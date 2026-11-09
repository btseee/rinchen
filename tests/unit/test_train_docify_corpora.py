"""Tests for `rinchen.train.docify` when documents of different corpora share a
`doc_id`: D4 names a document after the stem of its source file and numbers its
segments from 0, so two corpora of files with the same name collide on `(doc_id,
seg_idx)`. A document is `(src_lang, tgt_lang, corpus, doc_id)`; a window never holds
sentences of two corpora.

The sentences are the toy ones of `tests.support.doc_rows`; every sentence of the
`SENTENCES` is different, so a window says which corpus each of its sentences came from.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.errors import DataError
from rinchen.train.docify import DocWindows, docify, read_corpus_rows
from tests.support.corpus_rows import write_rows
from tests.support.doc_rows import JOIN, MN_EN, SENTENCES, doc_row, rows_frame
from tests.support.tiny_parallel import train_toy_spm

BUDGET = 1000


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("docify_corpora_spm"))


def windows_of(rows: list[dict[str, Any]], spm: Path, *, size: int) -> DocWindows:
    return docify(
        rows_frame(rows),
        spm_model=spm,
        min_size=size,
        max_size=size,
        max_pieces=BUDGET,
        seed=7,
    )


def corpus_rows(
    corpus: str, sentences: range, *, first_seg: int = 0, **kwargs: Any
) -> list[dict[str, Any]]:
    """The sentences `sentences` of `SENTENCES` as the document "en" of `corpus`, their
    `seg_idx` counted up from `first_seg`."""
    return [
        doc_row("en", first_seg + position, index, corpus=corpus, **kwargs)
        for position, index in enumerate(sentences)
    ]


def english(*indexes: int) -> str:
    return JOIN.join(SENTENCES[index][0] for index in indexes)


def sources(windows: DocWindows) -> list[str]:
    return windows.frame["src"].to_list()


def test_a_window_never_joins_the_last_sentences_of_one_corpus_to_the_next_ones(
    toy_spm: Path,
) -> None:
    # Corpus A is the document "en" at seg_idx 0..3 and corpus B another one called
    # "en" at 4..7: as one document they would be cut into 0-2, 3-5 (A3 B4 B5), 6-7.
    rows = corpus_rows("A", range(4)) + corpus_rows("B", range(4, 8), first_seg=4)

    windows = windows_of(rows, toy_spm, size=3)

    assert sources(windows) == [english(0, 1, 2), english(4, 5, 6)]
    assert windows.stats.excluded["tail"] == 2  # A3 and B7, one in each corpus
    assert windows.stats.rows_in_windows == 6


def test_two_corpora_numbered_from_zero_lose_no_row_as_a_duplicate(
    toy_spm: Path,
) -> None:
    # The same (doc_id, seg_idx) in two corpora is two sentences, not a repeated one.
    rows = corpus_rows("A", range(4)) + corpus_rows("B", range(4, 8))

    windows = windows_of(rows, toy_spm, size=3)

    assert sources(windows) == [english(0, 1, 2), english(4, 5, 6)]
    assert windows.stats.excluded["duplicate_seg_idx"] == 0
    assert windows.stats.excluded["tail"] == 2
    assert windows.stats.rows_in_windows == 6


def test_a_window_is_made_of_one_corpus_when_both_are_numbered_from_zero(
    toy_spm: Path,
) -> None:
    # As one document the rows would interleave: B0 A1 B2 (a duplicate position kept
    # by its id) instead of A0 A1 A2.
    rows = corpus_rows("A", range(4)) + corpus_rows("B", range(4, 8))

    windows = windows_of(rows, toy_spm, size=2)

    assert sorted(sources(windows)) == sorted(
        [english(0, 1), english(2, 3), english(4, 5), english(6, 7)]
    )


def test_two_files_of_different_corpora_with_the_same_document_id_are_not_one_document(
    toy_spm: Path, tmp_path: Path
) -> None:
    first = write_rows(tmp_path, "a", corpus_rows("A", range(4)))
    second = write_rows(tmp_path, "b", corpus_rows("B", range(4, 8)))

    windows = docify(
        read_corpus_rows([first, second]),
        spm_model=toy_spm,
        min_size=3,
        max_size=3,
        max_pieces=BUDGET,
        seed=7,
    )

    assert sources(windows) == [english(0, 1, 2), english(4, 5, 6)]
    assert windows.stats.excluded["duplicate_seg_idx"] == 0


def test_the_same_sentence_at_the_same_position_of_two_corpora_is_kept_in_both(
    toy_spm: Path,
) -> None:
    rows = corpus_rows("A", range(3)) + corpus_rows("B", range(3))

    windows = windows_of(rows, toy_spm, size=3)

    assert sources(windows) == [english(0, 1, 2)] * 2
    assert windows.stats.excluded["duplicate_seg_idx"] == 0


def test_a_repeated_position_inside_one_corpus_is_still_a_duplicate(
    toy_spm: Path,
) -> None:
    rows = (
        corpus_rows("A", range(3))
        + [doc_row("en", 1, 5, corpus="A")]
        + corpus_rows("B", range(3))
    )

    windows = windows_of(rows, toy_spm, size=3)

    assert windows.stats.excluded["duplicate_seg_idx"] == 1
    assert windows.stats.windows == 2


def test_the_two_directions_of_a_document_get_the_same_windows_in_each_corpus(
    toy_spm: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for corpus in ("A", "B"):
        rows += corpus_rows(corpus, range(8))
        rows += corpus_rows(corpus, range(8), languages=MN_EN)

    frame = docify(rows_frame(rows), spm_model=toy_spm, max_pieces=BUDGET, seed=3).frame

    spans = {
        language: sorted(
            zip(part["first_seg"], part["n_sentences"], strict=True), key=tuple
        )
        for (language,), part in frame.partition_by("src_lang", as_dict=True).items()
    }
    assert len(spans["eng_Latn"]) >= 4  # two windows at least in each of two corpora
    assert spans["eng_Latn"] == spans["khk_Cyrl"]


def test_one_corpus_is_cut_as_before_whatever_it_is_called(toy_spm: Path) -> None:
    rows = [doc_row("d1", position, position) for position in range(8)]

    named_a = [row | {"corpus": "A"} for row in rows]
    named_b = [row | {"corpus": "a-much-longer-name"} for row in rows]

    expected = docify(rows_frame(rows), spm_model=toy_spm, max_pieces=BUDGET, seed=5)
    for named in (named_a, named_b):
        windows = docify(
            rows_frame(named), spm_model=toy_spm, max_pieces=BUDGET, seed=5
        )
        assert windows.frame.equals(expected.frame)
        assert windows.stats == expected.stats


def test_documents_with_different_ids_in_two_corpora_are_cut_as_in_one(
    toy_spm: Path,
) -> None:
    one = [doc_row("d1", s, s, corpus="A") for s in range(8)]
    two = [doc_row("d2", s, s, corpus="B") for s in range(8)]
    together = [doc_row("d2", s, s, corpus="A") for s in range(8)]

    split_up = docify(
        rows_frame(one + two), spm_model=toy_spm, max_pieces=BUDGET, seed=5
    )
    as_one = docify(
        rows_frame(one + together), spm_model=toy_spm, max_pieces=BUDGET, seed=5
    )

    assert sorted(split_up.frame["src"]) == sorted(as_one.frame["src"])
    assert split_up.stats == as_one.stats


def test_windows_of_two_corpora_do_not_depend_on_the_order_of_the_rows(
    toy_spm: Path,
) -> None:
    rows = (
        corpus_rows("A", range(4))
        + corpus_rows("B", range(4, 8))
        + corpus_rows("B", range(5), first_seg=10)
    )
    expected = windows_of(rows, toy_spm, size=2).frame

    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert windows_of(shuffled, toy_spm, size=2).frame.equals(expected)


def test_the_corpus_is_one_of_the_columns_docify_needs(toy_spm: Path) -> None:
    rows = rows_frame(corpus_rows("A", range(4))).drop("corpus")

    with pytest.raises(DataError, match="corpus"):
        docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=1)


def test_a_row_that_would_make_a_window_but_has_no_corpus_is_a_data_error(
    toy_spm: Path,
) -> None:
    rows = rows_frame(corpus_rows("A", range(4))).with_columns(
        pl.when(pl.col("seg_idx") == 0)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("corpus"))
        .alias("corpus")
    )

    with pytest.raises(DataError, match="no corpus"):
        docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=1)


def test_a_row_without_a_corpus_that_is_left_out_anyway_is_counted_not_refused(
    toy_spm: Path,
) -> None:
    rows = rows_frame(corpus_rows("A", range(4), split="dev")).with_columns(
        pl.lit(None, dtype=pl.String).alias("corpus")
    )

    windows = docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=1)

    assert windows.stats.excluded["not_train"] == 4
