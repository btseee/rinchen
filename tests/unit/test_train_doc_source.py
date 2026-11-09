"""Tests for `rinchen.train.doc_source.read_corpus_rows`: which rows of the corpus files
it loads. The continuation code never reads the rows of a split it does not train or
validate on (the test rows): their text is not loaded, and they are still counted, as
`not_<split>`, by `docify`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.corpus.schema import corpus_frame
from rinchen.errors import DataError
from rinchen.train.docify import docify, read_corpus_rows
from tests.support.corpus_rows import write_frame, write_rows
from tests.support.doc_rows import SENTENCES, document, rows_frame
from tests.support.tiny_parallel import train_toy_spm

BUDGET = 1000
TEST_MARKER = "never-to-be-loaded"


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("doc_source_spm"))


def rows_of_four_splits() -> list[dict[str, Any]]:
    """Four train rows, three dev rows, two test rows and one holdout row, the text and
    the document id of the last two kinds marked."""
    marked: dict[str, Any] = {
        "src": f"{TEST_MARKER} source",
        "tgt": f"{TEST_MARKER} target",
    }
    return [
        *document("train-doc", range(4)),
        *document("dev-doc", range(3), split="dev"),
        *(
            row | marked | {"doc_id": f"{TEST_MARKER}-doc"}
            for row in document("test-doc", range(2), split="test")
        ),
        *(
            row | marked | {"doc_id": f"{TEST_MARKER}-doc"}
            for row in document("holdout-doc", range(1), split="holdout")
        ),
    ]


def cells(frame: pl.DataFrame) -> list[str]:
    """Every non-null cell of `frame` as text."""
    return [
        str(value) for row in frame.iter_rows() for value in row if value is not None
    ]


def test_the_text_of_the_rows_of_a_split_that_is_not_windowed_is_not_returned(
    tmp_path: Path,
) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())

    rows = read_corpus_rows([path])

    assert not [cell for cell in cells(rows) if TEST_MARKER in cell]
    left_out = rows.filter(pl.col("split").is_in(["test", "holdout"]))
    assert left_out.height == 3
    assert left_out.drop("split").null_count().row(0) == (3,) * 10


def test_every_row_is_still_in_the_frame_so_that_it_is_counted(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())

    rows = read_corpus_rows([path])

    assert rows.height == 10
    assert rows["split"].value_counts().sort("split").rows() == [
        ("dev", 3),
        ("holdout", 1),
        ("test", 2),
        ("train", 4),
    ]


def test_the_rows_of_the_windowed_splits_are_returned_whole(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())

    rows = read_corpus_rows([path])

    train = rows.filter(pl.col("split") == "train").sort("seg_idx")
    assert train["src"].to_list() == [english for english, _ in SENTENCES[:4]]
    assert train["doc_id"].to_list() == ["train-doc"] * 4
    dev = rows.filter(pl.col("split") == "dev").sort("seg_idx")
    assert dev["tgt"].to_list() == [mongolian for _, mongolian in SENTENCES[:3]]
    assert dev["origin"].dtype == pl.String


def test_the_counts_of_docify_are_those_of_the_unreduced_rows(
    toy_spm: Path, tmp_path: Path
) -> None:
    rows = rows_of_four_splits()
    path = write_rows(tmp_path, "rows", rows)

    for split, windowed in (("train", 4), ("dev", 3)):
        from_file = docify(
            read_corpus_rows([path]),
            spm_model=toy_spm,
            min_size=2,
            max_size=2,
            max_pieces=BUDGET,
            seed=3,
            split=split,
        )
        in_memory = docify(
            rows_frame(rows),
            spm_model=toy_spm,
            min_size=2,
            max_size=2,
            max_pieces=BUDGET,
            seed=3,
            split=split,
        )

        assert from_file.stats == in_memory.stats
        assert from_file.frame.equals(in_memory.frame)
        assert from_file.stats.rows_read == 10
        assert from_file.stats.excluded[f"not_{split}"] == 10 - windowed


def test_the_rows_of_every_file_are_filtered_and_counted(
    toy_spm: Path, tmp_path: Path
) -> None:
    first = write_rows(tmp_path, "a", rows_of_four_splits())
    second = write_rows(
        tmp_path,
        "b",
        document("other", range(5)) + document("t", range(4), split="test"),
    )

    rows = read_corpus_rows([first, second])

    assert rows.height == 19
    assert not [cell for cell in cells(rows) if TEST_MARKER in cell]
    windows = docify(
        rows, spm_model=toy_spm, min_size=2, max_size=2, max_pieces=BUDGET, seed=3
    )
    assert windows.stats.rows_read == 19
    assert windows.stats.excluded["not_train"] == 3 + 2 + 1 + 4
    assert windows.stats.rows_in_windows == 4 + 4  # the tail of the 5 rows is left out


def test_a_row_with_no_split_is_counted_under_not_split(
    toy_spm: Path, tmp_path: Path
) -> None:
    frame = corpus_frame(document("d1", range(4))).with_columns(
        pl.when(pl.col("seg_idx") == 3)
        .then(pl.lit(None))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    path = write_frame(tmp_path, "rows", frame)

    rows = read_corpus_rows([path])

    assert rows.height == 4
    windows = docify(
        rows, spm_model=toy_spm, min_size=2, max_size=2, max_pieces=BUDGET, seed=3
    )
    assert windows.stats.excluded["not_train"] == 1
    assert windows.stats.rows_in_windows == 2


def test_the_splits_that_are_loaded_can_be_named(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())

    rows = read_corpus_rows([path], splits=("test",))

    loaded = rows.filter(pl.col("src").is_not_null())
    assert loaded["split"].to_list() == ["test"] * 2
    assert all(TEST_MARKER in text for text in loaded["src"])
    assert rows.height == 10


# ---- a split that was not loaded --------------------------------------------------


def windows_of_split(rows: pl.DataFrame, spm: Path, split: str) -> Any:
    return docify(
        rows,
        spm_model=spm,
        min_size=2,
        max_size=2,
        max_pieces=BUDGET,
        seed=3,
        split=split,
    )


@pytest.mark.parametrize("split", ["test", "holdout", "Train", "", "train "])
def test_docify_refuses_a_split_that_makes_no_windows(
    toy_spm: Path, split: str
) -> None:
    rows = rows_frame(document("d1", range(4)) + document("t1", range(4), split="test"))

    with pytest.raises(ValueError, match=r"split must be one of \('train', 'dev'\)"):
        windows_of_split(rows, toy_spm, split)


def test_docify_refuses_a_split_that_makes_no_windows_before_it_reads_the_model(
    tmp_path: Path,
) -> None:
    rows = rows_frame(document("d1", range(4)))

    with pytest.raises(ValueError, match="split must be one of"):
        windows_of_split(rows, tmp_path / "absent.model", "test")


@pytest.mark.parametrize(
    ("loaded", "split"),
    [(("train",), "dev"), (("dev",), "train"), (("test",), "train")],
)
def test_docify_refuses_rows_that_were_read_without_the_split(
    toy_spm: Path, tmp_path: Path, loaded: tuple[str, ...], split: str
) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())
    rows = read_corpus_rows([path], splits=loaded)

    with pytest.raises(DataError) as raised:
        windows_of_split(rows, toy_spm, split)

    message = str(raised.value)
    assert message.startswith(f"every {split} row is empty apart from its split")
    assert "read_corpus_rows(splits=" in message


@pytest.mark.parametrize("loaded", [("train",), ("train", "dev"), ("dev", "train")])
def test_docify_accepts_rows_that_were_read_with_the_split(
    toy_spm: Path, tmp_path: Path, loaded: tuple[str, ...]
) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())
    rows = read_corpus_rows([path], splits=loaded)

    windows = windows_of_split(rows, toy_spm, "train")

    assert windows.stats.windows == 2
    assert windows.stats.excluded["not_train"] == 6


def test_one_empty_row_among_real_ones_is_counted_and_is_not_a_split_not_loaded(
    toy_spm: Path,
) -> None:
    # Only a split whose rows are ALL empty apart from `split` was left unread.
    others = [name for name in rows_frame([]).columns if name != "split"]
    rows = rows_frame(document("d1", range(5))).with_columns(
        pl.when(pl.col("seg_idx") == 4).then(None).otherwise(pl.col(name)).alias(name)
        for name in others
    )

    windows = windows_of_split(rows, toy_spm, "train")

    assert windows.stats.excluded["no_doc_id"] == 1
    assert windows.stats.windows == 2


def test_a_split_with_no_rows_at_all_is_not_a_split_that_was_not_loaded(
    toy_spm: Path, tmp_path: Path
) -> None:
    path = write_rows(tmp_path, "rows", document("d1", range(4)))
    rows = read_corpus_rows([path])

    windows = windows_of_split(rows, toy_spm, "dev")

    assert windows.stats.windows == 0
    assert windows.stats.excluded["not_dev"] == 4


@pytest.mark.parametrize("splits", ["train", "dev", "train,dev", ""])
def test_read_corpus_rows_refuses_one_string_for_the_splits(
    tmp_path: Path, splits: str
) -> None:
    path = write_rows(tmp_path, "rows", rows_of_four_splits())

    with pytest.raises(TypeError, match="splits must be a collection of split names"):
        read_corpus_rows([path], splits=splits)
