"""Tests for `rinchen.train.docify`: the windows of a document, how each row is counted,
and the files they are read from.

The expectations are hand-computed with the toy SentencePiece model: the piece counts of
the sentences in `tests.support.doc_rows` were counted once, and a window of k sentences
has their sum plus two pieces per `<sep>` between them (the word-boundary piece U+2581 and `<sep>`); the source also
counts the direction tag and `<doc>`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sentencepiece

from rinchen.corpus.schema import make_row_id
from rinchen.errors import ConfigError, DataError
from rinchen.train.docify import (
    DEV_WINDOW_SEED,
    DocWindows,
    PieceSummary,
    docify,
    exclusion_reasons,
    read_corpus_rows,
)
from tests.support.corpus_rows import write_rows
from tests.support.doc_rows import (
    JOIN,
    MN_EN,
    SENTENCES,
    SOURCE_PIECES,
    TARGET_PIECES,
    doc_row,
    document,
    rows_frame,
)
from tests.support.tiny_parallel import train_toy_spm

BUDGET = 1000


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("docify_spm"))


def windows_of(
    rows: list[dict[str, Any]],
    spm: Path,
    *,
    size: int | None = None,
    min_size: int = 2,
    max_size: int = 4,
    max_pieces: int = BUDGET,
    seed: int = 7,
    split: str = "train",
    filter_reasons: list[str | None] | None = None,
) -> DocWindows:
    """The windows of `rows`; `size` fixes both bounds."""
    return docify(
        rows_frame(rows, filter_reasons=filter_reasons),
        spm_model=spm,
        min_size=size or min_size,
        max_size=size or max_size,
        max_pieces=max_pieces,
        seed=seed,
        split=split,
    )


def spans(windows: DocWindows) -> list[tuple[str, int, int]]:
    """(document, first segment, sentences) of each window."""
    return [
        (row["doc_id"], row["first_seg"], row["n_sentences"])
        for row in windows.frame.iter_rows(named=True)
    ]


def test_the_hand_counted_pieces_are_the_toy_models(toy_spm: Path) -> None:
    # The constants the other tests rely on, checked against the model that cuts them.
    processor = sentencepiece.SentencePieceProcessor(model_file=str(toy_spm))

    for (english, mongolian), source, target in zip(
        SENTENCES, SOURCE_PIECES, TARGET_PIECES, strict=True
    ):
        assert len(processor.encode(english)) == source
        assert len(processor.encode(mongolian)) == target
    assert len(processor.encode(JOIN.join(e for e, _ in SENTENCES[:2]))) == 7 + 8 + 2


# ---- the windows ---------------------------------------------------------------------


def test_a_document_is_cut_into_windows_of_the_size_asked_for(toy_spm: Path) -> None:
    rows = document("d1", range(8))

    windows = windows_of(rows, toy_spm, size=2)

    assert spans(windows) == [("d1", 0, 2), ("d1", 2, 2), ("d1", 4, 2), ("d1", 6, 2)]


def test_a_window_joins_the_sentences_of_each_side_with_sep_and_counts_its_pieces(
    toy_spm: Path,
) -> None:
    rows = document("d1", [0, 1, 2])

    windows = windows_of(rows, toy_spm, size=3)

    (window,) = windows.frame.iter_rows(named=True)
    english, mongolian = zip(*SENTENCES[:3], strict=True)
    assert window == {
        "id": make_row_id(
            "eng_Latn", "khk_Cyrl", JOIN.join(english), JOIN.join(mongolian)
        ),
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": "This is the book. <sep> The water is here. <sep> Where is the dog?",
        "tgt": "Энэ бол ном. <sep> Ус энд байна. <sep> Нохой хаана байна вэ?",
        "doc_id": "d1",
        "first_seg": 0,
        "n_sentences": 3,
        # 7 + 8 + 8 + 2 * 2 pieces, and the tag and `<doc>` in front of the source
        "src_pieces": 2 + 27,
        # 7 + 6 + 12 + 2 * 2
        "tgt_pieces": 29,
    }


def test_the_frame_has_exactly_the_documented_columns(toy_spm: Path) -> None:
    windows = windows_of(document("d1", range(4)), toy_spm, size=2)

    assert windows.frame.columns == [
        "id",
        "src_lang",
        "tgt_lang",
        "src",
        "tgt",
        "doc_id",
        "first_seg",
        "n_sentences",
        "src_pieces",
        "tgt_pieces",
    ]
    assert windows.frame.schema["first_seg"] == pl.Int64


def test_no_window_means_an_empty_frame_with_the_same_columns(toy_spm: Path) -> None:
    windows = windows_of(document("d1", [0]), toy_spm, size=2)

    assert windows.frame.height == 0
    assert windows.frame.columns[0] == "id"
    assert windows.frame.schema["n_sentences"] == pl.Int64
    assert windows.stats.windows == 0


def test_the_sizes_are_drawn_from_the_range_by_a_hash_of_seed_document_and_start(
    toy_spm: Path,
) -> None:
    # sha256(f"{seed}\t{doc_id}\t{first segment}"): the sizes below were worked out by
    # hand for a document of 24 sentences; the last one of seed 8 is a single sentence
    # left over, a tail.
    rows = [doc_row("d1", position, position) for position in range(24)]

    seven = windows_of(rows, toy_spm, seed=7)
    eight = windows_of(rows, toy_spm, seed=8)

    assert [size for _, _, size in spans(seven)] == [4, 2, 4, 2, 3, 4, 3, 2]
    assert [size for _, _, size in spans(eight)] == [3, 3, 2, 3, 2, 2, 3, 2, 3]
    assert eight.stats.excluded["tail"] == 1


def test_the_validation_windows_are_cut_with_the_fixed_dev_seed(toy_spm: Path) -> None:
    rows = [doc_row("d1", position, position, split="dev") for position in range(24)]

    with_constant = windows_of(rows, toy_spm, split="dev", seed=DEV_WINDOW_SEED)
    with_zero = windows_of(rows, toy_spm, split="dev", seed=0)

    assert DEV_WINDOW_SEED == 0
    assert with_constant.frame.equals(with_zero.frame)


def test_the_same_seed_repeats_the_windows(toy_spm: Path) -> None:
    rows = document("d1", range(8)) + document("d2", range(8), languages=MN_EN)

    assert windows_of(rows, toy_spm).frame.equals(windows_of(rows, toy_spm).frame)


def test_windows_do_not_depend_on_the_order_of_the_rows(toy_spm: Path) -> None:
    rows = (
        document("d1", range(8))
        + document("d1", range(8), languages=MN_EN)
        + document("d2", range(5))
    )
    expected = windows_of(rows, toy_spm).frame

    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert windows_of(shuffled, toy_spm).frame.equals(expected)


def test_windows_do_not_depend_on_the_order_of_the_files(
    toy_spm: Path, tmp_path: Path
) -> None:
    first = document("d1", range(8))
    second = document("d2", range(6), languages=MN_EN)
    paths = {
        name: write_rows(tmp_path, name, rows)
        for name, rows in (("first", first), ("second", second))
    }

    def windows(order: list[str]) -> DocWindows:
        rows = read_corpus_rows([paths[name] for name in order])
        return docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=3)

    assert windows(["first", "second"]).frame.equals(windows(["second", "first"]).frame)


def test_a_window_never_has_fewer_sentences_than_the_drawn_size_allows(
    toy_spm: Path,
) -> None:
    rows = [doc_row("d1", position, position) for position in range(7)]

    windows = windows_of(rows, toy_spm, min_size=2, max_size=4)

    sizes = [size for _, _, size in spans(windows)]
    assert all(2 <= size <= 4 for size in sizes)
    assert sum(sizes) + windows.stats.excluded["tail"] == 7


# ---- consecutive segments --------------------------------------------------------------


def test_a_gap_in_seg_idx_ends_the_window(toy_spm: Path) -> None:
    rows = [doc_row("d1", seg, seg) for seg in (0, 1, 2, 4, 5, 6)]

    windows = windows_of(rows, toy_spm, size=3)

    assert spans(windows) == [("d1", 0, 3), ("d1", 4, 3)]


def test_a_run_shorter_than_the_smallest_window_is_counted_as_alone(
    toy_spm: Path,
) -> None:
    rows = [doc_row("d1", seg, seg) for seg in (0, 1, 3, 5, 6)]

    windows = windows_of(rows, toy_spm, size=2)

    assert spans(windows) == [("d1", 0, 2), ("d1", 5, 2)]
    assert windows.stats.excluded["gap_single"] == 1  # segment 3, between two gaps
    assert windows.stats.excluded["tail"] == 0


def test_the_seg_idx_need_not_start_at_zero(toy_spm: Path) -> None:
    rows = [doc_row("d1", seg, seg) for seg in (10, 11, 12, 13)]

    windows = windows_of(rows, toy_spm, size=2)

    assert spans(windows) == [("d1", 10, 2), ("d1", 12, 2)]


def test_a_window_never_crosses_into_the_next_document_or_direction(
    toy_spm: Path,
) -> None:
    # The segments go on from one document to the next, and from one direction of a
    # document to the other, so only the document and the languages tell them apart.
    rows = (
        [doc_row("d1", seg, seg) for seg in range(3)]
        + [doc_row("d2", seg, seg) for seg in range(3, 6)]
        + [doc_row("d2", seg, seg, languages=MN_EN) for seg in range(6, 9)]
    )

    windows = windows_of(rows, toy_spm, size=3)

    assert spans(windows) == [("d1", 0, 3), ("d2", 3, 3), ("d2", 6, 3)]


def test_a_one_sentence_document_is_counted_and_makes_no_window(toy_spm: Path) -> None:
    windows = windows_of([doc_row("d1", 0, 0)], toy_spm, size=2)

    assert windows.frame.height == 0
    assert windows.stats.excluded["gap_single"] == 1


def test_the_sentence_left_after_the_last_window_is_a_tail(toy_spm: Path) -> None:
    windows = windows_of(document("d1", range(5)), toy_spm, size=2)

    assert spans(windows) == [("d1", 0, 2), ("d1", 2, 2)]
    assert windows.stats.excluded["tail"] == 1
    assert windows.stats.excluded["gap_single"] == 0


def test_the_last_window_takes_what_is_left_when_that_is_at_least_the_smallest_size(
    toy_spm: Path,
) -> None:
    # By the hash of seed 7 the draws of a document "d1" are 4, 2 and 4 at segments 0,
    # 4 and 6: of eight sentences the last window asks for 4 and has 2, of nine it has 3.
    eight = windows_of([doc_row("d1", p, p) for p in range(8)], toy_spm, seed=7)
    nine = windows_of([doc_row("d1", p, p) for p in range(9)], toy_spm, seed=7)

    assert [size for _, _, size in spans(eight)] == [4, 2, 2]
    assert [size for _, _, size in spans(nine)] == [4, 2, 3]
    assert eight.stats.excluded["tail"] == nine.stats.excluded["tail"] == 0


def test_fewer_sentences_than_the_size_leave_a_tail_when_the_size_is_fixed(
    toy_spm: Path,
) -> None:
    three = windows_of(document("d1", range(3)), toy_spm, size=3)
    four = windows_of(document("d1", range(4)), toy_spm, size=3)

    assert spans(three) == [("d1", 0, 3)]
    assert spans(four) == [("d1", 0, 3)]
    assert four.stats.excluded["tail"] == 1


def test_two_rows_at_one_position_leave_one_and_count_the_other(toy_spm: Path) -> None:
    first = doc_row("d1", 0, 0)
    second = doc_row("d1", 1, 1)
    clash = doc_row("d1", 1, 4)
    rows = [first, second, clash, doc_row("d1", 2, 2)]
    kept = min(
        (second, clash),
        key=lambda row: make_row_id(
            row["src_lang"], row["tgt_lang"], row["src"], row["tgt"]
        ),
    )

    windows = windows_of(rows, toy_spm, size=3)

    assert windows.stats.excluded["duplicate_seg_idx"] == 1
    (window,) = windows.frame.iter_rows(named=True)
    assert window["src"].split(JOIN)[1] == kept["src"]
    # The same whatever the order the rows arrive in.
    assert windows_of(rows[::-1], toy_spm, size=3).frame.equals(windows.frame)


def test_the_rows_of_two_directions_of_one_document_are_windowed_apart(
    toy_spm: Path,
) -> None:
    rows = document("d1", range(4)) + document("d1", range(4), languages=MN_EN)

    windows = windows_of(rows, toy_spm, size=2)

    assert [
        (row["src_lang"], row["tgt_lang"], row["first_seg"])
        for row in windows.frame.iter_rows(named=True)
    ] == [
        ("eng_Latn", "khk_Cyrl", 0),
        ("eng_Latn", "khk_Cyrl", 2),
        ("khk_Cyrl", "eng_Latn", 0),
        ("khk_Cyrl", "eng_Latn", 2),
    ]
    assert windows.frame["src"][2] == "Энэ бол ном. <sep> Ус энд байна."


def test_two_documents_with_the_same_id_in_other_language_pairs_are_apart(
    toy_spm: Path,
) -> None:
    ru_mn = ("rus_Cyrl", "khk_Cyrl")
    rows = document("d1", range(2)) + [
        doc_row("d1", 0, 0, languages=ru_mn, src="Это книга.", tgt="Энэ бол ном."),
        doc_row("d1", 1, 1, languages=ru_mn, src="Вода здесь.", tgt="Ус энд байна."),
    ]

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.frame["src_lang"].to_list() == ["eng_Latn", "rus_Cyrl"]


def test_both_sides_of_every_window_have_one_sep_fewer_than_sentences(
    toy_spm: Path,
) -> None:
    rows = [doc_row("d1", position, position) for position in range(30)]

    windows = windows_of(rows, toy_spm)

    for row in windows.frame.iter_rows(named=True):
        assert row["src"].count("<sep>") == row["n_sentences"] - 1
        assert row["tgt"].count("<sep>") == row["n_sentences"] - 1


def test_a_huge_document_is_one_pass_over_its_sentences(toy_spm: Path) -> None:
    rows = [
        doc_row("big", position, 0, src=f"Line {position}.", tgt=f"Мөр {position}.")
        for position in range(3000)
    ]

    windows = windows_of(rows, toy_spm)

    stats = windows.stats
    assert stats.rows_read == 3000
    assert stats.rows_in_windows + sum(stats.excluded.values()) == 3000
    assert stats.windows > 900


# ---- the piece budget ------------------------------------------------------------------


def test_a_window_is_shortened_until_both_sides_fit_the_budget(toy_spm: Path) -> None:
    # Four sentences are 40 and 37 pieces (source with the tag and `<doc>`, then target);
    # three are 29 and 29; two are 19 and 15. The size is fixed at 4, so what is left after
    # a shortened window is a tail.
    rows = document("d1", range(4))

    whole = windows_of(rows, toy_spm, size=4, max_pieces=40)
    of_three = windows_of(rows, toy_spm, size=4, max_pieces=39)
    of_two = windows_of(rows, toy_spm, size=4, max_pieces=25)

    assert spans(whole) == [("d1", 0, 4)]
    assert spans(of_three) == [("d1", 0, 3)]
    assert of_three.stats.excluded["tail"] == 1
    # Shortened to two, the two that are left are fewer than the size asked for.
    assert spans(of_two) == [("d1", 0, 2)]
    assert of_two.stats.excluded["tail"] == 2


def test_the_target_side_alone_can_force_the_shortening(toy_spm: Path) -> None:
    # Sentences 2 and 5: source 8 + 7 + 2 + 2 = 19 pieces, target 12 + 7 + 2 = 21.
    rows = [doc_row("d1", 0, 2), doc_row("d1", 1, 5)]

    fits = windows_of(rows, toy_spm, size=2, max_pieces=21)
    short_of_target = windows_of(rows, toy_spm, size=2, max_pieces=20)

    assert spans(fits) == [("d1", 0, 2)]
    assert spans(short_of_target) == [("d1", 0, 1)]
    assert short_of_target.stats.excluded["tail"] == 1


def test_the_piece_counts_of_a_window_never_exceed_the_budget(toy_spm: Path) -> None:
    rows = [doc_row("d1", position, position) for position in range(40)]

    windows = windows_of(rows, toy_spm, max_pieces=30)

    assert windows.stats.windows > 0
    assert windows.frame["src_pieces"].max() <= 30
    assert windows.frame["tgt_pieces"].max() <= 30


def test_a_budget_that_shortens_a_window_below_the_smallest_size_leaves_a_single(
    toy_spm: Path,
) -> None:
    # A budget of 12: only one sentence fits at a time (7 + 2 and 8 + 2 pieces alone, 19
    # and 17 together); the single sentence left at the end is a tail.
    rows = document("d1", [0, 1, 5])

    windows = windows_of(rows, toy_spm, size=2, max_pieces=12)

    assert spans(windows) == [("d1", 0, 1), ("d1", 1, 1)]
    assert windows.stats.windows_by_size == {1: 2, 2: 0}
    assert windows.stats.excluded["tail"] == 1


def test_a_sentence_over_the_budget_alone_is_left_out_and_ends_the_run(
    toy_spm: Path,
) -> None:
    # The middle sentence is about 70 pieces on the target side, over a budget of 60;
    # its neighbours are not joined across it, and without a budget they would be.
    rows = document("d1", [0, 1, 2, 5, 6])
    rows[2]["tgt"] = " ".join(["Энэ бол ном."] * 10)

    open_budget = windows_of(rows, toy_spm, size=2, max_pieces=1000)
    tight = windows_of(rows, toy_spm, size=2, max_pieces=60)

    assert spans(open_budget) == [("d1", 0, 2), ("d1", 2, 2)]
    assert open_budget.stats.excluded["too_long"] == 0
    assert open_budget.stats.excluded["tail"] == 1
    assert spans(tight) == [("d1", 0, 2), ("d1", 3, 2)]
    assert tight.stats.excluded["too_long"] == 1
    assert tight.stats.excluded["tail"] == 0


def test_a_sentence_is_kept_when_it_fills_the_budget_exactly_and_dropped_above_it(
    toy_spm: Path,
) -> None:
    # Sentence 2 is 12 pieces on the target side (and 8 + 2 on the source side).
    rows = [doc_row("d1", 0, 2), doc_row("d1", 1, 2)]

    exact = windows_of(rows, toy_spm, size=1, max_pieces=12)
    below = windows_of(rows, toy_spm, size=1, max_pieces=11)

    assert (exact.stats.windows, exact.stats.excluded["too_long"]) == (2, 0)
    assert (below.stats.windows, below.stats.excluded["too_long"]) == (0, 2)


def test_the_tag_and_doc_count_against_the_source_budget(toy_spm: Path) -> None:
    # Sentence 3 is 9 pieces on the source side, 11 with the tag and `<doc>`.
    row = [doc_row("d1", 0, 3), doc_row("d1", 1, 3)]

    kept = windows_of(row, toy_spm, size=1, max_pieces=11)
    dropped = windows_of(row, toy_spm, size=1, max_pieces=10)

    assert kept.stats.windows == 2
    assert dropped.stats.windows == 0
    assert dropped.stats.excluded["too_long"] == 2


# ---- what a row must be -----------------------------------------------------------------


def test_a_sentence_with_a_literal_sep_is_left_out_and_ends_the_run(
    toy_spm: Path,
) -> None:
    rows = document("d1", range(7))
    rows[3]["src"] = "Look <sep> here."
    rows[5]["tgt"] = "Энд<sep>тэнд."

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.stats.excluded["contains_sep"] == 2
    assert spans(windows) == [("d1", 0, 2)]
    # Segments 0 to 2 are one run (a window and a tail), 4 and 6 are alone.
    assert windows.stats.excluded["tail"] == 1
    assert windows.stats.excluded["gap_single"] == 2


@pytest.mark.parametrize(
    "src, tgt",
    [
        ("", "Энэ бол ном."),
        ("This is the book.", ""),
        ("   ", "Энэ бол ном."),
        ("This is the book.", " " + chr(0x3000) + " "),
        (" \t\n ", "Энэ бол ном."),
        (chr(0x1C), "Энэ бол ном."),
    ],
    ids=["empty src", "empty tgt", "spaces", "ideographic space", "tab and lf", "FS"],
)
def test_an_empty_or_blank_sentence_is_counted_as_empty(
    toy_spm: Path, src: str, tgt: str
) -> None:
    rows = document("d1", range(5))
    rows[2]["src"], rows[2]["tgt"] = src, tgt

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.stats.excluded["empty"] == 1
    assert windows.stats.excluded["line_break"] == 0
    assert spans(windows) == [("d1", 0, 2), ("d1", 3, 2)]


def test_a_sentence_with_a_null_text_is_counted_as_empty(toy_spm: Path) -> None:
    rows = rows_frame(document("d1", range(4))).with_columns(
        pl.when(pl.col("seg_idx") == 1)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("tgt"))
        .alias("tgt")
    )

    windows = docify(
        rows, spm_model=toy_spm, min_size=2, max_size=2, max_pieces=BUDGET, seed=1
    )

    assert windows.stats.excluded["empty"] == 1


@pytest.mark.parametrize(
    "separator",
    [chr(0x0A), chr(0x0D), chr(0x0B), chr(0x0C), chr(0x85), chr(0x2028), chr(0x2029)],
    ids=["LF", "CR", "VT", "FF", "NEL", "LS", "PS"],
)
def test_a_line_break_inside_a_sentence_is_counted(
    toy_spm: Path, separator: str
) -> None:
    rows = document("d1", range(5))
    rows[2]["src"] = f"This is{separator}the book."

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.stats.excluded["line_break"] == 1
    assert windows.stats.excluded["empty"] == 0


@pytest.mark.parametrize("doc_id", [None, "", "   "], ids=["null", "empty", "blank"])
def test_a_row_without_a_document_id_is_counted_and_makes_no_window(
    toy_spm: Path, doc_id: str | None
) -> None:
    rows = [doc_row(doc_id, seg, seg) for seg in range(4)]

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.stats.windows == 0
    assert windows.stats.excluded["no_doc_id"] == 4


def test_a_row_without_a_seg_idx_is_counted(toy_spm: Path) -> None:
    rows = [doc_row("d1", None, 0), *document("d1", range(1, 3))]

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.stats.excluded["no_seg_idx"] == 1
    assert spans(windows) == [("d1", 0, 2)]


@pytest.mark.parametrize("origin", ["human", "web", "mined", "converted"])
def test_authentic_origins_make_windows(toy_spm: Path, origin: str) -> None:
    windows = windows_of(document("d1", range(4), origin=origin), toy_spm, size=2)

    assert windows.stats.windows == 2


@pytest.mark.parametrize("origin", ["bt", "ft", "synth_llm", "synth_nmt"])
def test_every_other_origin_is_excluded(toy_spm: Path, origin: str) -> None:
    windows = windows_of(document("d1", range(4), origin=origin), toy_spm, size=2)

    assert windows.stats.windows == 0
    assert windows.stats.excluded["excluded_origin"] == 4


def test_a_row_without_a_split_is_not_of_the_split(toy_spm: Path) -> None:
    rows = rows_frame(document("d1", range(5))).with_columns(
        pl.when(pl.col("seg_idx") == 2)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("split"))
        .alias("split")
    )

    windows = docify(
        rows, spm_model=toy_spm, min_size=2, max_size=2, max_pieces=BUDGET, seed=1
    )

    assert windows.stats.excluded["not_train"] == 1
    assert spans(windows) == [("d1", 0, 2), ("d1", 3, 2)]


def test_filtered_rows_are_left_out_and_counted(toy_spm: Path) -> None:
    rows = document("d1", range(5))

    windows = windows_of(
        rows, toy_spm, size=2, filter_reasons=[None, None, "lid", None, None]
    )

    assert windows.stats.excluded["filtered"] == 1
    assert spans(windows) == [("d1", 0, 2), ("d1", 3, 2)]


@pytest.mark.parametrize("split", ["dev", "test", "holdout"])
def test_only_the_rows_of_the_split_make_windows(toy_spm: Path, split: str) -> None:
    rows = document("d1", range(4)) + document("d2", range(4), split=split)

    windows = windows_of(rows, toy_spm, size=2)

    assert {row["doc_id"] for row in windows.frame.iter_rows(named=True)} == {"d1"}
    assert windows.stats.excluded["not_train"] == 4


def test_the_dev_windows_are_made_from_the_dev_rows(toy_spm: Path) -> None:
    rows = document("d1", range(4)) + document("d2", range(4), split="dev")

    windows = windows_of(rows, toy_spm, size=2, split="dev", seed=DEV_WINDOW_SEED)

    assert spans(windows) == [("d2", 0, 2), ("d2", 2, 2)]
    assert windows.stats.excluded["not_dev"] == 4
    assert "not_train" not in windows.stats.excluded


def test_the_first_reason_that_applies_is_the_one_counted(toy_spm: Path) -> None:
    rows = [
        doc_row("d1", 0, 0, split="dev", origin="bt", src=""),
        doc_row(None, 1, 1, origin="bt"),
        doc_row("d1", 2, 2, src=""),
    ]

    windows = windows_of(rows, toy_spm, size=2, filter_reasons=["x", "x", None])

    excluded = windows.stats.excluded
    assert (excluded["not_train"], excluded["filtered"], excluded["empty"]) == (1, 1, 1)
    assert sum(excluded.values()) == 3


def test_text_is_never_changed(toy_spm: Path) -> None:
    decomposed = "Cafe" + chr(0x301) + " is here."
    rows = [
        doc_row("d1", 0, 0, src=decomposed),
        doc_row("d1", 1, 1),
    ]

    windows = windows_of(rows, toy_spm, size=2)

    assert windows.frame["src"][0].startswith(decomposed + JOIN)


# ---- the counts -------------------------------------------------------------------------


def test_every_row_is_counted_once_in_a_window_or_under_one_reason(
    toy_spm: Path,
) -> None:
    rows = (
        document("d1", range(8))
        + document("d2", range(3), origin="bt")
        + document("d3", range(3), split="dev")
        + [doc_row("d4", 0, 0), doc_row("d4", 0, 4), doc_row("d4", 2, 2)]
        + [doc_row(None, 0, 0)]
    )

    windows = windows_of(
        rows, toy_spm, size=3, filter_reasons=[None] * (len(rows) - 1) + ["x"]
    )

    stats = windows.stats
    assert stats.rows_read == len(rows) == 18
    assert stats.rows_in_windows + sum(stats.excluded.values()) == 18
    assert stats.excluded == {
        "not_train": 3,
        "filtered": 1,
        "excluded_origin": 3,
        "no_doc_id": 0,
        "no_seg_idx": 0,
        "empty": 0,
        "line_break": 0,
        "contains_sep": 0,
        "duplicate_seg_idx": 1,
        "too_long": 0,
        "gap_single": 2,
        "tail": 2,
    }
    assert stats.windows == 2
    assert stats.rows_in_windows == 6


def test_the_reasons_are_listed_in_the_order_they_are_tried() -> None:
    assert exclusion_reasons("train") == (
        "not_train",
        "filtered",
        "excluded_origin",
        "no_doc_id",
        "no_seg_idx",
        "empty",
        "line_break",
        "contains_sep",
        "duplicate_seg_idx",
        "too_long",
        "gap_single",
        "tail",
    )
    assert exclusion_reasons("dev")[0] == "not_dev"


def test_the_windows_by_size_count_every_size_from_one(toy_spm: Path) -> None:
    rows = [doc_row("d1", position, position) for position in range(24)]

    windows = windows_of(rows, toy_spm, seed=7)

    # [4, 2, 4, 2, 3, 4, 3, 2] by the hash of seed 7
    assert windows.stats.windows_by_size == {1: 0, 2: 3, 3: 2, 4: 3}
    assert windows.stats.windows == 8
    assert windows.stats.rows_in_windows == 24


def test_the_piece_summaries_are_over_the_windows(toy_spm: Path) -> None:
    # Windows of two over sentences 0 to 3: source 19, 21 pieces (tag and `<doc>`
    # included) and target 15, 20.
    windows = windows_of(document("d1", range(4)), toy_spm, size=2)

    pieces = windows.stats.pieces
    assert pieces["src"] == PieceSummary(min=19, mean=20.0, p95=21, max=21)
    assert pieces["tgt"] == PieceSummary(min=15, mean=17.5, p95=20, max=20)


def test_the_piece_summary_of_no_window_is_zeros(toy_spm: Path) -> None:
    windows = windows_of(document("d1", [0]), toy_spm, size=2)

    assert windows.stats.pieces["src"] == PieceSummary(0, 0.0, 0, 0)
    assert windows.stats.pieces["tgt"] == PieceSummary(0, 0.0, 0, 0)


def test_the_stats_are_json_ready(toy_spm: Path) -> None:
    windows = windows_of(document("d1", range(4)), toy_spm, size=2)

    document_ = json.loads(json.dumps(windows.stats.to_dict()))

    assert document_["windows_by_size"] == {"1": 0, "2": 2}
    assert document_["pieces"]["src"] == {"min": 19, "mean": 20.0, "p95": 21, "max": 21}
    assert document_["excluded"]["tail"] == 0
    assert document_["rows_read"] == 4


# ---- what is refused ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"min_size": 0},
        {"min_size": 5, "max_size": 4},
        {"max_size": 0, "min_size": 1},
        {"min_size": True},
        {"max_size": 4.0},
        {"max_pieces": 0},
        {"max_pieces": True},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_arguments_that_make_no_sense_are_refused(
    toy_spm: Path, arguments: dict[str, Any]
) -> None:
    keywords: dict[str, Any] = {"max_pieces": BUDGET, "seed": 1} | arguments

    with pytest.raises(ValueError, match="min_size|max_size|max_pieces|seed"):
        docify(rows_frame(document("d1", range(4))), spm_model=toy_spm, **keywords)


def test_a_frame_without_a_column_is_a_data_error(toy_spm: Path) -> None:
    rows = rows_frame(document("d1", range(4))).drop("seg_idx")

    with pytest.raises(DataError, match="seg_idx"):
        docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=1)


@pytest.mark.parametrize("column", ["id", "src_lang", "tgt_lang", "origin"])
def test_a_row_that_would_make_a_window_but_lacks_its_identity_is_a_data_error(
    toy_spm: Path, column: str
) -> None:
    rows = rows_frame(document("d1", range(4))).with_columns(
        pl.when(pl.col("seg_idx") == 0)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(DataError, match=f"no {column}"):
        docify(rows, spm_model=toy_spm, max_pieces=BUDGET, seed=1)


def test_a_target_language_outside_the_registry_is_a_config_error(
    toy_spm: Path,
) -> None:
    rows = document("d1", range(4), languages=("eng_Latn", "xxx_Xxxx"))

    with pytest.raises(ConfigError, match="xxx_Xxxx"):
        windows_of(rows, toy_spm)


def test_a_model_without_the_separator_as_one_piece_is_a_config_error(
    toy_spm: Path, tmp_path: Path
) -> None:
    corpus = tmp_path / "plain.txt"
    corpus.write_text(
        "".join(f"line {i} of some text\n" for i in range(200)), encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "plain"),
        vocab_size=40,
        minloglevel=2,
    )

    with pytest.raises(ConfigError, match="<doc>"):
        windows_of(document("d1", range(4)), tmp_path / "plain.model")


def test_a_model_that_does_not_exist_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a file"):
        windows_of(document("d1", range(4)), tmp_path / "missing.model")


def test_the_arguments_are_checked_before_the_model_is_loaded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_size"):
        windows_of(document("d1", range(4)), tmp_path / "missing.model", min_size=0)


# ---- reading the files --------------------------------------------------------------------


def test_the_rows_of_every_file_are_read_with_the_columns_docify_needs(
    tmp_path: Path,
) -> None:
    first = write_rows(tmp_path, "a", document("d1", range(3)))
    second = write_rows(
        tmp_path, "b", document("d2", range(2), split="dev", origin="bt")
    )

    rows = read_corpus_rows([first, second])

    assert rows.height == 5
    assert rows.columns == [
        "id",
        "src_lang",
        "tgt_lang",
        "src",
        "tgt",
        "origin",
        "corpus",
        "split",
        "filter_reason",
        "doc_id",
        "seg_idx",
    ]
    assert rows.schema["origin"] == pl.String
    assert rows.schema["split"] == pl.String
    assert rows["split"].to_list() == ["train"] * 3 + ["dev"] * 2
    assert rows["origin"].to_list() == ["human"] * 3 + ["bt"] * 2


def test_a_file_name_with_glob_characters_is_read_as_it_is(tmp_path: Path) -> None:
    directory = tmp_path / "[a]*"
    directory.mkdir()
    path = write_rows(directory, "rows?", document("d1", range(2)))

    assert read_corpus_rows([path]).height == 2


def test_no_file_is_a_data_error() -> None:
    with pytest.raises(DataError, match="no corpus file"):
        read_corpus_rows([])


def test_a_missing_file_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        read_corpus_rows([tmp_path / "missing.parquet"])


def test_a_file_that_is_not_a_corpus_table_is_a_data_error(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)

    with pytest.raises(DataError, match="not a corpus table"):
        read_corpus_rows([path])


def test_a_damaged_file_is_a_data_error_whatever_polars_raises(tmp_path: Path) -> None:
    good = write_rows(tmp_path, "good", document("d1", range(3)))
    damaged = tmp_path / "damaged.parquet"
    damaged.write_bytes(good.read_bytes()[: good.stat().st_size // 2])

    with pytest.raises(DataError, match="damaged.parquet"):
        read_corpus_rows([damaged])


def test_the_lines_of_a_written_corpus_are_what_the_windows_hold(
    toy_spm: Path, tmp_path: Path
) -> None:
    # A round trip through Parquet gives the same windows as the in-memory frame.
    rows = document("d1", range(6)) + document("d2", range(5), languages=MN_EN)
    path = write_rows(tmp_path, "rows", rows)

    from_file = docify(
        read_corpus_rows([path]), spm_model=toy_spm, max_pieces=BUDGET, seed=9
    )
    in_memory = docify(rows_frame(rows), spm_model=toy_spm, max_pieces=BUDGET, seed=9)

    assert from_file.frame.equals(in_memory.frame)
