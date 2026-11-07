"""From alignments to mined corpus rows: which alignments become rows, how the
sentences of a group are joined, how rows are numbered and labelled."""

from __future__ import annotations

import polars as pl

from rinchen.corpus.schema import CORPUS_SCHEMA, make_row_id
from rinchen.mine.align import Alignment
from rinchen.mine.pairs import AlignmentCounts, aligned_corpus_frame, count_alignments

SRC = ["s0", "s1", "s2", "s3", "s4"]
TGT = ["t0", "t1", "t2", "t3"]
ALIGNMENTS = [
    Alignment((0,), (0,), 0.1),
    Alignment((1, 2), (1,), 0.2),
    Alignment((3,), (), 0.9),
    Alignment((), (2,), 0.9),
    Alignment((4,), (3,), 0.3),
]


def _frame(alignments: list[Alignment]) -> pl.DataFrame:
    return aligned_corpus_frame(
        alignments,
        SRC,
        TGT,
        src_lang="khk_Cyrl",
        tgt_lang="eng_Latn",
        corpus="unit",
        doc_id="doc",
    )


def test_each_alignment_with_both_sides_becomes_a_row_and_deletions_do_not() -> None:
    frame = _frame(ALIGNMENTS)

    assert frame["src"].to_list() == ["s0", "s1 s2", "s4"]
    assert frame["tgt"].to_list() == ["t0", "t1", "t3"]


def test_a_group_joins_its_sentences_with_one_space_on_each_side() -> None:
    frame = _frame([Alignment((0, 1, 2), (0, 1), 0.5)])

    assert frame["src"].to_list() == ["s0 s1 s2"]
    assert frame["tgt"].to_list() == ["t0 t1"]


def test_seg_idx_counts_the_emitted_rows_so_a_deletion_leaves_no_gap() -> None:
    assert _frame(ALIGNMENTS)["seg_idx"].to_list() == [0, 1, 2]


def test_rows_carry_the_run_labels_and_a_content_id() -> None:
    frame = _frame(ALIGNMENTS)

    assert frame.schema == CORPUS_SCHEMA
    assert frame["origin"].to_list() == ["mined"] * 3
    assert frame["split"].to_list() == ["train"] * 3
    assert frame["corpus"].to_list() == ["unit"] * 3
    assert frame["doc_id"].to_list() == ["doc"] * 3
    assert frame["src_lang"].to_list() == ["khk_Cyrl"] * 3
    assert frame["tgt_lang"].to_list() == ["eng_Latn"] * 3
    assert frame["id"].to_list() == [
        make_row_id("khk_Cyrl", "eng_Latn", src, tgt)
        for src, tgt in zip(frame["src"], frame["tgt"], strict=True)
    ]


def test_no_alignments_and_only_deletions_give_an_empty_corpus_frame() -> None:
    for alignments in ([], [Alignment((0,), (), 0.9), Alignment((), (0,), 0.9)]):
        frame = _frame(alignments)

        assert frame.height == 0
        assert frame.schema == CORPUS_SCHEMA


def test_the_counts_split_alignments_into_pairs_and_the_two_kinds_of_deletion() -> None:
    assert count_alignments(ALIGNMENTS) == AlignmentCounts(
        aligned=3, unaligned_src=1, unaligned_tgt=1
    )
    assert count_alignments([]) == AlignmentCounts(0, 0, 0)


def test_a_group_counts_once_however_many_sentences_it_holds() -> None:
    counts = count_alignments([Alignment((0, 1, 2), (0,), 0.5)])

    assert counts == AlignmentCounts(aligned=1, unaligned_src=0, unaligned_tgt=0)
