"""From alignments to mined corpus rows: the rules that decide which
alignments become rows and how a row is labelled."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from rinchen.corpus.schema import corpus_frame
from rinchen.mine.align import Alignment

MINED_ORIGIN = "mined"
MINED_SPLIT = "train"


@dataclass(frozen=True, slots=True)
class AlignmentCounts:
    """`aligned` counts groups with sentences on both sides, however many
    sentences a group holds; the other two count single-sentence deletions."""

    aligned: int
    unaligned_src: int
    unaligned_tgt: int


def count_alignments(alignments: Sequence[Alignment]) -> AlignmentCounts:
    return AlignmentCounts(
        aligned=sum(1 for a in alignments if a.src and a.tgt),
        unaligned_src=sum(1 for a in alignments if a.src and not a.tgt),
        unaligned_tgt=sum(1 for a in alignments if a.tgt and not a.src),
    )


def aligned_corpus_frame(
    alignments: Sequence[Alignment],
    src: Sequence[str],
    tgt: Sequence[str],
    *,
    src_lang: str,
    tgt_lang: str,
    corpus: str,
    doc_id: str,
) -> pl.DataFrame:
    """One corpus row per alignment with sentences on both sides, in order.
    The sentences of a group are joined by one space, `seg_idx` numbers the
    rows from 0, and a deletion emits no row."""
    pairs = [a for a in alignments if a.src and a.tgt]
    return corpus_frame(
        [
            {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": _joined(src, pair.src),
                "tgt": _joined(tgt, pair.tgt),
                "origin": MINED_ORIGIN,
                "corpus": corpus,
                "split": MINED_SPLIT,
                "doc_id": doc_id,
                "seg_idx": seg_idx,
            }
            for seg_idx, pair in enumerate(pairs)
        ]
    )


def _joined(sentences: Sequence[str], indexes: tuple[int, ...]) -> str:
    return " ".join(sentences[index] for index in indexes)
