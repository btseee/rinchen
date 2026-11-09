"""Cutting the sentences of a document into windows (D9), once `doc_source` has said
which rows can be in one.

A window holds only sentences whose `seg_idx` are consecutive (a gap, or a sentence that
was left out, ends a run) and is cut per `(src_lang, tgt_lang, corpus, doc_id)` over the
sentences sorted by `seg_idx`; a window never holds sentences of two corpora, whatever
their `doc_id` and `seg_idx`. Its size is drawn from `[min_size, max_size]` by a seeded
hash of `(seed, doc_id, seg_idx of its first sentence)`, so the windows do not depend on
the order of the rows or of the files and the two directions of a document get the same
windows. It is shortened, down to one sentence, while its source or its target would be
longer than `max_pieces` SentencePiece pieces; the source counts the direction tag and
`<doc>` that the prefix transform puts in front of it, and both sides are measured on
the JOINED text, so the word-boundary piece (U+2581) in front of a `<sep>` counts. The
last window of a run takes what is left when that is at least `min_size` sentences;
fewer are a `tail` (or `gap_single` when the whole run is shorter than `min_size`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl
import sentencepiece

from rinchen.corpus.schema import make_row_id
from rinchen.train.doc_source import SEGMENT_SEPARATOR

WINDOW_JOIN: Final = f" {SEGMENT_SEPARATOR} "
# The tag and `<doc>`, one piece each, in front of the source of every window.
PREFIX_PIECES: Final = 2

_ENCODE_CHUNK: Final = 20_000
# A document is named within its corpus: D4 names it after the file it came from, so two
# corpora can hold a document with the same id.
_DOCUMENT_KEY: Final = ("src_lang", "tgt_lang", "corpus", "doc_id")
WINDOW_COLUMNS: Final = (
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
)
_WINDOW_SCHEMA: Final = {
    "id": pl.String,
    "src_lang": pl.String,
    "tgt_lang": pl.String,
    "src": pl.String,
    "tgt": pl.String,
    "doc_id": pl.String,
    "first_seg": pl.Int64,
    "n_sentences": pl.Int64,
    "src_pieces": pl.Int64,
    "tgt_pieces": pl.Int64,
}


@dataclass(frozen=True, slots=True)
class Packing:
    """How windows are cut: the model that counts pieces, the bounds of a window's size
    in sentences, the piece budget of each side and the seed of the size draw."""

    processor: sentencepiece.SentencePieceProcessor
    min_size: int
    max_size: int
    max_pieces: int
    seed: int


@dataclass(frozen=True, slots=True)
class Window:
    src_lang: str
    tgt_lang: str
    doc_id: str
    first_seg: int
    sentences: int
    src: str
    tgt: str
    src_pieces: int
    tgt_pieces: int


def one_row_per_position(candidates: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """The candidates with one row per position `(src_lang, tgt_lang, corpus, doc_id,
    seg_idx)` (the first of the rows sorted by id, source and target, so no input order
    decides which) and the number of rows dropped."""
    position = (*_DOCUMENT_KEY, "seg_idx")
    ordered = candidates.sort(*position, "id", "src", "tgt")
    unique = ordered.filter(pl.struct(position).is_first_distinct())
    return unique, ordered.height - unique.height


def with_piece_counts(
    frame: pl.DataFrame, processor: sentencepiece.SentencePieceProcessor
) -> pl.DataFrame:
    """`frame` with `src_alone` and `tgt_alone`, the pieces of each sentence on its own."""
    return frame.with_columns(
        pl.Series("src_alone", _piece_counts(processor, frame["src"]), dtype=pl.Int64),
        pl.Series("tgt_alone", _piece_counts(processor, frame["tgt"]), dtype=pl.Int64),
    )


def _piece_counts(
    processor: sentencepiece.SentencePieceProcessor, texts: pl.Series
) -> list[int]:
    lines = texts.to_list()
    counts: list[int] = []
    for start in range(0, len(lines), _ENCODE_CHUNK):
        encoded = processor.encode(lines[start : start + _ENCODE_CHUNK], out_type=int)
        counts.extend(len(pieces) for pieces in encoded)
    return counts


def pack_windows(
    sentences: pl.DataFrame, packing: Packing
) -> tuple[list[Window], int, int]:
    """The windows of every document of `sentences` (the sentences that fit the budget
    alone, one per position, as `one_row_per_position` leaves them: sorted by language
    pair, corpus, document and `seg_idx`), the number of sentences alone in their run and
    the number left over after the last window of a run."""
    keys = list(
        zip(*(sentences[name].to_list() for name in _DOCUMENT_KEY), strict=True)
    )
    seg_idx = sentences["seg_idx"].to_list()
    src, tgt = sentences["src"].to_list(), sentences["tgt"].to_list()
    windows: list[Window] = []
    singles = tails = 0
    for run in _runs(keys, seg_idx):
        src_lang, tgt_lang, _corpus, doc_id = keys[run.start]
        starts = len(windows)
        position = run.start
        while position < run.stop:
            left = run.stop - position
            if left < packing.min_size:
                if len(windows) == starts:
                    singles += left
                else:
                    tails += left
                break
            window = _window(
                packing, (src_lang, tgt_lang, doc_id), seg_idx, src, tgt, position, left
            )
            windows.append(window)
            position += window.sentences
    return windows, singles, tails


def _runs(keys: Sequence[object], seg_idx: Sequence[int]) -> Iterator[range]:
    """The ranges of positions of one document whose `seg_idx` go up by one."""
    start = 0
    for position in range(1, len(keys) + 1):
        if (
            position == len(keys)
            or keys[position] != keys[start]
            or seg_idx[position] != seg_idx[position - 1] + 1
        ):
            yield range(start, position)
            start = position


def _window(
    packing: Packing,
    document: tuple[str, str, str],
    seg_idx: Sequence[int],
    src: Sequence[str],
    tgt: Sequence[str],
    position: int,
    left: int,
) -> Window:
    """The window that starts at `position` of the run, of the drawn size or the
    sentences that are left, shortened until both sides fit the budget (one sentence
    always does)."""
    src_lang, tgt_lang, doc_id = document
    size = min(_drawn_size(packing, doc_id, seg_idx[position]), left)
    while True:
        joined_src = WINDOW_JOIN.join(src[position : position + size])
        joined_tgt = WINDOW_JOIN.join(tgt[position : position + size])
        src_pieces = PREFIX_PIECES + len(packing.processor.encode(joined_src))
        tgt_pieces = len(packing.processor.encode(joined_tgt))
        if size == 1 or max(src_pieces, tgt_pieces) <= packing.max_pieces:
            return Window(
                src_lang,
                tgt_lang,
                doc_id,
                seg_idx[position],
                size,
                joined_src,
                joined_tgt,
                src_pieces,
                tgt_pieces,
            )
        size -= 1


def _drawn_size(packing: Packing, doc_id: str, first_seg: int) -> int:
    digest = hashlib.sha256(f"{packing.seed}\t{doc_id}\t{first_seg}".encode()).digest()
    span = packing.max_size - packing.min_size + 1
    return packing.min_size + int.from_bytes(digest[:8], "big") % span


def window_frame(windows: Sequence[Window]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [make_row_id(w.src_lang, w.tgt_lang, w.src, w.tgt) for w in windows],
            "src_lang": [w.src_lang for w in windows],
            "tgt_lang": [w.tgt_lang for w in windows],
            "src": [w.src for w in windows],
            "tgt": [w.tgt for w in windows],
            "doc_id": [w.doc_id for w in windows],
            "first_seg": [w.first_seg for w in windows],
            "n_sentences": [w.sentences for w in windows],
            "src_pieces": [w.src_pieces for w in windows],
            "tgt_pieces": [w.tgt_pieces for w in windows],
        },
        schema=_WINDOW_SCHEMA,
    ).select(WINDOW_COLUMNS)
