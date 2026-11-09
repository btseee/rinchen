"""Document windows (D9): the sentences of a document as runs of 2 to 4 consecutive
sentences, each side joined with ` <sep> `, which a document-level model trains on.

`docify` takes the corpus rows (`read_corpus_rows` reads the rows of the train and dev
splits and leaves the others empty, so that they are only counted) and, for one split,
cuts the rows that can be in a window (`doc_source.classify_rows`: authentic rows of the
split with a document id, a `seg_idx` and usable text) into windows per `(src_lang,
tgt_lang, corpus, doc_id)` (`doc_pack`: consecutive `seg_idx` only, never across two
corpora, sizes drawn by a seeded hash, shortened to a piece budget). The windows do not
depend on the order of the rows or of the files.

Every row of the input is counted once, in a window or under the reason it is in none
(`rows_read = rows_in_windows + the sum of the reasons`); nothing is dropped without a
count. `docify` is called once per split (`split="train"`, and `"dev"` with
`DEV_WINDOW_SEED` for the validation windows; no other split makes windows). The windows
are cut with the SentencePiece model of the run, which must have `<doc>`, `<sep>` and the
direction tags as single pieces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.config.languages import direction_tag
from rinchen.train.doc_pack import (
    PREFIX_PIECES,
    WINDOW_JOIN,
    Packing,
    one_row_per_position,
    pack_windows,
    window_frame,
    with_piece_counts,
)
from rinchen.train.doc_source import (
    DOC_MARKER,
    SEGMENT_SEPARATOR,
    classify_rows,
    exclusion_reasons,
    read_corpus_rows,
    require_windowed_split,
)
from rinchen.train.validation_corpus import load_processor, require_one_piece
from rinchen.utils.integers import is_strict_int

__all__ = [
    "DEV_WINDOW_SEED",
    "DOC_MARKER",
    "SEGMENT_SEPARATOR",
    "WINDOW_JOIN",
    "DocStats",
    "DocWindows",
    "PieceSummary",
    "docify",
    "exclusion_reasons",
    "read_corpus_rows",
]

# The seed the validation windows are cut with, whatever the seed of the training ones.
DEV_WINDOW_SEED: Final = 0
_P95: Final = 0.95


@dataclass(frozen=True, slots=True)
class PieceSummary:
    """The lowest, mean, 95th-percentile (nearest rank) and highest piece count of the
    windows of one side; zeros when there is no window."""

    min: int
    mean: float
    p95: int
    max: int


@dataclass(frozen=True, slots=True)
class DocStats:
    """What `docify` counted; `to_dict` is JSON-ready.

    `rows_read` is `rows_in_windows` plus the sum of `excluded`, which has every reason
    of `exclusion_reasons(split)` with 0 where nothing was left out. `windows_by_size`
    has every size from 1 to `max_size` (a window is shorter than `min_size` only when
    the piece budget shortened it). `pieces` has the keys `src` (with the tag and
    `<doc>`) and `tgt`."""

    rows_read: int
    rows_in_windows: int
    windows: int
    windows_by_size: dict[int, int]
    excluded: dict[str, int]
    pieces: dict[str, PieceSummary]

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["windows_by_size"] = {
            str(size): count for size, count in self.windows_by_size.items()
        }
        return document


@dataclass(frozen=True, slots=True, eq=False)
class DocWindows:
    """The windows (columns `id`, `src_lang`, `tgt_lang`, `src`, `tgt`, `doc_id`,
    `first_seg`, `n_sentences`, `src_pieces`, `tgt_pieces`, sorted by language pair,
    corpus, document and first segment; `doc_id` is the id the document has in its
    corpus, so it can repeat across corpora) and what was counted. `id` is the row id of
    the joined texts (`corpus.schema.make_row_id`)."""

    frame: pl.DataFrame
    stats: DocStats


def docify(
    rows: pl.DataFrame,
    *,
    spm_model: Path,
    min_size: int = 2,
    max_size: int = 4,
    max_pieces: int,
    seed: int,
    split: str = "train",
) -> DocWindows:
    """Form the windows of the `split` rows of `rows` (see the module docstring); `rows`
    holds the columns `read_corpus_rows` returns, whatever their order or the order of
    their rows.

    `max_pieces` bounds both sides of a window; Eole's own `filtertoolong` also counts
    the two sentence boundary tokens of a target, so a caller that trains with
    `tgt_seq_length` N passes at most N minus 2.

    Raises:
        ValueError: `min_size` is below 1 or above `max_size`; `max_pieces` or `seed` is
            not an integer; `max_pieces` is below 1; `split` is not `"train"` or `"dev"`.
        DataError: `rows` lacks a column; `rows` were read without the split (its rows
            are empty apart from their `split`); a row of the split has no id,
            language, corpus or origin.
        ConfigError: the SentencePiece model cannot be loaded or lacks `<doc>`, `<sep>`
            or the tag of a target language as one piece; a target language is not in
            the registry.
    """
    _require_arguments(min_size, max_size, max_pieces, seed)
    require_windowed_split(split)
    processor = load_processor(spm_model)
    for symbol in (DOC_MARKER, SEGMENT_SEPARATOR):
        require_one_piece(processor, symbol, spm_model, what="document symbol")
    classified = classify_rows(rows, split)
    for language in classified.candidates["tgt_lang"].unique():
        require_one_piece(processor, direction_tag(language), spm_model)

    unique, duplicates = one_row_per_position(classified.candidates)
    sized = with_piece_counts(unique, processor)
    fits = sized.filter(
        (pl.col("src_alone") + PREFIX_PIECES <= max_pieces)
        & (pl.col("tgt_alone") <= max_pieces)
    )
    windows, singles, tails = pack_windows(
        fits, Packing(processor, min_size, max_size, max_pieces, seed)
    )
    frame = window_frame(windows)
    reasons = Counter(
        {
            **classified.reasons,
            "duplicate_seg_idx": duplicates,
            "too_long": sized.height - fits.height,
            "gap_single": singles,
            "tail": tails,
        }
    )
    return DocWindows(frame, _stats(rows.height, frame, reasons, split, max_size))


def _require_arguments(
    min_size: int, max_size: int, max_pieces: int, seed: int
) -> None:
    for name, value in (
        ("min_size", min_size),
        ("max_size", max_size),
        ("max_pieces", max_pieces),
        ("seed", seed),
    ):
        if not is_strict_int(value):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if not 1 <= min_size <= max_size:
        raise ValueError(
            f"min_size and max_size must satisfy 1 <= min_size <= max_size, got "
            f"{min_size} and {max_size}"
        )
    if max_pieces < 1:
        raise ValueError(f"max_pieces must be at least 1, got {max_pieces}")


def _stats(
    rows_read: int,
    frame: pl.DataFrame,
    reasons: Counter[str],
    split: str,
    max_size: int,
) -> DocStats:
    by_size = _counts_by_size(frame)
    return DocStats(
        rows_read=rows_read,
        rows_in_windows=int(frame["n_sentences"].sum()),
        windows=frame.height,
        windows_by_size={size: by_size.get(size, 0) for size in range(1, max_size + 1)},
        excluded={reason: reasons[reason] for reason in exclusion_reasons(split)},
        pieces={
            "src": _summary(frame["src_pieces"]),
            "tgt": _summary(frame["tgt_pieces"]),
        },
    )


def _counts_by_size(frame: pl.DataFrame) -> dict[int, int]:
    return dict(frame.group_by("n_sentences").len().iter_rows())


def _summary(counts: pl.Series) -> PieceSummary:
    if counts.is_empty():
        return PieceSummary(0, 0.0, 0, 0)
    ordered = sorted(counts.to_list())
    nearest_rank = min(len(ordered) - 1, int(_P95 * len(ordered)))
    return PieceSummary(
        min=ordered[0],
        mean=sum(ordered) / len(ordered),
        p95=ordered[nearest_rank],
        max=ordered[-1],
    )
