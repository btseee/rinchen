"""The rows document windows are cut from (D9): reading the corpus files whole and
saying, row by row, why a row cannot be in a window.

The classification follows `mixture`: the first reason that applies decides, so every row
is counted once, and the empty and line-break rules are D7's (`mixture.blank_text`,
`mixture.has_line_break`). Only authentic rows make windows, as in D7's training corpora:
the rows of the other split, the filtered ones and the ones of an excluded origin (`ft`,
`synth_llm`, `synth_nmt`, and `bt`, which D9 does not window) are counted and left out. A
row with no document id or no `seg_idx` has no place in a document; a sentence that
holds the literal `<sep>` would be a false boundary in a window.

The reasons that need the other rows of a document (`duplicate_seg_idx`, `too_long`,
`gap_single`, `tail`) are `doc_pack`'s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from rinchen.corpus.readers import scan_corpus
from rinchen.errors import DataError
from rinchen.train.mixture import (
    BT_ORIGIN,
    EXCLUDED_ORIGINS,
    blank_text,
    has_line_break,
)

DOC_MARKER: Final = "<doc>"
SEGMENT_SEPARATOR: Final = "<sep>"

REQUIRED_COLUMNS: Final = (
    "id",
    "src_lang",
    "tgt_lang",
    "src",
    "tgt",
    "origin",
    "split",
    "filter_reason",
    "doc_id",
    "seg_idx",
)
_REASON: Final = "reason"


def exclusion_reasons(split: str) -> tuple[str, ...]:
    """The reasons a row of the corpus is in no window, in the order they are tried;
    `split` is the split that makes windows (its rows are `not_<split>` otherwise)."""
    return (
        f"not_{split}",
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


@dataclass(frozen=True, slots=True, eq=False)
class ClassifiedRows:
    """The rows that can be in a window (the columns of `REQUIRED_COLUMNS`) and the
    number of the others under each reason that `classify_rows` knows."""

    candidates: pl.DataFrame
    reasons: dict[str, int]


def read_corpus_rows(paths: Sequence[Path]) -> pl.DataFrame:
    """Every row of the corpus files, with the columns `docify` reads (`origin` and
    `split` as text). The files are read whole and by name, never as a glob.

    Raises:
        DataError: no file is given; a file is missing, cannot be read or is not a
            corpus table. A damaged file can make the Parquet reader panic, which polars
            raises as a BaseException (docs/VERIFY.md, D-POLARS-PANIC): it is a bad file
            too.
    """
    if not paths:
        raise DataError("no corpus file was given")
    frames = []
    for path in paths:
        try:
            frames.append(
                scan_corpus(path)
                .select(
                    *REQUIRED_COLUMNS[:5],
                    pl.col("origin").cast(pl.String),
                    pl.col("split").cast(pl.String),
                    *REQUIRED_COLUMNS[7:],
                )
                .collect()
            )
        except DataError:
            raise
        except (
            pl.exceptions.PolarsError,
            pl.exceptions.PanicException,
            OSError,
        ) as exc:
            raise DataError(f"cannot read corpus file {path}: {exc}") from exc
    return pl.concat(frames)


def classify_rows(rows: pl.DataFrame, split: str) -> ClassifiedRows:
    """Sort the rows of `rows` (the columns `read_corpus_rows` returns, in any order) into
    those that can be in a window of the `split` and those that cannot, and count the
    latter by the first reason that applies: `not_<split>`, `filtered`,
    `excluded_origin`, `no_doc_id`, `no_seg_idx`, `empty`, `line_break`, `contains_sep`.

    Raises:
        DataError: `rows` lacks a column; a row that can be in a window has no id,
            language or origin.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in rows.columns]
    if missing:
        raise DataError(f"the corpus rows lack the columns {missing}")
    classified = rows.select(REQUIRED_COLUMNS).with_columns(_reason(split))
    candidates = classified.filter(pl.col(_REASON).is_null())
    for column in ("id", "src_lang", "tgt_lang", "origin"):
        absent = candidates[column].null_count()
        if absent:
            raise DataError(
                f"{absent} {split} rows that would be windowed have no {column}"
            )
    left_out = classified.filter(pl.col(_REASON).is_not_null())
    return ClassifiedRows(
        candidates=candidates.drop(_REASON),
        reasons={
            reason: count
            for reason, count in left_out.group_by(_REASON).len().iter_rows()
        },
    )


def _reason(split: str) -> pl.Expr:
    """The first reason a row cannot be in a window, null for a row that can."""
    return (
        pl.when(pl.col("split").is_null() | (pl.col("split") != split))
        .then(pl.lit(f"not_{split}"))
        .when(pl.col("filter_reason").is_not_null())
        .then(pl.lit("filtered"))
        .when(pl.col("origin").is_in((BT_ORIGIN, *EXCLUDED_ORIGINS)))
        .then(pl.lit("excluded_origin"))
        .when(blank_text("doc_id"))
        .then(pl.lit("no_doc_id"))
        .when(pl.col("seg_idx").is_null())
        .then(pl.lit("no_seg_idx"))
        .when(blank_text("src") | blank_text("tgt"))
        .then(pl.lit("empty"))
        .when(has_line_break("src") | has_line_break("tgt"))
        .then(pl.lit("line_break"))
        .when(_has_separator("src") | _has_separator("tgt"))
        .then(pl.lit("contains_sep"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(_REASON)
    )


def _has_separator(column: str) -> pl.Expr:
    return pl.col(column).str.contains(SEGMENT_SEPARATOR, literal=True)
