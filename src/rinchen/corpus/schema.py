"""The corpus Parquet contract: column layout, row ids and frame construction.

This is the master plan's `<contracts>` "Corpus row" and "Monolingual row".
Every column is nullable; a null score means "not scored yet".
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl

from rinchen.errors import DataError

ORIGINS: tuple[str, ...] = (
    "human",
    "web",
    "mined",
    "bt",
    "ft",
    "synth_llm",
    "synth_nmt",
    "converted",
)
SPLITS: tuple[str, ...] = ("train", "dev", "test", "holdout")

SCORES_DTYPE = pl.Struct(
    {
        "lid_src": pl.Float32,
        "lid_tgt": pl.Float32,
        "script_src": pl.Float32,
        "script_tgt": pl.Float32,
        "len_ratio": pl.Float32,
        "num_ok": pl.Boolean,
        "labse": pl.Float32,
        "metricx_qe": pl.Float32,
        "cometkiwi": pl.Float32,
    }
)

CORPUS_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "src_lang": pl.String(),
        "tgt_lang": pl.String(),
        "src": pl.String(),
        "tgt": pl.String(),
        "origin": pl.Enum(ORIGINS),
        "corpus": pl.String(),
        "doc_id": pl.String(),
        "seg_idx": pl.Int32(),
        "split": pl.Enum(SPLITS),
        "scores": SCORES_DTYPE,
        "filter_reason": pl.String(),
        "created": pl.Datetime("us", "UTC"),
    }
)

MONO_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "lang": pl.String(),
        "text": pl.String(),
        "corpus": pl.String(),
        "doc_id": pl.String(),
        "seg_idx": pl.Int32(),
        "url": pl.String(),
        "scores": pl.Struct({"lid": pl.Float32, "script": pl.Float32}),
    }
)

_REQUIRED_KEYS = ("src_lang", "tgt_lang", "src", "tgt", "origin", "corpus", "split")
_OPTIONAL_KEYS = ("doc_id", "seg_idx")


def make_row_id(src_lang: str, tgt_lang: str, src: str, tgt: str) -> str:
    """First 16 hex characters of the SHA-256 of the four fields joined by TAB."""
    joined = f"{src_lang}\t{tgt_lang}\t{src}\t{tgt}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def validate_corpus_schema(schema: pl.Schema, *, source: str) -> None:
    """Raise DataError unless `schema` has exactly the corpus columns and
    dtypes. Column order is not checked; `source` names the data in the error."""
    missing = [name for name in CORPUS_SCHEMA.names() if name not in schema]
    extra = [name for name in schema.names() if name not in CORPUS_SCHEMA]
    mistyped = [
        f"{name} (expected {CORPUS_SCHEMA[name]}, got {schema[name]})"
        for name in CORPUS_SCHEMA.names()
        if name in schema and schema[name] != CORPUS_SCHEMA[name]
    ]
    problems = [
        f"{label}: {', '.join(names)}"
        for label, names in (
            ("missing columns", missing),
            ("unexpected columns", extra),
            ("wrong dtype", mistyped),
        )
        if names
    ]
    if problems:
        raise DataError(f"{source}: not a corpus table; {'; '.join(problems)}")


def corpus_frame(
    rows: Sequence[Mapping[str, Any]], *, created: datetime | None = None
) -> pl.DataFrame:
    """Build a corpus frame from row mappings.

    Each row needs the required keys (src_lang, tgt_lang, src, tgt, origin,
    corpus, split) and may carry `doc_id` and `seg_idx`. `id` is derived;
    `scores` and `filter_reason` start null. `created` (timezone-aware,
    default now) is stamped on every row."""
    stamp = _utc_timestamp(created)
    for index, row in enumerate(rows):
        _check_row(index, row)

    count = len(rows)
    columns: dict[str, list[Any]] = {
        "id": [
            make_row_id(r["src_lang"], r["tgt_lang"], r["src"], r["tgt"]) for r in rows
        ],
        "scores": [None] * count,
        "filter_reason": [None] * count,
        "created": [stamp] * count,
    }
    for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS):
        columns[key] = [row.get(key) for row in rows]
    return pl.DataFrame(
        [_series(name, columns[name]) for name in CORPUS_SCHEMA.names()]
    )


def _utc_timestamp(created: datetime | None) -> datetime:
    if created is None:
        return datetime.now(UTC)
    if created.tzinfo is None:
        raise DataError(f"created must carry a timezone, got naive {created!r}")
    return created.astimezone(UTC)


def _check_row(index: int, row: Mapping[str, Any]) -> None:
    unexpected = sorted(set(row) - {*_REQUIRED_KEYS, *_OPTIONAL_KEYS})
    if unexpected:
        raise DataError(f"row {index}: unexpected keys: {', '.join(unexpected)}")
    unusable = [key for key in _REQUIRED_KEYS if not isinstance(row.get(key), str)]
    if unusable:
        raise DataError(
            f"row {index}: required keys missing or not text: {', '.join(unusable)}"
        )
    for key, allowed in (("origin", ORIGINS), ("split", SPLITS)):
        if row[key] not in allowed:
            raise DataError(
                f"row {index}: {key} {row[key]!r} is not one of {', '.join(allowed)}"
            )


def _series(name: str, values: list[Any]) -> pl.Series:
    dtype = CORPUS_SCHEMA[name]
    try:
        return pl.Series(name, values, dtype=dtype)
    except (TypeError, OverflowError, pl.exceptions.PolarsError) as exc:
        raise DataError(f"column {name}: values do not fit {dtype}: {exc}") from exc
