"""Reading corpus Parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rinchen.corpus.schema import validate_corpus_schema
from rinchen.errors import DataError


def scan_corpus(path: Path) -> pl.LazyFrame:
    """Lazily scan a corpus Parquet file. Raises DataError when the file is
    missing or unreadable, or its schema is not the corpus contract."""
    if not path.is_file():
        raise DataError(f"corpus file not found: {path}")
    try:
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
    except pl.exceptions.PolarsError as exc:
        raise DataError(f"cannot read corpus file {path}: {exc}") from exc
    validate_corpus_schema(schema, source=str(path))
    return lazy
