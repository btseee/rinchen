"""Writing corpus Parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rinchen.corpus.schema import CORPUS_SCHEMA, validate_corpus_schema
from rinchen.utils.part_file import unique_part_path


def write_corpus(frame: pl.DataFrame | pl.LazyFrame, path: Path) -> int:
    """Validate `frame` against the corpus contract and write it as zstd
    Parquet in contract column order. Returns the row count.

    The file appears at `path` only once complete: it is written to a unique
    temporary file beside the target and renamed over it, so a failure leaves
    any earlier file intact and a concurrent writer's file alone."""
    lazy = frame.lazy()
    validate_corpus_schema(lazy.collect_schema(), source=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)

    part_path = unique_part_path(path)
    try:
        lazy.select(CORPUS_SCHEMA.names()).sink_parquet(part_path, compression="zstd")
        # The path is a file name: `[` or `*` in it must not match another file.
        written = pl.scan_parquet(part_path, glob=False)
        row_count = written.select(pl.len()).collect().item()
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)
    return int(row_count)
