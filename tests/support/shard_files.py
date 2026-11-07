"""Edits of a shard file on disk that leave it a well-formed shard, which is the
damage the store's structural checks cannot see: another text under the same
ids, or a shard written before shards carried a content digest."""

from __future__ import annotations

from pathlib import Path

import polars as pl

DIGEST_KEY = "rinchen.content_sha256"


def stored_digest(path: Path) -> str:
    """The content digest the store wrote into the shard file at `path`."""
    return pl.read_parquet_metadata(path)[DIGEST_KEY]


def change_a_candidate_keeping_the_digest(path: Path) -> None:
    """Rewrite the shard with `!` appended to its first candidate and the digest
    of the original content left as it was."""
    frame = pl.read_parquet(path, glob=False)
    digest = stored_digest(path)
    candidates = frame["candidates"].to_list()
    first = next(texts for texts in candidates if texts)
    first[0] += "!"
    changed = frame.with_columns(
        pl.Series("candidates", candidates, dtype=frame.schema["candidates"])
    )
    changed.write_parquet(path, compression="zstd", metadata={DIGEST_KEY: digest})


def drop_the_digest(path: Path) -> None:
    """Rewrite the shard unchanged, as a store that wrote no digest would have."""
    pl.read_parquet(path, glob=False).write_parquet(path, compression="zstd")
