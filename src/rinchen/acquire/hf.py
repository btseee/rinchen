"""HuggingFace `datasets` streaming -> Parquet shards.

`write_shards` takes an already-constructed row iterator so it is
unit-testable without any network access, and is exercised directly by
`tests/unit/test_acquire_hf.py`. `fetch` is the thin wrapper that actually
calls `datasets.load_dataset(..., streaming=True)`; a real network-backed
test would violate the "never network-access in unit tests" hard constraint,
so `tests/unit/test_acquire_hf_revision.py` (D13) runs it with
`datasets.load_dataset` replaced by a recording fake. It is NOT reachable via
the tiny-profile `--profile tiny` fixture path: every `sources.tiny.yaml`
entry uses `kind: local`, which `registry.fetch_source` dispatches to
`http.download`, never to `hf.fetch`. `rinchen eval suite fetch` (D13) is its
other caller."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl


def write_shards(
    rows: Iterable[Mapping[str, Any]],
    dest_dir: Path,
    *,
    shard_size: int = 10_000,
    prefix: str = "shard",
) -> list[Path]:
    """Write `rows` to `dest_dir` as Parquet shards of at most `shard_size`
    rows each. Returns the list of shard paths written, in order."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buffer: list[Mapping[str, Any]] = []
    shard_index = 0

    def _flush() -> None:
        nonlocal shard_index
        if not buffer:
            return
        frame = pl.DataFrame(buffer)
        path = dest_dir / f"{prefix}-{shard_index:05d}.parquet"
        frame.write_parquet(path)
        paths.append(path)
        shard_index += 1
        buffer.clear()

    for row in rows:
        buffer.append(dict(row))
        if len(buffer) >= shard_size:
            _flush()
    _flush()

    return paths


def fetch(
    dataset: str,
    dest_dir: Path,
    *,
    config: str | None = None,
    split: str = "train",
    shard_size: int = 10_000,
    revision: str | None = None,
) -> list[Path]:
    """Stream `dataset` (optionally a `config`/`split`) from the Hub and
    write it to `dest_dir` as Parquet shards. `revision` pins the dataset to
    a commit; without one the call to `load_dataset` is the one this function
    always made (no `revision` keyword)."""
    import datasets  # type: ignore[import-untyped]

    pin = {} if revision is None else {"revision": revision}
    stream = datasets.load_dataset(dataset, config, split=split, streaming=True, **pin)
    return write_shards(stream, dest_dir, shard_size=shard_size)
