# tests/unit/test_acquire_hf.py
from __future__ import annotations

import polars as pl

from rinchen.acquire import hf


def test_write_shards_splits_rows_into_shards_of_given_size(tmp_path):
    rows = [{"id": i, "text": f"row {i}"} for i in range(25)]

    paths = hf.write_shards(rows, tmp_path, shard_size=10)

    assert len(paths) == 3
    counts = [pl.read_parquet(p).height for p in paths]
    assert counts == [10, 10, 5]


def test_write_shards_preserves_row_content(tmp_path):
    rows = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

    paths = hf.write_shards(rows, tmp_path, shard_size=10)

    assert len(paths) == 1
    frame = pl.read_parquet(paths[0])
    assert frame.to_dicts() == rows


def test_write_shards_returns_empty_list_for_no_rows(tmp_path):
    paths = hf.write_shards([], tmp_path)

    assert paths == []
