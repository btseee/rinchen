from __future__ import annotations

import pytest

from rinchen.acquire.registry import (
    CommonCrawlSourceEntry,
    HfSourceEntry,
    LocalSourceEntry,
    OpusSourceEntry,
    SourceKind,
    fetch_source,
)
from rinchen.errors import DataError


def test_fetch_source_local_kind_copies_fixture(tmp_path, monkeypatch):
    import rinchen.acquire.http as http_module

    fixture_root = tmp_path / "repo"
    fixture = fixture_root / "tests" / "fixtures" / "raw" / "sample.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(http_module, "REPO_ROOT", fixture_root)

    entry = LocalSourceEntry(
        id="sample",
        kind=SourceKind.LOCAL,
        langs=["mn"],
        url="local://tests/fixtures/raw/sample.txt",
    )
    dest_dir = tmp_path / "out"

    result = fetch_source(entry, dest_dir)

    assert result.read_text(encoding="utf-8") == "hello"
    assert result.parent == dest_dir


def test_fetch_source_opus_kind_verifies_expected_sha(tmp_path, monkeypatch):
    """`expected_sha` on an `opus` entry must actually be checked -- not
    silently ignored like it previously was for every non-http/local kind."""
    import rinchen.acquire.registry as registry_module

    def fake_opus_fetch(corpus, src, tgt, dest, *, preprocessing, version):
        dest.write_bytes(b"wrong content")
        return dest

    monkeypatch.setattr(registry_module.opus, "fetch", fake_opus_fetch)

    entry = OpusSourceEntry(
        id="tatoeba-mn-en",
        kind=SourceKind.OPUS,
        langs=["mn", "en"],
        corpus="Tatoeba",
        src="mn",
        tgt="en",
        expected_sha="0" * 64,
    )
    dest_dir = tmp_path / "out"

    with pytest.raises(DataError, match="sha256 mismatch"):
        fetch_source(entry, dest_dir)


def test_fetch_source_hf_kind_does_not_double_nest_entry_id(tmp_path, monkeypatch):
    """The CLI (`src/rinchen/cli/data.py`) already passes
    `DATA_RAW_DIR / entry.id` as `dest_dir`; `fetch_source` must not append
    `entry.id` again, or output lands at `data/raw/<id>/<id>/...` instead
    of the intended single-nested `data/raw/<id>/...`."""
    import rinchen.acquire.registry as registry_module

    seen_dirs = []

    def fake_hf_fetch(dataset, dest_dir, *, config, split):
        seen_dirs.append(dest_dir)
        return []

    monkeypatch.setattr(registry_module.hf, "fetch", fake_hf_fetch)

    entry = HfSourceEntry(
        id="wikipedia-mn",
        kind=SourceKind.HF,
        langs=["mn"],
        dataset="wikimedia/wikipedia",
    )
    # Mirrors the CLI: the id is already baked into dest_dir once.
    dest_dir = tmp_path / "data" / "raw" / entry.id

    result = fetch_source(entry, dest_dir)

    assert seen_dirs == [dest_dir]
    assert result == dest_dir


def test_fetch_source_commoncrawl_kind_does_not_double_nest_entry_id(
    tmp_path, monkeypatch
):
    import rinchen.acquire.registry as registry_module

    seen_dirs = []

    def fake_query_cdx(url_pattern, collection):
        return [{"filename": "x", "offset": "0", "length": "1"}]

    def fake_fetch_records(records, dest_dir):
        seen_dirs.append(dest_dir)
        return []

    monkeypatch.setattr(registry_module.commoncrawl, "query_cdx", fake_query_cdx)
    monkeypatch.setattr(
        registry_module.commoncrawl, "fetch_records", fake_fetch_records
    )

    entry = CommonCrawlSourceEntry(
        id="cc-mn",
        kind=SourceKind.COMMONCRAWL,
        langs=["mn"],
        url_pattern="example.test/*",
        collection="CC-MAIN-2026-43",
    )
    dest_dir = tmp_path / "data" / "raw" / entry.id

    result = fetch_source(entry, dest_dir)

    assert seen_dirs == [dest_dir]
    assert result == dest_dir
