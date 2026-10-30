from __future__ import annotations

from rinchen.acquire.registry import LocalSourceEntry, SourceKind, fetch_source


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
