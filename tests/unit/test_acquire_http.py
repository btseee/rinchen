"""Tests for rinchen.acquire.http.download: resume, retry, sha256, local scheme."""
from __future__ import annotations

import httpx
import pytest

from rinchen.acquire import http as http_module
from rinchen.errors import DataError


class _TruncatingStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks
        raise httpx.ReadError("simulated truncated connection")

    def close(self) -> None:
        pass


def test_download_fetches_full_file_in_one_request(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "range" not in request.headers
        return httpx.Response(200, content=b"hello world")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.bin"

    result = http_module.download("http://example/file", dest, client=client)

    assert result == dest
    assert dest.read_bytes() == b"hello world"
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumes_after_truncated_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(http_module, "_sleep", lambda _seconds: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("range"))
        if len(calls) == 1:
            return httpx.Response(200, stream=_TruncatingStream([b"hello "]))
        assert request.headers.get("range") == "bytes=6-"
        return httpx.Response(206, content=b"world")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.bin"

    result = http_module.download("http://example/file", dest, client=client, retries=5)

    assert result == dest
    assert dest.read_bytes() == b"hello world"
    assert calls == [None, "bytes=6-"]


def test_download_raises_data_error_on_sha_mismatch(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello world")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.bin"

    with pytest.raises(DataError, match="sha256 mismatch"):
        http_module.download(
            "http://example/file",
            dest,
            client=client,
            sha256="0" * 64,
        )

    assert not dest.exists()


def test_download_local_scheme_copies_fixture_file(tmp_path, monkeypatch):
    fixture_root = tmp_path / "repo"
    fixture_root.mkdir()
    fixture = fixture_root / "tests" / "fixtures" / "raw" / "example.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("fixture content", encoding="utf-8")

    monkeypatch.setattr(http_module, "REPO_ROOT", fixture_root)
    dest = tmp_path / "dest" / "out.txt"

    result = http_module.download("local://tests/fixtures/raw/example.txt", dest)

    assert result == dest
    assert dest.read_text(encoding="utf-8") == "fixture content"


def test_download_local_scheme_raises_data_error_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(http_module, "REPO_ROOT", tmp_path)
    dest = tmp_path / "dest" / "out.txt"

    with pytest.raises(DataError, match="local source not found"):
        http_module.download("local://nonexistent.txt", dest)
