# tests/unit/test_acquire_opus.py
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rinchen.acquire import opus as opus_module
from rinchen.errors import DataError

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "raw"

FAKE_RESPONSE = {
    "corpora": [
        {
            "corpus": "Tatoeba",
            "source": "mn",
            "target": "en",
            "preprocessing": "moses",
            "version": "v2021-03-10",
            "latest": "False",
            "size": 23,
            "url": "https://example.test/OPUS-Tatoeba/v2021-03-10/moses/en-mn.txt.zip",
        },
        {
            "corpus": "Tatoeba",
            "source": "mn",
            "target": "en",
            "preprocessing": "moses",
            "version": "v2026-07-08",
            "latest": "True",
            "size": 21,
            "url": "https://example.test/OPUS-Tatoeba/v2026-07-08/moses/en-mn.txt.zip",
        },
        {
            "corpus": "QED",
            "source": "mn",
            "target": "en",
            "preprocessing": "moses",
            "version": "v2.0a",
            "latest": "True",
            "size": 1491,
            "url": "https://example.test/OPUS-QED/v2.0a/moses/en-mn.txt.zip",
        },
    ]
}


def test_list_corpora_never_sends_literal_version_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "version" not in request.url.params
        return httpx.Response(200, json=FAKE_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = opus_module.list_corpora("mn", "en", client=client)

    assert len(result) == 3
    assert result[0]["corpus"] == "Tatoeba"


_CONTENT_BY_URL = {
    "https://example.test/OPUS-Tatoeba/v2021-03-10/moses/en-mn.txt.zip": b"tatoeba-not-latest-bytes",
    "https://example.test/OPUS-Tatoeba/v2026-07-08/moses/en-mn.txt.zip": b"tatoeba-latest-bytes",
    "https://example.test/OPUS-QED/v2.0a/moses/en-mn.txt.zip": b"qed-latest-bytes",
}


def test_fetch_picks_the_entry_marked_latest_true(tmp_path):
    """Each mocked version's download URL returns distinguishable content,
    so this test actually fails if `fetch()` picks the wrong entry -- a
    mutant that always downloaded the first/last match (instead of the one
    tagged `latest: true`) previously passed undetected because every URL
    returned identical bytes."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "opus.nlpl.eu":
            return httpx.Response(200, json=FAKE_RESPONSE)
        content = _CONTENT_BY_URL[str(request.url)]
        return httpx.Response(200, content=content)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "tatoeba.zip"

    result = opus_module.fetch("Tatoeba", "mn", "en", dest, client=client)

    assert result == dest
    assert dest.read_bytes() == b"tatoeba-latest-bytes"


def test_fetch_raises_data_error_for_unknown_corpus(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=FAKE_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.zip"

    with pytest.raises(DataError, match="not found"):
        opus_module.fetch("NoSuchCorpus", "mn", "en", dest, client=client)


def test_list_corpora_parses_the_real_fixture_response_shape():
    import json

    fixture = FIXTURES_DIR / "opus_api_mn_en.json"
    fixture_data = json.loads(fixture.read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture_data)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = opus_module.list_corpora("mn", "en", client=client)

    assert len(result) >= 1
    assert "corpus" in result[0]
    assert "url" in result[0]
