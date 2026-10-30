# tests/unit/test_acquire_commoncrawl.py
from __future__ import annotations

import httpx

from rinchen.acquire import commoncrawl

CDX_NDJSON = (
    '{"urlkey": "test,example)/page1", "timestamp": "20261005112841", '
    '"url": "https://example.test/page1", "filename": "crawl-data/CC-MAIN-2026-43/segments/x/warc/foo.warc.gz", '
    '"offset": "1000", "length": "500", "languages": "mon"}\n'
    '{"urlkey": "test,example)/page2", "timestamp": "20261005112900", '
    '"url": "https://example.test/page2", "filename": "crawl-data/CC-MAIN-2026-43/segments/x/warc/foo.warc.gz", '
    '"offset": "2000", "length": "300", "languages": "eng"}\n'
)


def test_query_cdx_parses_ndjson_lines():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["output"] == "json"
        return httpx.Response(200, text=CDX_NDJSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    records = commoncrawl.query_cdx("example.test/*", "CC-MAIN-2026-43", client=client)

    assert len(records) == 2
    assert records[0]["languages"] == "mon"
    assert records[1]["offset"] == "2000"


def test_fetch_records_does_bounded_range_requests(tmp_path):
    seen_ranges = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        return httpx.Response(206, content=b"warc bytes here")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = [
        {"filename": "crawl-data/x/foo.warc.gz", "offset": "1000", "length": "500"},
        {"filename": "crawl-data/x/foo.warc.gz", "offset": "2000", "length": "300"},
    ]

    paths = commoncrawl.fetch_records(records, tmp_path, client=client)

    assert len(paths) == 2
    assert seen_ranges == ["bytes=1000-1499", "bytes=2000-2299"]
    assert all(p.read_bytes() == b"warc bytes here" for p in paths)
