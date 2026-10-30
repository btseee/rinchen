# scripts/build_sample_warc_fixture.py
"""One-time generator for tests/fixtures/raw/sample.warc.gz -- 5 synthetic
HTML pages, written as WARC response records, used to test
rinchen.acquire.warc. Re-run this script if the fixture ever needs
regenerating; the golden extracted text in
tests/unit/test_acquire_warc.py was computed from this exact content and
must be updated together with any change here."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from warcio.statusandheaders import StatusAndHeaders  # type: ignore[import-untyped]
from warcio.warcwriter import WARCWriter  # type: ignore[import-untyped]

PAGES = [
    (
        "http://example.test/page1",
        "First Article",
        [
            "This is the first paragraph of the first test article.",
            "It has a second sentence to give trafilatura enough content.",
        ],
    ),
    (
        "http://example.test/page2",
        "Second Article",
        [
            "This second article discusses a different synthetic topic entirely.",
            "Enough text is present here for reliable extraction as well.",
        ],
    ),
    (
        "http://example.test/page3",
        "Third Article",
        [
            "The third synthetic article continues the pattern for testing purposes.",
        ],
    ),
    (
        "http://example.test/page4",
        "Fourth Article",
        [
            "A fourth short article rounds out the fixture set nicely.",
            "Two sentences keep the extraction stable across all records.",
        ],
    ),
    (
        "http://example.test/page5",
        "Fifth Article",
        [
            "The fifth and final synthetic article closes out the sample WARC file.",
        ],
    ),
]


def _make_html(title: str, paragraphs: list[str]) -> bytes:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><article><h1>{title}</h1>{body}</article></body></html>"
    ).encode()


def main() -> int:
    assert sys.version_info[:2] == (3, 14), sys.version
    out_path = Path("tests/fixtures/raw/sample.warc.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        writer = WARCWriter(handle, gzip=True)
        for url, title, paragraphs in PAGES:
            html = _make_html(title, paragraphs)
            headers = StatusAndHeaders(
                "200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.0"
            )
            record = writer.create_warc_record(
                url, "response", payload=io.BytesIO(html), http_headers=headers
            )
            writer.write_record(record)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
