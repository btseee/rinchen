# tests/unit/test_acquire_warc.py
from __future__ import annotations

from pathlib import Path

from rinchen.acquire import warc

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "raw" / "sample.warc.gz"

GOLDEN = [
    "First Article\nThis is the first paragraph of the first test article.\nIt has a second sentence to give trafilatura enough content.",
    "Second Article\nThis second article discusses a different synthetic topic entirely.\nEnough text is present here for reliable extraction as well.",
    "Third Article\nThe third synthetic article continues the pattern for testing purposes.",
    "Fourth Article\nA fourth short article rounds out the fixture set nicely.\nTwo sentences keep the extraction stable across all records.",
    "Fifth Article\nThe fifth and final synthetic article closes out the sample WARC file.",
]


def test_iterate_yields_five_records():
    records = list(warc.iterate(FIXTURE))
    assert len(records) == 5
    assert records[0].url == "http://example.test/page1"


def test_extract_text_matches_golden_for_every_record():
    records = list(warc.iterate(FIXTURE))
    extracted = [warc.extract_text(r) for r in records]
    assert extracted == GOLDEN
