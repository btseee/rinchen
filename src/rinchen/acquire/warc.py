"""Iterate WARC files and extract article text from their response records."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from warcio.archiveiterator import ArchiveIterator  # type: ignore[import-untyped]
from warcio.recordloader import ArcWarcRecord  # type: ignore[import-untyped]


@dataclass(slots=True, frozen=True)
class WarcRecord:
    url: str
    content: bytes


def iterate(warc_path: Path) -> Iterator[WarcRecord]:
    """Yield each `response`-type record in a (possibly gzipped) WARC file."""
    with warc_path.open("rb") as stream:
        for record in ArchiveIterator(stream):
            if _is_response_record(record):
                yield WarcRecord(
                    url=record.rec_headers.get_header("WARC-Target-URI"),
                    content=record.content_stream().read(),
                )


def _is_response_record(record: ArcWarcRecord) -> bool:
    return record.rec_type == "response"


def extract_text(record: WarcRecord) -> str | None:
    """Extract the main article text from a WARC response record's HTML."""
    return trafilatura.extract(record.content)
