"""CommonCrawl CDX index queries and bounded-range WARC record fetches."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

CDX_URL_TEMPLATE = "https://index.commoncrawl.org/{collection}-index"
WARC_BASE_URL = "https://data.commoncrawl.org/"


def query_cdx(
    url_pattern: str,
    collection: str,
    *,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Query the CommonCrawl CDX index; returns one dict per matched record
    (real shape: newline-delimited JSON, one record per line)."""
    owns_client = client is None
    http_client = client or httpx.Client()
    try:
        response = http_client.get(
            CDX_URL_TEMPLATE.format(collection=collection),
            params={"url": url_pattern, "output": "json"},
        )
        response.raise_for_status()
        text = response.text
    finally:
        if owns_client:
            http_client.close()

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_records(
    records: list[dict],
    dest_dir: Path,
    *,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Fetch each CDX record's WARC bytes via a single bounded HTTP Range
    request (not a resumable large-file download -- each record's length
    is already known from the CDX entry)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http_client = client or httpx.Client()
    paths: list[Path] = []
    try:
        for index, record in enumerate(records):
            offset = int(record["offset"])
            length = int(record["length"])
            range_header = {"Range": f"bytes={offset}-{offset + length - 1}"}
            url = WARC_BASE_URL + record["filename"]
            response = http_client.get(url, headers=range_header)
            response.raise_for_status()
            out_path = dest_dir / f"record-{index}.warc.gz"
            out_path.write_bytes(response.content)
            paths.append(out_path)
    finally:
        if owns_client:
            http_client.close()
    return paths
