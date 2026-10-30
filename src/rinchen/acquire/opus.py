"""OPUS (https://opus.nlpl.eu) corpus listing and fetching.

Confirmed live against the real API during D1 design: the base URL must
have no trailing slash before the query string (a trailing slash breaks
the response), the API is source/target-order-independent, and passing
the literal string ``"latest"`` as the ``version`` query parameter
returns zero results -- it is not a recognized filter value. To honor a
caller's request for "the latest version", this module omits ``version``
from the actual query entirely and instead picks the entry (or entries,
for a corpus like Tatoeba that publishes many dated releases) whose
``latest`` field is the string ``"True"``.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from rinchen.acquire.http import download
from rinchen.errors import DataError

OPUS_API_URL = "https://opus.nlpl.eu/opusapi"


def list_corpora(
    src: str,
    tgt: str,
    *,
    preprocessing: str = "moses",
    client: httpx.Client | None = None,
) -> list[dict]:
    """Return every corpus entry OPUS has for `src`/`tgt`, in either order."""
    owns_client = client is None
    http_client = client or httpx.Client()
    try:
        response = http_client.get(
            OPUS_API_URL,
            params={"source": src, "target": tgt, "preprocessing": preprocessing},
        )
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            http_client.close()
    return data.get("corpora", [])


def fetch(
    corpus: str,
    src: str,
    tgt: str,
    dest: Path,
    *,
    preprocessing: str = "moses",
    version: str = "latest",
    client: httpx.Client | None = None,
) -> Path:
    """Find `corpus` in the real OPUS listing for `src`/`tgt` and download it."""
    entries = list_corpora(src, tgt, preprocessing=preprocessing, client=client)
    matches = [entry for entry in entries if entry.get("corpus") == corpus]
    if not matches:
        raise DataError(f"OPUS corpus {corpus!r} not found for {src}-{tgt}")

    if version == "latest":
        latest_matches = [m for m in matches if m.get("latest") == "True"]
        chosen = max(
            latest_matches or matches, key=lambda entry: entry.get("version", "")
        )
    else:
        version_matches = [m for m in matches if m.get("version") == version]
        if not version_matches:
            raise DataError(
                f"OPUS corpus {corpus!r} has no version {version!r} for {src}-{tgt}"
            )
        chosen = version_matches[0]

    return download(chosen["url"], dest, client=client)
