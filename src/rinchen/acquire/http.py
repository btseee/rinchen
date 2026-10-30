"""Generic HTTP download with resume, retry/backoff, sha256 verification,
and a `local://` scheme for offline fixture-backed fetches (used by the
registry's `kind: local` entries, see `registry.py`)."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import httpx

from rinchen.errors import DataError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_SCHEME = "local://"

_sleep = time.sleep


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    """Raise `DataError` if `path`'s sha256 doesn't match `expected`
    (deleting `path` first). A no-op when `expected` is None. Public so
    other acquire backends (e.g. `registry.fetch_source`'s `opus` branch)
    can reuse the same check and error type `download()` uses."""
    if expected is None:
        return
    actual = sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise DataError(
            f"sha256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    resume: bool = True,
    retries: int = 5,
    client: httpx.Client | None = None,
) -> Path:
    """Download `url` to `dest`, resuming a partial `.part` file on retry.

    `url` may be `local://<repo-relative-path>` to copy a local fixture
    file instead of making any network request.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if url.startswith(LOCAL_SCHEME):
        source = REPO_ROOT / url[len(LOCAL_SCHEME) :]
        if not source.is_file():
            raise DataError(f"local source not found: {source}")
        shutil.copyfile(source, dest)
        verify_sha256(dest, sha256)
        return dest

    part_path = dest.with_name(dest.name + ".part")
    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True)
    try:
        attempt = 0
        while True:
            attempt += 1
            existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
            headers = (
                {"Range": f"bytes={existing_bytes}-"}
                if resume and existing_bytes
                else {}
            )
            try:
                with http_client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    mode = "ab" if headers else "wb"
                    with part_path.open(mode) as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                break
            except httpx.HTTPError:
                if attempt >= retries:
                    raise
                _sleep(min(2**attempt, 30))
    finally:
        if owns_client:
            http_client.close()

    part_path.replace(dest)
    verify_sha256(dest, sha256)
    return dest
