# D1 — Data Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement D1 of the master plan: the data-acquisition machinery
(`acquire/http.py`, `acquire/opus.py`, `acquire/hf.py`,
`acquire/commoncrawl.py`, `acquire/warc.py`, `acquire/registry.py`), a
schema-validated registry config, the `rinchen data fetch`/`ls` CLI
commands, and proof that `rinchen data fetch --profile tiny --all` works
fully offline using `local://` fixtures — D1's own STOP gate.

**Architecture:** Each acquisition source "kind" (`http`, `opus`, `hf`,
`commoncrawl`, `local`) is its own small module with a narrow, independently
testable function. `registry.py` validates `configs/data/sources.yaml`
against a pydantic discriminated union and dispatches each entry to the
right module's fetch function. No real production sources are seeded in
`sources.yaml` — only the registry schema and offline test fixtures are
needed to prove the machinery works (see Global Constraints).

**Tech Stack:** `httpx` (HTTP client, already pinned), `pyyaml`, `pydantic`,
`warcio`, `trafilatura`, `datasets`, `polars`/`pyarrow` (all already
pinned), `typer`.

## Global Constraints

- Every Python invocation runs on CPython 3.14.x. No 3.12/3.13 fallback.
- **`configs/data/sources.yaml` ships with zero real production entries.**
  Before building this machinery, 7 candidate real-world parallel corpora
  were vetted with a 100-random-sample, zero-tolerance review; 6 of 7
  failed (structural misalignment, mistranslation, a galig/Latin-script
  hard-constraint violation, or character-encoding defects) and the user
  decided to build Rinchen's own standard dataset from scratch rather than
  seed from any of them — see `DECISIONS.md`'s 2026-10-30 entry for the
  full per-dataset verdict. Do not add any of the 7 reviewed candidates
  (TED2020, QED, NeuLab-TedTalks, GNOME, Tatoeba, `ugshanyu/1millionMongolianSentence`,
  `davidstap/NTREX`) to `sources.yaml` as part of this plan.
- **Never network-access in unit tests.** Every test in this plan uses
  `httpx.MockTransport` or local fixtures — no test makes a real HTTP
  request. `acquire/hf.py`'s actual `datasets.load_dataset(...)` call is
  the one real network-touching function in this milestone and is
  deliberately NOT unit-tested (see Task 6) — **correction, 2026-10-30
  post-hoc review:** it has no test coverage at all and is not reachable
  via the tiny-profile fixture path — every tiny-profile entry is
  `kind: local`, which dispatches to `http.download`, never to
  `hf.fetch`. See `STATE.md`'s "2026-10-30 post-hoc review fixes"
  paragraph.
- **All test fixtures are hand-authored, obviously-synthetic content** —
  none are derived from the 7 reviewed-and-rejected/borderline real
  corpora, keeping "data used to test the machinery" cleanly separate from
  "data judged fit for training."
- `ConfigError` is for malformed configuration (invalid `sources.yaml`
  schema); `DataError` is for corrupt/mismatched *data* (sha256 mismatch,
  a referenced local fixture missing). Both already exist in
  `src/rinchen/errors.py` from D0 — do not redefine them.
- The OPUS API has two confirmed-live gotchas, both already designed
  around in this plan's exact code (do not "fix" them differently):
  (1) passing the literal string `"latest"` as the `version` query
  parameter returns zero results from the real API — never send it;
  instead fetch the unfiltered list and pick the entry whose `latest`
  field is `"True"`. (2) The API is source/target-order-independent;
  don't assume a corpus only appears under one specific src/tgt ordering.

---

### Task 1: `acquire/http.py` — download with resume, retry, sha256, `local://` scheme

**Files:**
- Create: `src/rinchen/acquire/__init__.py` (empty)
- Create: `src/rinchen/acquire/http.py`
- Create: `tests/unit/test_acquire_http.py`

**Interfaces:**
- Consumes: `rinchen.errors.DataError`.
- Produces: `download(url, dest, *, sha256=None, resume=True, retries=5,
  client=None) -> Path`. Every later acquisition module (`opus.py`,
  `registry.py`'s dispatch in Task 7) calls this for any `http`/`opus`/
  `local`-kind fetch. Also produces `sha256_of(path) -> str` (used by
  other modules if they need to hash a file).

This exact implementation was verified live during planning (all 5 tests
below were written and run against it successfully) — transcribe it
exactly, don't redesign the resume/retry mechanism.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_acquire_http.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_acquire_http.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.acquire'`.

- [ ] **Step 3: Write `src/rinchen/acquire/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/rinchen/acquire/http.py`**

```python
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


def _verify_sha256(path: Path, expected: str | None) -> None:
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
        _verify_sha256(dest, sha256)
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

    part_path.rename(dest)
    _verify_sha256(dest, sha256)
    return dest
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_http.py -v`
Expected: all 5 pass.

- [ ] **Step 6: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire tests/unit/test_acquire_http.py`
Run: `uv run mypy src/rinchen/acquire`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add src/rinchen/acquire/__init__.py src/rinchen/acquire/http.py tests/unit/test_acquire_http.py
git commit -m "D1: add http.py (resumable download, retry, sha256, local:// scheme)"
```

---

### Task 2: `acquire/registry.py` — schema-validated source registry

**Files:**
- Create: `src/rinchen/acquire/registry.py`
- Create: `configs/data/sources.yaml`
- Create: `tests/unit/test_acquire_registry.py`

**Interfaces:**
- Consumes: `rinchen.errors.ConfigError`.
- Produces: `SourceKind` (enum), `SourceEntry` (discriminated-union type
  alias), `load_registry(path) -> list[SourceEntry]`,
  `find_entry(entries, source_id) -> SourceEntry`. Task 7 extends this
  file with a `fetch_source(entry, dest_dir) -> Path` dispatch function
  once `opus.py`/`hf.py`/`commoncrawl.py` exist (Tasks 3, 5, 6) — do not
  add dispatch logic in this task, only schema + loading.

This exact schema was verified live during planning (all 7 tests below
were run against it successfully) — transcribe it exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_acquire_registry.py
from __future__ import annotations

import pytest

from rinchen.acquire.registry import find_entry, load_registry
from rinchen.errors import ConfigError

VALID_YAML = """
sources:
  - id: tatoeba-mn-en
    kind: opus
    langs: [mn, en]
    corpus: Tatoeba
    src: mn
    tgt: en
  - id: wikipedia-mn
    kind: hf
    langs: [mn]
    dataset: wikimedia/wikipedia
    config: "20231101.mn"
  - id: fixture-source
    kind: local
    langs: [mn, en]
    url: "local://tests/fixtures/raw/moses_mn_en.zip"
"""

MISSING_FIELD_YAML = """
sources:
  - id: broken-opus
    kind: opus
    langs: [mn, en]
"""

UNKNOWN_KIND_YAML = """
sources:
  - id: broken
    kind: ftp
    langs: [mn]
    url: "ftp://example.test/file"
"""

EXTRA_FIELD_YAML = """
sources:
  - id: broken-local
    kind: local
    langs: [mn]
    url: "local://x"
    unexpected_field: 1
"""


def test_load_registry_parses_valid_entries(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    entries = load_registry(path)

    assert len(entries) == 3
    assert entries[0].id == "tatoeba-mn-en"
    assert entries[0].kind.value == "opus"
    assert entries[1].dataset == "wikimedia/wikipedia"
    assert entries[2].url == "local://tests/fixtures/raw/moses_mn_en.zip"


def test_load_registry_rejects_missing_required_field(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(MISSING_FIELD_YAML, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_registry(path)


def test_load_registry_rejects_unknown_kind(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(UNKNOWN_KIND_YAML, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_registry(path)


def test_load_registry_rejects_extra_field(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(EXTRA_FIELD_YAML, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_registry(path)


def test_load_registry_raises_config_error_when_file_missing(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_registry(tmp_path / "nonexistent.yaml")


def test_find_entry_returns_matching_entry(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    entries = load_registry(path)

    found = find_entry(entries, "wikipedia-mn")

    assert found.id == "wikipedia-mn"


def test_find_entry_raises_config_error_when_not_found(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    entries = load_registry(path)

    with pytest.raises(ConfigError, match="no source"):
        find_entry(entries, "nonexistent")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_acquire_registry.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.acquire.registry'`.

- [ ] **Step 3: Write `src/rinchen/acquire/registry.py`**

```python
"""Source registry: validates `configs/data/sources.yaml`-shaped files
against a `SourceEntry` discriminated union and loads them for fetching."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from rinchen.errors import ConfigError


class SourceKind(str, Enum):
    HTTP = "http"
    OPUS = "opus"
    HF = "hf"
    COMMONCRAWL = "commoncrawl"
    LOCAL = "local"


class _BaseSourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    langs: list[str]
    notes: str = ""
    expected_sha: str | None = None


class HttpSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.HTTP]
    url: str


class OpusSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.OPUS]
    corpus: str
    src: str
    tgt: str
    preprocessing: str = "moses"
    version: str = "latest"


class HfSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.HF]
    dataset: str
    config: str | None = None
    split: str = "train"


class CommonCrawlSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.COMMONCRAWL]
    url_pattern: str
    collection: str


class LocalSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.LOCAL]
    url: str


SourceEntry = Annotated[
    HttpSourceEntry
    | OpusSourceEntry
    | HfSourceEntry
    | CommonCrawlSourceEntry
    | LocalSourceEntry,
    Field(discriminator="kind"),
]

_LIST_ADAPTER = TypeAdapter(list[SourceEntry])


def load_registry(path: Path) -> list[SourceEntry]:
    """Load and validate a `sources.yaml`-shaped file. Raises ConfigError
    on any schema violation (missing field, unknown `kind`, extra field)."""
    if not path.is_file():
        raise ConfigError(f"registry file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries = raw.get("sources", [])
    try:
        return _LIST_ADAPTER.validate_python(entries)
    except ValidationError as exc:
        raise ConfigError(f"invalid registry {path}: {exc}") from exc


def find_entry(entries: list[SourceEntry], source_id: str) -> SourceEntry:
    for entry in entries:
        if entry.id == source_id:
            return entry
    raise ConfigError(f"no source with id {source_id!r} in registry")
```

- [ ] **Step 4: Write `configs/data/sources.yaml`**

```yaml
# Rinchen's real, production data-source registry.
#
# This file is deliberately empty of real entries as of D1. Before any
# source is added here, it must pass the same 100-random-sample,
# zero-tolerance translation/script/alignment review used during D1's own
# design -- see DECISIONS.md's 2026-10-30 entry, which rejected 6 of 7
# candidates reviewed that way (one hard-constraint violation: galig/Latin
# script; the rest: systematic misalignment, mistranslation, or character-
# encoding defects). Rinchen is building its own standard dataset from
# scratch rather than seeding from any of those candidates.
#
# Schema: see src/rinchen/acquire/registry.py's SourceEntry union.
sources: []
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_registry.py -v`
Expected: all 7 pass.

- [ ] **Step 6: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire tests/unit/test_acquire_registry.py`
Run: `uv run mypy src/rinchen/acquire`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add src/rinchen/acquire/registry.py configs/data/sources.yaml tests/unit/test_acquire_registry.py
git commit -m "D1: add schema-validated source registry (sources.yaml ships empty, see DECISIONS.md)"
```

---

### Task 3: `acquire/opus.py` — OPUS API client

**Files:**
- Create: `src/rinchen/acquire/opus.py`
- Create: `tests/fixtures/raw/opus_api_mn_en.json`
- Create: `tests/unit/test_acquire_opus.py`

**Interfaces:**
- Consumes: `rinchen.acquire.http.download`, `rinchen.errors.DataError`.
- Produces: `list_corpora(src, tgt, *, preprocessing="moses", client=None)
  -> list[dict]`, `fetch(corpus, src, tgt, dest, *, preprocessing="moses",
  version="latest", client=None) -> Path`.

This exact implementation was verified live during planning against both
the real OPUS API gotchas (see Global Constraints) and the tests below —
transcribe it exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_acquire_opus.py
from __future__ import annotations

import httpx
import pytest

from rinchen.acquire import opus as opus_module
from rinchen.errors import DataError

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


def test_fetch_picks_the_entry_marked_latest_true(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "opus.nlpl.eu":
            return httpx.Response(200, json=FAKE_RESPONSE)
        return httpx.Response(200, content=b"fake zip bytes")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "tatoeba.zip"

    result = opus_module.fetch("Tatoeba", "mn", "en", dest, client=client)

    assert result == dest
    assert dest.read_bytes() == b"fake zip bytes"


def test_fetch_raises_data_error_for_unknown_corpus(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=FAKE_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "out.zip"

    with pytest.raises(DataError, match="not found"):
        opus_module.fetch("NoSuchCorpus", "mn", "en", dest, client=client)


def test_list_corpora_parses_the_real_fixture_response_shape():
    import json
    from pathlib import Path

    fixture = Path("tests/fixtures/raw/opus_api_mn_en.json")
    fixture_data = json.loads(fixture.read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture_data)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = opus_module.list_corpora("mn", "en", client=client)

    assert len(result) >= 1
    assert "corpus" in result[0]
    assert "url" in result[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_acquire_opus.py -v`
Expected: `test_list_corpora_parses_the_real_fixture_response_shape` fails
with `FileNotFoundError` (fixture doesn't exist yet); the rest fail with
`ModuleNotFoundError: No module named 'rinchen.acquire.opus'`.

- [ ] **Step 3: Write `src/rinchen/acquire/opus.py`**

```python
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
```

- [ ] **Step 4: Write `tests/fixtures/raw/opus_api_mn_en.json`**

A realistic but entirely fictional OPUS API response (structurally
accurate to the real confirmed shape, invented corpus name/URL — this is
NOT a captured real response, per the Global Constraints' fixture rule):

```json
{
  "corpora": [
    {
      "alignment_pairs": 42,
      "corpus": "FixtureCorpus",
      "documents": "",
      "id": 999999,
      "latest": "True",
      "preprocessing": "moses",
      "size": 5,
      "source": "mn",
      "source_tokens": 300,
      "target": "en",
      "target_tokens": 280,
      "url": "https://example.test/OPUS-FixtureCorpus/v1/moses/en-mn.txt.zip",
      "version": "v1"
    }
  ]
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_opus.py -v`
Expected: all 4 pass.

- [ ] **Step 6: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire tests/unit/test_acquire_opus.py`
Run: `uv run mypy src/rinchen/acquire`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add src/rinchen/acquire/opus.py tests/fixtures/raw/opus_api_mn_en.json tests/unit/test_acquire_opus.py
git commit -m "D1: add OPUS API client (list_corpora, fetch)"
```

---

### Task 4: `acquire/warc.py` — WARC iteration and text extraction

**Files:**
- Create: `src/rinchen/acquire/warc.py`
- Create: `scripts/build_sample_warc_fixture.py`
- Create: `tests/fixtures/raw/sample.warc.gz` (generated by the script above)
- Create: `tests/unit/test_acquire_warc.py`

**Interfaces:**
- Produces: `WarcRecord` (dataclass: `url: str, content: bytes`),
  `iterate(warc_path) -> Iterator[WarcRecord]`, `extract_text(record) ->
  str | None`.

This exact implementation, fixture-generation script, and golden text
were verified live during planning — transcribe them exactly.

- [ ] **Step 1: Write `src/rinchen/acquire/warc.py`**

(Written before the fixture script here because the fixture-generation
script needs nothing from this module and the golden-text test needs
both to exist — order doesn't matter for TDD-failure purposes since both
land in the same task, but this keeps the module itself first.)

```python
"""Iterate WARC files and extract article text from their response records."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord


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
```

- [ ] **Step 2: Write the fixture-generation script**

```python
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

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

PAGES = [
    ("http://example.test/page1", "First Article", [
        "This is the first paragraph of the first test article.",
        "It has a second sentence to give trafilatura enough content.",
    ]),
    ("http://example.test/page2", "Second Article", [
        "This second article discusses a different synthetic topic entirely.",
        "Enough text is present here for reliable extraction as well.",
    ]),
    ("http://example.test/page3", "Third Article", [
        "The third synthetic article continues the pattern for testing purposes.",
    ]),
    ("http://example.test/page4", "Fourth Article", [
        "A fourth short article rounds out the fixture set nicely.",
        "Two sentences keep the extraction stable across all records.",
    ]),
    ("http://example.test/page5", "Fifth Article", [
        "The fifth and final synthetic article closes out the sample WARC file.",
    ]),
]


def _make_html(title: str, paragraphs: list[str]) -> bytes:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><article><h1>{title}</h1>{body}</article></body></html>"
    ).encode("utf-8")


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
```

- [ ] **Step 3: Run the script to generate the fixture**

Run: `uv run python scripts/build_sample_warc_fixture.py`
Expected: prints `wrote tests/fixtures/raw/sample.warc.gz (NNNN bytes)`.

- [ ] **Step 4: Write the failing tests (with golden text already known)**

```python
# tests/unit/test_acquire_warc.py
from __future__ import annotations

from pathlib import Path

from rinchen.acquire import warc

FIXTURE = Path("tests/fixtures/raw/sample.warc.gz")

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_warc.py -v`
Expected: both pass. (Verified live during planning with byte-identical
golden text to what's shown above — if your run produces different
extracted text, STOP and report the discrepancy rather than editing the
golden values to match, since that would mean either the fixture-
generation script or the installed `trafilatura` version behaves
differently than verified.)

- [ ] **Step 6: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire scripts/build_sample_warc_fixture.py tests/unit/test_acquire_warc.py`
Run: `uv run mypy src/rinchen/acquire scripts/build_sample_warc_fixture.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add src/rinchen/acquire/warc.py scripts/build_sample_warc_fixture.py tests/fixtures/raw/sample.warc.gz tests/unit/test_acquire_warc.py
git commit -m "D1: add WARC iteration + text extraction, with a real 5-record fixture"
```

---

### Task 5: `acquire/commoncrawl.py` — CDX query + bounded WARC range fetch

**Files:**
- Create: `src/rinchen/acquire/commoncrawl.py`
- Create: `tests/unit/test_acquire_commoncrawl.py`

**Interfaces:**
- Produces: `query_cdx(url_pattern, collection, *, client=None) ->
  list[dict]`, `fetch_records(records, dest_dir, *, client=None) ->
  list[Path]`.

This exact implementation was verified live during planning — transcribe
it exactly. Per the plan's Global Constraints and the design spec's
"Decisions resolved during design" section, this does NOT share code with
`http.py`'s resumable-download logic — it's a separate, simpler, bounded
single-range fetch.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_acquire_commoncrawl.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.acquire.commoncrawl'`.

- [ ] **Step 3: Write `src/rinchen/acquire/commoncrawl.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_commoncrawl.py -v`
Expected: both pass.

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire tests/unit/test_acquire_commoncrawl.py`
Run: `uv run mypy src/rinchen/acquire`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/rinchen/acquire/commoncrawl.py tests/unit/test_acquire_commoncrawl.py
git commit -m "D1: add CommonCrawl CDX query and bounded-range WARC fetch"
```

---

### Task 6: `acquire/hf.py` — HuggingFace streaming to Parquet shards

**Files:**
- Create: `src/rinchen/acquire/hf.py`
- Create: `tests/unit/test_acquire_hf.py`

**Interfaces:**
- Produces: `write_shards(rows, dest_dir, *, shard_size=10_000, prefix="shard")
  -> list[Path]` (unit-tested, no network), `fetch(dataset, dest_dir, *,
  config=None, split="train", shard_size=10_000) -> list[Path]` (the real
  `datasets.load_dataset(..., streaming=True)` wrapper — not unit-tested,
  per Global Constraints).

This exact implementation was verified live during planning — transcribe
it exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_acquire_hf.py
from __future__ import annotations

import polars as pl

from rinchen.acquire import hf


def test_write_shards_splits_rows_into_shards_of_given_size(tmp_path):
    rows = [{"id": i, "text": f"row {i}"} for i in range(25)]

    paths = hf.write_shards(rows, tmp_path, shard_size=10)

    assert len(paths) == 3
    counts = [pl.read_parquet(p).height for p in paths]
    assert counts == [10, 10, 5]


def test_write_shards_preserves_row_content(tmp_path):
    rows = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

    paths = hf.write_shards(rows, tmp_path, shard_size=10)

    assert len(paths) == 1
    frame = pl.read_parquet(paths[0])
    assert frame.to_dicts() == rows


def test_write_shards_returns_empty_list_for_no_rows(tmp_path):
    paths = hf.write_shards([], tmp_path)

    assert paths == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_acquire_hf.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.acquire.hf'`.

- [ ] **Step 3: Write `src/rinchen/acquire/hf.py`**

```python
"""HuggingFace `datasets` streaming -> Parquet shards.

`write_shards` takes an already-constructed row iterator so it is
unit-testable without any network access, and is exercised directly by
`tests/unit/test_acquire_hf.py`. `fetch` is the thin wrapper that actually
calls `datasets.load_dataset(..., streaming=True)`; it has no test
coverage at all (the master plan's own D1 test list names no such test,
and a real network-backed test would violate the "never network-access in
unit tests" hard constraint). It is NOT reachable via the tiny-profile
`--profile tiny` fixture path either: every `sources.tiny.yaml` entry uses
`kind: local`, which `registry.fetch_source` dispatches to
`http.download`, never to `hf.fetch`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl


def write_shards(
    rows: Iterable[Mapping[str, Any]],
    dest_dir: Path,
    *,
    shard_size: int = 10_000,
    prefix: str = "shard",
) -> list[Path]:
    """Write `rows` to `dest_dir` as Parquet shards of at most `shard_size`
    rows each. Returns the list of shard paths written, in order."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buffer: list[Mapping[str, Any]] = []
    shard_index = 0

    def _flush() -> None:
        nonlocal shard_index
        if not buffer:
            return
        frame = pl.DataFrame(buffer)
        path = dest_dir / f"{prefix}-{shard_index:05d}.parquet"
        frame.write_parquet(path)
        paths.append(path)
        shard_index += 1
        buffer.clear()

    for row in rows:
        buffer.append(dict(row))
        if len(buffer) >= shard_size:
            _flush()
    _flush()

    return paths


def fetch(
    dataset: str,
    dest_dir: Path,
    *,
    config: str | None = None,
    split: str = "train",
    shard_size: int = 10_000,
) -> list[Path]:
    """Stream `dataset` (optionally a `config`/`split`) from the Hub and
    write it to `dest_dir` as Parquet shards."""
    import datasets

    stream = datasets.load_dataset(dataset, config, split=split, streaming=True)
    return write_shards(stream, dest_dir, shard_size=shard_size)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_acquire_hf.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/acquire tests/unit/test_acquire_hf.py`
Run: `uv run mypy src/rinchen/acquire`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/rinchen/acquire/hf.py tests/unit/test_acquire_hf.py
git commit -m "D1: add HuggingFace streaming-to-Parquet-shards writer"
```

---

### Task 7: Registry dispatch, remaining fixtures, CLI, offline end-to-end test

**Files:**
- Modify: `src/rinchen/acquire/registry.py` (add `fetch_source`)
- Create: `tests/fixtures/raw/mn_news.html`
- Create: `tests/fixtures/raw/moses_mn_en.zip`
- Create: `configs/data/sources.tiny.yaml`
- Modify: `src/rinchen/cli/data.py` (currently an empty stub from D0)
- Create: `tests/unit/test_acquire_registry_fetch_source.py`
- Create: `tests/unit/test_cli_data.py`

**Interfaces:**
- Consumes: `http.download`, `opus.fetch`, `hf.fetch`,
  `commoncrawl.query_cdx`/`fetch_records` (Tasks 1, 3, 5, 6),
  `registry.load_registry`/`find_entry` (Task 2).
- Produces: `registry.fetch_source(entry, dest_dir) -> Path`. The CLI's
  `rinchen data fetch`/`ls` commands are the only consumers at D1.

- [ ] **Step 1: Write the two remaining fixture files**

`tests/fixtures/raw/mn_news.html` (a standalone synthetic HTML page,
simple fictional placeholder content, not claiming to be real news):

```html
<html>
<head><title>Fixture News Page</title></head>
<body>
<article>
<h1>Synthetic Test Headline</h1>
<p>This is a synthetic paragraph used only to test HTML text extraction.</p>
<p>It contains no real news content and no text from any reviewed corpus.</p>
</article>
</body>
</html>
```

`tests/fixtures/raw/moses_mn_en.zip`: a small zip in the real OPUS moses
format (two aligned `.txt` files, fictional content). Build it with:

```bash
mkdir -p /tmp/moses_fixture_build
cat > /tmp/moses_fixture_build/FixtureCorpus.en-mn.en << 'EOF'
This is a fixture sentence.
Here is a second fixture sentence.
A third synthetic sentence for testing.
EOF
cat > /tmp/moses_fixture_build/FixtureCorpus.en-mn.mn << 'EOF'
Энэ бол тестийн жишээ өгүүлбэр.
Энэ хоёр дахь тестийн жишээ өгүүлбэр.
Энэ бол тестийн гурав дахь зохиомол өгүүлбэр.
EOF
( cd /tmp/moses_fixture_build && zip -q /tmp/moses_fixture_build.zip FixtureCorpus.en-mn.en FixtureCorpus.en-mn.mn )
cp /tmp/moses_fixture_build.zip tests/fixtures/raw/moses_mn_en.zip
```

(If `/tmp` isn't writable in your environment, use any scratch directory
instead — the exact build location doesn't matter, only the final zip's
contents and its destination path in `tests/fixtures/raw/`.)

- [ ] **Step 2: Write `configs/data/sources.tiny.yaml`**

```yaml
# Offline, fixture-backed registry used only when `--profile tiny` is
# passed to `rinchen data fetch`/`ls`. Every entry uses `kind: local`, so
# fetching this registry makes zero network requests -- this is what
# proves D1's STOP gate: "rinchen data fetch --profile tiny --all works
# offline using local:// fixtures."
sources:
  - id: opus-api-sample
    kind: local
    langs: [mn, en]
    url: "local://tests/fixtures/raw/opus_api_mn_en.json"
  - id: warc-sample
    kind: local
    langs: [mn]
    url: "local://tests/fixtures/raw/sample.warc.gz"
  - id: news-html-sample
    kind: local
    langs: [mn]
    url: "local://tests/fixtures/raw/mn_news.html"
  - id: moses-zip-sample
    kind: local
    langs: [mn, en]
    url: "local://tests/fixtures/raw/moses_mn_en.zip"
```

- [ ] **Step 3: Write the failing test for `fetch_source`**

```python
# tests/unit/test_acquire_registry_fetch_source.py
from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_acquire_registry_fetch_source.py -v`
Expected: FAILS with `ImportError: cannot import name 'fetch_source'`.

- [ ] **Step 5: Add `fetch_source` to `src/rinchen/acquire/registry.py`**

Add this one new import at the top, alongside the existing ones (`Path`
is already imported in this file from Task 2 — do not duplicate it):

```python
from rinchen.acquire import commoncrawl, hf, http, opus
```

Then add this function at the end of the file:

```python
def fetch_source(entry: SourceEntry, dest_dir: Path) -> Path:
    """Dispatch `entry` to the right acquisition module based on its kind."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(entry, (HttpSourceEntry, LocalSourceEntry)):
        filename = entry.url.rstrip("/").split("/")[-1] or entry.id
        dest = dest_dir / filename
        return http.download(entry.url, dest, sha256=entry.expected_sha)

    if isinstance(entry, OpusSourceEntry):
        dest = dest_dir / f"{entry.id}.zip"
        return opus.fetch(
            entry.corpus,
            entry.src,
            entry.tgt,
            dest,
            preprocessing=entry.preprocessing,
            version=entry.version,
        )

    if isinstance(entry, HfSourceEntry):
        shard_dir = dest_dir / entry.id
        hf.fetch(entry.dataset, shard_dir, config=entry.config, split=entry.split)
        return shard_dir

    if isinstance(entry, CommonCrawlSourceEntry):
        shard_dir = dest_dir / entry.id
        records = commoncrawl.query_cdx(entry.url_pattern, entry.collection)
        commoncrawl.fetch_records(records, shard_dir)
        return shard_dir

    raise ConfigError(f"unknown source kind for entry {entry.id!r}")
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_acquire_registry_fetch_source.py -v`
Expected: passes.

- [ ] **Step 7: Write the CLI (`src/rinchen/cli/data.py`)**

Replace the file's stub contents entirely:

```python
"""`rinchen data`: fetch and list registry sources."""
from __future__ import annotations

from pathlib import Path

import typer

from rinchen.acquire.registry import fetch_source, find_entry, load_registry

app = typer.Typer(no_args_is_help=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCES_PATH = REPO_ROOT / "configs" / "data" / "sources.yaml"
SOURCES_TINY_PATH = REPO_ROOT / "configs" / "data" / "sources.tiny.yaml"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"


def _registry_path(profile: str | None) -> Path:
    return SOURCES_TINY_PATH if profile == "tiny" else SOURCES_PATH


@app.command()
def fetch(
    source: str | None = typer.Option(
        None, "--source", help="Fetch a single source by id."
    ),
    all_: bool = typer.Option(False, "--all", help="Fetch every registered source."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Use this profile's registry (e.g. 'tiny' for offline fixtures).",
    ),
) -> None:
    if not source and not all_:
        typer.echo("specify --source <id> or --all", err=True)
        raise typer.Exit(code=2)

    entries = load_registry(_registry_path(profile))
    targets = entries if all_ else [find_entry(entries, source)]

    for entry in targets:
        dest = fetch_source(entry, DATA_RAW_DIR / entry.id)
        typer.echo(f"fetched {entry.id} -> {dest}")


@app.command(name="ls")
def list_sources(
    profile: str | None = typer.Option(
        None, "--profile", help="Use this profile's registry."
    ),
) -> None:
    entries = load_registry(_registry_path(profile))
    for entry in entries:
        typer.echo(f"{entry.id}\t{entry.kind.value}\t{','.join(entry.langs)}")
```

- [ ] **Step 8: Write the end-to-end offline test (`tests/unit/test_cli_data.py`)**

```python
from __future__ import annotations

from typer.testing import CliRunner

from rinchen.cli import data as data_cli
from rinchen.cli.main import app

runner = CliRunner()


def test_data_fetch_tiny_profile_works_fully_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cli, "DATA_RAW_DIR", tmp_path)

    result = runner.invoke(app, ["data", "fetch", "--profile", "tiny", "--all"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "opus-api-sample" / "opus_api_mn_en.json").is_file()
    assert (tmp_path / "warc-sample" / "sample.warc.gz").is_file()
    assert (tmp_path / "news-html-sample" / "mn_news.html").is_file()
    assert (tmp_path / "moses-zip-sample" / "moses_mn_en.zip").is_file()


def test_data_ls_tiny_profile_lists_all_four_fixtures():
    result = runner.invoke(app, ["data", "ls", "--profile", "tiny"])

    assert result.exit_code == 0, result.output
    assert "opus-api-sample" in result.output
    assert "warc-sample" in result.output
    assert "news-html-sample" in result.output
    assert "moses-zip-sample" in result.output


def test_data_fetch_requires_source_or_all():
    result = runner.invoke(app, ["data", "fetch", "--profile", "tiny"])

    assert result.exit_code == 2
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_data.py -v`
Expected: all 3 pass. This is D1's actual STOP-gate proof: `rinchen data
fetch --profile tiny --all` works end-to-end with zero network calls.

- [ ] **Step 10: Run the full D1 test suite together**

Run: `uv run pytest tests/unit/test_acquire_*.py tests/unit/test_cli_data.py -v`
Expected: every test from Tasks 1-7 passes together (confirms no
cross-task interference).

- [ ] **Step 11: Run ruff and mypy**

Run: `uv run ruff check src/rinchen tests/unit`
Run: `uv run mypy src/rinchen`
Expected: both clean.

- [ ] **Step 12: Commit**

```bash
git add src/rinchen/acquire/registry.py src/rinchen/cli/data.py \
  tests/fixtures/raw/mn_news.html tests/fixtures/raw/moses_mn_en.zip \
  configs/data/sources.tiny.yaml \
  tests/unit/test_acquire_registry_fetch_source.py tests/unit/test_cli_data.py
git commit -m "D1: wire registry dispatch + CLI, prove offline tiny-profile fetch works end-to-end"
```

---

### Task 8: Runbook, docs updates, and final STOP-gate verification

**Files:**
- Create: `docs/runbooks/01_acquisition.md`
- Modify: `STATE.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7 — this task is the final
  acceptance check for the whole plan.

- [ ] **Step 1: Write `docs/runbooks/01_acquisition.md`**

```markdown
# Runbook: 01 — Data Acquisition

## Overview

`rinchen data fetch` pulls raw data into `data/raw/<source-id>/` per the
registry at `configs/data/sources.yaml` (real sources) or
`configs/data/sources.tiny.yaml` (offline fixtures, used by
`--profile tiny`). `rinchen data ls` lists registry entries without
fetching anything.

## Disk estimates

Real production sources are not yet populated in `sources.yaml` (see
`DECISIONS.md`'s 2026-10-30 entry — Rinchen is building its own standard
dataset rather than seeding from the 7 candidates reviewed during D1).
Once real sources are added, estimate disk usage per source from its
registry entry before fetching `--all` on a constrained machine; OPUS
corpora report their `size` (segment count) via the API, HuggingFace
datasets report their size on the dataset card, and CommonCrawl WARC
ranges are bounded by the `length` field in each CDX record queried.

## Parallelism

`rinchen data fetch --all` currently fetches sources sequentially. This
is intentional at D1: the registry and CLI are the proven mechanism;
parallelizing fetches (e.g. via a thread pool, respecting each remote's
rate limits) is a reasonable follow-up once there are enough real sources
for it to matter, not before.

## Rerun semantics

`acquire/http.py`'s `download()` resumes a truncated transfer via its
`.part` file and verifies `sha256` if the registry entry sets
`expected_sha`. It does not currently skip a fetch whose destination
already exists and is already complete (no hash-based rerun-skip wired to
`rinchen.manifest`'s stage runner yet) — that integration is part of the
still-deferred config→manifest seam noted in `DECISIONS.md`'s 2026-10-29
D0 wrap-up entry. For now, re-running `rinchen data fetch` for an
already-fetched source re-downloads it.

## How to add a new source

1. Vet the candidate with the same rigor used during D1's own design:
   download at least 100 random sentence pairs (for parallel data) or
   100 random documents (for monolingual data) and review every one for
   script correctness, translation fidelity (if parallel), and content
   cleanliness. Any issue found means the candidate is rejected — see
   `DECISIONS.md`'s 2026-10-30 entry for the bar this project holds to.
2. Add an entry to `configs/data/sources.yaml` matching one of the
   `SourceEntry` variants in `src/rinchen/acquire/registry.py` (`http`,
   `opus`, `hf`, `commoncrawl`).
3. Confirm it: `uv run rinchen data ls` should list it; `uv run rinchen
   data fetch --source <id>` should fetch it for real.
4. Record the vetting result (pass/fail, with the specific evidence) in
   `DECISIONS.md`, whether the source is accepted or rejected.
```

- [ ] **Step 2: Update `STATE.md`**

Read the current file first. Replace the "Current next step" line and
add a new paragraph documenting D1's completion, following the same
honest, hedged convention D0's equivalent entry used (distinguish "done
and verified locally" from "confirmed on CI" if CI still hasn't been
observed running — check the current file's wording on that point before
writing this, since it may have changed since D0 if the user pushed in
the meantime).

- [ ] **Step 3: Update `CLAUDE.md`**

Add `src/rinchen/acquire/`, `configs/data/`, and `docs/runbooks/01_acquisition.md`
to the repo layout block if not already implied by existing lines. Keep
the "Current phase" section to 1-2 sentences per its existing convention.

- [ ] **Step 4: Run the full real STOP-gate check**

Run: `just gate`
Expected: exits 0.

Run: `uv run rinchen data fetch --profile tiny --all`
Expected: exits 0, fetches all 4 fixtures into `data/raw/` for real (this
is the actual manual proof, beyond the automated test in Task 7 — run it
for real once here).

Run: `uv run rinchen data ls --profile tiny`
Expected: exits 0, lists all 4 fixture entries.

Run: `uv run pytest -q` (the full suite)
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add docs/runbooks/01_acquisition.md STATE.md CLAUDE.md
git commit -m "D1: add acquisition runbook, update STATE.md/CLAUDE.md for D1 completion"
```
