"""Source registry: validates `configs/data/sources.yaml`-shaped files
against a `SourceEntry` discriminated union and loads them for fetching."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from rinchen.acquire import commoncrawl, hf, http, opus
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

    @model_validator(mode="after")
    def _reject_expected_sha(self) -> HfSourceEntry:
        if self.expected_sha is not None:
            raise ValueError(
                "expected_sha is not supported for kind: hf -- an hf fetch "
                "writes multiple Parquet shards, so a single sha256 has no "
                "well-defined meaning here"
            )
        return self


class CommonCrawlSourceEntry(_BaseSourceEntry):
    kind: Literal[SourceKind.COMMONCRAWL]
    url_pattern: str
    collection: str

    @model_validator(mode="after")
    def _reject_expected_sha(self) -> CommonCrawlSourceEntry:
        if self.expected_sha is not None:
            raise ValueError(
                "expected_sha is not supported for kind: commoncrawl -- a "
                "commoncrawl fetch writes multiple WARC records, so a "
                "single sha256 has no well-defined meaning here"
            )
        return self


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
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid registry {path}: top level must be a mapping with a "
            f"'sources' key, got {type(raw).__name__}"
        )
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


def fetch_source(entry: SourceEntry, dest_dir: Path) -> Path:
    """Dispatch `entry` to the right acquisition module based on its kind."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(entry, (HttpSourceEntry, LocalSourceEntry)):
        filename = entry.url.rstrip("/").split("/")[-1] or entry.id
        dest = dest_dir / filename
        return http.download(entry.url, dest, sha256=entry.expected_sha)

    if isinstance(entry, OpusSourceEntry):
        dest = dest_dir / f"{entry.id}.zip"
        result = opus.fetch(
            entry.corpus,
            entry.src,
            entry.tgt,
            dest,
            preprocessing=entry.preprocessing,
            version=entry.version,
        )
        http.verify_sha256(result, entry.expected_sha)
        return result

    if isinstance(entry, HfSourceEntry):
        # `dest_dir` is already `<data-root>/<entry.id>` (the CLI builds it
        # that way), so it is used directly here rather than nesting
        # `entry.id` a second time underneath it.
        hf.fetch(entry.dataset, dest_dir, config=entry.config, split=entry.split)
        return dest_dir

    if isinstance(entry, CommonCrawlSourceEntry):
        records = commoncrawl.query_cdx(entry.url_pattern, entry.collection)
        commoncrawl.fetch_records(records, dest_dir)
        return dest_dir

    raise ConfigError(f"unknown source kind for entry {entry.id!r}")
