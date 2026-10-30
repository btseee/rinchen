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
