"""Language registry: validates `configs/languages.yaml` and exposes the
language codes and direction tags the rest of D3 is keyed on."""

from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

LANGUAGES_PATH = REPO_ROOT / "configs" / "languages.yaml"


class LanguageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    scripts: list[Script] = Field(min_length=1)
    direction_tag: str

    @model_validator(mode="after")
    def _direction_tag_matches_code(self) -> LanguageEntry:
        expected = f"<2{self.code}>"
        if self.direction_tag != expected:
            raise ValueError(
                f"direction_tag for {self.code!r} must be {expected!r}, "
                f"got {self.direction_tag!r}"
            )
        return self


_LIST_ADAPTER = TypeAdapter(list[LanguageEntry])


def load_languages(path: Path = LANGUAGES_PATH) -> list[LanguageEntry]:
    """Load and validate a `languages.yaml`-shaped file. Raises ConfigError
    on any violation (missing, unreadable or unparseable file, malformed top
    level, missing/extra field, unknown or empty scripts, wrong direction
    tag, duplicate code)."""
    if not path.is_file():
        raise ConfigError(f"language registry file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid language registry {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid language registry {path}: top level must be a mapping "
            f"with a 'languages' key, got {type(raw).__name__}"
        )
    entries = raw.get("languages")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(
            f"invalid language registry {path}: 'languages' must be a non-empty list"
        )
    try:
        languages = _LIST_ADAPTER.validate_python(entries)
    except ValidationError as exc:
        raise ConfigError(f"invalid language registry {path}: {exc}") from exc
    duplicates = sorted(
        code for code, count in Counter(e.code for e in languages).items() if count > 1
    )
    if duplicates:
        raise ConfigError(
            f"invalid language registry {path}: duplicate language codes {duplicates}"
        )
    return languages


@functools.cache
def known_language_codes() -> frozenset[str]:
    """Codes of the default registry; cached, so tests needing a custom
    registry call `load_languages(path)` directly."""
    return frozenset(entry.code for entry in load_languages())


@functools.cache
def _direction_tags() -> dict[str, str]:
    return {entry.code: entry.direction_tag for entry in load_languages()}


def direction_tag(code: str) -> str:
    """The direction tag of the language `code` in the default registry, such as
    `<2khk_Cyrl>`; cached like `known_language_codes`.

    Raises:
        ConfigError: `code` is not in the registry.
    """
    try:
        return _direction_tags()[code]
    except KeyError:
        raise ConfigError(
            f"language {code!r} is not in the language registry"
        ) from None
