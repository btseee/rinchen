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
