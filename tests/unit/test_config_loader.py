"""Tests for rinchen.config.loader.load's base -> profile -> overlay -> CLI merge."""

import pytest

from rinchen.config import loader
from rinchen.errors import ConfigError


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_load_merges_base_and_profile(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\npaths:\n  data_root: /base\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "paths:\n  data_root: /tiny\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny")

    assert config.profile == "tiny"
    assert config.seed == 1  # from base, not overridden by profile
    assert config.paths["data_root"] == "/tiny"  # profile overrides base


def test_load_applies_overlay_file(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "seed: 2\n")
    overlay = tmp_path / "overlay.yaml"
    _write(overlay, "seed: 3\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", overlay)

    assert config.seed == 3


def test_cli_override_wins_over_everything(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "seed: 2\n")
    overlay = tmp_path / "overlay.yaml"
    _write(overlay, "seed: 3\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", overlay, cli_overrides={"seed": "4"})

    assert config.seed == 4


def test_cli_override_sets_nested_path(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\npaths:\n  data_root: /base\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "{}\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", cli_overrides={"paths.data_root": "/overridden"})

    assert config.paths["data_root"] == "/overridden"


def test_load_raises_config_error_for_missing_profile(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    with pytest.raises(ConfigError):
        loader.load("nonexistent")


def test_env_var_overrides_seed(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "{}\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.setenv("RINCHEN_SEED", "99")

    config = loader.load("tiny")

    assert config.seed == 99
