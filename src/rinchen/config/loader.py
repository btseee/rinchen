"""Three-layer config loader: base -> profile -> overlay files -> CLI --set,
then RINCHEN_* env vars as the final overlay."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from rinchen.config.schema import RinchenConfig
from rinchen.config.settings import RinchenEnvSettings
from rinchen.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dotted_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        next_cursor = cursor.setdefault(part, {})
        if not isinstance(next_cursor, dict):
            raise ConfigError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = next_cursor
    cursor[parts[-1]] = value


def load(
    profile: str,
    *overlays: Path,
    cli_overrides: dict[str, str] | None = None,
) -> RinchenConfig:
    """Merge base -> profile -> overlay YAML files -> CLI --set overrides -> env vars."""
    merged: dict[str, Any] = {}
    if BASE_CONFIG_PATH.is_file():
        merged = _deep_merge(merged, _read_yaml(BASE_CONFIG_PATH))

    profile_path = PROFILES_DIR / f"{profile}.yaml"
    merged = _deep_merge(merged, _read_yaml(profile_path))
    merged["profile"] = profile

    for overlay_path in overlays:
        merged = _deep_merge(merged, _read_yaml(overlay_path))

    for dotted_key, raw_value in (cli_overrides or {}).items():
        _apply_dotted_override(merged, dotted_key, raw_value)

    env = RinchenEnvSettings()
    if env.seed is not None:
        merged["seed"] = env.seed

    try:
        return RinchenConfig(**merged)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"invalid resolved configuration: {exc}") from exc
