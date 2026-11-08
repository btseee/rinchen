"""The parts of an Eole config that every Rinchen model shares: the keys no config
may carry, the guard against `$`, and reading and writing the YAML.

What a particular model's config must hold (the converter's tags and corpora, say)
belongs to that model's own module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

from rinchen.errors import ConfigError
from rinchen.utils.part_file import unique_part_path

# Eole's own default, spelled out in every config so none depends on it.
POSITION_ENCODING: Final = "SinusoidalInterleaved"

# Field names of the installed Eole (0.6.4, `eole/config/models.py`) that select or
# tune a position scheme other than sinusoidal, or restrict or group attention.
# Eole has no field named for ALiBi (it is a `position_encoding_type` value, rejected
# by the validators). A key that is present is rejected even with a neutral value, so
# a config never holds a switch that could be flipped.
FORBIDDEN_KEYS: Final = frozenset(
    {
        # Rotary: `rope_config` holds every other rotary setting.
        "rope_config",
        "rotary_interleave",
        "rotary_theta",
        "rotary_theta_local",
        "rotary_dim",
        "rotary_dim_global",
        "partial_rotary_factor",
        "interleave_local",
        "multidimensional_rope",
        "xdrope_section",
        "qk_norm_post_rope",
        # Relative position bias.
        "relative_positions_buckets",
        # Sliding window and per-layer attention types.
        "sliding_window",
        "layer_types",
        # Grouped-query attention.
        "heads_kv",
        "global_heads_kv",
    }
)


def reject_dollar(what: str, value: Path | str) -> None:
    """Raise ConfigError, naming `what`, when `value` holds `$`."""
    # Eole runs `os.path.expandvars` over the text of a config file, so a `$NAME`
    # in a path would be silently replaced by an environment variable's value.
    if "$" in str(value):
        raise ConfigError(
            f"the {what} holds '$', which Eole would expand as a variable"
        )


def read_config(path: Path, what: str) -> Any:
    """What the YAML file at `path` holds, which the caller's validator then checks
    is a mapping of the right shape; `what` names the kind of config in messages.

    Raises:
        ConfigError: the file is missing, cannot be read or parsed, or its text holds
            `$`, which Eole would expand as a variable.
    """
    if not path.is_file():
        raise ConfigError(f"{what} file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        reject_dollar(f"text of the config {path}", text)
        return yaml.safe_load(text)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid {what} {path}: cannot read or parse: {exc}"
        ) from exc


def dump_config(cfg: dict[str, Any], path: Path) -> None:
    """Write `cfg` as YAML (UTF-8, LF, keys in their dict order) so the file appears
    whole: a failure leaves the old file, if any, and no temporary file. The
    directory of `path` must exist."""
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    part_path = unique_part_path(path)
    try:
        with part_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)
