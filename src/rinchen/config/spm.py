"""SentencePiece training config: the knobs that differ per profile
(`configs/spm/*.yaml`); the fixed policy invariants live in `spm.train`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError

# Strict so a quoted number or a YAML boolean is a typo to report, not a
# value to coerce.
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class SpmConfig(BaseModel):
    """One SentencePiece training run. Every field is required; relative
    paths are anchored at the repo root so the config means the same thing
    from any working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vocab_size: _PositiveInt
    input_sentence_size: _PositiveInt
    character_coverage: float = Field(strict=True, gt=0, le=1, allow_inf_nan=False)
    temperature: float = Field(strict=True, gt=0, allow_inf_nan=False)
    seed: int = Field(strict=True)
    corpus_dir: Path  # holds <lang_code>.txt files
    output_prefix: Path  # the trained model is <output_prefix>.model

    @field_validator("corpus_dir", "output_prefix", mode="before")
    @classmethod
    def _reject_blank_path(cls, value: object, info: ValidationInfo) -> object:
        # Before validation: Path("") already collapses to ".", which anchoring
        # would turn into the repo root itself (as output_prefix, that puts the
        # model outside the repository).
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{info.field_name} must not be empty or blank")
        return value

    @field_validator("corpus_dir", "output_prefix")
    @classmethod
    def _anchor_at_repo_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else REPO_ROOT / value


def load_spm_config(path: Path) -> SpmConfig:
    """Load and validate one `configs/spm/*.yaml`-shaped file. Raises
    ConfigError on any violation (missing, unreadable or unparseable file,
    non-mapping top level, missing/extra field, out-of-range or wrongly typed
    value)."""
    if not path.is_file():
        raise ConfigError(f"SPM config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid SPM config {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid SPM config {path}: top level must be a mapping of "
            f"field names to values, got {type(raw).__name__}"
        )
    try:
        return SpmConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid SPM config {path}: {exc}") from exc
