"""Filter pipeline config (`configs/data/filters.yaml`): one section per stage
plus the order the stages run in. Every number is an unvalidated prior until
real data exists; a missing section or field keeps its default."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, get_args

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rinchen.config.loader import REPO_ROOT, anchor_at_repo_root, reject_blank_path
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

FILTERS_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "filters.yaml"

# Strict so a quoted number or a YAML boolean is a typo to report, not a
# value to coerce.
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_Int = Annotated[int, Field(strict=True)]
_PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
_NonNegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_Probability = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
_Cosine = Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)]

_DIRECTION_KEY = re.compile(r"[a-z]{2,3}_[A-Za-z]+->[a-z]{2,3}_[A-Za-z]+")


# Relative paths mean the same thing from any working directory.
_RepoPath = Annotated[
    Path, BeforeValidator(reject_blank_path), AfterValidator(anchor_at_repo_root)
]

StageName = Literal[
    "script_ratio", "numbers", "length", "lid", "minhash", "labse", "qe"
]


class Bounds(BaseModel):
    """An accepted range of a ratio; both ends are inclusive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: _PositiveNumber
    max: _PositiveNumber

    @model_validator(mode="after")
    def _min_is_below_max(self) -> Bounds:
        if not self.min < self.max:
            raise ValueError(
                f"min must be less than max, got min={self.min} and max={self.max}"
            )
        return self


class ScriptRatioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_ratio: _Probability = 0.9


class DirectionBounds(BaseModel):
    """Replacement bounds for one translation direction; a side left unset
    keeps the stage-wide bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    char_ratio: Bounds | None = None
    token_ratio: Bounds | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> DirectionBounds:
        if self.char_ratio is None and self.token_ratio is None:
            raise ValueError("at least one of char_ratio, token_ratio must be set")
        return self


class LengthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    char_ratio: Bounds = Bounds(min=0.5, max=2.0)
    token_ratio: Bounds = Bounds(min=0.4, max=2.5)
    # Weight per character of a script; every other character weighs 1.0.
    char_weights: dict[Script, _PositiveNumber] = {
        Script.HAN: 2.5,
        Script.HANGUL: 2.0,
        Script.HIRAGANA: 1.5,
        Script.KATAKANA: 1.5,
    }
    # Keyed "<src_lang>-><tgt_lang>", e.g. "khk_Cyrl->eng_Latn".
    by_direction: dict[str, DirectionBounds] = {}
    # The token-ratio check runs only when a SentencePiece model is set.
    spm_model: _RepoPath | None = None

    @field_validator("by_direction")
    @classmethod
    def _directions_are_well_formed(
        cls, value: dict[str, DirectionBounds]
    ) -> dict[str, DirectionBounds]:
        malformed = sorted(key for key in value if not _DIRECTION_KEY.fullmatch(key))
        if malformed:
            raise ValueError(
                "by_direction keys must look like 'khk_Cyrl->eng_Latn', "
                f"got {malformed}"
            )
        return value


class LidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: _RepoPath = Field(
        default=Path("models/lid/glotlid_v3.bin"), validate_default=True
    )
    min_prob: _Probability = 0.5
    top_k: _PositiveInt = 3
    # Language code -> labels counted as that language; a code that is not
    # listed accepts only its own label. GlotLID v3 calls Mandarin cmn_Hani
    # (and sometimes wuu_Hani), never zho_Hans. A table you supply replaces
    # this one, like char_weights: list zho_Hans again if you still need it.
    accept: dict[str, list[str]] = Field(
        default_factory=lambda: {"zho_Hans": ["cmn_Hani", "wuu_Hani"]}
    )


class MinHashConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_perm: _PositiveInt = 128
    ngram: _PositiveInt = 5
    threshold: _Probability = 0.8
    seed: _Int = 1


class LabseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(default="sentence-transformers/LaBSE", min_length=1)
    min_cosine: _Cosine = 0.7
    batch_size: _PositiveInt = 128
    device: Literal["auto", "cpu", "cuda"] = "auto"


class QeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scorer: Literal["metricx_qe", "cometkiwi"] = "metricx_qe"
    max_metricx_qe: _NonNegativeNumber = 8.0  # lower is better
    min_cometkiwi: _Number = 0.55  # higher is better
    batch_size: _PositiveInt = 32


class FilterConfig(BaseModel):
    """The whole pipeline: `stages` lists the stages to run, in order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: list[StageName] = Field(
        default_factory=lambda: list(get_args(StageName)), min_length=1
    )
    script_ratio: ScriptRatioConfig = ScriptRatioConfig()
    length: LengthConfig = LengthConfig()
    lid: LidConfig = LidConfig()
    minhash: MinHashConfig = MinHashConfig()
    labse: LabseConfig = LabseConfig()
    qe: QeConfig = QeConfig()

    @field_validator("stages")
    @classmethod
    def _stages_are_unique(cls, value: list[StageName]) -> list[StageName]:
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(f"duplicate stage names: {repeated}")
        return value


def load_filter_config(path: Path = FILTERS_CONFIG_PATH) -> FilterConfig:
    """Load and validate a `configs/data/filters.yaml`-shaped file. Raises
    ConfigError on any violation (missing, unreadable or unparseable file,
    non-mapping top level, unknown field, out-of-range or wrongly typed
    value)."""
    if not path.is_file():
        raise ConfigError(f"filter config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid filter config {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid filter config {path}: top level must be a mapping of "
            f"section names to values, got {type(raw).__name__}"
        )
    try:
        return FilterConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid filter config {path}: {exc}") from exc
