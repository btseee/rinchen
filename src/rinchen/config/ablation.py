"""Ablation matrix config (`configs/eole/ablation_*.yaml`): the axes an ablation
varies (temperature of the corpus mixing, ratio of back-translated data, placement of
the direction tag), the seeds, the Eole base config every variant starts from, and
the limits that keep a run bounded.

`axes` and `seeds` each name a list of values and a variant is one choice from each
(`rinchen.train.variants.expand_variants`). A field left out keeps its default. This
module checks the matrix and nothing that needs Eole: the base config and `overrides`
are validated when a variant's Eole config is rendered, so this config layer imports
no feature package."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rinchen.config.loader import anchor_at_repo_root, reject_blank_path
from rinchen.errors import ConfigError

# Where the direction tag goes: the source line or the target line. A source suffix
# is out: Eole's `suffix` transform raises KeyError at the start of training unless a
# workaround corpus is configured.
TagPlacement = Literal["src", "tgt"]

# Strict so a quoted number or a YAML boolean is a typo to report, not a value to
# coerce.
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_Temperature = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
_BtRatio = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


def id_number(value: float) -> str:
    """How a variant id prints a temperature or a ratio: `0.5`, `1` for 1.0, `0.25`."""
    return f"{value:g}"


def _printed(value: object) -> str:
    return id_number(value) if isinstance(value, float) else str(value)


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _reject_suffix(value: object) -> object:
    if value == "suffix":
        raise ValueError(
            "suffix placement is not supported: Eole's suffix transform raises a "
            "KeyError when training starts unless a workaround corpus is configured; "
            "use 'src' or 'tgt'"
        )
    return value


def _reject_duplicates[T](values: list[T]) -> list[T]:
    # Compared as printed, because the printed values make the variant ids: values
    # that differ but print alike (0.1234567 and 0.1234568 as 0.123457) would name two
    # variants with one id.
    printed = [_printed(value) for value in values]
    repeated = sorted({text for text in printed if printed.count(text) > 1})
    if repeated:
        raise ValueError(
            f"duplicate values {repeated}: a variant id prints each value, so two "
            "values that print alike would name two variants alike"
        )
    return values


def _must_be_a_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"base config file not found: {path}")
    return path


_NonBlankString = Annotated[str, AfterValidator(_reject_blank)]
# Relative paths mean the same thing from any working directory.
_BaseFile = Annotated[
    Path,
    BeforeValidator(reject_blank_path),
    AfterValidator(anchor_at_repo_root),
    AfterValidator(_must_be_a_file),
]
_Placement = Annotated[TagPlacement, BeforeValidator(_reject_suffix)]
_Temperatures = Annotated[
    list[_Temperature], Field(min_length=1), AfterValidator(_reject_duplicates)
]
_BtRatios = Annotated[
    list[_BtRatio], Field(min_length=1), AfterValidator(_reject_duplicates)
]
_Placements = Annotated[
    list[_Placement], Field(min_length=1), AfterValidator(_reject_duplicates)
]
_Seeds = Annotated[
    list[_NonNegativeInt], Field(min_length=1), AfterValidator(_reject_duplicates)
]


def _default_placements() -> list[TagPlacement]:
    return ["src"]


class Axes(BaseModel):
    """What varies between variants; each list holds distinct values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Corpus mixing: size ** temperature over the directions; below 1 favours the
    # small ones, 1 is proportional.
    temperature: _Temperatures = Field(default_factory=lambda: [1.0])
    # Back-translated rows per authentic row of a direction; 0 uses none.
    bt_ratio: _BtRatios = Field(default_factory=lambda: [0.0])
    tag_placement: _Placements = Field(default_factory=_default_placements)


class AblationMatrix(BaseModel):
    """The matrix: `name` labels the run, `base` is the Eole config every variant
    starts from (an existing file), `overrides` is merged into every variant's Eole
    config, `weight_total` is the sum of the corpus weights of a variant,
    `dev_max_lines` caps the development lines per direction, and `max_variants`
    refuses a matrix that expands to more variants than that."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: _NonBlankString
    base: _BaseFile
    axes: Axes = Field(default_factory=Axes)
    seeds: _Seeds = Field(default_factory=lambda: [1234])
    weight_total: _PositiveInt = 100
    dev_max_lines: _PositiveInt = 1000
    max_variants: _PositiveInt = 64
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _expansion_fits_max_variants(self) -> AblationMatrix:
        count = (
            len(self.axes.temperature)
            * len(self.axes.bt_ratio)
            * len(self.axes.tag_placement)
            * len(self.seeds)
        )
        if count > self.max_variants:
            raise ValueError(
                f"the axes and seeds expand to {count} variants, more than "
                f"max_variants ({self.max_variants})"
            )
        return self


def load_ablation(path: Path) -> AblationMatrix:
    """Load and validate an ablation matrix file. Raises ConfigError on any
    violation (missing, unreadable or unparseable file, non-mapping top level,
    unknown field, out-of-range, duplicate or wrongly typed value, a tag placement
    that is not supported, a base file that does not exist, an expansion above
    `max_variants`)."""
    if not path.is_file():
        raise ConfigError(f"ablation config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid ablation config {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid ablation config {path}: top level must be a mapping of "
            f"field names to values, got {type(raw).__name__}"
        )
    try:
        return AblationMatrix.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid ablation config {path}: {exc}") from exc
