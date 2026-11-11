"""Configuration of the evaluation metrics (`configs/eval/metrics.yaml`, D13 T9 and T13): the
bootstrap settings, the metric a comparison is made on and the release targets, as data.

Every key is required: a missing `seed` is an error and never a seed nobody chose, and nothing
is read from the environment. Every number of the shipped table is an unvalidated prior.

A target is one of three kinds, and each kind has the fields it needs and no others:
`beats_baseline` (`min_directions` of `of_directions` directions where the named baseline system
is beaten; it has no `threshold`, the pair of counts is its threshold), `off_target` (a
`threshold` rate) and `robustness_value` (a `threshold` for the `key` of D12's robustness block).
What a target's status is, is decided where the report is built; this module only holds them.

The document is read from the bytes the caller hands in, so that the caller can hash the file
before it is read and parse the bytes it hashed.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rinchen.errors import ConfigError
from rinchen.metrics.registry import CountMetricName

TargetKind = Literal["beats_baseline", "off_target", "robustness_value"]

# The fields each kind of target holds (and must hold); every other field is absent.
_FIELDS_OF_KIND: Final[dict[str, frozenset[str]]] = {
    "beats_baseline": frozenset({"min_directions", "of_directions"}),
    "off_target": frozenset({"threshold"}),
    "robustness_value": frozenset({"threshold", "key"}),
}
_KIND_FIELDS: Final = ("threshold", "min_directions", "of_directions", "key")

_ID_PATTERN: Final = r"^[A-Za-z0-9._-]{1,64}$"
_KEY_PATTERN: Final = r"^[a-z][a-z0-9_]{0,63}$"


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _distinct(values: list[CountMetricName]) -> list[CountMetricName]:
    if len(set(values)) != len(values):
        raise ValueError("a metric is listed twice")
    return values


_Count = Annotated[int, Field(strict=True, ge=1)]
_Description = Annotated[str, AfterValidator(_not_blank)]


class BootstrapConfig(BaseModel):
    """The bootstrap: how many resamples, from which seed, at which level (`alpha` 0.05 is a
    95 % interval)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resamples: Annotated[int, Field(strict=True, ge=100)]
    seed: Annotated[int, Field(strict=True, ge=0)]
    alpha: Annotated[float, Field(strict=True, gt=0, lt=0.5)]


class TargetConfig(BaseModel):
    """One release target; see the module's text for the fields of each kind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=_ID_PATTERN)]
    description: _Description
    kind: TargetKind
    threshold: (
        Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None
    ) = None
    min_directions: _Count | None = None
    of_directions: _Count | None = None
    key: Annotated[str, Field(pattern=_KEY_PATTERN)] | None = None

    @model_validator(mode="after")
    def _holds_the_fields_of_its_kind(self) -> TargetConfig:
        needed = _FIELDS_OF_KIND[self.kind]
        given = frozenset(
            field for field in _KIND_FIELDS if getattr(self, field) is not None
        )
        if missing := sorted(needed - given):
            raise ValueError(f"a {self.kind} target needs {', '.join(missing)}")
        if foreign := sorted(given - needed):
            raise ValueError(f"a {self.kind} target has no {', '.join(foreign)}")
        if (
            self.min_directions is not None
            and self.of_directions is not None
            and self.min_directions > self.of_directions
        ):
            raise ValueError("min_directions is more than of_directions")
        return self


class EvalMetricsConfig(BaseModel):
    """The whole table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap: BootstrapConfig
    primary_metric: Literal["chrf_pp"]
    comparison_metrics: Annotated[
        list[CountMetricName], Field(min_length=1), AfterValidator(_distinct)
    ]
    baseline_system: Annotated[str, Field(pattern=_ID_PATTERN)] | None
    targets: list[TargetConfig]

    @model_validator(mode="after")
    def _the_primary_metric_is_compared(self) -> EvalMetricsConfig:
        if self.primary_metric not in self.comparison_metrics:
            raise ValueError(
                "comparison_metrics must include primary_metric "
                f"({self.primary_metric}): the release targets are judged on it"
            )
        return self

    @model_validator(mode="after")
    def _target_ids_are_distinct(self) -> EvalMetricsConfig:
        ids = [target.id for target in self.targets]
        if repeated := sorted({name for name in ids if ids.count(name) > 1}):
            raise ValueError(f"targets: an id is used twice: {', '.join(repeated)}")
        return self


def load_eval_metrics_config(data: bytes) -> EvalMetricsConfig:
    """The table in the YAML `data`.

    Raises:
        ConfigError: `data` is not UTF-8 or not YAML, is not a mapping, or breaks the schema;
            the message names every field at fault.
    """
    try:
        raw: Any = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError("invalid eval metrics config: not UTF-8 text") from error
    except yaml.YAMLError as error:
        raise ConfigError(
            f"invalid eval metrics config: cannot parse the YAML: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            "invalid eval metrics config: the top level must be a mapping of "
            f"field names to values, not {type(raw).__name__}"
        )
    try:
        return EvalMetricsConfig.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
        raise ConfigError(f"invalid eval metrics config: {problems}") from error
