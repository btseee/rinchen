"""Configuration of the tiny end-to-end pipeline (`configs/tiny/<profile>.yaml`, D15 spec P5 to
P8, P12, P13 and P14): the fixtures and their pinned digests, the configuration of every stage, the
schedule of the model, the thresholds, the wall-time budget and a timeout for each stage, as data.

Scaling up is a change of this file and no change of code (P14), so everything that differs between
a tiny and a larger run is here, and nothing in the pipeline looks at the name of a profile. The
document is closed: an unknown key is an error (so is a key that names a device, because no section
has one), every key is required unless a default is named (only the thresholds have defaults, the
bounds of the master plan, and no profile may hold a looser one: a miss is answered with more
`train.steps`), a number is a number and not a boolean or a quoted text, and what a
stage's own configuration would refuse is refused here, at load time, so that a bad profile fails
before anything is read or trained. A path is not blank, has only printable characters and holds no
`..` component; a relative one is anchored at the repository root and an absolute one is kept (the
control profiles of the gate point at copies in a temporary directory). The quantizations are the
two the CPU runs (`float32` and `int8`) and the tag is placed in the source. Every number is an
unvalidated prior from one machine's measurements.

The document is read from the bytes the caller hands in, so that the caller can hash the file before
it is read and parse the bytes it hashed; nothing is read from the environment. An error names the
fields at fault and quotes no value of the file: a key is named only when it is a short identifier.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, get_args

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rinchen.config.ablation import MAX_BT_RATIO
from rinchen.config.loader import REPO_ROOT, anchor_at_repo_root, reject_blank_path
from rinchen.errors import ConfigError
from rinchen.tiny.fixtures import FIXTURE_FILES

StageId = Literal[
    "environment",
    "fetch",
    "corpus",
    "filter",
    "spm",
    "converter",
    "synth",
    "prepare",
    "train",
    "average",
    "export",
    "parity",
    "serve",
    "robustness",
    "evaluate",
]
STAGE_IDS: Final[tuple[StageId, ...]] = get_args(StageId)
# The splits of a suite file (`rinchen.eval.suite_rules.SPLITS`; a test keeps the two equal).
Split = Literal["dev", "devtest", "test"]
Quantization = Literal["float32", "int8"]

# The bounds of the master plan's thresholds (milestone D15): "never lower a bound, raise the steps".
# A profile may hold a stricter bound and no looser one; a miss is answered with more `train.steps`.
TRAIN_CHRF_FLOOR: Final = 50.0
PARITY_EXACT_FLOOR: Final = 0.95
PARITY_DELTA_CEILING: Final = 0.001
INT8_CHRF_DROP_CEILING: Final = 1.0
SCRIPT_PURITY_FLOOR: Final = 1.0
PLACEHOLDER_ACC_FLOOR: Final = 1.0
_FLOORS: Final = (
    ("train_chrf_min", TRAIN_CHRF_FLOOR),
    ("parity_exact_min", PARITY_EXACT_FLOOR),
    ("script_purity_min", SCRIPT_PURITY_FLOOR),
    ("placeholder_acc_min", PLACEHOLDER_ACC_FLOOR),
)
_CEILINGS: Final = (
    ("parity_delta_max", PARITY_DELTA_CEILING),
    ("int8_chrf_drop_max", INT8_CHRF_DROP_CEILING),
)
MAX_SECONDS: Final = 86_400
MAX_THREADS: Final = 64
# The fewest resamples of the report's bootstrap (`rinchen.eval.bootstrap.MIN_RESAMPLES`).
MIN_RESAMPLES: Final = 100
_PROFILE: Final = re.compile(r"[a-z][a-z0-9_]{0,15}")
_CREATED: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_CREATED_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
# A key of the file is named in a message only when it looks like a name.
_NAMEABLE_KEY: Final = re.compile(r"[A-Za-z0-9_.-]{1,40}")
_KEY_MARK: Final = "[key]"


def _path_text(value: object, info: ValidationInfo) -> object:
    reject_blank_path(value, info)
    if isinstance(value, str):
        if not value.isprintable():
            raise ValueError("must hold only printable characters")
        if ".." in Path(value).parts:
            raise ValueError("must hold no .. component")
    return value


def _fixture_name(name: str) -> str:
    if name not in FIXTURE_FILES:
        raise ValueError("is not the name of a fixture file")
    return name


def _quoted_time(value: object) -> object:
    if isinstance(value, date):
        # Pydantic reports a ValueError as a validation error and a TypeError as a crash.
        raise ValueError(  # noqa: TRY004
            "must be quoted text: YAML reads an unquoted time as a date"
        )
    return value


def _utc_time(value: str) -> str:
    if _CREATED.fullmatch(value) is None:
        raise ValueError("must be a UTC time written YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, _CREATED_FORMAT)
    except ValueError as error:
        raise ValueError("must be a real date and time") from error
    return value


# Strict, so that a quoted number or a YAML boolean is a typo to report and not a value to coerce
# (an integer where a float is wanted is read as that float).
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_Seconds = Annotated[int, Field(strict=True, ge=1, le=MAX_SECONDS)]
_PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
_NonNegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_Probability = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
_Fraction = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
_Cosine = Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)]
_ChrF = Annotated[float, Field(strict=True, gt=0, le=100, allow_inf_nan=False)]
_ChrFDrop = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
_BtRatio = Annotated[
    float, Field(strict=True, ge=0, le=MAX_BT_RATIO, allow_inf_nan=False)
]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_FixtureName = Annotated[str, AfterValidator(_fixture_name)]
_Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
# Relative paths mean the same thing from any working directory.
_Path = Annotated[
    Path, BeforeValidator(_path_text), AfterValidator(anchor_at_repo_root)
]
_UtcTime = Annotated[
    str, Field(strict=True), BeforeValidator(_quoted_time), AfterValidator(_utc_time)
]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FixturesConfig(_Closed):
    """The directory of the seven fixture files and the SHA-256 of each, which must be exactly the
    seven names of `rinchen.tiny.fixtures.FIXTURE_FILES`."""

    directory: _Path
    sha256: dict[_FixtureName, _Digest]

    @field_validator("sha256")
    @classmethod
    def _one_digest_per_fixture(cls, digests: dict[str, str]) -> dict[str, str]:
        if missing := [name for name in FIXTURE_FILES if name not in digests]:
            raise ValueError(f"no digest for: {', '.join(missing)}")
        return {name: digests[name] for name in FIXTURE_FILES}


class SpmSection(_Closed):
    """The knobs of `rinchen spm train` that differ from the shipped configurations."""

    vocab_size: _PositiveInt
    input_sentence_size: _PositiveInt
    character_coverage: _Probability
    temperature: _PositiveNumber


class FilterSection(_Closed):
    """The filter's bounds: the length ratio, the near-duplicate threshold, the two stand-in
    scorers' bounds and the share of authentic rows they may flag."""

    char_ratio_min: _PositiveNumber
    char_ratio_max: _PositiveNumber
    minhash_threshold: _Probability
    labse_min_cosine: _Cosine
    qe_max_metricx: _NonNegativeNumber
    max_authentic_flagged_share: _Fraction

    @model_validator(mode="after")
    def _the_ratio_bounds_are_in_order(self) -> FilterSection:
        if not self.char_ratio_min < self.char_ratio_max:
            raise ValueError("char_ratio_min must be less than char_ratio_max")
        return self


class ConverterSection(_Closed):
    """The script converter's word list and the size of its character vocabulary."""

    pairs: _Path
    char_vocab_size: _PositiveInt


class SynthSection(_Closed):
    """The synthetic-data job, the size of its shards and the candidates asked per source."""

    job: _Identifier
    shard_size: _PositiveInt
    n: _PositiveInt


class TrainSection(_Closed):
    """The Eole base config, the steps and the corpus mixing of the one variant that is trained,
    and how many of the last checkpoints are averaged."""

    base: _Path
    steps: _PositiveInt
    average_last: _PositiveInt
    temperature: _PositiveNumber
    bt_ratio: _BtRatio
    tag_placement: Literal["src"]
    dev_max_lines: _PositiveInt


class ExportSection(_Closed):
    """The quantizations of the CTranslate2 export: the ones a CPU runs, each once."""

    quantizations: Annotated[tuple[Quantization, ...], Field(min_length=1)]

    @field_validator("quantizations")
    @classmethod
    def _each_once(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("a quantization is listed twice")
        return values


class ParitySection(_Closed):
    """How many lines `rinchen export parity` compares."""

    max_lines: _PositiveInt


class DecodeSection(_Closed):
    """The beam of every thresholded decode (greedy: the decode the parity command compares)."""

    beam_size: _PositiveInt


class ServeSection(_Closed):
    """The service configuration the in-process check starts."""

    config: _Path


class EvaluationSection(_Closed):
    """The split of the suites of the memorised rows and of the held-out ones (a fixture suite
    needs one of the three), and the bootstrap's resamples."""

    memorised_split: Split
    heldout_split: Split
    resamples: Annotated[int, Field(strict=True, ge=MIN_RESAMPLES)]

    @model_validator(mode="after")
    def _the_splits_differ_and_the_held_out_one_is_reported(self) -> EvaluationSection:
        if self.memorised_split == self.heldout_split:
            raise ValueError("the memorised and the held-out split must differ")
        if self.heldout_split == "dev":
            raise ValueError("the held-out split is reported, and a dev run is not")
        return self


class ThresholdsSection(_Closed):
    """The bounds of the report. The defaults are the master plan's, and a profile never lowers one
    (`TinyConfig` refuses it: a section built directly, as the tests of the thresholds do, may hold
    any bound); the wall-time bound is the budget of the profile."""

    train_chrf_min: _ChrF = TRAIN_CHRF_FLOOR
    parity_exact_min: _Probability = PARITY_EXACT_FLOOR
    parity_delta_max: _Probability = PARITY_DELTA_CEILING
    int8_chrf_drop_max: _ChrFDrop = INT8_CHRF_DROP_CEILING
    script_purity_min: _Probability = SCRIPT_PURITY_FLOOR
    placeholder_acc_min: _Probability = PLACEHOLDER_ACC_FLOOR


class TinyConfig(_Closed):
    """One profile of the pipeline. `created` is the UTC time stamped on every row of the corpus,
    so that two runs write equal files; `qe_fake` runs the planted stand-ins for LaBSE and the QE
    scorer."""

    fixtures: FixturesConfig
    registry: _Path
    spm: SpmSection
    filter: FilterSection
    converter: ConverterSection
    synth: SynthSection
    train: TrainSection
    export: ExportSection
    parity: ParitySection
    decode: DecodeSection
    serve: ServeSection
    evaluation: EvaluationSection
    thresholds: ThresholdsSection
    budget_seconds: _Seconds
    timeouts: dict[StageId, _Seconds]
    threads: Annotated[int, Field(strict=True, ge=1, le=MAX_THREADS)]
    qe_fake: Annotated[bool, Field(strict=True)]
    created: _UtcTime

    @field_validator("timeouts")
    @classmethod
    def _one_timeout_per_stage(cls, timeouts: dict[str, int]) -> dict[str, int]:
        if missing := [stage for stage in STAGE_IDS if stage not in timeouts]:
            raise ValueError(f"no timeout for: {', '.join(missing)}")
        return {stage: timeouts[stage] for stage in STAGE_IDS}

    @model_validator(mode="after")
    def _no_bound_of_the_master_plan_is_lowered(self) -> TinyConfig:
        lowered = [
            f"thresholds.{name} must be at least {floor}"
            for name, floor in _FLOORS
            if getattr(self.thresholds, name) < floor
        ]
        lowered += [
            f"thresholds.{name} must be at most {ceiling}"
            for name, ceiling in _CEILINGS
            if getattr(self.thresholds, name) > ceiling
        ]
        if lowered:
            raise ValueError(
                f"a profile never lowers a bound of the master plan ({'; '.join(lowered)})"
            )
        return self


def _where(location: tuple[int | str, ...]) -> str:
    parts = []
    for part in location:
        if part == _KEY_MARK:
            continue
        if isinstance(part, str) and _NAMEABLE_KEY.fullmatch(part) is None:
            part = "<key>"
        parts.append(str(part))
    return ".".join(parts)


def _describe(error: ValidationError) -> str:
    problems = []
    for item in error.errors(include_input=False, include_url=False):
        where = _where(item["loc"])
        problems.append(f"{where}: {item['msg']}" if where else item["msg"])
    return "; ".join(problems)


def load_tiny_config(data: bytes) -> TinyConfig:
    """The profile in the YAML `data`.

    Raises:
        ConfigError: `data` is not UTF-8 or not YAML, is not a mapping, or breaks the schema; the
            message names every field at fault and quotes none of the values.
    """
    try:
        raw: Any = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError("invalid tiny config: not UTF-8 text") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        line = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(
            f"invalid tiny config: the text is not valid YAML{line}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            "invalid tiny config: the top level must be a mapping of field names to values"
        )
    try:
        return TinyConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid tiny config: {_describe(error)}") from error


def config_path_for(profile: str) -> Path:
    """The file of the profile, `configs/tiny/<profile>.yaml` under the repository.

    The file is not looked at: the command that runs the profile refuses one that has none.

    Raises:
        ConfigError: `profile` is not a lower-case letter followed by up to 15 lower-case letters,
            digits or underscores; the message does not quote it.
    """
    if _PROFILE.fullmatch(profile) is None:
        raise ConfigError(
            "invalid profile name: a lower-case letter and up to 15 lower-case letters, "
            "digits or underscores"
        )
    return REPO_ROOT / "configs" / "tiny" / f"{profile}.yaml"
