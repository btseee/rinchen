"""Configuration of the maintenance check (`configs/maintain/maintain.yaml`, D14 spec M6 to M9 and
M13): the uv argument lists and their bounds, the pin rules and the parity reruns, as data.

Every key is required unless a default is named, nothing is read from the environment, and the
document is closed (an unknown key is an error). The guards of M6 are enforced here, at load
time, and not where uv is run: no configuration reaches the runner with an argument list that can
write the lock file or upgrade one package (the first argument is `lock`, every other one is one
of four flags, none repeats, the drift list holds `--dry-run`, and the lock check holds
`--check` and no flag but `--offline`). A rerun of kind `pytest` never selects a test of the
maintenance check itself, so a rerun cannot rerun the gate of the command that makes it. Every
number is an unvalidated prior.

The document is read from the bytes the caller hands in, so that the caller can hash the file
before it is read and parse the bytes it hashed. An error names the field and never quotes the
value that was refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rinchen.errors import ConfigError
from rinchen.maintain.lock import normalize_name
from rinchen.maintain.states import SKIP_REASONS

CADENCES: Final = ("nightly", "weekly")
GATE: Final = "gate"
ALLOWED_UV_FLAGS: Final = frozenset({"--upgrade", "--dry-run", "--offline", "--check"})
LOCK_CHECK_FLAGS: Final = frozenset({"--check", "--offline"})

_RERUN_ID: Final = r"^[a-z][a-z0-9_]{0,31}$"
_RULE_ID: Final = r"^[a-z0-9-]{1,40}$"
_PACKAGE: Final = r"^(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?|[a-z0-9][a-z0-9.-]*\*)$"
_CHECK_ID: Final = r"^D-[A-Z0-9-]{1,30}$"
_FORBIDDEN_IN_TARGET: Final = "maintain"
_MAX_TARGETS: Final = 20


def _one_line(value: str) -> str:
    if value.splitlines() != [value]:
        raise ValueError("must be one line")
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _checked_target(path: str) -> str:
    parts = path.split("/")
    if "\\" in path or not path.isprintable():
        raise ValueError("must be a printable path with / as its separator")
    if parts[0] != "tests" or len(parts) < 2:
        raise ValueError("must be a path under tests/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("must hold no empty, . or .. component")
    if _FORBIDDEN_IN_TARGET in path.casefold():
        raise ValueError("must not name a test of the maintenance check itself")
    return path


def _closed_skip_reason(reason: str) -> str:
    if reason not in SKIP_REASONS:
        raise ValueError("must be one of the closed skip reasons of the states module")
    return reason


_Line = Annotated[str, Field(min_length=1, max_length=300), AfterValidator(_one_line)]
_Seconds = Annotated[int, Field(strict=True, ge=1, le=14_400)]
_Target = Annotated[str, AfterValidator(_checked_target)]


class UvCommand(BaseModel):
    """A uv invocation: `uv <uv_args>` with a bound in seconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uv_args: Annotated[list[str], Field(min_length=1)]
    timeout_seconds: _Seconds

    @field_validator("uv_args")
    @classmethod
    def _only_the_closed_flags(cls, args: list[str]) -> list[str]:
        if args[0] != "lock":
            raise ValueError("must start with the argument lock")
        flags = args[1:]
        for position, flag in enumerate(flags, start=1):
            if flag not in ALLOWED_UV_FLAGS:
                allowed = ", ".join(sorted(ALLOWED_UV_FLAGS))
                raise ValueError(
                    f"argument {position} is not one of the closed list of flags ({allowed})"
                )
        if len(set(flags)) != len(flags):
            raise ValueError("a flag is repeated")
        return args


class PinRule(BaseModel):
    """A package, or a prefix ending in `*`, that a recorded decision pins, and the reruns (and
    the full gate, `gate`) that must be green before it is taken."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=_RULE_ID)]
    packages: Annotated[
        list[Annotated[str, Field(pattern=_PACKAGE)]],
        Field(min_length=1, max_length=12),
    ]
    decision: _Line
    requires: list[Annotated[str, Field(pattern=_RERUN_ID)]]


class DriftConfig(UvCommand):
    """The dependency drift step: uv's own dry run, which must stay a dry run."""

    rules: list[PinRule]

    @field_validator("uv_args")
    @classmethod
    def _stays_a_dry_run(cls, args: list[str]) -> list[str]:
        if "--dry-run" not in args:
            raise ValueError(
                "must hold --dry-run, the one guard that keeps the check read-only"
            )
        return args

    @field_validator("rules")
    @classmethod
    def _rule_ids_are_distinct(cls, rules: list[PinRule]) -> list[PinRule]:
        ids = [rule.id for rule in rules]
        if repeated := sorted({name for name in ids if ids.count(name) > 1}):
            raise ValueError(f"an id is used twice: {', '.join(repeated)}")
        return rules


class LockCheckConfig(UvCommand):
    """The check that the lock still satisfies `pyproject.toml`: `lock --check`, with `--offline`
    as the only other flag, so that it cannot be made to upgrade anything."""

    @field_validator("uv_args")
    @classmethod
    def _checks(cls, args: list[str]) -> list[str]:
        if "--check" not in args:
            raise ValueError("must hold --check")
        if any(flag not in LOCK_CHECK_FLAGS for flag in args[1:]):
            raise ValueError("must hold no flag but --check and --offline")
        return args


class RerunConfig(BaseModel):
    """A parity rerun. `pytest` runs `targets` and needs `min_passed` (the floor of tests that
    must pass); `export_parity` runs the parity command over an export and has neither."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=_RERUN_ID)]
    kind: Literal["pytest", "export_parity"]
    cadence: Literal["nightly", "weekly"]
    timeout_seconds: _Seconds
    why: _Line
    targets: Annotated[list[_Target], Field(max_length=_MAX_TARGETS)] = Field(
        default_factory=list
    )
    min_passed: Annotated[int, Field(strict=True, ge=1)] | None = None
    max_skipped: Annotated[int, Field(strict=True, ge=0)] = 0

    @field_validator("id")
    @classmethod
    def _is_not_the_gate(cls, rerun_id: str) -> str:
        if rerun_id == GATE:
            raise ValueError(f"must not be {GATE}: a rule's requires uses that word")
        return rerun_id

    @model_validator(mode="after")
    def _holds_the_fields_of_its_kind(self) -> RerunConfig:
        if self.kind == "pytest":
            if not self.targets:
                raise ValueError(f"targets: a pytest rerun needs 1 to {_MAX_TARGETS}")
            if self.min_passed is None:
                raise ValueError("min_passed: a pytest rerun needs a floor")
        else:
            for name in ("targets", "min_passed"):
                if getattr(self, name):
                    raise ValueError(f"{name}: an export_parity rerun has none")
            if self.max_skipped:
                raise ValueError("max_skipped: an export_parity rerun has none")
        return self


class MaintainConfig(BaseModel):
    """The whole table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skipped_doctor_checks: dict[
        Annotated[str, Field(pattern=_CHECK_ID)],
        Annotated[str, AfterValidator(_closed_skip_reason)],
    ]
    lock_check: LockCheckConfig
    drift: DriftConfig
    reruns: list[RerunConfig]

    @model_validator(mode="after")
    def _reruns_are_distinct_and_the_rules_name_them(self) -> MaintainConfig:
        ids = [rerun.id for rerun in self.reruns]
        if repeated := sorted({name for name in ids if ids.count(name) > 1}):
            raise ValueError(f"reruns: an id is used twice: {', '.join(repeated)}")
        known = {*ids, GATE}
        for rule in self.drift.rules:
            if unknown := [name for name in rule.requires if name not in known]:
                raise ValueError(
                    f"drift.rules.{rule.id}.requires: names no rerun of this file: "
                    f"{', '.join(unknown)}"
                )
        return self


def _describe(error: ValidationError) -> str:
    problems = []
    for item in error.errors(include_input=False):
        where = ".".join(str(part) for part in item["loc"])
        problems.append(f"{where}: {item['msg']}" if where else item["msg"])
    return "; ".join(problems)


def load_maintain_config(data: bytes) -> MaintainConfig:
    """The table in the YAML `data`.

    Raises:
        ConfigError: `data` is not UTF-8 or not YAML, is not a mapping, or breaks the schema; the
            message names every field at fault and quotes none of the values.
    """
    try:
        raw: Any = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError("invalid maintain config: not UTF-8 text") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        line = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(
            f"invalid maintain config: the text is not valid YAML{line}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            "invalid maintain config: the top level must be a mapping of field names to "
            "values"
        )
    try:
        return MaintainConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid maintain config: {_describe(error)}") from error


def reruns_due(config: MaintainConfig, cadence: str) -> tuple[RerunConfig, ...]:
    """The reruns a run at `cadence` makes: `nightly` the nightly ones, `weekly` all of them,
    each in file order.

    Raises:
        ConfigError: `cadence` is not one of `CADENCES`.
    """
    if cadence == "weekly":
        return tuple(config.reruns)
    if cadence == "nightly":
        return tuple(rerun for rerun in config.reruns if rerun.cadence == "nightly")
    raise ConfigError(f"cadence must be one of: {', '.join(CADENCES)}")


def _names(pattern: str, name: str) -> bool:
    if pattern.endswith("*"):
        return name.startswith(normalize_name(pattern[:-1]))
    return name == normalize_name(pattern)


def rule_for(rules: Sequence[PinRule], package: str) -> PinRule | None:
    """The first rule that names `package` (after the name is put in PEP 503 form) or whose
    prefix, written with a final `*`, the name starts with; `torch-*` covers `torch-cuda`, and
    does not cover `torchvision`."""
    name = normalize_name(package)
    for rule in rules:
        if any(_names(pattern, name) for pattern in rule.packages):
            return rule
    return None
