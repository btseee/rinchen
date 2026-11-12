"""The states of a stage of `rinchen tiny run` and the closed lists of its reasons (D15, spec P4).

Every stage ends in one of three states and, unless it is `ok`, in a reason that is one of the strings
written here and nowhere else: a report never carries a line of another program's output, an exception
text or a path. A reason that holds a number is a function of that number (an exit code, a count of
seconds or of packages), and its template, with the placeholders in angle brackets, is the string that
`ALL_REASONS` lists; one test pins every string of the list. A check of a stage is named by an
identifier (lower-case letters, digits and underscores), never by text.

A stage is never dropped: what was not looked at is `skipped` with a skip reason (the first ones of
the list, `SKIP_REASONS`), and what could not be done is `fail` with a fail reason (the others).
`StageRecord` refuses a skip or a failure without the reason of its kind, a reason on an `ok`, an id
that is not one of the fifteen stages, and a part whose state, reason or id is not allowed. A stage
that has parts (the filter and the evaluation) lists the state of each; it is `ok` only when no part
failed, and a part that was skipped is one skipped on purpose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

from rinchen.config.tiny import STAGE_IDS

State = Literal["ok", "skipped", "fail"]
STATES: Final[tuple[State, ...]] = ("ok", "skipped", "fail")

SKIP_NO_LID: Final = "no language-identification model: GlotLID is a download"
SKIP_NO_LABSE: Final = (
    "LaBSE needs a model that was not downloaded: --qe-fake runs the planted stand-in"
)
SKIP_NO_QE: Final = "the QE scorer needs a model that was not downloaded: --qe-fake runs the planted stand-in"
SKIP_NO_NEURAL: Final = (
    "no neural metric scorer: only the seam of score files was built"
)
SKIP_TOKEN_RATIO: Final = (
    "the token-ratio check needs a SentencePiece model, which does not exist yet"
)
SKIP_EARLIER_FAILED: Final = "an earlier stage failed"
SKIP_REASONS: Final = (
    SKIP_NO_LID,
    SKIP_NO_LABSE,
    SKIP_NO_QE,
    SKIP_NO_NEURAL,
    SKIP_TOKEN_RATIO,
    SKIP_EARLIER_FAILED,
)

_EXIT: Final = "exit code <n>"
_TIMEOUT: Final = "timed out after <s> s"
FAIL_BUDGET: Final = "the wall-time budget is spent"
_CHECK: Final = "a check of the stage failed: <check>"
FAIL_INPUT: Final = "an input is missing or differs"
_ENVIRONMENT: Final = "<m> packages differ from the lock, <e> are not in it"

# Every reason: the skip constants, then the constants and the templates of the functions below.
ALL_REASONS: Final = (
    *SKIP_REASONS,
    _EXIT,
    _TIMEOUT,
    FAIL_BUDGET,
    _CHECK,
    FAIL_INPUT,
    _ENVIRONMENT,
)
_FAIL_TEMPLATES: Final = ALL_REASONS[len(SKIP_REASONS) :]

# The stages that have parts, and the parts each one lists (spec P4).
PART_IDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "filter": (
            "script_ratio",
            "numbers",
            "length",
            "lid",
            "minhash",
            "labse",
            "qe",
        ),
        "evaluate": ("memorised", "heldout", "neural_seam", "report"),
    }
)

_CHECK_ID: Final = r"[a-z][a-z0-9_]{0,31}"
_MAX_DETAIL_DEPTH: Final = 16


def _closed_pattern(template: str) -> re.Pattern[str]:
    """The pattern of the strings a template stands for: a `<check>` is a check identifier and any
    other placeholder a whole number (ASCII digits, a minus sign allowed)."""
    pieces = []
    last = 0
    for found in re.finditer(r"<([a-z]+)>", template):
        pieces.append(re.escape(template[last : found.start()]))
        pieces.append(_CHECK_ID if found.group(1) == "check" else "-?[0-9]+")
        last = found.end()
    pieces.append(re.escape(template[last:]))
    return re.compile("".join(pieces))


_FAIL_PATTERNS: Final = tuple(_closed_pattern(reason) for reason in _FAIL_TEMPLATES)


def is_skip_reason(reason: str) -> bool:
    """Whether `reason` is one of the skip reasons."""
    return reason in SKIP_REASONS


def is_fail_reason(reason: str) -> bool:
    """Whether `reason` is one of the fail reasons or a template of them with every placeholder
    filled in."""
    return any(pattern.fullmatch(reason) for pattern in _FAIL_PATTERNS)


def is_closed_reason(reason: str) -> bool:
    """Whether `reason` is one of the closed strings or a template of the list with every
    placeholder filled in: anything else is free text, which a report never holds."""
    return is_skip_reason(reason) or is_fail_reason(reason)


def _fill(template: str, **values: int | str) -> str:
    """The template with each `<name>` replaced by the value given for it."""
    for name, value in values.items():
        template = template.replace(f"<{name}>", str(value))
    return template


def fail_exit(code: int) -> str:
    return _fill(_EXIT, n=code)


def fail_timeout(seconds: int) -> str:
    return _fill(_TIMEOUT, s=seconds)


def fail_check(check_id: str) -> str:
    return _fill(_CHECK, check=check_id)


def fail_environment(mismatched: int, extra: int) -> str:
    return _fill(_ENVIRONMENT, m=mismatched, e=extra)


def _check_state_and_reason(state: object, reason: str | None, what: str) -> None:
    if state not in STATES:
        raise ValueError(f"a {what}'s state is none of {', '.join(STATES)}")
    if state == "ok":
        if reason is not None:
            raise ValueError(f"an ok {what} has no reason")
    elif not isinstance(reason, str):
        raise ValueError(f"a skipped or failed {what} needs a reason")
    elif state == "skipped" and not is_skip_reason(reason):
        raise ValueError(f"a skipped {what}'s reason is none of the skip reasons")
    elif state == "fail" and not is_fail_reason(reason):
        raise ValueError(f"a failed {what}'s reason is none of the fail reasons")


def _check_json(value: object, depth: int = 0) -> None:
    if depth > _MAX_DETAIL_DEPTH:
        raise ValueError("a stage's detail is nested too deeply")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a stage's detail holds a number that is not finite")
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("a stage's detail has a key that is not text")
        for item in value.values():
            _check_json(item, depth + 1)
    elif isinstance(value, list | tuple):
        for item in value:
            _check_json(item, depth + 1)
    else:
        raise ValueError(  # noqa: TRY004 - one error type for every refused record
            "a stage's detail holds a value that JSON cannot hold"
        )


@dataclass(frozen=True, slots=True)
class PartRecord:
    """One part of a stage: its id (one of `PART_IDS` of the stage, checked by the stage's
    record), its state and the reason of a skip or a failure."""

    id: str
    state: State
    reason: str | None = None

    def __post_init__(self) -> None:
        _check_state_and_reason(self.state, self.reason, "part")


@dataclass(frozen=True, slots=True)
class StageRecord:
    """What a stage came to: its id, its state, the reason of a skip or a failure, a
    JSON-compatible `detail` of counts and closed identifiers, and the state of each part."""

    id: str
    state: State
    reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    parts: Sequence[PartRecord] = ()

    def __post_init__(self) -> None:
        if self.id not in STAGE_IDS:
            raise ValueError("a stage's id is none of the fifteen stages")
        _check_state_and_reason(self.state, self.reason, "stage")
        if not isinstance(self.detail, Mapping):
            raise ValueError(  # noqa: TRY004 - one error type for every refused record
                "a stage's detail is a mapping"
            )
        _check_json(self.detail)
        self._check_parts()

    def _check_parts(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        allowed = PART_IDS.get(self.id, ())
        listed = [part.id for part in self.parts if isinstance(part, PartRecord)]
        if len(listed) != len(self.parts):
            raise ValueError("a part of a stage is a PartRecord")
        if any(part_id not in allowed for part_id in listed):
            raise ValueError("a part of a stage is none of the parts of that stage")
        if len(set(listed)) != len(listed):
            raise ValueError("a part of a stage is listed twice")
        if self.state == "ok" and any(part.state == "fail" for part in self.parts):
            raise ValueError("an ok stage has no failed part")


def unlisted_parts(record: StageRecord) -> tuple[str, ...]:
    """The parts of the record's stage that the record does not list, in the order of the stage.

    A stage that did not end `ok` may list some or none of its parts (the report lists the ones it did
    not judge as skipped); an `ok` stage that leaves one out has dropped a part silently, and the
    pipeline and the report both refuse it.
    """
    listed = {part.id for part in record.parts}
    return tuple(part for part in PART_IDS.get(record.id, ()) if part not in listed)
