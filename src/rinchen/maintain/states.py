"""The states of a step of `rinchen maintain check` and the closed lists of its reasons.

Every step ends in one of four states and, unless it is `ok`, in a reason that is one of the
strings written here and nowhere else: a report never carries a line of another program's output,
an exception text or a path. A reason that holds a number is a function of that number (a count,
an exit code, a bound), and its template, with the placeholders in angle brackets, is the string
that `ALL_REASONS` lists; one test pins every string of the list.

A step is never dropped: what was not looked at is `skipped` with one of the skip reasons, and
what could not be looked at is `fail` with one of the fail reasons.
"""

from __future__ import annotations

import re
from typing import Final, Literal

State = Literal["ok", "drift", "fail", "skipped"]
STATES: Final[tuple[State, ...]] = ("ok", "drift", "fail", "skipped")

SKIP_GPU_CHECK: Final = "a GPU check: maintain check never asks a device question"
SKIP_NO_DRIFT: Final = "disabled by --no-drift"
SKIP_NO_RERUNS: Final = "disabled by --no-reruns"
SKIP_NO_EXPORT: Final = "no export was given"
SKIP_ENVIRONMENT: Final = "a doctor or environment check failed"
SKIP_DOCTOR_SELF: Final = "reported by the doctor itself"
# The skip reasons that are a fixed string: the ones a configuration may give a skipped check.
SKIP_REASONS: Final = (
    SKIP_GPU_CHECK,
    SKIP_NO_DRIFT,
    SKIP_NO_RERUNS,
    SKIP_NO_EXPORT,
    SKIP_ENVIRONMENT,
    SKIP_DOCTOR_SELF,
)
_NOT_DUE: Final = "not due at the <cadence> cadence"

FAIL_UV_MISSING: Final = "uv was not found"
_UV_EXIT: Final = "uv exited with code <n>"
_TIMEOUT: Final = "timed out after <s> s"
_UNRECOGNISED: Final = "the output of uv has <n> lines this version does not recognise"
_LOG_TOO_LONG: Final = "the output of uv is longer than <limit> bytes"
FAIL_NO_RESULT: Final = "uv printed no recognisable result"
FAIL_CONTRADICTION: Final = "uv printed a no-change line next to change lines"
_CROSS_CHECK: Final = (
    "uv reported a locked version that differs from uv.lock for <n> packages"
)
_ENVIRONMENT: Final = "<m> packages differ from the lock, <e> are not in it"
FAIL_DOCTOR: Final = "the check failed"
_PYTEST_EXIT: Final = "pytest exited with code <n>"
_TESTS: Final = "<f> failed and <e> errors"
_FLOOR: Final = "<p> passed, fewer than the floor <m>"
_SKIPPED: Final = "<k> skipped, more than the allowed <m>"
FAIL_NO_JUNIT: Final = "no JUnit report was written"
FAIL_TARGET: Final = "a target does not exist"
_COMMAND_EXIT: Final = "the command exited with code <n>"
FAIL_NO_FIGURES: Final = "the command wrote no readable parity.json"

# Every reason: the constants and the templates of the functions below, in the order of the spec.
ALL_REASONS: Final = (
    *SKIP_REASONS,
    _NOT_DUE,
    FAIL_UV_MISSING,
    _UV_EXIT,
    _TIMEOUT,
    _UNRECOGNISED,
    _LOG_TOO_LONG,
    FAIL_NO_RESULT,
    FAIL_CONTRADICTION,
    _CROSS_CHECK,
    _ENVIRONMENT,
    FAIL_DOCTOR,
    _PYTEST_EXIT,
    _TESTS,
    _FLOOR,
    _SKIPPED,
    FAIL_NO_JUNIT,
    FAIL_TARGET,
    _COMMAND_EXIT,
    FAIL_NO_FIGURES,
)


def _closed_pattern(template: str) -> re.Pattern[str]:
    """The pattern of the strings a template stands for: a `<cadence>` is one of the two words
    and any other placeholder a whole number (ASCII digits, a minus sign allowed)."""
    pieces = []
    last = 0
    for found in re.finditer(r"<([a-z]+)>", template):
        pieces.append(re.escape(template[last : found.start()]))
        pieces.append(
            "(?:nightly|weekly)" if found.group(1) == "cadence" else "-?[0-9]+"
        )
        last = found.end()
    pieces.append(re.escape(template[last:]))
    return re.compile("".join(pieces))


_CLOSED_PATTERNS: Final = tuple(_closed_pattern(reason) for reason in ALL_REASONS)


def is_closed_reason(reason: str) -> bool:
    """Whether `reason` is one of the closed strings or a template of the list with every
    placeholder filled in: anything else is free text, which a report never holds."""
    return any(pattern.fullmatch(reason) for pattern in _CLOSED_PATTERNS)


def _fill(template: str, **values: int | str) -> str:
    """The template with each `<name>` replaced by the value given for it."""
    for name, value in values.items():
        template = template.replace(f"<{name}>", str(value))
    return template


def skip_not_due(cadence: str) -> str:
    return _fill(_NOT_DUE, cadence=cadence)


def fail_uv_exit(code: int) -> str:
    return _fill(_UV_EXIT, n=code)


def fail_timeout(seconds: int) -> str:
    return _fill(_TIMEOUT, s=seconds)


def fail_unrecognised(count: int) -> str:
    return _fill(_UNRECOGNISED, n=count)


def fail_log_too_long(limit: int) -> str:
    return _fill(_LOG_TOO_LONG, limit=limit)


def fail_cross_check(count: int) -> str:
    return _fill(_CROSS_CHECK, n=count)


def fail_environment(mismatched: int, extra: int) -> str:
    return _fill(_ENVIRONMENT, m=mismatched, e=extra)


def fail_pytest_exit(code: int) -> str:
    return _fill(_PYTEST_EXIT, n=code)


def fail_tests(failed: int, errors: int) -> str:
    return _fill(_TESTS, f=failed, e=errors)


def fail_floor(passed: int, floor: int) -> str:
    return _fill(_FLOOR, p=passed, m=floor)


def fail_skipped(skipped: int, allowed: int) -> str:
    return _fill(_SKIPPED, k=skipped, m=allowed)


def fail_command_exit(code: int) -> str:
    return _fill(_COMMAND_EXIT, n=code)
