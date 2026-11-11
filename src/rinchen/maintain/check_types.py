"""The data a maintenance check takes and gives (D14, spec M2): the arguments of
`rinchen.maintain.check.run_check` and its result, which that module re-exports.

`Inputs` holds what the command read and hashed before the check began: the configuration, the lock
as read and the SHA-256 of the three files. `CheckOptions` is what the operator asked for and
`CheckSeams` everything the check takes from outside (the runner, `uv`, the interpreter, the doctor,
the installed versions, a clock, the parent environment), so that a test of the check starts
nothing. A `DoctorFinding` is one check of the doctor; its message goes to the command to print and
never into a report. Each type refuses the value that would put free text in a report: a
check id that is not a doctor id and a digest that is not SHA-256 hex.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from rinchen.config.maintain import MaintainConfig
from rinchen.maintain.commands import CommandRunner
from rinchen.maintain.lock import LockFile
from rinchen.maintain.report import Report
from rinchen.maintain.reruns import ExportInputs

DoctorStatus = Literal["pass", "fail", "skip"]
DOCTOR_STATUSES: Final[tuple[DoctorStatus, ...]] = ("pass", "fail", "skip")

_CHECK_ID: Final = re.compile(r"D-[A-Z0-9-]{1,30}")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    """One check of the doctor: its id, how it ended and its message (never put in a report)."""

    check_id: str
    status: DoctorStatus
    message: str

    def __post_init__(self) -> None:
        if not _CHECK_ID.fullmatch(self.check_id):
            raise ValueError("check_id is not a doctor check id (D- and capitals)")
        if self.status not in DOCTOR_STATUSES:
            raise ValueError(f"status is none of {', '.join(DOCTOR_STATUSES)}")


@dataclass(frozen=True, slots=True)
class CheckOptions:
    """What the operator asked for: the cadence, whether to run the drift step and the reruns, the
    reruns named with `--rerun` (they win over the cadence) and the export for `parity_export`."""

    cadence: str
    drift: bool
    reruns: bool
    selected: tuple[str, ...]
    export: ExportInputs | None


@dataclass(frozen=True, slots=True)
class CheckSeams:
    """Everything the check takes from outside: the runner, the `uv` to run (None when there is
    none), the interpreter of the reruns, the doctor, the installed versions, a clock and the
    parent environment of every command."""

    runner: CommandRunner
    uv: str | None
    python: str
    doctor: Callable[[], Sequence[DoctorFinding]]
    installed: Callable[[], Mapping[str, Sequence[str]]]
    clock: Callable[[], float]
    parent_env: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Inputs:
    """The configuration, the lock as read, and the SHA-256 of the three files the check read."""

    config: MaintainConfig
    lock: LockFile
    config_sha256: str
    lock_sha256: str
    pyproject_sha256: str

    def __post_init__(self) -> None:
        for name in ("config_sha256", "lock_sha256", "pyproject_sha256"):
            if not _DIGEST.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} is not a SHA-256 digest in lower-case hex")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The report; the seconds of each step; the `(id, message)` of every failed doctor check, for
    the command to print; the names, relative to the output directory, of the logs written."""

    report: Report
    seconds: Mapping[str, float]
    doctor_messages: tuple[tuple[str, str], ...]
    logs: tuple[str, ...]
