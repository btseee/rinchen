"""The doctor checks of `rinchen maintain check` (D14, spec M4): the six that answer a CPU question,
called in process in the order of the doctor.

The doctor also has a check that asks a device question. A maintenance command never asks one: its
row is skipped by the configuration (`skipped_doctor_checks`), whether or not a device is hidden.
So the list here is written by hand and not taken from `doctor.ALL_CHECKS`: a check that someone
adds to the doctor is not run by this command until it is put here, and a test pins that the
doctor's list is exactly this one plus the device check, so that the decision is forced.

A check that raises is a failed check of its own id, derived from the function's name
(`check_eole_cli` is `D-EOLE-CLI`), so the rest of the checks still run. The message of a finding
can hold an exception text and a path: the command prints it and never writes it to a file.

The checks run in this process, so the process must not see the caller's device either: a child
gets `CUDA_VISIBLE_DEVICES=""` from `rinchen.maintain.commands.child_environment`, and the command
holds `device_hidden` open for the whole of its own run.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

from rinchen.cli import doctor
from rinchen.cli.doctor import DoctorResult
from rinchen.maintain.check_types import DoctorFinding, DoctorStatus

CPU_CHECKS: Final[tuple[Callable[[], DoctorResult], ...]] = (
    doctor.check_py,
    doctor.check_wheels,
    doctor.check_eole_py,
    doctor.check_eole_cli,
    doctor.check_ct2_conv,
    doctor.check_ct2_import,
)

_DEVICE_VARIABLE: Final = "CUDA_VISIBLE_DEVICES"
_PREFIX: Final = "check_"
_STATUSES: Final[dict[str, DoctorStatus]] = {
    "pass": "pass",
    "fail": "fail",
    "skip": "skip",
}


def check_id_of(check: Callable[[], DoctorResult]) -> str:
    """The id of the doctor check that `check` is: `D-` and the rest of its name in capitals with
    `-` for `_`.

    Raises:
        ValueError: the function's name does not start with `check_`.
    """
    name = check.__name__
    if not name.startswith(_PREFIX):
        raise ValueError(f"a doctor check is a function named {_PREFIX}<what>")
    return "D-" + name.removeprefix(_PREFIX).upper().replace("_", "-")


def _finding_of(check: Callable[[], DoctorResult]) -> DoctorFinding:
    try:
        result = check()
    except Exception as error:  # noqa: BLE001 - whatever a check raises, the check failed
        return DoctorFinding(
            check_id_of(check), "fail", f"{type(error).__name__}: {error}"
        )
    status = _STATUSES.get(result.status)
    if status is None:
        return DoctorFinding(
            check_id_of(check), "fail", "the check returned a status of no known kind"
        )
    return DoctorFinding(result.check_id, status, result.message)


def run_cpu_checks() -> list[DoctorFinding]:
    """Each of `CPU_CHECKS` once, in order, as a finding."""
    return [_finding_of(check) for check in CPU_CHECKS]


@contextmanager
def device_hidden() -> Iterator[None]:
    """Hide every GPU from this process for the block (`CUDA_VISIBLE_DEVICES` is the empty text),
    and put the caller's value back, or remove the variable if there was none, whatever the block
    does."""
    previous = os.environ.get(_DEVICE_VARIABLE)
    os.environ[_DEVICE_VARIABLE] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_DEVICE_VARIABLE, None)
        else:
            os.environ[_DEVICE_VARIABLE] = previous
