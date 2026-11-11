"""The first four steps of a maintenance check (D14, spec M4, M5 and M6): the doctor, the lock
against `pyproject.toml`, the environment against the lock and the dependency drift.

Each builder returns a `Step` of the shapes that `rinchen.maintain.report` documents and takes
what it needs as arguments. What goes into a step is a closed vocabulary: a doctor check is its id
and a state (the message is handed back for the command to print), every reason is one of
`rinchen.maintain.states`, and a name or version copied from the environment is a plain token or
a placeholder. The lock file step has nothing to build (its figures are in the report's `lock`)
and the reruns are in `rinchen.maintain.check_reruns`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from rinchen.config.maintain import MaintainConfig
from rinchen.maintain import states
from rinchen.maintain.check_types import DoctorFinding
from rinchen.maintain.commands import (
    LOG_LIMIT,
    CommandResult,
    CommandRunner,
    child_environment,
    read_complete_log,
)
from rinchen.maintain.drift import DriftOutcome, drift_outcome
from rinchen.maintain.environment import EnvironmentResult, check_environment
from rinchen.maintain.lock import LockFile
from rinchen.maintain.report import Step, group_state
from rinchen.maintain.reruns import LOGS_DIR

MAX_NAMES: Final = 20
INVALID_TOKEN: Final = "(not a valid name or version)"
LOCK_CHECK_LOG: Final = "lock_consistent.log"
DRIFT_LOG: Final = "drift.log"

_TOKEN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}")
_DOCTOR_STATES: Final[Mapping[str, tuple[states.State, str | None]]] = {
    "pass": ("ok", None),
    "fail": ("fail", states.FAIL_DOCTOR),
    "skip": ("skipped", states.SKIP_DOCTOR_SELF),
}


def _check_row(
    check_id: str, state: states.State, reason: str | None
) -> dict[str, Any]:
    return {"id": check_id, "state": state, "reason": reason}


# ---- the doctor ---------------------------------------------------------------------------------------


def doctor_step(
    config: MaintainConfig, doctor: Callable[[], Sequence[DoctorFinding]]
) -> tuple[Step, tuple[tuple[str, str], ...]]:
    """The doctor's findings as checks, and the `(id, message)` of each failed one.

    A finding of a configured skip is dropped and that skip stands in its place; a configured skip
    the doctor did not report follows the findings. A doctor that raises is one failed check named
    `doctor`.
    """
    skips = config.skipped_doctor_checks
    rows: list[dict[str, Any]] = []
    messages: list[tuple[str, str]] = []
    try:
        findings = tuple(doctor())
    except Exception as error:  # noqa: BLE001 - any failure of the doctor is a failed check
        findings = ()
        rows.append(_check_row("doctor", "fail", states.FAIL_DOCTOR))
        messages.append(("doctor", f"{type(error).__name__}: {error}"))
    listed: set[str] = set()
    for found in findings:
        if found.check_id in skips:
            if found.check_id not in listed:
                rows.append(
                    _check_row(found.check_id, "skipped", skips[found.check_id])
                )
                listed.add(found.check_id)
            continue
        state, reason = _DOCTOR_STATES[found.status]
        rows.append(_check_row(found.check_id, state, reason))
        if state == "fail":
            messages.append((found.check_id, found.message))
    rows += [
        _check_row(check_id, "skipped", reason)
        for check_id, reason in skips.items()
        if check_id not in listed
    ]
    state = group_state([row["state"] for row in rows])
    reason = None
    if state == "skipped":
        reason = rows[0]["reason"] if rows else states.SKIP_DOCTOR_SELF
    return Step("doctor", state, reason, {"checks": rows}), tuple(messages)


# ---- the lock check ----------------------------------------------------------------------------------


def _uv_failure(outcome: CommandResult, timeout_seconds: int) -> str | None:
    """Why a run of uv that was to exit 0 did not, or None when it did."""
    if outcome.timed_out:
        return states.fail_timeout(timeout_seconds)
    if outcome.exit_code is None:
        return states.FAIL_UV_MISSING
    if outcome.exit_code != 0:
        return states.fail_uv_exit(outcome.exit_code)
    return None


def lock_check_step(
    config: MaintainConfig,
    uv: str | None,
    runner: CommandRunner,
    parent_env: Mapping[str, str],
    project_dir: Path,
    out: Path,
) -> Step:
    """`uv lock --check` (the arguments of the configuration): does the lock still satisfy
    `pyproject.toml`. Without a `uv` it is a failure and nothing runs."""
    if uv is None:
        return Step("lock_consistent", "fail", states.FAIL_UV_MISSING, {})
    outcome = runner.run(
        [uv, *config.lock_check.uv_args],
        cwd=project_dir,
        env=child_environment(parent_env, uv=True),
        log=out / LOGS_DIR / LOCK_CHECK_LOG,
        timeout=config.lock_check.timeout_seconds,
    )
    reason = _uv_failure(outcome, config.lock_check.timeout_seconds)
    return Step("lock_consistent", "ok" if reason is None else "fail", reason, {})


# ---- the environment ---------------------------------------------------------------------------------


def _token(text: str) -> str:
    return text if _TOKEN.fullmatch(text) else INVALID_TOKEN


def _names_and_more(names: Sequence[Any]) -> tuple[list[Any], int]:
    return list(names[:MAX_NAMES]), max(len(names) - MAX_NAMES, 0)


def _environment_detail(result: EnvironmentResult) -> dict[str, Any]:
    mismatched, mismatched_more = _names_and_more(result.mismatched)
    extra, extra_more = _names_and_more(result.extra)
    return {
        "installed": result.installed,
        "mismatched": [
            {
                "name": _token(item.name),
                "installed": [_token(version) for version in item.installed],
                "locked": [_token(version) for version in item.locked],
            }
            for item in mismatched
        ],
        "mismatched_more": mismatched_more,
        "extra": [_token(name) for name in extra],
        "extra_more": extra_more,
        "not_installed": result.not_installed,
    }


def environment_step(
    lock: LockFile, installed: Callable[[], Mapping[str, Sequence[str]]]
) -> Step:
    """The installed packages against the lock; the detail lists at most `MAX_NAMES` names of
    each kind and counts the rest, and counts the locked packages that are not installed."""
    result = check_environment(lock, installed())
    reason = (
        states.fail_environment(len(result.mismatched), len(result.extra))
        if result.state == "fail"
        else None
    )
    return Step("environment", result.state, reason, _environment_detail(result))


# ---- the drift ---------------------------------------------------------------------------------------


def _drift_detail(outcome: DriftOutcome) -> dict[str, Any]:
    entries = [
        {
            "package": entry.package,
            "change": entry.change,
            "locked": entry.locked,
            "available": entry.available,
            "class": entry.klass,
            "rule": entry.rule,
            "requires": list(entry.requires),
        }
        for entry in outcome.entries
    ]
    counted = [entry["change"] for entry in entries] + [
        entry["class"] for entry in entries
    ]
    return {
        "entries": entries,
        "counts": {
            name: counted.count(name)
            for name in ("update", "add", "remove", "drift", "blocked")
        },
        "warnings": outcome.warnings,
    }


def drift_step(
    config: MaintainConfig,
    lock: LockFile,
    uv: str | None,
    runner: CommandRunner,
    parent_env: Mapping[str, str],
    project_dir: Path,
    out: Path,
    *,
    enabled: bool,
) -> Step:
    """What `uv lock --upgrade --dry-run` (the arguments of the configuration) says, decided by
    `rinchen.maintain.drift`. Disabled it is skipped and nothing runs; without a `uv` it is a
    failure. The detail is empty unless the run was read and ended `ok` or `drift`."""
    if not enabled:
        return Step("drift", "skipped", states.SKIP_NO_DRIFT, {})
    if uv is None:
        return Step("drift", "fail", states.FAIL_UV_MISSING, {})
    log = out / LOGS_DIR / DRIFT_LOG
    outcome = runner.run(
        [uv, *config.drift.uv_args],
        cwd=project_dir,
        env=child_environment(parent_env, uv=True),
        log=log,
        timeout=config.drift.timeout_seconds,
    )
    decided = drift_outcome(
        exit_code=outcome.exit_code,
        timed_out=outcome.timed_out,
        text=read_complete_log(log, LOG_LIMIT),
        lock=lock,
        rules=config.drift.rules,
        timeout_seconds=config.drift.timeout_seconds,
        log_limit=LOG_LIMIT,
    )
    detail = _drift_detail(decided) if decided.state != "fail" else {}
    return Step("drift", decided.state, decided.reason, detail)
