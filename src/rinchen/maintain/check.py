"""`run_check`: the steps of a maintenance check, in order, with every seam an argument (D14, spec
M2 and M4 to M9).

The steps are attempted in the order of `STEP_IDS` and each once: the doctor, the lock file, the
lock against `pyproject.toml`, the environment against the lock, the dependency drift and the
parity reruns. They are independent, so a failure never stops the next, except that the reruns are
skipped when the doctor or the environment check failed: their numbers would be about another
stack. Nothing here touches the network, a model or a file other than the logs the commands are
given; the command layer owns the doctor, the real runner, the hashing and the files of a run.

The data a check takes and gives is in `rinchen.maintain.check_types` and is re-exported here; the
steps are in `check_steps` (the doctor, the lock, the environment, the drift) and `check_reruns`.
A step that holds several items (the doctor's checks, the reruns) lists each item's state and
reason in its detail and has no reason of its own, unless it is `skipped` as a whole.
`seconds` holds one measurement for each step, read from `seams.clock`; the report holds no time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from rinchen.config.maintain import CADENCES, MaintainConfig
from rinchen.errors import ConfigError
from rinchen.maintain.check_reruns import reruns_step
from rinchen.maintain.check_steps import (
    doctor_step,
    drift_step,
    environment_step,
    lock_check_step,
)
from rinchen.maintain.check_types import (
    CheckOptions,
    CheckResult,
    CheckSeams,
    DoctorFinding,
    Inputs,
)
from rinchen.maintain.commands import CommandResult, CommandRunner
from rinchen.maintain.report import Report, Step

__all__ = [
    "CheckOptions",
    "CheckResult",
    "CheckSeams",
    "DoctorFinding",
    "Inputs",
    "run_check",
]


class _RecordingRunner:
    """Passes every call on and remembers the log of each."""

    def __init__(self, inner: CommandRunner) -> None:
        self._inner = inner
        self.logs: list[Path] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        self.logs.append(log)
        return self._inner.run(argv, cwd=cwd, env=env, log=log, timeout=timeout)


def _validated(config: MaintainConfig, options: CheckOptions) -> None:
    if options.cadence not in CADENCES:
        raise ConfigError(f"cadence must be one of: {', '.join(CADENCES)}")
    known = {rerun.id for rerun in config.reruns}
    if any(name not in known for name in options.selected):
        raise ConfigError("--rerun names no rerun of the configuration")


@contextmanager
def _measured(
    clock: Callable[[], float], seconds: dict[str, float], step_id: str
) -> Iterator[None]:
    start = clock()
    try:
        yield
    finally:
        seconds[step_id] = clock() - start


def run_check(
    inputs: Inputs,
    options: CheckOptions,
    seams: CheckSeams,
    *,
    project_dir: Path,
    out_dir: Path,
) -> CheckResult:
    """Make the steps of the check, in order, and return the report.

    Raises:
        ConfigError: the cadence is none of the two, or `selected` names a rerun that the
            configuration does not hold. Nothing has run by then.
    """
    config, lock = inputs.config, inputs.lock
    _validated(config, options)
    runner = _RecordingRunner(seams.runner)
    out = out_dir.absolute()
    seconds: dict[str, float] = {}

    with _measured(seams.clock, seconds, "doctor"):
        doctor, messages = doctor_step(config, seams.doctor)
    with _measured(seams.clock, seconds, "lock_file"):
        lock_file = Step("lock_file", "ok", None, {})
    with _measured(seams.clock, seconds, "lock_consistent"):
        lock_consistent = lock_check_step(
            config, seams.uv, runner, seams.parent_env, project_dir, out
        )
    with _measured(seams.clock, seconds, "environment"):
        environment = environment_step(lock, seams.installed)
    with _measured(seams.clock, seconds, "drift"):
        drift = drift_step(
            config,
            lock,
            seams.uv,
            runner,
            seams.parent_env,
            project_dir,
            out,
            enabled=options.drift,
        )
    with _measured(seams.clock, seconds, "reruns"):
        reruns = reruns_step(
            config,
            options,
            runner,
            seams.python,
            seams.parent_env,
            project_dir,
            out,
            environment_failed=doctor.state == "fail" or environment.state == "fail",
        )

    report = Report(
        cadence=options.cadence,
        inputs={
            "config": inputs.config_sha256,
            "lock": inputs.lock_sha256,
            "pyproject": inputs.pyproject_sha256,
        },
        lock={
            "packages": lock.entries,
            "names": len(lock.versions),
            "format": {"version": lock.format_version, "revision": lock.revision},
        },
        steps=(doctor, lock_file, lock_consistent, environment, drift, reruns),
    )
    logs = tuple(
        log.relative_to(out).as_posix() for log in runner.logs if log.is_file()
    )
    return CheckResult(report, seconds, messages, logs)
