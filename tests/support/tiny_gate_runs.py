"""The runs of the D15 gate that watch a command while it runs, or run it with a control profile (R7, R10 and
R11): a whole pipeline under a profile of its own, a pipeline given a budget of a few seconds, and a pipeline
that is sent a signal once it is training.

Each run starts through `tests/support/tiny_gate.py` (guarded, in a session of its own, every wait bounded) and
reports what it saw: the exit code, the seconds, what it printed and where it wrote, and, for the runs that end
early, the processes of the run that were alive at the signal and the ones still alive after the command ended.
A process of the run is found by the marker of its environment and by the session the command was started in
(`tests/support/tiny_gate_processes.py`). What is alive is read at once after the command has ended and been
reaped, with a grace of at most 1.5 s for a process that is being torn down (`left_after_the_end`): an orphan
that ends by itself some seconds later is a leak all the same. Whatever is alive is then killed by pid, and only
that, by `stop_run`.
"""

from __future__ import annotations

import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.support.tiny_gate import (
    Started,
    finish,
    run_tiny,
    send_to_group,
    start_tiny,
    stop_run,
    wait_for_file,
)
from tests.support.tiny_gate_processes import (
    RunProcess,
    left_after_the_end,
    processes_of_the_run,
    wait_for_command,
)

__all__ = [
    "INTERRUPTS",
    "STALE_FILES",
    "BudgetRun",
    "ControlRun",
    "Interrupted",
    "interrupt_run",
    "run_with_budget",
    "run_with_profile",
]

# The log of `train launch` appears when the training starts, about 50 s into a run.
TRAIN_LOG: Final = ("07_train", "logs", "train-launch.log")
TRAIN_START_SECONDS: Final = 600
# The trainer starts a few seconds after that log appears (the vocabulary is built first).
EOLE: Final = "bin/eole"
TRAINER: Final = "bin/eole train"
TRAINER_START_SECONDS: Final = 120
# The trainer runs this long before the signal: it has started its helpers, and no command is starting a child.
TRAINER_SETTLE_SECONDS: Final = 3
EXIT_WITHIN_SECONDS: Final = 60
# A run with a budget of a few seconds ends by itself within seconds; a run that ignored its budget would take
# as long as a whole pipeline, so the bound is far above that and the test of the run asserts the short time.
BUDGET_RUN_BOUND: Final = 400
# What an earlier run left in an output directory, which a new run takes away first.
STALE_FILES: Final = ("manifest.json", "report.json", "report.md")
# The signals of R11 and the exit code of the command for each.
INTERRUPTS: Final = (
    (signal.SIGTERM, 143),
    (signal.SIGINT, 130),
    (signal.SIGHUP, 129),
)


def _empty_directory(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    return directory


@dataclass(frozen=True)
class ControlRun:
    """A run of the command: how it ended, what it printed and where it wrote."""

    exit_code: int
    seconds: float
    stdout: list[str]
    stderr: str
    out_dir: Path


def run_with_profile(root: Path, profile: Path) -> ControlRun:
    """The command with the pipeline file `profile`, writing into `root/out` from `root/cwd` (both made)."""
    cwd = _empty_directory(root, "cwd")
    out_dir = root / "out"
    invocation = run_tiny(["--config", profile, "--out-dir", out_dir], cwd=cwd)
    return ControlRun(
        invocation.returncode,
        invocation.seconds,
        invocation.stdout,
        invocation.stderr,
        out_dir,
    )


@dataclass(frozen=True)
class BudgetRun:
    """A run with a small budget, and what of it was alive after the command ended."""

    run: ControlRun
    left: list[RunProcess]


def run_with_budget(root: Path, seconds: int) -> BudgetRun:
    """The command with `--budget-seconds seconds`, writing into `root/out` from `root/cwd` (both made)."""
    cwd = _empty_directory(root, "cwd")
    out_dir = root / "out"
    started = start_tiny(
        ["--budget-seconds", str(seconds), "--out-dir", out_dir], cwd=cwd
    )
    try:
        invocation = finish(started, timeout=BUDGET_RUN_BOUND)
        left = left_after_the_end(started.marker, session=started.process.pid)
    finally:
        stop_run(started)
    return BudgetRun(
        ControlRun(
            invocation.returncode,
            invocation.seconds,
            invocation.stdout,
            invocation.stderr,
            out_dir,
        ),
        left,
    )


@dataclass(frozen=True)
class Interrupted:
    """How a run that was sent a signal ended: its exit code, the seconds from its start to its end, the
    seconds from the signal to its end, the processes of the run alive when the signal was sent (and how
    many of them were the trainer), what of the run was alive after it ended, and where it wrote."""

    exit_code: int
    seconds: float
    seconds_after_the_signal: float
    at_the_signal: int
    trainers: int
    left: list[RunProcess]
    out_dir: Path


def _settle(started: Started, seconds: float) -> None:
    """Let the command run for `seconds` more, failing when it ends meanwhile."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert started.process.poll() is None, "the command ended before the signal"
        time.sleep(0.2)


def interrupt_run(
    root: Path,
    number: signal.Signals,
    stale_from: Path,
    *,
    at_the_spawn: bool = False,
) -> Interrupted:
    """Run the command into `root/out`, which holds the manifest and the two documents of an earlier run
    (copied from `stale_from`), send `number` to it and see how it ended.

    The signal is sent once the log of `train launch` exists and the trainer has run for a few seconds, or,
    with `at_the_spawn`, the moment the first `eole` (the vocabulary build) shows in the process table: the
    command that starts it has hardly returned from starting it."""
    cwd = _empty_directory(root, "cwd")
    out_dir = _empty_directory(root, "out")
    for name in STALE_FILES:
        shutil.copyfile(stale_from / name, out_dir / name)
    started = start_tiny(["--out-dir", out_dir], cwd=cwd)
    try:
        wait_for_file(
            out_dir.joinpath(*TRAIN_LOG), started, seconds=TRAIN_START_SECONDS
        )
        session = started.process.pid
        trainers = wait_for_command(
            started.marker,
            EOLE if at_the_spawn else TRAINER,
            session=session,
            seconds=TRAINER_START_SECONDS,
        )
        if not at_the_spawn:
            _settle(started, TRAINER_SETTLE_SECONDS)
        # The command, the commands it started and the trainer all carry the marker.
        at_the_signal = len(processes_of_the_run(started.marker, session=session))
        sent = time.perf_counter()
        send_to_group(started, number)
        invocation = finish(started, timeout=EXIT_WITHIN_SECONDS)
        after = time.perf_counter() - sent
        left = left_after_the_end(started.marker, session=session)
    finally:
        stop_run(started)
    return Interrupted(
        invocation.returncode,
        invocation.seconds,
        after,
        at_the_signal,
        len(trainers),
        left,
        out_dir,
    )
