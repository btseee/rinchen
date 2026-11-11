"""The runner every process of a maintenance check goes through (D14, spec M10).

`uv`, a pytest rerun and the parity command are started by a `CommandRunner`, which hands back
how the process ended and leaves everything it printed in a log file. The library takes the runner
as an argument; the command builds a `SubprocessRunner`, and the tests of the library use a
recording fake, so that none of them needs `uv`, a network or a model.

`SubprocessRunner` starts the process without a shell and without standard input, with standard
output and error combined into the log, in a session of its own. A process that runs past its
bound is stopped through its whole process group (SIGTERM, then SIGKILL after a grace) and reaped:
`subprocess.run` would stop only the direct child, and a grandchild (a test's worker, a model
server) would keep running and keep the log open. An interrupt in the wait does the same, at once,
before it propagates, because the child's session also keeps it out of reach of the terminal's
Ctrl-C. It is the runner of `rinchen.utils.logged_process` with the one difference that a
maintenance check needs: the exit code is handed back and not turned into an error (a pytest
rerun reads 1 for a failure and 5 for nothing collected). POSIX only: the check runs in CI on
Linux.

The environment of a child is the parent's with the GPU hidden and the Hugging Face libraries
offline (`child_environment`), and for `uv` also without colour or progress bar and without
interpreter downloads. Nothing of the environment is written anywhere.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

UV_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType(
    {"NO_COLOR": "1", "UV_NO_PROGRESS": "1", "UV_PYTHON_DOWNLOADS": "never"}
)
HIDDEN_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)
LOG_LIMIT: Final = 64 * 1024

# With polars imported (every `rinchen` command has it) SIGINT is handled by a handler that
# restarts a blocking call: a wait without a timeout is not woken, while a wait with a timeout
# is interrupted at once. So the wait is made in slices of this length (docs/VERIFY.md,
# D-POLARS-SIGINT).
_INTERRUPT_POLL_SECONDS: Final = 0.2
_TERM_GRACE_SECONDS: Final = 2.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    """How a command ended: its exit code (negative for a signal), or `None` after a timeout and
    when it did not start; `timed_out` when it was stopped for running past its bound; `started`
    False when the system refused to start it (no such program, no permission)."""

    exit_code: int | None
    timed_out: bool
    started: bool = True


class CommandRunner(Protocol):
    """Runs a program to its end, or to its bound, and leaves its output in `log`."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult: ...


def child_environment(parent: Mapping[str, str], *, uv: bool = False) -> dict[str, str]:
    """A copy of `parent` with the device hidden and the Hugging Face libraries offline, and,
    for `uv`, without colour or progress bar and without interpreter downloads. The switches
    win over the parent's values; `parent` is never modified."""
    return {**parent, **HIDDEN_ENVIRONMENT, **(UV_ENVIRONMENT if uv else {})}


class SubprocessRunner:
    """The runner of the real command: see the module docstring."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        """Run `argv` in `cwd` with exactly `env`, its output replacing `log` (created, with
        its directory), for at most `timeout` seconds.

        A program that cannot be started gives `CommandResult(None, False, False)` and an empty
        log. A timeout stops the process group and gives `CommandResult(None, True)`. A
        `KeyboardInterrupt` or a `SystemExit` raised while waiting kills the group and
        propagates.
        """
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as sink:
            try:
                process = subprocess.Popen(
                    list(argv),
                    shell=False,
                    cwd=cwd,
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError:
                return CommandResult(exit_code=None, timed_out=False, started=False)
            try:
                exit_code = _wait_for_exit(process, timeout)
            except BaseException:
                _kill_group(process)
                raise
            if exit_code is None:
                _stop_group(process)
                return CommandResult(exit_code=None, timed_out=True)
            return CommandResult(exit_code=exit_code, timed_out=False)


def read_log(path: Path, limit: int = LOG_LIMIT) -> str:
    """The first `limit` bytes of the log at `path`, decoded as UTF-8 with replacement; the
    empty text when there is no such file.

    Raises:
        ValueError: `limit` is negative (a negative size would read the whole file).
    """
    if limit < 0:
        raise ValueError("limit must not be negative")
    try:
        with path.open("rb") as handle:
            data = handle.read(limit)
    except FileNotFoundError:
        return ""
    return data.decode("utf-8", errors="replace")


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: int) -> int | None:
    """The exit code of `process`, waited for in slices so that a pending interrupt is raised
    between them; `None` once `timeout` seconds have passed with the process still running."""
    deadline = time.monotonic() + timeout
    while True:
        # A slice never runs past the deadline, so the timeout is not stretched.
        slice_seconds = min(
            _INTERRUPT_POLL_SECONDS, max(deadline - time.monotonic(), 0.0)
        )
        try:
            return process.wait(timeout=slice_seconds)
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                return None


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    """Ask the process group to end with SIGTERM, give the leader `_TERM_GRACE_SECONDS` to
    exit, then SIGKILL the whole group (what the leader left behind too) and reap the leader."""
    _signal_group(process, signal.SIGTERM)
    try:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_TERM_GRACE_SECONDS)
    finally:
        _kill_group(process)


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the process group of `process`, which leads it, and reap the leader.

    The group id is the leader's pid, which stays reserved while any member lives; with none
    left the call finds no group. (A pid that the system recycled for a new group between the
    reap and this call would be hit; pids are handed out in sequence, so that takes a wrap-around
    within microseconds.)"""
    _signal_group(process, signal.SIGKILL)
    process.wait()


def _signal_group(process: subprocess.Popen[bytes], number: signal.Signals) -> None:
    # ProcessLookupError: the group has already gone. PermissionError: a member is not ours to
    # signal, and there is no more to be done about it.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, number)
