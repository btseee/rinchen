"""Run a command with its output kept in a log file.

Eole and the CTranslate2 converter run as subprocesses; both write the combined
output of the tool to a log and turn a failure into an `UpstreamError` that carries
the end of that log.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType

from rinchen.errors import UpstreamError

LOG_TAIL_LINES = 20
# Enough for 20 lines of a traceback without reading a multi-megabyte training log.
_TAIL_READ_BYTES = 64 * 1024
# A progress bar redrawn with carriage returns would otherwise fill the message.
_TAIL_LINE_CHARS = 300
_ELLIPSIS = "..."
_POSIX = os.name == "posix"
# With polars imported (every `rinchen` command has it) SIGINT is handled by a
# handler that restarts a blocking call: a wait without a timeout is not woken, and
# Ctrl-C would be raised only when the command ends (hours into an Eole training),
# while a wait with a timeout is interrupted at once. So the wait is made in slices
# of this length (docs/VERIFY.md, D-POLARS-SIGINT).
_INTERRUPT_POLL_S = 0.2
# The signals that end a process without Ctrl-C: `kill`, a service stop (SIGTERM) and
# a closed terminal or SSH session (SIGHUP, absent on Windows).
_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP) if _POSIX else ()


def run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
    what: str,
    timeout: float | None = None,
) -> None:
    """Run `command` with `env`, standard output and error combined into `log_path`
    (created, with its directory, and replaced if it exists).

    The child has no standard input. On POSIX it leads a session of its own, so that
    a timeout or an interrupt can kill everything it started: `subprocess.run`
    would kill only the direct child, and a grandchild (a data-loader worker, say)
    would keep running and keep the log open. `timeout` is in seconds; None waits
    for as long as the command runs.

    The child's own session also keeps it out of reach of a SIGTERM or SIGHUP sent
    to this process, which would then die and leave the command running. So while
    this call runs on the main thread, both signals raise `SystemExit(128 + signal
    number)` (the shell's status for a process ended by a signal) and the usual path
    kills the command's group; the caller's handlers are put back afterwards. A signal
    that is ignored when the call starts stays ignored: an inherited ignore is a
    request not to die on it (`nohup` ignores SIGHUP so that a closed terminal does
    not end a long run). SIGKILL cannot be caught: a child left running after `kill
    -9` of this process is a known limit.

    Raises:
        UpstreamError: the command exits with a non-zero code, or runs past
            `timeout` (it is then killed and reaped); `failure_with_log_tail` builds
            the message, which names `what`. Any other exception that interrupts the
            wait, such as KeyboardInterrupt or the SystemExit of a termination
            signal, kills the command first and propagates.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log, _exit_on_termination_signals():
        process = subprocess.Popen(
            list(command),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=_POSIX,
        )
        try:
            returncode = _wait_for_exit(process, timeout)
        except subprocess.TimeoutExpired:
            _kill_with_descendants(process)
            raise failure_with_log_tail(
                f"{what} timed out after {timeout} s and was killed", log_path
            ) from None
        except BaseException:
            _kill_with_descendants(process)
            raise
    if returncode != 0:
        raise failure_with_log_tail(f"{what} failed: {_describe(returncode)}", log_path)


@contextlib.contextmanager
def _exit_on_termination_signals() -> Iterator[None]:
    """While the block runs, SIGTERM and SIGHUP raise `SystemExit(128 + signal number)`
    in the main thread, except a signal that is ignored (`signal.SIG_IGN`), which is
    left alone; the handlers that were replaced are restored afterwards, whatever the
    block does. Handlers can only be set from the main thread, so elsewhere the block
    just runs."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def exit_with_status(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    previous = {
        number: signal.signal(number, exit_with_status)
        for number in _TERMINATION_SIGNALS
        if signal.getsignal(number) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: float | None) -> int:
    """The exit code of `process`, waited for in slices of `_INTERRUPT_POLL_S` so that
    a pending interrupt is raised between them. Raises `subprocess.TimeoutExpired`
    once `timeout` seconds have passed; None waits for as long as the process runs."""
    limit = math.inf if timeout is None else timeout
    deadline = time.monotonic() + limit
    while True:
        # A slice never runs past the deadline, so the timeout is not stretched.
        slice_s = min(_INTERRUPT_POLL_S, max(deadline - time.monotonic(), 0.0))
        try:
            return process.wait(timeout=slice_s)
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(process.args, limit) from None


def _describe(returncode: int) -> str:
    if returncode < 0:
        return f"killed by signal {-returncode} (exit code {returncode})"
    return f"exit code {returncode}"


def _kill_with_descendants(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's process group (the child is its leader) and reap the
    child; where there are no process groups, kill the child only."""
    if _POSIX:
        # ProcessLookupError: the group has already gone.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait()


def failure_with_log_tail(problem: str, log_path: Path) -> UpstreamError:
    """An `UpstreamError` for `problem` whose message names the log and ends with its
    last `LOG_TAIL_LINES` lines, each cut to 300 characters."""
    return UpstreamError(
        f"{problem}; log: {log_path}; last {LOG_TAIL_LINES} lines:\n{_log_tail(log_path)}"
    )


def _log_tail(log_path: Path) -> str:
    with log_path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - _TAIL_READ_BYTES))
        tail = log.read()
    lines = tail.decode("utf-8", errors="replace").splitlines()
    return "\n".join(_clip(line) for line in lines[-LOG_TAIL_LINES:])


def _clip(line: str) -> str:
    if len(line) <= _TAIL_LINE_CHARS:
        return line
    return line[:_TAIL_LINE_CHARS] + _ELLIPSIS
