"""Run a command with its output kept in a log file.

The converter starts Eole and the CTranslate2 converter as subprocesses; both
write the combined output of the tool to a log and turn a failure into an
`UpstreamError` that carries the end of that log.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rinchen.errors import UpstreamError

LOG_TAIL_LINES = 20
# Enough for 20 lines of a traceback without reading a multi-megabyte training log.
_TAIL_READ_BYTES = 64 * 1024
# A progress bar redrawn with carriage returns would otherwise fill the message.
_TAIL_LINE_CHARS = 300
_ELLIPSIS = "..."
_POSIX = os.name == "posix"


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

    Raises:
        UpstreamError: the command exits with a non-zero code, or runs past
            `timeout` (it is then killed and reaped); `failure_with_log_tail` builds
            the message, which names `what`. Any other exception that interrupts the
            wait, such as KeyboardInterrupt, kills the command first and propagates.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=_POSIX,
        )
        try:
            returncode = process.wait(timeout=timeout)
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
