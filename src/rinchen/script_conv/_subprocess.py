"""Run a command with its output kept in a log file.

The converter starts Eole and the CTranslate2 converter as subprocesses; both
write the combined output of the tool to a log and turn a failure into an
`UpstreamError` that carries the end of that log.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rinchen.errors import UpstreamError

LOG_TAIL_LINES = 20
# Enough for 20 lines of a traceback without reading a multi-megabyte training log.
_TAIL_READ_BYTES = 64 * 1024


def run_logged(
    command: Sequence[str], *, env: Mapping[str, str], log_path: Path, what: str
) -> None:
    """Run `command` with `env`, standard output and error combined into `log_path`
    (created, with its directory, and replaced if it exists).

    Raises:
        UpstreamError: the command exits with a non-zero code; `failure_with_log_tail`
            builds the message, which names `what`.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise failure_with_log_tail(
            f"{what} failed with exit code {completed.returncode}", log_path
        )


def failure_with_log_tail(problem: str, log_path: Path) -> UpstreamError:
    """An `UpstreamError` for `problem` whose message names the log and ends with its
    last `LOG_TAIL_LINES` lines."""
    return UpstreamError(
        f"{problem}; log: {log_path}; last {LOG_TAIL_LINES} lines:\n{_log_tail(log_path)}"
    )


def _log_tail(log_path: Path) -> str:
    with log_path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - _TAIL_READ_BYTES))
        tail = log.read()
    lines = tail.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])
