"""The processes a run of the D15 gate leaves behind, read from `/proc` (Linux only).

A process belongs to a run when its environment holds the run's marker (every child of a gate command
inherits it: the helper puts `RINCHEN_GATE_RUN=<hex>` into the environment it hands the command) or when
it is in the session the command was started in. The stages of the pipeline start their own children in
sessions of their own, so the session id alone would miss them; the marker alone would miss a process that
dropped the variable. Only the processes of the run are ever signalled, and by pid; a zombie has ended and
is not counted.

`proc` is a parameter of every function so that the tests read a made-up tree and start nothing.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROC: Final = Path("/proc")
MARKER_VARIABLE: Final = "RINCHEN_GATE_RUN"
_ZOMBIE: Final = "Z"
_POLL_SECONDS: Final = 0.05


@dataclass(frozen=True)
class RunProcess:
    """A live process of the run: its pid, its session and its command line (the program and its
    arguments, as `/proc` holds them, never its environment)."""

    pid: int
    session: int
    command: str


def _stat(directory: Path) -> tuple[str, int]:
    """The state and the session of the process whose `/proc` directory is `directory`. The name sits
    between the first `(` and the LAST `)` of the line and may hold spaces and parentheses."""
    text = directory.joinpath("stat").read_text(encoding="utf-8", errors="replace")
    fields = text.rsplit(")", 1)[1].split()
    # After the name: state, ppid, pgrp, session.
    return fields[0], int(fields[3])


def _holds_marker(directory: Path, marker: str) -> bool:
    try:
        environment = directory.joinpath("environ").read_bytes()
    except OSError:
        return False
    wanted = f"{MARKER_VARIABLE}={marker}".encode()
    return wanted in environment.split(b"\0")


def _command(directory: Path) -> str:
    try:
        raw = directory.joinpath("cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(
        part.decode("utf-8", "replace") for part in raw.split(b"\0") if part
    )


def processes_of_the_run(
    marker: str, *, session: int | None = None, proc: Path = PROC
) -> list[RunProcess]:
    """The live processes that hold `marker` in their environment or are in `session`, by pid."""
    found: list[RunProcess] = []
    for directory in sorted(
        (entry for entry in proc.iterdir() if entry.name.isdigit()),
        key=lambda entry: int(entry.name),
    ):
        try:
            state, process_session = _stat(directory)
            ours = process_session == session or _holds_marker(directory, marker)
        except OSError, IndexError, ValueError:
            continue  # it ended while it was being read
        if ours and state != _ZOMBIE:
            found.append(
                RunProcess(int(directory.name), process_session, _command(directory))
            )
    return found


def kill_the_run(
    marker: str,
    *,
    proc: Path = PROC,
    kill: Callable[[int, int], None] = os.kill,
) -> list[int]:
    """SIGKILL every live process whose environment holds `marker`, and return their pids. A process
    that is gone by the time it is signalled is not an error. Nothing without the marker is touched."""
    pids = [item.pid for item in processes_of_the_run(marker, proc=proc)]
    for pid in pids:
        try:
            kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def wait_for_command(
    marker: str,
    needle: str,
    *,
    session: int | None = None,
    seconds: float = 60.0,
    proc: Path = PROC,
) -> list[RunProcess]:
    """The processes of the run whose command line holds `needle`, as soon as there is one (at most
    `seconds` of waiting; an empty list when none appeared)."""
    deadline = time.monotonic() + seconds
    while True:
        found = [
            item
            for item in processes_of_the_run(marker, session=session, proc=proc)
            if needle in item.command
        ]
        if found or time.monotonic() >= deadline:
            return found
        time.sleep(_POLL_SECONDS)


def wait_until_none_left(
    marker: str,
    *,
    session: int | None = None,
    seconds: float = 10.0,
    proc: Path = PROC,
) -> list[RunProcess]:
    """What of the run is still alive after at most `seconds`: an empty list when nothing is."""
    deadline = time.monotonic() + seconds
    while True:
        left = processes_of_the_run(marker, session=session, proc=proc)
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(_POLL_SECONDS)
