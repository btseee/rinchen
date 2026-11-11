"""The signals of `rinchen maintain check` (D14, spec M3 and M10).

Every command the check starts runs in a session of its own (`rinchen.maintain.commands`), which
keeps it out of reach of a signal sent to this process: a SIGTERM that ended this process alone
would leave `uv` or a pytest rerun running until its bound. The runner kills the process group of
its command when an exception interrupts the wait, so while the check runs SIGTERM and SIGHUP raise
`SystemExit(128 + the signal number)` (143 and 129, the shell's status for a process that a signal
ended) and take the same path as Ctrl-C, whose `KeyboardInterrupt` typer ends with 130.

A signal that is ignored when the block starts stays ignored: an inherited ignore is a request not
to die on it (`nohup` ignores SIGHUP so that a closed terminal does not end a long run), and a
handler installed outside Python can be neither restored nor safely replaced. Handlers can only be
set from the main thread, so elsewhere the block just runs. SIGKILL cannot be caught: a command
left running after `kill -9` of this process is a known limit.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any, Final

_POSIX: Final = os.name == "posix"
_TERMINATION_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP) if _POSIX else ()


def _exit_with_status(number: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + number)


@contextmanager
def exit_on_termination_signals() -> Iterator[None]:
    """While the block runs, SIGTERM and SIGHUP raise `SystemExit(128 + signal number)` in the main
    thread, except a signal that is ignored or handled outside Python, which is left alone; the
    handlers that were replaced are put back afterwards, whatever the block does."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[signal.Signals, Any] = {}
    try:
        for number in _TERMINATION_SIGNALS:
            current = signal.getsignal(number)
            if current is signal.SIG_IGN or current is None:
                continue
            previous[number] = signal.signal(number, _exit_with_status)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
