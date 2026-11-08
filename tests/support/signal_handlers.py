"""Signal dispositions for the tests that depend on them.

A process started as a background job (`cmd &`, `nohup`) inherits SIGINT and SIGHUP as
ignored, and Python then leaves them ignored: a test that sends itself a real SIGINT
expecting `KeyboardInterrupt`, or that expects the default for SIGTERM and SIGHUP, would
pass in a terminal and fail there. Such a test installs the handlers it relies on for its
duration (`handlers_installed`; the `default_signal_handlers` fixture in `conftest.py`),
and a test of the ignored disposition sets `signal.SIG_IGN` itself.
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Iterator, Mapping
from typing import Any

# What a terminal session starts a command with: Ctrl-C raises KeyboardInterrupt and the
# termination signals act by default. SIGHUP does not exist on Windows.
DEFAULT_HANDLERS: Mapping[int, Any] = {
    signal.SIGINT: signal.default_int_handler,
    signal.SIGTERM: signal.SIG_DFL,
    **({signal.SIGHUP: signal.SIG_DFL} if hasattr(signal, "SIGHUP") else {}),
}


@contextlib.contextmanager
def handlers_installed(handlers: Mapping[int, Any]) -> Iterator[None]:
    """Install `handlers` (signal number to handler) for the block and put the previous
    ones back afterwards, however the block ends. Main thread only."""
    previous: dict[int, Any] = {}
    try:
        for number, handler in handlers.items():
            previous[number] = signal.signal(number, handler)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
