"""A stand-in for a `subprocess.Popen` child that has already finished.

The converter tests replace `subprocess.Popen` by a recorder that writes what the
real tool would and hands back a `FakeProcess`. Its `pid` is one that no system
hands out, so that code under test which signals the process group of a child that
failed (`run_logged` does) can only find no such group and never signals a real one.
"""

from __future__ import annotations

import math

# `run_logged` waits in slices of at most `_INTERRUPT_POLL_S` and never past the
# timeout. A test that sets the poll to infinity makes every wait the fake records
# infinity when no timeout was asked for, and what is left of the timeout when one
# was: no finite limit can pass for "no timeout". (The real poll is 0.2 s, so the
# real `Popen.wait` is never given infinity.)
LONG_POLL_S = math.inf

# Linux allows process ids up to 2**22; a signal to this group is ProcessLookupError.
NO_SUCH_PID = 2**31 - 2


class FakeProcess:
    """A child that exits with `returncode` as soon as it is waited for; every
    `wait` appends its `timeout` to `waits`."""

    pid = NO_SUCH_PID

    def __init__(self, returncode: int, waits: list[float | None]) -> None:
        self.returncode = returncode
        self._waits = waits

    def wait(self, timeout: float | None = None) -> int:
        self._waits.append(timeout)
        return self.returncode
