"""A stand-in for a `subprocess.Popen` child that has already finished.

The converter tests replace `subprocess.Popen` by a recorder that writes what the
real tool would and hands back a `FakeProcess`. It has no `pid` on purpose: code
under test that tried to signal it would fail loudly instead of signalling a real
process group.
"""

from __future__ import annotations

import math

# `run_logged` waits in slices of at most `_INTERRUPT_POLL_S` and never past the
# timeout. A test that sets the poll to infinity makes every wait the fake records
# infinity when no timeout was asked for, and what is left of the timeout when one
# was: no finite limit can pass for "no timeout". (The real poll is 0.2 s, so the
# real `Popen.wait` is never given infinity.)
LONG_POLL_S = math.inf


class FakeProcess:
    """A child that exits with `returncode` as soon as it is waited for; every
    `wait` appends its `timeout` to `waits`."""

    def __init__(self, returncode: int, waits: list[float | None]) -> None:
        self.returncode = returncode
        self._waits = waits

    def wait(self, timeout: float | None = None) -> int:
        self._waits.append(timeout)
        return self.returncode
