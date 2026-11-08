"""A stand-in for a `subprocess.Popen` child that has already finished.

The converter tests replace `subprocess.Popen` by a recorder that writes what the
real tool would and hands back a `FakeProcess`. It has no `pid` on purpose: code
under test that tried to signal it would fail loudly instead of signalling a real
process group.
"""

from __future__ import annotations

# `run_logged` waits in slices of at most `_INTERRUPT_POLL_S` and never past the
# timeout. A test that sets the poll to this long makes each wait it records the poll
# itself (no timeout) or what is left of the timeout, so `waits` still shows whether
# a timeout was asked for.
LONG_POLL_S = 1000.0


class FakeProcess:
    """A child that exits with `returncode` as soon as it is waited for; every
    `wait` appends its `timeout` to `waits`."""

    def __init__(self, returncode: int, waits: list[float | None]) -> None:
        self.returncode = returncode
        self._waits = waits

    def wait(self, timeout: float | None = None) -> int:
        self._waits.append(timeout)
        return self.returncode
