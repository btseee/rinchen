"""The lock that lets one launch at a time work in a run directory (D8).

The lock is an exclusive, non-blocking `flock` on a file. A `flock` belongs to the open
file description, not to the process or the path: it ends when the descriptor is closed
or the process dies (a crash or a kill leaves no stale lock to clean up), and two opens
of the file conflict, also inside one process.

The descriptor is not inheritable (`os.open`'s default) and becomes so only through the
explicit `pass_fds` of the one call that is meant to hand it on (`run_logged(...,
inherit_fds=...)`; `holding_the_run` yields its number). `launch` hands it to the Eole
commands it starts: Eole leads a session of its own and survives a SIGKILL of the
launcher, and the lock then stays held by that Eole until it exits. So the next launch is
refused while the orphan trains, instead of starting a second training into the same
checkpoint directory. The holder is Eole's leader (`eole train` itself): the per-rank
processes of a multi-GPU run are started by torch with their descriptors closed and do not
hold it, so the lock lasts as long as the leader, not as long as every rank. A leader that
dies on its own takes the group with it at the launcher (`run_logged`); after a SIGKILL of
the launcher itself nothing does, and a rank that outlives the leader is not covered.

The file is empty and is never removed (removing a lock file while others may open it
breaks the lock). The lock is advisory and local: it stops other callers of this module
on a file system that supports `flock`, not an operator's hand-run script.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rinchen.errors import ConfigError


@contextmanager
def holding_the_run(lock: Path) -> Iterator[int]:
    """Hold the lock file `lock` (created, with its parent directories, when missing)
    for the block, and let go of it however the block ends. It yields the number of the
    lock's descriptor, for `run_logged`'s `inherit_fds`: the descriptor is closed at the
    end of the block, but the lock lasts as long as any process holds a copy.

    Raises:
        ConfigError: another open file description holds the lock; the message names
            `lock`.
        OSError: the file cannot be created or opened.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        _take(descriptor, lock)
        yield descriptor
    finally:
        os.close(descriptor)  # releases the lock, unless a child holds a copy


def refuse_a_held_run(lock: Path) -> None:
    """Raise ConfigError when another open file description holds `lock`. The lock is
    taken and dropped to find out, and a missing file is not created: it means nobody
    holds it. This is a look, not a hold: a launch that starts after it is not stopped.

    Raises:
        ConfigError: the lock is held; the message names `lock`.
        OSError: the file exists and cannot be opened.
    """
    try:
        descriptor = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        _take(descriptor, lock)
    finally:
        os.close(descriptor)


def _take(descriptor: int, lock: Path) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ConfigError(
            f"another launch holds this run: {lock} is locked by a running "
            "`rinchen train launch` of the same run directory or by an Eole process it "
            "started (a launcher that was killed leaves its Eole training: wait for it "
            "to end or stop it; the lock goes away with the process)"
        ) from None
