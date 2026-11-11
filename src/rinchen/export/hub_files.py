"""The files of a stage as the upload plan reads them (D13, spec T18): which directory entries are
files of the upload, and each of those read ONCE, in one pass that gives its size, its SHA-256 and, for
a text file, its bytes (to scan them).

A link is never followed and a named pipe is never waited for: a file is opened with `O_NOFOLLOW` and
`O_NONBLOCK` and is read only when the descriptor it was opened as is a regular file (so a path that is
swapped for a link between the listing and the opening is refused, not followed). A problem is a pair
`(name, kind)`: the name of the file and the kind of problem, as a message names them, never a text.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.errors import DataError
from rinchen.export.hub_scan import kinds_in
from rinchen.export.release import STAGE_ONLY, UPLOAD_NAMES

__all__ = [
    "NOT_REGULAR",
    "NOT_UPLOADABLE",
    "TOO_BIG",
    "TOO_BIG_TO_SCAN",
    "UNREADABLE",
    "Problem",
    "Seen",
    "read_hashed",
    "read_stage",
    "shown",
]

# The kinds of problem a message names after a file.
NOT_UPLOADABLE: Final = "not an uploadable name"
NOT_REGULAR: Final = "not a regular file"
TOO_BIG: Final = "over the size limit"
TOO_BIG_TO_SCAN: Final = "too large to scan"
UNREADABLE: Final = "cannot be read"
Problem = tuple[str, str]
_TEXT_SUFFIXES: Final = (".json", ".md")
_CHUNK_BYTES: Final = 1 << 20
_NAME_SHOWN: Final = 80
# What `open` reports for a link that `O_NOFOLLOW` refuses to follow (Linux, other systems).
_LINK_ERRNOS: Final = (errno.ELOOP, errno.EMLINK)


@dataclass(frozen=True, slots=True)
class Seen:
    """A file as it was read: its size, its SHA-256 and, for a file that is read whole, its bytes."""

    size: int
    sha256: str
    data: bytes | None


def shown(name: str) -> str:
    """A file name as a message names it: a name of the upload as it is, any other escaped and cut
    (it is a name on the operator's disk, which may hold anything)."""
    if name in UPLOAD_NAMES or name in STAGE_ONLY:
        return name
    escaped = name.encode("unicode_escape").decode("ascii")
    return escaped if len(escaped) <= _NAME_SHOWN else f"{escaped[:_NAME_SHOWN]}..."


def read_stage(
    stage: Path, *, max_file_bytes: int, max_text_bytes: int
) -> tuple[dict[str, Seen], list[Problem]]:
    """Every entry of the stage directory, classified, and each file of the upload read once.

    A text file (`.json`, `.md`) that is read is scanned: what it holds that looks like a token or a
    local path is a problem of its own. The returned files are the ones that could be read; a file
    that was refused (a link, a pipe, a file over a limit) is named in the problems and is not among
    them.

    Raises:
        DataError: `stage` is not a directory that can be read.
    """
    try:
        entries = sorted(os.scandir(stage), key=lambda entry: entry.name)
    except OSError as error:
        raise DataError(f"{stage} is not a directory that can be read") from error
    seen: dict[str, Seen] = {}
    problems: list[Problem] = []
    for entry in entries:
        name = entry.name
        try:
            status = entry.stat(follow_symlinks=False)
        except OSError:
            problems.append((name, UNREADABLE))
            continue
        if not stat.S_ISREG(status.st_mode):
            problems.append((name, NOT_REGULAR))
        elif name in STAGE_ONLY:
            continue
        elif name not in UPLOAD_NAMES:
            problems.append((name, NOT_UPLOADABLE))
        else:
            _read_upload_file(
                Path(entry.path),
                status.st_size,
                (max_file_bytes, max_text_bytes),
                seen,
                problems,
            )
    return seen, problems


def _read_upload_file(
    path: Path,
    size: int,
    limits: tuple[int, int],
    seen: dict[str, Seen],
    problems: list[Problem],
) -> None:
    max_file_bytes, max_text_bytes = limits
    name = path.name
    text = name.endswith(_TEXT_SUFFIXES)
    if size > max_file_bytes:
        problems.append((name, TOO_BIG))
        return
    if text and size > max_text_bytes:
        problems.append((name, TOO_BIG_TO_SCAN))
        return
    read = read_hashed(
        path, keep=text, limit=max_text_bytes if text else max_file_bytes
    )
    if isinstance(read, str):
        problems.append((name, read))
        return
    seen[name] = read
    if read.data is not None:
        problems += [(name, kind) for kind in kinds_in(read.data)]


def read_hashed(path: Path, *, keep: bool, limit: int) -> Seen | str:
    """The size, SHA-256 and (when `keep`) the bytes of the regular file `path`, read once; or the
    kind of problem that stopped it: not a regular file when it was opened, over `limit` bytes, or
    unreadable. The bytes are kept only to be scanned or parsed."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        return NOT_REGULAR if error.errno in _LINK_ERRNOS else UNREADABLE
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return NOT_REGULAR
            return _digest_of(handle.read, keep=keep, limit=limit)
    except OSError:
        return UNREADABLE


def _digest_of(read: Callable[[int], bytes], *, keep: bool, limit: int) -> Seen | str:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    while chunk := read(_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
        if size > limit:
            return TOO_BIG
        if keep:
            chunks.append(chunk)
    return Seen(size, digest.hexdigest(), b"".join(chunks) if keep else None)
