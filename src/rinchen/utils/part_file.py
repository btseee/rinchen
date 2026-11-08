"""Writing a text file so that it appears whole: its bytes (one LF-ended line per
item), a temporary name beside it, then a rename over it."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path


def unique_part_path(path: Path) -> Path:
    """A `.part` path beside `path` that no other writer will pick, so that two
    runs into one directory never delete or overwrite each other's file. The
    file is not created: its permissions then follow the umask, as an ordinary
    new file's do (a `mkstemp` file would be private to its owner)."""
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")


def replace_file(path: Path, content: bytes) -> None:
    """Write `content` so that `path` appears whole: it is written to a temporary file
    beside it and renamed over it, and a failure leaves any earlier file."""
    part_path = unique_part_path(path)
    try:
        part_path.write_bytes(content)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def lines_as_bytes(lines: Iterable[str]) -> bytes:
    """`lines` as the bytes of a UTF-8 text file with one line each, every line ended
    by an LF (other line separators stay inside their line). The lines are expected
    not to be empty in number: no line gives a single empty one."""
    return ("\n".join(lines) + "\n").encode("utf-8")
