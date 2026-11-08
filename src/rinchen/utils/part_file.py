"""Writing a file so that it appears whole: a temporary name beside it, then a rename
over it."""

from __future__ import annotations

import uuid
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
