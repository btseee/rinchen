"""The temporary name a writer gives a file before it renames the file into
place."""

from __future__ import annotations

import uuid
from pathlib import Path


def unique_part_path(path: Path) -> Path:
    """A `.part` path beside `path` that no other writer will pick, so that two
    runs into one directory never delete or overwrite each other's file. The
    file is not created: its permissions then follow the umask, as an ordinary
    new file's do (a `mkstemp` file would be private to its owner)."""
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
