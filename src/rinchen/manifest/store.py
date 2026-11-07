"""Read/write a stage's manifest.json."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from rinchen.errors import DataError
from rinchen.manifest.model import Manifest
from rinchen.utils.part_file import unique_part_path


def manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "manifest.json"


def write(stage_dir: Path, manifest: Manifest) -> Path:
    """Write the manifest so that it appears complete or not at all: it is
    written to a unique temporary file beside the target and renamed over it,
    because a stage treats an existing manifest as proof that its outputs are
    done."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(stage_dir)
    part_path = unique_part_path(path)
    try:
        part_path.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)
    return path


def read(stage_dir: Path) -> Manifest | None:
    """The stage's manifest, or None when there is none. Raises DataError,
    naming the file, when it exists but is not valid JSON or not a manifest."""
    path = manifest_path(stage_dir)
    if not path.is_file():
        return None
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as exc:
        raise DataError(f"unreadable manifest {path}: {exc}") from exc
