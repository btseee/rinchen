"""Read/write a stage's manifest.json."""
from __future__ import annotations

from pathlib import Path

from rinchen.manifest.model import Manifest


def manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "manifest.json"


def write(stage_dir: Path, manifest: Manifest) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(stage_dir)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read(stage_dir: Path) -> Manifest | None:
    path = manifest_path(stage_dir)
    if not path.is_file():
        return None
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
