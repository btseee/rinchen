"""Stage runner: skip a stage iff inputs+config sha match an existing manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path

from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inputs(input_paths: list[Path]) -> list[ManifestFile]:
    return [
        ManifestFile(path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size)
        for path in input_paths
    ]


def should_skip(
    stage_dir: Path,
    input_paths: list[Path],
    config_sha256: str,
    *,
    force: bool = False,
) -> bool:
    """True iff a prior manifest exists with identical input hashes and config hash."""
    if force:
        return False
    existing = store.read(stage_dir)
    if existing is None:
        return False
    if existing.config_sha256 != config_sha256:
        return False
    existing_hashes = {item.path: item.sha256 for item in existing.inputs}
    for input_path in input_paths:
        key = str(input_path)
        if existing_hashes.get(key) != sha256_of(input_path):
            return False
    return True
