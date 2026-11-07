"""Stage runner: skip a stage iff inputs+config sha match an existing manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from collections.abc import Sequence
from pathlib import Path

from rinchen.config.loader import REPO_ROOT
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> str:
    """First 7 characters of the checkout's HEAD sha, plus `-dirty` when the
    working tree has uncommitted changes (untracked files included); `unknown`
    when git is unavailable or the repository has no commit."""
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
    except OSError, subprocess.SubprocessError:
        return "unknown"
    return head[:7] + ("-dirty" if dirty else "")


def tool_versions(packages: Sequence[str]) -> dict[str, str]:
    """Installed version of each package; `not installed` for a missing one."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def build_inputs(input_paths: list[Path]) -> list[ManifestFile]:
    return [
        ManifestFile(path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size)
        for path in input_paths
    ]


def output_file(path: Path, rows: int | None) -> ManifestFile:
    """The manifest entry of a file a stage has written; `rows` is its record
    count when it is a table, else None."""
    return ManifestFile(
        path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size, rows=rows
    )


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
