"""Stage runner: skip a stage iff inputs+config sha match an existing manifest,
and the commit protocol of a stage command: the manifest is written last."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from collections.abc import Sequence
from pathlib import Path

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.utils.logging import get as get_logger


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> str:
    """First 7 characters of the checkout's HEAD sha, plus `-dirty` when the
    working tree has uncommitted changes (untracked files included); `unknown`
    when git is unavailable or the repository has no commit; the reason is
    logged as a warning."""
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError) as error:
        _warn_git_failed(error)
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


def _warn_git_failed(error: OSError | subprocess.SubprocessError) -> None:
    fields = {"error": f"{type(error).__name__}: {error}"}
    if isinstance(error, subprocess.CalledProcessError):
        fields["git_stderr"] = (error.stderr or "").strip()
    get_logger(__name__).warning("code_version_unknown", **fields)


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
    return existing is not None and _records_run(existing, input_paths, config_sha256)


def outputs_current(
    stage_dir: Path,
    stage: str,
    input_paths: list[Path],
    config_sha256: str,
    outputs: list[Path],
    *,
    force: bool = False,
) -> bool:
    """True iff the previous run of `stage` in `stage_dir` can be reused: not
    forced, every file of `outputs` exists, and the manifest there is the one
    of `stage`, lists every one of `outputs`, and records the same input hashes
    and config hash as now. Raises DataError for a manifest that is unreadable,
    but only when every output exists: a rebuild needs no manifest."""
    if force or not all(path.is_file() for path in outputs):
        return False
    existing = store.read(stage_dir)
    if existing is None or existing.stage != stage:
        return False
    listed = {item.path for item in existing.outputs}
    return all(str(path) in listed for path in outputs) and _records_run(
        existing, input_paths, config_sha256
    )


def refuse_foreign_manifest(stage_dir: Path, stage: str) -> None:
    """Raises DataError when `stage_dir` holds the manifest of another stage,
    which `stage` must never replace. No manifest, one of `stage` and one that
    cannot be read pass: an unreadable manifest names no stage to protect, and
    a forced rerun is how the user repairs it."""
    try:
        existing = store.read(stage_dir)
    except DataError:
        return
    if existing is not None and existing.stage != stage:
        raise DataError(
            f"{stage_dir} holds the manifest of stage {existing.stage!r}, not of "
            f"{stage!r}; refusing to erase it. Use another output directory."
        )


def begin_commit(stage_dir: Path, stage: str) -> None:
    """Start writing the outputs of `stage` into `stage_dir`. The manifest is
    the commit marker: it is removed before any output is touched, so a
    failure part-way leaves no manifest that vouches for a mix of old and new
    outputs. Raises DataError, and removes nothing, for the manifest of another
    stage (see `refuse_foreign_manifest`)."""
    refuse_foreign_manifest(stage_dir, stage)
    store.manifest_path(stage_dir).unlink(missing_ok=True)


def _records_run(
    manifest: Manifest, input_paths: list[Path], config_sha256: str
) -> bool:
    if manifest.config_sha256 != config_sha256:
        return False
    recorded_hashes = {item.path: item.sha256 for item in manifest.inputs}
    return all(
        recorded_hashes.get(str(input_path)) == sha256_of(input_path)
        for input_path in input_paths
    )
