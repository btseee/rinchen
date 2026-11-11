"""The inputs of `rinchen maintain check` (D14, spec M3 and M13): the options that must agree with
each other, and the three files the check stands on, `pyproject.toml`, `uv.lock` and the
configuration, each hashed BEFORE it is read and parsed from the bytes that were hashed
(`verified_content`), so that a file that changed in between is refused and never read.

Nothing here writes a file or starts a process. An error names the option or the file and never
quotes a file: `pyproject.toml` is only checked to be TOML, the lock is read by
`rinchen.maintain.lock` and the configuration by `rinchen.config.maintain`.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.train_common import hash_input
from rinchen.config.maintain import CADENCES, MaintainConfig, load_maintain_config
from rinchen.errors import ConfigError, DataError
from rinchen.export.verified import verified_content
from rinchen.maintain.check_types import Inputs
from rinchen.maintain.lock import read_lock
from rinchen.maintain.reruns import ExportInputs
from rinchen.manifest.model import ManifestFile

PYPROJECT: Final = "pyproject.toml"
LOCK: Final = "uv.lock"
# A name the operator typed is cut to this many characters in a message.
_NAME_LIMIT: Final = 40


class HashedInputs(NamedTuple):
    """The three input files with the digest taken before anything read them."""

    pyproject: ManifestFile
    lock: ManifestFile
    config: ManifestFile

    def entries(self) -> list[ManifestFile]:
        """The three, in the order the manifest lists them."""
        return [self.pyproject, self.lock, self.config]


def absolute(path: Path) -> Path:
    """`path` made absolute and normalised, without resolving a link (the children run in the
    project's directory, which is not the one a relative path of the operator means)."""
    return Path(os.path.abspath(path))


def require_cadence(cadence: str) -> None:
    """Raise ConfigError unless `cadence` is one of the two."""
    if cadence not in CADENCES:
        raise ConfigError(f"--cadence must be one of: {', '.join(CADENCES)}")


def export_inputs(
    export: Path | None, checkpoint: Path | None, corpus: list[Path] | None
) -> ExportInputs | None:
    """The export, the checkpoint and the corpus files as the parity rerun takes them, or None
    when none was given.

    Raises:
        ConfigError: only some of the three options were given.
    """
    if export is None and checkpoint is None and not corpus:
        return None
    if export is None or checkpoint is None or not corpus:
        raise ConfigError(
            "--export, --checkpoint and --corpus must be given together, or none of them"
        )
    return ExportInputs(
        absolute(export),
        absolute(checkpoint),
        tuple(absolute(path) for path in corpus),
    )


def hash_inputs(project: Path, config: Path) -> HashedInputs:
    """Hash `pyproject.toml`, `uv.lock` and the configuration, reading none of them.

    Raises:
        DataError: the project file or the lock is missing or cannot be read.
        ConfigError: the configuration is not a file.
    """
    return HashedInputs(
        hash_input(project / PYPROJECT, what=PYPROJECT, missing=DataError),
        hash_input(project / LOCK, what=LOCK, missing=DataError),
        hash_input(config, what="maintain config", missing=ConfigError),
    )


def read_inputs(hashed: HashedInputs) -> Inputs:
    """Parse the three files from the bytes that were hashed.

    Raises:
        ConfigError: a file changed after it was hashed, or the configuration is invalid.
        DataError: `pyproject.toml` is not TOML, or the lock is not one this reader accepts.
    """
    _require_toml(
        verified_content(Path(hashed.pyproject.path), hashed.pyproject.sha256)
    )
    lock = read_lock(verified_content(Path(hashed.lock.path), hashed.lock.sha256))
    config = load_maintain_config(
        verified_content(Path(hashed.config.path), hashed.config.sha256)
    )
    return Inputs(
        config=config,
        lock=lock,
        config_sha256=hashed.config.sha256,
        lock_sha256=hashed.lock.sha256,
        pyproject_sha256=hashed.pyproject.sha256,
    )


def _require_toml(data: bytes) -> None:
    try:
        tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise DataError(f"{PYPROJECT}: not valid TOML") from error


def require_known_reruns(config: MaintainConfig, selected: Sequence[str]) -> None:
    """Raise ConfigError naming the first of `selected` that is no rerun of `config`."""
    known = [item.id for item in config.reruns]
    for name in selected:
        if name not in known:
            raise ConfigError(
                f"--rerun {name[:_NAME_LIMIT]!r} names no rerun of the configuration; "
                f"it holds: {', '.join(known)}"
            )
