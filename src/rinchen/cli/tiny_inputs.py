"""The inputs of `rinchen tiny run` (D15, spec P11 and P13): the pipeline file of the profile, the seven
fixtures, the Eole base config, the serve config, the converter's words file, the profile's own file (it
holds the seed) and `uv.lock`, each hashed BEFORE it is read, and the config, the lock, the seed and the
fixtures parsed from the bytes that were hashed (`verified_content`), so that a file that changed in between
is refused and never read.

The pipeline file is the one file read before the rest is hashed, because it says where the rest are; it is
hashed first and read through `verified_content`. A fixture that is missing, cannot be read or whose digest
differs from its pin is a `DataError` (exit 3), and so is a lock that is missing or is not one; a file that
changed after it was hashed, a missing or invalid pipeline file, a missing file the config names and a
profile file with no seed are a `ConfigError` (exit 2).

The Eole base config, the serve config and the words file are hashed for the manifest and are read by the
commands and stages that use them, later: a change after the hashing is not seen for those three. The
registry the fetch stage reads is not hashed here; the stage checks every copy it makes against its pin.
An error names the file or the field and never quotes a file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NamedTuple

import yaml

from rinchen.cli.train_common import hash_input
from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import TinyConfig, config_path_for, load_tiny_config
from rinchen.errors import ConfigError, DataError
from rinchen.export.verified import verified_content
from rinchen.maintain.lock import LockFile, read_lock
from rinchen.manifest.model import ManifestFile
from rinchen.tiny.fixtures import FIXTURE_FILES, Fixtures, read_fixtures

# Where the profile files (the seed) and the lock are; the tests point them at copies.
PROFILES_DIRECTORY: Final = REPO_ROOT / "configs" / "profiles"
LOCK_PATH: Final = REPO_ROOT / "uv.lock"
RUNBOOK: Final = "docs/runbooks/15_tiny_pipeline.md"


class HashedInputs(NamedTuple):
    """The inputs with the digest taken before anything of the run read them, and the pipeline file parsed
    from the bytes of that digest."""

    config_file: ManifestFile
    config: TinyConfig
    fixtures: Mapping[str, ManifestFile]
    base: ManifestFile
    serve: ManifestFile
    words: ManifestFile
    seed_file: ManifestFile
    lock: ManifestFile

    def entries(self) -> list[ManifestFile]:
        """Every file hashed, in the order of the manifest: the pipeline file, the seven fixtures, the Eole
        base config, the serve config, the words file, the profile file and the lock."""
        return [
            self.config_file,
            *self.fixtures.values(),
            self.base,
            self.serve,
            self.words,
            self.seed_file,
            self.lock,
        ]


class ReadInputs(NamedTuple):
    """What the run is given, parsed from the bytes that were hashed."""

    config: TinyConfig
    fixtures: Fixtures
    lock: LockFile
    lock_bytes: bytes
    seed: int


def absolute(path: Path) -> Path:
    """`path` made absolute and normalised, without resolving a link."""
    return Path(os.path.abspath(path))


def pipeline_config_path(profile: str, config: Path | None) -> Path:
    """The pipeline file of a run: `config` when it is given, else `configs/tiny/<profile>.yaml`.

    Raises:
        ConfigError: the profile is not a name of the right shape, or has no pipeline file and no
            `config` was given (the message names the profile and its file).
    """
    shipped = config_path_for(profile)
    if config is not None:
        return config
    if not shipped.is_file():
        relative = shipped.relative_to(REPO_ROOT).as_posix()
        raise ConfigError(
            f"no pipeline configuration for the profile {profile!r} ({relative}): "
            "`rinchen tiny run` runs the profiles that have one, and a larger profile is run "
            f"stage by stage ({RUNBOOK})"
        )
    return shipped


def _hash_fixtures(config: TinyConfig) -> dict[str, ManifestFile]:
    found: dict[str, ManifestFile] = {}
    for name in FIXTURE_FILES:
        entry = hash_input(
            config.fixtures.directory / name, what=f"fixture {name}", missing=DataError
        )
        if entry.sha256 != config.fixtures.sha256[name]:
            raise DataError(
                f"fixture {name}: the SHA-256 differs from its pin in the profile"
            )
        found[name] = entry
    return found


def hash_inputs(profile: str, config_file: Path) -> HashedInputs:
    """Hash the pipeline file, read it once through `verified_content`, then hash everything it and the
    profile name: the fixtures (each against its pin), the Eole base config, the serve config, the words
    file, the profile file and the lock. Nothing else is read.

    Raises:
        ConfigError: the profile name is not of the right shape, the pipeline file is missing, changed
            after it was hashed or is not valid, or another file it names is missing.
        DataError: a fixture is missing, cannot be read or is not its pin, or `uv.lock` is missing.
    """
    config_path_for(profile)
    entry = hash_input(config_file, what="tiny config", missing=ConfigError)
    config = load_tiny_config(verified_content(Path(entry.path), entry.sha256))
    fixtures = _hash_fixtures(config)
    return HashedInputs(
        config_file=entry,
        config=config,
        fixtures=fixtures,
        base=hash_input(
            config.train.base, what="Eole base config", missing=ConfigError
        ),
        serve=hash_input(config.serve.config, what="serve config", missing=ConfigError),
        words=hash_input(
            config.converter.pairs, what="converter words file", missing=ConfigError
        ),
        seed_file=hash_input(
            PROFILES_DIRECTORY / f"{profile}.yaml",
            what="profile file",
            missing=ConfigError,
        ),
        lock=hash_input(LOCK_PATH, what="uv.lock", missing=DataError),
    )


def _seed(data: bytes, name: str) -> int:
    """The `seed` of a profile file: a non-negative integer, as the matrix of the training takes it."""
    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{name}: not valid YAML, so it holds no seed") from error
    seed = document.get("seed") if isinstance(document, dict) else None
    if type(seed) is not int or seed < 0:
        raise ConfigError(f"{name}: has no seed that is a non-negative integer")
    return seed


def _fixtures(hashed: HashedInputs) -> Fixtures:
    pins = {name: entry.sha256 for name, entry in hashed.fixtures.items()}
    try:
        return read_fixtures(hashed.config.fixtures.directory, pins)
    except DataError as error:
        raise ConfigError(
            f"a fixture changed after it was hashed or cannot be read: {error}"
        ) from error


def read_inputs(hashed: HashedInputs) -> ReadInputs:
    """Parse the lock, the seed and the fixtures from the bytes that were hashed.

    Raises:
        ConfigError: a file changed after it was hashed (the fixtures included), or the profile file
            has no seed that is a non-negative integer.
        DataError: `uv.lock` is not a lock file this reader accepts, or cannot be read.
    """
    lock_bytes = verified_content(Path(hashed.lock.path), hashed.lock.sha256)
    lock = read_lock(lock_bytes)
    seed_file = Path(hashed.seed_file.path)
    seed = _seed(verified_content(seed_file, hashed.seed_file.sha256), seed_file.name)
    return ReadInputs(hashed.config, _fixtures(hashed), lock, lock_bytes, seed)
