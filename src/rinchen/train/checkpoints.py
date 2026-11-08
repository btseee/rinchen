"""The checkpoints of one Eole run: which step directories are complete, the identity
of the run, and the record of its launch.

Eole saves a checkpoint into `<model_path>/step_<N>/` and then links the files of the
newest one into `<model_path>` itself, one by one and without a temporary name, so a
kill in the middle of a save leaves a directory that is partly written and root links
that mix two steps (spike, D8). A resume therefore names a step directory that
`is_complete`, never `model_path`.

`is_complete` reads headers, not tensors: a file Eole did not finish writing is cut
short, and the checks catch every cut the spike tried. A flipped bit inside a complete
file is not caught here.

`launch.json` in the run directory is the record of what the run was started with. Its
`identity` is the hash of the effective config without the keys a run may change and
still be the same run (`identity_of`); a launch whose identity differs from the recorded
one would resume with options Eole silently ignores (the optimiser options come from the
checkpoint's own config).

`safetensors` is a dependency of Eole, not one of Rinchen's own: it is imported where it
is used, so importing this module stays light.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rinchen.errors import ConfigError, DataError
from rinchen.utils.part_file import replace_file

WEIGHTS_FILE: Final = "model.00.safetensors"
OPTIMIZER_FILE: Final = "optimizer.pt"
CONFIG_FILE: Final = "config.json"
VOCAB_FILE: Final = "vocab.json"
# The file `write_launch_record` writes into the run directory.
LAUNCH_FILE: Final = "launch.json"
# The `training` keys a run may change and stay the same run: the target, which a
# longer run extends, and the checkpoint a resume continues.
IDENTITY_EXCLUDED_KEYS: Final = ("train_steps", "train_from")

# The files that make a step directory complete, in the order they are checked: Eole
# writes `config.json` last, so a missing or cut one is the first sign of a save that
# did not finish.
_REQUIRED_FILES: Final = (CONFIG_FILE, VOCAB_FILE, OPTIMIZER_FILE, WEIGHTS_FILE)
# [0-9], not \d, which matches the digits of other scripts; no leading zero, so one step
# has one name.
_STEP_NAME: Final = re.compile(r"step_([1-9][0-9]*)")
# What `zipfile` raises for a damaged archive besides `BadZipFile`; ValueError includes
# the UnicodeDecodeError of a damaged name.
_UNREADABLE_ARCHIVE: Final = (ValueError, NotImplementedError, OverflowError, EOFError)
_MAX_RECORD_BYTES: Final = 64 * 1024
_SHA256_HEX: Final = r"^[0-9a-f]{64}$"
_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class StepDirectory:
    """A `step_<N>` directory of a checkpoint directory.

    `step` is N, `path` the directory (the checkpoint directory joined with its name),
    `problem` why it is not complete (`is_complete`) or None when it is.
    """

    step: int
    path: Path
    problem: str | None

    @property
    def complete(self) -> bool:
        return self.problem is None


@dataclass(frozen=True, slots=True)
class ResumePoint:
    """Where a run resumes: `directory` is the newest complete step directory (None when
    there is none) and `incomplete_newer` the incomplete directories above it (all of
    them when `directory` is None), oldest first. Eole overwrites them when the run
    reaches their step again."""

    directory: StepDirectory | None
    incomplete_newer: tuple[StepDirectory, ...]


def step_directories(model_path: Path) -> list[StepDirectory]:
    """The step directories of the checkpoint directory `model_path`, oldest step
    first, each with its completeness (`is_complete`).

    A step directory is a real directory (not a symbolic link) named `step_<N>` with N
    written in digits, positive and without a leading zero. Everything else in
    `model_path` is left out: the links Eole puts there, the SentencePiece artifact,
    `step_`, `step_x`, `step_007`, a file named `step_5`. An absent `model_path` holds
    no step directory.

    Raises:
        DataError: `model_path` exists and is not a directory (a link to nothing
            included), or it cannot be listed.
    """
    return [
        StepDirectory(step, path, is_complete(path))
        for step, path in _listed(model_path)
    ]


def step_of(directory: Path) -> int | None:
    """N when `directory` is named `step_<N>` (N in digits, positive, no leading zero:
    the names `step_directories` lists), else None. Only the name is read."""
    match = _STEP_NAME.fullmatch(directory.name)
    return int(match.group(1)) if match else None


def latest_complete(model_path: Path) -> ResumePoint:
    """The newest complete step directory of `model_path` and the incomplete ones above
    it. Only directories newer than the answer are examined; same errors as
    `step_directories`."""
    skipped: list[StepDirectory] = []
    for step, path in reversed(_listed(model_path)):
        problem = is_complete(path)
        if problem is None:
            return ResumePoint(
                StepDirectory(step, path, None), tuple(reversed(skipped))
            )
        skipped.append(StepDirectory(step, path, problem))
    return ResumePoint(None, tuple(reversed(skipped)))


def is_complete(step_dir: Path) -> str | None:
    """None when `step_dir` holds what Eole writes into a finished checkpoint, else a
    short phrase for the first thing wrong, starting with the name of the file.

    Needed, each a regular file and not a symbolic link: `config.json` and `vocab.json`,
    which parse as JSON objects; `optimizer.pt`, a zip archive whose central directory
    can be read and lists an entry; `model.00.safetensors`, which `safetensors` opens
    (it checks the header against the file size) and which holds a tensor. No tensor
    is loaded and no archive member is read. The SentencePiece artifact is not needed:
    it is written before `config.json`.

    Never raises for a damaged directory: a file that cannot be read is a problem too.
    """
    if not step_dir.is_dir():
        what = "is not a directory" if os.path.lexists(step_dir) else "does not exist"
        return f"{step_dir.name}: {what}"
    for name in _REQUIRED_FILES:
        try:
            problem = _file_problem(step_dir / name)
        except OSError as error:
            problem = f"{name}: cannot be read ({error.strerror or error!r})"
        if problem is not None:
            return problem
    return None


def identity_of(cfg: Mapping[str, Any]) -> str:
    """The SHA-256 (64 hex characters) of the canonical JSON of the effective config
    `cfg` without `training.train_steps` and `training.train_from`: the same for any
    key order, different for a change of any other value. `cfg` is not modified.

    The canonical JSON has sorted keys, no spaces and non-ASCII text as it is.

    Raises:
        ConfigError: `cfg` holds something JSON cannot write (a set, a path, NaN).
    """
    stripped = dict(cfg)
    training = stripped.get("training")
    if isinstance(training, Mapping):
        stripped["training"] = {
            key: value
            for key, value in training.items()
            if key not in IDENTITY_EXCLUDED_KEYS
        }
    try:
        text = json.dumps(
            stripped,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"cannot take the identity of the config, it is not plain data: {error}"
        ) from error
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunIdentity(BaseModel):
    """What `launch.json` records about the first launch of a run.

    `identity` is `identity_of` the effective config, `world_size` and `gpu_ranks` the
    effective GPU settings (they are inside the identity too; here they are for the
    reader), `created` the ISO 8601 time of the launch with its UTC offset. The model
    is strict, frozen and refuses any other field: a record that would not read back as
    written cannot be made.
    """

    model_config = ConfigDict(
        strict=True, frozen=True, extra="forbid", allow_inf_nan=False
    )

    identity: Annotated[str, Field(pattern=_SHA256_HEX)]
    world_size: Annotated[int, Field(ge=1)]
    gpu_ranks: tuple[Annotated[int, Field(ge=0)], ...]
    created: str

    @field_validator("created")
    @classmethod
    def _is_an_iso_time_with_an_offset(cls, value: str) -> str:
        try:
            stamp = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("must be an ISO 8601 time") from None
        if stamp.tzinfo is None:
            raise ValueError("must carry a UTC offset")
        return value


class _LaunchDocument(RunIdentity):
    """The content of `launch.json`: the record and the version of its format."""

    # Not `Literal[1]`: that accepts `true`.
    schema_version: Annotated[int, Field(ge=_SCHEMA_VERSION, le=_SCHEMA_VERSION)]


def write_launch_record(run_dir: Path, record: RunIdentity) -> Path:
    """Write `record` as `run_dir/launch.json`, whole or not at all (a temporary file
    and a rename), replacing any earlier record, and return its path. `run_dir` must
    exist.

    Raises:
        OSError: the file cannot be written.
    """
    path = run_dir / LAUNCH_FILE
    document = {"schema_version": _SCHEMA_VERSION, **record.model_dump(mode="json")}
    replace_file(path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
    return path


def read_launch_record(run_dir: Path) -> RunIdentity | None:
    """The record in `run_dir/launch.json`, or None when there is no such file.

    Raises:
        DataError: the file is not a record `write_launch_record` could have written
            (not JSON, not the schema, a field of another type, an unknown field, a
            wrong schema version, a file over 64 KiB), or it cannot be read.
    """
    path = run_dir / LAUNCH_FILE
    if not os.path.lexists(path):
        return None
    try:
        with path.open("rb") as handle:
            content = handle.read(_MAX_RECORD_BYTES + 1)
    except OSError as error:
        raise DataError(f"cannot read the launch record {path}: {error}") from error
    if len(content) > _MAX_RECORD_BYTES:
        raise DataError(f"{path} is too large to be a launch record")
    try:
        document = _LaunchDocument.model_validate_json(content)
    except ValidationError as error:
        raise DataError(
            f"{path} is not a Rinchen launch record: {_first_problems(error)}"
        ) from error
    return RunIdentity.model_validate(document.model_dump(exclude={"schema_version"}))


def _first_problems(error: ValidationError) -> str:
    # Field and reason only: the refused values stay out of the message.
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'the file'}: {item['msg']}"
        for item in error.errors()[:3]
    )


def _listed(model_path: Path) -> list[tuple[int, Path]]:
    if not model_path.is_dir():
        if os.path.lexists(model_path):
            raise DataError(
                f"the checkpoint directory {model_path} is not a directory "
                "(a file, or a link to nothing)"
            )
        return []
    found: list[tuple[int, Path]] = []
    try:
        with os.scandir(model_path) as entries:
            for entry in entries:
                path = model_path / entry.name
                step = step_of(path)
                if step is not None and entry.is_dir(follow_symlinks=False):
                    found.append((step, path))
    except OSError as error:
        raise DataError(
            f"cannot list the checkpoint directory {model_path}: {error}"
        ) from error
    return sorted(found)


def _file_problem(path: Path) -> str | None:
    name = path.name
    if path.is_symlink():
        return f"{name}: is a symbolic link, not a file Eole wrote"
    if not os.path.lexists(path):
        return f"{name}: is missing"
    if not path.is_file():
        return f"{name}: is not a regular file"
    if path.stat().st_size == 0:
        return f"{name}: is empty"
    if name == WEIGHTS_FILE:
        return _weights_problem(path)
    if name == OPTIMIZER_FILE:
        return _archive_problem(path)
    return _json_object_problem(path)


def _json_object_problem(path: Path) -> str | None:
    try:
        parsed = json.loads(path.read_bytes())
    except (ValueError, RecursionError) as error:
        return f"{path.name}: is not valid JSON ({error})"
    if not isinstance(parsed, dict):
        return f"{path.name}: is not a JSON object"
    return None


def _archive_problem(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                return f"{path.name}: is a zip archive with no entry"
    except zipfile.BadZipFile as error:
        return f"{path.name}: is not a complete zip archive ({error})"
    except _UNREADABLE_ARCHIVE as error:
        # Damage the reader trips over without calling it a bad zip: a name that is not
        # UTF-8 (a ValueError), a version it does not know, offsets or a record that
        # do not add up.
        return f"{path.name}: is not a readable zip archive ({type(error).__name__}: {error})"
    return None


def _weights_problem(path: Path) -> str | None:
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="numpy") as tensors:
            if not list(tensors.keys()):
                return f"{path.name}: holds no tensor"
    except SafetensorError as error:
        return f"{path.name}: is not a complete safetensors file ({error})"
    return None
