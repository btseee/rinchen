"""The reader of an export directory (D11): `verify_export`, and what a reader takes from the
files it vouched for.

`verify_export` is for everything that trusts an export: the engine that translates, the
commands that name it, the parity and the evaluations. It accepts a directory only when

- the manifest is of stage `export` with status `export: ok` and lists each of the five
  files once by file name,
- every file's SHA-256, taken once and BEFORE the file is parsed, is the one the manifest
  lists,
- the record (`rinchen.json`) states the SentencePiece model that is beside it and, when
  `shared_vocabulary.json` is a list of strings, the number of entries it has (a file that
  is anything else is refused by the reader that opens it), and
- every `tags` entry of the record is still the registry's tag of that language: the
  registry changed since the export otherwise, and the export is stale.

A directory that was moved is accepted: the manifest's paths are not compared, the names
are. A directory of the format D10 wrote (`export.json`, no `rinchen.json`) is refused with
the remedy: re-export the checkpoint with `rinchen export convert`.

A reader that opens one of the files afterwards (`verified_bytes`) reads its bytes ONCE and
refuses them unless they are the bytes that were hashed, and parses those bytes: what is
parsed is what was vouched for. The refusal is a guard against a stale or swapped directory,
not against a writer that swaps a file between the read and the use of the bytes, and not
against one that swaps `model.bin` between the hash and CTranslate2's load. Nothing about
the GPU is claimed: no export was loaded on one.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError, DataError
from rinchen.export.record import (
    CT2_FILES,
    EXPORT_FILES,
    EXPORT_STAGE,
    LEGACY_RECORD_FILE,
    RECORD_FILE,
    SPM_FILE,
    ModelRecord,
)
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.manifest.stage import sha256_of

_STATUS_KEY: Final = "export"
_VOCABULARY_FILE: Final = CT2_FILES[1]
_CONFIG_FILE: Final = CT2_FILES[2]
_DIGEST_SHOWN: Final = 12


@dataclass(frozen=True, slots=True)
class VerifiedExport:
    """An export directory `verify_export` found vouched for.

    `directory` is absolute and as given (not resolved). `sha256` maps each name of
    `EXPORT_FILES` to the SHA-256 the file had when it was hashed, once; `manifest_sha256`
    is that of `manifest.json`, taken before it was parsed. `record` is the parsed
    `rinchen.json`.
    """

    directory: Path
    record: ModelRecord
    sha256: Mapping[str, str]
    manifest_sha256: str


def verify_export(directory: Path) -> VerifiedExport:
    """The export in `directory` when its manifest vouches for its five files and its
    record agrees with them (see the module's text); raises when it is anything else.
    Reads the files and writes nothing.

    Raises:
        ConfigError: the directory is not a usable export: it is of the old format, has no
            manifest, the manifest is of another stage or lacks `export: ok`, a file is
            missing, not listed or listed twice, a file's SHA-256 is not the manifest's,
            the record states another SentencePiece model or vocabulary length than the
            files beside it, a tag is not the registry's any more, or a file changed after
            it was hashed. The message names the file and what is wrong.
        DataError: the manifest or the record cannot be read.
    """
    path = Path(os.path.abspath(directory))
    if not os.path.lexists(path):
        raise ConfigError(f"the export directory {path} does not exist")
    if not path.is_dir():
        raise ConfigError(f"the export directory {path} is not a directory")
    _refuse_the_old_format(path)
    manifest_file = store.manifest_path(path)
    if not manifest_file.is_file():
        raise _refuse(
            path,
            f"no manifest vouches for it: {manifest_file} does not exist (`rinchen "
            "export convert` writes it when it finishes)",
        )
    manifest_sha256 = _hashed(manifest_file)
    manifest = _vouching_manifest(path, manifest_file, manifest_sha256)
    listed = _listed_digests(path, manifest, manifest_file)
    for name in EXPORT_FILES:
        _require_a_regular_file(path, path / name)
    digests: dict[str, str] = {}
    for name in EXPORT_FILES:
        digest = _hashed(path / name)
        if digest != listed[name]:
            raise _refuse(
                path,
                f"{path / name} differs from the manifest {manifest_file}: its SHA-256 "
                f"is {digest[:_DIGEST_SHOWN]}... and the manifest lists "
                f"{listed[name][:_DIGEST_SHOWN]}...; the file changed after the "
                "manifest vouched for it",
            )
        digests[name] = digest
    record = _record_in(path / RECORD_FILE, digests[RECORD_FILE])
    if record.spm_sha256 != digests[SPM_FILE]:
        raise _refuse(
            path,
            f"{path / RECORD_FILE} records the SentencePiece model with SHA-256 "
            f"{record.spm_sha256[:_DIGEST_SHOWN]}... but {path / SPM_FILE} has "
            f"{digests[SPM_FILE][:_DIGEST_SHOWN]}...: the export holds another model "
            "than the one it says it was made with",
        )
    _require_the_vocabulary_to_agree(path, record, digests[_VOCABULARY_FILE])
    _require_the_registry_to_agree(path, record)
    return VerifiedExport(
        directory=path,
        record=record,
        sha256=MappingProxyType(digests),
        manifest_sha256=manifest_sha256,
    )


def verified_bytes(export: VerifiedExport, name: str) -> bytes:
    """The bytes of the file `name` of the export, which must be the ones `verify_export`
    hashed.

    Raises:
        DataError: the file cannot be read.
        ConfigError: its SHA-256 is not the verified one (it changed since).
    """
    file = export.directory / name
    try:
        content = file.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error
    if hashlib.sha256(content).hexdigest() != export.sha256[name]:
        raise ConfigError(
            f"{file} changed after the export was verified, so what would be read is "
            "not what was vouched for"
        )
    return content


def vocabulary_entries(export: VerifiedExport) -> tuple[str, ...]:
    """The entries of `shared_vocabulary.json` in the order of the file (the order of the
    ids CTranslate2 maps tokens to), each as the file has it.

    Raises:
        DataError: the file is not valid JSON or not a non-empty list of strings.
        ConfigError: the file changed after the export was verified.
    """
    file = export.directory / _VOCABULARY_FILE
    document = _json_document(verified_bytes(export, _VOCABULARY_FILE), file)
    if (
        not isinstance(document, list)
        or not document
        or not all(isinstance(entry, str) for entry in document)
    ):
        raise DataError(f"{file} must be a non-empty JSON list of strings")
    return tuple(document)


def shared_vocabulary(export: VerifiedExport) -> frozenset[str]:
    """The entries of `shared_vocabulary.json` as a set, the vocabulary CTranslate2 maps
    tokens to (a token outside it silently becomes `<unk>`).

    Raises:
        DataError: the file is not valid JSON or not a non-empty list of strings.
        ConfigError: the file changed after the export was verified.
    """
    return frozenset(vocabulary_entries(export))


def end_token(export: VerifiedExport) -> str:
    """The `eos_token` of the converter's `config.json`: the token a finished hypothesis
    ends with when `return_end_token` is asked for. There is no default.

    Raises:
        DataError: the file is not valid JSON, not an object, or has no non-empty string
            under `eos_token`.
        ConfigError: the file changed after the export was verified.
    """
    file = export.directory / _CONFIG_FILE
    document = _json_document(verified_bytes(export, _CONFIG_FILE), file)
    if not isinstance(document, Mapping):
        raise DataError(f"{file} must hold a JSON object")
    token = document.get("eos_token")
    if not isinstance(token, str) or not token:
        raise DataError(
            f"{file} names no 'eos_token' (a non-empty string), so a finished hypothesis "
            "cannot be told from one that was cut"
        )
    return token


def verified_content(file: Path, digest: str) -> bytes:
    """The bytes of `file`, read once, only when they are the bytes with SHA-256 `digest`:
    what is parsed is what was hashed.

    Raises:
        DataError: the file cannot be read.
        ConfigError: its SHA-256 is not `digest` (it changed after it was hashed).
    """
    try:
        content = file.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error
    if hashlib.sha256(content).hexdigest() != digest:
        raise ConfigError(
            f"{file} changed after it was hashed, so what would be read is not what "
            "was vouched for"
        )
    return content


def json_object(content: bytes, file: Path) -> dict[str, Any]:
    """The JSON object in `content`, the bytes of `file`.

    Raises:
        DataError: the bytes are not JSON or not an object; the message never quotes them.
    """
    try:
        document = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise DataError(f"{file}: is not valid JSON") from error
    if not isinstance(document, dict):
        raise DataError(f"{file}: is not a JSON object")
    return document


def string_list(content: bytes) -> list[str] | None:
    """The JSON list of strings in `content`, None when it is anything else (not JSON,
    nested too deeply, another kind of document, a list holding a non-string)."""
    try:
        document = json.loads(content)
    except ValueError, RecursionError:
        return None
    if isinstance(document, list) and all(isinstance(item, str) for item in document):
        return document
    return None


def _json_document(content: bytes, file: Path) -> object:
    try:
        return json.loads(content)
    except (ValueError, RecursionError) as error:
        # The decoder's message names a position, never the content.
        raise DataError(f"{file} is not valid JSON") from error


# ---- the checks of verify_export ---------------------------------------------------------


def _refuse(path: Path, problem: str) -> ConfigError:
    return ConfigError(f"cannot use the export {path}: {problem}")


def _refuse_the_old_format(path: Path) -> None:
    """A directory D10 wrote has `export.json` and no `rinchen.json`: nothing reads it,
    and the remedy is to export the checkpoint again."""
    if os.path.lexists(path / RECORD_FILE) or not os.path.lexists(
        path / LEGACY_RECORD_FILE
    ):
        return
    raise _refuse(
        path,
        f"it holds {LEGACY_RECORD_FILE}, the record of the format of the distillation "
        f"(D10), and no {RECORD_FILE}: export the checkpoint again with `rinchen export "
        "convert` (a few seconds for a small model)",
    )


def _hashed(file: Path) -> str:
    try:
        return sha256_of(file)
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error


def _vouching_manifest(path: Path, manifest_file: Path, digest: str) -> Manifest:
    content = verified_content(manifest_file, digest)
    try:
        manifest = Manifest.model_validate_json(content)
    except ValueError as error:
        # The reader's message holds the validation errors, which quote the input.
        raise DataError(
            f"{manifest_file}: is not a readable Rinchen manifest"
        ) from error
    if manifest.stage != EXPORT_STAGE:
        raise _refuse(
            path,
            f"{manifest_file} is the manifest of stage {manifest.stage!r}, not of "
            f"{EXPORT_STAGE!r}",
        )
    if manifest.status.get(_STATUS_KEY) != "ok":
        raise _refuse(
            path, f"the status in {manifest_file} does not say {_STATUS_KEY}: ok"
        )
    return manifest


def _listed_digests(
    path: Path, manifest: Manifest, manifest_file: Path
) -> dict[str, str]:
    """The SHA-256 the manifest lists for each file of an export, by file name (the paths
    are those the files had when it was written, and the directory may have moved since)."""
    by_name: dict[str, list[str]] = {}
    for item in manifest.outputs:
        by_name.setdefault(Path(item.path).name, []).append(item.sha256)
    listed: dict[str, str] = {}
    for name in EXPORT_FILES:
        entries = by_name.get(name, [])
        if not entries:
            raise _refuse(path, f"the manifest {manifest_file} does not list {name}")
        if len(entries) > 1:
            raise _refuse(
                path,
                f"the manifest {manifest_file} lists {name} more than once, so it "
                "does not say which file it means",
            )
        listed[name] = entries[0]
    return listed


def _require_a_regular_file(path: Path, file: Path) -> None:
    if not os.path.lexists(file):
        raise _refuse(path, f"{file} is missing")
    if not file.is_file():
        raise _refuse(path, f"{file} is not a regular file")


def _record_in(file: Path, digest: str) -> ModelRecord:
    content = verified_content(file, digest)
    try:
        return ModelRecord.from_bytes(content)
    except DataError as error:
        raise DataError(f"{file}: {error}") from error


def _require_the_vocabulary_to_agree(
    path: Path, record: ModelRecord, digest: str
) -> None:
    """The record states how many entries the vocabulary has; a file of another length
    maps the pieces to the wrong ids. A file that is not a non-empty list of strings is
    left to its reader (`shared_vocabulary`), which refuses it."""
    file = path / _VOCABULARY_FILE
    entries = string_list(verified_content(file, digest))
    if entries and len(entries) != record.vocabulary_entries:
        raise _refuse(
            path,
            f"{path / RECORD_FILE} records {record.vocabulary_entries} vocabulary "
            f"entries but {file} has {len(entries)}: the export holds another "
            "vocabulary than the one it says it was made with",
        )


def _require_the_registry_to_agree(path: Path, record: ModelRecord) -> None:
    """A tag of the record that is not the registry's tag of that language any more: the
    registry changed since the export, and a request would carry a tag the model never saw
    (or none it knows)."""
    for code, tag in sorted(record.tags.items()):
        if direction_tag(code) != tag:
            raise _refuse(
                path,
                f"the registry changed since the export: re-export. {path / RECORD_FILE} "
                f"records another direction tag for {code} than the registry gives now",
            )
