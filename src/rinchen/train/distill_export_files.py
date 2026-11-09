"""What a reader of a verified export (D10) takes from the files the converter wrote.

`verify_export` hashed the files of the export directory once and returned the digests. A
reader that opens one of them afterwards (the teacher that translates, the engine that
serves) reads its bytes ONCE through `verified_bytes`, which refuses them unless they are
the bytes that were hashed, and parses those bytes: what is parsed is what was vouched
for. The refusal is a guard against a stale or swapped directory, not against a writer
that swaps a file between the read and the use of the bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from rinchen.errors import ConfigError, DataError
from rinchen.train.distill_export import VerifiedExport

VOCABULARY_FILE = "shared_vocabulary.json"
CONFIG_FILE = "config.json"


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


def shared_vocabulary(export: VerifiedExport) -> frozenset[str]:
    """The entries of `shared_vocabulary.json`, the vocabulary CTranslate2 maps tokens to
    (a token outside it silently becomes `<unk>`).

    Raises:
        DataError: the file is not valid JSON or not a non-empty list of strings.
    """
    file = export.directory / VOCABULARY_FILE
    document = _json_document(verified_bytes(export, VOCABULARY_FILE), file)
    if (
        not isinstance(document, list)
        or not document
        or not all(isinstance(entry, str) for entry in document)
    ):
        raise DataError(f"{file} must be a non-empty JSON list of strings")
    return frozenset(document)


def end_token(export: VerifiedExport) -> str:
    """The `eos_token` of the converter's `config.json`: the token a finished hypothesis
    ends with when `return_end_token` is asked for. There is no default.

    Raises:
        DataError: the file is not valid JSON, not an object, or has no non-empty string
            under `eos_token`.
    """
    file = export.directory / CONFIG_FILE
    document = _json_document(verified_bytes(export, CONFIG_FILE), file)
    if not isinstance(document, Mapping):
        raise DataError(f"{file} must hold a JSON object")
    token = document.get("eos_token")
    if not isinstance(token, str) or not token:
        raise DataError(
            f"{file} names no 'eos_token' (a non-empty string), so a finished hypothesis "
            "cannot be told from one that was cut"
        )
    return token


def _json_document(content: bytes, file: Path) -> object:
    try:
        return json.loads(content)
    except ValueError as error:
        # The decoder's message names a position, never the content.
        raise DataError(f"{file} is not valid JSON") from error
