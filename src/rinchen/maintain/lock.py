"""The names and versions of `uv.lock`, read with `tomllib` and nothing else (D14, spec M5).

The lock file is TOML: a format `version`, a `revision` and one `[[package]]` table per locked
package, and a name can repeat (a vendored wheel per platform is one entry each). This module
reads the bytes it is handed into the table the environment check and the drift report stand on:
each name in PEP 503 form with its distinct versions. It guesses nothing: a format version other
than 1 is refused, because a newer layout may mean something else by a key, and a version is a
string that is kept and compared as text (`2.14.1+cu130` is a version; it is never parsed).

A `DataError` names the field at fault (`package[3].version`) and never quotes the bytes.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rinchen.errors import DataError

SUPPORTED_FORMAT_VERSION = 1
_SEPARATOR_RUNS = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """The PEP 503 form of a package name: lower case, each run of `-`, `_` and `.` one `-`."""
    return _SEPARATOR_RUNS.sub("-", name).lower()


@dataclass(frozen=True, slots=True)
class LockFile:
    """What a lock file holds that a check needs.

    `entries` counts the `[[package]]` tables, `versions` maps each normalised name to its
    distinct versions in sorted (text) order, and is read-only.
    """

    format_version: int
    revision: int | None
    entries: int
    versions: Mapping[str, tuple[str, ...]]


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text_field(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise DataError(
            f"uv.lock: package[{index}].{key} is missing, empty or not a string"
        )
    return value


def read_lock(data: bytes) -> LockFile:
    """The table of names and versions of the lock file in `data`.

    Raises:
        DataError: the bytes are not UTF-8 or not TOML, the format `version` is not the integer
            1, the `revision` is not an integer, `package` is missing or not a list, or an entry
            is not a table or has no non-empty string `name` or `version`.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError("uv.lock: not UTF-8 text") from error
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise DataError("uv.lock: not valid TOML") from error

    format_version = document.get("version")
    if not _is_integer(format_version) or format_version != SUPPORTED_FORMAT_VERSION:
        raise DataError(
            f"uv.lock: version is not the integer {SUPPORTED_FORMAT_VERSION}, "
            "the only format this reader knows"
        )
    revision = document.get("revision")
    if revision is not None and not _is_integer(revision):
        raise DataError("uv.lock: revision is not an integer")
    packages = document.get("package")
    if not isinstance(packages, list):
        raise DataError("uv.lock: package is missing or not a list of tables")

    found: dict[str, set[str]] = {}
    for index, entry in enumerate(packages):
        if not isinstance(entry, dict):
            raise DataError(f"uv.lock: package[{index}] is not a table")
        name = normalize_name(_text_field(entry, "name", index))
        found.setdefault(name, set()).add(_text_field(entry, "version", index))

    return LockFile(
        format_version=SUPPORTED_FORMAT_VERSION,
        revision=revision,
        entries=len(packages),
        versions=MappingProxyType(
            {name: tuple(sorted(found[name])) for name in sorted(found)}
        ),
    )
