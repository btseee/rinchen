"""The release record as the upload plan reads it (D13, spec T18): `release.json` of a stage, which says
which repository the stage was made for, what the model is called and which files it vouches for with
their sizes and digests. The plan compares the files with it and refuses the stage when they differ.

`read_record` is strict: the schema, the repository id, the model's name and, for every file listed,
exactly a size and a SHA-256. A refusal is a `NotARecord` that names the place (a field) and never what
the field held.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final, NamedTuple

from rinchen.errors import ConfigError
from rinchen.export.hub_files import shown
from rinchen.export.release import RELEASE_SCHEMA, check_repo_id

__all__ = ["Listed", "NotARecord", "read_record"]

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MODEL_NAME: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")


class Listed(NamedTuple):
    """What a release record says: the repository, the model's name and, for each file it vouches
    for, its size and SHA-256."""

    repo_id: str
    name: str
    files: Mapping[str, tuple[int, str]]


class NotARecord(Exception):
    """The document is not a release record; `detail` names the place and no text."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def read_record(data: bytes) -> Listed:
    """The release record in `data`, the bytes of `release.json`.

    Raises:
        NotARecord: it is not JSON, or a field is not what a release record holds.
    """
    try:
        document = json.loads(data)
    except (ValueError, RecursionError) as error:
        raise NotARecord("not a JSON document") from error
    if not isinstance(document, dict):
        raise NotARecord("not a JSON object")
    if document.get("schema") != RELEASE_SCHEMA:
        raise NotARecord("field 'schema'")
    repo_id = document.get("repo_id")
    try:
        check_repo_id(repo_id if isinstance(repo_id, str) else "")
    except ConfigError as error:
        raise NotARecord("field 'repo_id'") from error
    name = document.get("name")
    if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
        raise NotARecord("field 'name'")
    return Listed(str(repo_id), name, _listed_files(document.get("files")))


def _listed_files(files: object) -> dict[str, tuple[int, str]]:
    if not isinstance(files, dict):
        raise NotARecord("field 'files'")
    listed: dict[str, tuple[int, str]] = {}
    for name, entry in files.items():
        place = f"field 'files.{shown(str(name))}'"
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
            raise NotARecord(place)
        size, digest = entry["bytes"], entry["sha256"]
        if type(size) is not int or size < 0:
            raise NotARecord(place)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise NotARecord(place)
        listed[str(name)] = (size, digest)
    return listed
