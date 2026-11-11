"""The release record and the checks of `rinchen release stage` (D13, spec T17).

A release is a directory that is uploaded whole: the five files of an export, the card `README.md`, the
report (`report.json`, `report.md`) and the release record `release.json`, nine files in `UPLOAD_NAMES`,
and the manifest `manifest.json` of the stage that made it, which is NOT uploaded (like the export's it
holds the paths of the machine that made it).

`release.json` (schema `rinchen.release/1`) says what the release is and vouches for the files: the
repository, the model's name, version, quantization, tag placement and directions, the digests of the
export (its `model.bin`, its manifest, its SentencePiece model and the checkpoint it came from), the
digest of the report and the system of it the card describes, the evidence the figures rest on and the
size and SHA-256 of every uploaded file but itself. It is sorted JSON with two-space indent and one
final LF, and holds no timestamp (only manifests have one) and no path: it is uploaded.

`require_report_for_export` is the binding that stops a card from describing another model than the
one it ships with: the report must hold a system of that name of kind `rinchen` whose digests of the
model and of its manifest are the export's, and at least one result of it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Final

from rinchen.errors import ConfigError, DataError
from rinchen.eval.report_format import evidence_label
from rinchen.eval.report_markdown_cells import short
from rinchen.eval.report_read import ReportView
from rinchen.eval.robust_suite_rules import is_id
from rinchen.export.model_card import FileFacts
from rinchen.export.record import ModelRecord
from rinchen.export.verified import VerifiedExport

__all__ = [
    "RELEASE_SCHEMA",
    "STAGE_ONLY",
    "UPLOAD_NAMES",
    "check_repo_id",
    "check_system_name",
    "release_record",
    "release_to_bytes",
    "require_report_for_export",
]

RELEASE_SCHEMA: Final = "rinchen.release/1"
RELEASE_FILE: Final = "release.json"
# What a release uploads, in the order the plan lists it; `release.json` lists the others.
UPLOAD_NAMES: Final = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
    "README.md",
    "report.json",
    "report.md",
    RELEASE_FILE,
)
# What the stage writes and the upload leaves out.
STAGE_ONLY: Final = ("manifest.json",)
_MODEL_FILE: Final = "model.bin"
_REPO_PART: Final = r"[A-Za-z0-9._-]{1,96}"
_REPO_ID: Final = re.compile(f"{_REPO_PART}/{_REPO_PART}")
_DOT_PARTS: Final = (".", "..")


def check_repo_id(value: str) -> str:
    """`value` when it is a repository id, `ORG/NAME`: two parts of 1 to 96 characters (letters,
    digits, `.`, `_` and `-`), neither of them `.` or `..`.

    Raises:
        ConfigError: it is anything else; the message does not repeat it.
    """
    if not _REPO_ID.fullmatch(value) or any(
        part in _DOT_PARTS for part in value.split("/")
    ):
        raise ConfigError(
            "--repo-id must be ORG/NAME: two parts of 1 to 96 characters (letters, digits, "
            "'.', '_' and '-'), neither of them '.' or '..'"
        )
    return value


def check_system_name(value: str) -> str:
    """`value` when it is the id of a system of a report.

    Raises:
        ConfigError: it is not 1 to 64 characters of letters, digits, `.`, `_` and `-`.
    """
    if not is_id(value):
        raise ConfigError(
            "--system must be 1 to 64 characters: letters, digits, '.', '_' and '-'"
        )
    return value


def require_report_for_export(
    report: ReportView, system: str, verified: VerifiedExport
) -> None:
    """The report describes this export: it lists `system`, a system of kind `rinchen` whose
    `model_sha256` is the SHA-256 of the export's `model.bin` and whose `manifest_sha256` is that
    of the export's manifest, and it holds a result of it.

    Raises:
        DataError: it does not; the message names the system and, for a digest, the first twelve
            hexadecimal digits of each side.
    """
    entry = next((entry for entry in report.systems if entry.name == system), None)
    if entry is None:
        raise DataError(f"the report has no system {system!r}")
    if entry.kind != "rinchen":
        raise DataError(
            f"the system {system!r} of the report is of the kind {entry.kind}, not a rinchen "
            "export: a card describes a model of ours"
        )
    for what, found, expected in (
        ("model", entry.model_sha256, verified.sha256[_MODEL_FILE]),
        ("manifest", entry.manifest_sha256, verified.manifest_sha256),
    ):
        if found != expected:
            raise DataError(
                f"the system {system!r} of the report was measured on another {what} than "
                f"the export's: the report names {short(found)}... and the export has "
                f"{short(expected)}..."
            )
    if not any(result.system == system for result in report.results):
        raise DataError(f"the report holds no result of the system {system!r}")


def release_record(
    record: ModelRecord,
    report: ReportView,
    system: str,
    files: Sequence[FileFacts],
    repo_id: str,
    *,
    model_sha256: str,
    export_manifest_sha256: str,
    report_sha256: str,
) -> dict[str, Any]:
    """The document `release.json` holds: `files` are the uploaded files but the record itself,
    `model_sha256` and `export_manifest_sha256` the digests of the export's `model.bin` and manifest
    and `report_sha256` that of `report.json`, which describes `system`.

    Raises:
        ValueError: `files` names a file that is not uploaded, is the record itself or is listed
            twice.
    """
    names = [fact.name for fact in files]
    if any(name not in UPLOAD_NAMES or name == RELEASE_FILE for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError(
            "release.json lists the uploaded files other than itself, each once"
        )
    return {
        "schema": RELEASE_SCHEMA,
        "repo_id": repo_id,
        "name": record.name,
        "rinchen_version": record.rinchen_version,
        "quantization": record.quantization,
        "tag_placement": record.tag_placement,
        "directions": list(record.directions),
        "export": {
            "model_sha256": model_sha256,
            "manifest_sha256": export_manifest_sha256,
            "spm_sha256": record.spm_sha256,
            "checkpoint": asdict(record.checkpoint),
        },
        "report": {"sha256": report_sha256, "system": system},
        "evidence": evidence_label(report.evidence.kinds),
        "files": {
            fact.name: {"bytes": fact.bytes, "sha256": fact.sha256} for fact in files
        },
    }


def release_to_bytes(document: Mapping[str, Any]) -> bytes:
    """The UTF-8 bytes of `release.json`: sorted keys, two-space indent, one final LF."""
    text = json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    return f"{text}\n".encode()
