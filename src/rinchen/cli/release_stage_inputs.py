"""The report `rinchen release stage` stands on (D13): the report directory that `rinchen eval report`
wrote, its manifest, `report.json` and `report.md` hashed one by one BEFORE any is read, and what is
parsed is what was hashed (`verified_content`).

The manifest must be that of a finished `eval_report` stage and must list `report.json` and
`report.md`, each once, with the digests they have now: a report that changed after the command that
wrote it, or that nothing vouches for, is refused. `report.json` is then parsed from the bytes that
were hashed by the strict reader (`report_from_bytes`), so the card describes what the manifest vouches
for. `report.md` is not parsed: it is copied as it was hashed. No text of a file is in a message.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_report_output import MARKDOWN_FILE, REPORT_FILE
from rinchen.cli.eval_report_output import STAGE as REPORT_STAGE
from rinchen.cli.train_common import hash_input
from rinchen.errors import DataError
from rinchen.eval.report_read import ReportView, report_from_bytes
from rinchen.export.verified import verified_content
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile

__all__ = ["HashedReport", "LoadedReport", "hash_report_dir", "read_report_dir"]

_MANIFEST_FILE: Final = store.manifest_path(Path()).name
_MANIFEST: Final = "report manifest"
_REPORT: Final = "report file"
_STATUS_KEY: Final = "report"


class HashedReport(NamedTuple):
    """The report directory: where it is and its manifest, `report.json` and `report.md`, each
    with the digest taken before it was read."""

    directory: Path
    manifest: ManifestFile
    report: ManifestFile
    markdown: ManifestFile


class LoadedReport(NamedTuple):
    """The report that the manifest vouches for: the parsed view of `report.json` and the bytes of
    the two files as they were hashed."""

    view: ReportView
    report_bytes: bytes
    markdown_bytes: bytes


def hash_report_dir(directory: Path) -> HashedReport:
    """Hash the manifest, `report.json` and `report.md` of the report directory, reading none.

    Raises:
        DataError: one of the three is not a file or cannot be read.
    """
    return HashedReport(
        directory,
        hash_input(directory / _MANIFEST_FILE, what=_MANIFEST, missing=DataError),
        hash_input(directory / REPORT_FILE, what=_REPORT, missing=DataError),
        hash_input(directory / MARKDOWN_FILE, what=_REPORT, missing=DataError),
    )


def read_report_dir(hashed: HashedReport) -> LoadedReport:
    """The report of the directory, once its manifest is found to vouch for both files as they are.

    Raises:
        DataError: the manifest is not that of a finished `eval_report` stage, does not list a
            file once with the digest it has now, or `report.json` is not a report.
        ConfigError: a file changed after it was hashed.
    """
    directory = hashed.directory
    manifest = _manifest_of(
        verified_content(Path(hashed.manifest.path), hashed.manifest.sha256), directory
    )
    _require_a_finished_report(manifest, directory)
    listed = _listed_digests(manifest)
    for entry in (hashed.report, hashed.markdown):
        name = Path(entry.path).name
        if listed.get(name) != entry.sha256:
            raise DataError(
                f"the manifest of the report directory {directory} does not vouch for {name} "
                "as it is now: it is not listed once, or it changed after the report that "
                "wrote it"
            )
    report_bytes = verified_content(Path(hashed.report.path), hashed.report.sha256)
    return LoadedReport(
        report_from_bytes(report_bytes),
        report_bytes,
        verified_content(Path(hashed.markdown.path), hashed.markdown.sha256),
    )


def _manifest_of(data: bytes, directory: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(data)
    except ValueError as error:
        raise DataError(
            f"the manifest of the report directory {directory} is not a manifest of a stage"
        ) from error


def _require_a_finished_report(manifest: Manifest, directory: Path) -> None:
    if manifest.stage != REPORT_STAGE:
        raise DataError(
            f"the manifest of the report directory {directory} is not the manifest of an "
            f"{REPORT_STAGE} stage"
        )
    if manifest.status.get(_STATUS_KEY) != "ok":
        raise DataError(
            f"the manifest of the report directory {directory} does not say that the report "
            f"finished ({_STATUS_KEY}: ok)"
        )


def _listed_digests(manifest: Manifest) -> dict[str, str]:
    """The digest the manifest lists for each file name; a name listed twice is not listed (it
    does not say which file it means)."""
    names = Counter(Path(item.path).name for item in manifest.outputs)
    return {
        Path(item.path).name: item.sha256
        for item in manifest.outputs
        if names[Path(item.path).name] == 1
    }
