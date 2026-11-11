"""`rinchen release stage`: build the directory a release uploads, whole (D13). Parsing, the order of the
steps, the hashing and the exit codes only; the card and the release record are `rinchen.export`'s, the
report directory that is hashed and read is `rinchen.cli.release_stage_inputs`' and the files written and
the manifest are `rinchen.cli.release_stage_output`'s.

    rinchen release stage --export DIR --report-dir DIR --system NAME --repo-id ORG/NAME --out-dir DIR

It writes into `--out-dir` the nine files a release uploads (the export's `model.bin`,
`shared_vocabulary.json`, `config.json`, `spm.model` and `rinchen.json`, then `report.json`, `report.md`,
the card `README.md` of the system and the release record `release.json`) and, last, `manifest.json` of
the stage `release_stage`, which is not uploaded (it holds local paths). The export must be one our
manifest vouches for and the report must be the one of THIS export: it must hold a system `NAME` of
kind `rinchen` whose digests of the model and of its manifest are the export's, and a result of it.

The order: the options are checked; `verify_export` vouches for the export (the five files and the
manifest are hashed first); the manifest, `report.json` and `report.md` of the report directory are hashed
BEFORE any is read and the manifest must list both files with the digests found; `report.json` is parsed
from the bytes that were hashed; the report is bound to the export by their digests; an output directory
that is an input's or holds another stage's manifest is refused; only then are the files built (the export's
bytes are read again and must be the ones that were hashed), written and read back, and the manifest written
last. No text of a file is ever in a message of this command.

Exit codes: 0; 2 for a bad option, an export that is not vouched for, an output directory that is an
input's and a file that changed after it was hashed; 3 for a report directory that is missing, not what its
manifest vouches for or not a report, a report of another system or another model, and another stage's
manifest in the output directory; 1 when the output cannot be written or a copy differs from its source;
130 when interrupted. Nothing was run on a GPU, and nothing here touches one.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Annotated

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.release_stage_inputs import (
    HashedReport,
    LoadedReport,
    hash_report_dir,
    read_report_dir,
)
from rinchen.cli.release_stage_output import (
    STAGE,
    Prepared,
    commit,
    prepared,
    result_lines,
)
from rinchen.cli.train_common import require_out_apart_from_inputs
from rinchen.eval.report_format import evidence_label
from rinchen.export.model_card import FileFacts, render_card
from rinchen.export.record import EXPORT_FILES
from rinchen.export.release import (
    check_repo_id,
    check_system_name,
    release_record,
    release_to_bytes,
    require_report_for_export,
)
from rinchen.export.verified import VerifiedExport, verified_bytes, verify_export
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import refuse_foreign_manifest

_MANIFEST_FILE = "manifest.json"


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def stage_command(
    export: Annotated[
        Path,
        typer.Option(
            "--export", help="The export directory `rinchen export convert` wrote."
        ),
    ],
    report_dir: Annotated[
        Path,
        typer.Option(
            "--report-dir",
            help="The directory `rinchen eval report` wrote: manifest.json, report.json and report.md.",
        ),
    ],
    system: Annotated[
        str,
        typer.Option(
            "--system",
            help="The system of the report that is this export; the card describes it.",
        ),
    ],
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", help="The repository the release is for, ORG/NAME."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the nine files of the release and manifest.json (not uploaded).",
        ),
    ],
) -> None:
    """Stage a release: the export's files, the report, the model card and the release record.

    The report must be the one of this export: a system of the kind rinchen with the export's digests. Nothing is uploaded."""
    with rinchen_errors_as_exit_codes():
        target = Path(os.path.abspath(out_dir))
        ready = _build(
            Path(os.path.abspath(export)),
            Path(os.path.abspath(report_dir)),
            check_system_name(system),
            check_repo_id(repo_id),
            target,
        )
        manifest_path = commit(target, ready)
        for line in result_lines(ready, target, manifest_path):
            typer.echo(line)


# ---- the steps -------------------------------------------------------------------------------------


def _build(
    export: Path, report_dir: Path, system: str, repo_id: str, target: Path
) -> Prepared:
    """Everything up to the commit: vouch, hash, read, bind and build the bytes. Nothing is written
    and no directory is made."""
    verified = verify_export(export)
    hashed = hash_report_dir(report_dir)
    loaded = read_report_dir(hashed)
    require_report_for_export(loaded.view, system, verified)
    inputs = _inputs(verified, hashed)
    require_out_apart_from_inputs(
        target, [Path(entry.path) for entry in inputs], option="--out-dir"
    )
    refuse_foreign_manifest(target, STAGE)
    return prepared(
        files=_files_of(verified, loaded, system, repo_id),
        inputs=inputs,
        repo_id=repo_id,
        system=system,
        evidence=evidence_label(loaded.view.evidence.kinds),
    )


def _inputs(verified: VerifiedExport, hashed: HashedReport) -> list[ManifestFile]:
    """Every file that was hashed, in the order they are recorded: the export's five files and its
    manifest (with the digests `verify_export` took), then the report directory's manifest and the
    two reports."""
    export = [
        ManifestFile(
            path=str(verified.directory / name),
            sha256=verified.sha256[name],
            bytes=(verified.directory / name).stat().st_size,
        )
        for name in EXPORT_FILES
    ]
    manifest = verified.directory / _MANIFEST_FILE
    return [
        *export,
        ManifestFile(
            path=str(manifest),
            sha256=verified.manifest_sha256,
            bytes=manifest.stat().st_size,
        ),
        hashed.manifest,
        hashed.report,
        hashed.markdown,
    ]


def _files_of(
    verified: VerifiedExport, loaded: LoadedReport, system: str, repo_id: str
) -> dict[str, bytes]:
    """The bytes of the nine files in the order they are written: the export's five (the bytes that
    were hashed), the two reports (as hashed), the card and the release record."""
    files = {name: verified_bytes(verified, name) for name in EXPORT_FILES}
    files["report.json"] = loaded.report_bytes
    files["report.md"] = loaded.markdown_bytes
    report_sha256 = _digest(files["report.json"])
    facts = [FileFacts(name, len(data), _digest(data)) for name, data in files.items()]
    card = render_card(
        verified.record,
        loaded.view,
        system,
        facts,
        repo_id,
        report_sha256=report_sha256,
        export_manifest_sha256=verified.manifest_sha256,
    ).encode("utf-8")
    files["README.md"] = card
    facts.append(FileFacts("README.md", len(card), _digest(card)))
    files["release.json"] = release_to_bytes(
        release_record(
            verified.record,
            loaded.view,
            system,
            facts,
            repo_id,
            model_sha256=verified.sha256["model.bin"],
            export_manifest_sha256=verified.manifest_sha256,
            report_sha256=report_sha256,
        )
    )
    return files


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
