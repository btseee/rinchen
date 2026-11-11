"""A fabricated release for the tests of the model card and of `rinchen release stage` (D13, Task 9).

`fabricated_release(tmp_path)` is what the stage reads, made without a model: a fabricated export
that `verify_export` vouches for (`write_model_export`, whose `rinchen.json` names the six
directions of D12's toy: `eng_Latn-khk_Cyrl`, `khk_Cyrl-eng_Latn`, `khk_Cyrl-rus_Cyrl`,
`mn_Mong-eng_Latn`, `mn_Mong-rus_Cyrl` and `rus_Cyrl-khk_Cyrl`), the golden report of Task 8 in which
`alpha` is a system of kind `rinchen` carrying that export's digests, written into a report
directory as `rinchen eval report` leaves one (`report.json`, `report.md` and the manifest of stage
`eval_report`), and the directory the stage will be written to (`stage`, which does not exist
until the command made it: `fabricated_release(tmp_path, staged=True)` runs `rinchen release
stage` first, so that a later task has a stage the real command wrote).

Nothing here loads a model, imports `ctranslate2`, opens a socket or touches a GPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from click.testing import Result

from rinchen.eval.report_view import ReportView
from rinchen.export.model_card import FileFacts, render_card
from rinchen.export.record import ModelRecord
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from tests.support.eval_cli import invoke
from tests.support.model_exports import WrittenModel, write_model_export
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_FILE_SIZES,
    GOLDEN_MODEL_SHA256,
    GOLDEN_REPORT_SHA256,
    REPO_ID,
    SYSTEM,
    bound_document,
    markdown_bytes,
    report_bytes,
    view_of,
)

# The six directions of D12's toy export, sorted as a record holds them.
SIX_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
    "mn_Mong-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
REPORT_STAGE: Final = "eval_report"
MANIFEST: Final = "manifest.json"
_NOW: Final = "2026-11-10T10:00:00+00:00"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Release:
    """What `rinchen release stage` reads and where it writes: the export, the report directory
    (its documents and bytes), the system of the report that is the export, the repository id and
    the directory of the stage."""

    root: Path
    export: WrittenModel
    report_dir: Path
    document: dict[str, Any]
    report_bytes: bytes
    markdown_bytes: bytes
    system: str
    repo_id: str
    stage: Path

    @property
    def export_dir(self) -> Path:
        return self.export.directory

    @property
    def model_sha256(self) -> str:
        return self.export.sha256["model.bin"]

    @property
    def manifest_sha256(self) -> str:
        return self.export.manifest_sha256

    @property
    def view(self) -> ReportView:
        return view_of(self.document)

    def args(
        self,
        *extra: str,
        export: Path | None = None,
        report_dir: Path | None = None,
        system: str | None = None,
        repo_id: str | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """`release stage --export E --report-dir R --system S --repo-id I --out-dir O` and
        `extra`, each option the release's own unless it is given."""
        return [
            "release",
            "stage",
            "--export",
            str(self.export_dir if export is None else export),
            "--report-dir",
            str(self.report_dir if report_dir is None else report_dir),
            "--system",
            self.system if system is None else system,
            "--repo-id",
            self.repo_id if repo_id is None else repo_id,
            "--out-dir",
            str(self.stage if out_dir is None else out_dir),
            *extra,
        ]


def write_report_dir(
    directory: Path,
    report: bytes,
    markdown: bytes,
    *,
    stage: str = REPORT_STAGE,
    listed: Sequence[str] = ("report.json", "report.md"),
    status: Mapping[str, str] | None = None,
) -> Path:
    """A report directory as `rinchen eval report` leaves it: `report.json`, `report.md` and the
    manifest of `stage` listing the files named in `listed` with the digests of the bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    contents = {"report.json": report, "report.md": markdown}
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    store.write(
        directory,
        Manifest(
            stage=stage,
            created=_NOW,
            code_version="0000000",
            config_sha256="0" * 64,
            seed=20261110,
            inputs=[],
            outputs=[
                ManifestFile(
                    path=str(directory / name),
                    sha256=sha256_of(contents[name]),
                    bytes=len(contents[name]),
                )
                for name in listed
            ],
            metrics={},
            tool_versions={},
            status=dict(
                {"report": "ok", "evidence": "fixture"} if status is None else status
            ),
        ),
    )
    return directory


def another_report_dir(
    release: Release, document: dict[str, Any], name: str, **options: Any
) -> Path:
    """A second report directory (`name` beside the release's) holding `document`."""
    return write_report_dir(
        release.root / name,
        report_bytes(document),
        markdown_bytes(document),
        **options,
    )


def fabricated_release(
    tmp_path: Path,
    *,
    system: str = SYSTEM,
    repo_id: str = REPO_ID,
    staged: bool = False,
    directions: Sequence[str] = SIX_DIRECTIONS,
) -> Release:
    """The export, the report directory and the stage path under `tmp_path`; with `staged` the
    stage is made by `rinchen release stage` (which must exit 0)."""
    export = write_model_export(tmp_path / "export", directions=directions)
    document = bound_document(
        export.sha256["model.bin"], export.manifest_sha256, system=system
    )
    report = report_bytes(document)
    markdown = markdown_bytes(document)
    release = Release(
        root=tmp_path,
        export=export,
        report_dir=write_report_dir(tmp_path / "report", report, markdown),
        document=document,
        report_bytes=report,
        markdown_bytes=markdown,
        system=system,
        repo_id=repo_id,
        stage=tmp_path / "stage",
    )
    if staged:
        made = invoke_stage(release)
        assert made.exit_code == 0, made.output
    return release


def invoke_stage(release: Release, *extra: str, **options: Any) -> Result:
    """Run `rinchen release stage` over `release` in this process."""
    return invoke(release.args(*extra, **options))


def listing(directory: Path) -> list[str]:
    """The names of the files directly in `directory`, sorted."""
    return sorted(path.name for path in directory.iterdir())


def golden_facts() -> list[FileFacts]:
    """The facts of the seven files a card lists (the export's five and the two reports), with
    the sizes of `GOLDEN_FILE_SIZES` and, as digests, the SHA-256 of each file's name: values the
    golden card is rendered with, which no disk gives."""
    return [
        FileFacts(name, size, sha256_of(name.encode("ascii")))
        for name, size in GOLDEN_FILE_SIZES.items()
    ]


def disk_facts(release: Release) -> list[FileFacts]:
    """The facts of the same seven files as they are on the disk of `release`."""
    contents = {
        **{
            name: (release.export_dir / name).read_bytes()
            for name in release.export.sha256
        },
        "report.json": release.report_bytes,
        "report.md": release.markdown_bytes,
    }
    return [
        FileFacts(name, len(contents[name]), sha256_of(contents[name]))
        for name in GOLDEN_FILE_SIZES
    ]


NO_SCORER: Final = (
    "no segment scores were given: no scorer is built (B1 and B2 are open)"
)


def card_of(
    record: ModelRecord,
    document: dict[str, Any] | None = None,
    *,
    system: str = SYSTEM,
    files: list[FileFacts] | None = None,
    repo_id: str = REPO_ID,
) -> str:
    """The card of `system` of `document` (the golden report bound to a fabricated model)."""
    if document is None:
        document = bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)
    return render_card(
        record,
        view_of(document),
        system,
        golden_facts() if files is None else files,
        repo_id,
        report_sha256=GOLDEN_REPORT_SHA256,
        export_manifest_sha256=GOLDEN_EXPORT_MANIFEST_SHA256,
    )
