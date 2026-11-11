"""What `rinchen eval report` writes and prints (D13): `report.json` and `report.md`, the manifest
that ends the commit and the lines on standard output. The order of the steps before it, the
hashing and the exit codes are the command's (`rinchen.cli.eval_report`); the report, the Markdown
and the reader that checks the bytes are `rinchen.eval`'s.

`commit` writes nothing before everything is built. The files the old manifest of this stage
listed go first, each taken by its file name INSIDE the directory (a path in a manifest cannot reach
outside it) and no other file is touched; the old manifest goes before any of them, so that a
failure part-way leaves none that vouches for a mix of old and new files; `report.json`,
`report.md` and, LAST, `manifest.json` follow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

from rinchen.cli.eval_suite_output import options_digest
from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.errors import DataError
from rinchen.eval.report_format import evidence_label
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file
from rinchen.utils.part_file import replace_file

__all__ = [
    "MARKDOWN_FILE",
    "REPORT_FILE",
    "STAGE",
    "Prepared",
    "commit",
    "options_of",
    "prepared",
    "result_lines",
]

STAGE: Final = "eval_report"
REPORT_FILE: Final = "report.json"
MARKDOWN_FILE: Final = "report.md"
_OUTPUTS: Final = "evaluation report"
_PACKAGES: Final = ("sacrebleu", "numpy")


class Prepared(NamedTuple):
    """A report ready to be written: the bytes of the two files, the inputs in order with the
    digests taken before they were read, the digest of the options and the seed the manifest
    stands on, its status and metrics and the lines that are printed."""

    report: bytes
    markdown: bytes
    inputs: list[ManifestFile]
    config_sha256: str
    seed: int
    status: dict[str, str]
    metrics: dict[str, float]
    targets: list[str]


def options_of(*, metrics_sha256: str, baseline: str | None, seed: int) -> str:
    """The digest the manifest's `config_sha256` stands on: this command has no config file of its
    own, so the options that decide the report stand in for one (the SHA-256 of the metrics
    configuration, the baseline asked for and the seed)."""
    return options_digest(
        {
            "metrics_config": metrics_sha256,
            "baseline": "none" if baseline is None else baseline,
            "seed": str(seed),
        }
    )


def prepared(
    *,
    report: Mapping[str, Any],
    report_bytes: bytes,
    markdown: str,
    inputs: list[ManifestFile],
    config_sha256: str,
    seed: int,
) -> Prepared:
    """A report ready to be written: the manifest's status and metrics and the `targets:` lines
    from the report."""
    return Prepared(
        report=report_bytes,
        markdown=markdown.encode("utf-8"),
        inputs=inputs,
        config_sha256=config_sha256,
        seed=seed,
        status={
            "report": "ok",
            "evidence": evidence_label(report["evidence"]["kinds"]),
        },
        metrics={
            section: float(len(report[section]))
            for section in ("results", "comparisons", "systems", "suites")
        },
        targets=[
            f"targets: {target['id']} {target['status']}"
            for target in report["targets"]
        ],
    )


def commit(target: Path, ready: Prepared) -> Path:
    """Write the report into `target` (made when missing) and, last, the manifest; return the
    manifest's path. The files of the old manifest of this stage go first. Exits with code 1 when
    anything cannot be written.

    Raises:
        DataError: the directory holds the manifest of another stage.
    """
    create_out_dir(target, what=_OUTPUTS)
    try:
        old = _listed_by_old_manifest(target)
        begin_commit(target, STAGE)
        for name in old:
            (target / name).unlink(missing_ok=True)
        outputs: list[ManifestFile] = []
        for name, data in (
            (REPORT_FILE, ready.report),
            (MARKDOWN_FILE, ready.markdown),
        ):
            path = target / name
            replace_file(path, data)
            outputs.append(output_file(path, None))
        return write_manifest(
            target,
            stage=STAGE,
            config_sha256=ready.config_sha256,
            seed=ready.seed,
            inputs=ready.inputs,
            outputs=outputs,
            metrics=ready.metrics,
            packages=_PACKAGES,
            status=ready.status,
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)


def _listed_by_old_manifest(target: Path) -> Sequence[str]:
    """The names of the files the old manifest of this stage lists, as plain file names (the last
    part of each listed path). A manifest that cannot be read or is of another stage lists
    nothing; the manifest of another stage is refused by `begin_commit`."""
    try:
        manifest = store.read(target)
    except DataError:
        return []
    if manifest is None or manifest.stage != STAGE:
        return []
    names = [Path(item.path).name for item in manifest.outputs]
    return [name for name in names if name not in ("", ".", "..")]


def result_lines(ready: Prepared, target: Path, manifest_path: Path) -> list[str]:
    """`report:`, `markdown:` and `manifest:` with their paths, then a `targets:` line for every
    target."""
    return [
        f"report: {target / REPORT_FILE}",
        f"markdown: {target / MARKDOWN_FILE}",
        f"manifest: {manifest_path}",
        *ready.targets,
    ]
