"""`rinchen eval report`: turn runs of `rinchen eval run` into one report (D13). Parsing, the order of
the steps, the hashing and the exit codes only; the report, its comparisons, the Markdown and the
reader that checks the bytes are `rinchen.eval`'s, the files that are hashed and read
`rinchen.cli.eval_report_inputs`' and the files written and the manifest
`rinchen.cli.eval_report_output`'s.

    rinchen eval report --run DIR [--run DIR ...] --suite FILE [--suite FILE ...] --out-dir DIR
        [--baseline SYSTEM] [--metrics-config FILE]

It writes `<DIR>/report.json` (schema `rinchen.report/1`), `<DIR>/report.md` and, last,
`<DIR>/manifest.json` (stage `eval_report`). Each `--run` is the directory of one run (one suite, one
system); each `--suite` is the file a run scored, matched to the runs by the SHA-256 of its bytes. The
comparisons are against `--baseline` (the metrics configuration's `baseline_system` when it is not
given) and are RECOMPUTED from the suite and the outputs of the runs; the targets of the
configuration are measured and printed and never end the command with an error. Exit 0 whenever it
measured.

The order: the options are checked; every suite, the metrics configuration and the manifest and
`run.json` of every run are hashed BEFORE any is read; a run of a `dev` suite is refused at once (a
number computed on dev is never reported); each run is read through its manifest and `run.json` and
every file `run.json` names is hashed and checked against its listing; the suites are matched to the
runs; an output directory that is an input's or holds another stage's manifest is refused; only then
are the results and the outputs parsed (from the bytes that were hashed), the report built, written
back through the strict reader as a check, and anything written, the old files first and
`manifest.json` LAST. No text of a suite or of an output is ever in a message of this command.

Exit codes: 0; 2 for a bad option, a baseline that is no system of the runs, a suite that no run
scored, a file that changed after it was hashed, an output directory that is an input's; 3 for an
input that is missing, unreadable or invalid (a run directory without a manifest or with the
manifest of another stage, a manifest that does not vouch for `run.json`, a file that is not the one
`run.json` lists, a suite that was not given), a run of a `dev` suite, two runs of one system over
one suite and another stage's manifest in the output directory; 1 when the output cannot be written.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.eval_report_inputs import (
    HashedFiles,
    LoadedRun,
    hash_inputs,
    read_config,
    read_run_record,
    read_suites,
    run_input,
)
from rinchen.cli.eval_report_output import (
    STAGE,
    Prepared,
    commit,
    options_of,
    prepared,
    result_lines,
)
from rinchen.cli.eval_run_plan import DEFAULT_METRICS_PATH
from rinchen.cli.train_common import require_out_apart_from_inputs
from rinchen.errors import DataError
from rinchen.eval.report import build_report
from rinchen.eval.report_format import report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_read import report_from_bytes
from rinchen.eval.suites import Suite
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import refuse_foreign_manifest


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def report_command(
    run: Annotated[
        list[Path],
        typer.Option(
            "--run",
            help="A run directory `rinchen eval run` wrote (repeatable, at least one).",
        ),
    ],
    suite: Annotated[
        list[Path],
        typer.Option(
            "--suite",
            help="A suite file a run scored (repeatable); matched to the runs by SHA-256.",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: report.json, report.md and manifest.json.",
        ),
    ],
    baseline: Annotated[
        str | None,
        typer.Option(
            "--baseline",
            help="The system the others are compared with. Default: the metrics config's baseline_system.",
        ),
    ] = None,
    metrics_config: Annotated[
        Path,
        typer.Option(
            "--metrics-config",
            help="The metrics, bootstrap settings and targets (YAML).",
        ),
    ] = DEFAULT_METRICS_PATH,
) -> None:
    """Write the report of runs of `rinchen eval run`: scores with intervals, paired comparisons, targets.

    The comparisons are recomputed from the suites and the outputs. It measures and has no threshold: exit 0 whenever it measured."""
    with rinchen_errors_as_exit_codes():
        target = Path(os.path.abspath(out_dir))
        ready = _measure(
            [Path(os.path.abspath(path)) for path in run],
            [Path(os.path.abspath(path)) for path in suite],
            Path(os.path.abspath(metrics_config)),
            baseline,
            target,
        )
        manifest_path = commit(target, ready)
        for line in result_lines(ready, target, manifest_path):
            typer.echo(line)


# ---- the steps -------------------------------------------------------------------------------------


def _measure(
    runs: list[Path],
    suites: list[Path],
    metrics: Path,
    baseline: str | None,
    target: Path,
) -> Prepared:
    """Everything up to the commit: hash, read, check, build and check the bytes. Nothing is
    written and no directory is made."""
    hashed = hash_inputs(suites, metrics, runs)
    config = read_config(hashed.metrics)
    available = read_suites(hashed.suites)
    loaded = [read_run_record(one) for one in hashed.runs]
    _refuse_dev_runs(loaded, available)
    matched = [_suite_of(one, available) for one in loaded]
    inputs = _inputs(hashed, loaded)
    require_out_apart_from_inputs(
        target, [Path(entry.path) for entry in inputs], option="--out-dir"
    )
    refuse_foreign_manifest(target, STAGE)
    report = build_report(
        [run_input(one, suite) for one, suite in zip(loaded, matched, strict=True)],
        available,
        baseline=baseline,
        config=config,
    )
    report_bytes = report_to_bytes(report)
    # What is written must be a report the strict reader accepts.
    report_from_bytes(report_bytes)
    return prepared(
        report=report,
        report_bytes=report_bytes,
        markdown=render_markdown(report),
        inputs=inputs,
        config_sha256=options_of(
            metrics_sha256=hashed.metrics.sha256,
            baseline=baseline,
            seed=config.bootstrap.seed,
        ),
        seed=config.bootstrap.seed,
    )


def _refuse_dev_runs(loaded: list[LoadedRun], suites: dict[str, Suite]) -> None:
    """A run for selection, or over a `dev` suite, is refused before anything else: the numbers
    of a dev suite are for choosing a model and never become a result."""
    for one in loaded:
        suite = suites.get(one.record.suite.sha256)
        if one.record.purpose == "selection" or (
            suite is not None and suite.split == "dev"
        ):
            raise DataError(
                f"the suite {one.record.suite.name!r} is a dev suite: its runs are for "
                "choosing a model and never become a result"
            )


def _suite_of(one: LoadedRun, suites: dict[str, Suite]) -> Suite:
    suite = suites.get(one.record.suite.sha256)
    if suite is None:
        raise DataError(
            f"no --suite file matches the run of the system {one.record.system.name!r} over "
            f"the suite {one.record.suite.name!r}: a suite is matched to a run by the "
            "SHA-256 of its file"
        )
    return suite


def _inputs(hashed: HashedFiles, loaded: list[LoadedRun]) -> list[ManifestFile]:
    """Every file that was hashed, in the order they are recorded: the suites, the metrics
    configuration, then for each run its manifest, its `run.json` and the files it names."""
    return [
        *hashed.suites,
        hashed.metrics,
        *(entry for one in loaded for entry in (*one.inputs, *one.files.values())),
    ]
