"""`rinchen maintain check`: is this machine the stack the lock file names, what could `uv` take next,
and do the pinned numbers still come out the same (D14). Parsing, the order of the steps, the hashing,
the real seams and the exit codes only; the steps and the report are `rinchen.maintain`'s, the doctor's
CPU checks `rinchen.cli.maintain_doctor`'s, the files written, the manifest and the printed lines
`rinchen.cli.maintain_output`'s and the signals `rinchen.cli.maintain_signals`'.

    rinchen maintain check --out-dir DIR [--cadence nightly|weekly] [--config FILE] [--project-dir DIR]
        [--uv FILE] [--no-drift] [--no-reruns] [--rerun ID ...] [--fail-on-drift]
        [--export DIR --checkpoint DIR --corpus FILE [--corpus FILE ...]]

It writes `<DIR>/maintain_report.json` (schema `rinchen.maintain/1`), `<DIR>/maintain_report.md`, the
logs of the commands it ran (`logs/`, `junit/`) and, last, `<DIR>/manifest.json` (stage
`maintain_check`, which records every package version of the lock). It only reads: it never writes
`pyproject.toml` or `uv.lock` (the dry run is uv's own `--dry-run`, and the configuration cannot
make it anything else), never asks a device question and writes nothing outside `--out-dir`. The
program `uv` is `--uv` or the one on `PATH`; the environment variable `UV` is not read.

The order: the options are checked (`--no-reruns` and `--rerun` contradict each other); `pyproject.toml`,
`uv.lock` and the configuration are hashed BEFORE any is read; an output directory that is an
input's is refused, so is another stage's manifest in it, and so is an existing directory that
holds anything but the names this command writes; the three files are parsed from the bytes that
were hashed (a file that changed in between is refused) and the `--rerun` ids are checked against
the configuration; the code version of the checkout is read; only then is anything written: the
output directory is made, the old manifest and the files it listed are removed, the check runs, the messages of the doctor checks that failed are printed to standard
output as `[FAIL] <id>: <message>` (they reach no file), the version of uv is read for the manifest
and the report, the Markdown and the manifest are written. The command hides every GPU from its own
process for the whole run and puts the caller's setting back when it ends, whatever the exit.

What reaches the screen: the lines of the run, which hold states, counts and closed reasons, and
the `[FAIL] <id>: <message>` lines, the one place where the words of another program can appear. A
doctor message can hold an exception text, a path and the last 500 characters of what a program
printed (the `--help` runs of Eole and CTranslate2, one such tail for each); it is printed and
never written to a file. No text of a file is in any message of this command, and the output of
`uv` and of a rerun is in the logs only.

Exit codes: 0 when the run is `ok` or `drift`; 4 when it is `fail` (a measured result: everything is
written and printed) or when it is `drift` and `--fail-on-drift` was given; 2 for a bad option, an
invalid configuration, `--no-reruns` together with `--rerun`, a `--rerun` that names no rerun of the
configuration, a file that changed after it was hashed and an output directory that is the project's,
an input's or holds entries this command does not write; 3 for a missing or
invalid `pyproject.toml` or `uv.lock` and for the manifest of another stage in the output
directory; 1 when the output cannot be written; 130 when interrupted (SIGTERM and SIGHUP end it
with 143 and 129); an interrupt writes no manifest.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.maintain_doctor import run_cpu_checks
from rinchen.cli.maintain_inputs import (
    HashedInputs,
    absolute,
    export_inputs,
    hash_inputs,
    read_inputs,
    require_cadence,
    require_known_reruns,
    require_rerun_options_agree,
)
from rinchen.cli.maintain_output import (
    STAGE,
    UNKNOWN_VERSION,
    begin,
    commit,
    observe_code_version,
    parse_uv_version,
    require_only_own_entries,
    result_lines,
)
from rinchen.cli.maintain_signals import exit_on_termination_signals
from rinchen.cli.train_common import (
    create_out_dir,
    exit_cannot_write,
    require_out_apart_from_inputs,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.maintain.check import (
    CheckOptions,
    CheckResult,
    CheckSeams,
    DoctorFinding,
    Inputs,
    run_check,
)
from rinchen.maintain.commands import (
    CommandRunner,
    SubprocessRunner,
    child_environment,
    device_hidden,
    read_log,
)
from rinchen.maintain.environment import installed_versions
from rinchen.maintain.report import exit_code_of, report_to_bytes
from rinchen.maintain.report_markdown import render_markdown
from rinchen.maintain.reruns import LOGS_DIR, ExportInputs
from rinchen.manifest.stage import refuse_foreign_manifest

DEFAULT_CONFIG: Final = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
_OUTPUTS: Final = "maintenance report"
_VERSION_LOG: Final = "uv_version.log"
_VERSION_TIMEOUT_SECONDS: Final = 30

app = typer.Typer(no_args_is_help=True)


def make_runner() -> CommandRunner:
    """The runner of every command of a run; the tests of the command replace it."""
    return SubprocessRunner()


# The two other seams of the command, which the tests replace: the doctor's CPU checks and the
# installed versions.
doctor_findings: Callable[[], Sequence[DoctorFinding]] = run_cpu_checks
installed: Callable[[], Mapping[str, Sequence[str]]] = installed_versions


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def check_command(
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the report, its summary, the logs and manifest.json.",
        ),
    ],
    cadence: Annotated[
        str,
        typer.Option(
            "--cadence",
            help="Which reruns are due: `nightly` (the cheap ones) or `weekly` (all of them).",
        ),
    ] = "nightly",
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="The pin rules, the uv argument lists and the reruns (YAML).",
        ),
    ] = DEFAULT_CONFIG,
    project_dir: Annotated[
        Path,
        typer.Option(
            "--project-dir",
            help="The project: its pyproject.toml, its uv.lock and the tests the reruns select.",
        ),
    ] = REPO_ROOT,
    uv: Annotated[
        Path | None,
        typer.Option(
            "--uv",
            help="The uv program, as a path: `--uv uv` means `<cwd>/uv`, not the one on PATH. Default: the one on PATH (the variable UV is not read).",
        ),
    ] = None,
    no_drift: Annotated[
        bool,
        typer.Option(
            "--no-drift", help="Do not run the dependency drift step (it queries uv)."
        ),
    ] = False,
    no_reruns: Annotated[
        bool, typer.Option("--no-reruns", help="Run none of the parity reruns.")
    ] = False,
    rerun: Annotated[
        list[str] | None,
        typer.Option(
            "--rerun",
            help="Run exactly this rerun, whatever the cadence (repeatable; an id of the configuration).",
        ),
    ] = None,
    fail_on_drift: Annotated[
        bool,
        typer.Option(
            "--fail-on-drift", help="Exit 4 when the drift table is not empty."
        ),
    ] = False,
    export: Annotated[
        Path | None,
        typer.Option(
            "--export",
            help="An export directory for the `parity_export` rerun (with --checkpoint and --corpus).",
        ),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="The model directory the export was made from (with --export and --corpus).",
        ),
    ] = None,
    corpus: Annotated[
        list[Path] | None,
        typer.Option(
            "--corpus",
            help="Corpus Parquet file for the `parity_export` rerun; repeat it for more.",
        ),
    ] = None,
) -> None:
    """Check the stack: the doctor, the lock, the installed packages, the dependency drift and the parity reruns.

    Writes maintain_report.json, maintain_report.md and manifest.json, and only reads the project.

    Exit codes: 0 when the run is ok or drift; 4 when it fails (or drifts, with --fail-on-drift); 2 for a bad option or configuration; 3 for a missing or invalid pyproject.toml or uv.lock; 1 when the output cannot be written; 130 when interrupted."""
    with (
        rinchen_errors_as_exit_codes(),
        exit_on_termination_signals(),
        device_hidden(),
    ):
        plan = _plan(
            out_dir=out_dir,
            cadence=cadence,
            config=config,
            project_dir=project_dir,
            uv=uv,
            drift=not no_drift,
            reruns=not no_reruns,
            selected=tuple(rerun or ()),
            export=export_inputs(export, checkpoint, corpus),
        )
        result, uv_version = _run(plan)
        try:
            manifest_path = commit(
                plan.target,
                report_bytes=report_to_bytes(result.report),
                markdown=render_markdown(result.report),
                result=result,
                inputs=plan.hashed.entries(),
                lock=plan.inputs.lock,
                config_sha256=plan.hashed.config.sha256,
                code_version=plan.code_version,
                python_version=platform.python_version(),
                uv_version=uv_version,
            )
        except OSError as error:
            exit_cannot_write(plan.target, error, what=_OUTPUTS)
        for line in result_lines(result.report, plan.target, manifest_path):
            typer.echo(line)
        code = exit_code_of(result.report, fail_on_drift=fail_on_drift)
        if code:
            raise typer.Exit(code=code)


class _Plan(NamedTuple):
    """Everything a run needs once every input is accepted: nothing has been written yet."""

    target: Path
    project: Path
    hashed: HashedInputs
    inputs: Inputs
    options: CheckOptions
    uv: str | None
    code_version: str


def _plan(
    *,
    out_dir: Path,
    cadence: str,
    config: Path,
    project_dir: Path,
    uv: Path | None,
    drift: bool,
    reruns: bool,
    selected: tuple[str, ...],
    export: ExportInputs | None,
) -> _Plan:
    """Check the options and the files, in the order of the module docstring, and write nothing;
    the last thing it does is read the code version, before the output directory is made.

    Raises:
        ConfigError: a bad option, an invalid configuration, a rerun the configuration does not
            hold, a file that changed after it was hashed or an output directory that is an input's.
        DataError: a project file is missing or invalid, or the output directory holds the manifest
            of another stage.
    """
    require_cadence(cadence)
    require_rerun_options_agree(reruns=reruns, selected=selected)
    target = absolute(out_dir)
    project = absolute(project_dir)
    hashed = hash_inputs(project, absolute(config))
    require_out_apart_from_inputs(
        target, [Path(entry.path) for entry in hashed.entries()], option="--out-dir"
    )
    refuse_foreign_manifest(target, STAGE)
    require_only_own_entries(target)
    inputs = read_inputs(hashed)
    require_known_reruns(inputs.config, selected)
    options = CheckOptions(
        cadence=cadence,
        drift=drift,
        reruns=reruns,
        selected=selected,
        export=export,
    )
    return _Plan(
        target,
        project,
        hashed,
        inputs,
        options,
        _uv_program(uv),
        observe_code_version(),
    )


def _run(plan: _Plan) -> tuple[CheckResult, str]:
    """Make the output directory, clear the old run's files, run the check, print the messages of
    the doctor checks that failed and read the version of uv. Exits with code 1 when the directory
    cannot be made or cleared."""
    create_out_dir(plan.target, what=_OUTPUTS)
    try:
        begin(plan.target)
    except OSError as error:
        exit_cannot_write(plan.target, error, what=_OUTPUTS)
    runner = make_runner()
    environment = dict(os.environ)
    result = run_check(
        plan.inputs,
        plan.options,
        CheckSeams(
            runner=runner,
            uv=plan.uv,
            python=sys.executable,
            doctor=doctor_findings,
            installed=installed,
            clock=time.monotonic,
            parent_env=environment,
        ),
        project_dir=plan.project,
        out_dir=plan.target,
    )
    for check_id, message in result.doctor_messages:
        typer.echo(f"[FAIL] {check_id}: {message}")
    version = _uv_version(runner, plan.uv, plan.project, plan.target, environment)
    return result, version


# ---- the steps -------------------------------------------------------------------------------------


def _uv_program(uv: Path | None) -> str | None:
    """The program that runs uv: `--uv` (made absolute, because the commands run in the
    project's directory) or the one on `PATH`; None when there is none."""
    if uv is not None:
        return str(absolute(uv))
    return shutil.which("uv")


def _uv_version(
    runner: CommandRunner,
    program: str | None,
    project: Path,
    out: Path,
    environment: Mapping[str, str],
) -> str:
    """The version of uv for the manifest, or `unknown` when there is no uv, it fails or its
    first line is not a version. Its log is removed again: it is not an output of the run."""
    if program is None:
        return UNKNOWN_VERSION
    log = out / LOGS_DIR / _VERSION_LOG
    try:
        outcome = runner.run(
            [program, "--version"],
            cwd=project,
            env=child_environment(environment, uv=True),
            log=log,
            timeout=_VERSION_TIMEOUT_SECONDS,
        )
        if outcome.timed_out or outcome.exit_code != 0:
            return UNKNOWN_VERSION
        return parse_uv_version(read_log(log))
    finally:
        log.unlink(missing_ok=True)


app.command(name="check")(check_command)
