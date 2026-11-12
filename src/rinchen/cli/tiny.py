"""`rinchen tiny run`: the tiny end-to-end pipeline, from the committed fixtures to `report.json`, on the CPU
with no network and no model download (D15, spec P13). Parsing, the order of the work, the real seams and
the exit codes only; the stages and their order are `rinchen.tiny`'s, the hashing and reading of the inputs
`rinchen.cli.tiny_inputs`', and the files written, the manifest and the printed lines
`rinchen.cli.tiny_output`'s.

    rinchen tiny run [--profile NAME] [--out-dir DIR] [--config FILE] [--qe-fake | --no-qe-fake]
        [--threads N] [--budget-seconds N] [--dry-run]

It writes `<DIR>/report.json` (schema `rinchen.tiny/1`), `<DIR>/report.md`, the directory of each stage
and, last, `<DIR>/manifest.json` (stage `tiny_run`). A run never resumes: it removes what an earlier run
left and starts from nothing.

The order: the options are checked; the pipeline file of the profile is hashed, read through the hash and
parsed, and every other input (the seven fixtures against their pins, the Eole base config, the serve
config, the words file, the profile file that holds the seed, `uv.lock`) is hashed BEFORE any of them is
read; the lock, the seed and the fixtures are parsed from the bytes that were hashed (a file that changed
in between is refused); an output directory that holds another stage's manifest is refused, so is an
existing one that holds anything but the names this command writes; with `--dry-run` the commands of every
stage are listed and the command ends, having written nothing. Otherwise the code version of the checkout
is read, and only then is anything written: the output directory is made, the old manifest is removed first
and then the files of the old run, the stages run (the first failure ends the run, and a spent budget fails
the stage that would have started), the report is built, checked to read back as itself, and written with
its Markdown, and the manifest is written last. The command hides every GPU from its own process for the
whole of its run and puts the caller's setting back when it ends, whatever the exit.

What reaches the screen: a line for each stage with its state and closed reason, a line for each threshold
with its value, the verdict and the three paths, and no text of a file.

Exit codes: 0 when the verdict is pass; 4 when it is fail (a stage failed or a threshold was missed:
everything is written and printed); 2 for a bad option, profile, config or seed, a file that changed after
it was hashed and an output directory that holds entries this command does not write; 3 for a missing
fixture or one that is not its pin, a missing or invalid `uv.lock` and the manifest of another stage in the
output directory; 1 when the output cannot be written or the report is not a valid report (a defect);
130 when interrupted (SIGTERM and SIGHUP end it with 143 and 129); an interrupt writes no manifest.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import typer

from rinchen.cli import tiny_output
from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.maintain_signals import exit_on_termination_signals
from rinchen.cli.tiny_inputs import (
    HashedInputs,
    ReadInputs,
    absolute,
    hash_inputs,
    pipeline_config_path,
    read_inputs,
)
from rinchen.cli.train_common import create_out_dir, exit_cannot_write
from rinchen.config.tiny import MAX_SECONDS, MAX_THREADS
from rinchen.maintain.commands import CommandRunner, device_hidden
from rinchen.manifest.stage import refuse_foreign_manifest
from rinchen.tiny.graceful_runner import GracefulRunner
from rinchen.tiny.pipeline import (
    PipelinePlan,
    PipelineRun,
    StageFunction,
    plan_lines,
    run,
    stage_functions,
)
from rinchen.tiny.report import Inputs, Report, build_report, exit_code

_OUTPUTS: Final = "pipeline output"
DEFAULT_PROFILE: Final = "tiny"
DEFAULT_OUT_DIR: Final = Path("artifacts") / "tiny"

app = typer.Typer(no_args_is_help=True)


def make_runner() -> CommandRunner:
    """The runner of every command of a stage: it asks a command to end before it kills it, so that a
    launcher that is interrupted stops the trainer it started; the tests of the command replace it."""
    return GracefulRunner()


# The two other seams of the command, which the tests replace: the table of the fifteen stage functions
# (bound to the seed and to the bytes of the lock) and the clock of the wall-time budget.
make_stage_functions: Callable[[int, bytes], Mapping[str, StageFunction]] = (
    stage_functions
)
clock: Callable[[], float] = time.monotonic


class _Accepted(NamedTuple):
    """Everything a run needs once every input and option is accepted: nothing has been written yet."""

    target: Path
    hashed: HashedInputs
    read: ReadInputs
    plan: PipelinePlan


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each paragraph is
# one source line and the help wraps it by the width of the terminal.
def run_command(
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="The profile: selects configs/tiny/<profile>.yaml (the pipeline) and configs/profiles/<profile>.yaml (the seed). A profile without a pipeline file is refused.",
        ),
    ] = DEFAULT_PROFILE,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: report.json, report.md, the stage directories and manifest.json.",
        ),
    ] = DEFAULT_OUT_DIR,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="The pipeline file (YAML) in place of the one of the profile.",
        ),
    ] = None,
    qe_fake: Annotated[
        bool | None,
        typer.Option(
            "--qe-fake/--no-qe-fake",
            help="Run the planted stand-ins for LaBSE and the QE scorer (default: the profile's setting).",
        ),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            min=1,
            max=MAX_THREADS,
            help="Threads of each training run and of the CTranslate2 runs (default: the profile's).",
        ),
    ] = None,
    budget_seconds: Annotated[
        int | None,
        typer.Option(
            "--budget-seconds",
            min=1,
            max=MAX_SECONDS,
            help="The wall-time budget of the run in seconds (default: the profile's).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Check the inputs, list the commands of every stage and write nothing.",
        ),
    ] = False,
) -> None:
    """Run the tiny pipeline: the stages from the fixtures to the report, on the CPU.

    Writes report.json, report.md and manifest.json and a directory for each stage under --out-dir.

    Exit codes: 0 when the verdict is pass; 4 when it is fail; 2 for a bad option, profile, config or seed, a changed input or a directory with other entries; 3 for a missing or changed fixture, a bad uv.lock or another stage's manifest in the directory; 1 when the output cannot be written; 130 when interrupted (143 for SIGTERM, 129 for SIGHUP)."""
    with (
        rinchen_errors_as_exit_codes(),
        exit_on_termination_signals(),
        device_hidden(),
    ):
        accepted = _accept(
            profile=profile,
            out_dir=out_dir,
            config=config,
            qe_fake=qe_fake,
            threads=threads,
            budget_seconds=budget_seconds,
        )
        if dry_run:
            for line in plan_lines(accepted.plan):
                typer.echo(line)
            return
        version = tiny_output.observe_code_version()
        outcome = _run(accepted)
        report = _report(profile, accepted, outcome)
        manifest_path = _commit(accepted, report, outcome, version)
        for line in tiny_output.result_lines(
            report, outcome.elapsed, accepted.target, manifest_path
        ):
            typer.echo(line)
        code = exit_code(report)
        if code:
            raise typer.Exit(code=code)


def _accept(
    *,
    profile: str,
    out_dir: Path,
    config: Path | None,
    qe_fake: bool | None,
    threads: int | None,
    budget_seconds: int | None,
) -> _Accepted:
    """Check the options and hash and read the inputs, in the order of the module's text, and write
    nothing.

    Raises:
        ConfigError: a bad profile, config or seed, a file that changed after it was hashed or an output
            directory that holds entries this command does not write.
        DataError: a fixture is missing or is not its pin, `uv.lock` is missing or invalid, or the output
            directory holds the manifest of another stage.
    """
    config_file = pipeline_config_path(
        profile, None if config is None else absolute(config)
    )
    hashed = hash_inputs(profile, config_file)
    read = read_inputs(hashed)
    target = absolute(out_dir)
    refuse_foreign_manifest(target, tiny_output.STAGE)
    tiny_output.require_only_own_entries(target)
    chosen_qe = read.config.qe_fake if qe_fake is None else qe_fake
    chosen_threads = read.config.threads if threads is None else threads
    chosen_budget = (
        read.config.budget_seconds if budget_seconds is None else budget_seconds
    )
    effective = read.config.model_copy(
        update={
            "qe_fake": chosen_qe,
            "threads": chosen_threads,
            "budget_seconds": chosen_budget,
        }
    )
    plan = PipelinePlan(
        config=effective,
        fixtures=read.fixtures,
        out_dir=target,
        runner=make_runner(),
        qe_fake=chosen_qe,
        threads=chosen_threads,
        budget_seconds=chosen_budget,
        seed=read.seed,
        lock_bytes=read.lock_bytes,
    )
    return _Accepted(target, hashed, read, plan)


def _run(accepted: _Accepted) -> PipelineRun:
    """Make the output directory, clear the old run's files and run the stages. Exits with code 1 when
    the directory cannot be made or cleared."""
    target = accepted.target
    create_out_dir(target, what=_OUTPUTS)
    try:
        tiny_output.begin(target)
    except OSError as error:
        exit_cannot_write(target, error, what=_OUTPUTS)
    functions = make_stage_functions(accepted.read.seed, accepted.read.lock_bytes)
    return run(accepted.plan, clock, functions=functions)


def _report(profile: str, accepted: _Accepted, outcome: PipelineRun) -> Report:
    """The report of the run.

    Raises:
        ReportDefect: the observations do not make a valid report (a defect of the program).
    """
    hashed = accepted.hashed
    try:
        inputs = Inputs(
            config_sha256=hashed.config_file.sha256,
            fixtures={name: entry.sha256 for name, entry in hashed.fixtures.items()},
            lock_sha256=hashed.lock.sha256,
        )
        return build_report(
            profile,
            accepted.plan.qe_fake,
            inputs,
            outcome.results,
            outcome.elapsed,
            accepted.plan.config,
        )
    except ValueError as error:
        raise tiny_output.ReportDefect from error


def _commit(
    accepted: _Accepted, report: Report, outcome: PipelineRun, version: str
) -> Path:
    """Write the two documents and then the manifest, and return the manifest's path. Nothing is written
    when the report is not a valid report."""
    data, markdown = tiny_output.report_documents(report)
    target = accepted.target
    try:
        documents = tiny_output.write_documents(target, data, markdown)
        manifest = tiny_output.build_manifest(
            out_dir=target,
            config=accepted.plan.config,
            config_sha256=accepted.hashed.config_file.sha256,
            seed=accepted.read.seed,
            inputs=accepted.hashed.entries(),
            documents=documents,
            lock=accepted.read.lock,
            run=outcome,
            code_version=version,
            python_version=platform.python_version(),
        )
        return tiny_output.write_manifest(target, manifest)
    except OSError as error:
        exit_cannot_write(target, error, what=_OUTPUTS)


app.command(name="run")(run_command)
