"""`rinchen eval run`: score one system over an evaluation suite (D13). Parsing, the order of the
steps, the hashing and the exit codes only; the suite, the runner, the metrics, the bootstrap and
`result.json` are `rinchen.eval`'s, the systems and where a model is opened
`rinchen.cli.eval_systems`', the options `rinchen.cli.eval_run_plan`'s, the files that are hashed and
read `rinchen.cli.eval_run_inputs`' and the documents, the files written and the manifest
`rinchen.cli.eval_run_output`'s.

    rinchen eval run --suite FILE --out-dir DIR (--model DIR | --generator NAME | --hypotheses FILE)
        [--generators-config FILE] [--system NAME] [--direction D ...]
        [--beam-size 4] [--max-decoding-length 256] [--script-mask retry|always|off]
        [--device cpu|cuda] [--threads 1] [--robustness DIR] [--segment-scores METRIC=FILE ...]
        [--resamples N] [--seed N] [--metrics-config FILE]

Exactly one of three names the system: `--model` (also `RINCHEN_MODEL`), an export `rinchen export
convert` wrote, which the engine runs on the CPU unless an operator says `cuda` (never run here:
unmeasured; `auto` is refused) for the directions it was trained for; `--generator`, a baseline of D6's
configuration (`--generators-config`, default `configs/data/synth.yaml`), run with one candidate and,
for a server, greedy decoding, whatever the configuration says; or `--hypotheses`, a file in which
any other system wrote its outputs (`hypotheses.jsonl`, which every run writes, is such a file). The
options that set how the engine decodes go with `--model` only, `--segment-scores` with
`--hypotheses` only: per-segment scores of COMET or MetricX produced elsewhere (no scorer is built
here), whose header names the suite and the `hypotheses.jsonl` of this run.

It writes `<DIR>/<suite>/<system>/{hypotheses.jsonl, <direction>/result.json, run.json,
manifest.json}`: for every direction the suite declares there is a state (`run`, `not_trained`,
`unsupported_language`, `not_in_file` or `not_selected`) and only a direction that ran has a result;
a model is never asked for a direction it was not trained for, because the engine would answer it in
the wrong language. `--robustness` embeds D12's block of the same system, with its coverage, in every
result. The command measures and has no threshold: exit 0 whenever it measured. A run over a `dev`
suite is made (its purpose is `selection`) and `rinchen eval report` refuses it.

The order: the options are checked; for `--model` the export is vouched for (`verify_export`); the
suite, the metrics configuration, the generators configuration or hypotheses file, the robustness
files and the scores files are each hashed BEFORE any of them is read, and what is parsed is what was
hashed; an output directory (or run directory) that is an input's or holds another stage's manifest
is refused; the system opens, runs and is closed; every score and document is made; only then is
anything written, the old files first and `manifest.json` LAST. No text of a suite, an output or a
score is ever in a message of this command.

Exit codes: 0; 2 for a bad option, an export that is not vouched for, a vocabulary that lacks the
family's tokens, a file that changed after it was hashed, a robustness run of another system, an
output directory that is an input's, a device that is refused; 3 for a suite, outputs, scores or
robustness input that is missing, unreadable or invalid and another stage's manifest in the output
directory; 4 when CTranslate2 or a server fails; 1 when the output cannot be written; 130 when
interrupted (a run that stops before its first write leaves the last run as it was).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import typer

from rinchen.cli import eval_systems
from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.eval_run_inputs import (
    GeneratorSource,
    Hashed,
    ModelSource,
    entries,
    hash_inputs,
    read_robustness_input,
    read_segment_scores,
    system_entries,
    system_name,
)
from rinchen.cli.eval_run_output import (
    STAGE,
    Prepared,
    commit,
    documents_of,
    options_of,
    prepared,
)
from rinchen.cli.eval_run_plan import DEFAULT_METRICS_PATH, RunPlan, plan_of
from rinchen.cli.train_common import require_out_apart_from_inputs
from rinchen.config.eval_metrics import load_eval_metrics_config
from rinchen.errors import ConfigError, DataError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import direction_place, outputs_to_bytes
from rinchen.eval.runner import SuiteRun, run_suite
from rinchen.eval.suites import Suite
from rinchen.export.verified import VerifiedExport, verified_content
from rinchen.infer.engine_types import ModelInfo, ScriptMode
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import refuse_foreign_manifest


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def run_command(
    suite: Annotated[
        Path, typer.Option("--suite", help="The evaluation suite file (JSON).")
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the run goes into <out-dir>/<suite>/<system>/.",
        ),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            "--model",
            envvar="RINCHEN_MODEL",
            help="The export directory `rinchen export convert` made; the engine translates.",
        ),
    ] = None,
    generator: Annotated[
        str | None,
        typer.Option(
            "--generator",
            help="A baseline of the generators configuration (CTranslate2 or a server).",
        ),
    ] = None,
    hypotheses: Annotated[
        Path | None,
        typer.Option(
            "--hypotheses",
            help="A hypotheses file (JSON Lines): what some system wrote for the suite.",
        ),
    ] = None,
    generators_config: Annotated[
        Path | None,
        typer.Option(
            "--generators-config",
            help="The generators (YAML), with --generator. Default: configs/data/synth.yaml.",
        ),
    ] = None,
    system: Annotated[
        str | None,
        typer.Option(
            "--system",
            help=(
                "Name of the system, 1 to 64 characters: letters, digits, '.', '_' and '-'. "
                "Default: the name of the export, of the generator or the stem of the file."
            ),
        ),
    ] = None,
    direction: Annotated[
        list[str] | None,
        typer.Option(
            "--direction",
            help="Run only this direction of the suite (repeatable); the others are not_selected.",
        ),
    ] = None,
    beam_size: Annotated[
        int | None,
        typer.Option(
            "--beam-size",
            min=1,
            max=8,
            metavar="1..8",
            help="Beam width, 1 to 8 (--model; 4).",
        ),
    ] = None,
    max_decoding_length: Annotated[
        int | None,
        typer.Option(
            "--max-decoding-length",
            min=1,
            help="Longest hypothesis, in pieces (--model; 256).",
        ),
    ] = None,
    script_mask: Annotated[
        ScriptMode | None,
        typer.Option(
            "--script-mask",
            help="retry, always or off: how output in another script is suppressed (--model; retry).",
        ),
    ] = None,
    device: Annotated[
        Literal["cpu", "cuda"] | None,
        typer.Option(
            "--device",
            help="Where CTranslate2 runs; never auto, which would pick the GPU (--model; cpu).",
        ),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads", min=1, help="Threads CTranslate2 uses per batch (--model; 1)."
        ),
    ] = None,
    robustness: Annotated[
        Path | None,
        typer.Option(
            "--robustness",
            help="The output directory of `rinchen eval robustness` of the same system.",
        ),
    ] = None,
    segment_scores: Annotated[
        list[str] | None,
        typer.Option(
            "--segment-scores",
            help="METRIC=FILE: per-segment scores of comet22, metricx24 or metricx24_qe (--hypotheses).",
        ),
    ] = None,
    resamples: Annotated[
        int | None,
        typer.Option(
            "--resamples",
            help="Bootstrap resamples, at least 100. Default: the metrics config's.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Bootstrap seed. Default: the metrics config's."),
    ] = None,
    metrics_config: Annotated[
        Path,
        typer.Option(
            "--metrics-config", help="The metrics and bootstrap settings (YAML)."
        ),
    ] = DEFAULT_METRICS_PATH,
) -> None:
    """Score one system over an evaluation suite: BLEU, chrF and chrF++ with 95 % intervals.

    The system is a Rinchen export (--model), a baseline generator (--generator) or a file of outputs (--hypotheses). It measures and has no threshold: exit 0 whenever it measured."""
    with rinchen_errors_as_exit_codes():
        plan = plan_of(
            model=model,
            generator=generator,
            hypotheses=hypotheses,
            generators_config=generators_config,
            system=system,
            directions=direction,
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            script_mask=script_mask,
            device=device,
            threads=threads,
            robustness=robustness,
            segment_scores=segment_scores,
            resamples=resamples,
            seed=seed,
            metrics_config=metrics_config,
        )
        ready = _measure(
            plan, Path(os.path.abspath(suite)), Path(os.path.abspath(out_dir))
        )
        manifest_path = commit(ready)
        for line in ready.lines:
            typer.echo(line)
        typer.echo(f"run: {ready.run_dir / 'run.json'}")
        typer.echo(f"manifest: {manifest_path}")


class _Ran(NamedTuple):
    """A run and the files its system stood on (hashed before they were read)."""

    run: SuiteRun
    system_inputs: tuple[ManifestFile, ...]


# ---- the steps -------------------------------------------------------------------------------------


def _measure(plan: RunPlan, suite_path: Path, target: Path) -> Prepared:
    """Everything up to the commit: hash, read, run, score and build the documents. Nothing is
    written and no directory is made."""
    hashed = hash_inputs(plan, suite_path)
    suite = Suite.from_bytes(
        verified_content(Path(hashed.suite.path), hashed.suite.sha256)
    )
    config = load_eval_metrics_config(
        verified_content(Path(hashed.metrics.path), hashed.metrics.sha256)
    )
    settings = BootstrapSettings(
        resamples=config.bootstrap.resamples
        if plan.resamples is None
        else plan.resamples,
        seed=config.bootstrap.seed if plan.seed is None else plan.seed,
        alpha=config.bootstrap.alpha,
    )
    _require_declared(suite, plan.selection)
    name = system_name(plan, hashed.source)
    _require_names_that_make_a_directory(name, suite)
    run_dir = target / suite.name / name
    sources = [Path(entry.path) for entry in entries(hashed)]
    require_out_apart_from_inputs(target, sources, option="--out-dir")
    _require_run_dir_apart(run_dir, sources)
    refuse_foreign_manifest(run_dir, STAGE)
    robustness = read_robustness_input(hashed.robustness)
    ran = _run(hashed, name, suite, plan)
    outputs = outputs_to_bytes(ran.run.outputs)
    scores = read_segment_scores(
        hashed.scores, ran.run, hashlib.sha256(outputs).hexdigest()
    )
    documents = documents_of(
        suite=suite,
        suite_sha256=hashed.suite.sha256,
        run=ran.run,
        outputs=outputs,
        settings=settings,
        selection=plan.selection,
        robustness=robustness,
        scores=scores,
        score_files={metric: entry.sha256 for metric, entry in hashed.scores.items()},
    )
    inputs = [
        hashed.suite,
        hashed.metrics,
        *ran.system_inputs,
        *hashed.robustness,
        *hashed.scores.values(),
    ]
    return prepared(
        run_dir=run_dir,
        suite=suite,
        run=ran.run,
        documents=documents,
        inputs=inputs,
        config_sha256=options_of(
            run=ran.run,
            settings=settings,
            selection=plan.selection,
            metrics_sha256=hashed.metrics.sha256,
        ),
        settings=settings,
    )


def _require_names_that_make_a_directory(name: str, suite: Suite) -> None:
    """The run goes into `<out>/<suite>/<system>`: a name that is `.` or `..` matches the id
    pattern and would write beside or above the directory of the run."""
    if name in (".", ".."):
        raise ConfigError("the system name cannot be '.' or '..': it names a directory")
    if suite.name in (".", ".."):
        raise DataError("the suite name cannot be '.' or '..': it names a directory")


def _require_declared(suite: Suite, selection: tuple[str, ...] | None) -> None:
    for entry in selection or ():
        if entry not in suite.directions:
            raise ConfigError(
                "--direction names a direction the suite does not declare: "
                f"{direction_place(entry)}"
            )


def _require_run_dir_apart(run_dir: Path, sources: list[Path]) -> None:
    """The run writes into `run_dir`: an input that lives there (an earlier run's own
    `hypotheses.jsonl` fed back under the same name) would be replaced by an output."""
    resolved = run_dir.resolve()
    for held in sources:
        if resolved in (held.resolve().parent, held.parent.resolve()):
            raise ConfigError(
                f"the run writes into {run_dir}, which holds the input file {held}: "
                "give this run another --system name or another --out-dir"
            )


# ---- the run ----------------------------------------------------------------------------------------


def _run(hashed: Hashed, name: str, suite: Suite, plan: RunPlan) -> _Ran:
    source = hashed.source
    if isinstance(source, ModelSource):
        return _Ran(
            _run_model(source, name, suite, hashed.suite.sha256, plan.selection),
            system_entries(source),
        )
    if isinstance(source, GeneratorSource):
        system = eval_systems.generator_system(
            verified_content(Path(source.config.path), source.config.sha256),
            source.plan.generator,
            system_name=name,
        )
        run = run_suite(suite, hashed.suite.sha256, system, plan.selection)
        return _Ran(run, (source.config, *system.inputs))
    data = verified_content(Path(source.entry.path), source.entry.sha256)
    file_system = eval_systems.file_system(data, source.entry.sha256, suite, name)
    return _Ran(
        run_suite(suite, hashed.suite.sha256, file_system, plan.selection),
        (source.entry,),
    )


def _run_model(
    source: ModelSource,
    name: str,
    suite: Suite,
    suite_sha256: str,
    selection: tuple[str, ...] | None,
) -> SuiteRun:
    """The engine translates the suite. It opens only after every refusal that needs no model
    and is closed whatever happens."""
    engine = eval_systems.open_engine(
        source.verified.directory, source.plan.options.load()
    )
    try:
        _require_the_files_that_were_hashed(engine.info, source.verified)
        system = eval_systems.engine_system(engine, source.plan.options, name)
        return run_suite(suite, suite_sha256, system, selection)
    finally:
        engine.close()


def _require_the_files_that_were_hashed(
    info: ModelInfo, verified: VerifiedExport
) -> None:
    """The engine verifies the export again when it opens it: what it loaded must be what the
    manifest of this run vouches for."""
    loaded = (
        ("model.bin", info.model_sha256, verified.sha256["model.bin"]),
        ("manifest.json", info.manifest_sha256, verified.manifest_sha256),
    )
    for name, digest, hashed in loaded:
        if digest != hashed:
            raise ConfigError(
                f"the engine loaded a {name} that is not the one that was hashed: the "
                "export changed while the run started"
            )
