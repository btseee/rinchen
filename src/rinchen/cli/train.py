"""`rinchen train ablate`: expand an ablation matrix into variants, train each one from
random initialisation and score its development sets, and record the run in a
manifest. Parsing, file I/O and the manifest only; the work happens in
`rinchen.train`. The group's other commands, `launch` and `average` (in
`rinchen.cli.train_runs`) and `docify` (in `rinchen.cli.train_docify`), are registered
here; the checks the modules share are in `rinchen.cli.train_common`.

The variants are resumable work under the output directory (`variants/<id>/`), not
outputs of the run: the outputs are the three result tables, the rendered Eole config
of each variant and its checkpoint, and the manifest that vouches for them comes last.
(`train.json` is that work's record, not an output: the checkpoint it names is.)

`--prepare-only` renders what `ablate` would train on and trains nothing: for each
variant the corpora, the validation corpus and the Eole config, made by the same
function `run_ablation` uses (`prepare_variant`), so a long run (the teacher) can start
from the mixture the ablations chose. Its manifest has its own stage name,
`train_prepare`: it vouches for configs and data files, not for results and
checkpoints, so neither command may replace the other's manifest in an output
directory (`refuse_foreign_manifest`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from rinchen.cli import train_docify, train_runs
from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    corpus_manifest_files,
    create_out_dir,
    exit_cannot_write,
    hash_input,
    numbers_under,
    reject_repeated_corpora,
    require_file,
    require_out_apart_from_inputs,
    require_usable_timeout,
    write_manifest,
)
from rinchen.config.ablation import AblationMatrix, load_ablation
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    begin_commit,
    code_version,
    output_file,
    refuse_foreign_manifest,
    tool_versions,
)
from rinchen.train.ablation import run_ablation
from rinchen.train.ablation_layout import variant_paths
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.ablation_prepare import PreparedVariant, prepare_variant
from rinchen.train.ablation_records import AblationResult, VariantProgress
from rinchen.train.ablation_report import result_files
from rinchen.train.eole_run import run_training
from rinchen.train.evaluate import evaluate_variant

app = typer.Typer(no_args_is_help=True)
app.command("launch")(train_runs.launch_command)
app.command("average")(train_runs.average_command)
app.command("docify")(train_docify.docify_command)

_STAGE_NAME = "train_ablate"
_PREPARE_STAGE = "train_prepare"
_DEFAULT_THREADS = 2
_OUTPUTS = "ablation outputs"
_TOOL_PACKAGES = ("eole", "torch", "sentencepiece", "sacrebleu", "polars")
# How the messages name the inputs; the loaders of the corpus and the matrix say the
# same, so a missing file reads alike whichever check finds it.
_MATRIX = "ablation config file"
_BASE_CONFIG = "base Eole config"
_CORPUS = "corpus file"
_SPM_MODEL = "SentencePiece model"


@app.command("ablate")
def ablate(
    matrix: Annotated[
        Path,
        typer.Option(
            "--matrix",
            help="Ablation matrix (YAML): the axes, the seeds and the base Eole config.",
        ),
    ],
    corpus: Annotated[
        list[Path],
        typer.Option(
            "--corpus",
            help=(
                "Corpus Parquet file; repeat it for more. Its train rows train and "
                "its dev rows score the variants."
            ),
        ),
    ],
    spm_model: Annotated[
        Path,
        typer.Option(
            "--spm-model", help="SentencePiece model every variant tokenises with."
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the variants, the result tables and the manifest.",
        ),
    ],
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            min=1,
            help=(
                "OpenMP and MKL threads of each Eole run. Part of what a finished "
                "variant is reused for, so a rerun needs the same number. With "
                "--prepare-only nothing runs, and it is only recorded in the manifest."
            ),
        ),
    ] = _DEFAULT_THREADS,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help=(
                "Seconds after which each Eole command (the vocabulary, the "
                "training, the translation) is killed; no limit by default. Not "
                "with --prepare-only, which runs none."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Train every variant again, instead of reusing finished ones or "
                "only scoring ones that already trained. Not with --prepare-only."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Print the corpus statistics and, for each variant, the corpora with "
                "their probabilities, integer weights and upsampling; train and "
                "write nothing, not even the output directory. Wins over "
                "--prepare-only."
            ),
        ),
    ] = False,
    prepare_only: Annotated[
        bool,
        typer.Option(
            "--prepare-only",
            help=(
                "Write each variant's corpora, validation files and Eole config.yaml, "
                "exactly as a run would, and train and score nothing. The manifest is "
                "of stage train_prepare."
            ),
        ),
    ] = False,
) -> None:
    """Train every variant of an ablation matrix and score it on the dev sets.

    Each variant gets its own mixture of the corpora, Eole config, checkpoint and dev
    scores. Variants run one after the other; a rerun reuses the ones already
    finished (variants/<id>/result.json), only scores a trained checkpoint whose
    evaluation failed (variants/<id>/train.json) and trains the rest. The tables are
    results.tsv, results.json and results.md in the output directory.

    With --prepare-only it writes the corpora, validation files and config.yaml of
    every variant and prints one line per variant: `prepared <id>: <config.yaml>`."""
    with rinchen_errors_as_exit_codes():
        require_usable_timeout(timeout)
        reject_repeated_corpora(corpus)
        if prepare_only:
            # Refused for a dry run too: it must not be followed by a refusal for it.
            _reject_options_that_prepare_ignores(force=force, timeout=timeout)
        stage = _PREPARE_STAGE if prepare_only else _STAGE_NAME
        if dry_run:
            _print_plan(matrix, corpus, spm_model, out_dir, stage=stage)
        elif prepare_only:
            _prepare_variants(matrix, corpus, spm_model, out_dir, threads=threads)
        else:
            _train_variants(
                matrix,
                corpus,
                spm_model,
                out_dir,
                threads=threads,
                timeout=timeout,
                force=force,
            )


def _print_plan(
    matrix: Path, corpus: Sequence[Path], spm_model: Path, out_dir: Path, *, stage: str
) -> None:
    """The dry run: read the corpora, plan the variants and print the plan. `stage` is
    the one whose manifest the real run would replace."""
    ablation = _load_checked_matrix(matrix, corpus, spm_model, out_dir, stage=stage)
    # The corpora and the matrix are checked as they are read; this is the one input
    # nothing reads before the first variant, and a dry run that passes must not be
    # followed by a refusal for it.
    require_file(spm_model, what=_SPM_MODEL, missing=ConfigError)
    plan = plan_ablation(ablation, corpus, spm_model, out_dir)
    for line in _plan_lines(plan):
        typer.echo(line)


def _train_variants(
    matrix: Path,
    corpus: Sequence[Path],
    spm_model: Path,
    out_dir: Path,
    *,
    threads: int,
    timeout: float | None,
    force: bool,
) -> None:
    """The real run: hash the inputs, plan, train and evaluate every variant, write
    the tables, and record the run in the manifest, last."""
    # Each input is hashed before it is read, so that the manifest vouches for the
    # bytes the run started from.
    matrix_file = hash_input(matrix, what=_MATRIX, missing=ConfigError)
    ablation = _load_checked_matrix(
        matrix, corpus, spm_model, out_dir, stage=_STAGE_NAME
    )
    input_files = _hash_inputs(matrix_file, ablation, corpus, spm_model)
    plan = plan_ablation(ablation, corpus, spm_model, out_dir)
    # Before any variant is built or trained.
    create_out_dir(out_dir, what=_OUTPUTS)
    try:
        # Just before the first write, and after everything that can fail on bad
        # input: the old manifest goes first, so that a failure part-way leaves none.
        begin_commit(plan.out_dir, _STAGE_NAME)
        result = run_ablation(
            plan,
            threads=threads,
            timeout=timeout,
            force=force,
            train=run_training,
            evaluate=evaluate_variant,
            progress=_report_variant,
        )
        manifest_path = _write_manifest(
            result, inputs=input_files, config_sha256=matrix_file.sha256
        )
    except OSError as exc:
        exit_cannot_write(out_dir, exc, what=_OUTPUTS)
    counts = _outcome_counts(result)
    typer.echo(
        f"variants: {len(result.variants)} (trained {counts['trained']}, "
        f"evaluated {counts['evaluated']}, reused {counts['reused']})"
    )
    typer.echo(f"manifest: {manifest_path}")
    tables = result_files(result.plan.out_dir)
    typer.echo(f"results.tsv: {tables.tsv}")
    typer.echo(f"results.json: {tables.json}")
    typer.echo(f"results.md: {tables.md}")


def _reject_options_that_prepare_ignores(*, force: bool, timeout: float | None) -> None:
    """`--prepare-only` trains nothing and runs no Eole command: a `--force` would force
    nothing and a `--timeout` would bound nothing, so either is a mistake to report, not
    to ignore. (`--threads` is part of what a variant's fingerprint is made from, and is
    recorded.)"""
    if force:
        raise ConfigError(
            "--force has nothing to do with --prepare-only: it makes variants train "
            "again, and none trains"
        )
    if timeout is not None:
        raise ConfigError(
            "--timeout has nothing to do with --prepare-only: it bounds Eole "
            "commands, and none runs"
        )


def _prepare_variants(
    matrix: Path,
    corpus: Sequence[Path],
    spm_model: Path,
    out_dir: Path,
    *,
    threads: int,
) -> None:
    """Prepare every variant, as the run does and through the function it uses, and
    record the files in the manifest, last. Nothing is trained or scored."""
    matrix_file = hash_input(matrix, what=_MATRIX, missing=ConfigError)
    ablation = _load_checked_matrix(
        matrix, corpus, spm_model, out_dir, stage=_PREPARE_STAGE
    )
    input_files = _hash_inputs(matrix_file, ablation, corpus, spm_model)
    plan = plan_ablation(ablation, corpus, spm_model, out_dir)
    create_out_dir(out_dir, what=_OUTPUTS)
    try:
        # Just before the first write: the old manifest goes first, so that a failure
        # part-way leaves none.
        begin_commit(plan.out_dir, _PREPARE_STAGE)
        prepared: list[PreparedVariant] = []
        for planned in plan.variants:
            prepared.append(prepare_variant(plan, planned, threads=threads))
            typer.echo(f"prepared {planned.variant.id}: {prepared[-1].config_path}")
        manifest_path = write_manifest(
            plan.out_dir,
            stage=_PREPARE_STAGE,
            config_sha256=matrix_file.sha256,
            seed=plan.matrix.seeds[0],
            inputs=input_files,
            outputs=_prepared_files(prepared),
            metrics=_prepare_metrics(plan, prepared, threads),
            packages=_TOOL_PACKAGES,
            status={"prepare": "ok"},
        )
    except OSError as exc:
        exit_cannot_write(out_dir, exc, what=_OUTPUTS)
    typer.echo(f"manifest: {manifest_path}")


def _prepared_files(prepared: Sequence[PreparedVariant]) -> list[ManifestFile]:
    """Each variant's config and validation files, then every corpus file with its
    record, once (the variants share the files that are equal). The corpus files are
    listed with the digests and row counts the build recorded or checked against the
    files, not read again."""
    listed: dict[Path, ManifestFile] = {}
    for variant in prepared:
        for path in (variant.config_path, variant.valid.src, variant.valid.tgt):
            if path not in listed:
                listed[path] = output_file(path, None)
        for built in variant.corpora.corpora:
            for file in corpus_manifest_files(built):
                listed.setdefault(Path(file.path), file)
    return list(listed.values())


def _prepare_metrics(
    plan: AblationPlan, prepared: Sequence[PreparedVariant], threads: int
) -> dict[str, float]:
    """What the corpora held (as `ablate` records it), how many variants and distinct
    corpus files were prepared, the threads, and the seeds in matrix order."""
    corpora = {
        built.src_path for variant in prepared for built in variant.corpora.corpora
    }
    return {
        **numbers_under("count", plan.training.stats.to_dict()),
        **numbers_under("count.dev", plan.dev.stats.to_dict()),
        "variants.total": float(len(prepared)),
        "corpora.files": float(len(corpora)),
        "threads": float(threads),
        **{f"seed.{n}": float(seed) for n, seed in enumerate(plan.matrix.seeds)},
    }


def _load_checked_matrix(
    matrix: Path,
    corpus: Sequence[Path],
    spm_model: Path,
    out_dir: Path,
    *,
    stage: str,
) -> AblationMatrix:
    """The matrix, after the refusals that come before anything else is read or
    written, for a dry run too, which must not be followed by a refusal for them.
    `stage` is the one whose manifest the run replaces; another stage's is refused."""
    ablation = load_ablation(matrix)
    refuse_foreign_manifest(out_dir, stage)
    require_out_apart_from_inputs(
        out_dir, [matrix, ablation.base, *corpus, spm_model], option="--out-dir"
    )
    return ablation


def _hash_inputs(
    matrix_file: ManifestFile,
    ablation: AblationMatrix,
    corpus: Sequence[Path],
    spm_model: Path,
) -> list[ManifestFile]:
    """The manifest entries of the matrix (already hashed), the base config, the
    corpora and the SentencePiece model, each taken before anything reads it."""
    return [
        matrix_file,
        hash_input(ablation.base, what=_BASE_CONFIG, missing=ConfigError),
        *(hash_input(path, what=_CORPUS, missing=DataError) for path in corpus),
        hash_input(spm_model, what=_SPM_MODEL, missing=ConfigError),
    ]


def _report_variant(progress: VariantProgress) -> None:
    typer.echo(
        f"variant {progress.index}/{progress.total} {progress.variant_id} "
        f"{progress.outcome} train {progress.train_seconds:.1f} s "
        f"eval {progress.eval_seconds:.1f} s"
    )


def _plan_lines(plan: AblationPlan) -> list[str]:
    """What a run would do: the matrix, what was counted in the corpora and, for each
    variant, its axes, the back-translated rows it uses per direction and its corpora
    with their probabilities, integer weights and upsampling. One fact per line."""
    training, dev = plan.training.stats, plan.dev.stats
    lines = [
        f"matrix: {' '.join(plan.matrix.name.split())}",
        f"out dir: {plan.out_dir}",
        f"weight total: {plan.matrix.weight_total}",
        f"corpus files: {len(plan.corpus_paths)}",
        f"rows read: {training.rows_read}",
        f"rows kept: {training.rows_kept}",
        *(f"excluded {reason}: {n}" for reason, n in training.excluded.items()),
        *(
            f"excluded by origin {origin}: {n}"
            for origin, n in training.excluded_by_origin.items()
        ),
        f"duplicate ids: {training.duplicate_ids}",
        f"source collision rate: {training.source_collision_rate:.6f}",
        f"directions: {', '.join(plan.training.directions)}",
        f"dev rows read: {dev.rows_read}",
        f"dev rows usable: {dev.rows_usable}",
        *(f"dev lines {name}: {n}" for name, n in dev.lines_by_direction.items()),
        f"dev in train: {dev.dev_in_train}",
        f"variants: {len(plan.variants)}",
    ]
    for planned in plan.variants:
        variant = planned.variant
        lines.append(
            f"variant {variant.id}: temperature {variant.temperature:g}, "
            f"bt ratio {variant.bt_ratio:g}, tag placement {variant.tag_placement}, "
            f"seed {variant.seed}"
        )
        lines += [
            f"variant {variant.id} bt {direction}: available {usage.bt_available}, "
            f"requested {usage.bt_requested}, used {usage.bt_used}"
            for direction, usage in planned.bt_by_direction.items()
        ]
        lines += [
            f"variant {variant.id} corpus {corpus.name}: rows {corpus.rows}, "
            f"probability {corpus.probability:.6f}, weight {corpus.weight}, "
            f"upsampling {corpus.upsampling:.3f}"
            for corpus in planned.corpora
        ]
    lines.append("nothing was trained and nothing was written")
    return lines


def _write_manifest(
    result: AblationResult, *, inputs: list[ManifestFile], config_sha256: str
) -> Path:
    """Write the manifest of the run, which comes after the result tables so that it
    never describes outputs that are not there. The `seed` field is the first seed of
    the matrix; `metrics` lists them all."""
    manifest = Manifest(
        stage=_STAGE_NAME,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=config_sha256,
        seed=result.plan.matrix.seeds[0],
        inputs=inputs,
        outputs=_output_files(result),
        metrics=_metrics(result),
        tool_versions=tool_versions(_TOOL_PACKAGES),
        status={"ablation": "ok"},
    )
    return store.write(result.plan.out_dir, manifest)


def _output_files(result: AblationResult) -> list[ManifestFile]:
    """The three result tables, then the rendered config and the checkpoint of each
    variant. A checkpoint is listed with the digest its variant recorded when it
    trained and evaluated it (a reused variant's was checked against the file)."""
    tables = result_files(result.plan.out_dir)
    outputs = [
        output_file(tables.tsv, len(result.variants)),
        output_file(tables.json, None),
        output_file(tables.md, None),
    ]
    for variant in result.variants:
        config = variant_paths(result.plan.out_dir, variant.planned.variant.id).config
        checkpoint = variant.checkpoint_path
        outputs += [
            output_file(config, None),
            ManifestFile(
                path=str(checkpoint),
                sha256=variant.checkpoint_sha256,
                bytes=checkpoint.stat().st_size,
            ),
        ]
    return outputs


def _outcome_counts(result: AblationResult) -> dict[str, int]:
    """How many variants this run trained, evaluated from a checkpoint an earlier run
    trained, and reused whole."""
    return {
        outcome: sum(variant.outcome == outcome for variant in result.variants)
        for outcome in ("trained", "evaluated", "reused")
    }


def _metrics(result: AblationResult) -> dict[str, float]:
    """What the corpora held (`count.*` for the training rows, `count.dev.*` for the
    dev rows, named as the statistics name them), how many variants trained, were
    evaluated from an earlier training or were reused, and the run parameters: the
    threads and the seeds in matrix order."""
    plan = result.plan
    counts = _outcome_counts(result)
    return {
        **numbers_under("count", plan.training.stats.to_dict()),
        **numbers_under("count.dev", plan.dev.stats.to_dict()),
        "variants.total": float(len(result.variants)),
        "variants.trained": float(counts["trained"]),
        "variants.evaluated": float(counts["evaluated"]),
        "variants.reused": float(counts["reused"]),
        "threads": float(result.threads),
        **{f"seed.{n}": float(seed) for n, seed in enumerate(plan.matrix.seeds)},
    }
