"""`rinchen train ablate`: expand an ablation matrix into variants, train each one from
random initialisation and score its development sets, and record the run in a
manifest. Parsing, file I/O and the manifest only; the work happens in
`rinchen.train`.

The variants are resumable work under the output directory (`variants/<id>/`), not
outputs of the run: the outputs are the three result tables, the rendered Eole config
of each variant and its checkpoint, and the manifest that vouches for them comes last.
(`train.json` is that work's record, not an output: the checkpoint it names is.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from rinchen.cli.errors import (
    FAILURE_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.config.ablation import AblationMatrix, load_ablation
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    begin_commit,
    build_inputs,
    code_version,
    output_file,
    refuse_foreign_manifest,
    tool_versions,
)
from rinchen.train.ablation import run_ablation
from rinchen.train.ablation_layout import variant_paths
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.ablation_records import AblationResult, VariantProgress
from rinchen.train.ablation_report import result_files
from rinchen.train.eole_run import run_training
from rinchen.train.evaluate import evaluate_variant

app = typer.Typer(no_args_is_help=True)

_STAGE_NAME = "train_ablate"
_DEFAULT_THREADS = 2
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
                "variant is reused for, so a rerun needs the same number."
            ),
        ),
    ] = _DEFAULT_THREADS,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help=(
                "Seconds after which each Eole command (the vocabulary, the "
                "training, the translation) is killed; no limit by default."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Train every variant again, instead of reusing finished ones or "
                "only scoring ones that already trained."
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
                "write nothing, not even the output directory."
            ),
        ),
    ] = False,
) -> None:
    """Train every variant of an ablation matrix and score it on the dev sets.

    Each variant gets its own mixture of the corpora, Eole config, checkpoint and dev
    scores. Variants run one after the other; a rerun reuses the ones already
    finished (variants/<id>/result.json), only scores a trained checkpoint whose
    evaluation failed (variants/<id>/train.json) and trains the rest. The tables are
    results.tsv, results.json and results.md in the output directory."""
    with rinchen_errors_as_exit_codes():
        _require_usable_timeout(timeout)
        _reject_repeated_corpora(corpus)
        if dry_run:
            _print_plan(matrix, corpus, spm_model, out_dir)
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
    matrix: Path, corpus: Sequence[Path], spm_model: Path, out_dir: Path
) -> None:
    """The dry run: read the corpora, plan the variants and print the plan."""
    ablation = _load_checked_matrix(matrix, corpus, spm_model, out_dir)
    # The corpora and the matrix are checked as they are read; this is the one input
    # nothing reads before the first variant, and a dry run that passes must not be
    # followed by a refusal for it.
    _require_file(spm_model, what=_SPM_MODEL, missing=ConfigError)
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
    matrix_file = _hash_input(matrix, what=_MATRIX, missing=ConfigError)
    ablation = _load_checked_matrix(matrix, corpus, spm_model, out_dir)
    input_files = [
        matrix_file,
        _hash_input(ablation.base, what=_BASE_CONFIG, missing=ConfigError),
        *(_hash_input(path, what=_CORPUS, missing=DataError) for path in corpus),
        _hash_input(spm_model, what=_SPM_MODEL, missing=ConfigError),
    ]
    plan = plan_ablation(ablation, corpus, spm_model, out_dir)
    # Before any variant is built or trained.
    _create_out_dir(out_dir)
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
        _exit_cannot_write(out_dir, exc)
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


def _require_usable_timeout(timeout: float | None) -> None:
    if timeout is not None and not 0 < timeout < math.inf:
        raise ConfigError(
            f"--timeout must be a positive, finite number of seconds, got {timeout}"
        )


def _reject_repeated_corpora(paths: Sequence[Path]) -> None:
    """A corpus file given twice (whatever the spelling or link) would be read twice
    and double every row of it."""
    given: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in given:
            raise ConfigError(
                f"--corpus {path} names the same file as --corpus {given[resolved]}: "
                "it was given twice and would count every row of it twice"
            )
        given[resolved] = path


def _load_checked_matrix(
    matrix: Path, corpus: Sequence[Path], spm_model: Path, out_dir: Path
) -> AblationMatrix:
    """The matrix, after the refusals that come before anything else is read or
    written, for a dry run too, which must not be followed by a refusal for them."""
    ablation = load_ablation(matrix)
    refuse_foreign_manifest(out_dir, _STAGE_NAME)
    _require_out_apart_from_inputs(out_dir, [matrix, ablation.base, *corpus, spm_model])
    return ablation


def _require_file(
    path: Path, *, what: str, missing: type[ConfigError | DataError]
) -> None:
    if not path.is_file():
        raise missing(f"{what} not found: {path}")


def _hash_input(
    path: Path, *, what: str, missing: type[ConfigError | DataError]
) -> ManifestFile:
    """The manifest entry of the input file `path`, taken before anything reads it.
    A path that is not a file is `missing`, with the message of the loader that would
    have refused it; a file that cannot be read is a DataError."""
    _require_file(path, what=what, missing=missing)
    try:
        return build_inputs([path])[0]
    except OSError as exc:
        raise DataError(f"cannot read {what} {path}: {exc}") from exc


def _require_out_apart_from_inputs(out_dir: Path, sources: Sequence[Path]) -> None:
    """The run writes into `out_dir`, so an input file that lives there could be
    replaced by an output. Compares resolved paths, so a symlink or a `..` does not
    hide the clash, for each file as named and for the file it links to."""
    resolved = out_dir.resolve()
    for source in sources:
        if resolved in (source.resolve().parent, source.parent.resolve()):
            raise ConfigError(
                f"--out-dir {out_dir} is the directory of the input file {source}; "
                "pass a different --out-dir"
            )


def _create_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _exit_cannot_write(out_dir, exc)


def _exit_cannot_write(out_dir: Path, exc: OSError) -> NoReturn:
    exit_with_error(
        f"cannot write the ablation outputs under {out_dir}: {exc}", FAILURE_EXIT_CODE
    )


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
        **_numbers("count", plan.training.stats.to_dict()),
        **_numbers("count.dev", plan.dev.stats.to_dict()),
        "variants.total": float(len(result.variants)),
        "variants.trained": float(counts["trained"]),
        "variants.evaluated": float(counts["evaluated"]),
        "variants.reused": float(counts["reused"]),
        "threads": float(result.threads),
        **{f"seed.{n}": float(seed) for n, seed in enumerate(plan.matrix.seeds)},
    }


def _numbers(prefix: str, document: Mapping[str, Any]) -> dict[str, float]:
    """The numbers of a statistics document under dotted keys, `prefix.key` and
    `prefix.key.subkey`."""
    numbers: dict[str, float] = {}
    for key, value in document.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            numbers.update(_numbers(name, value))
        else:
            numbers[name] = float(value)
    return numbers
