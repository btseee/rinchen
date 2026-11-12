"""The CLI stages of the tiny pipeline that make and check the model: the preparation of the training, the
training, the averaging, the CTranslate2 exports and the parity check (D15, spec P2, P3, P7 and P9). The
first three stages with a command are in `model_stages`.

`prepare`, `train` and `average` share the directory `07_train`, and each stage finds the one variant
that the preparation made by looking in it, so no stage needs the facts of an earlier one. The
preparation is `rinchen train ablate --prepare-only`: it writes the corpora, the validation files and the
Eole config of the variant and trains nothing. The facts come from files: the steps from the manifest of
the launch, the rows from the records the preparation wrote beside its corpus files, the digests from the
weights and the exports, the parity from `parity.json`.

The seed is the pipeline's, read from the profile file the command hashed; the context has no field for
it, so the preparation takes it as a second argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from rinchen.errors import DataError
from rinchen.manifest import store
from rinchen.tiny import commands
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.model_stages import SPM_MODEL_FILE, SYNTH_CORPUS_FILE, SYNTH_RUN_DIR
from rinchen.tiny.step_runner import (
    StageCommands,
    StageStopped,
    file_digest,
    path_in,
    run_stage,
)
from rinchen.tiny.thresholds import REFERENCE_SYSTEM

TRAIN_FILE: Final = "train.parquet"
ABLATE_DIR: Final = "ablate"
MATRIX_FILE: Final = "matrix.yaml"
WEIGHTS_FILE: Final = "model.00.safetensors"
EXPORT_MODEL_FILE: Final = "model.bin"
PARITY_FILE: Final = "parity.json"
# What the parity file holds that the facts do not: the lines compared and the sentences that differ.
_PARITY_DROPPED: Final = ("lines", "mismatches")
_GROUPS: Final = ("auth", "bt")


def _variant(ctx: StageContext) -> Path:
    """The directory of the one variant the preparation made.

    Raises:
        DataError: there is none, or more than one.
    """
    made = sorted(path_in(ctx, "prepare", ABLATE_DIR, "variants").glob("*/config.yaml"))
    if len(made) != 1:
        raise DataError(f"{len(made)} prepared variants, not one")
    return made[0].parent


# ---- prepare ------------------------------------------------------------------------------------------


def _prepare(stage: StageCommands, seed: int) -> Mapping[str, Any]:
    ctx = stage.ctx
    matrix = stage.directory / MATRIX_FILE
    corpora = [
        path_in(ctx, "filter", TRAIN_FILE),
        path_in(ctx, "synth", SYNTH_RUN_DIR, SYNTH_CORPUS_FILE),
    ]
    commands.write_matrix(matrix, ctx.config.train, ctx.config.train.base, seed)
    stage.run(
        "ablate",
        commands.train_prepare(
            matrix,
            corpora,
            path_in(ctx, "spm", SPM_MODEL_FILE),
            stage.directory / ABLATE_DIR,
            ctx.threads,
        ),
    )
    return {"variant": _variant(ctx).name}


def run_prepare(ctx: StageContext, seed: int) -> StageResult:
    """Write the matrix of the one variant and run `rinchen train ablate --prepare-only` over the
    filtered corpus and the synthetic one, with the SentencePiece model, in `07_train/ablate`.

    The steps of the profile are the matrix's overrides of the base config, so the prepared config
    holds them. The fact is the id of the variant that was prepared.
    """
    return run_stage(ctx, "prepare", lambda stage: _prepare(stage, seed))


# ---- train --------------------------------------------------------------------------------------------


def _prepared_rows(data_dir: Path) -> tuple[dict[str, int], int]:
    """The rows the preparation wrote for each direction (authentic and back-translated together), by
    direction in order, and the back-translated rows among them, from the record beside each corpus."""
    by_direction: dict[str, int] = {}
    back_translated = 0
    records = sorted(data_dir.glob("**/*.json"))
    if not records:
        raise DataError("the preparation left no corpus record")
    for path in records:
        direction, _, group = path.stem.rpartition(".")
        rows = commands.read_json(path).get("rows")
        if not direction or group not in _GROUPS or type(rows) is not int or rows < 0:
            raise DataError(f"{path.name}: not the record of a corpus")
        by_direction[direction] = by_direction.get(direction, 0) + rows
        if group == "bt":
            back_translated += rows
    return dict(sorted(by_direction.items())), back_translated


def _recorded_steps(run_dir: Path) -> int:
    manifest = store.read(run_dir)
    steps = None if manifest is None else manifest.metrics.get("steps")
    if steps is None or not steps.is_integer() or steps < 1:
        raise DataError("the launch recorded no number of steps")
    return int(steps)


def _train(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    variant = _variant(ctx)
    run_dir = variant / "run"
    stage.run_bounded(
        "launch",
        lambda timeout: commands.train_launch(
            variant / "config.yaml", run_dir, ctx.threads, timeout
        ),
    )
    rows, back_translated = _prepared_rows(path_in(ctx, "prepare", ABLATE_DIR, "data"))
    return {
        "variant": variant.name,
        "steps": _recorded_steps(run_dir),
        "rows_by_direction": rows,
        "bt_rows": back_translated,
    }


def run_train(ctx: StageContext) -> StageResult:
    """Launch the training of the prepared variant with `rinchen train launch` on the CPU, in its run
    directory under `07_train`.

    The facts are the steps the run recorded in its manifest (not the profile's), the rows the
    preparation wrote for each direction and how many of them are back-translated.
    """
    return run_stage(ctx, "train", _train)


# ---- average ------------------------------------------------------------------------------------------


def _average(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    run_dir = _variant(ctx) / "run"
    stage.run(
        "average",
        commands.train_average(run_dir / "ckpt", ctx.config.train.average_last),
    )
    return {"weights_sha256": file_digest(run_dir / "averaged" / WEIGHTS_FILE)}


def run_average(ctx: StageContext) -> StageResult:
    """Average the last checkpoints of the run with `rinchen train average`, into `averaged` beside
    them. The fact is the SHA-256 of the averaged weights."""
    return run_stage(ctx, "average", _average)


# ---- export -------------------------------------------------------------------------------------------


def _export(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    averaged = _variant(ctx) / "run" / "averaged"
    spm_model = path_in(ctx, "spm", SPM_MODEL_FILE)
    digests: dict[str, str] = {}
    for quantization in ctx.config.export.quantizations:
        out_dir = stage.directory / quantization
        stage.run(
            quantization,
            commands.export_convert(averaged, spm_model, quantization, out_dir),
        )
        digests[quantization] = file_digest(out_dir / EXPORT_MODEL_FILE)
    return {"model_sha256": digests}


def run_export(ctx: StageContext) -> StageResult:
    """Convert the averaged weights to a CTranslate2 export for each quantization of the profile with
    `rinchen export convert`, the direction tag in the source, into `08_export/<quantization>`. The
    facts are the SHA-256 of each `model.bin`."""
    return run_stage(ctx, "export", _export)


# ---- parity -------------------------------------------------------------------------------------------


def _parity_facts(parity: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in parity.items() if key not in _PARITY_DROPPED}


def _parity(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    if REFERENCE_SYSTEM not in ctx.config.export.quantizations:
        raise DataError("the parity check needs the reference export")
    out_dir = stage.directory / REFERENCE_SYSTEM
    try:
        stage.run_bounded(
            REFERENCE_SYSTEM,
            lambda timeout: commands.export_parity(
                _variant(ctx) / "run" / "averaged",
                path_in(ctx, "export", REFERENCE_SYSTEM),
                path_in(ctx, "filter", TRAIN_FILE),
                out_dir,
                ctx.config.parity.max_lines,
                ctx.threads,
                timeout,
            ),
        )
    except StageStopped as stopped:
        # A parity that is missed exits with 4 after writing its measurement: the stage fails and
        # still carries it, so that the thresholds see the figures and not an absence.
        raise StageStopped(stopped.record, _measured(out_dir / PARITY_FILE)) from None
    return _parity_facts(commands.read_parity(out_dir / PARITY_FILE))


def _measured(path: Path) -> dict[str, Any]:
    try:
        return _parity_facts(commands.read_parity(path))
    except DataError:
        return {}


def run_parity(ctx: StageContext) -> StageResult:
    """Run `rinchen export parity` on the float32 export against the averaged weights, over the dev
    lines of the filtered corpus, into `09_parity/float32`.

    The facts are the content of `parity.json` without `lines` and `mismatches`. A parity that misses
    (exit code 4) fails the stage and keeps those facts when the file was written.
    """
    return run_stage(ctx, "parity", _parity)
