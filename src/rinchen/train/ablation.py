"""The run loop of an ablation: train and evaluate the variants of a plan one after the
other, reusing those already complete, and write the result tables.

It sequences; the rules live elsewhere: what the plan is in `ablation_plan`, the files
of a variant and the fingerprint of its inputs in `ablation_prepare`, when a recorded
result may be trusted in `ablation_records`, the tables in `ablation_report`, how
often each corpus is drawn in `sampling`, what a valid Eole config is in `eole_cfg`,
how a model is trained and scored in `eole_run` and `evaluate`. Eole itself and the
scoring are reached through the `train` and `evaluate` seams of `run_ablation`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from rinchen.config.ablation import TagPlacement
from rinchen.errors import UpstreamError
from rinchen.manifest.stage import sha256_of
from rinchen.train.ablation_plan import AblationPlan
from rinchen.train.ablation_prepare import PreparedVariant, prepare_variant
from rinchen.train.ablation_records import (
    AblationResult,
    VariantProgress,
    VariantResult,
    complete_result,
    scores_the_dev_sets,
)
from rinchen.train.ablation_report import write_results
from rinchen.train.eole_cfg import validate_train_config
from rinchen.train.eole_run import CHECKPOINT_FILE, run_training
from rinchen.train.evaluate import DevResult, evaluate_variant
from rinchen.train.mixture import DevSet
from rinchen.utils.part_file import replace_file


class TrainRunner(Protocol):
    """What `run_ablation` needs of a training: the shape of `eole_run.run_training`,
    the seam a unit test replaces. It returns the checkpoint directory."""

    def __call__(
        self,
        cfg_path: Path,
        run_dir: Path,
        *,
        validate: Callable[[Any], None],
        threads: int,
        timeout: float | None,
    ) -> Path: ...


class VariantEvaluator(Protocol):
    """What `run_ablation` needs of an evaluation: the shape of
    `evaluate.evaluate_variant`, the seam a unit test replaces."""

    def __call__(
        self,
        ckpt_dir: Path,
        dev: Mapping[str, DevSet],
        *,
        spm_model: Path,
        placement: TagPlacement,
        out_dir: Path,
        threads: int,
        timeout: float | None,
    ) -> DevResult: ...


def run_ablation(
    plan: AblationPlan,
    *,
    threads: int,
    timeout: float | None,
    force: bool,
    train: TrainRunner = run_training,
    evaluate: VariantEvaluator = evaluate_variant,
    progress: Callable[[VariantProgress], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> AblationResult:
    """Prepare, train and evaluate every variant of `plan` in order, then write the
    result tables (also when every variant was reused).

    A complete variant (`ablation_records.complete_result`) is reused, unless `force`; neither
    seam is called for it. Otherwise its stale `result.json` is removed, `train` is
    called as `eole_run.run_training` is (`validate_train_config` as the validator),
    then `evaluate` as `evaluate.evaluate_variant`, and `result.json` is written
    last, so a crash leaves the variant incomplete. `progress` gets a `VariantProgress`
    after each variant. `threads` goes to both seams and into the fingerprint;
    `timeout` (seconds, per Eole command) goes to both. `clock` times them.

    A failure of a seam, `KeyboardInterrupt` included, stops the run as it is, leaving
    the variants before it complete and the failed one without a `result.json`.

    Raises:
        UpstreamError: training left no checkpoint file, or the evaluation scored
            other dev sets than the plan's; what `train` and `evaluate` raise
            (UpstreamError, ConfigError, DataError) is not changed.
        OSError: a directory or file cannot be created or written.
        And what `prepare_variant` raises (`ValueError` for fewer than one thread,
        before anything is written).
    """
    total = len(plan.variants)
    results: list[VariantResult] = []
    for index, planned in enumerate(plan.variants, start=1):
        prepared = prepare_variant(plan, planned, threads=threads)
        result = None if force else complete_result(plan, prepared)
        if result is None:
            result = _train_and_evaluate(
                plan,
                prepared,
                threads=threads,
                timeout=timeout,
                train=train,
                evaluate=evaluate,
                clock=clock,
            )
        results.append(result)
        if progress is not None:
            progress(
                VariantProgress(
                    variant_id=planned.variant.id,
                    index=index,
                    total=total,
                    outcome="reused" if result.reused else "trained",
                    train_seconds=0.0 if result.reused else result.train_seconds,
                    eval_seconds=0.0 if result.reused else result.eval_seconds,
                )
            )
    outcome = AblationResult(plan=plan, threads=threads, variants=tuple(results))
    write_results(outcome)
    return outcome


def _train_and_evaluate(
    plan: AblationPlan,
    prepared: PreparedVariant,
    *,
    threads: int,
    timeout: float | None,
    train: TrainRunner,
    evaluate: VariantEvaluator,
    clock: Callable[[], float],
) -> VariantResult:
    variant = prepared.variant
    # First: the file vouches for a finished run, and this one is about to replace it.
    prepared.result_path.unlink(missing_ok=True)
    started = clock()
    ckpt_dir = train(
        prepared.config_path,
        prepared.run_dir,
        validate=validate_train_config,
        threads=threads,
        timeout=timeout,
    )
    train_seconds = clock() - started
    checkpoint = ckpt_dir / CHECKPOINT_FILE
    if not checkpoint.is_file():
        raise UpstreamError(
            f"training variant {variant.id} left no {CHECKPOINT_FILE} in {ckpt_dir}"
        )
    checkpoint_sha256 = sha256_of(checkpoint)
    started = clock()
    dev = evaluate(
        ckpt_dir,
        plan.dev.sets,
        spm_model=plan.spm_model,
        placement=variant.tag_placement,
        out_dir=prepared.eval_dir,
        threads=threads,
        timeout=timeout,
    )
    eval_seconds = clock() - started
    if not scores_the_dev_sets(dev, plan.dev.sets):
        raise UpstreamError(
            f"the evaluation of variant {variant.id} scored the directions "
            f"{sorted(dev.directions)}, not the dev sets of the plan "
            f"{sorted(plan.dev.sets)} line for line"
        )
    result = VariantResult(
        planned=prepared.planned,
        reused=False,
        fingerprint=prepared.fingerprint,
        dev=dev,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    # Last of all: a crash before this leaves the variant incomplete.
    document = json.dumps(result.to_dict(), indent=2) + "\n"
    replace_file(prepared.result_path, document.encode("utf-8"))
    return result
