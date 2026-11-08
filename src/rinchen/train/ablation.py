"""The ablation orchestrator: prepare each variant of a plan (corpora, validation
files, Eole config), train and evaluate them one after the other, and write the
results tables. The plan itself (`plan_ablation`) is in `ablation_plan` and the table
writers in `ablation_report`; this module is where a caller finds all of it.

It sequences; the rules live elsewhere: which rows train and how corpora are built in
`mixture` and `corpus_files`, how often each corpus is drawn in `sampling`, what a
valid Eole config is in `eole_cfg`, how a model is trained and scored in `eole_run`
and `evaluate`. Eole itself and the scoring are reached through the `train` and
`evaluate` seams of `run_ablation`.

Under the output directory:

    data/auth/, data/bt/r<ratio>_s<seed>/   the corpus files, shared where equal
    variants/<id>/config.yaml               the rendered Eole config
                  valid.src, valid.tgt      the pre-tokenised validation corpus
                  run/                      Eole's vocabulary, logs and checkpoint
                  eval/                     the dev evaluation's files
                  result.json               written last: the variant is complete
    results.tsv, results.json, results.md   the tables of the run

A variant is complete when its `result.json` parses, says `ok`, holds the fingerprint
of its inputs as they are now and the checkpoint file it names still has the recorded
digest: only then is it reused.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rinchen.config.ablation import TagPlacement
from rinchen.errors import UpstreamError
from rinchen.manifest.stage import sha256_of, tool_versions
from rinchen.train.ablation_plan import (
    RESERVED_OVERRIDE_KEYS,
    AblationPlan,
    PlannedCorpus,
    PlannedVariant,
    plan_ablation,
)
from rinchen.train.ablation_report import ResultFiles, result_files, write_results
from rinchen.train.corpus_files import BuiltCorpora, build_corpora
from rinchen.train.eole_cfg import read_config, render, validate_train_config
from rinchen.train.eole_data import data_section
from rinchen.train.eole_run import CHECKPOINT_FILE, run_training
from rinchen.train.evaluate import DevResult, evaluate_variant
from rinchen.train.mixture import DevSet
from rinchen.train.validation_corpus import ValidPaths, write_validation
from rinchen.train.variants import Variant
from rinchen.utils.part_file import replace_file

# The plan lives in `ablation_plan` and the result files in `ablation_report`; this
# module is the one entry point of the orchestrator, so it names what a caller needs.
__all__ = [
    "CONFIG_FILE",
    "DATA_DIR",
    "RESERVED_OVERRIDE_KEYS",
    "RESULT_FILE",
    "VARIANTS_DIR",
    "AblationPlan",
    "AblationResult",
    "PlannedCorpus",
    "PlannedVariant",
    "PreparedVariant",
    "ResultFiles",
    "TrainRunner",
    "VariantEvaluator",
    "VariantProgress",
    "VariantResult",
    "plan_ablation",
    "prepare_variant",
    "run_ablation",
]

DATA_DIR: Final = "data"
VARIANTS_DIR: Final = "variants"
CONFIG_FILE: Final = "config.yaml"
RESULT_FILE: Final = "result.json"

_STATUS_OK: Final = "ok"


# --- preparing a variant ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedVariant:
    """A variant ready to train: its files on disk and the fingerprint of its inputs.

    `fingerprint_inputs` is what the fingerprint is the SHA-256 of (as canonical JSON:
    sorted keys, no spaces): the digest of the rendered config text, of both files of
    every corpus and of the validation files, of the SentencePiece model, the thread
    count, the installed Eole version and the tag placement."""

    planned: PlannedVariant
    directory: Path
    config_path: Path
    config_text: str
    run_dir: Path
    eval_dir: Path
    result_path: Path
    valid: ValidPaths
    corpora: BuiltCorpora
    fingerprint_inputs: dict[str, Any]
    fingerprint: str

    @property
    def variant(self) -> Variant:
        return self.planned.variant

    @property
    def weights(self) -> dict[str, int]:
        return self.planned.weights

    @property
    def ckpt_dir(self) -> Path:
        """The config's `training.model_path`."""
        return self.run_dir / "ckpt"

    @property
    def checkpoint_path(self) -> Path:
        return self.ckpt_dir / CHECKPOINT_FILE


def prepare_variant(
    plan: AblationPlan, planned: PlannedVariant, *, threads: int
) -> PreparedVariant:
    """Write everything a variant needs before Eole runs: its corpus files (under
    `data/`, shared between variants where equal), its validation corpus, and its Eole
    config `variants/<id>/config.yaml` (`render`: the base, then the per-run values,
    then the matrix overrides over them, all validated). The config names the run
    directory's `vocab.src` and `ckpt`, the variant's seed, the SentencePiece model
    for both sides and the corpora with the planned weights, as absolute POSIX paths;
    the corpora run `filtertoolong` when the finished config configures it.

    Raises:
        ValueError: `threads` is below 1.
        ConfigError: the SentencePiece model cannot be loaded, the finished config is
            not valid, or a path holds `$`.
        DataError: a corpus or validation file cannot be built.
        OSError: a directory or file cannot be created or written.
    """
    _require_threads(threads)
    variant = planned.variant
    directory = plan.out_dir / VARIANTS_DIR / variant.id
    run_dir = directory / "run"
    built = build_corpora(
        plan.training,
        bt_ratio=variant.bt_ratio,
        seed=variant.seed,
        data_dir=plan.out_dir / DATA_DIR,
    )
    valid = write_validation(
        plan.dev.sets, plan.spm_model, variant.tag_placement, directory
    )
    base = read_config(plan.matrix.base, "base Eole config")
    data = data_section(
        built.corpora,
        planned.weights,
        variant.tag_placement,
        valid_paths=valid,
        filtertoolong=_configures_filtertoolong(base, plan.matrix.overrides),
    )
    spm = plan.spm_model.as_posix()
    per_run: dict[str, Any] = {
        "seed": variant.seed,
        "src_vocab": (run_dir / "vocab.src").as_posix(),
        "data": data,
        "transforms_configs": {
            "sentencepiece": {"src_subword_model": spm, "tgt_subword_model": spm}
        },
        "training": {"model_path": (run_dir / "ckpt").as_posix()},
    }
    config_path = render(
        plan.matrix.base,
        _layered(per_run, plan.matrix.overrides),
        directory / CONFIG_FILE,
    )
    config_text = config_path.read_text(encoding="utf-8")
    inputs = {
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "corpora": {
            corpus.name: [corpus.src_sha256, corpus.tgt_sha256]
            for corpus in built.corpora
        },
        "validation": [sha256_of(valid.src), sha256_of(valid.tgt)],
        "spm_sha256": sha256_of(plan.spm_model),
        "threads": threads,
        "eole": tool_versions(["eole"])["eole"],
        "placement": variant.tag_placement,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return PreparedVariant(
        planned=planned,
        directory=directory,
        config_path=config_path,
        config_text=config_text,
        run_dir=run_dir,
        eval_dir=directory / "eval",
        result_path=directory / RESULT_FILE,
        valid=valid,
        corpora=built,
        fingerprint_inputs=inputs,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _require_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")


def _layered(
    per_run: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """`per_run` with `overrides` laid over it: mappings join key by key, every other
    value of `overrides` (`None` included) is kept as it is. `render` merges the result
    over the base, so a `None` still deletes the base's key, also inside a section the
    run itself sets."""
    layered = dict(per_run)
    for key, value in overrides.items():
        current = layered.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            layered[key] = _layered(current, value)
        else:
            layered[key] = value
    return layered


def _configures_filtertoolong(base: object, overrides: Mapping[str, Any]) -> bool:
    """Whether the finished config has `transforms_configs.filtertoolong`: the base's
    unless the overrides set it (a `None` deletes it, as in `eole_cfg.merge`). An
    override of `transforms_configs` itself that is no mapping is refused when the
    plan is made."""
    sections = base.get("transforms_configs") if isinstance(base, Mapping) else None
    configured = isinstance(sections, Mapping) and "filtertoolong" in sections
    override = overrides.get("transforms_configs")
    if not isinstance(override, Mapping) or "filtertoolong" not in override:
        return configured
    return override["filtertoolong"] is not None


# --- running ---------------------------------------------------------------------------


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


@dataclass(frozen=True, slots=True)
class VariantProgress:
    """A variant is done, `trained` just now or `reused` from an earlier run (its
    `result.json` is on disk either way). `index` counts from 1. The seconds are those
    of this call: 0.0 for a reused variant."""

    variant_id: str
    index: int
    total: int
    outcome: Literal["reused", "trained"]
    train_seconds: float
    eval_seconds: float


@dataclass(frozen=True, slots=True)
class VariantResult:
    """A complete variant: its plan, the dev scores, the seconds training and
    evaluation took (those of the run that trained it, also when `reused`), and its
    checkpoint file with the SHA-256 of that file. `fingerprint` is that of the inputs
    it was trained from."""

    planned: PlannedVariant
    reused: bool
    fingerprint: str
    dev: DevResult
    train_seconds: float
    eval_seconds: float
    checkpoint_path: Path
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: what `result.json` holds. `reused` is left out: it says what a
        run did, not what the variant is."""
        return {
            **self.planned.to_dict(),
            "status": _STATUS_OK,
            "fingerprint": self.fingerprint,
            "train_seconds": self.train_seconds,
            "eval_seconds": self.eval_seconds,
            "checkpoint_file": self.checkpoint_path.as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "dev": self.dev.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class AblationResult:
    """A finished run: the plan, the threads it ran on and one `VariantResult` per
    variant in plan order."""

    plan: AblationPlan
    threads: int
    variants: tuple[VariantResult, ...]

    @property
    def files(self) -> ResultFiles:
        """The result tables `run_ablation` wrote."""
        return result_files(self.plan.out_dir)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: what `results.json` holds."""
        return {
            "name": self.plan.matrix.name,
            "threads": self.threads,
            "matrix": self.plan.matrix.model_dump(mode="json"),
            "statistics": self.plan.training.stats.to_dict(),
            "dev_statistics": self.plan.dev.stats.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
        }


class _Recorded(BaseModel):
    """What a `result.json` must say before it counts: strictly typed, extra keys
    ignored."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False, frozen=True)

    status: Literal["ok"]
    fingerprint: str
    train_seconds: Annotated[float, Field(ge=0)]
    eval_seconds: Annotated[float, Field(ge=0)]
    checkpoint_sha256: str
    dev: DevResult


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

    A complete variant (see the module docstring) is reused, unless `force`; neither
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
        result = None if force else _complete_result(plan, prepared)
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
    if not _scores_the_dev_sets(dev, plan.dev.sets):
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


def _complete_result(
    plan: AblationPlan, prepared: PreparedVariant
) -> VariantResult | None:
    """The result `prepared.result_path` records if it vouches for the variant as it
    is now, else None: the file does not parse as a result, says another status, holds
    another fingerprint, scores other dev sets than the plan's, or the checkpoint file
    is missing or has another digest. Nothing in the file is trusted before that."""
    try:
        recorded = _Recorded.model_validate_json(prepared.result_path.read_bytes())
        vouches = (
            recorded.fingerprint == prepared.fingerprint
            and _scores_the_dev_sets(recorded.dev, plan.dev.sets)
            and sha256_of(prepared.checkpoint_path) == recorded.checkpoint_sha256
        )
    except OSError, ValueError:
        return None
    if not vouches:
        return None
    return VariantResult(
        planned=prepared.planned,
        reused=True,
        fingerprint=recorded.fingerprint,
        dev=recorded.dev,
        train_seconds=recorded.train_seconds,
        eval_seconds=recorded.eval_seconds,
        checkpoint_path=prepared.checkpoint_path,
        checkpoint_sha256=recorded.checkpoint_sha256,
    )


def _scores_the_dev_sets(dev: DevResult, sets: Mapping[str, DevSet]) -> bool:
    return set(dev.directions) == set(sets) and all(
        scores.lines == len(sets[direction].src)
        for direction, scores in dev.directions.items()
    )
