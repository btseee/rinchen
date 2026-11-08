"""The files of one variant before Eole runs: its corpora, its validation corpus and
its rendered Eole config, and the fingerprint of the inputs they were made from.

Where the files go is `ablation_layout`; the config is assembled by `ablation_config`,
the same way the plan checked it. The plan that says which variants there are is
`ablation_plan`; the loop that trains and scores them is `ablation`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.manifest.stage import sha256_of, tool_versions
from rinchen.train.ablation_config import variant_overrides
from rinchen.train.ablation_layout import VariantPaths, data_dir, variant_paths
from rinchen.train.ablation_plan import AblationPlan, PlannedVariant
from rinchen.train.corpus_files import BuiltCorpora, build_corpora
from rinchen.train.eole_cfg import render
from rinchen.train.mixture import DevSet
from rinchen.train.validation_corpus import ValidPaths, write_validation
from rinchen.train.variants import Variant
from rinchen.utils.part_file import lines_as_bytes

# The packages whose version changes a model or a score: Eole trains and loads it, torch
# runs it, SentencePiece cuts the text and sacrebleu scores it. Not polars (it only
# reads the corpus files, whose bytes are in the fingerprint through the corpora) and
# not the code version (a rerun after a commit must not retrain every variant).
_FINGERPRINT_PACKAGES: Final = ("eole", "torch", "sentencepiece", "sacrebleu")


@dataclass(frozen=True, slots=True)
class PreparedVariant:
    """A variant ready to train: its files on disk and the fingerprint of its inputs.

    `fingerprint_inputs` is what the fingerprint is the SHA-256 of (as canonical JSON:
    sorted keys, no spaces): the digest of the rendered config text, of both files of
    every corpus and of the validation files, of the raw dev texts, of the SentencePiece
    model, the thread count, the installed versions of Eole, torch, SentencePiece and
    sacrebleu, and the tag placement. The rendered config holds absolute paths, so
    moving the output directory or the SentencePiece model changes it: a relocated run
    trains again, and the checkpoint the manifest hashes is the artefact of record."""

    planned: PlannedVariant
    paths: VariantPaths
    config_text: str
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
    def directory(self) -> Path:
        return self.paths.directory

    @property
    def config_path(self) -> Path:
        return self.paths.config

    @property
    def run_dir(self) -> Path:
        return self.paths.run_dir

    @property
    def ckpt_dir(self) -> Path:
        """The config's `training.model_path`."""
        return self.paths.ckpt_dir

    @property
    def checkpoint_path(self) -> Path:
        return self.paths.checkpoint

    @property
    def eval_dir(self) -> Path:
        return self.paths.eval_dir

    @property
    def train_record_path(self) -> Path:
        return self.paths.train_record

    @property
    def result_path(self) -> Path:
        return self.paths.result


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
    paths = variant_paths(plan.out_dir, variant.id)
    built = build_corpora(
        plan.training,
        bt_ratio=variant.bt_ratio,
        seed=variant.seed,
        data_dir=data_dir(plan.out_dir),
    )
    valid = write_validation(
        plan.dev.sets, plan.spm_model, variant.tag_placement, paths.directory
    )
    overrides = variant_overrides(
        plan.matrix,
        variant,
        spm_model=plan.spm_model,
        paths=paths,
        corpora=built.corpora,
        weights=planned.weights,
    )
    config_path = render(plan.matrix.base, overrides, paths.config)
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
        "versions": tool_versions(_FINGERPRINT_PACKAGES),
        "dev_raw": _raw_dev_digests(plan.dev.sets),
        "placement": variant.tag_placement,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return PreparedVariant(
        planned=planned,
        paths=paths,
        config_text=config_text,
        valid=valid,
        corpora=built,
        fingerprint_inputs=inputs,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _raw_dev_digests(sets: Mapping[str, DevSet]) -> dict[str, list[str]]:
    """The SHA-256 of the source lines and of the target lines of each dev set, as read
    from the corpus files: the tokenised validation file does not see a change that
    SentencePiece drops (a trailing space), but the evaluation scores the raw text."""
    return {
        direction: [
            hashlib.sha256(lines_as_bytes(sets[direction].src)).hexdigest(),
            hashlib.sha256(lines_as_bytes(sets[direction].tgt)).hexdigest(),
        ]
        for direction in sorted(sets)
    }


def _require_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
