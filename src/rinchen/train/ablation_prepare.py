"""The files of one variant before Eole runs: its corpora, its validation corpus and
its rendered Eole config, and the fingerprint of the inputs they were made from.

Under the output directory:

    data/auth/, data/bt/r<ratio>_s<seed>/   the corpus files, shared where equal
    variants/<id>/config.yaml               the rendered Eole config
                  valid.src, valid.tgt      the pre-tokenised validation corpus
                  run/                      Eole's vocabulary, logs and checkpoint
                  eval/                     the dev evaluation's files
                  result.json               written last: the variant is complete

The plan that says which variants there are is `ablation_plan`; the loop that trains
and scores them is `ablation`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.manifest.stage import sha256_of, tool_versions
from rinchen.train.ablation_plan import AblationPlan, PlannedVariant
from rinchen.train.corpus_files import BuiltCorpora, build_corpora
from rinchen.train.eole_cfg import read_config, render
from rinchen.train.eole_data import data_section
from rinchen.train.eole_run import CHECKPOINT_FILE
from rinchen.train.validation_corpus import ValidPaths, write_validation
from rinchen.train.variants import Variant

DATA_DIR: Final = "data"
VARIANTS_DIR: Final = "variants"
CONFIG_FILE: Final = "config.yaml"
RESULT_FILE: Final = "result.json"


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
