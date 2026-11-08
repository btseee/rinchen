"""Where an ablation keeps its files under the output directory.

    data/auth/, data/bt/r<ratio>_s<seed>/   the corpus files, shared where equal
    variants/<id>/config.yaml               the rendered Eole config
                  valid.src, valid.tgt      the pre-tokenised validation corpus
                  run/                      Eole's vocabulary, logs and checkpoint
                  eval/                     the dev evaluation's files
                  train.json                written once trained, until it is scored
                  result.json               written last: the variant is complete
    results.tsv, results.json, results.md   the tables of the run

The plan, the preparation, the run loop and the CLI name these files through this
module, so none of them can disagree about a path. Nothing here touches the disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.train.eole_run import CHECKPOINT_FILE
from rinchen.train.validation_corpus import ValidPaths, validation_paths

DATA_DIR: Final = "data"
VARIANTS_DIR: Final = "variants"
CONFIG_FILE: Final = "config.yaml"
RESULT_FILE: Final = "result.json"
TRAIN_FILE: Final = "train.json"


@dataclass(frozen=True, slots=True)
class VariantPaths:
    """The files and directories of one variant. `ckpt_dir` is the config's
    `training.model_path`, inside `run_dir`."""

    directory: Path
    config: Path
    valid: ValidPaths
    run_dir: Path
    vocab: Path
    ckpt_dir: Path
    checkpoint: Path
    eval_dir: Path
    train_record: Path
    result: Path


def data_dir(out_dir: Path) -> Path:
    """The directory of the corpus files of every variant."""
    return out_dir / DATA_DIR


def variant_paths(out_dir: Path, variant_id: str) -> VariantPaths:
    """The files of the variant `variant_id` of the run in `out_dir`."""
    directory = out_dir / VARIANTS_DIR / variant_id
    run_dir = directory / "run"
    ckpt_dir = run_dir / "ckpt"
    return VariantPaths(
        directory=directory,
        config=directory / CONFIG_FILE,
        valid=validation_paths(directory),
        run_dir=run_dir,
        vocab=run_dir / "vocab.src",
        ckpt_dir=ckpt_dir,
        checkpoint=ckpt_dir / CHECKPOINT_FILE,
        eval_dir=directory / "eval",
        train_record=directory / TRAIN_FILE,
        result=directory / RESULT_FILE,
    )
