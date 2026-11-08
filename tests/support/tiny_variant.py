"""A tiny multilingual variant trained for real, for the D7 integration tests.

`build_toy_inputs` turns the toy parallel data into what an ablation reads: a corpus
Parquet file, its training rows, the development sets of the four directions and a
SentencePiece model. `train_tiny_variant` then takes one variant through the same
chain the ablation run uses (corpora, mixing weights, the validation corpus, the
Eole `data` section, the rendered config, `run_training`) and returns the checkpoint.

A run this short learns little (chrF between 1 and 35 on the toy dev sets), so what a
test may check is plumbing, never a quality threshold; the quality gate is the
ablation's own. It is long enough that the model writes text: after 40 steps it
writes only the end-of-sentence token, an empty hypothesis for every line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.config.ablation import TagPlacement
from rinchen.config.loader import REPO_ROOT
from rinchen.train.corpus_files import build_corpora, corpus_sizes
from rinchen.train.eole_cfg import read_config, render, validate_train_config
from rinchen.train.eole_data import data_section
from rinchen.train.eole_run import run_training
from rinchen.train.mixture import (
    DevSet,
    TrainingRows,
    load_dev_rows,
    load_training_rows,
)
from rinchen.train.sampling import integer_weights, mixing_probabilities
from rinchen.train.validation_corpus import write_validation
from tests.support.tiny_converter import COMMAND_TIMEOUT_SECONDS
from tests.support.tiny_parallel import build_toy, train_toy_spm, write_corpus_parquet

BASE_CONFIG: Final = REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"
# About 15 seconds with build_vocab on two threads. 40 steps take 13 and teach the
# model nothing but the end-of-sentence token.
DEFAULT_STEPS: Final = 150
DEFAULT_SEED: Final = 1234
DEFAULT_TEMPERATURE: Final = 0.5
DEFAULT_BT_RATIO: Final = 0.5
WEIGHT_TOTAL: Final = 100
THREADS: Final = 2
# The toy generator holds 100 dev items per direction.
DEV_LINES_PER_DIRECTION: Final = 100


@dataclass(frozen=True)
class ToyInputs:
    """What every variant of a toy ablation reads: the training rows, the dev sets
    and the SentencePiece model. The dev and test splits of the toy data are
    separate; the Parquet file holds no test row."""

    spm_model: Path
    training: TrainingRows
    dev: dict[str, DevSet]


@dataclass(frozen=True)
class TinyVariant:
    """One trained variant: where its checkpoint is, how long training took and the
    integer mixing weights it trained with."""

    placement: TagPlacement
    run_dir: Path
    ckpt_dir: Path
    train_seconds: float
    weights: dict[str, int]


def build_toy_inputs(root: Path) -> ToyInputs:
    """Write the toy corpus (training, back-translated and dev rows, no test rows) as
    Parquet under `root`, read it back as an ablation does, and train the toy
    SentencePiece model beside it."""
    toy = build_toy()
    parquet = write_corpus_parquet(
        [*toy.train, *toy.bt, *toy.dev], root / "toy.parquet"
    )
    training = load_training_rows([parquet])
    dev = load_dev_rows(
        [parquet],
        max_lines=DEV_LINES_PER_DIRECTION,
        directions=training.directions,
        training=training,
    )
    return ToyInputs(
        spm_model=train_toy_spm(root / "spm"), training=training, dev=dev.sets
    )


def train_tiny_variant(
    inputs: ToyInputs,
    placement: TagPlacement,
    root: Path,
    *,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    bt_ratio: float = DEFAULT_BT_RATIO,
) -> TinyVariant:
    """Train one variant for `steps` steps under `root` (`data/`, `valid/`,
    `config.yaml` and `run/` are made there) and return it.

    The tiny base config is merged with the per-run values: the vocabulary, the
    checkpoint, the SentencePiece model, the corpora with their weights and the
    seed. The validation and the checkpoint come once, at the last step, so the run
    is as short as a run can be. Training runs on two threads with the GPU hidden.
    """
    run_dir = root / "run"
    built = build_corpora(
        inputs.training, bt_ratio=bt_ratio, seed=seed, data_dir=root / "data"
    )
    probabilities = mixing_probabilities(corpus_sizes(built.corpora), temperature)
    weights = integer_weights(probabilities, total=WEIGHT_TOTAL)
    valid = write_validation(inputs.dev, inputs.spm_model, placement, root / "valid")
    base = read_config(BASE_CONFIG, "base Eole config")
    data = data_section(
        built.corpora,
        weights,
        placement,
        valid_paths=valid,
        filtertoolong="filtertoolong" in base["transforms_configs"],
    )
    spm = inputs.spm_model.as_posix()
    overrides = {
        "seed": seed,
        "src_vocab": (run_dir / "vocab.src").as_posix(),
        "data": data,
        "transforms_configs": {
            "sentencepiece": {"src_subword_model": spm, "tgt_subword_model": spm}
        },
        "training": {
            "model_path": (run_dir / "ckpt").as_posix(),
            "train_steps": steps,
            "valid_steps": steps,
            "save_checkpoint_steps": steps,
        },
    }
    config = render(BASE_CONFIG, overrides, root / "config.yaml")
    started = time.perf_counter()
    ckpt_dir = run_training(
        config,
        run_dir,
        validate=validate_train_config,
        threads=THREADS,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    return TinyVariant(
        placement=placement,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        train_seconds=time.perf_counter() - started,
        weights=weights,
    )
