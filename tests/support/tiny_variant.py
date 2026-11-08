"""A tiny multilingual variant trained for real, for the D7 integration tests.

`build_toy_inputs` turns the toy parallel data into what an ablation reads: a corpus
Parquet file, the development sets of the four directions and a SentencePiece model.
`train_tiny_variant` then takes one variant through the chain the ablation run uses:
a one-variant plan, `prepare_variant` (corpora, mixing weights, the validation corpus,
the Eole `data` section, the rendered config) and `run_training`, and returns the
checkpoint.

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

from rinchen.config.ablation import AblationMatrix, TagPlacement
from rinchen.config.loader import REPO_ROOT
from rinchen.train.ablation_plan import plan_ablation
from rinchen.train.ablation_prepare import prepare_variant
from rinchen.train.eole_cfg import validate_train_config
from rinchen.train.eole_run import run_training
from rinchen.train.mixture import DevSet, load_dev_rows, load_training_rows
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
    """What every variant of a toy ablation reads: the corpus Parquet file, the
    SentencePiece model and the dev sets (as the plan reads them). The dev and test
    splits of the toy data are separate; the Parquet file holds no test row."""

    corpus: Path
    spm_model: Path
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
        corpus=parquet, spm_model=train_toy_spm(root / "spm"), dev=dev.sets
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
    """Train one variant for `steps` steps with `root` as the output directory of its
    plan (`data/` and `variants/<id>/` are made there) and return it.

    The tiny base config is merged with the per-run values by `prepare_variant`; the
    validation and the checkpoint come once, at the last step (a matrix override), so
    the run is as short as a run can be. Training runs on two threads with the GPU
    hidden.
    """
    matrix = AblationMatrix.model_validate(
        {
            "name": "tiny",
            "base": str(BASE_CONFIG),
            "axes": {
                "temperature": [temperature],
                "bt_ratio": [bt_ratio],
                "tag_placement": [placement],
            },
            "seeds": [seed],
            "weight_total": WEIGHT_TOTAL,
            "dev_max_lines": DEV_LINES_PER_DIRECTION,
            "overrides": {
                "training": {
                    "train_steps": steps,
                    "valid_steps": steps,
                    "save_checkpoint_steps": steps,
                }
            },
        }
    )
    plan = plan_ablation(matrix, [inputs.corpus], inputs.spm_model, root)
    prepared = prepare_variant(plan, plan.variants[0], threads=THREADS)
    started = time.perf_counter()
    ckpt_dir = run_training(
        prepared.config_path,
        prepared.run_dir,
        validate=validate_train_config,
        threads=THREADS,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    return TinyVariant(
        placement=placement,
        run_dir=prepared.run_dir,
        ckpt_dir=ckpt_dir,
        train_seconds=time.perf_counter() - started,
        weights=prepared.weights,
    )
