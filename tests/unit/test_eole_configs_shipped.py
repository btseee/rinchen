"""The Eole base configs and the ablation matrices shipped under `configs/eole/`:
`multilingual_tiny.yaml` (the spike's CPU recipe), `multilingual_proxy.yaml` (a GPU
shaped model whose every value is an unmeasured prior), and the matrices
`ablation_tiny.yaml` (2 variants) and `ablation_priors.yaml` (24).

A base holds no per-run value (`src_vocab`, the checkpoint path, the SentencePiece
model, the corpora): the ablation run merges those in. These tests merge in the
baseline's values for them and then ask both validators: ours, and Eole's own schema,
which checks the type of every value (ours checks the keys and a few rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.config.ablation import load_ablation
from rinchen.config.loader import REPO_ROOT
from rinchen.train.eole_cfg import (
    POSITION_ENCODING,
    merge,
    read_config,
    validate_train_config,
)
from rinchen.train.variants import expand_variants
from tests.support.eole_train_config import make_config

EOLE_DIR = REPO_ROOT / "configs" / "eole"
TINY_BASE = EOLE_DIR / "multilingual_tiny.yaml"
PROXY_BASE = EOLE_DIR / "multilingual_proxy.yaml"
TINY_MATRIX = EOLE_DIR / "ablation_tiny.yaml"
PRIORS_MATRIX = EOLE_DIR / "ablation_priors.yaml"

BASES = [pytest.param(TINY_BASE, id="tiny"), pytest.param(PROXY_BASE, id="proxy")]


def _per_run_values() -> dict[str, Any]:
    """What the ablation run adds to a base: the vocabulary, the checkpoint path,
    the SentencePiece model and the corpora (the working baseline's)."""
    run = make_config()
    return {
        "src_vocab": run["src_vocab"],
        "data": run["data"],
        "transforms_configs": {
            "sentencepiece": run["transforms_configs"]["sentencepiece"]
        },
        "training": {"model_path": run["training"]["model_path"]},
    }


def _base(path: Path) -> dict[str, Any]:
    raw = read_config(path, "base Eole config")
    assert isinstance(raw, dict)
    return raw


def _complete(path: Path) -> dict[str, Any]:
    return merge(_base(path), _per_run_values())


def _comment_lines(path: Path) -> list[str]:
    return [
        line.lstrip("# ").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    ]


# --- the bases ---------------------------------------------------------------------


@pytest.mark.parametrize("path", BASES)
def test_a_base_passes_the_validator_once_the_per_run_values_are_merged_in(
    path: Path,
) -> None:
    assert validate_train_config(_complete(path)) is None


@pytest.mark.parametrize("path", BASES)
def test_a_base_is_accepted_by_eoles_own_schema_once_the_per_run_values_are_merged_in(
    path: Path,
) -> None:
    # Constructing the config checks every key and the type of every value against
    # Eole's models and reads no file, so the placeholder paths need not exist.
    from eole.config.run import TrainConfig

    config = TrainConfig(**_complete(path))

    assert config.model.architecture == "transformer"
    assert config.model.embeddings.position_encoding_type == POSITION_ENCODING
    assert config.share_vocab is True
    assert config.overwrite is True


def test_eoles_schema_refuses_a_scalar_where_it_wants_a_list_as_the_test_above_needs() -> (
    None
):
    # The validator checks keys, not the type of a value: this is what the schema
    # check of the shipped bases protects against.
    from eole.config.run import TrainConfig
    from pydantic import ValidationError

    broken = _complete(TINY_BASE)
    broken["training"]["dropout"] = 0.1

    with pytest.raises(ValidationError, match="dropout"):
        TrainConfig(**broken)


@pytest.mark.parametrize("path", BASES)
def test_a_base_leaves_the_per_run_values_to_the_run(path: Path) -> None:
    base = _base(path)

    assert base["data"] == {}
    assert "src_vocab" not in base
    assert "tgt_vocab" not in base
    assert "model_path" not in base["training"]
    assert "sentencepiece" not in base["transforms_configs"]


@pytest.mark.parametrize("path", BASES)
def test_a_base_trains_from_random_init_with_one_shared_vocabulary(path: Path) -> None:
    base = _base(path)

    assert "train_from" not in base
    assert "train_from" not in base["training"]
    assert base["share_vocab"] is True
    assert base["overwrite"] is True
    assert base["model"]["share_embeddings"] is True
    assert base["model"]["share_decoder_embeddings"] is True


@pytest.mark.parametrize("path", BASES)
def test_a_base_says_the_position_scheme_in_the_embeddings(path: Path) -> None:
    assert _base(path)["model"]["embeddings"]["position_encoding_type"] == (
        "SinusoidalInterleaved"
    )


@pytest.mark.parametrize("path", BASES)
def test_a_base_configures_the_length_filter_the_corpora_then_use(path: Path) -> None:
    lengths = _base(path)["transforms_configs"]["filtertoolong"]

    assert set(lengths) == {"src_seq_length", "tgt_seq_length"}
    assert all(isinstance(value, int) and value > 0 for value in lengths.values())


def test_the_tiny_base_is_the_spikes_cpu_recipe() -> None:
    base = _base(TINY_BASE)

    assert base == {
        "seed": 1234,
        "share_vocab": True,
        "overwrite": True,
        "data": {},
        "transforms_configs": {
            "filtertoolong": {"src_seq_length": 64, "tgt_seq_length": 64}
        },
        "model": {
            "architecture": "transformer",
            "hidden_size": 64,
            "encoder": {"layers": 2},
            "decoder": {"layers": 2},
            "heads": 4,
            "transformer_ff": 128,
            "embeddings": {
                "word_vec_size": 64,
                "position_encoding_type": "SinusoidalInterleaved",
            },
            "share_embeddings": True,
            "share_decoder_embeddings": True,
        },
        "training": {
            "batch_type": "tokens",
            "batch_size": 2048,
            "accum_count": [1],
            "dropout": [0.0],
            "attention_dropout": [0.0],
            "optim": "adamw",
            "decay_method": "noam",
            "label_smoothing": 0.1,
            "world_size": 1,
            "train_steps": 1000,
            "valid_steps": 500,
            "save_checkpoint_steps": 1000,
            "keep_checkpoint": 2,
            "learning_rate": 0.5,
            "warmup_steps": 100,
            "compute_dtype": "fp32",
            "gpu_ranks": [],
            "bucket_size": 1024,
            "prefetch_factor": 8,
            "num_workers": 1,
        },
    }


def test_the_proxy_base_is_a_larger_model_for_one_gpu() -> None:
    base = _base(PROXY_BASE)

    assert base["model"]["hidden_size"] == 512
    assert base["model"]["embeddings"]["word_vec_size"] == 512
    assert base["model"]["encoder"] == {"layers": 6}
    assert base["model"]["decoder"] == {"layers": 6}
    assert base["model"]["heads"] == 8
    assert base["model"]["transformer_ff"] == 2048
    assert base["training"]["batch_type"] == "tokens"
    assert base["training"]["batch_size"] == 8192
    assert base["training"]["gpu_ranks"] == [0]
    assert base["training"]["world_size"] == 1
    assert base["training"]["compute_dtype"] == "fp16"


def test_the_proxy_base_asks_eole_for_half_precision() -> None:
    import torch
    from eole.config.run import TrainConfig

    config = TrainConfig(**_complete(PROXY_BASE))

    assert config.training.compute_dtype == torch.float16
    assert config.training.gpu_ranks == [0]


def test_the_proxy_base_says_in_a_comment_that_every_value_is_an_unmeasured_prior() -> (
    None
):
    comments = " ".join(_comment_lines(PROXY_BASE)).lower()

    assert "every value" in comments
    assert "unmeasured prior" in comments


# --- the matrices ------------------------------------------------------------------


def test_the_tiny_matrix_trains_the_two_placements_of_the_spike_at_one_seed() -> None:
    matrix = load_ablation(TINY_MATRIX)

    assert matrix.base == TINY_BASE
    assert [variant.id for variant in expand_variants(matrix)] == [
        "t0.5_bt0.5_src_s1234",
        "t0.5_bt0.5_tgt_s1234",
    ]


def test_the_priors_matrix_expands_to_24_variants_over_the_proxy_base() -> None:
    matrix = load_ablation(PRIORS_MATRIX)

    assert matrix.base == PROXY_BASE
    assert matrix.axes.temperature == [0.3, 0.5, 1.0]
    assert matrix.axes.bt_ratio == [0.0, 0.25, 0.5, 1.0]
    assert matrix.axes.tag_placement == ["src", "tgt"]
    assert matrix.seeds == [1234]
    assert len(expand_variants(matrix)) == 24


def test_the_priors_matrix_orders_its_variants_temperature_first() -> None:
    ids = [variant.id for variant in expand_variants(load_ablation(PRIORS_MATRIX))]

    assert ids[:3] == [
        "t0.3_bt0_src_s1234",
        "t0.3_bt0_tgt_s1234",
        "t0.3_bt0.25_src_s1234",
    ]
    assert ids[-1] == "t1_bt1_tgt_s1234"
    assert len(set(ids)) == 24


def test_the_priors_matrix_says_in_a_comment_that_every_value_is_an_unmeasured_prior() -> (
    None
):
    comments = " ".join(_comment_lines(PRIORS_MATRIX)).lower()

    assert "every value" in comments
    assert "unmeasured prior" in comments


@pytest.mark.parametrize("matrix_path", [TINY_MATRIX, PRIORS_MATRIX])
def test_every_variant_of_a_shipped_matrix_gives_a_valid_config(
    matrix_path: Path,
) -> None:
    matrix = load_ablation(matrix_path)
    complete = merge(_base(matrix.base), _per_run_values())

    for variant in expand_variants(matrix):
        validate_train_config(
            merge(complete, {"seed": variant.seed, **matrix.overrides})
        )


@pytest.mark.parametrize("matrix_path", [TINY_MATRIX, PRIORS_MATRIX])
def test_a_shipped_matrix_has_room_for_the_corpora_of_sixteen_directions(
    matrix_path: Path,
) -> None:
    # A weight per corpus: an authentic and a back-translated corpus for each of up
    # to 16 directions need 32 of the weight total.
    assert load_ablation(matrix_path).weight_total >= 32
