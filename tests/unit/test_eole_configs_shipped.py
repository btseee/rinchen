"""The Eole base configs and the ablation matrices shipped under `configs/eole/`:
`multilingual_tiny.yaml` (the spike's CPU recipe), `multilingual_proxy.yaml` (a GPU
shaped model whose every value is an unmeasured prior), `teacher.yaml` (the 8-GPU
teacher, every value an unmeasured prior), `teacher_tiny.yaml` (the CPU teacher gate
of D8), `teacher_doc.yaml` and `teacher_doc_tiny.yaml` (the document-level continuations
of the two teachers, D9), and the matrices `ablation_tiny.yaml` (2 variants) and `ablation_priors.yaml`
(24).

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
from rinchen.errors import ConfigError
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
TEACHER_BASE = EOLE_DIR / "teacher.yaml"
TEACHER_TINY_BASE = EOLE_DIR / "teacher_tiny.yaml"
TEACHER_DOC_BASE = EOLE_DIR / "teacher_doc.yaml"
TEACHER_DOC_TINY_BASE = EOLE_DIR / "teacher_doc_tiny.yaml"
TINY_MATRIX = EOLE_DIR / "ablation_tiny.yaml"
PRIORS_MATRIX = EOLE_DIR / "ablation_priors.yaml"

BASES = [
    pytest.param(TINY_BASE, id="tiny"),
    pytest.param(PROXY_BASE, id="proxy"),
    pytest.param(TEACHER_BASE, id="teacher"),
    pytest.param(TEACHER_TINY_BASE, id="teacher-tiny"),
    pytest.param(TEACHER_DOC_BASE, id="teacher-doc"),
    pytest.param(TEACHER_DOC_TINY_BASE, id="teacher-doc-tiny"),
]


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


# --- the teacher bases (D8) -------------------------------------------------------

TEACHER_BASES = [
    pytest.param(TEACHER_BASE, id="teacher"),
    pytest.param(TEACHER_TINY_BASE, id="teacher-tiny"),
]


def _entries(node: object) -> list[tuple[str, object, object]]:
    found: list[tuple[str, object, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(("", key, value))
            found.extend(_entries(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_entries(value))
    return found


@pytest.mark.parametrize("path", TEACHER_BASES)
def test_a_teacher_base_carries_the_position_scheme_and_no_forbidden_key(
    path: Path,
) -> None:
    from rinchen.train.eole_cfg import FORBIDDEN_KEYS

    base = _base(path)
    keys = {key for _, key, _ in _entries(base) if isinstance(key, str)}

    assert base["model"]["embeddings"]["position_encoding_type"] == (
        "SinusoidalInterleaved"
    )
    assert not keys & FORBIDDEN_KEYS
    assert "train_from" not in keys


@pytest.mark.parametrize("path", TEACHER_BASES)
def test_a_teacher_base_is_not_a_resume_config(path: Path) -> None:
    # `train_from` is allowed only in the resume file the launcher renders itself.
    with pytest.raises(ConfigError, match="train_from"):
        validate_train_config(
            merge(_complete(path), {"training": {"train_from": "run/ckpt/step_5"}})
        )


def test_the_teacher_base_is_the_8_gpu_prior_of_the_master_plan() -> None:
    base = _base(TEACHER_BASE)

    assert base["model"] == {
        "architecture": "transformer",
        "hidden_size": 1536,
        "encoder": {"layers": 24},
        "decoder": {"layers": 12},
        "heads": 16,
        "transformer_ff": 4096,
        "embeddings": {
            "word_vec_size": 1536,
            "position_encoding_type": "SinusoidalInterleaved",
        },
        "share_embeddings": True,
        "share_decoder_embeddings": True,
    }
    training = base["training"]
    assert training["world_size"] == 8
    assert training["gpu_ranks"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert training["gpu_backend"] == "nccl"
    assert training["compute_dtype"] == "bf16"
    assert training["batch_type"] == "tokens"
    assert training["batch_size"] == 16384
    assert training["accum_count"] == [8]
    assert training["dropout"] == [0.1]
    assert training["label_smoothing"] == 0.1
    assert training["valid_steps"] == 5000
    assert training["save_checkpoint_steps"] == 5000
    assert training["early_stopping"] == 6
    assert training["early_stopping_criteria"] == "ppl"
    assert training["keep_checkpoint"] == 8
    assert training["timeout"] == 7200
    assert training["batch_size_multiple"] == 8
    assert base["tensorboard"] is True


def test_the_teacher_base_keeps_the_best_checkpoint_it_can_stop_on() -> None:
    # Eole keeps the last `keep_checkpoint` checkpoints, never the best one: with a
    # validation at every save, early stopping needs one more kept than its patience.
    training = _base(TEACHER_BASE)["training"]

    assert training["valid_steps"] == training["save_checkpoint_steps"]
    assert training["keep_checkpoint"] >= training["early_stopping"] + 1


def test_the_teacher_base_leaves_the_data_and_the_run_paths_to_the_run() -> None:
    base = _base(TEACHER_BASE)

    assert base["data"] == {}
    assert "src_vocab" not in base
    assert "tgt_vocab" not in base
    assert "model_path" not in base["training"]
    assert "sentencepiece" not in base["transforms_configs"]
    assert "valid_metrics" not in base


def test_the_teacher_base_says_in_a_comment_that_every_value_is_an_unmeasured_prior() -> (
    None
):
    comments = " ".join(_comment_lines(TEACHER_BASE)).lower()

    assert "every value" in comments
    assert "unmeasured prior" in comments


def test_the_teacher_base_cites_the_master_plan_lines_its_numbers_come_from() -> None:
    comments = " ".join(_comment_lines(TEACHER_BASE))

    assert "2026-10-29-rinchen-dev-agent-prompt.md" in comments
    assert "511" in comments
    assert "514" in comments


def test_the_teacher_base_is_accepted_by_eoles_own_schema_with_its_8_gpu_settings() -> (
    None
):
    import torch
    from eole.config.run import TrainConfig

    config = TrainConfig(**_complete(TEACHER_BASE))

    assert config.training.compute_dtype == torch.bfloat16
    assert config.training.world_size == 8
    assert config.training.gpu_ranks == [0, 1, 2, 3, 4, 5, 6, 7]
    assert config.training.gpu_backend == "nccl"
    assert config.training.timeout == 7200
    assert config.training.early_stopping == 6
    assert config.training.early_stopping_criteria == "ppl"
    assert config.training.batch_size_multiple == 8
    assert config.tensorboard is True


def test_the_tiny_teacher_is_the_tiny_base_trained_for_200_steps_with_four_saves() -> (
    None
):
    base = _base(TEACHER_TINY_BASE)
    tiny = _base(TINY_BASE)

    assert base["model"] == tiny["model"]
    assert base["transforms_configs"] == tiny["transforms_configs"]
    assert base["report_every"] == 10
    assert base["seed"] == 1234
    expected = {
        **tiny["training"],
        "train_steps": 200,
        "valid_steps": 50,
        "save_checkpoint_steps": 50,
        "keep_checkpoint": 4,
    }
    assert base["training"] == expected
    assert base["training"]["dropout"] == [0.0]
    assert base["training"]["attention_dropout"] == [0.0]
    assert base["training"]["gpu_ranks"] == []
    assert base["training"]["world_size"] == 1
    assert base["training"]["compute_dtype"] == "fp32"


def test_the_tiny_teacher_leaves_the_data_and_the_run_paths_to_the_run() -> None:
    base = _base(TEACHER_TINY_BASE)

    assert base["data"] == {}
    assert "src_vocab" not in base
    assert "model_path" not in base["training"]
    assert "sentencepiece" not in base["transforms_configs"]
    assert "tensorboard" not in base


def test_the_tiny_teacher_keeps_every_checkpoint_of_its_four_saves() -> None:
    training = _base(TEACHER_TINY_BASE)["training"]

    assert training["train_steps"] // training["save_checkpoint_steps"] == 4
    assert training["keep_checkpoint"] == 4
    assert training["valid_steps"] == training["save_checkpoint_steps"]


# --- the document-level continuation bases (D9) -------------------------------------

DOC_BASES = [
    pytest.param(TEACHER_DOC_BASE, TEACHER_BASE, id="teacher-doc"),
    pytest.param(TEACHER_DOC_TINY_BASE, TEACHER_TINY_BASE, id="teacher-doc-tiny"),
]
# Eole fills an option a continuation leaves out from the config of the checkpoint it
# continues, so a continuation base states every option of the optimiser a Rinchen config
# may set (`adam_beta1` and `adam_eps` are not among them: they stay Eole's defaults).
OPTIMIZER_OPTIONS = (
    "optim",
    "adam_beta2",
    "weight_decay",
    "learning_rate",
    "decay_method",
    "warmup_steps",
)
# Keys only the launcher's continuation attempt may carry.
LAUNCHER_ONLY_KEYS = ("train_from", "reset_optim", "update_vocab", "keep_states")


@pytest.mark.parametrize(("path", "teacher"), DOC_BASES)
def test_a_doc_base_continues_the_architecture_of_its_teacher(
    path: Path, teacher: Path
) -> None:
    # `continuation.check_continuation` compares this section with the source's
    # `config.json`: a continuation says which architecture it believes it continues.
    assert _base(path)["model"] == _base(teacher)["model"]


@pytest.mark.parametrize(("path", "teacher"), DOC_BASES)
def test_a_doc_base_states_every_optimizer_option_itself(
    path: Path, teacher: Path
) -> None:
    training = _base(path)["training"]

    for option in OPTIMIZER_OPTIONS:
        assert option in training, option


@pytest.mark.parametrize(("path", "teacher"), DOC_BASES)
def test_a_doc_base_holds_no_key_of_the_launchers_continuation_attempt(
    path: Path, teacher: Path
) -> None:
    base = _base(path)
    keys = {key for _, key, _ in _entries(base) if isinstance(key, str)}

    assert not keys & set(LAUNCHER_ONLY_KEYS)
    # `launch --continue-from` adds `train_from` and `reset_optim`; a base that holds one
    # is refused like any other.
    for key in ("train_from", "reset_optim"):
        with pytest.raises(ConfigError, match=key):
            validate_train_config(
                merge(_complete(path), {"training": {key: "run/ckpt/step_5"}})
            )


@pytest.mark.parametrize(("path", "teacher"), DOC_BASES)
def test_a_doc_base_is_a_valid_continuation_base_with_no_corpora_of_its_own(
    path: Path, teacher: Path
) -> None:
    from rinchen.train.doc_corpus import check_continuation_base

    checked = check_continuation_base(path)

    assert checked["data"] == {}
    assert checked["model"] == _base(path)["model"]


@pytest.mark.parametrize(("path", "teacher"), DOC_BASES)
def test_a_doc_base_allows_windows_longer_than_the_teachers_sentences(
    path: Path, teacher: Path
) -> None:
    lengths = _base(path)["transforms_configs"]["filtertoolong"]
    teachers = _base(teacher)["transforms_configs"]["filtertoolong"]

    assert lengths["src_seq_length"] == lengths["tgt_seq_length"]
    assert lengths["src_seq_length"] > teachers["src_seq_length"]


def test_the_tiny_doc_base_is_the_tiny_teacher_continued_for_2500_steps() -> None:
    base = _base(TEACHER_DOC_TINY_BASE)
    teacher = _base(TEACHER_TINY_BASE)

    assert base["report_every"] == 50
    assert base["seed"] == teacher["seed"] == 1234
    assert base["transforms_configs"] == {
        "filtertoolong": {"src_seq_length": 128, "tgt_seq_length": 128}
    }
    assert base["training"] == {
        **teacher["training"],
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "train_steps": 2500,
        "valid_steps": 500,
        "save_checkpoint_steps": 500,
        "keep_checkpoint": 5,
    }


def test_the_tiny_doc_base_keeps_the_schedule_that_was_measured_on_the_toy() -> None:
    # The spike continued the tiny teacher with these values (learning rate 0.5, 100
    # warm-up steps, Noam, AdamW) and measured the alignment 0.97, 0.92 and 0.99 at 600,
    # 800 and 1000 steps (on overlapping windows).
    training = _base(TEACHER_DOC_TINY_BASE)["training"]

    assert training["optim"] == "adamw"
    assert training["decay_method"] == "noam"
    assert training["learning_rate"] == 0.5
    assert training["warmup_steps"] == 100
    assert training["compute_dtype"] == "fp32"
    assert training["world_size"] == 1
    assert training["gpu_ranks"] == []
    assert training["dropout"] == [0.0]
    assert training["attention_dropout"] == [0.0]


def test_the_tiny_doc_base_keeps_every_checkpoint_of_its_five_saves() -> None:
    training = _base(TEACHER_DOC_TINY_BASE)["training"]

    assert training["train_steps"] // training["save_checkpoint_steps"] == 5
    assert training["keep_checkpoint"] == 5
    assert training["valid_steps"] == training["save_checkpoint_steps"]
    assert training.get("early_stopping", 0) == 0


def test_the_tiny_doc_base_leaves_the_data_and_the_run_paths_to_the_run() -> None:
    base = _base(TEACHER_DOC_TINY_BASE)

    assert base["data"] == {}
    assert "src_vocab" not in base
    assert "model_path" not in base["training"]
    assert "sentencepiece" not in base["transforms_configs"]
    assert "tensorboard" not in base


def test_the_doc_base_is_the_teachers_hardware_and_batch_with_a_continuation_schedule() -> (
    None
):
    base = _base(TEACHER_DOC_BASE)
    teacher = _base(TEACHER_BASE)["training"]
    training = base["training"]

    for key in (
        "batch_type",
        "batch_size",
        "accum_count",
        "dropout",
        "label_smoothing",
        "compute_dtype",
        "world_size",
        "gpu_ranks",
        "gpu_backend",
        "timeout",
        "batch_size_multiple",
        "optim",
        "adam_beta2",
        "decay_method",
    ):
        assert training[key] == teacher[key], key
    assert base["tensorboard"] is True
    # A fresh optimiser on weights that already trained: a peak below the teacher's,
    # which the master plan's numbers give as 2.0 with 8000 warm-up steps.
    assert training["learning_rate"] < teacher["learning_rate"]
    assert training["warmup_steps"] > 0
    assert training["valid_steps"] == training["save_checkpoint_steps"]
    assert training["keep_checkpoint"] >= training["early_stopping"] + 1


def test_the_doc_base_is_accepted_by_eoles_own_schema_with_its_8_gpu_settings() -> None:
    import torch
    from eole.config.run import TrainConfig

    config = TrainConfig(**_complete(TEACHER_DOC_BASE))

    assert config.training.compute_dtype == torch.bfloat16
    assert config.training.world_size == 8
    assert config.training.gpu_ranks == [0, 1, 2, 3, 4, 5, 6, 7]
    assert (
        config.training.learning_rate
        == _base(TEACHER_DOC_BASE)["training"]["learning_rate"]
    )
    assert config.training.decay_method == "noam"


def test_the_tiny_doc_base_is_accepted_by_eoles_own_schema_on_the_cpu() -> None:
    from eole.config.run import TrainConfig

    config = TrainConfig(**_complete(TEACHER_DOC_TINY_BASE))

    assert config.training.gpu_ranks == []
    assert config.training.train_steps == 2500
    assert config.training.adam_beta2 == 0.999
    assert config.report_every == 50


@pytest.mark.parametrize("path", [TEACHER_DOC_BASE, TEACHER_DOC_TINY_BASE])
def test_a_doc_base_cites_the_spike_and_calls_its_unmeasured_values_priors(
    path: Path,
) -> None:
    comments = " ".join(_comment_lines(path)).lower()

    assert "continuation" in comments
    assert "alignment" in comments
    assert "1,000" in comments
    if path == TEACHER_DOC_BASE:
        assert "every value" in comments
        assert "unmeasured prior" in comments
