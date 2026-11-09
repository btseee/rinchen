"""The student configs of the distillation shipped under `configs/eole/` (D10):
`student_server.yaml` and `student_edge.yaml` (the two production students, every value an
unmeasured prior) and `student_tiny.yaml` (the CPU gate). That Eole's own schema and our
validator accept each of them is checked with every other base in
`test_eole_configs_shipped.py`; what is pinned here is what makes them students: the shape
of the master plan (a deep encoder and a shallow decoder), the vocabulary of the student's
own SentencePiece model, a recipe taken from the teacher's, no per-run value and no way to
start from a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import load_spm_config
from rinchen.errors import ConfigError
from rinchen.train.eole_cfg import (
    FORBIDDEN_KEYS,
    merge,
    read_config,
    validate_train_config,
)
from tests.support.eole_train_config import make_config

EOLE_DIR = REPO_ROOT / "configs" / "eole"
SERVER = EOLE_DIR / "student_server.yaml"
EDGE = EOLE_DIR / "student_edge.yaml"
TINY = EOLE_DIR / "student_tiny.yaml"
TEACHER = EOLE_DIR / "teacher.yaml"
TEACHER_TINY = EOLE_DIR / "teacher_tiny.yaml"
STUDENT_SPM_CONFIG = REPO_ROOT / "configs" / "spm" / "student_64k.yaml"
PRODUCTION = [pytest.param(SERVER, id="server"), pytest.param(EDGE, id="edge")]
STUDENTS = [*PRODUCTION, pytest.param(TINY, id="tiny")]
# Keys only the launcher's continuation attempt may carry: a student starts from nothing.
LAUNCHER_ONLY_KEYS = ("train_from", "reset_optim", "update_vocab", "keep_states")


def _base(path: Path) -> dict[str, Any]:
    raw = read_config(path, "base Eole config")
    assert isinstance(raw, dict)
    return raw


def _complete(path: Path) -> dict[str, Any]:
    """The base with the per-run values a run adds (the working baseline's)."""
    run = make_config()
    return merge(
        _base(path),
        {
            "src_vocab": run["src_vocab"],
            "data": run["data"],
            "transforms_configs": {
                "sentencepiece": run["transforms_configs"]["sentencepiece"]
            },
            "training": {"model_path": run["training"]["model_path"]},
        },
    )


def _comment_lines(path: Path) -> list[str]:
    return [
        line.lstrip("# ").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    ]


def _keys(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(str(key))
            found |= _keys(value)
    elif isinstance(node, list):
        for value in node:
            found |= _keys(value)
    return found


def _shape(path: Path) -> tuple[int, int, int, int, int, int]:
    model = _base(path)["model"]
    return (
        model["encoder"]["layers"],
        model["decoder"]["layers"],
        model["hidden_size"],
        model["heads"],
        model["transformer_ff"],
        model["embeddings"]["word_vec_size"],
    )


def parameters(
    hidden: int, feed_forward: int, encoder: int, decoder: int, vocabulary: int
) -> int:
    """The parameters of a shared-embedding transformer by the formula of runbook 08
    (`V` rounded up to a multiple of 8, as Eole pads the vocabulary)."""
    padded = -(-vocabulary // 8) * 8
    return (
        padded * (hidden + 1)
        + encoder * (4 * hidden**2 + 2 * hidden * feed_forward + 4 * hidden)
        + decoder * (8 * hidden**2 + 2 * hidden * feed_forward + 6 * hidden + 1)
        + 4 * hidden
    )


@pytest.mark.parametrize(
    ("path", "shape"),
    [
        pytest.param(SERVER, (16, 3, 768, 12, 3072, 768), id="server"),
        pytest.param(EDGE, (12, 2, 512, 8, 2048, 512), id="edge"),
        pytest.param(TINY, (2, 1, 64, 4, 128, 64), id="tiny"),
    ],
)
def test_a_student_has_the_shape_of_the_master_plan(
    path: Path, shape: tuple[int, ...]
) -> None:
    # (encoder layers, decoder layers, hidden, heads, feed-forward, word vector size)
    assert _shape(path) == shape


@pytest.mark.parametrize("path", STUDENTS)
def test_a_student_is_deeper_in_the_encoder_than_in_the_decoder(path: Path) -> None:
    encoder, decoder, hidden, heads, _, word_vector = _shape(path)

    assert encoder > decoder >= 1
    assert hidden % heads == 0
    assert word_vector == hidden


@pytest.mark.parametrize("path", STUDENTS)
def test_a_student_base_trains_from_random_init_and_holds_no_launcher_key(
    path: Path,
) -> None:
    base = _base(path)

    assert not _keys(base) & set(LAUNCHER_ONLY_KEYS)
    for key in ("train_from", "reset_optim"):
        with pytest.raises(ConfigError, match=key):
            validate_train_config(
                merge(_complete(path), {"training": {key: "run/ckpt/step_5"}})
            )


@pytest.mark.parametrize("path", STUDENTS)
def test_a_student_base_holds_no_corpus_and_the_position_scheme_of_every_model(
    path: Path,
) -> None:
    base = _base(path)

    assert base["data"] == {}
    assert base["model"]["embeddings"]["position_encoding_type"] == (
        "SinusoidalInterleaved"
    )
    assert not _keys(base) & FORBIDDEN_KEYS


@pytest.mark.parametrize("path", PRODUCTION)
def test_a_production_student_states_the_pieces_of_the_64k_model_plus_the_pad_token(
    path: Path,
) -> None:
    pieces = load_spm_config(STUDENT_SPM_CONFIG).vocab_size

    assert pieces == 64000
    assert _base(path)["src_vocab_size"] == pieces + 1


def test_the_tiny_student_leaves_the_vocabulary_size_to_its_sentencepiece_model() -> (
    None
):
    base = _base(TINY)

    assert "src_vocab_size" not in base
    assert "tgt_vocab_size" not in base


@pytest.mark.parametrize("path", PRODUCTION)
def test_a_production_student_says_it_is_unmeasured_and_cites_the_master_plan(
    path: Path,
) -> None:
    comments = " ".join(_comment_lines(path))

    assert "UNMEASURED" in comments
    assert "2026-10-29-rinchen-dev-agent-prompt.md" in comments
    assert "random initialisation" in comments


def test_the_tiny_student_says_what_it_is_and_what_the_run_adds() -> None:
    comments = " ".join(_comment_lines(TINY))

    assert "random initialisation" in comments
    assert "This file holds no per-run value" in comments
    assert "src_vocab_size" in comments


@pytest.mark.parametrize("path", PRODUCTION)
def test_a_production_student_keeps_the_schedule_of_the_teacher_it_is_distilled_from(
    path: Path,
) -> None:
    training = _base(path)["training"]
    teacher = _base(TEACHER)["training"]

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
        "learning_rate",
        "warmup_steps",
        "train_steps",
        "valid_steps",
        "save_checkpoint_steps",
        "keep_checkpoint",
        "early_stopping",
        "early_stopping_criteria",
    ):
        assert training[key] == teacher[key], key
    assert training["valid_steps"] == training["save_checkpoint_steps"]
    assert training["keep_checkpoint"] >= training["early_stopping"] + 1


def test_the_tiny_student_is_the_tiny_teachers_recipe_for_2000_steps() -> None:
    base = _base(TINY)
    teacher = _base(TEACHER_TINY)

    assert base["transforms_configs"] == teacher["transforms_configs"]
    assert base["report_every"] == 50
    assert base["training"] == {
        **teacher["training"],
        "train_steps": 2000,
        "valid_steps": 500,
        "save_checkpoint_steps": 500,
        "keep_checkpoint": 5,
    }
    assert base["training"]["gpu_ranks"] == []
    assert base["training"]["compute_dtype"] == "fp32"


def test_the_tiny_student_validates_at_every_save_and_keeps_them_all() -> None:
    training = _base(TINY)["training"]

    assert training["train_steps"] // training["save_checkpoint_steps"] == 4
    assert training["keep_checkpoint"] > 4
    assert training["valid_steps"] == training["save_checkpoint_steps"]


def test_the_parameter_formula_of_the_teacher_runbook_is_exact_on_the_tiny_checkpoints() -> (
    None
):
    # The two real tiny checkpoints of 2026-11-09 (a vocabulary of 344 entries): the
    # 2 + 2 layer teacher and the 2 + 1 layer student.
    assert parameters(64, 128, 2, 2, 344) == 187_738
    assert parameters(64, 128, 2, 1, 344) == 138_201


def test_the_production_students_are_far_smaller_than_the_teacher_by_the_same_formula() -> (
    None
):
    # ARITHMETIC, not a measurement: the formula applied to the shipped values.
    teacher = parameters(1536, 4096, 24, 12, 128_001)
    server = parameters(768, 3072, 16, 3, 64_001)
    edge = parameters(512, 2048, 12, 2, 64_001)

    assert (teacher, server, edge) == (1_102_982_164, 190_845_963, 79_006_218)
    for path, size in ((SERVER, server), (EDGE, edge)):
        encoder, decoder, hidden, _, feed_forward, _ = _shape(path)
        assert parameters(hidden, feed_forward, encoder, decoder, 64_001) == size
