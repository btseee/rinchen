"""Tests for the student's base config and its rendering (`check_student_base`,
`render_student_config` of `rinchen.train.distill_prepare`): a base of a student holds no
corpora and no way to start from a checkpoint, the finished config states the vocabulary size
of the student's own SentencePiece model, and a student starts from random initialisation
whatever the base says."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.train.distill_prepare import check_student_base, render_student_config
from rinchen.train.eole_cfg import read_config, validate_train_config
from tests.support.eole_train_config import make_config
from tests.support.tiny_parallel import train_toy_spm

CONFIGS = REPO_ROOT / "configs" / "eole"
TINY = CONFIGS / "student_tiny.yaml"


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("student_config_spm"))


def _write(directory: Path, config: dict[str, Any], name: str = "base.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _base(**changes: Any) -> dict[str, Any]:
    config = read_config(TINY, "tiny student")
    for key, value in changes.items():
        if key in ("train_from", "reset_optim"):
            config["training"][key] = value
        else:
            config[key] = value
    return config


def _render(base: Path, spm: Path, out: Path, **overrides: Any) -> Path:
    arguments: dict[str, Any] = {
        "data": make_config()["data"],
        "spm_model": spm,
        "src_vocab": out.parent / "run" / "vocab.src",
        "model_path": out.parent / "run" / "ckpt",
        "seed": 7,
        "out": out,
    } | overrides
    return render_student_config(base, **arguments)


# --- check_student_base --------------------------------------------------------------


@pytest.mark.parametrize("name", ["student_tiny", "student_edge", "student_server"])
def test_the_shipped_student_bases_pass(name: str) -> None:
    base = check_student_base(CONFIGS / f"{name}.yaml")

    assert (
        base["model"]["embeddings"]["position_encoding_type"] == "SinusoidalInterleaved"
    )
    assert base["data"] == {}
    assert "train_from" not in base["training"]
    assert "reset_optim" not in base["training"]


def test_the_checked_base_is_the_base_as_a_mapping() -> None:
    assert check_student_base(TINY) == read_config(TINY, "tiny student")


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"train_from": "runs/a/step_10"}, "train_from"),
        ({"reset_optim": "none"}, "reset_optim"),
        ({"reset_optim": "all"}, "reset_optim"),
        ({"data": {"en-mn.auth": make_config()["data"]["en-mn.auth"]}}, "data"),
        ({"data": {"valid": make_config()["data"]["valid"]}}, "data"),
    ],
)
def test_a_base_that_holds_corpora_or_a_checkpoint_to_start_from_is_refused(
    tmp_path: Path, changes: dict[str, Any], named: str
) -> None:
    base = _write(tmp_path, _base(**changes))

    with pytest.raises(ConfigError, match=named):
        check_student_base(base)


def test_the_refusal_of_a_checkpoint_says_a_student_starts_from_random_initialisation(
    tmp_path: Path,
) -> None:
    base = _write(tmp_path, _base(train_from="runs/a/step_10"))

    with pytest.raises(ConfigError, match="random initialisation"):
        check_student_base(base)


def test_a_base_that_is_not_a_valid_training_config_is_refused(tmp_path: Path) -> None:
    config = _base()
    config["model"]["embeddings"]["position_encoding_type"] = "Rotary"
    base = _write(tmp_path, config)

    with pytest.raises(ConfigError, match="position_encoding_type"):
        check_student_base(base)


def test_a_missing_base_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base"):
        check_student_base(tmp_path / "absent.yaml")


# --- render_student_config -------------------------------------------------------------


def test_the_rendered_config_is_the_base_with_the_run_values_laid_over_it(
    toy_spm: Path, tmp_path: Path
) -> None:
    out = tmp_path / "student" / "config.yaml"

    returned = _render(TINY, toy_spm, out)

    assert returned == out
    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    validate_train_config(config)
    assert config["seed"] == 7
    assert (
        config["src_vocab"] == (tmp_path / "student" / "run" / "vocab.src").as_posix()
    )
    assert (
        config["training"]["model_path"]
        == (tmp_path / "student" / "run" / "ckpt").as_posix()
    )
    assert config["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": toy_spm.as_posix(),
        "tgt_subword_model": toy_spm.as_posix(),
    }
    assert config["data"] == make_config()["data"]
    assert config["src_vocab_size"] == 641
    base = read_config(TINY, "tiny student")
    assert config["model"] == base["model"]
    assert config["training"]["train_steps"] == base["training"]["train_steps"]
    assert (
        config["transforms_configs"]["filtertoolong"]
        == (base["transforms_configs"]["filtertoolong"])
    )


def test_the_render_reads_the_base_and_leaves_it_as_it_is(
    toy_spm: Path, tmp_path: Path
) -> None:
    before = TINY.read_bytes()

    _render(TINY, toy_spm, tmp_path / "config.yaml")

    assert TINY.read_bytes() == before


def test_a_stated_size_that_holds_the_pieces_is_kept_and_a_smaller_one_is_refused(
    toy_spm: Path, tmp_path: Path
) -> None:
    kept = _write(tmp_path, _base(src_vocab_size=64001), "kept.yaml")
    smaller = _write(tmp_path, _base(src_vocab_size=640), "smaller.yaml")
    exact = _write(tmp_path, _base(src_vocab_size=641), "exact.yaml")

    _render(kept, toy_spm, tmp_path / "kept" / "config.yaml")
    _render(exact, toy_spm, tmp_path / "exact" / "config.yaml")
    with pytest.raises(ConfigError, match=r"src_vocab_size.*640.*641"):
        _render(smaller, toy_spm, tmp_path / "smaller" / "config.yaml")

    assert (
        yaml.safe_load((tmp_path / "kept" / "config.yaml").read_text())[
            "src_vocab_size"
        ]
        == 64001
    )
    assert (
        yaml.safe_load((tmp_path / "exact" / "config.yaml").read_text())[
            "src_vocab_size"
        ]
        == 641
    )
    assert not (tmp_path / "smaller").exists()


def test_a_stated_target_size_below_the_pieces_is_refused_too(
    toy_spm: Path, tmp_path: Path
) -> None:
    base = _write(tmp_path, _base(tgt_vocab_size=100))

    with pytest.raises(ConfigError, match=r"tgt_vocab_size.*100.*641"):
        _render(base, toy_spm, tmp_path / "out" / "config.yaml")


def test_a_stated_size_that_is_not_an_integer_is_a_config_error(
    toy_spm: Path, tmp_path: Path
) -> None:
    base = _write(tmp_path, _base(src_vocab_size="large"))

    with pytest.raises(ConfigError, match="src_vocab_size"):
        _render(base, toy_spm, tmp_path / "out" / "config.yaml")


def test_the_server_base_keeps_its_64001_for_a_model_with_fewer_pieces(
    toy_spm: Path, tmp_path: Path
) -> None:
    out = tmp_path / "config.yaml"

    _render(CONFIGS / "student_server.yaml", toy_spm, out)

    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert config["src_vocab_size"] == 64001
    assert config["model"]["encoder"] == {"layers": 16}
    assert config["model"]["decoder"] == {"layers": 3}


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"train_from": "runs/a/step_10"}, "train_from"),
        ({"reset_optim": "all"}, "reset_optim"),
        ({"data": {"en-mn.auth": make_config()["data"]["en-mn.auth"]}}, "data"),
    ],
)
def test_a_base_with_corpora_or_a_checkpoint_renders_nothing(
    toy_spm: Path, tmp_path: Path, changes: dict[str, Any], named: str
) -> None:
    base = _write(tmp_path, _base(**changes))

    with pytest.raises(ConfigError, match=named):
        _render(base, toy_spm, tmp_path / "out" / "config.yaml")

    assert not (tmp_path / "out").exists()


def test_a_model_that_cannot_be_loaded_is_a_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.model"

    with pytest.raises(ConfigError, match="SentencePiece model"):
        _render(TINY, missing, tmp_path / "out" / "config.yaml")

    assert not (tmp_path / "out").exists()


def test_the_output_may_not_be_the_base(toy_spm: Path, tmp_path: Path) -> None:
    base = _write(tmp_path, _base())

    with pytest.raises(ConfigError, match="base config"):
        _render(base, toy_spm, base)


@pytest.mark.parametrize("what", ["src_vocab", "model_path"])
def test_a_dollar_in_a_run_path_is_a_config_error(
    toy_spm: Path, tmp_path: Path, what: str
) -> None:
    with pytest.raises(ConfigError, match=r"\$"):
        _render(TINY, toy_spm, tmp_path / "out" / "config.yaml", **{what: Path("a/$b")})
