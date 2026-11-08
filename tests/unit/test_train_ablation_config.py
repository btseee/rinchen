"""Tests for `rinchen.train.ablation_config`: the Eole config of one variant, assembled
the same way for the plan's check (in memory, from the corpora a build would write) and
for the file the preparation renders (from the corpora it wrote).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError
from rinchen.train.ablation_config import variant_config, variant_overrides
from rinchen.train.ablation_layout import data_dir, variant_paths
from rinchen.train.ablation_plan import AblationPlan
from rinchen.train.ablation_prepare import prepare_variant
from rinchen.train.corpus_files import PlannedFiles, planned_corpus_files
from rinchen.train.eole_cfg import read_config
from rinchen.train.variants import expand_variants
from tests.support.corpus_rows import EN_MN
from tests.support.fake_ablation import (
    BASE_CONFIG,
    make_matrix,
    one_variant_axes,
    toy_plan,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

OUT = Path("/runs/ab")


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("config_spm"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_corpus(tmp_path_factory.mktemp("config_corpus"))


def _planned_config(plan: AblationPlan, index: int) -> dict[str, Any]:
    """The config of the variant as the plan would check it: no file is written."""
    planned = plan.variants[index]
    variant = planned.variant
    return variant_config(
        plan.matrix,
        variant,
        spm_model=plan.spm_model,
        paths=variant_paths(plan.out_dir, variant.id),
        corpora=planned_corpus_files(
            plan.training, variant.bt_ratio, variant.seed, data_dir(plan.out_dir)
        ),
        weights=planned.weights,
    )


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"overrides": {"training": {"dropout": [0.25], "keep_checkpoint": None}}},
        {"overrides": {"transforms_configs": {"filtertoolong": None}}},
        {"axes": {"temperature": [0.5, 2.0], "bt_ratio": [0.0, 0.5]}, "seeds": [1, 2]},
    ],
    ids=["default", "overrides", "no filtertoolong", "a grid"],
)
def test_the_config_the_plan_checks_is_the_config_the_preparation_renders(
    corpus: Path, toy_spm: Path, tmp_path: Path, fields: dict[str, Any]
) -> None:
    plan = toy_plan(tmp_path, toy_spm, corpus=corpus, **fields)

    for index, planned in enumerate(plan.variants):
        prepared = prepare_variant(plan, planned, threads=2)
        assert _planned_config(plan, index) == read_config(
            prepared.config_path, "config"
        )


def _files(group: str, name: str) -> PlannedFiles:
    return PlannedFiles(
        name=name,
        direction=EN_MN,
        group=group,
        tgt_lang="khk_Cyrl",
        src_path=Path(f"/runs/ab/data/{group}/{name}.src"),
        tgt_path=Path(f"/runs/ab/data/{group}/{name}.tgt"),
        rows=10,
    )


def _overrides(spm_model: Path, **matrix_fields: Any) -> dict[str, Any]:
    matrix = make_matrix(axes=one_variant_axes(), **matrix_fields)
    variant = expand_variants(matrix)[0]
    return variant_overrides(
        matrix,
        variant,
        spm_model=spm_model,
        paths=variant_paths(OUT, variant.id),
        corpora=[_files("auth", f"{EN_MN}.auth")],
        weights={f"{EN_MN}.auth": 7},
    )


def test_the_per_run_values_are_the_seed_the_run_paths_the_model_and_the_data(
    toy_spm: Path,
) -> None:
    overrides = _overrides(toy_spm)

    variant = "/runs/ab/variants/t0.5_bt0.5_src_s1234"
    assert overrides == {
        "seed": 1234,
        "src_vocab": f"{variant}/run/vocab.src",
        "data": {
            f"{EN_MN}.auth": {
                "path_src": f"/runs/ab/data/auth/{EN_MN}.auth.src",
                "path_tgt": f"/runs/ab/data/auth/{EN_MN}.auth.tgt",
                "transforms": ["sentencepiece", "prefix", "filtertoolong"],
                "weight": 7,
                "src_prefix": "<2khk_Cyrl>",
                "tgt_prefix": "",
            },
            "valid": {
                "path_src": f"{variant}/valid.src",
                "path_tgt": f"{variant}/valid.tgt",
            },
        },
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": toy_spm.as_posix(),
                "tgt_subword_model": toy_spm.as_posix(),
            }
        },
        "training": {"model_path": f"{variant}/run/ckpt"},
        "src_vocab_size": 641,
    }


def test_the_matrix_overrides_are_laid_over_the_per_run_values_key_by_key(
    toy_spm: Path,
) -> None:
    overrides = _overrides(
        toy_spm,
        overrides={
            "training": {"dropout": [0.25], "keep_checkpoint": None},
            "model": {"encoder": {"layers": 3}},
        },
    )

    assert overrides["training"] == {
        "model_path": "/runs/ab/variants/t0.5_bt0.5_src_s1234/run/ckpt",
        "dropout": [0.25],
        "keep_checkpoint": None,  # still there: merging it over the base deletes
    }
    assert overrides["model"] == {"encoder": {"layers": 3}}
    assert overrides["seed"] == 1234


def test_an_override_is_laid_over_the_per_run_value_it_names(toy_spm: Path) -> None:
    # The plan refuses overrides of the keys the run sets (`RESERVED_OVERRIDE_KEYS`);
    # the assembly itself just lays the overrides over, whatever they name.
    overrides = _overrides(
        toy_spm, overrides={"seed": 5, "training": {"model_path": "elsewhere"}}
    )

    assert overrides["seed"] == 5
    assert overrides["training"] == {"model_path": "elsewhere"}


def test_a_scalar_override_replaces_and_a_none_deletes_in_the_finished_config(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        overrides={"training": {"learning_rate": 0.7, "keep_checkpoint": None}},
    )

    cfg = _planned_config(plan, 0)

    assert cfg["training"]["learning_rate"] == 0.7
    assert "keep_checkpoint" not in cfg["training"]
    assert cfg["training"]["batch_size"] == 2048


def test_an_unknown_key_in_the_overrides_is_a_config_error_naming_it(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(tmp_path, toy_spm, corpus=corpus, axes=one_variant_axes())
    typo = plan.matrix.model_copy(
        update={"overrides": {"training": {"train_stepz": 100}}}
    )

    with pytest.raises(ConfigError, match=r"training\.train_stepz"):
        variant_config(
            typo,
            plan.variants[0].variant,
            spm_model=plan.spm_model,
            paths=variant_paths(plan.out_dir, plan.variants[0].variant.id),
            corpora=planned_corpus_files(
                plan.training, 0.5, 1234, data_dir(plan.out_dir)
            ),
            weights=plan.variants[0].weights,
        )

    assert not plan.out_dir.exists()


def test_planning_the_config_writes_nothing(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(tmp_path, toy_spm, corpus=corpus)
    before = sorted(tmp_path.rglob("*"))

    _planned_config(plan, 0)

    assert sorted(tmp_path.rglob("*")) == before


# ---- the vocabulary size ------------------------------------------------------------
#
# Eole cuts its vocabulary to `src_vocab_size` tokens (32,768 by default), special tokens
# included, and the reserved symbols that `add_reserved_symbols` appends are the last
# lines of the file: a size below the vocabulary would cut exactly them.


def _base_with(tmp_path: Path, **keys: Any) -> Path:
    base = read_config(BASE_CONFIG, "base")
    base.update(keys)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def _config_of(plan: AblationPlan) -> dict[str, Any]:
    return _planned_config(plan, 0)


def test_the_vocabulary_size_is_the_piece_count_and_eoles_pad_token_when_none_is_set(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(tmp_path, toy_spm, corpus=corpus, axes=one_variant_axes())

    cfg = _config_of(plan)

    assert cfg["src_vocab_size"] == 641
    assert "tgt_vocab_size" not in cfg


def test_the_rendered_config_holds_the_vocabulary_size(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(tmp_path, toy_spm, corpus=corpus, axes=one_variant_axes())

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    assert read_config(prepared.config_path, "config")["src_vocab_size"] == 641


@pytest.mark.parametrize("size", [641, 642, 32768])
def test_an_explicit_size_that_is_enough_is_kept_whether_the_base_or_the_overrides_set_it(
    corpus: Path, toy_spm: Path, tmp_path: Path, size: int
) -> None:
    from_base = toy_plan(
        tmp_path / "a",
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        base=str(_base_with(tmp_path, src_vocab_size=size)),
    )
    from_overrides = toy_plan(
        tmp_path / "b",
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        overrides={"src_vocab_size": size},
    )

    assert _config_of(from_base)["src_vocab_size"] == size
    assert _config_of(from_overrides)["src_vocab_size"] == size


def test_a_size_in_the_overrides_wins_over_one_in_the_base(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        base=str(_base_with(tmp_path, src_vocab_size=100)),
        overrides={"src_vocab_size": 700},
    )

    assert _config_of(plan)["src_vocab_size"] == 700


def test_a_size_deleted_by_the_overrides_is_the_default_again(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        base=str(_base_with(tmp_path, src_vocab_size=100)),
        overrides={"src_vocab_size": None},
    )

    assert _config_of(plan)["src_vocab_size"] == 641


@pytest.mark.parametrize("size", [640, 100, 1])
def test_an_explicit_size_below_the_vocabulary_is_refused_naming_both_numbers(
    corpus: Path, toy_spm: Path, tmp_path: Path, size: int
) -> None:
    with pytest.raises(ConfigError, match="src_vocab_size") as info:
        toy_plan(
            tmp_path,
            toy_spm,
            corpus=corpus,
            axes=one_variant_axes(),
            overrides={"src_vocab_size": size},
        )

    message = str(info.value)
    assert f"{size} is below 641" in message
    assert "640 pieces" in message
    assert not (tmp_path / "out").exists()


def test_a_target_size_below_the_vocabulary_is_refused_and_one_that_is_enough_is_kept(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"tgt_vocab_size.*500 is below 641"):
        toy_plan(
            tmp_path / "a",
            toy_spm,
            corpus=corpus,
            axes=one_variant_axes(),
            overrides={"tgt_vocab_size": 500},
        )

    plan = toy_plan(
        tmp_path / "b",
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        overrides={"tgt_vocab_size": 900},
    )

    cfg = _config_of(plan)
    assert (cfg["src_vocab_size"], cfg["tgt_vocab_size"]) == (641, 900)


@pytest.mark.parametrize(
    "size",
    ["5000", True, 0, -3, 700.0],
    ids=["text", "bool", "zero", "negative", "float"],
)
def test_a_size_that_is_not_a_positive_integer_is_refused(
    corpus: Path, toy_spm: Path, tmp_path: Path, size: object
) -> None:
    with pytest.raises(ConfigError, match="src_vocab_size: must be a positive integer"):
        toy_plan(
            tmp_path,
            toy_spm,
            corpus=corpus,
            axes=one_variant_axes(),
            overrides={"src_vocab_size": size},
        )


def test_a_model_that_cannot_be_loaded_is_a_config_error(tmp_path: Path) -> None:
    matrix = make_matrix(axes=one_variant_axes())
    variant = expand_variants(matrix)[0]

    with pytest.raises(ConfigError, match="SentencePiece model"):
        variant_overrides(
            matrix,
            variant,
            spm_model=tmp_path / "missing.model",
            paths=variant_paths(OUT, variant.id),
            corpora=[_files("auth", f"{EN_MN}.auth")],
            weights={f"{EN_MN}.auth": 7},
        )
