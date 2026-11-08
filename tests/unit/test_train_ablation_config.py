"""Tests for `rinchen.train.ablation_config`: the Eole config of one variant, assembled
the same way for the plan's check (in memory, from the corpora a build would write) and
for the file the preparation renders (from the corpora it wrote).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
    make_matrix,
    one_variant_axes,
    toy_plan,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

OUT = Path("/runs/ab")
SPM = Path("/models/toy.model")


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


def _overrides(**matrix_fields: Any) -> dict[str, Any]:
    matrix = make_matrix(axes=one_variant_axes(), **matrix_fields)
    variant = expand_variants(matrix)[0]
    return variant_overrides(
        matrix,
        variant,
        spm_model=SPM,
        paths=variant_paths(OUT, variant.id),
        corpora=[_files("auth", f"{EN_MN}.auth")],
        weights={f"{EN_MN}.auth": 7},
    )


def test_the_per_run_values_are_the_seed_the_run_paths_the_model_and_the_data() -> None:
    overrides = _overrides()

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
                "src_subword_model": "/models/toy.model",
                "tgt_subword_model": "/models/toy.model",
            }
        },
        "training": {"model_path": f"{variant}/run/ckpt"},
    }


def test_the_matrix_overrides_are_laid_over_the_per_run_values_key_by_key() -> None:
    overrides = _overrides(
        overrides={
            "training": {"dropout": [0.25], "keep_checkpoint": None},
            "model": {"encoder": {"layers": 3}},
        }
    )

    assert overrides["training"] == {
        "model_path": "/runs/ab/variants/t0.5_bt0.5_src_s1234/run/ckpt",
        "dropout": [0.25],
        "keep_checkpoint": None,  # still there: merging it over the base deletes
    }
    assert overrides["model"] == {"encoder": {"layers": 3}}
    assert overrides["seed"] == 1234


def test_an_override_is_laid_over_the_per_run_value_it_names() -> None:
    # The plan refuses overrides of the keys the run sets (`RESERVED_OVERRIDE_KEYS`);
    # the assembly itself just lays the overrides over, whatever they name.
    overrides = _overrides(
        overrides={"seed": 5, "training": {"model_path": "elsewhere"}}
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
