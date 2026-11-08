"""Tests for `rinchen.train.variants`: the variants an ablation matrix expands to,
in the order temperature, bt ratio, tag placement, seed, each with an id that names
its axes."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.ablation import AblationMatrix
from rinchen.config.loader import REPO_ROOT
from rinchen.train.variants import Variant, expand_variants


def _matrix(tmp_path: Path, **fields: Any) -> AblationMatrix:
    """A matrix over an existing base file with `fields` (axes, seeds, ...) set."""
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\n", encoding="utf-8")
    return AblationMatrix.model_validate({"name": "t", "base": str(base)} | fields)


def test_the_default_matrix_expands_to_one_variant_with_the_documented_id(
    tmp_path: Path,
) -> None:
    variants = expand_variants(_matrix(tmp_path))

    assert variants == [
        Variant(
            id="t1_bt0_src_s1234",
            temperature=1.0,
            bt_ratio=0.0,
            tag_placement="src",
            seed=1234,
        )
    ]


def test_the_id_of_the_spec_example_is_t0_5_bt0_5_src_s1234(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path, axes={"temperature": [0.5], "bt_ratio": [0.5]}, seeds=[1234]
    )

    assert [variant.id for variant in expand_variants(matrix)] == [
        "t0.5_bt0.5_src_s1234"
    ]


@pytest.mark.parametrize(
    ("temperature", "bt_ratio", "printed"),
    [
        (0.5, 0.5, "t0.5_bt0.5"),
        (1.0, 0.0, "t1_bt0"),
        (1, 1, "t1_bt1"),
        (0.3, 0.25, "t0.3_bt0.25"),
        (2.0, 0.125, "t2_bt0.125"),
        (0.05, 10.0, "t0.05_bt10"),
    ],
)
def test_an_id_prints_each_number_the_shortest_way_g_does(
    tmp_path: Path, temperature: float, bt_ratio: float, printed: str
) -> None:
    matrix = _matrix(
        tmp_path,
        axes={"temperature": [temperature], "bt_ratio": [bt_ratio]},
        seeds=[5],
    )

    (variant,) = expand_variants(matrix)

    assert variant.id == f"{printed}_src_s5"


def test_the_variants_are_the_product_with_the_seed_changing_fastest(
    tmp_path: Path,
) -> None:
    matrix = _matrix(
        tmp_path,
        axes={
            "temperature": [0.5, 1.0],
            "bt_ratio": [0.0, 0.5],
            "tag_placement": ["src", "tgt"],
        },
        seeds=[7],
    )

    assert [variant.id for variant in expand_variants(matrix)] == [
        "t0.5_bt0_src_s7",
        "t0.5_bt0_tgt_s7",
        "t0.5_bt0.5_src_s7",
        "t0.5_bt0.5_tgt_s7",
        "t1_bt0_src_s7",
        "t1_bt0_tgt_s7",
        "t1_bt0.5_src_s7",
        "t1_bt0.5_tgt_s7",
    ]


def test_the_seeds_come_in_the_order_given_and_innermost(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, axes={"temperature": [0.5, 1.0]}, seeds=[2, 1, 3])

    assert [variant.id for variant in expand_variants(matrix)] == [
        "t0.5_bt0_src_s2",
        "t0.5_bt0_src_s1",
        "t0.5_bt0_src_s3",
        "t1_bt0_src_s2",
        "t1_bt0_src_s1",
        "t1_bt0_src_s3",
    ]


def test_each_variant_carries_its_own_axes_and_seed(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path,
        axes={"temperature": [0.3], "bt_ratio": [0.25], "tag_placement": ["tgt"]},
        seeds=[11],
    )

    (variant,) = expand_variants(matrix)

    assert (
        variant.temperature,
        variant.bt_ratio,
        variant.tag_placement,
        variant.seed,
    ) == (0.3, 0.25, "tgt", 11)


def test_a_variant_is_a_frozen_record_of_five_fields(tmp_path: Path) -> None:
    (variant,) = expand_variants(_matrix(tmp_path))

    assert [field.name for field in dataclasses.fields(variant)] == [
        "id",
        "temperature",
        "bt_ratio",
        "tag_placement",
        "seed",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.seed = 1  # type: ignore[misc]


def test_expanding_twice_gives_equal_lists(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path, axes={"temperature": [0.5, 1.0], "tag_placement": ["src", "tgt"]}
    )

    assert expand_variants(matrix) == expand_variants(matrix)


def test_the_count_is_the_product_of_the_axis_lengths(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path,
        axes={
            "temperature": [0.3, 0.5, 1.0],
            "bt_ratio": [0.0, 0.25, 0.5, 1.0],
            "tag_placement": ["src", "tgt"],
        },
        seeds=[1, 2],
    )

    assert len(expand_variants(matrix)) == 3 * 4 * 2 * 2


def test_a_matrix_built_over_the_shipped_base_expands(tmp_path: Path) -> None:
    matrix = AblationMatrix.model_validate(
        {
            "name": "t",
            "base": str(REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"),
        }
    )

    assert len(expand_variants(matrix)) == 1
