"""Property tests for `rinchen.train.variants.expand_variants` over random axes: the
variants are the product of the axes in their order, and no two share an id."""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from rinchen.config.ablation import AblationMatrix
from rinchen.config.loader import REPO_ROOT
from rinchen.train.variants import expand_variants

_BASE = str(REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml")
# Distinct values that print alike in an id are rejected by the matrix, so the
# strategies draw values distinct under the same format.
_TEMPERATURES = st.lists(
    st.floats(min_value=0.01, max_value=4.0),
    min_size=1,
    max_size=4,
    unique_by=lambda value: f"{value:g}",
)
_RATIOS = st.lists(
    st.floats(min_value=0.0, max_value=4.0),
    min_size=1,
    max_size=4,
    unique_by=lambda value: f"{value:g}",
)
_PLACEMENTS = st.lists(
    st.sampled_from(["src", "tgt"]), min_size=1, max_size=2, unique=True
)
_SEEDS = st.lists(st.integers(0, 10**6), min_size=1, max_size=3, unique=True)


@given(
    temperatures=_TEMPERATURES,
    ratios=_RATIOS,
    placements=_PLACEMENTS,
    seeds=_SEEDS,
)
def test_the_variants_are_the_product_of_the_axes_with_unique_ids(
    temperatures: list[float],
    ratios: list[float],
    placements: list[str],
    seeds: list[int],
) -> None:
    matrix = AblationMatrix.model_validate(
        {
            "name": "p",
            "base": _BASE,
            "axes": {
                "temperature": temperatures,
                "bt_ratio": ratios,
                "tag_placement": placements,
            },
            "seeds": seeds,
            "max_variants": 1000,
        }
    )

    variants = expand_variants(matrix)

    expected = list(itertools.product(temperatures, ratios, placements, seeds))
    assert [
        (v.temperature, v.bt_ratio, v.tag_placement, v.seed) for v in variants
    ] == expected
    assert len({variant.id for variant in variants}) == len(variants)
