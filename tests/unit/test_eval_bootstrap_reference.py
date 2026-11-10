"""The bootstrap against an independent pure-Python reference (D13, T9).

The functions below are a FROZEN COPY of the loops of the design spike (`boot.py`): no numpy, one
Python integer at a time, the index rule written out as the spec states it. They are the
independent derivation the golden literals of `tests/support/eval_golden.py` come from, so they
are never imported from `rinchen`; if the implementation and this copy disagree on any input,
the implementation is wrong.

The index of position `j` of resample `b` over `n` units is `mix(seed + (b * n + j + 1) * GOLDEN)`,
its high 32 bits times `n`, shifted right by 32, every operation modulo 2^64. `mix` is splitmix64's
finaliser, which is why the first three outputs for seed 0 are the published splitmix64 vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.eval.bootstrap import (
    BootstrapSettings,
    draw_index,
    resample_indices,
    resample_totals,
)
from tests.support.eval_golden import (
    GOLDEN_ALPHA,
    GOLDEN_POSITIONS,
    GOLDEN_RESAMPLES,
    GOLDEN_SEED,
    MEAN_MICRO_A,
    MEAN_MICRO_B,
    MEAN_SUMS_A,
    MEAN_SUMS_A_MINUS_B,
    MEAN_SUMS_B,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
PUBLISHED_WORDS = (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F)


def mix_py(z: int) -> int:
    z &= MASK
    z = ((z ^ (z >> 30)) * C1) & MASK
    z = ((z ^ (z >> 27)) * C2) & MASK
    return z ^ (z >> 31)


def index_py(seed: int, b: int, j: int, n: int) -> int:
    word = mix_py(seed + (b * n + j + 1) * GOLDEN)
    return ((word >> 32) * n) >> 32


def resample_sums_py(
    stats: Sequence[Sequence[int]], seed: int, resamples: int
) -> list[list[int]]:
    """The sums of the drawn rows of every resample, one Python integer at a time."""
    n, k = len(stats), len(stats[0])
    out = []
    for b in range(resamples):
        acc = [0] * k
        for j in range(n):
            row = stats[index_py(seed, b, j, n)]
            for c in range(k):
                acc[c] += row[c]
        out.append(acc)
    return out


def column(rows: np.ndarray, index: int) -> list[int]:
    return [int(value) for value in rows[:, index]]


SEEDS = st.one_of(
    st.just(0),
    st.just(GOLDEN_SEED),
    st.just(MASK),
    st.integers(min_value=0, max_value=MASK),
)


def test_the_reference_reproduces_the_published_splitmix64_outputs() -> None:
    assert [mix_py(step * GOLDEN) for step in (1, 2, 3)] == list(PUBLISHED_WORDS)


@SETTINGS
@given(
    seed=SEEDS,
    resample=st.integers(min_value=0, max_value=10**9),
    units=st.integers(min_value=1, max_value=2**32),
    data=st.data(),
)
def test_draw_index_is_the_reference_index(
    seed: int, resample: int, units: int, data: st.DataObject
) -> None:
    position = data.draw(st.integers(min_value=0, max_value=min(units - 1, 10**6)))

    assert draw_index(seed, resample, position, units) == index_py(
        seed, resample, position, units
    )


@SETTINGS
@given(
    seed=SEEDS,
    first=st.integers(min_value=0, max_value=50),
    count=st.integers(min_value=0, max_value=6),
    units=st.integers(min_value=1, max_value=60),
)
def test_resample_indices_are_the_reference_indices(
    seed: int, first: int, count: int, units: int
) -> None:
    drawn = resample_indices(seed, first, first + count, units)

    expected = [
        [index_py(seed, b, j, units) for j in range(units)]
        for b in range(first, first + count)
    ]
    assert drawn.tolist() == expected


@SETTINGS
@given(
    seed=SEEDS,
    resamples=st.integers(min_value=100, max_value=200),
    data=st.data(),
)
def test_resample_totals_are_the_reference_sums_on_random_integer_rows(
    seed: int, resamples: int, data: st.DataObject
) -> None:
    units = data.draw(st.integers(min_value=1, max_value=40))
    width = data.draw(st.integers(min_value=1, max_value=5))
    rows = data.draw(
        st.lists(
            st.lists(
                st.integers(min_value=-(10**9), max_value=10**9),
                min_size=width,
                max_size=width,
            ),
            min_size=units,
            max_size=units,
        )
    )

    totals = resample_totals(
        np.array(rows, dtype=np.int64),
        BootstrapSettings(resamples=resamples, seed=seed),
    )

    assert totals.tolist() == resample_sums_py(rows, seed, resamples)


def mean_rows(micro: Sequence[int]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


def test_the_golden_mean_metric_sums_are_the_reference_sums_and_the_literals() -> None:
    settings_ = BootstrapSettings(
        resamples=GOLDEN_RESAMPLES, seed=GOLDEN_SEED, alpha=GOLDEN_ALPHA
    )
    low, high = GOLDEN_POSITIONS

    sums_a = resample_totals(mean_rows(MEAN_MICRO_A), settings_)
    sums_b = resample_totals(mean_rows(MEAN_MICRO_B), settings_)

    reference = resample_sums_py(mean_rows(MEAN_MICRO_A).tolist(), GOLDEN_SEED, 1000)
    assert sums_a.tolist() == reference
    sorted_a = sorted(column(sums_a, 0))
    sorted_b = sorted(column(sums_b, 0))
    sorted_difference = sorted(
        a - b for a, b in zip(column(sums_a, 0), column(sums_b, 0), strict=True)
    )
    assert (sorted_a[low], sorted_a[high]) == MEAN_SUMS_A
    assert (sorted_b[low], sorted_b[high]) == MEAN_SUMS_B
    assert (sorted_difference[low], sorted_difference[high]) == MEAN_SUMS_A_MINUS_B
