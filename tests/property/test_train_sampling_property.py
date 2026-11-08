"""Property tests for `rinchen.train.sampling` over random corpus sizes, temperatures
and probabilities: the probabilities are a distribution that smooths with the
temperature, and the integer weights keep their promises (at least 1, the sum is the
total, a larger probability never gets a smaller weight, close to the probabilities,
independent of the order the corpora are given in)."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rinchen.errors import ConfigError
from rinchen.train.sampling import CorpusSize, integer_weights, mixing_probabilities

_TEMPERATURES = st.floats(min_value=0.05, max_value=3.0)
_ROWS = st.integers(min_value=1, max_value=10**9)


@st.composite
def _corpora(draw: st.DrawFn) -> list[CorpusSize]:
    """Corpora with unique names spread over one to four directions."""
    rows = draw(st.lists(_ROWS, min_size=1, max_size=8))
    directions = draw(
        st.lists(
            st.sampled_from(["d0", "d1", "d2", "d3"]),
            min_size=len(rows),
            max_size=len(rows),
        )
    )
    return [
        CorpusSize(name=f"c{index}", direction=direction, rows=count)
        for index, (direction, count) in enumerate(zip(directions, rows, strict=True))
    ]


@st.composite
def _distributions(draw: st.DrawFn) -> dict[str, float]:
    """Positive probabilities summing to 1 (to rounding) under unique names."""
    raw = draw(
        st.lists(st.floats(min_value=1e-9, max_value=1.0), min_size=1, max_size=12)
    )
    total = sum(raw)
    return {f"c{index}": value / total for index, value in enumerate(raw)}


@st.composite
def _balanced_distributions(draw: st.DrawFn) -> dict[str, float]:
    """Probabilities none of which is below 1/36 (the largest is at most 5 times the
    smallest, over at most 8 corpora), so that a total of 36 or more gives every
    corpus a share of at least 1."""
    raw = draw(
        st.lists(st.floats(min_value=1.0, max_value=5.0), min_size=1, max_size=8)
    )
    total = sum(raw)
    return {f"c{index}": value / total for index, value in enumerate(raw)}


def _directions_total(corpora: list[CorpusSize]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for corpus in corpora:
        totals[corpus.direction] = totals.get(corpus.direction, 0) + corpus.rows
    return totals


@given(corpora=_corpora(), temperature=_TEMPERATURES)
def test_mixing_probabilities_are_positive_and_sum_to_one(
    corpora: list[CorpusSize], temperature: float
) -> None:
    probabilities = mixing_probabilities(corpora, temperature)

    assert set(probabilities) == {corpus.name for corpus in corpora}
    assert all(value > 0 for value in probabilities.values())
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)


@given(corpora=_corpora())
def test_temperature_one_is_proportional_to_the_rows(
    corpora: list[CorpusSize],
) -> None:
    probabilities = mixing_probabilities(corpora, 1.0)

    everything = sum(corpus.rows for corpus in corpora)
    for corpus in corpora:
        assert probabilities[corpus.name] == pytest.approx(
            corpus.rows / everything, rel=1e-9
        )


@given(corpora=_corpora(), temperature=_TEMPERATURES)
def test_corpora_of_one_direction_keep_the_ratio_of_their_rows(
    corpora: list[CorpusSize], temperature: float
) -> None:
    probabilities = mixing_probabilities(corpora, temperature)

    for first in corpora:
        for second in corpora:
            if first.direction == second.direction:
                assert probabilities[first.name] * second.rows == pytest.approx(
                    probabilities[second.name] * first.rows, rel=1e-9
                )


@given(
    corpora=_corpora(),
    low=st.floats(min_value=0.05, max_value=1.5),
    step=st.floats(min_value=0.0, max_value=1.5),
)
def test_a_smaller_temperature_never_lowers_the_smallest_directions_share(
    corpora: list[CorpusSize], low: float, step: float
) -> None:
    totals = _directions_total(corpora)
    smallest = min(totals, key=lambda direction: (totals[direction], direction))

    def share(temperature: float) -> float:
        probabilities = mixing_probabilities(corpora, temperature)
        return sum(
            probabilities[corpus.name]
            for corpus in corpora
            if corpus.direction == smallest
        )

    assert share(low) >= share(low + step) - 1e-12


@given(
    corpora=_corpora(),
    temperature=_TEMPERATURES,
    rng=st.randoms(use_true_random=False),
)
def test_the_order_of_the_corpora_changes_no_probability(
    corpora: list[CorpusSize], temperature: float, rng: random.Random
) -> None:
    shuffled = list(corpora)
    rng.shuffle(shuffled)

    assert mixing_probabilities(corpora, temperature) == mixing_probabilities(
        shuffled, temperature
    )


@given(probabilities=_distributions(), total=st.integers(min_value=1, max_value=300))
def test_every_weight_is_at_least_one_and_the_sum_is_the_total(
    probabilities: dict[str, float], total: int
) -> None:
    if len(probabilities) > total:
        with pytest.raises(ConfigError, match=f"at least {len(probabilities)}"):
            integer_weights(probabilities, total=total)
        return

    weights = integer_weights(probabilities, total=total)

    assert set(weights) == set(probabilities)
    assert min(weights.values()) >= 1
    assert sum(weights.values()) == total


@given(probabilities=_distributions(), total=st.integers(min_value=12, max_value=300))
def test_a_larger_probability_never_gets_a_smaller_weight(
    probabilities: dict[str, float], total: int
) -> None:
    weights = integer_weights(probabilities, total=total)

    for first, p_first in probabilities.items():
        for second, p_second in probabilities.items():
            if p_first > p_second:
                assert weights[first] >= weights[second], (first, second, weights)


@given(
    probabilities=_balanced_distributions(),
    total=st.integers(min_value=36, max_value=300),
)
def test_a_weight_is_within_one_of_its_share_when_no_share_is_below_one(
    probabilities: dict[str, float], total: int
) -> None:
    weights = integer_weights(probabilities, total=total)

    for name, value in probabilities.items():
        assert abs(weights[name] - total * value) < 1 + 1e-9, (name, weights)


@given(
    probabilities=_distributions(),
    total=st.integers(min_value=12, max_value=300),
    rng=st.randoms(use_true_random=False),
)
def test_the_order_the_probabilities_are_given_in_changes_no_weight(
    probabilities: dict[str, float], total: int, rng: random.Random
) -> None:
    names = list(probabilities)
    rng.shuffle(names)
    shuffled = {name: probabilities[name] for name in names}

    assert integer_weights(probabilities, total=total) == integer_weights(
        shuffled, total=total
    )


@given(
    counts=st.lists(st.integers(min_value=1, max_value=60), min_size=1, max_size=10),
)
def test_probabilities_that_are_multiples_of_one_over_the_total_are_returned_exactly(
    counts: list[int],
) -> None:
    total = sum(counts)
    probabilities = {f"c{index}": count / total for index, count in enumerate(counts)}

    weights = integer_weights(probabilities, total=total)

    assert weights == {f"c{index}": count for index, count in enumerate(counts)}


@given(corpora=_corpora(), temperature=_TEMPERATURES)
def test_the_weights_of_mixing_probabilities_sum_to_the_total(
    corpora: list[CorpusSize], temperature: float
) -> None:
    probabilities = mixing_probabilities(corpora, temperature)

    weights = integer_weights(probabilities, total=100)

    assert sum(weights.values()) == 100
    assert min(weights.values()) >= 1
