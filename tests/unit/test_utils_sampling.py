from __future__ import annotations

import pytest

from rinchen.utils.sampling import temperature_weights


def test_weights_sum_to_one() -> None:
    assert sum(temperature_weights([10, 1000, 50], 0.3)) == pytest.approx(1.0)


def test_temperature_one_is_proportional() -> None:
    assert temperature_weights([100, 300], 1.0) == pytest.approx([0.25, 0.75])


def test_low_temperature_gives_small_corpus_more_share() -> None:
    sizes = [10, 1000]

    assert temperature_weights(sizes, 0.3)[0] > temperature_weights(sizes, 1.0)[0]


def test_very_low_temperature_approaches_uniform() -> None:
    weights = temperature_weights([10, 1000], 0.01)

    assert all(abs(weight - 0.5) < 0.02 for weight in weights)


def test_high_temperature_exaggerates() -> None:
    sizes = [10, 1000]

    assert temperature_weights(sizes, 2.0)[0] < temperature_weights(sizes, 1.0)[0]


def test_zero_size_gets_zero_weight() -> None:
    assert temperature_weights([0, 5, 5], 0.3) == pytest.approx([0.0, 0.5, 0.5])


def test_empty_returns_empty() -> None:
    assert temperature_weights([], 0.3) == []


@pytest.mark.parametrize(
    ("sizes", "temperature", "problem"),
    [
        ([1, -1], 0.3, "non-negative"),
        ([1, 2], 0.0, "temperature"),
        ([1, 2], -0.5, "temperature"),
        ([0, 0], 0.3, "non-zero"),
    ],
)
def test_invalid_inputs_raise_value_error(
    sizes: list[int], temperature: float, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        temperature_weights(sizes, temperature)
