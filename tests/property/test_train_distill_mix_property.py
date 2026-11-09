"""Property tests for `rinchen.train.distill_mix`: whatever positive integers K and O and
whatever original weights, every direction that has a KD corpus draws exactly K of every
K + O examples from it, the directions keep their shares of the draws, and nothing depends
on the order the corpora are listed in. The arithmetic is on `Fraction`s: a float or a
rounding anywhere makes these fail."""

from __future__ import annotations

import math
import random
from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.train.distill_mix import Mix, mixed_weights, parse_mix, repeat_factors

_POSITIVE = st.integers(min_value=1, max_value=10**9)
_WEIGHT = st.integers(min_value=1, max_value=10_000)


@st.composite
def _corpora(draw: st.DrawFn) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    """Original weights of 1 to 6 directions with 1 to 3 corpora each, the direction of
    each original, and a KD corpus for a random subset of the directions (all of them
    for most draws)."""
    directions = [f"d{index}" for index in range(draw(st.integers(1, 6)))]
    original: dict[str, int] = {}
    direction_of: dict[str, str] = {}
    for direction in directions:
        for group in range(draw(st.integers(1, 3))):
            name = f"{direction}.g{group}"
            original[name] = draw(_WEIGHT)
            direction_of[name] = direction
    with_kd = draw(st.one_of(st.just(directions), st.sets(st.sampled_from(directions))))
    kd = {
        f"{direction}.kd": direction for direction in directions if direction in with_kd
    }
    return original, direction_of, kd


def _total_of(direction: str, weights: dict[str, int], names: dict[str, str]) -> int:
    return sum(weights[name] for name, home in names.items() if home == direction)


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora())
def test_every_kd_corpus_is_exactly_k_to_o_against_the_originals_of_its_direction(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
) -> None:
    weights_in, direction_of, kd_corpora = corpora

    weights = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))

    for name, direction in kd_corpora.items():
        scaled = _total_of(direction, weights, direction_of)
        assert Fraction(weights[name], scaled) == Fraction(kd, original)


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora())
def test_every_weight_is_a_positive_integer_and_the_names_are_the_inputs(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
) -> None:
    weights_in, direction_of, kd_corpora = corpora

    weights = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))

    assert set(weights) == set(weights_in) | set(kd_corpora)
    assert all(type(weight) is int and weight >= 1 for weight in weights.values())


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora())
def test_every_direction_keeps_its_share_of_the_draws(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
) -> None:
    weights_in, direction_of, kd_corpora = corpora
    names = {**direction_of, **kd_corpora}

    weights = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))

    before_total = sum(weights_in.values())
    after_total = sum(weights.values())
    for direction in set(direction_of.values()):
        before = Fraction(_total_of(direction, weights_in, direction_of), before_total)
        after = Fraction(_total_of(direction, weights, names), after_total)
        assert after == before


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora())
def test_the_weights_add_up_to_the_reduced_mix_times_the_original_sum(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
) -> None:
    weights_in, direction_of, kd_corpora = corpora
    divisor = math.gcd(kd, original)

    weights = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))

    assert sum(weights.values()) == (kd // divisor + original // divisor) * sum(
        weights_in.values()
    )


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, scale=st.integers(1, 1000), corpora=_corpora())
def test_a_mix_and_its_multiple_give_the_same_weights(
    kd: int,
    original: int,
    scale: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
) -> None:
    weights_in, direction_of, kd_corpora = corpora

    assert mixed_weights(
        weights_in, direction_of, kd_corpora, Mix(kd, original)
    ) == mixed_weights(
        weights_in, direction_of, kd_corpora, Mix(kd * scale, original * scale)
    )


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora(), seed=st.integers(0, 2**32))
def test_the_result_does_not_depend_on_the_order_of_the_mappings(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
    seed: int,
) -> None:
    weights_in, direction_of, kd_corpora = corpora
    rng = random.Random(seed)

    def shuffled[V](mapping: dict[str, V]) -> dict[str, V]:
        items = list(mapping.items())
        rng.shuffle(items)
        return dict(items)

    reference = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))
    reordered = mixed_weights(
        shuffled(weights_in),
        shuffled(direction_of),
        shuffled(kd_corpora),
        Mix(kd, original),
    )

    assert reordered == reference


@settings(max_examples=200)
@given(kd=_POSITIVE, original=_POSITIVE, corpora=_corpora(), rows=st.data())
def test_a_kd_row_is_drawn_k_over_o_times_the_ratio_of_the_row_counts_as_often(
    kd: int,
    original: int,
    corpora: tuple[dict[str, int], dict[str, str], dict[str, str]],
    rows: st.DataObject,
) -> None:
    weights_in, direction_of, kd_corpora = corpora
    counts = {
        name: rows.draw(st.integers(1, 10**6), label=name)
        for name in (*weights_in, *kd_corpora)
    }
    weights = mixed_weights(weights_in, direction_of, kd_corpora, Mix(kd, original))

    factors = repeat_factors(counts, weights, direction_of, kd_corpora)

    assert set(factors) == set(kd_corpora)
    for name, direction in kd_corpora.items():
        original_rows = _total_of(direction, counts, direction_of)
        assert factors[name] == Fraction(kd, original) * Fraction(
            original_rows, counts[name]
        )


@settings(max_examples=200)
@given(kd=st.integers(1, 10_000), original=st.integers(1, 10_000))
def test_a_mix_text_round_trips_through_the_reduced_integers(
    kd: int, original: int
) -> None:
    divisor = math.gcd(kd, original)

    assert parse_mix(f"{kd}:{original}") == Mix(kd // divisor, original // divisor)
