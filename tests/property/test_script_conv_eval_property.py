"""Property tests for the conversion metrics."""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from rinchen.script_conv.eval import (
    cer,
    edit_distance,
    length_ratio_quantiles,
    score_conversions,
    word_accuracy,
)

# A small alphabet makes equal and near-equal strings likely.
_SMALL_TEXT = st.text(alphabet="abc", max_size=12)
_PAIRED_TEXT = st.lists(st.tuples(_SMALL_TEXT, _SMALL_TEXT), min_size=1, max_size=6)


@given(st.text(max_size=20), st.text(max_size=20))
def test_edit_distance_is_symmetric(a: str, b: str) -> None:
    assert edit_distance(a, b) == edit_distance(b, a)


@given(st.text(max_size=20), st.text(max_size=20))
def test_edit_distance_is_at_most_the_longer_length(a: str, b: str) -> None:
    assert edit_distance(a, b) <= max(len(a), len(b))


@given(st.text(max_size=20), st.text(max_size=20))
def test_edit_distance_is_at_least_the_length_difference(a: str, b: str) -> None:
    assert edit_distance(a, b) >= abs(len(a) - len(b))


@given(_SMALL_TEXT, _SMALL_TEXT)
def test_edit_distance_is_zero_only_for_equal_strings(a: str, b: str) -> None:
    assert (edit_distance(a, b) == 0) == (a == b)


@given(_SMALL_TEXT, _SMALL_TEXT, _SMALL_TEXT)
def test_edit_distance_obeys_the_triangle_inequality(a: str, b: str, c: str) -> None:
    assert edit_distance(a, c) <= edit_distance(a, b) + edit_distance(b, c)


@given(_PAIRED_TEXT)
def test_cer_is_total_edits_over_total_reference_characters(
    pairs: list[tuple[str, str]],
) -> None:
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    total_characters = sum(len(reference) for reference in references)
    assume(total_characters > 0)
    total_edits = sum(edit_distance(r, h) for r, h in pairs)
    assert cer(references, hypotheses) == total_edits / total_characters


@given(st.lists(_SMALL_TEXT, min_size=1, max_size=6))
def test_converting_every_word_correctly_scores_perfectly(words: list[str]) -> None:
    assume(any(words))
    scores = score_conversions(words, words)
    assert scores.cer == 0.0
    assert scores.word_accuracy == 1.0
    assert scores.n == len(words)


@given(_PAIRED_TEXT)
def test_word_accuracy_is_a_share_between_zero_and_one(
    pairs: list[tuple[str, str]],
) -> None:
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    exact = sum(reference == hypothesis for reference, hypothesis in pairs)
    assert word_accuracy(references, hypotheses) == exact / len(pairs)


_LENGTH_PAIRS = st.lists(
    st.tuples(st.text(min_size=1, max_size=8), st.text(max_size=12)),
    min_size=1,
    max_size=8,
)
_QUANTILES = st.lists(
    st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=5, unique=True
)


@given(_LENGTH_PAIRS, _QUANTILES)
def test_length_ratio_quantiles_stay_within_the_observed_ratios(
    pairs: list[tuple[str, str]], quantiles: list[float]
) -> None:
    ratios = [len(script) / len(cyrillic) for cyrillic, script in pairs]
    result = length_ratio_quantiles(pairs, quantiles)
    assert set(result) == set(quantiles)
    assert all(min(ratios) <= value <= max(ratios) for value in result.values())


@given(_LENGTH_PAIRS)
def test_length_ratio_quantiles_do_not_decrease_with_the_quantile(
    pairs: list[tuple[str, str]],
) -> None:
    quantiles = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    values = [length_ratio_quantiles(pairs, quantiles)[q] for q in quantiles]
    assert values == sorted(values)


@given(_LENGTH_PAIRS, st.floats(min_value=1.0001, max_value=10.0))
def test_length_ratio_quantiles_reject_a_quantile_above_one(
    pairs: list[tuple[str, str]], quantile: float
) -> None:
    with pytest.raises(ValueError):
        length_ratio_quantiles(pairs, [quantile])
