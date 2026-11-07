"""Conversion metrics: edit distance, CER, word accuracy and length ratios."""

from __future__ import annotations

import pytest

from rinchen.script_conv.eval import (
    ConversionScores,
    cer,
    edit_distance,
    length_ratio_quantiles,
    score_conversions,
    word_accuracy,
)


def test_edit_distance_to_the_empty_string_is_the_length() -> None:
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "") == 0


def test_edit_distance_of_the_textbook_pair() -> None:
    assert edit_distance("kitten", "sitting") == 3


def test_edit_distance_is_symmetric() -> None:
    assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")


def test_edit_distance_of_equal_strings_is_zero() -> None:
    assert edit_distance("Монгол", "Монгол") == 0


def test_edit_distance_counts_code_points_not_bytes() -> None:
    latin_o = chr(0x006F)  # looks like the Cyrillic letter in "Монгол"
    assert edit_distance("Монгол", "Монг" + latin_o + "л") == 1
    assert edit_distance(chr(0x1820), chr(0x1821)) == 1
    assert edit_distance(chr(0x1820) * 3, "") == 3


def test_edit_distance_of_a_substitution_an_insertion_and_a_deletion() -> None:
    assert edit_distance("abcd", "abxd") == 1
    assert edit_distance("abcd", "abd") == 1
    assert edit_distance("abd", "abcd") == 1


def test_cer_of_identical_text_is_zero() -> None:
    assert cer(["abc"], ["abc"]) == 0.0


def test_cer_is_edits_over_reference_characters() -> None:
    assert cer(["abcd"], ["abxd"]) == 0.25


def test_cer_over_several_pairs_is_total_edits_over_total_characters() -> None:
    references = ["ab", "abcdefgh"]
    hypotheses = ["xy", "abcdefgh"]
    # 2 edits over 10 characters; the mean of the ratios would be 0.5.
    assert cer(references, hypotheses) == pytest.approx(0.2)


def test_cer_can_exceed_one_when_the_hypothesis_is_long() -> None:
    assert cer(["a"], ["abcd"]) == 3.0


def test_cer_accepts_an_empty_reference_beside_a_longer_one() -> None:
    assert cer(["", "abcd"], ["xy", "abcd"]) == 0.5


def test_cer_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="empty"):
        cer([], [])


def test_cer_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="2 references but 1 hypotheses"):
        cer(["a", "b"], ["a"])


def test_cer_rejects_a_reference_set_with_no_characters() -> None:
    with pytest.raises(ValueError, match="no characters"):
        cer(["", ""], ["", "a"])


def test_word_accuracy_counts_exact_matches() -> None:
    assert word_accuracy(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5


def test_word_accuracy_of_empty_strings_that_match_is_exact() -> None:
    assert word_accuracy(["", "a"], ["", "a"]) == 1.0


def test_word_accuracy_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="empty"):
        word_accuracy([], [])


def test_word_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="1 references but 2 hypotheses"):
        word_accuracy(["a"], ["a", "b"])


def test_score_conversions_bundles_cer_accuracy_and_count() -> None:
    scores = score_conversions(["abcd", "xy", "pq"], ["abxd", "xy", "pq"])
    assert scores == ConversionScores(cer=0.125, word_accuracy=2 / 3, n=3)


def test_score_conversions_rejects_what_cer_rejects() -> None:
    with pytest.raises(ValueError):
        score_conversions([], [])
    with pytest.raises(ValueError):
        score_conversions(["a"], [])
    with pytest.raises(ValueError):
        score_conversions([""], [""])


def test_length_ratio_quantiles_on_a_planted_list() -> None:
    # Script length over Cyrillic length: 0.5, 1, 1.5, 2, 2.5.
    pairs = [("ab", "x"), ("ab", "xy"), ("ab", "xyz"), ("ab", "xyzw"), ("ab", "xyzwv")]
    assert length_ratio_quantiles(pairs, [0.0, 0.5, 1.0]) == {
        0.0: 0.5,
        0.5: 1.5,
        1.0: 2.5,
    }


def test_length_ratio_quantiles_interpolate_linearly() -> None:
    pairs = [("ab", "x"), ("ab", "xy"), ("ab", "xyz"), ("ab", "xyzw"), ("ab", "xyzwv")]
    result = length_ratio_quantiles(pairs, [0.1, 0.25, 0.9])
    assert result[0.1] == pytest.approx(0.7)
    assert result[0.25] == pytest.approx(1.0)
    assert result[0.9] == pytest.approx(2.3)


def test_length_ratio_quantiles_divide_script_length_by_cyrillic_length() -> None:
    assert length_ratio_quantiles([("abcd", "xy")], [0.5]) == {0.5: 0.5}


def test_length_ratio_quantiles_of_one_pair_are_that_ratio() -> None:
    assert length_ratio_quantiles([("abc", "abcdef")], [0.0, 0.3, 1.0]) == {
        0.0: 2.0,
        0.3: 2.0,
        1.0: 2.0,
    }


def test_length_ratio_quantiles_are_floats() -> None:
    result = length_ratio_quantiles([("ab", "xyz")], [0.5])
    assert all(isinstance(value, float) for value in result.values())


def test_length_ratio_quantiles_reject_empty_pairs() -> None:
    with pytest.raises(ValueError, match="no pairs"):
        length_ratio_quantiles([], [0.5])


def test_length_ratio_quantiles_reject_a_zero_length_cyrillic_side() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        length_ratio_quantiles([("ab", "x"), ("", "y")], [0.5])


@pytest.mark.parametrize("quantile", [-0.01, 1.01, float("nan")])
def test_length_ratio_quantiles_reject_a_quantile_outside_zero_to_one(
    quantile: float,
) -> None:
    with pytest.raises(ValueError, match="quantile"):
        length_ratio_quantiles([("ab", "x")], [0.5, quantile])
