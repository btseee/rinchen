"""Tests for `rinchen.train.sampling`: `mixing_probabilities` (the temperature acts on
the directions, the corpora of a direction share its probability by their rows) and
`integer_weights` (the probabilities as Eole's whole-number corpus weights).

Every expected number below is worked out by hand in the comment of its test, from
the rule and not from the code. The properties over random inputs are in
`tests/property/test_train_sampling_property.py`.
"""

from __future__ import annotations

import math

import pytest

from rinchen.errors import ConfigError
from rinchen.train.sampling import CorpusSize, integer_weights, mixing_probabilities


def _single(name: str, rows: int) -> CorpusSize:
    """A corpus that is the only one of its direction."""
    return CorpusSize(name=name, direction=name, rows=rows)


# --- mixing_probabilities ---------------------------------------------------------


def test_temperature_one_gives_each_direction_its_share_of_the_rows() -> None:
    # 100 and 400 rows: 100/500 and 400/500.
    probabilities = mixing_probabilities([_single("en", 100), _single("ru", 400)], 1.0)

    assert probabilities == pytest.approx({"en": 0.2, "ru": 0.8})


def test_temperature_one_half_takes_the_square_root_of_each_direction() -> None:
    # sqrt(100) = 10 and sqrt(400) = 20: 10/30 and 20/30.
    probabilities = mixing_probabilities([_single("en", 100), _single("ru", 400)], 0.5)

    assert probabilities == pytest.approx({"en": 1 / 3, "ru": 2 / 3})


def test_corpora_of_a_direction_split_its_probability_in_proportion_to_their_rows() -> (
    None
):
    corpora = [
        CorpusSize("en-mn.auth", "en-mn", 100),
        CorpusSize("en-mn.bt", "en-mn", 300),
        _single("ru-mn.auth", 100),
    ]

    probabilities = mixing_probabilities(corpora, 1.0)

    # At T=1 the rows decide: 100/500, 300/500, 100/500.
    assert probabilities == pytest.approx(
        {"en-mn.auth": 0.2, "en-mn.bt": 0.6, "ru-mn.auth": 0.2}
    )


def test_the_temperature_acts_on_the_direction_total_and_not_on_each_corpus() -> None:
    corpora = [
        CorpusSize("en-mn.auth", "en-mn", 100),
        CorpusSize("en-mn.bt", "en-mn", 300),
        _single("ru-mn.auth", 100),
    ]

    probabilities = mixing_probabilities(corpora, 0.5)

    # The directions weigh sqrt(400) = 20 and sqrt(100) = 10, so en-mn gets 2/3 and
    # ru-mn 1/3; en-mn's 2/3 goes 1:3 to its two corpora: 1/6 and 1/2. Smoothing each
    # corpus alone would give 10, 17.3, 10 -> 0.27, 0.46, 0.27.
    assert probabilities == pytest.approx(
        {"en-mn.auth": 1 / 6, "en-mn.bt": 1 / 2, "ru-mn.auth": 1 / 3}
    )


def test_a_single_corpus_gets_the_whole_probability() -> None:
    assert mixing_probabilities([_single("en", 7)], 0.3) == {"en": 1.0}


def test_the_probabilities_keep_the_order_of_the_corpora() -> None:
    corpora = [_single("b", 5), _single("a", 5), _single("c", 5)]

    assert list(mixing_probabilities(corpora, 0.7)) == ["b", "a", "c"]


def test_a_smaller_temperature_lifts_the_smallest_direction() -> None:
    corpora = [_single("small", 10), _single("large", 1000)]

    at_one = mixing_probabilities(corpora, 1.0)["small"]
    at_low = mixing_probabilities(corpora, 0.3)["small"]

    # 10/1010 is about 0.0099; 10**0.3 / (10**0.3 + 1000**0.3) = 1/(1+100**0.3) is
    # about 0.2008.
    assert at_one == pytest.approx(10 / 1010)
    assert at_low == pytest.approx(1 / (1 + 100**0.3))


def test_the_input_order_changes_no_probability_by_a_single_bit() -> None:
    corpora = [
        CorpusSize("a.auth", "a", 123_457),
        CorpusSize("a.bt", "a", 61_729),
        _single("b", 3_000_001),
        _single("c", 999_999),
        _single("d", 17),
    ]

    forward = mixing_probabilities(corpora, 0.37)
    backward = mixing_probabilities(corpora[::-1], 0.37)

    assert forward == backward


def test_the_spikes_five_directions_at_temperature_0_3() -> None:
    corpora = [
        _single("en2mn.auth", 1_000_000),
        _single("en2mn.bt", 8_000_000),
        _single("mn2en", 1_000_000),
        _single("ru2mn", 600_000),
        _single("mn2ru", 600_000),
    ]

    probabilities = mixing_probabilities(corpora, 0.3)

    # size**0.3 in millions: 1, 2**0.9 = 1.866066, 1, 0.6**0.3 = 0.857955 (twice);
    # their sum is 5.581976.
    assert probabilities == pytest.approx(
        {
            "en2mn.auth": 0.179150,
            "en2mn.bt": 0.334307,
            "mn2en": 0.179150,
            "ru2mn": 0.153696,
            "mn2ru": 0.153696,
        },
        abs=1e-6,
    )


@pytest.mark.parametrize("rows", [0, -1])
def test_a_corpus_without_rows_is_an_error(rows: int) -> None:
    with pytest.raises(ValueError, match="empty.auth.*rows"):
        mixing_probabilities([_single("a", 5), _single("empty.auth", rows)], 1.0)


@pytest.mark.parametrize("temperature", [0.0, -0.5, math.nan, math.inf, -math.inf])
def test_a_temperature_that_is_not_a_positive_finite_number_is_an_error(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="temperature must be a positive finite"):
        mixing_probabilities([_single("a", 5), _single("b", 6)], temperature)


def test_a_temperature_too_large_for_the_sizes_is_an_error_and_not_an_overflow() -> (
    None
):
    # 10**9 ** 100 does not fit a float.
    with pytest.raises(ValueError, match="temperature 100.0"):
        mixing_probabilities([_single("a", 1), _single("b", 10**9)], 100.0)


def test_a_temperature_that_sums_past_the_float_range_is_an_error_and_not_zeros() -> (
    None
):
    # Each direction weighs (10**154) ** 2 = 1e308; the sum is infinite and each
    # share would be 0.0.
    with pytest.raises(ValueError, match="temperature 2.0"):
        mixing_probabilities([_single("a", 10**154), _single("b", 10**154)], 2.0)


def test_a_tiny_temperature_gives_every_direction_the_same_share() -> None:
    probabilities = mixing_probabilities([_single("a", 1), _single("b", 10**9)], 1e-12)

    assert probabilities == pytest.approx({"a": 0.5, "b": 0.5})


def test_two_corpora_with_one_name_are_an_error() -> None:
    with pytest.raises(ValueError, match="en-mn.auth.*twice|duplicate.*en-mn.auth"):
        mixing_probabilities(
            [CorpusSize("en-mn.auth", "en-mn", 5), CorpusSize("en-mn.auth", "xx", 5)],
            1.0,
        )


def test_no_corpus_at_all_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one corpus"):
        mixing_probabilities([], 1.0)


# --- integer_weights --------------------------------------------------------------


def test_the_spikes_example_gets_weights_that_sum_to_one_hundred() -> None:
    probabilities = {
        "en2mn.auth": 0.1791504561552639,
        "en2mn.bt": 0.33430657208345904,
        "mn2en": 0.1791504561552639,
        "ru2mn": 0.15369625780300658,
        "mn2ru": 0.15369625780300658,
    }

    # Times 100: 17.915, 33.431, 17.915, 15.370, 15.370. The floors 17, 33, 17, 15, 15
    # sum to 97, so the three largest remainders (.915, .915, .431) get one more:
    # 18, 34, 18, 15, 15. (The spike's 7/13/7/6/6 is the smallest vector near these
    # shares, a round of 39; the contract here is that the weights sum to the total.)
    assert integer_weights(probabilities) == {
        "en2mn.auth": 18,
        "en2mn.bt": 34,
        "mn2en": 18,
        "ru2mn": 15,
        "mn2ru": 15,
    }


def test_a_probability_of_a_billionth_still_gets_a_weight_of_one() -> None:
    weights = integer_weights({"big": 1 - 1e-9, "tiny": 1e-9})

    # The minimum of 1 for "tiny" leaves 99 for "big" (its ideal is 99.9999999).
    assert weights == {"big": 99, "tiny": 1}


def test_a_surplus_from_the_minimum_is_taken_from_the_largest_weight() -> None:
    probabilities = {"a": 0.6, "b": 0.396, "c": 0.002, "d": 0.002}

    # Times 100: 60, 39.6, 0.2, 0.2. The floors with a minimum of 1 are 60, 39, 1, 1,
    # which sum to 101: one is taken from the largest, 60 -> 59.
    assert integer_weights(probabilities) == {"a": 59, "b": 39, "c": 1, "d": 1}


def test_a_surplus_can_take_the_largest_weight_down_to_one_and_no_further() -> None:
    probabilities = {"a": 0.96, "b": 0.01, "c": 0.01, "d": 0.01, "e": 0.01}

    # Times 5: 4.8 and four times 0.05. The floors with a minimum of 1: 4, 1, 1, 1, 1
    # = 8, three too many; "a" gives up 4 -> 3 -> 2 -> 1 and the others stay at 1.
    assert integer_weights(probabilities, total=5) == dict.fromkeys("abcde", 1)


def test_the_surplus_is_taken_from_the_smaller_probability_of_two_equal_weights() -> (
    None
):
    probabilities = {"a": 0.46, "b": 0.44, "c": 0.05, "d": 0.05}

    # Times 5: 2.3, 2.2, 0.25, 0.25. The floors with a minimum of 1: 2, 2, 1, 1 = 6,
    # one too many. a and b tie at 2; taking from a would give a < b although
    # p(a) > p(b), so b loses one.
    assert integer_weights(probabilities, total=5) == {"a": 2, "b": 1, "c": 1, "d": 1}


def test_a_surplus_between_equal_probabilities_is_taken_from_the_later_name() -> None:
    probabilities = {"a": 0.49, "b": 0.49, "c": 0.01, "d": 0.01}

    # As above with equal probabilities: 2, 2, 1, 1 -> the later of the two, b, loses.
    assert integer_weights(probabilities, total=5) == {"a": 2, "b": 1, "c": 1, "d": 1}


def test_the_remainders_are_given_out_largest_first() -> None:
    probabilities = {"a": 0.55, "b": 0.35, "c": 0.10}

    # Times 7: 3.85, 2.45, 0.7. Floors with a minimum of 1: 3, 2, 1 = 6, so one more
    # goes to the largest remainder (.85, of a).
    assert integer_weights(probabilities, total=7) == {"a": 4, "b": 2, "c": 1}


def test_equal_remainders_go_to_the_earlier_names_whatever_the_input_order() -> None:
    third = 1 / 3

    forward = integer_weights({"a": third, "b": third, "c": third})
    backward = integer_weights({"c": third, "b": third, "a": third})

    # 33.33 each: floors 33 x 3 = 99, the one left over goes to the first name.
    assert forward == backward == {"a": 34, "b": 33, "c": 33}


def test_the_total_can_be_chosen() -> None:
    # Times 10: 2.5 and 7.5; floors 2 and 7; the tie of the remainders goes to "a".
    assert integer_weights({"a": 0.25, "b": 0.75}, total=10) == {"a": 3, "b": 7}


def test_a_single_corpus_takes_the_whole_total() -> None:
    assert integer_weights({"only": 1.0}) == {"only": 100}
    assert integer_weights({"only": 1.0}, total=7) == {"only": 7}


def test_as_many_corpora_as_the_total_all_get_one() -> None:
    probabilities = {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.05, "e": 0.05}

    assert integer_weights(probabilities, total=5) == dict.fromkeys("abcde", 1)


def test_probabilities_that_are_multiples_of_one_over_the_total_come_back_exactly() -> (
    None
):
    # 0.29 * 100 is 28.999999999999996 in floats; the weight is still 29.
    probabilities = {"a": 0.29, "b": 0.07, "c": 0.41, "d": 0.23}

    assert integer_weights(probabilities) == {"a": 29, "b": 7, "c": 41, "d": 23}


def test_a_larger_probability_never_gets_a_smaller_weight() -> None:
    probabilities = {"z": 0.5, "y": 0.3, "x": 0.2}

    weights = integer_weights(probabilities)

    assert weights == {"z": 50, "y": 30, "x": 20}


def test_more_corpora_than_the_total_is_a_config_error_that_names_the_number_needed() -> (
    None
):
    probabilities = dict.fromkeys("abcde", 0.2)

    with pytest.raises(ConfigError, match=r"weight_total 3.*at least 5"):
        integer_weights(probabilities, total=3)


def test_a_total_below_one_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"weight_total 0.*at least 1"):
        integer_weights({"a": 1.0}, total=0)


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
def test_a_probability_that_is_not_positive_and_finite_is_an_error(bad: float) -> None:
    with pytest.raises(ValueError, match="probability.*'b'"):
        integer_weights({"a": 0.5, "b": bad})


@pytest.mark.parametrize("second", [0.4, 0.6])
def test_probabilities_that_do_not_sum_to_one_are_an_error(second: float) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        integer_weights({"a": 0.5, "b": second})


def test_probabilities_that_sum_to_one_within_a_billionth_are_accepted() -> None:
    weights = integer_weights({"a": 0.5, "b": 0.5 + 5e-10})

    assert weights == {"a": 50, "b": 50}


def test_probabilities_that_sum_to_one_plus_two_billionths_are_an_error() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        integer_weights({"a": 0.5, "b": 0.5 + 2e-9})


def test_no_probability_at_all_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        integer_weights({})
