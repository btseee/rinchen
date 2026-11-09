"""Tests for `rinchen.train.distill_mix`: the mix `K:O`, the weights that make exactly K of
every K + O examples of a direction come from the KD corpus, and how much more often a KD row
is drawn than an original one.

The weights are worked out by hand from spec ruling T8 and the toy weights of the spike
(`d10-spike` findings, Q3): the original corpora of the toy at temperature 0.5 and a
back-translation ratio of 0.5 weigh 22 and 6 (en-mn and mn-en), 15 and 7 (mn-ru and ru-mn),
so the KD corpus of a direction weighs the sum of its original weights (28 or 22) for `1:1`.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.train.distill_mix import Mix, mixed_weights, parse_mix, repeat_factors

ORIGINAL = {
    "en-mn.auth": 22,
    "en-mn.bt": 6,
    "mn-en.auth": 22,
    "mn-en.bt": 6,
    "mn-ru.auth": 15,
    "mn-ru.bt": 7,
    "ru-mn.auth": 15,
    "ru-mn.bt": 7,
}
DIRECTION_OF = {name: name.split(".")[0] for name in ORIGINAL}
KD = {
    "en-mn.kd": "en-mn",
    "mn-en.kd": "mn-en",
    "mn-ru.kd": "mn-ru",
    "ru-mn.kd": "ru-mn",
}


def _mixed(mix: Mix, **replacements: Any) -> dict[str, int]:
    arguments: dict[str, Any] = {
        "original": ORIGINAL,
        "direction_of": DIRECTION_OF,
        "kd_corpora": KD,
    } | replacements
    return mixed_weights(
        arguments["original"],
        arguments["direction_of"],
        arguments["kd_corpora"],
        mix,
    )


# --- parse_mix -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kd", "original"),
    [
        ("1:1", 1, 1),
        ("3:1", 3, 1),
        ("1:3", 1, 3),
        ("2:2", 1, 1),
        ("6:4", 3, 2),
        ("10000:10000", 1, 1),
        ("10000:1", 10000, 1),
        ("1:10000", 1, 10000),
        ("9999:10000", 9999, 10000),
    ],
)
def test_a_mix_is_two_integers_reduced_by_their_greatest_common_divisor(
    text: str, kd: int, original: int
) -> None:
    assert parse_mix(text) == Mix(kd, original)


_ARABIC_INDIC_ONE = chr(0x0661)
_FULLWIDTH_ONE = chr(0xFF11)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1",
        ":",
        "1:",
        ":1",
        "0:1",
        "1:0",
        "0:0",
        "-1:1",
        "1:-1",
        "+1:1",
        "1.5:1",
        "1:1.5",
        "1: 1",
        " 1:1",
        "1:1 ",
        "1 :1",
        "1:1\n",
        "1:1:1",
        "1/1",
        "1;1",
        "one:one",
        "01:1",
        "1:01",
        "00:1",
        "10001:1",
        "1:10001",
        "99999:1",
        "1" + "0" * 5000 + ":1",
        f"{_ARABIC_INDIC_ONE}:1",
        f"1:{_ARABIC_INDIC_ONE}",
        f"1{_ARABIC_INDIC_ONE}:1",
        f"1:1{_ARABIC_INDIC_ONE}",
        f"{_FULLWIDTH_ONE}:1",
        "1" + chr(0xFF1A) + "1",
        "1_0:1",
        "0x1:1",
        "1e1:1",
        "None",
    ],
)
def test_a_text_that_is_not_two_positive_integers_of_at_most_10000_is_a_config_error(
    text: str,
) -> None:
    with pytest.raises(ConfigError, match=r"--mix|mix"):
        parse_mix(text)


def test_the_message_of_a_refused_mix_shows_the_text_and_the_rule() -> None:
    with pytest.raises(ConfigError) as caught:
        parse_mix("1.5:1")

    message = str(caught.value)
    assert "'1.5:1'" in message
    assert "K:O" in message
    assert "10000" in message


# --- Mix -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kd", "original"),
    [(0, 1), (1, 0), (-1, 1), (1, -3), (True, 1), (1, True), (1.0, 1), (1, "1")],
)
def test_a_mix_of_anything_but_two_positive_integers_is_a_value_error(
    kd: Any, original: Any
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Mix(kd, original)


def test_a_mix_that_is_not_reduced_can_be_built_and_reduced() -> None:
    assert Mix(6, 4).reduced() == Mix(3, 2)
    assert Mix(3, 2).reduced() == Mix(3, 2)
    assert Mix(7, 7).reduced() == Mix(1, 1)


# --- mixed_weights: the toy weights of the spike ---------------------------------------


def test_one_to_one_keeps_the_originals_and_gives_each_kd_corpus_the_sum_of_its_direction() -> (
    None
):
    weights = _mixed(Mix(1, 1))

    assert weights == {
        **ORIGINAL,
        "en-mn.kd": 28,
        "mn-en.kd": 28,
        "mn-ru.kd": 22,
        "ru-mn.kd": 22,
    }
    assert sum(weights.values()) == 200


def test_three_to_one_triples_the_kd_corpora_and_keeps_the_originals() -> None:
    weights = _mixed(Mix(3, 1))

    assert weights == {
        **ORIGINAL,
        "en-mn.kd": 84,
        "mn-en.kd": 84,
        "mn-ru.kd": 66,
        "ru-mn.kd": 66,
    }
    assert sum(weights.values()) == 400


def test_one_to_three_triples_the_originals_and_keeps_the_kd_corpora() -> None:
    weights = _mixed(Mix(1, 3))

    assert weights == {
        **{name: 3 * weight for name, weight in ORIGINAL.items()},
        "en-mn.kd": 28,
        "mn-en.kd": 28,
        "mn-ru.kd": 22,
        "ru-mn.kd": 22,
    }
    assert sum(weights.values()) == 400


def test_two_to_four_is_one_to_two() -> None:
    weights = _mixed(Mix(2, 4))

    assert weights == _mixed(Mix(1, 2))
    assert weights == {
        **{name: 2 * weight for name, weight in ORIGINAL.items()},
        "en-mn.kd": 28,
        "mn-en.kd": 28,
        "mn-ru.kd": 22,
        "ru-mn.kd": 22,
    }


def test_the_kd_weight_follows_the_originals_of_its_direction_not_the_scaled_ones() -> (
    None
):
    # 3:2 scales the originals by 2; the KD corpus is 3 times the UNSCALED sum (28), so that
    # the KD share is 3 / (3 + 2): 84 of 84 + 2 x 28.
    weights = _mixed(Mix(3, 2))

    assert weights["en-mn.kd"] == 84
    assert weights["en-mn.auth"] == 44
    assert weights["en-mn.bt"] == 12
    assert Fraction(84, 84 + 44 + 12) == Fraction(3, 5)


def test_the_mix_is_reduced_before_it_is_applied() -> None:
    assert _mixed(Mix(5, 5)) == _mixed(Mix(1, 1))
    assert _mixed(Mix(9, 3)) == _mixed(Mix(3, 1))


def test_a_direction_with_no_kd_corpus_keeps_its_share_of_the_draws() -> None:
    # Only en-mn has a KD corpus. mn-ru keeps no KD, but its originals are scaled by
    # K + O = 4 like every direction's total, so its share of the draws stays the
    # temperature's: 88 of 200 here, as 22 of 50 before the mix.
    original = {
        "en-mn.auth": 22,
        "en-mn.bt": 6,
        "mn-ru.auth": 15,
        "mn-ru.bt": 7,
    }
    directions = {name: name.split(".")[0] for name in original}

    weights = mixed_weights(original, directions, {"en-mn.kd": "en-mn"}, Mix(3, 1))

    assert weights == {
        "en-mn.auth": 22,
        "en-mn.bt": 6,
        "en-mn.kd": 84,
        "mn-ru.auth": 60,
        "mn-ru.bt": 28,
    }
    assert Fraction(22 + 6 + 84, sum(weights.values())) == Fraction(28, 28 + 22)


def test_no_kd_corpus_at_all_scales_every_original_by_the_sum_of_the_mix() -> None:
    weights = mixed_weights(
        {"a.auth": 3, "b.auth": 5}, {"a.auth": "a", "b.auth": "b"}, {}, Mix(1, 2)
    )

    assert weights == {"a.auth": 9, "b.auth": 15}


def test_nothing_in_nothing_out() -> None:
    assert mixed_weights({}, {}, {}, Mix(1, 1)) == {}


def test_the_inputs_are_not_changed() -> None:
    original = dict(ORIGINAL)
    directions = dict(DIRECTION_OF)
    kd = dict(KD)

    mixed_weights(original, directions, kd, Mix(3, 2))

    assert original == ORIGINAL
    assert directions == DIRECTION_OF
    assert kd == KD


def test_the_originals_come_first_in_the_order_given_then_the_kd_corpora() -> None:
    weights = _mixed(Mix(1, 1))

    assert list(weights) == [*ORIGINAL, *KD]


# --- mixed_weights: what it refuses --------------------------------------------------


def test_a_kd_corpus_of_a_direction_without_an_original_corpus_is_a_value_error() -> (
    None
):
    with pytest.raises(ValueError, match=r"xx-yy\.kd.*xx-yy.*no original corpus"):
        _mixed(Mix(1, 1), kd_corpora={**KD, "xx-yy.kd": "xx-yy"})


@pytest.mark.parametrize("weight", [0, -1, 1.0, "3", None, True])
def test_an_original_weight_that_is_not_a_positive_integer_is_a_value_error(
    weight: Any,
) -> None:
    with pytest.raises(ValueError, match=r"en-mn\.bt.*positive integer"):
        _mixed(Mix(1, 1), original={**ORIGINAL, "en-mn.bt": weight})


def test_a_name_that_is_both_an_original_and_a_kd_corpus_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"en-mn\.auth.*both"):
        _mixed(Mix(1, 1), kd_corpora={**KD, "en-mn.auth": "en-mn"})


def test_two_kd_corpora_of_one_direction_are_a_value_error() -> None:
    # Each would weigh K times the sum, so the KD share of the direction would be above
    # K / (K + O).
    with pytest.raises(ValueError, match=r"en-mn.*en-mn\.kd.*en-mn\.kd2"):
        _mixed(Mix(1, 1), kd_corpora={**KD, "en-mn.kd2": "en-mn"})


def test_an_original_without_a_direction_is_a_value_error() -> None:
    directions = {
        name: direction
        for name, direction in DIRECTION_OF.items()
        if name != "en-mn.bt"
    }

    with pytest.raises(ValueError, match=r"en-mn\.bt.*direction"):
        _mixed(Mix(1, 1), direction_of=directions)


def test_a_direction_for_a_corpus_that_is_no_original_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"direction.*ghost\.auth"):
        _mixed(Mix(1, 1), direction_of={**DIRECTION_OF, "ghost.auth": "en-mn"})


# --- repeat_factors ------------------------------------------------------------------


def test_the_repeat_factor_is_how_much_more_often_a_kd_row_is_drawn_than_an_original_one() -> (
    None
):
    # A KD row has weight 6 / 50 of the draws, an original row 3 / 100: four times more.
    rows = {"a.auth": 100, "a.kd": 50}
    weights = {"a.auth": 3, "a.kd": 6}

    factors = repeat_factors(rows, weights, {"a.auth": "a"}, {"a.kd": "a"})

    assert factors == {"a.kd": Fraction(4)}


def test_the_originals_of_a_direction_are_summed_in_weight_and_in_rows() -> None:
    rows = {"a.auth": 1200, "a.bt": 300, "a.kd": 1200}
    weights = {"a.auth": 22, "a.bt": 6, "a.kd": 28}

    factors = repeat_factors(rows, weights, {"a.auth": "a", "a.bt": "a"}, {"a.kd": "a"})

    # (28 / 1200) / (28 / 1500)
    assert factors == {"a.kd": Fraction(5, 4)}


def test_a_repeat_factor_is_an_exact_fraction_per_kd_corpus() -> None:
    rows = {"a.auth": 7, "b.auth": 9, "a.kd": 3, "b.kd": 11}
    weights = {"a.auth": 2, "b.auth": 5, "a.kd": 2, "b.kd": 5}

    factors = repeat_factors(
        rows,
        weights,
        {"a.auth": "a", "b.auth": "b"},
        {"a.kd": "a", "b.kd": "b"},
    )

    assert factors == {"a.kd": Fraction(7, 3), "b.kd": Fraction(9, 11)}
    assert all(isinstance(value, Fraction) for value in factors.values())


def test_the_repeat_factor_of_a_toy_mix_agrees_with_the_weights_of_the_mix() -> None:
    # 1200 KD rows against 1500 original rows of en-mn: at 1:1 each KD row is drawn 5/4 times
    # as often as an original row; at 3:1 three times that.
    rows = {
        "en-mn.auth": 1200,
        "en-mn.bt": 300,
        "en-mn.kd": 1200,
    }
    original = {"en-mn.auth": 22, "en-mn.bt": 6}
    directions = {"en-mn.auth": "en-mn", "en-mn.bt": "en-mn"}
    kd = {"en-mn.kd": "en-mn"}

    one = repeat_factors(
        rows, mixed_weights(original, directions, kd, Mix(1, 1)), directions, kd
    )
    three = repeat_factors(
        rows, mixed_weights(original, directions, kd, Mix(3, 1)), directions, kd
    )
    one_to_three = repeat_factors(
        rows, mixed_weights(original, directions, kd, Mix(1, 3)), directions, kd
    )

    assert one == {"en-mn.kd": Fraction(5, 4)}
    assert three == {"en-mn.kd": Fraction(15, 4)}
    assert one_to_three == {"en-mn.kd": Fraction(5, 12)}


@pytest.mark.parametrize(
    ("rows", "weights", "message"),
    [
        (
            {"a.auth": 10, "a.kd": 0},
            {"a.auth": 1, "a.kd": 1},
            r"rows of corpus 'a\.kd'",
        ),
        (
            {"a.auth": 0, "a.kd": 5},
            {"a.auth": 1, "a.kd": 1},
            r"rows of corpus 'a\.auth'",
        ),
        ({"a.auth": 10}, {"a.auth": 1, "a.kd": 1}, r"rows of corpus 'a\.kd'"),
        ({"a.auth": 10, "a.kd": 5}, {"a.kd": 1}, r"weight of corpus 'a\.auth'"),
        (
            {"a.auth": 10, "a.kd": 5},
            {"a.auth": 1, "a.kd": 0},
            r"weight of corpus 'a\.kd'",
        ),
    ],
)
def test_a_corpus_without_rows_or_a_positive_weight_is_a_value_error(
    rows: Mapping[str, int], weights: Mapping[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        repeat_factors(rows, weights, {"a.auth": "a"}, {"a.kd": "a"})


def test_a_kd_corpus_of_a_direction_without_original_rows_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"b\.kd.*no original corpus"):
        repeat_factors(
            {"a.auth": 10, "b.kd": 5},
            {"a.auth": 1, "b.kd": 1},
            {"a.auth": "a"},
            {"b.kd": "b"},
        )
