"""The figures the evaluation stage reports besides the metrics of each run (D15, spec P8 and P9, brief
G5.4): the corpus chrF of each direction and system in the shape the thresholds read, the drop of chrF
from the float32 export to the int8 one, and the agreement of what the two exports wrote.

The sign of a drop is `float32 - int8` (a quantisation that loses quality is a positive drop), the least
and the largest are over the directions given, and the agreement counts rows whose texts are equal.
Each function is checked against figures worked out by hand.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.tiny.eval_readers import Hypothesis
from rinchen.tiny.evaluate_facts import (
    agreement,
    chrf_table,
    drop_extremes,
    drops_of,
)

A = "eng_Latn-khk_Cyrl"
B = "khk_Cyrl-eng_Latn"


def run(chrf: float, rows: int = 10) -> dict[str, Any]:
    return {
        "rows": rows,
        "metrics": {"bleu": 1.0, "chrf": chrf, "chrf_pp": 2.0},
        "statuses": {"ok": rows},
    }


def test_the_chrf_table_has_a_figure_for_each_direction_and_system() -> None:
    table = {
        A: {"int8": run(90.0), "float32": run(91.5)},
        B: {"int8": run(80.25), "float32": run(80.25)},
    }

    assert chrf_table(table) == {
        A: {"int8": 90.0, "float32": 91.5},
        B: {"int8": 80.25, "float32": 80.25},
    }


def test_the_chrf_table_of_nothing_is_empty() -> None:
    assert chrf_table({}) == {}


def test_a_drop_is_the_float32_figure_minus_the_int8_one() -> None:
    chrf = {A: {"int8": 90.0, "float32": 91.5}, B: {"int8": 81.0, "float32": 80.25}}

    drops = drops_of(chrf, reference="float32", gated="int8")

    assert drops == {A: 1.5, B: -0.75}


def test_the_drop_follows_the_names_of_the_systems_it_is_given() -> None:
    chrf = {A: {"a": 10.0, "b": 4.0, "c": 99.0}}

    assert drops_of(chrf, reference="a", gated="b") == {A: 6.0}
    assert drops_of(chrf, reference="b", gated="a") == {A: -6.0}


def test_a_direction_without_both_systems_is_refused() -> None:
    with pytest.raises(DataError):
        drops_of({A: {"int8": 90.0}}, reference="float32", gated="int8")
    with pytest.raises(DataError):
        drops_of({A: {"float32": 90.0}}, reference="float32", gated="int8")


def test_the_least_and_the_largest_drop_are_over_every_direction() -> None:
    assert drop_extremes({A: 1.5, B: -0.75, "x": 0.0}) == {
        "least": -0.75,
        "largest": 1.5,
    }
    assert drop_extremes({A: 0.25}) == {"least": 0.25, "largest": 0.25}


def test_the_extremes_of_no_drop_are_refused() -> None:
    with pytest.raises(DataError):
        drop_extremes({})


def h(direction: str, row_id: str, text: str, status: str = "ok") -> Hypothesis:
    return Hypothesis(direction, row_id, status, text)


def test_agreement_counts_the_rows_with_equal_texts_by_direction() -> None:
    first = [h(A, "1", "a"), h(A, "2", "b"), h(B, "1", "c"), h(B, "2", "d")]
    second = [h(A, "1", "a"), h(A, "2", "X"), h(B, "1", "c"), h(B, "2", "d")]

    found = agreement(first, second)

    assert found == {
        "by_direction": {A: {"rows": 2, "equal": 1}, B: {"rows": 2, "equal": 2}},
        "rows": 4,
        "equal": 3,
        "share": 0.75,
    }


def test_agreement_does_not_depend_on_the_order_of_the_rows() -> None:
    first = [h(A, "1", "a"), h(A, "2", "b")]
    second = [h(A, "2", "b"), h(A, "1", "a")]

    assert agreement(first, second)["share"] == 1.0


def test_agreement_is_about_the_text_and_not_the_status() -> None:
    first = [h(A, "1", "a", "ok")]
    second = [h(A, "1", "a", "truncated")]

    assert agreement(first, second)["equal"] == 1


def test_agreement_of_texts_that_all_differ_is_zero() -> None:
    found = agreement([h(A, "1", "a")], [h(A, "1", "b")])

    assert (found["equal"], found["share"]) == (0, 0.0)


@pytest.mark.parametrize(
    "second",
    [
        [h(A, "1", "a")],
        [h(A, "1", "a"), h(A, "2", "b"), h(A, "3", "c")],
        [h(A, "1", "a"), h(A, "3", "b")],
        [h(A, "1", "a"), h(B, "2", "b")],
        [],
    ],
    ids=["fewer", "more", "other-id", "other-direction", "none"],
)
def test_two_outputs_that_do_not_answer_the_same_rows_are_refused(
    second: list[Hypothesis],
) -> None:
    with pytest.raises(DataError):
        agreement([h(A, "1", "a"), h(A, "2", "b")], second)


def test_agreement_of_no_rows_is_refused() -> None:
    with pytest.raises(DataError):
        agreement([], [])


def test_the_share_is_exact_for_a_count_that_has_no_short_decimal() -> None:
    first = [h(A, str(number), "a") for number in range(3)]
    second = [h(A, "0", "a"), h(A, "1", "a"), h(A, "2", "b")]

    assert math.isclose(agreement(first, second)["share"], 2 / 3)
