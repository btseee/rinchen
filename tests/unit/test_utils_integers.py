"""Tests for `rinchen.utils.integers.is_strict_int`: an `int` that is not a `bool`."""

from __future__ import annotations

import enum
from fractions import Fraction

import numpy as np
import pytest

from rinchen.utils.integers import is_strict_int


class Rank(enum.IntEnum):
    FIRST = 0


@pytest.mark.parametrize("value", [0, 1, -1, 7, 2**70, -(2**70), Rank.FIRST])
def test_an_integer_is_strict(value: object) -> None:
    assert is_strict_int(value) is True


@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_is_not_strict_although_python_calls_it_an_int(value: bool) -> None:
    assert isinstance(value, int)
    assert is_strict_int(value) is False


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        0.0,
        float("nan"),
        "1",
        "",
        b"1",
        None,
        [1],
        (1,),
        {"a": 1},
        1 + 0j,
        Fraction(1, 1),
        np.int64(1),
        object(),
    ],
)
def test_anything_else_is_not_strict(value: object) -> None:
    assert is_strict_int(value) is False
