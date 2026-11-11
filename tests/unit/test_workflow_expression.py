"""The model of the expressions of GitHub Actions that the static test of the nightly workflow uses
(D14, Task 6): `||` gives its first truthy operand, `&&` its first falsy one, `==` a boolean, the
precedence is `==` over `&&` over `||`, a name the context lacks is null, and what the model does
not know is refused."""

from __future__ import annotations

import pytest

from tests.support.workflow_expression import Expression


@pytest.mark.parametrize(
    ("source", "context", "value"),
    [
        ("${{ 'a' || 'b' }}", {}, "a"),
        ("${{ x || 'b' }}", {}, "b"),
        ("${{ x || 'b' }}", {"x": ""}, "b"),
        ("${{ 'a' && 'b' }}", {}, "b"),
        ("${{ x && 'b' }}", {}, None),
        ("${{ x == 'v' }}", {"x": "v"}, True),
        ("${{ x == 'v' }}", {"x": "w"}, False),
        ("${{ x == 'v' && 'yes' || 'no' }}", {"x": "v"}, "yes"),
        ("${{ x == 'v' && 'yes' || 'no' }}", {"x": "w"}, "no"),
        ("${{ y || (x == 'v' && 'yes' || 'no') }}", {"y": "set"}, "set"),
        ("${{ y || (x == 'v' && 'yes' || 'no') }}", {"x": "v"}, "yes"),
        ("${{ y || ('a' || 'b' && 'c') }}", {}, "a"),
        ("${{ a.b.c || 'd' }}", {"a.b.c": "dotted"}, "dotted"),
        ("  ${{   'padded'   }}  ", {}, "padded"),
    ],
)
def test_the_model_follows_the_operators_of_github_actions(
    source: str, context: dict[str, str], value: object
) -> None:
    assert Expression(source, context).value() == value


def test_and_binds_tighter_than_or() -> None:
    assert Expression("${{ 'a' || '' && 'c' }}", {}).value() == "a"
    assert Expression("${{ '' && 'b' || 'c' }}", {}).value() == "c"


def test_equality_binds_tighter_than_and() -> None:
    assert Expression("${{ 'x' && 'a' == 'b' }}", {}).value() is False
    assert Expression("${{ 'x' && 'a' == 'a' }}", {}).value() is True


@pytest.mark.parametrize(
    "source",
    [
        "x || y",
        "${{ x || y",
        "${{ x || }}",
        "${{ (x }}",
        "${{ x y }}",
        "${{ x ! y }}",
        "${{ x ?? 'y' }}",
        "${{ }}",
        "abc'a'de",
    ],
)
def test_the_model_refuses_what_it_does_not_know(source: str) -> None:
    with pytest.raises(ValueError):
        Expression(source, {}).value()
