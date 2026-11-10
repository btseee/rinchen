"""The refusals of a text of a request that a service may pass on (D11, fix wave).

The engine refuses a text it cannot take by a fixed rule (a lone surrogate, a reserved symbol)
and says so with a `RequestTextError`: a `DataError` whose message names the text by its place
and what is wrong with it, and never quotes it. Any other `DataError` may carry whatever its
raiser put in it, which is why the service passes on only this class.
"""

from __future__ import annotations

import pytest

from rinchen.errors import DataError
from rinchen.infer.engine_types import (
    RequestTextError,
    require_encodable,
    reserved_symbol_error,
)

LONE_SURROGATE = chr(0xD800)


def test_a_reserved_symbol_is_refused_with_a_request_text_error_that_names_its_place() -> (
    None
):
    error = reserved_symbol_error("row 3", "<2eng_Latn>")

    assert isinstance(error, RequestTextError)
    assert isinstance(error, DataError)
    assert str(error) == (
        "row 3 contains the reserved symbol <2eng_Latn>, which the model reads as a "
        "control piece and not as text (a direction tag would change the language asked "
        "for): remove it"
    )


def test_a_lone_surrogate_is_refused_with_a_request_text_error_that_names_its_place() -> (
    None
):
    with pytest.raises(RequestTextError) as caught:
        require_encodable("ab" + LONE_SURROGATE, "document 2")

    assert str(caught.value) == (
        "document 2 cannot be encoded as UTF-8: a lone surrogate at position 2"
    )
    assert LONE_SURROGATE not in str(caught.value)


def test_a_text_that_encodes_is_not_refused() -> None:
    require_encodable("Энэ бол од.", "row 0")


def test_a_plain_data_error_is_not_a_request_text_error() -> None:
    assert not isinstance(DataError("anything"), RequestTextError)


def test_the_place_and_the_problem_of_a_refusal_are_kept_apart() -> None:
    error = reserved_symbol_error("row 3", "<sep>")

    assert error.label == "row 3"
    assert error.problem.startswith("contains the reserved symbol <sep>, which")
    assert str(error) == f"{error.label} {error.problem}"
    with pytest.raises(RequestTextError) as caught:
        require_encodable("ab" + LONE_SURROGATE, "document 2")
    assert caught.value.label == "document 2"
    assert caught.value.problem == (
        "cannot be encoded as UTF-8: a lone surrogate at position 2"
    )
