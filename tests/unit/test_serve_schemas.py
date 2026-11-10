"""The request, the response and the errors of the service (D11, T15): what a body may
hold, which field a refusal names, and the one error shape every path answers in."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from rinchen.errors import ConfigError
from rinchen.serve.schemas import (
    ErrorBody,
    ErrorResponse,
    TranslateRequest,
    TranslateResponse,
    bounded_request_model,
    error_response,
    validation_error_response,
)

SECRET = "SECRET-TEXT-OF-A-USER"


def refused_fields(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The location of every complaint about `payload`."""
    with pytest.raises(ValidationError) as caught:
        TranslateRequest.model_validate(payload)
    return [tuple(error["loc"]) for error in caught.value.errors()]


def valid(**changes: Any) -> dict[str, Any]:
    return {"texts": ["a"], "tgt": "khk_Cyrl", **changes}


def test_a_minimal_request_gets_these_defaults() -> None:
    request = TranslateRequest.model_validate({"texts": ["Hello"], "tgt": "khk_Cyrl"})

    assert request.model_dump(mode="json") == {
        "texts": ["Hello"],
        "src": "auto",
        "tgt": "khk_Cyrl",
        "script": None,
        "beam_size": 4,
        "n_best": 1,
        "doc_mode": False,
    }


def test_a_full_request_round_trips_through_json() -> None:
    body = {
        "texts": ["one", "two", ""],
        "src": "eng_Latn",
        "tgt": "mn_Mong",
        "script": "Mong",
        "beam_size": 8,
        "n_best": 3,
        "doc_mode": True,
    }

    request = TranslateRequest.model_validate_json(json.dumps(body))

    assert request.model_dump(mode="json") == body
    assert TranslateRequest.model_validate_json(request.model_dump_json()) == request


def test_an_empty_text_is_a_valid_text() -> None:
    assert TranslateRequest.model_validate(valid(texts=[""])).texts == [""]


@pytest.mark.parametrize("script", ["Cyrl", "Mong", None])
def test_the_two_scripts_and_none_are_the_only_choices(script: str | None) -> None:
    assert TranslateRequest.model_validate(valid(script=script)).script == script


def test_the_limits_are_included_on_the_valid_side() -> None:
    request = TranslateRequest.model_validate(
        valid(texts=["x" * 5000] * 64, beam_size=8, n_best=8)
    )

    assert len(request.texts) == 64


@pytest.mark.parametrize(
    ("payload", "location"),
    [
        (valid(texts=[]), ("texts",)),
        (valid(texts=["a"] * 65), ("texts",)),
        (valid(texts=["x" * 5001]), ("texts", 0)),
        (valid(texts=["ok", 12]), ("texts", 1)),
        (valid(texts="one text"), ("texts",)),
        (valid(beam_size=0), ("beam_size",)),
        (valid(beam_size=9), ("beam_size",)),
        (valid(beam_size="4"), ("beam_size",)),
        (valid(beam_size=True), ("beam_size",)),
        (valid(beam_size=4, n_best=5), ("n_best",)),
        (valid(beam_size=2, n_best=3), ("n_best",)),
        (valid(n_best=0), ("n_best",)),
        (valid(terminology={"a": "b"}), ("terminology",)),
        (valid(script="Latin"), ("script",)),
        (valid(script="cyrl"), ("script",)),
        (valid(script="Latn"), ("script",)),
        (valid(script=""), ("script",)),
        (valid(src=""), ("src",)),
        (valid(src="   "), ("src",)),
        (valid(src="x" * 33), ("src",)),
        (valid(tgt=""), ("tgt",)),
        (valid(tgt=" "), ("tgt",)),
        (valid(tgt=None), ("tgt",)),
        (valid(doc_mode="yes"), ("doc_mode",)),
        ({"texts": ["a"]}, ("tgt",)),
        ({"tgt": "khk_Cyrl"}, ("texts",)),
    ],
    ids=lambda value: str(value)[:40],
)
def test_each_invalid_request_names_its_field(
    payload: dict[str, Any], location: tuple[Any, ...]
) -> None:
    assert refused_fields(payload) == [location]


def test_every_complaint_of_a_request_is_reported_not_only_the_first() -> None:
    locations = refused_fields(valid(texts=[], beam_size=9, script="Latin"))

    assert sorted(locations, key=str) == [("beam_size",), ("script",), ("texts",)]


def test_n_best_is_judged_against_a_beam_that_is_itself_valid_only() -> None:
    assert refused_fields(valid(beam_size=0, n_best=3)) == [("beam_size",)]


def test_hypotheses_are_judged_against_a_beam_only_when_the_request_names_one() -> None:
    # Without `beam_size` the service decodes with the beam of its config, which the schema does
    # not know: `decode_options` judges `n_best` against it.
    assert TranslateRequest.model_validate(valid(n_best=6)).n_best == 6
    assert TranslateRequest.model_validate(valid(n_best=6)).beam_size == 4
    assert refused_fields(valid(n_best=6, beam_size=4)) == [("n_best",)]
    assert TranslateRequest.model_validate(valid(n_best=6, beam_size=6)).n_best == 6


def test_the_refusal_of_hypotheses_over_the_named_beam_says_so_in_the_terms_of_the_api() -> (
    None
):
    with pytest.raises(ValidationError) as caught:
        TranslateRequest.model_validate_json(
            '{"texts": ["a"], "tgt": "x", "beam_size": 3, "n_best": 5}'
        )

    assert [(e["loc"], e["msg"]) for e in caught.value.errors()] == [
        (
            ("n_best",),
            (
                "Value error, n_best (5) must not exceed beam_size (3): the beam holds "
                "no more hypotheses"
            ),
        )
    ]


def test_a_request_cannot_be_changed_after_it_is_read() -> None:
    request = TranslateRequest.model_validate(valid())

    with pytest.raises(ValidationError):
        request.tgt = "eng_Latn"  # type: ignore[misc]


def test_the_limits_of_a_service_narrow_the_request() -> None:
    request_model = bounded_request_model(max_texts=2, max_text_chars=10)

    assert (
        request_model.model_validate(valid(texts=["x" * 10] * 2)).texts
        == ["x" * 10] * 2
    )
    with pytest.raises(ValidationError) as too_many:
        request_model.model_validate(valid(texts=["a"] * 3))
    with pytest.raises(ValidationError) as too_long:
        request_model.model_validate(valid(texts=["x" * 11]))
    assert [tuple(e["loc"]) for e in too_many.value.errors()] == [("texts",)]
    assert [tuple(e["loc"]) for e in too_long.value.errors()] == [("texts", 0)]


def test_the_narrowed_request_is_a_translate_request_by_the_same_name() -> None:
    request_model = bounded_request_model(max_texts=3, max_text_chars=7)

    assert issubclass(request_model, TranslateRequest)
    assert request_model.__name__ == "TranslateRequest"
    assert request_model.model_validate(valid()).beam_size == 4


def test_the_narrowed_request_keeps_every_other_rule() -> None:
    request_model = bounded_request_model(max_texts=3, max_text_chars=7)

    with pytest.raises(ValidationError, match="terminology"):
        request_model.model_validate(valid(terminology=1))
    with pytest.raises(ValidationError, match="n_best"):
        request_model.model_validate(valid(beam_size=1, n_best=2))


def test_the_limits_of_the_unnarrowed_request_are_the_defaults_of_the_config() -> None:
    assert refused_fields(valid(texts=["a"] * 65)) == [("texts",)]
    assert refused_fields(valid(texts=["x" * 5001])) == [("texts", 0)]


@pytest.mark.parametrize("limits", [(0, 10), (65, 10), (2, 0), (2, 100001)])
def test_limits_outside_what_a_config_allows_are_refused(
    limits: tuple[int, int],
) -> None:
    with pytest.raises(ConfigError, match="between"):
        bounded_request_model(max_texts=limits[0], max_text_chars=limits[1])


def test_the_error_response_has_one_shape() -> None:
    response = error_response(503, "overloaded", "the queue holds 5 of 5 sentences")

    assert response.status_code == 503
    assert response.media_type == "application/json"
    assert json.loads(response.body) == {
        "error": {
            "code": "overloaded",
            "message": "the queue holds 5 of 5 sentences",
        }
    }


def test_the_error_models_say_the_same_thing() -> None:
    body = ErrorResponse(error=ErrorBody(code="internal", message="x"))

    assert body.model_dump() == {"error": {"code": "internal", "message": "x"}}
    with pytest.raises(ValidationError):
        ErrorBody(code="internal", message="x", detail="y")  # type: ignore[call-arg]


def test_a_validation_error_is_a_422_in_the_error_shape() -> None:
    with pytest.raises(ValidationError) as caught:
        TranslateRequest.model_validate(valid(texts=[], beam_size=9))

    response = validation_error_response(caught.value.errors())
    body = json.loads(response.body)

    assert response.status_code == 422
    assert list(body) == ["error"]
    assert body["error"]["code"] == "validation_error"
    assert "texts" in body["error"]["message"]
    assert "beam_size" in body["error"]["message"]
    assert list(body["error"]) == ["code", "message"]


def test_the_message_of_a_validation_error_never_holds_the_input() -> None:
    with pytest.raises(ValidationError) as caught:
        TranslateRequest.model_validate(
            valid(texts=[SECRET * 300], terminology=SECRET, src=SECRET * 3)
        )

    message = json.loads(validation_error_response(caught.value.errors()).body)[
        "error"
    ]["message"]

    assert SECRET not in message
    assert "texts.0" in message
    assert "terminology" in message
    assert "src" in message


def test_a_validation_message_names_the_field_without_the_body_prefix() -> None:
    errors = [
        {"type": "x", "loc": ("body", "texts", 0), "msg": "too long", "input": SECRET},
        {"type": "y", "loc": ("body", 3), "msg": "JSON decode error", "input": {}},
        {"type": "z", "loc": (), "msg": "odd", "input": SECRET},
    ]

    message = json.loads(validation_error_response(errors).body)["error"]["message"]

    assert message == "texts.0: too long; body.3: JSON decode error; body: odd"


def test_a_response_has_the_fields_of_the_service() -> None:
    response = TranslateResponse(
        translations=["Сайн уу"],
        detected_src=["eng_Latn"],
        model="toy",
        latency_ms=12.5,
        statuses=["ok"],
        script_purity=[1.0],
    )

    assert response.model_dump(mode="json") == {
        "translations": ["Сайн уу"],
        "detected_src": ["eng_Latn"],
        "model": "toy",
        "latency_ms": 12.5,
        "statuses": ["ok"],
        "script_purity": [1.0],
    }


def test_the_alternatives_are_present_only_when_there_are_some() -> None:
    base: dict[str, Any] = {
        "translations": ["a", "b"],
        "detected_src": ["und", "und"],
        "model": "toy",
        "latency_ms": 1.0,
        "statuses": ["ok", "empty"],
        "script_purity": [1.0, None],
    }

    without = TranslateResponse(**base)
    with_them = TranslateResponse(**base, alternatives=[["a", "a2"], []])

    assert "alternatives" not in json.loads(without.model_dump_json())
    assert "alternatives" not in without.model_dump(mode="json")
    assert json.loads(with_them.model_dump_json())["alternatives"] == [["a", "a2"], []]
    assert json.loads(with_them.model_dump_json())["script_purity"] == [1.0, None]


def test_a_response_cannot_claim_a_negative_latency() -> None:
    with pytest.raises(ValidationError, match="latency_ms"):
        TranslateResponse(
            translations=[],
            detected_src=[],
            model="m",
            latency_ms=-0.5,
            statuses=[],
            script_purity=[],
        )


def test_a_response_with_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        TranslateResponse(  # type: ignore[call-arg]
            translations=[],
            detected_src=[],
            model="m",
            latency_ms=0.0,
            statuses=[],
            script_purity=[],
            terminology=1,
        )
