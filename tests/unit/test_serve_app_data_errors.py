"""The service (D11, fix wave): what a 400 says when the engine refuses a text.

The body of a 400 for a `DataError` of the engine is not the error's message unless the engine
raised it as a `RequestTextError` (a fixed refusal that names the text by its place and what is
wrong with it, and never quotes it). Anything else is one fixed sentence: a `DataError` can carry
whatever its raiser put in it, and a text of a request must not come back in a response.
"""

from __future__ import annotations

from typing import Any

from rinchen.errors import ConfigError, DataError
from rinchen.infer.engine_types import (
    require_encodable,
    reserved_symbol_error,
)
from tests.support.serve_app import (
    ENG,
    SECRET,
    FakeEngine,
    RecordingLog,
    error_of,
    make_client,
    post,
    run,
)

FIXED = "the model refused a text of this request"


def refused(log: RecordingLog, failure: BaseException, **fields: Any) -> Any:
    async def scenario() -> Any:
        engine = FakeEngine(translate_failure=failure, document_level=True)
        async with make_client(log, engine) as svc:
            return await post(svc.client, **fields)

    return run(scenario())


def test_a_data_error_of_any_other_kind_is_a_fixed_400_that_holds_none_of_its_text(
    recording_log: RecordingLog,
) -> None:
    response = refused(
        recording_log, DataError(f"row 0 holds {SECRET} which is not text"), texts=["a"]
    )

    assert response.status_code == 400
    assert error_of(response) == {"code": "bad_request", "message": FIXED}
    assert SECRET not in response.text


def test_a_data_error_of_a_document_request_is_the_same_fixed_400(
    recording_log: RecordingLog,
) -> None:
    response = refused(
        recording_log,
        DataError(f"sentence 2 contains {SECRET}"),
        texts=["a"],
        src=ENG,
        doc_mode=True,
    )

    assert error_of(response) == {"code": "bad_request", "message": FIXED}
    assert SECRET not in response.text


def test_a_reserved_symbol_is_a_400_that_names_the_row_and_the_symbol_only(
    recording_log: RecordingLog,
) -> None:
    response = refused(
        recording_log,
        reserved_symbol_error("row 1", "<2eng_Latn>"),
        texts=["a", f"<2eng_Latn> {SECRET}"],
    )

    assert response.status_code == 400
    assert error_of(response) == {
        "code": "bad_request",
        "message": (
            "row 1 contains the reserved symbol <2eng_Latn>, which the model reads as a "
            "control piece and not as text (a direction tag would change the language "
            "asked for): remove it"
        ),
    }
    assert SECRET not in response.text


def test_a_reserved_symbol_of_a_document_names_the_document(
    recording_log: RecordingLog,
) -> None:
    response = refused(
        recording_log,
        reserved_symbol_error("document 0", "<sep>"),
        texts=[f"a <sep> {SECRET}"],
        src=ENG,
        doc_mode=True,
    )

    assert error_of(response)["message"].startswith(
        "document 0 contains the reserved symbol <sep>, "
    )
    assert SECRET not in response.text


def test_a_lone_surrogate_is_a_400_that_names_the_row_and_the_position(
    recording_log: RecordingLog,
) -> None:
    try:
        require_encodable("ab" + chr(0xD800), "row 0")
    except DataError as failure:
        response = refused(recording_log, failure, texts=["a"])

    assert error_of(response) == {
        "code": "bad_request",
        "message": "row 0 cannot be encoded as UTF-8: a lone surrogate at position 2",
    }


def test_a_config_error_of_the_engine_is_still_passed_on(
    recording_log: RecordingLog,
) -> None:
    response = refused(
        recording_log, ConfigError("the model does not write xxx_Latn"), texts=["a"]
    )

    assert error_of(response) == {
        "code": "bad_request",
        "message": "the model does not write xxx_Latn",
    }
