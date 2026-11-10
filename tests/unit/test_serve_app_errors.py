"""The service (D11, Task 9): every way a request is refused and what the refusal looks like.

Every error, on every path, is `{"error": {"code": str, "message": str}}` with the status of its
code: 422 `validation_error`, 400 `bad_request`, 413 `payload_too_large`, 503 `overloaded`,
`draining` and `not_ready` (with `Retry-After: 1`), 504 `timeout`, 500 `internal`, 404
`not_found` and 405 `method_not_allowed`. A message names a field or a language, never a text a
client sent: the last tests plant a secret in every request and look for it in every response,
log event and metric.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from tests.support.serve_app import (
    ENG,
    KHK,
    SECRET,
    FakeEngine,
    RecordingLog,
    engine_started,
    error_of,
    make_client,
    post,
    run,
    sample,
    until,
)

JSON = {"content-type": "application/json"}


async def post_raw(
    svc: Any, content: bytes, headers: dict[str, str] | None = None
) -> Any:
    return await svc.client.post(
        "/v1/translate", content=content, headers=JSON if headers is None else headers
    )


# ---- 422: the body is not a request -------------------------------------------------------

REFUSED_BODIES = [
    ({"tgt": KHK}, "texts"),
    ({"texts": ["a"]}, "tgt"),
    ({"texts": [], "tgt": KHK}, "texts"),
    ({"texts": ["a"] * 4, "tgt": KHK}, "texts"),
    ({"texts": ["a", "b" * 11], "tgt": KHK}, "texts.1"),
    ({"texts": ["a"], "tgt": KHK, "terminology": {}}, "terminology"),
    ({"texts": ["a"], "tgt": KHK, "beam_size": "4"}, "beam_size"),
    ({"texts": ["a"], "tgt": KHK, "beam_size": 9}, "beam_size"),
    ({"texts": ["a"], "tgt": KHK, "beam_size": 2, "n_best": 3}, "n_best"),
    ({"texts": ["a"], "tgt": KHK, "script": "Latn"}, "script"),
    ({"texts": ["a"], "tgt": KHK, "doc_mode": "yes"}, "doc_mode"),
    ({"texts": ["a"], "tgt": ""}, "tgt"),
    ({"texts": ["a"], "tgt": KHK, "src": " "}, "src"),
    ({"texts": [1], "tgt": KHK}, "texts.0"),
]


@pytest.mark.parametrize(("payload", "field"), REFUSED_BODIES)
def test_a_body_that_is_not_a_request_is_a_422_that_names_the_field(
    payload: dict[str, Any], field: str, recording_log: RecordingLog
) -> None:
    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(
            recording_log, engine, max_texts=3, max_text_chars=10
        ) as svc:
            return await post_raw(svc, json.dumps(payload).encode()), engine

    response, engine = run(scenario())

    error = error_of(response)
    assert response.status_code == 422
    assert error["code"] == "validation_error"
    assert error["message"].split(":")[0] == field
    assert engine.calls == []


def test_a_body_that_is_not_json_is_a_422_in_the_error_shape(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[Any]:
        async with make_client(recording_log) as svc:
            return [
                await post_raw(svc, b'{"texts": ['),
                await post_raw(svc, b""),
                await post_raw(svc, b"[1, 2]"),
                await post_raw(svc, b'"text"'),
            ]

    responses = run(scenario())

    assert [r.status_code for r in responses] == [422] * 4
    assert [error_of(r)["code"] for r in responses] == ["validation_error"] * 4
    messages = [error_of(r)["message"] for r in responses]
    assert messages[0].startswith("body: Invalid JSON")
    assert messages[1].startswith("body: Invalid JSON")
    assert messages[2] == "body: Input should be an object"
    assert messages[3] == "body: Input should be an object"


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "text/plain"},
        {"content-type": "application/jsonx"},
        {"content-type": "application/x-www-form-urlencoded"},
        {},
    ],
)
def test_a_body_that_is_not_declared_as_json_is_a_422(
    headers: dict[str, str], recording_log: RecordingLog
) -> None:
    body = json.dumps({"texts": ["a"], "tgt": KHK}).encode()

    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(recording_log, engine) as svc:
            return await svc.client.post(
                "/v1/translate", content=body, headers=headers
            ), engine

    response, engine = run(scenario())

    assert response.status_code == 422
    assert error_of(response) == {
        "code": "validation_error",
        "message": "body: the content type must be application/json",
    }
    assert engine.calls == []


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "application/json; charset=utf-8", "Application/JSON"],
)
def test_the_content_type_of_a_json_body_may_carry_a_charset_and_any_case(
    content_type: str, recording_log: RecordingLog
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await post_raw(
                svc,
                json.dumps({"texts": ["a1"], "tgt": KHK}).encode(),
                {"content-type": content_type},
            )

    assert run(scenario()).status_code == 200


def test_a_422_never_echoes_what_the_client_sent(recording_log: RecordingLog) -> None:
    bodies = [
        json.dumps({"texts": [SECRET], "tgt": KHK, "beam_size": SECRET}).encode(),
        json.dumps({"texts": [SECRET * 1000], "tgt": KHK}).encode(),
        json.dumps({"texts": [SECRET], "tgt": SECRET * 5}).encode(),
        json.dumps({"texts": [SECRET], "tgt": KHK, "n_best": SECRET}).encode(),
        b'{"texts": ["' + SECRET.encode() + b'", ',
    ]

    async def scenario() -> list[Any]:
        async with make_client(recording_log, max_text_chars=100) as svc:
            return [await post_raw(svc, body) for body in bodies]

    responses = run(scenario())

    assert [r.status_code for r in responses] == [422] * 5
    for response in responses:
        assert SECRET not in response.text


# ---- 400: the engine refuses the request --------------------------------------------------


def bad_request(response: Any) -> str:
    assert response.status_code == 400, response.text
    error = error_of(response)
    assert error["code"] == "bad_request"
    return error["message"]


def test_a_target_the_model_does_not_write_is_a_400_that_names_what_it_writes(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> list[Any]:
        async with make_client(recording_log, scripted_engine(tmp_path).engine) as svc:
            return [
                await post(svc.client, texts=[SECRET], tgt="deu_Latn"),
                await post(svc.client, texts=[SECRET], tgt="xxx_Latn"),
            ]

    served, unknown = run(scenario())

    assert KHK in bad_request(served)
    assert "xxx_Latn" in bad_request(unknown)
    assert SECRET not in served.text + unknown.text


def test_an_unknown_source_or_the_same_source_and_target_is_a_400(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> list[Any]:
        async with make_client(recording_log, scripted_engine(tmp_path).engine) as svc:
            return [
                await post(svc.client, texts=[SECRET], src="xxx_Latn"),
                await post(svc.client, texts=[SECRET], src=KHK, tgt=KHK),
            ]

    unknown, same = run(scenario())

    assert "not in the language registry" in bad_request(unknown)
    assert "nothing to translate" in bad_request(same)
    assert SECRET not in unknown.text + same.text


def test_a_script_for_a_target_that_has_none_is_a_400(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> Any:
        async with make_client(recording_log, scripted_engine(tmp_path).engine) as svc:
            return await post(svc.client, texts=[SECRET], tgt=ENG, script="Mong")

    response = run(scenario())

    assert "a script applies to a Mongolian target only" in bad_request(response)
    assert SECRET not in response.text


def test_documents_for_a_sentence_model_or_without_a_source_are_a_400(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> list[Any]:
        sentence_model = scripted_engine(tmp_path / "sentence").engine
        async with make_client(recording_log, sentence_model) as svc:
            sentence = await post(svc.client, texts=[SECRET], src=ENG, doc_mode=True)
        document_model = scripted_engine(
            tmp_path / "document", document_level=True
        ).engine
        async with make_client(recording_log, document_model) as svc:
            auto = await post(svc.client, texts=[SECRET], doc_mode=True)
        return [sentence, auto]

    sentence, auto = run(scenario())

    assert "does not translate documents" in bad_request(sentence)
    assert "needs its source language" in bad_request(auto)
    assert SECRET not in sentence.text + auto.text


def test_a_document_with_a_reserved_marker_is_a_400_that_does_not_quote_it(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> Any:
        engine = scripted_engine(tmp_path, document_level=True).engine
        async with make_client(recording_log, engine) as svc:
            return await post(
                svc.client,
                texts=["fine", f"{SECRET} <sep> {SECRET}"],
                src=ENG,
                doc_mode=True,
            )

    response = run(scenario())

    message = bad_request(response)
    assert "document 1" in message
    assert SECRET not in response.text


def test_more_hypotheses_than_the_beam_the_service_was_given_is_a_400(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log, beam_size=2) as svc:
            return await post(svc.client, texts=[SECRET], n_best=3)

    response = run(scenario())

    assert "beam_size" in bad_request(response)
    assert SECRET not in response.text


# ---- 413: the body is too big -------------------------------------------------------------


def padded_body(size: int) -> bytes:
    """A valid request of exactly `size` bytes (spaces after the JSON)."""
    body = json.dumps({"texts": ["fits"], "tgt": KHK}).encode()
    assert len(body) <= size
    return body + b" " * (size - len(body))


def test_a_body_over_the_limit_by_its_length_is_a_413_and_is_not_read(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(recording_log, engine, max_body_bytes=1024) as svc:
            return await post_raw(svc, padded_body(2000)), engine

    response, engine = run(scenario())

    assert response.status_code == 413
    error = error_of(response)
    assert error["code"] == "payload_too_large"
    assert "1024" in error["message"]
    assert response.headers["connection"] == "close"
    assert engine.calls == []


def test_a_body_that_declares_more_than_the_limit_is_refused_before_a_byte_of_it_is_read(
    recording_log: RecordingLog,
) -> None:
    chunks = Chunks(5, 400)

    async def scenario() -> Any:
        async with make_client(recording_log, max_body_bytes=1024) as svc:
            return await svc.client.post(
                "/v1/translate",
                content=chunks,
                headers={**JSON, "content-length": "2000"},
            )

    response = run(scenario())

    assert response.status_code == 413
    assert chunks.pulled == 0


def test_a_body_of_exactly_the_limit_is_accepted_and_one_byte_more_is_not(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[int]:
        async with make_client(recording_log, max_body_bytes=1024) as svc:
            return [
                (await post_raw(svc, padded_body(1024))).status_code,
                (await post_raw(svc, padded_body(1025))).status_code,
            ]

    assert run(scenario()) == [200, 413]


class Chunks:
    """A request body of `count` chunks of `size` bytes that says how many were pulled."""

    def __init__(self, count: int, size: int, *, head: bytes = b"") -> None:
        self.count, self.size, self.head = count, size, head
        self.pulled = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index in range(self.count):
            self.pulled += 1
            yield self.head if index == 0 and self.head else b" " * self.size
            await asyncio.sleep(0)


def test_a_chunked_body_over_the_limit_is_a_413_and_is_read_only_to_the_chunk_that_crossed_it(
    recording_log: RecordingLog,
) -> None:
    chunks = Chunks(50, 100)

    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(recording_log, engine, max_body_bytes=1024) as svc:
            response = await svc.client.post(
                "/v1/translate", content=chunks, headers=JSON
            )
            return response, engine

    response, engine = run(scenario())

    assert response.status_code == 413
    assert error_of(response)["code"] == "payload_too_large"
    assert chunks.pulled == 11
    assert engine.calls == []


def test_a_chunked_body_within_the_limit_is_read_whole(
    recording_log: RecordingLog,
) -> None:
    body = json.dumps({"texts": ["fits"], "tgt": KHK}).encode()
    chunks = Chunks(1, 0, head=body)

    async def scenario() -> Any:
        async with make_client(recording_log, max_body_bytes=1024) as svc:
            return await svc.client.post("/v1/translate", content=chunks, headers=JSON)

    response = run(scenario())

    assert response.status_code == 200
    assert response.json()["translations"] == ["<fits>"]


class TwoChunks:
    """A request body sent as two chunks, the first of `split` bytes."""

    def __init__(self, body: bytes, split: int = 600) -> None:
        self.body, self.split = body, split

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body[: self.split]
        yield self.body[self.split :]


def test_a_chunked_body_of_exactly_the_limit_is_accepted_and_one_byte_more_is_not(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[int]:
        async with make_client(recording_log, max_body_bytes=1024) as svc:
            return [
                (
                    await svc.client.post(
                        "/v1/translate",
                        content=TwoChunks(padded_body(size)),
                        headers=JSON,
                    )
                ).status_code
                for size in (1024, 1025)
            ]

    assert run(scenario()) == [200, 413]


def test_a_413_is_counted_and_logged_under_translate(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_client(recording_log, max_body_bytes=1024) as svc:
            await post_raw(svc, padded_body(2000))
            return (await svc.client.get("/metrics")).text

    text = run(scenario())

    assert sample(text, "rinchen_requests_total", route="translate", status="413") == 1
    (event,) = [e for e in recording_log.named("request") if e["route"] == "translate"]
    assert (event["status"], event["code"]) == (413, "payload_too_large")


# ---- 503: the service cannot take the request now ------------------------------------------


def test_a_request_that_does_not_fit_the_queue_is_a_503_overloaded_with_retry_after(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, list[Any], str]:
        engine = FakeEngine()
        async with make_client(
            recording_log,
            engine,
            inter_threads=1,
            max_texts=2,
            max_queue_sentences=2,
        ) as svc:
            engine.hold()
            first = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            queued = asyncio.create_task(post(svc.client, texts=["a1", "b2"]))
            await until(lambda: svc.state.batcher.depth == 2)
            refused = await post(svc.client, texts=[SECRET])
            metrics = (await svc.client.get("/metrics")).text
            engine.release()
            return refused, list(await asyncio.gather(first, queued)), metrics

    refused, answered, metrics = run(scenario())

    assert refused.status_code == 503
    assert error_of(refused)["code"] == "overloaded"
    assert refused.headers["retry-after"] == "1"
    assert SECRET not in refused.text
    assert [r.status_code for r in answered] == [200, 200]
    assert (
        sample(metrics, "rinchen_requests_total", route="translate", status="503") == 1
    )


# ---- 504: it took too long ----------------------------------------------------------------


def test_a_request_that_takes_longer_than_the_timeout_is_a_504_and_its_queued_items_are_dropped(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[list[Any], str, FakeEngine, int]:
        engine = FakeEngine()
        async with make_client(
            recording_log, engine, inter_threads=1, request_timeout_seconds=0.2
        ) as svc:
            engine.hold()
            running = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            waiting = asyncio.create_task(post(svc.client, texts=["queued"]))
            answered = list(await asyncio.gather(running, waiting))
            depth = svc.state.batcher.depth
            engine.release()
            await asyncio.to_thread(engine.wait_for_finished, 1)
            return answered, (await svc.client.get("/metrics")).text, engine, depth

    answered, metrics, engine, depth = run(scenario())

    assert [r.status_code for r in answered] == [504, 504]
    assert [error_of(r)["code"] for r in answered] == ["timeout", "timeout"]
    assert depth == 0
    assert engine.texts_seen() == ["held"]
    assert (
        sample(metrics, "rinchen_requests_total", route="translate", status="504") == 2
    )


def test_the_service_answers_again_after_a_request_timed_out(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any]:
        engine = FakeEngine()
        async with make_client(
            recording_log, engine, request_timeout_seconds=0.2
        ) as svc:
            engine.hold()
            slow = await post(svc.client, texts=["slow"])
            engine.release()
            await asyncio.to_thread(engine.wait_for_finished, 1)
            return slow, await post(svc.client, texts=["quick"])

    slow, quick = run(scenario())

    assert slow.status_code == 504
    assert quick.status_code == 200
    assert quick.json()["translations"] == ["<quick>"]


# ---- 500: the engine fails ----------------------------------------------------------------


def test_an_engine_failure_is_a_generic_500_and_the_cause_is_logged(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine(translate_failure=UpstreamError("the library failed on row 0"))

    async def scenario() -> tuple[Any, str]:
        async with make_client(recording_log, engine) as svc:
            return await post(svc.client, texts=[SECRET]), (
                await svc.client.get("/metrics")
            ).text

    response, metrics = run(scenario())

    assert response.status_code == 500
    error = error_of(response)
    assert error["code"] == "internal"
    assert "library" not in error["message"]
    assert SECRET not in response.text
    (failed,) = recording_log.named("request_failed")
    assert failed == {
        "error_type": "UpstreamError",
        "cause": "the library failed on row 0",
    }
    assert (
        sample(metrics, "rinchen_requests_total", route="translate", status="500") == 1
    )
    assert sample(metrics, "rinchen_translated_sentences_total", status="ok") is None


def test_an_unexpected_failure_logs_its_type_and_never_its_message(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine(translate_failure=RuntimeError(f"cannot decode {SECRET}"))

    async def scenario() -> Any:
        async with make_client(recording_log, engine) as svc:
            return await post(svc.client, texts=[SECRET])

    response = run(scenario())

    assert response.status_code == 500
    assert error_of(response)["code"] == "internal"
    (failed,) = recording_log.named("request_failed")
    assert failed == {"error_type": "RuntimeError"}
    assert SECRET not in response.text + recording_log.text()


def test_a_timeout_of_the_engine_is_a_500_and_not_the_timeout_of_the_request(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine(translate_failure=TimeoutError("the engine's own"))

    async def scenario() -> Any:
        async with make_client(recording_log, engine) as svc:
            return await post(svc.client)

    response = run(scenario())

    assert response.status_code == 500
    assert error_of(response)["code"] == "internal"
    assert recording_log.named("request_failed") == [{"error_type": "TimeoutError"}]


def test_the_service_answers_again_after_an_engine_failure(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any]:
        engine = FakeEngine(translate_failure=UpstreamError("failed"))
        async with make_client(recording_log, engine) as svc:
            failed = await post(svc.client)
            engine.translate_failure = None
            return failed, await post(svc.client)

    failed, after = run(scenario())

    assert (failed.status_code, after.status_code) == (500, 200)


# ---- the other paths ----------------------------------------------------------------------


def test_an_unknown_path_and_a_wrong_method_answer_in_the_error_shape(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[Any]:
        async with make_client(recording_log) as svc:
            return [
                await svc.client.get("/no/such/path"),
                await svc.client.get("/v1/translate"),
                await svc.client.post("/healthz", json={}),
                await svc.client.get("/docs"),
                await svc.client.get("/openapi.json"),
            ]

    missing, wrong_method, probe, docs, schema = run(scenario())

    assert (missing.status_code, error_of(missing)["code"]) == (404, "not_found")
    assert (wrong_method.status_code, error_of(wrong_method)["code"]) == (
        405,
        "method_not_allowed",
    )
    assert wrong_method.headers["allow"] == "POST"
    assert (probe.status_code, error_of(probe)["code"]) == (405, "method_not_allowed")
    assert (docs.status_code, schema.status_code) == (404, 404)


def test_an_unexpected_exception_of_a_route_is_a_500_in_the_error_shape(
    recording_log: RecordingLog,
) -> None:
    async def boom() -> None:
        raise RuntimeError(f"a bug that mentions {SECRET}")

    async def scenario() -> tuple[Any, str]:
        async with make_client(recording_log) as svc:
            svc.app.add_api_route("/boom", boom, methods=["GET"])
            response = await svc.client.get("/boom")
            return response, (await svc.client.get("/metrics")).text

    response, metrics = run(scenario())

    assert response.status_code == 500
    assert error_of(response)["code"] == "internal"
    assert SECRET not in response.text
    assert sample(metrics, "rinchen_requests_total", route="other", status="500") == 1
    (event,) = recording_log.named("unhandled_error")
    assert event == {"error_type": "RuntimeError"}


def test_a_route_with_a_parameter_that_fails_validation_answers_in_the_error_shape(
    recording_log: RecordingLog,
) -> None:
    async def probe(x: int) -> dict[str, int]:
        return {"x": x}

    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            svc.app.add_api_route("/probe", probe, methods=["GET"])
            return await svc.client.get("/probe", params={"x": SECRET})

    response = run(scenario())

    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "validation_error"
    assert error["message"].startswith("query.x:")
    assert SECRET not in response.text
    (event,) = [e for e in recording_log.named("request") if e["route"] == "other"]
    assert (event["status"], event["code"]) == (422, "validation_error")


# ---- no text of a request is anywhere -----------------------------------------------------


def test_no_response_log_event_or_metric_holds_a_text_of_a_request(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[list[Any], str]:
        engine = FakeEngine()
        async with make_client(
            recording_log,
            engine,
            inter_threads=1,
            max_texts=2,
            max_queue_sentences=2,
            max_body_bytes=2048,
            request_timeout_seconds=0.3,
        ) as svc:
            refusals = []
            ok = await post(svc.client, texts=[SECRET, f"{SECRET} 2"], n_best=2)
            assert ok.status_code == 200
            refusals.append(await post(svc.client, texts=[SECRET], tgt="deu_Latn"))
            refusals.append(await post(svc.client, texts=[SECRET], beam_size="x"))
            refusals.append(
                await post_raw(
                    svc, json.dumps({"texts": [SECRET * 400], "tgt": KHK}).encode()
                )
            )
            refusals.append(await post_raw(svc, b'{"texts": ["' + SECRET.encode()))
            engine.translate_failure = UpstreamError("the library failed")
            refusals.append(await post(svc.client, texts=[SECRET]))
            engine.translate_failure = RuntimeError(f"failed on {SECRET}")
            refusals.append(await post(svc.client, texts=[SECRET]))
            engine.translate_failure = None
            engine.hold()
            running = asyncio.create_task(post(svc.client, texts=[SECRET]))
            await engine_started(engine, 4)
            queued = asyncio.create_task(post(svc.client, texts=[SECRET, SECRET]))
            await until(lambda: svc.state.batcher.depth == 2)
            refusals.append(await post(svc.client, texts=[SECRET]))
            refusals.extend(await asyncio.gather(running, queued))
            engine.release()
            metrics = (await svc.client.get("/metrics")).text
        return refusals, metrics

    refusals, metrics = run(scenario())

    assert sorted({r.status_code for r in refusals}) == [400, 413, 422, 500, 503, 504]
    for response in refusals:
        assert SECRET not in response.text
        assert error_of(response)
    assert SECRET not in metrics
    assert SECRET not in recording_log.text()
