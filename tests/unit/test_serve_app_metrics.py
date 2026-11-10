"""The service (D11, Task 9): what it counts and what it logs.

`GET /metrics` is Prometheus text of one registry per app: every family of the design, a count
that equals the number of requests made, the gauges of the queue and the batches in flight, the
counters of the engine mirrored as events. The log says each request with its route, status,
size and target, and never a text. Every test runs the app in process over a `FakeEngine`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rinchen.infer.engine_types import ENGINE_COUNTERS
from tests.support.serve_app import (
    KHK,
    FakeEngine,
    RecordingLog,
    engine_started,
    make_app,
    make_client,
    post,
    run,
    sample,
    serve_config,
    until,
)

METRIC_FAMILIES = {
    "rinchen_requests_total": "counter",
    "rinchen_request_duration_seconds": "histogram",
    "rinchen_translated_sentences_total": "counter",
    "rinchen_batch_size": "histogram",
    "rinchen_queue_depth": "gauge",
    "rinchen_inflight_batches": "gauge",
    "rinchen_script_retries_total": "counter",
    "rinchen_engine_events_total": "counter",
    "rinchen_ready": "gauge",
    "rinchen_draining": "gauge",
    "rinchen_model_info": "gauge",
}


# ---- the metrics --------------------------------------------------------------------------


async def scrape(svc: Any) -> str:
    response = await svc.client.get("/metrics")
    assert response.status_code == 200
    text: str = response.text
    return text


def test_metrics_are_prometheus_text_with_every_family_of_the_design(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, str]:
        async with make_client(recording_log) as svc:
            response = await svc.client.get("/metrics")
        return response, response.text

    response, text = run(scenario())

    assert (
        response.headers["content-type"] == "text/plain; version=1.0.0; charset=utf-8"
    )
    for name, kind in METRIC_FAMILIES.items():
        assert f"# TYPE {name} {kind}" in text
    assert "_created" not in text


def test_the_counts_are_the_number_of_requests_and_sentences_made(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[str, FakeEngine]:
        engine = FakeEngine()
        async with make_client(recording_log, engine) as svc:
            await asyncio.gather(
                *(post(svc.client, texts=[f"text{i}"]) for i in range(64))
            )
            return await scrape(svc), engine

    text, engine = run(scenario())

    assert sample(text, "rinchen_requests_total", route="translate", status="200") == 64
    assert sample(text, "rinchen_translated_sentences_total", status="ok") == 64
    assert sample(text, "rinchen_batch_size_count") == len(engine.calls)
    assert sample(text, "rinchen_batch_size_sum") == 64
    assert (
        sample(text, "rinchen_request_duration_seconds_count", route="translate") == 64
    )


def test_sentences_are_counted_by_the_status_they_got(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            await post(svc.client, texts=["hello", "", "silent", "cut...", "x" * 31])
            return await scrape(svc)

    text = run(scenario())

    for status in ("ok", "empty", "no_output", "truncated", "too_long"):
        assert sample(text, "rinchen_translated_sentences_total", status=status) == 1


def test_a_request_the_engine_refuses_is_counted_under_400_and_not_under_200(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            refused = await post(svc.client, tgt="deu_Latn")
            assert refused.status_code == 400
            return await scrape(svc)

    text = run(scenario())

    assert sample(text, "rinchen_requests_total", route="translate", status="400") == 1
    assert (
        sample(text, "rinchen_requests_total", route="translate", status="200") is None
    )
    assert sample(text, "rinchen_translated_sentences_total", status="ok") is None


def test_every_route_counts_its_own_responses(recording_log: RecordingLog) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            await svc.client.get("/healthz")
            await svc.client.get("/healthz")
            await svc.client.get("/readyz")
            await svc.client.get("/no/such/path")
            await svc.client.get("/v1/translate")
            await svc.client.get("/metrics")
            return await scrape(svc)

    text = run(scenario())

    expected = {
        ("healthz", "200"): 2,
        ("readyz", "200"): 1,
        ("other", "404"): 1,
        ("translate", "405"): 1,
        ("metrics", "200"): 1,
    }
    for (route, status), count in expected.items():
        assert (
            sample(text, "rinchen_requests_total", route=route, status=status) == count
        )


def test_the_gauges_show_the_queue_and_the_batches_in_flight(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[str, str]:
        engine = FakeEngine()
        async with make_client(recording_log, engine, inter_threads=1) as svc:
            engine.hold()
            first = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            queued = asyncio.create_task(post(svc.client, texts=["a1", "b2"]))
            await until(lambda: svc.state.batcher.depth == 2)
            while_held = await scrape(svc)
            engine.release()
            await asyncio.gather(first, queued)
            await until(lambda: svc.state.batcher.inflight == 0)
            return while_held, await scrape(svc)

    while_held, after = run(scenario())

    assert sample(while_held, "rinchen_inflight_batches") == 1
    assert sample(while_held, "rinchen_queue_depth") == 2
    assert sample(after, "rinchen_inflight_batches") == 0
    assert sample(after, "rinchen_queue_depth") == 0


def test_the_counters_of_the_engine_are_mirrored_as_events(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[str, str]:
        engine = FakeEngine(counter_values={"script_retries": 3, "zero_shot": 2})
        async with make_client(recording_log, engine) as svc:
            first = await scrape(svc)
            engine.counter_values["script_retries"] = 5
            return first, await scrape(svc)

    first, second = run(scenario())

    assert sample(first, "rinchen_engine_events_total", event="script_retries") == 3
    assert sample(first, "rinchen_engine_events_total", event="zero_shot") == 2
    assert sample(first, "rinchen_script_retries_total") == 3
    assert sample(second, "rinchen_engine_events_total", event="script_retries") == 5
    assert sample(second, "rinchen_script_retries_total") == 5
    for event in ENGINE_COUNTERS:
        assert sample(first, "rinchen_engine_events_total", event=event) is not None


def test_the_model_and_the_state_of_the_service_are_in_the_metrics(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            return await scrape(svc)

    text = run(scenario())

    assert (
        sample(
            text,
            "rinchen_model_info",
            name="toy-model",
            quantization="float32",
            device="cpu",
            compute_type="float32",
        )
        == 1
    )
    assert sample(text, "rinchen_ready") == 1
    assert sample(text, "rinchen_draining") == 0


def test_the_metrics_are_not_ready_before_the_start(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_app(recording_log) as svc:
            return await scrape(svc)

    text = run(scenario())

    assert sample(text, "rinchen_ready") == 0
    assert sample(text, "rinchen_draining") == 0


def test_two_apps_in_one_process_do_not_share_their_counts(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[str, str]:
        async with (
            make_client(recording_log) as first,
            make_client(recording_log) as second,
        ):
            for _ in range(3):
                await post(first.client)
            await post(second.client)
            return await scrape(first), await scrape(second)

    first, second = run(scenario())

    for text, count in ((first, 3), (second, 1)):
        assert (
            sample(text, "rinchen_requests_total", route="translate", status="200")
            == count
        )
        assert sample(text, "rinchen_translated_sentences_total", status="ok") == count


# ---- the log ------------------------------------------------------------------------------


def test_a_request_is_logged_with_its_route_status_size_and_target_and_no_text(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> None:
        async with make_client(recording_log) as svc:
            await post(svc.client, texts=["first text", "second text"])

    run(scenario())

    (event,) = [e for e in recording_log.named("request")]
    milliseconds = event.pop("milliseconds")
    assert isinstance(milliseconds, float)
    assert milliseconds >= 0
    assert event == {"route": "translate", "status": 200, "texts": 2, "target": KHK}
    assert "first text" not in recording_log.text()


def test_a_request_that_asks_for_a_target_the_model_does_not_write_is_logged_without_it(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> None:
        async with make_client(recording_log) as svc:
            await post(svc.client, tgt="not a language")

    run(scenario())

    (event,) = recording_log.named("request")
    assert "target" not in event
    assert (event["status"], event["code"], event["texts"]) == (400, "bad_request", 1)
    assert "not a language" not in recording_log.text()


def test_a_request_to_an_unknown_path_is_logged_under_other(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> None:
        async with make_client(recording_log) as svc:
            await svc.client.get("/some/secret-path")

    run(scenario())

    (event,) = recording_log.named("request")
    assert (event["route"], event["status"], event["code"]) == (
        "other",
        404,
        "not_found",
    )
    assert "texts" not in event


def test_the_start_of_the_service_is_logged_with_the_model_and_the_limits(
    recording_log: RecordingLog,
) -> None:
    config = serve_config(
        inter_threads=3, intra_threads=2, max_texts=10, max_body_bytes=2048
    )

    async def scenario() -> None:
        async with make_client(recording_log, config=config):
            pass

    run(scenario())

    (start,) = recording_log.named("serve_start")
    assert start["model"] == "toy-model"
    assert (start["quantization"], start["device"], start["compute_type"]) == (
        "float32",
        "cpu",
        "float32",
    )
    assert (start["inter_threads"], start["intra_threads"]) == (3, 2)
    assert (start["max_texts"], start["max_body_bytes"]) == (10, 2048)
    assert start["model_sha256"] == "a" * 64
    assert start["manifest_sha256"] == "b" * 64
