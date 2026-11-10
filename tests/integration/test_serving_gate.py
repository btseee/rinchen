"""The D11 serving gate, the service itself: `rinchen serve` over the real engine and the real
CTranslate2 on the CPU, in process (`TestClient`, `ASGITransport`) and as a real subprocess of the
test on the loopback. Nothing here touches a GPU (`CUDA_VISIBLE_DEVICES` is the empty string for
the session, every service runs on `device: cpu`, never `auto`); no image is built or run.

The data are the toy corpora of `tests/support/tiny_parallel.py`: templated sentences, so every
number below tests machinery and ranks nothing. The model is the float32 export of the source-tag
teacher and, for S7, its int8 export (both from `tests/integration/conftest.py`, which trains and
exports them once per session). The settings are `configs/serve/tiny.yaml`.

Gates (the 2 s is the gate and is never tightened or loosened; every other number is a count):

    S1  `TestClient` end to end over the real engine: `/readyz` names the model and its
        quantization, `This is the star.` into `khk_Cyrl` is the engine's translation, `/healthz`,
        an unknown target and a script on an English target are 400 with the engine's message,
        `n_best` 3 gives the engine's alternatives, `/metrics` has the names of T17
    S1b readiness is false while the warm-up runs (the real engine behind a gate) and true after
    S2  64 concurrent requests, in process and against the real `rinchen serve`: all 200, p95
        below 2.0 s, 64 requests of 8 sentences also, every translation the engine's, and the
        counters of `/metrics` equal to the requests and sentences sent (warm-up included)
    S2c a reader of the engine's counters never sees half a call while 8 workers commit
    S3b the stop with work queued (in process, the real engine held on a gate): readiness is false
        first, every request in flight is answered 200, `shutdown_drained` has failed 0. This is
        the only test that reaches the batcher's own drain: on the real server uvicorn waits for
        the requests in flight before the lifespan, and so the batcher's queue is already empty
        (S3 in `test_serving_gate_process.py`; the mutation check showed it)
    S7  the int8 export serves the same requests: 200, the engine's int8 translations, `/readyz`
        says `int8_float32` and `int8`

The real-server stop (S3), the compose command (S4), the missing GPU stack (S5) and the refused
models (S6) are in `test_serving_gate_process.py`.

Measured on this machine (CPU, Eole 0.6.4, CTranslate2 4.8.4, CPython 3.14.10, 2026-11-10; the
machine was shared with other work, so a time moves by a third from one run to the next, a count
or a score never does; the spike measured p95 24 to 28 ms in process and 48 ms over a socket):

    S2 in process      64 x 1 sentence: p50 23.5 ms, p95 26.8 ms; 64 x 8 sentences: p50 56.8 ms,
                       p95 84.3 ms; all 200; /metrics change 136 requests and 584 sentences, the
                       number sent (8 warm-up + 64 + 64 requests; 8 + 64 + 512 sentences), every
                       sentence `ok`, every translation the engine's
    S2 real server     ready after 0.66 s; 64 x 1: p50 54.5 ms, p95 91.1 ms; 64 x 8: p50 72.4 ms,
                       p95 120.4 ms; the same counts (the 2 s bound has a margin of 20 times)
    S2c                1,200 empty sentences committed by 8 workers with the switch interval at
                       1 microsecond while a thread read the counters 723,761 times: no reading
                       with `sources` different from `empty`; without the lock (mutant) 1,505
                       readings were torn in one run
    S3b                128 sentences queued and 64 given to the held engine when the stop began;
                       readiness 503 `draining`; a new request 503 `draining` with `Retry-After`
                       1; all 24 requests 200; `shutdown_drained` completed 192, failed 0
    S7                 int8 server ready after 0.66 s; compute_type `int8_float32`, quantization
                       `int8`; 64 x 1: p95 73.5 ms; 64 x 8: p95 138.7 ms; all 200 and every
                       translation the engine's int8 translation; 1 of the 100 sentences differs
                       from the float32 translation (a count on a toy, no quality claim)

The module takes about 17 s on top of the session fixtures (the teachers and exports, 119 s on a
machine with a load average of 3 to 5 and 3.5 minutes at a load of 12).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from fastapi.testclient import TestClient

from rinchen.errors import ConfigError
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import DecodeOptions, Row
from rinchen.serve.app import create_app
from tests.support.d11_gate import Exports, open_engine
from tests.support.serve_app import RecordingLog, error_of, run, sample, until
from tests.support.serving_gate import (
    CONCURRENT_REQUESTS,
    SENTENCES_PER_LARGE_REQUEST,
    WARM_UP_REQUESTS,
    HeldEngine,
    Load,
    Reply,
    Server,
    burst,
    differing_translations,
    engine_translations,
    free_port,
    many_sentence_bodies,
    run_load,
    running_server,
    send,
    serve_command,
    summarize,
    tiny_config,
    translate_body,
)

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures(
        "d11_gpu_hidden", "d11_serving_log", "d11_no_server_is_left"
    ),
]

# The gate. It is never tightened or loosened: the spike measured 24 to 48 ms.
P95_LIMIT_SECONDS: Final = 2.0
STAR: Final = "This is the star."
STAR_IN_KHALKHA: Final = "Энэ бол од."
# What a request that names a beam and no more gets on the tiny config.
DECODING: Final = DecodeOptions(beam_size=4, max_decoding_length=128)
# The families of `/metrics` the spec names (T17) and their types.
METRIC_FAMILIES: Final = {
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
LARGE_REQUESTS: Final = 24
# 100 requests of 4 empty texts fit the queue of `configs/serve/tiny.yaml` (512 sentences).
EMPTY_REQUESTS: Final = 100
EMPTY_ROUNDS: Final = 3


# ---------------------------------------------------------------------------------
# The fixtures: `tests/support/serving_gate_fixtures.py` has the logs, the sentences and the
# reference engines; these are the services.


@pytest.fixture(scope="module")
def service(
    d11_tiny_export_dir: Path, d11_serving_log: RecordingLog
) -> Iterator[TestClient]:
    """The app over the float32 export, started (the engine opened and warmed up in its
    lifespan) and stopped by the client's context."""
    app = create_app(tiny_config(), model=d11_tiny_export_dir)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def float32_server(
    d11_tiny_export_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Server]:
    port = free_port()
    with running_server(
        serve_command(d11_tiny_export_dir, port),
        tmp_path_factory.mktemp("serve_float32"),
        port=port,
        name="float32",
    ) as server:
        yield server


@pytest.fixture(scope="module")
def int8_server(
    d11_exports: Exports, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Server]:
    port = free_port()
    with running_server(
        serve_command(d11_exports.int8.directory, port),
        tmp_path_factory.mktemp("serve_int8"),
        port=port,
        name="int8",
    ) as server:
        yield server


def _async_client(app: Any) -> httpx.AsyncClient:
    """A client that calls `app` directly (this transport does not run the lifespan)."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://service",
        timeout=60.0,
    )


def _socket_client(server: Server) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=server.url,
        timeout=60.0,
        limits=httpx.Limits(max_connections=CONCURRENT_REQUESTS * 2),
    )


# ---------------------------------------------------------------------------------
# S1: the app over the real engine, through `TestClient`.


def test_readyz_names_the_model_and_its_quantization(
    service: TestClient, d11_exports: Exports
) -> None:
    response = service.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": d11_exports.float32.record["name"],
        "quantization": "float32",
        "device": "cpu",
        "compute_type": "float32",
    }


def test_healthz_answers_while_the_process_runs(service: TestClient) -> None:
    response = service.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_a_translation_is_the_engines_and_in_khalkha_cyrillic(
    service: TestClient, d11_float32_engine: Engine, d11_exports: Exports
) -> None:
    expected = d11_float32_engine.translate(
        [Row(STAR, "eng_Latn", "khk_Cyrl")], DECODING
    )[0]

    response = service.post("/v1/translate", json=translate_body([STAR]))

    assert response.status_code == 200
    body = response.json()
    assert expected.text == STAR_IN_KHALKHA
    assert body["translations"] == [STAR_IN_KHALKHA]
    assert body["statuses"] == ["ok"]
    assert body["detected_src"] == ["eng_Latn"]
    assert body["model"] == d11_exports.float32.record["name"]
    assert body["script_purity"] == [expected.script_purity]
    assert body["latency_ms"] > 0
    assert "alternatives" not in body


def test_an_unknown_target_is_400_with_the_engines_message(
    service: TestClient, d11_float32_engine: Engine
) -> None:
    with pytest.raises(ConfigError) as refused:
        d11_float32_engine.translate([Row(STAR, "eng_Latn", "xxx_Latn")], DECODING)

    response = service.post(
        "/v1/translate", json=translate_body([STAR], tgt="xxx_Latn")
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "bad_request", "message": str(refused.value)}
    }
    assert "xxx_Latn" in str(refused.value)


def test_a_mongolian_script_for_an_english_target_is_400_with_the_engines_message(
    service: TestClient, d11_float32_engine: Engine
) -> None:
    row = Row(STAR_IN_KHALKHA, "khk_Cyrl", "eng_Latn", "Mong")
    with pytest.raises(ConfigError) as refused:
        d11_float32_engine.translate([row], DECODING)

    response = service.post(
        "/v1/translate",
        json=translate_body(
            [STAR_IN_KHALKHA], src="khk_Cyrl", tgt="eng_Latn", script="Mong"
        ),
    )

    assert response.status_code == 400
    assert error_of(response) == {"code": "bad_request", "message": str(refused.value)}


def test_n_best_3_gives_the_engines_alternatives(
    service: TestClient, d11_float32_engine: Engine
) -> None:
    expected = d11_float32_engine.translate(
        [Row(STAR, "eng_Latn", "khk_Cyrl")],
        DecodeOptions(beam_size=4, n=3, max_decoding_length=128),
    )[0]

    response = service.post("/v1/translate", json=translate_body([STAR], n_best=3))

    assert response.status_code == 200
    body = response.json()
    assert body["translations"] == [expected.text]
    assert body["alternatives"] == [list(expected.alternatives)]
    assert len(body["alternatives"][0]) == 2
    assert expected.text not in body["alternatives"][0]


def test_metrics_has_every_name_of_the_spec_and_no_created_series(
    service: TestClient, d11_exports: Exports
) -> None:
    assert service.post("/v1/translate", json=translate_body([STAR])).status_code == 200

    response = service.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for family, kind in METRIC_FAMILIES.items():
        assert f"# TYPE {family} {kind}" in text, family
    assert "_created" not in text
    assert sample(text, "rinchen_ready") == 1.0
    assert sample(text, "rinchen_draining") == 0.0
    model = d11_exports.float32.record["name"]
    assert (
        sample(
            text,
            "rinchen_model_info",
            name=model,
            quantization="float32",
            device="cpu",
            compute_type="float32",
        )
        == 1.0
    )
    assert (sample(text, "rinchen_translated_sentences_total", status="ok") or 0) >= 1


# ---------------------------------------------------------------------------------
# S1b: readiness waits for the warm-up of the real engine.


def test_readiness_is_false_while_the_warm_up_runs_and_true_after_it(
    d11_tiny_export_dir: Path, d11_test_log: RecordingLog
) -> None:
    held = HeldEngine(open_engine(d11_tiny_export_dir))
    held.warmup_gate.clear()
    app = create_app(tiny_config(), engine=held)

    async def scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        async with _async_client(app) as client:
            stop, started = asyncio.Event(), asyncio.Event()

            async def lifespan() -> None:
                async with app.router.lifespan_context(app):
                    started.set()
                    await stop.wait()

            task = asyncio.create_task(lifespan())
            try:
                assert await asyncio.to_thread(held.warmup_entered.wait, 30)
                during_readyz = await client.get("/readyz")
                during_translate = await client.post(
                    "/v1/translate", json=translate_body([STAR])
                )
                assert not started.is_set()
                held.warmup_gate.set()
                await asyncio.wait_for(started.wait(), 30)
                after_readyz = await client.get("/readyz")
            finally:
                held.warmup_gate.set()
                stop.set()
                await task
            return during_readyz, during_translate, after_readyz

    during_readyz, during_translate, after_readyz = run(scenario())

    assert during_readyz.status_code == 503
    assert error_of(during_readyz)["code"] == "not_ready"
    assert during_translate.status_code == 503
    assert error_of(during_translate)["code"] == "not_ready"
    assert after_readyz.status_code == 200
    assert held.closed


# ---------------------------------------------------------------------------------
# S2: 64 concurrent requests.


def _assert_the_load_was_served(
    load: Load,
    expected: dict[str, str],
    record_property: Any,
    label: str,
) -> None:
    """The gate's assertions on a `Load`, which are the same in process and over a socket."""
    warm, singles, large = (
        summarize(each) for each in (load.warm, load.singles, load.large)
    )
    sentences_sent = (
        WARM_UP_REQUESTS
        + CONCURRENT_REQUESTS
        + CONCURRENT_REQUESTS * SENTENCES_PER_LARGE_REQUEST
    )
    requests_sent = WARM_UP_REQUESTS + 2 * CONCURRENT_REQUESTS

    assert warm.statuses == {200: WARM_UP_REQUESTS}
    assert singles.statuses == {200: CONCURRENT_REQUESTS}
    assert large.statuses == {200: CONCURRENT_REQUESTS}
    assert singles.p95 < P95_LIMIT_SECONDS
    assert large.p95 < P95_LIMIT_SECONDS
    assert singles.sentences == singles.ok_sentences == CONCURRENT_REQUESTS
    assert (
        large.sentences
        == large.ok_sentences
        == CONCURRENT_REQUESTS * SENTENCES_PER_LARGE_REQUEST
    )
    # Every translation is the one the engine writes for the same sentence on its own.
    assert differing_translations(load.replies, expected) == []
    # The counters of /metrics equal what was sent, the warm-up requests included.
    answered_ok = warm.ok_sentences + singles.ok_sentences + large.ok_sentences
    assert answered_ok == sentences_sent
    assert load.after.translate_200 - load.before.translate_200 == requests_sent
    assert load.after.ok_sentences - load.before.ok_sentences == answered_ok
    assert load.after.engine_sources - load.before.engine_sources == sentences_sent

    record_property(f"{label}_single_p50_ms", round(singles.p50 * 1000, 1))
    record_property(f"{label}_single_p95_ms", round(singles.p95 * 1000, 1))
    record_property(f"{label}_large_p50_ms", round(large.p50 * 1000, 1))
    record_property(f"{label}_large_p95_ms", round(large.p95 * 1000, 1))
    record_property(f"{label}_requests_counted", requests_sent)
    record_property(f"{label}_sentences_counted", sentences_sent)


def test_64_concurrent_requests_in_process_are_all_200_with_p95_under_two_seconds(
    d11_tiny_export_dir: Path,
    d11_sentences: list[str],
    d11_float32_engine: Engine,
    d11_test_log: RecordingLog,
    record_property: Any,
) -> None:
    expected = engine_translations(d11_float32_engine, d11_sentences)
    app = create_app(tiny_config(), model=d11_tiny_export_dir)

    async def scenario() -> Load:
        async with _async_client(app) as client, app.router.lifespan_context(app):
            return await run_load(client, d11_sentences)

    load = run(scenario())

    _assert_the_load_was_served(load, expected, record_property, "in_process")


def test_64_concurrent_requests_against_the_real_server_are_all_200_with_p95_under_two_seconds(
    float32_server: Server,
    d11_sentences: list[str],
    d11_float32_engine: Engine,
    record_property: Any,
) -> None:
    expected = engine_translations(d11_float32_engine, d11_sentences)

    async def scenario() -> Load:
        async with _socket_client(float32_server) as client:
            return await run_load(client, d11_sentences)

    load = run(scenario())

    record_property("server_ready_seconds", round(float32_server.ready_seconds, 2))
    _assert_the_load_was_served(load, expected, record_property, "server")
    assert float32_server.process.poll() is None


# ---------------------------------------------------------------------------------
# S2c: the engine's counters are whole while worker threads commit at once.


def test_a_reader_of_the_engine_counters_never_sees_half_a_call_while_eight_workers_commit(
    d11_tiny_export_dir: Path,
    d11_test_log: RecordingLog,
    record_property: Any,
) -> None:
    # Every text is empty, so each one-sentence batch adds 1 to `sources` and 1 to `empty`:
    # in any whole reading of the counters the two are equal.
    config = tiny_config(inter_threads=8, max_batch_sentences=1)
    app = create_app(config, model=d11_tiny_export_dir)
    bodies = [translate_body([""] * 4) for _ in range(EMPTY_REQUESTS)]
    sent = EMPTY_ROUNDS * EMPTY_REQUESTS * 4
    interval = sys.getswitchinterval()
    stop = threading.Event()
    torn: list[dict[str, int]] = []
    readings = 0

    def read(engine: Any) -> None:
        nonlocal readings
        while not stop.is_set():
            counters = engine.counters()
            readings += 1
            if counters["sources"] != counters["empty"]:
                torn.append(counters)

    async def scenario() -> list[Reply]:
        async with _async_client(app) as client, app.router.lifespan_context(app):
            reader = threading.Thread(target=read, args=(app.state.serving.engine,))
            # A thread is made to give way after a microsecond, so that the reader can run
            # between the two additions of a commit that is not under the engine's lock.
            sys.setswitchinterval(1e-6)
            reader.start()
            try:
                replies: list[Reply] = []
                for _ in range(EMPTY_ROUNDS):
                    replies += await burst(client, bodies)
                return replies
            finally:
                stop.set()
                reader.join(30)
                sys.setswitchinterval(interval)

    replies = run(scenario())

    assert {reply.status for reply in replies} == {200}
    assert {status for reply in replies for status in reply.body["statuses"]} == {
        "empty"
    }
    assert torn == []
    assert readings > 100
    engine = app.state.serving.engine
    counters = engine.counters()
    assert counters["sources"] == counters["empty"] == sent
    record_property("empty_sentences_committed", sent)
    record_property("counter_readings_during_the_commits", readings)


# ---------------------------------------------------------------------------------
# S3b: the stop with work queued, in process.


def test_a_stop_with_work_queued_closes_readiness_first_and_answers_every_request(
    d11_tiny_export_dir: Path,
    d11_sentences: list[str],
    d11_float32_engine: Engine,
    d11_test_log: RecordingLog,
    record_property: Any,
) -> None:
    expected = engine_translations(d11_float32_engine, d11_sentences)
    held = HeldEngine(open_engine(d11_tiny_export_dir))
    app = create_app(tiny_config(), engine=held)
    bodies = many_sentence_bodies(
        d11_sentences, LARGE_REQUESTS, SENTENCES_PER_LARGE_REQUEST
    )
    total = LARGE_REQUESTS * SENTENCES_PER_LARGE_REQUEST

    async def scenario() -> dict[str, Any]:
        async with _async_client(app) as client:
            state = app.state.serving
            stop, started = asyncio.Event(), asyncio.Event()

            async def lifespan() -> None:
                async with app.router.lifespan_context(app):
                    started.set()
                    await stop.wait()

            task = asyncio.create_task(lifespan())
            await asyncio.wait_for(started.wait(), 30)
            held.translate_gate.clear()
            requests = [asyncio.create_task(send(client, body)) for body in bodies]
            # Every sentence is either in a batch the held engine was given or queued.
            await until(lambda: held.rows_entered + state.batcher.depth == total)
            queued, given = state.batcher.depth, held.rows_entered
            stop.set()
            await until(lambda: not state.ready)
            during_readyz = await client.get("/readyz")
            during_translate = await client.post(
                "/v1/translate", json=translate_body([STAR])
            )
            held.translate_gate.set()
            replies = await asyncio.gather(*requests)
            await task
            return {
                "queued": queued,
                "given": given,
                "readyz": during_readyz,
                "translate": during_translate,
                "replies": replies,
            }

    outcome = run(scenario())

    assert outcome["queued"] > 0
    assert outcome["given"] > 0
    assert outcome["readyz"].status_code == 503
    assert error_of(outcome["readyz"])["code"] == "draining"
    assert outcome["translate"].status_code == 503
    assert error_of(outcome["translate"])["code"] == "draining"
    assert outcome["translate"].headers["retry-after"] == "1"
    replies = outcome["replies"]
    assert {reply.status for reply in replies} == {200}
    assert differing_translations(replies, expected) == []
    assert summarize(replies).ok_sentences == total
    (begin,) = d11_test_log.named("shutdown_begin")
    assert begin["queued"] + begin["inflight"] > 0
    (drained,) = d11_test_log.named("shutdown_drained")
    assert drained == {"completed": total, "failed": 0}
    assert held.closed
    record_property("drain_queued_sentences", outcome["queued"])
    record_property("drain_given_to_the_engine", outcome["given"])
    record_property("drain_completed", drained["completed"])


# ---------------------------------------------------------------------------------
# S7: the int8 export serves the same requests.


def test_the_int8_export_serves_the_same_requests_with_the_engines_int8_translations(
    int8_server: Server,
    d11_int8_engine: Engine,
    d11_float32_engine: Engine,
    d11_exports: Exports,
    d11_sentences: list[str],
    record_property: Any,
) -> None:
    ready = httpx.get(f"{int8_server.url}/readyz", timeout=10.0)
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model": d11_exports.int8.record["name"],
        "quantization": "int8",
        "device": "cpu",
        "compute_type": "int8_float32",
    }
    expected = engine_translations(d11_int8_engine, d11_sentences)

    async def scenario() -> Load:
        async with _socket_client(int8_server) as client:
            return await run_load(client, d11_sentences)

    load = run(scenario())

    _assert_the_load_was_served(load, expected, record_property, "int8")
    float32_expected = engine_translations(d11_float32_engine, d11_sentences)
    differing = sum(expected[text] != float32_expected[text] for text in d11_sentences)
    record_property(
        "int8_sentences_differing_from_float32", f"{differing}/{len(d11_sentences)}"
    )
    record_property("int8_server_ready_seconds", round(int8_server.ready_seconds, 2))
