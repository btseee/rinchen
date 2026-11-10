"""The service (D11, Task 9): the stop.

`uvicorn` stops accepting connections on SIGTERM, waits for the requests in flight and then runs
the end of the lifespan; the end of the lifespan flips readiness to false FIRST, then closes
the batcher (it runs what is queued, waits for what is in flight and fails with a 503 what is
still unanswered after its share of the shutdown budget) and only then closes the engine. These
tests run the end of the lifespan by hand, with the engine held on a gate so that requests are in
flight while it runs.

All of it is in process. Over a socket `uvicorn` has closed the listener and answered the
requests in flight before it runs the end of the lifespan, so the false `/readyz` and the 503
`draining` below are not observable over HTTP (a client gets a refused connection), and the
batcher's own drain has little left to do on a real server.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from rinchen.serve.batcher import MicroBatcher
from tests.support.serve_app import (
    FakeEngine,
    RecordingLog,
    engine_started,
    error_of,
    make_app,
    make_client,
    post,
    run,
    sample,
    until,
)


def test_the_service_is_not_ready_from_the_first_moment_of_the_stop_and_refuses_new_requests(
    recording_log: RecordingLog,
) -> None:
    """In process only: over a socket the listener is closed before this state exists."""

    async def scenario() -> dict[str, Any]:
        engine = FakeEngine()
        async with make_app(recording_log, engine) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            seen: dict[str, Any] = {"before": await svc.client.get("/readyz")}
            engine.hold()
            held = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            stopping = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await until(lambda: svc.state.draining)
            seen["ready"] = await svc.client.get("/readyz")
            seen["late"] = await post(svc.client)
            seen["health"] = await svc.client.get("/healthz")
            seen["metrics"] = (await svc.client.get("/metrics")).text
            seen["waiting"] = not stopping.done()
            engine.release()
            seen["held"] = await held
            await stopping
            seen["after"] = await svc.client.get("/readyz")
        return seen

    seen = run(scenario())

    assert seen["before"].status_code == 200
    assert seen["ready"].status_code == 503
    assert error_of(seen["ready"])["code"] == "draining"
    assert seen["late"].status_code == 503
    assert error_of(seen["late"])["code"] == "draining"
    assert seen["late"].headers["retry-after"] == "1"
    assert seen["health"].status_code == 200
    assert sample(seen["metrics"], "rinchen_ready") == 0
    assert sample(seen["metrics"], "rinchen_draining") == 1
    assert seen["waiting"] is True
    assert seen["held"].status_code == 200
    assert seen["held"].json()["translations"] == ["<held>"]
    assert seen["after"].status_code == 503


def test_30_requests_in_flight_when_the_stop_begins_are_all_answered_and_none_is_lost(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[list[Any], Any, FakeEngine, dict[str, int]]:
        engine = FakeEngine()
        async with make_app(
            recording_log, engine, inter_threads=2, max_batch_sentences=8
        ) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            engine.hold()
            requests = [
                asyncio.create_task(post(svc.client, texts=[f"text{i}"]))
                for i in range(30)
            ]
            await engine_started(engine, 2)
            await until(lambda: svc.state.batcher.depth + engine.started_items() == 30)
            at_begin = {
                "queued": svc.state.batcher.depth,
                "inflight": svc.state.batcher.inflight,
            }
            stopping = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await until(lambda: svc.state.draining)
            ready = await svc.client.get("/readyz")
            engine.release()
            answered = list(await asyncio.gather(*requests))
            await stopping
        return answered, ready, engine, at_begin

    answered, ready, engine, at_begin = run(scenario())

    assert ready.status_code == 503
    assert [r.status_code for r in answered] == [200] * 30
    assert [r.json()["translations"] for r in answered] == [
        [f"<text{i}>"] for i in range(30)
    ]
    assert recording_log.named("shutdown_begin") == [at_begin]
    assert at_begin["inflight"] == 2
    assert recording_log.named("shutdown_drained") == [{"completed": 30, "failed": 0}]
    names = recording_log.names()
    assert names.index("shutdown_begin") < names.index("shutdown_drained")
    assert engine.closed == 1
    assert engine.answered_when_closed == len(engine.calls)


def test_a_stop_with_nothing_in_flight_closes_the_engine_and_says_so(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> FakeEngine:
        engine = FakeEngine()
        async with make_app(recording_log, engine) as svc:
            async with svc.app.router.lifespan_context(svc.app):
                assert (await post(svc.client)).status_code == 200
            assert engine.closed == 1
        return engine

    run(scenario())

    assert recording_log.named("shutdown_begin") == [{"queued": 0, "inflight": 0}]
    assert recording_log.named("shutdown_drained") == [{"completed": 0, "failed": 0}]


def test_a_batch_that_fails_during_the_stop_is_counted_as_failed(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        engine = FakeEngine()
        async with make_app(recording_log, engine) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            engine.hold()
            held = asyncio.create_task(post(svc.client, texts=["a1", "b2"]))
            await engine_started(engine, 1)
            stopping = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await until(lambda: svc.state.draining)
            engine.translate_failure = UpstreamError("the library failed")
            engine.release()
            response = await held
            await stopping
        return response

    response = run(scenario())

    assert response.status_code == 500
    assert recording_log.named("shutdown_drained") == [{"completed": 0, "failed": 2}]


def test_a_request_the_engine_refuses_during_the_stop_is_counted_as_failed_and_the_others_as_completed(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[Any]:
        engine = FakeEngine()
        async with make_app(recording_log, engine, inter_threads=1) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            engine.hold()
            held = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            good = asyncio.create_task(post(svc.client, texts=["a1", "b2"]))
            bad = asyncio.create_task(post(svc.client, texts=["c3"], tgt="deu_Latn"))
            await until(lambda: svc.state.batcher.depth == 3)
            stopping = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await until(lambda: svc.state.draining)
            engine.release()
            answered = list(await asyncio.gather(held, good, bad))
            await stopping
        return answered

    answered = run(scenario())

    assert [r.status_code for r in answered] == [200, 200, 400]
    assert recording_log.named("shutdown_drained") == [{"completed": 3, "failed": 1}]


def test_what_is_still_unanswered_after_the_shutdown_timeout_is_a_503_and_the_engine_stays_loaded(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[list[Any], FakeEngine]:
        engine = FakeEngine()
        async with make_app(
            recording_log, engine, inter_threads=1, shutdown_timeout_seconds=1.5
        ) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            engine.hold()
            running = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            queued = asyncio.create_task(post(svc.client, texts=["a1", "b2"]))
            await until(lambda: svc.state.batcher.depth == 2)
            await lifespan.__aexit__(None, None, None)
            answered = list(await asyncio.gather(running, queued))
            closed_while_stuck = engine.closed
            engine.release()
            assert await asyncio.to_thread(engine.wait_for_finished, 1)
            assert closed_while_stuck == 0
        return answered, engine

    answered, engine = run(scenario())

    assert [r.status_code for r in answered] == [503, 503]
    assert [error_of(r)["code"] for r in answered] == ["draining", "draining"]
    assert [r.headers["retry-after"] for r in answered] == ["1", "1"]
    assert recording_log.named("shutdown_drained") == [{"completed": 0, "failed": 3}]
    assert recording_log.named("engine_left_loaded") == [{"batches": 1}]
    assert engine.closed == 0


def test_an_engine_that_fails_to_close_still_gets_its_drain_logged(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine(close_failure=UpstreamError("the library failed to unload"))

    async def scenario() -> None:
        async with make_app(recording_log, engine) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            await lifespan.__aenter__()
            with pytest.raises(UpstreamError, match="failed to unload"):
                await lifespan.__aexit__(None, None, None)

    run(scenario())

    assert recording_log.named("shutdown_drained") == [{"completed": 0, "failed": 0}]


def test_the_stop_gives_the_batcher_only_the_second_share_of_the_shutdown_budget(
    recording_log: RecordingLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server has waited for the connections and the requests in flight before the end of
    # the lifespan runs (the first share, `graceful_wait_seconds`): giving the batcher the
    # whole timeout as well would let a stop take twice the timeout.
    timeouts: list[float] = []
    close = MicroBatcher.close

    async def recording_close(self: MicroBatcher[Any, Any, Any], timeout: float) -> int:
        timeouts.append(timeout)
        return await close(self, timeout)

    monkeypatch.setattr(MicroBatcher, "close", recording_close)

    async def scenario() -> None:
        async with make_client(recording_log, shutdown_timeout_seconds=5.0):
            pass

    run(scenario())

    assert timeouts == [2.0]
