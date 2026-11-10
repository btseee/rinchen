"""The service (D11, Task 9) behind a real `uvicorn` on 127.0.0.1, in a thread of the test.

`ASGITransport` has no sockets and no HTTP parser, so what a server does with a body it does
not read, with `Connection: close` and with a stop that has a request in flight is checked here
once, over a real connection, on a free port chosen by the operating system. The server runs
over a `FakeEngine`; no model is loaded and nothing but the loopback is touched. The process's
own exit status after SIGTERM is not a thing a thread can show: the serving gate measures it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import uvicorn
from fastapi import FastAPI

from rinchen.serve.app import create_app
from tests.support.serve_app import (
    KHK,
    FakeEngine,
    RecordingLog,
    error_of,
    sample,
    serve_config,
)

START_SECONDS = 30.0
STOP_SECONDS = 30.0
JSON = {"content-type": "application/json"}


@dataclass
class Listening:
    url: str
    server: uvicorn.Server
    thread: threading.Thread


@contextmanager
def listening(app: FastAPI) -> Iterator[Listening]:
    """`app` served by a real uvicorn on a free loopback port until the block ends."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=10,
        server_header=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-under-test")
    thread.start()
    try:
        pause = threading.Event()
        for _ in range(int(START_SECONDS * 100)):
            if server.started or not thread.is_alive():
                break
            pause.wait(0.01)
        assert server.started, "the server did not start"
        port = server.servers[0].sockets[0].getsockname()[1]
        yield Listening(f"http://127.0.0.1:{port}", server, thread)
    finally:
        server.should_exit = True
        thread.join(STOP_SECONDS)
        assert not thread.is_alive(), "the server did not stop"


def request_body(*texts: str) -> bytes:
    return json.dumps({"texts": list(texts), "tgt": KHK}).encode()


def test_a_request_is_answered_over_a_real_socket_and_counted(
    recording_log: RecordingLog,
) -> None:
    app = create_app(serve_config(), engine=FakeEngine())  # type: ignore[arg-type]

    with (
        listening(app) as served,
        httpx.Client(base_url=served.url, timeout=30) as client,
    ):
        answer = client.post(
            "/v1/translate", content=request_body("one", "two"), headers=JSON
        )
        ready = client.get("/readyz")
        metrics = client.get("/metrics")

    assert answer.status_code == 200
    assert answer.json()["translations"] == ["<one>", "<two>"]
    assert ready.status_code == 200
    assert metrics.headers["content-type"] == "text/plain; version=1.0.0; charset=utf-8"
    assert (
        sample(metrics.text, "rinchen_requests_total", route="translate", status="200")
        == 1
    )


def test_a_body_over_the_limit_is_a_413_that_closes_the_connection_and_the_server_goes_on(
    recording_log: RecordingLog,
) -> None:
    app = create_app(serve_config(max_body_bytes=1024), engine=FakeEngine())  # type: ignore[arg-type]
    chunks = iter([b" " * 600, b" " * 600])

    with (
        listening(app) as served,
        httpx.Client(base_url=served.url, timeout=30) as client,
    ):
        by_length = client.post("/v1/translate", content=b" " * 2000, headers=JSON)
        chunked = client.post("/v1/translate", content=chunks, headers=JSON)
        after = client.post("/v1/translate", content=request_body("fits"), headers=JSON)

    for refused in (by_length, chunked):
        assert refused.status_code == 413
        assert error_of(refused)["code"] == "payload_too_large"
        assert refused.headers["connection"] == "close"
    assert chunked.request.headers["transfer-encoding"] == "chunked"
    assert after.status_code == 200


def test_a_stop_of_the_server_waits_for_the_request_in_flight_and_then_closes_the_engine(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine()
    app = create_app(serve_config(), engine=engine)  # type: ignore[arg-type]

    with (
        listening(app) as served,
        httpx.Client(base_url=served.url, timeout=30) as client,
    ):
        engine.hold()
        with ThreadPoolExecutor(max_workers=1) as pool:
            inflight = pool.submit(
                client.post, "/v1/translate", content=request_body("held"), headers=JSON
            )
            assert engine.wait_for_calls(1)
            served.server.should_exit = True
            engine.release()
            answer = inflight.result(timeout=STOP_SECONDS)
        served.thread.join(STOP_SECONDS)

    assert answer.status_code == 200
    assert answer.json()["translations"] == ["<held>"]
    assert not served.thread.is_alive()
    assert engine.closed == 1
    (drained,) = recording_log.named("shutdown_drained")
    assert drained["failed"] == 0
