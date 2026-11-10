"""The stop of the service on a real `uvicorn` is bounded by `shutdown_timeout_seconds` (D11, fix
wave).

`uvicorn` waits for the connections and the requests in flight first, cancels what is still
running and only then runs the end of the lifespan, which closes the micro-batcher and waits for
the same batch a second time. Both waits used to be the whole timeout, so a stop with a stuck
batch took twice as long, and `stop_grace_period` of the compose file, which covers the timeout
once, ended it with SIGKILL and no `shutdown_drained`. The timeout is now one budget the two waits
share (`ServeConfig.graceful_wait_seconds` and `drain_wait_seconds`).

The service runs over `FakeEngine` (tests/support/fake_serve_process.py): a request takes as long
as the test says inside the engine. Every server is on 127.0.0.1 with a free port, in a session of
its own, and is stopped through its process group only; no model, no GPU, no image.

Measured on this machine (CPU, 2026-11-10; the machine was shared, so a time moves by a third
from one run to the next): a batch stuck for 8 s with a shutdown timeout of 2 s ended the process
4.18 s after the signal before the fix (the request cancelled at +2.1 s, `engine_left_loaded` at
+4.1 s) and 2.1 to 2.2 s after it; with the default timeout of 20 s and a batch stuck for 30 s it
ended 20.2 s after the signal. The test below uses a timeout of 4 s so that a server that gave
both waits more than their share (`uvicorn` the whole timeout, or the batcher the whole timeout)
ends later than the margin allows.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx

from rinchen.config.loader import REPO_ROOT
from tests.support.serve_app import sample
from tests.support.serving_gate import (
    Server,
    free_port,
    server_environment,
    start_server,
    stop_server,
    translate_body,
    wait_for_exit,
    wait_until_ready,
)

# What the stop may take beyond the timeout: the loop of `uvicorn` looks at its flag and at the
# requests every 0.1 s, and the machine is shared. A server that gave both waits the whole
# timeout ends at about 8 s with the timeout of 4 s below, 1.5 s past the 6.5 s allowed here.
MARGIN_SECONDS: Final = 2.5
POLL_SECONDS: Final = 0.02
DEADLINE_SECONDS: Final = 60.0


@dataclass
class Client:
    """One translate request sent from a thread: when it ended and how."""

    server: Server
    status: int | str | None = None
    ended_at: float | None = None
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.thread = threading.Thread(target=self._send, daemon=True)
        self.thread.start()

    def _send(self) -> None:
        try:
            response = httpx.post(
                f"{self.server.url}/v1/translate",
                json=translate_body(["hello"]),
                timeout=DEADLINE_SECONDS,
            )
            self.status = response.status_code
        except httpx.HTTPError as error:
            self.status = type(error).__name__
        self.ended_at = time.perf_counter()


def start_fake_server(
    tmp_path: Path, *, delay_seconds: float, shutdown_seconds: float
) -> Server:
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "tests.support.fake_serve_process",
        str(port),
        str(delay_seconds),
        str(shutdown_seconds),
    ]
    # The sources of this tree first, so the server runs the code under test whichever tree
    # the environment's editable install points to.
    environment = server_environment(
        {
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
            "PYTHONUNBUFFERED": "1",
        }
    )
    server = start_server(command, tmp_path, port=port, environment=environment)
    wait_until_ready(server)
    return server


def wait_until_a_batch_is_running(server: Server) -> None:
    """Poll `/metrics` until the engine holds a batch: the request is inside the engine."""
    deadline = time.perf_counter() + DEADLINE_SECONDS
    while time.perf_counter() < deadline:
        metrics = httpx.get(f"{server.url}/metrics", timeout=5.0).text
        if sample(metrics, "rinchen_inflight_batches") == 1:
            return
        wait_for_exit(server.process, POLL_SECONDS)
    raise AssertionError("no batch ever ran in the engine")


def stop_and_time(server: Server, client: Client) -> tuple[float, float | None]:
    """Send SIGTERM to the server's group and return the seconds until the process ended and
    until the request was answered (`None` when it never was)."""
    sent = time.perf_counter()
    server.signal_group(signal.SIGTERM)
    assert wait_for_exit(server.process, DEADLINE_SECONDS), "the server never stopped"
    ended = time.perf_counter() - sent
    client.thread.join(DEADLINE_SECONDS)
    answered = None if client.ended_at is None else client.ended_at - sent
    return ended, answered


def test_a_stop_with_a_stuck_batch_ends_within_the_shutdown_timeout_and_a_margin(
    tmp_path: Path,
) -> None:
    server = start_fake_server(tmp_path, delay_seconds=12.0, shutdown_seconds=4.0)
    try:
        client = Client(server)
        wait_until_a_batch_is_running(server)

        ended, answered = stop_and_time(server, client)
    finally:
        returncode = stop_server(server, timeout=10.0)

    assert returncode == -signal.SIGTERM
    assert ended <= 4.0 + MARGIN_SECONDS, ended
    # The request is cut at the end of the server's wait: it is never left waiting for the
    # batch that is stuck.
    assert answered is not None and answered <= 4.0 + MARGIN_SECONDS, answered
    assert client.status == 500
    names = [event["event"] for event in server.events()]
    assert names.count("shutdown_begin") == 1
    assert names.count("engine_left_loaded") == 1
    assert names.count("shutdown_drained") == 1


def test_a_request_that_finishes_inside_the_budget_is_answered_and_the_stop_is_prompt(
    tmp_path: Path,
) -> None:
    server = start_fake_server(tmp_path, delay_seconds=0.8, shutdown_seconds=6.0)
    try:
        client = Client(server)
        wait_until_a_batch_is_running(server)

        ended, answered = stop_and_time(server, client)
    finally:
        returncode = stop_server(server, timeout=10.0)

    assert returncode == -signal.SIGTERM
    assert client.status == 200
    assert answered is not None
    # Not the whole timeout: the stop ends when the work does.
    assert ended < 3.0, ended
    assert not server.named("engine_left_loaded")
    (drained,) = server.named("shutdown_drained")
    assert drained["failed"] == 0
