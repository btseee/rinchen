"""A real `rinchen` service over a fake engine, as a process of its own, for the tests that
measure how a stop behaves on a real `uvicorn` (D11, fix wave).

    python -m tests.support.fake_serve_process PORT DELAY_SECONDS SHUTDOWN_SECONDS

serves `FakeEngine(delay_seconds=DELAY_SECONDS)` on the loopback (the default host) through `rinchen.cli.serve.
run_server`, the one place the command binds a port, with a shutdown timeout of SHUTDOWN_SECONDS.
A translate request then takes DELAY_SECONDS inside the engine, so a delay longer than the timeout
is a batch that is stuck when the stop begins. No model is loaded and no GPU is involved; the
process logs the JSON events of the service on standard error, like the command does.
"""

from __future__ import annotations

import sys

from rinchen.cli.serve import run_server
from rinchen.config.serve import ServeConfig
from rinchen.serve.app import create_app
from tests.support.serve_app import FakeEngine


def main(arguments: list[str]) -> None:
    assert sys.version_info[:2] == (3, 14)
    port, delay, timeout = int(arguments[0]), float(arguments[1]), float(arguments[2])
    config = ServeConfig(
        port=port,
        shutdown_timeout_seconds=timeout,
        max_wait_ms=1.0,
        inter_threads=1,
    )
    run_server(create_app(config, engine=FakeEngine(delay_seconds=delay)), config)


if __name__ == "__main__":
    main(sys.argv[1:])
