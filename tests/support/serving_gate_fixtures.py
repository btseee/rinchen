"""Fixtures the two modules of the D11 serving gate share (`tests/integration/test_serving_gate.py`
and `test_serving_gate_process.py`), re-exported by `tests/integration/conftest.py` so that a test
names them as a parameter.

The service and the engine log through `rinchen.utils.logging.get`, whose first call configures the
root logger for the rest of the session and `test_logging` expects to be the first: a module of the
gate takes `d11_serving_log`, which makes every logger of the module a `RecordingLog`, before it
opens an engine or an app. The reference engines are per module for that reason (a session fixture
would be set up before the module's patch).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from rinchen.infer.engine import Engine
from tests.support.d11_gate import Exports, open_engine
from tests.support.distill_gate_stages import Workspace
from tests.support.serve_app import RecordingLog
from tests.support.serving_gate import dev_sentences, running_servers


@pytest.fixture(scope="module")
def d11_serving_log() -> Iterator[RecordingLog]:
    """Every logger of the module is this `RecordingLog`."""
    log = RecordingLog()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("rinchen.utils.logging.get", lambda name: log)
        yield log


@pytest.fixture
def d11_test_log(
    monkeypatch: pytest.MonkeyPatch, d11_serving_log: RecordingLog
) -> RecordingLog:
    """A log of its own for the service of one test."""
    log = RecordingLog()
    monkeypatch.setattr("rinchen.utils.logging.get", lambda name: log)
    return log


@pytest.fixture(scope="module")
def d11_sentences(d11_workspace: Workspace) -> list[str]:
    """The 100 English dev sentences the gate sends."""
    return dev_sentences(d11_workspace)


@pytest.fixture(scope="module")
def d11_float32_engine(
    d11_tiny_export_dir: Path, d11_serving_log: RecordingLog
) -> Iterator[Engine]:
    """The engine over the float32 export, which the tests compare the services with."""
    engine = open_engine(d11_tiny_export_dir)
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def d11_int8_engine(
    d11_exports: Exports, d11_serving_log: RecordingLog
) -> Iterator[Engine]:
    engine = open_engine(d11_exports.int8.directory)
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def d11_no_server_is_left() -> Iterator[None]:
    """After the module, no server it started is running (the module's own fixtures, set up
    after this one, have stopped theirs by then)."""
    yield
    assert [server.process.pid for server in running_servers()] == []
