"""`run_synth` and its loopback listener (D15, spec P17, brief G4.4).

This module does NOT use the `no_network` guard: it is the one place where a test of the pipeline
connects, and only to the ports of the stage's own listener on 127.0.0.1 (and, once, to 127.0.0.2 to
show that the listener is not bound to more than the loopback address). The fake runner plays the child
here the way the real `synth run` does, one chat request to the server of the config for each text; the
stage's facts, its files and the end of the listener are checked on that traffic. Everything else about
the stage is in the two guarded modules.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import pytest
import yaml

from rinchen.corpus.readers import scan_corpus
from tests.support import tiny_run
from tests.support.tiny_run import (
    at,
    ready_for,
    run_one,
    stage_function,
)
from tests.support.tiny_stage_inputs import shipped_fixtures

# The fixtures of `tests/support/tiny_run.py`, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
out_dir = tiny_run.out_dir
LOOPBACK: Final = "127.0.0.1"


def port_of(out: Path) -> int:
    """The port of the listener, as the stage wrote it into the config of the command."""
    config = yaml.safe_load(at(out, "synth", "synth.yaml").read_text("utf-8"))
    parts = urlsplit(config["generators"]["tiny_oracle"]["base_url"])
    assert parts.hostname == LOOPBACK
    assert parts.port is not None
    return parts.port


def refuses(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError:
        return True
    return False


def test_the_child_reaches_the_listener_and_the_corpus_holds_what_the_oracle_said(
    out_dir: Path,
) -> None:
    result, runner = run_one(out_dir, "synth", contact_server=True)

    mono = shipped_fixtures().mono
    assert result.record.state == "ok"
    assert runner.requests_made == len(mono) == 64
    assert result.facts["chat_requests"] == 64
    assert result.facts["rows"] == 64
    corpus = scan_corpus(at(out_dir, "synth", "run", "corpus.parquet")).collect()
    assert corpus["src"].to_list() == [row.eng for row in mono]
    assert corpus["tgt"].to_list() == [row.khk for row in mono]


def test_the_listener_is_gone_when_the_stage_has_ended(out_dir: Path) -> None:
    result, _ = run_one(out_dir, "synth", contact_server=True)

    assert result.record.state == "ok"
    assert refuses(LOOPBACK, port_of(out_dir))


@pytest.mark.parametrize(
    "options",
    [
        {"fail": {"synth run": 4}},
        {"time_out": {"synth run"}},
        {"not_started": {"synth run"}},
    ],
    ids=["exit code", "timeout", "not started"],
)
def test_the_listener_is_gone_when_the_command_failed(
    out_dir: Path, options: dict[str, Any]
) -> None:
    result, _ = run_one(out_dir, "synth", **options)

    assert result.record.state == "fail"
    assert refuses(LOOPBACK, port_of(out_dir))


def test_the_listener_is_gone_when_the_runner_raises(out_dir: Path) -> None:
    ctx, _ = ready_for(out_dir, "synth", raises={"synth run": RuntimeError("boom")})

    with pytest.raises(RuntimeError, match="boom"):
        stage_function("synth")(ctx)

    assert refuses(LOOPBACK, port_of(out_dir))


def test_the_listener_is_up_while_the_command_runs_and_only_on_the_loopback(
    out_dir: Path,
) -> None:
    seen: dict[str, bool] = {}

    def probe(call: Any) -> None:
        port = port_of(out_dir)
        seen["loopback"] = not refuses(LOOPBACK, port)
        seen["other address"] = not refuses("127.0.0.2", port)

    run_one(out_dir, "synth", after={"synth run": probe})

    assert seen == {"loopback": True, "other address": False}
