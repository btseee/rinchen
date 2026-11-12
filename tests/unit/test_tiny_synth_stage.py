"""`run_synth` and its loopback listener (D15, spec P17, brief G4.4).

This module does NOT use the `no_network` guard: it is the one place where a test of the pipeline
connects, and only to the ports of the stage's own listener on 127.0.0.1 (and, once, to 127.0.0.2 to
show that the listener is not bound to more than the loopback address). The fake runner plays the child
here the way the real `synth run` does, one chat request to the server of the config for each text; the
stage's facts, its files and the end of the listener are checked on that traffic. Everything else about
the stage is in the two guarded modules.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import polars as pl
import pytest
import yaml

from rinchen.corpus.readers import scan_corpus
from rinchen.synth.prompts import render_messages
from rinchen.tiny import commands, model_stages
from rinchen.tiny.fakes import FakeChatServer
from rinchen.tiny.states import fail_check
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


# ---- the output of the run: a run that made no back-translated row is not ok ---------------------------


def leaving(*, selected: int, sources: int, rows: int) -> Callable[[Any], None]:
    """An `after` hook of `synth run`: the run directory is left with these counts and a corpus of
    `rows` rows, as a run would that the server could not answer."""

    def hook(call: Any) -> None:
        run = call.path("--out-dir")
        document = json.loads((run / "counts.json").read_text("utf-8"))
        document["counts"]["selected"] = selected
        document["counts"]["sources"] = sources
        (run / "counts.json").write_text(json.dumps(document), encoding="utf-8")
        corpus = pl.read_parquet(run / "corpus.parquet")
        pl.concat([corpus, corpus]).head(rows).write_parquet(run / "corpus.parquet")

    return hook


@pytest.mark.parametrize(
    ("selected", "sources", "rows"),
    [
        (0, 64, 0),
        (63, 64, 63),
        (64, 64, 63),
        (64, 64, 65),
        (0, 0, 0),
        # Everything selected and one row for each, but of fewer sources than the 64 rows the stage fed:
        # a run that read only part of the monolingual file.
        (10, 10, 10),
        (63, 63, 63),
        (65, 65, 65),
    ],
    ids=[
        "the server answered nothing",
        "one source without a row",
        "a row fewer than selected",
        "a row more than selected",
        "no source and no row",
        "the run read ten of the 64 sources",
        "the run read 63 of the 64 sources",
        "the run read a source more than it was fed",
    ],
)
def test_a_run_that_did_not_select_every_source_or_made_another_number_of_rows_fails_its_check(
    out_dir: Path, selected: int, sources: int, rows: int
) -> None:
    result, _ = run_one(
        out_dir,
        "synth",
        after={"synth run": leaving(selected=selected, sources=sources, rows=rows)},
    )

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("synth_output")
    assert result.record.detail == {}
    assert result.facts["rows"] == rows
    assert result.facts["counts"]["selected"] == selected
    assert result.facts["counts"]["sources"] == sources


def test_the_sources_are_compared_with_the_rows_the_stage_fed_whatever_their_number(
    out_dir: Path,
) -> None:
    # A stage fed ten rows is complete with ten sources, selected and made into one row each: the
    # number it compares with is the one of the fixture it wrote, not a figure of the shipped one.
    ctx, _ = ready_for(
        out_dir, "synth", after={"synth run": leaving(selected=10, sources=10, rows=10)}
    )
    fixtures = shipped_fixtures()
    fed = dataclasses.replace(fixtures, mono=fixtures.mono[:10])

    result = stage_function("synth")(dataclasses.replace(ctx, fixtures=fed))

    assert result.record.state == "ok"
    assert result.facts["counts"]["sources"] == result.facts["rows"] == 10


def test_the_check_is_the_closed_identifier_synth_output() -> None:
    assert model_stages.SYNTH_CHECK == fail_check("synth_output")
    assert model_stages.SYNTH_CHECK == "a check of the stage failed: synth_output"


def test_a_run_that_selected_every_source_into_one_row_each_is_still_ok(
    out_dir: Path,
) -> None:
    result, _ = run_one(
        out_dir,
        "synth",
        after={"synth run": leaving(selected=64, sources=64, rows=64)},
    )

    assert result.record.state == "ok"
    assert result.facts["rows"] == result.facts["counts"]["selected"] == 64
    assert result.facts["counts"]["sources"] == 64


# ---- the server and the layout of the prompt of the synthetic-data client ------------------------------


def asked(server: FakeChatServer, content: str) -> str:
    """The answer of the server to a chat request whose first message is `content`."""
    parts = urlsplit(server.base_url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request(
            "POST",
            f"{parts.path}/chat/completions",
            json.dumps({"messages": [{"role": "user", "content": content}]}),
            {"Content-Type": "application/json"},
        )
        reply = json.loads(connection.getresponse().read())
    finally:
        connection.close()
    answer: str = reply["choices"][0]["message"]["content"]
    return answer


def test_the_server_answers_the_prompt_of_the_real_client_with_the_english_line_of_the_oracle() -> (
    None
):
    mono = shipped_fixtures().mono
    oracle = {row.khk: row.eng for row in mono}

    with FakeChatServer(oracle) as server:
        answers = {
            row.khk: asked(
                server,
                render_messages(
                    commands.MONO_LANGUAGE, commands.OTHER_LANGUAGE, row.khk
                )[0]["content"],
            )
            for row in mono
        }

    assert len(mono) == 64
    assert answers == oracle
    assert "" not in answers.values()
