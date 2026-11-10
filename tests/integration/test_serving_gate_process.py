"""The D11 serving gate, the process: the real `rinchen serve` as a subprocess of the test on the
loopback with a free port, stopped, started the way the compose file starts it, started without the
GPU stack, and refused. Real `ctranslate2` on the CPU only (`CUDA_VISIBLE_DEVICES` is the empty
string for the session, `device: cpu`, never `auto`); no image is built or run, and nothing but the
loopback is used. Every server runs in a session of its own and is stopped through its process
group only.

The data are the toy corpora of `tests/support/tiny_parallel.py`: templated sentences, so every
number below tests machinery and ranks nothing.

Gates:

    S3  SIGTERM with 24 requests of 8 sentences inside the service (the micro-batcher holds them
        for 1.5 s, so that the work outlasts the signal whatever the speed): all 24 are answered
        200 with the engine's translations, a connection made after the signal is refused, the
        process exits by the signal (-15) within the shutdown timeout, and its standard error
        holds `shutdown_begin` and `shutdown_drained` with `failed` 0 and no text of a request.
        `shutdown_drained` says `completed` 0 and `failed` 0 here, which is vacuous as a proof of
        the batcher's drain: uvicorn's own wait answered the requests before the lifespan ran.
        The 503 `draining` is not observable on the socket either (the listener is closed first,
        a client gets a refused connection); S3b of the other module sees both in process
        The signal is sent once `/metrics` shows every sentence queued or in a batch
    S4  the `command` of `docker/compose.serve.yaml`, run natively with the model, the host, the
        port and the path of the config replaced (`/app` of the image is the repository root here)
        and the compose `environment` applied: `/readyz` is 200 and a translation is the engine's.
        The image is NOT built and `docker compose up` is NOT run
    S5  `torch`, `triton`, `eole`, `transformers`, `sentence_transformers` and `pyonmttok` are
        unimportable (an isolated interpreter with the six modules set to `None`): `rinchen
        translate` of 5 lines, `rinchen serve --help` and a running `rinchen serve` still work
    S6  a refused model or setting is exit 2 before a port is bound: a D10 directory, an unknown
        key of `--config`, one flipped byte in each of the five files of an export; an undamaged
        copy is served (the control)
    S8  no file of the serving code or of this gate asks a device question

Measured on this machine (CPU, Eole 0.6.4, CTranslate2 4.8.4, CPython 3.14.10, 2026-11-10; the
machine was shared with other work, so a time moves by a third from one run to the next, a count
or a score never does):

    S3  192 sentences queued, none answered and no batch running when the signal was sent;
        6 connection attempts until one was refused; all 24 answered 1.58 s after the signal (the
        batcher's 1.5 s wait), the process gone 1.69 s after it, exit status -15 (the shutdown
        timeout is 20 s); stderr events: serve_start 1, compute_type_effective 1, request 33,
        shutdown_begin 1 (queued 0), shutdown_drained 1 (completed 0, failed 0)
    S4  ready after 0.66 s with inter_threads 2 and intra_threads 1, exit status -15
    S5  all six modules are installed here and all six refused; `rinchen translate` of 5 lines
        0.77 s, byte-equal to the unblocked output; `serve --help` byte-equal; `rinchen serve`
        ready after 0.66 s and translated (Task 10's concern that the CLI imports something of the
        GPU stack: it does not)
    S6  each refusal takes 0.8 s and exits 2 with nothing on standard output, no `serve_start`
        and the port free afterwards

The module takes about 12 s on top of the session fixtures.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.export.record import EXPORT_FILES
from rinchen.infer.engine_types import DecodeOptions, Row
from tests.support.d11_gate import forbidden_gpu_findings
from tests.support.distill_gate import RINCHEN, source_findings
from tests.support.serve_app import run
from tests.support.serving_gate import (
    BLOCKED_MODULES,
    COMPOSE_FILE,
    CONCURRENT_REQUESTS,
    LOOPBACK,
    SERVE_CONFIG,
    Reply,
    Server,
    batch_size_sum,
    blocked_command,
    compose_environment,
    compose_service,
    differing_translations,
    engine_translations,
    free_port,
    import_report,
    many_sentence_bodies,
    native_compose_command,
    port_is_free,
    refused_connection,
    run_refused,
    running_server,
    send,
    serve_command,
    server_environment,
    summarize,
    translate_body,
    wait_for_exit,
    wait_until_submitted,
)

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures(
        "d11_gpu_hidden", "d11_serving_log", "d11_no_server_is_left"
    ),
]

STAR: Final = "This is the star."
STAR_IN_KHALKHA: Final = "Энэ бол од."
# What a request that names a beam and no more gets on the tiny config.
DECODING: Final = DecodeOptions(beam_size=4, max_decoding_length=128)
DRAIN_REQUESTS: Final = 24
DRAIN_SENTENCES_PER_REQUEST: Final = 8
DRAIN_BEAM_SIZE: Final = 8
# `shutdown_timeout_seconds` of configs/serve/tiny.yaml: the process must be gone within it.
SHUTDOWN_TIMEOUT_SECONDS: Final = 20.0
NO_LINES: Final = 5
STOP_WAIT_SECONDS: Final = 60.0
# The code that serves and the tests that gate it: none of it asks a device question.
_SCANNED_FILES: Final = (
    *sorted((REPO_ROOT / "src" / "rinchen" / "serve").glob("*.py")),
    REPO_ROOT / "src" / "rinchen" / "cli" / "serve.py",
    REPO_ROOT / "src" / "rinchen" / "config" / "serve.py",
    *sorted((REPO_ROOT / "tests" / "support").glob("serving_gate*.py")),
    *sorted((REPO_ROOT / "tests" / "integration").glob("test_serving_gate*.py")),
)


def _post(server: Server, body: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{server.url}/v1/translate", json=body, timeout=30.0)


# ---------------------------------------------------------------------------------
# S3: SIGTERM with work in flight, on the real server.


def test_sigterm_with_24_requests_inside_the_service_answers_them_all_and_exits_by_the_signal(
    d11_tiny_export_dir: Path,
    d11_float32_engine: Any,
    d11_sentences: list[str],
    tmp_path: Path,
    record_property: Any,
) -> None:
    bodies = many_sentence_bodies(
        d11_sentences,
        DRAIN_REQUESTS,
        DRAIN_SENTENCES_PER_REQUEST,
        beam_size=DRAIN_BEAM_SIZE,
    )
    total = DRAIN_REQUESTS * DRAIN_SENTENCES_PER_REQUEST
    expected = engine_translations(
        d11_float32_engine, d11_sentences, beam_size=DRAIN_BEAM_SIZE
    )
    port = free_port()
    # The micro-batcher holds the 192 sentences for 1.5 s (a batch of 256 is never full and the
    # wait is long), so the requests are inside the service, waiting, when the signal comes,
    # whatever the speed of the machine: the slow engine of the spike, made with two flags.
    command = serve_command(
        d11_tiny_export_dir,
        port,
        "--inter-threads",
        "1",
        "--max-batch-sentences",
        "256",
        "--max-wait-ms",
        "1500",
    )

    async def scenario(server: Server) -> dict[str, Any]:
        limits = httpx.Limits(max_connections=CONCURRENT_REQUESTS)
        async with (
            httpx.AsyncClient(
                base_url=server.url, timeout=60.0, limits=limits
            ) as client,
            httpx.AsyncClient(base_url=server.url, timeout=10.0) as probe,
        ):
            baseline = await batch_size_sum(probe)
            tasks = [asyncio.create_task(send(client, body)) for body in bodies]
            queued, inflight = await wait_until_submitted(
                probe, total, baseline=baseline
            )
            answered_before = sum(task.done() for task in tasks)
            signalled = time.perf_counter()
            server.signal_group(signal.SIGTERM)
            attempts = await refused_connection(server)
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            answered_after = time.perf_counter() - signalled
            exited = await asyncio.to_thread(
                wait_for_exit, server.process, STOP_WAIT_SECONDS
            )
            return {
                "queued": queued,
                "inflight": inflight,
                "answered_before": answered_before,
                "attempts": attempts,
                "outcomes": outcomes,
                "answered_after": answered_after,
                "exited": exited,
                "exit_seconds": time.perf_counter() - signalled,
            }

    with running_server(command, tmp_path / "drain", port=port, name="drain") as server:
        outcome = run(scenario(server), 180.0)

    outcomes = outcome["outcomes"]
    failures = [each for each in outcomes if not isinstance(each, Reply)]
    assert failures == []
    replies: list[Reply] = outcomes
    # The signal came while the service held the work: the stop had something to drain.
    assert outcome["queued"] + outcome["inflight"] > 0
    assert outcome["answered_before"] < DRAIN_REQUESTS
    # All 24 are answered, with what the engine writes.
    assert {reply.status for reply in replies} == {200}
    assert differing_translations(replies, expected) == []
    assert summarize(replies).ok_sentences == total
    # A connection made after the signal is refused, and the process ends by the signal.
    assert outcome["attempts"] >= 1
    assert outcome["exited"]
    assert server.process.returncode == -signal.SIGTERM
    assert outcome["exit_seconds"] < SHUTDOWN_TIMEOUT_SECONDS
    (begin,) = server.named("shutdown_begin")
    (drained,) = server.named("shutdown_drained")
    assert drained["failed"] == 0
    assert begin["queued"] == 0
    assert len(server.named("serve_start")) == 1
    served = [
        event
        for event in server.named("request")
        if event["route"] == "translate" and event["status"] == 200
    ]
    assert len(served) == DRAIN_REQUESTS
    # No text of a request is in the log.
    stderr = server.stderr()
    assert [text for body in bodies for text in body["texts"] if text in stderr] == []

    record_property("drain_sentences_queued_at_the_signal", int(outcome["queued"]))
    record_property("drain_batches_in_flight_at_the_signal", int(outcome["inflight"]))
    record_property(
        "drain_requests_answered_before_the_signal", outcome["answered_before"]
    )
    record_property("drain_refused_connection_attempts", outcome["attempts"])
    record_property("drain_seconds_to_answer_all", round(outcome["answered_after"], 2))
    record_property("drain_seconds_to_exit", round(outcome["exit_seconds"], 2))
    record_property("drain_exit_status", server.process.returncode)
    record_property(
        "drain_stderr_events",
        json.dumps(
            dict(Counter(each["event"] for each in server.events())), sort_keys=True
        ),
    )
    record_property(
        "drain_event_completed_failed", f"{drained['completed']}/{drained['failed']}"
    )


# ---------------------------------------------------------------------------------
# S4: the command of the compose file, run natively.


def test_the_compose_command_run_natively_serves_the_tiny_export(
    d11_tiny_export_dir: Path,
    d11_float32_engine: Any,
    tmp_path: Path,
    record_property: Any,
) -> None:
    port = free_port()
    command = native_compose_command(d11_tiny_export_dir, port)
    environment = server_environment(compose_environment())
    compose_words = [str(word) for word in compose_service()["command"]]

    # Only the four values named in the module's text differ from the compose file's words.
    changed = {
        compose_words[index - 1]
        for index, (image, native) in enumerate(
            zip(compose_words, command[1:], strict=True)
        )
        if image != native
    }
    assert command[0] == RINCHEN
    assert command[1] == compose_words[0] == "serve"
    assert changed == {"--model", "--config", "--host", "--port"}
    assert command[command.index("--model") + 1] == str(d11_tiny_export_dir)
    assert Path(command[command.index("--config") + 1]) == SERVE_CONFIG
    assert command[command.index("--host") + 1] == LOOPBACK
    assert command[command.index("--port") + 1] == str(port)
    assert COMPOSE_FILE.is_file()
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    expected = d11_float32_engine.translate(
        [Row(STAR, "eng_Latn", "khk_Cyrl")], DECODING
    )[0]

    with running_server(
        command,
        tmp_path / "compose",
        port=port,
        environment=environment,
        name="compose",
    ) as server:
        ready = httpx.get(f"{server.url}/readyz", timeout=10.0)
        translated = _post(server, translate_body([STAR]))
        ready_seconds = server.ready_seconds

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["device"] == "cpu"
    assert translated.status_code == 200
    assert translated.json()["translations"] == [expected.text] == [STAR_IN_KHALKHA]
    (start,) = server.named("serve_start")
    assert start["inter_threads"] == int(environment["RINCHEN_SERVE_INTER_THREADS"])
    assert start["intra_threads"] == int(environment["RINCHEN_SERVE_INTRA_THREADS"])
    assert start["device"] == "cpu"
    assert server.process.returncode == -signal.SIGTERM
    record_property("compose_command_ready_seconds", round(ready_seconds, 2))
    record_property(
        "compose_command_threads", f"{start['inter_threads']}x{start['intra_threads']}"
    )


# ---------------------------------------------------------------------------------
# S5: no GPU stack.


def test_the_gpu_stack_is_installed_here_and_the_wrapper_makes_it_unimportable(
    record_property: Any,
) -> None:
    installed = import_report(blocked=False)
    refused = import_report(blocked=True)

    # The block is not vacuous: the stack the slim image leaves out is installed in this
    # environment, and the wrapper refuses an import of every name.
    assert installed["torch"] is True
    assert installed["eole"] is True
    assert refused == dict.fromkeys(BLOCKED_MODULES, "refused")
    record_property(
        "installed_here", ",".join(name for name, found in installed.items() if found)
    )


def test_rinchen_translate_of_five_lines_works_without_the_gpu_stack(
    d11_tiny_export_dir: Path,
    d11_sentences: list[str],
    record_property: Any,
) -> None:
    lines = d11_sentences[:NO_LINES]
    arguments = (
        "translate",
        "--model",
        str(d11_tiny_export_dir),
        "--src",
        "eng_Latn",
        "--tgt",
        "khk_Cyrl",
    )
    stdin = "".join(f"{line}\n" for line in lines).encode("utf-8")

    def translate(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            timeout=300,
            check=False,
            env=server_environment(),
        )

    started = time.perf_counter()
    blocked = translate(blocked_command(*arguments))
    blocked_seconds = time.perf_counter() - started
    normal = translate([RINCHEN, *arguments])

    assert blocked.returncode == 0, blocked.stderr.decode("utf-8", "replace")
    assert normal.returncode == 0
    translated = blocked.stdout.decode("utf-8").splitlines()
    assert len(translated) == NO_LINES
    assert all(translated)
    assert blocked.stdout == normal.stdout
    assert f"ok {NO_LINES}" in blocked.stderr.decode("utf-8")
    assert "ImportError" not in blocked.stderr.decode("utf-8")
    record_property("blocked_translate_seconds", round(blocked_seconds, 2))
    record_property("blocked_translate_lines", len(translated))


def test_rinchen_serve_help_works_without_the_gpu_stack() -> None:
    def serve_help(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            command,
            capture_output=True,
            timeout=300,
            check=False,
            env=server_environment(),
        )

    blocked = serve_help(blocked_command("serve", "--help"))
    normal = serve_help([RINCHEN, "serve", "--help"])

    assert blocked.returncode == 0, blocked.stderr.decode("utf-8", "replace")
    assert normal.returncode == 0
    text = blocked.stdout.decode("utf-8")
    for option in ("--model", "--config", "--host", "--port", "--device"):
        assert option in text
    assert blocked.stdout == normal.stdout


def test_a_rinchen_serve_without_the_gpu_stack_answers_a_translation(
    d11_tiny_export_dir: Path,
    tmp_path: Path,
    record_property: Any,
) -> None:
    port = free_port()
    command = blocked_command(
        *[str(word) for word in serve_command(d11_tiny_export_dir, port)[1:]]
    )

    with running_server(
        command, tmp_path / "blocked", port=port, name="blocked"
    ) as server:
        translated = _post(server, translate_body([STAR]))
        ready_seconds = server.ready_seconds

    assert translated.status_code == 200
    assert translated.json()["translations"] == [STAR_IN_KHALKHA]
    assert "ImportError" not in server.stderr()
    assert server.process.returncode == -signal.SIGTERM
    record_property("blocked_serve_ready_seconds", round(ready_seconds, 2))


# ---------------------------------------------------------------------------------
# S6: a refused model or setting is exit 2 before a port is bound.


def _assert_refused_before_binding(refusal: Any, port: int, *, mentions: str) -> None:
    assert refusal.returncode == 2, refusal.stderr
    assert refusal.stdout == ""
    assert mentions in refusal.stderr
    # It never got as far as serving: no start event, no line of uvicorn's, and the port is free.
    assert "serve_start" not in refusal.stderr
    assert "Uvicorn running" not in refusal.stderr
    assert port_is_free(port)


def test_a_d10_export_directory_is_refused_with_exit_2_and_the_remedy(
    d11_tiny_export_dir: Path, tmp_path: Path
) -> None:
    old = tmp_path / "d10_export"
    shutil.copytree(d11_tiny_export_dir, old)
    (old / "rinchen.json").rename(old / "export.json")
    port = free_port()

    refusal = run_refused(
        serve_command(old, port), tmp_path / "refused_d10", name="d10"
    )

    _assert_refused_before_binding(refusal, port, mentions="rinchen export convert")


def test_a_config_with_an_unknown_key_is_refused_with_exit_2_naming_the_key(
    d11_tiny_export_dir: Path, tmp_path: Path
) -> None:
    config = tmp_path / "unknown_key.yaml"
    config.write_text("max_texts: 4\nnot_a_setting: 1\n", encoding="utf-8")
    port = free_port()

    refusal = run_refused(
        serve_command(d11_tiny_export_dir, port, config=config),
        tmp_path / "refused_key",
        name="key",
    )

    _assert_refused_before_binding(refusal, port, mentions="not_a_setting")


@pytest.mark.parametrize("damaged", EXPORT_FILES)
def test_an_export_with_one_flipped_byte_in_a_file_is_refused_with_exit_2(
    damaged: str, d11_tiny_export_dir: Path, tmp_path: Path
) -> None:
    export = tmp_path / "flipped"
    shutil.copytree(d11_tiny_export_dir, export)
    path = export / damaged
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0x01
    path.write_bytes(bytes(content))
    port = free_port()

    refusal = run_refused(
        serve_command(export, port), tmp_path / "refused_byte", name="byte"
    )

    _assert_refused_before_binding(refusal, port, mentions=damaged)


def test_an_undamaged_copy_of_the_export_is_served(
    d11_tiny_export_dir: Path, tmp_path: Path
) -> None:
    # The control of the refusals above: the same copy, with no byte changed, is a model.
    export = tmp_path / "intact"
    shutil.copytree(d11_tiny_export_dir, export)
    port = free_port()

    with running_server(
        serve_command(export, port), tmp_path / "served_copy", port=port, name="copy"
    ) as server:
        translated = _post(server, translate_body([STAR]))

    assert translated.status_code == 200
    assert translated.json()["translations"] == [STAR_IN_KHALKHA]


# ---------------------------------------------------------------------------------
# No GPU.


def test_no_gpu_question_is_asked_by_the_serving_code_or_its_gate() -> None:
    assert len(_SCANNED_FILES) >= 12
    for path in _SCANNED_FILES:
        assert forbidden_gpu_findings(path) == [], path
        assert set(source_findings(path).translator_devices) <= {"'cpu'"}, path
    # Every server of the gate is started with every device hidden, on the CPU.
    assert server_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert "device: cpu" in SERVE_CONFIG.read_text("utf-8")
