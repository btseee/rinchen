"""The helpers of the D11 serving gate (`tests/integration/test_serving_gate*.py`): a real
`rinchen serve` as a subprocess of the test, a client that drives it, and the readings the gate
makes of its answers, its `/metrics` and its standard error.

Every server the gate starts runs on `127.0.0.1` with a free port, in a session of its own
(`start_new_session=True`) and is stopped through that process group only, never by name:
`stop_server` sends the group SIGTERM and, only if it outlives its deadline, SIGKILL.
`running_server` stops its server even when the test fails, and a server that is still live when
the test process ends (or is told to end by SIGTERM) is stopped by an exit hook, so a hard stop of
the run does not leave a server behind. Standard output and standard error of a server go to files
beside it, so a chatty server cannot fill a pipe, and `Server.events` reads the JSON lines of the
second. Nothing here waits with a fixed pause: the readiness wait and every other wait are loops
with a deadline.

Nothing here starts a GPU: `server_environment` sets `CUDA_VISIBLE_DEVICES` to the empty string,
and the services run on `device: cpu` (`configs/serve/tiny.yaml`). The compose file of
`docker/compose.serve.yaml` is only read; the image is neither built nor run.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final

import httpx
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.serve import ServeConfig, load_serve_config
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import (
    DecodeOptions,
    DocumentResult,
    ModelInfo,
    Row,
    ScriptName,
    Translation,
)
from tests.support.distill_gate import RINCHEN
from tests.support.distill_gate_stages import Workspace
from tests.support.serve_app import sample

LOOPBACK: Final = "127.0.0.1"
SERVE_CONFIG: Final = REPO_ROOT / "configs" / "serve" / "tiny.yaml"
COMPOSE_FILE: Final = REPO_ROOT / "docker" / "compose.serve.yaml"
# Where the repository is in the image of `docker/Dockerfile.cpu` (its WORKDIR): the compose
# command names the config file there.
CONTAINER_APP_DIRECTORY: Final = "/app"
SOURCE_LANGUAGE: Final = "eng_Latn"
TARGET_LANGUAGE: Final = "khk_Cyrl"
# A server is ready in about a second; the deadlines are room for a loaded machine, not a
# measurement, and none of them is a gate.
READY_DEADLINE_SECONDS: Final = 120.0
STOP_DEADLINE_SECONDS: Final = 60.0
REQUEST_TIMEOUT_SECONDS: Final = 60.0
POLL_SECONDS: Final = 0.02
BLOCKED_MODULES: Final = (
    "torch",
    "triton",
    "eole",
    "transformers",
    "sentence_transformers",
    "pyonmttok",
)


# ---- the process ------------------------------------------------------------------------


def free_port() -> int:
    """A TCP port of the loopback that nothing listens on now (the operating system's choice)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        port: int = probe.getsockname()[1]
    return port


def port_is_free(port: int) -> bool:
    """Whether `port` of the loopback can be bound now (a server that bound it would hold it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((LOOPBACK, port))
        except OSError:
            return False
    return True


def server_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a server: the test's own without any `RINCHEN_*` variable (so the
    outer shell cannot change a setting), every device hidden, then `extra`."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("RINCHEN_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.update(extra or {})
    return environment


@dataclass
class Server:
    """A `rinchen serve` the gate started: its process, its port and the two files its output
    goes to."""

    process: subprocess.Popen[bytes]
    port: int
    stdout_path: Path
    stderr_path: Path
    ready_seconds: float = 0.0

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def stderr(self) -> str:
        return self.stderr_path.read_text("utf-8", errors="replace")

    def stdout(self) -> str:
        return self.stdout_path.read_text("utf-8", errors="replace")

    def events(self) -> list[dict[str, Any]]:
        """The JSON lines of standard error, in order (uvicorn's few plain lines are not)."""
        return events_of(self.stderr())

    def named(self, event: str) -> list[dict[str, Any]]:
        return [each for each in self.events() if each.get("event") == event]

    def signal_group(self, signum: signal.Signals) -> None:
        """Send `signum` to the server's process group, which it leads."""
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signum)


def events_of(text: str) -> list[dict[str, Any]]:
    """The JSON objects among the lines of `text`: the service's events."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


_LIVE: list[Server] = []
_PREVIOUS_SIGTERM: list[Any] = []


def running_servers() -> list[Server]:
    """The servers the gate started that are still running."""
    return [server for server in _LIVE if server.process.poll() is None]


def _end_the_process_on_sigterm(signum: int, frame: FrameType | None) -> None:
    # Raising ends the main thread through its `finally` blocks and exit hooks, which stop the
    # servers; the default action would end the test process with them still running.
    raise SystemExit(128 + signal.SIGTERM)


def _stop_every_live_server() -> None:
    # A copy: stopping a server takes it off the list.
    for server in tuple(_LIVE):
        with suppress(Exception):
            stop_server(server, timeout=10.0)


atexit.register(_stop_every_live_server)


def start_server(
    command: Sequence[str | Path],
    workdir: Path,
    *,
    port: int,
    environment: Mapping[str, str] | None = None,
    name: str = "server",
) -> Server:
    """Start `command` (a whole argument list, `rinchen serve ...` or a wrapper of it) in a
    session of its own, working in `workdir`, its output in `workdir/<name>.stdout|stderr`."""
    workdir.mkdir(parents=True, exist_ok=True)
    stdout_path = workdir / f"{name}.stdout"
    stderr_path = workdir / f"{name}.stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [str(word) for word in command],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=workdir,
            env=dict(environment) if environment is not None else server_environment(),
            start_new_session=True,
        )
    server = Server(process, port, stdout_path, stderr_path)
    if not _LIVE and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        _PREVIOUS_SIGTERM.append(
            signal.signal(signal.SIGTERM, _end_the_process_on_sigterm)
        )
    _LIVE.append(server)
    return server


def stop_server(server: Server, *, timeout: float = STOP_DEADLINE_SECONDS) -> int:
    """Stop the server through its process group: SIGTERM, and SIGKILL only when it is still
    running after `timeout` seconds. Returns its exit status (negative: ended by that signal)."""
    process = server.process
    if process.poll() is None:
        server.signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            server.signal_group(signal.SIGKILL)
            process.wait(timeout=timeout)
    if server in _LIVE:
        _LIVE.remove(server)
    if not _LIVE and _PREVIOUS_SIGTERM:
        signal.signal(signal.SIGTERM, _PREVIOUS_SIGTERM.pop())
    assert process.returncode is not None
    return process.returncode


def wait_until_ready(
    server: Server, *, deadline_seconds: float = READY_DEADLINE_SECONDS
) -> float:
    """Poll `/readyz` until it answers 200 and return the seconds that took; fail, with the
    tail of the server's standard error, when the server ends first or the deadline passes."""
    started = time.perf_counter()
    with httpx.Client(base_url=server.url, timeout=2.0) as client:
        while True:
            returncode = server.process.poll()
            if returncode is not None:
                raise AssertionError(
                    f"the server ended with status {returncode} before it was ready:\n"
                    f"{_tail(server.stderr())}"
                )
            try:
                if client.get("/readyz").status_code == 200:
                    server.ready_seconds = time.perf_counter() - started
                    return server.ready_seconds
            except httpx.TransportError:
                pass
            if time.perf_counter() - started > deadline_seconds:
                raise AssertionError(
                    f"the server was not ready after {deadline_seconds} s:\n"
                    f"{_tail(server.stderr())}"
                )
            wait_for_exit(server.process, POLL_SECONDS)


def wait_for_exit(process: subprocess.Popen[bytes], seconds: float) -> bool:
    """Wait up to `seconds` for `process` to end; whether it has."""
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


@contextmanager
def running_server(
    command: Sequence[str | Path],
    workdir: Path,
    *,
    port: int,
    environment: Mapping[str, str] | None = None,
    name: str = "server",
) -> Iterator[Server]:
    """A started, ready server, stopped through its process group when the block ends, whatever
    the block did."""
    server = start_server(
        command, workdir, port=port, environment=environment, name=name
    )
    try:
        wait_until_ready(server)
        yield server
    finally:
        stop_server(server)


def serve_command(
    model: Path, port: int, *extra: str | Path, config: Path | None = SERVE_CONFIG
) -> list[str | Path]:
    """`rinchen serve --model <model> [--config <config>] --port <port> <extra>` on the
    loopback (the default host), through the console script of the test's environment."""
    command: list[str | Path] = [RINCHEN, "serve", "--model", model]
    if config is not None:
        command += ["--config", config]
    return [*command, "--host", LOOPBACK, "--port", str(port), *extra]


# ---- the compose command, run natively ---------------------------------------------------


def compose_service(path: Path = COMPOSE_FILE) -> Mapping[str, Any]:
    """The one service of the compose file, as parsed YAML."""
    services = yaml.safe_load(path.read_text("utf-8"))["services"]
    assert len(services) == 1, sorted(services)
    service: Mapping[str, Any] = next(iter(services.values()))
    return service


def compose_environment(path: Path = COMPOSE_FILE) -> dict[str, str]:
    """The `environment` of the compose service, every value a string."""
    return {
        str(name): str(value)
        for name, value in compose_service(path)["environment"].items()
    }


def native_compose_command(
    model: Path, port: int, path: Path = COMPOSE_FILE
) -> list[str]:
    """The `command` of the compose service as it runs here, without a container: `rinchen`
    of the test's environment, then the compose words with exactly four values replaced: the
    model directory, the host (`127.0.0.1`), the port, and the repository root for the `/app`
    of the image in the path of the config file (the image copies `configs/` to `/app`, so the
    same file is the one under the repository root)."""
    words = [str(word) for word in compose_service(path)["command"]]
    replacements = {
        "--model": str(model),
        "--host": LOOPBACK,
        "--port": str(port),
    }
    replaced: set[str] = set()
    for index, word in enumerate(words):
        if word in replacements:
            words[index + 1] = replacements[word]
            replaced.add(word)
        elif word == "--config":
            image_path = words[index + 1]
            assert image_path.startswith(f"{CONTAINER_APP_DIRECTORY}/"), image_path
            words[index + 1] = str(
                REPO_ROOT / image_path.removeprefix(f"{CONTAINER_APP_DIRECTORY}/")
            )
            replaced.add(word)
    assert replaced == {"--model", "--host", "--port", "--config"}, replaced
    return [RINCHEN, *words]


# ---- no torch ----------------------------------------------------------------------------

_BLOCK_THE_GPU_STACK: Final = (
    f"import sys\nfor name in {BLOCKED_MODULES!r}:\n    sys.modules[name] = None\n"
)
_RUN_THE_CONSOLE_SCRIPT: Final = (
    "import importlib.metadata\n"
    "app = importlib.metadata.entry_points(group='console_scripts')['rinchen'].load()\n"
    "sys.exit(app(prog_name='rinchen'))\n"
)
_TRY_THE_BLOCKED_IMPORTS: Final = (
    "import importlib, json\n"
    "report = {}\n"
    f"for name in {BLOCKED_MODULES!r}:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except ImportError:\n"
    "        report[name] = 'refused'\n"
    "    else:\n"
    "        report[name] = 'imported'\n"
    "print(json.dumps(report))\n"
)
_FIND_THE_INSTALLED: Final = (
    "import importlib.util, json\n"
    f"print(json.dumps({{name: importlib.util.find_spec(name) is not None for name in {BLOCKED_MODULES!r}}}))\n"
)


def blocked_command(*arguments: str | Path) -> list[str]:
    """`rinchen <arguments>` in an isolated interpreter (`-I`: no environment variable, no
    user site, no `sitecustomize`) of the test's environment where the GPU stack cannot be
    imported: the modules of `BLOCKED_MODULES` are `None` in `sys.modules`, so an import of any
    raises `ImportError`, and the console script's target (`rinchen.cli.main:app`, read from the
    package's entry points) runs afterwards."""
    return [
        sys.executable,
        "-I",
        "-c",
        _BLOCK_THE_GPU_STACK + _RUN_THE_CONSOLE_SCRIPT,
        *map(str, arguments),
    ]


def import_report(*, blocked: bool) -> dict[str, Any]:
    """What an isolated interpreter of the test's environment says of `BLOCKED_MODULES`: with
    `blocked`, whether an import of each is refused after the block; without it, whether each is
    installed (found without being imported)."""
    code = (
        _BLOCK_THE_GPU_STACK + _TRY_THE_BLOCKED_IMPORTS
        if blocked
        else _FIND_THE_INSTALLED
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        timeout=120,
        check=True,
    )
    report: dict[str, Any] = json.loads(completed.stdout)
    return report


# ---- the client --------------------------------------------------------------------------


def translate_body(texts: Sequence[str], **fields: Any) -> dict[str, Any]:
    """A `POST /v1/translate` body: the texts from English into Khalkha Cyrillic, with `fields`
    replacing or adding members."""
    return {
        "texts": list(texts),
        "src": SOURCE_LANGUAGE,
        "tgt": TARGET_LANGUAGE,
        **fields,
    }


def single_sentence_bodies(
    sentences: Sequence[str], count: int
) -> list[dict[str, Any]]:
    """`count` requests of one sentence each, walking through `sentences`."""
    return [
        translate_body([sentences[index % len(sentences)]]) for index in range(count)
    ]


def many_sentence_bodies(
    sentences: Sequence[str], count: int, per_request: int, **fields: Any
) -> list[dict[str, Any]]:
    """`count` requests of `per_request` consecutive sentences each, walking through
    `sentences`."""
    return [
        translate_body(
            [
                sentences[(index * per_request + offset) % len(sentences)]
                for offset in range(per_request)
            ],
            **fields,
        )
        for index in range(count)
    ]


@dataclass(frozen=True)
class Reply:
    """One answered request: the status, the seconds from sending to the whole answer, the body
    as JSON and the texts that were asked for."""

    status: int
    seconds: float
    body: dict[str, Any]
    texts: tuple[str, ...]


async def send(client: httpx.AsyncClient, body: dict[str, Any]) -> Reply:
    started = time.perf_counter()
    response = await client.post("/v1/translate", json=body)
    seconds = time.perf_counter() - started
    return Reply(response.status_code, seconds, response.json(), tuple(body["texts"]))


async def burst(
    client: httpx.AsyncClient, bodies: Sequence[dict[str, Any]]
) -> list[Reply]:
    """Every body sent at once through `client`; the replies in the order of `bodies`."""
    return list(await asyncio.gather(*(send(client, body) for body in bodies)))


def percentile(seconds: Sequence[float], share: float) -> float:
    """The nearest-rank percentile: the smallest value that at least `share` of them do not
    exceed (for 64 values and 0.95, the 61st smallest)."""
    ordered = sorted(seconds)
    return ordered[math.ceil(share * len(ordered)) - 1]


@dataclass(frozen=True)
class Summary:
    """What a burst of replies shows: statuses, latency, and the sentences answered."""

    requests: int
    statuses: dict[int, int]
    p50: float
    p95: float
    slowest: float
    sentences: int
    ok_sentences: int


def summarize(replies: Sequence[Reply]) -> Summary:
    seconds = [reply.seconds for reply in replies]
    answered = [reply for reply in replies if reply.status == 200]
    return Summary(
        requests=len(replies),
        statuses=dict(Counter(reply.status for reply in replies)),
        p50=percentile(seconds, 0.5),
        p95=percentile(seconds, 0.95),
        slowest=max(seconds),
        sentences=sum(len(reply.body["translations"]) for reply in answered),
        ok_sentences=sum(reply.body["statuses"].count("ok") for reply in answered),
    )


def differing_translations(
    replies: Sequence[Reply], expected: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """The (text, answered, expected) of every sentence of a 200 reply whose translation is not
    the one in `expected`: empty when the service wrote what the engine writes."""
    differing: list[tuple[str, str, str]] = []
    for reply in replies:
        if reply.status != 200:
            continue
        for text, answered in zip(reply.texts, reply.body["translations"], strict=True):
            if answered != expected[text]:
                differing.append((text, answered, expected[text]))
    return differing


def engine_translations(
    engine: Engine, sentences: Sequence[str], *, beam_size: int = 4
) -> dict[str, str]:
    """What the engine itself writes for each of `sentences` (into `TARGET_LANGUAGE`, with the
    decode options `configs/serve/tiny.yaml` and a request that names no more give)."""
    options = DecodeOptions(beam_size=beam_size, max_decoding_length=128)
    rows = [Row(text, SOURCE_LANGUAGE, TARGET_LANGUAGE) for text in sentences]
    translations = engine.translate(rows, options)
    assert all(each.status == "ok" for each in translations)
    return {text: each.text for text, each in zip(sentences, translations, strict=True)}


# ---- /metrics ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """The three counters the gate reads of `/metrics`."""

    translate_200: float
    ok_sentences: float
    engine_sources: float


def reading_of(text: str) -> Reading:
    return Reading(
        translate_200=sample(
            text, "rinchen_requests_total", route="translate", status="200"
        )
        or 0.0,
        ok_sentences=sample(text, "rinchen_translated_sentences_total", status="ok")
        or 0.0,
        engine_sources=sample(text, "rinchen_engine_events_total", event="sources")
        or 0.0,
    )


async def read_metrics(client: httpx.AsyncClient) -> Reading:
    response = await client.get("/metrics")
    assert response.status_code == 200
    return reading_of(response.text)


# ---- the real engine, held ---------------------------------------------------------------

# A held call that nobody releases is a bug of the test: it ends after this many seconds.
RELEASE_TIMEOUT_SECONDS: Final = 60.0


class HeldEngine:
    """The real engine behind two gates, for the in-process tests of the service: `warmup` and
    `translate` wait until their event is set (both are set at the start), then run the real
    call. `warmup_entered` is set when the warm-up has begun, `rows_entered` counts the rows
    given to `translate` (before its gate), and `closed` says whether the service closed it."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.warmup_gate = threading.Event()
        self.translate_gate = threading.Event()
        self.warmup_gate.set()
        self.translate_gate.set()
        self.warmup_entered = threading.Event()
        self._lock = threading.Lock()
        self._rows_entered = 0
        self.closed = False

    @property
    def info(self) -> ModelInfo:
        return self._engine.info

    @property
    def rows_entered(self) -> int:
        with self._lock:
            return self._rows_entered

    def warmup(self) -> None:
        self.warmup_entered.set()
        if not self.warmup_gate.wait(RELEASE_TIMEOUT_SECONDS):
            raise AssertionError("a held warm-up was never released")
        self._engine.warmup()

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        with self._lock:
            self._rows_entered += len(rows)
        if not self.translate_gate.wait(RELEASE_TIMEOUT_SECONDS):
            raise AssertionError("a held call was never released")
        return self._engine.translate(rows, options)

    def translate_documents(
        self,
        documents: Sequence[str],
        src: str | None,
        tgt: str,
        options: DecodeOptions | None = None,
        *,
        script: ScriptName | None = None,
    ) -> list[DocumentResult]:
        return self._engine.translate_documents(
            documents, src, tgt, options, script=script
        )

    def counters(self) -> dict[str, int]:
        return self._engine.counters()

    def close(self) -> None:
        self.closed = True
        self._engine.close()


# ---- what the tests are given -------------------------------------------------------------


def tiny_config(**changes: Any) -> ServeConfig:
    """`configs/serve/tiny.yaml` as the command loads it (no environment layer), with
    `changes` laid over it as the flags of the command would."""
    return load_serve_config(SERVE_CONFIG, env={}, overrides=changes)


def dev_sentences(workspace: Workspace) -> list[str]:
    """The 100 English dev sentences of the toy's English to Khalkha direction: the texts
    the gate sends."""
    return list(workspace.inputs.dev["eng_Latn-khk_Cyrl"].src)


# ---- the load of the gate -----------------------------------------------------------------

WARM_UP_REQUESTS: Final = 8
CONCURRENT_REQUESTS: Final = 64
SENTENCES_PER_LARGE_REQUEST: Final = 8


@dataclass(frozen=True)
class Load:
    """The gate's load on a service and the two readings of `/metrics` around it: eight warm-up
    requests, 64 concurrent requests of one sentence and 64 of eight."""

    before: Reading
    after: Reading
    warm: list[Reply]
    singles: list[Reply]
    large: list[Reply]

    @property
    def replies(self) -> list[Reply]:
        return [*self.warm, *self.singles, *self.large]


async def run_load(client: httpx.AsyncClient, sentences: Sequence[str]) -> Load:
    """The gate's load through `client`, which reaches the service in process or over a socket.
    The metrics are read before the warm-up, so the warm-up requests are counted in the change."""
    before = await read_metrics(client)
    warm = await burst(client, single_sentence_bodies(sentences, WARM_UP_REQUESTS))
    singles = await burst(
        client, single_sentence_bodies(sentences, CONCURRENT_REQUESTS)
    )
    large = await burst(
        client,
        many_sentence_bodies(
            sentences, CONCURRENT_REQUESTS, SENTENCES_PER_LARGE_REQUEST
        ),
    )
    return Load(before, await read_metrics(client), warm, singles, large)


# ---- a command that is refused, and a stop with work in flight -----------------------------

REFUSAL_DEADLINE_SECONDS: Final = 120.0


@dataclass(frozen=True)
class Refusal:
    """What a `rinchen serve` that was refused left: its exit status and both outputs."""

    returncode: int
    stdout: str
    stderr: str


def run_refused(command: Sequence[str | Path], workdir: Path, *, name: str) -> Refusal:
    """Run `command` (a `rinchen serve` that must be refused) until it ends. A command that is
    still running after `REFUSAL_DEADLINE_SECONDS` was not refused (it is serving): its process
    group is stopped and the run fails."""
    server = start_server(command, workdir, port=0, name=name)
    try:
        if not wait_for_exit(server.process, REFUSAL_DEADLINE_SECONDS):
            raise AssertionError(
                f"the command was not refused: still running after "
                f"{REFUSAL_DEADLINE_SECONDS} s\n{_tail(server.stderr())}"
            )
    finally:
        returncode = stop_server(server)
    return Refusal(returncode, server.stdout(), server.stderr())


async def wait_until_submitted(
    probe: httpx.AsyncClient,
    total: int,
    *,
    baseline: float,
    deadline_seconds: float = 60.0,
) -> tuple[float, float]:
    """Poll `/metrics` until `total` sentences have reached the micro-batcher since the reading
    `baseline` of `rinchen_batch_size_sum` (each is queued or in a batch that started), and
    return the sentences queued and the batches in flight at that moment: one reading of the
    gauges, so the two agree. Every request is then inside the service."""
    started = time.perf_counter()
    while True:
        text = (await probe.get("/metrics")).text
        queued = sample(text, "rinchen_queue_depth") or 0.0
        begun = (sample(text, "rinchen_batch_size_sum") or 0.0) - baseline
        if queued + begun >= total:
            return queued, sample(text, "rinchen_inflight_batches") or 0.0
        if time.perf_counter() - started > deadline_seconds:
            raise AssertionError(
                f"only {queued + begun} of {total} sentences reached the batcher"
            )


async def batch_size_sum(probe: httpx.AsyncClient) -> float:
    """The sentences the micro-batcher has handed to the engine so far."""
    text = (await probe.get("/metrics")).text
    return sample(text, "rinchen_batch_size_sum") or 0.0


async def refused_connection(server: Server, *, deadline_seconds: float = 30.0) -> int:
    """Open a new connection to the server until one is refused (`ConnectError`: nothing
    listens) and return how many were tried. A connection that is accepted meanwhile is the
    server before it closed its listener."""
    started = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        async with httpx.AsyncClient(timeout=2.0) as fresh:
            try:
                await fresh.get(f"{server.url}/healthz")
            except httpx.ConnectError:
                return attempts
            except httpx.TransportError:
                pass
        if time.perf_counter() - started > deadline_seconds:
            raise AssertionError(
                f"the server still accepted connections {deadline_seconds} s after SIGTERM"
            )
