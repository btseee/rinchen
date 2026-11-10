"""The translation service (D11, T15 to T17): a FastAPI app over one engine.

    POST /v1/translate   the texts, the languages and how to decode them (`serve.schemas`)
    GET  /healthz        200 while the process runs
    GET  /readyz         200 once the engine is loaded and warmed up, until the stop begins
    GET  /metrics        Prometheus text, one registry per app (`serve.metrics`)

`create_app(config, engine=...)` takes an engine of the caller's (the tests', which the app then
owns and closes) or, with `model=`, opens the export in the lifespan: the engine is loaded and
one warm-up translation runs BEFORE readiness turns true, and a failure there ends the lifespan
with the error, so the server exits non-zero instead of listening with no model. The app holds
its state in `app.state.serving` (`ServeState`) and its routes are plain functions over it;
nothing is global.

A translate request is parsed by hand (one body, and `BodyLimit` of `serve._http` must be able to
stop it), validated by the bounded model of the config, refused with 503 `not_ready` or `draining`
when it reaches the app before the engine is ready or after the stop began, and its texts become
jobs of the micro-batcher: concurrent requests are decoded in one engine call when they share a
key. A request that waits longer than `request_timeout_seconds` is a 504. The batch the worker
thread runs is `run_jobs`, which also keeps one refused request from failing the others of its
batch.

What a client can see of the start and of the stop depends on the server in front of the app.
With `uvicorn` (`rinchen serve`) it runs the start of the lifespan BEFORE it binds the port and
closes the listener BEFORE it runs the end of the lifespan, so over the wire "not ready" and
"draining" are both a refused connection: the 503s and a false `/readyz` exist for an app driven
in process (the tests) or served by something that binds first, and nothing here makes them
visible on a socket. The port opens only after the export has been hashed (by the command and
again by the engine), loaded and warmed up: a health check on `/readyz` fails until then, so its
start period must be sized by the time of that start for the model (UNMEASURED for a real model).

The stop (T16): `uvicorn` has already closed the listener and waited for the requests in flight
(`graceful_wait_seconds` of the shutdown budget, then it cancels what is left); the end of the
lifespan flips readiness to false first (in process a probe sees a service that is going away;
over a socket nothing listens any more), then closes the micro-batcher with the rest of the
budget (`drain_wait_seconds`): it runs what is still queued and waits for what is in flight, and
fails the rest with a 503. With `uvicorn` that work is almost always done by then, because its own
wait answered the requests: a real SIGTERM logs `shutdown_drained` with `completed` 0 and
`failed` 0, and the batcher's drain matters only when the work outlasts the server's wait and the
requests were cancelled. The engine is closed last, but only when no batch is still running in it
(a timed-out batch keeps its thread; unloading the model under it is the engine's documented
hazard). No GPU was used and none is claimed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.types import Scope

from rinchen.config.serve import ServeConfig
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.infer.engine_types import LoadOptions
from rinchen.serve._http import (
    BodyLimit,
    EventLog,
    RequestRecorder,
    note_of,
    refusal,
    register_error_handlers,
)
from rinchen.serve._translate import (
    Answer,
    BatchKey,
    Job,
    Refused,
    ServingEngine,
    build_response,
    plan,
    run_jobs,
    statuses_of,
)
from rinchen.serve.batcher import Draining, MicroBatcher, QueueFull
from rinchen.serve.metrics import ServeMetrics
from rinchen.serve.schemas import (
    TranslateRequest,
    bounded_request_model,
    validation_message,
)
from rinchen.utils import logging as rinchen_logging

EngineOpener = Callable[[Path, ServeConfig], ServingEngine]

_JSON_TYPE: Final = "application/json"
_GENERIC_FAILURE: Final = "the model could not answer this request"
_DRAINING: Final = "the service is shutting down"
_NOT_READY: Final = "the model is not loaded yet"


def load_options(config: ServeConfig) -> LoadOptions:
    """How the engine is loaded for `config`: device, compute type and the two thread counts."""
    return LoadOptions(
        device=config.device,
        compute_type=config.compute_type,
        inter_threads=config.inter_threads,
        intra_threads=config.intra_threads,
    )


def _open_engine(model: Path, config: ServeConfig) -> ServingEngine:
    # Imported here: the engine is only needed when a model is opened, not to build the app.
    from rinchen.infer.engine import Engine

    return Engine.open(model, load_options(config))


@dataclass
class DrainTally:
    """The items answered and failed since the stop began, counted by the worker threads."""

    completed: int = 0
    failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *, completed: int = 0, failed: int = 0) -> None:
        with self._lock:
            self.completed += completed
            self.failed += failed


@dataclass
class ServeState:
    """Everything the routes of one app share. `engine` and `batcher` exist from the end of
    the start of the service to its stop."""

    config: ServeConfig
    metrics: ServeMetrics
    log: EventLog
    request_model: type[TranslateRequest]
    given_engine: ServingEngine | None
    model: Path | None
    open_engine: EngineOpener
    engine: ServingEngine | None = None
    batcher: MicroBatcher[BatchKey, Job, Answer | Refused] | None = None
    ready: bool = False
    draining: bool = False
    drain: DrainTally = field(default_factory=DrainTally)
    request_numbers: count[int] = field(default_factory=count)


def create_app(
    config: ServeConfig,
    *,
    engine: ServingEngine | None = None,
    model: Path | None = None,
    open_engine: EngineOpener | None = None,
) -> FastAPI:
    """The service for `config`, over `engine` or over the export in `model` (opened, with
    `open_engine` when given, in the lifespan).

    Raises:
        ConfigError: neither or both of `engine` and `model` are given, or the config names a
            script converter and no `open_engine` is given to load it (the service itself
            imports no converter).
    """
    if (engine is None) == (model is None):
        raise ConfigError("create_app needs an engine or a model directory, not both")
    if engine is None and config.script_converter is not None and open_engine is None:
        raise ConfigError(
            "the config names a script converter and nothing was given to load it: pass "
            "open_engine (the command `rinchen serve` does)"
        )
    log = rinchen_logging.get("rinchen.serve")
    state = ServeState(
        config=config,
        metrics=ServeMetrics(),
        log=log,
        request_model=bounded_request_model(
            max_texts=config.max_texts, max_text_chars=config.max_text_chars
        ),
        given_engine=engine,
        model=model,
        open_engine=open_engine if open_engine is not None else _open_engine,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await _start(state)
        try:
            yield
        finally:
            await _stop(state)

    # No OpenAPI document means no interactive documentation either: the surface is four routes.
    app = FastAPI(title="rinchen", lifespan=lifespan, openapi_url=None)
    app.state.serving = state
    app.add_middleware(BodyLimit, limit=config.max_body_bytes)
    app.add_middleware(RequestRecorder, metrics=state.metrics, log=log)
    register_error_handlers(app, log=log)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/readyz", readyz, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_route("/v1/translate", translate, methods=["POST"])
    return app


# ---- the start and the stop ---------------------------------------------------------------


async def _start(state: ServeState) -> None:
    """Load the engine, warm it up, build the batcher, and only then turn ready on."""
    engine = state.given_engine
    if engine is None:
        assert state.model is not None
        engine = await asyncio.to_thread(state.open_engine, state.model, state.config)
    try:
        await asyncio.to_thread(engine.warmup)
        state.batcher = _make_batcher(state, engine)
        state.engine = engine
        state.metrics.model(engine.info)
        state.ready = True
        state.metrics.ready(True)
    except BaseException:
        engine.close()
        raise
    config, info = state.config, engine.info
    state.log.info(
        "serve_start",
        model=info.name,
        quantization=info.quantization,
        device=info.device,
        compute_type=info.compute_type,
        model_sha256=info.model_sha256,
        manifest_sha256=info.manifest_sha256,
        inter_threads=config.inter_threads,
        intra_threads=config.intra_threads,
        beam_size=config.beam_size,
        max_batch_sentences=config.max_batch_sentences,
        max_wait_ms=config.max_wait_ms,
        max_queue_sentences=config.max_queue_sentences,
        max_texts=config.max_texts,
        max_text_chars=config.max_text_chars,
        max_body_bytes=config.max_body_bytes,
        request_timeout_seconds=config.request_timeout_seconds,
        shutdown_timeout_seconds=config.shutdown_timeout_seconds,
    )


def _make_batcher(
    state: ServeState, engine: ServingEngine
) -> MicroBatcher[BatchKey, Job, Answer | Refused]:
    config = state.config

    def run(key: BatchKey, jobs: list[Job]) -> list[Answer | Refused]:
        state.metrics.batch(len(jobs))
        try:
            answers = run_jobs(engine, key, jobs)
        except BaseException:
            _tally(state, failed=len(jobs))
            raise
        failed = sum(isinstance(answer, Refused) for answer in answers)
        _tally(state, completed=len(jobs) - failed, failed=failed)
        return answers

    return MicroBatcher(
        run,
        max_batch_sentences=config.max_batch_sentences,
        max_wait_seconds=config.max_wait_ms / 1000,
        max_queue_sentences=config.max_queue_sentences,
        workers=config.inter_threads,
    )


def _tally(state: ServeState, *, completed: int = 0, failed: int = 0) -> None:
    if state.draining:
        state.drain.add(completed=completed, failed=failed)


async def _stop(state: ServeState) -> None:
    """The end of the lifespan: readiness first, then the drain, then the engine."""
    state.ready = False
    state.draining = True
    state.metrics.ready(False)
    state.metrics.draining(True)
    engine, batcher = state.engine, state.batcher
    assert engine is not None and batcher is not None
    state.log.info("shutdown_begin", queued=batcher.depth, inflight=batcher.inflight)
    unanswered = await batcher.close(state.config.drain_wait_seconds)
    try:
        if batcher.inflight:
            state.log.warning("engine_left_loaded", batches=batcher.inflight)
        else:
            engine.close()
    finally:
        state.log.info(
            "shutdown_drained",
            completed=state.drain.completed,
            failed=state.drain.failed + unanswered,
        )


# ---- the probes and the metrics -----------------------------------------------------------


def _state(request: Request) -> ServeState:
    state: ServeState = request.app.state.serving
    return state


async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@dataclass(frozen=True)
class _Live:
    """The engine and the batcher of a service that is ready."""

    engine: ServingEngine
    batcher: MicroBatcher[BatchKey, Job, Answer | Refused]


def _live(state: ServeState) -> _Live | None:
    """The engine and the batcher while the service is ready: none before the start has
    finished, and none from the first moment of the stop."""
    if not state.ready or state.engine is None or state.batcher is None:
        return None
    return _Live(state.engine, state.batcher)


def _unavailable(state: ServeState, scope: Scope, *, retry: bool) -> Response:
    """The 503 of a service that is not ready: it is stopping, or it has not started."""
    if state.draining:
        return refusal(scope, 503, "draining", _DRAINING, retry=retry)
    return refusal(scope, 503, "not_ready", _NOT_READY, retry=retry)


async def readyz(request: Request) -> Response:
    state = _state(request)
    live = _live(state)
    if live is None:
        return _unavailable(state, request.scope, retry=False)
    info = live.engine.info
    return JSONResponse(
        {
            "status": "ready",
            "model": info.name,
            "quantization": info.quantization,
            "device": info.device,
            "compute_type": info.compute_type,
        }
    )


async def metrics(request: Request) -> Response:
    state = _state(request)
    if state.batcher is not None:
        state.metrics.queue_depth(state.batcher.depth)
        state.metrics.inflight(state.batcher.inflight)
    if state.engine is not None:
        state.metrics.engine_events(state.engine.counters())
    payload, content_type = state.metrics.render()
    return Response(payload, media_type=content_type)


# ---- translate ----------------------------------------------------------------------------


class _Expired(TimeoutError):
    """The request's own deadline passed (as opposed to a TimeoutError of the engine)."""


async def translate(request: Request) -> Response:
    state = _state(request)
    started = time.perf_counter()
    scope = request.scope
    parsed = await _parse(request, state)
    if isinstance(parsed, Response):
        return parsed
    body = parsed
    note = note_of(scope)
    note.texts = len(body.texts)
    live = _live(state)
    if live is None:
        return _unavailable(state, scope, retry=True)
    engine, batcher = live.engine, live.batcher
    if body.tgt in engine.info.targets:
        note.target = body.tgt
    try:
        key, jobs = plan(body, state.config, next(state.request_numbers))
        answers = await _submit(
            batcher, key, jobs, state.config.request_timeout_seconds
        )
        results = _answers_or_raise(answers)
    except QueueFull:
        return refusal(
            scope,
            503,
            "overloaded",
            "the service is busy: too many sentences are waiting",
            retry=True,
        )
    except Draining:
        return refusal(scope, 503, "draining", _DRAINING, retry=True)
    except (ConfigError, DataError) as error:
        return refusal(scope, 400, "bad_request", str(error))
    except _Expired:
        return refusal(
            scope, 504, "timeout", "the request took longer than the service allows"
        )
    except Exception as error:  # noqa: BLE001 -- anything else is a 500 with a generic message
        return _internal(state, scope, error)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    statuses = Counter(statuses_of(results))
    for status, number in statuses.items():
        state.metrics.sentences(status, number)
    response = build_response(
        body, results, model=engine.info.name, latency_ms=latency_ms
    )
    return JSONResponse(response.model_dump(mode="json"))


async def _submit(
    batcher: MicroBatcher[BatchKey, Job, Answer | Refused],
    key: BatchKey,
    jobs: Sequence[Job],
    seconds: float,
) -> list[Answer | Refused]:
    """The answers of `jobs`, or `_Expired` when they took longer than `seconds`."""
    deadline = asyncio.timeout(seconds)
    try:
        async with deadline:
            return await batcher.submit(key, jobs)
    except TimeoutError as error:
        if deadline.expired():
            raise _Expired from error
        raise


def _answers_or_raise(answers: Sequence[Answer | Refused]) -> list[Answer]:
    """`answers` as they are, or, when the engine refused this request in a batch, the error it
    gave."""
    results: list[Answer] = []
    for answer in answers:
        if isinstance(answer, Refused):
            raise answer.error
        results.append(answer)
    return results


async def _parse(request: Request, state: ServeState) -> TranslateRequest | Response:
    """The body as a request, or the 422 that says why it is none."""
    scope = request.scope
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != _JSON_TYPE:
        return refusal(
            scope,
            422,
            "validation_error",
            f"body: the content type must be {_JSON_TYPE}",
        )
    raw = await request.body()
    try:
        return state.request_model.model_validate_json(raw)
    except ValidationError as error:
        return refusal(
            scope,
            422,
            "validation_error",
            validation_message(error.errors(include_input=False)),
        )


def _internal(state: ServeState, scope: Scope, error: Exception) -> Response:
    """A 500 with a generic message. The log gets the type of the exception, and the message
    only of an `UpstreamError` (the engine's, which names a row and never quotes a text)."""
    fields = {"error_type": type(error).__name__}
    if isinstance(error, UpstreamError):
        fields["cause"] = str(error)
    state.log.error("request_failed", **fields)
    return refusal(scope, 500, "internal", _GENERIC_FAILURE)
