"""The `serve` stage of the tiny pipeline: the service, checked in process (D15, spec P2 and F10).

The stage starts the real service app over the int8 export with `fastapi.testclient.TestClient`, which
calls the app through an ASGI transport inside this process: no socket is opened, no port is bound and
no connection is made. It asks five questions in a fixed order and judges the answers: `/healthz` and
`/readyz` (200 both), a translation of two memorised sentences (200, the status `ok` and a script
purity of 1.0 for each), a target the model does not write (400 with an error code) and `/metrics`
(200, with exactly one translation counted as 200). The answers are never kept as text: the facts are
the status codes, the engine's statuses and purities, the error code (an identifier) and the counter.

The service logs JSON lines to the standard error stream through `rinchen.utils.logging`, whose first
use configures the root logger for the rest of the process, and httpx logs every request at INFO. A
pipeline's output is its own, so during the stage the logger the service asks for is a silent one
(`rinchen.utils.logging.get` is replaced for the length of the stage and put back) and logging is
switched off with `logging.disable`, whose previous setting is restored: nothing reaches either
stream and the logging state is exactly what it was, the first use of the process's logging included.

`config.serve.config` names the service's configuration; the stage overrides the threads to one each and
reads no `RINCHEN_SERVE_*` variable of the environment.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Final, Protocol
from unittest import mock

from fastapi.testclient import TestClient

from rinchen.config.eval_suites import HUB_LANGUAGE
from rinchen.config.serve import load_serve_config
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.serve.app import create_app
from rinchen.serve.engine import ServingEngine
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.fixtures import Fixtures
from rinchen.tiny.states import FAIL_INPUT, StageRecord, fail_check
from rinchen.tiny.step_runner import path_in
from rinchen.tiny.thresholds import GATED_SYSTEM
from rinchen.utils import logging as rinchen_logging

SERVE_OVERRIDES: Final = {"intra_threads": 1, "inter_threads": 1}
SOURCE_LANGUAGE: Final = "eng_Latn"
# A registry language the tiny model does not write.
REFUSED_TARGET: Final = "deu_Latn"
PROBE_CHECK: Final = fail_check("service_probe")
TRANSLATE_CHECK: Final = fail_check("service_translate")
REFUSAL_CHECK: Final = fail_check("service_refusal")
METRICS_CHECK: Final = fail_check("service_metrics")
SENTENCES: Final = 2
_COUNTER: Final = "rinchen_requests_total"
_COUNTER_LABELS: Final = {"route": "translate", "status": "200"}
_LABEL: Final = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_CODE: Final = re.compile(r"[a-z][a-z0-9_]{0,31}")
_OK: Final = 200
_REFUSED: Final = 400


class _Response(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


class _Client(Protocol):
    def get(self, url: str) -> _Response: ...

    def post(self, url: str, *, json: Any) -> _Response: ...


class _SilentLog:
    """The logger of the service while the stage runs: it takes every event and says nothing."""

    def debug(self, event: str, **fields: Any) -> None:
        return None

    def info(self, event: str, **fields: Any) -> None:
        return None

    def warning(self, event: str, **fields: Any) -> None:
        return None

    def error(self, event: str, **fields: Any) -> None:
        return None


@contextmanager
def _logging_silenced() -> Iterator[None]:
    """No line of any logger reaches a stream, and what logging was is what it is afterwards: the
    service gets a silent logger, so the process's logging is not configured by it, and the global
    switch of the `logging` module is put back to its previous value however the block ends."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with mock.patch.object(rinchen_logging, "get", lambda name: _SilentLog()):
            yield
    finally:
        logging.disable(previous)


def memorised_sentences(fixtures: Fixtures) -> tuple[str, str]:
    """Two English sentences the model was trained on: the two shortest `base` rows of the English
    pair file's train split (a short request is a cheap one), the first of the equally short ones
    first."""
    rows = sorted(
        (
            row.src
            for row in fixtures.pairs[f"{SOURCE_LANGUAGE}-{HUB_LANGUAGE}"]
            if row.split == "train" and row.kind == "base"
        ),
        key=len,
    )
    if len(rows) < SENTENCES:
        raise DataError("the fixtures have fewer than two memorised sentences")
    return rows[0], rows[1]


def _object(response: _Response) -> Mapping[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _translation(
    response: _Response,
) -> tuple[list[str] | None, list[float | None] | None]:
    """The statuses and the script purities a translate answer holds, when it holds both as lists of
    one entry per sentence."""
    body = _object(response)
    if body is None:
        return None, None
    statuses, purities = body.get("statuses"), body.get("script_purity")
    if not (
        isinstance(statuses, list)
        and len(statuses) == SENTENCES
        and all(isinstance(status, str) for status in statuses)
    ):
        statuses = None
    if not (
        isinstance(purities, list)
        and len(purities) == SENTENCES
        and all(value is None or _is_number(value) for value in purities)
    ):
        purities = None
    return statuses, purities


def _error_code(response: _Response) -> str | None:
    body = _object(response)
    error = None if body is None else body.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and _CODE.fullmatch(code):
        return code
    return None


def _counter(text: str) -> float | None:
    """The value of the sample of the translations answered with 200, when it is there once with
    exactly those labels (the labels may come in either order)."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        if name == _COUNTER and dict(_LABEL.findall(labels)) == _COUNTER_LABELS:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def exercise_service(
    client: _Client, texts: tuple[str, str], *, beam_size: int
) -> tuple[dict[str, Any], str | None]:
    """Ask the five questions of the stage of `client` and judge the answers.

    Returns the facts that the answers give (always the five status codes, in the order asked) and the
    check that failed first, `None` when none did: the probes, the translation, the refusal and the
    counters, in that order.
    """
    request = {"texts": list(texts), "src": SOURCE_LANGUAGE, "beam_size": beam_size}
    health = client.get("/healthz")
    ready = client.get("/readyz")
    translated = client.post("/v1/translate", json={**request, "tgt": HUB_LANGUAGE})
    refused = client.post("/v1/translate", json={**request, "tgt": REFUSED_TARGET})
    metrics = client.get("/metrics")
    statuses, purities = _translation(translated)
    code = _error_code(refused)
    counter = _counter(metrics.text)
    facts: dict[str, Any] = {
        "status_codes": [
            response.status_code
            for response in (health, ready, translated, refused, metrics)
        ],
        "statuses": statuses,
        "script_purity": purities,
        "error_code": code,
        "translate_200_total": counter,
    }
    return facts, _first_failure(facts, statuses, purities)


def _first_failure(
    facts: Mapping[str, Any],
    statuses: Sequence[str] | None,
    purities: Sequence[float | None] | None,
) -> str | None:
    health, ready, translated, refused, metrics = facts["status_codes"]
    if (health, ready) != (_OK, _OK):
        return PROBE_CHECK
    pure = purities is not None and all(
        value is not None and value == 1.0 for value in purities
    )
    if translated != _OK or statuses != ["ok"] * SENTENCES or not pure:
        return TRANSLATE_CHECK
    if refused != _REFUSED or facts["error_code"] is None:
        return REFUSAL_CHECK
    if metrics != _OK or facts["translate_200_total"] != 1.0:
        return METRICS_CHECK
    return None


def _failed(reason: str, facts: Mapping[str, Any]) -> StageResult:
    return StageResult(StageRecord("serve", "fail", reason), dict(facts))


def run_serve(ctx: StageContext, *, engine: ServingEngine | None = None) -> StageResult:
    """Check the service over the int8 export (`08_export/int8`) or, for a test, over `engine`.

    The service configuration is the profile's `serve.config` with one thread each and nothing from
    the environment. A configuration or an export that cannot be used is the failure `an input is
    missing or differs`; an engine that does not come up (an upstream failure) and a failed check are
    `a check of the stage failed` with the id of the check (`service_probe` for the engine). The stage
    writes no file. The facts are the five status codes, the statuses and purities of the
    translation, the error code of the refusal and the translation counter.
    """
    try:
        config = load_serve_config(
            ctx.config.serve.config, env={}, overrides=SERVE_OVERRIDES
        )
        texts = memorised_sentences(ctx.fixtures)
        model = None if engine is not None else path_in(ctx, "export", GATED_SYSTEM)
        with _logging_silenced():
            app = create_app(config, engine=engine, model=model)
            with TestClient(app) as client:
                facts, failure = exercise_service(
                    client, texts, beam_size=ctx.config.decode.beam_size
                )
    except ConfigError, DataError:
        return _failed(FAIL_INPUT, {})
    except UpstreamError:
        # The engine failed to come up (CTranslate2 refused the model): the probes would not pass.
        return _failed(PROBE_CHECK, {})
    if failure is not None:
        return _failed(failure, facts)
    return StageResult(StageRecord("serve", "ok", detail=copy.deepcopy(facts)), facts)
