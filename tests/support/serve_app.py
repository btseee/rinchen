"""A fake engine, a recording log and an in-process client for the tests of the service
(D11, Task 9).

`FakeEngine` is the engine as the service sees it: `info`, `translate`, `translate_documents`,
`warmup`, `counters` and `close`, honouring the engine's contract (a target the model does not
write, a script for a target that has none or the same source and target raise `ConfigError`
before anything is decoded; a document needs a document model and a source language; a call is
all-or-nothing). Its answer depends on the text of a row, so a test asks for each status by the
text it writes (`default_answer`). It can be held on a gate (a call that started waits for
`release()`), delayed, made to fail, and it records every call; nothing loads a model, imports
`ctranslate2` or touches a GPU.

`RecordingLog` stands in for the logger `rinchen.utils.logging.get` returns, so a test reads the
events the service logged instead of a stderr (the first call of the real `get` configures the
root logger for the rest of the session, and `test_logging` expects to be the first). The
fixture `serve_environment` installs it and makes `ctranslate2` unimportable.

`make_client` starts a service over an engine, in this event loop: an `httpx.AsyncClient` over
`ASGITransport` (which does not run the lifespan) inside `app.router.lifespan_context(app)`.
`make_app` gives the app and a client without starting it, for the tests that start and stop it
step by step. `run` runs a scenario in a fresh loop with a bound, so a test cannot hang the suite.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
import pytest
from fastapi import FastAPI

from rinchen.config.serve import ServeConfig
from rinchen.errors import ConfigError, DataError
from rinchen.infer.doc_chunker import ChunkStats
from rinchen.infer.engine_types import (
    ENGINE_COUNTERS,
    DecodeOptions,
    DocumentResult,
    ModelInfo,
    Row,
    ScriptName,
    Status,
    Translation,
    selected_target,
)
from rinchen.infer.script_mask import script_hint
from rinchen.serve.app import create_app

KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
MON: Final = "mn_Mong"
SECRET: Final = "SECRET-TEXT-OF-A-USER"
# A Mongolian-script text: three letters (U+1820, U+1821, U+1822).
MONGOLIAN: Final = "".join(chr(code) for code in (0x1820, 0x1821, 0x1822))
LONG_TEXT: Final = 30
NO_OUTPUT_TEXT: Final = "silent"
TRUNCATED_SUFFIX: Final = "..."
# A held call that nobody releases is a bug of the test: it ends after this many seconds.
RELEASE_TIMEOUT: Final = 30.0
SCENARIO_SECONDS: Final = 60.0
NO_STATS: Final = ChunkStats(
    sentences=0,
    chunks=0,
    windows=0,
    aligned=0,
    fallbacks=(),
    singles=0,
    oversize=0,
    empty=0,
)


def translation_of(
    row: Row,
    text: str,
    status: Status,
    *,
    alternatives: tuple[str, ...] = (),
    purity: float | None = None,
) -> Translation:
    """The `Translation` of `row` that says `text` with `status`: its `src` is the language
    given or the script hint, as the engine's is."""
    return Translation(
        text=text,
        status=status,
        alternatives=alternatives,
        src=row.src or script_hint(row.text),
        tgt=selected_target(row.tgt, row.script),
        script_purity=purity,
        retried=False,
        zero_shot=False,
    )


def default_answer(row: Row) -> Translation:
    """A text with no letter or digit is `empty`, one of more than `LONG_TEXT` characters
    `too_long`, `NO_OUTPUT_TEXT` is `no_output`, one ending in `...` is `truncated` (the
    partial text is kept) and everything else is `ok`, translated as `<text>`. The
    alternatives of an `ok` text are `<text>#2` and `<text>#3`."""
    text = row.text
    if not any(character.isalnum() for character in text):
        return translation_of(row, "", "empty")
    if len(text) > LONG_TEXT:
        return translation_of(row, "", "too_long")
    if text == NO_OUTPUT_TEXT:
        return translation_of(row, "", "no_output")
    if text.endswith(TRUNCATED_SUFFIX):
        return translation_of(row, f"<{text}>", "truncated", purity=1.0)
    return translation_of(
        row,
        f"<{text}>",
        "ok",
        alternatives=(f"<{text}>#2", f"<{text}>#3"),
        purity=1.0,
    )


@dataclass(frozen=True)
class Call:
    """One call of the engine: `kind` is `rows` or `documents`; `texts` are the texts of the
    rows or the documents; `rows` the rows (documents: none)."""

    kind: str
    texts: tuple[str, ...]
    options: DecodeOptions | None
    rows: tuple[Row, ...] = ()
    src: str | None = None
    tgt: str | None = None
    script: ScriptName | None = None


@dataclass
class FakeEngine:
    """The engine as the service sees it (see the module's text).

    `targets` are what the model writes; `answer` gives the translation of a row;
    `translate_failure` is raised by every call after the gate (a call that raises adds
    nothing); `warmup_failure` by `warmup`; `close_failure` by `close`; `delay_seconds` is
    how long a call takes (a bounded wait, never a sleep of the loop); `counter_values` are
    the counters the engine reports (the others are 0)."""

    name: str = "toy-model"
    targets: tuple[str, ...] = (KHK, ENG)
    document_level: bool = False
    answer: Callable[[Row], Translation] = default_answer
    translate_failure: BaseException | None = None
    warmup_failure: BaseException | None = None
    close_failure: BaseException | None = None
    delay_seconds: float = 0.0
    counter_values: dict[str, int] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    warmups: int = 0
    closed: int = 0
    answered_when_closed: int | None = None
    _condition: threading.Condition = field(default_factory=threading.Condition)
    _gate: threading.Event = field(default_factory=threading.Event)
    _warmup_gate: threading.Event = field(default_factory=threading.Event)
    _never: threading.Event = field(default_factory=threading.Event)
    _finished: int = 0
    _warmup_entered: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._gate.set()
        self._warmup_gate.set()

    # ---- what the service reads -----------------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            quantization="float32",
            device="cpu",
            compute_type="float32",
            placement="src",
            directions=(f"{ENG}-{KHK}",),
            targets=self.targets,
            document_level=self.document_level,
            model_sha256="a" * 64,
            manifest_sha256="b" * 64,
        )

    def counters(self) -> dict[str, int]:
        return dict.fromkeys(ENGINE_COUNTERS, 0) | self.counter_values

    # ---- what the tests steer -------------------------------------------------------------

    def hold(self) -> None:
        """Calls that start from now on wait for `release()`."""
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    def hold_warmup(self) -> None:
        self._warmup_gate.clear()

    def release_warmup(self) -> None:
        self._warmup_gate.set()

    def wait_for_warmup(self) -> bool:
        """Block (bounded) until `warmup` has been entered."""
        return self._warmup_entered.wait(RELEASE_TIMEOUT)

    def wait_for_calls(self, count: int) -> bool:
        """Block (bounded) until `count` calls have started."""
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.calls) >= count, RELEASE_TIMEOUT
            )

    def wait_for_finished(self, count: int) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._finished >= count, RELEASE_TIMEOUT
            )

    @property
    def finished(self) -> int:
        with self._condition:
            return self._finished

    def started_items(self) -> int:
        """The texts of every call that has started."""
        with self._condition:
            return sum(len(call.texts) for call in self.calls)

    def texts_seen(self) -> list[str]:
        with self._condition:
            return [text for call in self.calls for text in call.texts]

    # ---- the engine's own calls -----------------------------------------------------------

    def warmup(self) -> None:
        self._warmup_entered.set()
        if not self._warmup_gate.wait(RELEASE_TIMEOUT):
            raise AssertionError("a held warm-up was never released")
        self.warmups += 1
        if self.warmup_failure is not None:
            raise self.warmup_failure

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        for row in rows:
            self._require_servable(row.src, row.tgt, row.script)
        self._enter(Call("rows", tuple(r.text for r in rows), options, tuple(rows)))
        try:
            return [self.answer(row) for row in rows]
        finally:
            self._leave()

    def translate_documents(
        self,
        documents: Sequence[str],
        src: str | None,
        tgt: str,
        options: DecodeOptions | None = None,
        *,
        script: ScriptName | None = None,
    ) -> list[DocumentResult]:
        if not self.document_level:
            raise ConfigError(
                "the model does not translate documents: its record says "
                "document_level is false"
            )
        if src is None:
            raise ConfigError(
                "a document needs its source language (src) to be cut into sentences"
            )
        self._require_servable(src, tgt, script)
        for index, document in enumerate(documents):
            if "<sep>" in document:
                raise DataError(
                    f"document {index} contains the literal '<sep>', which a window "
                    "would read as the boundary of two sentences"
                )
        self._enter(Call("documents", tuple(documents), options, (), src, tgt, script))
        try:
            results = []
            for document in documents:
                translation = self.answer(Row(document, src, tgt, script))
                results.append(
                    DocumentResult(
                        text=translation.text,
                        translations=(translation,),
                        stats=NO_STATS,
                    )
                )
            return results
        finally:
            self._leave()

    def close(self) -> None:
        with self._condition:
            self.closed += 1
            self.answered_when_closed = self._finished
        if self.close_failure is not None:
            raise self.close_failure

    # ---- inside --------------------------------------------------------------------------

    def _require_servable(self, src: str | None, tgt: str, script: str | None) -> None:
        selected = selected_target(tgt, script)
        if selected not in self.targets:
            raise ConfigError(
                f"the model does not write {selected}: its targets are "
                f"{', '.join(self.targets)}"
            )
        if src == selected:
            raise ConfigError(
                f"the source and the target language are both {selected}: nothing to "
                "translate"
            )

    def _enter(self, call: Call) -> None:
        with self._condition:
            self.calls.append(call)
            self._condition.notify_all()
        if not self._gate.wait(RELEASE_TIMEOUT):
            raise AssertionError("a held call was never released")
        if self.delay_seconds:
            self._never.wait(self.delay_seconds)
        if self.translate_failure is not None:
            self._leave()
            raise self.translate_failure

    def _leave(self) -> None:
        with self._condition:
            self._finished += 1
            self._condition.notify_all()


class RecordingLog:
    """The logger of the service as a test sees it: every event with its level and fields."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def _add(self, level: str, event: str, fields: dict[str, Any]) -> None:
        with self._lock:
            self.events.append((level, event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self._add("info", event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._add("warning", event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._add("error", event, fields)

    def named(self, event: str) -> list[dict[str, Any]]:
        """The fields of every event called `event`, in order."""
        with self._lock:
            return [fields for _, name, fields in self.events if name == event]

    def names(self) -> list[str]:
        with self._lock:
            return [name for _, name, _ in self.events]

    def text(self) -> str:
        """Every event as one JSON line each, for a test that looks for a string in all of
        them."""
        with self._lock:
            return "\n".join(
                json.dumps(
                    {"level": level, "event": event, **fields},
                    default=repr,
                    ensure_ascii=False,
                )
                for level, event, fields in self.events
            )


@pytest.fixture
def recording_log(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    """The service's logger replaced by a `RecordingLog`, and the real `ctranslate2` made
    unimportable, so a test that reached it would fail instead of loading it."""
    log = RecordingLog()
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setattr("rinchen.utils.logging.get", lambda name: log)
    return log


def serve_config(**changes: Any) -> ServeConfig:
    """A `ServeConfig` for a test: a short batching wait and a short shutdown wait, the other
    values the defaults, with `changes` applied."""
    values: dict[str, Any] = {
        "max_wait_ms": 1.0,
        "shutdown_timeout_seconds": 5.0,
        **changes,
    }
    return ServeConfig(**values)


@dataclass
class Running:
    """A service under test: its client, its app, its engine and its log."""

    client: httpx.AsyncClient
    app: FastAPI
    engine: Any
    log: RecordingLog

    @property
    def state(self) -> Any:
        return self.app.state.serving


@asynccontextmanager
async def make_app(
    log: RecordingLog,
    engine: Any = None,
    *,
    config: ServeConfig | None = None,
    **changes: Any,
) -> AsyncIterator[Running]:
    """An app over `engine` (a `FakeEngine` when none is given) and a client for it, NOT
    started: the test enters and leaves `app.router.lifespan_context(app)` itself. An exception
    of the app reaches the client as a 500 response, not as a raised error."""
    fake = FakeEngine() if engine is None else engine
    app = create_app(config or serve_config(**changes), engine=fake)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://service"
    ) as client:
        yield Running(client, app, fake, log)


@asynccontextmanager
async def make_client(
    log: RecordingLog,
    engine: Any = None,
    *,
    config: ServeConfig | None = None,
    **changes: Any,
) -> AsyncIterator[Running]:
    """`make_app` with the service started (the engine warmed up, ready) and stopped when the
    block ends."""
    # The second context needs `running`, so the two cannot be one `async with`.
    async with make_app(log, engine, config=config, **changes) as running:  # noqa: SIM117
        async with running.app.router.lifespan_context(running.app):
            yield running


def run[T](scenario: Coroutine[Any, Any, T], seconds: float = SCENARIO_SECONDS) -> T:
    """Run `scenario` in a fresh event loop; it fails after `seconds` instead of hanging."""

    async def bounded() -> T:
        return await asyncio.wait_for(scenario, seconds)

    return asyncio.run(bounded())


async def until(condition: Callable[[], bool], seconds: float = 10.0) -> None:
    """Wait (bounded) for `condition` by yielding to the loop."""
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.001)


async def engine_started(engine: FakeEngine, count: int) -> None:
    """Wait (bounded, off the loop) until `engine` has been given `count` calls."""
    assert await asyncio.to_thread(engine.wait_for_calls, count), (
        f"fewer than {count} calls started"
    )


async def post(
    client: httpx.AsyncClient, payload: dict[str, Any] | None = None, **fields: Any
) -> httpx.Response:
    """`POST /v1/translate` with a minimal valid body (`texts` `["hello"]`, `tgt` the first
    target) and the changes of `payload` and `fields`."""
    body: dict[str, Any] = {"texts": ["hello"], "tgt": KHK, **(payload or {}), **fields}
    return await client.post("/v1/translate", json=body)


def error_of(response: httpx.Response) -> dict[str, str]:
    """The `error` member of an error response, asserting that it is the only member."""
    body = response.json()
    assert list(body) == ["error"], body
    error: dict[str, str] = body["error"]
    assert sorted(error) == ["code", "message"], error
    return error


_SAMPLE_LABEL: Final = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def sample(metrics_text: str, family: str, /, **labels: str) -> float | None:
    """The value of the sample `family` with exactly these `labels` (in any order) in a
    Prometheus text, or `None` when there is none."""
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, rest = head.partition("{")
        if name == family and dict(_SAMPLE_LABEL.findall(rest)) == labels:
            return float(value)
    return None
