"""Shared setup of the `rinchen translate` tests (D11, Task 5).

The command touches the model in ONE place, `rinchen.cli.translate.open_engine`; `install`
replaces it by a function that hands out a `FakeEngine` (or any engine of the test's own, such
as the scripted one of `tests.support.engines`) and records what it was asked to open. Nothing
here loads a model, imports `ctranslate2` or touches a GPU.

`FakeEngine` answers a row by its text, so that a test can ask for each status by the text it
writes: a text with no letter or digit is `empty`, one of more than `LONG_TEXT` characters is
`too_long`, `NO_OUTPUT_TEXT` is `no_output`, one ending in `...` is `truncated` (its partial
text is kept) and everything else is `ok`, translated as `<text>`. It records every call, and
a shared `events` list says in what order the model was opened, the input read and the engine
asked.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.infer.doc_chunker import ChunkStats
from rinchen.infer.engine_types import (
    ENGINE_COUNTERS,
    DecodeOptions,
    DocumentResult,
    LoadOptions,
    Row,
    Status,
    Translation,
    selected_target,
)

runner = CliRunner()

MODEL: Final = "models/tiny"
TGT: Final = "khk_Cyrl"
SRC: Final = "eng_Latn"
LONG_TEXT: Final = 30
NO_OUTPUT_TEXT: Final = "silent"
TRUNCATED_SUFFIX: Final = "..."
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

Answer = Callable[[Row], Translation]


def translation_of(
    row: Row,
    text: str,
    status: Status,
    *,
    alternatives: tuple[str, ...] = (),
    purity: float | None = None,
) -> Translation:
    """The `Translation` of `row` that says `text` with `status`: its `src` is the language
    given (`und` when none), its `tgt` the target the row selects."""
    return Translation(
        text=text,
        status=status,
        alternatives=alternatives,
        src=row.src or "und",
        tgt=selected_target(row.tgt, row.script),
        script_purity=purity,
        retried=False,
        zero_shot=False,
    )


def default_answer(row: Row) -> Translation:
    """The answer by text that the module's text describes."""
    text = row.text
    if not any(character.isalnum() for character in text):
        return translation_of(row, "", "empty")
    if len(text) > LONG_TEXT:
        return translation_of(row, "", "too_long")
    if text == NO_OUTPUT_TEXT:
        return translation_of(row, "", "no_output")
    if text.endswith(TRUNCATED_SUFFIX):
        return translation_of(row, f"<{text}>", "truncated", purity=1.0)
    return translation_of(row, f"<{text}>", "ok", purity=1.0)


@dataclass(frozen=True)
class RowsCall:
    """One call of `translate`: the rows and the options it was given."""

    rows: list[Row]
    options: DecodeOptions | None


@dataclass(frozen=True)
class DocumentsCall:
    """One call of `translate_documents`."""

    documents: list[str]
    src: str | None
    tgt: str
    options: DecodeOptions | None
    script: str | None


@dataclass(frozen=True)
class RequestCheck:
    """One call of `require_request`: the languages of the request and whether it is for
    documents."""

    src: str | None
    tgt: str
    script: str | None
    documents: bool


@dataclass
class FakeEngine:
    """The engine as the command sees it. `answer` gives the translation of a row;
    `on_call` is called with the number (from 1) of each `translate` or `translate_documents`
    call before it answers; `error` is raised by the call number `error_on` (default the
    first); `request_error` is raised by `require_request`, which records its calls in
    `checked` and is not a call of the numbering; `counter_values` are the engine counters
    (the others are 0)."""

    answer: Answer = default_answer
    on_call: Callable[[int], None] | None = None
    error: BaseException | None = None
    error_on: int = 1
    counter_values: dict[str, int] = field(default_factory=dict)
    request_error: BaseException | None = None
    checked: list[RequestCheck] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    calls: list[RowsCall] = field(default_factory=list)
    document_calls: list[DocumentsCall] = field(default_factory=list)
    closed: int = 0
    _call_number: int = 0

    def require_request(
        self,
        src: str | None,
        tgt: str,
        script: str | None = None,
        *,
        documents: bool = False,
    ) -> None:
        self.checked.append(RequestCheck(src, tgt, script, documents))
        if self.request_error is not None:
            raise self.request_error

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        self._enter("translate")
        self.calls.append(RowsCall(list(rows), options))
        return [self.answer(row) for row in rows]

    def translate_documents(
        self,
        documents: Sequence[str],
        src: str | None,
        tgt: str,
        options: DecodeOptions | None = None,
        *,
        script: Any = None,
    ) -> list[DocumentResult]:
        self._enter("translate")
        self.document_calls.append(
            DocumentsCall(list(documents), src, tgt, options, script)
        )
        results = []
        for document in documents:
            row = Row(document, src, tgt, script)
            translation = self.answer(row)
            results.append(
                DocumentResult(
                    text=translation.text, translations=(translation,), stats=NO_STATS
                )
            )
        return results

    def counters(self) -> dict[str, int]:
        return dict.fromkeys(ENGINE_COUNTERS, 0) | self.counter_values

    def close(self) -> None:
        self.closed += 1

    def _enter(self, event: str) -> None:
        self._call_number += 1
        self.events.append(event)
        if self.on_call is not None:
            self.on_call(self._call_number)
        if self.error is not None and self._call_number == self.error_on:
            raise self.error


@dataclass(frozen=True)
class OpenCall:
    """What the command asked `open_engine` for."""

    model: Path
    load: LoadOptions
    converter_paths: tuple[Path, Path] | None


@dataclass
class Opened:
    """The seam: the calls of `open_engine`, the engine it handed out and the events."""

    engine: Any
    calls: list[OpenCall] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def install(
    monkeypatch: pytest.MonkeyPatch,
    engine: Any | None = None,
    *,
    error: BaseException | None = None,
) -> Opened:
    """Replace `open_engine` by a seam that records its call and returns `engine` (a
    `FakeEngine` when none is given), or raises `error`. The engine's own `events` list (when
    it has one) is shared with the seam's, so `opened.events` reads `open`, then what the
    engine saw."""
    opened = Opened(FakeEngine() if engine is None else engine)
    if hasattr(opened.engine, "events"):
        opened.events = opened.engine.events

    def open_engine(
        model: Path, load: LoadOptions, converter_paths: tuple[Path, Path] | None
    ) -> Any:
        opened.events.append("open")
        opened.calls.append(OpenCall(model, load, converter_paths))
        if error is not None:
            raise error
        return opened.engine

    monkeypatch.setattr("rinchen.cli.translate.open_engine", open_engine)
    return opened


@dataclass
class BrokenStdout:
    """A standard output whose reader has gone away: writing and flushing it raises
    `BrokenPipeError`, as `rinchen translate ... | head -1` does once `head` has exited. It
    has a real descriptor (`stream.fileno()`), so that a test can see what the command does
    with it."""

    stream: Any
    descriptor: int


def broken_stdout(monkeypatch: pytest.MonkeyPatch) -> BrokenStdout:
    """Make the command's standard output a `BrokenStdout`; the caller closes `stream`."""
    reader, writer = os.pipe()
    os.close(reader)
    stdout = BrokenStdout(os.fdopen(writer, "wb"), writer)
    real = typer.get_binary_stream

    def get_binary_stream(name: Any, *args: Any, **kwargs: Any) -> Any:
        return stdout.stream if name == "stdout" else real(name, *args, **kwargs)

    monkeypatch.setattr(typer, "get_binary_stream", get_binary_stream)
    return stdout


class ReadProbe(io.BytesIO):
    """A standard input that says in `events` when it is read (once per call of `readline`,
    `read` or `readlines`)."""

    def __init__(self, content: bytes, events: list[str]) -> None:
        super().__init__(content)
        self._events = events

    def readline(self, size: int | None = -1, /) -> bytes:
        self._events.append("read")
        return super().readline(size)

    def read(self, size: int | None = -1, /) -> bytes:
        self._events.append("read")
        return super().read(size)

    def readlines(self, hint: int = -1, /) -> list[bytes]:
        self._events.append("read")
        return super().readlines(hint)


def probe_stdin(
    monkeypatch: pytest.MonkeyPatch, content: bytes, events: list[str]
) -> ReadProbe:
    """Make the command's standard input a `ReadProbe` of `content` (the runner takes bytes
    or text only), which says in `events` when it is read."""
    probe = ReadProbe(content, events)
    real = typer.get_binary_stream

    def get_binary_stream(name: Any, *args: Any, **kwargs: Any) -> Any:
        return probe if name == "stdin" else real(name, *args, **kwargs)

    monkeypatch.setattr(typer, "get_binary_stream", get_binary_stream)
    return probe


def translate_args(*extra: str, tgt: str | None = TGT) -> list[str]:
    """The arguments of a run: `translate --model MODEL --tgt TGT` and `extra`."""
    target = [] if tgt is None else ["--tgt", tgt]
    return ["translate", "--model", MODEL, *target, *extra]


def invoke(
    args: Sequence[str],
    *,
    stdin: bytes | str = b"",
    env: dict[str, str] | None = None,
) -> Result:
    """Run `rinchen` with `args` in this process, `stdin` as the standard input."""
    return runner.invoke(app, list(args), input=stdin, env=env)
