"""The hypotheses of D12 (spec T12): what a system wrote for each request of a suite, the runner
that asks an engine for them and the JSON Lines file that carries them from one system to
another.

A `Hypothesis` is a text and the status the engine gave it (`ok`, `empty`, `too_long`,
`truncated`, `no_output`); a status that is not `ok` is not an error, the text (possibly empty
or partial) is scored as it is and counted. `run_requests` makes ONE call to anything with a
`translate(rows, options)` method (`rinchen.infer.engine.Engine` has one, so do the fakes of the
tests) and returns one hypothesis per request, in order, never fewer.

A hypotheses file has one object per request with the keys `section`, `id`, `arm`, `status` and
`text` (`status` may be left out and then reads `ok`), sorted keys, text written as it is, one LF
per line. Reading takes the bytes ONCE: `read_hypotheses` returns the SHA-256 of the bytes it
parsed. The file must hold exactly the requests it is read against, each once; a refusal is a
`DataError` that names the line, the section, the id and the arm and never what the file held (a
part of a request is named only when it is well formed: a section of the format, a well-formed id,
an arm of the format). A baseline system's output enters D13's tables through this file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, get_args

from rinchen.errors import DataError, UpstreamError
from rinchen.eval.robust_suite import ARMS, SECTIONS, Request, RequestKey
from rinchen.eval.robust_suite_rules import JsonObject, is_id, parse_object
from rinchen.infer.engine_types import DecodeOptions, Row, Status, Translation
from rinchen.utils.part_file import replace_file

__all__ = [
    "MAX_HYPOTHESES_BYTES",
    "Hypothesis",
    "Translates",
    "hypotheses_from_bytes",
    "hypotheses_to_bytes",
    "read_hypotheses",
    "request_place",
    "run_requests",
    "write_hypotheses",
]

MAX_HYPOTHESES_BYTES: Final = 128 * 1024 * 1024
_STATUSES: Final = get_args(Status)
_FIELDS: Final = ("section", "id", "arm", "status", "text")
_REQUIRED: Final = ("section", "id", "arm", "text")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """What a system wrote for a request: its `text` and the engine's `status`.

    Raises:
        ValueError: `status` is not one of the five of the engine.
    """

    text: str
    status: Status

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {', '.join(_STATUSES)}")


class Translates(Protocol):
    """What the runner asks of an engine: `rinchen.infer.engine.Engine.translate`."""

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]: ...


def request_place(section: object, row_id: object, arm: object) -> str:
    """The request named by its parts, for a message. A part is named only when it is well
    formed (a section of the format, a well-formed id, an arm of the format): what a file
    holds in any other shape is never quoted. No part well formed: `a request`."""
    named = [
        f"section {section!r}" if section in SECTIONS else None,
        f"id {row_id!r}" if is_id(row_id) else None,
        f"arm {arm!r}" if arm in ARMS else None,
    ]
    return ", ".join(part for part in named if part is not None) or "a request"


def run_requests(
    requests: Sequence[Request], engine: Translates, options: DecodeOptions
) -> dict[RequestKey, Hypothesis]:
    """Ask `engine` to translate every request with ONE call, and return what it wrote
    in the order of the requests. The engine's own all-or-nothing errors propagate.

    Raises:
        ValueError: two requests have the same key.
        UpstreamError: the engine answered a different number of translations.
    """
    keys = _unique_keys(requests)
    rows = [
        Row(request.text, request.src_lang, request.tgt_lang, request.script)
        for request in requests
    ]
    answers = engine.translate(rows, options)
    if len(answers) != len(requests):
        raise UpstreamError(
            f"the engine answered {len(answers)} translations for "
            f"{len(requests)} requests"
        )
    return {
        key: Hypothesis(answer.text, answer.status)
        for key, answer in zip(keys, answers, strict=True)
    }


def read_hypotheses(
    path: Path, requests: Sequence[Request]
) -> tuple[dict[RequestKey, Hypothesis], str]:
    """The hypotheses in the file `path` for `requests`, in the order of the requests, and
    the SHA-256 of the bytes that were parsed. The file is opened once and at most one byte
    over the cap is read.

    Raises:
        DataError: the file cannot be read, or is not the hypotheses of `requests`.
        ValueError: two requests have the same key.
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_HYPOTHESES_BYTES + 1)
    except OSError as error:
        raise DataError(f"cannot read the hypotheses file {path}") from error
    digest = hashlib.sha256(data).hexdigest()
    return hypotheses_from_bytes(data, requests), digest


def hypotheses_from_bytes(
    data: bytes, requests: Sequence[Request]
) -> dict[RequestKey, Hypothesis]:
    """The hypotheses in `data`, a JSON Lines document, for `requests`: exactly those, each
    once, returned in the order of the requests whatever the order of the lines.

    Raises:
        DataError: `data` is too large, not UTF-8, or not the hypotheses of `requests`; the
            message names the line and the request and never quotes the file.
        ValueError: two requests have the same key.
    """
    keys = _unique_keys(requests)
    if len(data) > MAX_HYPOTHESES_BYTES:
        raise _too_large()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _refused("the file", "the text is not UTF-8") from None
    wanted = set(keys)
    found: dict[RequestKey, Hypothesis] = {}
    for number, line in enumerate(_lines(text), start=1):
        key, hypothesis = _read_line(number, line)
        place = f"line {number} ({request_place(*key)})"
        if key not in wanted:
            raise _refused(place, "is not a request of the suite")
        if key in found:
            raise _refused(place, "answers a request that is answered more than once")
        found[key] = hypothesis
    missing = [key for key in keys if key not in found]
    if missing:
        raise _refused(
            "the file",
            f"lacks {len(missing)} request(s), the first is {request_place(*missing[0])}",
        )
    return {key: found[key] for key in keys}


def write_hypotheses(hypotheses: Mapping[RequestKey, Hypothesis], path: Path) -> str:
    """Write `hypotheses` to `path`, in the order of the mapping, so that the file appears
    whole; return the SHA-256 of the bytes written.

    Raises:
        DataError: a text holds a lone surrogate, or the file would be larger than the
            reader accepts.
    """
    data = hypotheses_to_bytes(hypotheses)
    if len(data) > MAX_HYPOTHESES_BYTES:
        raise _too_large()
    replace_file(path, data)
    return hashlib.sha256(data).hexdigest()


def hypotheses_to_bytes(hypotheses: Mapping[RequestKey, Hypothesis]) -> bytes:
    """The UTF-8 bytes of the hypotheses file: one line each, in the order of the mapping.

    Raises:
        DataError: a text holds a lone surrogate.
    """
    chunks: list[bytes] = []
    for key, hypothesis in hypotheses.items():
        document = {
            "arm": key.arm,
            "id": key.id,
            "section": key.section,
            "status": hypothesis.status,
            "text": hypothesis.text,
        }
        line = json.dumps(document, sort_keys=True, ensure_ascii=False)
        try:
            chunks.append(f"{line}\n".encode())
        except UnicodeEncodeError:
            raise _refused(request_place(*key), "holds a lone surrogate") from None
    return b"".join(chunks)


def _unique_keys(requests: Sequence[Request]) -> list[RequestKey]:
    keys = [request.key for request in requests]
    if len(set(keys)) != len(keys):
        raise ValueError("the requests hold a key more than once")
    return keys


def _lines(text: str) -> list[str]:
    """The lines of `text`: split at LF only (a JSON string may hold any other separator),
    without the empty piece after the final LF."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _too_large() -> DataError:
    return DataError(
        "not a hypotheses file: the file is larger than "
        f"{MAX_HYPOTHESES_BYTES // 2**20} MiB"
    )


def _refused(place: str, problem: str) -> DataError:
    """The error for a file that breaks a rule. `place` is a line number and a request
    named by `request_place`: never a text or a key of the file."""
    return DataError(f"not a hypotheses file: {place}: {problem}")


def _read_line(number: int, line: str) -> tuple[RequestKey, Hypothesis]:
    """The request a line answers and what it says, checked field by field."""
    where = f"line {number}"
    try:
        parsed = json.loads(line, object_pairs_hook=parse_object)
    except ValueError, RecursionError:
        raise _refused(where, "is not valid JSON") from None
    if not isinstance(parsed, JsonObject):
        raise _refused(where, "is not a JSON object")
    place = f"{where} ({request_place(*_parts(parsed))})"
    if parsed.repeated:
        raise _refused(place, "a field appears more than once")
    if set(parsed) - set(_FIELDS):
        raise _refused(place, "has a field that is not part of the format")
    for field in _REQUIRED:
        if field not in parsed:
            raise _refused(place, f"field {field!r} is missing")
    for field in ("section", "id", "arm", "text"):
        if not isinstance(parsed[field], str):
            raise _refused(place, f"field {field!r} must be a string")
    status = parsed.get("status", "ok")
    if status not in _STATUSES:
        raise _refused(
            place, f"field 'status' must be one of {', '.join(_STATUSES)} or absent"
        )
    try:
        parsed["text"].encode("utf-8")
    except UnicodeEncodeError:
        raise _refused(place, "field 'text' holds a lone surrogate") from None
    return (
        RequestKey(parsed["section"], parsed["id"], parsed["arm"]),
        Hypothesis(parsed["text"], status),
    )


def _parts(parsed: JsonObject) -> tuple[object, object, object]:
    return parsed.get("section"), parsed.get("id"), parsed.get("arm")
