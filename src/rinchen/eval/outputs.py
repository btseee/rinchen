"""The outputs of D13 (spec T6): what a system wrote for each row of the directions of a suite, and
the JSON Lines file that carries them from one system to another.

An `Output` is a text and the status the engine gave it (`ok`, `empty`, `too_long`, `truncated`,
`no_output`; a baseline gives `ok` or `no_output`); a status that is not `ok` is not an error, the
text (possibly empty or partial) is scored as it is and counted. An `OutputKey` is a direction and
a row id of the suite.

The outputs file has one object per request with the keys `direction`, `id`, `status` and `text`
(`status` may be left out and then reads `ok`), sorted keys, text written as it is, one LF per
line. It is read against a suite: the directions PRESENT in the file must each be complete (every
row of the direction exactly once), a direction the file lacks is simply not in it (a system may
cover some directions), and what is returned follows the suite's directions and rows whatever the
order of the lines. A refusal is a `DataError` that names the line and, when they are well formed,
the direction and the id, and never what the file held. A file is at most 128 MiB, which is
checked before it is parsed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, get_args

from rinchen.errors import DataError
from rinchen.eval.robust_suite_rules import JsonObject, parse_object
from rinchen.eval.suite_rules import is_row_id
from rinchen.eval.suites import Suite
from rinchen.infer.engine_types import Status
from rinchen.utils.part_file import replace_file

__all__ = [
    "MAX_OUTPUTS_BYTES",
    "Output",
    "OutputKey",
    "direction_place",
    "output_place",
    "outputs_from_bytes",
    "outputs_to_bytes",
    "write_outputs",
]

MAX_OUTPUTS_BYTES: Final = 128 * 1024 * 1024
_STATUSES: Final = get_args(Status)
_FIELDS: Final = ("direction", "id", "status", "text")
_REQUIRED: Final = ("direction", "id", "text")
_DIRECTION_SHAPE: Final = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}-[a-z]{3}_[A-Z][a-z]{3}")


class OutputKey(NamedTuple):
    """Which output: the direction of the suite and the id of the row."""

    direction: str
    id: str


@dataclass(frozen=True, slots=True)
class Output:
    """What a system wrote for a row: its `text` and the engine's `status`.

    Raises:
        TypeError: `text` is not a string.
        ValueError: `status` is not one of the five of the engine.
    """

    text: str
    status: Status

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(
                f"Output.text must be a string, got {type(self.text).__name__}"
            )
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {', '.join(_STATUSES)}")


def direction_place(direction: object) -> str:
    """The direction named for a message, only when it has the shape of one (two codes of
    three letters, an underscore and a script of four, joined by a hyphen): what a file holds
    in any other shape is never quoted."""
    return _named_direction(direction) or "a direction that is not well formed"


def output_place(direction: object, row_id: object) -> str:
    """The output named by its parts, for a message. A part is named only when it is well
    formed (a direction of the format, a row id that matched its pattern). No part well
    formed: `a row`."""
    named = [
        _named_direction(direction),
        f"id {row_id!r}" if is_row_id(row_id) else None,
    ]
    return ", ".join(part for part in named if part is not None) or "a row"


def outputs_to_bytes(outputs: Mapping[OutputKey, Output]) -> bytes:
    """The UTF-8 bytes of the outputs file: one line each, in the order of the mapping.

    Raises:
        DataError: a text holds a lone surrogate.
    """
    chunks: list[bytes] = []
    for key, output in outputs.items():
        document = {
            "direction": key.direction,
            "id": key.id,
            "status": output.status,
            "text": output.text,
        }
        line = json.dumps(document, sort_keys=True, ensure_ascii=False)
        try:
            chunks.append(f"{line}\n".encode())
        except UnicodeEncodeError:
            raise _refused(
                output_place(key.direction, key.id), "a field holds a lone surrogate"
            ) from None
    return b"".join(chunks)


def write_outputs(outputs: Mapping[OutputKey, Output], path: Path) -> str:
    """Write `outputs` to `path`, in the order of the mapping, so that the file appears
    whole; return the SHA-256 of the bytes written.

    Raises:
        DataError: a text holds a lone surrogate, or the file would be larger than the
            reader accepts.
    """
    data = outputs_to_bytes(outputs)
    if len(data) > MAX_OUTPUTS_BYTES:
        raise _too_large()
    replace_file(path, data)
    return hashlib.sha256(data).hexdigest()


def outputs_from_bytes(data: bytes, suite: Suite) -> dict[OutputKey, Output]:
    """The outputs in `data`, a JSON Lines document, against `suite`: every line a row of a
    declared direction, each once, every direction that appears complete. They are returned in
    the order of the suite's directions and rows whatever the order of the lines.

    Raises:
        DataError: `data` is too large, not UTF-8, or not outputs of `suite`; the message
            names the line and the direction and id when they are well formed, never a text.
    """
    if len(data) > MAX_OUTPUTS_BYTES:
        raise _too_large()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _refused("the file", "the text is not UTF-8") from None
    rows = _RowIds(suite)
    found: dict[OutputKey, Output] = {}
    for number, line in enumerate(_lines(text), start=1):
        key, output = _read_line(number, line)
        place = f"line {number} ({output_place(key.direction, key.id)})"
        if not rows.declares(key.direction):
            raise _refused(place, "is not a direction of the suite")
        if key.id not in rows.of(key.direction).known:
            raise _refused(place, "is not a row of that direction")
        if key in found:
            raise _refused(place, "answers a row that is answered more than once")
        found[key] = output
    return _complete(found, suite, rows)


class _Ids(NamedTuple):
    """The ids of the rows of one direction: in file order, and as a set."""

    ordered: tuple[str, ...]
    known: frozenset[str]


class _RowIds:
    """The ids of the rows of each declared direction of a suite, looked up once."""

    def __init__(self, suite: Suite) -> None:
        self._suite = suite
        self._cache: dict[str, _Ids] = {}

    def declares(self, direction: str) -> bool:
        return direction in self._suite.directions

    def of(self, direction: str) -> _Ids:
        """The ids of a direction the suite declares."""
        if direction not in self._cache:
            ids = self._suite.view(direction).ids
            self._cache[direction] = _Ids(ids, frozenset(ids))
        return self._cache[direction]


def _complete(
    found: dict[OutputKey, Output], suite: Suite, rows: _RowIds
) -> dict[OutputKey, Output]:
    """`found` in the order of the suite, after checking that every direction that appears has
    every row of the suite."""
    present = {key.direction for key in found}
    ordered: dict[OutputKey, Output] = {}
    for direction in suite.directions:
        if direction not in present:
            continue
        keys = [OutputKey(direction, row_id) for row_id in rows.of(direction).ordered]
        missing = [key for key in keys if key not in found]
        if missing:
            raise _refused(
                direction_place(direction),
                f"lacks {len(missing)} of {len(keys)} rows, the first is "
                f"{output_place(*missing[0])}",
            )
        ordered.update((key, found[key]) for key in keys)
    return ordered


def _named_direction(direction: object) -> str | None:
    if isinstance(direction, str) and _DIRECTION_SHAPE.fullmatch(direction):
        return f"direction {direction!r}"
    return None


def _lines(text: str) -> list[str]:
    """The lines of `text`: split at LF only (a JSON string may hold any other separator),
    without the empty piece after the final LF."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _too_large() -> DataError:
    return DataError(
        f"not an outputs file: the file is larger than {MAX_OUTPUTS_BYTES // 2**20} MiB"
    )


def _refused(place: str, problem: str) -> DataError:
    """The error for a file that breaks a rule. `place` is a line number and an output named
    by `output_place`, or a direction named by `direction_place`: never a text or a key of
    the file."""
    return DataError(f"not an outputs file: {place}: {problem}")


def _read_line(number: int, line: str) -> tuple[OutputKey, Output]:
    """The row a line answers and what it says, checked field by field."""
    where = f"line {number}"
    try:
        parsed = json.loads(line, object_pairs_hook=parse_object)
    except ValueError, RecursionError:
        raise _refused(where, "is not valid JSON") from None
    if not isinstance(parsed, JsonObject):
        raise _refused(where, "is not a JSON object")
    place = f"{where} ({output_place(parsed.get('direction'), parsed.get('id'))})"
    if parsed.repeated:
        raise _refused(place, "a field appears more than once")
    if set(parsed) - set(_FIELDS):
        raise _refused(place, "has a field that is not part of the format")
    for field in _REQUIRED:
        if field not in parsed:
            raise _refused(place, f"field {field!r} is missing")
    for field in _REQUIRED:
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
    return OutputKey(parsed["direction"], parsed["id"]), Output(parsed["text"], status)
