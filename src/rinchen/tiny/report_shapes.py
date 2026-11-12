"""The closed shapes of the tiny pipeline's report (D15, spec P10): what a plain JSON value may be.

A shape is checked against a value of plain JSON types (a `dict`, a `list`, a `str`, an `int`, a
`float`, a `bool` or `None`) at a place, and a refusal is a `ShapeError` whose message is the place and
what is wrong there. The place is built from the names of the schema and from positions in a list; a key
that a document brought is counted and never quoted, and so is every text: a report is read by people and
by CI, and nothing it is handed may come back out in an error.

Every text leaf is held to the same rules, so that no shape lets a path, a time of day, a date, a
control character or a line of another program's output through: printable ASCII only, at most
`MAX_TEXT` characters, no slash and no backslash, and nothing shaped like a date or a time of day. The
one text a shape takes without those rules is the one it is told to expect (`equals`: the schema name
holds a slash).

A `Detail` is the free part of the report, the detail of a stage: any nesting of those types up to
`MAX_DEPTH`, whose keys are short names that name no duration or date.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

MAX_TEXT: Final = 512
MAX_DEPTH: Final = 8

_PRINTABLE: Final = re.compile(r"[ -~]*")
_PATH_CHARACTERS: Final = re.compile(r"[/\\]")
# A date (`2026-11-11`) or a time of day (`12:30`, also inside `T12:30:00Z`).
_WHEN: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}:[0-9]{2}")
_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_KEY: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
# A key of a detail that says how long something took or when it happened.
_CLOCK_KEY: Final = re.compile(
    r"(?:^|_)(?:seconds?|ms|duration|elapsed|time|timestamp|date)(?:_|$)",
    re.IGNORECASE,
)


class ShapeError(ValueError):
    """A value that is not of its shape: the message is the place, a colon and what is wrong."""

    def __init__(self, place: str, what: str) -> None:
        super().__init__(f"{place}: {what}")


class Shape(Protocol):
    """What the document is checked against: raises `ShapeError` for a value that does not fit."""

    def check(self, value: object, place: str) -> None: ...


def text_problem(value: str) -> str | None:
    """What is wrong with a text for the report, or `None`: it must be printable ASCII of at most
    `MAX_TEXT` characters, with no slash, no backslash, no date and no time of day."""
    if _PRINTABLE.fullmatch(value) is None:
        return "is not printable ASCII"
    if len(value) > MAX_TEXT:
        return f"is longer than {MAX_TEXT} characters"
    if _PATH_CHARACTERS.search(value):
        return "looks like a path"
    if _WHEN.search(value):
        return "looks like a date or a time of day"
    return None


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True, slots=True)
class Leaf:
    """A scalar shape: `what` completes "is not ..." in the message, `accepts` decides, and a text
    that is accepted is still held to the text rules unless `free_text` says it is expected."""

    what: str
    accepts: Callable[[object], bool]
    free_text: bool = False

    def check(self, value: object, place: str) -> None:
        if not self.accepts(value):
            raise ShapeError(place, f"is not {self.what}")
        if isinstance(value, str) and not self.free_text:
            problem = text_problem(value)
            if problem is not None:
                raise ShapeError(place, f"is a text that {problem}")


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _in_range(low: float, high: float) -> Callable[[object], bool]:
    def accepts(value: object) -> bool:
        return (
            isinstance(value, int | float)
            and _is_number(value)
            and low <= value <= high
        )

    return accepts


def _matches(pattern: re.Pattern[str]) -> Callable[[object], bool]:
    return lambda value: isinstance(value, str) and pattern.fullmatch(value) is not None


COUNT: Final = Leaf("a count, a whole number of 0 or more", _count)
NUMBER: Final = Leaf("a finite number", _is_number)
NONNEG: Final = Leaf("a finite number of 0 or more", _in_range(0.0, math.inf))
SHARE: Final = Leaf("a share, a number from 0 to 1", _in_range(0.0, 1.0))
SCORE: Final = Leaf("a score, a number from 0 to 100", _in_range(0.0, 100.0))
BOOL: Final = Leaf("true or false", lambda value: isinstance(value, bool))
NAME: Final = Leaf(
    "a name of letters, digits, dots, hyphens and underscores", _matches(_NAME)
)
DIGEST: Final = Leaf(
    "a SHA-256 digest of 64 lower-case hex characters", _matches(_DIGEST)
)
TEXT: Final = Leaf("a text", lambda value: isinstance(value, str) and value != "")


def choice(*members: str) -> Leaf:
    """A text that is one of `members`; the message lists them and never the value."""
    return Leaf(f"one of {', '.join(members)}", lambda value: value in members)


def equals(expected: Any) -> Leaf:
    """Exactly `expected`, of the same type: a constant of the schema (a text is not held to the text
    rules, because it is ours)."""
    return Leaf(
        "the value the schema fixes",
        lambda value: type(value) is type(expected) and value == expected,
        free_text=True,
    )


@dataclass(frozen=True, slots=True)
class Nullable:
    """Null, or a value of `shape`."""

    shape: Shape

    def check(self, value: object, place: str) -> None:
        if value is not None:
            self.shape.check(value, place)


@dataclass(frozen=True, slots=True)
class Keys:
    """A mapping with exactly these keys, each of its own shape."""

    shapes: Mapping[str, Shape]

    def check(self, value: object, place: str) -> None:
        if not isinstance(value, dict):
            raise ShapeError(place, "is not a mapping")
        if not all(isinstance(key, str) for key in value):
            raise ShapeError(place, "has a key that is not text")
        unknown = sum(1 for key in value if key not in self.shapes)
        if unknown:
            raise ShapeError(
                place, f"has {unknown} keys that are none of the expected ones"
            )
        for key, shape in self.shapes.items():
            if key not in value:
                raise ShapeError(place, f"lacks the key {key}")
            shape.check(value[key], f"{place}.{key}")


@dataclass(frozen=True, slots=True)
class Table:
    """A mapping of at least `minimum` entries whose keys are of the shape `key` and whose values
    are of the shape `value`."""

    key: Shape
    value: Shape
    minimum: int = 0

    def check(self, value: object, place: str) -> None:
        if not isinstance(value, dict):
            raise ShapeError(place, "is not a mapping")
        if len(value) < self.minimum:
            raise ShapeError(place, f"has fewer than {self.minimum} entries")
        for key, item in value.items():
            try:
                self.key.check(key, place)
            except ShapeError as error:
                raise ShapeError(
                    place, "has a key that is not an expected name"
                ) from error
            self.value.check(item, f"{place}.{key}")


@dataclass(frozen=True, slots=True)
class Seq:
    """A list of items of one shape, of exactly `length` entries when that is given and else of at
    least `minimum`."""

    item: Shape
    length: int | None = None
    minimum: int = 0

    def check(self, value: object, place: str) -> None:
        if not isinstance(value, list):
            raise ShapeError(place, "is not a list")
        if self.length is not None and len(value) != self.length:
            raise ShapeError(place, f"has {len(value)} entries, not {self.length}")
        if len(value) < self.minimum:
            raise ShapeError(place, f"has fewer than {self.minimum} entries")
        for position, item in enumerate(value):
            self.item.check(item, f"{place}[{position}]")


@dataclass(frozen=True, slots=True)
class Detail:
    """Free JSON for the detail of a stage: nulls, booleans, finite numbers, safe texts, lists and
    mappings of those, nested at most `MAX_DEPTH` deep, whose keys are short names that say nothing
    of a duration or a date."""

    def check(self, value: object, place: str, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            raise ShapeError(place, "is nested too deeply")
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int | float):
            if not _is_number(value):
                raise ShapeError(place, "is a number that is not finite")
            return
        if isinstance(value, str):
            problem = text_problem(value)
            if problem is not None:
                raise ShapeError(place, f"is a text that {problem}")
            return
        if isinstance(value, list):
            for position, item in enumerate(value):
                self.check(item, f"{place}[{position}]", depth + 1)
            return
        if isinstance(value, dict):
            self._check_mapping(value, place, depth)
            return
        raise ShapeError(place, "holds a value that plain JSON does not have")

    def _check_mapping(self, value: dict[Any, Any], place: str, depth: int) -> None:
        for key, item in value.items():
            if not isinstance(key, str) or _KEY.fullmatch(key) is None:
                raise ShapeError(place, "has a key that is not a short name")
            if _CLOCK_KEY.search(key):
                raise ShapeError(place, "has a key that names a duration or a date")
            self.check(item, f"{place}.{key}", depth + 1)


def plain(value: Any) -> Any:
    """`value` with every mapping a `dict` and every tuple or list a `list`, as JSON has them; the
    argument is not changed."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value
