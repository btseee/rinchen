"""The readers of what the evaluation commands of the tiny pipeline write (D15, spec P3 and P10).

A stage reads the results of a command from the files it wrote, never from what it printed. Three
files are read here: the `report.json` of `rinchen eval robustness` (the six values and the counts
each rests on), the `result.json` of one direction of one `rinchen eval run` (its rows, its six metrics
and the statuses of its outputs) and the `hypotheses.jsonl` of a run (what the system wrote for each
row). Each reader checks the shape of the file and refuses anything else with a `DataError` that names
the file and the field or the line and quotes no value: a text of a fixture or of a translation is never
in a message.

This module reads files of D12 and D13 by their own shapes, written again here so that the library of
the pipeline imports nothing of `rinchen.eval`: tests keep the statuses equal to the engine's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, NamedTuple

from rinchen.errors import DataError
from rinchen.tiny import commands

STATUSES: Final = ("ok", "empty", "too_long", "truncated", "no_output")
ROBUSTNESS_SCHEMA: Final = "rinchen.report.robustness/1"
ROBUSTNESS_VALUES: Final = (
    "entity_acc",
    "mong_source_retention",
    "numbers_acc",
    "placeholder_acc",
    "script_purity",
    "term_acc",
)
# The one value that is a ratio of two chrF scores and may exceed 1; the others are shares.
_RATIO_VALUE: Final = "mong_source_retention"
_COUNTS: Final = ("scored", "total", "excluded")
_HYPOTHESIS_KEYS: Final = ("direction", "id", "status", "text")


class Hypothesis(NamedTuple):
    """What a system wrote for a row of a direction, and the status the engine gave it."""

    direction: str
    id: str
    status: str
    text: str


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require(name: str, accepted: bool, what: str) -> None:
    if not accepted:
        raise DataError(f"{name}: {what}")


def _values(name: str, raw: object) -> dict[str, float | None]:
    _require(
        name,
        isinstance(raw, dict) and sorted(raw) == sorted(ROBUSTNESS_VALUES),
        "robustness is not exactly the six values",
    )
    assert isinstance(raw, dict)
    values: dict[str, float | None] = {}
    for key in ROBUSTNESS_VALUES:
        value = raw[key]
        if value is None:
            values[key] = None
            continue
        high = math.inf if key == _RATIO_VALUE else 1.0
        _require(
            name,
            _is_number(value) and 0.0 <= value <= high,
            f"robustness.{key} is not a number in its range",
        )
        values[key] = float(value)
    return values


def _coverage(name: str, raw: object) -> dict[str, dict[str, Any]]:
    _require(
        name,
        isinstance(raw, dict) and sorted(raw) == sorted(ROBUSTNESS_VALUES),
        "coverage is not exactly the six values",
    )
    assert isinstance(raw, dict)
    for key in ROBUSTNESS_VALUES:
        entry = raw[key]
        _require(
            name,
            isinstance(entry, dict)
            and all(_is_count(entry.get(count)) for count in _COUNTS)
            and entry["scored"] <= entry["total"],
            f"coverage.{key} does not give the counts it should",
        )
    return {key: dict(raw[key]) for key in ROBUSTNESS_VALUES}


def read_robustness_report(path: Path) -> dict[str, Any]:
    """The `values` and the `coverage` of the `report.json` `rinchen eval robustness` wrote.

    Raises:
        DataError: the file is missing or is not that report: another schema, other than the six
            values (each a number in its range or null) or the six counts beside them.
    """
    name = path.name
    document = commands.read_json(path)
    _require(
        name, document.get("schema") == ROBUSTNESS_SCHEMA, "not a robustness report"
    )
    return {
        "values": _values(name, document.get("robustness")),
        "coverage": _coverage(name, document.get("coverage")),
    }


def _statuses(name: str, raw: object, rows: int) -> dict[str, int]:
    _require(
        name,
        isinstance(raw, dict)
        and bool(raw)
        and set(raw) <= set(STATUSES)
        and all(_is_count(count) for count in raw.values()),
        "statuses is not a table of counts by status",
    )
    assert isinstance(raw, dict)
    _require(name, sum(raw.values()) == rows, "statuses do not add up to the rows")
    return {status: raw[status] for status in STATUSES if status in raw}


def read_run_result(
    path: Path, *, suite: str, system: str, direction: str
) -> dict[str, Any]:
    """The `rows`, the six `metrics` and the `statuses` of the `result.json` of `direction` that
    `rinchen eval run` wrote for `system` over `suite`.

    Raises:
        DataError: the file is missing or is not that result: of another suite, system or direction,
            without a positive count of rows that the statuses add up to, or without the six
            metrics (see `commands.read_result_metrics`).
    """
    name = path.name
    metrics = commands.read_result_metrics(path)
    document = commands.read_json(path)
    for key, expected in (
        ("suite", suite),
        ("system", system),
        ("direction", direction),
    ):
        _require(name, document.get(key) == expected, f"{key} is not the one asked for")
    rows = document.get("n_segments")
    _require(
        name,
        type(rows) is int and rows > 0,
        "n_segments is not a positive count",
    )
    assert isinstance(rows, int)
    return {
        "rows": rows,
        "metrics": metrics,
        "statuses": _statuses(name, document.get("statuses"), rows),
    }


def _no_repeated_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise ValueError("a key is repeated")
        found[key] = value
    return found


def _hypothesis(name: str, number: int, line: str) -> Hypothesis:
    place = f"{name}: line {number}"
    try:
        row = json.loads(line, object_pairs_hook=_no_repeated_key)
    except (ValueError, RecursionError) as error:
        raise DataError(f"{place}: not valid JSON") from error
    if not isinstance(row, dict) or sorted(row) != sorted(_HYPOTHESIS_KEYS):
        raise DataError(f"{place}: not an object of exactly the four fields")
    if not all(isinstance(row[key], str) for key in _HYPOTHESIS_KEYS):
        raise DataError(f"{place}: a field is not text")
    if row["status"] not in STATUSES:
        raise DataError(f"{place}: status is none of the five of the engine")
    return Hypothesis(row["direction"], row["id"], row["status"], row["text"])


def read_hypotheses(path: Path) -> list[Hypothesis]:
    """The rows of the `hypotheses.jsonl` of a run, in the order of the file, each `(direction, id)`
    once.

    Raises:
        DataError: the file is missing, empty, not UTF-8, holds a carriage return or a blank line, a
            line that is not an object of the four fields of text with a status of the engine, or a
            row twice; the message names the line.
    """
    name = path.name
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DataError(f"{name}: the file is missing or cannot be read") from error
    if not text.endswith("\n") or "\r" in text:
        raise DataError(
            f"{name}: lines must end with a line feed and hold no carriage return"
        )
    rows: list[Hypothesis] = []
    seen: set[tuple[str, str]] = set()
    for number, line in enumerate(text[:-1].split("\n"), start=1):
        row = _hypothesis(name, number, line)
        if (row.direction, row.id) in seen:
            raise DataError(f"{name}: line {number}: a row is answered twice")
        seen.add((row.direction, row.id))
        rows.append(row)
    return rows
