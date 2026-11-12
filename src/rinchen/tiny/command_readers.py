"""The readers of what the CLI stages of the tiny pipeline's commands wrote (D15, spec P3).

The results of a command are read from the files it wrote and never from what it printed. The one
exception is `convert-script eval`, which prints five lines and writes nothing: `read_converter_eval`
takes them by shape, the exact five in order, and refuses anything else. A refusal is a `DataError` that
names the file and the field and quotes no value. A file that says more than a reader needs is still read:
a reader checks the keys it hands on.

`rinchen.tiny.commands` re-exports every reader, which is where the stages and the tests take them from.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from rinchen.errors import DataError

_PERCENT_MAX: Final = 100.0


def _one_of_each_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("a key is repeated")
        document[key] = value
    return document


def _not_a_number(constant: str) -> Any:
    raise ValueError("NaN and infinity are not JSON")


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in the file `path`.

    Raises:
        DataError: the file cannot be read, is not UTF-8, is not JSON (a repeated key, NaN and infinity
            are refused) or is not one object; the message names the file and quotes nothing.
    """
    name = path.name
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DataError(f"{name}: the file is missing or cannot be read") from error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"{name}: not valid UTF-8") from error
    try:
        document = json.loads(
            text, object_pairs_hook=_one_of_each_key, parse_constant=_not_a_number
        )
    except (ValueError, RecursionError) as error:
        raise DataError(f"{name}: not valid JSON") from error
    if not isinstance(document, dict):
        raise DataError(f"{name}: not a JSON object")
    return document


def _is_number(value: object, low: float, high: float) -> bool:
    """A real number (a boolean is not one) that is finite and between `low` and `high`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and low <= value <= high


def _is_count(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _check(name: str, key: str, accepted: bool, what: str) -> None:
    if not accepted:
        raise DataError(f"{name}: {key} is {what}")


_PARITY_BOOLEANS: Final = ("passes", "gated")
_PARITY_SHARES: Final = ("exact_share", "min_exact_share")
_PARITY_DELTAS: Final = ("mean_abs_delta", "max_abs_delta", "max_mean_delta")
_PARITY_TEXTS: Final = ("quantization", "placement")


def read_parity(path: Path) -> dict[str, Any]:
    """The `parity.json` of `rinchen export parity`, whole, after its figures were checked.

    Raises:
        DataError: the file is not the command's: a key the report needs is missing or of the wrong
            kind, or more sentences are exact than were compared.
    """
    name = path.name
    document = read_json(path)
    needed = (
        *_PARITY_BOOLEANS,
        "exact",
        "sentences",
        *_PARITY_SHARES,
        *_PARITY_DELTAS,
        *_PARITY_TEXTS,
    )
    for key in needed:
        _check(name, key, key in document, "missing")
    for key in _PARITY_BOOLEANS:
        _check(name, key, type(document[key]) is bool, "not true or false")
    _check(name, "exact", _is_count(document["exact"]), "not a count")
    _check(
        name, "sentences", _is_count(document["sentences"], 1), "not a positive count"
    )
    _check(
        name, "exact", document["exact"] <= document["sentences"], "more than sentences"
    )
    for key in _PARITY_SHARES:
        _check(
            name, key, _is_number(document[key], 0.0, 1.0), "not a share from 0 to 1"
        )
    for key in _PARITY_DELTAS:
        _check(
            name,
            key,
            _is_number(document[key], 0.0, math.inf),
            "not a number of 0 or more",
        )
    for key in _PARITY_TEXTS:
        value = document[key]
        _check(name, key, isinstance(value, str) and bool(value), "not a text")
    return document


_EVAL_LINES: Final = (
    ("cer", "[0-9]+[.][0-9]{4}"),
    ("word_accuracy", "[0-9]+[.][0-9]{4}"),
    ("n", "[0-9]+"),
    ("rule_fallbacks", "[0-9]+"),
    ("invalid_outputs", "[0-9]+"),
)


def read_converter_eval(log_text: str) -> dict[str, float]:
    """The five numbers `convert-script eval` prints, in the order it prints them: `cer` and
    `word_accuracy` with four decimals, then `n`, `rule_fallbacks` and `invalid_outputs` as whole
    numbers, each line `<key> <number>` with one space and nothing else, one final line feed at most.

    Raises:
        DataError: the text is anything else; the message names the line's key and quotes no text.
    """
    lines = log_text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if len(lines) != len(_EVAL_LINES):
        raise DataError(
            f"convert-script eval: {len(lines)} lines, not the five it prints"
        )
    values: dict[str, float] = {}
    for line, (key, number) in zip(lines, _EVAL_LINES, strict=True):
        found = re.fullmatch(f"{key} ({number})", line)
        if found is None:
            raise DataError(f"convert-script eval: the line of {key} is not as printed")
        values[key] = float(found.group(1))
    return values


_METRICS: Final = ("bleu", "chrf", "chrf_pp", "comet22", "metricx24", "metricx24_qe")
_SURFACE_METRICS: Final = ("bleu", "chrf", "chrf_pp")


def read_result_metrics(path: Path) -> dict[str, float | None]:
    """The six metrics of a `result.json` of `rinchen eval run`: BLEU, chrF and chrF++ as numbers from
    0 to 100, and the three neural ones as a finite number or `None` when none was scored.

    Raises:
        DataError: `metrics` is not exactly those six, or a value is not as described.
    """
    name = path.name
    metrics = read_json(path).get("metrics")
    if not isinstance(metrics, dict):
        raise DataError(f"{name}: metrics is missing or not an object")
    if missing := [key for key in _METRICS if key not in metrics]:
        raise DataError(f"{name}: metrics lacks {', '.join(missing)}")
    if len(metrics) != len(_METRICS):
        raise DataError(f"{name}: metrics holds a key that is none of the six")
    read: dict[str, float | None] = {}
    for key in _METRICS:
        value = metrics[key]
        if value is None and key not in _SURFACE_METRICS:
            read[key] = None
            continue
        high = _PERCENT_MAX if key in _SURFACE_METRICS else math.inf
        low = 0.0 if key in _SURFACE_METRICS else -math.inf
        _check(name, key, _is_number(value, low, high), "not a number in its range")
        read[key] = float(value)
    return read


_COUNT_KEYS: Final = (
    "candidates",
    "dropped_copy",
    "dropped_empty",
    "dropped_purity",
    "dropped_qe",
    "no_candidates",
    "no_survivor",
    "selected",
    "sources",
)
_SHARD_KEYS: Final = ("generated", "planned", "regenerated_invalid", "reused")


def _counts(
    name: str, document: Mapping[str, Any], key: str, keys: Sequence[str]
) -> None:
    table = document.get(key)
    if not isinstance(table, dict):
        raise DataError(f"{name}: {key} is missing or not an object")
    for item in keys:
        _check(name, f"{key}.{item}", item in table, "missing")
        _check(name, f"{key}.{item}", _is_count(table[item]), "not a count")


def read_synth_counts(path: Path) -> dict[str, Any]:
    """The `counts.json` of `rinchen synth run`, whole, after its shape was checked: the counts of the
    run and of its shards, the job, the generator and the status of the quality floor.

    Raises:
        DataError: a key is missing or of the wrong kind, or more rows were selected than there are
            sources.
    """
    name = path.name
    document = read_json(path)
    _counts(name, document, "counts", _COUNT_KEYS)
    _counts(name, document, "shard_stats", _SHARD_KEYS)
    for key in ("qe_status", "job", "generator"):
        value = document.get(key)
        _check(
            name, key, isinstance(value, str) and bool(value), "missing or not a text"
        )
    counts = document["counts"]
    _check(
        name,
        "counts.selected",
        counts["selected"] <= counts["sources"],
        "more than the sources",
    )
    return document
