"""The scores file of D13 (spec T8): the seam through which per-segment scores of COMET-22,
MetricX-24 and MetricX-24-QE enter. D13 builds no neural scorer (the checkpoints cannot be
fetched here and their internals are unverified: B1 and B2 stay open); scores produced elsewhere
are handed in as a JSON Lines file:

    {"schema": "rinchen.eval.scores/1", "metric": ..., "suite_sha256": ..., "hypotheses_sha256": ...}
    {"direction": ..., "id": ..., "score": ...}   one per segment the run scored

The header must name the suite and the hypotheses file of THIS run (a score of other hypotheses
says nothing about these), every segment the run scored needs exactly one finite score of at most
1000 in magnitude, and a metric given for some segments or some directions only is refused.
Nothing here says whether a score is right. A file is at most 128 MiB, checked before it is
parsed. A refusal is a `DataError` that names the line and, when they are well formed, the
direction and the id, and never what the file held.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Final

from rinchen.errors import DataError
from rinchen.eval.outputs import OutputKey, output_place
from rinchen.eval.robust_suite_rules import JsonObject, parse_object
from rinchen.metrics.registry import SEGMENT_SCORE_METRICS

__all__ = ["MAX_SCORES_BYTES", "SCORES_SCHEMA", "read_scores", "scores_to_bytes"]

SCORES_SCHEMA: Final = "rinchen.eval.scores/1"
MAX_SCORES_BYTES: Final = 128 * 2**20
# The largest score magnitude a scorer of the seam may give (COMET and MetricX are a few units).
_MAX_ABS_SCORE: Final = 1000
_HEADER_FIELDS: Final = ("schema", "metric", "suite_sha256", "hypotheses_sha256")
_LINE_FIELDS: Final = ("direction", "id", "score")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


def scores_to_bytes(
    metric: str,
    suite_sha256: str,
    hypotheses_sha256: str,
    scores: Mapping[OutputKey, float],
) -> bytes:
    """The bytes of a scores file: the header and one line per score in the order of `scores`,
    sorted keys, text as it is, one LF per line. For tests and for external writers.

    Raises:
        ValueError: `metric` takes no segment scores, a digest is not 64 lowercase hexadecimal
            characters, or a score is not a finite number of at most 1000 in magnitude.
    """
    _require_arguments(metric, suite_sha256, hypotheses_sha256)
    header = {
        "schema": SCORES_SCHEMA,
        "metric": metric,
        "suite_sha256": suite_sha256,
        "hypotheses_sha256": hypotheses_sha256,
    }
    lines = [_dump(header)]
    for key, score in scores.items():
        if not _is_score(score):
            raise ValueError(
                f"a score must be a finite number of at most {_MAX_ABS_SCORE} in magnitude"
            )
        lines.append(_dump({"direction": key.direction, "id": key.id, "score": score}))
    return "".join(f"{line}\n" for line in lines).encode()


def read_scores(
    data: bytes,
    *,
    metric: str,
    suite_sha256: str,
    hypotheses_sha256: str,
    expected: Sequence[OutputKey],
) -> dict[OutputKey, float]:
    """The scores of `metric` in `data` for exactly the segments in `expected` (the outputs of
    the directions the run scored), in the order of `expected`.

    Raises:
        ValueError: `metric` takes no segment scores, or a digest is not 64 lowercase
            hexadecimal characters.
        DataError: the file is too large, not UTF-8 or not JSON Lines, its header does not name
            this metric, suite and hypotheses, or a segment has no score, two, or a score that
            is not a finite number of at most 1000 in magnitude; or a score is of no expected
            segment.
    """
    _require_arguments(metric, suite_sha256, hypotheses_sha256)
    if len(data) > MAX_SCORES_BYTES:
        raise _refused("the file", f"is larger than {MAX_SCORES_BYTES // 2**20} MiB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _refused("the file", "the text is not UTF-8") from None
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines:
        raise _refused("the file", "is empty: the header is missing")
    _check_header(_object(1, lines[0]), metric, suite_sha256, hypotheses_sha256)
    wanted = frozenset(expected)
    found: dict[OutputKey, float] = {}
    for number, line in enumerate(lines[1:], start=2):
        key, score = _score_line(number, line)
        place = f"line {number} ({output_place(key.direction, key.id)})"
        if key not in wanted:
            raise _refused(place, "is not a segment the run scored")
        if key in found:
            raise _refused(place, "scores a segment that is scored more than once")
        found[key] = score
    missing = [key for key in expected if key not in found]
    if missing:
        raise _refused(
            "the file",
            f"lacks {len(missing)} of {len(wanted)} scores, the first is "
            f"{output_place(*missing[0])}",
        )
    return {key: found[key] for key in expected}


def _require_arguments(metric: str, suite_sha256: str, hypotheses_sha256: str) -> None:
    if metric not in SEGMENT_SCORE_METRICS:
        raise ValueError(
            "that metric takes no segment scores: they are for "
            f"{', '.join(SEGMENT_SCORE_METRICS)}"
        )
    for name, digest in (
        ("suite_sha256", suite_sha256),
        ("hypotheses_sha256", hypotheses_sha256),
    ):
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def _is_score(value: object) -> bool:
    """A number (a boolean is not one) of at most `_MAX_ABS_SCORE` in magnitude: NaN and the
    infinities fail the comparison."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return abs(value) <= _MAX_ABS_SCORE


def _dump(document: Mapping[str, object]) -> str:
    return json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _refused(place: str, problem: str) -> DataError:
    """The error for a file that breaks a rule. `place` is a line number and a segment named
    by `output_place`, or `the file`: never a text or a key of the file."""
    return DataError(f"not a scores file: {place}: {problem}")


def _object(number: int, line: str) -> JsonObject:
    place = f"line {number}"
    try:
        parsed = json.loads(line, object_pairs_hook=parse_object)
    except ValueError, RecursionError:
        raise _refused(place, "is not valid JSON") from None
    if not isinstance(parsed, JsonObject):
        raise _refused(place, "is not a JSON object")
    if parsed.repeated:
        raise _refused(place, "a field appears more than once")
    return parsed


def _check_header(
    header: JsonObject, metric: str, suite_sha256: str, hypotheses_sha256: str
) -> None:
    place = "line 1 (the header)"
    for field in _HEADER_FIELDS:
        if field not in header:
            raise _refused(place, f"field {field!r} is missing")
    if len(header) != len(_HEADER_FIELDS):
        raise _refused(place, "has a field that is not part of the format")
    if header["schema"] != SCORES_SCHEMA:
        raise _refused(place, "field 'schema' is not the first version of the format")
    for field, own in (
        ("metric", metric),
        ("suite_sha256", suite_sha256),
        ("hypotheses_sha256", hypotheses_sha256),
    ):
        if header[field] != own:
            raise _refused(
                place, f"field {field!r} names another {_WHAT[field]} than this run's"
            )


_WHAT: Final = {
    "metric": "metric",
    "suite_sha256": "suite",
    "hypotheses_sha256": "hypotheses file",
}


def _score_line(number: int, line: str) -> tuple[OutputKey, float]:
    parsed = _object(number, line)
    place = f"line {number} ({output_place(parsed.get('direction'), parsed.get('id'))})"
    for field in _LINE_FIELDS:
        if field not in parsed:
            raise _refused(place, f"field {field!r} is missing")
    if len(parsed) != len(_LINE_FIELDS):
        raise _refused(place, "has a field that is not part of the format")
    for field in ("direction", "id"):
        if not isinstance(parsed[field], str):
            raise _refused(place, f"field {field!r} must be a string")
    score = parsed["score"]
    if not _is_score(score):
        raise _refused(
            place,
            f"field 'score' must be a finite number of at most {_MAX_ABS_SCORE} in "
            "magnitude",
        )
    return OutputKey(parsed["direction"], parsed["id"]), float(score)
