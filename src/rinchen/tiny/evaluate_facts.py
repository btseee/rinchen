"""The figures the evaluation stage reports besides the metrics of each run (D15, spec P8 and P9).

`chrf_table` reads the corpus chrF of each direction and system out of the runs' facts in the shape
the thresholds read, `drops_of` and `drop_extremes` give the drop of chrF from the reference export to the
gated one (a quantisation that loses quality is a positive drop: `reference - gated`) and its least and
largest value over the directions, and `agreement` counts the rows on which two systems wrote the same
text. Everything here works on facts the stage has already read and checked, imports nothing of
`rinchen.eval` and raises `DataError` for figures that cannot be compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rinchen.errors import DataError
from rinchen.tiny.eval_readers import Hypothesis


def chrf_table(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    """`{direction: {system: corpus chrF}}` from `{direction: {system: facts of the run}}`."""
    return {
        direction: {system: facts["metrics"]["chrf"] for system, facts in by.items()}
        for direction, by in runs.items()
    }


def drops_of(
    chrf: Mapping[str, Mapping[str, float]], *, reference: str, gated: str
) -> dict[str, float]:
    """`reference - gated` of the chrF of each direction.

    Raises:
        DataError: a direction has no figure for one of the two systems.
    """
    drops: dict[str, float] = {}
    for direction, by_system in chrf.items():
        if reference not in by_system or gated not in by_system:
            raise DataError("a direction has no chrF for one of the two systems")
        drops[direction] = by_system[reference] - by_system[gated]
    return drops


def drop_extremes(drops: Mapping[str, float]) -> dict[str, float]:
    """The `least` and the `largest` of the drops.

    Raises:
        DataError: there is no drop.
    """
    if not drops:
        raise DataError("there is no drop of chrF to compare")
    return {"least": min(drops.values()), "largest": max(drops.values())}


def agreement(
    first: Sequence[Hypothesis], second: Sequence[Hypothesis]
) -> dict[str, Any]:
    """How many of the rows both systems answered have the same text in both, by direction and in all.

    Raises:
        DataError: the two do not answer the same rows (a direction and an id), or there are none.
    """
    texts = {(row.direction, row.id): row.text for row in first}
    others = {(row.direction, row.id): row.text for row in second}
    if not texts or texts.keys() != others.keys():
        raise DataError("the two systems did not answer the same rows")
    by_direction: dict[str, dict[str, int]] = {}
    for (direction, row_id), text in texts.items():
        count = by_direction.setdefault(direction, {"rows": 0, "equal": 0})
        count["rows"] += 1
        count["equal"] += others[(direction, row_id)] == text
    rows = sum(count["rows"] for count in by_direction.values())
    equal = sum(count["equal"] for count in by_direction.values())
    return {
        "by_direction": by_direction,
        "rows": rows,
        "equal": equal,
        "share": equal / rows,
    }
