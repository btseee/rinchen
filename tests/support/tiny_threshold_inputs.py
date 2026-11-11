"""The inputs of the tiny pipeline's threshold tests (D15): good stage records and figures, and the
helpers that change one of them and read what the thresholds answer.

The tests of `rinchen.tiny.thresholds` (`tests/unit/test_tiny_thresholds.py` and
`test_tiny_thresholds_missing.py`) start from inputs that pass every threshold and change one figure
at a time, so that each verdict is a test of that figure alone.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

from rinchen.config.tiny import STAGE_IDS, ThresholdsSection
from rinchen.tiny.states import SKIP_NO_LID, SKIP_NO_QE, PartRecord, StageRecord
from rinchen.tiny.thresholds import (
    GATED_SYSTEM,
    MEMORISED_DIRECTIONS,
    REFERENCE_SYSTEM,
    ThresholdInputs,
    ThresholdResult,
    evaluate_thresholds,
)

BUDGET: Final = 1200
LIMITS: Final = ThresholdsSection()
FIRST: Final = "eng_Latn-khk_Cyrl"
LAST: Final = "mn_Mong-eng_Latn"
SENTINEL: Final = "ZZ-planted-value"


def good_stages() -> list[StageRecord]:
    """The fifteen stages, all ok; the filter skips two parts on purpose and the evaluation one."""
    records = []
    for stage in STAGE_IDS:
        if stage == "filter":
            parts = (
                PartRecord("script_ratio", "ok"),
                PartRecord("lid", "skipped", SKIP_NO_LID),
                PartRecord("qe", "skipped", SKIP_NO_QE),
            )
            records.append(StageRecord(stage, "ok", parts=parts))
        else:
            records.append(StageRecord(stage, "ok"))
    return records


def chrf_table(
    int8: float = 99.5, float32: float = 99.5, **changes: tuple[float, float]
) -> dict[str, dict[str, float]]:
    """Every memorised direction at the two values; `changes` maps a direction to `(int8, float32)`."""
    table = {
        direction: {GATED_SYSTEM: int8, REFERENCE_SYSTEM: float32}
        for direction in MEMORISED_DIRECTIONS
    }
    for direction, (low, high) in changes.items():
        table[direction] = {GATED_SYSTEM: low, REFERENCE_SYSTEM: high}
    return table


GOOD_PARITY: Final = {"exact_share": 1.0, "mean_abs_delta": 1.4e-06}
GOOD_ROBUSTNESS: Final = {
    "numbers_acc": 1.0,
    "placeholder_acc": 1.0,
    "entity_acc": 1.0,
    "term_acc": 1.0,
    "script_purity": 1.0,
    "mong_source_retention": 0.9976,
}


def good_inputs() -> ThresholdInputs:
    return ThresholdInputs(
        stages=good_stages(),
        memorised_chrf=chrf_table(),
        parity=dict(GOOD_PARITY),
        robustness=dict(GOOD_ROBUSTNESS),
        elapsed_seconds=123.4,
    )


def with_changes(**changes: Any) -> ThresholdInputs:
    return dataclasses.replace(good_inputs(), **changes)


def results_of(inputs: ThresholdInputs) -> dict[str, ThresholdResult]:
    results = evaluate_thresholds(inputs, LIMITS, BUDGET)
    return {result.id: result for result in results}


def one(threshold_id: str, inputs: ThresholdInputs) -> ThresholdResult:
    return results_of(inputs)[threshold_id]


def robustness_with(**changes: Any) -> dict[str, Any]:
    return {**GOOD_ROBUSTNESS, **changes}


def parity_with(**changes: Any) -> dict[str, Any]:
    return {**GOOD_PARITY, **changes}


def only_failing(inputs: ThresholdInputs) -> set[str]:
    return {
        r.id for r in evaluate_thresholds(inputs, LIMITS, BUDGET) if r.verdict == "fail"
    }


def memorised_without(direction: str, system: str) -> dict[str, dict[str, float]]:
    table = chrf_table()
    del table[direction][system]
    return table


def memorised_with(
    direction: str, system: str, value: Any
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = chrf_table()  # type: ignore[assignment]
    table[direction][system] = value
    return table
