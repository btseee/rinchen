"""The thresholds of `rinchen tiny run` (D15, spec P8): eight results, each `pass` or `fail`, in the
order of the table, from the records of the stages and the figures the stages measured.

A bound compares with `>=` or `<=` exactly as the table says, so a value equal to its bound passes.
A value that is missing (a `None`, a key that is not there, a stage that never ran) or that cannot be
a measurement (not a number, not finite, outside its range) is a `fail` whose detail names it, never a
pass: a gate that cannot fail proves nothing. `train_chrf` is the least of the int8 export's corpus
chrF over the five memorised directions and nothing else, and `int8_chrf_drop` the largest difference
between the float32 and the int8 figure over the same five: the other system and a direction that is
not named never feed either. A detail names only closed identifiers (a direction, a figure's key, a
stage id) and numbers, never a key or a value that the inputs held.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from rinchen.config.tiny import STAGE_IDS, ThresholdsSection
from rinchen.tiny.states import SKIP_EARLIER_FAILED, StageRecord, is_skip_reason

THRESHOLD_IDS: Final = (
    "stages",
    "train_chrf",
    "parity_exact",
    "parity_delta",
    "int8_chrf_drop",
    "script_purity",
    "placeholder_acc",
    "wall_time",
)
# The five directions the fixtures train and the memorised suites score.
MEMORISED_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "rus_Cyrl-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
)
GATED_SYSTEM: Final = "int8"
REFERENCE_SYSTEM: Final = "float32"

Verdict = Literal["pass", "fail"]

_CHRF_RANGE: Final = (0.0, 100.0)
_SHARE_RANGE: Final = (0.0, 1.0)
_NO_UPPER_BOUND: Final = math.inf
_STAGES_BOUND: Final = "<= 0"
_STAGES_FINE: Final = "no stage failed and every skip has a reason of the closed list"


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """One line of the table: the threshold's id, its bound as a comparison (`>= 50.0`), the
    value measured (`None` when there was none), the verdict and what the value is or why it is
    missing."""

    id: str
    bound: str
    value: float | None
    verdict: Verdict
    detail: str


@dataclass(frozen=True, slots=True)
class ThresholdInputs:
    """What the thresholds read: the stage records, the memorised corpus chrF by direction and then
    by system (`int8` and `float32`), the figures of the float32 export's parity run
    (`exact_share`, `mean_abs_delta`), D12's robustness block and the seconds the run took."""

    stages: Sequence[StageRecord]
    memorised_chrf: Mapping[str, Mapping[str, float]]
    parity: Mapping[str, float] | None
    robustness: Mapping[str, float | None] | None
    elapsed_seconds: float


def _number(value: object, low: float, high: float) -> float | None:
    """`value` as a float when it is a real number (a boolean is not one) that is finite and
    within `low` and `high`, else `None`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and low <= number <= high else None


def _figure(table: object, key: str, limits: tuple[float, float]) -> float | None:
    if not isinstance(table, Mapping):
        return None
    return _number(table.get(key), *limits)


def _bound(at_least: bool, limit: float) -> str:
    return f"{'>=' if at_least else '<='} {limit!r}"


def _result(
    threshold_id: str, at_least: bool, limit: float, value: float, detail: str
) -> ThresholdResult:
    passed = value >= limit if at_least else value <= limit
    return ThresholdResult(
        threshold_id,
        _bound(at_least, limit),
        value,
        "pass" if passed else "fail",
        detail,
    )


def _missing(
    threshold_id: str, at_least: bool, limit: float, what: str
) -> ThresholdResult:
    return ThresholdResult(
        threshold_id,
        _bound(at_least, limit),
        None,
        "fail",
        f"missing or invalid: {what}",
    )


def _is_at_fault(state: object, reason: object) -> bool:
    """Whether a stage or a part that came to `state` for `reason` is a problem for the run on its
    own: a failure, a skip without one of the skip reasons, an `ok` that holds a reason, or a state
    that is none of the three. (A stage skipped as a whole also needs a failure before it:
    `_stage_faults`.)"""
    if state == "ok":
        return reason is not None
    if state == "skipped":
        return not (isinstance(reason, str) and is_skip_reason(reason))
    return True


def _record_at_fault(record: StageRecord) -> bool:
    return _is_at_fault(record.state, record.reason) or any(
        _is_at_fault(part.state, part.reason) for part in record.parts
    )


def _stage_faults(stages: Sequence[StageRecord]) -> tuple[list[str], list[str]]:
    """The stages at fault by their own record, and the stages skipped as a whole that nothing
    before them explains, each in the order of `STAGE_IDS`.

    A stage is skipped as a whole for one reason only, `an earlier stage failed`, and only when a
    stage before it failed or was itself at fault; the other skip reasons are about a part of a
    stage (the filter's LID model, the evaluation's neural scorer) and explain no stage. The pipeline
    skips a stage only after a failure, so a skip that no fault precedes is a stage that was
    dropped without a cause.
    """
    own: set[str] = set()
    unexplained: set[str] = set()
    known = [record for record in stages if record.id in STAGE_IDS]
    for record in sorted(known, key=lambda record: STAGE_IDS.index(record.id)):
        position = STAGE_IDS.index(record.id)
        if _record_at_fault(record):
            own.add(record.id)
        elif record.state == "skipped" and not (
            record.reason == SKIP_EARLIER_FAILED
            and any(STAGE_IDS.index(stage) < position for stage in own | unexplained)
        ):
            unexplained.add(record.id)
    return (
        [stage for stage in STAGE_IDS if stage in own],
        [stage for stage in STAGE_IDS if stage in unexplained],
    )


def stages_threshold(stages: Sequence[StageRecord]) -> ThresholdResult:
    """The `stages` threshold over the stage records alone: the number of stages at fault (see
    `_stage_faults`), those with no record or two, and the records of a stage that does not exist;
    it needs no figure, so a report can check the one it holds against its own stages."""
    counts = Counter(record.id for record in stages)
    unknown = sum(count for stage, count in counts.items() if stage not in STAGE_IDS)
    failed, unexplained = _stage_faults(stages)
    missing = [stage for stage in STAGE_IDS if counts[stage] == 0]
    repeated = [stage for stage in STAGE_IDS if counts[stage] > 1]
    segments = []
    if failed:
        segments.append(
            f"failed or skipped without a closed reason: {', '.join(failed)}"
        )
    if unexplained:
        # True with no failure before the stage and with one that does not explain the skip.
        segments.append(
            "skipped for a reason that no failure before it explains: "
            f"{', '.join(unexplained)}"
        )
    if missing:
        segments.append(f"no record: {', '.join(missing)}")
    if repeated:
        segments.append(f"recorded twice: {', '.join(repeated)}")
    if unknown:
        segments.append(f"{unknown} records of an unknown stage")
    at_fault = len({*failed, *unexplained, *missing, *repeated}) + unknown
    return ThresholdResult(
        "stages",
        _STAGES_BOUND,
        at_fault,
        "pass" if at_fault == 0 else "fail",
        "; ".join(segments) or _STAGES_FINE,
    )


def _memorised(table: object, direction: str, system: str) -> float | None:
    if not isinstance(table, Mapping):
        return None
    return _figure(table.get(direction), system, _CHRF_RANGE)


def _names_missing(pairs: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{system} chrF of {direction}" for direction, system in pairs)


def _train_chrf(table: object, limit: float) -> ThresholdResult:
    figures: dict[str, float] = {}
    absent: list[tuple[str, str]] = []
    for direction in MEMORISED_DIRECTIONS:
        figure = _memorised(table, direction, GATED_SYSTEM)
        if figure is None:
            absent.append((direction, GATED_SYSTEM))
        else:
            figures[direction] = figure
    if absent:
        return _missing("train_chrf", True, limit, _names_missing(absent))
    least = min(figures, key=figures.__getitem__)
    return _result(
        "train_chrf",
        True,
        limit,
        figures[least],
        f"least of {len(figures)} directions, {least}, {GATED_SYSTEM} export",
    )


def _int8_drop(table: object, limit: float) -> ThresholdResult:
    drops: dict[str, float] = {}
    absent: list[tuple[str, str]] = []
    for direction in MEMORISED_DIRECTIONS:
        reference = _memorised(table, direction, REFERENCE_SYSTEM)
        gated = _memorised(table, direction, GATED_SYSTEM)
        if reference is None:
            absent.append((direction, REFERENCE_SYSTEM))
        if gated is None:
            absent.append((direction, GATED_SYSTEM))
        if reference is not None and gated is not None:
            drops[direction] = reference - gated
    if absent:
        return _missing("int8_chrf_drop", False, limit, _names_missing(absent))
    largest = max(drops, key=drops.__getitem__)
    return _result(
        "int8_chrf_drop",
        False,
        limit,
        drops[largest],
        f"largest {REFERENCE_SYSTEM} minus {GATED_SYSTEM} chrF over {len(drops)} directions, "
        f"{largest}",
    )


def _from_table(
    threshold_id: str,
    table: Mapping[str, Any] | None,
    key: str,
    what: str,
    *,
    at_least: bool,
    limit: float,
    limits: tuple[float, float],
) -> ThresholdResult:
    value = _figure(table, key, limits)
    if value is None:
        return _missing(threshold_id, at_least, limit, f"{key} of the {what} figures")
    return _result(threshold_id, at_least, limit, value, f"{key} of the {what} figures")


def _wall_time(elapsed: object, budget: int) -> ThresholdResult:
    value = _number(elapsed, 0.0, _NO_UPPER_BOUND)
    if value is None:
        return _missing("wall_time", False, budget, "the elapsed time of the run")
    return _result(
        "wall_time", False, budget, value, "seconds from the first line of the run"
    )


def evaluate_thresholds(
    inputs: ThresholdInputs, thresholds: ThresholdsSection, budget_seconds: int
) -> tuple[ThresholdResult, ...]:
    """The eight thresholds of the table, in order, over `inputs`, with the bounds of `thresholds`
    and the wall-time bound `budget_seconds`; none raises, and each missing value is a `fail`."""
    return (
        stages_threshold(inputs.stages),
        _train_chrf(inputs.memorised_chrf, thresholds.train_chrf_min),
        _from_table(
            "parity_exact",
            inputs.parity,
            "exact_share",
            "parity",
            at_least=True,
            limit=thresholds.parity_exact_min,
            limits=_SHARE_RANGE,
        ),
        _from_table(
            "parity_delta",
            inputs.parity,
            "mean_abs_delta",
            "parity",
            at_least=False,
            limit=thresholds.parity_delta_max,
            limits=(0.0, _NO_UPPER_BOUND),
        ),
        _int8_drop(inputs.memorised_chrf, thresholds.int8_chrf_drop_max),
        _from_table(
            "script_purity",
            inputs.robustness,
            "script_purity",
            "robustness",
            at_least=True,
            limit=thresholds.script_purity_min,
            limits=_SHARE_RANGE,
        ),
        _from_table(
            "placeholder_acc",
            inputs.robustness,
            "placeholder_acc",
            "robustness",
            at_least=True,
            limit=thresholds.placeholder_acc_min,
            limits=_SHARE_RANGE,
        ),
        _wall_time(inputs.elapsed_seconds, budget_seconds),
    )
