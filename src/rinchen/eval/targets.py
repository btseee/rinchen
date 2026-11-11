"""The release targets of D13 (spec T13) evaluated as MEASUREMENTS: each target of the metrics
configuration becomes a status, `met`, `not_met` or `unmeasured`, and a sentence saying what the
status rests on. Nothing fails because a target is not met, and `unmeasured` is the answer whenever
the evidence does not decide.

The targets are about a model of ours (a system of kind `rinchen`): a hypotheses file or a baseline
is never the candidate, however good its figures. Several candidates are each judged and the worse
one decides (`not_met` if any is, `met` only if all are). Everything here is a pure function of
what the report holds, given as the plain dictionaries of the document:

- `beats_baseline`: the directions of the benchmark `flores_plus`, split `devtest`, in which the
  candidate is significantly better than the baseline on the primary metric (the verdict of a
  comparison is its p-value against alpha: the report's `significant`). `met` at `min_directions`,
  `not_met` when the directions not measured yet cannot make up the difference, else `unmeasured`.
- `off_target`: never measured (no language identification is run).
- `robustness_value`: a value of D12's block against the threshold; `met` needs every unit scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from rinchen.config.eval_metrics import EvalMetricsConfig, TargetConfig

__all__ = ["TARGET_STATUSES", "TargetOutcome", "evaluate_targets"]

TARGET_STATUSES: Final = ("met", "not_met", "unmeasured")
BENCHMARK: Final = "flores_plus"
SPLIT: Final = "devtest"
CANDIDATE_KIND: Final = "rinchen"
_NO_LANGUAGE_IDENTIFICATION: Final = "no language identification is run"
_NO_ROBUSTNESS: Final = "no robustness run of a rinchen system was given"
_NULL_VALUE: Final = "the value is null: no unit was scored"

Number = float | int


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """One target as measured: its `id` and `description`, the `threshold` it asks for, the
    `value` that was found (None when nothing was), the `status` (`met`, `not_met` or
    `unmeasured`) and the `basis` sentence.

    Raises:
        ValueError: `status` is none of the three.
    """

    id: str
    description: str
    threshold: Number | None
    value: Number | None
    status: str
    basis: str

    def __post_init__(self) -> None:
        if self.status not in TARGET_STATUSES:
            raise ValueError(f"status must be one of {', '.join(TARGET_STATUSES)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "threshold": self.threshold,
            "value": self.value,
            "status": self.status,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class _Judged:
    """What one candidate came to: a status, the value found and the sentence."""

    status: str
    value: Number | None
    basis: str


def evaluate_targets(
    config: EvalMetricsConfig,
    comparisons: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    robustness: Mapping[str, Mapping[str, Any]],
    *,
    suites: Mapping[str, Mapping[str, Any]],
) -> tuple[TargetOutcome, ...]:
    """The outcome of every target of `config`, in its order.

    `comparisons`, `results` and `robustness` are the sections of the report of that name (a
    result is needed for the kind of its system) and `suites` the suites of the report by name
    (their `benchmark` and `split` decide which comparisons count).
    """
    candidates = {
        result["system"]
        for result in results
        if result["provenance"]["system"]["kind"] == CANDIDATE_KIND
    }
    outcomes: list[TargetOutcome] = []
    for target in config.targets:
        if target.kind == "beats_baseline":
            judged = _beats_baseline(
                target, config.primary_metric, comparisons, candidates, suites
            )
            threshold: Number | None = target.min_directions
        elif target.kind == "off_target":
            judged = _Judged("unmeasured", None, _NO_LANGUAGE_IDENTIFICATION)
            threshold = target.threshold
        else:
            judged = _robustness_value(target, candidates, robustness)
            threshold = target.threshold
        outcomes.append(
            TargetOutcome(
                target.id,
                target.description,
                threshold,
                judged.value,
                judged.status,
                judged.basis,
            )
        )
    return tuple(outcomes)


# ---- several candidates -----------------------------------------------------------------------


def _worst(judged: Mapping[str, _Judged]) -> _Judged:
    """One verdict from the verdict of each candidate, named in order: the worse decides."""
    if len(judged) == 1:
        return next(iter(judged.values()))
    statuses = {one.status for one in judged.values()}
    status = (
        "not_met"
        if "not_met" in statuses
        else "unmeasured"
        if "unmeasured" in statuses
        else "met"
    )
    values = [one.value for one in judged.values() if one.value is not None]
    basis = "; ".join(f"{name}: {judged[name].basis}" for name in sorted(judged))
    return _Judged(status, min(values) if values else None, basis)


# ---- beats_baseline ---------------------------------------------------------------------------


def _beats_baseline(
    target: TargetConfig,
    metric: str,
    comparisons: Sequence[Mapping[str, Any]],
    candidates: set[str],
    suites: Mapping[str, Mapping[str, Any]],
) -> _Judged:
    minimum, total = target.min_directions, target.of_directions
    assert minimum is not None and total is not None  # the configuration holds both
    verdicts: dict[str, dict[str, list[bool]]] = {}
    for comparison in comparisons:
        suite = suites.get(comparison["suite"])
        if (
            comparison["system"] not in candidates
            or comparison["metric"] != metric
            or comparison["p_value"] is None
            or suite is None
            or (suite["benchmark"], suite["split"]) != (BENCHMARK, SPLIT)
        ):
            continue
        by_direction = verdicts.setdefault(comparison["system"], {})
        by_direction.setdefault(comparison["direction"], []).append(
            bool(comparison["significant"])
        )
    if not verdicts:
        return _judge_directions(0, 0, minimum, total)
    return _worst(
        {
            system: _judge_directions(
                sum(all(verdict) for verdict in by_direction.values()),
                len(by_direction),
                minimum,
                total,
            )
            for system, by_direction in verdicts.items()
        }
    )


def _judge_directions(wins: int, measured: int, minimum: int, total: int) -> _Judged:
    left = max(total - measured, 0)
    value = wins if measured else None
    if wins >= minimum:
        return _Judged(
            "met",
            value,
            f"{wins} of {total} directions significantly better ({minimum} are needed)",
        )
    if wins + left < minimum:
        remaining = "none" if left == 0 else str(left)
        return _Judged(
            "not_met",
            value,
            f"{wins} of {total} directions significantly better and {remaining} left "
            f"to measure ({minimum} are needed)",
        )
    basis = f"{measured} of {total} directions measured"
    if measured:
        basis += f", {wins} significantly better"
    return _Judged("unmeasured", value, basis)


# ---- robustness_value -------------------------------------------------------------------------


def _robustness_value(
    target: TargetConfig,
    candidates: set[str],
    robustness: Mapping[str, Mapping[str, Any]],
) -> _Judged:
    assert target.key is not None and target.threshold is not None
    held = {name: entry for name, entry in robustness.items() if name in candidates}
    if not held:
        return _Judged("unmeasured", None, _NO_ROBUSTNESS)
    return _worst(
        {
            name: _judge_value(
                entry["block"][target.key],
                entry["coverage"][target.key],
                target.threshold,
            )
            for name, entry in held.items()
        }
    )


def _judge_value(
    value: float | None, coverage: Mapping[str, Any], threshold: float
) -> _Judged:
    if value is None:
        return _Judged("unmeasured", None, _NULL_VALUE)
    shown = f"{value:.4f}"
    if value < threshold:
        return _Judged("not_met", value, f"{shown} is below {threshold:g}")
    if coverage["scored"] != coverage["total"]:
        return _Judged(
            "unmeasured",
            value,
            f"{shown} is at least {threshold:g} on the scored units only: "
            f"{coverage['excluded']} of {coverage['total']} were left out",
        )
    return _Judged("met", value, f"{shown} is at least {threshold:g}")
