"""The report of D13 (spec T14): `build_report` turns runs and the suites they scored into
`rinchen.report/1`, the one document that says what was measured, how it compares and where every
number came from (the sections are described in `rinchen.eval.report_format`).

The runs are checked first (`rinchen.eval.report_runs`: a `dev` run, two runs of one system over one
suite, other bootstrap settings, a result that disagrees with its run and so on are refused). The
comparisons are RECOMPUTED here from the suite and the outputs of the runs, for each metric of
`config.comparison_metrics`: the candidate against the baseline for every other system of the same
suite and direction, one set of draws for the pair (`rinchen.eval.significance`). The verdict
`significant` is the p-value against alpha with the delta in favour of the candidate, never the
interval. A baseline without a result for a suite and direction gives no comparison, and the
coverage says why: its own row says what became of the direction, and in a suite where it has no run
at all a row of state `absent` says so for each direction.

The report holds no timestamp and no absolute path (`DataError` otherwise), and is a function of the
SET of runs: whatever order they are given in, the bytes are the same.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rinchen.config.eval_metrics import EvalMetricsConfig
from rinchen.errors import ConfigError, DataError
from rinchen.eval.bootstrap import RNG_NAME, BootstrapSettings
from rinchen.eval.report_format import (
    ABSENT_REASON,
    ABSENT_STATE,
    MULTIPLE_COMPARISONS_NOTE,
    REPORT_SCHEMA,
    count_intervals,
    multiple_comparisons_note,
    report_to_bytes,
    require_no_absolute_path,
)
from rinchen.eval.report_runs import (
    HeldRun,
    RunInput,
    hold_runs,
    robustness_of,
    verified_rows,
)
from rinchen.eval.robust_suite_rules import is_id
from rinchen.eval.scoring import INTERVAL_KIND, DirectionRows
from rinchen.eval.significance import PairedComparison, compare, is_significant
from rinchen.eval.suites import Suite, standard_directions
from rinchen.eval.targets import evaluate_targets
from rinchen.metrics.registry import HIGHER_IS_BETTER, CountMetricName
from rinchen.metrics.segment_stats import score_from_statistics

__all__ = [
    "MULTIPLE_COMPARISONS_NOTE",
    "REPORT_SCHEMA",
    "RunInput",
    "build_report",
    "multiple_comparisons_note",
    "report_to_bytes",
]

_ROBUSTNESS_FIELDS = ("robustness", "robustness_coverage", "robustness_source")


def build_report(
    runs: Sequence[RunInput],
    suites: Mapping[str, Suite],
    *,
    baseline: str | None,
    config: EvalMetricsConfig,
) -> dict[str, Any]:
    """The report of `runs`, whose suites are `suites` keyed by the SHA-256 of the suite file.
    The comparisons are against `baseline` (the configuration's `baseline_system` when it is
    None, none when both are).

    Raises:
        ConfigError: no run, a suite that was not given or that no run scored, a run of a `dev`
            suite, or a baseline that is no system of any run.
        DataError: runs that cannot be reported (see `rinchen.eval.report_runs`), or a report that
            would hold an absolute path.
    """
    settings = BootstrapSettings(
        config.bootstrap.resamples, config.bootstrap.seed, config.bootstrap.alpha
    )
    held = hold_runs(runs, suites, settings)
    named = baseline if baseline is not None else config.baseline_system
    _require_a_system_of_the_runs(named, held)
    rows = {(one.suite_name, one.system): verified_rows(one, settings) for one in held}

    comparisons = _comparisons(held, rows, named, config, settings)
    results = [
        _without_robustness(result.to_dict())
        for one in held
        for result in one.results.values()
    ]
    robustness = _robustness(held)
    suite_documents = _suites(held)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "evidence": _evidence(suite_documents),
        "systems": _systems(held),
        "suites": suite_documents,
        "runs": [
            {
                "suite": one.suite_name,
                "system": one.system,
                "run_sha256": one.input.run_sha256,
            }
            for one in held
        ],
        "coverage": _coverage(held, named),
        "results": results,
        "robustness": robustness,
        "comparisons": comparisons,
        "targets": [
            outcome.to_dict()
            for outcome in evaluate_targets(
                config,
                comparisons,
                results,
                robustness,
                suites={suite["name"]: suite for suite in suite_documents},
            )
        ],
        "bootstrap": {
            "resamples": settings.resamples,
            "seed": settings.seed,
            "alpha": settings.alpha,
            "interval": INTERVAL_KIND,
            "rng": RNG_NAME,
        },
        "note_multiple_comparisons": multiple_comparisons_note(
            count_intervals(results, comparisons), len(comparisons)
        ),
    }
    try:
        require_no_absolute_path(report)
    except ValueError as error:
        raise DataError(f"not an evaluation report: {error}") from error
    return report


def _require_a_system_of_the_runs(name: str | None, held: Sequence[HeldRun]) -> None:
    if name is None:
        return
    if not is_id(name):
        raise ConfigError(
            "the baseline must be a system name: 1 to 64 characters, letters, digits, "
            "'.', '_' and '-'"
        )
    if name not in {one.system for one in held}:
        raise ConfigError(f"the baseline {name!r} is not a system of any run")


# ---- the descriptions --------------------------------------------------------------------------


def _evidence(suites: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kinds = sorted({suite["kind"] for suite in suites})
    return {"kinds": kinds, "fixture_only": kinds == ["fixture"]}


def _systems(held: Sequence[HeldRun]) -> list[dict[str, Any]]:
    records = {one.system: one.input.record.system for one in held}
    return [
        {
            "name": name,
            "kind": record.kind,
            "model_sha256": record.model_sha256,
            "manifest_sha256": record.manifest_sha256,
            "options": dict(record.options),
        }
        for name, record in sorted(records.items())
    ]


def _suites(held: Sequence[HeldRun]) -> list[dict[str, Any]]:
    suites = {
        one.suite_name: (one.suite, one.input.record.suite.sha256) for one in held
    }
    return [
        {
            "name": name,
            "kind": suite.kind,
            "benchmark": suite.benchmark,
            "split": suite.split,
            "purpose": suite.purpose,
            "sha256": digest,
            "rows": len(suite.rows),
            "directions": _expected_directions(suite),
            "source": {
                "dataset": suite.source.dataset,
                "revision": suite.source.revision,
                "pinned": suite.source.pinned,
                "missing_languages": list(suite.source.missing_languages),
            },
        }
        for name, (suite, digest) in sorted(suites.items())
    ]


def _expected_directions(suite: Suite) -> list[dict[str, Any]]:
    """A row for each direction the report expects: the standard ones in their order with the
    number of rows the suite has (null when it does not declare the direction), then the suite's
    other directions in its order."""
    declared = {
        direction: len(suite.view(direction).ids) for direction in suite.directions
    }
    standard = standard_directions()
    return [
        *({"direction": d, "rows": declared.get(d)} for d in standard),
        *(
            {"direction": d, "rows": rows}
            for d, rows in declared.items()
            if d not in standard
        ),
    ]


def _coverage(held: Sequence[HeldRun], baseline: str | None) -> list[dict[str, Any]]:
    keyed: list[tuple[str, str, int, dict[str, Any]]] = []
    for one in held:
        for index, entry in enumerate(one.input.record.directions):
            keyed.append(
                (
                    one.suite_name,
                    one.system,
                    index,
                    _coverage_row(
                        one.suite_name,
                        one.system,
                        entry.direction,
                        entry.state,
                        entry.reason,
                    ),
                )
            )
    if baseline is not None:
        suites = {one.suite_name: one.suite for one in held}
        for suite_name, suite in suites.items():
            if not any(
                o.suite_name == suite_name and o.system == baseline for o in held
            ):
                for index, direction in enumerate(suite.directions):
                    row = _coverage_row(
                        suite_name, baseline, direction, ABSENT_STATE, ABSENT_REASON
                    )
                    keyed.append((suite_name, baseline, index, row))
    keyed.sort(key=lambda item: item[:3])
    return [row for *_, row in keyed]


def _coverage_row(
    suite: str, system: str, direction: str, state: str, reason: str | None
) -> dict[str, Any]:
    return {
        "suite": suite,
        "system": system,
        "direction": direction,
        "state": state,
        "reason": reason,
    }


def _without_robustness(result: dict[str, Any]) -> dict[str, Any]:
    for name in _ROBUSTNESS_FIELDS:
        del result[name]
    return result


def _robustness(held: Sequence[HeldRun]) -> dict[str, Any]:
    found: dict[str, dict[str, Any]] = {}
    for one in held:
        entry = robustness_of(one)
        if entry is None:
            continue
        if found.setdefault(one.system, entry) != entry:
            raise DataError(
                f"the system {one.system!r} carries different robustness blocks in "
                "different runs"
            )
    return dict(sorted(found.items()))


# ---- the comparisons ---------------------------------------------------------------------------


def _comparisons(
    held: Sequence[HeldRun],
    rows: Mapping[tuple[str, str], Mapping[str, DirectionRows]],
    baseline: str | None,
    config: EvalMetricsConfig,
    settings: BootstrapSettings,
) -> list[dict[str, Any]]:
    """The candidate against the baseline in every suite and direction both ran, for each metric
    of the configuration, recomputed from the statistics of the outputs."""
    if baseline is None:
        return []
    by_key = {(one.suite_name, one.system): one for one in held}
    comparisons: list[dict[str, Any]] = []
    for suite_name in sorted({one.suite_name for one in held}):
        base = by_key.get((suite_name, baseline))
        if base is None:
            continue
        candidates = sorted(
            one.system
            for one in held
            if one.suite_name == suite_name and one.system != baseline
        )
        for direction in base.suite.directions:
            if direction not in base.results:
                continue
            for candidate in candidates:
                if direction not in by_key[(suite_name, candidate)].results:
                    continue
                for metric in config.comparison_metrics:
                    comparisons.append(
                        _comparison(
                            suite_name,
                            direction,
                            candidate,
                            baseline,
                            metric,
                            rows[(suite_name, candidate)][direction],
                            rows[(suite_name, baseline)][direction],
                            settings,
                        )
                    )
    return comparisons


def _comparison(
    suite: str,
    direction: str,
    candidate: str,
    baseline: str,
    metric: CountMetricName,
    candidate_rows: DirectionRows,
    baseline_rows: DirectionRows,
    settings: BootstrapSettings,
) -> dict[str, Any]:
    counted = candidate_rows.metrics[metric]
    higher_is_better = HIGHER_IS_BETTER[metric]
    paired: PairedComparison = compare(
        counted.rows,
        baseline_rows.metrics[metric].rows,
        lambda totals: score_from_statistics(counted.metric, totals),
        settings,
        higher_is_better=higher_is_better,
    )
    return {
        "suite": suite,
        "direction": direction,
        "system": candidate,
        "baseline": baseline,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "delta": paired.delta,
        "ci95": None if paired.interval is None else list(paired.interval),
        "p_value": paired.p_value,
        "better_fraction": paired.better_fraction,
        "significant": is_significant(paired, settings.alpha),
        "unit": candidate_rows.unit,
        "units": paired.units,
    }
