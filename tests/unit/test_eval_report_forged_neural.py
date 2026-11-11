"""A report holds a neural metric only where the run names the scores it came from (D13 fix wave 2,
finding m1).

The neural metrics (COMET-22, MetricX-24 and its QE variant) cannot be recomputed in a report: it
holds no segment scores. What it can check is the presence: a result holds a value for a neural
metric exactly when `run.json` lists the scores file of that metric, and when it holds none, its
reason is the one the runner writes. A result that was edited and re-digested to plant a value, to
hide one or to carry another reason is refused (a `DataError`, exit 3 of the command), naming the run,
the suite, the direction and the field and never a value.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.outputs import OutputKey
from rinchen.eval.report import RunInput, build_report
from rinchen.eval.result import DirectionResult
from rinchen.metrics.registry import NO_SCORER_REASON, SEGMENT_SCORE_METRICS
from tests.support.eval_fakes import golden_suite
from tests.support.eval_report import (
    FIRST,
    SECOND,
    file_system,
    golden_run,
    golden_suites,
    run_input,
    shipped_config,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

WHERE = "the run of the system 'alpha' over the suite 'golden', direction " + FIRST
ROW_IDS = ("g01", "g02", "g03", "g04", "g05", "g06", "g07", "g08", "g09", "g10")


def report_of(runs: list[RunInput]) -> dict[str, Any]:
    return build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )


def scored_run(*metrics: str) -> RunInput:
    """The golden run of `alpha` with segment scores for each of `metrics`, as the command makes it."""
    scores = {
        metric: {
            OutputKey(direction, row_id): 0.5 + index / 100
            for direction in (FIRST, SECOND)
            for index, row_id in enumerate(ROW_IDS)
        }
        for metric in metrics
    }
    return run_input(golden_suite(), file_system("alpha"), scores=scores)


def with_first(run: RunInput, change: Any) -> list[RunInput]:
    """`run` with `change(result)` for its first direction, beside the genuine run of `beta`."""
    results = tuple(
        change(result) if result.direction == FIRST else result
        for result in run.results
    )
    return [replace(run, results=results), golden_run("beta")]


def planted(metric: str) -> Any:
    """A result that holds a value and an interval for `metric` and no reason."""

    def change(result: DirectionResult) -> DirectionResult:
        return replace(
            result,
            metrics={**result.metrics, metric: 0.5},
            ci95={**result.ci95, metric: [0.4, 0.6]},
            unmeasured={k: v for k, v in result.unmeasured.items() if k != metric},
        )

    return change


def hidden(metric: str) -> Any:
    """A result that holds no value for `metric` and the reason the runner writes."""

    def change(result: DirectionResult) -> DirectionResult:
        return replace(
            result,
            metrics={**result.metrics, metric: None},
            ci95={**result.ci95, metric: None},
            unmeasured={**result.unmeasured, metric: NO_SCORER_REASON},
        )

    return change


def reworded(metric: str, reason: str) -> Any:
    def change(result: DirectionResult) -> DirectionResult:
        return replace(result, unmeasured={**result.unmeasured, metric: reason})

    return change


# ---- the genuine runs still pass ----------------------------------------------------------------


def test_a_run_without_scores_holds_every_neural_metric_unmeasured_and_passes() -> None:
    result = golden_run("alpha").results[0]
    assert all(result.metrics[name] is None for name in SEGMENT_SCORE_METRICS)
    assert all(
        result.unmeasured[name] == NO_SCORER_REASON for name in SEGMENT_SCORE_METRICS
    )

    assert len(report_of([golden_run("alpha"), golden_run("beta")])["results"]) == 4


@pytest.mark.parametrize("metric", SEGMENT_SCORE_METRICS)
def test_a_run_whose_record_lists_the_scores_and_whose_result_holds_the_value_passes(
    metric: str,
) -> None:
    run = scored_run(metric)
    assert run.results[0].metrics[metric] is not None

    first = next(
        result
        for result in report_of([run, golden_run("beta")])["results"]
        if result["system"] == "alpha"
    )

    assert first["metrics"][metric] is not None


def test_one_metric_with_scores_leaves_the_others_to_their_reason() -> None:
    run = scored_run("metricx24")

    report = report_of([run, golden_run("beta")])

    first = next(r for r in report["results"] if r["system"] == "alpha")
    assert first["metrics"]["metricx24"] is not None
    assert first["metrics"]["comet22"] is None
    assert first["unmeasured"]["comet22"] == NO_SCORER_REASON


# ---- a value planted where the run lists no scores -----------------------------------------------


@pytest.mark.parametrize("metric", SEGMENT_SCORE_METRICS)
def test_a_value_the_run_never_scored_is_refused(metric: str) -> None:
    runs = with_first(golden_run("alpha"), planted(metric))

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the {metric} of the result has a value and the run lists no "
        "segment scores for it",
    ):
        report_of(runs)


def test_a_value_for_a_metric_the_run_did_not_score_is_refused_beside_one_it_did() -> (
    None
):
    runs = with_first(scored_run("comet22"), planted("metricx24"))

    with pytest.raises(DataError, match="the metricx24 of the result has a value"):
        report_of(runs)


def test_the_refusal_quotes_no_value() -> None:
    runs = with_first(golden_run("alpha"), planted("comet22"))

    with pytest.raises(DataError) as refused:
        report_of(runs)

    assert "0.5" not in str(refused.value)
    assert "0.4" not in str(refused.value)


# ---- a value the run scored, hidden by the result --------------------------------------------------


@pytest.mark.parametrize("metric", SEGMENT_SCORE_METRICS)
def test_a_result_that_hides_a_score_the_run_lists_is_refused(metric: str) -> None:
    runs = with_first(scored_run(metric), hidden(metric))

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the {metric} of the result has no value and the run lists "
        "segment scores for it",
    ):
        report_of(runs)


# ---- the reason of a metric that has no value ------------------------------------------------------


@pytest.mark.parametrize("metric", SEGMENT_SCORE_METRICS)
def test_another_reason_than_the_runner_writes_is_refused(metric: str) -> None:
    runs = with_first(golden_run("alpha"), reworded(metric, "the scorer crashed"))

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the unmeasured reason of the {metric} of the result is not "
        "the one the runner writes",
    ):
        report_of(runs)


def test_the_refusal_of_a_reason_quotes_no_text_of_it() -> None:
    runs = with_first(golden_run("alpha"), reworded("comet22", "SECRET-REASON-77"))

    with pytest.raises(DataError) as refused:
        report_of(runs)

    assert "SECRET-REASON-77" not in str(refused.value)


def test_the_second_direction_is_checked_too() -> None:
    run = golden_run("alpha")
    results = tuple(
        planted("comet22")(result) if result.direction == SECOND else result
        for result in run.results
    )

    with pytest.raises(
        DataError, match=f"direction {SECOND}: the comet22 of the result has a value"
    ):
        report_of([replace(run, results=results), golden_run("beta")])
