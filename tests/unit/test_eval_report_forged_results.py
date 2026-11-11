"""A report believes nothing a result file says about its own outputs (D13 fix wave 1, finding I1).

The digest chain catches a flipped byte; it cannot catch a result file that was edited and whose
digests were then updated in `run.json` and in the manifest. So the report recomputes from the suite
and the outputs of the run, for every direction that ran, the interval of each count metric, the
statuses, the three sacrebleu signatures and the `trained` flag, and refuses (a `DataError`, exit 3
of the command) a result that says anything else, naming the run, the suite, the direction and the
field and never a value. The neural metrics have no segment scores in a report, so their value and
interval stay what the result says (they are not recomputable here).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.report import RunInput, build_report
from rinchen.eval.report_runs import _same_interval
from tests.support.eval_fakes import golden_outputs, golden_suite
from tests.support.eval_report import (
    FIRST,
    SECOND,
    file_system,
    golden_run,
    golden_runs,
    golden_suites,
    run_input,
    shipped_config,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

WHERE = "the run of the system 'alpha' over the suite 'golden', direction " + FIRST


def report_of(runs: list[RunInput]) -> dict[str, Any]:
    return build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )


def flagged_run(name: str = "alpha") -> RunInput:
    """The golden run of `name` whose first direction holds one empty, one truncated and two
    `no_output` texts: the failures a system flagged."""
    outputs = golden_outputs(name)
    for row_id, text, status in (
        ("g01", "", "empty"),
        ("g02", "cut off", "truncated"),
        ("g03", "", "no_output"),
        ("g04", "", "no_output"),
    ):
        outputs[OutputKey(FIRST, row_id)] = Output(text, status)  # type: ignore[arg-type]
    return run_input(golden_suite(), file_system(name, outputs))


def with_first_result(run: RunInput, **changes: Any) -> RunInput:
    results = tuple(
        replace(result, **changes) if result.direction == FIRST else result
        for result in run.results
    )
    return replace(run, results=results)


def forged(run: RunInput, **changes: Any) -> list[RunInput]:
    return [with_first_result(run, **changes), golden_run("beta")]


# ---- the genuine results still pass -----------------------------------------------------------------


def test_a_run_with_flagged_outputs_is_reported_with_its_statuses() -> None:
    report = report_of([flagged_run(), golden_run("beta")])

    first = next(
        result
        for result in report["results"]
        if result["system"] == "alpha" and result["direction"] == FIRST
    )
    assert first["statuses"] == {"empty": 1, "no_output": 2, "ok": 6, "truncated": 1}


def test_the_genuine_golden_runs_pass_every_recomputation() -> None:
    assert len(report_of(golden_runs())["results"]) == 4


# ---- statuses -------------------------------------------------------------------------------------


def test_statuses_that_hide_the_flagged_outputs_are_refused() -> None:
    runs = forged(flagged_run(), statuses={"ok": 10})

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the statuses of the result do not match its outputs",
    ):
        report_of(runs)


def test_statuses_that_count_one_more_failure_than_the_outputs_hold_are_refused() -> (
    None
):
    runs = forged(
        flagged_run(), statuses={"empty": 2, "no_output": 2, "ok": 5, "truncated": 1}
    )

    with pytest.raises(DataError, match="the statuses of the result do not match"):
        report_of(runs)


def test_a_status_named_differently_with_the_same_counts_is_refused() -> None:
    runs = forged(
        flagged_run(), statuses={"empty": 1, "no_output": 2, "ok": 6, "too_long": 1}
    )

    with pytest.raises(DataError, match="the statuses of the result do not match"):
        report_of(runs)


# ---- the intervals -------------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["bleu", "chrf", "chrf_pp"])
def test_an_interval_that_the_outputs_do_not_give_is_refused(metric: str) -> None:
    run = golden_run("alpha")
    interval = [bound + 0.5 for bound in run.results[0].ci95[metric] or []]
    runs = forged(run, ci95={**run.results[0].ci95, metric: interval})

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the ci95 of the {metric} of the result does not match its "
        "outputs",
    ):
        report_of(runs)


@pytest.mark.parametrize("end", [0, 1])
def test_either_end_of_a_forged_interval_is_enough(end: int) -> None:
    run = golden_run("alpha")
    interval = list(run.results[0].ci95["chrf_pp"] or [])
    interval[end] += 1e-6
    runs = forged(run, ci95={**run.results[0].ci95, "chrf_pp": interval})

    with pytest.raises(DataError, match="the ci95 of the chrf_pp of the result"):
        report_of(runs)


def test_an_interval_within_rounding_of_the_recomputed_one_is_accepted() -> None:
    run = golden_run("alpha")
    interval = [bound * (1 + 1e-12) for bound in run.results[0].ci95["bleu"] or []]
    runs = forged(run, ci95={**run.results[0].ci95, "bleu": interval})

    assert len(report_of(runs)["results"]) == 4


def test_a_neural_metric_of_a_result_is_not_recomputed_and_not_refused() -> None:
    scores = {
        "comet22": {
            OutputKey(direction, row_id): 0.5 + index / 100
            for direction in (FIRST, SECOND)
            for index, row_id in enumerate(
                ("g01", "g02", "g03", "g04", "g05", "g06", "g07", "g08", "g09", "g10")
            )
        }
    }
    run = run_input(golden_suite(), file_system("alpha"), scores=scores)

    report = report_of([run, golden_run("beta")])

    first = next(r for r in report["results"] if r["system"] == "alpha")
    assert first["metrics"]["comet22"] is not None
    assert first["ci95"]["comet22"] is not None


# ---- the signatures ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["bleu_signature", "chrf_signature", "chrf_pp_signature"]
)
def test_a_signature_of_another_version_than_the_recomputed_one_is_refused(
    field: str,
) -> None:
    run = golden_run("alpha")
    signature = getattr(run.results[0], field)
    assert "version:" in signature
    stale = signature.rsplit("version:", 1)[0] + "version:9.9.9"
    runs = forged(run, **{field: stale})

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the {field} of the result does not match its outputs",
    ):
        report_of(runs)


def test_a_result_that_names_the_signature_of_another_tokenizer_is_refused() -> None:
    run = golden_run("alpha")
    signature = run.results[0].bleu_signature
    assert "tok:13a" in signature
    runs = forged(run, bleu_signature=signature.replace("tok:13a", "tok:char"))

    with pytest.raises(DataError, match="the bleu_signature of the result"):
        report_of(runs)


def test_a_result_of_an_older_sacrebleu_is_refused_even_when_its_figures_hold() -> None:
    # The figures of a result written by older code can be right and its signature still be
    # another stack's: the report only states what its own stack recomputes.
    run = golden_run("alpha")
    old = {
        field: getattr(run.results[0], field).replace("version:2.6.0", "version:2.5.1")
        for field in ("bleu_signature", "chrf_signature", "chrf_pp_signature")
    }
    assert old["bleu_signature"] != run.results[0].bleu_signature

    with pytest.raises(DataError, match="signature of the result does not match"):
        report_of(forged(run, **old))


# ---- trained --------------------------------------------------------------------------------------


def test_a_direction_that_ran_but_says_it_was_not_trained_is_refused() -> None:
    runs = forged(golden_run("alpha"), trained=False)

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the trained flag of the result does not match its run",
    ):
        report_of(runs)


# ---- the other direction is checked too -----------------------------------------------------------


def test_a_forged_second_direction_is_refused_naming_it() -> None:
    run = golden_run("alpha")
    results = tuple(
        replace(result, statuses={"ok": 10}) if result.direction == SECOND else result
        for result in flagged_second_direction(run).results
    )
    runs = [replace(flagged_second_direction(run), results=results), golden_run("beta")]

    with pytest.raises(
        DataError, match=f"direction {SECOND}: the statuses of the result"
    ):
        report_of(runs)


def flagged_second_direction(run: RunInput) -> RunInput:
    outputs = golden_outputs("alpha")
    outputs[OutputKey(SECOND, "g05")] = Output("", "empty")
    return run_input(golden_suite(), file_system("alpha", outputs))


# ---- the bootstrap record --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["resamples", "seed", "alpha"])
def test_a_bootstrap_record_that_is_not_the_one_the_report_uses_is_refused(
    field: str,
) -> None:
    run = golden_run("alpha")
    bootstrap = run.results[0].bootstrap
    other = {"resamples": 500, "seed": 7, "alpha": 0.1}[field]
    runs = forged(run, bootstrap=replace(bootstrap, **{field: other}))

    with pytest.raises(
        DataError,
        match=f"{WHERE}: the bootstrap of the result is not the one the report is "
        "made with",
    ):
        report_of(runs)


# ---- the comparison of two intervals ---------------------------------------------------------------


def test_an_interval_is_never_the_same_as_none_and_none_is_the_same_as_none() -> None:
    # A result cannot hold a null interval beside two or more units (its reader refuses that), so
    # the mixed case is only reachable by a direct call; it is the rule that decides it.
    assert _same_interval(None, None)
    assert not _same_interval(None, (1.0, 2.0))
    assert not _same_interval([1.0, 2.0], None)
    assert _same_interval([1.0, 2.0], (1.0, 2.0))
    assert not _same_interval([1.0, 2.0], (1.0, 2.5))
