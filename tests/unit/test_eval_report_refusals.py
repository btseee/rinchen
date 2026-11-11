"""What `build_report` refuses (D13, Task 8): runs that cannot be reported honestly.

A run of a `dev` suite is for choosing a model and never becomes a result (a `ConfigError` here, the
exit 3 of the command is its own check); two runs of one (suite, system) and a system whose record
changes from run to run are ambiguous; a run scored with other bootstrap settings than the
configuration would sit beside intervals of another kind; and a result that disagrees with the run
that lists it, or whose figures are not what its outputs score, is not trusted: the points are
recomputed from the suite and the outputs, so a result file is never believed to hold more than it
says. A message names suites and systems (ids) and never a text.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.report import RunInput, build_report
from rinchen.eval.result import Provenance
from rinchen.eval.systems import SystemRecord
from tests.support.eval_fakes import golden_suite
from tests.support.eval_report import (
    FIRST,
    SECOND,
    file_system,
    golden_run,
    golden_runs,
    golden_suites,
    robustness_input,
    run_input,
    shipped_config,
    suite_digest,
    without_documents,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def report_of(
    runs: Sequence[RunInput],
    suites: dict[str, Any] | None = None,
    baseline: str | None = "alpha",
) -> dict[str, Any]:
    return build_report(
        runs,
        golden_suites() if suites is None else suites,
        baseline=baseline,
        config=shipped_config(),
    )


# ---- what a report needs ------------------------------------------------------------------------


def test_a_report_of_no_run_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="a report needs at least one run"):
        report_of([])


def test_a_run_whose_suite_was_not_given_is_a_config_error_naming_both_ids() -> None:
    with pytest.raises(ConfigError) as caught:
        report_of(golden_runs(), suites={})

    assert "the suite golden" in str(caught.value) or "'golden'" in str(caught.value)
    assert "alpha" in str(caught.value)
    assert "was not given" in str(caught.value)


def test_a_suite_that_no_run_used_is_a_config_error() -> None:
    other = without_documents(golden_suite(), name="unused")
    suites = {**golden_suites(), suite_digest(other): other}

    with pytest.raises(ConfigError, match="no run scored the suite 'unused'"):
        report_of(golden_runs(), suites=suites)


def test_a_run_matched_to_another_suite_file_than_its_run_json_names_is_a_data_error() -> (
    None
):
    run = replace(golden_run("alpha"), suite_sha256="ab" * 32)

    with pytest.raises(
        DataError,
        match="the run of the system 'alpha' over the suite 'golden' was matched to a "
        "suite file other than the one its run.json names",
    ):
        report_of([run], baseline=None)


def test_a_baseline_that_is_no_system_of_any_run_is_a_config_error() -> None:
    with pytest.raises(
        ConfigError, match="the baseline 'nobody' is not a system of any run"
    ):
        report_of(golden_runs(), baseline="nobody")


# ---- the dev guard ------------------------------------------------------------------------------


def test_a_dev_suite_is_refused_with_a_config_error_naming_it() -> None:
    dev = replace(golden_suite(), split="dev")
    runs = [run_input(dev, file_system("alpha"))]

    with pytest.raises(ConfigError) as caught:
        report_of(runs, suites={suite_digest(dev): dev}, baseline=None)

    assert "the suite 'golden' is a dev suite" in str(caught.value)
    assert "choosing a model" in str(caught.value)


def test_a_run_that_says_it_is_for_selection_is_refused_even_over_a_devtest_suite() -> (
    None
):
    forged = golden_run("alpha")
    record = replace(forged.record, purpose="selection")

    with pytest.raises(ConfigError, match="dev suite"):
        report_of([replace(forged, record=record)], baseline=None)


def test_a_dev_suite_beside_a_good_one_still_refuses_the_whole_report() -> None:
    dev = without_documents(replace(golden_suite(), split="dev"), name="tuning")
    runs = [*golden_runs(), run_input(dev, file_system("alpha"))]
    suites = {**golden_suites(), suite_digest(dev): dev}

    with pytest.raises(ConfigError, match="'tuning' is a dev suite"):
        report_of(runs, suites=suites)


# ---- ambiguity ----------------------------------------------------------------------------------


def test_two_runs_of_one_system_over_one_suite_are_a_data_error() -> None:
    runs = [*golden_runs(), golden_run("alpha")]

    with pytest.raises(
        DataError, match="two runs of the system 'alpha' over the suite 'golden'"
    ):
        report_of(runs)


def test_two_suites_with_one_name_are_a_data_error() -> None:
    other = without_documents(golden_suite())  # the name is `golden`, the bytes differ
    runs = [golden_run("alpha"), run_input(other, file_system("beta"))]
    suites = {**golden_suites(), suite_digest(other): other}

    with pytest.raises(DataError, match="two different suites are named 'golden'"):
        report_of(runs, suites=suites)


def test_a_system_whose_record_differs_between_runs_is_a_data_error() -> None:
    other = without_documents(golden_suite(), name="other")
    first = golden_run("alpha")
    second = run_input(other, file_system("alpha"))
    changed = SystemRecord(
        kind="hypotheses",
        name="alpha",
        options={"beam_size": 4},
        outputs_sha256=second.record.system.outputs_sha256,
    )
    results = tuple(
        replace(r, provenance=replace(r.provenance, system=changed))
        for r in second.results
    )
    second = replace(
        second, record=replace(second.record, system=changed), results=results
    )

    with pytest.raises(
        DataError, match="the system 'alpha' is not the same in every run"
    ):
        report_of(
            [first, second],
            suites={**golden_suites(), suite_digest(other): other},
            baseline=None,
        )


def test_the_outputs_digest_of_a_file_system_may_differ_between_suites() -> None:
    # `outputs_sha256` is the digest of the file a run read: another suite, another file.
    other = without_documents(golden_suite(), name="other")

    report = report_of(
        [golden_run("alpha"), run_input(other, file_system("alpha"))],
        suites={**golden_suites(), suite_digest(other): other},
        baseline=None,
    )

    assert [s["name"] for s in report["systems"]] == ["alpha"]


def test_a_run_scored_with_other_bootstrap_settings_is_a_data_error() -> None:
    quick = run_input(
        golden_suite(),
        file_system("beta"),
        settings=BootstrapSettings(resamples=200, seed=20_261_110, alpha=0.05),
    )

    with pytest.raises(
        DataError,
        match="the run of the system 'beta' over the suite 'golden' was scored with "
        "other bootstrap settings",
    ):
        report_of([golden_run("alpha"), quick])


@pytest.mark.parametrize(
    "settings",
    [
        BootstrapSettings(resamples=1000, seed=7, alpha=0.05),
        BootstrapSettings(resamples=1000, seed=20_261_110, alpha=0.1),
    ],
)
def test_a_seed_or_an_alpha_that_differs_is_just_as_refused(
    settings: BootstrapSettings,
) -> None:
    run = run_input(golden_suite(), file_system("alpha"), settings=settings)

    with pytest.raises(DataError, match="other bootstrap settings"):
        report_of([run], baseline=None)


# ---- a result that disagrees with its run -------------------------------------------------------


def with_result(run: RunInput, direction: str, **changes: Any) -> RunInput:
    results = tuple(
        replace(result, **changes) if result.direction == direction else result
        for result in run.results
    )
    return replace(run, results=results)


def alpha_with(change: Callable[[RunInput], RunInput]) -> list[RunInput]:
    return [change(golden_run("alpha")), golden_run("beta")]


def test_a_run_without_the_result_of_a_direction_that_ran_is_a_data_error() -> None:
    runs = alpha_with(lambda run: replace(run, results=run.results[1:]))

    with pytest.raises(
        DataError,
        match="the run of the system 'alpha' over the suite 'golden' has no result "
        "for a direction that ran",
    ):
        report_of(runs)


def test_a_result_of_a_direction_that_did_not_run_is_a_data_error() -> None:
    def add(run: RunInput) -> RunInput:
        stray = replace(run.results[0], direction="eng_Latn-rus_Cyrl")
        return replace(run, results=(*run.results, stray))

    with pytest.raises(DataError, match="a result for a direction that did not run"):
        report_of(alpha_with(add))


def test_two_results_of_one_direction_are_a_data_error() -> None:
    runs = alpha_with(lambda run: replace(run, results=(*run.results, run.results[0])))

    with pytest.raises(DataError, match="two results"):
        report_of(runs)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"system": "beta"}, "the result is of another system"),
        ({"suite": "other"}, "the result is of another suite"),
        (
            {"provenance": "suite"},
            "the result was scored over another suite file than the run",
        ),
        (
            {"provenance": "outputs"},
            "the result was scored over other outputs than the run wrote",
        ),
        (
            {"provenance": "system"},
            "the result names another system record than the run",
        ),
    ],
)
def test_a_result_that_names_another_system_suite_or_file_than_its_run_is_refused(
    changes: dict[str, Any], match: str
) -> None:
    def change(run: RunInput) -> RunInput:
        result = run.results[0]
        update = dict(changes)
        if "provenance" in update:
            kind = update.pop("provenance")
            old = result.provenance
            if kind == "suite":
                update["provenance"] = replace(old, suite_sha256="e5" * 32)
            elif kind == "outputs":
                update["provenance"] = replace(old, outputs_sha256="e5" * 32)
            else:
                system = SystemRecord(
                    kind="hypotheses",
                    name="alpha",
                    options={"beam_size": 9},
                    outputs_sha256=old.system.outputs_sha256,
                )
                update["provenance"] = Provenance(
                    old.suite_sha256, old.outputs_sha256, system
                )
        if update.get("system") == "beta":
            update["provenance"] = replace(
                result.provenance,
                system=replace(result.provenance.system, name="beta"),
            )
        return with_result(run, result.direction, **update)

    with pytest.raises(DataError, match=match):
        report_of(alpha_with(change))


def test_a_result_whose_segment_count_is_not_the_suites_is_a_data_error() -> None:
    runs = alpha_with(
        lambda run: with_result(run, FIRST, n_segments=11, statuses={"ok": 11})
    )

    with pytest.raises(DataError, match="does not match its outputs: the segments"):
        report_of(runs)


def test_a_result_whose_unit_is_not_the_one_of_the_suite_is_a_data_error() -> None:
    def change(run: RunInput) -> RunInput:
        result = run.results[0]
        bootstrap = replace(result.bootstrap, unit="segment", units=10)
        return with_result(run, result.direction, bootstrap=bootstrap)

    with pytest.raises(
        DataError, match="does not match its outputs: the bootstrap unit"
    ):
        report_of(alpha_with(change))


@pytest.mark.parametrize("metric", ["bleu", "chrf", "chrf_pp"])
def test_a_point_that_the_outputs_do_not_score_is_a_data_error_never_a_comparison(
    metric: str,
) -> None:
    def change(run: RunInput) -> RunInput:
        result = run.results[0]
        metrics = {**result.metrics, metric: result.metrics[metric] + 1.0}
        return with_result(run, result.direction, metrics=metrics)

    with pytest.raises(
        DataError,
        match=f"the {metric} of the result does not match its outputs",
    ):
        report_of(alpha_with(change))


def test_the_outputs_of_a_run_that_are_not_the_ones_its_results_scored_are_refused() -> (
    None
):
    # `beta`'s results over `alpha`'s outputs: the recomputed scores differ from the results'.
    alpha = golden_run("alpha")
    swapped = replace(golden_run("beta"), outputs=alpha.outputs)

    with pytest.raises(DataError, match="does not match its outputs"):
        report_of([alpha, swapped])


def test_a_point_within_rounding_of_the_recomputed_one_is_accepted() -> None:
    def change(run: RunInput) -> RunInput:
        result = run.results[0]
        metrics = {**result.metrics, "bleu": result.metrics["bleu"] * (1 + 1e-12)}
        return with_result(run, result.direction, metrics=metrics)

    report = report_of(alpha_with(change))

    assert len(report["results"]) == 4


def test_the_comparison_comes_from_the_outputs_not_from_the_figures_of_a_result() -> (
    None
):
    # Outputs that do not match the results make the report refuse; they never make a comparison.
    swapped = replace(golden_run("beta"), outputs=golden_run("alpha").outputs)

    with pytest.raises(DataError):
        report_of([golden_run("alpha"), swapped])


def test_a_system_option_that_is_a_path_is_refused_by_the_report() -> None:
    def named(options: dict[str, Any]) -> RunInput:
        run = golden_run("alpha")
        system = replace(run.record.system, options=options)
        results = tuple(
            replace(r, provenance=replace(r.provenance, system=system))
            for r in run.results
        )
        return replace(run, record=replace(run.record, system=system), results=results)

    with pytest.raises(DataError, match="field 'systems.0': holds an absolute path"):
        report_of([named({"model_dir": "/home/user/model"})], baseline=None)


# ---- the robustness embed ----------------------------------------------------------------------


def alpha_with_robustness(**options: Any) -> RunInput:
    return run_input(
        golden_suite(),
        file_system("alpha"),
        robustness=robustness_input("alpha", **options),
    )


def test_the_robustness_block_comes_with_its_coverage_and_its_source() -> None:
    report = report_of([alpha_with_robustness()], baseline=None)

    entry = report["robustness"]["alpha"]
    assert sorted(entry) == ["block", "coverage", "source"]
    assert entry["block"]["numbers_acc"] == 0.9292
    assert entry["coverage"]["placeholder_acc"] == {
        "scored": 599,
        "excluded": 1,
        "total": 600,
    }
    assert entry["coverage"]["mong_source_retention"]["chrf_cyrl"] == 91.25
    assert entry["source"] == {
        "binding": "name",
        "report_sha256": "17" * 32,
        "robustness_sha256": "f6" * 32,
        "suite": "robust",
        "suite_sha256": "e5" * 32,
    }
    for result in report["results"]:
        assert "robustness" not in result
        assert "robustness_coverage" not in result
        assert "robustness_source" not in result


def test_the_same_robustness_run_in_two_suites_is_one_entry() -> None:
    other = without_documents(golden_suite(), name="other")
    runs = [
        alpha_with_robustness(),
        run_input(other, file_system("alpha"), robustness=robustness_input("alpha")),
    ]

    report = report_of(
        runs, suites={**golden_suites(), suite_digest(other): other}, baseline=None
    )

    assert list(report["robustness"]) == ["alpha"]


def test_two_runs_of_one_system_with_different_blocks_are_a_data_error() -> None:
    other = without_documents(golden_suite(), name="other")
    different = {
        "numbers_acc": 0.5,
        "placeholder_acc": 0.5,
        "term_acc": 0.5,
        "script_purity": 0.5,
        "entity_acc": 0.5,
        "mong_source_retention": 0.5,
    }
    runs = [
        alpha_with_robustness(),
        run_input(
            other,
            file_system("alpha"),
            robustness=robustness_input("alpha", block=different),
        ),
    ]

    with pytest.raises(
        DataError, match="the system 'alpha' carries different robustness blocks"
    ):
        report_of(
            runs, suites={**golden_suites(), suite_digest(other): other}, baseline=None
        )


def test_the_results_of_one_run_that_disagree_about_the_block_are_a_data_error() -> (
    None
):
    run = alpha_with_robustness()
    first, second = run.results
    other_block = {**first.robustness, "numbers_acc": 0.1}
    changed = replace(second, robustness=other_block)

    with pytest.raises(
        DataError,
        match="the results of the run of the system 'alpha' over the suite 'golden' "
        "do not carry the same robustness block",
    ):
        report_of([replace(run, results=(first, changed))], baseline=None)


def test_a_result_whose_robustness_source_is_not_the_one_of_its_run_is_a_data_error() -> (
    None
):
    run = alpha_with_robustness()
    record = replace(
        run.record,
        robustness_source={**run.record.robustness_source, "report_sha256": "99" * 32},
    )

    with pytest.raises(DataError, match="the robustness source of the run"):
        report_of([replace(run, record=record)], baseline=None)


def test_a_run_that_says_it_has_a_block_but_whose_results_have_none_is_a_data_error() -> (
    None
):
    plain = golden_run("alpha")
    with_block = alpha_with_robustness()
    record = replace(
        plain.record, robustness_source=with_block.record.robustness_source
    )

    with pytest.raises(DataError, match="the robustness source of the run"):
        report_of([replace(plain, record=record)], baseline=None)


def test_a_direction_left_out_by_the_selection_is_a_state_and_has_no_comparison() -> (
    None
):
    run = run_input(golden_suite(), file_system("alpha"), selection=[FIRST])

    report = report_of([run, golden_run("beta")], baseline="alpha")

    states = {(c["system"], c["direction"]): c["state"] for c in report["coverage"]}
    assert states[("alpha", SECOND)] == "not_selected"
    assert {c["direction"] for c in report["comparisons"]} == {FIRST}
    assert len(report["results"]) == 3


# ---- a run that does not fit the suite it was given -------------------------------------------


def test_a_suite_of_another_name_than_the_run_names_is_a_data_error() -> None:
    suite = golden_suite()
    renamed = replace(suite, name="other")

    with pytest.raises(
        DataError,
        match="the run of the system 'alpha' over the suite 'golden' was matched to a "
        "suite file of another name",
    ):
        report_of(
            [golden_run("alpha")], suites={suite_digest(suite): renamed}, baseline=None
        )


def test_a_suite_that_declares_other_directions_than_the_run_lists_is_a_data_error() -> (
    None
):
    suite = golden_suite()
    reordered = replace(suite, directions=(SECOND, FIRST))

    with pytest.raises(
        DataError,
        match="the run of the system 'alpha' over the suite 'golden' lists other "
        "directions than the suite declares",
    ):
        report_of(
            [golden_run("alpha")],
            suites={suite_digest(suite): reordered},
            baseline=None,
        )


@pytest.mark.parametrize("name", ["", "a b", "x" * 65, "a/b"])
def test_a_baseline_that_is_not_a_system_name_is_a_config_error(name: str) -> None:
    with pytest.raises(ConfigError, match="the baseline must be a system name"):
        report_of(golden_runs(), baseline=name)


def test_a_run_whose_system_name_is_not_an_id_is_a_data_error() -> None:
    run = golden_run("alpha")
    odd = replace(run.record.system, name="not an id")
    record = replace(run.record, system=odd)

    with pytest.raises(DataError, match="a run names a system that is not an id"):
        report_of([replace(run, record=record)], baseline=None)
