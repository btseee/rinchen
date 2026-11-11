"""Property tests of the report of D13 (Task 8): for random sets of runs over the golden suite (a
random system holding a random part of the directions, with random outputs) the report keeps
its promises: one coverage row for every system and declared direction, a verdict that is the
p-value against alpha, a delta that is the difference of the two results' figures, the same bytes
whatever the order of the runs, and a document the strict reader accepts and the Markdown renders.
Under the `no_network` guard like every D13 test module."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.config.eval_metrics import load_eval_metrics_config
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.report import RunInput, build_report, report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_read import report_from_bytes
from tests.support.eval_fakes import golden_suite
from tests.support.eval_golden import (
    ALPHA_ENG,
    ALPHA_KHK,
    BETA_ENG,
    BETA_KHK,
    GOLDEN_ENG,
    GOLDEN_IDS,
    GOLDEN_KHK,
)
from tests.support.eval_report import (
    FIRST,
    SECOND,
    SHIPPED_CONFIG,
    file_system,
    golden_suites,
    run_input,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=25,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
FAST = BootstrapSettings(resamples=100, seed=20_261_110, alpha=0.05)
CONFIG = load_eval_metrics_config(
    SHIPPED_CONFIG.read_text("utf-8")
    .replace("resamples: 1000", "resamples: 100")
    .encode()
)
SOURCES = {
    FIRST: (GOLDEN_ENG, ALPHA_ENG, BETA_ENG, ("",) * 10),
    SECOND: (GOLDEN_KHK, ALPHA_KHK, BETA_KHK, ("",) * 10),
}
SYSTEM = st.tuples(
    st.sets(st.sampled_from((FIRST, SECOND)), min_size=1),
    st.lists(st.integers(0, 3), min_size=20, max_size=20),
)


def outputs_of(directions: set[str], picks: list[int]) -> dict[OutputKey, Output]:
    held: dict[OutputKey, Output] = {}
    cells = iter(picks)
    for direction in (FIRST, SECOND):
        for row_id, index in zip(GOLDEN_IDS, range(10), strict=True):
            pick = next(cells)
            if direction in directions:
                held[OutputKey(direction, row_id)] = Output(
                    SOURCES[direction][pick][index], "ok"
                )
    return held


def runs_of(systems: list[tuple[set[str], list[int]]]) -> list[RunInput]:
    suite = golden_suite()
    return [
        run_input(suite, file_system(f"s{number}", outputs_of(*system)), settings=FAST)
        for number, system in enumerate(systems)
    ]


def report_of(runs: list[RunInput], baseline: str | None = "s0") -> dict[str, object]:
    return build_report(runs, golden_suites(), baseline=baseline, config=CONFIG)


@given(st.lists(SYSTEM, min_size=2, max_size=3))
@SETTINGS
def test_a_coverage_row_for_every_system_and_declared_direction(
    systems: list[tuple[set[str], list[int]]],
) -> None:
    report = report_of(runs_of(systems))

    rows = [(c["system"], c["direction"]) for c in report["coverage"]]  # type: ignore[index, union-attr]
    expected = list(
        itertools.product([f"s{i}" for i in range(len(systems))], (FIRST, SECOND))
    )
    assert sorted(rows) == sorted(expected)
    for coverage in report["coverage"]:  # type: ignore[union-attr]
        held = coverage["direction"] in systems[int(coverage["system"][1:])][0]
        assert (coverage["state"] == "run") == held
        assert (coverage["reason"] is None) == held


@given(st.lists(SYSTEM, min_size=2, max_size=3))
@SETTINGS
def test_significant_is_the_p_value_below_alpha_with_the_delta_in_favour(
    systems: list[tuple[set[str], list[int]]],
) -> None:
    report = report_of(runs_of(systems))

    for comparison in report["comparisons"]:  # type: ignore[union-attr]
        p_value = comparison["p_value"]
        expected = p_value is not None and p_value < 0.05 and comparison["delta"] > 0
        assert comparison["significant"] is expected


@given(st.lists(SYSTEM, min_size=2, max_size=3))
@SETTINGS
def test_a_delta_is_the_difference_of_the_two_results_figures(
    systems: list[tuple[set[str], list[int]]],
) -> None:
    report = report_of(runs_of(systems))
    results = {
        (r["system"], r["direction"]): r["metrics"]
        for r in report["results"]  # type: ignore[union-attr]
    }

    for comparison in report["comparisons"]:  # type: ignore[union-attr]
        candidate = results[(comparison["system"], comparison["direction"])]
        baseline = results[(comparison["baseline"], comparison["direction"])]
        difference = candidate[comparison["metric"]] - baseline[comparison["metric"]]
        assert comparison["delta"] == pytest.approx(difference, abs=1e-9)


@given(st.lists(SYSTEM, min_size=2, max_size=3), st.randoms(use_true_random=False))
@SETTINGS
def test_the_bytes_do_not_depend_on_the_order_of_the_runs_and_are_read_and_rendered(
    systems: list[tuple[set[str], list[int]]], random: object
) -> None:
    runs = runs_of(systems)
    shuffled = list(runs)
    random.shuffle(shuffled)  # type: ignore[attr-defined]

    report = report_of(runs)

    assert report_to_bytes(report_of(shuffled)) == report_to_bytes(report)
    view = report_from_bytes(report_to_bytes(report))
    assert len(view.results) == len(report["results"])  # type: ignore[arg-type]
    assert render_markdown(report) == render_markdown(report_of(shuffled))


@given(st.lists(SYSTEM, min_size=2, max_size=3))
@SETTINGS
def test_a_comparison_exists_exactly_where_both_systems_ran_the_direction(
    systems: list[tuple[set[str], list[int]]],
) -> None:
    report = report_of(runs_of(systems))

    found = {
        (c["system"], c["direction"], c["metric"])
        for c in report["comparisons"]  # type: ignore[union-attr]
    }
    expected = {
        (f"s{number}", direction, metric)
        for number, (directions, _) in enumerate(systems)
        if number > 0
        for direction in directions & systems[0][0]
        for metric in ("chrf_pp", "bleu")
    }
    assert found == expected
