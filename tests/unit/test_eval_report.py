"""The report document of D13 (Task 8, spec T14): `build_report` turns runs and the suites they
scored into `rinchen.report/1`, one JSON document that holds what was measured, how it compares and
where every number came from.

The figures are Task 2's and Task 6's golden tables (derived with the pure-Python reference of the
design spike) and, for the BLEU comparisons, the same reference run again for this task: the
paired comparison of `beta` against `alpha` at the DOCUMENT unit (the golden suite has five
documents), and `gamma`, `alpha` with three rows taken from `beta`, at the SEGMENT unit, where the
interval of the first chrF++ comparison excludes zero while its p-value 52/1001 is above 0.05. The
comparisons are recomputed from the suite and the outputs, never read from a result.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from rinchen.eval.outputs import Output
from rinchen.eval.report import (
    MULTIPLE_COMPARISONS_NOTE,
    REPORT_SCHEMA,
    build_report,
    multiple_comparisons_note,
    report_to_bytes,
)
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_read import report_from_bytes
from tests.support.eval_fakes import FakeSuiteSystem, golden_outputs, golden_suite
from tests.support.eval_golden import (
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_PAIRED,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SIGNATURES,
)
from tests.support.eval_report import (
    FIRST,
    MANIFEST_SHA256,
    MODEL_SHA256,
    ROBUSTNESS_BLOCK,
    SECOND,
    THIRD,
    file_system,
    golden_run,
    golden_runs,
    golden_suites,
    mixed_outputs,
    rinchen_system,
    robustness_input,
    run_input,
    shipped_config,
    suite_digest,
    with_third_direction,
    without_documents,
)
from tests.support.eval_report_golden import (
    GAMMA_PAIRED,
    GOLDEN_PAIRED_BLEU,
    STANDARD_DIRECTIONS,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

ABSOLUTE = re.compile(r"^(/|[A-Za-z]:[\\/])")
RESULT_KEYS = (
    "bleu_signature",
    "bootstrap",
    "chrf_pp_signature",
    "chrf_signature",
    "ci95",
    "direction",
    "manifest_sha",
    "metrics",
    "n_segments",
    "provenance",
    "schema",
    "statuses",
    "suite",
    "system",
    "trained",
    "unmeasured",
)


def golden_report(**options: Any) -> dict[str, Any]:
    return build_report(
        golden_runs(),
        golden_suites(),
        baseline=options.pop("baseline", "alpha"),
        config=shipped_config(),
    )


def strings_of(node: object) -> Iterator[str]:
    """Every string of a document, the keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from strings_of(value)
    elif isinstance(node, list):
        for item in node:
            yield from strings_of(item)


def keys_of(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from keys_of(value)
    elif isinstance(node, list):
        for item in node:
            yield from keys_of(item)


# ---- the document of the golden runs ----------------------------------------------------------


def test_the_report_has_exactly_the_keys_of_the_format() -> None:
    report = golden_report()

    assert sorted(report) == [
        "bootstrap",
        "comparisons",
        "coverage",
        "evidence",
        "note_multiple_comparisons",
        "results",
        "robustness",
        "runs",
        "schema",
        "suites",
        "systems",
        "targets",
    ]
    assert report["schema"] == REPORT_SCHEMA == "rinchen.report/1"


def test_a_fixture_suite_makes_the_evidence_fixture_only() -> None:
    assert golden_report()["evidence"] == {"kinds": ["fixture"], "fixture_only": True}


def test_the_evidence_lists_the_kinds_sorted_and_is_not_fixture_only_with_a_benchmark() -> (
    None
):
    benchmark = replace(
        golden_suite(), name="bench", kind="benchmark", benchmark="flores_plus"
    )
    runs = [*golden_runs(), run_input(benchmark, file_system("alpha"))]
    suites = {**golden_suites(), suite_digest(benchmark): benchmark}

    report = build_report(runs, suites, baseline=None, config=shipped_config())

    assert report["evidence"] == {
        "kinds": ["benchmark", "fixture"],
        "fixture_only": False,
    }


def test_the_systems_are_two_entries_sorted_by_name() -> None:
    systems = golden_report()["systems"]

    assert systems == [
        {
            "name": name,
            "kind": "hypotheses",
            "model_sha256": None,
            "manifest_sha256": None,
            "options": {},
        }
        for name in ("alpha", "beta")
    ]


def test_the_systems_are_sorted_whatever_the_order_of_the_runs() -> None:
    forward = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    backward = build_report(
        list(reversed(golden_runs())),
        golden_suites(),
        baseline="alpha",
        config=shipped_config(),
    )

    assert report_to_bytes(forward) == report_to_bytes(backward)
    assert [s["name"] for s in backward["systems"]] == ["alpha", "beta"]


def test_the_suite_is_described_with_its_digest_and_a_row_for_every_expected_direction() -> (
    None
):
    (suite,) = golden_report()["suites"]

    assert {key: suite[key] for key in suite if key != "directions"} == {
        "name": "golden",
        "kind": "fixture",
        "benchmark": "golden",
        "split": "devtest",
        "purpose": "report",
        "sha256": suite_digest(golden_suite()),
        "rows": 10,
        "source": {
            "dataset": None,
            "revision": None,
            "pinned": False,
            "missing_languages": [],
        },
    }
    assert [d["direction"] for d in suite["directions"]] == list(STANDARD_DIRECTIONS)
    assert {d["direction"]: d["rows"] for d in suite["directions"]} == {
        **dict.fromkeys(STANDARD_DIRECTIONS),
        FIRST: 10,
        SECOND: 10,
    }


def test_a_declared_direction_that_is_not_standard_follows_the_standard_ones() -> None:
    suite = with_third_direction(golden_suite())
    runs = [run_input(suite, file_system("alpha"))]

    report = build_report(
        runs, {suite_digest(suite): suite}, baseline=None, config=shipped_config()
    )

    directions = report["suites"][0]["directions"]
    assert [d["direction"] for d in directions] == [*STANDARD_DIRECTIONS, THIRD]
    assert directions[-1] == {"direction": THIRD, "rows": 10}


def test_the_coverage_has_a_row_for_every_system_and_declared_direction() -> None:
    coverage = golden_report()["coverage"]

    assert coverage == [
        {
            "suite": "golden",
            "system": system,
            "direction": direction,
            "state": "run",
            "reason": None,
        }
        for system in ("alpha", "beta")
        for direction in (FIRST, SECOND)
    ]


def test_a_direction_that_did_not_run_is_a_coverage_row_with_its_state_and_reason() -> (
    None
):
    suite = with_third_direction(golden_suite())
    runs = [
        run_input(suite, file_system("alpha")),
        run_input(suite, file_system("beta")),
    ]

    report = build_report(
        runs, {suite_digest(suite): suite}, baseline="alpha", config=shipped_config()
    )

    states = [(c["system"], c["direction"], c["state"]) for c in report["coverage"]]
    assert states == [
        ("alpha", FIRST, "run"),
        ("alpha", SECOND, "run"),
        ("alpha", THIRD, "not_in_file"),
        ("beta", FIRST, "run"),
        ("beta", SECOND, "run"),
        ("beta", THIRD, "not_in_file"),
    ]
    third = report["coverage"][2]
    assert third["reason"] == "the file holds no output for the direction"
    assert len(report["results"]) == 4
    assert {c["direction"] for c in report["comparisons"]} == {FIRST, SECOND}


def test_the_results_are_the_four_results_without_their_robustness_fields() -> None:
    results = golden_report()["results"]

    assert [(r["system"], r["direction"]) for r in results] == [
        ("alpha", FIRST),
        ("alpha", SECOND),
        ("beta", FIRST),
        ("beta", SECOND),
    ]
    for result in results:
        assert sorted(result) == list(RESULT_KEYS)
        assert result["suite"] == "golden"
        assert result["n_segments"] == 10
        assert result["trained"] is True
        assert result["statuses"] == {"ok": 10}
        assert result["bleu_signature"] == GOLDEN_SIGNATURES["bleu"]
        assert result["chrf_pp_signature"] == GOLDEN_SIGNATURES["chrf_pp"]
        assert result["bootstrap"]["unit"] == "document"
        assert result["bootstrap"]["units"] == 5


@pytest.mark.parametrize("system", ["alpha", "beta"])
@pytest.mark.parametrize("direction", [FIRST, SECOND])
def test_every_result_holds_the_golden_points_and_document_intervals(
    system: str, direction: str
) -> None:
    result = next(
        r
        for r in golden_report()["results"]
        if r["system"] == system and r["direction"] == direction
    )

    for metric in ("chrf", "chrf_pp"):
        assert (
            result["metrics"][metric]
            == GOLDEN_SEGMENT_ESTIMATES[(direction, system, metric)].point
        )
        assert result["ci95"][metric] == list(
            GOLDEN_DOCUMENT_INTERVALS[(direction, system, metric)]
        )
    assert result["metrics"]["bleu"] == pytest.approx(
        GOLDEN_SEGMENT_ESTIMATES[(direction, system, "bleu")].point, rel=1e-9
    )
    assert result["ci95"]["bleu"] == pytest.approx(
        list(GOLDEN_DOCUMENT_INTERVALS[(direction, system, "bleu")]), rel=1e-9
    )
    for metric in ("comet22", "metricx24", "metricx24_qe"):
        assert result["metrics"][metric] is None
        assert metric in result["unmeasured"]


def test_the_comparisons_are_beta_against_alpha_for_both_directions_and_both_metrics() -> (
    None
):
    comparisons = golden_report()["comparisons"]

    assert [(c["direction"], c["metric"]) for c in comparisons] == [
        (FIRST, "chrf_pp"),
        (FIRST, "bleu"),
        (SECOND, "chrf_pp"),
        (SECOND, "bleu"),
    ]
    for comparison in comparisons:
        assert sorted(comparison) == [
            "baseline",
            "better_fraction",
            "ci95",
            "delta",
            "direction",
            "higher_is_better",
            "metric",
            "p_value",
            "significant",
            "suite",
            "system",
            "unit",
            "units",
        ]
        assert comparison["suite"] == "golden"
        assert comparison["system"] == "beta"
        assert comparison["baseline"] == "alpha"
        assert comparison["higher_is_better"] is True
        assert comparison["significant"] is True
        assert comparison["unit"] == "document"
        assert comparison["units"] == 5


@pytest.mark.parametrize("direction", [FIRST, SECOND])
def test_the_chrf_pp_comparison_is_the_golden_paired_row_of_task_2(
    direction: str,
) -> None:
    golden = GOLDEN_PAIRED[(direction, "document")]
    comparison = next(
        c
        for c in golden_report()["comparisons"]
        if c["direction"] == direction and c["metric"] == "chrf_pp"
    )

    assert comparison["delta"] == golden.delta
    assert comparison["ci95"] == [golden.low, golden.high]
    assert comparison["p_value"] == golden.p_value
    assert comparison["better_fraction"] == golden.better_fraction


@pytest.mark.parametrize("direction", [FIRST, SECOND])
def test_the_bleu_comparison_is_the_reference_derivation(direction: str) -> None:
    golden = GOLDEN_PAIRED_BLEU[direction]
    comparison = next(
        c
        for c in golden_report()["comparisons"]
        if c["direction"] == direction and c["metric"] == "bleu"
    )

    assert comparison["delta"] == pytest.approx(golden.delta, rel=1e-9)
    assert comparison["ci95"] == pytest.approx([golden.low, golden.high], rel=1e-9)
    assert comparison["p_value"] == golden.p_value
    assert comparison["better_fraction"] == golden.better_fraction


def test_identical_systems_give_delta_zero_an_interval_of_zero_and_p_one() -> None:
    # The control sacrebleu's own paired test fails: the two systems share one set of draws, so
    # nothing moves the difference.
    twin = file_system("twin", golden_outputs("alpha"))
    runs = [golden_run("alpha"), run_input(golden_suite(), twin)]

    report = build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )

    assert len(report["comparisons"]) == 4
    for comparison in report["comparisons"]:
        assert comparison["delta"] == 0.0
        assert comparison["ci95"] == [0.0, 0.0]
        assert comparison["p_value"] == 1.0
        assert comparison["significant"] is False
        assert comparison["better_fraction"] == 0.0


def test_the_bootstrap_of_the_report_is_the_configuration() -> None:
    assert golden_report()["bootstrap"] == {
        "resamples": 1000,
        "seed": 20_261_110,
        "alpha": 0.05,
        "interval": "percentile",
        "rng": "splitmix64-counter/1",
    }


def test_the_note_gives_the_counts_of_what_the_report_holds() -> None:
    report = golden_report()

    # twelve intervals of the results (three count metrics, four results) and the four of the
    # comparisons; four paired comparisons.
    assert report["note_multiple_comparisons"] == (
        "16 intervals and 4 paired comparisons are reported with no correction for "
        "multiple comparisons"
    )


def test_the_note_template_is_a_function_of_the_two_counts() -> None:
    assert multiple_comparisons_note(3, 2) == (
        "3 intervals and 2 paired comparisons are reported with no correction for "
        "multiple comparisons"
    )
    assert MULTIPLE_COMPARISONS_NOTE.format(intervals=3, comparisons=2) == (
        multiple_comparisons_note(3, 2)
    )


def test_a_report_without_a_baseline_has_no_comparison_and_counts_only_intervals() -> (
    None
):
    report = build_report(
        golden_runs(), golden_suites(), baseline=None, config=shipped_config()
    )

    assert report["comparisons"] == []
    assert report["note_multiple_comparisons"] == multiple_comparisons_note(12, 0)


def test_the_baseline_of_the_configuration_stands_in_when_none_is_given() -> None:
    config = shipped_config().model_copy(update={"baseline_system": "alpha"})

    report = build_report(golden_runs(), golden_suites(), baseline=None, config=config)

    assert {c["baseline"] for c in report["comparisons"]} == {"alpha"}


def test_the_runs_are_named_by_suite_system_and_the_digest_of_their_run_json() -> None:
    runs = golden_runs()

    report = build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )

    assert report["runs"] == [
        {"suite": "golden", "system": "alpha", "run_sha256": runs[0].run_sha256},
        {"suite": "golden", "system": "beta", "run_sha256": runs[1].run_sha256},
    ]


def test_the_targets_of_the_golden_report_are_the_four_unmeasured_ones() -> None:
    targets = golden_report()["targets"]

    assert [(t["id"], t["status"]) for t in targets] == [
        ("beats_baseline", "unmeasured"),
        ("off_target", "unmeasured"),
        ("numbers", "unmeasured"),
        ("placeholders", "unmeasured"),
    ]
    assert targets[0]["basis"] == "0 of 16 directions measured"
    assert targets[0]["threshold"] == 12 and targets[0]["value"] is None


def test_a_robustness_run_of_a_model_of_ours_that_meets_the_numbers_target() -> None:
    system = rinchen_system("toy", golden_outputs("beta"))
    block = {**ROBUSTNESS_BLOCK, "numbers_acc": 0.995, "placeholder_acc": 1.0}
    inp = robustness_input(
        "toy", block=block, model_sha256=MODEL_SHA256, manifest_sha256=MANIFEST_SHA256
    )
    runs = [golden_run("alpha"), run_input(golden_suite(), system, robustness=inp)]

    report = build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )

    targets = {t["id"]: t for t in report["targets"]}
    assert targets["numbers"]["status"] == "met"
    assert targets["numbers"]["value"] == 0.995
    assert targets["placeholders"]["status"] == "unmeasured"  # one unit was left out
    assert report["robustness"]["toy"]["block"] == block


# ---- what the bytes are ------------------------------------------------------------------------


def test_the_report_bytes_are_sorted_indented_and_end_with_one_line_feed() -> None:
    report = golden_report()

    data = report_to_bytes(report)

    expected = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    assert data == f"{expected}\n".encode()
    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert b"\r" not in data


def test_the_report_built_twice_has_equal_bytes() -> None:
    assert report_to_bytes(golden_report()) == report_to_bytes(golden_report())


def test_the_bytes_keep_a_cyrillic_character_as_text() -> None:
    document = {"reason": "муур"}

    assert "муур".encode() in report_to_bytes(document)


def test_a_value_that_is_not_a_finite_number_cannot_be_written() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        report_to_bytes({"delta": float("nan")})


def test_the_report_holds_no_timestamp_key_and_no_absolute_path() -> None:
    report = golden_report()

    assert "created" not in set(keys_of(report))
    assert [s for s in strings_of(report) if ABSOLUTE.match(s)] == []


# ---- the verdict is the p-value, never the interval --------------------------------------------


def gamma_report() -> dict[str, Any]:
    plain = without_documents(golden_suite())
    runs = [
        run_input(plain, file_system("alpha")),
        run_input(plain, file_system("gamma", mixed_outputs())),
    ]
    return build_report(
        runs, {suite_digest(plain): plain}, baseline="alpha", config=shipped_config()
    )


def test_the_interval_excludes_zero_while_the_p_value_says_no() -> None:
    comparisons = gamma_report()["comparisons"]

    assert len(comparisons) == 4
    for comparison in comparisons:
        golden = GAMMA_PAIRED[(comparison["direction"], comparison["metric"])]
        assert comparison["unit"] == "segment" and comparison["units"] == 10
        assert comparison["delta"] == pytest.approx(golden.delta, rel=1e-9)
        assert comparison["ci95"] == pytest.approx([golden.low, golden.high], rel=1e-9)
        assert comparison["p_value"] == golden.p_value == 0.05194805194805195
        assert comparison["better_fraction"] == golden.better_fraction == 0.975
        # the interval is above zero ...
        assert comparison["ci95"][0] > 0
        # ... and the verdict is still no, because the p-value is above alpha
        assert comparison["significant"] is False


def test_a_win_with_a_p_value_below_alpha_is_significant() -> None:
    comparisons = golden_report()["comparisons"]

    assert all(c["p_value"] < 0.05 and c["delta"] > 0 for c in comparisons)
    assert all(c["significant"] for c in comparisons)


def test_a_system_that_is_worse_than_the_baseline_is_not_significantly_better() -> None:
    # alpha against beta as the baseline: the delta is negative and p is as small as it gets
    report = build_report(
        golden_runs(), golden_suites(), baseline="beta", config=shipped_config()
    )

    for comparison in report["comparisons"]:
        assert comparison["system"] == "alpha"
        assert comparison["delta"] < 0
        assert comparison["p_value"] == 0.001998001998001998
        assert comparison["better_fraction"] == 0.0
        assert comparison["significant"] is False


def test_a_swap_of_the_two_systems_negates_the_delta_and_leaves_the_p_value() -> None:
    forward = golden_report()["comparisons"]
    backward = build_report(
        golden_runs(), golden_suites(), baseline="beta", config=shipped_config()
    )["comparisons"]

    for one, other in zip(forward, backward, strict=True):
        assert other["delta"] == -one["delta"]
        assert other["ci95"] == [-one["ci95"][1], -one["ci95"][0]]
        assert other["p_value"] == one["p_value"]


# ---- a baseline that is missing ----------------------------------------------------------------


def test_a_baseline_that_has_no_run_in_a_suite_gives_a_coverage_row_and_no_comparison() -> (
    None
):
    other = without_documents(golden_suite(), name="other")
    runs = [*golden_runs(), run_input(other, file_system("beta"))]
    suites = {**golden_suites(), suite_digest(other): other}

    report = build_report(runs, suites, baseline="alpha", config=shipped_config())

    assert [c["suite"] for c in report["comparisons"]] == ["golden"] * 4
    absent = [row for row in report["coverage"] if row["state"] == "absent"]
    assert absent == [
        {
            "suite": "other",
            "system": "alpha",
            "direction": direction,
            "state": "absent",
            "reason": "baseline not in this suite",
        }
        for direction in (FIRST, SECOND)
    ]
    assert [(r["suite"], r["system"]) for r in report["runs"]] == [
        ("golden", "alpha"),
        ("golden", "beta"),
        ("other", "beta"),
    ]


def test_a_baseline_that_did_not_run_a_direction_leaves_that_direction_without_comparison() -> (
    None
):
    suite = with_third_direction(golden_suite())
    only_first = {
        key: output
        for key, output in golden_outputs("alpha").items()
        if key.direction == FIRST
    }
    runs = [
        run_input(suite, file_system("alpha", only_first)),
        run_input(suite, file_system("beta")),
    ]

    report = build_report(
        runs, {suite_digest(suite): suite}, baseline="alpha", config=shipped_config()
    )

    assert {c["direction"] for c in report["comparisons"]} == {FIRST}
    alpha_rows = {
        c["direction"]: c["state"] for c in report["coverage"] if c["system"] == "alpha"
    }
    assert alpha_rows[SECOND] == "not_in_file"


def test_the_document_has_one_coverage_row_for_every_run_and_declared_direction() -> (
    None
):
    suite = with_third_direction(golden_suite())
    runs = [run_input(suite, file_system(name)) for name in ("alpha", "beta")]

    report = build_report(
        runs, {suite_digest(suite): suite}, baseline="alpha", config=shipped_config()
    )

    assert len(report["coverage"]) == len(runs) * len(suite.directions)
    assert (
        len({(c["suite"], c["system"], c["direction"]) for c in report["coverage"]})
        == 6
    )


# ---- the release targets from a report -----------------------------------------------------------


def flores_suite() -> Any:
    """A FLORES+ devtest suite of the sixteen standard directions, six rows, no document."""
    from rinchen.eval.suites import Suite, SuiteRow, SuiteSource

    languages = {
        code for direction in STANDARD_DIRECTIONS for code in direction.split("-")
    }
    rows = tuple(
        SuiteRow(
            f"f{index:02d}",
            None,
            {
                code: f"{code} sentence {index} with some words"
                for code in sorted(languages)
            },
        )
        for index in range(1, 7)
    )
    return Suite(
        name="flores",
        kind="benchmark",
        benchmark="flores_plus",
        split="devtest",
        source=SuiteSource(None, None, False, {}, ()),
        directions=STANDARD_DIRECTIONS,
        rows=rows,
    )


def flores_report(wins: int) -> dict[str, Any]:
    from rinchen.eval.systems import SystemRecord
    from tests.support.eval_report import MANIFEST_SHA256, MODEL_SHA256

    suite = flores_suite()
    winning = set(STANDARD_DIRECTIONS[:wins])

    def toy(requests: Any) -> list[Output]:
        return [
            Output(
                suite.view(r.direction).references[
                    suite.view(r.direction).ids.index(r.id)
                ]
                if r.direction in winning
                else "x",
                "ok",
            )
            for r in requests
        ]

    toy_system = FakeSuiteSystem(
        answer=toy,
        record=SystemRecord(
            kind="rinchen",
            name="toy",
            model_sha256=MODEL_SHA256,
            manifest_sha256=MANIFEST_SHA256,
        ),
    )
    nllb = FakeSuiteSystem(
        answer=lambda requests: [Output("x", "ok") for _ in requests],
        record=SystemRecord(kind="baseline", name="nllb"),
    )
    runs = [run_input(suite, toy_system), run_input(suite, nllb)]
    return build_report(
        runs, {suite_digest(suite): suite}, baseline="nllb", config=shipped_config()
    )


@pytest.mark.parametrize(
    ("wins", "status", "value"),
    [(12, "met", 12), (16, "met", 16), (11, "not_met", 11), (0, "not_met", 0)],
)
def test_the_baseline_target_follows_the_significant_wins_over_a_flores_devtest_suite(
    wins: int, status: str, value: int
) -> None:
    report = flores_report(wins)

    target = next(t for t in report["targets"] if t["id"] == "beats_baseline")

    comparisons = [c for c in report["comparisons"] if c["metric"] == "chrf_pp"]
    assert len(comparisons) == 16
    assert sum(c["significant"] for c in comparisons) == wins
    assert (target["status"], target["value"]) == (status, value)
    assert target["threshold"] == 12


def test_a_suite_named_like_a_directory_is_only_a_name_in_the_report() -> None:
    for name in (".", ".."):
        suite = without_documents(golden_suite(), name=name)
        runs = [
            run_input(suite, file_system("alpha")),
            run_input(suite, file_system("beta")),
        ]

        report = build_report(
            runs,
            {suite_digest(suite): suite},
            baseline="alpha",
            config=shipped_config(),
        )

        assert report["suites"][0]["name"] == name
        assert {c["suite"] for c in report["comparisons"]} == {name}
        assert f"### {name}" in render_markdown(report)


# ---- a comparison with fewer than two units --------------------------------------------------


def one_row_report() -> dict[str, Any]:
    suite = golden_suite()
    single = replace(suite, rows=suite.rows[1:2])
    runs = [
        run_input(single, file_system("alpha")),
        run_input(single, file_system("beta")),
    ]
    return build_report(
        runs, {suite_digest(single): single}, baseline="alpha", config=shipped_config()
    )


def test_one_unit_gives_a_point_and_no_interval_p_value_or_verdict() -> None:
    report = one_row_report()

    assert len(report["comparisons"]) == 4
    for comparison in report["comparisons"]:
        assert comparison["units"] == 1
        assert comparison["delta"] > 0
        assert comparison["ci95"] is None
        assert comparison["p_value"] is None
        assert comparison["better_fraction"] is None
        assert comparison["significant"] is False
    for result in report["results"]:
        assert result["ci95"]["chrf_pp"] is None


def test_a_report_of_one_unit_is_read_back_and_rendered_with_n_a() -> None:
    report = one_row_report()

    view = report_from_bytes(report_to_bytes(report))
    markdown = render_markdown(report)

    assert all(c.p_value is None and c.ci95 is None for c in view.comparisons)
    assert "| khk_Cyrl-eng_Latn | beta | chrF++ | +" in markdown
    assert " | n/a | n/a | n/a (fewer than two units) |" in markdown
