"""The strict reader of the report of D13 (Task 8, spec T14): `report_from_bytes` turns the bytes of
a `report.json` into a `ReportView`, the one thing the model card and the release stage read.

What it holds: the schema `rinchen.report/1`, exactly the keys of the format and each of its type,
no absolute path in any string (a report that can be uploaded names no local file), every
comparison on a count metric, `significant` the verdict the p-value and the delta give, the
evidence the kinds of the suites, and the note the counts of what the report holds. A refusal is a
`DataError` that names the section (and the index of the entry) and never quotes a value.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.report import build_report, report_to_bytes
from rinchen.eval.report_read import (
    PURPOSES,
    SPLITS,
    SUITE_KINDS,
    ReportView,
    report_from_bytes,
)
from rinchen.eval.result import DirectionResult
from rinchen.eval.suite_rules import KINDS, PURPOSE_OF_SPLIT
from rinchen.eval.suite_rules import SPLITS as SUITE_SPLITS
from tests.support.eval_fakes import golden_suite
from tests.support.eval_report import (
    FIRST,
    file_system,
    golden_runs,
    golden_suites,
    robustness_input,
    run_input,
    shipped_config,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET MARKER 41"


def golden_report() -> dict[str, Any]:
    return build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )


def read(report: dict[str, Any]) -> ReportView:
    return report_from_bytes(report_to_bytes(report))


def edited(change: Callable[[dict[str, Any]], None]) -> bytes:
    report = copy.deepcopy(golden_report())
    change(report)
    return report_to_bytes(report)


# ---- the view ----------------------------------------------------------------------------------


def test_the_view_holds_every_section_of_the_report() -> None:
    report = golden_report()

    view = read(report)

    assert view.evidence.kinds == ("fixture",)
    assert view.evidence.fixture_only is True
    assert [s.name for s in view.systems] == ["alpha", "beta"]
    assert view.systems[0].kind == "hypotheses"
    assert view.systems[0].model_sha256 is None
    assert dict(view.systems[0].options) == {}
    suite = view.suites[0]
    assert (suite.name, suite.kind, suite.benchmark, suite.split, suite.purpose) == (
        "golden",
        "fixture",
        "golden",
        "devtest",
        "report",
    )
    assert suite.sha256 == report["suites"][0]["sha256"] and suite.rows == 10
    assert len(suite.directions) == 16
    assert {d.direction: d.rows for d in suite.directions if d.rows is not None} == {
        FIRST: 10,
        "eng_Latn-khk_Cyrl": 10,
    }
    assert suite.source.missing_languages == ()
    assert suite.source.pinned is False
    assert [(r.suite, r.system, r.run_sha256) for r in view.runs] == [
        (r["suite"], r["system"], r["run_sha256"]) for r in report["runs"]
    ]
    assert [(c.system, c.direction, c.state, c.reason) for c in view.coverage] == [
        (c["system"], c["direction"], c["state"], c["reason"])
        for c in report["coverage"]
    ]
    assert view.robustness == {}
    assert view.bootstrap.resamples == 1000 and view.bootstrap.seed == 20_261_110
    assert view.bootstrap.alpha == 0.05
    assert view.note_multiple_comparisons == report["note_multiple_comparisons"]


def test_the_results_of_the_view_are_the_direction_results_of_the_report() -> None:
    report = golden_report()

    view = read(report)

    assert len(view.results) == 4
    assert all(isinstance(result, DirectionResult) for result in view.results)
    for result, document in zip(view.results, report["results"], strict=True):
        assert (result.system, result.suite, result.direction) == (
            document["system"],
            document["suite"],
            document["direction"],
        )
        assert dict(result.metrics) == document["metrics"]
        assert {k: v for k, v in result.ci95.items()} == document["ci95"]
        assert result.robustness is None and result.robustness_coverage is None


def test_the_comparisons_of_the_view_are_those_of_the_report() -> None:
    report = golden_report()

    view = read(report)

    assert len(view.comparisons) == 4
    for comparison, document in zip(
        view.comparisons, report["comparisons"], strict=True
    ):
        assert comparison.suite == document["suite"]
        assert comparison.direction == document["direction"]
        assert comparison.system == "beta" and comparison.baseline == "alpha"
        assert comparison.metric == document["metric"]
        assert comparison.higher_is_better is True
        assert comparison.delta == document["delta"]
        assert comparison.ci95 == tuple(document["ci95"])
        assert comparison.p_value == document["p_value"]
        assert comparison.better_fraction == document["better_fraction"]
        assert comparison.significant is True
        assert (comparison.unit, comparison.units) == ("document", 5)


def test_the_targets_of_the_view_are_those_of_the_report() -> None:
    report = golden_report()

    view = read(report)

    assert [
        (t.id, t.description, t.threshold, t.value, t.status, t.basis)
        for t in view.targets
    ] == [
        (t["id"], t["description"], t["threshold"], t["value"], t["status"], t["basis"])
        for t in report["targets"]
    ]


def test_a_robustness_block_comes_back_with_its_coverage_and_source() -> None:
    inp = robustness_input("alpha")
    runs = [run_input(golden_suite(), file_system("alpha"), robustness=inp)]
    report = build_report(runs, golden_suites(), baseline=None, config=shipped_config())

    view = read(report)

    entry = view.robustness["alpha"]
    assert dict(entry.block) == report["robustness"]["alpha"]["block"]
    assert entry.coverage["placeholder_acc"]["excluded"] == 1
    assert entry.coverage["mong_source_retention"]["chrf_cyrl"] == 91.25
    assert dict(entry.source) == report["robustness"]["alpha"]["source"]
    assert entry.source["binding"] == "name"


def test_the_view_is_frozen() -> None:
    view = read(golden_report())

    with pytest.raises(FrozenInstanceError):
        view.evidence = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        view.systems[0].name = "x"  # type: ignore[misc]


def test_the_view_of_the_same_bytes_is_equal() -> None:
    data = report_to_bytes(golden_report())

    assert report_from_bytes(data) == report_from_bytes(data)


def test_the_constants_of_the_suite_format_are_those_of_the_suite_rules() -> None:
    # The reader keeps its own copy so that a card never needs the module that reads a suite.
    assert SUITE_KINDS == KINDS
    assert SPLITS == SUITE_SPLITS
    assert PURPOSES == tuple(dict.fromkeys(PURPOSE_OF_SPLIT.values()))


# ---- what is refused ---------------------------------------------------------------------------


def refuse(data: bytes) -> str:
    with pytest.raises(DataError) as caught:
        report_from_bytes(data)
    message = str(caught.value)
    assert message.startswith("not an evaluation report: ")
    assert MARKER not in message
    return message


def test_a_document_that_is_not_json_is_refused() -> None:
    assert "not valid JSON" in refuse(b"{nope")


def test_a_document_that_is_not_utf_8_is_refused() -> None:
    assert "not UTF-8" in refuse(b"\xff\xfe{}")


def test_a_document_that_is_not_an_object_is_refused() -> None:
    assert "not a JSON object" in refuse(b"[1, 2]")


def test_a_repeated_key_is_refused() -> None:
    data = b'{"schema": "rinchen.report/1", "schema": "rinchen.report/1"}'

    assert "appears more than once" in refuse(data)


def test_another_schema_is_refused_naming_the_field() -> None:
    message = refuse(edited(lambda r: r.update(schema="rinchen.report/2")))

    assert "field 'schema'" in message


def test_an_unknown_top_level_key_is_refused() -> None:
    message = refuse(edited(lambda r: r.update(created="2026-11-10")))

    assert "has a field that is not part of the format" in message
    assert "2026-11-10" not in message


def test_a_missing_top_level_key_is_refused_naming_it() -> None:
    message = refuse(edited(lambda r: r.pop("targets")))

    assert "the field 'targets' is missing" in message


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda r: r["evidence"].update(fixture_only="yes"), "evidence.fixture_only"),
        (
            lambda r: r["evidence"].update(kinds=["fixture", "fixture"]),
            "evidence.kinds",
        ),
        (lambda r: r["evidence"].update(kinds=["benchmark"]), "evidence.kinds"),
        (lambda r: r["evidence"].update(fixture_only=False), "evidence.fixture_only"),
        (lambda r: r["systems"][0].update(kind="model"), "systems.0.kind"),
        (lambda r: r["systems"][0].update(name="a b"), "systems.0.name"),
        (lambda r: r["systems"][1].update(name="alpha"), "systems"),
        (
            lambda r: r["systems"][0].update(model_sha256="abc"),
            "systems.0.model_sha256",
        ),
        (lambda r: r.update(systems={}), "systems"),
        (lambda r: r["suites"][0].update(split="train"), "suites.0.split"),
        (lambda r: r["suites"][0].update(purpose="tuning"), "suites.0.purpose"),
        (
            lambda r: r["suites"][0].update(split="dev", purpose="selection"),
            "suites.0.split",
        ),
        (lambda r: r["suites"][0].update(purpose="selection"), "suites.0.purpose"),
        (lambda r: r["suites"][0].update(sha256="x"), "suites.0.sha256"),
        (lambda r: r["suites"][0].update(rows=0), "suites.0.rows"),
        (
            lambda r: r["suites"][0]["directions"][0].update(rows=-1),
            "suites.0.directions.0.rows",
        ),
        (
            lambda r: r["suites"][0]["source"].update(pinned="no"),
            "suites.0.source.pinned",
        ),
        (lambda r: r["runs"][0].update(run_sha256="x"), "runs.0.run_sha256"),
        (lambda r: r["runs"][0].update(system="nobody"), "runs.0.system"),
        (lambda r: r["runs"].append(dict(r["runs"][0])), "runs"),
        (lambda r: r["coverage"][0].update(state="lost"), "coverage.0.state"),
        (lambda r: r["coverage"][0].update(reason="a reason"), "coverage.0.reason"),
        (
            lambda r: r["coverage"][0].update(state="not_trained", reason=None),
            "coverage.0.reason",
        ),
        (
            lambda r: r["coverage"][0].update(direction="khk_Mong-eng_Latn"),
            "coverage.0.direction",
        ),
        (lambda r: r["coverage"][0].update(suite="other"), "coverage.0.suite"),
        (
            lambda r: r["comparisons"][0].update(metric="comet22"),
            "comparisons.0.metric",
        ),
        (lambda r: r["comparisons"][0].update(metric="ter"), "comparisons.0.metric"),
        (
            lambda r: r["comparisons"][1].update(higher_is_better=False),
            "comparisons.1.higher_is_better",
        ),
        (lambda r: r["comparisons"][0].update(delta="big"), "comparisons.0.delta"),
        (lambda r: r["comparisons"][0].update(ci95=[2.0, 1.0]), "comparisons.0.ci95"),
        (lambda r: r["comparisons"][0].update(p_value=1.5), "comparisons.0.p_value"),
        (
            lambda r: r["comparisons"][0].update(significant=False),
            "comparisons.0.significant",
        ),
        (
            lambda r: r["comparisons"][0].update(p_value=0.06),
            "comparisons.0.significant",
        ),
        (lambda r: r["comparisons"][0].update(unit="paragraph"), "comparisons.0.unit"),
        (lambda r: r["comparisons"][0].update(system="nobody"), "comparisons.0.system"),
        (lambda r: r["comparisons"][0].update(p_value=None), "comparisons.0.p_value"),
        (lambda r: r["targets"][0].update(status="passed"), "targets.0.status"),
        (lambda r: r["targets"][0].update(basis=""), "targets.0.basis"),
        (lambda r: r["targets"][1].update(threshold="low"), "targets.1.threshold"),
        (lambda r: r["bootstrap"].update(resamples=10), "bootstrap.resamples"),
        (lambda r: r["bootstrap"].update(alpha=0.7), "bootstrap.alpha"),
        (lambda r: r["bootstrap"].update(rng="mersenne"), "bootstrap.rng"),
        (
            lambda r: r.update(note_multiple_comparisons="16 intervals"),
            "note_multiple_comparisons",
        ),
        (lambda r: r["results"][0]["statuses"].update(ok=9), "results.0"),
        (lambda r: r["results"][1].update(system="nobody"), "results.1"),
        (lambda r: r["results"][2].update(suite="other"), "results.2.suite"),
        (lambda r: r["results"][0].update(robustness=None), "results.0"),
    ],
)
def test_each_refusal_names_the_section_and_the_entry(
    change: Callable[[dict[str, Any]], None], field: str
) -> None:
    message = refuse(edited(change))

    assert field in message


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["systems"][0].update(name=MARKER),
        lambda r: r["systems"][0].update(kind=MARKER),
        lambda r: r["suites"][0].update(split=MARKER),
        lambda r: r["coverage"][0].update(state=MARKER),
        lambda r: r["coverage"][0].update(direction=MARKER),
        lambda r: r["comparisons"][0].update(metric=MARKER),
        lambda r: r["comparisons"][0].update(delta=MARKER),
        lambda r: r["targets"][0].update(status=MARKER),
        lambda r: r["bootstrap"].update(rng=MARKER),
        lambda r: r["evidence"].update(kinds=[MARKER]),
        lambda r: r.update(schema=MARKER),
        lambda r: r.update(note_multiple_comparisons=MARKER),
        lambda r: r["results"][0].update(system=MARKER),
        lambda r: r["results"][0]["statuses"].update({MARKER: 1}),
        lambda r: r.update({MARKER: 1}),
    ],
)
def test_a_refusal_never_quotes_the_value_or_the_key_it_is_about(
    change: Callable[[dict[str, Any]], None],
) -> None:
    refuse(edited(change))  # `refuse` asserts the marker is not in the message


@pytest.mark.parametrize(
    ("change", "section"),
    [
        (
            lambda r: r["systems"][0]["options"].update(model_dir="/home/user/model"),
            "systems.0",
        ),
        (
            lambda r: r["coverage"][0].update(state="not_trained", reason="/tmp/a/b"),
            "coverage.0",
        ),
        (lambda r: r["targets"][0].update(basis="see C:\\work\\x"), "targets.0"),
        (lambda r: r["targets"][1].update(description="at /var/log/x"), "targets.1"),
        (lambda r: r["suites"][0]["source"].update(dataset="/data/flores"), "suites.0"),
        (
            lambda r: r["results"][0]["unmeasured"].update(comet22="/opt/scorer"),
            "results.0",
        ),
    ],
)
def test_an_absolute_path_in_any_string_is_refused_naming_the_section(
    change: Callable[[dict[str, Any]], None], section: str
) -> None:
    message = refuse(edited(change))

    assert "absolute path" in message
    assert f"field '{section}'" in message
    assert "/home/user" not in message and "/tmp" not in message


@pytest.mark.parametrize(
    "text",
    ["a/b", "and/or", "http://example.org/x", "12/16", "khk_Cyrl-eng_Latn/result.json"],
)
def test_a_string_with_a_slash_that_is_not_a_path_is_accepted(text: str) -> None:
    report = golden_report()
    report["targets"][0]["basis"] = text

    assert read(report).targets[0].basis == text


def test_a_robustness_entry_that_breaks_the_format_is_refused() -> None:
    inp = robustness_input("alpha")
    report = build_report(
        [run_input(golden_suite(), file_system("alpha"), robustness=inp)],
        golden_suites(),
        baseline=None,
        config=shipped_config(),
    )
    for change, field in (
        (
            lambda r: r["robustness"]["alpha"]["block"].pop("numbers_acc"),
            "robustness.alpha",
        ),
        (
            lambda r: r["robustness"]["alpha"]["source"].update(binding="trust"),
            "robustness.alpha",
        ),
        (lambda r: r["robustness"]["alpha"].pop("coverage"), "robustness.alpha"),
        (
            lambda r: r["robustness"].update(nobody=r["robustness"]["alpha"]),
            "robustness",
        ),
    ):
        broken = copy.deepcopy(report)
        change(broken)

        assert field in refuse(report_to_bytes(broken))
