"""`rinchen.tiny.report`: the report of a tiny run and what it is built from (D15 Task 6, spec P4, P8,
P10, P12).

`build_report` lists every stage and every part of the pipeline whatever stopped the run, builds the
blocks from the facts of the stages (those of a record that failed included), evaluates the eight
thresholds over the same facts and keeps the elapsed time out of the document: it feeds the verdict of
the wall-time threshold and nothing else. The verdict is `pass` when no stage failed and every
threshold passed, and the exit code is 0 or 4.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS, ThresholdsSection
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import (
    UNMEASURED,
    Inputs,
    Report,
    build_report,
    exit_code,
    report_document,
    to_bytes,
)
from rinchen.tiny.report_schema import BLOCK_IDS, ROBUSTNESS_KEYS
from rinchen.tiny.report_shapes import ShapeError
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_LID,
    SKIP_NO_NEURAL,
    PartRecord,
    StageRecord,
    fail_check,
    fail_exit,
)
from rinchen.tiny.thresholds import THRESHOLD_IDS
from tests.support.tiny_report_build import build, config, golden_inputs
from tests.support.tiny_report_inputs import (
    CONFIG_SHA256,
    FIXTURE_DIGESTS,
    LOCK_SHA256,
    facts_of,
    failed,
    golden_results,
    replaced,
    skipped_after,
    stopped_after,
)


def by_id(report: Report) -> dict[str, Any]:
    return {t.id: t for t in report.thresholds}


def stage(report: Report, stage_id: str) -> StageRecord:
    return next(record for record in report.stages if record.id == stage_id)


def test_the_golden_run_passes_every_threshold_and_exits_0() -> None:
    report = build()
    assert report.verdict == "pass"
    assert exit_code(report) == 0
    assert [t.verdict for t in report.thresholds] == ["pass"] * 8
    assert all(record.state != "fail" for record in report.stages)


def test_the_report_echoes_the_profile_the_flag_and_the_inputs() -> None:
    report = build(profile="control", qe_fake=False)
    assert report.profile == "control"
    assert report.qe_fake is False
    assert report.inputs == golden_inputs()
    assert report.inputs.config_sha256 == CONFIG_SHA256
    assert report.inputs.lock_sha256 == LOCK_SHA256
    assert dict(report.inputs.fixtures) == FIXTURE_DIGESTS


def test_the_profile_is_data_and_changes_nothing_else() -> None:
    first = report_document(build(profile="tiny"))
    second = report_document(build(profile="full"))
    assert first.pop("profile") == "tiny"
    assert second.pop("profile") == "full"
    assert first == second


def test_the_thresholds_are_the_eight_of_the_table_in_order() -> None:
    assert THRESHOLD_IDS == (
        "stages",
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
        "wall_time",
    )
    assert tuple(t.id for t in build().thresholds) == THRESHOLD_IDS


def test_the_thresholds_read_the_bounds_and_the_budget_of_the_profile() -> None:
    changed = config().model_copy(
        update={
            "thresholds": ThresholdsSection(train_chrf_min=99.9),
            "budget_seconds": 100,
        }
    )
    report = build_report(
        "tiny", True, golden_inputs(), golden_results(), 120.01, changed
    )
    table = by_id(report)
    assert table["train_chrf"].bound == ">= 99.9"
    assert table["train_chrf"].verdict == "fail"
    assert table["wall_time"].bound == "<= 100"
    assert table["wall_time"].verdict == "fail"
    assert exit_code(report) == 4


def test_the_values_of_the_thresholds_are_the_figures_of_the_golden_run() -> None:
    table = by_id(build())
    assert table["stages"].value == 0
    assert table["train_chrf"].value == 99.35849111913797
    assert (
        table["train_chrf"].detail
        == "least of 5 directions, mn_Mong-eng_Latn, int8 export"
    )
    assert table["parity_exact"].value == 1.0
    assert table["parity_delta"].value == 1.6221680949143363e-06
    assert table["int8_chrf_drop"].value == 0.0
    assert table["script_purity"].value == 1.0
    assert table["placeholder_acc"].value == 1.0


def test_the_wall_time_threshold_holds_a_verdict_and_never_a_duration() -> None:
    for elapsed in (0.0, 120.01, 1199.9999, 1200, 1200.0):
        table = by_id(build(elapsed=elapsed))
        assert table["wall_time"].value is None
        assert table["wall_time"].verdict == "pass"
        assert table["wall_time"].bound == "<= 1200"
    for elapsed in (1200.001, 5000, 10**9):
        table = by_id(build(elapsed=elapsed))
        assert table["wall_time"].value is None
        assert table["wall_time"].verdict == "fail"


@pytest.mark.parametrize("elapsed", [math.nan, math.inf, -1.0, -0.001])
def test_an_elapsed_time_that_cannot_be_one_fails_the_wall_time_threshold(
    elapsed: float,
) -> None:
    table = by_id(build(elapsed=elapsed))
    assert table["wall_time"].verdict == "fail"
    assert table["wall_time"].value is None
    assert "missing or invalid" in table["wall_time"].detail


def test_the_elapsed_time_is_in_no_byte_of_the_document() -> None:
    marker = 123.456789
    document = to_bytes(build(elapsed=marker))
    assert b"123.45" not in document
    assert b"456789" not in document
    assert to_bytes(build(elapsed=1.0)) == to_bytes(build(elapsed=1199.0))
    assert to_bytes(build(elapsed=1.0)) != to_bytes(build(elapsed=1201.0))


def test_an_overrun_is_a_failed_verdict_even_when_no_stage_failed() -> None:
    report = build(elapsed=1500.0)
    assert all(record.state != "fail" for record in report.stages)
    assert report.verdict == "fail"
    assert exit_code(report) == 4


# ---- completeness: G6.3 ------------------------------------------------------------------------------


@pytest.mark.parametrize("count", range(15))
def test_every_stage_is_listed_in_order_whatever_stopped_the_run(count: int) -> None:
    report = build(golden_results()[:count])
    assert [record.id for record in report.stages] == list(STAGE_IDS)
    for record in report.stages[count:]:
        assert record.state == "skipped"
        assert record.reason == SKIP_EARLIER_FAILED
        assert dict(record.detail) == {}
    for record in report.stages[:count]:
        assert record.state == "ok"


@pytest.mark.parametrize("count", [0, 3, 14])
def test_every_part_of_the_filter_and_of_the_evaluation_is_listed_run_or_not(
    count: int,
) -> None:
    report = build(golden_results()[:count])
    for stage_id, parts in PART_IDS.items():
        record = stage(report, stage_id)
        assert [part.id for part in record.parts] == list(parts)
        if STAGE_IDS.index(stage_id) >= count:
            assert {(part.state, part.reason) for part in record.parts} == {
                ("skipped", SKIP_EARLIER_FAILED)
            }


def test_the_stages_that_have_no_parts_list_none() -> None:
    report = build(golden_results()[:2])
    for record in report.stages:
        if record.id not in PART_IDS:
            assert tuple(record.parts) == ()


def test_the_ok_filter_keeps_its_own_parts_with_their_reasons() -> None:
    record = stage(build(), "filter")
    states = {part.id: (part.state, part.reason) for part in record.parts}
    assert states["lid"] == ("skipped", SKIP_NO_LID)
    assert states["script_ratio"] == ("ok", None)
    assert len(record.parts) == 7
    assert record.detail["token_ratio_check"].startswith("the token-ratio check needs")


def test_a_run_that_stopped_after_a_failure_lists_the_rest_as_skipped() -> None:
    results = stopped_after(9, failing=True)
    report = build(results)
    assert stage(report, "average").state == "skipped"
    failing = stage(report, results[-1].record.id)
    assert failing.state == "fail"
    assert report.verdict == "fail"
    assert exit_code(report) == 4
    assert by_id(report)["stages"].verdict == "fail"


@pytest.mark.parametrize("stage_id", STAGE_IDS)
def test_a_failed_stage_alone_fails_the_verdict_and_the_exit_code(
    stage_id: str,
) -> None:
    report = build(replaced(golden_results(), stage_id, failed(stage_id)))
    assert report.verdict == "fail"
    assert exit_code(report) == 4
    assert stage(report, stage_id).state == "fail"
    assert stage(report, stage_id).reason == fail_exit(4)
    assert by_id(report)["stages"].verdict == "fail"
    assert by_id(report)["stages"].value == 1


def test_a_stage_that_failed_has_no_block_when_it_left_no_facts() -> None:
    report = build(replaced(golden_results(), "synth", failed("synth")))
    assert report.blocks["synth"] is None
    assert report.blocks["filter"] is not None


def test_a_failed_stage_with_parts_lists_every_part() -> None:
    record = StageRecord("filter", "fail", fail_exit(4))
    report = build(replaced(golden_results(), "filter", StageResult(record, {})))
    listed = stage(report, "filter")
    assert [part.id for part in listed.parts] == list(PART_IDS["filter"])
    assert {part.state for part in listed.parts} == {"skipped"}


def test_a_failed_filter_keeps_the_parts_it_judged_and_skips_the_rest() -> None:
    parts = (
        PartRecord("script_ratio", "ok"),
        PartRecord("numbers", "fail", fail_check("planted_noise")),
    )
    record = StageRecord("filter", "fail", fail_check("planted_noise"), parts=parts)
    report = build(replaced(golden_results(), "filter", StageResult(record, {})))
    listed = {part.id: part.state for part in stage(report, "filter").parts}
    assert listed["script_ratio"] == "ok"
    assert listed["numbers"] == "fail"
    assert listed["length"] == listed["qe"] == "skipped"
    assert list(listed) == list(PART_IDS["filter"])


def test_an_ok_stage_lists_the_parts_it_gave_in_the_order_of_the_stage() -> None:
    parts = tuple(PartRecord(part, "ok") for part in reversed(PART_IDS["filter"]))
    record = StageRecord("filter", "ok", parts=parts)
    report = build(replaced(golden_results(), "filter", StageResult(record, {})))
    assert [part.id for part in stage(report, "filter").parts] == list(
        PART_IDS["filter"]
    )


SOME_PARTS = [
    (stage_id, listed)
    for stage_id, parts in PART_IDS.items()
    for listed in range(len(parts))
]


@pytest.mark.parametrize(("stage_id", "listed"), SOME_PARTS)
def test_an_ok_stage_that_lists_only_some_of_its_parts_is_refused_and_not_dropped(
    stage_id: str, listed: int
) -> None:
    parts = tuple(PartRecord(part, "ok") for part in PART_IDS[stage_id][:listed])
    record = StageRecord(stage_id, "ok", parts=parts)
    place = rf"report\.stages\[{STAGE_IDS.index(stage_id)}\]"
    with pytest.raises(ShapeError, match=place) as raised:
        build(replaced(golden_results(), stage_id, StageResult(record, {})))
    assert "every one of its parts" in str(raised.value)


@pytest.mark.parametrize(("stage_id", "listed"), SOME_PARTS)
def test_the_report_model_refuses_an_ok_stage_that_lists_only_some_of_its_parts(
    stage_id: str, listed: int
) -> None:
    report = build()
    position = STAGE_IDS.index(stage_id)
    whole = report.stages[position]
    cut = dataclasses.replace(whole, parts=whole.parts[:listed])
    stages = (*report.stages[:position], cut, *report.stages[position + 1 :])
    with pytest.raises(ShapeError, match=rf"report\.stages\[{position}\]"):
        dataclasses.replace(report, stages=stages)


@pytest.mark.parametrize("stage_id", list(PART_IDS))
def test_a_stage_that_did_not_end_ok_may_list_none_of_its_parts_and_gets_them_all(
    stage_id: str,
) -> None:
    report = build(replaced(golden_results(), stage_id, failed(stage_id)))
    assert [part.id for part in stage(report, stage_id).parts] == list(
        PART_IDS[stage_id]
    )


def test_the_six_robustness_values_and_their_coverage_are_in_the_blocks() -> None:
    block = build().blocks["robustness"]
    assert tuple(block["values"]) == ROBUSTNESS_KEYS
    assert tuple(block["coverage"]) == ROBUSTNESS_KEYS
    assert block["coverage"]["placeholder_acc"] == {
        "excluded": 0,
        "scored": 238,
        "total": 238,
    }


def test_the_blocks_are_the_eleven_in_the_order_of_the_schema() -> None:
    assert tuple(build().blocks) == BLOCK_IDS
    assert len(BLOCK_IDS) == 11


def test_the_unmeasured_list_is_the_fixed_one() -> None:
    document = report_document(build())
    assert document["unmeasured"] == list(UNMEASURED)
    assert len(UNMEASURED) == 6


# ---- a stage that never ran, a stage that failed with facts ------------------------------------------


def test_a_stage_that_never_ran_has_no_block_and_its_thresholds_name_what_is_missing() -> (
    None
):
    report = build(golden_results()[:2])
    for name in BLOCK_IDS:
        assert report.blocks[name] is None
    table = by_id(report)
    assert table["train_chrf"].verdict == "fail"
    assert table["train_chrf"].value is None
    assert table["train_chrf"].detail.startswith("missing or invalid: int8 chrF of ")
    assert (
        table["parity_exact"].detail
        == "missing or invalid: exact_share of the parity figures"
    )
    assert table["parity_delta"].verdict == "fail"
    assert table["int8_chrf_drop"].verdict == "fail"
    assert (
        table["script_purity"].detail
        == "missing or invalid: script_purity of the robustness figures"
    )
    assert table["placeholder_acc"].verdict == "fail"
    assert table["wall_time"].verdict == "pass"


def test_a_missed_parity_gate_keeps_its_figures_for_the_thresholds_and_the_block() -> (
    None
):
    facts = dict(facts_of(golden_results(), "parity"))
    facts.update(exact=50, exact_share=50 / 70, passes=False)
    results = replaced(golden_results(), "parity", failed("parity", facts=facts))
    report = build(results)
    table = by_id(report)
    assert table["parity_exact"].value == 50 / 70
    assert table["parity_exact"].verdict == "fail"
    assert table["parity_exact"].detail == "exact_share of the parity figures"
    assert table["parity_delta"].value == 1.6221680949143363e-06
    assert table["parity_delta"].verdict == "pass"
    assert report.blocks["parity"]["exact"] == 50
    assert report.blocks["parity"]["passes"] is False
    assert stage(report, "parity").reason == fail_exit(4)
    assert report.verdict == "fail"


def test_a_parity_that_wrote_no_figures_is_a_missing_value_and_not_a_pass() -> None:
    report = build(replaced(golden_results(), "parity", failed("parity")))
    table = by_id(report)
    assert table["parity_exact"].verdict == "fail"
    assert table["parity_exact"].detail.startswith("missing or invalid: ")
    assert table["parity_delta"].verdict == "fail"
    assert report.blocks["parity"] is None


def evaluate_failing_after(*done: str, reason: str = FAIL_INPUT) -> StageResult:
    """The evaluation as the stage returns it when its first parts ran and a later one failed: the
    facts of the parts before it, an empty detail, the parts after it skipped."""
    parts = []
    failing_seen = False
    for part_id in PART_IDS["evaluate"]:
        if failing_seen:
            parts.append(PartRecord(part_id, "skipped", SKIP_EARLIER_FAILED))
        elif part_id in done:
            parts.append(PartRecord(part_id, "ok"))
        else:
            parts.append(PartRecord(part_id, "fail", reason))
            failing_seen = True
    facts = {
        key: value
        for key, value in facts_of(golden_results(), "evaluate").items()
        if key in _FACTS_OF[done]
    }
    return StageResult(StageRecord("evaluate", "fail", reason, {}, parts), facts)


_FACTS_OF = {
    (): set(),
    ("memorised",): {"memorised", "memorised_chrf", "int8_chrf_drop", "agreement"},
    ("memorised", "heldout"): {
        "memorised",
        "memorised_chrf",
        "int8_chrf_drop",
        "agreement",
        "heldout",
    },
}


def test_an_evaluation_that_failed_after_its_first_part_still_feeds_its_thresholds() -> (
    None
):
    results = replaced(
        golden_results(), "evaluate", evaluate_failing_after("memorised")
    )
    report = build(results)
    table = by_id(report)
    assert table["train_chrf"].value == 99.35849111913797
    assert table["train_chrf"].verdict == "pass"
    assert table["int8_chrf_drop"].value == 0.0
    assert report.blocks["memorised"] is not None
    assert report.blocks["int8"] is not None
    assert report.blocks["heldout"] is None
    assert report.blocks["neural_seam"] is None
    record = stage(report, "evaluate")
    assert record.state == "fail"
    assert dict(record.detail) == {}
    states = [(part.id, part.state, part.reason) for part in record.parts]
    assert states == [
        ("memorised", "ok", None),
        ("heldout", "fail", FAIL_INPUT),
        ("neural_seam", "skipped", SKIP_EARLIER_FAILED),
        ("report", "skipped", SKIP_EARLIER_FAILED),
    ]
    assert report.verdict == "fail"


def test_an_evaluation_that_failed_in_its_first_part_has_no_figures() -> None:
    results = replaced(golden_results(), "evaluate", evaluate_failing_after())
    report = build(results)
    table = by_id(report)
    assert table["train_chrf"].verdict == "fail"
    assert table["train_chrf"].detail.startswith("missing or invalid")
    for name in ("memorised", "heldout", "neural_seam", "int8"):
        assert report.blocks[name] is None


def test_a_robustness_stage_that_failed_leaves_the_two_thresholds_that_read_it_missing() -> (
    None
):
    report = build(replaced(golden_results(), "robustness", failed("robustness")))
    table = by_id(report)
    assert table["script_purity"].verdict == "fail"
    assert table["placeholder_acc"].verdict == "fail"
    assert report.blocks["robustness"] is None
    assert table["train_chrf"].verdict == "pass"


def test_a_skipped_neural_part_is_a_null_block_and_a_part_with_its_closed_reason() -> (
    None
):
    parts = [PartRecord(part, "ok") for part in PART_IDS["evaluate"]]
    parts[2] = PartRecord("neural_seam", "skipped", SKIP_NO_NEURAL)
    facts = dict(facts_of(golden_results(), "evaluate"))
    facts["neural_seam"] = None
    record = StageRecord("evaluate", "ok", detail=copy.deepcopy(facts), parts=parts)
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))
    assert report.blocks["neural_seam"] is None
    neural = [
        part for part in stage(report, "evaluate").parts if part.id == "neural_seam"
    ]
    assert (neural[0].state, neural[0].reason) == ("skipped", SKIP_NO_NEURAL)
    assert report.verdict == "pass"


def test_a_figure_under_its_bound_fails_its_threshold_and_the_verdict() -> None:
    facts = copy.deepcopy(dict(facts_of(golden_results(), "evaluate")))
    facts["memorised_chrf"]["rus_Cyrl-khk_Cyrl"]["int8"] = 40.0
    record = StageRecord(
        "evaluate",
        "ok",
        detail=copy.deepcopy(facts),
        parts=[PartRecord(p, "ok") for p in PART_IDS["evaluate"]],
    )
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))
    table = by_id(report)
    assert table["train_chrf"].value == 40.0
    assert table["train_chrf"].verdict == "fail"
    assert report.verdict == "fail"
    assert exit_code(report) == 4


# ---- the details, the order and the duplicates -------------------------------------------------------


def test_the_results_may_come_in_any_order() -> None:
    results = golden_results()
    assert to_bytes(build(list(reversed(results)))) == to_bytes(build(results))


def test_two_results_for_one_stage_are_refused() -> None:
    results = golden_results()
    with pytest.raises(ValueError, match="two results"):
        build([*results, results[3]])


def test_a_detail_the_report_cannot_hold_is_left_out_and_the_state_is_kept() -> None:
    results = golden_results()
    record = StageRecord("fetch", "ok", detail={"files": 7, "where": "/home/someone/x"})
    report = build(replaced(results, "fetch", StageResult(record, {"files": 7})))
    listed = stage(report, "fetch")
    assert listed.state == "ok"
    assert dict(listed.detail) == {}
    assert "/home" not in to_bytes(report).decode()


@pytest.mark.parametrize(
    "detail",
    [
        {"seconds": 3},
        {"elapsed_time": 3},
        {"note": "2026-11-11T00:00:00Z"},
        {"a b": 1},
        {"text": "caf" + chr(0xE9)},
    ],
)
def test_a_detail_with_a_duration_a_date_or_an_odd_key_is_left_out(
    detail: dict[str, Any],
) -> None:
    record = StageRecord("fetch", "ok", detail=detail)
    report = build(replaced(golden_results(), "fetch", StageResult(record, {})))
    assert dict(stage(report, "fetch").detail) == {}


def test_a_detail_that_holds_a_tuple_comes_out_as_a_list() -> None:
    record = StageRecord("spm", "ok", detail={"languages": ("eng_Latn", "khk_Cyrl")})
    report = build(replaced(golden_results(), "spm", StageResult(record, {})))
    assert stage(report, "spm").detail["languages"] == ["eng_Latn", "khk_Cyrl"]


def test_building_a_report_changes_neither_the_records_nor_the_facts() -> None:
    results = golden_results()
    before = copy.deepcopy([(r.record, dict(r.facts)) for r in results])
    build(results)
    assert [(r.record, dict(r.facts)) for r in results] == before


def test_a_bad_profile_name_is_refused() -> None:
    for profile in ("Tiny", "", "t" * 17, "a b", "a/b"):
        with pytest.raises(ShapeError):
            build(profile=profile)


# ---- the report as a value ---------------------------------------------------------------------------


def test_a_report_is_frozen() -> None:
    report = build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.profile = "full"  # type: ignore[misc]


def test_the_verdict_is_a_property_of_the_stages_and_the_thresholds() -> None:
    report = build()
    broken = dataclasses.replace(
        report,
        thresholds=(
            *report.thresholds[:7],
            dataclasses.replace(report.thresholds[7], verdict="fail"),
        ),
    )
    assert report.verdict == "pass"
    assert broken.verdict == "fail"
    assert exit_code(broken) == 4


def test_a_report_with_fewer_stages_or_in_another_order_is_refused() -> None:
    report = build()
    with pytest.raises(ShapeError):
        dataclasses.replace(report, stages=report.stages[:14])
    with pytest.raises(ShapeError, match="in order"):
        dataclasses.replace(report, stages=(*reversed(report.stages),))
    with pytest.raises(ShapeError, match="in order"):
        dataclasses.replace(report, thresholds=(*reversed(report.thresholds),))


def test_a_threshold_whose_verdict_does_not_follow_from_its_value_is_refused() -> None:
    report = build()
    table = list(report.thresholds)
    table[1] = dataclasses.replace(table[1], verdict="fail")
    with pytest.raises(ShapeError, match=r"thresholds\[1\]"):
        dataclasses.replace(report, thresholds=tuple(table))
    table[1] = dataclasses.replace(report.thresholds[1], value=10.0)
    with pytest.raises(ShapeError, match=r"thresholds\[1\]"):
        dataclasses.replace(report, thresholds=tuple(table))


def test_only_the_wall_time_threshold_may_pass_without_a_value() -> None:
    report = build()
    table = list(report.thresholds)
    table[2] = dataclasses.replace(table[2], value=None)
    with pytest.raises(ShapeError, match=r"thresholds\[2\]"):
        dataclasses.replace(report, thresholds=tuple(table))
    table[2] = dataclasses.replace(table[2], verdict="fail")
    assert dataclasses.replace(report, thresholds=tuple(table)).verdict == "fail"
    wall = dataclasses.replace(report.thresholds[7], verdict="fail")
    assert (
        dataclasses.replace(report, thresholds=(*report.thresholds[:7], wall)).verdict
        == "fail"
    )


def test_a_report_refuses_blocks_that_are_not_the_schema() -> None:
    report = build()
    blocks = copy.deepcopy(dict(report.blocks))
    blocks["filter"]["unexpected"] = 1
    with pytest.raises(ShapeError, match=r"blocks\.filter"):
        dataclasses.replace(report, blocks=blocks)
    del blocks["filter"]["unexpected"]
    blocks["serve"] = None
    assert dataclasses.replace(report, blocks=blocks).blocks["serve"] is None


def test_inputs_are_digests_of_the_profile_the_seven_fixtures_and_the_lock() -> None:
    inputs = golden_inputs()
    assert isinstance(inputs, Inputs)
    with pytest.raises(ShapeError, match=r"inputs\.config_sha256"):
        Inputs("xyz", dict(FIXTURE_DIGESTS), LOCK_SHA256)
    with pytest.raises(ShapeError, match=r"inputs\.lock_sha256"):
        Inputs(CONFIG_SHA256, dict(FIXTURE_DIGESTS), "A" * 64)
    with pytest.raises(ShapeError, match=r"inputs\.fixtures"):
        Inputs(CONFIG_SHA256, {"noise.tsv": "0" * 64}, LOCK_SHA256)
    extra = {**FIXTURE_DIGESTS, "other.tsv": "0" * 64}
    with pytest.raises(ShapeError, match=r"inputs\.fixtures"):
        Inputs(CONFIG_SHA256, extra, LOCK_SHA256)


def test_a_skipped_stage_result_for_a_stage_that_never_ran_is_the_same_as_none() -> (
    None
):
    results = golden_results()
    explicit = replaced(results, "evaluate", skipped_after("evaluate"))
    absent = results[:-1]
    assert to_bytes(build(explicit)) == to_bytes(build(absent))


def test_the_budget_spent_failure_is_a_failed_stage_like_any_other() -> None:
    record = StageRecord("train", "fail", FAIL_BUDGET)
    report = build(replaced(golden_results()[:9], "train", StageResult(record, {})))
    assert stage(report, "train").reason == FAIL_BUDGET
    assert report.verdict == "fail"
