"""`rinchen.tiny.report_markdown.render`: the Markdown of a tiny run's report (D15 Task 6, spec P10).

The Markdown is rendered from the report alone, so that `render(parse_report(to_bytes(report)))` is
`render(report)`: a heading with the verdict and the profile, the table of the stages with the parts of
the filter and of the evaluation (and the note of the filter's skipped token-ratio check beside its
`length` part), the table of the thresholds with their bounds, values and verdicts, the held-out table
with the line that says it is never gated and the neural score or the reason it was not measured, the
six robustness values beside D12's reference floors, and the fixed list of what was not measured. A
float beside a bound or a floor is written with the fewest significant digits (four or more) that keep
it on its bound's side, a float elsewhere with two decimals, a null as a hyphen, and nothing comes from
the clock.
"""

from __future__ import annotations

import dataclasses
import math
import re

import pytest

from rinchen.tiny.context import StageResult
from rinchen.tiny.report import Report, parse_report, to_bytes
from rinchen.tiny.report_markdown import figure_beside_bound, render
from rinchen.tiny.report_schema import ROBUSTNESS_KEYS, UNMEASURED
from rinchen.tiny.states import (
    FAIL_INPUT,
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_LID,
    SKIP_NO_NEURAL,
    SKIP_TOKEN_RATIO,
    PartRecord,
    StageRecord,
    fail_exit,
)
from tests.support.tiny_report_build import VARIANT_NAMES, build, variant_reports
from tests.support.tiny_report_inputs import (
    evaluate_facts,
    golden_results,
    replaced,
    stopped_after,
    without_neural_part,
)
from tests.support.tiny_report_scans import text_violations
from tests.support.tiny_report_walk import GOLDEN_MD

FLOORS = {
    "entity_acc": "0.90",
    "mong_source_retention": "0.95",
    "numbers_acc": "0.90",
    "placeholder_acc": "0.99",
    "script_purity": "0.999",
    "term_acc": "0.99",
}
SECTIONS = [
    "## Stages",
    "## Thresholds",
    "## Held-out",
    "## Robustness",
    "## Not measured",
]


def lines_of(report: Report) -> list[str]:
    return render(report).split("\n")


def section(report: Report, heading: str) -> list[str]:
    """The lines of the section that starts at `heading`, up to the next heading."""
    lines = lines_of(report)
    start = lines.index(heading) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    return lines[start:end]


def rows(report: Report, heading: str) -> list[list[str]]:
    """The cells of the data rows of the table in a section."""
    table = [line for line in section(report, heading) if line.startswith("| ")]
    return [
        [cell.strip() for cell in line.strip("|").split(" | ")] for line in table[2:]
    ]


# ---- the golden file and the round trip --------------------------------------------------------------


def test_the_markdown_of_the_golden_run_is_the_golden_markdown() -> None:
    assert render(build()) == GOLDEN_MD.read_text(encoding="utf-8")
    assert render(build()).encode() == GOLDEN_MD.read_bytes()


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_the_markdown_is_the_same_before_and_after_a_trip_through_the_bytes(
    name: str,
) -> None:
    report = variant_reports()[name]
    assert render(parse_report(to_bytes(report))) == render(report)


def test_the_golden_report_renders_the_same_after_the_trip() -> None:
    report = build()
    assert render(parse_report(to_bytes(report))) == render(report)


# ---- the shape of the text ---------------------------------------------------------------------------


def test_the_text_has_line_feeds_only_and_ends_in_exactly_one() -> None:
    for report in (build(), *variant_reports().values()):
        text = render(report)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
        assert "\r" not in text
        assert "\t" not in text
        assert text.isascii()


def test_the_heading_names_the_verdict_and_the_profile() -> None:
    assert lines_of(build())[0] == "# Tiny pipeline: pass (profile tiny)"
    assert (
        lines_of(build(profile="control"))[0]
        == "# Tiny pipeline: pass (profile control)"
    )
    failing = build(stopped_after(9, failing=True))
    assert lines_of(failing)[0] == "# Tiny pipeline: fail (profile tiny)"
    assert lines_of(build(elapsed=9000.0))[0] == "# Tiny pipeline: fail (profile tiny)"


def test_the_five_sections_come_in_order_each_after_a_blank_line() -> None:
    for report in (build(), *variant_reports().values()):
        lines = lines_of(report)
        positions = [lines.index(heading) for heading in SECTIONS]
        assert positions == sorted(positions)
        for position in positions:
            assert lines[position - 1] == ""
            assert lines[position + 1] == ""
        assert [line for line in lines if line.startswith("## ")] == SECTIONS
        assert [line for line in lines if line.startswith("# ")] == [lines[0]]


def test_there_are_no_runs_of_blank_lines() -> None:
    assert "\n\n\n" not in render(build())


# ---- the stages --------------------------------------------------------------------------------------


def test_the_stage_table_lists_the_fifteen_stages_and_the_parts_of_two_of_them() -> (
    None
):
    table = rows(build(), "## Stages")
    assert section(build(), "## Stages")[1].startswith("| stage | state | reason |")
    assert len(table) == 15 + 7 + 4
    names = [row[0] for row in table]
    assert [name for name in names if "." not in name] == [
        "environment",
        "fetch",
        "corpus",
        "filter",
        "spm",
        "converter",
        "synth",
        "prepare",
        "train",
        "average",
        "export",
        "parity",
        "serve",
        "robustness",
        "evaluate",
    ]
    assert names[names.index("filter") + 1] == "filter.script_ratio"
    assert [name for name in names if name.startswith("filter.")] == [
        f"filter.{part}" for part in PART_IDS["filter"]
    ]
    assert [name for name in names if name.startswith("evaluate.")] == [
        f"evaluate.{part}" for part in PART_IDS["evaluate"]
    ]


def test_the_state_and_the_reason_of_every_row_are_the_records() -> None:
    table = {row[0]: row for row in rows(build(), "## Stages")}
    assert table["environment"][1:] == ["ok", ""]
    assert table["filter.lid"][1:] == ["skipped", SKIP_NO_LID]
    failing = build(stopped_after(9, failing=True))
    table = {row[0]: row for row in rows(failing, "## Stages")}
    assert table["train"][1:] == ["fail", fail_exit(4)]
    assert table["average"][1:] == ["skipped", SKIP_EARLIER_FAILED]
    assert table["evaluate.memorised"][1:] == ["skipped", SKIP_EARLIER_FAILED]


def test_the_note_of_the_skipped_token_ratio_check_is_beside_the_length_part() -> None:
    table = {row[0]: row for row in rows(build(), "## Stages")}
    assert table["filter.length"][1] == "ok"
    assert table["filter.length"][2] == f"token_ratio_check skipped: {SKIP_TOKEN_RATIO}"
    assert SKIP_TOKEN_RATIO in render(build())


def test_the_note_is_on_the_stage_row_when_the_filter_lists_no_length_part() -> None:
    # The report refuses an ok filter that leaves a part out, so the report is damaged after it was
    # built: the renderer must still put the note where the part is not.
    record = StageRecord(
        "filter",
        "ok",
        detail={"token_ratio_check": SKIP_TOKEN_RATIO},
        parts=(PartRecord("script_ratio", "ok"),),
    )
    report = build()
    stages = tuple(record if stage.id == "filter" else stage for stage in report.stages)
    object.__setattr__(report, "stages", stages)
    table = {row[0]: row for row in rows(report, "## Stages")}
    assert table["filter"][2] == f"token_ratio_check skipped: {SKIP_TOKEN_RATIO}"
    assert "filter.length" not in table
    assert table["filter.script_ratio"][2] == ""


def test_the_names_of_the_variants_are_those_of_the_reports() -> None:
    assert tuple(variant_reports()) == VARIANT_NAMES


def test_the_note_is_not_shown_for_a_filter_that_has_none() -> None:
    record = StageRecord(
        "filter",
        "ok",
        detail={"rows": 770},
        parts=tuple(PartRecord(p, "ok") for p in PART_IDS["filter"]),
    )
    table = {
        row[0]: row
        for row in rows(
            build(replaced(golden_results(), "filter", StageResult(record, {}))),
            "## Stages",
        )
    }
    assert table["filter.length"][2] == ""
    assert "token_ratio_check" not in render(
        build(replaced(golden_results(), "filter", StageResult(record, {})))
    )


def test_the_note_of_the_check_is_also_shown_when_the_filter_failed() -> None:
    record = StageRecord(
        "filter",
        "fail",
        FAIL_INPUT,
        detail={"token_ratio_check": SKIP_TOKEN_RATIO},
        parts=(PartRecord("script_ratio", "ok"), PartRecord("length", "ok")),
    )
    report = build(replaced(golden_results(), "filter", StageResult(record, {})))
    table = {row[0]: row for row in rows(report, "## Stages")}
    assert table["filter.length"][2].startswith("token_ratio_check skipped")


def test_a_cell_cannot_break_the_table() -> None:
    detail = {"token_ratio_check": "x | y `z` w"}
    record = StageRecord(
        "filter",
        "ok",
        detail=detail,
        parts=tuple(PartRecord(p, "ok") for p in PART_IDS["filter"]),
    )
    report = build(replaced(golden_results(), "filter", StageResult(record, {})))
    lines = [line for line in section(report, "## Stages") if line.startswith("| ")]
    unescaped = re.compile(r"(?<!\\)\|")
    for line in lines:
        assert len(unescaped.findall(line)) == 4, line
    assert "x \\| y \\`z\\` w" in render(report)


# ---- the thresholds ----------------------------------------------------------------------------------


def test_the_threshold_table_has_the_eight_rows_with_four_significant_digits() -> None:
    table = rows(build(), "## Thresholds")
    assert [row[0] for row in table] == [
        "stages",
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
        "wall_time",
    ]
    assert section(build(), "## Thresholds")[1].startswith(
        "| threshold | bound | value | verdict | detail |"
    )
    cells = {row[0]: row[1:4] for row in table}
    assert cells["stages"] == ["<= 0", "0", "pass"]
    assert cells["train_chrf"] == [">= 50.0", "99.36", "pass"]
    assert cells["parity_exact"] == [">= 0.95", "1", "pass"]
    assert cells["parity_delta"] == ["<= 0.001", "1.622e-06", "pass"]
    assert cells["int8_chrf_drop"] == ["<= 1.0", "0", "pass"]
    assert cells["script_purity"] == [">= 1.0", "1", "pass"]
    assert cells["placeholder_acc"] == [">= 1.0", "1", "pass"]
    assert cells["wall_time"] == ["<= 1200", "-", "pass"]


def test_a_float_is_never_written_in_full() -> None:
    text = render(build())
    assert "99.35849111913797" not in text
    assert "1.6221680949143363e-06" not in text


def test_a_float_beside_a_bound_has_four_significant_digits_when_it_is_far_from_it() -> (
    None
):
    # The golden figures are far from their bounds: none needs more than four digits.
    for name in ("## Thresholds", "## Robustness"):
        for row in rows(build(), name):
            figure = row[2] if name == "## Thresholds" else row[1]
            digits = re.sub(r"e[+-][0-9]+$", "", figure).replace(".", "").lstrip("0-")
            assert len(digits) <= 4, (row[0], figure)


def test_the_detail_of_each_threshold_is_in_the_table() -> None:
    table = {row[0]: row for row in rows(build(), "## Thresholds")}
    assert (
        table["train_chrf"][4] == "least of 5 directions, mn_Mong-eng_Latn, int8 export"
    )
    assert (
        table["stages"][4]
        == "no stage failed and every skip has a reason of the closed list"
    )


def test_a_missing_figure_is_a_hyphen_with_its_reason_and_a_failed_verdict() -> None:
    report = build(stopped_after(3, failing=True))
    table = {row[0]: row for row in rows(report, "## Thresholds")}
    for name in (
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
    ):
        assert table[name][2] == "-", name
        assert table[name][3] == "fail", name
        assert table[name][4].startswith("missing or invalid: "), name
    # The stage that failed is the one fault: the stages after it are skipped for it.
    assert table["stages"][2:4] == ["1", "fail"]
    assert table["wall_time"][2:4] == ["-", "pass"]


def with_thresholds(**changes: tuple[float, str]) -> Report:
    """The golden report with the value and the verdict of some thresholds changed."""
    report = build()
    table = []
    for result in report.thresholds:
        if result.id in changes:
            value, verdict = changes[result.id]
            result = dataclasses.replace(result, value=value, verdict=verdict)
        table.append(result)
    return dataclasses.replace(report, thresholds=tuple(table))


def test_a_failing_value_is_never_written_as_one_that_meets_its_bound() -> None:
    # Two decimals wrote 49.996 as `50.00` beside `>= 50.0` and 237/238 as `1.00` beside `>= 1.0`,
    # both with the verdict fail: a reader of the table saw a value that met its bound.
    report = with_thresholds(
        train_chrf=(49.996, "fail"), placeholder_acc=(237 / 238, "fail")
    )
    table = {row[0]: row for row in rows(report, "## Thresholds")}
    assert table["train_chrf"][1:4] == [">= 50.0", "49.996", "fail"]
    assert table["placeholder_acc"][1:4] == [">= 1.0", "0.9958", "fail"]


def test_a_value_that_passes_is_never_written_as_one_that_misses_its_bound() -> None:
    report = with_thresholds(
        parity_delta=(1.6221680949143363e-06, "pass"),
        parity_exact=(0.9500000000000001, "pass"),
        int8_chrf_drop=(1.0, "pass"),
    )
    table = {row[0]: row for row in rows(report, "## Thresholds")}
    assert table["parity_delta"][1:4] == ["<= 0.001", "1.622e-06", "pass"]
    assert table["parity_exact"][1:4] == [">= 0.95", "0.95", "pass"]
    assert table["int8_chrf_drop"][1:4] == ["<= 1.0", "1", "pass"]


def test_a_value_that_misses_its_bound_by_a_hair_is_written_with_the_digits_that_show_it() -> (
    None
):
    report = with_thresholds(
        parity_delta=(0.0010004, "fail"), parity_exact=(0.94999, "fail")
    )
    table = {row[0]: row for row in rows(report, "## Thresholds")}
    assert table["parity_delta"][1:4] == ["<= 0.001", "0.0010004", "fail"]
    assert table["parity_exact"][1:4] == [">= 0.95", "0.94999", "fail"]


def test_a_robustness_value_under_its_floor_is_not_written_as_meeting_it() -> None:
    results = golden_results()
    facts = dict(results[13].facts)
    facts["values"] = {**facts["values"], "term_acc": 0.98999}
    record = StageRecord("robustness", "ok", detail={})
    report = build(replaced(results, "robustness", StageResult(record, facts)))
    table = {row[0]: row for row in rows(report, "## Robustness")}
    assert table["term_acc"][1:3] == ["0.98999", "0.99"]


# The figure a cell holds beside a bound: a null, a whole number, and floats that need only four
# digits or more of them.
BESIDE = [
    (None, ">= 50.0", "-"),
    (3, "<= 0", "3"),
    (49.996, ">= 50.0", "49.996"),
    (237 / 238, ">= 1.0", "0.9958"),
    (1.6221680949143363e-06, "<= 0.001", "1.622e-06"),
    (99.35849111913797, ">= 50.0", "99.36"),
    (1.0, ">= 0.95", "1"),
    (0.0, "<= 1.0", "0"),
    (0.99996, ">= 1.0", "0.99996"),
    (0.0010004, "<= 0.001", "0.0010004"),
    (50.0, ">= 50.0", "50"),
    (50.00001, ">= 50.0", "50"),
    (math.nextafter(50.0, 0.0), ">= 50.0", "49.99999999999999"),
    (math.nextafter(0.001, 1.0), "<= 0.001", "0.0010000000000000002"),
    (0.9999999999999999, ">= 1.0", "0.9999999999999999"),
    (0.99999999123, ">= 1.0", "0.99999999"),
    (-1e-15, "<= 1.0", "-1e-15"),
]


@pytest.mark.parametrize(("value", "bound", "text"), BESIDE)
def test_a_figure_beside_its_bound_is_written_as_the_table_says(
    value: float | None, bound: str, text: str
) -> None:
    assert figure_beside_bound(value, bound) == text


def test_a_figure_that_is_not_beside_a_bound_keeps_two_decimals() -> None:
    held = {row[0]: row for row in rows(build(), "## Held-out")}
    assert held["eng_Latn-khk_Cyrl"][3:6] == ["72.00", "71.02", "54.55"]
    assert "3.88 over 30 rows" in "\n".join(section(build(), "## Held-out"))


def test_a_failed_threshold_shows_its_value_and_its_verdict() -> None:
    facts = evaluate_facts()
    facts["memorised_chrf"]["rus_Cyrl-khk_Cyrl"]["int8"] = 40.0
    record = StageRecord(
        "evaluate",
        "ok",
        detail={},
        parts=tuple(PartRecord(p, "ok") for p in PART_IDS["evaluate"]),
    )
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))
    table = {row[0]: row for row in rows(report, "## Thresholds")}
    assert table["train_chrf"][1:4] == [">= 50.0", "40", "fail"]


def test_the_wall_time_row_has_a_hyphen_and_the_sentence_says_where_the_seconds_are() -> (
    None
):
    for elapsed, verdict in ((120.01, "pass"), (5000.0, "fail")):
        report = build(elapsed=elapsed)
        table = {row[0]: row for row in rows(report, "## Thresholds")}
        assert table["wall_time"][1:4] == ["<= 1200", "-", verdict]
    text = section(build(), "## Thresholds")
    assert (
        "The wall-time figure is in manifest.json: this report holds no duration and no timestamp."
        in text
    )


# ---- the held-out table ------------------------------------------------------------------------------


def test_the_held_out_section_says_it_is_reported_and_never_gated() -> None:
    lines = section(build(), "## Held-out")
    assert (
        "Held-out rows: generalisation on templated sentences, reported and never gated."
        in lines
    )
    assert "generalisation on templated sentences, reported and never gated" in render(
        build()
    )


def test_the_held_out_table_has_a_row_per_direction_with_chrf_chrf_pp_bleu_and_the_statuses() -> (
    None
):
    assert section(build(), "## Held-out")[3].startswith(
        "| direction | system | rows | chrF | chrF++ | BLEU | statuses |"
    )
    table = rows(build(), "## Held-out")
    assert table == [
        ["eng_Latn-khk_Cyrl", "int8", "30", "72.00", "71.02", "54.55", "ok 30"],
        ["khk_Cyrl-eng_Latn", "int8", "30", "64.19", "63.37", "48.13", "ok 30"],
        ["khk_Cyrl-rus_Cyrl", "int8", "30", "55.31", "54.62", "43.66", "ok 30"],
        ["rus_Cyrl-khk_Cyrl", "int8", "30", "60.80", "59.95", "46.78", "ok 30"],
    ]


def test_the_statuses_are_listed_in_the_order_of_the_engine() -> None:
    facts = evaluate_facts()
    run = facts["heldout"]["eng_Latn-khk_Cyrl"]["int8"]
    run["statuses"] = {"no_output": 1, "empty": 2, "ok": 25, "truncated": 2}
    record = StageRecord(
        "evaluate",
        "ok",
        detail={},
        parts=tuple(PartRecord(p, "ok") for p in PART_IDS["evaluate"]),
    )
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))
    assert (
        rows(report, "## Held-out")[0][-1] == "ok 25, empty 2, truncated 2, no_output 1"
    )


def test_a_held_out_block_that_is_null_is_said_so() -> None:
    report = build(stopped_after(14))
    lines = section(report, "## Held-out")
    assert (
        "Held-out rows: generalisation on templated sentences, reported and never gated."
        in lines
    )
    assert any(line.startswith("Not measured") for line in lines)
    assert not any(line.startswith("| direction") for line in lines)


def test_the_neural_score_is_shown_with_its_metric_system_rows_and_two_decimals() -> (
    None
):
    lines = section(build(), "## Held-out")
    expected = (
        "Neural score (metricx24_qe of tiny-int8-qe, planted scores): "
        "eng_Latn-khk_Cyrl 3.88 over 30 rows, khk_Cyrl-eng_Latn 4.31 over 30 rows."
    )
    assert expected in lines


def test_a_skipped_neural_part_says_it_was_not_measured_and_why() -> None:
    report = build(replaced(golden_results(), "evaluate", without_neural_part()))
    lines = section(report, "## Held-out")
    assert f"Neural score: not measured ({SKIP_NO_NEURAL})." in lines
    assert not any(line.startswith("Neural score (") for line in lines)


def test_a_neural_part_that_never_ran_says_the_closed_reason_of_the_part() -> None:
    lines = section(build(stopped_after(5)), "## Held-out")
    assert f"Neural score: not measured ({SKIP_EARLIER_FAILED})." in lines


def test_a_neural_part_that_failed_says_the_reason_it_failed_for() -> None:
    parts = (
        PartRecord("memorised", "ok"),
        PartRecord("heldout", "ok"),
        PartRecord("neural_seam", "fail", fail_exit(4)),
        PartRecord("report", "skipped", SKIP_EARLIER_FAILED),
    )
    facts = {k: v for k, v in evaluate_facts().items() if k != "neural_seam"}
    record = StageRecord("evaluate", "fail", fail_exit(4), {}, parts)
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))
    assert f"Neural score: not measured ({fail_exit(4)})." in section(
        report, "## Held-out"
    )


# ---- robustness --------------------------------------------------------------------------------------


def test_the_robustness_table_lists_the_six_values_beside_the_reference_floors() -> (
    None
):
    assert section(build(), "## Robustness")[3].startswith(
        "| measure | value | reference floor | scored | total | excluded |"
    )
    table = rows(build(), "## Robustness")
    assert [row[0] for row in table] == list(ROBUSTNESS_KEYS)
    assert {row[0]: row[2] for row in table} == FLOORS
    assert {row[0]: row[1] for row in table} == {
        "entity_acc": "0.9792",
        "mong_source_retention": "0.9936",
        "numbers_acc": "1",
        "placeholder_acc": "1",
        "script_purity": "1",
        "term_acc": "0.9991",
    }
    assert {row[0]: row[3:] for row in table}["term_acc"] == ["1134", "1134", "0"]
    assert {row[0]: row[3:] for row in table}["mong_source_retention"] == [
        "48",
        "48",
        "0",
    ]


def test_the_floors_are_labelled_as_references_and_not_as_gates() -> None:
    text = "\n".join(section(build(), "## Robustness"))
    assert "reference floors" in text
    assert "not gates of this run" in text


def test_a_robustness_value_that_could_not_be_measured_is_a_hyphen() -> None:
    results = golden_results()
    facts = {key: value for key, value in dict(results[13].facts).items()}
    facts["values"] = {**facts["values"], "mong_source_retention": None}
    record = StageRecord("robustness", "ok", detail={})
    report = build(replaced(results, "robustness", StageResult(record, facts)))
    table = {row[0]: row for row in rows(report, "## Robustness")}
    assert table["mong_source_retention"][1] == "-"


def test_a_robustness_block_that_is_null_is_said_so() -> None:
    report = build(stopped_after(5))
    lines = section(report, "## Robustness")
    assert any(line.startswith("Not measured") for line in lines)
    assert not any(line.startswith("| measure") for line in lines)


# ---- not measured ------------------------------------------------------------------------------------


def test_the_fixed_list_of_what_was_not_measured_closes_the_text() -> None:
    lines = section(build(), "## Not measured")
    assert [line for line in lines if line] == [f"- {item}" for item in UNMEASURED]
    assert lines[-1] == ""
    assert render(build()).endswith(f"- {UNMEASURED[-1]}\n")


# ---- what the text must not hold ---------------------------------------------------------------------


def test_the_markdown_holds_no_time_path_duration_environment_value_or_fixture_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "HOME": "/zz-unique-home-4f1c",
        "USER": "zz-unique-user-9a2e",
        "PATH": "/zz-unique-path-77d0:/zz-unique-bin",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for name, report in {"golden": build(), **variant_reports()}.items():
        assert text_violations(render(report), tuple(environment.values())) == [], name


def test_the_markdown_does_not_depend_on_the_elapsed_time() -> None:
    assert render(build(elapsed=1.0)) == render(build(elapsed=1199.0))


def test_the_markdown_shows_nothing_the_report_does_not_hold() -> None:
    text = render(build())
    for word in ("elapsed", "manifest_sha256", "hypotheses_sha256", "fingerprint"):
        assert word not in text
    assert text.count("manifest.json") == 1


def test_a_report_with_nothing_run_renders_and_every_section_says_so() -> None:
    report = build([])
    text = render(report)
    assert lines_of(report)[0] == "# Tiny pipeline: fail (profile tiny)"
    assert len(rows(report, "## Stages")) == 15 + 7 + 4
    assert {row[1] for row in rows(report, "## Stages")} == {"skipped"}
    assert text.count("Not measured") >= 3
    assert "Neural score: not measured (an earlier stage failed)." in text


def test_the_markdown_of_every_variant_keeps_the_same_sections() -> None:
    for report in variant_reports().values():
        assert [line for line in lines_of(report) if line.startswith("## ")] == SECTIONS
