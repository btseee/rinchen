"""What `report.md` shows beside a figure (D13 fix wave 1: I3, I4, M6, M7).

- I3: a failure the system flagged stays visible beside the metric. Every score table has, for each
  system, a column `<system> ok` with `n of n` when every output was `ok` and `k of n (1 empty, 3
  truncated)` otherwise, derived from the `statuses` of the report's results (no new key).
- I4: the Suites table names the dataset and the revision of each suite: the revision, `unpinned`
  when none was pinned (or `<revision> (unpinned)` when one was recorded and not pinned) and
  `n/a (no source record)` for a suite with no source record, a fixture for one.
- M6: the comparison tables give a verdict in words, `better`, `worse` or `no significant
  difference` (`n/a (<reason>)` without a p-value), and not a star: a system that is significantly
  worse than the baseline must not read as `no`. `report.json`'s `significant` stays "the first system
  is better".
- M7: the robustness block of D12 has no suite kind, and the report says so.
"""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.report import build_report
from rinchen.eval.report_format import SECTIONS
from rinchen.eval.report_markdown import render_markdown
from tests.support.eval_fakes import golden_outputs, golden_suite
from tests.support.eval_report import (
    FIRST,
    SECOND,
    file_system,
    golden_run,
    golden_runs,
    golden_suites,
    mixed_outputs,
    robustness_input,
    run_input,
    shipped_config,
    suite_digest,
    without_documents,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import golden_document

pytestmark = pytest.mark.usefixtures("no_network")


def lines_of(report: dict[str, Any]) -> list[str]:
    return render_markdown(report).split("\n")


def flagged_alpha() -> Any:
    """`alpha` whose first direction holds one empty text, two `no_output` and one truncated."""
    outputs = golden_outputs("alpha")
    for row_id, text, status in (
        ("g01", "", "empty"),
        ("g02", "", "no_output"),
        ("g03", "", "no_output"),
        ("g04", "cut off", "truncated"),
    ):
        outputs[OutputKey(FIRST, row_id)] = Output(text, status)  # type: ignore[arg-type]
    return run_input(golden_suite(), file_system("alpha", outputs))


def report_with_flagged_alpha() -> dict[str, Any]:
    return build_report(
        [flagged_alpha(), golden_run("beta")],
        golden_suites(),
        baseline="alpha",
        config=shipped_config(),
    )


def metric_table(lines: list[str], heading: str) -> list[str]:
    start = lines.index(f"#### {heading}")
    end = next(i for i in range(start + 2, len(lines)) if lines[i] == "")
    return lines[start + 2 : end]


# ---- I3: the statuses beside the metric ---------------------------------------------------------------


def test_every_score_table_has_an_ok_column_for_each_system() -> None:
    lines = lines_of(
        build_report(
            golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
        )
    )

    for heading in ("chrF++", "chrF", "BLEU"):
        assert (
            metric_table(lines, heading)[0]
            == "| direction | alpha | alpha ok | beta | beta ok |"
        )


def test_an_all_ok_direction_says_n_of_n() -> None:
    lines = lines_of(
        build_report(
            golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
        )
    )

    row = next(
        line
        for line in metric_table(lines, "chrF++")
        if line.startswith(f"| {FIRST} |")
    )
    assert (
        row
        == f"| {FIRST} | 73.08 [54.91, 85.48] | 10 of 10 | 93.87 [85.24, 100.00] | 10 of 10 |"
    )


def test_flagged_outputs_are_named_beside_the_metric_with_their_counts() -> None:
    lines = lines_of(report_with_flagged_alpha())

    rows = [
        line
        for line in metric_table(lines, "chrF++")
        if line.startswith(f"| {FIRST} |")
    ]
    assert len(rows) == 1
    assert "| 6 of 10 (1 empty, 2 no_output, 1 truncated) |" in rows[0]
    assert rows[0].endswith("| 10 of 10 |")


@pytest.mark.parametrize("heading", ["chrF++", "chrF", "BLEU"])
def test_the_flagged_counts_are_in_every_metric_table_not_only_the_first(
    heading: str,
) -> None:
    lines = lines_of(report_with_flagged_alpha())

    row = next(
        line for line in metric_table(lines, heading) if line.startswith(f"| {FIRST} |")
    )
    assert "6 of 10 (1 empty, 2 no_output, 1 truncated)" in row


def test_the_other_direction_of_the_same_system_stays_n_of_n() -> None:
    lines = lines_of(report_with_flagged_alpha())

    row = next(
        line
        for line in metric_table(lines, "chrF++")
        if line.startswith(f"| {SECOND} |")
    )
    assert row.count("10 of 10") == 2


def test_a_direction_that_is_not_in_the_suite_has_no_counts() -> None:
    lines = lines_of(report_with_flagged_alpha())

    row = next(
        line
        for line in metric_table(lines, "chrF++")
        if line.startswith("| khk_Cyrl-deu_Latn |")
    )
    assert (
        row
        == "| khk_Cyrl-deu_Latn | n/a (not in the suite) | n/a | n/a (not in the suite) | n/a |"
    )


def test_the_counts_are_read_from_the_statuses_of_the_results_and_nothing_else() -> (
    None
):
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    result = next(
        r
        for r in report["results"]
        if (r["system"], r["direction"]) == ("beta", SECOND)
    )
    result["statuses"] = {"ok": 7, "too_long": 3}

    lines = lines_of(report)

    row = next(
        line for line in metric_table(lines, "BLEU") if line.startswith(f"| {SECOND} |")
    )
    assert row.endswith("| 7 of 10 (3 too_long) |")


def test_no_key_is_added_to_the_report_for_the_counts() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )

    assert set(report) == set(SECTIONS)
    assert set(report["results"][0]) == {
        "schema",
        "system",
        "suite",
        "direction",
        "n_segments",
        "trained",
        "statuses",
        "metrics",
        "bleu_signature",
        "chrf_signature",
        "chrf_pp_signature",
        "unmeasured",
        "ci95",
        "bootstrap",
        "manifest_sha",
        "provenance",
    }


# ---- I4: the dataset and the revision ------------------------------------------------------------------


SUITES_HEADER = "| suite | kind | benchmark | split | purpose | rows | directions | dataset | revision | sha256 |"


def suite_row(report: dict[str, Any], name: str = "golden") -> list[str]:
    lines = lines_of(report)
    assert SUITES_HEADER in lines
    row = next(line for line in lines if line.startswith(f"| {name} |"))
    return [cell.strip() for cell in row.strip("|").split("|")]


def with_source(**source: Any) -> dict[str, Any]:
    report = golden_document()
    report["suites"][0]["source"].update(source)
    return report


def test_a_fixture_has_no_source_record_and_says_why() -> None:
    row = suite_row(golden_document())

    assert row[7:9] == ["n/a (no source record)", "n/a (no source record)"]


def test_a_pinned_suite_names_the_dataset_and_the_revision() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    row = suite_row(
        with_source(dataset="org/some-benchmark", revision=revision, pinned=True)
    )

    assert row[7:9] == ["org/some-benchmark", revision]


def test_a_dataset_without_a_revision_is_unpinned() -> None:
    row = suite_row(
        with_source(dataset="org/some-benchmark", revision=None, pinned=False)
    )

    assert row[7:9] == ["org/some-benchmark", "unpinned"]


def test_a_revision_that_was_recorded_and_not_pinned_says_so() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    row = suite_row(
        with_source(dataset="org/some-benchmark", revision=revision, pinned=False)
    )

    assert row[7:9] == ["org/some-benchmark", f"{revision} (unpinned)"]


def test_the_suites_table_names_no_path_and_no_text() -> None:
    markdown = render_markdown(
        with_source(dataset="org/some-benchmark", revision=None, pinned=False)
    )

    assert "/home" not in markdown and "/tmp" not in markdown


# ---- M6: the verdict ------------------------------------------------------------------------------------


COMPARISON_HEADER = "| direction | system | metric | delta | interval | p | verdict |"


def test_a_significantly_better_system_reads_better() -> None:
    lines = lines_of(
        build_report(
            golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
        )
    )

    assert COMPARISON_HEADER in lines
    assert (
        f"| {FIRST} | beta | chrF++ | +20.79 | [+13.92, +33.41] | 0.002 | better |"
        in lines
    )


def test_a_significantly_worse_system_reads_worse_and_not_no() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="beta", config=shipped_config()
    )

    lines = lines_of(report)

    assert (
        f"| {FIRST} | alpha | chrF++ | -20.79 | [-33.41, -13.92] | 0.002 | worse |"
        in lines
    )
    assert not any("| no |" in line for line in lines)


def test_a_difference_that_is_not_significant_says_so_in_words() -> None:
    plain = without_documents(golden_suite())
    report = build_report(
        [
            run_input(plain, file_system("alpha")),
            run_input(plain, file_system("gamma", mixed_outputs())),
        ],
        {suite_digest(plain): plain},
        baseline="alpha",
        config=shipped_config(),
    )

    lines = lines_of(report)

    assert (
        f"| {FIRST} | gamma | chrF++ | +8.22 | [+1.95, +16.52] | 0.052 | no significant difference |"
        in lines
    )


def test_a_comparison_without_a_p_value_says_n_a_with_the_reason() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    report["comparisons"][0].update(
        ci95=None, p_value=None, better_fraction=None, significant=False
    )

    lines = lines_of(report)

    assert (
        f"| {FIRST} | beta | chrF++ | +20.79 | n/a | n/a | n/a (fewer than two units) |"
        in lines
    )


def test_the_sentence_above_the_table_explains_the_three_verdicts() -> None:
    lines = lines_of(
        build_report(
            golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
        )
    )

    sentence = next(line for line in lines if line.startswith("Baseline: alpha. "))
    assert sentence == (
        "Baseline: alpha. The verdict is the p-value's, not the interval's: `better` is a "
        "p-value below 0.05 with the difference in the system's favour, `worse` is a p-value "
        "below 0.05 with the difference against it, and `no significant difference` is "
        "anything else."
    )


def test_the_star_and_the_significant_column_are_gone_from_the_markdown() -> None:
    markdown = render_markdown(
        build_report(
            golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
        )
    )

    assert "| significant |" not in markdown
    assert "A star marks" not in markdown
    assert " * |" not in markdown


def test_report_json_keeps_the_boolean_meaning_the_first_system_is_better() -> None:
    worse = build_report(
        golden_runs(), golden_suites(), baseline="beta", config=shipped_config()
    )

    assert all(c["significant"] is False for c in worse["comparisons"])
    assert all(c["delta"] < 0 and c["p_value"] < 0.05 for c in worse["comparisons"])


# ---- M7: the robustness suite has no kind --------------------------------------------------------------


def test_the_robustness_measurement_says_that_its_suite_kind_is_not_recorded() -> None:
    report = build_report(
        [
            run_input(
                golden_suite(),
                file_system("alpha"),
                robustness=robustness_input("alpha"),
            ),
            golden_run("beta"),
        ],
        golden_suites(),
        baseline="alpha",
        config=shipped_config(),
    )

    markdown = render_markdown(report)

    assert (
        "Measured on the suite robust (e5e5e5e5e5e5, suite kind not recorded), bound to the "
        "system by name."
    ) in markdown
    assert "so the evidence banner above does not count it" in markdown


# ---- the verdict follows the p-value, alpha and the direction of the metric ----------------------------


def comparison_row(report: dict[str, Any], suffix: str) -> str:
    return next(
        line
        for line in lines_of(report)
        if line.startswith(f"| {FIRST} | beta | chrF++ |") and line.endswith(suffix)
    )


@pytest.mark.parametrize(
    ("p_value", "verdict"),
    [
        (0.049, "better"),
        (0.0499, "better"),
        (0.05, "no significant difference"),
        (0.052, "no significant difference"),
        (0.011, "better"),
    ],
)
def test_the_verdict_is_the_p_value_against_alpha_nothing_else(
    p_value: float, verdict: str
) -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    report["comparisons"][0]["p_value"] = p_value

    assert comparison_row(report, f"| {verdict} |")


def test_the_verdict_uses_the_alpha_of_the_report_and_says_it() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    report["bootstrap"]["alpha"] = 0.1
    report["comparisons"][0]["p_value"] = 0.08

    markdown = render_markdown(report)

    assert comparison_row(report, "| better |")
    assert "`better` is a p-value below 0.1 with the difference" in markdown


def test_a_metric_that_is_lower_is_better_reads_the_other_way() -> None:
    # No count metric is lower-is-better (MetricX would be); the renderer follows the comparison's
    # own `higher_is_better` all the same, so a positive delta of such a metric is worse.
    report = build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )
    report["comparisons"][0]["higher_is_better"] = False

    assert comparison_row(report, "| worse |")
