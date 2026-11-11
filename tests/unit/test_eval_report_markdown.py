"""The Markdown report of D13 (Task 8, spec T14): `render_markdown` is a pure function of the parsed
report, so the same document gives the same bytes in every process, locale and time zone.

The sections come in a fixed order; a score is `value [low, high]` with two decimals, a p-value has
three, a delta a sign; whatever was not measured is `n/a (<state or reason>)` and never a number;
a star marks a significant comparison; the evidence banner is a function of the kinds of suites.
The whole golden report is pinned byte for byte in `test_eval_report_golden.py`; here the lines
the brief names are pinned one by one, and the rest of the behaviour with small reports.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from typing import Any

import pytest

from rinchen.eval.outputs import OutputKey
from rinchen.eval.report import build_report, report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.systems import NOT_TRAINED_REASON, SystemRecord, Verdict
from tests.support.eval_fakes import FakeSuiteSystem, golden_outputs, golden_suite
from tests.support.eval_report import (
    MANIFEST_SHA256,
    MODEL_SHA256,
    ROOT,
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
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

BANNER = (
    "These figures come from generated fixture suites. They test the tooling and say "
    "nothing about translation quality."
)
SECTIONS = [
    "## Systems",
    "## Suites",
    "## Scores",
    "## Paired comparisons",
    "## Robustness",
    "## Not measured",
    "## Release targets",
    "## Provenance",
]


def golden_report() -> dict[str, Any]:
    return build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )


def golden_markdown() -> str:
    return render_markdown(golden_report())


def lines_of(markdown: str) -> list[str]:
    return markdown.split("\n")


def block_after(markdown: str, *headings: str) -> list[str]:
    """The lines under the last of `headings` (found one after the other) up to the next
    heading of any level."""
    lines = lines_of(markdown)
    position = 0
    for heading in headings:
        position = lines.index(heading, position) + 1
    end = next(
        (i for i in range(position, len(lines)) if lines[i].startswith("#")), len(lines)
    )
    return lines[position:end]


def not_trained_report() -> dict[str, Any]:
    """`alpha` (a file) and `toy` (a model of ours that was not trained for the third direction)
    over the golden suite with a third direction."""
    suite = with_third_direction(golden_suite())
    toy = FakeSuiteSystem(
        verdicts={THIRD: Verdict("not_trained", NOT_TRAINED_REASON)},
        answer=lambda requests: [
            golden_outputs("beta")[OutputKey(r.direction, r.id)] for r in requests
        ],
        record=SystemRecord(
            kind="rinchen",
            name="toy",
            model_sha256=MODEL_SHA256,
            manifest_sha256=MANIFEST_SHA256,
        ),
    )
    runs = [run_input(suite, file_system("alpha")), run_input(suite, toy)]
    return build_report(
        runs, {suite_digest(suite): suite}, baseline="alpha", config=shipped_config()
    )


# ---- the layout --------------------------------------------------------------------------------


def test_the_sections_come_in_the_fixed_order() -> None:
    headings = [line for line in lines_of(golden_markdown()) if line.startswith("## ")]

    assert headings == SECTIONS


def test_the_document_starts_with_its_title_and_the_banner() -> None:
    lines = lines_of(golden_markdown())

    assert lines[0] == "# Evaluation report"
    assert lines[1] == ""
    assert lines[2] == BANNER


def test_the_text_ends_with_one_line_feed_and_has_no_carriage_return() -> None:
    markdown = golden_markdown()

    assert markdown.endswith("\n") and not markdown.endswith("\n\n")
    assert "\r" not in markdown


def test_the_fixture_banner_is_the_sentence_of_the_brief() -> None:
    assert BANNER in lines_of(golden_markdown())


def test_a_report_with_a_benchmark_beside_a_fixture_says_which_kinds_were_measured() -> (
    None
):
    benchmark = replace(
        golden_suite(), name="bench", kind="benchmark", benchmark="flores_plus"
    )
    runs = [*golden_runs(), run_input(benchmark, file_system("alpha"))]
    report = build_report(
        runs,
        {**golden_suites(), suite_digest(benchmark): benchmark},
        baseline=None,
        config=shipped_config(),
    )

    banner = lines_of(render_markdown(report))[2]

    assert BANNER not in banner
    assert banner == (
        "Evidence: suites of the kinds benchmark, fixture. The suites of the kind fixture "
        "are generated: they test the tooling and say nothing about translation quality."
    )


def test_a_report_without_a_fixture_says_what_a_measurement_is() -> None:
    benchmark = replace(
        golden_suite(), name="bench", kind="benchmark", benchmark="flores_plus"
    )
    report = build_report(
        [run_input(benchmark, file_system("alpha"))],
        {suite_digest(benchmark): benchmark},
        baseline=None,
        config=shipped_config(),
    )

    banner = lines_of(render_markdown(report))[2]

    assert banner == (
        "Evidence: suites of the kind benchmark. Each figure is a measurement of the "
        "system named, on the suite named, and says nothing about other text."
    )


# ---- the scores --------------------------------------------------------------------------------


def test_the_chrf_pp_table_has_the_system_columns_and_the_document_unit_cells() -> None:
    table = block_after(golden_markdown(), "### golden", "#### chrF++")

    assert table[1] == "| direction | alpha | beta |"
    assert table[2] == "| --- | --- | --- |"
    assert (
        "| khk_Cyrl-eng_Latn | 73.08 [54.91, 85.48] | 93.87 [85.24, 100.00] |" in table
    )
    assert (
        "| eng_Latn-khk_Cyrl | 68.06 [50.67, 80.98] | 91.34 [85.07, 97.86] |" in table
    )


def test_the_rows_are_the_standard_directions_in_order_then_the_suites_own() -> None:
    suite = with_third_direction(golden_suite())
    report = build_report(
        [run_input(suite, file_system("alpha"))],
        {suite_digest(suite): suite},
        baseline=None,
        config=shipped_config(),
    )

    table = block_after(render_markdown(report), "### golden", "#### chrF++")

    rows = [line.split(" | ")[0].removeprefix("| ") for line in table[3:] if line]
    assert rows == [
        "khk_Cyrl-deu_Latn",
        "deu_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-jpn_Jpan",
        "jpn_Jpan-khk_Cyrl",
        "khk_Cyrl-kaz_Cyrl",
        "kaz_Cyrl-khk_Cyrl",
        "khk_Cyrl-kor_Hang",
        "kor_Hang-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-tur_Latn",
        "tur_Latn-khk_Cyrl",
        "khk_Cyrl-zho_Hans",
        "zho_Hans-khk_Cyrl",
        THIRD,
    ]


def test_a_direction_the_suite_does_not_declare_is_n_a_not_in_the_suite() -> None:
    table = block_after(golden_markdown(), "### golden", "#### chrF++")

    assert (
        "| khk_Cyrl-deu_Latn | n/a (not in the suite) | n/a (not in the suite) |"
        in table
    )


def test_a_direction_a_system_did_not_run_is_n_a_with_its_state() -> None:
    table = block_after(
        render_markdown(not_trained_report()), "### golden", "#### chrF++"
    )

    row = next(line for line in table if line.startswith(f"| {THIRD} |"))
    assert row == f"| {THIRD} | n/a (not_in_file) | n/a (not_trained) |"


def test_a_not_run_cell_is_never_a_number() -> None:
    markdown = render_markdown(not_trained_report())

    for line in lines_of(markdown):
        if line.startswith(f"| {THIRD} |"):
            assert "0.00" not in line


def test_the_three_count_metrics_have_a_table_each_in_the_order_chrf_pp_chrf_bleu() -> (
    None
):
    headings = [
        line for line in lines_of(golden_markdown()) if line.startswith("#### ")
    ]

    assert headings == ["#### chrF++", "#### chrF", "#### BLEU"]


def test_the_resampling_unit_of_a_suite_is_named_under_its_heading() -> None:
    block = block_after(golden_markdown(), "### golden")

    assert "Resampling unit: document (5 units in each direction)." in block


def test_the_resampling_unit_is_the_segment_when_no_row_has_a_document() -> None:
    plain = without_documents(golden_suite())
    report = build_report(
        [run_input(plain, file_system("alpha"))],
        {suite_digest(plain): plain},
        baseline=None,
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "### golden")

    assert "Resampling unit: segment (10 units in each direction)." in block


def test_an_interval_that_is_not_there_is_n_a_and_never_zero() -> None:
    report = golden_report()
    for result in report["results"]:
        result["ci95"]["chrf_pp"] = None

    table = block_after(render_markdown(report), "### golden", "#### chrF++")

    assert "| khk_Cyrl-eng_Latn | 73.08 [n/a] | 93.87 [n/a] |" in table


def test_a_score_has_two_decimals_and_a_p_value_three() -> None:
    report = golden_report()
    report["results"][0]["metrics"]["chrf_pp"] = 5.0
    report["results"][0]["ci95"]["chrf_pp"] = [4.9, 5.1]
    report["comparisons"][0]["p_value"] = 0.0004

    markdown = render_markdown(report)

    assert "| khk_Cyrl-eng_Latn | 5.00 [4.90, 5.10] |" in markdown
    assert "| 0.000 | * |" in markdown


def test_a_neural_metric_with_values_gets_its_table_and_one_without_is_listed_as_unmeasured() -> (
    None
):
    suite = golden_suite()
    scores = {
        "comet22": {key: 0.8 for key in golden_outputs("alpha")},
        "metricx24": {key: 2.0 for key in golden_outputs("alpha")},
    }
    alpha = run_input(suite, file_system("alpha"), scores=scores)
    report = build_report(
        [alpha], golden_suites(), baseline=None, config=shipped_config()
    )

    markdown = render_markdown(report)

    assert "#### COMET-22" in markdown
    assert "#### MetricX-24 (lower is better)" in markdown
    assert "#### MetricX-24-QE" not in markdown
    assert any(
        line.startswith("- metricx24_qe: no segment scores were given")
        for line in lines_of(markdown)
    )
    assert not any(line.startswith("- comet22:") for line in lines_of(markdown))


# ---- the comparisons ---------------------------------------------------------------------------


def test_a_comparison_row_has_the_delta_the_interval_the_p_value_and_a_star() -> None:
    markdown = golden_markdown()

    assert (
        "| khk_Cyrl-eng_Latn | beta | chrF++ | +20.79 | [+13.92, +33.41] | 0.002 | * |"
        in lines_of(markdown)
    )
    assert (
        "| khk_Cyrl-eng_Latn | beta | BLEU | +35.01 | [+22.91, +54.78] | 0.002 | * |"
        in lines_of(markdown)
    )


def test_the_comparison_table_names_the_baseline_and_has_its_header() -> None:
    lines = lines_of(golden_markdown())

    assert any(line.startswith("Baseline: alpha. ") for line in lines)
    assert (
        "| direction | system | metric | delta | interval | p | significant |" in lines
    )


def test_a_comparison_that_is_not_significant_has_an_empty_mark() -> None:
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

    lines = lines_of(render_markdown(report))

    assert (
        "| khk_Cyrl-eng_Latn | gamma | chrF++ | +8.22 | [+1.95, +16.52] | 0.052 |  |"
        in lines
    )


def test_a_negative_delta_has_its_sign() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline="beta", config=shipped_config()
    )

    lines = lines_of(render_markdown(report))

    assert (
        "| khk_Cyrl-eng_Latn | alpha | chrF++ | -20.79 | [-33.41, -13.92] | 0.002 |  |"
        in lines
    )


def test_a_comparison_without_an_interval_or_a_p_value_says_n_a() -> None:
    report = golden_report()
    comparison = report["comparisons"][0]
    comparison.update(ci95=None, p_value=None, better_fraction=None, significant=False)

    lines = lines_of(render_markdown(report))

    assert "| khk_Cyrl-eng_Latn | beta | chrF++ | +20.79 | n/a | n/a |  |" in lines


def test_the_note_about_multiple_comparisons_opens_the_section() -> None:
    block = block_after(golden_markdown(), "## Paired comparisons")

    assert block[1] == (
        "16 intervals and 4 paired comparisons are reported with no correction for "
        "multiple comparisons."
    )


def test_a_baseline_with_no_run_in_a_suite_is_said_and_nothing_is_compared() -> None:
    other = without_documents(golden_suite(), name="other")
    report = build_report(
        [*golden_runs(), run_input(other, file_system("beta"))],
        {**golden_suites(), suite_digest(other): other},
        baseline="alpha",
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "## Paired comparisons", "### other")

    assert block == [
        "",
        "The baseline alpha has no run on this suite: nothing is compared.",
        "",
    ]


def test_a_report_without_a_comparison_says_so() -> None:
    report = build_report(
        golden_runs(), golden_suites(), baseline=None, config=shipped_config()
    )

    block = block_after(render_markdown(report), "## Paired comparisons")

    assert (
        "No paired comparison was made: no baseline was named, or no other system ran "
        "with it."
    ) in block


# ---- the rest of the sections ----------------------------------------------------------------


def test_the_systems_table_has_the_twelve_digits_of_a_digest_and_n_a_for_none() -> None:
    suite = golden_suite()
    toy = rinchen_system("toy", golden_outputs("beta"))
    report = build_report(
        [run_input(suite, toy)], golden_suites(), baseline=None, config=shipped_config()
    )

    block = block_after(render_markdown(report), "## Systems")

    assert block[1] == "| system | kind | model | manifest | options |"
    assert block[3] == (
        f"| toy | rinchen | {MODEL_SHA256[:12]} | {MANIFEST_SHA256[:12]} | none |"
    )


def test_the_systems_table_lists_the_options_sorted() -> None:
    report = golden_report()
    report["systems"][0]["options"] = {"temperature": 0.0, "beam_size": 4}

    block = block_after(render_markdown(report), "## Systems")

    assert "| alpha | hypotheses | n/a | n/a | beam_size=4, temperature=0.0 |" in block


def test_the_suites_table_has_the_counts_and_the_digest_prefix() -> None:
    block = block_after(golden_markdown(), "## Suites")

    assert block[1] == (
        "| suite | kind | benchmark | split | purpose | rows | directions | sha256 |"
    )
    assert block[3] == (
        f"| golden | fixture | golden | devtest | report | 10 | 2 | {suite_digest(golden_suite())[:12]} |"
    )


def test_a_suite_that_lacks_languages_says_which() -> None:
    report = golden_report()
    report["suites"][0]["source"]["missing_languages"] = ["deu_Latn", "tur_Latn"]

    block = block_after(render_markdown(report), "## Suites")

    assert "The suite golden lacks deu_Latn, tur_Latn." in block


def test_the_robustness_section_without_a_run_says_so() -> None:
    block = block_after(golden_markdown(), "## Robustness")

    assert "No robustness run was given." in block


def test_the_robustness_section_has_the_six_values_with_their_counts() -> None:
    inp = robustness_input("alpha")
    report = build_report(
        [run_input(golden_suite(), file_system("alpha"), robustness=inp)],
        golden_suites(),
        baseline=None,
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "## Robustness", "### alpha")

    assert block[1] == "| value | result | scored | excluded | total |"
    assert "| numbers_acc | 0.9292 | 960 | 0 | 960 |" in block
    assert "| placeholder_acc | 0.9983 | 599 | 1 | 600 |" in block
    assert "| mong_source_retention | 0.9976 | 900 | 0 | 900 |" in block
    assert (
        "Measured on the suite robust (e5e5e5e5e5e5), bound to the system by name."
        in block
    )


def test_a_null_robustness_value_is_n_a_with_its_reason() -> None:
    inp = robustness_input(
        "alpha",
        block={
            "numbers_acc": None,
            "placeholder_acc": 1.0,
            "term_acc": None,
            "script_purity": 1.0,
            "entity_acc": None,
            "mong_source_retention": None,
        },
    )
    report = build_report(
        [run_input(golden_suite(), file_system("alpha"), robustness=inp)],
        golden_suites(),
        baseline=None,
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "## Robustness", "### alpha")

    assert "| numbers_acc | n/a (no scored unit) | 960 | 0 | 960 |" in block


def test_the_not_measured_list_names_the_neural_metrics_with_their_counts() -> None:
    block = block_after(golden_markdown(), "## Not measured")

    reason = "no segment scores were given: no scorer is built (B1 and B2 are open)"
    assert f"- comet22: {reason} (4 of 4 results)" in block
    assert f"- metricx24: {reason} (4 of 4 results)" in block
    assert f"- metricx24_qe: {reason} (4 of 4 results)" in block
    assert "- golden: 14 expected directions are not in the suite." in block
    assert (
        "- The share of outputs in the wrong language: no language identification is run."
        in block
    )


def test_the_not_measured_list_counts_the_directions_that_did_not_run() -> None:
    block = block_after(render_markdown(not_trained_report()), "## Not measured")

    assert "- alpha on golden: 1 direction did not run (not_in_file: 1)." in block
    assert "- toy on golden: 1 direction did not run (not_trained: 1)." in block


def test_the_targets_section_lists_the_four_targets_with_their_statuses() -> None:
    block = block_after(golden_markdown(), "## Release targets")

    assert block[1].startswith("A target is a measurement")
    assert "| target | status | value | threshold | basis |" in block
    assert any(
        line.startswith(
            "| beats_baseline | unmeasured | n/a | 12 | 0 of 16 directions measured"
        )
        for line in block
    )
    assert [line.split(" | ")[0] for line in block if line.startswith("| ")][2:] == [
        "| beats_baseline",
        "| off_target",
        "| numbers",
        "| placeholders",
    ]
    assert sum("| unmeasured |" in line for line in block) == 4


def test_the_provenance_has_the_bootstrap_the_signatures_and_the_run_digests() -> None:
    runs = golden_runs()
    report = build_report(
        runs, golden_suites(), baseline="alpha", config=shipped_config()
    )

    block = block_after(render_markdown(report), "## Provenance")

    assert (
        "- Bootstrap: 1000 resamples from the seed 20261110, percentile intervals at the "
        "level 0.95 (alpha 0.05), draws splitmix64-counter/1."
    ) in block
    assert (
        "- BLEU signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0"
        in block
    )
    assert (
        "- chrF++ signature: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0"
        in block
    )
    assert f"| golden | alpha | {runs[0].run_sha256[:12]} |" in block
    assert f"| golden | beta | {runs[1].run_sha256[:12]} |" in block


# ---- purity and determinism --------------------------------------------------------------------


def test_no_line_holds_a_path_or_a_timestamp() -> None:
    absolute = re.compile(r"(^|[\s(\[\"'])(/[A-Za-z0-9._~-]|[A-Za-z]:[\\/])")
    stamp = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")

    for line in lines_of(golden_markdown()):
        assert absolute.search(line) is None, line
        assert stamp.search(line) is None, line


def test_rendering_does_not_change_the_report_it_is_given() -> None:
    report = golden_report()
    before = copy.deepcopy(report)

    render_markdown(report)

    assert report == before


def test_the_report_read_back_from_its_bytes_renders_the_same_text() -> None:
    report = golden_report()

    again = json.loads(report_to_bytes(report))

    assert render_markdown(again) == render_markdown(report)


def test_rendering_twice_gives_equal_text() -> None:
    assert golden_markdown() == golden_markdown()


_SUBPROCESS = """
import sys
sys.path.insert(0, sys.argv[1])
from rinchen.eval.report import build_report, report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from tests.support.eval_report import golden_runs, golden_suites, shipped_config
report = build_report(golden_runs(), golden_suites(), baseline="alpha", config=shipped_config())
sys.stdout.buffer.write(report_to_bytes(report) + b"\\0" + render_markdown(report).encode("utf-8"))
"""


@pytest.mark.parametrize(
    "environment",
    [
        {"PYTHONHASHSEED": "1", "LC_ALL": "C", "TZ": "UTC"},
        {"PYTHONHASHSEED": "2", "LC_ALL": "C.UTF-8", "TZ": "Asia/Ulaanbaatar"},
        {"PYTHONHASHSEED": "random", "LANG": "tr_TR.UTF-8", "TZ": "America/New_York"},
    ],
)
def test_the_json_and_the_markdown_are_byte_identical_in_a_fresh_process(
    environment: dict[str, str],
) -> None:
    expected = report_to_bytes(golden_report()) + b"\0" + golden_markdown().encode()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LC_", "LANG", "PYTHONHASHSEED", "TZ"))
    }
    env.update(environment, CUDA_VISIBLE_DEVICES="")

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(ROOT)],
        capture_output=True,
        env=env,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == expected


def test_the_unmeasured_count_is_the_results_that_lack_the_metric_of_all_the_results() -> (
    None
):
    scores = {"comet22": {key: 0.8 for key in golden_outputs("alpha")}}
    alpha = run_input(golden_suite(), file_system("alpha"), scores=scores)
    report = build_report(
        [alpha, golden_run("beta")],
        golden_suites(),
        baseline=None,
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "## Not measured")

    reason = "no segment scores were given: no scorer is built (B1 and B2 are open)"
    assert f"- comet22: {reason} (2 of 4 results)" in block
    assert f"- metricx24: {reason} (4 of 4 results)" in block
    table = block_after(render_markdown(report), "### golden", "#### COMET-22")
    assert "| khk_Cyrl-eng_Latn | 0.80 [0.80, 0.80] | n/a (" in "\n".join(table)


def test_a_baseline_that_has_no_run_in_a_suite_is_not_a_direction_that_did_not_run() -> (
    None
):
    other = without_documents(golden_suite(), name="other")
    report = build_report(
        [*golden_runs(), run_input(other, file_system("beta"))],
        {**golden_suites(), suite_digest(other): other},
        baseline="alpha",
        config=shipped_config(),
    )

    block = block_after(render_markdown(report), "## Not measured")

    assert not any(" on other" in line for line in block)
    assert not any("absent" in line for line in block)


def test_a_text_with_a_bar_or_a_line_break_stays_one_cell() -> None:
    report = golden_report()
    report["targets"][0]["basis"] = "one | two\nthree"
    report["systems"][0]["options"] = {"note": "a|b"}

    lines = lines_of(render_markdown(report))

    assert any(line.endswith("| one \\| two three |") for line in lines)
    assert "| alpha | hypotheses | n/a | n/a | note=a\\|b |" in lines
