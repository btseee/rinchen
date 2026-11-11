"""The Markdown summary of a maintenance report (D14, spec M11): rendered from the report alone.

Pinned cell by cell on small reports built by hand (`tests/support/maintain_reports.py`): the
heading with the state and the cadence, the table of checks, the drift table and its cap of 200
rows, the rerun table, one `What to do` line for each failing step and the fixed sentence of the
blind spot. A cell escapes the characters that would break a table. The golden file of the whole
run is pinned in `test_maintain_report_golden.py`.
"""

from __future__ import annotations

import pytest

from rinchen.maintain.report import STEP_IDS
from rinchen.maintain.report_markdown import MAX_DRIFT_ROWS, render_markdown
from tests.support.maintain_reports import (
    check_row,
    doctor_detail,
    drift_detail,
    drift_entry,
    report_of,
    rerun_item,
    reruns_detail,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

BLIND_SPOT = (
    "The drift table is what uv could take within the constraints of pyproject.toml; "
    "a package pinned with == never appears in it (docs/runbooks/14_maintenance.md)."
)


def lines_of(markdown: str) -> list[str]:
    return markdown.split("\n")


def test_the_heading_names_the_state_and_the_cadence() -> None:
    assert lines_of(render_markdown(report_of()))[0] == (
        "# Maintenance check: ok (nightly cadence)"
    )
    markdown = render_markdown(report_of({"drift": "drift"}, cadence="weekly"))
    assert lines_of(markdown)[0] == "# Maintenance check: drift (weekly cadence)"
    failing = render_markdown(report_of({"reruns": "fail"}))
    assert lines_of(failing)[0] == "# Maintenance check: fail (nightly cadence)"


def test_the_checks_table_has_one_row_per_doctor_check_and_one_per_other_step() -> None:
    report = report_of(
        {
            "doctor": "fail",
            "environment": "fail",
            "drift": "skipped",
            "reruns": "skipped",
        },
        {
            "environment": "1 packages differ from the lock, 0 are not in it",
            "drift": "disabled by --no-drift",
        },
        doctor=doctor_detail(
            check_row("D-PY"),
            check_row("D-EOLE-PY", "fail", "the check failed"),
            check_row(
                "D-TORCH-CUDA",
                "skipped",
                "a GPU check: maintain check never asks a device question",
            ),
        ),
    )

    markdown = render_markdown(report)

    assert (
        "## Checks\n\n"
        "| step | check | state | reason |\n"
        "| --- | --- | --- | --- |\n"
        "| doctor | D-PY | ok |  |\n"
        "| doctor | D-EOLE-PY | fail | the check failed |\n"
        "| doctor | D-TORCH-CUDA | skipped | a GPU check: maintain check never asks a device question |\n"
        "| lock_file | - | ok |  |\n"
        "| lock_consistent | - | ok |  |\n"
        "| environment | - | fail | 1 packages differ from the lock, 0 are not in it |\n"
        "| drift | - | skipped | disabled by --no-drift |\n"
        "| reruns | - | skipped |  |\n"
    ) in markdown


def test_a_doctor_with_no_checks_listed_is_one_row_of_the_step() -> None:
    markdown = render_markdown(report_of())

    assert "| doctor | - | ok |  |\n" in markdown


def test_the_drift_table_lists_the_entries_in_the_order_of_the_report() -> None:
    entries = [
        drift_entry(
            "huggingface-hub",
            locked="2.3.0",
            available="2.3.1",
            klass="blocked",
            rule="hub-client",
            requires=["evaluation_tiny", "gate"],
        ),
        drift_entry("rich", locked="15.0.0", available="15.0.1"),
        drift_entry("tomli-w", "add", None, "1.2.0"),
        drift_entry("sortedcontainers", "remove", "2.4.0", None),
    ]

    markdown = render_markdown(
        report_of({"drift": "drift"}, drift=drift_detail(entries))
    )

    assert (
        "## Dependency drift\n\n"
        "State: drift. 4 entries (2 update, 1 add, 1 remove; 3 drift, 1 blocked).\n\n"
        "| package | change | locked | available | class | needs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| huggingface-hub | update | 2.3.0 | 2.3.1 | blocked | evaluation_tiny, gate |\n"
        "| rich | update | 15.0.0 | 15.0.1 | drift | - |\n"
        "| tomli-w | add | - | 1.2.0 | drift | - |\n"
        "| sortedcontainers | remove | 2.4.0 | - | drift | - |\n\n"
    ) in markdown


@pytest.mark.parametrize(
    ("state", "reason", "line"),
    [
        ("ok", None, "State: ok."),
        (
            "skipped",
            "disabled by --no-drift",
            "State: skipped. Reason: disabled by --no-drift.",
        ),
        (
            "fail",
            "uv exited with code 1",
            "State: fail. Reason: uv exited with code 1.",
        ),
    ],
)
def test_a_drift_step_without_a_table_says_its_state_and_reason(
    state: str, reason: str | None, line: str
) -> None:
    reasons = {} if reason is None else {"drift": reason}

    markdown = render_markdown(report_of({"drift": state}, reasons))

    assert f"## Dependency drift\n\n{line}\n\n{BLIND_SPOT}\n" in markdown
    assert "| package |" not in markdown


def test_a_table_of_250_entries_shows_200_rows_and_says_how_many_more_the_json_holds() -> (
    None
):
    entries = [drift_entry(f"package-{n:03d}") for n in range(250)]

    markdown = render_markdown(
        report_of({"drift": "drift"}, drift=drift_detail(entries))
    )

    assert MAX_DRIFT_ROWS == 200
    rows = [line for line in lines_of(markdown) if line.startswith("| package-")]
    assert len(rows) == 200
    assert rows[0].startswith("| package-000 |") and rows[-1].startswith(
        "| package-199 |"
    )
    assert "50 more entries are in maintain_report.json" in lines_of(markdown)
    assert "package-200" not in markdown


@pytest.mark.parametrize(("count", "more"), [(200, None), (201, 1), (199, None)])
def test_the_count_line_appears_only_when_rows_were_cut(
    count: int, more: int | None
) -> None:
    entries = [drift_entry(f"package-{n:03d}") for n in range(count)]

    markdown = render_markdown(
        report_of({"drift": "drift"}, drift=drift_detail(entries))
    )

    if more is None:
        assert "more entries are in maintain_report.json" not in markdown
    else:
        assert f"{more} more entries are in maintain_report.json\n" in markdown


def test_the_rerun_table_lists_every_rerun_with_its_counters_and_reason() -> None:
    items = reruns_detail(
        rerun_item("numerics", tests=(1110, 0, 0, 0)),
        rerun_item("parity_tiny", "fail", "3 failed and 1 errors", tests=(9, 3, 1, 0)),
        rerun_item(
            "parity_export", "skipped", "no export was given", kind="export_parity"
        ),
        rerun_item(
            "serving_tiny",
            "skipped",
            "not due at the nightly cadence",
            cadence="weekly",
        ),
    )

    markdown = render_markdown(report_of({"reruns": "fail"}, reruns=items))

    assert (
        "## Parity reruns\n\n"
        "| id | state | passed | failed | skipped | reason |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| numerics | ok | 1110 | 0 | 0 |  |\n"
        "| parity_tiny | fail | 9 | 3 | 0 | 3 failed and 1 errors |\n"
        "| parity_export | skipped | - | - | - | no export was given |\n"
        "| serving_tiny | skipped | - | - | - | not due at the nightly cadence |\n"
    ) in markdown


def test_a_reruns_step_with_no_items_says_its_state() -> None:
    markdown = render_markdown(
        report_of({"reruns": "skipped"}, {"reruns": "disabled by --no-reruns"})
    )

    assert (
        "## Parity reruns\n\nState: skipped. Reason: disabled by --no-reruns.\n"
        in markdown
    )
    assert "| id | state |" not in markdown


def test_every_failing_step_has_one_what_to_do_line() -> None:
    report = report_of({"doctor": "fail", "lock_consistent": "fail", "reruns": "fail"})

    lines = lines_of(render_markdown(report))

    todo = [line for line in lines if "What to do" in line]
    assert todo == [
        "- What to do (doctor): docs/runbooks/14_maintenance.md#failure-doctor",
        "- What to do (lock_consistent): docs/runbooks/14_maintenance.md#failure-lock_consistent",
        "- What to do (reruns): docs/runbooks/14_maintenance.md#failure-reruns",
    ]
    assert lines[lines.index("## Failures") + 1] == ""


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_each_step_can_fail_and_points_at_its_own_section(step_id: str) -> None:
    states = {step_id: "fail"}

    markdown = render_markdown(report_of(states))

    assert f"#failure-{step_id}\n" in markdown
    assert markdown.count("What to do") == 1


def test_a_report_without_a_failure_has_no_failures_section() -> None:
    markdown = render_markdown(report_of({"drift": "drift", "reruns": "skipped"}))

    assert "What to do" not in markdown
    assert "## Failures" not in markdown


def test_the_blind_spot_sentence_is_in_every_summary_exactly_once() -> None:
    for states in ({}, {"drift": "drift"}, {"drift": "fail"}, {"doctor": "fail"}):
        assert render_markdown(report_of(states)).count(BLIND_SPOT) == 1


def test_a_cell_escapes_the_pipe_the_backtick_and_the_backslash() -> None:
    report = report_of(
        {"drift": "drift"},
        drift=drift_detail([drift_entry("a|b`c\\d", locked="1|2", available="3`4")]),
        doctor=doctor_detail(check_row("D-X|Y", "fail", "bad | `reason`")),
    )

    markdown = render_markdown(report)

    assert "| a\\|b\\`c\\\\d | update | 1\\|2 | 3\\`4 | drift | - |\n" in markdown
    assert "| doctor | D-X\\|Y | fail | bad \\| \\`reason\\` |\n" in markdown
    for line in lines_of(markdown):
        if line.startswith("| a\\|b"):
            assert line.replace("\\\\", "").replace("\\|", "").count("|") == 7


def test_a_cell_never_breaks_a_line() -> None:
    report = report_of(
        {"doctor": "fail"},
        doctor=doctor_detail(check_row("D-X", "fail", "one\ntwo\r\nthree")),
    )

    markdown = render_markdown(report)

    assert "| doctor | D-X | fail | one two three |\n" in markdown


def test_the_markdown_has_line_feeds_only_and_one_final_line_feed() -> None:
    markdown = render_markdown(
        report_of(
            {"drift": "drift", "doctor": "fail"}, drift=drift_detail([drift_entry("a")])
        )
    )

    assert markdown.endswith("\n") and not markdown.endswith("\n\n")
    assert "\r" not in markdown
    assert "\n\n\n" not in markdown


def test_the_markdown_is_a_function_of_the_report() -> None:
    assert render_markdown(report_of({"drift": "drift"})) == render_markdown(
        report_of({"drift": "drift"})
    )
    assert render_markdown(report_of()) != render_markdown(report_of(cadence="weekly"))
