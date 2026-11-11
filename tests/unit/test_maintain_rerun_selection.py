"""Why a rerun was not made when `--rerun` named others (D14 fix wave 1, finding M6).

`--rerun X` runs exactly the reruns it names, whatever the cadence. The other reruns of the
configuration were reported as `not due at the <cadence> cadence`, which is false for a rerun
that IS due and was left out by the selection. They are now `skipped` with the reason `not
selected with --rerun`. Without a selection nothing changes: a rerun that is not due says so. A
selected rerun that was not run keeps its own reason (the environment, no export), and the step
as a whole is judged by the first selected rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.maintain import states
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.report import report_document
from rinchen.maintain.report_markdown import render_markdown
from rinchen.maintain.states import (
    SKIP_ENVIRONMENT,
    SKIP_NO_EXPORT,
    SKIP_NOT_SELECTED,
    skip_not_due,
)
from tests.support.maintain_check import (
    Doctor,
    execute,
    finding,
    make_options,
    make_seams,
    script_passing_reruns,
    step_of,
    uv_call,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.maintain_scans import closed_reason
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import NO_CHANGE_MEASURED

pytestmark = pytest.mark.usefixtures("no_network")

NIGHTLY = ("numerics", "parity_tiny", "parity_export")
WEEKLY = ("robustness_tiny", "evaluation_tiny", "serving_tiny")


def uv_runner() -> FakeRunner:
    return (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )


def reasons_of_items(
    tmp_path: Path, runner: FakeRunner, **options: Any
) -> dict[str, tuple[str, str | None]]:
    run = execute(tmp_path, runner, options=make_options(**options))
    items = step_of(report_document(run.result.report), "reruns")["detail"]["items"]  # type: ignore[index]
    return {item["id"]: (item["state"], item["reason"]) for item in items}


def test_the_reason_is_a_fixed_closed_string() -> None:
    assert SKIP_NOT_SELECTED == "not selected with --rerun"
    assert SKIP_NOT_SELECTED in states.ALL_REASONS
    assert SKIP_NOT_SELECTED not in states.SKIP_REASONS
    assert closed_reason(SKIP_NOT_SELECTED)


def test_the_other_reruns_are_not_selected_whatever_their_cadence(
    tmp_path: Path,
) -> None:
    runner = script_passing_reruns(uv_runner(), "numerics")

    by_id = reasons_of_items(tmp_path, runner, selected=("numerics",))

    assert by_id["numerics"] == ("ok", None)
    for rerun_id in (*NIGHTLY[1:], *WEEKLY):
        assert by_id[rerun_id] == ("skipped", SKIP_NOT_SELECTED), rerun_id


def test_a_selected_weekly_rerun_leaves_the_due_nightly_ones_not_selected(
    tmp_path: Path,
) -> None:
    runner = script_passing_reruns(uv_runner(), "serving_tiny")

    by_id = reasons_of_items(tmp_path, runner, selected=("serving_tiny",))

    assert by_id["serving_tiny"] == ("ok", None)
    assert by_id["numerics"] == ("skipped", SKIP_NOT_SELECTED)
    assert by_id["parity_tiny"] == ("skipped", SKIP_NOT_SELECTED)


def test_with_the_weekly_cadence_a_due_rerun_that_was_left_out_is_not_selected(
    tmp_path: Path,
) -> None:
    runner = script_passing_reruns(uv_runner(), "numerics")

    by_id = reasons_of_items(tmp_path, runner, cadence="weekly", selected=("numerics",))

    assert by_id["serving_tiny"] == ("skipped", SKIP_NOT_SELECTED)
    assert by_id["parity_tiny"] == ("skipped", SKIP_NOT_SELECTED)


def test_two_selected_reruns_run_and_the_rest_are_not_selected(tmp_path: Path) -> None:
    runner = script_passing_reruns(uv_runner(), "numerics", "serving_tiny")

    by_id = reasons_of_items(tmp_path, runner, selected=("numerics", "serving_tiny"))

    assert by_id["numerics"] == ("ok", None)
    assert by_id["serving_tiny"] == ("ok", None)
    assert by_id["parity_tiny"] == ("skipped", SKIP_NOT_SELECTED)
    assert by_id["evaluation_tiny"] == ("skipped", SKIP_NOT_SELECTED)


def test_without_a_selection_a_rerun_that_is_not_due_says_so(tmp_path: Path) -> None:
    runner = script_passing_reruns(uv_runner(), "numerics", "parity_tiny")

    by_id = reasons_of_items(tmp_path, runner)

    for rerun_id in WEEKLY:
        assert by_id[rerun_id] == ("skipped", skip_not_due("nightly")), rerun_id


def test_a_selected_rerun_that_could_not_run_keeps_its_own_reason(
    tmp_path: Path,
) -> None:
    runner = uv_runner()

    by_id = reasons_of_items(tmp_path, runner, selected=("parity_export",))

    assert by_id["parity_export"] == ("skipped", SKIP_NO_EXPORT)
    assert by_id["numerics"] == ("skipped", SKIP_NOT_SELECTED)


def test_a_selected_rerun_after_a_failed_doctor_is_skipped_for_the_environment(
    tmp_path: Path,
) -> None:
    runner = uv_runner()
    doctor = Doctor([finding("D-PY", "fail")])

    run = execute(
        tmp_path,
        runner,
        options=make_options(selected=("serving_tiny",)),
        seams=make_seams(runner, doctor=doctor),
    )

    items = step_of(report_document(run.result.report), "reruns")["detail"]["items"]  # type: ignore[index]
    by_id = {item["id"]: item["reason"] for item in items}
    assert by_id["serving_tiny"] == SKIP_ENVIRONMENT
    assert by_id["numerics"] == SKIP_NOT_SELECTED


def test_the_step_that_ran_nothing_is_judged_by_the_first_selected_rerun(
    tmp_path: Path,
) -> None:
    run = execute(
        tmp_path, uv_runner(), options=make_options(selected=("parity_export",))
    )

    step = step_of(report_document(run.result.report), "reruns")

    assert (step["state"], step["reason"]) == ("skipped", SKIP_NO_EXPORT)


def test_the_summary_shows_the_reason(tmp_path: Path) -> None:
    runner = script_passing_reruns(uv_runner(), "numerics")
    run = execute(tmp_path, runner, options=make_options(selected=("numerics",)))

    markdown = render_markdown(run.result.report)

    assert (
        "| parity_tiny | skipped | - | - | - | not selected with --rerun |" in markdown
    )
    assert "not due" not in markdown
