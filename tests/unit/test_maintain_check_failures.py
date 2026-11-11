"""The failures of a maintenance check (D14, spec M2, M4, M5, M6 and M8): each step ends in `fail`
with a reason of the closed list and the others still run, except the reruns, which are skipped
when the doctor or the environment check failed.

Against the scripted runner and the doubles of `tests/support/maintain_check.py`. The doctor's
message (it can hold an exception text and a path) goes to `doctor_messages` for the command to
print and into the report not at all. `D-TORCH-CUDA` is a configured skip: a finding of it is
dropped and the configured row stands in its place; a configured skip the doctor never reports is
listed all the same. The environment detail shows at most 20 names of each kind and counts the
rest; a locked package that is not installed is a count.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from rinchen.maintain.check import DoctorFinding
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.report import exit_code_of, report_document, report_to_bytes
from rinchen.maintain.report_markdown import render_markdown
from rinchen.maintain.states import (
    FAIL_DOCTOR,
    FAIL_UV_MISSING,
    SKIP_DOCTOR_SELF,
    SKIP_ENVIRONMENT,
    SKIP_GPU_CHECK,
    SKIP_NO_DRIFT,
    SKIP_NO_RERUNS,
    fail_environment,
    fail_floor,
    fail_timeout,
    fail_uv_exit,
    skip_not_due,
)
from tests.support.maintain_check import (
    CPU_CHECKS,
    DOCTOR_MESSAGE,
    GPU_CHECK,
    SHIPPED_CONFIG,
    UV,
    Doctor,
    Run,
    config_with,
    execute,
    finding,
    golden_doctor,
    golden_runner,
    make_inputs,
    make_options,
    make_seams,
    rerun_call,
    rerun_id_of,
    script_passing_reruns,
    step_of,
    uv_call,
)
from tests.support.maintain_fakes import FakeRunner, junit_xml
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    NO_CHANGE_MEASURED,
    OFFLINE_FAILURE_MEASURED,
    SAMPLE_LOCKED,
    lock_of,
)

pytestmark = pytest.mark.usefixtures("no_network")

WEEKLY_THREE = ("robustness_tiny", "evaluation_tiny", "serving_tiny")


def document_of(run: Run) -> dict[str, object]:
    return report_document(run.result.report)


def item_states(run: Run) -> dict[str, tuple[str, str | None]]:
    detail = step_of(document_of(run), "reruns")["detail"]
    return {i["id"]: (i["state"], i["reason"]) for i in detail["items"]}  # type: ignore[index]


def states_of(run: Run) -> dict[str, str]:
    return {step.id: step.state for step in run.result.report.steps}


def failing_doctor(*failed: str) -> Doctor:
    findings = [
        finding(check_id, "fail" if check_id in failed else "pass")
        for check_id in CPU_CHECKS
    ]
    return Doctor([*findings, finding(GPU_CHECK, "skip")])


def uv_only_runner(
    *,
    check: CommandResult | None = None,
    dry_run: CommandResult | None = None,
    dry_text: str = NO_CHANGE_MEASURED,
) -> FakeRunner:
    return (
        FakeRunner()
        .script(check or CommandResult(0, False), when=uv_call("--check"))
        .script(
            dry_run or CommandResult(0, False),
            log_text=dry_text,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )


def rerun_ids(runner: FakeRunner) -> list[str]:
    return [rerun_id_of(argv) for argv in runner.argvs if argv[0] != UV]


# ---- the doctor ---------------------------------------------------------------------------------------


def test_a_failed_doctor_check_fails_the_step_and_skips_every_selected_rerun(
    tmp_path: Path,
) -> None:
    runner = uv_only_runner()
    doctor = failing_doctor("D-EOLE-PY")

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=doctor))

    assert rerun_ids(runner) == []
    assert states_of(run)["doctor"] == "fail"
    assert states_of(run)["reruns"] == "skipped"
    assert item_states(run) == {
        "numerics": ("skipped", SKIP_ENVIRONMENT),
        "parity_tiny": ("skipped", SKIP_ENVIRONMENT),
        "parity_export": ("skipped", SKIP_ENVIRONMENT),
        **{rerun_id: ("skipped", skip_not_due("nightly")) for rerun_id in WEEKLY_THREE},
    }
    reruns = step_of(document_of(run), "reruns")
    assert (reruns["state"], reruns["reason"]) == ("skipped", SKIP_ENVIRONMENT)
    assert document_of(run)["state"] == "fail"
    assert exit_code_of(run.result.report, fail_on_drift=False) == 4


def test_the_other_steps_still_run_after_a_doctor_failure(tmp_path: Path) -> None:
    runner = uv_only_runner(dry_text=NO_CHANGE_MEASURED)

    run = execute(
        tmp_path, runner, seams=make_seams(runner, doctor=failing_doctor("D-PY"))
    )

    assert runner.argvs == [
        (UV, "lock", "--check"),
        (UV, "lock", "--upgrade", "--dry-run"),
    ]
    assert states_of(run) == {
        "doctor": "fail",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "ok",
        "reruns": "skipped",
    }


def test_a_failed_check_is_listed_with_the_closed_reason_and_its_message_goes_to_the_command(
    tmp_path: Path,
) -> None:
    runner = uv_only_runner()
    doctor = failing_doctor("D-EOLE-PY", "D-CT2-CONV")

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=doctor))

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    failed = [c for c in checks if c["state"] == "fail"]
    assert failed == [
        {"id": "D-EOLE-PY", "state": "fail", "reason": FAIL_DOCTOR},
        {"id": "D-CT2-CONV", "state": "fail", "reason": FAIL_DOCTOR},
    ]
    assert run.result.doctor_messages == (
        ("D-EOLE-PY", DOCTOR_MESSAGE),
        ("D-CT2-CONV", DOCTOR_MESSAGE),
    )
    everything = (
        report_to_bytes(run.result.report) + render_markdown(run.result.report).encode()
    )
    for text in (DOCTOR_MESSAGE, "eole failed", "/srv/", "site-packages"):
        assert text.encode() not in everything


def test_only_the_failed_checks_have_a_message_for_the_command(tmp_path: Path) -> None:
    run = execute(tmp_path, golden_runner())

    assert run.result.doctor_messages == ()


def test_a_skip_by_the_doctor_is_a_skip_with_the_closed_reason(tmp_path: Path) -> None:
    doctor = Doctor([finding("D-PY"), finding("D-WHEELS", "skip")])

    run = execute(
        tmp_path, golden_runner(), seams=make_seams(golden_runner(), doctor=doctor)
    )

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    assert [(c["id"], c["state"], c["reason"]) for c in checks] == [
        ("D-PY", "ok", None),
        ("D-WHEELS", "skipped", SKIP_DOCTOR_SELF),
        (GPU_CHECK, "skipped", SKIP_GPU_CHECK),
    ]
    assert states_of(run)["doctor"] == "ok"


def test_a_doctor_that_raises_is_one_failed_check_named_doctor(tmp_path: Path) -> None:
    runner = uv_only_runner()

    def broken() -> Sequence[DoctorFinding]:
        raise RuntimeError("the doctor fell over at /srv/example/x")

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=broken))

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    assert checks == [
        {"id": "doctor", "state": "fail", "reason": FAIL_DOCTOR},
        {"id": GPU_CHECK, "state": "skipped", "reason": SKIP_GPU_CHECK},
    ]
    assert states_of(run)["doctor"] == "fail"
    assert rerun_ids(runner) == []
    (message,) = run.result.doctor_messages
    assert message[0] == "doctor" and "RuntimeError" in message[1]
    assert b"fell over" not in report_to_bytes(run.result.report)


def test_a_finding_of_a_configured_skip_is_dropped_and_the_configured_row_stands_in(
    tmp_path: Path,
) -> None:
    doctor = Doctor([finding(GPU_CHECK, "fail"), *(finding(i) for i in CPU_CHECKS)])
    runner = golden_runner()

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=doctor))

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    assert [c["id"] for c in checks] == [GPU_CHECK, *CPU_CHECKS]
    assert checks[0] == {"id": GPU_CHECK, "state": "skipped", "reason": SKIP_GPU_CHECK}
    assert states_of(run)["doctor"] == "ok"
    assert run.result.doctor_messages == ()
    assert doctor.calls == 1
    assert rerun_ids(runner) == ["numerics", "parity_tiny"]


def test_a_configured_skip_the_doctor_never_reports_is_listed_after_its_findings(
    tmp_path: Path,
) -> None:
    doctor = Doctor([finding(i) for i in CPU_CHECKS])

    run = execute(
        tmp_path, golden_runner(), seams=make_seams(golden_runner(), doctor=doctor)
    )

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    assert [c["id"] for c in checks] == [*CPU_CHECKS, GPU_CHECK]
    assert checks[-1]["reason"] == SKIP_GPU_CHECK


def test_the_configured_skips_come_with_the_reasons_of_the_configuration(
    tmp_path: Path,
) -> None:
    config = config_with(
        skipped_doctor_checks={"D-WHEELS": SKIP_NO_DRIFT, GPU_CHECK: SKIP_GPU_CHECK}
    )
    runner = golden_runner()

    run = execute(
        tmp_path,
        runner,
        inputs=make_inputs(config),
        seams=make_seams(runner, doctor=golden_doctor()),
    )

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    by_id = {c["id"]: c for c in checks}
    assert (by_id["D-WHEELS"]["state"], by_id["D-WHEELS"]["reason"]) == (
        "skipped",
        SKIP_NO_DRIFT,
    )
    assert [c["id"] for c in checks].count("D-WHEELS") == 1


def test_a_configured_skip_reported_twice_is_listed_once_and_other_findings_as_given(
    tmp_path: Path,
) -> None:
    doctor = Doctor(
        [
            finding(GPU_CHECK, "skip"),
            finding("D-PY"),
            finding(GPU_CHECK, "fail"),
            finding("D-PY"),
        ]
    )
    runner = golden_runner()

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=doctor))

    checks = step_of(document_of(run), "doctor")["detail"]["checks"]  # type: ignore[index]
    assert [c["id"] for c in checks] == [GPU_CHECK, "D-PY", "D-PY"]
    assert checks[0]["reason"] == SKIP_GPU_CHECK


def test_a_doctor_with_nothing_to_report_is_a_skipped_step(tmp_path: Path) -> None:
    config = config_with(skipped_doctor_checks={})
    runner = golden_runner()

    run = execute(
        tmp_path,
        runner,
        inputs=make_inputs(config),
        seams=make_seams(runner, doctor=Doctor([])),
    )

    step = step_of(document_of(run), "doctor")
    assert (step["state"], step["reason"]) == ("skipped", SKIP_DOCTOR_SELF)
    assert step["detail"] == {"checks": []}
    assert rerun_ids(runner) == ["numerics", "parity_tiny"]


# ---- the environment --------------------------------------------------------------------------------


def test_an_installed_version_the_lock_does_not_list_fails_the_step_and_skips_the_reruns(
    tmp_path: Path,
) -> None:
    runner = uv_only_runner()
    installed = {**SAMPLE_LOCKED, "numpy": ("9.9.9",)}

    run = execute(
        tmp_path, runner, seams=make_seams(runner, installed=lambda: installed)
    )

    step = step_of(document_of(run), "environment")
    assert (step["state"], step["reason"]) == ("fail", fail_environment(1, 0))
    assert step["detail"]["mismatched"] == [  # type: ignore[index]
        {"name": "numpy", "installed": ["9.9.9"], "locked": ["2.5.4"]}
    ]
    assert rerun_ids(runner) == []
    assert item_states(run)["numerics"] == ("skipped", SKIP_ENVIRONMENT)
    assert states_of(run) == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "fail",
        "drift": "ok",
        "reruns": "skipped",
    }
    assert document_of(run)["state"] == "fail"


def test_a_package_the_lock_does_not_name_fails_the_step(tmp_path: Path) -> None:
    runner = uv_only_runner()
    installed = {**SAMPLE_LOCKED, "extra-one": ("1.0",), "extra-two": ("2.0",)}

    run = execute(
        tmp_path, runner, seams=make_seams(runner, installed=lambda: installed)
    )

    step = step_of(document_of(run), "environment")
    assert (step["state"], step["reason"]) == ("fail", fail_environment(0, 2))
    assert step["detail"]["extra"] == ["extra-one", "extra-two"]  # type: ignore[index]
    assert rerun_ids(runner) == []


def test_a_locked_package_that_is_not_installed_is_a_count_and_no_failure(
    tmp_path: Path,
) -> None:
    runner = golden_runner()
    installed = {k: v for k, v in SAMPLE_LOCKED.items() if k not in ("rich", "numpy")}

    run = execute(
        tmp_path, runner, seams=make_seams(runner, installed=lambda: installed)
    )

    step = step_of(document_of(run), "environment")
    assert step["state"] == "ok"
    assert step["detail"]["not_installed"] == 2  # type: ignore[index]
    assert step["detail"]["installed"] == 4  # type: ignore[index]
    assert "rich" not in str(step["detail"])


def test_twenty_five_mismatched_names_show_twenty_and_a_count_of_five(
    tmp_path: Path,
) -> None:
    names = [f"pkg{n:02d}" for n in range(25)]
    lock = lock_of(
        {**{n: ("1.0",) for n in names}, "unused-a": ("1",), "unused-b": ("1",)}
    )
    installed = {n: ("9.9",) for n in names} | {
        f"stray{n:02d}": ("1",) for n in range(25)
    }
    runner = uv_only_runner(dry_text=NO_CHANGE_MEASURED)

    run = execute(
        tmp_path,
        runner,
        inputs=make_inputs(lock=lock),
        seams=make_seams(runner, installed=lambda: installed),
    )

    step = step_of(document_of(run), "environment")
    detail = step["detail"]
    assert step["reason"] == fail_environment(25, 25)
    assert [m["name"] for m in detail["mismatched"]] == names[:20]  # type: ignore[index]
    assert detail["mismatched_more"] == 5  # type: ignore[index]
    assert detail["extra"] == [f"stray{n:02d}" for n in range(20)]  # type: ignore[index]
    assert detail["extra_more"] == 5  # type: ignore[index]
    assert detail["not_installed"] == 2  # type: ignore[index]
    assert detail["installed"] == 50  # type: ignore[index]
    assert "unused-a" not in str(detail)


def test_a_name_or_a_version_that_is_not_a_token_is_not_copied(tmp_path: Path) -> None:
    runner = uv_only_runner()
    installed = {
        **SAMPLE_LOCKED,
        "numpy": ("/home/ghost/numpy-9",),
        "/srv/odd": ("1.0",),
    }

    run = execute(
        tmp_path, runner, seams=make_seams(runner, installed=lambda: installed)
    )

    detail = step_of(document_of(run), "environment")["detail"]
    text = str(detail)
    assert "/home" not in text and "/srv" not in text
    assert detail["mismatched"][0]["name"] == "numpy"  # type: ignore[index]
    assert detail["mismatched"][0]["installed"] == ["(not a valid name or version)"]  # type: ignore[index]
    assert detail["extra"] == ["(not a valid name or version)"]  # type: ignore[index]


# ---- uv: the lock check and the dry run --------------------------------------------------------------


def test_no_uv_fails_the_lock_check_and_the_drift_and_the_reruns_still_run(
    tmp_path: Path,
) -> None:
    runner = script_passing_reruns(FakeRunner(), "numerics", "parity_tiny")

    run = execute(tmp_path, runner, seams=make_seams(runner, uv=None))

    assert [rerun_id_of(a) for a in runner.argvs] == ["numerics", "parity_tiny"]
    for step_id in ("lock_consistent", "drift"):
        step = step_of(document_of(run), step_id)
        assert (step["state"], step["reason"]) == ("fail", FAIL_UV_MISSING)
    assert states_of(run)["reruns"] == "ok"
    assert document_of(run)["state"] == "fail"
    assert run.result.logs == ("logs/numerics.log", "logs/parity_tiny.log")


def test_no_uv_with_no_drift_leaves_the_drift_skipped_and_the_lock_check_failed(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()

    run = execute(
        tmp_path,
        runner,
        options=make_options(drift=False, reruns=False),
        seams=make_seams(runner, uv=None),
    )

    assert runner.calls == []
    assert states_of(run)["drift"] == "skipped"
    assert step_of(document_of(run), "drift")["reason"] == SKIP_NO_DRIFT
    assert states_of(run)["lock_consistent"] == "fail"


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (CommandResult(None, False, False), FAIL_UV_MISSING),
        (CommandResult(None, True), fail_timeout(120)),
        (CommandResult(1, False), fail_uv_exit(1)),
        (CommandResult(2, False), fail_uv_exit(2)),
        (CommandResult(-9, False), fail_uv_exit(-9)),
        (CommandResult(None, False), FAIL_UV_MISSING),
    ],
)
def test_the_lock_check_fails_with_the_reason_of_how_uv_ended(
    tmp_path: Path, result: CommandResult, reason: str
) -> None:
    runner = script_passing_reruns(
        uv_only_runner(check=result), "numerics", "parity_tiny"
    )

    run = execute(tmp_path, runner)

    step = step_of(document_of(run), "lock_consistent")
    assert (step["state"], step["reason"], step["detail"]) == ("fail", reason, {})
    assert rerun_ids(runner) == ["numerics", "parity_tiny"]
    assert states_of(run)["drift"] == "ok"
    assert document_of(run)["state"] == "fail"


@pytest.mark.parametrize(
    ("result", "text", "reason"),
    [
        (CommandResult(None, False, False), "", FAIL_UV_MISSING),
        (CommandResult(None, True), "", fail_timeout(300)),
        (CommandResult(1, False), OFFLINE_FAILURE_MEASURED, fail_uv_exit(1)),
        (CommandResult(0, False), "", "uv printed no recognisable result"),
        (
            CommandResult(0, False),
            "Update numpy v2.5.4 -> v2.5.5\nsomething new\nanother\n",
            "the output of uv has 2 lines this version does not recognise",
        ),
        (
            CommandResult(0, False),
            "Update numpy v1.0 -> v2.0\n",
            "uv reported a locked version that differs from uv.lock for 1 packages",
        ),
    ],
)
def test_a_drift_failure_has_the_reason_of_the_dry_run_and_does_not_stop_the_reruns(
    tmp_path: Path, result: CommandResult, text: str, reason: str
) -> None:
    runner = script_passing_reruns(
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(result, log_text=text, when=uv_call("--upgrade", "--dry-run")),
        "numerics",
        "parity_tiny",
    )

    run = execute(tmp_path, runner)

    step = step_of(document_of(run), "drift")
    assert (step["state"], step["reason"], step["detail"]) == ("fail", reason, {})
    assert rerun_ids(runner) == ["numerics", "parity_tiny"]
    assert states_of(run)["reruns"] == "ok"
    assert document_of(run)["state"] == "fail"
    everything = (
        report_to_bytes(run.result.report) + render_markdown(run.result.report).encode()
    )
    for line in ("Using CPython", "Resolved", "error: No solution", "something new"):
        assert line.encode() not in everything


# ---- the reruns --------------------------------------------------------------------------------------


def test_a_failed_rerun_does_not_stop_the_next_and_fails_the_step(
    tmp_path: Path,
) -> None:
    runner = script_passing_reruns(uv_only_runner(), "parity_tiny")
    runner.script(
        CommandResult(1, False),
        junit=junit_xml(1110, 2, 0, 0),
        when=rerun_call("numerics"),
    )

    run = execute(tmp_path, runner)

    assert rerun_ids(runner) == ["numerics", "parity_tiny"]
    assert item_states(run)["numerics"] == ("fail", "2 failed and 0 errors")
    assert item_states(run)["parity_tiny"] == ("ok", None)
    step = step_of(document_of(run), "reruns")
    assert (step["state"], step["reason"]) == ("fail", None)
    assert states_of(run)["doctor"] == "ok"
    assert document_of(run)["state"] == "fail"
    assert exit_code_of(run.result.report, fail_on_drift=False) == 4


def test_a_rerun_below_its_floor_is_a_failure_with_its_counters(tmp_path: Path) -> None:
    runner = script_passing_reruns(uv_only_runner(), "parity_tiny")
    runner.script(
        CommandResult(0, False),
        junit=junit_xml(1000, 0, 0, 0),
        when=rerun_call("numerics"),
    )

    run = execute(tmp_path, runner)

    detail = step_of(document_of(run), "reruns")["detail"]
    numerics = detail["items"][0]  # type: ignore[index]
    assert (numerics["state"], numerics["reason"]) == ("fail", fail_floor(1000, 1100))
    assert numerics["tests"] == {"passed": 1000, "failed": 0, "errors": 0, "skipped": 0}


def test_a_failed_drift_and_a_drift_table_do_not_change_what_the_reruns_do(
    tmp_path: Path,
) -> None:
    run = execute(tmp_path, golden_runner())

    assert states_of(run)["drift"] == "drift"
    assert item_states(run)["numerics"] == ("ok", None)


def test_the_environment_gate_wins_over_a_missing_export_and_loses_to_no_reruns(
    tmp_path: Path,
) -> None:
    runner = uv_only_runner()
    run = execute(
        tmp_path,
        runner,
        options=make_options(selected=("parity_export",)),
        seams=make_seams(runner, doctor=failing_doctor("D-PY")),
    )
    assert item_states(run)["parity_export"] == ("skipped", SKIP_ENVIRONMENT)

    other = tmp_path / "other"
    other.mkdir()
    runner2 = uv_only_runner()
    run2 = execute(
        other,
        runner2,
        options=make_options(reruns=False),
        seams=make_seams(runner2, doctor=failing_doctor("D-PY")),
    )
    assert item_states(run2)["numerics"] == ("skipped", SKIP_NO_RERUNS)


def test_a_failed_doctor_with_a_selected_weekly_rerun_skips_it_with_the_environment_reason(
    tmp_path: Path,
) -> None:
    runner = uv_only_runner()

    run = execute(
        tmp_path,
        runner,
        options=make_options(selected=("serving_tiny",)),
        seams=make_seams(runner, doctor=failing_doctor("D-PY")),
    )

    states = item_states(run)
    assert states["serving_tiny"] == ("skipped", SKIP_ENVIRONMENT)
    assert states["numerics"] == ("skipped", skip_not_due("nightly"))
    reruns = step_of(document_of(run), "reruns")
    assert (reruns["state"], reruns["reason"]) == ("skipped", SKIP_ENVIRONMENT)


def test_with_no_nightly_rerun_in_the_file_the_step_is_skipped_as_not_due(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    weekly = [rerun for rerun in document["reruns"] if rerun["cadence"] == "weekly"]
    config = config_with(reruns=weekly, drift={**document["drift"], "rules": []})
    runner = uv_only_runner()

    run = execute(tmp_path, runner, inputs=make_inputs(config))

    step = step_of(document_of(run), "reruns")
    assert (step["state"], step["reason"]) == ("skipped", skip_not_due("nightly"))
    assert rerun_ids(runner) == []


def test_with_nothing_due_and_no_reruns_asked_for_the_step_says_it_was_disabled(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    weekly = [rerun for rerun in document["reruns"] if rerun["cadence"] == "weekly"]
    config = config_with(reruns=weekly, drift={**document["drift"], "rules": []})
    runner = uv_only_runner()

    run = execute(
        tmp_path, runner, inputs=make_inputs(config), options=make_options(reruns=False)
    )

    step = step_of(document_of(run), "reruns")
    assert (step["state"], step["reason"]) == ("skipped", SKIP_NO_RERUNS)
