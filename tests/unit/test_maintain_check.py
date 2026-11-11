"""`run_check` (D14, spec M2, M4 to M9 and M11): the steps of a maintenance check, their order and
the reruns' selection, against the scripted runner and the doubles of `tests/support/maintain_check.py`.

The golden scenario (the nightly run of the shipped configuration over the sample lock) is pinned
here by value: the six steps, the doctor's seven checks, the seven rows of the drift table with
their pin rules, the six reruns. The calls the runner sees are pinned in order (the lock check, the
dry run, then the reruns in file order) together with the order in which the seams are used, and
the selection of the reruns case by case: the cadence, `selected`, `--no-reruns`, `--no-drift` and
the export. The failure cases are in `test_maintain_check_failures.py`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

from rinchen.errors import ConfigError
from rinchen.maintain.check import CheckResult, DoctorFinding, Inputs, run_check
from rinchen.maintain.commands import HIDDEN_ENVIRONMENT, UV_ENVIRONMENT, CommandResult
from rinchen.maintain.lock import LockFile
from rinchen.maintain.report import (
    STEP_IDS,
    exit_code_of,
    report_document,
    report_to_bytes,
)
from rinchen.maintain.reruns import ExportInputs, parity_argv
from rinchen.maintain.states import (
    SKIP_GPU_CHECK,
    SKIP_NO_DRIFT,
    SKIP_NO_EXPORT,
    SKIP_NO_RERUNS,
    SKIP_NOT_SELECTED,
    skip_not_due,
)
from tests.support.maintain_check import (
    CPU_CHECKS,
    GPU_CHECK,
    PYTHON,
    SECRET_ENVIRONMENT,
    UV,
    Run,
    execute,
    finding,
    golden_doctor,
    golden_run,
    golden_runner,
    make_inputs,
    make_options,
    make_seams,
    rerun_call,
    rerun_id_of,
    script_passing_reruns,
    shipped_config,
    step_of,
    uv_call,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.maintain_project import make_project
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    SAMPLE_LOCKED,
    lock_for_samples,
)

pytestmark = pytest.mark.usefixtures("no_network")

RERUN_IDS = (
    "numerics",
    "parity_tiny",
    "parity_export",
    "robustness_tiny",
    "evaluation_tiny",
    "serving_tiny",
)
PARITY_JSON = {"exact": 200, "sentences": 200, "mean_abs_delta": 7.19e-07}
# What a child inherits of SECRET_ENVIRONMENT: its two secrets (a name that ends in `_SECRET` and
# one that ends in `_TOKEN`) are left out of every child's environment.
INHERITED_ENVIRONMENT = {"HOME": SECRET_ENVIRONMENT["HOME"]}
LOCK_CHECK = (UV, "lock", "--check")
DRY_RUN = (UV, "lock", "--upgrade", "--dry-run")


def document_of(run: Run) -> dict[str, object]:
    return report_document(run.result.report)


def items_of(run: Run) -> dict[str, dict[str, object]]:
    detail = step_of(document_of(run), "reruns")["detail"]
    assert isinstance(detail, dict)
    return {item["id"]: item for item in detail["items"]}


def states_of(run: Run) -> dict[str, str]:
    return {step.id: step.state for step in run.result.report.steps}


# ---- the golden scenario, by value ----------------------------------------------------------------


def test_the_golden_run_lists_the_six_steps_in_order_with_their_states(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    document = document_of(run)

    assert [step["id"] for step in document["steps"]] == list(STEP_IDS)  # type: ignore[attr-defined]
    assert states_of(run) == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "drift",
        "reruns": "ok",
    }
    assert document["state"] == "drift"
    assert document["cadence"] == "nightly"
    assert exit_code_of(run.result.report, fail_on_drift=False) == 0
    assert exit_code_of(run.result.report, fail_on_drift=True) == 4
    assert run.runner.unused() == 0


def test_the_report_carries_the_digests_and_the_figures_of_the_lock(
    tmp_path: Path,
) -> None:
    document = document_of(golden_run(tmp_path))

    assert set(document["inputs"]) == {"config", "lock", "pyproject"}  # type: ignore[call-overload]
    assert document["inputs"] == {  # same digests as the Inputs were given
        "config": make_inputs().config_sha256,
        "lock": make_inputs().lock_sha256,
        "pyproject": make_inputs().pyproject_sha256,
    }
    assert document["lock"] == {
        "packages": 6,
        "names": 6,
        "format": {"version": 1, "revision": 5},
    }


def test_the_doctor_lists_seven_checks_and_the_gpu_one_is_the_configured_skip(
    tmp_path: Path,
) -> None:
    doctor = golden_doctor()
    run = execute(
        tmp_path, golden_runner(), seams=make_seams(golden_runner(), doctor=doctor)
    )
    step = step_of(document_of(run), "doctor")

    assert doctor.calls == 1
    assert (step["state"], step["reason"]) == ("ok", None)
    checks = step["detail"]["checks"]  # type: ignore[index]
    assert [check["id"] for check in checks] == [*CPU_CHECKS, GPU_CHECK]
    assert [(c["state"], c["reason"]) for c in checks[:6]] == [("ok", None)] * 6
    assert checks[6] == {"id": GPU_CHECK, "state": "skipped", "reason": SKIP_GPU_CHECK}
    assert set(checks[0]) == {"id", "state", "reason"}


def test_the_lock_steps_and_the_environment_are_ok_with_their_figures(
    tmp_path: Path,
) -> None:
    document = document_of(golden_run(tmp_path))

    for step_id in ("lock_file", "lock_consistent"):
        step = step_of(document, step_id)
        assert (step["state"], step["reason"], step["detail"]) == ("ok", None, {})
    environment = step_of(document, "environment")
    assert (environment["state"], environment["reason"]) == ("ok", None)
    assert environment["detail"] == {
        "installed": 6,
        "mismatched": [],
        "mismatched_more": 0,
        "extra": [],
        "extra_more": 0,
        "not_installed": 0,
    }


def test_the_drift_step_holds_the_seven_entries_of_the_hand_written_run(
    tmp_path: Path,
) -> None:
    step = step_of(document_of(golden_run(tmp_path)), "drift")

    assert (step["state"], step["reason"]) == ("drift", None)
    detail = step["detail"]
    assert detail["counts"] == {  # type: ignore[index]
        "update": 5,
        "add": 1,
        "remove": 1,
        "drift": 3,
        "blocked": 4,
    }
    assert detail["warnings"] == 0  # type: ignore[index]
    assert detail["entries"] == [  # type: ignore[index]
        {
            "package": "huggingface-hub",
            "change": "update",
            "locked": "2.3.0",
            "available": "2.3.1",
            "class": "blocked",
            "rule": "hub-client",
            "requires": ["evaluation_tiny", "gate"],
        },
        {
            "package": "numpy",
            "change": "update",
            "locked": "2.5.4",
            "available": "2.5.5",
            "class": "blocked",
            "rule": "sacrebleu-numpy",
            "requires": ["numerics", "evaluation_tiny"],
        },
        {
            "package": "nvidia-cudnn-cu13",
            "change": "update",
            "locked": "9.24.0.43",
            "available": "9.25.0.10",
            "class": "blocked",
            "rule": "torch-family",
            "requires": ["parity_tiny", "robustness_tiny", "gate"],
        },
        {
            "package": "rich",
            "change": "update",
            "locked": "15.0.0",
            "available": "15.0.1",
            "class": "drift",
            "rule": None,
            "requires": [],
        },
        {
            "package": "safetensors",
            "change": "update",
            "locked": "0.8.0",
            "available": "0.8.1",
            "class": "blocked",
            "rule": "transformers-stack",
            "requires": ["gate"],
        },
        {
            "package": "sortedcontainers",
            "change": "remove",
            "locked": "2.4.0",
            "available": None,
            "class": "drift",
            "rule": None,
            "requires": [],
        },
        {
            "package": "tomli-w",
            "change": "add",
            "locked": None,
            "available": "1.2.0",
            "class": "drift",
            "rule": None,
            "requires": [],
        },
    ]


def test_the_reruns_are_the_six_of_the_file_two_run_one_without_an_export_three_not_due(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    step = step_of(document_of(run), "reruns")

    assert (step["state"], step["reason"]) == ("ok", None)
    items = step["detail"]["items"]  # type: ignore[index]
    assert [item["id"] for item in items] == list(RERUN_IDS)
    assert items[0] == {
        "id": "numerics",
        "kind": "pytest",
        "cadence": "nightly",
        "state": "ok",
        "reason": None,
        "tests": {"passed": 1110, "failed": 0, "errors": 0, "skipped": 0},
        "figures": None,
    }
    assert items[1]["tests"] == {"passed": 13, "failed": 0, "errors": 0, "skipped": 0}
    assert (items[2]["state"], items[2]["reason"], items[2]["kind"]) == (
        "skipped",
        SKIP_NO_EXPORT,
        "export_parity",
    )
    assert (items[2]["tests"], items[2]["figures"]) == (None, None)
    for item in items[3:]:
        assert (item["state"], item["reason"], item["cadence"]) == (
            "skipped",
            skip_not_due("nightly"),
            "weekly",
        )
        assert (item["tests"], item["figures"]) == (None, None)


def test_the_seconds_hold_one_measurement_for_each_step(tmp_path: Path) -> None:
    result = golden_run(tmp_path).result

    assert dict(result.seconds) == {step_id: 0.25 for step_id in STEP_IDS}


def test_the_logs_are_the_relative_names_of_the_files_the_commands_wrote(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    assert run.result.logs == (
        "logs/lock_consistent.log",
        "logs/drift.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
    )
    assert all((run.out / name).is_file() for name in run.result.logs)


def test_the_check_writes_nothing_of_its_own_and_leaves_the_project_as_it_was(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path, config=shipped_config())
    before = {p: p.read_bytes() for p in sorted(project.rglob("*")) if p.is_file()}
    out = tmp_path / "out"

    result = run_check(
        make_inputs(),
        make_options(),
        make_seams(golden_runner()),
        project_dir=project,
        out_dir=out,
    )

    assert {
        p: p.read_bytes() for p in sorted(project.rglob("*")) if p.is_file()
    } == before
    written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert written == {*result.logs, "junit/numerics.xml", "junit/parity_tiny.xml"}


# ---- what the commands are given -------------------------------------------------------------------


def test_the_calls_are_the_lock_check_the_dry_run_then_the_reruns_in_file_order(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    argvs = run.runner.argvs
    assert argvs[0] == LOCK_CHECK
    assert argvs[1] == DRY_RUN
    assert [rerun_id_of(argv) for argv in argvs[2:]] == ["numerics", "parity_tiny"]
    assert len(argvs) == 4


def test_the_seams_are_used_in_the_order_of_the_steps(tmp_path: Path) -> None:
    runner = golden_runner()
    doctor = golden_doctor()
    seen: list[tuple[str, int]] = []

    def doctor_seam() -> Sequence[DoctorFinding]:
        seen.append(("doctor", len(runner.calls)))
        return doctor()

    def installed() -> Mapping[str, Sequence[str]]:
        seen.append(("environment", len(runner.calls)))
        return SAMPLE_LOCKED

    execute(
        tmp_path,
        runner,
        seams=make_seams(runner, doctor=doctor_seam, installed=installed),
    )

    # The doctor before every command; the environment after the lock check and before the
    # dry run (calls 0 and 1 of the runner).
    assert seen == [("doctor", 0), ("environment", 1)]
    assert runner.argvs[:2] == [LOCK_CHECK, DRY_RUN]


def test_each_command_runs_in_the_project_with_its_bound_and_its_environment(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    lock_check, dry_run, numerics, parity = run.runner.calls

    for call in run.runner.calls:
        assert call.cwd == run.project
        assert call.env["CUDA_VISIBLE_DEVICES"] == ""
        assert call.env["HOME"] == SECRET_ENVIRONMENT["HOME"]
    assert (lock_check.timeout, dry_run.timeout) == (120, 300)
    assert (numerics.timeout, parity.timeout) == (300, 600)
    assert lock_check.env == {
        **INHERITED_ENVIRONMENT,
        **HIDDEN_ENVIRONMENT,
        **UV_ENVIRONMENT,
    }
    assert dry_run.env == lock_check.env
    assert numerics.env == {**INHERITED_ENVIRONMENT, **HIDDEN_ENVIRONMENT}
    assert lock_check.log == run.out / "logs" / "lock_consistent.log"
    assert dry_run.log == run.out / "logs" / "drift.log"
    assert numerics.log == run.out / "logs" / "numerics.log"
    assert numerics.argv[0] == PYTHON


def test_the_parent_environment_is_not_modified(tmp_path: Path) -> None:
    parent = dict(SECRET_ENVIRONMENT)

    execute(
        tmp_path, golden_runner(), seams=make_seams(golden_runner(), parent_env=parent)
    )

    assert parent == dict(SECRET_ENVIRONMENT)


# ---- the selection of the reruns --------------------------------------------------------------------


def weekly_runner() -> FakeRunner:
    runner = FakeRunner().script(CommandResult(0, False), when=uv_call("--check"))
    runner.script(
        CommandResult(0, False),
        log_text=NO_CHANGE_MEASURED,
        when=uv_call("--upgrade", "--dry-run"),
    )
    return script_passing_reruns(
        runner,
        "numerics",
        "parity_tiny",
        "robustness_tiny",
        "evaluation_tiny",
        "serving_tiny",
    )


def test_a_weekly_run_runs_the_five_runnable_reruns_and_lists_none_as_not_due(
    tmp_path: Path,
) -> None:
    runner = weekly_runner()

    run = execute(tmp_path, runner, options=make_options(cadence="weekly"))

    assert [rerun_id_of(argv) for argv in runner.argvs[2:]] == [
        "numerics",
        "parity_tiny",
        "robustness_tiny",
        "evaluation_tiny",
        "serving_tiny",
    ]
    items = items_of(run)
    assert {i: (items[i]["state"], items[i]["reason"]) for i in RERUN_IDS} == {
        "numerics": ("ok", None),
        "parity_tiny": ("ok", None),
        "parity_export": ("skipped", SKIP_NO_EXPORT),
        "robustness_tiny": ("ok", None),
        "evaluation_tiny": ("ok", None),
        "serving_tiny": ("ok", None),
    }
    assert document_of(run)["cadence"] == "weekly"
    assert states_of(run)["drift"] == "ok"


def test_selected_runs_exactly_those_whatever_the_cadence(tmp_path: Path) -> None:
    runner = script_passing_reruns(
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        ),
        "serving_tiny",
    )

    run = execute(tmp_path, runner, options=make_options(selected=("serving_tiny",)))

    assert [rerun_id_of(argv) for argv in runner.argvs[2:]] == ["serving_tiny"]
    items = items_of(run)
    assert items["serving_tiny"]["state"] == "ok"
    for rerun_id in RERUN_IDS:
        if rerun_id != "serving_tiny":
            assert (items[rerun_id]["state"], items[rerun_id]["reason"]) == (
                "skipped",
                SKIP_NOT_SELECTED,
            )


def test_selecting_the_export_rerun_without_an_export_runs_nothing_and_says_why(
    tmp_path: Path,
) -> None:
    runner = (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )

    run = execute(tmp_path, runner, options=make_options(selected=("parity_export",)))

    assert runner.argvs == [LOCK_CHECK, DRY_RUN]
    items = items_of(run)
    assert (items["parity_export"]["state"], items["parity_export"]["reason"]) == (
        "skipped",
        SKIP_NO_EXPORT,
    )
    assert items["numerics"]["reason"] == SKIP_NOT_SELECTED
    step = step_of(document_of(run), "reruns")
    assert (step["state"], step["reason"]) == ("skipped", SKIP_NO_EXPORT)


def test_the_export_rerun_runs_the_parity_command_over_the_given_export(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    export = ExportInputs(
        tmp_path / "export", tmp_path / "ckpt", (tmp_path / "a.parquet",)
    )
    runner = (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
        .script(
            CommandResult(0, False),
            files={
                out / "parity_export" / "parity.json": json.dumps(PARITY_JSON).encode()
            },
            when=rerun_call("parity_export"),
        )
    )

    run = execute(
        tmp_path,
        runner,
        options=make_options(selected=("parity_export",), export=export),
    )

    rerun = shipped_config().reruns[2]
    assert runner.argvs[2] == tuple(
        parity_argv(PYTHON, rerun, export, out / "parity_export")
    )
    call = runner.calls[2]
    assert call.cwd == run.project
    assert call.timeout == 600
    assert call.env == {**INHERITED_ENVIRONMENT, **HIDDEN_ENVIRONMENT}
    assert call.log == out / "logs" / "parity_export.log"
    item = items_of(run)["parity_export"]
    assert (item["state"], item["reason"]) == ("ok", None)
    assert item["figures"] == PARITY_JSON
    assert item["tests"] is None
    assert run.result.logs[-1] == "logs/parity_export.log"


def test_the_export_rerun_runs_at_the_nightly_cadence_when_an_export_is_given(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    export = ExportInputs(
        tmp_path / "export", tmp_path / "ckpt", (tmp_path / "a.parquet",)
    )
    runner = golden_runner().script(
        CommandResult(0, False),
        files={out / "parity_export" / "parity.json": json.dumps(PARITY_JSON).encode()},
        when=rerun_call("parity_export"),
    )

    run = execute(tmp_path, runner, options=make_options(export=export))

    assert [rerun_id_of(argv) for argv in runner.argvs[2:]] == [
        "numerics",
        "parity_tiny",
        "parity_export",
    ]
    assert items_of(run)["parity_export"]["state"] == "ok"


def test_no_drift_runs_only_the_lock_check_of_the_uv_calls(tmp_path: Path) -> None:
    runner = script_passing_reruns(
        FakeRunner().script(CommandResult(0, False), when=uv_call("--check")),
        "numerics",
        "parity_tiny",
    )

    run = execute(tmp_path, runner, options=make_options(drift=False))

    assert [a for a in runner.argvs if a[0] == UV] == [LOCK_CHECK]
    step = step_of(document_of(run), "drift")
    assert (step["state"], step["reason"], step["detail"]) == (
        "skipped",
        SKIP_NO_DRIFT,
        {},
    )
    assert states_of(run)["reruns"] == "ok"
    assert document_of(run)["state"] == "ok"
    assert run.result.logs == (
        "logs/lock_consistent.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
    )


def test_no_reruns_lists_all_six_as_skipped_and_runs_none(tmp_path: Path) -> None:
    runner = (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )

    run = execute(tmp_path, runner, options=make_options(reruns=False))

    assert runner.argvs == [LOCK_CHECK, DRY_RUN]
    items = items_of(run)
    assert list(items) == list(RERUN_IDS)
    assert all(
        (item["state"], item["reason"], item["tests"], item["figures"])
        == ("skipped", SKIP_NO_RERUNS, None, None)
        for item in items.values()
    )
    step = step_of(document_of(run), "reruns")
    assert (step["state"], step["reason"]) == ("skipped", SKIP_NO_RERUNS)


def test_no_reruns_wins_over_selected(tmp_path: Path) -> None:
    runner = (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )

    run = execute(
        tmp_path, runner, options=make_options(reruns=False, selected=("numerics",))
    )

    assert len(runner.argvs) == 2
    assert items_of(run)["numerics"]["reason"] == SKIP_NO_RERUNS


def test_selected_wins_over_the_weekly_cadence_too(tmp_path: Path) -> None:
    runner = script_passing_reruns(
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        ),
        "numerics",
    )

    run = execute(
        tmp_path, runner, options=make_options(cadence="weekly", selected=("numerics",))
    )

    assert [rerun_id_of(argv) for argv in runner.argvs[2:]] == ["numerics"]
    assert items_of(run)["serving_tiny"]["reason"] == SKIP_NOT_SELECTED


# ---- refusals of the arguments ---------------------------------------------------------------------


@pytest.mark.parametrize("selected", [("nope",), ("numerics", "nope"), ("",)])
def test_a_selection_that_names_no_rerun_of_the_file_is_refused_before_anything_runs(
    tmp_path: Path, selected: tuple[str, ...]
) -> None:
    runner = FakeRunner()
    doctor = golden_doctor()

    with pytest.raises(ConfigError, match="names no rerun") as raised:
        execute(
            tmp_path,
            runner,
            options=make_options(selected=selected),
            seams=make_seams(runner, doctor=doctor),
        )

    assert runner.calls == [] and doctor.calls == 0
    assert "nope" not in str(raised.value)


@pytest.mark.parametrize("cadence", ["daily", "", "Nightly", "weekly "])
def test_a_cadence_that_is_none_of_the_two_is_refused_even_with_a_selection(
    tmp_path: Path, cadence: str
) -> None:
    runner = FakeRunner()

    with pytest.raises(ConfigError, match="cadence"):
        execute(
            tmp_path,
            runner,
            options=make_options(cadence=cadence, selected=("numerics",)),
        )

    assert runner.calls == []


@pytest.mark.parametrize("field", ["config_sha256", "lock_sha256", "pyproject_sha256"])
@pytest.mark.parametrize(
    "bad", ["", "abc", "A" * 64, "g" * 64, "a" * 63, "/tmp/" + "a" * 59]
)
def test_an_input_digest_that_is_not_a_sha256_is_refused(field: str, bad: str) -> None:
    good = make_inputs()
    values = {
        "config_sha256": good.config_sha256,
        "lock_sha256": good.lock_sha256,
        "pyproject_sha256": good.pyproject_sha256,
    }
    values[field] = bad

    with pytest.raises(ValueError, match=field):
        Inputs(config=good.config, lock=lock_for_samples(), **values)


def test_a_finding_needs_a_doctor_check_id_and_one_of_the_three_statuses() -> None:
    assert finding("D-PY").status == "pass"
    for bad_id in ("", "PY", "d-py", "D-" + "X" * 31, "D-PY\n", "/srv/x"):
        with pytest.raises(ValueError, match="check_id"):
            finding(bad_id)
    with pytest.raises(ValueError, match="status"):
        finding("D-PY", "passed")  # type: ignore[arg-type]


def test_the_result_holds_a_report_seconds_messages_and_logs(tmp_path: Path) -> None:
    result: CheckResult = golden_run(tmp_path).result

    assert result.report.cadence == "nightly"
    assert isinstance(result.seconds, Mapping)
    assert result.doctor_messages == ()
    assert isinstance(result.logs, tuple)


# ---- the lock's figures, the logs and the warnings ---------------------------------------------------


def test_the_lock_figures_count_entries_and_names_apart_and_keep_a_missing_revision(
    tmp_path: Path,
) -> None:
    versions = {"alpha": ("1.0", "2.0"), "beta": ("3.0",)}
    lock = LockFile(
        format_version=1, revision=None, entries=7, versions=MappingProxyType(versions)
    )
    runner = FakeRunner().script(CommandResult(0, False), when=uv_call("--check"))
    runner.script(
        CommandResult(0, False),
        log_text=NO_CHANGE_MEASURED,
        when=uv_call("--upgrade", "--dry-run"),
    )

    run = execute(
        tmp_path,
        runner,
        options=make_options(reruns=False),
        inputs=make_inputs(lock=lock),
        seams=make_seams(runner, installed=lambda: versions),
    )

    assert document_of(run)["lock"] == {
        "packages": 7,
        "names": 2,
        "format": {"version": 1, "revision": None},
    }
    assert states_of(run)["environment"] == "ok"


def test_a_relative_output_directory_gives_the_same_relative_log_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project = make_project(tmp_path, config=shipped_config())

    result = run_check(
        make_inputs(),
        make_options(),
        make_seams(golden_runner()),
        project_dir=project,
        out_dir=Path("out"),
    )

    assert result.logs == (
        "logs/lock_consistent.log",
        "logs/drift.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
    )
    assert all((tmp_path / "out" / name).is_file() for name in result.logs)


class SilentRunner:
    """A runner whose commands succeed and leave no log."""

    def run(self, argv: Sequence[str], **kwargs: object) -> CommandResult:
        return CommandResult(0, False)


def test_a_command_that_left_no_log_is_not_among_the_logs(tmp_path: Path) -> None:
    project = make_project(tmp_path, config=shipped_config())
    seams = dataclasses.replace(make_seams(FakeRunner()), runner=SilentRunner())

    result = run_check(
        make_inputs(),
        make_options(reruns=False),
        seams,
        project_dir=project,
        out_dir=tmp_path / "out",
    )

    assert result.logs == ()
    assert result.report.steps[2].state == "ok"


@pytest.mark.parametrize(
    ("text", "state", "warnings", "entries"),
    [
        (NO_CHANGE_MEASURED + "warning: a thing\nwarning:\n", "ok", 2, 0),
        (CHANGES_HAND_WRITTEN + "warning: odd\n", "drift", 1, 7),
    ],
)
def test_the_warning_lines_of_a_dry_run_are_counted_and_never_quoted(
    tmp_path: Path, text: str, state: str, warnings: int, entries: int
) -> None:
    runner = script_passing_reruns(
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=text,
            when=uv_call("--upgrade", "--dry-run"),
        ),
        "numerics",
        "parity_tiny",
    )

    run = execute(tmp_path, runner)

    step = step_of(document_of(run), "drift")
    assert step["state"] == state
    assert step["detail"]["warnings"] == warnings  # type: ignore[index]
    assert len(step["detail"]["entries"]) == entries  # type: ignore[index]
    assert "a thing" not in report_to_bytes(run.result.report).decode()
