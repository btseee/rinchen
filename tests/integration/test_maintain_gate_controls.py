"""The D14 STOP gate, the controls (R7 to R10 and the shim's own): the reruns that must fail, the drift outputs
that must be refused, the refusals and the interrupt of the real `rinchen maintain check` binary, and the hygiene
of the files D14 adds. The night itself (R1 to R6) is in `test_maintain_gate.py`, whose docstring says what
the toy is and is not and what the shim is.

Every run is the real binary on the CPU with every device hidden; the reruns are the real pytest over a scratch
project (the repository's own `pyproject.toml` and `uv.lock`, so that the real lock check and the real
environment check pass, and a handful of tiny tests whose outcomes are known). The `uv` is the shim of
`tests/support/fake_uv.py` and no output of `lock --upgrade --dry-run` ever comes from an index.

Controls, every floor a literal of the test that holds it and never lowered (the shim's own behaviour, which
these runs rely on, is pinned in `tests/unit/test_fake_uv.py`):

    R7  the reruns through the real pytest: a failing test is `fail` (`1 failed and 0 errors`), an erroring
        one too, a selection with fewer passing tests than its floor is `fail`, a skipping test with no skips
        allowed is `fail`, an empty target (pytest exit 5) is `fail` and never `ok`, a rerun that runs past
        its bound is `fail` (`timed out after 3 s`) within 10 s and the process it started is gone; a
        passing rerun sees the device hidden and the Hub offline; every other step is still reported and the
        exit is 4
    R8  the drift: random text, the measured offline failure, a sample whose old version disagrees with the
        lock are each `fail` and exit 4; the measured no-change text is `ok`; `--fail-on-drift` turns a table
        into exit 4 and without it the same table exits 0; `--no-drift` skips the step and the shim never sees
        a dry run; a configuration whose drift arguments lack `--dry-run` is refused (exit 2, nothing written)
    R9  the refusals (a missing `uv.lock` exit 3, another stage's manifest exit 3 with the old files untouched,
        `--export` alone, an unknown `--rerun` and an output directory that is the project's exit 2), each
        with nothing written; an interrupt (SIGINT exit 130, SIGTERM exit 143) leaves no manifest, takes the
        old one away and leaves no process
    R10 the hygiene: no file D14 adds asks a device question, none names `uv.lock` or `pyproject.toml` next to
        a write, no argument list holds an add, a sync, a pip or `--upgrade-package`, the one `--upgrade` of the
        code base is the shipped configuration's next to `--dry-run` (and the loader's closed list of flags),
        every subprocess has a timeout and the workflow holds none of the words of its scan

Measured on this machine (CPU, 2026-11-11):

    R7     seven reruns in one run (about 15 s with the doctor): `1 failed and 0 errors`, `0 failed and 1
           errors`, `2 passed, fewer than the floor 3`, `1 skipped, more than the allowed 0`, `0 passed,
           fewer than the floor 1` (pytest exit 5), the passing ones `ok` with 2 passed and with the device
           hidden; the 3 s bound: `timed out after 3 s`, the reruns step ends within 10 s of it and both pids
           of the sleeping test are gone
    R8     each drift run 13 s (the doctor): `lines this version does not recognise` for the random text,
           `uv exited with code 1`, `differs from uv.lock for 1 packages`; the no-change text `ok`
    R9     the five refusals under 1 s each; the interrupted runs end with 130 and 143 in 13.4 s
    R10    the scans of the 20 modules of the command and its library and the 7 helper files of the gate, no
           finding, and 22 sensitivity cases that each break one rule
    time   this module's tests took about 160 s in a run of their own (each real run of the command about
           13 s: the doctor)

UNMEASURED: GitHub Actions, the real state of PyPI, the lines uv prints for a change and anything on a GPU.
"""

from __future__ import annotations

import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.manifest.stage import output_file
from tests.support.distill_gate import Invocation
from tests.support.fake_uv import read_calls, write_uv_shim
from tests.support.maintain_gate import (
    REPOSITORY,
    WORKFLOW,
    absolute_paths,
    finish,
    gate_environment,
    project_files_unchanged,
    read_manifest,
    read_report,
    real_uv,
    rerun_item,
    run_maintain,
    scratch_project,
    start_maintain,
    step_of,
    stop_group,
    timestamps,
    workflow_command,
    write_config,
)
from tests.support.maintain_gate_hygiene import (
    LOADER,
    SHIPPED_CONFIG,
    changing_argument_lists,
    device_questions,
    helper_files,
    python_files_of_the_code_base,
    runtime_files,
    untimed_subprocesses,
    upgrade_flags_in_configurations,
    upgrade_flags_in_python,
    workflow_word_findings,
    writes_a_read_only_file,
)
from tests.support.maintain_project import (
    ERRORS,
    FAILS,
    HIDDEN_ENVIRONMENT_TEST,
    PASSES_A,
    PASSES_B,
    PIDS_FILE,
    SKIPS,
    SLEEPS,
    is_running,
    kill_quietly,
    read_pids,
    wait_until_gone,
)
from tests.support.nightly_workflow import CHECK_PREFIX
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    OFFLINE_FAILURE_MEASURED,
)

pytestmark = [pytest.mark.subprocess, pytest.mark.usefixtures("d11_gpu_hidden")]

EMPTY_TARGET = "tests/empty"
NO_SUCH_UV = "/nonexistent/uv"
SHIM_TIMEOUT_SECONDS = 30
RANDOM_TEXT = "zxqv plmk 3817\nwoeiru ytqp 20 jjj\n\nkk\n"
DISAGREEING_SAMPLE = CHANGES_HAND_WRITTEN.replace("numpy v2.5.4", "numpy v2.5.3")
SLEEP_BOUND_SECONDS = 3
SLEEP_GRACE_SECONDS = 10
INTERRUPT_WAIT_SECONDS = 150


@pytest.fixture(autouse=True)
def the_project_files_keep_their_bytes_and_times() -> Iterator[None]:
    """No test of this module may change `pyproject.toml` or `uv.lock` of the repository."""
    with project_files_unchanged():
        yield


# =================================================================================
# R7: the reruns through the real pytest.


def _rerun(
    rerun_id: str,
    targets: Sequence[str],
    *,
    min_passed: int = 1,
    timeout: int = 60,
) -> dict[str, Any]:
    return {
        "id": rerun_id,
        "kind": "pytest",
        "cadence": "nightly",
        "timeout_seconds": timeout,
        "why": "a control of the maintenance gate",
        "min_passed": min_passed,
        "targets": list(targets),
    }


@dataclass(frozen=True)
class ScratchRun:
    """A run of the real command over a scratch project: what it printed, where it wrote."""

    invocation: Invocation
    project: Path
    out_dir: Path
    shim_dir: Path

    @property
    def report(self) -> dict[str, Any]:
        return read_report(self.out_dir)


def _scratch_run(
    root: Path,
    reruns: Sequence[Mapping[str, Any]],
    *,
    extra: Sequence[str] = (),
    dry_run: str = NO_CHANGE_MEASURED,
) -> ScratchRun:
    project = scratch_project(root)
    config = write_config(root / "config" / "maintain.yaml", reruns)
    shim = write_uv_shim(
        root / "bin", real_uv=real_uv(), project=REPOSITORY, dry_run_output=dry_run
    )
    out_dir = root / "out"
    invocation = run_maintain(
        [
            "--out-dir",
            out_dir,
            "--project-dir",
            project,
            "--config",
            config,
            "--uv",
            shim,
            *extra,
        ],
        cwd=root,
        env=gate_environment(tmpdir=root / "tmp"),
    )
    return ScratchRun(invocation, project, out_dir, shim.parent)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> ScratchRun:
    """One run with a rerun for each outcome of pytest that matters, all at the nightly cadence."""
    reruns = [
        _rerun("passes", [PASSES_A, PASSES_B], min_passed=2),
        _rerun("fails", [FAILS]),
        _rerun("errors", [ERRORS]),
        _rerun("below_floor", [PASSES_A, PASSES_B], min_passed=3),
        _rerun("skips", [PASSES_A, SKIPS]),
        _rerun("empty", [EMPTY_TARGET]),
        _rerun("hidden", [HIDDEN_ENVIRONMENT_TEST]),
    ]
    return _scratch_run(tmp_path_factory.mktemp("r7_outcomes"), reruns)


def _outcome(run: ScratchRun, rerun_id: str) -> tuple[str, str | None]:
    item = rerun_item(run.report, rerun_id)
    return item["state"], item["reason"]


def test_a_failing_test_fails_its_rerun_and_the_exit_is_four(
    outcomes: ScratchRun,
) -> None:
    assert outcomes.invocation.returncode == 4, outcomes.invocation.stderr
    assert _outcome(outcomes, "fails") == ("fail", "1 failed and 0 errors")
    assert outcomes.report["state"] == "fail"


def test_an_erroring_test_fails_its_rerun(outcomes: ScratchRun) -> None:
    assert _outcome(outcomes, "errors") == ("fail", "0 failed and 1 errors")


def test_a_selection_with_fewer_passing_tests_than_its_floor_fails(
    outcomes: ScratchRun,
) -> None:
    assert _outcome(outcomes, "below_floor") == (
        "fail",
        "2 passed, fewer than the floor 3",
    )


def test_a_skipping_test_fails_a_rerun_that_allows_no_skips(
    outcomes: ScratchRun,
) -> None:
    assert _outcome(outcomes, "skips") == ("fail", "1 skipped, more than the allowed 0")


def test_an_empty_target_is_a_failure_and_never_a_pass(outcomes: ScratchRun) -> None:
    # pytest exits 5 here and the JUnit report counts no test: the floor names the failure.
    assert _outcome(outcomes, "empty") == ("fail", "0 passed, fewer than the floor 1")


def test_a_passing_rerun_passes_and_sees_the_device_hidden_and_the_hub_offline(
    outcomes: ScratchRun,
) -> None:
    passes = rerun_item(outcomes.report, "passes")

    assert _outcome(outcomes, "passes") == ("ok", None)
    assert passes["tests"] == {"passed": 2, "failed": 0, "errors": 0, "skipped": 0}
    assert _outcome(outcomes, "hidden") == ("ok", None)


def test_a_failed_rerun_stops_nothing_else_and_everything_is_written(
    outcomes: ScratchRun,
) -> None:
    report = outcomes.report
    states = {step["id"]: step["state"] for step in report["steps"]}

    assert [step["id"] for step in report["steps"]] == [
        "doctor",
        "lock_file",
        "lock_consistent",
        "environment",
        "drift",
        "reruns",
    ]
    assert states == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "ok",
        "reruns": "fail",
    }
    assert (outcomes.out_dir / "manifest.json").is_file()
    assert (outcomes.out_dir / "maintain_report.md").is_file()
    assert [item["id"] for item in step_of(report, "reruns")["detail"]["items"]] == [
        "passes",
        "fails",
        "errors",
        "below_floor",
        "skips",
        "empty",
        "hidden",
    ]


@pytest.fixture(scope="module")
def sleeper(tmp_path_factory: pytest.TempPathFactory) -> ScratchRun:
    return _scratch_run(
        tmp_path_factory.mktemp("r7_sleeper"),
        [_rerun("sleeper", [SLEEPS], timeout=SLEEP_BOUND_SECONDS)],
        extra=("--no-drift",),
    )


def test_a_rerun_that_runs_past_its_bound_fails_within_ten_seconds_of_it(
    sleeper: ScratchRun,
) -> None:
    seconds = read_manifest(sleeper.out_dir)["metrics"]["seconds.reruns"]

    assert sleeper.invocation.returncode == 4, sleeper.invocation.stderr
    assert _outcome(sleeper, "sleeper") == ("fail", "timed out after 3 s")
    assert SLEEP_BOUND_SECONDS <= seconds < SLEEP_BOUND_SECONDS + SLEEP_GRACE_SECONDS


def test_the_process_a_timed_out_rerun_started_and_its_child_are_gone(
    sleeper: ScratchRun,
) -> None:
    pids = read_pids(sleeper.project / PIDS_FILE)

    try:
        assert len(pids) == 2
        assert all(wait_until_gone(pid) for pid in pids)
    finally:
        kill_quietly([pid for pid in pids if is_running(pid)])


# =================================================================================
# R8: the drift.


@dataclass(frozen=True)
class DriftRun:
    invocation: Invocation
    out_dir: Path
    shim_dir: Path

    @property
    def drift(self) -> dict[str, Any]:
        return step_of(read_report(self.out_dir), "drift")


def _drift_run(
    tmp_path: Path,
    output: str,
    *,
    exit_code: int = 0,
    extra: Sequence[str] = (),
    config: Path | None = None,
) -> DriftRun:
    shim = write_uv_shim(
        tmp_path / "bin",
        real_uv=real_uv(),
        project=REPOSITORY,
        dry_run_output=output,
        dry_run_exit=exit_code,
    )
    out_dir = tmp_path / "out"
    arguments: list[str | Path] = ["--out-dir", out_dir, "--no-reruns", "--uv", shim]
    if config is not None:
        arguments += ["--config", config]
    invocation = run_maintain([*arguments, *extra], cwd=tmp_path)
    return DriftRun(invocation, out_dir, shim.parent)


def test_an_output_of_random_text_is_a_failure_and_never_a_table(
    tmp_path: Path,
) -> None:
    run = _drift_run(tmp_path, RANDOM_TEXT)

    assert run.invocation.returncode == 4, run.invocation.stderr
    assert run.drift["state"] == "fail"
    assert re.fullmatch(
        r"the output of uv has [1-9][0-9]* lines this version does not recognise",
        run.drift["reason"],
    )
    assert run.drift["detail"] == {}
    assert read_report(run.out_dir)["state"] == "fail"
    assert (run.out_dir / "manifest.json").is_file()


def test_the_measured_offline_failure_with_its_exit_code_is_a_failure(
    tmp_path: Path,
) -> None:
    run = _drift_run(tmp_path, OFFLINE_FAILURE_MEASURED, exit_code=1)

    assert run.invocation.returncode == 4
    assert (run.drift["state"], run.drift["reason"]) == (
        "fail",
        "uv exited with code 1",
    )


def test_a_sample_whose_old_version_disagrees_with_the_lock_is_a_failure(
    tmp_path: Path,
) -> None:
    run = _drift_run(tmp_path, DISAGREEING_SAMPLE)

    assert run.invocation.returncode == 4
    assert (run.drift["state"], run.drift["reason"]) == (
        "fail",
        "uv reported a locked version that differs from uv.lock for 1 packages",
    )


def test_the_measured_no_change_output_is_ok_and_the_run_is_ok(tmp_path: Path) -> None:
    run = _drift_run(tmp_path, NO_CHANGE_MEASURED)
    report = read_report(run.out_dir)

    assert run.invocation.returncode == 0, run.invocation.stderr
    assert (run.drift["state"], run.drift["reason"]) == ("ok", None)
    assert report["state"] == "ok"
    assert run.invocation.stdout[0] == "maintain: ok (nightly)"


def test_fail_on_drift_turns_a_table_into_exit_four_and_without_it_the_exit_is_zero(
    tmp_path: Path,
) -> None:
    strict = _drift_run(
        tmp_path / "strict", CHANGES_HAND_WRITTEN, extra=["--fail-on-drift"]
    )
    plain = _drift_run(tmp_path / "plain", CHANGES_HAND_WRITTEN)

    assert strict.invocation.returncode == 4
    assert plain.invocation.returncode == 0, plain.invocation.stderr
    for run in (strict, plain):
        assert run.drift["state"] == "drift"
        assert read_report(run.out_dir)["state"] == "drift"


def test_no_drift_skips_the_step_and_the_shim_never_sees_a_dry_run(
    tmp_path: Path,
) -> None:
    run = _drift_run(tmp_path, CHANGES_HAND_WRITTEN, extra=["--no-drift"])
    calls = read_calls(run.shim_dir)

    assert run.invocation.returncode == 0, run.invocation.stderr
    assert (run.drift["state"], run.drift["reason"]) == (
        "skipped",
        "disabled by --no-drift",
    )
    assert not any("--dry-run" in call["argv"] for call in calls)
    assert [call["argv"] for call in calls if call["forwarded"]] == [
        ["lock", "--check"],
        ["--version"],
    ]


def test_a_configuration_whose_drift_arguments_lack_the_dry_run_is_refused(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text("utf-8"))
    document["drift"]["uv_args"] = ["lock", "--upgrade"]
    config = tmp_path / "no_dry_run.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    run = _drift_run(tmp_path, CHANGES_HAND_WRITTEN, config=config)

    assert run.invocation.returncode == 2
    assert "--dry-run" in run.invocation.stderr
    assert run.invocation.stdout == []
    assert not run.out_dir.exists()
    assert read_calls(run.shim_dir) == []


# =================================================================================
# R9: the refusals and the interrupt.


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _write_old_run(out_dir: Path, stage: str) -> None:
    """Leave a manifest of `stage` and one file it lists in `out_dir`."""
    out_dir.mkdir(parents=True)
    old_report = out_dir / "maintain_report.json"
    old_report.write_text('{"old": true}\n', encoding="utf-8")
    store.write(
        out_dir,
        Manifest(
            stage=stage,
            created="2026-01-01T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=-1,
            outputs=[output_file(old_report, None)],
        ),
    )


def _refuse_missing_lock(root: Path, project: Path, out_dir: Path) -> list[str | Path]:
    (project / "uv.lock").unlink()
    return ["--project-dir", project]


def _refuse_foreign_manifest(
    root: Path, project: Path, out_dir: Path
) -> list[str | Path]:
    _write_old_run(out_dir, "export_parity")
    return ["--project-dir", project]


def _refuse_half_an_export(
    root: Path, project: Path, out_dir: Path
) -> list[str | Path]:
    return ["--project-dir", project, "--export", root / "an_export"]


def _refuse_unknown_rerun(root: Path, project: Path, out_dir: Path) -> list[str | Path]:
    return ["--project-dir", project, "--rerun", "nosuch"]


def _refuse_out_dir_of_the_project(
    root: Path, project: Path, out_dir: Path
) -> list[str | Path]:
    return ["--project-dir", project, "--out-dir", project]


REFUSALS: list[tuple[Callable[[Path, Path, Path], list[str | Path]], int, str]] = [
    (_refuse_missing_lock, 3, "uv.lock"),
    (_refuse_foreign_manifest, 3, "export_parity"),
    (_refuse_half_an_export, 2, "--export, --checkpoint and --corpus"),
    (_refuse_unknown_rerun, 2, "nosuch"),
    (_refuse_out_dir_of_the_project, 2, "--out-dir"),
]


@pytest.mark.parametrize(
    ("prepare", "exit_code", "message"),
    REFUSALS,
    ids=[refusal[0].__name__.removeprefix("_refuse_") for refusal in REFUSALS],
)
def test_the_real_binary_refuses_and_writes_nothing(
    tmp_path: Path,
    prepare: Callable[[Path, Path, Path], list[str | Path]],
    exit_code: int,
    message: str,
) -> None:
    project = scratch_project(tmp_path)
    out_dir = tmp_path / "out"
    arguments = prepare(tmp_path, project, out_dir)
    if "--out-dir" not in arguments:
        arguments = [*arguments, "--out-dir", out_dir]
    before = _tree(tmp_path)

    # `--rerun` contradicts `--no-reruns`, so the case that names a rerun leaves the latter out
    # (an unknown rerun is refused before anything runs).
    no_reruns = [] if "--rerun" in arguments else ["--no-reruns"]
    run = run_maintain(
        [*arguments, "--no-drift", *no_reruns, "--uv", NO_SUCH_UV], cwd=tmp_path
    )

    assert run.returncode == exit_code, run.stderr
    assert message in run.stderr
    assert run.stdout == []
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(
    ("signal_number", "exit_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
    ids=["sigint", "sigterm"],
)
def _wait_for_pids(pids_file: Path, process: subprocess.Popen[bytes]) -> list[int]:
    """The two pids the sleeping test writes, once it has written them (bounded)."""
    deadline = time.monotonic() + INTERRUPT_WAIT_SECONDS
    while True:
        assert process.poll() is None, "the command ended before its rerun started"
        assert time.monotonic() < deadline, "the rerun did not start in time"
        if pids_file.exists() and len(pids_file.read_text().split()) == 2:
            return read_pids(pids_file)
        time.sleep(0.2)


@pytest.mark.parametrize(
    ("signal_number", "exit_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
    ids=["sigint", "sigterm"],
)
def test_an_interrupted_run_leaves_no_manifest_and_no_process_and_takes_the_old_run_away(
    tmp_path: Path, signal_number: signal.Signals, exit_code: int
) -> None:
    project = scratch_project(tmp_path)
    config = write_config(
        tmp_path / "config" / "maintain.yaml",
        [_rerun("sleeper", [SLEEPS], timeout=120)],
    )
    shim = write_uv_shim(
        tmp_path / "bin",
        real_uv=real_uv(),
        project=REPOSITORY,
        dry_run_output=NO_CHANGE_MEASURED,
    )
    out_dir = tmp_path / "out"
    _write_old_run(out_dir, "maintain_check")
    process = start_maintain(
        [
            "--out-dir",
            out_dir,
            "--project-dir",
            project,
            "--config",
            config,
            "--no-drift",
            "--uv",
            shim,
        ],
        cwd=tmp_path,
        env=gate_environment(tmpdir=tmp_path / "tmp"),
    )
    pids: list[int] = []
    try:
        pids = _wait_for_pids(project / PIDS_FILE, process)
        process.send_signal(signal_number)
        run = finish(process, timeout=60)
        assert run.returncode == exit_code, run.stderr
        assert not (out_dir / "manifest.json").exists()
        assert not (out_dir / "maintain_report.json").exists()
        assert all(wait_until_gone(pid) for pid in pids)
    finally:
        if process.poll() is None:
            stop_group(process)
        kill_quietly([pid for pid in pids if is_running(pid)])


# =================================================================================
# The gate's own guard: a command starts only where every device is hidden.


def test_a_command_is_refused_unless_the_device_is_hidden_for_the_process_and_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `--help` is the whole command: a guard that failed would start a command that does nothing.
    shown = {**gate_environment(), "CUDA_VISIBLE_DEVICES": "0"}

    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        run_maintain(["--help"], cwd=tmp_path, env=shown)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        run_maintain(["--help"], cwd=tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "UV_OFFLINE",
        "HF_TOKEN",
    ],
)
def test_the_variables_that_hide_the_devices_and_the_hub_cannot_be_changed(
    name: str,
) -> None:
    with pytest.raises(AssertionError, match=name):
        gate_environment(extra={name: "x"})


def test_the_environment_of_a_command_hides_the_device_and_the_hub_and_holds_no_token() -> (
    None
):
    environment = gate_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == environment["UV_OFFLINE"] == "1"
    assert "HF_TOKEN" not in environment
    assert not [name for name in environment if name.startswith("PYTEST_")]


def test_the_command_is_read_from_the_workflow_for_either_cadence() -> None:
    nightly = workflow_command(WORKFLOW, "nightly")

    assert nightly == workflow_command(WORKFLOW, "weekly")
    assert nightly.startswith(CHECK_PREFIX)
    assert '--cadence "$CADENCE"' in nightly


def test_a_workflow_whose_cadence_expression_gives_another_cadence_is_refused(
    tmp_path: Path,
) -> None:
    text = WORKFLOW.read_text("utf-8")
    broken = tmp_path / "nightly.yml"
    broken.write_text(
        text.replace("github.event.inputs.cadence ||", "'weekly' ||"), encoding="utf-8"
    )

    assert broken.read_text("utf-8") != text
    with pytest.raises(AssertionError, match="CADENCE expression"):
        workflow_command(broken, "nightly")


@pytest.mark.parametrize(
    "text",
    [
        "/home/someone/project",
        "see (/tmp/a/b.txt) now",
        "path: /var/lib/x",
        "report: /srv/out/maintain_report.json",
        "C:" + chr(0x5C) + "Users",
    ],
)
def test_the_path_scan_sees_an_absolute_path_wherever_it_stands(text: str) -> None:
    assert absolute_paths(text) != []


@pytest.mark.parametrize(
    "text",
    [
        "docs/runbooks/14_maintenance.md#failure-drift",
        "n/a",
        "1/2",
        "a/b and/or c",
        "| - | ok |",
    ],
)
def test_the_path_scan_leaves_relative_paths_and_slashes_alone(text: str) -> None:
    assert absolute_paths(text) == []


@pytest.mark.parametrize(
    "text", ["2026-11-11", "created 2026-11-11T13:51:56", "at 13:51:56 today"]
)
def test_the_timestamp_scan_sees_a_date_and_a_time_of_day(text: str) -> None:
    assert timestamps(text) != []


@pytest.mark.parametrize(
    "text", ["version 0.13.7", "3.14.10", "numpy 2.5.4", "1100 passed"]
)
def test_the_timestamp_scan_leaves_versions_and_counts_alone(text: str) -> None:
    assert timestamps(text) == []


def _project_of_two_files(root: Path) -> Path:
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def test_the_guard_on_the_project_files_lets_an_untouched_pair_pass(
    tmp_path: Path,
) -> None:
    root = _project_of_two_files(tmp_path / "project")

    with project_files_unchanged(root):
        pass


@pytest.mark.parametrize("name", ["pyproject.toml", "uv.lock"])
def test_the_guard_on_the_project_files_sees_a_changed_byte_or_a_touch(
    tmp_path: Path, name: str
) -> None:
    root = _project_of_two_files(tmp_path / "project")

    with pytest.raises(AssertionError, match=name), project_files_unchanged(root):
        (root / name).write_bytes((root / name).read_bytes() + b"\n")
    with pytest.raises(AssertionError, match=name), project_files_unchanged(root):
        (root / name).touch()


# =================================================================================
# R10: the hygiene of the files D14 adds.

RUNTIME = runtime_files()
FILES = [*RUNTIME, *helper_files()]
BROKEN_SOURCES = {
    "device": [
        "import torch",
        "from torch import cuda",
        "x = torch" + ".cuda.is_available()",
        "n = get_cuda_device_count()",
        "x = lib.get_cuda_device_count",
        "subprocess.run(['nvidia" + "-smi'])",
        "doctor.check_torch_cuda()",
        "from rinchen.cli.doctor import check_torch_cuda",
        "CHECKS = (doctor.check_py, doctor.check_torch_cuda)",
    ],
    "write": [
        "def f():\n    open('uv.lock', 'w')",
        "def f(p):\n    (p / 'pyproject.toml').write_text('x')",
        "def f(p):\n    (p / 'uv.lock').unlink()",
        "def f(p):\n    p.replace(p / 'uv.lock')",
        "replace_file(project / 'uv.lock', b'')",
    ],
    "change": [
        "run(['uv', 'sync'])",
        "run(('uv', 'pip', 'install'))",
        "run(['uv', 'lock', '--upgrade-package', 'x'])",
        "run(['uv', 'add', 'x'])",
    ],
    "untimed": [
        "import subprocess\nsubprocess.run(['x'])",
        "import subprocess\nsubprocess.check_output(['x'])",
        "import subprocess\nsubprocess.Popen(['x'])",
        "import subprocess\np = subprocess.Popen(['x'])\np.communicate()",
    ],
}


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_no_file_that_runs_or_builds_a_command_asks_a_device_question(
    path: Path,
) -> None:
    assert device_questions(path.read_text("utf-8")) == []


@pytest.mark.parametrize("source", BROKEN_SOURCES["device"])
def test_the_device_scan_sees_each_question_a_command_could_ask(source: str) -> None:
    assert device_questions(source) != []


@pytest.mark.parametrize("path", RUNTIME, ids=lambda path: path.name)
def test_no_module_of_the_command_writes_next_to_the_lock_or_the_project_file(
    path: Path,
) -> None:
    assert writes_a_read_only_file(path.read_text("utf-8")) == []


@pytest.mark.parametrize("source", BROKEN_SOURCES["write"])
def test_the_write_scan_sees_a_scope_that_names_a_read_only_file_and_writes(
    source: str,
) -> None:
    assert writes_a_read_only_file(source) != []


@pytest.mark.parametrize("path", RUNTIME, ids=lambda path: path.name)
def test_no_argument_list_of_the_command_holds_a_change_word(path: Path) -> None:
    assert changing_argument_lists(path.read_text("utf-8")) == []


@pytest.mark.parametrize("source", BROKEN_SOURCES["change"])
def test_the_argument_scan_sees_each_word_that_could_change_the_stack(
    source: str,
) -> None:
    assert changing_argument_lists(source) != []


def test_the_only_upgrade_flag_of_the_code_base_is_the_configurations_next_to_the_dry_run() -> (
    None
):
    in_python = {
        path.relative_to(REPOSITORY).as_posix(): upgrade_flags_in_python(
            path.read_text("utf-8")
        )
        for path in python_files_of_the_code_base()
    }

    assert {name: found for name, found in in_python.items() if found} == {
        LOADER.relative_to(REPOSITORY).as_posix(): ["--upgrade"]
    }
    assert upgrade_flags_in_configurations() == [
        ("configs/maintain/maintain.yaml", ["lock", "--upgrade", "--dry-run"])
    ]


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_every_subprocess_of_the_command_and_the_gate_has_a_timeout(path: Path) -> None:
    assert untimed_subprocesses(path.read_text("utf-8")) == []


@pytest.mark.parametrize("source", BROKEN_SOURCES["untimed"])
def test_the_timeout_scan_sees_a_subprocess_that_is_not_bounded(source: str) -> None:
    assert untimed_subprocesses(source) != []


def test_the_workflow_holds_none_of_the_words_of_its_scan() -> None:
    text = WORKFLOW.read_text("utf-8")

    assert workflow_word_findings(text) == []
    for broken in (
        text + "\n# a docker image\n",
        text.replace("just tiny", "uv lock --upgrade"),
        text.replace("just tiny", "git push"),
        text + "\n# ${{ secrets.TOKEN }}\n",
    ):
        assert workflow_word_findings(broken) != []
