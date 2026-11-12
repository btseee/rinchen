"""`rinchen maintain check` (D14, spec M3, M9 and M12): a full run and every option, through typer's
runner over a project on disk, with the runner, the doctor and the installed versions replaced by
doubles (`tests.support.maintain_cli`). Nothing starts a process; the fake runner records every
command the run would have made.

The nightly run of the shipped configuration over the sample lock is the golden case: its JSON is
the bytes the library makes from the same observations (the check is run a second time directly,
with no command around it), its manifest is pinned by value, and its printed lines are listed in
full. The options are checked one at a time against the commands the runner sees. The refusals are
in `test_cli_maintain_refusals.py`, the failure paths and the commit protocol in
`test_cli_maintain_failures.py`.
"""

from __future__ import annotations

import json
import platform
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.cli import maintain_inputs
from rinchen.config.loader import REPO_ROOT
from rinchen.maintain.check import run_check
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.lock import read_lock
from rinchen.maintain.report import report_to_bytes
from rinchen.maintain.report_markdown import render_markdown
from rinchen.maintain.reruns import ExportInputs, parity_argv
from rinchen.maintain.states import SKIP_NO_DRIFT, SKIP_NO_EXPORT, SKIP_NO_RERUNS
from tests.support.maintain_check import (
    SHIPPED_CONFIG,
    golden_doctor,
    golden_runner,
    make_inputs,
    make_options,
    make_seams,
    script_passing_reruns,
    shipped_config,
    step_of,
)
from tests.support.maintain_cli import (
    CODE_VERSION,
    LOCKED,
    UV,
    UV_VERSION,
    Workspace,
    cli_runner,
    install,
    invoke,
    lines_of,
    make_workspace,
    manifest_of,
    parity_script,
    quiet_runner,
    sha256_of,
    version_call,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.maintain_scans import violations
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

STEPS = ("doctor", "lock_file", "lock_consistent", "environment", "drift", "reruns")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def report_of_run(ws: Workspace) -> dict[str, Any]:
    document = json.loads((ws.out / "maintain_report.json").read_bytes())
    assert isinstance(document, dict)
    return document


def uv_arguments(runner: FakeRunner) -> list[list[str]]:
    """The arguments (after the program) of every command that went to uv, in order."""
    return [list(argv[1:]) for argv in runner.argvs if argv[0] == UV]


# ---- a full run ---------------------------------------------------------------------------------------


def test_the_nightly_run_writes_every_file_and_the_manifest_last(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    order: list[str] = []
    real = Path.replace

    def spy(self: Path, target: Any) -> Any:
        order.append(Path(target).name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", spy)

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert sorted(ws.written()) == [
        "junit/numerics.xml",
        "junit/parity_tiny.xml",
        "logs/drift.log",
        "logs/lock_consistent.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
        "maintain_report.json",
        "maintain_report.md",
        "manifest.json",
    ]
    assert order == ["maintain_report.json", "maintain_report.md", "manifest.json"]


def test_the_report_is_the_bytes_the_library_makes_from_the_same_observations(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    inputs = make_inputs(
        shipped_config(), read_lock((ws.project / "uv.lock").read_bytes())
    )
    inputs = type(inputs)(
        config=inputs.config,
        lock=inputs.lock,
        config_sha256=sha256_of(ws.config),
        lock_sha256=sha256_of(ws.project / "uv.lock"),
        pyproject_sha256=sha256_of(ws.project / "pyproject.toml"),
    )
    expected = run_check(
        inputs,
        make_options(),
        make_seams(cli_runner(), doctor=golden_doctor(), installed=lambda: LOCKED),
        project_dir=ws.project,
        out_dir=tmp_path / "library-out",
    )

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert (ws.out / "maintain_report.json").read_bytes() == report_to_bytes(
        expected.report
    )
    assert (ws.out / "maintain_report.md").read_text(
        encoding="utf-8"
    ) == render_markdown(expected.report)


def test_the_manifest_is_pinned_by_value(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)

    assert invoke(ws.args()).exit_code == 0

    manifest = manifest_of(ws.out)
    assert manifest["stage"] == "maintain_check"
    assert manifest["seed"] == -1
    assert manifest["code_version"] == CODE_VERSION
    assert manifest["config_sha256"] == sha256_of(ws.config)
    assert manifest["inputs"] == [
        {
            "path": str(path),
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "rows": None,
        }
        for path in (
            ws.project / "pyproject.toml",
            ws.project / "uv.lock",
            ws.config,
        )
    ]
    outputs = {
        Path(item["path"]).relative_to(ws.out).as_posix(): item
        for item in manifest["outputs"]
    }
    assert list(outputs) == [
        "maintain_report.json",
        "maintain_report.md",
        "logs/lock_consistent.log",
        "logs/drift.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
        "junit/numerics.xml",
        "junit/parity_tiny.xml",
    ]
    for name, item in outputs.items():
        assert item["sha256"] == sha256_of(ws.out / name)
        assert item["bytes"] == (ws.out / name).stat().st_size
    assert manifest["tool_versions"] == {
        "numpy": "2.5.4",
        "rich": "15.0.0",
        "huggingface-hub": "2.4.0",
        "safetensors": "0.8.0",
        "nvidia-cudnn-cu13": "9.24.0.43",
        "sortedcontainers": "2.4.0",
        "twin": "1.0, 2.0",
        "python": platform.python_version(),
        "uv": UV_VERSION,
    }
    assert {
        key: value for key, value in manifest["metrics"].items() if "." not in key
    } == {
        "packages": 8.0,
        "entries": 7.0,
        "blocked": 4.0,
        "reruns_run": 2.0,
    }
    assert sorted(key for key in manifest["metrics"] if key.startswith("seconds.")) == [
        f"seconds.{step}" for step in sorted(STEPS)
    ]
    assert manifest["status"] == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "drift",
        "reruns": "ok",
        "state": "drift",
    }


def test_the_printed_lines_are_the_interface_in_order(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)

    result = invoke(ws.args())

    assert lines_of(result) == [
        "maintain: drift (nightly)",
        "doctor: ok (6 ok, 0 fail, 1 skipped)",
        "lock: ok (8 entries, 7 names)",
        "environment: ok (7 installed, 0 differ, 0 extra, 0 not installed)",
        "drift: drift (7 entries: 5 update, 1 add, 1 remove; 4 blocked)",
        "rerun numerics: ok (1110 passed)",
        "rerun parity_tiny: ok (13 passed)",
        "rerun parity_export: skipped (no export was given)",
        "rerun robustness_tiny: skipped (not due at the nightly cadence)",
        "rerun evaluation_tiny: skipped (not due at the nightly cadence)",
        "rerun serving_tiny: skipped (not due at the nightly cadence)",
        f"report: {ws.out / 'maintain_report.json'}",
        f"summary: {ws.out / 'maintain_report.md'}",
        f"manifest: {ws.out / 'manifest.json'}",
    ]


def test_the_commands_of_the_run_are_the_lock_check_the_dry_run_the_reruns_and_the_version(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    assert invoke(ws.args()).exit_code == 0

    assert uv_arguments(runner) == [
        ["lock", "--check"],
        ["lock", "--upgrade", "--dry-run"],
        ["--version"],
    ]
    pytest_calls = [argv for argv in runner.argvs if argv[0] != UV]
    assert [argv[0] for argv in pytest_calls] == [sys.executable, sys.executable]
    assert all(call.cwd == ws.project for call in runner.calls)


def test_the_version_log_is_not_left_in_the_output_directory(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    assert invoke(ws.args()).exit_code == 0

    version_log = next(call.log for call in runner.calls if version_call(call.argv))
    assert version_log.parent == ws.out / "logs"
    assert not version_log.exists()


def test_the_version_of_uv_in_the_manifest_is_unknown_when_uv_fails(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = golden_runner().script(
        CommandResult(2, False), log_text="uv 0.13.7 (broken)\n", when=version_call
    )
    install(monkeypatch, runner=runner)

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert manifest_of(ws.out)["tool_versions"]["uv"] == "unknown"


def test_the_version_of_uv_in_the_manifest_is_unknown_when_the_output_is_not_a_version(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = golden_runner().script(
        CommandResult(0, False), log_text="uv /home/someone/bin\n", when=version_call
    )
    install(monkeypatch, runner=runner)

    assert invoke(ws.args()).exit_code == 0

    assert manifest_of(ws.out)["tool_versions"]["uv"] == "unknown"


def test_each_input_is_hashed_before_it_is_read(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    names = {"pyproject.toml", "uv.lock", ws.config.name}
    events: list[tuple[str, str]] = []
    real_hash = maintain_inputs.hash_input
    real_read = Path.read_bytes

    def hashing(path: Path, **options: Any) -> Any:
        events.append(("hash", path.name))
        return real_hash(path, **options)

    def reading(self: Path) -> bytes:
        if self.name in names:
            events.append(("read", self.name))
        return real_read(self)

    monkeypatch.setattr(maintain_inputs, "hash_input", hashing)
    monkeypatch.setattr(Path, "read_bytes", reading)

    assert invoke(ws.args()).exit_code == 0

    for name in names:
        assert ("hash", name) in events and ("read", name) in events
        assert events.index(("hash", name)) < events.index(("read", name))
    assert [kind for kind, _ in events[:3]] == ["hash"] * 3


# ---- the options --------------------------------------------------------------------------------------


def test_no_drift_skips_the_drift_step_and_runs_no_uv_call_but_the_lock_check(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    result = invoke(ws.args("--no-drift"))

    assert result.exit_code == 0, result.output
    assert uv_arguments(runner) == [["lock", "--check"], ["--version"]]
    drift = step_of(report_of_run(ws), "drift")
    assert (drift["state"], drift["reason"], drift["detail"]) == (
        "skipped",
        SKIP_NO_DRIFT,
        {},
    )
    assert "drift: skipped (disabled by --no-drift)" in lines_of(result)
    assert lines_of(result)[0] == "maintain: ok (nightly)"


def test_no_reruns_skips_every_rerun_and_starts_no_pytest(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    result = invoke(ws.args("--no-reruns"))

    assert result.exit_code == 0, result.output
    assert [argv for argv in runner.argvs if argv[0] != UV] == []
    reruns = step_of(report_of_run(ws), "reruns")
    assert (reruns["state"], reruns["reason"]) == ("skipped", SKIP_NO_RERUNS)
    assert {item["reason"] for item in reruns["detail"]["items"]} == {SKIP_NO_RERUNS}
    assert sorted(ws.written()) == [
        "logs/drift.log",
        "logs/lock_consistent.log",
        "maintain_report.json",
        "maintain_report.md",
        "manifest.json",
    ]
    assert manifest_of(ws.out)["metrics"]["reruns_run"] == 0.0


def test_a_named_rerun_without_an_export_is_skipped_with_its_reason(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    result = invoke(ws.args("--rerun", "parity_export"))

    assert result.exit_code == 0, result.output
    items = {
        item["id"]: item
        for item in step_of(report_of_run(ws), "reruns")["detail"]["items"]
    }
    assert items["parity_export"]["state"] == "skipped"
    assert items["parity_export"]["reason"] == SKIP_NO_EXPORT
    assert items["numerics"]["state"] == "skipped"
    assert [argv for argv in runner.argvs if argv[0] != UV] == []


def test_a_named_rerun_runs_whatever_the_cadence(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = script_passing_reruns(cli_runner(), "serving_tiny")
    install(monkeypatch, runner=runner)

    result = invoke(ws.args("--rerun", "serving_tiny", "--no-drift"))

    assert result.exit_code == 0, result.output
    items = {
        item["id"]: item["state"]
        for item in step_of(report_of_run(ws), "reruns")["detail"]["items"]
    }
    assert items["serving_tiny"] == "ok"
    assert items["numerics"] == "skipped"


def test_an_export_runs_the_parity_command_with_every_corpus_and_lists_its_result(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = parity_script(cli_runner(), ws.out)
    install(monkeypatch, runner=runner)
    export, checkpoint = tmp_path / "export", tmp_path / "checkpoint"
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"

    result = invoke(
        ws.args(
            "--export",
            str(export),
            "--checkpoint",
            str(checkpoint),
            "--corpus",
            str(first),
            "--corpus",
            str(second),
        )
    )

    assert result.exit_code == 0, result.output
    rerun = next(r for r in shipped_config().reruns if r.id == "parity_export")
    expected = parity_argv(
        sys.executable,
        rerun,
        ExportInputs(export, checkpoint, (first, second)),
        ws.out / "parity_export",
    )
    assert expected in [list(argv) for argv in runner.argvs]
    assert "rerun parity_export: ok (200/200 exact, mean |delta| 7.19e-07)" in lines_of(
        result
    )
    listed = [
        Path(item["path"]).relative_to(ws.out).as_posix()
        for item in manifest_of(ws.out)["outputs"]
    ]
    assert "parity_export/parity.json" in listed
    assert "logs/parity_export.log" in listed


def test_a_weekly_cadence_runs_every_rerun_that_needs_no_export(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = script_passing_reruns(
        cli_runner(), "robustness_tiny", "evaluation_tiny", "serving_tiny"
    )
    install(monkeypatch, runner=runner)

    result = invoke(ws.args("--cadence", "weekly"))

    assert result.exit_code == 0, result.output
    document = report_of_run(ws)
    assert document["cadence"] == "weekly"
    states = {
        item["id"]: item["state"]
        for item in step_of(document, "reruns")["detail"]["items"]
    }
    assert states == {
        "numerics": "ok",
        "parity_tiny": "ok",
        "parity_export": "skipped",
        "robustness_tiny": "ok",
        "evaluation_tiny": "ok",
        "serving_tiny": "ok",
    }
    assert lines_of(result)[0] == "maintain: drift (weekly)"


def test_fail_on_drift_turns_a_drift_run_into_exit_4_with_the_same_files(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    assert invoke(ws.args()).exit_code == 0
    plain = (ws.out / "maintain_report.json").read_bytes()
    plain_markdown = (ws.out / "maintain_report.md").read_bytes()
    install(monkeypatch)

    result = invoke(ws.args("--fail-on-drift", out_dir=tmp_path / "strict"))

    assert result.exit_code == 4, result.output
    assert (tmp_path / "strict" / "maintain_report.json").read_bytes() == plain
    assert (tmp_path / "strict" / "maintain_report.md").read_bytes() == plain_markdown
    assert (tmp_path / "strict" / "manifest.json").is_file()
    assert lines_of(result)[0] == "maintain: drift (nightly)"


def test_fail_on_drift_does_not_change_a_run_without_drift(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, runner=quiet_runner())

    result = invoke(ws.args("--fail-on-drift"))

    assert result.exit_code == 0, result.output
    assert lines_of(result)[0] == "maintain: ok (nightly)"
    assert "drift: ok (no changes)" in lines_of(result)


# ---- the program uv -----------------------------------------------------------------------------------


def test_the_uv_option_is_the_program_of_every_uv_command(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)

    assert invoke(ws.args(uv="/x/uv")).exit_code == 0

    assert {argv[0] for argv in runner.argvs if argv[0] != sys.executable} == {"/x/uv"}


def test_a_relative_uv_option_is_made_absolute_because_the_child_runs_elsewhere(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = install(monkeypatch)
    monkeypatch.chdir(tmp_path)

    assert invoke(ws.args(uv="bin/uv")).exit_code == 0

    assert str(tmp_path / "bin" / "uv") in {argv[0] for argv in runner.argvs}


def make_uv_on_path(directory: Path) -> Path:
    directory.mkdir()
    program = directory / "uv"
    program.write_text("#!/bin/sh\n")
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    return program


def test_without_the_option_uv_is_the_one_on_the_path(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = install(monkeypatch)
    program = make_uv_on_path(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(program.parent))

    result = invoke(ws.args(uv=None))

    assert result.exit_code == 0, result.output
    assert {argv[0] for argv in runner.argvs if argv[0] != sys.executable} == {
        str(program)
    }


def test_without_a_uv_both_uv_steps_fail_and_nothing_is_run_for_them(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = install(monkeypatch)
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    result = invoke(ws.args(uv=None))

    assert result.exit_code == 4, result.output
    document = report_of_run(ws)
    for step in ("lock_consistent", "drift"):
        found = step_of(document, step)
        assert (found["state"], found["reason"]) == ("fail", "uv was not found")
    assert [argv for argv in runner.argvs if argv[0] != sys.executable] == []
    assert manifest_of(ws.out)["tool_versions"]["uv"] == "unknown"
    assert manifest_of(ws.out)["status"]["state"] == "fail"


def test_the_environment_variable_uv_is_never_read(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = install(monkeypatch)
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("UV", "/opt/from-the-environment/uv")

    result = invoke(ws.args(uv=None))

    assert result.exit_code == 4, result.output
    assert not any("from-the-environment" in " ".join(argv) for argv in runner.argvs)
    assert step_of(report_of_run(ws), "drift")["reason"] == "uv was not found"


def test_the_environment_variable_uv_does_not_override_the_option_or_the_path(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)
    monkeypatch.setenv("UV", "/opt/from-the-environment/uv")

    assert invoke(ws.args(uv="/x/uv")).exit_code == 0

    assert not any("from-the-environment" in argv[0] for argv in runner.argvs)


def test_the_children_run_with_the_environment_of_the_command_and_the_hidden_one(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch)
    monkeypatch.setenv("RINCHEN_CHILD_PROBE", "visible")

    assert invoke(ws.args()).exit_code == 0

    for call in runner.calls:
        assert call.env["RINCHEN_CHILD_PROBE"] == "visible"
        assert call.env["CUDA_VISIBLE_DEVICES"] == ""
        assert call.env["HF_HUB_OFFLINE"] == "1"
    uv_calls = [call for call in runner.calls if call.argv[0] == UV]
    assert all(call.env["NO_COLOR"] == "1" for call in uv_calls)


# ---- the configuration is the file that was given -----------------------------------------------------


def test_the_default_configuration_is_the_one_in_the_repository(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    arguments = [
        "maintain",
        "check",
        "--out-dir",
        str(ws.out),
        "--project-dir",
        str(ws.project),
        "--uv",
        UV,
    ]

    result = invoke(arguments)

    assert result.exit_code == 0, result.output
    shipped = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
    assert manifest_of(ws.out)["config_sha256"] == sha256_of(shipped)


def test_a_configuration_with_other_reruns_changes_what_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["reruns"] = [
        rerun for rerun in document["reruns"] if rerun["id"] == "numerics"
    ]
    document["drift"]["rules"] = []
    ws = make_workspace(tmp_path, config_text=yaml.safe_dump(document))
    install(monkeypatch)

    result = invoke(ws.args("--no-drift"))

    assert result.exit_code == 0, result.output
    items = step_of(report_of_run(ws), "reruns")["detail"]["items"]
    assert [item["id"] for item in items] == ["numerics"]


def test_the_environment_of_the_command_is_in_no_file(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    monkeypatch.setenv("RINCHEN_TEST_SECRET", "swordfish-0451")

    assert invoke(ws.args()).exit_code == 0

    written = ws.written()
    for name in ("maintain_report.json", "maintain_report.md"):
        assert violations(written[name].decode(), secrets=["swordfish-0451"]) == []
    assert all(b"swordfish-0451" not in data for data in written.values())
