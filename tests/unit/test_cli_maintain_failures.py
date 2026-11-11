"""The failure paths, the commit protocol and the untouched inputs of `rinchen maintain check`
(D14, spec M3, M4 and M12).

A measured failure (a rerun that fails, a doctor check that fails) is a result: everything is
written and printed and the exit code is 4. A doctor message is printed to standard output as
`[FAIL] <id>: <message>` and is in no file. A seam that raises is never turned into a pass. The
commit protocol is D13's: the old manifest goes before anything else, then the files it listed (and
no other), the documents are written whole, and the manifest comes last, so that an interrupt or a
failed write leaves no manifest. Two runs over the same inputs give byte-equal documents and a
manifest that differs only in `created` and the seconds. Whatever happens, `pyproject.toml`, the
lock and the configuration keep their bytes and their modification times.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import maintain_output
from rinchen.maintain.check_types import DoctorFinding
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.states import SKIP_ENVIRONMENT
from tests.support.maintain_check import (
    DOCTOR_MESSAGE,
    Doctor,
    finding,
    golden_doctor,
    script_passing_reruns,
    step_of,
)
from tests.support.maintain_cli import (
    LOCKED,
    Workspace,
    cli_runner,
    identity,
    install,
    invoke,
    lines_of,
    make_workspace,
    manifest_of,
    night_runner,
    snapshot,
)
from tests.support.maintain_fakes import FakeRunner, junit_xml
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def report_of_run(ws: Workspace) -> dict[str, Any]:
    document = json.loads((ws.out / "maintain_report.json").read_bytes())
    assert isinstance(document, dict)
    return document


class Interrupting:
    """A runner that raises `error` when the command that matches `when` is run and passes every
    other command on to `inner`."""

    def __init__(
        self,
        error: BaseException,
        when: Callable[[Sequence[str]], bool],
        inner: FakeRunner,
    ) -> None:
        self.error, self.when, self.inner = error, when, inner

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        if self.when(argv):
            raise self.error
        return self.inner.run(argv, cwd=cwd, env=env, log=log, timeout=timeout)


def is_dry_run(argv: Sequence[str]) -> bool:
    return "--upgrade" in argv


# ---- a measured failure is a result ---------------------------------------------------------------------


def test_a_failing_rerun_writes_everything_prints_it_and_exits_4(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = night_runner(numerics=(1, junit_xml(1110, 2, 0, 0)))
    install(monkeypatch, runner=runner)

    result = invoke(ws.args())

    assert result.exit_code == 4, result.output
    assert "manifest.json" in ws.written()
    numerics = next(
        item
        for item in step_of(report_of_run(ws), "reruns")["detail"]["items"]
        if item["id"] == "numerics"
    )
    assert (numerics["state"], numerics["reason"]) == ("fail", "2 failed and 0 errors")
    assert lines_of(result)[0] == "maintain: fail (nightly)"
    assert "rerun numerics: fail (2 failed and 0 errors)" in lines_of(result)
    assert manifest_of(ws.out)["status"]["reruns"] == "fail"
    assert manifest_of(ws.out)["status"]["state"] == "fail"


def test_a_failed_doctor_check_is_printed_as_a_fail_line_and_is_in_no_file(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    doctor = Doctor(
        [
            finding("D-PY"),
            finding("D-EOLE-CLI", "fail"),
            finding("D-TORCH-CUDA", "skip"),
        ]
    )
    install(monkeypatch, doctor=doctor)

    result = invoke(ws.args())

    assert result.exit_code == 4, result.output
    assert lines_of(result)[0] == f"[FAIL] D-EOLE-CLI: {DOCTOR_MESSAGE}"
    assert lines_of(result)[1] == "maintain: fail (nightly)"
    assert "doctor: fail (1 ok, 1 fail, 1 skipped)" in lines_of(result)
    for name, data in ws.written().items():
        assert DOCTOR_MESSAGE.encode() not in data, name
        assert b"/srv/example" not in data, name
    check = step_of(report_of_run(ws), "doctor")["detail"]["checks"]
    assert {"id": "D-EOLE-CLI", "state": "fail", "reason": "the check failed"} in check


def test_the_reruns_are_skipped_when_the_doctor_failed(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = install(monkeypatch, doctor=Doctor([finding("D-EOLE-CLI", "fail")]))

    assert invoke(ws.args()).exit_code == 4

    reruns = step_of(report_of_run(ws), "reruns")
    assert reruns["state"] == "skipped"
    assert {item["reason"] for item in reruns["detail"]["items"]} == {
        SKIP_ENVIRONMENT,
        "not due at the nightly cadence",
    }
    assert [argv for argv in runner.argvs if "pytest" in argv] == []


def test_only_failed_checks_print_a_line(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)

    result = invoke(ws.args())

    assert not any(line.startswith("[") for line in lines_of(result))


def test_a_doctor_seam_that_raises_is_a_failed_check_and_never_a_pass(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> Sequence[DoctorFinding]:
        raise RuntimeError("the doctor broke")

    install(monkeypatch, doctor=broken)

    result = invoke(ws.args())

    assert result.exit_code == 4, result.output
    assert lines_of(result)[0] == "[FAIL] doctor: RuntimeError: the doctor broke"
    assert step_of(report_of_run(ws), "doctor")["state"] == "fail"
    assert all(b"the doctor broke" not in data for data in ws.written().values())


def test_an_installed_seam_that_raises_ends_the_run_with_the_error_and_no_manifest(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> Any:
        raise RuntimeError("the environment could not be read")

    install(monkeypatch, installed=broken)

    result = invoke(ws.args())

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert not (ws.out / "manifest.json").exists()
    assert not (ws.out / "maintain_report.json").exists()


def test_an_environment_that_differs_from_the_lock_fails_the_run(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = {**LOCKED, "numpy": ("9.9.9",), "stray": ("1.0",)}
    install(monkeypatch, versions=installed)

    result = invoke(ws.args())

    assert result.exit_code == 4, result.output
    assert "environment: fail (1 packages differ from the lock, 1 are not in it)" in (
        lines_of(result)
    )


# ---- the commit protocol --------------------------------------------------------------------------------


def weekly_runner() -> FakeRunner:
    return script_passing_reruns(
        cli_runner(), "robustness_tiny", "evaluation_tiny", "serving_tiny"
    )


class Watching:
    """A runner that notes whether the manifest was there when each command was started."""

    def __init__(self, inner: FakeRunner, manifest: Path) -> None:
        self.inner, self.manifest = inner, manifest
        self.manifest_present: list[bool] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        self.manifest_present.append(self.manifest.exists())
        return self.inner.run(argv, cwd=cwd, env=env, log=log, timeout=timeout)


def test_the_old_manifest_is_removed_before_anything_is_run(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, runner=weekly_runner())
    assert invoke(ws.args("--cadence", "weekly")).exit_code == 0
    watching = Watching(cli_runner(), ws.out / "manifest.json")
    install(monkeypatch, runner=watching)  # type: ignore[arg-type]

    assert invoke(ws.args()).exit_code == 0

    assert watching.manifest_present and not any(watching.manifest_present)


def test_the_files_the_old_manifest_listed_go_and_no_others(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, runner=weekly_runner())
    assert invoke(ws.args("--cadence", "weekly")).exit_code == 0
    assert (ws.out / "logs" / "robustness_tiny.log").is_file()
    assert (ws.out / "junit" / "robustness_tiny.xml").is_file()
    (ws.out / "notes.txt").write_text("the operator's own note")
    (ws.out / "logs" / "extra.log").write_text("not listed by the manifest")
    install(monkeypatch)

    assert invoke(ws.args()).exit_code == 0

    after = ws.written()
    assert "logs/robustness_tiny.log" not in after
    assert "junit/robustness_tiny.xml" not in after
    assert "logs/serving_tiny.log" not in after
    assert after["notes.txt"] == b"the operator's own note"
    assert after["logs/extra.log"] == b"not listed by the manifest"
    assert sorted(after) == [
        "junit/numerics.xml",
        "junit/parity_tiny.xml",
        "logs/drift.log",
        "logs/extra.log",
        "logs/lock_consistent.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
        "maintain_report.json",
        "maintain_report.md",
        "manifest.json",
        "notes.txt",
    ]


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [(KeyboardInterrupt(), 130), (SystemExit(143), 143), (SystemExit(129), 129)],
    ids=["interrupt", "sigterm", "sighup"],
)
def test_an_interrupt_leaves_no_manifest_and_no_report(
    ws: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    exit_code: int,
) -> None:
    install(monkeypatch)
    assert invoke(ws.args()).exit_code == 0
    interrupting = Interrupting(error, is_dry_run, cli_runner())
    install(monkeypatch, runner=interrupting)

    result = invoke(ws.args())

    assert result.exit_code == exit_code, result.output
    after = ws.written()
    assert "manifest.json" not in after
    assert "maintain_report.json" not in after
    assert "maintain_report.md" not in after


def test_an_interrupt_in_a_first_run_leaves_no_manifest(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        runner=Interrupting(KeyboardInterrupt(), is_dry_run, cli_runner()),
    )

    result = invoke(ws.args())

    assert result.exit_code == 130
    assert not (ws.out / "manifest.json").exists()
    assert "maintain:" not in result.stdout


@pytest.mark.parametrize("failing", ["maintain_report.json", "maintain_report.md"])
def test_a_failure_to_write_a_document_leaves_no_manifest(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    install(monkeypatch)
    real = maintain_output.replace_file

    def replace(path: Path, content: bytes, **options: Any) -> None:
        if path.name == failing:
            raise OSError(28, "No space left on device")
        real(path, content, **options)

    monkeypatch.setattr(maintain_output, "replace_file", replace)

    result = invoke(ws.args())

    assert result.exit_code == 1, result.output
    assert "cannot write" in result.stderr
    assert not (ws.out / "manifest.json").exists()
    assert lines_of(result) == []


def test_a_failure_to_write_the_manifest_is_exit_1_and_leaves_none(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)

    def refuse(directory: Path, manifest: Any) -> Path:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(maintain_output.store, "write", refuse)

    result = invoke(ws.args())

    assert result.exit_code == 1, result.output
    assert not (ws.out / "manifest.json").exists()
    assert lines_of(result) == []


def test_a_second_run_gives_the_same_documents_and_a_manifest_that_differs_only_in_time(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    assert invoke(ws.args()).exit_code == 0
    first = ws.written()
    first_manifest = manifest_of(ws.out)
    install(monkeypatch)

    assert invoke(ws.args()).exit_code == 0

    second = ws.written()
    assert second["maintain_report.json"] == first["maintain_report.json"]
    assert second["maintain_report.md"] == first["maintain_report.md"]
    assert sorted(second) == sorted(first)
    second_manifest = manifest_of(ws.out)

    def stable(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            **manifest,
            "created": None,
            "metrics": {
                key: value
                for key, value in manifest["metrics"].items()
                if not key.startswith("seconds.")
            },
        }

    assert stable(second_manifest) == stable(first_manifest)
    assert second_manifest["created"] >= first_manifest["created"]


# ---- the inputs are untouched ---------------------------------------------------------------------------


def nightly(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch)
    return ws.args()


def weekly(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch, runner=weekly_runner())
    return ws.args("--cadence", "weekly")


def preflight(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch)
    return ws.args("--no-drift", "--no-reruns")


def strict(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch)
    return ws.args("--fail-on-drift")


def failing_rerun(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch, runner=night_runner(numerics=(1, junit_xml(5, 5, 0, 0))))
    return ws.args()


def failing_doctor(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch, doctor=Doctor([finding("D-PY", "fail")]))
    return ws.args()


def interrupted(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(
        monkeypatch, runner=Interrupting(KeyboardInterrupt(), is_dry_run, cli_runner())
    )
    return ws.args()


def broken_environment(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    install(monkeypatch, doctor=golden_doctor(), versions={"stray": ("1",)})
    return ws.args()


@pytest.mark.parametrize(
    "scenario",
    [
        nightly,
        weekly,
        preflight,
        strict,
        failing_rerun,
        failing_doctor,
        interrupted,
        broken_environment,
    ],
    ids=lambda scenario: scenario.__name__,
)
def test_the_project_files_and_the_configuration_keep_their_bytes_and_times(
    ws: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Callable[[Workspace, pytest.MonkeyPatch], list[str]],
) -> None:
    watched = [ws.project / "pyproject.toml", ws.project / "uv.lock", ws.config]
    before = [identity(path) for path in watched]
    arguments = scenario(ws, monkeypatch)

    invoke(arguments)

    assert [identity(path) for path in watched] == before


def test_nothing_outside_the_output_directory_is_written(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    before = {
        name: data
        for name, data in snapshot(ws.root).items()
        if not name.startswith("out/")
    }

    assert invoke(ws.args()).exit_code == 0

    after = {
        name: data
        for name, data in snapshot(ws.root).items()
        if not name.startswith("out/")
    }
    assert after == before
