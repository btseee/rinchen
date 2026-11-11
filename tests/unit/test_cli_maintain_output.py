"""What `rinchen maintain check` writes and prints (D14, spec M12): `begin` and `commit` of
`rinchen.cli.maintain_output`, the manifest they write and the lines `result_lines` makes.

The commit protocol is tested on a real output directory that a scripted run of the check filled
(`tests.support.maintain_check.golden_run`): the old manifest goes first and then the files it
listed, by their place inside the directory and never one outside it; the two documents are written
through `replace_file`; the logs and the JUnit files that exist are listed; and the manifest comes
LAST (the order of `Path.replace` calls is the order of the writes). The manifest's contents are
pinned by value: the versions of every locked name, the counts, the seconds of each step and the
state of each step, and no word of it is a path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import maintain_output
from rinchen.cli.maintain_output import (
    REPORT_FILE,
    STAGE,
    SUMMARY_FILE,
    begin,
    commit,
    parse_uv_version,
    result_lines,
)
from rinchen.errors import DataError
from rinchen.maintain.check_types import CheckResult
from rinchen.maintain.lock import LockFile
from rinchen.maintain.report import STEP_IDS, report_to_bytes
from rinchen.maintain.report_markdown import render_markdown
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import build_inputs
from tests.support.maintain_check import Run, golden_run
from tests.support.maintain_check import sha256_of as digest_of_label
from tests.support.maintain_cli import CODE_VERSION, manifest_of, sha256_of
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
from tests.support.uv_outputs import lock_for_samples, lock_of

pytestmark = pytest.mark.usefixtures("no_network")

PYTHON = "3.14.10"
UV = "0.13.7"
ENVIRONMENT = {
    "installed": 6,
    "mismatched": [],
    "mismatched_more": 0,
    "extra": [],
    "extra_more": 0,
    "not_installed": 0,
}


@pytest.fixture(autouse=True)
def constant_code_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maintain_output, "code_version", lambda: CODE_VERSION)


def written(run: Run, *, lock: LockFile | None = None, uv: str = UV) -> Path:
    """Commit the documents and the manifest of a scripted run; return the manifest's path."""
    project = run.project
    return commit(
        run.out,
        report_bytes=report_to_bytes(run.result.report),
        markdown=render_markdown(run.result.report),
        result=run.result,
        inputs=build_inputs([project / "pyproject.toml", project / "uv.lock"]),
        lock=lock_for_samples() if lock is None else lock,
        config_sha256=digest_of_label("maintain.yaml of the golden run"),
        python_version=PYTHON,
        uv_version=uv,
    )


def relative(manifest: dict[str, Any], key: str, out: Path) -> list[str]:
    """The paths a manifest lists under `key`, relative to the output directory."""
    return [Path(item["path"]).relative_to(out).as_posix() for item in manifest[key]]


# ---- commit -------------------------------------------------------------------------------------------


def test_the_stage_and_the_names_of_the_two_documents() -> None:
    assert (STAGE, REPORT_FILE, SUMMARY_FILE) == (
        "maintain_check",
        "maintain_report.json",
        "maintain_report.md",
    )


def test_commit_writes_the_documents_and_lists_every_output_with_its_digest(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    manifest_path = written(run)

    assert manifest_path == run.out / "manifest.json"
    manifest = manifest_of(run.out)
    assert relative(manifest, "outputs", run.out) == [
        "maintain_report.json",
        "maintain_report.md",
        "logs/lock_consistent.log",
        "logs/drift.log",
        "logs/numerics.log",
        "logs/parity_tiny.log",
        "junit/numerics.xml",
        "junit/parity_tiny.xml",
    ]
    for item in manifest["outputs"]:
        path = Path(item["path"])
        assert item["sha256"] == sha256_of(path)
        assert item["bytes"] == path.stat().st_size
    assert (run.out / REPORT_FILE).read_bytes() == report_to_bytes(run.result.report)
    assert (run.out / SUMMARY_FILE).read_text(encoding="utf-8") == render_markdown(
        run.result.report
    )


def test_the_manifest_is_written_after_every_file_it_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = golden_run(tmp_path)
    order: list[str] = []
    real = Path.replace

    def spy(self: Path, target: Any) -> Any:
        order.append(Path(target).name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", spy)

    written(run)

    assert order == [REPORT_FILE, SUMMARY_FILE, "manifest.json"]


def test_the_manifest_records_the_stage_the_inputs_the_seed_and_the_config_digest(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    manifest = manifest_of(run.out)
    assert manifest["stage"] == STAGE
    assert manifest["schema"] == "rinchen.manifest/1"
    assert manifest["seed"] == -1
    assert manifest["code_version"] == CODE_VERSION
    assert manifest["config_sha256"] == digest_of_label(
        "maintain.yaml of the golden run"
    )
    assert [Path(item["path"]).name for item in manifest["inputs"]] == [
        "pyproject.toml",
        "uv.lock",
    ]
    Manifest.model_validate(manifest)


def test_the_tool_versions_are_every_locked_name_and_the_two_tools(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    assert manifest_of(run.out)["tool_versions"] == {
        "numpy": "2.5.4",
        "rich": "15.0.0",
        "huggingface-hub": "2.3.0",
        "safetensors": "0.8.0",
        "nvidia-cudnn-cu13": "9.24.0.43",
        "sortedcontainers": "2.4.0",
        "python": PYTHON,
        "uv": UV,
    }


def test_a_name_with_several_versions_lists_all_of_them_in_order(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    lock = lock_of({"twin": ("2.0", "1.0", "1.5"), "solo": ("3.0",)})

    written(run, lock=lock)

    versions = manifest_of(run.out)["tool_versions"]
    assert versions["twin"] == "1.0, 1.5, 2.0"
    assert versions["solo"] == "3.0"


@pytest.mark.parametrize(
    ("name", "key"), [("uv", "uv (tool)"), ("python", "python (tool)")]
)
def test_a_locked_package_named_like_a_tool_keeps_its_own_version(
    tmp_path: Path, name: str, key: str
) -> None:
    run = golden_run(tmp_path)

    written(run, lock=lock_of({name: ("9.9",), "numpy": ("2.5.4",)}))

    versions = manifest_of(run.out)["tool_versions"]
    assert versions[name] == "9.9"
    assert versions[key] == (UV if name == "uv" else PYTHON)


def test_the_metrics_are_the_counts_and_the_seconds_of_each_step(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    metrics = manifest_of(run.out)["metrics"]
    assert metrics == {
        "packages": 6.0,
        "entries": 7.0,
        "blocked": 4.0,
        "reruns_run": 2.0,
        **{f"seconds.{step}": run.result.seconds[step] for step in STEP_IDS},
    }
    assert all(metrics[f"seconds.{step}"] > 0 for step in STEP_IDS)


def test_the_manifest_is_dated_in_utc_at_the_time_it_is_written(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    before = datetime.now(UTC)

    written(run)

    after = datetime.now(UTC)
    created = datetime.fromisoformat(manifest_of(run.out)["created"])
    assert created.utcoffset() == timedelta(0)
    assert before <= created <= after


def test_the_status_is_the_state_of_each_step_and_the_overall_state(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    assert manifest_of(run.out)["status"] == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "drift",
        "reruns": "ok",
        "state": "drift",
    }


def test_nothing_in_the_status_the_versions_or_the_metrics_is_a_path_or_a_reason(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    manifest = manifest_of(run.out)
    for table in ("status", "tool_versions"):
        for value in manifest[table].values():
            assert "/" not in value and " " not in value.replace(", ", "")
    assert all(isinstance(value, float) for value in manifest["metrics"].values())


def test_the_junit_file_of_a_rerun_that_did_not_run_is_not_listed(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    (run.out / "junit").mkdir(exist_ok=True)
    (run.out / "junit" / "robustness_tiny.xml").write_text("<stale/>")

    written(run)

    assert "junit/robustness_tiny.xml" not in relative(
        manifest_of(run.out), "outputs", run.out
    )


def test_a_junit_file_that_is_gone_is_not_listed(tmp_path: Path) -> None:
    run = golden_run(tmp_path)
    (run.out / "junit" / "numerics.xml").unlink()

    written(run)

    listed = relative(manifest_of(run.out), "outputs", run.out)
    assert "junit/numerics.xml" not in listed
    assert "junit/parity_tiny.xml" in listed


def test_commit_fails_with_an_os_error_when_a_document_cannot_be_written(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    (run.out / REPORT_FILE).mkdir()

    with pytest.raises(OSError):
        written(run)

    assert not (run.out / "manifest.json").exists()


# ---- begin --------------------------------------------------------------------------------------------


def old_run(tmp_path: Path) -> Run:
    """A finished run whose manifest is in place."""
    run = golden_run(tmp_path)
    written(run)
    return run


def test_begin_removes_the_old_manifest_and_then_the_files_it_listed(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    (run.out / "notes.txt").write_text("mine")
    (run.out / "logs" / "other.log").write_text("not listed")

    begin(run.out)

    assert sorted(path.name for path in run.out.rglob("*") if path.is_file()) == [
        "notes.txt",
        "other.log",
    ]


def test_begin_removes_the_manifest_before_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = old_run(tmp_path)
    seen: list[bool] = []
    real = Path.unlink

    def spy(self: Path, missing_ok: bool = False) -> None:
        if self.name != "manifest.json":
            seen.append((run.out / "manifest.json").exists())
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spy)

    begin(run.out)

    assert seen and not any(seen)


def test_begin_never_removes_a_file_outside_the_output_directory(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    outside = tmp_path / "outside" / "important.txt"
    outside.parent.mkdir()
    outside.write_text("keep")
    (outside.parent / "logs").mkdir()
    (outside.parent / "logs" / "drift.log").write_text("keep")
    escape = tmp_path / "escape.txt"
    escape.write_text("keep")
    manifest = store.read(run.out)
    assert manifest is not None
    manifest.outputs = [
        ManifestFile(path=str(outside), sha256="0" * 64),
        ManifestFile(path=str(run.out / ".." / "escape.txt"), sha256="0" * 64),
        ManifestFile(path="../escape.txt", sha256="0" * 64),
        ManifestFile(
            path=str(tmp_path / "outside" / "logs" / "drift.log"), sha256="0" * 64
        ),
        *manifest.outputs,
    ]
    store.write(run.out, manifest)

    begin(run.out)

    assert outside.read_text() == "keep"
    assert (outside.parent / "logs" / "drift.log").read_text() == "keep"
    assert escape.read_text() == "keep"
    assert not (run.out / "logs" / "drift.log").exists()


def test_begin_removes_only_the_files_this_command_writes(tmp_path: Path) -> None:
    # A manifest that lists a file of another shape (one that this command never writes) does
    # not make that file ours to remove, even inside the output directory.
    run = old_run(tmp_path)
    keep = [
        run.out / "notes.txt",
        run.out / "logs" / "deeper" / "x.log",
        run.out / "other" / "x.log",
    ]
    for path in keep:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep")
    manifest = store.read(run.out)
    assert manifest is not None
    manifest.outputs = [
        *(ManifestFile(path=str(path), sha256="0" * 64) for path in keep),
        *manifest.outputs,
    ]
    store.write(run.out, manifest)

    begin(run.out)

    assert all(path.read_text() == "keep" for path in keep)
    assert not (run.out / "logs" / "drift.log").exists()
    assert not (run.out / REPORT_FILE).exists()


def test_begin_does_not_follow_a_linked_directory_out_of_the_output_directory(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (run.out / "logs").rename(elsewhere)
    (run.out / "logs").symlink_to(elsewhere, target_is_directory=True)

    begin(run.out)

    assert sorted(path.name for path in elsewhere.iterdir()) == [
        "drift.log",
        "lock_consistent.log",
        "numerics.log",
        "parity_tiny.log",
    ]
    assert not (run.out / "manifest.json").exists()


def test_begin_leaves_a_directory_that_the_manifest_names_alone(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    manifest = store.read(run.out)
    assert manifest is not None
    manifest.outputs = [
        ManifestFile(path=str(run.out / "logs"), sha256="0" * 64),
    ]
    store.write(run.out, manifest)

    begin(run.out)

    assert (run.out / "logs" / "drift.log").is_file()
    assert not (run.out / "manifest.json").exists()


def test_begin_refuses_the_manifest_of_another_stage_and_removes_nothing(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    manifest = store.read(run.out)
    assert manifest is not None
    manifest.stage = "export_parity"
    store.write(run.out, manifest)
    before = {path: path.read_bytes() for path in run.out.rglob("*") if path.is_file()}

    with pytest.raises(DataError, match="export_parity"):
        begin(run.out)

    assert {
        path: path.read_bytes() for path in run.out.rglob("*") if path.is_file()
    } == before


def test_begin_on_an_unreadable_manifest_removes_it_and_nothing_else(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    (run.out / "manifest.json").write_text("{not json")

    begin(run.out)

    assert not (run.out / "manifest.json").exists()
    assert (run.out / "logs" / "drift.log").is_file()


def test_begin_in_a_directory_that_does_not_exist_makes_nothing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nowhere"

    begin(target)

    assert not target.exists()


def test_a_manifest_moved_with_its_directory_still_lets_begin_find_its_files(
    tmp_path: Path,
) -> None:
    run = old_run(tmp_path)
    moved = tmp_path / "moved"
    run.out.rename(moved)

    begin(moved)

    assert not (moved / "logs" / "drift.log").exists()
    assert not (moved / "junit" / "numerics.xml").exists()
    assert not (moved / "manifest.json").exists()


# ---- the version of uv --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("uv 0.13.7 (x86_64-unknown-linux-gnu)\n", "0.13.7"),
        ("uv 0.13.7\n", "0.13.7"),
        ("uv 0.14.0-alpha.1+local (abc 2026-01-01)\r\n", "0.14.0-alpha.1+local"),
        ("uv 0.13.7 (x)\nsecond line\n", "0.13.7"),
        ("", "unknown"),
        ("uv\n", "unknown"),
        ("uv /home/someone/bin\n", "unknown"),
        ("pip 24.0 from /usr/lib\n", "unknown"),
        ("error: no such command\n", "unknown"),
        ("uv " + "9" * 40 + "\n", "unknown"),
        ("\nuv 0.13.7\n", "unknown"),
    ],
)
def test_the_version_of_uv_is_a_token_of_its_first_line_or_unknown(
    text: str, version: str
) -> None:
    assert parse_uv_version(text) == version


# ---- the printed lines --------------------------------------------------------------------------------


def test_the_lines_of_the_golden_run_are_one_per_step_and_rerun_and_then_the_paths(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)
    manifest = run.out / "manifest.json"

    lines = result_lines(run.result.report, run.out, manifest)

    assert lines == [
        "maintain: drift (nightly)",
        "doctor: ok (6 ok, 0 fail, 1 skipped)",
        "lock: ok (6 entries, 6 names)",
        "environment: ok (6 installed, 0 differ, 0 extra, 0 not installed)",
        "drift: drift (7 entries: 5 update, 1 add, 1 remove; 4 blocked)",
        "rerun numerics: ok (1110 passed)",
        "rerun parity_tiny: ok (13 passed)",
        "rerun parity_export: skipped (no export was given)",
        "rerun robustness_tiny: skipped (not due at the nightly cadence)",
        "rerun evaluation_tiny: skipped (not due at the nightly cadence)",
        "rerun serving_tiny: skipped (not due at the nightly cadence)",
        f"report: {run.out / REPORT_FILE}",
        f"summary: {run.out / SUMMARY_FILE}",
        f"manifest: {manifest}",
    ]


def full_report(**steps: tuple[str, str | None, dict[str, Any]]) -> Any:
    """A report whose steps are complete (shapes of the document) and as named: a step is
    `(state, reason, detail)`."""
    standard: dict[str, tuple[str, str | None, dict[str, Any]]] = {
        "doctor": ("ok", None, doctor_detail(check_row("D-PY"))),
        "lock_file": ("ok", None, {}),
        "lock_consistent": ("ok", None, {}),
        "environment": ("ok", None, ENVIRONMENT),
        "drift": ("ok", None, drift_detail([])),
        "reruns": (
            "ok",
            None,
            reruns_detail(rerun_item("numerics", tests=(5, 0, 0, 0))),
        ),
    }
    standard.update(steps)
    return report_of(
        {step: state for step, (state, _, _) in standard.items()},
        {step: reason for step, (_, reason, _) in standard.items() if reason},
        **{step: detail for step, (_, _, detail) in standard.items()},
    )


def lines_for(report: Any, tmp_path: Path) -> list[str]:
    return result_lines(report, tmp_path / "out", tmp_path / "out" / "manifest.json")


def test_a_failed_doctor_counts_its_checks_by_state() -> None:
    report = full_report(
        doctor=(
            "fail",
            None,
            doctor_detail(
                check_row("D-PY"),
                check_row("D-EOLE-CLI", "fail", "the check failed"),
                check_row("D-TORCH-CUDA", "skipped", "a GPU check"),
                check_row("D-WHEELS", "fail", "the check failed"),
            ),
        )
    )

    lines = lines_for(report, Path("/x"))

    assert lines[1] == "doctor: fail (1 ok, 2 fail, 1 skipped)"
    assert lines[0] == "maintain: fail (nightly)"


def test_a_failed_lock_check_prints_its_reason(tmp_path: Path) -> None:
    report = full_report(
        lock_consistent=("fail", "uv exited with code 2", {}),
    )

    lines = lines_for(report, tmp_path)

    assert lines[2] == "lock: fail (uv exited with code 2)"


def test_a_failed_environment_prints_its_reason_and_a_skipped_drift_its_own(
    tmp_path: Path,
) -> None:
    report = full_report(
        environment=(
            "fail",
            "2 packages differ from the lock, 1 are not in it",
            {**ENVIRONMENT, "extra": ["x"]},
        ),
        drift=("skipped", "disabled by --no-drift", {}),
    )

    lines = lines_for(report, tmp_path)

    assert (
        lines[3]
        == "environment: fail (2 packages differ from the lock, 1 are not in it)"
    )
    assert lines[4] == "drift: skipped (disabled by --no-drift)"


def test_an_environment_counts_what_differs_beyond_the_names_it_lists(
    tmp_path: Path,
) -> None:
    detail = {
        **ENVIRONMENT,
        "installed": 30,
        "mismatched": [{"name": "a", "installed": ["1"], "locked": ["2"]}],
        "mismatched_more": 4,
        "extra": ["b", "c"],
        "extra_more": 3,
        "not_installed": 7,
    }

    lines = lines_for(full_report(environment=("ok", None, detail)), tmp_path)

    assert (
        lines[3] == "environment: ok (30 installed, 5 differ, 5 extra, 7 not installed)"
    )


def test_a_drift_step_without_changes_says_so_and_a_failed_one_prints_its_reason(
    tmp_path: Path,
) -> None:
    quiet = lines_for(full_report(), tmp_path)
    failed = lines_for(
        full_report(drift=("fail", "uv printed no recognisable result", {})), tmp_path
    )

    assert quiet[4] == "drift: ok (no changes)"
    assert failed[4] == "drift: fail (uv printed no recognisable result)"


def test_the_drift_line_counts_the_changes_and_the_blocked_entries(
    tmp_path: Path,
) -> None:
    entries = [
        drift_entry("a", "update", klass="blocked", rule="r"),
        drift_entry("b", "add", "1.0", None),
        drift_entry("c", "remove", "1.0", None),
    ]

    lines = lines_for(
        full_report(drift=("drift", None, drift_detail(entries))), tmp_path
    )

    assert lines[4] == "drift: drift (3 entries: 1 update, 1 add, 1 remove; 1 blocked)"


def test_a_rerun_line_gives_the_counters_the_figures_or_the_reason(
    tmp_path: Path,
) -> None:
    items = reruns_detail(
        rerun_item("numerics", "fail", "3 failed and 0 errors", tests=(10, 3, 0, 0)),
        rerun_item(
            "parity_export",
            kind="export_parity",
            figures={"exact": 199, "sentences": 200, "mean_abs_delta": 1e-06},
        ),
        rerun_item("serving_tiny", "skipped", "no export was given"),
        rerun_item("odd", "fail", "no JUnit report was written"),
    )

    lines = lines_for(full_report(reruns=("fail", None, items)), tmp_path)

    assert lines[5:9] == [
        "rerun numerics: fail (3 failed and 0 errors)",
        "rerun parity_export: ok (199/200 exact, mean |delta| 1.00e-06)",
        "rerun serving_tiny: skipped (no export was given)",
        "rerun odd: fail (no JUnit report was written)",
    ]


def test_only_the_last_three_lines_hold_a_path(tmp_path: Path) -> None:
    run = golden_run(tmp_path)

    lines = result_lines(run.result.report, run.out, run.out / "manifest.json")

    assert [line for line in lines if str(tmp_path) in line] == lines[-3:]
    assert all("/" not in line for line in lines[:-3])


def test_a_report_without_a_rerun_has_no_rerun_lines(tmp_path: Path) -> None:
    report = full_report(reruns=("skipped", "disabled by --no-reruns", reruns_detail()))

    lines = lines_for(report, tmp_path)

    assert not any(line.startswith("rerun ") for line in lines)
    assert len(lines) == 8


def test_the_summary_json_of_a_commit_is_the_report_that_was_passed(
    tmp_path: Path,
) -> None:
    run = golden_run(tmp_path)

    written(run)

    document: dict[str, Any] = json.loads((run.out / REPORT_FILE).read_bytes())
    assert document["state"] == "drift"
    assert isinstance(run.result, CheckResult)
