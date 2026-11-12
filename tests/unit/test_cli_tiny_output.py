"""What `rinchen tiny run` writes and prints (D15 Task 7, brief G7.3, spec P11 and P13): the names an output
directory may hold, the old run's files that go before a new one starts, the two documents, the manifest
that ends the commit and the lines on standard output.

The commit has two halves: `begin` removes the old manifest FIRST (so a run that is interrupted leaves none
that vouches for a mix of old and new files) and then the stage directories and the documents the run owns;
`write_documents` and `write_manifest` end it, the manifest last. The manifest records the lock's resolved
versions (every name, a name locked at several versions joined by `, ` in sorted order, plus `python`), the
checkpoint files that exist when it is written, the seconds of each stage and the two figures a run gives.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import tiny_output
from rinchen.cli.tiny_output import (
    OWN_ENTRIES,
    STAGE,
    ReportDefect,
    begin,
    build_manifest,
    checkpoint_files,
    clear_run,
    observe_code_version,
    report_documents,
    require_only_own_entries,
    result_lines,
    write_documents,
    write_manifest,
)
from rinchen.config.tiny import STAGE_IDS
from rinchen.errors import ConfigError, DataError
from rinchen.maintain.lock import read_lock
from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.tiny.context import StageResult
from rinchen.tiny.pipeline import PipelineRun, run
from rinchen.tiny.report import to_bytes
from rinchen.tiny.report_markdown import render
from rinchen.tiny.states import StageRecord
from tests.support import tiny_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import (
    FakeClock,
    install,
    invoke,
    manifest_of,
    run_args,
    watch_first_stage,
)
from tests.support.tiny_pipeline_runs import make_plan, real_functions
from tests.support.tiny_report_build import build
from tests.support.tiny_report_inputs import golden_results
from tests.support.tiny_run_eval import EvalRunner
from tests.support.tiny_stage_inputs import shipped_config

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of `tests/support/tiny_run.py`, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
out_dir = tiny_run.out_dir

STAGE_DIRECTORIES = (
    "01_fetch",
    "02_corpus",
    "03_filter",
    "04_spm",
    "05_converter",
    "06_synth",
    "07_train",
    "08_export",
    "09_parity",
    "10_robustness",
    "11_evaluate",
)


# ---- the names of an output directory -----------------------------------------------------------------------


def test_the_names_an_output_directory_may_hold_are_the_documents_the_logs_and_the_stage_directories() -> (
    None
):
    assert set(OWN_ENTRIES) == {
        "manifest.json",
        "report.json",
        "report.md",
        "logs",
        *STAGE_DIRECTORIES,
    }
    assert len(OWN_ENTRIES) == len(set(OWN_ENTRIES)) == 15
    assert STAGE == "tiny_run"


def test_a_directory_that_does_not_exist_or_is_empty_or_holds_only_own_names_passes(
    tmp_path: Path,
) -> None:
    require_only_own_entries(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    require_only_own_entries(empty)
    own = tmp_path / "own"
    own.mkdir()
    for name in OWN_ENTRIES:
        (own / name).write_text("x", encoding="utf-8")
    require_only_own_entries(own)


@pytest.mark.parametrize(
    "name", ["src", "notes.txt", ".hidden", "01_fetch.bak", "Report.json"]
)
def test_an_entry_the_command_does_not_write_is_refused_and_not_named(
    name: str, tmp_path: Path
) -> None:
    (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        require_only_own_entries(tmp_path)

    message = str(raised.value)
    assert "holds 1 entry" in message
    assert name not in message
    assert ".part" not in message


def test_the_message_counts_the_entries(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()

    with pytest.raises(ConfigError, match="holds 3 entries"):
        require_only_own_entries(tmp_path)


def test_a_part_file_counts_and_the_message_says_to_delete_it(tmp_path: Path) -> None:
    (tmp_path / "x.part").write_text("x", encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        require_only_own_entries(tmp_path)

    assert "holds 1 entry" in str(raised.value)
    assert "delete it" in str(raised.value)
    assert "x.part" not in str(raised.value)


# ---- the old run goes --------------------------------------------------------------------------------------


def lay_down_a_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in STAGE_DIRECTORIES:
        (root / name / "logs").mkdir(parents=True)
        (root / name / "logs" / "x.log").write_text("x", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "fetch-raised.log").write_text("x", encoding="utf-8")
    for name in ("report.json", "report.md", "manifest.json"):
        (root / name).write_text("old", encoding="utf-8")


def test_clear_run_removes_the_stage_directories_the_logs_and_the_two_documents(
    tmp_path: Path,
) -> None:
    lay_down_a_run(tmp_path)
    (tmp_path / "keep.txt").write_text("not ours", encoding="utf-8")

    clear_run(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "keep.txt",
        "manifest.json",
    ]


def test_clear_run_on_a_directory_that_does_not_exist_is_nothing(
    tmp_path: Path,
) -> None:
    clear_run(tmp_path / "missing")

    assert not (tmp_path / "missing").exists()


def test_clear_run_takes_away_a_link_and_a_file_in_the_place_of_a_stage_directory(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious.txt").write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "01_fetch").symlink_to(elsewhere, target_is_directory=True)
    (out / "02_corpus").write_text("a file", encoding="utf-8")

    clear_run(out)

    assert list(out.iterdir()) == []
    assert (elsewhere / "precious.txt").read_text(encoding="utf-8") == "x"


def test_begin_removes_the_old_manifest_first_and_then_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lay_down_a_run(tmp_path)
    store.write(
        tmp_path,
        Manifest(
            stage=STAGE, created="x", code_version="x", config_sha256="0" * 64, seed=1
        ),
    )
    seen: list[tuple[str, bool, bool]] = []
    real_begin_commit = tiny_output.begin_commit
    real_clear = tiny_output.clear_run

    def commit(directory: Path, name: str) -> None:
        seen.append(
            (
                "begin_commit",
                (tmp_path / "manifest.json").exists(),
                (tmp_path / "report.json").exists(),
            )
        )
        real_begin_commit(directory, name)
        seen.append(
            (
                "after begin_commit",
                (tmp_path / "manifest.json").exists(),
                (tmp_path / "report.json").exists(),
            )
        )

    def clearing(directory: Path) -> None:
        seen.append(
            (
                "clear_run",
                (tmp_path / "manifest.json").exists(),
                (tmp_path / "report.json").exists(),
            )
        )
        real_clear(directory)

    monkeypatch.setattr(tiny_output, "begin_commit", commit)
    monkeypatch.setattr(tiny_output, "clear_run", clearing)

    begin(tmp_path)

    assert seen == [
        ("begin_commit", True, True),
        ("after begin_commit", False, True),
        ("clear_run", False, True),
    ]
    assert list(tmp_path.iterdir()) == []


def test_the_manifest_of_another_stage_is_refused_and_nothing_is_removed(
    tmp_path: Path,
) -> None:
    lay_down_a_run(tmp_path)
    store.write(
        tmp_path,
        Manifest(
            stage="spm_train",
            created="x",
            code_version="x",
            config_sha256="0" * 64,
            seed=1,
        ),
    )
    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}

    with pytest.raises(DataError, match="another|not of"):
        begin(tmp_path)

    assert {
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
    } == before


# ---- the code version --------------------------------------------------------------------------------------


def test_the_code_version_is_the_one_of_the_manifest_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tiny_output, "code_version", lambda: "abc1234")

    assert observe_code_version() == "abc1234"


def git(repo: Path, *args: str) -> str:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def committed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    (repo / "tracked.txt").write_text("one", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "--quiet", "-m", "first")
    monkeypatch.setattr(stage, "REPO_ROOT", repo)
    return repo


@pytest.mark.subprocess
def test_a_clean_checkout_gets_its_plain_sha_and_a_changed_one_is_dirty(
    committed_repo: Path,
) -> None:
    head = git(committed_repo, "rev-parse", "HEAD")[:7]

    assert observe_code_version() == head

    (committed_repo / "tracked.txt").write_text("two", encoding="utf-8")
    assert observe_code_version() == f"{head}-dirty"


# ---- the documents -----------------------------------------------------------------------------------------


def test_the_documents_are_written_whole_and_listed_with_their_digests(
    tmp_path: Path,
) -> None:
    written = write_documents(tmp_path, b'{"a": 1}\n', "# text\n")

    assert [entry.path for entry in written] == [
        str(tmp_path / "report.json"),
        str(tmp_path / "report.md"),
    ]
    assert (tmp_path / "report.json").read_bytes() == b'{"a": 1}\n'
    assert (tmp_path / "report.md").read_bytes() == b"# text\n"
    assert [entry.bytes for entry in written] == [9, 7]
    assert all(len(entry.sha256) == 64 for entry in written)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "report.json",
        "report.md",
    ]


def test_the_documents_replace_old_ones(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("old", encoding="utf-8")

    write_documents(tmp_path, b"new\n", "new\n")

    assert (tmp_path / "report.json").read_bytes() == b"new\n"


def test_a_document_that_cannot_be_written_is_an_os_error_and_leaves_no_part_file_and_no_half_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Path.write_bytes

    def failing(self: Path, data: bytes) -> int:
        if self.name.startswith("report.md"):
            real(self, data[: len(data) // 2])
            raise OSError("disk full")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", failing)

    with pytest.raises(OSError, match="disk full"):
        write_documents(tmp_path, b"a\n", "bcdefgh\n")

    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


# ---- the report as it is written -------------------------------------------------------------------------


def test_the_documents_of_a_report_are_its_bytes_and_its_markdown() -> None:
    report = build()

    data, markdown = report_documents(report)

    assert data == to_bytes(report)
    assert markdown == render(report)


def test_a_report_whose_markdown_does_not_survive_a_round_trip_is_a_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = tiny_output.render

    def drifting(report: Any) -> str:
        calls.append(1)
        return real(report) + ("x" if len(calls) > 1 else "")

    monkeypatch.setattr(tiny_output, "render", drifting)

    with pytest.raises(ReportDefect):
        report_documents(build())


def test_bytes_that_do_not_parse_back_are_a_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tiny_output, "to_bytes", lambda report: b"{}\n")

    with pytest.raises(ReportDefect):
        report_documents(build())


# ---- the checkpoint files ------------------------------------------------------------------------------------


def test_with_no_stage_run_there_are_no_checkpoint_files(tmp_path: Path) -> None:
    assert checkpoint_files(tmp_path, shipped_config()) == []


def test_the_checkpoint_files_that_exist_are_listed_in_the_order_of_the_stages(
    tmp_path: Path,
) -> None:
    files = [
        tmp_path / "04_spm" / "tiny.model",
        tmp_path / "05_converter" / "model" / "model.bin",
        tmp_path
        / "07_train"
        / "ablate"
        / "variants"
        / "v1"
        / "run"
        / "averaged"
        / "model.00.safetensors",
        tmp_path / "08_export" / "float32" / "model.bin",
        tmp_path / "08_export" / "int8" / "model.bin",
    ]
    for path in files:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")

    found = checkpoint_files(tmp_path, shipped_config())

    assert found == [files[0], files[1], files[2], files[3], files[4]]


def test_a_failed_run_lists_fewer_checkpoint_files(tmp_path: Path) -> None:
    for relative in ("04_spm/tiny.model", "08_export/int8/model.bin"):
        (tmp_path / relative).parent.mkdir(parents=True)
        (tmp_path / relative).write_bytes(b"x")

    found = checkpoint_files(tmp_path, shipped_config())

    assert [path.relative_to(tmp_path).as_posix() for path in found] == [
        "04_spm/tiny.model",
        "08_export/int8/model.bin",
    ]


def test_the_checkpoint_files_are_where_the_real_stages_write_them(
    tmp_path: Path,
) -> None:
    """The paths of the manifest are the stages' own: the real stage functions, over the runner that
    plays every command, write the five files, and the manifest finds every one."""
    runner = EvalRunner()
    plan = make_plan(tmp_path / "out", runner=runner)

    result = run(plan, FakeClock(), functions=real_functions())

    assert [item.record.state for item in result.results] == ["ok"] * 15
    found = checkpoint_files(plan.out_dir, plan.config)
    assert [path.name for path in found] == [
        "tiny.model",
        "model.bin",
        "model.00.safetensors",
        "model.bin",
        "model.bin",
    ]
    assert [path.parent.name for path in found if path.name == "model.bin"] == [
        "model",
        "float32",
        "int8",
    ]
    assert all(path.is_file() for path in found)


# ---- the manifest --------------------------------------------------------------------------------------------


def a_run(
    seconds: float = 2.0, results: list[StageResult] | None = None
) -> PipelineRun:
    chosen = results if results is not None else golden_results()
    return PipelineRun(
        tuple(chosen),
        {item.record.id: seconds for item in chosen},
        seconds * len(chosen),
    )


LOCK_TEXT = b"""version = 1
revision = 3

[[package]]
name = "Alpha_Pkg"
version = "1.0"

[[package]]
name = "beta"
version = "2.1"

[[package]]
name = "beta"
version = "2.0"

[[package]]
name = "python"
version = "9.9"
"""


def manifest_with(tmp_path: Path, **changes: Any) -> Manifest:
    arguments: dict[str, Any] = {
        "out_dir": tmp_path,
        "config": shipped_config(),
        "config_sha256": "a" * 64,
        "seed": 7,
        "inputs": [ManifestFile(path="/x/in", sha256="b" * 64, bytes=3)],
        "documents": [
            ManifestFile(path=str(tmp_path / "report.json"), sha256="c" * 64, bytes=9)
        ],
        "lock": read_lock(LOCK_TEXT),
        "run": a_run(),
        "code_version": "abc1234",
        "python_version": "3.14.10",
    }
    arguments.update(changes)
    return build_manifest(**arguments)


def test_the_manifest_is_the_one_of_the_stage_tiny_run(tmp_path: Path) -> None:
    manifest = manifest_with(tmp_path)

    assert manifest.stage == "tiny_run"
    assert manifest.code_version == "abc1234"
    assert manifest.config_sha256 == "a" * 64
    assert manifest.seed == 7
    assert manifest.inputs == [ManifestFile(path="/x/in", sha256="b" * 64, bytes=3)]
    assert manifest.created.endswith("+00:00")


def test_the_outputs_are_the_documents_and_the_checkpoint_files_that_exist(
    tmp_path: Path,
) -> None:
    model = tmp_path / "08_export" / "int8" / "model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")

    manifest = manifest_with(tmp_path)

    assert [item.path for item in manifest.outputs] == [
        str(tmp_path / "report.json"),
        str(model),
    ]
    assert manifest.outputs[1].bytes == 7
    assert len(manifest.outputs[1].sha256) == 64


def test_the_tool_versions_are_every_name_of_the_lock_and_python(
    tmp_path: Path,
) -> None:
    manifest = manifest_with(tmp_path)

    assert manifest.tool_versions == {
        "alpha-pkg": "1.0",
        "beta": "2.0, 2.1",
        "python": "9.9",
        "python (tool)": "3.14.10",
    }


def test_a_lock_without_a_package_named_python_gives_the_tool_its_plain_name(
    tmp_path: Path,
) -> None:
    lock = read_lock(b'version = 1\n\n[[package]]\nname = "a"\nversion = "1"\n')

    manifest = manifest_with(tmp_path, lock=lock)

    assert manifest.tool_versions == {"a": "1", "python": "3.14.10"}


def test_the_real_lock_gives_every_one_of_its_186_names_and_python(
    tmp_path: Path,
) -> None:
    real_lock = read_lock(
        Path(__file__).resolve().parents[2].joinpath("uv.lock").read_bytes()
    )

    manifest = manifest_with(tmp_path, lock=real_lock)

    assert len(real_lock.versions) == 186
    assert len(manifest.tool_versions) == 187
    assert set(real_lock.versions) <= set(manifest.tool_versions)


def test_the_metrics_are_the_seconds_of_every_stage_and_the_two_figures(
    tmp_path: Path,
) -> None:
    seconds = {stage_id: float(index) + 0.5 for index, stage_id in enumerate(STAGE_IDS)}
    run_ = PipelineRun(tuple(golden_results()), seconds, 99.0)

    manifest = manifest_with(tmp_path, run=run_)

    assert manifest.metrics == {
        **{f"seconds.{stage_id}": value for stage_id, value in seconds.items()},
        "rows.kept": 756.0,
        "train.steps": 1500.0,
    }


def test_a_figure_a_stage_did_not_give_is_left_out_and_not_made_up(
    tmp_path: Path,
) -> None:
    results = golden_results()[:3]
    results.append(StageResult(StageRecord("filter", "fail", "exit code 4"), {}))
    results += [
        StageResult(
            StageRecord(item.record.id, "skipped", "an earlier stage failed"), {}
        )
        for item in golden_results()[4:]
    ]

    manifest = manifest_with(tmp_path, run=a_run(results=results))

    assert "rows.kept" not in manifest.metrics
    assert "train.steps" not in manifest.metrics
    assert len([key for key in manifest.metrics if key.startswith("seconds.")]) == 15


@pytest.mark.parametrize("value", [True, "756", 1.5, -1, None])
def test_a_figure_that_is_not_a_count_is_left_out(value: Any, tmp_path: Path) -> None:
    results = golden_results()
    results[3] = StageResult(results[3].record, {**results[3].facts, "kept": value})

    manifest = manifest_with(tmp_path, run=a_run(results=results))

    assert "rows.kept" not in manifest.metrics


def test_the_status_is_the_state_of_each_stage_and_nothing_else(tmp_path: Path) -> None:
    results = golden_results()
    results[5] = StageResult(
        StageRecord("converter", "skipped", "an earlier stage failed"), {}
    )
    results[6] = StageResult(StageRecord("synth", "fail", "exit code 4"), {})

    manifest = manifest_with(tmp_path, run=a_run(results=results))

    assert list(manifest.status) == list(STAGE_IDS)
    assert manifest.status["converter"] == "skipped"
    assert manifest.status["synth"] == "fail"
    assert manifest.status["fetch"] == "ok"


def test_the_manifest_is_written_whole_and_reads_back(tmp_path: Path) -> None:
    manifest = manifest_with(tmp_path)

    path = write_manifest(tmp_path, manifest)

    assert path == tmp_path / "manifest.json"
    assert store.read(tmp_path) == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "tiny_run"
    assert [item.name for item in tmp_path.iterdir()] == ["manifest.json"]


# ---- the printed lines ---------------------------------------------------------------------------------------


def shown(value: float | None) -> str:
    return "-" if value is None else format(value, ".6g")


def test_the_lines_are_the_stages_the_thresholds_the_verdict_and_the_three_paths(
    tmp_path: Path,
) -> None:
    report = build()

    lines = result_lines(report, 123.456, tmp_path, tmp_path / "manifest.json")

    wanted = [f"{stage_id}: ok" for stage_id in STAGE_IDS]
    for threshold in report.thresholds:
        value = 123.456 if threshold.id == "wall_time" else threshold.value
        wanted.append(
            f"threshold {threshold.id}: {threshold.verdict} (value {shown(value)}, bound {threshold.bound})"
        )
    wanted += [
        "verdict: pass",
        f"report: {tmp_path / 'report.json'}",
        f"summary: {tmp_path / 'report.md'}",
        f"manifest: {tmp_path / 'manifest.json'}",
    ]
    assert lines == wanted


def test_a_skip_and_a_failure_are_printed_with_their_closed_reasons(
    tmp_path: Path,
) -> None:
    from tests.support.tiny_report_inputs import failed, replaced, skipped_after

    results = replaced(golden_results(), "spm", failed("spm"))
    results = replaced(results, "converter", skipped_after("converter"))
    report = build(results)

    lines = result_lines(report, 1.0, tmp_path, tmp_path / "manifest.json")

    assert "spm: fail (exit code 4)" in lines
    assert "converter: skipped (an earlier stage failed)" in lines
    assert "verdict: fail" in lines


def test_the_lines_before_the_paths_hold_no_path_and_no_text_of_a_file(
    tmp_path: Path,
) -> None:
    lines = result_lines(build(), 1.0, tmp_path, tmp_path / "manifest.json")

    assert not any("/" in line for line in lines[:-3])


# ---- the commit protocol of the command (G7.3) -----------------------------------------------------------


def lay_down_an_earlier_run(out: Path) -> None:
    """The files of a finished run: every stage directory, the logs, both documents and its manifest."""
    lay_down_a_run(out)
    store.write(
        out,
        Manifest(
            stage=STAGE, created="x", code_version="x", config_sha256="0" * 64, seed=1
        ),
    )


def test_the_old_manifest_and_the_old_run_are_gone_before_the_first_stage_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    lay_down_an_earlier_run(out)
    seen: list[dict[str, Any]] = []

    def observe(ctx: Any) -> None:
        seen.append(
            {
                "present": sorted(
                    name for name in OWN_ENTRIES if (out / name).exists()
                ),
                "out_dir": ctx.out_dir,
            }
        )

    recording, _ = watch_first_stage(monkeypatch, observe)

    result = invoke(run_args(out))

    assert result.exit_code == 0, result.output
    assert seen == [{"present": [], "out_dir": out}]
    assert recording.stages == list(STAGE_IDS)
    assert sorted(path.name for path in out.iterdir()) == [
        "manifest.json",
        "report.json",
        "report.md",
    ]


def test_the_code_version_is_asked_before_the_output_directory_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    seen: list[bool] = []
    install(monkeypatch)
    monkeypatch.setattr(
        tiny_output, "code_version", lambda: seen.append(out.exists()) or "abc1234"
    )

    result = invoke(run_args(out))

    assert result.exit_code == 0, result.output
    assert seen == [False]
    assert manifest_of(out)["code_version"] == "abc1234"


def test_the_manifest_is_written_after_every_stage_and_after_both_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    recording, _ = install(monkeypatch)
    events: list[str] = []
    real_documents = tiny_output.write_documents
    real_manifest = tiny_output.write_manifest

    def documents(*args: Any, **keywords: Any) -> Any:
        events.append(f"documents after {len(recording.stages)} stages")
        found = real_documents(*args, **keywords)
        events.append("documents written")
        return found

    def manifest(directory: Path, *args: Any, **keywords: Any) -> Any:
        events.append(
            f"manifest with {sorted(path.name for path in directory.iterdir())}"
        )
        return real_manifest(directory, *args, **keywords)

    monkeypatch.setattr(tiny_output, "write_documents", documents)
    monkeypatch.setattr(tiny_output, "write_manifest", manifest)

    result = invoke(run_args(out))

    assert result.exit_code == 0, result.output
    assert events == [
        "documents after 15 stages",
        "documents written",
        "manifest with ['report.json', 'report.md']",
    ]


def test_the_manifest_lists_the_checkpoint_files_the_run_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    def write_a_model(ctx: Any) -> None:
        path = out / "04_spm" / "tiny.model"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"a model")

    watch_first_stage(monkeypatch, write_a_model)

    result = invoke(run_args(out))

    assert result.exit_code == 0, result.output
    listed = [item["path"] for item in manifest_of(out)["outputs"]]
    assert listed == [
        str(out / "report.json"),
        str(out / "report.md"),
        str(out / "04_spm" / "tiny.model"),
    ]
