"""`rinchen tiny run` over fakes (D15 Task 7, brief G7.2, G7.5 and G7.6, spec P13 and P14): every option and
every exit code, the dry run and the scale-up.

The three seams of the command are replaced (`tests/support/tiny_cli.py`): the table of the fifteen stage
functions (each a fake that says what it does and for how many seconds of a fake clock), the runner and the
clock, and the code version of the manifest is a constant. The files the command reads (`uv.lock`, the
profile file, the fixtures, the pipeline file) are the repository's own, or copies a test may change
(`make_workspace`); nothing is written outside `tmp_path`.

Exit codes: 0 a pass; 4 a stage failed or a threshold was missed (everything is written and printed); 2 a
bad option, profile, config or seed, a changed input and an output directory that holds entries the command
does not write; 3 a missing fixture, a fixture that is not its pin and the manifest of another stage in the
output directory; 1 an output that cannot be written and a report that is not a valid report.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Final

import pytest
from typer.testing import CliRunner

from rinchen.cli import tiny, tiny_output
from rinchen.cli.main import app
from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import STAGE_IDS
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.tiny import pipeline
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import (
    Outcome,
    Recording,
    Workspace,
    append_a_line,
    changing_after_hashing,
    entries_of,
    go,
    identity,
    install,
    invoke,
    lines_of,
    make_workspace,
    manifest_of,
    report_of,
    run_args,
    watch_first_stage,
)
from tests.support.tiny_pipeline_runs import make_plan, real_table
from tests.support.tiny_profiles import control_profile
from tests.support.tiny_run_eval import EvalRunner
from tests.support.tiny_stage_inputs import shipped_config

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    return make_workspace(tmp_path / "inputs", monkeypatch)


# ---- the exit codes 0 and 4 ---------------------------------------------------------------------------------


def test_a_pass_exits_0_and_writes_the_two_documents_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording, _ = go(monkeypatch, out)

    assert result.exit_code == 0, result.output
    assert entries_of(out) == ["manifest.json", "report.json", "report.md"]
    assert recording.stages == list(STAGE_IDS)
    assert report_of(out)["verdict"] == "pass"
    assert manifest_of(out)["stage"] == "tiny_run"


def test_a_failed_stage_exits_4_and_still_writes_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording, _ = go(monkeypatch, out, outcomes={"spm": Outcome("fail")})

    assert result.exit_code == 4, result.output
    assert entries_of(out) == ["manifest.json", "report.json", "report.md"]
    assert recording.stages == list(STAGE_IDS[:5])
    assert report_of(out)["verdict"] == "fail"
    assert manifest_of(out)["status"]["spm"] == "fail"
    assert manifest_of(out)["status"]["converter"] == "skipped"
    assert "spm: fail (exit code 4)" in lines_of(result)
    assert "verdict: fail" in lines_of(result)


def test_a_threshold_that_is_missed_exits_4_although_no_stage_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, _, _ = go(monkeypatch, out, outcomes={"parity": Outcome("no_facts")})

    assert result.exit_code == 4
    assert not [line for line in lines_of(result)[:15] if ": fail" in line]
    thresholds = {item["id"]: item["verdict"] for item in report_of(out)["thresholds"]}
    assert thresholds["parity_exact"] == "fail"
    assert thresholds["stages"] == "pass"


def test_a_run_that_spends_its_budget_exits_4_with_the_stage_failed_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    outcomes = {stage: Outcome("ok", 6.0) for stage in STAGE_IDS}

    result, recording, _ = go(
        monkeypatch, out, "--budget-seconds", "10", outcomes=outcomes
    )

    assert result.exit_code == 4
    assert recording.stages == ["environment", "fetch"]
    assert "corpus: fail (the wall-time budget is spent)" in lines_of(result)
    assert report_of(out)["thresholds"][-1]["verdict"] == "fail"
    assert manifest_of(out)["metrics"]["seconds.fetch"] == 6.0
    assert manifest_of(out)["metrics"]["seconds.corpus"] == 0.0


def test_the_lines_are_the_stages_the_thresholds_the_verdict_and_the_three_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, _, _ = go(monkeypatch, out)

    lines = lines_of(result)
    assert lines[:15] == [f"{stage}: ok" for stage in STAGE_IDS]
    assert [line.split(":")[0] for line in lines[15:23]] == [
        f"threshold {item['id']}" for item in report_of(out)["thresholds"]
    ]
    assert lines[23] == "verdict: pass"
    assert lines[24:] == [
        f"report: {out / 'report.json'}",
        f"summary: {out / 'report.md'}",
        f"manifest: {out / 'manifest.json'}",
    ]
    assert result.stderr == ""


def test_a_relative_out_dir_is_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result, _, _ = go(monkeypatch, Path("artifacts") / "tiny")

    assert result.exit_code == 0, result.output
    assert entries_of(tmp_path / "artifacts" / "tiny") == [
        "manifest.json",
        "report.json",
        "report.md",
    ]
    assert f"report: {tmp_path / 'artifacts' / 'tiny' / 'report.json'}" in lines_of(
        result
    )
    assert manifest_of(tmp_path / "artifacts" / "tiny")["outputs"][0]["path"] == str(
        tmp_path / "artifacts" / "tiny" / "report.json"
    )


def test_the_default_out_dir_is_artifacts_tiny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    install(monkeypatch)

    result = invoke(["tiny", "run"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "artifacts" / "tiny" / "report.json").is_file()


def test_a_run_into_a_directory_of_an_earlier_run_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    go(monkeypatch, out)
    (out / "07_train").mkdir()
    (out / "07_train" / "stale.txt").write_text("stale", encoding="utf-8")
    (out / "logs").mkdir()
    (out / "logs" / "old-raised.log").write_text("stale", encoding="utf-8")
    first = manifest_of(out)

    result, _, _ = go(monkeypatch, out)

    assert result.exit_code == 0, result.output
    assert entries_of(out) == ["manifest.json", "report.json", "report.md"]
    assert manifest_of(out)["created"] >= first["created"]


# ---- the options -----------------------------------------------------------------------------------------


def test_a_relative_config_path_is_made_absolute_for_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = control_profile(tmp_path, budget_seconds=77)
    monkeypatch.chdir(tmp_path)

    result, _, _ = go(monkeypatch, tmp_path / "out", config=Path(config.name))

    assert result.exit_code == 0, result.output
    assert manifest_of(tmp_path / "out")["inputs"][0]["path"] == str(config)


def test_the_stages_get_the_runner_of_the_seam_and_the_out_dir_of_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = object()
    recording, _ = install(monkeypatch, runner=runner)

    result = invoke(run_args(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    assert recording.runners == [runner]
    assert all(ctx.runner is runner for ctx in recording.contexts)
    assert {ctx.out_dir for ctx in recording.contexts} == {tmp_path / "out"}


def test_the_seed_is_the_one_of_the_profile_file_and_the_lock_is_given_to_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, recording, _ = go(monkeypatch, tmp_path / "out")

    assert result.exit_code == 0
    assert recording.seeds == [1234]
    assert recording.lock_bytes == [(REPO_ROOT / "uv.lock").read_bytes()]


def test_the_profile_selects_the_file_of_the_seed_and_is_data_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    workspace.profile.with_name("other.yaml").write_text("seed: 5\n", encoding="utf-8")
    out = tmp_path / "out"

    result, recording, _ = go(
        monkeypatch, out, "--profile", "other", config=workspace.config
    )

    assert result.exit_code == 0, result.output
    assert recording.seeds == [5]
    assert report_of(out)["profile"] == "other"
    assert manifest_of(out)["seed"] == 5


def test_a_config_option_replaces_the_pipeline_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = control_profile(tmp_path, budget_seconds=77)

    result, recording, _ = go(monkeypatch, tmp_path / "out", config=config)

    assert result.exit_code == 0, result.output
    assert recording.contexts[0].config.budget_seconds == 77
    assert recording.remaining[0] == 77.0
    assert (
        manifest_of(tmp_path / "out")["config_sha256"]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("flag", "profile_says", "wanted"),
    [
        (None, True, True),
        (None, False, False),
        ("--qe-fake", False, True),
        ("--no-qe-fake", True, False),
    ],
    ids=["profile-true", "profile-false", "flag-on", "flag-off"],
)
def test_the_stand_in_switch_defaults_to_the_profile_and_the_flags_override_it(
    flag: str | None,
    profile_says: bool,
    wanted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = control_profile(tmp_path, qe_fake=profile_says)
    extra = [] if flag is None else [flag]
    out = tmp_path / "out"

    result, recording, _ = go(monkeypatch, out, *extra, config=config)

    assert result.exit_code == 0, result.output
    assert recording.contexts[0].qe_fake is wanted
    assert report_of(out)["qe_fake"] is wanted


def test_the_threads_default_to_the_profiles_and_the_option_overrides_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plain, _ = go(monkeypatch, tmp_path / "one")
    _, given, _ = go(monkeypatch, tmp_path / "two", "--threads", "5")

    assert plain.contexts[0].threads == shipped_config().threads == 2
    assert given.contexts[0].threads == 5


def test_the_budget_defaults_to_the_profiles_and_the_option_is_the_bound_of_the_run_and_of_the_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plain, _ = go(monkeypatch, tmp_path / "one")
    result, given, _ = go(monkeypatch, tmp_path / "two", "--budget-seconds", "900")

    assert plain.remaining[0] == 1200.0
    assert given.remaining[0] == 900.0
    assert given.contexts[0].config.budget_seconds == 900
    assert report_of(tmp_path / "one")["thresholds"][-1]["bound"] == "<= 1200"
    assert report_of(tmp_path / "two")["thresholds"][-1]["bound"] == "<= 900"
    assert result.exit_code == 0


def test_the_command_and_its_table_run_with_the_device_hidden_and_the_variable_is_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen: list[str | None] = []
    recording, clock = install(monkeypatch)
    inner = tiny.make_stage_functions

    def table(seed: int, lock_bytes: bytes) -> Any:
        seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return inner(seed, lock_bytes)

    monkeypatch.setattr(tiny, "make_stage_functions", table)

    result = invoke(run_args(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    assert seen == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert recording.stages == list(STAGE_IDS)
    assert clock.now > 1000.0


# ---- the exit code 2 ----------------------------------------------------------------------------------------


def refused(
    monkeypatch: pytest.MonkeyPatch, out: Path, *extra: str, config: Path | None = None
) -> tuple[Any, Recording]:
    result, recording, _ = go(monkeypatch, out, *extra, config=config)
    return result, recording


@pytest.mark.parametrize(
    "profile", ["Full", "FULL", "a/b", "x" * 17, "1x", "tiny.yaml"]
)
def test_a_profile_outside_the_pattern_is_exit_2_and_nothing_is_written(
    profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording = refused(monkeypatch, out, "--profile", profile)

    assert result.exit_code == 2
    assert not out.exists()
    assert recording.stages == []
    assert "profile name" in result.stderr


@pytest.mark.parametrize("profile", ["full", "another"])
def test_a_profile_with_no_pipeline_file_is_exit_2_with_the_message_of_the_spec(
    profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording = refused(monkeypatch, out, "--profile", profile)

    assert result.exit_code == 2
    assert f"no pipeline configuration for the profile '{profile}'" in result.stderr
    assert f"(configs/tiny/{profile}.yaml)" in result.stderr
    assert (
        "https://github.com/btseee/rinchen/wiki/Runbook-15-Tiny-Pipeline"
        in result.stderr
    )
    assert "docs/" not in result.stderr
    assert not out.exists()
    assert recording.stages == []
    assert result.stdout == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--no-such-option"],
        ["--threads", "0"],
        ["--threads", "-1"],
        ["--threads", "65"],
        ["--threads", "many"],
        ["--budget-seconds", "0"],
        ["--budget-seconds", "86401"],
        ["--budget-seconds", "soon"],
        ["--qe-fake", "--no-qe-fake-at-all"],
        ["--dev" + "ice", "gpu"],
    ],
    ids=lambda arguments: " ".join(arguments),
)
def test_a_usage_error_is_exit_2_and_nothing_is_written(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording = refused(monkeypatch, out, *arguments)

    assert result.exit_code == 2
    assert not out.exists()
    assert recording.stages == []


def test_a_config_that_does_not_exist_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = refused(monkeypatch, tmp_path / "out", config=tmp_path / "nope.yaml")

    assert result.exit_code == 2
    assert "tiny config" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_config_that_is_not_valid_is_exit_2_and_says_which_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = control_profile(tmp_path, validate=False, threads=0)

    result, recording = refused(monkeypatch, tmp_path / "out", config=config)

    assert result.exit_code == 2
    assert "threads" in result.stderr
    assert recording.stages == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["src", "x.part", "notes.txt"])
def test_an_output_directory_that_holds_an_entry_the_command_does_not_write_is_exit_2(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text("x", encoding="utf-8")
    (out / "report.json").write_text("an earlier report", encoding="utf-8")
    before = identity(out)

    result, recording = refused(monkeypatch, out)

    assert result.exit_code == 2
    assert "holds 1 entry" in result.stderr
    assert name not in result.stderr
    assert ("delete it" in result.stderr) == name.endswith(".part")
    assert identity(out) == before
    assert recording.stages == []


@pytest.mark.parametrize(
    "text",
    ["seed: true\n", "seed: '1'\n", "paths: {}\n", "seed: [1]\n", "seed: -3\n"],
    ids=["bool", "text", "missing", "list", "negative"],
)
def test_a_seed_that_is_not_a_non_negative_integer_is_exit_2(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    workspace.profile.write_text(text, encoding="utf-8")

    result, recording = refused(monkeypatch, tmp_path / "out", config=workspace.config)

    assert result.exit_code == 2
    assert "seed" in result.stderr
    assert recording.stages == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("target", ["config", "fixture", "profile", "lock"])
def test_an_input_that_changed_between_hashing_and_reading_is_exit_2_and_is_not_read(
    target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    path = {
        "config": workspace.config,
        "fixture": workspace.fixtures / "noise.tsv",
        "profile": workspace.profile,
        "lock": workspace.lock,
    }[target]
    recording, _ = install(monkeypatch)
    changing_after_hashing(monkeypatch, path, append_a_line)
    out = tmp_path / "out"

    result = invoke(run_args(out, config=workspace.config))

    assert result.exit_code == 2, result.output
    assert "changed" in result.stderr
    assert recording.stages == []
    assert recording.seeds == []
    assert not out.exists()


# ---- the exit code 3 ----------------------------------------------------------------------------------------


def test_a_fixture_whose_digest_differs_from_its_pin_is_exit_3_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    path = workspace.fixtures / "eng_Latn-khk_Cyrl.tsv"
    path.write_bytes(path.read_bytes() + b"x")
    out = tmp_path / "out"

    result, recording = refused(monkeypatch, out, config=workspace.config)

    assert result.exit_code == 3
    assert "eng_Latn-khk_Cyrl.tsv" in result.stderr
    assert not out.exists()
    assert recording.stages == []


def test_a_missing_fixture_is_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    (workspace.fixtures / "mono_khk_Cyrl.tsv").unlink()

    result, _ = refused(monkeypatch, tmp_path / "out", config=workspace.config)

    assert result.exit_code == 3
    assert "mono_khk_Cyrl.tsv" in result.stderr


def test_a_lock_that_is_missing_or_not_a_lock_is_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    workspace.lock.write_text("not = = toml\n", encoding="utf-8")
    broken, _ = refused(monkeypatch, tmp_path / "one", config=workspace.config)
    workspace.lock.unlink()

    missing, _ = refused(monkeypatch, tmp_path / "two", config=workspace.config)

    assert (broken.exit_code, missing.exit_code) == (3, 3)
    assert "uv.lock" in broken.stderr
    assert not (tmp_path / "one").exists()


def test_the_manifest_of_another_stage_is_exit_3_and_nothing_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    store.write(
        out,
        Manifest(
            stage="spm_train",
            created="x",
            code_version="x",
            config_sha256="0" * 64,
            seed=1,
        ),
    )
    (out / "report.json").write_text("an earlier report", encoding="utf-8")
    (out / "01_fetch").mkdir()
    (out / "01_fetch" / "a.tsv").write_text("x", encoding="utf-8")
    before = identity(out)

    result, recording = refused(monkeypatch, out)

    assert result.exit_code == 3
    assert "spm_train" in result.stderr
    assert identity(out) == before
    assert recording.stages == []


@pytest.mark.parametrize(
    "refusal",
    ["profile", "foreign entry", "usage", "fixture", "seed"],
)
def test_a_refusal_leaves_the_files_of_an_earlier_run_byte_identical(
    refusal: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    out = tmp_path / "out"
    first, _, _ = go(monkeypatch, out)
    assert first.exit_code == 0
    (out / "01_fetch").mkdir()
    (out / "01_fetch" / "kept.txt").write_text("from the earlier run", encoding="utf-8")
    arguments: list[str] = []
    config: Path | None = None
    if refusal == "profile":
        arguments = ["--profile", "full"]
    elif refusal == "foreign entry":
        (out / "src").mkdir()
    elif refusal == "usage":
        arguments = ["--threads", "0"]
    elif refusal == "fixture":
        path = workspace.fixtures / "nouns.tsv"
        path.write_bytes(path.read_bytes() + b"x")
        config = workspace.config
    else:
        workspace.profile.write_text("seed: nope\n", encoding="utf-8")
        config = workspace.config
    before = identity(out)

    result, recording = refused(monkeypatch, out, *arguments, config=config)

    assert result.exit_code in (2, 3)
    assert identity(out) == before
    assert recording.stages == []


def test_the_refusals_happen_before_the_commit_begins_and_the_code_version_is_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    calls: list[str] = []
    install(monkeypatch)
    monkeypatch.setattr(
        tiny_output, "code_version", lambda: calls.append("version") or "x"
    )
    monkeypatch.setattr(
        tiny_output, "begin_commit", lambda *args: calls.append("begin")
    )
    (workspace.fixtures / "nouns.tsv").write_bytes(b"changed")
    out = tmp_path / "out"
    out.mkdir()
    (out / "src").mkdir()

    changed = invoke(run_args(out, config=workspace.config))
    (workspace.fixtures / "nouns.tsv").unlink()
    missing = invoke(run_args(out, config=workspace.config))
    foreign = invoke(run_args(out))
    other = tmp_path / "other"
    store.write(
        other,
        Manifest(
            stage="spm_train",
            created="x",
            code_version="x",
            config_sha256="0" * 64,
            seed=1,
        ),
    )
    stranger = invoke(run_args(other))

    assert (changed.exit_code, missing.exit_code, foreign.exit_code) == (3, 3, 2)
    assert stranger.exit_code == 3
    assert calls == []


# ---- the exit code 1 ----------------------------------------------------------------------------------------


def test_an_output_directory_that_cannot_be_made_is_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the directory should go", encoding="utf-8")

    result, recording, _ = go(monkeypatch, blocked / "out")

    assert result.exit_code == 1
    assert "cannot write" in result.stderr
    assert recording.stages == []


@pytest.mark.parametrize("failing", ["write_documents", "write_manifest", "begin"])
def test_a_file_that_cannot_be_written_or_removed_is_exit_1_and_no_manifest_is_left(
    failing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    install(monkeypatch)

    def refuse(*args: Any, **keywords: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(tiny_output, failing, refuse)

    result = invoke(run_args(out))

    assert result.exit_code == 1
    assert "cannot write" in result.stderr
    assert not (out / "manifest.json").exists()


DEFECT_WORDS: Final = (
    "the stages that ran wrote their directories, which remain",
    "report.json, report.md and manifest.json were not written",
    "the files of an earlier run were removed when this run began",
)


def test_a_report_that_is_not_a_valid_report_is_a_defect_exit_1_and_no_document_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    def leave_a_directory(ctx: Any) -> None:
        (ctx.out_dir / "01_fetch").mkdir(parents=True)

    watch_first_stage(monkeypatch, leave_a_directory)

    def invalid(*args: Any, **keywords: Any) -> Any:
        raise ValueError("secret-text-of-a-fixture")

    monkeypatch.setattr(tiny, "build_report", invalid)

    result = invoke(run_args(out))

    message = " ".join(result.stderr.split())
    assert result.exit_code == 1
    assert "defect" in message
    assert all(words in message for words in DEFECT_WORDS)
    assert "nothing of the run was written" not in message
    assert "secret-text-of-a-fixture" not in result.output
    assert (out / "01_fetch").is_dir()
    assert not (out / "report.json").exists()
    assert not (out / "report.md").exists()
    assert not (out / "manifest.json").exists()


def test_a_markdown_that_does_not_survive_a_round_trip_is_a_defect_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    install(monkeypatch)
    calls: list[int] = []
    real = tiny_output.render

    def drifting(report: Any) -> str:
        calls.append(1)
        return real(report) + ("x" if len(calls) > 1 else "")

    monkeypatch.setattr(tiny_output, "render", drifting)

    result = invoke(run_args(out))

    message = " ".join(result.stderr.split())
    assert result.exit_code == 1
    assert "defect" in message
    assert all(words in message for words in DEFECT_WORDS)
    assert not (out / "report.json").exists()
    assert not (out / "manifest.json").exists()


# ---- the dry run (G7.5) ----------------------------------------------------------------------------------------


def test_a_dry_run_prints_the_fifteen_stages_with_the_commands_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"

    result, recording, _ = go(monkeypatch, out, "--dry-run")

    assert result.exit_code == 0, result.output
    lines = lines_of(result)
    assert [line.split(":")[0] for line in lines if not line.startswith(" ")] == list(
        STAGE_IDS
    )
    assert lines == pipeline.plan_lines(make_plan(out))
    assert not out.exists()
    assert recording.stages == []


def test_a_dry_run_writes_nothing_into_an_existing_directory_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    first, _, _ = go(monkeypatch, out)
    before = identity(out)
    assert first.exit_code == 0

    result, _, _ = go(monkeypatch, out, "--dry-run")

    assert result.exit_code == 0
    assert identity(out) == before


def test_a_dry_run_refuses_a_changed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Workspace
) -> None:
    path = workspace.fixtures / "noise.tsv"
    path.write_bytes(path.read_bytes() + b"x")

    result, recording, _ = go(
        monkeypatch, tmp_path / "out", "--dry-run", config=workspace.config
    )

    assert result.exit_code == 3
    assert result.stdout == ""
    assert recording.stages == []


def test_a_dry_run_also_refuses_a_directory_the_run_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "src").mkdir()

    result, _, _ = go(monkeypatch, out, "--dry-run")

    assert result.exit_code == 2
    assert result.stdout == ""


def test_a_dry_run_never_asks_the_code_version_or_starts_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    install(monkeypatch)
    monkeypatch.setattr(
        tiny_output, "code_version", lambda: calls.append("version") or "x"
    )
    monkeypatch.setattr(
        tiny_output, "begin_commit", lambda *args: calls.append("begin")
    )

    result = invoke(run_args(tmp_path / "out", "--dry-run"))

    assert result.exit_code == 0
    assert calls == []


# ---- scale-up is a change of the profile (G7.6) ---------------------------------------------------------


def run_real_stages(
    monkeypatch: pytest.MonkeyPatch, out: Path, config: Path | None
) -> tuple[Any, list[str]]:
    """The command over the real stage functions (the environment and the service are fakes), the runner
    that plays every command, and the function each stage ran."""
    used: list[str] = []

    def table(seed: int, lock_bytes: bytes) -> Any:
        functions = real_table(seed, lock_bytes)
        for stage_id, function in functions.items():
            used.append(f"{stage_id}:{getattr(function, 'func', function).__name__}")
        return functions

    install(monkeypatch, runner=EvalRunner(), table=table)
    return invoke(run_args(out, config=config)), used


def test_a_profile_with_other_steps_runs_the_same_stage_functions_and_records_its_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = control_profile(tmp_path, train_steps=400)
    default_out = tmp_path / "default"
    control_out = tmp_path / "control"

    default, default_used = run_real_stages(monkeypatch, default_out, None)
    scaled, scaled_used = run_real_stages(monkeypatch, control_out, control)

    assert default.exit_code == 0, default.output
    assert scaled.exit_code == 0, scaled.output
    assert manifest_of(default_out)["metrics"]["train.steps"] == 1500.0
    assert manifest_of(control_out)["metrics"]["train.steps"] == 400.0
    assert default_used == scaled_used
    assert len(default_used) == 15
    assert (
        report_of(default_out)["profile"] == report_of(control_out)["profile"] == "tiny"
    )
    assert manifest_of(control_out)["inputs"][0]["path"] == str(control)


# ---- the help --------------------------------------------------------------------------------------------


def test_the_help_lists_every_option_and_the_exit_codes_and_offers_no_device() -> None:
    result = CliRunner().invoke(app, ["tiny", "run", "--help"], terminal_width=200)

    assert result.exit_code == 0
    for option in (
        "--profile",
        "--out-dir",
        "--config",
        "--qe-fake",
        "--no-qe-fake",
        "--threads",
        "--budget-seconds",
        "--dry-run",
    ):
        assert option in result.output, option
    for word in ("Exit", "codes", "130", "143", "129"):
        assert word in result.output, word
    assert "--dev" + "ice" not in result.output


def test_the_group_lists_the_run_command() -> None:
    result = CliRunner().invoke(app, ["tiny", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
