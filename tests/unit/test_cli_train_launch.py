"""`rinchen train launch`: its options, the facts it prints, the manifest it writes last,
the order of its checks and writes, and the exit code of each failure.

Eole never starts. The CLI runs the real launcher (`rinchen.train.launch`), and the launcher's
`run_logged` is replaced by `tests.support.launch_cli.LinkingFakeEole`, which writes the logs
and step directories a run leaves, links the newest step as Eole does and fails the way
`run_logged` does when a script says so. The rules of the launcher have their own tests
(`test_train_launch.py`, `test_train_launch_lock.py`); here is what the command makes of them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import train_common, train_runs
from rinchen.cli.main import app
from rinchen.errors import ConfigError, DataError, RinchenError, UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.fake_launch import Script
from tests.support.launch_cli import (
    LOCK_HOLDER,
    LaunchWorkspace,
    LinkingFakeEole,
    assert_one_error,
    install_fake_eole,
    make_launch_workspace,
    snapshot,
)

runner = CliRunner()

OPTIONS = [
    "--config",
    "--run-dir",
    "--world-size",
    "--gpu-ranks",
    "--threads",
    "--timeout",
    "--cuda-visible-devices",
    "--write-only",
]
TOOL_PACKAGES = ["eole", "torch", "safetensors", "sentencepiece"]
STEP_FILES = ["model.00.safetensors", "optimizer.pt", "config.json", "vocab.json"]
FAILED = "eole train failed: exit code 1; log: launch/attempt-1.log"


@pytest.fixture
def workspace(tmp_path: Path) -> LaunchWorkspace:
    return make_launch_workspace(tmp_path)


@pytest.fixture
def fake(
    monkeypatch: pytest.MonkeyPatch, workspace: LaunchWorkspace
) -> LinkingFakeEole:
    return install_fake_eole(monkeypatch, workspace)


def run_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def manifest_of(workspace: LaunchWorkspace) -> Manifest:
    manifest = store.read(workspace.run_dir)
    assert manifest is not None
    return manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def other_config(workspace: LaunchWorkspace, name: str, **training: object) -> Path:
    """A copy of the workspace's config with keys of `training` changed."""
    document = yaml.safe_load(workspace.config.read_text(encoding="utf-8"))
    document["training"].update(training)
    path = workspace.root / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def crash(step: int) -> Script:
    """A run killed after it reached `step`: it saved nothing there."""
    return Script(to_step=step, skip_final_save=True, error=UpstreamError(FAILED))


def stdout_of(workspace: LaunchWorkspace, *extra: str) -> list[str]:
    return run_ok(workspace.args(*extra)).stdout.splitlines()


# --- help and options --------------------------------------------------------------


def test_the_train_group_lists_every_command() -> None:
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    for command in ("ablate", "launch", "average"):
        assert command in result.stdout


def test_launch_help_names_every_option() -> None:
    result = runner.invoke(app, ["train", "launch", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout


@pytest.mark.parametrize("option", ["--config", "--run-dir"])
def test_a_required_option_that_is_missing_is_a_usage_error(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, option: str
) -> None:
    args = workspace.args()
    index = args.index(option)
    del args[index : index + 2]

    result = runner.invoke(app, args)

    assert result.exit_code == 2, result.output
    assert f"Missing option '{option}'" in result.stderr
    assert not workspace.run_dir.exists()
    assert fake.calls == []


def test_the_defaults_are_two_threads_and_no_timeout(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    assert [(call.env["OMP_NUM_THREADS"], call.timeout) for call in fake.calls] == [
        ("2", None)
    ] * 2


def test_the_threads_and_the_timeout_reach_both_eole_commands(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args("--threads", "3", "--timeout", "12.5"))

    assert [
        (
            call.subcommand,
            call.env["OMP_NUM_THREADS"],
            call.env["MKL_NUM_THREADS"],
            call.timeout,
        )
        for call in fake.calls
    ] == [("build_vocab", "3", "3", 12.5), ("train", "3", "3", 12.5)]


@pytest.mark.parametrize(
    ("option", "value"), [("--threads", "0"), ("--world-size", "0")]
)
def test_a_count_below_one_is_a_usage_error(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, option: str, value: str
) -> None:
    result = runner.invoke(app, workspace.args(option, value))

    assert result.exit_code == 2, result.output
    assert option in result.stderr
    assert not workspace.run_dir.exists()
    assert fake.calls == []


@pytest.mark.parametrize("timeout", ["0", "-5", "nan", "inf"])
def test_a_timeout_that_is_not_a_positive_number_is_a_config_error(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, timeout: str
) -> None:
    result = runner.invoke(app, workspace.args("--timeout", timeout))

    assert_one_error(result, 2, "--timeout must be a positive, finite number")
    assert not workspace.run_dir.exists()
    assert fake.calls == []


# --- --gpu-ranks, --world-size and --cuda-visible-devices ---------------------------


@pytest.mark.parametrize(
    ("text", "ranks"),
    [
        ("0,1", [0, 1]),
        ("0,1,2,3", [0, 1, 2, 3]),
        (" 0 , 1 ,2", [0, 1, 2]),
        ("1,0", [1, 0]),
        ("0", [0]),
        ("00,01", [0, 1]),
    ],
)
def test_ranks_are_comma_separated_numbers_with_spaces_tolerated(
    tmp_path: Path, text: str, ranks: list[int]
) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)

    result = runner.invoke(
        app,
        workspace.args(
            "--world-size", str(len(ranks)), "--gpu-ranks", text, "--write-only"
        ),
    )

    assert result.exit_code == 0, result.output
    record = json.loads(workspace.record.read_text(encoding="utf-8"))
    assert (record["world_size"], record["gpu_ranks"]) == (len(ranks), ranks)


NOT_A_NUMBER = "is not a whole number"


@pytest.mark.parametrize(
    ("text", "why"),
    [
        pytest.param("", "an item is empty", id="empty"),
        pytest.param(" ", "an item is empty", id="a space"),
        pytest.param("0,,1", "an item is empty", id="a gap"),
        pytest.param("0,", "an item is empty", id="a trailing comma"),
        pytest.param(",0", "an item is empty", id="a leading comma"),
        pytest.param(",", "an item is empty", id="a comma"),
        pytest.param("-1", "-1 is negative", id="negative"),
        pytest.param("0,-1", "-1 is negative", id="negative second"),
        pytest.param("1,1", "[1] is repeated", id="repeated"),
        pytest.param("0,1,0", "[0] is repeated", id="repeated first"),
        pytest.param("a", NOT_A_NUMBER, id="letter"),
        pytest.param("0,a", NOT_A_NUMBER, id="letter second"),
        pytest.param("1.5", NOT_A_NUMBER, id="decimal point"),
        pytest.param("0 1", NOT_A_NUMBER, id="space inside"),
        pytest.param("+1", NOT_A_NUMBER, id="plus sign"),
        pytest.param("0x1", NOT_A_NUMBER, id="hexadecimal"),
        pytest.param("1e1", NOT_A_NUMBER, id="exponent"),
        pytest.param(chr(0x663), NOT_A_NUMBER, id="arabic-indic digit"),
        pytest.param("9" * 5000, "too many digits", id="5000 digits"),
    ],
)
def test_a_list_of_ranks_that_is_not_one_is_a_config_error_before_anything_happens(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, text: str, why: str
) -> None:
    result = runner.invoke(app, workspace.args("--gpu-ranks", text))

    assert_one_error(result, 2, "--gpu-ranks must be distinct GPU numbers")
    assert why in result.stderr
    assert not workspace.run_dir.exists()
    assert fake.calls == []


def test_a_gpu_run_needs_ranks_and_a_world_size_that_agree(
    tmp_path: Path, fake: LinkingFakeEole
) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)

    only_world = runner.invoke(app, workspace.args("--world-size", "2"))
    only_ranks = runner.invoke(app, workspace.args("--gpu-ranks", "0,1"))

    assert_one_error(only_world, 2, "invalid Eole config: training.gpu_ranks")
    assert_one_error(only_ranks, 2, "invalid Eole config: training.world_size")
    assert not workspace.run_dir.exists()
    assert fake.calls == []


def test_world_size_and_ranks_are_laid_over_the_config_and_recorded(
    tmp_path: Path, fake: LinkingFakeEole
) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)

    run_ok(workspace.args("--world-size", "2", "--gpu-ranks", "0,1"))

    rendered = yaml.safe_load(
        (workspace.run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
    )
    assert (rendered["training"]["world_size"], rendered["training"]["gpu_ranks"]) == (
        2,
        [0, 1],
    )
    record = json.loads(workspace.record.read_text(encoding="utf-8"))
    assert (record["world_size"], record["gpu_ranks"]) == (2, [0, 1])
    # A GPU run leaves the visible devices to the environment.
    assert fake.trains[0].env.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )


def test_the_visible_devices_are_exported_by_command_sh(tmp_path: Path) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)

    run_ok(
        workspace.args(
            "--world-size",
            "2",
            "--gpu-ranks",
            "0,1",
            "--cuda-visible-devices",
            "2,3",
            "--write-only",
        )
    )

    assert "export CUDA_VISIBLE_DEVICES=2,3\n" in workspace.command_sh.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("devices", ["", "a", "0, 1", "0,", ",0", "-1", "0;1"])
def test_visible_devices_that_are_not_numbers_with_commas_are_a_config_error(
    tmp_path: Path, fake: LinkingFakeEole, devices: str
) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)

    result = runner.invoke(
        app,
        workspace.args(
            "--world-size", "2", "--gpu-ranks", "0,1", "--cuda-visible-devices", devices
        ),
    )

    assert_one_error(result, 2, "cuda_visible_devices must be device numbers")
    assert not workspace.run_dir.exists()
    assert fake.calls == []


@pytest.mark.parametrize(
    ("ranks", "devices", "why"),
    [
        ("0,1", "0", "names 1 device(s) for 2 ranks"),
        (None, "0", "uses no GPU"),
    ],
)
def test_visible_devices_that_do_not_fit_the_run_are_a_config_error(
    tmp_path: Path,
    fake: LinkingFakeEole,
    ranks: str | None,
    devices: str,
    why: str,
) -> None:
    workspace = make_launch_workspace(tmp_path, timeout=3600)
    extra = ["--world-size", "2", "--gpu-ranks", ranks] if ranks else []

    result = runner.invoke(
        app, workspace.args(*extra, "--cuda-visible-devices", devices)
    )

    assert_one_error(result, 2, "cuda_visible_devices")
    assert why in result.stderr
    assert fake.calls == []


# --- what it prints ------------------------------------------------------------------


def test_a_fresh_launch_prints_one_fact_per_line(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    assert stdout_of(workspace) == [
        "status: completed",
        "steps: 200",
        "resumed from: none",
        "attempt: 1",
        f"checkpoint: {workspace.step_dir(200)}",
        f"log: {workspace.attempt_log(1)}",
        f"command.sh: {workspace.command_sh}",
        f"manifest: {workspace.manifest}",
    ]


def test_a_resumed_launch_says_where_it_resumed_from(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(crash(120))
    assert runner.invoke(app, workspace.args()).exit_code == 4

    assert stdout_of(workspace) == [
        "status: completed",
        "steps: 200",
        "resumed from: 100",
        "attempt: 2",
        f"checkpoint: {workspace.step_dir(200)}",
        f"log: {workspace.attempt_log(2)}",
        f"command.sh: {workspace.command_sh}",
        f"manifest: {workspace.manifest}",
    ]


def test_the_incomplete_steps_a_resume_ignores_are_listed(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(
        Script(
            to_step=130, skip_final_save=True, cut=(100,), error=UpstreamError(FAILED)
        )
    )
    assert runner.invoke(app, workspace.args()).exit_code == 4

    lines = stdout_of(workspace)

    assert "resumed from: 50" in lines
    assert lines[-2:] == [
        "incomplete step directories skipped: [100]",
        f"manifest: {workspace.manifest}",
    ]
    assert not any(
        line.startswith("incomplete") for line in stdout_of(workspace, "--write-only")
    )


def test_a_finished_run_prints_already_complete_and_runs_nothing(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    calls = len(fake.calls)

    assert stdout_of(workspace) == [
        "status: already_complete",
        "steps: 200",
        "resumed from: none",
        "attempt: 0",
        f"checkpoint: {workspace.step_dir(200)}",
        "log: none",
        "command.sh: none",
        f"manifest: {workspace.manifest}",
    ]
    assert len(fake.calls) == calls


def test_an_early_stop_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_launch_workspace(
        tmp_path, early_stopping=2, early_stopping_criteria="ppl", train_steps=600
    )
    stand_in = install_fake_eole(monkeypatch, workspace)
    stand_in.queue(Script(early_stop_at=150))

    lines = stdout_of(workspace)

    assert lines[:2] == ["status: early_stopped", "steps: 150"]
    manifest = manifest_of(workspace)
    assert manifest.metrics["early_stopped"] == 1.0
    assert manifest.metrics["steps"] == 150.0


def test_write_only_prints_the_path_of_command_sh_and_nothing_else(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    result = run_ok(workspace.args("--write-only"))

    assert result.stdout == f"{workspace.command_sh}\n"
    assert fake.calls == []
    assert workspace.command_sh.is_file()


# --- --write-only --------------------------------------------------------------------


def test_write_only_writes_no_manifest_and_runs_nothing(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args("--write-only"))

    assert not workspace.manifest.exists()
    assert fake.calls == []
    assert workspace.record.is_file()


def test_write_only_leaves_an_existing_manifest_alone(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    before = workspace.manifest.read_bytes()
    longer = other_config(workspace, "longer.yaml", train_steps=300)

    run_ok(workspace.args("--write-only", config=longer))

    assert workspace.manifest.read_bytes() == before
    # The script was written again, for the attempt that would resume the run.
    text = workspace.command_sh.read_text(encoding="utf-8")
    assert "attempt 2" in text
    assert "# It resumes from step 200" in text


def test_write_only_of_a_finished_run_writes_nothing_and_says_so_on_stderr(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    before = snapshot(workspace.run_dir)

    result = run_ok(workspace.args("--write-only"))

    assert result.stdout == ""
    assert "already complete" in result.stderr
    assert "step 200" in result.stderr
    assert snapshot(workspace.run_dir) == before


def test_write_only_applies_the_refusals_of_a_launch(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(crash(120))
    assert runner.invoke(app, workspace.args()).exit_code == 4
    other = other_config(workspace, "other.yaml", learning_rate=0.3)
    before = snapshot(workspace.run_dir)

    result = runner.invoke(app, workspace.args("--write-only", config=other))

    assert_one_error(result, 2, "the config is not the one this run was started with")
    assert snapshot(workspace.run_dir) == before


def test_write_only_is_refused_for_a_foreign_manifest_too(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    write_foreign_manifest(workspace.run_dir)

    result = runner.invoke(app, workspace.args("--write-only"))

    assert_one_error(
        result, 3, f"{workspace.run_dir} holds the manifest of stage 'train_ablate'"
    )
    assert not workspace.command_sh.exists()


# --- the manifest --------------------------------------------------------------------


def write_foreign_manifest(directory: Path) -> None:
    store.write(
        directory,
        Manifest(
            stage="train_ablate",
            created="2026-11-08T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=0,
        ),
    )


def test_the_manifest_names_the_stage_the_config_and_the_seed(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert manifest.stage == "train_launch"
    assert manifest.seed == 1234
    assert manifest.config_sha256 == sha256(workspace.config)
    assert [(item.path, item.sha256, item.bytes) for item in manifest.inputs] == [
        (
            str(workspace.config),
            sha256(workspace.config),
            workspace.config.stat().st_size,
        )
    ]
    assert manifest.status == {"launch": "ok"}
    assert manifest.tool_versions == {
        package: importlib.metadata.version(package) for package in TOOL_PACKAGES
    }


def test_the_manifest_lists_the_four_files_of_the_final_step_and_the_run_records(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    step = workspace.step_dir(200)
    assert [item.path for item in manifest.outputs] == [
        *(str(step / name) for name in STEP_FILES),
        str(workspace.record),
        str(workspace.command_sh),
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path


def test_the_manifest_never_lists_the_links_in_the_checkpoint_directory(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    assert (workspace.model_path / "model.00.safetensors").is_symlink()
    paths = [Path(item.path) for item in manifest_of(workspace).outputs]
    assert not any(path.is_symlink() for path in paths)
    assert not any(path.parent == workspace.model_path for path in paths)


def test_the_manifest_of_a_fresh_run_carries_the_counts_and_the_curve(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    assert manifest_of(workspace).metrics == {
        "steps": 200.0,
        "attempt": 1.0,
        "incomplete_skipped": 0.0,
        "early_stopped": 0.0,
        "xent_first": 5.33,
        "xent_last": 2.17,
        "valid_ppl_first": 50.0,
        "valid_ppl_last": 14.2857,
    }


def test_the_manifest_of_a_resumed_run_says_where_it_resumed_from(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(
        Script(
            to_step=130, skip_final_save=True, cut=(100,), error=UpstreamError(FAILED)
        )
    )
    assert runner.invoke(app, workspace.args()).exit_code == 4

    run_ok(workspace.args())

    metrics = manifest_of(workspace).metrics
    assert metrics["resumed_from"] == 50.0
    assert metrics["attempt"] == 2.0
    assert metrics["incomplete_skipped"] == 1.0
    # The curve is the last attempt's: it starts at the resume.
    assert metrics["xent_first"] == 4.5
    assert metrics["valid_ppl_first"] == 27.2727


def test_a_fresh_run_has_no_resumed_from_metric(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())

    assert "resumed_from" not in manifest_of(workspace).metrics


def test_the_manifest_is_written_after_every_file_it_lists(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[bool], int]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                [Path(item.path).is_file() for item in manifest.outputs],
                sum(1 for path in workspace.run_dir.rglob("*") if path.is_file()),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(workspace.args())

    (((exist, files_before)),) = seen
    assert all(exist) and len(exist) == 6
    # Nothing but the manifest is written after it.
    after = sum(1 for path in workspace.run_dir.rglob("*") if path.is_file())
    assert after == files_before + 1


def test_the_manifest_vouches_for_the_bytes_the_config_had_before_it_was_read(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = sha256(workspace.config)
    real = train_runs.launch_run

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        # The file changes just before the launcher reads it.
        workspace.config.write_text(
            workspace.config.read_text(encoding="utf-8") + "# edited\n",
            encoding="utf-8",
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(train_runs, "launch_run", after_a_change)

    run_ok(workspace.args())

    assert sha256(workspace.config) != before
    manifest = manifest_of(workspace)
    assert [item.sha256 for item in manifest.inputs] == [before]
    assert manifest.config_sha256 == before


def test_a_config_that_cannot_be_read_is_a_data_error_naming_it(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = Path.open

    def open_unless_unreadable(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if self == workspace.config and "b" in mode:
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_unless_unreadable)

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 3, f"cannot read Eole config {workspace.config}: ")
    assert not workspace.run_dir.exists()
    assert fake.calls == []


def test_a_missing_config_is_a_config_error(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    missing = workspace.root / "nowhere.yaml"

    result = runner.invoke(app, workspace.args(config=missing))

    assert_one_error(result, 2, f"Eole config not found: {missing}")
    assert not workspace.run_dir.exists()


# --- the checks and the order of the writes -----------------------------------------


def test_a_foreign_manifest_in_the_run_directory_is_refused_and_kept(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    write_foreign_manifest(workspace.run_dir)
    before = workspace.manifest.read_bytes()

    result = runner.invoke(app, workspace.args())

    assert_one_error(
        result, 3, f"{workspace.run_dir} holds the manifest of stage 'train_ablate'"
    )
    assert workspace.manifest.read_bytes() == before
    assert fake.calls == []
    assert not workspace.command_sh.exists()


def test_a_run_directory_that_holds_the_config_is_refused(
    tmp_path: Path, fake: LinkingFakeEole
) -> None:
    run = tmp_path / "run"
    workspace = make_launch_workspace(tmp_path)
    inside = run / "config.yaml"
    run.mkdir()
    inside.write_text(workspace.config.read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(app, workspace.args(config=inside))

    assert_one_error(
        result, 2, f"--run-dir {run} is the directory of the input file {inside}"
    )
    assert fake.calls == []
    assert not (run / "launch").exists()


def test_a_run_directory_reached_through_a_link_or_dotdot_is_refused_too(
    tmp_path: Path, fake: LinkingFakeEole
) -> None:
    workspace = make_launch_workspace(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(workspace.root, target_is_directory=True)

    for spelling in (alias, workspace.root / "x" / ".."):
        (workspace.root / "x").mkdir(exist_ok=True)
        result = runner.invoke(app, workspace.args(run_dir=spelling))

        assert_one_error(result, 2, f"--run-dir {spelling} is the directory of")
    assert fake.calls == []


def test_a_refused_config_leaves_the_manifest_of_a_finished_run_alone(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    before = snapshot(workspace.run_dir)
    other = other_config(workspace, "other.yaml", learning_rate=0.3)

    result = runner.invoke(app, workspace.args(config=other))

    assert_one_error(result, 2, "the config is not the one this run was started with")
    # The launch took the lock before it looked at the run: its empty file is the same.
    assert snapshot(workspace.run_dir) == before


def test_a_failed_attempt_removes_the_manifest_that_described_the_run_before_it(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    assert workspace.manifest.is_file()
    longer = other_config(workspace, "longer.yaml", train_steps=300)
    fake.queue(crash(250))

    result = runner.invoke(app, workspace.args(config=longer))

    assert_one_error(result, 4, "eole train failed")
    assert not workspace.manifest.exists()


def test_a_run_that_is_extended_gets_a_manifest_for_its_new_end(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    longer = other_config(workspace, "longer.yaml", train_steps=300)

    lines = run_ok(workspace.args(config=longer)).stdout.splitlines()

    assert lines[:4] == [
        "status: completed",
        "steps: 300",
        "resumed from: 200",
        "attempt: 2",
    ]
    manifest = manifest_of(workspace)
    assert manifest.config_sha256 == sha256(longer)
    assert str(workspace.step_dir(300) / "model.00.safetensors") in [
        item.path for item in manifest.outputs
    ]


def test_the_manifest_is_removed_before_eole_starts(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ok(workspace.args())
    longer = other_config(workspace, "longer.yaml", train_steps=300)
    seen: list[bool] = []
    real = fake.__class__.__call__

    def looking(self: LinkingFakeEole, *args: Any, **kwargs: Any) -> None:
        seen.append(workspace.manifest.exists())
        real(self, *args, **kwargs)

    monkeypatch.setattr(fake.__class__, "__call__", looking)

    run_ok(workspace.args(config=longer))

    assert seen == [False]


def test_a_finished_run_changes_nothing_on_disk_when_launched_again(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    before = snapshot(workspace.run_dir)

    run_ok(workspace.args())

    assert snapshot(workspace.run_dir) == before


def test_a_finished_run_without_a_manifest_gets_one_without_a_new_attempt(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    first = workspace.manifest.read_bytes()
    workspace.manifest.unlink()
    calls = len(fake.calls)

    run_ok(workspace.args())

    assert len(fake.calls) == calls
    rewritten = manifest_of(workspace)
    assert rewritten.stage == "train_launch"
    assert [item.path for item in rewritten.outputs] == [
        item.path for item in Manifest.model_validate_json(first).outputs
    ]
    assert rewritten.metrics["steps"] == 200.0
    assert rewritten.metrics["attempt"] == 1.0


def test_a_manifest_that_no_longer_matches_the_files_is_written_again(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    workspace.command_sh.write_text(
        workspace.command_sh.read_text(encoding="utf-8") + "# by hand\n",
        encoding="utf-8",
    )

    run_ok(workspace.args())

    entry = next(
        item
        for item in manifest_of(workspace).outputs
        if item.path == str(workspace.command_sh)
    )
    assert entry.sha256 == sha256(workspace.command_sh)


def test_a_manifest_of_another_config_is_written_again_for_a_finished_run(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    copy = workspace.root / "copy.yaml"
    copy.write_bytes(workspace.config.read_bytes())

    run_ok(workspace.args(config=copy))

    manifest = manifest_of(workspace)
    assert [item.path for item in manifest.inputs] == [str(copy)]


# --- exit codes -----------------------------------------------------------------------


def test_a_failing_eole_exits_4_with_one_error_line_and_no_manifest(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(crash(120))

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 4, "eole train failed")
    assert not workspace.manifest.exists()
    assert result.stdout == ""


def test_a_nan_loss_exits_4_and_writes_no_manifest(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(Script(nan_from=100))

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 4, "the training loss is not finite")
    assert not workspace.manifest.exists()


def test_a_run_that_stopped_without_its_final_step_exits_4(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(Script(to_step=120, skip_final_save=True))

    result = runner.invoke(app, workspace.args())

    assert_one_error(
        result, 4, "eole train exited with code 0 but the run was interrupted"
    )
    assert not workspace.manifest.exists()


def test_another_config_than_the_runs_exits_2(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(crash(120))
    assert runner.invoke(app, workspace.args()).exit_code == 4
    other = other_config(workspace, "other.yaml", learning_rate=0.3)

    result = runner.invoke(app, workspace.args(config=other))

    assert_one_error(result, 2, "the config is not the one this run was started with")


def test_a_launch_record_that_is_not_one_exits_3(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    fake.queue(crash(120))
    assert runner.invoke(app, workspace.args()).exit_code == 4
    workspace.record.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 3, f"{workspace.record} is not a Rinchen launch record")


def test_an_error_of_any_other_kind_exits_1(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise RinchenError("something else")

    monkeypatch.setattr(train_runs, "launch_run", failing)

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 1, "something else")


def test_an_output_that_cannot_be_written_exits_1_naming_the_run_directory(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(workspace.run_dir))

    monkeypatch.setattr(train_runs, "launch_run", failing)

    result = runner.invoke(app, workspace.args())

    assert_one_error(
        result, 1, f"cannot write the launch outputs under {workspace.run_dir}: "
    )


def test_a_manifest_that_cannot_be_written_exits_1_and_leaves_none(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(out_dir: Path, manifest: Manifest) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(train_common.store, "write", failing)

    result = runner.invoke(app, workspace.args())

    assert_one_error(
        result, 1, f"cannot write the launch outputs under {workspace.run_dir}: "
    )
    assert not workspace.manifest.exists()


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (KeyboardInterrupt(), 130),
        (SystemExit(143), 143),
        (SystemExit(129), 129),
    ],
    ids=["interrupt", "terminate", "hang-up"],
)
def test_a_signal_ends_the_launch_with_its_code_and_leaves_a_run_that_resumes(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    error: BaseException,
    exit_code: int,
) -> None:
    fake.queue(Script(to_step=120, skip_final_save=True, error=error))

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == exit_code, result.output
    assert not workspace.manifest.exists()
    assert "Traceback" not in result.output
    again = run_ok(workspace.args())
    assert "resumed from: 100" in again.stdout.splitlines()


# --- one launch at a time --------------------------------------------------------------


def test_a_launch_into_a_run_that_another_process_holds_exits_2_and_changes_nothing(
    workspace: LaunchWorkspace, fake: LinkingFakeEole
) -> None:
    run_ok(workspace.args())
    longer = other_config(workspace, "longer.yaml", train_steps=300)
    lock = workspace.run_dir / "launch" / "lock"
    holder = subprocess.Popen(
        [sys.executable, "-I", "-c", LOCK_HOLDER, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        before = snapshot(workspace.run_dir)
        calls = len(fake.calls)

        result = runner.invoke(app, workspace.args(config=longer))

        assert_one_error(result, 2, "another launch holds this run")
        assert str(lock) in result.stderr
        assert snapshot(workspace.run_dir) == before
        assert len(fake.calls) == calls
        # The manifest of the finished run survives a launch that was turned away.
        assert workspace.manifest.is_file()
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=30)
        if holder.stdout is not None:
            holder.stdout.close()
    assert run_ok(workspace.args(config=longer)).stdout.startswith("status: completed")


def test_the_command_makes_no_directory_before_the_launcher_does(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise ConfigError("refused by the launcher")

    monkeypatch.setattr(train_runs, "launch_run", failing)

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 2, "refused by the launcher")
    assert not workspace.run_dir.exists()


def test_a_data_error_of_the_launcher_leaves_an_existing_manifest(
    workspace: LaunchWorkspace, fake: LinkingFakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ok(workspace.args())
    before = workspace.manifest.read_bytes()

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise DataError("cannot list the checkpoint directory")

    monkeypatch.setattr(train_runs, "launch_run", failing)

    result = runner.invoke(app, workspace.args())

    assert_one_error(result, 3, "cannot list the checkpoint directory")
    assert workspace.manifest.read_bytes() == before


def test_the_command_passes_its_options_to_the_launcher(
    workspace: LaunchWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise ConfigError("stop here")

    monkeypatch.setattr(train_runs, "launch_run", recording)

    runner.invoke(
        app,
        workspace.args(
            "--world-size",
            "4",
            "--gpu-ranks",
            "0, 1,2 ,3",
            "--threads",
            "5",
            "--timeout",
            "9.5",
            "--cuda-visible-devices",
            "4,5,6,7",
            "--write-only",
        ),
    )

    ((args, kwargs),) = calls
    assert args == (workspace.config, workspace.run_dir)
    assert {key: kwargs[key] for key in kwargs if key != "before_writing"} == {
        "world_size": 4,
        "gpu_ranks": [0, 1, 2, 3],
        "threads": 5,
        "timeout": 9.5,
        "cuda_visible_devices": "4,5,6,7",
        "write_only": True,
    }
    # A write-only launch begins no commit: it writes no manifest.
    assert kwargs["before_writing"] is None


def test_a_real_launch_begins_its_commit_through_the_hook_of_the_launcher(
    workspace: LaunchWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks: list[Callable[[], None] | None] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        hooks.append(kwargs["before_writing"])
        raise ConfigError("stop here")

    monkeypatch.setattr(train_runs, "launch_run", recording)

    runner.invoke(app, workspace.args())

    (hook,) = hooks
    assert callable(hook)


def test_relative_paths_work_and_a_rerun_from_the_same_place_changes_nothing(
    workspace: LaunchWorkspace,
    fake: LinkingFakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace.root)
    arguments = ["train", "launch", "--config", "base.yaml", "--run-dir", "run"]

    first = run_ok(arguments)

    assert first.stdout.splitlines()[0] == "status: completed"
    assert [item.path for item in manifest_of(workspace).inputs] == ["base.yaml"]
    before = snapshot(workspace.run_dir)
    again = run_ok(arguments)
    assert again.stdout.splitlines()[0] == "status: already_complete"
    assert snapshot(workspace.run_dir) == before
