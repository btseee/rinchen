"""Tests for `rinchen.train.launch`: the resumable launcher of an Eole training.

Eole never starts. `run_logged` is replaced, in the module that calls it, by
`tests.support.fake_launch.FakeEole`, which writes the log and the step directories a
real command would leave and fails the way `run_logged` does when a script says so. The
rules the launcher applies (`check_gpu_rules`, the identity of a run, the completeness of
step directories, the log curve) have their own tests; here is what the launcher does
with them: the order of its work, what it writes, and how each way a run can end is
named. The real binary is the integration tests'.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.train import launch as launch_module
from rinchen.train.checkpoints import (
    LAUNCH_FILE,
    identity_of,
    read_launch_record,
    step_directories,
)
from rinchen.train.launch import LaunchResult, launch
from tests.support import step_dirs
from tests.support.fake_launch import (
    EOLE,
    FakeEole,
    Script,
    launch_config,
    write_config,
)

FAILED = "eole train failed: exit code 1; log: x; last 20 lines:\nboom"


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def base_cfg(run_dir: Path) -> dict[str, Any]:
    return launch_config(run_dir)


@pytest.fixture
def config_path(tmp_path: Path, base_cfg: dict[str, Any]) -> Path:
    return write_config(tmp_path / "base.yaml", base_cfg)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> FakeEole:
    stand_in = FakeEole(run_dir)
    monkeypatch.setattr(launch_module, "run_logged", stand_in)
    return stand_in


def go(config_path: Path, run_dir: Path, **options: Any) -> LaunchResult:
    return launch(config_path, run_dir, eole=EOLE, **options)


def crash(step: int) -> Script:
    """A run killed after it reached `step` (it saved nothing there)."""
    return Script(to_step=step, skip_final_save=True, error=UpstreamError(FAILED))


def start_and_crash(
    fake: FakeEole, config_path: Path, run_dir: Path, step: int = 120
) -> None:
    fake.queue(crash(step))
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)


def files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def config_with(base: dict[str, Any], **training: Any) -> dict[str, Any]:
    changed = json.loads(json.dumps(base))
    changed["training"].update(training)
    return changed


# ---- a fresh launch ---------------------------------------------------------------


def test_a_fresh_launch_builds_the_vocabulary_then_trains_and_reports_what_it_did(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    result = go(config_path, run_dir)

    train_config = str(run_dir / "launch" / "train-1.yaml")
    assert [call.command for call in fake.calls] == [
        (EOLE, "build_vocab", "-config", train_config, "-n_sample", "-1"),
        (EOLE, "train", "-config", train_config),
    ]
    assert result == LaunchResult(
        status="completed",
        steps=200,
        resumed_from=None,
        checkpoint=run_dir / "ckpt" / "step_200",
        command_sh=run_dir / "command.sh",
        log=run_dir / "launch" / "attempt-1.log",
        attempt=1,
        incomplete_skipped=(),
    )


def test_each_command_writes_its_own_log_in_the_launch_directory(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)

    assert [call.log_path for call in fake.calls] == [
        run_dir / "launch" / "build_vocab.log",
        run_dir / "launch" / "attempt-1.log",
    ]
    assert [call.what for call in fake.calls] == ["eole build_vocab", "eole train"]


def test_the_launch_record_is_written_before_eole_starts(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    go(config_path, run_dir)

    for call in fake.calls:
        assert call.launch_json is not None, call.command
        document = json.loads(call.launch_json)
        assert document["identity"] == identity_of(base_cfg)
        assert document["world_size"] == 1
        assert document["gpu_ranks"] == []
        assert datetime.fromisoformat(document["created"]).tzinfo is not None


def test_the_launch_record_reads_back_as_the_run_identity(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    go(config_path, run_dir)

    record = read_launch_record(run_dir)
    assert record is not None
    assert record.identity == identity_of(base_cfg)


def test_the_vocabulary_is_built_only_when_the_vocabulary_file_is_missing(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    run_dir.mkdir()
    (run_dir / "vocab.src").write_text("a\nb\n", encoding="utf-8")

    go(config_path, run_dir)

    assert [call.subcommand for call in fake.calls] == ["train"]


def test_the_attempt_config_is_the_base_config_with_nothing_added(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    before = config_path.read_bytes()

    go(config_path, run_dir)

    rendered = yaml.safe_load(
        (run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
    )
    assert rendered == base_cfg
    assert "train_from" not in rendered["training"]
    assert config_path.read_bytes() == before


def test_both_commands_read_the_same_rendered_config(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    go(config_path, run_dir)

    assert [call.config for call in fake.calls] == [base_cfg, base_cfg]


def test_settings_the_config_leaves_out_are_written_into_the_attempt_config(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    del base_cfg["training"]["world_size"]
    del base_cfg["training"]["gpu_ranks"]
    config = write_config(tmp_path / "bare.yaml", base_cfg)

    go(config, run_dir)

    training = fake.trains[0].config["training"]
    assert (training["world_size"], training["gpu_ranks"]) == (1, [])


def test_a_second_launch_of_a_finished_run_runs_nothing(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)
    calls, files = len(fake.calls), files_under(run_dir)

    result = go(config_path, run_dir)

    assert result == LaunchResult(
        status="already_complete",
        steps=200,
        resumed_from=None,
        checkpoint=run_dir / "ckpt" / "step_200",
        command_sh=None,
        log=None,
        attempt=0,
        incomplete_skipped=(),
    )
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_finished_run_is_recognised_whatever_incomplete_steps_lie_around(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)
    leftover = step_dirs.make_step_dir(run_dir / "ckpt", 250)
    step_dirs.truncate(leftover, step_dirs.WEIGHTS_FILE, 10)

    result = go(config_path, run_dir)

    assert (result.status, result.steps, result.incomplete_skipped) == (
        "already_complete",
        200,
        (250,),
    )


def test_the_result_cannot_be_changed(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    result = go(config_path, run_dir)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "written"  # type: ignore[misc]


# ---- the process environment ------------------------------------------------------


def test_a_cpu_run_pins_the_threads_and_hides_the_gpu(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    go(config_path, run_dir)

    for call in fake.calls:
        assert call.env["CUDA_VISIBLE_DEVICES"] == ""
        assert call.env["OMP_NUM_THREADS"] == "2"
        assert call.env["MKL_NUM_THREADS"] == "2"


def test_the_thread_count_is_the_one_asked_for(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir, threads=5)

    assert {c.env["OMP_NUM_THREADS"] for c in fake.calls} == {"5"}
    assert {c.env["MKL_NUM_THREADS"] for c in fake.calls} == {"5"}


def test_fewer_than_one_thread_is_refused_before_anything_is_written(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    with pytest.raises(ValueError, match="threads"):
        go(config_path, run_dir, threads=0)

    assert not run_dir.exists()
    assert fake.calls == []


def gpu_options() -> dict[str, Any]:
    return {"world_size": 2, "gpu_ranks": [0, 1]}


def test_a_gpu_run_leaves_the_process_environment_alone(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")

    go(config, run_dir, **gpu_options())

    assert {call.env["CUDA_VISIBLE_DEVICES"] for call in fake.calls} == {"5,6"}


def test_a_gpu_run_adds_no_device_variable_when_the_process_has_none(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    go(config, run_dir, **gpu_options())

    assert all("CUDA_VISIBLE_DEVICES" not in call.env for call in fake.calls)


def test_requested_devices_are_set_for_a_gpu_run(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")

    go(config, run_dir, cuda_visible_devices="2,3", **gpu_options())

    assert {call.env["CUDA_VISIBLE_DEVICES"] for call in fake.calls} == {"2,3"}


@pytest.mark.parametrize(
    "devices", ["", "0;1", "a", "0,", ",0", "0,,1", "0 1", "-1", "0,1\n", "$X", "0,1x"]
)
def test_devices_that_are_not_digits_separated_by_commas_are_refused(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    devices: str,
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))

    with pytest.raises(ConfigError, match="cuda_visible_devices"):
        go(config, run_dir, cuda_visible_devices=devices, **gpu_options())

    assert not run_dir.exists()
    assert fake.calls == []


def test_devices_are_refused_for_a_run_that_uses_no_gpu(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    with pytest.raises(ConfigError, match=r"cuda_visible_devices.*no GPU"):
        go(config_path, run_dir, cuda_visible_devices="0")

    assert fake.calls == []


def test_fewer_devices_than_ranks_are_refused(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))

    with pytest.raises(ConfigError, match=r"cuda_visible_devices.*2 ranks"):
        go(config, run_dir, cuda_visible_devices="0", **gpu_options())


def test_more_devices_than_ranks_are_fine(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))

    go(config, run_dir, cuda_visible_devices="0,1,2,3", **gpu_options())

    assert fake.trains


def test_extra_environment_is_laid_over_everything_else(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(
        config_path,
        run_dir,
        extra_env={
            "RINCHEN_SHIM": "1",
            "OMP_NUM_THREADS": "9",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )

    for call in fake.calls:
        assert call.env["RINCHEN_SHIM"] == "1"
        assert call.env["OMP_NUM_THREADS"] == "9"
        assert call.env["MKL_NUM_THREADS"] == "2"
        assert call.env["CUDA_VISIBLE_DEVICES"] == "0"


def test_no_extra_variable_is_added_without_a_request(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RINCHEN_SHIM", "from the process")

    go(config_path, run_dir)

    expected = dict(os.environ)
    expected.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    assert fake.trains[0].env == expected


def test_the_timeout_bounds_every_command_and_defaults_to_none(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)
    assert [call.timeout for call in fake.calls] == [None, None]

    shutil.rmtree(run_dir)
    fake.calls.clear()
    go(config_path, run_dir, timeout=90.5)
    assert [call.timeout for call in fake.calls] == [90.5, 90.5]


# ---- the eole command -------------------------------------------------------------


def test_without_an_injected_path_the_console_script_next_to_the_interpreter_is_used(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_module, "eole_command", lambda: "/found/by/which/eole")

    launch(config_path, run_dir)

    assert {call.command[0] for call in fake.calls} == {"/found/by/which/eole"}


def test_a_missing_eole_stops_everything_before_any_file_is_written(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> str:
        raise UpstreamError("the `eole` command is not installed in /x")

    monkeypatch.setattr(launch_module, "eole_command", missing)

    with pytest.raises(UpstreamError, match="not installed"):
        launch(config_path, run_dir)
    with pytest.raises(UpstreamError, match="not installed"):
        launch(config_path, run_dir, write_only=True)

    assert not run_dir.exists()


# ---- overrides --------------------------------------------------------------------


def test_the_world_size_and_ranks_are_laid_over_the_config_and_recorded(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))

    go(config, run_dir, world_size=2, gpu_ranks=[0, 1])

    rendered = yaml.safe_load(
        (run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
    )
    assert (rendered["training"]["world_size"], rendered["training"]["gpu_ranks"]) == (
        2,
        [0, 1],
    )
    record = json.loads((run_dir / LAUNCH_FILE).read_text(encoding="utf-8"))
    assert (record["world_size"], record["gpu_ranks"]) == (2, [0, 1])
    effective = config_with(base_cfg, timeout=3600, world_size=2, gpu_ranks=[0, 1])
    assert record["identity"] == identity_of(effective)


def test_an_override_that_contradicts_the_config_is_refused_before_anything_runs(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    eight = config_with(base_cfg, timeout=7200, world_size=8, gpu_ranks=list(range(8)))
    config = write_config(tmp_path / "eight.yaml", eight)

    with pytest.raises(ConfigError, match=r"world_size.*below the 8 entries"):
        go(config, run_dir, world_size=1)

    assert not run_dir.exists()
    assert fake.calls == []


def test_a_gpu_rule_broken_by_the_overrides_is_refused_before_anything_runs(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    with pytest.raises(ConfigError, match="no process"):
        go(config_path, run_dir, world_size=2)

    assert not run_dir.exists()
    assert fake.calls == []


def test_a_gpu_rule_broken_by_the_config_is_refused_before_anything_runs(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(
        tmp_path / "bad.yaml",
        config_with(base_cfg, early_stopping=2, early_stopping_criteria=["ppl", "acc"]),
    )

    with pytest.raises(ConfigError, match="early_stopping_criteria"):
        go(config, run_dir)

    assert not run_dir.exists()


# ---- the config -------------------------------------------------------------------


@pytest.mark.parametrize("steps", [None, 0, -5, True, "200", 1.5, [200]])
def test_train_steps_must_be_an_explicit_positive_integer(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    steps: object,
) -> None:
    cfg = config_with(base_cfg)
    if steps is None:
        del cfg["training"]["train_steps"]
    else:
        cfg["training"]["train_steps"] = steps
    config = write_config(tmp_path / "steps.yaml", cfg)

    with pytest.raises(ConfigError, match=r"training\.train_steps.*100000"):
        go(config, run_dir)

    assert not run_dir.exists()
    assert fake.calls == []


@pytest.mark.parametrize("model_path", [None, "", 3, "/elsewhere/ckpt", "RUN"])
def test_the_checkpoint_directory_must_be_a_path_inside_the_run_directory(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    model_path: object,
) -> None:
    cfg = config_with(base_cfg)
    if model_path is None:
        del cfg["training"]["model_path"]
    else:
        cfg["training"]["model_path"] = (
            str(run_dir) if model_path == "RUN" else model_path
        )
    config = write_config(tmp_path / "model.yaml", cfg)

    with pytest.raises(ConfigError, match=r"model_path"):
        go(config, run_dir)

    assert not run_dir.exists()


def test_the_vocabulary_file_must_be_named(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    del base_cfg["src_vocab"]
    config = write_config(tmp_path / "novocab.yaml", base_cfg)

    with pytest.raises(ConfigError, match="src_vocab"):
        go(config, run_dir)


def test_an_invalid_config_is_refused_by_the_schema_before_anything_else(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    base_cfg["model"]["embeddings"]["position_encoding_type"] = "Rotary"
    config = write_config(tmp_path / "rotary.yaml", base_cfg)

    with pytest.raises(ConfigError, match="position_encoding_type"):
        go(config, run_dir)

    assert not run_dir.exists()


def test_a_train_from_in_the_base_config_is_an_unknown_key(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    base_cfg["training"]["train_from"] = str(run_dir / "ckpt" / "step_50")
    config = write_config(tmp_path / "smuggled.yaml", base_cfg)

    with pytest.raises(ConfigError, match=r"training\.train_from: unknown key"):
        go(config, run_dir)

    assert fake.calls == []


def test_a_checkpoint_directory_with_a_parent_reference_is_refused_up_front(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    # The resume config would refuse such a train_from hours into the run.
    base_cfg["training"]["model_path"] = str(run_dir / "other" / ".." / "ckpt")
    config = write_config(tmp_path / "dots.yaml", base_cfg)

    with pytest.raises(ConfigError, match=r"model_path.*'\.\.'"):
        go(config, run_dir)

    assert not run_dir.exists()
    assert fake.calls == []


def test_the_resume_config_is_validated_as_a_resume_config(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = launch_module.validate_train_config
    seen: list[tuple[bool, str | None]] = []

    def spy(cfg: Any, *, resumes: bool = False) -> None:
        seen.append((resumes, cfg["training"].get("train_from")))
        real(cfg, resumes=resumes)

    monkeypatch.setattr(launch_module, "validate_train_config", spy)
    start_and_crash(fake, config_path, run_dir)

    go(config_path, run_dir)

    assert (True, str(run_dir / "ckpt" / "step_100")) in seen
    assert (False, None) in seen


def test_a_missing_config_file_is_a_config_error(
    tmp_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    with pytest.raises(ConfigError, match="not found"):
        go(tmp_path / "nothing.yaml", run_dir)


def test_a_dollar_sign_in_the_config_path_or_the_run_directory_is_refused(
    tmp_path: Path, config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    dollar_config = tmp_path / "a$b.yaml"
    shutil.copyfile(config_path, dollar_config)

    with pytest.raises(ConfigError, match=r"config path.*'\$'"):
        go(dollar_config, run_dir)
    with pytest.raises(ConfigError, match=r"run directory.*'\$'"):
        go(config_path, tmp_path / "a$b")

    assert fake.calls == []


def test_a_glob_character_in_the_checkpoint_directory_is_refused(
    tmp_path: Path, base_cfg: dict[str, Any], run_dir: Path, fake: FakeEole
) -> None:
    base_cfg["training"]["model_path"] = str(run_dir / "ck[pt]")
    config = write_config(tmp_path / "glob.yaml", base_cfg)

    with pytest.raises(ConfigError, match="glob"):
        go(config, run_dir)

    assert not run_dir.exists()


def test_a_relative_config_path_and_run_directory_work(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = launch(Path("base.yaml"), Path("run"), eole=EOLE)

    assert result.status == "completed"
    assert (run_dir / "launch.json").is_file()
    assert Path(fake.trains[0].command[3]).is_absolute()


def test_a_config_with_unusual_characters_in_its_paths_is_launched_and_quoted_in_command_sh(
    tmp_path: Path, fake: FakeEole
) -> None:
    run = tmp_path / "it's a run"
    cfg = launch_config(run)
    config = write_config(tmp_path / "odd.yaml", cfg)
    fake.run_dir = run

    launch(config, run, eole=EOLE, write_only=True)

    done = subprocess.run(
        ["bash", "-n", str(run / "command.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (done.returncode, done.stderr) == (0, "")


# ---- resuming ---------------------------------------------------------------------


def test_a_crashed_run_resumes_from_its_newest_complete_step(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, step=120)
    before = config_path.read_bytes()

    result = go(config_path, run_dir)

    assert [d.step for d in step_directories(run_dir / "ckpt")][:2] == [50, 100]
    assert result.status == "completed"
    assert (result.resumed_from, result.attempt, result.steps) == (100, 2, 200)
    resume = yaml.safe_load(
        (run_dir / "launch" / "train-2.yaml").read_text(encoding="utf-8")
    )
    expected = config_with(base_cfg, train_from=str(run_dir / "ckpt" / "step_100"))
    assert resume == expected
    assert fake.trains[1].config == expected
    assert fake.trains[1].command[3] == str(run_dir / "launch" / "train-2.yaml")
    assert config_path.read_bytes() == before


def test_the_first_attempts_config_and_log_are_kept_when_the_run_resumes(
    config_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, step=120)
    first_log = (run_dir / "launch" / "attempt-1.log").read_bytes()

    go(config_path, run_dir)

    assert (
        yaml.safe_load(
            (run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
        )
        == base_cfg
    )
    assert (run_dir / "launch" / "attempt-1.log").read_bytes() == first_log
    assert (run_dir / "launch" / "attempt-2.log").is_file()


def test_a_resume_does_not_rebuild_the_vocabulary(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir)

    go(config_path, run_dir)

    assert [call.subcommand for call in fake.calls] == ["build_vocab", "train", "train"]


def test_an_incomplete_newer_step_is_reported_and_the_resume_is_from_the_one_before(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(
        Script(
            to_step=130, skip_final_save=True, cut=(100,), error=UpstreamError(FAILED)
        )
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from, result.incomplete_skipped) == (
        "completed",
        50,
        (100,),
    )
    assert "step_50" in fake.trains[1].config["training"]["train_from"]
    # Eole saved step 100 again on the way to the end, so it is complete now.
    assert [d.complete for d in step_directories(run_dir / "ckpt")] == [True] * 4


def test_with_no_complete_step_the_run_starts_again_from_scratch(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(
        Script(to_step=60, skip_final_save=True, cut=(50,), error=UpstreamError(FAILED))
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert (result.resumed_from, result.incomplete_skipped, result.attempt) == (
        None,
        (50,),
        2,
    )
    assert "train_from" not in fake.trains[1].config["training"]


def test_a_longer_run_resumes_from_the_end_of_the_shorter_one(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    go(config_path, run_dir)
    longer = write_config(
        tmp_path / "longer.yaml", config_with(base_cfg, train_steps=400)
    )

    result = go(longer, run_dir)

    assert (result.status, result.resumed_from, result.steps) == ("completed", 200, 400)
    assert fake.trains[1].config["training"]["train_steps"] == 400
    assert fake.trains[1].config["training"]["train_from"] == str(
        run_dir / "ckpt" / "step_200"
    )


def test_a_lowered_train_steps_below_the_newest_complete_step_is_refused_with_both_numbers(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    go(config_path, run_dir)
    shorter = write_config(
        tmp_path / "shorter.yaml", config_with(base_cfg, train_steps=150)
    )
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError) as raised:
        go(shorter, run_dir)

    assert "200" in str(raised.value)
    assert "150" in str(raised.value)
    assert "train_steps" in str(raised.value)
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_train_steps_may_stop_exactly_at_a_step_that_was_saved(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, step=120)
    equal = write_config(
        tmp_path / "equal.yaml", config_with(base_cfg, train_steps=100)
    )

    result = go(equal, run_dir)

    assert (result.status, result.steps, result.resumed_from) == (
        "already_complete",
        100,
        None,
    )


# ---- the identity of a run --------------------------------------------------------


def changed_data_path(cfg: dict[str, Any]) -> None:
    cfg["data"]["en-mn.auth"]["path_src"] += "x"


def changed_learning_rate(cfg: dict[str, Any]) -> None:
    cfg["training"]["learning_rate"] = 0.51


def changed_model_size(cfg: dict[str, Any]) -> None:
    cfg["model"]["hidden_size"] = 128
    cfg["model"]["embeddings"]["word_vec_size"] = 128


def changed_seed(cfg: dict[str, Any]) -> None:
    cfg["seed"] = 1235


def changed_batch(cfg: dict[str, Any]) -> None:
    cfg["training"]["batch_size"] = 4096


@pytest.mark.parametrize(
    "change",
    [
        changed_data_path,
        changed_learning_rate,
        changed_model_size,
        changed_seed,
        changed_batch,
    ],
)
def test_a_config_that_differs_from_the_runs_is_refused_before_anything_runs(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    change: Any,
) -> None:
    start_and_crash(fake, config_path, run_dir)
    other = config_with(base_cfg)
    change(other)
    other_path = write_config(tmp_path / "other.yaml", other)
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError) as raised:
        go(other_path, run_dir)

    assert "launch.json" in str(raised.value)
    assert "train_steps" in str(raised.value)
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_another_world_size_through_the_override_is_another_run(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))
    start_and_crash(fake, config, run_dir)
    calls = len(fake.calls)

    with pytest.raises(ConfigError, match=r"world_size 1.*world_size 2"):
        go(config, run_dir, world_size=2, gpu_ranks=[0, 1])

    assert len(fake.calls) == calls


def test_the_same_settings_spelled_out_are_the_same_run(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir)

    result = go(config_path, run_dir, world_size=1, gpu_ranks=[])

    assert (result.status, result.resumed_from) == ("completed", 100)


def test_the_order_of_the_keys_does_not_make_another_run(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir)
    reordered = write_config(
        tmp_path / "reordered.yaml", dict(reversed(list(base_cfg.items())))
    )

    result = go(reordered, run_dir)

    assert result.resumed_from == 100


def test_step_directories_without_a_launch_record_are_refused(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    step_dirs.make_run(run_dir / "ckpt", (50, 100))

    with pytest.raises(ConfigError, match=r"step directories.*launch\.json"):
        go(config_path, run_dir)

    assert fake.calls == []


def test_an_incomplete_step_directory_alone_is_also_refused_without_a_launch_record(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    step = step_dirs.make_step_dir(run_dir / "ckpt", 50)
    step_dirs.truncate(step, step_dirs.WEIGHTS_FILE, 10)

    with pytest.raises(ConfigError, match=r"launch\.json"):
        go(config_path, run_dir)


def test_a_launch_record_that_is_not_a_record_is_a_data_error(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    run_dir.mkdir()
    (run_dir / LAUNCH_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(DataError, match="launch record"):
        go(config_path, run_dir)

    assert fake.calls == []


def test_a_launch_record_of_another_shape_is_a_data_error(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    run_dir.mkdir()
    (run_dir / LAUNCH_FILE).write_text('{"identity": 3}', encoding="utf-8")

    with pytest.raises(DataError, match="launch record"):
        go(config_path, run_dir)


def test_a_record_without_step_directories_means_a_start_from_scratch_with_a_new_record(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    # The first attempt died before it saved anything; the config was fixed since.
    fake.vocabulary_error = UpstreamError("eole build_vocab failed: exit code 1")
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)
    fixed = config_with(base_cfg, learning_rate=0.4)
    fixed_path = write_config(tmp_path / "fixed.yaml", fixed)
    fake.vocabulary_error = None

    result = go(fixed_path, run_dir)

    assert result.status == "completed"
    record = read_launch_record(run_dir)
    assert record is not None
    assert record.identity == identity_of(fixed)


def test_a_record_that_matches_is_left_as_it_was(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir)
    # An old time stamp, so that a rewrite could not give the same bytes.
    document = json.loads((run_dir / LAUNCH_FILE).read_text(encoding="utf-8"))
    document["created"] = "2020-01-01T00:00:00+00:00"
    (run_dir / LAUNCH_FILE).write_text(json.dumps(document), encoding="utf-8")
    record = (run_dir / LAUNCH_FILE).read_bytes()

    go(config_path, run_dir)

    assert (run_dir / LAUNCH_FILE).read_bytes() == record


# ---- how a run ends ---------------------------------------------------------------


def early_stopping_config(base: dict[str, Any]) -> dict[str, Any]:
    return config_with(
        base, early_stopping=2, early_stopping_criteria="ppl", train_steps=600
    )


def test_a_run_that_early_stops_is_reported_with_the_step_it_stopped_at(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150))

    result = go(config, run_dir)

    assert (result.status, result.steps) == ("early_stopped", 150)
    assert result.checkpoint == run_dir / "ckpt" / "step_150"


def test_an_early_stop_is_recognised_whatever_the_exit_code(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    # With several processes Eole's early stop ends the job with exit code 1 after the
    # collective timeout, once the final checkpoint is written.
    cfg = early_stopping_config(config_with(base_cfg, timeout=3600))
    config = write_config(tmp_path / "es.yaml", cfg)
    fake.queue(Script(early_stop_at=150, error=UpstreamError(FAILED)))

    result = go(config, run_dir, world_size=2, gpu_ranks=[0, 1])

    assert (result.status, result.steps) == ("early_stopped", 150)


def test_an_early_stop_whose_checkpoint_is_not_complete_is_an_upstream_error(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150, cut=(150,), error=UpstreamError(FAILED)))

    with pytest.raises(UpstreamError, match="boom"):
        go(config, run_dir)


def test_an_early_stop_at_exit_code_zero_with_a_cut_checkpoint_is_not_a_success(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150, cut=(150,)))

    with pytest.raises(UpstreamError, match="interrupted"):
        go(config, run_dir)


def test_launching_a_run_that_already_early_stopped_runs_nothing(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150))
    go(config, run_dir)
    calls, files = len(fake.calls), files_under(run_dir)

    result = go(config, run_dir)

    assert (result.status, result.steps, result.attempt) == ("early_stopped", 150, 0)
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_later_train_steps_does_not_restart_an_early_stopped_run(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    cfg = early_stopping_config(base_cfg)
    fake.queue(Script(early_stop_at=150))
    go(write_config(tmp_path / "es.yaml", cfg), run_dir)
    longer = write_config(tmp_path / "longer.yaml", config_with(cfg, train_steps=900))
    calls = len(fake.calls)

    result = go(longer, run_dir)

    assert result.status == "early_stopped"
    assert len(fake.calls) == calls


def test_an_early_stop_older_than_the_newest_complete_step_does_not_finish_the_run(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150))
    go(config, run_dir)
    # Someone went on past the stop: the log's stop is not where the run is.
    step_dirs.make_step_dir(run_dir / "ckpt", 200)

    result = go(config, run_dir)

    assert (result.status, result.resumed_from) == ("completed", 200)


def test_a_run_that_fails_is_an_upstream_error_and_keeps_the_message(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(to_step=60, skip_final_save=True, error=UpstreamError(FAILED)))

    with pytest.raises(UpstreamError) as raised:
        go(config_path, run_dir)

    assert str(raised.value) == FAILED


def test_a_timeout_is_an_upstream_error_and_the_run_stays_resumable(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    timed_out = UpstreamError("eole train timed out after 30 s and was killed")
    fake.queue(Script(to_step=120, skip_final_save=True, error=timed_out))
    with pytest.raises(UpstreamError, match="timed out"):
        go(config_path, run_dir, timeout=30)

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from) == ("completed", 100)


def test_an_exit_code_zero_without_the_final_step_is_an_interrupted_run(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    # Ctrl-C makes Eole exit 0 without a final checkpoint.
    fake.queue(Script(to_step=130, skip_final_save=True))

    with pytest.raises(UpstreamError) as raised:
        go(config_path, run_dir)

    message = str(raised.value)
    assert "interrupted" in message
    assert "step 100" in message
    assert "200" in message
    assert str(run_dir / "launch" / "attempt-1.log") in message


def test_an_exit_code_zero_that_saved_nothing_is_an_interrupted_run_too(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(to_step=20, skip_final_save=True))

    with pytest.raises(UpstreamError, match="interrupted"):
        go(config_path, run_dir)


@pytest.mark.parametrize("field", ["xent", "lr"])
def test_a_nan_in_the_log_fails_a_run_that_finished(
    config_path: Path, run_dir: Path, fake: FakeEole, field: str
) -> None:
    fake.queue(Script(nan_from=100, nan_field=field))

    with pytest.raises(UpstreamError) as raised:
        go(config_path, run_dir)

    message = str(raised.value)
    assert f"{field} at step 100" in message
    assert str(run_dir / "launch" / "attempt-1.log") in message
    assert "checkpoints" in message


def test_a_nan_fails_an_early_stopped_run_too(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "es.yaml", early_stopping_config(base_cfg))
    fake.queue(Script(early_stop_at=150, nan_from=50))

    with pytest.raises(UpstreamError, match="xent at step 50"):
        go(config, run_dir)


def test_a_nan_after_the_checkpoint_resumed_from_does_not_fail_a_later_clean_attempt(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    # The loss went NaN at step 110, after the last checkpoint (step 100): the resume
    # starts from sound weights and its own log is clean.
    fake.queue(
        Script(
            to_step=130,
            skip_final_save=True,
            nan_from=110,
            error=UpstreamError(FAILED),
        )
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from) == ("completed", 100)


def test_only_the_last_attempts_log_decides_whether_a_checkpoint_holds_a_nan(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    nan_after_the_checkpoint = Script(
        to_step=130, skip_final_save=True, nan_from=110, error=UpstreamError(FAILED)
    )
    fake.queue(nan_after_the_checkpoint, crash(170))
    for _ in range(2):
        with pytest.raises(UpstreamError):
            go(config_path, run_dir)

    # Attempt 2 resumed from step 100 with a clean log and saved step 150.
    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from, result.attempt) == ("completed", 150, 3)


def test_a_run_whose_loss_went_nan_is_not_passed_off_as_finished_on_the_next_launch(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(nan_from=100))
    with pytest.raises(UpstreamError, match="not finite"):
        go(config_path, run_dir)
    calls = len(fake.calls)

    with pytest.raises(UpstreamError, match=r"not finite.*xent at step 100"):
        go(config_path, run_dir)

    assert len(fake.calls) == calls


def test_a_run_is_not_resumed_from_a_checkpoint_saved_after_its_loss_went_nan(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(
        Script(
            to_step=130,
            skip_final_save=True,
            nan_from=100,
            error=UpstreamError(FAILED),
        )
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)
    calls = len(fake.calls)

    with pytest.raises(UpstreamError) as raised:
        go(config_path, run_dir)

    message = str(raised.value)
    assert "xent at step 100" in message
    assert "step 100" in message
    assert str(run_dir / "launch" / "attempt-1.log") in message
    assert len(fake.calls) == calls


def test_the_nan_guard_also_stops_a_write_only_launch(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(nan_from=100))
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)
    files = files_under(run_dir)

    with pytest.raises(UpstreamError, match="not finite"):
        go(config_path, run_dir, write_only=True)

    assert files_under(run_dir) == files


def test_a_keyboard_interrupt_leaves_the_run_resumable(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(to_step=120, skip_final_save=True, error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from, result.attempt) == ("completed", 100, 2)


@pytest.mark.parametrize("code", [129, 143])
def test_a_termination_signal_leaves_the_run_resumable(
    config_path: Path, run_dir: Path, fake: FakeEole, code: int
) -> None:
    fake.queue(Script(to_step=70, skip_final_save=True, error=SystemExit(code)))
    with pytest.raises(SystemExit) as raised:
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert raised.value.code == code
    assert (result.status, result.resumed_from) == ("completed", 50)


def test_an_interrupt_during_the_vocabulary_leaves_a_start_from_scratch(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.vocabulary_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        go(config_path, run_dir)
    fake.vocabulary_error = None

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from, result.attempt) == (
        "completed",
        None,
        1,
    )
    assert [call.subcommand for call in fake.calls] == [
        "build_vocab",
        "build_vocab",
        "train",
    ]


def test_a_failed_vocabulary_stops_before_training_and_writes_no_attempt_log(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.vocabulary_error = UpstreamError("eole build_vocab failed: exit code 1")

    with pytest.raises(UpstreamError, match="build_vocab"):
        go(config_path, run_dir)

    assert fake.trains == []
    assert not list((run_dir / "launch").glob("attempt-*.log"))


# ---- the files it writes ---------------------------------------------------------


def test_attempts_are_numbered_one_after_the_logs_there_are(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    for step in (70, 120):
        fake.queue(crash(step))
        with pytest.raises(UpstreamError):
            go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert result.attempt == 3
    assert sorted(p.name for p in (run_dir / "launch").iterdir()) == [
        "attempt-1.log",
        "attempt-2.log",
        "attempt-3.log",
        "build_vocab.log",
        "train-1.yaml",
        "train-2.yaml",
        "train-3.yaml",
    ]


def test_a_log_is_never_overwritten_when_the_numbers_have_a_gap(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    (run_dir / "launch").mkdir(parents=True)
    (run_dir / "launch" / "attempt-7.log").write_text(
        "an older attempt", encoding="utf-8"
    )

    result = go(config_path, run_dir)

    assert result.attempt == 8
    assert (run_dir / "launch" / "attempt-7.log").read_text(
        encoding="utf-8"
    ) == "an older attempt"


def test_nothing_is_written_outside_the_run_directory_and_the_paths_the_config_names(
    tmp_path: Path, config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    before = {p for p in tmp_path.rglob("*") if p.is_file()}

    go(config_path, run_dir)

    created = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert created
    assert all(path.is_relative_to(run_dir) for path in created)
    assert {p.name for p in created if p.parent == run_dir} == {
        "command.sh",
        "launch.json",
        "vocab.src",
    }


def test_a_refused_launch_writes_nothing(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    del base_cfg["training"]["train_steps"]
    config = write_config(tmp_path / "x.yaml", base_cfg)
    before = sorted(p.name for p in tmp_path.rglob("*"))

    with pytest.raises(ConfigError):
        go(config, run_dir)

    assert sorted(p.name for p in tmp_path.rglob("*")) == before


# ---- command.sh and --write-only --------------------------------------------------

GOLDEN_HEADER = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt {attempt} of the run in {run_json}.
{origin}
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# Unlike `rinchen train launch` it writes no launch.json and does not look for NaN.
set -euo pipefail
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
{devices}cd {cwd}
[ -f {run}/vocab.src ] || {eole} build_vocab -config {run}/launch/train-{attempt}.yaml -n_sample -1
{eole} train -config {run}/launch/train-{attempt}.yaml
"""
FROM_SCRATCH = "# It starts the run from scratch: no checkpoint of it was complete when this was written."


def golden(
    run_dir: Path,
    cwd: Path,
    *,
    attempt: int = 1,
    threads: int = 2,
    devices: str = "export CUDA_VISIBLE_DEVICES=''\n",
    origin: str = FROM_SCRATCH,
) -> str:
    return GOLDEN_HEADER.format(
        attempt=attempt,
        run_json=json.dumps(str(run_dir)),
        origin=origin,
        threads=threads,
        devices=devices,
        cwd=cwd,
        run=run_dir,
        eole=EOLE,
    )


def test_write_only_writes_the_script_and_the_attempt_config_and_runs_nothing(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = config_path.read_bytes()

    result = go(config_path, run_dir, write_only=True)

    assert result == LaunchResult(
        status="written",
        steps=None,
        resumed_from=None,
        checkpoint=None,
        command_sh=run_dir / "command.sh",
        log=None,
        attempt=1,
        incomplete_skipped=(),
    )
    assert fake.calls == []
    assert sorted(files_under(run_dir)) == ["command.sh", "launch/train-1.yaml"]
    assert (
        yaml.safe_load(
            (run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
        )
        == base_cfg
    )
    assert (run_dir / "command.sh").read_text(encoding="utf-8") == golden(
        run_dir, tmp_path
    )
    assert config_path.read_bytes() == before


def test_write_only_creates_no_launch_record_and_leaves_an_existing_one_alone(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir, write_only=True)
    assert not (run_dir / LAUNCH_FILE).exists()
    start_and_crash(fake, config_path, run_dir)
    record = (run_dir / LAUNCH_FILE).read_bytes()

    go(config_path, run_dir, write_only=True)

    assert (run_dir / LAUNCH_FILE).read_bytes() == record


def test_write_only_twice_gives_the_same_script(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir, write_only=True)
    first = files_under(run_dir)

    result = go(config_path, run_dir, write_only=True)

    assert files_under(run_dir) == first
    assert result.attempt == 1


def test_write_only_names_the_step_a_run_resumes_from(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    start_and_crash(fake, config_path, run_dir)
    calls = len(fake.calls)

    result = go(config_path, run_dir, write_only=True)

    assert (result.status, result.steps, result.resumed_from, result.attempt) == (
        "written",
        100,
        100,
        2,
    )
    assert result.checkpoint is None
    origin = "# It resumes from step 100 (training.train_from of the config it names)."
    assert (run_dir / "command.sh").read_text(encoding="utf-8") == golden(
        run_dir, tmp_path, attempt=2, origin=origin
    )
    resume = yaml.safe_load(
        (run_dir / "launch" / "train-2.yaml").read_text(encoding="utf-8")
    )
    assert resume == config_with(
        base_cfg, train_from=str(run_dir / "ckpt" / "step_100")
    )
    assert len(fake.calls) == calls


def test_write_only_reports_the_incomplete_steps_it_ignores(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(
        Script(
            to_step=130, skip_final_save=True, cut=(100,), error=UpstreamError(FAILED)
        )
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)

    result = go(config_path, run_dir, write_only=True)

    assert (result.resumed_from, result.incomplete_skipped) == (50, (100,))


def test_write_only_applies_the_same_refusals(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir)
    other = write_config(
        tmp_path / "other.yaml", config_with(base_cfg, learning_rate=0.3)
    )
    files = files_under(run_dir)

    with pytest.raises(ConfigError, match="launch.json"):
        go(other, run_dir, write_only=True)

    assert files_under(run_dir) == files


def test_write_only_of_a_finished_run_has_nothing_to_write(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)
    (run_dir / "command.sh").unlink()

    result = go(config_path, run_dir, write_only=True)

    assert result.status == "already_complete"
    assert not (run_dir / "command.sh").exists()


def test_the_eight_gpu_script_exports_the_devices_and_names_the_rendered_config(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = write_config(tmp_path / "eight.yaml", config_with(base_cfg, timeout=7200))

    go(
        config,
        run_dir,
        world_size=8,
        gpu_ranks=list(range(8)),
        cuda_visible_devices="0,1,2,3,4,5,6,7",
        threads=4,
        write_only=True,
    )

    text = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert text == golden(
        run_dir,
        tmp_path,
        threads=4,
        devices="export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n",
    )
    rendered = yaml.safe_load(
        (run_dir / "launch" / "train-1.yaml").read_text(encoding="utf-8")
    )
    assert rendered["training"]["world_size"] == 8
    assert rendered["training"]["gpu_ranks"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_a_gpu_script_without_requested_devices_leaves_the_variable_alone(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = write_config(tmp_path / "gpu.yaml", config_with(base_cfg, timeout=3600))

    go(config, run_dir, write_only=True, **gpu_options())

    text = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" not in text


def test_the_script_is_a_plain_text_file_without_execute_bits(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    result = go(config_path, run_dir)

    assert result.command_sh is not None
    assert stat.S_IMODE(result.command_sh.stat().st_mode) & 0o111 == 0


def test_a_launch_rewrites_the_script_for_its_own_attempt(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    start_and_crash(fake, config_path, run_dir)
    go(config_path, run_dir)

    text = (run_dir / "command.sh").read_text(encoding="utf-8")
    origin = "# It resumes from step 100 (training.train_from of the config it names)."
    assert text == golden(run_dir, tmp_path, attempt=2, origin=origin)


def test_bash_accepts_the_scripts_launch_writes(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    go(config_path, run_dir, write_only=True)
    other = tmp_path / "other_run"
    fake.run_dir = other
    config = write_config(
        tmp_path / "eight.yaml", config_with(launch_config(other), timeout=7200)
    )
    go(
        config,
        other,
        world_size=8,
        gpu_ranks=list(range(8)),
        cuda_visible_devices="0,1,2,3,4,5,6,7",
        write_only=True,
    )

    for script in (run_dir / "command.sh", other / "command.sh"):
        done = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert (done.returncode, done.stderr) == (0, "")
