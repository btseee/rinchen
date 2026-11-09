"""`rinchen train average`: which checkpoints it picks, what it prints, the manifest it
writes last into the averaged directory, and the exit code of each failure.

The checkpoints are small fabricated step directories (`tests.support.step_dirs`) with
real safetensors files; nothing starts Eole. The averaging itself is tested in
`test_train_average.py`; here is what the command makes of it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from rinchen.cli import train_common, train_runs
from rinchen.cli.main import app
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support import step_dirs
from tests.support.launch_cli import assert_one_error, snapshot

runner = CliRunner()

OPTIONS = ["--checkpoint-dir", "--last", "--out"]
STEPS = (50, 100, 150, 200, 250)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def checkpoints(run_dir: Path) -> Path:
    model_path = run_dir / "ckpt"
    step_dirs.make_run(model_path, STEPS)
    step_dirs.write_run_record(run_dir)
    return model_path


def args(checkpoints: Path, *extra: str) -> list[str]:
    return ["train", "average", "--checkpoint-dir", str(checkpoints), *extra]


def run_ok(arguments: list[str]) -> Any:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return result


def manifest_in(directory: Path) -> Manifest:
    manifest = store.read(directory)
    assert manifest is not None
    return manifest


# --- help and options --------------------------------------------------------------


def test_average_help_names_every_option() -> None:
    result = runner.invoke(app, ["train", "average", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout


@pytest.mark.parametrize("option", ["--checkpoint-dir", "--last"])
def test_a_required_option_that_is_missing_is_a_usage_error(
    checkpoints: Path, option: str
) -> None:
    arguments = args(checkpoints, "--last", "3")
    index = arguments.index(option)
    del arguments[index : index + 2]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2, result.output
    assert f"Missing option '{option}'" in result.stderr


@pytest.mark.parametrize("last", ["1", "0", "-3"])
def test_fewer_than_two_checkpoints_is_a_config_error(
    checkpoints: Path, run_dir: Path, last: str
) -> None:
    result = runner.invoke(app, args(checkpoints, "--last", last))

    assert_one_error(result, 2, f"--last must be at least 2, got {last}")
    assert not (run_dir / "averaged").exists()


def test_a_last_that_is_not_a_number_is_a_usage_error(checkpoints: Path) -> None:
    result = runner.invoke(app, args(checkpoints, "--last", "three"))

    assert result.exit_code == 2, result.output
    assert "--last" in result.stderr


# --- what it picks and prints --------------------------------------------------------


def test_it_averages_the_newest_complete_steps_into_averaged_beside_the_checkpoints(
    checkpoints: Path, run_dir: Path
) -> None:
    result = run_ok(args(checkpoints, "--last", "3"))

    out = run_dir / "averaged"
    lines = result.stdout.splitlines()
    assert lines[0] == "averaged: 3 checkpoints (steps 150, 200, 250)"
    assert lines[1] == f"output: {out}"
    assert lines[2] == f"weights sha256: {sha256(out / 'model.00.safetensors')}"
    assert lines[3] == f"manifest: {out / 'manifest.json'}"
    assert len(lines) == 4
    tensors = step_dirs.read_safetensors_raw(out / "model.00.safetensors")
    expected = step_dirs.default_tensors(200)["encoder.layer.0.weight"]
    dtype, shape, data = tensors["encoder.layer.0.weight"]
    assert (dtype, shape) == ("F32", (3, 4))
    assert np.frombuffer(data, dtype="<f4").reshape(3, 4).tolist() == expected.tolist()


def test_it_names_the_sources_in_the_record_of_the_average(
    checkpoints: Path, run_dir: Path
) -> None:
    run_ok(args(checkpoints, "--last", "2"))

    record = json.loads(
        (run_dir / "averaged" / "averaged_from.json").read_text(encoding="utf-8")
    )
    assert [(source["step"], source["path"]) for source in record["sources"]] == [
        (200, str(checkpoints / "step_200")),
        (250, str(checkpoints / "step_250")),
    ]


def test_out_names_another_directory(checkpoints: Path, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere" / "mean"
    out.parent.mkdir()

    result = run_ok(args(checkpoints, "--last", "2", "--out", str(out)))

    assert f"output: {out}" in result.stdout.splitlines()
    assert (out / "model.00.safetensors").is_file()
    assert not (tmp_path / "run" / "averaged").exists()


def test_the_default_output_is_beside_a_relative_checkpoint_directory(
    checkpoints: Path, run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    run_ok(["train", "average", "--checkpoint-dir", "run/ckpt/", "--last", "2"])

    assert (run_dir / "averaged" / "model.00.safetensors").is_file()


def test_the_default_output_is_beside_a_checkpoint_directory_that_is_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disk = tmp_path / "other_disk"
    step_dirs.make_run(disk / "exp1", (100, 200))
    run = tmp_path / "run"
    run.mkdir()
    step_dirs.write_run_record(run)
    (run / "ckpt").symlink_to(disk / "exp1", target_is_directory=True)
    monkeypatch.chdir(run)

    result = run_ok(["train", "average", "--checkpoint-dir", "ckpt", "--last", "2"])

    assert f"output: {run / 'averaged'}" in result.stdout.splitlines()
    assert (run / "averaged" / "model.00.safetensors").is_file()
    assert not (disk / "averaged").exists()


def test_the_source_checkpoints_are_left_as_they_are(checkpoints: Path) -> None:
    before = snapshot(checkpoints)

    run_ok(args(checkpoints, "--last", "5"))

    assert snapshot(checkpoints) == before


def test_an_incomplete_step_above_the_oldest_picked_is_left_out_and_listed(
    checkpoints: Path, run_dir: Path
) -> None:
    step_dirs.truncate(checkpoints / "step_250", step_dirs.WEIGHTS_FILE, 10)

    result = run_ok(args(checkpoints, "--last", "3"))

    lines = result.stdout.splitlines()
    assert lines[0] == "averaged: 3 checkpoints (steps 100, 150, 200)"
    assert "incomplete step directories skipped: [250]" in lines
    assert lines[-1] == f"manifest: {run_dir / 'averaged' / 'manifest.json'}"


def test_an_incomplete_step_below_the_oldest_picked_is_not_mentioned(
    checkpoints: Path,
) -> None:
    step_dirs.truncate(checkpoints / "step_50", step_dirs.WEIGHTS_FILE, 10)

    result = run_ok(args(checkpoints, "--last", "3"))

    assert "incomplete step directories" not in result.stdout


def test_files_in_the_checkpoint_directory_that_are_not_steps_are_ignored(
    checkpoints: Path,
) -> None:
    step_dirs.link_newest_step(checkpoints, 250)
    (checkpoints / "step_x").mkdir()
    (checkpoints / "notes.txt").write_text("hello", encoding="utf-8")

    result = run_ok(args(checkpoints, "--last", "2"))

    assert result.stdout.splitlines()[0] == "averaged: 2 checkpoints (steps 200, 250)"


# --- refusals --------------------------------------------------------------------------


def test_an_existing_output_is_refused_and_left_alone(
    checkpoints: Path, run_dir: Path
) -> None:
    run_ok(args(checkpoints, "--last", "2"))
    before = snapshot(run_dir / "averaged")

    result = runner.invoke(app, args(checkpoints, "--last", "3"))

    assert_one_error(result, 2, f"the output directory {run_dir / 'averaged'} exists")
    assert snapshot(run_dir / "averaged") == before


def test_checkpoints_that_no_launch_wrote_are_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    # Two step directories laid out by hand: averaged, they would make a model that
    # passes for ours, and `launch --continue-from` accepts any averaged model that a
    # `train_average` manifest vouches for.
    foreign = tmp_path / "foreign" / "ckpt"
    step_dirs.make_run(foreign, (10, 20))
    before = snapshot(tmp_path)

    result = runner.invoke(app, args(foreign, "--last", "2"))

    assert_one_error(
        result, 2, f"cannot average {foreign}: no launch.json within 3 levels above"
    )
    assert snapshot(tmp_path) == before


def test_the_refusal_says_what_is_looked_for_and_not_that_a_launch_wrote_the_steps(
    tmp_path: Path,
) -> None:
    foreign = tmp_path / "foreign" / "ckpt"
    step_dirs.make_run(foreign, (10, 20))

    result = runner.invoke(app, args(foreign, "--last", "2"))

    message = next(
        line for line in result.stderr.splitlines() if line.startswith("error:")
    )
    assert message.endswith(
        "no launch.json within 3 levels above its step directories; only a checkpoint "
        "directory inside a run directory that has one (`rinchen train launch` writes "
        "it) is averaged, because an average is accepted as the model to continue from"
    )


def test_steps_laid_out_by_hand_inside_a_run_directory_with_a_record_are_averaged(
    tmp_path: Path,
) -> None:
    # The guard looks for the file and no more: it is a guard against a mistake.
    run = tmp_path / "run"
    step_dirs.make_run(run / "ckpt", (10, 20))
    step_dirs.write_run_record(run)

    result = run_ok(args(run / "ckpt", "--last", "2"))

    assert result.stdout.startswith("averaged: 2 checkpoints")


def test_the_launch_record_may_be_up_to_three_levels_above_the_steps(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "run" / "models" / "ckpt"  # steps -> ckpt -> models -> run
    step_dirs.make_run(nested, (10, 20))
    step_dirs.write_run_record(tmp_path / "run")

    result = run_ok(args(nested, "--last", "2"))

    assert (tmp_path / "run" / "models" / "averaged" / "manifest.json").is_file()
    assert result.stdout.startswith("averaged: 2 checkpoints")


def test_the_run_is_found_above_a_checkpoint_directory_named_relative_to_the_cwd(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `.` has no parent in the path as given: the run directory is above the cwd.
    monkeypatch.chdir(checkpoints)

    run_ok(["train", "average", "--checkpoint-dir", ".", "--last", "2"])

    assert (run_dir / "averaged" / "manifest.json").is_file()


def test_a_launch_record_four_levels_above_the_steps_is_not_looked_for(
    tmp_path: Path,
) -> None:
    deep = tmp_path / "run" / "a" / "b" / "ckpt"
    step_dirs.make_run(deep, (10, 20))
    step_dirs.write_run_record(tmp_path / "run")

    result = runner.invoke(app, args(deep, "--last", "2"))

    assert_one_error(result, 2, f"cannot average {deep}: no launch.json")
    assert not (deep.parent / "averaged").exists()


def test_a_launch_record_that_cannot_be_read_is_a_data_error_that_writes_nothing(
    checkpoints: Path, run_dir: Path
) -> None:
    (run_dir / "launch.json").write_text("{not json", encoding="utf-8")
    before = snapshot(run_dir)

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert_one_error(result, 3, f"cannot average {checkpoints}: ")
    assert "launch.json" in result.stderr
    assert snapshot(run_dir) == before


def test_fewer_complete_steps_than_asked_for_is_a_data_error_naming_the_count(
    checkpoints: Path, run_dir: Path
) -> None:
    step_dirs.truncate(checkpoints / "step_250", step_dirs.WEIGHTS_FILE, 10)
    step_dirs.truncate(checkpoints / "step_200", step_dirs.WEIGHTS_FILE, 10)

    result = runner.invoke(app, args(checkpoints, "--last", "4"))

    assert_one_error(
        result,
        3,
        f"{checkpoints} holds 3 complete step directories (2 incomplete), "
        "--last 4 needs 4",
    )
    assert not (run_dir / "averaged").exists()


@pytest.mark.parametrize("name", ["missing", "a_file"])
def test_a_checkpoint_directory_that_is_not_one_is_a_data_error(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    if name == "a_file":
        path.write_text("x", encoding="utf-8")

    result = runner.invoke(app, args(path, "--last", "2"))

    assert_one_error(result, 3, f"checkpoint directory not found: {path}")


def test_checkpoints_of_different_shapes_are_a_data_error_that_leaves_nothing(
    checkpoints: Path, run_dir: Path
) -> None:
    step_dirs.make_step_dir(
        checkpoints,
        300,
        tensors={"encoder.layer.0.weight": np.zeros((2, 2), dtype=np.float32)},
    )

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert result.exit_code == 3, result.output
    assert "Traceback" not in result.output
    assert sorted(path.name for path in run_dir.iterdir()) == ["ckpt", "launch.json"]


def test_a_nan_in_a_checkpoint_is_a_data_error_that_leaves_nothing(
    checkpoints: Path, run_dir: Path
) -> None:
    tensors = step_dirs.default_tensors(300)
    tensors["generator.bias"] = np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)
    step_dirs.make_step_dir(checkpoints, 300, tensors=tensors)

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert result.exit_code == 3, result.output
    assert sorted(path.name for path in run_dir.iterdir()) == ["ckpt", "launch.json"]


# --- the manifest ------------------------------------------------------------------------


def test_the_manifest_vouches_for_the_sources_and_the_files_it_wrote(
    checkpoints: Path, run_dir: Path
) -> None:
    run_ok(args(checkpoints, "--last", "3"))

    out = run_dir / "averaged"
    manifest = manifest_in(out)
    assert manifest.stage == "train_average"
    assert manifest.seed == 0
    assert manifest.status == {"average": "ok"}
    assert [(item.path, item.sha256, item.bytes) for item in manifest.inputs] == [
        (
            str(checkpoints / f"step_{step}" / "model.00.safetensors"),
            sha256(checkpoints / f"step_{step}" / "model.00.safetensors"),
            (checkpoints / f"step_{step}" / "model.00.safetensors").stat().st_size,
        )
        for step in (150, 200, 250)
    ]
    assert [item.path for item in manifest.outputs] == [
        str(out / name)
        for name in (
            "model.00.safetensors",
            "averaged_from.json",
            "config.json",
            "toy.model",
            "vocab.json",
        )
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path


def test_the_manifest_carries_the_run_figures_and_the_tool_versions(
    checkpoints: Path, run_dir: Path
) -> None:
    run_ok(args(checkpoints, "--last", "3"))

    manifest = manifest_in(run_dir / "averaged")
    assert manifest.metrics == {
        "checkpoints": 3.0,
        "first_step": 150.0,
        "last_step": 250.0,
        "tensors": 3.0,
        "tensors_averaged": 3.0,
    }
    assert manifest.tool_versions == {
        package: importlib.metadata.version(package)
        for package in ("safetensors", "numpy")
    }
    canonical = json.dumps(
        {"last": 3, "steps": [150, 200, 250]}, sort_keys=True, separators=(",", ":")
    )
    assert (
        manifest.config_sha256 == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )


def test_the_manifest_is_the_last_file_written(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[bool], list[str]]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                [Path(item.path).is_file() for item in manifest.outputs],
                sorted(path.name for path in out_dir.iterdir()),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(args(checkpoints, "--last", "2"))

    ((exist, names),) = seen
    assert all(exist)
    assert "manifest.json" not in names
    assert sorted(path.name for path in (run_dir / "averaged").iterdir()) == sorted(
        [*names, "manifest.json"]
    )


def test_each_weights_file_is_hashed_once(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[Path] = []
    from rinchen.manifest import stage

    real = stage.sha256_of

    def counting(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr("rinchen.train.average.sha256_of", counting)
    monkeypatch.setattr(stage, "sha256_of", counting)

    run_ok(args(checkpoints, "--last", "3"))

    weights = [path for path in hashed if path.name == "model.00.safetensors"]
    sources = [path for path in weights if path.parent.name.startswith("step_")]
    assert sorted(path.parent.name for path in sources) == [
        "step_150",
        "step_200",
        "step_250",
    ]
    # The average's own weights were hashed once, by the averaging, before the rename.
    assert len(weights) == len(sources) + 1


def test_a_manifest_that_cannot_be_written_removes_the_output_and_exits_1(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(out_dir: Path, manifest: Manifest) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(train_common.store, "write", failing)

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert_one_error(
        result, 1, f"cannot write the averaged model under {run_dir / 'averaged'}: "
    )
    assert sorted(path.name for path in run_dir.iterdir()) == ["ckpt", "launch.json"]
    monkeypatch.undo()
    assert runner.invoke(app, args(checkpoints, "--last", "2")).exit_code == 0


def test_an_interrupt_while_the_manifest_is_written_removes_the_output(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(out_dir: Path, manifest: Manifest) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_common.store, "write", interrupted)

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert result.exit_code == 130, result.output
    assert sorted(path.name for path in run_dir.iterdir()) == ["ckpt", "launch.json"]


def test_an_output_that_cannot_be_written_exits_1_naming_it(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*arguments: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(run_dir))

    monkeypatch.setattr(train_runs, "average_run", failing)

    result = runner.invoke(app, args(checkpoints, "--last", "2"))

    assert_one_error(
        result, 1, f"cannot write the averaged model under {run_dir / 'averaged'}: "
    )


def test_the_command_hands_the_newest_directories_and_the_output_to_the_averaging(
    checkpoints: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[Path], Path]] = []
    real = train_runs.average_run

    def recording(step_dirs_: list[Path], out: Path) -> Any:
        calls.append((list(step_dirs_), out))
        return real(step_dirs_, out)

    monkeypatch.setattr(train_runs, "average_run", recording)

    run_ok(args(checkpoints, "--last", "2"))

    assert calls == [
        ([checkpoints / "step_200", checkpoints / "step_250"], run_dir / "averaged")
    ]
