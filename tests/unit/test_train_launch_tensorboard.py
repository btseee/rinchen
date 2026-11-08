"""Where the training curves of a launch go.

Eole's own default for `tensorboard_log_dir` is the relative `runs/eole`, so a run that
sets `tensorboard: true` and no directory would write its event files into the working
directory of whoever launches it, one directory per attempt, outside the run. `launch`
puts the directory into the effective config instead: `<run directory>/tensorboard`,
spelled as an absolute path. It is an effective value like any other, so it is in every
rendered attempt config and in the identity of the run.

Eole never starts here: `run_logged` is replaced by the launcher tests' `FakeEole`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train import launch as launch_module
from rinchen.train.checkpoints import identity_of, read_launch_record
from rinchen.train.launch import LaunchResult, launch
from tests.support.fake_launch import (
    EOLE,
    FakeEole,
    Script,
    launch_config,
    write_config,
)

FAILED = "eole train failed: exit code 1; log: x; last 20 lines:\nboom"

pytestmark = pytest.mark.usefixtures("fake")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> FakeEole:
    stand_in = FakeEole(run_dir)
    monkeypatch.setattr(launch_module, "run_logged", stand_in)
    return stand_in


def go(base: dict[str, Any], run_dir: Path, **options: Any) -> LaunchResult:
    config = write_config(run_dir.parent / "base.yaml", base)
    return launch(config, run_dir, eole=EOLE, **options)


def rendered(run_dir: Path, attempt: int = 1) -> dict[str, Any]:
    text = (run_dir / "launch" / f"train-{attempt}.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def test_tensorboard_without_a_directory_gets_one_inside_the_run(
    run_dir: Path,
) -> None:
    go({**launch_config(run_dir), "tensorboard": True}, run_dir)

    assert rendered(run_dir)["tensorboard_log_dir"] == f"{run_dir}/tensorboard"


def test_both_eole_commands_read_the_default_directory(
    run_dir: Path, fake: FakeEole
) -> None:
    go({**launch_config(run_dir), "tensorboard": True}, run_dir)

    assert [call.config["tensorboard_log_dir"] for call in fake.calls] == [
        f"{run_dir}/tensorboard"
    ] * 2


def test_only_the_directory_is_added_to_the_config(run_dir: Path) -> None:
    base = {**launch_config(run_dir), "tensorboard": True}

    go(base, run_dir)

    assert rendered(run_dir) == {
        **base,
        "tensorboard_log_dir": f"{run_dir}/tensorboard",
    }


def test_a_relative_run_directory_gives_an_absolute_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_dir: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    base = {**launch_config(run_dir), "tensorboard": True}

    go(base, Path("run"))

    assert rendered(Path("run"))["tensorboard_log_dir"] == f"{run_dir}/tensorboard"


def test_the_directory_of_a_resume_is_the_same(run_dir: Path, fake: FakeEole) -> None:
    base = {**launch_config(run_dir), "tensorboard": True}
    fake.queue(Script(to_step=120, skip_final_save=True, error=UpstreamError(FAILED)))
    with pytest.raises(UpstreamError):
        go(base, run_dir)

    go(base, run_dir)

    assert rendered(run_dir, 2)["tensorboard_log_dir"] == f"{run_dir}/tensorboard"
    assert rendered(run_dir, 2)["training"]["train_from"] == f"{run_dir}/ckpt/step_100"


def test_write_only_renders_the_directory_too(run_dir: Path) -> None:
    go({**launch_config(run_dir), "tensorboard": True}, run_dir, write_only=True)

    assert rendered(run_dir)["tensorboard_log_dir"] == f"{run_dir}/tensorboard"


def test_a_directory_the_config_names_is_kept(run_dir: Path, tmp_path: Path) -> None:
    named = str(tmp_path / "curves" / "teacher")
    base = {
        **launch_config(run_dir),
        "tensorboard": True,
        "tensorboard_log_dir": named,
    }

    go(base, run_dir)

    assert rendered(run_dir) == base


@pytest.mark.parametrize("switch", [False, None], ids=["off", "absent"])
def test_without_tensorboard_nothing_is_added(
    run_dir: Path, switch: bool | None
) -> None:
    base = launch_config(run_dir)
    if switch is not None:
        base["tensorboard"] = switch

    go(base, run_dir)

    assert rendered(run_dir) == base
    assert "tensorboard_log_dir" not in rendered(run_dir)


def test_an_empty_directory_name_is_refused_and_not_replaced_by_the_default(
    run_dir: Path,
) -> None:
    base = {**launch_config(run_dir), "tensorboard": True, "tensorboard_log_dir": ""}

    with pytest.raises(ConfigError, match="tensorboard_log_dir: must be a non-empty"):
        go(base, run_dir)

    assert not run_dir.exists()


def test_the_default_directory_is_part_of_the_identity_of_the_run(
    run_dir: Path,
) -> None:
    base = {**launch_config(run_dir), "tensorboard": True}

    go(base, run_dir)

    record = read_launch_record(run_dir)
    assert record is not None
    assert record.identity == identity_of(rendered(run_dir))
    assert record.identity != identity_of(base)


def test_a_run_started_with_tensorboard_resumes_from_another_spelling_of_its_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    # As for `training.model_path`: what makes the run is the effective config, and a
    # relative and an absolute spelling of the run directory name the same one.
    monkeypatch.chdir(tmp_path)
    base = {**launch_config(run_dir), "tensorboard": True}
    fake.queue(Script(to_step=120, skip_final_save=True, error=UpstreamError(FAILED)))
    with pytest.raises(UpstreamError):
        go(base, Path("run"))

    result = go(base, run_dir)

    assert (result.status, result.resumed_from) == ("completed", 100)
