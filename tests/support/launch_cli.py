"""Shared setup of the tests of `rinchen train launch` and `rinchen train average`: a
workspace with an Eole config laid out for a launch (the one of `fake_launch.launch_config`),
and the stand-in for Eole behind `rinchen.train.launch.run_logged`.

The CLI runs the real launcher against `LinkingFakeEole`, which writes the logs and the
step directories a run leaves (see `tests.support.fake_launch`) and, as Eole does after a
save, links the files of the newest step into the checkpoint directory itself. So a test
of the manifest can see that it lists the files of the step directory and never those
links. Nothing starts Eole and nothing leaves `tmp_path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import Result

from rinchen.train import launch as launch_module
from rinchen.train.checkpoints import step_of
from tests.support import step_dirs
from tests.support.fake_launch import (
    EOLE,
    Call,
    FakeEole,
    launch_config,
    write_config,
)

STEP_FILES = ("model.00.safetensors", "optimizer.pt", "config.json", "vocab.json")


@dataclass
class LinkingFakeEole(FakeEole):
    """`FakeEole` that links the newest step into the checkpoint directory after every
    train call, however the call ends (Eole links during the save)."""

    def _train(self, call: Call) -> None:
        model_path = Path(call.config["training"]["model_path"])
        try:
            super()._train(call)
        finally:
            steps = [
                step
                for entry in model_path.glob("step_*")
                if (step := step_of(entry)) is not None
            ]
            if steps:
                step_dirs.link_newest_step(model_path, max(steps))


@dataclass(frozen=True, slots=True)
class LaunchWorkspace:
    """The files of one `rinchen train launch` test: `config` exists, `run_dir` does
    not."""

    root: Path
    config: Path
    run_dir: Path

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def command_sh(self) -> Path:
        return self.run_dir / "command.sh"

    @property
    def record(self) -> Path:
        return self.run_dir / "launch.json"

    @property
    def model_path(self) -> Path:
        return self.run_dir / "ckpt"

    def step_dir(self, step: int) -> Path:
        return self.model_path / f"step_{step}"

    def attempt_log(self, attempt: int) -> Path:
        return self.run_dir / "launch" / f"attempt-{attempt}.log"

    def args(
        self,
        *extra: str,
        config: Path | None = None,
        run_dir: Path | None = None,
    ) -> list[str]:
        """The command line `rinchen train launch --config .. --run-dir .. *extra`."""
        return [
            "train",
            "launch",
            "--config",
            str(self.config if config is None else config),
            "--run-dir",
            str(self.run_dir if run_dir is None else run_dir),
            *extra,
        ]


def make_launch_workspace(root: Path, **training: object) -> LaunchWorkspace:
    """An Eole config for a 200-step launch into `root/run` (`training` overrides keys of
    its `training` section), written to `root/base.yaml`."""
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / "run"
    config = write_config(root / "base.yaml", launch_config(run_dir, **training))
    return LaunchWorkspace(root=root, config=config, run_dir=run_dir)


def install_fake_eole(
    monkeypatch: pytest.MonkeyPatch, workspace: LaunchWorkspace
) -> LinkingFakeEole:
    """Make the launcher run `LinkingFakeEole` instead of Eole, under the fixed command
    path `EOLE` (the real `eole` is never looked for)."""
    stand_in = LinkingFakeEole(workspace.run_dir)
    monkeypatch.setattr(launch_module, "run_logged", stand_in)
    monkeypatch.setattr(launch_module, "eole_command", lambda: EOLE)
    return stand_in


# A process that holds the lock of a run the way a launch does, says so, and waits.
LOCK_HOLDER = """
import fcntl, os, sys, time
descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("held", flush=True)
time.sleep(120)
"""


def assert_one_error(result: Result, exit_code: int, message_start: str) -> None:
    """The command failed with `exit_code` and said so in one line that starts with
    `error: ` and `message_start`, which is the last line of stderr and no traceback."""
    assert result.exit_code == exit_code, result.output
    errors = [line for line in result.stderr.splitlines() if line.startswith("error:")]
    assert len(errors) == 1, result.stderr
    assert errors[0].startswith(f"error: {message_start}"), errors[0]
    assert result.stderr.splitlines()[-1] == errors[0]
    assert "Traceback" not in result.output


def snapshot(directory: Path) -> dict[str, bytes | str | None]:
    """Every file (its bytes), symbolic link (its target) and directory (None) below
    `directory`, by relative path."""
    found: dict[str, bytes | str | None] = {}
    for path in sorted(directory.rglob("*")):
        key = str(path.relative_to(directory))
        if path.is_symlink():
            found[key] = f"-> {path.readlink()}"
        elif path.is_file():
            found[key] = path.read_bytes()
        else:
            found[key] = None
    return found
