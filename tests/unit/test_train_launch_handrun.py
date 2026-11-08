"""A run started by hand from `command.sh` leaves the log that `launch` reads.

`rinchen train launch --write-only` writes `command.sh`; the operator runs it with bash
(the 8-GPU workflow). The script appends the output of `eole train` to
`launch/attempt-<k>.log`, so the next `rinchen train launch` finds a NaN or an early stop of
the hand-run attempt as it would in one of its own: a NaN run is refused, an early-stopped
run is not resumed to the end. These tests run the written script with bash against a fake
`eole` process (`tests.support.fake_eole_process`); the launches that follow use the real
`run_logged` against the same fake, which must not be started when the launch is turned
away.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from rinchen.train.checkpoints import step_directories
from rinchen.train.launch import LaunchResult, launch
from tests.support import fake_eole_process
from tests.support.fake_launch import launch_config, write_config

pytestmark = pytest.mark.subprocess  # the tests start bash and a fake eole process

BASH = shutil.which("bash")


class HandRun:
    """One run directory, a fake `eole` and the means to write, run and launch."""

    def __init__(self, tmp_path: Path, **training: Any) -> None:
        self.run_dir = tmp_path / "run"
        self.eole = fake_eole_process.executable(tmp_path / "fake-eole")
        self.calls = tmp_path / "eole-calls"
        self.config = write_config(
            tmp_path / "base.yaml", launch_config(self.run_dir, **training)
        )

    def write_script(self) -> Path:
        result = launch(self.config, self.run_dir, eole=str(self.eole), write_only=True)
        assert result.command_sh is not None
        return result.command_sh

    def run_script(self, mode: str, *, exit_status: int = 0) -> int:
        assert BASH is not None
        done = subprocess.run(
            [BASH, str(self.run_dir / "command.sh")],
            env=self.environment(mode, exit_status),
            capture_output=True,
            text=True,
            check=False,
        )
        return done.returncode

    def launch(self, **options: Any) -> LaunchResult:
        return launch(
            self.config,
            self.run_dir,
            eole=str(self.eole),
            extra_env=self.environment("ok", 0),
            **options,
        )

    def environment(self, mode: str, exit_status: int) -> dict[str, str]:
        return {
            **os.environ,
            "FAKE_EOLE_MODE": mode,
            "FAKE_EOLE_EXIT": str(exit_status),
            "FAKE_EOLE_CALLS": str(self.calls),
        }

    def eole_calls(self) -> list[str]:
        return self.calls.read_text().split() if self.calls.exists() else []

    @property
    def attempt_log(self) -> Path:
        return self.run_dir / "launch" / "attempt-1.log"


def test_a_hand_run_leaves_the_attempt_log_with_the_steps_eole_printed(
    tmp_path: Path,
) -> None:
    run = HandRun(tmp_path, train_steps=100)
    run.write_script()

    assert run.run_script("ok") == 0

    log = run.attempt_log.read_text(encoding="utf-8")
    assert "Step 100/  100;" in log
    assert run.eole_calls() == ["build_vocab", "train"]


def test_a_finished_hand_run_is_found_complete_by_the_next_launch(
    tmp_path: Path,
) -> None:
    run = HandRun(tmp_path, train_steps=100)
    run.write_script()
    run.run_script("ok")

    result = run.launch()

    assert (result.status, result.steps, result.attempt) == ("already_complete", 100, 0)
    assert run.eole_calls() == ["build_vocab", "train"]  # nothing new was started


def test_a_hand_run_with_a_nan_is_refused_by_the_next_launch(tmp_path: Path) -> None:
    run = HandRun(tmp_path, train_steps=100)
    run.write_script()
    # Eole goes on training with a NaN loss and exits 0.
    assert run.run_script("nan:20") == 0
    assert "nan" in run.attempt_log.read_text(encoding="utf-8")

    with pytest.raises(UpstreamError, match="not finite") as raised:
        run.launch()

    assert "xent at step 20" in str(raised.value)
    assert str(run.attempt_log) in str(raised.value)
    assert run.eole_calls() == ["build_vocab", "train"]  # nothing new was started


def test_a_hand_run_that_early_stopped_is_not_resumed_to_the_end(
    tmp_path: Path,
) -> None:
    run = HandRun(tmp_path, train_steps=500)
    run.write_script()
    # With several processes Eole exits with 1 after the collective timeout, once the
    # final checkpoint is written; `pipefail` keeps that status.
    assert run.run_script("early:100", exit_status=1) == 1

    result = run.launch()

    assert (result.status, result.steps) == ("early_stopped", 100)
    assert run.eole_calls() == ["build_vocab", "train"]  # nothing new was started
    assert [d.step for d in step_directories(run.run_dir / "ckpt")] == [50, 100]


def test_a_write_only_after_a_hand_run_is_the_next_attempt(tmp_path: Path) -> None:
    run = HandRun(tmp_path, train_steps=500)
    run.write_script()
    run.run_script("interrupted:120")

    result = run.launch(write_only=True)

    assert (result.status, result.attempt, result.resumed_from) == ("written", 2, 100)
    assert (run.run_dir / "launch" / "train-2.yaml").is_file()
