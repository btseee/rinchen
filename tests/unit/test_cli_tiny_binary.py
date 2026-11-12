"""`rinchen tiny` through the real binary (`python -m rinchen`) and the registry of commands (D15 Task 7, brief
G7.7, spec P13): the help, the registration and the refusals that need no stage.

These are the only real processes of the task besides the children of the signal tests. Each runs with the
device hidden, the Hugging Face libraries and uv offline, no secret of the environment, a bound on its time
and a working directory of its own, and none starts a stage: the help and a refused profile end before any
file is read, and the dry run lists the commands of the stages without running one. A real run of the
pipeline is the gate on the real binaries, not this module.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.maintain.commands import child_environment
from tests.support.maintain_project import parent_environment
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = [pytest.mark.usefixtures("no_network"), pytest.mark.subprocess]

TIMEOUT_SECONDS = 180
# The refusal of a profile costs the start of the interpreter and the import of the commands (about a
# second when the machine is quiet) and nothing else: the bound only has to tell it from a run.
REFUSAL_SECONDS = 10.0
OPTIONS = (
    "--profile",
    "--out-dir",
    "--config",
    "--qe-fake",
    "--no-qe-fake",
    "--threads",
    "--budget-seconds",
    "--dry-run",
)


def binary(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """`python -m rinchen <arguments>` in `cwd`: bounded, the device hidden, the package indexes offline."""
    environment = child_environment(parent_environment())
    environment["UV_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "rinchen", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def test_the_help_of_the_group_lists_the_run_command(tmp_path: Path) -> None:
    result = binary("tiny", "--help", cwd=tmp_path)

    assert result.returncode == 0
    assert "run" in result.stdout


def test_the_help_of_the_command_lists_every_option_and_offers_no_device(
    tmp_path: Path,
) -> None:
    result = binary("tiny", "run", "--help", cwd=tmp_path)

    assert result.returncode == 0
    for option in OPTIONS:
        assert option in result.stdout, option
    assert "--dev" + "ice" not in result.stdout
    for word in ("Exit", "130", "143", "129"):
        assert word in result.stdout, word
    assert list(tmp_path.iterdir()) == []


def test_the_profile_full_is_refused_quickly_with_exit_2_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result = binary("tiny", "run", "--profile", "full", cwd=tmp_path)
    elapsed = time.monotonic() - started

    assert result.returncode == 2
    assert elapsed < REFUSAL_SECONDS
    assert "no pipeline configuration for the profile 'full'" in result.stderr
    assert "(configs/tiny/full.yaml)" in result.stderr
    assert "docs/runbooks/15_tiny_pipeline.md" in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_a_bad_option_is_exit_2_before_anything_runs(tmp_path: Path) -> None:
    result = binary("tiny", "run", "--threads", "0", cwd=tmp_path)

    assert result.returncode == 2
    assert list(tmp_path.iterdir()) == []


def test_a_dry_run_lists_the_commands_of_the_stages_and_writes_nothing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"

    result = binary("tiny", "run", "--dry-run", "--out-dir", str(out), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    stage_lines = [
        line.split(":")[0]
        for line in result.stdout.splitlines()
        if not line.startswith(" ")
    ]
    assert stage_lines == list(STAGE_IDS)
    commands = [line for line in result.stdout.splitlines() if line.startswith(" ")]
    assert len(commands) == 24
    assert all(" -m rinchen " in command for command in commands)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
