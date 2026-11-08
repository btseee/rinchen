"""A newest step whose `optimizer.pt` has a damaged zip central directory is skipped, not
a traceback, by the real `rinchen` commands.

A bit flipped in the UTF-8 name of a member of the archive makes `zipfile` raise a decoding
error that is not a `BadZipFile`; `is_complete` reports it as a problem of `optimizer.pt`,
so `rinchen train launch` and `rinchen train average` fall back to the previous complete
step. The commands run as subprocesses of the `rinchen` binary beside the interpreter, on a
fabricated run (`tests.support.step_dirs`); the run is finished at step 50, so `launch` has
no Eole to start.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.support import step_dirs
from tests.support.fake_launch import launch_config, write_config

pytestmark = pytest.mark.subprocess  # the tests start the rinchen command

RINCHEN = str(Path(sys.executable).with_name("rinchen"))
FINAL_STEP = 50
DAMAGED_STEP = 100
TIMEOUT_S = 120


def rinchen(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [RINCHEN, *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )


def damage_the_name_in_the_central_directory(optimizer: Path) -> None:
    """Make the UTF-8 name of a member, as the central directory holds it, invalid UTF-8
    (one byte inside a two-byte letter)."""
    content = bytearray(optimizer.read_bytes())
    name = step_dirs.UTF8_MEMBER.encode("utf-8")
    content[content.rindex(name) + 11] = 0xFD
    optimizer.write_bytes(bytes(content))
    with pytest.raises(UnicodeDecodeError):  # the damage is the one meant
        zipfile.ZipFile(optimizer)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A run directory whose config is recorded, with complete steps 25 and 50 and a
    damaged newest step 100."""
    run_dir = tmp_path / "run"
    config = write_config(
        tmp_path / "base.yaml", launch_config(run_dir, train_steps=FINAL_STEP)
    )
    recorded = rinchen(
        "train",
        "launch",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--write-only",
    )
    assert recorded.returncode == 0, recorded.stderr
    step_dirs.make_run(run_dir / "ckpt", (25, FINAL_STEP, DAMAGED_STEP))
    damage_the_name_in_the_central_directory(
        run_dir / "ckpt" / f"step_{DAMAGED_STEP}" / "optimizer.pt"
    )
    return run_dir


def test_launch_skips_the_damaged_newest_step_and_prints_no_traceback(
    tmp_path: Path, run: Path
) -> None:
    done = rinchen(
        "train",
        "launch",
        "--config",
        str(tmp_path / "base.yaml"),
        "--run-dir",
        str(run),
    )

    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr + done.stdout
    facts = dict(line.split(": ", 1) for line in done.stdout.splitlines())
    assert facts["status"] == "already_complete"
    assert facts["steps"] == str(FINAL_STEP)
    assert facts["incomplete step directories skipped"] == f"[{DAMAGED_STEP}]"


def test_average_skips_the_damaged_newest_step_and_prints_no_traceback(
    run: Path,
) -> None:
    done = rinchen(
        "train", "average", "--checkpoint-dir", str(run / "ckpt"), "--last", "2"
    )

    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr + done.stdout
    facts = dict(line.split(": ", 1) for line in done.stdout.splitlines())
    assert facts["averaged"] == "2 checkpoints (steps 25, 50)"
    assert facts["incomplete step directories skipped"] == f"[{DAMAGED_STEP}]"
