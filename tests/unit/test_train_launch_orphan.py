"""The lock of a run outlives a launcher that was killed, as long as the Eole it started.

`run_logged` puts Eole in a session of its own, so a SIGKILL of the launcher (which
nothing can catch) leaves Eole training. The launcher hands the lock descriptor to both
Eole commands (`pass_fds`), so the lock stays held by that Eole and the next `launch` is
refused instead of starting a second training into the same checkpoint directory. These
tests start real processes: a fake `eole` script (the test interpreter) that waits until
the test lets it finish, and real `rinchen train launch` commands in processes of their
own. Nothing starts the real Eole and nothing touches a GPU.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from rinchen.errors import UpstreamError
from rinchen.train.launch import launch
from tests.support.fake_launch import launch_config, write_config

pytestmark = [
    pytest.mark.subprocess,  # the tests start real child processes
    pytest.mark.skipif(
        sys.platform != "linux", reason="needs flock, sessions and /proc"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS = 200
WAIT_S = 60.0

# `eole build_vocab` and `eole train` as far as the lock is concerned: it says which
# descriptors it holds on the lock file (and whether it could pass them on), and, for
# `train`, keeps running until the test creates `finish`, then saves the last step.
FAKE_EOLE = """#!{python}
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, {repo!r})
from tests.support import step_dirs

fake_dir = Path(os.environ["FAKE_DIR"])
subcommand = sys.argv[1]
lock = os.stat(os.environ["FAKE_LOCK"])
holding = []
for name in os.listdir("/proc/self/fd"):
    try:
        status = os.stat("/proc/self/fd/" + name)
    except OSError:
        continue
    if (status.st_dev, status.st_ino) == (lock.st_dev, lock.st_ino):
        holding.append(int(name))
(fake_dir / (subcommand + ".json")).write_text(
    json.dumps(
        {{"holding": holding, "inheritable": [os.get_inheritable(n) for n in holding]}}
    )
)
with (fake_dir / "invocations").open("a") as invocations:
    invocations.write(str(os.getpid()) + " " + subcommand + "\\n")
if subcommand == "train" and os.environ.get("FAKE_WAIT") == "1":
    (fake_dir / "running.tmp").write_text(str(os.getpid()))
    os.replace(fake_dir / "running.tmp", fake_dir / "running")
    while not (fake_dir / "finish").exists():
        time.sleep(0.05)
    step_dirs.make_step_dir(Path(os.environ["FAKE_MODEL_PATH"]), {steps})
"""

# What `python -m rinchen train launch ...` is, with the fake `eole` in place of the
# real one: the arguments after the fake's path are the command line.
CLI_RUNNER = """
import sys

from rinchen.train import launch as launch_module

fake = sys.argv[1]
launch_module.eole_command = lambda: fake
from rinchen.cli.main import app

app(sys.argv[2:])
"""


@dataclass(frozen=True)
class World:
    root: Path
    config: Path
    run_dir: Path
    fake_dir: Path
    eole: Path

    @property
    def lock(self) -> Path:
        return self.run_dir / "launch" / "lock"

    @property
    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "FAKE_DIR": str(self.fake_dir),
            "FAKE_LOCK": str(self.lock),
            "FAKE_MODEL_PATH": str(self.run_dir / "ckpt"),
            "FAKE_WAIT": "1",
        }

    def launch_command(self) -> list[str]:
        return [
            sys.executable,
            "-I",
            "-c",
            CLI_RUNNER,
            str(self.eole),
            "train",
            "launch",
            "--config",
            str(self.config),
            "--run-dir",
            str(self.run_dir),
            "--threads",
            "1",
        ]

    def invocations(self) -> list[tuple[int, str]]:
        path = self.fake_dir / "invocations"
        if not path.exists():
            return []
        return [
            (int(pid), subcommand)
            for pid, subcommand in (
                line.split() for line in path.read_text().splitlines()
            )
        ]


@pytest.fixture
def world(tmp_path: Path) -> World:
    run_dir = tmp_path / "run"
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    eole = tmp_path / "fake-eole"
    eole.write_text(
        FAKE_EOLE.format(python=sys.executable, repo=str(REPO_ROOT), steps=STEPS),
        encoding="utf-8",
    )
    eole.chmod(0o755)
    config = write_config(
        tmp_path / "base.yaml", launch_config(run_dir, train_steps=STEPS)
    )
    return World(tmp_path, config, run_dir, fake_dir, eole)


def is_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"  # a zombie has exited; only its parent has not reaped it


def lock_is_held(lock: Path) -> bool:
    descriptor = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


def wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"gave up waiting for {what}")
        time.sleep(0.02)


@pytest.fixture
def started_processes(world: World) -> Iterator[list[subprocess.Popen[bytes]]]:
    """The launchers a test starts; on the way out, they and every fake Eole of the
    test (each leads a session of its own) are killed if they still run. Only
    processes this test started, known by their own pids."""
    launchers: list[subprocess.Popen[bytes]] = []
    try:
        yield launchers
    finally:
        for process in launchers:
            if process.poll() is None:
                process.kill()
            process.wait()
        for pid, _ in world.invocations():
            if is_running(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_each_eole_command_holds_the_lock_through_a_descriptor_it_inherited(
    world: World,
) -> None:
    environment = {**world.environment, "FAKE_WAIT": "0"}

    with pytest.raises(UpstreamError, match="interrupted"):
        launch(
            world.config,
            world.run_dir,
            eole=str(world.eole),
            extra_env=environment,
        )

    reports = {
        subcommand: json.loads((world.fake_dir / f"{subcommand}.json").read_text())
        for subcommand in ("build_vocab", "train")
    }
    # Exactly one descriptor on the lock file, open in the command.
    assert {
        name: (len(report["holding"]), report["inheritable"])
        for name, report in reports.items()
    } == {"build_vocab": (1, [True]), "train": (1, [True])}
    assert not lock_is_held(world.lock)


def test_the_lock_outlives_a_launcher_killed_with_sigkill_until_its_eole_ends(
    world: World, started_processes: list[subprocess.Popen[bytes]]
) -> None:
    log = world.root / "launcher-1.log"
    with log.open("wb") as output:
        launcher = subprocess.Popen(
            world.launch_command(),
            env=world.environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    started_processes.append(launcher)
    wait_until(
        lambda: (world.fake_dir / "running").exists() or launcher.poll() is not None,
        "the fake Eole to start training",
    )
    assert launcher.poll() is None, log.read_text(errors="replace")
    orphan = int((world.fake_dir / "running").read_text())

    launcher.send_signal(signal.SIGKILL)
    launcher.wait()

    # The launcher is gone and its Eole trains on, still holding the run.
    assert is_running(orphan)
    assert lock_is_held(world.lock)
    second = subprocess.run(
        world.launch_command(),
        env=world.environment,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert second.returncode == 2, second.stdout + second.stderr
    assert "another launch holds this run" in second.stderr
    assert "or by an Eole process it started" in second.stderr
    assert str(world.lock) in second.stderr
    assert [subcommand for _, subcommand in world.invocations()].count("train") == 1
    assert not (world.run_dir / "launch" / "attempt-2.log").exists()
    assert is_running(orphan)

    # Eole ends: the lock goes with it and the run is finished, as its checkpoint says.
    (world.fake_dir / "finish").write_text("1")
    wait_until(lambda: not is_running(orphan), "the fake Eole to finish")
    wait_until(lambda: not lock_is_held(world.lock), "the lock to be released")
    third = subprocess.run(
        world.launch_command(),
        env=world.environment,
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert third.returncode == 0, third.stdout + third.stderr
    assert "status: already_complete" in third.stdout
    assert [subcommand for _, subcommand in world.invocations()].count("train") == 1
