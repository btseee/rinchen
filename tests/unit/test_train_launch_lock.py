"""Tests for the lock a launch holds on its run directory (`rinchen.train.launch`).

A real launch takes an exclusive, non-blocking `flock` on `<run_dir>/launch/lock` for its
whole duration, so that two launches never write the same attempt files, `launch.json`
and checkpoint directory at once. The lock belongs to the open file, so it dies with the
process: a crash leaves nothing to clean. `write_only` takes no lock but is refused while
one is held. Eole never starts (`run_logged` is replaced by `FakeEole`); one test runs a
second, real process.
"""

from __future__ import annotations

import copy
import fcntl
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train import launch as launch_module
from rinchen.train.launch import LaunchResult, launch
from tests.support import step_dirs
from tests.support.fake_launch import (
    EOLE,
    FakeEole,
    Script,
    launch_config,
    write_config,
)
from tests.support.launch_cli import LOCK_HOLDER


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


def lock_file(run_dir: Path) -> Path:
    return run_dir / "launch" / "lock"


def lock_is_held(run_dir: Path) -> bool:
    """Whether another open file description holds the lock (a try, released at once)."""
    descriptor = os.open(lock_file(run_dir), os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


@pytest.fixture
def held_by_a_test_descriptor(run_dir: Path) -> Iterator[Path]:
    """The lock of `run_dir` held by this test, as a launch of another process would."""
    lock_file(run_dir).parent.mkdir(parents=True)
    descriptor = os.open(lock_file(run_dir), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield lock_file(run_dir)
    finally:
        os.close(descriptor)


def files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Observing:
    """Stands in for `run_logged` like `FakeEole`, and asks `look` at each command and
    the options it was given before the fake runs it."""

    def __init__(
        self, fake: FakeEole, look: Callable[[Sequence[str], dict[str, Any]], None]
    ) -> None:
        self.fake = fake
        self.look = look

    def __call__(self, command: Sequence[str], **options: Any) -> None:
        self.look(command, options)
        self.fake(command, **options)


# ---- what the lock is --------------------------------------------------------------


def test_a_launch_creates_an_empty_lock_file_in_the_launch_directory(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)

    assert lock_file(run_dir).is_file()
    assert lock_file(run_dir).read_bytes() == b""


def test_the_lock_is_held_while_each_eole_command_runs(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        launch_module,
        "run_logged",
        Observing(
            fake,
            lambda command, options: seen.append((command[1], lock_is_held(run_dir))),
        ),
    )

    go(config_path, run_dir)

    assert seen == [("build_vocab", True), ("train", True)]


def test_the_lock_is_held_while_the_state_of_the_run_is_read(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = launch_module.latest_complete
    seen: list[bool] = []

    def looking(model_path: Path, **options: Any) -> Any:
        seen.append(lock_is_held(run_dir))
        return real(model_path, **options)

    monkeypatch.setattr(launch_module, "latest_complete", looking)

    go(config_path, run_dir)

    # Once before the attempt and once to see how it ended.
    assert seen == [True, True]


def lock_identity(run_dir: Path) -> tuple[int, int]:
    """(st_dev, st_ino) of the lock file: what `FakeEole` records for a descriptor."""
    status = os.stat(lock_file(run_dir))
    return (status.st_dev, status.st_ino)


def test_both_eole_commands_are_handed_the_lock_descriptor_and_nothing_else(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    # The Eole process keeps the lock alive when the launcher is killed (see
    # test_train_launch_orphan.py); the descriptor goes to these two commands only.
    go(config_path, run_dir)

    assert [
        (call.subcommand, len(call.inherit_fds), call.inherited_files)
        for call in fake.calls
    ] == [
        ("build_vocab", 1, (lock_identity(run_dir),)),
        ("train", 1, (lock_identity(run_dir),)),
    ]


def test_the_descriptor_a_command_gets_is_the_open_file_the_launch_locked(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Locking again through the handed descriptor succeeds only when it is the very
    # open file the launch holds the lock on: a second open of the lock file would
    # be refused. So the command's copy keeps that lock alive.
    relocked: list[int] = []

    def look(command: Sequence[str], options: dict[str, Any]) -> None:
        (descriptor,) = options["inherit_fds"]
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        relocked.append(descriptor)

    monkeypatch.setattr(launch_module, "run_logged", Observing(fake, look))

    go(config_path, run_dir)

    assert len(relocked) == 2


def test_a_write_only_launch_starts_no_command_and_hands_out_no_descriptor(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir, write_only=True)

    assert fake.calls == []


def test_the_lock_message_names_the_launch_and_the_eole_it_started(
    config_path: Path, run_dir: Path, fake: FakeEole, held_by_a_test_descriptor: Path
) -> None:
    with pytest.raises(ConfigError) as raised:
        go(config_path, run_dir)

    message = str(raised.value)
    assert str(held_by_a_test_descriptor) in message
    assert "another launch holds this run" in message
    assert "or by an Eole process it started" in message


def test_a_launch_of_a_finished_run_takes_the_lock_and_gives_it_back(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)
    files = files_under(run_dir)

    result = go(config_path, run_dir)

    assert result.status == "already_complete"
    assert not lock_is_held(run_dir)
    assert files_under(run_dir) == files


# ---- released on every way out ----------------------------------------------------


@pytest.mark.parametrize(
    ("script", "error"),
    [
        pytest.param(
            Script(to_step=120, skip_final_save=True, error=UpstreamError("boom")),
            UpstreamError,
            id="failure",
        ),
        pytest.param(
            Script(to_step=120, skip_final_save=True, error=KeyboardInterrupt()),
            KeyboardInterrupt,
            id="keyboard-interrupt",
        ),
        pytest.param(
            Script(to_step=120, skip_final_save=True, error=SystemExit(143)),
            SystemExit,
            id="system-exit",
        ),
        pytest.param(Script(nan_from=100), UpstreamError, id="nan-at-the-end"),
    ],
)
def test_the_lock_is_released_after_a_launch_that_does_not_succeed(
    config_path: Path,
    run_dir: Path,
    fake: FakeEole,
    script: Script,
    error: type[BaseException],
) -> None:
    fake.queue(script)

    with pytest.raises(error):
        go(config_path, run_dir)

    assert not lock_is_held(run_dir)


def test_the_lock_is_released_after_a_launch_that_succeeds(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)

    assert not lock_is_held(run_dir)


def test_the_lock_is_released_after_a_launch_that_is_refused_once_the_run_is_read(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    fake: FakeEole,
) -> None:
    fake.queue(Script(to_step=120, skip_final_save=True, error=UpstreamError("boom")))
    with pytest.raises(UpstreamError):
        go(config_path, run_dir)
    changed = copy.deepcopy(base_cfg)
    changed["training"]["learning_rate"] = 0.3
    other = write_config(tmp_path / "other.yaml", changed)

    with pytest.raises(ConfigError, match="not the one this run was started with"):
        go(other, run_dir)

    assert not lock_is_held(run_dir)


def test_a_launch_after_a_failed_one_takes_the_lock_again_and_resumes(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    fake.queue(Script(to_step=120, skip_final_save=True, error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        go(config_path, run_dir)

    result = go(config_path, run_dir)

    assert (result.status, result.resumed_from) == ("completed", 100)


# ---- a second launch ---------------------------------------------------------------


class Pausing:
    """Stands in for `run_logged`: the first `eole train` waits until `release` is set,
    after saying so through `started`; everything else goes to the fake."""

    def __init__(self, fake: FakeEole) -> None:
        self.fake = fake
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, command: Sequence[str], **options: Any) -> None:
        if command[1] == "train":
            self.started.set()
            assert self.release.wait(timeout=30), "the test never released the launch"
        self.fake(command, **options)


def test_a_second_launch_into_a_run_that_a_launch_holds_is_refused(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    pausing = Pausing(fake)
    monkeypatch.setattr(launch_module, "run_logged", pausing)
    outcome: list[LaunchResult] = []
    first = threading.Thread(target=lambda: outcome.append(go(config_path, run_dir)))
    first.start()
    try:
        assert pausing.started.wait(timeout=30)
        snapshot = files_under(run_dir)

        with pytest.raises(ConfigError) as raised:
            go(config_path, run_dir)

        message = str(raised.value)
        assert "another launch" in message
        assert str(lock_file(run_dir)) in message
        assert files_under(run_dir) == snapshot
        # The refused launch closed its own descriptor, not the first launch's lock.
        assert lock_is_held(run_dir)
        with pytest.raises(ConfigError, match="another launch"):
            go(config_path, run_dir)
    finally:
        pausing.release.set()
        first.join(timeout=30)

    assert [result.status for result in outcome] == ["completed"]
    assert not lock_is_held(run_dir)
    assert len(fake.trains) == 1


def test_a_second_launch_with_a_write_only_script_is_refused_too(
    config_path: Path, run_dir: Path, fake: FakeEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    pausing = Pausing(fake)
    monkeypatch.setattr(launch_module, "run_logged", pausing)
    first = threading.Thread(target=lambda: go(config_path, run_dir))
    first.start()
    try:
        assert pausing.started.wait(timeout=30)
        snapshot = files_under(run_dir)

        with pytest.raises(ConfigError, match="another launch"):
            go(config_path, run_dir, write_only=True)

        assert files_under(run_dir) == snapshot
    finally:
        pausing.release.set()
        first.join(timeout=30)


def test_a_real_second_process_holding_the_lock_refuses_a_launch_until_it_dies(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    lock_file(run_dir).parent.mkdir(parents=True)
    holder = subprocess.Popen(
        [sys.executable, "-I", "-c", LOCK_HOLDER, str(lock_file(run_dir))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"

        with pytest.raises(ConfigError) as raised:
            go(config_path, run_dir)

        assert "another launch" in str(raised.value)
        assert str(lock_file(run_dir)) in str(raised.value)
        assert fake.calls == []
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=30)

        result = go(config_path, run_dir)

        assert result.status == "completed"
    finally:
        holder.kill()
        holder.wait(timeout=30)
        if holder.stdout is not None:
            holder.stdout.close()


# ---- write_only --------------------------------------------------------------------


def test_write_only_takes_no_lock_and_creates_no_lock_file(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir, write_only=True)

    assert not lock_file(run_dir).exists()


def test_write_only_works_beside_a_lock_file_nobody_holds(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    go(config_path, run_dir)

    result = go(config_path, run_dir, write_only=True)

    assert result.status == "already_complete"
    assert not lock_is_held(run_dir)


def test_write_only_is_refused_while_a_launch_holds_the_run_and_writes_nothing(
    config_path: Path, run_dir: Path, fake: FakeEole, held_by_a_test_descriptor: Path
) -> None:
    before = files_under(run_dir)

    with pytest.raises(ConfigError) as raised:
        go(config_path, run_dir, write_only=True)

    assert "another launch" in str(raised.value)
    assert str(held_by_a_test_descriptor) in str(raised.value)
    assert files_under(run_dir) == before
    assert lock_is_held(run_dir)


def test_write_only_next_to_a_resumable_run_is_refused_while_it_is_held(
    config_path: Path, run_dir: Path, fake: FakeEole, held_by_a_test_descriptor: Path
) -> None:
    step_dirs.make_step_dir(run_dir / "ckpt", 50)
    before = files_under(run_dir)

    with pytest.raises(ConfigError, match="another launch"):
        go(config_path, run_dir, write_only=True)

    assert files_under(run_dir) == before
