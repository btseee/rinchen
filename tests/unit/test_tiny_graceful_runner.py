"""The runner of the commands of `rinchen tiny run` (D15, found by the first real run of the pipeline): an
interrupt of the command asks the command it is running to end, with SIGTERM, and kills its process group
only when that fails, so that `rinchen train launch` can stop the Eole it started.

The runner of the maintenance check (`SubprocessRunner`) kills the group of an interrupted command with SIGKILL
at once. `rinchen train launch` leads no group that holds Eole: Eole leads a session of its own, and a launcher
that dies of SIGKILL leaves it running (`rinchen.train.launch` says so). A SIGHUP, SIGTERM or SIGINT sent to
`rinchen tiny run` therefore left `eole build_vocab` or `eole train` running for as long as it took to end
(measured: 5 to 15 s of `build_vocab` after the command had exited, with `eole train` the same by reading the
code). The tests below use a command that does what the launcher does, by a signal handler that writes a file
before it exits, and stand-ins for the interrupt (a wait that raises `KeyboardInterrupt`): nothing here starts
the pipeline, Eole or a model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from rinchen.cli import tiny as tiny_command
from rinchen.maintain import commands as maintenance
from rinchen.maintain.commands import CommandResult, SubprocessRunner
from rinchen.tiny import graceful_runner
from rinchen.tiny.graceful_runner import GracefulRunner
from tests.support.maintain_project import is_running, wait_until_gone

# A command that ends cleanly when it is asked to: its SIGTERM handler writes `terminated` and exits. It also
# starts a helper of its own in its group and writes the helper's pid, as a launcher's workers would.
COMMAND: Final = """
import pathlib, signal, subprocess, sys, time
ready, terminated, helper_pid = map(pathlib.Path, sys.argv[1:4])
def on_term(number, frame):
    terminated.write_text("terminated")
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
helper = subprocess.Popen(["sleep", "60"])
helper_pid.write_text(str(helper.pid))
ready.write_text("ready")
time.sleep(60)
"""
# A command that ignores SIGTERM: only a kill ends it.
STUBBORN: Final = """
import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text("ready")
time.sleep(60)
"""
GRACE_UPPER_BOUND_SECONDS: Final = 10.0


def _argv(code: str, *paths: Path) -> list[str]:
    return [sys.executable, "-I", "-c", code, *map(str, paths)]


def _environment() -> dict[str, str]:
    return {"PATH": os.environ["PATH"]}


def _interrupt_when_ready(
    ready: Path,
) -> Callable[[subprocess.Popen[bytes], int], int | None]:
    """A stand-in for the wait of the runner: it raises `KeyboardInterrupt` once the command has written
    `ready`."""

    def wait(process: subprocess.Popen[bytes], timeout: int) -> int | None:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert time.monotonic() < deadline, "the command did not become ready"
            time.sleep(0.02)
        raise KeyboardInterrupt

    return wait


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "ready", tmp_path / "terminated", tmp_path / "helper"


# ---- an interrupt ----------------------------------------------------------------------------------


def test_an_interrupt_asks_the_command_to_end_before_it_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, terminated, helper = _files(tmp_path)
    monkeypatch.setattr(graceful_runner, "_wait_for_exit", _interrupt_when_ready(ready))

    with pytest.raises(KeyboardInterrupt):
        GracefulRunner().run(
            _argv(COMMAND, ready, terminated, helper),
            cwd=tmp_path,
            env=_environment(),
            log=tmp_path / "log",
            timeout=60,
        )

    assert terminated.read_text(encoding="utf-8") == "terminated"


def test_an_interrupt_leaves_nothing_of_the_command_alive_the_helper_in_its_group_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, terminated, helper = _files(tmp_path)
    monkeypatch.setattr(graceful_runner, "_wait_for_exit", _interrupt_when_ready(ready))

    with pytest.raises(KeyboardInterrupt):
        GracefulRunner().run(
            _argv(COMMAND, ready, terminated, helper),
            cwd=tmp_path,
            env=_environment(),
            log=tmp_path / "log",
            timeout=60,
        )

    assert wait_until_gone(int(helper.read_text(encoding="utf-8")), 5.0)


def test_a_command_that_ignores_the_request_is_killed_after_the_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "ready"
    monkeypatch.setattr(graceful_runner, "_wait_for_exit", _interrupt_when_ready(ready))
    started = time.monotonic()

    with pytest.raises(KeyboardInterrupt):
        GracefulRunner().run(
            _argv(STUBBORN, ready),
            cwd=tmp_path,
            env=_environment(),
            log=tmp_path / "log",
            timeout=60,
        )

    assert time.monotonic() - started < GRACE_UPPER_BOUND_SECONDS
    assert ready.exists()


def test_a_system_exit_is_an_interrupt_too_and_propagates_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, terminated, helper = _files(tmp_path)

    def wait(process: subprocess.Popen[bytes], timeout: int) -> int | None:
        _interrupt_when_ready(ready)(process, timeout)
        raise AssertionError("unreachable")

    def terminate(process: subprocess.Popen[bytes], timeout: int) -> int | None:
        try:
            wait(process, timeout)
        except KeyboardInterrupt:
            raise SystemExit(143) from None
        return None

    monkeypatch.setattr(graceful_runner, "_wait_for_exit", terminate)

    with pytest.raises(SystemExit) as raised:
        GracefulRunner().run(
            _argv(COMMAND, ready, terminated, helper),
            cwd=tmp_path,
            env=_environment(),
            log=tmp_path / "log",
            timeout=60,
        )

    assert raised.value.code == 143
    assert terminated.read_text(encoding="utf-8") == "terminated"


def test_the_runner_of_the_maintenance_check_kills_at_once_and_the_command_cannot_clean_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason for the runner of this package: the one it replaces never asks."""
    ready, terminated, helper = _files(tmp_path)
    monkeypatch.setattr(maintenance, "_wait_for_exit", _interrupt_when_ready(ready))

    with pytest.raises(KeyboardInterrupt):
        SubprocessRunner().run(
            _argv(COMMAND, ready, terminated, helper),
            cwd=tmp_path,
            env=_environment(),
            log=tmp_path / "log",
            timeout=60,
        )

    assert not terminated.exists()


# ---- what the runner does as the one it extends does ------------------------------------------------


def test_a_command_that_runs_past_its_bound_is_asked_and_then_stopped_and_says_so(
    tmp_path: Path,
) -> None:
    ready, terminated, helper = _files(tmp_path)

    result = GracefulRunner().run(
        _argv(COMMAND, ready, terminated, helper),
        cwd=tmp_path,
        env=_environment(),
        log=tmp_path / "log",
        timeout=1,
    )

    assert result == CommandResult(exit_code=None, timed_out=True)
    assert terminated.read_text(encoding="utf-8") == "terminated"
    assert not is_running(int(helper.read_text(encoding="utf-8")))


def test_a_command_that_ends_gives_its_exit_code_and_its_output_is_the_log(
    tmp_path: Path,
) -> None:
    log = tmp_path / "deeper" / "log"

    result = GracefulRunner().run(
        _argv("import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"),
        cwd=tmp_path,
        env=_environment(),
        log=log,
        timeout=60,
    )

    assert result == CommandResult(exit_code=3, timed_out=False)
    assert sorted(log.read_text(encoding="utf-8").split()) == ["err", "out"]


def test_a_program_that_cannot_be_started_is_a_result_and_not_an_error(
    tmp_path: Path,
) -> None:
    log = tmp_path / "log"

    result = GracefulRunner().run(
        [str(tmp_path / "no-such-program")],
        cwd=tmp_path,
        env=_environment(),
        log=log,
        timeout=60,
    )

    assert result == CommandResult(exit_code=None, timed_out=False, started=False)
    assert log.read_bytes() == b""


def test_the_command_of_the_pipeline_runs_its_stages_with_this_runner() -> None:
    assert type(tiny_command.make_runner()) is GracefulRunner
    assert isinstance(tiny_command.make_runner(), SubprocessRunner)
