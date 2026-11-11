"""The signals of `rinchen maintain check` (D14, spec M3 and M10): SIGINT, SIGTERM and SIGHUP stop a
run with the exit status of a process that a signal ended (130, 143, 129), kill the process group
of the command that is running and write no manifest.

The runner starts every command in a session of its own, which keeps it out of reach of a signal
sent to this process, so a SIGTERM that killed the command alone would leave a child running for
as long as its bound. `exit_on_termination_signals` turns SIGTERM and SIGHUP into the
`SystemExit` that the runner's cleanup handles for Ctrl-C. Two halves: the context manager in
process (what it installs, what it leaves alone, what it puts back), and the real command in a
real process whose `uv` is a shell script that starts a sleeping child, signalled from outside,
with the pids of both checked afterwards.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli.maintain_signals import exit_on_termination_signals
from rinchen.maintain.commands import child_environment
from tests.support.maintain_cli import Workspace, make_workspace
from tests.support.maintain_project import (
    is_running,
    kill_quietly,
    parent_environment,
    wait_until_gone,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.signal_handlers import DEFAULT_HANDLERS, handlers_installed

pytestmark = pytest.mark.usefixtures("no_network")

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
START_SECONDS = 120
EXIT_SECONDS = 60

# The command in a real process, over doubles for the doctor and the installed versions and the
# real runner: the `uv` of the run is the script of the test.
CHILD = """
import signal, sys
from rinchen.cli import maintain
from rinchen.cli.main import app
from rinchen.maintain.check_types import DoctorFinding

for number, handler in (
    (signal.SIGINT, signal.default_int_handler),
    (signal.SIGTERM, signal.SIG_DFL),
    (signal.SIGHUP, signal.SIG_DFL),
):
    signal.signal(number, handler)
maintain.doctor_findings = lambda: [DoctorFinding("D-PY", "pass", "ok")]
maintain.installed = lambda: {}
app(sys.argv[1:], prog_name="rinchen")
"""
SLEEPING_UV = """#!/bin/sh
sleep 120 &
echo "$$ $!" > "{pids}"
wait
"""


def sentinel(number: int, frame: Any) -> None:  # pragma: no cover - never called
    raise AssertionError("the previous handler was called")


# ---- the context manager --------------------------------------------------------------------------------


@POSIX_ONLY
@pytest.mark.parametrize(
    ("number", "status"), [(signal.SIGTERM, 143), (signal.SIGHUP, 129)]
)
def test_a_termination_signal_raises_the_exit_status_of_a_process_it_ended(
    number: signal.Signals, status: int
) -> None:
    with handlers_installed({number: sentinel}), exit_on_termination_signals():
        handler = signal.getsignal(number)
        assert callable(handler) and handler is not sentinel
        with pytest.raises(SystemExit) as raised:
            handler(number, None)
    assert raised.value.code == status


@POSIX_ONLY
def test_the_previous_handlers_come_back_afterwards_whatever_the_block_does() -> None:
    with handlers_installed({signal.SIGTERM: sentinel, signal.SIGHUP: sentinel}):
        with pytest.raises(RuntimeError), exit_on_termination_signals():
            raise RuntimeError("the block failed")

        assert signal.getsignal(signal.SIGTERM) is sentinel
        assert signal.getsignal(signal.SIGHUP) is sentinel


@POSIX_ONLY
def test_sigint_is_left_to_python() -> None:
    with handlers_installed(DEFAULT_HANDLERS), exit_on_termination_signals():
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@POSIX_ONLY
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_that_is_ignored_stays_ignored(number: signal.Signals) -> None:
    # `nohup` ignores SIGHUP on purpose: a request not to die when the terminal closes.
    with handlers_installed({number: signal.SIG_IGN}):
        with exit_on_termination_signals():
            assert signal.getsignal(number) is signal.SIG_IGN
        assert signal.getsignal(number) is signal.SIG_IGN


@POSIX_ONLY
def test_the_one_that_is_not_ignored_is_still_replaced_when_the_other_is() -> None:
    with handlers_installed({signal.SIGHUP: signal.SIG_IGN, signal.SIGTERM: sentinel}):
        with exit_on_termination_signals():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is not sentinel
        assert signal.getsignal(signal.SIGTERM) is sentinel


@POSIX_ONLY
def test_a_handler_installed_outside_python_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = signal.getsignal

    def foreign(number: int) -> Any:
        return None if number == signal.SIGTERM else real(number)

    installs: list[int] = []
    real_signal = signal.signal

    def recording(number: int, handler: Any) -> Any:
        installs.append(number)
        return real_signal(number, handler)

    monkeypatch.setattr(signal, "getsignal", foreign)
    monkeypatch.setattr(signal, "signal", recording)

    with handlers_installed({signal.SIGHUP: sentinel}):
        installs.clear()
        with exit_on_termination_signals():
            pass

    assert signal.SIGTERM not in installs
    assert signal.SIGHUP in installs


def test_outside_the_main_thread_the_block_just_runs() -> None:
    seen: dict[str, Any] = {}

    def work() -> None:
        before = signal.getsignal(signal.SIGTERM)
        with exit_on_termination_signals():
            seen["inside"] = signal.getsignal(signal.SIGTERM)
        seen["unchanged"] = seen["inside"] is before

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(timeout=10)

    assert seen["unchanged"] is True


# ---- the command in a real process ----------------------------------------------------------------------


def start(ws: Workspace, tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    """The command, running with a `uv` that sleeps; returns the process and the file the script
    writes its pid and its child's pid to."""
    pids = tmp_path / "uv.pids"
    shim = tmp_path / "bin" / "uv"
    shim.parent.mkdir()
    shim.write_text(SLEEPING_UV.format(pids=pids))
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    arguments = ws.args("--no-drift", "--no-reruns", uv=str(shim))
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD, *arguments],
        env=child_environment(parent_environment()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, pids


def wait_for_pids(pids: Path, process: subprocess.Popen[bytes]) -> list[int]:
    """The pids of the script and its child, once it has written them."""
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("the command ended before its uv started")
        if pids.exists() and len(pids.read_text().split()) == 2:
            return [int(pid) for pid in pids.read_text().split()]
        time.sleep(0.1)
    raise AssertionError("the uv of the test did not start in time")


@POSIX_ONLY
@pytest.mark.parametrize(
    ("number", "status"),
    [(signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)],
    ids=["sigterm", "sighup", "sigint"],
)
def test_a_signal_stops_the_command_kills_its_child_and_writes_no_manifest(
    tmp_path: Path, number: signal.Signals, status: int
) -> None:
    ws = make_workspace(tmp_path)
    process, pids = start(ws, tmp_path)
    children: list[int] = []
    try:
        children = wait_for_pids(pids, process)
        assert all(is_running(pid) for pid in children)

        process.send_signal(number)
        process.wait(timeout=EXIT_SECONDS)

        assert process.returncode == status
        assert all(wait_until_gone(pid, seconds=10) for pid in children)
        assert not (ws.out / "manifest.json").exists()
        assert not (ws.out / "maintain_report.json").exists()
    finally:
        process.kill()
        process.wait()
        kill_quietly(children)
