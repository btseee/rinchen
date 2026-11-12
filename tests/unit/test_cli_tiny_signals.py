"""The signals of `rinchen tiny run` (D15 Task 7, brief G7.4, spec P13): SIGINT, SIGTERM and SIGHUP stop a run
with the exit status of a process that a signal ended (130, 143, 129), kill the process group of the command
that is running and write no manifest.

The runner starts every command in a session of its own, which keeps it out of reach of a signal sent to
this process, so a SIGTERM that killed the command alone would leave a child running for as long as its
bound; `exit_on_termination_signals` turns SIGTERM and SIGHUP into the `SystemExit` that the runner's
cleanup handles for Ctrl-C. Two halves: the command in process (what the signal handlers are during the
run and afterwards, and a real signal sent to this process from a stage), and the real command in a real
process whose first stage runs a shell script through the real runner, a script that starts a sleeping
child; the process is signalled from outside and the pids of the script and its child are checked
afterwards. A SIGHUP that is ignored when the command starts stays ignored, in both.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import tiny, tiny_output
from rinchen.maintain.commands import child_environment
from rinchen.tiny.context import StageContext, StageResult
from tests.support.maintain_project import (
    is_running,
    kill_quietly,
    parent_environment,
    wait_until_gone,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.signal_handlers import DEFAULT_HANDLERS, handlers_installed
from tests.support.tiny_cli import fake_table, install, invoke, run_args

pytestmark = pytest.mark.usefixtures("no_network")

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
START_SECONDS = 180
EXIT_SECONDS = 60

# The command in a real process: the real runner and the real everything, but for the table of the stage
# functions, whose first stage runs the script of the test through the runner and whose others do nothing.
CHILD = """
import signal, sys
from rinchen.cli import tiny
from rinchen.cli.main import app
from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.context import StageResult
from rinchen.tiny.states import StageRecord

SCRIPT = sys.argv[1]
for number, handler in (
    (signal.SIGINT, signal.default_int_handler),
    (signal.SIGTERM, signal.SIG_DFL),
    (signal.SIGHUP, {hup}),
):
    signal.signal(number, handler)


def table(seed, lock_bytes):
    def first(ctx):
        ctx.runner.run(
            ["/bin/sh", SCRIPT],
            cwd=ctx.out_dir,
            env={{"PATH": "/usr/bin:/bin"}},
            log=ctx.out_dir / "logs" / "holding.log",
            timeout=120,
        )
        return StageResult(StageRecord("environment", "ok"), {{}})

    def quiet(stage_id):
        return lambda ctx: StageResult(StageRecord(stage_id, "ok"), {{}})

    return {{
        stage_id: first if stage_id == STAGE_IDS[0] else quiet(stage_id)
        for stage_id in STAGE_IDS
    }}


tiny.make_stage_functions = table
app(sys.argv[2:], prog_name="rinchen")
"""
HOLDING_SCRIPT = """#!/bin/sh
sleep 120 &
echo "$$ $!" > "{pids}"
wait
"""


# ---- the command in process ---------------------------------------------------------------------------------


def sentinel(number: int, frame: Any) -> None:  # pragma: no cover - never called
    raise AssertionError("the previous handler was called")


@POSIX_ONLY
def test_the_handlers_are_replaced_during_the_run_and_come_back_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    during: dict[str, Any] = {}
    recording, clock = install(monkeypatch)
    functions = fake_table({}, clock, recording)

    def watching(seed: int, lock_bytes: bytes) -> dict[str, Any]:
        table = functions(seed, lock_bytes)
        inner = table["environment"]

        def first(ctx: StageContext) -> StageResult:
            during["term"] = signal.getsignal(signal.SIGTERM)
            during["hup"] = signal.getsignal(signal.SIGHUP)
            return inner(ctx)

        table["environment"] = first
        return table

    monkeypatch.setattr(tiny, "make_stage_functions", watching)
    with handlers_installed({signal.SIGTERM: sentinel, signal.SIGHUP: sentinel}):
        result = invoke(run_args(tmp_path / "out"))

        assert result.exit_code == 0, result.output
        assert callable(during["term"]) and during["term"] is not sentinel
        assert callable(during["hup"]) and during["hup"] is not sentinel
        assert signal.getsignal(signal.SIGTERM) is sentinel
        assert signal.getsignal(signal.SIGHUP) is sentinel


@POSIX_ONLY
def test_a_hangup_that_is_ignored_when_the_command_starts_stays_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    recording, clock = install(monkeypatch)
    functions = fake_table({}, clock, recording)

    def watching(seed: int, lock_bytes: bytes) -> dict[str, Any]:
        table = functions(seed, lock_bytes)
        inner = table["environment"]

        def first(ctx: StageContext) -> StageResult:
            seen.append(signal.getsignal(signal.SIGHUP))
            return inner(ctx)

        table["environment"] = first
        return table

    monkeypatch.setattr(tiny, "make_stage_functions", watching)
    with handlers_installed({signal.SIGHUP: signal.SIG_IGN, signal.SIGTERM: sentinel}):
        result = invoke(run_args(tmp_path / "out"))

        assert result.exit_code == 0, result.output
        assert seen == [signal.SIG_IGN]
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is sentinel


@POSIX_ONLY
@pytest.mark.parametrize(
    ("number", "status"),
    [(signal.SIGTERM, 143), (signal.SIGHUP, 129)],
    ids=["sigterm", "sighup"],
)
def test_a_signal_sent_to_the_process_during_a_stage_ends_the_command_and_writes_no_manifest(
    number: signal.Signals, status: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    recording, clock = install(monkeypatch)
    functions = fake_table({}, clock, recording)

    def signalling(seed: int, lock_bytes: bytes) -> dict[str, Any]:
        table = functions(seed, lock_bytes)

        def stage(ctx: StageContext) -> StageResult:
            os.kill(os.getpid(), number)
            raise AssertionError("the signal did not end the stage")

        table["spm"] = stage
        return table

    monkeypatch.setattr(tiny, "make_stage_functions", signalling)
    with handlers_installed(DEFAULT_HANDLERS):
        result = invoke(run_args(out))

        assert result.exit_code == status
        assert signal.getsignal(number) is DEFAULT_HANDLERS[number]
    assert not (out / "manifest.json").exists()
    assert not (out / "report.json").exists()
    # The stage that was signalled is the one that does not record itself: no stage ran after it.
    assert recording.stages == ["environment", "fetch", "corpus", "filter"]


def test_an_exit_between_the_documents_and_the_manifest_leaves_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    install(monkeypatch)

    def interrupted(*args: Any, **keywords: Any) -> Any:
        raise SystemExit(143)

    monkeypatch.setattr(tiny_output, "write_manifest", interrupted)

    result = invoke(run_args(out))

    assert result.exit_code == 143
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("option", ["--clock", "--runner", "--stage-functions"])
def test_the_seams_of_the_tests_are_not_options_of_the_command(option: str) -> None:
    result = invoke(["tiny", "run", option, "x"])

    assert result.exit_code == 2


# ---- the command in a real process ----------------------------------------------------------------------------


def start(tmp_path: Path, *, hangup: str) -> tuple[subprocess.Popen[bytes], Path, Path]:
    """The command in a real process, running a script that starts a sleeping child; returns the process,
    the file the script writes its pid and its child's pid to, and the output directory."""
    pids = tmp_path / "script.pids"
    script = tmp_path / "holding.sh"
    script.write_text(HOLDING_SCRIPT.format(pids=pids), encoding="utf-8")
    script.chmod(0o755)
    out = tmp_path / "out"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD.format(hup=hangup),
            str(script),
            "tiny",
            "run",
            "--out-dir",
            str(out),
        ],
        cwd=tmp_path,
        env=child_environment(parent_environment()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, pids, out


def wait_for_pids(pids: Path, process: subprocess.Popen[bytes]) -> list[int]:
    """The pids of the script and its child, once it has written them."""
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("the command ended before its first stage started")
        if pids.exists() and len(pids.read_text().split()) == 2:
            return [int(pid) for pid in pids.read_text().split()]
        time.sleep(0.1)
    raise AssertionError("the script of the test did not start in time")


@POSIX_ONLY
@pytest.mark.parametrize(
    ("number", "status"),
    [(signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)],
    ids=["sigterm", "sighup", "sigint"],
)
def test_a_signal_stops_the_command_kills_the_child_of_its_runner_and_writes_no_manifest(
    number: signal.Signals, status: int, tmp_path: Path
) -> None:
    process, pids, out = start(tmp_path, hangup="signal.SIG_DFL")
    children: list[int] = []
    try:
        children = wait_for_pids(pids, process)
        assert all(is_running(pid) for pid in children)

        process.send_signal(number)
        process.wait(timeout=EXIT_SECONDS)

        assert process.returncode == status
        assert all(wait_until_gone(pid, seconds=10) for pid in children)
        assert not (out / "manifest.json").exists()
        assert not (out / "report.json").exists()
        assert not (out / "report.md").exists()
    finally:
        process.kill()
        process.wait()
        kill_quietly(children)


@POSIX_ONLY
def test_a_hangup_ignored_at_the_start_does_not_end_the_command_and_a_termination_still_does(
    tmp_path: Path,
) -> None:
    process, pids, out = start(tmp_path, hangup="signal.SIG_IGN")
    children: list[int] = []
    try:
        children = wait_for_pids(pids, process)

        process.send_signal(signal.SIGHUP)
        time.sleep(1.0)
        assert process.poll() is None
        assert all(is_running(pid) for pid in children)

        process.send_signal(signal.SIGTERM)
        process.wait(timeout=EXIT_SECONDS)

        assert process.returncode == 143
        assert all(wait_until_gone(pid, seconds=10) for pid in children)
        assert not (out / "manifest.json").exists()
    finally:
        process.kill()
        process.wait()
        kill_quietly(children)
