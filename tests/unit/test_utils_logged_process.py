"""Tests for `rinchen.utils.logged_process`: the logged subprocess runner that
Eole and the CTranslate2 converter run through.

These tests start real children (the test interpreter running a few lines), because
what they pin is process behaviour: the combined log, the session of the child, and
that a timeout or an interrupt kills the child's whole process group.
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from rinchen.utils import logged_process
from rinchen.utils.logged_process import LOG_TAIL_LINES, run_logged

pytestmark = pytest.mark.subprocess  # the tests start real child processes

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="needs process groups, fork and /proc"
)

# The child forks a grandchild, records both pids and then sleeps for the seconds
# of its second argument. The default 20 s is far longer than any timeout below, so a
# child that survives is a defect and not luck.
_CHILD_WITH_GRANDCHILD = """
import os, sys, time
lifetime = float(sys.argv[2]) if len(sys.argv) > 2 else 20
grandchild = os.fork()
if grandchild == 0:
    time.sleep(lifetime)
    os._exit(0)
with open(sys.argv[1] + ".tmp", "w") as handle:
    handle.write(f"{os.getpid()} {grandchild}")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
print("child started", flush=True)
time.sleep(lifetime)
"""


def _python(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


def _run(command: list[str], log_path: Path, **kwargs: Any) -> None:
    run_logged(
        command, env=dict(os.environ), log_path=log_path, what="the child", **kwargs
    )


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except ProcessLookupError, FileNotFoundError:
        return False
    return state != "Z"  # a zombie has exited; only its parent has not reaped it


def _wait_until_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _is_running(pid):
            return True
        time.sleep(0.05)
    return not _is_running(pid)


def _kill_quietly(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _read_pids(pid_file: Path) -> list[int]:
    if not pid_file.exists():
        pytest.fail("the child did not get to start its grandchild in time")
    return [int(pid) for pid in pid_file.read_text().split()]


# ---- the log and the exit status ------------------------------------------------


def test_run_logged_writes_standard_output_and_error_to_one_log(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "print('to stdout', flush=True)\n"
        "print('to stderr', file=sys.stderr, flush=True)\n"
        "print('to stdout again')\n"
    )
    log_path = tmp_path / "nested" / "child.log"

    _run(_python(code), log_path)

    assert log_path.read_text(encoding="utf-8") == (
        "to stdout\nto stderr\nto stdout again\n"
    )


def test_run_logged_gives_the_child_the_environment_it_was_passed(
    tmp_path: Path,
) -> None:
    code = "import os; print(os.environ['CHILD_SETTING'])"
    log_path = tmp_path / "child.log"

    run_logged(
        _python(code),
        env={**os.environ, "CHILD_SETTING": "visible"},
        log_path=log_path,
        what="the child",
    )

    assert log_path.read_text(encoding="utf-8") == "visible\n"


def test_run_logged_gives_the_child_no_standard_input(tmp_path: Path) -> None:
    # A child that read the terminal would hang a batch run.
    log_path = tmp_path / "child.log"

    _run(_python("import sys; print(repr(sys.stdin.read()))"), log_path)

    assert log_path.read_text(encoding="utf-8") == "''\n"


def test_run_logged_reports_a_failing_exit_code_and_the_log_tail(
    tmp_path: Path,
) -> None:
    code = "import sys; print('first'); print('last words'); sys.exit(3)"
    log_path = tmp_path / "child.log"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), log_path)

    message = str(raised.value)
    assert message.startswith("the child failed: exit code 3;")
    assert str(log_path) in message
    assert message.endswith("first\nlast words")


@linux_only
def test_run_logged_words_a_negative_exit_code_as_a_signal(tmp_path: Path) -> None:
    code = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), tmp_path / "child.log")

    assert "killed by signal 15 (exit code -15)" in str(raised.value)


def test_run_logged_keeps_the_last_twenty_lines_of_a_long_log(tmp_path: Path) -> None:
    code = "import sys\nfor n in range(1, 5001): print(f'line {n}')\nsys.exit(1)"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), tmp_path / "child.log")

    expected = [f"line {number}" for number in range(5001 - LOG_TAIL_LINES, 5001)]
    assert str(raised.value).endswith("\n".join(expected))
    assert f"line {5000 - LOG_TAIL_LINES}\n" not in str(raised.value)


def test_run_logged_reads_only_the_end_of_a_huge_log(tmp_path: Path) -> None:
    code = "import sys\nfor n in range(300000): print(f'row {n}')\nsys.exit(1)"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), tmp_path / "child.log")

    assert str(raised.value).endswith("row 299999")
    assert len(str(raised.value)) < 10_000


def test_run_logged_cuts_each_tail_line_to_300_characters(tmp_path: Path) -> None:
    code = "import sys; print('x' * 1000); print('y' * 300); sys.exit(1)"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), tmp_path / "child.log")

    lines = str(raised.value).split("\n")[-2:]
    assert lines == ["x" * 300 + "...", "y" * 300]


def test_run_logged_returns_normally_for_a_child_inside_its_timeout(
    tmp_path: Path,
) -> None:
    _run(_python("print('quick')"), tmp_path / "child.log", timeout=30)


@linux_only
def test_run_logged_starts_the_child_in_a_session_of_its_own(tmp_path: Path) -> None:
    # The own session is what lets a timeout signal the child's whole group.
    code = "import os; print(os.getpgid(0) == os.getpid())"
    log_path = tmp_path / "child.log"

    _run(_python(code), log_path)

    assert log_path.read_text(encoding="utf-8") == "True\n"


# ---- timeout and interrupt ------------------------------------------------------


@linux_only
def test_run_logged_kills_the_child_and_its_grandchild_on_a_timeout(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pids"
    log_path = tmp_path / "child.log"
    started = time.monotonic()

    try:
        with pytest.raises(UpstreamError) as raised:
            _run(_python(_CHILD_WITH_GRANDCHILD, str(pid_file)), log_path, timeout=2.0)
        elapsed = time.monotonic() - started
        pids = _read_pids(pid_file)
        survivors = [pid for pid in pids if not _wait_until_gone(pid)]
    finally:
        if pid_file.exists():
            _kill_quietly(_read_pids(pid_file))

    message = str(raised.value)
    assert "the child timed out after 2.0 s" in message
    assert str(log_path) in message
    assert message.endswith("child started")
    assert elapsed < 10, "the run waited for the child instead of killing it"
    assert survivors == [], f"processes still running after the timeout: {survivors}"


@linux_only
def test_run_logged_kills_the_process_group_when_it_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The child's own session means Ctrl-C no longer reaches it from the terminal.
    pid_file = tmp_path / "pids"
    original_wait = subprocess.Popen.wait
    interrupted = []

    def interrupt_once(
        self: subprocess.Popen[bytes], timeout: float | None = None
    ) -> int:
        if not interrupted:
            interrupted.append(True)
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise KeyboardInterrupt
        return original_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)

    try:
        with pytest.raises(KeyboardInterrupt):
            _run(_python(_CHILD_WITH_GRANDCHILD, str(pid_file)), tmp_path / "child.log")
        pids = _read_pids(pid_file)
        survivors = [pid for pid in pids if not _wait_until_gone(pid)]
    finally:
        if pid_file.exists():
            _kill_quietly(_read_pids(pid_file))

    assert survivors == [], f"processes still running after the interrupt: {survivors}"


@linux_only
@pytest.mark.timeout(30)
def test_a_real_sigint_interrupts_the_wait_while_the_child_runs(tmp_path: Path) -> None:
    # Once polars is imported (every `rinchen` command has it) a SIGINT sent to the
    # process does not wake a blocking wait that has no timeout: the interrupt is
    # raised only when the child ends by itself, which for an Eole training is hours
    # later (docs/VERIFY.md, D-POLARS-SIGINT). The child here lives 6 s, so a wait
    # that is not woken fails the assertion below and does not hang the suite.
    importlib.import_module("polars")
    pid_file = tmp_path / "pids"
    signalled_at: list[float] = []
    stop = threading.Event()

    def send_sigint_once_the_child_runs() -> None:
        while not pid_file.exists():  # a loaded machine may start the child late
            if stop.wait(0.02):
                return
        signalled_at.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Timer(0.4, send_sigint_once_the_child_runs)
    interrupter.start()

    try:
        with pytest.raises(KeyboardInterrupt):
            _run(
                _python(_CHILD_WITH_GRANDCHILD, str(pid_file), "6"),
                tmp_path / "child.log",
            )
        interrupted_after = time.monotonic() - signalled_at[0]
        pids = _read_pids(pid_file)
        survivors = [pid for pid in pids if not _wait_until_gone(pid)]
    finally:
        stop.set()
        interrupter.cancel()
        interrupter.join()
        if pid_file.exists():
            _kill_quietly(_read_pids(pid_file))

    assert interrupted_after < 2.0, (
        f"the interrupt was raised {interrupted_after:.2f} s after the signal"
    )
    assert survivors == [], f"processes still running after the interrupt: {survivors}"


# ---- SIGTERM and SIGHUP -----------------------------------------------------------
#
# The child has a session of its own, so a signal sent to `rinchen` (a `kill`, a closed
# terminal, a service stop) never reaches it: without a handler the parent dies and
# leaves an `eole train` running for hours with nobody to read its result.

_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _sentinel_handler(signum: int, frame: object) -> None:
    raise AssertionError(f"signal {signum} reached a handler that was not replaced")


def _handlers() -> dict[int, Any]:
    return {number: signal.getsignal(number) for number in _TERMINATION_SIGNALS}


@pytest.fixture
def sentinel_handlers() -> Iterator[dict[int, Any]]:
    """Handlers of the test's own for SIGTERM and SIGHUP, so that a `run_logged` that
    installs none of its own fails the test (the sentinel raises) instead of killing
    the pytest process; restored afterwards."""
    previous = {
        number: signal.signal(number, _sentinel_handler)
        for number in _TERMINATION_SIGNALS
    }
    try:
        yield _handlers()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@linux_only
@pytest.mark.timeout(30)
@pytest.mark.parametrize("signum", _TERMINATION_SIGNALS, ids=["SIGTERM", "SIGHUP"])
def test_a_real_termination_signal_ends_the_run_and_kills_the_process_group(
    tmp_path: Path, sentinel_handlers: dict[int, Any], signum: signal.Signals
) -> None:
    importlib.import_module("polars")  # as in every `rinchen` command
    pid_file = tmp_path / "pids"
    signalled_at: list[float] = []
    stop = threading.Event()

    def send_the_signal_once_the_child_runs() -> None:
        while not pid_file.exists():  # a loaded machine may start the child late
            if stop.wait(0.02):
                return
        signalled_at.append(time.monotonic())
        os.kill(os.getpid(), signum)

    sender = threading.Timer(0.4, send_the_signal_once_the_child_runs)
    sender.start()

    try:
        with pytest.raises(SystemExit) as raised:
            _run(
                _python(_CHILD_WITH_GRANDCHILD, str(pid_file), "6"),
                tmp_path / "child.log",
            )
        ended_after = time.monotonic() - signalled_at[0]
        pids = _read_pids(pid_file)
        survivors = [pid for pid in pids if not _wait_until_gone(pid)]
    finally:
        stop.set()
        sender.cancel()
        sender.join()
        if pid_file.exists():
            _kill_quietly(_read_pids(pid_file))

    # The shell's convention for a process ended by a signal: 128 + its number.
    assert raised.value.code == 128 + signum
    assert ended_after < 1.5, f"the run ended {ended_after:.2f} s after the signal"
    assert survivors == [], f"processes still running after the signal: {survivors}"
    assert _handlers() == sentinel_handlers


@pytest.mark.parametrize("scenario", ["success", "failure", "timeout", "no command"])
def test_the_handlers_of_the_caller_are_back_after_every_way_a_run_can_end(
    tmp_path: Path, sentinel_handlers: dict[int, Any], scenario: str
) -> None:
    code = {
        "success": "print('fine')",
        "failure": "import sys; sys.exit(3)",
        "timeout": "import time; time.sleep(5)",
        "no command": "",
    }[scenario]
    command = _python(code)
    if scenario == "no command":
        command = [str(tmp_path / "no-such-command")]
    expected = {
        "success": None,
        "failure": UpstreamError,
        "timeout": UpstreamError,
        "no command": FileNotFoundError,
    }[scenario]

    if expected is None:
        _run(command, tmp_path / "child.log")
    else:
        with pytest.raises(expected):
            _run(
                command,
                tmp_path / "child.log",
                timeout=0.3 if scenario == "timeout" else None,
            )

    assert _handlers() == sentinel_handlers


def test_run_logged_does_not_touch_the_handlers_off_the_main_thread(
    tmp_path: Path, sentinel_handlers: dict[int, Any]
) -> None:
    # Handlers can only be set from the main thread; a run in a worker thread (a test,
    # a future caller) must neither fail on that nor change them.
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(_run, _python("print('fine')"), tmp_path / "child.log")
        worker.result(timeout=30)  # raises what the worker raised

    assert (tmp_path / "child.log").read_text(encoding="utf-8") == "fine\n"
    assert _handlers() == sentinel_handlers


@pytest.mark.parametrize("timeout", [None, 30.0])
def test_run_logged_waits_through_several_polls_for_a_child_that_fails_late(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None
) -> None:
    monkeypatch.setattr(logged_process, "_INTERRUPT_POLL_S", 0.1)
    code = "import sys, time; time.sleep(0.6); print('finished late'); sys.exit(3)"

    with pytest.raises(UpstreamError) as raised:
        _run(_python(code), tmp_path / "child.log", timeout=timeout)

    message = str(raised.value)
    assert message.startswith("the child failed: exit code 3;")
    assert message.endswith("finished late")


@pytest.mark.parametrize("poll_s", [0.2, 5.0])
@pytest.mark.parametrize(("timeout", "wording"), [(0, "0"), (0.5, "0.5")])
def test_run_logged_stops_waiting_at_its_timeout_whatever_the_poll(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    poll_s: float,
    timeout: float,
    wording: str,
) -> None:
    # A slice longer than what is left of the timeout would let the child run on:
    # this one would return normally after 3 s.
    monkeypatch.setattr(logged_process, "_INTERRUPT_POLL_S", poll_s)
    started = time.monotonic()

    with pytest.raises(UpstreamError) as raised:
        _run(
            _python("import time; time.sleep(3)"),
            tmp_path / "child.log",
            timeout=timeout,
        )

    assert f"the child timed out after {wording} s and was killed" in str(raised.value)
    assert time.monotonic() - started < 2.5
