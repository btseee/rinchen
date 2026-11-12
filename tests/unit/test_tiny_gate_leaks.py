"""The leak check of the interrupt controls of the D15 gate (`tests/support/tiny_gate_processes.py`,
`left_after_the_end`), tested over a made-up `/proc` and a made-up clock: no process is started, no signal
is sent and no test waits.

The check reads the process table at once, after the command has ended and been reaped, and allows a grace of at
most 1.5 s for a process that is being torn down. A zombie is not in the count at all (its state is `Z`), and
a process that is gone by the end of the grace was dying; one that is alive then is a leak. The first version
of the control waited 10 s and passed whenever an orphan of the run ended by itself within that time: the
vocabulary build that D14's runner left behind ran for 5 to 15 s.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support import tiny_gate_runs
from tests.support.tiny_gate_processes import (
    REAPING_GRACE_SECONDS,
    RunProcess,
    left_after_the_end,
)

MARKER: Final = "RINCHEN_GATE_RUN=abc123"
RUNS: Final = REPO_ROOT / "tests" / "support" / "tiny_gate_runs.py"


def _process(proc: Path, pid: int, *, session: int, state: str = "S") -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "stat").write_text(
        f"{pid} (eole) {state} 1 {pid} {session} 0 -1 4194560\n", encoding="utf-8"
    )
    (directory / "environ").write_bytes(MARKER.encode() + b"\0")
    (directory / "cmdline").write_bytes(b"eole\0build_vocab\0")


class Timeline:
    """A clock that only moves when the check waits, and a `/proc` in which processes end at given times."""

    def __init__(self, proc: Path) -> None:
        self.proc = proc
        self.now = 0.0
        self.sleeps: list[float] = []
        self._ends: list[tuple[float, int]] = []

    def ends_at(self, seconds: float, pid: int) -> None:
        self._ends.append((seconds, pid))

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        for when, pid in self._ends:
            if when <= self.now and (self.proc / str(pid)).exists():
                shutil.rmtree(self.proc / str(pid))

    def left(self, **options: float) -> list[RunProcess]:
        return left_after_the_end(
            "abc123",
            proc=self.proc,
            clock=self.clock,
            sleep=self.sleep,
            **options,
        )


@pytest.fixture
def timeline(tmp_path: Path) -> Timeline:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "self").mkdir()
    return Timeline(proc)


def test_the_grace_is_at_most_a_second_and_a_half() -> None:
    assert 0 < REAPING_GRACE_SECONDS <= 1.5


def test_nothing_left_is_known_at_once_and_nothing_is_waited_for(
    timeline: Timeline,
) -> None:
    assert timeline.left() == []
    assert timeline.sleeps == []
    assert timeline.now == 0.0


def test_a_process_that_ends_three_seconds_after_the_command_is_a_leak(
    timeline: Timeline,
) -> None:
    _process(timeline.proc, 90, session=90)
    timeline.ends_at(3.0, 90)

    left = timeline.left()

    assert [item.pid for item in left] == [90]
    assert left[0].command == "eole build_vocab"
    # It did not wait for the process to end by itself.
    assert timeline.now <= REAPING_GRACE_SECONDS + 0.1


@pytest.mark.parametrize("seconds", [0.1, 0.5, 1.0])
def test_a_process_that_ends_within_the_grace_was_dying_and_is_not_a_leak(
    timeline: Timeline, seconds: float
) -> None:
    _process(timeline.proc, 91, session=91)
    timeline.ends_at(seconds, 91)

    assert timeline.left() == []
    assert timeline.now <= REAPING_GRACE_SECONDS + 0.1


@pytest.mark.parametrize("seconds", [2.0, 5.0, 10.0, 15.0])
def test_a_process_that_ends_after_the_grace_is_a_leak_however_soon_it_ends_by_itself(
    timeline: Timeline, seconds: float
) -> None:
    _process(timeline.proc, 92, session=92)
    timeline.ends_at(seconds, 92)

    assert [item.pid for item in timeline.left()] == [92]


def test_a_zombie_has_ended_and_is_not_a_leak_whatever_the_grace(
    timeline: Timeline,
) -> None:
    _process(timeline.proc, 93, session=93, state="Z")

    assert timeline.left() == []
    assert timeline.sleeps == []


def test_what_is_returned_is_what_is_alive_when_the_grace_is_over(
    timeline: Timeline,
) -> None:
    _process(timeline.proc, 94, session=94)
    _process(timeline.proc, 95, session=95)
    timeline.ends_at(0.5, 94)

    assert [item.pid for item in timeline.left()] == [95]


def test_a_process_of_the_session_of_the_run_is_found_without_the_marker(
    timeline: Timeline,
) -> None:
    directory = timeline.proc / "96"
    directory.mkdir()
    (directory / "stat").write_text("96 (sh) S 1 96 77 0 -1 0\n", encoding="utf-8")

    found = left_after_the_end(
        "abc123",
        session=77,
        proc=timeline.proc,
        clock=timeline.clock,
        sleep=timeline.sleep,
    )

    assert [item.pid for item in found] == [96]


def test_with_no_grace_the_table_is_read_once_and_what_is_there_is_returned(
    timeline: Timeline,
) -> None:
    _process(timeline.proc, 97, session=97)
    timeline.ends_at(0.01, 97)

    assert [item.pid for item in timeline.left(grace=0)] == [97]
    assert timeline.sleeps == []


@pytest.mark.parametrize("grace", [1.51, 2, 10, -0.1])
def test_a_grace_beyond_the_bound_is_refused_so_that_the_wait_cannot_grow_back(
    timeline: Timeline, grace: float
) -> None:
    with pytest.raises(ValueError, match="grace"):
        timeline.left(grace=grace)


def test_the_poll_is_short_against_the_grace(timeline: Timeline) -> None:
    _process(timeline.proc, 98, session=98)

    timeline.left()

    assert timeline.sleeps
    assert max(timeline.sleeps) <= REAPING_GRACE_SECONDS / 10


# ---- the runs that use the check -------------------------------------------------------------------------


def _calls_of(source: str, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_both_runs_read_what_is_left_through_the_check_and_name_no_wait_of_their_own() -> (
    None
):
    source = RUNS.read_text(encoding="utf-8")
    calls = _calls_of(source, "left_after_the_end")

    assert tiny_gate_runs.left_after_the_end is left_after_the_end
    # `interrupt_run` and `run_with_budget`, each with the marker and the session and no grace of its own.
    assert len(calls) == 2
    for call in calls:
        assert {keyword.arg for keyword in call.keywords} == {"session"}
    assert "wait_until_none_left" not in source
    assert not hasattr(tiny_gate_runs, "LEFT_OVER_SECONDS")


def test_the_only_sleep_of_the_runs_is_the_one_before_the_signal() -> None:
    """A pause between the end of the command and the read of the table would give an orphan the time to
    end by itself, as the old wait did."""
    functions = [
        node
        for node in ast.walk(ast.parse(RUNS.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    ]
    sleepers = {
        function.name
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sleep"
    }

    assert sleepers == {"_settle"}
