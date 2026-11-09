"""The engine from many threads at once (D11, ruling T11): every result is the one its own
rows earned, the counters are the exact sums of what the calls did (no update is lost), and a
suppression list is built once however many threads ask for it at the same moment.

The library stand-in sleeps a little inside every call, so the threads really overlap; the
interpreter is told to switch threads very often, which makes a lost update likely when the
engine does not hold its lock.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rinchen.infer import engine as engine_module
from rinchen.infer.engine_types import DecodeOptions, Row, Translation
from tests.support.engines import (
    END,
    ENG,
    KHK,
    SP,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    hyp,
    scripted_engine,
)
from tests.support.fake_ct2 import Hypothesis

THREADS = 8
CALLS = 50
OFF = DecodeOptions(script_mask="off")

pytestmark = pytest.mark.usefixtures("engine_environment")


@pytest.fixture
def switching_often() -> Iterator[None]:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def slow_echo(line: tuple[str, ...]) -> Sequence[Hypothesis]:
    """The echo of the source; the word `cut` is answered without the end token and the word
    `bad` with a reserved piece."""
    time.sleep(0.0005)
    words = line[1:]
    if SP + "cut" in words:
        return [hyp(*words)]
    if SP + "bad" in words:
        return [hyp(*words, "<bt>", END)]
    return [hyp(*words, END)]


def run_threads(work: Any) -> list[Any]:
    barrier = threading.Barrier(THREADS)
    results: list[Any] = [None] * THREADS
    failures: list[BaseException] = []

    def target(number: int) -> None:
        try:
            barrier.wait()
            results[number] = work(number)
        except BaseException as error:  # noqa: BLE001 - reported to the test below
            failures.append(error)

    threads = [threading.Thread(target=target, args=(n,)) for n in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not failures, failures
    return results


def test_every_thread_gets_its_own_results_and_the_counters_are_the_exact_sums(
    tmp_path: Path, switching_often: None
) -> None:
    fixture = scripted_engine(tmp_path, script=slow_echo)

    def work(number: int) -> list[list[Translation]]:
        out = []
        for call in range(CALLS):
            words = f"w{number}x{call}"
            rows = [
                Row(words, None, ENG),
                Row(f"cut {words}", None, ENG),
                Row(f"bad {words}", None, ENG),
                Row("", None, ENG),
            ]
            out.append(fixture.engine.translate(rows, OFF))
        return out

    results = run_threads(work)

    for number, per_thread in enumerate(results):
        for call, translations in enumerate(per_thread):
            words = f"w{number}x{call}"
            assert [(t.status, t.text) for t in translations] == [
                ("ok", words),
                ("truncated", f"cut {words}"),
                ("no_output", ""),
                ("empty", ""),
            ]
    total = THREADS * CALLS
    counters = fixture.engine.counters()
    assert counters["sources"] == 4 * total
    assert counters["truncated"] == total
    assert counters["no_output"] == total
    assert counters["empty"] == total
    assert counters["reserved_symbol"] == total
    assert sum(counters.values()) == 4 * total + 4 * total
    assert len(fixture.ct2.calls) == total


def test_the_retry_counters_are_exact_too(
    tmp_path: Path, switching_often: None
) -> None:
    latin = SP + "hello"

    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        time.sleep(0.0005)
        return [hyp(SP + "Энэ", END)] if latin in suppressed else [hyp(latin, END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    def work(number: int) -> None:
        for _ in range(CALLS):
            fixture.engine.translate([Row("x", None, KHK), Row("y", None, KHK)])

    run_threads(work)

    counters = fixture.engine.counters()
    assert counters["sources"] == 2 * THREADS * CALLS
    assert counters["script_retries"] == 2 * THREADS * CALLS
    assert counters["script_retry_improved"] == 2 * THREADS * CALLS


def test_a_suppression_list_is_built_once_when_every_thread_asks_at_the_same_moment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    real = engine_module.suppression_list

    def slow(*args: Any, **kwargs: Any) -> list[list[str]]:
        time.sleep(0.05)
        built.append(args[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "suppression_list", slow)
    fixture = scripted_engine(tmp_path, script=slow_echo)

    def work(number: int) -> list[Translation]:
        return fixture.engine.translate(
            [Row("hello", None, ENG)], DecodeOptions(script_mask="always")
        )

    results = run_threads(work)

    assert built == [ENG]
    assert all(r[0].text == "hello" for r in results)
    lists = {
        tuple(map(tuple, call.suppress_sequences or [])) for call in fixture.ct2.calls
    }
    assert len(lists) == 1


def test_a_failing_thread_adds_nothing_while_the_others_count(
    tmp_path: Path, switching_often: None
) -> None:
    fixture = scripted_engine(tmp_path, script=slow_echo)

    def work(number: int) -> int:
        failed = 0
        for call in range(CALLS):
            if number == 0 and call % 2 == 0:
                try:
                    fixture.engine.translate([Row("ok", None, "deu_Latn")], OFF)
                except Exception:  # noqa: BLE001 - the refusal is the point
                    failed += 1
                continue
            fixture.engine.translate([Row("ok", None, ENG)], OFF)
        return failed

    results = run_threads(work)

    assert results[0] == CALLS // 2
    expected = THREADS * CALLS - CALLS // 2
    assert fixture.engine.counters()["sources"] == expected


class YieldingCounters(dict[str, int]):
    """The counters dict, handing the interpreter to another thread between a read and the
    write that follows it, as a preemption would."""

    def __getitem__(self, key: str) -> int:
        value = super().__getitem__(key)
        time.sleep(0.0002)
        return value


def test_no_update_of_a_counter_is_lost_between_its_read_and_its_write(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, script=slow_echo)
    # The structure the engine's lock protects: swapped for one that gives the thread away.
    fixture.engine._counters = YieldingCounters(fixture.engine.counters())

    def work(number: int) -> None:
        for _ in range(5):
            fixture.engine.translate([Row("hello", None, ENG)], OFF)

    run_threads(work)

    assert fixture.engine.counters()["sources"] == THREADS * 5


class OverlapDetectingConverter:
    """A script converter that notes whether two calls ever ran at the same time (D5's
    converter keeps a cache in the instance and is not thread-safe)."""

    def __init__(self) -> None:
        self.running = 0
        self.overlapped = False
        self.calls = 0

    def convert(self, text: str, target: str) -> SimpleNamespace:
        self.running += 1
        if self.running > 1:
            self.overlapped = True
        time.sleep(0.001)
        self.calls += 1
        self.running -= 1
        return SimpleNamespace(text=text, rule_fallbacks=0, invalid_outputs=0)


def test_the_script_converter_is_never_called_by_two_threads_at_once(
    tmp_path: Path, switching_often: None
) -> None:
    converter = OverlapDetectingConverter()
    fixture = scripted_engine(tmp_path, converter=converter)  # type: ignore[arg-type]

    def work(number: int) -> None:
        for _ in range(5):
            fixture.engine.translate([Row("hello", ENG, KHK, "Mong")], OFF)

    run_threads(work)

    assert converter.overlapped is False
    assert converter.calls == THREADS * 5
    assert fixture.engine.counters()["script_conversions"] == THREADS * 5
