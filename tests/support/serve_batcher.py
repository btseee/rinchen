"""A `run` for the micro-batcher tests (D11): it records every batch it is given, counts how
many run at once, answers each item with `<key>:<item>`, and can hold batches on a gate or
fail them, so a test decides which batches are in flight and which are queued."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager

from rinchen.serve.batcher import MicroBatcher

# A held batch that nobody releases is a bug of the test: it fails after this many seconds
# instead of hanging the suite.
RELEASE_TIMEOUT = 30.0


class BatchProbe:
    """The `run(key, items)` of a batcher under test.

    `hold` is `True` (every batch waits for `release()`), a set of keys (only their batches
    do) or `False`. `fail_on` is a set of items: a batch holding one raises `ValueError`.
    `short_on` is a set of items too: a batch holding one returns one result too few."""

    def __init__(
        self,
        *,
        hold: bool | Iterable[str] = False,
        fail_on: Iterable[str] = (),
        short_on: Iterable[str] = (),
    ) -> None:
        self._condition = threading.Condition()
        self._gate = threading.Event()
        self._held_keys = None if hold is True else frozenset(hold or ())
        if hold is False:
            self._gate.set()
        self._fail_on = frozenset(fail_on)
        self._short_on = frozenset(short_on)
        self.calls: list[tuple[str, list[str]]] = []
        self.active = 0
        self.peak = 0
        self.finished = 0

    def __call__(self, key: str, items: list[str]) -> list[str]:
        with self._condition:
            self.calls.append((key, list(items)))
            self.active += 1
            self.peak = max(self.peak, self.active)
            self._condition.notify_all()
        try:
            if self._held(key) and not self._gate.wait(RELEASE_TIMEOUT):
                raise AssertionError("a held batch was never released")
            if self._fail_on.intersection(items):
                raise ValueError("a scripted failure")
            answers = [f"{key}:{item}" for item in items]
            return answers[:-1] if self._short_on.intersection(items) else answers
        finally:
            with self._condition:
                self.active -= 1
                self.finished += 1
                self._condition.notify_all()

    def _held(self, key: str) -> bool:
        return self._held_keys is None or key in self._held_keys

    def release(self) -> None:
        self._gate.set()

    def wait_for_calls(self, count: int) -> bool:
        """Block (up to `RELEASE_TIMEOUT`) until `count` batches have started."""
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.calls) >= count, RELEASE_TIMEOUT
            )

    def wait_for_finished(self, count: int) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self.finished >= count, RELEASE_TIMEOUT
            )

    def items_seen(self) -> list[str]:
        return [item for _, items in self.calls for item in items]


def make_batcher(
    run: Callable[[str, list[str]], list[str]], **options: float
) -> MicroBatcher[str, str, str]:
    settings: dict[str, float] = {
        "max_batch_sentences": 4,
        "max_wait_seconds": 0.05,
        "max_queue_sentences": 100,
        "workers": 2,
    }
    settings.update(options)
    return MicroBatcher(
        run,
        max_batch_sentences=int(settings["max_batch_sentences"]),
        max_wait_seconds=settings["max_wait_seconds"],
        max_queue_sentences=int(settings["max_queue_sentences"]),
        workers=int(settings["workers"]),
    )


@asynccontextmanager
async def batching(
    probe: BatchProbe, **options: float
) -> AsyncIterator[MicroBatcher[str, str, str]]:
    """A batcher over `probe` that is released and closed however the test ends."""
    batcher = make_batcher(probe, **options)
    try:
        yield batcher
    finally:
        probe.release()
        await batcher.close(10.0)


async def until(condition: Callable[[], bool], seconds: float = 10.0) -> None:
    """Wait (bounded) for `condition` to become true by yielding to the loop."""
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.001)


async def started(probe: BatchProbe, count: int) -> None:
    """Wait (bounded, off the loop) until `probe` has been given `count` batches."""
    assert await asyncio.to_thread(probe.wait_for_calls, count), (
        f"fewer than {count} batches started"
    )
