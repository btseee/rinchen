"""The micro-batcher of the service (D11, T16).

Requests arrive one at a time and a decoder is fastest on a batch, so the batcher queues the
items of every request per `key` (the decode options CTranslate2 takes once per call: a batch
never mixes keys) and a dispatcher task hands a key's items to a thread pool as one batch:

* when the key holds `max_batch_sentences` items, or its oldest item has waited
  `max_wait_seconds`, whichever comes first, and only while fewer than `workers` batches are
  in flight, so under load the queue grows into larger batches by itself;
* at once, whatever the wait, once `close` began.

`await submit(key, items)` answers one result per item, in order, or raises: `QueueFull` when
the request does not fit the queue whole (nothing of it is queued: all or none), `Draining`
once `close` began, and what `run` raised for a batch that held one of its items. `run(key,
items)` runs in a worker thread and returns one result per item; any other count fails that
batch with `UpstreamError`. The batcher is generic: it knows nothing of an engine, a request or a
text, and nothing here puts an item in a message.

Cancelling a submitter drops the items of its request that are still queued, and an item that
is already in a batch is run with its batch and its result is discarded: a batch is never
split or recalled. All state is touched on the event loop's thread only.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

from rinchen.errors import ConfigError, RinchenError, UpstreamError
from rinchen.utils.integers import is_strict_int

# A timer may fire a hair before its time by the loop's clock resolution; a deadline this close
# counts as reached, so the dispatcher does not wake again for it.
_CLOCK_SLACK_SECONDS = 1e-6


class QueueFull(RinchenError):
    """The request does not fit the queue: it would hold more sentences than the limit."""


class Draining(RinchenError):
    """The batcher was closed: it takes no new work, or the work was given up on."""


@dataclass(eq=False, slots=True)
class _Entry[K, I, R]:
    """One item of a request: its key, the item, the future its submitter waits on, when
    it was queued, and whether it is still counted in the queue."""

    key: K
    item: I
    future: asyncio.Future[R]
    queued_at: float
    queued: bool = True


class MicroBatcher[K: Hashable, I, R]:
    """Batches the items of concurrent requests for `run`, a blocking function of a key and
    a list of items that returns one result per item (see the module's text).

    Raises:
        ConfigError: a count is not an integer of at least 1, or `max_wait_seconds` is not a
            finite number above 0.
    """

    def __init__(
        self,
        run: Callable[[K, list[I]], Sequence[R]],
        *,
        max_batch_sentences: int,
        max_wait_seconds: float,
        max_queue_sentences: int,
        workers: int,
    ) -> None:
        for name, value in (
            ("max_batch_sentences", max_batch_sentences),
            ("max_queue_sentences", max_queue_sentences),
            ("workers", workers),
        ):
            if not is_strict_int(value) or value < 1:
                raise ConfigError(
                    f"{name} must be an integer of at least 1, got {value!r}"
                )
        if (
            isinstance(max_wait_seconds, bool)
            or not isinstance(max_wait_seconds, int | float)
            or not math.isfinite(max_wait_seconds)
            or max_wait_seconds <= 0
        ):
            raise ConfigError(
                f"max_wait_seconds must be a finite number above 0, got {max_wait_seconds!r}"
            )
        self._run = run
        self._max_batch = max_batch_sentences
        self._max_wait = float(max_wait_seconds)
        self._max_queue = max_queue_sentences
        self._workers = workers
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rinchen-batch"
        )
        self._queues: dict[K, deque[_Entry[K, I, R]]] = {}
        self._depth = 0
        self._flights: dict[asyncio.Future[Sequence[R]], list[_Entry[K, I, R]]] = {}
        self._wake = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._closing = False
        self._closer: asyncio.Task[int] | None = None

    @property
    def depth(self) -> int:
        """The items queued and not yet in a batch."""
        return self._depth

    @property
    def inflight(self) -> int:
        """The batches handed to the workers and not yet finished."""
        return len(self._flights)

    async def submit(self, key: K, items: Sequence[I]) -> list[R]:
        """Queue `items` under `key` and wait for their results, in order.

        Raises:
            Draining: `close` began.
            QueueFull: the queue cannot hold all of `items`; none was queued.
            Exception: what `run` raised for a batch that held an item of this request.
        """
        if self._closing:
            raise Draining("the service is shutting down and takes no new work")
        if not items:
            return []
        if self._depth + len(items) > self._max_queue:
            raise QueueFull(
                f"{len(items)} sentences do not fit: the queue holds {self._depth} of "
                f"{self._max_queue}"
            )
        loop = asyncio.get_running_loop()
        if self._dispatcher is None:
            self._dispatcher = loop.create_task(
                self._dispatch(), name="rinchen-micro-batcher"
            )
        now = loop.time()
        entries = [_Entry(key, item, loop.create_future(), now) for item in items]
        self._queues.setdefault(key, deque()).extend(entries)
        self._depth += len(entries)
        self._wake.set()
        try:
            outcomes = await asyncio.gather(
                *(entry.future for entry in entries), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._drop_queued(entries)
            raise
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return cast("list[R]", outcomes)

    async def close(self, timeout: float) -> int:
        """Stop taking work, run what is queued at once, wait up to `timeout` seconds for
        everything to finish, and return the number of items that were still unanswered then
        and were failed with `Draining` (queued ones and the ones in a batch still running:
        their results are discarded). Calling it again returns the same number."""
        if self._closer is None:
            self._closing = True
            self._wake.set()
            self._closer = asyncio.get_running_loop().create_task(self._finish(timeout))
        return await asyncio.shield(self._closer)

    async def _finish(self, timeout: float) -> int:
        dispatcher = self._dispatcher
        if dispatcher is not None:
            await asyncio.wait({dispatcher}, timeout=timeout)
        unanswered = self._give_up_on_the_rest()
        # A batch still running keeps its thread; the loop is not made to wait for it.
        idle = not self._flights
        self._pool.shutdown(wait=idle, cancel_futures=not idle)
        return unanswered

    def _give_up_on_the_rest(self) -> int:
        """Fail every item still unanswered with `Draining`; stop the dispatcher."""
        if self._dispatcher is not None and not self._dispatcher.done():
            self._dispatcher.cancel()
        entries = [e for queue in self._queues.values() for e in queue]
        entries += [e for batch in self._flights.values() for e in batch]
        self._queues.clear()
        self._depth = 0
        for entry in entries:
            entry.queued = False
        return self._fail(entries, Draining("the service stopped before this was done"))

    async def _dispatch(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._start_ready_batches(loop)
            if self._closing and not self._depth and not self._flights:
                return
            self._wake.clear()
            try:
                await asyncio.wait_for(
                    self._wake.wait(), self._until_next_deadline(loop)
                )
            except TimeoutError:
                pass

    def _start_ready_batches(self, loop: asyncio.AbstractEventLoop) -> None:
        while len(self._flights) < self._workers:
            key = self._oldest_ready_key(loop.time())
            if key is None:
                return
            self._start_batch(loop, key)

    def _oldest_ready_key(self, now: float) -> K | None:
        """The key whose oldest item has waited longest among those ready to go out: full,
        past their wait, or any when closing."""
        chosen: K | None = None
        chosen_since = math.inf
        for key, queue in self._queues.items():
            since = queue[0].queued_at
            ready = (
                self._closing
                or len(queue) >= self._max_batch
                or now >= since + self._max_wait - _CLOCK_SLACK_SECONDS
            )
            if ready and since < chosen_since:
                chosen, chosen_since = key, since
        return chosen

    def _until_next_deadline(self, loop: asyncio.AbstractEventLoop) -> float | None:
        """Seconds until a queued key is due, or `None` when waiting for a wake-up is all
        there is to do (nothing queued, or every worker busy: a finished batch wakes us)."""
        if not self._queues or len(self._flights) >= self._workers:
            return None
        due = (
            min(queue[0].queued_at for queue in self._queues.values()) + self._max_wait
        )
        return max(0.0, due - loop.time())

    def _start_batch(self, loop: asyncio.AbstractEventLoop, key: K) -> None:
        queue = self._queues[key]
        entries: list[_Entry[K, I, R]] = []
        while queue and len(entries) < self._max_batch:
            entry = queue.popleft()
            entry.queued = False
            entries.append(entry)
        if not queue:
            del self._queues[key]
        self._depth -= len(entries)
        try:
            flight = loop.run_in_executor(
                self._pool, self._run, key, [entry.item for entry in entries]
            )
        except RuntimeError as error:
            # A pool that was shut down refuses work: fail the batch rather than lose it.
            self._fail(entries, error)
            return
        self._flights[flight] = entries
        flight.add_done_callback(self._on_flight_done)

    def _on_flight_done(self, flight: asyncio.Future[Sequence[R]]) -> None:
        entries = self._flights.pop(flight)
        if flight.cancelled():
            self._fail(entries, Draining("the service stopped before this was done"))
        elif (error := flight.exception()) is not None:
            self._fail(entries, error)
        else:
            results = flight.result()
            if len(results) != len(entries):
                self._fail(
                    entries,
                    UpstreamError(
                        f"a batch of {len(entries)} items came back with "
                        f"{len(results)} results"
                    ),
                )
            else:
                for entry, result in zip(entries, results, strict=True):
                    if not entry.future.done():
                        entry.future.set_result(result)
        self._wake.set()

    def _fail(self, entries: Sequence[_Entry[K, I, R]], error: BaseException) -> int:
        """Fail the entries that are still waiting; return how many that was."""
        failed = 0
        for entry in entries:
            if not entry.future.done():
                entry.future.set_exception(error)
                failed += 1
        return failed

    def _drop_queued(self, entries: Sequence[_Entry[K, I, R]]) -> None:
        """Take the items of a cancelled request that are still queued out of the queue."""
        for entry in entries:
            if not entry.queued:
                continue
            queue = self._queues[entry.key]
            queue.remove(entry)
            if not queue:
                del self._queues[entry.key]
            entry.queued = False
            self._depth -= 1
        self._wake.set()
