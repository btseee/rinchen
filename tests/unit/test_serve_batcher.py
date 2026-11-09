"""The micro-batcher of the service (D11, T16), on a real event loop and real threads: when a
batch goes out, who shares one, how many run at once, what a full queue, a failure, a
cancellation and a close do, and that every item is answered once, to its own submitter."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.serve.batcher import Draining, MicroBatcher, QueueFull
from tests.support.serve_batcher import (
    BatchProbe,
    batching,
    make_batcher,
    started,
    until,
)

# Long enough that a batch which goes out only when it is full, or when the batcher is
# closing, cannot have waited for it.
FOREVER = 60.0


def run_scenario(scenario: Callable[[], Awaitable[None]]) -> None:
    async def bounded() -> None:
        async with asyncio.timeout(60):
            await scenario()

    asyncio.run(bounded())


def test_a_full_batch_goes_out_before_the_wait_is_over() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(
            probe, max_batch_sentences=4, max_wait_seconds=FOREVER
        ) as b:
            async with asyncio.timeout(10):
                result = await b.submit("k", ["a", "b", "c", "d"])

        assert result == ["k:a", "k:b", "k:c", "k:d"]
        assert probe.calls == [("k", ["a", "b", "c", "d"])]

    run_scenario(scenario)


def test_a_lone_item_waits_the_wait_and_not_much_longer() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.2) as b:
            began = time.perf_counter()
            result = await b.submit("k", ["a"])
            waited = time.perf_counter() - began

        assert result == ["k:a"]
        assert 0.2 - 0.001 <= waited <= 0.2 + 2.0

    run_scenario(scenario)


def test_the_wait_is_counted_from_the_oldest_item_of_a_key() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.6) as b:
            began = time.perf_counter()
            first = asyncio.create_task(b.submit("k", ["a"]))
            await asyncio.sleep(0.4)
            second = asyncio.create_task(b.submit("k", ["b"]))
            assert await first == ["k:a"]
            first_waited = time.perf_counter() - began
            assert await second == ["k:b"]

        assert probe.calls == [("k", ["a", "b"])]
        # A wait restarted by the second arrival would end near 1.0 seconds.
        assert 0.6 - 0.001 <= first_waited <= 0.9

    run_scenario(scenario)


def test_items_of_different_keys_never_share_a_batch() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_batch_sentences=8, max_wait_seconds=0.05) as b:
            first, second = await asyncio.gather(
                b.submit("a", ["a1", "a2"]), b.submit("b", ["b1", "b2"])
            )

        assert first == ["a:a1", "a:a2"]
        assert second == ["b:b1", "b:b2"]
        assert sorted(probe.calls) == [("a", ["a1", "a2"]), ("b", ["b1", "b2"])]

    run_scenario(scenario)


def test_requests_of_one_key_that_arrive_together_share_a_batch() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_batch_sentences=8, max_wait_seconds=0.05) as b:
            first, second = await asyncio.gather(
                b.submit("k", ["a1", "a2"]), b.submit("k", ["b1", "b2", "b3"])
            )

        assert first == ["k:a1", "k:a2"]
        assert second == ["k:b1", "k:b2", "k:b3"]
        assert probe.calls == [("k", ["a1", "a2", "b1", "b2", "b3"])]

    run_scenario(scenario)


@pytest.mark.parametrize(
    ("queued", "sizes"), [(5, [5]), (8, [8]), (12, [8, 4]), (20, [8, 8, 4])]
)
def test_under_a_held_worker_the_queue_grows_into_larger_batches(
    queued: int, sizes: list[int]
) -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(
            probe, workers=1, max_batch_sentences=8, max_wait_seconds=0.02
        ) as b:
            first = asyncio.create_task(b.submit("k", ["first"]))
            await started(probe, 1)
            later = [
                asyncio.create_task(b.submit("k", [f"x{index}"]))
                for index in range(queued)
            ]
            # Many times the wait: a batch would have gone out had a worker been free.
            await asyncio.sleep(0.15)
            assert (b.depth, b.inflight, len(probe.calls)) == (queued, 1, 1)

            probe.release()
            assert await first == ["k:first"]
            results = [await task for task in later]

        assert results == [[f"k:x{index}"] for index in range(queued)]
        assert [len(items) for _, items in probe.calls[1:]] == sizes
        assert probe.items_seen()[1:] == [f"x{index}" for index in range(queued)]

    run_scenario(scenario)


def test_at_most_workers_batches_run_at_once() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(
            probe, workers=2, max_batch_sentences=1, max_wait_seconds=0.01
        ) as b:
            tasks = [
                asyncio.create_task(b.submit("k", [f"i{index}"])) for index in range(6)
            ]
            await started(probe, 2)
            await asyncio.sleep(0.1)
            assert (probe.peak, len(probe.calls)) == (2, 2)
            assert (b.inflight, b.depth) == (2, 4)

            probe.release()
            results = [await task for task in tasks]

        assert results == [[f"k:i{index}"] for index in range(6)]
        assert probe.peak == 2

    run_scenario(scenario)


def test_the_results_come_back_to_the_right_submitter_whatever_finishes_first() -> None:
    probe = BatchProbe(hold={"slow"})

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.01) as b:
            slow = asyncio.create_task(b.submit("slow", ["s1", "s2"]))
            await started(probe, 1)
            fast = asyncio.create_task(b.submit("fast", ["f1", "f2", "f3"]))

            assert await fast == ["fast:f1", "fast:f2", "fast:f3"]
            assert not slow.done()
            probe.release()
            assert await slow == ["slow:s1", "slow:s2"]

    run_scenario(scenario)


def test_a_request_of_more_than_a_batch_is_split_and_answered_whole_in_order() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(
            probe, max_batch_sentences=4, workers=2, max_wait_seconds=0.01
        ) as b:
            items = [f"i{index}" for index in range(10)]
            result = await b.submit("k", items)

        assert result == [f"k:{item}" for item in items]
        assert sorted(len(batch) for _, batch in probe.calls) == [2, 4, 4]

    run_scenario(scenario)


def test_a_request_that_does_not_fit_the_queue_is_refused_whole() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(
            probe,
            workers=1,
            max_batch_sentences=1,
            max_queue_sentences=5,
            max_wait_seconds=0.01,
        ) as b:
            running = asyncio.create_task(b.submit("k", ["running"]))
            await started(probe, 1)
            kept = asyncio.create_task(b.submit("k", ["a1", "a2", "a3", "a4"]))
            await asyncio.sleep(0)
            assert b.depth == 4

            with pytest.raises(QueueFull):
                await b.submit("k", ["b1", "b2"])
            assert b.depth == 4

            exactly_full = asyncio.create_task(b.submit("k", ["c1"]))
            await asyncio.sleep(0)
            assert b.depth == 5
            with pytest.raises(QueueFull):
                await b.submit("k", ["d1"])
            assert b.depth == 5

            probe.release()
            assert await running == ["k:running"]
            assert await kept == ["k:a1", "k:a2", "k:a3", "k:a4"]
            assert await exactly_full == ["k:c1"]

        assert sorted(probe.items_seen()) == sorted(
            ["running", "a1", "a2", "a3", "a4", "c1"]
        )

    run_scenario(scenario)


def test_a_request_larger_than_the_whole_queue_is_always_refused() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_queue_sentences=3) as b:
            with pytest.raises(QueueFull, match="3"):
                await b.submit("k", ["a", "b", "c", "d"])
            assert b.depth == 0
            assert await b.submit("k", ["a", "b", "c"]) == ["k:a", "k:b", "k:c"]

        assert probe.items_seen() == ["a", "b", "c"]

    run_scenario(scenario)


def test_the_refusal_names_the_numbers_and_no_text() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_queue_sentences=2) as b:
            with pytest.raises(QueueFull) as caught:
                await b.submit("k", ["secret text one", "two", "three"])

        assert "secret" not in str(caught.value)
        assert "3" in str(caught.value)

    run_scenario(scenario)


def test_an_exception_of_run_fails_exactly_the_items_of_its_batch() -> None:
    probe = BatchProbe(fail_on={"bad"})

    async def scenario() -> None:
        # One worker: the second batch is still queued when the first one fails.
        async with batching(
            probe, workers=1, max_batch_sentences=2, max_wait_seconds=0.01
        ) as b:
            failing, fine = await asyncio.gather(
                b.submit("k", ["a", "bad"]),
                b.submit("k", ["c", "d"]),
                return_exceptions=True,
            )
            assert isinstance(failing, ValueError)
            assert fine == ["k:c", "k:d"]

            assert await b.submit("k", ["after"]) == ["k:after"]
            assert (b.depth, b.inflight) == (0, 0)

        assert sorted(probe.calls) == [
            ("k", ["a", "bad"]),
            ("k", ["after"]),
            ("k", ["c", "d"]),
        ]

    run_scenario(scenario)


def test_two_requests_in_one_failing_batch_both_fail() -> None:
    probe = BatchProbe(fail_on={"bad"})

    async def scenario() -> None:
        async with batching(probe, max_batch_sentences=8, max_wait_seconds=0.05) as b:
            first, second = await asyncio.gather(
                b.submit("k", ["x"]), b.submit("k", ["bad"]), return_exceptions=True
            )

        assert isinstance(first, ValueError)
        assert isinstance(second, ValueError)
        assert probe.calls == [("k", ["x", "bad"])]

    run_scenario(scenario)


def test_a_request_with_items_in_a_failed_and_a_good_batch_fails() -> None:
    probe = BatchProbe(fail_on={"bad"})

    async def scenario() -> None:
        async with batching(probe, max_batch_sentences=2, max_wait_seconds=0.01) as b:
            with pytest.raises(ValueError, match="scripted"):
                await b.submit("k", ["a", "bad", "c", "d"])
            assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_a_run_that_returns_too_few_results_fails_its_batch_and_the_batcher_goes_on() -> (
    None
):
    probe = BatchProbe(short_on={"short"})

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.01) as b:
            with pytest.raises(UpstreamError, match="3 items came back with 2 results"):
                await b.submit("k", ["a", "b", "short"])

            assert await b.submit("k", ["x"]) == ["k:x"]
            assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_a_pool_that_refuses_a_batch_fails_it_and_nothing_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = BatchProbe()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cannot schedule new futures after shutdown")

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.01) as b:
            monkeypatch.setattr(ThreadPoolExecutor, "submit", refuse)
            with pytest.raises(RuntimeError, match="cannot schedule"):
                await b.submit("k", ["a", "b"])
            assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_close_answers_everything_queued_without_waiting_for_the_wait() -> None:
    probe = BatchProbe(hold=True)
    requests = [[f"r{n}a", f"r{n}b"] for n in range(25)]

    async def scenario() -> None:
        b = make_batcher(
            probe,
            workers=1,
            max_batch_sentences=4,
            max_queue_sentences=100,
            max_wait_seconds=FOREVER,
        )
        tasks = [asyncio.create_task(b.submit("k", items)) for items in requests]
        await started(probe, 1)
        assert b.depth == 46

        closing = asyncio.create_task(b.close(10.0))
        await asyncio.sleep(0)
        with pytest.raises(Draining):
            await b.submit("k", ["late"])
        probe.release()

        assert await closing == 0
        results = [await task for task in tasks]

        assert results == [[f"k:{item}" for item in items] for items in requests]
        assert sorted(probe.items_seen()) == sorted(
            item for items in requests for item in items
        )
        assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_a_submit_after_close_began_is_draining_and_queues_nothing() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        b = make_batcher(probe)
        assert await b.close(1.0) == 0

        with pytest.raises(Draining):
            await b.submit("k", ["a"])
        assert b.depth == 0
        assert probe.calls == []

    run_scenario(scenario)


def test_close_is_idempotent_and_repeats_its_count() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        b = make_batcher(probe, workers=1, max_batch_sentences=2, max_wait_seconds=0.01)
        tasks = [asyncio.create_task(b.submit("k", ["a", "b"])) for _ in range(2)]
        await started(probe, 1)

        first = await b.close(0.05)
        second = await b.close(5.0)
        probe.release()
        await asyncio.gather(*tasks, return_exceptions=True)
        await until(lambda: b.inflight == 0)

        assert first == second == 4

    run_scenario(scenario)


def test_close_with_a_timeout_shorter_than_the_work_fails_the_rest_with_draining() -> (
    None
):
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        b = make_batcher(probe, workers=1, max_batch_sentences=2, max_wait_seconds=0.01)
        tasks = [
            asyncio.create_task(b.submit("k", [f"r{n}a", f"r{n}b"])) for n in range(3)
        ]
        await started(probe, 1)
        await asyncio.sleep(0)
        assert (b.depth, b.inflight) == (4, 1)

        began = time.perf_counter()
        failed = await b.close(0.05)
        closing_took = time.perf_counter() - began

        assert failed == 6
        # The batch in flight is still held: giving up does not wait for it.
        assert closing_took < 5.0
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(outcome, Draining) for outcome in outcomes)
        assert b.depth == 0
        probe.release()
        await until(lambda: b.inflight == 0)
        assert probe.items_seen() == ["r0a", "r0b"]

    run_scenario(scenario)


def test_the_items_already_answered_are_not_counted_as_failed_by_a_close() -> None:
    probe = BatchProbe(hold={"slow"})

    async def scenario() -> None:
        b = make_batcher(probe, workers=2, max_batch_sentences=8, max_wait_seconds=0.01)
        slow = asyncio.create_task(b.submit("slow", ["s1", "s2"]))
        await started(probe, 1)
        assert await b.submit("fast", ["f1"]) == ["fast:f1"]

        failed = await b.close(0.05)

        assert failed == 2
        assert isinstance(
            (await asyncio.gather(slow, return_exceptions=True))[0], Draining
        )
        probe.release()
        await until(lambda: b.inflight == 0)

    run_scenario(scenario)


def test_close_of_a_batcher_that_was_never_used_returns_zero() -> None:
    async def scenario() -> None:
        assert await make_batcher(BatchProbe()).close(0.1) == 0

    run_scenario(scenario)


def test_cancelling_a_submitter_drops_its_queued_items_and_keeps_the_others() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(
            probe, workers=1, max_batch_sentences=1, max_wait_seconds=0.01
        ) as b:
            running = asyncio.create_task(b.submit("k", ["running"]))
            await started(probe, 1)
            cancelled = asyncio.create_task(b.submit("k", ["x1", "x2", "x3"]))
            kept = asyncio.create_task(b.submit("k", ["y1", "y2"]))
            await asyncio.sleep(0)
            assert b.depth == 5

            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert b.depth == 2

            probe.release()
            assert await running == ["k:running"]
            assert await kept == ["k:y1", "k:y2"]

        assert probe.items_seen() == ["running", "y1", "y2"]
        assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_cancelling_a_submitter_whose_items_are_running_loses_nobody_else() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(probe, workers=1, max_wait_seconds=0.01) as b:
            cancelled = asyncio.create_task(b.submit("k", ["x"]))
            other = asyncio.create_task(b.submit("k", ["y"]))
            await started(probe, 1)
            assert probe.calls == [("k", ["x", "y"])]

            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            probe.release()
            assert await other == ["k:y"]
            await until(lambda: b.inflight == 0)

        assert probe.calls == [("k", ["x", "y"])]
        assert b.depth == 0

    run_scenario(scenario)


def test_a_request_cancelled_just_as_the_service_gives_up_ends_cancelled() -> None:
    probe = BatchProbe(hold=True)
    original = MicroBatcher._give_up_on_the_rest

    async def scenario(monkeypatch: pytest.MonkeyPatch) -> None:
        b = make_batcher(probe, workers=1, max_batch_sentences=1, max_wait_seconds=0.01)
        loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(b.submit("k", [f"r{n}"])) for n in range(3)]
        await started(probe, 1)
        await asyncio.sleep(0)

        def give_up_then_cancel(self: MicroBatcher[str, str, str]) -> int:
            failed = original(self)
            # The items are failed and the submitters have not run yet: cancel them now.
            loop.call_soon(lambda: [task.cancel() for task in tasks])
            return failed

        monkeypatch.setattr(MicroBatcher, "_give_up_on_the_rest", give_up_then_cancel)
        assert await b.close(0.05) == 3
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        assert [type(outcome) for outcome in outcomes] == [asyncio.CancelledError] * 3
        probe.release()
        await until(lambda: b.inflight == 0)

    with pytest.MonkeyPatch.context() as monkeypatch:
        run_scenario(lambda: scenario(monkeypatch))


def test_a_dispatcher_with_nothing_to_do_does_not_spin() -> None:
    probe = BatchProbe(hold=True)

    async def spent_while_waiting(seconds: float) -> float:
        began = time.process_time()
        await asyncio.sleep(seconds)
        return time.process_time() - began

    async def scenario() -> None:
        async with batching(
            probe, workers=1, max_batch_sentences=1, max_wait_seconds=0.01
        ) as b:
            idle = await spent_while_waiting(0.4)

            running = asyncio.create_task(b.submit("k", ["running"]))
            await started(probe, 1)
            queued = [asyncio.create_task(b.submit("k", [f"q{n}"])) for n in range(3)]
            busy = await spent_while_waiting(0.4)

            probe.release()
            await running
            await asyncio.gather(*queued)

        # A dispatcher that wakes up in a loop spends the whole wait; one that sleeps
        # until it is woken spends a few milliseconds.
        assert idle < 0.2
        assert busy < 0.2

    run_scenario(scenario)


def test_the_gauges_are_zero_when_everything_is_answered() -> None:
    probe = BatchProbe(fail_on={"bad"})

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.01) as b:
            assert (b.depth, b.inflight) == (0, 0)
            await b.submit("k", ["a", "b", "c", "d", "e"])
            assert (b.depth, b.inflight) == (0, 0)
            with pytest.raises(ValueError):
                await b.submit("k", ["bad"])
            assert (b.depth, b.inflight) == (0, 0)

    run_scenario(scenario)


def test_the_batch_that_waited_longest_goes_first() -> None:
    probe = BatchProbe(hold=True)

    async def scenario() -> None:
        async with batching(
            probe, workers=1, max_batch_sentences=8, max_wait_seconds=0.01
        ) as b:
            first = asyncio.create_task(b.submit("busy", ["x"]))
            await started(probe, 1)
            older = asyncio.create_task(b.submit("old", ["o"]))
            await asyncio.sleep(0.02)
            newer = asyncio.create_task(b.submit("new", ["n"]))
            await asyncio.sleep(0.05)
            probe.release()
            await asyncio.gather(first, older, newer)

        assert [key for key, _ in probe.calls] == ["busy", "old", "new"]

    run_scenario(scenario)


def test_an_empty_request_is_answered_with_nothing_and_queues_nothing() -> None:
    probe = BatchProbe()

    async def scenario() -> None:
        async with batching(probe, max_wait_seconds=0.01) as b:
            assert await b.submit("k", []) == []
            assert (b.depth, b.inflight) == (0, 0)
            # An empty request must leave no empty queue behind for the dispatcher to trip
            # on: give it the turns to look.
            await asyncio.sleep(0.05)
            assert await b.submit("k", ["a"]) == ["k:a"]

        assert probe.calls == [("k", ["a"])]

    run_scenario(scenario)


def test_the_threads_of_a_closed_batcher_are_gone() -> None:
    probe = BatchProbe()

    def worker_threads() -> list[str]:
        return [
            t.name for t in threading.enumerate() if t.name.startswith("rinchen-batch")
        ]

    async def scenario() -> None:
        b = make_batcher(probe, workers=3, max_batch_sentences=1)
        await asyncio.gather(*(b.submit("k", [str(n)]) for n in range(6)))
        assert worker_threads()
        assert await b.close(5.0) == 0
        # Checked while the batcher is still referenced: a pool nobody shut down keeps its
        # idle threads until it is collected.
        assert worker_threads() == []

    run_scenario(scenario)


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_sentences": 0},
        {"max_batch_sentences": True},
        {"max_batch_sentences": 1.5},
        {"max_queue_sentences": 0},
        {"workers": 0},
        {"workers": False},
        {"max_wait_seconds": 0},
        {"max_wait_seconds": -1.0},
        {"max_wait_seconds": math.nan},
        {"max_wait_seconds": math.inf},
        {"max_wait_seconds": True},
    ],
    ids=lambda options: str(options),
)
def test_a_bad_option_is_a_config_error_naming_it(options: dict[str, Any]) -> None:
    settings: dict[str, Any] = {
        "max_batch_sentences": 4,
        "max_wait_seconds": 0.05,
        "max_queue_sentences": 100,
        "workers": 2,
    }
    settings.update(options)

    with pytest.raises(ConfigError, match=next(iter(options))):
        MicroBatcher(BatchProbe(), **settings)


def test_the_exceptions_are_distinct_and_carry_a_message() -> None:
    assert not issubclass(QueueFull, Draining)
    assert not issubclass(Draining, QueueFull)
    assert str(Draining("closing")) == "closing"
