"""Property tests of the micro-batcher (D11, T16): whatever the arrivals, the keys, the batch
size, the workers and the queue limit, every item of an accepted request is run exactly once,
answered to its own submitter in the order of its request, in a batch of one key that never
exceeds the size limit; a refused request never ran any of its items; and nothing is left
queued or in flight once the batcher is closed."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.serve.batcher import QueueFull
from tests.support.serve_batcher import BatchProbe, make_batcher

KEYS = ("a", "b", "c")


@dataclass(frozen=True)
class Arrival:
    key: str
    size: int
    after: int


arrivals = st.lists(
    st.builds(
        Arrival,
        key=st.sampled_from(KEYS),
        size=st.integers(min_value=1, max_value=7),
        after=st.integers(min_value=0, max_value=3),
    ),
    min_size=1,
    max_size=12,
)


async def play(
    plan: list[Arrival],
    *,
    max_batch: int,
    workers: int,
    max_queue: int,
    max_wait: float,
) -> tuple[BatchProbe, list[list[str] | QueueFull], int, tuple[int, int]]:
    probe = BatchProbe()
    batcher = make_batcher(
        probe,
        max_batch_sentences=max_batch,
        workers=workers,
        max_queue_sentences=max_queue,
        max_wait_seconds=max_wait,
    )

    async def request(index: int, arrival: Arrival) -> list[str] | QueueFull:
        await asyncio.sleep(arrival.after * 0.002)
        items = [f"r{index}.{position}" for position in range(arrival.size)]
        try:
            return await batcher.submit(arrival.key, items)
        except QueueFull as refused:
            return refused

    async with asyncio.timeout(30):
        outcomes = await asyncio.gather(
            *(request(index, arrival) for index, arrival in enumerate(plan))
        )
        failed = await batcher.close(10.0)
    return probe, list(outcomes), failed, (batcher.depth, batcher.inflight)


@settings(max_examples=60)
@given(
    plan=arrivals,
    max_batch=st.integers(min_value=1, max_value=8),
    workers=st.integers(min_value=1, max_value=3),
    max_queue=st.integers(min_value=7, max_value=60),
    max_wait=st.sampled_from([0.001, 0.004]),
)
def test_every_item_is_answered_once_in_order_in_a_batch_of_its_key(
    plan: list[Arrival], max_batch: int, workers: int, max_queue: int, max_wait: float
) -> None:
    probe, outcomes, failed, (depth, inflight) = asyncio.run(
        play(
            plan,
            max_batch=max_batch,
            workers=workers,
            max_queue=max_queue,
            max_wait=max_wait,
        )
    )

    accepted = [
        (index, arrival)
        for index, (arrival, outcome) in enumerate(zip(plan, outcomes, strict=True))
        if not isinstance(outcome, QueueFull)
    ]
    expected_items = sorted(
        f"r{index}.{position}"
        for index, arrival in accepted
        for position in range(arrival.size)
    )

    # Every accepted item ran exactly once, and a refused request ran none.
    assert sorted(probe.items_seen()) == expected_items
    # The sum of the batch sizes is the number of items, and no batch is too big.
    assert sum(len(items) for _, items in probe.calls) == len(expected_items)
    assert all(1 <= len(items) <= max_batch for _, items in probe.calls)
    # A batch holds the items of one key only.
    for key, items in probe.calls:
        assert {plan[int(item[1:].split(".")[0])].key for item in items} == {key}
    # Each submitter got its own items' answers, in the order of its request.
    for index, arrival in accepted:
        assert outcomes[index] == [
            f"{arrival.key}:r{index}.{position}" for position in range(arrival.size)
        ]
    # Nothing is left behind.
    assert (failed, depth, inflight) == (0, 0, 0)
    assert probe.peak <= workers


@settings(max_examples=40)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=9), min_size=1, max_size=8),
    max_batch=st.integers(min_value=1, max_value=8),
)
def test_a_queue_that_holds_everything_refuses_nothing_and_loses_nothing(
    sizes: list[int], max_batch: int
) -> None:
    plan = [Arrival(KEYS[index % 2], size, 0) for index, size in enumerate(sizes)]

    probe, outcomes, failed, _ = asyncio.run(
        play(plan, max_batch=max_batch, workers=2, max_queue=1000, max_wait=0.001)
    )

    assert not any(isinstance(outcome, QueueFull) for outcome in outcomes)
    assert sorted(probe.items_seen()) == sorted(
        f"r{index}.{position}"
        for index, size in enumerate(sizes)
        for position in range(size)
    )
    assert failed == 0
