"""The metrics of the service (D11, T17), in the Prometheus text format.

One `CollectorRegistry` per `ServeMetrics`, so two services in one process (the tests start
many) never share a count and nothing is registered globally. The library's created-timestamp
series are noise; they are left out of the text here, and not by the library's process-wide
switch.

A label takes a value of a closed set (a route of the service, a status of the engine, an event
the engine counts) or a field of the model's record, so the number of series is bounded and no
text of a request can become one. A bad value raises `ValueError` before anything is counted.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Final, get_args

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.metrics_core import Metric

from rinchen.infer.engine_types import ENGINE_COUNTERS, Status
from rinchen.utils.integers import is_strict_int

if TYPE_CHECKING:
    from rinchen.infer.engine_types import ModelInfo

# The endpoints of the service; anything else a client asks for is `other`.
ROUTES: Final = ("translate", "healthz", "readyz", "metrics", "other")

_STATUSES: Final = frozenset(get_args(Status))
_BATCH_BUCKETS: Final = (1, 2, 4, 8, 16, 32, 64, 128)
_DURATION_BUCKETS: Final = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
)


def _require_count(what: str, value: object, *, minimum: int = 0) -> int:
    if not is_strict_int(value) or value < minimum:
        raise ValueError(
            f"{what} must be an integer of at least {minimum}, got {value!r}"
        )
    return value


class _WithoutCreatedSeries:
    """A collector that shows a registry's metrics minus their `_created` samples."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self._registry = registry

    def collect(self) -> Iterator[Metric]:
        for family in self._registry.collect():
            kept = [s for s in family.samples if not s.name.endswith("_created")]
            shown = Metric(family.name, family.documentation, family.type, family.unit)
            shown.samples = kept
            yield shown


class ServeMetrics:
    """The instruments of one service. Each method records one thing; `render` is the text
    `GET /metrics` answers.

    Raises:
        ValueError: (at construction) the registry already holds one of the names."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self._engine_seen: dict[str, int] = {}
        self._requests = Counter(
            "rinchen_requests_total",
            "Requests answered, by route and status.",
            ["route", "status"],
            registry=self.registry,
        )
        self._duration = Histogram(
            "rinchen_request_duration_seconds",
            "Seconds from the request arriving to its answer, by route.",
            ["route"],
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._sentences = Counter(
            "rinchen_translated_sentences_total",
            "Sentences answered, by the engine's status.",
            ["status"],
            registry=self.registry,
        )
        self._batch_size = Histogram(
            "rinchen_batch_size",
            "Sentences in a batch given to the engine.",
            buckets=_BATCH_BUCKETS,
            registry=self.registry,
        )
        self._queue_depth = Gauge(
            "rinchen_queue_depth",
            "Sentences waiting for a batch.",
            registry=self.registry,
        )
        self._inflight = Gauge(
            "rinchen_inflight_batches",
            "Batches the engine is decoding.",
            registry=self.registry,
        )
        self._script_retries = Counter(
            "rinchen_script_retries_total",
            "Sentences decoded again with a script list.",
            registry=self.registry,
        )
        self._engine_events = Counter(
            "rinchen_engine_events_total",
            "What the engine counts, by event.",
            ["event"],
            registry=self.registry,
        )
        self._ready = Gauge(
            "rinchen_ready",
            "1 while the service answers translate requests.",
            registry=self.registry,
        )
        self._draining = Gauge(
            "rinchen_draining",
            "1 once the service stopped accepting work.",
            registry=self.registry,
        )
        self._model_info = Gauge(
            "rinchen_model_info",
            "The model that is loaded (always 1).",
            ["name", "quantization", "device", "compute_type"],
            registry=self.registry,
        )
        self._ready.set(0)
        self._draining.set(0)

    def request(self, route: str, status: int, seconds: float) -> None:
        """One request answered: counted by route and status, timed by route."""
        valid_status = is_strict_int(status) and 100 <= status <= 599
        valid_seconds = (
            isinstance(seconds, int | float)
            and not isinstance(seconds, bool)
            and math.isfinite(seconds)
            and seconds >= 0
        )
        if route not in ROUTES or not valid_status or not valid_seconds:
            raise ValueError(
                f"request must be one of the routes {', '.join(ROUTES)}, a status of "
                f"100 to 599 and a finite time of at least 0 seconds; got "
                f"{route!r}, {status!r}, {seconds!r}"
            )
        self._requests.labels(route, str(status)).inc()
        self._duration.labels(route).observe(seconds)

    def sentences(self, status: str, count: int) -> None:
        """`count` sentences answered with the engine's `status`."""
        if status not in _STATUSES or not is_strict_int(count) or count < 0:
            raise ValueError(
                f"sentences must be a status of the engine ({', '.join(sorted(_STATUSES))}) "
                f"and a count of at least 0; got {status!r}, {count!r}"
            )
        self._sentences.labels(status).inc(count)

    def batch(self, size: int) -> None:
        """A batch of `size` sentences went to the engine."""
        if not is_strict_int(size) or size < 1:
            raise ValueError(
                f"batch size must be an integer of at least 1, got {size!r}"
            )
        self._batch_size.observe(size)

    def queue_depth(self, depth: int) -> None:
        """The sentences waiting for a batch right now."""
        self._queue_depth.set(_require_count("queue_depth", depth))

    def inflight(self, batches: int) -> None:
        """The batches the engine is decoding right now."""
        self._inflight.set(_require_count("inflight", batches))

    def engine_events(self, counters: Mapping[str, int]) -> None:
        """The engine's counters (`Engine.counters()`): each is cumulative since the engine
        opened, so only the part not yet seen is added, and a counter that went down is a
        restart and counts from zero. `script_retries` is also its own series.

        Raises:
            ValueError: an event the engine does not count, or a count that is not an
                integer of at least 0; nothing is recorded then.
        """
        unknown = sorted(set(counters) - set(ENGINE_COUNTERS))
        if unknown:
            raise ValueError(
                f"engine events must be among the engine's: unknown {unknown}"
            )
        for event, value in counters.items():
            _require_count(f"engine event {event!r}", value)
        for event, value in counters.items():
            before = self._engine_seen.get(event, 0)
            added = value - before if value >= before else value
            self._engine_seen[event] = value
            self._engine_events.labels(event).inc(added)
            if event == "script_retries":
                self._script_retries.inc(added)

    def ready(self, is_ready: bool) -> None:
        self._ready.set(1 if is_ready else 0)

    def draining(self, is_draining: bool) -> None:
        self._draining.set(1 if is_draining else 0)

    def model(self, info: ModelInfo) -> None:
        """The loaded model, as one series; a model loaded later replaces it."""
        self._model_info.clear()
        self._model_info.labels(
            info.name, info.quantization, info.device, info.compute_type
        ).set(1)

    def render(self) -> tuple[bytes, str]:
        """The text of every metric and its content type."""
        return generate_latest(
            _WithoutCreatedSeries(self.registry)
        ), CONTENT_TYPE_LATEST
