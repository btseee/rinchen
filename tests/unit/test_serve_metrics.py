"""The metrics of the service (D11, T17): every instrument of the spec in the Prometheus text
format, one registry per instance (no global state), no created-timestamp series."""

from __future__ import annotations

import prometheus_client
import pytest

from rinchen.infer.engine_types import ENGINE_COUNTERS, ModelInfo
from rinchen.serve.metrics import ROUTES, ServeMetrics

TYPES = [
    "# TYPE rinchen_requests_total counter",
    "# TYPE rinchen_request_duration_seconds histogram",
    "# TYPE rinchen_translated_sentences_total counter",
    "# TYPE rinchen_batch_size histogram",
    "# TYPE rinchen_queue_depth gauge",
    "# TYPE rinchen_inflight_batches gauge",
    "# TYPE rinchen_script_retries_total counter",
    "# TYPE rinchen_engine_events_total counter",
    "# TYPE rinchen_ready gauge",
    "# TYPE rinchen_draining gauge",
    "# TYPE rinchen_model_info gauge",
]


def model_info(**changes: object) -> ModelInfo:
    fields: dict[str, object] = {
        "name": "toy",
        "quantization": "int8",
        "device": "cpu",
        "compute_type": "int8_float32",
        "placement": "src",
        "directions": ("eng_Latn-khk_Cyrl",),
        "targets": ("khk_Cyrl",),
        "document_level": False,
        "model_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    return ModelInfo(**{**fields, **changes})  # type: ignore[arg-type]


def text_of(metrics: ServeMetrics) -> str:
    return metrics.render()[0].decode("utf-8")


def lines_of(metrics: ServeMetrics) -> list[str]:
    return text_of(metrics).splitlines()


def used_everywhere() -> ServeMetrics:
    metrics = ServeMetrics()
    metrics.request("translate", 200, 0.01)
    metrics.sentences("ok", 3)
    metrics.batch(4)
    metrics.queue_depth(2)
    metrics.inflight(1)
    metrics.engine_events({"sources": 3, "script_retries": 1})
    metrics.ready(True)
    metrics.draining(False)
    metrics.model(model_info())
    return metrics


def test_every_instrument_of_the_spec_is_exposed_with_its_type() -> None:
    lines = lines_of(used_everywhere())

    for type_line in TYPES:
        assert type_line in lines


def test_an_instrument_nobody_used_still_has_its_type_line() -> None:
    lines = lines_of(ServeMetrics())

    for type_line in TYPES:
        assert type_line in lines


def test_the_text_is_prometheus_text_with_its_content_type() -> None:
    body, content_type = ServeMetrics().render()

    assert isinstance(body, bytes)
    assert body.endswith(b"\n")
    assert content_type == "text/plain; version=1.0.0; charset=utf-8"


def test_no_created_series_is_exposed() -> None:
    metrics = used_everywhere()

    assert "_created" not in text_of(metrics)


def test_the_filter_has_something_to_remove() -> None:
    # The library emits the created series unless a process-wide switch turned them off.
    metrics = used_everywhere()

    assert b"_created" in prometheus_client.generate_latest(metrics.registry)


def test_each_instance_has_a_registry_of_its_own() -> None:
    first, second = ServeMetrics(), ServeMetrics()

    first.request("translate", 200, 0.01)
    first.request("translate", 200, 0.02)
    second.request("translate", 200, 0.03)

    line = 'rinchen_requests_total{route="translate",status="200"}'
    assert f"{line} 2.0" in lines_of(first)
    assert f"{line} 1.0" in lines_of(second)
    assert first.registry is not second.registry


def test_nothing_is_registered_in_the_global_registry() -> None:
    used_everywhere()

    assert b"rinchen_" not in prometheus_client.generate_latest(
        prometheus_client.REGISTRY
    )


def test_a_registry_can_be_given() -> None:
    registry = prometheus_client.CollectorRegistry()

    metrics = ServeMetrics(registry)
    metrics.ready(True)

    assert metrics.registry is registry
    assert b"rinchen_ready 1.0" in prometheus_client.generate_latest(registry)


def test_a_registry_that_has_the_names_already_is_refused() -> None:
    registry = prometheus_client.CollectorRegistry()
    ServeMetrics(registry)

    with pytest.raises(ValueError, match="rinchen_"):
        ServeMetrics(registry)


def test_a_request_is_counted_by_route_and_status_and_timed_by_route() -> None:
    metrics = ServeMetrics()

    metrics.request("translate", 200, 0.03125)
    metrics.request("translate", 200, 0.25)
    metrics.request("translate", 503, 0.0)
    metrics.request("readyz", 503, 0.0)

    lines = lines_of(metrics)
    assert 'rinchen_requests_total{route="translate",status="200"} 2.0' in lines
    assert 'rinchen_requests_total{route="translate",status="503"} 1.0' in lines
    assert 'rinchen_requests_total{route="readyz",status="503"} 1.0' in lines
    assert 'rinchen_request_duration_seconds_count{route="translate"} 3.0' in lines
    assert 'rinchen_request_duration_seconds_sum{route="translate"} 0.28125' in lines
    assert (
        'rinchen_request_duration_seconds_bucket{le="0.05",route="translate"} 2.0'
        in (lines)
    )
    assert (
        'rinchen_request_duration_seconds_bucket{le="0.5",route="translate"} 3.0'
        in (lines)
    )


def test_the_routes_are_the_endpoints_and_one_for_the_rest() -> None:
    assert ROUTES == ("translate", "healthz", "readyz", "metrics", "other")


@pytest.mark.parametrize(
    ("route", "status", "seconds"),
    [
        ("/v1/translate", 200, 0.1),
        ("", 200, 0.1),
        ("translate", 99, 0.1),
        ("translate", 600, 0.1),
        ("translate", "200", 0.1),
        ("translate", True, 0.1),
        ("translate", 200, -0.1),
        ("translate", 200, float("nan")),
        ("translate", 200, float("inf")),
    ],
)
def test_a_request_the_set_cannot_hold_is_refused_not_counted(
    route: str, status: object, seconds: float
) -> None:
    metrics = ServeMetrics()

    with pytest.raises(ValueError, match="request"):
        metrics.request(route, status, seconds)  # type: ignore[arg-type]

    assert not any(
        line.startswith("rinchen_requests_total{") for line in lines_of(metrics)
    )


def test_sentences_are_counted_by_status() -> None:
    metrics = ServeMetrics()

    metrics.sentences("ok", 3)
    metrics.sentences("ok", 2)
    metrics.sentences("too_long", 1)
    metrics.sentences("empty", 0)

    lines = lines_of(metrics)
    assert 'rinchen_translated_sentences_total{status="ok"} 5.0' in lines
    assert 'rinchen_translated_sentences_total{status="too_long"} 1.0' in lines
    assert 'rinchen_translated_sentences_total{status="empty"} 0.0' in lines


@pytest.mark.parametrize(
    ("status", "count"), [("fine", 1), ("", 1), ("ok", -1), ("ok", 1.5), ("ok", True)]
)
def test_sentences_of_a_status_the_engine_has_not_or_a_bad_count_are_refused(
    status: str, count: object
) -> None:
    with pytest.raises(ValueError, match="sentences"):
        ServeMetrics().sentences(status, count)  # type: ignore[arg-type]


def test_the_batch_size_histogram_runs_from_one_to_128() -> None:
    metrics = ServeMetrics()

    metrics.batch(1)
    metrics.batch(5)
    metrics.batch(64)
    metrics.batch(200)

    lines = lines_of(metrics)
    buckets = [line for line in lines if line.startswith("rinchen_batch_size_bucket")]
    assert [line.split("{")[1].split("}")[0] for line in buckets] == [
        'le="1.0"',
        'le="2.0"',
        'le="4.0"',
        'le="8.0"',
        'le="16.0"',
        'le="32.0"',
        'le="64.0"',
        'le="128.0"',
        'le="+Inf"',
    ]
    assert 'rinchen_batch_size_bucket{le="1.0"} 1.0' in lines
    assert 'rinchen_batch_size_bucket{le="8.0"} 2.0' in lines
    assert 'rinchen_batch_size_bucket{le="64.0"} 3.0' in lines
    assert 'rinchen_batch_size_bucket{le="128.0"} 3.0' in lines
    assert 'rinchen_batch_size_bucket{le="+Inf"} 4.0' in lines
    assert "rinchen_batch_size_count 4.0" in lines
    assert "rinchen_batch_size_sum 270.0" in lines


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_a_batch_that_is_not_a_positive_count_is_refused(size: object) -> None:
    with pytest.raises(ValueError, match="batch"):
        ServeMetrics().batch(size)  # type: ignore[arg-type]


def test_the_queue_and_the_batches_in_flight_are_gauges_that_follow_the_last_value() -> (
    None
):
    metrics = ServeMetrics()

    metrics.queue_depth(7)
    metrics.inflight(2)
    metrics.queue_depth(3)
    metrics.inflight(0)

    lines = lines_of(metrics)
    assert "rinchen_queue_depth 3.0" in lines
    assert "rinchen_inflight_batches 0.0" in lines


@pytest.mark.parametrize("method", ["queue_depth", "inflight"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_a_gauge_of_a_count_refuses_what_is_not_one(method: str, value: object) -> None:
    with pytest.raises(ValueError, match=method):
        getattr(ServeMetrics(), method)(value)


def test_readiness_and_the_drain_start_at_zero_and_follow_the_flags() -> None:
    metrics = ServeMetrics()
    assert "rinchen_ready 0.0" in lines_of(metrics)
    assert "rinchen_draining 0.0" in lines_of(metrics)

    metrics.ready(True)
    assert "rinchen_ready 1.0" in lines_of(metrics)

    metrics.ready(False)
    metrics.draining(True)
    assert "rinchen_ready 0.0" in lines_of(metrics)
    assert "rinchen_draining 1.0" in lines_of(metrics)


def test_the_engine_counters_are_exposed_by_event() -> None:
    metrics = ServeMetrics()

    metrics.engine_events({"sources": 10, "truncated": 2, "empty": 0})

    lines = lines_of(metrics)
    assert 'rinchen_engine_events_total{event="sources"} 10.0' in lines
    assert 'rinchen_engine_events_total{event="truncated"} 2.0' in lines
    assert 'rinchen_engine_events_total{event="empty"} 0.0' in lines


def test_the_engine_counters_are_cumulative_so_a_second_reading_adds_only_the_difference() -> (
    None
):
    metrics = ServeMetrics()

    metrics.engine_events({"sources": 10, "truncated": 2})
    metrics.engine_events({"sources": 25, "truncated": 2})
    metrics.engine_events({"sources": 25, "truncated": 2})

    lines = lines_of(metrics)
    assert 'rinchen_engine_events_total{event="sources"} 25.0' in lines
    assert 'rinchen_engine_events_total{event="truncated"} 2.0' in lines


def test_a_counter_that_went_back_down_is_a_restart_and_counts_from_zero() -> None:
    metrics = ServeMetrics()

    metrics.engine_events({"sources": 25})
    metrics.engine_events({"sources": 4})
    metrics.engine_events({"sources": 9})

    assert 'rinchen_engine_events_total{event="sources"} 34.0' in lines_of(metrics)


def test_the_script_retries_are_their_own_series_too() -> None:
    metrics = ServeMetrics()

    metrics.engine_events({"script_retries": 3, "sources": 3})
    metrics.engine_events({"script_retries": 5, "sources": 3})

    lines = lines_of(metrics)
    assert "rinchen_script_retries_total 5.0" in lines
    assert 'rinchen_engine_events_total{event="script_retries"} 5.0' in lines


def test_every_counter_the_engine_keeps_is_accepted_whole() -> None:
    metrics = ServeMetrics()

    metrics.engine_events(dict.fromkeys(ENGINE_COUNTERS, 1))

    lines = lines_of(metrics)
    for name in ENGINE_COUNTERS:
        assert f'rinchen_engine_events_total{{event="{name}"}} 1.0' in lines


@pytest.mark.parametrize(
    "counters",
    [{"surprise": 1}, {"sources": -1}, {"sources": 1.5}, {"sources": True}],
)
def test_an_event_the_engine_does_not_have_or_a_bad_count_is_refused_whole(
    counters: dict[str, object],
) -> None:
    metrics = ServeMetrics()

    with pytest.raises(ValueError, match="engine"):
        metrics.engine_events({"empty": 1, **counters})  # type: ignore[arg-type]

    assert "rinchen_engine_events_total{" not in text_of(metrics)


def test_the_model_is_one_series_of_four_labels() -> None:
    metrics = ServeMetrics()

    metrics.model(model_info())

    assert (
        'rinchen_model_info{compute_type="int8_float32",device="cpu",'
        'name="toy",quantization="int8"} 1.0'
    ) in lines_of(metrics)


def test_a_new_model_replaces_the_old_series() -> None:
    metrics = ServeMetrics()

    metrics.model(model_info())
    metrics.model(
        model_info(name="big", quantization="float32", compute_type="float32")
    )

    series = [
        line for line in lines_of(metrics) if line.startswith("rinchen_model_info{")
    ]
    assert series == [
        (
            'rinchen_model_info{compute_type="float32",device="cpu",'
            'name="big",quantization="float32"} 1.0'
        )
    ]
