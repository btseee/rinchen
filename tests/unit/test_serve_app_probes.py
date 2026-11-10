"""The service (D11, fix wave): the probes and the paths around them.

A path with a trailing slash is not a route (a 404 in the error shape, never a redirect whose
`Location` is built from the `Host` header of the client); a load balancer may probe with HEAD;
and a probe that comes every few seconds is a debug event in the log, while it stays counted in
`rinchen_requests_total`.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.serve_app import (
    RecordingLog,
    error_of,
    make_client,
    post,
    run,
    sample,
)

PROBES = ["/healthz", "/readyz", "/metrics"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/translate/"),
        ("GET", "/healthz/"),
        ("GET", "/readyz/"),
        ("GET", "/metrics/"),
        ("GET", "//healthz"),
    ],
)
def test_a_path_with_a_trailing_slash_is_a_404_in_the_error_shape_and_never_a_redirect(
    recording_log: RecordingLog, method: str, path: str
) -> None:
    async def scenario() -> tuple[Any, str]:
        async with make_client(recording_log) as svc:
            response = await svc.client.request(method, path, json={})
            return response, (await svc.client.get("/metrics")).text

    response, metrics = run(scenario())

    assert response.status_code == 404
    assert error_of(response) == {"code": "not_found", "message": "Not Found"}
    assert "location" not in response.headers
    assert sample(metrics, "rinchen_requests_total", route="other", status="404") == 1
    assert (
        sample(metrics, "rinchen_requests_total", route="other", status="307") is None
    )


@pytest.mark.parametrize("path", PROBES)
def test_a_probe_may_use_head_and_gets_the_status_of_the_get_without_a_body(
    recording_log: RecordingLog, path: str
) -> None:
    async def scenario() -> tuple[Any, Any]:
        async with make_client(recording_log) as svc:
            return await svc.client.get(path), await svc.client.head(path)

    got, head = run(scenario())

    assert (got.status_code, head.status_code) == (200, 200)
    assert head.headers["content-type"] == got.headers["content-type"]
    assert head.content == b""


@pytest.mark.parametrize("path", PROBES)
def test_head_is_listed_among_the_methods_a_probe_allows(
    recording_log: RecordingLog, path: str
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await svc.client.post(path, json={})

    response = run(scenario())

    assert response.status_code == 405
    # The order of the methods is a set's, which changes from one process to the next.
    assert sorted(response.headers["allow"].split(", ")) == ["GET", "HEAD"]


def test_translate_stays_post_only_and_a_head_of_it_is_a_405(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await svc.client.head("/v1/translate")

    response = run(scenario())

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


def test_a_head_probe_is_counted_under_its_route(recording_log: RecordingLog) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            await svc.client.head("/healthz")
            await svc.client.head("/readyz")
            return (await svc.client.get("/metrics")).text

    metrics = run(scenario())

    assert sample(metrics, "rinchen_requests_total", route="healthz", status="200") == 1
    assert sample(metrics, "rinchen_requests_total", route="readyz", status="200") == 1


def test_the_events_of_the_probes_are_debug_and_their_counts_stay(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> str:
        async with make_client(recording_log) as svc:
            for path in PROBES:
                await svc.client.get(path)
            await svc.client.head("/healthz")
            await post(svc.client)
            return (await svc.client.get("/metrics")).text

    metrics = run(scenario())

    levels = [
        (level, fields["route"])
        for level, event, fields in recording_log.events
        if event == "request"
    ]
    assert levels == [
        ("debug", "healthz"),
        ("debug", "readyz"),
        ("debug", "metrics"),
        ("debug", "healthz"),
        ("info", "translate"),
        ("debug", "metrics"),
    ]
    assert sample(metrics, "rinchen_requests_total", route="healthz", status="200") == 2
    assert (
        sample(metrics, "rinchen_requests_total", route="translate", status="200") == 1
    )


def test_a_probe_that_fails_is_still_an_info_event(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> None:
        async with make_client(recording_log) as svc:
            svc.state.ready = False
            await svc.client.get("/readyz")

    run(scenario())

    (event,) = [
        (level, fields)
        for level, name, fields in recording_log.events
        if name == "request" and fields["route"] == "readyz"
    ]
    assert event[0] == "info"
    assert event[1]["status"] == 503
