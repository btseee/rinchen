"""The fake chat server the synthetic-data client tests run against: request
capture, scripted failures, forced answer order and in-flight accounting."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tests.support.fake_llm_server import (
    FakeLlmServer,
    chat_reply,
    echo_responder,
    rendezvous_responder,
    source_text,
)

URL = "http://fake.test/v1/chat/completions"


def _body(text: str) -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": f"Translate.\n\n{text}"}],
        "n": 1,
    }


def _post(server: FakeLlmServer, text: str) -> httpx.Response:
    client = httpx.Client(transport=server.transport)
    return client.post(URL, json=_body(text), timeout=3.0)


def test_chat_reply_carries_the_fields_a_real_server_adds():
    reply = chat_reply(["a"])

    assert reply["object"] == "chat.completion"
    assert reply["id"].startswith("chatcmpl-")
    assert reply["model"] == "fake-model"
    assert reply["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


def test_chat_reply_has_one_choice_per_candidate_in_order():
    reply = chat_reply(["a", "b"])

    assert [choice["message"]["content"] for choice in reply["choices"]] == ["a", "b"]
    assert [choice["index"] for choice in reply["choices"]] == [0, 1]


def test_source_text_is_everything_after_the_first_blank_line():
    body = {"messages": [{"role": "user", "content": "Translate.\n\nfirst\n\nsecond"}]}

    assert source_text(body) == "first\n\nsecond"


def test_echo_responder_answers_each_request_with_its_own_text():
    server = FakeLlmServer(echo_responder(2))

    reply = _post(server, "hello").json()

    assert [c["message"]["content"] for c in reply["choices"]] == ["hello/0", "hello/1"]


def test_a_request_is_answered_with_the_responder_reply_and_recorded():
    server = FakeLlmServer(lambda body: chat_reply(["ok:" + source_text(body)]))

    first = _post(server, "one")
    second = _post(server, "two")

    assert first.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "ok:one"
    assert second.json()["choices"][0]["message"]["content"] == "ok:two"
    assert server.requests == [_body("one"), _body("two")]
    assert [request.method for request in server.http_requests] == ["POST", "POST"]
    assert [str(request.url) for request in server.http_requests] == [URL, URL]
    assert server.answered == ["one", "two"]


def test_the_timeout_of_each_request_is_visible():
    server = FakeLlmServer(echo_responder())

    _post(server, "one")

    assert server.http_requests[0].extensions["timeout"] == {
        "connect": 3.0,
        "read": 3.0,
        "write": 3.0,
        "pool": 3.0,
    }


def test_scripted_failures_apply_to_the_first_requests_in_arrival_order():
    def too_many(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3"})

    server = FakeLlmServer(
        lambda body: chat_reply(["fine"]),
        failures=[503, httpx.ConnectError("refused"), too_many],
    )
    client = httpx.Client(transport=server.transport)

    first = client.post(URL, json=_body("a"))
    with pytest.raises(httpx.ConnectError, match="refused"):
        client.post(URL, json=_body("a"))
    third = client.post(URL, json=_body("a"))
    fourth = client.post(URL, json=_body("a"))

    assert first.status_code == 503
    assert third.status_code == 429
    assert third.headers["Retry-After"] == "3"
    assert fourth.status_code == 200
    assert len(server.requests) == 4
    assert server.answered == ["a"]


def test_a_scripted_failure_can_be_a_malformed_ok_response():
    server = FakeLlmServer(
        echo_responder(),
        failures=[lambda _request: httpx.Response(200, content=b"not json")],
    )

    response = _post(server, "a")

    assert response.status_code == 200
    assert response.content == b"not json"


def test_forced_answer_order_makes_early_requests_wait_for_later_ones():
    server = FakeLlmServer(echo_responder(), answer_order=["c", "b", "a"])
    results: dict[str, str] = {}

    def ask(text: str) -> None:
        reply = _post(server, text).json()
        results[text] = reply["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(ask, text) for text in ("a", "b", "c")]:
            future.result()

    assert server.answered == ["c", "b", "a"]
    assert results == {"a": "a/0", "b": "b/0", "c": "c/0"}


def test_a_request_whose_turn_never_comes_fails_loudly():
    server = FakeLlmServer(
        echo_responder(), answer_order=["first", "second"], turn_timeout_s=0.05
    )

    with pytest.raises(AssertionError, match="second"):
        _post(server, "second")


def test_requests_outside_the_forced_order_are_answered_at_once():
    server = FakeLlmServer(echo_responder(), answer_order=["a", "b"])

    reply = _post(server, "unlisted")

    assert reply.status_code == 200


def test_the_order_key_can_be_any_function_of_the_request_body():
    server = FakeLlmServer(
        echo_responder(),
        answer_order=[2, 1],
        order_key=lambda body: len(source_text(body)),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_post, server, text) for text in ("x", "yy")]
        for future in futures:
            future.result()

    assert server.answered == [2, 1]


def test_a_rendezvous_responder_answers_only_once_enough_requests_are_waiting():
    server = FakeLlmServer(rendezvous_responder(3, echo_responder()))

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_post, server, str(i)) for i in range(3)]
        replies = [future.result().json() for future in futures]

    assert server.max_in_flight == 3
    assert sorted(r["choices"][0]["message"]["content"] for r in replies) == [
        "0/0",
        "1/0",
        "2/0",
    ]


def test_a_rendezvous_that_never_gathers_fails_loudly():
    server = FakeLlmServer(rendezvous_responder(2, echo_responder(), timeout_s=0.05))

    with pytest.raises(threading.BrokenBarrierError):
        _post(server, "alone")


def test_delay_keeps_requests_in_flight_long_enough_to_overlap():
    server = FakeLlmServer(echo_responder(), delay_s=0.2)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(_post, server, str(i)) for i in range(3)]:
            future.result()

    assert server.max_in_flight == 3


def test_max_in_flight_is_one_for_sequential_requests():
    server = FakeLlmServer(echo_responder())

    _post(server, "a")
    _post(server, "b")

    assert server.max_in_flight == 1
