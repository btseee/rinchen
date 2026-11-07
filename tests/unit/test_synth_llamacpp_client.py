"""`LlamaCppTranslator`: `n` separate requests per text, each asking for one
candidate with its own seed, merged in request order. Retries, parsing and the
rest of the contract are the shared chat-completions behaviour, covered by
`test_synth_vllm_client`; the cases here only show the llama.cpp client uses it."""

from __future__ import annotations

import httpx
import pytest

from rinchen.config.synth import LlamaCppConfig
from rinchen.errors import UpstreamError
from rinchen.synth import _chat_completions as chat_module
from rinchen.synth.llamacpp_client import LlamaCppTranslator
from rinchen.synth.prompts import render_messages
from tests.support.fake_llm_server import (
    FakeLlmServer,
    chat_reply,
    echo_responder,
    source_text,
)

ENG, KHK = "eng_Latn", "khk_Cyrl"
BASE_URL = "http://llama.test:8080/v1"
URL = "http://llama.test:8080/v1/chat/completions"
# The vLLM client's seed for ("Hello world", eng -> khk) at n=4.
HELLO_SEED = 798292229


def _translator(server: FakeLlmServer, **fields):
    sleeps: list[float] = []
    fields.setdefault("base_url", BASE_URL)
    translator = LlamaCppTranslator(
        LlamaCppConfig(**fields),
        client=httpx.Client(transport=server.transport),
        sleep=sleeps.append,
    )
    return translator, sleeps


def _by_seed(body: dict) -> str:
    return f"{source_text(body)}@{body['seed']}"


def _seed_echo(body: dict) -> dict:
    return chat_reply([_by_seed(body)])


def test_it_issues_exactly_n_requests_per_text_each_asking_for_one_candidate():
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(server, n=3)

    translator.translate(["a", "b"], ENG, KHK)

    assert len(server.requests) == 6
    assert {body["n"] for body in server.requests} == {1}
    for text in ("a", "b"):
        assert sum(source_text(body) == text for body in server.requests) == 3


def test_each_request_is_the_vllm_request_with_one_candidate_and_its_own_seed():
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(
        server, model="hy-test", temperature=0.3, top_p=0.9, max_tokens=64, n=4
    )

    translator.translate(["Hello world"], ENG, KHK)

    bodies = sorted(server.requests, key=lambda body: body["seed"])
    assert bodies == [
        {
            "model": "hy-test",
            "messages": render_messages(ENG, KHK, "Hello world"),
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 64,
            "n": 1,
            "seed": HELLO_SEED + k,
        }
        for k in range(4)
    ]


def test_the_endpoint_is_chat_completions_under_the_base_url_without_a_slash():
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(server, base_url=BASE_URL + "/", n=2)

    translator.translate(["a"], ENG, KHK)

    assert {str(request.url) for request in server.http_requests} == {URL}
    assert {request.method for request in server.http_requests} == {"POST"}


def test_the_seeds_of_a_text_are_its_base_seed_plus_the_candidate_index():
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(server, n=4)

    translator.translate(["Hello world"], ENG, KHK)

    assert sorted(body["seed"] for body in server.requests) == [
        HELLO_SEED,
        HELLO_SEED + 1,
        HELLO_SEED + 2,
        HELLO_SEED + 3,
    ]


def test_seeds_wrap_around_at_two_to_the_thirty_first(monkeypatch):
    monkeypatch.setattr(chat_module, "request_seed", lambda *args: 2**31 - 1)
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(server, n=3)

    translator.translate(["a"], ENG, KHK)

    assert sorted(body["seed"] for body in server.requests) == [0, 1, 2**31 - 1]


def test_the_candidates_are_merged_in_request_order_whatever_the_answer_order():
    # The first request of a text has the smallest seed (base + 0); the server
    # answers the last request first.
    answer_order = [f"Hello world@{HELLO_SEED + k}" for k in (3, 2, 1, 0)]
    server = FakeLlmServer(_seed_echo, answer_order=answer_order, order_key=_by_seed)
    translator, _ = _translator(server, n=4, concurrency=4)

    result = translator.translate(["Hello world"], ENG, KHK)

    assert server.answered == answer_order
    assert result == [[f"Hello world@{HELLO_SEED + k}" for k in range(4)]]


def test_results_keep_input_order_across_texts():
    server = FakeLlmServer(
        echo_responder(), answer_order=["c", "b", "a"], turn_timeout_s=5.0
    )
    translator, _ = _translator(server, n=1, concurrency=3)

    result = translator.translate(["a", "b", "c"], ENG, KHK)

    assert server.answered == ["c", "b", "a"]
    assert result == [["a/0"], ["b/0"], ["c/0"]]


def test_merged_candidates_are_stripped_and_deduplicated_across_requests():
    by_seed = {HELLO_SEED: " same ", HELLO_SEED + 1: "other", HELLO_SEED + 2: ""}
    server = FakeLlmServer(lambda body: chat_reply([by_seed.get(body["seed"], "same")]))
    translator, _ = _translator(server, n=4)

    assert translator.translate(["Hello world"], ENG, KHK) == [["same", "other"]]


def test_total_concurrency_over_all_requests_is_bounded_by_the_config():
    server = FakeLlmServer(_seed_echo, delay_s=0.05)
    translator, _ = _translator(server, n=4, concurrency=3)

    result = translator.translate(["a", "b"], ENG, KHK)

    assert len(result) == 2
    assert len(server.requests) == 8
    assert server.max_in_flight == 3


def test_one_text_can_use_the_whole_concurrency_for_its_candidates():
    server = FakeLlmServer(_seed_echo, delay_s=0.05)
    translator, _ = _translator(server, n=4, concurrency=8)

    translator.translate(["a"], ENG, KHK)

    assert server.max_in_flight == 4


def test_empty_texts_make_no_request():
    server = FakeLlmServer(_seed_echo)
    translator, _ = _translator(server)

    assert translator.translate([], ENG, KHK) == []
    assert server.requests == []


def test_transient_failures_are_retried_with_the_shared_backoff():
    server = FakeLlmServer(_seed_echo, failures=[429, 503])
    translator, sleeps = _translator(server, n=1)

    result = translator.translate(["a"], ENG, KHK)

    assert len(result) == 1
    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_exhausted_retries_name_the_status_attempts_and_endpoint():
    server = FakeLlmServer(_seed_echo, failures=[503] * 3)
    translator, _ = _translator(server, n=1, retries=2)

    with pytest.raises(UpstreamError) as raised:
        translator.translate(["a"], ENG, KHK)

    assert "HTTP 503" in str(raised.value)
    assert "attempts: 3" in str(raised.value)
    assert URL in str(raised.value)


def test_one_failed_request_fails_the_whole_call_with_no_partial_list():
    server = FakeLlmServer(_seed_echo, failures=[400])
    translator, sleeps = _translator(server, n=3, concurrency=2)

    with pytest.raises(UpstreamError, match="HTTP 400"):
        translator.translate(["a", "b"], ENG, KHK)

    assert sleeps == []


def test_a_server_that_ignores_n_equals_one_is_an_error():
    server = FakeLlmServer(lambda body: chat_reply(["one", "two"]))
    translator, _ = _translator(server, n=2)

    with pytest.raises(UpstreamError, match="2 choices for n=1"):
        translator.translate(["a"], ENG, KHK)


def test_a_reply_without_choices_is_an_error():
    server = FakeLlmServer(lambda body: {"error": {"message": "busy"}})
    translator, _ = _translator(server, n=2)

    with pytest.raises(UpstreamError, match="error body"):
        translator.translate(["a"], ENG, KHK)
