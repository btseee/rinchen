"""`OpenAICompatTranslator`: the request it sends, the seeds it asks for, the
order of its results, how it retries and what it does with a bad reply. Every
test runs against `FakeLlmServer`; nothing opens a socket."""

from __future__ import annotations

import httpx
import pytest

from rinchen.config.synth import VllmConfig
from rinchen.errors import ConfigError, UpstreamError
from rinchen.synth import _chat_completions as chat_module
from rinchen.synth.prompts import render_messages
from rinchen.synth.vllm_client import OpenAICompatTranslator
from tests.support.fake_llm_server import (
    FakeLlmServer,
    chat_reply,
    echo_responder,
    source_text,
    table_responder,
)

ENG, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"
URL = "http://llm.test:8000/v1/chat/completions"
BASE_URL = "http://llm.test:8000/v1"


def _translator(server: FakeLlmServer, **fields):
    sleeps: list[float] = []
    fields.setdefault("base_url", BASE_URL)
    translator = OpenAICompatTranslator(
        VllmConfig(**fields),
        client=httpx.Client(transport=server.transport),
        sleep=sleeps.append,
    )
    return translator, sleeps


def _seeds_by_text(server: FakeLlmServer) -> dict[str, list[int]]:
    seeds: dict[str, list[int]] = {}
    for body in server.requests:
        seeds.setdefault(source_text(body), []).append(body["seed"])
    return seeds


def _status(code: int, **headers: str):
    return lambda _request: httpx.Response(code, headers=headers)


def test_translate_takes_its_arguments_by_the_protocol_names():
    translator, _ = _translator(FakeLlmServer(echo_responder()))

    result = translator.translate(texts=["a"], src_lang=ENG, tgt_lang=KHK)

    assert result == [["a/0"]]


def test_the_request_body_has_exactly_the_documented_fields():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(
        server, model="hy-test", temperature=0.3, top_p=0.9, max_tokens=64, n=2
    )

    translator.translate(["Hello world"], ENG, KHK)

    (body,) = server.requests
    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "n",
        "seed",
    }
    assert body["model"] == "hy-test"
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 64
    assert body["n"] == 2
    assert body["messages"] == render_messages(ENG, KHK, "Hello world")
    assert [message["role"] for message in body["messages"]] == ["user"]
    assert body["messages"][0]["content"].endswith("\n\nHello world")


@pytest.mark.parametrize("base_url", [BASE_URL, BASE_URL + "/"])
def test_the_request_is_a_post_to_chat_completions_under_the_base_url(base_url):
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server, base_url=base_url)

    translator.translate(["a"], ENG, KHK)

    (request,) = server.http_requests
    assert request.method == "POST"
    assert str(request.url) == URL


def test_one_request_per_text_with_the_text_and_languages_in_the_prompt():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server)

    translator.translate(["alpha", "beta", "gamma"], KHK, ENG)

    assert sorted(source_text(body) for body in server.requests) == [
        "alpha",
        "beta",
        "gamma",
    ]
    prompt = server.requests[0]["messages"][0]["content"]
    assert "Mongolian" in prompt.split("\n\n")[0]
    assert "English" in prompt.split("\n\n")[0]


def test_mongolian_text_and_replies_survive_the_json_round_trip():
    server = FakeLlmServer(table_responder({"Сайн байна уу": ["Hello", "Hi there"]}))
    translator, _ = _translator(server)

    result = translator.translate(["Сайн байна уу"], KHK, ENG)

    assert result == [["Hello", "Hi there"]]
    assert server.requests[0]["messages"][0]["content"].endswith("\n\nСайн байна уу")


def test_an_unsupported_direction_raises_before_any_request():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server)

    with pytest.raises(ConfigError, match="unsupported translation direction"):
        translator.translate(["a"], ENG, RUS)

    assert server.requests == []


def test_the_seed_is_pinned_to_the_documented_formula():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server, n=4)

    translator.translate(["Hello world"], ENG, KHK)

    assert server.requests[0]["seed"] == 798292229


def test_a_text_asks_for_the_same_seed_in_every_run_and_position():
    first = FakeLlmServer(echo_responder())
    second = FakeLlmServer(echo_responder())
    translator_one, _ = _translator(first)
    translator_two, _ = _translator(second)

    translator_one.translate(["x", "y"], ENG, KHK)
    translator_two.translate(["z", "y", "x", "x"], ENG, KHK)

    seeds_one = _seeds_by_text(first)
    seeds_two = _seeds_by_text(second)
    assert seeds_two["x"] == [seeds_one["x"][0]] * 2
    assert seeds_two["y"] == seeds_one["y"]


def test_the_seed_differs_per_text_language_pair_and_candidate_count():
    seen = []
    for text, src, tgt, n in [
        ("Hello world", ENG, KHK, 4),
        ("Hello there", ENG, KHK, 4),
        ("Hello world", KHK, ENG, 4),
        ("Hello world", RUS, KHK, 4),
        ("Hello world", ENG, KHK, 2),
    ]:
        server = FakeLlmServer(echo_responder())
        translator, _ = _translator(server, n=n)
        translator.translate([text], src, tgt)
        seen.append(server.requests[0]["seed"])

    assert seen == [798292229, 2116189699, 243216805, 572529373, 1220420973]


def test_seeds_stay_below_two_to_the_thirty_first():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server)

    translator.translate([f"text {i}" for i in range(200)], ENG, KHK)

    assert all(0 <= body["seed"] < 2**31 for body in server.requests)


def test_results_keep_input_order_when_the_server_answers_out_of_order():
    server = FakeLlmServer(echo_responder(), answer_order=["c", "b", "a"])
    translator, _ = _translator(server, concurrency=3)

    result = translator.translate(["a", "b", "c"], ENG, KHK)

    assert server.answered == ["c", "b", "a"]
    assert result == [["a/0"], ["b/0"], ["c/0"]]


def test_the_same_text_twice_is_translated_twice():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server)

    result = translator.translate(["a", "a"], ENG, KHK)

    assert result == [["a/0"], ["a/0"]]
    assert len(server.requests) == 2


def test_no_more_requests_are_in_flight_than_the_configured_concurrency():
    server = FakeLlmServer(echo_responder(), delay_s=0.05)
    translator, _ = _translator(server, concurrency=2)

    result = translator.translate([f"t{i}" for i in range(6)], ENG, KHK)

    assert result == [[f"t{i}/0"] for i in range(6)]
    assert server.max_in_flight == 2


def test_a_concurrency_larger_than_the_number_of_texts_is_fine():
    server = FakeLlmServer(echo_responder(), delay_s=0.05)
    translator, _ = _translator(server, concurrency=32)

    result = translator.translate(["a", "b"], ENG, KHK)

    assert result == [["a/0"], ["b/0"]]
    assert server.max_in_flight == 2


def test_empty_texts_make_no_request_and_no_client(monkeypatch):
    def no_client(*args, **kwargs):
        raise AssertionError("a client was created for an empty batch")

    monkeypatch.setattr(chat_module.httpx, "Client", no_client)
    translator = OpenAICompatTranslator(VllmConfig())

    assert translator.translate([], ENG, KHK) == []


def test_empty_texts_send_nothing_to_an_injected_client():
    server = FakeLlmServer(echo_responder())
    translator, _ = _translator(server)

    assert translator.translate([], ENG, KHK) == []
    assert server.requests == []


def test_every_request_carries_the_configured_timeout():
    server = FakeLlmServer(echo_responder(), failures=[503])
    translator, _ = _translator(server, timeout_s=7.5)

    translator.translate(["a", "b"], ENG, KHK)

    assert len(server.http_requests) == 3
    for request in server.http_requests:
        assert request.extensions["timeout"] == {
            "connect": 7.5,
            "read": 7.5,
            "write": 7.5,
            "pool": 7.5,
        }


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_transient_status_is_retried_until_the_server_answers(status):
    server = FakeLlmServer(echo_responder(), failures=[status])
    translator, sleeps = _translator(server)

    result = translator.translate(["a"], ENG, KHK)

    assert result == [["a/0"]]
    assert len(server.requests) == 2
    assert sleeps == [1.0]


def test_the_backoff_doubles_from_the_configured_start_before_each_retry():
    server = FakeLlmServer(echo_responder(), failures=[429, 503, 503])
    translator, sleeps = _translator(server, backoff_s=0.5)

    result = translator.translate(["a"], ENG, KHK)

    assert result == [["a/0"]]
    assert sleeps == [0.5, 1.0, 2.0]


def test_the_backoff_is_capped_at_thirty_seconds():
    server = FakeLlmServer(echo_responder(), failures=[503] * 7)
    translator, sleeps = _translator(server, retries=7, backoff_s=1.0)

    translator.translate(["a"], ENG, KHK)

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


@pytest.mark.parametrize(
    ("status", "retry_after", "expected"),
    [
        pytest.param(429, "3", 3.0, id="429-seconds"),
        pytest.param(503, "3", 3.0, id="503-seconds"),
        pytest.param(429, "0", 0.0, id="zero"),
        pytest.param(429, "3600", 30.0, id="a-long-wait-is-capped"),
        pytest.param(429, "Wed, 21 Oct 2026 07:28:00 GMT", 1.0, id="http-date"),
        pytest.param(429, "1.5", 1.0, id="fraction"),
        pytest.param(429, "-5", 1.0, id="negative"),
        pytest.param(429, "", 1.0, id="blank"),
        pytest.param(500, "3", 1.0, id="not-honoured-on-500"),
        pytest.param(502, "3", 1.0, id="not-honoured-on-502"),
    ],
)
def test_retry_after_in_seconds_replaces_the_computed_delay(
    status, retry_after, expected
):
    server = FakeLlmServer(
        echo_responder(), failures=[_status(status, **{"Retry-After": retry_after})]
    )
    translator, sleeps = _translator(server)

    translator.translate(["a"], ENG, KHK)

    assert sleeps == [expected]


def test_exhausting_the_retries_names_the_status_the_attempts_and_the_endpoint():
    server = FakeLlmServer(echo_responder(), failures=[503] * 3)
    translator, sleeps = _translator(server, retries=2)

    with pytest.raises(UpstreamError) as raised:
        translator.translate(["secret source text"], ENG, KHK)

    message = str(raised.value)
    assert "HTTP 503" in message
    assert "attempts: 3" in message
    assert URL in message
    assert "secret source text" not in message
    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_zero_retries_means_one_attempt_and_no_sleep():
    server = FakeLlmServer(echo_responder(), failures=[503])
    translator, sleeps = _translator(server, retries=0)

    with pytest.raises(UpstreamError, match="attempts: 1"):
        translator.translate(["a"], ENG, KHK)

    assert len(server.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 404, 422, 501])
def test_other_statuses_are_not_retried(status):
    server = FakeLlmServer(echo_responder(), failures=[status])
    translator, sleeps = _translator(server)

    with pytest.raises(UpstreamError) as raised:
        translator.translate(["a"], ENG, KHK)

    assert f"HTTP {status}" in str(raised.value)
    assert URL in str(raised.value)
    assert len(server.requests) == 1
    assert sleeps == []


def test_a_transport_error_is_retried():
    server = FakeLlmServer(
        echo_responder(),
        failures=[httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
    )
    translator, sleeps = _translator(server)

    result = translator.translate(["a"], ENG, KHK)

    assert result == [["a/0"]]
    assert sleeps == [1.0, 2.0]


def test_exhausted_transport_errors_name_the_exception_class_and_keep_the_cause():
    server = FakeLlmServer(echo_responder(), failures=[httpx.ReadTimeout("slow")] * 2)
    translator, _ = _translator(server, retries=1)

    with pytest.raises(UpstreamError, match="ReadTimeout") as raised:
        translator.translate(["a"], ENG, KHK)

    assert "attempts: 2" in str(raised.value)
    assert URL in str(raised.value)
    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize(
    ("failure", "fragment"),
    [
        pytest.param(
            lambda _request: httpx.Response(200, content=b"<html>oops</html>"),
            "not JSON",
            id="not-json",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json=["choices"]),
            "not an object",
            id="json-list",
        ),
        pytest.param(
            lambda _request: httpx.Response(
                200, json={"error": {"message": "model gone"}}
            ),
            "error body",
            id="error-key",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json={"id": "x"}),
            "without choices",
            id="no-choices-key",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json={"choices": []}),
            "without choices",
            id="empty-choices",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json={"choices": "text"}),
            "without choices",
            id="choices-not-a-list",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json={"choices": ["text"]}),
            "message.content",
            id="choice-not-an-object",
        ),
        pytest.param(
            lambda _request: httpx.Response(200, json={"choices": [{"index": 0}]}),
            "message.content",
            id="choice-without-message",
        ),
        pytest.param(
            lambda _request: httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant"}}]}
            ),
            "message.content",
            id="message-without-content",
        ),
        pytest.param(
            lambda _request: httpx.Response(
                200, json={"choices": [{"message": {"content": None}}]}
            ),
            "message.content",
            id="content-null",
        ),
        pytest.param(
            lambda _request: httpx.Response(
                200, json={"choices": [{"message": {"content": 7}}]}
            ),
            "message.content",
            id="content-not-a-string",
        ),
        pytest.param(
            lambda _request: httpx.Response(
                200, json={"choices": chat_reply(["fine"])["choices"] + [{"index": 1}]}
            ),
            "message.content",
            id="one-good-one-bad-choice",
        ),
    ],
)
def test_a_malformed_ok_reply_raises_and_is_not_retried(failure, fragment):
    server = FakeLlmServer(echo_responder(), failures=[failure])
    translator, sleeps = _translator(server)

    with pytest.raises(UpstreamError) as raised:
        translator.translate(["a"], ENG, KHK)

    assert fragment in str(raised.value)
    assert URL in str(raised.value)
    assert len(server.requests) == 1
    assert sleeps == []


def test_an_error_body_carries_the_servers_message():
    server = FakeLlmServer(
        echo_responder(),
        failures=[
            lambda _r: httpx.Response(200, json={"error": {"message": "model gone"}})
        ],
    )
    translator, _ = _translator(server)

    with pytest.raises(UpstreamError, match="model gone"):
        translator.translate(["a"], ENG, KHK)


def test_more_choices_than_requested_is_an_error_but_exactly_n_is_fine():
    server = FakeLlmServer(table_responder({"a": ["1", "2", "3"], "b": ["1", "2"]}))
    translator, _ = _translator(server, n=2)

    with pytest.raises(UpstreamError, match="3 choices for n=2"):
        translator.translate(["a"], ENG, KHK)
    assert translator.translate(["b"], ENG, KHK) == [["1", "2"]]


def test_fewer_choices_than_requested_are_accepted():
    server = FakeLlmServer(table_responder({"a": ["only one"]}))
    translator, _ = _translator(server, n=4)

    assert translator.translate(["a"], ENG, KHK) == [["only one"]]


def test_candidates_are_stripped_blank_ones_dropped_and_duplicates_removed():
    server = FakeLlmServer(
        table_responder(
            {"a": ["  one  ", "", "   ", "two", "one", "\ntwo\n", "three", "t  w"]}
        )
    )
    translator, _ = _translator(server, n=8)

    assert translator.translate(["a"], ENG, KHK) == [["one", "two", "three", "t  w"]]


def test_a_text_whose_candidates_are_all_blank_gets_an_empty_list():
    server = FakeLlmServer(table_responder({"a": ["", " \n"], "b": ["kept"]}))
    translator, _ = _translator(server, n=2)

    assert translator.translate(["a", "b"], ENG, KHK) == [[], ["kept"]]


def test_one_failed_request_fails_the_whole_call():
    server = FakeLlmServer(echo_responder(), failures=[400])
    translator, _ = _translator(server, concurrency=2)

    with pytest.raises(UpstreamError, match="HTTP 400"):
        translator.translate([f"t{i}" for i in range(4)], ENG, KHK)


@pytest.mark.timeout(30)
def test_a_failure_stops_the_requests_still_queued_and_nothing_hangs():
    server = FakeLlmServer(echo_responder(), failures=[400], delay_s=0.05)
    translator, _ = _translator(server, concurrency=2)

    with pytest.raises(UpstreamError, match="HTTP 400"):
        translator.translate([f"t{i}" for i in range(30)], ENG, KHK)

    assert len(server.requests) < 10


def test_a_client_that_was_passed_in_is_left_open():
    server = FakeLlmServer(echo_responder())
    client = httpx.Client(transport=server.transport)
    translator = OpenAICompatTranslator(
        VllmConfig(base_url=BASE_URL), client=client, sleep=lambda _s: None
    )

    translator.translate(["a"], ENG, KHK)

    assert client.is_closed is False
    assert (
        client.post(URL, json={"messages": [{"content": "x\n\ny"}]}).status_code == 200
    )


def _record_clients(monkeypatch, server: FakeLlmServer) -> list[httpx.Client]:
    real_client = httpx.Client
    made: list[httpx.Client] = []

    def make_client(*args, **kwargs):
        made.append(real_client(transport=server.transport))
        return made[-1]

    monkeypatch.setattr(chat_module.httpx, "Client", make_client)
    return made


def test_a_client_the_translator_made_is_closed_once_the_call_is_done(monkeypatch):
    server = FakeLlmServer(echo_responder())
    made = _record_clients(monkeypatch, server)
    translator = OpenAICompatTranslator(VllmConfig(base_url=BASE_URL))

    translator.translate(["a", "b", "c"], ENG, KHK)

    assert len(made) == 1
    assert made[0].is_closed is True


def test_a_client_the_translator_made_is_closed_when_the_call_fails(monkeypatch):
    server = FakeLlmServer(echo_responder(), failures=[400])
    made = _record_clients(monkeypatch, server)
    translator = OpenAICompatTranslator(VllmConfig(base_url=BASE_URL))

    with pytest.raises(UpstreamError):
        translator.translate(["a"], ENG, KHK)

    assert made[0].is_closed is True
