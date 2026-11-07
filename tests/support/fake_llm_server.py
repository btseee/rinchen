"""A scripted stand-in for an OpenAI-compatible chat server, served through an
`httpx.MockTransport` so no test opens a socket.

`FakeLlmServer(responder)` answers every request with `responder(body)` (the
decoded JSON request body in, a JSON reply out) and records what it was asked.
On top of that a test can:

- script failures for the first requests, in arrival order (`failures`): a status
  code, an exception to raise from the transport, or a function from the request
  to a response (for a 429 with a `Retry-After` header or a malformed 200);
- force the order in which requests are answered (`answer_order`), so requests
  sent together come back out of order without any timing assumption;
- hold every answer for `delay_s` so requests overlap, and read `max_in_flight`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Hashable, Mapping, Sequence

import httpx

Responder = Callable[[dict], dict]
Failure = int | BaseException | Callable[[httpx.Request], httpx.Response]


def source_text(body: dict) -> str:
    """The text a chat request asks to translate: what follows the first blank
    line of its user message, which is how `render_prompt` lays it out."""
    content = body["messages"][0]["content"]
    return content.partition("\n\n")[2]


def chat_reply(candidates: Sequence[str]) -> dict:
    """The JSON body of a successful chat completion with one choice each, with
    the extra fields a real server sends (a client must ignore them)."""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": candidate},
                "finish_reason": "stop",
            }
            for index, candidate in enumerate(candidates)
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def echo_responder(candidates_per_text: int = 1) -> Responder:
    """Answer with `candidates_per_text` choices, the k-th being `<text>/<k>`."""

    def respond(body: dict) -> dict:
        text = source_text(body)
        return chat_reply([f"{text}/{k}" for k in range(candidates_per_text)])

    return respond


def table_responder(candidates: Mapping[str, Sequence[str]]) -> Responder:
    """Answer each source text with its listed candidates; an unlisted text is
    a test mistake and raises KeyError out of the transport."""

    def respond(body: dict) -> dict:
        return chat_reply(candidates[source_text(body)])

    return respond


def rendezvous_responder(
    parties: int, responder: Responder, *, timeout_s: float = 5.0
) -> Responder:
    """Hold every request until `parties` are waiting together, then answer them
    with `responder`. That makes "N requests are in flight at once" a certainty
    instead of a timing window; if they never gather, `BrokenBarrierError`
    surfaces out of the transport. The request count has to be a multiple of
    `parties`."""
    barrier = threading.Barrier(parties, timeout=timeout_s)

    def respond(body: dict) -> dict:
        barrier.wait()
        return responder(body)

    return respond


class FakeLlmServer:
    def __init__(
        self,
        responder: Responder,
        *,
        failures: Sequence[Failure] = (),
        answer_order: Sequence[Hashable] = (),
        order_key: Callable[[dict], Hashable] = source_text,
        turn_timeout_s: float = 5.0,
        delay_s: float = 0.0,
    ) -> None:
        self.transport = httpx.MockTransport(self._handle)
        # Every request in arrival order, scripted failures and retries included.
        self.requests: list[dict] = []
        self.http_requests: list[httpx.Request] = []
        # The keys of the requests the responder answered, in answer order.
        self.answered: list[Hashable] = []
        self.max_in_flight = 0
        self._responder = responder
        self._failures = tuple(failures)
        self._answer_order = tuple(answer_order)
        self._order_key = order_key
        self._turn_timeout_s = turn_timeout_s
        self._delay_s = delay_s
        self._in_flight = 0
        self._changed = threading.Condition()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._changed:
            arrival = len(self.requests)
            self.requests.append(body)
            self.http_requests.append(request)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if arrival < len(self._failures):
                return self._fail(self._failures[arrival], request)
            return self._answer(body)
        finally:
            with self._changed:
                self._in_flight -= 1

    @staticmethod
    def _fail(failure: Failure, request: httpx.Request) -> httpx.Response:
        if isinstance(failure, BaseException):
            raise failure
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "scripted"}})
        return failure(request)

    def _answer(self, body: dict) -> httpx.Response:
        key = self._order_key(body)
        self._wait_for_turn(key)
        if self._delay_s:
            time.sleep(self._delay_s)
        reply = self._responder(body)
        with self._changed:
            self.answered.append(key)
            self._changed.notify_all()
        return httpx.Response(200, json=reply)

    def _wait_for_turn(self, key: Hashable) -> None:
        if key not in self._answer_order:
            return
        before = set(self._answer_order[: self._answer_order.index(key)])
        with self._changed:
            turn = self._changed.wait_for(
                lambda: before.issubset(self.answered), self._turn_timeout_s
            )
        if not turn:
            raise AssertionError(
                f"the fake server never reached the turn of request {key!r} "
                f"in the answer order {self._answer_order!r}"
            )
