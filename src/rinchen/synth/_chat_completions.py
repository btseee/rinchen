"""The chat-completions protocol both LLM clients speak: a request per text (or
per candidate), bounded concurrency, retries with backoff and the checks on the
reply. `OpenAICompatTranslator` (vLLM) and `LlamaCppTranslator` differ only in how
many candidates one request asks for, which is `_request_shapes`."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from itertools import chain, islice
from typing import NamedTuple

import httpx

from rinchen.config.synth import LlamaCppConfig, VllmConfig
from rinchen.errors import UpstreamError
from rinchen.synth.prompts import render_messages

_SEED_MODULUS = 2**31
_MAX_BACKOFF_S = 30.0
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_AFTER_STATUSES = frozenset({429, 503})
_ERROR_PREVIEW_CHARS = 200


def request_seed(src_lang: str, tgt_lang: str, n: int, text: str) -> int:
    """The sampling seed of a text's request, from its content only, so a
    resumed run asks for the seed an uninterrupted one would have: the first four
    bytes of sha256 over `"{src_lang}|{tgt_lang}|{n}|{text}"` (UTF-8) read as a
    big-endian integer, modulo 2**31."""
    key = f"{src_lang}|{tgt_lang}|{n}|{text}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % _SEED_MODULUS


class RequestShape(NamedTuple):
    """A request sent for every text: how many candidates it asks for and what
    is added to the text's seed."""

    n: int
    seed_offset: int


class _Transient(Exception):
    """An attempt that failed in a way worth repeating."""

    def __init__(self, failure: str, retry_after_s: float | None = None) -> None:
        super().__init__(failure)
        self.failure = failure
        self.retry_after_s = retry_after_s


class ChatCompletionsTranslator[Config: (VllmConfig, LlamaCppConfig)](ABC):
    """Posts to `{base_url}/chat/completions`; at most `cfg.concurrency` requests
    are in flight, and any request that fails for good fails the whole call.
    A `client` passed in is never closed; one the translator makes is closed
    when each call ends."""

    def __init__(
        self,
        cfg: Config,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg: Config = cfg
        self._client = client
        self._sleep = sleep
        self._url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    @abstractmethod
    def _request_shapes(self) -> Sequence[RequestShape]:
        """The requests sent per text; their candidates are merged in this order."""

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        if not texts:
            return []
        bodies_per_text = [self._bodies_for(text, src_lang, tgt_lang) for text in texts]
        replies = iter(self._send_all(list(chain.from_iterable(bodies_per_text))))
        return [
            _clean(chain.from_iterable(islice(replies, len(bodies))))
            for bodies in bodies_per_text
        ]

    def _bodies_for(self, text: str, src_lang: str, tgt_lang: str) -> list[dict]:
        messages = render_messages(src_lang, tgt_lang, text)
        seed = request_seed(src_lang, tgt_lang, self._cfg.n, text)
        return [
            {
                "model": self._cfg.model,
                "messages": messages,
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "max_tokens": self._cfg.max_tokens,
                "n": shape.n,
                "seed": (seed + shape.seed_offset) % _SEED_MODULUS,
            }
            for shape in self._request_shapes()
        ]

    def _send_all(self, bodies: list[dict]) -> list[list[str]]:
        owns_client = self._client is None
        http = httpx.Client() if self._client is None else self._client
        try:
            return self._fan_out(http, bodies)
        finally:
            if owns_client:
                http.close()

    def _fan_out(self, http: httpx.Client, bodies: list[dict]) -> list[list[str]]:
        workers = min(self._cfg.concurrency, len(bodies))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._candidates, http, body) for body in bodies]
            _, queued = wait(futures, return_when=FIRST_EXCEPTION)
            # A call that has failed returns nothing, so the requests that have
            # not started are not worth sending. The pool starts requests in
            # submission order, so `result` reaches the failure before it can
            # reach a cancelled request.
            for future in queued:
                future.cancel()
            return [future.result() for future in futures]

    def _candidates(self, http: httpx.Client, body: dict) -> list[str]:
        return self._parse(self._post_with_retry(http, body), body["n"])

    def _post_with_retry(self, http: httpx.Client, body: dict) -> httpx.Response:
        attempt = 0
        while True:
            try:
                return self._post_once(http, body)
            except _Transient as transient:
                if attempt >= self._cfg.retries:
                    raise UpstreamError(
                        f"{self._url} failed with {transient.failure}, "
                        f"attempts: {attempt + 1}"
                    ) from transient.__cause__
                self._sleep(self._backoff(attempt, transient.retry_after_s))
                attempt += 1

    def _post_once(self, http: httpx.Client, body: dict) -> httpx.Response:
        try:
            response = http.post(self._url, json=body, timeout=self._cfg.timeout_s)
        except httpx.TransportError as error:
            raise _Transient(type(error).__name__) from error
        if response.is_success:
            return response
        status = response.status_code
        if status not in _RETRYABLE_STATUSES:
            raise UpstreamError(f"{self._url} answered HTTP {status}, not retried")
        retry_after_s = (
            _retry_after_seconds(response) if status in _RETRY_AFTER_STATUSES else None
        )
        raise _Transient(f"HTTP {status}", retry_after_s)

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        if retry_after_s is None:
            return min(self._cfg.backoff_s * 2**attempt, _MAX_BACKOFF_S)
        return min(retry_after_s, _MAX_BACKOFF_S)

    def _parse(self, response: httpx.Response, n: int) -> list[str]:
        """The reply's candidate strings. A malformed reply is not transient, so
        it raises without a retry."""
        try:
            payload = response.json()
        except ValueError as error:
            raise UpstreamError(
                f"{self._url} answered a body that is not JSON"
            ) from error
        if not isinstance(payload, dict):
            raise UpstreamError(
                f"{self._url} answered a JSON body that is not an object"
            )
        if "error" in payload:
            preview = repr(payload["error"])[:_ERROR_PREVIEW_CHARS]
            raise UpstreamError(f"{self._url} answered an error body: {preview}")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UpstreamError(f"{self._url} answered without choices")
        if len(choices) > n:
            raise UpstreamError(
                f"{self._url} answered {len(choices)} choices for n={n}"
            )
        return [self._content_of(choice) for choice in choices]

    def _content_of(self, choice: object) -> str:
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise UpstreamError(
                f"{self._url} answered a choice without a message.content string"
            )
        return content


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """A `Retry-After` given as a whole number of seconds; an HTTP date or any
    other value is ignored."""
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isascii() and value.isdigit() else None


def _clean(candidates: Iterable[str]) -> list[str]:
    """Stripped, non-empty and without duplicates, in first-seen order."""
    stripped = (candidate.strip() for candidate in candidates)
    return list(dict.fromkeys(candidate for candidate in stripped if candidate))
