"""The chat-completions protocol both LLM clients speak: a request per text (or
per candidate), bounded concurrency, retries with backoff and the checks on the
reply. `OpenAICompatTranslator` (vLLM) and `LlamaCppTranslator` differ only in how
many candidates one request asks for, which is `_request_shapes`."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from itertools import chain
from typing import NamedTuple

import httpx

from rinchen.config.synth import LlamaCppConfig, VllmConfig
from rinchen.errors import ConfigError, UpstreamError
from rinchen.synth._candidates import clean_candidates
from rinchen.synth._texts import require_encodable
from rinchen.synth.prompts import render_messages

_SEED_MODULUS = 2**31
_MAX_BACKOFF_S = 30.0
# 2**10 seconds is already past the cap; a larger exponent only risks overflow.
_MAX_BACKOFF_EXPONENT = 10
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_AFTER_STATUSES = frozenset({429, 503})
_PREVIEW_CHARS = 200
# With polars imported (every `rinchen` command has it) SIGINT is handled by a
# handler that restarts a blocking call: an untimed `wait` is not woken and the
# KeyboardInterrupt would be raised only after every queued request was sent,
# while a wait with a timeout is interrupted at once. So the wait is made in
# slices of this length (Wiki page Verification-Log, D-POLARS-SIGINT).
_INTERRUPT_POLL_S = 0.2


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
    are in flight, and any request that fails for good fails the whole call: the
    requests not yet started are cancelled, as they are on an interrupt. A
    `client` passed in is never closed; one the translator makes is closed when
    each call ends.

    What it makes of a reply: any 2xx with a valid body is accepted; a choice
    whose `message.content` is null, absent or empty (a refusal, an empty
    generation) and an empty `choices` list give no candidate; `finish_reason`
    is ignored, so a generation cut at `max_tokens` is kept and the selection
    stage's filters decide. Candidates are stripped with `str.strip` (which also
    removes narrow no-break spaces at the ends, harmless for Cyrillic output,
    the only output a job may request).

    Error messages carry a preview of the server's response body. A server may
    echo fields of the request in its own error body, so the preview is not a
    guarantee about what it shows; the translator itself adds nothing from the
    request to a message."""

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
        _require_http_url(cfg.base_url, self._url)

    @abstractmethod
    def _request_shapes(self) -> Sequence[RequestShape]:
        """The requests sent per text; their candidates are merged in this order."""

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        if not texts:
            return []
        require_encodable(texts)
        bodies = [
            body
            for text in texts
            for body in self._bodies_for(text, src_lang, tgt_lang)
        ]
        replies = self._send_all(bodies)
        per_text = len(self._request_shapes())
        return [
            clean_candidates(chain.from_iterable(replies[start : start + per_text]))
            for start in range(0, len(replies), per_text)
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
        if self._client is None:
            limits = httpx.Limits(max_connections=self._cfg.concurrency)
            http = httpx.Client(limits=limits)
        else:
            http = self._client
        try:
            return self._fan_out(http, bodies)
        finally:
            if owns_client:
                http.close()

    def _fan_out(self, http: httpx.Client, bodies: list[dict]) -> list[list[str]]:
        workers = min(self._cfg.concurrency, len(bodies))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                futures = [pool.submit(self._candidates, http, body) for body in bodies]
                _wait_until_all_done_or_one_failed(futures)
            finally:
                # However the wait ended (a request failed, all are done, an
                # interrupt), requests that have not started are not worth
                # sending. The pool starts requests in submission order, so
                # `result` reaches a failure before it can reach a cancelled
                # request.
                pool.shutdown(wait=False, cancel_futures=True)
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
                    raise self._gave_up(transient.failure, attempt + 1) from (
                        transient.__cause__
                    )
                self._sleep(self._backoff(attempt, transient.retry_after_s))
                attempt += 1
            except (httpx.HTTPError, httpx.InvalidURL) as error:
                # Not a transport failure (an undecodable body, a redirect loop,
                # an invalid URL): trying again would not help.
                raise self._gave_up(type(error).__name__, attempt + 1) from error

    def _gave_up(self, failure: str, attempts: int) -> UpstreamError:
        return UpstreamError(f"{self._url} failed with {failure}, attempts: {attempts}")

    def _post_once(self, http: httpx.Client, body: dict) -> httpx.Response:
        try:
            response = http.post(self._url, json=body, timeout=self._cfg.timeout_s)
        except httpx.TransportError as error:
            raise _Transient(type(error).__name__) from error
        if response.is_success:
            return response
        failure = _status_failure(response)
        if response.status_code not in _RETRYABLE_STATUSES:
            raise UpstreamError(f"{self._url} answered {failure}, not retried")
        retry_after_s = (
            _retry_after_seconds(response)
            if response.status_code in _RETRY_AFTER_STATUSES
            else None
        )
        raise _Transient(failure, retry_after_s)

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        if retry_after_s is None:
            exponent = min(attempt, _MAX_BACKOFF_EXPONENT)
            return min(self._cfg.backoff_s * 2**exponent, _MAX_BACKOFF_S)
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
        if payload.get("error") is not None:
            raise UpstreamError(
                f"{self._url} answered an error body: {_preview(response)}"
            )
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise UpstreamError(f"{self._url} answered without choices")
        if len(choices) > n:
            raise UpstreamError(
                f"{self._url} answered {len(choices)} choices for n={n}"
            )
        return [self._content_of(choice) for choice in choices]

    def _content_of(self, choice: object) -> str:
        """The text of a choice; a null or absent content, which is how a refusal
        or an empty generation arrives, is the empty string."""
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise UpstreamError(
                f"{self._url} answered a choice without a message object"
            )
        content = message.get("content")
        if content is None:
            return ""
        if not isinstance(content, str):
            raise UpstreamError(
                f"{self._url} answered a message.content that is not a string"
            )
        return content


def _wait_until_all_done_or_one_failed(futures: Sequence[Future]) -> None:
    """Return when every future has finished or one has raised, waking every
    `_INTERRUPT_POLL_S` so that a pending interrupt can be raised."""
    pending = set(futures)
    while pending:
        done, pending = wait(
            pending, timeout=_INTERRUPT_POLL_S, return_when=FIRST_EXCEPTION
        )
        if any(
            not future.cancelled() and future.exception() is not None for future in done
        ):
            return


def _require_http_url(base_url: str, endpoint: str) -> None:
    try:
        url = httpx.URL(endpoint)
    except httpx.InvalidURL as error:
        raise ConfigError(
            f"base_url {base_url!r} is not a valid URL: {error}"
        ) from error
    if url.scheme not in ("http", "https") or not url.host:
        raise ConfigError(
            f"base_url {base_url!r} must be an http or https URL with a host"
        )


def _preview(response: httpx.Response) -> str:
    """The start of a reply's body for an error message: decoded leniently,
    whitespace collapsed, cut at 200 characters."""
    text = response.content.decode("utf-8", errors="replace")
    return " ".join(text.split())[:_PREVIEW_CHARS]


def _status_failure(response: httpx.Response) -> str:
    preview = _preview(response)
    status = f"HTTP {response.status_code}"
    return f"{status} ({preview})" if preview else status


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """A `Retry-After` given as a whole number of seconds; an HTTP date or any
    other value is ignored."""
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isascii() and value.isdigit() else None
