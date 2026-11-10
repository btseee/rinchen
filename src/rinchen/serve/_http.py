"""The HTTP layer of the service that is not about translating (D11, T15, T17): the limit on
a body, the record of every response, and the one error shape.

Two pure ASGI middlewares (no `BaseHTTPMiddleware`, whose body handling would hide what they do):

- `BodyLimit` refuses a body over `max_body_bytes` with a 413: by its `Content-Length`, before a
  byte is read, and, for a body that has none (chunked), by counting what the app reads: the chunk
  that crosses the limit makes `receive` raise `BodyTooLarge`, and nothing after it is read. The
  one route that reads a body reads it itself (`Request.body`), so the exception reaches the
  handler registered for it; a framework that parses the body for a route would turn it into a
  400, which is why the service parses its one body by hand.
- `RequestRecorder` counts every response in `rinchen_requests_total` and times it by route, and
  logs one `request` event, at the moment the response starts (so a client that has seen the
  response finds it counted). It sees 404s, 405s, 413s and 500s as it sees a 200.

`register_error_handlers` makes every error the service answers `{"error": {"code": str,
"message": str}}`: an unknown path (404), a wrong method (405), a refused parameter (422), a body
over the limit (413) and an exception nobody handled (500: a generic message; the log gets the
type of the exception, never its message). A message names a field or a limit and never what a
client sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Final, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rinchen.serve.metrics import ServeMetrics
from rinchen.serve.schemas import error_response, validation_error_response

# Where the recorder keeps what a handler tells it about its request (`RequestNote`).
NOTE_KEY: Final = "rinchen.request_note"
RETRY_AFTER_SECONDS: Final = "1"

_ROUTE_BY_PATH: Final = {
    "/v1/translate": "translate",
    "/healthz": "healthz",
    "/readyz": "readyz",
    "/metrics": "metrics",
}
_CODE_BY_STATUS: Final = {404: "not_found", 405: "method_not_allowed"}


class EventLog(Protocol):
    """What the service asks of its logger (`rinchen.utils.logging.get` returns one): events
    with named fields, none of which is ever a text of a request."""

    def info(self, event: str, **fields: Any) -> None: ...

    def warning(self, event: str, **fields: Any) -> None: ...

    def error(self, event: str, **fields: Any) -> None: ...


class BodyTooLarge(Exception):
    """The body of a request is longer than the service accepts."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"the body is over the limit of {limit} bytes")
        self.limit = limit


@dataclass(slots=True)
class RequestNote:
    """What the log says of a request besides its route and status: how many texts it held,
    the target it asked for (only a target the model writes) and the code of its refusal."""

    texts: int | None = None
    target: str | None = None
    code: str | None = None


def note_of(scope: Scope) -> RequestNote:
    """The note of the request `scope` belongs to (a fresh, unrecorded one when no recorder
    is in front of the app)."""
    note = scope.get(NOTE_KEY)
    if isinstance(note, RequestNote):
        return note
    return RequestNote()


def route_of(scope: Scope) -> str:
    """The route label of a request: one of the four routes, else `other`."""
    return _ROUTE_BY_PATH.get(scope["path"], "other")


def refusal(
    scope: Scope,
    status: int,
    code: str,
    message: str,
    *,
    retry: bool = False,
    close: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """An error response in the one shape, whose `code` the log of the request will carry.
    `retry` adds `Retry-After: 1` (the service is busy or stopping: try again), `close` adds
    `Connection: close` (the rest of the request was not read)."""
    note_of(scope).code = code
    response = error_response(status, code, message)
    for name, value in (headers or {}).items():
        response.headers[name] = value
    if retry:
        response.headers["Retry-After"] = RETRY_AFTER_SECONDS
    if close:
        response.headers["Connection"] = "close"
    return response


# ---- the limit on a body ------------------------------------------------------------------


class BodyLimit:
    """Refuses a body of more than `limit` bytes with a 413 (see the module's text)."""

    def __init__(self, app: ASGIApp, *, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None and declared > self.limit:
            await too_large(scope, self.limit)(scope, receive, send)
            return
        seen = 0

        async def bounded() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise BodyTooLarge(self.limit)
            return message

        await self.app(scope, bounded, send)


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            digits = value.strip()
            return int(digits) if digits.isdigit() else None
    return None


def too_large(scope: Scope, limit: int) -> JSONResponse:
    return refusal(
        scope,
        413,
        "payload_too_large",
        f"the body is over the limit of {limit} bytes",
        close=True,
    )


# ---- the record of every response ---------------------------------------------------------


class RequestRecorder:
    """Counts, times and logs every response (see the module's text)."""

    def __init__(self, app: ASGIApp, *, metrics: ServeMetrics, log: EventLog) -> None:
        self.app = app
        self._metrics = metrics
        self._log = log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        note = RequestNote()
        scope[NOTE_KEY] = note
        started = time.perf_counter()
        recorded = False

        def record(status: int) -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True
            seconds = time.perf_counter() - started
            route = route_of(scope)
            self._metrics.request(route, status, seconds)
            fields: dict[str, Any] = {
                "route": route,
                "status": status,
                "milliseconds": round(seconds * 1000, 3),
            }
            if note.texts is not None:
                fields["texts"] = note.texts
            if note.target is not None:
                fields["target"] = note.target
            if note.code is not None:
                fields["code"] = note.code
            self._log.info("request", **fields)

        async def recording_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                record(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, recording_send)
        except Exception:
            # Nothing answered yet: the server's own 500 follows, which is what is counted.
            record(500)
            raise


# ---- the error shape on every path --------------------------------------------------------


def register_error_handlers(app: FastAPI, *, log: EventLog) -> None:
    """Make every error of `app` the one shape: see the module's text."""

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        code = _CODE_BY_STATUS.get(exc.status_code, "http_error")
        phrase = HTTPStatus(exc.status_code).phrase
        headers = dict(exc.headers) if exc.headers else None
        return refusal(request.scope, exc.status_code, code, phrase, headers=headers)

    async def invalid(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        note_of(request.scope).code = "validation_error"
        return validation_error_response(exc.errors())

    async def too_big(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, BodyTooLarge)
        return too_large(request.scope, exc.limit)

    async def unexpected(request: Request, exc: Exception) -> Response:
        log.error("unhandled_error", error_type=type(exc).__name__)
        return refusal(
            request.scope, 500, "internal", "the service failed to answer this request"
        )

    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(RequestValidationError, invalid)
    app.add_exception_handler(BodyTooLarge, too_big)
    app.add_exception_handler(Exception, unexpected)
