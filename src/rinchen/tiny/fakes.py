"""The planted stand-ins of `--qe-fake`: a sentence encoder and a quality scorer that need no model
(D15, spec P6).

`ConceptEncoder` stands in for LaBSE. It is not a trained network: a text is read as a BAG OF
CONCEPTS, each concept is one fixed pseudo-random 64-dimensional vector seeded by the CRC-32 of its
name, and the bag is summed in the order of the concepts' names and normalised. The concepts are the
nouns (the three words of a row and the traditional spelling of its Khalkha word), the entities (each
spelling and its traditional spelling), digit strings (the group and decimal separators are not part of
a number) and the numbered placeholders; any other word names none and is ignored. A Khalkha case
suffix glued to a known name is cut off by trying the longest known prefix of three characters or more.
A word is cut from the next at white space and at `. , ? ! : ;` only, so the Mongolian full stop and comma
(U+1803 and U+1802) stay glued to the word before them and that word names no concept: of the 760
authentic fixture rows the one that ends in a full stop, `mm-0031`, embeds under the 0.7 bound of the
filter and is rejected by this stand-in (a fix would move the kept rows and every digest, so it is left).
Nothing is random at run time and nothing depends on the hash seed of the process, so a text has the
same vector in every process.

`ConceptQE` stands in for the MetricX scorer: `25 * max(0, 1 - cosine)` of the two sides, on MetricX's
scale, where lower is better. Neither ranks a real system; they give the filter something that can
tell a pair that says one thing from a pair that says two.

`FakeChatServer` stands in for the OpenAI-compatible server of the synthetic-data stage. It is a
standard-library HTTP server on the loopback address (the one socket the pipeline opens) that answers
`POST /v1/chat/completions` with the oracle's English line for the text a request asks to translate, as
many times over as the request asks for choices, and every other request with a 404. It lives for the
length of a `with` block and is shut down, its socket closed and its thread joined, on every way out.
"""

from __future__ import annotations

import json
import re
import socketserver
import threading
import zlib
from collections.abc import Mapping, Sequence
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from rinchen.errors import ConfigError
from rinchen.mine.embed import Embedder
from rinchen.script_conv.rules import transliterate

DIMENSIONS: Final = 64
_MIN_PREFIX: Final = 3
_QE_SCALE: Final = 25.0
_EMPTY_CONCEPT: Final = "<empty>"
_PLACEHOLDER_OPEN: Final = chr(0x27E6)
# A numbered placeholder, a number (digits, then a separator and one to three digits, as often as it
# goes) or a run of any other characters that are not white space or one of . , ? ! : ;
_TOKEN: Final = re.compile(r"\u27e6\d+\u27e7|\d+(?:[ .,]\d{1,3})*|[^\s.,?!:;]+")
_NOT_A_DIGIT: Final = re.compile(r"\D")


@cache
def _vector(concept: str) -> NDArray[np.float32]:
    """The fixed vector of a concept, before it is summed and normalised (read-only: it is shared)."""
    rng = np.random.default_rng(zlib.crc32(concept.encode("utf-8")))
    vector = rng.standard_normal(DIMENSIONS).astype(np.float32)
    vector.setflags(write=False)
    return vector


def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return _unit(_vector(_EMPTY_CONCEPT))
    return (vector / norm).astype(np.float32)


class ConceptEncoder:
    """An `Embedder` of bags of concepts. `nouns` are rows whose first column is the Khalkha word
    (the others are the same noun in other languages) and `entities` map an id to a language to the
    spellings of the name; both come from the fixtures."""

    def __init__(
        self,
        nouns: Sequence[Sequence[str]],
        entities: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> None:
        concept_of: dict[str, str] = {}
        for index, row in enumerate(nouns):
            for word in row:
                concept_of[word.lower()] = f"n{index}"
            concept_of[transliterate(row[0])] = f"n{index}"
        for entity_id, by_language in entities.items():
            for spellings in by_language.values():
                for spelling in spellings:
                    concept_of[spelling.lower()] = f"e:{entity_id}"
                    concept_of[transliterate(spelling)] = f"e:{entity_id}"
        self._concept_of = concept_of

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One unit row of 64 dimensions per text, in order; a text with no concept has the one fixed
        vector of the empty bag."""
        rows = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        for index, text in enumerate(texts):
            rows[index] = self._embed_one(text)
        return rows

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        total = np.zeros(DIMENSIONS, dtype=np.float32)
        concepts = sorted(self._concepts(text))
        if not concepts:
            return _unit(_vector(_EMPTY_CONCEPT))
        for concept in concepts:
            total += _vector(concept)
        return _unit(total)

    def _concepts(self, text: str) -> list[str]:
        found: list[str] = []
        for token in _TOKEN.findall(text):
            if token[0].isdecimal():
                found.append("d:" + _NOT_A_DIGIT.sub("", token))
            elif token.startswith(_PLACEHOLDER_OPEN):
                found.append("p:" + token)
            elif (concept := self._word_concept(token)) is not None:
                found.append(concept)
        return found

    def _word_concept(self, token: str) -> str | None:
        """The concept of a word, or of the longest known prefix of it of at least three characters
        (a case suffix glued to a name); None when the word names none."""
        concept = self._concept_of.get(token.lower())
        cut = len(token) - 1
        while concept is None and cut >= _MIN_PREFIX:
            concept = self._concept_of.get(token[:cut].lower())
            cut -= 1
        return concept


class ConceptQE:
    """A `QEScorer` over the cosine of the two sides' bags; it runs on the CPU."""

    requires_cuda: bool = False

    def __init__(self, encoder: Embedder) -> None:
        self._encoder = encoder

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        """`25 * max(0, 1 - cosine)` per pair, in order; 0 is a pair that says one thing.

        Raises:
            ValueError: the two lists differ in length.
        """
        if len(sources) != len(translations):
            raise ValueError(
                f"got {len(sources)} sources and {len(translations)} translations"
            )
        cosine = np.einsum(
            "ij,ij->i", self._encoder.embed(sources), self._encoder.embed(translations)
        )
        return [_QE_SCALE * max(0.0, 1.0 - float(value)) for value in cosine]


LOOPBACK: Final = "127.0.0.1"
_CHAT_PATH: Final = "/v1/chat/completions"
_MAX_BODY: Final = 1 << 20
_MAX_CHOICES: Final = 64
# The default backlog of a socket server (5) drops connections when a client opens many at once.
_BACKLOG: Final = 128
# A client that sends fewer bytes than it announced is let go after this long.
_CLIENT_SECONDS: Final = 10
_POLL_SECONDS: Final = 0.05
_THREAD_NAME: Final = "tiny-fake-chat-server"


class _BadRequest(Exception):
    """A request the server refuses with a 400; the message says what is wrong, never with what."""


class _Tally:
    """A count that the handler threads add to and the owner reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def add(self) -> None:
        with self._lock:
            self._value += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


def _completion(answer: str, choices: int) -> dict[str, Any]:
    """The body of a successful chat completion with `choices` identical choices."""
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
            for index in range(choices)
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat_request(body: bytes) -> tuple[str, int]:
    """The source text and the number of choices a chat request asks for: the text after the first blank
    line of the first message, which is how the synthetic-data prompt lays out its source."""
    try:
        document = json.loads(body)
    except ValueError as error:
        raise _BadRequest("the body is not JSON") from error
    if not isinstance(document, dict):
        raise _BadRequest("the body is not a JSON object")
    messages = document.get("messages")
    first = messages[0] if isinstance(messages, list) and messages else None
    content = first.get("content") if isinstance(first, dict) else None
    if not isinstance(content, str):
        raise _BadRequest("the first message has no text")
    choices = document.get("n", 1)
    if type(choices) is not int or not 1 <= choices <= _MAX_CHOICES:
        raise _BadRequest("n is not a whole number from 1 to 64")
    return content.partition("\n\n")[2], choices


class _ChatServer(ThreadingHTTPServer):
    """The HTTP server: the oracle and the count are what its handlers share."""

    request_queue_size = _BACKLOG

    def __init__(self, oracle: Mapping[str, str], tally: _Tally) -> None:
        super().__init__((LOOPBACK, 0), _ChatHandler)
        self.oracle = oracle
        self.tally = tally

    def server_bind(self) -> None:
        """Bind, without the reverse lookup of the address that `HTTPServer` makes after it: no name
        is resolved."""
        socketserver.TCPServer.server_bind(self)
        address = self.server_address
        assert isinstance(address, tuple)
        self.server_name = LOOPBACK
        self.server_port = address[1]

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that hangs up or sends garbage is not the stage's concern, and nothing it sent
        is printed."""


class _ChatHandler(BaseHTTPRequestHandler):
    server: _ChatServer
    timeout = _CLIENT_SECONDS

    def log_message(self, format: str, *args: Any) -> None:
        """Nothing is logged: a request holds the text of a fixture."""

    def __getattr__(self, name: str) -> Any:
        # Every method that is not POST, however odd, is answered 404 and not 501.
        if name.startswith("do_"):
            return self._not_found
        raise AttributeError(name)

    def _reply(self, status: int, document: Mapping[str, Any]) -> None:
        data = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._reply(
            status, {"error": {"message": message, "type": "fake_server_error"}}
        )

    def _not_found(self) -> None:
        self._error(404, "not found")

    def _body(self) -> bytes | None:
        """The body of the request, or None after the refusal that was sent for it."""
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0:
            self._error(400, "a body with a length is needed")
            return None
        if length > _MAX_BODY:
            self._error(413, "the body is too large")
            return None
        return self.rfile.read(length)

    def do_POST(self) -> None:
        if self.path != _CHAT_PATH:
            self._not_found()
            return
        body = self._body()
        if body is None:
            return
        try:
            source, choices = _chat_request(body)
        except _BadRequest as refusal:
            self._error(400, str(refusal))
            return
        self.server.tally.add()
        self._reply(200, _completion(self.server.oracle.get(source, ""), choices))


class FakeChatServer:
    """An OpenAI-compatible chat server over a table: `oracle` maps a source text to its answer.

    Used as a context manager it binds `127.0.0.1` on a port the system picks, serves from a daemon
    thread and, on leaving the block by any path, shuts down, closes the socket and joins the thread.
    `base_url` is what a client's config names, `http://127.0.0.1:<port>/v1`. A text that is not in
    the oracle is answered with an empty text.

    Raises:
        ConfigError: `host` is anything but the loopback address `127.0.0.1`.
    """

    def __init__(
        self, oracle: Mapping[str, str] | None = None, *, host: str = LOOPBACK
    ) -> None:
        if host != LOOPBACK:
            raise ConfigError(
                "the fake chat server listens on the loopback address only"
            )
        self._oracle = dict(oracle or {})
        self._tally = _Tally()
        self._server: _ChatServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """`http://127.0.0.1:<port>/v1` of the running server.

        Raises:
            RuntimeError: the server is not running.
        """
        if self._server is None:
            raise RuntimeError("the fake chat server is not running")
        return f"http://{LOOPBACK}:{self._server.server_port}/v1"

    @property
    def requests_seen(self) -> int:
        """How many chat requests were answered, all runs of this server together."""
        return self._tally.value

    def __enter__(self) -> Self:
        if self._server is not None:
            raise RuntimeError("the fake chat server is already running")
        server = _ChatServer(self._oracle, self._tally)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": _POLL_SECONDS},
            name=_THREAD_NAME,
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            server.server_close()
            raise
        self._server, self._thread = server, thread
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is None or thread is None:
            return
        try:
            server.shutdown()
        finally:
            server.server_close()
            thread.join()
