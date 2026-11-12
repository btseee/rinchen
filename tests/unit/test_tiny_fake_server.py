"""`rinchen.tiny.fakes.FakeChatServer`: the loopback chat server of the synth stage (D15, spec P17, brief
G4.1).

The tests connect to the server's OWN listener on 127.0.0.1, so this module does not use the
`no_network` guard (which refuses every connect). Every client here talks to the loopback address and
nothing else, through the standard library or `httpx` (the library the real command uses), with the
environment's proxy settings ignored. The reply is compared with the body that the test double of the
earlier milestones, `tests/support/fake_llm_server.chat_reply`, builds.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
import pytest

from rinchen.errors import ConfigError
from rinchen.tiny.fakes import FakeChatServer
from tests.support.fake_llm_server import chat_reply

CHAT: Final = "/v1/chat/completions"
ORACLE: Final = {
    "one text": "its answer",
    "second": "another answer",
    "": "for nothing",
}
THREAD: Final = "tiny-fake-chat-server"


@pytest.fixture
def server() -> Iterator[FakeChatServer]:
    with FakeChatServer(ORACLE) as running:
        yield running


def address(server: FakeChatServer) -> tuple[str, int]:
    parts = urlsplit(server.base_url)
    assert parts.hostname is not None and parts.port is not None
    return parts.hostname, parts.port


def exchange(
    server: FakeChatServer,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = address(server)
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def request_body(
    text: str, n: int = 1, *, prompt: str = "Translate to English."
) -> bytes:
    return json.dumps(
        {
            "model": "any",
            "messages": [{"role": "user", "content": f"{prompt}\n\n{text}"}],
            "n": n,
            "temperature": 0.0,
        }
    ).encode()


def chat(server: FakeChatServer, text: str, n: int = 1) -> tuple[int, Any]:
    status, _, data = exchange(
        server,
        "POST",
        CHAT,
        request_body(text, n),
        {"Content-Type": "application/json"},
    )
    return status, json.loads(data)


def refuses(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError:
        return True
    return False


def alive() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == THREAD]


# ---- the address ------------------------------------------------------------------------------------


def test_the_base_url_is_the_loopback_with_a_port_and_the_v1_prefix(
    server: FakeChatServer,
) -> None:
    parts = urlsplit(server.base_url)

    assert parts.scheme == "http"
    assert parts.hostname == "127.0.0.1"
    assert parts.port is not None and parts.port > 0
    assert parts.path == "/v1"
    assert parts.query == "" and parts.fragment == "" and parts.username is None


def test_no_name_is_resolved_to_start_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a name was resolved")

    for name in ("getfqdn", "gethostbyaddr", "gethostbyname", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)

    with FakeChatServer(ORACLE) as server:
        assert address(server)[0] == "127.0.0.1"


def test_two_servers_run_side_by_side_on_their_own_ports() -> None:
    with FakeChatServer(ORACLE) as first, FakeChatServer({"x": "y"}) as second:
        assert address(first) != address(second)
        assert (
            chat(first, "one text")[1]["choices"][0]["message"]["content"]
            == "its answer"
        )
        assert chat(second, "x")[1]["choices"][0]["message"]["content"] == "y"
        assert chat(second, "one text")[1]["choices"][0]["message"]["content"] == ""


def test_the_listener_is_bound_to_the_loopback_address_and_no_other(
    server: FakeChatServer,
) -> None:
    host, port = address(server)

    assert not refuses(host, port)
    assert refuses("127.0.0.2", port)


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "localhost",
        "",
        "::1",
        "::",
        "127.0.0.2",
        "192.168.0.1",
        "127.0.0.1 ",
        "LOCALHOST",
    ],
)
def test_a_host_that_is_not_the_loopback_address_is_a_configuration_error(
    host: str,
) -> None:
    with pytest.raises(ConfigError) as error:
        FakeChatServer(host=host)
    with pytest.raises(ConfigError):
        FakeChatServer(ORACLE, host=host)

    if host.strip():
        assert host not in str(error.value)


def test_the_loopback_address_is_the_default_and_can_be_named() -> None:
    with FakeChatServer(ORACLE, host="127.0.0.1") as named, FakeChatServer() as default:
        assert address(named)[0] == address(default)[0] == "127.0.0.1"


# ---- what it answers --------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 64])
def test_a_known_text_gets_the_oracle_text_in_each_of_n_choices(
    server: FakeChatServer, n: int
) -> None:
    status, reply = chat(server, "one text", n)

    assert status == 200
    assert reply == chat_reply(["its answer"] * n)


def test_an_unknown_text_gets_an_empty_choice() -> None:
    with FakeChatServer(ORACLE) as server:
        status, reply = chat(server, "not in the oracle", 3)

    assert status == 200
    assert reply == chat_reply(["", "", ""])


def test_the_source_is_what_follows_the_first_blank_line() -> None:
    oracle = {"first\n\nsecond": "both", "second": "the tail only"}
    with FakeChatServer(oracle) as server:
        _, both = chat(server, "first\n\nsecond")
        _, tail = chat(server, "second")

    assert both["choices"][0]["message"]["content"] == "both"
    assert tail["choices"][0]["message"]["content"] == "the tail only"


def test_a_message_without_a_blank_line_has_the_empty_source(
    server: FakeChatServer,
) -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": "only one line"}]}
    ).encode()

    status, _, data = exchange(server, "POST", CHAT, body)

    assert status == 200
    assert json.loads(data) == chat_reply(["for nothing"])


def test_only_the_first_message_is_read(server: FakeChatServer) -> None:
    messages = [
        {"role": "system", "content": "head\n\nsecond"},
        {"role": "user", "content": "x\n\none text"},
    ]
    body = json.dumps({"messages": messages}).encode()

    _, _, data = exchange(server, "POST", CHAT, body)

    assert json.loads(data)["choices"][0]["message"]["content"] == "another answer"


def test_the_number_of_choices_defaults_to_one(server: FakeChatServer) -> None:
    body = json.dumps({"messages": [{"content": "p\n\none text"}]}).encode()

    _, _, data = exchange(server, "POST", CHAT, body)

    assert json.loads(data) == chat_reply(["its answer"])


def test_the_reply_is_json_with_its_length(server: FakeChatServer) -> None:
    status, headers, data = exchange(
        server,
        "POST",
        CHAT,
        request_body("one text"),
        {"Content-Type": "application/json"},
    )

    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert int(headers["Content-Length"]) == len(data)


def test_the_real_client_library_gets_the_same_reply(server: FakeChatServer) -> None:
    with httpx.Client(trust_env=False, timeout=10) as client:
        reply = client.post(
            f"{server.base_url}/chat/completions",
            json={"messages": [{"role": "user", "content": "p\n\none text"}], "n": 2},
        )

    assert reply.status_code == 200
    assert reply.json() == chat_reply(["its answer"] * 2)


def test_the_oracle_is_copied_when_the_server_is_made() -> None:
    oracle = {"a": "b"}
    server = FakeChatServer(oracle)
    oracle["a"] = "changed"
    oracle["c"] = "added"

    with server:
        assert chat(server, "a")[1]["choices"][0]["message"]["content"] == "b"
        assert chat(server, "c")[1]["choices"][0]["message"]["content"] == ""


# ---- what it refuses --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/models"),
        ("GET", CHAT),
        ("GET", "/"),
        ("POST", "/v1/models"),
        ("POST", "/"),
        ("POST", CHAT + "/"),
        ("POST", CHAT + "?stream=true"),
        ("POST", "/v1/chat"),
        ("POST", "/v2/chat/completions"),
        ("POST", "v1/chat/completions"),
        ("PUT", CHAT),
        ("DELETE", CHAT),
        ("PATCH", CHAT),
        ("HEAD", CHAT),
        ("OPTIONS", CHAT),
        ("BREW", CHAT),
    ],
)
def test_any_other_path_or_method_is_a_404(
    server: FakeChatServer, method: str, path: str
) -> None:
    status, _, data = exchange(
        server, method, path, request_body("one text") if method != "GET" else None
    )

    assert status == 404
    if method != "HEAD":
        assert "one text" not in data.decode()
    assert server.requests_seen == 0


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b"3",
        b'{"messages": []}',
        b'{"messages": "x"}',
        b'{"messages": [3]}',
        b'{"messages": [{}]}',
        b'{"messages": [{"content": 3}]}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": 0}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": -1}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": 65}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": "2"}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": 1.0}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": true}',
        b'{"messages": [{"content": "a\\n\\nb"}], "n": null}',
        b"\xff\xfe",
    ],
)
def test_a_body_that_is_not_a_chat_request_is_a_400_that_quotes_nothing(
    server: FakeChatServer, body: bytes
) -> None:
    status, _, data = exchange(server, "POST", CHAT, body)

    assert status == 400
    assert json.loads(data)["error"]["message"]
    assert server.requests_seen == 0


def test_a_refusal_does_not_echo_what_it_was_sent(server: FakeChatServer) -> None:
    body = json.dumps(
        {"messages": [{"content": "SENTINEL-TEXT"}], "n": "SENTINEL-N"}
    ).encode()

    status, _, data = exchange(server, "POST", CHAT, body)

    assert status == 400
    assert b"SENTINEL" not in data


def test_a_request_without_a_length_is_a_400(server: FakeChatServer) -> None:
    host, port = address(server)
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.putrequest("POST", CHAT)
        connection.endheaders()
        response = connection.getresponse()
        status, data = response.status, response.read()
    finally:
        connection.close()

    assert status == 400
    assert json.loads(data)["error"]["message"] == "a body with a length is needed"


@pytest.mark.parametrize("declared", ["abc", "-1", "2097152"])
def test_a_length_that_is_not_a_number_or_is_too_large_is_refused_before_the_body_is_read(
    server: FakeChatServer, declared: str
) -> None:
    host, port = address(server)
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.putrequest("POST", CHAT)
        connection.putheader("Content-Length", declared)
        connection.endheaders()
        status = connection.getresponse().status
    finally:
        connection.close()

    assert status in (400, 413)
    assert server.requests_seen == 0


def test_a_client_that_sends_garbage_or_hangs_up_does_not_stop_the_server(
    server: FakeChatServer,
) -> None:
    host, port = address(server)
    with socket.create_connection((host, port), timeout=3) as garbage:
        garbage.sendall(b"\x00\x01 not http at all\r\n\r\n")
    with socket.create_connection((host, port), timeout=3):
        pass

    assert chat(server, "one text")[0] == 200


# ---- the count --------------------------------------------------------------------------------------


def test_the_count_is_the_number_of_chat_requests_that_were_answered(
    server: FakeChatServer,
) -> None:
    assert server.requests_seen == 0

    chat(server, "one text")
    chat(server, "unknown")
    exchange(server, "GET", "/v1/models")
    exchange(server, "POST", CHAT, b"not json")
    chat(server, "second", 3)

    assert server.requests_seen == 3


def test_the_count_can_be_read_after_the_server_has_stopped() -> None:
    with FakeChatServer(ORACLE) as server:
        chat(server, "one text")
        chat(server, "one text")

    assert server.requests_seen == 2


# ---- the end ----------------------------------------------------------------------------------------


def test_after_the_block_the_port_refuses_a_connection_and_the_thread_is_joined() -> (
    None
):
    before = len(alive())
    with FakeChatServer(ORACLE) as server:
        host, port = address(server)
        assert len(alive()) == before + 1
        assert not refuses(host, port)

    assert refuses(host, port)
    assert len(alive()) == before


def test_the_listening_socket_is_closed_by_the_block_and_not_left_to_the_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    original = socket.socket.close

    def spy(self: socket.socket) -> None:
        closed.append(self.fileno())
        original(self)

    monkeypatch.setattr(socket.socket, "close", spy)

    with FakeChatServer(ORACLE):
        assert closed == []

    assert len(closed) == 1
    assert closed[0] >= 0


def test_the_serving_thread_is_joined_on_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joined: list[str] = []
    original = threading.Thread.join

    def spy(self: threading.Thread, timeout: float | None = None) -> None:
        joined.append(self.name)
        original(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", spy)

    with FakeChatServer(ORACLE):
        assert joined == []

    assert joined == [THREAD]


def test_after_an_exception_in_the_block_the_port_refuses_a_connection_and_it_propagates() -> (
    None
):
    before = len(alive())
    server = FakeChatServer(ORACLE)
    with pytest.raises(RuntimeError, match="inside"), server:
        host, port = address(server)
        raise RuntimeError("inside")

    assert refuses(host, port)
    assert len(alive()) == before


def test_the_block_ends_at_once_when_nothing_was_ever_asked() -> None:
    with FakeChatServer(ORACLE):
        pass


def test_a_server_that_was_stopped_can_run_again_on_another_port() -> None:
    server = FakeChatServer(ORACLE)
    with server:
        first = address(server)
    with server:
        second = address(server)
        assert chat(server, "one text")[0] == 200

    assert refuses(*second)
    assert server.requests_seen == 1
    assert first[0] == second[0]


def test_the_base_url_is_not_there_outside_the_block() -> None:
    server = FakeChatServer(ORACLE)

    with pytest.raises(RuntimeError, match="not running"):
        server.base_url  # noqa: B018 - the access is the test
    with server:
        server.base_url  # noqa: B018
    with pytest.raises(RuntimeError, match="not running"):
        server.base_url  # noqa: B018


def test_a_server_that_is_running_cannot_be_entered_again() -> None:
    with FakeChatServer(ORACLE) as server:
        with pytest.raises(RuntimeError, match="already running"):
            server.__enter__()
        assert chat(server, "one text")[0] == 200
    assert alive() == []


# ---- concurrency ------------------------------------------------------------------------------------


def test_sixty_four_concurrent_requests_are_all_answered() -> None:
    oracle = {f"text {n}": f"answer {n}" for n in range(64)}
    with FakeChatServer(oracle) as server, ThreadPoolExecutor(max_workers=64) as pool:
        replies = list(pool.map(lambda n: chat(server, f"text {n}", 2), range(64)))

    assert [status for status, _ in replies] == [200] * 64
    for n, (_, reply) in enumerate(replies):
        assert reply == chat_reply([f"answer {n}"] * 2)
    assert server.requests_seen == 64
