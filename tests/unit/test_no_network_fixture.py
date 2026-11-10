"""The `no_network` guard of the D13 tests: each of the four ways to reach the network raises
while it is in force, and the guard goes away with the test."""

from __future__ import annotations

import socket

import pytest

from tests.support.no_network import no_network  # noqa: F401 - the fixture


@pytest.mark.usefixtures("no_network")
def test_every_way_to_connect_or_resolve_raises_while_the_guard_is_in_force() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        for attempt in (
            lambda: probe.connect(("127.0.0.1", 9)),
            lambda: probe.connect_ex(("127.0.0.1", 9)),
            lambda: socket.create_connection(("127.0.0.1", 9)),
            lambda: socket.getaddrinfo("localhost", 9),
        ):
            with pytest.raises(AssertionError, match="network access in a test"):
                attempt()


@pytest.mark.usefixtures("no_network")
def test_creating_a_connection_is_refused_by_the_guard_and_not_only_by_the_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `create_connection` resolves the name first; with the resolver out of the way it is
    # the guard's own refusal that stops it, not an error of the operating system.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [])

    with pytest.raises(AssertionError, match="network access in a test"):
        socket.create_connection(("127.0.0.1", 9))


@pytest.mark.usefixtures("no_network")
def test_a_socket_that_is_only_bound_to_the_loopback_is_not_affected() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))

        assert server.getsockname()[0] == "127.0.0.1"


def test_the_guard_does_not_outlive_the_test_that_asked_for_it() -> None:
    # Not requested here: the library's own functions are back, none of them the guard's.
    assert socket.socket.connect.__name__ == "connect"
    assert socket.socket.connect_ex.__name__ == "connect_ex"
    assert socket.create_connection.__name__ == "create_connection"
    assert socket.getaddrinfo.__name__ == "getaddrinfo"
