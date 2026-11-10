"""The guard of the D13 tests: while `no_network` is in force no socket connects and no name is
resolved, so a library that reaches for the network (sacrebleu fetching a tokenizer model on its
first use, a dataset or a Hub client) fails the test at once instead of downloading something.

The guard refuses `socket.socket.connect`, `connect_ex`, `socket.create_connection` and
`socket.getaddrinfo` with an `AssertionError`. A socket that is only created or bound is not
affected, so a test that serves on the loopback still can. A module applies it with
`pytestmark = pytest.mark.usefixtures("no_network")`.
"""

from __future__ import annotations

import socket
from typing import NoReturn

import pytest

REFUSAL = "network access in a test"


def _refuse(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(REFUSAL)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
