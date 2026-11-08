"""Tests for `tests.support.signal_handlers`: the handlers a signal test installs are in
place for its block only. SIGUSR1 stands in, so no test here changes the dispositions the
suite relies on."""

from __future__ import annotations

import signal

import pytest

from tests.support.signal_handlers import DEFAULT_HANDLERS, handlers_installed

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs SIGUSR1")


def _handler(signum: int, frame: object) -> None:
    raise AssertionError("not meant to run")


def test_the_handlers_are_in_place_inside_the_block_and_gone_after_it() -> None:
    before = signal.getsignal(signal.SIGUSR1)

    with handlers_installed({signal.SIGUSR1: _handler}):
        assert signal.getsignal(signal.SIGUSR1) is _handler

    assert signal.getsignal(signal.SIGUSR1) == before


def test_the_previous_handlers_come_back_when_the_block_raises() -> None:
    before = signal.getsignal(signal.SIGUSR1)

    with pytest.raises(RuntimeError), handlers_installed({signal.SIGUSR1: _handler}):
        raise RuntimeError("the test failed")

    assert signal.getsignal(signal.SIGUSR1) == before


def test_an_ignored_signal_is_restored_as_ignored() -> None:
    before = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        with handlers_installed({signal.SIGUSR1: _handler}):
            pass

        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGUSR1, before)


def test_the_defaults_are_what_a_terminal_session_starts_with() -> None:
    assert DEFAULT_HANDLERS[signal.SIGINT] is signal.default_int_handler
    assert DEFAULT_HANDLERS[signal.SIGTERM] is signal.SIG_DFL
    assert DEFAULT_HANDLERS[signal.SIGHUP] is signal.SIG_DFL
