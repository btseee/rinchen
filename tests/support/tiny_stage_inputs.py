"""What the tests of the library stages of the tiny pipeline hand them (D15): the shipped profile,
the committed fixtures and a `StageContext` over a temporary output directory.

The stages that run in process start no command, so the runner of the context is one that fails the
test when it is called; the clock of the budget is a constant.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Final, NoReturn

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import TinyConfig, load_tiny_config
from rinchen.maintain.commands import CommandResult
from rinchen.tiny.context import StageContext
from rinchen.tiny.fixtures import Fixtures, read_fixtures
from tests.support.no_network import REFUSAL

CONFIG_PATH: Final = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"
LOCK_PATH: Final = REPO_ROOT / "uv.lock"
BUDGET_LEFT: Final = 1200.0


class NoCommands:
    """A runner that fails the test: a library stage must not start a command."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        self._refuse()

    @staticmethod
    def _refuse() -> NoReturn:
        raise AssertionError("a library stage started a command")


def _refuse(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(REFUSAL)


@contextmanager
def offline_in(directory: Path) -> Iterator[None]:
    """Run from `directory` (made if need be) with the network refused, as the autouse fixtures of
    the stage tests do, for a fixture of a wider scope than a test, which they do not reach."""
    directory.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(directory)
        patch.setattr(socket.socket, "connect", _refuse)
        patch.setattr(socket.socket, "connect_ex", _refuse)
        patch.setattr(socket, "create_connection", _refuse)
        patch.setattr(socket, "getaddrinfo", _refuse)
        yield


@cache
def shipped_config() -> TinyConfig:
    return load_tiny_config(CONFIG_PATH.read_bytes())


@cache
def shipped_fixtures() -> Fixtures:
    config = shipped_config()
    return read_fixtures(config.fixtures.directory, config.fixtures.sha256)


def make_context(
    out_dir: Path,
    *,
    config: TinyConfig | None = None,
    fixtures: Fixtures | None = None,
    qe_fake: bool = True,
) -> StageContext:
    """A context over `out_dir` with the shipped profile and fixtures unless others are given."""
    used = config if config is not None else shipped_config()
    return StageContext(
        config=used,
        fixtures=fixtures if fixtures is not None else shipped_fixtures(),
        out_dir=out_dir,
        runner=NoCommands(),
        qe_fake=qe_fake,
        threads=used.threads,
        remaining_seconds=lambda: BUDGET_LEFT,
    )
