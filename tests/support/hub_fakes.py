"""Fakes of the Hub for the tests of the upload plan and of `rinchen release hub` (D13, Task 10).

No test of D13 talks to the Hub: `RecordingHub` stands in for the `HubClient` that `upload` is given
(it records every call, with the paths of the operations, and can be told to raise an error whose
text holds the token), `FakeHfApi` stands in for `huggingface_hub.HfApi` under the real adapter
`HfHubClient` (it records its calls and raises if anything that writes or reads a token is touched)
and `install_open_client` replaces the seam `open_client` of the command, so that no real client is
ever built. Nothing here opens a socket, reads the environment or touches a token file; the token of
a test is `fake_token()`, a recognisably fake string that is put together when the test runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NoReturn

import pytest

from rinchen.export.hub import CommitResult, UploadOperation

COMMIT_OID: Final = "0123456789abcdef0123456789abcdef01234567"
_FAKE_PIECES: Final = ("hf", "_", "FAKE" * 8)


def fake_token(suffix: str = "") -> str:
    """A token-like string that is not one, built from pieces so that no source file holds it (it
    matches the scan of the plan, which is the point: if it ever reached a file the plan would
    refuse the file, and a test that looks for it in every output would find it)."""
    return "".join(_FAKE_PIECES) + suffix


class TokenInError(RuntimeError):
    """An upstream error whose text holds the token, as a library may report a rejected request."""


@dataclass(frozen=True)
class RepoCall:
    repo_id: str
    private: bool
    exist_ok: bool
    token: str


@dataclass(frozen=True)
class CommitCall:
    repo_id: str
    operations: tuple[tuple[str, Path], ...]
    commit_message: str
    token: str


@dataclass
class RecordingHub:
    """A `HubClient` that records `create_repo` and `create_commit` in `calls` (in order) and
    answers with `oid`. `repo_error` and `commit_error` are raised instead of answering (after the
    call is recorded)."""

    oid: str = COMMIT_OID
    repo_error: BaseException | None = None
    commit_error: BaseException | None = None
    calls: list[RepoCall | CommitCall] = field(default_factory=list)

    def create_repo(
        self, repo_id: str, *, private: bool, exist_ok: bool, token: str
    ) -> None:
        self.calls.append(RepoCall(repo_id, private, exist_ok, token))
        if self.repo_error is not None:
            raise self.repo_error

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[UploadOperation],
        *,
        commit_message: str,
        token: str,
    ) -> CommitResult:
        self.calls.append(
            CommitCall(
                repo_id,
                tuple((op.path_in_repo, op.path) for op in operations),
                commit_message,
                token,
            )
        )
        if self.commit_error is not None:
            raise self.commit_error
        return CommitResult(self.oid)

    @property
    def repo_calls(self) -> list[RepoCall]:
        return [call for call in self.calls if isinstance(call, RepoCall)]

    @property
    def commit_calls(self) -> list[CommitCall]:
        return [call for call in self.calls if isinstance(call, CommitCall)]


@dataclass(frozen=True)
class FakeCommitInfo:
    """What the library's `CommitInfo` gives the adapter: the commit's id."""

    oid: str


@dataclass
class FakeHfApi:
    """A stand-in for `huggingface_hub.HfApi`. The class records the token each instance was
    built with (`built_with`) and the calls of every instance (`calls`); anything else asked of an
    instance (a `login`, a `whoami`, a token read) raises, so a call the adapter must not make fails
    the test at once."""

    built_with: list[str | None] = field(default_factory=list, repr=False)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    oid: str = COMMIT_OID

    def __call__(self, token: str | None = None) -> FakeHfApi:
        self.built_with.append(token)
        return self

    def create_repo(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("create_repo", args, kwargs))

    def create_commit(self, *args: Any, **kwargs: Any) -> FakeCommitInfo:
        self.calls.append(("create_commit", args, kwargs))
        return FakeCommitInfo(self.oid)

    def __getattr__(self, name: str) -> NoReturn:
        raise AssertionError(f"the adapter touched HfApi.{name}")


def forbid_token_handling(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make `huggingface_hub.login`, `HfFolder` and `get_token` raise when they are used; the list
    that is returned stays empty while nothing touches them."""
    import huggingface_hub

    touched: list[str] = []

    def refuse(name: str) -> Callable[..., NoReturn]:
        def refused(*args: Any, **kwargs: Any) -> NoReturn:
            touched.append(name)
            raise AssertionError(f"the adapter used huggingface_hub.{name}")

        return refused

    # `HfFolder` is gone from the installed library; the guard is there for a version that has it.
    for name in ("login", "HfFolder", "get_token", "whoami"):
        monkeypatch.setattr(huggingface_hub, name, refuse(name), raising=False)
    return touched


def install_open_client(
    monkeypatch: pytest.MonkeyPatch, hub: RecordingHub | None = None
) -> list[str]:
    """Make the command's seam `open_client` hand out `hub` (or fail the test when none is given);
    the list that is returned holds the token of each call."""
    opened: list[str] = []

    def opening(token: str) -> RecordingHub:
        opened.append(token)
        assert hub is not None, "a client was opened"
        return hub

    monkeypatch.setattr("rinchen.cli.release_hub.open_client", opening)
    return opened
