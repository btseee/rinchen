"""The upload of a staged release to the Hub: its plan, and the one commit that carries it (D13, spec T18).

`plan_upload(stage, repo_id)` lists the nine files of `UPLOAD_NAMES` of a stage that
`rinchen release stage` wrote, each with the size and SHA-256 it has now (every file is read once, in
one pass that also scans the text files), and refuses the stage, naming each FILE and the kind of
problem and never a text, when anything is not as it should be: a name outside the list, a directory,
a link or another file that is not a regular one, a file over `MAX_FILE_BYTES`, a file that is not
what `release.json` lists, a name `release.json` lists that is absent, a text file that looks like a
token or holds an absolute local path (`rinchen.export.hub_scan`). `manifest.json` is the stage's own
and is known and left out (it holds the paths of the machine that made it). `UploadPlan.to_bytes()` is
`hub_plan.json`: sorted keys, LF, no timestamp, no path and no token. The reading of the stage is
`rinchen.export.hub_files`' and the release record's is `rinchen.export.hub_record`'s.

`upload(plan, stage, client, token)` re-hashes every file of the plan just before the operations are
built and refuses a change, creates the repository (private unless `public`), and makes ONE commit of
the nine files in the order of `UPLOAD_NAMES`. It talks to the Hub only through a `HubClient`. An error
of the client is turned into an `UpstreamError` that names its class and nothing else (its text may
hold the token), raised outside the `except` block so that the error is not kept as its cause or its
context. A file could still change between the re-hash and the moment the library reads it: the stage
is the operator's, and the commit records the digest of `release.json` that vouches for the others.

`HfHubClient` is the client for the real Hub and the one place that imports the Hub library, inside its
constructor. It is built with the token and passes it to the library's client; it never asks the library
to log in, never reads a token file and never writes the token anywhere. Nothing in this module reads
the environment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple, Protocol

from rinchen.errors import ConfigError, DataError, RinchenError, UpstreamError
from rinchen.export.hub_files import Problem, Seen, read_hashed, read_stage, shown
from rinchen.export.hub_record import Listed, NotARecord, read_record
from rinchen.export.release import (
    RELEASE_FILE,
    UPLOAD_NAMES,
    check_repo_id,
)

__all__ = [
    "HUB_COMMIT_SCHEMA",
    "HUB_PLAN_SCHEMA",
    "MAX_FILE_BYTES",
    "MAX_TEXT_BYTES",
    "CommitResult",
    "HfHubClient",
    "HubClient",
    "PlannedFile",
    "UploadOperation",
    "UploadPlan",
    "commit_to_bytes",
    "plan_upload",
    "upload",
]

HUB_PLAN_SCHEMA: Final = "rinchen.hub_plan/1"
HUB_COMMIT_SCHEMA: Final = "rinchen.hub_commit/1"
# What the Hub takes in one file, and what a text file may weigh to be read whole and scanned.
MAX_FILE_BYTES: Final = 10 * 2**30
MAX_TEXT_BYTES: Final = 64 * 2**20
_OID: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_PROBLEMS_SHOWN: Final = 12
# The kinds of problem a message names after a file (those of reading it are `hub_files`').
_CHANGED: Final = "differs from what release.json lists"
_LISTED_ABSENT: Final = "listed in release.json but absent"
_UNLISTED: Final = "not listed in release.json"
_ABSENT: Final = "absent from the stage"
_NOT_LISTABLE: Final = "not a name release.json may list"
_LISTABLE: Final = tuple(name for name in UPLOAD_NAMES if name != RELEASE_FILE)


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """One file of the upload: its name in the repository and in the stage, its size and SHA-256."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class UploadPlan:
    """The nine files of an upload in the order of `UPLOAD_NAMES`, their total size and the SHA-256
    of `release.json`, which vouches for the others."""

    repo_id: str
    files: tuple[PlannedFile, ...]
    total_bytes: int
    release_sha256: str

    def to_bytes(self) -> bytes:
        """The bytes of `hub_plan.json`."""
        return _json_bytes(
            {
                "schema": HUB_PLAN_SCHEMA,
                "repo_id": self.repo_id,
                "release_sha256": self.release_sha256,
                "total_bytes": self.total_bytes,
                "files": [
                    {"name": file.name, "bytes": file.bytes, "sha256": file.sha256}
                    for file in self.files
                ],
            }
        )


class UploadOperation(NamedTuple):
    """A file to add to the commit: the name it has in the repository and where it is."""

    path_in_repo: str
    path: Path


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What the Hub made of the commit: its id."""

    commit_oid: str


class HubClient(Protocol):
    """The two calls an upload makes. `token` is the one the upload was given."""

    def create_repo(
        self, repo_id: str, *, private: bool, exist_ok: bool, token: str
    ) -> None: ...

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[UploadOperation],
        *,
        commit_message: str,
        token: str,
    ) -> CommitResult: ...


# ---- the plan ---------------------------------------------------------------------------------------------------


def plan_upload(stage: Path, repo_id: str) -> UploadPlan:
    """The plan of the upload of `stage` to `repo_id`.

    Raises:
        ConfigError: `repo_id` is not `ORG/NAME`, or is not the repository the release record of the
            stage names.
        DataError: `stage` is not a directory, or is not a release that can be uploaded; the
            message names every file at fault and the kind of problem, never a text.
    """
    check_repo_id(repo_id)
    seen, problems = read_stage(
        stage, max_file_bytes=MAX_FILE_BYTES, max_text_bytes=MAX_TEXT_BYTES
    )
    listed = _record_of(seen, problems)
    if listed is not None and listed.repo_id != repo_id:
        raise ConfigError(
            "--repo-id is not the repository that release.json says this stage was made for"
        )
    problems += _against_the_record(seen, listed, {name for name, _ in problems})
    _refuse(problems)
    files = tuple(
        PlannedFile(name, seen[name].size, seen[name].sha256) for name in UPLOAD_NAMES
    )
    return UploadPlan(
        repo_id=repo_id,
        files=files,
        total_bytes=sum(file.bytes for file in files),
        release_sha256=seen[RELEASE_FILE].sha256,
    )


def _record_of(seen: Mapping[str, Seen], problems: list[Problem]) -> Listed | None:
    """The release record of the stage, or None (with a problem recorded) when there is none."""
    record = seen.get(RELEASE_FILE)
    if record is None or record.data is None:
        if all(name != RELEASE_FILE for name, _ in problems):
            problems.append((RELEASE_FILE, _ABSENT))
        return None
    try:
        return read_record(record.data)
    except NotARecord as error:
        problems.append((RELEASE_FILE, f"not a release record: {error.detail}"))
        return None


def _against_the_record(
    seen: Mapping[str, Seen], listed: Listed | None, named: set[str]
) -> list[Problem]:
    """What the files found say against what the release record lists (nothing is said of a file
    that was refused before it was read: it is already named). Without a usable record only the
    absent files are named."""
    problems: list[Problem] = []
    vouched_for = {} if listed is None else listed.files
    problems += [
        (name, _NOT_LISTABLE) for name in sorted(vouched_for) if name not in _LISTABLE
    ]
    for name in _LISTABLE:
        entry = seen.get(name)
        if entry is None:
            if name not in named:
                problems.append(
                    (name, _LISTED_ABSENT if name in vouched_for else _ABSENT)
                )
        elif listed is not None and name not in vouched_for:
            problems.append((name, _UNLISTED))
        elif listed is not None and vouched_for[name] != (entry.size, entry.sha256):
            problems.append((name, _CHANGED))
    return problems


def _refuse(problems: Sequence[Problem]) -> None:
    if not problems:
        return
    shown_problems = [
        f"{shown(name)} ({kind})" for name, kind in problems[:_PROBLEMS_SHOWN]
    ]
    if len(problems) > _PROBLEMS_SHOWN:
        shown_problems.append(f"and {len(problems) - _PROBLEMS_SHOWN} more")
    raise DataError(
        "the stage is not a release that can be uploaded: " + "; ".join(shown_problems)
    )


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    return f"{text}\n".encode()


# ---- the upload ----------------------------------------------------------------------------------------------------


def upload(
    plan: UploadPlan,
    stage: Path,
    client: HubClient,
    token: str,
    *,
    public: bool = False,
) -> CommitResult:
    """Create the repository of `plan` (private unless `public`) and make ONE commit of its nine
    files, through `client`.

    Raises:
        ConfigError: `token` is blank, or a file of the plan is not what it was when it was
            planned (nothing was sent).
        UpstreamError: the Hub, or the client, failed; the message names the class of the error and
            never its text, which may hold the token.
    """
    if not token.strip():
        raise ConfigError("the token is blank")
    name = _name_of_the_model(_require_unchanged(plan, stage))
    operations = [UploadOperation(file.name, stage / file.name) for file in plan.files]
    _through_the_client(
        "the Hub refused the repository",
        lambda: client.create_repo(
            plan.repo_id, private=not public, exist_ok=True, token=token
        ),
    )
    result = _through_the_client(
        "the Hub refused the upload",
        lambda: client.create_commit(
            plan.repo_id,
            operations,
            commit_message=f"Rinchen {name}",
            token=token,
        ),
    )
    if not isinstance(result.commit_oid, str) or not _OID.fullmatch(result.commit_oid):
        raise UpstreamError("the Hub answered with a commit id that is not one")
    return result


def commit_to_bytes(plan: UploadPlan, result: CommitResult) -> bytes:
    """The bytes of `hub_commit.json`: the repository, the commit and the release it carried."""
    return _json_bytes(
        {
            "schema": HUB_COMMIT_SCHEMA,
            "repo_id": plan.repo_id,
            "commit_oid": result.commit_oid,
            "release_sha256": plan.release_sha256,
        }
    )


def _require_unchanged(plan: UploadPlan, stage: Path) -> bytes:
    """Hash every file of the plan again and return the bytes of `release.json` as they are.

    Raises:
        ConfigError: a file is gone, is no longer a regular file or is not what the plan says.
    """
    release = b""
    for file in plan.files:
        keep = file.name == RELEASE_FILE
        read = read_hashed(
            stage / file.name,
            keep=keep,
            limit=MAX_TEXT_BYTES if keep else MAX_FILE_BYTES,
        )
        if isinstance(read, str) or (read.size, read.sha256) != (
            file.bytes,
            file.sha256,
        ):
            raise ConfigError(
                f"{file.name} changed after the plan was made, so what would be uploaded is "
                "not what was planned"
            )
        if read.data is not None:
            release = read.data
    return release


def _name_of_the_model(release: bytes) -> str:
    try:
        return read_record(release).name
    except NotARecord as error:
        raise ConfigError(
            f"{RELEASE_FILE} is not a release record: {error.detail}"
        ) from None


def _through_the_client[T](failure: str, call: Callable[[], T]) -> T:
    """The answer of `call`, or an `UpstreamError` that says `failure` and names the class of
    the error the client raised: the error itself is not kept, its text may hold the token. An
    error of ours and an interrupt pass through."""
    try:
        return call()
    except RinchenError:
        raise
    except Exception as error:  # noqa: BLE001 -- any error of the client is named by its class
        refused = type(error).__name__
    raise UpstreamError(f"{failure} ({refused})")


# ---- the client for the real Hub ----------------------------------------------------------------------------------------


class HfHubClient:
    """The `HubClient` for the Hub: the library's `HfApi`, built with the token, for a repository
    of the kind `model`. The library is imported here and nowhere else."""

    def __init__(self, token: str) -> None:
        if not token.strip():
            raise ConfigError("the token is blank")
        from huggingface_hub import CommitOperationAdd, HfApi

        self._token = token
        self._add: Callable[..., Any] = CommitOperationAdd
        self._api: Any = HfApi(token=token)

    def create_repo(
        self, repo_id: str, *, private: bool, exist_ok: bool, token: str
    ) -> None:
        self._require_own(token)
        self._api.create_repo(
            repo_id, private=private, exist_ok=exist_ok, repo_type="model"
        )

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[UploadOperation],
        *,
        commit_message: str,
        token: str,
    ) -> CommitResult:
        self._require_own(token)
        added = [
            self._add(path_in_repo=item.path_in_repo, path_or_fileobj=str(item.path))
            for item in operations
        ]
        info = self._api.create_commit(
            repo_id=repo_id,
            operations=added,
            commit_message=commit_message,
            repo_type="model",
        )
        return CommitResult(info.oid)

    def _require_own(self, token: str) -> None:
        if token != self._token:
            raise ConfigError(
                "the client was built with another token than the call gave"
            )
