"""How the upload plan reads one file (D13, Task 10, spec T18): `read_hashed` opens a file without
following a link and without waiting for a pipe, reads it once and answers its size, SHA-256 and (when
asked) its bytes, or the kind of problem that stopped it: not a regular file, over the limit, or cannot be
read. A name in a message is shown as it is when it is a name of the upload and escaped and cut otherwise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from rinchen.export.hub_files import (
    NOT_REGULAR,
    TOO_BIG,
    UNREADABLE,
    Seen,
    read_hashed,
    shown,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_file_is_read_once_into_its_size_and_digest(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"abc" * 100)

    assert read_hashed(tmp_path / "f", keep=False, limit=1000) == Seen(
        300, digest(b"abc" * 100), None
    )


def test_the_bytes_are_kept_only_when_asked_for(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"abc")

    assert read_hashed(tmp_path / "f", keep=True, limit=10) == Seen(
        3, digest(b"abc"), b"abc"
    )


def test_a_file_larger_than_one_chunk_is_hashed_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rinchen.export.hub_files as files

    monkeypatch.setattr(files, "_CHUNK_BYTES", 7)
    data = bytes(range(256)) * 3
    (tmp_path / "f").write_bytes(data)

    assert read_hashed(tmp_path / "f", keep=True, limit=10_000) == Seen(
        len(data), digest(data), data
    )


def test_an_empty_file_has_the_digest_of_nothing(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"")

    assert read_hashed(tmp_path / "f", keep=True, limit=0) == Seen(0, digest(b""), b"")


@pytest.mark.parametrize("size, answer", [(9, "seen"), (10, "seen"), (11, TOO_BIG)])
def test_a_file_over_the_limit_is_refused_while_it_is_read(
    tmp_path: Path, size: int, answer: str
) -> None:
    (tmp_path / "f").write_bytes(b"x" * size)

    read = read_hashed(tmp_path / "f", keep=True, limit=10)

    assert (read if isinstance(read, str) else "seen") == answer


def test_a_link_is_never_followed_even_to_a_regular_file(tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"abc")
    (tmp_path / "link").symlink_to(tmp_path / "target")

    assert read_hashed(tmp_path / "link", keep=True, limit=10) == NOT_REGULAR


def test_a_missing_file_cannot_be_read(tmp_path: Path) -> None:
    assert read_hashed(tmp_path / "missing", keep=True, limit=10) == UNREADABLE


def test_a_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()

    assert read_hashed(tmp_path / "d", keep=True, limit=10) in (NOT_REGULAR, UNREADABLE)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="this platform has no FIFO")
@pytest.mark.timeout(30)
def test_a_pipe_is_refused_and_never_waited_for(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    assert read_hashed(tmp_path / "pipe", keep=True, limit=10) == NOT_REGULAR


# ---- how a name is shown ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["model.bin", "release.json", "README.md", "manifest.json"]
)
def test_the_name_of_a_file_of_the_upload_is_shown_as_it_is(name: str) -> None:
    assert shown(name) == name


def test_any_other_name_is_escaped_and_cut() -> None:
    assert shown("notes.txt") == "notes.txt"
    assert shown("a" + chr(0xE9) + chr(0x0A)) == "a\\xe9\\n"
    cut = shown("y" * 200)
    assert cut == "y" * 80 + "..."
    assert shown("y" * 80) == "y" * 80
