"""The unique temporary name a writer uses before it renames its file into place,
`replace_file`, which writes a file whole, and the bytes of a file of lines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from rinchen.utils.part_file import lines_as_bytes, replace_file, unique_part_path


def test_the_part_path_is_a_part_file_beside_the_target(tmp_path: Path) -> None:
    target = tmp_path / "out" / "corpus.parquet"

    part = unique_part_path(target)

    assert part.parent == target.parent
    assert part.name.startswith("corpus.parquet.")
    assert part.suffix == ".part"
    assert part != target


def test_two_part_paths_for_one_target_differ(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"

    assert unique_part_path(target) != unique_part_path(target)


def test_choosing_a_part_path_creates_nothing(tmp_path: Path) -> None:
    unique_part_path(tmp_path / "manifest.json")

    assert list(tmp_path.iterdir()) == []


def test_replace_file_writes_the_bytes_and_leaves_no_part_file(tmp_path: Path) -> None:
    target = tmp_path / "corpus.src"

    replace_file(target, b"one\ntwo\n")

    assert target.read_bytes() == b"one\ntwo\n"
    assert [path.name for path in tmp_path.iterdir()] == ["corpus.src"]


def test_replace_file_replaces_an_earlier_file(tmp_path: Path) -> None:
    target = tmp_path / "corpus.src"
    target.write_bytes(b"old content that is longer\n")

    replace_file(target, b"new\n")

    assert target.read_bytes() == b"new\n"


def test_replace_file_writes_the_bytes_as_given(tmp_path: Path) -> None:
    target = tmp_path / "raw.bin"
    content = ("a\r\nb" + chr(0) + "c" + "э").encode()

    replace_file(target, content)

    assert target.read_bytes() == content


@pytest.mark.parametrize("mode", [0o640, 0o600, 0o444, 0o755])
def test_replace_file_gives_the_file_the_mode_it_is_asked_for(
    tmp_path: Path, mode: int
) -> None:
    target = tmp_path / "vocab.src"

    replace_file(target, b"a\n", mode=mode)

    assert target.stat().st_mode & 0o7777 == mode
    assert target.read_bytes() == b"a\n"
    assert [path.name for path in tmp_path.iterdir()] == ["vocab.src"]


def test_replace_file_without_a_mode_leaves_it_to_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        replace_file(tmp_path / "plain", b"x")
    finally:
        os.umask(previous)

    assert (tmp_path / "plain").stat().st_mode & 0o7777 == 0o640


def test_the_mode_is_set_before_the_rename_so_the_file_never_shows_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    real_replace = Path.replace

    def recording_replace(self: Path, destination: Any) -> Path:
        seen.append(self.stat().st_mode & 0o7777)
        return real_replace(self, destination)

    monkeypatch.setattr(Path, "replace", recording_replace)

    replace_file(tmp_path / "vocab.src", b"a\n", mode=0o600)

    assert seen == [0o600]


def test_replace_file_into_a_missing_directory_is_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replace_file(tmp_path / "missing" / "corpus.src", b"x")

    assert list(tmp_path.iterdir()) == []


def test_a_failed_rename_leaves_the_earlier_file_and_no_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "corpus.src"
    target.write_bytes(b"earlier\n")

    def failing_replace(self: Path, destination: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        replace_file(target, b"later\n")

    assert target.read_bytes() == b"earlier\n"
    assert [path.name for path in tmp_path.iterdir()] == ["corpus.src"]


def test_a_failed_write_leaves_no_part_file_and_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "corpus.src"

    def failing_write(self: Path, data: bytes) -> int:
        self.touch()
        raise OSError("no space")

    monkeypatch.setattr(Path, "write_bytes", failing_write)

    with pytest.raises(OSError, match="no space"):
        replace_file(target, b"later\n")

    assert list(tmp_path.iterdir()) == []


def test_lines_are_each_ended_by_one_lf_and_encoded_as_utf_8() -> None:
    assert lines_as_bytes(["one", "два", ""]) == "one\nдва\n\n".encode()


def test_lines_as_bytes_leaves_other_line_separators_inside_their_line() -> None:
    line = "a" + chr(0x2028) + "b\r" + chr(0x85) + "c"

    assert lines_as_bytes([line, "d"]) == (line + "\nd\n").encode("utf-8")


def test_lines_as_bytes_takes_any_iterable_of_text() -> None:
    assert lines_as_bytes(str(number) for number in range(3)) == b"0\n1\n2\n"
    assert lines_as_bytes(("x",)) == b"x\n"
