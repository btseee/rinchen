"""The unique temporary name a writer uses before it renames its file into
place."""

from __future__ import annotations

from pathlib import Path

from rinchen.utils.part_file import unique_part_path


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
