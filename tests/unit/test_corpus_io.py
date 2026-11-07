"""Unit tests for `rinchen.corpus.readers` and `rinchen.corpus.writers`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import DataError

CREATED = datetime(2026, 11, 6, 12, 30, 15, 123456, tzinfo=UTC)


def _frame() -> pl.DataFrame:
    return corpus_frame(
        [
            {
                "src_lang": "eng_Latn",
                "tgt_lang": "khk_Cyrl",
                "src": "Good morning.",
                "tgt": "Өдрийн мэнд.",
                "origin": "human",
                "corpus": "tiny",
                "split": "train",
                "doc_id": "d1",
                "seg_idx": 0,
            },
            {
                "src_lang": "eng_Latn",
                "tgt_lang": "khk_Cyrl",
                "src": "Thanks.",
                "tgt": "Баярлалаа.",
                "origin": "web",
                "corpus": "tiny",
                "split": "dev",
            },
        ],
        created=CREATED,
    )


def test_write_then_scan_round_trips_and_is_zstd(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    frame = _frame()

    write_corpus(frame, path)

    assert scan_corpus(path).collect().equals(frame)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_write_creates_parent_dirs_and_returns_row_count(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "corpus.parquet"

    assert write_corpus(_frame(), path) == 2
    assert path.is_file()


def test_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"

    write_corpus(_frame(), path)

    assert [entry.name for entry in tmp_path.iterdir()] == ["corpus.parquet"]


def test_write_refuses_wrong_schema_and_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "out" / "corpus.parquet"
    wrong = _frame().drop("src")

    with pytest.raises(DataError, match=r"missing columns: src\b"):
        write_corpus(wrong, path)

    assert not path.exists()
    assert not path.parent.exists()


def test_write_that_fails_while_computing_keeps_the_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.parquet"
    write_corpus(_frame(), path)

    def explode(series: pl.Series) -> pl.Series:
        raise ValueError("planted failure")

    failing = (
        _frame()
        .lazy()
        .with_columns(src=pl.col("src").map_batches(explode, return_dtype=pl.String))
    )
    with pytest.raises(ValueError, match="planted failure"):
        write_corpus(failing, path)

    assert [entry.name for entry in tmp_path.iterdir()] == ["corpus.parquet"]
    assert scan_corpus(path).collect().equals(_frame())


def test_failed_rename_removes_the_part_file_and_keeps_the_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.parquet"
    write_corpus(_frame(), path)
    part_existed_at_rename: list[bool] = []

    def failing_replace(self: Path, target: Path) -> Path:
        part_existed_at_rename.append(self.is_file())
        raise OSError("planted rename failure")

    monkeypatch.setattr(Path, "replace", failing_replace)
    replacement = _frame().with_columns(src=pl.lit("changed"))

    with pytest.raises(OSError, match="planted rename failure"):
        write_corpus(replacement, path)

    assert part_existed_at_rename == [True]
    assert [entry.name for entry in tmp_path.iterdir()] == ["corpus.parquet"]
    assert scan_corpus(path).collect().equals(_frame())


def test_write_orders_columns_as_the_contract(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    shuffled = _frame().select(list(reversed(CORPUS_SCHEMA.names())))

    write_corpus(shuffled, path)

    assert scan_corpus(path).collect_schema().names() == CORPUS_SCHEMA.names()


def test_scan_missing_file_is_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing.parquet"):
        scan_corpus(tmp_path / "missing.parquet")


def test_scan_wrong_schema_is_data_error(tmp_path: Path) -> None:
    path = tmp_path / "wrong.parquet"
    _frame().drop("src").write_parquet(path)

    with pytest.raises(DataError, match=r"missing columns: src\b"):
        scan_corpus(path)


def test_scan_unreadable_file_is_data_error(tmp_path: Path) -> None:
    path = tmp_path / "junk.parquet"
    path.write_bytes(b"not a parquet file")

    with pytest.raises(DataError, match="junk.parquet"):
        scan_corpus(path)


def test_write_accepts_a_lazyframe(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"

    assert write_corpus(_frame().lazy(), path) == 2
    assert scan_corpus(path).collect().equals(_frame())


def test_empty_frame_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    empty = pl.DataFrame(schema=CORPUS_SCHEMA)

    assert write_corpus(empty, path) == 0
    result = scan_corpus(path).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_a_writer_never_touches_the_temporary_file_of_another_writer(
    tmp_path: Path,
) -> None:
    # What a concurrent run into the same directory has in flight, under the
    # name an earlier version of the writer used for itself.
    other_writers_file = tmp_path / "corpus.parquet.part"
    other_writers_file.write_bytes(b"in flight")
    path = tmp_path / "corpus.parquet"

    write_corpus(_frame(), path)

    assert other_writers_file.read_bytes() == b"in flight"
    assert scan_corpus(path).collect().equals(_frame())
    assert sorted(entry.name for entry in tmp_path.glob("*.part")) == [
        "corpus.parquet.part"
    ]
