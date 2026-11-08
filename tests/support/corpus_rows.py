"""Corpus rows, corpus files and the training rows read from them, for the tests of
`rinchen.train.mixture` and the modules that take its rows (`corpus_files`,
`eole_data`, `validation_corpus`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.train.mixture import TrainingRows, load_training_rows
from tests.support.tiny_parallel import build_toy, write_corpus_parquet

EN_MN = "eng_Latn-khk_Cyrl"
MN_EN = "khk_Cyrl-eng_Latn"
RU_MN = "rus_Cyrl-khk_Cyrl"
MN_RU = "khk_Cyrl-rus_Cyrl"


def corpus_row(
    src: str = "Hello",
    tgt: str = "Сайн",
    *,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "khk_Cyrl",
    origin: str = "human",
    split: str = "train",
) -> dict[str, str]:
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src": src,
        "tgt": tgt,
        "origin": origin,
        "corpus": "test",
        "split": split,
    }


def numbered_rows(count: int, **kwargs: str) -> list[dict[str, str]]:
    """`count` rows with the sources `s0`, `s1`, ... (`bt0`, `web0` ... when an
    origin other than human is given, so no two origins share a row id)."""
    origin = kwargs.get("origin", "human")
    stem = "s" if origin == "human" else origin
    return [
        corpus_row(f"{stem}{index}", f"t{index}", **kwargs) for index in range(count)
    ]


def write_rows(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    ids: Sequence[str] | None = None,
    filter_reasons: Sequence[str | None] | None = None,
) -> Path:
    frame = corpus_frame(rows)
    if ids is not None:
        frame = frame.with_columns(pl.Series("id", list(ids), dtype=pl.String))
    if filter_reasons is not None:
        frame = frame.with_columns(
            pl.Series("filter_reason", list(filter_reasons), dtype=pl.String)
        )
    return write_frame(directory, name, frame)


def write_frame(directory: Path, name: str, frame: pl.DataFrame) -> Path:
    path = directory / f"{name}.parquet"
    write_corpus(frame, path)
    return path


def training_from_rows(
    directory: Path, rows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> TrainingRows:
    directory.mkdir(parents=True, exist_ok=True)
    return load_training_rows([write_rows(directory, "rows", rows, **kwargs)])


def file_lines(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8")
    assert text.endswith("\n")
    return text.split("\n")[:-1]


def toy_training_rows(tmp_path: Path) -> TrainingRows:
    toy = build_toy()
    path = write_corpus_parquet(
        [*toy.train, *toy.bt, *toy.dev], tmp_path / "toy.parquet"
    )
    return load_training_rows([path])
