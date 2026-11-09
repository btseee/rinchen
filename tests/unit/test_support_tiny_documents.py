"""Tests for the toy documents of `tests.support.tiny_parallel` (D9): documents of 4 to
8 consecutive sentences, split by document, that the gate of the document-level model
trains and is scored on."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from rinchen.train.docify import read_corpus_rows
from tests.support.corpus_rows import write_rows
from tests.support.tiny_parallel import (
    LANGUAGES,
    DocRow,
    ToyDocuments,
    build_toy,
    build_toy_documents,
)

PAIRS = (("eng_Latn", "khk_Cyrl"), ("rus_Cyrl", "khk_Cyrl"))


@pytest.fixture(scope="module")
def documents() -> ToyDocuments:
    return build_toy_documents()


def _splits(documents: ToyDocuments) -> dict[str, tuple[DocRow, ...]]:
    return {"train": documents.train, "dev": documents.dev, "test": documents.test}


def _documents_of(rows: tuple[DocRow, ...]) -> dict[tuple[str, str, str], list[DocRow]]:
    grouped: dict[tuple[str, str, str], list[DocRow]] = {}
    for row in rows:
        key = (row["src_lang"], row["tgt_lang"], row["doc_id"])
        grouped.setdefault(key, []).append(row)
    return grouped


def test_the_sizes_are_the_ones_documented(documents: ToyDocuments) -> None:
    # Rows are items times two directions; documents are counted once per language pair.
    assert [
        (len(rows), len({row["doc_id"] for row in rows}))
        for rows in (documents.train, documents.dev, documents.test)
    ] == [(3600, 302), (400, 33), (400, 33)]


def test_the_same_seed_gives_the_same_documents_and_another_seed_others(
    documents: ToyDocuments,
) -> None:
    other = build_toy_documents(seed=8)
    lengths = [len(rows) for rows in _documents_of(documents.train).values()]

    assert build_toy_documents() == documents
    assert other != documents
    # Not only other sentences: the documents are cut at other places.
    assert [len(rows) for rows in _documents_of(other.train).values()] != lengths


def test_every_document_has_four_to_eight_consecutive_sentences_from_zero(
    documents: ToyDocuments,
) -> None:
    for rows in _splits(documents).values():
        for key, sentences in _documents_of(rows).items():
            assert 4 <= len(sentences) <= 8, key
            assert [row["seg_idx"] for row in sentences] == list(range(len(sentences)))


def test_document_ids_are_numbered_and_each_is_one_pair_in_both_directions(
    documents: ToyDocuments,
) -> None:
    pattern = re.compile(r"doc-[0-9]+")
    for rows in _splits(documents).values():
        directions: dict[str, set[tuple[str, str]]] = {}
        for row in rows:
            assert pattern.fullmatch(row["doc_id"])
            directions.setdefault(row["doc_id"], set()).add(
                (row["src_lang"], row["tgt_lang"])
            )
        for doc_id, seen in directions.items():
            assert len(seen) == 2, doc_id
            (first, second) = sorted(seen)
            assert first == second[::-1], doc_id
            assert first in PAIRS or first[::-1] in PAIRS


def test_no_document_id_is_in_two_splits(documents: ToyDocuments) -> None:
    ids = {
        name: {row["doc_id"] for row in rows}
        for name, rows in _splits(documents).items()
    }

    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]
    assert sum(map(len, ids.values())) == 302 + 33 + 33


def test_no_sentence_of_any_language_is_in_two_splits(documents: ToyDocuments) -> None:
    sentences: dict[str, dict[str, set[str]]] = {}
    for name, rows in _splits(documents).items():
        by_language: dict[str, set[str]] = {language: set() for language in LANGUAGES}
        for row in rows:
            by_language[row["src_lang"]].add(row["src"])
            by_language[row["tgt_lang"]].add(row["tgt"])
        sentences[name] = by_language

    for language in LANGUAGES:
        assert not sentences["train"][language] & sentences["dev"][language]
        assert not sentences["train"][language] & sentences["test"][language]
        assert not sentences["dev"][language] & sentences["test"][language]


def test_the_two_directions_of_a_document_are_mirror_images(
    documents: ToyDocuments,
) -> None:
    for rows in _splits(documents).values():
        grouped = _documents_of(rows)
        for (source, target, doc_id), sentences in grouped.items():
            mirror = grouped[(target, source, doc_id)]
            assert [(r["tgt"], r["src"]) for r in sentences] == [
                (r["src"], r["tgt"]) for r in mirror
            ]


def test_the_toy_is_three_way_parallel_in_the_held_out_documents(
    documents: ToyDocuments,
) -> None:
    # Dev and test hold one set of items for both language pairs: the Mongolian
    # sentences of the English documents are those of the Russian documents. The two
    # pairs of the train split have content of their own.
    def mongolian(rows: tuple[DocRow, ...], language: str) -> set[str]:
        return {
            row["tgt"]
            for row in rows
            if row["src_lang"] == language and row["tgt_lang"] == "khk_Cyrl"
        }

    for rows in (documents.dev, documents.test):
        assert mongolian(rows, "eng_Latn") == mongolian(rows, "rus_Cyrl")
    assert not mongolian(documents.train, "eng_Latn") & mongolian(
        documents.train, "rus_Cyrl"
    )


def test_the_documents_are_made_of_the_toy_items_of_their_split(
    documents: ToyDocuments,
) -> None:
    toy = build_toy()
    for name, rows in _splits(documents).items():
        items = {
            (row["src_lang"], row["tgt_lang"], row["src"], row["tgt"])
            for row in getattr(toy, name)
        }
        assert {
            (row["src_lang"], row["tgt_lang"], row["src"], row["tgt"]) for row in rows
        } == items


def test_every_row_is_human_text_of_its_split_with_a_corpus_name(
    documents: ToyDocuments,
) -> None:
    for name, rows in _splits(documents).items():
        assert {row["split"] for row in rows} == {name}
        assert {row["origin"] for row in rows} == {"human"}
    assert Counter(row["corpus"] for row in documents.train) == {
        "toy-en-mn": 2400,
        "toy-ru-mn": 1200,
    }


def test_the_rows_make_a_corpus_file_that_docify_reads(
    documents: ToyDocuments, tmp_path: Path
) -> None:
    path = write_rows(tmp_path, "docs", [*documents.train, *documents.dev])

    rows = read_corpus_rows([path])

    assert rows.height == 4000
    assert rows["doc_id"].null_count() == 0
    assert rows["seg_idx"].dtype == pl.Int32
    assert rows["seg_idx"].min() == 0
