"""Unit tests for `rinchen.corpus.schema` (the corpus Parquet contract)."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import polars as pl
import pytest

from rinchen.corpus.schema import (
    CORPUS_SCHEMA,
    MONO_SCHEMA,
    ORIGINS,
    SCORES_DTYPE,
    SPLITS,
    corpus_frame,
    make_row_id,
    validate_corpus_schema,
)
from rinchen.errors import DataError

CREATED = datetime(2026, 11, 6, 12, 30, 15, 123456, tzinfo=UTC)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": "Good morning.",
        "tgt": "Өдрийн мэнд.",
        "origin": "human",
        "corpus": "tiny",
        "split": "train",
    }
    row.update(overrides)
    return row


def test_corpus_schema_matches_master_plan_contract() -> None:
    assert CORPUS_SCHEMA.names() == [
        "id",
        "src_lang",
        "tgt_lang",
        "src",
        "tgt",
        "origin",
        "corpus",
        "doc_id",
        "seg_idx",
        "split",
        "scores",
        "filter_reason",
        "created",
    ]
    assert CORPUS_SCHEMA["seg_idx"] == pl.Int32
    assert CORPUS_SCHEMA["origin"] == pl.Enum(ORIGINS)
    assert CORPUS_SCHEMA["split"] == pl.Enum(SPLITS)
    assert CORPUS_SCHEMA["scores"] == SCORES_DTYPE
    assert CORPUS_SCHEMA["created"] == pl.Datetime("us", "UTC")
    assert ORIGINS == (
        "human",
        "web",
        "mined",
        "bt",
        "ft",
        "synth_llm",
        "synth_nmt",
        "converted",
    )
    assert SPLITS == ("train", "dev", "test", "holdout")
    assert isinstance(SCORES_DTYPE, pl.Struct)
    assert [(field.name, field.dtype) for field in SCORES_DTYPE.fields] == [
        ("lid_src", pl.Float32),
        ("lid_tgt", pl.Float32),
        ("script_src", pl.Float32),
        ("script_tgt", pl.Float32),
        ("len_ratio", pl.Float32),
        ("num_ok", pl.Boolean),
        ("labse", pl.Float32),
        ("metricx_qe", pl.Float32),
        ("cometkiwi", pl.Float32),
    ]


def test_mono_schema_fields() -> None:
    assert MONO_SCHEMA == pl.Schema(
        {
            "id": pl.String,
            "lang": pl.String,
            "text": pl.String,
            "corpus": pl.String,
            "doc_id": pl.String,
            "seg_idx": pl.Int32,
            "url": pl.String,
            "scores": pl.Struct({"lid": pl.Float32, "script": pl.Float32}),
        }
    )


def test_make_row_id_is_16_hex_and_depends_on_all_four_fields() -> None:
    fields = ("eng_Latn", "khk_Cyrl", "hello", "сайн уу")
    row_id = make_row_id(*fields)

    assert re.fullmatch(r"[0-9a-f]{16}", row_id)
    assert make_row_id(*fields) == row_id
    for index in range(4):
        changed = list(fields)
        changed[index] += "x"
        assert make_row_id(*changed) != row_id


def test_make_row_id_is_sha256_of_tab_joined_fields_prefix() -> None:
    expected = hashlib.sha256(b"a\tb\tc\td").hexdigest()[:16]

    assert make_row_id("a", "b", "c", "d") == expected


def test_corpus_frame_fills_defaults() -> None:
    rows = [_row(), _row(src="Thanks.", tgt="Баярлалаа.", origin="web")]

    frame = corpus_frame(rows, created=CREATED)

    validate_corpus_schema(frame.schema, source="test")
    assert frame.schema == CORPUS_SCHEMA
    assert frame["id"].to_list() == [
        make_row_id(r["src_lang"], r["tgt_lang"], r["src"], r["tgt"]) for r in rows
    ]
    assert frame["origin"].to_list() == ["human", "web"]
    assert frame["doc_id"].null_count() == 2
    assert frame["seg_idx"].null_count() == 2
    assert frame["scores"].null_count() == 2
    assert frame["filter_reason"].null_count() == 2
    assert frame["created"].to_list() == [CREATED, CREATED]


def test_corpus_frame_keeps_optional_doc_id_and_seg_idx() -> None:
    frame = corpus_frame([_row(doc_id="d1", seg_idx=3), _row(tgt="x")])

    assert frame["doc_id"].to_list() == ["d1", None]
    assert frame["seg_idx"].to_list() == [3, None]


def test_corpus_frame_defaults_created_to_now_in_utc() -> None:
    before = datetime.now(UTC)
    frame = corpus_frame([_row()])
    after = datetime.now(UTC)

    (created,) = frame["created"].to_list()
    assert before <= created <= after


def test_corpus_frame_converts_created_to_utc() -> None:
    plus_nine = timezone(timedelta(hours=9))

    frame = corpus_frame([_row()], created=CREATED.astimezone(plus_nine))

    assert frame["created"].to_list() == [CREATED]


def test_corpus_frame_rejects_naive_created() -> None:
    with pytest.raises(DataError, match="timezone"):
        corpus_frame([_row()], created=datetime(2026, 11, 6, 12, 0, 0))


def test_corpus_frame_accepts_no_rows() -> None:
    frame = corpus_frame([], created=CREATED)

    assert frame.height == 0
    assert frame.schema == CORPUS_SCHEMA


def test_corpus_frame_rejects_unknown_origin() -> None:
    with pytest.raises(DataError, match="oracle"):
        corpus_frame([_row(origin="oracle")])


def test_corpus_frame_rejects_unknown_split() -> None:
    with pytest.raises(DataError, match="validation"):
        corpus_frame([_row(split="validation")])


@pytest.mark.parametrize(
    "key", ["src_lang", "tgt_lang", "src", "tgt", "origin", "corpus", "split"]
)
def test_corpus_frame_rejects_row_missing_required_key(key: str) -> None:
    row = _row()
    del row[key]

    with pytest.raises(DataError, match=key):
        corpus_frame([_row(), row])


def test_corpus_frame_rejects_unexpected_key() -> None:
    with pytest.raises(DataError, match="docid"):
        corpus_frame([_row(docid="d1")])


@pytest.mark.parametrize("seg_idx", ["3", 2**40])
def test_corpus_frame_rejects_unrepresentable_seg_idx(seg_idx: object) -> None:
    with pytest.raises(DataError, match="seg_idx"):
        corpus_frame([_row(seg_idx=seg_idx)])


def test_validate_corpus_schema_accepts_the_contract() -> None:
    validate_corpus_schema(CORPUS_SCHEMA, source="contract")


def test_validate_corpus_schema_reports_missing_extra_and_mistyped() -> None:
    broken = {name: dtype for name, dtype in CORPUS_SCHEMA.items() if name != "src"}
    broken["x"] = pl.String
    broken["seg_idx"] = pl.Int64

    with pytest.raises(DataError) as excinfo:
        validate_corpus_schema(pl.Schema(broken), source="data/clean/some.parquet")

    message = str(excinfo.value)
    assert "src" in message
    assert re.search(r"\bx\b", message)
    assert "seg_idx" in message
    assert "data/clean/some.parquet" in message
