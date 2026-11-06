# tests/unit/test_filter_reasons.py
from __future__ import annotations

import polars as pl
import pytest

from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.filter.reasons import FilterReason, flag, with_scores


def _corpus(count: int = 2) -> pl.LazyFrame:
    rows = [
        {
            "src_lang": "khk_Cyrl",
            "tgt_lang": "eng_Latn",
            "src": f"source {index}",
            "tgt": f"target {index}",
            "origin": "human",
            "corpus": "unit",
            "split": "train",
        }
        for index in range(count)
    ]
    return corpus_frame(rows).lazy()


def test_reason_values_are_exactly_the_ten_strings():
    assert [reason.value for reason in FilterReason] == [
        "script_src",
        "script_tgt",
        "lid_src",
        "lid_tgt",
        "len_ratio_char",
        "len_ratio_token",
        "numbers",
        "duplicate",
        "labse",
        "qe",
    ]


def test_reasons_are_plain_strings():
    assert FilterReason.LID_SRC == "lid_src"
    assert f"{FilterReason.QE}" == "qe"


def test_flag_sets_the_reason_only_where_failed_and_not_yet_flagged():
    frame = pl.DataFrame(
        {
            "failed": [True, True, False, None],
            "filter_reason": [None, "x", None, None],
        }
    ).lazy()

    result = flag(frame, pl.col("failed"), FilterReason.LABSE).collect()

    assert result["filter_reason"].to_list() == ["labse", "x", None, None]
    assert result["failed"].to_list() == [True, True, False, None]


def test_flag_keeps_the_corpus_schema():
    frame = _corpus(3).with_columns(
        pl.col("src").str.ends_with("1").alias("hit"),
    )

    result = flag(frame, pl.col("hit"), FilterReason.NUMBERS).collect()

    assert result["filter_reason"].to_list() == [None, "numbers", None]
    assert result.drop("hit").schema == CORPUS_SCHEMA


def test_flag_is_first_failure_wins_across_calls():
    frame = _corpus(1)

    result = flag(
        flag(frame, pl.lit(True), FilterReason.SCRIPT_SRC),
        pl.lit(True),
        FilterReason.DUPLICATE,
    ).collect()

    assert result["filter_reason"].to_list() == ["script_src"]


def test_with_scores_fills_a_fresh_frame_whose_scores_are_null():
    result = with_scores(_corpus(2), lid_src=pl.lit(0.75)).collect()

    assert result.schema == CORPUS_SCHEMA
    assert result["scores"].to_list() == [
        {**_all_none(), "lid_src": 0.75},
        {**_all_none(), "lid_src": 0.75},
    ]


def test_with_scores_replaces_one_field_and_keeps_the_others():
    first = with_scores(_corpus(2), lid_src=pl.lit(0.25), num_ok=pl.lit(True))

    result = with_scores(first, lid_src=pl.lit(0.5), labse=pl.lit(0.9)).collect()

    assert result.schema == CORPUS_SCHEMA
    assert result["scores"].to_list()[0] == {
        **_all_none(),
        "lid_src": 0.5,
        "num_ok": True,
        "labse": pytest.approx(0.9),
    }


def test_with_scores_reads_other_columns_per_row():
    result = with_scores(
        _corpus(2),
        len_ratio=pl.col("tgt").str.len_chars() / pl.col("src").str.len_chars(),
    ).collect()

    assert result.schema == CORPUS_SCHEMA
    assert result["scores"].struct.field("len_ratio").to_list() == [1.0, 1.0]


def test_with_scores_casts_to_the_declared_field_type():
    result = with_scores(_corpus(1), lid_tgt=pl.lit(0.5, dtype=pl.Float64))

    assert result.collect_schema() == CORPUS_SCHEMA


def test_with_scores_with_no_fields_changes_nothing():
    frame = _corpus(2)

    assert with_scores(frame).collect().equals(frame.collect())


def test_with_scores_rejects_a_name_that_is_not_a_score_field():
    with pytest.raises(ValueError, match="unknown"):
        with_scores(_corpus(1), unknown=pl.lit(1.0))


def test_with_scores_names_every_unknown_field_and_lists_the_valid_ones():
    with pytest.raises(ValueError) as excinfo:
        with_scores(_corpus(1), lid_src=pl.lit(1.0), nope=pl.lit(1.0), bad=pl.lit(1.0))

    message = str(excinfo.value)
    assert "bad" in message
    assert "nope" in message
    assert "metricx_qe" in message


def _all_none() -> dict[str, object]:
    return {
        "lid_src": None,
        "lid_tgt": None,
        "script_src": None,
        "script_tgt": None,
        "len_ratio": None,
        "num_ok": None,
        "labse": None,
        "metricx_qe": None,
        "cometkiwi": None,
    }
