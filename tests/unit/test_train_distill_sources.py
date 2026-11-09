"""The sources of a knowledge-distillation run (D10): which rows of the corpus files are
sources (only the authentic training rows), one source per direction and text with all its
authentic targets as references, their ids, their order and the refusals.

The corpus files are the planted ones of `tests.support.planted_kd`; the expected ids are
literals (`make_row_id(src_lang, tgt_lang, text, "")` computed once)."""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from rinchen.errors import ConfigError
from rinchen.train.distill_sources import KdSources, kd_sources
from rinchen.train.mixture import load_training_rows
from tests.support.corpus_rows import (
    EN_MN,
    MN_EN,
    RU_MN,
    corpus_row,
    training_from_rows,
)
from tests.support.planted_kd import (
    PLANTED_REFERENCES,
    SOURCE_IDS,
    planted_training_rows,
)


def _by_direction(sources):
    return {source.direction: source for source in sources}


def _texts(source: KdSources) -> list[str]:
    return source.frame["text"].to_list()


# -- what a source is


def test_only_the_authentic_train_rows_without_a_filter_reason_are_sources(tmp_path):
    planted = planted_training_rows(tmp_path)

    sources = _by_direction(kd_sources(planted.training))

    assert sorted(_texts(sources[EN_MN])) == [
        "Good morning",
        "Hello",
        "Thank you",
        "Water",
    ]
    assert sorted(_texts(sources[RU_MN])) == ["Книга", "Привет", "Спасибо"]


@pytest.mark.parametrize(
    "excluded",
    [
        "Dev only",
        "Test only",
        "Holdout only",
        "Back translated",
        "Forward translated",
        "Synthetic",
        "Filtered only",
    ],
)
def test_a_text_that_only_an_excluded_row_has_is_not_a_source(tmp_path, excluded):
    sources = kd_sources(planted_training_rows(tmp_path).training)

    assert all(excluded not in _texts(source) for source in sources)


def test_the_sources_are_the_directions_that_have_authentic_rows_sorted(tmp_path):
    sources = kd_sources(planted_training_rows(tmp_path).training)

    assert [source.direction for source in sources] == [EN_MN, RU_MN]
    assert [(source.src_lang, source.tgt_lang) for source in sources] == [
        ("eng_Latn", "khk_Cyrl"),
        ("rus_Cyrl", "khk_Cyrl"),
    ]


def test_a_text_the_corpus_repeats_is_one_source_and_all_its_targets_are_references(
    tmp_path,
):
    sources = _by_direction(kd_sources(planted_training_rows(tmp_path).training))

    english = sources[EN_MN]
    assert english.frame.height == 4
    assert english.references[SOURCE_IDS["Thank you"]] == frozenset(
        {"Баярлалаа", "Баярлалаа танд"}
    )


def test_a_dev_a_back_translated_and_a_filtered_row_of_a_source_are_no_reference(
    tmp_path,
):
    english = _by_direction(kd_sources(planted_training_rows(tmp_path).training))[EN_MN]

    # The dev row says "Усан", the filtered row "Шүүсэн ус", the bt and ft rows of Hello
    # say "Мэнд" and "Мэндчилгээ".
    assert english.references[SOURCE_IDS["Water"]] == frozenset({"Ус"})
    assert english.references[SOURCE_IDS["Hello"]] == frozenset({"Сайн байна уу"})


def test_the_references_hold_exactly_the_planted_targets_of_every_source(tmp_path):
    sources = _by_direction(kd_sources(planted_training_rows(tmp_path).training))

    for direction, expected in PLANTED_REFERENCES.items():
        assert sources[direction].references == {
            SOURCE_IDS[text]: frozenset(targets) for text, targets in expected.items()
        }


# -- the ids, the order and the frame


def test_the_ids_are_the_row_id_of_the_source_with_an_empty_target(tmp_path):
    sources = _by_direction(kd_sources(planted_training_rows(tmp_path).training))

    for source in sources.values():
        assert {
            text: row_id
            for row_id, text in zip(source.frame["id"], source.frame["text"])
        } == {text: SOURCE_IDS[text] for text in _texts(source)}


def test_the_frame_is_sorted_by_id(tmp_path):
    for source in kd_sources(planted_training_rows(tmp_path).training):
        ids = source.frame["id"].to_list()
        assert ids == sorted(ids)
    english = _by_direction(kd_sources(planted_training_rows(tmp_path).training))[EN_MN]
    assert english.frame["id"].to_list() == [
        "29bc7ec8efab9ec2",
        "53855621fbc23c82",
        "b7a3e2e9e81912de",
        "ce3c60bd22eb00ce",
    ]


def test_the_frame_has_exactly_the_columns_id_lang_and_text(tmp_path):
    for source in kd_sources(planted_training_rows(tmp_path).training):
        assert source.frame.columns == ["id", "lang", "text"]
        assert source.frame.schema == {
            "id": pl.String,
            "lang": pl.String,
            "text": pl.String,
        }
        assert source.frame["lang"].unique().to_list() == [source.src_lang]


def test_shuffled_rows_and_files_in_another_order_give_the_same_sources(tmp_path):
    straight = planted_training_rows(tmp_path / "straight")
    shuffled = planted_training_rows(tmp_path / "shuffled", shuffle_seed=11)
    reversed_files = load_training_rows(list(reversed(shuffled.paths)))

    expected = kd_sources(straight.training)
    for rows in (shuffled.training, reversed_files):
        found = kd_sources(rows)
        assert [source.direction for source in found] == [
            source.direction for source in expected
        ]
        for got, want in zip(found, expected, strict=True):
            assert got.frame.equals(want.frame)
            assert got.references == want.references


# -- the directions


def test_a_direction_list_restricts_the_sources(tmp_path):
    rows = planted_training_rows(tmp_path).training

    (only,) = kd_sources(rows, [RU_MN])

    assert only.direction == RU_MN
    assert sorted(_texts(only)) == ["Книга", "Привет", "Спасибо"]


def test_requested_directions_come_out_sorted_and_once(tmp_path):
    rows = planted_training_rows(tmp_path).training

    found = kd_sources(rows, [RU_MN, EN_MN, RU_MN])

    assert [source.direction for source in found] == [EN_MN, RU_MN]


def test_an_unknown_direction_is_refused_naming_it_and_the_available_ones(tmp_path):
    rows = planted_training_rows(tmp_path).training

    with pytest.raises(ConfigError) as caught:
        kd_sources(rows, [EN_MN, "xxx_Latn-khk_Cyrl"])

    message = str(caught.value)
    assert "xxx_Latn-khk_Cyrl" in message
    assert EN_MN in message
    assert RU_MN in message


def test_a_direction_with_only_back_translated_rows_has_no_sources(tmp_path):
    rows = training_from_rows(
        tmp_path,
        [
            corpus_row("Hello", "Сайн"),
            corpus_row(
                "Сайн", "Hello", src_lang="khk_Cyrl", tgt_lang="eng_Latn", origin="bt"
            ),
        ],
    )

    with pytest.raises(ConfigError) as caught:
        kd_sources(rows, [MN_EN])

    assert MN_EN in str(caught.value)
    assert EN_MN in str(caught.value)
    assert [source.direction for source in kd_sources(rows)] == [EN_MN]


def test_no_direction_at_all_is_refused(tmp_path):
    rows = planted_training_rows(tmp_path).training

    with pytest.raises(ConfigError, match="no direction"):
        kd_sources(rows, [])


# -- the invariants of KdSources itself


def _valid_sources() -> KdSources:
    return KdSources(
        direction=EN_MN,
        src_lang="eng_Latn",
        tgt_lang="khk_Cyrl",
        frame=pl.DataFrame(
            {"id": ["a"], "lang": ["eng_Latn"], "text": ["Hello"]},
            schema={"id": pl.String, "lang": pl.String, "text": pl.String},
        ),
        references={"a": frozenset({"Сайн"})},
    )


def test_a_direction_that_is_not_the_language_pair_is_refused():
    with pytest.raises(ValueError, match="direction"):
        dataclasses.replace(_valid_sources(), direction=RU_MN)


def test_a_frame_with_other_columns_than_id_lang_and_text_is_refused():
    frame = _valid_sources().frame.with_columns(extra=pl.lit(1))

    with pytest.raises(ValueError, match="columns"):
        dataclasses.replace(_valid_sources(), frame=frame)


def test_a_frame_with_a_row_in_another_language_is_refused():
    frame = _valid_sources().frame.with_columns(lang=pl.lit("rus_Cyrl"))

    with pytest.raises(ValueError, match="lang"):
        dataclasses.replace(_valid_sources(), frame=frame)
