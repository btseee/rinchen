"""Tests for what the saved `config.json` of a checkpoint of ours says about the DIRECTIONS the
model was trained for and about whether it decodes `<doc>` windows (D11, T1 and T10).

The `data` section the launcher leaves names each corpus `<src>-<tgt>.<group>` (`auth`, `bt`,
`kd`): the directions of the model are the pairs of codes of the registry in those names, and
a document-level run writes `<doc>` as a whole word of the `src_prefix` of its corpora. The
vocabulary says neither (it holds every reserved symbol), so this file is the only evidence an
export has. The fabricated sections come from `tests.support.step_dirs.launcher_data`.
"""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.train.distill_saved_config import (
    saved_directions,
    saved_document_level,
)
from tests.support.step_dirs import launcher_data

TOY_CORPORA = (
    ("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),
    ("khk_Cyrl-eng_Latn.auth", "<2eng_Latn>", None),
    ("khk_Cyrl-rus_Cyrl.auth", "<2rus_Cyrl>", None),
    ("rus_Cyrl-khk_Cyrl.auth", "<2khk_Cyrl>", None),
)
TOY_DIRECTIONS = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)


def config_of(data: Any) -> dict[str, Any]:
    return {"training": {"model_path": "/ckpt"}, "data": data}


def entry(src_prefix: Any) -> dict[str, Any]:
    return {"path_src": "/a.src", "path_tgt": "/a.tgt", "src_prefix": src_prefix,
            "tgt_prefix": ""}  # fmt: skip


# ---- saved_directions ----------------------------------------------------------------------


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_toy_teachers_corpora_give_its_four_directions(placement: str) -> None:
    config = config_of(launcher_data(placement, TOY_CORPORA))

    assert saved_directions(config) == TOY_DIRECTIONS


def test_back_translated_and_distilled_corpora_of_the_same_directions_add_none() -> (
    None
):
    corpora = (
        *TOY_CORPORA,
        ("eng_Latn-khk_Cyrl.bt", "<2khk_Cyrl>", "<bt>"),
        ("khk_Cyrl-eng_Latn.bt", "<2eng_Latn>", "<bt>"),
        ("eng_Latn-khk_Cyrl.kd", "<2khk_Cyrl>", "<ft>"),
    )

    assert saved_directions(config_of(launcher_data("src", corpora))) == TOY_DIRECTIONS


def test_a_corpus_of_one_group_only_is_enough_to_name_its_direction() -> None:
    corpora = (("eng_Latn-khk_Cyrl.kd", "<2khk_Cyrl>", "<ft>"),)

    assert saved_directions(config_of(launcher_data("src", corpora))) == (
        "eng_Latn-khk_Cyrl",
    )


def test_the_directions_are_sorted_whatever_the_order_of_the_corpora() -> None:
    corpora = tuple(reversed(TOY_CORPORA))

    assert saved_directions(config_of(launcher_data("src", corpora))) == TOY_DIRECTIONS


def test_the_valid_entry_names_no_direction() -> None:
    data = launcher_data("src", (("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),))
    assert "valid" in data

    assert saved_directions(config_of(data)) == ("eng_Latn-khk_Cyrl",)


@pytest.mark.parametrize(
    "name",
    [
        "eng_Latn-xxx_Latn.auth",
        "xxx_Latn-khk_Cyrl.auth",
        "abc-def.auth",
        "eng_Latn-eng_Latn.auth",
    ],
    ids=[
        "unknown target",
        "unknown source",
        "no registry code",
        "a language to itself",
    ],
)
def test_a_corpus_with_a_code_outside_the_registry_or_a_pair_of_one_language_is_ignored(
    name: str,
) -> None:
    corpora = (
        ("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),
        (name, "<2khk_Cyrl>", None),
    )

    assert saved_directions(config_of(launcher_data("src", corpora))) == (
        "eng_Latn-khk_Cyrl",
    )


@pytest.mark.parametrize(
    "name",
    [
        "eng_Latn-khk_Cyrl",
        "eng_Latn-khk_Cyrl.",
        "eng_Latn-khk_Cyrl.AUTH",
        "eng_Latn-khk_Cyrl.auth.extra",
        "eng_Latn-khk_Cyrl.auth2",
        "eng_Latn-khk_Cyrl-rus_Cyrl.auth",
        " eng_Latn-khk_Cyrl.auth",
        "train",
    ],
)
def test_a_name_that_is_not_a_direction_and_a_group_is_ignored(name: str) -> None:
    corpora = (
        ("khk_Cyrl-rus_Cyrl.auth", "<2rus_Cyrl>", None),
        (name, "<2khk_Cyrl>", None),
    )

    assert saved_directions(config_of(launcher_data("src", corpora))) == (
        "khk_Cyrl-rus_Cyrl",
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"training": {}},
        {"data": {}},
        {"data": {"valid": {"path_src": "/v.src", "path_tgt": "/v.tgt"}}},
        {"data": {"train": entry("<2khk_Cyrl>")}},
        {"data": {"eng_Latn-xxx_Latn.auth": entry("<2khk_Cyrl>")}},
    ],
    ids=[
        "empty config",
        "no data section",
        "empty data section",
        "valid only",
        "no direction in the name",
        "no registry code",
    ],
)
def test_a_config_that_names_no_direction_gives_none(config: dict[str, Any]) -> None:
    assert saved_directions(config) is None


@pytest.mark.parametrize("data", [[], "src", 7, ["eng_Latn-khk_Cyrl.auth"]])
def test_a_data_section_that_is_not_an_object_is_a_data_error(data: Any) -> None:
    with pytest.raises(DataError, match="'data' is not an object"):
        saved_directions(config_of(data))


# ---- saved_document_level --------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "<2khk_Cyrl> <doc>",
        "<doc> <2khk_Cyrl>",
        "<doc>",
        "<2khk_Cyrl> <bt> <doc>",
        "  <2khk_Cyrl>\t<doc>  ",
    ],
)
def test_the_word_doc_in_a_source_prefix_says_the_model_decodes_windows(
    prefix: str,
) -> None:
    config = config_of({"corpus": entry(prefix)})

    assert saved_document_level(config) is True


@pytest.mark.parametrize(
    "prefix",
    [
        "<doc><sep>",
        "x<doc>",
        "<doc>x",
        "<docs>",
        "<doc",
        "doc",
        "<2khk_Cyrl> <document>",
        "<DOC>",
        "",
    ],
)
def test_doc_inside_another_word_or_spelled_otherwise_does_not_count(
    prefix: str,
) -> None:
    config = config_of({"corpus": entry(prefix)})

    assert saved_document_level(config) is False


def test_corpora_without_doc_say_the_model_is_not_a_document_model() -> None:
    config = config_of(launcher_data("src"))

    assert saved_document_level(config) is False


def test_one_window_corpus_among_sentence_corpora_is_enough() -> None:
    data = launcher_data("src")
    data["eng_Latn-khk_Cyrl.auth"]["src_prefix"] = "<2khk_Cyrl> <doc>"

    assert saved_document_level(config_of(data)) is True


def test_only_the_source_prefix_counts() -> None:
    config = config_of({"corpus": {"src_prefix": "<2khk_Cyrl>", "tgt_prefix": "<doc>"}})

    assert saved_document_level(config) is False


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"training": {}},
        {"data": {}},
        {"data": {"valid": {"path_src": "/v.src", "path_tgt": "/v.tgt"}}},
        {"data": {"corpus": {"path_src": "/c.src", "path_tgt": "/c.tgt"}}},
    ],
    ids=[
        "empty config",
        "no data section",
        "empty data section",
        "valid only",
        "no source prefix",
    ],
)
def test_a_config_with_no_corpus_to_read_gives_none(config: dict[str, Any]) -> None:
    assert saved_document_level(config) is None


@pytest.mark.parametrize("data", [[], "src", 7])
def test_a_data_section_that_is_not_an_object_is_a_data_error_for_documents_too(
    data: Any,
) -> None:
    with pytest.raises(DataError, match="'data' is not an object"):
        saved_document_level(config_of(data))


@pytest.mark.parametrize("prefix", [7, ["<doc>"], {"a": 1}, True])
def test_a_source_prefix_that_is_not_text_is_a_data_error(prefix: Any) -> None:
    with pytest.raises(DataError, match=r"src_prefix is not text"):
        saved_document_level(config_of({"corpus": entry(prefix)}))


@pytest.mark.parametrize("corpus", ["<2khk_Cyrl> <doc>", 7, ["<doc>"], None])
def test_a_corpus_entry_that_is_not_an_object_is_a_data_error_for_documents(
    corpus: Any,
) -> None:
    with pytest.raises(DataError, match="is not an object"):
        saved_document_level(config_of({"corpus": corpus}))
