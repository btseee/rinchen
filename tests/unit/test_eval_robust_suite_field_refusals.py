"""Refusals of the suite reader of D12 (spec T11) about what a field holds: texts (length, blank,
lone surrogate, the reserved symbols of the project), languages, the script of a row, the
`mong` section, the entity table and the entities of a row, the terms of a row. Each test plants
`MARKER` in the very value a refusal is about, so a message that echoed it would show it; the
messages name the section, the row index and id and the field. Every expectation is a literal."""

from __future__ import annotations

import json

import pytest

from rinchen.errors import DataError
from rinchen.eval.robust_suite import MAX_TEXT_CHARS, SECTIONS, Suite
from rinchen.infer.source_tokens import source_symbols
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from tests.support.robust_suite_documents import (
    MARKER,
    MONGOLIAN,
    Document,
    accepted,
    assert_names,
    at,
    changed,
    document,
    refused,
    repeating,
    row_of,
    set_row_field,
)

# ---- the texts ----------------------------------------------------------------------------

TEXT_FIELDS = [
    ("numbers", "source"),
    ("numbers", "reference"),
    ("placeholders", "source"),
    ("placeholders", "reference"),
    ("entities", "source"),
    ("entities", "reference"),
    ("terms", "source"),
    ("terms", "reference"),
    ("script", "source"),
    ("mong", "source"),
    ("mong", "reference"),
    ("mong", "source_mong"),
]


def with_text(section: str, field: str, text: object) -> bytes:
    return changed(set_row_field(section, field, text))


@pytest.mark.parametrize(("section", "field"), TEXT_FIELDS)
def test_a_text_that_is_too_long_is_refused(section: str, field: str) -> None:
    text = (
        MONGOLIAN + MARKER + "x" * (MAX_TEXT_CHARS - len(MARKER) - len(MONGOLIAN) + 1)
    )
    assert len(text) == MAX_TEXT_CHARS + 1

    message = refused(with_text(section, field, text))

    assert_names(message, *at(section, field), f"{MAX_TEXT_CHARS} characters")


@pytest.mark.parametrize(("section", "field"), TEXT_FIELDS)
@pytest.mark.parametrize("blank", ["", "   ", chr(0xA0) * 2, "\n\t"])
def test_a_blank_text_is_refused(section: str, field: str, blank: str) -> None:
    assert_names(
        refused(with_text(section, field, blank)), *at(section, field), "blank"
    )


@pytest.mark.parametrize(("section", "field"), TEXT_FIELDS)
@pytest.mark.parametrize("value", [5, [MARKER], {"a": MARKER}, True])
def test_a_text_that_is_not_a_string_is_refused(
    section: str, field: str, value: object
) -> None:
    assert_names(refused(with_text(section, field, value)), *at(section, field))


@pytest.mark.parametrize(("section", "field"), TEXT_FIELDS)
def test_a_lone_surrogate_is_refused(section: str, field: str) -> None:
    sentinel = "SURROGATE-SENTINEL"
    text = json.dumps(
        changed_document(section, field, MONGOLIAN + sentinel), ensure_ascii=False
    )
    data = text.replace(sentinel, MARKER + "\\ud800").encode("utf-8")

    assert_names(refused(data), *at(section, field), "surrogate")


def changed_document(section: str, field: str, value: object) -> Document:
    doc = document()
    row_of(doc, section)[field] = value
    return doc


def test_a_pair_of_escaped_surrogates_is_one_character_and_accepted() -> None:
    sentinel = "PAIR-SENTINEL"
    text = json.dumps(
        changed_document("numbers", "source", sentinel), ensure_ascii=False
    )
    data = text.replace(sentinel, "\\ud83d\\ude00 15").encode("utf-8")

    assert Suite.from_bytes(data).sections["numbers"][0].source == chr(0x1F600) + " 15"


RESERVED_EXAMPLES = [
    "<2eng_Latn>",
    "<2khk_Cyrl>",
    "<2mn_Mong>",
    "<t>",
    "</t>",
    "<r>",
    "</r>",
]


@pytest.mark.parametrize(("section", "field"), TEXT_FIELDS)
@pytest.mark.parametrize("symbol", RESERVED_EXAMPLES)
def test_a_reserved_symbol_of_the_project_is_refused_in_every_text(
    section: str, field: str, symbol: str
) -> None:
    text = MONGOLIAN + f" {MARKER} {symbol} 15"

    message = refused(with_text(section, field, text))

    assert_names(message, *at(section, field), "reserved symbol")
    assert symbol not in message


def test_every_reserved_symbol_but_the_numbered_placeholders_is_refused() -> None:
    symbols = source_symbols()
    assert len(symbols) == 18

    for symbol in symbols:
        text = f"15 {symbol} 15"
        with pytest.raises(DataError, match="reserved symbol"):
            Suite.from_bytes(with_text("numbers", "source", text))
    placeholders = " ".join(user_defined_symbols()[-PLACEHOLDER_COUNT:])
    assert accepted(set_row_field("numbers", "source", f"15 {placeholders}"))


@pytest.mark.parametrize("section", ["numbers", "placeholders", "mong"])
def test_a_reference_the_section_needs_cannot_be_null(section: str) -> None:
    message = refused(with_text(section, "reference", None))

    assert_names(message, *at(section, "reference"), "text")


# ---- the languages and the script ---------------------------------------------------------


@pytest.mark.parametrize(
    "section", ["numbers", "placeholders", "entities", "terms", "mong"]
)
def test_a_source_language_is_required_outside_the_script_section(section: str) -> None:
    message = refused(with_text(section, "src_lang", None))

    assert_names(message, *at(section, "src_lang"), "null")


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("field", ["src_lang", "tgt_lang"])
@pytest.mark.parametrize("value", [f"{MARKER}_Latn", 5, "", "eng_latn", ["eng_Latn"]])
def test_a_language_that_is_not_a_code_of_the_registry_is_refused(
    section: str, field: str, value: object
) -> None:
    message = refused(with_text(section, field, value))

    assert_names(message, *at(section, field), "registry")


@pytest.mark.parametrize("section", SECTIONS)
def test_a_target_language_is_required_in_every_section(section: str) -> None:
    assert_names(
        refused(with_text(section, "tgt_lang", None)), *at(section, "tgt_lang")
    )


@pytest.mark.parametrize(
    ("tgt_lang", "script"),
    [
        ("eng_Latn", "Mong"),
        ("rus_Cyrl", "Cyrl"),
        ("khk_Cyrl", "Latn"),
        ("khk_Cyrl", MARKER),
    ],
)
def test_a_script_not_allowed_for_the_target_is_refused(
    tgt_lang: str, script: str
) -> None:
    def change(doc: Document) -> None:
        row_of(doc, "script").update(tgt_lang=tgt_lang, script=script)

    assert_names(refused(changed(change)), *at("script", "script"))


@pytest.mark.parametrize("script", [5, ["Mong"], {"Mong": 1}, "mong", True])
def test_a_script_that_is_not_a_script_name_is_refused(script: object) -> None:
    assert_names(
        refused(with_text("script", "script", script)), *at("script", "script")
    )


# ---- the mong section ---------------------------------------------------------------------


@pytest.mark.parametrize("src_lang", ["eng_Latn", "mn_Mong", "rus_Cyrl"])
def test_a_mong_row_has_a_khalkha_source(src_lang: str) -> None:
    message = refused(with_text("mong", "src_lang", src_lang))

    assert_names(message, *at("mong", "src_lang"), "khk_Cyrl")


@pytest.mark.parametrize("tgt_lang", ["khk_Cyrl", "mn_Mong"])
def test_a_mong_row_translates_into_neither_khalkha_nor_traditional_mongolian(
    tgt_lang: str,
) -> None:
    message = refused(with_text("mong", "tgt_lang", tgt_lang))

    assert_names(message, *at("mong", "tgt_lang"))


@pytest.mark.parametrize(
    "text", ["Энэ бол ном.", "This is a book.", f"15 {chr(0x20AE)} 20", f"{MARKER} 12"]
)
def test_a_source_in_traditional_script_holds_a_letter_of_it(text: str) -> None:
    message = refused(with_text("mong", "source_mong", text))

    assert_names(message, *at("mong", "source_mong"), "Mongolian")


# ---- the entity table and the entities of a row ---------------------------------------


def test_a_table_that_is_not_an_object_is_refused() -> None:
    def change(doc: Document) -> None:
        doc["entities"] = [MARKER]

    assert_names(refused(changed(change)), "entity table", "object")


@pytest.mark.parametrize("table_id", ["Bat", "", "x" * 33, f"{MARKER}-1", "a b", "бат"])
def test_an_entity_table_id_is_one_to_32_lowercase_letters_digits_and_underscores(
    table_id: str,
) -> None:
    def change(doc: Document) -> None:
        doc["entities"] = {"bat": doc["entities"]["bat"], table_id: {"eng_Latn": ["x"]}}

    assert_names(refused(changed(change)), "entity table", "index 1", "'id'")


def test_an_entity_table_id_of_32_characters_is_accepted() -> None:
    longest = "a_9" * 10 + "bc"
    assert len(longest) == 32

    def change(doc: Document) -> None:
        doc["entities"] = {longest: {"khk_Cyrl": ["x"]}}
        row_of(doc, "entities")["entities"] = [longest]

    assert list(accepted(change).entities.spellings) == [longest]


@pytest.mark.parametrize("languages", [[MARKER], MARKER, None, 5])
def test_the_languages_of_an_entity_must_be_an_object(languages: object) -> None:
    def change(doc: Document) -> None:
        doc["entities"]["bat"] = languages

    assert_names(
        refused(changed(change)), "entity table", "'bat'", "'languages'", "object"
    )


@pytest.mark.parametrize("code", [f"{MARKER}_Latn", "eng", "ENG_LATN", ""])
def test_a_language_of_the_table_must_be_a_code_of_the_registry(code: str) -> None:
    def change(doc: Document) -> None:
        doc["entities"]["bat"][code] = ["x"]

    assert_names(
        refused(changed(change)), "entity table", "'bat'", "'languages'", "registry"
    )


@pytest.mark.parametrize("spellings", [[], ["x"] * 9, MARKER, None, {"a": 1}, 5])
def test_the_spellings_of_a_language_are_a_list_of_one_to_eight(
    spellings: object,
) -> None:
    def change(doc: Document) -> None:
        doc["entities"]["bat"]["khk_Cyrl"] = spellings

    assert_names(
        refused(changed(change)), "entity table", "'bat'", "'khk_Cyrl'", "1 to 8"
    )


@pytest.mark.parametrize(
    "spelling",
    ["", "  ", chr(0xA0), 5, None, ["a"], "x" * 201, f"{MARKER} <2eng_Latn>", "<t>"],
)
def test_a_spelling_is_a_non_blank_text_of_at_most_200_characters(
    spelling: object,
) -> None:
    def change(doc: Document) -> None:
        doc["entities"]["bat"]["khk_Cyrl"] = ["Бат", spelling]

    assert_names(refused(changed(change)), "entity table", "'bat'", "'khk_Cyrl[1]'")


def test_a_lone_surrogate_in_a_spelling_is_refused() -> None:
    sentinel = "SPELLING-SENTINEL"
    doc = document()
    doc["entities"]["bat"]["khk_Cyrl"] = [sentinel]
    data = json.dumps(doc, ensure_ascii=False).replace(sentinel, MARKER + "\\ud800")

    assert_names(refused(data.encode("utf-8")), "entity table", "'bat'", "surrogate")


def test_a_repeated_entity_id_in_the_table_is_refused() -> None:
    data = repeating(document(), '"entities": {"bat": ', '{}, "bat": ')

    assert_names(refused(data), "entity table", "more than once")


def test_a_repeated_language_of_an_entity_is_refused() -> None:
    data = repeating(document(), '"khk_Cyrl": ["Бат"]', ', "khk_Cyrl": ["Бат"]')

    assert_names(refused(data), "entity table", "'bat'", "more than once")


@pytest.mark.parametrize("ids", [["nobody"], ["bat", MARKER], [MARKER], ["Bat"], [""]])
def test_an_entity_id_that_is_not_in_the_table_is_refused(ids: list[str]) -> None:
    message = refused(changed(set_row_field("entities", "entities", ids)))

    assert_names(message, *at("entities", f"entities[{len(ids) - 1}]"), "table")


def test_an_entity_without_a_spelling_in_the_target_language_of_the_row_is_refused() -> (
    None
):
    def change(doc: Document) -> None:
        row_of(doc, "entities")["tgt_lang"] = "rus_Cyrl"

    message = refused(changed(change))

    assert_names(message, *at("entities", "entities[0]"), "spelling")


def test_an_entity_id_listed_twice_in_a_row_is_refused() -> None:
    message = refused(changed(set_row_field("entities", "entities", ["bat", "bat"])))

    assert_names(message, *at("entities", "entities[1]"), "more than once")


@pytest.mark.parametrize("value", [[], MARKER, None, 5, {"bat": 1}])
def test_the_entities_of_a_row_are_a_list_of_at_least_one(value: object) -> None:
    message = refused(changed(set_row_field("entities", "entities", value)))

    assert_names(message, *at("entities", "entities"), "list")


@pytest.mark.parametrize("item", [5, None, ["bat"], {"id": "bat"}])
def test_an_entity_id_of_a_row_is_a_string(item: object) -> None:
    message = refused(changed(set_row_field("entities", "entities", ["bat", item])))

    assert_names(message, *at("entities", "entities[1]"))


# ---- the terms of a row -------------------------------------------------------------------


@pytest.mark.parametrize("value", [[], MARKER, None, 5, {"src": "a", "tgt": "b"}])
def test_the_terms_of_a_row_are_a_list_of_at_least_one_pair(value: object) -> None:
    assert_names(
        refused(changed(set_row_field("terms", "terms", value))),
        *at("terms", "terms"),
        "list",
    )


@pytest.mark.parametrize("pair", [MARKER, 5, None, ["a", "b"]])
def test_a_term_pair_is_an_object(pair: object) -> None:
    message = refused(changed(set_row_field("terms", "terms", [pair])))

    assert_names(message, *at("terms", "terms[0]"), "object")


@pytest.mark.parametrize("missing", ["src", "tgt"])
def test_a_term_pair_has_both_sides(missing: str) -> None:
    pair = {"src": "table", "tgt": "ширээ"}
    del pair[missing]

    message = refused(changed(set_row_field("terms", "terms", [pair])))

    assert_names(message, *at("terms", f"terms[0].{missing}"), "missing")


def test_a_term_pair_has_no_other_field() -> None:
    pair = {"src": "table", "tgt": "ширээ", MARKER: "x"}

    message = refused(changed(set_row_field("terms", "terms", [pair])))

    assert_names(message, *at("terms", "terms[0]"), "not part of the format")


def test_a_repeated_field_of_a_term_pair_is_refused() -> None:
    data = repeating(document(), '"src": "table"', f', "src": "{MARKER}"')

    assert_names(refused(data), *at("terms", "terms[0]"), "'src'", "more than once")


@pytest.mark.parametrize("side", ["src", "tgt"])
@pytest.mark.parametrize(
    "term",
    ["", "   ", 5, None, "x" * 201, f"{MARKER} <2eng_Latn>", f"{MARKER} </r>"],
)
def test_a_term_is_a_non_blank_text_of_at_most_200_characters(
    side: str, term: object
) -> None:
    pair = {"src": "table", "tgt": "ширээ", side: term}

    message = refused(changed(set_row_field("terms", "terms", [pair])))

    assert_names(message, *at("terms", f"terms[0].{side}"))


def test_the_second_term_pair_of_a_row_is_named_by_its_position() -> None:
    pairs = [{"src": "table", "tgt": "ширээ"}, {"src": "", "tgt": "x"}]

    message = refused(changed(set_row_field("terms", "terms", pairs)))

    assert_names(message, *at("terms", "terms[1].src"))
