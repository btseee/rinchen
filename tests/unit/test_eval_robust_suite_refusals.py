"""Refusals of the suite reader of D12 (spec T11) about the file, the sections and the fields of
a row every section shares: the file is exactly the document the writer makes, and anything else
is a `DataError` that names where it looked (the section, the row index and id, the field) and
never quotes what the file held. The refusals about texts, languages, scripts, the entity table
and the terms are in `test_eval_robust_suite_field_refusals`. Every expectation is a literal."""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.eval.robust_suite import (
    MAX_SUITE_BYTES,
    MAX_TEXT_CHARS,
    SECTIONS,
    SUITE_SCHEMA,
    Suite,
)
from tests.support.robust_suite_documents import (
    COMMON,
    EXTRA,
    MARKER,
    PLACEHOLDER_5,
    Document,
    accepted,
    assert_names,
    at,
    changed,
    document,
    drop_row_field,
    encode,
    refused,
    repeating,
    row_of,
    set_row_field,
)

# ---- what is accepted ---------------------------------------------------------------------


def test_the_document_the_tests_start_from_is_a_valid_suite() -> None:
    suite = Suite.from_bytes(encode(document()))

    assert suite.name == "tiny"
    assert suite.row_counts() == {section: 1 for section in SECTIONS}
    assert suite.entities.spellings_of("bat", "eng_Latn") == ("Bat",)


def test_a_numbered_placeholder_is_text_a_row_may_hold() -> None:
    suite = Suite.from_bytes(encode(document()))

    placeholder_row = suite.sections["placeholders"][0]
    assert PLACEHOLDER_5 in placeholder_row.source
    assert PLACEHOLDER_5 in (placeholder_row.reference or "")


def test_texts_of_the_largest_size_are_accepted() -> None:
    suite = accepted(
        set_row_field("numbers", "source", "a" * MAX_TEXT_CHARS),
    )

    assert len(suite.sections["numbers"][0].source) == MAX_TEXT_CHARS


def test_the_same_id_may_be_used_by_two_sections() -> None:
    def change(doc: Document) -> None:
        row_of(doc, "placeholders")["id"] = "n1"

    suite = accepted(change)

    assert suite.sections["numbers"][0].id == suite.sections["placeholders"][0].id


def test_a_source_language_may_be_missing_in_a_script_row_only() -> None:
    suite = accepted(set_row_field("script", "src_lang", None))

    assert suite.sections["script"][0].src_lang is None


@pytest.mark.parametrize(
    ("tgt_lang", "script"),
    [
        ("khk_Cyrl", "Mong"),
        ("khk_Cyrl", "Cyrl"),
        ("mn_Mong", "Mong"),
        ("mn_Mong", "Cyrl"),
        ("eng_Latn", None),
    ],
)
def test_a_script_belongs_to_a_target_that_is_khalkha_or_traditional_mongolian(
    tgt_lang: str, script: str | None
) -> None:
    def change(doc: Document) -> None:
        row_of(doc, "script").update(tgt_lang=tgt_lang, script=script)

    assert accepted(change).sections["script"][0].script == script


@pytest.mark.parametrize("section", ["entities", "terms", "script"])
def test_a_reference_may_be_null_where_the_format_says_so(section: str) -> None:
    assert (
        accepted(set_row_field(section, "reference", None))
        .sections[section][0]
        .reference
        is None
    )


def test_a_section_may_be_empty_when_another_has_rows() -> None:
    def change(doc: Document) -> None:
        doc["sections"] = {"numbers": doc["sections"]["numbers"], "terms": []}

    assert accepted(change).row_counts() == {"numbers": 1, "terms": 0}


def test_eight_spellings_of_200_characters_and_a_term_of_200_characters_are_accepted() -> (
    None
):
    def change(doc: Document) -> None:
        doc["entities"]["bat"]["khk_Cyrl"] = [f"{index}" * 200 for index in range(8)]
        row_of(doc, "terms")["terms"] = [{"src": "s" * 200, "tgt": "t" * 200}]

    suite = accepted(change)

    assert len(suite.entities.spellings_of("bat", "khk_Cyrl")) == 8
    assert suite.sections["terms"][0].terms[0].tgt == "t" * 200


def test_a_reference_of_64_numbers_is_accepted_and_one_of_65_is_not() -> None:
    sixty_four = ", ".join(str(number) for number in range(1, 65))
    sixty_five = ", ".join(str(number) for number in range(1, 66))

    for section in ("numbers", "placeholders"):
        assert accepted(set_row_field(section, "reference", sixty_four))
        message = refused(changed(set_row_field(section, "reference", sixty_five)))
        assert_names(message, *at(section, "reference"), "64 numbers")


@pytest.mark.parametrize("section", ["entities", "terms", "script", "mong"])
def test_the_limit_of_numbers_is_a_rule_of_the_numbers_and_placeholders_sections(
    section: str,
) -> None:
    sixty_five = ", ".join(str(number) for number in range(1, 66))

    assert accepted(set_row_field(section, "reference", sixty_five))


# ---- the file as a whole ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (b'{"name": "\xff"}', "UTF-8"),
        (b"\xef\xbb\xbf" + encode(document()), "JSON"),
        (b"\xff\xfe{}", "UTF-8"),
        (b"{not json", "JSON"),
        (f'{{"name": "{MARKER}" 5}}'.encode(), "JSON"),
        (f'{{"name": "{MARKER}"'.encode() + b"\xff", "UTF-8"),
        (f'["{MARKER}"]'.encode(), "JSON object"),
        (b"", "JSON"),
        (b"[" * 100_000, "JSON"),
        (b"[]", "JSON object"),
        (b'"text"', "JSON object"),
        (b"7", "JSON object"),
        (b"null", "JSON object"),
    ],
)
def test_bytes_that_are_not_a_json_object_are_refused(data: bytes, named: str) -> None:
    assert named in refused(data)


@pytest.mark.parametrize(
    "schema", ["rinchen.robustness.suite/2", "", 1, None, [SUITE_SCHEMA]]
)
def test_another_schema_is_refused(schema: object) -> None:
    def change(doc: Document) -> None:
        doc["schema"] = schema

    message = refused(changed(change))

    assert_names(message, "'schema'", SUITE_SCHEMA)


@pytest.mark.parametrize("key", ["schema", "name", "entities", "sections"])
def test_each_top_level_key_is_required(key: str) -> None:
    def change(doc: Document) -> None:
        del doc[key]

    assert_names(refused(changed(change)), "suite", f"'{key}'", "missing")


def test_a_top_level_key_the_format_does_not_define_is_refused_without_naming_it() -> (
    None
):
    def change(doc: Document) -> None:
        doc[MARKER] = 1

    assert_names(refused(changed(change)), "suite", "not part of the format")


@pytest.mark.parametrize("name", ["", "a b", "x" * 65, 5, None, f"{MARKER} 2", ["a"]])
def test_a_name_that_is_not_one_to_64_name_characters_is_refused(name: object) -> None:
    def change(doc: Document) -> None:
        doc["name"] = name

    assert_names(refused(changed(change)), "suite", "'name'")


def test_a_repeated_top_level_key_is_refused() -> None:
    data = repeating(document(), '"name": "tiny"', ', "name": "again"')

    assert_names(refused(data), "suite", "'name'", "more than once")


def test_a_repeated_key_that_is_no_field_of_the_format_is_refused_without_naming_it() -> (
    None
):
    data = repeating(document(), '"name": "tiny"', f', "{MARKER}": 1, "{MARKER}": 2')

    assert_names(refused(data), "suite", "more than once")


# ---- the sections -------------------------------------------------------------------------


def test_a_section_the_format_does_not_have_is_refused_without_naming_it() -> None:
    def change(doc: Document) -> None:
        doc["sections"][MARKER] = [row_of(doc, "numbers")]

    assert_names(refused(changed(change)), "'sections'", "not a section of the format")


@pytest.mark.parametrize("sections", [[], "numbers", None, 3])
def test_sections_must_be_an_object(sections: object) -> None:
    def change(doc: Document) -> None:
        doc["sections"] = sections

    assert_names(refused(changed(change)), "'sections'", "object")


@pytest.mark.parametrize("sections", [{}, {"numbers": []}, {"numbers": [], "mong": []}])
def test_a_suite_with_no_row_at_all_is_refused(sections: dict[str, list[Any]]) -> None:
    def change(doc: Document) -> None:
        doc["sections"] = sections

    assert_names(refused(changed(change)), "'sections'", "at least one row")


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("value", [{}, "text", None, 5])
def test_a_section_must_be_a_list(section: str, value: object) -> None:
    def change(doc: Document) -> None:
        doc["sections"][section] = value

    assert_names(refused(changed(change)), f"'{section}'", "list")


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("value", [MARKER, 5, None, [MARKER]])
def test_a_row_must_be_an_object(section: str, value: object) -> None:
    def change(doc: Document) -> None:
        doc["sections"][section].append(value)

    assert_names(refused(changed(change)), f"'{section}'", "index 1", "object")


def test_a_repeated_section_is_refused() -> None:
    data = repeating(document(), '"sections": {', '"numbers": [], ')

    assert_names(refused(data), "'sections'", "'numbers'", "more than once")


# ---- the fields of a row ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "field"), [(s, f) for s in SECTIONS for f in (*COMMON, *EXTRA[s])]
)
def test_each_field_of_a_row_is_required(section: str, field: str) -> None:
    message = refused(changed(drop_row_field(section, field)))

    assert_names(message, *at(section, field, named_id=field != "id"), "missing")


@pytest.mark.parametrize("section", SECTIONS)
def test_a_field_the_row_does_not_define_is_refused_without_naming_it(
    section: str,
) -> None:
    message = refused(changed(set_row_field(section, MARKER, 1)))

    assert_names(message, *at(section, None), "not part of the format")


FOREIGN_FIELD = {
    "numbers": "terms",
    "placeholders": "script",
    "entities": "terms",
    "terms": "entities",
    "script": "source_mong",
    "mong": "script",
}


@pytest.mark.parametrize("section", SECTIONS)
def test_a_field_of_another_section_is_refused_in_this_one(section: str) -> None:
    message = refused(changed(set_row_field(section, FOREIGN_FIELD[section], None)))

    assert_names(message, *at(section, None), "not part of the format")


@pytest.mark.parametrize("section", SECTIONS)
def test_a_repeated_field_of_a_row_is_refused(section: str) -> None:
    source = document()["sections"][section][0]["source"]
    data = repeating(document(), f'"source": "{source}"', f', "source": "{MARKER}"')

    assert_names(refused(data), *at(section, "source"), "more than once")


@pytest.mark.parametrize("section", SECTIONS)
def test_a_repeated_field_that_is_no_field_of_the_format_is_refused(
    section: str,
) -> None:
    source = document()["sections"][section][0]["source"]
    data = repeating(
        document(), f'"source": "{source}"', f', "{MARKER}": 1, "{MARKER}": 2'
    )

    assert_names(refused(data), *at(section, None), "more than once")


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize(
    "value", [5, None, [], f"{MARKER} id", "", "x" * 65, f"{MARKER}.{'x' * 60}", "a/b"]
)
def test_an_id_is_one_to_64_characters_of_letters_digits_dot_underscore_and_dash(
    section: str, value: object
) -> None:
    message = refused(changed(set_row_field(section, "id", value)))

    assert_names(message, f"'{section}'", "index 0", "'id'")


def test_an_id_of_64_characters_and_every_allowed_character_is_accepted() -> None:
    allowed = "aZ09._-" + "x" * 57

    assert (
        accepted(set_row_field("numbers", "id", allowed)).sections["numbers"][0].id
        == allowed
    )


@pytest.mark.parametrize("section", SECTIONS)
def test_an_id_repeated_in_a_section_is_refused_at_the_second_row(section: str) -> None:
    def change(doc: Document) -> None:
        doc["sections"][section].append(dict(row_of(doc, section)))

    assert_names(
        refused(changed(change)), *at(section, "id", index=1), "more than once"
    )


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize(
    "value", [5, None, "Upper", "", "x" * 25, f"{MARKER} k", "a-b"]
)
def test_a_category_is_one_to_24_lowercase_letters_digits_and_underscores(
    section: str, value: object
) -> None:
    message = refused(changed(set_row_field(section, "category", value)))

    assert_names(message, *at(section, "category"))


def test_a_category_of_24_characters_is_accepted() -> None:
    assert accepted(set_row_field("numbers", "category", "a_9" * 8))


# ---- the size ----------------------------------------------------------------------------


def test_a_suite_of_more_than_the_largest_size_is_refused_by_the_parser_too() -> None:
    data = encode(document()) + b" " * MAX_SUITE_BYTES

    assert_names(refused(data), "8 MiB")
