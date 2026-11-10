"""Refusals of the suite reader of D13 (spec T2 and T3): the file is exactly the document the
writer makes, and anything else is a `DataError` that names where it looked (a field of the
format, a row index and a well-formed id) and never quotes what the file held. Each test plants
`MARKER` in the value a refusal is about; `refused` fails when a message holds it. Every
expectation is a literal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

import pytest

from rinchen.errors import DataError
from rinchen.eval import suite_rules, suites
from rinchen.eval.suite_rules import SUITE_SCHEMA
from rinchen.eval.suites import Suite, read_suite
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from tests.support.eval_suite_documents import (
    DIGEST,
    MARKER,
    REVISION,
    Document,
    accepted,
    assert_names,
    at_row,
    changed,
    document,
    encode,
    refused,
    repeating,
    set_row,
    set_sentence,
    set_source,
    set_top,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

RULE_WORDS = ("khk_Mong", "refused by name", "Inner Mongolian")
MONGOLIAN = "".join(chr(code) for code in (0x1820, 0x1828, 0x1821))
PLACEHOLDER_3 = user_defined_symbols()[-PLACEHOLDER_COUNT:][3]
TOP_KEYS = (
    "schema",
    "name",
    "kind",
    "benchmark",
    "split",
    "source",
    "directions",
    "rows",
)
SOURCE_KEYS = ("dataset", "revision", "pinned", "files", "missing_languages")


# ---- what is accepted ---------------------------------------------------------------------


def test_the_document_the_tests_start_from_is_a_valid_suite() -> None:
    suite = Suite.from_bytes(encode(document()))

    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "tiny",
        "fixture",
        "tiny",
        "devtest",
    )
    assert suite.directions == ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl")
    assert [row.id for row in suite.rows] == ["r1", "r2"]


def test_a_numbered_placeholder_is_text_a_sentence_may_hold() -> None:
    suite = accepted(set_sentence("eng_Latn", f"Where is {PLACEHOLDER_3}?"))

    assert suite.rows[0].text["eng_Latn"] == f"Where is {PLACEHOLDER_3}?"


def test_a_sentence_of_5000_characters_is_accepted_and_one_of_5001_is_not() -> None:
    suite = accepted(set_sentence("eng_Latn", "x" * 5000))
    assert len(suite.rows[0].text["eng_Latn"]) == 5000

    message = refused(changed(set_sentence("eng_Latn", "x" * 5001)))

    assert_names(message, *at_row(0, "text.eng_Latn", "r1"), "5000 characters")


def test_a_human_suite_may_hold_traditional_script_mongolian() -> None:
    def change(doc: Document) -> None:
        doc.update(kind="human", split="test", benchmark="annotators")
        doc["directions"] = ["khk_Cyrl-mn_Mong"]
        doc["rows"][0]["text"]["mn_Mong"] = MONGOLIAN

    suite = accepted(change)

    assert suite.directions == ("khk_Cyrl-mn_Mong",)
    assert suite.rows[0].text["mn_Mong"] == MONGOLIAN


def test_every_kind_and_split_pair_of_t2_is_accepted() -> None:
    pairs = [
        ("benchmark", "dev"),
        ("benchmark", "devtest"),
        ("human", "test"),
        ("fixture", "dev"),
        ("fixture", "devtest"),
        ("fixture", "test"),
    ]
    for kind, split in pairs:

        def change(doc: Document, kind: str = kind, split: str = split) -> None:
            doc.update(kind=kind, split=split)

        suite = accepted(change)
        assert (suite.kind, suite.split) == (kind, split)


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
    "schema",
    [
        "rinchen.eval.suite/2",
        "rinchen.robustness.suite/1",
        "",
        1,
        None,
        [SUITE_SCHEMA],
        f"{MARKER}",
    ],
)
def test_another_schema_is_refused(schema: object) -> None:
    message = refused(changed(set_top("schema", schema)))

    assert_names(message, "'schema'", SUITE_SCHEMA)


@pytest.mark.parametrize("key", TOP_KEYS)
def test_each_top_level_key_is_required(key: str) -> None:
    def change(doc: Document) -> None:
        del doc[key]

    assert_names(refused(changed(change)), "suite", f"'{key}'", "missing")


def test_a_top_level_key_the_format_does_not_define_is_refused_without_naming_it() -> (
    None
):
    assert_names(
        refused(changed(set_top(MARKER, 1))), "suite", "not part of the format"
    )


def test_a_repeated_top_level_key_is_refused_by_name() -> None:
    data = repeating(document(), '"name": "tiny"', ', "name": "again"')

    assert_names(refused(data), "suite", "'name'", "more than once")


def test_a_repeated_key_that_is_no_field_of_the_format_is_refused_without_naming_it() -> (
    None
):
    data = repeating(document(), '"name": "tiny"', f', "{MARKER}": 1, "{MARKER}": 2')

    assert_names(refused(data), "suite", "more than once")


def test_a_repeated_field_of_a_row_is_refused_by_name_and_names_the_row() -> None:
    data = repeating(document(), '"id": "r1"', ', "id": "again"')

    assert_names(refused(data), *at_row(0, None, "again"), "'id'", "more than once")


def test_a_repeated_language_of_a_row_is_refused_by_its_code() -> None:
    data = repeating(
        document(), '"eng_Latn": "The cat sat."', f', "eng_Latn": "{MARKER}"'
    )

    assert_names(refused(data), "row index 0", "'eng_Latn'", "more than once")


def test_a_repeated_key_of_the_source_is_refused() -> None:
    data = repeating(document(), '"pinned": false', ', "pinned": true')

    assert_names(refused(data), "suite", "'source'", "'pinned'", "more than once")


# ---- name, kind, benchmark, split ----------------------------------------------------------


@pytest.mark.parametrize("field", ["name", "benchmark"])
@pytest.mark.parametrize(
    "value", ["", "a b", "x" * 65, "a/b", 5, None, f"{MARKER} 2", ["a"], "é"]
)
def test_a_name_that_is_not_one_to_64_name_characters_is_refused(
    field: str, value: object
) -> None:
    assert_names(refused(changed(set_top(field, value))), "suite", f"'{field}'")


@pytest.mark.parametrize("value", ["fixtures", "", None, 3, ["fixture"], MARKER])
def test_a_kind_that_is_not_one_of_the_three_is_refused(value: object) -> None:
    message = refused(changed(set_top("kind", value)))

    assert_names(message, "'kind'", "benchmark, human, fixture")


@pytest.mark.parametrize("value", ["train", "", None, 3, ["dev"], MARKER, "Dev"])
def test_a_split_that_is_not_one_of_the_three_is_refused(value: object) -> None:
    message = refused(changed(set_top("split", value)))

    assert_names(message, "'split'", "dev, devtest, test")


@pytest.mark.parametrize(
    ("kind", "split", "named"),
    [
        ("benchmark", "test", "dev or devtest"),
        ("human", "dev", "test"),
        ("human", "devtest", "test"),
    ],
)
def test_a_kind_and_a_split_that_do_not_go_together_are_refused(
    kind: str, split: str, named: str
) -> None:
    def change(doc: Document) -> None:
        doc.update(kind=kind, split=split)

    assert_names(refused(changed(change)), "'split'", kind, named)


# ---- the source ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [[], "source", None, 3])
def test_the_source_must_be_an_object(value: object) -> None:
    assert_names(refused(changed(set_top("source", value))), "'source'", "object")


@pytest.mark.parametrize("key", SOURCE_KEYS)
def test_each_key_of_the_source_is_required(key: str) -> None:
    def change(doc: Document) -> None:
        del doc["source"][key]

    assert_names(refused(changed(change)), f"source, field '{key}'", "missing")


def test_a_key_of_the_source_the_format_does_not_define_is_refused_unnamed() -> None:
    message = refused(changed(set_source(MARKER, 1)))

    assert_names(message, "source", "not part of the format")


@pytest.mark.parametrize(
    "value", ["", "a b", "x" * 201, 5, ["a"], f"{MARKER} x", "é/b", "a\nb"]
)
def test_a_dataset_that_is_not_a_hub_id_is_refused(value: object) -> None:
    assert_names(refused(changed(set_source("dataset", value))), "'dataset'")


def test_a_dataset_is_a_hub_id_or_null() -> None:
    for value in (None, "openlanguagedata/flores_plus", "x" * 200, "ntrex"):
        assert accepted(set_source("dataset", value)).source.dataset == value


@pytest.mark.parametrize(
    "revision",
    [
        "0123456789abcdef0123456789abcdef0123456",
        "0123456789abcdef0123456789abcdef012345678",
        "0123456789ABCDEF0123456789ABCDEF01234567",
        "0123456789abcdef0123456789abcdef0123456g",
        "",
        5,
        True,
        ["0123456789abcdef0123456789abcdef01234567"],
        f"{MARKER}",
    ],
)
def test_a_revision_that_is_not_40_lowercase_hex_characters_is_refused(
    revision: object,
) -> None:
    def change(doc: Document) -> None:
        doc["source"].update(revision=revision, pinned=True)

    assert_names(refused(changed(change)), "source, field 'revision'", "40")


def test_a_revision_of_40_hex_characters_is_accepted_pinned_or_not() -> None:
    for pinned in (True, False):

        def change(doc: Document, pinned: bool = pinned) -> None:
            doc["source"].update(revision=REVISION, pinned=pinned)

        suite = accepted(change)
        assert (suite.source.revision, suite.source.pinned) == (REVISION, pinned)


def test_a_suite_cannot_be_pinned_without_a_revision() -> None:
    message = refused(changed(set_source("pinned", True)))

    assert_names(message, "source, field 'pinned'", "revision")


@pytest.mark.parametrize("pinned", ["true", 1, 0, None, [True], MARKER])
def test_pinned_must_be_true_or_false(pinned: object) -> None:
    assert_names(refused(changed(set_source("pinned", pinned))), "'pinned'", "true or")


@pytest.mark.parametrize("files", [[], "files", None, 3])
def test_the_files_must_be_an_object(files: object) -> None:
    assert_names(refused(changed(set_source("files", files))), "'files'", "object")


@pytest.mark.parametrize(
    "digest",
    [
        "ab" * 31 + "a",
        "ab" * 33,
        "AB" * 32,
        "xy" * 32,
        "",
        5,
        None,
        MARKER,
        ["ab" * 32],
    ],
)
def test_a_digest_that_is_not_64_lowercase_hex_characters_is_refused(
    digest: object,
) -> None:
    message = refused(changed(set_source("files", {"shard-0.parquet": digest})))

    assert_names(message, "source, field 'files[0]'", "64")


@pytest.mark.parametrize(
    "name",
    [
        f"../{MARKER}.parquet",
        f"a/../{MARKER}",
        f"/{MARKER}",
        f"a//{MARKER}",
        f"./{MARKER}",
        f"{MARKER}/",
        f"{MARKER}/..",
        "x" * (201 - len(MARKER)) + MARKER,
        "",
        f"{MARKER}\nb",
        f"{MARKER}\x00",
    ],
)
def test_a_file_name_that_is_not_a_plain_relative_path_is_refused(name: str) -> None:
    message = refused(changed(set_source("files", {name: DIGEST})))

    assert_names(message, "source, field 'files[0]'", "relative path")


def test_the_second_file_is_named_by_its_position() -> None:
    files = {"a.parquet": DIGEST, "../b": DIGEST}

    assert_names(refused(changed(set_source("files", files))), "'files[1]'")


def test_relative_paths_of_the_largest_size_and_with_dots_inside_a_name_are_accepted() -> (
    None
):
    files = {
        "khk_Cyrl/shard-00000.parquet": DIGEST,
        "a..b.txt": DIGEST,
        "x" * 200: DIGEST,
        "дэд/файл.txt": DIGEST,
    }

    suite = accepted(set_source("files", files))

    assert dict(suite.source.files) == files


@pytest.mark.parametrize("languages", [{}, "eng_Latn", None, 3])
def test_missing_languages_must_be_a_list(languages: object) -> None:
    message = refused(changed(set_source("missing_languages", languages)))

    assert_names(message, "'missing_languages'", "list")


@pytest.mark.parametrize(
    "languages",
    [["kor_Hang", "deu_Latn"], ["deu_Latn", "deu_Latn"], ["tur_Latn", "kor_Hang"]],
)
def test_missing_languages_are_sorted_and_have_no_repeat(
    languages: list[str],
) -> None:
    message = refused(changed(set_source("missing_languages", languages)))

    assert_names(message, "source, field 'missing_languages'", "sorted")


def test_missing_languages_of_the_registry_in_order_are_accepted() -> None:
    suite = accepted(set_source("missing_languages", ["deu_Latn", "kor_Hang"]))

    assert suite.source.missing_languages == ("deu_Latn", "kor_Hang")


# ---- languages ----------------------------------------------------------------------------


def test_khk_mong_is_refused_by_name_in_a_direction_a_row_and_the_missing_languages() -> (
    None
):
    def in_direction(doc: Document) -> None:
        doc["directions"] = ["khk_Mong-eng_Latn"]

    def in_target(doc: Document) -> None:
        doc["directions"] = ["eng_Latn-khk_Mong"]

    def in_row(doc: Document) -> None:
        doc["rows"][1]["text"]["khk_Mong"] = "x"

    def in_missing(doc: Document) -> None:
        doc["source"]["missing_languages"] = ["khk_Mong"]

    for change, place in (
        (in_direction, "directions[0]"),
        (in_target, "directions[0]"),
        (in_row, "row index 1"),
        (in_missing, "missing_languages[0]"),
    ):
        message = refused(changed(change))
        assert_names(message, place, *RULE_WORDS)


def test_khk_mong_is_refused_in_every_kind_of_suite() -> None:
    for kind, split in (
        ("benchmark", "devtest"),
        ("human", "test"),
        ("fixture", "dev"),
    ):

        def change(doc: Document, kind: str = kind, split: str = split) -> None:
            doc.update(kind=kind, split=split)
            doc["rows"][0]["text"]["khk_Mong"] = "x"

        assert_names(refused(changed(change)), *RULE_WORDS)


@pytest.mark.parametrize("code", ["xx_Latn", "eng", "", f"{MARKER}", "ENG_LATN", "5"])
def test_a_language_that_is_not_in_the_registry_is_refused_and_not_quoted(
    code: str,
) -> None:
    def in_row(doc: Document) -> None:
        doc["rows"][0]["text"] = {code: "x", "eng_Latn": "y"}

    def in_direction(doc: Document) -> None:
        doc["directions"] = [f"{code}-eng_Latn"]

    row_message = refused(changed(in_row))
    direction_message = refused(changed(in_direction))

    assert_names(row_message, "language code of the registry", "row index 0")
    assert_names(direction_message, "'directions[0]'")
    for message in (row_message, direction_message):
        assert "xx_Latn" not in message and "ENG_LATN" not in message


def test_a_code_shaped_like_a_language_that_the_registry_lacks_is_named() -> None:
    def change(doc: Document) -> None:
        doc["rows"][0]["text"]["abc_Latn"] = "x"

    assert_names(refused(changed(change)), "abc_Latn is not a language code")


def test_mn_mong_is_refused_in_a_flores_plus_benchmark_wherever_it_is_named() -> None:
    def base(doc: Document) -> None:
        doc.update(kind="benchmark", benchmark="flores_plus", split="devtest")

    def in_row(doc: Document) -> None:
        base(doc)
        doc["rows"][0]["text"]["mn_Mong"] = "x"

    def in_direction(doc: Document) -> None:
        # No row has the language: only the direction itself can be what is refused.
        base(doc)
        doc["directions"].append("khk_Cyrl-mn_Mong")

    def in_missing(doc: Document) -> None:
        base(doc)
        doc["source"]["missing_languages"] = ["mn_Mong"]

    for change, place in (
        (in_row, "row index 0"),
        (in_direction, "'directions[2]'"),
        (in_missing, "'missing_languages[0]'"),
    ):
        assert_names(
            refused(changed(change)),
            place,
            "mn_Mong is not a language of this benchmark",
        )


def test_mn_mong_is_accepted_in_a_flores_plus_fixture_or_another_benchmark() -> None:
    def fixture(doc: Document) -> None:
        doc.update(benchmark="flores_plus", kind="fixture")
        doc["rows"][0]["text"]["mn_Mong"] = "x"

    def other(doc: Document) -> None:
        doc.update(kind="benchmark", benchmark="ntrex", split="devtest")
        doc["rows"][0]["text"]["mn_Mong"] = "x"

    for change in (fixture, other):
        assert accepted(change).rows[0].text["mn_Mong"] == "x"


# ---- the directions -----------------------------------------------------------------------


@pytest.mark.parametrize("directions", [{}, "khk_Cyrl-eng_Latn", None, 3])
def test_directions_must_be_a_list(directions: object) -> None:
    assert_names(
        refused(changed(set_top("directions", directions))), "'directions'", "list"
    )


def test_a_suite_with_no_direction_is_refused() -> None:
    assert_names(
        refused(changed(set_top("directions", []))), "'directions'", "at least one"
    )


@pytest.mark.parametrize(
    "direction",
    [
        "khk_Cyrl",
        "khk_Cyrl-eng_Latn-deu_Latn",
        "khk_Cyrl--eng_Latn",
        "-eng_Latn",
        "khk_Cyrl-",
        "",
        5,
        None,
        ["khk_Cyrl-eng_Latn"],
        f"{MARKER}",
    ],
)
def test_a_direction_that_is_not_two_codes_joined_by_a_hyphen_is_refused(
    direction: object,
) -> None:
    message = refused(changed(set_top("directions", [direction])))

    assert_names(message, "'directions[0]'")


def test_a_direction_with_equal_languages_is_refused() -> None:
    message = refused(changed(set_top("directions", ["eng_Latn-eng_Latn"])))

    assert_names(message, "'directions[0]'", "two different languages")


def test_a_direction_twice_is_refused_at_its_second_place() -> None:
    directions = ["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"]

    message = refused(changed(set_top("directions", directions)))

    assert_names(message, "'directions[2]'", "more than once")


def test_a_direction_no_row_has_both_languages_of_is_refused() -> None:
    def change(doc: Document) -> None:
        doc["directions"].append("khk_Cyrl-deu_Latn")
        doc["rows"][1]["text"] = {"eng_Latn": "A dog runs.", "deu_Latn": "Hund"}

    message = refused(changed(change))

    assert_names(message, "'directions[2]'", "no row with both languages")


def test_a_direction_that_one_row_alone_makes_is_accepted() -> None:
    def change(doc: Document) -> None:
        doc["directions"].append("khk_Cyrl-deu_Latn")
        doc["rows"][1]["text"]["deu_Latn"] = "Hund"

    assert "khk_Cyrl-deu_Latn" in accepted(change).directions


# ---- the rows -----------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [{}, "rows", None, 3])
def test_rows_must_be_a_list(rows: object) -> None:
    assert_names(refused(changed(set_top("rows", rows))), "'rows'", "list")


def test_a_suite_with_no_row_is_refused() -> None:
    assert_names(refused(changed(set_top("rows", []))), "'rows'", "at least one row")


def test_more_rows_than_the_cap_are_refused_before_a_row_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suite_rules, "MAX_ROWS", 3)

    def change(doc: Document) -> None:
        del doc["rows"][0]["id"]
        doc["rows"] = [doc["rows"][0], *(dict(doc["rows"][1]) for _ in range(3))]

    message = refused(changed(change))

    assert_names(message, "'rows'", "at most 3 rows")


def test_rows_up_to_the_cap_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suite_rules, "MAX_ROWS", 3)

    def change(doc: Document) -> None:
        doc["rows"] = [
            {"id": f"r{n}", "doc": None, "text": dict(doc["rows"][0]["text"])}
            for n in range(3)
        ]

    assert len(accepted(change).rows) == 3


@pytest.mark.parametrize("row", [[], "row", None, 3, [{"id": "r1"}]])
def test_a_row_must_be_an_object(row: object) -> None:
    def change(doc: Document) -> None:
        doc["rows"][1] = row

    assert_names(refused(changed(change)), *at_row(1, None, None), "object")


@pytest.mark.parametrize("key", ["id", "doc", "text"])
def test_each_field_of_a_row_is_required(key: str) -> None:
    def change(doc: Document) -> None:
        del doc["rows"][0][key]

    assert_names(
        refused(changed(change)),
        *at_row(0, key, "r1" if key != "id" else None),
        "missing",
    )


def test_a_field_of_a_row_the_format_does_not_define_is_refused_unnamed() -> None:
    message = refused(changed(set_row(MARKER, 1)))

    assert_names(message, *at_row(0, None, "r1"), "not part of the format")


@pytest.mark.parametrize(
    "row_id",
    ["", "a b", "x" * 97, "é", "a/b", 5, None, ["r"], f"{MARKER} 2", "a\nb"],
)
def test_an_id_that_does_not_match_its_pattern_is_refused_and_not_named(
    row_id: object,
) -> None:
    message = refused(changed(set_row("id", row_id)))

    assert_names(message, *at_row(0, "id", None), "'id'")


def test_an_id_of_96_characters_and_every_character_of_the_pattern_is_accepted() -> (
    None
):
    for row_id in ("x" * 96, "A.b_c-d:e9", "0"):
        assert accepted(set_row("id", row_id)).rows[0].id == row_id


def test_a_duplicate_id_is_refused_at_its_second_row() -> None:
    message = refused(changed(set_row("id", "r1", index=1)))

    assert_names(message, *at_row(1, "id", "r1"), "more than once")


@pytest.mark.parametrize(
    "doc", ["", "a b", "x" * 97, "é", 5, ["d"], f"{MARKER} 2", "a\nb", True]
)
def test_a_document_that_does_not_match_its_pattern_is_refused(doc: object) -> None:
    message = refused(changed(set_row("doc", doc)))

    assert_names(message, *at_row(0, "doc", "r1"))


def test_a_document_is_an_id_or_null() -> None:
    assert accepted(set_row("doc", None)).rows[0].doc is None
    assert accepted(set_row("doc", "doc:12-3")).rows[0].doc == "doc:12-3"


@pytest.mark.parametrize("text", [{}, [], "text", None, 3, [["eng_Latn", "x"]]])
def test_the_text_of_a_row_must_be_an_object_with_at_least_one_language(
    text: object,
) -> None:
    message = refused(changed(set_row("text", text)))

    assert_names(message, *at_row(0, "text", "r1"))


@pytest.mark.parametrize("sentence", ["", " ", "\t\t", chr(0x2003), 5, None, ["a"], {}])
def test_a_blank_sentence_or_a_non_text_is_refused(sentence: object) -> None:
    message = refused(changed(set_sentence("eng_Latn", sentence)))

    assert_names(message, *at_row(0, "text.eng_Latn", "r1"), "non-blank")


@pytest.mark.parametrize(
    "code", [0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029]
)
def test_a_sentence_with_a_line_break_is_refused(code: int) -> None:
    message = refused(changed(set_sentence("khk_Cyrl", f"{MARKER}{chr(code)}x")))

    assert_names(message, *at_row(0, "text.khk_Cyrl", "r1"), "line break")


def test_a_sentence_with_a_lone_surrogate_is_refused() -> None:
    doc = document()
    doc["rows"][0]["text"]["eng_Latn"] = f"{MARKER}{chr(0xD800)}"
    data = encode(doc, ensure_ascii=True)
    assert b"\\ud800" in data

    message = refused(data)

    assert_names(message, *at_row(0, "text.eng_Latn", "r1"), "lone surrogate")


def test_a_sentence_with_a_reserved_symbol_is_refused_and_a_placeholder_is_not() -> (
    None
):
    for symbol in ("<2eng_Latn>", "<bt>", "<doc>", "<sep>", "</t>"):
        message = refused(changed(set_sentence("eng_Latn", f"{MARKER} {symbol}")))
        assert_names(message, *at_row(0, "text.eng_Latn", "r1"), "reserved symbol")
        assert symbol not in message


def test_the_second_language_of_the_row_is_named_in_its_own_field() -> None:
    message = refused(changed(set_sentence("khk_Cyrl", "", index=1)))

    assert_names(message, *at_row(1, "text.khk_Cyrl", "r2"))


# ---- the cap on the size and reading a file -------------------------------------------------


def test_a_file_over_the_size_cap_is_refused_before_anything_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = encode(document())
    monkeypatch.setattr(suites, "MAX_SUITE_BYTES", len(good))
    path = tmp_path / "huge.json"
    path.write_bytes(good + b" ")

    def never_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a file over the cap was parsed")

    monkeypatch.setattr(json, "loads", never_parse)

    with pytest.raises(DataError, match="larger than"):
        read_suite(path)
    with pytest.raises(DataError, match="larger than"):
        Suite.from_bytes(good + b" ")


def test_read_suite_asks_for_at_most_one_byte_over_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = encode(document())
    monkeypatch.setattr(suites, "MAX_SUITE_BYTES", len(good))
    path = tmp_path / "huge.json"
    path.write_bytes(good + b" " * 1000)
    asked: list[int] = []
    real_open = Path.open

    class Recording:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __enter__(self) -> Self:
            self.inner.__enter__()
            return self

        def __exit__(self, *exc: object) -> object:
            return self.inner.__exit__(*exc)

        def read(self, size: int = -1) -> bytes:
            asked.append(size)
            data: bytes = self.inner.read(size)
            return data

    def opening(self: Path, *args: Any, **kwargs: Any) -> Recording:
        return Recording(real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", opening)

    with pytest.raises(DataError, match="larger than"):
        read_suite(path)

    assert asked == [len(good) + 1]


def test_a_file_of_exactly_the_cap_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = encode(document())
    monkeypatch.setattr(suites, "MAX_SUITE_BYTES", len(good))
    path = tmp_path / "largest.json"
    path.write_bytes(good)

    suite, _ = read_suite(path)

    assert suite == Suite.from_bytes(good)


def test_the_message_about_the_size_cap_gives_it_in_mebibytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suites, "MAX_SUITE_BYTES", 2 * 2**20)

    with pytest.raises(DataError) as raised:
        Suite.from_bytes(b" " * (2 * 2**20 + 1))

    assert str(raised.value) == "not an evaluation suite: the file is larger than 2 MiB"


def test_a_path_that_is_missing_or_a_directory_is_a_data_error(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(DataError, match="cannot read the suite file") as raised:
            read_suite(path)
        assert str(path) in str(raised.value)
        assert isinstance(raised.value.__cause__, OSError)
