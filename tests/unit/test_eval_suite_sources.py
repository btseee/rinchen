"""The readers that turn rows of LOCAL files into a suite (D13, T4): `lines_rows` (a text file with
one sentence per line, the line number is the id), `build_multiway` (the rows of several languages
joined by id, with the hub language Khalkha as the one that sets the order and the directions) and
`human_rows` (D2's export of a human set). All of it works over plain rows, so no test needs a
dataset or the network; the Parquet, TSV and JSON Lines reading is the command's
(`test_cli_eval_suite_*`).

Every expectation is a literal. Each refusal is checked for what it names (a language, a line, a
count, a row index) and for what it must never say (a text of the file)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.eval.suite_rules import SuiteSource
from rinchen.eval.suite_sources import (
    HUMAN_COLUMNS,
    SourceRow,
    build_multiway,
    human_rows,
    lines_rows,
)
from rinchen.eval.suites import Suite
from tests.support.eval_suite_cli import EXPORT_COLUMNS, export_row, golden_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-83"
NINE = (
    "khk_Cyrl",
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
)
SIX_OTHERS = ("deu_Latn", "jpn_Jpan", "kaz_Cyrl", "kor_Hang", "tur_Latn", "zho_Hans")
IDS = ("a", "b", "c", "d")


def plain_source(**changes: Any) -> SuiteSource:
    fields: dict[str, Any] = {
        "dataset": None,
        "revision": None,
        "pinned": False,
        "files": {},
        "missing_languages": (),
    }
    fields.update(changes)
    return SuiteSource(**fields)


def language_rows(
    language: str, ids: Sequence[str] = IDS, docs: Sequence[str | None] | None = None
) -> list[SourceRow]:
    given = docs if docs is not None else ("d1", "d1", "d2", "d2")
    return [
        SourceRow(row_id, f"{language} sentence {row_id}", given[index % len(given)])
        for index, row_id in enumerate(ids)
    ]


def three_languages() -> dict[str, list[SourceRow]]:
    return {name: language_rows(name) for name in ("khk_Cyrl", "eng_Latn", "rus_Cyrl")}


def multiway(
    languages: dict[str, Sequence[SourceRow]] | None = None, **changes: Any
) -> Suite:
    arguments: dict[str, Any] = {
        "name": "mini-devtest",
        "kind": "benchmark",
        "benchmark": "flores_plus",
        "split": "devtest",
        "languages": three_languages() if languages is None else languages,
        "configured": NINE,
        "source": plain_source(),
    }
    arguments.update(changes)
    return build_multiway(**arguments)


def refused(call: Callable[[], object], error: type[Exception] = DataError) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


# ---- lines_rows ---------------------------------------------------------------------------


def test_three_lines_give_the_ids_l000001_to_l000003() -> None:
    rows = lines_rows("eng_Latn", b"one\ntwo\nthree\n")

    assert rows == [
        SourceRow("l000001", "one", None),
        SourceRow("l000002", "two", None),
        SourceRow("l000003", "three", None),
    ]


def test_a_final_line_end_is_optional() -> None:
    assert lines_rows("eng_Latn", b"one\ntwo") == lines_rows("eng_Latn", b"one\ntwo\n")


def test_the_lines_are_read_as_utf_8_text() -> None:
    assert [row.text for row in lines_rows("khk_Cyrl", "муур\nнохой\n".encode())] == [
        "муур",
        "нохой",
    ]


@pytest.mark.parametrize(
    ("data", "line"),
    [
        (b"one\n\nthree\n", 2),
        (b"one\ntwo\n\n", 3),
        (b"\none\n", 1),
        (b"one\n   \nthree\n", 2),
        (b"one\n\t\nthree", 2),
    ],
)
def test_a_blank_line_is_refused_with_its_number_and_the_language(
    data: bytes, line: int
) -> None:
    message = refused(lambda: lines_rows("eng_Latn", data))

    assert "eng_Latn" in message
    assert f"line {line}" in message
    assert "one" not in message
    assert "three" not in message


def test_a_blank_line_message_never_quotes_a_neighbour() -> None:
    data = f"{MARKER}\n\n{MARKER}\n".encode()

    assert MARKER not in refused(lambda: lines_rows("eng_Latn", data))


def test_a_file_with_windows_line_ends_is_refused_naming_the_first_line() -> None:
    message = refused(lambda: lines_rows("eng_Latn", b"one\r\ntwo\r\n"))

    assert "eng_Latn" in message
    assert "line 1" in message
    assert "one" not in message


def test_a_carriage_return_later_in_the_file_names_its_line() -> None:
    assert "line 3" in refused(lambda: lines_rows("eng_Latn", b"a\nb\nc\r\nd\n"))


def test_bytes_that_are_not_utf_8_are_refused_without_quoting_them() -> None:
    message = refused(lambda: lines_rows("eng_Latn", b"ok\n\xff\xfe\n"))

    assert "eng_Latn" in message
    assert "UTF-8" in message
    assert "xff" not in message


def test_an_empty_file_is_refused() -> None:
    assert "eng_Latn" in refused(lambda: lines_rows("eng_Latn", b""))


def test_a_line_break_that_is_not_a_line_feed_stays_in_its_line_for_the_suite_to_refuse() -> (
    None
):
    rows = lines_rows("eng_Latn", f"one{chr(0x2028)}two\nthree\nfour\nfive\n".encode())
    languages = {
        "eng_Latn": rows,
        "khk_Cyrl": [SourceRow(row.id, f"k{row.id}", None) for row in rows],
    }

    message = refused(lambda: multiway(languages))

    assert "l000001" in message
    assert "line break" in message


@pytest.mark.parametrize("language", ["khk_Mong", "xx_Latn", "english"])
def test_a_language_outside_the_registry_is_refused_by_the_lines_reader(
    language: str,
) -> None:
    message = refused(lambda: lines_rows(language, b"one\n"))

    if language == "khk_Mong":
        assert "khk_Mong is refused by name" in message
    assert "one" not in message


# ---- build_multiway -----------------------------------------------------------------------


def test_three_languages_make_the_four_directions_between_the_hub_and_each_partner() -> (
    None
):
    suite = multiway()

    assert suite.directions == (
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
    )
    assert "eng_Latn-rus_Cyrl" not in suite.directions
    assert "rus_Cyrl-eng_Latn" not in suite.directions


def test_the_languages_the_source_lacks_are_the_configured_ones_not_present_sorted() -> (
    None
):
    suite = multiway()

    assert suite.source.missing_languages == SIX_OTHERS


def test_the_missing_languages_are_sorted_whatever_the_order_of_the_configuration() -> (
    None
):
    suite = multiway(configured=tuple(reversed(NINE)))

    assert suite.source.missing_languages == SIX_OTHERS


def test_the_missing_languages_are_computed_and_not_taken_from_the_source_given() -> (
    None
):
    suite = multiway(source=plain_source(missing_languages=("zho_Hans",)))

    assert suite.source.missing_languages == SIX_OTHERS


def test_nothing_is_missing_when_every_configured_language_is_present() -> None:
    languages = {name: language_rows(name) for name in NINE}

    suite = multiway(languages)

    assert suite.source.missing_languages == ()
    assert len(suite.directions) == 16


def test_the_directions_follow_the_order_of_the_standard_directions() -> None:
    languages = {
        name: language_rows(name) for name in ("zho_Hans", "khk_Cyrl", "deu_Latn")
    }

    suite = multiway(languages)

    assert suite.directions == (
        "khk_Cyrl-deu_Latn",
        "deu_Latn-khk_Cyrl",
        "khk_Cyrl-zho_Hans",
        "zho_Hans-khk_Cyrl",
    )


def test_the_rows_follow_the_order_of_the_hub_and_are_joined_by_id() -> None:
    languages = {
        "khk_Cyrl": language_rows("khk_Cyrl", ids=("c", "a", "d", "b")),
        "eng_Latn": language_rows("eng_Latn", ids=("b", "d", "a", "c")),
    }

    suite = multiway(languages)

    assert [row.id for row in suite.rows] == ["c", "a", "d", "b"]
    assert [dict(row.text) for row in suite.rows][:2] == [
        {"khk_Cyrl": "khk_Cyrl sentence c", "eng_Latn": "eng_Latn sentence c"},
        {"khk_Cyrl": "khk_Cyrl sentence a", "eng_Latn": "eng_Latn sentence a"},
    ]


def test_the_document_of_a_row_is_the_hubs() -> None:
    languages = three_languages()
    languages["eng_Latn"] = language_rows("eng_Latn", docs=("other", "other"))

    suite = multiway(languages)

    assert [row.doc for row in suite.rows] == ["d1", "d1", "d2", "d2"]


def test_the_suite_a_builder_returns_is_the_one_its_own_bytes_read_back_to() -> None:
    suite = multiway()

    assert Suite.from_bytes(suite.to_bytes()).to_bytes() == suite.to_bytes()
    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "mini-devtest",
        "benchmark",
        "flores_plus",
        "devtest",
    )


def test_the_files_of_the_source_are_kept() -> None:
    digest = "ab" * 32
    suite = multiway(
        source=plain_source(files={"eng_Latn/shard-00000.parquet": digest})
    )

    assert dict(suite.source.files) == {"eng_Latn/shard-00000.parquet": digest}


def test_a_suite_without_the_hub_language_is_a_configuration_error() -> None:
    languages = three_languages()
    del languages["khk_Cyrl"]

    message = refused(lambda: multiway(languages), ConfigError)

    assert "khk_Cyrl" in message


def test_a_language_with_one_id_fewer_is_refused_with_the_counts_and_never_intersected() -> (
    None
):
    languages = three_languages()
    languages["eng_Latn"] = language_rows("eng_Latn", ids=("a", "b", "c"))

    message = refused(lambda: multiway(languages))

    assert "eng_Latn" in message
    assert "3" in message
    assert "4" in message
    assert "khk_Cyrl" in message
    assert "sentence" not in message


def test_a_language_with_another_id_of_the_same_count_is_refused() -> None:
    languages = three_languages()
    languages["rus_Cyrl"] = language_rows("rus_Cyrl", ids=("a", "b", "c", "z"))

    message = refused(lambda: multiway(languages))

    assert "rus_Cyrl" in message
    assert "4" in message


def test_a_language_with_more_ids_than_the_hub_is_refused() -> None:
    languages = three_languages()
    languages["rus_Cyrl"] = language_rows("rus_Cyrl", ids=("a", "b", "c", "d", "e"))

    message = refused(lambda: multiway(languages))

    assert "rus_Cyrl" in message
    assert "5" in message


@pytest.mark.parametrize("language", ["khk_Cyrl", "eng_Latn"])
def test_a_repeated_id_is_refused_naming_the_language(language: str) -> None:
    languages = three_languages()
    # The same set of ids as the hub's, one of them twice: only the check for a repeated id
    # sees it (a set of the ids would give the hub's).
    languages[language] = language_rows(language, ids=("a", "b", "c", "d", "d"))

    message = refused(lambda: multiway(languages))

    assert language in message
    assert "more than once" in message
    assert "sentence" not in message


def test_inner_mongolian_as_a_language_key_is_refused_by_name() -> None:
    languages = three_languages()
    # Rows that do not agree with the hub's: the language is refused before they are joined.
    languages["khk_Mong"] = language_rows("khk_Mong", ids=("a", "b"))

    assert "khk_Mong is refused by name" in refused(lambda: multiway(languages))


def test_inner_mongolian_among_the_configured_languages_is_refused_by_name() -> None:
    assert "khk_Mong is refused by name" in refused(
        lambda: multiway(configured=(*NINE, "khk_Mong"))
    )


def test_a_code_outside_the_registry_is_refused() -> None:
    languages = three_languages()
    languages["abc_Latn"] = language_rows("abc_Latn", ids=("a", "b"))

    message = refused(lambda: multiway(languages))

    assert "abc_Latn" in message
    assert "registry" in message


def test_traditional_script_mongolian_is_not_a_language_of_flores_plus() -> None:
    languages = three_languages()
    languages["mn_Mong"] = language_rows("mn_Mong", ids=("a", "b"))

    message = refused(lambda: multiway(languages))

    assert "mn_Mong" in message
    assert "benchmark" in message


def test_traditional_script_mongolian_is_a_language_of_a_fixture_and_follows_the_standard_pairs() -> (
    None
):
    languages = three_languages()
    languages["mn_Mong"] = language_rows("mn_Mong")

    suite = multiway(languages, kind="fixture", benchmark="tiny", name="tiny-devtest")

    assert suite.directions == (
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-mn_Mong",
        "mn_Mong-khk_Cyrl",
    )


def test_a_suite_with_no_partner_language_has_no_direction_and_is_refused() -> None:
    assert "direction" in refused(
        lambda: multiway({"khk_Cyrl": language_rows("khk_Cyrl")})
    )


def test_a_document_that_is_not_an_id_is_refused_by_the_round_trip_naming_the_field() -> (
    None
):
    languages = three_languages()
    languages["khk_Cyrl"] = language_rows(
        "khk_Cyrl", docs=("https://example.org/news/first",)
    )

    message = refused(lambda: multiway(languages))

    assert "doc" in message
    assert "row index 0" in message
    assert "example.org" not in message


def test_a_blank_sentence_is_refused_naming_the_row_and_never_quoted() -> None:
    languages = three_languages()
    rows = language_rows("eng_Latn")
    rows[1] = SourceRow("b", "   ", "d1")
    languages["eng_Latn"] = rows

    message = refused(lambda: multiway(languages))

    assert "row index 1" in message
    assert "'b'" in message
    assert "text.eng_Latn" in message


def test_a_refusal_never_quotes_a_text_of_the_source() -> None:
    languages = three_languages()
    languages["eng_Latn"] = [
        SourceRow(row.id, f"{MARKER} {row.text}", row.doc)
        for row in language_rows("eng_Latn", ids=("a", "b", "c"))
    ]

    assert MARKER not in refused(lambda: multiway(languages))


# ---- human_rows ---------------------------------------------------------------------------


def human(rows: list[dict[str, Any]], **changes: Any) -> Suite:
    arguments: dict[str, Any] = {
        "name": "human-test",
        "source": plain_source(files={"export.jsonl": "cd" * 32}),
    }
    arguments.update(changes)
    return human_rows(rows, **arguments)


def test_the_export_columns_are_the_nine_of_the_annotation_app() -> None:
    assert HUMAN_COLUMNS == EXPORT_COLUMNS


def test_a_golden_export_gives_the_ids_of_its_documents_and_segments() -> None:
    suite = human(golden_export())

    assert [row.id for row in suite.rows] == [
        "d1-00001",
        "d1-00002",
        "d1-00003",
        "d2-00001",
        "d2-00002",
        "d2-00003",
    ]
    assert [row.doc for row in suite.rows] == ["d1", "d1", "d1", "d2", "d2", "d2"]
    assert dict(suite.rows[0].text) == {
        "eng_Latn": "The cat sat.",
        "khk_Cyrl": "Муур сууж байв.",
    }
    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "human-test",
        "human",
        "human",
        "test",
    )
    assert dict(suite.source.files) == {"export.jsonl": "cd" * 32}
    assert suite.source.missing_languages == ()


def test_an_english_to_khalkha_set_has_no_khalkha_to_english_direction() -> None:
    suite = human(golden_export())

    assert suite.directions == ("eng_Latn-khk_Cyrl",)
    assert suite.view("eng_Latn-khk_Cyrl").references[0] == "Муур сууж байв."


def test_the_directions_are_the_pairs_present_sorted() -> None:
    rows = [
        *golden_export()[:2],
        export_row("d9", 1, "Привет.", "Сайн байна уу.", src_lang="rus_Cyrl"),
    ]

    suite = human(rows)

    assert suite.directions == ("eng_Latn-khk_Cyrl", "rus_Cyrl-khk_Cyrl")
    assert suite.view("eng_Latn-khk_Cyrl").ids == ("d1-00001", "d1-00002")
    assert suite.view("rus_Cyrl-khk_Cyrl").ids == ("d9-00001",)


def test_both_ways_of_a_pair_in_one_export_are_refused_because_a_row_cannot_say_which_way_it_goes() -> (
    None
):
    rows = [
        *golden_export()[:2],
        export_row(
            "d3", 1, "Сайн.", "Hello.", src_lang="khk_Cyrl", tgt_lang="eng_Latn"
        ),
    ]

    message = refused(lambda: human(rows))

    assert "eng_Latn" in message
    assert "khk_Cyrl" in message
    assert "Hello" not in message


def test_the_order_of_the_export_does_not_matter() -> None:
    rows = golden_export()
    expected = human(list(rows)).to_bytes()

    for seed in (1, 2, 3, 4, 5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        assert human(shuffled).to_bytes() == expected
    assert human(list(reversed(rows))).to_bytes() == expected


def test_the_segment_zero_is_a_segment_and_gives_the_id_of_its_document() -> None:
    rows = [export_row("d1", 0, "First.", "F."), export_row("d1", 1, "Second.", "S.")]

    assert [row.id for row in human(rows).rows] == ["d1-00000", "d1-00001"]


def test_segments_sort_by_number_and_not_by_text() -> None:
    rows = [
        export_row("d1", number, f"Sentence {number}.", f"S{number}.")
        for number in (10, 2, 1)
    ]

    assert [row.id for row in human(rows).rows] == ["d1-00001", "d1-00002", "d1-00010"]


def test_a_row_without_a_document_gets_the_number_of_its_place_in_the_sorted_rows() -> (
    None
):
    rows = [
        export_row(None, None, "Beta.", "B."),
        export_row("d1", 1, "Doc.", "D."),
        export_row(None, None, "Alpha.", "A."),
    ]

    suite = human(rows)

    assert [row.id for row in suite.rows] == ["s000001", "s000002", "d1-00001"]
    assert [row.text["eng_Latn"] for row in suite.rows] == ["Alpha.", "Beta.", "Doc."]
    assert [row.doc for row in suite.rows] == [None, None, "d1"]


def test_rows_that_tie_on_every_sort_key_but_the_target_still_sort_the_same_way() -> (
    None
):
    rows = [
        export_row(None, None, "Same.", "Second."),
        export_row(None, None, "Same.", "First."),
    ]

    assert human(rows).to_bytes() == human(list(reversed(rows))).to_bytes()
    assert [row.text["khk_Cyrl"] for row in human(rows).rows] == ["First.", "Second."]


@pytest.mark.parametrize(
    ("field", "value"),
    [("origin", "web"), ("origin", "synthetic"), ("split", "train"), ("split", "dev")],
)
def test_a_row_that_is_not_a_human_test_row_is_refused_naming_its_index(
    field: str, value: str
) -> None:
    rows = golden_export()
    rows[2][field] = value

    message = refused(lambda: human(rows))

    assert "row index 2" in message
    assert field in message
    assert value not in message


def test_two_sets_in_one_export_are_refused() -> None:
    rows = golden_export()
    rows[4]["set"] = "other"

    message = refused(lambda: human(rows))

    assert "set" in message
    assert "row index 4" in message
    assert "other" not in message


def test_a_set_that_is_not_text_is_refused() -> None:
    rows = golden_export()
    rows[0]["set"] = None

    assert "row index 0" in refused(lambda: human(rows))


def test_a_repeated_document_and_segment_is_refused_as_a_repeated_id() -> None:
    rows = golden_export()
    rows[1]["seg_idx"] = 1

    message = refused(lambda: human(rows))

    assert "d1-00001" in message
    assert "more than once" in message


@pytest.mark.parametrize("column", ["src_lang", "tgt_lang"])
def test_inner_mongolian_is_refused_by_name_in_either_language_column(
    column: str,
) -> None:
    rows = golden_export()
    rows[3][column] = "khk_Mong"

    message = refused(lambda: human(rows))

    assert "khk_Mong is refused by name" in message
    assert "row index 3" in message


def test_a_language_outside_the_registry_is_refused() -> None:
    rows = golden_export()
    rows[0]["src_lang"] = "xx_Latn"

    assert "row index 0" in refused(lambda: human(rows))


def test_a_row_cannot_translate_a_language_into_itself() -> None:
    rows = golden_export()
    rows[0]["tgt_lang"] = "eng_Latn"

    assert "row index 0" in refused(lambda: human(rows))


def test_traditional_script_mongolian_is_a_language_of_a_human_set() -> None:
    mongolian = "".join(chr(code) for code in (0x1830, 0x1820, 0x1822, 0x1828))
    rows = [export_row("d1", 1, "Hello.", mongolian, tgt_lang="mn_Mong")]

    assert human(rows).directions == ("eng_Latn-mn_Mong",)


@pytest.mark.parametrize("field", ["src", "tgt"])
def test_a_blank_sentence_is_refused_naming_the_input_row_and_never_quoted(
    field: str,
) -> None:
    rows = golden_export()
    rows[5][field] = "  "
    rows[5]["tgt" if field == "src" else "src"] = MARKER

    message = refused(lambda: human(rows))

    assert "row index 5" in message
    assert field in message
    assert MARKER not in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seg_idx", "1"),
        ("seg_idx", True),
        ("seg_idx", -1),
        ("seg_idx", 1.5),
        ("doc_id", 7),
        ("doc_id", ""),
        ("doc_id", "has space"),
        ("doc_id", "x" * 97),
        ("src", 5),
        ("src", None),
        ("origin", None),
    ],
)
def test_a_value_of_the_wrong_type_or_shape_is_refused_naming_its_field(
    field: str, value: object
) -> None:
    rows = golden_export()
    rows[1][field] = value

    message = refused(lambda: human(rows))

    assert "row index 1" in message
    assert field in message


def test_a_document_without_a_segment_number_has_its_row_numbered_by_place() -> None:
    rows = golden_export()[:2]
    rows[0]["seg_idx"] = None

    suite = human(rows)

    assert [row.id for row in suite.rows] == ["s000001", "d1-00002"]
    assert [row.doc for row in suite.rows] == ["d1", "d1"]


@pytest.mark.parametrize("column", EXPORT_COLUMNS)
def test_a_missing_column_is_named(column: str) -> None:
    rows = golden_export()
    del rows[3][column]

    message = refused(lambda: human(rows))

    assert column in message
    assert "row index 3" in message


def test_a_column_that_is_not_part_of_the_export_is_refused_without_naming_it() -> None:
    rows = golden_export()
    rows[2][MARKER] = "x"

    message = refused(lambda: human(rows))

    assert "row index 2" in message
    assert MARKER not in message


def test_an_empty_export_is_refused() -> None:
    assert "no row" in refused(lambda: human([]))


def test_an_export_with_a_sentence_over_the_limit_is_refused_by_the_suite() -> None:
    rows = golden_export()
    rows[0]["src"] = "x" * 5001

    assert "row index 0" in refused(lambda: human(rows))


def test_the_suite_of_a_human_export_is_the_one_its_own_bytes_read_back_to() -> None:
    suite = human(golden_export())

    assert Suite.from_bytes(suite.to_bytes()).to_bytes() == suite.to_bytes()
