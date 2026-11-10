"""The rules of the suite file of D13 (spec T2 and T3), one function at a time: the constants, the
refusal and what it never quotes, the language check (`khk_Mong` refused by name, whatever the
registry holds), the id and sentence checks and the direction names. Every expectation is a
literal. The whole reader is in `test_eval_suite_refusals` and `test_eval_suites`."""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval import suite_rules
from rinchen.eval.suite_rules import (
    FORBIDDEN_LANGUAGES,
    HUB_LANGUAGE,
    KINDS,
    MAX_ROWS,
    MAX_SENTENCE_CHARS,
    MAX_SUITE_BYTES,
    PURPOSE_OF_SPLIT,
    SPLITS,
    SPLITS_OF_KIND,
    SUITE_SCHEMA,
    checked_language,
    checked_sentence,
    direction_name,
    is_row_id,
    refusal,
    split_direction,
)
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-84"
PLAIN = "SECRETMARKER84"  # no hyphen: a direction is split at hyphens
RULE_WORDS = ("khk_Mong", "refused by name", "Inner Mongolian")
PLACEHOLDER_3 = user_defined_symbols()[-PLACEHOLDER_COUNT:][3]


def message_of(error: pytest.ExceptionInfo[DataError]) -> str:
    return str(error.value)


# ---- the constants ------------------------------------------------------------------------


def test_the_constants_are_the_ones_of_the_spec() -> None:
    assert SUITE_SCHEMA == "rinchen.eval.suite/1"
    assert KINDS == ("benchmark", "human", "fixture")
    assert SPLITS == ("dev", "devtest", "test")
    assert MAX_ROWS == 100_000
    assert MAX_SUITE_BYTES == 64 * 2**20
    assert MAX_SENTENCE_CHARS == 5000
    assert FORBIDDEN_LANGUAGES == ("khk_Mong",)
    assert HUB_LANGUAGE == "khk_Cyrl"


def test_the_splits_each_kind_may_have_are_those_of_t2() -> None:
    assert dict(SPLITS_OF_KIND) == {
        "benchmark": ("dev", "devtest"),
        "human": ("test",),
        "fixture": ("dev", "devtest", "test"),
    }


def test_the_purpose_of_a_split_is_selection_for_dev_and_report_for_the_others() -> (
    None
):
    assert dict(PURPOSE_OF_SPLIT) == {
        "dev": "selection",
        "devtest": "report",
        "test": "report",
    }


def test_the_tables_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        SPLITS_OF_KIND["benchmark"] = ("test",)  # type: ignore[index]
    with pytest.raises(TypeError):
        PURPOSE_OF_SPLIT["devtest"] = "selection"  # type: ignore[index]


def test_the_hub_language_is_in_the_registry_and_the_forbidden_one_is_not() -> None:
    assert HUB_LANGUAGE in known_language_codes()
    assert not set(FORBIDDEN_LANGUAGES) & known_language_codes()


# ---- the refusal --------------------------------------------------------------------------


def test_a_refusal_is_a_data_error_that_says_what_kind_of_file_it_is_about() -> None:
    error = refusal("suite", "kind", "must be one of a, b")

    assert isinstance(error, DataError)
    assert str(error) == (
        "not an evaluation suite: suite, field 'kind': must be one of a, b"
    )


def test_a_refusal_without_a_field_names_the_place_only() -> None:
    assert str(refusal("row index 4", None, "is not an object")) == (
        "not an evaluation suite: row index 4: is not an object"
    )


# ---- the language -------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(known_language_codes()))
def test_every_code_of_the_registry_is_accepted(code: str) -> None:
    assert checked_language(code, "suite", "text") == code


def test_khk_mong_is_refused_by_name_with_the_rule_and_the_code() -> None:
    with pytest.raises(DataError) as raised:
        checked_language("khk_Mong", "row index 2", "text")

    message = message_of(raised)
    for fragment in (*RULE_WORDS, "row index 2", "'text'"):
        assert fragment in message
    assert message.startswith("not an evaluation suite:")


def test_khk_mong_stays_refused_if_the_registry_were_to_hold_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        suite_rules,
        "known_language_codes",
        lambda: frozenset({*known_language_codes(), "khk_Mong"}),
    )

    with pytest.raises(DataError) as raised:
        checked_language("khk_Mong", "suite", "directions[0]")

    for fragment in RULE_WORDS:
        assert fragment in message_of(raised)


@pytest.mark.parametrize("value", ["xx_Latn", "eng", "", MARKER, f"{MARKER}_Latn"])
def test_a_string_that_is_not_shaped_like_a_code_is_refused_and_never_quoted(
    value: str,
) -> None:
    with pytest.raises(DataError) as raised:
        checked_language(value, "suite", "text")

    message = message_of(raised)
    assert "language code of the registry" in message
    assert value not in message or value == ""
    assert MARKER not in message


def test_a_string_shaped_like_a_code_that_the_registry_lacks_is_named() -> None:
    with pytest.raises(DataError) as raised:
        checked_language("abc_Latn", "suite", "text")

    assert "abc_Latn is not a language code of the registry" in message_of(raised)


@pytest.mark.parametrize("value", [5, None, ["eng_Latn"], b"eng_Latn", 1.5])
def test_a_value_that_is_not_a_string_is_refused(value: Any) -> None:
    with pytest.raises(DataError) as raised:
        checked_language(value, "suite", "text")

    assert "language code of the registry" in message_of(raised)


def test_a_code_the_caller_excludes_is_refused_even_though_the_registry_has_it() -> (
    None
):
    assert checked_language("mn_Mong", "suite", "text") == "mn_Mong"

    with pytest.raises(DataError) as raised:
        checked_language("mn_Mong", "suite", "text", excluded=("mn_Mong",))

    message = message_of(raised)
    assert "mn_Mong is not a language of this benchmark" in message
    assert checked_language("eng_Latn", "suite", "text", excluded=("mn_Mong",)) == (
        "eng_Latn"
    )


# ---- the row id ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["g01", "a", "A.b_c-d:e", "0", "x" * 96, "doc:12-3", "..", "a..b"]
)
def test_an_id_of_the_pattern_is_a_row_id(value: str) -> None:
    assert is_row_id(value)


@pytest.mark.parametrize(
    "value",
    ["", "a b", "x" * 97, "é", "a/b", "a\n", "a\tb", " a", "a ", 5, None, b"ab", ["a"]],
)
def test_anything_else_is_not_a_row_id(value: object) -> None:
    assert not is_row_id(value)


# ---- the sentence -------------------------------------------------------------------------


def test_an_ordinary_sentence_is_returned_as_it_is() -> None:
    sentence = "  Муур сууж байв.  "

    assert checked_sentence(sentence, "row index 0", "text.khk_Cyrl") == sentence


def test_a_numbered_placeholder_is_text_a_sentence_may_hold() -> None:
    sentence = f"Where is {PLACEHOLDER_3}?"

    assert checked_sentence(sentence, "row index 0", "text.eng_Latn") == sentence


def test_a_sentence_of_the_largest_size_is_accepted_and_one_more_is_not() -> None:
    assert len(checked_sentence("x" * 5000, "p", "f")) == 5000

    with pytest.raises(DataError) as raised:
        checked_sentence("x" * 5001, "row index 0", "text.eng_Latn")

    assert "at most 5000 characters" in message_of(raised)


@pytest.mark.parametrize("value", ["", " ", "\t", "   ", chr(0x2003), 5, None, ["a"]])
def test_a_blank_or_a_non_text_is_not_a_sentence(value: object) -> None:
    with pytest.raises(DataError) as raised:
        checked_sentence(value, "row index 1", "text.eng_Latn")

    message = message_of(raised)
    assert "non-blank sentence" in message
    assert "row index 1" in message and "'text.eng_Latn'" in message


@pytest.mark.parametrize(
    "code", [0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029]
)
def test_every_line_break_the_engine_splits_at_is_refused(code: int) -> None:
    with pytest.raises(DataError) as raised:
        checked_sentence(f"{MARKER} a{chr(code)}b", "row index 0", "text.eng_Latn")

    message = message_of(raised)
    assert "line break" in message
    assert MARKER not in message


def test_a_lone_surrogate_is_refused() -> None:
    with pytest.raises(DataError) as raised:
        checked_sentence(f"{MARKER}{chr(0xD800)}", "row index 0", "text.eng_Latn")

    message = message_of(raised)
    assert "lone surrogate" in message
    assert MARKER not in message


def test_every_reserved_symbol_but_the_numbered_placeholders_is_refused() -> None:
    reserved = user_defined_symbols()[:-PLACEHOLDER_COUNT]

    assert len(reserved) >= 10
    for symbol in reserved:
        with pytest.raises(DataError) as raised:
            checked_sentence(f"{MARKER} {symbol} x", "row index 0", "text.eng_Latn")
        message = message_of(raised)
        assert "reserved symbol" in message
        assert MARKER not in message and symbol not in message


def test_every_numbered_placeholder_is_text() -> None:
    for symbol in user_defined_symbols()[-PLACEHOLDER_COUNT:]:
        assert checked_sentence(f"a {symbol} b", "p", "f")


# ---- the directions -----------------------------------------------------------------------


def test_a_direction_name_joins_the_languages_with_a_hyphen() -> None:
    assert direction_name("khk_Cyrl", "eng_Latn") == "khk_Cyrl-eng_Latn"
    assert direction_name("eng_Latn", "khk_Cyrl") == "eng_Latn-khk_Cyrl"


def test_a_direction_name_splits_back_into_its_languages() -> None:
    assert split_direction("khk_Cyrl-eng_Latn") == ("khk_Cyrl", "eng_Latn")
    assert split_direction(direction_name("tur_Latn", "zho_Hans")) == (
        "tur_Latn",
        "zho_Hans",
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "-",
        PLAIN,
        f"{PLAIN}-",
        f"-{PLAIN}",
        f"a-b-{PLAIN}",
        f"{PLAIN}--x",
        f"{PLAIN}-eng_Latn-khk_Cyrl",
    ],
)
def test_a_malformed_direction_is_a_data_error_that_does_not_quote_it(
    value: str,
) -> None:
    with pytest.raises(DataError) as raised:
        split_direction(value)

    message = message_of(raised)
    assert message.startswith("not an evaluation suite:")
    assert "<source>-<target>" in message
    assert PLAIN not in message


def test_the_place_of_a_malformed_direction_can_be_named() -> None:
    with pytest.raises(DataError) as raised:
        split_direction("khk_Cyrl", place="the option --direction")

    assert "the option --direction" in message_of(raised)
