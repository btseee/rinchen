"""Rule baseline: lexicon first, phonographic fallback second.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable. The fallback's own letters and rules are tested in
`test_script_conv_fallback.py`.
"""

from __future__ import annotations

import pytest

from rinchen.errors import DataError
from rinchen.script_conv import fallback, rules
from rinchen.script_conv.alphabet import NNBSP
from rinchen.script_conv.fallback import development_split, transliterate_fallback
from rinchen.script_conv.rules import (
    LEXICON,
    WordResult,
    _load_lexicon,
    _parse_lexicon,
    transliterate,
    transliterate_word,
)
from rinchen.script_conv.validate import validate_traditional
from tests.support.script_reference import (
    FIXTURE_DIR,
    all_reference_rows,
    read_tsv_rows,
)

GOLDEN_PATH = FIXTURE_DIR / "rules_golden.tsv"
SCRIPT_COMMA = chr(0x1802)
SCRIPT_FULL_STOP = chr(0x1803)
MONGOL = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
SUFFIX_UN = chr(0x1824) + chr(0x1828)
SCRIPT_A = chr(0x1820)
HEADER = "cyrillic\tscript\n"

# A lexicon key whose verified spelling differs from the fallback's, so a lookup
# that silently fell back would give a different answer.
LEXICON_ONLY_KEY = next(
    word
    for word, script in sorted(LEXICON.items())
    if transliterate_fallback(word) != script
)


def test_word_result_is_a_frozen_value() -> None:
    assert WordResult("x", "fallback") == WordResult("x", "fallback")
    with pytest.raises(AttributeError):
        WordResult("x", "fallback").text = "y"  # type: ignore[misc]


def test_the_fallback_names_resolve_from_rules() -> None:
    assert rules.transliterate_fallback is fallback.transliterate_fallback
    assert rules.FALLBACK_LETTERS is fallback.FALLBACK_LETTERS
    assert rules.development_split is fallback.development_split


def test_lexicon_is_read_only() -> None:
    with pytest.raises(TypeError):
        LEXICON["x"] = "y"  # type: ignore[index]


def test_every_lexicon_value_passes_the_validator() -> None:
    assert LEXICON
    for word, script in LEXICON.items():
        assert validate_traditional(script) == (), word


def test_lexicon_keys_are_the_development_words() -> None:
    assert all(development_split(word) for word in LEXICON)


def test_golden_rows_convert_exactly_with_their_recorded_path() -> None:
    rows = read_tsv_rows(GOLDEN_PATH)

    assert len(rows) == 100
    for row in rows:
        result = transliterate_word(row["cyrillic"])
        assert result == WordResult(row["expected"], row["path"]), row["cyrillic"]


def test_lexicon_hit_returns_the_verified_spelling() -> None:
    result = transliterate_word(LEXICON_ONLY_KEY)

    assert result == WordResult(LEXICON[LEXICON_ONLY_KEY], "lexicon")
    assert result.text != transliterate_fallback(LEXICON_ONLY_KEY)
    assert result.source == "lexicon"


def test_lookup_ignores_case() -> None:
    expected = WordResult(LEXICON[LEXICON_ONLY_KEY], "lexicon")

    for spelling in (
        LEXICON_ONLY_KEY.upper(),
        LEXICON_ONLY_KEY.capitalize(),
        LEXICON_ONLY_KEY.swapcase(),
    ):
        result = transliterate_word(spelling)
        assert result == expected, spelling
        assert result.source == "lexicon", spelling


def test_a_word_with_an_uppercase_letter_outside_the_lexicon_falls_back() -> None:
    assert "бббб" not in LEXICON

    result = transliterate_word("Бббб")

    assert result == WordResult(transliterate_fallback("бббб"), "fallback")
    assert result.source == "fallback"


def test_transliterate_word_output_over_the_whole_reference_passes_the_validator() -> (
    None
):
    for row in all_reference_rows():
        result = transliterate_word(row["cyrillic"])
        assert validate_traditional(result.text) == (), row["cyrillic"]


def test_transliterate_maps_comma_and_full_stop_and_keeps_the_space() -> None:
    text = transliterate("Монгол, улс.")

    first = transliterate_word("Монгол").text
    second = transliterate_word("улс").text
    assert text == first + SCRIPT_COMMA + " " + second + SCRIPT_FULL_STOP
    assert text.endswith(SCRIPT_FULL_STOP)
    assert text.index(SCRIPT_COMMA) == len(first)


def test_transliterate_keeps_digits_hyphen_and_latin_and_converts_cyrillic() -> None:
    text = transliterate("2024оны н-р iPhone-ыг")

    expected = (
        "2024"
        + transliterate_word("оны").text
        + " "
        + transliterate_word("н").text
        + "-"
        + transliterate_word("р").text
        + " iPhone-"
        + transliterate_word("ыг").text
    )
    assert text == expected


def test_transliterate_empty_text_is_empty() -> None:
    assert transliterate("") == ""


def test_transliterate_passes_text_already_in_script_through() -> None:
    text = MONGOL + NNBSP + SUFFIX_UN + " " + MONGOL + SCRIPT_FULL_STOP

    assert transliterate(text) == text


def test_transliterate_keeps_other_punctuation_and_spaces() -> None:
    assert transliterate("(1) 50%!?") == "(1) 50%!?"
    assert transliterate("  ") == "  "


@pytest.mark.parametrize(
    "text",
    ["3.5", "1,5", "1,000", "2024.01.05", "a.b", "iPhone.com", "https://a.b/"],
)
def test_a_dot_or_comma_between_ascii_letters_or_digits_stays(text: str) -> None:
    assert transliterate(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5.", "5" + SCRIPT_FULL_STOP),
        (".5", SCRIPT_FULL_STOP + "5"),
        ("...", SCRIPT_FULL_STOP * 3),
        ("1,", "1" + SCRIPT_COMMA),
        ("a, b", "a" + SCRIPT_COMMA + " b"),
        ("1.5.", "1.5" + SCRIPT_FULL_STOP),
        ("https://a.b/.", "https://a.b/" + SCRIPT_FULL_STOP),
    ],
)
def test_a_dot_or_comma_not_between_ascii_letters_or_digits_maps(
    text: str, expected: str
) -> None:
    assert transliterate(text) == expected


def test_a_dot_between_cyrillic_words_maps() -> None:
    expected = (
        transliterate_word("Монгол").text
        + SCRIPT_FULL_STOP
        + transliterate_word("Улс").text
    )

    assert transliterate("Монгол.Улс") == expected


def test_dots_after_cyrillic_abbreviation_letters_map() -> None:
    expected = (
        transliterate_word("т").text
        + SCRIPT_FULL_STOP
        + transliterate_word("м").text
        + SCRIPT_FULL_STOP
    )

    assert transliterate("т.м.") == expected


def test_a_number_next_to_a_cyrillic_suffix_keeps_its_decimal_point() -> None:
    assert transliterate("3.5оны") == "3.5" + transliterate_word("оны").text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cyrillic\tscript\n", {}),
        ("cyrillic\tscript\nаа\tX\nбб\tY\n", {"аа": "X", "бб": "Y"}),
        ("cyrillic\tscript\nаа\tX", {"аа": "X"}),
    ],
)
def test_parse_lexicon_reads_the_pairs(text: str, expected: dict[str, str]) -> None:
    assert _parse_lexicon(text) == expected


@pytest.mark.parametrize(
    ("text", "line", "reason"),
    [
        ("", 1, "empty"),
        ("\n", 1, "header"),
        ("word\tscript\nаа\tX\n", 1, "header"),
        ("cyrillic\n", 1, "header"),
        ("cyrillic\tscript\textra\n", 1, "header"),
        ("cyrillic\tscript\nаа\n", 2, "2 tab-separated fields, found 1"),
        ("cyrillic\tscript\nаа\tX\tY\n", 2, "2 tab-separated fields, found 3"),
        ("cyrillic\tscript\nаа\t\n", 2, "empty field"),
        ("cyrillic\tscript\n\tX\n", 2, "empty field"),
        ("cyrillic\tscript\nаа\tX\nаа\tY\n", 3, "duplicate"),
        ("cyrillic\tscript\nаа\tX\n\nбб\tY\n", 3, "blank line"),
        ("cyrillic\tscript\nаа\tX\n\n", 3, "blank line"),
        ("cyrillic\tscript\nАа\tX\n", 2, "not lower-case"),
        ("cyrillic\tscript\nаа\tX\nБб\tY\n", 3, "not lower-case"),
    ],
)
def test_parse_lexicon_rejects_a_bad_file_and_names_the_line(
    text: str, line: int, reason: str
) -> None:
    with pytest.raises(DataError, match=rf"lexicon\.tsv:{line}\b.*{reason}"):
        _parse_lexicon(text)


def test_parse_lexicon_accepts_the_script_columns_it_is_given_verbatim() -> None:
    script = SCRIPT_A + NNBSP + SCRIPT_A

    assert _parse_lexicon(HEADER + "аа\t" + script + "\n") == {"аа": script}


class _MissingResource:
    def joinpath(self, *names: str) -> _MissingResource:
        return self

    def read_text(self, encoding: str) -> str:
        raise FileNotFoundError("lexicon.tsv is gone")


def test_a_missing_lexicon_file_is_a_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rules.resources, "files", lambda package: _MissingResource())

    with pytest.raises(DataError, match=r"lexicon\.tsv"):
        _load_lexicon()
