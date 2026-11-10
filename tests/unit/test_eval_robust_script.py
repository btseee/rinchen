"""The script purity metric of D12 (spec T8): the scripts a row's output must be in
(`target_scripts`: the registry's scripts of the target, or those of `mn_Mong` or `khk_Cyrl`
when the row names a script) and the purity of a text in them (`score_script`: the mean's
unit, `null` for a text with no counted letter, the floor inclusive). The vectors are the
plan's and the golden rows s1 to s6; every expectation is a literal. Traditional-script,
Greek, Han and kana letters are built with `chr()`; the placeholder glyphs come from
`spm.symbols`."""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.errors import ConfigError
from rinchen.eval import robust_script
from rinchen.eval.robust_script import (
    PURITY_FLOOR,
    ScriptOutcome,
    score_script,
    target_scripts,
)
from rinchen.infer import script_mask
from rinchen.lang.scripts import Script
from rinchen.spm.symbols import user_defined_symbols

MARKER = "SECRET-MARKER-91"
PLACEHOLDER_0 = user_defined_symbols()[-100]

MONG_E = chr(0x1821)
MONG_N = chr(0x1828)
MONG_B = chr(0x182A)
MONG_O = chr(0x1823)
MONG_L = chr(0x182F)
MONG_M = chr(0x182E)
MONG_FULL_STOP = chr(0x1803)
MONG_ENE = MONG_E + MONG_N + MONG_E
MONG_BOL = MONG_B + MONG_O + MONG_L
MONG_NOM = MONG_N + MONG_O + MONG_M
MONG_TEXT = f"{MONG_ENE} {MONG_BOL} {MONG_NOM}"
MONG_AND_CYRILLIC = f"{MONG_ENE} бол"

GREEK = chr(0x3B1) + chr(0x3B2) + chr(0x3B3)
HAN = chr(0x65E5) + chr(0x672C) + chr(0x8A9E)
HIRAGANA = chr(0x3042) + chr(0x3044)
KATAKANA = chr(0x30A2) + chr(0x30A4)
HANGUL = chr(0xD55C) + chr(0xAE00)
LONG_VOWEL_MARK = chr(0x30FC)
MODIFIER_LETTER_APOSTROPHE = chr(0x2BC)
ACUTE = chr(0x301)

CYRILLIC_9 = "Энэ бол ном"
CYRILLIC_49 = "а" * 49


# --- target_scripts ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tgt_lang", "script", "expected"),
    [
        ("khk_Cyrl", None, {Script.CYRILLIC}),
        ("mn_Mong", None, {Script.MONGOLIAN}),
        ("eng_Latn", None, {Script.LATIN}),
        ("rus_Cyrl", None, {Script.CYRILLIC}),
        ("jpn_Jpan", None, {Script.HAN, Script.HIRAGANA, Script.KATAKANA}),
        ("khk_Cyrl", "Mong", {Script.MONGOLIAN}),
        ("khk_Cyrl", "Cyrl", {Script.CYRILLIC}),
        ("mn_Mong", "Mong", {Script.MONGOLIAN}),
        ("mn_Mong", "Cyrl", {Script.CYRILLIC}),
    ],
    ids=[
        "khalkha",
        "mn_mong",
        "english",
        "russian",
        "japanese",
        "khalkha_as_mong",
        "khalkha_as_cyrl",
        "mn_mong_as_mong",
        "mn_mong_as_cyrl",
    ],
)
def test_the_scripts_a_row_must_write_in(
    tgt_lang: str, script: str | None, expected: set[Script]
) -> None:
    found = target_scripts(tgt_lang, script)

    assert found == frozenset(expected)
    assert isinstance(found, frozenset)


@pytest.mark.parametrize(
    ("tgt_lang", "script"),
    [
        ("eng_Latn", "Mong"),
        ("eng_Latn", "Cyrl"),
        ("rus_Cyrl", "Cyrl"),
        ("jpn_Jpan", "Mong"),
        ("khk_Cyrl", "Latn"),
        ("mn_Mong", "Latn"),
        ("eng_Latn", "Latn"),
        ("khk_Cyrl", "mong"),
        ("khk_Cyrl", "CYRL"),
        ("khk_Cyrl", "Cyrillic"),
        ("khk_Cyrl", "Mongolian"),
        ("khk_Cyrl", ""),
        ("xxx_Latn", "Mong"),
    ],
)
def test_a_script_for_another_target_or_of_another_name_is_a_config_error(
    tgt_lang: str, script: str
) -> None:
    with pytest.raises(ConfigError):
        target_scripts(tgt_lang, script)


def test_a_target_the_registry_does_not_know_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        target_scripts("xxx_Latn", None)


def test_a_refusal_does_not_quote_the_script_it_refuses() -> None:
    with pytest.raises(ConfigError) as raised:
        target_scripts("khk_Cyrl", MARKER)

    assert MARKER not in str(raised.value)
    assert "Mong" in str(raised.value)
    assert "Cyrl" in str(raised.value)


# --- score_script: the vectors of the plan ------------------------------------------------


def purity_case(
    label: str,
    text: str,
    tgt_lang: str,
    script: str | None,
    purity: float,
    below_floor: bool,
) -> object:
    return pytest.param(text, tgt_lang, script, purity, below_floor, id=label)


VECTORS = [
    purity_case("cyrillic_text", "Энэ бол ном.", "khk_Cyrl", None, 1.0, False),
    purity_case(
        "english_text_for_khalkha", "This is a book.", "khk_Cyrl", None, 0.0, True
    ),
    purity_case("nine_of_ten", CYRILLIC_9 + "x", "khk_Cyrl", None, 0.9, True),
    purity_case("floor_49_of_50", CYRILLIC_49 + "x", "khk_Cyrl", None, 0.98, False),
    purity_case("floor_98_of_100", "а" * 98 + "xy", "khk_Cyrl", None, 0.98, False),
    purity_case("under_48_of_50", "а" * 48 + "xy", "khk_Cyrl", None, 0.96, True),
    purity_case("under_97_of_100", "а" * 97 + "xyz", "khk_Cyrl", None, 0.97, True),
    purity_case("greek_only", GREEK, "khk_Cyrl", None, 0.0, True),
    purity_case("greek_and_cyrillic", "Энэ " + GREEK, "khk_Cyrl", None, 0.5, True),
    purity_case("english", "It is fine.", "eng_Latn", None, 1.0, False),
    purity_case("han_for_japanese", HAN, "jpn_Jpan", None, 1.0, False),
    purity_case(
        "kana_for_japanese", HAN + HIRAGANA + KATAKANA, "jpn_Jpan", None, 1.0, False
    ),
    purity_case("hangul_for_japanese", HANGUL, "jpn_Jpan", None, 0.0, True),
    purity_case("han_for_korean", HAN, "kor_Hang", None, 0.0, True),
    purity_case("mong_text_as_mong", MONG_TEXT, "khk_Cyrl", "Mong", 1.0, False),
    purity_case("cyrillic_text_as_mong", "Энэ бол ном.", "khk_Cyrl", "Mong", 0.0, True),
    purity_case(
        "half_mong_half_cyrillic", MONG_AND_CYRILLIC, "khk_Cyrl", "Mong", 0.5, True
    ),
    purity_case("mong_text_for_khalkha", MONG_TEXT, "khk_Cyrl", None, 0.0, True),
    purity_case("mong_text_for_mn_mong", MONG_TEXT, "mn_Mong", None, 1.0, False),
    purity_case(
        "cyrillic_as_cyrl_for_mn_mong", "Энэ бол ном.", "mn_Mong", "Cyrl", 1.0, False
    ),
    purity_case("mong_text_as_cyrl", MONG_TEXT, "mn_Mong", "Cyrl", 0.0, True),
    purity_case("cyrillic_for_mn_mong", "Энэ бол ном.", "mn_Mong", None, 0.0, True),
]


@pytest.mark.parametrize(
    ("text", "tgt_lang", "script", "purity", "below_floor"), VECTORS
)
def test_the_purity_of_a_text_and_whether_it_is_below_the_floor(
    text: str,
    tgt_lang: str,
    script: str | None,
    purity: float,
    below_floor: bool,
) -> None:
    outcome = score_script(text, tgt_lang, script)

    assert outcome == ScriptOutcome(purity=purity, below_floor=below_floor)


@pytest.mark.parametrize(
    ("text", "tgt_lang", "expected"),
    [
        ("а", "khk_Cyrl", ScriptOutcome(1.0, False)),
        ("x", "khk_Cyrl", ScriptOutcome(0.0, True)),
        ("а.", "khk_Cyrl", ScriptOutcome(1.0, False)),
        ("x", "eng_Latn", ScriptOutcome(1.0, False)),
    ],
    ids=["one_cyrillic", "one_latin", "one_cyrillic_and_a_stop", "one_english"],
)
def test_one_counted_letter_is_enough_for_a_purity(
    text: str, tgt_lang: str, expected: ScriptOutcome
) -> None:
    assert score_script(text, tgt_lang) == expected


def test_the_floor_is_the_one_of_the_decode_check_and_a_text_on_it_passes() -> None:
    assert PURITY_FLOOR == 0.98
    assert PURITY_FLOOR is script_mask.PURITY_FLOOR
    on_the_floor = score_script(CYRILLIC_49 + "x", "khk_Cyrl")
    under_the_floor = score_script("а" * 48 + "xy", "khk_Cyrl")

    assert on_the_floor.purity == PURITY_FLOOR
    assert on_the_floor.below_floor is False
    assert under_the_floor.purity is not None
    assert under_the_floor.purity < PURITY_FLOOR
    assert under_the_floor.below_floor is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\t",
        "123",
        "123 " + PLACEHOLDER_0 + " , .",
        PLACEHOLDER_0,
        "..., !?",
        "3,5 %",
        ACUTE,
        LONG_VOWEL_MARK,
        MODIFIER_LETTER_APOSTROPHE,
        MONG_FULL_STOP,
    ],
    ids=[
        "empty",
        "space",
        "newline_tab",
        "digits",
        "digits_placeholder_punctuation",
        "placeholder",
        "punctuation",
        "number_and_percent",
        "combining_mark",
        "long_vowel_mark",
        "modifier_letter",
        "mongolian_full_stop",
    ],
)
def test_a_text_with_no_counted_letter_has_no_purity_and_is_not_below_the_floor(
    text: str,
) -> None:
    for tgt_lang, script in (
        ("khk_Cyrl", None),
        ("khk_Cyrl", "Mong"),
        ("eng_Latn", None),
    ):
        assert score_script(text, tgt_lang, script) == ScriptOutcome(None, None)


def test_digits_punctuation_marks_and_placeholders_around_letters_do_not_count() -> (
    None
):
    text = f"Энэ 3 бол {PLACEHOLDER_0} ном{ACUTE} 12.5!"

    assert score_script(text, "khk_Cyrl") == ScriptOutcome(1.0, False)


def test_a_letter_of_a_script_outside_the_enum_counts_against_the_text() -> None:
    assert score_script("а" + GREEK[0], "khk_Cyrl") == ScriptOutcome(0.5, True)


def test_the_script_of_a_row_is_judged_before_its_text() -> None:
    for text in ("", "Энэ бол ном.", "123"):
        with pytest.raises(ConfigError):
            score_script(text, "eng_Latn", "Mong")
        with pytest.raises(ConfigError):
            score_script(text, "khk_Cyrl", "Latn")
        with pytest.raises(ConfigError):
            score_script(text, "xxx_Latn")


def test_a_bad_row_never_quotes_the_text_it_was_given() -> None:
    text = f"{MARKER} Энэ бол ном."

    for tgt_lang, script in (("eng_Latn", "Mong"), ("khk_Cyrl", "Latn")):
        with pytest.raises(ConfigError) as raised:
            score_script(text, tgt_lang, script)
        assert MARKER not in str(raised.value)


def test_no_script_is_the_same_as_the_target_of_the_row() -> None:
    for text in ("Энэ бол ном.", "This is a book.", MONG_TEXT):
        assert score_script(text, "khk_Cyrl", None) == score_script(text, "khk_Cyrl")
    assert score_script(MONG_TEXT, "khk_Cyrl", "Mong") != score_script(
        MONG_TEXT, "khk_Cyrl"
    )


def test_the_outcome_is_a_frozen_value_without_a_dict() -> None:
    outcome = score_script("Энэ бол ном.", "khk_Cyrl")

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.purity = 0.0  # type: ignore[misc]
    assert not hasattr(outcome, "__dict__")
    assert [field.name for field in dataclasses.fields(ScriptOutcome)] == [
        "purity",
        "below_floor",
    ]


def test_the_module_names_what_it_offers() -> None:
    assert sorted(robust_script.__all__) == [
        "PURITY_FLOOR",
        "ScriptOutcome",
        "score_script",
        "target_scripts",
    ]


# --- the golden rows --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "text", "tgt_lang", "script", "expected"),
    [
        ("s1", "Энэ бол ном.", "khk_Cyrl", None, ScriptOutcome(1.0, False)),
        ("s2", "This is Эн.", "khk_Cyrl", None, ScriptOutcome(0.25, True)),
        ("s3", "", "khk_Cyrl", None, ScriptOutcome(None, None)),
        ("s4", MONG_TEXT, "khk_Cyrl", "Mong", ScriptOutcome(1.0, False)),
        ("s5", MONG_AND_CYRILLIC, "khk_Cyrl", "Mong", ScriptOutcome(0.5, True)),
        ("s6", "It is fine.", "eng_Latn", None, ScriptOutcome(1.0, False)),
    ],
    ids=["s1", "s2", "s3", "s4", "s5", "s6"],
)
def test_the_golden_rows(
    row: str,
    text: str,
    tgt_lang: str,
    script: str | None,
    expected: ScriptOutcome,
) -> None:
    assert score_script(text, tgt_lang, script) == expected
