from __future__ import annotations

import pytest

from rinchen.lang.scripts import Script, script_of


@pytest.mark.parametrize(
    ("script", "text"),
    [
        (Script.CYRILLIC, "Монгол"),
        (Script.MONGOLIAN, "ᠠᠡᠢ"),
        (Script.LATIN, "Mongol"),
        (Script.HAN, "中文"),
        (Script.HANGUL, "한국어"),
        (Script.HIRAGANA, "ひらがな"),
        (Script.KATAKANA, "カタカナ"),
    ],
)
def test_pure_script_is_ratio_one(script: Script, text: str) -> None:
    assert script_of(text) == {script: 1.0}


def test_mixed_text_gives_proportions() -> None:
    assert script_of("abМо") == {Script.LATIN: 0.5, Script.CYRILLIC: 0.5}
    assert script_of("日本語のカタカナ") == {
        Script.HAN: 0.375,
        Script.HIRAGANA: 0.125,
        Script.KATAKANA: 0.5,
    }


def test_non_letters_are_ignored() -> None:
    assert script_of("12 ,.— Мо") == {Script.CYRILLIC: 1.0}
    # FVS (U+180B), NNBSP (U+202F) and the Mongolian full stop (U+1803) are not letters.
    assert script_of("ᠠ᠋ ᠃") == {Script.MONGOLIAN: 1.0}


@pytest.mark.parametrize("text", ["", "   ", "123 !?"])
def test_letterless_text_returns_empty(text: str) -> None:
    assert script_of(text) == {}


def test_letters_of_unlisted_scripts_count_in_denominator() -> None:
    assert script_of("Мо" + "αβ") == {Script.CYRILLIC: 0.5}


KATAKANA_LONG_VOWEL = chr(0x30FC)
MODIFIER_LETTER_APOSTROPHE = chr(0x02BC)


def test_common_script_letters_are_neutral_in_numerator_and_denominator() -> None:
    coffee = chr(0x30B3) + KATAKANA_LONG_VOWEL + chr(0x30D2) + KATAKANA_LONG_VOWEL

    assert script_of(coffee) == {Script.KATAKANA: 1.0}
    assert script_of("Mongol" + MODIFIER_LETTER_APOSTROPHE) == {Script.LATIN: 1.0}


def test_text_of_only_common_script_letters_returns_empty() -> None:
    assert script_of(KATAKANA_LONG_VOWEL * 3 + MODIFIER_LETTER_APOSTROPHE) == {}


def test_neutral_letters_do_not_dilute_a_mixed_text() -> None:
    assert script_of("aМ" + KATAKANA_LONG_VOWEL * 4 + "αβ") == {
        Script.LATIN: 0.25,
        Script.CYRILLIC: 0.25,
    }
