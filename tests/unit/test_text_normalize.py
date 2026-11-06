"""Unit tests for `rinchen.text.normalize`.

Every string that mixes scripts or contains invisible characters is written
with code-point escapes so the exact input is reviewable; a trailing comment
names the intended word.
"""

from __future__ import annotations

import pytest

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError
from rinchen.text.normalize import normalize

# Mongolian letters U+1820..U+1842 (A through CHI), joined without separators.
_MONGOLIAN_LETTERS = "".join(chr(code) for code in range(0x1820, 0x1843))

_MONGOL_WORD = "\u041c\u043e\u043d\u0433\u043e\u043b"  # Cyrillic "Mongol"


def test_nfc_composes() -> None:
    assert normalize("e\u0301", "eng_Latn") == "\u00e9"


def test_whitespace_collapses_and_strips() -> None:
    assert normalize("  a \t\n b\u00a0\u3000c  ", "eng_Latn") == "a b c"


def test_nnbsp_collapses_outside_mn_mong() -> None:
    assert normalize("a\u202fb", "eng_Latn") == "a b"


def test_nnbsp_preserved_in_mn_mong() -> None:
    s = "\u1820\u202f\u1821"
    assert normalize(s, "mn_Mong") == s


def test_quotes_canonicalized() -> None:
    text = "\u201cHi\u201d \u2018x\u2019 \u00abY\u00bb \u201eZ\u201f"
    assert normalize(text, "eng_Latn") == '"Hi" \'x\' "Y" "Z"'


@pytest.mark.parametrize("quote", "\u201c\u201d\u201e\u201f\u00ab\u00bb")
def test_every_double_quote_variant_becomes_ascii_double_quote(quote: str) -> None:
    assert normalize(f"a{quote}b", "eng_Latn") == 'a"b'


@pytest.mark.parametrize("quote", "\u2018\u2019\u201a\u201b")
def test_every_single_quote_variant_becomes_ascii_apostrophe(quote: str) -> None:
    assert normalize(f"a{quote}b", "eng_Latn") == "a'b"


def test_zero_width_stripped_without_double_space() -> None:
    assert normalize("a \u200b b", "eng_Latn") == "a b"
    assert normalize("a\u200bb\ufeff\u2060", "eng_Latn") == "ab"


def test_fvs_preserved_only_in_mn_mong() -> None:
    s = "\u1820\u180b\u1821"
    assert normalize(s, "mn_Mong") == s
    assert normalize(s, "khk_Cyrl") == "\u1820\u1821"


def test_zwj_and_zwnj_are_stripped_even_in_mn_mong() -> None:
    assert (
        normalize("\u1820\u200d\u1821\u200c\u1822", "mn_Mong") == "\u1820\u1821\u1822"
    )


def test_homoglyph_repairs_latin_inside_cyrillic_token() -> None:
    # Latin "o" at two positions inside Cyrillic "Mongol".
    assert normalize("\u041co\u043d\u0433o\u043b", "khk_Cyrl") == _MONGOL_WORD


def test_homoglyph_uppercase() -> None:
    # Latin "P" followed by Cyrillic: "Rossiya".
    expected = "\u0420\u043e\u0441\u0441\u0438\u044f"
    assert normalize("P\u043e\u0441\u0441\u0438\u044f", "rus_Cyrl") == expected


@pytest.mark.parametrize(
    ("latin", "cyrillic"),
    [
        ("a", "\u0430"),
        ("e", "\u0435"),
        ("o", "\u043e"),
        ("p", "\u0440"),
        ("c", "\u0441"),
        ("x", "\u0445"),
        ("y", "\u0443"),
        ("A", "\u0410"),
        ("E", "\u0415"),
        ("O", "\u041e"),
        ("P", "\u0420"),
        ("C", "\u0421"),
        ("X", "\u0425"),
        ("Y", "\u0423"),
    ],
)
def test_homoglyph_table_maps_each_latin_letter(latin: str, cyrillic: str) -> None:
    # One Cyrillic anchor plus the Latin lookalike: 50 percent, so add a second anchor.
    anchor = "\u043c\u043d"  # Cyrillic "mn"
    assert normalize(f"{anchor}{latin}", "khk_Cyrl") == f"{anchor}{cyrillic}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pop", "pop"),
        ("\u041cop", "\u041cop"),  # 1 of 3 letters Cyrillic
        ("\u041co", "\u041co"),  # exactly 50 percent: unchanged
        ("\u041c\u043eo", "\u041c\u043e\u043e"),  # 2 of 3 Cyrillic: fixed
        ("\u0430\u04301o", "\u0430\u04301\u043e"),  # the digit is not counted
    ],
)
def test_homoglyph_gated_by_majority(text: str, expected: str) -> None:
    assert normalize(text, "khk_Cyrl") == expected


def test_fita_folds_to_o_umlaut_letters() -> None:
    assert normalize("\u0472\u0473", "khk_Cyrl") == "\u04e8\u04e9"


def test_other_u_o_variants_untouched() -> None:
    assert (
        normalize("\u04ae\u04af\u04e8\u04e9", "khk_Cyrl") == "\u04ae\u04af\u04e8\u04e9"
    )
    assert normalize("\u04b0\u04b1", "kaz_Cyrl") == "\u04b0\u04b1"


def test_final_nfc_pass_keeps_idempotence() -> None:
    assert normalize("\u0472\u0308", "khk_Cyrl") == "\u04ea"
    assert normalize("\u0443\u0443y\u0306", "rus_Cyrl") == "\u0443\u0443\u045e"


def test_mn_mong_letters_untouched() -> None:
    assert normalize(_MONGOLIAN_LETTERS, "mn_Mong") == _MONGOLIAN_LETTERS


@pytest.mark.parametrize("lang", sorted(known_language_codes()))
def test_empty_string_stays_empty(lang: str) -> None:
    assert normalize("", lang) == ""


def test_whitespace_only_collapses_by_language() -> None:
    assert normalize(" \u00a0\t ", "eng_Latn") == ""
    assert normalize("\u202f", "khk_Cyrl") == ""
    assert normalize(" \u202f ", "mn_Mong") == "\u202f"


def test_letterless_tokens_do_not_crash() -> None:
    assert normalize("12.5 \u2014 !?", "khk_Cyrl") == "12.5 \u2014 !?"


@pytest.mark.parametrize("lang", ["mn_mong", ""])
def test_unknown_language_raises_config_error(lang: str) -> None:
    with pytest.raises(ConfigError):
        normalize("text", lang)
