"""Property tests for `rinchen.text.normalize` (hypothesis profile: see conftest)."""

from __future__ import annotations

import string

import regex
from hypothesis import given
from hypothesis import strategies as st

from rinchen.config.languages import known_language_codes
from rinchen.text.normalize import normalize

_LATIN_LOOKALIKES = "aeopcxyAEOPCXY"
_LATIN_PLAIN = "bd"
# Cyrillic letters the normalizer never rewrites (it folds Fita U+0472/U+0473
# into U+04E8/U+04E9, so Fita cannot appear in a "left unchanged" token).
_CYRILLIC_STABLE = "\u0430\u043e\u04e8\u04e9\u04ae\u04af\u04b1\u043c\u043d"
_CYRILLIC_FITA = "\u0472"
_PUNCTUATION = ".,!?\u2014"
_SPACES = " \t\n\u00a0\u202f"
_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"
_MONGOLIAN_FVS = "\u180b\u180e\u180f"
_QUOTES = "\u201c\u201d\u2018\u2019\u00ab\u00bb"
_COMBINING_MARKS = "\u0301\u0306\u0308"
_MONGOLIAN_LETTERS = "".join(chr(code) for code in range(0x1820, 0x1830))

_ALPHABET = st.sampled_from(
    _LATIN_LOOKALIKES
    + _LATIN_PLAIN
    + _CYRILLIC_STABLE
    + _CYRILLIC_FITA
    + string.digits
    + _PUNCTUATION
    + _SPACES
    + _ZERO_WIDTH
    + _MONGOLIAN_FVS
    + _QUOTES
    + _COMBINING_MARKS
    + _MONGOLIAN_LETTERS
)
# A base letter, then characters normalization strips, then a combining mark:
# stripping makes the pair adjacent, so only the closing NFC pass composes it.
# Plain character sampling almost never builds this three-part shape.
_MARK_AFTER_STRIPPED_CHARS = st.tuples(
    _ALPHABET,
    st.text(alphabet=st.sampled_from(_ZERO_WIDTH + _MONGOLIAN_FVS), max_size=2),
    st.sampled_from(_COMBINING_MARKS),
).map("".join)
_TEXTS = st.lists(st.one_of(_ALPHABET, _MARK_AFTER_STRIPPED_CHARS)).map("".join)
_LANGUAGES = st.sampled_from(sorted(known_language_codes()))

_MONGOLIAN_BLOCK_LETTER = regex.compile(r"[\u1800-\u18af]")
_LETTER = regex.compile(r"\p{L}")


@st.composite
def _latin_heavy_tokens(draw: st.DrawFn) -> str:
    """A Latin/Cyrillic token whose Cyrillic letters are at most half of all."""
    latin = draw(
        st.lists(st.sampled_from(_LATIN_LOOKALIKES + _LATIN_PLAIN), min_size=1)
    )
    cyrillic = draw(st.lists(st.sampled_from(_CYRILLIC_STABLE), max_size=len(latin)))
    return "".join(draw(st.permutations(latin + cyrillic)))


@given(text=_TEXTS, lang=_LANGUAGES)
def test_idempotent(text: str, lang: str) -> None:
    once = normalize(text, lang)
    assert normalize(once, lang) == once


@given(text=_TEXTS)
def test_mn_mong_letters_never_altered(text: str) -> None:
    def mongolian_letters(value: str) -> list[str]:
        return [
            ch
            for ch in value
            if _LETTER.match(ch) and _MONGOLIAN_BLOCK_LETTER.match(ch)
        ]

    assert mongolian_letters(normalize(text, "mn_Mong")) == mongolian_letters(text)


@given(token=_latin_heavy_tokens())
def test_homoglyph_only_fires_in_cyrillic_majority_tokens(token: str) -> None:
    assert normalize(token, "eng_Latn") == token
