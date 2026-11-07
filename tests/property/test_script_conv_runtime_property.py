"""Property tests for `ScriptConverter` over arbitrary and script-mixed text.

The translator is a stub that answers every word with one fixed valid script word,
so the model is never wrong and what is left to check is the converter's own
handling of the text around the words.
"""

from __future__ import annotations

import regex
from hypothesis import given
from hypothesis import strategies as st

from rinchen.script_conv.alphabet import (
    FVS,
    JOINERS,
    LETTERS,
    MVS,
    NIRUGU,
    NNBSP,
    SCRIPT_TO_CYRILLIC_PUNCTUATION,
)
from rinchen.script_conv.runtime import ScriptConverter
from rinchen.script_conv.segments import Segment, segment
from tests.support.stub_runtime import StubProcessor, StubTranslator

FIXED_SCRIPT_WORD = chr(0x1820) + chr(0x1828)
FIXED_CYRILLIC_WORD = "ан"
_RUN_CHARACTERS = LETTERS | FVS | JOINERS | {MVS, NIRUGU}
_CYRILLIC_LETTER = regex.compile(r"(?=\p{L})\p{Script=Cyrillic}")
_MARKS_TO_ASCII = str.maketrans(dict(SCRIPT_TO_CYRILLIC_PUNCTUATION))

# Cyrillic and script letters with the characters that decide where a word ends:
# NNBSP, the selectors, a joiner, ASCII and the two script marks.
_CHARACTER_TEXT = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x0430, max_codepoint=0x044F),
        st.sampled_from(sorted(LETTERS)[:6]),
        st.sampled_from([NNBSP, MVS, *sorted(FVS), *JOINERS, chr(0x1802), chr(0x1803)]),
        st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    ),
    max_size=40,
)

# Whole words and the stretches between them, glued in any order, so that a script
# word with a suffix gap and a word beside a number or a mark come up often.
_SCRIPT_WORD = chr(0x182E) + chr(0x1823) + chr(0x1829)
_PIECES = [
    "монгол",
    "Улс",
    _SCRIPT_WORD,
    _SCRIPT_WORD + NNBSP + chr(0x1824) + chr(0x1828),
    _SCRIPT_WORD + chr(0x180B) + NNBSP + chr(0x1824),
    " ",
    "\n",
    NNBSP,
    ".",
    ",",
    "3.5",
    "a.b",
    "-",
    chr(0x1802),
    chr(0x1803),
    chr(0x180B),
]
_WORD_TEXT = st.lists(st.sampled_from(_PIECES), max_size=12).map("".join)

_MIXED_TEXT = st.one_of(_CHARACTER_TEXT, _WORD_TEXT)


def _script_converter() -> ScriptConverter:
    translator = StubTranslator(lambda word: FIXED_SCRIPT_WORD)
    return ScriptConverter.from_parts(translator, StubProcessor())


def _cyrillic_converter() -> ScriptConverter:
    translator = StubTranslator(lambda word: FIXED_CYRILLIC_WORD)
    return ScriptConverter.from_parts(translator, StubProcessor())


def _is_cyrillic_letter(char: str) -> bool:
    return _CYRILLIC_LETTER.match(char) is not None


def _is_script_word(piece: Segment) -> bool:
    # A run of selectors with no letter is not a word and passes through.
    return piece.kind == "mongolian" and not LETTERS.isdisjoint(piece.text)


def _marks_as_ascii(text: str) -> str:
    return text.translate(_MARKS_TO_ASCII)


@given(st.text())
def test_convert_never_raises_for_any_text(text: str) -> None:
    for target in ("Mong", "Cyrl"):
        _script_converter().convert(text, target)  # type: ignore[arg-type]


@given(st.text(alphabet=st.characters(exclude_categories=())))
def test_convert_never_raises_for_text_with_surrogates(text: str) -> None:
    for target in ("Mong", "Cyrl"):
        _script_converter().convert(text, target)  # type: ignore[arg-type]


@given(_MIXED_TEXT)
def test_convert_never_raises_for_text_of_both_scripts_and_ascii(text: str) -> None:
    for target in ("Mong", "Cyrl"):
        _script_converter().convert(text, target)  # type: ignore[arg-type]


@given(_MIXED_TEXT)
def test_no_cyrillic_letter_is_left_in_text_converted_to_script(text: str) -> None:
    converted = _script_converter().convert(text, "Mong").text

    assert not any(_is_cyrillic_letter(char) for char in converted)


@given(_MIXED_TEXT)
def test_no_script_letter_is_left_in_text_converted_to_cyrillic(text: str) -> None:
    converted = _cyrillic_converter().convert(text, "Cyrl").text

    assert not LETTERS & set(converted)


@given(_MIXED_TEXT)
def test_text_around_the_words_is_kept_apart_from_the_two_marks_to_script(
    text: str,
) -> None:
    # What is neither a script letter, selector or joiner nor part of a Cyrillic word
    # is the text around the words. It keeps its order, and only "." and "," may
    # change, so both sides are compared with the script marks written as ASCII.
    converted = _script_converter().convert(text, "Mong").text

    around_input = "".join(
        char
        for piece in segment(text)
        if piece.kind != "cyrillic"
        for char in piece.text
        if char not in _RUN_CHARACTERS
    )
    around_output = "".join(char for char in converted if char not in _RUN_CHARACTERS)
    assert _marks_as_ascii(around_output) == _marks_as_ascii(around_input)


@given(_MIXED_TEXT)
def test_text_around_the_words_is_kept_apart_from_the_two_marks_to_cyrillic(
    text: str,
) -> None:
    # The Cyrillic letters of the output are the input's own words and the converted
    # ones; without them, what is left is the text around the script words, with the
    # same freedom for the marks as above.
    converted = _cyrillic_converter().convert(text, "Cyrl").text

    around_input = "".join(
        char
        for piece in segment(text)
        if not _is_script_word(piece)
        for char in piece.text
        if not _is_cyrillic_letter(char)
    )
    around_output = "".join(char for char in converted if not _is_cyrillic_letter(char))
    assert _marks_as_ascii(around_output) == _marks_as_ascii(around_input)
