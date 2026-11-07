"""Property tests for the rule baseline over arbitrary Cyrillic and mixed text."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from rinchen.script_conv.alphabet import LETTERS
from rinchen.script_conv.rules import transliterate, transliterate_fallback
from rinchen.script_conv.segments import segment
from rinchen.script_conv.validate import validate_traditional

# The whole Cyrillic block, so letters the fallback does not map are drawn too.
_CYRILLIC_BLOCK = [chr(code_point) for code_point in range(0x0400, 0x0530)]
# The letters of the Mongolian Cyrillic alphabet, drawn more often than the rest.
_MONGOLIAN_CYRILLIC = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюяөү")


@given(st.text(alphabet=st.sampled_from(_MONGOLIAN_CYRILLIC), max_size=20))
def test_fallback_of_a_khalkha_cyrillic_word_is_valid_letters(word: str) -> None:
    output = transliterate_fallback(word)

    assert validate_traditional(output) == ()
    assert set(output) <= LETTERS


@given(st.text(alphabet=st.sampled_from(_CYRILLIC_BLOCK), max_size=20))
def test_fallback_of_any_cyrillic_block_text_is_valid_letters(word: str) -> None:
    output = transliterate_fallback(word)

    assert validate_traditional(output) == ()
    assert set(output) <= LETTERS


@given(st.text(alphabet=st.sampled_from(_MONGOLIAN_CYRILLIC), max_size=20))
def test_fallback_is_the_same_for_upper_and_lower_case(word: str) -> None:
    assert transliterate_fallback(word.upper()) == transliterate_fallback(word)


@given(st.text(max_size=60))
def test_transliterate_never_raises_and_leaves_no_cyrillic_letter(text: str) -> None:
    output = transliterate(text)

    assert not any(piece.kind == "cyrillic" for piece in segment(output))


@given(st.text(alphabet=st.sampled_from(list("abc 019-()!?")), max_size=40))
def test_text_without_cyrillic_or_script_punctuation_is_unchanged(text: str) -> None:
    assert transliterate(text) == text
