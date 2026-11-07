"""Property tests for `transliterate` over text that mixes Cyrillic, Latin, digits,
punctuation and traditional-script characters.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable.
"""

from __future__ import annotations

from hypothesis import event, given
from hypothesis import strategies as st

from rinchen.script_conv.alphabet import CYRILLIC_TO_SCRIPT_PUNCTUATION, NNBSP
from rinchen.script_conv.fallback import FALLBACK_LETTERS
from rinchen.script_conv.rules import transliterate, transliterate_word
from rinchen.script_conv.segments import segment
from rinchen.script_conv.validate import validate_traditional

# Cyrillic capitals and small letters, the Mongolian-only letters (ө, ү, ё and their
# capitals) and one Kazakh letter, so the fallback meets a letter it does not map.
_CYRILLIC = [chr(code_point) for code_point in range(0x0410, 0x0450)] + [
    chr(code_point) for code_point in (0x0401, 0x0451, 0x04E8, 0x04E9, 0x04AE, 0x04AF)
]
_CYRILLIC.append(chr(0x049B))
_ASCII = list("abcXYZ0123456789")
_PUNCTUATION = list(".,.,- /:()")
# Three script letters, a variation selector, a vowel separator and an NNBSP: the
# characters that make a Mongolian run, including its degenerate forms.
_SCRIPT = [chr(code_point) for code_point in (0x1820, 0x1828, 0x182E, 0x180B, 0x180E)]
_SCRIPT.append(NNBSP)

# Cyrillic is drawn three times as often as any of the other groups.
_CHARACTER = st.one_of(
    st.sampled_from(_CYRILLIC),
    st.sampled_from(_CYRILLIC),
    st.sampled_from(_CYRILLIC),
    st.sampled_from(_ASCII),
    st.sampled_from(_PUNCTUATION),
    st.sampled_from(_SCRIPT),
)
_TEXT = st.text(alphabet=_CHARACTER, max_size=40)


def _changes_only_dots_and_commas(before: str, after: str) -> bool:
    return len(before) == len(after) and all(
        old == new or CYRILLIC_TO_SCRIPT_PUNCTUATION.get(old) == new
        for old, new in zip(before, after, strict=True)
    )


@given(_TEXT)
def test_no_cyrillic_letter_survives(text: str) -> None:
    pieces = segment(text)
    event(
        "contains cyrillic"
        if any(p.kind == "cyrillic" for p in pieces)
        else "no cyrillic"
    )

    output = transliterate(text)

    assert not any(piece.kind == "cyrillic" for piece in segment(output))


@given(_TEXT)
def test_a_non_cyrillic_segment_changes_only_in_its_dots_and_commas(text: str) -> None:
    for piece in segment(text):
        if piece.kind == "mongolian":
            assert transliterate(piece.text) == piece.text
        elif piece.kind == "other":
            assert _changes_only_dots_and_commas(piece.text, transliterate(piece.text))


@given(_TEXT)
def test_the_conversion_is_the_concatenation_of_its_segments(text: str) -> None:
    pieces = segment(text)

    assert transliterate(text) == "".join(transliterate(p.text) for p in pieces)


@given(_TEXT)
def test_every_converted_word_is_valid_and_fallback_words_use_fallback_letters(
    text: str,
) -> None:
    for piece in segment(text):
        if piece.kind != "cyrillic":
            continue
        result = transliterate_word(piece.text)

        assert transliterate(piece.text) == result.text
        assert validate_traditional(result.text) == ()
        if result.source == "fallback":
            assert set(result.text) <= FALLBACK_LETTERS
