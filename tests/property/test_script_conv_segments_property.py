"""Property tests for `segment` over arbitrary and script-mixed text."""

from __future__ import annotations

from itertools import pairwise

import regex
from hypothesis import given
from hypothesis import strategies as st

from rinchen.script_conv.alphabet import FVS, JOINERS, LETTERS, MVS, NIRUGU, NNBSP
from rinchen.script_conv.segments import segment

_RUN_CHARACTERS = LETTERS | FVS | JOINERS | {MVS, NIRUGU}
_CYRILLIC_LETTER = regex.compile(r"(?=\p{L})\p{Script=Cyrillic}")
_COMBINING_MARK = regex.compile(r"\p{M}")

# The Cyrillic, Mongolian and ASCII blocks, plus the characters the segmenter
# treats specially: NNBSP, the joiners and a combining mark.
_BLOCK_TEXT = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x0400, max_codepoint=0x04FF),
        st.characters(min_codepoint=0x1800, max_codepoint=0x18AF),
        st.characters(min_codepoint=0x20, max_codepoint=0x7E),
        st.sampled_from([NNBSP, *JOINERS, chr(0x0301)]),
    )
)
# Cyrillic letters beside the selectors, which are category Mn like a combining mark.
_CYRILLIC_WITH_SELECTORS = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x0430, max_codepoint=0x044F),
        st.sampled_from([*sorted(FVS), chr(0x0301)]),
    )
)
_LETTER_WORD = st.text(alphabet=st.sampled_from(sorted(LETTERS)), min_size=1)


def _joined(text: str) -> str:
    return "".join(piece.text for piece in segment(text))


@given(st.text())
def test_concatenating_the_segments_reproduces_any_text(text: str) -> None:
    assert _joined(text) == text


@given(st.text(alphabet=st.characters(exclude_categories=())))
def test_concatenating_the_segments_reproduces_text_with_surrogates(text: str) -> None:
    assert _joined(text) == text


@given(_BLOCK_TEXT)
def test_concatenating_the_segments_reproduces_script_mixed_text(text: str) -> None:
    assert _joined(text) == text


@given(_BLOCK_TEXT)
def test_segments_are_never_empty_and_never_repeat_a_kind_next_to_each_other(
    text: str,
) -> None:
    pieces = segment(text)
    assert all(piece.text for piece in pieces)
    assert all(left.kind != right.kind for left, right in pairwise(pieces))


@given(st.one_of(_BLOCK_TEXT, _CYRILLIC_WITH_SELECTORS))
def test_a_cyrillic_segment_is_letters_with_their_marks(text: str) -> None:
    for piece in segment(text):
        if piece.kind == "cyrillic":
            assert _CYRILLIC_LETTER.match(piece.text[0])
            assert not set(piece.text) & _RUN_CHARACTERS
            assert all(
                _CYRILLIC_LETTER.match(char) or _COMBINING_MARK.match(char)
                for char in piece.text
            )


@given(_BLOCK_TEXT)
def test_a_mongolian_segment_has_only_run_characters_and_inner_nnbsp(
    text: str,
) -> None:
    for piece in segment(text):
        if piece.kind != "mongolian":
            continue
        assert piece.text[0] in _RUN_CHARACTERS
        assert piece.text[-1] in _RUN_CHARACTERS
        for index, char in enumerate(piece.text):
            if char == NNBSP:
                assert piece.text[index - 1] in _RUN_CHARACTERS
                assert piece.text[index + 1] in LETTERS
            else:
                assert char in _RUN_CHARACTERS


@given(_BLOCK_TEXT)
def test_an_other_segment_holds_no_cyrillic_letter_and_no_run_character(
    text: str,
) -> None:
    for piece in segment(text):
        if piece.kind == "other":
            assert not set(piece.text) & _RUN_CHARACTERS
            assert not any(_CYRILLIC_LETTER.match(char) for char in piece.text)


@given(st.lists(_LETTER_WORD, min_size=1, max_size=4))
def test_letter_words_joined_by_nnbsp_are_one_mongolian_segment(
    words: list[str],
) -> None:
    text = NNBSP.join(words)
    assert [(piece.text, piece.kind) for piece in segment(text)] == [
        (text, "mongolian")
    ]


@given(st.lists(_LETTER_WORD, min_size=2, max_size=4))
def test_letter_words_joined_by_a_space_are_split_at_the_space(
    words: list[str],
) -> None:
    mongolian = [
        piece.text for piece in segment(" ".join(words)) if piece.kind != "other"
    ]
    assert mongolian == words
