"""Property tests for the span matching of D12: a word between spaces is found and the same
word glued to letters on both sides is not; whatever a model writes, asking for a spelling
never raises; and a text asked about in its decomposed, composed, upper-case or
case-folded form gives the answer of the text itself."""

from __future__ import annotations

import unicodedata

from hypothesis import example, given
from hypothesis import strategies as st

from rinchen.eval.robust_spans import span_present

SPELLINGS = ("Улаанбаатар", "Ulan Bator", "Бат", "Дорж", "Сайхан")
# The filler letters of the glued text are not in the alphabet of the words, so no word
# can begin at the filler.
WORD_LETTERS = st.characters(categories=("Ll", "Lu"), exclude_characters="xyXY")
WORDS = st.text(alphabet=WORD_LETTERS, min_size=1, max_size=12)
FRAGMENTS = st.sampled_from(
    [
        *SPELLINGS,
        "Улаанбаатарт",
        "Улаанбаатарынхаа",
        "Батор",
        "ULAN BATOR",
        "ulan   bator",
        " ",
        "  ",
        ".",
        "'s",
        "2",
        "аас",
        "-",
        "й",
        "и" + chr(0x306),
        "ө",
        "Ө",
    ]
)
TEXTS = st.one_of(
    st.lists(FRAGMENTS, max_size=12).map("".join),
    st.text(max_size=40),
    st.text(alphabet=st.characters(exclude_categories=()), max_size=40),
)
SAFE_FOR_CASING = st.lists(FRAGMENTS, max_size=12).map("".join)


@given(WORDS)
def test_a_word_between_spaces_is_found_and_the_same_word_glued_is_not(
    word: str,
) -> None:
    assert span_present(f"x {word} y", [word])
    assert not span_present(f"x{word}y", [word])


@given(st.one_of(TEXTS, st.text(alphabet=st.characters(exclude_categories=()))))
@example(chr(0xD800))
@example(chr(0xDFFF) + "Бат" + chr(0xD83D))
@example("")
def test_asking_for_a_spelling_never_raises_and_answers_a_bool(text: str) -> None:
    for spellings in ([SPELLINGS[0]], list(SPELLINGS)):
        assert span_present(text, spellings) in (True, False)


@given(SAFE_FOR_CASING)
def test_the_answer_does_not_depend_on_the_case_of_the_text(text: str) -> None:
    answer = span_present(text, SPELLINGS)

    assert span_present(text.upper(), SPELLINGS) is answer
    assert span_present(text.lower(), SPELLINGS) is answer
    assert span_present(text.casefold(), SPELLINGS) is answer


@given(TEXTS)
def test_the_answer_does_not_depend_on_the_normalization_form_of_the_text(
    text: str,
) -> None:
    answer = span_present(text, SPELLINGS)

    assert span_present(unicodedata.normalize("NFD", text), SPELLINGS) is answer
    assert span_present(unicodedata.normalize("NFC", text), SPELLINGS) is answer


@given(SAFE_FOR_CASING)
def test_the_answer_does_not_depend_on_the_form_of_the_spelling(text: str) -> None:
    decomposed = [unicodedata.normalize("NFD", spelling) for spelling in SPELLINGS]
    upper = [spelling.upper() for spelling in SPELLINGS]

    assert span_present(text, decomposed) is span_present(text, SPELLINGS)
    assert span_present(text, upper) is span_present(text, SPELLINGS)


@given(
    st.text(alphabet="вгдж", min_size=1, max_size=8),
    st.integers(min_value=0, max_value=6),
    st.text(alphabet="аб", max_size=8),
)
def test_the_suffix_allowance_decides_how_many_letters_may_follow(
    word: str, allowance: int, suffix: str
) -> None:
    # The letters of the suffix are not those of the word, so the word occurs once.
    text = f"{word}{suffix} z"

    assert span_present(text, [word], max_suffix=allowance) is (
        len(suffix) <= allowance
    )
