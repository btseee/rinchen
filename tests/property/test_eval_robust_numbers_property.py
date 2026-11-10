"""Property tests for the numbers metric of D12: whatever text a model writes, extracting its
numbers never raises, is deterministic and yields at most 65 tokens; a text without a digit
character has no number; a text of at most 64 numbers scored against itself is exact, in any
order of its numbers; a score never claims more matches than either side holds."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from rinchen.eval.robust_numbers import (
    MAX_NUMBER_TOKENS,
    NumberToken,
    numbers_in,
    score_numbers,
)
from rinchen.spm.symbols import user_defined_symbols

NBSP = chr(0xA0)
NARROW_NBSP = chr(0x202F)
THIN_SPACE = chr(0x2009)
MINUS = chr(0x2212)
TUGRIK = chr(0x20AE)
FOREIGN_DIGITS = [chr(0x661), chr(0xFF11), chr(0x1811), chr(0x966)]
CJK_NUMERALS = "".join(
    chr(code)
    for code in (
        *(0x3007, 0x96F6, 0x4E00, 0x4E8C, 0x4E09, 0x56DB, 0x4E94, 0x516D),
        *(0x4E03, 0x516B, 0x4E5D, 0x5341, 0x767E, 0x5343, 0x4E07),
    )
)
PLACEHOLDER_GLYPHS = [chr(0x27E6), chr(0x27E7)]

LANGUAGES = st.sampled_from(
    [None, "eng_Latn", "rus_Cyrl", "khk_Cyrl", "mn_Mong", "deu_Latn", "zho_Hans"]
)
NUMERIC_PIECES = st.sampled_from(
    [*"0123456789" * 3, *" .,:/-()%$x", NBSP, NARROW_NBSP, THIN_SPACE, MINUS, TUGRIK]
    + FOREIGN_DIGITS
    + list(CJK_NUMERALS)
    + PLACEHOLDER_GLYPHS
    + [user_defined_symbols()[-1], user_defined_symbols()[-100]]
)
NUMERIC_TEXT = st.lists(NUMERIC_PIECES, max_size=70).map("".join)
NO_DIGIT_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=("Nd", "Cs"), blacklist_characters=CJK_NUMERALS
    ),
    max_size=70,
)
ANY_TEXT = st.one_of(st.text(max_size=70), NUMERIC_TEXT)
NUMBER_ITEMS = [
    "1,250.50",
    "3,5",
    "12.03.2024",
    "10:30",
    "-5",
    "7",
    "1 250",
    "1,250",
    "15%",
    TUGRIK + "1500",
    "20 $",
    "0,500",
    "2024-03-12",
]


def parse(text: str, lang: str | None) -> tuple[NumberToken, ...]:
    return numbers_in(text, lang)


@given(ANY_TEXT, LANGUAGES)
def test_numbers_in_never_raises_and_returns_at_most_sixty_five_tokens(
    text: str, lang: str | None
) -> None:
    tokens = parse(text, lang)

    assert len(tokens) <= MAX_NUMBER_TOKENS + 1
    assert all(token.readings for token in tokens)
    assert all(token.surface for token in tokens)


@given(ANY_TEXT, LANGUAGES)
def test_numbers_in_is_deterministic(text: str, lang: str | None) -> None:
    assert parse(text, lang) == parse(text, lang)


@given(NO_DIGIT_TEXT, LANGUAGES)
def test_a_text_without_a_digit_character_has_no_number(
    text: str, lang: str | None
) -> None:
    assert parse(text, lang) == ()


@given(ANY_TEXT, LANGUAGES)
def test_a_text_scored_against_itself_is_exact_when_it_holds_at_most_sixty_four_numbers(
    text: str, lang: str | None
) -> None:
    count = len(parse(text, lang))

    outcome = score_numbers(text, text, tgt_lang=lang)

    fits = count <= MAX_NUMBER_TOKENS
    assert outcome.exact is fits
    assert outcome.surface_exact is fits
    assert outcome.units_ok
    assert outcome.found == outcome.expected == min(count, MAX_NUMBER_TOKENS)
    assert outcome.extra == count - outcome.found
    assert outcome.capped is (not fits)


@given(st.lists(st.sampled_from(NUMBER_ITEMS), max_size=20), LANGUAGES, st.data())
def test_the_order_of_the_numbers_never_matters(
    items: list[str], lang: str | None, data: st.DataObject
) -> None:
    shuffled = data.draw(st.permutations(items))
    reference = " x ".join(items)
    hypothesis = " x ".join(shuffled)

    outcome = score_numbers(reference, hypothesis, tgt_lang=lang)

    assert outcome.exact
    assert outcome.surface_exact


@given(ANY_TEXT, ANY_TEXT, LANGUAGES)
def test_a_score_never_raises_and_counts_within_both_sides(
    reference: str, hypothesis: str, lang: str | None
) -> None:
    wanted = len(parse(reference, lang))
    seen = len(parse(hypothesis, lang))

    outcome = score_numbers(reference, hypothesis, tgt_lang=lang)

    assert outcome.expected == min(wanted, MAX_NUMBER_TOKENS)
    assert outcome.found <= min(outcome.expected, MAX_NUMBER_TOKENS)
    assert outcome.found + outcome.extra == seen
    assert outcome.exact == (
        outcome.found == outcome.expected == seen and outcome.units_ok
    )
