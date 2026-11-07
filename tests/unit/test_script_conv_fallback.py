"""Phonographic fallback: one letter map plus five contextual rules.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable. The held-out figures at the bottom were measured once,
after the rules were frozen.
"""

from __future__ import annotations

import pytest

from rinchen.script_conv.alphabet import LETTERS
from rinchen.script_conv.eval import cer, word_accuracy
from rinchen.script_conv.fallback import (
    FALLBACK_LETTERS,
    development_split,
    transliterate_fallback,
)
from rinchen.script_conv.validate import validate_traditional
from tests.support.script_reference import (
    all_reference_rows,
    is_development_word,
    valid_reference_pairs,
)

KAZAKH_QA = chr(0x049B)  # a Cyrillic letter Khalkha does not use

# Letters of the script by their Unicode names, as code points.
A, E, I, O, U, OE, UE = 0x1820, 0x1821, 0x1822, 0x1823, 0x1824, 0x1825, 0x1826
NA, BA, QA, GA, MA, LA = 0x1828, 0x182A, 0x182C, 0x182D, 0x182E, 0x182F
SA, SHA, TA, DA, CHA, JA, YA, RA = (
    0x1830,
    0x1831,
    0x1832,
    0x1833,
    0x1834,
    0x1835,
    0x1836,
    0x1837,
)


def _script(*code_points: int) -> str:
    return "".join(chr(code_point) for code_point in code_points)


# Exact-match rate and CER of `transliterate_fallback` on the held-out half of the
# Wiktionary reference. Measured once, after the rules were frozen; the rules were
# developed on the other half only. The test allows 0.02 of slack either way.
HELD_OUT_COUNT = 1022
HELD_OUT_EXACT_MATCH = 0.2319
HELD_OUT_CER = 0.2342
HELD_OUT_SLACK = 0.02


def test_development_split_hashes_the_utf8_word() -> None:
    words = [cyrillic for cyrillic, _ in valid_reference_pairs()]

    assert all(development_split(word) == is_development_word(word) for word in words)
    shares = {development_split(word) for word in words}
    assert shares == {True, False}


def test_fallback_ignores_case() -> None:
    assert transliterate_fallback("МОНГОЛ") == transliterate_fallback("монгол")
    assert transliterate_fallback("Өлзий") == transliterate_fallback("өлзий")


def test_fallback_drops_a_letter_outside_the_map() -> None:
    assert transliterate_fallback("а" + KAZAKH_QA) == transliterate_fallback("а")
    assert transliterate_fallback(KAZAKH_QA) == ""


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # The letter map: в is written ba, й and ь are written i, я/ю/е/ё are ya
        # plus a vowel, ж and з are ja, ц and ч are cha, ъ is dropped.
        ("вайн", _script(BA, A, I, NA)),
        ("минь", _script(MA, I, NA, I)),
        ("баъ", _script(BA, A)),
        ("яс", _script(YA, A, SA)),
        ("юм", _script(YA, U, MA)),
        ("ем", _script(YA, E, MA)),
        ("ёл", _script(YA, O, LA)),
        ("жил", _script(JA, I, LA)),
        ("зам", _script(JA, A, MA)),
        ("цаг", _script(CHA, A, GA)),
        ("чам", _script(CHA, A, MA)),
        ("шон", _script(SHA, O, NA)),
    ],
)
def test_fallback_letter_map(word: str, expected: str) -> None:
    assert transliterate_fallback(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Rule: after the first vowel letter, о is written U and ө is written UE.
        ("толог", _script(TA, O, LA, U, GA)),
        ("өгөл", _script(OE, GA, UE, LA)),
        # Rule: the second of two equal vowel letters is written GA, then the vowel.
        ("баал", _script(BA, A, GA, A, LA)),
        ("сээр", _script(SA, E, GA, E, RA)),
        ("муур", _script(MA, U, GA, U, RA)),
        ("ааа", _script(A, GA, A, A)),
        # Rule: a word-final х takes U in a back word and UE in a front word.
        ("бах", _script(BA, A, QA, U)),
        ("хэх", _script(QA, E, QA, UE)),
        # Rule: a word-final ч or ж takes I.
        ("ач", _script(A, CHA, I)),
        ("муж", _script(MA, U, JA, I)),
        # Rule: a word-final т, ц or з takes A in a back word and E in a front word.
        ("ат", _script(A, TA, A)),
        ("эт", _script(E, TA, E)),
        ("рац", _script(RA, A, CHA, A)),
        ("лаз", _script(LA, A, JA, A)),
    ],
)
def test_fallback_context_rules(word: str, expected: str) -> None:
    assert transliterate_fallback(word) == expected


def test_fallback_final_rules_apply_only_at_the_end_of_the_word() -> None:
    assert transliterate_fallback("ачаа") == _script(A, CHA, A, GA, A)
    assert transliterate_fallback("хах") == _script(QA, A, QA, U)
    assert transliterate_fallback("тат") == _script(TA, A, TA, A)


def test_fallback_of_an_empty_word_is_empty() -> None:
    assert transliterate_fallback("") == ""


def test_fallback_emits_only_letters_and_is_never_empty_for_a_vowel() -> None:
    for word in ("а", "эхэ", "өөр", "ян", "ёс", "ёл"):
        output = transliterate_fallback(word)
        assert output
        assert set(output) <= FALLBACK_LETTERS <= LETTERS, word


def test_fallback_output_over_the_whole_reference_passes_the_validator() -> None:
    for row in all_reference_rows():
        output = transliterate_fallback(row["cyrillic"])
        assert validate_traditional(output) == (), row["cyrillic"]


def test_held_out_reference_accuracy_matches_the_one_time_measurement() -> None:
    held_out = [
        (cyrillic, script)
        for cyrillic, script in valid_reference_pairs()
        if not development_split(cyrillic)
    ]
    assert len(held_out) == HELD_OUT_COUNT
    references = [script for _, script in held_out]
    hypotheses = [transliterate_fallback(cyrillic) for cyrillic, _ in held_out]

    assert word_accuracy(references, hypotheses) == pytest.approx(
        HELD_OUT_EXACT_MATCH, abs=HELD_OUT_SLACK
    )
    assert cer(references, hypotheses) == pytest.approx(
        HELD_OUT_CER, abs=HELD_OUT_SLACK
    )
