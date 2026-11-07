"""Deterministic pseudo-text for the D4 test support.

Words are random concatenations of syllables from a per-language inventory, in
the style of the tiny SentencePiece fixture (`scripts/build_spm_fixture.py`):
they carry no meaning. The planted corpus and the tiny LID model's training
text both draw from these inventories, so the model has seen the kind of text
the corpus plants. It has not seen the 22 generated planted rows; the 28 clean
rows are fixture lines, which its training text holds as well.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from rinchen.config.loader import REPO_ROOT

TINY_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"


def _consonant_vowel(consonants: str, vowels: str) -> tuple[str, ...]:
    return tuple(consonant + vowel for consonant in consonants for vowel in vowels)


def _german_syllables() -> tuple[str, ...]:
    onsets = ("b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "w")
    clusters = ("sch", "ch", "tz", "pf", "kr", "gr", "tr", "str", "schl")
    vowels = ("a", "e", "i", "o", "u", "ä", "ö", "ü")
    codas = ("", "", "n", "r", "ng", "ck", "ß", "st", "ch", "t")
    return tuple(
        onset + vowel + coda
        for onset in (*onsets, *clusters)
        for vowel in vowels
        for coda in codas
    )


KHK_SYLLABLES = _consonant_vowel("бвгдзклмнпрстхцчшй", "аэиоуөүя")
RUS_SYLLABLES = _consonant_vowel("бвгджзклмнпрстфхцчшщ", "аэиоуыяеёю")
ENG_SYLLABLES = _consonant_vowel("bcdfghjklmnprstvwz", "aeiou")
# Letters the tiny SentencePiece model never saw: each one falls back to a
# byte piece, so words made of them fragment into one token per character.
RARE_ENG_SYLLABLES = _consonant_vowel("qxy", "aeiou")
# The same for Cyrillic: the soft sign is in no fixture language.
RARE_KHK_SYLLABLES = _consonant_vowel("бвгдзклмнпрстхцчшй", "ь")
DEU_SYLLABLES = _german_syllables()

DIGIT_TOKENS = ("2024", "3.5")


def make_word(
    rng: random.Random,
    syllables: Sequence[str],
    *,
    min_syllables: int = 1,
    max_syllables: int = 4,
) -> str:
    count = rng.randint(min_syllables, max_syllables)
    return "".join(rng.choice(syllables) for _ in range(count))


def make_words(
    rng: random.Random,
    syllables: Sequence[str],
    count: int,
    *,
    min_syllables: int = 1,
    max_syllables: int = 4,
) -> list[str]:
    return [
        make_word(
            rng, syllables, min_syllables=min_syllables, max_syllables=max_syllables
        )
        for _ in range(count)
    ]


def make_line(
    rng: random.Random,
    syllables: Sequence[str],
    *,
    min_words: int = 4,
    max_words: int = 10,
    min_syllables: int = 1,
    max_syllables: int = 4,
    digit_share: float = 0.0,
) -> str:
    """A line of 4-10 words ending in a full stop; with probability
    `digit_share` one number token is inserted somewhere in it."""
    words = make_words(
        rng,
        syllables,
        rng.randint(min_words, max_words),
        min_syllables=min_syllables,
        max_syllables=max_syllables,
    )
    if rng.random() < digit_share:
        words.insert(rng.randint(0, len(words)), rng.choice(DIGIT_TOKENS))
    return " ".join(words) + "."


def make_line_of_length(
    rng: random.Random,
    syllables: Sequence[str],
    target: int,
    *,
    min_syllables: int = 1,
    max_syllables: int = 3,
) -> str:
    """A line ending in a full stop whose non-space length reaches `target`
    with the fewest words, so it overshoots by less than one word."""
    words: list[str] = []
    length = 1  # the full stop
    while length < target:
        word = make_word(
            rng, syllables, min_syllables=min_syllables, max_syllables=max_syllables
        )
        words.append(word)
        length += len(word)
    return " ".join(words) + "."


def make_mixed_line(
    rng: random.Random,
    native: Sequence[str],
    foreign: Sequence[str],
    *,
    native_words: int,
    foreign_words: int,
) -> str:
    """Short native words with a few long foreign ones, as a line quoting
    brand names or URLs does. The foreign words carry roughly 40% of the
    letters but a minority of the words, so a language identifier still
    follows the native language."""
    words = make_words(rng, native, native_words, min_syllables=1, max_syllables=2)
    words += make_words(rng, foreign, foreign_words, min_syllables=4, max_syllables=4)
    rng.shuffle(words)
    return " ".join(words) + "."


def non_space_length(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def fixture_lines(code: str) -> list[str]:
    """The lines of one language of the tiny SentencePiece corpus fixture."""
    path: Path = TINY_CORPUS_DIR / f"{code}.txt"
    return path.read_text(encoding="utf-8").splitlines()
