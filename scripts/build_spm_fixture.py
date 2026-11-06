# scripts/build_spm_fixture.py
"""One-time generator for tests/fixtures/spm/tiny_corpus/*.txt -- the tiny
synthetic corpus the D3 SentencePiece tests train on (1000-piece vocabulary).

The texts are deliberately SYNTHETIC pseudo-words (consonant+vowel syllables),
not real sentences: they carry no meaning, no real-world text and no
transliterated or Inner Mongolian material. They only have to be varied enough
for SentencePiece's unigram trainer to reach its HARD vocab limit of 1000
pieces. Re-run this script if the fixture ever needs regenerating; tests in
tests/unit/test_spm_fixture.py compare the committed files to this output.

Determinism: every language draws from its own random.Random stream, seeded
with the string f"{SEED}-{code}" (e.g. "1234-khk_Cyrl"). A string seed is
hashed with SHA-512, so the streams are independent of PYTHONHASHSEED and
adding or reordering languages never shifts another language's text.

Mongolian syllables are built from code points (chr) so this source stays
ASCII apart from the Cyrillic and Latin inventories below."""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SEED = 1234
MIN_SYLLABLES_PER_WORD = 1
MAX_SYLLABLES_PER_WORD = 4
MIN_WORDS_PER_LINE = 4
MAX_WORDS_PER_LINE = 10
DIGIT_LINE_SHARE = 0.15
DIGIT_TOKENS = ("2024", "3.5")

MONGOLIAN_TERMINATOR = chr(0x1803)  # MONGOLIAN FULL STOP


def _cv_syllables(consonants: str, vowels: str) -> list[str]:
    return [consonant + vowel for consonant in consonants for vowel in vowels]


def _mongolian_syllables() -> list[str]:
    first_letters = range(0x1820, 0x1842, 3)
    second_letters = (0x1820, 0x1822, 0x1824, 0x1826)
    return [chr(a) + chr(b) for a in first_letters for b in second_letters]


@dataclass(frozen=True, slots=True)
class _LanguageSpec:
    code: str
    syllables: Sequence[str]
    terminator: str
    line_count: int
    has_digits: bool


SPECS = (
    _LanguageSpec(
        code="khk_Cyrl",
        syllables=_cv_syllables("бвгдзклмнпрстхцчшй", "аэиоуөүя"),
        terminator=".",
        line_count=400,
        has_digits=True,
    ),
    _LanguageSpec(
        code="rus_Cyrl",
        syllables=_cv_syllables("бвгджзклмнпрстфхцчшщ", "аэиоуыяеёю"),
        terminator=".",
        line_count=200,
        has_digits=True,
    ),
    _LanguageSpec(
        code="eng_Latn",
        syllables=_cv_syllables("bcdfghjklmnprstvwz", "aeiou"),
        terminator=".",
        line_count=100,
        has_digits=True,
    ),
    _LanguageSpec(
        code="mn_Mong",
        syllables=_mongolian_syllables(),
        terminator=MONGOLIAN_TERMINATOR,
        line_count=100,
        has_digits=False,
    ),
)


def _word(rng: random.Random, syllables: Sequence[str]) -> str:
    count = rng.randint(MIN_SYLLABLES_PER_WORD, MAX_SYLLABLES_PER_WORD)
    return "".join(rng.choice(syllables) for _ in range(count))


def _line(rng: random.Random, spec: _LanguageSpec) -> str:
    word_count = rng.randint(MIN_WORDS_PER_LINE, MAX_WORDS_PER_LINE)
    tokens = [_word(rng, spec.syllables) for _ in range(word_count)]
    if spec.has_digits and rng.random() < DIGIT_LINE_SHARE:
        tokens.insert(rng.randint(0, len(tokens)), rng.choice(DIGIT_TOKENS))
    return " ".join(tokens) + spec.terminator


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        rng = random.Random(f"{SEED}-{spec.code}")
        lines = [_line(rng, spec) for _ in range(spec.line_count)]
        with (out_dir / f"{spec.code}.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.writelines(line + "\n" for line in lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "tests" / "fixtures" / "spm" / "tiny_corpus"
    build(out_dir)
    print(f"wrote {len(SPECS)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    raise SystemExit(main())
