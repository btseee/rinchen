"""A tiny supervised fastText language identifier for the filter tests.

It tells the four pseudo-languages of `pseudo_text` apart (`khk_Cyrl`,
`rus_Cyrl`, `eng_Latn`, `deu_Latn`), trains in well under a second from
deterministic text, and is never committed. Besides plain lines the training
text holds the shapes the planted corpus uses: Cyrillic words with rare
letters, English words made of letters the tiny SentencePiece model has never
seen, and lines of one language that quote long words of another.
"""

from __future__ import annotations

import random
from pathlib import Path

import fasttext

from tests.support import pseudo_text as pt

LABELS = ("khk_Cyrl", "rus_Cyrl", "eng_Latn", "deu_Latn")

_SEED = "tiny-lid"
_PLAIN_LINES = 300
_RARE_LINES = 150
_MIXED_LINES = 200
_DIGIT_SHARE = 0.15
_MIXED_NATIVE_WORDS = (5, 9)
_MIXED_FOREIGN_WORDS = (1, 3)

# Measured: with these settings every pseudo-language line of the planted
# corpus scores at least 0.9 for its language in about 0.2 seconds.
_TRAINING_OPTIONS = {
    "lr": 1.0,
    "epoch": 30,
    "dim": 16,
    "minn": 1,
    "maxn": 4,
    "wordNgrams": 1,
    "loss": "softmax",
    "thread": 1,
    "verbose": 0,
}


def train_tiny_lid_model(path: Path) -> Path:
    """Train the model and save it at `path`; the training text is written
    beside it as `<path>.train.txt`. Returns `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    training_path = path.with_name(path.name + ".train.txt")
    training_path.write_text(
        "".join(f"__label__{label} {line}\n" for label, line in _training_lines()),
        encoding="utf-8",
        newline="\n",
    )
    model = fasttext.train_supervised(input=str(training_path), **_TRAINING_OPTIONS)
    model.save_model(str(path))
    return path


def _training_lines() -> list[tuple[str, str]]:
    rng = random.Random(_SEED)
    lines: list[tuple[str, str]] = []

    def add(label: str, texts: list[str]) -> None:
        lines.extend((label, text) for text in texts)

    def plain(syllables: tuple[str, ...], **options: int) -> list[str]:
        return [
            pt.make_line(rng, syllables, digit_share=_DIGIT_SHARE, **options)
            for _ in range(_PLAIN_LINES)
        ]

    def rare(syllables: tuple[str, ...], **options: int) -> list[str]:
        return [pt.make_line(rng, syllables, **options) for _ in range(_RARE_LINES)]

    def mixed(native: tuple[str, ...], foreign: tuple[str, ...]) -> list[str]:
        return [
            pt.make_mixed_line(
                rng,
                native,
                foreign,
                native_words=rng.randint(*_MIXED_NATIVE_WORDS),
                foreign_words=rng.randint(*_MIXED_FOREIGN_WORDS),
            )
            for _ in range(_MIXED_LINES)
        ]

    add("khk_Cyrl", pt.fixture_lines("khk_Cyrl") + plain(pt.KHK_SYLLABLES))
    add("khk_Cyrl", rare(pt.RARE_KHK_SYLLABLES, max_syllables=3, max_words=14))
    add("khk_Cyrl", mixed(pt.KHK_SYLLABLES, pt.ENG_SYLLABLES))
    add("rus_Cyrl", pt.fixture_lines("rus_Cyrl") + plain(pt.RUS_SYLLABLES))
    add("eng_Latn", pt.fixture_lines("eng_Latn") + plain(pt.ENG_SYLLABLES))
    add("eng_Latn", rare(pt.RARE_ENG_SYLLABLES, max_syllables=2))
    add(
        "eng_Latn",
        rare(pt.RARE_ENG_SYLLABLES, max_syllables=1, min_words=5, max_words=30),
    )
    add("eng_Latn", mixed(pt.ENG_SYLLABLES, pt.KHK_SYLLABLES))
    add("deu_Latn", plain(pt.DEU_SYLLABLES, max_syllables=3))
    rng.shuffle(lines)
    return lines
