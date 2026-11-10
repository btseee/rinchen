"""The robustness toy of D12 (spec T17): a deterministic corpus of number, placeholder, entity
and traditional-script sentences over the D7 nouns and templates, small enough that a tiny model
trained on it for 190 seconds can be measured by the robustness suites.

A PORT of the design spike's generator (`rtoy.py`): `build_robust_toy(7)` writes the same 29,352
rows in the same order. The random draws are the whole contract. Every draw comes from ONE
`random.Random(seed)`, in the order of `_draw_items` (the base sentences, the seven kinds of
number, the two-number sentences, the placeholder sentences, the entity sentences) and, inside a
draw, in the order its function makes them; reordering two calls or two arguments changes every
row after it. `robust_toy_digest` is the SHA-256 of the rows (`4136d80a...bad1` for seed 7, pinned
by `tests/unit/test_support_robust_toy.py`), so a change of the generator, of the D7 nouns and
templates or of D5's lexicon (the traditional-script strings are `rules.transliterate` of the
Khalkha sentences) fails loudly instead of moving every number of the gate.

An item is one content in three languages (English, Khalkha, Russian). Every item makes the four
pair rows `eng-khk`, `khk-eng`, `rus-khk`, `khk-rus` and, for a base, placeholder or entity item,
two rows with the Khalkha side in traditional script (`mn_Mong` to English and to Russian, origin
`converted`). The rows come train, then dev, then test, in the order the items were drawn. No item
is drawn twice, and a test pins that no sentence of one split is in another. The numbered
placeholders are the ten indices 0 to 9 (a model copies an index only if it was trained on it).
At seed 7 the training split takes every sentence of the single-noun templates and of the
placeholder templates with one or two placeholders (the genitive one among them: the pool is smaller
than the draw), so the dev and test splits hold only pair and triple base sentences and
three-placeholder sentences.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

from rinchen.config.spm import SpmConfig
from rinchen.script_conv.rules import transliterate
from rinchen.spm.train import train_spm
from rinchen.train.mixture import load_dev_rows, load_training_rows
from tests.support.robust_toy_text import (
    CITY_IDS,
    ENG,
    ENTITIES,
    GENITIVE_TEMPLATE,
    KHK,
    NUMBER_KINDS,
    PERSON_IDS,
    RUS,
    NumberValue,
    entity_sentence,
    noun_sentence,
    number_sentence,
    placeholder_sentence,
    two_numbers_sentence,
)
from tests.support.tiny_parallel import LANGUAGES, NOUNS, write_corpus_parquet
from tests.support.tiny_variant import DEV_LINES_PER_DIRECTION, ToyInputs

assert sys.version_info[:2] == (3, 14), sys.version

__all__ = [
    "ENTITIES",
    "GENITIVE_TEMPLATE",
    "ROBUST_SEED",
    "ROBUST_SPM_VOCAB_SIZE",
    "NumberValue",
    "RobustRow",
    "RobustToy",
    "build_robust_toy",
    "robust_toy_digest",
    "robust_toy_inputs",
    "train_robust_spm",
]

ROBUST_SEED: Final = 7
# `train_spm` accepts 600 to 1,100 pieces for this text and fails at 1,400; the 118 reserved
# symbols, the 256 byte pieces and the 3 control symbols are inside the 800.
ROBUST_SPM_VOCAB_SIZE: Final = 800

_SPLITS: Final = ("train", "dev", "test")
_SPM_LANGUAGES: Final = (ENG, KHK, RUS, "mn_Mong")
_PAIRS: Final = ((ENG, KHK), (RUS, KHK))
_MONG_KINDS: Final = ("base", "placeholder", "entity")
# The number of nouns (or placeholders) each sentence template takes: the six D7 templates and
# the genitive one.
_NOUNS_IN: Final = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 2}
_BASE_TEMPLATES: Final = (1, 2, 3, 4, 5, 6)
_PLACEHOLDER_TEMPLATES: Final = (1, 2, 3, 4, 5, 6, 7)
_ENTITY_TEMPLATES: Final = (1, 2, 3)
_PLACEHOLDER_INDICES: Final = 10
_GUARD_PER_ITEM: Final = 50

_NumberParts = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RobustRow:
    """A corpus row (`src_lang`, `tgt_lang`, `src`, `tgt`, `origin`, `corpus`, `split`) and what
    its item holds: the `kind` (`base`, `number`, `number2`, `placeholder`, `entity`), the
    `template` of a base, placeholder or entity item (None for a number), the `numbers` of a
    number item (one or two), the `placeholders` indices, the `entities` ids (person, then
    city) and the `nouns` of a base item (indices into the D7 noun table)."""

    src_lang: str
    tgt_lang: str
    src: str
    tgt: str
    origin: str
    corpus: str
    split: str
    kind: str
    template: int | None = None
    numbers: tuple[NumberValue, ...] = ()
    placeholders: tuple[int, ...] = ()
    entities: tuple[str, ...] = ()
    nouns: tuple[int, ...] = ()

    def corpus_row(self) -> dict[str, str]:
        """The seven keys `write_corpus_parquet` takes."""
        return {
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "src": self.src,
            "tgt": self.tgt,
            "origin": self.origin,
            "corpus": self.corpus,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class RobustToy:
    """The rows of the toy: train, dev and test, in that order."""

    rows: tuple[RobustRow, ...]


@dataclass(frozen=True, slots=True)
class _Item:
    kind: str
    template: int | None
    texts: Mapping[str, str]
    numbers: tuple[NumberValue, ...] = ()
    placeholders: tuple[int, ...] = ()
    entities: tuple[str, ...] = ()
    nouns: tuple[int, ...] = ()


_Draw = Callable[[random.Random], tuple[_Item, Hashable]]


def _texts(render: Callable[[str], str]) -> dict[str, str]:
    return {lang: render(lang) for lang in LANGUAGES}


# ---- the draws: one item and the key that stops it being drawn twice ----------------------------


def _draw_base(rng: random.Random) -> tuple[_Item, Hashable]:
    template = rng.choice(_BASE_TEMPLATES)
    nouns = tuple(rng.sample(range(len(NOUNS)), _NOUNS_IN[template]))
    item = _Item(
        "base",
        template,
        _texts(lambda lang: noun_sentence(template, nouns, lang)),
        nouns=nouns,
    )
    return item, ("base", template, nouns)


_NUMBER_PARTS: Final[dict[str, Callable[[random.Random], _NumberParts]]] = {
    "int": lambda rng: {"n": rng.randint(2, 9999)},
    "decimal": lambda rng: {
        "whole": rng.randint(0, 99),
        "frac": rng.choice(("5", "25", "75", "4", "8", "05")),
    },
    "grouped": lambda rng: {
        "n": rng.randint(1000, 999_999),
        "frac": rng.choice((None, None, "50", "5")),
    },
    "percent": lambda rng: {
        "n": rng.randint(1, 99),
        "frac": rng.choice((None, None, "5")),
    },
    "currency": lambda rng: {
        "n": rng.choice((rng.randint(100, 9999), rng.randint(10_000, 999_999)))
    },
    "date": lambda rng: {
        "y": rng.randint(2000, 2029),
        "m": rng.randint(1, 12),
        "d": rng.randint(1, 28),
    },
    "time": lambda rng: {"h": rng.randint(0, 23), "mi": rng.randint(0, 59)},
}


def _draw_number_value(rng: random.Random, kind: str) -> NumberValue:
    parts = _NUMBER_PARTS[kind](rng)
    return NumberValue(kind, tuple(sorted(parts.items(), key=str)))


def _draw_number(rng: random.Random, kind: str) -> tuple[_Item, Hashable]:
    noun = rng.randrange(len(NOUNS))
    number = _draw_number_value(rng, kind)
    item = _Item(
        "number",
        None,
        _texts(lambda lang: number_sentence(noun, number, lang)),
        numbers=(number,),
    )
    return item, ("num", kind, noun, number.values)


def _draw_two_numbers(rng: random.Random) -> tuple[_Item, Hashable]:
    first_noun, second_noun = rng.sample(range(len(NOUNS)), 2)
    first_kind, second_kind = rng.choice(NUMBER_KINDS), rng.choice(NUMBER_KINDS)
    first = _draw_number_value(rng, first_kind)
    second = _draw_number_value(rng, second_kind)
    item = _Item(
        "number2",
        None,
        _texts(
            lambda lang: two_numbers_sentence(
                (first_noun, second_noun), (first, second), lang
            )
        ),
        numbers=(first, second),
    )
    key = (
        "num2",
        first_noun,
        second_noun,
        first_kind,
        second_kind,
        first.values,
        second.values,
    )
    return item, key


def _draw_placeholders(rng: random.Random) -> tuple[_Item, Hashable]:
    template = rng.choice(_PLACEHOLDER_TEMPLATES)
    indices = tuple(rng.sample(range(_PLACEHOLDER_INDICES), _NOUNS_IN[template]))
    item = _Item(
        "placeholder",
        template,
        _texts(lambda lang: placeholder_sentence(template, indices, lang)),
        placeholders=indices,
    )
    return item, ("ph", template, indices)


def _draw_entities(rng: random.Random) -> tuple[_Item, Hashable]:
    template = rng.choice(_ENTITY_TEMPLATES)
    person, city = rng.choice(PERSON_IDS), rng.choice(CITY_IDS)
    item = _Item(
        "entity",
        template,
        _texts(lambda lang: entity_sentence(template, person, city, lang)),
        entities=(person, city),
    )
    return item, ("ent", template, person, city)


# ---- the allocation -----------------------------------------------------------------------------


def _allocate(
    rng: random.Random,
    items: dict[str, list[Any]],
    seen: set[Any],
    counts: Mapping[str, int],
    draw: _Draw,
) -> None:
    """Draw `counts[split]` items nobody drew before for each split, in the order of
    `counts`, and add them to `items[split]`. A draw whose key was seen is dropped and drawn
    again; the supply is not endless, so a split that cannot be filled in
    `50 * count` attempts is an error.

    Raises:
        ValueError: the draw ran out of new items before a split was full.
    """
    for split, count in counts.items():
        made = 0
        attempts = 0
        while made < count and attempts < count * _GUARD_PER_ITEM:
            attempts += 1
            item, key = draw(rng)
            if key in seen:
                continue
            seen.add(key)
            items[split].append(item)
            made += 1
        if made < count:
            raise ValueError(
                f"the {split} split got {made} new items of {count} wanted: "
                "the draw ran out of items nobody drew"
            )


def _draw_items(rng: random.Random) -> dict[str, list[_Item]]:
    items: dict[str, list[_Item]] = {split: [] for split in _SPLITS}
    seen: set[Hashable] = set()

    def allocate(counts: Mapping[str, int], draw: _Draw) -> None:
        _allocate(rng, items, seen, counts, draw)

    allocate({"train": 1500, "dev": 60, "test": 100}, _draw_base)
    for kind in NUMBER_KINDS:
        counts = {"train": 520 if kind == "date" else 260, "dev": 20, "test": 30}
        allocate(counts, partial(_draw_number, kind=kind))
    allocate({"train": 700, "dev": 20, "test": 30}, _draw_two_numbers)
    allocate({"train": 700, "dev": 60, "test": 100}, _draw_placeholders)
    allocate({"train": 200, "dev": 12, "test": 40}, _draw_entities)
    return items


# ---- the rows -----------------------------------------------------------------------------------


def _row(
    item: _Item,
    split: str,
    languages: tuple[str, str],
    texts: tuple[str, str],
    *,
    origin: str,
    corpus: str,
) -> RobustRow:
    return RobustRow(
        src_lang=languages[0],
        tgt_lang=languages[1],
        src=texts[0],
        tgt=texts[1],
        origin=origin,
        corpus=corpus,
        split=split,
        kind=item.kind,
        template=item.template,
        numbers=item.numbers,
        placeholders=item.placeholders,
        entities=item.entities,
        nouns=item.nouns,
    )


def _item_rows(item: _Item, split: str) -> list[RobustRow]:
    """The four pair rows of an item and, for a base, placeholder or entity item, the two
    rows with the Khalkha side in traditional script."""
    corpus = f"toy-{item.kind}"
    rows = [
        _row(
            item,
            split,
            (source, target),
            (item.texts[source], item.texts[target]),
            origin="human",
            corpus=corpus,
        )
        for first, second in _PAIRS
        for source, target in ((first, second), (second, first))
    ]
    if item.kind in _MONG_KINDS:
        traditional = transliterate(item.texts[KHK])
        rows.extend(
            _row(
                item,
                split,
                ("mn_Mong", target),
                (traditional, item.texts[target]),
                origin="converted",
                corpus=f"{corpus}-mong",
            )
            for target in (ENG, RUS)
        )
    return rows


def build_robust_toy(seed: int = ROBUST_SEED) -> RobustToy:
    """The toy rows for `seed`: the same seed gives the same rows, another seed other
    sentences in the same sizes (train 25,520 rows, dev 1,432, test 2,400)."""
    items = _draw_items(random.Random(seed))
    return RobustToy(
        rows=tuple(
            row
            for split in _SPLITS
            for item in items[split]
            for row in _item_rows(item, split)
        )
    )


def robust_toy_digest(toy: RobustToy) -> str:
    """The SHA-256 of the lines `split TAB src_lang TAB tgt_lang TAB src TAB tgt` of every row
    joined by LF, with no final LF."""
    lines = "\n".join(
        f"{row.split}\t{row.src_lang}\t{row.tgt_lang}\t{row.src}\t{row.tgt}"
        for row in toy.rows
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


# ---- the model and the inputs of a training --------------------------------------------------------


def train_robust_spm(directory: Path, toy: RobustToy) -> Path:
    """Train the SentencePiece model of the four languages on the TRAINING rows of `toy` with
    the project's `train_spm` (one `<language>.txt` per language in `directory/corpus`, each
    text once, in the order of the rows) and return `directory/rtoy.model`. Every direction tag
    and every numbered placeholder is one piece of it; so are the ten digits and the tugrik.

    `directory` may be relative: it is made absolute here, because `SpmConfig` would anchor a
    relative path at the repository root.

    Raises:
        UpstreamError: SentencePiece rejected the run.
    """
    directory = directory.resolve()
    texts: dict[str, dict[str, None]] = {lang: {} for lang in _SPM_LANGUAGES}
    for row in toy.rows:
        if row.split == "train":
            texts[row.src_lang][row.src] = None
            texts[row.tgt_lang][row.tgt] = None
    corpus_dir = directory / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for language, lines in texts.items():
        (corpus_dir / f"{language}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
        )
    config = SpmConfig(
        vocab_size=ROBUST_SPM_VOCAB_SIZE,
        input_sentence_size=100_000,
        character_coverage=0.9999,
        temperature=0.3,
        seed=1234,
        corpus_dir=corpus_dir,
        output_prefix=directory / "rtoy",
    )
    return train_spm(config)


def robust_toy_inputs(root: Path) -> ToyInputs:
    """What an ablation or a teacher run reads: the corpus Parquet file of the training and dev
    rows (no test row) in `root`, read back as the plan does to find the six directions and the
    dev set of each (100 lines), and the SentencePiece model in `root/spm`."""
    toy = build_robust_toy()
    parquet = write_corpus_parquet(
        [row.corpus_row() for row in toy.rows if row.split != "test"],
        root / "rtoy.parquet",
    )
    training = load_training_rows([parquet])
    dev = load_dev_rows(
        [parquet],
        max_lines=DEV_LINES_PER_DIRECTION,
        directions=training.directions,
        training=training,
    )
    return ToyInputs(
        corpus=parquet, spm_model=train_robust_spm(root / "spm", toy), dev=dev.sets
    )
