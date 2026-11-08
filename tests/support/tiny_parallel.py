"""Deterministic templated parallel data in khk_Cyrl, eng_Latn and rus_Cyrl for the
D7 tests, and a tiny SentencePiece model trained on it.

Ported from the D7 spike's generator. Every sentence is one of six templates over
37 basic nouns, with no agreement: English always says "the", and the Mongolian and
Russian nouns stay in their bare nominative form. A content item is a template and
the nouns that fill it; one item is rendered in all three languages, so the data is
three-way parallel.

What `build_toy` returns:

  train  en-mn: 1,200 items and ru-mn: 600 items, each in both directions. The two
         language pairs use disjoint content, and they differ in size so that a
         temperature changes the mixing weights (the direction totals are 1,200 and
         600). 600 is as many as keeps the ru directions near the 1,200 pairs the
         spike trained to chrF 98 and more.
  bt     300 items per language pair in both directions, origin `bt`, content that
         no other group has.
  dev    100 held-out items (12 single-noun, 44 pair and 44 triple sentences) in the
         four directions; test holds another 100.

No sentence of one group is in another, in any language (the generator asserts it).
Nothing is read or written here but through the functions that take a path.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Any

from rinchen.config.spm import SpmConfig
from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.spm.train import train_spm

assert sys.version_info[:2] == (3, 14), sys.version

# (khk_Cyrl, eng_Latn, rus_Cyrl): only very basic words, in the bare forms.
NOUNS: tuple[tuple[str, str, str], ...] = (
    ("ном", "book", "книга"),
    ("ус", "water", "вода"),
    ("гэр", "home", "дом"),
    ("нар", "sun", "солнце"),
    ("сар", "moon", "луна"),
    ("од", "star", "звезда"),
    ("хүн", "person", "человек"),
    ("нохой", "dog", "собака"),
    ("морь", "horse", "лошадь"),
    ("мод", "tree", "дерево"),
    ("цэцэг", "flower", "цветок"),
    ("хот", "city", "город"),
    ("уул", "mountain", "гора"),
    ("гол", "river", "река"),
    ("талх", "bread", "хлеб"),
    ("цай", "tea", "чай"),
    ("сүү", "milk", "молоко"),
    ("загас", "fish", "рыба"),
    ("шувуу", "bird", "птица"),
    ("өдөр", "day", "день"),
    ("хаалга", "door", "дверь"),
    ("цонх", "window", "окно"),
    ("сандал", "chair", "стул"),
    ("ширээ", "table", "стол"),
    ("машин", "car", "машина"),
    ("алим", "apple", "яблоко"),
    ("бороо", "rain", "дождь"),
    ("цас", "snow", "снег"),
    ("салхи", "wind", "ветер"),
    ("сургууль", "school", "школа"),
    ("багш", "teacher", "учитель"),
    ("ээж", "mother", "мама"),
    ("аав", "father", "отец"),
    ("зам", "road", "дорога"),
    ("гар", "hand", "рука"),
    ("нүд", "eye", "глаз"),
    ("дэлгүүр", "shop", "магазин"),
)

LANGUAGES = ("khk_Cyrl", "eng_Latn", "rus_Cyrl")
TOY_DIRECTIONS = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "rus_Cyrl-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
)

# The vocabulary of the toy SentencePiece model. For this text the project's
# `train_spm` accepts 458 (118 reserved symbols, 256 byte pieces, 3 control symbols
# and the 81 characters) up to 674 (nothing more to learn). 640 sits clear of both
# ends and near enough the top that most nouns are one piece: about 7 pieces per
# sentence, against 14 at 500.
TOY_SPM_VOCAB_SIZE = 640
TOY_SPM_VOCAB_RANGE = (458, 674)

DEFAULT_SEED = 7
HELD_OUT_ITEMS = 100

_NOUN_COUNT = len(NOUNS)
_SINGLE_TEMPLATES = (1, 2, 3, 4)
_PAIR_TEMPLATE = 5
_TRIPLE_TEMPLATE = 6
# Items per kind: (singles, pairs, triples).
_EN_MN_ITEMS = (62, 600, 538)  # 1,200
_RU_MN_ITEMS = (31, 300, 269)  # 600
_BT_ITEMS = (0, 100, 200)  # 300 per language pair
_HELD_OUT_SINGLES_PER_TEMPLATE = 3
_HELD_OUT_PAIRS = 44
_HELD_OUT_TRIPLES = 44

_CREATED = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

ToyRow = dict[str, str]


@dataclass(frozen=True, slots=True)
class _Item:
    template: int
    nouns: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ToyData:
    """Rows in the form `corpus_frame` takes, in direction order within a group."""

    train: tuple[ToyRow, ...]
    bt: tuple[ToyRow, ...]
    dev: tuple[ToyRow, ...]
    test: tuple[ToyRow, ...]


# One format string per template and language; `{0}`, `{1}` and `{2}` are the nouns.
# The first letter of a sentence is capitalised after the nouns are put in.
_TEMPLATES: tuple[dict[int, str], ...] = (
    {  # khk_Cyrl
        1: "Энэ бол {0}.",
        2: "{0} энд байна.",
        3: "{0} хаана байна вэ?",
        4: "Тэнд {0} байна.",
        5: "{0} ба {1}.",
        6: "{0}, {1} ба {2}.",
    },
    {  # eng_Latn
        1: "This is the {0}.",
        2: "The {0} is here.",
        3: "Where is the {0}?",
        4: "The {0} is there.",
        5: "The {0} and the {1}.",
        6: "The {0}, the {1} and the {2}.",
    },
    {  # rus_Cyrl
        1: "Это {0}.",
        2: "{0} здесь.",
        3: "Где {0}?",
        4: "Там {0}.",
        5: "{0} и {1}.",
        6: "{0}, {1} и {2}.",
    },
)

_EN_MN = ("eng_Latn", "khk_Cyrl")
_RU_MN = ("rus_Cyrl", "khk_Cyrl")


def _render(item: _Item, column: int) -> str:
    """The sentence of `item` in the language of `LANGUAGES[column]`."""
    words = [NOUNS[index][column] for index in item.nouns]
    sentence = _TEMPLATES[column][item.template].format(*words)
    return sentence[0].upper() + sentence[1:]


def _rows(
    items: Sequence[_Item],
    languages: tuple[str, str],
    *,
    origin: str,
    corpus: str,
    split: str,
) -> list[ToyRow]:
    """The items as rows of both directions between `languages`: every row of the
    first direction, then every row of the reverse."""
    return [
        {
            "src_lang": source,
            "tgt_lang": target,
            "src": _render(item, LANGUAGES.index(source)),
            "tgt": _render(item, LANGUAGES.index(target)),
            "origin": origin,
            "corpus": corpus,
            "split": split,
        }
        for source, target in (languages, languages[::-1])
        for item in items
    ]


class _Pool:
    """Content items handed out in slices that never overlap."""

    def __init__(self, items: list[_Item]) -> None:
        self._items = items
        self._taken = 0

    def take(self, count: int) -> list[_Item]:
        chunk = self._items[self._taken : self._taken + count]
        if len(chunk) != count:
            raise ValueError(f"the pool has {len(chunk)} items left, {count} wanted")
        self._taken += count
        return chunk

    def rest(self) -> list[_Item]:
        return self._items[self._taken :]


def _shuffled(rng: random.Random, items: list[_Item]) -> list[_Item]:
    rng.shuffle(items)
    return items


def build_toy(seed: int = DEFAULT_SEED) -> ToyData:
    """The toy rows for `seed`: the same seed gives the same rows, another seed
    other sentences in the same sizes."""
    rng = random.Random(seed)
    singles = {
        template: _Pool(
            _shuffled(rng, [_Item(template, (noun,)) for noun in range(_NOUN_COUNT)])
        )
        for template in _SINGLE_TEMPLATES
    }
    pairs = _Pool(
        _shuffled(
            rng, [_Item(_PAIR_TEMPLATE, n) for n in permutations(range(_NOUN_COUNT), 2)]
        )
    )
    triples = _Pool(
        _shuffled(
            rng,
            [_Item(_TRIPLE_TEMPLATE, n) for n in permutations(range(_NOUN_COUNT), 3)],
        )
    )

    def held_out() -> list[_Item]:
        return [
            *(
                item
                for template in _SINGLE_TEMPLATES
                for item in singles[template].take(_HELD_OUT_SINGLES_PER_TEMPLATE)
            ),
            *pairs.take(_HELD_OUT_PAIRS),
            *triples.take(_HELD_OUT_TRIPLES),
        ]

    dev_items, test_items = held_out(), held_out()
    spare_singles = _Pool(
        _shuffled(
            rng,
            [
                item
                for template in _SINGLE_TEMPLATES
                for item in singles[template].rest()
            ],
        )
    )

    def take(counts: tuple[int, int, int]) -> list[_Item]:
        single_count, pair_count, triple_count = counts
        return [
            *spare_singles.take(single_count),
            *pairs.take(pair_count),
            *triples.take(triple_count),
        ]

    en_items, ru_items = take(_EN_MN_ITEMS), take(_RU_MN_ITEMS)
    bt_en_items, bt_ru_items = take(_BT_ITEMS), take(_BT_ITEMS)
    rng.shuffle(en_items)
    rng.shuffle(ru_items)

    data = ToyData(
        train=(
            *_rows(en_items, _EN_MN, origin="human", corpus="toy-en-mn", split="train"),
            *_rows(ru_items, _RU_MN, origin="human", corpus="toy-ru-mn", split="train"),
        ),
        bt=(
            *_rows(bt_en_items, _EN_MN, origin="bt", corpus="toy-bt-en", split="train"),
            *_rows(bt_ru_items, _RU_MN, origin="bt", corpus="toy-bt-ru", split="train"),
        ),
        dev=_held_out_rows(dev_items, "dev"),
        test=_held_out_rows(test_items, "test"),
    )
    _check_no_shared_sentence(data)
    return data


def _held_out_rows(items: Sequence[_Item], split: str) -> tuple[ToyRow, ...]:
    return (
        *_rows(items, _EN_MN, origin="human", corpus=f"toy-{split}", split=split),
        *_rows(items, _RU_MN, origin="human", corpus=f"toy-{split}", split=split),
    )


def _check_no_shared_sentence(data: ToyData) -> None:
    groups = {"train": data.train, "bt": data.bt, "dev": data.dev, "test": data.test}
    sentences: dict[str, dict[str, set[str]]] = {
        name: {language: set() for language in LANGUAGES} for name in groups
    }
    for name, rows in groups.items():
        for row in rows:
            sentences[name][row["src_lang"]].add(row["src"])
            sentences[name][row["tgt_lang"]].add(row["tgt"])
    names = list(groups)
    for language in LANGUAGES:
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                shared = sentences[first][language] & sentences[second][language]
                assert not shared, (language, first, second, sorted(shared)[:3])


def write_corpus_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write `rows` as a corpus Parquet file at `path` (directories are made) and
    return `path`. The timestamp is fixed, so equal rows give equal bytes."""
    write_corpus(corpus_frame(rows, created=_CREATED), path)
    return path


def train_toy_spm(directory: Path, *, vocab_size: int = TOY_SPM_VOCAB_SIZE) -> Path:
    """Train the toy SentencePiece model with the project's `train_spm` on the
    training and back-translated text of `build_toy()` (`<language>.txt` files in
    `directory/corpus`) and return the path of `directory/toy.model`. Every
    direction tag and `<bt>` is a user-defined symbol of the project's model, so one
    piece each.

    Raises:
        UpstreamError: `vocab_size` is outside what the text supports.
    """
    toy = build_toy()
    texts: dict[str, dict[str, None]] = {language: {} for language in LANGUAGES}
    for row in (*toy.train, *toy.bt):
        texts[row["src_lang"]][row["src"]] = None
        texts[row["tgt_lang"]][row["tgt"]] = None
    corpus_dir = directory / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for language, lines in texts.items():
        (corpus_dir / f"{language}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
        )
    cfg = SpmConfig(
        vocab_size=vocab_size,
        input_sentence_size=100_000,
        character_coverage=0.9999,
        temperature=0.3,
        seed=1234,
        corpus_dir=corpus_dir,
        output_prefix=directory / "toy",
    )
    return train_spm(cfg)
