# scripts/build_parallel_fixtures.py
"""Generator of tests/fixtures/parallel/*.tsv, the committed fixtures of the tiny
end-to-end pipeline (D15, spec P5).

usage: build_parallel_fixtures.py OUT_DIR
Regenerate the committed files with `uv run python scripts/build_parallel_fixtures.py
tests/fixtures/parallel`; the new digests are a decision, not an update (see the tests).

Seven tab-separated files, UTF-8, LF, one header line each:

- eng_Latn-khk_Cyrl.tsv and rus_Cyrl-khk_Cyrl.tsv: 345 rows each (300 train, 15 dev,
  30 test) over six kinds of sentence: `base` (six templates over 37 nouns), `int`,
  `decimal` and `grouped` (a noun and a number written in each language's own format),
  `placeholder` (a template over the numbered placeholders 0 to 9) and `entity` (a person
  and a city). One item is rendered in all three languages.
- mn_Mong.tsv: 70 rows, the traditional-script spelling of Khalkha sentences that the
  English pair file holds, in the same split.
- mono_khk_Cyrl.tsv: 64 Khalkha lines with their English oracle (the synthetic job's source).
- noise.tsv: ten planted bad pairs; `expect` names the reason the filter must give.
- nouns.tsv and entities.tsv: the tables the sentences are made from.

Every sentence is templated and deterministic: nothing here is a real text. The tests
(tests/unit/test_parallel_fixtures.py) compare the committed files with this output, pin
their digests and check the composition; a change to this generator that moves one byte is a
change of the fixtures, and of every measurement made on them.

Determinism: each file draws from its own random.Random stream, seeded with the string
f"{SEED}-{name}"; a string seed is hashed with SHA-512, so the streams do not depend on
PYTHONHASHSEED or on the order the files are written in. The sets below answer membership
only and are never iterated. The generator is self-contained: it embeds its nouns, entities
and templates and imports nothing but the standard library and the transliteration.

Characters a reader could mistake for another are written with chr(): the two brackets of a
placeholder and the dash of a Russian number sentence.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.script_conv.rules import transliterate

SEED = 7
KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
RUS: Final = "rus_Cyrl"
LANGUAGES: Final = (KHK, ENG, RUS)
MONGOLIAN: Final = "mn_Mong"

PLACEHOLDER_OPEN: Final = chr(0x27E6)
PLACEHOLDER_CLOSE: Final = chr(0x27E7)
PLACEHOLDER_COUNT: Final = 10
EM_DASH: Final = chr(0x2014)

KINDS: Final = ("base", "int", "decimal", "grouped", "placeholder", "entity")
# Items per kind, in the order of KINDS, of each split of a pair file.
SPLITS: Final = {
    "train": (150, 30, 18, 12, 60, 30),
    "dev": (8, 2, 1, 1, 2, 1),
    "test": (15, 3, 2, 2, 5, 3),
}
MONGOLIAN_SPLITS: Final = (("train", 50), ("dev", 10), ("test", 10))
MONGOLIAN_KINDS: Final = ("base", "entity", "placeholder")
MONO_ITEMS: Final = (40, 0, 0, 0, 12, 12)
NOUNS_OF_TEMPLATE: Final = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3}
BASE_TEMPLATE_DRAW: Final = (1, 2, 3, 4, 5, 5, 6)
DECIMAL_FRACTIONS: Final = ("5", "25", "75", "4", "8")
ENTITY_TEMPLATES: Final = (1, 2, 3)

# (khk_Cyrl, eng_Latn, rus_Cyrl): only very basic words, in the bare forms.
NOUNS: Final[tuple[tuple[str, str, str], ...]] = (
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

# id -> (khk_Cyrl, eng_Latn, rus_Cyrl) cells, spellings separated by `|`. The first spelling
# of a language is what a sentence writes; the others are accepted too.
PERSONS: Final = {
    "bat": ("Бат", "Bat", "Бат"),
    "saraa": ("Сараа", "Saraa", "Сараа"),
    "dorj": ("Дорж", "Dorj", "Дорж"),
    "tuya": ("Туяа", "Tuya", "Туя"),
    "naran": ("Наран", "Naran", "Наран"),
    "bold": ("Болд", "Bold", "Болд"),
    "oyun": ("Оюун", "Oyun", "Оюн"),
    "ganaa": ("Ганаа", "Ganaa", "Ганаа"),
    "munkh": ("Мөнх", "Munkh", "Мунх"),
    "enkh": ("Энх", "Enkh", "Энх"),
}
CITIES: Final = {
    "ulaanbaatar": ("Улаанбаатар", "Ulaanbaatar|Ulan Bator", "Улан-Батор"),
    "darkhan": ("Дархан", "Darkhan", "Дархан"),
    "erdenet": ("Эрдэнэт", "Erdenet", "Эрдэнэт"),
    "khovd": ("Ховд", "Khovd", "Ховд"),
    "choibalsan": ("Чойбалсан", "Choibalsan", "Чойбалсан"),
    "murun": ("Мөрөн", "Murun", "Мурэн"),
    "bayankhongor": ("Баянхонгор", "Bayankhongor", "Баянхонгор"),
    "sukhbaatar": ("Сүхбаатар", "Sukhbaatar", "Сухэ-Батор"),
    "zuunmod": ("Зуунмод", "Zuunmod", "Зуунмод"),
    "uliastai": ("Улиастай", "Uliastai", "Улиастай"),
}
ENTITIES: Final = {**PERSONS, **CITIES}
PERSON_IDS: Final = tuple(PERSONS)
CITY_IDS: Final = tuple(CITIES)

# One format string per template and language; `{0}`, `{1}` and `{2}` are the nouns (or the
# placeholders). The first letter of a sentence is capitalised after they are put in.
TEMPLATES: Final = {
    KHK: {
        1: "Энэ бол {0}.",
        2: "{0} энд байна.",
        3: "{0} хаана байна вэ?",
        4: "Тэнд {0} байна.",
        5: "{0} ба {1}.",
        6: "{0}, {1} ба {2}.",
    },
    ENG: {
        1: "This is the {0}.",
        2: "The {0} is here.",
        3: "Where is the {0}?",
        4: "The {0} is there.",
        5: "The {0} and the {1}.",
        6: "The {0}, the {1} and the {2}.",
    },
    RUS: {
        1: "Это {0}.",
        2: "{0} здесь.",
        3: "Где {0}?",
        4: "Там {0}.",
        5: "{0} и {1}.",
        6: "{0}, {1} и {2}.",
    },
}
# The entity sentences; `{person}` and `{city}` take the first spelling of the language, and
# the Khalkha side glues a case suffix to the city.
ENTITY_SENTENCES: Final = {
    1: {
        ENG: "{person} lives in {city}.",
        KHK: "{person} {city}д амьдардаг.",
        RUS: "{person} живёт в {city}.",
    },
    2: {
        ENG: "{person} is from {city}.",
        KHK: "{person} {city}аас ирсэн.",
        RUS: "{person} из {city}.",
    },
    3: {
        ENG: "Is {person} in {city}?",
        KHK: "{person} {city}д байна уу?",
        RUS: "{person} в {city}?",
    },
}

# The planted pairs as (expect, src, tgt), English to Khalkha. The two `duplicate` rows repeat
# the first train rows of the English pair file and are put in after the sixth row.
NOISE_ROWS: Final = (
    ("script_tgt", "This is the book.", "This is the book."),
    ("script_tgt", "Where is the dog?", "Where is the dog?"),
    ("len_ratio_char", "The book is here.", "Ном."),
    ("len_ratio_char", "The book, the water and the home.", "Ном."),
    ("numbers", "The book is 12.", "Ном 13 байна."),
    ("numbers", "The dog is 7.", "Нохой 8 байна."),
    ("labse", "The book is here.", "Ус хаана байна вэ?"),
    ("labse", "Where is the dog?", "Ном энд байна."),
)

Texts = dict[str, str]
Row = tuple[str, ...]


class Number(NamedTuple):
    kind: str
    value: int
    fraction: str | None


class Item(NamedTuple):
    """What one draw makes: a key that says which content it is, and its text per language."""

    kind: str
    key: tuple[object, ...]
    texts: Texts


def _capitalised(text: str) -> str:
    return text[0].upper() + text[1:]


def _placeholder(index: int) -> str:
    return f"{PLACEHOLDER_OPEN}{index}{PLACEHOLDER_CLOSE}"


def _template_sentence(template: int, words: Sequence[str], language: str) -> str:
    return _capitalised(TEMPLATES[language][template].format(*words))


def _number_text(number: Number, language: str) -> str:
    if number.fraction is not None:
        point = "." if language == ENG else ","
        return f"{number.value}{point}{number.fraction}"
    if number.kind != "grouped":
        return str(number.value)
    return (
        f"{number.value:,}"
        if language == ENG
        else f"{number.value:,}".replace(",", " ")
    )


def _number_sentence(noun: int, number: Number, language: str) -> str:
    word = NOUNS[noun][LANGUAGES.index(language)]
    text = _number_text(number, language)
    if language == ENG:
        return f"The {word} is {text}."
    if language == KHK:
        return _capitalised(f"{word} {text} байна.")
    return _capitalised(f"{word} {EM_DASH} {text}.")


def _entity_sentence(template: int, person: str, city: str, language: str) -> str:
    spelled = {
        id_: ENTITIES[id_][LANGUAGES.index(language)].split("|")[0]
        for id_ in (person, city)
    }
    return ENTITY_SENTENCES[template][language].format(
        person=spelled[person], city=spelled[city]
    )


def _draw_number(rng: random.Random, kind: str) -> Number:
    if kind == "int":
        return Number(kind, rng.randint(1, 999), None)
    if kind == "decimal":
        return Number(kind, rng.randint(0, 99), rng.choice(DECIMAL_FRACTIONS))
    return Number(kind, rng.randint(1000, 99_999), None)


def _draw(rng: random.Random, kind: str) -> Item:
    """One item of `kind`. The order of the draws from `rng` is part of what the files hold."""
    if kind == "base":
        template = rng.choice(BASE_TEMPLATE_DRAW)
        nouns = tuple(rng.sample(range(len(NOUNS)), NOUNS_OF_TEMPLATE[template]))
        texts = {
            language: _template_sentence(
                template, [NOUNS[i][LANGUAGES.index(language)] for i in nouns], language
            )
            for language in LANGUAGES
        }
        return Item(kind, ("b", template, nouns), texts)
    if kind in ("int", "decimal", "grouped"):
        noun = rng.randrange(len(NOUNS))
        number = _draw_number(rng, kind)
        texts = {
            language: _number_sentence(noun, number, language) for language in LANGUAGES
        }
        return Item(kind, ("n", noun, number), texts)
    if kind == "placeholder":
        template = rng.choice(tuple(NOUNS_OF_TEMPLATE))
        indices = tuple(
            rng.sample(range(PLACEHOLDER_COUNT), NOUNS_OF_TEMPLATE[template])
        )
        words = [_placeholder(index) for index in indices]
        texts = {
            language: _template_sentence(template, words, language)
            for language in LANGUAGES
        }
        return Item(kind, ("p", template, indices), texts)
    template = rng.choice(ENTITY_TEMPLATES)
    person, city = rng.choice(PERSON_IDS), rng.choice(CITY_IDS)
    texts = {
        language: _entity_sentence(template, person, city, language)
        for language in LANGUAGES
    }
    return Item(kind, ("e", template, person, city), texts)


def _items(rng: random.Random, counts: Sequence[int], taken: set[object]) -> list[Item]:
    """`counts[i]` new items of `KINDS[i]`: a draw whose content or any sentence is in `taken`
    is dropped, so no sentence of any language appears twice across the pair and monolingual
    files (the planted noise rows are written by hand and reuse some train sentences)."""
    items: list[Item] = []
    for kind, count in zip(KINDS, counts, strict=True):
        found = 0
        while found < count:
            item = _draw(rng, kind)
            if item.key in taken or any(text in taken for text in item.texts.values()):
                continue
            taken.add(item.key)
            taken.update(item.texts.values())
            items.append(item)
            found += 1
    return items


def _pair_rows(
    rng: random.Random, prefix: str, other: str, taken: set[object]
) -> list[Row]:
    rows: list[Row] = []
    for split, counts in SPLITS.items():
        for item in _items(rng, counts, taken):
            rows.append(
                (
                    f"{prefix}-{len(rows) + 1:04d}",
                    split,
                    item.kind,
                    item.texts[other],
                    item.texts[KHK],
                )
            )
    return rows


def _mongolian_rows(english_pair: Sequence[Row]) -> list[Row]:
    """Traditional-script copies of Khalkha sentences the English pair file already holds."""
    rows: list[Row] = []
    for split, count in MONGOLIAN_SPLITS:
        chosen = [
            row for row in english_pair if row[1] == split and row[2] in MONGOLIAN_KINDS
        ][:count]
        for _, _, kind, english, khalkha in chosen:
            rows.append(
                (
                    f"mm-{len(rows) + 1:04d}",
                    split,
                    kind,
                    transliterate(khalkha),
                    khalkha,
                    english,
                )
            )
    return rows


def _noise_rows(english_pair: Sequence[Row]) -> list[Row]:
    """Ten planted pairs. A `duplicate` row repeats a train row of the English pair file."""
    train = [row for row in english_pair if row[1] == "train" and row[2] == "base"]
    planted = list(NOISE_ROWS)
    planted[6:6] = [("duplicate", row[3], row[4]) for row in train[:2]]
    return [
        (f"no-{number:04d}", expect, ENG, KHK, src, tgt)
        for number, (expect, src, tgt) in enumerate(planted, start=1)
    ]


def _write_tsv(path: Path, header: Sequence[str], rows: Sequence[Row]) -> None:
    lines = ["\t".join(row) for row in [tuple(header), *rows]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build(out_dir: Path) -> None:
    """Write the seven fixture files into `out_dir` (created when missing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    taken: set[object] = set()
    pairs: dict[str, list[Row]] = {}
    for pair, prefix, other in (
        (f"{ENG}-{KHK}", "ek", ENG),
        (f"{RUS}-{KHK}", "rk", RUS),
    ):
        pairs[pair] = _pair_rows(random.Random(f"{SEED}-{pair}"), prefix, other, taken)
        _write_tsv(
            out_dir / f"{pair}.tsv", ("id", "split", "kind", other, KHK), pairs[pair]
        )
    english_pair = pairs[f"{ENG}-{KHK}"]
    _write_tsv(
        out_dir / f"{MONGOLIAN}.tsv",
        ("id", "split", "kind", MONGOLIAN, KHK, ENG),
        _mongolian_rows(english_pair),
    )
    mono = _items(random.Random(f"{SEED}-mono"), MONO_ITEMS, taken)
    _write_tsv(
        out_dir / f"mono_{KHK}.tsv",
        ("id", KHK, ENG),
        [
            (f"mo-{i:04d}", item.texts[KHK], item.texts[ENG])
            for i, item in enumerate(mono, start=1)
        ],
    )
    _write_tsv(
        out_dir / "noise.tsv",
        ("id", "expect", "src_lang", "tgt_lang", "src", "tgt"),
        _noise_rows(english_pair),
    )
    _write_tsv(out_dir / "nouns.tsv", LANGUAGES, NOUNS)
    _write_tsv(
        out_dir / "entities.tsv",
        ("id", *LANGUAGES),
        [(id_, *cells) for id_, cells in ENTITIES.items()],
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: build_parallel_fixtures.py OUT_DIR", file=sys.stderr)
        return 2
    build(Path(arguments[0]))
    print(f"wrote 7 files to {arguments[0]}")
    return 0


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    raise SystemExit(main())
