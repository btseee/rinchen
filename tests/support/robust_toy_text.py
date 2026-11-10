"""What the robustness toy says (D12, spec T17): the entity table, the number formats of the three
languages and the templates that put nouns, numbers, placeholders and names into a sentence.

Ported from the design spike's generator (`rtoy.py`). Everything here is a pure function of what
it is given: the order of the random draws of `robust_toy` alone decides what the toy says, so a
change in this file that alters a sentence changes the digest the toy is pinned by.

Conventions per language (priors, as in the spec): English writes `1,250.50` and `03/12/2024`,
Khalkha `1 250,50` and `2024.03.12`, Russian `1 250,50` and `12.03.2024`, the group space being a
plain U+0020 (the engine's `normalize` turns a no-break space into one). The currency symbol and
the dash of a Russian sentence are written with `chr()`, like every character a reader could
mistake for another.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Final

from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from tests.support.tiny_parallel import (
    _TEMPLATES,  # the D7 sentence templates, indexed like LANGUAGES
    LANGUAGES,
    NOUNS,
)

ENG: Final = "eng_Latn"
KHK: Final = "khk_Cyrl"
RUS: Final = "rus_Cyrl"
TUGRIK: Final = chr(0x20AE)
EM_DASH: Final = chr(0x2014)
GENITIVE_TEMPLATE: Final = 7
NUMBER_KINDS: Final = (
    "int",
    "decimal",
    "grouped",
    "percent",
    "currency",
    "date",
    "time",
)

# ---- the entities --------------------------------------------------------------------------


def _spellings(
    english: tuple[str, ...], khalkha: tuple[str, ...], russian: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    return {ENG: english, KHK: khalkha, RUS: russian}


# The first spelling of a language is what the toy writes; the others are accepted too.
PERSONS: Final = {
    "bat": _spellings(("Bat",), ("Бат",), ("Бат",)),
    "saraa": _spellings(("Saraa",), ("Сараа",), ("Сараа",)),
    "dorj": _spellings(("Dorj",), ("Дорж",), ("Дорж",)),
    "tuya": _spellings(("Tuya",), ("Туяа",), ("Туя",)),
    "naran": _spellings(("Naran",), ("Наран",), ("Наран",)),
    "bold": _spellings(("Bold",), ("Болд",), ("Болд",)),
    "oyun": _spellings(("Oyun",), ("Оюун",), ("Оюн",)),
    "ganaa": _spellings(("Ganaa",), ("Ганаа",), ("Ганаа",)),
    "munkh": _spellings(("Munkh",), ("Мөнх",), ("Мунх",)),
    "enkh": _spellings(("Enkh",), ("Энх",), ("Энх",)),
}
CITIES: Final = {
    "ulaanbaatar": _spellings(
        ("Ulaanbaatar", "Ulan Bator"), ("Улаанбаатар",), ("Улан-Батор",)
    ),
    "darkhan": _spellings(("Darkhan",), ("Дархан",), ("Дархан",)),
    "erdenet": _spellings(("Erdenet",), ("Эрдэнэт",), ("Эрдэнэт",)),
    "khovd": _spellings(("Khovd",), ("Ховд",), ("Ховд",)),
    "choibalsan": _spellings(("Choibalsan",), ("Чойбалсан",), ("Чойбалсан",)),
    "murun": _spellings(("Murun",), ("Мөрөн",), ("Мурэн",)),
    "bayankhongor": _spellings(("Bayankhongor",), ("Баянхонгор",), ("Баянхонгор",)),
    "sukhbaatar": _spellings(("Sukhbaatar",), ("Сүхбаатар",), ("Сухэ-Батор",)),
    "zuunmod": _spellings(("Zuunmod",), ("Зуунмод",), ("Зуунмод",)),
    "uliastai": _spellings(("Uliastai",), ("Улиастай",), ("Улиастай",)),
}
PERSON_IDS: Final = tuple(PERSONS)
CITY_IDS: Final = tuple(CITIES)
ENTITIES: Final = {**PERSONS, **CITIES}

# The template of an entity sentence, by language; `{person}` and `{city}` take the first
# spelling of the language. The Khalkha side glues a case suffix to the city.
_ENTITY_TEMPLATES: Final = {
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


def entity_sentence(template: int, person: str, city: str, lang: str) -> str:
    """The sentence of an entity item: `person` and `city` are ids of `ENTITIES`."""
    return _ENTITY_TEMPLATES[template][lang].format(
        person=ENTITIES[person][lang][0], city=ENTITIES[city][lang][0]
    )


# ---- numbers -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NumberValue:
    """One number of a sentence: its `kind` and its parts as `(name, value)` pairs in the
    order of `sorted(..., key=str)` (the whole `n`, the fraction digits `frac` that keep a
    leading zero, the year, month and day of a date...). The same pairs make the item's key."""

    kind: str
    values: tuple[tuple[str, Any], ...]


def _group_int(n: int, lang: str, *, grouped: bool) -> str:
    text = f"{n:,}" if grouped else str(n)
    return text if lang == ENG else text.replace(",", " ")


def _decimal(whole: int, fraction: str, lang: str, *, grouped: bool) -> str:
    whole_text = _group_int(whole, lang, grouped=grouped)
    return f"{whole_text}.{fraction}" if lang == ENG else f"{whole_text},{fraction}"


def _number(v: Mapping[str, Any], lang: str, *, grouped: bool) -> str:
    if v["frac"] is None:
        return _group_int(v["n"], lang, grouped=grouped)
    return _decimal(v["n"], v["frac"], lang, grouped=grouped)


def _date(v: Mapping[str, Any], lang: str) -> str:
    year, month, day = v["y"], v["m"], v["d"]
    if lang == ENG:
        return f"{month:02d}/{day:02d}/{year}"
    if lang == KHK:
        return f"{year}.{month:02d}.{day:02d}"
    return f"{day:02d}.{month:02d}.{year}"


_NUMBER_TEXT: Final[dict[str, Callable[[Mapping[str, Any], str], str]]] = {
    "int": lambda v, lang: _group_int(v["n"], lang, grouped=False),
    "decimal": lambda v, lang: _decimal(v["whole"], v["frac"], lang, grouped=False),
    "grouped": lambda v, lang: _number(v, lang, grouped=True),
    "percent": lambda v, lang: f"{_number(v, lang, grouped=False)}%",
    "currency": lambda v, lang: f"{_group_int(v['n'], lang, grouped=True)}{TUGRIK}",
    "date": _date,
    "time": lambda v, lang: f"{v['h']:02d}:{v['mi']:02d}",
}


def number_text(number: NumberValue, lang: str) -> str:
    """The number as `lang` writes it."""
    return _NUMBER_TEXT[number.kind](dict(number.values), lang)


# ---- sentences -----------------------------------------------------------------------------


def sentence_case(text: str) -> str:
    return text[0].upper() + text[1:]


def _column(lang: str) -> int:
    """The column of `lang` in the noun table and in the templates."""
    return LANGUAGES.index(lang)


def _template_sentence(template: int, words: Sequence[str], lang: str) -> str:
    return sentence_case(_TEMPLATES[_column(lang)][template].format(*words))


def noun_sentence(template: int, nouns: Sequence[int], lang: str) -> str:
    """The D7 sentence of `template` over the nouns (indices into `NOUNS`)."""
    column = _column(lang)
    return _template_sentence(template, [NOUNS[i][column] for i in nouns], lang)


@cache
def _placeholders() -> tuple[str, ...]:
    return tuple(user_defined_symbols()[-PLACEHOLDER_COUNT:])


def placeholder(index: int) -> str:
    """The reserved symbol of the numbered placeholder `index` (0 to 99)."""
    return _placeholders()[index]


def placeholder_sentence(template: int, indices: Sequence[int], lang: str) -> str:
    """A D7 sentence with the nouns replaced by placeholders; the genitive template swaps
    the order of its two placeholders in Khalkha."""
    words = [placeholder(index) for index in indices]
    if template != GENITIVE_TEMPLATE:
        return _template_sentence(template, words, lang)
    first, second = words
    if lang == ENG:
        return f"The {first} of the {second}."
    if lang == KHK:
        return f"{second}ийн {first}."
    return f"{first} {second}."


def number_sentence(noun: int, number: NumberValue, lang: str) -> str:
    """`The book is 1,250.` and its Khalkha and Russian counterparts."""
    text = number_text(number, lang)
    word = NOUNS[noun][_column(lang)]
    if lang == ENG:
        return f"The {word} is {text}."
    if lang == KHK:
        return sentence_case(f"{word} {text} байна.")
    return sentence_case(f"{word} {EM_DASH} {text}.")


def two_numbers_sentence(
    nouns: tuple[int, int], numbers: tuple[NumberValue, NumberValue], lang: str
) -> str:
    """`The book: 5, the dog: 12.` and its counterparts."""
    column = _column(lang)
    first, second = (NOUNS[noun][column] for noun in nouns)
    one, other = (number_text(number, lang) for number in numbers)
    if lang == ENG:
        return f"The {first}: {one}, the {second}: {other}."
    return sentence_case(f"{first}: {one}, {second}: {other}.")
