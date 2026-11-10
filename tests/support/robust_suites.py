"""The suites of D12 built from the robustness toy (spec T17, T18): the main suite the gate scores
and three controls, each a pure function of the toy (`build_unseen_placeholder_suite` of nothing
at all), so the same toy gives the same bytes.

The main suite is made of the held-out rows (the TEST split, in the order the toy lists them):

  numbers       the 840 `number` and 120 `number2` rows of the four pair directions; category the
                kind of the number (`int`, `decimal`, `grouped`, `percent`, `currency`, `date`,
                `time`) or `two_numbers`
  placeholders  the 600 `placeholder` rows of the six directions; category `noun`, or `reorder`
                for the genitive sentence (the toy's held-out split holds none: its training
                split takes every sentence of the one- and two-placeholder templates, so
                `reorder` is a category the builder can write and this toy never gives it a row)
  entities      the 240 `entity` rows of the six directions; the person and the city of each
  terms         the 400 `base` rows of the four pair directions; every noun of the sentence is a
                glossary entry (the noun of the source language, the noun of the target); 1,044
  script        `trained`: the same 400 base rows; `untrained`: for each of the pairs eng-eng,
                eng-rus, khk-khk, rus-rus, rus-eng the first 100 held-out base sources of the
                pair's source language, with no source language given (500 rows). Khalkha has two
                held-out rows for every sentence (to English and to Russian), so its 100 sources
                are 50 sentences, each twice. A script row judges the output alone: no reference
  mong          the 480 `base`, `entity` and `placeholder` rows from Khalkha into English and
                Russian, with the Khalkha written in traditional script (`rules.transliterate`,
                as the toy's `mn_Mong` rows have it); category the kind

The controls: the terms section alone with a glossary the model never writes
(`build_term_control_suite`), placeholders of indices 10 and up that no toy row trains
(`build_unseen_placeholder_suite`) and the first 100 English to Khalkha base rows asked in
traditional script (`build_mong_output_suite`). Ids are `num-0001`, `ph-0001`, `ent-0001`,
`term-0001`, `trained-0001`, `untrained-eng-rus-001`, `mong-0001`, `unseen-eng-khk-10` and
`mong-out-0001`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, Final

from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite import Suite, SuiteRow, TermPair
from rinchen.script_conv.rules import transliterate
from tests.support.robust_toy import GENITIVE_TEMPLATE, RobustRow, RobustToy
from tests.support.robust_toy_text import ENG, ENTITIES, KHK, RUS, placeholder
from tests.support.tiny_parallel import LANGUAGES, NOUNS

__all__ = [
    "CONTROL_TERMS",
    "UNSEEN_INDICES",
    "build_main_suite",
    "build_mong_output_suite",
    "build_term_control_suite",
    "build_unseen_placeholder_suite",
]

# What the term control asks for instead of the noun: a word that is no sentence of the toy.
CONTROL_TERMS: Final = {KHK: "тавиур", ENG: "stand", RUS: "подставка"}
# Placeholder indices no row of the toy trains (it trains 0 to 9).
UNSEEN_INDICES: Final = (10, 11, 12, 25, 50, 75, 99)
_UNSEEN_SENTENCES: Final = {ENG: "This is {}.", KHK: "Энэ бол {}.", RUS: "Это {}."}
_UNSEEN_DIRECTIONS: Final = ((ENG, KHK), (KHK, ENG), (RUS, KHK), (KHK, RUS))
_UNTRAINED_PAIRS: Final = (
    (ENG, ENG),
    (ENG, RUS),
    (KHK, KHK),
    (RUS, RUS),
    (RUS, ENG),
)
_UNTRAINED_SOURCES: Final = 100
_MONG_OUTPUT_ROWS: Final = 100
_NO_ENTITIES: Final = EntityTable({})


def _held_out(toy: RobustToy, *kinds: str) -> list[RobustRow]:
    """The test rows of the kinds, in the order of the toy."""
    return [row for row in toy.rows if row.split == "test" and row.kind in kinds]


def _pair_rows(rows: Iterable[RobustRow]) -> list[RobustRow]:
    return [row for row in rows if row.src_lang != "mn_Mong"]


def _row(
    row_id: str,
    row: RobustRow,
    category: str,
    **fields: Any,
) -> SuiteRow:
    return SuiteRow(
        id=row_id,
        src_lang=row.src_lang,
        tgt_lang=row.tgt_lang,
        source=row.src,
        reference=row.tgt,
        category=category,
        **fields,
    )


def _numbered(
    prefix: str,
    rows: list[RobustRow],
    category: Callable[[RobustRow], str],
    **fields: Callable[[RobustRow], Any],
) -> tuple[SuiteRow, ...]:
    return tuple(
        _row(
            f"{prefix}-{index:04d}",
            row,
            category(row),
            **{name: make(row) for name, make in fields.items()},
        )
        for index, row in enumerate(rows, start=1)
    )


def _number_category(row: RobustRow) -> str:
    return "two_numbers" if row.kind == "number2" else row.numbers[0].kind


def _placeholder_category(row: RobustRow) -> str:
    return "reorder" if row.template == GENITIVE_TEMPLATE else "noun"


def _terms(row: RobustRow) -> tuple[TermPair, ...]:
    source, target = LANGUAGES.index(row.src_lang), LANGUAGES.index(row.tgt_lang)
    return tuple(TermPair(NOUNS[i][source], NOUNS[i][target]) for i in row.nouns)


def _short(language: str) -> str:
    return language.partition("_")[0]


def _script_section(toy: RobustToy) -> tuple[SuiteRow, ...]:
    base = _held_out(toy, "base")
    trained = tuple(
        SuiteRow(
            id=f"trained-{index:04d}",
            src_lang=row.src_lang,
            tgt_lang=row.tgt_lang,
            source=row.src,
            reference=None,
            category="trained",
        )
        for index, row in enumerate(_pair_rows(base), start=1)
    )
    untrained = tuple(
        SuiteRow(
            id=f"untrained-{_short(source)}-{_short(target)}-{index:03d}",
            src_lang=None,
            tgt_lang=target,
            source=row.src,
            reference=None,
            category="untrained",
        )
        for source, target in _UNTRAINED_PAIRS
        for index, row in enumerate(
            [row for row in base if row.src_lang == source][:_UNTRAINED_SOURCES],
            start=1,
        )
    )
    return trained + untrained


def _mong_section(toy: RobustToy) -> tuple[SuiteRow, ...]:
    rows = [
        row
        for row in _held_out(toy, "base", "entity", "placeholder")
        if row.src_lang == KHK and row.tgt_lang in (ENG, RUS)
    ]
    return _numbered(
        "mong",
        rows,
        lambda row: row.kind,
        source_mong=lambda row: transliterate(row.src),
    )


def _terms_section(toy: RobustToy) -> tuple[SuiteRow, ...]:
    return _numbered(
        "term",
        _pair_rows(_held_out(toy, "base")),
        lambda _: "glossary",
        terms=_terms,
    )


def build_main_suite(toy: RobustToy) -> Suite:
    """The six sections of the held-out rows of `toy` (see the module docstring)."""
    return Suite(
        name="robust-main",
        entities=EntityTable(
            {name: dict(by_lang) for name, by_lang in ENTITIES.items()}
        ),
        sections={
            "numbers": _numbered(
                "num", _held_out(toy, "number", "number2"), _number_category
            ),
            "placeholders": _numbered(
                "ph", _held_out(toy, "placeholder"), _placeholder_category
            ),
            "entities": _numbered(
                "ent",
                _held_out(toy, "entity"),
                lambda _: "person_city",
                entities=lambda row: row.entities,
            ),
            "terms": _terms_section(toy),
            "script": _script_section(toy),
            "mong": _mong_section(toy),
        },
    )


def build_term_control_suite(toy: RobustToy) -> Suite:
    """The terms section of the main suite with every glossary target replaced by a word the
    model never writes (`CONTROL_TERMS`): a system that is right about the real glossary
    scores 0 here."""
    rows = tuple(
        replace(
            row,
            terms=tuple(
                TermPair(pair.src, CONTROL_TERMS[row.tgt_lang]) for pair in row.terms
            ),
        )
        for row in _terms_section(toy)
    )
    return Suite(
        name="robust-term-control", entities=_NO_ENTITIES, sections={"terms": rows}
    )


def build_unseen_placeholder_suite() -> Suite:
    """One short sentence per index of `UNSEEN_INDICES` in each of four directions: the
    numbered placeholders no row of the toy trains (28 rows)."""
    rows = tuple(
        SuiteRow(
            id=f"unseen-{_short(source)}-{_short(target)}-{index}",
            src_lang=source,
            tgt_lang=target,
            source=_UNSEEN_SENTENCES[source].format(placeholder(index)),
            reference=_UNSEEN_SENTENCES[target].format(placeholder(index)),
            category="noun",
        )
        for source, target in _UNSEEN_DIRECTIONS
        for index in UNSEEN_INDICES
    )
    return Suite(
        name="robust-unseen-placeholders",
        entities=_NO_ENTITIES,
        sections={"placeholders": rows},
    )


def build_mong_output_suite(toy: RobustToy) -> Suite:
    """The first 100 held-out English to Khalkha base sentences, asked in traditional script."""
    english = [
        row
        for row in _held_out(toy, "base")
        if (row.src_lang, row.tgt_lang) == (ENG, KHK)
    ][:_MONG_OUTPUT_ROWS]
    rows = tuple(
        SuiteRow(
            id=f"mong-out-{index:04d}",
            src_lang=ENG,
            tgt_lang=KHK,
            source=row.src,
            reference=None,
            category="mong_output",
            script="Mong",
        )
        for index, row in enumerate(english, start=1)
    )
    return Suite(
        name="robust-mong-output", entities=_NO_ENTITIES, sections={"script": rows}
    )
