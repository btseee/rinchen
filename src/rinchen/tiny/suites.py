"""The suites of the tiny pipeline, made from the fixtures (D15, spec P2 and P5, findings F8 and F11).

`robustness_suite` is the suite `rinchen eval robustness` measures: its rows come from the KEPT TRAIN
rows only (the rows a model was trained on, which is what the gate's placeholders and numbers are
measured over), every direction of every kept pair. The sections are `numbers` and `placeholders` (a
row of that kind of the fixture), `entities` and `terms` (found by whole tokens, `suite_matching`),
`script` (the `base` rows, with no reference) and `mong` (the kept traditional-script rows: the
Khalkha source, its traditional spelling and the English reference). A row of a pair the filter
dropped is not in it, and a section that has no row is left out.

`memorised_suites` and `heldout_suites` are the evaluation suites, as plain data: the memorised ones
hold the same kept train rows (the model is asked for what it learned) and the held-out ones the
fixture's `test` rows, which no model trained on. A fixture suite needs one of the splits `dev`,
`devtest` and `test`; the profile names the split of each kind and the held-out one is never `dev` (a
`dev` run is not reported, and the pipeline selects nothing). The one module that writes an evaluation
suite file is `rinchen.eval.fixture_suites` and the robustness suite has its own writer: nothing here
reads a suite.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Final

from rinchen.config.eval_suites import HUB_LANGUAGE
from rinchen.eval.fixture_suites import (
    FixtureRow,
    FixtureSuite,
    fixture_suite_path,
    write_fixture_suite,
)
from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite import Suite, SuiteRow, TermPair, write_suite
from rinchen.tiny.fixtures import LANGUAGES, Fixtures, PairRow
from rinchen.tiny.suite_matching import EntityFinder, TermFinder

ROBUSTNESS_NAME: Final = "tiny-robust"
MEMORISED_NAMES: Final = (
    "tiny-memorised-eng",
    "tiny-memorised-rus",
    "tiny-memorised-mong",
)
HELDOUT_NAMES: Final = ("tiny-heldout-eng", "tiny-heldout-rus")
BENCHMARK: Final = "tiny"
MEMORISED_SPLIT: Final = "test"
HELDOUT_SPLIT: Final = "devtest"

_ENGLISH: Final = "eng_Latn"
_RUSSIAN: Final = "rus_Cyrl"
_MONGOLIAN: Final = "mn_Mong"
# The source language of each pair file, in the order of the suites.
_PAIR_LANGUAGES: Final = (_ENGLISH, _RUSSIAN)
_NUMBER_KINDS: Final = ("int", "decimal", "grouped")
_PLACEHOLDER_KIND: Final = "placeholder"
_ENTITY_KIND: Final = "entity"
_TERM_KIND: Final = "base"
_PREFIXES: Final = {
    "numbers": "num",
    "placeholders": "ph",
    "entities": "ent",
    "terms": "term",
    "script": "scr",
    "mong": "mong",
}
_MONG_ROW_PREFIX: Final = "mm"


def _pair_rows(fixtures: Fixtures, language: str) -> tuple[PairRow, ...]:
    return fixtures.pairs[f"{language}-{HUB_LANGUAGE}"]


def _kept_train_rows(
    fixtures: Fixtures, language: str, kept: frozenset[tuple[str, str]]
) -> Iterable[PairRow]:
    return (
        row
        for row in _pair_rows(fixtures, language)
        if row.split == "train" and (row.src, row.tgt) in kept
    )


class _Sections:
    """The rows of the sections of one suite, numbered as they are added."""

    def __init__(self) -> None:
        self._rows: dict[str, list[SuiteRow]] = {name: [] for name in _PREFIXES}

    def add(
        self,
        section: str,
        source_language: str,
        target_language: str,
        source: str,
        reference: str | None,
        category: str,
        *,
        entities: tuple[str, ...] = (),
        terms: tuple[TermPair, ...] = (),
        source_mong: str | None = None,
    ) -> None:
        rows = self._rows[section]
        identifier = f"{_PREFIXES[section]}-{len(rows) + 1:04d}"
        rows.append(
            SuiteRow(
                identifier,
                source_language,
                target_language,
                source,
                reference,
                category,
                entities=entities,
                terms=terms,
                source_mong=source_mong,
            )
        )

    def non_empty(self) -> dict[str, tuple[SuiteRow, ...]]:
        return {name: tuple(rows) for name, rows in self._rows.items() if rows}


class _Finders:
    """What finds the entities and the terms of a sentence."""

    def __init__(self, fixtures: Fixtures) -> None:
        self.entities = EntityFinder(fixtures.entities)
        self.terms = TermFinder(fixtures.nouns, LANGUAGES)


def _add_direction(
    sections: _Sections,
    finders: _Finders,
    kind: str,
    languages: tuple[str, str],
    texts: tuple[str, str],
) -> None:
    """The rows one direction of a kept train row gives: `languages` and `texts` are the source's
    and the target's."""
    (src, tgt), (source, reference) = languages, texts
    if kind in _NUMBER_KINDS:
        sections.add("numbers", src, tgt, source, reference, kind)
    elif kind == _PLACEHOLDER_KIND:
        sections.add("placeholders", src, tgt, source, reference, "noun")
    elif kind == _ENTITY_KIND:
        if ids := finders.entities.ids_in(source, src):
            sections.add(
                "entities", src, tgt, source, reference, "person_city", entities=ids
            )
    elif kind == _TERM_KIND:
        if pairs := finders.terms.pairs_in(source, src, tgt):
            sections.add(
                "terms",
                src,
                tgt,
                source,
                reference,
                "glossary",
                terms=tuple(TermPair(*pair) for pair in pairs),
            )
        sections.add("script", src, tgt, source, None, kind)


def robustness_suite(
    fixtures: Fixtures, kept_pairs: Collection[tuple[str, str]]
) -> Suite:
    """The robustness suite `tiny-robust` of the kept train rows of `fixtures`.

    `kept_pairs` are the `(source, target)` texts of the rows the filter kept: the source and the
    Khalkha text of a row of a pair file, the traditional-script text and the English text of a
    `mn_Mong` row. A row of the fixtures whose pair is not among them is not in the suite.
    """
    kept = frozenset(kept_pairs)
    finders = _Finders(fixtures)
    sections = _Sections()
    for language in _PAIR_LANGUAGES:
        for row in _kept_train_rows(fixtures, language, kept):
            for languages, texts in (
                ((language, HUB_LANGUAGE), (row.src, row.tgt)),
                ((HUB_LANGUAGE, language), (row.tgt, row.src)),
            ):
                _add_direction(sections, finders, row.kind, languages, texts)
    for mong in fixtures.mong:
        if mong.split == "train" and (mong.mong, mong.eng) in kept:
            sections.add(
                "mong",
                HUB_LANGUAGE,
                _ENGLISH,
                mong.khk,
                mong.eng,
                mong.kind,
                source_mong=mong.mong,
            )
    return Suite(
        name=ROBUSTNESS_NAME,
        entities=EntityTable(
            {
                entity: {language: tuple(names) for language, names in by.items()}
                for entity, by in fixtures.entities.items()
            }
        ),
        sections=sections.non_empty(),
    )


def _pair_suite(
    name: str, language: str, rows: Iterable[PairRow], split: str
) -> FixtureSuite:
    prefix = language[:2]
    return FixtureSuite(
        name=name,
        benchmark=BENCHMARK,
        split=split,
        directions=(f"{language}-{HUB_LANGUAGE}", f"{HUB_LANGUAGE}-{language}"),
        rows=tuple(
            FixtureRow(f"{prefix}-{row.id}", {language: row.src, HUB_LANGUAGE: row.tgt})
            for row in rows
        ),
    )


def memorised_suites(
    fixtures: Fixtures,
    kept_pairs: Collection[tuple[str, str]],
    *,
    split: str = MEMORISED_SPLIT,
) -> tuple[FixtureSuite, ...]:
    """The suites of what the model trained on, `tiny-memorised-eng`, `-rus` and `-mong`: the kept
    train rows of the two pair files (two directions each) and of the `mn_Mong` file (the one
    direction `mn_Mong-eng_Latn`), split `split`."""
    kept = frozenset(kept_pairs)
    made = [
        _pair_suite(name, language, _kept_train_rows(fixtures, language, kept), split)
        for name, language in zip(MEMORISED_NAMES[:2], _PAIR_LANGUAGES, strict=True)
    ]
    made.append(
        FixtureSuite(
            name=MEMORISED_NAMES[-1],
            benchmark=BENCHMARK,
            split=split,
            directions=(f"{_MONGOLIAN}-{_ENGLISH}",),
            rows=tuple(
                FixtureRow(
                    f"{_MONG_ROW_PREFIX}-{row.id}",
                    {_MONGOLIAN: row.mong, _ENGLISH: row.eng},
                )
                for row in fixtures.mong
                if row.split == "train" and (row.mong, row.eng) in kept
            ),
        )
    )
    return tuple(made)


def heldout_suites(
    fixtures: Fixtures, *, split: str = HELDOUT_SPLIT
) -> tuple[FixtureSuite, ...]:
    """The suites of what no model trained on, `tiny-heldout-eng` and `tiny-heldout-rus`: the
    `test` rows of the two pair files, two directions each, split `split` (a name for the report:
    nothing is selected or tuned on them)."""
    return tuple(
        _pair_suite(
            name,
            language,
            (row for row in _pair_rows(fixtures, language) if row.split == "test"),
            split,
        )
        for name, language in zip(HELDOUT_NAMES, _PAIR_LANGUAGES, strict=True)
    )


def write_suites(
    directory: Path,
    *,
    robustness: Suite | None = None,
    evaluation: Sequence[FixtureSuite] = (),
) -> dict[str, Path]:
    """Write the robustness suite (when given) and each evaluation suite as `<directory>/<name>.json`
    and return the path of each by suite name, in the order written. The directory is made.

    Raises:
        ValueError: two suites have the same name (nothing is written then).
        DataError: a name is not one a file can have, or an evaluation suite breaks the rules of its
            format; what was written before stays.
        ConfigError: the label of an evaluation suite is the name of a configured benchmark.
    """
    names = [spec.name for spec in evaluation]
    if robustness is not None:
        names.insert(0, robustness.name)
    if len(set(names)) != len(names):
        raise ValueError("two suites have the same name")
    paths: dict[str, Path] = {}
    if robustness is not None:
        path = fixture_suite_path(directory, robustness.name)
        directory.mkdir(parents=True, exist_ok=True)
        write_suite(robustness, path)
        paths[robustness.name] = path
    for spec in evaluation:
        write_fixture_suite(spec, directory)
        paths[spec.name] = fixture_suite_path(directory, spec.name)
    return paths
