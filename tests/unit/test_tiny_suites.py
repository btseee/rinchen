"""The robustness suite and the evaluation suites of the tiny pipeline, made from the fixtures (D15, spec
P2 and P5, findings F8 and F11, brief G5.1 and G5.2).

A robustness suite is made of the KEPT TRAIN rows only (the rows a model was trained on, which is what
the placeholders and the numbers of the gate are measured over): every direction of every kept pair,
the entities and the glossary terms found by whole tokens. The memorised evaluation suites hold the same
kept train rows and are split `test`; the held-out ones hold the fixture's `test` rows and are split
`devtest`; no suite is split `dev`, because the pipeline never selects. What the suites write is read
back here with the readers, which only a test may use.

Sentences are never typed into this file: they come from the fixtures.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from rinchen.corpus.readers import scan_corpus
from rinchen.eval import fixture_suites, robust_suite, suites
from rinchen.tiny import suite_matching
from rinchen.tiny import suites as tiny_suites
from rinchen.tiny.fixtures import Fixtures
from rinchen.tiny.suite_matching import EntityFinder, tokens_of
from rinchen.tiny.suites import (
    heldout_suites,
    memorised_suites,
    robustness_suite,
    write_suites,
)
from tests.support import tiny_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_stage_inputs import shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

library_outputs = tiny_run.library_outputs

KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
RUS: Final = "rus_Cyrl"
MONG: Final = "mn_Mong"
SECTIONS: Final = ("numbers", "placeholders", "entities", "terms", "script", "mong")
COUNTS: Final = {
    "numbers": 240,
    "placeholders": 240,
    "entities": 120,
    "terms": 600,
    "script": 600,
    "mong": 50,
}
PREFIX: Final = {
    "numbers": "num",
    "placeholders": "ph",
    "entities": "ent",
    "terms": "term",
    "script": "scr",
    "mong": "mong",
}


@pytest.fixture(scope="module")
def fixtures() -> Fixtures:
    return shipped_fixtures()


def train_pairs(fixtures: Fixtures) -> set[tuple[str, str]]:
    """Every train pair of the fixtures, the way the filtered corpus keys it: the source and the
    Khalkha text of a pair row, the traditional and the English text of a `mn_Mong` row."""
    pairs = {
        (row.src, row.tgt)
        for rows in fixtures.pairs.values()
        for row in rows
        if row.split == "train"
    }
    pairs |= {(row.mong, row.eng) for row in fixtures.mong if row.split == "train"}
    return pairs


@pytest.fixture(scope="module")
def everything(fixtures: Fixtures) -> frozenset[tuple[str, str]]:
    return frozenset(train_pairs(fixtures))


@pytest.fixture(scope="module")
def robust(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> robust_suite.Suite:
    return robustness_suite(fixtures, everything)


def pair_row_of(fixtures: Fixtures) -> dict[str, tuple[str, str, str]]:
    """The pair row (source language, its split and its kind) of each sentence of a pair file."""
    found: dict[str, tuple[str, str, str]] = {}
    for pair, rows in fixtures.pairs.items():
        source_language = pair.split("-")[0]
        for row in rows:
            found[row.src] = (source_language, row.split, row.kind)
            found[row.tgt] = (KHK, row.split, row.kind)
    return found


# ---- G5.1: the robustness suite over the fixtures ----------------------------------------------------


def test_the_suite_has_the_name_and_the_sections_of_the_fixtures(
    robust: robust_suite.Suite,
) -> None:
    assert robust.name == "tiny-robust"
    assert robust.row_counts() == COUNTS
    assert tuple(robust.sections) == SECTIONS


def test_the_entity_rows_hold_two_hundred_and_forty_mentions(
    robust: robust_suite.Suite,
) -> None:
    mentions = sum(len(row.entities) for row in robust.sections["entities"])

    assert mentions == 240
    assert all(len(row.entities) == 2 for row in robust.sections["entities"])


def test_the_ids_are_sequential_in_each_section(robust: robust_suite.Suite) -> None:
    for section, rows in robust.sections.items():
        assert [row.id for row in rows] == [
            f"{PREFIX[section]}-{number:04d}" for number in range(1, len(rows) + 1)
        ], section


def test_every_kept_pair_gives_a_row_in_both_directions(
    robust: robust_suite.Suite,
) -> None:
    for section in ("numbers", "placeholders", "entities", "terms", "script"):
        by_direction: dict[tuple[str | None, str], int] = {}
        for row in robust.sections[section]:
            key = (row.src_lang, row.tgt_lang)
            by_direction[key] = by_direction.get(key, 0) + 1
        assert by_direction == {
            (ENG, KHK): COUNTS[section] // 4,
            (KHK, ENG): COUNTS[section] // 4,
            (RUS, KHK): COUNTS[section] // 4,
            (KHK, RUS): COUNTS[section] // 4,
        }, section


def test_a_row_and_its_reverse_hold_the_same_two_sentences(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    pairs = {
        frozenset((row.src, row.tgt))
        for rows in fixtures.pairs.values()
        for row in rows
        if row.kind == "placeholder" and row.split == "train"
    }
    seen = {
        frozenset((row.source, row.reference))
        for row in robust.sections["placeholders"]
    }

    assert seen == pairs
    assert len(pairs) == 120


def test_the_categories_are_the_kinds_of_the_fixtures(
    robust: robust_suite.Suite,
) -> None:
    def categories(section: str) -> set[str]:
        return {row.category for row in robust.sections[section]}

    assert categories("numbers") == {"int", "decimal", "grouped"}
    assert categories("placeholders") == {"noun"}
    assert categories("entities") == {"person_city"}
    assert categories("terms") == {"glossary"}
    assert categories("script") == {"base"}
    assert categories("mong") == {"base"}


def test_only_train_rows_are_in_the_suite(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    sentences = pair_row_of(fixtures)
    for section in ("numbers", "placeholders", "entities", "terms", "script"):
        for row in robust.sections[section]:
            language, split, _ = sentences[row.source]
            assert split == "train", (section, row.id)
            assert language == row.src_lang
            if row.reference is not None:
                assert sentences[row.reference][1] == "train"
    for row in robust.sections["mong"]:
        assert sentences[row.source][1] == "train"


def test_the_numbers_and_placeholders_rows_have_their_reference(
    robust: robust_suite.Suite,
) -> None:
    for section in ("numbers", "placeholders", "mong"):
        assert all(row.reference for row in robust.sections[section]), section
    assert all(row.reference is None for row in robust.sections["script"])
    assert all(row.script is None for row in robust.sections["script"])


def test_the_entity_ids_of_a_row_are_the_whole_token_names_of_its_source(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    finder = EntityFinder(fixtures.entities)
    sentences = pair_row_of(fixtures)
    pair_of = {row.src: row.tgt for rows in fixtures.pairs.values() for row in rows} | {
        row.tgt: row.src for rows in fixtures.pairs.values() for row in rows
    }

    for row in robust.sections["entities"]:
        assert row.src_lang is not None
        found = finder.ids_in(row.source, row.src_lang)
        assert row.entities == found
        # The other side of the same pair names the same two entities, so none was found by chance.
        other = finder.ids_in(pair_of[row.source], sentences[pair_of[row.source]][0])
        assert set(other) == set(row.entities)


def test_a_name_that_is_part_of_another_name_is_not_a_second_entity(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    # In Russian the person `Bat` is a piece of the city names `Ulan-Bator` and `Sukhe-Bator`.
    bat = fixtures.entities["bat"][RUS][0].casefold()
    cities = {
        fixtures.entities[city][RUS][0].casefold()
        for city in ("ulaanbaatar", "sukhbaatar")
    }
    checked = 0
    for row in robust.sections["entities"]:
        if row.src_lang != RUS:
            continue
        tokens = set(tokens_of(row.source))
        if tokens & cities and bat not in tokens:
            checked += 1
            assert "bat" not in row.entities
    # Three rows of the fixtures (a digest pins them) name such a city and not the person.
    assert checked == 3


def test_the_terms_of_a_row_are_whole_token_nouns_of_its_two_sides(
    robust: robust_suite.Suite,
) -> None:
    for row in robust.sections["terms"]:
        assert row.terms
        assert row.reference is not None
        source, reference = set(tokens_of(row.source)), set(tokens_of(row.reference))
        for pair in row.terms:
            assert pair.src.casefold() in source, row.id
            assert pair.tgt.casefold() in reference, row.id


def test_the_entity_table_is_the_one_of_the_fixtures(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    assert {
        entity: {language: tuple(spellings) for language, spellings in by.items()}
        for entity, by in robust.entities.spellings.items()
    } == {
        entity: {language: tuple(spellings) for language, spellings in by.items()}
        for entity, by in fixtures.entities.items()
    }


def test_the_mong_rows_are_the_kept_traditional_script_train_rows(
    fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    train = [row for row in fixtures.mong if row.split == "train"]

    assert [
        (row.source, row.source_mong, row.reference, row.category)
        for row in robust.sections["mong"]
    ] == [(row.khk, row.mong, row.eng, row.kind) for row in train]
    assert all(
        (row.src_lang, row.tgt_lang) == (KHK, ENG) for row in robust.sections["mong"]
    )


def test_the_suite_is_written_and_read_back_by_the_reader_of_the_robustness_suite(
    tmp_path: Path, robust: robust_suite.Suite
) -> None:
    paths = write_suites(tmp_path, robustness=robust)
    read, digest = robust_suite.read_suite(paths["tiny-robust"])

    assert read.to_bytes() == robust.to_bytes()
    assert read.row_counts() == COUNTS
    assert digest == robust_suite.write_suite(robust, tmp_path / "again.json")


def test_the_rows_of_a_pair_the_filter_dropped_are_not_in_the_suite(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    placeholder_rows = [
        row
        for row in fixtures.pairs["eng_Latn-khk_Cyrl"]
        if row.kind == "placeholder" and row.split == "train"
    ]
    dropped = placeholder_rows[0]

    suite = robustness_suite(fixtures, everything - {(dropped.src, dropped.tgt)})

    assert suite.row_counts() == {**COUNTS, "placeholders": 238}
    assert dropped.src not in {row.source for row in suite.sections["placeholders"]}
    assert dropped.tgt not in {row.source for row in suite.sections["placeholders"]}
    # The ids stay sequential after a row is gone.
    assert [row.id for row in suite.sections["placeholders"]][-1] == "ph-0238"


@pytest.mark.parametrize("kind", ["base", "int", "entity", "placeholder"])
def test_a_dropped_pair_removes_its_rows_from_the_matching_sections_only(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]], kind: str
) -> None:
    row = next(
        item
        for item in fixtures.pairs["rus_Cyrl-khk_Cyrl"]
        if item.kind == kind and item.split == "train"
    )

    suite = robustness_suite(fixtures, everything - {(row.src, row.tgt)})

    expected = dict(COUNTS)
    for section in {
        "base": ("terms", "script"),
        "int": ("numbers",),
        "entity": ("entities",),
        "placeholder": ("placeholders",),
    }[kind]:
        expected[section] -= 2
    assert suite.row_counts() == expected


def test_a_dropped_traditional_script_row_leaves_the_mong_section(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    row = next(item for item in fixtures.mong if item.split == "train")

    suite = robustness_suite(fixtures, everything - {(row.mong, row.eng)})

    assert suite.row_counts() == {**COUNTS, "mong": 49}


def test_a_pair_of_the_dev_or_test_split_is_never_in_the_suite_even_when_kept(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    held_out = {
        (row.src, row.tgt)
        for rows in fixtures.pairs.values()
        for row in rows
        if row.split != "train"
    }

    suite = robustness_suite(fixtures, everything | held_out)

    assert suite.row_counts() == COUNTS


def test_without_a_kept_pair_there_is_no_section(fixtures: Fixtures) -> None:
    assert dict(robustness_suite(fixtures, []).sections) == {}


def test_a_base_row_with_no_noun_is_a_script_row_and_no_terms_row(
    fixtures: Fixtures,
) -> None:
    pair = "eng_Latn-khk_Cyrl"
    rows = list(fixtures.pairs[pair])
    index = next(n for n, row in enumerate(rows) if row.kind == "base")
    rows[index] = dataclasses.replace(rows[index], src="Nothing is here.", tgt="Zzz.")
    altered = dataclasses.replace(fixtures, pairs={**fixtures.pairs, pair: tuple(rows)})

    suite = robustness_suite(altered, train_pairs(altered))

    counts = suite.row_counts()
    assert counts["script"] == COUNTS["script"]
    assert counts["terms"] == COUNTS["terms"] - 2
    assert {row.source for row in suite.sections["script"]} >= {
        "Nothing is here.",
        "Zzz.",
    }
    assert "Zzz." not in {row.source for row in suite.sections["terms"]}


def test_a_list_of_rows_given_to_a_suite_is_not_changed_by_it(
    fixtures: Fixtures,
) -> None:
    kept = list(train_pairs(fixtures))

    robustness_suite(fixtures, kept)

    assert sorted(kept) == sorted(train_pairs(fixtures))


@pytest.mark.parametrize("make", [list, set, frozenset, tuple])
def test_the_kept_pairs_may_be_any_collection(
    fixtures: Fixtures,
    everything: frozenset[tuple[str, str]],
    make: type[Collection[tuple[str, str]]],
) -> None:
    assert robustness_suite(fixtures, make(everything)).row_counts() == COUNTS


def test_the_suite_does_not_change_the_fixtures(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    def snapshot() -> object:
        return (
            [(pair, rows) for pair, rows in fixtures.pairs.items()],
            fixtures.mong,
            fixtures.mono,
            fixtures.noise,
            fixtures.nouns,
            {
                entity: {language: spellings for language, spellings in by.items()}
                for entity, by in fixtures.entities.items()
            },
        )

    before = snapshot()

    robustness_suite(fixtures, everything)
    memorised_suites(fixtures, everything)
    heldout_suites(fixtures)

    assert snapshot() == before


def kept_pairs_of(directory: Path) -> set[tuple[str, str]]:
    kept = scan_corpus(directory / "03_filter" / "kept.parquet").collect()
    train = kept.filter(pl.col("split").cast(pl.String) == "train")
    return set(zip(train["src"].to_list(), train["tgt"].to_list(), strict=True))


def test_the_suite_of_the_rows_the_real_filter_kept_has_at_most_the_rows_of_all(
    fixtures: Fixtures, library_outputs: Path
) -> None:
    kept = kept_pairs_of(library_outputs)

    suite = robustness_suite(fixtures, kept)

    counts = suite.row_counts()
    assert 0 < counts["placeholders"] <= 240
    assert all(counts[section] <= COUNTS[section] for section in COUNTS)
    # The filter flags some authentic train rows: the suite is smaller than the one of every row.
    assert sum(COUNTS.values()) - sum(counts.values()) > 0


# ---- G5.2: the evaluation suites ----------------------------------------------------------------------


def rows_of(spec: fixture_suites.FixtureSuite) -> list[tuple[str, dict[str, str]]]:
    return [(row.id, dict(row.text)) for row in spec.rows]


def test_the_memorised_suites_hold_the_kept_train_rows(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    english, russian, mong = memorised_suites(fixtures, everything)

    assert [spec.name for spec in (english, russian, mong)] == [
        "tiny-memorised-eng",
        "tiny-memorised-rus",
        "tiny-memorised-mong",
    ]
    assert (len(english.rows), len(russian.rows), len(mong.rows)) == (300, 300, 50)
    assert english.directions == (f"{ENG}-{KHK}", f"{KHK}-{ENG}")
    assert russian.directions == (f"{RUS}-{KHK}", f"{KHK}-{RUS}")
    assert mong.directions == (f"{MONG}-{ENG}",)
    for spec, languages in (
        (english, {ENG, KHK}),
        (russian, {RUS, KHK}),
        (mong, {MONG, ENG}),
    ):
        assert all(set(row.text) == languages for row in spec.rows), spec.name
        assert (spec.benchmark, spec.split) == ("tiny", "test")


def test_the_memorised_texts_are_the_fixture_rows(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    english, russian, mong = memorised_suites(fixtures, everything)
    train = [row for row in fixtures.pairs["eng_Latn-khk_Cyrl"] if row.split == "train"]

    assert [(row.id, dict(row.text)) for row in english.rows[: len(train)]] == [
        (f"en-{item.id}", {ENG: item.src, KHK: item.tgt}) for item in train
    ]
    assert english.rows[0].id == "en-ek-0001"
    assert russian.rows[0].id == "ru-rk-0001"
    assert mong.rows[0].id == "mm-mm-0001"
    assert [row.doc for row in english.rows] == [None] * 300


def test_a_pair_the_filter_dropped_is_not_in_a_memorised_suite(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    english_row = next(
        row for row in fixtures.pairs["eng_Latn-khk_Cyrl"] if row.split == "train"
    )
    mong_row = next(row for row in fixtures.mong if row.split == "train")
    kept = everything - {
        (english_row.src, english_row.tgt),
        (mong_row.mong, mong_row.eng),
    }

    english, russian, mong = memorised_suites(fixtures, kept)

    assert (len(english.rows), len(russian.rows), len(mong.rows)) == (299, 300, 49)
    assert f"en-{english_row.id}" not in {row.id for row in english.rows}
    assert f"mm-{mong_row.id}" not in {row.id for row in mong.rows}


def test_the_held_out_suites_hold_the_test_rows_of_the_fixtures(
    fixtures: Fixtures,
) -> None:
    english, russian = heldout_suites(fixtures)

    assert [spec.name for spec in (english, russian)] == [
        "tiny-heldout-eng",
        "tiny-heldout-rus",
    ]
    assert (len(english.rows), len(russian.rows)) == (30, 30)
    for spec, pair, source in (
        (english, "eng_Latn-khk_Cyrl", ENG),
        (russian, "rus_Cyrl-khk_Cyrl", RUS),
    ):
        expected = [row for row in fixtures.pairs[pair] if row.split == "test"]
        assert [(row.id, dict(row.text)) for row in spec.rows] == [
            (f"{source[:2]}-{item.id}", {source: item.src, KHK: item.tgt})
            for item in expected
        ]
        assert spec.directions == (f"{source}-{KHK}", f"{KHK}-{source}")
        assert (spec.benchmark, spec.split) == ("tiny", "devtest")


def test_the_held_out_rows_are_no_train_rows(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    trained = {sentence for pair in everything for sentence in pair}
    for spec in heldout_suites(fixtures):
        for row in spec.rows:
            assert not set(row.text.values()) & trained, row.id


def test_the_splits_follow_the_profile_and_the_default_is_the_shipped_one(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    assert {spec.split for spec in memorised_suites(fixtures, everything)} == {"test"}
    assert {spec.split for spec in heldout_suites(fixtures)} == {"devtest"}
    assert {
        spec.split for spec in memorised_suites(fixtures, everything, split="devtest")
    } == {"devtest"}
    assert {spec.split for spec in heldout_suites(fixtures, split="test")} == {"test"}


def test_no_suite_the_pipeline_makes_is_split_dev(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    every = [*memorised_suites(fixtures, everything), *heldout_suites(fixtures)]

    assert len(every) == 5
    assert {spec.split for spec in every} == {"test", "devtest"}


def test_the_ids_of_a_suite_are_unique_and_the_suites_are_the_same_on_a_second_call(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    first = [*memorised_suites(fixtures, everything), *heldout_suites(fixtures)]
    second = [*memorised_suites(fixtures, everything), *heldout_suites(fixtures)]

    assert first == second
    for spec in first:
        ids = [row.id for row in spec.rows]
        assert len(set(ids)) == len(ids), spec.name


def test_every_suite_is_written_as_a_fixture_suite_the_reader_accepts(
    tmp_path: Path, fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    specs = [*memorised_suites(fixtures, everything), *heldout_suites(fixtures)]

    paths = write_suites(tmp_path, evaluation=specs)

    assert list(paths) == [spec.name for spec in specs]
    for spec in specs:
        read, _ = suites.read_suite(paths[spec.name])
        assert paths[spec.name] == tmp_path / f"{spec.name}.json"
        assert (read.kind, read.benchmark, read.split) == (
            "fixture",
            "tiny",
            spec.split,
        )
        assert read.directions == tuple(spec.directions)
        assert [(row.id, dict(row.text)) for row in read.rows] == rows_of(spec)
        assert read.split in ("test", "devtest")
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        f"{spec.name}.json" for spec in specs
    )


def test_the_directions_of_the_memorised_suites_are_the_five_the_model_is_trained_for(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    directions = [
        direction
        for spec in memorised_suites(fixtures, everything)
        for direction in spec.directions
    ]

    assert directions == [
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "mn_Mong-eng_Latn",
    ]


def test_the_memorised_suites_of_the_rows_the_real_filter_kept(
    fixtures: Fixtures, library_outputs: Path
) -> None:
    english, russian, mong = memorised_suites(fixtures, kept_pairs_of(library_outputs))

    assert len(russian.rows) == 300
    assert 290 <= len(english.rows) <= 300
    assert 45 <= len(mong.rows) <= 50
    assert len(english.rows) + len(mong.rows) < 350


# ---- writing -------------------------------------------------------------------------------------------


def test_the_directory_is_made_and_both_kinds_of_suite_are_written_into_it(
    tmp_path: Path,
    fixtures: Fixtures,
    everything: frozenset[tuple[str, str]],
    robust: robust_suite.Suite,
) -> None:
    target = tmp_path / "one" / "two"

    paths = write_suites(
        target,
        robustness=robust,
        evaluation=heldout_suites(fixtures),
    )

    assert list(paths) == ["tiny-robust", "tiny-heldout-eng", "tiny-heldout-rus"]
    assert sorted(path.name for path in target.iterdir()) == sorted(
        f"{name}.json" for name in paths
    )
    assert all(path.parent == target for path in paths.values())


def test_nothing_is_written_when_there_is_nothing_to_write(tmp_path: Path) -> None:
    assert write_suites(tmp_path / "out") == {}


def test_two_suites_of_one_name_are_refused_and_nothing_is_written(
    tmp_path: Path, fixtures: Fixtures, robust: robust_suite.Suite
) -> None:
    spec = heldout_suites(fixtures)[0]
    same_as_the_robust_one = dataclasses.replace(spec, name=robust.name)

    with pytest.raises(ValueError, match="same name"):
        write_suites(tmp_path / "one", evaluation=[spec, spec])
    with pytest.raises(ValueError, match="same name"):
        write_suites(
            tmp_path / "two", robustness=robust, evaluation=[same_as_the_robust_one]
        )

    assert not (tmp_path / "one").exists()
    assert not (tmp_path / "two").exists()


# ---- what the module imports ------------------------------------------------------------------------------


def eval_imports(module: object) -> set[str]:
    path = Path(getattr(module, "__file__", ""))
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name for name in found if name.startswith("rinchen.eval")}


def test_the_suites_import_the_two_writers_and_the_span_table_and_no_reader() -> None:
    allowed = (
        "rinchen.eval.fixture_suites",
        "rinchen.eval.robust_spans",
        "rinchen.eval.robust_suite",
    )
    imported = eval_imports(tiny_suites)

    assert imported
    for name in imported - {"rinchen.eval"}:
        assert any(name == own or name.startswith(f"{own}.") for own in allowed), name
    assert {"rinchen.eval.fixture_suites", "rinchen.eval.robust_suite"} <= imported


def test_the_matching_module_imports_nothing_of_the_eval_package() -> None:
    assert eval_imports(suite_matching) == set()


def test_a_list_of_names_the_stages_use_is_the_suites_the_module_makes(
    fixtures: Fixtures, everything: frozenset[tuple[str, str]]
) -> None:
    made: Sequence[str] = [
        spec.name
        for spec in (*memorised_suites(fixtures, everything), *heldout_suites(fixtures))
    ]

    assert made == [
        *tiny_suites.MEMORISED_NAMES,
        *tiny_suites.HELDOUT_NAMES,
    ]
    assert tiny_suites.ROBUSTNESS_NAME == "tiny-robust"
