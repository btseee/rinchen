"""Finding the entities and the glossary terms of a sentence of the fixtures by whole tokens (D15, spec P2,
brief G5.1).

A name or a noun is found when a WHOLE TOKEN of the sentence is it, never when it is a piece of a longer
word: the prototype matched by substring and found `Bat` in `Sukhbaatar`. The one allowance is a Khalkha
case suffix glued to a name, found by the longest known single-token spelling that is a proper prefix of
the token and has at least three characters. The matchers are checked against a brute-force reader on
generated sentences and on every sentence of the fixtures, and on sentences written to be traps.

No Cyrillic is typed into this file: a name is read from the fixtures and a suffix is made from its code
points.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from rinchen.tiny.fixtures import LANGUAGES, Fixtures
from rinchen.tiny.suite_matching import (
    EntityFinder,
    TermFinder,
    tokens_of,
)
from tests.support.tiny_stage_inputs import shipped_fixtures

KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
RUS: Final = "rus_Cyrl"
EDGE: Final = ".,?!:;"
# Khalkha case suffixes, by code point: d (locative), aas (ablative), iig, tai.
SUFFIXES: Final = (
    (0x434,),
    (0x430, 0x430, 0x441),
    (0x438, 0x439, 0x433),
    (0x442, 0x430, 0x439),
)


def suffixes() -> list[str]:
    return ["".join(chr(code) for code in codes) for codes in SUFFIXES]


def name(fixtures: Fixtures, entity: str, language: str = KHK) -> str:
    return fixtures.entities[entity][language][0]


def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


# ---- the tokens ---------------------------------------------------------------------------------


def test_a_sentence_is_split_into_folded_tokens_without_the_marks_at_their_edges() -> (
    None
):
    assert tokens_of("Bat, lives: in  Murun?!") == ["bat", "lives", "in", "murun"]
    assert tokens_of("  ") == []
    assert tokens_of("") == []
    assert tokens_of(".,?") == []


def test_a_token_is_folded_in_the_normal_form_and_without_case() -> None:
    decomposed = "Cafe" + chr(0x301)
    assert tokens_of(decomposed) == ["caf" + chr(0xE9)]
    assert tokens_of("STRASSE") == tokens_of("strasse")
    assert tokens_of(chr(0xDF)) == ["ss"]


def test_a_hyphen_inside_a_token_stays_and_one_at_the_edge_stays_too() -> None:
    assert tokens_of("Ulan-Bator is -here-.") == ["ulan-bator", "is", "-here-"]


# ---- the entities -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixtures() -> Fixtures:
    return shipped_fixtures()


@pytest.fixture(scope="module")
def finder(fixtures: Fixtures) -> EntityFinder:
    return EntityFinder(fixtures.entities)


def test_a_name_that_is_a_whole_token_is_found(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    for language in LANGUAGES:
        for entity in fixtures.entities:
            spelling = name(fixtures, entity, language)
            assert finder.ids_in(f"{spelling}.", language) == (entity,), (
                entity,
                language,
            )


def test_a_name_inside_a_longer_name_is_not_found(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    # In Russian the person `Bat` is a piece of both city names `Ulan-Bator` and `Sukhe-Bator`: the
    # prototype's substring match found it in every sentence that names either city.
    bat = name(fixtures, "bat", RUS)
    for city in ("sukhbaatar", "ulaanbaatar"):
        written = name(fixtures, city, RUS)
        assert bat.casefold() in written.casefold()

        assert finder.ids_in(f"Mother lives in {written}.", RUS) == (city,)
        assert finder.ids_in(f"{written} {written}", RUS) == (city,)
        assert finder.ids_in(f"{bat} {written}", RUS) == ("bat", city)
    # In Khalkha the suffixed city is not the person either.
    sukhbaatar = name(fixtures, "sukhbaatar")
    assert finder.ids_in(f"{sukhbaatar}{suffixes()[0]}.", KHK) == ("sukhbaatar",)
    assert finder.ids_in(f"{sukhbaatar}{suffixes()[1]}.", KHK) == ("sukhbaatar",)


def test_a_name_is_not_found_in_the_middle_or_at_the_end_of_a_longer_word(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    bat = name(fixtures, "bat", ENG)

    assert finder.ids_in(f"The {bat}man is here.", ENG) == ()
    assert finder.ids_in(f"The a{bat} is here.", ENG) == ()
    assert finder.ids_in(f"The {bat}s are here.", ENG) == ()


def test_the_case_suffix_of_khalkha_is_allowed_on_a_name(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    city = name(fixtures, "ulaanbaatar")
    person = name(fixtures, "bat")

    for suffix in suffixes():
        assert finder.ids_in(f"{person} {city}{suffix}.", KHK) == (
            "bat",
            "ulaanbaatar",
        )
        assert finder.ids_in(f"{person}{suffix}.", KHK) == ("bat",)


def test_the_case_suffix_is_for_khalkha_only(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    for language in (ENG, RUS):
        spelling = name(fixtures, "ulaanbaatar", language)
        assert finder.ids_in(f"{spelling}{suffixes()[0]}", language) == ()
        assert finder.ids_in(f"{spelling}x", language) == ()


def test_a_known_prefix_of_fewer_than_three_characters_is_not_a_name() -> None:
    entities = {"yan": {KHK: ("Yan",)}, "yo": {KHK: ("Yo",)}}
    short = EntityFinder(entities)

    assert short.ids_in("Yo.", KHK) == ("yo",)
    assert short.ids_in("Yos.", KHK) == ()
    assert short.ids_in("Yans.", KHK) == ("yan",)
    assert short.ids_in("Yan.", KHK) == ("yan",)


def test_the_longest_known_prefix_is_the_name_of_a_suffixed_token() -> None:
    entities = {"bat": {KHK: ("Bat",)}, "bator": {KHK: ("Bator",)}}
    both = EntityFinder(entities)

    assert both.ids_in("Bator.", KHK) == ("bator",)
    assert both.ids_in("Batort.", KHK) == ("bator",)
    assert both.ids_in("Batt.", KHK) == ("bat",)
    assert both.ids_in("Bat Bator", KHK) == ("bat", "bator")


def test_a_name_of_several_words_is_found_as_that_run_of_whole_tokens() -> None:
    entities = {"ulan": {ENG: ("Ulan Bator", "Ulaanbaatar")}}
    cities = EntityFinder(entities)

    assert cities.ids_in("He lives in Ulan Bator.", ENG) == ("ulan",)
    assert cities.ids_in("He lives in Ulan.", ENG) == ()
    assert cities.ids_in("He lives in Ulan Batory.", ENG) == ()
    assert cities.ids_in("Ulaanbaatar and Ulan  Bator", ENG) == ("ulan",)


def test_the_longest_run_of_tokens_is_the_name_and_it_covers_its_tokens() -> None:
    entities = {
        "short": {ENG: ("Ulan",)},
        "long": {ENG: ("Ulan Bator",)},
        "tail": {ENG: ("Bator",)},
    }
    names = EntityFinder(entities)

    assert names.ids_in("Ulan Bator", ENG) == ("long",)
    assert names.ids_in("Ulan", ENG) == ("short",)
    assert names.ids_in("Ulan and Bator", ENG) == ("short", "tail")
    assert names.ids_in("Bator Ulan Bator Ulan", ENG) == ("tail", "long", "short")


def test_the_ids_of_a_sentence_come_once_each_in_the_order_of_the_sentence(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    first, second = name(fixtures, "oyun"), name(fixtures, "darkhan")

    assert finder.ids_in(f"{second} {first} {second} {first}.", KHK) == (
        "darkhan",
        "oyun",
    )


def test_case_and_the_marks_at_the_edge_of_a_token_do_not_matter(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    spelling = name(fixtures, "murun", ENG)

    assert finder.ids_in(f"{spelling.upper()}?", ENG) == ("murun",)
    assert finder.ids_in(f"{spelling.lower()},", ENG) == ("murun",)


def test_an_unknown_language_or_an_empty_sentence_has_no_entity(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    assert finder.ids_in("anything", "deu_Latn") == ()
    assert finder.ids_in("", KHK) == ()
    assert finder.ids_in("... ,,,", KHK) == ()


def test_the_same_name_of_two_entities_is_the_first_ones() -> None:
    twin = EntityFinder({"a": {ENG: ("Same",)}, "b": {ENG: ("Same",)}})

    assert twin.ids_in("Same.", ENG) == ("a",)


def test_every_entity_row_of_the_fixtures_names_the_same_entities_on_both_sides(
    fixtures: Fixtures, finder: EntityFinder
) -> None:
    rows = 0
    for pair, pair_rows in fixtures.pairs.items():
        source_language = pair.split("-")[0]
        for row in pair_rows:
            if row.kind != "entity":
                continue
            rows += 1
            on_source = finder.ids_in(row.src, source_language)
            on_target = finder.ids_in(row.tgt, KHK)
            assert len(on_source) == 2, row.id
            assert set(on_source) == set(on_target), row.id
    assert rows == 2 * (30 + 1 + 3)


# ---- the terms ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def terms(fixtures: Fixtures) -> TermFinder:
    return TermFinder(fixtures.nouns, LANGUAGES)


def test_a_noun_that_is_a_whole_token_gives_its_pair(
    fixtures: Fixtures, terms: TermFinder
) -> None:
    khk, eng, rus = fixtures.nouns[0]

    assert terms.pairs_in(f"The {eng}.", ENG, KHK) == ((eng, khk),)
    assert terms.pairs_in(f"{khk}, {khk}!", KHK, RUS) == ((khk, rus),)
    assert terms.pairs_in(f"{rus}?", RUS, ENG) == ((rus, eng),)


def test_a_noun_inside_a_longer_word_is_not_a_term(
    fixtures: Fixtures, terms: TermFinder
) -> None:
    khk, eng, _ = fixtures.nouns[0]

    assert terms.pairs_in(f"The {eng}s and {eng}ish.", ENG, KHK) == ()
    assert terms.pairs_in(f"x{eng}.", ENG, KHK) == ()
    assert terms.pairs_in(f"{khk}{suffixes()[0]}.", KHK, ENG) == ()


def test_the_pairs_of_a_sentence_are_distinct_and_in_the_order_of_the_sentence(
    fixtures: Fixtures, terms: TermFinder
) -> None:
    (k0, e0, _), (k1, e1, _) = fixtures.nouns[0], fixtures.nouns[1]

    assert terms.pairs_in(f"{e1} and {e0} and {e1}.", ENG, KHK) == (
        (e1, k1),
        (e0, k0),
    )


def test_the_case_of_a_token_does_not_matter_and_the_pair_is_the_table_s(
    fixtures: Fixtures, terms: TermFinder
) -> None:
    khk, eng, _ = fixtures.nouns[2]

    assert terms.pairs_in(eng.upper() + ".", ENG, KHK) == ((eng, khk),)


def test_a_language_the_table_does_not_have_is_refused(terms: TermFinder) -> None:
    with pytest.raises(ValueError, match="language"):
        terms.pairs_in("book", "deu_Latn", KHK)
    with pytest.raises(ValueError, match="language"):
        terms.pairs_in("book", ENG, "deu_Latn")


def test_a_noun_row_of_a_wrong_width_is_refused() -> None:
    with pytest.raises(ValueError, match="width"):
        TermFinder([("a", "b")], LANGUAGES)


def test_every_base_row_of_the_fixtures_has_at_least_one_term_on_each_side(
    fixtures: Fixtures, terms: TermFinder
) -> None:
    base = 0
    for pair, pair_rows in fixtures.pairs.items():
        source_language = pair.split("-")[0]
        for row in pair_rows:
            if row.kind != "base":
                continue
            base += 1
            assert terms.pairs_in(row.src, source_language, KHK), row.id
            assert terms.pairs_in(row.tgt, KHK, source_language), row.id
    assert base == 2 * (150 + 8 + 15)


# ---- against a brute-force reader -----------------------------------------------------------------


def brute_tokens(text: str) -> list[str]:
    folded = [fold(word.strip(EDGE)) for word in text.split()]
    return [word for word in folded if word]


def brute_entities(
    text: str, language: str, entities: Mapping[str, Mapping[str, Sequence[str]]]
) -> tuple[str, ...]:
    """Every start position is tried against every spelling of every entity: an exact run of whole
    tokens (the longest wins) or, in Khalkha, a single known spelling of three or more characters that
    is a proper prefix of the token (the longest wins); the tokens a hit covers are skipped."""
    tokens = brute_tokens(text)
    found: list[str] = []
    position = 0
    while position < len(tokens):
        exact: tuple[int, str] | None = None
        prefix: tuple[int, str] | None = None
        for entity, by_language in entities.items():
            for spelling in by_language.get(language, ()):
                words = brute_tokens(spelling)
                if not words:
                    continue
                if tokens[position : position + len(words)] == words and (
                    exact is None or len(words) > exact[0]
                ):
                    exact = (len(words), entity)
                if (
                    language == KHK
                    and len(words) == 1
                    and len(words[0]) >= 3
                    and tokens[position].startswith(words[0])
                    and len(tokens[position]) > len(words[0])
                    and (prefix is None or len(words[0]) > prefix[0])
                ):
                    prefix = (len(words[0]), entity)
        if exact is not None:
            hit, width = exact[1], exact[0]
        elif prefix is not None:
            hit, width = prefix[1], 1
        else:
            position += 1
            continue
        if hit not in found:
            found.append(hit)
        position += width
    return tuple(found)


def brute_terms(
    text: str, source: str, target: str, nouns: Sequence[Sequence[str]]
) -> tuple[tuple[str, str], ...]:
    tokens = brute_tokens(text)
    column = {language: index for index, language in enumerate(LANGUAGES)}
    found: list[tuple[str, str]] = []
    for token in tokens:
        for row in nouns:
            if fold(row[column[source]]) == token:
                pair = (row[column[source]], row[column[target]])
                if pair not in found:
                    found.append(pair)
                break
    return tuple(found)


def generated_sentences(fixtures: Fixtures, count: int) -> list[tuple[str, str]]:
    """Sentences of the three languages made of names, nouns, suffixed and damaged names, hyphens and
    marks: (language, sentence)."""
    rng = random.Random(20261112)
    words: dict[str, list[str]] = {language: [] for language in LANGUAGES}
    for by_language in fixtures.entities.values():
        for language, spellings in by_language.items():
            words[language].extend(spellings)
    for row in fixtures.nouns:
        for language, word in zip(LANGUAGES, row, strict=True):
            words[language].append(word)
    plain = ["and", "lives", "in", "is", "from", "x", "ab"]
    makers = [
        lambda word: word,
        lambda word: word.upper(),
        lambda word: word.lower(),
        lambda word: word + rng.choice(suffixes()),
        lambda word: word + "x",
        lambda word: "x" + word,
        lambda word: word[:-1],
        lambda word: word[:2],
        lambda word: word + "-" + word,
        lambda word: word + rng.choice([",", ".", "?", "!", ":", ";"]),
    ]
    sentences: list[tuple[str, str]] = []
    for _ in range(count):
        language = rng.choice(LANGUAGES)
        pieces = []
        for _ in range(rng.randint(1, 7)):
            if rng.random() < 0.2:
                pieces.append(rng.choice(plain))
            else:
                pieces.append(rng.choice(makers)(rng.choice(words[language])))
        sentences.append((language, " ".join(pieces)))
    return sentences


def fixture_sentences(fixtures: Fixtures) -> list[tuple[str, str]]:
    sentences: list[tuple[str, str]] = []
    for pair, rows in fixtures.pairs.items():
        source_language = pair.split("-")[0]
        for row in rows:
            sentences.append((source_language, row.src))
            sentences.append((KHK, row.tgt))
    for row in fixtures.mong:
        sentences.append((KHK, row.khk))
        sentences.append((ENG, row.eng))
    for row in fixtures.mono:
        sentences.append((KHK, row.khk))
        sentences.append((ENG, row.eng))
    return sentences


def test_the_matchers_agree_with_the_brute_force_reader_on_generated_sentences(
    fixtures: Fixtures, finder: EntityFinder, terms: TermFinder
) -> None:
    sentences = generated_sentences(fixtures, 1000)
    hits = 0

    for language, sentence in sentences:
        expected = brute_entities(sentence, language, fixtures.entities)
        assert finder.ids_in(sentence, language) == expected, (language, sentence)
        hits += bool(expected)
        for target in LANGUAGES:
            if target != language:
                assert terms.pairs_in(sentence, language, target) == brute_terms(
                    sentence, language, target, fixtures.nouns
                ), (language, target, sentence)

    assert len(sentences) == 1000
    # The generator makes traps and hits both: a reader that finds nothing would pass otherwise.
    assert 300 < hits < 1000


def test_the_matchers_agree_with_the_brute_force_reader_on_every_sentence_of_the_fixtures(
    fixtures: Fixtures, finder: EntityFinder, terms: TermFinder
) -> None:
    sentences = fixture_sentences(fixtures)
    found = 0

    for language, sentence in sentences:
        expected = brute_entities(sentence, language, fixtures.entities)
        assert finder.ids_in(sentence, language) == expected
        found += bool(expected)
        for target in LANGUAGES:
            if target != language:
                assert terms.pairs_in(sentence, language, target) == brute_terms(
                    sentence, language, target, fixtures.nouns
                )

    assert len(sentences) == 2 * (345 + 345) + 2 * 70 + 2 * 64
    # Both sides of every entity row of both pair files name entities; the reader finds them all.
    assert found >= 2 * 2 * (30 + 1 + 3)
