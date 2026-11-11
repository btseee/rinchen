"""Property tests of the planted encoder of the tiny pipeline (D15, `rinchen.tiny.fakes`, brief G3.4),
300 generated cases each, over the words of the committed fixtures.

The encoder reads a text as a bag of concepts, so (1) any permutation of the words of a sentence
embeds to the same vector, exactly, because the bag is summed in one order; (2) the separators of a
number are not part of it; and (3) a sentence with one concept replaced by another is further from
the original. How much further is arithmetic: the vectors are random, so a bag of one concept
replaced by another stays below the filter's 0.7 bound for every pair of the lexicon (exhaustive
maximum 0.446), while a bag of two concepts reaches 0.83 and of three 0.90 for the worst
replacements of the lexicon (about 1 % and 35 % of them are at or above 0.7). The brief's statement
that a replacement is below 0.7 "when the bag has at most three concepts" is false for the second
and third, so those bags are held to the bound that is true over the whole lexicon, 0.95.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.tiny.fakes import ConceptEncoder
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_fakes_reference import (
    SUFFIXES,
    concept_spellings,
    unit_pools,
)
from tests.support.tiny_stage_inputs import shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
FIXTURES: Final = shipped_fixtures()
ENCODER: Final = ConceptEncoder(FIXTURES.nouns, FIXTURES.entities)
POOLS: Final = unit_pools(FIXTURES.nouns, FIXTURES.entities)
SPELLINGS: Final = concept_spellings(FIXTURES.nouns, FIXTURES.entities)
CONCEPTS: Final = sorted(SPELLINGS)
SINGLE_BOUND: Final = 0.7
SEVERAL_BOUND: Final = 0.95
DIGITS: Final = "0123456789"


def embed_one(text: str) -> np.ndarray:
    return ENCODER.embed([text])[0]


@st.composite
def sentences(draw: st.DrawFn) -> tuple[str, list[str]]:
    """A separator and the words of a sentence. A number is drawn only when the separator is not a
    bare space: two numbers next to each other with a space between them read as one grouped
    number, so their order is not free."""
    separator = draw(st.sampled_from([" ", ", ", "; ", " - "]))
    kinds = ["word", "suffixed", "placeholder", "unknown"]
    if separator != " ":
        kinds.append("number")
    words = []
    for _ in range(draw(st.integers(min_value=1, max_value=8))):
        kind = draw(st.sampled_from(kinds))
        word = draw(st.sampled_from(POOLS[kind]))
        if kind == "suffixed":
            word += draw(st.sampled_from(SUFFIXES))
        words.append(word)
    return separator, words


@SETTINGS
@given(data=st.data(), sentence=sentences())
def test_any_permutation_of_the_words_of_a_sentence_embeds_to_the_same_vector(
    data: st.DataObject, sentence: tuple[str, list[str]]
) -> None:
    separator, words = sentence
    shuffled = data.draw(st.permutations(words))

    assert np.array_equal(
        embed_one(separator.join(words)), embed_one(separator.join(shuffled))
    )


@st.composite
def grouped_numbers(draw: st.DrawFn) -> tuple[str, str]:
    """A number with group and decimal separators, and its digits."""
    head = draw(st.text(alphabet=DIGITS, min_size=1, max_size=4))
    groups = draw(
        st.lists(
            st.tuples(
                st.sampled_from(" .,"),
                st.text(alphabet=DIGITS, min_size=1, max_size=3),
            ),
            max_size=3,
        )
    )
    written = head + "".join(separator + group for separator, group in groups)
    return written, head + "".join(group for _, group in groups)


@SETTINGS
@given(number=grouped_numbers(), before=st.sampled_from(["", "x ", "n: "]))
def test_the_separators_of_a_number_are_not_part_of_it(
    number: tuple[str, str], before: str
) -> None:
    written, digits = number

    assert np.array_equal(
        embed_one(f"{before}{written}."), embed_one(f"{before}{digits}")
    )


@st.composite
def replacements(draw: st.DrawFn) -> tuple[int, str, str]:
    """A sentence of one to three concepts, each in some spelling of the lexicon, and the same
    sentence with one concept replaced by another."""
    size = draw(st.integers(min_value=1, max_value=3))
    chosen = draw(
        st.lists(
            st.sampled_from(CONCEPTS),
            min_size=size + 1,
            max_size=size + 1,
            unique=True,
        )
    )
    separator = draw(st.sampled_from([" ", ", "]))
    before = [draw(st.sampled_from(SPELLINGS[name])) for name in chosen[:size]]
    after = list(before)
    after[draw(st.integers(min_value=0, max_value=size - 1))] = draw(
        st.sampled_from(SPELLINGS[chosen[size]])
    )
    return size, separator.join(before), separator.join(after)


@SETTINGS
@given(replacement=replacements())
def test_a_replaced_concept_moves_the_sentence_away_from_the_original(
    replacement: tuple[int, str, str],
) -> None:
    size, original, replaced = replacement

    similarity = float(embed_one(original) @ embed_one(replaced))

    assert similarity < (SINGLE_BOUND if size == 1 else SEVERAL_BOUND)
