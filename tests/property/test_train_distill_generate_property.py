"""Property tests of the KD generation (D10): for arbitrary scripted answers, the rows plus
the three counts of the sources that have none account for every source, and the corpus and
the counts are the same for any shard size and for sources given in any order.

Sources are English words of a few letters (always sendable); the scripted answers are drawn
from a small pool that holds good Mongolian texts, Latin ones (impure), the empty text, and
copies of the source. The hypothesis profile is loaded in `tests/conftest.py`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.train.distill_export import VerifiedExport
from tests.support.corpus_rows import EN_MN
from tests.support.distill_teachers import verified_export
from tests.support.planted_kd import ScriptedTeacher, kd_run, sources_of

_WORDS = st.text("abcde", min_size=1, max_size=4)
_POOL = ("Сайн", "Сайн байна уу", "Ус", "Hello", "", "  ", "Баярлалаа")
# Each source is answered with up to three of the pool's texts and, sometimes, a copy of itself.
_ANSWERS = st.tuples(st.lists(st.sampled_from(_POOL), max_size=3), st.booleans())
_CASES = st.dictionaries(_WORDS, _ANSWERS, min_size=1, max_size=12)
_SHARD_SIZES = st.integers(min_value=1, max_value=6)


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> VerifiedExport:
    return verified_export(tmp_path_factory.mktemp("teacher"))


def _script(case: dict[str, tuple[list[str], bool]]) -> dict[str, list[str]]:
    return {
        word: [*candidates, word] if copy else list(candidates)
        for word, (candidates, copy) in case.items()
    }


def _generate(export, case, order, shard_size):
    sources = [sources_of(EN_MN, {word: [] for word in order})]
    with tempfile.TemporaryDirectory() as directory:
        return kd_run(
            Path(directory),
            export,
            ScriptedTeacher(_script(case)),
            sources=sources,
            shard_size=shard_size,
        )


@settings(max_examples=60)
@given(case=_CASES, shard_size=_SHARD_SIZES)
def test_the_rows_and_the_unselected_sources_account_for_every_source(
    export, case, shard_size
):
    result = _generate(export, case, sorted(case), shard_size)

    counts = result.counts["total"]
    assert counts["sources"] == len(case)
    assert result.corpus.height == counts["selected"]
    assert result.corpus.height + counts["no_candidates"] + counts[
        "no_survivor"
    ] == len(case)
    assert result.counts[EN_MN] == counts


@settings(max_examples=60)
@given(
    case=_CASES, shard_size=_SHARD_SIZES, shard_size_again=_SHARD_SIZES, data=st.data()
)
def test_the_result_is_the_same_for_any_shard_size_and_any_order_of_the_sources(
    export, case, shard_size, shard_size_again, data
):
    shuffled = data.draw(st.permutations(sorted(case)))

    straight = _generate(export, case, sorted(case), shard_size)
    other = _generate(export, case, shuffled, shard_size_again)

    assert other.corpus.drop("created").equals(straight.corpus.drop("created"))
    assert other.counts == straight.counts
    assert other.matches_reference == straight.matches_reference
    assert other.letterless_sources == straight.letterless_sources
