"""Property tests for `rinchen.synth.shards`: the plan, the fingerprint and the
shard store hold for arbitrary rows, not only the hand-picked ones.

Ids are unique by construction (a dictionary keyed by id), as `read_monolingual`
guarantees. Texts and candidates avoid lone surrogates, which Parquet cannot
hold, and NUL. The hypothesis profile is loaded in `tests/conftest.py`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import polars as pl
from hypothesis import assume, given
from hypothesis import strategies as st

from rinchen.config.synth import JobConfig, VllmConfig
from rinchen.synth.shards import ShardStore, job_fingerprint, plan_shards

_CHARACTERS = st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00")
_TEXTS = st.text(_CHARACTERS, max_size=12)
_IDS = st.text(_CHARACTERS, min_size=1, max_size=6)
_ROWS = st.dictionaries(_IDS, _TEXTS, max_size=40)  # id -> text
_SHARD_SIZES = st.integers(min_value=1, max_value=50)

_JOB = JobConfig(
    kind="bt", mono_lang="khk_Cyrl", other_lang="eng_Latn", generator="vllm_default"
)
_GENERATOR = VllmConfig()


def _frame(rows: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {"id": list(rows), "text": list(rows.values())},
        schema={"id": pl.String, "text": pl.String},
    )


@given(rows=_ROWS, shard_size=_SHARD_SIZES)
def test_the_shards_concatenate_to_the_id_sorted_input(
    rows: dict[str, str], shard_size: int
) -> None:
    plan = plan_shards(_frame(rows), shard_size)

    expected = sorted(rows.items())
    assert [pair for shard in plan for pair in zip(shard.ids, shard.texts)] == expected


@given(rows=_ROWS, shard_size=_SHARD_SIZES)
def test_every_shard_is_full_except_possibly_the_last(
    rows: dict[str, str], shard_size: int
) -> None:
    plan = plan_shards(_frame(rows), shard_size)

    sizes = [len(shard.ids) for shard in plan]
    assert all(size == shard_size for size in sizes[:-1])
    assert all(1 <= size <= shard_size for size in sizes)
    assert sum(sizes) == len(rows)
    assert len(plan) == -(-len(rows) // shard_size)  # ceiling division


@given(rows=_ROWS, shard_size=_SHARD_SIZES)
def test_the_shard_indexes_run_from_zero_without_gaps(
    rows: dict[str, str], shard_size: int
) -> None:
    plan = plan_shards(_frame(rows), shard_size)

    assert [shard.index for shard in plan] == list(range(len(plan)))


@given(rows=_ROWS, shard_size=_SHARD_SIZES, data=st.data())
def test_the_plan_is_the_same_for_every_input_order(
    rows: dict[str, str], shard_size: int, data: st.DataObject
) -> None:
    order = data.draw(st.permutations(list(rows)))
    shuffled = {key: rows[key] for key in order}

    assert plan_shards(_frame(shuffled), shard_size) == plan_shards(
        _frame(rows), shard_size
    )


@given(rows=_ROWS, shard_size=_SHARD_SIZES)
def test_the_same_input_and_size_give_identical_shards(
    rows: dict[str, str], shard_size: int
) -> None:
    assert plan_shards(_frame(rows), shard_size) == plan_shards(
        _frame(rows), shard_size
    )


def _fingerprint(rows: dict[str, str]) -> str:
    return job_fingerprint(_frame(rows), _JOB, _GENERATOR, 4, 1)


@given(first=_ROWS, second=_ROWS)
def test_equal_fingerprints_mean_equal_rows_whatever_their_order(
    first: dict[str, str], second: dict[str, str]
) -> None:
    assert (_fingerprint(first) == _fingerprint(second)) == (
        sorted(first.items()) == sorted(second.items())
    )


@given(rows=_ROWS, data=st.data())
def test_the_fingerprint_ignores_the_input_order(
    rows: dict[str, str], data: st.DataObject
) -> None:
    order = data.draw(st.permutations(list(rows)))

    assert _fingerprint({key: rows[key] for key in order}) == _fingerprint(rows)


@given(rows=_ROWS.filter(bool), other_text=_TEXTS, data=st.data())
def test_changing_one_text_changes_the_fingerprint(
    rows: dict[str, str], other_text: str, data: st.DataObject
) -> None:
    victim = data.draw(st.sampled_from(sorted(rows)))
    assume(rows[victim] != other_text)

    assert _fingerprint({**rows, victim: other_text}) != _fingerprint(rows)


@given(rows=_ROWS.filter(bool), new_id=_IDS, data=st.data())
def test_renaming_one_id_changes_the_fingerprint(
    rows: dict[str, str], new_id: str, data: st.DataObject
) -> None:
    victim = data.draw(st.sampled_from(sorted(rows)))
    assume(new_id not in rows)
    renamed = {(new_id if key == victim else key): text for key, text in rows.items()}

    assert _fingerprint(renamed) != _fingerprint(rows)


_Settings = tuple[Literal["bt", "ft"], int, float, int, int]
_SETTINGS: st.SearchStrategy[_Settings] = st.tuples(
    st.sampled_from(["bt", "ft"]),  # job kind
    st.integers(min_value=1, max_value=64),  # generator n
    st.sampled_from([0.0, 0.3, 0.7, 1.0]),  # generator temperature
    st.integers(min_value=1, max_value=9),  # shard_size
    st.integers(min_value=0, max_value=9),  # seed
)


def _settings_fingerprint(settings: _Settings) -> str:
    kind, n, temperature, shard_size, seed = settings
    job = JobConfig(
        kind=kind,
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="vllm_default",
    )
    generator = VllmConfig(n=n, temperature=temperature)
    return job_fingerprint(_frame({"a": "x"}), job, generator, shard_size, seed)


@given(first=_SETTINGS, second=_SETTINGS)
def test_equal_fingerprints_mean_equal_settings(
    first: _Settings, second: _Settings
) -> None:
    assert (_settings_fingerprint(first) == _settings_fingerprint(second)) == (
        first == second
    )


_CANDIDATE_LISTS = st.lists(
    st.lists(st.text(_CHARACTERS, max_size=10), max_size=4), max_size=8
)


@given(candidates=_CANDIDATE_LISTS)
def test_the_store_round_trips_any_candidates_including_empty_lists(
    candidates: list[list[str]],
) -> None:
    ids = [f"id-{n}" for n in range(len(candidates))]
    with tempfile.TemporaryDirectory() as directory:
        store = ShardStore(Path(directory) / "shards")

        store.write(0, ids, candidates)

        assert store.read(0, ids) == candidates
