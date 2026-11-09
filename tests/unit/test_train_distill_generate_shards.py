"""The shards of a KD generation (D10): where the files are, that a rerun reuses them, that a
crash between shards loses nothing, that a damaged shard is regenerated and counted, that
`force` regenerates all, that another teacher or option names another directory (a stale
shard is never reused) and that the batch size, the thread count and the selection do not.

Same fakes as `test_train_distill_generate.py`: a scripted teacher, hand-built sources, the
fabricated export, a fake scorer."""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.config.synth import SelectionConfig
from rinchen.errors import UpstreamError
from rinchen.train.distill_teacher import TeacherOptions
from tests.support.corpus_rows import EN_MN, RU_MN
from tests.support.distill_exports import sha256_of_bytes
from tests.support.distill_teachers import verified_export
from tests.support.planted_kd import (
    ALL_SELECTED_SCRIPT,
    PLANTED_SCRIPT,
    ScriptedTeacher,
    kd_run,
    planted_sources,
    sources_of,
)
from tests.support.planted_synth import FakeScorer
from tests.support.shard_files import change_a_candidate_keeping_the_digest

# Two shards of the English direction (4 sources) and two of the Russian one (3 sources).
SHARD_SIZE = 2
ZERO_WIDTH = chr(0x200B) * 3
FIRST_RUN = {"planned": 4, "reused": 0, "generated": 4, "regenerated_invalid": 0}


@pytest.fixture
def export(tmp_path):
    return verified_export(tmp_path / "teacher")


def run(tmp_path, export, teacher=None, **keywords):
    return kd_run(
        tmp_path,
        export,
        ScriptedTeacher(ALL_SELECTED_SCRIPT) if teacher is None else teacher,
        shard_size=SHARD_SIZE,
        **keywords,
    )


def shard_files(result):
    return sorted(
        path.relative_to(result.shard_directory).as_posix()
        for path in result.shard_directory.rglob("*.parquet")
    )


def no_call_teacher() -> ScriptedTeacher:
    """A teacher that fails the test when it is asked anything: every text is unscripted."""
    return ScriptedTeacher({})


# -- the files


def test_the_shards_of_every_direction_are_files_under_one_directory(tmp_path, export):
    result = run(tmp_path, export)

    assert result.shard_directory.name == f"shards-{result.fingerprint[:12]}"
    assert shard_files(result) == [
        f"{EN_MN}/shard-00000.parquet",
        f"{EN_MN}/shard-00001.parquet",
        f"{RU_MN}/shard-00000.parquet",
        f"{RU_MN}/shard-00001.parquet",
    ]


def test_a_first_run_generates_every_shard(tmp_path, export):
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    result = run(tmp_path, export, teacher)

    assert result.shard_stats == FIRST_RUN
    assert len(teacher.calls) == 4


# -- reuse and resume


def test_a_rerun_reuses_every_shard_and_asks_the_teacher_nothing(tmp_path, export):
    first = run(tmp_path, export)
    teacher = no_call_teacher()

    second = run(tmp_path, export, teacher)

    assert second.shard_stats == {
        "planned": 4,
        "reused": 4,
        "generated": 0,
        "regenerated_invalid": 0,
    }
    assert teacher.calls == []
    assert second.corpus.drop("created").equals(first.corpus.drop("created"))
    assert second.counts == first.counts


def test_a_rerun_reports_the_same_letterless_sources_and_matches(tmp_path, export):
    sources = [sources_of(EN_MN, {"Hello": ["Сайн байна уу"], ZERO_WIDTH: []})]
    teacher = ScriptedTeacher({"Hello": ["Сайн байна уу"]})
    first = run(tmp_path, export, teacher, sources=sources)

    second = run(tmp_path, export, no_call_teacher(), sources=sources)

    assert second.shard_stats["reused"] == 1
    assert (
        second.letterless_sources == first.letterless_sources == {EN_MN: 1, "total": 1}
    )
    assert second.matches_reference == first.matches_reference == {EN_MN: 1, "total": 1}
    assert second.counts == first.counts


def test_a_translator_that_fails_leaves_the_finished_shards_and_the_rerun_does_the_rest(
    tmp_path, export
):
    failing = ScriptedTeacher(
        ALL_SELECTED_SCRIPT, fail_on_call={1: UpstreamError("down")}
    )

    with pytest.raises(UpstreamError) as caught:
        run(tmp_path, export, failing)

    assert f"direction {EN_MN}" in str(caught.value)
    assert "shard 1 of 2" in str(caught.value)
    (written,) = [path.name for path in (tmp_path / "out").rglob("shard-*.parquet")]
    assert written == "shard-00000.parquet"

    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)
    result = run(tmp_path, export, teacher)

    assert result.shard_stats == {
        "planned": 4,
        "reused": 1,
        "generated": 3,
        "regenerated_invalid": 0,
    }
    assert len(teacher.calls) == 3


def test_a_shard_is_on_disk_before_the_selection_of_the_shard_starts(tmp_path, export):
    class Crashing(FakeScorer):
        def score(self, sources, translations):
            raise UpstreamError("the scorer is down")

    with pytest.raises(UpstreamError, match="scorer"):
        run(tmp_path, export, scorer=Crashing({}))

    assert [path.name for path in (tmp_path / "out").rglob("shard-*.parquet")] == [
        "shard-00000.parquet"
    ]


# -- damaged shards


def _shard_path(result, direction=EN_MN, index=0):
    return result.shard_directory / direction / f"shard-{index:05d}.parquet"


def test_a_truncated_shard_is_regenerated_and_counted(tmp_path, export):
    first = run(tmp_path, export)
    damaged = _shard_path(first)
    damaged.write_bytes(damaged.read_bytes()[:20])
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    second = run(tmp_path, export, teacher)

    assert second.shard_stats == {
        "planned": 4,
        "reused": 3,
        "generated": 1,
        "regenerated_invalid": 1,
    }
    assert len(teacher.calls) == 1
    assert second.corpus.drop("created").equals(first.corpus.drop("created"))


def test_a_shard_with_a_changed_candidate_is_regenerated_not_trusted(tmp_path, export):
    first = run(tmp_path, export)
    change_a_candidate_keeping_the_digest(_shard_path(first, RU_MN, 0))
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    second = run(tmp_path, export, teacher)

    assert second.shard_stats["regenerated_invalid"] == 1
    assert second.corpus.drop("created").equals(first.corpus.drop("created"))


# -- force


def test_force_generates_every_shard_again_and_counts_none_as_invalid(tmp_path, export):
    run(tmp_path, export)
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    result = run(tmp_path, export, teacher, force=True)

    assert result.shard_stats == FIRST_RUN
    assert len(teacher.calls) == 4


def test_force_replaces_a_damaged_shard_without_calling_it_invalid(tmp_path, export):
    first = run(tmp_path, export)
    _shard_path(first).write_bytes(b"damaged")

    result = run(tmp_path, export, force=True)

    assert result.shard_stats["regenerated_invalid"] == 0
    assert run(tmp_path, export, no_call_teacher()).shard_stats["reused"] == 4


# -- what names the shard directory


def test_another_teacher_digest_names_another_directory_and_reuses_nothing(
    tmp_path, export
):
    first = run(tmp_path, export)
    other = dataclasses.replace(
        export, sha256={**export.sha256, "model.bin": sha256_of_bytes(b"another model")}
    )
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    second = run(tmp_path, other, teacher)

    assert second.shard_directory != first.shard_directory
    assert second.shard_stats == FIRST_RUN
    assert len(teacher.calls) == 4
    assert shard_files(first)  # the stale shards stay where they are


def test_another_decoding_option_names_another_directory(tmp_path, export):
    first = run(tmp_path, export)

    second = run(
        tmp_path,
        export,
        options=dataclasses.replace(TeacherOptions(), beam_size=5),
    )

    assert second.shard_directory != first.shard_directory
    assert second.shard_stats == FIRST_RUN


@pytest.mark.parametrize("field", ["batch_size", "threads"])
def test_the_batch_size_and_the_threads_do_not_change_the_directory(
    tmp_path, export, field
):
    first = run(tmp_path, export)

    second = run(
        tmp_path,
        export,
        no_call_teacher(),
        options=dataclasses.replace(TeacherOptions(), **{field: 9}),
    )

    assert second.shard_directory == first.shard_directory
    assert second.shard_stats["reused"] == 4


def test_another_selection_reuses_the_shards_and_may_select_other_rows(
    tmp_path, export
):
    mixed = {**PLANTED_SCRIPT, "Спасибо": ["Thank you Баярлалаа"]}
    first = run(tmp_path, export, ScriptedTeacher(mixed))
    # About half the letters of that answer are Cyrillic: impure at 0.9, pure enough at 0.5.
    lenient = SelectionConfig(min_script_share=0.5)

    second = run(tmp_path, export, no_call_teacher(), selection=lenient)

    assert second.shard_directory == first.shard_directory
    assert second.shard_stats["reused"] == 4
    assert first.counts["total"]["selected"] == 4
    assert second.counts["total"]["selected"] == 5


def test_another_seed_or_shard_size_names_another_directory(tmp_path, export):
    first = run(tmp_path, export)

    assert run(tmp_path, export, seed=99).shard_directory != first.shard_directory
    assert (
        kd_run(
            tmp_path, export, ScriptedTeacher(ALL_SELECTED_SCRIPT), shard_size=3
        ).shard_directory
        != first.shard_directory
    )


def test_one_source_text_changed_names_another_directory(tmp_path, export):
    first = run(tmp_path, export)
    sources = planted_sources()
    sources[0] = dataclasses.replace(
        sources[0],
        frame=sources[0].frame.with_columns(
            text=sources[0].frame["text"].str.replace("Water", "Waters")
        ),
    )
    teacher = ScriptedTeacher({**ALL_SELECTED_SCRIPT, "Waters": ["Ус"]})

    second = run(tmp_path, export, teacher, sources=sources)

    assert second.shard_directory != first.shard_directory
    assert second.shard_stats == FIRST_RUN
