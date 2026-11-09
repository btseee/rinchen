"""The generation of a KD corpus (D10) over D6's shards and selection: the rows, their
labels and order, the counts and their invariants, the selection (copies, script purity,
empty answers, MBR among several candidates), the QE hook and its orientation, the sources
no letter of which reaches the teacher, the refusals of a reply that breaks the translator's
contract, the counters and `matches_reference`.

The shard files, the resume and the fingerprint are in the neighbouring test files. The
teacher is a `ScriptedTeacher` (literal scripts), the sources are built by hand, the export
is the fabricated one and the QE scorer a fake: nothing runs a model or touches a GPU."""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.config.filters import QeConfig
from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, make_row_id
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.synth.select import COUNT_KEYS
from rinchen.train.distill_export import verify_export
from rinchen.train.distill_generate import KdResult, ShardProgress
from rinchen.train.distill_teacher import TeacherOptions
from tests.support.corpus_rows import EN_MN, RU_MN
from tests.support.distill_teachers import (
    END,
    SP,
    TAG,
    hyp,
    scripted_teacher,
    verified_export,
)
from tests.support.planted_kd import (
    ALL_SELECTED_SCRIPT,
    PLANTED_COUNTS,
    PLANTED_SCRIPT,
    ScriptedTeacher,
    kd_run,
    planted_sources,
    sources_of,
)
from tests.support.planted_synth import FakeScorer

ENG, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"
ZERO_WIDTH = chr(0x200B) * 3


@pytest.fixture
def export(tmp_path):
    return verified_export(tmp_path / "teacher")


def planted_teacher(**options) -> ScriptedTeacher:
    return ScriptedTeacher(PLANTED_SCRIPT, **options)


def rows_of(result: KdResult) -> list[dict]:
    return result.corpus.rows(named=True)


# -- the rows


def test_the_rows_are_forward_translations_labelled_with_the_name_in_id_order(
    tmp_path, export
):
    result = kd_run(tmp_path, export, planted_teacher())

    rows = rows_of(result)
    assert [(row["src"], row["tgt"]) for row in rows] == [
        ("Hello", "Сайн байна уу"),
        ("Good morning", "Өглөөний мэнд"),
        ("Water", "Хүйтэн ус"),
        ("Thank you", "Баярлалаа танд"),
    ]
    for row in rows:
        assert (row["src_lang"], row["tgt_lang"]) == (ENG, KHK)
        assert row["origin"] == "ft"
        assert row["corpus"] == "ft:toy"
        assert row["split"] == "train"
        assert row["doc_id"] is None
        assert row["seg_idx"] is None
        assert row["filter_reason"] is None
        assert row["id"] == make_row_id(ENG, KHK, row["src"], row["tgt"])


def test_a_row_scores_only_the_script_purity_of_its_target(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    for row in rows_of(result):
        assert row["scores"] == {
            "lid_src": None,
            "lid_tgt": None,
            "script_src": None,
            "script_tgt": 1.0,
            "len_ratio": None,
            "num_ok": None,
            "labse": None,
            "metricx_qe": None,
            "cometkiwi": None,
        }


def test_the_corpus_has_the_corpus_schema(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    assert result.corpus.schema == CORPUS_SCHEMA


def test_every_row_of_every_shard_and_direction_has_the_same_created_stamp(
    tmp_path, export
):
    result = kd_run(
        tmp_path, export, ScriptedTeacher(ALL_SELECTED_SCRIPT), shard_size=2
    )

    assert result.corpus.height == 7
    assert result.corpus["created"].n_unique() == 1


def test_the_rows_of_both_directions_follow_in_direction_then_id_order(
    tmp_path, export
):
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    result = kd_run(tmp_path, export, teacher)

    assert [(row["src_lang"], row["src"]) for row in rows_of(result)] == [
        (ENG, "Hello"),
        (ENG, "Good morning"),
        (ENG, "Water"),
        (ENG, "Thank you"),
        (RUS, "Привет"),
        (RUS, "Спасибо"),
        (RUS, "Книга"),
    ]


def test_sources_in_another_order_give_the_same_rows(tmp_path, export):
    straight = kd_run(tmp_path / "a", export, ScriptedTeacher(ALL_SELECTED_SCRIPT))

    reordered = kd_run(
        tmp_path / "b",
        export,
        ScriptedTeacher(ALL_SELECTED_SCRIPT),
        sources=list(reversed(planted_sources())),
    )

    assert reordered.corpus.drop("created").equals(straight.corpus.drop("created"))
    assert reordered.fingerprint == straight.fingerprint


def test_nothing_selected_is_an_empty_frame_with_the_corpus_schema(tmp_path, export):
    teacher = ScriptedTeacher({text: [] for text in PLANTED_SCRIPT})

    result = kd_run(tmp_path, export, teacher)

    assert result.corpus.height == 0
    assert result.corpus.schema == CORPUS_SCHEMA
    assert result.counts["total"]["no_candidates"] == 7
    assert result.counts["total"]["selected"] == 0


def test_a_frame_in_another_row_order_gives_the_same_rows_and_fingerprint(
    tmp_path, export
):
    straight = kd_run(tmp_path / "a", export, ScriptedTeacher(ALL_SELECTED_SCRIPT))
    turned = [
        dataclasses.replace(source, frame=source.frame.reverse())
        for source in planted_sources()
    ]

    reversed_rows = kd_run(
        tmp_path / "b", export, ScriptedTeacher(ALL_SELECTED_SCRIPT), sources=turned
    )

    assert reversed_rows.corpus.drop("created").equals(straight.corpus.drop("created"))
    assert reversed_rows.fingerprint == straight.fingerprint


def test_a_direction_without_sources_has_no_shards_and_no_rows(tmp_path, export):
    sources = [sources_of(EN_MN, {})]

    result = kd_run(tmp_path, export, ScriptedTeacher({}), sources=sources)

    assert result.corpus.height == 0
    assert result.corpus.schema == CORPUS_SCHEMA
    assert result.counts[EN_MN] == dict.fromkeys(COUNT_KEYS, 0)
    assert result.shard_stats == {
        "planned": 0,
        "reused": 0,
        "generated": 0,
        "regenerated_invalid": 0,
    }


# -- what the teacher is asked


def test_the_teacher_gets_the_texts_of_each_shard_in_id_order_and_the_languages(
    tmp_path, export
):
    teacher = planted_teacher()

    kd_run(tmp_path, export, teacher, shard_size=3)

    assert teacher.calls == [
        (("Hello", "Good morning", "Water"), ENG, KHK),
        (("Thank you",), ENG, KHK),
        (("Привет", "Спасибо", "Книга"), RUS, KHK),
    ]


def test_the_teacher_counters_are_the_translators(tmp_path, export):
    teacher = planted_teacher(counters={"sources": 7, "truncated": 2})

    result = kd_run(tmp_path, export, teacher)

    assert result.teacher_counters == {"sources": 7, "truncated": 2}


# -- the counts


def test_the_counts_of_each_direction_are_the_planted_ones(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    assert result.counts[EN_MN] == PLANTED_COUNTS[EN_MN]
    assert result.counts[RU_MN] == PLANTED_COUNTS[RU_MN]


def test_the_total_is_the_sum_of_the_directions_and_has_every_count_key(
    tmp_path, export
):
    result = kd_run(tmp_path, export, planted_teacher())

    assert set(result.counts) == {EN_MN, RU_MN, "total"}
    for counts in result.counts.values():
        assert tuple(counts) == COUNT_KEYS
    assert result.counts["total"] == {
        key: PLANTED_COUNTS[EN_MN][key] + PLANTED_COUNTS[RU_MN][key]
        for key in COUNT_KEYS
    }


def test_the_counts_obey_the_invariants_of_d6(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    for counts in result.counts.values():
        assert counts["sources"] == (
            counts["no_candidates"] + counts["no_survivor"] + counts["selected"]
        )
        survivors = counts["candidates"] - (
            counts["dropped_empty"]
            + counts["dropped_purity"]
            + counts["dropped_copy"]
            + counts["dropped_qe"]
        )
        assert survivors >= counts["selected"] >= 0
    assert result.counts["total"]["selected"] == result.corpus.height


def test_a_candidate_equal_to_its_source_is_a_copy_and_a_latin_one_for_a_mongolian_target_is_impure(
    tmp_path, export
):
    result = kd_run(tmp_path, export, planted_teacher())

    assert result.counts[RU_MN]["dropped_copy"] == 1  # Книга -> Книга
    assert result.counts[RU_MN]["dropped_purity"] == 1  # Спасибо -> Thank you very much
    assert result.counts[RU_MN]["no_candidates"] == 1  # Привет -> []


# -- several candidates and one


def test_with_several_candidates_the_mbr_winner_is_the_row(tmp_path, export):
    script = {
        "Hello": [
            "Огт өөр өгүүлбэр энд",
            "Сайн байна уу бүгдээрээ",
            "Сайн байна уу бүгдээс",
        ]
    }
    sources = [sources_of(EN_MN, {"Hello": []})]

    result = kd_run(
        tmp_path,
        export,
        ScriptedTeacher(script),
        sources=sources,
        options=TeacherOptions(n=3),
    )

    assert [row["tgt"] for row in rows_of(result)] == ["Сайн байна уу бүгдээрээ"]
    assert result.counts["total"]["candidates"] == 3
    assert result.counts["total"]["selected"] == 1


def test_with_one_candidate_no_mbr_happens(tmp_path, export, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the vote of MBR ran for a lone candidate")

    monkeypatch.setattr("rinchen.synth.select.mbr.select", refuse)

    result = kd_run(tmp_path, export, planted_teacher())

    assert result.corpus.height == 4


# -- the sources no letter of which reaches the teacher


def test_a_source_without_a_letter_or_digit_is_not_sent_and_is_counted(
    tmp_path, export
):
    sources = [sources_of(EN_MN, {"Hello": [], ZERO_WIDTH: [], "...": []})]
    teacher = ScriptedTeacher({"Hello": ["Сайн байна уу"]})

    result = kd_run(tmp_path, export, teacher, sources=sources)

    (call,) = teacher.calls
    assert call[0] == ("Hello",)
    assert result.counts[EN_MN]["sources"] == 3
    assert result.counts[EN_MN]["no_candidates"] == 2
    assert result.counts[EN_MN]["selected"] == 1
    assert result.letterless_sources == {EN_MN: 2, "total": 2}


def test_a_shard_of_sources_without_letters_is_not_sent_at_all(tmp_path, export):
    sources = [sources_of(EN_MN, {ZERO_WIDTH: [], "!!!": []})]
    teacher = ScriptedTeacher({})

    result = kd_run(tmp_path, export, teacher, sources=sources)

    assert teacher.calls == []
    assert result.corpus.height == 0
    assert result.counts["total"]["no_candidates"] == 2
    assert result.letterless_sources["total"] == 2


def test_a_digit_is_a_letter_enough_to_be_sent(tmp_path, export):
    sources = [sources_of(EN_MN, {"2024": []})]
    teacher = ScriptedTeacher({"2024": ["Хоёр мянга хорин дөрөв"]})

    result = kd_run(tmp_path, export, teacher, sources=sources)

    assert result.letterless_sources == {EN_MN: 0, "total": 0}
    assert result.corpus.height == 1


def test_the_planted_sources_have_no_letterless_one(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    assert result.letterless_sources == {EN_MN: 0, RU_MN: 0, "total": 0}


# -- the QE hook


def test_the_qe_pairs_are_the_real_source_and_the_candidate(tmp_path, export):
    scorer = FakeScorer({})

    kd_run(tmp_path, export, planted_teacher(), scorer=scorer)

    pairs = {
        pair
        for sources, translations in scorer.calls
        for pair in zip(sources, translations, strict=True)
    }
    # The copy and the impure candidate of the Russian direction never reach the scorer.
    assert pairs == {
        ("Hello", "Сайн байна уу"),
        ("Good morning", "Өглөөний мэнд"),
        ("Thank you", "Баярлалаа танд"),
        ("Water", "Хүйтэн ус"),
    }


def test_a_candidate_the_scorer_rates_above_the_floor_is_dropped_and_counted(
    tmp_path, export
):
    scorer = FakeScorer({("Hello", "Сайн байна уу"): 9.0}, default=1.0)

    result = kd_run(tmp_path, export, planted_teacher(), scorer=scorer)

    assert result.counts[EN_MN]["dropped_qe"] == 1
    assert result.counts[EN_MN]["no_survivor"] == 1
    assert result.counts[EN_MN]["selected"] == 3
    assert "Hello" not in [row["src"] for row in rows_of(result)]


def test_the_score_of_the_winner_is_in_the_scorers_column_and_the_status_is_ok(
    tmp_path, export
):
    scorer = FakeScorer({("Water", "Хүйтэн ус"): 2.5}, default=1.0)

    result = kd_run(tmp_path, export, planted_teacher(), scorer=scorer)

    assert result.qe_status == "ok"
    by_source = {row["src"]: row["scores"] for row in rows_of(result)}
    assert by_source["Water"]["metricx_qe"] == 2.5
    assert by_source["Hello"]["metricx_qe"] == 1.0
    assert all(scores["cometkiwi"] is None for scores in by_source.values())


def test_the_configured_scorer_names_the_column(tmp_path, export):
    selection = SelectionConfig(qe=QeConfig(scorer="cometkiwi"))
    scorer = FakeScorer({}, default=0.9)

    result = kd_run(
        tmp_path, export, planted_teacher(), selection=selection, scorer=scorer
    )

    assert all(
        row["scores"]["cometkiwi"] == pytest.approx(0.9) for row in rows_of(result)
    )
    assert all(row["scores"]["metricx_qe"] is None for row in rows_of(result))


def test_without_a_scorer_the_status_says_so_and_the_column_is_null(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    assert result.qe_status == "skipped: no scorer"
    assert result.corpus["scores"].struct.field("metricx_qe").null_count() == 4


# -- the real teacher behind the protocol


def test_the_teacher_translator_is_a_kd_translator_and_its_counters_are_reported(
    tmp_path,
):
    script = {(TAG, SP + "hello", SP + "world"): [hyp(SP + "сайн", SP + "уу", END)]}
    teacher, ct2 = scripted_teacher(tmp_path, script)
    sources = [sources_of(EN_MN, {"hello world": ["сайн уу"], ZERO_WIDTH: []})]

    result = kd_run(
        tmp_path, verify_export(tmp_path / "export"), teacher, sources=sources
    )

    assert [(row["src"], row["tgt"]) for row in rows_of(result)] == [
        ("hello world", "сайн уу")
    ]
    # The source without a letter never reached the teacher; the counters are its own.
    assert result.teacher_counters == {
        "sources": 1,
        "source_too_long": 0,
        "empty_source": 0,
        "truncated": 0,
        "line_break": 0,
    }
    assert result.matches_reference == {EN_MN: 1, "total": 1}
    assert len(ct2.calls) == 1


# -- a translator that breaks its contract


def _generate_with_reply(tmp_path, export, reply):
    return kd_run(tmp_path, export, planted_teacher(replies={0: reply}), shard_size=3)


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        ("not a list", "str"),
        ((["a"], ["b"], ["c"]), "tuple"),
        ([["a"], ["b"]], "2 candidate lists for 3 texts"),
        ([["a"], ["b"], "c"], "text 2"),
        ([["a"], [1], ["c"]], "text 1"),
        ([["a"], None, ["c"]], "text 1"),
    ],
)
def test_a_reply_that_is_not_one_list_of_strings_per_text_is_a_data_error(
    tmp_path, export, reply, problem
):
    with pytest.raises(DataError) as caught:
        _generate_with_reply(tmp_path, export, reply)

    message = str(caught.value)
    assert EN_MN in message
    assert "shard 0 of 2" in message
    assert problem in message


def test_a_reply_that_breaks_the_contract_writes_no_shard(tmp_path, export):
    with pytest.raises(DataError):
        _generate_with_reply(tmp_path, export, [["a"]])

    assert not list((tmp_path / "out").rglob("*.parquet"))


@pytest.mark.parametrize("error_type", [DataError, ConfigError])
def test_a_data_or_config_error_of_the_translator_keeps_its_type_and_names_the_shard(
    tmp_path, export, error_type
):
    teacher = planted_teacher(fail_on_call={0: error_type("the cause")})

    with pytest.raises(error_type) as caught:
        kd_run(tmp_path, export, teacher, shard_size=3)

    assert "the cause" in str(caught.value)
    assert EN_MN in str(caught.value)
    assert "shard 0 of 2" in str(caught.value)


@pytest.mark.parametrize("error", [UpstreamError("down"), OSError("no route")])
def test_an_upstream_or_os_error_of_the_translator_is_an_upstream_error(
    tmp_path, export, error
):
    teacher = planted_teacher(fail_on_call={0: error})

    with pytest.raises(UpstreamError) as caught:
        kd_run(tmp_path, export, teacher, shard_size=3)

    assert str(error) in str(caught.value)
    assert EN_MN in str(caught.value)
    assert caught.value.__cause__ is error


def test_any_other_error_of_the_translator_propagates_as_it_is(tmp_path, export):
    teacher = planted_teacher(fail_on_call={0: RuntimeError("a bug")})

    with pytest.raises(RuntimeError, match="a bug"):
        kd_run(tmp_path, export, teacher, shard_size=3)


def test_a_candidate_utf8_cannot_encode_is_a_data_error_naming_the_shard(
    tmp_path, export
):
    reply = [[chr(0xD800)], [], []]

    with pytest.raises(DataError) as caught:
        _generate_with_reply(tmp_path, export, reply)

    assert EN_MN in str(caught.value)
    assert "shard 0 of 2" in str(caught.value)
    assert not list((tmp_path / "out").rglob("*.parquet"))


# -- matches_reference


def test_matches_reference_counts_the_rows_whose_target_is_an_authentic_one(
    tmp_path, export
):
    result = kd_run(tmp_path, export, planted_teacher())

    # Hello, Good morning and Thank you (its second target) agree; Water does not.
    assert result.matches_reference == {EN_MN: 3, RU_MN: 0, "total": 3}


def test_matches_reference_adds_up_the_shards_and_the_directions(tmp_path, export):
    teacher = ScriptedTeacher(ALL_SELECTED_SCRIPT)

    result = kd_run(tmp_path, export, teacher, shard_size=2)

    # English: Hello and Good morning in the first shard, Thank you in the second (Water
    # differs). Russian: Спасибо and Книга agree, Привет does not.
    assert result.matches_reference == {EN_MN: 3, RU_MN: 2, "total": 5}


def test_a_candidate_the_selection_drops_is_not_a_match(tmp_path, export):
    scorer = FakeScorer({("Hello", "Сайн байна уу"): 9.0}, default=1.0)

    result = kd_run(tmp_path, export, planted_teacher(), scorer=scorer)

    # Hello's answer equals its reference but the QE floor drops it: no row, no match.
    assert result.matches_reference == {EN_MN: 2, RU_MN: 0, "total": 2}


def test_matches_reference_looks_at_the_references_of_the_source_of_the_row(
    tmp_path, export
):
    sources = [sources_of(EN_MN, {"Alpha": ["Нэг"], "Beta": ["Хоёр"]})]
    teacher = ScriptedTeacher({"Alpha": ["Хоёр"], "Beta": ["Хоёр"]})

    result = kd_run(tmp_path, export, teacher, sources=sources)

    # Alpha is answered with the reference of Beta, which is no reference of Alpha.
    assert result.matches_reference == {EN_MN: 1, "total": 1}


def test_a_source_without_a_reference_never_matches(tmp_path, export):
    sources = [dataclasses.replace(planted_sources()[0], references={})]

    result = kd_run(tmp_path, export, planted_teacher(), sources=sources)

    assert result.matches_reference == {EN_MN: 0, "total": 0}


# -- progress


def test_progress_is_called_once_per_shard_in_order_after_the_shard_is_on_disk_and_before_its_selection(
    tmp_path, export
):
    scorer = FakeScorer({})
    seen: list[tuple[ShardProgress, int, int]] = []

    def progress(event: ShardProgress) -> None:
        shards = len(list((tmp_path / "out").rglob("shard-*.parquet")))
        seen.append((event, shards, len(scorer.calls)))

    kd_run(
        tmp_path,
        export,
        ScriptedTeacher(ALL_SELECTED_SCRIPT),
        shard_size=2,
        scorer=scorer,
        progress=progress,
    )

    assert [event for event, _, _ in seen] == [
        ShardProgress(EN_MN, 0, 2, False),
        ShardProgress(EN_MN, 1, 2, False),
        ShardProgress(RU_MN, 0, 2, False),
        ShardProgress(RU_MN, 1, 2, False),
    ]
    # The shard is on disk when the event is made (one more file each time) and the
    # scorer has not seen it: its pairs come after, one call per shard.
    assert [shards for _, shards, _ in seen] == [1, 2, 3, 4]
    assert [calls for _, _, calls in seen] == [0, 1, 2, 3]


def test_a_shard_read_from_disk_is_reported_as_reused(tmp_path, export):
    kd_run(tmp_path, export, planted_teacher(), shard_size=2)
    seen: list[ShardProgress] = []

    kd_run(tmp_path, export, ScriptedTeacher({}), shard_size=2, progress=seen.append)

    assert [event.reused for event in seen] == [True, True, True, True]


# -- the refusals of the call itself


def test_no_sources_are_refused(tmp_path, export):
    with pytest.raises(ConfigError, match="no sources"):
        kd_run(tmp_path, export, planted_teacher(), sources=[])


def test_two_frames_of_one_direction_are_refused_before_anything_is_generated(
    tmp_path, export
):
    teacher = planted_teacher()

    with pytest.raises(ConfigError, match=EN_MN):
        kd_run(
            tmp_path,
            export,
            teacher,
            sources=[*planted_sources(), planted_sources()[0]],
        )

    assert teacher.calls == []


def test_a_bad_name_is_refused_before_anything_is_generated(tmp_path, export):
    teacher = planted_teacher()

    with pytest.raises(ConfigError, match="name"):
        kd_run(tmp_path, export, teacher, name="a b")

    assert teacher.calls == []


def test_a_shard_size_below_one_is_refused_before_anything_is_generated(
    tmp_path, export
):
    teacher = planted_teacher()

    with pytest.raises(ValueError, match="shard_size"):
        kd_run(tmp_path, export, teacher, shard_size=0)

    assert teacher.calls == []


def test_the_shard_directory_and_the_fingerprint_are_in_the_result(tmp_path, export):
    result = kd_run(tmp_path, export, planted_teacher())

    assert len(result.fingerprint) == 64
    assert (
        result.shard_directory == tmp_path / "out" / f"shards-{result.fingerprint[:12]}"
    )
    assert result.shard_directory.is_dir()
