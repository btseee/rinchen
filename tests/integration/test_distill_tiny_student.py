"""The D10 STOP gate: a tiny student is distilled from a tiny teacher and exported. Everything
runs through the real `rinchen` commands with the real Eole and the real CTranslate2 on the
CPU; nothing here touches a GPU (`CUDA_VISIBLE_DEVICES` is the empty string for the whole
module, the Eole configs train with `gpu_ranks: []` and the converter runs with the device
hidden), and the in-process CTranslate2 translator is built on `device="cpu"`.

The data are the toy corpora of `tests/support/tiny_parallel.py`: templated sentences, so
every number below tests machinery and ranks nothing (three seeds of the same student
span 1.9 chrF in the D10 spike). The corpus file holds the train, back-translated and dev
rows; the toy TEST split is written to no file a command reads and is looked at here, in
the test process, only to show that none of its sentences reached the generator.

Module fixtures run the expensive commands once, in the order the scenarios need, and take
a snapshot of what the tests look at, so no test depends on the order the tests run in:

    T  the tiny teacher: 2,000 steps of `teacher_tiny.yaml` (source tag), `average --last 2`
    X  `distill export` of the averaged teacher
    G  `distill generate` over the 3,600 authentic training sources, and a rerun
    P  `distill prepare --mix 1:1` for the tiny student, and two dry runs (`--mix 3:1`)
    S  `train launch` of the student: 2,000 steps from random initialisation
    E  `distill evaluate` of the teacher and of the student on the same dev lines
    Q  `distill export` of the student, and CTranslate2 against Eole on 200 dev sentences
    V  a second student with a 500-piece SentencePiece model: 200 steps
    R  fifteen refusals, each exit 2 with nothing written and neither Eole nor the
       converter started (`tests/support/distill_gate_refusals.py`)

Measured on this machine (CPU, two threads, Eole 0.6.4, CTranslate2 4.8.4, 2026-11-09; the
machine was shared with other work, so a time moves by a third from one run to the next, a
count or a score never does; the spike of
`.superpowers/sdd/2026-11-09-d10-distillation/spike-findings.md` measured the same counts and
scores). The whole module takes 3 min 18 s to 3 min 41 s:

    T  ablate --prepare-only 1.2 s, launch 65 to 111 s, average 1.0 s
    X  export 3.4 s; model.bin 2,034,311 bytes (2 encoder, 2 decoder layers, 344 entries)
    G  3,600 sources, 3,599 rows (1 dropped for script purity), 3,583 equal to an authentic
       target (99.6 %), 10 shards, 1.6 s; the rerun reuses 10 of 10
    P  weights 22, 6, 22, 6, 15, 7, 15, 7 (originals) and 28, 28, 22, 22 (KD), 200 in all;
       `--mix 3:1` gives 84, 84, 66, 66; 12 corpora, 400 validation lines, 641 pieces; 1.1 s
    S  launch 59 to 77 s: 2 encoder and 1 decoder layers, loss 5.58 at step 50, 0.95 at 2000
    E  teacher chrF 98.20 (min 97.79), student chrF 98.58 (min 97.54), 3.8 s each, no tag
       leak, no dev line in training
    Q  export 3.2 s; model.bin 1,835,438 bytes against the teacher's 2,034,311; CTranslate2
       equals Eole's greedy output on 200 of 200 dev sentences (2.6 s for both);
       "This is the star." gives "Энэ бол од."
    V  a 500-piece model: `src_vocab_size` 501, a 200-step launch (18 s with the model's
       training and the preparation), 118 reserved symbols
    R  15 refusals, each 1.0 to 1.3 s (11 s with the damaged copies)

The thresholds are the gate: teacher chrF at least 90.0, student chrF at least 95.0 and at
least the teacher's minus 3.0, CTranslate2 equal to Eole on at least 95 % of 200, at least
3,550 KD rows. A miss is answered by raising the steps of the config, never by lowering a
threshold.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
import sentencepiece

from rinchen.config.languages import direction_tag
from rinchen.corpus.readers import scan_corpus
from rinchen.spm.symbols import user_defined_symbols
from rinchen.synth.select import COUNT_KEYS
from rinchen.train.eole_cfg import read_config
from tests.support import distill_gate
from tests.support.distill_gate import (
    Invocation,
    is_cyrillic_text,
    sha256_of,
    source_findings,
)
from tests.support.distill_gate_refusals import (
    Refusal,
    RefusalInputs,
    run_refusals,
)
from tests.support.distill_gate_stages import (
    DEV_LINES,
    DIRECTIONS,
    EXPORT_FILES,
    SECOND_STUDENT_STEPS,
    STEPS,
    STUDENT_BASE,
    WEIGHTS,
    Evaluations,
    Exported,
    Generated,
    Prepared,
    SecondStudent,
    Smoke,
    Student,
    Teacher,
    Workspace,
    assert_model_record_facts,
    evaluate_checkpoint,
    export_checkpoint,
    generate_kd,
    listed_digest,
    make_workspace,
    manifest_of,
    prepare_student_run,
    smoke_check,
    train_second_student,
    train_student,
    train_teacher,
)
from tests.support.tiny_parallel import train_toy_spm

pytestmark = pytest.mark.subprocess

SECOND_SPM_PIECES: Final = 500

# The gate. They are never lowered: a miss is answered with more steps.
MIN_KD_ROWS: Final = 3550
MIN_REFERENCE_SHARE: Final = 0.9
MIN_TEACHER_CHRF: Final = 90.0
MIN_STUDENT_CHRF: Final = 95.0
MAX_STUDENT_SHORTFALL: Final = 3.0
MIN_CT2_MATCH_SHARE: Final = 0.95


# ---------------------------------------------------------------------------------
# The fixtures: each runs one stage of `tests/support/distill_gate_stages.py` once per
# module, in the order the scenarios need.


@pytest.fixture(scope="module", autouse=True)
def gpu_hidden() -> Iterator[None]:
    """No device is visible to any process of this module: Eole, the converter and the
    CTranslate2 translator of the test inherit the empty list."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    return make_workspace(tmp_path_factory.mktemp("distill_gate"))


@pytest.fixture(scope="module")
def teacher(workspace: Workspace) -> Teacher:
    return train_teacher(workspace)


@pytest.fixture(scope="module")
def teacher_export(workspace: Workspace, teacher: Teacher) -> Exported:
    return export_checkpoint(
        workspace, teacher.averaged, workspace.root / "teacher_ct2"
    )


@pytest.fixture(scope="module")
def kd(workspace: Workspace, teacher_export: Exported) -> Generated:
    return generate_kd(workspace, teacher_export)


@pytest.fixture(scope="module")
def prepared(workspace: Workspace, kd: Generated) -> Prepared:
    return prepare_student_run(workspace, kd)


@pytest.fixture(scope="module")
def student(prepared: Prepared) -> Student:
    return train_student(prepared)


@pytest.fixture(scope="module")
def evaluations(
    workspace: Workspace, teacher: Teacher, student: Student
) -> Evaluations:
    return Evaluations(
        teacher=evaluate_checkpoint(workspace, teacher.averaged, "eval_teacher"),
        student=evaluate_checkpoint(workspace, student.final_step, "eval_student"),
        student_run_directory=evaluate_checkpoint(
            workspace, student.run_dir / "ckpt", "eval_student_ckpt"
        ),
    )


@pytest.fixture(scope="module")
def student_export(workspace: Workspace, student: Student) -> Exported:
    return export_checkpoint(
        workspace, student.final_step, workspace.root / "student_ct2"
    )


@pytest.fixture(scope="module")
def smoke(workspace: Workspace, student: Student, student_export: Exported) -> Smoke:
    return smoke_check(workspace, student, student_export)


@pytest.fixture(scope="module")
def second_spm(workspace: Workspace) -> Path:
    return train_toy_spm(workspace.root / "spm500", vocab_size=SECOND_SPM_PIECES)


@pytest.fixture(scope="module")
def second_student(
    workspace: Workspace, kd: Generated, second_spm: Path
) -> SecondStudent:
    return train_second_student(workspace, kd, second_spm)


@pytest.fixture(scope="module")
def refusals(
    workspace: Workspace,
    teacher: Teacher,
    teacher_export: Exported,
    kd: Generated,
    student: Student,
    second_spm: Path,
) -> dict[str, Refusal]:
    return run_refusals(
        RefusalInputs(
            scratch=workspace.root / "refusals",
            corpus=workspace.inputs.corpus,
            spm_model=workspace.inputs.spm_model,
            other_spm_model=second_spm,
            student_base=STUDENT_BASE,
            teacher_run=teacher.run_dir,
            teacher_export=teacher_export.directory,
            kd_directory=kd.directory,
            student_step=student.final_step,
        )
    )


# =================================================================================
# T: the teacher.


def test_the_tiny_teacher_trains_2000_steps_and_two_checkpoints_are_averaged(
    teacher: Teacher, record_property: Any
) -> None:
    record_property("launch_seconds", round(teacher.launch.seconds, 1))
    record_property("prepare_seconds", round(teacher.prepare_seconds, 1))
    record_property("average_seconds", round(teacher.average.seconds, 1))

    assert teacher.launch.facts["status"] == "completed"
    assert teacher.launch.facts["steps"] == str(STEPS)
    assert teacher.saved_steps == [500, 1000, 1500, 2000]
    assert teacher.curve.non_finite_step is None
    assert "averaged: 2 checkpoints (steps 1500, 2000)" in teacher.average.stdout
    assert manifest_of(teacher.averaged).stage == "train_average"


# X: the teacher's export.


def test_the_teacher_export_holds_the_five_files_and_a_manifest_that_vouches_for_them(
    teacher_export: Exported, record_property: Any
) -> None:
    record_property("seconds", round(teacher_export.invocation.seconds, 1))
    directory = teacher_export.directory
    manifest = teacher_export.manifest

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [*EXPORT_FILES, "manifest.json"]
    )
    assert manifest.stage == "export"
    assert manifest.status["export"] == "ok"
    assert manifest.status["tag_placement_check"] == "checked"
    assert {Path(item.path).name for item in manifest.outputs} == set(EXPORT_FILES)
    for name in EXPORT_FILES:
        assert listed_digest(manifest, name) == sha256_of(directory / name), name
    assert f"exported to {directory}" in teacher_export.invocation.stdout
    record_property("model_bytes", (directory / "model.bin").stat().st_size)


def test_the_teacher_export_record_says_where_the_model_came_from(
    workspace: Workspace, teacher: Teacher, teacher_export: Exported
) -> None:
    record = teacher_export.record
    checkpoint = record["checkpoint"]
    averaged = manifest_of(teacher.averaged)

    assert record["tag_placement"] == "src"
    assert_model_record_facts(teacher_export, DIRECTIONS)
    assert (record["encoder_layers"], record["decoder_layers"]) == (2, 2)
    assert record["hidden_size"] == 64
    assert record["vocabulary_entries"] == 344
    assert checkpoint["kind"] == "averaged"
    assert checkpoint["step"] is None
    assert checkpoint["manifest_stage"] == "train_average"
    assert checkpoint["weights_sha256"] == listed_digest(averaged, WEIGHTS)
    assert checkpoint["weights_sha256"] == sha256_of(teacher.averaged / WEIGHTS)
    assert record["spm_sha256"] == sha256_of(workspace.inputs.spm_model)
    assert sha256_of(teacher_export.directory / "spm.model") == record["spm_sha256"]


def test_the_exported_vocabulary_keeps_every_reserved_symbol(
    teacher_export: Exported,
) -> None:
    entries = json.loads(
        (teacher_export.directory / "shared_vocabulary.json").read_text("utf-8")
    )
    symbols = user_defined_symbols()

    assert len(symbols) == 118
    assert len(entries) == teacher_export.record["vocabulary_entries"]
    assert set(symbols) <= set(entries)


# G: the KD corpus.


def test_generate_sends_every_authentic_training_source_and_keeps_almost_all(
    workspace: Workspace, kd: Generated, record_property: Any
) -> None:
    record_property("seconds", round(kd.invocation.seconds, 1))
    total = kd.counts["counts"]["total"]
    record_property("kd_rows", kd.frame.height)

    assert len(workspace.toy.train) == 3600
    assert total["sources"] == 3600
    assert kd.frame.height == total["selected"]
    assert kd.frame.height >= MIN_KD_ROWS
    assert kd.counts["teacher_counters"] == {
        "sources": 3600,
        "source_too_long": 0,
        "empty_source": 0,
        "truncated": 0,
        "line_break": 0,
        "reserved_symbol": 0,
    }
    assert kd.counts["letterless_sources"]["total"] == 0


def test_the_counts_of_a_generation_are_conserved_in_every_direction(
    kd: Generated,
) -> None:
    by_scope = kd.counts["counts"]

    assert set(by_scope) == {*DIRECTIONS, "total"}
    for scope, counts in by_scope.items():
        assert set(counts) == set(COUNT_KEYS), scope
        assert counts["sources"] == (
            counts["no_candidates"] + counts["no_survivor"] + counts["selected"]
        ), scope
        # One hypothesis per source (n = 1): every survivor is a winner.
        assert counts["candidates"] == (
            counts["dropped_empty"]
            + counts["dropped_purity"]
            + counts["dropped_copy"]
            + counts["dropped_qe"]
            + counts["selected"]
        ), scope
    for key in COUNT_KEYS:
        assert by_scope["total"][key] == sum(by_scope[d][key] for d in DIRECTIONS), key
    shards = kd.counts["shard_stats"]
    assert shards == {
        "planned": 10,
        "reused": 0,
        "generated": 10,
        "regenerated_invalid": 0,
    }


def test_every_kd_row_is_a_train_row_of_origin_ft_with_the_label_of_the_teacher(
    teacher_export: Exported, kd: Generated
) -> None:
    frame = kd.frame
    model_digest = sha256_of(teacher_export.directory / "model.bin")

    assert set(frame["origin"].cast(str).to_list()) == {"ft"}
    assert set(frame["split"].cast(str).to_list()) == {"train"}
    assert frame["corpus"].unique().to_list() == [f"ft:rinchen-{model_digest[:12]}"]
    assert re.fullmatch(r"ft:rinchen-[0-9a-f]{12}", frame["corpus"][0])
    assert frame["filter_reason"].null_count() == frame.height
    assert frame["doc_id"].null_count() == frame.height
    assert frame["tgt"].str.strip_chars().str.len_chars().min() > 0
    assert set(zip(frame["src_lang"], frame["tgt_lang"], strict=True)) == {
        tuple(direction.split("-")) for direction in DIRECTIONS
    }


def test_most_kd_targets_equal_an_authentic_target_of_their_source(
    workspace: Workspace, kd: Generated
) -> None:
    references: dict[tuple[str, str, str], set[str]] = {}
    for row in workspace.toy.train:
        references.setdefault(
            (row["src_lang"], row["tgt_lang"], row["src"]), set()
        ).add(row["tgt"])
    frame = kd.frame
    equal = sum(
        target in references[(src_lang, tgt_lang, source)]
        for src_lang, tgt_lang, source, target in zip(
            frame["src_lang"],
            frame["tgt_lang"],
            frame["src"],
            frame["tgt"],
            strict=True,
        )
    )

    assert equal == kd.counts["matches_reference"]["total"]
    assert equal / frame.height >= MIN_REFERENCE_SHARE


def test_no_kd_source_is_a_dev_a_test_or_a_back_translated_sentence(
    workspace: Workspace, kd: Generated
) -> None:
    toy = workspace.toy
    kd_sources = set(kd.frame["src"].to_list())
    corpus_file = scan_corpus(workspace.inputs.corpus).collect()
    in_the_file = set(
        corpus_file.filter(corpus_file["split"].cast(str) == "dev")["src"].to_list()
    )
    dev_sources = {row["src"] for row in toy.dev}
    test_sources = {row["src"] for row in toy.test}
    train_pairs = {(r["src_lang"], r["tgt_lang"], r["src"]) for r in toy.train}
    kd_pairs = set(
        zip(kd.frame["src_lang"], kd.frame["tgt_lang"], kd.frame["src"], strict=True)
    )

    # The corpus file DOES hold the dev rows; they never reach the generator.
    assert dev_sources == in_the_file
    assert dev_sources and test_sources
    assert not kd_sources & dev_sources
    assert not kd_sources & test_sources
    assert not kd_sources & {row["src"] for row in toy.bt}
    assert kd_pairs <= train_pairs


def test_the_generate_manifest_vouches_for_the_teacher_and_the_outputs(
    teacher_export: Exported, kd: Generated
) -> None:
    manifest = kd.manifest
    record = teacher_export.record
    teacher_files = {
        Path(item.path).name: item.sha256
        for item in manifest.inputs
        if Path(item.path).parent == teacher_export.directory
    }

    assert manifest.stage == "distill_generate"
    assert manifest.status["generate"] == "ok"
    assert manifest.status["teacher.tag_placement"] == "src"
    assert (
        manifest.status["teacher.checkpoint_weights_sha256"]
        == record["checkpoint"]["weights_sha256"]
    )
    assert (
        manifest.status["teacher.checkpoint_manifest_sha256"]
        == record["checkpoint"]["manifest_sha256"]
    )
    assert manifest.status["teacher.model_sha256"] == teacher_files["model.bin"]
    assert teacher_files == {
        **{name: listed_digest(teacher_export.manifest, name) for name in EXPORT_FILES},
        "manifest.json": sha256_of(teacher_export.directory / "manifest.json"),
    }
    assert {Path(item.path).name: item.rows for item in manifest.outputs} == {
        "kd.parquet": kd.frame.height,
        "counts.json": None,
    }
    assert listed_digest(manifest, "kd.parquet") == sha256_of(
        kd.directory / "kd.parquet"
    )
    assert manifest.metrics["count.total.selected"] == kd.frame.height


def test_a_rerun_reuses_every_shard_and_gives_the_same_rows(kd: Generated) -> None:
    assert kd.rerun.returncode == 0
    assert "shards: 10 (reusable 10)" in kd.rerun.stdout
    assert kd.rerun_counts["shard_stats"] == {
        "planned": 10,
        "reused": 10,
        "generated": 0,
        "regenerated_invalid": 0,
    }
    assert kd.rerun_counts["fingerprint"] == kd.counts["fingerprint"]
    assert kd.rerun_counts["counts"] == kd.counts["counts"]
    assert kd.rerun_frame.drop("created").equals(kd.frame.drop("created"))


# P: the student's data and config.


def test_prepare_writes_twelve_corpora_and_a_manifest_of_stage_distill_prepare(
    prepared: Prepared, record_property: Any
) -> None:
    record_property("seconds", round(prepared.invocation.seconds, 1))
    facts = prepared.invocation.facts

    assert facts["corpora"] == "12 (reused 0)"
    assert facts["validation lines"] == str(len(DIRECTIONS) * DEV_LINES)
    assert facts["config"] == str(prepared.out_dir / "config.yaml")
    assert facts["mix"] == "1:1"
    assert facts["kd tag"] == "ft"
    assert prepared.manifest.stage == "distill_prepare"
    assert prepared.manifest.status["prepare"] == "ok"
    assert prepared.manifest.metrics["kd.count.rows_kept"] == 3599
    assert prepared.manifest.metrics["count.dev.dev_in_train"] == 0


def test_every_kd_corpus_is_tagged_ft_after_the_direction_tag(
    prepared: Prepared,
) -> None:
    data = prepared.config["data"]
    kd_corpora = {name: spec for name, spec in data.items() if name.endswith(".kd")}

    assert sorted(kd_corpora) == [f"{d}.kd" for d in DIRECTIONS]
    for name, spec in kd_corpora.items():
        target = name.removesuffix(".kd").split("-")[1]
        assert spec["src_prefix"] == f"{direction_tag(target)} <ft>", name
        assert spec["tgt_prefix"] == "", name
    originals = [
        spec
        for name, spec in data.items()
        if name != "valid" and name not in kd_corpora
    ]
    assert len(originals) == 8
    assert all("<ft>" not in spec["src_prefix"] for spec in originals)


def test_the_weight_of_each_kd_corpus_is_exactly_the_sum_of_the_originals_of_its_direction(
    prepared: Prepared,
) -> None:
    data = prepared.config["data"]
    weights = {name: spec["weight"] for name, spec in data.items() if name != "valid"}

    assert weights == {
        "eng_Latn-khk_Cyrl.auth": 22,
        "eng_Latn-khk_Cyrl.bt": 6,
        "khk_Cyrl-eng_Latn.auth": 22,
        "khk_Cyrl-eng_Latn.bt": 6,
        "khk_Cyrl-rus_Cyrl.auth": 15,
        "khk_Cyrl-rus_Cyrl.bt": 7,
        "rus_Cyrl-khk_Cyrl.auth": 15,
        "rus_Cyrl-khk_Cyrl.bt": 7,
        "eng_Latn-khk_Cyrl.kd": 28,
        "khk_Cyrl-eng_Latn.kd": 28,
        "khk_Cyrl-rus_Cyrl.kd": 22,
        "rus_Cyrl-khk_Cyrl.kd": 22,
    }
    for direction in DIRECTIONS:
        assert weights[f"{direction}.kd"] == (
            weights[f"{direction}.auth"] + weights[f"{direction}.bt"]
        ), direction
    assert sum(weights.values()) == 200


def test_the_student_config_trains_from_random_init_with_the_pieces_of_its_model(
    workspace: Workspace, prepared: Prepared
) -> None:
    config = prepared.config

    assert "train_from" not in config["training"]
    assert "reset_optim" not in config["training"]
    assert "train_from" not in config
    assert "reset_optim" not in config
    assert config["src_vocab_size"] == workspace.processor.get_piece_size() + 1 == 641
    assert (
        config["model"]["embeddings"]["position_encoding_type"]
        == "SinusoidalInterleaved"
    )
    assert (
        config["model"]["encoder"]["layers"],
        config["model"]["decoder"]["layers"],
    ) == (2, 1)
    assert config["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": str(workspace.inputs.spm_model),
        "tgt_subword_model": str(workspace.inputs.spm_model),
    }


def _printed_weights(invocation: Invocation) -> dict[str, int]:
    pattern = re.compile(r"corpus (\S+): rows \d+, weight (\d+)")
    return {
        match[1]: int(match[2])
        for match in map(pattern.fullmatch, invocation.stdout)
        if match
    }


def test_a_dry_run_of_three_to_one_prints_a_ratio_of_exactly_three_and_writes_nothing(
    prepared: Prepared,
) -> None:
    printed = _printed_weights(prepared.three_to_one)

    for direction in DIRECTIONS:
        originals = printed[f"{direction}.auth"] + printed[f"{direction}.bt"]
        assert printed[f"{direction}.kd"] == 3 * originals, direction
    assert printed["eng_Latn-khk_Cyrl.kd"] == 84
    assert printed["khk_Cyrl-rus_Cyrl.kd"] == 66
    assert prepared.three_to_one.facts["mix"] == "3:1"
    assert "nothing was written" in prepared.three_to_one.stdout
    assert not prepared.three_to_one_out_dir.exists()
    assert _printed_weights(prepared.three_to_one_over_the_run) == printed
    assert prepared.digest_after_dry_run == prepared.digest_before_dry_run


# S: the student's training.


def test_the_student_trains_2000_steps_from_random_init_with_two_encoder_and_one_decoder_layer(
    student: Student, record_property: Any
) -> None:
    record_property("seconds", round(student.launch.seconds, 1))
    model = student.saved_config["model"]
    record_property("xent_first", student.curve.reports[0].xent)
    record_property("xent_last", student.curve.reports[-1].xent)

    assert student.launch.facts["status"] == "completed"
    assert student.launch.facts["steps"] == str(STEPS)
    assert student.launch.facts["resumed from"] == "none"
    assert student.launch.facts["checkpoint"] == str(student.final_step)
    assert student.saved_steps == [500, 1000, 1500, 2000]
    assert (model["encoder"]["layers"], model["decoder"]["layers"]) == (2, 1)
    assert student.curve.non_finite_step is None
    assert student.curve.reports[-1].xent < student.curve.reports[0].xent


def test_nothing_of_the_student_run_names_a_checkpoint_to_start_from(
    student: Student,
) -> None:
    for text in (
        json.dumps(student.effective_config),
        json.dumps(student.launch_record),
        (student.run_dir / "launch" / "train-1.yaml").read_text("utf-8"),
        (student.run_dir / "launch.json").read_text("utf-8"),
    ):
        assert "train_from" not in text
        assert "reset_optim" not in text
    assert student.launch_manifest.stage == "train_launch"


# E: teacher against student.


def test_the_student_is_within_three_chrf_of_the_teacher_and_above_95(
    evaluations: Evaluations, record_property: Any
) -> None:
    teacher, student = evaluations.teacher.result, evaluations.student.result
    record_property("teacher_chrf", round(teacher["chrf_mean"], 2))
    record_property("student_chrf", round(student["chrf_mean"], 2))
    record_property("teacher_min_chrf", round(teacher["chrf_min"], 2))
    record_property("student_min_chrf", round(student["chrf_min"], 2))
    record_property(
        "student_by_direction",
        {d: round(v["chrf"], 2) for d, v in student["directions"].items()},
    )
    record_property(
        "teacher_by_direction",
        {d: round(v["chrf"], 2) for d, v in teacher["directions"].items()},
    )

    assert teacher["chrf_mean"] >= MIN_TEACHER_CHRF
    assert student["chrf_mean"] >= MIN_STUDENT_CHRF
    assert student["chrf_mean"] >= teacher["chrf_mean"] - MAX_STUDENT_SHORTFALL


def test_both_evaluations_score_the_same_dev_lines_without_a_leak(
    workspace: Workspace, evaluations: Evaluations, teacher: Teacher, student: Student
) -> None:
    left, right = evaluations.teacher, evaluations.student

    assert sorted(left.result["directions"]) == sorted(DIRECTIONS)
    assert (
        {d: v["lines"] for d, v in left.result["directions"].items()}
        == {d: v["lines"] for d, v in right.result["directions"].items()}
        == dict.fromkeys(DIRECTIONS, DEV_LINES)
    )
    for evaluated in (left, right):
        assert evaluated.result["tag_leaks"] == 0
        assert evaluated.result["placement"] == "src"
        assert evaluated.result["max_lines"] == DEV_LINES
        assert evaluated.result["spm_sha256"] == sha256_of(workspace.inputs.spm_model)
        assert evaluated.invocation.facts["tag leaks"] == "0"
        assert evaluated.invocation.facts["dev lines in training"] == "0"
        assert evaluated.manifest.status["tag_placement_check"] == "checked"
        assert evaluated.manifest.stage == "distill_evaluate"
        assert evaluated.manifest.metrics["dev_in_train"] == 0
    assert left.result["checkpoint"] == str(teacher.averaged)
    assert right.result["checkpoint"] == str(student.final_step)


def test_the_ckpt_directory_of_a_run_is_scored_as_its_newest_step(
    evaluations: Evaluations, student: Student
) -> None:
    link, step = evaluations.student_run_directory, evaluations.student

    assert link.result["checkpoint"] == str(student.run_dir / "ckpt")
    assert link.result["directions"] == step.result["directions"]
    assert link.result["chrf_mean"] == step.result["chrf_mean"]
    for direction in DIRECTIONS:
        name = f"hyp.{direction}.txt"
        assert (link.out_dir / name).read_text("utf-8") == (
            step.out_dir / name
        ).read_text("utf-8"), name


# Q: the student's export and CTranslate2 against Eole.


def test_the_student_export_records_a_two_one_layer_model_made_from_the_final_step(
    student: Student,
    student_export: Exported,
    teacher_export: Exported,
    record_property: Any,
) -> None:
    record_property("seconds", round(student_export.invocation.seconds, 1))
    directory = student_export.directory
    record = student_export.record
    checkpoint = record["checkpoint"]
    student_bytes = (directory / "model.bin").stat().st_size
    teacher_bytes = (teacher_export.directory / "model.bin").stat().st_size
    record_property("student_model_bytes", student_bytes)
    record_property("teacher_model_bytes", teacher_bytes)

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [*EXPORT_FILES, "manifest.json"]
    )
    assert (record["encoder_layers"], record["decoder_layers"]) == (2, 1)
    assert record["tag_placement"] == "src"
    assert_model_record_facts(student_export, DIRECTIONS)
    assert checkpoint["kind"] == "step"
    assert checkpoint["step"] == STEPS
    assert checkpoint["manifest_stage"] == "train_launch"
    assert checkpoint["weights_sha256"] == listed_digest(
        student.launch_manifest, WEIGHTS
    )
    assert student_export.manifest.stage == "export"
    for name in EXPORT_FILES:
        assert listed_digest(student_export.manifest, name) == sha256_of(
            directory / name
        ), name
    assert student_bytes < teacher_bytes


def test_ctranslate2_and_eole_write_the_same_greedy_output_for_200_dev_sentences(
    smoke: Smoke, record_property: Any
) -> None:
    record_property("equal_pieces", smoke.equal_pieces)
    record_property("equal_text", smoke.equal_text)
    record_property("first_mismatches", smoke.first_mismatches)

    assert smoke.sentences == 200
    assert smoke.equal_pieces / smoke.sentences >= MIN_CT2_MATCH_SHARE


def test_the_exported_student_translates_an_english_sentence_into_cyrillic(
    smoke: Smoke, record_property: Any
) -> None:
    record_property("star_text", smoke.star_text)

    assert smoke.star_text.strip()
    assert is_cyrillic_text(smoke.star_text)


# V: a student with its own SentencePiece model.


def test_a_student_with_a_500_piece_model_states_501_entries_and_cuts_its_validation_with_it(
    workspace: Workspace, second_student: SecondStudent, kd: Generated
) -> None:
    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(second_student.spm_model)
    )
    first_line = (
        (second_student.out_dir / "valid.src").read_text("utf-8").splitlines()[0]
    )
    tag, *pieces = first_line.split(" ")
    text = processor.decode(pieces)

    assert processor.get_piece_size() == SECOND_SPM_PIECES
    assert second_student.config["src_vocab_size"] == SECOND_SPM_PIECES + 1
    assert second_student.config["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": str(second_student.spm_model),
        "tgt_subword_model": str(second_student.spm_model),
    }
    # The line is a dev sentence cut by the 500-piece model, not by the 640-piece one.
    assert tag == "<2khk_Cyrl>"
    assert pieces == processor.encode(text, out_type=str)
    assert pieces != workspace.processor.encode(text, out_type=str)
    assert text in {row["src"] for row in workspace.toy.dev}


def test_the_second_student_trains_and_its_vocabulary_holds_every_reserved_symbol(
    second_student: SecondStudent, record_property: Any
) -> None:
    record_property("seconds", round(second_student.launch.seconds, 1))
    vocabulary = second_student.run_dir / "vocab.src"
    listed = {
        line.split("\t")[0]
        for line in vocabulary.read_text("utf-8").splitlines()
        if line
    }

    assert second_student.launch.facts["status"] == "completed"
    assert second_student.launch.facts["steps"] == str(SECOND_STUDENT_STEPS)
    assert second_student.curve.non_finite_step is None
    assert set(user_defined_symbols()) <= listed


# R: the refusals.

REFUSALS: Final = {
    "export_a_step_without_its_run": "no launch.json within 3 levels",
    "export_a_flipped_weights_byte": "differs from the manifest",
    "export_a_non_final_step": "only the final step of a finished run is vouched for",
    "export_another_sentencepiece_model": "is not the SentencePiece model the checkpoint was trained with",
    "export_into_an_existing_directory": "exists: remove it",
    "generate_a_model_with_a_flipped_byte": "differs from the manifest",
    "generate_an_export_without_its_manifest": "no manifest vouches for it",
    "generate_a_vocabulary_without_a_tag": "<2eng_Latn> is not an entry of",
    "generate_n_above_the_beam": "--n (5) must not exceed --beam-size (4)",
    "prepare_a_kd_file_with_a_flipped_byte": "the manifest vouches for",
    "prepare_a_kd_file_without_its_manifest": "has no manifest.json beside it",
    "prepare_a_mix_of_zero": "--mix '0:1' is not K:O",
    "prepare_a_base_with_train_from": "training.train_from",
    "prepare_a_stated_vocabulary_size_below_the_pieces": "src_vocab_size: 100 is below 641",
    "evaluate_a_checkpoint_path_with_a_bracket": "holds '['",
}


def test_every_refusal_of_the_gate_has_a_case_and_every_case_a_refusal(
    refusals: dict[str, Refusal],
) -> None:
    assert set(refusals) == set(REFUSALS)


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_damaged_input_is_refused_with_exit_2_before_any_tool_starts(
    refusals: dict[str, Refusal], name: str
) -> None:
    refusal = refusals[name]

    assert refusal.invocation.returncode == 2, refusal.invocation.stderr
    assert refusal.invocation.stderr.startswith("error: ")
    assert REFUSALS[name] in refusal.invocation.stderr
    assert refusal.inputs_unchanged
    assert refusal.converter_logs == []
    # Nothing is written into an output directory. A vocabulary without a tag is found
    # when the teacher is loaded, after the directory was made, so that one may exist, and
    # empty; the existing directory a refused export was given keeps what was in it.
    expected = ["kept.txt"] if name == "export_into_an_existing_directory" else []
    assert refusal.out_dir_files == expected


# D: the GPU stayed hidden.


def test_the_eole_logs_show_a_cpu_run_and_the_converter_logs_no_device(
    teacher: Teacher,
    student: Student,
    second_student: SecondStudent,
    teacher_export: Exported,
    student_export: Exported,
    evaluations: Evaluations,
) -> None:
    for log in (teacher.log, student.log, second_student.log):
        assert "GPU ranks: []" in log
        assert "Starting training in cpu mode" in log
        assert "nccl" not in log.lower()
        assert "cuda:" not in log
    for run_dir in (teacher.run_dir, student.run_dir, second_student.run_dir):
        effective = read_config(run_dir / "launch" / "train-1.yaml", "config")
        assert effective["training"]["gpu_ranks"] == []
        assert effective["training"]["world_size"] == 1
    for exported in (teacher_export, student_export):
        converter_log = exported.converter_log.read_text("utf-8", errors="replace")
        assert "cuda" not in converter_log.lower()
        assert "device" not in converter_log.lower()
    for evaluated in (
        evaluations.teacher,
        evaluations.student,
        evaluations.student_run_directory,
    ):
        prediction_log = (evaluated.out_dir / "dev.log").read_text(
            "utf-8", errors="replace"
        )
        assert "device=cpu" in prediction_log
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_the_test_code_never_asks_a_device_question_and_builds_ctranslate2_on_the_cpu() -> (
    None
):
    probe = "nvidia" + "-smi"
    cuda_namespace = "torch" + ".cuda"
    here = source_findings(Path(__file__))
    support = source_findings(Path(distill_gate.__file__))

    for found in (here, support):
        assert not [text for text in found.strings if probe in text]
        assert cuda_namespace not in found.dotted_names
        assert set(found.translator_devices) <= {"'cpu'"}
    assert support.translator_devices == ["'cpu'"]
