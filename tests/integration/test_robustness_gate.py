"""The D12 STOP gate: the robustness metrics measured on a real model, through the real
`rinchen eval robustness` binary. Everything runs with the real Eole (training), the real
CTranslate2 (the engine of the command and the D5 converter) and the real SentencePiece on the
CPU; nothing here touches a GPU (`CUDA_VISIBLE_DEVICES` is the empty string for the session, Eole
trains with `gpu_ranks: []`, the command runs on `cpu` and `auto` is refused).

The data are the robustness toy of `tests/support/robust_toy.py`: templated sentences, numbers,
placeholders, names and traditional-script sources over the D7 nouns. A tiny model trained on them
for 6,000 steps reads and writes them, so what the six metrics say about it can be checked; it
ranks nothing (a model this size on templated text is not evidence about a real model, a real
corpus, a real glossary or a GPU) and every figure below is a measurement of the machinery.

Session fixtures (`tests/integration/conftest.py`) pay for the expensive stages once: the toy
teacher (6,000 steps, averaged over its last two saves, `rinchen export convert` to float32), the
D11 teacher that never saw a digit or a placeholder (`d11_tiny_export_dir`) and the D5 tiny script
converter (`tiny_run`, `tiny_ct2`). The suites are generated from the toy
(`tests/support/robust_suites.py`); nothing is committed but the golden files.

Gates, every threshold a literal of this file and never lowered (a miss of a floor is answered with
more steps of the toy, `ROBUST_TEACHER_STEPS`, which is deterministic here):

    R1  `rinchen eval robustness --suite main --model export` with the defaults (beam 4, mask
        `retry`, float32): exit 0, `report.json` and `robustness.json` agree on the six values,
        every request is answered `ok`, no row is excluded, and numbers at least 0.90 (each of
        the seven single-number categories at least 0.95), placeholders 0.99, entities 0.90,
        terms 0.99, script purity 0.999, `mn_Mong` source retention 0.95; the manifest lists
        the suite, the export and the files written with the digests of the bytes
    R2  non-vacuity: the hypotheses of R1 with the texts moved 7 rows inside each (section, arm)
        group and scored through `--hypotheses`: numbers, placeholders, terms at most 0.02,
        entities at most 0.10; with only the `mong` arm moved, retention at most 0.2
    R3  the D11 teacher (never trained on a digit or a placeholder) over the main suite:
        placeholders at most 0.02, numbers at most 0.02, retention at most 0.5
    R4  a glossary the model never writes (the term control suite): term accuracy at most 0.02
    R5  placeholder indices 10 and up, which no row of the toy trains: at most 0.2 (all four
        directions are recorded)
    R6  script masks over the main suite: `off` leaves at least 50 `script` rows below the
        purity floor and a mean below 0.97; `always` and `retry` none and a mean of at least
        0.999; the numbers, placeholders, entities, terms and retention move by at most 0.02
    R7  rescoring: `--hypotheses` over R1's `hypotheses.jsonl` gives the same `robustness`,
        `sections`, `rows` and `suite`; the command run twice with `--model` writes equal bytes
    R8  the D5 tiny converter attached: the five other sections equal R1's, the converted arm
        retains less than the direct arm, a traditional-script output suite is pure (at least
        0.99) and every request `ok`; without a converter that suite exits 2 and writes nothing
    R9  refusals through the real binary: an export with a flipped byte (2), a suite with an
        unknown key (3), `--device auto` (2), each with nothing written
        (`tests/integration/test_robustness_gate_refusals.py`)
    R10 no GPU: an AST scan of the D12 code and tests, every command in a process that hides the
        devices (and one that does not is refused before it starts), every subprocess bounded
        by 900 seconds (the same file); the Eole log says `GPU ranks: []` and `cpu mode` and the
        command's own options say `cpu` (here)
    R11 the exclusions: the hand-made golden suite and hypotheses (`tests/fixtures/robustness`)
        through the real binary give the golden `robustness.json` byte for byte: the rows no
        metric may score (a number-free reference, a source whose numbers the reference does not
        keep, a text with no letter, a glossary term the source lacks) are counted under their
        reason and never counted as passes

Measured on this machine (CPU, one thread per command, Eole 0.6.4, CTranslate2 4.8.4, SentencePiece
and sacrebleu as locked, 2026-11-10; the machine was shared with other work, so a time moves by a
third from one run to the next, a count or a score never does: the toy's training is bit-for-bit
reproducible, two trainings gave the same `model.bin`, SHA-256 `e28d1566...a6`, 2,080,071 bytes, and
every count and score below equals what the design spike measured except where it says so):

    setup  the toy teacher (6,000 steps, saves 4,500 to 6,000 kept, the last two averaged): launch
           180 s (215 s in a scratch run on a loaded machine; the spike 185 to 193 s), prepare 0.9 s,
           average 0.8 s, `export convert` about 4 s, 186 s for the fixture; the two D11 teachers
           and their four exports 125 s; the D5 converter (`tiny_run`, `tiny_ct2`) 45 s; the 31
           tests of the two modules 6 min 14 s in all, 3 min 6 s of it the toy teacher. Every
           command of the gate takes 0.8 to 2.3 s (the main suite is 4,060 requests at beam 4
           in 2.0 s, import and model load included)
    R1     beam 4, mask `retry`, float32, 2.0 s: numbers 892 of 960 = 0.9292 (the design spike's
           prototype said 895, 0.9323: it ignored the currency unit, and the model drops the tugrik
           in three rows; the 0.90 floor stands), categories int 120/120, decimal 120/120, percent
           120/120, date 120/120, time 120/120, currency 117/120, grouped 118/120, and two numbers
           in one sentence 57/120 (no floor: the hard rows); placeholders 599 of 600 = 0.9983 (one
           `mn_Mong` row loses a placeholder at beam 4; the spike: none at beam 1), entities 447 of 480
           mentions = 0.9312 (208 of 240 rows with both found), terms 1,044 of 1,044 mentions =
           1.0, script purity 1.0 over 900 rows (0 below the floor), `mn_Mong` retention 0.9976
           (chrF 99.49 for the traditional-script arm over 99.73 for the Cyrillic one, 467 of 480
           texts equal); every request `ok` (960, 600, 240, 400, 900 and 960 hypotheses), no row
           excluded, the scored counts 960, 600, 480 mentions, 1,044 mentions, 900, 480
    R2     the hypotheses of R1 moved 7 rows inside each group: numbers 3 of 960, placeholders 2 of
           600, entities 10 of 480 (0.0208), terms 1 of 1,044; only the `mong` arm moved: retention
           0.0598 (a rotated Cyrillic arm gave 0.06 in the spike)
    R3     the D11 teacher over the same suite: numbers 0 of 960, placeholders 0 of 600, entities 0
           of 480, terms 1,041 of 1,044 (the words are its own), retention 0.3172 (0 of 480 texts
           equal)
    R4     a glossary the model never writes: 0 of 1,044 mentions
    R5     placeholder indices 10 and up: 0 of 7 in each of the four directions (`eng_Latn-khk_Cyrl`,
           `khk_Cyrl-eng_Latn`, `khk_Cyrl-rus_Cyrl`, `rus_Cyrl-khk_Cyrl`), 0 of 28 in all
    R6     mask `off`: 101 of the 900 `script` rows below the 0.98 floor (all in the 500 untrained
           requests, 399 of them pure: 99 of 200 for English, 100 of 100 for Khalkha, 200 of 200
           for Russian), mean purity 0.9389, minimum 0.0; `retry` and `always`: none below, mean
           1.0; with `off` the other five values equal R1's, with `always` (the other scripts
           suppressed throughout) numbers 891 of 960, placeholders 598 of 600, entities 444 of 480
           (0.925), terms 1.0, retention 0.9977: every move well under 0.02
    R7     rescoring R1's `hypotheses.jsonl`: `robustness`, `sections`, `rows` and `suite` equal
           (the `system` names no model); the command run twice with `--model`: the three files
           equal byte for byte
    R8     with the D5 tiny converter: the five other sections equal R1's; the converted arm
           retains 0.5473 (chrF 54.58 against 99.49 for the direct arm and 99.73 for the Cyrillic
           one; the spike's 0.562 against 1.003 was beam 1); the traditional-script output suite
           (100 rows): purity 1.0, all `ok`; without a converter it exits 2 with nothing written
    R9, R10, R11  as stated: the golden suite scores to the golden `robustness.json` byte for byte

What the toy cannot say (unmeasured): anything about a real model, a real corpus, real numbers,
names or glossaries in Mongolian, Russian or English, a real converter or a mechanism that places a
placeholder or a term (none exists); anything on a GPU (CTranslate2 on CUDA, float16,
int8_float16); a run in CI. The model ranks nothing, int8 was not measured here and nothing here
says it is lossless.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.distill_gate import sha256_of
from tests.support.distill_gate_stages import Teacher, manifest_of
from tests.support.robust_gate import (
    HYPOTHESES_FILE,
    MANIFEST_FILE,
    MONG_ARM,
    REPORT_FILE,
    RESULT_FILE,
    ROBUST_TEACHER_STEPS,
    Run,
    Suites,
    hypotheses_run,
    model_run,
    rotate_hypotheses,
    run_robustness,
)
from tests.support.robust_golden import GOLDEN_HYPOTHESES, GOLDEN_RESULT, GOLDEN_SUITE
from tests.support.tiny_converter import TinyRun

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden"),
]

# The gates. They are never lowered: a miss is answered with more steps of the toy.
MIN_NUMBERS_ACC: Final = 0.90
MIN_NUMBER_CATEGORY_ACC: Final = 0.95
MIN_PLACEHOLDER_ACC: Final = 0.99
MIN_ENTITY_ACC: Final = 0.90
MIN_TERM_ACC: Final = 0.99
MIN_SCRIPT_PURITY: Final = 0.999
MIN_MONG_RETENTION: Final = 0.95
# What a measurement that is paired with the wrong rows, or made by a model that cannot do the
# thing, is allowed to reach.
MAX_ROTATED_NUMBERS: Final = 0.02
MAX_ROTATED_PLACEHOLDERS: Final = 0.02
MAX_ROTATED_ENTITIES: Final = 0.10
MAX_ROTATED_TERMS: Final = 0.02
MAX_ROTATED_MONG_RETENTION: Final = 0.2
MAX_D11_PLACEHOLDERS: Final = 0.02
MAX_D11_NUMBERS: Final = 0.02
MAX_D11_RETENTION: Final = 0.5
MAX_NEVER_WRITTEN_TERMS: Final = 0.02
MAX_UNSEEN_PLACEHOLDERS: Final = 0.2
MIN_ROWS_BELOW_THE_FLOOR_WITH_THE_MASK_OFF: Final = 50
MAX_MEAN_PURITY_WITH_THE_MASK_OFF: Final = 0.97
MAX_MOVE_OF_AN_UNRELATED_SECTION: Final = 0.02
MIN_MONG_OUTPUT_PURITY: Final = 0.99
ROTATION: Final = 7

# The main suite (`tests/support/robust_suites.py`): the rows of each section, the hypotheses a
# run holds for them (the `mong` rows are asked twice, in Cyrillic and in traditional script),
# and what is scored (entities and terms count the mentions of their rows).
SUITE_ROWS: Final = {
    "numbers": 960,
    "placeholders": 600,
    "entities": 240,
    "terms": 400,
    "script": 900,
    "mong": 480,
}
HYPOTHESES: Final = {**SUITE_ROWS, "mong": 960}
SCORED: Final = {**SUITE_ROWS, "entities": 480, "terms": 1044}
SINGLE_NUMBER_CATEGORIES: Final = (
    "int",
    "decimal",
    "grouped",
    "percent",
    "currency",
    "date",
    "time",
)
KEYS: Final = {
    "numbers": "numbers_acc",
    "placeholders": "placeholder_acc",
    "entities": "entity_acc",
    "terms": "term_acc",
    "script": "script_purity",
    "mong": "mong_source_retention",
}
SIX_KEYS: Final = tuple(KEYS.values())
EXPORT_DIRECTIONS: Final = {
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
    "mn_Mong-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
}
OPTIONS_OF_THE_DEFAULT_RUN: Final = {
    "beam_size": 4,
    "converter": False,
    "device": "cpu",
    "max_decoding_length": 256,
    "script_mask": "retry",
    "threads": 1,
}


# ---------------------------------------------------------------------------------
# The fixtures: each runs the real command once per module.


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory the runs of this module write into (one subdirectory each)."""
    return tmp_path_factory.mktemp("d12_runs")


@pytest.fixture(scope="module")
def main_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(d12_suites.main, d12_robust_export, runs / "main")


@pytest.fixture(scope="module")
def rotated_run(d12_suites: Suites, main_run: Run, runs: Path) -> Run:
    rotated = runs / "rotated.jsonl"
    rotate_hypotheses(main_run.out_dir / HYPOTHESES_FILE, rotated, ROTATION)
    return hypotheses_run(d12_suites.main, rotated, runs / "rotated")


@pytest.fixture(scope="module")
def mong_rotated_run(d12_suites: Suites, main_run: Run, runs: Path) -> Run:
    rotated = runs / "mong_rotated.jsonl"
    rotate_hypotheses(
        main_run.out_dir / HYPOTHESES_FILE, rotated, ROTATION, groups={MONG_ARM}
    )
    return hypotheses_run(d12_suites.main, rotated, runs / "mong_rotated")


@pytest.fixture(scope="module")
def d11_run(d11_tiny_export_dir: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(d12_suites.main, d11_tiny_export_dir, runs / "d11_teacher")


@pytest.fixture(scope="module")
def term_control_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(d12_suites.term_control, d12_robust_export, runs / "term_control")


@pytest.fixture(scope="module")
def unseen_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(d12_suites.unseen_placeholders, d12_robust_export, runs / "unseen")


@pytest.fixture(scope="module")
def mask_off_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(
        d12_suites.main,
        d12_robust_export,
        runs / "mask_off",
        "--script-mask",
        "off",
    )


@pytest.fixture(scope="module")
def mask_always_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(
        d12_suites.main,
        d12_robust_export,
        runs / "mask_always",
        "--script-mask",
        "always",
    )


@pytest.fixture(scope="module")
def rescored_run(d12_suites: Suites, main_run: Run, runs: Path) -> Run:
    return hypotheses_run(
        d12_suites.main, main_run.out_dir / HYPOTHESES_FILE, runs / "rescored"
    )


@pytest.fixture(scope="module")
def second_main_run(d12_robust_export: Path, d12_suites: Suites, runs: Path) -> Run:
    return model_run(d12_suites.main, d12_robust_export, runs / "main_again")


def _converter_options(tiny_run: TinyRun, tiny_ct2: Path) -> tuple[str, ...]:
    return (
        "--script-converter-ct2",
        str(tiny_ct2),
        "--script-converter-spm",
        str(tiny_run.spm_model),
    )


@pytest.fixture(scope="module")
def converter_run(
    d12_robust_export: Path,
    d12_suites: Suites,
    tiny_run: TinyRun,
    tiny_ct2: Path,
    runs: Path,
) -> Run:
    return model_run(
        d12_suites.main,
        d12_robust_export,
        runs / "converter",
        *_converter_options(tiny_run, tiny_ct2),
    )


@pytest.fixture(scope="module")
def mong_output_run(
    d12_robust_export: Path,
    d12_suites: Suites,
    tiny_run: TinyRun,
    tiny_ct2: Path,
    runs: Path,
) -> Run:
    return model_run(
        d12_suites.mong_output,
        d12_robust_export,
        runs / "mong_output",
        *_converter_options(tiny_run, tiny_ct2),
    )


def _value(run: Run, section: str) -> float:
    value = run.block[KEYS[section]]
    assert value is not None, section
    return value


def _record_run(record_property: Any, name: str, run: Run) -> None:
    """The values the run measured, to be read from the report of the session."""
    record_property(f"{name}_seconds", round(run.invocation.seconds, 1))
    for section, key in KEYS.items():
        value = run.block[key]
        if value is not None:
            record_property(f"{name}_{key}", round(value, 6))
            scored = run.sections[section]
            record_property(
                f"{name}_{section}_ok_of_scored", f"{scored['ok']}/{scored['scored']}"
            )


# =================================================================================
# The model the gate measures.


def test_the_toy_teacher_trained_on_the_cpu_and_its_export_is_float32(
    d12_robust_teacher: Teacher, d12_robust_export: Path, record_property: Any
) -> None:
    teacher = d12_robust_teacher
    record_property("teacher_launch_seconds", round(teacher.launch.seconds, 1))
    record_property("teacher_prepare_seconds", round(teacher.prepare_seconds, 1))
    record_property("teacher_average_seconds", round(teacher.average.seconds, 1))
    record_property("teacher_steps", ROBUST_TEACHER_STEPS)
    record_property("export_model_sha256", sha256_of(d12_robust_export / "model.bin"))
    record_property(
        "export_model_bytes", (d12_robust_export / "model.bin").stat().st_size
    )
    record = json.loads((d12_robust_export / "rinchen.json").read_text("utf-8"))

    assert teacher.saved_steps[-1] == ROBUST_TEACHER_STEPS
    assert teacher.curve.non_finite_step is None
    assert "GPU ranks: []" in teacher.log
    assert "Starting training in cpu mode" in teacher.log
    assert record["quantization"] == "float32"
    assert record["tag_placement"] == "src"
    assert set(record["directions"]) == EXPORT_DIRECTIONS
    assert (d12_robust_export / MANIFEST_FILE).is_file()


# =================================================================================
# R1: the main suite, with the defaults.


def test_the_command_scores_the_main_suite_and_prints_a_line_per_section(
    main_run: Run, d12_robust_export: Path, record_property: Any
) -> None:
    _record_run(record_property, "main", main_run)
    out = main_run.out_dir
    lines = main_run.invocation.stdout
    record = json.loads((d12_robust_export / "rinchen.json").read_text("utf-8"))

    assert lines[:2] == ["suite: robust-main", f"system: {record['name']}"]
    assert [line.split(":")[0] for line in lines[2:8]] == list(SUITE_ROWS)
    assert lines[8:] == [
        f"result: {out / RESULT_FILE}",
        f"report: {out / REPORT_FILE}",
        f"manifest: {out / MANIFEST_FILE}",
    ]
    assert sorted(path.name for path in out.iterdir()) == sorted(
        (HYPOTHESES_FILE, MANIFEST_FILE, REPORT_FILE, RESULT_FILE)
    )
    assert main_run.result["system"]["options"] == OPTIONS_OF_THE_DEFAULT_RUN


def test_the_report_and_the_result_agree_on_the_six_values(main_run: Run) -> None:
    block = main_run.block

    assert set(block) == set(SIX_KEYS)
    assert block == main_run.result["robustness"]
    for section, key in KEYS.items():
        assert block[key] == main_run.sections[section]["value"]
    assert main_run.report["schema"] == "rinchen.report.robustness/1"
    assert main_run.report["robustness_sha256"] == sha256_of(
        main_run.out_dir / RESULT_FILE
    )
    assert main_run.report["suite"] == {
        "name": "robust-main",
        "sha256": main_run.result["suite"]["sha256"],
    }


def test_the_six_values_clear_their_floors_for_a_model_trained_on_the_toy(
    main_run: Run,
) -> None:
    assert _value(main_run, "numbers") >= MIN_NUMBERS_ACC
    assert _value(main_run, "placeholders") >= MIN_PLACEHOLDER_ACC
    assert _value(main_run, "entities") >= MIN_ENTITY_ACC
    assert _value(main_run, "terms") >= MIN_TERM_ACC
    assert _value(main_run, "script") >= MIN_SCRIPT_PURITY
    assert _value(main_run, "mong") >= MIN_MONG_RETENTION


def test_each_single_number_category_clears_its_floor(
    main_run: Run, record_property: Any
) -> None:
    categories = main_run.sections["numbers"]["by_category"]
    for name, counts in categories.items():
        record_property(f"numbers_{name}", f"{counts['ok']}/{counts['scored']}")

    assert set(categories) == {*SINGLE_NUMBER_CATEGORIES, "two_numbers"}
    for name in SINGLE_NUMBER_CATEGORIES:
        counts = categories[name]
        assert counts["scored"] > 0, name
        assert counts["ok"] / counts["scored"] >= MIN_NUMBER_CATEGORY_ACC, name


def test_every_request_is_answered_ok_and_every_row_is_scored(main_run: Run) -> None:
    for section, rows in SUITE_ROWS.items():
        facts = main_run.sections[section]
        assert facts["rows"] == rows, section
        assert facts["statuses"] == {"ok": HYPOTHESES[section]}, section
        assert facts["scored"] == SCORED[section], section
        assert facts["excluded"] == {}, section
        assert len(main_run.result["rows"][section]) == rows, section
    assert main_run.result["suite"]["rows"] == SUITE_ROWS
    assert sum(
        1
        for line in (main_run.out_dir / HYPOTHESES_FILE).read_text("utf-8").splitlines()
    ) == sum(HYPOTHESES.values())


def test_the_manifest_vouches_for_the_suite_the_export_and_the_files_written(
    main_run: Run, d12_robust_export: Path, d12_suites: Suites
) -> None:
    manifest = manifest_of(main_run.out_dir)
    inputs = {Path(item.path): item.sha256 for item in manifest.inputs}
    outputs = {Path(item.path).name: item for item in manifest.outputs}

    assert manifest.stage == "eval_robustness"
    assert manifest.status["robustness"] == "ok"
    assert manifest.status["mode"] == "model"
    assert manifest.status["quantization"] == "float32"
    assert inputs[d12_suites.main] == sha256_of(d12_suites.main)
    export_files = sorted(path.name for path in d12_robust_export.iterdir())
    assert sorted(path.name for path in inputs if path.parent == d12_robust_export) == (
        export_files
    )
    for path, digest in inputs.items():
        assert digest == sha256_of(path), path
    assert set(outputs) == {HYPOTHESES_FILE, REPORT_FILE, RESULT_FILE}
    for name, item in outputs.items():
        assert item.sha256 == sha256_of(main_run.out_dir / name), name
    assert outputs[HYPOTHESES_FILE].rows == sum(HYPOTHESES.values())
    assert {name: manifest.metrics[f"rows.{name}"] for name in SUITE_ROWS} == {
        name: float(count) for name, count in SUITE_ROWS.items()
    }
    for key in SIX_KEYS:
        assert manifest.metrics[key] == main_run.block[key], key
    assert {"ctranslate2", "sentencepiece", "sacrebleu", "regex"} <= set(
        manifest.tool_versions
    )


# =================================================================================
# R2: the same hypotheses paired with the wrong rows score near zero.


def test_the_rotation_moves_the_texts_and_leaves_the_keys_alone(
    main_run: Run, rotated_run: Run, mong_rotated_run: Run, runs: Path
) -> None:
    original = _read_lines(main_run.out_dir / HYPOTHESES_FILE)
    rotated = _read_lines(runs / "rotated.jsonl")
    mong_only = _read_lines(runs / "mong_rotated.jsonl")

    assert len(rotated) == len(original) == len(mong_only)
    moved = 0
    for before, after in zip(original, rotated, strict=True):
        assert {k: v for k, v in before.items() if k != "text"} == {
            k: v for k, v in after.items() if k != "text"
        }
        moved += before["text"] != after["text"]
    assert moved >= 0.9 * len(original)
    only_mong = {
        (line["section"], line["arm"])
        for before, line in zip(original, mong_only, strict=True)
        if before["text"] != line["text"]
    }
    assert only_mong == {MONG_ARM}


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_hypotheses_paired_with_another_rows_reference_score_near_zero(
    rotated_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "rotated", rotated_run)

    assert rotated_run.result["system"]["model_sha256"] is None
    assert _value(rotated_run, "numbers") <= MAX_ROTATED_NUMBERS
    assert _value(rotated_run, "placeholders") <= MAX_ROTATED_PLACEHOLDERS
    assert _value(rotated_run, "entities") <= MAX_ROTATED_ENTITIES
    assert _value(rotated_run, "terms") <= MAX_ROTATED_TERMS


def test_a_traditional_script_arm_paired_with_the_wrong_rows_retains_little(
    mong_rotated_run: Run, record_property: Any
) -> None:
    record_property(
        "mong_rotated_retention", round(_value(mong_rotated_run, "mong"), 6)
    )

    assert _value(mong_rotated_run, "mong") <= MAX_ROTATED_MONG_RETENTION


# =================================================================================
# R3: a model that never saw a digit or a placeholder.


def test_the_d11_teacher_that_never_saw_a_digit_or_a_placeholder_scores_near_zero(
    d11_run: Run, main_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "d11_teacher", d11_run)

    assert _value(d11_run, "placeholders") <= MAX_D11_PLACEHOLDERS
    assert _value(d11_run, "numbers") <= MAX_D11_NUMBERS
    assert _value(d11_run, "mong") <= MAX_D11_RETENTION
    # The same suite, the same command: only the model differs.
    assert d11_run.result["suite"] == main_run.result["suite"]
    assert (
        d11_run.result["system"]["model_sha256"]
        != main_run.result["system"]["model_sha256"]
    )


# =================================================================================
# R4: a glossary the model never writes.


def test_a_glossary_the_model_never_writes_scores_near_zero(
    term_control_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "term_control", term_control_run)
    terms = term_control_run.sections["terms"]

    assert list(term_control_run.sections) == ["terms"]
    assert terms["scored"] == SCORED["terms"]
    assert terms["excluded"] == {}
    assert terms["value"] is not None
    assert terms["value"] <= MAX_NEVER_WRITTEN_TERMS


# =================================================================================
# R5: placeholder indices the toy never trained.


def test_placeholder_indices_that_were_never_trained_are_not_copied(
    unseen_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "unseen_placeholders", unseen_run)
    placeholders = unseen_run.sections["placeholders"]
    for direction, counts in placeholders["by_direction"].items():
        record_property(f"unseen_{direction}", f"{counts['ok']}/{counts['scored']}")

    assert placeholders["scored"] == 28
    assert placeholders["excluded"] == {}
    assert set(placeholders["by_direction"]) == {
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
    }
    assert placeholders["value"] <= MAX_UNSEEN_PLACEHOLDERS


# =================================================================================
# R6: the script mask.


def test_with_the_mask_off_the_untrained_pairs_leave_rows_below_the_purity_floor(
    mask_off_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "mask_off", mask_off_run)
    script = mask_off_run.sections["script"]
    record_property("mask_off_below_floor", script["details"]["below_floor"])
    record_property("mask_off_mean_purity", round(script["details"]["mean"], 6))
    record_property("mask_off_min_purity", round(script["details"]["min"], 6))

    assert (
        script["details"]["below_floor"] >= MIN_ROWS_BELOW_THE_FLOOR_WITH_THE_MASK_OFF
    )
    assert script["value"] < MAX_MEAN_PURITY_WITH_THE_MASK_OFF
    assert mask_off_run.result["system"]["options"]["script_mask"] == "off"


@pytest.mark.parametrize("mask", ["retry", "always"])
def test_with_the_mask_retry_or_always_no_row_is_below_the_floor(
    mask: str,
    main_run: Run,
    mask_always_run: Run,
    record_property: Any,
) -> None:
    run = main_run if mask == "retry" else mask_always_run
    script = run.sections["script"]
    record_property(f"mask_{mask}_mean_purity", round(script["details"]["mean"], 6))

    assert run.result["system"]["options"]["script_mask"] == mask
    assert script["details"]["below_floor"] == 0
    assert script["details"]["mean"] >= MIN_SCRIPT_PURITY
    assert script["value"] >= MIN_SCRIPT_PURITY


@pytest.mark.parametrize("mask", ["off", "always"])
def test_the_mask_is_not_the_business_of_the_other_sections(
    mask: str, main_run: Run, mask_off_run: Run, mask_always_run: Run
) -> None:
    run = mask_off_run if mask == "off" else mask_always_run
    for section, key in KEYS.items():
        if section == "script":
            continue
        moved = abs(_value(run, section) - _value(main_run, section))
        assert moved <= MAX_MOVE_OF_AN_UNRELATED_SECTION, key


# =================================================================================
# R7: rescoring and determinism.


def test_rescoring_the_hypotheses_a_run_wrote_gives_the_same_measurement(
    main_run: Run, rescored_run: Run
) -> None:
    for part in ("robustness", "sections", "rows", "suite"):
        assert rescored_run.result[part] == main_run.result[part], part
    assert rescored_run.report["robustness"] == main_run.report["robustness"]
    system = rescored_run.result["system"]
    assert system["model_sha256"] is None
    assert system["manifest_sha256"] is None
    assert system["options"] is None
    assert main_run.result["system"]["model_sha256"] is not None
    # A rescoring has no model, so it writes no hypotheses of its own.
    assert not (rescored_run.out_dir / HYPOTHESES_FILE).exists()


def test_the_command_run_twice_with_the_model_writes_the_same_bytes(
    main_run: Run, second_main_run: Run
) -> None:
    for name in (RESULT_FILE, REPORT_FILE, HYPOTHESES_FILE):
        assert (second_main_run.out_dir / name).read_bytes() == (
            main_run.out_dir / name
        ).read_bytes(), name


# =================================================================================
# R8: the D5 script converter.


def test_the_converter_changes_only_the_converted_arm_of_the_traditional_sources(
    converter_run: Run, main_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "converter", converter_run)
    mong = converter_run.sections["mong"]
    direct = main_run.sections["mong"]["details"]
    record_property(
        "converted_retention", round(mong["details"]["retention_converted"], 6)
    )
    record_property("converted_chrf", round(mong["details"]["chrf_converted"], 4))
    record_property("direct_chrf", round(mong["details"]["chrf_mong"], 4))
    record_property("cyrillic_chrf", round(mong["details"]["chrf_cyrl"], 4))

    for section in ("numbers", "placeholders", "entities", "terms", "script"):
        assert converter_run.sections[section] == main_run.sections[section], section
        assert converter_run.result["rows"][section] == main_run.result["rows"][section]
    assert converter_run.block == main_run.block
    assert converter_run.result["system"]["options"]["converter"] is True
    assert mong["statuses"] == {"ok": 3 * SUITE_ROWS["mong"]}
    assert direct["retention_converted"] is None
    assert direct["chrf_converted"] is None
    assert mong["details"]["retention_converted"] is not None
    assert mong["details"]["retention_converted"] < mong["value"]
    assert mong["details"]["chrf_converted"] < mong["details"]["chrf_mong"]


def test_the_traditional_script_the_converter_writes_is_pure_and_every_request_is_ok(
    mong_output_run: Run, record_property: Any
) -> None:
    _record_run(record_property, "mong_output", mong_output_run)
    script = mong_output_run.sections["script"]

    assert list(mong_output_run.sections) == ["script"]
    assert script["rows"] == script["scored"] == 100
    assert script["statuses"] == {"ok": 100}
    assert script["excluded"] == {}
    assert script["details"]["below_floor"] == 0
    assert _value(mong_output_run, "script") >= MIN_MONG_OUTPUT_PURITY


def test_without_a_converter_the_traditional_script_cannot_be_written(
    d12_robust_export: Path, d12_suites: Suites, runs: Path
) -> None:
    out_dir = runs / "mong_output_without_a_converter"
    invocation = run_robustness(
        "--suite",
        d12_suites.mong_output,
        "--model",
        d12_robust_export,
        "--out-dir",
        out_dir,
    )

    assert invocation.returncode == 2, invocation.stderr
    assert invocation.stdout == []
    assert "script converter" in invocation.stderr
    assert not out_dir.exists()


# =================================================================================
# R10 (the CPU part): the training and the engine ran on the CPU.


def test_the_training_and_the_engine_ran_on_the_cpu(
    d12_robust_teacher: Teacher, main_run: Run
) -> None:
    log = d12_robust_teacher.log
    assert "nccl" not in log.lower()
    assert "cuda:" not in log
    assert main_run.result["system"]["options"]["device"] == "cpu"
    assert main_run.invocation.stderr.lower().count("cuda") == 0


# =================================================================================
# R11: the exclusions, with hand-made expectations.


def test_the_golden_suite_scores_to_the_golden_bytes_and_no_excluded_row_passes(
    runs: Path,
) -> None:
    run = hypotheses_run(
        GOLDEN_SUITE, GOLDEN_HYPOTHESES, runs / "golden", "--system", "golden"
    )
    sections = run.sections

    assert (run.out_dir / RESULT_FILE).read_bytes() == GOLDEN_RESULT.read_bytes()
    assert sections["numbers"]["excluded"] == {
        "no_numbers": 1,
        "source_reference_disagree": 1,
    }
    assert (sections["numbers"]["ok"], sections["numbers"]["scored"]) == (3, 6)
    assert sections["placeholders"]["excluded"] == {"no_placeholders": 1}
    assert sections["terms"]["excluded"] == {"absent_in_source": 1}
    assert sections["script"]["excluded"] == {"no_letters": 1}
    assert run.block == {
        "entity_acc": 0.5714285714285714,
        "mong_source_retention": 0.5017570743156547,
        "numbers_acc": 0.5,
        "placeholder_acc": 0.4,
        "script_purity": 0.75,
        "term_acc": 0.75,
    }
    assert hashlib.sha256(GOLDEN_RESULT.read_bytes()).hexdigest() == (
        "995e81766fd139349abc2ed19ef1953b3036cedcbf28b4f0440e0ce088ee76d0"
    )
