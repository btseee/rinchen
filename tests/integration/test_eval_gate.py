"""The D13 STOP gate, evaluation half (R1 to R5): `rinchen eval suite|run` through the real binary over
the toy of D12, the real Eole-trained model and its CTranslate2 export, the real SentencePiece and the
real sacrebleu, on the CPU. Nothing here touches a GPU (`CUDA_VISIBLE_DEVICES` is the empty string for
the session and every command refuses to start otherwise; the device is `cpu`, `auto` is never used),
nothing opens a socket (`HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` are on, no token is in the
environment, and the Hub library and `datasets` are BLOCKED for every command: a package of each name
that raises `ImportError` comes first on the command's path, so no client can be made) and nothing
downloads anything. The paired test, the report, the guards and the robustness block (R6 to R8) are in
`test_eval_gate_controls.py`; the staging, the Hub plan and the hygiene scans (R9 to R11) in
`test_release_gate.py`.

The data are the held-out rows of the robustness toy (`tests/support/robust_toy.py`), joined into the
fixture suites `toy-devtest` (480 rows) and `toy-dev` (292 rows) of `tests/support/eval_toy.py`: one
sentence in each of `eng_Latn`, `khk_Cyrl` and `rus_Cyrl`, the six directions of the three languages.
The systems are the toy teacher of D12 (the session fixture, 6,000 steps, float32 export), the D11
teacher that never saw those sentences (`d11_tiny_export_dir`) and a hypotheses file. The model ranks
nothing (a tiny model on templated text is no evidence about a real model, a real corpus or a GPU);
every figure below measures the machinery.

Gates, every floor a literal of `tests/support/eval_gate.py` and never lowered (a miss of a model floor
is answered with more steps of the toy, `ROBUST_TEACHER_STEPS`, which is deterministic here):

    R1  `toy_suite`: 480 `devtest` rows and 292 `dev` rows, three languages, six directions, no
        document; the suite files are what the writer wrote and `rinchen eval suite inspect` prints them
    R2  `eval run --model <D12 export>`: exit 0; four directions `run` (480 segments, every status `ok`),
        `eng_Latn-rus_Cyrl` and `rus_Cyrl-eng_Latn` `not_trained` with no result file; chrF++ at least
        95.0, BLEU at least 93.0, each interval containing its value and narrower than 5.0, the segment
        unit, 480 units, seed 20261110, the neural metrics null with their reasons
    R3  the D11 teacher as `d11-teacher`: chrF++ at most 60.0 in every trained direction
    R4  R2's `hypotheses.jsonl` through `--hypotheses` as `a-file`: every metric, interval, status and
        signature equal to R2's, the system kind `hypotheses`, the two other directions `not_in_file`
    R5  determinism: R2 run again, and under `PYTHONHASHSEED=1` with `LC_ALL=C` and under
        `PYTHONHASHSEED=2`, gives byte-equal `result.json` of every direction, `hypotheses.jsonl` and
        `run.json`; a command started in a process that does not hide the devices or the Hub is
        refused before it starts

Measured on this machine (CPU, one thread per command, Eole 0.6.4, CTranslate2 4.8.4, sacrebleu 2.6.0,
2026-11-11; the machine was shared, so a time moves by a third between runs, a count or a score never does:
the toy's training is bit-for-bit reproducible):

    setup  the session fixtures (the D12 toy teacher, 6,000 steps, averaged and exported to float32; the two D11
           teachers and their exports) and the three shared runs: 358 s once per session, 357.6 s of it
           before the first test; the export's `model.bin` is 2,080,071 bytes, SHA-256 `e28d1566...a6`, the
           one D12's gate and the design spike measured (the training is bit-for-bit reproducible); the
           suite files are 113,679 bytes (`toy-devtest`) and 69,392 bytes (`toy-dev`)
    R1     480 and 292 rows, 6 directions, `inspect` prints name, kind, split, purpose, digest, rows per
           language and the directions
    R2     the D12 export, beam 4, script mask `retry`, float32, about 1.6 s for a whole run (the model load
           and the 1,920 requests of the four trained directions included), every status `ok`; chrF++ with
           its 95 % interval (1,000 resamples, the segment unit, 480 units, seed 20261110), then chrF and BLEU
           with theirs:
               eng_Latn-khk_Cyrl  97.03 [95.93, 98.04]  chrF 96.99 [95.78, 98.08]  BLEU 95.84 [94.60, 97.08]
               khk_Cyrl-eng_Latn  99.05 [98.61, 99.46]  chrF 99.14 [98.73, 99.51]  BLEU 98.12 [97.20, 98.93]
               khk_Cyrl-rus_Cyrl  98.97 [98.37, 99.49]  chrF 99.02 [98.43, 99.54]  BLEU 98.15 [97.13, 99.04]
               rus_Cyrl-khk_Cyrl  98.38 [97.66, 99.05]  chrF 98.53 [97.83, 99.20]  BLEU 96.83 [95.47, 98.10]
           the widest interval is 2.63 (BLEU, `rus_Cyrl-khk_Cyrl`) against the limit of 5.0; `run.json` is
           2,344 bytes, a `result.json` about 2,230 bytes, `hypotheses.jsonl` 210,614 bytes (1,920 lines)
    R3     the D11 teacher: chrF++ 36.50 [33.63, 39.16], 46.09 [42.89, 48.93], 29.30 [25.52, 32.67] and
           31.67 [28.66, 34.52] in the same four directions (BLEU 29.53, 33.11, 26.05, 27.55), 5.5 to 7.1
           wide: the limit of 60.0 is 14 points above the highest, 46.09
    R4     the outputs of R2 as `a-file`: every metric, interval, status and signature equal to R2's (the
           comparison is `==` on the parsed documents), `not_in_file` for the two other directions
    R5     the three reruns (plain, `PYTHONHASHSEED=1` with `LC_ALL=C`, `PYTHONHASHSEED=2`) wrote the same
           bytes as R2 for `hypotheses.jsonl`, `run.json` and the four `result.json`
    time   this module's 30 tests took 212.7 s in the full gate (four parts in parallel on the shared machine),
           190 s of it the toy teacher's training (the D11 teachers had been trained earlier in that
           session); the three modules of the gate alone take 423.9 s, 357.6 s of it the session fixtures

What the toy cannot say (UNMEASURED): anything about a real model, FLORES+, NTREX or a human set, a real
baseline (NLLB, MADLAD, Hy-MT2), COMET or MetricX (no scorer exists: only the seam), the real Hub, GitHub
Actions and anything on a GPU (CTranslate2 on CUDA, float16, int8_float16). The toy ranks nothing, and
nothing here says that any system is better than another except as a check that the instrument sees a
difference the construction planted.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.distill_gate import sha256_of
from tests.support.eval_gate import (
    COMET_AND_METRICX,
    DEVTEST_SUITE,
    HUB_BLOCKED_VARIABLE,
    HYPOTHESES_FILE,
    MAX_CI_WIDTH,
    MAX_TEACHER_CHRF_PP,
    MIN_BLEU,
    MIN_CHRF_PP,
    NOT_TRAINED_DIRECTIONS,
    PROTECTED_VARIABLES,
    TRAINED_DIRECTIONS,
    Runs,
    block_hub_libraries,
    changed_environment,
    environment_problems,
    read_result,
    read_run,
    run_eval,
)
from tests.support.eval_toy import (
    DEV_ROWS,
    DEVTEST_ROWS,
    LANGUAGES,
    SIX_DIRECTIONS,
    Suites,
    toy_suite,
)
from tests.support.robust_toy import RobustToy, build_robust_toy

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden", "d13_offline"),
]

BOOTSTRAP: Final = {
    "alpha": 0.05,
    "interval": "percentile",
    "resamples": 1000,
    "rng": "splitmix64-counter/1",
    "seed": 20261110,
    "unit": "segment",
    "units": DEVTEST_ROWS,
}
COUNT_METRICS: Final = ("bleu", "chrf", "chrf_pp")
# What a result says about the system, not about the scores: it differs between a model and a file.
_SAME_IN_A_FILE: Final = (
    "metrics",
    "ci95",
    "statuses",
    "bootstrap",
    "n_segments",
    "bleu_signature",
    "chrf_signature",
    "chrf_pp_signature",
    "unmeasured",
    "suite",
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory the reruns of this module write into."""
    return tmp_path_factory.mktemp("d13_eval_gate")


@pytest.fixture(scope="module")
def toy() -> RobustToy:
    return build_robust_toy()


# =================================================================================
# R1: the toy's suites.


def test_the_toy_suites_hold_one_row_per_held_out_item_in_three_languages(
    toy: RobustToy,
) -> None:
    for split, rows in (("devtest", DEVTEST_ROWS), ("dev", DEV_ROWS)):
        suite = toy_suite(toy, split)

        assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
            f"toy-{split}",
            "fixture",
            "toy",
            split,
        )
        assert len(suite.rows) == rows
        assert len({row.id for row in suite.rows}) == rows
        assert all(set(row.text) == set(LANGUAGES) for row in suite.rows)
        assert all(row.doc is None for row in suite.rows)
        assert sorted(suite.directions) == sorted(SIX_DIRECTIONS)
        assert suite.source.dataset is None and suite.source.pinned is False
        assert suite.purpose == ("report" if split == "devtest" else "selection")


def test_the_two_toy_suites_share_no_sentence(toy: RobustToy) -> None:
    held_out = {row.text["khk_Cyrl"] for row in toy_suite(toy, "devtest").rows}
    dev = {row.text["khk_Cyrl"] for row in toy_suite(toy, "dev").rows}

    assert held_out and dev and not held_out & dev


def test_an_item_that_lacks_a_language_is_refused_and_so_is_a_split_that_is_not_a_suite(
    toy: RobustToy,
) -> None:
    first = next(
        row for row in toy.rows if row.split == "test" and row.src_lang == "eng_Latn"
    )
    khalkha_of_first = (
        first.tgt
    )  # the English to Khalkha row's target is its Khalkha text

    def of_the_first_item_in_russian(row: Any) -> bool:
        khalkha = row.src if row.src_lang == "khk_Cyrl" else row.tgt
        return (
            row.split == "test"
            and "rus_Cyrl" in (row.src_lang, row.tgt_lang)
            and khalkha == khalkha_of_first
        )

    without_russian = RobustToy(
        rows=tuple(row for row in toy.rows if not of_the_first_item_in_russian(row))
    )
    assert len(without_russian.rows) == len(toy.rows) - 2

    with pytest.raises(ValueError, match="rus_Cyrl"):
        toy_suite(without_russian, "devtest")
    with pytest.raises(ValueError, match="split"):
        toy_suite(toy, "train")


def test_the_suite_files_are_the_bytes_of_the_writer_and_inspect_prints_them(
    d13_suites: Suites, toy: RobustToy
) -> None:
    for path, split, rows in (
        (d13_suites.devtest, "devtest", DEVTEST_ROWS),
        (d13_suites.dev, "dev", DEV_ROWS),
    ):
        assert path.read_bytes() == toy_suite(toy, split).to_bytes()

        inspected = run_eval("suite", "inspect", "--suite", path)

        assert inspected.returncode == 0, inspected.stderr
        assert inspected.stdout[:7] == [
            f"name: toy-{split}",
            "kind: fixture",
            f"split: {split}",
            f"purpose: {'report' if split == 'devtest' else 'selection'}",
            f"sha256: {sha256_of(path)}",
            f"rows: eng_Latn {rows}, khk_Cyrl {rows}, rus_Cyrl {rows}",
            "directions: 6",
        ]
        assert sorted(line.strip() for line in inspected.stdout[7:]) == sorted(
            SIX_DIRECTIONS
        )


# =================================================================================
# R2: the D12 export over the devtest suite.


def test_the_command_prints_a_line_per_direction_and_a_state_for_the_untrained_two(
    d13_runs: Runs,
) -> None:
    invocation = d13_runs.invocations["a-model"]

    assert invocation.returncode == 0
    assert invocation.stdout[:3] == [
        "suite: toy-devtest",
        "system: a-model",
        "purpose: report",
    ]
    for direction in TRAINED_DIRECTIONS:
        metrics = read_result(d13_runs.model, direction)["metrics"]
        line = invocation.facts[direction]
        assert line.startswith(f"chrF++ {metrics['chrf_pp']:.2f} [")
        assert line.endswith(f"({DEVTEST_ROWS} segments, {DEVTEST_ROWS} ok)")
    for direction in NOT_TRAINED_DIRECTIONS:
        assert invocation.facts[direction] == "n/a (not_trained)"
    assert invocation.facts["run"] == str(d13_runs.model / "run.json")


def test_four_directions_ran_two_were_never_trained_and_have_no_result_file(
    d13_runs: Runs,
) -> None:
    run = read_run(d13_runs.model)
    states = {entry["direction"]: entry for entry in run["directions"]}

    assert set(states) == set(SIX_DIRECTIONS)
    for direction in TRAINED_DIRECTIONS:
        assert states[direction] == {
            "direction": direction,
            "reason": None,
            "result": f"{direction}/result.json",
            "rows": DEVTEST_ROWS,
            "state": "run",
        }
        assert (d13_runs.model / direction / "result.json").is_file()
    for direction in NOT_TRAINED_DIRECTIONS:
        entry = states[direction]
        assert entry["state"] == "not_trained"
        assert isinstance(entry["reason"], str) and entry["reason"]
        assert entry["result"] is None and entry["rows"] == DEVTEST_ROWS
        assert not (d13_runs.model / direction).exists()
    assert sorted(path.name for path in d13_runs.model.iterdir()) == sorted(
        [*TRAINED_DIRECTIONS, HYPOTHESES_FILE, "manifest.json", "run.json"]
    )


def test_the_run_record_names_the_suite_the_system_and_every_file_with_its_digest(
    d13_runs: Runs, d13_suites: Suites
) -> None:
    run = read_run(d13_runs.model)

    assert run["schema"] == "rinchen.eval.run/1"
    assert run["purpose"] == "report"
    assert run["suite"] == {
        "name": "toy-devtest",
        "sha256": sha256_of(d13_suites.devtest),
    }
    assert (run["system"]["kind"], run["system"]["name"]) == ("rinchen", "a-model")
    assert run["settings"] == {
        "alpha": 0.05,
        "resamples": 1000,
        "seed": 20261110,
        "selection": None,
    }
    assert run["robustness_source"] is None and run["segment_scores"] == {}
    assert set(run["files"]) == {
        HYPOTHESES_FILE,
        *(f"{direction}/result.json" for direction in TRAINED_DIRECTIONS),
    }
    for name, digest in run["files"].items():
        assert digest == sha256_of(d13_runs.model / name), name


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_export_clears_the_floors_in_a_trained_direction(
    d13_runs: Runs, direction: str, record_property: Any
) -> None:
    result = read_result(d13_runs.model, direction)
    metrics, intervals = result["metrics"], result["ci95"]
    for name in COUNT_METRICS:
        record_property(f"export_{direction}_{name}", round(metrics[name], 2))
        record_property(
            f"export_{direction}_{name}_ci95",
            [round(end, 2) for end in intervals[name]],
        )

    assert result["trained"] is True
    assert result["n_segments"] == DEVTEST_ROWS
    assert result["statuses"] == {"ok": DEVTEST_ROWS}
    assert metrics["chrf_pp"] >= MIN_CHRF_PP
    assert metrics["bleu"] >= MIN_BLEU
    for name in COUNT_METRICS:
        low, high = intervals[name]
        assert low <= metrics[name] <= high, name
        assert high - low < MAX_CI_WIDTH, name
    assert result["bootstrap"] == BOOTSTRAP
    assert result["suite"] == "toy-devtest" and result["system"] == "a-model"
    assert result["robustness"] is None
    assert result["robustness_coverage"] is None and result["robustness_source"] is None


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_neural_metrics_are_null_with_their_reasons_and_never_zero(
    d13_runs: Runs, direction: str
) -> None:
    result = read_result(d13_runs.model, direction)

    for name in COMET_AND_METRICX:
        assert result["metrics"][name] is None, name
        assert result["ci95"][name] is None, name
        reason = result["unmeasured"][name]
        assert isinstance(reason, str) and reason, name
    assert set(result["unmeasured"]) == set(COMET_AND_METRICX)


def test_the_signatures_name_the_tokenizer_and_the_version_the_scores_were_made_with(
    d13_runs: Runs,
) -> None:
    result = read_result(d13_runs.model, "eng_Latn-khk_Cyrl")

    assert result["bleu_signature"].split("|")[:4] == [
        "nrefs:1",
        "case:mixed",
        "eff:no",
        "tok:13a",
    ]
    assert result["chrf_signature"].endswith("nw:0|space:no|version:2.6.0")
    assert result["chrf_pp_signature"].endswith("nw:2|space:no|version:2.6.0")


# =================================================================================
# R3: the D11 teacher, which never saw the sentences, is far below.


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_d11_teacher_is_far_below_the_floor_of_the_export(
    d13_runs: Runs, direction: str, record_property: Any
) -> None:
    result = read_result(d13_runs.teacher, direction)
    record_property(
        f"teacher_{direction}_chrf_pp", round(result["metrics"]["chrf_pp"], 2)
    )
    record_property(f"teacher_{direction}_bleu", round(result["metrics"]["bleu"], 2))

    assert result["trained"] is True
    assert result["statuses"] == {"ok": DEVTEST_ROWS}
    assert result["metrics"]["chrf_pp"] <= MAX_TEACHER_CHRF_PP
    assert result["system"] == "d11-teacher"


def test_the_d11_teacher_has_the_same_two_directions_it_never_learned(
    d13_runs: Runs,
) -> None:
    run = read_run(d13_runs.teacher)
    states = {entry["direction"]: entry["state"] for entry in run["directions"]}

    assert {d: states[d] for d in TRAINED_DIRECTIONS} == dict.fromkeys(
        TRAINED_DIRECTIONS, "run"
    )
    assert {d: states[d] for d in NOT_TRAINED_DIRECTIONS} == dict.fromkeys(
        NOT_TRAINED_DIRECTIONS, "not_trained"
    )
    for direction in NOT_TRAINED_DIRECTIONS:
        assert not (d13_runs.teacher / direction).exists()


# =================================================================================
# R4: the outputs of R2 fed back as a file say the same.


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_outputs_of_the_export_fed_back_as_a_file_give_every_number_again(
    d13_runs: Runs, direction: str
) -> None:
    from_model = read_result(d13_runs.model, direction)
    from_file = read_result(d13_runs.file, direction)

    for key in _SAME_IN_A_FILE:
        assert from_file[key] == from_model[key], key
    assert from_file["system"] == "a-file"
    assert from_file["provenance"]["system"]["kind"] == "hypotheses"
    assert (
        from_file["provenance"]["outputs_sha256"]
        == from_model["provenance"]["outputs_sha256"]
        == sha256_of(d13_runs.model / HYPOTHESES_FILE)
    )


def test_a_file_system_is_a_hypotheses_system_and_lists_what_it_does_not_hold(
    d13_runs: Runs,
) -> None:
    run = read_run(d13_runs.file)
    states = {entry["direction"]: entry["state"] for entry in run["directions"]}

    assert run["system"]["kind"] == "hypotheses"
    assert run["system"]["model_sha256"] is None
    assert {d: states[d] for d in NOT_TRAINED_DIRECTIONS} == dict.fromkeys(
        NOT_TRAINED_DIRECTIONS, "not_in_file"
    )
    assert (d13_runs.file / HYPOTHESES_FILE).read_bytes() == (
        d13_runs.model / HYPOTHESES_FILE
    ).read_bytes()


# =================================================================================
# R5: the same bytes every time.


@pytest.fixture(scope="module")
def rerun_roots(
    d13_suites: Suites, d12_robust_export: Path, scratch: Path
) -> list[Path]:
    """R2 again, three times: plain, under hash seed 1 with the C locale, under hash seed 2."""
    roots: list[Path] = []
    for name, environ in (
        ("again", {}),
        ("hashseed1_c_locale", {"PYTHONHASHSEED": "1", "LC_ALL": "C"}),
        ("hashseed2", {"PYTHONHASHSEED": "2"}),
    ):
        root = scratch / name
        invocation = run_eval(
            "run",
            "--suite",
            d13_suites.devtest,
            "--model",
            d12_robust_export,
            "--system",
            "a-model",
            "--out-dir",
            root,
            environ=environ,
        )
        assert invocation.returncode == 0, invocation.stderr
        roots.append(root / DEVTEST_SUITE / "a-model")
    return roots


def test_the_run_gives_the_same_bytes_again_and_under_other_hash_seeds_and_locales(
    d13_runs: Runs, rerun_roots: list[Path]
) -> None:
    names = [
        HYPOTHESES_FILE,
        "run.json",
        *(f"{direction}/result.json" for direction in TRAINED_DIRECTIONS),
    ]

    assert len(rerun_roots) == 3
    for root in rerun_roots:
        assert sorted(path.name for path in root.iterdir()) == sorted(
            path.name for path in d13_runs.model.iterdir()
        )
        for name in names:
            assert (root / name).read_bytes() == (d13_runs.model / name).read_bytes(), (
                root.parent.parent.name,
                name,
            )


def test_a_variable_that_protects_the_gate_cannot_be_changed_by_a_test() -> None:
    assert {
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_TOKEN",
    } <= set(PROTECTED_VARIABLES)
    for name in PROTECTED_VARIABLES:
        with (
            pytest.raises(AssertionError, match=name),
            changed_environment({name: "x"}),
        ):
            pass


def test_a_command_in_a_process_that_does_not_hide_the_devices_or_the_hub_is_refused(
    d13_suites: Suites, d13_runs: Runs, scratch: Path
) -> None:
    assert environment_problems() == []
    for name, value in (
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("HF_HUB_OFFLINE", "0"),
        ("HF_DATASETS_OFFLINE", "0"),
        ("HF_TOKEN", "a-value"),
        (HUB_BLOCKED_VARIABLE, None),
    ):
        out_dir = scratch / f"must_not_start_{name}"
        with pytest.MonkeyPatch.context() as patch:
            if value is None:
                patch.delenv(name)
            else:
                patch.setenv(name, value)
            with pytest.raises(AssertionError, match=name):
                run_eval(
                    "run",
                    "--suite",
                    d13_suites.devtest,
                    "--hypotheses",
                    d13_runs.model / HYPOTHESES_FILE,
                    "--system",
                    "must-not-start",
                    "--out-dir",
                    out_dir,
                )
        assert not out_dir.exists()
    assert environment_problems() == []


def test_the_blocked_hub_libraries_raise_when_they_are_imported(
    tmp_path: Path,
) -> None:
    blocked = block_hub_libraries(tmp_path / "blocked")

    for library in ("huggingface_hub", "datasets"):
        assert (blocked / library / "__init__.py").is_file()
        with pytest.raises(ImportError, match="blocked"):
            runpy.run_path(str(blocked / library / "__init__.py"))
