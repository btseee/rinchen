"""The D13 STOP gate, evaluation half (R6 to R8): the report and the controls that show the paired test
and the guards work, through the real `rinchen eval report|run|suite` binary on the CPU with every device
hidden and no network (the setting and the data are those of `test_eval_gate.py`, whose docstring says
what the toy is and is not).

Gates, every floor a literal of `tests/support/eval_gate.py` and never lowered:

    R6  `eval report` over the export (`a-model`), the D11 teacher (`d11-teacher`), the outputs of the
        export fed back as a file (`a-file`) and `a-perturbed` (the export's outputs with every 20th text
        replaced by the next row's): against `d11-teacher` the export is above it by at least 40.0
        chrF++ in each trained direction, the lower end of the interval above 40.0 and p at most 0.01;
        against `a-model` the identical system `a-file` has delta exactly 0.0, interval [0.0, 0.0] and p
        exactly 1.0 (the control that sacrebleu's own paired test fails), and `a-perturbed` has delta at
        most -2.0 with the upper end of the interval below -0.5 and p at most 0.01 (the sensitivity
        control; the verdict `significant` means "the first system is better", so it is false for the
        worse system and true for the export against `a-perturbed`); `report.json` and `report.md` are
        byte-equal when generated twice (the second time under another hash seed and the C locale),
        hold no absolute path and no timestamp, say fixture in the banner and leave the four targets
        `unmeasured` (the toy suite is never `flores_plus` `devtest`, no run carries a robustness block)
    R7  the guards: a run over the `dev` suite is made (purpose `selection`) and `eval report` refuses it
        with exit 3 naming the suite, alone or among good runs; `eval suite build` refuses a lines
        directory that holds the file of the Inner Mongolian variety with exit 3 and writes nothing, and
        builds the four standard directions from the toy's three languages
    R8  robustness attached: D12's `eval robustness` over its main suite, then `eval run --robustness`:
        the six values and their coverage in every `result.json` equal the robustness run's, the binding
        is `model_digest`, the scores are unchanged; the same directory attached to the D11 teacher's
        run exits 2 with nothing written

Measured on this machine (CPU, one thread per command, Eole 0.6.4, CTranslate2 4.8.4, sacrebleu 2.6.0,
2026-11-11; a time moves by a third between runs, a count or a score never does):

    R6     export (`a-model`) minus `d11-teacher`, chrF++, paired, 1,000 resamples, p = 0.001998 (the
           smallest p, 2/1,001) in every direction:
               eng_Latn-khk_Cyrl  +60.528 [+57.934, +63.433]    khk_Cyrl-eng_Latn  +52.965 [+50.240, +56.143]
               khk_Cyrl-rus_Cyrl  +69.668 [+66.371, +73.529]    rus_Cyrl-khk_Cyrl  +66.707 [+63.981, +69.802]
           (BLEU +66.306, +65.016, +72.091, +69.277, p 0.001998), so the limit of +40.0 stands 12.9 points
           below the lowest lower end, 50.240; the identical system `a-file` against `a-model`: delta 0.0,
           interval [0.0, 0.0], p 1.0, `significant` false, in all four directions and both metrics;
           `a-perturbed` (96 of the 1,920 texts replaced, 24 per direction) against `a-model`:
               eng_Latn-khk_Cyrl  -3.274 [-4.696, -2.064]    khk_Cyrl-eng_Latn  -3.205 [-4.635, -2.049]
               khk_Cyrl-rus_Cyrl  -4.030 [-5.858, -2.511]    rus_Cyrl-khk_Cyrl  -3.402 [-4.889, -2.115]
           p 0.001998, `better_fraction` 0.0, `significant` false (the worse system is not better); the
           export against `a-perturbed` (the baseline of a third report): the same deltas with the sign
           reversed, `significant` true; the reports: `report.json` 57,134 bytes and `report.md` 12,012 for
           the four runs against the teacher, equal when generated again under `PYTHONHASHSEED=3` with
           `LC_ALL=C`; the four targets `unmeasured` (`beats_baseline`: 0 of 16 directions measured)
    R7     a run over `toy-dev` (292 segments) is made, `purpose` `selection`, and the report refuses it with
           exit 3 (alone and among good runs); the lines directory with the file of the Inner Mongolian
           variety exits 3 ("refused by name") and writes nothing; the lines directory of the toy's three
           languages builds `khk_Cyrl-eng_Latn`, `eng_Latn-khk_Cyrl`, `khk_Cyrl-rus_Cyrl` and
           `rus_Cyrl-khk_Cyrl` over 480 rows, six languages of the benchmark's list missing
    R8     D12's robustness run over its main suite, 2 s: numbers_acc 0.9291666666666667 (892 of 960, the
           floor of D12 is 0.90), placeholder_acc 0.9983 (599 of 600), term_acc 1.0 (1,044 of 1,044),
           script_purity 1.0 (900 of 900), entity_acc 0.93125 (447 of 480), mong_source_retention 0.9976
           (480 of 480); the same values and counts are in each of the four `result.json`, binding
           `model_digest`, the scores unchanged; the D11 teacher with that directory exits 2 ("another
           model") with nothing written
    time   this module's 37 tests took 23.9 s in the full gate (the session fixtures were paid for by the
           module before it; every command takes 1 to 2.5 s)

UNMEASURED: every real dataset, baseline and neural metric, the Hub, GitHub Actions and anything on a GPU.
The toy ranks nothing: a delta here shows that the instrument sees a difference that was planted, not that
a system is better.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.support.distill_gate import read_json, sha256_of
from tests.support.eval_gate import (
    DEVTEST_SUITE,
    EVERY_NTH_TEXT,
    HYPOTHESES_FILE,
    MAX_DELTA_OF_PERTURBED,
    MAX_P_OF_A_SEEN_DIFFERENCE,
    MAX_UPPER_END_OF_PERTURBED,
    MIN_DELTA_OVER_TEACHER,
    MIN_LOWER_END_OF_PERTURBED,
    MIN_NUMBERS_ACC,
    TRAINED_DIRECTIONS,
    Report,
    Runs,
    absolute_paths_in,
    comparison,
    make_report,
    perturb_outputs,
    read_result,
    read_run,
    run_eval,
    run_robustness,
)
from tests.support.eval_toy import (
    DEV_ROWS,
    DEVTEST_ROWS,
    INNER_MONGOLIAN_FILE,
    LANGUAGES,
    Suites,
    toy_suite,
    write_lines_directory,
)
from tests.support.robust_gate import Suites as RobustSuites
from tests.support.robust_toy import RobustToy, build_robust_toy

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden", "d13_offline"),
]

STANDARD_DIRECTIONS = {
    "khk_Cyrl-eng_Latn",
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
}
BLOCK_KEYS = {
    "numbers_acc",
    "placeholder_acc",
    "term_acc",
    "script_purity",
    "entity_acc",
    "mong_source_retention",
}


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory the runs, reports and builds of this module write into."""
    return tmp_path_factory.mktemp("d13_eval_controls")


@pytest.fixture(scope="module")
def toy() -> RobustToy:
    return build_robust_toy()


@pytest.fixture(scope="module")
def perturbed_run(d13_suites: Suites, d13_runs: Runs, scratch: Path) -> Path:
    """The export's outputs with every 20th text replaced by the next row's, scored as a file."""
    outputs = scratch / "perturbed.jsonl"
    perturb_outputs(d13_runs.model / HYPOTHESES_FILE, outputs, every=EVERY_NTH_TEXT)
    invocation = run_eval(
        "run",
        "--suite",
        d13_suites.devtest,
        "--hypotheses",
        outputs,
        "--system",
        "a-perturbed",
        "--out-dir",
        d13_runs.root,
    )
    assert invocation.returncode == 0, invocation.stderr
    return d13_runs.run_dir("a-perturbed")


def _report(
    name: str, baseline: str, runs: Runs, suites: Suites, perturbed: Path, scratch: Path
) -> Report:
    return make_report(
        [runs.model, runs.teacher, runs.file, perturbed],
        [suites.devtest],
        scratch / name,
        baseline=baseline,
    )


@pytest.fixture(scope="module")
def teacher_report(
    d13_suites: Suites, d13_runs: Runs, perturbed_run: Path, scratch: Path
) -> Report:
    """Every run against the D11 teacher."""
    return _report(
        "report_teacher", "d11-teacher", d13_runs, d13_suites, perturbed_run, scratch
    )


@pytest.fixture(scope="module")
def model_report(
    d13_suites: Suites, d13_runs: Runs, perturbed_run: Path, scratch: Path
) -> Report:
    """Every run against the export: the identical and the perturbed control live here."""
    return _report(
        "report_model", "a-model", d13_runs, d13_suites, perturbed_run, scratch
    )


@pytest.fixture(scope="module")
def perturbed_baseline_report(
    d13_suites: Suites, d13_runs: Runs, perturbed_run: Path, scratch: Path
) -> Report:
    """Every run against the perturbed system: the export is the candidate that is better."""
    return _report(
        "report_perturbed", "a-perturbed", d13_runs, d13_suites, perturbed_run, scratch
    )


# =================================================================================
# R6: the report, and the controls of the paired test.


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_export_is_far_above_the_d11_teacher_and_the_paired_test_sees_it(
    teacher_report: Report, direction: str, record_property: Any
) -> None:
    for metric in ("chrf_pp", "bleu"):
        paired = comparison(teacher_report, "a-model", direction, metric)
        record_property(
            f"export_vs_teacher_{direction}_{metric}_delta", round(paired["delta"], 3)
        )
        record_property(f"export_vs_teacher_{direction}_{metric}_p", paired["p_value"])

        assert paired["baseline"] == "d11-teacher"
        assert paired["delta"] >= MIN_DELTA_OVER_TEACHER, metric
        assert paired["ci95"][0] > MIN_DELTA_OVER_TEACHER, metric
        assert paired["p_value"] <= MAX_P_OF_A_SEEN_DIFFERENCE, metric
        assert paired["better_fraction"] == 1.0, metric
        assert paired["significant"] is True, metric
        assert (paired["unit"], paired["units"]) == ("segment", DEVTEST_ROWS)


def test_the_comparisons_cover_four_directions_two_metrics_and_three_systems(
    teacher_report: Report,
) -> None:
    comparisons = teacher_report.document["comparisons"]
    keys = {(c["system"], c["direction"], c["metric"]) for c in comparisons}

    assert len(comparisons) == len(keys) == 3 * 4 * 2
    assert {system for system, _, _ in keys} == {"a-model", "a-file", "a-perturbed"}
    assert {direction for _, direction, _ in keys} == set(TRAINED_DIRECTIONS)
    assert (
        f"{len(keys)} paired comparisons"
        in teacher_report.document["note_multiple_comparisons"]
    )


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
@pytest.mark.parametrize("metric", ["chrf_pp", "bleu"])
def test_a_system_compared_with_itself_has_delta_zero_and_p_one(
    model_report: Report, direction: str, metric: str
) -> None:
    paired = comparison(model_report, "a-file", direction, metric)

    assert paired["baseline"] == "a-model"
    assert paired["delta"] == 0.0
    assert paired["ci95"] == [0.0, 0.0]
    assert paired["p_value"] == 1.0
    assert paired["significant"] is False
    assert paired["better_fraction"] == 0.0


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_a_perturbed_system_is_seen_below_the_export(
    model_report: Report, direction: str, record_property: Any
) -> None:
    paired = comparison(model_report, "a-perturbed", direction, "chrf_pp")
    record_property(f"perturbed_vs_export_{direction}_delta", round(paired["delta"], 3))
    record_property(
        f"perturbed_vs_export_{direction}_ci95",
        [round(end, 3) for end in paired["ci95"]],
    )

    assert paired["baseline"] == "a-model"
    assert paired["delta"] <= MAX_DELTA_OF_PERTURBED
    assert paired["ci95"][1] < MAX_UPPER_END_OF_PERTURBED
    assert paired["p_value"] <= MAX_P_OF_A_SEEN_DIFFERENCE
    assert paired["better_fraction"] == 0.0
    # "Significant" says the first system is better: the worse system is not.
    assert paired["significant"] is False


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_export_is_seen_above_the_perturbed_system_and_the_verdict_says_so(
    perturbed_baseline_report: Report, direction: str, record_property: Any
) -> None:
    paired = comparison(perturbed_baseline_report, "a-model", direction, "chrf_pp")
    record_property(f"export_vs_perturbed_{direction}_delta", round(paired["delta"], 3))
    record_property(
        f"export_vs_perturbed_{direction}_ci95",
        [round(end, 3) for end in paired["ci95"]],
    )

    assert paired["baseline"] == "a-perturbed"
    assert paired["delta"] >= -MAX_DELTA_OF_PERTURBED
    assert paired["ci95"][0] > MIN_LOWER_END_OF_PERTURBED
    assert paired["p_value"] <= MAX_P_OF_A_SEEN_DIFFERENCE
    assert paired["better_fraction"] == 1.0
    assert paired["significant"] is True


def test_the_perturbed_system_changes_one_text_in_twenty_and_nothing_else(
    d13_runs: Runs, perturbed_run: Path
) -> None:
    def rows_of(run: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (run / HYPOTHESES_FILE).read_text("utf-8").splitlines()
        ]

    original, perturbed = rows_of(d13_runs.model), rows_of(perturbed_run)
    planted = len(TRAINED_DIRECTIONS) * (DEVTEST_ROWS // EVERY_NTH_TEXT)

    assert len(original) == len(perturbed) == len(TRAINED_DIRECTIONS) * DEVTEST_ROWS
    assert [(r["direction"], r["id"], r["status"]) for r in original] == [
        (r["direction"], r["id"], r["status"]) for r in perturbed
    ]
    changed = sum(
        a["text"] != b["text"] for a, b in zip(original, perturbed, strict=True)
    )
    # A neighbouring row can carry the same text, so a planted change may leave it as it was.
    assert 0.75 * planted <= changed <= planted


@pytest.mark.parametrize("name", ["report.json", "report.md"])
def test_the_report_is_byte_equal_when_generated_again_under_another_hash_seed(
    teacher_report: Report,
    d13_suites: Suites,
    d13_runs: Runs,
    perturbed_run: Path,
    scratch: Path,
    name: str,
) -> None:
    again = make_report(
        [d13_runs.model, d13_runs.teacher, d13_runs.file, perturbed_run],
        [d13_suites.devtest],
        scratch / "report_teacher_again",
        baseline="d11-teacher",
        environ={"PYTHONHASHSEED": "3", "LC_ALL": "C"},
    )

    assert (again.directory / name).read_bytes() == (
        teacher_report.directory / name
    ).read_bytes()


def test_no_document_of_the_runs_or_the_reports_holds_an_absolute_path_or_a_timestamp(
    teacher_report: Report, model_report: Report, d13_runs: Runs, perturbed_run: Path
) -> None:
    files = [
        report.directory / name
        for report in (teacher_report, model_report)
        for name in ("report.json", "report.md")
    ]
    for run in (d13_runs.model, d13_runs.teacher, d13_runs.file, perturbed_run):
        files.append(run / "run.json")
        files.extend(sorted(run.glob("*/result.json")))

    assert len(files) == 4 + 4 * (1 + len(TRAINED_DIRECTIONS))
    for path in files:
        text = path.read_text("utf-8")
        assert absolute_paths_in(text) == [], path.name
        assert re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text) is None, path.name
        assert '"created"' not in text, path.name


def test_the_report_says_fixture_in_its_banner_and_measures_no_target_that_needs_real_data(
    teacher_report: Report,
) -> None:
    document = teacher_report.document
    targets = {target["id"]: target for target in document["targets"]}
    banner = teacher_report.markdown.splitlines()[2]

    assert document["evidence"] == {"fixture_only": True, "kinds": ["fixture"]}
    assert banner.startswith("These figures come from generated fixture suites.")
    assert "say nothing about translation quality" in banner
    assert set(targets) == {"beats_baseline", "off_target", "numbers", "placeholders"}
    for name, target in targets.items():
        assert target["status"] == "unmeasured", name
        assert target["value"] is None, name
    assert targets["beats_baseline"]["basis"] == "0 of 16 directions measured"
    assert {(r["suite"], r["system"]) for r in document["runs"]} == {
        (DEVTEST_SUITE, system)
        for system in ("a-model", "d11-teacher", "a-file", "a-perturbed")
    }
    target_lines = [
        line
        for line in teacher_report.invocation.stdout
        if line.startswith("targets: ")
    ]
    assert len(target_lines) == 4
    assert all(line.endswith(" unmeasured") for line in target_lines)


def test_the_scores_of_the_report_are_the_scores_of_the_runs(
    teacher_report: Report, d13_runs: Runs
) -> None:
    by_key = {
        (result["system"], result["direction"]): result
        for result in teacher_report.document["results"]
    }

    assert len(by_key) == 4 * len(TRAINED_DIRECTIONS)
    for system, run in (("a-model", d13_runs.model), ("d11-teacher", d13_runs.teacher)):
        for direction in TRAINED_DIRECTIONS:
            from_run = read_result(run, direction)
            assert by_key[(system, direction)]["metrics"] == from_run["metrics"]
            assert by_key[(system, direction)]["ci95"] == from_run["ci95"]


# =================================================================================
# R7: the guards.


@pytest.fixture(scope="module")
def dev_run(d13_suites: Suites, d12_robust_export: Path, scratch: Path) -> Path:
    invocation = run_eval(
        "run",
        "--suite",
        d13_suites.dev,
        "--model",
        d12_robust_export,
        "--system",
        "a-model",
        "--out-dir",
        scratch / "dev_runs",
    )
    assert invocation.returncode == 0, invocation.stderr
    assert invocation.stdout[:3] == [
        "suite: toy-dev",
        "system: a-model",
        "purpose: selection",
    ]
    return scratch / "dev_runs" / "toy-dev" / "a-model"


def test_a_run_over_the_dev_suite_is_made_for_selection_and_the_report_refuses_it(
    dev_run: Path, d13_suites: Suites, scratch: Path
) -> None:
    out_dir = scratch / "dev_report"

    assert read_run(dev_run)["purpose"] == "selection"
    assert read_result(dev_run, "eng_Latn-khk_Cyrl")["n_segments"] == DEV_ROWS
    refused = run_eval(
        "report", "--run", dev_run, "--suite", d13_suites.dev, "--out-dir", out_dir
    )

    assert refused.returncode == 3, refused.stderr
    assert refused.stdout == []
    assert "'toy-dev'" in refused.stderr
    assert "dev suite" in refused.stderr
    assert not out_dir.exists()


def test_a_dev_run_among_good_ones_still_stops_the_report(
    dev_run: Path, d13_suites: Suites, d13_runs: Runs, scratch: Path
) -> None:
    out_dir = scratch / "mixed_report"

    refused = run_eval(
        "report",
        "--run",
        d13_runs.model,
        "--run",
        dev_run,
        "--suite",
        d13_suites.devtest,
        "--suite",
        d13_suites.dev,
        "--out-dir",
        out_dir,
    )

    assert refused.returncode == 3, refused.stderr
    assert "'toy-dev'" in refused.stderr
    assert not out_dir.exists()


def _build_lines(lines: Path, out_dir: Path) -> Any:
    return run_eval(
        "suite",
        "build",
        "--benchmark",
        "ntrex",
        "--split",
        "devtest",
        "--from",
        lines,
        "--kind",
        "fixture",
        "--name",
        "toy-lines",
        "--out-dir",
        out_dir,
    )


def test_a_lines_directory_with_the_inner_mongolian_variety_is_refused_and_writes_nothing(
    toy: RobustToy, scratch: Path
) -> None:
    lines = scratch / "lines_with_inner_mongolian"
    write_lines_directory(
        lines, toy_suite(toy, "devtest"), extra_files=(INNER_MONGOLIAN_FILE,)
    )
    out_dir = scratch / "refused_build"

    refused = _build_lines(lines, out_dir)

    assert refused.returncode == 3, refused.stderr
    assert refused.stdout == []
    assert "refused by name" in refused.stderr
    assert not out_dir.exists()


def test_a_lines_directory_of_the_toys_three_languages_builds_the_four_standard_directions(
    toy: RobustToy, scratch: Path
) -> None:
    lines = scratch / "lines"
    write_lines_directory(lines, toy_suite(toy, "devtest"))
    out_dir = scratch / "built"

    built = _build_lines(lines, out_dir)

    assert built.returncode == 0, built.stderr
    suite_file = out_dir / "toy-lines.json"
    assert built.facts["suite"] == str(suite_file)
    assert built.facts["sha256"] == sha256_of(suite_file)
    document = read_json(suite_file)
    assert set(document["directions"]) == STANDARD_DIRECTIONS
    assert len(document["directions"]) == 4
    assert (document["kind"], document["benchmark"], document["name"]) == (
        "fixture",
        "ntrex",
        "toy-lines",
    )
    assert len(document["rows"]) == DEVTEST_ROWS
    assert set(document["source"]["files"]) == {f"{code}.txt" for code in LANGUAGES}
    missing = document["source"]["missing_languages"]
    assert missing == sorted(missing) and missing
    assert not set(missing) & set(LANGUAGES)
    assert "khk_Mong" not in suite_file.read_text("utf-8")
    inspected = run_eval("suite", "inspect", "--suite", suite_file)
    assert inspected.returncode == 0, inspected.stderr
    assert {
        line.strip() for line in inspected.stdout if line.startswith("  ")
    } == STANDARD_DIRECTIONS


# =================================================================================
# R8: D12's robustness block, attached with its coverage.


@pytest.fixture(scope="module")
def robustness_dir(
    d12_robust_export: Path, d12_suites: RobustSuites, scratch: Path
) -> Path:
    """D12's `rinchen eval robustness` over its main suite with the export."""
    out_dir = scratch / "robustness"
    invocation = run_robustness(
        "--suite", d12_suites.main, "--model", d12_robust_export, "--out-dir", out_dir
    )
    assert invocation.returncode == 0, invocation.stderr
    return out_dir


@pytest.fixture(scope="module")
def robust_run(
    d13_suites: Suites, d12_robust_export: Path, robustness_dir: Path, scratch: Path
) -> Path:
    out_dir = scratch / "robust_runs"
    invocation = run_eval(
        "run",
        "--suite",
        d13_suites.devtest,
        "--model",
        d12_robust_export,
        "--system",
        "a-model",
        "--robustness",
        robustness_dir,
        "--out-dir",
        out_dir,
    )
    assert invocation.returncode == 0, invocation.stderr
    return out_dir / DEVTEST_SUITE / "a-model"


@pytest.mark.parametrize("direction", TRAINED_DIRECTIONS)
def test_the_six_values_and_their_coverage_are_in_every_result_equal_to_the_robustness_run(
    robust_run: Path,
    robustness_dir: Path,
    d12_suites: RobustSuites,
    direction: str,
    record_property: Any,
) -> None:
    narrow = read_json(robustness_dir / "report.json")
    result = read_result(robust_run, direction)
    record_property("numbers_acc", result["robustness"]["numbers_acc"])

    assert result["robustness"] == narrow["robustness"]
    assert result["robustness_coverage"] == narrow["coverage"]
    assert set(result["robustness"]) == BLOCK_KEYS
    assert result["robustness"]["numbers_acc"] >= MIN_NUMBERS_ACC
    for key, counts in result["robustness_coverage"].items():
        assert counts["excluded"] == 0, key
        assert counts["scored"] == counts["total"] > 0, key
    assert result["robustness_source"] == {
        "binding": "model_digest",
        "report_sha256": sha256_of(robustness_dir / "report.json"),
        "robustness_sha256": sha256_of(robustness_dir / "robustness.json"),
        "suite": "robust-main",
        "suite_sha256": sha256_of(d12_suites.main),
    }


def test_the_run_record_names_where_the_robustness_came_from_and_the_scores_are_unchanged(
    robust_run: Path, d13_runs: Runs, robustness_dir: Path
) -> None:
    run = read_run(robust_run)

    assert run["robustness_source"]["binding"] == "model_digest"
    assert run["robustness_source"]["robustness_sha256"] == sha256_of(
        robustness_dir / "robustness.json"
    )
    for direction in TRAINED_DIRECTIONS:
        with_block = read_result(robust_run, direction)
        plain = read_result(d13_runs.model, direction)
        assert with_block["metrics"] == plain["metrics"]
        assert with_block["ci95"] == plain["ci95"]
    assert (robust_run / HYPOTHESES_FILE).read_bytes() == (
        d13_runs.model / HYPOTHESES_FILE
    ).read_bytes()


def test_the_robustness_of_the_export_is_refused_for_the_d11_teacher_and_writes_nothing(
    d13_suites: Suites, d11_tiny_export_dir: Path, robustness_dir: Path, scratch: Path
) -> None:
    out_dir = scratch / "teacher_with_the_robustness_of_the_export"

    refused = run_eval(
        "run",
        "--suite",
        d13_suites.devtest,
        "--model",
        d11_tiny_export_dir,
        "--system",
        "d11-teacher",
        "--robustness",
        robustness_dir,
        "--out-dir",
        out_dir,
    )

    assert refused.returncode == 2, refused.stderr
    assert refused.stdout == []
    assert "another model" in refused.stderr
    assert not out_dir.exists()
