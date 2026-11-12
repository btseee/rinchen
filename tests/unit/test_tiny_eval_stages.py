"""The `evaluate` stage of the tiny pipeline over a recording fake runner (D15, spec P2, P8, P9 and F8, brief
G5.4). The neural seam is in `test_tiny_neural_seam.py`.

The stage writes the five evaluation suites from the fixtures and the rows the filter kept, runs
`rinchen eval run` for the int8 export over the three memorised suites and the two held-out ones (the
float32 export over the memorised ones), `rinchen eval report` over the held-out runs, and reads the
`result.json` and the `hypotheses.jsonl` the commands wrote. The runner (`tests/support/tiny_run_eval.py`)
plays the commands over the recordings of the spike; the real binaries run in the gate.

The figures expected here are worked out from the recordings and from the files the runner wrote, with
code that is not the stage's.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest

from rinchen.config.tiny import DecodeSection, EvaluationSection, ExportSection
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.writers import write_corpus
from rinchen.eval import suites as suite_reader
from rinchen.tiny import commands
from rinchen.tiny.states import (
    FAIL_INPUT,
    SKIP_EARLIER_FAILED,
    fail_exit,
    fail_timeout,
    is_closed_reason,
)
from rinchen.tiny.step_runner import NOT_STARTED, command_timeout
from rinchen.tiny.thresholds import (
    MEMORISED_DIRECTIONS,
    ThresholdInputs,
    evaluate_thresholds,
)
from tests.support import tiny_run, tiny_run_eval
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_help import check_argument_list
from tests.support.tiny_run import at, changed_config, recorded_text, tree
from tests.support.tiny_run_eval import (
    ALL_KEYS,
    FLOAT32,
    HELDOUT,
    INT8,
    MEMORISED,
    MODEL_RUNS,
    run_evaluation,
    stage_dir,
)
from tests.support.tiny_stage_inputs import shipped_config, shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of the support modules, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
model_outputs = tiny_run_eval.model_outputs
evaluated_inputs = tiny_run_eval.evaluated_inputs
eval_out = tiny_run_eval.eval_out

HELDOUT_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "rus_Cyrl-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
)
SUITE_OF: Final = dict(
    zip(
        MEMORISED_DIRECTIONS,
        [MEMORISED[0], MEMORISED[0], MEMORISED[1], MEMORISED[1], MEMORISED[2]],
        strict=True,
    )
)
FORBIDDEN: Final = (
    "au" + "to",
    "cu" + "da",
    "fe" + "tch",
    "up" + "load",
    "pub" + "lic",
)


run = run_evaluation


def option(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def result_of(out: Path, suite: str, system: str, direction: str) -> dict[str, Any]:
    path = stage_dir(out) / "runs" / suite / system / direction / "result.json"
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def recorded(quantization: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        recorded_text(f"result_memorised_eng_khk_{quantization}.json")
    )
    return document


def kept_pairs(out: Path) -> set[tuple[str, str]]:
    kept = scan_corpus(at(out, "filter", "kept.parquet")).collect()
    train = kept.filter(pl.col("split").cast(pl.String) == "train")
    return set(zip(train["src"].to_list(), train["tgt"].to_list(), strict=True))


# ---- the commands -------------------------------------------------------------------------------------


def test_the_stage_is_ok_and_runs_the_commands_in_the_order_of_its_parts(
    eval_out: Path,
) -> None:
    result, runner = run(eval_out)

    assert result.record.id == "evaluate"
    assert result.record.state == "ok"
    assert result.record.reason is None
    assert runner.keys == ALL_KEYS


def test_without_the_stand_ins_the_neural_run_is_not_made(eval_out: Path) -> None:
    result, runner = run(eval_out, qe_fake=False)

    assert result.record.state == "ok"
    assert runner.keys == [*MODEL_RUNS, "eval report"]
    assert not (stage_dir(eval_out) / "scores").exists()


def test_every_run_of_a_model_decodes_at_the_beam_of_the_profile(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    for call in runner.calls[: len(MODEL_RUNS)]:
        assert option(call.argv, "--beam-size") == "1"
    assert "--beam-size" not in runner.calls[len(MODEL_RUNS)].argv
    assert "--beam-size" not in runner.calls[-1].argv


def test_the_beam_is_the_one_of_the_profile_and_not_a_number_of_the_stage(
    eval_out: Path,
) -> None:
    config = changed_config(decode=DecodeSection(beam_size=3))

    _, runner = run(eval_out, config=config)

    assert {option(call.argv, "--beam-size") for call in runner.calls[:8]} == {"3"}


def test_the_robustness_block_is_attached_to_the_held_out_runs_only(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    robustness = str(eval_out.resolve() / "10_robustness")
    with_block = [call.key for call in runner.calls if "--robustness" in call.argv]
    assert with_block == [f"eval run {suite} {INT8}" for suite in HELDOUT]
    for call in runner.calls:
        if "--robustness" in call.argv:
            assert option(call.argv, "--robustness") == robustness


def test_every_run_is_told_the_bootstrap_draws_of_the_profile_and_the_report_is_not(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    for call in runner.calls[:-1]:
        assert option(call.argv, "--resamples") == "1000"
    assert "--resamples" not in runner.calls[-1].argv


def test_the_draws_are_the_profile_s(eval_out: Path) -> None:
    section = EvaluationSection(
        memorised_split="test", heldout_split="devtest", resamples=250
    )

    _, runner = run(eval_out, config=changed_config(evaluation=section))

    assert {option(call.argv, "--resamples") for call in runner.calls[:-1]} == {"250"}


def test_each_system_runs_its_own_export(eval_out: Path) -> None:
    _, runner = run(eval_out)

    exports = eval_out.resolve() / "08_export"
    for call in runner.calls[: len(MODEL_RUNS)]:
        system = option(call.argv, "--system")
        assert option(call.argv, "--model") == str(
            exports / system.removeprefix("tiny-")
        ), call.key


def test_the_runs_the_suites_and_the_report_are_in_the_directories_of_the_stage(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    directory = stage_dir(eval_out)
    for call in runner.calls[:-1]:
        suite = Path(option(call.argv, "--suite"))
        assert suite.parent == directory / "suites"
        assert suite.name == f"{call.key.split()[2]}.json"
        assert option(call.argv, "--out-dir") == str(directory / "runs")
    report = runner.calls[-1].argv
    assert option(report, "--out-dir") == str(directory / "report")
    assert [report[i + 1] for i, part in enumerate(report) if part == "--suite"] == [
        str(directory / "suites" / f"{suite}.json") for suite in HELDOUT
    ]
    assert [report[i + 1] for i, part in enumerate(report) if part == "--run"] == [
        str(directory / "runs" / suite / INT8) for suite in HELDOUT
    ]


def test_every_list_is_a_call_of_the_real_command_with_options_it_has(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    for call in runner.calls:
        words = ("eval", "report") if call.key == "eval report" else ("eval", "run")
        assert check_argument_list(call.argv, words) == [], call.key


def test_no_argument_holds_a_device_a_download_or_an_upload(eval_out: Path) -> None:
    _, runner = run(eval_out)

    for call in runner.calls:
        words = {part.lower() for part in call.argv[3:]}
        assert not words & set(FORBIDDEN), call.key
        assert not any(word.startswith("--dev" + "ice") for word in words), call.key


def test_every_command_runs_in_the_stage_directory_with_its_own_log(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    for call in runner.calls:
        assert call.cwd == stage_dir(eval_out)
    names = [call.log.name for call in runner.calls]
    assert names[0] == "evaluate-int8-tiny-memorised-eng.log"
    assert names[3] == "evaluate-float32-tiny-memorised-eng.log"
    assert names[6] == "evaluate-int8-tiny-heldout-eng.log"
    assert names[-2:] == ["evaluate-neural.log", "evaluate-report.log"]
    assert {call.log.parent for call in runner.calls} == {stage_dir(eval_out) / "logs"}


def test_the_budget_is_read_again_before_each_command(eval_out: Path) -> None:
    left = iter([1000.0, 400.0, 300.0, 299.5, 250.2, 100.0, 50.0, 7.9, 2.0, 0.1])

    _, runner = run(eval_out, remaining=lambda: next(left))

    assert [call.timeout for call in runner.calls] == [
        300,
        300,
        300,
        299,
        250,
        100,
        50,
        7,
        2,
        1,
    ]
    assert command_timeout(300, 299.5) == 299


# ---- the suites ----------------------------------------------------------------------------------------


def test_the_five_suites_are_written_and_read_back_as_fixture_suites(
    eval_out: Path,
) -> None:
    run(eval_out)

    directory = stage_dir(eval_out) / "suites"
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        f"{name}.json" for name in (*MEMORISED, *HELDOUT)
    )
    for name in MEMORISED:
        read, _ = suite_reader.read_suite(directory / f"{name}.json")
        assert (read.kind, read.benchmark, read.split) == ("fixture", "tiny", "test")
    for name in HELDOUT:
        read, _ = suite_reader.read_suite(directory / f"{name}.json")
        assert (read.kind, read.benchmark, read.split) == ("fixture", "tiny", "devtest")


def test_no_suite_is_split_dev(eval_out: Path) -> None:
    run(eval_out)

    splits = {
        suite_reader.read_suite(path)[0].split
        for path in (stage_dir(eval_out) / "suites").iterdir()
    }
    assert splits == {"test", "devtest"}


def test_the_splits_are_the_ones_of_the_profile(eval_out: Path) -> None:
    section = EvaluationSection(
        memorised_split="devtest", heldout_split="test", resamples=1000
    )

    run(eval_out, config=changed_config(evaluation=section))

    directory = stage_dir(eval_out) / "suites"
    assert suite_reader.read_suite(directory / "tiny-memorised-eng.json")[0].split == (
        "devtest"
    )
    assert (
        suite_reader.read_suite(directory / "tiny-heldout-eng.json")[0].split == "test"
    )


def test_the_memorised_suites_hold_the_rows_the_filter_kept(eval_out: Path) -> None:
    run(eval_out)

    fixtures = shipped_fixtures()
    kept = kept_pairs(eval_out)
    directory = stage_dir(eval_out) / "suites"
    for name, pair in (
        ("tiny-memorised-eng", "eng_Latn-khk_Cyrl"),
        ("tiny-memorised-rus", "rus_Cyrl-khk_Cyrl"),
    ):
        expected = sum(
            1
            for row in fixtures.pairs[pair]
            if row.split == "train" and (row.src, row.tgt) in kept
        )
        assert (
            len(suite_reader.read_suite(directory / f"{name}.json")[0].rows) == expected
        )
    mong = sum(
        1
        for row in fixtures.mong
        if row.split == "train" and (row.mong, row.eng) in kept
    )
    sizes = [
        len(suite_reader.read_suite(directory / f"{name}.json")[0].rows)
        for name in MEMORISED
    ]
    assert sizes[2] == mong
    # The filter flags authentic rows, so some train row is not in a suite.
    assert sum(sizes) < 300 + 300 + 50


def test_a_row_the_kept_file_lacks_is_not_in_a_memorised_suite(eval_out: Path) -> None:
    row = next(
        item
        for item in shipped_fixtures().pairs["rus_Cyrl-khk_Cyrl"]
        if item.split == "train" and item.kind == "int"
    )
    kept = scan_corpus(at(eval_out, "filter", "kept.parquet")).collect()
    write_corpus(
        kept.filter(~((pl.col("src") == row.src) & (pl.col("tgt") == row.tgt))),
        at(eval_out, "filter", "kept.parquet"),
    )

    run(eval_out)

    suite, _ = suite_reader.read_suite(
        stage_dir(eval_out) / "suites" / "tiny-memorised-rus.json"
    )
    assert len(suite.rows) == 299
    assert f"ru-{row.id}" not in {item.id for item in suite.rows}


def test_the_held_out_suites_do_not_depend_on_the_kept_file(eval_out: Path) -> None:
    run(eval_out)
    names = [f"{name}.json" for name in HELDOUT]
    first = [(stage_dir(eval_out) / "suites" / name).read_bytes() for name in names]
    kept = scan_corpus(at(eval_out, "filter", "kept.parquet")).collect()
    write_corpus(kept.slice(10), at(eval_out, "filter", "kept.parquet"))
    shutil.rmtree(stage_dir(eval_out))

    run(eval_out)

    again = [(stage_dir(eval_out) / "suites" / name).read_bytes() for name in names]
    assert again == first


# ---- the facts -------------------------------------------------------------------------------------------


def test_the_facts_have_the_keys_the_report_is_built_from(eval_out: Path) -> None:
    result, _ = run(eval_out)

    assert sorted(result.facts) == [
        "agreement",
        "heldout",
        "int8_chrf_drop",
        "memorised",
        "memorised_chrf",
        "neural_seam",
        "report",
    ]
    assert result.facts["report"] == {"runs": 2, "suites": 2}
    assert list(result.facts["memorised"]) == list(MEMORISED_DIRECTIONS)
    assert list(result.facts["heldout"]) == list(HELDOUT_DIRECTIONS)


def test_the_detail_is_a_copy_of_the_facts_and_holds_json(eval_out: Path) -> None:
    result, _ = run(eval_out)

    assert result.record.detail == result.facts
    result.facts["memorised_chrf"]["eng_Latn-khk_Cyrl"]["int8"] = -1.0
    assert result.record.detail["memorised_chrf"]["eng_Latn-khk_Cyrl"]["int8"] != -1.0
    json.dumps(result.record.detail)


def test_the_chrf_of_each_direction_and_system_is_read_from_the_results(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out)

    expected = {
        direction: {
            "int8": recorded("int8")["metrics"]["chrf"],
            "float32": recorded("float32")["metrics"]["chrf"],
        }
        for direction in MEMORISED_DIRECTIONS
    }
    assert result.facts["memorised_chrf"] == expected


def test_each_run_gives_its_rows_its_six_metrics_and_its_statuses(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out)

    for direction, suite in (
        ("eng_Latn-khk_Cyrl", "tiny-memorised-eng"),
        ("mn_Mong-eng_Latn", "tiny-memorised-mong"),
    ):
        for quantization in ("int8", "float32"):
            facts = result.facts["memorised"][direction][quantization]
            document = result_of(eval_out, suite, f"tiny-{quantization}", direction)
            assert facts["rows"] == document["n_segments"]
            assert facts["metrics"] == document["metrics"]
            assert facts["statuses"] == document["statuses"] == {"ok": facts["rows"]}
            assert sorted(facts["metrics"]) == [
                "bleu",
                "chrf",
                "chrf_pp",
                "comet22",
                "metricx24",
                "metricx24_qe",
            ]


def test_the_rows_of_a_direction_are_the_rows_of_its_suite(eval_out: Path) -> None:
    result, _ = run(eval_out)

    for suite in MEMORISED:
        read, _ = suite_reader.read_suite(
            stage_dir(eval_out) / "suites" / f"{suite}.json"
        )
        for direction in read.directions:
            assert result.facts["memorised"][direction]["int8"]["rows"] == len(
                read.view(direction).ids
            )


def test_the_held_out_facts_are_the_int8_runs_only(eval_out: Path) -> None:
    result, _ = run(eval_out)

    for direction in HELDOUT_DIRECTIONS:
        by_system = result.facts["heldout"][direction]
        assert list(by_system) == ["int8"]
        assert by_system["int8"]["rows"] == 30
        assert by_system["int8"]["statuses"] == {"ok": 30}
    assert (
        result.facts["heldout"]["eng_Latn-khk_Cyrl"]["int8"]["metrics"]["chrf"]
        == (recorded("int8")["metrics"]["chrf"])
    )


def test_the_drop_is_float32_minus_int8_over_the_five_memorised_directions(
    eval_out: Path,
) -> None:
    chrf = {
        ("tiny-memorised-rus", INT8, "khk_Cyrl-rus_Cyrl"): 60.0,
        ("tiny-memorised-mong", FLOAT32, "mn_Mong-eng_Latn"): 10.0,
        ("tiny-memorised-eng", FLOAT32, "eng_Latn-khk_Cyrl"): 88.5,
        ("tiny-memorised-eng", INT8, "eng_Latn-khk_Cyrl"): 80.0,
    }

    result, _ = run(eval_out, chrf=chrf)

    int8, float32 = (recorded(q)["metrics"]["chrf"] for q in ("int8", "float32"))
    drops = [
        float32 - int8,
        float32 - 60.0,
        10.0 - int8,
        88.5 - 80.0,
    ]
    assert result.facts["int8_chrf_drop"] == {
        "least": min(drops),
        "largest": max(drops),
    }
    assert result.facts["int8_chrf_drop"]["least"] == 10.0 - int8
    assert result.facts["int8_chrf_drop"]["largest"] == float32 - 60.0


def test_a_quantisation_that_gains_has_a_negative_drop(eval_out: Path) -> None:
    result, _ = run(
        eval_out,
        chrf={
            (SUITE_OF[d], system, d): figure
            for d in MEMORISED_DIRECTIONS
            for system, figure in ((INT8, 99.99), (FLOAT32, 99.5))
        },
    )

    assert result.facts["int8_chrf_drop"] == {
        "least": pytest.approx(-0.49, abs=1e-9),
        "largest": pytest.approx(-0.49, abs=1e-9),
    }


def test_the_figures_of_the_held_out_runs_never_feed_the_drop_or_the_chrf_table(
    eval_out: Path,
) -> None:
    chrf = {("tiny-heldout-eng", INT8, d): 1.0 for d in HELDOUT_DIRECTIONS[:2]}

    result, _ = run(eval_out, chrf=chrf)

    # The memorised int8 run of the same directions is another run of the same system.
    assert (
        result.facts["heldout"]["eng_Latn-khk_Cyrl"]["int8"]["metrics"]["chrf"] == 1.0
    )
    assert result.facts["memorised_chrf"]["eng_Latn-khk_Cyrl"]["int8"] != 1.0
    assert result.facts["int8_chrf_drop"]["largest"] == (
        recorded("float32")["metrics"]["chrf"] - recorded("int8")["metrics"]["chrf"]
    )


def test_the_facts_feed_the_thresholds_in_the_shape_they_read(eval_out: Path) -> None:
    result, _ = run(eval_out)

    inputs = ThresholdInputs(
        stages=[],
        memorised_chrf=result.facts["memorised_chrf"],
        parity=None,
        robustness=None,
        elapsed_seconds=1.0,
    )
    found = {
        item.id: item
        for item in evaluate_thresholds(
            inputs, shipped_config().thresholds, shipped_config().budget_seconds
        )
    }

    assert found["train_chrf"].verdict == "pass"
    assert found["train_chrf"].value == recorded("int8")["metrics"]["chrf"]
    assert found["int8_chrf_drop"].verdict == "pass"
    assert found["int8_chrf_drop"].value == pytest.approx(
        recorded("float32")["metrics"]["chrf"] - recorded("int8")["metrics"]["chrf"]
    )


# ---- what goes wrong ---------------------------------------------------------------------------------------


def parts_of(result: Any) -> list[tuple[str, str, str | None]]:
    return [(part.id, part.state, part.reason) for part in result.record.parts]


@pytest.mark.parametrize("code", [1, 3])
def test_a_command_of_the_first_part_that_fails_stops_the_stage_and_skips_the_rest(
    eval_out: Path, code: int
) -> None:
    key = MODEL_RUNS[2]

    result, runner = run(eval_out, fail={key: code})

    assert result.record.state == "fail"
    assert result.record.reason == fail_exit(code)
    assert result.record.detail == {}
    assert result.facts == {}
    assert runner.keys == MODEL_RUNS[:3]
    assert parts_of(result) == [
        ("memorised", "fail", fail_exit(code)),
        ("heldout", "skipped", SKIP_EARLIER_FAILED),
        ("neural_seam", "skipped", SKIP_EARLIER_FAILED),
        ("report", "skipped", SKIP_EARLIER_FAILED),
    ]


def test_a_command_of_the_held_out_part_that_fails_keeps_the_facts_of_the_first(
    eval_out: Path,
) -> None:
    result, runner = run(eval_out, fail={MODEL_RUNS[-1]: 4})

    assert result.record.reason == fail_exit(4)
    assert runner.keys == MODEL_RUNS
    assert parts_of(result)[:2] == [
        ("memorised", "ok", None),
        ("heldout", "fail", fail_exit(4)),
    ]
    assert sorted(result.facts) == [
        "agreement",
        "int8_chrf_drop",
        "memorised",
        "memorised_chrf",
    ]
    assert result.record.detail == {}


def test_a_command_that_times_out_or_cannot_start_fails_with_that_reason(
    eval_out: Path,
) -> None:
    timed, runner = run(eval_out, time_out=[MODEL_RUNS[0]])
    assert timed.record.reason == fail_timeout(runner.calls[0].timeout)

    refused, _ = run(eval_out, not_started=[MODEL_RUNS[0]])
    assert refused.record.reason == NOT_STARTED


def test_the_report_that_fails_is_the_last_part_and_the_others_stay_ok(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out, fail={"eval report": 3})

    assert parts_of(result) == [
        ("memorised", "ok", None),
        ("heldout", "ok", None),
        ("neural_seam", "ok", None),
        ("report", "fail", fail_exit(3)),
    ]
    assert "neural_seam" in result.facts


def test_an_exception_of_the_runner_is_not_hidden(eval_out: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        run(eval_out, raises={MODEL_RUNS[1]: RuntimeError("boom")})


def test_a_result_the_command_did_not_write_is_a_failed_input(eval_out: Path) -> None:
    def remove(call: Any) -> None:
        run_dir = Path(call.one("--out-dir")) / "tiny-memorised-rus" / INT8
        (run_dir / "khk_Cyrl-rus_Cyrl" / "result.json").unlink()

    result, runner = run(eval_out, after={MODEL_RUNS[1]: remove})

    assert result.record.reason == FAIL_INPUT
    assert parts_of(result)[0] == ("memorised", "fail", FAIL_INPUT)
    assert runner.keys == MODEL_RUNS[:2]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(suite="tiny-memorised-eng"),
        lambda d: d.update(system="tiny-float32"),
        lambda d: d.update(direction="khk_Cyrl-eng_Latn"),
        lambda d: d.update(n_segments=1),
        lambda d: d.pop("metrics"),
        lambda d: d["metrics"].update(chrf=None),
    ],
    ids=["suite", "system", "direction", "rows", "no-metrics", "null-chrf"],
)
def test_a_result_that_is_not_the_one_asked_for_is_a_failed_input(
    eval_out: Path, change: Callable[[dict[str, Any]], Any]
) -> None:
    def alter(call: Any) -> None:
        path = (
            Path(call.one("--out-dir"))
            / "tiny-memorised-rus"
            / INT8
            / "rus_Cyrl-khk_Cyrl"
            / "result.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        change(document)
        path.write_text(json.dumps(document), encoding="utf-8")

    result, _ = run(eval_out, after={MODEL_RUNS[1]: alter})

    assert result.record.reason == FAIL_INPUT


def test_hypotheses_that_are_not_there_are_a_failed_input_for_the_agreement(
    eval_out: Path,
) -> None:
    def remove(call: Any) -> None:
        (
            Path(call.one("--out-dir"))
            / "tiny-memorised-mong"
            / FLOAT32
            / "hypotheses.jsonl"
        ).unlink()

    result, _ = run(eval_out, after={MODEL_RUNS[5]: remove})

    assert result.record.reason == FAIL_INPUT
    assert parts_of(result)[0] == ("memorised", "fail", FAIL_INPUT)


@pytest.mark.parametrize("missing", ["int8", "float32"])
def test_a_profile_that_lacks_one_of_the_two_exports_cannot_be_evaluated(
    eval_out: Path, missing: str
) -> None:
    kept = tuple(name for name in ("float32", "int8") if name != missing)
    config = changed_config(export=ExportSection(quantizations=kept))

    result, runner = run(eval_out, config=config)

    assert result.record.reason == FAIL_INPUT
    assert runner.calls == []
    assert parts_of(result)[0] == ("memorised", "fail", FAIL_INPUT)


def test_a_missing_kept_file_is_a_failed_input_and_no_command_runs(
    eval_out: Path,
) -> None:
    (eval_out / "03_filter" / "kept.parquet").unlink()

    result, runner = run(eval_out)

    assert result.record.reason == FAIL_INPUT
    assert runner.calls == []
    assert parts_of(result)[0] == ("memorised", "fail", FAIL_INPUT)


def test_a_missing_robustness_report_fails_the_held_out_part_before_its_commands(
    eval_out: Path,
) -> None:
    (eval_out / "10_robustness" / "report.json").unlink()

    result, runner = run(eval_out)

    assert runner.keys == MODEL_RUNS[:6]
    assert parts_of(result)[:3] == [
        ("memorised", "ok", None),
        ("heldout", "fail", FAIL_INPUT),
        ("neural_seam", "skipped", SKIP_EARLIER_FAILED),
    ]


def test_a_report_the_command_did_not_write_or_of_another_schema_is_a_failed_input(
    eval_out: Path,
) -> None:
    def remove(call: Any) -> None:
        (Path(call.one("--out-dir")) / "report.json").unlink()

    def other(call: Any) -> None:
        (Path(call.one("--out-dir")) / "report.json").write_text(
            '{"schema": "other"}', encoding="utf-8"
        )

    for hook in (remove, other):
        result, _ = run(eval_out, after={"eval report": hook})
        assert parts_of(result)[-1] == ("report", "fail", FAIL_INPUT)
        shutil.rmtree(stage_dir(eval_out))


def test_every_reason_the_stage_and_its_parts_give_is_of_the_closed_list(
    eval_out: Path,
) -> None:
    for options in ({"fail": {MODEL_RUNS[0]: 7}}, {"time_out": [MODEL_RUNS[4]]}):
        result, _ = run(eval_out, **options)
        reasons = [result.record.reason, *(part.reason for part in result.record.parts)]
        assert all(is_closed_reason(reason) for reason in reasons if reason is not None)
        shutil.rmtree(stage_dir(eval_out))


# ---- where it writes ------------------------------------------------------------------------------------------


def test_the_stage_writes_only_into_its_own_directory(eval_out: Path) -> None:
    before = {
        path: data
        for path, data in tree(eval_out).items()
        if not path.startswith("11_evaluate/")
    }

    run(eval_out)

    after = {
        path: data
        for path, data in tree(eval_out).items()
        if not path.startswith("11_evaluate/")
    }
    assert after == before
    assert {child.name for child in (eval_out / "11_evaluate").iterdir()} == {
        "suites",
        "runs",
        "report",
        "scores",
        "logs",
    }


def test_two_runs_in_two_directories_give_the_same_facts_and_the_same_files(
    tmp_path: Path, evaluated_inputs: Path
) -> None:
    results = []
    for name in ("one", "two"):
        out = tmp_path / name
        shutil.copytree(evaluated_inputs, out)
        result, _ = run(out)
        results.append((result.facts, tree(out / "11_evaluate" / "suites")))

    assert results[0] == results[1]


def test_no_text_of_a_fixture_is_in_the_record_or_the_facts(eval_out: Path) -> None:
    result, _ = run(eval_out)

    text = json.dumps([result.record.detail, result.facts])
    sentences = [row.src for row in shipped_fixtures().pairs["eng_Latn-khk_Cyrl"][:50]]
    assert not any(sentence in text for sentence in sentences)


def test_the_commands_of_the_stage_are_the_builders_lists(eval_out: Path) -> None:
    _, runner = run(eval_out)

    directory = stage_dir(eval_out)
    root = eval_out.resolve()
    expected_first = commands.eval_run(
        directory / "suites" / "tiny-memorised-eng.json",
        root / "08_export" / "int8",
        INT8,
        directory / "runs",
        1,
        None,
        resamples=1000,
    )
    assert list(runner.calls[0].argv) == expected_first
    assert list(runner.calls[6].argv) == commands.eval_run(
        directory / "suites" / "tiny-heldout-eng.json",
        root / "08_export" / "int8",
        INT8,
        directory / "runs",
        1,
        root / "10_robustness",
        resamples=1000,
    )
