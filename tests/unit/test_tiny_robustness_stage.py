"""The `robustness` stage of the tiny pipeline over a recording fake runner (D15, spec P2 and P8, brief
G5.4).

The stage makes the robustness suite from the rows the filter kept (read from `03_filter/kept.parquet`),
runs `rinchen eval robustness` over the int8 export at the beam of the profile with the script converter
attached, and reads the `report.json` the command wrote. The runner (`tests/support/tiny_run_eval.py`)
plays the command: it reads the suite with the real reader and refuses what the real command would
refuse, so a stage that wires a path wrongly fails on it. The real binaries run in the gate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest

from rinchen.config.tiny import DecodeSection
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.writers import write_corpus
from rinchen.eval import robust_suite
from rinchen.tiny import commands
from rinchen.tiny.eval_stages import run_robustness
from rinchen.tiny.states import FAIL_INPUT, fail_exit, fail_timeout, is_closed_reason
from rinchen.tiny.step_runner import NOT_STARTED, command_timeout
from rinchen.tiny.suites import robustness_suite
from tests.support import tiny_run, tiny_run_eval
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_run import at, changed_config, context, tree
from tests.support.tiny_run_eval import EvalRunner
from tests.support.tiny_stage_inputs import shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of the support modules, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
model_outputs = tiny_run_eval.model_outputs

KEY: Final = "eval robustness"
FORBIDDEN: Final = (
    "au" + "to",
    "cu" + "da",
    "fe" + "tch",
    "up" + "load",
    "pub" + "lic",
)


@pytest.fixture
def model_out(tmp_path: Path, model_outputs: Path) -> Path:
    out = tmp_path / "out"
    shutil.copytree(model_outputs, out)
    return out


def run(out: Path, **options: Any) -> tuple[Any, EvalRunner]:
    runner = EvalRunner(**options)
    return run_robustness(context(out, runner=runner)), runner


def option(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def kept_rows(out: Path) -> pl.DataFrame:
    return scan_corpus(at(out, "filter", "kept.parquet")).collect()


def kept_pairs_of(frame: pl.DataFrame) -> set[tuple[str, str]]:
    train = frame.filter(pl.col("split").cast(pl.String) == "train")
    return set(zip(train["src"].to_list(), train["tgt"].to_list(), strict=True))


def without(out: Path, keep: pl.Expr) -> None:
    """Rewrite the kept corpus of `out` with the rows `keep` selects."""
    write_corpus(kept_rows(out).filter(keep), at(out, "filter", "kept.parquet"))


# ---- the command ------------------------------------------------------------------------------------


def test_the_stage_is_ok_after_one_command_in_its_own_directory(
    model_out: Path,
) -> None:
    result, runner = run(model_out)

    assert result.record.id == "robustness"
    assert result.record.state == "ok"
    assert result.record.reason is None
    assert runner.keys == [KEY]
    assert runner.calls[0].cwd == at(model_out, "robustness").resolve()
    assert runner.calls[0].log == at(
        model_out, "robustness", "logs", "robustness-run.log"
    )


def test_the_argument_list_is_the_builders_over_the_stages_own_paths(
    model_out: Path,
) -> None:
    _, runner = run(model_out)

    root = model_out.resolve()
    directory = root / "10_robustness"
    assert list(runner.calls[0].argv) == commands.eval_robustness(
        directory / "suites" / "tiny-robust.json",
        root / "08_export" / "int8",
        directory,
        1,
        root / "05_converter" / "model",
        root / "05_converter" / "run" / "converter_char.model",
    )


def test_the_suite_is_not_in_the_output_directory_the_command_writes_into(
    model_out: Path,
) -> None:
    _, runner = run(model_out)

    argv = runner.calls[0].argv
    assert Path(option(argv, "--suite")).parent != Path(option(argv, "--out-dir"))


def test_the_beam_is_the_one_of_the_profile(model_out: Path) -> None:
    config = changed_config(decode=DecodeSection(beam_size=3))

    runner = EvalRunner()
    run_robustness(context(model_out, runner=runner, config=config))

    assert option(runner.calls[0].argv, "--beam-size") == "3"


def test_the_model_is_the_int8_export_and_the_converter_is_attached_whole(
    model_out: Path,
) -> None:
    _, runner = run(model_out)

    argv = runner.calls[0].argv
    root = model_out.resolve()
    assert option(argv, "--model") == str(root / "08_export" / "int8")
    assert option(argv, "--script-converter-ct2") == str(
        root / "05_converter" / "model"
    )
    assert option(argv, "--script-converter-spm") == str(
        root / "05_converter" / "run" / "converter_char.model"
    )
    assert "--hypotheses" not in argv


def test_no_argument_holds_a_device_a_download_or_an_upload(model_out: Path) -> None:
    _, runner = run(model_out)

    words = {part.lower() for part in runner.calls[0].argv[3:]}
    assert not words & set(FORBIDDEN)
    assert not any(word.startswith("--dev" + "ice") for word in words)


def test_the_command_gets_the_stage_timeout_or_what_is_left(model_out: Path) -> None:
    _, runner = run(model_out)
    assert runner.calls[0].timeout == 120

    tight = EvalRunner()
    run_robustness(context(model_out, runner=tight, remaining=lambda: 7.5))
    assert tight.calls[0].timeout == command_timeout(120, 7.5) == 7


# ---- the suite ----------------------------------------------------------------------------------------


def test_the_suite_is_made_of_the_rows_the_filter_kept(model_out: Path) -> None:
    _, _ = run(model_out)

    expected = robustness_suite(shipped_fixtures(), kept_pairs_of(kept_rows(model_out)))
    written, _ = robust_suite.read_suite(
        at(model_out, "robustness", "suites", "tiny-robust.json")
    )
    assert written.to_bytes() == expected.to_bytes()
    assert written.name == "tiny-robust"
    assert 0 < written.row_counts()["placeholders"] <= 240


def test_a_row_the_kept_file_lacks_is_not_in_the_suite(model_out: Path) -> None:
    before = run(model_out)[0].facts["rows"]
    row = next(
        item
        for item in shipped_fixtures().pairs["rus_Cyrl-khk_Cyrl"]
        if item.kind == "placeholder" and item.split == "train"
    )
    without(model_out, ~((pl.col("src") == row.src) & (pl.col("tgt") == row.tgt)))

    after = run(model_out)[0].facts["rows"]

    assert after["placeholders"] == before["placeholders"] - 2
    assert {key: after[key] for key in after if key != "placeholders"} == {
        key: before[key] for key in before if key != "placeholders"
    }


def test_a_row_that_is_not_a_train_row_of_the_kept_file_is_not_used(
    model_out: Path,
) -> None:
    frame = kept_rows(model_out)
    held_out = pl.col("split").cast(pl.String) != "train"
    assert frame.filter(held_out).height > 0
    before = run(model_out)[0].facts["rows"]

    without(model_out, pl.col("split").cast(pl.String) == "train")

    assert run(model_out)[0].facts["rows"] == before


def test_a_pair_the_kept_file_calls_a_dev_row_is_not_a_train_row_of_the_suite(
    model_out: Path,
) -> None:
    before = run(model_out)[0].facts["rows"]
    row = next(
        item
        for item in shipped_fixtures().pairs["eng_Latn-khk_Cyrl"]
        if item.kind == "placeholder" and item.split == "train"
    )
    frame = kept_rows(model_out)
    named = (pl.col("src") == row.src) & (pl.col("tgt") == row.tgt)
    relabelled = frame.with_columns(
        pl.when(named)
        .then(pl.lit("dev").cast(frame.schema["split"]))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    write_corpus(relabelled, at(model_out, "filter", "kept.parquet"))

    after = run(model_out)[0].facts["rows"]

    assert after["placeholders"] == before["placeholders"] - 2


def test_the_stage_reads_the_kept_file_and_not_the_fixtures(model_out: Path) -> None:
    without(model_out, pl.col("src_lang").cast(pl.String) == "mn_Mong")

    rows = run(model_out)[0].facts["rows"]

    assert "mong" in rows
    assert "numbers" not in rows
    assert "placeholders" not in rows


# ---- the facts -----------------------------------------------------------------------------------------


def test_the_facts_are_the_six_values_their_coverage_and_the_rows_of_the_suite(
    model_out: Path,
) -> None:
    result, _ = run(model_out)

    recorded = json.loads(tiny_run.recorded_text("robustness_report.json"))
    assert sorted(result.facts) == ["coverage", "rows", "values"]
    assert result.facts["values"] == recorded["robustness"]
    assert result.facts["coverage"] == recorded["coverage"]
    expected = robustness_suite(shipped_fixtures(), kept_pairs_of(kept_rows(model_out)))
    assert result.facts["rows"] == expected.row_counts()


def test_the_detail_is_a_copy_of_the_facts(model_out: Path) -> None:
    result, _ = run(model_out)

    assert result.record.detail == result.facts
    result.facts["rows"]["numbers"] = -1
    assert result.record.detail["rows"]["numbers"] != -1
    json.dumps(result.record.detail)


def test_a_value_under_its_floor_is_a_measurement_and_not_a_failure(
    model_out: Path,
) -> None:
    result, _ = run(model_out, robustness={"script_purity": 0.5, "term_acc": None})

    assert result.record.state == "ok"
    assert result.facts["values"]["script_purity"] == 0.5
    assert result.facts["values"]["term_acc"] is None


def test_the_stage_writes_only_into_its_own_directory(model_out: Path) -> None:
    before = {
        path: data
        for path, data in tree(model_out).items()
        if not path.startswith("10_robustness/")
    }

    run(model_out)

    after = {
        path: data
        for path, data in tree(model_out).items()
        if not path.startswith("10_robustness/")
    }
    assert after == before
    assert (model_out / "10_robustness" / "report.json").is_file()


# ---- what goes wrong -------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [1, 2, 4])
def test_a_command_that_exits_with_a_code_fails_the_stage_with_it(
    model_out: Path, code: int
) -> None:
    result, _ = run(model_out, fail={KEY: code})

    assert result.record.state == "fail"
    assert result.record.reason == fail_exit(code)
    assert result.record.detail == {}
    assert result.facts == {}


def test_a_command_that_times_out_fails_the_stage_with_its_bound(
    model_out: Path,
) -> None:
    result, runner = run(model_out, time_out=[KEY])

    assert result.record.reason == fail_timeout(runner.calls[0].timeout)


def test_a_command_that_cannot_start_fails_the_stage(model_out: Path) -> None:
    result, _ = run(model_out, not_started=[KEY])

    assert result.record.reason == NOT_STARTED


def test_an_exception_of_the_runner_is_not_hidden(model_out: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        run(model_out, raises={KEY: RuntimeError("boom")})


@pytest.mark.parametrize("missing", ["05_converter/model", "08_export/int8"])
def test_an_input_the_command_cannot_find_is_the_exit_code_of_its_refusal(
    model_out: Path, missing: str
) -> None:
    shutil.rmtree(model_out / missing)

    result, _ = run(model_out)

    assert result.record.reason == fail_exit(2)


def test_a_missing_converter_model_file_is_refused_too(model_out: Path) -> None:
    (model_out / "05_converter" / "run" / "converter_char.model").unlink()

    assert run(model_out)[0].record.reason == fail_exit(2)


def test_a_report_the_command_did_not_write_is_a_failed_input(model_out: Path) -> None:
    result, _ = run(
        model_out,
        after={
            KEY: lambda call: (Path(call.one("--out-dir")) / "report.json").unlink()
        },
    )

    assert result.record.state == "fail"
    assert result.record.reason == FAIL_INPUT
    assert result.facts == {}


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"schema": "other"}'])
def test_a_report_that_is_not_the_one_the_command_writes_is_a_failed_input(
    model_out: Path, content: str
) -> None:
    result, _ = run(
        model_out,
        after={
            KEY: lambda call: (Path(call.one("--out-dir")) / "report.json").write_text(
                content, encoding="utf-8"
            )
        },
    )

    assert result.record.reason == FAIL_INPUT


@pytest.mark.parametrize("name", ["03_filter/kept.parquet"])
def test_a_missing_kept_file_is_a_failed_input_and_no_command_runs(
    model_out: Path, name: str
) -> None:
    (model_out / name).unlink()

    result, runner = run(model_out)

    assert result.record.reason == FAIL_INPUT
    assert runner.calls == []


def test_a_kept_file_that_is_not_a_corpus_is_a_failed_input(model_out: Path) -> None:
    (model_out / "03_filter" / "kept.parquet").write_bytes(b"not a parquet file")

    result, runner = run(model_out)

    assert result.record.reason == FAIL_INPUT
    assert runner.calls == []


def test_a_kept_file_without_a_train_row_makes_no_suite_and_no_command_runs(
    model_out: Path,
) -> None:
    without(model_out, pl.col("split").cast(pl.String) == "dev")

    result, runner = run(model_out)

    assert result.record.reason == FAIL_INPUT
    assert runner.calls == []


def test_every_reason_the_stage_gives_is_of_the_closed_list(model_out: Path) -> None:
    for options in ({"fail": {KEY: 3}}, {"time_out": [KEY]}, {"not_started": [KEY]}):
        result, _ = run(model_out, **options)
        assert result.record.reason is not None
        assert is_closed_reason(result.record.reason)


def test_no_text_of_a_fixture_is_in_the_record_or_the_facts(model_out: Path) -> None:
    result, _ = run(model_out)
    text = json.dumps([result.record.detail, result.facts])
    sentences = [row.src for row in shipped_fixtures().pairs["eng_Latn-khk_Cyrl"][:50]]

    assert not any(sentence in text for sentence in sentences)
