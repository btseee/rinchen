"""The CLI stages of the tiny pipeline: what each one writes for its commands and reports (D15, spec P2,
P7, P10 and P17, brief G4.4), over the recording fake runner of `tests/support/tiny_run.py` and the real
output of the corpus and filter stages. What every stage does with its commands (order, budget, failure) is
in `test_tiny_model_stages.py`.

A stage's facts are read from the files its commands wrote (the vocabulary file, the manifest of the
launch, the corpus counts of the preparation, the weights, the exports, `parity.json`), never from what a
command printed, and each is checked against a figure worked out independently from the inputs or equal
to the spike's recording.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Final

import polars as pl
import pytest
import yaml

from rinchen.config.synth import load_synth_config
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import MONO_SCHEMA
from rinchen.tiny.states import FAIL_INPUT, fail_exit, is_closed_reason
from tests.support import tiny_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_run import (
    COMMANDS_OF,
    MODEL_STAGES,
    VARIANT,
    at,
    changed_config,
    context,
    ready_for,
    recorded_text,
    run_one,
    run_stages,
    stage_function,
)
from tests.support.tiny_stage_inputs import shipped_config, shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of `tests/support/tiny_run.py`, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
out_dir = tiny_run.out_dir

WEIGHTS: Final = "model.00.safetensors"


def train_rows(out: Path) -> pl.DataFrame:
    return scan_corpus(at(out, "filter", "train.parquet")).collect()


def is_train(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("split").cast(pl.String) == "train")


# ---- spm ---------------------------------------------------------------------------------------------


def test_spm_reports_the_vocabulary_of_the_model_and_the_languages_it_saw(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "spm")

    assert result.facts == {
        "vocab_size": 640,
        "languages": ["eng_Latn", "khk_Cyrl", "mn_Mong", "rus_Cyrl"],
    }


def test_spm_vocabulary_is_counted_in_the_file_the_command_wrote(out_dir: Path) -> None:
    section = shipped_config().spm.model_copy(update={"vocab_size": 700})
    ctx, _ = ready_for(out_dir, "spm", config=changed_config(spm=section))

    result = stage_function("spm")(ctx)

    assert result.facts["vocab_size"] == 700
    assert len(at(out_dir, "spm", "tiny.vocab").read_text("utf-8").splitlines()) == 700
    assert (
        yaml.safe_load(at(out_dir, "spm", "spm.yaml").read_text("utf-8"))["vocab_size"]
        == 700
    )


def test_spm_vocabulary_is_what_the_file_holds_and_not_what_the_profile_asked_for(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "spm", vocab_pieces=631)

    assert shipped_config().spm.vocab_size == 640
    assert result.facts["vocab_size"] == 631


def test_spm_trains_on_the_kept_train_text_of_each_language(out_dir: Path) -> None:
    run_one(out_dir, "spm")
    kept = scan_corpus(at(out_dir, "filter", "kept.parquet")).collect()
    expected: dict[str, list[str]] = {}
    for row in is_train(kept).iter_rows(named=True):
        expected.setdefault(row["src_lang"], []).append(row["src"])
        expected.setdefault(row["tgt_lang"], []).append(row["tgt"])

    files = sorted(at(out_dir, "spm", "text").iterdir())
    assert [path.name for path in files] == sorted(f"{lang}.txt" for lang in expected)
    for path in files:
        assert path.read_text(encoding="utf-8").splitlines() == list(
            dict.fromkeys(expected[path.stem])
        )
        assert path.read_bytes().endswith(b"\n")
    assert "tiny-noise" not in set(is_train(kept)["corpus"].to_list())


def test_spm_writes_the_config_its_command_reads(out_dir: Path) -> None:
    run_one(out_dir, "spm")

    config = yaml.safe_load(at(out_dir, "spm", "spm.yaml").read_text("utf-8"))
    assert config["seed"] == 1234
    assert config["corpus_dir"] == str(at(out_dir, "spm", "text"))
    assert config["output_prefix"] == str(at(out_dir, "spm", "tiny"))
    assert at(out_dir, "spm", "tiny.model").is_file()


# ---- the converter -----------------------------------------------------------------------------------


def test_the_converter_reports_the_five_numbers_of_its_evaluation(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "converter")

    assert result.facts == {
        "cer": 0.0285,
        "word_accuracy": 0.916,
        "n": 500,
        "rule_fallbacks": 0,
        "invalid_outputs": 0,
    }
    for key in ("n", "rule_fallbacks", "invalid_outputs"):
        assert type(result.facts[key]) is int


def test_the_converter_is_built_trained_exported_and_evaluated_on_the_words_of_the_profile(
    out_dir: Path,
) -> None:
    _, runner = run_one(out_dir, "converter")
    directory = at(out_dir, "converter")
    words = str(shipped_config().converter.pairs)
    by_key = {call.key: call.argv for call in runner.calls}

    def value(key: str, option: str) -> str:
        argv = by_key[key]
        return argv[argv.index(option) + 1]

    assert value("convert-script build-data", "--pairs") == words
    assert value("convert-script build-data", "--out-dir") == str(directory / "corpus")
    assert value("convert-script train", "--data-dir") == str(directory / "corpus")
    assert value("convert-script train", "--run-dir") == str(directory / "run")
    assert value("convert-script train", "--spm-config") == str(directory / "char.yaml")
    assert value("convert-script train", "--threads") == "2"
    assert value("convert-script export", "--run-dir") == str(directory / "run")
    assert value("convert-script export", "--out-dir") == str(directory / "model")
    assert value("convert-script eval", "--model-dir") == str(directory / "model")
    assert value("convert-script eval", "--pairs") == words
    assert yaml.safe_load((directory / "char.yaml").read_text("utf-8")) == {
        "vocab_size": 200
    }


# ---- synth -------------------------------------------------------------------------------------------


def test_synth_reports_the_counts_of_the_job_and_its_rows_and_never_its_fingerprint(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "synth")
    recorded = json.loads(recorded_text("counts.json"))

    assert result.facts == {
        "job": "bt_eng_khk",
        "generator": "tiny_oracle",
        "qe_status": "skipped: --no-qe",
        "counts": recorded["counts"],
        "shard_stats": recorded["shard_stats"],
        "rows": 64,
        "chat_requests": 0,
    }
    assert "fingerprint" not in json.dumps(result.facts)


def test_synth_counts_the_rows_of_the_corpus_the_command_wrote(out_dir: Path) -> None:
    result, _ = run_one(out_dir, "synth")

    corpus = scan_corpus(at(out_dir, "synth", "run", "corpus.parquet")).collect()
    assert result.facts["rows"] == corpus.height == len(shipped_fixtures().mono)


def test_the_monolingual_corpus_is_the_khalkha_column_of_the_fixture(
    out_dir: Path,
) -> None:
    run_one(out_dir, "synth")

    frame = pl.read_parquet(at(out_dir, "synth", "mono.parquet"))
    mono = shipped_fixtures().mono
    assert frame.schema == MONO_SCHEMA
    assert frame["id"].to_list() == [row.id for row in mono]
    assert frame["text"].to_list() == [row.khk for row in mono]
    assert set(frame["lang"].to_list()) == {"khk_Cyrl"}


def test_the_synthetic_config_names_the_job_and_a_listener_on_the_loopback(
    out_dir: Path,
) -> None:
    result, runner = run_one(out_dir, "synth")
    directory = at(out_dir, "synth")

    config = load_synth_config(directory / "synth.yaml")
    generator = config.generators["tiny_oracle"]
    assert generator.base_url.startswith("http://127.0.0.1:")  # type: ignore[union-attr]
    assert generator.n == shipped_config().synth.n
    assert list(config.jobs) == [shipped_config().synth.job]
    assert config.jobs["bt_eng_khk"].shard_size == shipped_config().synth.shard_size
    argv = runner.calls[0].argv
    assert argv[5] == "bt_eng_khk"
    assert argv[argv.index("--config") + 1] == str(directory / "synth.yaml")
    assert argv[argv.index("--input") + 1] == str(directory / "mono.parquet")
    assert argv[argv.index("--out-dir") + 1] == str(directory / "run")
    assert "--no-qe" in argv
    assert result.record.state == "ok"


# ---- prepare -----------------------------------------------------------------------------------------


def test_prepare_names_the_variant_it_prepared(out_dir: Path) -> None:
    result, _ = run_one(out_dir, "prepare")

    assert result.facts == {"variant": VARIANT}


def test_prepare_trains_on_the_filtered_corpus_and_the_synthetic_rows(
    out_dir: Path,
) -> None:
    _, runner = run_one(out_dir, "prepare")
    argv = runner.calls[0].argv

    corpora = [argv[i + 1] for i, token in enumerate(argv) if token == "--corpus"]
    assert corpora == [
        str(at(out_dir, "filter", "train.parquet")),
        str(at(out_dir, "synth", "run", "corpus.parquet")),
    ]
    assert argv[argv.index("--spm-model") + 1] == str(at(out_dir, "spm", "tiny.model"))
    assert argv[argv.index("--out-dir") + 1] == str(at(out_dir, "prepare", "ablate"))
    assert argv[argv.index("--matrix") + 1] == str(
        at(out_dir, "prepare", "matrix.yaml")
    )
    assert argv[argv.index("--threads") + 1] == "2"


def test_the_prepared_config_holds_the_steps_of_the_profile(out_dir: Path) -> None:
    section = shipped_config().train.model_copy(update={"steps": 400})
    ctx, _ = ready_for(out_dir, "prepare", config=changed_config(train=section))

    stage_function("prepare")(ctx)

    prepared = yaml.safe_load(
        (
            at(out_dir, "prepare", "ablate", "variants", VARIANT, "config.yaml")
        ).read_text("utf-8")
    )
    assert prepared["training"]["train_steps"] == 400
    assert prepared["training"]["valid_steps"] == 400
    assert prepared["training"]["keep_checkpoint"] == 4


# ---- train -------------------------------------------------------------------------------------------


def test_train_reports_the_steps_the_run_recorded_and_the_rows_it_trained_on(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "train")
    direction = pl.col("src_lang") + "-" + pl.col("tgt_lang")
    authentic = dict(
        is_train(train_rows(out_dir)).group_by(direction.alias("d")).len().iter_rows()
    )
    synthetic = min(64, round(0.2 * authentic["eng_Latn-khk_Cyrl"]))
    expected = dict(authentic)
    expected["eng_Latn-khk_Cyrl"] += synthetic

    assert synthetic > 0
    assert result.facts == {
        "variant": VARIANT,
        "steps": 1500,
        "rows_by_direction": dict(sorted(expected.items())),
        "bt_rows": synthetic,
    }
    assert list(result.facts["rows_by_direction"]) == sorted(expected)


def test_train_launches_the_prepared_config_in_its_run_directory(out_dir: Path) -> None:
    _, runner = run_one(out_dir, "train")
    variant = at(out_dir, "prepare", "ablate", "variants", VARIANT)
    argv = runner.calls[0].argv

    assert argv[argv.index("--config") + 1] == str(variant / "config.yaml")
    assert argv[argv.index("--run-dir") + 1] == str(variant / "run")
    assert argv[argv.index("--threads") + 1] == "2"


def test_the_steps_are_read_from_the_manifest_of_the_run_and_not_from_the_profile(
    out_dir: Path,
) -> None:
    section = shipped_config().train.model_copy(update={"steps": 400})
    ctx, _ = ready_for(out_dir, "train", config=changed_config(train=section))

    result = stage_function("train")(ctx)

    assert result.facts["steps"] == 400


def test_the_steps_are_what_the_launch_recorded_and_not_what_the_profile_asked_for(
    out_dir: Path,
) -> None:
    result, _ = run_one(out_dir, "train", launched_steps=1234)

    assert shipped_config().train.steps == 1500
    assert result.facts["steps"] == 1234


@pytest.mark.parametrize("recorded", [0, -5, 1.5])
def test_a_launch_that_recorded_no_whole_number_of_steps_fails_on_its_input(
    out_dir: Path, recorded: float
) -> None:
    result, _ = run_one(out_dir, "train", launched_steps=recorded)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.facts == {}


# ---- average -----------------------------------------------------------------------------------------


def test_average_reports_the_digest_of_the_averaged_weights(out_dir: Path) -> None:
    result, runner = run_one(out_dir, "average")
    run = at(out_dir, "prepare", "ablate", "variants", VARIANT, "run")
    weights = (run / "averaged" / WEIGHTS).read_bytes()

    assert result.facts == {"weights_sha256": hashlib.sha256(weights).hexdigest()}
    argv = runner.calls[0].argv
    assert argv[argv.index("--checkpoint-dir") + 1] == str(run / "ckpt")
    assert argv[argv.index("--last") + 1] == "2"
    assert "--out" not in argv
    assert weights.startswith(b"average of step_1400,step_1500")


def test_average_takes_as_many_checkpoints_as_the_profile_says(out_dir: Path) -> None:
    section = shipped_config().train.model_copy(update={"average_last": 3})
    ctx, runner = ready_for(out_dir, "average", config=changed_config(train=section))

    result = stage_function("average")(ctx)

    assert runner.calls[0].argv[-1] == "3"
    weights = at(
        out_dir, "prepare", "ablate", "variants", VARIANT, "run", "averaged", WEIGHTS
    )
    assert weights.read_bytes().startswith(b"average of step_1300,step_1400,step_1500")
    assert result.record.state == "ok"


# ---- export ------------------------------------------------------------------------------------------


def test_export_reports_the_digest_of_each_model_it_converted(out_dir: Path) -> None:
    result, runner = run_one(out_dir, "export")
    averaged = at(out_dir, "prepare", "ablate", "variants", VARIANT, "run", "averaged")

    assert set(result.facts) == {"model_sha256"}
    for quantization in ("float32", "int8"):
        model = at(out_dir, "export", quantization, "model.bin").read_bytes()
        assert (
            result.facts["model_sha256"][quantization]
            == hashlib.sha256(model).hexdigest()
        )
    assert (
        result.facts["model_sha256"]["float32"] != result.facts["model_sha256"]["int8"]
    )
    for call in runner.calls:
        argv = call.argv
        quantization = argv[argv.index("--quantization") + 1]
        assert argv[argv.index("--checkpoint") + 1] == str(averaged)
        assert argv[argv.index("--out-dir") + 1] == str(
            at(out_dir, "export", quantization)
        )
        assert argv[argv.index("--spm-model") + 1] == str(
            at(out_dir, "spm", "tiny.model")
        )
        assert argv[argv.index("--tag-placement") + 1] == "src"


def test_export_converts_the_quantizations_of_the_profile_in_its_order(
    out_dir: Path,
) -> None:
    section = shipped_config().export.model_copy(
        update={"quantizations": ("int8", "float32")}
    )
    ctx, runner = ready_for(out_dir, "export", config=changed_config(export=section))

    result = stage_function("export")(ctx)

    assert runner.keys == ["export convert int8", "export convert float32"]
    assert list(result.facts["model_sha256"]) == ["int8", "float32"]


# ---- parity ------------------------------------------------------------------------------------------


def test_parity_reports_the_recorded_run_without_its_lines_and_mismatches(
    out_dir: Path,
) -> None:
    result, runner = run_one(out_dir, "parity")
    document = json.loads(recorded_text("parity.json"))
    del document["lines"], document["mismatches"]

    assert result.facts == document
    assert result.facts["exact_share"] == 1.0
    argv = runner.calls[0].argv
    run = at(out_dir, "prepare", "ablate", "variants", VARIANT, "run")
    assert argv[argv.index("--export") + 1] == str(at(out_dir, "export", "float32"))
    assert argv[argv.index("--out-dir") + 1] == str(at(out_dir, "parity", "float32"))
    assert argv[argv.index("--checkpoint") + 1] == str(run / "averaged")
    assert argv[argv.index("--corpus") + 1] == str(
        at(out_dir, "filter", "train.parquet")
    )
    assert argv[argv.index("--max-lines") + 1] == "50"
    assert argv[argv.index("--threads") + 1] == "2"


def test_a_parity_that_misses_fails_the_stage_and_still_carries_the_measurement(
    out_dir: Path,
) -> None:
    missed = {"passes": False, "exact": 60, "exact_share": 60 / 70}

    result, _ = run_one(
        out_dir,
        "parity",
        fail={"export parity": 4},
        write_on_failure={"export parity"},
        parity=missed,
    )

    assert result.record.state == "fail"
    assert result.record.reason == "exit code 4"
    assert result.facts["passes"] is False
    assert result.facts["exact_share"] == 60 / 70
    assert "lines" not in result.facts and "mismatches" not in result.facts
    assert result.record.detail == {}


def test_a_parity_that_missed_and_wrote_nothing_carries_no_facts(out_dir: Path) -> None:
    result, _ = run_one(out_dir, "parity", fail={"export parity": 4})

    assert result.record.reason == "exit code 4"
    assert result.facts == {}


def test_parity_without_a_float32_export_fails_on_its_input(out_dir: Path) -> None:
    section = shipped_config().export.model_copy(update={"quantizations": ("int8",)})
    ctx, _ = ready_for(out_dir, "parity", config=changed_config(export=section))

    result = stage_function("parity")(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.facts == {}


# ---- the seed ----------------------------------------------------------------------------------------


def test_the_seed_reaches_the_sentencepiece_config_the_synthetic_run_and_the_matrix(
    out_dir: Path,
) -> None:
    ctx = context(out_dir)
    for stage in ("spm", "synth", "prepare"):
        assert stage_function(stage, 77)(ctx).record.state == "ok"

    assert (
        yaml.safe_load(at(out_dir, "spm", "spm.yaml").read_text("utf-8"))["seed"] == 77
    )
    synth = next(call for call in ctx.runner.calls if call.key == "synth run")  # type: ignore[attr-defined]
    assert synth.argv[synth.argv.index("--seed") + 1] == "77"
    assert yaml.safe_load(at(out_dir, "prepare", "matrix.yaml").read_text("utf-8"))[
        "seeds"
    ] == [77]
    assert at(
        out_dir, "prepare", "ablate", "variants", "t1_bt0.2_src_s77", "config.yaml"
    ).is_file()


# ---- an input that is not there --------------------------------------------------------------------

RUN: Final = f"ablate/variants/{VARIANT}/run"


@pytest.mark.parametrize(
    ("stage", "owner", "name", "reason", "ran"),
    [
        ("spm", "filter", "kept.parquet", FAIL_INPUT, 0),
        ("prepare", "filter", "train.parquet", fail_exit(2), 1),
        ("prepare", "synth", "run/corpus.parquet", fail_exit(2), 1),
        ("train", "prepare", f"ablate/variants/{VARIANT}/config.yaml", FAIL_INPUT, 0),
        ("average", "train", f"{RUN}/ckpt", fail_exit(2), 1),
        ("export", "average", f"{RUN}/averaged/{WEIGHTS}", fail_exit(2), 1),
        ("parity", "export", "float32/model.bin", fail_exit(2), 1),
        ("parity", "filter", "train.parquet", fail_exit(2), 1),
    ],
)
def test_an_input_that_an_earlier_stage_should_have_written_is_a_failure(
    out_dir: Path, stage: str, owner: str, name: str, reason: str, ran: int
) -> None:
    ctx, runner = ready_for(out_dir, stage)
    target = at(out_dir, owner, name)
    shutil.rmtree(target) if target.is_dir() else target.unlink()

    result = stage_function(stage)(ctx)

    assert (result.record.state, result.record.reason) == ("fail", reason)
    assert is_closed_reason(reason)
    assert len(runner.calls) == ran
    assert result.facts == {}


@pytest.mark.parametrize(
    ("stage", "key", "name"),
    [
        ("spm", "spm train", "tiny.vocab"),
        ("synth", "synth run", "run/counts.json"),
        ("synth", "synth run", "run/corpus.parquet"),
        ("train", "train launch", f"ablate/variants/{VARIANT}/run/manifest.json"),
        ("average", "train average", f"{RUN}/averaged/{WEIGHTS}"),
        ("export", "export convert int8", "int8/model.bin"),
        ("parity", "export parity", "float32/parity.json"),
    ],
)
def test_a_command_that_succeeds_but_leaves_out_a_file_the_stage_reads_fails_on_its_input(
    out_dir: Path, stage: str, key: str, name: str
) -> None:
    target = at(out_dir, stage, name)
    ctx, runner = ready_for(out_dir, stage, after={key: lambda call: target.unlink()})

    result = stage_function(stage)(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.facts == {}
    assert key in runner.keys


def test_a_filter_output_that_is_not_a_corpus_is_a_failure_of_the_input_and_runs_nothing(
    out_dir: Path,
) -> None:
    ctx, runner = ready_for(out_dir, "spm")
    at(out_dir, "filter", "kept.parquet").write_bytes(b"not a table")

    result = stage_function("spm")(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert runner.calls == []


def test_two_prepared_variants_are_a_failure_of_the_input_and_run_nothing(
    out_dir: Path,
) -> None:
    ctx, runner = ready_for(out_dir, "train")
    other = at(out_dir, "prepare", "ablate", "variants", "other")
    other.mkdir()
    (other / "config.yaml").write_text("seed: 1\n", encoding="utf-8")

    result = stage_function("train")(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert runner.calls == []


@pytest.mark.parametrize("stage", ["train", "average", "export", "parity"])
def test_a_stage_that_needs_the_prepared_variant_fails_on_its_input_when_there_is_none(
    out_dir: Path, stage: str
) -> None:
    ctx = context(out_dir)

    result = stage_function(stage)(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert ctx.runner.calls == []  # type: ignore[attr-defined]


def test_a_whole_run_over_the_recorded_commands_is_ok_stage_by_stage(
    out_dir: Path,
) -> None:
    ctx = context(out_dir)

    results = run_stages(ctx, "parity")

    assert list(results) == list(MODEL_STAGES)
    assert [result.record.state for result in results.values()] == ["ok"] * 8
    assert len(ctx.runner.calls) == sum(len(keys) for keys in COMMANDS_OF.values())  # type: ignore[attr-defined]
