"""`rinchen distill evaluate`: what it refuses, with the exit code and the one-line error the
command documents (2 for a bad option, a checkpoint that is not a directory with the files
Eole loads, a glob character or `$` in a path Eole reads, an output directory that is an
input's; 3 for missing or unreadable data and another stage's manifest; 4 when the prediction
fails), and the order of its work: every input is hashed before the prediction, the old
result and manifest go before it, and the manifest is written last.

A refusal leaves the output directory as it was found and never starts the prediction. The
prints and the files of a successful run are in `test_cli_distill_evaluate.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import sentencepiece

from rinchen.cli import distill_evaluate
from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import UpstreamError
from rinchen.manifest import store
from tests.support.distill_cli import (
    CHECKPOINT_FILES,
    EvaluateWorkspace,
    OraclePredict,
    PredictCall,
    install_predictor,
    invoke,
    make_evaluate_workspace,
    run_ok,
    write_checkpoint,
    write_foreign_manifest,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.planted_student import Inputs, build_inputs
from tests.support.tiny_parallel import build_toy, write_corpus_parquet

GLOB_CHARACTERS = ("$", "[", "]", "*", "?")
FAST = ("--max-lines", "3")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("cli_distill_evaluate_refusals"))


@pytest.fixture
def workspace(tmp_path: Path, inputs: Inputs) -> EvaluateWorkspace:
    return make_evaluate_workspace(tmp_path, inputs)


@pytest.fixture
def predictor(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> OraclePredict:
    oracle = OraclePredict(inputs.spm, build_toy().dev)
    install_predictor(monkeypatch, oracle)
    return oracle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- options -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        ("--max-lines", "0"),
        ("--threads", "0"),
        ("--timeout", "x"),
    ],
)
def test_a_usage_error_is_exit_2_and_nothing_is_predicted(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    arguments: tuple[str, str],
) -> None:
    result = invoke(workspace.args(*arguments))

    assert result.exit_code == 2, result.output
    assert arguments[0] in result.stderr
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("timeout", ["0", "-5", "nan", "inf"])
def test_a_timeout_that_is_not_a_positive_finite_number_is_a_config_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict, timeout: str
) -> None:
    result = invoke(workspace.args("--timeout", timeout))

    assert_one_error(result, 2, "--timeout must be a positive, finite number")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("placement", [None, "both"])
def test_the_tag_placement_is_required_and_is_src_or_tgt(
    workspace: EvaluateWorkspace, predictor: OraclePredict, placement: str | None
) -> None:
    result = invoke(workspace.args(tag_placement=placement))

    assert result.exit_code == 2, result.output
    assert "--tag-placement" in result.stderr
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_corpus_given_twice_is_refused_whatever_its_spelling(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    again = (
        workspace.corpus.parent
        / ".."
        / workspace.corpus.parent.name
        / workspace.corpus.name
    )

    result = invoke(workspace.args(corpora=[workspace.corpus, again]))

    assert_one_error(result, 2, f"--corpus {again} names the same file as --corpus")
    assert predictor.calls == []


# --- the checkpoint ------------------------------------------------------------------------------


def test_a_checkpoint_that_is_not_a_directory_is_a_config_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    missing = workspace.root / "nowhere"

    result = invoke(workspace.args(checkpoint=missing))

    assert_one_error(
        result, 2, f"checkpoint config.json not found: {missing}/config.json"
    )
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("name", list(CHECKPOINT_FILES))
def test_a_checkpoint_directory_without_a_file_eole_loads_is_a_config_error_naming_it(
    workspace: EvaluateWorkspace, predictor: OraclePredict, name: str
) -> None:
    (workspace.checkpoint / name).unlink()

    result = invoke(workspace.args())

    assert result.exit_code == 2, result.output
    assert str(workspace.checkpoint / name) in result.stderr
    assert "not found" in result.stderr or "no weights" in result.stderr
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_checkpoint_with_several_weight_shards_has_every_one_hashed(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    shard = workspace.checkpoint / "model.01.safetensors"
    shard.write_bytes(b"a second shard")

    run_ok(workspace.args(*FAST))

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert recorded[shard] == sha256(shard)
    assert recorded[workspace.checkpoint / "model.00.safetensors"] == sha256(
        workspace.checkpoint / "model.00.safetensors"
    )


def test_a_checkpoint_is_evaluated_without_a_manifest_and_whatever_manifest_it_holds(
    workspace: EvaluateWorkspace, predictor: OraclePredict, tmp_path: Path
) -> None:
    assert not (workspace.checkpoint / "manifest.json").exists()
    run_ok(workspace.args(*FAST))
    write_foreign_manifest(workspace.checkpoint, stage="anything")

    run_ok(workspace.args(*FAST, out_dir=tmp_path / "second"))

    assert (tmp_path / "second" / "manifest.json").is_file()


@pytest.mark.parametrize("character", GLOB_CHARACTERS)
def test_a_character_eole_reads_specially_in_the_checkpoint_path_is_refused(
    workspace: EvaluateWorkspace, predictor: OraclePredict, character: str
) -> None:
    checkpoint = write_checkpoint(workspace.root / f"ck{character}pt")

    result = invoke(workspace.args(checkpoint=checkpoint))

    assert result.exit_code == 2, result.output
    assert "checkpoint directory" in result.stderr
    assert character in result.stderr
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_dollar_in_the_output_directory_is_refused_and_a_glob_character_is_fine(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    refused = invoke(workspace.args(out_dir=workspace.root / "out$dir"))

    assert refused.exit_code == 2, refused.output
    assert "output directory" in refused.stderr
    assert predictor.calls == []

    run_ok(workspace.args(*FAST, out_dir=workspace.root / "out[1]"))

    assert (workspace.root / "out[1]" / "result.json").is_file()


# --- the other inputs ---------------------------------------------------------------------------


def test_a_missing_corpus_is_a_data_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    missing = workspace.root / "absent.parquet"

    result = invoke(workspace.args(corpora=[missing]))

    assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_missing_spm_model_is_a_config_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    missing = workspace.root / "absent.model"

    result = invoke(workspace.args(spm_model=missing))

    assert_one_error(result, 2, f"SentencePiece model not found: {missing}")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_corpus_without_a_training_row_is_a_data_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    dev_only = write_corpus_parquet(
        build_toy().dev, workspace.root / "dev_only.parquet"
    )

    result = invoke(workspace.args(corpora=[dev_only]))

    assert_one_error(result, 3, "no training rows: ")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_a_training_direction_without_a_dev_row_is_a_data_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    toy = build_toy()
    rows = [
        *toy.train,
        *(
            row
            for row in toy.dev
            if (row["src_lang"], row["tgt_lang"]) != ("rus_Cyrl", "khk_Cyrl")
        ),
    ]
    path = workspace.root / "short.parquet"
    write_corpus(corpus_frame(rows), path)

    result = invoke(workspace.args(corpora=[path]))

    assert_one_error(result, 3, "no usable dev rows for direction rus_Cyrl-khk_Cyrl")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def model_without_tags(directory: Path) -> Path:
    """A small SentencePiece model with no direction tag among its pieces."""
    text = directory / "plain.txt"
    text.write_text(
        "".join(f"{row['src']}\n{row['tgt']}\n" for row in build_toy().train[:400]),
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(directory / "plain"),
        vocab_size=120,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return directory / "plain.model"


def test_a_spm_model_without_a_direction_tag_is_refused_before_the_old_result_is_touched(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    run_ok(workspace.args(*FAST))
    before = snapshot(workspace.out_dir)
    calls = len(predictor.calls)

    result = invoke(workspace.args(spm_model=model_without_tags(workspace.root)))

    assert result.exit_code == 2, result.output
    assert "the direction tag <2" in result.stderr
    assert "is not one piece of the SentencePiece model" in result.stderr
    assert len(predictor.calls) == calls
    assert snapshot(workspace.out_dir) == before


# --- the output directory -----------------------------------------------------------------------


@pytest.mark.parametrize("input_name", ["corpus", "spm", "checkpoint"])
def test_an_output_directory_that_is_the_directory_of_an_input_is_refused_and_untouched(
    workspace: EvaluateWorkspace, predictor: OraclePredict, input_name: str
) -> None:
    path = {
        "corpus": workspace.corpus,
        "spm": workspace.spm_model,
        "checkpoint": workspace.checkpoint / "config.json",
    }[input_name]
    before = snapshot(path.parent)

    result = invoke(workspace.args(out_dir=path.parent))

    assert_one_error(
        result, 2, f"--out-dir {path.parent} is the directory of the input file "
    )
    assert snapshot(path.parent) == before
    assert predictor.calls == []


def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_before_anything_is_read(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_foreign_manifest(workspace.out_dir)
    before = snapshot(workspace.out_dir)

    def read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the corpus was read before the foreign manifest was refused")

    monkeypatch.setattr(distill_evaluate, "load_training_rows", read)

    result = invoke(workspace.args())

    assert_one_error(
        result, 3, f"{workspace.out_dir} holds the manifest of stage 'train_docify'"
    )
    assert snapshot(workspace.out_dir) == before
    assert predictor.calls == []


# --- a prediction that fails ---------------------------------------------------------------------


def test_a_prediction_that_fails_is_exit_4_with_no_manifest_and_no_result(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(
        monkeypatch,
        OraclePredict(
            inputs.spm,
            build_toy().dev,
            fail=UpstreamError("eole predict exited with code 1"),
        ),
    )

    result = invoke(workspace.args(*FAST))

    assert_one_error(result, 4, "eole predict exited with code 1")
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()
    assert not list(workspace.out_dir.glob("hyp.*"))


def test_a_failed_rerun_leaves_neither_the_old_manifest_nor_the_old_result(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(monkeypatch, OraclePredict(inputs.spm, build_toy().dev))
    run_ok(workspace.args(*FAST))
    assert workspace.manifest.is_file()
    assert workspace.result.is_file()
    install_predictor(
        monkeypatch,
        OraclePredict(
            inputs.spm, build_toy().dev, fail=UpstreamError("eole predict timed out")
        ),
    )

    result = invoke(workspace.args(*FAST))

    assert_one_error(result, 4, "eole predict timed out")
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_an_interrupt_exits_130_and_writes_no_manifest(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(
        monkeypatch,
        OraclePredict(inputs.spm, build_toy().dev, fail=KeyboardInterrupt()),
    )

    result = invoke(workspace.args(*FAST))

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_a_result_that_cannot_be_written_exits_1_and_leaves_no_manifest(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(distill_evaluate, "replace_file", failing)

    result = invoke(workspace.args(*FAST))

    assert_one_error(
        result, 1, f"cannot write the evaluation under {workspace.out_dir}"
    )
    assert "disk full" in result.stderr
    assert not workspace.manifest.exists()


def test_an_output_directory_that_cannot_be_made_exits_1(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    blocked = workspace.root / "blocked"
    blocked.write_text("a file where the directory should be", encoding="utf-8")

    result = invoke(workspace.args(out_dir=blocked / "out"))

    assert result.exit_code == 1, result.output
    assert "cannot write the evaluation under" in result.stderr
    assert predictor.calls == []


# --- hashing before the prediction ----------------------------------------------------------------


def test_every_input_is_hashed_before_the_prediction_starts(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real = distill_evaluate.hash_input

    def hashing(path: Path, **kwargs: Any) -> Any:
        events.append(f"hash:{Path(path).name}")
        return real(path, **kwargs)

    def predicting(call: PredictCall) -> None:
        events.append("predict")

    monkeypatch.setattr(distill_evaluate, "hash_input", hashing)
    install_predictor(
        monkeypatch, OraclePredict(inputs.spm, build_toy().dev, during=predicting)
    )

    run_ok(workspace.args(*FAST))

    assert events == [
        "hash:toy.parquet",
        "hash:toy.model",
        "hash:config.json",
        "hash:vocab.json",
        "hash:model.00.safetensors",
        "predict",
    ]


def test_the_manifest_vouches_for_the_checkpoint_as_it_was_before_the_prediction_read_it(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = workspace.checkpoint / "model.00.safetensors"
    before = sha256(weights)

    def swapping(call: PredictCall) -> None:
        weights.write_bytes(b"swapped while the prediction ran")

    install_predictor(
        monkeypatch, OraclePredict(inputs.spm, build_toy().dev, during=swapping)
    )

    run_ok(workspace.args(*FAST))

    assert sha256(weights) != before
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert {Path(item.path): item.sha256 for item in manifest.inputs}[weights] == before


def test_the_old_manifest_and_result_are_gone_before_the_prediction_runs(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(monkeypatch, OraclePredict(inputs.spm, build_toy().dev))
    run_ok(workspace.args(*FAST))
    seen: list[tuple[bool, bool]] = []

    def looking(call: PredictCall) -> None:
        seen.append((workspace.manifest.exists(), workspace.result.exists()))

    install_predictor(
        monkeypatch, OraclePredict(inputs.spm, build_toy().dev, during=looking)
    )

    run_ok(workspace.args(*FAST))

    assert seen == [(False, False)]


def test_the_manifest_is_written_after_the_result_and_every_hypothesis_file_it_lists(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = distill_evaluate.write_manifest
    present: list[bool] = []

    def checking(out_dir: Path, **kwargs: Any) -> Path:
        present.extend(Path(item.path).is_file() for item in kwargs["outputs"])
        present.append(not (out_dir / "manifest.json").exists())
        return real(out_dir, **kwargs)

    monkeypatch.setattr(distill_evaluate, "write_manifest", checking)

    run_ok(workspace.args(*FAST))

    assert present == [True] * 6
