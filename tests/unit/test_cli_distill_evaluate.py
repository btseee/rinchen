"""`rinchen distill evaluate`: what it prints and writes (the scores of each direction, the
record `result.json`, the hypotheses and the manifest) and what it asks the prediction for.
What it refuses and the order of its work are in `test_cli_distill_evaluate_refusals.py`.

No test starts Eole: the command calls the prediction through its module-level
`predict_tokens`, which `tests.support.distill_cli.OraclePredict` replaces. The oracle answers
every dev line with its reference, so each direction scores chrF 100.00 and BLEU 100.00,
except where a test silences the Mongolian targets: then the two directions into Mongolian
score 0.00 and the mean of the four is 50.00. The toy corpus holds 100 dev rows in each of its
four directions and no test row; the first lines of a direction are those with the smallest
ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sentencepiece

from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.corpus_rows import EN_MN, RU_MN
from tests.support.distill_cli import (
    CHECKPOINT_FILES,
    EvaluateWorkspace,
    OraclePredict,
    install_predictor,
    invoke,
    make_evaluate_workspace,
    run_ok,
)
from tests.support.planted_student import DIRECTIONS, Inputs, build_inputs
from tests.support.tiny_parallel import build_toy


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("cli_distill_evaluate_inputs"))


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


def manifest_of(workspace: EvaluateWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def result_of(workspace: EvaluateWorkspace) -> dict[str, Any]:
    return json.loads(workspace.result.read_text(encoding="utf-8"))


def expected_lines(
    workspace: EvaluateWorkspace,
    *,
    lines: int,
    scores: dict[str, tuple[str, str]] | None = None,
    mean: str = "100.00",
    low: str = "100.00",
    leaks: int = 0,
) -> list[str]:
    """What a run prints: every direction at the scores in `scores` (chrF and BLEU text),
    chrF 100.00 and BLEU 100.00 for the others, over `lines` dev lines."""
    given = {} if scores is None else scores
    return [
        *(
            f"direction {direction}: chrF {given.get(direction, ('100.00',) * 2)[0]} "
            f"BLEU {given.get(direction, ('100.00',) * 2)[1]} ({lines} lines)"
            for direction in DIRECTIONS
        ),
        f"chrF mean: {mean}",
        f"chrF min: {low}",
        f"tag leaks: {leaks}",
        "dev lines in training: 0",
        "tag placement: stated, not recorded in the checkpoint",
        f"result: {workspace.out_dir}/result.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]


def first_dev_targets(inputs: Inputs, direction: str, count: int) -> list[str]:
    """The targets of the `count` dev rows of `direction` with the smallest ids: what the
    oracle's hypotheses are, read from the corpus file itself."""
    src_lang, tgt_lang = direction.split("-")
    frame = pl.read_parquet(inputs.corpus)
    chosen = (
        frame.filter(
            (pl.col("split") == "dev")
            & (pl.col("src_lang") == src_lang)
            & (pl.col("tgt_lang") == tgt_lang)
        )
        .sort("id", "src", "tgt")
        .head(count)
    )
    return chosen["tgt"].to_list()


# --- help --------------------------------------------------------------------------------------


def test_help_names_every_option_and_marks_exactly_the_five_required_ones() -> None:
    result = invoke(["distill", "evaluate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--checkpoint",
        "--spm-model",
        "--corpus",
        "--tag-placement",
        "--out-dir",
        "--max-lines",
        "--threads",
        "--timeout",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 5
    assert "<src|tgt>" in result.stdout


def test_the_help_states_the_defaults_and_that_the_checkpoint_needs_no_manifest() -> (
    None
):
    text = " ".join(invoke(["distill", "evaluate", "--help"]).stdout.split())

    for option, default in (("--max-lines", "1000"), ("--threads", "2")):
        assert (
            f"[default: {default}]" in text.split(f"{option} ", 1)[1].split(" --", 1)[0]
        ), option
    assert "need not have a manifest" in text
    assert "baseline or a foreign model" in text


# --- what it prints and writes -----------------------------------------------------------------


def test_a_run_prints_the_scores_of_every_direction_then_the_summary_and_the_paths(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    result = run_ok(workspace.args("--max-lines", "5"))

    assert result.stdout.splitlines() == expected_lines(workspace, lines=5)


def test_the_default_evaluates_every_dev_line_up_to_a_thousand_of_each_direction(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    result = run_ok(workspace.args())

    assert result.stdout.splitlines() == expected_lines(workspace, lines=100)
    assert len(predictor.calls[0].source_tokens) == 400


def test_max_lines_limits_the_lines_of_each_direction(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    result = run_ok(workspace.args("--max-lines", "3"))

    assert result.stdout.splitlines() == expected_lines(workspace, lines=3)
    assert len(predictor.calls[0].source_tokens) == 12


def test_the_scores_are_those_of_the_hypotheses_a_direction_that_gets_nothing_scores_zero(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(
        monkeypatch, OraclePredict(inputs.spm, build_toy().dev, silent=["khk_Cyrl"])
    )

    result = run_ok(workspace.args("--max-lines", "5"))

    zero = ("0.00", "0.00")
    assert result.stdout.splitlines() == expected_lines(
        workspace,
        lines=5,
        scores={EN_MN: zero, RU_MN: zero},
        mean="50.00",
        low="0.00",
    )
    document = result_of(workspace)
    assert document["chrf_mean"] == 50.0
    assert document["chrf_min"] == 0.0
    assert document["directions"][EN_MN] == {"lines": 5, "chrf": 0.0, "bleu": 0.0}


def test_a_hypothesis_that_holds_a_direction_tag_is_a_leak_and_is_counted(
    workspace: EvaluateWorkspace, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_predictor(
        monkeypatch, OraclePredict(inputs.spm, build_toy().dev, leaks=[0, 1, 7])
    )

    result = run_ok(workspace.args("--max-lines", "5"))

    assert "tag leaks: 3" in result.stdout.splitlines()
    assert result_of(workspace)["tag_leaks"] == 3
    assert manifest_of(workspace).metrics["tag_leaks"] == 3.0


def test_result_json_holds_the_scores_and_what_they_were_made_from(
    workspace: EvaluateWorkspace, inputs: Inputs, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "5"))

    text = workspace.result.read_text(encoding="utf-8")
    document = json.loads(text)
    assert text.startswith('{\n  "bleu_signature": ')
    assert text.endswith("\n}\n")
    assert "\r" not in text
    assert list(document) == sorted(document)
    assert document["checkpoint"] == str(workspace.checkpoint)
    assert document["spm_sha256"] == sha256(inputs.spm)
    assert document["placement"] == "src"
    assert document["max_lines"] == 5
    assert document["chrf_mean"] == 100.0
    assert document["chrf_min"] == 100.0
    assert document["tag_leaks"] == 0
    assert document["chrf_signature"].startswith("nrefs:1|case:mixed|")
    assert document["bleu_signature"].startswith("nrefs:1|case:mixed|")
    assert document["directions"] == {
        direction: {"lines": 5, "chrf": 100.0, "bleu": 100.0}
        for direction in DIRECTIONS
    }
    assert sorted(document) == [
        "bleu_signature",
        "checkpoint",
        "chrf_mean",
        "chrf_min",
        "chrf_signature",
        "directions",
        "max_lines",
        "placement",
        "spm_sha256",
        "tag_leaks",
    ]


def test_the_checkpoint_is_recorded_as_given_and_handed_to_the_prediction_in_full(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace.root)

    run_ok(workspace.args("--max-lines", "2", checkpoint=Path("ckpt")))

    assert result_of(workspace)["checkpoint"] == "ckpt"
    assert predictor.calls[0].ckpt_dir == workspace.checkpoint


def test_the_hypothesis_files_hold_one_line_per_dev_line_in_the_order_of_the_ids(
    workspace: EvaluateWorkspace, inputs: Inputs, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "5"))

    for direction in DIRECTIONS:
        path = workspace.out_dir / f"hyp.{direction}.txt"
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text.splitlines() == first_dev_targets(inputs, direction, 5), direction


def test_the_manifest_vouches_for_the_inputs_the_checkpoint_the_outputs_and_the_numbers(
    workspace: EvaluateWorkspace, inputs: Inputs, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "5"))

    manifest = manifest_of(workspace)
    assert manifest.stage == "distill_evaluate"
    assert manifest.seed == -1
    assert manifest.status == {
        "evaluate": "ok",
        "tag_placement": "src",
        "tag_placement_check": "stated",
    }
    expected_inputs = [
        workspace.corpus,
        workspace.spm_model,
        *workspace.checkpoint_files,
    ]
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), sha256(path)) for path in expected_inputs
    ]
    assert [item.bytes for item in manifest.inputs[2:]] == [
        len(content) for content in CHECKPOINT_FILES.values()
    ]
    out = workspace.out_dir
    assert [(Path(item.path), item.rows) for item in manifest.outputs] == [
        (out / "result.json", None),
        *((out / f"hyp.{direction}.txt", 5) for direction in DIRECTIONS),
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
    assert manifest.metrics == {
        "chrf_mean": 100.0,
        "chrf_min": 100.0,
        "tag_leaks": 0.0,
        "dev_in_train": 0.0,
        **{f"lines.{direction}": 5.0 for direction in DIRECTIONS},
        **{f"chrf.{direction}": 100.0 for direction in DIRECTIONS},
        **{f"bleu.{direction}": 100.0 for direction in DIRECTIONS},
    }
    assert len(manifest.config_sha256) == 64


def test_the_manifest_config_digest_follows_the_options_that_decide_the_numbers(
    tmp_path: Path, inputs: Inputs, predictor: OraclePredict
) -> None:
    def digest_of(name: str, *extra: str, placement: str = "src") -> str:
        workspace = make_evaluate_workspace(tmp_path / name, inputs)
        run_ok(workspace.args(*extra, tag_placement=placement))
        return manifest_of(workspace).config_sha256

    base = digest_of("a", "--max-lines", "5")

    assert digest_of("b", "--max-lines", "5") == base
    assert digest_of("c", "--max-lines", "6") != base
    assert digest_of("d", "--max-lines", "5", placement="tgt") != base
    assert digest_of("e", "--max-lines", "5", "--threads", "3") == base


# --- what it asks the prediction for ---------------------------------------------------------


def test_the_prediction_is_one_call_with_the_tagged_dev_lines_of_every_direction(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "2"))

    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    assert call.ckpt_dir == workspace.checkpoint
    assert call.out_dir == workspace.out_dir
    assert call.stem == "dev"
    assert call.threads == 2
    assert call.timeout is None
    assert call.forced_prefix_tokens is None
    assert len(call.source_tokens) == 8
    tags = [line.split(" ")[0] for line in call.source_tokens]
    assert (
        tags
        == ["<2khk_Cyrl>"] * 2
        + ["<2eng_Latn>"] * 2
        + ["<2rus_Cyrl>"] * 2
        + ["<2khk_Cyrl>"] * 2
    )
    assert call.stale_files == tuple(
        workspace.out_dir / f"hyp.{direction}.txt" for direction in DIRECTIONS
    )


def test_with_the_target_placement_the_tag_is_forced_as_the_start_of_the_output(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    result = run_ok(workspace.args("--max-lines", "2", tag_placement="tgt"))

    call = predictor.calls[0]
    assert call.forced_prefix_tokens == (
        ("<2khk_Cyrl>",) * 2
        + ("<2eng_Latn>",) * 2
        + ("<2rus_Cyrl>",) * 2
        + ("<2khk_Cyrl>",) * 2
    )
    assert not any(line.startswith("<2") for line in call.source_tokens)
    assert result.stdout.splitlines() == expected_lines(workspace, lines=2)
    assert result_of(workspace)["placement"] == "tgt"
    assert manifest_of(workspace).status["tag_placement"] == "tgt"


def test_the_threads_and_the_timeout_reach_the_prediction(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "2", "--threads", "3", "--timeout", "60"))

    call = predictor.calls[0]
    assert call.threads == 3
    assert call.timeout == 60.0


# --- the rows it reads -------------------------------------------------------------------------


def corpus_with_test_rows(directory: Path, *, null_text: bool) -> Path:
    """The toy's training and dev rows and its 400 test rows, which have valid text or,
    with `null_text`, no text at all."""
    toy = build_toy()
    frame = corpus_frame([*toy.train, *toy.dev, *toy.test])
    if null_text:
        frame = frame.with_columns(
            *(
                pl.when(pl.col("split") == "test")
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col(column))
                .alias(column)
                for column in ("src", "tgt")
            )
        )
    path = directory / ("null_test.parquet" if null_text else "valid_test.parquet")
    write_corpus(frame, path)
    return path


@pytest.mark.parametrize("null_text", [False, True])
def test_a_test_row_is_never_read_whatever_it_holds(
    tmp_path: Path,
    inputs: Inputs,
    predictor: OraclePredict,
    null_text: bool,
) -> None:
    workspace = make_evaluate_workspace(tmp_path, inputs)
    corpus = corpus_with_test_rows(tmp_path, null_text=null_text)
    toy = build_toy()
    dev_sources = {row["src"] for row in toy.dev}
    test_only = {row["src"] for row in toy.test} - dev_sources
    assert len(test_only) > 100

    result = run_ok(workspace.args("--max-lines", "5", corpora=[corpus]))

    assert result.stdout.splitlines() == expected_lines(workspace, lines=5)
    processor = sentencepiece.SentencePieceProcessor(model_file=str(inputs.spm))
    sent = {
        processor.decode(line.split(" ")[1:])
        for line in predictor.calls[0].source_tokens
    }
    assert len(predictor.calls[0].source_tokens) == 20
    assert sent <= dev_sources


def test_a_dev_line_that_is_also_a_training_row_is_counted_and_recorded(
    tmp_path: Path, inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_evaluate_workspace(tmp_path, inputs)
    toy = build_toy()
    copied = {**toy.train[0], "split": "dev", "corpus": "toy-leak"}
    path = tmp_path / "leaky.parquet"
    write_corpus(corpus_frame([*toy.train, *toy.dev, copied]), path)
    install_predictor(monkeypatch, OraclePredict(inputs.spm, [*toy.dev, copied]))

    result = run_ok(workspace.args("--max-lines", "1000", corpora=[path]))

    lines = result.stdout.splitlines()
    assert "dev lines in training: 1" in lines
    assert f"direction {EN_MN}: chrF 100.00 BLEU 100.00 (101 lines)" in lines
    assert manifest_of(workspace).metrics["dev_in_train"] == 1.0


# --- a rerun ----------------------------------------------------------------------------------


def test_two_runs_of_the_same_inputs_give_byte_identical_results(
    tmp_path: Path, inputs: Inputs, predictor: OraclePredict
) -> None:
    first = make_evaluate_workspace(tmp_path / "first", inputs)
    second = make_evaluate_workspace(tmp_path / "second", inputs)
    # The record names the checkpoint as given: the same path for both runs.
    run_ok(first.args("--max-lines", "5"))
    run_ok(second.args("--max-lines", "5", checkpoint=first.checkpoint))

    first_bytes = first.result.read_bytes()
    assert second.result.read_bytes() == first_bytes
    for direction in DIRECTIONS:
        name = f"hyp.{direction}.txt"
        assert (second.out_dir / name).read_bytes() == (
            first.out_dir / name
        ).read_bytes()


def test_a_rerun_into_the_same_directory_replaces_the_result_and_the_manifest(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "5"))
    first = manifest_of(workspace)
    first_result = workspace.result.read_bytes()

    result = run_ok(workspace.args("--max-lines", "5"))

    second = manifest_of(workspace)
    assert second.created != first.created
    assert second.outputs == first.outputs
    assert workspace.result.read_bytes() == first_result
    assert result.stdout.splitlines() == expected_lines(workspace, lines=5)
    assert len(predictor.calls) == 2


def test_a_rerun_with_another_number_of_lines_rewrites_the_result(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    run_ok(workspace.args("--max-lines", "5"))

    run_ok(workspace.args("--max-lines", "3"))

    assert result_of(workspace)["max_lines"] == 3
    assert result_of(workspace)["directions"][EN_MN]["lines"] == 3
    assert (workspace.out_dir / f"hyp.{EN_MN}.txt").read_text().count("\n") == 3
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == sorted(
        ["manifest.json", "result.json", *(f"hyp.{d}.txt" for d in DIRECTIONS)]
    )
