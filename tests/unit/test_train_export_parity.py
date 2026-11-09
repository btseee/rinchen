"""`rinchen.train.export_parity`: what a strict parity run asks of Eole and of CTranslate2 and
how it turns their answers into a report.

No test starts Eole or loads a model: `tests.support.fake_parity_eole` stands in for both
(the real run is the integration test of Task 7). The numbers are worked out by hand: Eole
scores a sentence of two pieces -0.375 (-0.125 for each piece and for the end token), and the
library's log-probabilities are the same plus `deltas[i]` on the end token, so a delta `d`
is `d / 3` per token. What the run refuses is in `test_train_export_parity_refusals.py`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train.eole_run import predict_files
from rinchen.train.export_parity import (
    MAX_MEAN_DELTA,
    MIN_EXACT_SHARE,
    ParityReport,
    parity_config,
    parse_scored_line,
    strict_parity,
)
from tests.support.fake_ct2 import SPIECE
from tests.support.fake_parity_eole import TAG, ParityCase, make_case

NNBSP = chr(0x202F)


def run(
    case: ParityCase, *, threads: int = 2, timeout: float | None = None
) -> ParityReport:
    return strict_parity(
        case.checkpoint,
        case.export,
        case.sources,
        out_dir=case.out_dir,
        threads=threads,
        timeout=timeout,
        engine=case.ct2,
        processor=case.processor,
    )


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParityCase:
    return make_case(tmp_path, monkeypatch)


# --- the thresholds ----------------------------------------------------------------------------


def test_the_thresholds_are_the_master_plans() -> None:
    assert MIN_EXACT_SHARE == 0.95
    assert MAX_MEAN_DELTA == 1e-3


# --- parse_scored_line -------------------------------------------------------------------------


def test_a_scored_line_is_its_pieces_and_its_score() -> None:
    line = f"{SPIECE}a {SPIECE}b\t-1.3368405\t1.0"

    assert parse_scored_line(line) == ([f"{SPIECE}a", f"{SPIECE}b"], -1.3368405)


def test_a_scored_line_without_pieces_is_an_empty_hypothesis() -> None:
    assert parse_scored_line("\t-0.25\t1.0") == ([], -0.25)


def test_a_scored_line_splits_its_pieces_at_a_space_only() -> None:
    piece = f"{SPIECE}a{NNBSP}b"

    assert parse_scored_line(f"{piece} c\t-1.0\t1.0") == ([piece, "c"], -1.0)


@pytest.mark.parametrize(
    "line",
    ["a b\t-1.0", "a b\t-1.0\t1.0\textra", "a b", ""],
    ids=["two fields", "four fields", "one field", "nothing"],
)
def test_a_scored_line_of_another_number_of_fields_is_refused(line: str) -> None:
    with pytest.raises(UpstreamError, match="fields"):
        parse_scored_line(line)


@pytest.mark.parametrize("score", ["fast", "nan", "inf", "-inf", ""])
def test_a_score_that_is_no_finite_number_is_refused(score: str) -> None:
    with pytest.raises(UpstreamError, match="score"):
        parse_scored_line(f"a b\t{score}\t1.0")


def test_a_refused_scored_line_does_not_quote_the_pieces() -> None:
    with pytest.raises(UpstreamError) as raised:
        parse_scored_line("secretpiece\toops\t1.0")

    assert "secretpiece" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


# --- parity_config -----------------------------------------------------------------------------


def config_for(tmp_path: Path, placement: str) -> dict[str, object]:
    files = predict_files(tmp_path / "out", "parity")
    return parity_config(tmp_path / "ckpt", files, placement=placement)


def test_the_prediction_config_of_the_placement_src(tmp_path: Path) -> None:
    out = tmp_path / "out"

    assert config_for(tmp_path, "src") == {
        "model_path": (tmp_path / "ckpt").as_posix(),
        "src": (out / "parity.tokens.src").as_posix(),
        "output": (out / "parity.tokens.out").as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
        "max_length": 128,
        "max_length_ratio": 0,
        "with_score": True,
        "length_penalty": "none",
    }


def test_the_prediction_config_of_the_placement_tgt_forces_the_tag(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"

    assert config_for(tmp_path, "tgt") == {
        "model_path": (tmp_path / "ckpt").as_posix(),
        "src": (out / "parity.tokens.src").as_posix(),
        "tgt": (out / "parity.tokens.tgt").as_posix(),
        "tgt_file_prefix": True,
        "output": (out / "parity.tokens.out").as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
        "max_length": 128,
        "max_length_ratio": 0,
        "with_score": True,
        "length_penalty": "none",
    }


def test_a_placement_that_is_neither_src_nor_tgt_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="placement"):
        config_for(tmp_path, "middle")


@pytest.mark.parametrize("where", ["ckpt", "out"])
def test_a_dollar_in_a_path_of_the_config_is_refused(
    tmp_path: Path, where: str
) -> None:
    files = predict_files(
        tmp_path / ("out$HOME" if where == "out" else "out"), "parity"
    )
    checkpoint = tmp_path / ("ckpt$HOME" if where == "ckpt" else "ckpt")

    with pytest.raises(ConfigError, match=r"\$"):
        parity_config(checkpoint, files, placement="src")


@pytest.mark.parametrize("character", ["[", "]", "*", "?"])
def test_a_glob_character_in_the_checkpoint_path_is_refused(
    tmp_path: Path, character: str
) -> None:
    files = predict_files(tmp_path / "out", "parity")

    with pytest.raises(ConfigError, match="glob"):
        parity_config(tmp_path / f"ck{character}pt", files, placement="src")


def test_a_glob_character_in_the_output_directory_is_fine(tmp_path: Path) -> None:
    files = predict_files(tmp_path / "out[1]", "parity")

    config = parity_config(tmp_path / "ckpt", files, placement="src")

    assert config["output"] == (tmp_path / "out[1]" / "parity.tokens.out").as_posix()


# --- strict_parity: what it asks of Eole and of the library -------------------------------------


def test_strict_parity_runs_eole_once_on_the_cpu_with_the_gpu_hidden(
    case: ParityCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A device is visible to the process that runs the test: Eole must not see it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    run(case, threads=3, timeout=90.0)

    (launch,) = case.eole.launches
    assert launch.command == [
        "eole",
        "predict",
        "-config",
        str(case.out_dir / "parity.predict.yaml"),
    ]
    assert launch.what == "eole predict"
    assert launch.timeout == 90.0
    assert launch.log_path == case.out_dir / "parity.log"
    assert launch.env["CUDA_VISIBLE_DEVICES"] == ""
    assert launch.env["OMP_NUM_THREADS"] == "3"
    assert launch.env["MKL_NUM_THREADS"] == "3"
    assert launch.config["gpu_ranks"] == []
    assert launch.config["model_path"] == case.checkpoint.as_posix()


def test_the_source_lines_eole_reads_are_the_tag_and_the_pieces(
    case: ParityCase,
) -> None:
    run(case)

    (launch,) = case.eole.launches
    assert launch.source_lines == [
        f"{TAG} {SPIECE}w0",
        f"{TAG} {SPIECE}w1",
        f"{TAG} {SPIECE}w2",
    ]
    assert launch.prefix_lines == []
    assert "tgt" not in launch.config


def test_the_library_is_given_the_same_source_lines_and_greedy_options(
    case: ParityCase,
) -> None:
    run(case)

    (call,) = case.ct2.calls
    assert call.source == [
        [TAG, f"{SPIECE}w0"],
        [TAG, f"{SPIECE}w1"],
        [TAG, f"{SPIECE}w2"],
    ]
    assert call.target_prefix is None
    assert call.options == {
        "beam_size": 1,
        "max_decoding_length": 128,
        "return_end_token": False,
        "max_input_length": 0,
    }


def test_the_library_scores_the_tokens_eole_wrote_over_the_same_sources(
    case: ParityCase,
) -> None:
    run(case)

    (call,) = case.ct2.score_calls
    assert call.source == case.ct2.calls[0].source
    assert call.target == [tokens for tokens, _ in case.answers]
    assert call.options == {"max_input_length": 0}


def test_every_text_is_tokenised_with_the_export_s_own_model(case: ParityCase) -> None:
    run(case)

    assert case.processor.encoded == ["w0", "w1", "w2"]


def test_a_source_of_another_target_gets_the_tag_of_that_target(
    case: ParityCase,
) -> None:
    mixed = [case.sources[0], replace(case.sources[1], tgt_lang="eng_Latn")]
    case.eole.answers = case.answers[:2]
    case.ct2.script = {
        (TAG, f"{SPIECE}w0"): [(case.answers[0][0], -0.5)],
        ("<2eng_Latn>", f"{SPIECE}w1"): [(case.answers[1][0], -0.5)],
    }
    case.ct2.score_script = lambda line, target: [-0.125] * (len(target) + 1)

    strict_parity(
        case.checkpoint,
        case.export,
        mixed,
        out_dir=case.out_dir,
        engine=case.ct2,
        processor=case.processor,
    )

    (launch,) = case.eole.launches
    assert launch.source_lines == [f"{TAG} {SPIECE}w0", f"<2eng_Latn> {SPIECE}w1"]


# --- strict_parity: the report -------------------------------------------------------------------


def test_equal_translations_with_equal_scores_are_a_perfect_report(
    case: ParityCase,
) -> None:
    report = run(case)

    assert report == ParityReport(
        sentences=3,
        exact=3,
        exact_share=1.0,
        mean_abs_delta=0.0,
        max_abs_delta=0.0,
        quantization="float32",
        placement="src",
        gated=True,
        mismatches=(),
    )
    assert report.passes


def test_a_delta_of_one_millionth_per_token_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 3e-6 on a sum over three log-probabilities (two pieces and the end token).
    case = make_case(tmp_path, monkeypatch, count=2, deltas=[3e-6, 3e-6])

    report = run(case)

    assert report.mean_abs_delta == pytest.approx(1e-6, rel=1e-6)
    assert report.max_abs_delta == pytest.approx(1e-6, rel=1e-6)
    assert report.passes


def test_the_delta_is_per_token_with_the_end_token_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two pieces and the end token make three log-probabilities: 0.003 / 3, not 0.003 / 2.
    case = make_case(tmp_path, monkeypatch, count=1, deltas=[0.003])

    assert run(case).mean_abs_delta == pytest.approx(0.001, rel=1e-9)


def test_the_delta_is_the_absolute_difference_of_the_sums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=2, deltas=[-0.003, 0.006])

    report = run(case)

    assert report.mean_abs_delta == pytest.approx(0.0015, rel=1e-9)
    assert report.max_abs_delta == pytest.approx(0.002, rel=1e-9)


def test_the_mean_is_the_mean_and_not_the_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Per token 0, 0 and 0.003: the mean is 0.001 and the median 0.
    case = make_case(tmp_path, monkeypatch, count=3, deltas=[0.0, 0.0, 0.009])

    report = run(case)

    assert report.mean_abs_delta == pytest.approx(0.001, rel=1e-9)
    assert report.max_abs_delta == pytest.approx(0.003, rel=1e-9)


def test_the_end_token_s_log_probability_is_part_of_the_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Eole's score holds it: a sum of the two pieces only would differ by 0.125 over two tokens.
    case = make_case(tmp_path, monkeypatch, count=1)

    assert run(case).max_abs_delta == 0.0


@pytest.mark.parametrize(
    ("different", "passes"),
    [(10, True), (11, False)],
    ids=["190 of 200 is 0.95 exactly", "189 of 200"],
)
def test_the_exact_share_must_reach_95_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    different: int,
    passes: bool,
) -> None:
    case = make_case(tmp_path, monkeypatch, count=200, differ=range(different))

    report = run(case)

    assert report.exact == 200 - different
    assert report.exact_share == (200 - different) / 200
    assert report.passes is passes


def test_nineteen_of_twenty_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=20, differ=[7])

    assert run(case).passes


def test_a_mean_delta_of_1_5_thousandths_fails_a_float32_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=2, deltas=[0.0045, 0.0045])

    report = run(case)

    assert report.exact_share == 1.0
    assert report.mean_abs_delta == pytest.approx(0.0015, rel=1e-9)
    assert report.gated
    assert not report.passes


def test_a_mean_delta_of_exactly_one_thousandth_passes_a_float32_export() -> None:
    report = ParityReport(10, 10, 1.0, 1e-3, 2e-3, "float32", "src", True, ())

    assert report.passes


def test_a_mean_delta_just_above_one_thousandth_fails_a_float32_export() -> None:
    report = ParityReport(10, 10, 1.0, 1.0000001e-3, 2e-3, "float32", "src", True, ())

    assert not report.passes


@pytest.mark.parametrize("quantization", ["int8", "int8_float16", "float16"])
def test_the_same_numbers_pass_a_quantised_export_that_is_not_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: str
) -> None:
    case = make_case(
        tmp_path,
        monkeypatch,
        count=2,
        quantization=quantization,
        deltas=[0.0045, 0.0045],
    )

    report = run(case)

    assert report.quantization == quantization
    assert report.mean_abs_delta == pytest.approx(0.0015, rel=1e-9)
    assert not report.gated
    assert report.passes


def test_a_quantised_export_still_needs_the_exact_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(
        tmp_path, monkeypatch, count=200, quantization="int8", differ=range(11)
    )

    report = run(case)

    assert not report.gated
    assert not report.passes


def test_the_report_carries_the_placement_of_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=1, placement="tgt")

    assert run(case).placement == "tgt"


def test_the_first_ten_mismatches_are_kept_as_token_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=30, differ=range(3, 23))

    report = run(case)

    assert report.exact == 10
    assert len(report.mismatches) == 10
    assert report.mismatches[0] == (
        f"{SPIECE}h3x0 {SPIECE}h3x1",
        f"{SPIECE}h3x0 {SPIECE}other",
    )
    assert report.mismatches[-1] == (
        f"{SPIECE}h12x0 {SPIECE}h12x1",
        f"{SPIECE}h12x0 {SPIECE}other",
    )


def test_no_mismatch_is_kept_when_everything_is_equal(case: ParityCase) -> None:
    assert run(case).mismatches == ()


def test_the_mismatches_of_a_report_are_lists_of_two_lines_in_its_dict() -> None:
    report = ParityReport(
        2, 1, 0.5, 0.0, 0.0, "float32", "src", True, (("a b", "a c"),)
    )

    assert report.to_dict()["mismatches"] == [["a b", "a c"]]


def test_the_report_is_json_ready(case: ParityCase) -> None:
    document = run(case).to_dict()

    assert document == {
        "sentences": 3,
        "exact": 3,
        "exact_share": 1.0,
        "mean_abs_delta": 0.0,
        "max_abs_delta": 0.0,
        "quantization": "float32",
        "placement": "src",
        "gated": True,
        "mismatches": [],
    }


# --- strict_parity: the end token and the forced prefix ------------------------------------------


def test_the_end_token_is_not_part_of_the_pieces_that_are_compared(
    case: ParityCase,
) -> None:
    # The library's hypotheses carry the end token only when asked for it.
    assert run(case).exact == 3
    assert case.ct2.calls[0].options["return_end_token"] is False


def test_the_placement_tgt_forces_the_tag_as_the_prefix_of_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=2, placement="tgt")

    report = run(case)

    (launch,) = case.eole.launches
    assert launch.source_lines == [f"{SPIECE}w0", f"{SPIECE}w1"]
    assert launch.prefix_lines == [TAG, TAG]
    assert launch.config["tgt_file_prefix"] is True
    (call,) = case.ct2.calls
    assert call.source == [[f"{SPIECE}w0"], [f"{SPIECE}w1"]]
    assert call.target_prefix == [[TAG], [TAG]]
    assert report.exact == 2


def test_the_placement_tgt_scores_eole_s_line_with_its_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=1, placement="tgt")

    run(case)

    (call,) = case.ct2.score_calls
    assert call.target == [[TAG, f"{SPIECE}h0x0", f"{SPIECE}h0x1"]]


def test_the_placement_tgt_drops_the_log_probability_of_the_forced_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tag and seven pieces: with the end token nine log-probabilities, the first -0.09.
    # Kept, they differ from Eole's score by 0.09, which is 0.01 per token; dropped, by
    # nothing.
    case = make_case(
        tmp_path,
        monkeypatch,
        count=1,
        placement="tgt",
        piece_count=7,
        prefix_log_prob=-0.09,
    )
    eole_tokens, eole_score = case.answers[0]
    (scoring,) = case.ct2.score_batch([[f"{SPIECE}w0"]], [eole_tokens])
    kept = abs(sum(scoring.log_probs) - eole_score) / len(scoring.log_probs)

    report = run(case)

    assert len(scoring.log_probs) == 9
    assert kept == pytest.approx(0.01)
    assert report.mean_abs_delta == 0.0
    assert report.max_abs_delta == 0.0


def test_the_placement_tgt_divides_by_the_log_probabilities_it_compares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two pieces after the tag: the end token and two pieces are three log-probabilities
    # once the tag's is dropped (not four).
    case = make_case(tmp_path, monkeypatch, count=1, placement="tgt", deltas=[0.003])

    assert run(case).mean_abs_delta == pytest.approx(0.001, rel=1e-9)


def test_the_mismatches_of_the_placement_tgt_are_shown_without_the_forced_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=2, placement="tgt", differ=[1])

    report = run(case)

    assert report.exact == 1
    assert report.mismatches == (
        (f"{SPIECE}h1x0 {SPIECE}h1x1", f"{SPIECE}h1x0 {SPIECE}other"),
    )


def test_an_eole_line_that_does_not_start_with_the_forced_tag_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=1, placement="tgt")
    case.eole.answers = [([f"{SPIECE}h0x0", f"{SPIECE}h0x1"], -0.375)]

    with pytest.raises(UpstreamError, match="line 1") as raised:
        run(case)

    assert "forced tag" in str(raised.value)
    assert case.ct2.score_calls == []


def test_a_hypothesis_of_the_library_that_does_not_start_with_the_prefix_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(tmp_path, monkeypatch, count=1, placement="tgt")
    case.ct2.keeps_prefix = False

    with pytest.raises(UpstreamError, match="prefix"):
        run(case)
