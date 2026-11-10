"""`rinchen eval run` over a hypotheses file and over a model (D13, Task 7): what it prints and
writes for a run that measured, with the golden suite and the golden outputs of `alpha`.

The golden figures are Task 2's and Task 6's (derived with the pure-Python reference of the design
spike): the intervals are the DOCUMENT unit, the golden suite having five documents. The model is
a fake engine behind the seam of `rinchen.cli.eval_systems`; what the command refuses is in
`test_cli_eval_run_refusals.py`, the manifest and the files it replaces in
`test_cli_eval_run_manifest.py`, the inputs of the robustness and of the scores in
`test_cli_eval_run_inputs.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.eval.outputs import Output
from rinchen.eval.result import result_from_bytes, run_from_bytes
from rinchen.infer.engine_types import DecodeOptions, LoadOptions, Row
from tests.support.eval_cli import (
    ALPHA_LINES,
    FIRST,
    SECOND,
    THIRD,
    EvalWorkspace,
    alpha_texts,
    install,
    invoke,
    make_workspace,
    read_json,
    run_dir_of,
    three_language_suite,
)
from tests.support.eval_fakes import (
    GOLDEN_OUTPUTS_FILES,
    FakeCandidateTranslator,
    answering,
    golden_outputs,
    golden_requests,
)
from tests.support.eval_golden import GOLDEN_ENG, GOLDEN_KHK
from tests.support.eval_result_documents import GOLDEN_RESULT_FILE
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

GOLDEN_KHK_TEXTS = GOLDEN_KHK
GOLDEN_ENG_TEXTS = GOLDEN_ENG


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def listing(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


# --- a hypotheses file -----------------------------------------------------------------------------


def test_a_hypotheses_run_prints_the_golden_lines_and_the_two_paths(
    workspace: EvalWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        *ALPHA_LINES,
        f"run: {workspace.run_dir / 'run.json'}",
        f"manifest: {workspace.run_dir / 'manifest.json'}",
    ]


def test_a_hypotheses_run_writes_exactly_the_files_of_a_run(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    assert listing(workspace.run_dir) == [
        f"{SECOND}/result.json",
        "hypotheses.jsonl",
        f"{FIRST}/result.json",
        "manifest.json",
        "run.json",
    ]


def test_the_result_of_the_first_direction_is_the_golden_result_byte_for_byte(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    written = (workspace.run_dir / FIRST / "result.json").read_bytes()

    assert written == GOLDEN_RESULT_FILE.read_bytes()


def test_the_outputs_file_of_a_run_is_the_canonical_rewrite_of_what_it_used(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    written = (workspace.run_dir / "hypotheses.jsonl").read_bytes()

    # The golden files are canonical, so the rewrite is the file itself.
    assert written == GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()


def test_the_results_name_the_digest_of_the_outputs_file_the_run_wrote(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    written = sha((workspace.run_dir / "hypotheses.jsonl").read_bytes())

    for direction in (FIRST, SECOND):
        result = result_from_bytes(
            (workspace.run_dir / direction / "result.json").read_bytes()
        )
        assert result.provenance.outputs_sha256 == written
        assert result.provenance.system.outputs_sha256 == sha(
            workspace.alpha.read_bytes()
        )
        assert (result.system, result.trained, result.manifest_sha) == (
            "alpha",
            True,
            None,
        )
        assert result.robustness is None
        assert result.metrics["comet22"] is None
        assert "comet22" in result.unmeasured


def test_a_file_that_is_not_canonical_is_rewritten_and_the_results_name_the_digest_of_the_rewrite(
    workspace: EvalWorkspace,
) -> None:
    # The same outputs, the lines in the reverse order: the file the operator gave is not the file
    # the run writes and re-scores from, and the results must say which is which.
    lines = workspace.alpha.read_text("utf-8").splitlines()
    workspace.alpha.write_text("\n".join(reversed(lines)) + "\n", "utf-8")
    given = sha(workspace.alpha.read_bytes())

    assert invoke(workspace.hypotheses_args()).exit_code == 0

    written = (workspace.run_dir / "hypotheses.jsonl").read_bytes()
    assert written == GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()
    assert sha(written) != given
    result = result_from_bytes((workspace.run_dir / FIRST / "result.json").read_bytes())
    assert result.provenance.outputs_sha256 == sha(written)
    assert result.provenance.system.outputs_sha256 == given
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert record.files["hypotheses.jsonl"] == sha(written)
    assert record.system.outputs_sha256 == given


def test_run_json_lists_every_direction_and_the_digest_of_every_file(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    run_dir = workspace.run_dir

    record = run_from_bytes((run_dir / "run.json").read_bytes())

    assert (record.suite.name, record.purpose) == ("golden", "report")
    assert record.suite.sha256 == sha(workspace.suite.read_bytes())
    assert [
        (d.direction, d.state, d.reason, d.rows, d.result) for d in record.directions
    ] == [
        (FIRST, "run", None, 10, f"{FIRST}/result.json"),
        (SECOND, "run", None, 10, f"{SECOND}/result.json"),
    ]
    assert dict(record.files) == {
        name: sha((run_dir / name).read_bytes())
        for name in (
            "hypotheses.jsonl",
            f"{FIRST}/result.json",
            f"{SECOND}/result.json",
        )
    }
    assert (record.settings.resamples, record.settings.seed, record.settings.alpha) == (
        1000,
        20261110,
        0.05,
    )
    assert record.settings.selection is None
    assert (record.system.kind, record.system.name) == ("hypotheses", "alpha")
    assert record.robustness_source is None
    assert dict(record.segment_scores) == {}


def test_two_runs_give_equal_run_json_results_and_hypotheses_and_manifests_that_differ_only_in_created(
    workspace: EvalWorkspace,
) -> None:
    first = workspace.out_dir / "one"
    second = workspace.out_dir / "two"
    assert invoke(workspace.hypotheses_args(out_dir=first)).exit_code == 0
    assert invoke(workspace.hypotheses_args(out_dir=second)).exit_code == 0
    one, two = run_dir_of(first, "alpha"), run_dir_of(second, "alpha")

    for name in (
        "run.json",
        "hypotheses.jsonl",
        f"{FIRST}/result.json",
        f"{SECOND}/result.json",
    ):
        assert (one / name).read_bytes() == (two / name).read_bytes(), name

    def stripped(directory: Path) -> dict[str, Any]:
        manifest = read_json(directory / "manifest.json")
        manifest.pop("created")
        for entry in manifest["outputs"]:
            entry["path"] = Path(entry["path"]).name
        return manifest

    assert stripped(one) == stripped(two)


def test_resamples_and_seed_are_recorded_and_move_the_intervals(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    default = read_json(workspace.run_dir / FIRST / "result.json")

    assert (
        invoke(workspace.hypotheses_args("--resamples", "500", "--seed", "7")).exit_code
        == 0
    )
    moved = read_json(workspace.run_dir / FIRST / "result.json")
    record = read_json(workspace.run_dir / "run.json")

    assert (moved["bootstrap"]["resamples"], moved["bootstrap"]["seed"]) == (500, 7)
    assert moved["bootstrap"]["alpha"] == 0.05
    assert (record["settings"]["resamples"], record["settings"]["seed"]) == (500, 7)
    assert moved["metrics"] == default["metrics"], (
        "the point estimates are not resampled"
    )
    # Five documents allow only 5**5 resamples, so chrF++ ends can coincide across seeds; the
    # BLEU interval of the golden suite does move (36.34 against 38.62 at the low end).
    assert moved["ci95"]["bleu"] != default["ci95"]["bleu"]


def test_the_bootstrap_settings_of_the_metrics_config_are_the_defaults(
    workspace: EvalWorkspace,
) -> None:
    workspace.metrics.write_text(
        workspace.metrics.read_text("utf-8").replace(
            "resamples: 1000", "resamples: 300"
        ),
        "utf-8",
    )

    assert (
        invoke(
            workspace.hypotheses_args("--metrics-config", str(workspace.metrics))
        ).exit_code
        == 0
    )

    bootstrap = read_json(workspace.run_dir / FIRST / "result.json")["bootstrap"]
    assert (bootstrap["resamples"], bootstrap["seed"]) == (300, 20261110)


def test_a_run_of_another_system_name_goes_to_its_own_directory(
    workspace: EvalWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args("--system", "a-file"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1] == "system: a-file"
    assert (run_dir_of(workspace.out_dir, "a-file") / "run.json").is_file()
    assert not workspace.run_dir.exists()


def test_a_file_that_holds_one_direction_runs_it_and_lists_the_other_not_in_the_file(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    from tests.support.eval_cli import one_direction_file

    only = one_direction_file(tmp_path / "only.jsonl")

    result = invoke(workspace.hypotheses_args(hypotheses=only))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3:5] == [
        ALPHA_LINES[3],
        f"{SECOND}: n/a (not_in_file)",
    ]
    record = run_from_bytes(
        (run_dir_of(workspace.out_dir, "only") / "run.json").read_bytes()
    )
    states = {d.direction: (d.state, d.result) for d in record.directions}
    assert states == {
        FIRST: ("run", f"{FIRST}/result.json"),
        SECOND: ("not_in_file", None),
    }
    assert record.directions[1].reason
    assert not (run_dir_of(workspace.out_dir, "only") / SECOND).exists()


def test_a_status_that_is_not_ok_is_counted_beside_the_ok_ones_in_the_line(
    workspace: EvalWorkspace,
) -> None:
    lines = workspace.alpha.read_text("utf-8").splitlines()
    marked = [
        line.replace('"status": "ok"', '"status": "truncated"')
        if number in (1, 4)
        else line
        for number, line in enumerate(lines)
    ]
    workspace.alpha.write_text("\n".join(marked) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert f"{FIRST}: chrF++ 73.08 [54.91, 85.48] (10 segments, 8 ok, 2 truncated)" in (
        result.stdout.splitlines()
    )
    written = read_json(workspace.run_dir / FIRST / "result.json")
    assert written["statuses"] == {"ok": 8, "truncated": 2}


def test_a_direction_of_one_segment_has_no_interval_and_the_line_says_so(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    document = json.loads(workspace.suite.read_text("utf-8"))
    document["rows"] = document["rows"][:1]
    one_row = tmp_path / "one_row.json"
    one_row.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
    lines = workspace.alpha.read_text("utf-8").splitlines()
    kept = [line for line in lines if '"id": "g01"' in line]
    only = tmp_path / "only.jsonl"
    only.write_text("\n".join(kept) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args(hypotheses=only, suite=one_row))

    assert result.exit_code == 0, result.output
    assert f"{FIRST}: chrF++ 100.00 n/a (1 segment, 1 ok)" in result.stdout.splitlines()
    written = read_json(
        run_dir_of(workspace.out_dir, "only", "golden") / FIRST / "result.json"
    )
    assert written["ci95"]["chrf_pp"] is None
    assert written["bootstrap"]["units"] == 1


def test_a_run_over_a_dev_suite_is_made_and_says_selection(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    document = json.loads(workspace.suite.read_text("utf-8"))
    document["split"] = "dev"
    dev = tmp_path / "dev_suite.json"
    dev.write_text(json.dumps(document, ensure_ascii=False), "utf-8")

    result = invoke(workspace.hypotheses_args(suite=dev))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[2] == "purpose: selection"
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert record.purpose == "selection"
    assert (
        read_json(workspace.run_dir / "manifest.json")["status"]["purpose"]
        == "selection"
    )


# --- the selection of directions ------------------------------------------------------------------


def test_a_selected_direction_is_run_and_the_other_is_listed_not_selected(
    workspace: EvalWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args("--direction", FIRST))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3:5] == [
        ALPHA_LINES[3],
        f"{SECOND}: n/a (not_selected)",
    ]
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert record.settings.selection == (FIRST,)
    assert [(d.direction, d.state) for d in record.directions] == [
        (FIRST, "run"),
        (SECOND, "not_selected"),
    ]
    assert record.directions[1].reason
    assert not (workspace.run_dir / SECOND).exists()
    lines = (workspace.run_dir / "hypotheses.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 10
    assert all(f'"direction": "{FIRST}"' in line for line in lines)


def test_two_selected_directions_are_kept_in_the_order_of_the_suite_in_the_run(
    workspace: EvalWorkspace,
) -> None:
    result = invoke(
        workspace.hypotheses_args("--direction", SECOND, "--direction", FIRST)
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3:5] == [ALPHA_LINES[3], ALPHA_LINES[4]]
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert record.settings.selection == (SECOND, FIRST)


# --- a model --------------------------------------------------------------------------------------


def test_the_engine_gets_one_call_of_the_runnable_requests_in_order_with_the_decode_flags(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    result = invoke(
        workspace.model_args(
            "--beam-size",
            "2",
            "--max-decoding-length",
            "64",
            "--script-mask",
            "always",
            "--threads",
            "3",
        )
    )

    assert result.exit_code == 0, result.output
    (call,) = engine.calls
    assert call.rows == [Row(r.text, r.src_lang, r.tgt_lang) for r in golden_requests()]
    assert call.options == DecodeOptions(
        beam_size=2, max_decoding_length=64, script_mask="always"
    )
    (opened,) = seams.opens
    assert opened.model == workspace.export.directory
    assert opened.load == LoadOptions(device="cpu", intra_threads=3)
    assert engine.closed == 1
    assert seams.events == ["open", "translate", "close"]


def test_a_model_run_with_no_flag_uses_the_defaults_of_d12(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    assert invoke(workspace.model_args()).exit_code == 0

    assert engine.calls[0].options == DecodeOptions(
        beam_size=4, max_decoding_length=256, script_mask="retry"
    )
    assert seams.opens[0].load == LoadOptions(device="cpu", intra_threads=1)


def test_the_model_can_come_from_the_environment(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    result = invoke(
        workspace.base(), env={"RINCHEN_MODEL": str(workspace.export.directory)}
    )

    assert result.exit_code == 0, result.output
    assert seams.opens[0].model == workspace.export.directory


def test_a_model_run_names_its_system_with_the_model_the_digests_and_the_options(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--beam-size", "2"))

    assert result.exit_code == 0, result.output
    name = workspace.export.record.name
    run_dir = run_dir_of(workspace.out_dir, name)
    record = run_from_bytes((run_dir / "run.json").read_bytes())
    system = record.system
    assert (system.kind, system.name) == ("rinchen", name)
    assert system.model_sha256 == workspace.export.sha256["model.bin"]
    assert system.manifest_sha256 == workspace.export.manifest_sha256
    assert dict(system.options) == {
        "beam_size": 2,
        "max_decoding_length": 256,
        "script_mask": "retry",
        "device": "cpu",
        "threads": 1,
    }
    written = read_json(run_dir / FIRST / "result.json")
    assert written["manifest_sha"] == workspace.export.manifest_sha256
    assert (
        written["provenance"]["system"]["model_sha256"]
        == workspace.export.sha256["model.bin"]
    )
    assert written["trained"] is True
    assert result.stdout.splitlines()[1] == f"system: {name}"


def test_a_system_name_overrides_the_name_of_the_model(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--system", "toy"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1] == "system: toy"
    assert (run_dir_of(workspace.out_dir, "toy") / "run.json").is_file()


def test_the_outputs_of_a_model_run_fed_back_as_a_file_give_the_same_results(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args("--system", "toy")).exit_code == 0
    model_dir = run_dir_of(workspace.out_dir, "toy")

    result = invoke(
        workspace.hypotheses_args(
            "--system", "toy-file", hypotheses=model_dir / "hypotheses.jsonl"
        )
    )

    assert result.exit_code == 0, result.output
    file_dir = run_dir_of(workspace.out_dir, "toy-file")
    assert (file_dir / "hypotheses.jsonl").read_bytes() == (
        model_dir / "hypotheses.jsonl"
    ).read_bytes()
    for direction in (FIRST, SECOND):
        model_result = read_json(model_dir / direction / "result.json")
        file_result = read_json(file_dir / direction / "result.json")
        for key in (
            "metrics",
            "ci95",
            "statuses",
            "n_segments",
            "bootstrap",
            "bleu_signature",
        ):
            assert file_result[key] == model_result[key], key


def test_a_status_the_engine_flags_is_scored_as_it_is_and_counted(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [
        Output(output.text, "truncated" if number in (2, 5) else output.status)
        for number, output in enumerate(golden_outputs("alpha").values())
    ]
    install(monkeypatch, workspace.engine(answer=answering(outputs)))

    result = invoke(workspace.model_args("--system", "toy"))

    assert result.exit_code == 0, result.output
    assert f"{FIRST}: chrF++ 73.08 [54.91, 85.48] (10 segments, 8 ok, 2 truncated)" in (
        result.stdout.splitlines()
    )


def test_a_direction_the_model_was_not_trained_for_is_listed_and_neither_run_nor_scored(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = workspace.engine()
    install(monkeypatch, engine)
    suite = three_language_suite(tmp_path / "three.json")

    result = invoke(workspace.model_args("--system", "toy", suite=suite))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3:6] == [
        ALPHA_LINES[3],
        ALPHA_LINES[4],
        f"{THIRD}: n/a (not_trained)",
    ]
    (call,) = engine.calls
    assert len(call.rows) == 20
    assert all(row.tgt != "rus_Cyrl" for row in call.rows)
    run_dir = run_dir_of(workspace.out_dir, "toy")
    record = run_from_bytes((run_dir / "run.json").read_bytes())
    third = record.directions[2]
    assert (third.direction, third.state, third.rows, third.result) == (
        THIRD,
        "not_trained",
        10,
        None,
    )
    assert third.reason
    assert not (run_dir / THIRD).exists()
    assert THIRD not in (run_dir / "hypotheses.jsonl").read_text("utf-8")


def test_a_selection_keeps_the_engine_from_translating_the_other_direction(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine(
        answer=answering(list(golden_outputs("alpha").values())[:10])
    )
    install(monkeypatch, engine)

    result = invoke(workspace.model_args("--direction", FIRST))

    assert result.exit_code == 0, result.output
    (call,) = engine.calls
    assert len(call.rows) == 10
    assert {(row.src, row.tgt) for row in call.rows} == {("khk_Cyrl", "eng_Latn")}


# --- a baseline -----------------------------------------------------------------------------------


def alpha_translator() -> FakeCandidateTranslator:
    """A translator that says what `alpha` wrote for the direction it is asked for."""
    return FakeCandidateTranslator(
        answer=lambda texts, src, tgt: [[text] for text in alpha_texts(f"{src}-{tgt}")]
    )


def test_a_ct2_baseline_runs_every_direction_with_one_call_each_and_records_its_settings(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    translator = alpha_translator()
    seams = install(monkeypatch, None, translator)

    result = invoke(workspace.generator_args("nllb_toy"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[:5] == [
        "suite: golden",
        "system: nllb_toy",
        "purpose: report",
        ALPHA_LINES[3],
        ALPHA_LINES[4],
    ]
    assert [(c.src, c.tgt) for c in translator.calls] == [
        ("khk_Cyrl", "eng_Latn"),
        ("eng_Latn", "khk_Cyrl"),
    ]
    assert [c.texts for c in translator.calls] == [
        list(GOLDEN_KHK_TEXTS),
        list(GOLDEN_ENG_TEXTS),
    ]
    (built,) = seams.builds
    assert (built.n, built.beam_size) == (1, 4)
    run_dir = run_dir_of(workspace.out_dir, "nllb_toy")
    record = run_from_bytes((run_dir / "run.json").read_bytes())
    assert (record.system.kind, record.system.name) == ("baseline", "nllb_toy")
    assert (record.system.model_sha256, record.system.manifest_sha256) == (None, None)
    assert dict(record.system.options) == {
        "generator": "nllb_toy",
        "kind": "ct2",
        "family": "nllb",
        "device": "cpu",
        "beam_size": 4,
        "batch_size": 16,
        "n": 1,
        "max_tokens": 128,
    }
    written = read_json(run_dir / FIRST / "result.json")
    assert (written["trained"], written["manifest_sha"]) == (True, None)
    assert written["provenance"]["system"]["kind"] == "baseline"


def test_a_server_baseline_runs_greedy_with_one_candidate_whatever_the_configuration_says(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, alpha_translator())

    result = invoke(workspace.generator_args("vllm_toy", "--system", "hy-mt2"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1] == "system: hy-mt2"
    (built,) = seams.builds
    assert (built.n, built.temperature, built.top_p) == (1, 0.0, 1.0)
    record = run_from_bytes(
        (run_dir_of(workspace.out_dir, "hy-mt2") / "run.json").read_bytes()
    )
    assert dict(record.system.options) == {
        "generator": "vllm_toy",
        "kind": "vllm",
        "model": "Hy-MT2-7B",
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 200,
    }


def test_a_server_cannot_run_a_pair_that_is_not_khalkha_to_or_from_a_partner_a_ct2_model_can(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = three_language_suite(tmp_path / "three.json")
    install(monkeypatch, None, FakeCandidateTranslator())

    ct2 = invoke(workspace.generator_args("nllb_toy", suite=suite))
    server = invoke(workspace.generator_args("vllm_toy", suite=suite))

    assert ct2.exit_code == 0, ct2.output
    assert ct2.stdout.splitlines()[5].startswith(f"{THIRD}: chrF++ ")
    assert server.exit_code == 0, server.output
    assert server.stdout.splitlines()[5] == f"{THIRD}: n/a (unsupported_language)"
    record = run_from_bytes(
        (run_dir_of(workspace.out_dir, "vllm_toy") / "run.json").read_bytes()
    )
    assert record.directions[2].state == "unsupported_language"
    assert record.directions[2].reason
    assert not (run_dir_of(workspace.out_dir, "vllm_toy") / THIRD).exists()


def test_a_selection_keeps_a_baseline_from_translating_the_other_direction(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    translator = alpha_translator()
    install(monkeypatch, None, translator)

    result = invoke(workspace.generator_args("vllm_toy", "--direction", SECOND))

    assert result.exit_code == 0, result.output
    assert [(c.src, c.tgt) for c in translator.calls] == [("eng_Latn", "khk_Cyrl")]
    assert result.stdout.splitlines()[3:5] == [
        f"{FIRST}: n/a (not_selected)",
        ALPHA_LINES[4],
    ]
    run_dir = run_dir_of(workspace.out_dir, "vllm_toy")
    assert not (run_dir / FIRST).exists()
    assert run_from_bytes((run_dir / "run.json").read_bytes()).settings.selection == (
        SECOND,
    )


def test_a_candidate_list_that_is_empty_is_no_output_and_is_counted_in_the_line(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def answer(texts: Any, src: str, tgt: str) -> list[list[str]]:
        lists = [[text] for text in alpha_texts(f"{src}-{tgt}")]
        lists[3] = []
        return lists

    install(monkeypatch, None, FakeCandidateTranslator(answer=answer))

    result = invoke(workspace.generator_args("vllm_toy"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3].endswith("(10 segments, 9 ok, 1 no_output)")
    written = read_json(
        run_dir_of(workspace.out_dir, "vllm_toy") / FIRST / "result.json"
    )
    assert written["statuses"] == {"no_output": 1, "ok": 9}


# --- a run in which nothing ran ------------------------------------------------------------------------


def test_a_run_in_which_no_direction_ran_is_still_written_with_every_state(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    empty = tmp_path / "nothing.jsonl"
    empty.write_bytes(b"")

    result = invoke(workspace.hypotheses_args(hypotheses=empty))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[3:5] == [
        f"{FIRST}: n/a (not_in_file)",
        f"{SECOND}: n/a (not_in_file)",
    ]
    run_dir = run_dir_of(workspace.out_dir, "nothing")
    assert sorted(path.name for path in run_dir.rglob("*") if path.is_file()) == [
        "hypotheses.jsonl",
        "manifest.json",
        "run.json",
    ]
    assert (run_dir / "hypotheses.jsonl").read_bytes() == b""
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["status"]["unit"] == "none"
    assert manifest["metrics"] == {}


# --- the help ----------------------------------------------------------------------------------------


def test_the_help_of_eval_lists_suite_robustness_and_run() -> None:
    result = invoke(["eval", "--help"])

    assert result.exit_code == 0
    for command in ("suite", "robustness", "run"):
        assert command in result.stdout


def test_the_help_of_run_lists_every_option_and_marks_the_required_ones() -> None:
    result = invoke(["eval", "run", "--help"])

    assert result.exit_code == 0
    for option in (
        "--suite",
        "--out-dir",
        "--model",
        "--generator",
        "--generators-config",
        "--hypotheses",
        "--system",
        "--direction",
        "--beam-size",
        "--max-decoding-length",
        "--script-mask",
        "--device",
        "--threads",
        "--robustness",
        "--segment-scores",
        "--resamples",
        "--seed",
        "--metrics-config",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 2


def test_the_help_of_run_wraps_its_description_by_width_not_by_the_docstring() -> None:
    result = invoke(["eval", "run", "--help"])

    above_the_options = result.stdout.split("Options", 1)[0].splitlines()[:-1]
    paragraphs = "\n".join(line.rstrip() for line in above_the_options).split("\n\n")
    description = [p for p in paragraphs if "Usage:" not in p and p.strip()]
    assert description
    for paragraph in description:
        lines = paragraph.splitlines()
        assert all(len(line) >= 60 for line in lines[:-1]), paragraph
