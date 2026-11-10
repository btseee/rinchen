"""`rinchen eval robustness`: what it prints and writes, which requests it asks the engine
for and how (D12, Task 7). What it refuses and the order of its work are in
`test_cli_eval_robustness_refusals.py`, what its manifest says in
`test_cli_eval_robustness_manifest.py`.

No test loads a model: the command reaches the engine and the converter through `open_engine`
and `load_converter`, which `tests.support.robust_cli.install` replaces. The suite and the
hypotheses are the golden files (`tests/fixtures/robustness`), whose scores are derived by hand
(`tests.support.robust_golden`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result

from rinchen.cli.eval_robustness_output import result_lines
from rinchen.eval.robustness import RobustnessSystem, compute_robustness
from rinchen.infer.engine_types import DecodeOptions, LoadOptions, Row
from tests.support.engines import FakeConverter
from tests.support.robust_cli import (
    FakeRobustEngine,
    RobustWorkspace,
    Seams,
    echo_answers,
    install,
    invoke,
    make_workspace,
    sha256_of,
)
from tests.support.robust_golden import (
    GOLDEN,
    GOLDEN_COVERAGE,
    GOLDEN_HYPOTHESES_SHA256,
    golden_requests,
    golden_suite,
    suite_of,
)
from tests.support.robust_suite_documents import document, encode

# The cyrillic and traditional sources of the three `mong` rows of the golden suite.
MONG_ROWS = {
    row.id: (row.source, row.source_mong) for row in golden_suite().sections["mong"]
}

GOLDEN_LINES = [
    "suite: golden",
    "system: golden",
    "numbers: 0.5000 (3/6), excluded 2",
    "placeholders: 0.4000 (2/5), excluded 1",
    "entities: 0.5714 (4/7 mentions)",
    "terms: 0.7500 (3/4 mentions), excluded 1",
    "script: 0.7500 (mean of 5 rows), below floor 2, excluded 1",
    "mong: 0.5018 (retention over 3 rows, chrF 50.18 over 100.00 Cyrillic)",
]


@pytest.fixture
def workspace(tmp_path: Path) -> RobustWorkspace:
    return make_workspace(tmp_path)


def paths_lines(workspace: RobustWorkspace) -> list[str]:
    out = workspace.out_dir
    return [
        f"result: {out / 'robustness.json'}",
        f"report: {out / 'report.json'}",
        f"manifest: {out / 'manifest.json'}",
    ]


def lines_of(result: Result) -> list[str]:
    return result.stdout.splitlines()


# --- --hypotheses ------------------------------------------------------------------------------


def test_a_hypotheses_run_prints_the_figures_of_every_section_and_the_three_paths(
    workspace: RobustWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args("--system", "golden"))

    assert result.exit_code == 0, result.output
    assert lines_of(result) == [*GOLDEN_LINES, *paths_lines(workspace)]
    assert result.stderr == ""


def test_a_hypotheses_run_writes_the_result_the_report_and_the_manifest_and_nothing_else(
    workspace: RobustWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args("--system", "golden"))

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        "manifest.json",
        "report.json",
        "robustness.json",
    ]
    assert not workspace.hypotheses_out.exists()


def test_the_result_is_the_golden_result_byte_for_byte(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    assert (
        workspace.result.read_bytes()
        == (GOLDEN / "golden_robustness.json").read_bytes()
    )


def test_the_report_holds_the_golden_values_their_counts_and_the_digest_of_the_result(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    report = json.loads(workspace.report.read_text(encoding="utf-8"))

    assert report == {
        "schema": "rinchen.report.robustness/1",
        "system": {"name": "golden", "model_sha256": None, "manifest_sha256": None},
        "suite": {
            "name": "golden",
            "sha256": "2767112dc06c61b62fbc66ad7336b3dffeff81c33ce7e7e124875a44503e75b0",
        },
        "robustness": {
            "numbers_acc": 0.5,
            "placeholder_acc": 0.4,
            "term_acc": 0.75,
            "script_purity": 0.75,
            "entity_acc": 0.5714285714285714,
            "mong_source_retention": 0.5017570743156547,
        },
        "coverage": GOLDEN_COVERAGE,
        "robustness_file": "robustness.json",
        "robustness_sha256": sha256_of(workspace.result),
    }


def test_two_runs_give_the_same_result_and_the_same_report_bytes(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    again = tmp_path / "again"

    invoke(workspace.hypotheses_args("--system", "golden"))
    invoke(workspace.hypotheses_args("--system", "golden", out_dir=again))

    assert (again / "robustness.json").read_bytes() == workspace.result.read_bytes()
    assert (again / "report.json").read_bytes() == workspace.report.read_bytes()


def test_without_a_name_the_system_is_the_stem_of_the_hypotheses_file(
    workspace: RobustWorkspace,
) -> None:
    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert lines_of(result)[:2] == ["suite: golden", "system: golden_hypotheses"]
    assert json.loads(workspace.result.read_text(encoding="utf-8"))["system"] == {
        "name": "golden_hypotheses",
        "model_sha256": None,
        "manifest_sha256": None,
        "hypotheses_sha256": GOLDEN_HYPOTHESES_SHA256,
        "options": None,
    }


def test_a_run_into_an_existing_directory_replaces_the_files_of_the_last_run(
    workspace: RobustWorkspace,
) -> None:
    workspace.out_dir.mkdir()
    workspace.result.write_text("old result", encoding="utf-8")
    workspace.report.write_text("old report", encoding="utf-8")
    (workspace.out_dir / "notes.txt").write_text("not ours", encoding="utf-8")

    result = invoke(workspace.hypotheses_args("--system", "golden"))

    assert result.exit_code == 0, result.output
    assert (
        workspace.result.read_bytes()
        == (GOLDEN / "golden_robustness.json").read_bytes()
    )
    assert (
        json.loads(workspace.report.read_text(encoding="utf-8"))["system"]["name"]
        == "golden"
    )
    assert (workspace.out_dir / "notes.txt").read_text(encoding="utf-8") == "not ours"


# --- the lines of the sections -----------------------------------------------------------------


def test_a_section_without_a_scored_row_prints_n_a_and_the_exclusions_it_keeps(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    doc = document()
    doc["sections"] = {
        "numbers": doc["sections"]["numbers"],
        "script": doc["sections"]["script"],
    }
    doc["sections"]["numbers"][0]["source"] = "It costs a lot."
    doc["sections"]["numbers"][0]["reference"] = "Үнэ их."
    doc["sections"]["script"][0]["source"] = "1 2 3"
    suite = tmp_path / "null_suite.json"
    suite.write_bytes(encode(doc))
    hypotheses = tmp_path / "null_hypotheses.jsonl"
    hypotheses.write_text(
        '{"arm": "main", "id": "n1", "section": "numbers", "text": "Үнэ их."}\n'
        '{"arm": "main", "id": "s1", "section": "script", "text": "1 2 3"}\n',
        encoding="utf-8",
    )

    result = invoke(
        workspace.hypotheses_args(
            "--system", "nothing", suite=suite, hypotheses=hypotheses
        )
    )

    assert result.exit_code == 0, result.output
    assert lines_of(result)[:4] == [
        "suite: tiny",
        "system: nothing",
        "numbers: n/a (no scored row), excluded 1",
        "script: n/a (no scored row), below floor 0, excluded 1",
    ]
    report = json.loads(workspace.report.read_text(encoding="utf-8"))
    assert report["robustness"]["numbers_acc"] is None
    assert report["robustness"]["script_purity"] is None


def test_a_suite_with_one_section_prints_that_section_only(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    doc = document()
    doc["sections"] = {"mong": doc["sections"]["mong"]}
    suite = tmp_path / "mong_suite.json"
    suite.write_bytes(encode(doc))
    hypotheses = tmp_path / "mong_hypotheses.jsonl"
    hypotheses.write_text(
        '{"arm": "cyrl", "id": "m1", "section": "mong", "text": "This is a book."}\n'
        '{"arm": "mong", "id": "m1", "section": "mong", "text": "This is a book."}\n',
        encoding="utf-8",
    )

    result = invoke(
        workspace.hypotheses_args("--system", "one", suite=suite, hypotheses=hypotheses)
    )

    assert result.exit_code == 0, result.output
    assert lines_of(result)[2:-3] == [
        "mong: 1.0000 (retention over 1 rows, chrF 100.00 over 100.00 Cyrillic)"
    ]
    report = json.loads(workspace.report.read_text(encoding="utf-8"))
    assert report["robustness"] == {
        "numbers_acc": None,
        "placeholder_acc": None,
        "term_acc": None,
        "script_purity": None,
        "entity_acc": None,
        "mong_source_retention": 1.0,
    }


def test_a_mong_section_without_rows_prints_n_a_and_no_chrf() -> None:
    # The suite reader refuses an empty section: only a suite built in memory has one.
    result = compute_robustness(
        suite_of(mong=()), {}, suite_sha256="0" * 64, system=RobustnessSystem(name="s")
    )

    assert result_lines(result)[2:] == ["mong: n/a (no scored row)"]


def test_a_control_that_scores_zero_leaves_no_retention_and_shows_both_chrfs(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    doc = document()
    doc["sections"] = {"mong": doc["sections"]["mong"]}
    suite = tmp_path / "mong_suite.json"
    suite.write_bytes(encode(doc))
    hypotheses = tmp_path / "mong_hypotheses.jsonl"
    hypotheses.write_text(
        '{"arm": "cyrl", "id": "m1", "section": "mong", "text": "Zq"}\n'
        '{"arm": "mong", "id": "m1", "section": "mong", "text": "This is a book."}\n',
        encoding="utf-8",
    )

    result = invoke(
        workspace.hypotheses_args("--system", "one", suite=suite, hypotheses=hypotheses)
    )

    assert result.exit_code == 0, result.output
    assert lines_of(result)[2:-3] == [
        "mong: n/a (retention over 1 rows, chrF 100.00 over 0.00 Cyrillic)"
    ]
    report = json.loads(workspace.report.read_text(encoding="utf-8"))
    assert report["robustness"]["mong_source_retention"] is None
    assert report["coverage"]["mong_source_retention"] == {
        "scored": 1,
        "excluded": 0,
        "total": 1,
        "chrf_cyrl": 0.0,
        "chrf_mong": 100.0,
    }


# --- --model -----------------------------------------------------------------------------------


def test_a_model_run_asks_the_engine_once_for_every_request_of_the_suite_in_order(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert result.exit_code == 0, result.output
    assert len(engine.calls) == 1
    expected = [
        Row(request.text, request.src_lang, request.tgt_lang, request.script)
        for request in golden_requests()
    ]
    assert engine.calls[0].rows == expected
    assert len(expected) == 33


def test_the_mong_rows_are_asked_once_as_cyrillic_and_once_as_the_traditional_script(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    install(monkeypatch, engine)

    invoke(workspace.model_args())

    rows = engine.calls[0].rows
    source, traditional = MONG_ROWS["m1"]
    assert Row(source, "khk_Cyrl", "eng_Latn", None) in rows
    assert Row(traditional, "mn_Mong", "eng_Latn", None) in rows
    assert rows.index(Row(source, "khk_Cyrl", "eng_Latn", None)) + 1 == rows.index(
        Row(traditional, "mn_Mong", "eng_Latn", None)
    )


def test_the_decode_options_are_the_flags(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    install(monkeypatch, engine)

    result = invoke(
        workspace.model_args(
            "--beam-size", "2", "--max-decoding-length", "64", "--script-mask", "always"
        )
    )

    assert result.exit_code == 0, result.output
    assert engine.calls[0].options == DecodeOptions(
        beam_size=2, max_decoding_length=64, script_mask="always"
    )


def test_the_default_decode_options_are_beam_4_length_256_and_retry(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    install(monkeypatch, engine)

    invoke(workspace.model_args())

    assert engine.calls[0].options == DecodeOptions(
        beam_size=4, max_decoding_length=256, script_mask="retry"
    )


def test_the_engine_is_opened_on_the_cpu_with_the_threads_asked_for(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, workspace.engine())

    invoke(workspace.model_args("--threads", "3"))

    assert [(call.model, call.load, call.converter) for call in seams.opens] == [
        (workspace.export.directory, LoadOptions(device="cpu", intra_threads=3), None)
    ]
    assert seams.loads == []


def test_one_thread_and_the_cpu_are_the_defaults(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, workspace.engine())

    invoke(workspace.model_args())

    assert seams.opens[0].load == LoadOptions(device="cpu", intra_threads=1)


def test_the_device_cuda_is_passed_on_and_never_run_here(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--device", "cuda"))

    assert result.exit_code == 0, result.output
    assert seams.opens[0].load == LoadOptions(device="cuda", intra_threads=1)
    options = json.loads(workspace.result.read_text(encoding="utf-8"))["system"][
        "options"
    ]
    assert options["device"] == "cuda"


def test_the_model_may_come_from_the_environment(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, workspace.engine())
    arguments = workspace.model_args()
    at = arguments.index("--model")
    del arguments[at : at + 2]

    result = invoke(arguments, env={"RINCHEN_MODEL": str(workspace.export.directory)})

    assert result.exit_code == 0, result.output
    assert seams.opens[0].model == workspace.export.directory


def test_the_engine_is_closed_after_a_good_run(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    invoke(workspace.model_args())

    assert engine.closed == 1
    assert seams.events == ["open", "translate", "close"]


def test_a_model_run_names_the_system_after_the_export_and_prints_it(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args())

    assert lines_of(result)[:2] == [
        "suite: golden",
        f"system: {workspace.export.record.name}",
    ]


def test_the_name_given_replaces_the_name_of_the_export(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--system", "my-teacher_v1.2"))

    assert lines_of(result)[1] == "system: my-teacher_v1.2"
    system = json.loads(workspace.result.read_text(encoding="utf-8"))["system"]
    assert system["name"] == "my-teacher_v1.2"


def test_the_result_names_the_model_by_the_engine_and_records_the_options(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    invoke(
        workspace.model_args(
            "--beam-size", "2", "--max-decoding-length", "64", "--script-mask", "always"
        )
    )

    system = json.loads(workspace.result.read_text(encoding="utf-8"))["system"]
    assert system == {
        "name": workspace.export.record.name,
        "model_sha256": workspace.export.sha256["model.bin"],
        "manifest_sha256": workspace.export.manifest_sha256,
        "hypotheses_sha256": None,
        "options": {
            "beam_size": 2,
            "converter": False,
            "device": "cpu",
            "max_decoding_length": 64,
            "script_mask": "always",
            "threads": 1,
        },
    }
    report = json.loads(workspace.report.read_text(encoding="utf-8"))
    assert report["system"] == {
        "name": workspace.export.record.name,
        "model_sha256": workspace.export.sha256["model.bin"],
        "manifest_sha256": workspace.export.manifest_sha256,
    }


def test_the_hypotheses_file_holds_the_answers_of_the_engine_in_request_order(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    invoke(workspace.model_args())

    assert workspace.hypotheses_out.read_bytes() == workspace.hypotheses.read_bytes()


def test_a_model_run_prints_the_same_figures_and_the_same_paths(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--system", "golden"))

    assert lines_of(result) == [*GOLDEN_LINES, *paths_lines(workspace)]
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        "hypotheses.jsonl",
        "manifest.json",
        "report.json",
        "robustness.json",
    ]


def test_the_hypotheses_file_a_model_run_wrote_gives_the_same_figures_when_read_back(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch, workspace.engine())
    invoke(workspace.model_args())
    again = tmp_path / "again"

    result = invoke(
        workspace.hypotheses_args(hypotheses=workspace.hypotheses_out, out_dir=again)
    )

    assert result.exit_code == 0, result.output
    first = json.loads(workspace.result.read_text(encoding="utf-8"))
    second = json.loads((again / "robustness.json").read_text(encoding="utf-8"))
    for part in ("robustness", "sections", "rows", "suite"):
        assert second[part] == first[part]
    assert second["system"]["model_sha256"] is None
    assert not (again / "hypotheses.jsonl").exists()


def test_the_statuses_the_engine_gave_are_written_and_counted(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    invoke(workspace.model_args())

    lines = workspace.hypotheses_out.read_text(encoding="utf-8").splitlines()
    statuses = [json.loads(line)["status"] for line in lines]
    assert statuses.count("empty") == 3
    assert len(lines) == 33
    document = json.loads(workspace.result.read_text(encoding="utf-8"))
    assert (
        sum(
            section["statuses"].get("empty", 0)
            for section in document["sections"].values()
        )
        == 3
    )


# --- the converter -----------------------------------------------------------------------------


def converter_run(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeConverter, FakeRobustEngine, Seams]:
    converter = FakeConverter(prefix="[cyrl]")
    engine = workspace.engine(answer=echo_answers)
    seams = install(monkeypatch, engine, converter=converter)
    result = invoke(workspace.model_args(*workspace.converter_args()))
    assert result.exit_code == 0, result.output
    return converter, engine, seams


def test_the_converter_is_opened_first_on_the_device_of_the_engine_and_attached_to_it(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter, _, seams = converter_run(workspace, monkeypatch)

    assert [(c.ct2_dir, c.spm_model, c.device) for c in seams.loads] == [
        (workspace.converter_ct2, workspace.converter_spm, "cpu")
    ]
    assert seams.events[:2] == ["open_converter", "open"]
    assert seams.opens[0].converter is converter


def test_the_converter_turns_the_traditional_source_of_every_mong_row_into_cyrillic(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter, _, _ = converter_run(workspace, monkeypatch)

    assert converter.calls == [
        (traditional, "Cyrl") for _, traditional in MONG_ROWS.values()
    ]


def test_every_mong_row_gains_a_converted_request_with_the_text_the_converter_made(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, engine, _ = converter_run(workspace, monkeypatch)

    rows = engine.calls[0].rows
    assert len(rows) == 36
    for source, traditional in MONG_ROWS.values():
        converted = Row(f"[cyrl]{traditional}", "khk_Cyrl", "eng_Latn", None)
        assert rows.count(converted) == 1
        at = rows.index(converted)
        assert rows[at - 2].text == source
        assert rows[at - 1].text == traditional


def test_the_converted_arm_is_scored_and_the_options_say_a_converter_was_used(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter_run(workspace, monkeypatch)

    document = json.loads(workspace.result.read_text(encoding="utf-8"))
    details = document["sections"]["mong"]["details"]
    assert details["chrf_converted"] is not None
    assert details["retention_converted"] is not None
    assert document["system"]["options"]["converter"] is True
    assert len(workspace.hypotheses_out.read_text(encoding="utf-8").splitlines()) == 36


def test_the_hypotheses_of_a_run_with_a_converter_can_be_read_back_without_one(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter_run(workspace, monkeypatch)
    again = tmp_path / "again"

    result = invoke(
        workspace.hypotheses_args(hypotheses=workspace.hypotheses_out, out_dir=again)
    )

    assert result.exit_code == 0, result.output
    first = json.loads(workspace.result.read_text(encoding="utf-8"))
    second = json.loads((again / "robustness.json").read_text(encoding="utf-8"))
    assert second["sections"] == first["sections"]
    assert second["robustness"] == first["robustness"]
    assert second["sections"]["mong"]["details"]["retention_converted"] is not None


def test_a_converter_run_is_closed_like_any_other(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, engine, _ = converter_run(workspace, monkeypatch)

    assert engine.closed == 1


def test_the_requests_of_a_suite_without_a_mong_section_need_no_conversion(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc = document()
    doc["sections"] = {"numbers": doc["sections"]["numbers"]}
    suite = tmp_path / "numbers_only.json"
    suite.write_bytes(encode(doc))
    converter = FakeConverter()
    engine = workspace.engine(answer=echo_answers)
    install(monkeypatch, engine, converter=converter)

    result = invoke(workspace.model_args(*workspace.converter_args(), suite=suite))

    assert result.exit_code == 0, result.output
    assert converter.calls == []
    assert len(engine.calls[0].rows) == 1


# --- help --------------------------------------------------------------------------------------


def test_the_help_of_eval_lists_robustness() -> None:
    result = invoke(["eval", "--help"])

    assert result.exit_code == 0
    assert "robustness" in result.stdout


def test_the_help_of_robustness_lists_every_option_and_marks_the_required_ones() -> (
    None
):
    result = invoke(["eval", "robustness", "--help"])

    assert result.exit_code == 0
    for option in (
        "--suite",
        "--model",
        "--hypotheses",
        "--out-dir",
        "--system",
        "--beam-size",
        "--max-decoding-length",
        "--script-mask",
        "--device",
        "--threads",
        "--script-converter-ct2",
        "--script-converter-spm",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 2
