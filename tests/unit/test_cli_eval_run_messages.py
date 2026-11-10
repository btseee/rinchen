"""No way for `rinchen eval run` to fail prints a text of a suite, of a hypotheses or scores file, of
a robustness file, of a generators configuration, of a vocabulary or of what a system answered (D13,
Task 7, spec T15: "no text of a suite, an output or a score file is ever in a message").

Every scenario plants `MARKER` in the very text a refusal is about, runs the command through the
way it fails and asserts that the marker is in no standard output, no standard error and no log
line, and that the failure is one `error:` line with the exit code the command documents. The scan
of the interpolations that a message is built from is in `test_eval_error_messages_d13.py`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.errors import UpstreamError
from rinchen.eval.outputs import Output
from tests.support.eval_cli import (
    MARKER,
    EvalWorkspace,
    install,
    invoke,
    make_workspace,
    rewrite_robustness_file,
    robustness_directories,
    scores_bytes,
)
from tests.support.eval_fakes import (
    GOLDEN_OUTPUTS_FILES,
    FakeCandidateTranslator,
    answering,
    golden_outputs,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

Scenario = Callable[[EvalWorkspace, pytest.MonkeyPatch], tuple[list[str], int]]


def alpha_lines(workspace: EvalWorkspace) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in workspace.alpha.read_text("utf-8").splitlines()
    ]


def write_lines(path: Path, lines: list[Any]) -> None:
    path.write_text(
        "".join(
            (line if isinstance(line, str) else json.dumps(line, ensure_ascii=False))
            + "\n"
            for line in lines
        ),
        "utf-8",
    )


def marked_alpha(workspace: EvalWorkspace) -> list[dict[str, Any]]:
    lines = alpha_lines(workspace)
    for line in lines:
        line["text"] = f"{MARKER} {line['text']}"
    return lines


# ---- the hypotheses file ----------------------------------------------------------------------------


def outputs_with_an_unknown_id(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[3]["id"] = f"{MARKER} not an id"
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_with_a_well_formed_unknown_id(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[3]["id"] = "g99"
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_with_a_bad_status(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[0]["status"] = MARKER
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_with_a_field_of_the_marker_name(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[0][MARKER] = MARKER
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_of_a_direction_that_is_not_one(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[0]["direction"] = MARKER
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_that_miss_a_row(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    write_lines(workspace.alpha, marked_alpha(workspace)[:-1])
    return workspace.hypotheses_args(), 3


def outputs_that_are_not_json(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines: list[Any] = marked_alpha(workspace)
    lines[2] = f"{MARKER} is not json"
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_with_a_repeated_key(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines: list[Any] = marked_alpha(workspace)
    lines[1] = f'{{"text": "{MARKER}", "text": "{MARKER}"}}'
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_with_a_number_for_text(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = marked_alpha(workspace)
    lines[1]["text"] = 7
    lines[2]["id"] = f"{MARKER} x"
    write_lines(workspace.alpha, lines)
    return workspace.hypotheses_args(), 3


def outputs_that_are_not_utf_8(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    workspace.alpha.write_bytes(MARKER.encode() + b"\xff\n")
    return workspace.hypotheses_args(), 3


# ---- the suite --------------------------------------------------------------------------------------------


def suite_document(workspace: EvalWorkspace) -> dict[str, Any]:
    document = json.loads(workspace.suite.read_text("utf-8"))
    for row in document["rows"]:
        for language in row["text"]:
            row["text"][language] = f"{MARKER} {row['text'][language]}"
    return document


def write_suite(workspace: EvalWorkspace, document: dict[str, Any]) -> list[str]:
    path = workspace.root / "in" / "marked_suite.json"
    path.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
    return workspace.hypotheses_args(suite=path)


def suite_with_a_key_named_by_the_marker(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    document = suite_document(workspace)
    document[MARKER] = MARKER
    return write_suite(workspace, document), 3


def suite_with_a_language_named_by_the_marker(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    document = suite_document(workspace)
    document["rows"][0]["text"][MARKER] = MARKER
    return write_suite(workspace, document), 3


def suite_with_a_blank_sentence(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    document = suite_document(workspace)
    document["rows"][2]["text"]["eng_Latn"] = "   "
    document["rows"][3]["id"] = f"{MARKER} spaces"
    return write_suite(workspace, document), 3


def suite_that_is_not_json(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    path = workspace.root / "in" / "marked_suite.json"
    path.write_text(f"{MARKER} is not json", "utf-8")
    return workspace.hypotheses_args(suite=path), 3


# ---- the scores ------------------------------------------------------------------------------------------


def scores_lines(workspace: EvalWorkspace) -> list[Any]:
    suite_sha256 = hashlib.sha256(workspace.suite.read_bytes()).hexdigest()
    data = scores_bytes("comet22", suite_sha256, _alpha_sha(), value=0.5)
    return [json.loads(line) for line in data.decode().splitlines()]


def _alpha_sha() -> str:
    return hashlib.sha256(GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()).hexdigest()


def scored(workspace: EvalWorkspace, lines: list[Any]) -> list[str]:
    path = workspace.root / "in" / "comet22.jsonl"
    write_lines(path, lines)
    return workspace.hypotheses_args("--segment-scores", f"comet22={path}")


def scores_with_a_header_of_the_marker(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = scores_lines(workspace)
    lines[0]["metric"] = MARKER
    lines[0]["suite_sha256"] = MARKER
    return scored(workspace, lines), 3


def scores_with_an_id_of_the_marker(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = scores_lines(workspace)
    lines[2]["id"] = f"{MARKER} x"
    lines[3]["direction"] = MARKER
    return scored(workspace, lines), 3


def scores_that_are_text(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = scores_lines(workspace)
    lines[4]["score"] = MARKER
    return scored(workspace, lines), 3


def scores_that_are_not_json(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines: list[Any] = scores_lines(workspace)
    lines[5] = f"{MARKER} is not json"
    return scored(workspace, lines), 3


def scores_that_miss_a_segment(
    workspace: EvalWorkspace, _: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    lines = scores_lines(workspace)
    lines[1]["id"] = f"{MARKER} x"
    return scored(workspace, lines[:-1]), 3


# ---- the robustness directory ----------------------------------------------------------------------------


def robustness_report_of_a_system_named_by_the_marker(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    directory = robustness_directories(workspace, monkeypatch).named_alpha
    rewrite_robustness_file(
        directory, "report.json", lambda d: d["system"].__setitem__("name", MARKER)
    )
    return workspace.hypotheses_args("--robustness", str(directory)), 3


def robustness_block_with_a_value_of_the_marker(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    directory = robustness_directories(workspace, monkeypatch).named_alpha
    rewrite_robustness_file(
        directory,
        "report.json",
        lambda d: d["robustness"].__setitem__("numbers_acc", MARKER),
    )
    return workspace.hypotheses_args("--robustness", str(directory)), 3


def robustness_manifest_of_a_stage_named_by_the_marker(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    directory = robustness_directories(workspace, monkeypatch).named_alpha
    rewrite_robustness_file(
        directory, "manifest.json", lambda d: d.__setitem__("stage", MARKER)
    )
    return workspace.hypotheses_args("--robustness", str(directory)), 3


def robustness_manifest_that_is_not_json(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    directory = robustness_directories(workspace, monkeypatch).named_alpha
    (directory / "manifest.json").write_text(f"{MARKER} is not json", "utf-8")
    return workspace.hypotheses_args("--robustness", str(directory)), 3


def robustness_of_a_system_the_run_is_not(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    directory = robustness_directories(workspace, monkeypatch).named_alpha
    rewrite_robustness_file(
        directory, "report.json", lambda d: d["system"].__setitem__("name", "not an id")
    )
    rewrite_robustness_file(
        directory, "robustness.json", lambda d: d["system"].__setitem__("name", MARKER)
    )
    return workspace.hypotheses_args("--robustness", str(directory)), 3


# ---- the generators ------------------------------------------------------------------------------------------


def generators_with_values_of_the_marker(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    workspace.generators.write_text(
        f"generators:\n  x:\n    kind: vllm\n    model: {MARKER}\n    n: {MARKER}\n"
        f"    base_url: {MARKER}\n",
        "utf-8",
    )
    install(monkeypatch, None, FakeCandidateTranslator())
    return workspace.generator_args("x"), 2


def generators_that_are_not_yaml(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    workspace.generators.write_text(
        f"generators: [{MARKER}\n  x: {MARKER}: {MARKER}", "utf-8"
    )
    install(monkeypatch, None, FakeCandidateTranslator())
    return workspace.generator_args("x"), 2


def vocabulary_of_the_marker(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    (workspace.root / "nllb" / "shared_vocabulary.json").write_text(
        json.dumps([MARKER, "<unk>"]), "utf-8"
    )
    install(monkeypatch, None, FakeCandidateTranslator())
    return workspace.generator_args("nllb_toy"), 2


def vocabulary_that_is_not_a_list_of_strings(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    (workspace.root / "nllb" / "shared_vocabulary.json").write_text(
        json.dumps({MARKER: [MARKER, 7]}), "utf-8"
    )
    install(monkeypatch, None, FakeCandidateTranslator())
    return workspace.generator_args("nllb_toy"), 2


# ---- what a system answers -----------------------------------------------------------------------------------


def a_translator_that_answers_too_few_lists(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    translator = FakeCandidateTranslator(
        answer=lambda texts, src, tgt: [[f"{MARKER} {text}"] for text in texts][:-1]
    )
    install(monkeypatch, None, translator)
    return workspace.generator_args("nllb_toy"), 4


def a_translator_that_fails_upstream(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    def fail(texts: Any, src: str, tgt: str) -> Any:
        raise UpstreamError("the server answered 503")

    install(monkeypatch, None, FakeCandidateTranslator(answer=fail))
    return workspace.generator_args("vllm_toy"), 4


def an_engine_that_answers_too_few_texts(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    outputs = [
        Output(f"{MARKER} {o.text}", "ok") for o in golden_outputs("alpha").values()
    ]
    install(monkeypatch, workspace.engine(answer=answering(outputs[:-1])))
    return workspace.model_args(), 4


def an_engine_that_answers_a_lone_surrogate(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    outputs = [
        Output(f"{MARKER} {o.text}", "ok") for o in golden_outputs("alpha").values()
    ]
    outputs[4] = Output(f"{MARKER}{chr(0xD800)}", "ok")
    install(monkeypatch, workspace.engine(answer=answering(outputs)))
    return workspace.model_args(), 3


def an_engine_whose_files_are_not_the_hashed_ones(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    import dataclasses

    engine = workspace.engine()
    engine.info = dataclasses.replace(engine.info, model_sha256="f" * 64)
    install(monkeypatch, engine)
    return workspace.model_args(), 2


SCENARIOS: Final[dict[str, Scenario]] = {
    "outputs-unknown-id-not-an-id": outputs_with_an_unknown_id,
    "outputs-unknown-id-well-formed": outputs_with_a_well_formed_unknown_id,
    "outputs-bad-status": outputs_with_a_bad_status,
    "outputs-field-of-the-marker-name": outputs_with_a_field_of_the_marker_name,
    "outputs-direction-not-one": outputs_of_a_direction_that_is_not_one,
    "outputs-missing-row": outputs_that_miss_a_row,
    "outputs-not-json": outputs_that_are_not_json,
    "outputs-repeated-key": outputs_with_a_repeated_key,
    "outputs-number-for-text": outputs_with_a_number_for_text,
    "outputs-not-utf-8": outputs_that_are_not_utf_8,
    "suite-key-of-the-marker": suite_with_a_key_named_by_the_marker,
    "suite-language-of-the-marker": suite_with_a_language_named_by_the_marker,
    "suite-blank-sentence": suite_with_a_blank_sentence,
    "suite-not-json": suite_that_is_not_json,
    "scores-header-of-the-marker": scores_with_a_header_of_the_marker,
    "scores-id-of-the-marker": scores_with_an_id_of_the_marker,
    "scores-text-score": scores_that_are_text,
    "scores-not-json": scores_that_are_not_json,
    "scores-missing-segment": scores_that_miss_a_segment,
    "robustness-system-of-the-marker": robustness_report_of_a_system_named_by_the_marker,
    "robustness-value-of-the-marker": robustness_block_with_a_value_of_the_marker,
    "robustness-stage-of-the-marker": robustness_manifest_of_a_stage_named_by_the_marker,
    "robustness-manifest-not-json": robustness_manifest_that_is_not_json,
    "robustness-other-system": robustness_of_a_system_the_run_is_not,
    "generators-values-of-the-marker": generators_with_values_of_the_marker,
    "generators-not-yaml": generators_that_are_not_yaml,
    "vocabulary-of-the-marker": vocabulary_of_the_marker,
    "vocabulary-not-a-list-of-strings": vocabulary_that_is_not_a_list_of_strings,
    "translator-too-few-lists": a_translator_that_answers_too_few_lists,
    "translator-upstream": a_translator_that_fails_upstream,
    "engine-too-few-texts": an_engine_that_answers_too_few_texts,
    "engine-lone-surrogate": an_engine_that_answers_a_lone_surrogate,
    "engine-other-files": an_engine_whose_files_are_not_the_hashed_ones,
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_no_way_for_a_run_to_fail_prints_a_text_of_its_inputs_or_of_a_system(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = make_workspace(tmp_path)
    install(monkeypatch, workspace.engine(), FakeCandidateTranslator())
    arguments, code = SCENARIOS[scenario](workspace, monkeypatch)

    result = invoke(arguments)

    assert result.exit_code == code, result.output
    assert result.stderr.startswith("error: "), result.stderr
    assert result.stdout == ""
    assert MARKER not in result.stdout
    assert MARKER not in result.stderr
    assert MARKER not in caplog.text
    assert not workspace.out_dir.exists()
