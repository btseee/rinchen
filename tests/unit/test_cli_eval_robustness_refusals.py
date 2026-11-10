"""`rinchen eval robustness`: what it refuses before it runs, with the exit code and the one-line
error the command documents (2 for a bad option, an export that is not vouched for, a file that
changed after it was hashed, an output directory that is an input's; 3 for a suite or a hypotheses
file that cannot be read or is not one, and another stage's manifest in the output directory; 1
when the output directory cannot be made).

A refusal never creates the output directory and never opens the engine or the converter. What a
run that fails after that leaves behind is in `test_cli_eval_robustness_failures.py`, the texts
that no message holds in `test_cli_eval_robustness_messages.py`, the prints and the files of a
good run in `test_cli_eval_robustness.py` and the manifest in
`test_cli_eval_robustness_manifest.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import eval_robustness
from tests.support.distill_cli import write_foreign_manifest
from tests.support.engines import FakeConverter
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.robust_cli import (
    CONVERTER_FILES,
    RobustWorkspace,
    Seams,
    install,
    invoke,
    make_workspace,
)
from tests.support.robust_suite_documents import document, encode


@pytest.fixture
def workspace(tmp_path: Path) -> RobustWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def seams(workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch) -> Seams:
    return install(monkeypatch, workspace.engine(), converter=FakeConverter())


def assert_refused(
    workspace: RobustWorkspace, seams: Seams, result: Any, code: int
) -> None:
    """Refused with `code`, before the engine or the converter was opened and with nothing
    on standard output and nothing written."""
    assert result.exit_code == code, result.output
    assert result.stdout == ""
    assert seams.opens == []
    assert seams.loads == []
    assert not workspace.out_dir.exists()


# --- the options -------------------------------------------------------------------------------


def test_neither_a_model_nor_hypotheses_is_a_config_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    arguments = workspace.hypotheses_args()
    at = arguments.index("--hypotheses")
    del arguments[at : at + 2]

    result = invoke(arguments)

    assert_one_error(result, 2, "give --model or --hypotheses")
    assert_refused(workspace, seams, result, 2)


def test_a_model_and_hypotheses_together_are_a_config_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    arguments = workspace.model_args("--hypotheses", str(workspace.hypotheses))

    result = invoke(arguments)

    assert_one_error(result, 2, "give --model or --hypotheses, not both")
    assert "RINCHEN_MODEL" in result.stderr
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("option", ["--suite", "--out-dir"])
def test_the_suite_and_the_output_directory_are_required(
    workspace: RobustWorkspace, seams: Seams, option: str
) -> None:
    arguments = workspace.hypotheses_args()
    at = arguments.index(option)
    del arguments[at : at + 2]

    result = invoke(arguments)

    assert_refused(workspace, seams, result, 2)
    assert option in result.stderr


@pytest.mark.parametrize("name", ["a b", "x" * 65, "", "a/b", "caf" + chr(0xE9)])
def test_a_system_name_that_is_not_an_id_is_a_config_error(
    workspace: RobustWorkspace, seams: Seams, name: str
) -> None:
    result = invoke(workspace.hypotheses_args("--system", name))

    assert_one_error(result, 2, "--system must be 1 to 64 characters")
    assert_refused(workspace, seams, result, 2)


def test_a_name_of_64_characters_and_every_allowed_one_is_accepted(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args("--system", "a" * 63 + "."))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[1] == "system: " + "a" * 63 + "."


def test_a_file_stem_that_is_not_an_id_needs_a_system_name(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    odd = workspace.hypotheses.with_name("two words.jsonl")
    workspace.hypotheses.rename(odd)

    result = invoke(workspace.hypotheses_args(hypotheses=odd))

    assert_one_error(result, 2, "the name of the hypotheses file")
    assert "--system" in result.stderr
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--beam-size", "0"),
        ("--beam-size", "9"),
        ("--max-decoding-length", "0"),
        ("--threads", "0"),
        ("--device", "auto"),
        ("--script-mask", "sometimes"),
    ],
)
def test_a_usage_error_is_exit_2_and_nothing_runs(
    workspace: RobustWorkspace, seams: Seams, arguments: tuple[str, str]
) -> None:
    result = invoke(workspace.model_args(*arguments))

    assert_refused(workspace, seams, result, 2)
    assert arguments[0] in result.stderr


@pytest.mark.parametrize("beam_size", ["1", "8"])
def test_the_beam_sizes_at_the_bounds_are_accepted(
    workspace: RobustWorkspace, seams: Seams, beam_size: str
) -> None:
    result = invoke(workspace.model_args("--beam-size", beam_size))

    assert result.exit_code == 0, result.output


# --- the converter pair ------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [0, 1])
def test_one_of_the_converter_options_alone_is_a_config_error(
    workspace: RobustWorkspace, seams: Seams, keep: int
) -> None:
    pair = workspace.converter_args()
    alone = pair[2 * keep : 2 * keep + 2]

    result = invoke(workspace.model_args(*alone))

    assert_one_error(
        result, 2, "--script-converter-ct2 and --script-converter-spm go together"
    )
    assert_refused(workspace, seams, result, 2)


def test_the_converter_pair_is_refused_with_hypotheses(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args(*workspace.converter_args()))

    assert_one_error(result, 2, "the script converter options need --model")
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("option", ["--script-converter-ct2", "--script-converter-spm"])
def test_one_converter_option_is_refused_with_hypotheses_too(
    workspace: RobustWorkspace, seams: Seams, option: str
) -> None:
    pair = workspace.converter_args()
    at = pair.index(option)

    result = invoke(workspace.hypotheses_args(*pair[at : at + 2]))

    assert_one_error(result, 2, "the script converter options need --model")
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("name", CONVERTER_FILES)
def test_a_converter_directory_without_one_of_its_files_is_a_config_error_naming_it(
    workspace: RobustWorkspace, seams: Seams, name: str
) -> None:
    (workspace.converter_ct2 / name).unlink()

    result = invoke(workspace.model_args(*workspace.converter_args()))

    assert_one_error(result, 2, "")
    assert str(workspace.converter_ct2 / name) in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_converter_without_its_sentencepiece_model_is_a_config_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    workspace.converter_spm.unlink()

    result = invoke(workspace.model_args(*workspace.converter_args()))

    assert_one_error(result, 2, "")
    assert str(workspace.converter_spm) in result.stderr
    assert_refused(workspace, seams, result, 2)


# --- the export --------------------------------------------------------------------------------


def test_a_directory_that_is_no_export_is_refused_and_the_engine_is_never_opened(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    empty = workspace.root / "empty"
    empty.mkdir()

    result = invoke(workspace.model_args(model=empty))

    assert_one_error(result, 2, f"cannot use the export {empty}: no manifest vouches")
    assert_refused(workspace, seams, result, 2)


def test_a_model_that_does_not_exist_is_refused(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    missing = workspace.root / "nowhere"

    result = invoke(workspace.model_args(model=missing))

    assert_one_error(result, 2, f"the export directory {missing} does not exist")
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize(
    "name",
    ["model.bin", "shared_vocabulary.json", "config.json", "spm.model", "rinchen.json"],
)
def test_a_flipped_byte_in_a_file_of_the_export_is_refused_naming_the_file(
    workspace: RobustWorkspace, seams: Seams, name: str
) -> None:
    path = workspace.export.directory / name
    content = bytearray(path.read_bytes())
    content[0] ^= 0xFF
    path.write_bytes(bytes(content))

    result = invoke(workspace.model_args())

    assert_one_error(result, 2, "cannot use the export")
    assert name in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_the_export_is_vouched_for_before_the_suite_is_looked_at(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    (workspace.export.directory / "model.bin").write_bytes(b"flipped")
    workspace.suite.unlink()

    result = invoke(workspace.model_args())

    assert_one_error(result, 2, "cannot use the export")
    assert_refused(workspace, seams, result, 2)


# --- the suite ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["model", "hypotheses"])
def test_a_missing_suite_is_a_data_error(
    workspace: RobustWorkspace, seams: Seams, mode: str
) -> None:
    missing = workspace.root / "absent.json"
    arguments = (
        workspace.model_args(suite=missing)
        if mode == "model"
        else workspace.hypotheses_args(suite=missing)
    )

    result = invoke(arguments)

    assert_one_error(result, 3, f"suite file not found: {missing}")
    assert_refused(workspace, seams, result, 3)


def test_a_directory_as_the_suite_is_a_data_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args(suite=workspace.root))

    assert_one_error(result, 3, f"suite file not found: {workspace.root}")
    assert_refused(workspace, seams, result, 3)


def test_a_suite_with_a_field_of_the_wrong_type_is_refused_naming_section_row_and_field(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    doc = document()
    doc["sections"]["numbers"][0]["source"] = 5
    workspace.suite.write_bytes(encode(doc))

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result,
        3,
        "not a robustness suite: section 'numbers', row index 0 (id 'n1'), field 'source'",
    )
    assert_refused(workspace, seams, result, 3)


def test_a_suite_with_an_unknown_key_is_refused_naming_section_and_row(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    doc = document()
    doc["sections"]["numbers"][0]["bogus"] = 1
    workspace.suite.write_bytes(encode(doc))

    result = invoke(workspace.model_args())

    assert_one_error(
        result,
        3,
        "not a robustness suite: section 'numbers', row index 0 (id 'n1'): has a field "
        "that is not part of the format",
    )
    assert_refused(workspace, seams, result, 3)


def test_a_suite_that_is_not_json_is_a_data_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    workspace.suite.write_bytes(b"{not json")

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result, 3, "not a robustness suite: suite: the text is not valid JSON"
    )
    assert_refused(workspace, seams, result, 3)


def test_a_suite_that_changed_after_it_was_hashed_is_refused_as_a_config_error(
    workspace: RobustWorkspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = eval_robustness.verified_content

    def changing(file: Path, digest: str) -> bytes:
        if file == workspace.suite:
            file.write_bytes(file.read_bytes() + b"\n")
        return real(file, digest)

    monkeypatch.setattr(eval_robustness, "verified_content", changing)

    result = invoke(workspace.hypotheses_args())

    assert_one_error(result, 2, f"{workspace.suite} changed after it was hashed")
    assert_refused(workspace, seams, result, 2)


def test_a_hypotheses_file_that_changed_after_it_was_hashed_is_refused(
    workspace: RobustWorkspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = eval_robustness.verified_content

    def changing(file: Path, digest: str) -> bytes:
        if file == workspace.hypotheses:
            file.write_bytes(file.read_bytes() + b"\n")
        return real(file, digest)

    monkeypatch.setattr(eval_robustness, "verified_content", changing)

    result = invoke(workspace.hypotheses_args())

    assert_one_error(result, 2, f"{workspace.hypotheses} changed after it was hashed")
    assert_refused(workspace, seams, result, 2)


# --- the hypotheses ----------------------------------------------------------------------------


def hypotheses_lines(workspace: RobustWorkspace) -> list[bytes]:
    return workspace.hypotheses.read_bytes().splitlines(keepends=True)


def test_a_missing_hypotheses_file_is_a_data_error(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    missing = workspace.root / "absent.jsonl"

    result = invoke(workspace.hypotheses_args(hypotheses=missing))

    assert_one_error(result, 3, f"hypotheses file not found: {missing}")
    assert_refused(workspace, seams, result, 3)


def test_a_hypotheses_file_that_lacks_a_request_is_refused_naming_section_id_and_arm(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    lines = hypotheses_lines(workspace)
    workspace.hypotheses.write_bytes(b"".join(lines[1:]))

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result,
        3,
        "not a hypotheses file: the file: lacks 1 request(s), the first is "
        "section 'numbers', id 'n1', arm 'main'",
    )
    assert_refused(workspace, seams, result, 3)


def test_a_hypotheses_file_with_a_request_the_suite_does_not_have_is_refused(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    extra = json.dumps(
        {"section": "numbers", "id": "zz9", "arm": "main", "text": "x"}
    ).encode()
    workspace.hypotheses.write_bytes(workspace.hypotheses.read_bytes() + extra + b"\n")

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result,
        3,
        "not a hypotheses file: line 34 (section 'numbers', id 'zz9', arm 'main'): "
        "is not a request of the suite",
    )
    assert_refused(workspace, seams, result, 3)


def test_a_hypotheses_file_that_answers_a_request_twice_is_refused(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    lines = hypotheses_lines(workspace)
    workspace.hypotheses.write_bytes(b"".join([*lines, lines[0]]))

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result,
        3,
        "not a hypotheses file: line 34 (section 'numbers', id 'n1', arm 'main'): "
        "answers a request that is answered more than once",
    )
    assert_refused(workspace, seams, result, 3)


def test_the_converted_arm_for_some_mong_rows_only_is_refused(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    line = json.dumps(
        {"section": "mong", "id": "m1", "arm": "converted", "text": "x"}
    ).encode()
    workspace.hypotheses.write_bytes(workspace.hypotheses.read_bytes() + line + b"\n")

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 3, result.output
    assert "section 'mong', id 'm2', arm 'converted'" in result.stderr
    assert_refused(workspace, seams, result, 3)


# --- the output directory ----------------------------------------------------------------------


@pytest.mark.parametrize("input_name", ["suite", "hypotheses"])
def test_an_output_directory_that_is_the_directory_of_an_input_is_refused_and_untouched(
    workspace: RobustWorkspace, seams: Seams, input_name: str
) -> None:
    directory = (
        workspace.suite if input_name == "suite" else workspace.hypotheses
    ).parent
    before = snapshot(directory)

    result = invoke(workspace.hypotheses_args(out_dir=directory))

    assert_one_error(
        result, 2, f"--out-dir {directory} is the directory of the input file "
    )
    assert snapshot(directory) == before
    assert seams.opens == []


@pytest.mark.parametrize("input_name", ["export", "converter", "suite"])
def test_a_model_run_refuses_the_directory_of_every_input_as_its_output(
    workspace: RobustWorkspace, seams: Seams, input_name: str
) -> None:
    directory = {
        "export": workspace.export.directory,
        "converter": workspace.converter_ct2,
        "suite": workspace.suite.parent,
    }[input_name]
    before = snapshot(directory)

    result = invoke(
        workspace.model_args(*workspace.converter_args(), out_dir=directory)
    )

    assert_one_error(
        result, 2, f"--out-dir {directory} is the directory of the input file "
    )
    assert snapshot(directory) == before
    assert seams.opens == []
    assert seams.loads == []


@pytest.mark.parametrize("mode", ["model", "hypotheses"])
def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_and_untouched(
    workspace: RobustWorkspace, seams: Seams, mode: str
) -> None:
    write_foreign_manifest(workspace.out_dir)
    before = snapshot(workspace.out_dir)
    arguments = (
        workspace.model_args() if mode == "model" else workspace.hypotheses_args()
    )

    result = invoke(arguments)

    assert_one_error(
        result, 3, f"{workspace.out_dir} holds the manifest of stage 'train_docify'"
    )
    assert snapshot(workspace.out_dir) == before
    assert seams.opens == []


def test_an_output_directory_that_cannot_be_made_exits_1(
    workspace: RobustWorkspace, seams: Seams
) -> None:
    blocked = workspace.root / "blocked"
    blocked.write_text("a file where the directory should be", encoding="utf-8")

    result = invoke(workspace.hypotheses_args(out_dir=blocked / "out"))

    assert_one_error(result, 1, "cannot write the robustness result under")
    assert result.stdout == ""
