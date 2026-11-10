"""`rinchen eval run`: what it refuses, with the exit code and the one-line error the command
documents (D13, Task 7). 2 is a bad option, a model that is not vouched for, a vocabulary that
lacks the family's tokens, a file that changed after it was hashed, an output directory that is an
input's, an error of the engine that is a configuration; 3 an unreadable or invalid input (the
suite, the outputs, a foreign manifest in the output directory); 4 an upstream failure; 130 an
interrupt.

A refusal before the run creates no output directory and never opens the engine or builds a
translator. What a run that fails part-way leaves is checked where it is raised: nothing is written
before every score is computed, so the files of the last run are as they were.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.export.record import EXPORT_FILES
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_cli import (
    FIRST,
    EvalWorkspace,
    Seams,
    assert_refused,
    install,
    invoke,
    make_workspace,
    run_dir_of,
)
from tests.support.eval_fakes import FakeCandidateTranslator
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def seams(workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch) -> Seams:
    return install(monkeypatch, workspace.engine(), FakeCandidateTranslator())


def edited_suite(workspace: EvalWorkspace, change: Any) -> Path:
    document = json.loads(workspace.suite.read_text("utf-8"))
    change(document)
    path = workspace.root / "in" / "edited_suite.json"
    path.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
    return path


# --- the system options --------------------------------------------------------------------------------


def test_no_system_option_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.base())

    assert_one_error(
        result, 2, "give exactly one of --model, --generator and --hypotheses"
    )
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize(
    "extra",
    [
        ("--model", "x", "--hypotheses", "y"),
        ("--generator", "vllm_toy", "--hypotheses", "y"),
        ("--model", "x", "--generator", "vllm_toy"),
        ("--model", "x", "--generator", "vllm_toy", "--hypotheses", "y"),
    ],
)
def test_two_system_options_are_a_config_error(
    workspace: EvalWorkspace, seams: Seams, extra: tuple[str, ...]
) -> None:
    result = invoke(workspace.base(*extra))

    assert_one_error(
        result, 2, "give exactly one of --model, --generator and --hypotheses"
    )
    assert "RINCHEN_MODEL" in result.stderr
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("name", ["a b", "x" * 65, "", "a/b", "caf" + chr(0xE9)])
def test_a_system_name_that_is_not_an_id_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, name: str
) -> None:
    result = invoke(workspace.hypotheses_args("--system", name))

    assert_one_error(result, 2, "--system must be 1 to 64 characters")
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("name", [".", ".."])
def test_a_system_name_that_names_a_directory_is_a_config_error_and_nothing_is_written(
    workspace: EvalWorkspace, seams: Seams, name: str
) -> None:
    result = invoke(workspace.hypotheses_args("--system", name))

    assert_one_error(result, 2, "the system name cannot be '.' or '..'")
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("name", [".", ".."])
def test_a_suite_named_like_a_directory_is_a_data_error_and_nothing_is_written(
    workspace: EvalWorkspace, seams: Seams, name: str
) -> None:
    suite = edited_suite(workspace, lambda d: d.__setitem__("name", name))

    result = invoke(workspace.hypotheses_args(suite=suite))

    assert_one_error(result, 3, "the suite name cannot be '.' or '..'")
    assert_refused(workspace, seams, result, 3)
    assert not (workspace.root / "sys").exists()


def test_a_file_stem_that_is_not_an_id_needs_a_system_name(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    odd = workspace.alpha.with_name("two words.jsonl")
    workspace.alpha.rename(odd)

    result = invoke(workspace.hypotheses_args(hypotheses=odd))

    assert_one_error(result, 2, "the name of the hypotheses file")
    assert "--system" in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_generator_name_that_is_not_an_id_needs_a_system_name(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.generator_args("two words"))

    assert_one_error(result, 2, "the name of the generator")
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
def test_a_model_option_outside_its_range_is_a_usage_error_with_nothing_written(
    workspace: EvalWorkspace, seams: Seams, arguments: tuple[str, ...]
) -> None:
    result = invoke(workspace.model_args(*arguments))

    assert_refused(workspace, seams, result, 2)
    assert arguments[0] in result.stderr


@pytest.mark.parametrize(
    "option",
    [
        ("--beam-size", "3"),
        ("--max-decoding-length", "64"),
        ("--script-mask", "always"),
        ("--device", "cpu"),
        ("--threads", "2"),
    ],
)
@pytest.mark.parametrize("system", ["hypotheses", "generator"])
def test_a_model_option_given_with_anything_but_a_model_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, option: tuple[str, ...], system: str
) -> None:
    arguments = (
        workspace.hypotheses_args(*option)
        if system == "hypotheses"
        else workspace.generator_args("vllm_toy", *option)
    )

    result = invoke(arguments)

    assert_one_error(
        result, 2, "these options set how the engine decodes and need --model"
    )
    assert option[0] in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_the_generators_config_needs_a_generator(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(
        workspace.hypotheses_args("--generators-config", str(workspace.generators))
    )

    assert_one_error(result, 2, "--generators-config needs --generator")
    assert_refused(workspace, seams, result, 2)


def test_a_device_cuda_is_validated_and_handed_to_the_engine_seam_never_run(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.model_args("--device", "cuda"))

    assert result.exit_code == 0, result.output
    assert seams.opens[0].load.device == "cuda"


@pytest.mark.parametrize(
    ("option", "value", "start"),
    [
        ("--resamples", "50", "resamples"),
        ("--resamples", "0", "resamples"),
        ("--seed", "-1", "seed"),
    ],
)
def test_a_bootstrap_setting_the_bootstrap_cannot_have_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, option: str, value: str, start: str
) -> None:
    result = invoke(workspace.hypotheses_args(option, value))

    assert result.exit_code == 2, result.output
    assert start in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_metrics_config_that_is_missing_or_unusable_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, tmp_path: Path
) -> None:
    missing = invoke(
        workspace.hypotheses_args("--metrics-config", str(tmp_path / "none.yaml"))
    )
    broken = tmp_path / "broken.yaml"
    broken.write_text("bootstrap: [", "utf-8")
    unusable = invoke(workspace.hypotheses_args("--metrics-config", str(broken)))

    assert_one_error(missing, 2, "eval metrics config not found")
    # PyYAML's own text can run over several lines, so only the start of the message is pinned.
    assert unusable.exit_code == 2
    assert unusable.stderr.startswith("error: invalid eval metrics config")
    assert_refused(workspace, seams, unusable, 2)


# --- the selection ------------------------------------------------------------------------------------


def test_a_direction_the_suite_does_not_declare_is_a_config_error_naming_it(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args("--direction", "khk_Cyrl-deu_Latn"))

    assert_one_error(
        result, 2, "--direction names a direction the suite does not declare"
    )
    assert "khk_Cyrl-deu_Latn" in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_direction_that_is_not_well_formed_is_refused_and_never_quoted(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args("--direction", "SECRET-MARKER-91"))

    assert_one_error(
        result, 2, "--direction names a direction the suite does not declare"
    )
    assert "SECRET-MARKER-91" not in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_a_direction_given_twice_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(
        workspace.hypotheses_args("--direction", FIRST, "--direction", FIRST)
    )

    assert_one_error(result, 2, "--direction")
    assert_refused(workspace, seams, result, 2)


def test_an_unknown_direction_never_opens_the_engine(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.model_args("--direction", "khk_Cyrl-deu_Latn"))

    assert_one_error(
        result, 2, "--direction names a direction the suite does not declare"
    )
    assert_refused(workspace, seams, result, 2)


# --- the suite ------------------------------------------------------------------------------------------


def test_a_suite_that_is_missing_or_a_directory_is_a_data_error(
    workspace: EvalWorkspace, seams: Seams, tmp_path: Path
) -> None:
    missing = invoke(workspace.hypotheses_args(suite=tmp_path / "none.json"))
    directory = invoke(workspace.hypotheses_args(suite=tmp_path))

    assert_one_error(missing, 3, "suite file not found")
    assert_one_error(directory, 3, "suite file not found")
    assert_refused(workspace, seams, directory, 3)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.__setitem__("surprise", 1),
        lambda d: (
            d["directions"].append("khk_Mong-eng_Latn"),
            d["rows"][0]["text"].__setitem__("khk_Mong", "x"),
        ),
        lambda d: d["rows"][0].__setitem__("text", {}),
    ],
    ids=["unknown-key", "khk_Mong", "empty-text"],
)
def test_a_suite_that_is_not_one_is_a_data_error_and_nothing_is_written(
    workspace: EvalWorkspace, seams: Seams, change: Any
) -> None:
    result = invoke(workspace.hypotheses_args(suite=edited_suite(workspace, change)))

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: not an evaluation suite:")
    assert_refused(workspace, seams, result, 3)


def test_khk_mong_is_refused_by_name(workspace: EvalWorkspace, seams: Seams) -> None:
    suite = edited_suite(
        workspace,
        lambda d: (
            d["directions"].append("khk_Mong-eng_Latn"),
            d["rows"][0]["text"].__setitem__("khk_Mong", "x"),
        ),
    )

    result = invoke(workspace.hypotheses_args(suite=suite))

    assert_one_error(result, 3, "not an evaluation suite:")
    assert "khk_Mong is refused by name" in result.stderr


def test_a_suite_with_a_repeated_key_is_a_data_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    broken = workspace.root / "in" / "repeated.json"
    broken.write_text(
        workspace.suite.read_text("utf-8").replace('"kind"', '"name": "x", "kind"', 1),
        "utf-8",
    )

    result = invoke(workspace.hypotheses_args(suite=broken))

    assert result.exit_code == 3, result.output
    assert_refused(workspace, seams, result, 3)


# --- the outputs file ------------------------------------------------------------------------------------


def test_a_hypotheses_file_that_is_missing_or_a_directory_is_a_data_error(
    workspace: EvalWorkspace, seams: Seams, tmp_path: Path
) -> None:
    missing = invoke(
        workspace.hypotheses_args("--system", "x", hypotheses=tmp_path / "none.jsonl")
    )
    directory = invoke(workspace.hypotheses_args("--system", "x", hypotheses=tmp_path))

    assert_one_error(missing, 3, "hypotheses file not found")
    assert_one_error(directory, 3, "hypotheses file not found")
    assert_refused(workspace, seams, directory, 3)


def test_a_hypotheses_file_that_misses_a_row_of_a_direction_is_a_data_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    lines = workspace.alpha.read_text("utf-8").splitlines()
    workspace.alpha.write_text("\n".join(lines[:-1]) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 3, result.output
    assert "eng_Latn-khk_Cyrl" in result.stderr
    assert_refused(workspace, seams, result, 3)


def test_a_hypotheses_file_with_an_unknown_id_is_a_data_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    lines = workspace.alpha.read_text("utf-8").splitlines()
    lines.append(lines[0].replace('"g01"', '"g99"'))
    workspace.alpha.write_text("\n".join(lines) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 3, result.output
    assert "g99" in result.stderr
    assert_refused(workspace, seams, result, 3)


# --- the export -------------------------------------------------------------------------------------------


def test_an_export_without_a_manifest_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    (workspace.export.directory / "manifest.json").unlink()

    result = invoke(workspace.model_args())

    assert_one_error(result, 2, "")
    assert "no manifest vouches for it" in result.stderr
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_an_export_with_a_flipped_byte_is_refused_naming_the_file(
    workspace: EvalWorkspace, seams: Seams, name: str
) -> None:
    file = workspace.export.directory / name
    data = bytearray(file.read_bytes())
    data[0] ^= 0xFF
    file.write_bytes(bytes(data))

    result = invoke(workspace.model_args())

    assert result.exit_code == 2, result.output
    assert name in result.stderr
    assert_refused(workspace, seams, result, 2)


def test_an_export_that_is_not_a_directory_is_a_config_error(
    workspace: EvalWorkspace, seams: Seams, tmp_path: Path
) -> None:
    result = invoke(workspace.model_args(model=tmp_path / "none"))

    assert result.exit_code == 2, result.output
    assert_refused(workspace, seams, result, 2)


@pytest.mark.parametrize("which", ["model_sha256", "manifest_sha256"])
def test_an_engine_that_loaded_other_files_than_the_ones_hashed_is_refused(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    import dataclasses

    engine = workspace.engine()
    engine.info = dataclasses.replace(engine.info, **{which: "f" * 64})
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, 2, "the engine loaded")
    assert engine.closed == 1
    assert engine.calls == []
    assert not workspace.out_dir.exists()


# --- the output directory -----------------------------------------------------------------------------------


def test_an_output_directory_holding_the_manifest_of_another_stage_is_refused_untouched(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    run_dir = run_dir_of(workspace.out_dir, "alpha")
    before = write_foreign_manifest(run_dir)

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 3, result.output
    assert "train_docify" in result.stderr
    assert result.stdout == ""
    assert (run_dir / "manifest.json").read_bytes() == before
    assert snapshot(workspace.out_dir) == {
        "golden": None,
        "golden/alpha": None,
        "golden/alpha/manifest.json": before,
    }
    assert seams.opens == []


def test_a_foreign_manifest_stops_a_model_run_before_the_engine_is_opened(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    write_foreign_manifest(run_dir_of(workspace.out_dir, workspace.export.record.name))

    result = invoke(workspace.model_args())

    assert result.exit_code == 3, result.output
    assert seams.opens == []


def test_an_output_directory_that_is_the_directory_of_the_suite_is_refused(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    result = invoke(workspace.hypotheses_args(out_dir=workspace.suite.parent))

    assert_one_error(result, 2, "--out-dir")
    assert "the directory of the input file" in result.stderr
    assert seams.opens == []
    assert not (workspace.suite.parent / "golden").exists()


def test_a_run_directory_that_holds_an_input_is_refused(
    workspace: EvalWorkspace, seams: Seams
) -> None:
    # `alpha` again from its own earlier output would replace the file it was read from.
    run_dir = run_dir_of(workspace.out_dir, "alpha")
    run_dir.mkdir(parents=True)
    shutil.copyfile(workspace.alpha, run_dir / "hypotheses.jsonl")

    result = invoke(
        workspace.hypotheses_args(
            "--system", "alpha", hypotheses=run_dir / "hypotheses.jsonl"
        )
    )

    assert_one_error(result, 2, "")
    assert "give this run another --system" in result.stderr
    assert snapshot(run_dir) == {"hypotheses.jsonl": workspace.alpha.read_bytes()}
