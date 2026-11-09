"""`rinchen train docify`: what it refuses. A bad option, a missing or invalid input, an
output directory it may not use, a corpus with no window to cut and a write that fails each
end with the exit code and the one-line error the command documents, and leave the output
directory as it was found.

What a successful run prints and writes is in `test_cli_train_docify.py`. Nothing here
starts Eole.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rinchen.cli import train_common, train_docify
from rinchen.cli.main import app
from rinchen.errors import RinchenError
from tests.support.doc_cli import (
    PAIRS,
    DocifyWorkspace,
    hand_built_rows,
    made_dir,
    make_docify_workspace,
    run_ok,
)
from tests.support.launch_cli import assert_one_error, snapshot

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path) -> DocifyWorkspace:
    return make_docify_workspace(tmp_path)


def test_a_dry_run_is_refused_for_a_directory_with_another_stages_manifest(
    workspace: DocifyWorkspace,
) -> None:
    train_common.write_manifest(
        made_dir(workspace.out_dir),
        stage="train_launch",
        config_sha256="0" * 64,
        seed=1,
        inputs=[],
        outputs=[],
        metrics={},
        packages=[],
        status={"launch": "ok"},
    )

    result = runner.invoke(app, workspace.args("--dry-run"))

    assert_one_error(
        result,
        3,
        f"{workspace.out_dir} holds the manifest of stage 'train_launch', not of "
        "'train_docify'",
    )


def test_a_budget_above_what_filtertoolong_keeps_is_a_config_error(
    workspace: DocifyWorkspace,
) -> None:
    result = runner.invoke(app, workspace.args("--max-pieces", "63"))

    assert_one_error(result, 2, "--max-pieces 63 is above what the base config keeps")
    assert "src_seq_length 64" in result.stderr
    assert "tgt_seq_length 64" in result.stderr
    assert "62" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_budget_below_the_shortest_sentence_is_a_data_error_that_writes_nothing(
    workspace: DocifyWorkspace,
) -> None:
    result = runner.invoke(app, workspace.args("--max-pieces", "3", *PAIRS))

    assert_one_error(result, 3, "no document windows")
    assert "--max-pieces 3" in result.stderr
    assert "too_long" in result.stderr
    assert not workspace.out_dir.exists()


def test_the_same_refusal_comes_from_a_dry_run(workspace: DocifyWorkspace) -> None:
    result = runner.invoke(app, workspace.args("--max-pieces", "3", "--dry-run"))

    assert_one_error(result, 3, "no document windows")
    assert not workspace.out_dir.exists()


def test_a_corpus_with_no_training_row_is_a_data_error(tmp_path: Path) -> None:
    rows = [row for row in hand_built_rows() if row["split"] != "train"]
    workspace = make_docify_workspace(tmp_path, rows=rows)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(result, 3, "no document windows")
    assert "not_train" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_corpus_with_no_dev_window_is_a_data_error_that_writes_nothing(
    tmp_path: Path,
) -> None:
    rows = [row for row in hand_built_rows() if row["split"] != "dev"]
    workspace = make_docify_workspace(tmp_path, rows=rows)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(result, 3, "no dev document windows")
    assert not workspace.out_dir.exists()


def test_a_minimum_above_the_maximum_is_a_config_error(
    workspace: DocifyWorkspace,
) -> None:
    result = runner.invoke(app, workspace.args("--min-size", "3", "--max-size", "2"))

    assert_one_error(result, 2, "--min-size 3 is above --max-size 2")
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("temperature", ["0", "-1", "inf", "nan", "1e9"])
def test_a_temperature_that_cannot_weight_the_directions_is_a_config_error(
    workspace: DocifyWorkspace, temperature: str
) -> None:
    result = runner.invoke(app, workspace.args("--temperature", temperature, *PAIRS))

    assert_one_error(result, 2, "")
    assert "temperature" in result.stderr
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    "option",
    ["--min-size", "--max-size", "--max-pieces", "--dev-max-windows", "--weight-total"],
)
def test_a_count_below_one_is_a_usage_error(
    workspace: DocifyWorkspace, option: str
) -> None:
    result = runner.invoke(app, workspace.args(option, "0"))

    assert result.exit_code == 2, result.output
    assert "Invalid value" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_weight_total_below_the_number_of_corpora_is_a_config_error(
    workspace: DocifyWorkspace,
) -> None:
    result = runner.invoke(app, workspace.args("--weight-total", "1", *PAIRS))

    assert_one_error(result, 2, "weight_total 1 is below the number of corpora (2)")
    assert not workspace.out_dir.exists()


def test_a_corpus_given_twice_is_refused(workspace: DocifyWorkspace) -> None:
    result = runner.invoke(
        app, workspace.args(corpora=[workspace.corpus, workspace.corpus])
    )

    assert_one_error(result, 2, f"--corpus {workspace.corpus} names the same file")
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("fault", ["train_from", "reset_optim", "data"])
def test_a_base_config_that_cannot_start_a_continuation_is_refused_before_any_write(
    tmp_path: Path, fault: str
) -> None:
    def edit(cfg: dict[str, Any]) -> None:
        if fault == "data":
            cfg["data"] = {
                "other.auth": {
                    "path_src": "a.src",
                    "path_tgt": "a.tgt",
                    "transforms": [],
                    "weight": 1,
                    "src_prefix": "",
                    "tgt_prefix": "",
                }
            }
        else:
            cfg["training"][fault] = "all" if fault == "reset_optim" else "/x/step_1"

    workspace = make_docify_workspace(tmp_path, edit_base=edit)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert fault.replace("data", "other.auth") in result.stderr
    assert not workspace.out_dir.exists()


def test_an_output_directory_holding_another_stages_manifest_is_refused_untouched(
    workspace: DocifyWorkspace,
) -> None:
    train_common.write_manifest(
        made_dir(workspace.out_dir),
        stage="train_prepare",
        config_sha256="0" * 64,
        seed=1,
        inputs=[],
        outputs=[],
        metrics={},
        packages=[],
        status={"prepare": "ok"},
    )
    before = snapshot(workspace.out_dir)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(
        result,
        3,
        f"{workspace.out_dir} holds the manifest of stage 'train_prepare', not of "
        "'train_docify'",
    )
    assert snapshot(workspace.out_dir) == before


@pytest.mark.parametrize("character", ["$", "[", "]", "*", "?"])
def test_a_character_eole_would_expand_or_read_as_a_pattern_is_refused_in_the_directory(
    workspace: DocifyWorkspace, character: str
) -> None:
    out_dir = workspace.root / f"out{character}x"

    result = runner.invoke(app, workspace.args(out_dir=out_dir))

    assert result.exit_code == 2, result.output
    assert "output directory" in result.stderr
    assert repr(character) in result.stderr
    assert not out_dir.exists()


@pytest.mark.parametrize("flags", [[], ["--dry-run"]])
def test_a_character_in_a_parent_of_the_directory_is_refused_as_well(
    workspace: DocifyWorkspace, flags: list[str]
) -> None:
    out_dir = workspace.root / "a*b" / "out"

    result = runner.invoke(app, workspace.args(*flags, out_dir=out_dir))

    assert result.exit_code == 2, result.output
    assert not out_dir.exists()


def test_a_dollar_sign_in_the_spm_model_path_is_refused(
    workspace: DocifyWorkspace,
) -> None:
    model = workspace.root / "a$b.model"
    model.write_bytes(workspace.spm_model.read_bytes())

    result = runner.invoke(app, workspace.args(spm_model=model))

    assert result.exit_code == 2, result.output
    assert "$" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_write_failure_exits_with_one_and_leaves_no_manifest_not_even_the_old_one(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ok(workspace.args(*PAIRS))

    def full_disk(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(train_docify, "build_doc_corpora", full_disk)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(
        result, 1, f"cannot write the document corpora under {workspace.out_dir}: "
    )
    assert not workspace.manifest.exists()


def test_an_output_directory_that_cannot_be_made_exits_with_one(
    workspace: DocifyWorkspace,
) -> None:
    (workspace.root / "blocked").write_text("a file, not a directory", encoding="utf-8")

    result = runner.invoke(
        app, workspace.args(*PAIRS, out_dir=workspace.root / "blocked" / "out")
    )

    assert_one_error(result, 1, "cannot write the document corpora under")


def test_a_domain_error_while_writing_exits_with_its_code_and_leaves_no_manifest(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise RinchenError("something else")

    monkeypatch.setattr(train_docify, "write_doc_validation", failing)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(result, 1, "something else")
    assert not workspace.manifest.exists()


def test_an_interrupt_exits_130_and_leaves_no_manifest(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_docify, "build_doc_corpora", interrupted)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry run"])
@pytest.mark.parametrize(
    ("which", "exit_code", "what"),
    [
        ("corpus", 3, "corpus file"),
        ("base_config", 2, "base Eole config"),
        ("spm_model", 2, "SentencePiece model"),
    ],
)
def test_a_missing_input_is_reported_alike_by_a_run_and_a_dry_run(
    workspace: DocifyWorkspace,
    dry_run: bool,
    which: str,
    exit_code: int,
    what: str,
) -> None:
    missing = workspace.root / "nowhere"
    replaced: dict[str, Any] = {
        "corpus": {"corpora": [missing]},
        "base_config": {"config": missing},
        "spm_model": {"spm_model": missing},
    }[which]

    result = runner.invoke(
        app, workspace.args(*(["--dry-run"] if dry_run else []), **replaced)
    )

    assert_one_error(result, exit_code, f"{what} not found: {missing}")
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("key", ["src_seq_length", "tgt_seq_length"])
@pytest.mark.parametrize("value", [None, 0, -3, True, "64", 64.0])
def test_a_filtertoolong_length_that_is_not_a_positive_integer_is_refused(
    tmp_path: Path, key: str, value: object
) -> None:
    def edit(cfg: dict[str, Any]) -> None:
        cfg["transforms_configs"]["filtertoolong"][key] = value

    workspace = make_docify_workspace(tmp_path, edit_base=edit)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(
        result,
        2,
        f"invalid base Eole config: transforms_configs.filtertoolong.{key}: must be a "
        "positive integer",
    )
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("tgt_seq_length", [2, 1])
def test_a_base_that_leaves_no_room_for_a_window_is_refused(
    tmp_path: Path, tgt_seq_length: int
) -> None:
    def edit(cfg: dict[str, Any]) -> None:
        cfg["transforms_configs"]["filtertoolong"]["tgt_seq_length"] = tgt_seq_length

    workspace = make_docify_workspace(tmp_path, edit_base=edit)

    result = runner.invoke(app, workspace.args(*PAIRS))

    assert_one_error(result, 2, "the base Eole config keeps no window")
    assert f"tgt_seq_length {tgt_seq_length}" in result.stderr
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("which", ["corpus", "base_config", "spm_model"])
def test_the_output_directory_may_not_be_the_directory_of_an_input(
    workspace: DocifyWorkspace, which: str
) -> None:
    directory = getattr(workspace, which).parent
    before = snapshot(workspace.root)

    result = runner.invoke(app, workspace.args(out_dir=directory))

    assert_one_error(
        result, 2, f"--out-dir {directory} is the directory of the input file"
    )
    assert snapshot(workspace.root) == before


def test_a_glob_character_in_the_working_directory_refuses_a_relative_directory(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    working = made_dir(workspace.root / "w[1]d")
    monkeypatch.chdir(working)

    result = runner.invoke(app, workspace.args(out_dir=Path("out")))

    assert result.exit_code == 2, result.output
    assert "output directory" in result.stderr
    assert "'['" in result.stderr
    assert not (working / "out").exists()


@pytest.mark.parametrize("temperature", ["0", "-1", "inf", "nan"])
def test_a_temperature_that_is_not_positive_and_finite_is_a_config_error(
    workspace: DocifyWorkspace, temperature: str
) -> None:
    result = runner.invoke(app, workspace.args("--temperature", temperature, *PAIRS))

    assert_one_error(result, 2, "--temperature must be a positive, finite number")
    assert not workspace.out_dir.exists()


def test_a_temperature_so_large_that_the_weights_overflow_is_a_config_error(
    workspace: DocifyWorkspace,
) -> None:
    result = runner.invoke(app, workspace.args("--temperature", "1e9", *PAIRS))

    assert_one_error(
        result, 2, "--temperature 1000000000.0: temperature 1000000000.0 is too large"
    )
    assert not workspace.out_dir.exists()


def test_a_bad_temperature_is_reported_before_any_input_is_read(
    workspace: DocifyWorkspace,
) -> None:
    missing = workspace.root / "nowhere.parquet"

    result = runner.invoke(app, workspace.args("--temperature", "0", corpora=[missing]))

    assert_one_error(result, 2, "--temperature must be a positive, finite number")
