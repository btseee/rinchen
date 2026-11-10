"""`rinchen export parity`: what it refuses, with the exit code and the one-line error the
command documents (2 for a bad option, an export that is not vouched for, a checkpoint of
other weights than the export's, a path Eole reads specially, an output directory that is an
input's; 3 for missing or unreadable data and another stage's manifest; 4 when the parity
fails to run), and the order of its work: every input is hashed before it is read, the old
result and manifest go before the run, and the manifest is written last.

A refusal leaves the output directory as it was found and never starts the parity. The prints
and the files of a successful run are in `test_cli_export_parity.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import export_parity
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.manifest import store
from tests.support.corpus_rows import corpus_row, write_rows
from tests.support.distill_cli import write_foreign_manifest
from tests.support.fake_parity_eole import WEIGHTS_SHA256
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.parity_cli import (
    FakeStrictParity,
    ParityCall,
    ParityWorkspace,
    corpus_rows,
    install_parity,
    invoke,
    make_workspace,
)

GLOB_CHARACTERS = ("$", "[", "]", "*", "?")


@pytest.fixture
def workspace(tmp_path: Path) -> ParityWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def parity(monkeypatch: pytest.MonkeyPatch) -> FakeStrictParity:
    return install_parity(monkeypatch)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_refused(
    workspace: ParityWorkspace, parity: FakeStrictParity, result: Any, code: int
) -> None:
    assert result.exit_code == code, result.output
    assert result.stdout == ""
    assert parity.calls == []
    assert not workspace.out_dir.exists()


# --- options -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [("--max-lines", "0"), ("--threads", "0"), ("--timeout", "x")],
)
def test_a_usage_error_is_exit_2_and_nothing_runs(
    workspace: ParityWorkspace, parity: FakeStrictParity, arguments: tuple[str, str]
) -> None:
    result = invoke(workspace.args(*arguments))

    assert_refused(workspace, parity, result, 2)
    assert arguments[0] in result.stderr


@pytest.mark.parametrize(
    "option", ["--checkpoint", "--export", "--corpus", "--out-dir"]
)
def test_every_required_option_is_required(
    workspace: ParityWorkspace, parity: FakeStrictParity, option: str
) -> None:
    arguments = workspace.args()
    at = arguments.index(option)
    del arguments[at : at + 2]

    result = invoke(arguments)

    assert_refused(workspace, parity, result, 2)
    assert option in result.stderr


@pytest.mark.parametrize("timeout", ["0", "-5", "nan", "inf"])
def test_a_timeout_that_is_not_a_positive_finite_number_is_a_config_error(
    workspace: ParityWorkspace, parity: FakeStrictParity, timeout: str
) -> None:
    result = invoke(workspace.args("--timeout", timeout))

    assert_one_error(result, 2, "--timeout must be a positive, finite number")
    assert parity.calls == []
    assert not workspace.out_dir.exists()


def test_a_corpus_given_twice_is_refused_whatever_its_spelling(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    again = (
        workspace.corpus.parent
        / ".."
        / workspace.corpus.parent.name
        / workspace.corpus.name
    )

    result = invoke(workspace.args(corpora=[workspace.corpus, again]))

    assert_one_error(result, 2, f"--corpus {again} names the same file as --corpus")
    assert parity.calls == []


# --- the export ----------------------------------------------------------------------------------


def test_a_directory_that_is_no_export_is_refused_with_nothing_on_stdout(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    empty = workspace.root / "empty"
    empty.mkdir()

    result = invoke(workspace.args(export=empty))

    assert_one_error(result, 2, f"cannot use the export {empty}: no manifest vouches")
    assert_refused(workspace, parity, result, 2)


def test_an_export_of_the_format_of_d10_is_refused_with_the_remedy(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    old = workspace.root / "old"
    old.mkdir()
    (old / "export.json").write_text("{}", encoding="utf-8")

    result = invoke(workspace.args(export=old))

    assert_one_error(result, 2, f"cannot use the export {old}")
    assert "rinchen export convert" in result.stderr
    assert_refused(workspace, parity, result, 2)


def test_an_export_file_that_changed_after_its_manifest_is_refused(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    (workspace.export.directory / "model.bin").write_bytes(b"flipped")

    result = invoke(workspace.args())

    assert_one_error(result, 2, "cannot use the export")
    assert "model.bin" in result.stderr
    assert_refused(workspace, parity, result, 2)


# --- the checkpoint ------------------------------------------------------------------------------


def test_a_checkpoint_of_other_weights_than_the_export_s_is_refused_before_anything_is_made(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    workspace.weights.write_bytes(b"the weights of another model")

    result = invoke(workspace.args())

    assert_one_error(result, 2, "the checkpoint's weights")
    assert WEIGHTS_SHA256[:12] in result.stderr
    assert_refused(workspace, parity, result, 2)


@pytest.mark.parametrize("name", ["config.json", "vocab.json"])
def test_a_checkpoint_of_the_export_s_weights_but_another_config_or_vocabulary_is_refused(
    workspace: ParityWorkspace, parity: FakeStrictParity, name: str
) -> None:
    (workspace.checkpoint / name).write_bytes(b"the file of another model")

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"the checkpoint's {name} has SHA-256 ")
    assert "a parity of two different models says nothing" in result.stderr
    assert_refused(workspace, parity, result, 2)


@pytest.mark.parametrize("name", ["model.00.safetensors", "config.json", "vocab.json"])
def test_a_checkpoint_without_a_file_the_manifest_names_is_a_config_error_naming_it(
    workspace: ParityWorkspace, parity: FakeStrictParity, name: str
) -> None:
    (workspace.checkpoint / name).unlink()

    result = invoke(workspace.args())

    assert result.exit_code == 2, result.output
    assert str(workspace.checkpoint / name) in result.stderr
    assert "not found" in result.stderr
    assert parity.calls == []
    assert not workspace.out_dir.exists()


def test_a_checkpoint_that_is_not_a_directory_is_a_config_error(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    missing = workspace.root / "nowhere"

    result = invoke(workspace.args(checkpoint=missing))

    assert_one_error(result, 2, "checkpoint weights not found: ")
    assert parity.calls == []


@pytest.mark.parametrize("character", GLOB_CHARACTERS)
def test_a_character_eole_reads_specially_in_the_checkpoint_path_is_refused(
    workspace: ParityWorkspace, parity: FakeStrictParity, character: str
) -> None:
    checkpoint = workspace.root / f"ck{character}pt"
    workspace.checkpoint.rename(checkpoint)

    result = invoke(workspace.args(checkpoint=checkpoint))

    assert result.exit_code == 2, result.output
    assert "checkpoint directory" in result.stderr
    assert character in result.stderr
    assert parity.calls == []
    assert not workspace.out_dir.exists()


def test_a_dollar_in_the_output_directory_is_refused_and_a_glob_character_is_fine(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    refused = invoke(workspace.args(out_dir=workspace.root / "out$dir"))

    assert refused.exit_code == 2, refused.output
    assert "output directory" in refused.stderr
    assert parity.calls == []

    done = invoke(workspace.args(out_dir=workspace.root / "out[1]"))

    assert done.exit_code == 0, done.output
    assert (workspace.root / "out[1]" / "parity.json").is_file()


# --- the corpus ----------------------------------------------------------------------------------


def test_a_missing_corpus_is_a_data_error(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    missing = workspace.root / "absent.parquet"

    result = invoke(workspace.args(corpora=[missing]))

    assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert parity.calls == []
    assert not workspace.out_dir.exists()


def test_a_corpus_without_dev_rows_for_a_direction_of_the_model_is_a_data_error(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    only = write_rows(
        workspace.root / "short",
        "short",
        [
            corpus_row("a", "b", src_lang="eng_Latn", tgt_lang="khk_Cyrl", split="dev"),
        ],
    )

    result = invoke(workspace.args(corpora=[only]))

    assert_one_error(result, 3, "no usable dev rows for direction khk_Cyrl-eng_Latn")
    assert parity.calls == []
    assert not workspace.out_dir.exists()


# --- the output directory ------------------------------------------------------------------------


@pytest.mark.parametrize("input_name", ["corpus", "export", "checkpoint"])
def test_an_output_directory_that_is_the_directory_of_an_input_is_refused_and_untouched(
    workspace: ParityWorkspace, parity: FakeStrictParity, input_name: str
) -> None:
    directory = {
        "corpus": workspace.corpus.parent,
        "export": workspace.export.directory,
        "checkpoint": workspace.checkpoint,
    }[input_name]
    before = snapshot(directory)

    result = invoke(workspace.args(out_dir=directory))

    assert_one_error(
        result, 2, f"--out-dir {directory} is the directory of the input file "
    )
    assert snapshot(directory) == before
    assert parity.calls == []


def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_before_anything_is_read(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_foreign_manifest(workspace.out_dir)
    before = snapshot(workspace.out_dir)

    def read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the corpus was read before the foreign manifest was refused")

    monkeypatch.setattr(export_parity, "load_dev_rows", read)

    result = invoke(workspace.args())

    assert_one_error(
        result, 3, f"{workspace.out_dir} holds the manifest of stage 'train_docify'"
    )
    assert snapshot(workspace.out_dir) == before
    assert parity.calls == []


def test_an_output_directory_that_cannot_be_made_exits_1(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    blocked = workspace.root / "blocked"
    blocked.write_text("a file where the directory should be", encoding="utf-8")

    result = invoke(workspace.args(out_dir=blocked / "out"))

    assert result.exit_code == 1, result.output
    assert "cannot write the parity result under" in result.stderr
    assert parity.calls == []


# --- a parity that fails to run ------------------------------------------------------------------


def test_a_parity_that_fails_is_exit_4_with_no_manifest_and_no_result(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch, error=UpstreamError("eole predict exited with code 1"))

    result = invoke(workspace.args())

    assert_one_error(result, 4, "eole predict exited with code 1")
    assert result.stdout == ""
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_a_refusal_of_the_parity_itself_is_exit_2(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch, error=ConfigError("the target mn_Mong has no tag"))

    result = invoke(workspace.args())

    assert_one_error(result, 2, "the target mn_Mong has no tag")


def test_a_failed_rerun_leaves_neither_the_old_manifest_nor_the_old_result(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch)
    assert invoke(workspace.args()).exit_code == 0
    assert workspace.manifest.is_file()
    assert workspace.result.is_file()
    install_parity(monkeypatch, error=UpstreamError("eole predict timed out"))

    result = invoke(workspace.args())

    assert_one_error(result, 4, "eole predict timed out")
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_an_interrupt_exits_130_and_writes_no_manifest(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch, error=KeyboardInterrupt())

    result = invoke(workspace.args())

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_a_result_that_cannot_be_written_exits_1_and_leaves_no_manifest(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(export_parity, "replace_file", failing)

    result = invoke(workspace.args())

    assert_one_error(
        result, 1, f"cannot write the parity result under {workspace.out_dir}"
    )
    assert "disk full" in result.stderr
    assert not workspace.manifest.exists()


def test_a_file_of_the_parity_that_cannot_be_read_is_exit_3_not_a_write_failure(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(
        monkeypatch,
        error=DataError("cannot read the output of eole predict out/parity.tokens.out"),
    )

    result = invoke(workspace.args())

    assert_one_error(result, 3, "cannot read the output of eole predict")
    assert "cannot write" not in result.stderr
    assert not workspace.manifest.exists()
    assert not workspace.result.exists()


def test_a_file_of_the_parity_that_cannot_be_written_is_exit_1(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch, error=OSError("read-only file system"))

    result = invoke(workspace.args())

    assert_one_error(
        result, 1, f"cannot write the parity result under {workspace.out_dir}"
    )
    assert "read-only file system" in result.stderr
    assert not workspace.manifest.exists()


# --- hashing before reading ----------------------------------------------------------------------


def test_every_input_is_hashed_before_the_parity_starts(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real = export_parity.hash_input

    def hashing(path: Path, **kwargs: Any) -> Any:
        events.append(f"hash:{Path(path).name}")
        return real(path, **kwargs)

    def running(call: ParityCall) -> None:
        events.append("parity")

    monkeypatch.setattr(export_parity, "hash_input", hashing)
    parity.during = running

    assert invoke(workspace.args()).exit_code == 0

    assert events == [
        "hash:toy.parquet",
        "hash:model.00.safetensors",
        "hash:config.json",
        "hash:vocab.json",
        "parity",
    ]


def test_the_corpus_is_hashed_before_it_is_read(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = sha256(workspace.corpus)
    real = export_parity.load_dev_rows

    def reading(paths: Any, **kwargs: Any) -> Any:
        # Something else replaces the corpus between the hash and the read.
        extra = corpus_row("added", "row", split="dev")
        write_rows(workspace.corpus.parent, "toy", [*corpus_rows(), extra])
        return real(paths, **kwargs)

    monkeypatch.setattr(export_parity, "load_dev_rows", reading)

    invoke(workspace.args())

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert manifest.inputs[0].path == str(workspace.corpus)
    assert manifest.inputs[0].sha256 == before
    assert sha256(workspace.corpus) != before


def test_the_manifest_vouches_for_the_inputs_as_they_were_before_the_parity_read_them(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    weights_before = sha256(workspace.weights)
    model = workspace.export.directory / "model.bin"
    model_before = sha256(model)

    def swapping(call: ParityCall) -> None:
        workspace.weights.write_bytes(b"swapped while the parity ran")
        model.write_bytes(b"swapped as well")

    parity.during = swapping

    assert invoke(workspace.args()).exit_code == 0

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert recorded[workspace.weights] == weights_before
    assert recorded[model] == model_before
    assert parity.calls[0].weights_sha256 == weights_before


def test_the_old_manifest_and_result_are_gone_before_the_parity_runs(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
) -> None:
    assert invoke(workspace.args()).exit_code == 0
    seen: list[tuple[bool, bool]] = []

    def looking(call: ParityCall) -> None:
        seen.append((workspace.manifest.exists(), workspace.result.exists()))

    parity.during = looking

    assert invoke(workspace.args()).exit_code == 0

    assert seen == [(False, False)]


def test_the_manifest_is_written_after_the_result_it_lists(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = export_parity.write_manifest
    present: list[bool] = []

    def checking(out_dir: Path, **kwargs: Any) -> Path:
        present.extend(Path(item.path).is_file() for item in kwargs["outputs"])
        present.append(not (out_dir / "manifest.json").exists())
        return real(out_dir, **kwargs)

    monkeypatch.setattr(export_parity, "write_manifest", checking)

    assert invoke(workspace.args()).exit_code == 0

    assert present == [True, True]
