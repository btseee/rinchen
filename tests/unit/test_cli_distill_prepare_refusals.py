"""`rinchen distill prepare`: what it refuses, with the exit code and the one-line error the
command documents (2 for a bad option, config, provenance or output directory, 3 for missing
or unreadable data and another stage's manifest), and the order of its work: every input is
hashed before anything reads it (the KD files once), the old manifest goes only when the
first file is about to be written, and the manifest is written last.

A refusal leaves everything as it was found: no file written and no directory made (an output
directory that already held a run keeps its manifest). The statistics and the files of a
successful run are in `test_cli_distill_prepare.py`.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import rinchen.train.distill_corpus as distill_corpus_module
import rinchen.train.distill_prepare as library
from rinchen.cli import distill_prepare
from rinchen.corpus.schema import corpus_frame
from rinchen.manifest import stage as stage_module
from rinchen.manifest import store
from tests.support.distill_cli import (
    TOY_OPTIONS,
    PrepareWorkspace,
    invoke,
    make_prepare_workspace,
    run_ok,
    write_foreign_manifest,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.planted_kd import write_kd_files
from tests.support.planted_student import Inputs, base_with, build_inputs
from tests.support.step_dirs import flip_one_byte
from tests.support.tiny_parallel import build_toy, write_corpus_parquet

GLOB_CHARACTERS = ("$", "[", "]", "*", "?")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("cli_distill_prepare_refusals"))


@pytest.fixture
def workspace(tmp_path: Path, inputs: Inputs) -> PrepareWorkspace:
    return make_prepare_workspace(tmp_path, inputs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Spies:
    """Records, in order, what the command hashes and what it hands to the library: the
    files it hashes (by name), `prepare_student` and the removal of the old manifest."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[str] = []
        self._wrap(monkeypatch, distill_prepare, "hash_input", self._hash_label)
        self._wrap(
            monkeypatch, distill_prepare, "prepare_student", lambda *a, **k: "read"
        )
        self._wrap(
            monkeypatch, distill_prepare, "begin_commit", lambda *a, **k: "commit"
        )
        self._wrap(
            monkeypatch, distill_prepare, "write_manifest", lambda *a, **k: "manifest"
        )

    @staticmethod
    def _hash_label(path: Path, **_: Any) -> str:
        return f"hash:{Path(path).name}"

    def _wrap(
        self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str, label: Any
    ) -> None:
        real = getattr(module, name)

        def spying(*args: Any, **kwargs: Any) -> Any:
            self.events.append(label(*args, **kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, spying)


# --- options, before anything is read -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1",
        "1:",
        ":1",
        "0:1",
        "1:0",
        "-1:1",
        "+1:1",
        "1.5:1",
        "1: 1",
        "01:1",
        "1:1:1",
        "3/1",
        f"{chr(0x0661)}:1",
        "10001:1",
    ],
)
def test_a_malformed_mix_is_refused_before_any_file_is_read(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    spies = Spies(monkeypatch)

    result = invoke(workspace.args(*TOY_OPTIONS, "--mix", text))

    assert_one_error(result, 2, f"--mix {text!r} is not K:O")
    assert spies.events == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    ("option", "value", "named"),
    [
        ("--temperature", "0", "--temperature"),
        ("--temperature", "-1", "--temperature"),
        ("--temperature", "nan", "--temperature"),
        ("--temperature", "inf", "--temperature"),
        ("--bt-ratio", "-0.5", "--bt-ratio"),
        ("--bt-ratio", "nan", "--bt-ratio"),
        ("--bt-ratio", "inf", "--bt-ratio"),
    ],
)
def test_a_temperature_or_ratio_that_is_not_a_usable_number_is_refused_before_any_file_is_read(
    workspace: PrepareWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
    named: str,
) -> None:
    spies = Spies(monkeypatch)

    result = invoke(workspace.args(option, value))

    assert_one_error(result, 2, named)
    assert spies.events == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--dev-max-lines", "0"),
        ("--weight-total", "0"),
        ("--tag-placement", "both"),
        ("--bt-ratio", "many"),
        ("--seed", "x"),
    ],
)
def test_a_usage_error_is_exit_2_and_nothing_is_read(
    workspace: PrepareWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, str],
) -> None:
    spies = Spies(monkeypatch)

    result = invoke(workspace.args(*arguments))

    assert result.exit_code == 2, result.output
    assert arguments[0] in result.stderr
    assert spies.events == []
    assert not workspace.out_dir.exists()


def test_a_temperature_so_large_that_the_weights_overflow_is_a_config_error(
    workspace: PrepareWorkspace,
) -> None:
    result = invoke(workspace.args("--temperature", "100000"))

    assert_one_error(result, 2, "cannot plan the student run: temperature 100000.0")
    assert not workspace.out_dir.exists()


def test_a_weight_total_below_the_number_of_corpora_is_a_config_error(
    workspace: PrepareWorkspace,
) -> None:
    result = invoke(workspace.args(*TOY_OPTIONS, "--weight-total", "3"))

    assert result.exit_code == 2, result.output
    assert "weight_total" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_corpus_given_twice_is_refused_whatever_its_spelling(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    spies = Spies(monkeypatch)
    again = (
        workspace.corpus.parent
        / ".."
        / workspace.corpus.parent.name
        / workspace.corpus.name
    )

    result = invoke(workspace.args(corpora=[workspace.corpus, again]))

    assert_one_error(result, 2, f"--corpus {again} names the same file as --corpus")
    assert spies.events == []


def test_a_kd_file_given_twice_is_refused_and_named_as_kd(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    spies = Spies(monkeypatch)
    again = workspace.kd.parent / ".." / "kd" / workspace.kd.name

    result = invoke(workspace.args(kd_files=[workspace.kd, again]))

    assert_one_error(
        result, 2, f"--kd {again} names the same file as --kd {workspace.kd}"
    )
    assert "every row of it twice" in result.stderr
    assert spies.events == []


@pytest.mark.parametrize("character", GLOB_CHARACTERS)
def test_a_character_eole_reads_specially_in_the_output_directory_is_refused(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch, character: str
) -> None:
    spies = Spies(monkeypatch)
    out = workspace.root / f"out{character}dir"

    result = invoke(workspace.args(out_dir=out))

    assert result.exit_code == 2, result.output
    assert "output directory" in result.stderr
    assert character in result.stderr
    assert spies.events == []
    assert not out.exists()


@pytest.mark.parametrize("character", GLOB_CHARACTERS)
def test_a_character_eole_reads_specially_in_the_spm_model_path_is_refused(
    workspace: PrepareWorkspace,
    inputs: Inputs,
    monkeypatch: pytest.MonkeyPatch,
    character: str,
) -> None:
    spies = Spies(monkeypatch)
    model = workspace.root / f"spm{character}dir" / "toy.model"
    model.parent.mkdir()
    shutil.copyfile(inputs.spm, model)

    result = invoke(workspace.args(spm_model=model))

    assert result.exit_code == 2, result.output
    assert "SentencePiece model path" in result.stderr
    assert character in result.stderr
    assert spies.events == []
    assert not workspace.out_dir.exists()


# --- missing inputs ---------------------------------------------------------------------------


def test_a_missing_corpus_is_a_data_error(workspace: PrepareWorkspace) -> None:
    missing = workspace.root / "absent.parquet"

    result = invoke(workspace.args(corpora=[missing]))

    assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert not workspace.out_dir.exists()


def test_a_missing_kd_file_is_a_data_error(workspace: PrepareWorkspace) -> None:
    missing = workspace.root / "absent_kd.parquet"

    result = invoke(workspace.args(kd_files=[missing]))

    assert_one_error(result, 3, f"KD file not found: {missing}")
    assert not workspace.out_dir.exists()


def test_a_missing_spm_model_is_a_config_error(workspace: PrepareWorkspace) -> None:
    missing = workspace.root / "absent.model"

    result = invoke(workspace.args(spm_model=missing))

    assert_one_error(result, 2, f"SentencePiece model not found: {missing}")
    assert not workspace.out_dir.exists()


def test_a_missing_base_config_is_a_config_error(workspace: PrepareWorkspace) -> None:
    missing = workspace.root / "absent.yaml"

    result = invoke(workspace.args(config=missing))

    assert_one_error(result, 2, f"base Eole config not found: {missing}")
    assert not workspace.out_dir.exists()


# --- what the KD files must be -----------------------------------------------------------------


def test_a_kd_file_without_its_manifest_is_a_config_error_and_nothing_is_written(
    workspace: PrepareWorkspace,
) -> None:
    (workspace.kd.parent / "manifest.json").unlink()

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(
        result, 2, f"KD file {workspace.kd} has no manifest.json beside it"
    )
    assert not workspace.out_dir.exists()


def test_a_kd_file_with_a_changed_byte_is_a_config_error_and_nothing_is_written(
    workspace: PrepareWorkspace,
) -> None:
    flip_one_byte(workspace.kd)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(result, 2, f"KD file {workspace.kd}: its SHA-256 is ")
    assert "the manifest vouches for" in result.stderr
    assert not workspace.out_dir.exists()


def test_the_manifest_of_another_stage_does_not_vouch_for_a_kd_file(
    workspace: PrepareWorkspace,
) -> None:
    write_foreign_manifest(workspace.kd.parent, stage="synth")

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(
        result, 2, f"KD file {workspace.kd}: the manifest beside it is of stage 'synth'"
    )
    assert not workspace.out_dir.exists()


def test_a_kd_manifest_that_cannot_be_read_is_a_data_error(
    workspace: PrepareWorkspace,
) -> None:
    (workspace.kd.parent / "manifest.json").write_text("not json", encoding="utf-8")

    result = invoke(workspace.args(*TOY_OPTIONS))

    manifest = workspace.kd.parent / "manifest.json"
    assert result.exit_code == 3, result.output
    # The manifest store's own message spans several lines (what the parser found wrong).
    assert result.stderr.startswith(f"error: unreadable manifest {manifest}: ")
    assert "Traceback" not in result.output
    assert not workspace.out_dir.exists()


def test_a_kd_file_with_no_row_that_trains_is_a_data_error(
    tmp_path: Path, inputs: Inputs
) -> None:
    rows = [
        {**row, "origin": "ft", "corpus": "ft:toy", "split": "dev"}
        for row in build_toy().train
    ]
    kd = write_kd_files(tmp_path / "dev_only_kd", corpus_frame(rows))
    workspace = make_prepare_workspace(tmp_path / "w", inputs, kd=kd)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(result, 3, "no KD rows: 3600 rows read, 0 kept")
    assert not workspace.out_dir.exists()


# --- the base config and the student's model -------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"train_from": "step_100"}, "train_from"),
        ({"reset_optim": "all"}, "reset_optim"),
    ],
)
def test_a_base_that_starts_from_a_checkpoint_is_a_config_error_naming_the_key(
    workspace: PrepareWorkspace, changes: dict[str, Any], named: str
) -> None:
    path = base_with(workspace.root, **changes)

    result = invoke(workspace.args(*TOY_OPTIONS, config=path))

    assert_one_error(result, 2, f"invalid base Eole config {path}: training.{named}")
    assert "a student starts from random initialisation" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_base_with_corpora_of_its_own_is_a_config_error(
    workspace: PrepareWorkspace,
) -> None:
    other = base_with(workspace.root, data={"x": {"path_src": "a", "path_tgt": "b"}})

    result = invoke(workspace.args(*TOY_OPTIONS, config=other))

    assert_one_error(result, 2, f"invalid base Eole config {other}: data:")
    assert not workspace.out_dir.exists()


def test_a_spm_model_that_lacks_the_kd_tag_is_a_config_error_naming_the_tag(
    workspace: PrepareWorkspace, inputs: Inputs
) -> None:
    result = invoke(workspace.args(*TOY_OPTIONS, spm_model=inputs.no_ft_model))

    assert result.exit_code == 2, result.output
    assert "<ft>" in result.stderr
    assert "origin tag" in result.stderr
    assert not workspace.out_dir.exists()


def test_without_the_kd_tag_a_model_that_lacks_it_is_fine(
    workspace: PrepareWorkspace, inputs: Inputs
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS, "--no-kd-tag", spm_model=inputs.no_ft_model))

    assert workspace.manifest.is_file()


def test_a_stated_vocabulary_size_below_the_models_pieces_is_a_config_error_naming_both_numbers(
    workspace: PrepareWorkspace,
) -> None:
    small = base_with(workspace.root, src_vocab_size=600)

    result = invoke(workspace.args(*TOY_OPTIONS, config=small))

    assert result.exit_code == 2, result.output
    assert "600" in result.stderr
    assert "641" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_corpus_without_a_training_row_is_a_data_error(
    tmp_path: Path, inputs: Inputs
) -> None:
    workspace = make_prepare_workspace(tmp_path, inputs)
    dev_only = write_corpus_parquet(build_toy().dev, tmp_path / "dev_only.parquet")

    result = invoke(workspace.args(*TOY_OPTIONS, corpora=[dev_only]))

    assert_one_error(result, 3, "no training rows: ")
    assert not workspace.out_dir.exists()


# --- the output directory ------------------------------------------------------------------------


@pytest.mark.parametrize("input_name", ["corpus", "kd", "base", "spm", "kd manifest"])
def test_an_output_directory_that_is_the_directory_of_an_input_is_refused_and_untouched(
    workspace: PrepareWorkspace, input_name: str
) -> None:
    path = {
        "corpus": workspace.corpus,
        "kd": workspace.kd,
        "base": workspace.base,
        "spm": workspace.spm_model,
        "kd manifest": workspace.kd.parent / "manifest.json",
    }[input_name]
    before = snapshot(path.parent)

    result = invoke(workspace.args(*TOY_OPTIONS, out_dir=path.parent))

    assert_one_error(
        result, 2, f"--out-dir {path.parent} is the directory of the input file "
    )
    assert snapshot(path.parent) == before


def test_an_output_directory_that_links_to_an_inputs_directory_is_refused(
    workspace: PrepareWorkspace,
) -> None:
    link = workspace.root / "link_to_kd"
    link.symlink_to(workspace.kd.parent, target_is_directory=True)
    before = snapshot(workspace.kd.parent)

    result = invoke(workspace.args(*TOY_OPTIONS, out_dir=link))

    assert_one_error(result, 2, f"--out-dir {link} is the directory of the input file ")
    assert snapshot(workspace.kd.parent) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_before_anything_is_read(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    write_foreign_manifest(workspace.out_dir)
    before = snapshot(workspace.out_dir)
    spies = Spies(monkeypatch)

    result = invoke(workspace.args(*TOY_OPTIONS, *(["--dry-run"] if dry_run else [])))

    assert_one_error(
        result, 3, f"{workspace.out_dir} holds the manifest of stage 'train_docify'"
    )
    assert snapshot(workspace.out_dir) == before
    assert "read" not in spies.events


def test_an_input_that_cannot_be_read_is_a_data_error_not_a_write_error(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args: Any, **kwargs: Any) -> Any:
        raise OSError("input/output error")

    monkeypatch.setattr(library, "load_training_rows", unreadable)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(
        result, 3, "cannot read an input of the student run: input/output error"
    )
    assert not workspace.out_dir.exists()


# --- hashing before reading ----------------------------------------------------------------------


def test_every_input_is_hashed_before_the_library_reads_anything(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    spies = Spies(monkeypatch)

    run_ok(workspace.args(*TOY_OPTIONS))

    assert spies.events == [
        "hash:base.yaml",
        "hash:toy.parquet",
        "hash:kd.parquet",
        "hash:manifest.json",
        "hash:toy.model",
        "read",
        "commit",
        "manifest",
    ]


def test_a_dry_run_hashes_the_inputs_and_reads_but_commits_and_writes_nothing(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    spies = Spies(monkeypatch)

    run_ok(workspace.args(*TOY_OPTIONS, "--dry-run"))

    assert spies.events == [
        "hash:base.yaml",
        "hash:toy.parquet",
        "hash:kd.parquet",
        "hash:manifest.json",
        "hash:toy.model",
        "read",
    ]


def test_the_kd_file_is_hashed_once_and_the_loader_does_not_hash_it_again(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[Path] = []
    real = stage_module.sha256_of

    def counting(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(stage_module, "sha256_of", counting)
    monkeypatch.setattr(
        distill_corpus_module,
        "sha256_of",
        lambda path: pytest.fail(f"the loader hashed {path} again"),
    )

    run_ok(workspace.args(*TOY_OPTIONS))

    assert hashed.count(workspace.kd) == 1
    assert hashed.count(workspace.kd.parent / "manifest.json") == 1


def rewrite_parquet(path: Path) -> None:
    pl.read_parquet(path).write_parquet(path, compression="uncompressed")


def append_a_comment(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")


@pytest.mark.parametrize("which", ["corpus", "base config"])
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    if which == "corpus":
        # A copy, so that rewriting it leaves the shared toy corpus alone.
        path = workspace.root / "corpus.parquet"
        shutil.copyfile(workspace.corpus, path)
        arguments = workspace.args(*TOY_OPTIONS, corpora=[path])
        rewrite = rewrite_parquet
    else:
        path = workspace.base
        arguments = workspace.args(*TOY_OPTIONS)
        rewrite = append_a_comment
    before = sha256(path)
    real = distill_prepare.prepare_student
    changed: list[bool] = []

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        if not changed:
            changed.append(True)
            rewrite(path)
        return real(*args, **kwargs)

    monkeypatch.setattr(distill_prepare, "prepare_student", after_a_change)

    run_ok(arguments)

    assert sha256(path) != before
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert recorded[path] == before


# --- the commit: the old manifest goes late, the new one comes last ---------------------------------


def previous_run(workspace: PrepareWorkspace) -> dict[str, bytes | str | None]:
    run_ok(workspace.args(*TOY_OPTIONS))
    return snapshot(workspace.out_dir)


def test_a_refusal_after_a_run_leaves_the_earlier_run_and_its_manifest_untouched(
    workspace: PrepareWorkspace,
) -> None:
    before = previous_run(workspace)
    flip_one_byte(workspace.kd)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert result.exit_code == 2, result.output
    assert snapshot(workspace.out_dir) == before


def test_a_failure_while_writing_leaves_no_manifest_not_even_the_old_one(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_run(workspace)

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(library, "write_validation", failing)

    result = invoke(workspace.args(*TOY_OPTIONS, "--mix", "3:1"))

    assert_one_error(
        result, 1, f"cannot write the student run under {workspace.out_dir}"
    )
    assert "disk full" in result.stderr
    assert not workspace.manifest.exists()


def test_a_failure_while_writing_the_validation_files_leaves_no_config_and_no_manifest(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(library, "write_validation", failing)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(
        result, 1, f"cannot write the student run under {workspace.out_dir}"
    )
    assert not workspace.config.exists()
    assert not workspace.manifest.exists()
    # The corpora written before the failure stay: they are reused by the next run.
    assert (workspace.out_dir / "data" / "kd" / "eng_Latn-khk_Cyrl.kd.src").is_file()


def test_the_manifest_is_written_after_every_output_it_lists(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = distill_prepare.write_manifest
    present: list[bool] = []

    def checking(out_dir: Path, **kwargs: Any) -> Path:
        present.extend(Path(item.path).is_file() for item in kwargs["outputs"])
        present.append(not (out_dir / "manifest.json").exists())
        return real(out_dir, **kwargs)

    monkeypatch.setattr(distill_prepare, "write_manifest", checking)

    run_ok(workspace.args(*TOY_OPTIONS))

    assert len(present) == 3 + 12 * 3 + 1
    assert all(present)


def test_a_failure_to_write_the_manifest_exits_1(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("read-only file system")

    monkeypatch.setattr(distill_prepare, "write_manifest", failing)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert_one_error(
        result, 1, f"cannot write the student run under {workspace.out_dir}"
    )
    assert "read-only file system" in result.stderr
    assert not workspace.manifest.exists()


def test_an_output_directory_that_cannot_be_made_exits_1(
    workspace: PrepareWorkspace,
) -> None:
    blocked = workspace.root / "blocked"
    blocked.write_text("a file where the directory should be", encoding="utf-8")

    result = invoke(workspace.args(*TOY_OPTIONS, out_dir=blocked / "out"))

    assert result.exit_code == 1, result.output
    assert "cannot write the student run under" in result.stderr


def test_an_interrupt_exits_130_and_writes_no_manifest(
    workspace: PrepareWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(library, "write_validation", interrupted)

    result = invoke(workspace.args(*TOY_OPTIONS))

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()
