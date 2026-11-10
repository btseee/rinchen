"""`rinchen distill generate`: what it refuses, with the exit code and the one-line error the
command documents (2 for a bad option, config or export, 3 for unreadable or empty data), and
that every input is hashed before anything reads it.

A refusal leaves the output directory as it was found, builds no teacher and translates
nothing. The teacher is a scripted fake behind the command's `build_teacher` seam, or the real
`TeacherTranslator` over a scripted CTranslate2 where the refusal is its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.cli import distill_generate, train_common
from rinchen.errors import DataError
from rinchen.export.record import EXPORT_FILES
from rinchen.manifest import store
from tests.support.corpus_rows import corpus_row
from tests.support.distill_cli import (
    GenerateWorkspace,
    TeacherSeam,
    install_ct2_teacher,
    install_teacher,
    invoke,
    make_generate_workspace,
    run_ok,
    write_selection_config,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.step_dirs import flip_one_byte, sha256


@pytest.fixture
def workspace(tmp_path: Path) -> GenerateWorkspace:
    return make_generate_workspace(tmp_path)


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> TeacherSeam:
    return install_teacher(monkeypatch)


def assert_nothing_done(workspace: GenerateWorkspace, seam: TeacherSeam) -> None:
    assert not workspace.out_dir.exists()
    assert seam.built == []


# --- the teacher ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_a_flipped_byte_in_a_file_of_the_export_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam, name: str
) -> None:
    flip_one_byte(workspace.teacher / name)

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"cannot use the export {workspace.teacher}: ")
    assert f"{workspace.teacher / name} differs from the manifest" in result.stderr
    assert_nothing_done(workspace, seam)


def test_an_export_without_its_manifest_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    (workspace.teacher / "manifest.json").unlink()

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"cannot use the export {workspace.teacher}: ")
    assert "no manifest vouches for it" in result.stderr
    assert_nothing_done(workspace, seam)


def test_a_directory_with_the_manifest_of_another_stage_is_not_an_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_generate_workspace(tmp_path, stage="synth")
    seam = install_teacher(monkeypatch)

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"cannot use the export {workspace.teacher}: ")
    assert "of stage 'synth', not of 'export'" in result.stderr
    assert_nothing_done(workspace, seam)


def test_a_manifest_that_cannot_be_read_is_a_data_error(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    (workspace.teacher / "manifest.json").write_text("not json", encoding="utf-8")

    result = invoke(workspace.args())

    assert_one_error(
        result, 3, f"{workspace.teacher / 'manifest.json'}: is not a readable"
    )
    assert_nothing_done(workspace, seam)


def test_a_teacher_directory_that_does_not_exist_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    missing = workspace.root / "nowhere"

    result = invoke(workspace.args(teacher=missing))

    assert_one_error(result, 2, f"the export directory {missing} does not exist")
    assert_nothing_done(workspace, seam)


def test_a_direction_tag_the_teacher_does_not_have_is_refused_before_any_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_generate_workspace(
        tmp_path, tokens=("<unk>", "<blank>", "<s>", "</s>", "a")
    )
    library = install_ct2_teacher(monkeypatch)

    result = invoke(workspace.args())

    assert_one_error(result, 2, "the direction tag <2khk_Cyrl> is not an entry of ")
    assert f"{workspace.teacher / 'shared_vocabulary.json'}" in result.stderr
    assert library.calls == []
    assert workspace.shard_files() == []
    assert not workspace.manifest.exists()
    assert not workspace.kd_corpus.exists()


# --- the options ---------------------------------------------------------------------------------


def test_more_hypotheses_than_the_beam_holds_are_refused_before_any_work(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args("--n", "3", "--beam-size", "2"))

    assert_one_error(result, 2, "--n (3) must not exceed --beam-size (2)")
    assert_nothing_done(workspace, seam)


def test_a_source_limit_above_what_ctranslate2_cuts_silently_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args("--max-source-pieces", "1024"))

    assert_one_error(result, 2, "--max-source-pieces (1024) must be at most 1023")
    assert_nothing_done(workspace, seam)


def test_the_auto_device_is_not_a_choice(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args("--device", "auto"))

    assert result.exit_code == 2, result.output
    assert "auto" in result.stderr
    assert_nothing_done(workspace, seam)


def test_the_auto_compute_type_is_not_a_choice(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args("--compute-type", "auto"))

    assert result.exit_code == 2, result.output
    assert "compute type `auto` is refused" in result.stderr
    assert_nothing_done(workspace, seam)


@pytest.mark.parametrize(
    "option",
    ["--beam-size", "--n", "--max-decoding-length", "--max-source-pieces"]
    + ["--batch-size", "--shard-size", "--threads"],
)
def test_a_number_below_one_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam, option: str
) -> None:
    result = invoke(workspace.args(option, "0"))

    assert result.exit_code == 2, result.output
    assert option in result.stderr
    assert_nothing_done(workspace, seam)


@pytest.mark.parametrize("name", ["", "a b", "a/b", "x" * 65, "naim" + chr(0x301)])
def test_a_name_that_is_not_a_plain_label_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam, name: str
) -> None:
    result = invoke(workspace.args("--name", name))

    assert_one_error(result, 2, "the name of a KD corpus must be 1 to 64 characters")
    assert_nothing_done(workspace, seam)


def test_a_direction_without_authentic_training_rows_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args("--direction", "xxx_Latn-khk_Cyrl"))

    assert_one_error(
        result, 2, "no authentic training rows for direction(s) ['xxx_Latn-khk_Cyrl']"
    )
    assert "eng_Latn-khk_Cyrl" in result.stderr
    assert_nothing_done(workspace, seam)


def test_a_corpus_given_twice_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    result = invoke(workspace.args(corpora=[*workspace.corpora, workspace.corpora[0]]))

    assert_one_error(result, 2, f"--corpus {workspace.corpora[0]} names the same file")
    assert_nothing_done(workspace, seam)


# --- the output directory --------------------------------------------------------------------------


@pytest.mark.parametrize("character", ["$", "[", "]", "*", "?"])
def test_an_output_directory_with_a_dollar_or_a_glob_character_is_refused(
    workspace: GenerateWorkspace, seam: TeacherSeam, character: str
) -> None:
    target = workspace.root / f"out{character}x"

    result = invoke(workspace.args(out_dir=target))

    assert result.exit_code == 2, result.output
    assert "output directory" in result.stderr
    assert repr(character) in result.stderr or character == "$"
    assert not target.exists()
    assert seam.built == []


@pytest.mark.parametrize("character", ["$", "[", "]", "*", "?"])
def test_the_refusal_of_such_a_character_gives_the_path_rule_not_a_reason_about_eole(
    workspace: GenerateWorkspace, seam: TeacherSeam, character: str
) -> None:
    # This command starts no Eole: the reason is the project's path rule for outputs.
    target = workspace.root / f"out{character}x"

    result = invoke(workspace.args(out_dir=target))

    assert_one_error(
        result,
        2,
        f"the output directory {target} holds {character!r}"
        if character != "$"
        else "the output directory holds '$'",
    )
    assert "path rule for outputs" in result.stderr
    assert "Eole" not in result.stderr
    assert "loads a checkpoint" not in result.stderr


def test_a_glob_character_in_the_working_directory_refuses_a_relative_directory(
    workspace: GenerateWorkspace,
    seam: TeacherSeam,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working = workspace.root / "w[1]d"
    working.mkdir()
    monkeypatch.chdir(working)

    result = invoke(workspace.args(out_dir=Path("out")))

    assert result.exit_code == 2, result.output
    assert "'['" in result.stderr
    assert not (working / "out").exists()
    assert seam.built == []


@pytest.mark.parametrize("which", ["corpus", "teacher", "selection config"])
def test_the_output_directory_may_not_be_the_directory_of_an_input(
    workspace: GenerateWorkspace, seam: TeacherSeam, which: str
) -> None:
    directory = {
        "corpus": workspace.corpora[0].parent,
        "teacher": workspace.teacher,
        "selection config": workspace.root,
    }[which]
    extra: list[str] = []
    if which == "selection config":
        selection = write_selection_config(
            workspace.root / "sel.yaml", min_script_share=0.9
        )
        extra = ["--selection-config", str(selection)]
    before = snapshot(workspace.root)

    result = invoke(workspace.args(*extra, out_dir=directory))

    assert_one_error(
        result, 2, f"--out-dir {directory} is the directory of the input file"
    )
    assert snapshot(workspace.root) == before
    assert seam.built == []


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry run"])
def test_another_stages_manifest_in_the_output_directory_is_refused_untouched(
    workspace: GenerateWorkspace, seam: TeacherSeam, dry_run: bool
) -> None:
    train_common.write_manifest(
        workspace.out_dir,
        stage="train_docify",
        config_sha256="0" * 64,
        seed=1,
        inputs=[],
        outputs=[],
        metrics={},
        packages=[],
        status={"docify": "ok"},
    )
    before = snapshot(workspace.out_dir)

    result = invoke(workspace.args(*(["--dry-run"] if dry_run else [])))

    assert_one_error(
        result,
        3,
        f"{workspace.out_dir} holds the manifest of stage 'train_docify', not of "
        "'distill_generate'",
    )
    assert snapshot(workspace.out_dir) == before
    assert seam.built == []


# --- the corpus and the selection config ------------------------------------------------------------------


def test_a_corpus_file_that_is_missing_is_a_data_error(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    missing = workspace.root / "nowhere.parquet"

    result = invoke(workspace.args(corpora=[missing]))

    assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert_nothing_done(workspace, seam)


def test_a_corpus_with_no_training_row_is_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_generate_workspace(
        tmp_path, rows=[corpus_row("Hello", "Сайн", split="dev")]
    )
    seam = install_teacher(monkeypatch)

    result = invoke(workspace.args())

    assert_one_error(result, 3, "no training rows: 1 rows read, 0 kept")
    assert_nothing_done(workspace, seam)


def test_a_corpus_with_only_back_translated_training_rows_is_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_generate_workspace(
        tmp_path, rows=[corpus_row("Hello", "Сайн", origin="bt")]
    )
    seam = install_teacher(monkeypatch)

    result = invoke(workspace.args())

    assert_one_error(result, 3, "no authentic training rows")
    assert_nothing_done(workspace, seam)


def test_a_selection_config_that_is_missing_is_a_configuration_error(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    missing = workspace.root / "nowhere.yaml"

    result = invoke(workspace.args("--selection-config", str(missing)))

    assert_one_error(result, 2, f"selection config file not found: {missing}")
    assert_nothing_done(workspace, seam)


def test_a_selection_config_that_does_not_validate_is_a_configuration_error(
    workspace: GenerateWorkspace, seam: TeacherSeam
) -> None:
    bad = write_selection_config(workspace.root / "bad.yaml", min_script_share=5)

    result = invoke(workspace.args("--selection-config", str(bad)))

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith(f"error: invalid synth config {bad}")
    assert_nothing_done(workspace, seam)


# --- hashing before reading --------------------------------------------------------------------------------


def watched(
    monkeypatch: pytest.MonkeyPatch, events: list[str], name: str, label: str
) -> None:
    real = getattr(distill_generate, name)

    def spying(*args: Any, **kwargs: Any) -> Any:
        events.append(
            label if name != "hash_input" else f"{label}:{Path(args[0]).name}"
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(distill_generate, name, spying)


def test_every_input_is_hashed_before_it_is_read(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = write_selection_config(
        workspace.root / "sel.yaml", min_script_share=0.9
    )
    events: list[str] = []
    install_teacher(monkeypatch)
    for name, label in (
        ("hash_input", "hash"),
        ("verify_export", "verify_export"),
        ("load_synth_config", "read selection config"),
        ("load_training_rows", "read corpus"),
        ("build_teacher", "build_teacher"),
    ):
        watched(monkeypatch, events, name, label)

    run_ok(workspace.args("--selection-config", str(selection)))

    assert events[:5] == [f"hash:{path.name}" for path in workspace.corpora] + [
        "hash:sel.yaml"
    ]
    assert sorted(events[5:]) == sorted(
        ["verify_export", "read selection config", "read corpus", "build_teacher"]
    )
    assert events[-1] == "build_teacher"
    assert events.index("verify_export") < events.index("build_teacher")


@pytest.mark.parametrize("which", ["corpus", "selection config", "teacher model"])
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    selection = write_selection_config(
        workspace.root / "sel.yaml", min_script_share=0.9
    )
    path, seam_name, rewrite = {
        "corpus": (workspace.corpora[0], "load_training_rows", rewrite_parquet),
        "selection config": (selection, "load_synth_config", append_a_comment),
        "teacher model": (workspace.teacher / "model.bin", "build_teacher", swap_bytes),
    }[which]
    install_teacher(monkeypatch)
    before = sha256(path)
    real = getattr(distill_generate, seam_name)
    changed: list[bool] = []

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        if not changed:
            changed.append(True)
            rewrite(path)
        return real(*args, **kwargs)

    monkeypatch.setattr(distill_generate, seam_name, after_a_change)

    run_ok(workspace.args("--selection-config", str(selection)))

    assert sha256(path) != before
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert recorded[path] == before


def rewrite_parquet(path: Path) -> None:
    pl.read_parquet(path).write_parquet(path, compression="uncompressed")


def append_a_comment(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")


def swap_bytes(path: Path) -> None:
    path.write_bytes(b"swapped after the verification")


# --- interrupts ------------------------------------------------------------------------------------------------


def test_an_interrupt_before_the_first_shard_exits_130_and_writes_no_manifest(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    install_teacher(monkeypatch)
    monkeypatch.setattr(distill_generate, "generate_kd", interrupted)

    result = invoke(workspace.args())

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()


def test_a_failure_to_write_the_outputs_is_a_write_error_and_leaves_no_manifest(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(distill_generate, "write_corpus", failing)

    result = invoke(workspace.args())

    assert_one_error(result, 1, f"cannot write the KD corpus under {workspace.out_dir}")
    assert "disk full" in result.stderr
    assert not workspace.manifest.exists()


def test_a_domain_error_while_writing_exits_with_its_code(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    def failure(*args: Any, **kwargs: Any) -> Any:
        raise DataError("the frame is not a corpus")

    monkeypatch.setattr(distill_generate, "write_corpus", failure)

    result = invoke(workspace.args())

    assert_one_error(result, 3, "the frame is not a corpus")
    assert not workspace.manifest.exists()
