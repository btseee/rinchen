"""Tests for `rinchen.train.continuation`: which checkpoint may start a continuation, and
whether a config may continue it.

A continuation trains a NEW run on from OUR OWN checkpoint, the one case where weights
other than a run's own enter a run. So the source is accepted only when it is the final
step of a run that `rinchen train launch` finished, or an averaged model that
`rinchen train average` wrote, AND the manifest that command wrote vouches for the files
as they are now (their SHA-256). Everything else is refused with a message that names the
file and what is wrong, never what is in it. The runs here are fabricated by
`tests.support.step_dirs` with real manifests written through `rinchen.manifest.store`;
the real commands' own manifests are checked against this reader in
`test_cli_train_launch.py`. Eole never starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.spm.symbols import user_defined_symbols
from rinchen.train import continuation as continuation_module
from rinchen.train.checkpoints import ContinuedFrom
from rinchen.train.continuation import (
    AVERAGE_STAGE,
    LAUNCH_STAGE,
    SourceProvenance,
    check_continuation,
    require_source_apart,
    source_path,
    verify_source,
)
from tests.support import step_dirs
from tests.support.fake_launch import continuation_config
from tests.support.step_dirs import (
    SOURCE_VOCABULARY,
    FabricatedRun,
    make_averaged_dir,
    make_finished_run,
    sha256,
)

STEP_FILES = ("model.00.safetensors", "optimizer.pt", "config.json", "vocab.json")
AVERAGED_FILES = ("model.00.safetensors", "config.json", "vocab.json")
# A text of each file that a one-byte change leaves a complete directory with: JSON stays
# JSON, the archive stays an archive, and the weights change in their last tensor.
FLIP_MARKERS = {
    "model.00.safetensors": None,
    "optimizer.pt": b"not a pickle",
    "config.json": b"saved_at",
    "vocab.json": b"<sep>",
    "averaged_from.json": b"schema_version",
}


@pytest.fixture
def teacher(tmp_path: Path) -> FabricatedRun:
    return make_finished_run(tmp_path / "teacher")


@pytest.fixture
def averaged(tmp_path: Path) -> Path:
    return make_averaged_dir(tmp_path / "averaged")


def refusal(source: Path) -> str:
    with pytest.raises(ConfigError) as raised:
        verify_source(source)
    return str(raised.value)


# ---- what is accepted ---------------------------------------------------------------


def test_the_final_step_of_a_finished_run_is_accepted_with_its_provenance(
    teacher: FabricatedRun,
) -> None:
    found = verify_source(teacher.final)

    assert found == SourceProvenance(
        kind="step",
        path=teacher.final,
        weights_sha256=sha256(teacher.final / "model.00.safetensors"),
        optimizer_sha256=sha256(teacher.final / "optimizer.pt"),
        config_sha256=sha256(teacher.final / "config.json"),
        vocab_sha256=sha256(teacher.final / "vocab.json"),
        manifest_sha256=sha256(teacher.manifest),
        manifest_stage=LAUNCH_STAGE,
        step=200,
        source_run=teacher.run_dir,
    )


def test_an_averaged_model_is_accepted_with_its_provenance(averaged: Path) -> None:
    found = verify_source(averaged)

    assert found == SourceProvenance(
        kind="averaged",
        path=averaged,
        weights_sha256=sha256(averaged / "model.00.safetensors"),
        optimizer_sha256=None,
        config_sha256=sha256(averaged / "config.json"),
        vocab_sha256=sha256(averaged / "vocab.json"),
        manifest_sha256=sha256(averaged / "manifest.json"),
        manifest_stage=AVERAGE_STAGE,
        step=None,
        source_run=None,
    )


def test_the_stage_names_are_those_the_commands_write() -> None:
    assert (LAUNCH_STAGE, AVERAGE_STAGE) == ("train_launch", "train_average")


def test_the_path_is_made_absolute_and_not_resolved(
    teacher: FabricatedRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    found = verify_source(Path("teacher") / "run" / "ckpt" / ".." / "ckpt" / "step_200")

    assert found.path == teacher.final
    assert found.path.is_absolute()


def test_a_source_that_was_moved_with_its_run_is_accepted_by_file_name(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    # The manifest lists the files by the paths they had when it was written; those
    # paths are gone now, and what vouches for a file is its name in the step directory
    # and its hash.
    moved = tmp_path / "elsewhere" / "run"
    moved.parent.mkdir()
    before = verify_source(teacher.final)
    shutil.move(teacher.run_dir, moved)

    found = verify_source(moved / "ckpt" / "step_200")

    assert found.path == moved / "ckpt" / "step_200"
    assert found.source_run == moved
    assert found.weights_sha256 == before.weights_sha256
    assert found.manifest_sha256 == before.manifest_sha256


def test_an_averaged_model_that_was_moved_or_renamed_is_accepted(
    averaged: Path, tmp_path: Path
) -> None:
    before = verify_source(averaged)
    renamed = tmp_path / "somewhere" / "teacher-average"
    renamed.parent.mkdir()
    shutil.move(averaged, renamed)

    found = verify_source(renamed)

    assert found.path == renamed
    assert found.weights_sha256 == before.weights_sha256


@pytest.mark.parametrize(
    ("nest", "found"),
    [(0, True), (1, True), (2, False)],
    ids=["two-levels", "three-levels", "four-levels"],
)
def test_the_run_directory_is_looked_for_three_levels_above_the_step(
    teacher: FabricatedRun, nest: int, found: bool
) -> None:
    # `ckpt/step_200` is two levels below the run directory; a checkpoint directory
    # nested deeper has the run's `launch.json` within three levels or not at all.
    final = teacher.final
    if nest:
        nested = teacher.run_dir.joinpath(*["nest"] * nest)
        nested.mkdir(parents=True)
        shutil.move(teacher.model_path, nested / "ckpt")
        final = nested / "ckpt" / "step_200"

    if found:
        assert verify_source(final).source_run == teacher.run_dir
    else:
        assert "launch.json" in refusal(final)


def test_a_step_directory_with_no_run_record_within_three_levels_is_refused(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    alone = tmp_path / "loose" / "step_200"
    shutil.copytree(teacher.final, alone)

    message = refusal(alone)

    assert "launch.json" in message
    assert str(alone) in message


# ---- a path that cannot be a source -------------------------------------------------


def test_a_symbolic_link_is_refused_even_when_it_leads_to_a_good_source(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    link = tmp_path / "link"
    link.symlink_to(teacher.final)

    assert "symbolic link" in refusal(link)


def test_a_link_to_an_averaged_model_is_refused_too(
    averaged: Path, tmp_path: Path
) -> None:
    link = tmp_path / "link"
    link.symlink_to(averaged)

    assert "symbolic link" in refusal(link)


def test_a_file_is_not_a_source(tmp_path: Path) -> None:
    file = tmp_path / "step_200"
    file.write_text("x", encoding="utf-8")

    assert "not a directory" in refusal(file)


def test_a_path_that_does_not_exist_is_not_a_source(tmp_path: Path) -> None:
    assert "does not exist" in refusal(tmp_path / "step_200")


def test_a_directory_that_is_neither_a_step_nor_an_averaged_model_is_refused(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    other = tmp_path / "model"
    shutil.copytree(teacher.final, other)

    message = refusal(other)

    assert "step_<N>" in message
    assert "averaged_from.json" in message


def test_an_averaged_directory_without_its_record_is_refused(
    averaged: Path,
) -> None:
    (averaged / "averaged_from.json").unlink()

    message = refusal(averaged)

    assert "averaged_from.json" in message


# ---- the record of an averaged model ---------------------------------------------------
# `averaged_from.json` and the manifest are two accounts of what an average was made from,
# written by one command; a source whose accounts differ is not what the command made.


def two_sources(directory: Path) -> tuple[step_dirs.AveragedSource, ...]:
    return step_dirs.averaged_sources(directory)


def test_an_averaged_model_that_was_moved_is_still_accepted(
    averaged: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "elsewhere" / "averaged"
    moved.parent.mkdir()
    shutil.move(averaged, moved)

    assert verify_source(moved).kind == "averaged"


@pytest.mark.parametrize(
    "case",
    ["other_sha256", "other_path", "one_input_less", "one_input_more", "reordered"],
)
def test_an_averaged_model_whose_record_and_manifest_name_other_checkpoints_is_refused(
    tmp_path: Path, case: str
) -> None:
    directory = tmp_path / "avg"
    first, second = two_sources(directory)
    extra = step_dirs.AveragedSource(
        str(tmp_path / "elsewhere" / "step_250"), 250, "3" * 64
    )
    inputs = {
        "other_sha256": (replace(first, sha256="9" * 64), second),
        "other_path": (replace(first, directory=extra.directory), second),
        "one_input_less": (first,),
        "one_input_more": (first, second, extra),
        "reordered": (second, first),
    }[case]
    make_averaged_dir(directory, inputs=inputs)

    message = refusal(directory)

    assert "averaged_from.json" in message
    assert "manifest" in message


@pytest.mark.parametrize("count", [0, 1])
def test_an_average_of_fewer_than_two_checkpoints_is_refused(
    tmp_path: Path, count: int
) -> None:
    directory = tmp_path / "avg"
    sources = two_sources(directory)[:count]
    make_averaged_dir(directory, recorded=sources)

    message = refusal(directory)

    assert "averaged_from.json" in message
    assert "fewer than two" in message


def test_an_averaged_model_whose_record_has_other_weights_than_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    averaged = make_averaged_dir(tmp_path / "avg", recorded_weights_sha256="e" * 64)

    message = refusal(averaged)

    assert "averaged_from.json" in message
    assert "weights" in message


@pytest.mark.parametrize(
    "record",
    [
        "SECRET-NOT-JSON",
        '["SECRET"]',
        '{"sources": "SECRET", "weights": {"sha256": "x"}}',
        '{"sources": 5, "weights": {"sha256": "x"}}',
        '{"sources": ["SECRET", "SECRET"], "weights": {"sha256": "x"}}',
        '{"sources": [{"path": "SECRET"}, {"path": "SECRET"}], "weights": {"sha256": "x"}}',
        '{"sources": [{"path": 1, "step": 2, "sha256": "SECRET"}], "weights": {"sha256": "x"}}',
        '{"sources": [{"path": "a", "step": "SECRET", "sha256": "b"}], "weights": {"sha256": "x"}}',
        '{"sources": [{"path": "a", "step": true, "sha256": "SECRET"}], "weights": {"sha256": "x"}}',
        '{"sources": [{"path": "a", "step": 1, "sha256": 7}], "weights": {"sha256": "x"}}',
        '{"sources": [], "weights": "SECRET"}',
        '{"sources": [], "weights": {"sha256": 3}}',
        '{"sources": []}',
    ],
)
def test_an_averaged_model_whose_record_is_not_what_average_writes_is_a_data_error(
    tmp_path: Path, record: str
) -> None:
    averaged = make_averaged_dir(tmp_path / "avg", record_text=record)

    with pytest.raises(DataError) as raised:
        verify_source(averaged)

    assert "averaged_from.json" in str(raised.value)
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize("char", ["$", "[", "]", "*", "?"])
def test_a_path_that_holds_a_variable_or_glob_character_is_refused(
    tmp_path: Path, char: str
) -> None:
    run = make_finished_run(tmp_path / f"teacher{char}1")

    message = refusal(run.final)

    assert char in message
    assert "continuation source" in message


def test_source_path_is_absolute_and_refuses_the_same_characters(
    tmp_path: Path,
) -> None:
    assert source_path(Path("a") / ".." / "b").is_absolute()
    with pytest.raises(ConfigError, match="'\\$'"):
        source_path(tmp_path / "$HOME")


# ---- the manifest ---------------------------------------------------------------------


def test_a_run_without_a_manifest_is_refused(tmp_path: Path) -> None:
    run = make_finished_run(tmp_path / "t", manifest=False)

    message = refusal(run.final)

    assert str(run.run_dir / "manifest.json") in message
    assert "no manifest" in message


def test_an_averaged_model_without_a_manifest_is_refused(tmp_path: Path) -> None:
    averaged = make_averaged_dir(tmp_path / "avg", manifest=False)

    message = refusal(averaged)

    assert str(averaged / "manifest.json") in message


@pytest.mark.parametrize("stage", ["train_ablate", "train_average", "train_docify"])
def test_the_manifest_of_another_stage_does_not_vouch_for_a_step(
    tmp_path: Path, stage: str
) -> None:
    run = make_finished_run(tmp_path / "t", stage=stage, status={"launch": "ok"})

    message = refusal(run.final)

    assert f"stage {stage!r}" in message
    assert "train_launch" in message


@pytest.mark.parametrize("stage", ["train_launch", "train_ablate"])
def test_the_manifest_of_another_stage_does_not_vouch_for_an_averaged_model(
    tmp_path: Path, stage: str
) -> None:
    averaged = make_averaged_dir(
        tmp_path / "avg", stage=stage, status={"average": "ok"}
    )

    message = refusal(averaged)

    assert f"stage {stage!r}" in message
    assert "train_average" in message


@pytest.mark.parametrize(
    "status",
    [{}, {"launch": "partial: killed"}, {"launch": "skipped: x"}, {"average": "ok"}],
)
def test_a_manifest_whose_status_is_not_ok_is_refused(
    tmp_path: Path, status: dict[str, str]
) -> None:
    run = make_finished_run(tmp_path / "t", status=status)

    message = refusal(run.final)

    assert "status" in message
    assert "manifest.json" in message


@pytest.mark.parametrize("status", [{}, {"average": "partial: x"}, {"launch": "ok"}])
def test_an_averaged_manifest_whose_status_is_not_ok_is_refused(
    tmp_path: Path, status: dict[str, str]
) -> None:
    averaged = make_averaged_dir(tmp_path / "avg", status=status)

    assert "status" in refusal(averaged)


@pytest.mark.parametrize("name", STEP_FILES)
def test_a_manifest_that_does_not_list_a_file_of_the_step_is_refused_naming_it(
    tmp_path: Path, name: str
) -> None:
    listed = [other for other in STEP_FILES if other != name]
    run = make_finished_run(tmp_path / "t", listed=listed)

    message = refusal(run.final)

    assert f"does not list {name}" in message


@pytest.mark.parametrize("name", [*AVERAGED_FILES, "averaged_from.json"])
def test_a_manifest_that_does_not_list_a_file_of_the_averaged_model_is_refused(
    tmp_path: Path, name: str
) -> None:
    listed = [
        other for other in (*AVERAGED_FILES, "averaged_from.json") if other != name
    ]
    averaged = make_averaged_dir(tmp_path / "avg", listed=listed)

    assert f"does not list {name}" in refusal(averaged)


def test_a_manifest_that_lists_a_file_name_twice_is_refused_as_ambiguous(
    teacher: FabricatedRun,
) -> None:
    document = json.loads(teacher.manifest.read_text(encoding="utf-8"))
    twin = copy.deepcopy(document["outputs"][0])
    twin["path"] = twin["path"].replace("step_200", "step_150")
    document["outputs"].append(twin)
    teacher.manifest.write_text(json.dumps(document), encoding="utf-8")

    message = refusal(teacher.final)

    assert "model.00.safetensors" in message
    assert "more than once" in message


def test_a_step_that_is_not_the_final_one_is_refused_saying_so(
    teacher: FabricatedRun,
) -> None:
    message = refusal(teacher.step(150))

    assert "step_150" in message
    assert "step_200" in message
    assert "final step" in message


def test_a_step_directory_renamed_to_another_step_is_refused(
    teacher: FabricatedRun,
) -> None:
    renamed = teacher.model_path / "step_300"
    shutil.copytree(teacher.final, renamed)

    message = refusal(renamed)

    assert "step_300" in message
    assert "final step" in message


def test_an_unreadable_manifest_is_a_data_error_that_names_the_file_not_its_content(
    teacher: FabricatedRun,
) -> None:
    teacher.manifest.write_text('{"stage": "SECRET-IN-THE-MANIFEST"', encoding="utf-8")

    with pytest.raises(DataError) as raised:
        verify_source(teacher.final)

    assert str(teacher.manifest) in str(raised.value)
    assert "SECRET-IN-THE-MANIFEST" not in str(raised.value)


def test_a_file_that_is_not_a_manifest_is_a_data_error(
    teacher: FabricatedRun,
) -> None:
    teacher.manifest.write_text(
        json.dumps({"command": "SECRET-TORCHRUN", "nodes": 2}), encoding="utf-8"
    )

    with pytest.raises(DataError) as raised:
        verify_source(teacher.final)

    assert "SECRET-TORCHRUN" not in str(raised.value)


def test_an_unreadable_manifest_of_an_averaged_model_is_a_data_error(
    averaged: Path,
) -> None:
    (averaged / "manifest.json").write_text("not json at all", encoding="utf-8")

    with pytest.raises(DataError, match="manifest.json"):
        verify_source(averaged)


def test_a_run_record_that_is_not_one_is_a_data_error(teacher: FabricatedRun) -> None:
    (teacher.run_dir / "launch.json").write_text(
        "SECRET-NOT-A-RECORD", encoding="utf-8"
    )

    with pytest.raises(DataError) as raised:
        verify_source(teacher.final)

    assert "launch.json" in str(raised.value)
    assert "SECRET-NOT-A-RECORD" not in str(raised.value)


# ---- the files themselves -------------------------------------------------------------


@pytest.mark.parametrize("name", STEP_FILES)
def test_one_changed_byte_in_a_file_of_the_step_is_refused_naming_the_file(
    teacher: FabricatedRun, name: str
) -> None:
    step_dirs.flip_one_byte(teacher.final / name, marker=FLIP_MARKERS[name])

    message = refusal(teacher.final)

    assert name in message
    assert "differs" in message
    assert "manifest" in message


@pytest.mark.parametrize("name", [*AVERAGED_FILES, "averaged_from.json"])
def test_one_changed_byte_in_a_file_of_the_averaged_model_is_refused_naming_it(
    averaged: Path, name: str
) -> None:
    step_dirs.flip_one_byte(averaged / name, marker=FLIP_MARKERS[name])

    message = refusal(averaged)

    assert name in message
    assert "differs" in message


def test_a_file_that_was_legitimately_written_again_after_the_manifest_is_refused(
    teacher: FabricatedRun,
) -> None:
    # The manifest vouches for the bytes it saw; a run extended since (its step files
    # written again) is not vouched for until a launch writes its manifest again.
    step_dirs.overwrite(teacher.final, "config.json", b'{"saved_at": 201}')

    assert "config.json" in refusal(teacher.final)


def test_the_message_never_holds_the_content_of_a_file(teacher: FabricatedRun) -> None:
    step_dirs.overwrite(
        teacher.final,
        "vocab.json",
        json.dumps({"src": ["SECRET-TOKEN"], "tgt": []}).encode(),
    )

    assert "SECRET-TOKEN" not in refusal(teacher.final)


@pytest.mark.parametrize("name", STEP_FILES)
def test_an_incomplete_step_directory_is_refused_naming_the_problem(
    teacher: FabricatedRun, name: str
) -> None:
    step_dirs.delete(teacher.final, name)

    message = refusal(teacher.final)

    assert "not complete" in message
    assert f"{name}: is missing" in message


def test_a_cut_weights_file_is_an_incomplete_directory(teacher: FabricatedRun) -> None:
    step_dirs.truncate(teacher.final, "model.00.safetensors", 10)

    assert "not complete" in refusal(teacher.final)


def test_a_file_that_is_a_link_is_not_a_file_eole_wrote(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "other-weights"
    shutil.copyfile(teacher.final / "model.00.safetensors", elsewhere)
    step_dirs.replace_with_symlink_to(teacher.final, "model.00.safetensors", elsewhere)

    assert "symbolic link" in refusal(teacher.final)


@pytest.mark.parametrize("sibling", ["model.01.safetensors", "model.99.safetensors"])
def test_a_shard_the_manifest_does_not_list_is_refused_because_eole_loads_it(
    teacher: FabricatedRun, sibling: str
) -> None:
    # Eole loads every `model.*.safetensors` of the directory it is given.
    shutil.copyfile(teacher.final / "model.00.safetensors", teacher.final / sibling)

    message = refusal(teacher.final)

    assert sibling in message


def test_a_shard_beside_the_averaged_weights_is_refused_too(averaged: Path) -> None:
    shutil.copyfile(
        averaged / "model.00.safetensors", averaged / "model.01.safetensors"
    )

    assert "model.01.safetensors" in refusal(averaged)


@pytest.mark.parametrize("name", ["model.00.safetensors", "averaged_from.json"])
def test_a_link_in_an_averaged_model_is_refused_even_when_its_target_has_the_same_bytes(
    averaged: Path, tmp_path: Path, name: str
) -> None:
    target = tmp_path / "elsewhere"
    shutil.copyfile(averaged / name, target)
    (averaged / name).unlink()
    (averaged / name).symlink_to(target)

    message = refusal(averaged)

    assert name in message
    assert "not a regular file" in message


def test_an_averaged_model_with_an_optimizer_file_is_refused(
    averaged: Path, tmp_path: Path
) -> None:
    other = step_dirs.make_step_dir(tmp_path / "other", 5)
    shutil.copyfile(other / "optimizer.pt", averaged / "optimizer.pt")

    message = refusal(averaged)

    assert "optimizer.pt" in message


@pytest.mark.parametrize("name", ["config.json", "vocab.json"])
def test_an_averaged_model_missing_a_json_file_is_refused_naming_it(
    averaged: Path, name: str
) -> None:
    (averaged / name).unlink()

    message = refusal(averaged)

    assert f"{name}: is missing" in message


# ---- each file is hashed once -----------------------------------------------------------


def test_every_file_is_hashed_once_and_the_checks_reuse_the_digests(
    teacher: FabricatedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[Path] = []
    real = continuation_module.sha256_of

    def counting(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(continuation_module, "sha256_of", counting)

    found = verify_source(teacher.final)
    check_continuation(continuation_config(Path("/runs/new")), found)

    assert sorted(path.name for path in hashed) == sorted(
        [*STEP_FILES, "manifest.json"]
    )


# ---- the source must be apart from the new run ---------------------------------------------


def apart(source: Path, run_dir: Path, vocabulary: Path | None = None) -> None:
    require_source_apart(
        source, run_dir=run_dir, vocabulary=vocabulary or run_dir / "vocab.src"
    )


def test_a_source_apart_from_the_run_and_its_vocabulary_passes(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    apart(teacher.final, tmp_path / "new")


@pytest.mark.parametrize(
    "inside",
    ["", "ckpt", "ckpt/step_200", "launch"],
    ids=["run-dir", "model-path", "step", "launch-dir"],
)
def test_a_source_that_is_the_new_run_or_inside_it_is_refused(
    tmp_path: Path, inside: str
) -> None:
    run_dir = tmp_path / "new"
    source = run_dir / inside if inside else run_dir

    with pytest.raises(ConfigError) as raised:
        apart(source, run_dir)

    assert str(run_dir) in str(raised.value)
    assert "inside" in str(raised.value) or "is the" in str(raised.value)


def test_a_run_directory_inside_the_source_is_refused(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    # The vocabulary is put outside the source, so that this refusal is the run's.
    with pytest.raises(ConfigError, match="new run directory .* is inside the source"):
        apart(teacher.final, teacher.final / "new-run", vocabulary=tmp_path / "v.src")


def test_a_vocabulary_file_inside_the_source_is_refused(
    teacher: FabricatedRun, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="vocabulary"):
        apart(teacher.final, tmp_path / "new", vocabulary=teacher.final / "vocab.src")


def test_a_link_to_the_run_does_not_hide_that_the_source_is_inside_it(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "new"
    (run_dir / "ckpt" / "step_200").mkdir(parents=True)
    link = tmp_path / "shortcut"
    link.symlink_to(run_dir / "ckpt" / "step_200")

    with pytest.raises(ConfigError, match="inside"):
        apart(link, run_dir)


# ---- the record of the source -----------------------------------------------------------------


def test_the_provenance_becomes_the_record_and_back(teacher: FabricatedRun) -> None:
    found = verify_source(teacher.final)

    record = found.as_record()

    assert isinstance(record, ContinuedFrom)
    assert record.kind == "step"
    assert record.path == str(teacher.final)
    assert record.weights_sha256 == found.weights_sha256
    assert record.step == 200
    assert record.source_run == str(teacher.run_dir)
    assert SourceProvenance.from_record(record) == found


def test_an_averaged_record_has_no_step_run_or_optimizer(averaged: Path) -> None:
    record = verify_source(averaged).as_record()

    assert (record.step, record.source_run, record.optimizer_sha256) == (
        None,
        None,
        None,
    )
    assert SourceProvenance.from_record(record) == verify_source(averaged)


def test_the_files_of_a_step_source_are_its_four_files_and_the_manifest(
    teacher: FabricatedRun,
) -> None:
    found = verify_source(teacher.final)

    assert found.files() == (
        (teacher.final / "model.00.safetensors", found.weights_sha256),
        (teacher.final / "optimizer.pt", found.optimizer_sha256),
        (teacher.final / "config.json", found.config_sha256),
        (teacher.final / "vocab.json", found.vocab_sha256),
        (teacher.manifest, found.manifest_sha256),
    )


def test_the_files_of_an_averaged_source_have_no_optimizer(averaged: Path) -> None:
    found = verify_source(averaged)

    assert found.files() == (
        (averaged / "model.00.safetensors", found.weights_sha256),
        (averaged / "config.json", found.config_sha256),
        (averaged / "vocab.json", found.vocab_sha256),
        (averaged / "manifest.json", found.manifest_sha256),
    )


# ---- check_continuation ----------------------------------------------------------------------

MODEL_KEYS = [
    ("hidden_size",),
    ("embeddings", "word_vec_size"),
    ("heads",),
    ("transformer_ff",),
    ("encoder", "layers"),
    ("decoder", "layers"),
    ("embeddings", "position_encoding_type"),
]
OPTIMIZER_KEYS = [
    "learning_rate",
    "warmup_steps",
    "decay_method",
    "optim",
    "adam_beta2",
]


def good_config() -> dict[str, Any]:
    return continuation_config(Path("/runs/new"))


def problems_of(cfg: dict[str, Any], source: SourceProvenance) -> str:
    with pytest.raises(ConfigError) as raised:
        check_continuation(cfg, source)
    return str(raised.value)


def test_a_config_that_states_everything_continues_a_teacher_that_has_everything(
    teacher: FabricatedRun,
) -> None:
    assert check_continuation(good_config(), verify_source(teacher.final)) is None


def test_the_same_holds_for_an_averaged_teacher(averaged: Path) -> None:
    assert check_continuation(good_config(), verify_source(averaged)) is None


def test_the_check_changes_nothing(teacher: FabricatedRun) -> None:
    cfg = good_config()
    before = copy.deepcopy(cfg)
    files = step_dirs.tree_digest(teacher.run_dir)

    check_continuation(cfg, verify_source(teacher.final))

    assert cfg == before
    assert step_dirs.tree_digest(teacher.run_dir) == files


@pytest.mark.parametrize("path", MODEL_KEYS, ids=lambda path: ".".join(path))
def test_a_config_that_does_not_state_a_key_of_the_architecture_is_refused_by_name(
    teacher: FabricatedRun, path: tuple[str, ...]
) -> None:
    cfg = good_config()
    node = cfg["model"]
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]

    message = problems_of(cfg, verify_source(teacher.final))

    assert f"model.{'.'.join(path)}" in message
    assert "architecture" in message


def test_a_config_with_no_model_section_names_every_key(teacher: FabricatedRun) -> None:
    cfg = good_config()
    del cfg["model"]

    message = problems_of(cfg, verify_source(teacher.final))

    for path in MODEL_KEYS:
        assert f"model.{'.'.join(path)}" in message


@pytest.mark.parametrize(
    ("path", "value", "saved"),
    [
        (("heads",), 8, 4),
        (("hidden_size",), 128, 64),
        (("transformer_ff",), 256, 128),
        (("encoder", "layers"), 6, 2),
        (("decoder", "layers"), 3, 2),
    ],
    ids=["heads", "hidden", "ff", "encoder", "decoder"],
)
def test_a_config_whose_architecture_differs_from_the_sources_is_refused_with_both_values(
    teacher: FabricatedRun, path: tuple[str, ...], value: int, saved: int
) -> None:
    cfg = good_config()
    node = cfg["model"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    message = problems_of(cfg, verify_source(teacher.final))

    assert (
        f"model.{'.'.join(path)} is {saved!r} in the checkpoint but {value!r} "
        "in the config"
    ) in message


def test_a_source_that_records_no_model_cannot_be_continued(tmp_path: Path) -> None:
    run = make_finished_run(tmp_path / "t")
    config = json.loads((run.final / "config.json").read_text(encoding="utf-8"))
    del config["model"]
    step_dirs.overwrite(run.final, "config.json", json.dumps(config).encode())
    step_dirs.write_manifest(
        run.run_dir,
        [run.final / name for name in STEP_FILES]
        + [run.run_dir / "launch.json", run.run_dir / "command.sh"],
    )

    message = problems_of(good_config(), verify_source(run.final))

    assert "config.json" in message
    assert "does not record model.heads" in message


def test_the_problems_of_one_config_are_all_listed_each_with_its_own_message(
    teacher: FabricatedRun,
) -> None:
    cfg = good_config()
    cfg["model"]["heads"] = 8
    del cfg["training"]["adam_beta2"]

    message = problems_of(cfg, verify_source(teacher.final))

    assert "model.heads is 4 in the checkpoint but 8 in the config" in message
    assert "training.adam_beta2" in message


@pytest.mark.parametrize("key", OPTIMIZER_KEYS)
def test_an_optimizer_option_left_out_is_refused_by_name(
    teacher: FabricatedRun, key: str
) -> None:
    cfg = good_config()
    del cfg["training"][key]

    message = problems_of(cfg, verify_source(teacher.final))

    assert f"training.{key}" in message
    assert "inherit" in message


def test_every_optimizer_option_left_out_is_named(teacher: FabricatedRun) -> None:
    cfg = good_config()
    for key in OPTIMIZER_KEYS:
        del cfg["training"][key]

    message = problems_of(cfg, verify_source(teacher.final))

    for key in OPTIMIZER_KEYS:
        assert f"training.{key}" in message


def test_a_config_with_no_training_section_names_the_options(
    teacher: FabricatedRun,
) -> None:
    cfg = good_config()
    del cfg["training"]

    assert "training.learning_rate" in problems_of(cfg, verify_source(teacher.final))


def vocabulary_without(*symbols: str) -> list[str]:
    return [token for token in SOURCE_VOCABULARY if token not in symbols]


def test_a_teacher_that_predates_the_vocabulary_invariant_is_refused(
    tmp_path: Path,
) -> None:
    run = make_finished_run(
        tmp_path / "old", vocab_tokens=vocabulary_without("<doc>", "<sep>")
    )

    message = problems_of(good_config(), verify_source(run.final))

    assert "<doc>" in message
    assert "<sep>" in message
    assert "predates the vocabulary invariant" in message


def test_one_missing_structural_symbol_is_named_alone(tmp_path: Path) -> None:
    run = make_finished_run(tmp_path / "old", vocab_tokens=vocabulary_without("<sep>"))

    message = problems_of(good_config(), verify_source(run.final))

    assert "<sep>" in message
    assert "<doc>" not in message


@pytest.mark.parametrize("tag", ["<2khk_Cyrl>", "<bt>"])
def test_a_tag_the_data_uses_that_the_teacher_never_saw_is_refused_by_name(
    tmp_path: Path, tag: str
) -> None:
    run = make_finished_run(tmp_path / "old", vocab_tokens=vocabulary_without(tag))

    message = problems_of(good_config(), verify_source(run.final))

    assert tag in message
    assert "<doc>" not in message
    assert "predates" not in message


def test_a_symbol_the_data_does_not_use_is_not_required(tmp_path: Path) -> None:
    # `<2rus_Cyrl>` appears in no prefix of this config.
    run = make_finished_run(
        tmp_path / "t",
        vocab_tokens=vocabulary_without(
            "<2rus_Cyrl>", "<ft>", user_defined_symbols()[-1]
        ),
    )

    assert check_continuation(good_config(), verify_source(run.final)) is None


def test_the_tokens_of_a_prefix_are_found_by_their_angle_brackets(
    tmp_path: Path,
) -> None:
    run = make_finished_run(
        tmp_path / "t", vocab_tokens=vocabulary_without("<2rus_Cyrl>", "<t>")
    )
    cfg = good_config()
    cfg["data"]["en-mn.auth"]["src_prefix"] = "<2rus_Cyrl> <doc> plain <t>"
    cfg["data"]["en-mn.bt"]["tgt_prefix"] = "<bt>"

    message = problems_of(cfg, verify_source(run.final))

    assert "<2rus_Cyrl>" in message
    assert "<t>" in message
    assert "plain" not in message


def test_a_symbol_missing_from_the_target_side_only_is_refused(tmp_path: Path) -> None:
    document = step_dirs.eole_vocabulary(SOURCE_VOCABULARY)
    document["tgt"] = vocabulary_without("<sep>")
    run = make_finished_run(tmp_path / "t", vocab_document=document)

    assert "<sep>" in problems_of(good_config(), verify_source(run.final))


@pytest.mark.parametrize(
    "document",
    [
        {"src": ["<doc>"]},
        {"tgt": ["<doc>"]},
        {"src": "<doc>", "tgt": ["<doc>"]},
        {"src": [["<doc>", 0]], "tgt": [["<doc>", 0]]},
        {"src": [1, 2], "tgt": [1, 2]},
        {"src": None, "tgt": None},
        {"specials": {}},
    ],
    ids=["no-tgt", "no-src", "src-text", "pairs", "numbers", "null", "no-lists"],
)
def test_a_vocabulary_file_that_is_not_what_eole_writes_is_a_data_error(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    run = make_finished_run(tmp_path / "t", vocab_document=document)
    found = verify_source(run.final)

    with pytest.raises(DataError) as raised:
        check_continuation(good_config(), found)

    assert "vocab.json" in str(raised.value)


def test_a_config_file_changed_after_it_was_verified_is_refused(
    teacher: FabricatedRun,
) -> None:
    found = verify_source(teacher.final)
    step_dirs.overwrite(teacher.final, "config.json", b'{"model": {"heads": 99}}')

    message = problems_of(good_config(), found)

    assert "config.json" in message
    assert "changed since" in message


def test_a_vocabulary_changed_after_it_was_verified_is_refused(
    teacher: FabricatedRun,
) -> None:
    found = verify_source(teacher.final)
    step_dirs.overwrite(
        teacher.final,
        "vocab.json",
        json.dumps(step_dirs.eole_vocabulary(["a"])).encode(),
    )

    message = problems_of(good_config(), found)

    assert "vocab.json" in message
    assert "changed since" in message


def test_the_config_check_reads_what_was_hashed_not_what_is_there_now(
    teacher: FabricatedRun,
) -> None:
    found = verify_source(teacher.final)
    config = (teacher.final / "config.json").read_bytes()

    assert hashlib.sha256(config).hexdigest() == found.config_sha256
    assert check_continuation(good_config(), found) is None
