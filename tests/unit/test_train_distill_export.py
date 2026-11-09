"""Tests for `rinchen.train.distill_export`: turning a checkpoint of OURS into a CTranslate2
model directory with a record of where it came from (D10), and reading such a directory back.

`export_checkpoint` accepts exactly what `continuation.verify_source` accepts (the final step
of a run `rinchen train launch` finished, an averaged model `rinchen train average` wrote),
the SentencePiece model the caller names must be the one the checkpoint was trained with,
and any failure leaves no output directory behind. The runs are fabricated by
`tests.support.step_dirs` with real manifests; the converter is replaced by
`tests.support.distill_exports.fake_converter`, so nothing starts a process (the real
converter runs in the integration test of the distillation gate).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.eole_ct2 import export_log_path
from rinchen.train import distill_export
from rinchen.train.distill_export import (
    CT2_FILES,
    EXPORT_FILES,
    EXPORT_STAGE,
    RECORD_FILE,
    SPM_FILE,
    CheckpointRecord,
    ExportRecord,
    export_checkpoint,
)
from rinchen.utils import logged_process
from tests.support.distill_exports import FakeConverter, fake_converter
from tests.support.fake_process import LONG_POLL_S, FakeProcess
from tests.support.sized_spm import sized_spm_bytes
from tests.support.step_dirs import (
    FabricatedRun,
    delete,
    eole_model_section,
    eole_vocabulary,
    flip_one_byte,
    make_averaged_dir,
    make_finished_run,
    sha256,
    tree_digest,
    write_spm_model,
)

# 4 specials, `a`, `b` and the 118 reserved symbols of the project.
SOURCE_VOCABULARY_ENTRIES = 124
STUDENT_SHAPE = {"hidden_size": 96, "heads": 4, "transformer_ff": 192}


@pytest.fixture
def teacher(tmp_path: Path) -> FabricatedRun:
    return make_finished_run(
        tmp_path / "teacher",
        model=eole_model_section(encoder_layers=2, decoder_layers=1),
    )


@pytest.fixture
def averaged(tmp_path: Path) -> Path:
    return make_averaged_dir(
        tmp_path / "averaged",
        model=eole_model_section(encoder_layers=2, decoder_layers=1),
    )


@pytest.fixture
def spm_model(tmp_path: Path) -> Path:
    return write_spm_model(tmp_path / "spm")


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConverter:
    return fake_converter(monkeypatch)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def read_nothing(*_: object) -> None:
    raise AssertionError("the checkpoint was read")


def export(
    checkpoint: Path,
    target: Path,
    spm_model: Path,
    *,
    placement: Any = "src",
    timeout: float | None = None,
) -> ExportRecord:
    return export_checkpoint(
        checkpoint,
        target,
        spm_model=spm_model,
        tag_placement=placement,
        timeout=timeout,
    )


def refusal(
    checkpoint: Path,
    target: Path,
    spm_model: Path,
    *,
    error: type[Exception] = ConfigError,
) -> str:
    with pytest.raises(error) as raised:
        export(checkpoint, target, spm_model)
    return str(raised.value)


def assert_nothing_was_made(
    converter: FakeConverter, target: Path, *, source: Path, digest: str
) -> None:
    assert not target.exists()
    assert converter.calls == []
    assert tree_digest(source) == digest


# ---- what is exported -------------------------------------------------------------


def test_the_final_step_of_a_run_is_exported_with_a_record_of_where_it_came_from(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    before = tree_digest(teacher.run_dir)

    record = export(teacher.final, out_dir, spm_model)

    assert record == ExportRecord(
        tag_placement="src",
        checkpoint=CheckpointRecord(
            kind="step",
            step=200,
            weights_sha256=sha256(teacher.final / "model.00.safetensors"),
            config_sha256=sha256(teacher.final / "config.json"),
            vocab_sha256=sha256(teacher.final / "vocab.json"),
            manifest_sha256=sha256(teacher.manifest),
            manifest_stage="train_launch",
        ),
        spm_sha256=sha256(spm_model),
        encoder_layers=2,
        decoder_layers=1,
        hidden_size=64,
        vocabulary_entries=SOURCE_VOCABULARY_ENTRIES,
    )
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "config.json",
        "export.json",
        "model.bin",
        "shared_vocabulary.json",
        "spm.model",
    ]
    assert (out_dir / "export.json").read_bytes() == record.to_bytes()
    assert (out_dir / "spm.model").read_bytes() == spm_model.read_bytes()
    assert converter.calls[0].checkpoint == teacher.final
    assert converter.calls[0].out_dir == out_dir
    assert tree_digest(teacher.run_dir) == before


def test_an_averaged_model_is_exported_as_such(
    averaged: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    record = export(averaged, out_dir, spm_model, placement="tgt")

    assert record.tag_placement == "tgt"
    assert record.checkpoint == CheckpointRecord(
        kind="averaged",
        step=None,
        weights_sha256=sha256(averaged / "model.00.safetensors"),
        config_sha256=sha256(averaged / "config.json"),
        vocab_sha256=sha256(averaged / "vocab.json"),
        manifest_sha256=sha256(averaged / "manifest.json"),
        manifest_stage="train_average",
    )
    assert (out_dir / "export.json").read_bytes() == record.to_bytes()


def test_the_shape_is_read_from_the_checkpoints_config_not_assumed(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(
        tmp_path / "student",
        model=eole_model_section(encoder_layers=3, decoder_layers=1, **STUDENT_SHAPE),
    )

    record = export(run.final, out_dir, spm_model)

    assert (record.encoder_layers, record.decoder_layers, record.hidden_size) == (
        3,
        1,
        96,
    )


def test_the_hidden_size_is_the_encoders_and_the_vocabulary_is_the_source_side(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    # Eole writes the same value for both stacks and both sides of a shared-vocabulary
    # model; a document that differs says which key and which side the record reads.
    model = eole_model_section(encoder_layers=2, decoder_layers=1)
    model["encoder"]["hidden_size"] = 96
    document = eole_vocabulary(("<unk>", "<blank>", "<s>", "</s>", "a"))
    document["tgt"] = ["<unk>", "<blank>", "<s>"]
    run = make_finished_run(tmp_path / "odd", model=model, vocab_document=document)

    record = export(run.final, out_dir, spm_model)

    assert (record.hidden_size, record.vocabulary_entries) == (96, 5)


def test_the_vocabulary_entries_are_those_of_the_checkpoints_source_vocabulary(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(
        tmp_path / "small", vocab_tokens=("<unk>", "<blank>", "<s>", "</s>", "a")
    )

    assert export(run.final, out_dir, spm_model).vocabulary_entries == 5


def test_the_saved_copy_is_found_by_the_name_the_saved_config_gives_it(
    tmp_path: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(tmp_path / "named", spm_name="student_pieces.model")
    model = tmp_path / "spm" / "elsewhere.model"
    model.parent.mkdir()
    model.write_bytes((run.final / "student_pieces.model").read_bytes())

    record = export(run.final, out_dir, model)

    assert record.spm_sha256 == sha256(model)


def test_the_timeout_and_the_checkpoint_directory_reach_the_converter(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    export(teacher.final, out_dir, spm_model, timeout=12.5)

    (call,) = converter.calls
    assert (call.checkpoint, call.out_dir, call.timeout) == (
        teacher.final,
        out_dir,
        12.5,
    )


def test_a_relative_checkpoint_path_reaches_the_converter_made_absolute(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The path Eole is given is the path the manifests and the record name, not a
    # resolved one: the converter is handed the absolute spelling of what was typed.
    monkeypatch.chdir(tmp_path)

    export(Path("teacher/run/ckpt/../ckpt/step_200"), out_dir, spm_model)

    assert converter.calls[0].checkpoint == teacher.final


def test_a_placement_other_than_src_or_tgt_is_refused_before_anything_is_read(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distill_export, "verify_source", read_nothing)

    with pytest.raises(ValueError, match="placement must be one of src, tgt"):
        export(teacher.final, out_dir, spm_model, placement="both")

    assert not out_dir.exists()


# ---- what is refused, with nothing made ----------------------------------------------


def test_a_run_without_our_manifest_is_refused(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(tmp_path / "bare", manifest=False)
    digest = tree_digest(run.run_dir)

    message = refusal(run.final, out_dir, spm_model)

    assert "no manifest vouches for it" in message
    assert str(run.manifest) in message
    assert_nothing_was_made(converter, out_dir, source=run.run_dir, digest=digest)


def test_a_manifest_of_another_stage_is_refused(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(tmp_path / "foreign", stage="synth")
    digest = tree_digest(run.run_dir)

    message = refusal(run.final, out_dir, spm_model)

    assert "'synth'" in message
    assert "'train_launch'" in message
    assert_nothing_was_made(converter, out_dir, source=run.run_dir, digest=digest)


def test_a_checkpoint_whose_weights_changed_after_the_manifest_is_refused(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    flip_one_byte(teacher.final / "model.00.safetensors")
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, spm_model)

    assert "model.00.safetensors differs from the manifest" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


@pytest.mark.parametrize("name", ["config.json", "vocab.json"])
def test_a_checkpoint_file_that_changed_after_the_manifest_is_refused(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    name: str,
) -> None:
    flip_one_byte(teacher.final / name, marker=b"saved_at" if name[0] == "c" else b"a")
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, spm_model)

    assert f"{name} differs from the manifest" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


def test_a_step_that_is_not_the_final_one_is_refused(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.step(150), out_dir, spm_model)

    assert "only the final step of a finished run is vouched for" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


def test_an_incomplete_step_is_refused(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    delete(teacher.final, "optimizer.pt")
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, spm_model)

    assert "optimizer.pt" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


def test_a_sentencepiece_model_that_is_not_the_one_the_checkpoint_was_trained_with_is_refused(
    tmp_path: Path, teacher: FabricatedRun, converter: FakeConverter, out_dir: Path
) -> None:
    other = tmp_path / "other" / "other.model"
    other.parent.mkdir()
    other.write_bytes(sized_spm_bytes(13))
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, other)

    saved = sha256(teacher.final / "toy.model")
    assert saved != sha256(other)
    assert str(other) in message
    assert sha256(other)[:12] in message
    assert saved[:12] in message
    assert saved not in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


def test_a_checkpoint_whose_config_records_no_sentencepiece_model_is_refused(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    run = make_finished_run(tmp_path / "unrecorded", record_spm=False)
    digest = tree_digest(run.run_dir)

    message = refusal(run.final, out_dir, spm_model)

    assert "src_subword_model" in message
    assert str(run.final / "config.json") in message
    assert_nothing_was_made(converter, out_dir, source=run.run_dir, digest=digest)


def test_a_checkpoint_that_lost_its_saved_copy_of_the_model_is_refused(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    delete(teacher.final, "toy.model")
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, spm_model)

    assert "no copy of its SentencePiece model" in message
    assert str(teacher.final / "toy.model") in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_a_sentencepiece_model_that_is_not_a_file_is_refused(
    teacher: FabricatedRun,
    tmp_path: Path,
    converter: FakeConverter,
    out_dir: Path,
    kind: str,
) -> None:
    model = tmp_path / "no-such.model"
    if kind == "directory":
        model.mkdir()
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, model)

    assert f"{model} is not a file" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


@pytest.mark.parametrize("char", ["$", "[", "]", "*", "?"])
def test_a_checkpoint_path_holding_a_character_eole_would_read_is_refused(
    teacher: FabricatedRun,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    char: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moved = tmp_path / f"te{char}acher"
    shutil.move(teacher.run_dir.parent, moved)
    checkpoint = moved / "run" / "ckpt" / "step_200"
    digest = tree_digest(moved)
    monkeypatch.setattr(distill_export, "verify_source", read_nothing)

    message = refusal(checkpoint, out_dir, spm_model)

    assert repr(char) in message
    assert_nothing_was_made(converter, out_dir, source=moved, digest=digest)


@pytest.mark.parametrize("char", ["$", "[", "]", "*", "?"])
def test_an_output_directory_the_converter_would_get_with_such_a_character_is_refused(
    teacher: FabricatedRun,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    char: str,
) -> None:
    target = tmp_path / f"out{char}dir"
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, target, spm_model)

    assert repr(char) in message
    assert_nothing_was_made(converter, target, source=teacher.run_dir, digest=digest)


def test_a_sentencepiece_path_holding_a_dollar_is_refused(
    teacher: FabricatedRun, tmp_path: Path, converter: FakeConverter, out_dir: Path
) -> None:
    model = write_spm_model(tmp_path / "sp$m")
    digest = tree_digest(teacher.run_dir)

    message = refusal(teacher.final, out_dir, model)

    assert "'$'" in message
    assert_nothing_was_made(converter, out_dir, source=teacher.run_dir, digest=digest)


def test_an_output_directory_that_exists_is_refused_before_anything_is_hashed(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run is not even a run: the first thing refused is the directory the converter
    # would refuse, and it is left as it was.
    out_dir.mkdir()
    (out_dir / "mine.txt").write_text("keep", encoding="utf-8")
    digest = tree_digest(out_dir)

    message = refusal(tmp_path / "not-a-checkpoint", out_dir, spm_model)

    assert f"{out_dir} exists" in message
    assert converter.calls == []
    assert tree_digest(out_dir) == digest


def test_a_directory_made_while_the_checkpoint_was_hashed_is_refused_and_kept(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hashing a real checkpoint takes a while; a directory someone else made in the
    # meantime is not ours to remove when the export fails.
    real = distill_export.verify_source

    def meanwhile(checkpoint: Path) -> Any:
        found = real(checkpoint)
        out_dir.mkdir()
        (out_dir / "theirs.txt").write_text("keep", encoding="utf-8")
        return found

    monkeypatch.setattr(distill_export, "verify_source", meanwhile)

    message = refusal(teacher.final, out_dir, spm_model)

    assert f"{out_dir} exists" in message
    assert converter.calls == []
    assert (out_dir / "theirs.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("deleted", "key"),
    [
        (("encoder", "layers"), "model.encoder.layers"),
        (("decoder", "layers"), "model.decoder.layers"),
        (("encoder", "hidden_size"), "model.encoder.hidden_size"),
    ],
)
def test_a_saved_config_that_does_not_say_the_shape_is_a_data_error_naming_the_key(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    deleted: tuple[str, str],
    key: str,
) -> None:
    model = eole_model_section(encoder_layers=2, decoder_layers=1)
    del model[deleted[0]][deleted[1]]
    run = make_finished_run(tmp_path / "shapeless", model=model)
    digest = tree_digest(run.run_dir)

    message = refusal(run.final, out_dir, spm_model, error=DataError)

    assert str(run.final / "config.json") in message
    assert key in message
    assert_nothing_was_made(converter, out_dir, source=run.run_dir, digest=digest)


@pytest.mark.parametrize("value", ["2", True, 0, -1, 2.0, None])
def test_a_layer_count_that_is_not_a_positive_integer_is_a_data_error(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    value: object,
) -> None:
    model = eole_model_section(encoder_layers=2, decoder_layers=1)
    model["encoder"]["layers"] = value
    run = make_finished_run(tmp_path / "odd", model=model)

    message = refusal(run.final, out_dir, spm_model, error=DataError)

    assert "model.encoder.layers" in message
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "document",
    [{"tgt": ["a"]}, {"src": "a b"}, {"src": []}, {"src": [["a", 0]]}],
    ids=["no-src", "src-not-a-list", "empty-src", "src-not-tokens"],
)
def test_a_saved_vocabulary_that_is_not_a_list_of_tokens_is_a_data_error(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    document: dict[str, Any],
) -> None:
    run = make_finished_run(tmp_path / "vocab", vocab_document=document)

    message = refusal(run.final, out_dir, spm_model, error=DataError)

    assert str(run.final / "vocab.json") in message
    assert "'src'" in message
    assert not out_dir.exists()


@pytest.mark.parametrize("name", ["config.json", "vocab.json"])
def test_a_file_that_changes_after_it_was_hashed_is_not_the_one_that_is_read(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    # Valid JSON saying another shape: only the comparison with the digest `verify_source`
    # took tells it from the file that was vouched for.
    real = distill_export.verify_source
    other = {"model": {"encoder": {"layers": 9, "hidden_size": 9}}, "src": ["x"]}

    def swap(checkpoint: Path) -> Any:
        found = real(checkpoint)
        (found.path / name).write_text(json.dumps(other), encoding="utf-8")
        return found

    monkeypatch.setattr(distill_export, "verify_source", swap)

    message = refusal(teacher.final, out_dir, spm_model)

    assert f"{teacher.final / name} changed after it was hashed" in message
    assert converter.calls == []
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("content", "problem"),
    [(b"{not json", "is not valid JSON"), (b"[1, 2]", "is not a JSON object")],
)
def test_a_saved_config_that_does_not_parse_is_a_data_error(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    problem: str,
) -> None:
    # The manifest vouches for the bytes, so the digest is the broken file's.
    real = distill_export.verify_source

    def broken(checkpoint: Path) -> Any:
        found = real(checkpoint)
        (found.path / "config.json").write_bytes(content)
        return replace(found, config_sha256=sha256(found.path / "config.json"))

    monkeypatch.setattr(distill_export, "verify_source", broken)

    message = refusal(teacher.final, out_dir, spm_model, error=DataError)

    assert f"{teacher.final / 'config.json'}: {problem}" in message
    assert not out_dir.exists()


# ---- a failure leaves no directory ---------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [UpstreamError("the converter failed"), KeyboardInterrupt()],
    ids=["failed", "interrupted"],
)
def test_a_converter_that_fails_half_way_leaves_no_directory_and_keeps_its_log(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    converter = fake_converter(monkeypatch, error=error)
    digest = tree_digest(teacher.run_dir)

    with pytest.raises(type(error)) as raised:
        export(teacher.final, out_dir, spm_model)

    assert raised.value is error
    assert len(converter.calls) == 1
    assert not out_dir.exists()
    assert export_log_path(out_dir).read_text(encoding="utf-8") == "converter log\n"
    assert tree_digest(teacher.run_dir) == digest


def test_a_converter_that_exits_zero_without_a_model_leaves_no_directory(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real `export_ct2` over a converter process that makes the directory, writes
    # a line and exits 0 with no `model.bin`.
    def silent_converter(argv: list[str], **kwargs: Any) -> FakeProcess:
        Path(argv[argv.index("--output_dir") + 1]).mkdir()
        kwargs["stdout"].write(b"converting\n")
        return FakeProcess(0, [])

    monkeypatch.setattr(subprocess, "Popen", silent_converter)
    monkeypatch.setattr(logged_process, "_INTERRUPT_POLL_S", LONG_POLL_S)

    with pytest.raises(UpstreamError, match="left no model.bin"):
        export(teacher.final, out_dir, spm_model)

    assert not out_dir.exists()
    assert export_log_path(out_dir).read_text(encoding="utf-8") == "converting\n"


@pytest.mark.parametrize("name", [name for name in CT2_FILES if name != "model.bin"])
def test_a_converter_that_leaves_one_of_its_other_files_out_leaves_no_directory(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    fake_converter(monkeypatch, skip=(name,))

    with pytest.raises(UpstreamError, match=f"left no {name}"):
        export(teacher.final, out_dir, spm_model)

    assert not out_dir.exists()


@pytest.mark.parametrize("name", [SPM_FILE, RECORD_FILE])
def test_a_failure_writing_the_files_beside_the_models_leaves_no_directory(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    real = distill_export.replace_file

    def full_disk(path: Path, content: bytes, **options: Any) -> None:
        if path.name == name:
            raise OSError(28, "No space left on device")
        real(path, content, **options)

    monkeypatch.setattr(distill_export, "replace_file", full_disk)

    with pytest.raises(OSError, match="No space left"):
        export(teacher.final, out_dir, spm_model)

    assert len(converter.calls) == 1
    assert not out_dir.exists()


def test_the_export_files_are_the_five_the_manifest_lists() -> None:
    assert CT2_FILES == ("model.bin", "shared_vocabulary.json", "config.json")
    assert EXPORT_FILES == (
        "model.bin",
        "shared_vocabulary.json",
        "config.json",
        "spm.model",
        "export.json",
    )
    assert EXPORT_STAGE == "distill_export"
