"""Tests for the record of an export (`ExportRecord`, `export.json`) and for
`verify_export`, which every later step of the distillation uses to trust an export
directory: the teacher that translates, the generation that fingerprints it and the
commands that record it.

`export.json` and `manifest.json` are two accounts of one directory, written by one
command; `verify_export` accepts a directory only when they agree with the files, taking
each file's SHA-256 once and before it parses the file. The directories are fabricated by
`tests.support.distill_exports.write_export`, so each test damages one thing.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.manifest.model import Manifest
from rinchen.train import distill_export
from rinchen.train.distill_export import (
    EXPORT_FILES,
    RECORD_FILE,
    CheckpointRecord,
    ExportRecord,
    VerifiedExport,
    verify_export,
)
from tests.support.distill_exports import (
    WrittenExport,
    checkpoint_record,
    sha256_of_bytes,
    write_export,
)
from tests.support.step_dirs import flip_one_byte, sha256

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64


def a_record(
    *, checkpoint: CheckpointRecord | None = None, placement: Any = "src"
) -> ExportRecord:
    return ExportRecord(
        tag_placement=placement,
        checkpoint=checkpoint
        or CheckpointRecord(
            kind="step",
            step=2000,
            weights_sha256=DIGEST_A,
            config_sha256=DIGEST_B,
            vocab_sha256=DIGEST_C,
            manifest_sha256=DIGEST_D,
            manifest_stage="train_launch",
        ),
        spm_sha256=DIGEST_E,
        encoder_layers=2,
        decoder_layers=1,
        hidden_size=64,
        vocabulary_entries=344,
    )


def document_of(record: ExportRecord) -> dict[str, Any]:
    document = json.loads(record.to_bytes())
    assert isinstance(document, dict)
    return document


def encoded(document: object) -> bytes:
    return json.dumps(document).encode("utf-8")


# ---- the record ---------------------------------------------------------------------


def test_a_record_is_written_as_sorted_two_space_indented_json_with_a_final_line_feed() -> (
    None
):
    expected = (
        "{\n"
        '  "checkpoint": {\n'
        f'    "config_sha256": "{DIGEST_B}",\n'
        '    "kind": "step",\n'
        f'    "manifest_sha256": "{DIGEST_D}",\n'
        '    "manifest_stage": "train_launch",\n'
        '    "step": 2000,\n'
        f'    "vocab_sha256": "{DIGEST_C}",\n'
        f'    "weights_sha256": "{DIGEST_A}"\n'
        "  },\n"
        '  "decoder_layers": 1,\n'
        '  "encoder_layers": 2,\n'
        '  "hidden_size": 64,\n'
        '  "schema": "rinchen.distill_export/1",\n'
        f'  "spm_sha256": "{DIGEST_E}",\n'
        '  "tag_placement": "src",\n'
        '  "vocabulary_entries": 344\n'
        "}\n"
    )

    assert a_record().to_bytes() == expected.encode("utf-8")


def test_an_averaged_checkpoint_is_written_with_a_null_step() -> None:
    record = a_record(checkpoint=checkpoint_record("averaged"))

    assert b'"kind": "averaged"' in record.to_bytes()
    assert b'"step": null' in record.to_bytes()


@pytest.mark.parametrize(
    "record",
    [
        a_record(),
        a_record(placement="tgt"),
        a_record(checkpoint=checkpoint_record("averaged")),
    ],
    ids=["step", "tgt", "averaged"],
)
def test_a_record_reads_back_as_it_was_written(record: ExportRecord) -> None:
    assert ExportRecord.from_bytes(record.to_bytes()) == record


def refuses_to_read(content: bytes) -> str:
    with pytest.raises(DataError) as raised:
        ExportRecord.from_bytes(content)
    return str(raised.value)


def test_bytes_that_are_not_utf_8_are_not_a_record() -> None:
    assert "UTF-8" in refuses_to_read(b"\xff\xfe{}")


def test_text_that_is_not_json_is_not_a_record() -> None:
    assert "not valid JSON" in refuses_to_read(b'{"schema": ')


def test_json_that_is_not_an_object_is_not_a_record() -> None:
    assert "not a JSON object" in refuses_to_read(b"[1, 2]")


def test_a_document_with_another_schema_is_not_a_record() -> None:
    document = document_of(a_record())
    document["schema"] = "rinchen.distill_export/2"

    message = refuses_to_read(encoded(document))

    assert "'schema'" in message
    assert "rinchen.distill_export/1" in message


def test_a_document_with_a_key_twice_is_not_a_record() -> None:
    text = a_record().to_bytes().decode("utf-8")
    twice = text.replace(
        '"hidden_size": 64,', '"hidden_size": 64,\n  "hidden_size": 128,'
    )
    assert twice != text

    assert "'hidden_size' twice" in refuses_to_read(twice.encode("utf-8"))


TOP_LEVEL_KEYS = (
    "tag_placement",
    "checkpoint",
    "spm_sha256",
    "encoder_layers",
    "decoder_layers",
    "hidden_size",
    "vocabulary_entries",
)
CHECKPOINT_KEYS = (
    "kind",
    "step",
    "weights_sha256",
    "config_sha256",
    "vocab_sha256",
    "manifest_sha256",
    "manifest_stage",
)


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_a_record_with_a_field_missing_is_refused_naming_it(key: str) -> None:
    document = document_of(a_record())
    del document[key]

    assert f"'{key}' is missing" in refuses_to_read(encoded(document))


@pytest.mark.parametrize("key", CHECKPOINT_KEYS)
def test_a_record_with_a_checkpoint_field_missing_is_refused_naming_it(
    key: str,
) -> None:
    document = document_of(a_record())
    del document["checkpoint"][key]

    assert f"'checkpoint.{key}' is missing" in refuses_to_read(encoded(document))


def test_a_record_with_an_extra_field_is_refused_naming_it() -> None:
    document = document_of(a_record())
    document["quantization"] = "int8"

    assert "'quantization'" in refuses_to_read(encoded(document))


def test_a_record_with_an_extra_checkpoint_field_is_refused_naming_it() -> None:
    document = document_of(a_record())
    document["checkpoint"]["optimizer_sha256"] = DIGEST_A

    assert "'checkpoint.optimizer_sha256'" in refuses_to_read(encoded(document))


def test_a_checkpoint_that_is_not_an_object_is_refused() -> None:
    document = document_of(a_record())
    document["checkpoint"] = "step_2000"

    assert "'checkpoint' must be an object" in refuses_to_read(encoded(document))


@pytest.mark.parametrize("placement", ["both", "", 1, None, "SRC"])
def test_a_placement_other_than_src_or_tgt_is_refused(placement: object) -> None:
    document = document_of(a_record())
    document["tag_placement"] = placement

    assert "'tag_placement' must be 'src' or 'tgt'" in refuses_to_read(
        encoded(document)
    )


@pytest.mark.parametrize("key", ["encoder_layers", "decoder_layers", "hidden_size"])
@pytest.mark.parametrize("value", ["2", True, 2.0, None, -1, 0])
def test_a_count_that_is_not_a_positive_integer_is_refused_naming_the_field(
    key: str, value: object
) -> None:
    document = document_of(a_record())
    document[key] = value

    assert f"'{key}' must be a positive integer" in refuses_to_read(encoded(document))


@pytest.mark.parametrize("value", ["344", True, 1.5, None, -1])
def test_a_vocabulary_that_has_no_whole_number_of_entries_is_refused(
    value: object,
) -> None:
    document = document_of(a_record())
    document["vocabulary_entries"] = value

    assert "'vocabulary_entries' must be a positive integer" in refuses_to_read(
        encoded(document)
    )


@pytest.mark.parametrize(
    "key",
    ["weights_sha256", "config_sha256", "vocab_sha256", "manifest_sha256"],
)
@pytest.mark.parametrize(
    "value", ["abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n", 5, None]
)
def test_a_checkpoint_digest_that_is_not_a_sha256_is_refused_naming_the_field(
    key: str, value: object
) -> None:
    document = document_of(a_record())
    document["checkpoint"][key] = value

    assert f"'checkpoint.{key}' must be a SHA-256" in refuses_to_read(encoded(document))


@pytest.mark.parametrize("value", ["abc", "A" * 64, "a" * 65, 5, None])
def test_a_sentencepiece_digest_that_is_not_a_sha256_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["spm_sha256"] = value

    assert "'spm_sha256' must be a SHA-256" in refuses_to_read(encoded(document))


@pytest.mark.parametrize("value", ["both", 1, None])
def test_a_checkpoint_kind_other_than_step_or_averaged_is_refused(
    value: object,
) -> None:
    document = document_of(a_record())
    document["checkpoint"]["kind"] = value

    assert "'checkpoint.kind' must be 'step' or 'averaged'" in refuses_to_read(
        encoded(document)
    )


@pytest.mark.parametrize("value", [None, "2000", True, 0, -3, 1.5])
def test_a_step_checkpoint_needs_a_positive_step(value: object) -> None:
    document = document_of(a_record())
    document["checkpoint"]["step"] = value

    assert "'checkpoint.step' must be a positive integer" in refuses_to_read(
        encoded(document)
    )


def test_an_averaged_checkpoint_has_no_step() -> None:
    document = document_of(a_record(checkpoint=checkpoint_record("averaged")))
    document["checkpoint"]["step"] = 200

    assert "'checkpoint.step' must be null" in refuses_to_read(encoded(document))


def test_a_manifest_stage_that_is_not_text_is_refused() -> None:
    document = document_of(a_record())
    document["checkpoint"]["manifest_stage"] = 7

    assert "'checkpoint.manifest_stage' must be text" in refuses_to_read(
        encoded(document)
    )


# ---- verify_export: what is accepted ------------------------------------------------


@pytest.fixture
def written(tmp_path: Path) -> WrittenExport:
    return write_export(tmp_path / "export")


def refusal(directory: Path, *, error: type[Exception] = ConfigError) -> str:
    with pytest.raises(error) as raised:
        verify_export(directory)
    return str(raised.value)


def test_a_complete_export_is_verified_with_the_digest_of_each_of_its_files(
    written: WrittenExport,
) -> None:
    verified = verify_export(written.directory)

    assert isinstance(verified, VerifiedExport)
    assert verified.directory == written.directory
    assert verified.record == written.record
    assert dict(verified.sha256) == {
        name: sha256(written.directory / name) for name in EXPORT_FILES
    }
    assert verified.manifest_sha256 == sha256(written.directory / "manifest.json")


def test_the_digests_of_a_verified_export_are_the_ones_the_manifest_lists(
    written: WrittenExport,
) -> None:
    listed = {
        Path(item.path).name: item.sha256
        for item in Manifest.model_validate_json(
            (written.directory / "manifest.json").read_bytes()
        ).outputs
    }

    assert dict(verify_export(written.directory).sha256) == listed


def test_the_directory_is_made_absolute_and_not_resolved(
    written: WrittenExport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    verified = verify_export(Path("export") / ".." / "export")

    assert verified.directory == written.directory
    assert verified.directory.is_absolute()


def test_an_export_that_was_moved_is_accepted_by_file_name(
    written: WrittenExport, tmp_path: Path
) -> None:
    # The manifest lists the files by the paths they had when it was written.
    moved = tmp_path / "elsewhere" / "teacher-export"
    moved.parent.mkdir()
    shutil.move(written.directory, moved)

    verified = verify_export(moved)

    assert verified.directory == moved
    assert verified.record == written.record
    assert verified.manifest_sha256 == written.manifest_sha256


def test_an_averaged_checkpoint_and_a_target_tag_are_read_back(tmp_path: Path) -> None:
    written = write_export(
        tmp_path / "tgt", tag_placement="tgt", checkpoint=checkpoint_record("averaged")
    )

    record = verify_export(written.directory).record

    assert record.tag_placement == "tgt"
    assert (record.checkpoint.kind, record.checkpoint.step) == ("averaged", None)


# ---- verify_export: what is refused -------------------------------------------------


def damaged_manifest(directory: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    file = directory / "manifest.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    edit(document)
    file.write_text(json.dumps(document), encoding="utf-8")


def no_manifest(directory: Path) -> None:
    (directory / "manifest.json").unlink()


def other_stage(directory: Path) -> None:
    damaged_manifest(directory, lambda d: d.update(stage="synth"))


def status_not_ok(directory: Path) -> None:
    damaged_manifest(directory, lambda d: d.update(status={"export": "partial: x"}))


def leaves_out(name: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        damaged_manifest(
            directory,
            lambda d: d.update(
                outputs=[o for o in d["outputs"] if not o["path"].endswith(name)]
            ),
        )

    return damage


def lists_twice(name: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        damaged_manifest(
            directory,
            lambda d: d["outputs"].append(
                next(o for o in d["outputs"] if o["path"].endswith(name))
            ),
        )

    return damage


def flips_a_byte(name: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        flip_one_byte(directory / name)

    return damage


def loses(name: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        (directory / name).unlink()

    return damage


def is_a_directory(name: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        (directory / name).unlink()
        (directory / name).mkdir()

    return damage


# (what is wrong, how to make it so, a text of the message that says it)
REFUSALS: dict[str, tuple[Callable[[Path], None], str]] = {
    "no-manifest": (no_manifest, "no manifest vouches for it"),
    "other-stage": (other_stage, "is the manifest of stage 'synth'"),
    "status-not-ok": (status_not_ok, "does not say export: ok"),
    **{
        f"flipped-{name}": (flips_a_byte(name), f"{name} differs from the manifest")
        for name in EXPORT_FILES
    },
    **{f"lost-{name}": (loses(name), f"{name} is missing") for name in EXPORT_FILES},
    **{
        f"directory-{name}": (is_a_directory(name), f"{name} is not a regular file")
        for name in EXPORT_FILES
    },
    **{
        f"unlisted-{name}": (leaves_out(name), f"does not list {name}")
        for name in EXPORT_FILES
    },
    **{
        f"twice-{name}": (lists_twice(name), f"lists {name} more than once")
        for name in EXPORT_FILES
    },
}


@pytest.mark.parametrize("scene", sorted(REFUSALS))
def test_a_damaged_export_is_refused_with_a_message_naming_what_is_wrong(
    tmp_path: Path, scene: str
) -> None:
    damage, says = REFUSALS[scene]
    directory = write_export(tmp_path / "export").directory
    damage(directory)

    message = refusal(directory)

    assert says in message
    assert str(directory) in message


def test_every_kind_of_damage_has_a_message_of_its_own(tmp_path: Path) -> None:
    messages: dict[str, str] = {}
    for scene in sorted(REFUSALS):
        damage, _ = REFUSALS[scene]
        directory = write_export(tmp_path / scene / "export").directory
        damage(directory)
        messages[scene] = refusal(directory).replace(str(directory), "<dir>")

    assert len(set(messages.values())) == len(messages)


@pytest.mark.parametrize(
    "status",
    [{}, {"export": "failed"}, {"launch": "ok"}, {"export": "ok ", "x": "ok"}],
    ids=["none", "not-ok", "another-key", "padded"],
)
def test_a_manifest_whose_status_does_not_say_export_ok_does_not_vouch_for_an_export(
    tmp_path: Path, status: dict[str, str]
) -> None:
    directory = write_export(tmp_path / "export", status=status).directory

    assert "does not say export: ok" in refusal(directory)


def test_a_record_that_states_another_sentencepiece_model_than_the_one_beside_it_is_refused(
    tmp_path: Path, written: WrittenExport
) -> None:
    directory = write_export(tmp_path / "liar", record_spm_sha256=DIGEST_E).directory

    message = refusal(directory)

    assert RECORD_FILE in message
    assert "spm.model" in message
    assert DIGEST_E[:12] in message
    assert written.sha256["spm.model"][:12] in message


def test_a_directory_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    message = refusal(tmp_path / "nothing")

    assert f"{tmp_path / 'nothing'} does not exist" in message


def test_a_file_in_place_of_the_directory_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x", encoding="utf-8")

    assert f"{file} is not a directory" in refusal(file)


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"{}", b"[]", b"\xff\xfe\x00"],
    ids=["not-json", "empty-object", "list", "not-utf-8"],
)
def test_a_manifest_that_cannot_be_read_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    directory = write_export(tmp_path / "export").directory
    (directory / "manifest.json").write_bytes(content)

    message = refusal(directory, error=DataError)

    assert (
        f"{directory / 'manifest.json'}: is not a readable Rinchen manifest" in message
    )


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"[1]", b"{}", b"\xff\xfe\x00"],
    ids=["not-json", "list", "empty-object", "not-utf-8"],
)
def test_a_record_that_cannot_be_read_is_a_data_error_naming_the_file(
    tmp_path: Path, content: bytes
) -> None:
    # The manifest vouches for the bytes, so only the reading of them is refused.
    directory = write_export(tmp_path / "export", record_text=content).directory

    message = refusal(directory, error=DataError)

    assert message.startswith(f"{directory / RECORD_FILE}: ")


# ---- verify_export: each file is hashed once, before it is parsed ----------------------


class Trace:
    """What `verify_export` did, in order: `hash:<name>` for each SHA-256 taken and
    `parse:<name>` for each of the two files it parses."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[str] = []
        real_hash = distill_export.sha256_of
        real_manifest = Manifest.model_validate_json
        real_record = ExportRecord.from_bytes

        def hashing(file: Path) -> str:
            self.events.append(f"hash:{file.name}")
            return real_hash(file)

        def parsing_the_manifest(content: bytes) -> Manifest:
            self.events.append("parse:manifest.json")
            return real_manifest(content)

        def parsing_the_record(content: bytes) -> ExportRecord:
            self.events.append(f"parse:{RECORD_FILE}")
            return real_record(content)

        monkeypatch.setattr(distill_export, "sha256_of", hashing)
        monkeypatch.setattr(Manifest, "model_validate_json", parsing_the_manifest)
        monkeypatch.setattr(ExportRecord, "from_bytes", parsing_the_record)


def test_each_file_and_the_manifest_is_hashed_exactly_once_and_before_it_is_parsed(
    written: WrittenExport, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = Trace(monkeypatch)

    verify_export(written.directory)

    hashed = [event for event in trace.events if event.startswith("hash:")]
    assert sorted(hashed) == sorted(
        [f"hash:{name}" for name in (*EXPORT_FILES, "manifest.json")]
    )
    for name in ("manifest.json", RECORD_FILE):
        assert trace.events.index(f"hash:{name}") < trace.events.index(f"parse:{name}")
        assert trace.events.count(f"parse:{name}") == 1


def test_a_refused_export_hashes_nothing_beyond_what_it_needed_to_refuse(
    written: WrittenExport, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A manifest of another stage is refused before any of the five files is hashed (the
    # model of a real teacher is a gigabyte).
    other_stage(written.directory)
    trace = Trace(monkeypatch)

    refusal(written.directory)

    assert trace.events == ["hash:manifest.json", "parse:manifest.json"]


@pytest.mark.parametrize(
    "name", ["manifest.json", RECORD_FILE], ids=["manifest", "record"]
)
def test_a_file_that_changes_after_it_was_hashed_is_not_the_one_that_is_parsed(
    written: WrittenExport, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # A document that parses and says something else: only the comparison of the bytes
    # that are parsed with the digest that was taken tells it from the vouched one.
    real = distill_export.sha256_of
    other = encoded({"schema": "rinchen.distill_export/1"})

    def swapping(file: Path) -> str:
        digest = real(file)
        if file.name == name:
            file.write_bytes(other)
        return digest

    monkeypatch.setattr(distill_export, "sha256_of", swapping)

    message = refusal(written.directory)

    assert f"{written.directory / name} changed after it was hashed" in message


def test_the_hash_of_the_manifest_is_that_of_the_bytes_that_were_parsed(
    written: WrittenExport,
) -> None:
    content = (written.directory / "manifest.json").read_bytes()

    assert verify_export(written.directory).manifest_sha256 == sha256_of_bytes(content)
