"""Tests for `verify_export` (`rinchen.export.verified`), which every later step that trusts
an export directory uses: the engine that translates, the commands that name it, the parity
and the evaluations.

`rinchen.json` and `manifest.json` are two accounts of one directory, written by one
function (`convert`); `verify_export` accepts a directory only when they agree with the
files, taking each file's SHA-256 once and before it parses the file, and when the language
registry still gives the tags the record states. The directories are fabricated by
`tests.support.model_exports.write_model_export`, so each test damages one thing.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.export import verified as verified_module
from rinchen.export.record import EXPORT_FILES, RECORD_FILE, ModelRecord
from rinchen.export.verified import VerifiedExport, verify_export
from rinchen.manifest.model import Manifest
from tests.support.distill_exports import sha256_of_bytes
from tests.support.model_exports import (
    WrittenModel,
    checkpoint_record,
    write_model_export,
)
from tests.support.step_dirs import SOURCE_VOCABULARY, flip_one_byte, sha256

DIGEST_E = "e" * 64


@pytest.fixture
def written(tmp_path: Path) -> WrittenModel:
    return write_model_export(tmp_path / "export")


def refusal(directory: Path, *, error: type[Exception] = ConfigError) -> str:
    with pytest.raises(error) as raised:
        verify_export(directory)
    return str(raised.value)


# ---- what is accepted ----------------------------------------------------------------


def test_a_complete_export_is_verified_with_the_digest_of_each_of_its_files(
    written: WrittenModel,
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
    written: WrittenModel,
) -> None:
    listed = {
        Path(item.path).name: item.sha256
        for item in Manifest.model_validate_json(
            (written.directory / "manifest.json").read_bytes()
        ).outputs
    }

    assert dict(verify_export(written.directory).sha256) == listed


def test_the_digests_cannot_be_changed_by_the_caller(written: WrittenModel) -> None:
    verified = verify_export(written.directory)

    with pytest.raises(TypeError):
        verified.sha256["model.bin"] = "0" * 64  # type: ignore[index]


def test_the_directory_is_made_absolute_and_not_resolved(
    written: WrittenModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    verified = verify_export(Path("export") / ".." / "export")

    assert verified.directory == written.directory
    assert verified.directory.is_absolute()


def test_a_link_to_the_directory_is_kept_as_it_was_given_and_not_followed(
    written: WrittenModel, tmp_path: Path
) -> None:
    link = tmp_path / "link"
    link.symlink_to(written.directory, target_is_directory=True)

    verified = verify_export(link)

    assert verified.directory == link
    assert verified.record == written.record


def test_an_export_that_was_moved_is_accepted_by_file_name(
    written: WrittenModel, tmp_path: Path
) -> None:
    # The manifest lists the files by the paths they had when it was written.
    moved = tmp_path / "elsewhere" / "teacher-export"
    moved.parent.mkdir()
    shutil.move(written.directory, moved)

    verified = verify_export(moved)

    assert verified.directory == moved
    assert verified.record == written.record
    assert verified.manifest_sha256 == written.manifest_sha256


@pytest.mark.parametrize(
    "options",
    [
        {"placement": "tgt", "checkpoint": checkpoint_record("averaged")},
        {"quantization": "int8", "document_level": True},
        {"quantization": "int8_float16", "directions": ("khk_Cyrl-eng_Latn",)},
        {"directions": ("eng_Latn-mn_Mong", "khk_Cyrl-mn_Mong")},
    ],
    ids=["tgt-averaged", "int8-document", "int8_float16-one-way", "traditional-script"],
)
def test_the_record_of_each_kind_of_export_is_read_back(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    model = write_model_export(tmp_path / "other", **options)

    assert verify_export(model.directory).record == model.record


# ---- what is refused -----------------------------------------------------------------


def damaged_manifest(directory: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    file = directory / "manifest.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    edit(document)
    file.write_text(json.dumps(document), encoding="utf-8")


def no_manifest(directory: Path) -> None:
    (directory / "manifest.json").unlink()


def another_stage(stage: str) -> Callable[[Path], None]:
    def damage(directory: Path) -> None:
        damaged_manifest(directory, lambda d: d.update(stage=stage))

    return damage


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
    "manifest-of-the-distillation": (
        another_stage("distill_export"),
        "is the manifest of stage 'distill_export', not of 'export'",
    ),
    "manifest-of-synth": (
        another_stage("synth"),
        "is the manifest of stage 'synth', not of 'export'",
    ),
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
    directory = write_model_export(tmp_path / "export").directory
    damage(directory)

    message = refusal(directory)

    assert says in message
    assert str(directory) in message


def test_every_kind_of_damage_has_a_message_of_its_own(tmp_path: Path) -> None:
    messages: dict[str, str] = {}
    for scene in sorted(REFUSALS):
        damage, _ = REFUSALS[scene]
        directory = write_model_export(tmp_path / scene / "export").directory
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
    directory = write_model_export(tmp_path / "export", status=status).directory

    assert "does not say export: ok" in refusal(directory)


def test_a_manifest_that_leaves_a_file_out_does_not_vouch_for_it(
    tmp_path: Path,
) -> None:
    directory = write_model_export(
        tmp_path / "export", listed=("model.bin", "spm.model", "rinchen.json")
    ).directory

    assert "does not list shared_vocabulary.json" in refusal(directory)


def test_a_record_that_states_another_sentencepiece_model_than_the_one_beside_it_is_refused(
    tmp_path: Path, written: WrittenModel
) -> None:
    directory = write_model_export(
        tmp_path / "liar", record_spm_sha256=DIGEST_E
    ).directory

    message = refusal(directory)

    assert RECORD_FILE in message
    assert "spm.model" in message
    assert DIGEST_E[:12] in message
    assert written.sha256["spm.model"][:12] in message


@pytest.mark.parametrize("stated", [123, 125])
def test_a_record_that_states_another_vocabulary_length_than_the_file_is_refused(
    tmp_path: Path, stated: int
) -> None:
    directory = write_model_export(
        tmp_path / "export", record_vocabulary_entries=stated
    ).directory

    message = refusal(directory)

    assert f"cannot use the export {directory}" in message
    assert f"records {stated} vocabulary entries" in message
    assert f"{directory / 'shared_vocabulary.json'} has {len(SOURCE_VOCABULARY)}" in (
        message
    )


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"a": 1}', b'["a", 2]', b"[]"],
    ids=["not json", "object", "non-string", "empty"],
)
def test_a_vocabulary_that_is_not_a_list_of_strings_is_left_to_its_reader(
    tmp_path: Path, content: bytes
) -> None:
    directory = write_model_export(
        tmp_path / "export", vocabulary_bytes=content
    ).directory

    assert verify_export(directory).record.vocabulary_entries == len(SOURCE_VOCABULARY)


def test_a_tag_the_registry_no_longer_gives_is_refused_naming_its_language_and_saying_to_re_export(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        verified_module,
        "direction_tag",
        lambda code: "<2changed>" if code == "khk_Cyrl" else f"<2{code}>",
    )

    message = refusal(written.directory)

    assert "the registry changed since the export: re-export" in message
    assert str(written.directory / RECORD_FILE) in message
    assert "direction tag for khk_Cyrl than the registry gives now" in message


def test_a_registry_that_still_gives_the_tags_of_the_record_changes_nothing(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verified_module, "direction_tag", lambda code: f"<2{code}>")

    assert verify_export(written.directory).record == written.record


def test_only_the_tag_of_a_language_the_record_lists_is_compared_with_the_registry(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def registry(code: str) -> str:
        asked.append(code)
        return f"<2{code}>"

    monkeypatch.setattr(verified_module, "direction_tag", registry)

    verify_export(written.directory)

    assert sorted(asked) == sorted(written.record.tags)


def test_a_directory_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    message = refusal(tmp_path / "nothing")

    assert f"{tmp_path / 'nothing'} does not exist" in message


def test_a_file_in_place_of_the_directory_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x", encoding="utf-8")

    assert f"{file} is not a directory" in refusal(file)


# ---- the export of the distillation (D10) ---------------------------------------------


def a_distillation_export(directory: Path) -> Path:
    """The layout D10 wrote: the converter's files, `spm.model`, `export.json` (the old
    record) and a manifest of stage `distill_export`; there is no `rinchen.json`."""
    model = write_model_export(directory, stage="distill_export")
    (model.directory / "rinchen.json").rename(model.directory / "export.json")
    return model.directory


def test_an_export_of_the_distillation_is_refused_with_the_remedy(
    tmp_path: Path,
) -> None:
    directory = a_distillation_export(tmp_path / "old")

    message = refusal(directory)

    assert "export.json" in message
    assert "rinchen.json" in message
    assert "rinchen export convert" in message
    assert str(directory) in message


def test_a_directory_with_only_the_old_record_is_refused_with_the_remedy(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "export.json").write_text("{}", encoding="utf-8")

    assert "rinchen export convert" in refusal(directory)


def test_the_old_record_is_refused_before_anything_is_hashed_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = a_distillation_export(tmp_path / "old")
    trace = Trace(monkeypatch)

    refusal(directory)

    assert trace.events == []


def test_the_old_record_does_not_matter_when_the_new_one_is_there(
    written: WrittenModel,
) -> None:
    (written.directory / "export.json").write_text("{}", encoding="utf-8")

    assert verify_export(written.directory).record == written.record


# ---- what cannot be read -------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"{}", b"[]", b"\xff\xfe\x00"],
    ids=["not-json", "empty-object", "list", "not-utf-8"],
)
def test_a_manifest_that_cannot_be_read_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    directory = write_model_export(tmp_path / "export").directory
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
    directory = write_model_export(tmp_path / "export", record_text=content).directory

    message = refusal(directory, error=DataError)

    assert message.startswith(f"{directory / RECORD_FILE}: ")


def test_a_record_of_the_old_schema_in_the_new_place_is_a_data_error(
    tmp_path: Path,
) -> None:
    old = b'{"schema": "rinchen.distill_export/1"}'
    directory = write_model_export(tmp_path / "export", record_text=old).directory

    message = refusal(directory, error=DataError)

    assert message.startswith(f"{directory / RECORD_FILE}: ")
    assert "rinchen.model/1" in message


# ---- each file is hashed once, before it is parsed -----------------------------------


class Trace:
    """What `verify_export` did, in order: `hash:<name>` for each SHA-256 taken and
    `parse:<name>` for each of the two files it parses."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[str] = []
        real_hash = verified_module.sha256_of
        real_manifest = Manifest.model_validate_json
        real_record = ModelRecord.from_bytes

        def hashing(file: Path) -> str:
            self.events.append(f"hash:{file.name}")
            return real_hash(file)

        def parsing_the_manifest(content: bytes) -> Manifest:
            self.events.append("parse:manifest.json")
            return real_manifest(content)

        def parsing_the_record(content: bytes) -> ModelRecord:
            self.events.append(f"parse:{RECORD_FILE}")
            return real_record(content)

        monkeypatch.setattr(verified_module, "sha256_of", hashing)
        monkeypatch.setattr(Manifest, "model_validate_json", parsing_the_manifest)
        monkeypatch.setattr(ModelRecord, "from_bytes", parsing_the_record)


def test_each_file_and_the_manifest_is_hashed_exactly_once_and_before_it_is_parsed(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
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


def test_the_record_is_not_parsed_before_every_file_was_hashed(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = Trace(monkeypatch)

    verify_export(written.directory)

    last_hash = max(i for i, e in enumerate(trace.events) if e.startswith("hash:"))
    first_parse_of_the_record = trace.events.index(f"parse:{RECORD_FILE}")
    assert last_hash < first_parse_of_the_record


def test_a_refused_export_hashes_nothing_beyond_what_it_needed_to_refuse(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A manifest of another stage is refused before any of the five files is hashed (the
    # model of a real teacher is a gigabyte).
    another_stage("synth")(written.directory)
    trace = Trace(monkeypatch)

    refusal(written.directory)

    assert trace.events == ["hash:manifest.json", "parse:manifest.json"]


@pytest.mark.parametrize(
    "name", ["manifest.json", RECORD_FILE], ids=["manifest", "record"]
)
def test_a_file_that_changes_after_it_was_hashed_is_not_the_one_that_is_parsed(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # A document that parses and says something else: only the comparison of the bytes
    # that are parsed with the digest that was taken tells it from the vouched one.
    real = verified_module.sha256_of
    other = json.dumps({"schema": "rinchen.model/1"}).encode("utf-8")

    def swapping(file: Path) -> str:
        digest = real(file)
        if file.name == name:
            file.write_bytes(other)
        return digest

    monkeypatch.setattr(verified_module, "sha256_of", swapping)

    message = refusal(written.directory)

    assert f"{written.directory / name} changed after it was hashed" in message


def test_the_vocabulary_is_read_as_the_bytes_that_were_hashed(
    written: WrittenModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = written.directory / "shared_vocabulary.json"
    real = verified_module.verified_content

    def swapping(path: Path, digest: str) -> bytes:
        if path == file:
            path.write_bytes(json.dumps(["a", "b", "c"]).encode("utf-8"))
        return real(path, digest)

    monkeypatch.setattr(verified_module, "verified_content", swapping)

    with pytest.raises(ConfigError, match=r"shared_vocabulary\.json changed after"):
        verify_export(written.directory)


def test_the_hash_of_the_manifest_is_that_of_the_bytes_that_were_parsed(
    written: WrittenModel,
) -> None:
    content = (written.directory / "manifest.json").read_bytes()

    assert verify_export(written.directory).manifest_sha256 == sha256_of_bytes(content)
