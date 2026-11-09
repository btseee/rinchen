"""Tests for `rinchen.export.ct2.convert` (D11): the CTranslate2 converter's three files, a
copy of the SentencePiece model, the record and, last, the manifest, in a directory that
appears whole or not at all.

The converter refuses a directory that exists, so it writes into a child of a private
directory `convert` makes beside the output, and the child is renamed onto the output once
the manifest is in it. The converter is `tests.support.model_exports.fake_convert_converter`:
nothing starts a process. What goes wrong half way is in `test_export_convert_failures.py`.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest
from rinchen.export.ct2 import ConvertResult, ExportFacts, convert
from rinchen.export.record import EXPORT_FILES, ModelRecord
from rinchen.export.verified import verify_export

from rinchen.errors import ConfigError, DataError
from rinchen.export import ct2
from rinchen.export.eole_ct2 import export_log_path
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.distill_exports import (
    CONVERTER_CONFIG,
    MODEL_BYTES,
    sha256_of_bytes,
)
from tests.support.model_exports import (
    TOY_DIRECTIONS,
    FakeConvertConverter,
    checkpoint_inputs,
    checkpoint_record,
    export_facts,
    fake_convert_converter,
    write_model_export,
)
from tests.support.step_dirs import SOURCE_VOCABULARY, SPM_BYTES, tree_digest

MODEL_NAME = "rinchen-07148803f29a"
# The SHA-256 of {"quantization":"int8","tag_placement":"src"}, canonical JSON.
INT8_SRC_CONFIG_SHA256 = (
    "0fbdcc6aaf8d5519b858ec436506e0b7e59133a8f097fc0d55129bf144e0e690"
)


@pytest.fixture(autouse=True)
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ct2, "code_version", lambda: "abc1234")


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConvertConverter:
    return fake_convert_converter(monkeypatch)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "run" / "ckpt" / "step_200"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def run_convert(
    checkpoint: Path,
    out_dir: Path,
    *,
    using: ExportFacts | None = None,
    **options: Any,
) -> ConvertResult:
    return convert(
        checkpoint,
        out_dir,
        facts=export_facts() if using is None else using,
        spm_bytes=SPM_BYTES,
        inputs=checkpoint_inputs(checkpoint),
        **options,
    )


def names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def manifest_of(directory: Path) -> Manifest:
    return Manifest.model_validate_json((directory / "manifest.json").read_bytes())


# ---- what is made ---------------------------------------------------------------------


def test_the_output_directory_holds_the_five_files_and_the_manifest_and_nothing_else(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path, tmp_path: Path
) -> None:
    before = names(tmp_path)

    run_convert(checkpoint, out_dir)

    assert names(out_dir) == {*EXPORT_FILES, "manifest.json"}
    assert names(tmp_path) - before == {"export", "export.export.log"}
    assert export_log_path(out_dir).read_text(encoding="utf-8") == "converter log\n"


def test_the_record_says_what_the_export_is(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    result = run_convert(checkpoint, out_dir, using=export_facts(quantization="int8"))

    assert result.record == ModelRecord(
        name=MODEL_NAME,
        rinchen_version=importlib.metadata.version("rinchen"),
        tag_placement="src",
        directions=TOY_DIRECTIONS,
        languages=("eng_Latn", "khk_Cyrl", "rus_Cyrl"),
        tags={
            "eng_Latn": "<2eng_Latn>",
            "khk_Cyrl": "<2khk_Cyrl>",
            "rus_Cyrl": "<2rus_Cyrl>",
        },
        document_level=False,
        quantization="int8",
        checkpoint=checkpoint_record(),
        spm_sha256=sha256_of_bytes(SPM_BYTES),
        encoder_layers=2,
        decoder_layers=1,
        hidden_size=64,
        vocabulary_entries=len(SOURCE_VOCABULARY),
    )


def test_the_record_file_is_the_record_and_is_read_back_as_it_is(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    result = run_convert(checkpoint, out_dir)

    content = (out_dir / "rinchen.json").read_bytes()
    assert content == result.record.to_bytes()
    assert ModelRecord.from_bytes(content) == result.record


def test_the_files_beside_the_record_are_the_converters_and_a_copy_of_the_sentencepiece_model(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(checkpoint, out_dir)

    assert (out_dir / "spm.model").read_bytes() == SPM_BYTES
    assert (out_dir / "model.bin").read_bytes() == MODEL_BYTES
    assert json.loads((out_dir / "config.json").read_bytes()) == CONVERTER_CONFIG
    assert json.loads((out_dir / "shared_vocabulary.json").read_bytes()) == list(
        SOURCE_VOCABULARY
    )


def test_a_directory_that_convert_made_is_one_verify_export_accepts(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    result = run_convert(checkpoint, out_dir, using=export_facts(quantization="int8"))

    verified = verify_export(out_dir)

    assert verified.record == result.record
    assert result.manifest_path == out_dir / "manifest.json"


@pytest.mark.parametrize(
    "changes",
    [
        {"tag_placement": "tgt"},
        {"document_level": True},
        {"quantization": "float16"},
        {"quantization": "int8_float16"},
        {"directions": ("khk_Cyrl-eng_Latn",)},
        {"checkpoint": checkpoint_record("averaged")},
    ],
    ids=["tgt", "document", "float16", "int8_float16", "one-way", "averaged"],
)
def test_every_kind_of_export_is_accepted_by_its_reader(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    changes: dict[str, Any],
) -> None:
    result = run_convert(checkpoint, out_dir, using=export_facts(**changes))

    assert verify_export(out_dir).record == result.record


def test_the_same_inputs_give_the_same_record_and_the_same_name(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path
) -> None:
    first = run_convert(checkpoint, tmp_path / "first")
    second = run_convert(checkpoint, tmp_path / "second")

    assert (tmp_path / "first" / "rinchen.json").read_bytes() == (
        tmp_path / "second" / "rinchen.json"
    ).read_bytes()
    assert first.record == second.record


# ---- the name -------------------------------------------------------------------------


def test_an_export_is_named_after_its_model_unless_it_is_given_a_name(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path, tmp_path: Path
) -> None:
    default = run_convert(checkpoint, out_dir)
    named = run_convert(checkpoint, tmp_path / "named", name="mongolian-teacher.v1")

    assert default.record.name == MODEL_NAME
    assert named.record.name == "mongolian-teacher.v1"
    assert ModelRecord.from_bytes(
        (tmp_path / "named" / "rinchen.json").read_bytes()
    ).name == ("mongolian-teacher.v1")


def test_the_default_name_follows_the_model_not_the_sentencepiece_model(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_convert_converter(monkeypatch, replace={"model.bin": b"another model"})

    result = convert(
        checkpoint,
        tmp_path / "export",
        facts=export_facts(),
        spm_bytes=b"another sentencepiece model",
        inputs=[],
    )

    assert result.record.name == "rinchen-" + sha256_of_bytes(b"another model")[:12]


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "a/b", "line\nbreak"])
def test_a_name_that_is_not_usable_is_refused_before_anything_is_made(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    name: str,
) -> None:
    before = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match="1 to 64"):
        run_convert(checkpoint, out_dir, name=name)

    assert converter.calls == []
    assert tree_digest(tmp_path) == before


def test_the_version_of_the_package_is_unknown_when_it_has_no_metadata(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)

    assert run_convert(checkpoint, out_dir).record.rinchen_version == "unknown"


# ---- how the converter is called ---------------------------------------------------------


def test_the_converter_writes_into_a_private_directory_beside_the_output_not_into_it(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(checkpoint, out_dir, timeout=12.5)

    (call,) = converter.calls
    assert call.out_dir != out_dir
    assert call.out_dir.parent.parent == out_dir.parent
    assert call.out_dir.parent.name.startswith(".export.")
    assert call.out_dir.name == "export"
    assert call.checkpoint == checkpoint
    assert call.timeout == 12.5
    assert call.log_path == export_log_path(out_dir)
    assert call.reported_dir == out_dir


@pytest.mark.parametrize("quantization", ["float32", "float16", "int8", "int8_float16"])
def test_the_quantization_reaches_the_converter(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    quantization: str,
) -> None:
    result = run_convert(
        checkpoint, out_dir, using=export_facts(quantization=quantization)
    )

    (call,) = converter.calls
    assert call.quantization == quantization
    assert result.record.quantization == quantization
    assert manifest_of(out_dir).status["quantization"] == quantization


def test_a_missing_parent_directory_of_the_output_is_made(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path
) -> None:
    target = tmp_path / "deep" / "er" / "export"

    run_convert(checkpoint, target)

    assert (target / "model.bin").is_file()
    assert names(target.parent) == {"export", "export.export.log"}


def test_the_private_directory_of_another_call_is_not_touched(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    # What a killed run leaves behind cannot be told from a run in progress.
    stale = tmp_path / ".export.stale"
    stale.mkdir()
    (stale / "model.bin").write_bytes(b"theirs")

    run_convert(checkpoint, out_dir)

    assert (stale / "model.bin").read_bytes() == b"theirs"


# ---- the manifest ---------------------------------------------------------------------


def test_the_manifest_lists_the_five_files_under_the_final_directory_with_their_digests(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(checkpoint, out_dir)

    manifest = manifest_of(out_dir)
    assert [item.path for item in manifest.outputs] == [
        str(out_dir / name) for name in EXPORT_FILES
    ]
    for item in manifest.outputs:
        content = Path(item.path).read_bytes()
        assert item.sha256 == sha256_of_bytes(content)
        assert item.bytes == len(content)
        assert item.rows is None
    assert not any(".export." in item.path for item in manifest.outputs)


def test_the_manifest_lists_the_inputs_exactly_as_they_were_given(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(checkpoint, out_dir)

    assert manifest_of(out_dir).inputs == checkpoint_inputs(checkpoint)


def test_the_manifest_says_what_the_stage_did(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(checkpoint, out_dir, using=export_facts(quantization="int8"))

    manifest = manifest_of(out_dir)
    assert manifest.schema == "rinchen.manifest/1"
    assert manifest.stage == "export"
    assert manifest.seed == -1
    assert manifest.code_version == "abc1234"
    assert manifest.config_sha256 == INT8_SRC_CONFIG_SHA256
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "int8",
        "checkpoint_kind": "step",
    }
    assert manifest.metrics == {
        "encoder_layers": 2.0,
        "decoder_layers": 1.0,
        "hidden_size": 64.0,
        "vocabulary_entries": float(len(SOURCE_VOCABULARY)),
        "model_bytes": float(len(MODEL_BYTES)),
    }
    assert set(manifest.tool_versions) == {"ctranslate2", "eole"}
    created = datetime.datetime.fromisoformat(manifest.created)
    assert created.tzinfo is not None


def test_the_checkpoint_kind_of_an_averaged_model_is_in_the_status(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run_convert(
        checkpoint,
        out_dir,
        using=export_facts(checkpoint=checkpoint_record("averaged")),
    )

    assert manifest_of(out_dir).status["checkpoint_kind"] == "averaged"


def test_the_manifest_is_the_last_file_written_and_into_the_private_directory(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Path, set[str], bool]] = []
    real = store.write

    def spying(stage_dir: Path, manifest: Manifest) -> Path:
        seen.append((stage_dir, names(stage_dir), out_dir.exists()))
        return real(stage_dir, manifest)

    monkeypatch.setattr(store, "write", spying)

    run_convert(checkpoint, out_dir)

    ((stage_dir, present, published),) = seen
    assert stage_dir != out_dir
    assert stage_dir.parent.name.startswith(".export.")
    assert present == set(EXPORT_FILES)
    assert published is False


def test_each_file_is_hashed_once_and_the_inputs_never(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed: list[str] = []
    real = ct2.sha256_of

    def spying(file: Path) -> str:
        hashed.append(file.name)
        return real(file)

    monkeypatch.setattr(ct2, "sha256_of", spying)

    run_convert(checkpoint, out_dir)

    assert sorted(hashed) == sorted(EXPORT_FILES)


# ---- what is refused before anything is made -------------------------------------------


def assert_nothing_was_made(
    converter: FakeConvertConverter, tmp_path: Path, digest: str
) -> None:
    assert converter.calls == []
    assert tree_digest(tmp_path) == digest


@pytest.mark.parametrize("quantization", ["int4", "int16", "auto", "INT8", ""])
def test_a_quantization_the_export_does_not_offer_is_refused(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    quantization: str,
) -> None:
    digest = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match="float32, float16, int8, int8_float16"):
        run_convert(checkpoint, out_dir, using=export_facts(quantization=quantization))

    assert_nothing_was_made(converter, tmp_path, digest)


def test_a_placement_that_is_neither_src_nor_tgt_is_refused(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    digest = tree_digest(tmp_path)

    with pytest.raises(ValueError, match="placement must be one of src, tgt"):
        run_convert(checkpoint, out_dir, using=export_facts(tag_placement="up"))  # type: ignore[arg-type]

    assert_nothing_was_made(converter, tmp_path, digest)


@pytest.mark.parametrize(
    "directions",
    [
        (),
        ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"),
        ("eng_Latn-xxx_Latn",),
        ("eng_Latn-eng_Latn",),
    ],
    ids=["none", "unsorted", "unknown-language", "same-language"],
)
def test_directions_that_a_record_could_not_hold_are_refused(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    directions: tuple[str, ...],
) -> None:
    digest = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match="'directions'"):
        run_convert(checkpoint, out_dir, using=export_facts(directions=directions))

    assert_nothing_was_made(converter, tmp_path, digest)


@pytest.mark.parametrize(
    "changes",
    [{"hidden_size": 0}, {"encoder_layers": True}, {"vocabulary_entries": -3}],
    ids=["no-width", "bool-layers", "negative-vocabulary"],
)
def test_facts_that_a_record_could_not_hold_are_refused_before_the_converter_runs(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    digest = tree_digest(tmp_path)

    with pytest.raises(DataError, match="not a model record"):
        run_convert(checkpoint, out_dir, using=export_facts(**changes))

    assert_nothing_was_made(converter, tmp_path, digest)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_an_output_that_exists_is_refused_and_left_as_it_was(
    checkpoint: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    if kind == "directory":
        out_dir.mkdir()
        (out_dir / "mine.txt").write_text("keep", encoding="utf-8")
    else:
        out_dir.write_text("keep", encoding="utf-8")
    digest = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert_nothing_was_made(converter, tmp_path, digest)


def test_a_dangling_link_at_the_output_path_counts_as_existing(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path, tmp_path: Path
) -> None:
    out_dir.symlink_to(tmp_path / "nowhere")

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert converter.calls == []
    assert out_dir.is_symlink()


@pytest.mark.parametrize("char", ["$", "[", "]", "*", "?"])
def test_a_checkpoint_path_holding_a_character_eole_would_read_is_refused(
    converter: FakeConvertConverter,
    out_dir: Path,
    tmp_path: Path,
    char: str,
) -> None:
    checkpoint = tmp_path / f"te{char}acher" / "step_200"
    checkpoint.mkdir(parents=True)
    digest = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match="checkpoint path") as raised:
        run_convert(checkpoint, out_dir)

    assert repr(char) in str(raised.value)
    assert_nothing_was_made(converter, tmp_path, digest)


@pytest.mark.parametrize("char", ["$", "[", "]", "*", "?"])
def test_an_output_directory_holding_such_a_character_is_refused(
    checkpoint: Path,
    converter: FakeConvertConverter,
    tmp_path: Path,
    char: str,
) -> None:
    target = tmp_path / "deep" / f"out{char}dir"
    digest = tree_digest(tmp_path)

    with pytest.raises(ConfigError, match="output directory") as raised:
        run_convert(checkpoint, target)

    assert repr(char) in str(raised.value)
    assert_nothing_was_made(converter, tmp_path, digest)
    assert not (tmp_path / "deep").exists()


def test_the_texts_of_the_dollar_and_glob_refusals_are_the_ones_of_the_distillation(
    converter: FakeConvertConverter, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError) as dollar:
        run_convert(tmp_path / "c$", tmp_path / "out")
    with pytest.raises(ConfigError) as glob:
        run_convert(tmp_path / "c*", tmp_path / "out")

    assert str(dollar.value) == (
        "the checkpoint path holds '$', which Eole would expand as a variable"
    )
    assert str(glob.value) == (
        f"the checkpoint path {tmp_path / 'c*'} holds '*', which Eole reads as a glob "
        "pattern when it loads a checkpoint from it"
    )


def test_the_fabricated_export_of_the_tests_carries_the_files_convert_writes(
    checkpoint: Path, converter: FakeConvertConverter, out_dir: Path, tmp_path: Path
) -> None:
    run_convert(checkpoint, out_dir)
    fabricated = write_model_export(tmp_path / "fabricated")

    assert names(out_dir) == names(fabricated.directory)
