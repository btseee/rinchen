"""Fabricated exports of the model format (`rinchen.json`) for the tests of D11.

`write_model_export` is a COMPLETE export directory as `rinchen export convert` leaves one:
the converter's three files, `spm.model`, `rinchen.json` and, last, an `export` manifest
with every file's real SHA-256, written through `rinchen.manifest.store`. The tests of the
reader, of the engine and of the commands start from it, so each of them can damage one
thing and leave the rest valid.

`fake_convert_converter` replaces the converter call of `rinchen.export.ct2` by a function
that writes what CTranslate2's converter writes for an Eole checkpoint (the three files of
`tests.support.distill_exports.converter_files`) into the output directory and the log beside
it, records the call (the quantisation included) and can fail the way the real one does,
with or without a partial `model.bin`. Nothing starts a process.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import pytest

from rinchen.config.ablation import TagPlacement
from rinchen.config.languages import direction_tag
from rinchen.export.ct2 import ExportFacts
from rinchen.export.eole_ct2 import export_log_path
from rinchen.export.record import (
    CT2_FILES,
    EXPORT_FILES,
    EXPORT_STAGE,
    RECORD_FILE,
    SPM_FILE,
    CheckpointRecord,
    ModelRecord,
    default_name,
    languages_of,
    targets_of,
)
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.utils.part_file import replace_file
from tests.support.distill_exports import (
    CHECKPOINT_MANIFEST_SHA256,
    CONFIG_SHA256,
    VOCAB_SHA256,
    WEIGHTS_SHA256,
    converter_files,
    sha256_of_bytes,
)
from tests.support.step_dirs import SOURCE_VOCABULARY, SPM_BYTES

# The directions of the toy teacher, sorted as a record holds them.
TOY_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
RINCHEN_VERSION: Final = "0.0.0"
_NOW: Final = "2026-11-09T10:00:00+00:00"


def checkpoint_record(
    kind: Literal["step", "averaged"] = "step", step: int | None = 200
) -> CheckpointRecord:
    """The checkpoint a fabricated export says it came from: a final step, or an
    averaged model (`step` None)."""
    return CheckpointRecord(
        kind=kind,
        step=step if kind == "step" else None,
        weights_sha256=WEIGHTS_SHA256,
        config_sha256=CONFIG_SHA256,
        vocab_sha256=VOCAB_SHA256,
        manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
        manifest_stage="train_launch" if kind == "step" else "train_average",
    )


def export_facts(**changes: Any) -> ExportFacts:
    """The facts of the toy teacher's export (`ExportFacts`), with `changes` applied: what
    `convert` is given for a vocabulary of `SOURCE_VOCABULARY`."""
    return replace(
        ExportFacts(
            tag_placement="src",
            directions=TOY_DIRECTIONS,
            document_level=False,
            quantization="float32",
            checkpoint=checkpoint_record(),
            encoder_layers=2,
            decoder_layers=1,
            hidden_size=64,
            vocabulary_entries=len(SOURCE_VOCABULARY),
        ),
        **changes,
    )


def checkpoint_inputs(checkpoint: Path) -> list[ManifestFile]:
    """The inputs a command names for an export of `checkpoint`: its weights, saved
    config and vocabulary and the SentencePiece model, with digests that vouch for
    nothing on disk (`convert` never hashes an input)."""
    return [
        ManifestFile(
            path=str(checkpoint / "model.00.safetensors"), sha256=WEIGHTS_SHA256
        ),
        ManifestFile(path=str(checkpoint / "config.json"), sha256=CONFIG_SHA256),
        ManifestFile(path=str(checkpoint / "vocab.json"), sha256=VOCAB_SHA256),
        ManifestFile(path="/models/spm/toy.model", sha256=sha256_of_bytes(SPM_BYTES)),
    ]


@dataclass(frozen=True)
class WrittenModel:
    """A fabricated export: its directory, the record in `rinchen.json`, the SHA-256 of
    each of its files and that of its manifest."""

    directory: Path
    record: ModelRecord
    sha256: Mapping[str, str]
    manifest_sha256: str


def write_model_export(
    directory: Path,
    *,
    placement: TagPlacement = "src",
    directions: Sequence[str] = TOY_DIRECTIONS,
    quantization: str = "float32",
    document_level: bool = False,
    tokens: Sequence[str] = SOURCE_VOCABULARY,
    name: str | None = None,
    encoder_layers: int = 2,
    decoder_layers: int = 1,
    hidden_size: int = 64,
    checkpoint: CheckpointRecord | None = None,
    spm_bytes: bytes = SPM_BYTES,
    tags: Mapping[str, str] | None = None,
    record_text: bytes | None = None,
    record_spm_sha256: str | None = None,
    record_vocabulary_entries: int | None = None,
    vocabulary_bytes: bytes | None = None,
    stage: str = EXPORT_STAGE,
    status: Mapping[str, str] | None = None,
    listed: Sequence[str] = EXPORT_FILES,
) -> WrittenModel:
    """A complete export in `directory` (made when missing): the converter's files for
    `tokens`, `spm.model` of `spm_bytes` and `rinchen.json`, and the manifest of `stage`
    (default the export's own) with `status` (default `export: ok`) listing the files named
    in `listed`, by the path they have now.

    The record is that of a model with these `directions`, `placement`, `quantization` and
    `document_level`; its `name` is `name`, else the default of the bytes of `model.bin`;
    its `tags` are the registry's unless `tags` says otherwise; it states `len(tokens)`
    vocabulary entries and the SentencePiece digest of `spm_bytes` unless
    `record_vocabulary_entries` or `record_spm_sha256` say otherwise. `record_text` is the
    whole content of `rinchen.json` instead of the record's; `vocabulary_bytes` the content
    of `shared_vocabulary.json` instead of the tokens'. The manifest vouches for whatever
    bytes the files have."""
    directory.mkdir(parents=True, exist_ok=True)
    contents = converter_files(tokens)
    if vocabulary_bytes is not None:
        contents["shared_vocabulary.json"] = vocabulary_bytes
    contents[SPM_FILE] = spm_bytes
    record = ModelRecord(
        name=default_name(sha256_of_bytes(contents["model.bin"]))
        if name is None
        else name,
        rinchen_version=RINCHEN_VERSION,
        tag_placement=placement,
        directions=tuple(directions),
        languages=languages_of(directions),
        tags={target: direction_tag(target) for target in targets_of(directions)}
        if tags is None
        else tags,
        document_level=document_level,
        quantization=quantization,
        checkpoint=checkpoint_record() if checkpoint is None else checkpoint,
        spm_sha256=sha256_of_bytes(spm_bytes)
        if record_spm_sha256 is None
        else record_spm_sha256,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        hidden_size=hidden_size,
        vocabulary_entries=len(tokens)
        if record_vocabulary_entries is None
        else record_vocabulary_entries,
    )
    contents[RECORD_FILE] = record.to_bytes() if record_text is None else record_text
    for name_of_file in (*CT2_FILES, SPM_FILE, RECORD_FILE):
        replace_file(directory / name_of_file, contents[name_of_file])
    digests = {n: sha256_of_bytes(contents[n]) for n in EXPORT_FILES}
    manifest = Manifest(
        stage=stage,
        created=_NOW,
        code_version="0000000",
        config_sha256="0" * 64,
        seed=-1,
        inputs=[],
        outputs=[
            ManifestFile(
                path=str(directory / n), sha256=digests[n], bytes=len(contents[n])
            )
            for n in listed
        ],
        metrics={},
        tool_versions={"ctranslate2": "4.8.4"},
        status=dict({"export": "ok"} if status is None else status),
    )
    manifest_file = store.write(directory, manifest)
    return WrittenModel(
        directory=directory,
        record=record,
        sha256=digests,
        manifest_sha256=sha256_of_bytes(manifest_file.read_bytes()),
    )


# ---- the converter ------------------------------------------------------------------


@dataclass(frozen=True)
class ConverterCall:
    """One call of the converter: the checkpoint directory, the output directory, the
    timeout it was given, the log, the directory its messages name and the quantisation."""

    checkpoint: Path
    out_dir: Path
    timeout: float | None
    log_path: Path | None = None
    reported_dir: Path | None = None
    quantization: str | None = None


@dataclass
class FakeConvertConverter:
    """Stands in for `export_ct2`. It makes `out_dir` (and fails, as the converter does,
    when it exists), writes the log (`log_path`, else the one beside `out_dir`),
    `model.bin` and, unless `error` is set, the other two files, then returns `out_dir`.
    The vocabulary is the `src` list of the `vocab.json` of the checkpoint when it has one,
    else `tokens`.

    With `error` it writes `partial_model` (nothing when None) as `model.bin` and raises
    `error`. `skip` names files it leaves out of a successful conversion; `replace` gives
    the bytes of files it writes instead of the usual ones; `during` is called first, as
    something else that happens while the converter runs."""

    calls: list[ConverterCall] = field(default_factory=list)
    error: BaseException | None = None
    partial_model: bytes | None = b"cut"
    skip: tuple[str, ...] = ()
    replace: Mapping[str, bytes] = field(default_factory=dict)
    tokens: Sequence[str] = SOURCE_VOCABULARY
    during: Callable[[], None] | None = None

    def __call__(
        self,
        ckpt_dir: Path,
        out_dir: Path,
        *,
        timeout: float | None = None,
        log_path: Path | None = None,
        reported_dir: Path | None = None,
        quantization: str | None = None,
    ) -> Path:
        self.calls.append(
            ConverterCall(
                ckpt_dir, out_dir, timeout, log_path, reported_dir, quantization
            )
        )
        if self.during is not None:
            self.during()
        log = export_log_path(out_dir) if log_path is None else log_path
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("converter log\n", encoding="utf-8")
        out_dir.mkdir()
        if self.error is not None:
            if self.partial_model is not None:
                (out_dir / "model.bin").write_bytes(self.partial_model)
            raise self.error
        for name, content in converter_files(self._vocabulary(ckpt_dir)).items():
            if name not in self.skip:
                (out_dir / name).write_bytes(self.replace.get(name, content))
        return out_dir

    def _vocabulary(self, ckpt_dir: Path) -> Sequence[str]:
        file = ckpt_dir / "vocab.json"
        if file.is_file():
            return json.loads(file.read_text(encoding="utf-8"))["src"]
        return self.tokens


def fake_convert_converter(
    monkeypatch: pytest.MonkeyPatch, **options: Any
) -> FakeConvertConverter:
    """Install a `FakeConvertConverter(**options)` as the converter of `convert` and
    return it."""
    fake = FakeConvertConverter(**options)
    monkeypatch.setattr("rinchen.export.ct2.export_ct2", fake)
    return fake
