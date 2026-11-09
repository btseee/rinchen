"""Fabricated CTranslate2 exports for the tests of D10.

`fake_converter` replaces the converter call of `rinchen.train.distill_export` by a function
that writes what CTranslate2's converter writes for an Eole checkpoint (`model.bin`,
`shared_vocabulary.json` holding the `src` vocabulary of the checkpoint, and the small
`config.json`) into the output directory, the log beside it, and records the call; it can
also fail the way the real one does, with or without a partial `model.bin`. Nothing starts a
process.

`write_export` is a COMPLETE export directory as `rinchen distill export` leaves one: the
converter's three files, `spm.model`, `export.json` and, last, a `distill_export` manifest
with every file's real SHA-256, written through `rinchen.manifest.store`. The tests of the
teacher, of the generation and of the commands start from it, so each of them can damage one
thing and leave the rest valid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import pytest

from rinchen.config.ablation import TagPlacement
from rinchen.export.eole_ct2 import export_log_path
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.train.distill_export import (
    CT2_FILES,
    EXPORT_FILES,
    EXPORT_STAGE,
    RECORD_FILE,
    SPM_FILE,
    CheckpointRecord,
    ExportRecord,
)
from rinchen.utils.part_file import replace_file
from tests.support.step_dirs import SOURCE_VOCABULARY, SPM_BYTES

MODEL_BYTES: Final = b"fabricated ctranslate2 weights"
# What CTranslate2 4.8.4's Eole converter writes into `config.json` (spike, D10).
CONVERTER_CONFIG: Final = {
    "add_source_bos": False,
    "add_source_eos": False,
    "bos_token": "<s>",
    "decoder_start_token": "<s>",
    "eos_token": "</s>",
    "layer_norm_epsilon": 1e-06,
    "multi_query_attention": False,
    "unk_token": "<unk>",
}
# Digests of the checkpoint a fabricated export says it was made from: they vouch for no
# file, and differ from one another so that a test sees which one it reads.
WEIGHTS_SHA256: Final = "1" * 64
CONFIG_SHA256: Final = "2" * 64
VOCAB_SHA256: Final = "3" * 64
CHECKPOINT_MANIFEST_SHA256: Final = "4" * 64
_NOW: Final = "2026-11-09T10:00:00+00:00"


def converter_files(tokens: Sequence[str]) -> dict[str, bytes]:
    """The three files the converter writes for a model whose vocabulary is `tokens`."""
    return {
        "model.bin": MODEL_BYTES,
        "shared_vocabulary.json": json.dumps(list(tokens)).encode("utf-8"),
        "config.json": json.dumps(CONVERTER_CONFIG, sort_keys=True).encode("utf-8"),
    }


@dataclass(frozen=True)
class ConverterCall:
    """One call of the converter: the checkpoint directory, the output directory, the
    timeout it was given, the log and the directory its messages name."""

    checkpoint: Path
    out_dir: Path
    timeout: float | None
    log_path: Path | None = None
    reported_dir: Path | None = None


@dataclass
class FakeConverter:
    """Stands in for `export_ct2`. It makes `out_dir` (and fails, as the converter does,
    when it exists), writes the log (`log_path`, else the one beside `out_dir`),
    `model.bin` and, unless `error` is set, the other two files, then returns `out_dir`.
    With `error` it writes `partial_model` (nothing when None) as `model.bin` and raises
    `error`. `skip` names files it leaves out of a successful conversion; `replace` gives
    the bytes of files it writes instead of the usual ones."""

    calls: list[ConverterCall] = field(default_factory=list)
    error: BaseException | None = None
    partial_model: bytes | None = b"cut"
    skip: tuple[str, ...] = ()
    replace: Mapping[str, bytes] = field(default_factory=dict)

    def __call__(
        self,
        ckpt_dir: Path,
        out_dir: Path,
        *,
        timeout: float | None = None,
        log_path: Path | None = None,
        reported_dir: Path | None = None,
    ) -> Path:
        self.calls.append(
            ConverterCall(ckpt_dir, out_dir, timeout, log_path, reported_dir)
        )
        log = export_log_path(out_dir) if log_path is None else log_path
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("converter log\n", encoding="utf-8")
        out_dir.mkdir()
        if self.error is not None:
            if self.partial_model is not None:
                (out_dir / "model.bin").write_bytes(self.partial_model)
            raise self.error
        vocabulary = json.loads((ckpt_dir / "vocab.json").read_text(encoding="utf-8"))
        for name, content in converter_files(vocabulary["src"]).items():
            if name not in self.skip:
                (out_dir / name).write_bytes(self.replace.get(name, content))
        return out_dir


def fake_converter(monkeypatch: pytest.MonkeyPatch, **options: Any) -> FakeConverter:
    """Install a `FakeConverter(**options)` as the converter of the export and return it."""
    fake = FakeConverter(**options)
    monkeypatch.setattr("rinchen.train.distill_export.export_ct2", fake)
    return fake


@dataclass(frozen=True)
class WrittenExport:
    """A fabricated export: its directory, the record in `export.json`, the SHA-256 of
    each of its files and that of its manifest."""

    directory: Path
    record: ExportRecord
    sha256: Mapping[str, str]
    manifest_sha256: str


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def write_export(
    directory: Path,
    *,
    tag_placement: TagPlacement = "src",
    tokens: Sequence[str] = SOURCE_VOCABULARY,
    encoder_layers: int = 2,
    decoder_layers: int = 1,
    hidden_size: int = 64,
    checkpoint: CheckpointRecord | None = None,
    spm_bytes: bytes = SPM_BYTES,
    record_text: bytes | None = None,
    record_spm_sha256: str | None = None,
    record_vocabulary_entries: int | None = None,
    vocabulary_bytes: bytes | None = None,
    stage: str = EXPORT_STAGE,
    status: Mapping[str, str] | None = None,
    listed: Sequence[str] = EXPORT_FILES,
) -> WrittenExport:
    """A complete export in `directory` (made when missing): the converter's files for
    `tokens`, `spm.model` of `spm_bytes` and `export.json`, which records the layers and
    `hidden_size` and `len(tokens)` as vocabulary entries, and the manifest of `stage`
    (default the export's own) with `status` (default `export: ok`) listing the files
    named in `listed`, by the path they have now.

    `record_text` is the whole content of `export.json` instead of the record's;
    `record_spm_sha256` is the SentencePiece digest the record states instead of
    `spm.model`'s; `record_vocabulary_entries` the entry count the record states instead
    of `len(tokens)`; `vocabulary_bytes` the content of `shared_vocabulary.json` instead
    of the tokens'. The manifest vouches for whatever bytes the files have."""
    directory.mkdir(parents=True, exist_ok=True)
    contents = converter_files(tokens)
    if vocabulary_bytes is not None:
        contents["shared_vocabulary.json"] = vocabulary_bytes
    contents[SPM_FILE] = spm_bytes
    record = ExportRecord(
        tag_placement=tag_placement,
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
    for name in CT2_FILES + (SPM_FILE, RECORD_FILE):
        replace_file(directory / name, contents[name])
    digests = {name: sha256_of_bytes(contents[name]) for name in EXPORT_FILES}
    manifest = Manifest(
        stage=stage,
        created=_NOW,
        code_version="0000000",
        config_sha256="0" * 64,
        seed=-1,
        inputs=[],
        outputs=[
            ManifestFile(
                path=str(directory / name),
                sha256=digests[name],
                bytes=len(contents[name]),
            )
            for name in listed
        ],
        metrics={},
        tool_versions={"ctranslate2": "4.8.4"},
        status=dict({"export": "ok"} if status is None else status),
    )
    manifest_file = store.write(directory, manifest)
    return WrittenExport(
        directory=directory,
        record=record,
        sha256=digests,
        manifest_sha256=sha256_of_bytes(manifest_file.read_bytes()),
    )
