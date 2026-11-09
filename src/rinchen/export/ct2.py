"""`convert`: a checkpoint of ours becomes an export directory (D11).

    model.bin, shared_vocabulary.json, config.json   what CTranslate2's converter writes
    spm.model                                        a copy of the SentencePiece model
    rinchen.json                                     the record (`ModelRecord`)
    manifest.json                                    stage `export`, written last

The caller (`rinchen.train.distill_export`, which needs `continuation.verify_source`) has
verified the checkpoint and the SentencePiece model and gathered the facts of the record
(`ExportFacts`); `convert` makes the directory and vouches for it. Nothing is hashed twice:
`inputs` carry the digests the caller took before it read anything, and `convert` never
hashes them again; each of the five files is hashed once, after it is written, and the
manifest lists those digests. The converter reads the weights AFTER the caller hashed them,
as every command of this project reads an input after it hashed it: the digests in the
record guard against a mistake or a stale checkpoint, not against a writer that swaps the
weights in between.

The converter refuses a directory that exists, so it writes into a child of a private
directory this call makes beside the output; the two files beside the model and the
manifest are written there; and the child is renamed onto the output only when all of it is
there. The output therefore appears whole, manifest included, or not at all. Any failure
removes the private directory, and the output only if this call put it there: a directory
someone else makes at the output path in the meantime is refused and never touched. The
converter's log stays beside the output (`<out_dir>.export.log`).

The converter runs as a subprocess with the GPU hidden (`export_environment`); this module
imports no CTranslate2. Nothing about the GPU is claimed: no export was loaded on one.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError, UpstreamError
from rinchen.export.eole_ct2 import export_ct2, export_log_path
from rinchen.export.quantize import require_quantization
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
    require_directions,
    require_name,
    targets_of,
)
from rinchen.export.verified import string_list
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import code_version, sha256_of, tool_versions
from rinchen.utils.part_file import replace_file

_TOOL_PACKAGES: Final = ("ctranslate2", "eole")
# The child of the private directory that the converter makes.
_BUILT: Final = "export"
_MODEL_FILE: Final = CT2_FILES[0]
_VOCABULARY_FILE: Final = CT2_FILES[1]
# The export has no seed: nothing random happens in it.
_NO_SEED: Final = -1
_GLOB_CHARACTERS: Final = frozenset("[]*?")
_UNKNOWN_VERSION: Final = "unknown"
# The name a trial record carries before the model that gives the real one exists.
_TRIAL_NAME: Final = default_name("0" * 64)
# What `_manifest_of` says about every export; a caller's extra status adds to it.
_OWN_STATUS_KEYS: Final = ("export", "tag_placement", "quantization", "checkpoint_kind")


@dataclass(frozen=True, slots=True)
class ExportFacts:
    """What `convert` records about the checkpoint, which only the caller can know: where
    the direction tag goes, the sorted `<src>-<tgt>` directions the model was trained for,
    whether it decodes `<doc>` windows, the quantisation to convert with, the checkpoint
    and the sizes read from it (`vocabulary_entries` is the length of the checkpoint's
    source vocabulary, which the converter's `shared_vocabulary.json` must have)."""

    tag_placement: TagPlacement
    directions: tuple[str, ...]
    document_level: bool
    quantization: str
    checkpoint: CheckpointRecord
    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    vocabulary_entries: int


@dataclass(frozen=True, slots=True)
class ConvertResult:
    """What `convert` made: the `record` in `rinchen.json` and the path of the manifest."""

    record: ModelRecord
    manifest_path: Path


def convert(
    checkpoint: Path,
    out_dir: Path,
    *,
    facts: ExportFacts,
    spm_bytes: bytes,
    inputs: Sequence[ManifestFile],
    name: str | None = None,
    timeout: float | None = None,
    extra_status: Mapping[str, str] | None = None,
) -> ConvertResult:
    """Convert the checkpoint in `checkpoint` to an export directory `out_dir` (which must
    not exist) and return its record and the path of its manifest.

    `facts` are the checkpoint-derived fields of the record; `spm_bytes` the bytes of the
    SentencePiece model the checkpoint was trained with, whose SHA-256 the caller
    verified; `inputs` the manifest's inputs with the digests the caller took; `name` the
    model's name (default: `default_name` of the SHA-256 of `model.bin`); `timeout`
    (seconds) bounds the converter, which runs with the GPU hidden. The quantisation of
    `facts` is passed to the converter and recorded. `extra_status` are facts the caller
    settled (how the tag placement, the directions and the document level were found out)
    that the manifest carries beside its own status; the record does not hold them.

    Raises:
        ValueError: the placement of `facts` is neither `src` nor `tgt`, or `extra_status`
            names a key the status of an export already has.
        ConfigError: the quantisation is not offered, the directions or the name are not
            what a record holds, `checkpoint` or `out_dir` holds `$` or a glob character,
            or `out_dir` exists (also when it appears while the converter runs).
        DataError: another fact is not what a record holds (a count that is not a
            positive integer, a digest that is not a SHA-256).
        UpstreamError: the converter fails, leaves a file out or writes a vocabulary that
            is not a list of strings or has another number of entries than
            `facts.vocabulary_entries`.
        OSError: a file cannot be written. After the checks, any failure removes what this
            call made before it is raised.
    """
    _check_arguments(checkpoint, out_dir, facts, spm_bytes, name)
    status = _status_with(extra_status)
    staging = _make_private_directory(out_dir)
    built = staging / _BUILT
    published = False
    try:
        export_ct2(
            checkpoint,
            built,
            timeout=timeout,
            log_path=export_log_path(out_dir),
            reported_dir=out_dir,
            quantization=facts.quantization,
        )
        _require_the_converters_files(built, out_dir)
        _require_the_vocabulary_of(built / _VOCABULARY_FILE, out_dir, facts)
        digests = {_MODEL_FILE: sha256_of(built / _MODEL_FILE)}
        record = _record_of(
            facts,
            spm_bytes,
            default_name(digests[_MODEL_FILE]) if name is None else name,
        )
        replace_file(built / SPM_FILE, spm_bytes)
        replace_file(built / RECORD_FILE, record.to_bytes())
        for file in EXPORT_FILES:
            if file not in digests:
                digests[file] = sha256_of(built / file)
        store.write(
            built, _manifest_of(built, out_dir, record, inputs, digests, status)
        )
        _publish(built, out_dir)
        published = True
        with contextlib.suppress(OSError):
            staging.rmdir()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return ConvertResult(record, store.manifest_path(out_dir))


def _status_with(extra_status: Mapping[str, str] | None) -> dict[str, str]:
    """The `extra_status` as a dict; a key the status of an export has is a mistake of the
    caller, and an extra status never replaces `export: ok`."""
    extra = dict(extra_status or {})
    for key in _OWN_STATUS_KEYS:
        if key in extra:
            raise ValueError(
                f"extra_status holds {key!r}, which the status of an export already has"
            )
    return extra


def _check_arguments(
    checkpoint: Path,
    out_dir: Path,
    facts: ExportFacts,
    spm_bytes: bytes,
    name: str | None,
) -> None:
    """Everything that can be refused before anything is made, the converter included."""
    require_placement(facts.tag_placement)
    require_quantization(facts.quantization)
    require_directions(facts.directions)
    if name is not None:
        require_name(name)
    _reject_unsafe_characters("checkpoint path", checkpoint)
    _reject_unsafe_characters("output directory", out_dir)
    _require_no_directory(out_dir)
    # The other facts come from the checkpoint; a trial record that its own reader reads
    # back tells now, not after the converter ran, that they are not what a record holds.
    trial = _record_of(facts, spm_bytes, _TRIAL_NAME if name is None else name)
    ModelRecord.from_bytes(trial.to_bytes())


def _reject_unsafe_characters(what: str, path: Path) -> None:
    """Refuse a path the converter's loader would misread: `$` is expanded as a variable
    and `[`, `]`, `*` and `?` make a glob pattern of the checkpoint directory."""
    text = str(path)
    if "$" in text:
        raise ConfigError(
            f"the {what} holds '$', which Eole would expand as a variable"
        )
    found = next((char for char in text if char in _GLOB_CHARACTERS), None)
    if found is not None:
        raise ConfigError(
            f"the {what} {text} holds {found!r}, which Eole reads as a glob pattern when "
            "it loads a checkpoint from it"
        )


def _require_no_directory(out_dir: Path) -> None:
    if os.path.lexists(out_dir):
        raise ConfigError(
            f"{out_dir} exists: remove it, or name another output directory (the "
            "converter refuses a directory that exists)"
        )


def _make_private_directory(out_dir: Path) -> Path:
    """A new directory beside `out_dir` that only this call knows (made, with the parents of
    `out_dir`, as the converter would have made them)."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))


def _publish(built: Path, out_dir: Path) -> None:
    """Rename `built` onto `out_dir`, which must not exist. A rename also replaces an EMPTY
    directory that appears between the check and the call (and nothing else); that is the
    window that is left."""
    _require_no_directory(out_dir)
    try:
        os.rename(built, out_dir)
    except OSError:
        # Something took the path in the last instant: refuse in the usual words.
        _require_no_directory(out_dir)
        raise


def _require_the_converters_files(built: Path, out_dir: Path) -> None:
    """The three files of the converter are in `built`; the message names `out_dir`,
    where they will be."""
    for name in CT2_FILES:
        if not (built / name).is_file():
            raise UpstreamError(
                f"the CTranslate2 export left no {name} in {out_dir} (the converter's "
                f"log is {export_log_path(out_dir)})"
            )


def _require_the_vocabulary_of(file: Path, out_dir: Path, facts: ExportFacts) -> None:
    """The converter wrote its vocabulary from the checkpoint's, so `file` must hold as
    many entries as the facts state; the message names `out_dir`, where it will be."""
    shown = out_dir / file.name
    try:
        entries = string_list(file.read_bytes())
    except OSError as error:
        raise UpstreamError(f"cannot read {shown}: {error.strerror}") from error
    if entries is None:
        raise UpstreamError(
            f"{shown} is not a JSON list of strings, as the converter writes it"
        )
    if len(entries) != facts.vocabulary_entries:
        raise UpstreamError(
            f"{shown} has {len(entries)} entries but the checkpoint's vocabulary has "
            f"{facts.vocabulary_entries}: the export would map pieces to the wrong ids"
        )


def _record_of(facts: ExportFacts, spm_bytes: bytes, name: str) -> ModelRecord:
    return ModelRecord(
        name=name,
        rinchen_version=_rinchen_version(),
        tag_placement=facts.tag_placement,
        directions=facts.directions,
        languages=languages_of(facts.directions),
        tags={target: direction_tag(target) for target in targets_of(facts.directions)},
        document_level=facts.document_level,
        quantization=facts.quantization,
        checkpoint=facts.checkpoint,
        spm_sha256=hashlib.sha256(spm_bytes).hexdigest(),
        encoder_layers=facts.encoder_layers,
        decoder_layers=facts.decoder_layers,
        hidden_size=facts.hidden_size,
        vocabulary_entries=facts.vocabulary_entries,
    )


def _rinchen_version() -> str:
    try:
        return importlib.metadata.version("rinchen")
    except importlib.metadata.PackageNotFoundError:
        # Run from a source tree that was never installed.
        return _UNKNOWN_VERSION


def _manifest_of(
    built: Path,
    out_dir: Path,
    record: ModelRecord,
    inputs: Sequence[ManifestFile],
    digests: dict[str, str],
    extra_status: Mapping[str, str],
) -> Manifest:
    """The manifest of the export in `built`, which is about to become `out_dir`: the five
    files are named by the path they will have there, with the digests taken once."""
    sizes = {name: (built / name).stat().st_size for name in EXPORT_FILES}
    return Manifest(
        stage=EXPORT_STAGE,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=_choices_sha256(record),
        seed=_NO_SEED,
        inputs=list(inputs),
        outputs=[
            ManifestFile(
                path=str(out_dir / name), sha256=digests[name], bytes=sizes[name]
            )
            for name in EXPORT_FILES
        ],
        metrics={
            "encoder_layers": float(record.encoder_layers),
            "decoder_layers": float(record.decoder_layers),
            "hidden_size": float(record.hidden_size),
            "vocabulary_entries": float(record.vocabulary_entries),
            "model_bytes": float(sizes[_MODEL_FILE]),
        },
        tool_versions=tool_versions(_TOOL_PACKAGES),
        status={
            "export": "ok",
            "tag_placement": record.tag_placement,
            "quantization": record.quantization,
            "checkpoint_kind": record.checkpoint.kind,
            **extra_status,
        },
    )


def _choices_sha256(record: ModelRecord) -> str:
    """The export has no config file, so its two choices stand in for one: the SHA-256 of
    the canonical JSON (sorted keys, no spaces) of `{"quantization": ..., "tag_placement":
    ...}`."""
    canonical = json.dumps(
        {"quantization": record.quantization, "tag_placement": record.tag_placement},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
