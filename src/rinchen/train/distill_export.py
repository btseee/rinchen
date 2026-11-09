"""The CTranslate2 export of a checkpoint of OURS, with a record of where it came from (D10).

`export_checkpoint` turns the final step of a run `rinchen train launch` finished, or a
directory `rinchen train average` wrote, into a directory CTranslate2 loads:

    model.bin, shared_vocabulary.json, config.json   what CTranslate2's converter writes
    spm.model                                        a copy of the SentencePiece model
    export.json                                      the record (`ExportRecord`)
    manifest.json                                    stage `distill_export`, written last
                                                     by the command, never by this module

The checkpoint must be one `continuation.verify_source` vouches for (its manifest lists
the files and their SHA-256 are those on disk), and the SentencePiece model must be the
one the checkpoint was trained with: its SHA-256 must equal that of the copy Eole saved in
the checkpoint directory. The tag placement is not in CTranslate2's files, so the caller
states it and the record keeps it; the saved `config.json` of a checkpoint our launcher
wrote does hold it (the prefixes of its `data` section, `distill_saved_config`), and a
statement that disagrees is refused before anything is made. A saved config without a
`data` section, or without a direction tag in it, leaves the statement unchecked and the
record cannot tell which of the two it was: `export_checkpoint` returns it beside the record
(`ExportOutcome.placement_check`) for the command to report. A saved config that names one
SentencePiece model for the source and another for the target cannot be exported: an
export holds one.
Nothing is read from `config.json` or `vocab.json` before its bytes matched the digest
`verify_source` took, and the checkpoint is never written. The converter reads the weights
AFTER that digest was taken, as every command of this project reads an input after it
hashed it: the digests in the record guard against a mistake or a stale checkpoint, not
against a writer that swaps the weights in between.

The converter refuses a directory that exists, so it writes into a child of a private
directory this call makes beside the output, and the child is renamed onto the output only
when the converter's three files are there, the vocabulary it wrote has the number of entries
the record states and the two files beside them are written. Any failure removes the private
directory, and the output only if this call put it there: a directory someone else makes at
the output path in the meantime is refused and never touched (the converter's log beside
the output stays).

`verify_export` is the reader of such a directory, for everything that trusts one (the
teacher that translates, the generation that fingerprints it): the manifest must be of
stage `distill_export` with status `export: ok` and list each of the five files once by
file name, every file's SHA-256, taken once and before the file is parsed, must be the one
the manifest lists, the record must state the SentencePiece model that is beside it and, when
`shared_vocabulary.json` is a list of strings, the number of entries it has (a file that is
anything else is refused by the reader that opens it). A
directory that was moved is accepted: the manifest's paths are not compared, the names are.
The verification is not atomic with the later load of `model.bin` by CTranslate2: it guards
against a mistake or a stale directory, not against a writer that swaps a file after the
hash. Nothing about the GPU is claimed: the converter runs with the GPU hidden
(`export_environment`) and no export was loaded on one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, get_args

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.eole_ct2 import export_ct2, export_log_path
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.manifest.stage import sha256_of
from rinchen.train.checkpoints import CONFIG_FILE, VOCAB_FILE, ContinuationKind
from rinchen.train.continuation import SourceProvenance, verify_source
from rinchen.train.distill_saved_config import (
    PlacementCheck,
    json_object,
    require_stated_placement,
    verified_content,
)
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.utils.integers import is_strict_int
from rinchen.utils.part_file import replace_file

EXPORT_STAGE: Final = "distill_export"
RECORD_FILE: Final = "export.json"
SPM_FILE: Final = "spm.model"
# What CTranslate2's converter writes for an Eole checkpoint (CTranslate2 4.8.4).
CT2_FILES: Final = ("model.bin", "shared_vocabulary.json", "config.json")
EXPORT_FILES: Final = (*CT2_FILES, SPM_FILE, RECORD_FILE)
RECORD_SCHEMA: Final = "rinchen.distill_export/1"

_STATUS_KEY: Final = "export"
# The child of the private directory that the converter makes.
_BUILT: Final = "export"
_VOCABULARY_FILE: Final = CT2_FILES[1]
_DIGEST_SHOWN: Final = 12
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_PLACEMENTS: Final = get_args(TagPlacement)
_KINDS: Final = get_args(ContinuationKind)
_RECORD_KEYS: Final = (
    "tag_placement",
    "checkpoint",
    "spm_sha256",
    "encoder_layers",
    "decoder_layers",
    "hidden_size",
    "vocabulary_entries",
)
_CHECKPOINT_KEYS: Final = (
    "kind",
    "step",
    "weights_sha256",
    "config_sha256",
    "vocab_sha256",
    "manifest_sha256",
    "manifest_stage",
)
_CHECKPOINT_DIGESTS: Final = (
    "weights_sha256",
    "config_sha256",
    "vocab_sha256",
    "manifest_sha256",
)
_COUNT_KEYS: Final = (
    "encoder_layers",
    "decoder_layers",
    "hidden_size",
    "vocabulary_entries",
)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """The checkpoint an export was made from, as `verify_source` found it: its kind, the
    step (None for an averaged model), the SHA-256 of its weights, `config.json`,
    `vocab.json` and of the manifest that vouched for them, and that manifest's stage."""

    kind: ContinuationKind
    step: int | None
    weights_sha256: str
    config_sha256: str
    vocab_sha256: str
    manifest_sha256: str
    manifest_stage: str


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """`export.json`: what the export directory says about itself.

    `tag_placement` is where the direction tag goes (`src`: in the source, `tgt`: a forced
    prefix of the target): the caller's statement, which the export refused when the saved
    config of the checkpoint places the tag elsewhere; `spm_sha256` the SHA-256 of
    `spm.model`; the layer counts, `hidden_size` and `vocabulary_entries` (the length of
    the checkpoint's source vocabulary, which the export and `verify_export` check is the
    length of `shared_vocabulary.json`) are read from the checkpoint's saved `config.json`
    and `vocab.json`.
    """

    tag_placement: TagPlacement
    checkpoint: CheckpointRecord
    spm_sha256: str
    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    vocabulary_entries: int

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of `export.json`: sorted keys, two-space indent, one final LF."""
        document = {"schema": RECORD_SCHEMA, **asdict(self)}
        text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
        return f"{text}\n".encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ExportRecord:
        """The record in `data`, which must be exactly the document `to_bytes` writes in
        shape: this schema, every field and no other, each of its type.

        Raises:
            DataError: `data` is not UTF-8, not JSON, not an object, or not that document;
                the message names the field and never quotes what it held.
        """
        document = _record_document(data)
        _require_exactly(document, _RECORD_KEYS, "")
        placement = document["tag_placement"]
        if placement not in _PLACEMENTS:
            raise _not_a_record("'tag_placement' must be 'src' or 'tgt'")
        checkpoint = document["checkpoint"]
        if not isinstance(checkpoint, dict):
            raise _not_a_record("'checkpoint' must be an object")
        _require_exactly(checkpoint, _CHECKPOINT_KEYS, "checkpoint.")
        spm_sha256 = _digest_field(document, "spm_sha256", "spm_sha256")
        counts = {key: _count_field(document, key, key) for key in _COUNT_KEYS}
        return cls(
            tag_placement=placement,
            checkpoint=_checkpoint_record(checkpoint),
            spm_sha256=spm_sha256,
            encoder_layers=counts["encoder_layers"],
            decoder_layers=counts["decoder_layers"],
            hidden_size=counts["hidden_size"],
            vocabulary_entries=counts["vocabulary_entries"],
        )


@dataclass(frozen=True, slots=True)
class VerifiedExport:
    """An export directory `verify_export` found vouched for.

    `directory` is absolute and as given (not resolved). `sha256` maps each name of
    `EXPORT_FILES` to the SHA-256 the file had when it was hashed, once; `manifest_sha256`
    is that of `manifest.json`, taken before it was parsed. `record` is the parsed
    `export.json`.
    """

    directory: Path
    record: ExportRecord
    sha256: Mapping[str, str]
    manifest_sha256: str


# ---- export_checkpoint -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """What `export_checkpoint` made: the `record` it wrote into `export.json` and how the
    tag placement of the record was settled, `placement_check`: `checked` when the saved
    config of the checkpoint records the placement and it is the stated one, `stated` when
    it records none. The record cannot say which, so the command that reports it reads it
    here."""

    record: ExportRecord
    placement_check: PlacementCheck


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    spm_model: Path,
    tag_placement: TagPlacement,
    timeout: float | None = None,
) -> ExportOutcome:
    """Export the checkpoint of ours in `checkpoint` to a CTranslate2 model in `out_dir`
    (which must not exist) with `spm.model` and `export.json`, and return the record and
    whether its tag placement was checked or only stated.

    `spm_model` must be the SentencePiece model the checkpoint was trained with;
    `tag_placement` says where its direction tag goes (it must be the one the saved config
    of the checkpoint records, when that records one); `timeout` (seconds) bounds the
    converter. The converter runs as a subprocess with the GPU hidden. No manifest is
    written: the command that calls this writes it, last.

    Raises:
        ValueError: `tag_placement` is neither `src` nor `tgt`.
        ConfigError: a path holds `$` (or, for the checkpoint and `out_dir`, a glob
            character), `spm_model` is not a file or not the model of the checkpoint,
            `tag_placement` is not the one the saved config records, the saved config
            names two SentencePiece models, `out_dir` exists (also when it appears while
            the converter runs), or the checkpoint is not one our manifest vouches for
            (`continuation.verify_source`).
        DataError: the checkpoint's `config.json` or `vocab.json` does not say what an
            export records (a `data` section that is not what Eole writes or whose corpora
            place the tag differently included), or a file cannot be read.
        UpstreamError: the converter fails, leaves a file out or writes a vocabulary
            that is not a list of strings or has another number of entries than the
            checkpoint's. After the checks, any failure removes what this call made
            before it is raised.
    """
    _check_arguments(checkpoint, out_dir, spm_model, tag_placement)
    provenance = verify_source(checkpoint)
    saved = _saved_model(provenance, spm_model, tag_placement)
    spm_bytes = _sentencepiece_trained_with(spm_model, saved.spm_copy)
    record = ExportRecord(
        tag_placement=tag_placement,
        checkpoint=_checkpoint_of(provenance),
        spm_sha256=hashlib.sha256(spm_bytes).hexdigest(),
        encoder_layers=saved.encoder_layers,
        decoder_layers=saved.decoder_layers,
        hidden_size=saved.hidden_size,
        vocabulary_entries=saved.vocabulary_entries,
    )
    # Hashing a real checkpoint takes a while and the first check is older than that.
    _require_no_directory(out_dir)
    _build_export(provenance.path, out_dir, record, spm_bytes, timeout)
    return ExportOutcome(record, saved.placement_check)


def _build_export(
    checkpoint: Path,
    out_dir: Path,
    record: ExportRecord,
    spm_bytes: bytes,
    timeout: float | None,
) -> None:
    """Make `out_dir` from the converter's files, `spm_bytes` and `record`.

    The converter refuses a directory that exists, so it writes into a child of a private
    directory this call makes beside `out_dir`, and so do the two files beside the model;
    only then is the child renamed onto `out_dir`. A failure removes the private directory,
    and `out_dir` only if this call has put it in place: a directory someone else makes at
    that path in the meantime is refused and never touched. The converter's log stays,
    beside `out_dir`."""
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
        )
        _require_the_converters_files(built, out_dir)
        _require_the_vocabulary_of(built / _VOCABULARY_FILE, out_dir, record)
        replace_file(built / SPM_FILE, spm_bytes)
        replace_file(built / RECORD_FILE, record.to_bytes())
        _publish(built, out_dir)
        published = True
        with contextlib.suppress(OSError):
            staging.rmdir()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise


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


def _check_arguments(
    checkpoint: Path, out_dir: Path, spm_model: Path, tag_placement: object
) -> None:
    """Everything that can be refused before the checkpoint is hashed."""
    require_placement(tag_placement)
    reject_dollar("checkpoint path", checkpoint)
    reject_glob_characters("checkpoint path", checkpoint)
    reject_dollar("output directory", out_dir)
    reject_glob_characters("output directory", out_dir)
    reject_dollar("SentencePiece model path", spm_model)
    if not spm_model.is_file():
        raise ConfigError(f"the SentencePiece model {spm_model} is not a file")
    _require_no_directory(out_dir)


def _require_no_directory(out_dir: Path) -> None:
    if os.path.lexists(out_dir):
        raise ConfigError(
            f"{out_dir} exists: remove it, or name another output directory (the "
            "converter refuses a directory that exists)"
        )


@dataclass(frozen=True, slots=True)
class _SavedModel:
    """What the saved `config.json` and `vocab.json` of a checkpoint say about its model:
    the numbers an export records, the copy of the SentencePiece model it was trained
    with and whether the stated tag placement was checked against them."""

    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    vocabulary_entries: int
    spm_copy: Path
    placement_check: PlacementCheck


def _saved_model(
    provenance: SourceProvenance, spm_model: Path, tag_placement: TagPlacement
) -> _SavedModel:
    """Read `config.json` and `vocab.json` of the checkpoint, each only when its bytes
    are those `verify_source` hashed, and refuse a `tag_placement` the saved config
    contradicts."""
    config_file = provenance.path / CONFIG_FILE
    config = _saved_json(config_file, provenance.config_sha256)
    placement_check = require_stated_placement(config, config_file, tag_placement)
    vocab_file = provenance.path / VOCAB_FILE
    vocabulary = _saved_json(vocab_file, provenance.vocab_sha256)
    return _SavedModel(
        encoder_layers=_count_at(config, ("model", "encoder", "layers"), config_file),
        decoder_layers=_count_at(config, ("model", "decoder", "layers"), config_file),
        hidden_size=_count_at(config, ("model", "encoder", "hidden_size"), config_file),
        vocabulary_entries=_vocabulary_entries(vocabulary, vocab_file),
        spm_copy=_saved_copy(config, config_file, spm_model),
        placement_check=placement_check,
    )


def _checkpoint_of(provenance: SourceProvenance) -> CheckpointRecord:
    return CheckpointRecord(
        kind=provenance.kind,
        step=provenance.step,
        weights_sha256=provenance.weights_sha256,
        config_sha256=provenance.config_sha256,
        vocab_sha256=provenance.vocab_sha256,
        manifest_sha256=provenance.manifest_sha256,
        manifest_stage=provenance.manifest_stage,
    )


def _saved_json(file: Path, digest: str) -> dict[str, Any]:
    """The JSON object in `file`, read once and only when its bytes are those with
    SHA-256 `digest`."""
    return json_object(verified_content(file, digest), file)


def _count_at(document: Mapping[str, Any], path: tuple[str, ...], file: Path) -> int:
    node: Any = document
    for key in path:
        node = node.get(key) if isinstance(node, Mapping) else None
    if not is_strict_int(node) or node < 1:
        raise DataError(
            f"{file}: {'.'.join(path)} is missing or is not a positive integer, as "
            "Eole saves it"
        )
    return node


def _vocabulary_entries(vocabulary: Mapping[str, Any], file: Path) -> int:
    tokens = vocabulary.get("src")
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(token, str) for token in tokens)
    ):
        raise DataError(
            f"{file}: 'src' is missing or is not a non-empty list of tokens, as Eole "
            "saves it"
        )
    return len(tokens)


def _saved_copy(config: Mapping[str, Any], config_file: Path, spm_model: Path) -> Path:
    """The copy of its SentencePiece model that Eole saved into the checkpoint directory:
    the file the saved config names under `${MODEL_PATH}/`, by its last path component."""
    configs = config.get("transforms_configs")
    section = configs.get("sentencepiece") if isinstance(configs, Mapping) else None
    recorded = (
        section.get("src_subword_model") if isinstance(section, Mapping) else None
    )
    target = section.get("tgt_subword_model") if isinstance(section, Mapping) else None
    if isinstance(recorded, str) and isinstance(target, str) and target != recorded:
        raise ConfigError(
            f"{config_file} records two SentencePiece models, {recorded} for the source "
            f"and {target} for the target: an export holds one spm.model"
        )
    name = PurePosixPath(recorded).name if isinstance(recorded, str) else ""
    if not name:
        raise ConfigError(
            f"{config_file} does not record transforms_configs.sentencepiece."
            f"src_subword_model, so there is no SentencePiece model of the checkpoint "
            f"to compare {spm_model} with"
        )
    copy = config_file.parent / name
    if not copy.is_file():
        raise ConfigError(
            f"the checkpoint has no copy of its SentencePiece model: {config_file} "
            f"names {name}, which is not a file ({copy})"
        )
    return copy


def _sentencepiece_trained_with(spm_model: Path, copy: Path) -> bytes:
    """The bytes of `spm_model`, read once, when they are those of `copy`, the model the
    checkpoint was trained with; what is written into the export is what was compared."""
    try:
        content = spm_model.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read {spm_model}: {error.strerror}") from error
    mine = hashlib.sha256(content).hexdigest()
    theirs = _hashed(copy)
    if mine != theirs:
        raise ConfigError(
            f"{spm_model} (SHA-256 {mine[:_DIGEST_SHOWN]}...) is not the SentencePiece "
            f"model the checkpoint was trained with ({copy}, SHA-256 "
            f"{theirs[:_DIGEST_SHOWN]}...): the checkpoint's vocabulary is the pieces that "
            "model gave its data, and a piece it lacks would be read as <unk>"
        )
    return content


def _require_the_converters_files(built: Path, out_dir: Path) -> None:
    """The three files of the converter are in `built`; the message names `out_dir`,
    where they will be."""
    for name in CT2_FILES:
        if not (built / name).is_file():
            raise UpstreamError(
                f"the CTranslate2 export left no {name} in {out_dir} (the converter's "
                f"log is {export_log_path(out_dir)})"
            )


def _require_the_vocabulary_of(file: Path, out_dir: Path, record: ExportRecord) -> None:
    """The converter wrote its vocabulary from the checkpoint's, so `file` must hold as
    many entries as the record states; the message names `out_dir`, where it will be."""
    shown = out_dir / file.name
    try:
        entries = _string_list(file.read_bytes())
    except OSError as error:
        raise UpstreamError(f"cannot read {shown}: {error.strerror}") from error
    if entries is None:
        raise UpstreamError(
            f"{shown} is not a JSON list of strings, as the converter writes it"
        )
    if len(entries) != record.vocabulary_entries:
        raise UpstreamError(
            f"{shown} has {len(entries)} entries but the checkpoint's vocabulary has "
            f"{record.vocabulary_entries}: the export would map pieces to the wrong ids"
        )


def _string_list(content: bytes) -> list[str] | None:
    """The JSON list of strings in `content`, None when it is anything else."""
    try:
        document = json.loads(content)
    except ValueError, RecursionError:
        return None
    if isinstance(document, list) and all(isinstance(item, str) for item in document):
        return document
    return None


# ---- the record ----------------------------------------------------------------------


def _not_a_record(problem: str) -> DataError:
    return DataError(f"not an export record: {problem}")


def _record_document(data: bytes) -> dict[str, Any]:
    def no_key_twice(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key, value in pairs:
            if key in found:
                raise _not_a_record(f"it has the key {key!r} twice")
            found[key] = value
        return found

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _not_a_record("the text is not UTF-8") from error
    try:
        document = json.loads(text, object_pairs_hook=no_key_twice)
    except (ValueError, RecursionError) as error:
        raise _not_a_record("the text is not valid JSON") from error
    if not isinstance(document, dict):
        raise _not_a_record("the document is not a JSON object")
    if document.pop("schema", None) != RECORD_SCHEMA:
        raise _not_a_record(f"'schema' must be {RECORD_SCHEMA!r}")
    return document


def _require_exactly(
    document: Mapping[str, Any], keys: tuple[str, ...], prefix: str
) -> None:
    for key in keys:
        if key not in document:
            raise _not_a_record(f"{prefix + key!r} is missing")
    extra = sorted(set(document) - set(keys))
    if extra:
        raise _not_a_record(f"it has the unknown field {prefix + extra[0]!r}")


def _digest_field(document: Mapping[str, Any], key: str, name: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _not_a_record(
            f"{name!r} must be a SHA-256 (64 lowercase hexadecimal characters)"
        )
    return value


def _count_field(document: Mapping[str, Any], key: str, name: str) -> int:
    value = document[key]
    if not is_strict_int(value) or value < 1:
        raise _not_a_record(f"{name!r} must be a positive integer")
    return value


def _checkpoint_record(checkpoint: Mapping[str, Any]) -> CheckpointRecord:
    kind = checkpoint["kind"]
    if kind not in _KINDS:
        raise _not_a_record("'checkpoint.kind' must be 'step' or 'averaged'")
    step = checkpoint["step"]
    if kind == "step":
        step = _count_field(checkpoint, "step", "checkpoint.step")
    elif step is not None:
        raise _not_a_record("'checkpoint.step' must be null for an averaged model")
    digests = {
        key: _digest_field(checkpoint, key, f"checkpoint.{key}")
        for key in _CHECKPOINT_DIGESTS
    }
    stage = checkpoint["manifest_stage"]
    if not isinstance(stage, str):
        raise _not_a_record("'checkpoint.manifest_stage' must be text")
    return CheckpointRecord(
        kind=kind,
        step=step,
        weights_sha256=digests["weights_sha256"],
        config_sha256=digests["config_sha256"],
        vocab_sha256=digests["vocab_sha256"],
        manifest_sha256=digests["manifest_sha256"],
        manifest_stage=stage,
    )


# ---- verify_export ---------------------------------------------------------------------


def verify_export(directory: Path) -> VerifiedExport:
    """The export in `directory` when its manifest vouches for its five files and its
    record agrees with them (see the module's text); raises when it is anything else.
    Reads the files and writes nothing.

    Raises:
        ConfigError: the directory is not a usable export: no manifest, the manifest of
            another stage or without `export: ok`, a file missing, not listed or listed
            twice, a file whose SHA-256 is not the manifest's, a record that states
            another SentencePiece model than the one beside it, or a file that changed
            after it was hashed. The message names the file and what is wrong.
        DataError: the manifest or the record cannot be read.
    """
    path = Path(os.path.abspath(directory))
    if not os.path.lexists(path):
        raise ConfigError(f"the export directory {path} does not exist")
    if not path.is_dir():
        raise ConfigError(f"the export directory {path} is not a directory")
    manifest_file = store.manifest_path(path)
    if not manifest_file.is_file():
        raise _refuse(
            path,
            f"no manifest vouches for it: {manifest_file} does not exist (`rinchen "
            "distill export` writes it when it finishes)",
        )
    manifest_sha256 = _hashed(manifest_file)
    manifest = _vouching_manifest(path, manifest_file, manifest_sha256)
    listed = _listed_digests(path, manifest, manifest_file)
    for name in EXPORT_FILES:
        _require_a_regular_file(path, path / name)
    digests: dict[str, str] = {}
    for name in EXPORT_FILES:
        digest = _hashed(path / name)
        if digest != listed[name]:
            raise _refuse(
                path,
                f"{path / name} differs from the manifest {manifest_file}: its SHA-256 "
                f"is {digest[:_DIGEST_SHOWN]}... and the manifest lists "
                f"{listed[name][:_DIGEST_SHOWN]}...; the file changed after the "
                "manifest vouched for it",
            )
        digests[name] = digest
    record = _record_in(path / RECORD_FILE, digests[RECORD_FILE])
    if record.spm_sha256 != digests[SPM_FILE]:
        raise _refuse(
            path,
            f"{path / RECORD_FILE} records the SentencePiece model with SHA-256 "
            f"{record.spm_sha256[:_DIGEST_SHOWN]}... but {path / SPM_FILE} has "
            f"{digests[SPM_FILE][:_DIGEST_SHOWN]}...: the export holds another model "
            "than the one it says it was made with",
        )
    _require_the_vocabulary_to_agree(path, record, digests[_VOCABULARY_FILE])
    return VerifiedExport(
        directory=path,
        record=record,
        sha256=MappingProxyType(digests),
        manifest_sha256=manifest_sha256,
    )


def _require_the_vocabulary_to_agree(
    path: Path, record: ExportRecord, digest: str
) -> None:
    """The record states how many entries the vocabulary has; a file of another length
    maps the pieces to the wrong ids. A file that is not a non-empty list of strings is
    left to its reader (`distill_export_files.shared_vocabulary`), which refuses it."""
    file = path / _VOCABULARY_FILE
    entries = _string_list(verified_content(file, digest))
    if entries and len(entries) != record.vocabulary_entries:
        raise _refuse(
            path,
            f"{path / RECORD_FILE} records {record.vocabulary_entries} vocabulary "
            f"entries but {file} has {len(entries)}: the export holds another "
            "vocabulary than the one it says it was made with",
        )


def _refuse(path: Path, problem: str) -> ConfigError:
    return ConfigError(f"cannot use the export {path}: {problem}")


def _hashed(file: Path) -> str:
    try:
        return sha256_of(file)
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error


def _vouching_manifest(path: Path, manifest_file: Path, digest: str) -> Manifest:
    content = verified_content(manifest_file, digest)
    try:
        manifest = Manifest.model_validate_json(content)
    except ValueError as error:
        # The reader's message holds the validation errors, which quote the input.
        raise DataError(
            f"{manifest_file}: is not a readable Rinchen manifest"
        ) from error
    if manifest.stage != EXPORT_STAGE:
        raise _refuse(
            path,
            f"{manifest_file} is the manifest of stage {manifest.stage!r}, not of "
            f"{EXPORT_STAGE!r}",
        )
    if manifest.status.get(_STATUS_KEY) != "ok":
        raise _refuse(
            path, f"the status in {manifest_file} does not say {_STATUS_KEY}: ok"
        )
    return manifest


def _listed_digests(
    path: Path, manifest: Manifest, manifest_file: Path
) -> dict[str, str]:
    """The SHA-256 the manifest lists for each file of an export, by file name (the paths
    are those the files had when it was written, and the directory may have moved since)."""
    by_name: dict[str, list[str]] = {}
    for item in manifest.outputs:
        by_name.setdefault(Path(item.path).name, []).append(item.sha256)
    listed: dict[str, str] = {}
    for name in EXPORT_FILES:
        entries = by_name.get(name, [])
        if not entries:
            raise _refuse(path, f"the manifest {manifest_file} does not list {name}")
        if len(entries) > 1:
            raise _refuse(
                path,
                f"the manifest {manifest_file} lists {name} more than once, so it "
                "does not say which file it means",
            )
        listed[name] = entries[0]
    return listed


def _require_a_regular_file(path: Path, file: Path) -> None:
    if not os.path.lexists(file):
        raise _refuse(path, f"{file} is missing")
    if not file.is_file():
        raise _refuse(path, f"{file} is not a regular file")


def _record_in(file: Path, digest: str) -> ExportRecord:
    content = verified_content(file, digest)
    try:
        return ExportRecord.from_bytes(content)
    except DataError as error:
        raise DataError(f"{file}: {error}") from error
