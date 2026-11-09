"""The CTranslate2 export of a checkpoint of ours, as a front-end of `rinchen.export` (D10, D11).

`export_checkpoint` turns the final step of a run `rinchen train launch` finished, or a
directory `rinchen train average` wrote, into the export directory `rinchen.export.ct2.convert`
makes (`rinchen.json`, the converter's three files, `spm.model` and, last, the manifest; the
record, its reader `verify_export` and the staging live there). What stays here is what needs
the training side: `continuation.verify_source` vouches for the checkpoint, and the saved
`config.json` and `vocab.json` of its directory say what the record states.

The SentencePiece model must be the one the checkpoint was trained with: its SHA-256 must
equal that of the copy Eole saved in the checkpoint directory. Three facts are not in
CTranslate2's files, so the caller may state them and the record keeps them, but the saved
`config.json` of a checkpoint our launcher wrote holds them (the `data` section,
`distill_saved_config`), and a statement that contradicts it is refused before anything is
made:

- the tag placement (`src` or `tgt`): from the prefixes of the corpora;
- the directions the model was trained for: from the names of the corpora
  (`saved_directions`); required from the caller when the config names none;
- whether it decodes `<doc>` windows: from the `src_prefix` of the corpora
  (`saved_document_level`); the default, when the config says nothing, is False.

A config that says nothing leaves the statement unchecked: `export_checkpoint` returns how
each fact was settled beside the record (`ExportOutcome`: `checked` or `stated` for the
placement, `derived` or `stated` for the other two; a statement equal to what the config says
is `derived`), because the record cannot say which. A saved config that names one SentencePiece
model for the source and another for the target cannot be exported: an export holds one.

Nothing is read from `config.json` or `vocab.json` before its bytes matched the digest
`verify_source` took, and the checkpoint is never written. The converter reads the weights
AFTER that digest was taken, as every command of this project reads an input after it hashed
it: the digests in the record guard against a mistake or a stale checkpoint, not against a
writer that swaps the weights in between. Nothing about the GPU is claimed: the converter runs
with the GPU hidden and no export was loaded on one.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.errors import ConfigError, DataError
from rinchen.export.ct2 import ExportFacts, convert
from rinchen.export.quantize import require_quantization
from rinchen.export.record import (
    CheckpointRecord,
    ModelRecord,
    require_directions,
    require_name,
)
from rinchen.export.verified import json_object, verified_content
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import sha256_of
from rinchen.train.checkpoints import CONFIG_FILE, VOCAB_FILE, WEIGHTS_FILE
from rinchen.train.continuation import SourceProvenance, verify_source
from rinchen.train.distill_saved_config import (
    PlacementCheck,
    require_stated_placement,
    saved_directions,
    saved_document_level,
)
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.utils.integers import is_strict_int

# How a fact the record holds was settled: `derived` from the saved config of the checkpoint
# (a statement equal to it counts), or `stated` by the caller because the config says nothing.
FactCheck = Literal["derived", "stated"]

_DIGEST_SHOWN: Final = 12


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """What `export_checkpoint` made: the `record` it wrote into `rinchen.json` and how the
    facts of the record were settled. `placement_check` is `checked` when the saved config of
    the checkpoint records the tag placement and it is the stated one, `stated` when it
    records none; `directions_check` and `document_level_check` are `derived` when the saved
    config names them (the statement, if any, equals it) and `stated` when it does not. The
    record cannot say which, so the command that reports it reads it here."""

    record: ModelRecord
    placement_check: PlacementCheck
    directions_check: FactCheck
    document_level_check: FactCheck


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    spm_model: Path,
    tag_placement: TagPlacement,
    quantization: str = "float32",
    directions: Sequence[str] | None = None,
    document_level: bool | None = None,
    name: str | None = None,
    timeout: float | None = None,
) -> ExportOutcome:
    """Export the checkpoint of ours in `checkpoint` to an export directory `out_dir` (which
    must not exist), with `spm.model`, `rinchen.json` and the manifest, and return the record
    and how its facts were settled.

    `spm_model` must be the SentencePiece model the checkpoint was trained with;
    `tag_placement` says where its direction tag goes (it must be the one the saved config
    of the checkpoint records, when that records one); `quantization` is what the converter
    stores (one of `QUANTIZATIONS`); `directions` the `<src>-<tgt>` pairs the model was
    trained for (sorted and deduplicated here; required when the saved config names none,
    and equal to the config's when it does); `document_level` whether the model decodes
    `<doc>` windows (None: the config's, else False); `name` the model's name (None: the
    default of its weights); `timeout` (seconds) bounds the converter, which runs as a
    subprocess with the GPU hidden. The manifest is written by `convert`, last.

    Raises:
        ValueError: `tag_placement` is neither `src` nor `tgt`.
        ConfigError: a path holds `$` (or, for the checkpoint and `out_dir`, a glob
            character), `spm_model` is not a file or not the model of the checkpoint, the
            quantisation, the name or the stated directions are not what a record holds,
            `tag_placement`, the directions or the document level are not the ones the saved
            config records, the saved config names no directions and none were stated, the
            saved config names two SentencePiece models, `out_dir` exists (also when it
            appears while the converter runs), or the checkpoint is not one our manifest
            vouches for (`continuation.verify_source`).
        DataError: the checkpoint's `config.json` or `vocab.json` does not say what an
            export records (a `data` section that is not what Eole writes or whose corpora
            place the tag differently included), or a file cannot be read.
        UpstreamError: the converter fails, leaves a file out or writes a vocabulary
            that is not a list of strings or has another number of entries than the
            checkpoint's.
        OSError: the output cannot be written. After the checks, any failure removes what
            this call made before it is raised.
    """
    stated_directions = _check_arguments(
        checkpoint, out_dir, spm_model, tag_placement, quantization, directions, name
    )
    provenance = verify_source(checkpoint)
    saved = _saved_model(provenance, spm_model, tag_placement)
    spm_bytes = _sentencepiece_trained_with(spm_model, saved.spm_copy)
    settled_directions, directions_check = _settled_directions(
        saved.directions, stated_directions, saved.config_file
    )
    settled_level, level_check = _settled_document_level(
        saved.document_level, document_level, saved.config_file
    )
    facts = ExportFacts(
        tag_placement=tag_placement,
        directions=settled_directions,
        document_level=settled_level,
        quantization=quantization,
        checkpoint=_checkpoint_of(provenance),
        encoder_layers=saved.encoder_layers,
        decoder_layers=saved.decoder_layers,
        hidden_size=saved.hidden_size,
        vocabulary_entries=saved.vocabulary_entries,
    )
    # Hashing a real checkpoint takes a while and the first check is older than that:
    # `convert` looks at `out_dir` again before it starts the converter.
    result = convert(
        provenance.path,
        out_dir,
        facts=facts,
        spm_bytes=spm_bytes,
        inputs=_inputs(provenance, spm_model, spm_bytes),
        name=name,
        timeout=timeout,
        extra_status={
            "tag_placement_check": saved.placement_check,
            "directions_check": directions_check,
            "document_level_check": level_check,
        },
    )
    return ExportOutcome(
        result.record, saved.placement_check, directions_check, level_check
    )


def _check_arguments(
    checkpoint: Path,
    out_dir: Path,
    spm_model: Path,
    tag_placement: object,
    quantization: str,
    directions: Sequence[str] | None,
    name: str | None,
) -> tuple[str, ...] | None:
    """Everything that can be refused before the checkpoint is hashed. Returns the stated
    directions, sorted and without repeats (None when none were stated)."""
    require_placement(tag_placement)
    require_quantization(quantization)
    if name is not None:
        require_name(name)
    stated = None if directions is None else require_directions(sorted(set(directions)))
    reject_dollar("checkpoint path", checkpoint)
    reject_glob_characters("checkpoint path", checkpoint)
    reject_dollar("output directory", out_dir)
    reject_glob_characters("output directory", out_dir)
    reject_dollar("SentencePiece model path", spm_model)
    if not spm_model.is_file():
        raise ConfigError(f"the SentencePiece model {spm_model} is not a file")
    _require_no_directory(out_dir)
    return stated


def _require_no_directory(out_dir: Path) -> None:
    if os.path.lexists(out_dir):
        raise ConfigError(
            f"{out_dir} exists: remove it, or name another output directory (the "
            "converter refuses a directory that exists)"
        )


def _settled_directions(
    derived: tuple[str, ...] | None, stated: tuple[str, ...] | None, config_file: Path
) -> tuple[tuple[str, ...], FactCheck]:
    """The directions of the record: the saved config's when it names any (a statement must
    equal them), else the statement, which is then required."""
    if derived is None:
        if stated is None:
            raise ConfigError(
                f"{config_file} names no directions the model was trained for (its data "
                "section has no corpus named <src>-<tgt>.<group>): state them with "
                "--direction <src>-<tgt>, once for each"
            )
        return stated, "stated"
    if stated is not None and stated != derived:
        raise ConfigError(
            f"the directions {', '.join(stated)} are not the ones the checkpoint was "
            f"trained for: {config_file} names {', '.join(derived)} in its data section; "
            "leave out --direction, or pass exactly those"
        )
    return derived, "derived"


def _settled_document_level(
    derived: bool | None, stated: bool | None, config_file: Path
) -> tuple[bool, FactCheck]:
    """The document level of the record: the saved config's when its corpora say (a
    statement must equal it), else the statement, else False."""
    if derived is None:
        return (False if stated is None else stated), "stated"
    if stated is not None and stated != derived:
        raise ConfigError(
            f"the document level {_word(stated)} is not the one the checkpoint was "
            f"trained with: the corpora of {config_file} "
            f"{'have' if derived else 'do not have'} <doc> in their source prefix, so "
            f"it is {_word(derived)}; leave out --document-level, or pass "
            f"{'--document-level' if derived else '--no-document-level'}"
        )
    return derived, "derived"


def _word(level: bool) -> str:
    return "true" if level else "false"


def _inputs(
    provenance: SourceProvenance, spm_model: Path, spm_bytes: bytes
) -> list[ManifestFile]:
    """The checkpoint's files and the SentencePiece model, with the SHA-256 the
    verification took before it read them (the SentencePiece model's are those of the bytes
    that were compared and are written into the export): never taken again."""
    return [
        ManifestFile(
            path=str(provenance.path / WEIGHTS_FILE), sha256=provenance.weights_sha256
        ),
        ManifestFile(
            path=str(provenance.path / CONFIG_FILE), sha256=provenance.config_sha256
        ),
        ManifestFile(
            path=str(provenance.path / VOCAB_FILE), sha256=provenance.vocab_sha256
        ),
        ManifestFile(
            path=str(spm_model),
            sha256=hashlib.sha256(spm_bytes).hexdigest(),
        ),
    ]


@dataclass(frozen=True, slots=True)
class _SavedModel:
    """What the saved `config.json` and `vocab.json` of a checkpoint say about its model:
    the numbers an export records, the copy of the SentencePiece model it was trained
    with, whether the stated tag placement was checked against them and the directions and
    document level its corpora name (None when they name none)."""

    config_file: Path
    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    vocabulary_entries: int
    spm_copy: Path
    placement_check: PlacementCheck
    directions: tuple[str, ...] | None
    document_level: bool | None


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
        config_file=config_file,
        encoder_layers=_count_at(config, ("model", "encoder", "layers"), config_file),
        decoder_layers=_count_at(config, ("model", "decoder", "layers"), config_file),
        hidden_size=_count_at(config, ("model", "encoder", "hidden_size"), config_file),
        vocabulary_entries=_vocabulary_entries(vocabulary, vocab_file),
        spm_copy=_saved_copy(config, config_file, spm_model),
        placement_check=placement_check,
        directions=saved_directions(config),
        document_level=saved_document_level(config),
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


def _hashed(file: Path) -> str:
    try:
        return sha256_of(file)
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error
