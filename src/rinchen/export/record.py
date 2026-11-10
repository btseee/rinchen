"""The record of a model export: `rinchen.json`, schema `rinchen.model/1` (D11).

An export directory (`rinchen.export.verified`, `rinchen.export.ct2`) holds

    model.bin, shared_vocabulary.json, config.json   what CTranslate2's converter writes
    spm.model                                        a copy of the SentencePiece model
    rinchen.json                                     the record, `ModelRecord`
    manifest.json                                    stage `export`, written last

`rinchen.json` says what the converter's files cannot: where the direction tag goes, which
directions the model was trained for and the tag of each target (the vocabulary does not say:
every vocabulary holds all the reserved symbols), whether it decodes `<doc>` windows, how it
was quantised, which checkpoint of ours it came from, and the sizes read from that
checkpoint. It is UTF-8 JSON with sorted keys, two-space indent, one final LF and no
timestamp, so the same inputs give the same bytes.

`ModelRecord.from_bytes` is as strict as the writer is exact: this schema, every field and no
other, each of its type, `languages` the union of the languages of `directions`, `tags` one
entry for each target of `directions`. A refusal is a `DataError` that names the field and
never quotes what it held: the file may come from anywhere. Whether a `tags` entry is still
the registry's tag is not the record's to know: `verify_export` checks it.

The record replaces D10's `export.json` (schema `rinchen.distill_export/1`); there is no
second reader.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from rinchen.config.ablation import TagPlacement
from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, DataError
from rinchen.export.quantize import QUANTIZATIONS
from rinchen.utils.integers import is_strict_int

EXPORT_STAGE: Final = "export"
RECORD_FILE: Final = "rinchen.json"
# The record of the format D10 wrote; `verify_export` refuses a directory that has only it.
LEGACY_RECORD_FILE: Final = "export.json"
SPM_FILE: Final = "spm.model"
# What CTranslate2's converter writes for an Eole checkpoint (CTranslate2 4.8.4).
CT2_FILES: Final = ("model.bin", "shared_vocabulary.json", "config.json")
EXPORT_FILES: Final = (*CT2_FILES, SPM_FILE, RECORD_FILE)
RECORD_SCHEMA: Final = "rinchen.model/1"
# How a checkpoint was made: the final step of a run, or an averaged directory. Defined
# here, once; `rinchen.train.checkpoints` imports it.
ContinuationKind = Literal["step", "averaged"]

_NAME: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")
# The characters of a package version (PEP 440), at most 64 of them.
_VERSION: Final = re.compile(r"[A-Za-z0-9._+!-]{1,64}")
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_PLACEMENTS: Final = get_args(TagPlacement)
_KINDS: Final = get_args(ContinuationKind)
_RECORD_KEYS: Final = (
    "name",
    "rinchen_version",
    "tag_placement",
    "directions",
    "languages",
    "tags",
    "document_level",
    "quantization",
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
_NAME_RULE: Final = "1 to 64 characters: letters, digits, '.', '_' and '-'"


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
class ModelRecord:
    """`rinchen.json`: what the export directory says about itself.

    `name` is how the model is called in logs and metrics (default `default_name`);
    `rinchen_version` the installed package's version at export; `tag_placement` where the
    direction tag goes (`src`: in the source, `tgt`: a forced prefix of the target);
    `directions` the sorted `<src>-<tgt>` registry codes the model was trained for;
    `languages` the sorted union of their codes; `tags` the direction tag of each TARGET of
    `directions`; `document_level` whether the model decodes `<doc>` windows; `quantization`
    what the converter was asked for; `checkpoint` where the weights came from;
    `spm_sha256` the SHA-256 of `spm.model`; the layer counts, `hidden_size` and
    `vocabulary_entries` (the length of `shared_vocabulary.json`) are read from the
    checkpoint. `tags` is read-only whatever mapping it was built from.
    """

    name: str
    rinchen_version: str
    tag_placement: TagPlacement
    directions: tuple[str, ...]
    languages: tuple[str, ...]
    tags: Mapping[str, str]
    document_level: bool
    quantization: str
    checkpoint: CheckpointRecord
    spm_sha256: str
    encoder_layers: int
    decoder_layers: int
    hidden_size: int
    vocabulary_entries: int

    def __post_init__(self) -> None:
        # A frozen record with a mutable member would not be one.
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "languages", tuple(self.languages))
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    @property
    def targets(self) -> tuple[str, ...]:
        """The languages the model can be asked to write: the sorted keys of `tags`."""
        return tuple(sorted(self.tags))

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of `rinchen.json`: sorted keys, two-space indent, one final LF."""
        document = {
            "schema": RECORD_SCHEMA,
            "name": self.name,
            "rinchen_version": self.rinchen_version,
            "tag_placement": self.tag_placement,
            "directions": list(self.directions),
            "languages": list(self.languages),
            "tags": dict(self.tags),
            "document_level": self.document_level,
            "quantization": self.quantization,
            "checkpoint": asdict(self.checkpoint),
            "spm_sha256": self.spm_sha256,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "hidden_size": self.hidden_size,
            "vocabulary_entries": self.vocabulary_entries,
        }
        text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
        return f"{text}\n".encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelRecord:
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
        quantization = document["quantization"]
        if quantization not in QUANTIZATIONS:
            raise _not_a_record(
                f"'quantization' must be one of {', '.join(QUANTIZATIONS)}"
            )
        if not isinstance(document["document_level"], bool):
            raise _not_a_record("'document_level' must be true or false")
        checkpoint = document["checkpoint"]
        if not isinstance(checkpoint, dict):
            raise _not_a_record("'checkpoint' must be an object")
        _require_exactly(checkpoint, _CHECKPOINT_KEYS, "checkpoint.")
        directions = _directions(document["directions"])
        counts = {key: _count_field(document, key, key) for key in _COUNT_KEYS}
        return cls(
            name=_text_field(document, "name", _NAME, f"must be {_NAME_RULE}"),
            rinchen_version=_text_field(
                document,
                "rinchen_version",
                _VERSION,
                "must be a package version (1 to 64 characters of a version)",
            ),
            tag_placement=placement,
            directions=directions,
            languages=_languages(document["languages"], directions),
            tags=_tags(document["tags"], directions),
            document_level=document["document_level"],
            quantization=quantization,
            checkpoint=_checkpoint_record(checkpoint),
            spm_sha256=_digest_field(document, "spm_sha256", "spm_sha256"),
            encoder_layers=counts["encoder_layers"],
            decoder_layers=counts["decoder_layers"],
            hidden_size=counts["hidden_size"],
            vocabulary_entries=counts["vocabulary_entries"],
        )


def default_name(model_sha256: str) -> str:
    """The name of an export that was given none: `rinchen-` and the first 12 hexadecimal
    digits of the SHA-256 of its `model.bin`."""
    return f"rinchen-{model_sha256[:12]}"


def require_name(value: str) -> str:
    """`value` when it is a name a record can hold.

    Raises:
        ConfigError: it is not 1 to 64 characters of letters, digits, `.`, `_` and `-`.
    """
    if not _NAME.fullmatch(value):
        raise ConfigError(f"a model name must be {_NAME_RULE}")
    return value


def require_directions(directions: Sequence[str]) -> tuple[str, ...]:
    """`directions` as a tuple when a record can hold them: at least one, each
    `<source>-<target>` of two distinct language codes of the registry, sorted and without
    repeats.

    Raises:
        ConfigError: they are anything else; the message says what is wrong and shows them.
    """
    problem = _directions_problem(directions)
    if problem is not None:
        raise ConfigError(f"{problem} (got {list(directions)!r})")
    return tuple(directions)


def languages_of(directions: Sequence[str]) -> tuple[str, ...]:
    """The sorted union of the languages of `directions`, which must be well formed
    (`require_directions`)."""
    return tuple(
        sorted({code for direction in directions for code in _pair(direction)})
    )


def targets_of(directions: Sequence[str]) -> tuple[str, ...]:
    """The sorted distinct targets of `directions`, which must be well formed
    (`require_directions`)."""
    return tuple(sorted({_pair(direction)[1] for direction in directions}))


def _pair(direction: str) -> tuple[str, str]:
    source, target = direction.split("-")
    return source, target


# ---- reading the document ---------------------------------------------------------------


def _not_a_record(problem: str) -> DataError:
    return DataError(f"not a model record: {problem}")


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


def _text_field(
    document: Mapping[str, Any], key: str, pattern: re.Pattern[str], must: str
) -> str:
    value = document[key]
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise _not_a_record(f"{key!r} {must}")
    return value


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


# ---- directions, languages and tags --------------------------------------------------------


def _directions(value: object) -> tuple[str, ...]:
    problem = _directions_problem(value)
    if problem == _UNKNOWN_LANGUAGE:
        # The record was written when the registry had the language: the export is stale.
        problem += (
            ": the registry changed since the export: re-export the checkpoint with "
            "`rinchen export convert`"
        )
    if problem is not None:
        raise _not_a_record(problem)
    assert isinstance(value, list)
    return tuple(value)


_UNKNOWN_LANGUAGE: Final = (
    "'directions' holds a language that is not in the language registry"
)


def _directions_problem(value: object) -> str | None:
    """What is wrong with `value` as a record's `directions`, in words that quote none of
    it; None when nothing is."""
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(entry, str) for entry in value
    ):
        return "'directions' must be a list of text"
    if not value:
        return "'directions' must name at least one direction"
    known = known_language_codes()
    for direction in value:
        parts = direction.split("-")
        if len(parts) != 2 or not all(parts):
            return "'directions' holds an entry that is not <source>-<target>"
        if not set(parts) <= known:
            return _UNKNOWN_LANGUAGE
        if parts[0] == parts[1]:
            return "'directions' holds a direction from a language to itself"
    if list(value) != sorted(set(value)):
        return "'directions' must be sorted and without repeats"
    return None


def _languages(value: object, directions: tuple[str, ...]) -> tuple[str, ...]:
    expected = languages_of(directions)
    if value != list(expected):
        raise _not_a_record("'languages' must be the sorted languages of 'directions'")
    return expected


def _tags(value: object, directions: tuple[str, ...]) -> Mapping[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(tag, str) and tag for tag in value.values()
    ):
        raise _not_a_record("'tags' must be an object of text")
    if sorted(value) != list(targets_of(directions)):
        raise _not_a_record(
            "'tags' must have one entry for each target of 'directions'"
        )
    return value
