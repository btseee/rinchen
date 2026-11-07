"""The converter's own character-level SentencePiece model.

The converter reads and writes characters: one piece per character, plus the two
target tags as single pieces. This model is not one of the language models of
`rinchen.spm`, which are unigram, byte-fallback and trained per language.
`train_char_model` trains it, `check_char_model` verifies a trained one,
`load_char_processor` opens one and `load_char_model_config` reads
`configs/spm/converter_char.yaml`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import sentencepiece
import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from rinchen.config.loader import anchor_at_repo_root, reject_blank_path
from rinchen.errors import ConfigError, DataError, UpstreamError

# The single source for the symbols the model must keep whole: each one tells the
# converter which script to write.
CONVERTER_TAGS = ("<2mn_Mong>", "<2mn_Cyrl>")

# U+2581 LOWER ONE EIGHTH BLOCK: SentencePiece's default dummy prefix, which it puts
# in front of a text that does not start with a space.
_DUMMY_PREFIX = chr(0x2581)

# 2 keeps the trainer's ERROR and FATAL messages and drops its INFO/WARNING flood.
_TRAINER_MIN_LOG_LEVEL = 2

# Strict so a quoted number or a YAML boolean is a typo to report, not a value to
# coerce.
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
# Anchoring a blank path would put the model outside the repository.
_RepoPath = Annotated[
    Path, BeforeValidator(reject_blank_path), AfterValidator(anchor_at_repo_root)
]


class CharModelConfig(BaseModel):
    """One training run of the converter's character model. There is no seed: the
    character trainer is deterministic. A relative `output_prefix` is anchored at the
    repo root, so the config means the same thing from any working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vocab_size: _PositiveInt
    output_prefix: _RepoPath  # the trained model is <output_prefix>.model


def load_char_model_config(path: Path) -> CharModelConfig:
    """Load and validate one `configs/spm/converter_char.yaml`-shaped file. Raises
    ConfigError on any violation (missing, unreadable or unparseable file,
    non-mapping top level, missing/extra field, out-of-range or wrongly typed
    value)."""
    if not path.is_file():
        raise ConfigError(f"char model config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid char model config {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid char model config {path}: top level must be a mapping of "
            f"field names to values, got {type(raw).__name__}"
        )
    try:
        return CharModelConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid char model config {path}: {exc}") from exc


def train_char_model(texts: Iterable[str], prefix: Path, *, vocab_size: int) -> Path:
    """Train the character model on `texts`, one word per line, and return its path,
    `<prefix>.model`; the trainer writes `<prefix>.vocab` beside it. The texts pass
    through `<prefix>.input.txt`, which is removed afterwards. The directory of
    `prefix` is created even when training then fails. Blank texts are skipped.

    `vocab_size` is only a cap: the model has one piece per distinct character plus
    the special pieces and tags, so the tiny reference list trains to 73 pieces at a
    `vocab_size` of 512. A cap below that count is not an error either: the trainer
    drops the rarest characters. Run `check_char_model` afterwards on the training
    texts, as a collection that can be read again (a generator is spent by this call).

    Training is deterministic: the same texts into the same `prefix` give
    byte-identical `.model` and `.vocab` files. The model file embeds the absolute
    prefix, so its hash holds per location only; the `.vocab` file (pieces and
    scores) is the location-independent identity of the model. Record both hashes.

    Raises:
        DataError: `texts` holds no non-blank text.
        UpstreamError: SentencePiece rejected the training run, for example because
            `vocab_size` cannot hold the special pieces and tags; the trainer's
            `RuntimeError` is chained as the cause.
    """
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Nothing is renamed into place, and a fixed name keeps the model file free of a
    # random path (the trainer records its input path in the model).
    input_path = Path(f"{prefix}.input.txt")
    try:
        if _write_non_blank_lines(texts, input_path) == 0:
            raise DataError(f"cannot train {prefix}: no non-blank text")
        _run_trainer(input_path, prefix, vocab_size)
    finally:
        input_path.unlink(missing_ok=True)
    return Path(f"{prefix}.model")


def _write_non_blank_lines(texts: Iterable[str], path: Path) -> int:
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            # Only U+0020 is a space for the converter's words; U+202F is part of one.
            if text.strip(" "):
                handle.write(f"{text}\n")
                written += 1
    return written


def _run_trainer(input_path: Path, prefix: Path, vocab_size: int) -> None:
    try:
        sentencepiece.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type="char",
            character_coverage=1.0,
            byte_fallback=False,
            normalization_rule_name="identity",
            add_dummy_prefix=True,
            user_defined_symbols=list(CONVERTER_TAGS),
            hard_vocab_limit=False,
            minloglevel=_TRAINER_MIN_LOG_LEVEL,
        )
    except RuntimeError as exc:
        raise UpstreamError(
            f"sentencepiece training failed for {prefix}: {exc}"
        ) from exc


def check_char_model(model: Path, texts: Iterable[str]) -> None:
    """Check the model at `model` against the converter's invariants: each of
    `CONVERTER_TAGS` is a single piece, and every text in `texts` has only known
    characters and decodes back to itself after encoding. Texts are whole words, so
    a gap or selector inside a word is exercised as the converter meets it. Stops at
    the first failure.

    `texts` must be a collection that can be read again, not the generator that fed
    `train_char_model`: a spent generator, like an empty `texts`, checks only the tags.

    Raises:
        DataError: the model cannot be loaded (the message names the file, and the
            cause is chained except for a directory), a tag is not a single piece, or
            a text has an unknown character or does not round-trip; the message names
            the tag, the character (by code point) or the text.
    """
    processor = load_char_processor(model)
    for tag in CONVERTER_TAGS:
        defect = _tag_defect(processor, tag)
        if defect is not None:
            raise DataError(f"{model}: tag {tag} {defect}")
    for text in texts:
        _check_text(processor, model, text)


def load_char_processor(model: Path) -> sentencepiece.SentencePieceProcessor:
    """The SentencePiece processor of the model file at `model`.

    Raises:
        DataError: `model` is a directory, cannot be read or is not a model; the
            library's error is chained except for a directory.
    """
    # SentencePiece aborts the whole interpreter on a directory, so it never gets to
    # see one.
    if model.is_dir():
        raise DataError(f"cannot load SentencePiece model {model}: it is a directory")
    try:
        return sentencepiece.SentencePieceProcessor(model_file=str(model))
    except (OSError, RuntimeError) as exc:
        # OSError: missing or unreadable file. RuntimeError: not a model.
        raise DataError(f"cannot load SentencePiece model {model}: {exc}") from exc


def _tag_defect(
    processor: sentencepiece.SentencePieceProcessor, tag: str
) -> str | None:
    """Why `tag` is not a single piece, or None when it is: it encodes to its own id,
    alone or after the dummy prefix. Ids, not piece strings, because a run of unknown
    characters comes back as one piece spelled like the original text, and a piece
    that exists but is not a user-defined symbol (a control symbol) is never matched
    in text."""
    tag_id = processor.piece_to_id(tag)
    if tag_id == processor.unk_id():
        return "is not a piece of the model"
    ids = processor.encode(tag)
    if ids == [tag_id] or ids == [processor.piece_to_id(_DUMMY_PREFIX), tag_id]:
        return None
    return (
        f"is not matched as a symbol: it encodes to ids {ids}, not to its id {tag_id}"
    )


def _check_text(
    processor: sentencepiece.SentencePieceProcessor, model: Path, text: str
) -> None:
    unknown = _first_unknown_character(processor, text)
    if unknown is not None:
        raise DataError(
            f"{model}: character U+{ord(unknown):04X} of {text!a} "
            "is unknown to the model"
        )
    decoded = processor.decode(processor.encode(text))
    if decoded != text:
        raise DataError(
            f"{model}: {text!a} does not round-trip: it decodes to {decoded!a}"
        )


def _first_unknown_character(
    processor: sentencepiece.SentencePieceProcessor, text: str
) -> str | None:
    unknown_id = processor.unk_id()
    if unknown_id not in processor.encode(text):
        return None
    return next((char for char in text if unknown_id in processor.encode(char)), None)
