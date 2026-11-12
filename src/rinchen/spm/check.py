"""Validate a trained SentencePiece model against the project invariants.

`check_spm` reports, as values, whether every reserved symbol is a single
piece, whether byte fallback is on, and how much of each language's text the
model covers without falling back to unknown or byte pieces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import sentencepiece

from rinchen.errors import DataError
from rinchen.spm.corpus import read_lines
from rinchen.spm.symbols import user_defined_symbols

# U+2581 LOWER ONE EIGHTH BLOCK: SentencePiece's default dummy prefix, which
# it puts in front of a text that does not start with a space.
_DUMMY_PREFIX = chr(0x2581)
_BYTE_PIECES = tuple(f"<0x{value:02X}>" for value in range(256))


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    """How many pieces a language's sample encoded to, and how many of them
    were unknown or byte pieces."""

    lang: str
    pieces: int
    fallback_pieces: int

    @property
    def coverage(self) -> float:
        """The share of pieces that did not fall back; 1.0 for an empty sample."""
        if self.pieces == 0:
            return 1.0
        return 1 - self.fallback_pieces / self.pieces


@dataclass(frozen=True, slots=True)
class SpmCheckReport:
    """What `check_spm` found. `non_atomic_symbols` follow the order of
    `user_defined_symbols`; `coverage` follows the order of the corpus mapping."""

    non_atomic_symbols: tuple[str, ...]
    byte_fallback_on: bool
    coverage: tuple[LanguageCoverage, ...]

    @property
    def ok(self) -> bool:
        """True when every reserved symbol is atomic and byte fallback is on.
        Coverage is informational and does not decide it."""
        return not self.non_atomic_symbols and self.byte_fallback_on


def check_spm(
    model_path: Path,
    corpus_files: Mapping[str, Path],
    *,
    sample_lines: int = 1000,
) -> SpmCheckReport:
    """Check the model at `model_path` and measure its coverage of each
    language's first `sample_lines` non-blank lines in `corpus_files`. A file is
    not read past that sample, so corruption beyond it goes unnoticed.

    A symbol is atomic when it encodes to itself, alone or after the dummy
    prefix. Byte fallback is on when all 256 byte pieces `<0x00>`..`<0xFF>`
    exist. A failed check is reported in the returned value, not raised.

    Raises:
        ValueError: `sample_lines` is not positive.
        DataError: the model cannot be loaded, or a corpus file cannot be read
            or decoded; the message names the file and the cause is chained
            (except for a model path that is a directory).
    """
    if sample_lines <= 0:
        raise ValueError(f"sample_lines must be > 0, got {sample_lines}")
    processor = _load_processor(model_path)
    return SpmCheckReport(
        non_atomic_symbols=tuple(
            symbol
            for symbol in user_defined_symbols()
            if not _is_atomic(processor, symbol)
        ),
        byte_fallback_on=_has_byte_fallback(processor),
        coverage=tuple(
            _measure_coverage(processor, lang, path, sample_lines)
            for lang, path in corpus_files.items()
        ),
    )


def _load_processor(model_path: Path) -> sentencepiece.SentencePieceProcessor:
    # SentencePiece aborts the whole interpreter on a directory, so it never
    # gets to see one.
    if model_path.is_dir():
        raise DataError(
            f"cannot load SentencePiece model {model_path}: it is a directory"
        )
    try:
        return sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    except (OSError, RuntimeError) as exc:
        # SentencePiece 0.2.2 raises RuntimeError for a missing or unreadable file and for a file
        # that is not a model; OSError is kept for a file error raised by Python itself.
        raise DataError(f"cannot load SentencePiece model {model_path}: {exc}") from exc


def _is_atomic(processor: sentencepiece.SentencePieceProcessor, symbol: str) -> bool:
    pieces = processor.encode(symbol, return_type=str)
    return pieces == [symbol] or pieces == [_DUMMY_PREFIX, symbol]


def _has_byte_fallback(processor: sentencepiece.SentencePieceProcessor) -> bool:
    return all(_is_byte_piece(processor, piece) for piece in _BYTE_PIECES)


def _is_byte_piece(processor: sentencepiece.SentencePieceProcessor, piece: str) -> bool:
    # A piece the model lacks maps to the unknown id.
    piece_id = processor.piece_to_id(piece)
    return piece_id != processor.unk_id() and processor.is_byte(piece_id)


def _measure_coverage(
    processor: sentencepiece.SentencePieceProcessor,
    lang: str,
    path: Path,
    sample_lines: int,
) -> LanguageCoverage:
    lines = read_lines(path, limit=sample_lines)
    encoded = processor.encode(lines, return_type=int)
    piece_ids = [piece_id for ids in encoded for piece_id in ids]
    fallback_pieces = sum(
        1
        for piece_id in piece_ids
        if processor.is_unknown(piece_id) or processor.is_byte(piece_id)
    )
    return LanguageCoverage(
        lang=lang, pieces=len(piece_ids), fallback_pieces=fallback_pieces
    )
