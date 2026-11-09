"""Script masking: the scripts a target language writes and must not write, how pure a text is
in them, and the entries of `shared_vocabulary.json` CTranslate2 is told never to produce
(`suppress_sequences`, one single-token entry each).

A model of ours can emit a piece of the wrong script (a Latin piece in Cyrillic output; the
D5 converter, which wants to write traditional script, in a Cyrillic one). The forbidden
scripts of a target are every script of `rinchen.lang.scripts.Script` the registry does not
give to it; a letter of a script outside that enum (Greek) is in nobody's forbidden set, and
counts against a text's purity all the same.

`suppression_list` is the list of one (target, `allow`) pair; a caller builds it once and
passes it to every call of a decode. It never holds the end token (a model that cannot end
never stops) and holds only entries of the vocabulary it was given (CTranslate2 refuses any
other). The cost is the conversion of the list on every call, not a cost per decoding step.
Whether a hard mask helps a real model is unmeasured: the toy leaks only for requests it was
never trained for.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from rinchen.config.languages import load_languages
from rinchen.errors import ConfigError
from rinchen.infer.source_tokens import PieceLookup, reserved_pieces
from rinchen.lang.scripts import Script, counted_letters, script_of
from rinchen.spm.symbols import user_defined_symbols

# A prior, never calibrated: below it a decode that was not masked is decoded again with the
# list of its target.
PURITY_FLOOR: Final = 0.98

_DUMMY_PREFIX: Final = chr(0x2581)
_BYTE_PIECE: Final = re.compile(r"<0x([0-9A-F]{2})>")


class ProcessorLike(PieceLookup, Protocol):
    """What the suppression list asks of a SentencePiece processor (`SentencePieceProcessor`
    has it): the piece types on top of the lookups of the reserved-piece rule."""

    def is_control(self, piece_id: int) -> bool: ...

    def is_unknown(self, piece_id: int) -> bool: ...

    def is_byte(self, piece_id: int) -> bool: ...


def script_hint(text: str) -> Literal["mn_Mong", "und"]:
    """`mn_Mong` when `text` holds a letter of the Mongolian script, else `und`: the only
    distinction the normalisation of a text makes by language, and all the script says
    about a source."""
    return "mn_Mong" if Script.MONGOLIAN in script_of(text) else "und"


@functools.cache
def _registry_scripts() -> dict[str, frozenset[Script]]:
    return {entry.code: frozenset(entry.scripts) for entry in load_languages()}


def allowed_scripts(tgt: str) -> frozenset[Script]:
    """The scripts the registry gives to the language `tgt`.

    Raises:
        ConfigError: `tgt` is not in the language registry.
    """
    try:
        return _registry_scripts()[tgt]
    except KeyError:
        raise ConfigError(
            f"language {tgt!r} is not in the language registry: no scripts to allow"
        ) from None


def forbidden_scripts(tgt: str) -> frozenset[Script]:
    """Every script of `Script` the registry does not give to the language `tgt`.

    Raises:
        ConfigError: `tgt` is not in the language registry.
    """
    return frozenset(Script) - allowed_scripts(tgt)


def script_purity(text: str, scripts: Collection[Script]) -> float:
    """The share of the letters of `text` that are in `scripts`, counted as
    `rinchen.lang.scripts.script_of` counts them: digits, punctuation, marks, whitespace and
    the letters of no script are ignored, a letter of a script outside `Script` counts
    against the text. 1.0 for a text without a counted letter (nothing is out of script)."""
    if counted_letters(text) == 0:
        return 1.0
    shares = script_of(text)
    return sum(shares.get(script, 0.0) for script in set(scripts))


def suppression_list(
    vocabulary: Sequence[str],
    processor: ProcessorLike,
    tgt: str,
    *,
    end_token: str,
    allow: Collection[str] = (),
    scripts: bool = True,
) -> list[list[str]]:
    """The entries of `vocabulary` a decode into the language `tgt` must never produce, as
    CTranslate2's `suppress_sequences` wants them: a list of one-token lists, in the order of
    `vocabulary`, each entry once, never the `end_token`.

    The control entries are always listed: the unknown piece, the control pieces of the model
    (`<s>`) and every entry that is no piece of the model (`<blank>`, the padding Eole adds):
    none is ever text. With `scripts` (the default) the list also holds

    - the reserved symbols the model must not write (direction tags, structural symbols; not
      the placeholders and not those in `allow`),
    - every ordinary piece with a letter of a script forbidden for `tgt` (the dummy prefix
      U+2581 is read as a space), and
    - when Latin is forbidden, the byte-fallback pieces of the ASCII letters. A byte piece of
      any other byte stays: alone it says no script.

    Raises:
        ConfigError: `tgt` is not in the language registry.
        ValueError: `allow` names something that is no reserved symbol.
    """
    rules = _ScriptRules(
        processor=processor,
        forbidden=forbidden_scripts(tgt),
        reserved=reserved_pieces(processor, end_token, vocabulary, allow=allow),
        symbols=frozenset(user_defined_symbols()),
    )
    listed: dict[str, None] = {}
    for entry in vocabulary:
        if entry == end_token:
            continue
        piece_id = processor.piece_to_id(entry)
        if rules.is_control(piece_id) or (
            scripts and rules.is_unwritable(entry, piece_id)
        ):
            listed[entry] = None
    return [[entry] for entry in listed]


@dataclass(frozen=True)
class _ScriptRules:
    """What one (target, `allow`) pair decides about the entries of a vocabulary."""

    processor: ProcessorLike
    forbidden: frozenset[Script]
    reserved: frozenset[str]
    symbols: frozenset[str]

    def is_control(self, piece_id: int) -> bool:
        return self.processor.is_unknown(piece_id) or self.processor.is_control(
            piece_id
        )

    def is_unwritable(self, entry: str, piece_id: int) -> bool:
        if entry in self.reserved:
            return True
        if entry in self.symbols:
            return False
        if self.processor.is_byte(piece_id):
            return Script.LATIN in self.forbidden and _is_ascii_letter_byte(entry)
        letters = script_of(entry.replace(_DUMMY_PREFIX, " "))
        return any(script in self.forbidden for script in letters)


def _is_ascii_letter_byte(piece: str) -> bool:
    match = _BYTE_PIECE.fullmatch(piece)
    if match is None:
        return False
    character = chr(int(match.group(1), 16))
    return character.isascii() and character.isalpha()
