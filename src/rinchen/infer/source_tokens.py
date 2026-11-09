"""The rules CTranslate2 does not have, shared by the teacher of D10 and the engine of D11:
where the direction tag goes, and the pieces no translation of ours holds.

A translation never holds the unknown piece of the SentencePiece model (decoded, it reads as
the text of U+2047 between spaces), the end token inside the text, a reserved symbol of the
project (a direction tag, `<bt>`, `<doc>`; decoded, each reads as its own letters) or an entry
of `shared_vocabulary.json` that is no piece of the model (Eole's `<blank>` and the padding it
adds to the vocabulary: decoded, they read as their letters). The numbered placeholders are
reserved symbols too, and the one exception: the engine is meant to write them (D12), so
`reserved_pieces` leaves them out and the teacher of D10, which never writes one, adds them
back. The pieces of a byte-fallback model (`<0x41>`) are ordinary text.

Nothing here imports SentencePiece or CTranslate2: a processor is passed in.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Protocol

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols


class PieceLookup(Protocol):
    """What the reserved-piece rule asks of a SentencePiece processor (`SentencePieceProcessor`
    has it)."""

    def piece_to_id(self, piece: str) -> int: ...

    def unk_id(self) -> int: ...

    def id_to_piece(self, piece_id: int) -> str: ...


def ct2_source_tokens(
    placement: TagPlacement, tag: str, pieces: Sequence[str]
) -> tuple[list[str], list[str] | None]:
    """The source line and the forced target prefix CTranslate2 is given for a text cut into
    `pieces` and asked for the language `tag` stands for: for placement `src` the tag in
    front of the pieces and no prefix, for `tgt` the pieces alone and the prefix `[tag]`.

    Raises:
        ValueError: `placement` is neither `src` nor `tgt`.
    """
    require_placement(placement)
    if placement == "src":
        return [tag, *pieces], None
    return list(pieces), [tag]


def reserved_pieces(
    processor: PieceLookup,
    end_token: str,
    vocabulary: Iterable[str],
    *,
    allow: Collection[str] = (),
) -> frozenset[str]:
    """The pieces a translation never holds: the unknown piece of `processor`, the
    `end_token` (a hypothesis ends with it once), every reserved symbol of the project
    (every SentencePiece model of ours keeps them as single pieces) except the numbered
    placeholders and the symbols in `allow`, and every entry of the export's `vocabulary`
    that is no piece of `processor`: Eole adds its pad token and pads the vocabulary with
    made-up tokens, which the model could emit and `processor.decode` would write as their
    letters.

    `allow` lets a request write a reserved symbol: a document window is cut at `<sep>`.
    A vocabulary entry that is no piece of the model stays reserved whatever `allow` says.

    Raises:
        TypeError: `allow` is a bare string, which would be read as a set of letters.
        ValueError: `allow` names something that is no reserved symbol (a misspelt symbol
            would otherwise be ignored without a sign).
    """
    symbols = user_defined_symbols()
    allowed = _checked_allowance(allow, symbols)
    placeholders = frozenset(symbols[-PLACEHOLDER_COUNT:])
    unknown_id = processor.unk_id()
    foreign = (e for e in vocabulary if processor.piece_to_id(e) == unknown_id)
    unknown = processor.id_to_piece(unknown_id)
    written_by_the_model = placeholders | allowed
    reserved_symbols = (s for s in symbols if s not in written_by_the_model)
    return frozenset({unknown, end_token, *reserved_symbols, *foreign})


def _checked_allowance(
    allow: Collection[str], symbols: Sequence[str]
) -> frozenset[str]:
    if isinstance(allow, str):
        raise TypeError(f"allow is a collection of symbols, not the string {allow!r}")
    unknown = sorted(set(allow) - set(symbols))
    if unknown:
        raise ValueError(f"allow names pieces that are no reserved symbol: {unknown}")
    return frozenset(allow)
