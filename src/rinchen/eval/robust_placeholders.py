"""The placeholder metric of D12: did a hypothesis keep the numbered placeholders of its
source?

A placeholder is the glyph pair of `spm.symbols` around one or two digits without a leading
zero, `0` to `99`: exactly the hundred reserved symbols, so a pair around `03`, around `100`
or around a digit with spaces is no placeholder and its glyphs are damage. A hypothesis is
scored as the engine wrote it: nothing is repaired, because a policy that appended a lost
placeholder would make every row pass and measure nothing. The order of the placeholders is a
detail that is kept beside the verdict and never part of it: it is legitimately free across
languages."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols


@dataclass(frozen=True, slots=True)
class PlaceholderOutcome:
    """How a hypothesis kept the placeholders of a source: those the source holds (a
    repeated index counts each time), those of them the hypothesis holds as often
    (`found`), the rest (`missing`), the hypothesis's placeholders the source does not
    hold (`extra`) and the broken glyphs (`damaged`). `exact` is no missing, no extra and
    no damage; `order_ok` says, for an exact row only, whether the hypothesis writes them
    in the reference's order."""

    expected: int
    found: int
    missing: int
    extra: int
    damaged: int
    exact: bool
    order_ok: bool | None


def placeholders_in(text: str) -> tuple[int, ...]:
    """The indices of the well-formed placeholders of `text`, in written order."""
    index_of = _index_by_symbol()
    return tuple(index_of[match.group()] for match in _placeholders().finditer(text))


def damaged_brackets(text: str) -> int:
    """How many glyphs of the pair are not part of a well-formed placeholder."""
    glyphs = len(_glyphs().findall(text))
    return glyphs - 2 * len(_placeholders().findall(text))


def score_placeholders(
    source: str, reference: str, hypothesis: str
) -> PlaceholderOutcome:
    """Compare the placeholders of `hypothesis` with those of `source` as multisets. The
    reference only gives the order of an exact row."""
    wanted = Counter(placeholders_in(source))
    written = placeholders_in(hypothesis)
    seen = Counter(written)
    found = (wanted & seen).total()
    missing = wanted.total() - found
    extra = seen.total() - found
    damaged = damaged_brackets(hypothesis)
    exact = missing == extra == damaged == 0
    return PlaceholderOutcome(
        expected=wanted.total(),
        found=found,
        missing=missing,
        extra=extra,
        damaged=damaged,
        exact=exact,
        order_ok=written == placeholders_in(reference) if exact else None,
    )


def placeholders_agree(source: str, reference: str) -> bool:
    """Whether the source holds a placeholder and the reference holds the same ones, as
    often."""
    wanted = Counter(placeholders_in(source))
    return bool(wanted) and wanted == Counter(placeholders_in(reference))


@cache
def _index_by_symbol() -> dict[str, int]:
    symbols = user_defined_symbols()[-PLACEHOLDER_COUNT:]
    return {symbol: index for index, symbol in enumerate(symbols)}


@cache
def _placeholders() -> re.Pattern[str]:
    """The hundred reserved placeholders and nothing that merely looks like one."""
    return re.compile("|".join(re.escape(symbol) for symbol in _index_by_symbol()))


@cache
def _glyphs() -> re.Pattern[str]:
    first = user_defined_symbols()[-PLACEHOLDER_COUNT]
    return re.compile(re.escape(first[0]) + "|" + re.escape(first[-1]))
