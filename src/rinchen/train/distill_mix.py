"""How a student's training data is mixed from the original corpora and the KD corpora (D10):
the mix `K:O`, the integer weights that realise it exactly, and how much more often a KD row
is drawn than an original one.

Eole's corpus `weight` is the number of consecutive examples taken from that corpus per round
of a cyclic round robin (`sampling`), so the share of a corpus in the draws is its weight over
the sum of the weights, whatever its size. The mix `K:O` (reduced by the greatest common
divisor) asks that in every direction with a KD corpus exactly K of every K + O examples come
from it. The original corpora keep D7's weights (`sampling.integer_weights`), which carry the
temperature over the directions; then, with `S` the sum of the original weights of a
direction,

- every original corpus weighs O times what D7 gave it, so the originals of the direction
  weigh `O x S`, and
- its KD corpus weighs `K x S`.

So the KD corpus against the originals of its direction is exactly `K : O`, whatever `S`.
Every direction then weighs `(K + O) x S`, which makes the share of each direction of the
draws the temperature's, unchanged: a direction that has no KD corpus gets its originals
scaled by `K + O` instead, to keep that share. The sum of the weights is `(K + O) x` D7's.
All of it is integer arithmetic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from rinchen.errors import ConfigError
from rinchen.utils.integers import is_strict_int

# The largest number either side of a mix text may be: a bound on what a typo can ask for.
MAX_MIX_TERM: Final = 10_000
_MIX_TERM: Final = r"[1-9][0-9]{0,4}"
_MIX_TEXT: Final = re.compile(rf"({_MIX_TERM}):({_MIX_TERM})")


@dataclass(frozen=True, slots=True)
class Mix:
    """The share of KD examples in the draws of every direction: `kd` of every
    `kd + original`. Both are positive integers; `reduced` divides them by their greatest
    common divisor, which does not change the share.

    Raises:
        ValueError: either is not a positive integer.
    """

    kd: int
    original: int

    def __post_init__(self) -> None:
        for name, value in (("kd", self.kd), ("original", self.original)):
            if not is_strict_int(value) or value < 1:
                raise ValueError(
                    f"the {name} part of a mix must be a positive integer, got {value!r}"
                )

    def reduced(self) -> Mix:
        divisor = math.gcd(self.kd, self.original)
        return Mix(self.kd // divisor, self.original // divisor)


def parse_mix(text: str) -> Mix:
    """The mix `text` writes as `K:O`: two positive decimal integers of at most 10,000
    each, in ASCII digits with no sign, zero in front or space, reduced by their greatest
    common divisor (`2:2` is `1:1`, `6:4` is `3:2`).

    Raises:
        ConfigError: `text` is anything else.
    """
    found = _MIX_TEXT.fullmatch(text)
    if found is None or max(int(part) for part in found.groups()) > MAX_MIX_TERM:
        raise ConfigError(
            f"--mix {text!r} is not K:O: two positive whole numbers of at most "
            f"{MAX_MIX_TERM}, written with digits only and no sign or space, such as 1:1 "
            "or 3:1"
        )
    kd, original = (int(part) for part in found.groups())
    return Mix(kd, original).reduced()


def mixed_weights(
    original: Mapping[str, int],
    direction_of: Mapping[str, str],
    kd_corpora: Mapping[str, str],
    mix: Mix,
) -> dict[str, int]:
    """The weights of all the corpora of a student's data, the original corpora first in
    the order given, then the KD corpora in the order given (see the module's text).

    `original` maps each original corpus to its D7 weight and `direction_of` each of them
    to its direction; `kd_corpora` maps each KD corpus to its direction (one KD corpus per
    direction, and its direction has original corpora).

    Raises:
        ValueError: a weight that is not a positive integer; `original` and `direction_of`
            that do not name the same corpora; a corpus that is both original and KD; a KD
            corpus whose direction has no original corpus; two KD corpora of one
            direction.
    """
    reduced = mix.reduced()
    _require_consistent(original, direction_of, kd_corpora)
    direction_sums: dict[str, int] = {}
    for name, weight in original.items():
        direction_sums[direction_of[name]] = (
            direction_sums.get(direction_of[name], 0) + weight
        )
    kd_directions = set(kd_corpora.values())
    scale_of = {
        direction: reduced.original
        if direction in kd_directions
        else reduced.kd + reduced.original
        for direction in direction_sums
    }
    weights = {
        name: scale_of[direction_of[name]] * weight for name, weight in original.items()
    }
    for name, direction in kd_corpora.items():
        weights[name] = reduced.kd * direction_sums[direction]
    return weights


def repeat_factors(
    rows: Mapping[str, int],
    weights: Mapping[str, int],
    direction_of: Mapping[str, str],
    kd_corpora: Mapping[str, str],
) -> dict[str, Fraction]:
    """For each KD corpus, how many times more often one of its rows is drawn than one
    row of the original corpora of its direction: `(KD weight / KD rows)` over
    `(sum of the original weights of the direction / their rows)`. `weights` are the
    weights Eole gets (`mixed_weights`), `rows` the row count of every corpus written.

    Raises:
        ValueError: a corpus of `direction_of` or `kd_corpora` without rows or without a
            positive weight; a KD corpus whose direction has no original corpus.
    """
    original_rows: dict[str, int] = {}
    original_weights: dict[str, int] = {}
    for name, direction in direction_of.items():
        original_rows[direction] = original_rows.get(direction, 0) + _positive(
            rows, name, "rows"
        )
        original_weights[direction] = original_weights.get(direction, 0) + _positive(
            weights, name, "weight"
        )
    factors: dict[str, Fraction] = {}
    for name, direction in kd_corpora.items():
        kd_rows = _positive(rows, name, "rows")
        kd_weight = _positive(weights, name, "weight")
        if direction not in original_rows:
            raise ValueError(
                f"the KD corpus {name!r} is of direction {direction!r}, which has no "
                "original corpus"
            )
        factors[name] = Fraction(kd_weight, kd_rows) / Fraction(
            original_weights[direction], original_rows[direction]
        )
    return factors


def _positive(counts: Mapping[str, int], name: str, what: str) -> int:
    value = counts.get(name)
    if not is_strict_int(value) or value < 1:
        raise ValueError(
            f"the {what} of corpus {name!r} must be a positive integer, got {value!r}"
        )
    return value


def _require_consistent(
    original: Mapping[str, int],
    direction_of: Mapping[str, str],
    kd_corpora: Mapping[str, str],
) -> None:
    for name, weight in original.items():
        if not is_strict_int(weight) or weight < 1:
            raise ValueError(
                f"the weight of the original corpus {name!r} must be a positive "
                f"integer, got {weight!r}"
            )
        if name not in direction_of:
            raise ValueError(f"the original corpus {name!r} has no direction")
    ghosts = sorted(set(direction_of) - set(original))
    if ghosts:
        raise ValueError(
            f"a direction is given for {ghosts[0]!r}, which is no original corpus"
        )
    both = sorted(set(original) & set(kd_corpora))
    if both:
        raise ValueError(
            f"corpus {both[0]!r} is both an original and a KD corpus; a name is one"
        )
    original_directions = set(direction_of.values())
    seen: dict[str, str] = {}
    for name, direction in kd_corpora.items():
        if direction not in original_directions:
            raise ValueError(
                f"the KD corpus {name!r} is of direction {direction!r}, which has no "
                "original corpus"
            )
        if direction in seen:
            raise ValueError(
                f"direction {direction!r} has two KD corpora, {seen[direction]!r} and "
                f"{name!r}: its KD share would be above that of the mix"
            )
        seen[direction] = name
