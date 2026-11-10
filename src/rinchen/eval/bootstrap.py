"""The bootstrap of D13 (T9): which units a resample draws, and the sums of their statistics.

A corpus score is a function of INTEGER statistics added up (`rinchen.metrics.segment_stats`), so
the bootstrap is integer arithmetic from end to end: it draws units, adds up their rows and hands
the sums to the score. Nothing in it depends on a float, on numpy's random generator (whose
stream numpy does not promise across versions), on the environment or on the hash seed.

The stream of draws is counter-based. The index of position `j` of resample `b` over `n` units is

    ((mix(seed + (b * n + j + 1) * 0x9E3779B97F4A7C15) >> 32) * n) >> 32

with every operation modulo 2^64 and `mix` the finaliser of splitmix64. It reproduces the
published splitmix64 outputs (for seed 0 the high 32 bits of the first three words are the first
three indices over 2^32 units), a resample's draws do not depend on the chunk it is computed in,
and the bias of the multiplication is at most `n / 2^32`. `draw_index` is the rule in Python
integers; `resample_indices` is the same rule in uint64 arithmetic with no loop.

The interval of a score is the percentile interval `[s[k], s[B - 1 - k]]` of the sorted scores
of `B` resamples, with `k = floor(B * alpha / 2 + 1e-9)` and no interpolation: the positions 25
and 974 for the defaults. The seed is fixed and recorded: it moves an end by about 0.02.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rinchen.errors import ConfigError

RNG_NAME: Final = "splitmix64-counter/1"
BOOTSTRAP_SEED: Final = 20_261_110
DEFAULT_RESAMPLES: Final = 1000
DEFAULT_ALPHA: Final = 0.05
MIN_RESAMPLES: Final = 100

# The draws of a chunk of resamples are held as a (resamples, units) array: at most this many
# cells, so that memory is bounded whatever the number of resamples.
CHUNK_CELLS: Final = 1 << 20
MAX_UNITS: Final = 2**32

_MASK: Final = (1 << 64) - 1
_INT64_MAX: Final = 2**63 - 1
_GOLDEN: Final = 0x9E3779B97F4A7C15
_MIX_FIRST: Final = 0xBF58476D1CE4E5B9
_MIX_SECOND: Final = 0x94D049BB133111EB


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """How many resamples, from which seed, at which level (`alpha` 0.05 is a 95 % interval)."""

    resamples: int = DEFAULT_RESAMPLES
    seed: int = BOOTSTRAP_SEED
    alpha: float = DEFAULT_ALPHA

    def __post_init__(self) -> None:
        if not _is_int(self.resamples) or self.resamples < MIN_RESAMPLES:
            raise ConfigError(
                f"resamples must be an integer of at least {MIN_RESAMPLES}"
            )
        if not _is_int(self.seed) or not 0 <= self.seed <= _MASK:
            raise ConfigError("seed must be an integer from 0 to 2^64 - 1")
        if not isinstance(self.alpha, float) or not 0 < self.alpha < 0.5:
            raise ConfigError("alpha must be a number above 0 and below 0.5")


def _is_int(number: object) -> bool:
    return isinstance(number, int) and not isinstance(number, bool)


def _require_stream(seed: int, units: int) -> None:
    if not 0 <= seed <= _MASK:
        raise ValueError("the seed must be an integer from 0 to 2^64 - 1")
    if not 1 <= units <= MAX_UNITS:
        raise ValueError("the number of units must be from 1 to 2^32")


def _mix(word: int) -> int:
    word &= _MASK
    word = ((word ^ (word >> 30)) * _MIX_FIRST) & _MASK
    word = ((word ^ (word >> 27)) * _MIX_SECOND) & _MASK
    return word ^ (word >> 31)


def draw_index(seed: int, resample: int, position: int, units: int) -> int:
    """The unit that position `position` of resample `resample` draws, among `units`.

    Raises:
        ValueError: the seed is not in [0, 2^64), `units` is not in [1, 2^32], `resample` is
            negative or `position` is not in [0, units).
    """
    _require_stream(seed, units)
    if resample < 0 or not 0 <= position < units:
        raise ValueError(
            "a draw is the position of a resample: from 0, below the units"
        )
    word = _mix(seed + (resample * units + position + 1) * _GOLDEN)
    return ((word >> 32) * units) >> 32


def resample_indices(seed: int, first: int, last: int, units: int) -> NDArray[np.int64]:
    """The draws of resamples `first` to `last - 1`, as a `(last - first, units)` array: cell for
    cell what `draw_index` gives, in uint64 arithmetic that wraps modulo 2^64, with no loop.

    Raises:
        ValueError: as `draw_index` for the seed and the units, and `first` negative or above
            `last`.
    """
    _require_stream(seed, units)
    if first < 0 or last < first:
        raise ValueError("resamples are a range from 0 upwards")
    resample = np.arange(first, last, dtype=np.uint64)[:, None]
    position = np.arange(units, dtype=np.uint64)[None, :]
    word = np.uint64(seed) + (resample * np.uint64(units) + position + np.uint64(1)) * (
        np.uint64(_GOLDEN)
    )
    word = (word ^ (word >> np.uint64(30))) * np.uint64(_MIX_FIRST)
    word = (word ^ (word >> np.uint64(27))) * np.uint64(_MIX_SECOND)
    word = word ^ (word >> np.uint64(31))
    drawn = ((word >> np.uint64(32)) * np.uint64(units)) >> np.uint64(32)
    return drawn.astype(np.int64)


def require_unit_rows(rows: object) -> NDArray[np.int64]:
    """`rows` as the `(units, columns)` int64 array the bootstrap works on.

    Raises:
        ValueError: `rows` is not a two-dimensional int64 array with a column, it has no unit,
            or the sum of a resample could overflow int64.
    """
    if (
        not isinstance(rows, np.ndarray)
        or rows.dtype != np.int64
        or rows.ndim != 2
        or rows.shape[1] < 1
    ):
        raise ValueError(
            "rows must be a two-dimensional int64 array with at least one column"
        )
    if rows.shape[0] < 1:
        raise ValueError("at least one unit is needed")
    _require_no_overflow(rows, rows.shape[0])
    return rows


def _require_no_overflow(rows: NDArray[np.int64], added: int) -> None:
    largest = max(int(rows.max()), -int(rows.min()))
    if largest * added > _INT64_MAX:
        raise ValueError("a sum of these rows could overflow int64")


def resample_totals(
    rows: NDArray[np.int64], settings: BootstrapSettings
) -> NDArray[np.int64]:
    """The sums of the drawn rows of every resample: a `(resamples, columns)` int64 array whose
    row `b` adds up the rows of the units that resample `b` drew, with repetition.

    The resamples are computed in chunks of at most `CHUNK_CELLS` draws; a resample's draws do
    not depend on the chunk it is in.

    Raises:
        ValueError: see `require_unit_rows`.
    """
    matrix = require_unit_rows(rows)
    units = matrix.shape[0]
    totals = np.empty((settings.resamples, matrix.shape[1]), dtype=np.int64)
    per_chunk = max(1, CHUNK_CELLS // units)
    for first in range(0, settings.resamples, per_chunk):
        last = min(first + per_chunk, settings.resamples)
        drawn = resample_indices(settings.seed, first, last, units)
        offsets = np.arange(last - first, dtype=np.int64)[:, None] * units
        counts = np.bincount(
            (drawn + offsets).ravel(), minlength=(last - first) * units
        ).reshape(last - first, units)
        totals[first:last] = counts @ matrix
    return totals


def group_rows(
    rows: Sequence[Sequence[int]], groups: Sequence[str] | None
) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    """The rows added up per group, as the units of a bootstrap, and the groups in the order of
    their FIRST APPEARANCE (never sorted). Without groups every row is a unit, labelled `"0"`,
    `"1"` and so on by position.

    Raises:
        ValueError: no row, rows of different or no width, a cell that is not an integer of
            int64, a number of groups other than the number of rows, or a sum that could
            overflow int64.
    """
    if not rows:
        raise ValueError("at least one row is needed")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("rows must all have the same number of columns, at least one")
    if groups is not None and len(groups) != len(rows):
        raise ValueError("there must be one group for each row")
    matrix = np.array(rows)
    if matrix.dtype != np.int64:
        raise ValueError("every cell of the rows must be an integer that fits int64")
    _require_no_overflow(matrix, len(rows))
    if groups is None:
        return matrix, tuple(str(position) for position in range(len(rows)))
    numbers: dict[str, int] = {}
    codes = np.array(
        [numbers.setdefault(group, len(numbers)) for group in groups], dtype=np.int64
    )
    sums = np.zeros((len(numbers), width), dtype=np.int64)
    np.add.at(sums, codes, matrix)
    return sums, tuple(numbers)


def interval_positions(resamples: int, alpha: float) -> tuple[int, int]:
    """The positions of the interval's ends among `resamples` sorted scores:
    `(k, resamples - 1 - k)` with `k = floor(resamples * alpha / 2 + 1e-9)`.

    Raises:
        ValueError: no resample, or an `alpha` outside (0, 0.5).
    """
    if resamples < 1:
        raise ValueError("an interval needs at least one resample")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be a number above 0 and below 0.5")
    k = math.floor(resamples * alpha / 2 + 1e-9)
    return k, resamples - 1 - k


def interval_bounds(values: Iterable[float], alpha: float) -> tuple[float, float]:
    """The ends of the percentile interval of `values`: the sorted values at
    `interval_positions`, as Python floats.

    Raises:
        ValueError: no value, a value that is not a number, or an `alpha` outside (0, 0.5).
    """
    scores = [float(value) for value in values]
    if any(math.isnan(score) for score in scores):
        raise ValueError("a score is not a number")
    low, high = interval_positions(len(scores), alpha)
    scores.sort()
    return scores[low], scores[high]
