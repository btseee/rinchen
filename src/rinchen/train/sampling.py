"""How often each training corpus is drawn: temperature-smoothed probabilities and
the whole-number weights Eole's data mixer takes.

Eole's corpus `weight` is the number of consecutive examples taken from that corpus
per round of a cyclic round robin, so the share of a corpus is its weight over the
sum of the weights, whatever its size. Temperature sampling therefore needs integer
weights; `integer_weights` turns the probabilities into them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rinchen.errors import ConfigError
from rinchen.utils.sampling import temperature_weights

# Probabilities arrive from float arithmetic: their sum is 1 only to rounding.
_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CorpusSize:
    """A training corpus: its unique name, the translation direction it belongs to
    (any text; corpora with equal text share a direction) and its row count."""

    name: str
    direction: str
    rows: int


def mixing_probabilities(
    corpora: Sequence[CorpusSize], temperature: float
) -> dict[str, float]:
    """The probability of drawing each corpus, keyed by name in the order given.

    The temperature smooths the DIRECTIONS (`utils.sampling.temperature_weights` over
    the row total of each): below 1 a small direction gets more than its share of the
    rows, 1 is proportional. Inside a direction its probability is shared in
    proportion to the rows of its corpora, so the ratio between two corpora of one
    direction (authentic and back-translated rows, say) does not depend on the
    temperature.

    The probabilities are positive and sum to 1, and no one of them depends on the
    order of `corpora`.

    Raises:
        ValueError: no corpus; two corpora with one name; a corpus without rows; a
            temperature that is not a positive finite number or that is so large
            that the sizes raised to it do not fit a float.
    """
    if not corpora:
        raise ValueError("at least one corpus is needed")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"temperature must be a positive finite number, got {temperature}"
        )
    _check_names_and_rows(corpora)
    rows_by_direction: dict[str, int] = {}
    for corpus in corpora:
        rows_by_direction[corpus.direction] = (
            rows_by_direction.get(corpus.direction, 0) + corpus.rows
        )
    direction_probability = _direction_probabilities(rows_by_direction, temperature)
    probabilities = {
        corpus.name: direction_probability[corpus.direction]
        * (corpus.rows / rows_by_direction[corpus.direction])
        for corpus in corpora
    }
    if not all(value > 0 for value in probabilities.values()):
        raise _temperature_too_large(temperature)
    return probabilities


def integer_weights(
    probabilities: Mapping[str, float], *, total: int = 100
) -> dict[str, int]:
    """Whole-number corpus weights close to `probabilities`, keyed by name in the
    order given.

    Every weight is at least 1 (Eole takes a corpus of weight 0 never), and they sum
    to `total` whenever there are at most `total` corpora. The rule: floor each
    `total * probability`, with a minimum of 1; if that falls short of `total`, give
    one more to the corpora with the largest remainders (largest remainder rounding);
    if the minimum made it overshoot, take one at a time from the largest weight, never
    from a weight of 1. A larger probability never gets a smaller weight, and ties
    (equal remainders, equal weights) go by name, never by the order of `probabilities`:
    an extra unit goes to the earlier name, a unit taken comes from the later one.

    Raises:
        ValueError: no probability; a probability that is not positive and finite;
            probabilities whose sum is not 1 within 1e-9.
        ConfigError: more corpora than `total`; the message names the number needed.
    """
    _check_distribution(probabilities)
    if total < len(probabilities):
        raise ConfigError(
            f"weight_total {total} is below the number of corpora "
            f"({len(probabilities)}): every corpus needs a weight of at least 1, so "
            f"weight_total must be at least {len(probabilities)}"
        )
    ideal = {name: total * value for name, value in probabilities.items()}
    weights = {name: max(1, math.floor(value)) for name, value in ideal.items()}
    missing = total - sum(weights.values())
    if missing > 0:
        _give_to_largest_remainders(weights, ideal, missing)
    elif missing < 0:
        _take_from_largest_weights(weights, probabilities, -missing)
    return {name: weights[name] for name in probabilities}


def _check_names_and_rows(corpora: Sequence[CorpusSize]) -> None:
    seen: set[str] = set()
    for corpus in corpora:
        if corpus.name in seen:
            raise ValueError(f"corpus name {corpus.name!r} appears twice")
        seen.add(corpus.name)
        if corpus.rows < 1:
            raise ValueError(
                f"corpus {corpus.name!r} has {corpus.rows} rows: a corpus without "
                "rows has no probability and must be left out"
            )


def _direction_probabilities(
    rows_by_direction: Mapping[str, int], temperature: float
) -> dict[str, float]:
    directions = list(rows_by_direction)
    try:
        shares = temperature_weights(
            [rows_by_direction[direction] for direction in directions], temperature
        )
    except OverflowError:
        raise _temperature_too_large(temperature) from None
    return dict(zip(directions, shares, strict=True))


def _temperature_too_large(temperature: float) -> ValueError:
    return ValueError(
        f"temperature {temperature} is too large for these corpus sizes: the sizes "
        "raised to it, or the sum of those powers, do not fit a float"
    )


def _check_distribution(probabilities: Mapping[str, float]) -> None:
    if not probabilities:
        raise ValueError("at least one probability is needed")
    for name, value in probabilities.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"the probability of {name!r} must be positive and finite, got {value}"
            )
    total = math.fsum(probabilities.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(
            f"the probabilities must sum to 1 (within {_SUM_TOLERANCE}), got {total}"
        )


def _give_to_largest_remainders(
    weights: dict[str, int], ideal: Mapping[str, float], count: int
) -> None:
    """Add 1 to `count` weights: those furthest below their ideal value, the earlier
    name first among equals. Floors leave less than one unit per corpus to give, so
    `count` never exceeds the number of weights."""
    by_remainder = sorted(weights, key=lambda key: (weights[key] - ideal[key], key))
    for name in by_remainder[:count]:
        weights[name] += 1


def _take_from_largest_weights(
    weights: dict[str, int], probabilities: Mapping[str, float], count: int
) -> None:
    """Take 1 from a weight `count` times: the largest, and of equal weights the one
    of the smaller probability (so no larger probability is left with less), of
    those the later name. The weights sum to more than the number of corpora here, so
    a weight of 2 or more always exists."""
    for _ in range(count):
        largest = max(weights, key=lambda key: (weights[key], -probabilities[key], key))
        weights[largest] -= 1
