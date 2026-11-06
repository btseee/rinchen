"""Temperature-smoothed sampling weights for multi-corpus mixing."""

from __future__ import annotations

from collections.abc import Sequence


def temperature_weights(sizes: Sequence[int], temperature: float) -> list[float]:
    """Return per-corpus sampling weights ``size**T / sum(size**T)``.

    A temperature below 1 lifts small corpora above their proportional share
    (XLM-R style exponential smoothing); 1 is proportional; above 1 widens the gap.
    A zero-size corpus always gets weight 0.0.

    Raises:
        ValueError: if ``temperature`` is not positive, any size is negative, or
            every size is zero.
    """
    # Checked before exponentiation: Python evaluates 0 ** 0.0 as 1.0, which would
    # hand a zero-size corpus a nonzero weight.
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if any(size < 0 for size in sizes):
        raise ValueError(f"sizes must be non-negative, got {list(sizes)}")
    if not sizes:
        return []
    if not any(sizes):
        raise ValueError("at least one size must be non-zero")

    smoothed = [size**temperature for size in sizes]
    total = sum(smoothed)
    return [value / total for value in smoothed]
