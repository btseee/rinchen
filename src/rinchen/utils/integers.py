"""Whether a value is an integer in the sense of a config file.

A boolean is an `int` to Python (`isinstance(True, int)`), and a count, a rank, a port or a
step to a config reader. YAML and JSON write `true` and `1` differently, so the config
checks of the launcher and of the Eole config reject the first where the second is meant.
"""

from __future__ import annotations

from typing import TypeGuard


def is_strict_int(value: object) -> TypeGuard[int]:
    """True for an `int` (a subclass of it included) that is not a `bool`."""
    return isinstance(value, int) and not isinstance(value, bool)
