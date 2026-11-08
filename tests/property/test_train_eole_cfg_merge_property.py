"""Property tests for `rinchen.train.eole_cfg.merge` over random nested configs:
merging the same overrides twice changes nothing, neither input is modified, and a
key whose override is `None` is gone from the result at any depth."""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from rinchen.train.eole_cfg import merge

_KEYS = st.sampled_from(["a", "b", "c", "d"])
_SCALARS = st.one_of(
    st.integers(-3, 3), st.booleans(), st.text(alphabet="xyz", max_size=3)
)
_LEAVES = st.one_of(_SCALARS, st.lists(_SCALARS, max_size=3))
_MAPPINGS = st.recursive(
    st.dictionaries(_KEYS, _LEAVES, max_size=3),
    lambda children: st.dictionaries(_KEYS, st.one_of(_LEAVES, children), max_size=3),
    max_leaves=8,
)
# Overrides may also carry `None`, which deletes.
_OVERRIDE_LEAVES = st.one_of(_LEAVES, st.none())
_OVERRIDES = st.recursive(
    st.dictionaries(_KEYS, _OVERRIDE_LEAVES, max_size=3),
    lambda children: st.dictionaries(
        _KEYS, st.one_of(_OVERRIDE_LEAVES, children), max_size=3
    ),
    max_leaves=8,
)


def _none_paths(
    overrides: dict[str, Any], path: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    for key, value in overrides.items():
        if value is None:
            found.append((*path, key))
        elif isinstance(value, dict):
            found.extend(_none_paths(value, (*path, key)))
    return found


def _present(cfg: dict[str, Any], path: tuple[str, ...]) -> bool:
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


@given(base=_MAPPINGS, overrides=_OVERRIDES)
def test_merging_the_same_overrides_twice_changes_nothing(
    base: dict[str, Any], overrides: dict[str, Any]
) -> None:
    once = merge(base, overrides)

    assert merge(once, overrides) == once


@given(base=_MAPPINGS, overrides=_OVERRIDES)
def test_merge_modifies_neither_input(
    base: dict[str, Any], overrides: dict[str, Any]
) -> None:
    base_before = copy.deepcopy(base)
    overrides_before = copy.deepcopy(overrides)

    merged = merge(base, overrides)
    _scribble(merged)

    assert base == base_before
    assert overrides == overrides_before


def _scribble(node: Any) -> None:
    """Change every list and mapping of `node` in place, to show nothing is shared."""
    if isinstance(node, dict):
        for value in list(node.values()):
            _scribble(value)
        node["__scribble__"] = 1
    elif isinstance(node, list):
        for item in node:
            _scribble(item)
        node.append("__scribble__")


@given(base=_MAPPINGS, overrides=_OVERRIDES)
def test_a_none_override_leaves_no_key_behind_at_any_depth(
    base: dict[str, Any], overrides: dict[str, Any]
) -> None:
    merged = merge(base, overrides)

    # A later override of the same path by a non-None value cannot exist: paths are
    # unique within one mapping, so every None path names a key that must be gone.
    assert [path for path in _none_paths(overrides) if _present(merged, path)] == []


@given(base=_MAPPINGS)
def test_merging_no_overrides_gives_an_equal_copy(base: dict[str, Any]) -> None:
    merged = merge(base, {})

    assert merged == base
    assert merged is not base


def _holds_none(node: Any) -> bool:
    if isinstance(node, dict):
        return any(value is None or _holds_none(value) for value in node.values())
    return False


@given(overrides=_OVERRIDES)
def test_merging_into_an_empty_base_keeps_no_none_the_overrides_brought(
    overrides: dict[str, Any],
) -> None:
    assert not _holds_none(merge({}, overrides))
