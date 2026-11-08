"""Property tests for `rinchen.train.checkpoints`: the identity of a run over generated
configs, and which directory names count as step directories."""

from __future__ import annotations

import copy
import random
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.train.checkpoints import identity_of, step_directories

_KEYS = st.text(alphabet="abcdefgh_", min_size=1, max_size=4)
_LEAVES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-5, 5),
    st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    st.text(max_size=6),
)
_SECTIONS = st.recursive(
    _LEAVES,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(_KEYS, inner, max_size=3)
    ),
    max_leaves=8,
)
# A config as `identity_of` takes it: sections and a `training` mapping of its own.
_CONFIGS = st.fixed_dictionaries(
    {"training": st.dictionaries(_KEYS, _SECTIONS, max_size=4)},
    optional={
        "seed": st.integers(),
        "data": st.dictionaries(_KEYS, _SECTIONS, max_size=3),
    },
)


def _reordered(node: Any, rng: random.Random) -> Any:
    if isinstance(node, dict):
        items = list(node.items())
        rng.shuffle(items)
        return {key: _reordered(value, rng) for key, value in items}
    if isinstance(node, list):
        return [_reordered(value, rng) for value in node]
    return node


@given(_CONFIGS, st.integers())
def test_the_identity_is_the_same_for_any_order_of_the_keys(
    cfg: dict[str, Any], seed: int
) -> None:
    assert identity_of(_reordered(cfg, random.Random(seed))) == identity_of(cfg)


@given(_CONFIGS, st.integers(), st.one_of(st.none(), st.text(max_size=8)))
def test_the_identity_ignores_the_two_keys_a_run_may_change(
    cfg: dict[str, Any], steps: int, train_from: str | None
) -> None:
    changed = copy.deepcopy(cfg)
    changed["training"]["train_steps"] = steps
    changed["training"]["train_from"] = train_from
    plain = copy.deepcopy(cfg)
    plain["training"].pop("train_steps", None)
    plain["training"].pop("train_from", None)

    assert identity_of(changed) == identity_of(plain) == identity_of(cfg)


@given(_CONFIGS)
def test_the_identity_does_not_modify_its_input(cfg: dict[str, Any]) -> None:
    before = copy.deepcopy(cfg)

    identity_of(cfg)

    assert cfg == before


@given(_CONFIGS, _KEYS, _SECTIONS)
def test_adding_or_changing_any_other_top_level_key_changes_the_identity(
    cfg: dict[str, Any], key: str, value: Any
) -> None:
    changed = copy.deepcopy(cfg)
    changed[key] = value if cfg.get(key) != value else [value]

    if key == "training":
        return  # the training section has its own property below
    assert identity_of(changed) != identity_of(cfg)


@given(_CONFIGS, _KEYS, _SECTIONS)
def test_adding_or_changing_any_other_training_key_changes_the_identity(
    cfg: dict[str, Any], key: str, value: Any
) -> None:
    if key in ("train_steps", "train_from"):
        return
    changed = copy.deepcopy(cfg)
    changed["training"][key] = value if cfg["training"].get(key) != value else [value]

    assert identity_of(changed) != identity_of(cfg)


def _is_a_step_name(name: str) -> bool:
    digits = name[len("step_") :]
    return (
        name.startswith("step_")
        and digits.isascii()
        and digits.isdigit()
        and not digits.startswith("0")
    )


@settings(max_examples=60)
@given(
    st.sets(
        st.text(
            alphabet="step_0123456789 x-+." + chr(0x665), min_size=1, max_size=12
        ).filter(lambda name: name not in (".", "..")),
        max_size=12,
    )
)
def test_a_directory_is_a_step_directory_exactly_when_its_name_is_step_and_a_canonical_number(
    names: set[str],
) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        model_path = Path(scratch)
        for name in names:
            (model_path / name).mkdir()

        listed = step_directories(model_path)

    expected = sorted(int(name[5:]) for name in names if _is_a_step_name(name))
    assert [directory.step for directory in listed] == expected
    assert all(directory.path.name == f"step_{directory.step}" for directory in listed)
