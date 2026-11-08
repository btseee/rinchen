"""Property test for `rinchen.train.eole_cfg.validate_train_config`: whatever shape a
section of the working config is replaced by, or whatever keys it is given, the
verdict is None or a `ConfigError`, never an `AttributeError`, `TypeError` or
`KeyError` from a section that was not the mapping the checks expected."""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from rinchen.errors import ConfigError
from rinchen.train.eole_cfg import validate_train_config
from tests.support.eole_train_config import CORPUS, make_config

_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-2, 2),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(alphabet="ab$ ", max_size=4),
)
_JUNK = st.recursive(
    _SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(alphabet="abcd", max_size=3), children, max_size=3),
    ),
    max_leaves=6,
)
# Where the junk goes: every section the checks look into.
_SECTIONS = [
    (),
    ("data",),
    ("data", CORPUS),
    ("data", "valid"),
    ("transforms_configs",),
    ("transforms_configs", "sentencepiece"),
    ("model",),
    ("model", "encoder"),
    ("model", "embeddings"),
    ("training",),
]


def _verdict(cfg: object) -> str | None:
    try:
        validate_train_config(cfg)
    except ConfigError as exc:
        return str(exc)
    return None


@given(section=st.sampled_from(_SECTIONS), junk=_JUNK)
def test_a_section_replaced_by_anything_gets_a_verdict_not_a_crash(
    section: tuple[str, ...], junk: Any
) -> None:
    cfg = make_config()
    if not section:
        cfg = junk
    else:
        node = cfg
        for key in section[:-1]:
            node = node[key]
        node[section[-1]] = junk

    verdict = _verdict(cfg)

    assert verdict is None or verdict.startswith("invalid Eole config: ")


@given(
    section=st.sampled_from(_SECTIONS[:1] + _SECTIONS[2:]),
    extra=st.dictionaries(st.text(alphabet="abcd_", max_size=4), _JUNK, max_size=3),
)
def test_keys_added_to_a_section_get_a_verdict_not_a_crash(
    section: tuple[str, ...], extra: dict[str, Any]
) -> None:
    cfg = make_config()
    node = cfg
    for key in section:
        node = node[key]
    node.update(extra)

    verdict = _verdict(cfg)

    assert verdict is None or verdict.startswith("invalid Eole config: ")
