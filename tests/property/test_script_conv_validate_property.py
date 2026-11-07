"""Property tests for `validate_traditional` over arbitrary text."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from rinchen.script_conv.alphabet import FVS, JOINERS, LETTERS, MVS, NIRUGU, NNBSP
from rinchen.script_conv.validate import (
    FOREIGN_CHARACTER,
    JOINER,
    VIOLATION_CODES,
    validate_traditional,
)

# Characters the validator treats specially, so random text reaches every code.
_SCRIPT_ALPHABET = sorted(LETTERS | FVS | JOINERS | {MVS, NNBSP, NIRUGU}) + [
    " ",
    "X",
    "9",
    chr(0x0430),
]

# The script alphabet mixed with any character of the Cyrillic block.
_SCRIPT_AND_CYRILLIC = st.sampled_from(_SCRIPT_ALPHABET) | st.characters(
    min_codepoint=0x0400, max_codepoint=0x04FF
)


@given(st.text())
def test_never_raises_and_reports_in_index_order(text: str) -> None:
    violations = validate_traditional(text)
    indexes = [violation.index for violation in violations]
    assert indexes == sorted(indexes)
    assert all(0 <= index < len(text) for index in indexes)
    assert {violation.code for violation in violations} <= VIOLATION_CODES


@given(st.text(alphabet=_SCRIPT_AND_CYRILLIC))
def test_each_joiner_and_each_cyrillic_letter_is_reported(text: str) -> None:
    violations = validate_traditional(text)
    assert len({violation.index for violation in violations}) == len(violations)
    joiners = sum(char in JOINERS for char in text)
    assert sum(violation.code == JOINER for violation in violations) == joiners
    cyrillic_indexes = [
        index for index, char in enumerate(text) if 0x0400 <= ord(char) <= 0x04FF
    ]
    foreign_indexes = [
        violation.index
        for violation in violations
        if violation.code == FOREIGN_CHARACTER
    ]
    assert foreign_indexes == cyrillic_indexes


@given(st.text(alphabet=st.sampled_from(_SCRIPT_ALPHABET)))
def test_text_free_of_violations_has_no_joiner_and_no_cyrillic(text: str) -> None:
    if not validate_traditional(text):
        assert not set(text) & JOINERS
        assert chr(0x0430) not in text


@given(st.text(alphabet=st.sampled_from(sorted(LETTERS)), max_size=30))
def test_letter_only_text_is_valid(text: str) -> None:
    assert validate_traditional(text) == ()
