"""The placeholder metric of D12 (spec T5): which numbered placeholders a text holds
(`placeholders_in`), which brackets of the pair are broken (`damaged_brackets`), how a raw
hypothesis is scored against the placeholders of its source (`score_placeholders`) and when
a source and a reference carry the same ones (`placeholders_agree`). A placeholder is the
glyph pair of `spm.symbols` around one or two digits without a leading zero, 0 to 99; the
tables are literals, the expected values of the golden rows p1 to p6 come from the spike's
independent prototypes, and the glyphs are built with `chr()`."""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.eval.robust_placeholders import (
    PlaceholderOutcome,
    damaged_brackets,
    placeholders_agree,
    placeholders_in,
    score_placeholders,
)
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols

OPEN = chr(0x27E6)
CLOSE = chr(0x27E7)
ARABIC_INDIC_THREE = chr(0x663)


def ph(index: int) -> str:
    return f"{OPEN}{index}{CLOSE}"


def shape(
    outcome: PlaceholderOutcome,
) -> tuple[int, int, int, int, int, bool, bool | None]:
    return (
        outcome.expected,
        outcome.found,
        outcome.missing,
        outcome.extra,
        outcome.damaged,
        outcome.exact,
        outcome.order_ok,
    )


# --- placeholders_in and damaged_brackets -----------------------------------------------


@pytest.mark.parametrize(
    ("text", "indices"),
    [
        (f"Where is {ph(3)}?", (3,)),
        (f"{ph(0)} {ph(12)} {ph(99)}", (0, 12, 99)),
        (f"{ph(1)}ийн {ph(0)}.", (1, 0)),
        (f"{ph(1)}{ph(2)}{ph(1)}", (1, 2, 1)),
        (f"{ph(5)}{CLOSE}", (5,)),
        (f"{OPEN}{ph(5)}", (5,)),
        ("", ()),
        ("no placeholder here", ()),
    ],
)
def test_placeholders_in_lists_the_well_formed_ones_in_order(
    text: str, indices: tuple[int, ...]
) -> None:
    assert placeholders_in(text) == indices


@pytest.mark.parametrize(
    ("text", "damaged"),
    [
        (f"{OPEN}03{CLOSE}", 2),
        (f"{OPEN}100{CLOSE}", 2),
        (f"{OPEN} 3 {CLOSE}", 2),
        (f"{OPEN}3", 1),
        (f"3{CLOSE}", 1),
        ("[[3]]", 0),
        ("<3>", 0),
        (f"{OPEN}{ARABIC_INDIC_THREE}{CLOSE}", 2),
        (f"{OPEN}{CLOSE}", 2),
        (f"{OPEN}3{OPEN}", 2),
    ],
)
def test_a_malformed_placeholder_is_no_placeholder_and_its_glyphs_are_damage(
    text: str, damaged: int
) -> None:
    assert placeholders_in(text) == ()
    assert damaged_brackets(text) == damaged


def test_a_stray_glyph_beside_a_well_formed_placeholder_is_damage_and_not_a_match() -> (
    None
):
    assert placeholders_in(f"{ph(3)}{CLOSE}") == (3,)
    assert damaged_brackets(f"{ph(3)}{CLOSE}") == 1
    assert placeholders_in(f"{OPEN}{ph(3)}") == (3,)
    assert damaged_brackets(f"{OPEN}{ph(3)}") == 1
    assert damaged_brackets(f"{ph(3)} {ph(4)}") == 0
    assert damaged_brackets("") == 0


def test_every_one_of_the_hundred_symbols_matches_once_with_its_own_index() -> None:
    symbols = user_defined_symbols()[-PLACEHOLDER_COUNT:]

    assert len(symbols) == 100
    for index, symbol in enumerate(symbols):
        assert placeholders_in(symbol) == (index,), symbol
        assert placeholders_in(f"a {symbol} b") == (index,)
        assert damaged_brackets(symbol) == 0
    assert placeholders_in("".join(symbols)) == tuple(range(100))


def test_no_other_reserved_symbol_and_no_direction_tag_matches() -> None:
    others = user_defined_symbols()[:-PLACEHOLDER_COUNT]

    assert len(others) == 18
    for symbol in others:
        assert placeholders_in(symbol) == (), symbol
        assert damaged_brackets(symbol) == 0, symbol


def test_the_glyphs_are_the_ones_the_symbols_use() -> None:
    first = user_defined_symbols()[-PLACEHOLDER_COUNT]

    assert (first[0], first[-1]) == (OPEN, CLOSE)
    assert first == ph(0)


# --- score_placeholders -----------------------------------------------------------------

SOURCE = f"Where is {ph(3)}?"
REFERENCE = f"{ph(3)} хаана байна вэ?"

SCORES = [
    # source, reference, hypothesis, (expected, found, missing, extra, damaged, exact, order_ok)
    pytest.param(SOURCE, REFERENCE, REFERENCE, (1, 1, 0, 0, 0, True, True), id="kept"),
    pytest.param(
        SOURCE,
        REFERENCE,
        "Хаана байна вэ?",
        (1, 0, 1, 0, 0, False, None),
        id="dropped",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)} {ph(3)}",
        (1, 1, 0, 1, 0, False, None),
        id="doubled",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)} {ph(3)} {ph(3)}",
        (1, 1, 0, 2, 0, False, None),
        id="tripled",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        ph(4),
        (1, 0, 1, 1, 0, False, None),
        id="another_index",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)} {ph(4)}",
        (1, 1, 0, 1, 0, False, None),
        id="a_foreign_one_beside",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{OPEN} 3 {CLOSE} хаана",
        (1, 0, 1, 0, 2, False, None),
        id="spaces_inside",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)} {ph(3)}{CLOSE}",
        (1, 1, 0, 1, 1, False, None),
        id="doubled_and_a_stray_glyph",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)}{CLOSE}",
        (1, 1, 0, 0, 1, False, None),
        id="only_a_stray_glyph",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{ph(3)} {OPEN}",
        (1, 1, 0, 0, 1, False, None),
        id="a_trailing_open_glyph",
    ),
    pytest.param(
        SOURCE,
        REFERENCE,
        f"{OPEN}03{CLOSE} {ph(3)}",
        (1, 1, 0, 0, 2, False, None),
        id="a_leading_zero_beside_the_real_one",
    ),
    pytest.param(
        SOURCE, REFERENCE, "", (1, 0, 1, 0, 0, False, None), id="empty_hypothesis"
    ),
    pytest.param(
        f"{ph(0)} {ph(1)}",
        f"{ph(1)} {ph(0)}",
        f"{ph(0)} {ph(1)}",
        (2, 2, 0, 0, 0, True, False),
        id="source_order_is_not_the_reference_order",
    ),
    pytest.param(
        f"{ph(0)} {ph(1)}",
        f"{ph(1)} {ph(0)}",
        f"{ph(1)} {ph(0)}",
        (2, 2, 0, 0, 0, True, True),
        id="reference_order",
    ),
    pytest.param(
        f"{ph(0)} and {ph(0)}",
        f"{ph(0)} ба {ph(0)}",
        f"{ph(0)} ба",
        (2, 1, 1, 0, 0, False, None),
        id="a_repeated_index_with_one_left",
    ),
    pytest.param(
        f"{ph(0)} and {ph(0)}",
        f"{ph(0)} ба {ph(0)}",
        f"{ph(0)} ба {ph(0)}",
        (2, 2, 0, 0, 0, True, True),
        id="a_repeated_index_with_both",
    ),
    pytest.param(
        f"{ph(0)} and {ph(0)}",
        f"{ph(0)} ба {ph(0)}",
        f"{ph(0)} ба {ph(0)} {ph(0)}",
        (2, 2, 0, 1, 0, False, None),
        id="a_repeated_index_with_one_too_many",
    ),
    pytest.param(
        f"{ph(1)} and {ph(2)}",
        f"{ph(1)} ба {ph(2)}",
        f"{ph(1)} ба {ph(1)}",
        (2, 1, 1, 1, 0, False, None),
        id="one_index_for_two",
    ),
    pytest.param(
        SOURCE,
        f"{ph(2)} хаана байна вэ?",
        f"{ph(2)} хаана байна вэ?",
        (1, 0, 1, 1, 0, False, None),
        id="scored_against_the_source_and_never_the_reference",
    ),
    pytest.param(
        f"{ph(7)} and {ph(12)} and {ph(99)}",
        f"{ph(99)} {ph(12)} {ph(7)}",
        f"{ph(12)} {ph(99)} {ph(7)}",
        (3, 3, 0, 0, 0, True, False),
        id="two_digit_indices",
    ),
    # The golden rows p1 to p5 (p6 has no placeholder in its source).
    pytest.param(
        f"Where is {ph(3)}?",
        f"{ph(3)} хаана байна вэ?",
        f"{ph(3)} хаана байна вэ?",
        (1, 1, 0, 0, 0, True, True),
        id="golden_p1",
    ),
    pytest.param(
        f"The {ph(0)} of the {ph(1)}.",
        f"{ph(1)}ийн {ph(0)}.",
        f"{ph(0)} {ph(1)}ийн.",
        (2, 2, 0, 0, 0, True, False),
        id="golden_p2",
    ),
    pytest.param(
        f"This is {ph(4)}.",
        f"Энэ бол {ph(4)}.",
        "Энэ бол.",
        (1, 0, 1, 0, 0, False, None),
        id="golden_p3",
    ),
    pytest.param(
        f"This is {ph(5)}.",
        f"Энэ бол {ph(5)}.",
        f"Энэ бол {ph(5)} {ph(5)}.",
        (1, 1, 0, 1, 0, False, None),
        id="golden_p4",
    ),
    pytest.param(
        f"This is {ph(6)}.",
        f"Энэ бол {ph(6)}.",
        f"Энэ бол {OPEN} 6 {CLOSE}.",
        (1, 0, 1, 0, 2, False, None),
        id="golden_p5",
    ),
]


@pytest.mark.parametrize(("source", "reference", "hypothesis", "expected"), SCORES)
def test_score_placeholders(
    source: str,
    reference: str,
    hypothesis: str,
    expected: tuple[int, int, int, int, int, bool, bool | None],
) -> None:
    outcome = score_placeholders(source, reference, hypothesis)

    assert shape(outcome) == expected


def test_a_hypothesis_that_is_the_raw_output_is_never_repaired() -> None:
    # A policy that appended the lost placeholder would make every row pass; the metric
    # reads what the engine wrote and nothing else.
    outcome = score_placeholders(SOURCE, REFERENCE, "Хаана байна вэ?")

    assert not outcome.exact
    assert outcome.missing == 1


def test_the_order_of_a_row_that_is_not_exact_is_not_asked() -> None:
    inexact = score_placeholders(
        f"{ph(0)} {ph(1)}", f"{ph(0)} {ph(1)}", f"{ph(1)} {ph(1)}"
    )

    assert inexact.order_ok is None
    assert not inexact.exact


def test_the_outcome_is_frozen_and_slotted() -> None:
    outcome = score_placeholders(SOURCE, REFERENCE, REFERENCE)

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.exact = False  # type: ignore[misc]
    assert not hasattr(outcome, "__dict__")
    assert [field.name for field in dataclasses.fields(PlaceholderOutcome)] == [
        "expected",
        "found",
        "missing",
        "extra",
        "damaged",
        "exact",
        "order_ok",
    ]


def test_scoring_is_deterministic_and_leaves_the_texts_alone() -> None:
    first = score_placeholders(SOURCE, REFERENCE, f"{ph(3)} {ph(3)}")
    second = score_placeholders(SOURCE, REFERENCE, f"{ph(3)} {ph(3)}")

    assert first == second


# --- placeholders_agree -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reference", "agree"),
    [
        (f"a {ph(1)}", f"b {ph(1)}", True),
        (f"a {ph(1)}", f"b {ph(2)}", False),
        ("no", "no", False),
        ("No placeholder.", "Тэмдэг алга.", False),
        (f"a {ph(1)}", "b", False),
        ("a", f"b {ph(1)}", False),
        (f"{ph(0)} {ph(1)}", f"{ph(1)} {ph(0)}", True),
        (f"{ph(0)} {ph(0)}", f"{ph(0)}", False),
        (f"{ph(0)}", f"{ph(0)} {ph(0)}", False),
        (f"{ph(0)} {ph(0)}", f"{ph(0)} {ph(0)}", True),
        (f"{ph(0)} {ph(1)}", f"{ph(0)} {ph(0)}", False),
        (f"{ph(3)}", f"{OPEN}03{CLOSE}", False),
        ("", "", False),
    ],
)
def test_placeholders_agree(source: str, reference: str, agree: bool) -> None:
    assert placeholders_agree(source, reference) is agree
