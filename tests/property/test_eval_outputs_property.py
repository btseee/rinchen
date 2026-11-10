"""Property tests of the outputs file of D13: outputs generated over a generated suite survive a
trip through their bytes (the file read back is the outputs written, in the order of the suite),
and one flipped byte of a golden file is outputs or a `DataError` and never anything else. Under
the `no_network` guard like every D13 test module."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.eval.outputs import Output, OutputKey, outputs_from_bytes, outputs_to_bytes
from rinchen.eval.suites import Suite, SuiteRow, SuiteSource
from tests.support.eval_fakes import (
    GOLDEN_OUTPUTS_FILES,
    golden_outputs,
    golden_suite,
    no_network,  # noqa: F401 - the fixture of the mark
)

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
LANGUAGES = ("eng_Latn", "khk_Cyrl", "rus_Cyrl", "deu_Latn")
STATUSES = ("ok", "empty", "too_long", "truncated", "no_output")
# Any text a system may write: no lone surrogate (the writer refuses it), line breaks included.
TEXTS = st.text(alphabet=st.characters(codec="utf-8"), max_size=40)
OUTPUTS = st.builds(Output, text=TEXTS, status=st.sampled_from(STATUSES))
GOLDEN_BYTES = GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()


@st.composite
def suites(draw: st.DrawFn) -> Suite:
    """A suite of one to five rows, every row holding all four languages, and a random
    non-empty list of distinct directions between them."""
    pairs = [(a, b) for a in LANGUAGES for b in LANGUAGES if a != b]
    directions = draw(
        st.lists(st.sampled_from(pairs), min_size=1, max_size=5, unique=True)
    )
    rows = tuple(
        SuiteRow(id=f"r{index}", doc=None, text={code: "x" for code in LANGUAGES})
        for index in range(draw(st.integers(min_value=1, max_value=5)))
    )
    return Suite(
        name="gen",
        kind="fixture",
        benchmark="gen",
        split="devtest",
        source=SuiteSource(None, None, False, {}, ()),
        directions=tuple(f"{a}-{b}" for a, b in directions),
        rows=rows,
    )


@st.composite
def suite_and_outputs(draw: st.DrawFn) -> tuple[Suite, dict[OutputKey, Output]]:
    """A suite and complete outputs for a random subset of its directions, built in the order
    of the suite's directions and rows."""
    suite = draw(suites())
    chosen = draw(
        st.lists(
            st.sampled_from(suite.directions),
            unique=True,
            max_size=len(suite.directions),
        )
    )
    outputs: dict[OutputKey, Output] = {}
    for direction in suite.directions:
        if direction in chosen:
            for row_id in suite.view(direction).ids:
                outputs[OutputKey(direction, row_id)] = draw(OUTPUTS)
    return suite, outputs


@SETTINGS
@given(pair=suite_and_outputs())
def test_outputs_survive_a_trip_through_their_bytes(
    pair: tuple[Suite, dict[OutputKey, Output]],
) -> None:
    suite, outputs = pair

    data = outputs_to_bytes(outputs)
    again = outputs_from_bytes(data, suite)

    assert again == outputs
    assert list(again) == list(outputs)
    assert outputs_to_bytes(again) == data
    assert data == b"" or data.endswith(b"\n")


@SETTINGS
@given(pair=suite_and_outputs(), seed=st.randoms(use_true_random=False))
def test_the_order_of_the_lines_never_changes_what_is_read(
    pair: tuple[Suite, dict[OutputKey, Output]], seed: object
) -> None:
    suite, outputs = pair
    lines = outputs_to_bytes(outputs).split(b"\n")[:-1] if outputs else []
    seed.shuffle(lines)  # type: ignore[attr-defined]

    again = outputs_from_bytes(b"".join(line + b"\n" for line in lines), suite)

    assert list(again) == list(outputs)
    assert again == outputs


@SETTINGS
@given(
    position=st.integers(min_value=0, max_value=len(GOLDEN_BYTES) - 1),
    mask=st.integers(min_value=1, max_value=255),
)
def test_one_changed_byte_of_the_golden_file_is_outputs_or_a_data_error(
    position: int, mask: int
) -> None:
    suite = golden_suite()
    changed = bytearray(GOLDEN_BYTES)
    changed[position] ^= mask

    try:
        parsed = outputs_from_bytes(bytes(changed), suite)
    except DataError as error:
        assert str(error).startswith("not an outputs file:")
        return

    assert outputs_from_bytes(outputs_to_bytes(parsed), suite) == parsed
    assert (parsed == golden_outputs("alpha")) == (
        outputs_to_bytes(parsed) == GOLDEN_BYTES
    )


@SETTINGS
@given(cut=st.integers(min_value=0, max_value=len(GOLDEN_BYTES) - 1))
def test_a_file_cut_short_is_a_data_error_or_a_complete_direction(cut: int) -> None:
    suite = golden_suite()

    try:
        parsed = outputs_from_bytes(GOLDEN_BYTES[:cut], suite)
    except DataError:
        return

    # Only a cut at a boundary that leaves whole directions (nothing, or the first ten lines)
    # can be a file; a cut inside the second direction leaves it incomplete.
    assert len(parsed) in (0, 10, 20)
