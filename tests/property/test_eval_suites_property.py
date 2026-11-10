"""Property tests of the suite file of D13: a generated valid suite survives a trip through its
bytes in both directions, one changed byte of the golden file is a suite or a `DataError` and
never anything else, and the view of every declared direction of a generated suite has columns
of equal length that hold exactly the rows with both languages. Under the `no_network` guard like
every D13 test module."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval.suites import Suite, SuiteRow, SuiteSource
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=80,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
GOLDEN_BYTES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "golden_suite.json"
).read_bytes()
LANGUAGES = sorted(known_language_codes())
# Every reserved symbol of the project holds a `<`, and a line break is a control character or
# one of the two separators, so none is drawn.
SENTENCES = st.text(
    alphabet=st.characters(
        codec="utf-8",
        exclude_categories=("Cc", "Cs", "Zl", "Zp"),
        exclude_characters="<",
    ),
    min_size=1,
    max_size=20,
).filter(lambda sentence: sentence.strip() != "")
DOCS = st.one_of(st.none(), st.sampled_from(["d1", "d2", "d3", "doc:4"]))
FILE_NAMES = st.from_regex(
    r"[a-z]{1,8}(/[a-z0-9_-]{1,8}){0,2}\.parquet", fullmatch=True
)
DIGESTS = st.from_regex(r"[0-9a-f]{64}", fullmatch=True)
REVISIONS = st.from_regex(r"[0-9a-f]{40}", fullmatch=True)
KINDS_AND_SPLITS = st.sampled_from(
    [
        ("fixture", "dev"),
        ("fixture", "devtest"),
        ("fixture", "test"),
        ("benchmark", "dev"),
        ("benchmark", "devtest"),
        ("human", "test"),
    ]
)


@st.composite
def sources(draw: st.DrawFn) -> SuiteSource:
    revision = draw(st.one_of(st.none(), REVISIONS))
    return SuiteSource(
        dataset=draw(st.one_of(st.none(), st.sampled_from(["org/name", "ntrex"]))),
        revision=revision,
        pinned=revision is not None and draw(st.booleans()),
        files=draw(st.dictionaries(FILE_NAMES, DIGESTS, max_size=3)),
        missing_languages=tuple(
            sorted(draw(st.lists(st.sampled_from(LANGUAGES), unique=True, max_size=3)))
        ),
    )


@st.composite
def rows_of(draw: st.DrawFn, languages: list[str]) -> list[SuiteRow]:
    rows: list[SuiteRow] = []
    for index in range(draw(st.integers(min_value=1, max_value=6))):
        # The first row has every language, so every pair of languages has a row.
        held = languages
        if index:
            held = draw(
                st.lists(
                    st.sampled_from(languages),
                    min_size=1,
                    max_size=len(languages),
                    unique=True,
                )
            )
        text = {language: draw(SENTENCES) for language in held}
        rows.append(SuiteRow(id=f"r{index}", doc=draw(DOCS), text=text))
    return rows


@st.composite
def suites(draw: st.DrawFn) -> Suite:
    languages = draw(
        st.lists(st.sampled_from(LANGUAGES), min_size=2, max_size=4, unique=True)
    )
    rows = draw(rows_of(languages))
    pairs = [
        f"{first}-{second}"
        for first in languages
        for second in languages
        if first != second
        and any(first in row.text and second in row.text for row in rows)
    ]
    kind, split = draw(KINDS_AND_SPLITS)
    return Suite(
        name=draw(st.sampled_from(["mini", "a.b_c-d", "x" * 64])),
        kind=kind,
        benchmark=draw(st.sampled_from(["ntrex", "golden", "annotators"])),
        split=split,
        source=draw(sources()),
        directions=tuple(
            draw(st.lists(st.sampled_from(pairs), min_size=1, unique=True))
        ),
        rows=rows,
    )


@SETTINGS
@given(suite=suites())
def test_a_generated_suite_survives_a_trip_through_its_bytes(suite: Suite) -> None:
    data = suite.to_bytes()

    again = Suite.from_bytes(data)

    assert again == suite
    assert again.to_bytes() == data
    assert data.endswith(b"}\n")


@SETTINGS
@given(
    position=st.integers(min_value=0, max_value=len(GOLDEN_BYTES) - 1),
    mask=st.integers(min_value=1, max_value=255),
)
def test_one_changed_byte_of_the_golden_file_is_a_suite_or_a_data_error(
    position: int, mask: int
) -> None:
    golden = Suite.from_bytes(GOLDEN_BYTES)
    changed = bytearray(GOLDEN_BYTES)
    changed[position] ^= mask

    try:
        parsed = Suite.from_bytes(bytes(changed))
    except DataError:
        return

    assert (parsed == golden) == (parsed.to_bytes() == GOLDEN_BYTES)
    assert Suite.from_bytes(parsed.to_bytes()) == parsed


@SETTINGS
@given(suite=suites())
def test_the_view_of_every_declared_direction_has_columns_of_equal_length(
    suite: Suite,
) -> None:
    for direction in suite.directions:
        view = suite.view(direction)
        source_language, target_language = direction.split("-")
        both = [
            row
            for row in suite.rows
            if source_language in row.text and target_language in row.text
        ]

        assert len(view.ids) == len(view.sources) == len(view.references) == len(both)
        assert view.ids == tuple(row.id for row in both)
        assert view.sources == tuple(row.text[source_language] for row in both)
        assert view.references == tuple(row.text[target_language] for row in both)
        if view.docs is not None:
            assert view.docs == tuple(row.doc for row in both)
        assert (view.docs is not None) == all(row.doc is not None for row in both)
