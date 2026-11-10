"""Property tests for the readers of the sources of a suite (D13, Task 4). The join of several
languages by id gives the rows of the hub language in the hub's order with every language's text
for the id, whatever order the other languages list their rows in; a human export gives the same
suite bytes in any order; the lines of a text file are numbered from one and read back as the text
they were written from."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.eval.suite_rules import SuiteSource
from rinchen.eval.suite_sources import (
    SourceRow,
    build_multiway,
    human_rows,
    lines_rows,
)
from rinchen.eval.suites import Suite
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# Every reserved symbol of the project holds a `<`, and a sentence holds no line break.
SENTENCES = st.text(
    alphabet=st.characters(
        codec="utf-8",
        exclude_characters="<",
        exclude_categories=("Cc", "Cs", "Zl", "Zp"),
    ),
    min_size=1,
    max_size=20,
).filter(lambda text: text.strip() != "")
ROW_IDS = st.from_regex(r"[A-Za-z0-9._:-]{1,12}", fullmatch=True)
PARTNERS = ("eng_Latn", "rus_Cyrl", "zho_Hans", "deu_Latn")


def source() -> SuiteSource:
    return SuiteSource(
        dataset=None, revision=None, pinned=False, files={}, missing_languages=()
    )


@st.composite
def multiway_inputs(draw: st.DrawFn) -> dict[str, list[SourceRow]]:
    ids = draw(st.lists(ROW_IDS, min_size=1, max_size=8, unique=True))
    partners = draw(st.lists(st.sampled_from(PARTNERS), min_size=1, unique=True))
    inputs: dict[str, list[SourceRow]] = {}
    for language in ("khk_Cyrl", *partners):
        order = draw(st.permutations(ids))
        inputs[language] = [
            SourceRow(row_id, draw(SENTENCES), None) for row_id in order
        ]
    return inputs


@SETTINGS
@given(multiway_inputs())
def test_the_join_follows_the_hub_and_holds_the_text_of_every_language(
    inputs: dict[str, list[SourceRow]],
) -> None:
    suite = build_multiway(
        name="mini",
        kind="fixture",
        benchmark="mini",
        split="devtest",
        languages=inputs,
        configured=tuple(inputs),
        source=source(),
    )

    hub_ids = [row.id for row in inputs["khk_Cyrl"]]
    assert [row.id for row in suite.rows] == hub_ids
    for row in suite.rows:
        for language, rows in inputs.items():
            expected = next(item.text for item in rows if item.id == row.id)
            assert row.text[language] == expected
    assert len(suite.directions) == 2 * (len(inputs) - 1)
    assert Suite.from_bytes(suite.to_bytes()).to_bytes() == suite.to_bytes()


@SETTINGS
@given(multiway_inputs())
def test_a_language_that_lacks_an_id_is_refused_and_never_intersected(
    inputs: dict[str, list[SourceRow]],
) -> None:
    partner = next(language for language in inputs if language != "khk_Cyrl")
    shortened = {**inputs, partner: inputs[partner][1:]}
    if len(inputs["khk_Cyrl"]) == 1:
        shortened[partner] = []

    with pytest.raises(DataError) as caught:
        build_multiway(
            name="mini",
            kind="fixture",
            benchmark="mini",
            split="devtest",
            languages=shortened,
            configured=tuple(shortened),
            source=source(),
        )

    assert partner in str(caught.value)


EXPORT_ROWS = st.lists(
    st.tuples(
        st.sampled_from(("d1", "d2", "d3", None)),
        st.integers(min_value=0, max_value=30),
        SENTENCES,
        SENTENCES,
        st.sampled_from(("eng_Latn", "rus_Cyrl")),
    ),
    min_size=1,
    max_size=10,
    unique_by=lambda item: (item[0], item[1]) if item[0] is not None else item[2:4],
)


def export_of(items: list[tuple[str | None, int, str, str, str]]) -> list[dict]:
    return [
        {
            "src": src,
            "tgt": tgt,
            "src_lang": language,
            "tgt_lang": "khk_Cyrl",
            "doc_id": doc,
            "seg_idx": None if doc is None else seg,
            "set": "news",
            "origin": "human",
            "split": "test",
        }
        for doc, seg, src, tgt, language in items
    ]


@SETTINGS
@given(EXPORT_ROWS, st.randoms(use_true_random=False))
def test_a_human_export_gives_the_same_suite_in_any_order(
    items: list[tuple[str | None, int, str, str, str]], random: object
) -> None:
    rows = export_of(items)
    shuffled = list(rows)
    random.shuffle(shuffled)  # type: ignore[attr-defined]

    first = human_rows(rows, name="human-test", source=source())
    second = human_rows(shuffled, name="human-test", source=source())

    assert first.to_bytes() == second.to_bytes()
    for direction in first.directions:
        reverse = "-".join(reversed(direction.split("-")))
        assert reverse not in first.directions


@SETTINGS
@given(st.lists(SENTENCES, min_size=1, max_size=12), st.booleans())
def test_the_lines_of_a_file_are_numbered_from_one_and_read_back(
    sentences: list[str], final_line_end: bool
) -> None:
    text = "\n".join(sentences) + ("\n" if final_line_end else "")

    rows = lines_rows("eng_Latn", text.encode("utf-8"))

    assert [row.id for row in rows] == [
        f"l{n:06d}" for n in range(1, len(sentences) + 1)
    ]
    assert [row.text for row in rows] == sentences
