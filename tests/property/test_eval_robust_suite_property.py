"""Property tests for the suite file of D12: a generated valid suite survives a trip through its
bytes in both directions, one changed byte of a valid file is a suite or a `DataError` and never
anything else, and the requests of a suite are as many as its rows and arms and never repeat a
key."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from hypothesis import given
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite import (
    SECTIONS,
    Suite,
    SuiteRow,
    TermPair,
    suite_requests,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "robustness"
GOLDEN_BYTES = (GOLDEN / "golden_suite.json").read_bytes()
LANGUAGES = ("khk_Cyrl", "eng_Latn", "rus_Cyrl")
MONGOLIAN_LETTERS = st.text(
    alphabet="".join(chr(code) for code in range(0x1820, 0x1843)),
    min_size=1,
    max_size=8,
)
# Every reserved symbol of the project holds a `<`, so none is drawn.
TEXTS = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="<"),
    min_size=1,
    max_size=24,
).filter(lambda text: text.strip() != "")
IDS = st.from_regex(r"[A-Za-z0-9._-]{1,64}", fullmatch=True)
CATEGORIES = st.from_regex(r"[a-z0-9_]{1,24}", fullmatch=True)
TABLE_IDS = st.from_regex(r"[a-z0-9_]{1,32}", fullmatch=True)
OPTIONAL_TEXTS = st.one_of(st.none(), TEXTS)


@st.composite
def entity_tables(draw: st.DrawFn) -> EntityTable:
    ids = draw(st.lists(TABLE_IDS, min_size=1, max_size=3, unique=True))
    return EntityTable(
        {
            entity_id: {
                language: tuple(draw(st.lists(TEXTS, min_size=1, max_size=3)))
                for language in LANGUAGES
            }
            for entity_id in ids
        }
    )


@st.composite
def rows(draw: st.DrawFn, section: str, row_id: str, table: EntityTable) -> SuiteRow:
    category = draw(CATEGORIES)
    source = draw(TEXTS)
    languages = st.sampled_from(LANGUAGES)
    common = {"id": row_id, "category": category, "source": source}
    if section in ("numbers", "placeholders"):
        return SuiteRow(
            **common,
            src_lang=draw(languages),
            tgt_lang=draw(languages),
            reference=draw(TEXTS),
        )
    if section == "entities":
        mentions = draw(
            st.lists(st.sampled_from(sorted(table.spellings)), min_size=1, unique=True)
        )
        return SuiteRow(
            **common,
            src_lang=draw(languages),
            tgt_lang=draw(languages),
            reference=draw(OPTIONAL_TEXTS),
            entities=tuple(mentions),
        )
    if section == "terms":
        pairs = draw(st.lists(st.tuples(TEXTS, TEXTS), min_size=1, max_size=3))
        return SuiteRow(
            **common,
            src_lang=draw(languages),
            tgt_lang=draw(languages),
            reference=draw(OPTIONAL_TEXTS),
            terms=tuple(TermPair(src, tgt) for src, tgt in pairs),
        )
    if section == "script":
        target = draw(st.sampled_from((*LANGUAGES, "mn_Mong")))
        script: Literal["Mong", "Cyrl"] | None = None
        if target in ("khk_Cyrl", "mn_Mong"):
            script = draw(st.sampled_from((None, "Mong", "Cyrl")))
        return SuiteRow(
            **common,
            src_lang=draw(st.one_of(st.none(), languages)),
            tgt_lang=target,
            reference=draw(OPTIONAL_TEXTS),
            script=script,
        )
    return SuiteRow(
        **common,
        src_lang="khk_Cyrl",
        tgt_lang=draw(st.sampled_from(("eng_Latn", "rus_Cyrl"))),
        reference=draw(TEXTS),
        source_mong=draw(MONGOLIAN_LETTERS),
    )


@st.composite
def suites(draw: st.DrawFn) -> Suite:
    table = draw(entity_tables())
    sections: dict[str, tuple[SuiteRow, ...]] = {}
    for section in SECTIONS:
        ids = draw(st.lists(IDS, max_size=3, unique=True))
        sections[section] = tuple(draw(rows(section, row_id, table)) for row_id in ids)
    if not any(sections.values()):
        sections["numbers"] = (draw(rows("numbers", "n1", table)),)
    present = draw(st.sets(st.sampled_from(SECTIONS)))
    kept = {s: r for s, r in sections.items() if r or s in present}
    return Suite(name=draw(IDS), entities=table, sections=kept)


@given(suites())
def test_a_suite_survives_its_bytes(suite: Suite) -> None:
    assert Suite.from_bytes(suite.to_bytes()) == suite


@given(suites())
def test_the_bytes_of_a_suite_are_what_the_suite_read_from_them_writes(
    suite: Suite,
) -> None:
    data = suite.to_bytes()

    assert Suite.from_bytes(data).to_bytes() == data


def suite_or_refusal(data: bytes) -> None:
    try:
        Suite.from_bytes(data)
    except DataError:
        pass


@given(suites(), st.integers(min_value=0), st.integers(min_value=0, max_value=255))
def test_one_changed_byte_of_a_valid_file_is_a_suite_or_a_data_error(
    suite: Suite, position: int, value: int
) -> None:
    data = bytearray(suite.to_bytes())
    data[position % len(data)] = value

    suite_or_refusal(bytes(data))


@given(st.integers(min_value=0), st.integers(min_value=0, max_value=255))
def test_one_changed_byte_of_the_golden_file_is_a_suite_or_a_data_error(
    position: int, value: int
) -> None:
    data = bytearray(GOLDEN_BYTES)
    data[position % len(data)] = value

    suite_or_refusal(bytes(data))


@given(st.integers(min_value=0, max_value=len(GOLDEN_BYTES) - 2))
def test_a_golden_file_cut_short_is_a_data_error(
    cut: int,
) -> None:
    try:
        Suite.from_bytes(GOLDEN_BYTES[:cut])
    except DataError:
        return
    raise AssertionError("a cut file was read as a suite")


@given(st.binary(max_size=300))
def test_bytes_that_are_not_a_suite_are_a_data_error(data: bytes) -> None:
    try:
        Suite.from_bytes(data)
    except DataError:
        return
    raise AssertionError("arbitrary bytes were read as a suite")


@given(suites(), st.data())
def test_the_requests_are_one_per_row_and_arm_and_never_repeat_a_key(
    suite: Suite, data: st.DataObject
) -> None:
    mong_rows = suite.sections.get("mong", ())
    converted_ids = data.draw(
        st.sets(st.sampled_from([r.id for r in mong_rows]))
        if mong_rows
        else st.just(set())
    )

    requests = suite_requests(suite, {i: "converted text" for i in converted_ids})

    keys = [tuple(request.key) for request in requests]
    other_rows = sum(len(r) for s, r in suite.sections.items() if s != "mong")
    assert len(requests) == other_rows + 2 * len(mong_rows) + len(converted_ids)
    assert len(set(keys)) == len(keys)
    order = [SECTIONS.index(section) for section, _, _ in keys]
    assert order == sorted(order)
