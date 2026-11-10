"""The suites D12 builds from the robustness toy (spec T17, T18): the main suite the gate scores,
the control suites that must score far below it, and the traditional-script output suite.

The counts are those of the brief (a row per direction of every held-out item). The suites hold
no row the metrics exclude and their references satisfy the metrics: scoring a suite with its own
references as the hypotheses gives 1.0 everywhere (a suite that fails that is a suite that
cannot be told from a broken system).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rinchen.eval.robust_hypotheses import Hypothesis
from rinchen.eval.robust_suite import (
    Request,
    RequestKey,
    Suite,
    SuiteRow,
    read_suite,
    suite_requests,
    write_suite,
)
from rinchen.eval.robustness import (
    RobustnessResult,
    RobustnessSystem,
    compute_robustness,
)
from rinchen.script_conv.rules import transliterate
from tests.support.robust_suites import (
    build_main_suite,
    build_mong_output_suite,
    build_term_control_suite,
    build_unseen_placeholder_suite,
)
from tests.support.robust_toy import (
    ENTITIES,
    RobustRow,
    RobustToy,
    build_robust_toy,
)
from tests.support.tiny_parallel import NOUNS

OPEN, CLOSE = chr(0x27E6), chr(0x27E7)
SIX_DIRECTIONS = {
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "mn_Mong-eng_Latn",
    "mn_Mong-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
}
NUMBER_KINDS = ("int", "decimal", "grouped", "percent", "currency", "date", "time")
CONTROL_TERMS = {"khk_Cyrl": "тавиур", "eng_Latn": "stand", "rus_Cyrl": "подставка"}
UNTRAINED_PAIRS = (
    ("eng", "eng", "eng_Latn"),
    ("eng", "rus", "rus_Cyrl"),
    ("khk", "khk", "khk_Cyrl"),
    ("rus", "rus", "rus_Cyrl"),
    ("rus", "eng", "eng_Latn"),
)
SIZE_LIMIT = 2_000_000


@pytest.fixture(scope="module")
def toy() -> RobustToy:
    return build_robust_toy()


@pytest.fixture(scope="module")
def main(toy: RobustToy) -> Suite:
    return build_main_suite(toy)


@pytest.fixture(scope="module")
def term_control(toy: RobustToy) -> Suite:
    return build_term_control_suite(toy)


@pytest.fixture(scope="module")
def unseen() -> Suite:
    return build_unseen_placeholder_suite()


@pytest.fixture(scope="module")
def mong_output(toy: RobustToy) -> Suite:
    return build_mong_output_suite(toy)


def _test_rows(toy: RobustToy, *kinds: str) -> list[RobustRow]:
    return [row for row in toy.rows if row.split == "test" and row.kind in kinds]


# ---- the main suite ------------------------------------------------------------------------


def test_the_main_suite_has_the_rows_of_the_brief(main: Suite) -> None:
    assert main.row_counts() == {
        "numbers": 960,
        "placeholders": 600,
        "entities": 240,
        "terms": 400,
        "script": 900,
        "mong": 480,
    }
    assert sum(len(row.entities) for row in main.sections["entities"]) == 480
    assert sum(len(row.terms) for row in main.sections["terms"]) == 1_044


def test_the_main_suite_is_named_and_holds_the_whole_entity_table(
    main: Suite,
) -> None:
    assert main.name == "robust-main"
    assert dict(main.entities.spellings) == ENTITIES
    assert len(main.entities.spellings) == 20


def test_a_numbers_row_is_a_held_out_number_sentence_of_a_pair_direction(
    toy: RobustToy, main: Suite
) -> None:
    rows = _test_rows(toy, "number", "number2")
    suite_rows = main.sections["numbers"]

    assert len(rows) == 960
    assert [row.id for row in suite_rows] == [f"num-{i:04d}" for i in range(1, 961)]
    for source, row in zip(rows, suite_rows, strict=True):
        assert (row.src_lang, row.tgt_lang) == (source.src_lang, source.tgt_lang)
        assert (row.source, row.reference) == (source.src, source.tgt)
        assert row.script is None


def test_the_numbers_are_split_by_the_kind_of_number(main: Suite) -> None:
    categories = Counter(row.category for row in main.sections["numbers"])

    assert categories == {kind: 120 for kind in (*NUMBER_KINDS, "two_numbers")}


def test_a_two_numbers_row_is_a_two_number_sentence(
    toy: RobustToy, main: Suite
) -> None:
    rows = _test_rows(toy, "number", "number2")
    pairs = list(zip(rows, main.sections["numbers"], strict=True))

    assert sum(row.category == "two_numbers" for _, row in pairs) == 120
    for source, row in pairs:
        assert (row.category == "two_numbers") == (len(source.numbers) == 2)
        assert row.category in ("two_numbers", source.numbers[0].kind)


def test_a_placeholders_row_is_a_held_out_placeholder_sentence_of_six_directions(
    toy: RobustToy, main: Suite
) -> None:
    rows = _test_rows(toy, "placeholder")
    suite_rows = main.sections["placeholders"]

    assert [row.id for row in suite_rows] == [f"ph-{i:04d}" for i in range(1, 601)]
    assert {f"{row.src_lang}-{row.tgt_lang}" for row in suite_rows} == SIX_DIRECTIONS
    assert Counter(f"{row.src_lang}-{row.tgt_lang}" for row in suite_rows) == {
        direction: 100 for direction in SIX_DIRECTIONS
    }
    for source, row in zip(rows, suite_rows, strict=True):
        assert (row.source, row.reference) == (source.src, source.tgt)
        assert OPEN in row.source
        assert OPEN in (row.reference or "")


def test_the_held_out_placeholder_sentences_all_have_three_placeholders(
    main: Suite,
) -> None:
    assert {row.category for row in main.sections["placeholders"]} == {"noun"}
    assert {row.source.count(OPEN) for row in main.sections["placeholders"]} == {3}


def test_the_genitive_sentence_is_the_reorder_category() -> None:
    genitive = _hand_row(
        f"The {_ph(0)} of the {_ph(1)}.",
        f"{_ph(1)}ийн {_ph(0)}.",
        kind="placeholder",
        template=7,
    )
    plain = _hand_row(
        f"Where is {_ph(2)}?",
        f"{_ph(2)} хаана байна вэ?",
        kind="placeholder",
        template=3,
    )

    suite = build_main_suite(RobustToy(rows=(genitive, plain)))

    assert [row.category for row in suite.sections["placeholders"]] == [
        "reorder",
        "noun",
    ]


def _ph(index: int) -> str:
    return f"{OPEN}{index}{CLOSE}"


def _hand_row(src: str, tgt: str, **changes: object) -> RobustRow:
    fields: dict[str, object] = {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "src": src,
        "tgt": tgt,
        "origin": "human",
        "corpus": "toy-x",
        "split": "test",
        "kind": "base",
    }
    fields.update(changes)
    return RobustRow(**fields)  # type: ignore[arg-type]


def test_an_entities_row_names_its_person_and_its_city_in_six_directions(
    toy: RobustToy, main: Suite
) -> None:
    rows = _test_rows(toy, "entity")
    suite_rows = main.sections["entities"]

    assert [row.id for row in suite_rows] == [f"ent-{i:04d}" for i in range(1, 241)]
    assert {row.category for row in suite_rows} == {"person_city"}
    assert {f"{row.src_lang}-{row.tgt_lang}" for row in suite_rows} == SIX_DIRECTIONS
    for source, row in zip(rows, suite_rows, strict=True):
        assert row.entities == source.entities
        assert (row.source, row.reference) == (source.src, source.tgt)
        person, city = row.entities
        assert ENTITIES[person]["eng_Latn"] and ENTITIES[city]["rus_Cyrl"]


def test_a_terms_row_holds_every_noun_of_a_held_out_base_sentence(
    toy: RobustToy, main: Suite
) -> None:
    rows = [row for row in _test_rows(toy, "base") if row.src_lang != "mn_Mong"]
    suite_rows = main.sections["terms"]
    columns = {"khk_Cyrl": 0, "eng_Latn": 1, "rus_Cyrl": 2}

    assert len(rows) == 400
    assert [row.id for row in suite_rows] == [f"term-{i:04d}" for i in range(1, 401)]
    assert {row.category for row in suite_rows} == {"glossary"}
    for source, row in zip(rows, suite_rows, strict=True):
        assert (row.source, row.reference) == (source.src, source.tgt)
        assert len(row.terms) == len(source.nouns)
        for pair, noun in zip(row.terms, source.nouns, strict=True):
            assert pair.src == NOUNS[noun][columns[source.src_lang]]
            assert pair.tgt == NOUNS[noun][columns[source.tgt_lang]]


def test_the_terms_rows_are_the_four_pair_directions(main: Suite) -> None:
    directions = Counter(
        f"{row.src_lang}-{row.tgt_lang}" for row in main.sections["terms"]
    )

    assert directions == {
        "eng_Latn-khk_Cyrl": 100,
        "khk_Cyrl-eng_Latn": 100,
        "khk_Cyrl-rus_Cyrl": 100,
        "rus_Cyrl-khk_Cyrl": 100,
    }


def test_the_script_section_has_the_trained_rows_and_the_untrained_pairs(
    main: Suite,
) -> None:
    rows = main.sections["script"]

    assert Counter(row.category for row in rows) == {"trained": 400, "untrained": 500}
    assert [row.id for row in rows[:400]] == [f"trained-{i:04d}" for i in range(1, 401)]


def test_the_trained_rows_are_the_terms_rows_with_their_source_language(
    main: Suite,
) -> None:
    trained = main.sections["script"][:400]

    for script_row, terms_row in zip(trained, main.sections["terms"], strict=True):
        assert script_row.src_lang == terms_row.src_lang
        assert script_row.tgt_lang == terms_row.tgt_lang
        assert script_row.source == terms_row.source
        assert script_row.script is None
        assert script_row.reference is None


def test_the_untrained_rows_have_no_source_language_and_five_untrained_pairs(
    toy: RobustToy, main: Suite
) -> None:
    untrained = main.sections["script"][400:]

    assert all(row.src_lang is None for row in untrained)
    assert all(row.script is None and row.reference is None for row in untrained)
    start = 0
    for source, label, target in UNTRAINED_PAIRS:
        block = untrained[start : start + 100]
        start += 100
        expected = [
            row.src
            for row in _test_rows(toy, "base")
            if row.src_lang.startswith(source)
        ][:100]
        assert [row.id for row in block] == [
            f"untrained-{source}-{label}-{i:03d}" for i in range(1, 101)
        ]
        assert {row.tgt_lang for row in block} == {target}
        assert [row.source for row in block] == expected


def test_the_khalkha_untrained_sources_are_the_first_hundred_rows_so_each_sentence_twice(
    main: Suite,
) -> None:
    block = [
        row for row in main.sections["script"] if row.id.startswith("untrained-khk-")
    ]

    assert len(block) == 100
    assert len({row.source for row in block}) == 50
    assert block[0].source == block[1].source


def test_the_script_rows_hold_the_sentences_of_the_toy(
    toy: RobustToy, main: Suite
) -> None:
    sentences = {row.src for row in toy.rows if row.split == "test"}

    assert {row.source for row in main.sections["script"]} <= sentences


def test_a_mong_row_is_a_khalkha_sentence_and_the_same_sentence_in_traditional_script(
    toy: RobustToy, main: Suite
) -> None:
    rows = [
        row
        for row in _test_rows(toy, "base", "entity", "placeholder")
        if row.src_lang == "khk_Cyrl" and row.tgt_lang in ("eng_Latn", "rus_Cyrl")
    ]
    suite_rows = main.sections["mong"]

    assert len(rows) == 480
    assert [row.id for row in suite_rows] == [f"mong-{i:04d}" for i in range(1, 481)]
    for source, row in zip(rows, suite_rows, strict=True):
        assert row.src_lang == "khk_Cyrl"
        assert row.tgt_lang == source.tgt_lang
        assert (row.source, row.reference) == (source.src, source.tgt)
        assert row.source_mong == transliterate(source.src)
        assert row.category == source.kind


def test_the_mong_rows_are_split_by_the_kind_of_sentence(main: Suite) -> None:
    categories = Counter(row.category for row in main.sections["mong"])

    assert categories == {"base": 200, "entity": 80, "placeholder": 200}


def test_every_source_of_the_main_suite_is_held_out(
    toy: RobustToy, main: Suite
) -> None:
    held_in: dict[str, set[str]] = {}
    for row in toy.rows:
        if row.split != "test":
            held_in.setdefault(row.src_lang, set()).add(row.src)
            held_in.setdefault(row.tgt_lang, set()).add(row.tgt)
    anywhere = set().union(*held_in.values())

    for section, rows in main.sections.items():
        for row in rows:
            assert row.source not in anywhere, section
            if row.reference is not None:
                assert row.reference not in anywhere, section
            if row.source_mong is not None:
                assert row.source_mong not in held_in["mn_Mong"], section


# ---- the control suites ---------------------------------------------------------------------


def test_the_term_control_is_the_terms_section_alone(
    main: Suite, term_control: Suite
) -> None:
    assert term_control.name == "robust-term-control"
    assert term_control.row_counts() == {"terms": 400}
    assert sum(len(row.terms) for row in term_control.sections["terms"]) == 1_044
    for control, original in zip(
        term_control.sections["terms"], main.sections["terms"], strict=True
    ):
        assert (control.id, control.source, control.reference, control.category) == (
            original.id,
            original.source,
            original.reference,
            original.category,
        )
        assert [pair.src for pair in control.terms] == [
            pair.src for pair in original.terms
        ]


def test_every_glossary_target_of_the_control_is_a_word_the_model_never_writes(
    toy: RobustToy, term_control: Suite
) -> None:
    for row in term_control.sections["terms"]:
        assert {pair.tgt for pair in row.terms} == {CONTROL_TERMS[row.tgt_lang]}
    texts = [text for row in toy.rows for text in (row.src, row.tgt)]
    for word in CONTROL_TERMS.values():
        assert not [text for text in texts if word in text], word


def test_the_unseen_placeholder_suite_has_seven_indices_in_four_directions(
    unseen: Suite,
) -> None:
    rows = unseen.sections["placeholders"]

    assert unseen.name == "robust-unseen-placeholders"
    assert unseen.row_counts() == {"placeholders": 28}
    assert len({row.id for row in rows}) == 28
    assert Counter(f"{row.src_lang}-{row.tgt_lang}" for row in rows) == {
        "eng_Latn-khk_Cyrl": 7,
        "khk_Cyrl-eng_Latn": 7,
        "rus_Cyrl-khk_Cyrl": 7,
        "khk_Cyrl-rus_Cyrl": 7,
    }
    assert {row.category for row in rows} == {"noun"}


def test_the_unseen_placeholders_are_the_indices_the_toy_never_trains(
    unseen: Suite,
) -> None:
    rows = unseen.sections["placeholders"]

    indices = {int(row.source.split(OPEN)[1].split(CLOSE)[0]) for row in rows}
    assert indices == {10, 11, 12, 25, 50, 75, 99}
    assert all(index >= 10 for index in indices)


@pytest.mark.parametrize(
    ("src_lang", "tgt_lang", "source", "reference"),
    [
        ("eng_Latn", "khk_Cyrl", "This is {}.", "Энэ бол {}."),
        ("khk_Cyrl", "eng_Latn", "Энэ бол {}.", "This is {}."),
        ("rus_Cyrl", "khk_Cyrl", "Это {}.", "Энэ бол {}."),
        ("khk_Cyrl", "rus_Cyrl", "Энэ бол {}.", "Это {}."),
    ],
)
def test_an_unseen_placeholder_sentence_is_the_d7_sentence_with_the_placeholder(
    unseen: Suite, src_lang: str, tgt_lang: str, source: str, reference: str
) -> None:
    rows = {
        (row.src_lang, row.tgt_lang, row.source): row.reference
        for row in unseen.sections["placeholders"]
    }

    for index in (10, 11, 12, 25, 50, 75, 99):
        glyph = f"{OPEN}{index}{CLOSE}"
        assert rows[(src_lang, tgt_lang, source.format(glyph))] == reference.format(
            glyph
        )


def test_the_traditional_script_output_suite_asks_for_the_script(
    toy: RobustToy, mong_output: Suite
) -> None:
    rows = mong_output.sections["script"]
    base = [
        row
        for row in _test_rows(toy, "base")
        if (row.src_lang, row.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    ][:100]

    assert mong_output.name == "robust-mong-output"
    assert mong_output.row_counts() == {"script": 100}
    assert [row.id for row in rows] == [f"mong-out-{i:04d}" for i in range(1, 101)]
    assert [row.source for row in rows] == [row.src for row in base]
    for row in rows:
        assert (row.src_lang, row.tgt_lang) == ("eng_Latn", "khk_Cyrl")
        assert row.script == "Mong"
        assert row.category == "mong_output"
        assert row.reference is None


# ---- the files -------------------------------------------------------------------------------


def _all_suites(
    main: Suite, term_control: Suite, unseen: Suite, mong_output: Suite
) -> dict[str, Suite]:
    return {
        "main": main,
        "term_control": term_control,
        "unseen": unseen,
        "mong_output": mong_output,
    }


def test_every_suite_round_trips_through_the_file_and_is_small(
    tmp_path: Path,
    main: Suite,
    term_control: Suite,
    unseen: Suite,
    mong_output: Suite,
) -> None:
    for name, suite in _all_suites(main, term_control, unseen, mong_output).items():
        path = tmp_path / f"{name}.json"

        digest = write_suite(suite, path)
        read, read_digest = read_suite(path)

        assert read_digest == digest
        assert read.to_bytes() == suite.to_bytes() == path.read_bytes()
        assert read.row_counts() == suite.row_counts()
        assert read.name == suite.name
        assert path.stat().st_size < SIZE_LIMIT, name


def test_the_builders_are_deterministic(
    toy: RobustToy, main: Suite, term_control: Suite, unseen: Suite, mong_output: Suite
) -> None:
    again = build_robust_toy()

    assert build_main_suite(again).to_bytes() == main.to_bytes()
    assert build_term_control_suite(again).to_bytes() == term_control.to_bytes()
    assert build_unseen_placeholder_suite().to_bytes() == unseen.to_bytes()
    assert build_mong_output_suite(again).to_bytes() == mong_output.to_bytes()


def test_the_builders_take_the_test_split_only() -> None:
    held_in = _hand_row("This is a book.", "Энэ бол ном.", split="train", nouns=(0,))
    held_out = _hand_row("This is a star.", "Энэ бол од.", nouns=(5,))
    toy = RobustToy(rows=(held_in, held_out))

    terms = build_main_suite(toy).sections["terms"]
    control = build_term_control_suite(toy).sections["terms"]

    assert [row.source for row in terms] == ["This is a star."]
    assert [row.source for row in control] == ["This is a star."]


# ---- the references satisfy the metrics ----------------------------------------------------------


def _script_text(row: SuiteRow) -> str:
    plain = {
        "eng_Latn": "The book is here.",
        "khk_Cyrl": "Ном энд байна.",
        "rus_Cyrl": "Книга здесь.",
    }[row.tgt_lang]
    return transliterate(plain) if row.script == "Mong" else plain


def _reference_answers(suite: Suite) -> dict[RequestKey, Hypothesis]:
    """Every request answered with the row's reference (a script row has none: it is answered
    with a sentence in the script of its target)."""
    rows = {
        (section, row.id): row
        for section, section_rows in suite.sections.items()
        for row in section_rows
    }
    answers: dict[RequestKey, Hypothesis] = {}
    for request in suite_requests(suite):
        row = rows[(request.key.section, request.key.id)]
        text = row.reference if row.reference is not None else _script_text(row)
        answers[request.key] = Hypothesis(text, "ok")
    return answers


def _score(suite: Suite, answers: dict[RequestKey, Hypothesis]) -> RobustnessResult:
    return compute_robustness(
        suite,
        answers,
        suite_sha256="0" * 64,
        system=RobustnessSystem(name="the-references"),
    )


def test_the_references_of_the_main_suite_score_one_everywhere_and_exclude_nothing(
    main: Suite,
) -> None:
    result = _score(main, _reference_answers(main))
    sections = result.sections

    for name, scored, ok in (
        ("numbers", 960, 960),
        ("placeholders", 600, 600),
        ("entities", 480, 480),
        ("terms", 1_044, 1_044),
        ("script", 900, 900),
        ("mong", 480, 480),
    ):
        section = sections[name]
        assert dict(section.excluded) == {}, name
        assert (section.scored, section.ok, section.value) == (scored, ok, 1.0), name
    assert sections["script"].details["below_floor"] == 0
    assert sections["mong"].details["agreement"] == 1.0
    assert (
        sections["mong"].details["chrf_cyrl"] == sections["mong"].details["chrf_mong"]
    )
    assert dict(sections["numbers"].statuses) == {"ok": 960}


def test_the_references_exclude_no_mention_of_an_entity_or_a_term(main: Suite) -> None:
    sections = _score(main, _reference_answers(main)).sections

    assert sections["entities"].details["rows_all_found"] == 240
    assert sections["terms"].details["rows_all_found"] == 400


def test_the_term_control_scores_zero_with_nothing_excluded_when_the_references_answer(
    term_control: Suite,
) -> None:
    section = _score(term_control, _reference_answers(term_control)).sections["terms"]

    assert (section.scored, section.ok, section.value) == (1_044, 0, 0.0)
    assert dict(section.excluded) == {}


def test_the_references_of_the_unseen_placeholder_suite_score_one(
    unseen: Suite,
) -> None:
    section = _score(unseen, _reference_answers(unseen)).sections["placeholders"]

    assert (section.scored, section.ok, section.value) == (28, 28, 1.0)
    assert dict(section.excluded) == {}


def test_a_traditional_script_answer_to_the_output_suite_is_pure(
    mong_output: Suite,
) -> None:
    section = _score(mong_output, _reference_answers(mong_output)).sections["script"]

    assert (section.scored, section.value) == (100, 1.0)
    assert dict(section.excluded) == {}


def test_a_cyrillic_answer_to_the_output_suite_is_not_pure(mong_output: Suite) -> None:
    answers = {
        request.key: Hypothesis("Ном энд байна.", "ok")
        for request in suite_requests(mong_output)
    }

    section = _score(mong_output, answers).sections["script"]

    assert section.value == 0.0
    assert section.details["below_floor"] == 100


def test_the_requests_of_the_main_suite_are_the_rows_and_the_two_arms(
    main: Suite,
) -> None:
    requests: list[Request] = suite_requests(main)
    arms = Counter((request.key.section, request.key.arm) for request in requests)

    assert len(requests) == 960 + 600 + 240 + 400 + 900 + 2 * 480
    assert arms[("mong", "cyrl")] == arms[("mong", "mong")] == 480
