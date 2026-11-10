"""The span metrics of D12 (spec T6 and T7): whether a name or a term is PRESENT in a text
(`span_present`: NFC and casefolded, at a boundary, followed by at most four letters and a
boundary), the entity table of a suite (`EntityTable`), and the scores of the entity rows
(`score_entities`) and the terminology rows (`score_terms`). The vectors are the spike's
(`s19_span_vectors`) and the golden rows e1 to e4 and t1 to t3; every expectation is a
literal, and the invisible or look-alike characters are built with `chr()`."""

from __future__ import annotations

import dataclasses

import pytest
import regex

from rinchen.errors import DataError
from rinchen.eval import robust_spans
from rinchen.eval.robust_spans import (
    MAX_SUFFIX_LETTERS,
    EntityTable,
    SpanOutcome,
    score_entities,
    score_terms,
    span_present,
)

BREVE = chr(0x306)
ACUTE = chr(0x301)
NBSP = chr(0xA0)
ARABIC_INDIC_THREE = chr(0x663)
SHARP_S = chr(0xDF)
MARKER = "SECRET-MARKER-91"

TABLE = EntityTable(
    {
        "bat": {"eng_Latn": ("Bat",), "khk_Cyrl": ("Бат",)},
        "ulaanbaatar": {
            "eng_Latn": ("Ulaanbaatar", "Ulan Bator"),
            "khk_Cyrl": ("Улаанбаатар",),
        },
    }
)


def span_case(
    text: str, spellings: tuple[str, ...], present: bool, label: str
) -> object:
    return pytest.param(text, spellings, present, id=label)


# --- span_present -----------------------------------------------------------------------

PRESENT = [
    span_case("Бат Улаанбаатарт амьдардаг.", ("Улаанбаатар",), True, "case_suffix_t"),
    span_case("Улаанбаатараас ирсэн", ("Улаанбаатар",), True, "case_suffix_aas"),
    span_case("Улаанбаатарынхаа", ("Улаанбаатар",), False, "five_letter_suffix"),
    span_case("ULAANBAATAR is", ("Ulaanbaatar",), True, "upper_case_text"),
    span_case("улаанбаатар", ("УЛААНБААТАР",), True, "upper_case_spelling"),
    span_case("Ulan   Bator is", ("Ulan Bator",), True, "several_spaces_in_text"),
    span_case("Ulan Bator is", ("Ulan   Bator",), True, "several_spaces_in_spelling"),
    span_case("Ulan\tBator", ("Ulan Bator",), True, "tab_between_words"),
    span_case("Ulan\nBator", ("Ulan Bator",), True, "newline_between_words"),
    span_case(f"Ulan{NBSP}Bator", ("Ulan Bator",), True, "no_break_space"),
    span_case("UlanBator", ("Ulan Bator",), False, "no_space_at_all"),
    span_case("Ulan Bator is", ("Ulaanbaatar", "Ulan Bator"), True, "second_spelling"),
    span_case("Ulaanbaatar is", ("Ulaanbaatar", "Ulan Bator"), True, "first_spelling"),
    span_case("Ulan Ator is", ("Ulaanbaatar", "Ulan Bator"), False, "neither"),
    span_case("Батор", ("Бат",), True, "a_short_name_is_a_prefix_of_a_longer_one"),
    span_case("Сбат", ("Бат",), False, "a_letter_before"),
    span_case("Бат2", ("Бат",), False, "a_digit_after"),
    span_case("2Бат", ("Бат",), False, "a_digit_before"),
    span_case(f"Бат{ARABIC_INDIC_THREE}", ("Бат",), False, "a_foreign_digit_after"),
    span_case("Бат ааа2", ("Бат",), True, "a_digit_after_another_word"),
    span_case("Батааа2", ("Бат",), False, "a_digit_after_the_suffix"),
    span_case("Bat's book", ("Bat",), True, "english_possessive"),
    span_case("bat", ("Bat",), True, "lower_case_text"),
    span_case("Тэнд Улан-Баторе байна", ("Улан-Батор",), True, "hyphenated_name"),
    span_case("Дорж-Бат", ("Бат",), True, "hyphen_before"),
    span_case("(Бат)", ("Бат",), True, "parentheses"),
    span_case("Бат.", ("Бат",), True, "full_stop_after"),
    span_case("", ("Бат",), False, "empty_text"),
    span_case("Бат", ("Дорж",), False, "another_name"),
    span_case("Батаааа", ("Бат",), True, "four_letters_after"),
    span_case("Батааааа", ("Бат",), False, "five_letters_after"),
    span_case("Бат аааа", ("Бат",), True, "the_next_word_is_not_a_suffix"),
    span_case(f"Бат{ACUTE}", ("Бат",), True, "a_mark_after_is_a_suffix_letter"),
    span_case(f"{ACUTE}Бат", ("Бат",), False, "a_mark_before"),
    span_case(f"Бат{ACUTE * 5}", ("Бат",), False, "five_marks_after"),
    span_case("STRASSE", (f"Stra{SHARP_S}e",), True, "casefold_is_not_lower"),
    span_case("a+b", ("a+b",), True, "plus_is_literal"),
    span_case("aab", ("a+b",), False, "plus_is_not_a_quantifier"),
    span_case("a|b", ("a|b",), True, "bar_is_literal"),
    span_case("a", ("a|b",), False, "bar_is_not_an_alternation"),
    span_case("Bax", ("Ba.",), False, "dot_is_literal"),
    span_case("Ba.", ("Ba.",), True, "dot_matches_itself"),
    span_case("C++ rocks", ("C++",), True, "a_name_of_pluses"),
    span_case(" Bat ", (" Bat ",), True, "edge_spaces_of_a_spelling_do_not_count"),
    span_case("Bat", (" Bat ",), True, "edge_spaces_need_no_space_in_the_text"),
]


@pytest.mark.parametrize(("text", "spellings", "present"), PRESENT)
def test_span_present(text: str, spellings: tuple[str, ...], present: bool) -> None:
    assert span_present(text, spellings) is present


def test_a_decomposed_i_with_a_breve_is_the_composed_letter() -> None:
    decomposed = "Ба" + "и" + BREVE + "р"
    composed = "Ба" + "й" + "р"

    assert decomposed != composed
    assert span_present(decomposed, [composed])
    assert span_present(composed, [decomposed])
    assert span_present(decomposed, [decomposed])
    assert not span_present(composed, ["Баир"])


def test_the_suffix_allowance_is_four_letters_and_can_be_set() -> None:
    assert MAX_SUFFIX_LETTERS == 4
    assert span_present("Батор", ["Бат"], max_suffix=2)
    assert not span_present("Батор", ["Бат"], max_suffix=1)
    assert not span_present("Батор", ["Бат"], max_suffix=0)
    assert span_present("Бат", ["Бат"], max_suffix=0)
    assert span_present("Бат. Батор", ["Бат"], max_suffix=0)
    assert span_present("Батааааа", ["Бат"], max_suffix=5)
    assert not span_present("Батааааа", ["Бат"], max_suffix=4)


def test_a_found_span_does_not_hide_a_later_one() -> None:
    # The first `Бат` is part of a longer word; the second stands alone.
    assert span_present("Сбат Бат", ["Бат"])
    assert span_present("Бат2 Бат", ["Бат"])
    assert span_present("Батааааа Бат", ["Бат"])


@pytest.mark.parametrize(
    "spellings", [[], (), [""], ["  "], ["Бат", ""], ["Бат", f" {NBSP}\t"]]
)
def test_no_spelling_or_a_blank_one_is_refused(spellings: list[str]) -> None:
    with pytest.raises(ValueError, match="spelling"):
        span_present("x", spellings)


def test_a_string_in_place_of_a_list_of_spellings_is_refused() -> None:
    with pytest.raises(TypeError, match="spellings"):
        span_present("Бат", "Бат")


def test_a_negative_allowance_is_refused() -> None:
    with pytest.raises(ValueError, match="max_suffix"):
        span_present("Бат", ["Бат"], max_suffix=-1)


def test_the_refusal_names_no_text() -> None:
    with pytest.raises(ValueError) as caught:
        span_present(MARKER, [MARKER, ""])

    assert MARKER not in str(caught.value)


def test_the_pattern_is_built_once_per_spellings_and_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robust_spans._pattern.cache_clear()
    built: list[str] = []
    compile_pattern = regex.compile

    def counting_compile(pattern: str, *args: object, **kwargs: object) -> object:
        built.append(pattern)
        return compile_pattern(pattern, *args, **kwargs)

    monkeypatch.setattr(regex, "compile", counting_compile)

    for _ in range(3):
        span_present("Бат", ["Бат"])
        span_present("Батор", ["Бат"])
    assert len(built) == 1
    span_present("Бат", ["Бат"], max_suffix=2)
    assert len(built) == 2
    span_present("Бат", ["Бат", "Дорж"])
    assert len(built) == 3
    span_present("Бат", ["Дорж"])
    assert len(built) == 4
    span_present("Бат", ("Бат",))
    assert len(built) == 4


def test_a_cached_pattern_does_not_leak_into_another_spelling_or_allowance() -> None:
    assert span_present("Батор", ["Бат"], max_suffix=4)
    assert not span_present("Батор", ["Бат"], max_suffix=0)
    assert not span_present("Дорж", ["Бат"], max_suffix=4)
    assert span_present("Дорж", ["Дорж"], max_suffix=4)
    assert span_present("Батор", ["Бат"], max_suffix=4)


# --- EntityTable ------------------------------------------------------------------------


def test_the_table_lists_the_spellings_of_an_entity_in_a_language() -> None:
    assert TABLE.spellings_of("bat", "khk_Cyrl") == ("Бат",)
    assert TABLE.spellings_of("ulaanbaatar", "eng_Latn") == (
        "Ulaanbaatar",
        "Ulan Bator",
    )


@pytest.mark.parametrize(
    ("entity_id", "language"),
    [("dorj", "khk_Cyrl"), ("bat", "rus_Cyrl"), ("dorj", "rus_Cyrl")],
)
def test_an_entity_or_a_language_the_table_lacks_is_a_data_error_naming_both(
    entity_id: str, language: str
) -> None:
    with pytest.raises(DataError) as caught:
        TABLE.spellings_of(entity_id, language)

    assert entity_id in str(caught.value)
    assert language in str(caught.value)


def test_the_refusal_of_the_table_quotes_none_of_its_spellings() -> None:
    table = EntityTable({"bat": {"khk_Cyrl": (MARKER,)}})

    with pytest.raises(DataError) as caught:
        table.spellings_of("bat", "eng_Latn")

    assert MARKER not in str(caught.value)


def test_a_language_with_an_empty_spelling_list_is_a_data_error() -> None:
    table = EntityTable({"bat": {"khk_Cyrl": ()}})

    with pytest.raises(DataError, match="bat"):
        table.spellings_of("bat", "khk_Cyrl")


def test_the_table_and_the_outcome_are_frozen_and_the_outcome_slotted() -> None:
    outcome = SpanOutcome(1, 1, 0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        TABLE.spellings = {}  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.found = 0  # type: ignore[misc]
    assert not hasattr(outcome, "__dict__")
    assert [field.name for field in dataclasses.fields(SpanOutcome)] == [
        "expected",
        "found",
        "excluded",
    ]


# --- score_entities ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hypothesis", "entity_ids", "tgt_lang", "counts"),
    [
        # The golden rows e1 to e4.
        ("Бат Улаанбаатарт амьдардаг.", ["bat", "ulaanbaatar"], "khk_Cyrl", (2, 2, 0)),
        ("Дорж Дарханд амьдардаг.", ["bat", "ulaanbaatar"], "khk_Cyrl", (2, 0, 0)),
        (
            "Bat is in Ulan Bator's office.",
            ["bat", "ulaanbaatar"],
            "eng_Latn",
            (2, 2, 0),
        ),
        ("Улаанбаатарынхаа зам.", ["ulaanbaatar"], "khk_Cyrl", (1, 0, 0)),
        # What the golden rows leave out.
        ("Бат Дархан.", ["ulaanbaatar", "bat"], "khk_Cyrl", (2, 1, 0)),
        ("Улаанбаатар", ["bat", "ulaanbaatar"], "khk_Cyrl", (2, 1, 0)),
        ("Бат", ["bat"], "eng_Latn", (1, 0, 0)),
        ("Bat", ["bat"], "khk_Cyrl", (1, 0, 0)),
        ("Ulaanbaatar", ["ulaanbaatar"], "eng_Latn", (1, 1, 0)),
        ("", ["bat"], "khk_Cyrl", (1, 0, 0)),
        ("Бат", [], "khk_Cyrl", (0, 0, 0)),
        ("Бат", ["bat", "bat"], "khk_Cyrl", (2, 2, 0)),
    ],
)
def test_score_entities(
    hypothesis: str,
    entity_ids: list[str],
    tgt_lang: str,
    counts: tuple[int, int, int],
) -> None:
    outcome = score_entities(hypothesis, entity_ids, tgt_lang, TABLE)

    assert (outcome.expected, outcome.found, outcome.excluded) == counts


@pytest.mark.parametrize(
    ("entity_id", "language"),
    [("dorj", "khk_Cyrl"), ("bat", "rus_Cyrl")],
)
def test_an_entity_the_table_cannot_spell_is_a_data_error_that_quotes_no_text(
    entity_id: str, language: str
) -> None:
    with pytest.raises(DataError) as caught:
        score_entities(f"{MARKER} Бат", ["bat", entity_id], language, TABLE)

    assert entity_id in str(caught.value)
    assert language in str(caught.value)
    assert MARKER not in str(caught.value)


# --- score_terms ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "hypothesis", "terms", "counts"),
    [
        # The golden rows t1 to t3.
        (
            "Where is the table?",
            "Ширээ хаана байна вэ?",
            [("table", "ширээ")],
            (1, 1, 0),
        ),
        (
            "The table and the window.",
            "Ширээ ба цонхнууд.",
            [("table", "ширээ"), ("window", "цонх")],
            (2, 2, 0),
        ),
        (
            "The chair is here.",
            "Тавиур энд байна.",
            [("chair", "сандал"), ("lamp", "чийдэн")],
            (1, 0, 1),
        ),
        # What the golden rows leave out.
        ("The lamp.", "Чийдэн.", [("table", "ширээ")], (0, 0, 1)),
        (
            "The lamp and the lamp.",
            "Чийдэн.",
            [("table", "ширээ"), ("window", "цонх")],
            (0, 0, 2),
        ),
        ("The Table.", "Ширээ.", [("table", "ширээ")], (1, 1, 0)),
        ("The tables.", "Ширээнүүд.", [("table", "ширээ")], (1, 1, 0)),
        ("The table.", "Сандал.", [("table", "ширээ")], (1, 0, 0)),
        ("The table.", "", [("table", "ширээ")], (1, 0, 0)),
        (
            "The table and the table.",
            "Ширээ.",
            [("table", "ширээ"), ("table", "ширээ")],
            (2, 2, 0),
        ),
        ("The tablecloth.", "Ширээ.", [("table", "ширээ")], (0, 0, 1)),
        ("The table.", "Ширээ.", [], (0, 0, 0)),
        (
            "The window and the table.",
            "Цонх, сандал.",
            [("table", "ширээ"), ("window", "цонх")],
            (2, 1, 0),
        ),
    ],
)
def test_score_terms(
    source: str,
    hypothesis: str,
    terms: list[tuple[str, str]],
    counts: tuple[int, int, int],
) -> None:
    outcome = score_terms(source, hypothesis, terms)

    assert (outcome.expected, outcome.found, outcome.excluded) == counts


def test_a_term_absent_from_the_source_is_excluded_even_when_its_target_is_there() -> (
    None
):
    outcome = score_terms("The chair.", "Чийдэн, сандал.", [("lamp", "чийдэн")])

    assert outcome == SpanOutcome(expected=0, found=0, excluded=1)


def test_a_blank_term_is_refused_and_names_no_text() -> None:
    with pytest.raises(ValueError) as caught:
        score_terms(MARKER, MARKER, [("", "ширээ")])

    assert MARKER not in str(caught.value)
    with pytest.raises(ValueError) as target_caught:
        score_terms(f"{MARKER} table", MARKER, [("table", " ")])

    assert MARKER not in str(target_caught.value)
