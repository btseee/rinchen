"""The numbers metric of D12 (spec T2 to T4): what a number is (`numbers_in`), how a
hypothesis is scored against a reference (`score_numbers`) and when a source and a reference
carry the same numbers (`numbers_agree`). The tables are the vectors of the plan, produced
with the spike's reference implementation; every one is a literal, never recomputed.

A token is shown as `values [/unit] [?]`: the readings in increasing order joined by `|`,
the currency unit after a slash, a question mark for an ambiguous group."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterator
from decimal import Decimal

import pytest

from rinchen.eval import robust_numbers
from rinchen.eval.robust_numbers import (
    COMMA_DECIMAL_LANGUAGES,
    CURRENCY_SYMBOLS,
    GROUP_SPACES,
    MAX_NUMBER_TOKENS,
    POINT_DECIMAL_LANGUAGES,
    NumberOutcome,
    NumberToken,
    numbers_agree,
    numbers_in,
    score_numbers,
)
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols

NBSP = chr(0xA0)
NARROW_NBSP = chr(0x202F)
THIN_SPACE = chr(0x2009)
MINUS = chr(0x2212)
TUGRIK = chr(0x20AE)
EURO = chr(0x20AC)
ARABIC_INDIC = "".join(chr(0x660 + digit) for digit in range(10))
FULLWIDTH = "".join(chr(0xFF10 + digit) for digit in range(10))
MONGOLIAN = "".join(chr(0x1810 + digit) for digit in range(10))
THREE_IN_CJK = chr(0x4E09)


def placeholder(index: int) -> str:
    return user_defined_symbols()[-PLACEHOLDER_COUNT:][index]


def plain(value: Decimal) -> str:
    return format(value.normalize(), "f")


def view(token: NumberToken) -> str:
    shown = "|".join(plain(reading) for reading in sorted(token.readings))
    unit = f" /{token.unit}" if token.unit else ""
    return f"{shown}{unit}{' ?' if token.ambiguous else ''}"


def views(text: str, lang: str | None = None) -> list[str]:
    return [view(token) for token in numbers_in(text, lang)]


def surfaces(text: str, lang: str | None = None) -> list[str]:
    return [token.surface for token in numbers_in(text, lang)]


def case(text: str, lang: str | None, *expected: str) -> object:
    return pytest.param(text, lang, list(expected), id=f"{text!r}-{lang}")


# --- the constants ----------------------------------------------------------------------


def test_the_constants_are_the_ones_the_spec_rules() -> None:
    assert MAX_NUMBER_TOKENS == 64
    assert POINT_DECIMAL_LANGUAGES == {"eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"}
    assert COMMA_DECIMAL_LANGUAGES == {"rus_Cyrl", "deu_Latn", "tur_Latn", "kaz_Cyrl"}
    assert set(GROUP_SPACES) == {" ", NBSP, NARROW_NBSP, THIN_SPACE}
    assert set(CURRENCY_SYMBOLS) == set(TUGRIK + "$" + EURO) | {
        chr(0xA3),
        chr(0xA5),
        chr(0x20BD),
        chr(0x20A9),
    }


def test_the_records_are_frozen_and_slotted() -> None:
    token = NumberToken(frozenset({Decimal(1)}), None, "1", False)
    outcome = score_numbers("1", "1", tgt_lang=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        token.unit = "$"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.found = 2  # type: ignore[misc]
    assert not hasattr(token, "__dict__")
    assert not hasattr(outcome, "__dict__")


# --- numbers_in: the grammar --------------------------------------------------------------

GROUPS_AND_DECIMALS = [
    case("1,250.50", "eng_Latn", "1250.5"),
    case("1,250.50", None, "1250.5"),
    case("1.250,50", "deu_Latn", "1250.5"),
    case("1,250.50", "rus_Cyrl", "1250.5"),
    case("1 250,50", "khk_Cyrl", "1250.5"),
    case("1 250.5", None, "1250.5"),
    case(f"1{NBSP}250,50", "rus_Cyrl", "1250.5"),
    case(f"1{NARROW_NBSP}250,50", "rus_Cyrl", "1250.5"),
    case(f"1{THIN_SPACE}250,50", "rus_Cyrl", "1250.5"),
    case("1,250", "eng_Latn", "1250"),
    case("1,250", "rus_Cyrl", "1.25"),
    case("1,250", "khk_Cyrl", "1.25|1250 ?"),
    case("1,250", None, "1.25|1250 ?"),
    case("1,250", "mn_Mong", "1.25|1250 ?"),
    case("1.250", "eng_Latn", "1.25"),
    case("1.250", "deu_Latn", "1250"),
    case("1.250", "khk_Cyrl", "1.25|1250 ?"),
    case("1.250.000", "eng_Latn", "1250000"),
    case("1,250,000", "rus_Cyrl", "1250000"),
    case("1 250 000", None, "1250000"),
    case("3,5", "khk_Cyrl", "3.5"),
    case("3.5", "khk_Cyrl", "3.5"),
    case("0,500", "khk_Cyrl", "0.5"),
    case("0.500", "eng_Latn", "0.5"),
    case("12 500", None, "12500"),
    case("12 50", None, "12", "50"),
    case("3 5000", None, "3", "5000"),
]


@pytest.mark.parametrize(("text", "lang", "expected"), GROUPS_AND_DECIMALS)
def test_groups_and_decimal_marks(text: str, lang: str | None, expected: list[str]):
    assert views(text, lang) == expected


COMPONENTS = [
    case("12.03.2024", None, "12", "3", "2024"),
    case("2024-03-12", None, "2024", "3", "12"),
    case("03/12/2024", None, "3", "12", "2024"),
    case("12.03.24", None, "12", "3", "24"),
    case("10:30", None, "10", "30"),
    case("10:30:15", None, "10", "30", "15"),
    case("5-6", None, "5", "6"),
    case("3-5 range", None, "3", "5"),
    case("COVID-19", None, "19"),
    case("A4 and mp3", None, "4", "3"),
    case("version 1.2.3", None, "1", "2", "3"),
    case("10.0.0.1", None, "10", "0", "0", "1"),
    case("1,2,3", None, "1", "2", "3"),
    case("3, 500", None, "3", "500"),
    case("0", None, "0"),
    case("007", None, "7"),
    case("1e5", None, "1", "5"),
    case("no digits", None),
    case("", None),
    case("2024 оны 3 дугаар сарын 12", "khk_Cyrl", "2024", "3", "12"),
    case("1 250-5", None, "1250", "5"),
    case("1 250.50.3", None, "1250", "50.3"),
]


@pytest.mark.parametrize(("text", "lang", "expected"), COMPONENTS)
def test_dates_times_ranges_and_runs_that_are_not_one_number(
    text: str, lang: str | None, expected: list[str]
):
    assert views(text, lang) == expected


SIGNS_UNITS_AND_FOREIGN_DIGITS = [
    case("-5", None, "-5"),
    case(f"{MINUS}5", None, "-5"),
    case("a -5 b", None, "-5"),
    case("(-5)", None, "-5"),
    case("-3,5", "khk_Cyrl", "-3.5"),
    case("5 -6", None, "5", "-6"),
    case("1,500" + TUGRIK, None, f"1.5|1500 /{TUGRIK} ?"),
    case(TUGRIK + "1500", None, f"1500 /{TUGRIK}"),
    case("$20", None, "20 /$"),
    case("20 $", None, "20 /$"),
    case(f"1 500 {TUGRIK}", None, f"1500 /{TUGRIK}"),
    case(f"1 500{NBSP}{TUGRIK}", None, f"1500 /{TUGRIK}"),
    case(f"20{EURO}", None, f"20 /{EURO}"),
    case("20  $", None, "20"),
    case("$20 " + TUGRIK, None, f"20 /{TUGRIK}"),
    case(TUGRIK + "20$", None, "20 /$"),
    case(TUGRIK + "\n5", None, "5"),
    case("5\n" + TUGRIK, None, "5"),
    case("-1,250", "khk_Cyrl", "-1250|-1.25 ?"),
    case("5 and $", None, "5"),
    case("15%", None, "15"),
    case("15 %", None, "15"),
    case(f"{placeholder(12)} and 5", None, "5"),
    case(f"{placeholder(3)}{placeholder(4)}7", None, "7"),
    case(f"12{placeholder(3)}4", None, "12", "4"),
    case(ARABIC_INDIC[1:4], None, "123"),
    case(FULLWIDTH[1:3], None, "12"),
    case(MONGOLIAN[1:3], None, "12"),
    case(f"2{THREE_IN_CJK}", None, "2"),
    case(f"x {chr(0x4E8C)}{chr(0x3007)}{chr(0x4E8C)}{chr(0x56DB)} y", None, "2024"),
    # A symbol between two numbers is the unit of the one before it, never of both.
    case(
        f"859 571{TUGRIK} 3 ширхэг",
        "khk_Cyrl",
        f"859571 /{TUGRIK}",
        "3",
    ),
    case(
        f"3 ширхэг 859 571{TUGRIK}",
        "khk_Cyrl",
        "3",
        f"859571 /{TUGRIK}",
    ),
    case(f"5 {TUGRIK} 3", None, f"5 /{TUGRIK}", "3"),
    case(f"5{TUGRIK}3", None, f"5 /{TUGRIK}", "3"),
    case(f"5 {TUGRIK} $3", None, f"5 /{TUGRIK}", "3 /$"),
    case(f"5{TUGRIK}$3", None, f"5 /{TUGRIK}", "3 /$"),
    case("5 and $3", None, "5", "3 /$"),
    case("$5 x $3", None, "5 /$", "3 /$"),
    case(f"{TUGRIK}5 3", None, f"5 /{TUGRIK}", "3"),
    case(f"price {TUGRIK} 3", None, f"3 /{TUGRIK}"),
    # A symbol after a number is the number's when it touches the number's digits or
    # stands after a space and is not glued to a digit; a symbol after a space that is
    # glued to the next number's digits is that number's own.
    case("5 $7", None, "5", "7 /$"),
    case("$5 $7", None, "5 /$", "7 /$"),
    case("$7 $5", None, "7 /$", "5 /$"),
    case("$5 $7 $9", None, "5 /$", "7 /$", "9 /$"),
    case(f"{TUGRIK}5 {TUGRIK}7", None, f"5 /{TUGRIK}", f"7 /{TUGRIK}"),
    case(f"5{TUGRIK} 7{TUGRIK}", None, f"5 /{TUGRIK}", f"7 /{TUGRIK}"),
    case(f"5{TUGRIK} 7", None, f"5 /{TUGRIK}", "7"),
    case("5$ 7", None, "5 /$", "7"),
    case("5 $ 7", None, "5 /$", "7"),
    case("5 $x7", None, "5 /$", "7"),
    case("$5 $ 7", None, "5 /$", "7"),
    case(f"5{NBSP}${NBSP}7", None, "5 /$", "7"),
    case("$5, $7", None, "5 /$", "7 /$"),
    case("$5 and $7", None, "5 /$", "7 /$"),
    case("5$7", None, "5 /$", "7"),
    case(f"12:30{TUGRIK} 5", None, "12", "30", f"5 /{TUGRIK}"),
]


@pytest.mark.parametrize(("text", "lang", "expected"), SIGNS_UNITS_AND_FOREIGN_DIGITS)
def test_signs_units_placeholders_and_foreign_digits(
    text: str, lang: str | None, expected: list[str]
):
    assert views(text, lang) == expected


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("1 250,50", "khk_Cyrl", ["1 250,50"]),
        (f"1{NBSP}250,50", "rus_Cyrl", ["1 250,50"]),
        (f"1{NARROW_NBSP}250,50", "rus_Cyrl", ["1 250,50"]),
        (f"1{THIN_SPACE}250,50", "rus_Cyrl", ["1 250,50"]),
        # The components of one run share the run as written: the layout of a date.
        ("12.03.2024", None, ["12.03.2024"] * 3),
        ("03/12/2024", None, ["03/12/2024"] * 3),
        ("10:30", None, ["10:30"] * 2),
        ("5-6", None, ["5-6"] * 2),
        ("10.0.0.1", None, ["10.0.0.1"] * 4),
        (f"1{NBSP}250-5", None, ["1 250-5"] * 2),
        ("1 5 12.03", None, ["1", "5", "12.03"]),
        ("-5", None, ["-5"]),
        (f"{MINUS}5", None, ["-5"]),
        ("a -5 b", None, ["-5"]),
        ("(-5)", None, ["-5"]),
        ("007", None, ["007"]),
        (f"1 500 {TUGRIK}", None, ["1 500"]),
        ("1,250.50", "eng_Latn", ["1,250.50"]),
    ],
)
def test_the_surface_is_the_number_as_written_with_plain_group_spaces(
    text: str, lang: str | None, expected: list[str]
) -> None:
    assert surfaces(text, lang) == expected


def test_only_the_hundred_reserved_placeholders_are_removed_before_reading() -> None:
    for index in range(PLACEHOLDER_COUNT):
        assert numbers_in(f"{placeholder(index)} 5") == numbers_in("5")
    open_glyph, close_glyph = placeholder(0)[0], placeholder(0)[-1]

    assert views(f"{open_glyph}100{close_glyph}") == ["100"]
    assert views(f"{open_glyph}05{close_glyph}") == ["5"]


def test_numbers_in_gives_exact_values_not_floats() -> None:
    (token,) = numbers_in("0.1")

    assert token.readings == frozenset({Decimal("0.1")})
    assert all(isinstance(reading, Decimal) for reading in token.readings)


def test_a_number_with_many_digits_is_read_exactly_and_never_raises() -> None:
    digits = "9" * 5_000

    (positive,) = numbers_in(digits)
    (negative,) = numbers_in("-" + digits)

    assert positive.readings == frozenset({Decimal(digits)})
    # The unary minus would round to the context's 28 digits; the sign must not.
    assert negative.readings == frozenset({Decimal("-" + digits)})


# --- numbers_in: a runaway text ---------------------------------------------------------


def seconds(call: Callable[[], object]) -> float:
    """The best of three runs, so that a busy machine does not decide the test."""
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return best


@pytest.mark.parametrize(("count", "limit"), [(1_000, 0.1), (100_000, 0.5)])
def test_a_runaway_text_stops_after_the_sixty_fifth_token(
    count: int, limit: float
) -> None:
    text = " ".join(["1"] * count)

    tokens = numbers_in(text)

    assert len(tokens) == MAX_NUMBER_TOKENS + 1 == 65
    assert seconds(lambda: numbers_in(text)) < limit


def test_the_scan_stops_reading_runs_after_the_sixty_fifth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_read = []
    read_a_run = robust_numbers._run_numbers

    def counting(run: str, lang: str | None) -> Iterator[NumberToken]:
        runs_read.append(run)
        return read_a_run(run, lang)

    monkeypatch.setattr(robust_numbers, "_run_numbers", counting)

    tokens = numbers_in(" ".join(["1"] * 1_000))

    assert len(tokens) == 65
    assert len(runs_read) == 65


@pytest.mark.parametrize("joiner", ["-", ",", ":", "/", "."])
def test_a_runaway_run_is_cut_as_fast_as_a_runaway_text(joiner: str) -> None:
    text = joiner.join(["1"] * 100_000)

    assert len(numbers_in(text)) == 65
    assert seconds(lambda: numbers_in(text)) < 0.5


# --- score_numbers ----------------------------------------------------------------------


def shape(outcome: NumberOutcome) -> tuple[int, int, int, bool, bool, bool, bool, bool]:
    return (
        outcome.expected,
        outcome.found,
        outcome.extra,
        outcome.units_ok,
        outcome.exact,
        outcome.surface_exact,
        outcome.ambiguous,
        outcome.capped,
    )


def scored(
    reference: str,
    hypothesis: str,
    lang: str | None,
    *,
    counts: tuple[int, int, int],
    exact: bool,
    units_ok: bool = True,
    surface: bool = False,
    ambiguous: bool = False,
    capped: bool = False,
) -> object:
    expected, found, extra = counts
    return pytest.param(
        reference,
        hypothesis,
        lang,
        (expected, found, extra, units_ok, exact, surface, ambiguous, capped),
        id=f"{reference!r}-{hypothesis!r}",
    )


T = TUGRIK


def unit_score(
    reference: str,
    hypothesis: str,
    *,
    counts: tuple[int, int, int],
    exact: bool,
    units: bool = True,
    same: bool = True,
    lang: str = "eng_Latn",
) -> object:
    """A row of English text unless `lang` says otherwise: `same` says whether the
    surfaces (the written numbers without units) agree, `units` whether the units do."""
    return scored(
        reference,
        hypothesis,
        lang,
        counts=counts,
        exact=exact,
        units_ok=units,
        surface=same,
    )


SCORES = [
    scored(
        "Ном 1 250,50 байна.",
        "Book 1,250.50.",
        "eng_Latn",
        counts=(1, 1, 0),
        exact=True,
    ),
    scored("The price is 3.5.", "Цена 3,5.", "rus_Cyrl", counts=(1, 1, 0), exact=True),
    scored("3,5", "35", "khk_Cyrl", counts=(1, 0, 1), exact=False),
    scored("120", "210", None, counts=(1, 0, 1), exact=False),
    scored("7529", "752", None, counts=(1, 0, 1), exact=False),
    scored("7529", "7592", None, counts=(1, 0, 1), exact=False),
    # A date is one run: written in another layout it keeps its values (`exact`) and not
    # its written form (`surface_exact`), which compares the run with its separators.
    scored("2024.03.12", "12.03.2024", None, counts=(3, 3, 0), exact=True),
    scored("12.03.2024", "03/12/2024", None, counts=(3, 3, 0), exact=True),
    scored(
        "12.03.2024", "12.03.2024", None, counts=(3, 3, 0), exact=True, surface=True
    ),
    scored("12.03.2024", "12/03/2024", None, counts=(3, 3, 0), exact=True),
    scored(
        "2024-03-12", "2024-03-12", None, counts=(3, 3, 0), exact=True, surface=True
    ),
    scored("10:30", "10:30", None, counts=(2, 2, 0), exact=True, surface=True),
    scored(
        "Meeting 12.03.2024 at 10:30.",
        "Date 03/12/2024, 10:30.",
        None,
        counts=(5, 5, 0),
        exact=True,
    ),
    scored(
        "Meeting 12.03.2024 at 10:30.",
        "At 10:30 on 12.03.2024.",
        None,
        counts=(5, 5, 0),
        exact=True,
        surface=True,
    ),
    scored("10:30", "10.30", None, counts=(2, 0, 1), exact=False),
    scored("5 and 5", "5", None, counts=(2, 1, 0), exact=False),
    scored("5", "5 and 5", None, counts=(1, 1, 1), exact=False),
    scored("5 and 7", "7 and 5", None, counts=(2, 2, 0), exact=True, surface=True),
    scored(
        "1,500" + TUGRIK,
        f"1 500 {TUGRIK}",
        None,
        counts=(1, 1, 0),
        exact=True,
        ambiguous=True,
    ),
    scored(
        "1,500" + TUGRIK,
        "1 500",
        None,
        counts=(1, 1, 0),
        exact=False,
        units_ok=False,
        ambiguous=True,
    ),
    scored(
        "1,500",
        f"1 500{TUGRIK}",
        None,
        counts=(1, 1, 0),
        exact=True,
        ambiguous=True,
    ),
    scored("1 500", "1,500", None, counts=(1, 1, 0), exact=True, ambiguous=True),
    scored("15%", "15", None, counts=(1, 1, 0), exact=True, surface=True),
    scored("-5", "5", None, counts=(1, 0, 1), exact=False),
    scored("1.250", "1,250", "khk_Cyrl", counts=(1, 1, 0), exact=True, ambiguous=True),
    scored("1,250", "1.25", "eng_Latn", counts=(1, 0, 1), exact=False),
    scored(
        "1,250 and 1.25",
        "1,250 and 1 250",
        "khk_Cyrl",
        counts=(2, 2, 0),
        exact=True,
        ambiguous=True,
    ),
    scored("5", "", None, counts=(1, 0, 0), exact=False),
    scored("", "5", None, counts=(0, 0, 1), exact=False),
    scored(
        "no numbers", "no numbers", None, counts=(0, 0, 0), exact=True, surface=True
    ),
    scored(ARABIC_INDIC[1:3], "12", None, counts=(1, 1, 0), exact=True, surface=True),
    scored("5-6", "6, 5", None, counts=(2, 2, 0), exact=True),
    scored("5-6", "5-6", None, counts=(2, 2, 0), exact=True, surface=True),
    scored("5", "-5", None, counts=(1, 0, 1), exact=False),
    scored("12", "012", None, counts=(1, 1, 0), exact=True),
    # One value written twice, once with a unit: a text is its own best translation.
    unit_score(f"5 and 5 {T}", f"5 and 5 {T}", counts=(2, 2, 0), exact=True),
    unit_score(f"5 {T} and 5", f"5 {T} and 5", counts=(2, 2, 0), exact=True),
    unit_score(f"{T}5 and 5", f"{T}5 and 5", counts=(2, 2, 0), exact=True),
    unit_score(f"100 {T} and 100", f"100 {T} and 100", counts=(2, 2, 0), exact=True),
    unit_score(f"5 and 5 {T}", f"5 {T} and 5", counts=(2, 2, 0), exact=True),
    unit_score("5 and 5", f"5 {T} and 5", counts=(2, 2, 0), exact=True),
    unit_score(f"5 {T} and 5 $", f"5 $ and 5 {T}", counts=(2, 2, 0), exact=True),
    # A unit the reference states and no hypothesis number carries fails the row.
    unit_score(f"5 and 5 {T}", "5 and 5", counts=(2, 2, 0), exact=False, units=False),
    unit_score(
        f"5 {T} and 5 {T}", f"5 {T} and 5", counts=(2, 2, 0), exact=False, units=False
    ),
    unit_score(
        f"5 {T} and 5 $", f"5 {T} and 5 {T}", counts=(2, 2, 0), exact=False, units=False
    ),
    unit_score(f"5 {T} and 5", "5 and 5 $", counts=(2, 2, 0), exact=False, units=False),
    unit_score(
        f"5 {T}", "5 and 5 $", counts=(1, 1, 1), exact=False, units=False, same=False
    ),
    # A spare number takes the unit it needs, and a missing number is not a unit failure.
    unit_score(f"5 {T}", f"5 and 5 {T}", counts=(1, 1, 1), exact=False, same=False),
    unit_score(f"5 {T} and 5", "5 and 7", counts=(2, 1, 1), exact=False, same=False),
    unit_score(f"5 {T} and 5", f"5 {T}", counts=(2, 1, 0), exact=False, same=False),
    # A symbol between two numbers is the first one's: the reordering keeps every unit.
    unit_score(
        f"859 571{T} 3 ширхэг",
        f"3 ширхэг 859 571{T}",
        counts=(2, 2, 0),
        exact=True,
        lang="khk_Cyrl",
    ),
    unit_score(
        f"3 ширхэг 859 571{T}",
        f"859 571{T} 3 ширхэг",
        counts=(2, 2, 0),
        exact=True,
        lang="khk_Cyrl",
    ),
    # A symbol glued to the next number belongs to it: reordering such text keeps units.
    unit_score("$5 $7", "$7 $5", counts=(2, 2, 0), exact=True),
    unit_score("$5 $7 $9", "$9 $7 $5", counts=(3, 3, 0), exact=True),
    unit_score(f"{T}5 {T}7", f"{T}7 {T}5", counts=(2, 2, 0), exact=True),
    unit_score(f"5{T} 7{T}", f"{T}5 {T}7", counts=(2, 2, 0), exact=True),
    unit_score(f"{T}5 {T}7", f"5{T} 7{T}", counts=(2, 2, 0), exact=True),
    unit_score("5$ 7", "7 $5", counts=(2, 2, 0), exact=True),
    unit_score("$5, $7", "$7, $5", counts=(2, 2, 0), exact=True),
    unit_score("$5 and $7", "$7 and $5", counts=(2, 2, 0), exact=True),
    # The symbol did not move with its number: `5 $7` is 5 and 7 $, `7 $5` is 7 and 5 $.
    unit_score("5 $7", "7 $5", counts=(2, 2, 0), exact=False, units=False),
    unit_score("7 $5", "5 $7", counts=(2, 2, 0), exact=False, units=False),
    unit_score("5 $7", "5 $7", counts=(2, 2, 0), exact=True),
]


@pytest.mark.parametrize(("reference", "hypothesis", "lang", "expected"), SCORES)
def test_score_numbers(
    reference: str,
    hypothesis: str,
    lang: str | None,
    expected: tuple[int, int, int, bool, bool, bool, bool, bool],
) -> None:
    outcome = score_numbers(reference, hypothesis, tgt_lang=lang)

    assert shape(outcome) == expected


def test_a_matching_that_fits_every_reference_token_is_found_where_a_greedy_one_fails() -> (
    None
):
    # The first reference token reads 1.25 or 1250 and takes the first hypothesis token
    # that shares a reading; the second can only read 1.25 and needs that very token.
    outcome = score_numbers("1,250 and 1.25", "1,250 and 1 250", tgt_lang="khk_Cyrl")

    assert (outcome.found, outcome.extra, outcome.exact) == (2, 0, True)


def test_a_reference_of_one_hundred_numbers_is_cut_at_sixty_four_and_flagged() -> None:
    text = " ".join(["1"] * 100)

    outcome = score_numbers(text, text, tgt_lang=None)

    assert (outcome.expected, outcome.found, outcome.extra) == (64, 64, 1)
    assert outcome.capped
    assert not outcome.exact
    assert seconds(lambda: score_numbers(text, text, tgt_lang=None)) < 0.1


def test_a_number_the_hypothesis_lacks_is_not_a_match_of_its_neighbour() -> None:
    outcome = score_numbers("5 and 7", "7", tgt_lang=None)

    assert (outcome.expected, outcome.found, outcome.extra) == (2, 1, 0)
    assert not outcome.exact


def test_the_surface_of_a_signed_number_keeps_its_sign() -> None:
    assert not score_numbers("-5", "5", tgt_lang=None).surface_exact
    assert score_numbers("-5", f"{MINUS}5", tgt_lang=None).surface_exact


# --- numbers_agree ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "reference", "src_lang", "tgt_lang", "agree"),
    [
        ("3.5", "3,5", "eng_Latn", "khk_Cyrl", True),
        ("1,250", "1 250", "eng_Latn", "khk_Cyrl", True),
        ("about 3 million", "3 000 000", "eng_Latn", "rus_Cyrl", False),
        ("2 items", "два", "eng_Latn", "rus_Cyrl", False),
        ("a b", "c d", None, None, False),
        ("5 and 7", "7 and 5", "eng_Latn", "khk_Cyrl", True),
        ("5 and 7 and 9", "7 and 5", "eng_Latn", "khk_Cyrl", False),
        ("5", "5 and 7", "eng_Latn", "khk_Cyrl", False),
        ("5 and 8", "7 and 5", "eng_Latn", "khk_Cyrl", False),
        ("1,250", "1.25", "eng_Latn", "eng_Latn", False),
    ],
)
def test_numbers_agree(
    source: str, reference: str, src_lang: str | None, tgt_lang: str | None, agree: bool
) -> None:
    assert (
        numbers_agree(source, reference, src_lang=src_lang, tgt_lang=tgt_lang) is agree
    )


def test_a_reference_cut_at_the_cap_never_agrees() -> None:
    text = " ".join(["1"] * 100)

    assert not numbers_agree(text, text, src_lang=None, tgt_lang=None)
