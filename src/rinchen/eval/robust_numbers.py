"""The numbers metric of D12: what a number is, and whether a hypothesis kept the numbers
of a reference.

A number is read by its VALUE, whatever the convention that wrote it: `1,250.50`,
`1 250,50` and `1250.5` are the same number, `12.03.2024` and `2024-03-12` are the same
three numbers, and `7529` is not `7592` (D4's digit multiset cannot tell). One separator
followed by exactly three digits (`1,250`) is a decimal point or a grouping: the language
of the text decides when it is known, and an unknown or mixed language keeps both readings
in one token flagged `ambiguous`.

Limits, on purpose: number and magnitude words (`three`, `million`) are not read, `1e5` is
1 and 5, `.5` is 5, `10.30` is the decimal 10.3 and not a time, digits glued to letters
(`COVID-19`, `A4`) count, and an ordinal (`3-r`) is its digit."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cache
from itertools import chain, islice
from typing import Final

from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from rinchen.text.numerals import fold_numerals

MAX_NUMBER_TOKENS: Final = 64
"""A side is matched on its first 64 numbers; a text is read up to its 65th, which is the
sign that it holds more."""

POINT_DECIMAL_LANGUAGES: Final = frozenset(
    {"eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"}
)
COMMA_DECIMAL_LANGUAGES: Final = frozenset(
    {"rus_Cyrl", "deu_Latn", "tur_Latn", "kaz_Cyrl"}
)
GROUP_SPACES: Final = "".join(chr(code) for code in (0x20, 0xA0, 0x202F, 0x2009))
CURRENCY_SYMBOLS: Final = "".join(
    chr(code) for code in (0x20AE, 0x24, 0x20AC, 0xA3, 0xA5, 0x20BD, 0x20A9)
)
_MINUS_SIGNS: Final = "-" + chr(0x2212)

_SPACES = re.escape(GROUP_SPACES)
_CURRENCY = re.escape(CURRENCY_SYMBOLS)
# A run: digits in groups of three behind one to three, or a plain digit string, then any
# number of `. , : / -` each directly followed by digits.
_RUN = re.compile(
    rf"(?<![0-9])(?:[0-9]{{1,3}}(?:[{_SPACES}][0-9]{{3}})+|[0-9]+)"
    r"(?:[.,:/\-](?=[0-9])[0-9]+)*(?![0-9])"
)
_GROUPED_HEAD = re.compile(rf"([0-9]{{1,3}}(?:[{_SPACES}][0-9]{{3}})+)(.*)", re.DOTALL)
_DECIMAL_TAIL = re.compile(r"[.,][0-9]+")
_COMPONENT_BREAK = re.compile(r"[:/\-]")
_DECIMAL_MARK = re.compile(r"[.,]")
_DIGITS = re.compile(r"[0-9]+")
_GROUP_SPACE = re.compile(f"[{_SPACES}]")
# A symbol touching the digits, or one after a group space that is not glued to the next
# number's digits (that one is the next number's).
_UNIT_AFTER = re.compile(f"(?:([{_CURRENCY}])|[{_SPACES}]([{_CURRENCY}])(?![0-9]))")
_UNIT_BEFORE = re.compile(f"([{_CURRENCY}])[{_SPACES}]?\\Z")


@dataclass(frozen=True, slots=True)
class NumberToken:
    """A number of a text: the values it can mean (two for an ambiguous group), the
    currency symbol written next to it, the number as written (the components of a date, a
    time or a range share the run as written, separators included, so that a date in
    another layout has another surface), and whether its reading depends on a convention
    the language did not settle."""

    readings: frozenset[Decimal]
    unit: str | None
    surface: str
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class NumberOutcome:
    """How a hypothesis kept the numbers of a reference: the reference's numbers (at
    most 64), those matched one to one, the hypothesis numbers left over (those beyond
    the 64th included), and whether the units, the written form (`surface_exact`) and the
    whole row (`exact`) agree."""

    expected: int
    found: int
    extra: int
    units_ok: bool
    exact: bool
    surface_exact: bool
    ambiguous: bool
    capped: bool


def numbers_in(text: str, lang: str | None = None) -> tuple[NumberToken, ...]:
    """The numbers of `text` in written order, read under the conventions of `lang`
    (`None` when unknown). Numbered placeholders are not numbers. A currency symbol is the
    unit of one number: the symbol that touches a number's digits or follows it after a
    space and is not glued to the next number's digits is that number's, so `$5 $7` is
    `5 $` and `7 $`, `5 $7` is `5` and `7 $`, and `5 $ 7` is `5 $` and `7`. At most
    `MAX_NUMBER_TOKENS + 1` tokens are produced, and the scan stops there."""
    bare = _placeholder_pattern().sub(" ", fold_numerals(text))
    return tuple(islice(_scan(bare, lang), MAX_NUMBER_TOKENS + 1))


def score_numbers(
    reference: str, hypothesis: str, *, tgt_lang: str | None
) -> NumberOutcome:
    """Match the numbers of `hypothesis` to those of `reference` one to one by value (a
    maximum matching, so that no assignment is missed), both read as `tgt_lang`. Order
    never matters; a missing, extra or changed number fails the row (`exact`). A unit the
    reference states must be the hypothesis's (a reference without a unit accepts any):
    `units_ok` holds when some maximum matching gives every reference number a hypothesis
    number of its own unit, so a value written twice with different units is never failed
    by the pairing."""
    wanted = numbers_in(reference, tgt_lang)[:MAX_NUMBER_TOKENS]
    seen = numbers_in(hypothesis, tgt_lang)
    kept = seen[:MAX_NUMBER_TOKENS]
    pairs = _maximum_matching(wanted, kept)
    with_units = _maximum_matching(wanted, kept, _same_unit)
    units_ok = len(with_units) == len(pairs)
    return NumberOutcome(
        expected=len(wanted),
        found=len(pairs),
        extra=len(seen) - len(pairs),
        units_ok=units_ok,
        exact=len(pairs) == len(wanted) == len(seen) and units_ok,
        surface_exact=_surfaces(wanted) == _surfaces(seen),
        ambiguous=any(token.ambiguous for token in (*wanted, *kept)),
        capped=len(seen) > MAX_NUMBER_TOKENS,
    )


def numbers_agree(
    source: str, reference: str, *, src_lang: str | None, tgt_lang: str | None
) -> bool:
    """Whether the reference has numbers and the source carries the same ones, one for
    one (a number of the source may read as the reference's under either convention)."""
    in_source = numbers_in(source, src_lang)
    in_reference = numbers_in(reference, tgt_lang)
    if not in_reference or len(in_source) != len(in_reference):
        return False
    pairs = _maximum_matching(in_reference, in_source[:MAX_NUMBER_TOKENS])
    return len(pairs) == len(in_reference)


@cache
def _placeholder_pattern() -> re.Pattern[str]:
    """The hundred reserved placeholders and nothing that merely looks like one."""
    symbols = user_defined_symbols()[-PLACEHOLDER_COUNT:]
    return re.compile("|".join(re.escape(symbol) for symbol in symbols))


def _scan(text: str, lang: str | None) -> Iterator[NumberToken]:
    claimed_up_to = 0
    for run in _RUN.finditer(text):
        numbers = _run_numbers(run.group(), lang)
        first_two = list(islice(numbers, 2))
        if len(first_two) == 1:
            token, claimed_up_to = _with_sign_and_unit(
                first_two[0], text, run.start(), run.end(), claimed_up_to
            )
            yield token
        else:
            layout = _GROUP_SPACE.sub(" ", run.group())
            for token in chain(first_two, numbers):
                yield replace(token, surface=layout)


def _run_numbers(run: str, lang: str | None) -> Iterator[NumberToken]:
    """The numbers of one run: a space-grouped head is one number (a decimal when a
    `.` or `,` and digits close the run), the rest breaks at `:`, `/` and `-`."""
    rest = run
    grouped = _GROUPED_HEAD.fullmatch(run)
    if grouped:
        head, tail = grouped.groups()
        digits = _GROUP_SPACE.sub("", head)
        if _DECIMAL_TAIL.fullmatch(tail):
            yield _token(Decimal(f"{digits}.{tail[1:]}"), run)
            return
        yield _token(Decimal(digits), head)
        rest = tail[1:]
    for component in _COMPONENT_BREAK.split(rest):
        if component:
            yield from _component_numbers(component, lang)


def _component_numbers(component: str, lang: str | None) -> Iterator[NumberToken]:
    """One number when the digits and marks of `component` form one, else one number
    per digit group."""
    number = _one_number(component, lang)
    if number is not None:
        yield number
        return
    for digits in _DIGITS.findall(component):
        yield _token(Decimal(digits), digits)


def _one_number(component: str, lang: str | None) -> NumberToken | None:
    marks = _DECIMAL_MARK.findall(component)
    if not marks:
        return _token(Decimal(component), component)
    parts = _DECIMAL_MARK.split(component)
    if len(set(marks)) == 2:
        return _marks_of_two_kinds(component, marks, parts)
    if len(marks) > 1:
        return _repeated_grouping(component, parts)
    return _single_mark(component, parts, marks[0], lang)


def _marks_of_two_kinds(
    component: str, marks: list[str], parts: list[str]
) -> NumberToken | None:
    """`1,250.50` or `1.250,50`: the last mark is the decimal point when the other kind
    groups threes."""
    groups_with_the_other_mark = set(marks[:-1]) == set(marks) - {marks[-1]}
    if (
        groups_with_the_other_mark
        and 1 <= len(parts[0]) <= 3
        and all(len(part) == 3 for part in parts[1:-1])
    ):
        return _token(Decimal("".join(parts[:-1]) + "." + parts[-1]), component)
    return None


def _repeated_grouping(component: str, parts: list[str]) -> NumberToken | None:
    """`1.250.000` or `1,250,000`; `10.0.0.1` and `1,2,3` are not one number."""
    if 1 <= len(parts[0]) <= 3 and all(len(part) == 3 for part in parts[1:]):
        return _token(Decimal("".join(parts)), component)
    return None


def _single_mark(
    component: str, parts: list[str], mark: str, lang: str | None
) -> NumberToken:
    whole, fraction = parts
    decimal = Decimal(f"{whole}.{fraction}")
    could_group = len(fraction) == 3 and 1 <= len(whole) <= 3 and whole[0] != "0"
    if not could_group:
        return _token(decimal, component)
    grouped = Decimal(whole + fraction)
    if lang in POINT_DECIMAL_LANGUAGES:
        return _token(decimal if mark == "." else grouped, component)
    if lang in COMMA_DECIMAL_LANGUAGES:
        return _token(decimal if mark == "," else grouped, component)
    return NumberToken(frozenset({decimal, grouped}), None, component, True)


def _token(value: Decimal, surface: str) -> NumberToken:
    """A number read as one value; its group spaces are written as plain spaces."""
    return NumberToken(frozenset({value}), None, _GROUP_SPACE.sub(" ", surface), False)


def _with_sign_and_unit(
    token: NumberToken, text: str, start: int, end: int, claimed_up_to: int
) -> tuple[NumberToken, int]:
    """The sign written before a lone number and the currency symbol beside it, and the
    end of the symbol this number took after itself (the next number may not claim it)."""
    surface = token.surface
    readings = token.readings
    if _is_negated(text, start):
        readings = frozenset(reading.copy_negate() for reading in readings)
        surface = "-" + surface
    unit, claimed_up_to = _currency_beside(text, start, end, claimed_up_to)
    return replace(token, readings=readings, surface=surface, unit=unit), claimed_up_to


def _is_negated(text: str, start: int) -> bool:
    """A minus sign right before the number, itself after the start, a space or `(`."""
    if start == 0 or text[start - 1] not in _MINUS_SIGNS:
        return False
    return start == 1 or text[start - 2].isspace() or text[start - 2] == "("


def _currency_beside(
    text: str, start: int, end: int, claimed_up_to: int
) -> tuple[str | None, int]:
    """The currency symbol after the number (touching its digits, or behind one group
    space and not glued to a digit: that one is the next number's), else the one before it,
    with at most one group space between. A symbol belongs to one number: a number before
    this one that took it after itself keeps it (`claimed_up_to` is the end of that
    symbol). Returns the unit and the end of the symbol this number took after itself (0
    when it took none)."""
    after = _UNIT_AFTER.match(text, end)
    if after:
        return after.group(1) or after.group(2), after.end()
    window = max(start - 2, 0)
    before = _UNIT_BEFORE.search(text[window:start])
    if before and window + before.start(1) >= claimed_up_to:
        return before.group(1), 0
    return None, 0


def _surfaces(tokens: Sequence[NumberToken]) -> list[str]:
    return sorted(token.surface for token in tokens)


def _same_unit(want: NumberToken, token: NumberToken) -> bool:
    """A reference number that states a unit needs that unit; one without accepts any."""
    return want.unit is None or token.unit == want.unit


def _maximum_matching(
    wanted: Sequence[NumberToken],
    seen: Sequence[NumberToken],
    compatible: Callable[[NumberToken, NumberToken], bool] | None = None,
) -> list[tuple[int, int]]:
    """Kuhn's algorithm on tokens that share a reading (and, when `compatible` is given,
    satisfy it): wanted tokens are tried in order, each with its candidates in order, so
    the pairs are the same on every run. Returns `(wanted index, seen index)` pairs."""
    candidates = [
        [
            index
            for index, token in enumerate(seen)
            if token.readings & want.readings
            and (compatible is None or compatible(want, token))
        ]
        for want in wanted
    ]
    owner = [-1] * len(seen)

    def place(want: int, visited: set[int]) -> bool:
        for index in candidates[want]:
            if index in visited:
                continue
            visited.add(index)
            if owner[index] == -1 or place(owner[index], visited):
                owner[index] = want
                return True
        return False

    for want in range(len(wanted)):
        place(want, set())
    return sorted((want, index) for index, want in enumerate(owner) if want != -1)
