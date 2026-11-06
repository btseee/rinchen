"""Rule-based sentence splitting, one rule family per script.

`split` routes a language code to its rule family. A family finds the offsets
where a sentence ends; `_parts_between` turns those offsets into the stripped,
non-empty parts. A new family registers its splitter in `_SPLITTERS`.

The Latin and Cyrillic families (`_SpacedFamily`) end a sentence only before
whitespace and a sentence start; a lone "." after an abbreviation (also one
written with spaces, "z. B."), an initial or an ordinal number does not end one.
The Cyrillic languages share one boundary rule and a base of Russian-origin
abbreviations; Mongolian and Kazakh each add entries of their own, so each
language has its own family and an entry listed for one does not apply to the
others.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

import regex

from rinchen.errors import ConfigError

# Quotes and brackets that close a sentence; they stay with the sentence they close.
_CLOSERS = "\"'\u00bb\u201d\u2019)]\uff09\u300d\u300f"

# U+1803 MONGOLIAN FULL STOP and U+1809 MONGOLIAN MANCHU FULL STOP.
# U+1802 MONGOLIAN COMMA is absent on purpose: a comma never ends a sentence.
_MN_MONG_TERMINATORS = "\u1803\u1809?!"
# U+3002 IDEOGRAPHIC FULL STOP, U+FF01 FULLWIDTH EXCLAMATION MARK,
# U+FF1F FULLWIDTH QUESTION MARK.
_CJK_TERMINATORS = "\u3002\uff01\uff1f"


def _boundary_after(terminators: str) -> re.Pattern[str]:
    """Match a run of `terminators` plus closers, only when more text follows.

    The quantifiers are possessive so that a run is never matched in part: at
    the end of the text, backtracking would otherwise cut "?!" between its
    characters, because the "!" counts as text that follows the "?". The
    lookbehind lets a match start only at the first terminator of a run: a
    failed attempt there fails at every later offset of the run too, and
    retrying them would take quadratic time on a long run.
    """
    run = re.escape(terminators)
    return re.compile(rf"(?<![{run}])[{run}]++[{re.escape(_CLOSERS)}]*+(?=\s*\S)")


_MN_MONG_BOUNDARY = _boundary_after(_MN_MONG_TERMINATORS)
_CJK_BOUNDARY = _boundary_after(_CJK_TERMINATORS)

# Latin and Cyrillic terminators: U+2026 HORIZONTAL ELLIPSIS joins the ASCII ones.
_SPACED_TERMINATORS = ".!?\u2026"
# Besides an uppercase letter, Cyrillic dialogue opens a sentence with a quote
# (" ' and U+00AB) or a dash (U+2014 em dash, U+2013 en dash, hyphen-minus).
_CYRILLIC_OPENERS = "\"'\u00ab\u2014\u2013-"

# What may start a sentence, as regex source for a lookahead. Hangul has no case,
# so the Latin family accepts any letter.
_LATIN_START = r"\p{L}"
_CYRILLIC_START = (
    rf"[\p{{Lu}}&&\p{{Script=Cyrillic}}]|[{regex.escape(_CYRILLIC_OPENERS)}]"
)

# Entries are lowercase and compact, as `_is_known_abbreviation` compares them: inner
# dots kept, no whitespace, final dot removed ("z.b" also matches "z. B.").
_ENGLISH_ABBREVIATIONS = frozenset(
    {
        # Titles: Mr. Mrs. Ms. Dr. Prof. St. Capt. Col. Gen. Gov. Lt. Mt. Rev. Sen. Sgt.
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "capt",
        "col",
        "gen",
        "gov",
        "lt",
        "mt",
        "rev",
        "sen",
        "sgt",
        # References and listings: vs. etc. e.g. i.e. cf. viz. fig. vol. inc.
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "viz",
        "fig",
        "vol",
        "inc",
    }
)
_GERMAN_ABBREVIATIONS = frozenset(
    {
        # z.B. d.h. u.a. u.v.m. i.d.R. s.o. s.u. z.T. bzw. usw. etc. ggf. evtl. bspw. inkl.
        "z.b",
        "d.h",
        "u.a",
        "u.v.m",
        "i.d.r",
        "s.o",
        "s.u",
        "z.t",
        "bzw",
        "usw",
        "etc",
        "ggf",
        "evtl",
        "bspw",
        "inkl",
        # Dr. Prof. Nr. Str. ca. vgl. hrsg. ebd. Abb. Mio. Mrd.
        "dr",
        "prof",
        "nr",
        "str",
        "ca",
        "vgl",
        "hrsg",
        "ebd",
        "abb",
        "mio",
        "mrd",
    }
)
# The Cyrillic entries are code-point escapes to keep this file ASCII; each comment
# glosses its entry.
_CYRILLIC_COMMON_ABBREVIATIONS = frozenset(
    {
        # Russian in origin, and used in Mongolian and Kazakh text as well.
        "\u0442.\u0434",  # and so on
        "\u0442.\u0435",  # that is
        "\u0442.\u043f",  # and the like
        "\u0442.\u043a",  # because
        "\u0442.\u043d",  # so-called
        "\u043d.\u044d",  # of the common era
        "\u043d\u0430\u043f\u0440",  # for example
        "\u0434\u0440",  # others
        "\u0441\u043c",  # see
        "\u0443\u043b",  # street
        "\u043f\u0440",  # avenue
        "\u043f\u0435\u0440",  # lane
        "\u043f\u043b",  # square
        "\u043e\u0431\u043b",  # region
        "\u0433",  # year or city
        "\u0433\u0433",  # years
        "\u0438\u043c",  # named after
        "\u043f\u0440\u043e\u0444",  # professor
        "\u0430\u043a\u0430\u0434",  # academician
        "\u0434\u043e\u0446",  # associate professor
        "\u0442\u044b\u0441",  # thousand
        "\u043c\u043b\u043d",  # million
        "\u043c\u043b\u0440\u0434",  # billion
        "\u0440\u0443\u0431",  # rouble
        "\u043a\u043e\u043f",  # kopeck
        "\u0440\u0438\u0441",  # figure
        "\u0441\u0442\u0440",  # page
    }
)
_MONGOLIAN_ABBREVIATIONS = frozenset(
    {
        "\u0433.\u043c",  # and so on
        "\u0436.\u043d\u044c",  # for example
        "\u04e9.\u0445",  # in other words
        "\u043c.\u044d.\u04e9",  # before the common era
    }
)
_KAZAKH_ABBREVIATIONS = frozenset(
    {
        # From the official Kazakh punctuation rules: the dotted forms of
        # "and so on" and "and the like", and the units year, century, second,
        # minute and hour.
        "\u0442.\u0431",  # and so on
        "\u0442.\u0441",  # and the like
        "\u0442.\u0442",  # and so on
        "\u0436",  # year
        "\u0493",  # century
        "\u0441\u0435\u043a",  # second
        "\u043c\u0438\u043d",  # minute
        "\u0441\u0430\u0493",  # hour
        # Found in a Kazakh abbreviation list, not in the official text read.
        "\u0442.\u0441.\u0441",  # and the like
        "\u0431.\u0437.\u0434",  # before the common era
    }
)


def _spaced_boundary(sentence_start: str) -> regex.Pattern[str]:
    """Match a terminator run and its closers when whitespace and a start follow.

    The quantifiers are possessive so that a run is never matched in part, which
    would cut "?!" between its characters. The lookbehind lets a match start only
    at the first terminator of a run, as in `_boundary_after`.
    """
    run = regex.escape(_SPACED_TERMINATORS)
    return regex.compile(
        rf"(?<![{run}])(?P<terminator>[{run}]++)"
        rf"[{regex.escape(_CLOSERS)}]*+(?=\s+(?:{sentence_start}))",
        regex.V1,
    )


_LATIN_BOUNDARY = _spaced_boundary(_LATIN_START)
_CYRILLIC_BOUNDARY = _spaced_boundary(_CYRILLIC_START)
# The token before a "." is a name initial when it is one uppercase letter or a dotted
# chain of them ("J.K", whose final dot is the "." under test).
_INITIALS = regex.compile(r"\p{Lu}(?:\.\p{Lu})*")

# An abbreviation may be written across this many whitespace-separated tokens
# ("i. d. R."); the bound also keeps the scan around each "." constant.
_MAX_ABBREVIATION_TOKENS = 4


def _parts_between(text: str, cut_offsets: Iterable[int]) -> list[str]:
    """Cut `text` at `cut_offsets`, which must be ascending, into stripped parts."""
    bounds = [0, *cut_offsets, len(text)]
    pieces = (text[start:end].strip() for start, end in pairwise(bounds))
    return [piece for piece in pieces if piece]


def _split_at_boundaries(boundary: re.Pattern[str], text: str) -> list[str]:
    return _parts_between(text, (match.end() for match in boundary.finditer(text)))


def _split_mn_mong(text: str) -> list[str]:
    return _split_at_boundaries(_MN_MONG_BOUNDARY, text)


def _split_cjk(text: str) -> list[str]:
    return _split_at_boundaries(_CJK_BOUNDARY, text)


@dataclass(frozen=True, slots=True)
class _SpacedFamily:
    """Rules for text whose sentences are separated by whitespace.

    A lone "." is ambiguous there. `abbreviations`, `exempts_initials` and
    `exempts_ordinals` name the tokens around it that keep it from ending a sentence.
    """

    boundary: regex.Pattern[str]
    abbreviations: frozenset[str] = frozenset()
    exempts_initials: bool = False
    exempts_ordinals: bool = False


_ENGLISH = _SpacedFamily(_LATIN_BOUNDARY, _ENGLISH_ABBREVIATIONS, exempts_initials=True)
_GERMAN = _SpacedFamily(
    _LATIN_BOUNDARY,
    _GERMAN_ABBREVIATIONS,
    exempts_initials=True,
    exempts_ordinals=True,
)
_PLAIN_LATIN = _SpacedFamily(_LATIN_BOUNDARY)
_KHK_CYRL = _SpacedFamily(
    _CYRILLIC_BOUNDARY,
    _CYRILLIC_COMMON_ABBREVIATIONS | _MONGOLIAN_ABBREVIATIONS,
    exempts_initials=True,
)
_RUS_CYRL = _SpacedFamily(
    _CYRILLIC_BOUNDARY, _CYRILLIC_COMMON_ABBREVIATIONS, exempts_initials=True
)
_KAZ_CYRL = _SpacedFamily(
    _CYRILLIC_BOUNDARY,
    _CYRILLIC_COMMON_ABBREVIATIONS | _KAZAKH_ABBREVIATIONS,
    exempts_initials=True,
)


def _token_start(text: str, end: int) -> int:
    """Offset where the whitespace-delimited token that ends at `end` begins."""
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return start


def _dotted_tokens_before(text: str, start: int, limit: int) -> list[str]:
    """Up to `limit` tokens ending in "." right before offset `start`, nearest first.

    The scan stops at the first token that does not end in a dot.
    """
    tokens: list[str] = []
    while len(tokens) < limit:
        end = start
        while end > 0 and text[end - 1].isspace():
            end -= 1
        if end == 0 or text[end - 1] != ".":
            break
        start = _token_start(text, end)
        tokens.append(text[start:end])
    return tokens


def _dotted_tokens_after(text: str, end: int, limit: int) -> list[str]:
    """Up to `limit` tokens ending in "." right after offset `end`, nearest first.

    The scan stops at the first token that does not end in a dot.
    """
    tokens: list[str] = []
    while len(tokens) < limit:
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        end = start
        while end < len(text) and not text[end].isspace():
            end += 1
        if end == start or text[end - 1] != ".":
            break
        tokens.append(text[start:end])
    return tokens


def _windows_around(
    before: list[str], current: str, after: list[str]
) -> Iterator[list[str]]:
    """Every run of up to `_MAX_ABBREVIATION_TOKENS` consecutive tokens with `current`.

    `before` lists the tokens left of `current`, nearest first; `after`, those right.
    """
    for count_before in range(len(before) + 1):
        room = _MAX_ABBREVIATION_TOKENS - 1 - count_before
        for count_after in range(min(len(after), room) + 1):
            yield [*reversed(before[:count_before]), current, *after[:count_after]]


def _is_known_abbreviation(text: str, dot: int, family: _SpacedFamily) -> bool:
    """Whether the lone "." at `dot` ends a listed abbreviation, however it is spaced.

    A window of consecutive dotted tokens around the dot is joined without
    whitespace and looked up, so "z. B." matches "z.b" from both of its dots.
    Windows reach past the dot's own token only when whitespace follows the dot.
    With a closer after it the token ends in the closer, the neighbouring dots
    would not see the same window, and the abbreviation could be cut apart.
    """
    if not family.abbreviations:
        return False
    start = _token_start(text, dot)
    current = text[start : dot + 1]
    has_neighbours = text[dot + 1 : dot + 2].isspace()
    room = _MAX_ABBREVIATION_TOKENS - 1
    before = _dotted_tokens_before(text, start, room) if has_neighbours else []
    after = _dotted_tokens_after(text, dot + 1, room) if has_neighbours else []
    return any(
        "".join(window)[:-1].lower() in family.abbreviations
        for window in _windows_around(before, current, after)
    )


def _is_suppressed(text: str, dot: int, family: _SpacedFamily) -> bool:
    """Whether the lone "." at `dot` abbreviates or numbers instead of ending."""
    token = text[_token_start(text, dot) : dot]
    return (
        (family.exempts_initials and _INITIALS.fullmatch(token) is not None)
        or (family.exempts_ordinals and token.isdecimal())
        or _is_known_abbreviation(text, dot, family)
    )


def _ends_sentence(match: regex.Match[str], text: str, family: _SpacedFamily) -> bool:
    if match["terminator"] != ".":
        return True
    return not _is_suppressed(text, match.start(), family)


def _split_spaced(family: _SpacedFamily, text: str) -> list[str]:
    return _parts_between(
        text,
        (
            match.end()
            for match in family.boundary.finditer(text)
            if _ends_sentence(match, text, family)
        ),
    )


_SPLITTERS: dict[str, Callable[[str], list[str]]] = {
    "mn_Mong": _split_mn_mong,
    "zho_Hans": _split_cjk,
    "jpn_Jpan": _split_cjk,
    "eng_Latn": partial(_split_spaced, _ENGLISH),
    "deu_Latn": partial(_split_spaced, _GERMAN),
    "tur_Latn": partial(_split_spaced, _PLAIN_LATIN),
    "kor_Hang": partial(_split_spaced, _PLAIN_LATIN),
    "khk_Cyrl": partial(_split_spaced, _KHK_CYRL),
    "rus_Cyrl": partial(_split_spaced, _RUS_CYRL),
    "kaz_Cyrl": partial(_split_spaced, _KAZ_CYRL),
}


def split(text: str, lang: str) -> list[str]:
    """Split `text` into sentences using the rules for `lang`.

    Parts are verbatim slices of `text`, in order, stripped and non-empty; only
    whitespace at the edges of a part is dropped. Empty or whitespace-only text
    gives `[]`. Raises ConfigError if `lang` has no splitting rules.
    """
    splitter = _SPLITTERS.get(lang)
    if splitter is None:
        raise ConfigError(
            f"no sentence-splitting rules for language {lang!r}; "
            f"expected one of {sorted(_SPLITTERS)}"
        )
    return splitter(text)
