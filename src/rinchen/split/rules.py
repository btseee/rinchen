"""Rule-based sentence splitting, one rule family per script.

`split` routes a language code to its rule family. A family finds the offsets
where a sentence ends; `_parts_between` turns those offsets into the stripped,
non-empty parts. A new family registers its splitter in `_SPLITTERS`.

The Latin and Cyrillic families (`_SpacedFamily`) end a sentence only before
whitespace and a sentence start; a lone "." after an abbreviation, an initial or
an ordinal number does not end one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
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
    characters, because the "!" counts as text that follows the "?".
    """
    return re.compile(
        rf"[{re.escape(terminators)}]++[{re.escape(_CLOSERS)}]*+(?=\s*\S)"
    )


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

# Entries are lowercase with the final dot removed, as `_is_suppressed` compares them.
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
_CYRILLIC_ABBREVIATIONS = frozenset(
    {
        # Russian, as used in Mongolian and Kazakh text too.
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
        # Mongolian.
        "\u0433.\u043c",  # and so on
        "\u0436.\u043d\u044c",  # for example
        "\u04e9.\u0445",  # in other words
        "\u043c.\u044d.\u04e9",  # before the common era
        # Kazakh.
        "\u0442.\u0431",  # and so on
        "\u0442.\u0441.\u0441",  # and the like
        "\u0431.\u0437.\u0434",  # before the common era
    }
)


def _spaced_boundary(sentence_start: str) -> regex.Pattern[str]:
    """Match a terminator run and its closers when whitespace and a start follow.

    The quantifiers are possessive so that a run is never matched in part, which
    would cut "?!" between its characters.
    """
    return regex.compile(
        rf"(?P<terminator>[{regex.escape(_SPACED_TERMINATORS)}]++)"
        rf"[{regex.escape(_CLOSERS)}]*+(?=\s+(?:{sentence_start}))",
        regex.V1,
    )


_LATIN_BOUNDARY = _spaced_boundary(_LATIN_START)
_CYRILLIC_BOUNDARY = _spaced_boundary(_CYRILLIC_START)
# A single uppercase letter before a "." is a name initial ("J. K. Rowling").
_INITIAL = regex.compile(r"\p{Lu}")


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
    `exempts_ordinals` name the tokens before it that keep it from ending a sentence.
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
_CYRILLIC = _SpacedFamily(
    _CYRILLIC_BOUNDARY, _CYRILLIC_ABBREVIATIONS, exempts_initials=True
)


def _token_before(text: str, end: int) -> str:
    """The whitespace-delimited token that ends at offset `end`."""
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end]


def _is_suppressed(token: str, family: _SpacedFamily) -> bool:
    """Whether a lone "." after `token` abbreviates or numbers instead of ending."""
    return (
        token.lower() in family.abbreviations
        or (family.exempts_initials and _INITIAL.fullmatch(token) is not None)
        or (family.exempts_ordinals and token.isdecimal())
    )


def _ends_sentence(match: regex.Match[str], text: str, family: _SpacedFamily) -> bool:
    if match["terminator"] != ".":
        return True
    return not _is_suppressed(_token_before(text, match.start()), family)


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
    "khk_Cyrl": partial(_split_spaced, _CYRILLIC),
    "rus_Cyrl": partial(_split_spaced, _CYRILLIC),
    "kaz_Cyrl": partial(_split_spaced, _CYRILLIC),
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
