"""Property tests for `rinchen.split.split` across its four rule families.

Five representative languages cover the families: `khk_Cyrl` (Cyrillic), `mn_Mong`
(traditional script), `zho_Hans` (CJK), `eng_Latn` and `deu_Latn` (Latin). German
is here because the spaced-abbreviation windows, the ordinal exemption and the
closer-after-dot guard only run in families that carry abbreviations.

Each language gets random characters from its own alphabet. The Latin and Cyrillic
languages also get whitespace-joined tokens (initials, abbreviations, ordinals),
which reach the abbreviation, initial and window paths that a random character
stream almost never builds. Texts pinned in `_PINNED_TEXTS` are replayed on every
property. The hypothesis profile is loaded in `tests/conftest.py`.

Mongolian code points and invisible characters are written as code-point escapes
so the exact input stays reviewable.
"""

from __future__ import annotations

import string
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from rinchen.split import split

NBSP = "\N{NO-BREAK SPACE}"
_SPACING = f" \n\t{NBSP}"
_STOPS = ".!?\u2026"  # ASCII stops and U+2026 HORIZONTAL ELLIPSIS
_CLOSERS = "\"')]\u00bb\u201d\u2019"
_OPENERS = "\"'([\u00ab\u201c\u2018"
_DASHES = "\u2014\u2013-"  # em dash, en dash, hyphen-minus
_SHARED_MARKS = string.digits + _SPACING + _STOPS + _CLOSERS + _OPENERS + _DASHES + ","

# U+1803 MONGOLIAN FULL STOP, U+1809 MONGOLIAN MANCHU FULL STOP, U+1802 MONGOLIAN
# COMMA (never ends a sentence) and U+202F NARROW NO-BREAK SPACE (Mongolian suffix
# space, which counts as whitespace).
_MN_MONG_MARKS = "\u1803\u1809\u1802\u202f"
# U+3002 IDEOGRAPHIC FULL STOP, U+FF01 and U+FF1F fullwidth "!" and "?", U+FF0C
# fullwidth comma, U+FF09/U+300D/U+300F closers, U+FF08/U+300C/U+300E openers.
_CJK_MARKS = "\u3002\uff01\uff1f\uff0c\uff09\u300d\u300f\uff08\u300c\u300e"
_MN_MONG_LETTERS = "".join(chr(code) for code in range(0x1820, 0x1830))

_MAX_CHARS = 120
_LONG_CHARS = 20  # lower bound of the longer texts, which hold several sentences
_MAX_TOKENS = 14
_TOKEN_SEPARATORS = (" ", " ", "  ", "\n", "\t", NBSP)
# ASCII spaces and controls plus NBSP, em space, narrow NBSP and ideographic space.
_WHITESPACE_ONLY = " \t\n\r\x0b\x0c\u00a0\u2003\u202f\u3000"


@dataclass(frozen=True, slots=True)
class _Family:
    """A language, the alphabet of its random texts and its token vocabulary.

    `runs` are abbreviations written as consecutive dotted tokens ("z. B."); a
    random token stream would seldom put their pieces side by side.
    """

    lang: str
    alphabet: str
    tokens: tuple[str, ...] = ()
    runs: tuple[tuple[str, ...], ...] = ()

    def texts(self) -> st.SearchStrategy[str]:
        characters = st.one_of(
            st.text(alphabet=self.alphabet, max_size=_MAX_CHARS),
            st.text(alphabet=self.alphabet, min_size=_LONG_CHARS, max_size=_MAX_CHARS),
        )
        if not self.tokens:
            return characters
        return st.one_of(characters, _token_texts(self.tokens, self.runs))


@st.composite
def _token_texts(
    draw: st.DrawFn, tokens: tuple[str, ...], runs: tuple[tuple[str, ...], ...]
) -> str:
    """Single tokens and runs in random order, each token followed by random space."""
    single_tokens = st.sampled_from(tokens).map(lambda token: (token,))
    units = draw(
        st.lists(st.one_of(single_tokens, st.sampled_from(runs)), max_size=_MAX_TOKENS)
    )
    pieces = [piece for unit in units for piece in unit]
    gaps = draw(
        st.lists(
            st.sampled_from(_TOKEN_SEPARATORS),
            min_size=len(pieces),
            max_size=len(pieces),
        )
    )
    return "".join(piece + gap for piece, gap in zip(pieces, gaps, strict=True))


# Tokens mix capitalised words (they can start a sentence), single capital initials,
# abbreviations whole and in the pieces of a spaced one, ordinals and sentence ends.
_KHK_CYRL_TOKENS = (
    "Тэр",
    "Би",
    "Энэ",
    "Баяр",
    "Сайн",
    "ирлээ",
    "дараа",
    "юм",
    "Б.",
    "Д.",
    "А.Б.",
    "т.",
    "д.",
    "г.",
    "ул.",
    "др.",
    "см.",
    "т.д.",
    "т.е.",
    "1.5",
    "3.",
    "д.)",
    "ирлээ.)",
    "Тийм.",
    "Явав.",
    "Хот.",
    "Үнэн?",
    "Явъя!",
    '"Явъя."',
    "\u2014",
)
_KHK_CYRL_RUNS = (
    ("т.", "д."),
    ("т.", "е."),
    ("т.", "п."),
    ("н.", "э."),
    ("т.", "д.", "Б."),
)
_ENG_LATN_TOKENS = (
    "the",
    "cat",
    "He",
    "Then",
    "Foo",
    "B.",
    "J.",
    "K.",
    "J.K.",
    "Dr.",
    "Mr.",
    "St.",
    "e.",
    "g.",
    "i.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "3.",
    "3.14",
    "Done.",
    "Really?",
    "Wait\u2026",
    "a.)",
    '"stop."',
)
_ENG_LATN_RUNS = (("e.", "g."), ("i.", "e."), ("J.", "K."), ("e.", "g.", "Foo"))
_DEU_LATN_TOKENS = (
    "das",
    "ist",
    "Der",
    "Test",
    "Foo",
    "B.",
    "J.",
    "R.",
    "z.",
    "u.",
    "a.",
    "d.",
    "h.",
    "i.",
    "o.",
    "Dr.",
    "bzw.",
    "usw.",
    "z.B.",
    "u.a.",
    "3.",
    "12.",
    "Ende.",
    "Wirklich?",
    "Wow!",
    "a.)",
    '"ja."',
)

_DEU_LATN_RUNS = (
    ("z.", "B."),
    ("u.", "a."),
    ("d.", "h."),
    ("i.", "d.", "R."),
    ("u.", "a.)", "o."),
    ("z.", "B.", "Foo"),
)

_FAMILIES = (
    _Family("khk_Cyrl", "аБдөүТгӨҮ" + _SHARED_MARKS, _KHK_CYRL_TOKENS, _KHK_CYRL_RUNS),
    _Family("mn_Mong", _MN_MONG_LETTERS + _MN_MONG_MARKS + _SHARED_MARKS),
    _Family("zho_Hans", "今天气很好你吗我是的" + _CJK_MARKS + _SHARED_MARKS),
    _Family(
        "eng_Latn", "abdeiorsDJKMX" + _SHARED_MARKS, _ENG_LATN_TOKENS, _ENG_LATN_RUNS
    ),
    _Family(
        "deu_Latn", "abdhiouzBDJRÄäß" + _SHARED_MARKS, _DEU_LATN_TOKENS, _DEU_LATN_RUNS
    ),
)

# Texts earlier reviews found unpinned, replayed on every property for their language.
_PINNED_TEXTS: dict[str, tuple[str, ...]] = {
    "deu_Latn": (
        # A closer after the dot hides it from the neighbouring dots' windows.
        "u. a.) o. Foo",
        "Das ist d. h. ein Test. Gut.",
    ),
    # The "т. д." window suppresses both dots: one part, no Latin letters in it.
    "khk_Cyrl": ("Мы купили хлеб, молоко и т. д. Потом ушли.",),
}

_EACH_FAMILY = pytest.mark.parametrize(
    "family", _FAMILIES, ids=lambda family: family.lang
)


def _holds_for_family_texts(family: _Family, holds: Callable[[str], None]) -> None:
    """Run `holds` on random texts of `family` and on its pinned texts."""
    runner = given(text=family.texts())(holds)
    for pinned in _PINNED_TEXTS.get(family.lang, ()):
        runner = example(text=pinned)(runner)
    runner()


@_EACH_FAMILY
def test_parts_reassemble_to_input_modulo_whitespace(family: _Family) -> None:
    def holds(text: str) -> None:
        parts = split(text, family.lang)
        assert "".join("".join(part.split()) for part in parts) == "".join(text.split())

    _holds_for_family_texts(family, holds)


@_EACH_FAMILY
def test_no_empty_or_unstripped_parts(family: _Family) -> None:
    def holds(text: str) -> None:
        for part in split(text, family.lang):
            assert part
            assert part == part.strip()

    _holds_for_family_texts(family, holds)


@_EACH_FAMILY
def test_each_part_is_a_fixed_point(family: _Family) -> None:
    def holds(text: str) -> None:
        for part in split(text, family.lang):
            assert split(part, family.lang) == [part]

    _holds_for_family_texts(family, holds)


@_EACH_FAMILY
def test_parts_are_verbatim_in_order_slices(family: _Family) -> None:
    def holds(text: str) -> None:
        position = 0
        for part in split(text, family.lang):
            position = text.index(part, position) + len(part)

    _holds_for_family_texts(family, holds)


@pytest.mark.parametrize("lang", [family.lang for family in _FAMILIES])
@given(text=st.text(alphabet=_WHITESPACE_ONLY, max_size=30))
def test_whitespace_only_text_gives_empty_list(lang: str, text: str) -> None:
    assert split(text, lang) == []
