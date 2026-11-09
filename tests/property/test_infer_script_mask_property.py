"""Property tests for `rinchen.infer.script_mask`: whatever vocabulary, target and `allow` a
caller gives, the suppression list holds only entries of that vocabulary, never the end token,
each once, in the order of the vocabulary, and the same list every time; the off mode and an
allowance only ever shorten it. The purity of a text is a share of its letters: it adds up
over the allowed and the forbidden scripts and weighs the parts of a concatenation by their
letters."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from rinchen.config.languages import load_languages
from rinchen.infer.script_mask import (
    allowed_scripts,
    forbidden_scripts,
    script_purity,
    suppression_list,
)
from rinchen.lang.scripts import Script, counted_letters
from tests.support.fake_ct2 import SPIECE, FakeProcessor

END = "</s>"
TARGETS = [entry.code for entry in load_languages()]
PLACEHOLDER = chr(0x27E6) + "5" + chr(0x27E7)
MONGOLIAN_PIECE = SPIECE + chr(0x1820) + chr(0x1821)
HAN_PIECE = SPIECE + chr(0x4E2D) + chr(0x6587)
HANGUL_PIECE = chr(0xD55C) + chr(0xAE00)
KANA_PIECE = SPIECE + chr(0x3072) + chr(0x3089) + chr(0x30AB)
GREEK_PIECE = SPIECE + chr(0x3B1) + chr(0x3B2)
# "ba" in Cyrillic, spelled by code point: the second letter looks like a Latin one.
CYRILLIC_PIECE = SPIECE + chr(0x431) + chr(0x430)

FOREIGN = ["<blank>", "averyunlikelytoken0", "<unk>"]
PIECES = [
    "<s>",
    END,
    "<2khk_Cyrl>",
    "<2eng_Latn>",
    "<2mn_Mong>",
    "<bt>",
    "<doc>",
    "<sep>",
    PLACEHOLDER,
    SPIECE,
    CYRILLIC_PIECE,
    SPIECE + "Th",
    "e",
    SPIECE + "Э" + "x",
    SPIECE + "123",
    GREEK_PIECE,
    MONGOLIAN_PIECE,
    HAN_PIECE,
    HANGUL_PIECE,
    KANA_PIECE,
    "<0x41>",
    "<0x7A>",
    "<0x30>",
    "<0xE0>",
]
POOL = [*FOREIGN, *PIECES]
# `<unk>` is the processor's own unknown piece: it is not given as a piece.
PROCESSOR = FakeProcessor(PIECES)

ALLOWANCES = [(), ("<sep>",), ("<doc>", "<sep>"), (PLACEHOLDER,)]
vocabularies = st.lists(st.sampled_from(POOL), max_size=40)
targets = st.sampled_from(TARGETS)
allowances = st.sampled_from(ALLOWANCES)


def flat(listed: list[list[str]]) -> list[str]:
    assert all(len(entry) == 1 for entry in listed)
    return [entry[0] for entry in listed]


@given(vocabularies, targets, allowances, st.booleans())
def test_the_list_is_a_deterministic_ordered_subset_of_the_vocabulary_without_the_end(
    vocabulary: list[str], tgt: str, allow: tuple[str, ...], scripts: bool
) -> None:
    listed = suppression_list(
        vocabulary, PROCESSOR, tgt, end_token=END, allow=allow, scripts=scripts
    )
    again = suppression_list(
        vocabulary, PROCESSOR, tgt, end_token=END, allow=allow, scripts=scripts
    )

    entries = flat(listed)
    assert listed == again
    assert set(entries) <= set(vocabulary)
    assert END not in entries
    assert len(entries) == len(set(entries))
    assert [vocabulary.index(entry) for entry in entries] == sorted(
        vocabulary.index(entry) for entry in entries
    )


@given(vocabularies, targets, allowances)
def test_the_off_mode_only_ever_shortens_the_list(
    vocabulary: list[str], tgt: str, allow: tuple[str, ...]
) -> None:
    on = flat(suppression_list(vocabulary, PROCESSOR, tgt, end_token=END, allow=allow))
    off = flat(
        suppression_list(
            vocabulary, PROCESSOR, tgt, end_token=END, allow=allow, scripts=False
        )
    )

    assert set(off) <= set(on)
    assert set(off) <= {"<blank>", "averyunlikelytoken0", "<unk>", "<s>"}


@given(vocabularies, targets)
def test_an_allowance_only_ever_removes_entries(
    vocabulary: list[str], tgt: str
) -> None:
    plain = flat(suppression_list(vocabulary, PROCESSOR, tgt, end_token=END))
    allowed = flat(
        suppression_list(
            vocabulary, PROCESSOR, tgt, end_token=END, allow=("<doc>", "<sep>")
        )
    )

    assert set(allowed) <= set(plain)
    assert set(plain) - set(allowed) <= {"<doc>", "<sep>"}


@given(vocabularies, targets)
def test_a_piece_of_a_forbidden_script_is_listed_and_one_of_an_allowed_script_is_not(
    vocabulary: list[str], tgt: str
) -> None:
    listed = set(flat(suppression_list(vocabulary, PROCESSOR, tgt, end_token=END)))
    forbidden = forbidden_scripts(tgt)
    pieces_by_script = {
        Script.CYRILLIC: CYRILLIC_PIECE,
        Script.LATIN: SPIECE + "Th",
        Script.MONGOLIAN: MONGOLIAN_PIECE,
        Script.HAN: HAN_PIECE,
        Script.HANGUL: HANGUL_PIECE,
        Script.KATAKANA: KANA_PIECE,
    }

    for script, piece in pieces_by_script.items():
        if piece in vocabulary:
            assert (piece in listed) == (script in forbidden), (tgt, script)


# -- the purity of a text

LETTERS = st.sampled_from(
    [
        chr(0x430),
        "Э",
        "ү",
        "a",
        "Z",
        chr(0x1820),
        chr(0x1821),
        chr(0x4E2D),
        chr(0xD55C),
        chr(0x3072),
        chr(0x30AB),
        chr(0x3B1),
    ]
)
NEUTRAL = st.sampled_from(
    [" ", "1", "7", ",", "!", "-", chr(0x180B), chr(0x30FC), "\n"]
)
texts = st.lists(st.one_of(LETTERS, NEUTRAL), max_size=30).map("".join)
script_sets = st.sets(st.sampled_from(list(Script)))


@given(texts, script_sets)
def test_a_purity_is_a_share(text: str, scripts: set[Script]) -> None:
    assert 0.0 <= script_purity(text, scripts) <= 1.0 + 1e-9


@given(texts, targets)
def test_the_allowed_and_the_forbidden_scripts_split_the_named_letters(
    text: str, tgt: str
) -> None:
    named = counted_letters(text) - sum(1 for char in text if char == chr(0x3B1))
    expected = 1.0 if counted_letters(text) == 0 else named / counted_letters(text)

    total = script_purity(text, allowed_scripts(tgt)) + script_purity(
        text, forbidden_scripts(tgt)
    )

    if counted_letters(text) == 0:
        assert total == 2.0
    else:
        assert math.isclose(total, expected, abs_tol=1e-9)


@given(texts, texts, script_sets)
def test_the_purity_of_a_concatenation_weighs_the_parts_by_their_letters(
    first: str, second: str, scripts: set[Script]
) -> None:
    whole = counted_letters(first + second)
    if whole == 0:
        assert script_purity(first + second, scripts) == 1.0
        return

    in_scripts = sum(
        script_purity(part, scripts) * counted_letters(part)
        for part in (first, second)
        if counted_letters(part)
    )

    assert math.isclose(
        script_purity(first + second, scripts) * whole, in_scripts, abs_tol=1e-9
    )
