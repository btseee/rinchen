"""Property tests for `rinchen.infer.doc_chunker` over random documents: whatever the
sentences (blank ones, long ones, any number of them) and whatever a window translator
does to the separators of its output, every sentence gets exactly one translation, in
order, and exactly the windows whose output does not line up fall back to the sentence
translator.

The piece counter is the number of whitespace-separated words; a window costs the two
fixed pieces plus the words of its joined text.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from hypothesis import given
from hypothesis import strategies as st

from rinchen.infer.doc_chunker import (
    PREFIX_PIECES,
    Chunk,
    chunk_document,
    join_window,
    translate_document,
)

_WORDS: Final = ("a", "b", "cc", "ddd", "e", "ff", "g", "hh", "i")
_BLANKS: Final = ("", " ", "\t", "  \n ", chr(0x2028), chr(0xA0), chr(0x3000))

# What the window translator does to the output of a window.
_MODES: Final = (
    "echo",
    "drop",
    "add",
    "merge_all",
    "empty_first",
    "empty_last",
    "none",
)


def words(text: str) -> int:
    return len(text.split())


@st.composite
def documents(draw: st.DrawFn) -> list[str]:
    """Up to 14 sentences of 1 to 9 words, some of them blank."""
    sentences: list[str] = []
    for _ in range(draw(st.integers(0, 14))):
        if draw(st.integers(0, 5)) == 0:
            sentences.append(draw(st.sampled_from(_BLANKS)))
        else:
            sentences.append(
                " ".join(
                    draw(st.lists(st.sampled_from(_WORDS), min_size=1, max_size=9))
                )
            )
    return sentences


budgets = st.integers(1, 40)
sizes = st.integers(1, 5)


def chunks(
    sentences: Sequence[str], max_sentences: int, max_pieces: int
) -> tuple[Chunk, ...]:
    return chunk_document(
        sentences,
        count_pieces=words,
        max_sentences=max_sentences,
        max_pieces=max_pieces,
    )


def blank(sentence: str) -> bool:
    return not sentence.strip()


def cost(sentences: Sequence[str]) -> int:
    return PREFIX_PIECES + words(join_window(list(sentences)))


@given(documents(), sizes, budgets)
def test_every_position_is_in_exactly_one_chunk(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    found = [
        position
        for chunk in chunks(sentences, max_sentences, max_pieces)
        for position in chunk.positions
    ]

    assert sorted(found) == list(range(len(sentences)))


@given(documents(), sizes, budgets)
def test_the_kinds_follow_the_blanks_the_budget_and_the_size_limit(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    for chunk in chunks(sentences, max_sentences, max_pieces):
        texts = [sentences[position] for position in chunk.positions]
        assert chunk.positions == tuple(sorted(chunk.positions))
        if chunk.kind == "empty":
            assert len(texts) == 1
            assert blank(texts[0])
            continue
        assert not any(blank(text) for text in texts)
        if chunk.kind == "oversize":
            assert len(texts) == 1
            assert cost(texts) > max_pieces
        elif chunk.kind == "single":
            assert len(texts) == 1
            assert cost(texts) <= max_pieces
        else:
            assert chunk.kind == "window"
            assert 2 <= len(texts) <= max_sentences
            assert cost(texts) <= max_pieces


@given(documents(), sizes, budgets)
def test_a_window_is_made_of_neighbours_and_the_chunks_come_in_order(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    found = chunks(sentences, max_sentences, max_pieces)

    for chunk in found:
        first = chunk.positions[0]
        assert chunk.positions == tuple(range(first, first + len(chunk.positions)))
    assert [chunk.positions[0] for chunk in found] == sorted(
        chunk.positions[0] for chunk in found
    )


@given(documents(), sizes, budgets)
def test_the_chunks_are_cut_greedily(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    """A window or single is followed by a sentence that could not have joined it: the
    window was full, the next sentence would have passed the budget, or a blank sentence
    came between them."""
    runs = [
        chunk
        for chunk in chunks(sentences, max_sentences, max_pieces)
        if chunk.kind in ("window", "single", "oversize")
    ]
    ordered = sorted(runs, key=lambda chunk: chunk.positions[0])
    for before, after in pairwise(ordered):
        if before.kind == "oversize":
            continue
        texts = [sentences[position] for position in before.positions]
        following = sentences[after.positions[0]]
        full = len(texts) == max_sentences
        between = sentences[before.positions[-1] + 1 : after.positions[0]]
        assert (
            full or any(map(blank, between)) or cost([*texts, following]) > max_pieces
        ), (before, after)


@given(documents(), sizes, budgets)
def test_chunking_is_deterministic(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    assert chunks(sentences, max_sentences, max_pieces) == chunks(
        sentences, max_sentences, max_pieces
    )


def mangled(text: str, mode: str) -> str:
    """The output a window translator of `mode` gives for the window `text`."""
    segments = text.split(" <sep> ")
    if mode == "echo":
        return text
    if mode == "drop":
        return " <sep> ".join(segments[:-1])
    if mode == "add":
        return text + " <sep> extra"
    if mode == "merge_all":
        return " ".join(segments)
    if mode == "empty_first":
        return " <sep> ".join(["", *segments[1:]])
    if mode == "empty_last":
        return " <sep> ".join([*segments[:-1], ""])
    return "no separator, any words"


@given(
    documents(),
    sizes,
    budgets,
    st.lists(st.sampled_from(_MODES), min_size=15, max_size=15),
)
def test_every_sentence_gets_one_translation_and_only_mangled_windows_fall_back(
    sentences: list[str], max_sentences: int, max_pieces: int, modes: list[str]
) -> None:
    window_calls: list[str] = []
    sentence_calls: list[str] = []

    def translate_window(text: str) -> str:
        mode = modes[len(window_calls)]
        window_calls.append(text)
        return mangled(text, mode)

    def translate_sentence(text: str) -> str:
        sentence_calls.append(text)
        return f"S[{text}]"

    result = translate_document(
        sentences,
        translate_window=translate_window,
        translate_sentence=translate_sentence,
        count_pieces=words,
        max_sentences=max_sentences,
        max_pieces=max_pieces,
    )

    assert len(result.translations) == len(sentences)
    plan = chunks(sentences, max_sentences, max_pieces)
    windows = [chunk for chunk in plan if chunk.kind == "window"]
    assert len(window_calls) == len(windows)
    expected_fallbacks: list[int] = []
    expected_sentence_calls: list[str] = []
    for chunk in plan:
        texts = [" ".join(sentences[position].split()) for position in chunk.positions]
        if chunk.kind == "empty":
            assert result.translations[chunk.positions[0]] == ""
        elif chunk.kind != "window":
            expected_sentence_calls.append(texts[0])
            assert result.translations[chunk.positions[0]] == f"S[{texts[0]}]"
    for index, chunk in enumerate(windows):
        texts = [" ".join(sentences[position].split()) for position in chunk.positions]
        assert window_calls[index] == " <sep> ".join(texts)
        # Only an echo keeps the window's own sentences; every other mode breaks the
        # count or leaves an empty segment.
        if modes[index] == "echo":
            assert [result.translations[p] for p in chunk.positions] == texts
        else:
            expected_fallbacks.append(index)
            assert [result.translations[p] for p in chunk.positions] == [
                f"S[{text}]" for text in texts
            ]
    # The sentence translator saw the fallbacks' sentences and the lone ones, window by
    # window in document order of the chunks; each exactly once.
    assert sorted(sentence_calls) == sorted(
        expected_sentence_calls
        + [
            " ".join(sentences[p].split())
            for index in expected_fallbacks
            for p in windows[index].positions
        ]
    )
    assert result.stats.fallbacks == tuple(expected_fallbacks)
    assert result.stats.windows == len(windows)
    assert result.stats.aligned == len(windows) - len(expected_fallbacks)
    assert result.stats.sentences == len(sentences)
    assert result.stats.chunks == len(plan)
    assert result.stats.empty == sum(1 for chunk in plan if chunk.kind == "empty")
    assert result.stats.oversize == sum(1 for chunk in plan if chunk.kind == "oversize")
    assert result.stats.singles == sum(1 for chunk in plan if chunk.kind == "single")


@given(documents(), sizes, budgets)
def test_an_echoing_window_translator_returns_the_sentences_themselves(
    sentences: list[str], max_sentences: int, max_pieces: int
) -> None:
    result = translate_document(
        sentences,
        translate_window=lambda text: text,
        translate_sentence=lambda text: text,
        count_pieces=words,
        max_sentences=max_sentences,
        max_pieces=max_pieces,
    )

    assert result.translations == tuple(" ".join(text.split()) for text in sentences)
    assert result.stats.fallbacks == ()
