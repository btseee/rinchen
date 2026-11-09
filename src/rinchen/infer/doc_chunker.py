"""Translating a document in windows of consecutive sentences (D9, spec ruling T7), with
the sentence-level translation as the fallback of a window whose output does not line up.

A document-level model reads a window of 2 to 4 sentences joined with ` <sep> ` and
writes the translations joined the same way. `translate_document` cuts a document into
windows (`chunk_document`), joins each one (`join_window`), splits the model's output at
the separators (`split_output`) and, when a window comes back with another number of
segments than it had sentences, translates every sentence of that window on its own.
Nothing is repaired in part: the fallback is structural, so a window is either the
model's output, one segment per sentence, or the sentence-level translation of all its
sentences. The result has exactly one translation per input sentence, in order.

The module takes callables, not models: `count_pieces` (the pieces of a text, which the
caller's SentencePiece model counts), `translate_window` and `translate_sentence`. It
knows nothing of Eole or SentencePiece, and the training side is not imported: the
markers below repeat the ones `rinchen.train.docify` windows are trained with, and a test
holds the two in step.

How a document is cut:

- A sentence that is empty or only whitespace (Python's `str.strip`) is passed through
  with an empty translation, calls no translator, is left out of the windows and ends
  the window it is in, as `docify` ends a run at any sentence it leaves out: the
  sentences on either side of it are never in one window.
- A sentence is one line. Its whitespace, line breaks included, becomes single spaces
  (`one_line`), in the window and for the sentence-level translator, so a window never
  spans lines of a file Eole reads.
- The windows are greedy: each takes the next sentences while it holds at most
  `max_sentences` and its source costs at most `max_pieces` pieces. The source of a window
  is the direction tag and `<doc>`, two pieces that `PREFIX_PIECES` counts, and the
  joined text, which `count_pieces` counts as the model will see it, so what a `<sep>`
  costs is the counter's own (one for a word counter, two for SentencePiece, whose
  word-boundary piece stands in front of it).
- A sentence whose own window would be over the budget is `oversize` and goes to the
  sentence translator alone, and no window spans it; a sentence left alone by the windows
  around it is `single` and goes there too (a model trained on windows of at least two
  sentences is not asked for one).
- A sentence that holds the literal `<sep>` is refused for the whole document
  (`DataError`), before anything is translated: it would be read as a boundary.

A window falls back when its output has another number of segments than the window has
sentences, and also when it has the right number but a segment is empty: every sentence of
a window has text, so an empty segment is a sentence lost, which is never passed on
silently. What the callables raise is never caught.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from rinchen.errors import DataError
from rinchen.utils.integers import is_strict_int

__all__ = [
    "DEFAULT_MAX_SENTENCES",
    "DOC_MARKER",
    "PREFIX_PIECES",
    "SEGMENT_SEPARATOR",
    "WINDOW_JOIN",
    "Chunk",
    "ChunkKind",
    "ChunkStats",
    "DocumentTranslation",
    "chunk_document",
    "join_window",
    "one_line",
    "split_output",
    "translate_document",
]

# The same symbols and joint as `rinchen.train.docify` and `doc_pack`: the inference side
# must not import the training side, and `tests/unit/test_infer_doc_chunker.py` checks the
# two agree.
DOC_MARKER: Final = "<doc>"
SEGMENT_SEPARATOR: Final = "<sep>"
WINDOW_JOIN: Final = f" {SEGMENT_SEPARATOR} "
# The direction tag and `<doc>`, one piece each, in front of the source of every window.
PREFIX_PIECES: Final = 2
# The largest window `docify` cuts by default.
DEFAULT_MAX_SENTENCES: Final = 4

# What decoding gives for a `<sep>`: `A. <sep> B.` or, written bare, `A.<sep> B`.
_SEPARATOR_WITH_SPACE: Final = re.compile(rf"\s*{re.escape(SEGMENT_SEPARATOR)}\s*")

ChunkKind = Literal["window", "single", "oversize", "empty"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """The sentences of a document that are translated together: `positions` are their
    places in the document, ascending. A `window` has two or more, the other kinds one:
    `single` is a sentence the windows around it left alone, `oversize` one that does not
    fit the budget even alone, `empty` one with no text."""

    kind: ChunkKind
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChunkStats:
    """How a document was translated. `chunks` is `windows + singles + oversize + empty`.
    `windows` went to the window translator and `aligned` of them came back as one segment
    per sentence; `fallbacks` are the indices (from 0, in the order of the document) of
    the windows that did not and were translated sentence by sentence."""

    sentences: int
    chunks: int
    windows: int
    aligned: int
    fallbacks: tuple[int, ...]
    singles: int
    oversize: int
    empty: int


@dataclass(frozen=True, slots=True)
class DocumentTranslation:
    """One translation per input sentence, in order (an empty one for a sentence with no
    text), and what it took to get them."""

    translations: tuple[str, ...]
    stats: ChunkStats


def one_line(sentence: str) -> str:
    """`sentence` with every run of whitespace (what `str.split` splits at: line breaks,
    no-break and ideographic spaces included) as one space and none at either end."""
    return " ".join(sentence.split())


def join_window(sentences: Sequence[str]) -> str:
    """The source text of a window: its sentences as `one_line`, joined with ` <sep> `.

    Raises:
        ValueError: no sentence is given.
        DataError: a sentence holds the literal `<sep>`.
    """
    if not sentences:
        raise ValueError("a window needs at least one sentence")
    _refuse_separators(sentences)
    return WINDOW_JOIN.join(one_line(sentence) for sentence in sentences)


def split_output(text: str, expected: int) -> list[str] | None:
    """The segments of a window's output, split at every `<sep>` with the whitespace
    around it dropped, or None unless there are exactly `expected` of them. A segment may
    be empty (`A <sep> <sep> B`, a leading or a trailing `<sep>`): the count is the
    `<sep>` count plus one, and what an empty segment means is the caller's to decide.

    Raises:
        ValueError: `expected` is not a positive integer.
    """
    if not is_strict_int(expected) or expected < 1:
        raise ValueError(f"expected must be a positive integer, got {expected!r}")
    segments = _SEPARATOR_WITH_SPACE.split(text.strip())
    return segments if len(segments) == expected else None


def chunk_document(
    sentences: Sequence[str],
    *,
    count_pieces: Callable[[str], int],
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_pieces: int,
) -> tuple[Chunk, ...]:
    """Cut `sentences` into chunks (see the module's text): every position is in exactly
    one chunk, and the chunks are in the order of their first position.

    `count_pieces` gives the pieces of the joined text of a window (without the two
    fixed pieces, which are added here).

    Raises:
        ValueError: `max_sentences` or `max_pieces` is not a positive integer;
            `count_pieces` returns something that is not a count.
        TypeError: `sentences` is a single string or holds something that is not text.
        DataError: a sentence holds the literal `<sep>`.
    """
    _require_limits(max_sentences, max_pieces)
    _require_sentences(sentences)
    _refuse_separators(sentences)

    def fits(positions: Sequence[int]) -> bool:
        joined = join_window([sentences[position] for position in positions])
        return PREFIX_PIECES + _pieces(count_pieces, joined) <= max_pieces

    chunks: list[Chunk] = []
    current: list[int] = []

    def close() -> None:
        if current:
            chunks.append(
                Chunk("window" if len(current) > 1 else "single", tuple(current))
            )
            current.clear()

    for position, sentence in enumerate(sentences):
        if not sentence.strip():
            close()
            chunks.append(Chunk("empty", (position,)))
        elif not fits([position]):
            close()
            chunks.append(Chunk("oversize", (position,)))
        else:
            if current and (
                len(current) == max_sentences or not fits([*current, position])
            ):
                close()
            current.append(position)
    close()
    return tuple(chunks)


def translate_document(
    sentences: Sequence[str],
    *,
    translate_window: Callable[[str], str],
    translate_sentence: Callable[[str], str],
    count_pieces: Callable[[str], int],
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_pieces: int,
) -> DocumentTranslation:
    """Translate `sentences`, one translation each, in order (see the module's text).

    `translate_window` takes the joined text of a window (`join_window`) and returns the
    model's output for it, `<sep>` included; `translate_sentence` takes one sentence
    (`one_line`) and returns its translation. Chunks are translated in the order of their
    first sentence, and the sentences of a window that falls back in order. What the
    callables raise propagates.

    Raises:
        ValueError, TypeError, DataError: as `chunk_document`; also TypeError when a
            callable returns something that is not text.
    """
    chunks = chunk_document(
        sentences,
        count_pieces=count_pieces,
        max_sentences=max_sentences,
        max_pieces=max_pieces,
    )
    translations = [""] * len(sentences)
    fallbacks: list[int] = []
    windows = 0
    for chunk in chunks:
        if chunk.kind == "empty":
            continue
        texts = [one_line(sentences[position]) for position in chunk.positions]
        if chunk.kind == "window":
            segments = _window_segments(translate_window, texts)
            windows += 1
            if segments is not None:
                for position, segment in zip(chunk.positions, segments, strict=True):
                    translations[position] = segment
                continue
            fallbacks.append(windows - 1)
        for position, text in zip(chunk.positions, texts, strict=True):
            translations[position] = _text(
                translate_sentence, text, "translate_sentence"
            )
    return DocumentTranslation(
        translations=tuple(translations),
        stats=ChunkStats(
            sentences=len(sentences),
            chunks=len(chunks),
            windows=windows,
            aligned=windows - len(fallbacks),
            fallbacks=tuple(fallbacks),
            singles=_count(chunks, "single"),
            oversize=_count(chunks, "oversize"),
            empty=_count(chunks, "empty"),
        ),
    )


def _window_segments(
    translate_window: Callable[[str], str], texts: Sequence[str]
) -> list[str] | None:
    """The segments of the window of `texts` when its output has one non-empty segment
    per sentence, else None."""
    output = _text(translate_window, join_window(texts), "translate_window")
    segments = split_output(output, len(texts))
    if segments is None or not all(segments):
        return None
    return segments


def _text(translate: Callable[[str], str], text: str, name: str) -> str:
    translated = translate(text)
    if not isinstance(translated, str):
        raise TypeError(
            f"{name} must return text, got {type(translated).__name__}: {translated!r}"
        )
    return translated


def _count(chunks: Sequence[Chunk], kind: ChunkKind) -> int:
    return sum(1 for chunk in chunks if chunk.kind == kind)


def _pieces(count_pieces: Callable[[str], int], text: str) -> int:
    count = count_pieces(text)
    if not is_strict_int(count) or count < 0:
        raise ValueError(
            f"count_pieces must return a non-negative integer, got {count!r}"
        )
    return count


def _require_limits(max_sentences: object, max_pieces: object) -> None:
    for name, value in (("max_sentences", max_sentences), ("max_pieces", max_pieces)):
        if not is_strict_int(value) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_sentences(sentences: Sequence[object]) -> None:
    if isinstance(sentences, str):
        raise TypeError(
            "sentences must be a sequence of sentences, not one string: "
            f"{sentences[:30]!r}"
        )
    for position, sentence in enumerate(sentences):
        if not isinstance(sentence, str):
            raise TypeError(
                f"sentence {position} must be text, got {type(sentence).__name__}"
            )


def _refuse_separators(sentences: Sequence[str]) -> None:
    for position, sentence in enumerate(sentences):
        if SEGMENT_SEPARATOR in sentence:
            raise DataError(
                f"sentence {position} contains the literal {SEGMENT_SEPARATOR!r}, "
                "which a window would read as the boundary of two sentences: "
                f"{sentence[:60]!r}"
            )
