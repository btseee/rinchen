"""Tests for `rinchen.infer.doc_chunker`: cutting a document into windows, joining a
window with `<sep>`, splitting a window's output and the sentence-level fallback of a
window whose output does not line up.

The piece counter is the number of whitespace-separated words, so every expectation can be
counted by hand: a window costs the two fixed pieces (the direction tag and `<doc>`) plus
the words of its joined text, `<sep>` being a word of its own. The sentences below have
these word counts: "a b" 2, "c d e" 3, "f" 1, "g h" 2, "i j k l" 4.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.infer import doc_chunker
from rinchen.infer.doc_chunker import (
    DEFAULT_MAX_SENTENCES,
    Chunk,
    ChunkStats,
    DocumentTranslation,
    chunk_document,
    join_window,
    split_output,
    translate_document,
)
from rinchen.train import doc_pack, docify

SEP = "<sep>"
SENTENCES = ("a b", "c d e", "f", "g h", "i j k l")


def words(text: str) -> int:
    return len(text.split())


def chunks_of(
    sentences: Sequence[str],
    *,
    max_sentences: int = 4,
    max_pieces: int = 100,
    count: Callable[[str], int] = words,
) -> list[tuple[str, tuple[int, ...]]]:
    """(kind, positions) of every chunk, in the order `chunk_document` gives them."""
    return [
        (chunk.kind, chunk.positions)
        for chunk in chunk_document(
            sentences,
            count_pieces=count,
            max_sentences=max_sentences,
            max_pieces=max_pieces,
        )
    ]


# --- the layering and the vocabulary shared with the training side ----------------------


def imported_modules(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def test_the_chunker_imports_nothing_from_training_or_the_command_line() -> None:
    imported = imported_modules(Path(doc_chunker.__file__))

    for module in imported:
        assert not module.startswith(("rinchen.train", "rinchen.cli")), module
    assert "rinchen.errors" in imported


def test_the_whole_infer_package_stays_free_of_training_and_the_command_line() -> None:
    files = sorted(Path(doc_chunker.__file__).parent.glob("*.py"))

    assert {path.name for path in files} == {"__init__.py", "doc_chunker.py"}
    for path in files:
        for module in imported_modules(path):
            assert not module.startswith(("rinchen.train", "rinchen.cli")), (
                path.name,
                module,
            )


def test_the_constants_are_the_ones_the_windows_are_trained_with() -> None:
    assert doc_chunker.DOC_MARKER == docify.DOC_MARKER == "<doc>"
    assert doc_chunker.SEGMENT_SEPARATOR == docify.SEGMENT_SEPARATOR == SEP
    assert doc_chunker.WINDOW_JOIN == doc_pack.WINDOW_JOIN == " <sep> "
    assert doc_chunker.PREFIX_PIECES == doc_pack.PREFIX_PIECES == 2


def test_the_default_window_size_is_the_largest_one_docify_cuts() -> None:
    assert DEFAULT_MAX_SENTENCES == 4


# --- chunk_document: greedy windows under a size and a piece budget ------------------------


def test_a_generous_budget_gives_a_window_of_the_largest_size_and_a_lone_sentence() -> (
    None
):
    # "a b <sep> c d e <sep> f <sep> g h" is 11 words, 13 pieces with the two fixed ones.
    assert chunks_of(SENTENCES) == [("window", (0, 1, 2, 3)), ("single", (4,))]


def test_a_window_is_cut_before_the_sentence_that_would_pass_the_budget() -> None:
    # (0,1) costs 2 + 6 = 8 <= 9, (0,1,2) 2 + 8 = 10 > 9; (2,3) 2 + 4 = 6, (2,3,4) 2 + 9 = 11.
    assert chunks_of(SENTENCES, max_pieces=9) == [
        ("window", (0, 1)),
        ("window", (2, 3)),
        ("single", (4,)),
    ]


def test_a_window_may_cost_exactly_the_budget() -> None:
    assert chunks_of(SENTENCES, max_pieces=8)[0] == ("window", (0, 1))
    assert chunks_of(SENTENCES, max_pieces=7)[0] == ("single", (0,))


def test_a_sentence_that_leaves_no_room_for_the_next_one_is_alone() -> None:
    # (1,2) costs 2 + 5 = 7 fits, (1,2,3) costs 2 + 8 = 10; (3,4) costs 2 + 7 = 9 > 7.
    assert chunks_of(SENTENCES, max_pieces=7) == [
        ("single", (0,)),
        ("window", (1, 2)),
        ("single", (3,)),
        ("single", (4,)),
    ]


def test_the_separators_and_the_two_fixed_pieces_count_toward_the_budget() -> None:
    sentences = ("a b", "c d")
    # "a b <sep> c d" is 5 words, 7 pieces with the tag and <doc>.
    assert chunks_of(sentences, max_pieces=7) == [("window", (0, 1))]
    assert chunks_of(sentences, max_pieces=6) == [("single", (0,)), ("single", (1,))]


def test_the_counter_is_asked_about_the_joined_text_of_the_window() -> None:
    asked: list[str] = []

    def counting(text: str) -> int:
        asked.append(text)
        return words(text)

    chunks_of(("a b", "c d"), max_pieces=7, count=counting)

    assert "a b <sep> c d" in asked
    assert "a b" in asked


def test_a_sentence_over_the_budget_alone_is_an_oversize_chunk_between_the_windows() -> (
    None
):
    sentences = ("a b", "c d", "x " * 12, "e f", "g h")

    # "x x ... x" is 12 words, 14 pieces alone: over 10.
    assert chunks_of(sentences, max_pieces=10) == [
        ("window", (0, 1)),
        ("oversize", (2,)),
        ("window", (3, 4)),
    ]


def test_a_document_of_oversize_sentences_only_has_no_window() -> None:
    assert chunks_of(("a b c", "d e f"), max_pieces=4) == [
        ("oversize", (0,)),
        ("oversize", (1,)),
    ]


def test_a_budget_of_one_makes_every_sentence_oversize() -> None:
    assert chunks_of(SENTENCES, max_pieces=1) == [
        ("oversize", (position,)) for position in range(5)
    ]


def test_one_sentence_per_window_makes_every_sentence_single() -> None:
    assert chunks_of(SENTENCES, max_sentences=1) == [
        ("single", (position,)) for position in range(5)
    ]


def test_two_sentences_per_window() -> None:
    assert chunks_of(SENTENCES, max_sentences=2) == [
        ("window", (0, 1)),
        ("window", (2, 3)),
        ("single", (4,)),
    ]


def test_the_window_size_limit_applies_before_the_budget() -> None:
    assert chunks_of(("a", "b", "c", "d", "e", "f", "g"), max_sentences=3) == [
        ("window", (0, 1, 2)),
        ("window", (3, 4, 5)),
        ("single", (6,)),
    ]


def test_a_one_sentence_document_is_a_single_chunk() -> None:
    assert chunks_of(("only one",)) == [("single", (0,))]


def test_an_empty_document_has_no_chunk() -> None:
    assert chunks_of(()) == []


def test_empty_and_blank_sentences_are_chunks_of_their_own_and_left_out_of_windows() -> (
    None
):
    blanks = (
        "",
        "  \t",
        chr(0x2028),
        chr(0x85),
        chr(0xA0),
        chr(0x3000),
        chr(0x1C),
        "\n",
    )
    sentences = ("a b", blanks[0], blanks[1], "c d e", *blanks[2:], "f")

    # The three sentences with text are one window around the blanks, which stay where
    # they were and each have a chunk of their own.
    assert chunks_of(sentences) == [
        ("window", (0, 3, 10)),
        *(("empty", (position,)) for position in (1, 2, 4, 5, 6, 7, 8, 9)),
    ]


def test_a_document_of_blanks_only_is_all_empty_chunks() -> None:
    assert chunks_of(("", "   ")) == [("empty", (0,)), ("empty", (1,))]


def test_the_chunks_are_immutable_values_in_order_of_their_first_sentence() -> None:
    chunks = chunk_document(
        ("a b", "", "c d"), count_pieces=words, max_sentences=4, max_pieces=100
    )

    assert chunks == (Chunk("window", (0, 2)), Chunk("empty", (1,)))
    with pytest.raises(AttributeError):
        chunks[0].kind = "single"  # type: ignore[misc]


def test_a_sentence_with_the_separator_is_refused_naming_its_position() -> None:
    with pytest.raises(DataError, match=r"sentence 2 .*<sep>"):
        chunk_document(
            ("fine", "also fine", "not <sep> fine"),
            count_pieces=words,
            max_sentences=4,
            max_pieces=100,
        )


def test_a_separator_glued_to_a_word_is_refused_too() -> None:
    with pytest.raises(DataError, match="sentence 0"):
        chunk_document(
            ("x<sep>y",), count_pieces=words, max_sentences=4, max_pieces=100
        )


def test_the_refusal_comes_before_the_counter_is_called() -> None:
    def never(text: str) -> int:
        raise AssertionError("the counter was called")

    with pytest.raises(DataError):
        chunk_document(
            ("a", "<sep>"), count_pieces=never, max_sentences=4, max_pieces=100
        )


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "4", None])
def test_max_sentences_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="max_sentences"):
        chunk_document(
            ("a",),
            count_pieces=words,
            max_sentences=bad,  # type: ignore[arg-type]
            max_pieces=10,
        )


@pytest.mark.parametrize("bad", [0, -5, True, 10.0, None])
def test_max_pieces_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="max_pieces"):
        chunk_document(
            ("a",),
            count_pieces=words,
            max_sentences=4,
            max_pieces=bad,  # type: ignore[arg-type]
        )


def test_a_bare_string_is_not_a_document() -> None:
    with pytest.raises(TypeError, match="sentences"):
        chunk_document("Hello", count_pieces=words, max_sentences=4, max_pieces=10)


def test_a_sentence_that_is_not_text_is_refused_naming_its_position() -> None:
    with pytest.raises(TypeError, match="sentence 1"):
        chunk_document(
            ("a", 3),  # type: ignore[arg-type]
            count_pieces=words,
            max_sentences=4,
            max_pieces=10,
        )


@pytest.mark.parametrize("bad", [-1, True, 2.5, "3", None])
def test_a_counter_that_does_not_return_a_count_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="count_pieces"):
        chunk_document(
            ("a",),
            count_pieces=lambda text: bad,  # type: ignore[arg-type,return-value]
            max_sentences=4,
            max_pieces=10,
        )


def test_a_counter_that_raises_is_not_swallowed() -> None:
    def failing(text: str) -> int:
        raise RuntimeError("no tokenizer")

    with pytest.raises(RuntimeError, match="no tokenizer"):
        chunk_document(("a",), count_pieces=failing, max_sentences=4, max_pieces=10)


# --- join_window ---------------------------------------------------------------------------


def test_a_window_is_its_sentences_joined_with_a_spaced_separator() -> None:
    assert join_window(["A.", "B.", "C."]) == "A. <sep> B. <sep> C."


def test_one_sentence_window_has_no_separator() -> None:
    assert join_window(["Only."]) == "Only."


def test_the_whitespace_inside_and_around_a_sentence_becomes_single_spaces() -> None:
    line_separator = chr(0x2028)
    assert (
        join_window([f"  a  b\n c{line_separator}d ", "\te f\r\n"])
        == "a b c d <sep> e f"
    )


def test_a_window_is_one_line_whatever_line_breaks_the_sentences_hold() -> None:
    breaks = "".join(chr(code) for code in (0x0A, 0x0D, 0x0B, 0x0C, 0x85, 0x2029))
    joined = join_window([f"a{breaks}b", f"c{breaks}d"])

    assert joined == "a b <sep> c d"
    assert len(joined.splitlines()) == 1


def test_a_window_without_sentences_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one sentence"):
        join_window([])


def test_a_separator_inside_a_sentence_is_refused_by_the_join_too() -> None:
    with pytest.raises(DataError, match="sentence 1"):
        join_window(["a", "b <sep> c"])


# --- split_output --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "segments"),
    [
        ("A. <sep> B.", 2, ["A.", "B."]),
        ("A.<sep> B", 2, ["A.", "B"]),
        ("A <sep> B <sep> C", 3, ["A", "B", "C"]),
        ("  A <sep> B  ", 2, ["A", "B"]),
        ("A\t<sep>\nB", 2, ["A", "B"]),
        ("A <sep> <sep> B", 3, ["A", "", "B"]),
        ("<sep> B", 2, ["", "B"]),
        ("A <sep>", 2, ["A", ""]),
        ("<sep>", 2, ["", ""]),
        ("A", 1, ["A"]),
        ("", 1, [""]),
        ("a b  c", 1, ["a b  c"]),
    ],
)
def test_the_output_is_split_at_every_separator_with_the_whitespace_around_it(
    text: str, expected: int, segments: list[str]
) -> None:
    assert split_output(text, expected) == segments


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A. <sep> B.", 3),
        ("A. <sep> B.", 1),
        ("A", 2),
        ("A <sep> <sep> B", 2),
        ("<sep>", 3),
        ("A <sep> B <sep> C <sep> D", 3),
    ],
)
def test_a_count_other_than_the_expected_one_gives_none(
    text: str, expected: int
) -> None:
    assert split_output(text, expected) is None


def test_the_split_follows_the_separator_count_plus_one() -> None:
    for count in range(1, 7):
        text = " <sep> ".join(f"s{n}" for n in range(count))
        assert split_output(text, count) == [f"s{n}" for n in range(count)]
        assert split_output(text, count + 1) is None


def test_other_unicode_whitespace_around_a_separator_is_dropped_too() -> None:
    padded = f"A{chr(0xA0)}{chr(0x3000)}<sep>{chr(0x2028)}B"

    assert split_output(padded, 2) == ["A", "B"]


def test_text_that_only_looks_like_the_separator_does_not_split() -> None:
    assert split_output("A <SEP> B < sep > C <se p>", 1) == [
        "A <SEP> B < sep > C <se p>"
    ]


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, None])
def test_the_expected_count_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="expected"):
        split_output("A", bad)  # type: ignore[arg-type]


# --- translate_document --------------------------------------------------------------------


class Log:
    """The calls of both callables in order, and what each returns."""

    def __init__(
        self,
        window: Callable[[str], str] = lambda text: text,
        sentence: Callable[[str], str] = lambda text: f"S[{text}]",
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._window = window
        self._sentence = sentence

    def translate_window(self, text: str) -> str:
        self.calls.append(("window", text))
        return self._window(text)

    def translate_sentence(self, text: str) -> str:
        self.calls.append(("sentence", text))
        return self._sentence(text)


def translate(
    sentences: Sequence[str],
    log: Log,
    *,
    max_sentences: int = 4,
    max_pieces: int = 100,
) -> DocumentTranslation:
    return translate_document(
        sentences,
        translate_window=log.translate_window,
        translate_sentence=log.translate_sentence,
        count_pieces=words,
        max_sentences=max_sentences,
        max_pieces=max_pieces,
    )


def test_a_callable_that_echoes_the_window_returns_every_sentence() -> None:
    log = Log()

    result = translate(SENTENCES, log)

    assert result.translations == SENTENCES[:4] + ("S[i j k l]",)
    assert log.calls == [
        ("window", "a b <sep> c d e <sep> f <sep> g h"),
        ("sentence", "i j k l"),
    ]
    assert result.stats == ChunkStats(
        sentences=5,
        chunks=2,
        windows=1,
        aligned=1,
        fallbacks=(),
        singles=1,
        oversize=0,
        empty=0,
    )


def test_the_translations_are_a_tuple_with_one_text_per_sentence() -> None:
    result = translate(SENTENCES, Log(), max_pieces=9)

    assert isinstance(result.translations, tuple)
    assert len(result.translations) == len(SENTENCES)


def test_a_window_that_merges_two_segments_falls_back_alone() -> None:
    def window(text: str) -> str:
        # The first window is translated well, the second loses a separator.
        return text if text == "a b <sep> c d e" else text.replace(" <sep> ", " ")

    log = Log(window=window)

    result = translate(SENTENCES, log, max_pieces=9)

    assert result.translations == ("a b", "c d e", "S[f]", "S[g h]", "S[i j k l]")
    assert log.calls == [
        ("window", "a b <sep> c d e"),
        ("window", "f <sep> g h"),
        ("sentence", "f"),
        ("sentence", "g h"),
        ("sentence", "i j k l"),
    ]
    assert result.stats.windows == 2
    assert result.stats.aligned == 1
    assert result.stats.fallbacks == (1,)
    assert result.stats.singles == 1


def test_a_window_that_adds_a_segment_falls_back() -> None:
    log = Log(window=lambda text: text + " <sep> extra")

    result = translate(("a", "b"), log)

    assert result.translations == ("S[a]", "S[b]")
    assert result.stats.fallbacks == (0,)
    assert result.stats.aligned == 0


def test_a_window_that_drops_every_separator_falls_back() -> None:
    log = Log(window=lambda text: "no separators at all")

    result = translate(("a", "b", "c"), log)

    assert result.translations == ("S[a]", "S[b]", "S[c]")


@pytest.mark.parametrize("output", ["X <sep>", "<sep> X", "<sep>", "X <sep> <sep> Y"])
def test_an_empty_segment_for_a_sentence_that_has_text_falls_back(output: str) -> None:
    expected = 3 if output == "X <sep> <sep> Y" else 2
    sentences = ("a", "b", "c")[:expected]
    log = Log(window=lambda text: output)

    result = translate(sentences, log)

    assert result.translations == tuple(f"S[{s}]" for s in sentences)
    assert result.stats.fallbacks == (0,)


def test_the_fallback_translates_each_sentence_of_the_window_once_in_order() -> None:
    log = Log(window=lambda text: "broken")

    translate(("a b", "c", "d e"), log)

    assert log.calls == [
        ("window", "a b <sep> c <sep> d e"),
        ("sentence", "a b"),
        ("sentence", "c"),
        ("sentence", "d e"),
    ]


def test_the_fallbacks_are_numbered_among_the_windows_of_the_document() -> None:
    sentences = ("a", "b", "c", "d", "e", "f", "g")

    def window(text: str) -> str:
        # Windows 0 and 2 come back broken, window 1 is fine.
        return "broken" if text in ("a <sep> b", "e <sep> f") else text

    result = translate(sentences, Log(window=window), max_sentences=2)

    assert result.stats.windows == 3
    assert result.stats.aligned == 1
    assert result.stats.fallbacks == (0, 2)
    assert result.translations == ("S[a]", "S[b]", "c", "d", "S[e]", "S[f]", "S[g]")


def test_an_error_of_the_window_translator_is_not_swallowed() -> None:
    def failing(text: str) -> str:
        raise RuntimeError("the model is gone")

    with pytest.raises(RuntimeError, match="the model is gone"):
        translate(("a", "b"), Log(window=failing))


def test_a_domain_error_of_the_window_translator_is_not_swallowed_either() -> None:
    def failing(text: str) -> str:
        raise DataError("bad window")

    with pytest.raises(DataError, match="bad window"):
        translate(("a", "b"), Log(window=failing))


def test_an_error_of_the_sentence_translator_is_not_swallowed_on_any_path() -> None:
    def failing(text: str) -> str:
        raise RuntimeError("sentence model is gone")

    one_sentence = ("a",)
    broken_window = Log(window=lambda text: "broken", sentence=failing)
    oversize = ("x " * 20,)
    for log, sentences in (
        (Log(sentence=failing), one_sentence),
        (broken_window, ("a", "b")),
        (Log(sentence=failing), oversize),
    ):
        with pytest.raises(RuntimeError, match="sentence model is gone"):
            translate(sentences, log, max_pieces=10)


def test_a_window_output_that_is_not_text_is_refused() -> None:
    log = Log(window=lambda text: None)  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="translate_window"):
        translate(("a", "b"), log)


def test_a_sentence_translation_that_is_not_text_is_refused() -> None:
    log = Log(sentence=lambda text: 3)  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="translate_sentence"):
        translate(("a",), log)


def test_one_translation_per_sentence_with_empty_sentences_passed_through() -> None:
    log = Log()

    result = translate(("a b", "", "c d e", "  ", "f"), log)

    assert result.translations == ("a b", "", "c d e", "", "f")
    assert log.calls == [("window", "a b <sep> c d e <sep> f")]
    assert result.stats.empty == 2
    assert result.stats.sentences == 5


def test_a_document_of_blank_sentences_calls_nothing() -> None:
    log = Log()

    result = translate(("", " ", chr(0x2029)), log)

    assert result.translations == ("", "", "")
    assert log.calls == []
    assert result.stats == ChunkStats(
        sentences=3,
        chunks=3,
        windows=0,
        aligned=0,
        fallbacks=(),
        singles=0,
        oversize=0,
        empty=3,
    )


def test_an_empty_document_has_no_translation_and_no_call() -> None:
    log = Log()

    result = translate((), log)

    assert result.translations == ()
    assert log.calls == []
    assert result.stats == ChunkStats(
        sentences=0,
        chunks=0,
        windows=0,
        aligned=0,
        fallbacks=(),
        singles=0,
        oversize=0,
        empty=0,
    )


def test_a_one_sentence_document_takes_the_sentence_path_with_no_window_call() -> None:
    log = Log()

    result = translate(("Hello there",), log)

    assert result.translations == ("S[Hello there]",)
    assert log.calls == [("sentence", "Hello there")]
    assert result.stats.singles == 1
    assert result.stats.windows == 0


def test_a_lone_sentence_between_blanks_is_a_single_not_a_window() -> None:
    log = Log()

    result = translate(("", "only", ""), log)

    assert result.translations == ("", "S[only]", "")
    assert [kind for kind, _ in log.calls] == ["sentence"]


def test_an_oversize_sentence_goes_to_the_sentence_translator_and_is_counted() -> None:
    long = "x " * 12
    log = Log()

    result = translate(("a b", "c d", long, "e f"), log, max_pieces=10)

    assert result.translations[:2] == ("a b", "c d")
    assert result.translations[2] == f"S[{long.strip()}]"
    assert result.translations[3] == "S[e f]"
    assert result.stats.oversize == 1
    assert result.stats.windows == 1
    assert result.stats.singles == 1


def test_max_sentences_of_one_sends_every_sentence_to_the_sentence_translator() -> None:
    log = Log()

    result = translate(("a", "b", "c"), log, max_sentences=1)

    assert result.translations == ("S[a]", "S[b]", "S[c]")
    assert [kind for kind, _ in log.calls] == ["sentence"] * 3
    assert result.stats.singles == 3


def test_the_sentence_translator_gets_the_sentence_on_one_line() -> None:
    log = Log()

    translate(("only  one\nline",), log)

    assert log.calls == [("sentence", "only one line")]


def test_a_sentence_with_a_line_separator_inside_reaches_the_window_on_one_line() -> (
    None
):
    log = Log()

    translate((f"a{chr(0x2028)}b", "c"), log)

    assert log.calls == [("window", "a b <sep> c")]


def test_a_sentence_with_the_separator_is_refused_before_anything_is_translated() -> (
    None
):
    log = Log()

    with pytest.raises(DataError, match="sentence 1"):
        translate(("a b", "c <sep> d", "e"), log)

    assert log.calls == []


def test_a_bare_string_is_not_a_document_for_the_translation_either() -> None:
    with pytest.raises(TypeError, match="sentences"):
        translate("Hello", Log())  # type: ignore[arg-type]


def test_the_stats_account_for_every_chunk_and_sentence() -> None:
    sentences = ("a b", "", "c d e", "f", "x " * 12, "g h", "i j k l")

    result = translate(sentences, Log(), max_sentences=2, max_pieces=10)

    stats = result.stats
    assert stats.sentences == 7
    assert stats.chunks == stats.windows + stats.singles + stats.oversize + stats.empty
    assert (stats.windows, stats.singles, stats.oversize, stats.empty) == (2, 1, 1, 1)
    assert len(result.translations) == 7


def test_the_statistics_are_immutable_values() -> None:
    stats = translate(("a", "b"), Log()).stats

    with pytest.raises(AttributeError):
        stats.aligned = 5  # type: ignore[misc]
    assert stats.fallbacks == ()
