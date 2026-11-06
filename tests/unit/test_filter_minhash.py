"""The `minhash` stage: a row whose `src + TAB + tgt` is a near-duplicate of an
earlier surviving row is rejected as `duplicate`."""

from __future__ import annotations

import random

import polars as pl
import pytest

from rinchen.config.filters import MinHashConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.filter import minhash
from rinchen.filter.minhash import duplicate_flags
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus

DEFAULT_CONFIG = MinHashConfig()
LONG_TEXT = "the quick brown fox jumps over the lazy dog and keeps running far away"
OTHER_TEXT = "pack my box with five dozen liquor jugs before the sun goes down"

# The recall/precision gate: a seeded set of 200 base texts, 100 one-character
# edits of distinct bases and 100 unrelated decoys.
GATE_SEED = 7
BASE_COUNT = 200
NEAR_DUPLICATE_COUNT = 100
DECOY_COUNT = 100
TEXT_LENGTH = 120
LETTERS = "abcdefghijklmnopqrstuvwxyz"
MIN_PLANTED_JACCARD = 0.9  # clearly above the 0.8 threshold
MAX_DECOY_JACCARD = 0.5  # clearly below it
MIN_RECALL = 0.95
MIN_PRECISION = 0.95


def _row(src: str, tgt: str) -> dict[str, str]:
    return {
        "src_lang": "khk_Cyrl",
        "tgt_lang": "eng_Latn",
        "src": src,
        "tgt": tgt,
        "origin": "human",
        "corpus": "unit",
        "split": "train",
    }


def _frame(*pairs: tuple[str, str]) -> pl.DataFrame:
    return corpus_frame([_row(src, tgt) for src, tgt in pairs])


def _lazy(*pairs: tuple[str, str]) -> pl.LazyFrame:
    return _frame(*pairs).lazy()


def _with_reasons(df: pl.DataFrame, reasons: list[str | None]) -> pl.DataFrame:
    return df.with_columns(pl.Series("filter_reason", reasons, dtype=pl.String))


def _reasons(result: pl.DataFrame) -> list[str | None]:
    return result["filter_reason"].to_list()


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


# --- duplicate_flags ----------------------------------------------------------------


def test_an_exact_copy_is_flagged_and_the_first_is_kept() -> None:
    flags = duplicate_flags(
        [LONG_TEXT, OTHER_TEXT, LONG_TEXT], [True] * 3, DEFAULT_CONFIG
    )

    assert flags == [False, False, True]


def test_a_near_duplicate_is_flagged_by_row_order_not_by_content() -> None:
    edited = LONG_TEXT[:30] + "Q" + LONG_TEXT[31:]

    assert duplicate_flags([LONG_TEXT, edited], [True] * 2, DEFAULT_CONFIG) == [
        False,
        True,
    ]
    assert duplicate_flags([edited, LONG_TEXT], [True] * 2, DEFAULT_CONFIG) == [
        False,
        True,
    ]


def test_a_row_that_is_not_alive_is_never_flagged_and_never_shadows() -> None:
    flags = duplicate_flags(
        [LONG_TEXT, LONG_TEXT, LONG_TEXT], [False, True, True], DEFAULT_CONFIG
    )

    assert flags == [False, False, True]


def test_a_row_that_is_not_alive_is_not_flagged_even_as_a_copy_of_a_live_row() -> None:
    flags = duplicate_flags([LONG_TEXT, LONG_TEXT], [True, False], DEFAULT_CONFIG)

    assert flags == [False, False]


def test_no_rows_give_no_flags() -> None:
    assert duplicate_flags([], [], DEFAULT_CONFIG) == []


def test_flags_and_texts_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="length"):
        duplicate_flags([LONG_TEXT, OTHER_TEXT], [True], DEFAULT_CONFIG)


def test_two_empty_texts_are_exact_duplicates() -> None:
    assert duplicate_flags(["", ""], [True, True], DEFAULT_CONFIG) == [False, True]


def test_a_text_shorter_than_the_shingle_is_one_whole_shingle() -> None:
    cfg = MinHashConfig(ngram=500)

    flags = duplicate_flags([LONG_TEXT, OTHER_TEXT, LONG_TEXT, "x"], [True] * 4, cfg)

    assert flags == [False, False, True, False]


def test_a_text_of_exactly_the_shingle_length_is_one_shingle() -> None:
    cfg = MinHashConfig(ngram=5)

    assert duplicate_flags(["abcde", "abcde", "abcdf"], [True] * 3, cfg) == [
        False,
        True,
        False,
    ]


def test_the_same_input_gives_the_same_flags_every_time() -> None:
    texts = [LONG_TEXT, OTHER_TEXT, LONG_TEXT, OTHER_TEXT[:-1] + "!", "short"]
    alive = [True, True, True, False, True]

    first = duplicate_flags(texts, alive, DEFAULT_CONFIG)

    assert duplicate_flags(texts, alive, DEFAULT_CONFIG) == first


# --- apply --------------------------------------------------------------------------


def test_apply_flags_the_later_copy_as_duplicate_and_keeps_the_first() -> None:
    result = minhash.apply(
        _lazy(
            ("src one", LONG_TEXT), ("other src", OTHER_TEXT), ("src one", LONG_TEXT)
        ),
        DEFAULT_CONFIG,
    ).collect()

    assert _reasons(result) == [None, None, "duplicate"]


def test_apply_compares_both_sides_of_the_pair() -> None:
    # Same target and a different source is half the joined text, far below
    # the threshold; the identical pair is a copy.
    result = minhash.apply(
        _lazy(
            (LONG_TEXT, OTHER_TEXT),
            (OTHER_TEXT, OTHER_TEXT),
            (LONG_TEXT, OTHER_TEXT),
        ),
        DEFAULT_CONFIG,
    ).collect()

    assert _reasons(result) == [None, None, "duplicate"]


def test_apply_ignores_a_row_that_already_has_a_reason_when_looking_for_copies() -> (
    None
):
    df = _with_reasons(
        _frame(("s", LONG_TEXT), ("s", LONG_TEXT), ("s", LONG_TEXT)),
        ["lid_src", None, None],
    )

    result = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()

    # The rejected first row neither shadows the second nor loses its reason.
    assert _reasons(result) == ["lid_src", None, "duplicate"]


def test_apply_keeps_the_earlier_reason_of_a_rejected_copy() -> None:
    df = _with_reasons(_frame(("s", LONG_TEXT), ("s", LONG_TEXT)), [None, "numbers"])

    result = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()

    assert _reasons(result) == [None, "numbers"]


def test_apply_reads_a_null_side_as_empty_text() -> None:
    df = _frame(("", ""), ("x", "y"), ("", "")).with_columns(
        pl.when(pl.col("src") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("src"))
        .alias("src"),
        pl.when(pl.col("tgt") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("tgt"))
        .alias("tgt"),
    )
    assert df["src"].null_count() == 2

    result = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()

    assert _reasons(result) == [None, None, "duplicate"]


def test_apply_treats_two_empty_rows_as_exact_duplicates() -> None:
    result = minhash.apply(_lazy(("", ""), ("", "")), DEFAULT_CONFIG).collect()

    assert _reasons(result) == [None, "duplicate"]


def test_apply_on_an_empty_frame_returns_an_empty_frame_of_the_same_schema() -> None:
    result = minhash.apply(corpus_frame([]).lazy(), DEFAULT_CONFIG).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_apply_with_a_shingle_longer_than_every_text_does_not_crash() -> None:
    cfg = MinHashConfig(ngram=500)

    result = minhash.apply(
        _lazy(("ab", "cd"), ("ab", "cd"), ("", ""), ("ab", "ce")), cfg
    ).collect()

    assert _reasons(result) == [None, "duplicate", None, None]


def test_apply_keeps_the_corpus_schema_and_every_other_column() -> None:
    df = _frame(("a", LONG_TEXT), ("b", OTHER_TEXT), ("a", LONG_TEXT))

    result = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()

    assert result.schema == CORPUS_SCHEMA
    assert result.drop("filter_reason").equals(df.drop("filter_reason"))


def test_apply_gives_identical_output_on_two_runs() -> None:
    df = _frame(
        ("a", LONG_TEXT),
        ("b", OTHER_TEXT),
        ("a", LONG_TEXT),
        ("c", LONG_TEXT[:-1] + "!"),
        ("", ""),
        ("", ""),
    )

    first = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()
    second = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect()

    assert first.equals(second)


def test_apply_finds_a_copy_in_a_different_chunk_of_the_column() -> None:
    # The stage looks across rows, so it has to see the whole column and not
    # one chunk of it at a time.
    chunked = pl.concat(
        [_frame(("s", LONG_TEXT), ("s", OTHER_TEXT)), _frame(("s", LONG_TEXT))],
        rechunk=False,
    )
    assert chunked["src"].n_chunks() == 2

    result = minhash.apply(chunked.lazy(), DEFAULT_CONFIG).collect()

    assert _reasons(result) == [None, None, "duplicate"]


def test_apply_finds_copies_under_the_streaming_engine() -> None:
    df = _frame(("s", LONG_TEXT), ("s", OTHER_TEXT), ("s", LONG_TEXT))

    result = minhash.apply(df.lazy(), DEFAULT_CONFIG).collect(engine="streaming")

    assert _reasons(result) == [None, None, "duplicate"]


# --- the planted corpus -------------------------------------------------------------


def test_planted_corpus_flags_exactly_the_planted_duplicates(
    planted: PlantedCorpus,
) -> None:
    result = minhash.apply(planted.frame.lazy(), DEFAULT_CONFIG).collect()

    flagged = {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }

    planted_duplicates = {
        row_id
        for row_id, reason in planted.expected_reason.items()
        if reason == "duplicate"
    }
    assert set(flagged) == planted.isolated_failures["minhash"]
    assert set(flagged) == planted_duplicates
    assert set(flagged.values()) == {"duplicate"}
    assert len(flagged) == 3
    assert result.schema == CORPUS_SCHEMA


# --- recall and precision against the exact Jaccard ---------------------------------


def _pseudo_text(rng: random.Random) -> str:
    words: list[str] = []
    while len(" ".join(words)) < TEXT_LENGTH:
        words.append("".join(rng.choices(LETTERS, k=rng.randint(3, 8))))
    return " ".join(words)[:TEXT_LENGTH]


def _one_character_edit(rng: random.Random, text: str) -> str:
    position = rng.randrange(len(text))
    replacement = rng.choice([letter for letter in LETTERS if letter != text[position]])
    return text[:position] + replacement + text[position + 1 :]


def _split_pair(text: str) -> tuple[str, str]:
    half = len(text) // 2
    return text[:half], text[half:]


def _joined(pair: tuple[str, str]) -> str:
    return pair[0] + "\t" + pair[1]


def _shingles(text: str) -> set[str]:
    length = DEFAULT_CONFIG.ngram
    return {text[i : i + length] for i in range(len(text) - length + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def _gate_rows() -> tuple[list[tuple[str, str]], list[tuple[int, int]], list[int]]:
    """The shuffled (src, tgt) rows, the (base row, edited row) pairs and the
    decoy rows."""
    rng = random.Random(GATE_SEED)
    bases = [_pseudo_text(rng) for _ in range(BASE_COUNT)]
    edited_bases = rng.sample(range(BASE_COUNT), NEAR_DUPLICATE_COUNT)
    tagged = [
        *((f"base{index}", text) for index, text in enumerate(bases)),
        *(
            (f"edit{index}", _one_character_edit(rng, bases[index]))
            for index in edited_bases
        ),
        *((f"decoy{index}", _pseudo_text(rng)) for index in range(DECOY_COUNT)),
    ]
    rng.shuffle(tagged)
    row_of = {tag: row for row, (tag, _) in enumerate(tagged)}
    pairs = [(row_of[f"base{index}"], row_of[f"edit{index}"]) for index in edited_bases]
    decoys = [row_of[f"decoy{index}"] for index in range(DECOY_COUNT)]
    return [_split_pair(text) for _, text in tagged], pairs, decoys


def test_recall_and_precision_against_exact_shingle_jaccard() -> None:
    rows, edit_pairs, decoy_rows = _gate_rows()
    sets = [_shingles(_joined(pair)) for pair in rows]

    # The planted data sits well clear of the threshold on both sides.
    for base, edit in edit_pairs:
        assert _jaccard(sets[base], sets[edit]) >= MIN_PLANTED_JACCARD
    for decoy in decoy_rows:
        assert (
            max(
                _jaccard(sets[decoy], sets[other])
                for other in range(len(rows))
                if other != decoy
            )
            <= MAX_DECOY_JACCARD
        )

    expected = [
        any(
            _jaccard(sets[earlier], sets[row]) >= DEFAULT_CONFIG.threshold
            for earlier in range(row)
        )
        for row in range(len(rows))
    ]
    assert sum(expected) == NEAR_DUPLICATE_COUNT

    result = minhash.apply(_lazy(*rows), DEFAULT_CONFIG).collect()
    actual = [reason == "duplicate" for reason in _reasons(result)]

    true_positives = sum(a and e for a, e in zip(actual, expected, strict=True))
    false_positives = sum(a and not e for a, e in zip(actual, expected, strict=True))
    false_negatives = sum(e and not a for a, e in zip(actual, expected, strict=True))
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    assert recall >= MIN_RECALL, (true_positives, false_positives, false_negatives)
    assert precision >= MIN_PRECISION, (
        true_positives,
        false_positives,
        false_negatives,
    )
