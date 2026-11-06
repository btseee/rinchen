"""The `labse` stage: a row whose two sides do not embed close together is
rejected as `labse`. Every row gets a score."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from rinchen.config.filters import LabseConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.filter import labse
from rinchen.mine import embed as embed_module
from rinchen.mine.embed import Embedder
from tests.support.embedders import ConceptEmbedder
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus

DEFAULT_CONFIG = LabseConfig()
IDENTICAL_PAIR_COUNT = 400


class RecordingEmbedder:
    """Wraps an embedder and records the texts of every `embed` call."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return self._inner.embed(texts)


class RealShapeEmbedder:
    """Like the real embedder, answers no texts with an array of shape (0, 0)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._inner.embed(texts)


class NanEmbedder:
    """Embeds the text "nan" as a vector of NaN and everything else as `inner`."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self._inner.embed(texts)
        for position, text in enumerate(texts):
            if text == "nan":
                vectors[position] = np.nan
        return vectors


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


def _scores(result: pl.DataFrame) -> list[float | None]:
    return result["scores"].struct.field("labse").to_list()


def _reasons(result: pl.DataFrame) -> list[str | None]:
    return result["filter_reason"].to_list()


def _concepts(*groups: Sequence[str]) -> ConceptEmbedder:
    """Texts of one group share a concept; every group has its own."""
    return ConceptEmbedder(
        {text: index for index, group in enumerate(groups, start=1) for text in group}
    )


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


# --- the planted corpus -------------------------------------------------------------


def test_planted_corpus_flags_exactly_the_planted_labse_rows(
    planted: PlantedCorpus,
) -> None:
    result = labse.apply(
        planted.frame.lazy(),
        DEFAULT_CONFIG,
        embedder=ConceptEmbedder(planted.concepts),
    ).collect()

    flagged = {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }
    assert set(flagged) == planted.isolated_failures["labse"]
    assert len(flagged) == 3
    assert set(flagged.values()) == {"labse"}
    assert result.schema == CORPUS_SCHEMA


def test_planted_corpus_scores_every_row_within_the_cosine_range(
    planted: PlantedCorpus,
) -> None:
    result = labse.apply(
        planted.frame.lazy(),
        DEFAULT_CONFIG,
        embedder=ConceptEmbedder(planted.concepts),
    ).collect()

    scores = _scores(result)
    assert None not in scores
    assert all(-1.0 <= score <= 1.0 for score in scores if score is not None)
    flagged = planted.isolated_failures["labse"]
    for row_id, score in zip(result["id"].to_list(), scores, strict=True):
        assert score is not None
        assert (score < DEFAULT_CONFIG.min_cosine) == (row_id in flagged)


def test_the_stage_writes_only_the_labse_score(planted: PlantedCorpus) -> None:
    result = labse.apply(
        planted.frame.lazy(),
        DEFAULT_CONFIG,
        embedder=ConceptEmbedder(planted.concepts),
    ).collect()

    others = [name for name in result["scores"].struct.fields if name != "labse"]
    assert others
    for name in others:
        assert result["scores"].struct.field(name).null_count() == result.height


def test_planted_corpus_other_columns_are_unchanged(planted: PlantedCorpus) -> None:
    result = labse.apply(
        planted.frame.lazy(),
        DEFAULT_CONFIG,
        embedder=ConceptEmbedder(planted.concepts),
    ).collect()

    assert result.drop("scores", "filter_reason").equals(
        planted.frame.drop("scores", "filter_reason")
    )


# --- the threshold ------------------------------------------------------------------


def test_a_row_below_the_threshold_is_flagged_and_a_matching_row_is_not() -> None:
    embedder = _concepts(["a1", "a2"], ["b1"], ["c1"])

    result = labse.apply(
        _frame(("a1", "a2"), ("b1", "c1")).lazy(), DEFAULT_CONFIG, embedder=embedder
    ).collect()

    assert _reasons(result) == [None, "labse"]


def test_a_score_exactly_at_the_threshold_is_not_flagged() -> None:
    embedder = _concepts(["a1", "a2"], ["b1"], ["c1"])
    frame = _frame(("a1", "a2"), ("b1", "c1")).lazy()
    unrelated_score = _scores(
        labse.apply(frame, DEFAULT_CONFIG, embedder=embedder).collect()
    )[1]
    assert unrelated_score is not None

    at_score = LabseConfig(min_cosine=unrelated_score)
    above_score = LabseConfig(
        min_cosine=float(np.nextafter(np.float32(unrelated_score), np.float32(1)))
    )

    at = labse.apply(frame, at_score, embedder=embedder).collect()
    above = labse.apply(frame, above_score, embedder=embedder).collect()
    assert _reasons(at) == [None, None]
    assert _reasons(above) == [None, "labse"]


def test_a_nan_score_is_flagged_not_kept() -> None:
    embedder = NanEmbedder(_concepts(["a1", "a2"]))
    frame = _frame(("a1", "a2"), ("nan", "a1"), ("a1", "nan"))

    result = labse.apply(frame.lazy(), DEFAULT_CONFIG, embedder=embedder).collect()

    scores = _scores(result)
    assert scores[0] is not None
    assert all(score is not None and math.isnan(score) for score in scores[1:])
    assert _reasons(result) == [None, "labse", "labse"]


def test_a_row_that_already_has_a_reason_keeps_it() -> None:
    embedder = _concepts(["a1"], ["b1"])
    df = _frame(("a1", "b1"), ("a1", "b1")).with_columns(
        pl.Series("filter_reason", ["numbers", None], dtype=pl.String)
    )

    result = labse.apply(df.lazy(), DEFAULT_CONFIG, embedder=embedder).collect()

    assert _reasons(result) == ["numbers", "labse"]
    assert None not in _scores(result)


def test_identical_sides_never_score_above_one() -> None:
    # Float32 rounding pushes the dot product of a unit vector with itself
    # slightly past 1 for some vectors.
    texts = [f"sentence {index}" for index in range(IDENTICAL_PAIR_COUNT)]
    embedder = ConceptEmbedder({text: index for index, text in enumerate(texts)})

    result = labse.apply(
        _frame(*[(text, text) for text in texts]).lazy(),
        DEFAULT_CONFIG,
        embedder=embedder,
    ).collect()

    scores = [score for score in _scores(result) if score is not None]
    assert len(scores) == IDENTICAL_PAIR_COUNT
    assert max(scores) == 1.0
    assert min(scores) > 0.999
    assert set(_reasons(result)) == {None}


# --- empty and missing text ---------------------------------------------------------


def test_empty_sides_are_embedded_like_any_text() -> None:
    embedder = _concepts([""], ["x"])

    result = labse.apply(
        _frame(("", ""), ("", "x"), ("x", "")).lazy(), DEFAULT_CONFIG, embedder=embedder
    ).collect()

    scores = _scores(result)
    assert None not in scores
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(scores[2], abs=1e-6)
    assert _reasons(result) == [None, "labse", "labse"]


def test_whitespace_only_sides_get_a_score_and_do_not_crash() -> None:
    result = labse.apply(
        _frame((" ", "\t"), ("\n", "x")).lazy(),
        DEFAULT_CONFIG,
        embedder=ConceptEmbedder({}),
    ).collect()

    assert None not in _scores(result)


def test_a_null_side_is_embedded_as_empty_text() -> None:
    embedder = ConceptEmbedder({})
    empty = _frame(("", "x"), ("y", ""))
    nulled = empty.with_columns(
        pl.when(pl.col("src") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("src"))
        .alias("src"),
        pl.when(pl.col("tgt") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("tgt"))
        .alias("tgt"),
    )
    assert nulled["src"].null_count() == 1
    assert nulled["tgt"].null_count() == 1

    from_empty = labse.apply(empty.lazy(), DEFAULT_CONFIG, embedder=embedder).collect()
    from_null = labse.apply(nulled.lazy(), DEFAULT_CONFIG, embedder=embedder).collect()

    assert None not in _scores(from_null)
    assert _scores(from_null) == _scores(from_empty)
    assert _reasons(from_null) == _reasons(from_empty)


def test_apply_on_an_empty_frame_returns_an_empty_frame_of_the_same_schema() -> None:
    result = labse.apply(
        corpus_frame([]).lazy(), DEFAULT_CONFIG, embedder=ConceptEmbedder({})
    ).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_an_embedder_that_answers_no_texts_with_shape_zero_by_zero_is_accepted() -> (
    None
):
    embedder = RealShapeEmbedder(ConceptEmbedder({}))

    result = labse.apply(
        corpus_frame([]).lazy(), DEFAULT_CONFIG, embedder=embedder
    ).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_the_built_embedder_loads_no_model_for_an_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, str]] = []

    def load_model(model_name: str, device: str) -> None:
        loads.append((model_name, device))

    monkeypatch.setattr(embed_module, "_load_model", load_model)

    result = labse.apply(corpus_frame([]).lazy(), LabseConfig(device="cpu")).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA
    assert loads == []


# --- batching across rows -----------------------------------------------------------


def test_the_embedder_receives_each_side_once_for_all_rows() -> None:
    pairs = [(f"src {index}", f"tgt {index}") for index in range(9)]
    spy = RecordingEmbedder(ConceptEmbedder({}))

    labse.apply(_frame(*pairs).lazy(), DEFAULT_CONFIG, embedder=spy).collect()

    assert [len(texts) for texts in spy.calls] == [9, 9]
    assert spy.calls[0] == [src for src, _ in pairs]
    assert spy.calls[1] == [tgt for _, tgt in pairs]


def test_a_column_in_several_chunks_is_scored_as_one_batch() -> None:
    pairs = [(f"src {index}", f"tgt {index}") for index in range(6)]
    concepts = {text: index for index, pair in enumerate(pairs) for text in pair}
    chunked = pl.concat(
        [_frame(*pairs[:2]), _frame(*pairs[2:5]), _frame(*pairs[5:])], rechunk=False
    )
    assert chunked["src"].n_chunks() == 3
    spy = RecordingEmbedder(ConceptEmbedder(concepts))

    result = labse.apply(chunked.lazy(), DEFAULT_CONFIG, embedder=spy).collect()

    assert [len(texts) for texts in spy.calls] == [6, 6]
    expected = labse.apply(
        _frame(*pairs).lazy(), DEFAULT_CONFIG, embedder=ConceptEmbedder(concepts)
    ).collect()
    assert _scores(result) == _scores(expected)
    assert set(_reasons(result)) == {None}


def test_scores_are_the_row_wise_dot_product_in_float32() -> None:
    embedder = ConceptEmbedder({"a": 1, "b": 1, "c": 2})
    frame = _frame(("a", "b"), ("a", "c"), ("c", "c"))

    result = labse.apply(frame.lazy(), DEFAULT_CONFIG, embedder=embedder).collect()

    src = embedder.embed(["a", "a", "c"])
    tgt = embedder.embed(["b", "c", "c"])
    assert result["scores"].struct.field("labse").dtype == pl.Float32
    assert _scores(result) == pytest.approx(
        [float(s @ t) for s, t in zip(src, tgt, strict=True)], abs=1e-6
    )


def test_scoring_under_the_streaming_engine_gives_the_same_result() -> None:
    pairs = [(f"src {index}", f"tgt {index}") for index in range(5)]
    spy = RecordingEmbedder(ConceptEmbedder({}))
    lazy = labse.apply(_frame(*pairs).lazy(), DEFAULT_CONFIG, embedder=spy)

    streamed = lazy.collect(engine="streaming")

    in_memory = labse.apply(
        _frame(*pairs).lazy(), DEFAULT_CONFIG, embedder=ConceptEmbedder({})
    ).collect()
    assert [len(texts) for texts in spy.calls] == [5, 5]
    assert _scores(streamed) == _scores(in_memory)


# --- building the embedder from the config ------------------------------------------


def test_without_an_embedder_one_is_built_from_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []

    def build(model_name: str, *, batch_size: int, device: str) -> ConceptEmbedder:
        built.append({"model": model_name, "batch_size": batch_size, "device": device})
        return ConceptEmbedder({})

    monkeypatch.setattr(labse, "SentenceTransformerEmbedder", build)
    cfg = LabseConfig(model="some/model", batch_size=16, device="cpu")

    result = labse.apply(_frame(("a", "b")).lazy(), cfg).collect()

    assert built == [{"model": "some/model", "batch_size": 16, "device": "cpu"}]
    assert None not in _scores(result)
