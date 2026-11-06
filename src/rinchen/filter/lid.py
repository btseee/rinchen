"""B7 `GlotLID` wrapper and the `lid` stage.

`GlotLID` wraps a fastText language-identification model (GlotLID). It returns
labels without the `__label__` prefix, and labels text written in traditional
Mongolian script `mn_Mong` before the model runs, because GlotLID does not
know that code. The `lid` stage scores both sides of every row with it and
rejects a row whose side is not in the language the row says it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import fasttext
import polars as pl

from rinchen.config.filters import LidConfig
from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, DataError
from rinchen.filter.reasons import FilterReason, flag, with_scores

_LABEL_PREFIX = "__label__"
TRADITIONAL_MONGOLIAN_LABEL = "mn_Mong"
# The Mongolian block, U+1800..U+18AF.
_TRADITIONAL_FIRST = 0x1800
_TRADITIONAL_LAST = 0x18AF


class _FastTextModel(Protocol):
    def predict(self, texts: list[str], k: int) -> tuple[Any, Any]: ...


class GlotLID:
    """A GlotLID model read from `model_path`.

    Raises ConfigError, naming the path, when the file is missing or is not a
    fastText model."""

    def __init__(self, model_path: Path) -> None:
        self._model = _load_model(model_path)

    def predict(
        self, texts: Sequence[str], k: int = 3
    ) -> list[list[tuple[str, float]]]:
        """The `k` most probable labels of each text, most probable first,
        with probabilities clamped to [0, 1]. A label is a GlotLID code such
        as `khk_Cyrl`. A model with fewer than `k` labels gives all it has.

        A text whose non-whitespace characters are at least half traditional
        Mongolian script (U+1800..U+18AF) is `[("mn_Mong", 1.0)]`, without the
        model. A blank text has no language to name and gets `[]`. A newline is
        read as a space, because fastText predicts one line at a time. The
        model is asked once for all remaining texts."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        results: list[list[tuple[str, float]]] = [[] for _ in texts]
        positions: list[int] = []
        lines: list[str] = []
        for position, text in enumerate(texts):
            if _is_traditional_script(text):
                results[position] = [(TRADITIONAL_MONGOLIAN_LABEL, 1.0)]
            elif text.strip():
                positions.append(position)
                lines.append(text.replace("\n", " "))
        if lines:
            all_labels, all_probs = self._model.predict(lines, k=k)
            for position, labels, probs in zip(
                positions, all_labels, all_probs, strict=True
            ):
                results[position] = [
                    (label.removeprefix(_LABEL_PREFIX), _clamp_probability(prob))
                    for label, prob in zip(labels, probs, strict=True)
                ]
        return results


def apply(df: pl.LazyFrame, cfg: LidConfig) -> pl.LazyFrame:
    """Score both sides, then reject with `lid_src` first and `lid_tgt`
    second.

    A side scores the summed probability of the labels accepted for its
    language (`cfg.accept`, else the language's own code) among the model's
    top `cfg.top_k` guesses; a blank side scores 0.0. The model is loaded once
    here, so a missing model file raises ConfigError at this call. A language
    code that is not in the registry raises DataError when the frame is
    collected."""
    model = GlotLID(cfg.model_path)
    scored = with_scores(
        df,
        lid_src=_side_score("src", "src_lang", model, cfg),
        lid_tgt=_side_score("tgt", "tgt_lang", model, cfg),
    )
    floor = pl.lit(cfg.min_prob, dtype=pl.Float32)
    src_score = pl.col("scores").struct.field("lid_src")
    tgt_score = pl.col("scores").struct.field("lid_tgt")
    src_flagged = flag(scored, src_score < floor, FilterReason.LID_SRC)
    return flag(src_flagged, tgt_score < floor, FilterReason.LID_TGT)


def _load_model(model_path: Path) -> _FastTextModel:
    if not model_path.is_file():
        raise ConfigError(f"LID model file not found: {model_path}")
    try:
        model: _FastTextModel = fasttext.load_model(str(model_path))
    except (ValueError, OSError) as exc:
        raise ConfigError(f"cannot read LID model {model_path}: {exc}") from exc
    return model


def _is_traditional_script(text: str) -> bool:
    traditional = 0
    other = 0
    for char in text:
        if char.isspace():
            continue
        if _TRADITIONAL_FIRST <= ord(char) <= _TRADITIONAL_LAST:
            traditional += 1
        else:
            other += 1
    return traditional > 0 and traditional >= other


def _clamp_probability(probability: float) -> float:
    # fastText reports 1.00001 for a certain answer.
    return min(max(float(probability), 0.0), 1.0)


def _side_score(
    text_column: str, lang_column: str, model: GlotLID, cfg: LidConfig
) -> pl.Expr:
    def score_batch(batch: pl.Series) -> pl.Series:
        texts = batch.struct.field("text").to_list()
        langs = batch.struct.field("lang").to_list()
        # Resolved before the model runs: a bad code must not cost a prediction.
        accepted = {
            lang: _accepted_labels(lang, cfg.accept) for lang in dict.fromkeys(langs)
        }
        predictions = model.predict(texts, k=cfg.top_k)
        return pl.Series(
            [
                _accepted_probability(row_predictions, accepted[lang])
                for row_predictions, lang in zip(predictions, langs, strict=True)
            ],
            dtype=pl.Float32,
        )

    # Not elementwise, although a score depends on its own row only: polars
    # would split the column into one morsel per thread, down to one row per
    # model call on a small frame, and call the model from several threads.
    # One sequential call per side is what the model is built for.
    return pl.struct(
        pl.col(text_column).fill_null("").alias("text"),
        pl.col(lang_column).alias("lang"),
    ).map_batches(score_batch, return_dtype=pl.Float32)


def _accepted_labels(
    lang: str | None, accept: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    if lang is None or lang not in known_language_codes():
        raise DataError(
            f"language code {lang!r} is not in the language registry "
            "(configs/languages.yaml)"
        )
    return frozenset(accept.get(lang, [lang]))


def _accepted_probability(
    predictions: Sequence[tuple[str, float]], accepted: frozenset[str]
) -> float:
    return _clamp_probability(
        sum(prob for label, prob in predictions if label in accepted)
    )
