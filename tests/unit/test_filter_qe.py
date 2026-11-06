"""The `qe` stage: a translation whose reference-free quality score is on the
wrong side of the configured threshold is rejected as `qe`. The scorer is
injected, so these tests run a fake one and never need a GPU."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rinchen.config.filters import QeConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.errors import UpstreamError
from rinchen.filter import qe
from rinchen.filter.qe import EoleMetricXQEScorer, QEScorer
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus

METRICX = QeConfig(scorer="metricx_qe", max_metricx_qe=8.0, batch_size=32)
COMETKIWI = QeConfig(scorer="cometkiwi", min_cometkiwi=0.55, batch_size=32)


class FakeScorer:
    """Scores a pair by its translation: looked up in `by_target`, else the
    translation's length. Records every call."""

    def __init__(
        self,
        by_target: Mapping[str, float] | None = None,
        *,
        requires_cuda: bool = False,
    ) -> None:
        self.requires_cuda = requires_cuda
        self._by_target = dict(by_target or {})
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.calls.append((list(sources), list(translations)))
        return [self._by_target.get(text, float(len(text))) for text in translations]


class ScriptedScorer:
    """Answers each call with whatever `answer` makes of the call's sources."""

    requires_cuda = False

    def __init__(self, answer: Callable[[Sequence[str]], list[float]]) -> None:
        self._answer = answer

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        return self._answer(sources)


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


def _score_column(result: pl.DataFrame, field: str) -> list[float | None]:
    return result["scores"].struct.field(field).to_list()


def _reasons(result: pl.DataFrame) -> list[str | None]:
    return result["filter_reason"].to_list()


def _next_float32_up(value: float) -> float:
    return float(np.nextafter(np.float32(value), np.float32(np.inf)))


def _next_float32_down(value: float) -> float:
    return float(np.nextafter(np.float32(value), np.float32(-np.inf)))


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


# --- which score column, which direction --------------------------------------------


def test_metricx_scores_go_to_metricx_qe_and_flag_above_the_maximum() -> None:
    scorer = FakeScorer({"fine": 3.0, "poor": 12.5})

    result = qe.apply(
        _frame(("a", "fine"), ("b", "poor")).lazy(), METRICX, scorer=scorer
    ).collect()

    assert _score_column(result, "metricx_qe") == [3.0, 12.5]
    assert _score_column(result, "cometkiwi") == [None, None]
    assert _reasons(result) == [None, "qe"]


def test_cometkiwi_scores_go_to_cometkiwi_and_flag_below_the_minimum() -> None:
    scorer = FakeScorer({"fine": 0.9, "poor": 0.2})

    result = qe.apply(
        _frame(("a", "fine"), ("b", "poor")).lazy(), COMETKIWI, scorer=scorer
    ).collect()

    assert _score_column(result, "cometkiwi") == pytest.approx([0.9, 0.2], abs=1e-6)
    assert _score_column(result, "metricx_qe") == [None, None]
    assert _reasons(result) == [None, "qe"]


def test_a_good_metricx_score_is_low_and_a_good_cometkiwi_score_is_high() -> None:
    # The same high number is a failure for MetricX-QE and a pass for CometKiwi.
    scorer = FakeScorer({"x": 20.0})
    frame = _frame(("a", "x")).lazy()

    metricx = qe.apply(frame, METRICX, scorer=scorer).collect()
    cometkiwi = qe.apply(frame, COMETKIWI, scorer=scorer).collect()

    assert _reasons(metricx) == ["qe"]
    assert _reasons(cometkiwi) == [None]


def test_scores_are_float32() -> None:
    result = qe.apply(_frame(("a", "b")).lazy(), METRICX, scorer=FakeScorer()).collect()

    assert result["scores"].struct.field("metricx_qe").dtype == pl.Float32


def test_only_the_configured_scorers_field_is_written() -> None:
    result = qe.apply(
        _frame(("a", "b")).lazy(), COMETKIWI, scorer=FakeScorer({"b": 0.9})
    ).collect()

    others = [name for name in result["scores"].struct.fields if name != "cometkiwi"]
    assert others
    for name in others:
        assert result["scores"].struct.field(name).null_count() == result.height


# --- the thresholds -----------------------------------------------------------------


def test_a_metricx_score_exactly_at_the_maximum_is_not_flagged() -> None:
    above = _next_float32_up(METRICX.max_metricx_qe)
    scorer = FakeScorer({"at": METRICX.max_metricx_qe, "over": above})

    result = qe.apply(
        _frame(("a", "at"), ("b", "over")).lazy(), METRICX, scorer=scorer
    ).collect()

    assert _reasons(result) == [None, "qe"]


def test_a_cometkiwi_score_exactly_at_the_minimum_is_not_flagged() -> None:
    below = _next_float32_down(COMETKIWI.min_cometkiwi)
    scorer = FakeScorer({"at": COMETKIWI.min_cometkiwi, "under": below})

    result = qe.apply(
        _frame(("a", "at"), ("b", "under")).lazy(), COMETKIWI, scorer=scorer
    ).collect()

    assert _reasons(result) == [None, "qe"]


@pytest.mark.parametrize("cfg", [METRICX, COMETKIWI], ids=["metricx_qe", "cometkiwi"])
def test_a_nan_score_is_flagged_not_kept(cfg: QeConfig) -> None:
    good = 1.0 if cfg.scorer == "metricx_qe" else 0.99
    scorer = FakeScorer({"nan": math.nan, "good": good})

    result = qe.apply(
        _frame(("a", "good"), ("b", "nan")).lazy(), cfg, scorer=scorer
    ).collect()

    field = "metricx_qe" if cfg.scorer == "metricx_qe" else "cometkiwi"
    nan_score = _score_column(result, field)[1]
    assert nan_score is not None and math.isnan(nan_score)
    assert _reasons(result) == [None, "qe"]


def test_a_row_that_already_has_a_reason_keeps_it_and_is_still_scored() -> None:
    df = _frame(("a", "x" * 20), ("b", "x" * 20)).with_columns(
        pl.Series("filter_reason", ["numbers", None], dtype=pl.String)
    )

    result = qe.apply(df.lazy(), METRICX, scorer=FakeScorer()).collect()

    assert _reasons(result) == ["numbers", "qe"]
    assert _score_column(result, "metricx_qe") == [20.0, 20.0]


# --- chunking by batch size ---------------------------------------------------------


def test_the_scorer_gets_chunks_of_at_most_batch_size_rows_in_order() -> None:
    pairs = [(f"src {index}", f"tgt {index}") for index in range(5)]
    scorer = FakeScorer()
    cfg = QeConfig(batch_size=2)

    qe.apply(_frame(*pairs).lazy(), cfg, scorer=scorer).collect()

    assert [len(sources) for sources, _ in scorer.calls] == [2, 2, 1]
    assert [src for sources, _ in scorer.calls for src in sources] == [
        src for src, _ in pairs
    ]
    assert [tgt for _, targets in scorer.calls for tgt in targets] == [
        tgt for _, tgt in pairs
    ]


def test_every_row_goes_to_the_scorer_with_the_source_first() -> None:
    scorer = FakeScorer()

    qe.apply(_frame(("hello", "world")).lazy(), METRICX, scorer=scorer).collect()

    assert scorer.calls == [(["hello"], ["world"])]


def test_a_batch_size_larger_than_the_frame_is_one_call() -> None:
    scorer = FakeScorer()
    pairs = [(f"src {index}", f"tgt {index}") for index in range(3)]

    qe.apply(_frame(*pairs).lazy(), QeConfig(batch_size=50), scorer=scorer).collect()

    assert [len(sources) for sources, _ in scorer.calls] == [3]


def test_a_frame_of_exactly_batch_size_rows_is_one_call() -> None:
    scorer = FakeScorer()
    pairs = [(f"src {index}", f"tgt {index}") for index in range(4)]

    qe.apply(_frame(*pairs).lazy(), QeConfig(batch_size=4), scorer=scorer).collect()

    assert [len(sources) for sources, _ in scorer.calls] == [4]


def test_a_column_in_several_chunks_scores_the_same_as_one_chunk() -> None:
    pairs = [(f"src {index}", "t" * index) for index in range(6)]
    chunked = pl.concat(
        [_frame(*pairs[:2]), _frame(*pairs[2:5]), _frame(*pairs[5:])], rechunk=False
    )
    assert chunked["src"].n_chunks() == 3
    cfg = QeConfig(batch_size=4, max_metricx_qe=3.0)

    result = qe.apply(chunked.lazy(), cfg, scorer=FakeScorer()).collect()

    expected = qe.apply(_frame(*pairs).lazy(), cfg, scorer=FakeScorer()).collect()
    assert _score_column(result, "metricx_qe") == _score_column(expected, "metricx_qe")
    assert _reasons(result) == _reasons(expected)
    assert _reasons(result) == [None, None, None, None, "qe", "qe"]


def test_scoring_under_the_streaming_engine_gives_the_same_result() -> None:
    pairs = [(f"src {index}", "t" * (index + 1)) for index in range(5)]
    spy = FakeScorer()
    cfg = QeConfig(batch_size=2, max_metricx_qe=3.0)
    lazy = qe.apply(_frame(*pairs).lazy(), cfg, scorer=spy)

    streamed = lazy.collect(engine="streaming")

    in_memory = qe.apply(_frame(*pairs).lazy(), cfg, scorer=FakeScorer()).collect()
    assert [len(sources) for sources, _ in spy.calls] == [2, 2, 1]
    assert _score_column(streamed, "metricx_qe") == _score_column(
        in_memory, "metricx_qe"
    )
    assert _reasons(streamed) == _reasons(in_memory)


# --- a scorer that misbehaves -------------------------------------------------------


@pytest.mark.parametrize("returned", [1, 3], ids=["too_few", "too_many"])
def test_a_result_of_the_wrong_length_is_an_upstream_error(returned: int) -> None:
    scorer = ScriptedScorer(lambda sources: [1.0] * returned)

    with pytest.raises(
        UpstreamError, match=re.escape(f"returned {returned} scores for 2 rows")
    ):
        qe.apply(
            _frame(("a", "b"), ("c", "d")).lazy(), METRICX, scorer=scorer
        ).collect()


def test_a_short_result_for_the_last_chunk_is_an_upstream_error() -> None:
    def short_last(sources: Sequence[str]) -> list[float]:
        return [1.0] * (len(sources) - (1 if len(sources) == 1 else 0))

    scorer = ScriptedScorer(short_last)
    pairs = [(f"src {index}", f"tgt {index}") for index in range(5)]

    with pytest.raises(UpstreamError):
        qe.apply(_frame(*pairs).lazy(), QeConfig(batch_size=2), scorer=scorer).collect()


# --- empty and missing text ---------------------------------------------------------


def test_apply_on_an_empty_frame_returns_an_empty_frame_of_the_same_schema() -> None:
    scorer = FakeScorer()

    result = qe.apply(corpus_frame([]).lazy(), METRICX, scorer=scorer).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA
    assert scorer.calls == []


def test_a_null_side_is_scored_as_empty_text() -> None:
    frame = _frame(("", "x"), ("y", "")).with_columns(
        pl.when(pl.col("src") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("src"))
        .alias("src"),
        pl.when(pl.col("tgt") == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("tgt"))
        .alias("tgt"),
    )
    assert frame["src"].null_count() == 1
    assert frame["tgt"].null_count() == 1
    scorer = FakeScorer()

    result = qe.apply(frame.lazy(), METRICX, scorer=scorer).collect()

    assert scorer.calls == [(["", "y"], ["x", ""])]
    assert _score_column(result, "metricx_qe") == [1.0, 0.0]


# --- the planted corpus -------------------------------------------------------------


def test_planted_corpus_stays_schema_valid_with_a_fixed_score_per_row(
    planted: PlantedCorpus,
) -> None:
    scorer = ScriptedScorer(lambda sources: [2.5] * len(sources))

    result = qe.apply(planted.frame.lazy(), METRICX, scorer=scorer).collect()

    assert result.schema == CORPUS_SCHEMA
    assert result.height == planted.frame.height
    assert _score_column(result, "metricx_qe") == [2.5] * planted.frame.height
    assert result["filter_reason"].null_count() == result.height
    assert result.drop("scores", "filter_reason").equals(
        planted.frame.drop("scores", "filter_reason")
    )


def test_planted_corpus_rows_over_the_maximum_are_all_flagged_qe(
    planted: PlantedCorpus,
) -> None:
    scorer = ScriptedScorer(lambda sources: [9.0] * len(sources))

    result = qe.apply(planted.frame.lazy(), METRICX, scorer=scorer).collect()

    assert result.schema == CORPUS_SCHEMA
    assert set(_reasons(result)) == {"qe"}


# --- skipping -----------------------------------------------------------------------


def test_no_scorer_means_the_stage_is_skipped_whatever_cuda_says() -> None:
    for cuda in (True, False):
        assert qe.skip_reason(METRICX, None, cuda_available=cuda) == (
            "no QE scorer configured"
        )


def test_a_cuda_scorer_without_a_cuda_device_is_skipped() -> None:
    scorer = FakeScorer(requires_cuda=True)

    assert qe.skip_reason(METRICX, scorer, cuda_available=False) == "no CUDA device"


def test_a_cuda_scorer_with_a_cuda_device_runs() -> None:
    scorer = FakeScorer(requires_cuda=True)

    assert qe.skip_reason(METRICX, scorer, cuda_available=True) is None


def test_a_scorer_that_needs_no_cuda_runs_without_it() -> None:
    assert qe.skip_reason(COMETKIWI, FakeScorer(), cuda_available=False) is None


def test_cuda_is_only_looked_up_when_a_cuda_scorer_needs_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups: list[None] = []

    def cuda_is_available() -> bool:
        lookups.append(None)
        return False

    monkeypatch.setattr(qe, "_cuda_is_available", cuda_is_available)

    assert qe.skip_reason(METRICX, None) == "no QE scorer configured"
    assert qe.skip_reason(METRICX, FakeScorer()) is None
    assert (
        qe.skip_reason(METRICX, FakeScorer(requires_cuda=True), cuda_available=True)
        is None
    )
    assert lookups == []

    assert qe.skip_reason(METRICX, FakeScorer(requires_cuda=True)) == "no CUDA device"
    assert lookups == [None]


@pytest.mark.parametrize("available", [True, False])
def test_the_cuda_lookup_asks_torch(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    reason = qe.skip_reason(METRICX, FakeScorer(requires_cuda=True))

    assert reason == (None if available else "no CUDA device")


def test_a_fake_scorer_satisfies_the_protocol() -> None:
    scorer: QEScorer = FakeScorer()

    assert scorer.requires_cuda is False


# --- the Eole MetricX-QE adapter ----------------------------------------------------


def test_the_eole_scorer_needs_cuda_and_is_a_scorer() -> None:
    scorer: QEScorer = EoleMetricXQEScorer()

    assert scorer.requires_cuda is True


def test_the_eole_scorer_scores_no_pairs_without_loading_a_model(
    tmp_path: Path,
) -> None:
    scorer = EoleMetricXQEScorer(str(tmp_path / "not-a-model"))

    assert scorer.score([], []) == []


# Importing Eole warns about its missing C++ extension and flash_attn and about
# deprecations in Eole, pydantic and torch; none of it is ours to fix.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Could not import:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.torch.jit.script. is not supported:FutureWarning")
def test_a_model_that_is_neither_a_directory_nor_a_repo_is_an_upstream_error(
    tmp_path: Path,
) -> None:
    scorer = EoleMetricXQEScorer(str(tmp_path / "not-a-model"))

    with pytest.raises(UpstreamError, match="not-a-model"):
        scorer.score(["hello"], ["world"])


@pytest.mark.subprocess
def test_importing_the_qe_stage_does_not_import_torch_or_eole() -> None:
    snippet = (
        "import sys\n"
        "import rinchen.filter.qe\n"
        "heavy = [m for m in ('torch', 'eole', 'transformers')"
        " if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


# --- the real model -----------------------------------------------------------------


@pytest.mark.network
def test_metricx_qe_scores_a_translation_lower_than_a_mismatched_pair() -> None:
    if not qe._cuda_is_available():
        pytest.skip("MetricX-QE through Eole needs a CUDA device")
    sources = [
        "The weather is nice today.",
        "The weather is nice today.",
        "I bought three apples at the market.",
        "I bought three apples at the market.",
    ]
    translations = [
        "Өнөөдөр цаг агаар сайхан байна.",
        "Хадны оройд цас орж байна.",
        "Би зах дээрээс гурван алим худалдаж авлаа.",
        "Манай компани шинэ оффис нээлээ.",
    ]

    scores = EoleMetricXQEScorer().score(sources, translations)

    assert len(scores) == 4
    assert all(0.0 <= score <= 25.0 for score in scores)
    assert scores[0] < scores[1]
    assert scores[2] < scores[3]
