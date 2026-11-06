"""The filter pipeline: stages run in the configured order against the planted
corpus, with the tiny LID model, the tiny SentencePiece model and the concept
embedder standing in for the real models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from rinchen.config.filters import FilterConfig, LengthConfig, LidConfig, StageName
from rinchen.corpus.schema import CORPUS_SCHEMA, SCORES_DTYPE, corpus_frame
from rinchen.errors import ConfigError, DataError
from rinchen.filter.pipeline import PipelineResult, kept_rows, run_pipeline
from rinchen.filter.reasons import FilterReason
from tests.support.embedders import ConceptEmbedder
from tests.support.planted_corpus import (
    PLANTED_REASON_COUNTS,
    PlantedCorpus,
    build_planted_corpus,
    tiny_spm_model,
)
from tests.support.tiny_lid import train_tiny_lid_model

NO_SCORER_STATUS = "skipped: no QE scorer configured"
PARTIAL_LENGTH_STATUS = "partial: token ratio skipped (no spm_model)"
ALL_REASONS = [reason.value for reason in FilterReason]
SCORE_FIELDS = list(SCORES_DTYPE.to_schema())


class ConstantScorer:
    """A QE scorer that gives every pair the same score, except the
    translations listed in `by_target`; it records how many pairs it saw."""

    requires_cuda = False

    def __init__(
        self, score: float = 1.0, by_target: dict[str, float] | None = None
    ) -> None:
        self._score = score
        self._by_target = by_target or {}
        self.pairs_seen = 0

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.pairs_seen += len(sources)
        return [self._by_target.get(text, self._score) for text in translations]


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


@pytest.fixture(scope="module")
def lid_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_tiny_lid_model(tmp_path_factory.mktemp("pipeline_lid") / "lid.bin")


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_spm_model(tmp_path_factory.mktemp("pipeline_spm"))


@pytest.fixture(scope="module")
def tiny_config(lid_model: Path, spm_model: Path) -> FilterConfig:
    return FilterConfig(
        lid=LidConfig(model_path=lid_model),
        length=LengthConfig(spm_model=spm_model),
    )


@pytest.fixture(scope="module")
def embedder(planted: PlantedCorpus) -> ConceptEmbedder:
    return ConceptEmbedder(planted.concepts)


@pytest.fixture(scope="module")
def full_run(
    planted: PlantedCorpus, tiny_config: FilterConfig, embedder: ConceptEmbedder
) -> PipelineResult:
    return run_pipeline(planted.frame.lazy(), tiny_config, embedder=embedder)


def _reasons_by_id(result: PipelineResult) -> dict[str, str | None]:
    return dict(
        zip(
            result.frame["id"].to_list(),
            result.frame["filter_reason"].to_list(),
            strict=True,
        )
    )


def _only(config: FilterConfig, *stages: StageName) -> FilterConfig:
    return config.model_copy(update={"stages": list(stages)})


def _score_nulls(frame: pl.DataFrame, field: str) -> int:
    return frame["scores"].struct.field(field).null_count()


def test_planted_corpus_reaches_the_planted_reason_counts(
    full_run: PipelineResult,
) -> None:
    assert full_run.reason_counts == {**PLANTED_REASON_COUNTS, "qe": 0}


def test_every_row_gets_the_reason_the_planted_ledger_expects(
    planted: PlantedCorpus, full_run: PipelineResult
) -> None:
    assert _reasons_by_id(full_run) == planted.expected_reason


def test_result_frame_keeps_the_input_rows_in_order_and_stays_a_corpus(
    planted: PlantedCorpus, full_run: PipelineResult
) -> None:
    assert full_run.frame["id"].to_list() == planted.frame["id"].to_list()
    assert full_run.frame.schema == CORPUS_SCHEMA


def test_status_is_ok_for_every_stage_but_a_qe_stage_without_a_scorer(
    tiny_config: FilterConfig, full_run: PipelineResult
) -> None:
    assert list(full_run.stage_status) == tiny_config.stages
    assert full_run.stage_status == {
        **{name: "ok" for name in tiny_config.stages},
        "qe": NO_SCORER_STATUS,
    }


def test_reason_counts_name_every_reason_even_when_unused(
    full_run: PipelineResult,
) -> None:
    assert sorted(full_run.reason_counts) == sorted(ALL_REASONS)
    assert full_run.reason_counts["qe"] == 0


def test_a_skipped_qe_stage_adds_no_scores(full_run: PipelineResult) -> None:
    assert _score_nulls(full_run.frame, "metricx_qe") == full_run.frame.height
    assert _score_nulls(full_run.frame, "cometkiwi") == full_run.frame.height


def test_kept_rows_are_the_rows_without_a_reason(
    planted: PlantedCorpus, full_run: PipelineResult
) -> None:
    kept = kept_rows(full_run.frame)

    clean_ids = [
        row_id for row_id, reason in planted.expected_reason.items() if reason is None
    ]
    assert kept["id"].to_list() == clean_ids
    assert kept["filter_reason"].null_count() == kept.height
    assert kept.schema == CORPUS_SCHEMA


def test_stage_order_decides_which_reason_a_row_keeps(
    tiny_config: FilterConfig,
) -> None:
    # Latin letters in a Cyrillic source fail script_ratio; 12 against 13
    # fails numbers.
    frame = corpus_frame(
        [
            {
                "src_lang": "khk_Cyrl",
                "tgt_lang": "eng_Latn",
                "src": "abc def 12",
                "tgt": "xyz uvw 13",
                "origin": "web",
                "corpus": "unit",
                "split": "train",
            }
        ]
    )

    numbers_first = run_pipeline(
        frame.lazy(), _only(tiny_config, "numbers", "script_ratio")
    )
    script_first = run_pipeline(
        frame.lazy(), _only(tiny_config, "script_ratio", "numbers")
    )

    assert numbers_first.frame["filter_reason"].to_list() == ["numbers"]
    assert script_first.frame["filter_reason"].to_list() == ["script_src"]
    assert numbers_first.reason_counts["numbers"] == 1
    assert script_first.reason_counts["script_src"] == 1


def test_a_stage_left_out_of_stages_adds_no_scores(
    planted: PlantedCorpus, tiny_config: FilterConfig
) -> None:
    result = run_pipeline(planted.frame.lazy(), _only(tiny_config, "script_ratio"))

    assert _score_nulls(result.frame, "script_src") == 0
    assert _score_nulls(result.frame, "script_tgt") == 0
    untouched = [
        field for field in SCORE_FIELDS if field not in {"script_src", "script_tgt"}
    ]
    for field in untouched:
        assert _score_nulls(result.frame, field) == result.frame.height, field
    assert result.stage_status == {"script_ratio": "ok"}
    assert result.reason_counts["script_src"] == 2
    assert result.reason_counts["script_tgt"] == 2
    assert result.reason_counts["labse"] == 0


def test_a_reason_already_on_the_input_is_kept_and_counted(
    planted: PlantedCorpus, tiny_config: FilterConfig
) -> None:
    number_row = next(
        row_id
        for row_id, reason in planted.expected_reason.items()
        if reason == "numbers"
    )
    frame = planted.frame.with_columns(
        pl.when(pl.col("id") == number_row)
        .then(pl.lit("manual"))
        .otherwise(pl.col("filter_reason"))
        .alias("filter_reason")
    )

    result = run_pipeline(frame.lazy(), _only(tiny_config, "numbers"))

    assert _reasons_by_id(result)[number_row] == "manual"
    assert result.reason_counts["manual"] == 1
    assert result.reason_counts["numbers"] == PLANTED_REASON_COUNTS["numbers"] - 1


def test_a_corpus_without_rows_passes_through_every_stage(
    tiny_config: FilterConfig, embedder: ConceptEmbedder
) -> None:
    result = run_pipeline(corpus_frame([]).lazy(), tiny_config, embedder=embedder)

    assert result.frame.height == 0
    assert result.frame.schema == CORPUS_SCHEMA
    assert result.reason_counts == dict.fromkeys(ALL_REASONS, 0)
    assert kept_rows(result.frame).height == 0
    assert result.stage_status["qe"] == NO_SCORER_STATUS


def test_a_corpus_of_wrong_schema_is_a_data_error(
    planted: PlantedCorpus, tiny_config: FilterConfig
) -> None:
    missing_column = planted.frame.drop("scores")
    wrong_dtype = planted.frame.with_columns(pl.col("seg_idx").cast(pl.Int64))

    with pytest.raises(DataError, match="scores"):
        run_pipeline(missing_column.lazy(), tiny_config)
    with pytest.raises(DataError, match="seg_idx"):
        run_pipeline(wrong_dtype.lazy(), tiny_config)


def test_length_without_an_spm_model_reports_a_partial_status(
    planted: PlantedCorpus,
) -> None:
    result = run_pipeline(planted.frame.lazy(), _only(FilterConfig(), "length"))

    assert result.stage_status == {"length": PARTIAL_LENGTH_STATUS}
    assert result.reason_counts["len_ratio_token"] == 0
    assert result.reason_counts["len_ratio_char"] == 3


def test_a_qe_scorer_scores_every_row_and_rejects_poor_translations(
    planted: PlantedCorpus,
) -> None:
    clean_pair = planted.frame.filter(pl.col("filter_reason").is_null()).row(
        0, named=True
    )
    scorer = ConstantScorer(score=1.0, by_target={clean_pair["tgt"]: 20.0})

    result = run_pipeline(
        planted.frame.lazy(), _only(FilterConfig(), "qe"), qe_scorer=scorer
    )

    assert result.stage_status == {"qe": "ok"}
    assert scorer.pairs_seen == planted.frame.height
    assert _score_nulls(result.frame, "metricx_qe") == 0
    assert result.reason_counts["qe"] == 1
    assert _reasons_by_id(result)[clean_pair["id"]] == "qe"


def test_a_stage_list_of_only_a_skipped_stage_returns_the_input_unchanged(
    planted: PlantedCorpus,
) -> None:
    result = run_pipeline(planted.frame.lazy(), _only(FilterConfig(), "qe"))

    assert result.stage_status == {"qe": NO_SCORER_STATUS}
    assert result.frame.equals(planted.frame)
    assert result.reason_counts == dict.fromkeys(ALL_REASONS, 0)


def test_a_missing_lid_model_is_a_config_error(
    planted: PlantedCorpus, tmp_path: Path
) -> None:
    config = FilterConfig(lid=LidConfig(model_path=tmp_path / "absent.bin"))

    with pytest.raises(ConfigError, match="absent.bin"):
        run_pipeline(planted.frame.lazy(), _only(config, "lid"))
