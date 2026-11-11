"""The names of the metrics an evaluation result reports (D13, T7 and T8): their order, their
kinds, which way is better, the reason of a neural metric nobody measured, and the lookup that
names the valid ones."""

from __future__ import annotations

from typing import get_args

import pytest

from rinchen.errors import ConfigError
from rinchen.metrics.registry import (
    COUNT_METRICS,
    HIGHER_IS_BETTER,
    METRIC_NAMES,
    NO_SCORER_REASON,
    SEGMENT_SCORE_METRICS,
    MetricName,
    require_count_metric,
    require_metric,
)
from tests.support.module_imports import project_imports, source_of
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def test_the_six_metrics_come_in_the_order_of_the_result() -> None:
    assert METRIC_NAMES == (
        "bleu",
        "chrf",
        "chrf_pp",
        "comet22",
        "metricx24",
        "metricx24_qe",
    )


def test_the_literal_type_names_the_same_six_metrics() -> None:
    assert get_args(MetricName) == METRIC_NAMES


def test_the_count_metrics_and_the_segment_score_metrics_split_the_six() -> None:
    assert COUNT_METRICS == ("bleu", "chrf", "chrf_pp")
    assert SEGMENT_SCORE_METRICS == ("comet22", "metricx24", "metricx24_qe")
    assert COUNT_METRICS + SEGMENT_SCORE_METRICS == METRIC_NAMES


def test_only_the_metricx_scores_are_lower_is_better() -> None:
    assert dict(HIGHER_IS_BETTER) == {
        "bleu": True,
        "chrf": True,
        "chrf_pp": True,
        "comet22": True,
        "metricx24": False,
        "metricx24_qe": False,
    }


def test_the_direction_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        HIGHER_IS_BETTER["metricx24"] = True  # type: ignore[index]


def test_the_reason_of_an_unmeasured_neural_metric_is_the_one_of_the_spec() -> None:
    assert NO_SCORER_REASON == (
        "no segment scores were given: the COMET and MetricX scorers are not built"
    )


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_every_valid_name_is_returned_as_it_was_given(name: str) -> None:
    assert require_metric(name) == name


@pytest.mark.parametrize("name", ["blue", "BLEU", "chrf++", "", " bleu", "comet"])
def test_a_name_that_is_no_metric_is_a_config_error_that_names_the_six(
    name: str,
) -> None:
    with pytest.raises(ConfigError) as caught:
        require_metric(name)

    for valid in METRIC_NAMES:
        assert valid in str(caught.value)


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_a_count_metric_is_returned_as_it_was_given(name: str) -> None:
    assert require_count_metric(name) == name


@pytest.mark.parametrize("name", [*SEGMENT_SCORE_METRICS, "blue", ""])
def test_a_name_that_is_not_a_count_metric_is_refused_naming_the_count_metrics(
    name: str,
) -> None:
    with pytest.raises(ConfigError) as caught:
        require_count_metric(name)

    for valid in COUNT_METRICS:
        assert valid in str(caught.value)


def test_the_registry_imports_nothing_of_the_project_but_the_errors() -> None:
    # It is a table that every layer above reads: it reads nothing of theirs.
    assert project_imports(source_of("rinchen.metrics.registry")) == {"rinchen.errors"}
