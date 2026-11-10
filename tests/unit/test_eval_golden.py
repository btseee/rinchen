"""The golden literals of D13 in one file: the parity check D14's maintenance loop reruns after a
bump of sacrebleu or numpy (the runbook names it). Everything here is a literal derived with the
pure-Python reference of the design spike and sacrebleu's own score-from-statistics, never with
the code under test, and each one is reached through the public path a result takes:

- Task 1, the metrics: the integer statistics of the golden corpus (the first segment and the
  totals), the score of the sums for four corpora and three metrics against the figures AND
  against sacrebleu's `corpus_score`, and the signatures;
- Task 2, the bootstrap: the published splitmix64 vectors through the index rule, the draws of
  the default seed, the intervals of the golden suite over the segments and over the documents,
  the paired comparison and the mean-type metric;
- Task 6, the result: the golden `result.json`, byte for byte.

A figure that moves with the machine is BLEU's last digit (the C library's `exp` and `log`), so a
BLEU interval end is compared at 1e-9 relative and BLEU's point estimate with `==` (it is exact
on the machine of the spike; the failure on another machine is the intended alarm).
Regenerating a golden after an intended change: check each changed literal independently first.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
from sacrebleu.metrics import BLEU, CHRF

from rinchen.eval.bootstrap import (
    BootstrapSettings,
    draw_index,
    interval_positions,
    resample_totals,
)
from rinchen.eval.outputs import Output
from rinchen.eval.result import result_from_bytes
from rinchen.eval.scoring import direction_rows, score_direction
from rinchen.eval.significance import compare, estimate
from rinchen.eval.suites import DirectionView, SuiteRow
from rinchen.metrics.segment_stats import (
    score_from_statistics,
    segment_statistics,
    statistics_metric,
    totals_of,
)
from tests.support.eval_fakes import golden_outputs, golden_suite
from tests.support.eval_golden import (
    ALPHA_ENG,
    GOLDEN_COUNT_METRICS,
    GOLDEN_DIRECTIONS,
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_ENG,
    GOLDEN_PAIRED,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SIGNATURES,
    GOLDEN_SYSTEMS,
    MEAN_INTERVAL_A,
    MEAN_MICRO_A,
    MEAN_MICRO_B,
    MEAN_P_VALUE,
    MEAN_POINT_A,
    MEAN_SUMS_A,
)
from tests.support.eval_result_documents import GOLDEN_RESULT_FILE
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = BootstrapSettings()
CHRF_TOTALS = (147, 180, 144, 138, 170, 130, 129, 160, 116, 120, 150, 102, 111, 140)
CHRF_TOTALS += (88, 102, 130, 75)
FIRST_SEGMENT = {
    "bleu": (6, 6, 6, 5, 4, 3, 6, 5, 4, 3),
    "chrf": (17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12),
    "chrf_pp": (17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12)
    + (6, 6, 6, 5, 5, 5),
}
TOTALS = {
    "bleu": (42, 50, 40, 26, 16, 8, 42, 33, 24, 15),
    "chrf": CHRF_TOTALS,
    "chrf_pp": CHRF_TOTALS + (42, 50, 40, 33, 40, 26),
}
RESULT_FILE_SHA256 = "05212b4fdd52448d4981e5e21f3f6b9315f16d7cfd38fd7c195d62172eb08dde"


def outputs_of(system: str, direction: str) -> list[Output]:
    return [o for k, o in golden_outputs(system).items() if k.direction == direction]


def view_of(direction: str, *, documents: bool = True) -> DirectionView:
    suite = golden_suite()
    if not documents:
        suite = dataclasses.replace(
            suite, rows=tuple(SuiteRow(r.id, None, r.text) for r in suite.rows)
        )
    return suite.view(direction)


def close(
    actual: list[float] | tuple[float, float],
    expected: tuple[float, float],
    metric: str,
) -> None:
    if metric == "bleu":
        assert list(actual) == pytest.approx(list(expected), rel=1e-9)
    else:
        assert list(actual) == list(expected)


# ---- Task 1: the metrics ---------------------------------------------------------------------


@pytest.mark.parametrize("name", GOLDEN_COUNT_METRICS)
def test_the_statistics_of_the_golden_corpus_are_the_reference_ones(name: str) -> None:
    metric = statistics_metric(name, "eng_Latn")

    rows = segment_statistics(metric, ALPHA_ENG, GOLDEN_ENG).rows

    assert rows[0] == FIRST_SEGMENT[name]
    assert totals_of(rows) == TOTALS[name]


@pytest.mark.parametrize(
    ("direction", "system", "name"),
    [
        (d, s, m)
        for d in GOLDEN_DIRECTIONS
        for s in GOLDEN_SYSTEMS
        for m in GOLDEN_COUNT_METRICS
    ],
    ids=" ".join,
)
def test_the_twelve_scores_are_the_reference_ones_and_sacrebleus_own(
    direction: str, system: str, name: str
) -> None:
    view = view_of(direction)
    hypotheses = [o.text for o in outputs_of(system, direction)]
    metric = statistics_metric(name, view.tgt_lang)
    rows = segment_statistics(metric, hypotheses, view.references).rows
    figure = GOLDEN_SEGMENT_ESTIMATES[direction, system, name].point

    assert score_from_statistics(metric, totals_of(rows)) == figure
    scorer = (
        BLEU() if name == "bleu" else CHRF(word_order=2 if name == "chrf_pp" else 0)
    )
    assert (
        min(
            100.0, float(scorer.corpus_score(hypotheses, [list(view.references)]).score)
        )
        == figure
    )


def test_the_signatures_are_the_ones_of_the_pinned_sacrebleu() -> None:
    scores = score_direction(
        view_of("khk_Cyrl-eng_Latn"), outputs_of("alpha", "khk_Cyrl-eng_Latn"), SETTINGS
    )

    assert dict(scores.signatures) == dict(GOLDEN_SIGNATURES)
    assert GOLDEN_SIGNATURES["bleu"].endswith("version:2.6.0")


# ---- Task 2: the bootstrap ---------------------------------------------------------------------


def test_the_index_rule_reproduces_the_published_splitmix64_words() -> None:
    # The high 32 bits of 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F (seed 0).
    assert [draw_index(0, 0, j, 2**32) for j in range(3)] == [
        3793791033,
        1853398634,
        113532184,
    ]
    assert [draw_index(0, 0, j, 1000) for j in range(5)] == [883, 431, 26, 970, 106]


def test_the_draws_of_the_default_seed_are_the_reference_ones() -> None:
    seed = SETTINGS.seed
    assert [draw_index(seed, 0, j, 7) for j in range(7)] == [5, 2, 4, 4, 2, 3, 2]
    assert [draw_index(seed, 1, j, 7) for j in range(7)] == [0, 6, 2, 1, 2, 4, 1]
    assert [draw_index(seed, 999, j, 7) for j in range(7)] == [5, 2, 6, 0, 2, 1, 4]
    assert [draw_index(seed, 0, j, 1012) for j in range(5)] == [788, 407, 682, 697, 335]


def test_the_interval_sits_at_positions_25_and_974() -> None:
    assert interval_positions(1000, 0.05) == (25, 974)


def mean_rows(micro: tuple[int, ...]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


def mean_score(totals: list[int]) -> float:
    return totals[0] / (totals[1] * 1_000_000)


def test_the_mean_type_golden_has_the_reference_sums_the_interval_and_the_p_value() -> (
    None
):
    sums = sorted(resample_totals(mean_rows(MEAN_MICRO_A), SETTINGS)[:, 0].tolist())

    assert (sums[25], sums[974]) == MEAN_SUMS_A
    result = estimate(mean_rows(MEAN_MICRO_A), mean_score, SETTINGS)
    assert (result.value, result.interval) == (MEAN_POINT_A, MEAN_INTERVAL_A)
    paired = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), mean_score, SETTINGS
    )
    assert paired.p_value == MEAN_P_VALUE


# ---- the two together: what a result and a report hold -----------------------------------------


@pytest.mark.parametrize("direction", GOLDEN_DIRECTIONS)
@pytest.mark.parametrize("system", GOLDEN_SYSTEMS)
def test_the_intervals_of_the_golden_suite_over_its_documents(
    direction: str, system: str
) -> None:
    scores = score_direction(
        view_of(direction), outputs_of(system, direction), SETTINGS
    )

    for name in GOLDEN_COUNT_METRICS:
        assert (
            scores.metrics[name]
            == GOLDEN_SEGMENT_ESTIMATES[direction, system, name].point
        )
        interval = scores.ci95[name]
        assert interval is not None
        close(interval, GOLDEN_DOCUMENT_INTERVALS[direction, system, name], name)
    assert scores.bootstrap.unit == "document"
    assert scores.bootstrap.units == 5


@pytest.mark.parametrize("direction", GOLDEN_DIRECTIONS)
@pytest.mark.parametrize("unit", ["segment", "document"])
def test_the_paired_comparison_of_the_two_golden_systems(
    direction: str, unit: str
) -> None:
    view = view_of(direction, documents=unit == "document")
    beta = direction_rows(view, outputs_of("beta", direction)).metrics["chrf_pp"]
    alpha = direction_rows(view, outputs_of("alpha", direction)).metrics["chrf_pp"]
    golden = GOLDEN_PAIRED[direction, unit]

    result = compare(
        beta.rows,
        alpha.rows,
        lambda totals: score_from_statistics(alpha.metric, totals),
        SETTINGS,
    )

    assert (result.delta, result.interval) == (golden.delta, (golden.low, golden.high))
    assert (result.p_value, result.better_fraction) == (
        golden.p_value,
        golden.better_fraction,
    )


# ---- Task 6: the golden result ---------------------------------------------------------------


def test_the_golden_result_file_is_the_one_that_was_checked_independently() -> None:
    data = GOLDEN_RESULT_FILE.read_bytes()

    assert hashlib.sha256(data).hexdigest() == RESULT_FILE_SHA256
    result = result_from_bytes(data)
    assert (result.system, result.suite, result.direction) == (
        "alpha",
        "golden",
        "khk_Cyrl-eng_Latn",
    )
    assert (
        result.metrics["chrf_pp"]
        == GOLDEN_SEGMENT_ESTIMATES["khk_Cyrl-eng_Latn", "alpha", "chrf_pp"].point
    )
