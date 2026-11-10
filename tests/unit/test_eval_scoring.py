"""Scoring one direction of D13 (spec T7 to T9): `score_direction` turns a view of the suite and
the outputs of a system into the value, the interval and the signature of every metric, from
per-segment integer statistics grouped into the units of the bootstrap (the document when every
row has one, else the segment).

Every figure is a literal of `tests/support/eval_golden.py`, derived with the pure-Python
reference of the design spike and sacrebleu's own score-from-statistics, never with the code under
test. BLEU's `exp` and `log` come from the C library, so a BLEU interval end is compared at 1e-9
relative; everything else with `==`. A status that is not `ok` is counted, and the text is scored
as it is. The neural metrics are `None` with a reason unless segment scores are handed in; then
their mean is bootstrapped through integer micro-units with the same draws as the count metrics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest

from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output
from rinchen.eval.scoring import (
    BootstrapRecord,
    DirectionRows,
    DirectionScores,
    direction_rows,
    score_direction,
)
from rinchen.eval.significance import compare
from rinchen.eval.suites import DirectionView, Suite, SuiteRow
from rinchen.metrics.registry import (
    COUNT_METRICS,
    HIGHER_IS_BETTER,
    METRIC_NAMES,
    SEGMENT_SCORE_METRICS,
)
from rinchen.metrics.segment_stats import STATISTICS_WIDTH, score_from_statistics
from tests.support.eval_fakes import golden_outputs, golden_suite
from tests.support.eval_golden import (
    GOLDEN_BOOTSTRAP_DOCUMENT,
    GOLDEN_BOOTSTRAP_SEGMENT,
    GOLDEN_COUNT_METRICS,
    GOLDEN_DIRECTIONS,
    GOLDEN_DOCS,
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_NO_SCORER_REASON,
    GOLDEN_PAIRED,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SIGNATURES,
    GOLDEN_SYSTEMS,
    MEAN_INTERVAL_A,
    MEAN_MICRO_A,
    MEAN_POINT_A,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = BootstrapSettings()
MARKER = "SECRET-MARKER-61"
KEYS = [
    (direction, system, metric)
    for direction in GOLDEN_DIRECTIONS
    for system in GOLDEN_SYSTEMS
    for metric in GOLDEN_COUNT_METRICS
]


def outputs_in_order(system: str, direction: str) -> list[Output]:
    return [
        output
        for key, output in golden_outputs(system).items()
        if key.direction == direction
    ]


def without_documents(suite: Suite) -> Suite:
    return dataclasses.replace(
        suite,
        rows=tuple(SuiteRow(row.id, None, row.text) for row in suite.rows),
    )


def view_of(direction: str, *, documents: bool = True) -> DirectionView:
    suite = golden_suite() if documents else without_documents(golden_suite())
    return suite.view(direction)


def score(direction: str, system: str, *, documents: bool = True) -> DirectionScores:
    return score_direction(
        view_of(direction, documents=documents),
        outputs_in_order(system, direction),
        SETTINGS,
    )


def assert_interval(
    actual: Sequence[float] | None, expected: tuple[float, float], metric: str
) -> None:
    assert actual is not None
    if metric == "bleu":
        assert list(actual) == pytest.approx(list(expected), rel=1e-9)
    else:
        assert list(actual) == list(expected)


def free_view(
    src: str, tgt: str, sources: Sequence[str], references: Sequence[str]
) -> DirectionView:
    """A view over made-up rows, without documents (a suite that has none)."""
    return DirectionView(
        direction=f"{src}-{tgt}",
        src_lang=src,
        tgt_lang=tgt,
        ids=tuple(f"r{number}" for number in range(len(sources))),
        docs=None,
        sources=tuple(sources),
        references=tuple(references),
    )


# ---- the golden suite ---------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS, ids=" ".join)
def test_the_document_unit_gives_the_golden_point_and_interval(
    key: tuple[str, str, str],
) -> None:
    direction, system, metric = key

    scores = score(direction, system)

    assert scores.metrics[metric] == GOLDEN_SEGMENT_ESTIMATES[key].point
    assert_interval(scores.ci95[metric], GOLDEN_DOCUMENT_INTERVALS[key], metric)


@pytest.mark.parametrize("key", KEYS, ids=" ".join)
def test_without_documents_the_segment_is_the_unit_and_gives_the_segment_interval(
    key: tuple[str, str, str],
) -> None:
    direction, system, metric = key
    golden = GOLDEN_SEGMENT_ESTIMATES[key]

    scores = score(direction, system, documents=False)

    assert scores.metrics[metric] == golden.point
    assert_interval(scores.ci95[metric], (golden.low, golden.high), metric)


@pytest.mark.parametrize("direction", GOLDEN_DIRECTIONS)
@pytest.mark.parametrize("system", GOLDEN_SYSTEMS)
def test_the_bootstrap_record_says_which_unit_and_how_many(
    direction: str, system: str
) -> None:
    with_documents = score(direction, system)
    without = score(direction, system, documents=False)

    assert dataclasses.asdict(with_documents.bootstrap) == GOLDEN_BOOTSTRAP_DOCUMENT
    assert dataclasses.asdict(without.bootstrap) == GOLDEN_BOOTSTRAP_SEGMENT


def test_the_record_of_the_bootstrap_follows_the_settings_that_were_given() -> None:
    settings = BootstrapSettings(resamples=500, seed=7, alpha=0.1)

    scores = score_direction(
        view_of("khk_Cyrl-eng_Latn"),
        outputs_in_order("alpha", "khk_Cyrl-eng_Latn"),
        settings,
    )

    assert scores.bootstrap == BootstrapRecord(
        resamples=500,
        seed=7,
        alpha=0.1,
        interval="percentile",
        rng="splitmix64-counter/1",
        unit="document",
        units=5,
    )
    assert scores.ci95["chrf_pp"] != score("khk_Cyrl-eng_Latn", "alpha").ci95["chrf_pp"]


@pytest.mark.parametrize("direction", GOLDEN_DIRECTIONS)
def test_the_signatures_are_the_ones_sacrebleu_reports(direction: str) -> None:
    scores = score(direction, "alpha")

    assert dict(scores.signatures) == dict(GOLDEN_SIGNATURES)


def test_the_counts_the_statuses_and_the_unmeasured_metrics_of_a_plain_run() -> None:
    scores = score("khk_Cyrl-eng_Latn", "alpha")

    assert scores.n_segments == 10
    assert dict(scores.statuses) == {"ok": 10}
    assert list(scores.metrics) == list(METRIC_NAMES)
    assert list(scores.ci95) == list(METRIC_NAMES)
    for name in SEGMENT_SCORE_METRICS:
        assert scores.metrics[name] is None
        assert scores.ci95[name] is None
        assert scores.unmeasured[name] == GOLDEN_NO_SCORER_REASON
    assert list(scores.unmeasured) == list(SEGMENT_SCORE_METRICS)


def test_a_count_metric_is_never_unmeasured_and_never_written_as_zero() -> None:
    scores = score("khk_Cyrl-eng_Latn", "alpha")

    for name in COUNT_METRICS:
        assert name not in scores.unmeasured
        assert scores.metrics[name] is not None
        assert scores.metrics[name] > 0.0
    # The system that wrote nothing right scores 0.0 as a number, not as a missing value.
    blank = score_direction(
        view_of("khk_Cyrl-eng_Latn"),
        [Output("", "no_output")] * 10,
        SETTINGS,
    )
    assert blank.metrics["chrf_pp"] == 0.0
    assert "chrf_pp" not in blank.unmeasured


def test_the_values_leave_the_module_as_python_floats_and_lists() -> None:
    scores = score("khk_Cyrl-eng_Latn", "alpha")

    for name in COUNT_METRICS:
        assert type(scores.metrics[name]) is float
        interval = scores.ci95[name]
        assert type(interval) is list
        assert [type(end) for end in interval] == [float, float]


def test_the_scores_cannot_be_changed_through_the_mappings() -> None:
    scores = score("khk_Cyrl-eng_Latn", "alpha")

    for mapping in (
        scores.metrics,
        scores.ci95,
        scores.signatures,
        scores.unmeasured,
        scores.statuses,
    ):
        with pytest.raises(TypeError):
            mapping["bleu"] = 0  # type: ignore[index]


# ---- a status that is not ok ----------------------------------------------------------------


def test_a_status_that_is_not_ok_is_counted_and_the_text_is_scored_as_it_is() -> None:
    plain = outputs_in_order("alpha", "khk_Cyrl-eng_Latn")
    flagged = [
        Output(output.text, status)
        for output, status in zip(
            plain,
            ["truncated", "ok", "ok", "no_output", "ok", "ok", "truncated"]
            + ["ok"] * 3,
            strict=True,
        )
    ]

    scores = score_direction(view_of("khk_Cyrl-eng_Latn"), flagged, SETTINGS)

    assert dict(scores.statuses) == {"no_output": 1, "ok": 7, "truncated": 2}
    assert list(scores.statuses) == ["no_output", "ok", "truncated"]
    reference = score("khk_Cyrl-eng_Latn", "alpha")
    assert scores.metrics == reference.metrics
    assert scores.ci95 == reference.ci95


def test_every_status_of_the_engine_is_counted() -> None:
    statuses = ["ok", "empty", "too_long", "truncated", "no_output"] * 2
    outputs = [
        Output(output.text, status)  # type: ignore[arg-type]
        for output, status in zip(
            outputs_in_order("beta", "eng_Latn-khk_Cyrl"), statuses, strict=True
        )
    ]

    scores = score_direction(view_of("eng_Latn-khk_Cyrl"), outputs, SETTINGS)

    assert dict(scores.statuses) == {
        "empty": 2,
        "no_output": 2,
        "ok": 2,
        "too_long": 2,
        "truncated": 2,
    }


# ---- the tokenizer follows the target ---------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "tgt", "tokenizer"),
    [
        ("eng_Latn", "zho_Hans", "zh"),
        ("eng_Latn", "jpn_Jpan", "char"),
        ("eng_Latn", "kor_Hang", "char"),
        ("zho_Hans", "eng_Latn", "13a"),
        ("jpn_Jpan", "khk_Cyrl", "13a"),
        ("kor_Hang", "rus_Cyrl", "13a"),
        ("khk_Cyrl", "mn_Mong", "13a"),
        ("eng_Latn", "tur_Latn", "13a"),
    ],
)
def test_the_tokenizer_of_bleu_follows_the_target_language_and_never_the_source(
    src: str, tgt: str, tokenizer: str
) -> None:
    view = free_view(src, tgt, ["a b", "c d"], ["a b", "c e"])

    scores = score_direction(view, [Output("a b", "ok"), Output("c d", "ok")], SETTINGS)

    assert scores.signatures["bleu"].split("|")[3] == f"tok:{tokenizer}"
    assert scores.signatures["chrf"] == GOLDEN_SIGNATURES["chrf"]


# ---- units ----------------------------------------------------------------------------------


def test_one_segment_has_a_value_and_no_interval() -> None:
    view = free_view("khk_Cyrl", "eng_Latn", ["x"], ["the cat sat on the mat"])

    scores = score_direction(view, [Output("the cat sat on the mat", "ok")], SETTINGS)

    assert scores.n_segments == 1
    assert scores.metrics["chrf_pp"] == 100.0
    assert scores.metrics["bleu"] == 100.0
    assert all(scores.ci95[name] is None for name in METRIC_NAMES)
    assert scores.bootstrap.units == 1
    assert scores.bootstrap.unit == "segment"


def test_one_document_is_one_unit_and_has_no_interval() -> None:
    suite = golden_suite()
    one_document = dataclasses.replace(
        suite, rows=tuple(SuiteRow(r.id, "only", r.text) for r in suite.rows)
    )

    scores = score_direction(
        one_document.view("khk_Cyrl-eng_Latn"),
        outputs_in_order("alpha", "khk_Cyrl-eng_Latn"),
        SETTINGS,
    )

    assert scores.n_segments == 10
    assert scores.bootstrap.unit == "document"
    assert scores.bootstrap.units == 1
    assert scores.ci95["chrf_pp"] is None
    assert (
        scores.metrics["chrf_pp"]
        == GOLDEN_SEGMENT_ESTIMATES[("khk_Cyrl-eng_Latn", "alpha", "chrf_pp")].point
    )


def test_no_segment_is_a_value_error() -> None:
    view = free_view("khk_Cyrl", "eng_Latn", [], [])

    with pytest.raises(ValueError, match="at least one segment"):
        score_direction(view, [], SETTINGS)


def test_outputs_of_another_length_are_a_value_error_that_names_the_counts() -> None:
    view = view_of("khk_Cyrl-eng_Latn")
    outputs = outputs_in_order("alpha", "khk_Cyrl-eng_Latn")

    for wrong in (outputs[:9], [*outputs, outputs[0]]):
        with pytest.raises(ValueError, match=rf"{len(wrong)} outputs.*10 segments"):
            score_direction(view, wrong, SETTINGS)


def test_the_messages_never_hold_a_text() -> None:
    view = free_view("khk_Cyrl", "eng_Latn", [MARKER, MARKER], [MARKER, MARKER])

    with pytest.raises(ValueError) as caught:
        score_direction(view, [Output(MARKER, "ok")], SETTINGS)

    assert MARKER not in str(caught.value)


# ---- the neural seam ---------------------------------------------------------------------------


def twelve_row_view() -> DirectionView:
    texts = [f"sentence number {number}" for number in range(12)]
    return free_view("khk_Cyrl", "eng_Latn", texts, texts)


def twelve_outputs() -> list[Output]:
    return [Output(f"sentence number {number}", "ok") for number in range(12)]


def comet_scores() -> list[float]:
    return [micro / 1_000_000 for micro in MEAN_MICRO_A]


def test_comet_is_the_mean_of_the_segment_scores_with_the_interval_of_the_golden() -> (
    None
):
    scores = score_direction(
        twelve_row_view(),
        twelve_outputs(),
        SETTINGS,
        segment_scores={"comet22": comet_scores()},
    )

    assert scores.metrics["comet22"] == MEAN_POINT_A == 0.836
    assert scores.ci95["comet22"] == list(MEAN_INTERVAL_A) == [0.7785, 0.89]
    assert scores.bootstrap.units == 12
    assert scores.bootstrap.unit == "segment"
    for name in ("metricx24", "metricx24_qe"):
        assert scores.metrics[name] is None
        assert scores.ci95[name] is None
        assert scores.unmeasured[name] == GOLDEN_NO_SCORER_REASON
    assert "comet22" not in scores.unmeasured


def test_metricx_scores_are_accepted_and_the_score_does_not_consult_the_direction() -> (
    None
):
    scores = score_direction(
        twelve_row_view(),
        twelve_outputs(),
        SETTINGS,
        segment_scores={"metricx24": comet_scores(), "metricx24_qe": comet_scores()},
    )

    assert scores.metrics["metricx24"] == scores.metrics["metricx24_qe"] == 0.836
    assert scores.ci95["metricx24"] == [0.7785, 0.89]
    # Higher or lower is better is the comparison's business, never the score's.
    assert HIGHER_IS_BETTER["metricx24"] is False


def test_scores_for_part_of_the_segments_are_a_value_error() -> None:
    for wrong in (comet_scores()[:11], [*comet_scores(), 0.5], []):
        with pytest.raises(ValueError, match=rf"{len(wrong)} scores.*12 segments"):
            score_direction(
                twelve_row_view(),
                twelve_outputs(),
                SETTINGS,
                segment_scores={"comet22": wrong},
            )


def test_a_name_that_is_no_segment_score_metric_is_a_value_error() -> None:
    for name in ("bleu", "chrf_pp", "comet", ""):
        with pytest.raises(ValueError, match="segment scores"):
            score_direction(
                twelve_row_view(),
                twelve_outputs(),
                SETTINGS,
                segment_scores={name: comet_scores()},
            )


def test_a_score_that_is_not_finite_or_too_large_is_a_value_error() -> None:
    for bad in (float("nan"), float("inf"), 1000.5):
        scores = comet_scores()
        scores[3] = bad
        with pytest.raises(ValueError, match="score 3"):
            score_direction(
                twelve_row_view(),
                twelve_outputs(),
                SETTINGS,
                segment_scores={"comet22": scores},
            )


def test_a_mean_over_documents_is_the_mean_of_the_segments_not_of_the_documents() -> (
    None
):
    # Documents of one and of three sentences: the mean of the document means would be 0.5,
    # the mean of the four segment scores is 0.25.
    texts = [f"t {number}" for number in range(4)]
    suite = Suite(
        name="docs",
        kind="fixture",
        benchmark="docs",
        split="devtest",
        source=golden_suite().source,
        directions=("khk_Cyrl-eng_Latn",),
        rows=tuple(
            SuiteRow(f"r{number}", doc, {"khk_Cyrl": text, "eng_Latn": text})
            for number, (doc, text) in enumerate(
                zip(["a", "b", "b", "b"], texts, strict=True)
            )
        ),
    )

    scores = score_direction(
        suite.view("khk_Cyrl-eng_Latn"),
        [Output(text, "ok") for text in texts],
        SETTINGS,
        segment_scores={"comet22": [1.0, 0.0, 0.0, 0.0]},
    )

    assert scores.metrics["comet22"] == 0.25
    assert scores.bootstrap.unit == "document"
    assert scores.bootstrap.units == 2
    # Two units, so a resample is two documents: both the lone one (1.0), both of the long
    # one (0.0) or one of each (0.25), a quarter, a quarter and a half of the draws: the ends
    # of the interval are 0.0 and 1.0 (over four segment units the top would be 0.75).
    assert scores.ci95["comet22"] == [0.0, 1.0]


# ---- the rows of a direction -------------------------------------------------------------------


@pytest.mark.parametrize("direction", GOLDEN_DIRECTIONS)
def test_direction_rows_has_the_integer_statistics_of_every_count_metric(
    direction: str,
) -> None:
    rows = direction_rows(view_of(direction), outputs_in_order("alpha", direction))

    assert isinstance(rows, DirectionRows)
    assert rows.unit == "document"
    assert rows.labels == ("d1", "d2", "d3", "d4", "d5")
    assert list(rows.metrics) == list(COUNT_METRICS)
    for name, metric_rows in rows.metrics.items():
        assert metric_rows.rows.dtype == np.int64
        assert metric_rows.rows.shape == (5, STATISTICS_WIDTH[name])
        assert metric_rows.signature == GOLDEN_SIGNATURES[name]
        assert metric_rows.metric.name == name


def test_direction_rows_without_documents_has_one_row_per_segment_in_file_order() -> (
    None
):
    rows = direction_rows(
        view_of("khk_Cyrl-eng_Latn", documents=False),
        outputs_in_order("alpha", "khk_Cyrl-eng_Latn"),
    )

    assert rows.unit == "segment"
    assert rows.labels == tuple(str(number) for number in range(10))
    # The first segment's chrF statistics, from the golden corpus of Task 1.
    assert rows.metrics["chrf"].rows[0].tolist() == [
        17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12,
    ]  # fmt: skip
    assert rows.metrics["bleu"].rows[0].tolist() == [6, 6, 6, 5, 4, 3, 6, 5, 4, 3]


def test_the_rows_of_two_systems_compare_as_the_golden_paired_comparison() -> None:
    # What the report does: the rows of both systems of a direction, one set of draws.
    for direction in GOLDEN_DIRECTIONS:
        for unit, documents in (("segment", False), ("document", True)):
            alpha = direction_rows(
                view_of(direction, documents=documents),
                outputs_in_order("alpha", direction),
            ).metrics["chrf_pp"]
            beta = direction_rows(
                view_of(direction, documents=documents),
                outputs_in_order("beta", direction),
            ).metrics["chrf_pp"]
            golden = GOLDEN_PAIRED[direction, unit]

            result = compare(
                beta.rows,
                alpha.rows,
                lambda totals, metric=alpha.metric: score_from_statistics(
                    metric, totals
                ),
                SETTINGS,
            )

            assert result.delta == golden.delta
            assert result.interval == (golden.low, golden.high)
            assert result.p_value == golden.p_value
            assert result.better_fraction == golden.better_fraction


def test_direction_rows_refuses_what_score_direction_refuses() -> None:
    view = view_of("khk_Cyrl-eng_Latn")

    with pytest.raises(ValueError, match="9 outputs"):
        direction_rows(view, outputs_in_order("alpha", "khk_Cyrl-eng_Latn")[:9])
    with pytest.raises(ValueError, match="at least one segment"):
        direction_rows(free_view("khk_Cyrl", "eng_Latn", [], []), [])


def test_the_documents_of_the_golden_suite_are_the_ones_of_the_literals() -> None:
    assert view_of("khk_Cyrl-eng_Latn").docs == GOLDEN_DOCS
