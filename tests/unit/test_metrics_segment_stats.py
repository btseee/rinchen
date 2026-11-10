"""The metrics from integer statistics (D13, T7 and T8): the statistics of every segment, the
score of a sum of them (equal, with `==`, to what sacrebleu scores over the whole corpus), the
signature as sacrebleu reports it after the statistics were extracted, the clamp to 100.0, the
tokenizer that follows the target language and the six that are refused by name before sacrebleu
is asked to build them, and the mean seam of the neural metrics through integer micro-units.

Every expected figure of the golden corpus comes from the design spike's pure-Python reference
and sacrebleu's own score-from-statistics (see `tests/support/eval_golden.py`), never from the
code under test. Every test runs under the `no_network` guard: sacrebleu downloads a tokenizer
model on the first use of `flores200`, `flores101`, `spm` or `spBLEU-1K`, and that must never
happen here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import numpy
import pytest
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, UpstreamError
from rinchen.metrics import segment_stats
from rinchen.metrics.bleu_chrf import corpus_scores
from rinchen.metrics.registry import COUNT_METRICS
from rinchen.metrics.segment_stats import (
    ALLOWED_TOKENIZERS,
    BLEU_TOKENIZER_BY_TARGET,
    DEFAULT_TOKENIZER,
    MICRO,
    STATISTICS_WIDTH,
    SegmentStatistics,
    StatisticsMetric,
    mean_from_statistics,
    mean_statistics,
    metric_with_tokenizer,
    score_from_statistics,
    segment_statistics,
    statistics_metric,
    totals_of,
)
from tests.support.eval_golden import (
    ALPHA_ENG,
    ALPHA_KHK,
    BETA_ENG,
    BETA_KHK,
    GOLDEN_ENG,
    GOLDEN_KHK,
)
from tests.support.module_imports import (
    loaded_by_importing,
    project_imports,
    source_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

REFUSED_TOKENIZERS = (
    "flores200",
    "flores101",
    "spm",
    "spBLEU-1K",
    "ja-mecab",
    "ko-mecab",
    "13A",
    "",
)

CHRF_FIRST_SEGMENT = (17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13)
CHRF_FIRST_SEGMENT += (12, 12, 12)
FIRST_SEGMENT = {
    "bleu": (6, 6, 6, 5, 4, 3, 6, 5, 4, 3),
    "chrf": CHRF_FIRST_SEGMENT,
    "chrf_pp": CHRF_FIRST_SEGMENT + (6, 6, 6, 5, 5, 5),
}
CHRF_TOTALS = (147, 180, 144, 138, 170, 130, 129, 160, 116, 120, 150, 102, 111, 140)
CHRF_TOTALS += (88, 102, 130, 75)
TOTALS = {
    "bleu": (42, 50, 40, 26, 16, 8, 42, 33, 24, 15),
    "chrf": CHRF_TOTALS,
    "chrf_pp": CHRF_TOTALS + (42, 50, 40, 33, 40, 26),
}

# (hypotheses, references, target language) and the score of each metric.
GOLDEN_CASES: dict[str, tuple[Sequence[str], Sequence[str], str, dict[str, float]]] = {
    "khk_Cyrl-eng_Latn alpha": (
        ALPHA_ENG,
        GOLDEN_ENG,
        "eng_Latn",
        {
            "bleu": 59.40481339822236,
            "chrf": 72.43951493593927,
            "chrf_pp": 73.08276683393322,
        },
    ),
    "khk_Cyrl-eng_Latn beta": (
        BETA_ENG,
        GOLDEN_ENG,
        "eng_Latn",
        {
            "bleu": 94.41920983915246,
            "chrf": 93.46167452409257,
            "chrf_pp": 93.87300204175257,
        },
    ),
    "eng_Latn-khk_Cyrl alpha": (
        ALPHA_KHK,
        GOLDEN_KHK,
        "khk_Cyrl",
        {
            "bleu": 50.86894326533609,
            "chrf": 67.41404584044409,
            "chrf_pp": 68.06307127183763,
        },
    ),
    "eng_Latn-khk_Cyrl beta": (
        BETA_KHK,
        GOLDEN_KHK,
        "khk_Cyrl",
        {
            "bleu": 91.06781579902749,
            "chrf": 91.06726556207575,
            "chrf_pp": 91.33781025459385,
        },
    ),
}
GOLDEN_PARAMETERS = [
    (case, name) for case in GOLDEN_CASES for name in ("bleu", "chrf", "chrf_pp")
]

BLEU_SIGNATURE = "nrefs:1|case:mixed|eff:no|tok:{tokenizer}|smooth:exp|version:2.6.0"
CHRF_SIGNATURE = "nrefs:1|case:mixed|eff:yes|nc:6|nw:{words}|space:no|version:2.6.0"


@pytest.fixture(autouse=True)
def fresh_scorers() -> Iterator[None]:
    """The scorers are cached per metric; a test that swaps sacrebleu's classes must not leave
    its stand-in in the cache, nor find another test's."""
    segment_stats._scorer.cache_clear()
    yield
    segment_stats._scorer.cache_clear()


def golden_statistics(case: str, name: str) -> SegmentStatistics:
    hypotheses, references, target, _ = GOLDEN_CASES[case]
    return segment_statistics(statistics_metric(name, target), hypotheses, references)


def sacrebleu_score(name: str, case: str) -> float:
    """What sacrebleu itself scores over the whole corpus of the case (clamped as
    `corpus_scores` clamps), through the public API only."""
    hypotheses, references, _, _ = GOLDEN_CASES[case]
    if name == "chrf_pp":
        return float(
            CHRF(word_order=2).corpus_score(list(hypotheses), [list(references)]).score
        )
    scores = corpus_scores(hypotheses, references)
    return scores.bleu if name == "bleu" else scores.chrf


# ---- the constants ----------------------------------------------------------------------


def test_the_constants_of_the_module_are_the_ones_of_the_spec() -> None:
    assert ALLOWED_TOKENIZERS == frozenset({"13a", "zh", "char", "intl", "none"})
    assert dict(BLEU_TOKENIZER_BY_TARGET) == {
        "zho_Hans": "zh",
        "jpn_Jpan": "char",
        "kor_Hang": "char",
    }
    assert DEFAULT_TOKENIZER == "13a"
    assert dict(STATISTICS_WIDTH) == {"bleu": 10, "chrf": 18, "chrf_pp": 24}
    assert MICRO == 1_000_000


def test_every_count_metric_has_a_width_and_every_table_entry_is_valid() -> None:
    assert set(STATISTICS_WIDTH) == set(COUNT_METRICS)
    assert set(BLEU_TOKENIZER_BY_TARGET.values()) <= ALLOWED_TOKENIZERS
    assert set(BLEU_TOKENIZER_BY_TARGET) <= known_language_codes()
    assert DEFAULT_TOKENIZER in ALLOWED_TOKENIZERS


@pytest.mark.parametrize(
    "table", [BLEU_TOKENIZER_BY_TARGET, STATISTICS_WIDTH], ids=["tokenizers", "widths"]
)
def test_the_tables_are_read_only(table: Any) -> None:
    with pytest.raises(TypeError):
        table["eng_Latn"] = "zh"


# ---- the statistics and the scores of the golden corpus ---------------------------------


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_the_first_segment_of_the_golden_corpus_has_the_statistics_of_the_reference(
    name: str,
) -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", name)

    assert statistics.rows[0] == FIRST_SEGMENT[name]
    assert len(statistics.rows) == len(ALPHA_ENG)
    assert {len(row) for row in statistics.rows} == {STATISTICS_WIDTH[name]}


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_the_statistics_of_the_golden_corpus_add_up_to_the_totals_of_the_reference(
    name: str,
) -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", name)

    assert totals_of(statistics.rows) == TOTALS[name]


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_every_cell_of_the_statistics_is_a_python_int(name: str) -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", name)

    assert all(type(cell) is int for row in statistics.rows for cell in row)
    assert isinstance(statistics.rows, tuple)
    assert all(isinstance(row, tuple) for row in statistics.rows)
    assert all(type(cell) is int for cell in totals_of(statistics.rows))


@pytest.mark.parametrize(("case", "name"), GOLDEN_PARAMETERS)
def test_the_score_of_the_summed_statistics_is_the_figure_of_the_reference(
    case: str, name: str
) -> None:
    statistics = golden_statistics(case, name)

    assert (
        score_from_statistics(statistics.metric, totals_of(statistics.rows))
        == GOLDEN_CASES[case][3][name]
    )


@pytest.mark.parametrize(("case", "name"), GOLDEN_PARAMETERS)
def test_the_score_of_the_summed_statistics_is_what_sacrebleu_scores_over_the_corpus(
    case: str, name: str
) -> None:
    statistics = golden_statistics(case, name)

    assert score_from_statistics(
        statistics.metric, totals_of(statistics.rows)
    ) == sacrebleu_score(name, case)


def test_the_corpus_score_is_not_the_mean_of_the_segment_scores() -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", "chrf")
    metric = statistics.metric

    segment_scores = [score_from_statistics(metric, row) for row in statistics.rows]
    corpus = score_from_statistics(metric, totals_of(statistics.rows))

    assert corpus == 72.43951493593927
    assert sum(segment_scores) / len(segment_scores) != corpus


# ---- the signatures ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "tokenizer"),
    [
        ("eng_Latn", "13a"),
        ("khk_Cyrl", "13a"),
        ("zho_Hans", "zh"),
        ("jpn_Jpan", "char"),
        ("kor_Hang", "char"),
    ],
)
def test_the_bleu_signature_names_the_tokenizer_of_the_target(
    target: str, tokenizer: str
) -> None:
    statistics = segment_statistics(
        statistics_metric("bleu", target), GOLDEN_ENG, GOLDEN_ENG
    )

    assert statistics.signature == BLEU_SIGNATURE.format(tokenizer=tokenizer)


def test_the_chrf_signatures_name_the_word_order() -> None:
    chrf = segment_statistics(
        statistics_metric("chrf", "eng_Latn"), ALPHA_ENG, GOLDEN_ENG
    )
    chrf_pp = segment_statistics(
        statistics_metric("chrf_pp", "eng_Latn"), ALPHA_ENG, GOLDEN_ENG
    )

    assert chrf.signature == CHRF_SIGNATURE.format(words=0)
    assert chrf_pp.signature == CHRF_SIGNATURE.format(words=2)


def test_the_signatures_are_the_ones_corpus_scores_reports() -> None:
    scores = corpus_scores(ALPHA_ENG, GOLDEN_ENG)

    bleu = segment_statistics(
        statistics_metric("bleu", "eng_Latn"), ALPHA_ENG, GOLDEN_ENG
    )
    chrf = segment_statistics(
        statistics_metric("chrf", "eng_Latn"), ALPHA_ENG, GOLDEN_ENG
    )

    assert bleu.signature == scores.bleu_signature
    assert chrf.signature == scores.chrf_signature


@pytest.mark.parametrize("tokenizer", sorted(ALLOWED_TOKENIZERS))
def test_every_allowed_tokenizer_builds_offline_and_is_named_in_the_signature(
    tokenizer: str,
) -> None:
    statistics = segment_statistics(
        metric_with_tokenizer("bleu", tokenizer), ["a b c d e"], ["a b c d e"]
    )

    assert statistics.signature == BLEU_SIGNATURE.format(tokenizer=tokenizer)


# ---- the clamp --------------------------------------------------------------------------


def test_identical_text_scores_exactly_100_on_every_metric() -> None:
    # sacrebleu reports 100.00000000000004 for the BLEU of identical text: float rounding.
    for name in COUNT_METRICS:
        statistics = segment_statistics(
            statistics_metric(name, "eng_Latn"), GOLDEN_ENG, GOLDEN_ENG
        )

        assert (
            score_from_statistics(statistics.metric, totals_of(statistics.rows))
            == 100.0
        )


class _FixedScore:
    """A sacrebleu metric that scores every statistics vector `score`."""

    def __init__(self, score: float) -> None:
        self._score = score

    def _compute_score_from_stats(self, stats: Sequence[int]) -> Any:
        return type("Score", (), {"score": self._score})()


@pytest.mark.parametrize(
    ("reported", "returned"),
    [(100.00000000000004, 100.0), (100.0, 100.0), (73.5, 73.5), (0.0, 0.0)],
)
@pytest.mark.parametrize("name", COUNT_METRICS)
def test_a_score_is_clamped_to_100_and_otherwise_returned_as_sacrebleu_has_it(
    monkeypatch: pytest.MonkeyPatch, name: str, reported: float, returned: float
) -> None:
    monkeypatch.setattr(
        segment_stats, "_scorer", lambda name, tokenizer: _FixedScore(reported)
    )
    metric = statistics_metric(name, "eng_Latn")

    assert score_from_statistics(metric, (0,) * STATISTICS_WIDTH[name]) == returned


# ---- the edges --------------------------------------------------------------------------


@pytest.mark.parametrize("name", COUNT_METRICS)
@pytest.mark.parametrize(
    ("hypothesis", "reference"), [("", "abc"), ("abc", ""), ("", "")]
)
def test_an_empty_hypothesis_or_reference_scores_zero_as_sacrebleu_scores_it(
    name: str, hypothesis: str, reference: str
) -> None:
    statistics = segment_statistics(
        statistics_metric(name, "eng_Latn"), [hypothesis], [reference]
    )

    assert score_from_statistics(statistics.metric, totals_of(statistics.rows)) == 0.0


def test_no_segment_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        segment_statistics(statistics_metric("bleu", "eng_Latn"), [], [])


@pytest.mark.parametrize(
    ("hypotheses", "references", "counts"),
    [
        (["X1", "X2"], ["Y1"], ("2", "1")),
        (["X1"], ["Y1", "Y2", "Y3"], ("1", "3")),
    ],
)
def test_unequal_lengths_are_a_value_error_that_names_the_counts_and_never_a_text(
    hypotheses: list[str], references: list[str], counts: tuple[str, str]
) -> None:
    with pytest.raises(ValueError) as caught:
        segment_statistics(
            statistics_metric("chrf", "eng_Latn"), hypotheses, references
        )

    message = str(caught.value)
    assert counts[0] in message
    assert counts[1] in message
    assert not any(text in message for text in (*hypotheses, *references))


def test_the_totals_of_no_row_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        totals_of(())


def test_rows_of_different_widths_have_no_totals() -> None:
    with pytest.raises(ValueError, match="width"):
        totals_of(((1, 2, 3), (1, 2)))


@pytest.mark.parametrize("name", COUNT_METRICS)
@pytest.mark.parametrize("delta", [-1, 1])
def test_a_vector_of_the_wrong_width_cannot_be_scored(name: str, delta: int) -> None:
    metric = statistics_metric(name, "eng_Latn")

    with pytest.raises(ValueError, match=str(STATISTICS_WIDTH[name])):
        score_from_statistics(metric, (1,) * (STATISTICS_WIDTH[name] + delta))


class _GuardedList(list[int]):
    """A list that fails the test when anything but a read touches it."""

    def _refuse(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the caller's totals were changed")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = extend = insert = pop = remove = reverse = sort = clear = _refuse


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_totals_that_are_numpy_integers_score_as_the_python_integers_do(
    name: str,
) -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", name)
    totals = totals_of(statistics.rows)

    assert score_from_statistics(
        statistics.metric, numpy.array(totals, dtype=numpy.int64)
    ) == score_from_statistics(statistics.metric, totals)


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_a_total_that_is_not_an_integer_is_refused_and_never_truncated(
    name: str,
) -> None:
    width = STATISTICS_WIDTH[name]

    with pytest.raises(TypeError):
        score_from_statistics(statistics_metric(name, "eng_Latn"), [1.5] * width)
    with pytest.raises(TypeError):
        score_from_statistics(
            statistics_metric(name, "eng_Latn"), numpy.ones(width, dtype=numpy.float64)
        )


@pytest.mark.parametrize("name", COUNT_METRICS)
def test_scoring_leaves_the_totals_it_was_given_as_they_were(name: str) -> None:
    statistics = golden_statistics("khk_Cyrl-eng_Latn alpha", name)
    totals = totals_of(statistics.rows)
    given = _GuardedList(totals)

    score = score_from_statistics(statistics.metric, given)

    assert tuple(given) == totals
    assert score == GOLDEN_CASES["khk_Cyrl-eng_Latn alpha"][3][name]


def test_scoring_sees_every_digit_of_a_total_beyond_what_a_float32_holds() -> None:
    # 2**24 + 1 is the first integer a float32 cannot hold. A BLEU of totals that large moves
    # in its eighth digit when the totals are rounded through a float32.
    big = 2**24 + 1
    totals = [3 * big + 1, 3 * big, big, big - 1, big - 3, big - 5]
    totals += [3 * big, 3 * big - 1, 3 * big - 2, 3 * big - 3]
    scorer = BLEU()
    exact = float(scorer._compute_score_from_stats(list(totals)).score)
    through_float32 = float(
        scorer._compute_score_from_stats(
            [int(numpy.float32(total)) for total in totals]
        ).score
    )
    assert through_float32 != exact, "the premise of the test"

    assert score_from_statistics(statistics_metric("bleu", "eng_Latn"), totals) == exact


# ---- the tokenizers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "tokenizer"),
    [
        ("zho_Hans", "zh"),
        ("jpn_Jpan", "char"),
        ("kor_Hang", "char"),
        ("eng_Latn", "13a"),
        ("khk_Cyrl", "13a"),
        ("rus_Cyrl", "13a"),
        ("deu_Latn", "13a"),
        ("kaz_Cyrl", "13a"),
        ("tur_Latn", "13a"),
        ("mn_Mong", "13a"),
    ],
)
def test_the_bleu_tokenizer_follows_the_target_language(
    target: str, tokenizer: str
) -> None:
    assert statistics_metric("bleu", target).tokenizer == tokenizer


@pytest.mark.parametrize("name", ["chrf", "chrf_pp"])
@pytest.mark.parametrize("target", ["eng_Latn", "zho_Hans", "jpn_Jpan"])
def test_a_chrf_metric_has_no_tokenizer(name: str, target: str) -> None:
    assert statistics_metric(name, target).tokenizer is None


@pytest.mark.parametrize("tokenizer", REFUSED_TOKENIZERS)
def test_a_tokenizer_outside_the_allowed_set_is_refused_by_name(
    tokenizer: str,
) -> None:
    with pytest.raises(ConfigError) as caught:
        metric_with_tokenizer("bleu", tokenizer)

    assert repr(tokenizer) in str(caught.value)
    for allowed in sorted(ALLOWED_TOKENIZERS):
        assert allowed in str(caught.value)


@pytest.mark.parametrize("tokenizer", sorted(ALLOWED_TOKENIZERS))
def test_an_allowed_tokenizer_is_accepted(tokenizer: str) -> None:
    assert metric_with_tokenizer("bleu", tokenizer) == StatisticsMetric(
        "bleu", tokenizer
    )


@pytest.mark.parametrize("name", ["chrf", "chrf_pp"])
def test_a_tokenizer_is_refused_for_a_chrf_metric(name: str) -> None:
    with pytest.raises(ConfigError, match="no tokenizer"):
        metric_with_tokenizer(name, "13a")


@pytest.mark.parametrize("tokenizer", REFUSED_TOKENIZERS)
def test_a_metric_cannot_be_built_by_hand_with_a_refused_tokenizer(
    tokenizer: str,
) -> None:
    with pytest.raises(ConfigError):
        StatisticsMetric("bleu", tokenizer)


@pytest.mark.parametrize("name", ["comet22", "metricx24_qe", "blue", ""])
def test_a_metric_cannot_be_built_by_hand_with_a_name_that_is_not_a_count_metric(
    name: str,
) -> None:
    with pytest.raises(ConfigError, match="chrf_pp"):
        StatisticsMetric(name, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["comet22", "metricx24", "metricx24_qe", "blue", ""])
def test_a_name_that_is_not_a_count_metric_has_no_statistics(name: str) -> None:
    with pytest.raises(ConfigError, match="chrf_pp"):
        statistics_metric(name, "eng_Latn")
    with pytest.raises(ConfigError, match="chrf_pp"):
        metric_with_tokenizer(name, "13a")


@pytest.mark.parametrize("name", COUNT_METRICS)
@pytest.mark.parametrize("target", ["khk_Mong", "xx_Latn", "", "eng_latn", "khk"])
def test_a_target_outside_the_language_registry_is_refused_by_name(
    name: str, target: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        statistics_metric(name, target)

    assert repr(target) in str(caught.value)
    assert "registry" in str(caught.value)


class _Recorder:
    """A stand-in for sacrebleu's `BLEU` or `CHRF` that records what is asked of it, in order,
    on one list shared by every instance."""

    events: ClassVar[list[str]]

    def __init__(self, **options: object) -> None:
        self.events.append(f"build {sorted(options.items())}")
        # The width of the statistics of the metric that was asked for.
        self.width = {"tokenize": 10, "word_order": 18}[next(iter(options))]
        self.width += 6 if options.get("word_order") == 2 else 0

    def _extract_corpus_statistics(
        self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]
    ) -> list[list[int]]:
        self.events.append("extract")
        return [[1] * self.width for _ in hypotheses]

    def get_signature(self) -> str:
        self.events.append("signature")
        return "recorded"

    def _compute_score_from_stats(self, stats: Sequence[int]) -> Any:
        self.events.append("score")
        return type("Score", (), {"score": 1.0})()


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    recorder = type("Recorder", (_Recorder,), {"events": events})
    monkeypatch.setattr(segment_stats, "BLEU", recorder)
    monkeypatch.setattr(segment_stats, "CHRF", recorder)
    return events


@pytest.mark.parametrize("tokenizer", REFUSED_TOKENIZERS)
def test_a_refused_tokenizer_builds_nothing(
    recorders: list[str], tokenizer: str
) -> None:
    with pytest.raises(ConfigError):
        metric_with_tokenizer("bleu", tokenizer)
    with pytest.raises(ConfigError):
        StatisticsMetric("bleu", tokenizer)

    assert recorders == []


@pytest.mark.parametrize("target", ["khk_Mong", "xx_Latn"])
def test_a_refused_target_builds_nothing(recorders: list[str], target: str) -> None:
    with pytest.raises(ConfigError):
        statistics_metric("bleu", target)

    assert recorders == []


# ---- when sacrebleu is asked ------------------------------------------------------------


def test_each_call_builds_a_fresh_metric_and_reads_its_signature_after_the_statistics(
    recorders: list[str],
) -> None:
    metric = statistics_metric("bleu", "zho_Hans")

    first = segment_statistics(metric, ["a"], ["a"])
    second = segment_statistics(metric, ["b", "c"], ["b", "c"])

    assert recorders == [
        "build [('tokenize', 'zh')]",
        "extract",
        "signature",
        "build [('tokenize', 'zh')]",
        "extract",
        "signature",
    ]
    assert (first.signature, second.signature) == ("recorded", "recorded")


def test_the_chrf_metrics_are_built_with_their_word_order(recorders: list[str]) -> None:
    segment_statistics(statistics_metric("chrf", "eng_Latn"), ["a"], ["a"])
    segment_statistics(statistics_metric("chrf_pp", "eng_Latn"), ["a"], ["a"])

    assert [event for event in recorders if event.startswith("build")] == [
        "build [('word_order', 0)]",
        "build [('word_order', 2)]",
    ]


def test_one_scorer_is_built_per_metric_and_tokenizer_and_kept(
    recorders: list[str],
) -> None:
    # Interleaved, so that a cache that holds only the last scorer builds again and again.
    for _ in range(2):
        score_from_statistics(statistics_metric("bleu", "eng_Latn"), [1] * 10)
        score_from_statistics(statistics_metric("bleu", "zho_Hans"), [1] * 10)
        score_from_statistics(statistics_metric("bleu", "khk_Cyrl"), [1] * 10)
        score_from_statistics(statistics_metric("chrf", "eng_Latn"), [1] * 18)
        score_from_statistics(statistics_metric("chrf_pp", "eng_Latn"), [1] * 24)

    assert [event for event in recorders if event.startswith("build")] == [
        "build [('tokenize', '13a')]",
        "build [('tokenize', 'zh')]",
        "build [('word_order', 0)]",
        "build [('word_order', 2)]",
    ]


def test_a_statistics_matrix_of_the_wrong_shape_is_an_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Narrow(_Recorder):
        events: ClassVar[list[str]] = []

        def _extract_corpus_statistics(
            self, hypotheses: Sequence[str], references: Sequence[Sequence[str]]
        ) -> list[list[int]]:
            return [[1, 2, 3] for _ in hypotheses]

    monkeypatch.setattr(segment_stats, "BLEU", Narrow)

    with pytest.raises(UpstreamError, match="10"):
        segment_statistics(statistics_metric("bleu", "eng_Latn"), ["a"], ["a"])


def test_the_private_methods_the_module_relies_on_are_those_of_the_pinned_sacrebleu() -> (
    None
):
    # A new sacrebleu may rename or change them: the golden figures above then fail too, and
    # this says why.
    assert sacrebleu.__version__ == "2.6.0"
    for metric_class in (BLEU, CHRF):
        assert callable(metric_class._extract_corpus_statistics)
        assert callable(metric_class._compute_score_from_stats)
    rows = BLEU()._extract_corpus_statistics(["a b c d e"], [["a b c d e"]])
    assert rows == [[5, 5, 5, 4, 3, 2, 5, 4, 3, 2]]


# ---- the mean seam ----------------------------------------------------------------------


def test_a_score_becomes_a_pair_of_integer_micro_units_and_a_count() -> None:
    assert mean_statistics((0.912, -0.25, 24.999999)) == (
        (912000, 1),
        (-250000, 1),
        (24999999, 1),
    )


def test_a_micro_unit_is_rounded_and_never_truncated() -> None:
    # 1.001 * 1e6 is 1000999.9999999999 and 0.0157 * 1e6 is 15699.999999999998 in floats.
    assert mean_statistics((1.001, 0.0157, -1.001)) == (
        (1001000, 1),
        (15700, 1),
        (-1001000, 1),
    )


def test_the_scores_of_twelve_segments_have_the_mean_they_have() -> None:
    scores = (0.912, 0.85, 0.77, 0.99, 0.64, 0.88, 0.93, 0.72, 0.81, 0.96, 0.70, 0.87)

    totals = totals_of(mean_statistics(scores))

    assert totals == (10_032_000, 12)
    assert mean_from_statistics(totals) == 0.836
    assert mean_from_statistics((10032000, 12)) == 0.836


@pytest.mark.parametrize("score", [1000.0, -1000.0, 0.0])
def test_a_score_up_to_a_thousand_is_accepted(score: float) -> None:
    assert mean_statistics((score,)) == ((round(score * MICRO), 1),)


@pytest.mark.parametrize(
    "score", [math.nan, math.inf, -math.inf, 1000.5, -1000.5, 1e300]
)
def test_a_score_that_is_not_finite_or_beyond_a_thousand_is_a_value_error(
    score: float,
) -> None:
    with pytest.raises(ValueError, match="score 1"):
        mean_statistics((0.5, score))


@pytest.mark.parametrize("totals", [(5, 0), (5, -1)])
def test_a_mean_over_no_count_is_a_value_error(totals: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="count"):
        mean_from_statistics(totals)


@pytest.mark.parametrize("totals", [(), (5,), (5, 1, 1)])
def test_a_mean_needs_exactly_a_sum_and_a_count(totals: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="width"):
        mean_from_statistics(totals)


# ---- layering ---------------------------------------------------------------------------


def test_the_module_imports_of_the_project_only_the_registries_and_the_errors() -> None:
    assert project_imports(source_of("rinchen.metrics.segment_stats")) == {
        "rinchen.config.languages",
        "rinchen.errors",
        "rinchen.metrics.registry",
    }


def test_importing_the_metrics_loads_no_model_library_and_no_dataset_or_hub_client() -> (
    None
):
    loaded = loaded_by_importing(
        ["rinchen.metrics.registry", "rinchen.metrics.segment_stats"],
        (
            "ctranslate2",
            "sentencepiece",
            "eole",
            "torch",
            "polars",
            "datasets",
            "huggingface_hub",
            "httpx",
            "rinchen.eval",
            "rinchen.train",
            "rinchen.synth",
            "rinchen.serve",
            "rinchen.cli",
            "rinchen.export",
        ),
    )

    assert loaded == []
