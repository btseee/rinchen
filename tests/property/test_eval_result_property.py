"""Property tests of the scoring and the documents of D13 (Task 6): the statuses of the outputs
never move a score, what is scored is written and read back unchanged and with the same bytes, and
one flipped byte of a golden document (`result.json`, `run.json`, a scores file) is a document or
a `DataError` and never another exception (the readers are strict, not fragile). Under the
`no_network` guard like every D13 test module."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.errors import DataError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.result import build_result, result_from_bytes, run_from_bytes
from rinchen.eval.scoring import score_direction
from rinchen.eval.segment_scores import read_scores, scores_to_bytes
from rinchen.eval.systems import SystemRecord
from tests.support.eval_fakes import (
    OUTPUTS_SHA256,
    SUITE_SHA256,
    golden_outputs,
    golden_suite,
    no_network,  # noqa: F401 - the fixture of the mark
)
from tests.support.eval_result_documents import (
    GOLDEN_RESULT_FILE,
    golden_result_document,
    run_document,
    serialised,
)

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=60,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
FAST = BootstrapSettings(resamples=100)
DIRECTION = "khk_Cyrl-eng_Latn"
STATUSES = ("ok", "empty", "too_long", "truncated", "no_output")
TEXTS = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="\n"), max_size=30
)
VIEW = golden_suite().view(DIRECTION)
GOLDEN_RESULT = GOLDEN_RESULT_FILE.read_bytes()
GOLDEN_RUN = serialised(run_document())
EXPECTED = list(golden_outputs("alpha"))
GOLDEN_SCORES = scores_to_bytes(
    "comet22",
    SUITE_SHA256,
    OUTPUTS_SHA256,
    {key: 0.5 + index / 100 for index, key in enumerate(EXPECTED)},
)


@given(
    st.lists(TEXTS, min_size=10, max_size=10),
    st.lists(st.sampled_from(STATUSES), min_size=10, max_size=10),
)
@SETTINGS
def test_the_statuses_are_counted_and_never_move_a_score(
    texts: list[str], statuses: list[str]
) -> None:
    plain = score_direction(VIEW, [Output(t, "ok") for t in texts], FAST)
    flagged = score_direction(
        VIEW,
        [Output(t, s) for t, s in zip(texts, statuses, strict=True)],  # type: ignore[arg-type]
        FAST,
    )

    assert flagged.metrics == plain.metrics
    assert flagged.ci95 == plain.ci95
    assert sum(flagged.statuses.values()) == 10
    assert dict(flagged.statuses) == {
        s: statuses.count(s) for s in sorted(set(statuses))
    }


@given(st.lists(TEXTS, min_size=10, max_size=10))
@SETTINGS
def test_what_is_scored_is_written_and_read_back_with_the_same_bytes(
    texts: list[str],
) -> None:
    scores = score_direction(VIEW, [Output(t, "ok") for t in texts], FAST)
    result = build_result(
        scores=scores,
        system=SystemRecord(
            kind="hypotheses", name="alpha", outputs_sha256=OUTPUTS_SHA256
        ),
        suite_name="golden",
        suite_sha256=SUITE_SHA256,
        direction=DIRECTION,
        outputs_sha256=OUTPUTS_SHA256,
        trained=True,
    )

    data = result.to_bytes()

    assert result_from_bytes(data) == result
    assert result_from_bytes(data).to_bytes() == data
    assert result.metrics["chrf_pp"] is not None
    assert 0.0 <= result.metrics["chrf_pp"] <= 100.0


@given(st.integers(0, len(GOLDEN_RESULT) - 1), st.integers(0, 255))
@SETTINGS
def test_one_changed_byte_of_the_golden_result_is_a_result_or_a_data_error(
    index: int, byte: int
) -> None:
    mutated = GOLDEN_RESULT[:index] + bytes([byte]) + GOLDEN_RESULT[index + 1 :]

    try:
        result = result_from_bytes(mutated)
    except DataError:
        return
    assert result.n_segments >= 1
    assert result_from_bytes(result.to_bytes()) == result


@given(st.integers(0, len(GOLDEN_RUN) - 1), st.integers(0, 255))
@SETTINGS
def test_one_changed_byte_of_a_run_is_a_run_or_a_data_error(
    index: int, byte: int
) -> None:
    mutated = GOLDEN_RUN[:index] + bytes([byte]) + GOLDEN_RUN[index + 1 :]

    try:
        record = run_from_bytes(mutated)
    except DataError:
        return
    assert run_from_bytes(record.to_bytes()) == record


@given(st.integers(0, len(GOLDEN_SCORES) - 1), st.integers(0, 255))
@SETTINGS
def test_one_changed_byte_of_a_scores_file_is_scores_or_a_data_error(
    index: int, byte: int
) -> None:
    mutated = GOLDEN_SCORES[:index] + bytes([byte]) + GOLDEN_SCORES[index + 1 :]

    try:
        found = read_scores(
            mutated,
            metric="comet22",
            suite_sha256=SUITE_SHA256,
            hypotheses_sha256=OUTPUTS_SHA256,
            expected=EXPECTED,
        )
    except DataError:
        return
    assert list(found) == EXPECTED
    assert all(abs(score) <= 1000 for score in found.values())


@given(st.lists(st.floats(-1000, 1000, allow_nan=False), min_size=20, max_size=20))
@SETTINGS
def test_scores_are_read_back_in_the_order_of_the_expected_keys(
    scores: list[float],
) -> None:
    shuffled = dict(zip(reversed(EXPECTED), reversed(scores), strict=True))
    data = scores_to_bytes("metricx24", SUITE_SHA256, OUTPUTS_SHA256, shuffled)

    found = read_scores(
        data,
        metric="metricx24",
        suite_sha256=SUITE_SHA256,
        hypotheses_sha256=OUTPUTS_SHA256,
        expected=EXPECTED,
    )

    assert list(found) == EXPECTED
    assert found == shuffled
    assert OutputKey in (type(key) for key in found)


def test_the_golden_documents_are_the_fixtures_of_the_properties() -> None:
    assert result_from_bytes(GOLDEN_RESULT).system == "alpha"
    assert golden_result_document()["n_segments"] == 10
    assert run_from_bytes(GOLDEN_RUN).purpose == "report"
