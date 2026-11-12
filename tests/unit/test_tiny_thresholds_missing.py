"""The missing and the impossible values of the thresholds of `rinchen.tiny.thresholds` (D15, spec P8),
and the `stages` threshold.

A value that is missing (a `None`, a key that is not there, a stage that never ran) or that cannot be
a measurement (not a number, a boolean, not finite, outside its range) is a `fail` whose detail names
it, and its value is `None`: it is never read as zero, which would pass every upper bound. A detail
names only closed identifiers and never a key or a value of the inputs. The `stages` threshold counts
the stages that failed, were skipped without a reason of the closed list, have no record or have two,
and it does not trust a record that was damaged after it was built.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.states import (
    SKIP_EARLIER_FAILED,
    SKIP_NO_LID,
    SKIP_REASONS,
    PartRecord,
    StageRecord,
    fail_exit,
)
from rinchen.tiny.thresholds import (
    GATED_SYSTEM,
    MEMORISED_DIRECTIONS,
    ThresholdResult,
    evaluate_thresholds,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_threshold_inputs import (
    BUDGET,
    FIRST,
    LAST,
    LIMITS,
    SENTINEL,
    chrf_table,
    good_stages,
    memorised_with,
    memorised_without,
    one,
    only_failing,
    parity_with,
    results_of,
    robustness_with,
    with_changes,
)

pytestmark = pytest.mark.usefixtures("no_network")


MISSING_CASES: list[tuple[str, dict[str, Any], set[str], str]] = [
    ("no table", {"memorised_chrf": {}}, {"train_chrf", "int8_chrf_drop"}, FIRST),
    (
        "no int8 figure",
        {"memorised_chrf": memorised_without("rus_Cyrl-khk_Cyrl", "int8")},
        {"train_chrf", "int8_chrf_drop"},
        "rus_Cyrl-khk_Cyrl",
    ),
    (
        "no float32 figure",
        {"memorised_chrf": memorised_without(LAST, "float32")},
        {"int8_chrf_drop"},
        LAST,
    ),
    (
        "a direction without systems",
        {"memorised_chrf": {**chrf_table(), FIRST: {}}},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure that is None",
        {"memorised_chrf": memorised_with(FIRST, "int8", None)},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure that is not a number",
        {"memorised_chrf": memorised_with(FIRST, "int8", "99.5")},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure that is a bool",
        {"memorised_chrf": memorised_with(FIRST, "int8", True)},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure that is nan",
        {"memorised_chrf": memorised_with(FIRST, "int8", math.nan)},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure above 100",
        {"memorised_chrf": memorised_with(FIRST, "int8", 100.01)},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "an int8 figure below 0",
        {"memorised_chrf": memorised_with(FIRST, "int8", -0.01)},
        {"train_chrf", "int8_chrf_drop"},
        FIRST,
    ),
    (
        "a float32 figure that is inf",
        {"memorised_chrf": memorised_with(FIRST, "float32", math.inf)},
        {"int8_chrf_drop"},
        FIRST,
    ),
    (
        "a float32 figure above 100",
        {"memorised_chrf": memorised_with(LAST, "float32", 150.0)},
        {"int8_chrf_drop"},
        LAST,
    ),
    ("no parity", {"parity": None}, {"parity_exact", "parity_delta"}, "parity"),
    ("an empty parity", {"parity": {}}, {"parity_exact", "parity_delta"}, "parity"),
    (
        "no exact share",
        {"parity": {"mean_abs_delta": 1e-6}},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "no mean delta",
        {"parity": {"exact_share": 1.0}},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "a share above 1",
        {"parity": parity_with(exact_share=1.01)},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a share below 0",
        {"parity": parity_with(exact_share=-0.01)},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a share that is nan",
        {"parity": parity_with(exact_share=math.nan)},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a share that is a bool",
        {"parity": parity_with(exact_share=True)},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a share that is None",
        {"parity": parity_with(exact_share=None)},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a share that is text",
        {"parity": parity_with(exact_share="1.0")},
        {"parity_exact"},
        "exact_share",
    ),
    (
        "a delta below 0",
        {"parity": parity_with(mean_abs_delta=-1e-9)},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "a delta that is nan",
        {"parity": parity_with(mean_abs_delta=math.nan)},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "a delta that is inf",
        {"parity": parity_with(mean_abs_delta=math.inf)},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "a delta that is a bool",
        {"parity": parity_with(mean_abs_delta=False)},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "a delta that is None",
        {"parity": parity_with(mean_abs_delta=None)},
        {"parity_delta"},
        "mean_abs_delta",
    ),
    (
        "no robustness",
        {"robustness": None},
        {"script_purity", "placeholder_acc"},
        "robustness",
    ),
    (
        "an empty robustness",
        {"robustness": {}},
        {"script_purity", "placeholder_acc"},
        "robustness",
    ),
    (
        "no purity",
        {"robustness": {"placeholder_acc": 1.0}},
        {"script_purity"},
        "script_purity",
    ),
    (
        "no placeholder figure",
        {"robustness": {"script_purity": 1.0}},
        {"placeholder_acc"},
        "placeholder_acc",
    ),
    (
        "a purity of None",
        {"robustness": robustness_with(script_purity=None)},
        {"script_purity"},
        "script_purity",
    ),
    (
        "a purity above 1",
        {"robustness": robustness_with(script_purity=1.5)},
        {"script_purity"},
        "script_purity",
    ),
    (
        "a purity below 0",
        {"robustness": robustness_with(script_purity=-0.5)},
        {"script_purity"},
        "script_purity",
    ),
    (
        "a purity that is nan",
        {"robustness": robustness_with(script_purity=math.nan)},
        {"script_purity"},
        "script_purity",
    ),
    (
        "a purity that is a bool",
        {"robustness": robustness_with(script_purity=True)},
        {"script_purity"},
        "script_purity",
    ),
    (
        "a placeholder figure of None",
        {"robustness": robustness_with(placeholder_acc=None)},
        {"placeholder_acc"},
        "placeholder_acc",
    ),
    (
        "a placeholder figure above 1",
        {"robustness": robustness_with(placeholder_acc=1.0001)},
        {"placeholder_acc"},
        "placeholder_acc",
    ),
    (
        "a placeholder figure that is a bool",
        {"robustness": robustness_with(placeholder_acc=True)},
        {"placeholder_acc"},
        "placeholder_acc",
    ),
    (
        "an elapsed time that is nan",
        {"elapsed_seconds": math.nan},
        {"wall_time"},
        "elapsed",
    ),
    (
        "an elapsed time that is inf",
        {"elapsed_seconds": math.inf},
        {"wall_time"},
        "elapsed",
    ),
    ("a negative elapsed time", {"elapsed_seconds": -1.0}, {"wall_time"}, "elapsed"),
    (
        "an elapsed time that is None",
        {"elapsed_seconds": None},
        {"wall_time"},
        "elapsed",
    ),
    (
        "an elapsed time that is a bool",
        {"elapsed_seconds": True},
        {"wall_time"},
        "elapsed",
    ),
]


@pytest.mark.parametrize(
    ("changes", "failing", "named"),
    [case[1:] for case in MISSING_CASES],
    ids=[case[0] for case in MISSING_CASES],
)
def test_a_missing_or_impossible_value_is_a_fail_that_names_it(
    changes: dict[str, Any], failing: set[str], named: str
) -> None:
    results = results_of(with_changes(**changes))
    assert only_failing(with_changes(**changes)) == failing
    for threshold_id in failing:
        result = results[threshold_id]
        assert result.value is None
        assert result.verdict == "fail"
        assert result.detail.startswith("missing or invalid: ")
        assert named in result.detail
        assert "/" not in result.detail


def test_every_missing_direction_is_named_not_only_the_first() -> None:
    result = one("train_chrf", with_changes(memorised_chrf={}))
    for direction in MEMORISED_DIRECTIONS:
        assert direction in result.detail
    result = one("int8_chrf_drop", with_changes(memorised_chrf={}))
    for direction in MEMORISED_DIRECTIONS:
        assert direction in result.detail


def test_a_missing_system_is_named_with_its_direction() -> None:
    result = one(
        "int8_chrf_drop",
        with_changes(memorised_chrf=memorised_without(FIRST, "float32")),
    )
    assert "float32" in result.detail
    assert FIRST in result.detail
    result = one(
        "train_chrf", with_changes(memorised_chrf=memorised_without(FIRST, "int8"))
    )
    assert "int8" in result.detail
    assert FIRST in result.detail


def test_a_missing_value_is_never_read_as_zero() -> None:
    # Zero would pass every upper bound: a delta, a drop and a time of 0 are the best possible.
    inputs = with_changes(
        parity={}, robustness={}, memorised_chrf={}, elapsed_seconds=math.nan
    )
    assert only_failing(inputs) == {
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
        "wall_time",
    }


def test_no_detail_quotes_a_key_or_a_value_it_was_handed() -> None:
    inputs = with_changes(
        parity={SENTINEL: 5.0, "exact_share": math.nan, "mean_abs_delta": SENTINEL},
        robustness={SENTINEL: 1.0, "script_purity": SENTINEL},
        memorised_chrf={SENTINEL: {GATED_SYSTEM: 1.0}, FIRST: {SENTINEL: 1.0}},
    )
    for result in evaluate_thresholds(inputs, LIMITS, BUDGET):
        assert SENTINEL not in result.detail
        assert SENTINEL not in result.bound


# ---- the stages threshold ----------------------------------------------------------------------


def replaced(
    records: list[StageRecord], stage: str, new: StageRecord
) -> list[StageRecord]:
    return [new if record.id == stage else record for record in records]


def corrupted(record: StageRecord, **fields: Any) -> StageRecord:
    """A record that was damaged after it was built: the constructor refuses these, so the
    threshold must not trust it."""
    for name, value in fields.items():
        object.__setattr__(record, name, value)
    return record


def stages_result(records: list[StageRecord]) -> ThresholdResult:
    return one("stages", with_changes(stages=records))


def after_a_failure(failing: str) -> list[StageRecord]:
    """The stages of a run that stopped: `failing` failed and every later one is skipped for it."""
    records = []
    for stage in STAGE_IDS:
        if stage == failing:
            records.append(StageRecord(stage, "fail", fail_exit(4)))
        elif STAGE_IDS.index(stage) > STAGE_IDS.index(failing):
            records.append(StageRecord(stage, "skipped", SKIP_EARLIER_FAILED))
        else:
            records.append(StageRecord(stage, "ok"))
    return records


def test_all_stages_ok_is_a_pass_of_value_zero() -> None:
    result = stages_result(good_stages())
    assert (result.verdict, result.value) == ("pass", 0)


@pytest.mark.parametrize("failing", STAGE_IDS)
def test_the_stages_skipped_for_an_earlier_failure_are_not_faults_of_their_own(
    failing: str,
) -> None:
    # A stopped run has one fault, the stage that failed; the stages after it are what it caused.
    result = stages_result(after_a_failure(failing))
    assert (result.verdict, result.value) == ("fail", 1)
    assert failing in result.detail
    assert "no failure before" not in result.detail


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_a_stage_skipped_for_an_earlier_failure_when_nothing_failed_is_a_fault(
    stage: str,
) -> None:
    records = replaced(
        good_stages(), stage, StageRecord(stage, "skipped", SKIP_EARLIER_FAILED)
    )
    result = stages_result(records)
    assert (result.verdict, result.value) == ("fail", 1)
    assert stage in result.detail
    assert "no failure before" in result.detail


@pytest.mark.parametrize("reason", SKIP_REASONS)
@pytest.mark.parametrize("failing", ["train", None])
def test_a_whole_stage_skip_is_explained_by_a_failure_before_it_and_by_nothing_else(
    reason: str, failing: str | None
) -> None:
    # `serve` comes after `train`: the reasons about a part of a stage (no LID model, no neural
    # scorer) say nothing about a stage, so they are faults whether something failed before or not.
    records = after_a_failure(failing) if failing else good_stages()
    records = replaced(records, "serve", StageRecord("serve", "skipped", reason))
    result = stages_result(records)
    explained = failing is not None and reason == SKIP_EARLIER_FAILED
    assert result.value == (1 if explained else 1 + (failing is not None))
    assert result.verdict == "fail"
    assert ("serve" in result.detail) is (not explained)


def test_a_skip_for_an_earlier_failure_is_not_explained_by_a_failure_after_it() -> None:
    records = replaced(
        good_stages(), "spm", StageRecord("spm", "skipped", SKIP_EARLIER_FAILED)
    )
    records = replaced(records, "train", StageRecord("train", "fail", fail_exit(4)))
    for ordered in (records, list(reversed(records))):
        result = stages_result(ordered)
        assert (result.verdict, result.value) == ("fail", 2)
        assert "spm" in result.detail
        assert "train" in result.detail


def test_a_stage_skipped_for_no_reason_explains_the_skips_after_it() -> None:
    # The first stage that went wrong is counted; the ones that say "an earlier stage failed" after
    # it are explained by it, even though it did not fail.
    records = replaced(
        good_stages(), "serve", StageRecord("serve", "skipped", SKIP_EARLIER_FAILED)
    )
    for stage in ("robustness", "evaluate"):
        records = replaced(
            records, stage, StageRecord(stage, "skipped", SKIP_EARLIER_FAILED)
        )
    for ordered in (records, list(reversed(records))):
        result = stages_result(ordered)
        assert (result.verdict, result.value) == ("fail", 1)
        assert "serve" in result.detail
        assert "robustness" not in result.detail


@pytest.mark.parametrize("reason", SKIP_REASONS)
def test_a_part_keeps_every_closed_skip_reason_on_an_ok_stage(reason: str) -> None:
    part = PartRecord("lid", "skipped", reason)
    record = StageRecord("filter", "ok", parts=(part,))
    result = stages_result(replaced(good_stages(), "filter", record))
    assert (result.verdict, result.value) == ("pass", 0)


def test_one_failed_stage_fails_the_threshold_and_is_named() -> None:
    records = replaced(
        good_stages(), "train", StageRecord("train", "fail", fail_exit(4))
    )
    result = stages_result(records)
    assert (result.verdict, result.value) == ("fail", 1)
    assert "train" in result.detail


def test_two_failed_stages_count_two() -> None:
    records = replaced(
        good_stages(), "train", StageRecord("train", "fail", fail_exit(4))
    )
    records = replaced(records, "export", StageRecord("export", "fail", fail_exit(1)))
    result = stages_result(records)
    assert (result.verdict, result.value) == ("fail", 2)
    assert "train" in result.detail
    assert "export" in result.detail


def test_a_skip_without_a_closed_reason_fails_the_threshold_and_is_named() -> None:
    skipped = corrupted(
        StageRecord("spm", "skipped", SKIP_EARLIER_FAILED), reason="free text"
    )
    result = stages_result(replaced(good_stages(), "spm", skipped))
    assert (result.verdict, result.value) == ("fail", 1)
    assert "spm" in result.detail
    assert "free text" not in result.detail


@pytest.mark.parametrize(
    "reason", [None, "", "exit code 4", "an earlier stage failed."]
)
def test_a_skip_with_no_skip_reason_fails_the_threshold(reason: str | None) -> None:
    skipped = corrupted(
        StageRecord("spm", "skipped", SKIP_EARLIER_FAILED), reason=reason
    )
    assert stages_result(replaced(good_stages(), "spm", skipped)).verdict == "fail"


def test_an_ok_stage_that_holds_a_reason_fails_the_threshold() -> None:
    odd = corrupted(StageRecord("spm", "ok"), reason=SKIP_NO_LID)
    assert stages_result(replaced(good_stages(), "spm", odd)).verdict == "fail"


def test_a_state_outside_the_three_fails_the_threshold() -> None:
    odd = corrupted(StageRecord("spm", "ok"), state="drift")
    assert stages_result(replaced(good_stages(), "spm", odd)).verdict == "fail"


def test_a_failed_part_of_an_ok_stage_fails_the_threshold() -> None:
    record = StageRecord("filter", "ok", parts=(PartRecord("numbers", "ok"),))
    corrupted(record.parts[0], state="fail", reason=fail_exit(1))
    result = stages_result(replaced(good_stages(), "filter", record))
    assert (result.verdict, result.value) == ("fail", 1)
    assert "filter" in result.detail


def test_a_skipped_part_without_a_closed_reason_fails_the_threshold() -> None:
    part = PartRecord("lid", "skipped", SKIP_NO_LID)
    record = StageRecord("filter", "ok", parts=(part,))
    corrupted(part, reason="free text")
    assert stages_result(replaced(good_stages(), "filter", record)).verdict == "fail"


def test_a_stage_that_has_no_record_fails_the_threshold_and_is_named() -> None:
    records = [record for record in good_stages() if record.id != "parity"]
    result = stages_result(records)
    assert (result.verdict, result.value) == ("fail", 1)
    assert "parity" in result.detail


def test_a_record_of_a_stage_that_does_not_exist_fails_the_threshold_without_a_quote() -> (
    None
):
    odd = corrupted(StageRecord("spm", "ok"), id="ZZ-planted-stage")
    result = stages_result([*good_stages(), odd])
    assert (result.verdict, result.value) == ("fail", 1)
    assert "unknown stage" in result.detail
    assert "ZZ-planted-stage" not in result.detail


def test_a_stage_with_two_records_fails_the_threshold_and_is_named() -> None:
    records = [*good_stages(), StageRecord("parity", "ok")]
    result = stages_result(records)
    assert (result.verdict, result.value) == ("fail", 1)
    assert "parity" in result.detail


def test_no_records_at_all_is_every_stage_missing() -> None:
    result = stages_result([])
    assert (result.verdict, result.value) == ("fail", 15)
    for stage in STAGE_IDS:
        assert stage in result.detail


def test_the_stage_threshold_does_not_depend_on_the_figures() -> None:
    inputs = with_changes(parity=None, robustness=None, memorised_chrf={})
    assert results_of(inputs)["stages"].verdict == "pass"


def test_a_stage_that_failed_fails_no_other_threshold() -> None:
    records = replaced(
        good_stages(), "train", StageRecord("train", "fail", fail_exit(4))
    )
    assert only_failing(with_changes(stages=records)) == {"stages"}
