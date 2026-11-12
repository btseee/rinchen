"""The drift step's decision and table (D14, spec M6 and M7): what `drift_outcome` makes of the exit
status, the timeout and the text of one `uv lock --upgrade --dry-run`, the lock file and the pin
rules.

The order of the decision is pinned case by case: a timeout, a uv that did not run to an exit, a
non-zero exit, then the text (an unrecognised line, a no-change line next to change lines, no
result at all), then the cross-check of every change against the lock file, and only then the
table. The text is the HAND-WRITTEN sample of `tests/support/uv_outputs.py` (the real lines of a
change are UNMEASURED), so the refusals are the point: a mis-parse must end in `fail`, never in a
table that looks right. A reason is one of the strings of `rinchen.maintain.states` and holds
nothing of the output.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import PinRule, load_maintain_config
from rinchen.maintain import states
from rinchen.maintain.drift import DriftEntry, DriftOutcome, EntryClass, drift_outcome
from rinchen.maintain.lock import LockFile
from rinchen.maintain.uv_output import ChangeKind
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    OFFLINE_FAILURE_MEASURED,
    lock_for_samples,
    lock_of,
)

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
CANARY = "SECRETCANARY"
POSITIVE = "No lockfile changes detected\n"
BOILERPLATE = (
    "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3\n"
    "Resolved 189 packages in 2.31s\n"
)


def shipped_rules() -> list[PinRule]:
    return load_maintain_config(SHIPPED.read_bytes()).drift.rules


def outcome(
    text: str,
    *,
    exit_code: int | None = 0,
    timed_out: bool = False,
    lock: LockFile | None = None,
    rules: Sequence[PinRule] | None = None,
    timeout_seconds: int = 300,
) -> DriftOutcome:
    return drift_outcome(
        exit_code=exit_code,
        timed_out=timed_out,
        text=text,
        lock=lock if lock is not None else lock_for_samples(),
        rules=shipped_rules() if rules is None else rules,
        timeout_seconds=timeout_seconds,
    )


def failed(result: DriftOutcome, reason: str) -> None:
    assert result.state == "fail"
    assert result.reason == reason
    assert result.entries == ()


# --- the exit status and the timeout decide before any line is read ----------------------------


@pytest.mark.parametrize("exit_code", [0, 1, 2, None])
@pytest.mark.parametrize("text", [CHANGES_HAND_WRITTEN, NO_CHANGE_MEASURED, ""])
def test_a_timeout_is_a_failure_whatever_the_exit_code_and_the_text(
    exit_code: int | None, text: str
) -> None:
    result = outcome(text, exit_code=exit_code, timed_out=True)
    failed(result, states.fail_timeout(300))
    assert result.warnings == 0


def test_the_timeout_in_the_reason_is_the_bound_that_was_given() -> None:
    failed(outcome("", timed_out=True, timeout_seconds=42), states.fail_timeout(42))


@pytest.mark.parametrize("text", [CHANGES_HAND_WRITTEN, NO_CHANGE_MEASURED, ""])
def test_a_uv_that_did_not_run_to_an_exit_is_the_missing_uv(text: str) -> None:
    failed(outcome(text, exit_code=None), states.FAIL_UV_MISSING)


@pytest.mark.parametrize("code", [1, 2, 3, 127, 255, -9, -15])
@pytest.mark.parametrize(
    "text", [OFFLINE_FAILURE_MEASURED, CHANGES_HAND_WRITTEN, NO_CHANGE_MEASURED, ""]
)
def test_a_non_zero_exit_is_a_failure_whatever_the_text_parsed_to(
    code: int, text: str
) -> None:
    result = outcome(text, exit_code=code)
    failed(result, states.fail_uv_exit(code))
    assert result.warnings == 0


def test_the_measured_offline_failure_with_exit_1_names_the_exit_code_only() -> None:
    failed(outcome(OFFLINE_FAILURE_MEASURED, exit_code=1), states.fail_uv_exit(1))


# --- the text -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "\n\n", BOILERPLATE, "warning: a\n", BOILERPLATE + "warning: a\nwarning: b\n"],
)
def test_an_output_without_a_result_is_a_failure(text: str) -> None:
    failed(outcome(text), states.FAIL_NO_RESULT)


def test_an_output_with_no_result_keeps_the_count_of_its_warnings() -> None:
    assert outcome("warning: a\nwarning: b\n").warnings == 2


def test_one_unknown_line_next_to_the_sample_is_a_failure_that_counts_it() -> None:
    result = outcome(CHANGES_HAND_WRITTEN + "Frobnicated numpy\n")
    failed(result, states.fail_unrecognised(1))


def test_each_unknown_line_is_counted() -> None:
    text = CHANGES_HAND_WRITTEN + "Frobnicated numpy\nUpdate numpy\nx\n"
    failed(outcome(text), states.fail_unrecognised(3))


def test_the_offline_failure_text_read_with_exit_0_is_four_unrecognised_lines() -> None:
    failed(outcome(OFFLINE_FAILURE_MEASURED), states.fail_unrecognised(4))


def test_an_unknown_line_next_to_the_positive_line_is_a_failure_and_not_ok() -> None:
    failed(
        outcome(NO_CHANGE_MEASURED + "Lockfile changes detected\n"),
        states.fail_unrecognised(1),
    )


@pytest.mark.parametrize("before", [True, False])
def test_the_positive_line_next_to_change_lines_is_a_contradiction(
    before: bool,
) -> None:
    text = (
        POSITIVE + CHANGES_HAND_WRITTEN if before else CHANGES_HAND_WRITTEN + POSITIVE
    )
    failed(outcome(text), states.FAIL_CONTRADICTION)


def test_an_unknown_line_is_judged_before_a_contradiction() -> None:
    text = POSITIVE + CHANGES_HAND_WRITTEN + "Frobnicated numpy\n"
    failed(outcome(text), states.fail_unrecognised(1))


def test_an_unknown_line_is_judged_before_the_cross_check() -> None:
    text = "Update numpy v2.5.3 -> v2.5.5\nFrobnicated numpy\n"
    failed(outcome(text), states.fail_unrecognised(1))


def test_a_contradiction_is_judged_before_the_cross_check() -> None:
    text = POSITIVE + "Update numpy v2.5.3 -> v2.5.5\n"
    failed(outcome(text), states.FAIL_CONTRADICTION)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (CHANGES_HAND_WRITTEN + "Frobnicated\n", states.fail_unrecognised(1)),
        (POSITIVE + CHANGES_HAND_WRITTEN, states.FAIL_CONTRADICTION),
        ("Update numpy v2.5.3 -> v2.5.5\n", states.fail_cross_check(1)),
        (BOILERPLATE, states.FAIL_NO_RESULT),
    ],
    ids=["unrecognised", "contradiction", "cross-check", "no-result"],
)
def test_a_failure_after_the_parse_keeps_the_count_of_warnings(
    body: str, reason: str
) -> None:
    result = outcome("warning: a\n" + body + "warning: b\n")
    failed(result, reason)
    assert result.warnings == 2


# --- the cross-check against the lock file ---------------------------------------------------------

BAD_CHANGES = [
    pytest.param(
        "Update numpy v2.5.3 -> v2.5.5\n", id="update-of-a-version-the-lock-lacks"
    ),
    pytest.param("Update numpy v2.5.4.0 -> v2.5.5\n", id="update-compared-as-text"),
    pytest.param(
        "Update numpy v02.5.4 -> v2.5.5\n", id="update-with-a-zero-padded-version"
    ),
    pytest.param(
        "Update nosuchpackage v1.0 -> v1.1\n", id="update-of-a-name-the-lock-lacks"
    ),
    pytest.param("Add numpy v2.6.0\n", id="add-of-a-name-the-lock-has"),
    pytest.param(
        "Add Numpy v2.6.0\n", id="add-of-a-name-the-lock-has-in-another-spelling"
    ),
    pytest.param(
        "Remove sortedcontainers v2.3.0\n", id="remove-of-a-version-the-lock-lacks"
    ),
    pytest.param("Remove nosuchpackage v1.0\n", id="remove-of-a-name-the-lock-lacks"),
]


@pytest.mark.parametrize("line", BAD_CHANGES)
def test_a_change_that_disagrees_with_the_lock_is_a_cross_check_failure(
    line: str,
) -> None:
    failed(outcome(BOILERPLATE + line), states.fail_cross_check(1))


@pytest.mark.parametrize("line", BAD_CHANGES)
def test_one_bad_change_among_good_ones_fails_the_whole_table(line: str) -> None:
    failed(outcome(CHANGES_HAND_WRITTEN + line), states.fail_cross_check(1))


def test_the_same_update_twice_is_one_failure() -> None:
    line = "Update numpy v2.5.4 -> v2.5.5\n"
    failed(outcome(line + line), states.fail_cross_check(1))


def test_two_updates_of_one_package_to_different_versions_are_one_failure() -> None:
    text = "Update numpy v2.5.4 -> v2.5.5\nUpdate numpy v2.5.4 -> v2.5.6\n"
    failed(outcome(text), states.fail_cross_check(1))


def test_the_same_remove_twice_and_the_same_add_twice_are_one_failure_each() -> None:
    failed(outcome("Remove sortedcontainers v2.4.0\n" * 2), states.fail_cross_check(1))
    failed(outcome("Add tomli-w v1.2.0\n" * 2), states.fail_cross_check(1))


def test_three_copies_of_one_change_are_two_failures() -> None:
    failed(outcome("Add tomli-w v1.2.0\n" * 3), states.fail_cross_check(2))


def test_two_bad_changes_are_two_failures() -> None:
    text = "Update numpy v2.5.3 -> v2.5.5\nRemove nosuchpackage v1.0\n"
    failed(outcome(text), states.fail_cross_check(2))


def test_a_bad_change_written_twice_counts_each_line() -> None:
    line = "Update numpy v2.5.3 -> v2.5.5\n"
    failed(outcome(line + line), states.fail_cross_check(2))


def test_a_change_that_is_bad_and_a_duplicate_is_counted_once_per_line() -> None:
    text = (
        "Update numpy v2.5.4 -> v2.5.5\nUpdate numpy v2.5.3 -> v2.5.5\nAdd numpy v1\n"
    )
    failed(outcome(text), states.fail_cross_check(2))


def test_an_update_and_a_remove_of_one_package_are_not_duplicates() -> None:
    text = "Update numpy v2.5.4 -> v2.5.5\nRemove numpy v2.5.4\n"
    result = outcome(text)
    assert result.state == "drift"
    assert [(entry.package, entry.change) for entry in result.entries] == [
        ("numpy", "remove"),
        ("numpy", "update"),
    ]


def test_an_add_of_a_name_the_lock_has_is_refused_in_the_updated_spelling_too() -> None:
    failed(outcome("Added rich 15.0.1\n"), states.fail_cross_check(1))


def test_the_names_of_the_output_and_of_the_lock_are_compared_in_pep_503_form() -> None:
    result = outcome("Update Huggingface_Hub v2.4.0 -> v2.4.1\n", rules=[])
    assert result.state == "drift"
    assert result.entries[0].package == "huggingface-hub"


def test_an_old_version_that_is_any_of_several_locked_versions_passes() -> None:
    lock = lock_of({"pyonmttok": ["1.38.1", "1.38.0"]})
    for old in ("1.38.1", "1.38.0"):
        assert (
            outcome(f"Update pyonmttok v{old} -> v1.39\n", lock=lock).state == "drift"
        )
    failed(
        outcome("Update pyonmttok v1.37 -> v1.39\n", lock=lock),
        states.fail_cross_check(1),
    )


def test_a_failure_in_the_cross_check_reports_no_name_and_no_line_of_the_output() -> (
    None
):
    result = outcome(
        f"Update {CANARY.lower()} v1.0 -> v1.1\nAdd numpy v1.0+{CANARY.lower()}\n"
    )
    assert result.reason == states.fail_cross_check(2)
    assert CANARY.lower() not in str(result).lower()


def test_an_unrecognised_failure_reports_no_line_of_the_output() -> None:
    result = outcome(CHANGES_HAND_WRITTEN + f"Frobnicated {CANARY}\n")
    assert result.reason == states.fail_unrecognised(1)
    assert CANARY not in str(result)


# --- the table ------------------------------------------------------------------------------------


def test_nothing_to_change_is_ok_with_no_entries() -> None:
    result = outcome(NO_CHANGE_MEASURED)
    assert result == DriftOutcome(state="ok", reason=None, entries=(), warnings=0)


def test_the_positive_line_alone_is_ok() -> None:
    assert outcome(POSITIVE).state == "ok"


def test_the_ok_state_ignores_the_warnings_and_counts_them() -> None:
    result = outcome("warning: a\n" + NO_CHANGE_MEASURED + "warning: b\n")
    assert (result.state, result.warnings, result.entries) == ("ok", 2, ())


def entry(
    package: str,
    change: ChangeKind,
    locked: str | None,
    available: str | None,
    klass: EntryClass,
    rule: str | None = None,
    requires: tuple[str, ...] = (),
) -> DriftEntry:
    return DriftEntry(package, change, locked, available, klass, rule, requires)


def test_the_sample_with_the_shipped_rules_is_seven_entries_sorted_by_package() -> None:
    torch_requires = tuple(
        next(rule for rule in shipped_rules() if rule.id == "torch-family").requires
    )
    result = outcome(CHANGES_HAND_WRITTEN)
    assert result.state == "drift"
    assert result.reason is None
    assert result.warnings == 0
    assert result.entries == (
        entry(
            "huggingface-hub",
            "update",
            "2.4.0",
            "2.4.1",
            "blocked",
            "hub-client",
            ("evaluation_tiny", "gate"),
        ),
        entry(
            "numpy",
            "update",
            "2.5.4",
            "2.5.5",
            "blocked",
            "sacrebleu-numpy",
            ("numerics", "evaluation_tiny"),
        ),
        entry(
            "nvidia-cudnn-cu13",
            "update",
            "9.24.0.43",
            "9.25.0.10",
            "blocked",
            "torch-family",
            torch_requires,
        ),
        entry("rich", "update", "15.0.0", "15.0.1", "drift"),
        entry(
            "safetensors",
            "update",
            "0.8.0",
            "0.8.1",
            "blocked",
            "transformers-stack",
            ("gate",),
        ),
        entry("sortedcontainers", "remove", "2.4.0", None, "drift"),
        entry("tomli-w", "add", None, "1.2.0", "drift"),
    )
    classes = [item.klass for item in result.entries]
    assert (classes.count("blocked"), classes.count("drift")) == (4, 3)


def test_a_blocked_table_is_still_the_drift_state_and_never_a_failure() -> None:
    result = outcome("Update numpy v2.5.4 -> v2.5.5\n")
    assert [item.klass for item in result.entries] == ["blocked"]
    assert (result.state, result.reason) == ("drift", None)


def test_without_rules_every_entry_is_drift_with_no_rule_and_no_requirements() -> None:
    result = outcome(CHANGES_HAND_WRITTEN, rules=[])
    assert result.state == "drift"
    assert len(result.entries) == 7
    for item in result.entries:
        assert (item.klass, item.rule, item.requires) == ("drift", None, ())


def test_the_first_rule_that_names_a_package_wins() -> None:
    rules = shipped_rules()
    first = PinRule(
        id="first", packages=["num*"], decision="a decision", requires=["numerics"]
    )
    result = outcome("Update numpy v2.5.4 -> v2.5.5\n", rules=[first, *rules])
    assert (result.entries[0].rule, result.entries[0].requires) == (
        "first",
        ("numerics",),
    )


def test_a_package_the_output_spells_in_another_form_is_still_blocked() -> None:
    result = outcome("Update Huggingface_Hub v2.4.0 -> v2.4.1\n")
    assert (result.entries[0].package, result.entries[0].klass) == (
        "huggingface-hub",
        "blocked",
    )
    assert result.entries[0].rule == "hub-client"


def test_the_locked_column_is_the_lock_s_string_and_never_the_outputs() -> None:
    lock = lock_of(
        {"numpy": ["2.5.3", "2.5.4"], "sortedcontainers": ["2.4.0", "2.4.1"]}
    )
    text = "Update numpy v2.5.4 -> v2.5.5\nRemove sortedcontainers v2.4.0\n"
    result = outcome(text, lock=lock)
    assert [(item.package, item.locked) for item in result.entries] == [
        ("numpy", "2.5.3, 2.5.4"),
        ("sortedcontainers", "2.4.0, 2.4.1"),
    ]


def test_the_locked_column_of_a_single_version_is_that_version_alone() -> None:
    result = outcome(
        "Update numpy v2.5.4 -> v2.5.5\n", lock=lock_of({"numpy": ["2.5.4"]})
    )
    assert result.entries[0].locked == "2.5.4"


def test_an_add_has_no_locked_version_and_a_remove_has_no_available_one() -> None:
    result = outcome("Add tomli-w v1.2.0\nRemove sortedcontainers v2.4.0\n", rules=[])
    by_package = {item.package: item for item in result.entries}
    assert (by_package["tomli-w"].locked, by_package["tomli-w"].available) == (
        None,
        "1.2.0",
    )
    assert (
        by_package["sortedcontainers"].locked,
        by_package["sortedcontainers"].available,
    ) == (
        "2.4.0",
        None,
    )


def test_the_available_version_is_the_new_one_of_the_output() -> None:
    result = outcome("Update numpy v2.5.4 -> v9.9.9+local\n", rules=[])
    assert result.entries[0].available == "9.9.9+local"


def test_entries_are_sorted_by_package_and_not_by_change() -> None:
    lock = lock_of({"aaa": ["1"]})
    for text in (
        "Update aaa v1 -> v2\nAdd zzz v1\n",
        "Add zzz v1\nUpdate aaa v1 -> v2\n",
    ):
        entries = outcome(text, lock=lock, rules=[]).entries
        assert [(item.package, item.change) for item in entries] == [
            ("aaa", "update"),
            ("zzz", "add"),
        ]


def test_entries_of_one_package_are_sorted_by_change() -> None:
    text = "Update numpy v2.5.4 -> v2.5.5\nRemove numpy v2.5.4\n"
    result = outcome(text, rules=[])
    assert [(item.package, item.change) for item in result.entries] == [
        ("numpy", "remove"),
        ("numpy", "update"),
    ]


def test_the_order_of_the_text_does_not_change_the_entries() -> None:
    lines = CHANGES_HAND_WRITTEN.splitlines()[2:]
    forward = outcome("\n".join(lines) + "\n")
    backward = outcome("\n".join(reversed(lines)) + "\n")
    assert forward.entries == backward.entries


def test_a_rerun_gives_equal_outcomes() -> None:
    assert outcome(CHANGES_HAND_WRITTEN) == outcome(CHANGES_HAND_WRITTEN)
    assert outcome(NO_CHANGE_MEASURED) == outcome(NO_CHANGE_MEASURED)


def test_warnings_are_counted_in_a_drift_outcome_and_are_never_entries() -> None:
    result = outcome("warning: a\n" + CHANGES_HAND_WRITTEN + "warning: b\nwarning: c\n")
    assert result.state == "drift"
    assert result.warnings == 3
    assert len(result.entries) == 7


def test_the_outcome_and_its_entries_are_frozen_slotted_values() -> None:
    result = outcome(CHANGES_HAND_WRITTEN)
    for value, field in ((result, "state"), (result.entries[0], "klass")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "x")
        assert not hasattr(value, "__dict__")
    assert isinstance(result.entries, tuple)
    assert isinstance(result.entries[0].requires, tuple)


def test_the_lock_and_the_rules_are_not_changed_by_a_decision() -> None:
    lock = lock_for_samples()
    rules = shipped_rules()
    before = (dict(lock.versions), [rule.model_dump() for rule in rules])
    outcome(CHANGES_HAND_WRITTEN, lock=lock, rules=rules)
    assert (dict(lock.versions), [rule.model_dump() for rule in rules]) == before
