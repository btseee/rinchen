"""What a `Step` of a maintenance report refuses (D14 fix wave 1, finding M9).

A step is built from observations, and the closed lists of `rinchen.maintain.states` are what
keeps free text out of a report. The step enforces what the report's readers rely on: an `ok` or
`drift` step has no reason; a `skipped` step has a reason; a `fail` step has one unless it is a
group step (the doctor and the reruns list the reason of each item in their detail and have none
of their own, unless they were skipped as a whole); and a reason is always one of the closed
strings, or an instance of one of the closed templates.
"""

from __future__ import annotations

import re

import pytest

from rinchen.maintain import states
from rinchen.maintain.report import GROUP_STEP_IDS, STEP_IDS, Step
from rinchen.maintain.states import is_closed_reason
from tests.support.maintain_scans import closed_reason
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SINGLE_STEP_IDS = [step_id for step_id in STEP_IDS if step_id not in GROUP_STEP_IDS]
FREE_TEXT = [
    "",
    "because",
    "the check failed.",
    "uv exited with code x",
    "uv exited with code ",
    "uv exited with code 1 ",
    "not due at the monthly cadence",
    "disabled by --no-drift\n",
    "Disabled by --no-drift",
    "error: boom",
    "/home/user/uv exited with code 1",
]


def filled_reasons() -> list[str]:
    """Every closed reason, each template with a number (a negative one too) or a cadence."""
    found = []
    for reason in states.ALL_REASONS:
        if "<cadence>" in reason:
            found += [
                reason.replace("<cadence>", word) for word in ("nightly", "weekly")
            ]
        else:
            found += [
                re.sub(r"<[a-z]+>", number, reason) for number in ("0", "7", "-9")
            ]
    return found


# ---- the group steps ---------------------------------------------------------------------------------


def test_the_group_steps_are_the_doctor_and_the_reruns() -> None:
    assert GROUP_STEP_IDS == ("doctor", "reruns")
    assert set(GROUP_STEP_IDS) <= set(STEP_IDS)


# ---- ok and drift: no reason ---------------------------------------------------------------------


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_an_ok_step_has_no_reason(step_id: str) -> None:
    Step(step_id, "ok", None, {})
    with pytest.raises(ValueError, match="reason"):
        Step(step_id, "ok", states.SKIP_NO_DRIFT, {})


def test_a_drift_step_has_no_reason() -> None:
    Step("drift", "drift", None, {})
    with pytest.raises(ValueError, match="reason"):
        Step("drift", "drift", states.fail_uv_exit(1), {})


# ---- skipped: a closed reason ----------------------------------------------------------------------


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_a_skipped_step_needs_a_reason(step_id: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        Step(step_id, "skipped", None, {})


@pytest.mark.parametrize("step_id", STEP_IDS)
@pytest.mark.parametrize(
    "reason", [*states.SKIP_REASONS, states.skip_not_due("weekly")]
)
def test_a_skipped_step_takes_a_closed_reason(step_id: str, reason: str) -> None:
    assert Step(step_id, "skipped", reason, {}).reason == reason


@pytest.mark.parametrize("step_id", STEP_IDS)
@pytest.mark.parametrize("reason", FREE_TEXT)
def test_a_skipped_step_refuses_free_text(step_id: str, reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        Step(step_id, "skipped", reason, {})


# ---- fail: a closed reason unless the step is a group ------------------------------------------------


@pytest.mark.parametrize("step_id", SINGLE_STEP_IDS)
def test_a_failed_single_step_needs_a_reason(step_id: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        Step(step_id, "fail", None, {})


@pytest.mark.parametrize("step_id", GROUP_STEP_IDS)
def test_a_failed_group_step_has_no_reason_of_its_own(step_id: str) -> None:
    assert Step(step_id, "fail", None, {}).reason is None


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_a_failed_step_takes_a_closed_reason(step_id: str) -> None:
    assert Step(step_id, "fail", states.fail_uv_exit(2), {}).reason == (
        "uv exited with code 2"
    )


@pytest.mark.parametrize("step_id", STEP_IDS)
@pytest.mark.parametrize("reason", FREE_TEXT)
def test_a_failed_step_refuses_free_text(step_id: str, reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        Step(step_id, "fail", reason, {})


# ---- the closed reasons --------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", filled_reasons())
def test_every_closed_reason_filled_in_is_closed(reason: str) -> None:
    assert is_closed_reason(reason)
    assert closed_reason(reason)


@pytest.mark.parametrize("reason", FREE_TEXT)
def test_free_text_is_not_a_closed_reason(reason: str) -> None:
    assert not is_closed_reason(reason)
    assert not closed_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "uv exited with code " + chr(0x0663),
        "uv exited with code 1\n",
        "uv exited with code 1.5",
        "uv exited with code +1",
        "timed out after 5 s ",
        "5 failed and 3",
    ],
)
def test_a_template_filled_with_anything_but_a_whole_number_is_not_closed(
    reason: str,
) -> None:
    assert not is_closed_reason(reason)
    assert not closed_reason(reason)


def test_the_message_of_a_refused_reason_quotes_none_of_it() -> None:
    with pytest.raises(ValueError) as caught:
        Step("drift", "fail", "swordfish-0451", {})

    assert "swordfish" not in str(caught.value)
