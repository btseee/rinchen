"""What a configured skip of a doctor check may silence (D14 fix wave 1, finding M2).

A configuration may skip a doctor check, but only one that asks a device question (the one id in
`SKIPPABLE_DOCTOR_CHECKS`: this command never asks a device question), and a skip never covers a
failure: a finding whose status is `fail` stays a failure of the doctor step whatever the
configuration says, so that a skip can never turn a broken stack into a green night. The probe of
the review (a skip of `D-WHEELS` over a failing `D-WHEELS`, which gave a step without a failure)
is the last test here, on a configuration built around the validation, because the loader now
refuses that skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.config.maintain import (
    SKIPPABLE_DOCTOR_CHECKS,
    MaintainConfig,
    load_maintain_config,
)
from rinchen.errors import ConfigError
from rinchen.maintain.check_steps import doctor_step
from rinchen.maintain.report import report_document
from rinchen.maintain.states import (
    FAIL_DOCTOR,
    SKIP_DOCTOR_SELF,
    SKIP_ENVIRONMENT,
    SKIP_GPU_CHECK,
    SKIP_NO_DRIFT,
)
from tests.support.maintain_check import (
    CPU_CHECKS,
    GPU_CHECK,
    SHIPPED_CONFIG,
    Doctor,
    execute,
    finding,
    golden_runner,
    make_seams,
    shipped_config,
    step_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def loaded_with_skips(skips: dict[str, Any]) -> MaintainConfig:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["skipped_doctor_checks"] = skips
    return load_maintain_config(yaml.safe_dump(document).encode())


def config_skipping(skips: dict[str, str]) -> MaintainConfig:
    """The shipped configuration with these skips, set without validating them."""
    return shipped_config().model_copy(update={"skipped_doctor_checks": skips})


def checks_of(config: MaintainConfig, findings: list[Any]) -> list[tuple[Any, ...]]:
    step, _ = doctor_step(config, Doctor(findings))
    return [(row["id"], row["state"], row["reason"]) for row in step.detail["checks"]]


# ---- the closed set of ids the loader accepts ------------------------------------------------------


def test_the_one_check_a_configuration_may_skip_is_the_device_check() -> None:
    assert SKIPPABLE_DOCTOR_CHECKS == frozenset({GPU_CHECK})


def test_the_shipped_configuration_skips_only_the_device_check() -> None:
    assert set(shipped_config().skipped_doctor_checks) <= SKIPPABLE_DOCTOR_CHECKS


def test_the_device_check_may_be_skipped_with_a_closed_reason() -> None:
    config = loaded_with_skips({GPU_CHECK: SKIP_NO_DRIFT})

    assert config.skipped_doctor_checks == {GPU_CHECK: SKIP_NO_DRIFT}


@pytest.mark.parametrize("check_id", [*CPU_CHECKS, "D-A", "D-TORCH", "D-TORCH-CUDA2"])
def test_any_other_check_cannot_be_skipped(check_id: str) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["skipped_doctor_checks"] = {check_id: SKIP_GPU_CHECK}

    with pytest.raises(ConfigError) as caught:
        load_maintain_config(yaml.safe_dump(document).encode())

    message = str(caught.value)
    assert message.startswith("invalid maintain config")
    assert "skipped_doctor_checks" in message
    assert GPU_CHECK in message


def test_a_check_that_cannot_be_skipped_is_refused_among_others() -> None:
    with pytest.raises(ConfigError, match="skipped_doctor_checks"):
        loaded_with_skips({GPU_CHECK: SKIP_GPU_CHECK, "D-PY": SKIP_GPU_CHECK})


# ---- a skip never covers a failure ---------------------------------------------------------------


@pytest.mark.parametrize("status", ["pass", "skip"])
def test_a_configured_skip_stands_in_for_a_finding_that_did_not_fail(
    status: str,
) -> None:
    config = config_skipping({GPU_CHECK: SKIP_NO_DRIFT})

    checks = checks_of(config, [finding("D-PY"), finding(GPU_CHECK, status)])  # type: ignore[arg-type]

    assert checks == [("D-PY", "ok", None), (GPU_CHECK, "skipped", SKIP_NO_DRIFT)]


def test_a_failure_of_a_configured_skip_stays_a_failure() -> None:
    config = config_skipping({GPU_CHECK: SKIP_GPU_CHECK})
    doctor = Doctor([finding("D-PY"), finding(GPU_CHECK, "fail")])

    step, messages = doctor_step(config, doctor)

    assert step.state == "fail"
    assert [(r["id"], r["state"], r["reason"]) for r in step.detail["checks"]] == [
        ("D-PY", "ok", None),
        (GPU_CHECK, "fail", FAIL_DOCTOR),
    ]
    assert [check_id for check_id, _ in messages] == [GPU_CHECK]


def test_a_failure_is_listed_once_and_not_also_as_a_skip() -> None:
    config = config_skipping({GPU_CHECK: SKIP_GPU_CHECK})

    checks = checks_of(config, [finding(GPU_CHECK, "fail")])

    assert checks == [(GPU_CHECK, "fail", FAIL_DOCTOR)]


def test_a_skip_that_came_first_does_not_hide_a_failure_that_followed() -> None:
    config = config_skipping({GPU_CHECK: SKIP_GPU_CHECK})

    checks = checks_of(
        config,
        [finding(GPU_CHECK, "skip"), finding("D-PY"), finding(GPU_CHECK, "fail")],
    )

    assert checks == [
        (GPU_CHECK, "skipped", SKIP_GPU_CHECK),
        ("D-PY", "ok", None),
        (GPU_CHECK, "fail", FAIL_DOCTOR),
    ]


def test_a_failed_device_check_fails_the_night_and_skips_the_reruns(
    tmp_path: Path,
) -> None:
    runner = golden_runner()
    doctor = Doctor([*(finding(i) for i in CPU_CHECKS), finding(GPU_CHECK, "fail")])

    run = execute(tmp_path, runner, seams=make_seams(runner, doctor=doctor))

    document = report_document(run.result.report)
    assert document["state"] == "fail"
    assert step_of(document, "doctor")["state"] == "fail"
    items = step_of(document, "reruns")["detail"]["items"]  # type: ignore[index]
    by_id = {item["id"]: (item["state"], item["reason"]) for item in items}
    assert by_id["numerics"] == ("skipped", SKIP_ENVIRONMENT)
    assert by_id["parity_tiny"] == ("skipped", SKIP_ENVIRONMENT)
    assert {state for state, _ in by_id.values()} == {"skipped"}
    assert [argv for argv in runner.argvs if argv[0] != "uv"] == []


def test_the_probe_of_a_skip_over_a_failing_wheels_check() -> None:
    # The review's probe: D-WHEELS configured as skipped, D-WHEELS failing. The loader refuses
    # that configuration now; built around it, the step must still not call the doctor green.
    config = config_skipping({"D-WHEELS": SKIP_DOCTOR_SELF})
    findings = [
        finding("D-PY"),
        finding("D-WHEELS", "fail"),
        finding(GPU_CHECK, "skip"),
    ]

    step, messages = doctor_step(config, Doctor(findings))

    assert step.state == "fail"
    assert [(r["id"], r["state"]) for r in step.detail["checks"]] == [
        ("D-PY", "ok"),
        ("D-WHEELS", "fail"),
        (GPU_CHECK, "skipped"),
    ]
    assert [check_id for check_id, _ in messages] == ["D-WHEELS"]
