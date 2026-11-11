"""The doctor checks of `rinchen maintain check` (D14, spec M4): the six CPU checks, called in
process in the order of the doctor, and the one GPU check that the command never calls.

The partition is pinned from both sides: the names of `doctor.ALL_CHECKS` are exactly the CPU list
plus `check_torch_cuda`, so a check added to the doctor fails this module until someone decides
whether it is a CPU or a GPU question. `run_cpu_checks` is exercised over doubles (the real checks
start Eole and CTranslate2 and take seconds), with a spy in the place of the GPU check that fails
the test if anything calls it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rinchen.cli import doctor, maintain_doctor
from rinchen.cli.doctor import DoctorResult
from rinchen.cli.maintain_doctor import CPU_CHECKS, run_cpu_checks
from rinchen.config.maintain import SKIPPABLE_DOCTOR_CHECKS
from rinchen.maintain.check_types import DoctorFinding

CPU_NAMES = (
    "check_py",
    "check_wheels",
    "check_eole_py",
    "check_eole_cli",
    "check_ct2_conv",
    "check_ct2_import",
)
CPU_IDS = (
    "D-PY",
    "D-WHEELS",
    "D-EOLE-PY",
    "D-EOLE-CLI",
    "D-CT2-CONV",
    "D-CT2-IMPORT",
)


def fakes(
    monkeypatch: pytest.MonkeyPatch,
    results: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    """Put a double for each CPU check in `CPU_CHECKS` and a spy for the GPU check in the doctor;
    returns the list that the doubles append their names to, in the order they are called."""
    calls: list[str] = []
    wanted = results or {}

    def double(name: str, check_id: str) -> Callable[[], DoctorResult]:
        status, message = wanted.get(check_id, ("pass", f"message of {check_id}"))

        def check() -> DoctorResult:
            calls.append(name)
            return DoctorResult(check_id, status, message)

        check.__name__ = name
        return check

    monkeypatch.setattr(
        maintain_doctor,
        "CPU_CHECKS",
        tuple(double(n, i) for n, i in zip(CPU_NAMES, CPU_IDS, strict=True)),
    )

    def never() -> DoctorResult:
        raise AssertionError("the GPU check was called")

    monkeypatch.setattr(doctor, "check_torch_cuda", never)
    monkeypatch.setattr(doctor, "ALL_CHECKS", [*maintain_doctor.CPU_CHECKS, never])
    return calls


# ---- the list ---------------------------------------------------------------------------------------


def test_the_cpu_list_names_the_six_checks_in_the_order_of_the_doctor() -> None:
    assert tuple(check.__name__ for check in CPU_CHECKS) == CPU_NAMES
    assert all(check is getattr(doctor, check.__name__) for check in CPU_CHECKS)


def test_the_doctor_holds_the_cpu_list_and_the_gpu_check_and_nothing_else() -> None:
    names = [check.__name__ for check in doctor.ALL_CHECKS]

    assert names == [*(check.__name__ for check in CPU_CHECKS), "check_torch_cuda"]


def test_the_gpu_check_is_not_in_the_cpu_list() -> None:
    assert doctor.check_torch_cuda not in CPU_CHECKS


def test_the_checks_a_configuration_may_skip_are_the_doctors_device_checks() -> None:
    device_checks = [check for check in doctor.ALL_CHECKS if check not in CPU_CHECKS]

    assert {maintain_doctor.check_id_of(check) for check in device_checks} == (
        SKIPPABLE_DOCTOR_CHECKS
    )


# ---- running them -----------------------------------------------------------------------------------


def test_every_cpu_check_is_called_once_in_order_and_the_gpu_check_never(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fakes(monkeypatch)

    findings = run_cpu_checks()

    assert calls == list(CPU_NAMES)
    assert [finding.check_id for finding in findings] == list(CPU_IDS)


@pytest.mark.parametrize("status", ["pass", "fail", "skip"])
def test_a_result_becomes_a_finding_unchanged(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    fakes(monkeypatch, {"D-EOLE-PY": (status, "import eole failed: boom")})

    findings = run_cpu_checks()

    assert findings[2] == DoctorFinding(
        "D-EOLE-PY",
        status,
        "import eole failed: boom",  # type: ignore[arg-type]
    )


def test_the_findings_are_findings_of_the_check_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes(monkeypatch)

    assert all(isinstance(item, DoctorFinding) for item in run_cpu_checks())


def test_a_check_that_raises_is_a_failed_check_of_its_own_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fakes(monkeypatch)
    doubles = list(maintain_doctor.CPU_CHECKS)

    def broken() -> DoctorResult:
        raise RuntimeError("the check broke")

    broken.__name__ = "check_eole_cli"
    doubles[3] = broken
    monkeypatch.setattr(maintain_doctor, "CPU_CHECKS", tuple(doubles))

    findings = run_cpu_checks()

    assert [(item.check_id, item.status) for item in findings] == [
        *((check_id, "pass") for check_id in CPU_IDS[:3]),
        ("D-EOLE-CLI", "fail"),
        *((check_id, "pass") for check_id in CPU_IDS[4:]),
    ]
    assert findings[3].message == "RuntimeError: the check broke"
    assert calls == [*CPU_NAMES[:3], *CPU_NAMES[4:]]


def test_a_check_with_an_unknown_status_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes(monkeypatch, {"D-PY": ("maybe", "unsure")})

    findings = run_cpu_checks()

    assert (findings[0].check_id, findings[0].status) == ("D-PY", "fail")


def test_the_ids_derived_from_the_function_names_are_the_doctors_ids() -> None:
    derived = [maintain_doctor.check_id_of(check) for check in CPU_CHECKS]

    assert derived == list(CPU_IDS)


def test_a_name_that_is_no_check_name_has_no_id() -> None:
    def helper() -> DoctorResult:
        return DoctorResult("D-X", "pass", "")

    with pytest.raises(ValueError, match="check_"):
        maintain_doctor.check_id_of(helper)
