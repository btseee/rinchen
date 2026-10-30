"""Tests for rinchen.cli.doctor's checks."""
import sys

from rinchen.cli import doctor


def test_check_py_passes_on_real_cpython_314():
    result = doctor.check_py()
    assert result.check_id == "D-PY"
    assert result.status == "pass"


def test_check_py_fails_when_version_is_not_314(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
    result = doctor.check_py()
    assert result.check_id == "D-PY"
    assert result.status == "fail"


def test_check_wheels_passes_for_real_installed_environment():
    result = doctor.check_wheels()
    assert result.check_id == "D-WHEELS"
    assert result.status == "pass"
    assert "torch" in result.message


def test_run_all_returns_one_result_per_check():
    results = doctor.run_all()
    check_ids = {result.check_id for result in results}
    assert check_ids == {
        "D-PY",
        "D-WHEELS",
        "D-EOLE-PY",
        "D-EOLE-CLI",
        "D-CT2-CONV",
        "D-CT2-IMPORT",
        "D-TORCH-CUDA",
    }
