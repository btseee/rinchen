from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from probe_env import (
    build_manifest,
    classify_wheel_tags,
    derive_status,
    detect_gpu,
    fetch_pypi_json,
    fetch_torch_cu130_index,
    parse_pinned_packages,
    render_report,
    run_uv_lock,
)


def test_classify_cp314_wheel():
    tags = classify_wheel_tags(["torch-2.14.1-cp314-cp314-win_amd64.whl"])
    assert tags == {"cp314"}


def test_classify_cp314t_wheel():
    tags = classify_wheel_tags(
        ["ctranslate2-4.8.4-cp314-cp314t-manylinux_2_28_x86_64.whl"]
    )
    assert tags == {"cp314t"}


def test_classify_abi3_wheel():
    tags = classify_wheel_tags(
        ["fasttext_new-0.1.0-cp39-abi3-manylinux_2_28_x86_64.whl"]
    )
    assert tags == {"abi3"}


def test_classify_pure_python_wheel():
    tags = classify_wheel_tags(["eole-0.6.4-py3-none-any.whl"])
    assert tags == {"py3-none"}


def test_classify_sdist_only():
    tags = classify_wheel_tags(["eole-0.6.4.tar.gz"])
    assert tags == {"sdist"}


def test_classify_no_files():
    tags = classify_wheel_tags([])
    assert tags == {"none"}


def test_classify_mixed_files_combine_tags():
    tags = classify_wheel_tags(
        [
            "pkg-1.0-cp314-cp314-win_amd64.whl",
            "pkg-1.0-cp314-cp314t-manylinux_2_28_x86_64.whl",
            "pkg-1.0.tar.gz",
        ]
    )
    assert tags == {"cp314", "cp314t", "sdist"}


def test_classify_compressed_py2_py3_tag_counts_as_py3():
    tags = classify_wheel_tags(
        ["warcio-1.8.1-py2.py3-none-any.whl", "warcio-1.8.1.tar.gz"]
    )
    assert tags == {"py3-none", "sdist"}


def test_classify_py2_only_tag_does_not_count_as_py3():
    tags = classify_wheel_tags(["sixexample-1.0-py2-none-any.whl"])
    assert tags == {"none"}


def test_derive_status_ok_when_real_tag_present():
    assert derive_status({"cp314"}) == "ok"
    assert derive_status({"py3-none"}) == "ok"
    assert derive_status({"abi3", "sdist"}) == "ok"


def test_derive_status_sdist_only():
    assert derive_status({"sdist"}) == "sdist-only"


def test_derive_status_no_wheel_found():
    assert derive_status({"none"}) == "no-wheel-found"


def test_parse_pinned_packages_reads_pinned_and_unpinned():
    content = """
[project]
dependencies = [
    "torch==2.14.1",
    "numpy",
]

[dependency-groups]
dev = ["pytest", "ruff==0.1.0"]
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pyproject.toml"
        path.write_text(content, encoding="utf-8")
        packages = parse_pinned_packages(path)
    assert ("torch", "2.14.1") in packages
    assert ("numpy", None) in packages
    assert ("pytest", None) in packages
    assert ("ruff", "0.1.0") in packages


def test_build_manifest_assembles_expected_structure():
    manifest = build_manifest(
        python_version="3.14.7",
        platform="win32",
        cpu_count=8,
        gpu=None,
        uv_version="0.13.5",
        uv_lock_result={"success": True, "stderr_tail": None},
        packages={"torch": {"requested": "2.14.1", "wheel_tags": ["cp314"], "status": "ok"}},
    )
    assert manifest["python_version"] == "3.14.7"
    assert manifest["gpu"] is None
    assert manifest["uv_lock"]["success"] is True
    assert manifest["packages"]["torch"]["status"] == "ok"
    json.dumps(manifest)  # must be JSON-serializable


def test_build_manifest_includes_probed_at_when_given():
    manifest = build_manifest(
        python_version="3.14.7",
        platform="win32",
        cpu_count=8,
        gpu=None,
        uv_version="0.13.5",
        uv_lock_result={"success": True, "stderr_tail": None},
        packages={},
        probed_at="2026-10-28T00:00:00+00:00",
    )
    assert manifest["probed_at"] == "2026-10-28T00:00:00+00:00"


def test_render_report_includes_python_and_package_table():
    manifest = build_manifest(
        python_version="3.14.7",
        platform="win32",
        cpu_count=8,
        gpu=None,
        uv_version="0.13.5",
        uv_lock_result={"success": True, "stderr_tail": None},
        packages={
            "torch": {"requested": "2.14.1", "wheel_tags": ["cp314"], "status": "ok"},
            "pyonmttok": {"requested": None, "wheel_tags": ["none"], "status": "no-wheel-found"},
        },
    )
    report = render_report(manifest)
    assert "3.14.7" in report
    assert "torch" in report
    assert "pyonmttok" in report
    assert "⚠️" in report  # the flagged pyonmttok row
    assert "## Blockers" in report
    assert "## Open questions" in report


def test_render_report_flags_uv_lock_failure():
    manifest = build_manifest(
        python_version="3.14.7",
        platform="win32",
        cpu_count=8,
        gpu=None,
        uv_version="0.13.5",
        uv_lock_result={"success": False, "stderr_tail": "error: no solution found"},
        packages={},
    )
    report = render_report(manifest)
    assert "FAILED" in report
    assert "error: no solution found" in report
    assert "## Blockers" in report


@pytest.mark.network
def test_fetch_pypi_json_known_package():
    data = fetch_pypi_json("packaging", None)
    assert data is not None
    assert "info" in data


@pytest.mark.network
def test_fetch_pypi_json_unknown_package_returns_none():
    data = fetch_pypi_json("this-package-definitely-does-not-exist-rinchen-probe", None)
    assert data is None


@pytest.mark.network
def test_fetch_torch_cu130_index_returns_filenames():
    filenames = fetch_torch_cu130_index()
    assert isinstance(filenames, list)
    assert any("torch" in f for f in filenames)


@pytest.mark.subprocess
def test_detect_gpu_returns_none_or_dict():
    result = detect_gpu()
    assert result is None or {"name", "driver_version", "memory_total"} <= result.keys()


@pytest.mark.network
@pytest.mark.subprocess
def test_run_uv_lock_on_real_pyproject():
    repo_root = Path(__file__).resolve().parent.parent
    result = run_uv_lock(repo_root)
    assert "success" in result
    assert "stderr_tail" in result


@pytest.mark.network
@pytest.mark.subprocess
def test_main_writes_manifest_and_report(tmp_path):
    import probe_env

    exit_code = probe_env.main(out_root=tmp_path)
    assert exit_code == 0

    manifest_path = tmp_path / "manifests" / "phase0_env_probe.json"
    report_path = tmp_path / "reports" / "phase_0_env_probe.md"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["python_version"].startswith("3.14")
    assert "packages" in manifest

    report = report_path.read_text(encoding="utf-8")
    assert "# Phase 0 Report: Environment Probe (P0.1)" in report


def test_main_fails_fast_on_wrong_python_version(monkeypatch):
    import probe_env

    monkeypatch.setattr(probe_env.sys, "version_info", (3, 13, 0, "final", 0))
    exit_code = probe_env.main()
    assert exit_code == 1
