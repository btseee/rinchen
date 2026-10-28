from probe_env import classify_wheel_tags


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


import json
import tempfile
from pathlib import Path

from probe_env import build_manifest, parse_pinned_packages, render_report


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
        path.write_text(content)
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
