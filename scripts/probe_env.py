"""Phase 0 environment probe: checks Python version, dry-resolves the pinned
dependency stack via `uv lock`, and checks PyPI for per-package wheel
availability on cp314."""

from __future__ import annotations

import tomllib
from pathlib import Path


def classify_wheel_tags(filenames: list[str]) -> set[str]:
    """Classify a release's file list into wheel-tag categories.

    Returns a set that may contain any of: "cp314", "cp314t", "abi3",
    "py3-none", "sdist", or "none" (only when the input is empty).
    """
    tags: set[str] = set()
    for name in filenames:
        if name.endswith((".tar.gz", ".zip")):
            tags.add("sdist")
            continue
        if not name.endswith(".whl"):
            continue
        stem = name[: -len(".whl")]
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        python_tag, abi_tag = parts[-3], parts[-2]
        if "cp314t" in python_tag or "cp314t" in abi_tag:
            tags.add("cp314t")
        elif "cp314" in python_tag:
            tags.add("cp314")
        if abi_tag == "abi3":
            tags.add("abi3")
        if python_tag.startswith("py3") and abi_tag == "none":
            tags.add("py3-none")
    if not tags:
        tags.add("none")
    return tags


def parse_pinned_packages(pyproject_path: Path) -> list[tuple[str, str | None]]:
    """Read [project.dependencies] and [dependency-groups].dev from a
    pyproject.toml, returning (name, version_or_none) pairs."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    specs = list(data.get("project", {}).get("dependencies", []))
    specs += list(data.get("dependency-groups", {}).get("dev", []))
    results: list[tuple[str, str | None]] = []
    for spec in specs:
        if "==" in spec:
            name, version = spec.split("==", 1)
            results.append((name.strip(), version.strip()))
        else:
            results.append((spec.strip(), None))
    return results


def build_manifest(
    python_version: str,
    platform: str,
    cpu_count: int,
    gpu: dict | None,
    uv_version: str,
    uv_lock_result: dict,
    packages: dict,
) -> dict:
    """Assemble the Phase 0 probe manifest structure."""
    return {
        "python_version": python_version,
        "platform": platform,
        "cpu_count": cpu_count,
        "gpu": gpu,
        "uv_version": uv_version,
        "uv_lock": uv_lock_result,
        "packages": packages,
    }


def render_report(manifest: dict) -> str:
    """Render the Phase 0 STOP-gate report as Markdown."""
    lines = ["# Phase 0 Report: Environment Probe", ""]
    py_ok = manifest["python_version"].startswith("3.14")
    lines.append(
        f"**Python version:** {manifest['python_version']} "
        f"({'OK' if py_ok else 'FAIL - must be 3.14.x'})"
    )
    lines.append(f"**Platform:** {manifest['platform']}")
    lines.append(f"**CPU count:** {manifest['cpu_count']}")
    gpu = manifest.get("gpu")
    lines.append(f"**GPU:** {gpu['name'] if gpu else 'none detected'}")
    lines.append(f"**uv version:** {manifest['uv_version']}")
    lock = manifest["uv_lock"]
    lines.append(f"**uv lock:** {'SUCCESS' if lock['success'] else 'FAILED'}")
    if not lock["success"]:
        lines.append("")
        lines.append("```")
        lines.append(lock.get("stderr_tail") or "(no stderr captured)")
        lines.append("```")
    lines.append("")
    lines.append("## Package wheel-tag check")
    lines.append("")
    lines.append("| package | requested | wheel tags | status |")
    lines.append("|---|---|---|---|")
    for name in sorted(manifest["packages"]):
        pkg = manifest["packages"][name]
        if "wheel_tags" in pkg:
            tags_display = ", ".join(sorted(pkg["wheel_tags"])) or "none"
        else:
            tags_display = (
                f"pypi={sorted(pkg.get('torch_pypi', []))}, "
                f"cu130={sorted(pkg.get('torch_cu130_index', []))}"
            )
        flag = " ⚠️" if pkg["status"] != "ok" else ""
        requested = pkg.get("requested") or "latest"
        lines.append(f"| {name} | {requested} | {tags_display} | {pkg['status']}{flag} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    sys.exit(0)
