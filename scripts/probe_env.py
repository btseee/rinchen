"""Phase 0 environment probe: checks Python version, dry-resolves the pinned
dependency stack via `uv lock`, and checks PyPI for per-package wheel
availability on cp314."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


def fetch_pypi_json(name: str, version: str | None) -> dict | None:
    """Fetch a package's PyPI JSON metadata. Returns None if not found."""
    url = (
        f"https://pypi.org/pypi/{name}/{version}/json"
        if version
        else f"https://pypi.org/pypi/{name}/json"
    )
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_torch_cu130_index() -> list[str]:
    """Fetch the filenames listed on PyTorch's cu130 wheel index for torch."""
    url = "https://download.pytorch.org/whl/cu130/torch/"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    filenames = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        filename = href.split("/")[-1].split("#")[0]
        if filename:
            filenames.append(filename)
    return filenames


def run_uv_lock(repo_root: Path) -> dict:
    """Dry-resolve the pinned stack via `uv lock`, in a temp copy of the repo
    so the real uv.lock is never created or mutated by this probe."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(repo_root / "pyproject.toml", tmp_path / "pyproject.toml")
        src_dir = repo_root / "src"
        if src_dir.exists():
            shutil.copytree(src_dir, tmp_path / "src")
        result = subprocess.run(
            ["uv", "lock"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "success": result.returncode == 0,
            "stderr_tail": None if result.returncode == 0 else result.stderr[-4000:],
        }


def detect_gpu() -> dict | None:
    """Detect an NVIDIA GPU via nvidia-smi. Returns None if absent."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}


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
