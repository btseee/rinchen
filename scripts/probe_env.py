"""Phase 0 environment probe: checks Python version, dry-resolves the pinned
dependency stack via `uv lock`, and checks PyPI for per-package wheel
availability on cp314."""

from __future__ import annotations

import datetime
import http.client
import json
import os
import platform as platform_module
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class ProbeFetchError(Exception):
    """Raised when a network fetch fails for a reason other than 'not found'."""


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.hrefs.append(value)


def classify_wheel_tags(filenames: list[str]) -> set[str]:
    """Classify a release's file list into wheel-tag categories.

    Returns a set that may contain any of: "cp314", "cp314t", "abi3",
    "py3-none", "sdist", or "none" (only when no file matched a category).
    Handles compressed tags like "py2.py3" by checking each dot-separated
    component.
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
        python_subtags = python_tag.split(".")
        abi_subtags = abi_tag.split(".")
        if any("cp314t" in t for t in python_subtags) or any(
            "cp314t" in t for t in abi_subtags
        ):
            tags.add("cp314t")
        elif any("cp314" in t for t in python_subtags):
            tags.add("cp314")
        if "abi3" in abi_subtags:
            tags.add("abi3")
        if any(t.startswith("py3") for t in python_subtags) and "none" in abi_subtags:
            tags.add("py3-none")
    if not tags:
        tags.add("none")
    return tags


def derive_status(tags: set[str]) -> str:
    """Derive a package's probe status from its classified wheel tags."""
    real_tags = tags - {"sdist", "none"}
    if real_tags:
        return "ok"
    if "sdist" in tags:
        return "sdist-only"
    return "no-wheel-found"


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
    probed_at: str | None = None,
) -> dict:
    """Assemble the Phase 0 probe manifest structure."""
    return {
        "probed_at": probed_at,
        "python_version": python_version,
        "platform": platform,
        "cpu_count": cpu_count,
        "gpu": gpu,
        "uv_version": uv_version,
        "uv_lock": uv_lock_result,
        "packages": packages,
    }


def render_report(manifest: dict) -> str:
    """Render the Phase 0 / P0.1 STOP-gate report as Markdown."""
    lines = [
        "# Phase 0 Report: Environment Probe (P0.1)",
        "",
        (
            "**Goal:** determine whether the pinned Python 3.14 dependency stack "
            "resolves cleanly, and record per-package wheel availability, before "
            "P0.2-P0.7 build on top of it."
        ),
        "",
    ]
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
    if manifest.get("probed_at"):
        lines.append(f"**Probed at:** {manifest['probed_at']}")
    lock = manifest["uv_lock"]
    lines.append(f"**uv lock:** {'SUCCESS' if lock['success'] else 'FAILED'}")
    lines.append("")

    lines.append("## Blockers")
    lines.append("")
    blockers = []
    if not lock["success"]:
        stderr_display = (lock.get("stderr_tail") or "(no stderr captured)").replace(
            "```", "` ` `"
        )
        blockers.append(
            "`uv lock` fails to resolve the pinned stack on this Python version. "
            "Raw resolver output:\n\n```\n" + stderr_display + "\n```"
        )
    non_ok = {
        name: pkg
        for name, pkg in manifest["packages"].items()
        if pkg["status"] != "ok"
    }
    if non_ok:
        for name, pkg in sorted(non_ok.items()):
            if pkg["status"] == "sdist-only":
                note = "non-blocking for install (pure sdist build, just slower)"
            else:
                note = "may block installation — investigate"
            blockers.append(f"`{name}`: status `{pkg['status']}` ({note}).")
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- None found.")
    lines.append("")

    lines.append("## Package wheel-tag check")
    lines.append("")
    lines.append("| package | requested | checked version | wheel tags | status |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(manifest["packages"]):
        pkg = manifest["packages"][name]
        if "wheel_tags" in pkg:
            tags_display = ", ".join(sorted(pkg["wheel_tags"])) or "none"
        else:
            pypi_tags = ", ".join(sorted(pkg.get("torch_pypi", []))) or "none"
            cu130_tags = ", ".join(sorted(pkg.get("torch_cu130_index", []))) or "none"
            tags_display = f"pypi={pypi_tags}; cu130={cu130_tags}"
        flag = " ⚠️" if pkg["status"] != "ok" else ""
        requested = pkg.get("requested") or "latest"
        checked_version = pkg.get("checked_version") or "-"
        lines.append(
            f"| {name} | {requested} | {checked_version} | {tags_display} | "
            f"{pkg['status']}{flag} |"
        )
    lines.append("")

    lines.append("## Open questions")
    lines.append("")
    lines.append(
        "- Does `eole --no-deps` (installing only the dependencies that "
        "`import eole.bin.main` and `eole train --help` require) avoid the "
        "`pyonmttok` cp314 gap, or does eole's own code import pyonmttok "
        "eagerly? See DECISIONS.md."
    )
    lines.append(
        "- Should this probe be re-run once `pyonmttok` or `eole` ship a "
        "cp314 wheel, to confirm the gap has closed?"
    )
    return "\n".join(lines) + "\n"


def fetch_pypi_json(name: str, version: str | None) -> dict | None:
    """Fetch a package's PyPI JSON metadata.

    Returns None if the specific name/version combination is confirmed not
    to exist (HTTP 404). Raises ProbeFetchError for any other failure
    (network error, timeout, non-404 HTTP error, invalid JSON) so callers
    can distinguish "not found" from "couldn't check".
    """
    url = (
        f"https://pypi.org/pypi/{name}/{version}/json"
        if version
        else f"https://pypi.org/pypi/{name}/json"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ProbeFetchError(f"{name}: HTTP {exc.code}") from exc
    except (OSError, http.client.HTTPException, UnicodeDecodeError) as exc:
        raise ProbeFetchError(f"{name}: {exc}") from exc
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ProbeFetchError(f"{name}: invalid JSON response: {exc}") from exc


def fetch_torch_cu130_index() -> list[str]:
    """Fetch the filenames listed on PyTorch's cu130 wheel index for torch.

    Raises ProbeFetchError on any network failure.
    """
    url = "https://download.pytorch.org/whl/cu130/torch/"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, UnicodeDecodeError) as exc:
        raise ProbeFetchError(f"torch cu130 index: {exc}") from exc
    parser = _LinkExtractor()
    parser.feed(html)
    filenames = []
    for href in parser.hrefs:
        filename = href.split("/")[-1].split("#")[0]
        if filename:
            filenames.append(urllib.parse.unquote(filename))
    return filenames


def run_uv_lock(repo_root: Path) -> dict:
    """Dry-resolve the pinned stack via `uv lock`, in a temp copy of the
    repo so the real uv.lock is never created or mutated by this probe."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(repo_root / "pyproject.toml", tmp_path / "pyproject.toml")
        src_dir = repo_root / "src"
        if src_dir.exists():
            shutil.copytree(src_dir, tmp_path / "src")
        try:
            result = subprocess.run(
                ["uv", "lock"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"success": False, "stderr_tail": f"uv lock failed to run: {exc}"}
        stderr_tail = None if result.returncode == 0 else result.stderr[-4000:]
        if stderr_tail:
            stderr_tail = stderr_tail.replace(str(Path.home()), "~")
        return {"success": result.returncode == 0, "stderr_tail": stderr_tail}


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
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}


def check_python_version() -> bool:
    """Return True if running on Python 3.14.x."""
    return sys.version_info[:2] == (3, 14)


REPO_ROOT = Path(__file__).resolve().parent.parent


def main(out_root: Path | None = None) -> int:
    if not check_python_version():
        print(
            f"ERROR: Rinchen requires Python 3.14.x, found "
            f"{sys.version_info[0]}.{sys.version_info[1]}",
            file=sys.stderr,
        )
        return 1

    repo_root = REPO_ROOT
    output_root = out_root or repo_root
    python_version = platform_module.python_version()
    plat = sys.platform
    cpu_count = os.cpu_count() or 0
    gpu = detect_gpu()

    try:
        uv_version_result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if uv_version_result.returncode == 0:
            uv_version = uv_version_result.stdout.strip()
        else:
            uv_version = (
                f"error (exit {uv_version_result.returncode}): "
                f"{uv_version_result.stderr.strip()}"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        uv_version = f"unavailable: {exc}"

    uv_lock_result = run_uv_lock(repo_root)
    probed_at = datetime.datetime.now(datetime.UTC).isoformat()

    packages: dict = {}
    for name, version in parse_pinned_packages(repo_root / "pyproject.toml"):
        if name == "torch":
            try:
                pypi_data = fetch_pypi_json(name, version)
            except ProbeFetchError as exc:
                packages[name] = {
                    "requested": version,
                    "torch_pypi": [],
                    "torch_cu130_index": [],
                    "status": "fetch-error",
                    "error": str(exc),
                }
                continue
            pypi_filenames = (
                [f["filename"] for f in pypi_data.get("urls", [])] if pypi_data else []
            )
            pypi_tags = classify_wheel_tags(pypi_filenames)
            try:
                cu130_filenames = fetch_torch_cu130_index()
            except ProbeFetchError as exc:
                packages[name] = {
                    "requested": version,
                    "torch_pypi": sorted(pypi_tags),
                    "torch_cu130_index": [],
                    "status": "fetch-error",
                    "error": str(exc),
                }
                continue
            if version:
                cu130_filenames = [
                    f
                    for f in cu130_filenames
                    if f.startswith((f"torch-{version}+", f"torch-{version}-"))
                ]
            cu130_tags = classify_wheel_tags(cu130_filenames)
            packages[name] = {
                "requested": version,
                "checked_version": (pypi_data or {}).get("info", {}).get("version"),
                "torch_pypi": sorted(pypi_tags),
                "torch_cu130_index": sorted(cu130_tags),
                "status": derive_status(cu130_tags),
            }
            continue

        try:
            data = fetch_pypi_json(name, version)
        except ProbeFetchError as exc:
            packages[name] = {
                "requested": version,
                "wheel_tags": [],
                "status": "fetch-error",
                "error": str(exc),
            }
            continue
        if data is None:
            packages[name] = {
                "requested": version,
                "wheel_tags": ["none"],
                "status": "version-not-found",
            }
            continue
        filenames = [f["filename"] for f in data.get("urls", [])]
        tags = classify_wheel_tags(filenames)
        packages[name] = {
            "requested": version,
            "checked_version": data.get("info", {}).get("version"),
            "wheel_tags": sorted(tags),
            "status": derive_status(tags),
        }

    manifest = build_manifest(
        python_version=python_version,
        platform=plat,
        cpu_count=cpu_count,
        gpu=gpu,
        uv_version=uv_version,
        uv_lock_result=uv_lock_result,
        packages=packages,
        probed_at=probed_at,
    )
    report = render_report(manifest)

    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    manifests_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    (manifests_dir / "phase0_env_probe.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "phase_0_env_probe.md").write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
