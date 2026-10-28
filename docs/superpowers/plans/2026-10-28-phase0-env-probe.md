# Phase 0 / P0.1 Environment Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/probe_env.py`, a script that checks this machine's Python version, dry-resolves the pinned dependency stack via `uv lock`, and checks PyPI for per-package cp314/cp314t/abi3 wheel availability — writing `manifests/phase0_env_probe.json` and `reports/phase_0.md` so Phase 0's known blockers (pyonmttok cp314 gap, vLLM exclusion) get real evidence instead of guesses.

**Architecture:** One script, `scripts/probe_env.py`, with deterministic pure functions (wheel-tag classification, pyproject.toml parsing, manifest/report building) separated from thin IO wrappers (PyPI HTTP fetch, `uv lock` subprocess, `nvidia-smi` subprocess), wired together by a `main()` entrypoint. Pure functions get full unit tests; IO wrappers get one integration test each, marked so they can be excluded in constrained environments.

**Tech Stack:** Python 3.14, `httpx` (HTTP), `beautifulsoup4` (HTML parsing of the PyTorch index), `tomllib` (stdlib, parsing pyproject.toml), `pytest` with custom markers.

## Global Constraints

- Hard-fail (non-zero exit) if `sys.version_info[:2] != (3, 14)`, checked at the very start of `main()`.
- No package installation anywhere in this plan — `uv lock` only, run in a temp copy of the repo so the real `uv.lock` (which doesn't exist yet) isn't created as a side effect of testing.
- Every dependency this plan uses (`httpx`, `beautifulsoup4`) is already pinned in `pyproject.toml` — do not add new dependencies.
- Exact function names and signatures below are binding — later tasks import them by these exact names.
- Write tests before implementation for every function (TDD).

---

### Task 1: pytest config + wheel-tag classifier

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/probe_env.py`
- Create: `tests/test_probe_env.py`

**Interfaces:**
- Produces: `classify_wheel_tags(filenames: list[str]) -> set[str]` in `scripts/probe_env.py`, used by Task 3 (IO wrappers feed it real filenames) and Task 4 (main orchestration).
- Produces: pytest config (`pythonpath = ["scripts"]`, markers `network` and `subprocess` registered) in `pyproject.toml`, which every later task's tests rely on to `import probe_env` and to mark IO tests.

- [ ] **Step 1: Add pytest config to `pyproject.toml`**

Open `pyproject.toml` and add this section (anywhere after the existing tables, e.g. after `[tool.hatch.build.targets.wheel]`):

```toml
[tool.pytest.ini_options]
pythonpath = ["scripts"]
markers = [
    "network: tests that make real network calls",
    "subprocess: tests that spawn real subprocesses",
]
```

- [ ] **Step 2: Write the failing test for `classify_wheel_tags`**

Create `tests/test_probe_env.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_probe_env.py -v`
Expected: `ModuleNotFoundError: No module named 'probe_env'` (the file doesn't exist yet) — or collection error. Either confirms the test can't pass yet.

- [ ] **Step 4: Create `scripts/probe_env.py` with `classify_wheel_tags`**

```python
"""Phase 0 environment probe: checks Python version, dry-resolves the pinned
dependency stack via `uv lock`, and checks PyPI for per-package wheel
availability on cp314."""

from __future__ import annotations


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


if __name__ == "__main__":
    import sys

    sys.exit(0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_probe_env.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml scripts/probe_env.py tests/test_probe_env.py
git commit -m "feat: add wheel-tag classifier for Phase 0 env probe"
```

---

### Task 2: pyproject.toml dependency parser + manifest/report builders

**Files:**
- Modify: `scripts/probe_env.py`
- Modify: `tests/test_probe_env.py`

**Interfaces:**
- Consumes: nothing new from Task 1 besides the file existing.
- Produces: `parse_pinned_packages(pyproject_path: Path) -> list[tuple[str, str | None]]`, `build_manifest(python_version: str, platform: str, cpu_count: int, gpu: dict | None, uv_version: str, uv_lock_result: dict, packages: dict) -> dict`, and `render_report(manifest: dict) -> str` — all used by Task 4's `main()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_probe_env.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_probe_env.py -v -k "parse_pinned or build_manifest or render_report"`
Expected: `ImportError` (names not defined yet in `probe_env`).

- [ ] **Step 3: Implement `parse_pinned_packages`, `build_manifest`, `render_report`**

Add to `scripts/probe_env.py` (above the `if __name__ == "__main__":` block):

```python
import tomllib
from pathlib import Path


def parse_pinned_packages(pyproject_path: Path) -> list[tuple[str, str | None]]:
    """Read [project.dependencies] and [dependency-groups].dev from a
    pyproject.toml, returning (name, version_or_none) pairs."""
    data = tomllib.loads(pyproject_path.read_text())
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
```

Add `from pathlib import Path` and `import tomllib` at the top of the file if not already present (they are new in this step).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_probe_env.py -v`
Expected: all tests from Task 1 and Task 2 PASS (11 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_env.py tests/test_probe_env.py
git commit -m "feat: add pyproject parser and manifest/report builders for env probe"
```

---

### Task 3: IO wrappers (PyPI fetch, torch cu130 index, uv lock, GPU detection)

**Files:**
- Modify: `scripts/probe_env.py`
- Modify: `tests/test_probe_env.py`

**Interfaces:**
- Consumes: `classify_wheel_tags` from Task 1 (not called directly by these functions, but by Task 4's orchestration using their output).
- Produces: `fetch_pypi_json(name: str, version: str | None) -> dict | None`, `fetch_torch_cu130_index() -> list[str]`, `run_uv_lock(repo_root: Path) -> dict`, `detect_gpu() -> dict | None` — all used by Task 4's `main()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_probe_env.py`:

```python
import pytest

from probe_env import detect_gpu, fetch_pypi_json, fetch_torch_cu130_index, run_uv_lock


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
def test_run_uv_lock_on_real_pyproject(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    result = run_uv_lock(repo_root)
    assert "success" in result
    assert "stderr_tail" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_probe_env.py -v -k "fetch_pypi or fetch_torch or detect_gpu or run_uv_lock"`
Expected: `ImportError` (names not defined yet).

- [ ] **Step 3: Implement the IO wrappers**

Add to `scripts/probe_env.py` (below the imports, above `classify_wheel_tags` or anywhere at module level — keep related functions grouped):

```python
import subprocess
import tempfile
import shutil

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_probe_env.py -v`
Expected: all tests PASS (16 total). The network-marked tests require internet access from this machine; if they fail due to connectivity rather than a code defect, note that in your self-review but do not alter the implementation to work around a connectivity problem.

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_env.py tests/test_probe_env.py
git commit -m "feat: add PyPI, torch index, uv lock, and GPU detection IO wrappers"
```

---

### Task 4: main() orchestration, CLI entrypoint, end-to-end smoke test

**Files:**
- Modify: `scripts/probe_env.py`
- Modify: `tests/test_probe_env.py`

**Interfaces:**
- Consumes: `classify_wheel_tags`, `parse_pinned_packages`, `build_manifest`, `render_report` (Tasks 1-2), `fetch_pypi_json`, `fetch_torch_cu130_index`, `run_uv_lock`, `detect_gpu` (Task 3).
- Produces: `main() -> int`, the script's CLI entrypoint, used by Task 5 (running the probe for real).

- [ ] **Step 1: Write the failing end-to-end smoke test**

Append to `tests/test_probe_env.py`:

```python
import json


@pytest.mark.network
@pytest.mark.subprocess
def test_main_writes_manifest_and_report(tmp_path, monkeypatch):
    import probe_env

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(probe_env, "REPO_ROOT", repo_root)
    exit_code = probe_env.main()
    assert exit_code == 0

    manifest_path = repo_root / "manifests" / "phase0_env_probe.json"
    report_path = repo_root / "reports" / "phase_0.md"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["python_version"].startswith("3.14")
    assert "packages" in manifest

    report = report_path.read_text()
    assert "# Phase 0 Report: Environment Probe" in report


def test_main_fails_fast_on_wrong_python_version(monkeypatch):
    import probe_env

    monkeypatch.setattr(probe_env.sys, "version_info", (3, 13, 0, "final", 0))
    exit_code = probe_env.main()
    assert exit_code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_probe_env.py -v -k "test_main"`
Expected: FAIL with an `AttributeError` — `probe_env` has neither `REPO_ROOT` nor `main` defined yet (the current `if __name__ == "__main__":` block has no `main()` function or module-level `REPO_ROOT`).

- [ ] **Step 3: Implement `main()` and wire the CLI entrypoint**

Replace the `if __name__ == "__main__":` block at the bottom of `scripts/probe_env.py` with:

```python
import platform as platform_module
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if sys.version_info[:2] != (3, 14):
        print(
            f"ERROR: Rinchen requires Python 3.14.x, found "
            f"{sys.version_info[0]}.{sys.version_info[1]}",
            file=sys.stderr,
        )
        return 1

    repo_root = REPO_ROOT
    python_version = platform_module.python_version()
    plat = sys.platform
    cpu_count = __import__("os").cpu_count() or 0
    gpu = detect_gpu()

    uv_version_result = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True
    )
    uv_version = uv_version_result.stdout.strip()

    uv_lock_result = run_uv_lock(repo_root)

    packages: dict = {}
    for name, version in parse_pinned_packages(repo_root / "pyproject.toml"):
        if name == "torch":
            pypi_data = fetch_pypi_json(name, version)
            pypi_filenames = (
                [f["filename"] for f in pypi_data.get("urls", [])]
                if pypi_data
                else []
            )
            pypi_tags = classify_wheel_tags(pypi_filenames)
            cu130_filenames = fetch_torch_cu130_index()
            if version:
                cu130_filenames = [
                    f for f in cu130_filenames if f.startswith(f"torch-{version}")
                ]
            cu130_tags = classify_wheel_tags(cu130_filenames)
            real_tags = cu130_tags - {"sdist", "none"}
            packages[name] = {
                "requested": version,
                "torch_pypi": sorted(pypi_tags),
                "torch_cu130_index": sorted(cu130_tags),
                "status": "ok" if real_tags else "needs-review",
            }
            continue

        data = fetch_pypi_json(name, version)
        if data is None:
            packages[name] = {
                "requested": version,
                "wheel_tags": ["none"],
                "status": "version-not-found",
            }
            continue
        filenames = [f["filename"] for f in data.get("urls", [])]
        tags = classify_wheel_tags(filenames)
        real_tags = tags - {"sdist", "none"}
        if real_tags:
            status = "ok"
        elif "sdist" in tags:
            status = "sdist-only"
        else:
            status = "no-wheel-found"
        packages[name] = {
            "requested": version,
            "wheel_tags": sorted(tags),
            "status": status,
        }

    manifest = build_manifest(
        python_version=python_version,
        platform=plat,
        cpu_count=cpu_count,
        gpu=gpu,
        uv_version=uv_version,
        uv_lock_result=uv_lock_result,
        packages=packages,
    )
    report = render_report(manifest)

    manifests_dir = repo_root / "manifests"
    reports_dir = repo_root / "reports"
    manifests_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    (manifests_dir / "phase0_env_probe.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (reports_dir / "phase_0.md").write_text(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Also add `import json` and `import os` is avoided by using `__import__("os")` inline above — instead, add a clean `import os` at the top of the file with the other imports, and change `__import__("os").cpu_count()` to `os.cpu_count()`. Add `import json` near the top imports too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_probe_env.py -v`
Expected: all tests PASS (18 total). The `test_main_writes_manifest_and_report` test will actually write `manifests/phase0_env_probe.json` and `reports/phase_0.md` to the real repo as a side effect — that's expected and matches Task 5's goal; do not treat those files appearing as test pollution.

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_env.py tests/test_probe_env.py
git commit -m "feat: wire env probe main() orchestration and CLI entrypoint"
```

---

### Task 5: Run the probe for real and record Phase 0 findings

**Files:**
- Create (generated, not hand-written): `manifests/phase0_env_probe.json`
- Create (generated, not hand-written): `reports/phase_0.md`
- Modify: `STATE.md`
- Modify: `DECISIONS.md`

**Interfaces:**
- Consumes: `main()` from Task 4.
- Produces: nothing consumed by other tasks — this is the terminal task, producing the actual Phase 0 P0.1 evidence.

- [ ] **Step 1: Run the probe for real**

Run: `python scripts/probe_env.py`
Expected: exit code 0 (confirmed Python 3.14 on this machine). This may have already written the output files as a side effect of Task 4's test suite — re-run it directly here anyway so the files reflect a clean, intentional run, not leftover test output.

- [ ] **Step 2: Inspect the generated report**

Run: `cat reports/phase_0.md` (or open the file) and read it. Specifically check:
- Whether `eole`'s dependency `pyonmttok` was actually probed — it isn't a direct dependency in `pyproject.toml`, so this script does NOT check it directly (only direct dependencies are probed). Note in your report file (Step 4 below) that `pyonmttok`'s cp314 status remains unverified by this script — probing transitive dependencies is out of scope for P0.1 as specced, and would need `uv lock`'s success/failure as the signal instead (check whether `uv_lock.success` is `true` or `false` in the manifest, and if `false`, check whether the stderr mentions `pyonmttok`).
- Whether `uv lock` succeeded or failed, and why if it failed.
- Which packages, if any, show `status` other than `"ok"`.

- [ ] **Step 3: Update STATE.md with the real findings**

Read the current `STATE.md`, then edit it to replace the "Known blockers for Phase 0 to resolve" section with the actual findings from Step 2 — for example, if `uv lock` failed because of `pyonmttok`, state that plainly with the exact error line; if it succeeded, say so and remove that blocker. Keep the file's overall structure (heading, phase status line) intact; only update the blockers content to reflect reality instead of a prediction. Add a line noting P0.1 is complete and pointing to `reports/phase_0.md` for the full table.

- [ ] **Step 4: Update DECISIONS.md if the findings change a prior decision**

If Step 2 confirms the `pyonmttok` cp314 gap is real (either `uv lock` fails naming it, or — if you want higher confidence — manually check `https://pypi.org/pypi/pyonmttok/json` for wheel tags), add a new `DECISIONS.md` bullet: `- Confirmed: pyonmttok has no cp314 wheel as of 2026-10-28. [EVIDENCE: Phase 0 P0.1 probe, see reports/phase_0.md]. Follow-up (P0.1's STATE.md blocker) needed before eole can be installed on this machine.` If instead `uv lock` succeeds cleanly with no issues, add a bullet recording that positive result instead: `- Confirmed: the full pinned stack dry-resolves cleanly via uv lock on Python 3.14.7. [EVIDENCE: Phase 0 P0.1 probe, 2026-10-28, see reports/phase_0.md].` Use whichever is actually true — do not guess.

- [ ] **Step 5: Commit**

```bash
git add manifests/phase0_env_probe.json reports/phase_0.md STATE.md DECISIONS.md
git commit -m "chore: run Phase 0 P0.1 environment probe and record findings"
```

---

## Final verification

- [ ] Run `python -m pytest tests/test_probe_env.py -v` and confirm all tests pass (network/subprocess tests included, since this machine has internet and uv).
- [ ] Run `git log --oneline -5` and confirm five commits exist for Tasks 1-5 in order.
- [ ] Open `reports/phase_0.md` and confirm it reads as a genuine Phase 0 STOP-gate report a human could act on.
