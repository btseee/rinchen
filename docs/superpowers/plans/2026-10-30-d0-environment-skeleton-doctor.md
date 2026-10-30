# D0 — Environment, Skeleton, Doctor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement D0 of the master plan (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`): pyproject restructure, justfile/CI/pre-commit, the config loader, logging, seed utility, manifest model/store/stage runner, and `rinchen doctor`, with D0's own STOP gate (`just gate` green, doctor CPU checks pass, `docs/VERIFY.md` populated) as the acceptance target — not the project-wide definition of done, which matures over D1-D15.

**Architecture:** A minimal, real `src/rinchen/` package: `config/` (3-layer YAML + CLI + env merge), `utils/` (seed, logging), `manifest/` (hash-based skip mechanism), `errors.py` (domain exceptions), and `cli/` (a typer app with 13 subcommand groups, 12 of them stubs — only `doctor` has real logic). Every doctor check is verified against the real, already-synced project environment during this plan's own execution, not assumed.

**Tech Stack:** `typer`, `pydantic` + `pydantic-settings`, `structlog`, `PyYAML`, `pytest` + `hypothesis` + `pytest-xdist/cov/timeout`, `just`, `pre-commit`.

## Global Constraints

- Every Python invocation runs on CPython 3.14.x. No 3.12/3.13 fallback.
- Keep every already-verified core-ML pin in `pyproject.toml` exactly
  as-is (`torch`, `eole`, `pyonmttok`, `ctranslate2`, `transformers`,
  `tokenizers`, `sentencepiece`, `sacrebleu`, `kiwipiepy`, `fasttext-new`,
  `fasttext-wheel`, `datasketch`) along with the existing
  `[tool.uv.sources]`/`[tool.uv.environments]`/`[tool.uv.index]` entries
  — do not touch those lines except to add new entries alongside them.
- vLLM is NOT added to `pyproject.toml` in any form (already decided;
  documented deviation from the master doc's toolchain section).
- D0's own STOP gate is the acceptance target for this whole plan: `just
  gate` green locally, doctor CPU checks pass, `docs/VERIFY.md` has every
  row (that exists at this point) filled in with a result. The broader
  project-wide "definition of done" (85% coverage, `mypy --strict`
  everywhere, `just tiny` actually working) is NOT this plan's gate.
- `just` (the command runner) is already installed on this dev machine
  (verified: `just 1.59.0` via winget). CI installs it via
  `extractions/setup-just@v2`.
- Domain exceptions (`rinchen.errors`) are used instead of bare
  exceptions anywhere this plan's own code can raise a domain-specific
  error (`ConfigError` in the config loader, `DoctorCheckFailed` is
  available for `cli/doctor.py` though the checks themselves report
  structured results rather than raising).
- A real, already-installed `eole` 0.6.4 environment is available on
  this machine (`uv sync --dev` succeeded earlier this session) — the
  doctor task verifies its checks against this real environment, not a
  mock.

---

### Task 1: Restructure `pyproject.toml` and record resolved versions

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/STACK.md`

**Interfaces:**
- Produces: an updated, real `uv.lock` and a synced `.venv` with every
  new package from the master doc's toolchain section installed. Every
  later task in this plan assumes these packages are importable.

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `[project]` `dependencies` list, add `[project.scripts]`, and
replace the `dev` group, keeping every other section (`[tool.uv]`,
`[[tool.uv.index]]`, `[tool.uv.sources]`, `[build-system]`,
`[tool.hatch.build.targets.wheel]`, `[tool.pytest.ini_options]`,
`[tool.ruff.lint.per-file-ignores]`) exactly as they are today:

```toml
[project]
name = "rinchen"
version = "0.0.0"
description = "Open-source, Mongolian-centric machine translation trained from scratch."
requires-python = "==3.14.*"
dependencies = [
    "torch==2.14.1",
    "eole==0.6.4",
    "pyonmttok==1.38.1",
    "ctranslate2==4.8.4",
    "transformers==5.21.0",
    "tokenizers==0.23.3",
    "sentencepiece==0.2.1",
    "sacrebleu==2.6.0",
    "kiwipiepy==0.24.0",
    "fasttext-new==0.1.0",
    "fasttext-wheel==0.9.2",
    "datasketch==2.0.0",
    "datasets",
    "huggingface_hub",
    "pyarrow",
    "polars",
    "duckdb",
    "numpy",
    "pandas",
    "scipy",
    "trafilatura",
    "warcio",
    "httpx",
    "selectolax",
    "beautifulsoup4",
    "sentence-transformers",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "typer",
    "rich",
    "structlog",
    "jinja2",
    "python-multipart",
    "prometheus-client",
    "regex",
    "PyYAML",
]

[project.scripts]
rinchen = "rinchen.cli.main:app"

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "mypy",
    "hypothesis",
    "pytest-xdist",
    "pytest-cov",
    "pytest-timeout",
    "pre-commit",
    "ty",
]
```

(Everything from `[tool.uv]` onward in the current file is unchanged —
only insert the new dependency entries, the `[project.scripts]` table,
and the expanded `dev` group shown above.)

- [ ] **Step 2: Run the real `uv lock`**

Run: `uv lock`
Expected: exits 0.

- [ ] **Step 3: Run the real `uv sync --dev`**

Run: `uv sync --dev`
Expected: exits 0. If any NEW package (not one of the already-solved
`pyonmttok`/`fasttext-wheel` ones) fails to install, stop and report the
exact error — do not guess at a fix. Every new package was verified to
have a cp314-safe release during this plan's design (pure-Python wheel,
explicit cp314 wheel, or a Rust binary shipped as a CPython-ABI-
independent `py3-none-<platform>` wheel), so a real failure here would
be a new, unplanned finding.

- [ ] **Step 4: Record resolved versions in `docs/STACK.md`**

```bash
mkdir -p docs
{
  echo "# STACK.md"
  echo
  echo "Resolved package versions for the current \`uv.lock\`, recorded per"
  echo "CLAUDE.md's hard constraint (\"record every resolved package version\")."
  echo "Regenerate after any \`uv lock --upgrade\`."
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo '```'
  uv pip list
  echo '```'
} > docs/STACK.md
cat docs/STACK.md
```

Expected: `docs/STACK.md` contains a real, non-empty package list
including `torch`, `eole`, `typer`, `structlog`, `pydantic-settings`,
`hypothesis`, etc.

- [ ] **Step 5: Confirm the existing test suite still passes**

Run: `uv run pytest -q`
Expected: all previously-passing tests still pass (no regression from
the dependency bump).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock docs/STACK.md
git commit -m "D0: restructure pyproject.toml to the master plan's toolchain, record resolved versions"
```

---

### Task 2: Tooling files (`justfile`, CI, pre-commit) and old stub cleanup

**Files:**
- Create: `justfile`
- Create: `.pre-commit-config.yaml`
- Modify: `.github/workflows/ci.yml`
- Delete: `src/rinchen/align/`, `src/rinchen/convert/`, `src/rinchen/data/`
  (orphaned stub packages from the superseded Phase 0-14 structure — no
  equivalent in the new `<repo_layout>`; confirmed each `__init__.py` is
  completely empty before deleting)

**Interfaces:**
- Produces: `just gate`, `just test`, `just doctor`, `just tiny`, `just
  accept` recipes. Later tasks in this plan (and all of D1+) invoke
  `just gate` as their own verification step.

- [ ] **Step 1: Confirm the three old stub packages are empty, then remove them**

```bash
for d in align convert data; do
  if [ -s "src/rinchen/$d/__init__.py" ]; then
    echo "NOT EMPTY, STOP: src/rinchen/$d/__init__.py" >&2
    exit 1
  fi
done
rm -rf src/rinchen/align src/rinchen/convert src/rinchen/data
```

Expected: the loop prints nothing (all three were empty) and the three
directories are gone. If the loop DOES print "NOT EMPTY, STOP", halt and
report back — do not delete a non-empty file.

- [ ] **Step 2: Write the justfile**

```just
# Rinchen development commands. Run `just --list` to see all recipes.

gate:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy src
    uv run pytest -m "not network" -q

test:
    uv run pytest -q

doctor:
    uv run rinchen doctor

tiny:
    echo "just tiny: not implemented until D15 (tiny end-to-end pipeline). See docs/research/2026-10-29-rinchen-dev-agent-prompt.md milestone D15."

accept:
    echo "just accept: not implemented until the full project-wide definition of done is met. See <mission> in docs/research/2026-10-29-rinchen-dev-agent-prompt.md."
```

- [ ] **Step 3: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.3
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix]
```

(`v0.17.3` matches the `ruff` version already resolved into `uv.lock`
from Task 1 — confirm this with `uv pip show ruff` and adjust the `rev`
if it differs; do not leave a mismatched pin.)

- [ ] **Step 4: Update `.github/workflows/ci.yml`**

Replace the file's contents entirely with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python 3.14
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install just
        uses: extractions/setup-just@v2

      - name: Sync dependencies
        run: uv sync --frozen --group dev

      - name: Gate (ruff format, ruff check, mypy, fast tests)
        run: just gate

      - name: Tiny pipeline (stub until D15)
        run: just tiny

      - name: Upload coverage
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage.xml
          if-no-files-found: ignore
```

- [ ] **Step 5: Run `just gate` locally and confirm it runs (not necessarily green yet — most modules don't exist until later tasks in this plan)**

Run: `just gate`
Expected: `ruff format --check .` and `ruff check .` run (likely clean,
since there's little new code yet); `mypy src` runs against the existing
`src/rinchen/` stub tree; `pytest -m "not network" -q` runs the existing
test suite. Record the actual output in your report — this step is
about confirming the plumbing works, not about D0 being done yet (later
tasks add the real modules this gate will check).

- [ ] **Step 6: Commit**

```bash
git add justfile .pre-commit-config.yaml .github/workflows/ci.yml
git add -u src/rinchen
git commit -m "D0: add justfile, update CI to use just, remove orphaned src/rinchen stub packages"
```

---

### Task 3: `errors.py`, `utils/seed.py`, `utils/logging.py`

**Files:**
- Create: `src/rinchen/errors.py`
- Create: `src/rinchen/utils/__init__.py`
- Create: `src/rinchen/utils/seed.py`
- Create: `src/rinchen/utils/logging.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_errors.py`
- Create: `tests/unit/test_seed.py`
- Create: `tests/unit/test_logging.py`

**Interfaces:**
- Produces: `rinchen.errors.{RinchenError,ConfigError,DataError,
  UpstreamError,ParityError,DoctorCheckFailed}`. Task 4 (`config/`)
  imports `ConfigError`.
- Produces: `rinchen.utils.seed.set_all(seed: int) -> None`.
- Produces: `rinchen.utils.logging.get(name: str) -> BoundLogger`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/__init__.py`: empty file.

`tests/unit/test_errors.py`:

```python
"""Tests for rinchen.errors exception hierarchy."""
import pytest

from rinchen.errors import (
    ConfigError,
    DataError,
    DoctorCheckFailed,
    ParityError,
    RinchenError,
    UpstreamError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [ConfigError, DataError, UpstreamError, ParityError, DoctorCheckFailed],
)
def test_each_domain_exception_is_a_rinchen_error(exc_cls):
    assert issubclass(exc_cls, RinchenError)
    assert issubclass(exc_cls, Exception)


def test_domain_exception_carries_message():
    with pytest.raises(ConfigError, match="bad config"):
        raise ConfigError("bad config")
```

`tests/unit/test_seed.py`:

```python
"""Tests for rinchen.utils.seed.set_all."""
import random

from rinchen.utils import seed


def test_set_all_makes_random_deterministic():
    seed.set_all(42)
    first = [random.random() for _ in range(5)]

    seed.set_all(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_set_all_different_seeds_differ():
    seed.set_all(1)
    first = random.random()

    seed.set_all(2)
    second = random.random()

    assert first != second
```

`tests/unit/test_logging.py`:

```python
"""Tests for rinchen.utils.logging.get."""
import json

from rinchen.utils import logging as rinchen_logging


def test_get_returns_logger_that_emits_json(capsys):
    logger = rinchen_logging.get("test_logger")
    logger.info("stage_start", stage="demo", count=3)

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["event"] == "stage_start"
    assert payload["stage"] == "demo"
    assert payload["count"] == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/ -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'rinchen.errors'` (and similarly for `rinchen.utils`).

- [ ] **Step 3: Write `src/rinchen/errors.py`**

```python
"""Domain exceptions for Rinchen, per <coding_standards> in the master plan."""


class RinchenError(Exception):
    """Base class for all Rinchen domain exceptions."""


class ConfigError(RinchenError):
    """Raised for invalid or missing configuration."""


class DataError(RinchenError):
    """Raised for corrupt, missing, or invalid data."""


class UpstreamError(RinchenError):
    """Raised when an upstream dependency (Eole, CTranslate2, etc.) misbehaves."""


class ParityError(RinchenError):
    """Raised when a fallback implementation fails to match its reference within tolerance."""


class DoctorCheckFailed(RinchenError):
    """Raised by a `rinchen doctor` check that fails."""
```

- [ ] **Step 4: Write `src/rinchen/utils/__init__.py`** (empty file)

- [ ] **Step 5: Write `src/rinchen/utils/seed.py`**

```python
"""Deterministic seeding across every RNG Rinchen touches."""
from __future__ import annotations

import random


def set_all(seed: int) -> None:
    """Seed Python's random, NumPy, and (if importable) PyTorch."""
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
```

- [ ] **Step 6: Write `src/rinchen/utils/logging.py`**

```python
"""Structured JSON logging setup via structlog."""
from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(stream=sys.stderr, format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger configured for JSON output on stderr."""
    _configure()
    return structlog.get_logger(name)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/ -v`
Expected: all pass.

- [ ] **Step 8: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/errors.py src/rinchen/utils tests/unit`
Run: `uv run mypy src/rinchen/errors.py src/rinchen/utils`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add src/rinchen/errors.py src/rinchen/utils tests/unit
git commit -m "D0: add errors, seed, and logging utilities"
```

---

### Task 4: Config loader (`config/`)

**Files:**
- Create: `src/rinchen/config/__init__.py`
- Create: `src/rinchen/config/schema.py`
- Create: `src/rinchen/config/settings.py`
- Create: `src/rinchen/config/loader.py`
- Create: `configs/base.yaml`
- Create: `configs/profiles/tiny.yaml`
- Create: `configs/profiles/full.yaml`
- Create: `tests/unit/test_config_loader.py`

**Interfaces:**
- Consumes: `rinchen.errors.ConfigError` from Task 3.
- Produces: `rinchen.config.loader.load(profile: str, *overlays: Path,
  cli_overrides: dict[str, str] | None = None) -> RinchenConfig`. Later
  milestones (D1+) import this to resolve their own stage configs.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config_loader.py`:

```python
"""Tests for rinchen.config.loader.load's base -> profile -> overlay -> CLI merge."""
import pytest

from rinchen.config import loader
from rinchen.errors import ConfigError


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_load_merges_base_and_profile(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\npaths:\n  data_root: /base\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "paths:\n  data_root: /tiny\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny")

    assert config.profile == "tiny"
    assert config.seed == 1  # from base, not overridden by profile
    assert config.paths["data_root"] == "/tiny"  # profile overrides base


def test_load_applies_overlay_file(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "seed: 2\n")
    overlay = tmp_path / "overlay.yaml"
    _write(overlay, "seed: 3\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", overlay)

    assert config.seed == 3


def test_cli_override_wins_over_everything(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "seed: 2\n")
    overlay = tmp_path / "overlay.yaml"
    _write(overlay, "seed: 3\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", overlay, cli_overrides={"seed": "4"})

    assert config.seed == 4


def test_cli_override_sets_nested_path(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\npaths:\n  data_root: /base\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "{}\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    config = loader.load("tiny", cli_overrides={"paths.data_root": "/overridden"})

    assert config.paths["data_root"] == "/overridden"


def test_load_raises_config_error_for_missing_profile(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.delenv("RINCHEN_SEED", raising=False)

    with pytest.raises(ConfigError):
        loader.load("nonexistent")


def test_env_var_overrides_seed(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write(base, "seed: 1\n")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _write(profiles_dir / "tiny.yaml", "{}\n")

    monkeypatch.setattr(loader, "BASE_CONFIG_PATH", base)
    monkeypatch.setattr(loader, "PROFILES_DIR", profiles_dir)
    monkeypatch.setenv("RINCHEN_SEED", "99")

    config = loader.load("tiny")

    assert config.seed == 99
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_loader.py -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'rinchen.config'`.

- [ ] **Step 3: Write `src/rinchen/config/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/rinchen/config/schema.py`**

```python
"""Pydantic schema for Rinchen's resolved configuration.

Minimal at D0: just enough fields to prove the merge mechanism. Later
milestones add real fields as they need profile-driven config.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RinchenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    seed: int = 0
    paths: dict[str, str] = {}
```

- [ ] **Step 5: Write `src/rinchen/config/settings.py`**

```python
"""Environment-variable overlay for Rinchen configuration, via pydantic-settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RinchenEnvSettings(BaseSettings):
    """`RINCHEN_*` environment variables, applied as the final overlay."""

    model_config = SettingsConfigDict(env_prefix="RINCHEN_", extra="ignore")

    seed: int | None = None
```

- [ ] **Step 6: Write `src/rinchen/config/loader.py`**

```python
"""Three-layer config loader: base -> profile -> overlay files -> CLI --set,
then RINCHEN_* env vars as the final overlay."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from rinchen.config.schema import RinchenConfig
from rinchen.config.settings import RinchenEnvSettings
from rinchen.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dotted_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        next_cursor = cursor.setdefault(part, {})
        if not isinstance(next_cursor, dict):
            raise ConfigError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = next_cursor
    cursor[parts[-1]] = value


def load(
    profile: str,
    *overlays: Path,
    cli_overrides: dict[str, str] | None = None,
) -> RinchenConfig:
    """Merge base -> profile -> overlay YAML files -> CLI --set overrides -> env vars."""
    merged: dict[str, Any] = {}
    if BASE_CONFIG_PATH.is_file():
        merged = _deep_merge(merged, _read_yaml(BASE_CONFIG_PATH))

    profile_path = PROFILES_DIR / f"{profile}.yaml"
    merged = _deep_merge(merged, _read_yaml(profile_path))
    merged["profile"] = profile

    for overlay_path in overlays:
        merged = _deep_merge(merged, _read_yaml(overlay_path))

    for dotted_key, raw_value in (cli_overrides or {}).items():
        _apply_dotted_override(merged, dotted_key, raw_value)

    env = RinchenEnvSettings()
    if env.seed is not None:
        merged["seed"] = env.seed

    try:
        return RinchenConfig(**merged)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"invalid resolved configuration: {exc}") from exc
```

- [ ] **Step 7: Write the real config files**

`configs/base.yaml`:

```yaml
seed: 1234
paths: {}
```

`configs/profiles/tiny.yaml`:

```yaml
seed: 1234
paths:
  data_root: data/tiny
```

`configs/profiles/full.yaml`:

```yaml
# Populated by later milestones as real stages are built; not used by D0.
seed: 1234
paths: {}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_loader.py -v`
Expected: all 6 pass.

- [ ] **Step 9: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/config tests/unit/test_config_loader.py`
Run: `uv run mypy src/rinchen/config`
Expected: both clean.

- [ ] **Step 10: Commit**

```bash
git add src/rinchen/config configs/base.yaml configs/profiles tests/unit/test_config_loader.py
git commit -m "D0: add 3-layer config loader (base, profile, overlay, CLI, env)"
```

---

### Task 5: Manifest model, store, and stage runner (`manifest/`)

**Files:**
- Create: `src/rinchen/manifest/__init__.py`
- Create: `src/rinchen/manifest/model.py`
- Create: `src/rinchen/manifest/store.py`
- Create: `src/rinchen/manifest/stage.py`
- Create: `tests/unit/test_manifest.py`

**Interfaces:**
- Produces: `Manifest` (pydantic model matching the `<contracts>`
  section's schema), `store.write(stage_dir, manifest) -> Path`,
  `store.read(stage_dir) -> Manifest | None`, `stage.sha256_of(path) ->
  str`, `stage.build_inputs(paths) -> list[ManifestFile]`,
  `stage.should_skip(stage_dir, input_paths, config_sha256, *, force=False)
  -> bool`. D1+ stage implementations consume these directly.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_manifest.py`:

```python
"""Tests for the manifest model, store, and hash-based stage skip mechanism."""
from pathlib import Path

from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest


def _write_manifest(stage_dir: Path, config_sha256: str, inputs: list[Path]) -> None:
    manifest = Manifest(
        stage="test_stage",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256=config_sha256,
        seed=0,
        inputs=stage.build_inputs(inputs),
    )
    store.write(stage_dir, manifest)


def test_store_round_trips(tmp_path):
    stage_dir = tmp_path / "stage"
    manifest = Manifest(
        stage="demo",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
    )

    path = store.write(stage_dir, manifest)
    assert path.is_file()

    loaded = store.read(stage_dir)
    assert loaded is not None
    assert loaded.stage == "demo"
    assert loaded.schema == "rinchen.manifest/1"


def test_store_read_returns_none_when_missing(tmp_path):
    assert store.read(tmp_path / "nonexistent") is None


def test_identical_inputs_and_config_are_skipped(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is True


def test_changed_config_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-different") is False


def test_changed_input_contents_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])
    input_file.write_text("changed", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_no_prior_manifest_never_skips(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_force_always_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc", force=True) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_manifest.py -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'rinchen.manifest'`.

- [ ] **Step 3: Write `src/rinchen/manifest/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/rinchen/manifest/model.py`**

```python
"""Pydantic model for a stage's manifest.json, per <contracts> in the master plan."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    bytes: int | None = None
    rows: int | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema: str = "rinchen.manifest/1"
    stage: str
    created: str
    code_version: str
    config_sha256: str
    seed: int
    inputs: list[ManifestFile] = []
    outputs: list[ManifestFile] = []
    metrics: dict[str, float] = {}
    tool_versions: dict[str, str] = {}
```

- [ ] **Step 5: Write `src/rinchen/manifest/store.py`**

```python
"""Read/write a stage's manifest.json."""
from __future__ import annotations

from pathlib import Path

from rinchen.manifest.model import Manifest


def manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "manifest.json"


def write(stage_dir: Path, manifest: Manifest) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(stage_dir)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read(stage_dir: Path) -> Manifest | None:
    path = manifest_path(stage_dir)
    if not path.is_file():
        return None
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
```

- [ ] **Step 6: Write `src/rinchen/manifest/stage.py`**

```python
"""Stage runner: skip a stage iff inputs+config sha match an existing manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path

from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inputs(input_paths: list[Path]) -> list[ManifestFile]:
    return [
        ManifestFile(path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size)
        for path in input_paths
    ]


def should_skip(
    stage_dir: Path,
    input_paths: list[Path],
    config_sha256: str,
    *,
    force: bool = False,
) -> bool:
    """True iff a prior manifest exists with identical input hashes and config hash."""
    if force:
        return False
    existing = store.read(stage_dir)
    if existing is None:
        return False
    if existing.config_sha256 != config_sha256:
        return False
    existing_hashes = {item.path: item.sha256 for item in existing.inputs}
    for input_path in input_paths:
        key = str(input_path)
        if existing_hashes.get(key) != sha256_of(input_path):
            return False
    return True
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_manifest.py -v`
Expected: all 7 pass.

- [ ] **Step 8: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/manifest tests/unit/test_manifest.py`
Run: `uv run mypy src/rinchen/manifest`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add src/rinchen/manifest tests/unit/test_manifest.py
git commit -m "D0: add manifest model, store, and hash-based stage skip mechanism"
```

---

### Task 6: CLI skeleton (`cli/`)

**Files:**
- Modify: `src/rinchen/cli/__init__.py` (currently empty; stays empty —
  package marker only)
- Create: `src/rinchen/cli/main.py`
- Create: `src/rinchen/cli/data.py`
- Create: `src/rinchen/cli/spm.py`
- Create: `src/rinchen/cli/filter.py`
- Create: `src/rinchen/cli/mine.py`
- Create: `src/rinchen/cli/convert_script.py`
- Create: `src/rinchen/cli/synth.py`
- Create: `src/rinchen/cli/train.py`
- Create: `src/rinchen/cli/export.py`
- Create: `src/rinchen/cli/eval.py`
- Create: `src/rinchen/cli/serve.py`
- Create: `src/rinchen/cli/annotate.py`
- Create: `src/rinchen/cli/tiny.py`
- Create: `src/rinchen/__main__.py`
- Create: `src/rinchen/py.typed` (empty marker file)
- Create: `tests/unit/test_cli_help.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `rinchen.cli.main.app` (a `typer.Typer`), registering 13
  subcommand groups by name: `data`, `spm`, `filter`, `mine`,
  `convert-script`, `synth`, `train`, `export`, `eval`, `serve`,
  `annotate`, `doctor`, `tiny`. Task 7 (`cli/doctor.py`) is the one real
  implementation this task's `main.py` registers; this task creates
  `main.py` assuming `cli/doctor.py` will exist (Task 7 creates it) —
  if executed out of order, `main.py`'s import of `doctor` will fail
  until Task 7 lands. Keep these two tasks in the written order.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli_help.py`:

```python
"""Every rinchen CLI subcommand must respond to --help with exit code 0."""
import pytest
from typer.testing import CliRunner

from rinchen.cli.main import app

runner = CliRunner()

SUBCOMMANDS = [
    "data",
    "spm",
    "filter",
    "mine",
    "convert-script",
    "synth",
    "train",
    "export",
    "eval",
    "serve",
    "annotate",
    "doctor",
    "tiny",
]


def test_root_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand):
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_cli_help.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.cli.main'`.

- [ ] **Step 3: Write the 12 stub sub-apps**

Each of these files (`data.py`, `spm.py`, `filter.py`, `mine.py`,
`convert_script.py`, `synth.py`, `train.py`, `export.py`, `eval.py`,
`serve.py`, `annotate.py`, `tiny.py`) gets the same two-line body, with
a docstring naming its own subcommand and owning milestone. Example for
`src/rinchen/cli/data.py`:

```python
"""Stub CLI sub-app for `rinchen data` -- real commands land in D1 (Data acquisition)."""
import typer

app = typer.Typer(no_args_is_help=True)
```

Repeat for the other 11, changing only the docstring's command name and
milestone reference:
- `spm.py`: "`rinchen spm`" / "D3 (Normalization & SentencePiece)"
- `filter.py`: "`rinchen filter`" / "D4 (Filtering & mining)"
- `mine.py`: "`rinchen mine`" / "D4 (Filtering & mining)"
- `convert_script.py`: "`rinchen convert-script`" / "D5 (Cyrillic<->traditional converter)"
- `synth.py`: "`rinchen synth`" / "D6 (Synthetic data)"
- `train.py`: "`rinchen train`" / "D7 (Multilingual ablations) / D8 (Teacher training)"
- `export.py`: "`rinchen export`" / "D11 (Export, quantization, serving)"
- `eval.py`: "`rinchen eval`" / "D12 (Robustness) / D13 (Final eval)"
- `serve.py`: "`rinchen serve`" / "D11 (Export, quantization, serving)"
- `annotate.py`: "`rinchen annotate`" / "D2 (Human test-set tooling)"
- `tiny.py`: "`rinchen tiny`" / "D15 (Tiny end-to-end pipeline)"

- [ ] **Step 4: Write `src/rinchen/cli/main.py`**

```python
"""Top-level `rinchen` typer app, registering one sub-app per CLI module."""
from __future__ import annotations

import typer

from rinchen.cli import (
    annotate,
    convert_script,
    data,
    doctor,
    eval as eval_cli,
    export,
    filter as filter_cli,
    mine,
    serve,
    spm,
    synth,
    tiny,
    train,
)

app = typer.Typer(name="rinchen", no_args_is_help=True)

app.add_typer(data.app, name="data")
app.add_typer(spm.app, name="spm")
app.add_typer(filter_cli.app, name="filter")
app.add_typer(mine.app, name="mine")
app.add_typer(convert_script.app, name="convert-script")
app.add_typer(synth.app, name="synth")
app.add_typer(train.app, name="train")
app.add_typer(export.app, name="export")
app.add_typer(eval_cli.app, name="eval")
app.add_typer(serve.app, name="serve")
app.add_typer(annotate.app, name="annotate")
app.add_typer(doctor.app, name="doctor")
app.add_typer(tiny.app, name="tiny")
```

(This imports `rinchen.cli.doctor`, created in Task 7 — run Task 7
immediately after this one, before trying to get this test green.)

- [ ] **Step 5: Write `src/rinchen/__main__.py`**

```python
"""Entry point for `python -m rinchen`."""
from rinchen.cli.main import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Create `src/rinchen/py.typed`** (empty file — PEP 561 marker)

- [ ] **Step 7: Confirm `src/rinchen/cli/__init__.py` is still empty** (it already is; no change)

- [ ] **Step 8: Run the test — expect it to still fail until Task 7's `doctor.py` exists**

Run: `uv run pytest tests/unit/test_cli_help.py -v`
Expected at this point: FAILS with `ModuleNotFoundError: No module named
'rinchen.cli.doctor'` (not the earlier `main` error) — this confirms
every piece built so far is wired correctly and the only missing piece
is Task 7's real doctor implementation. Do not mark this task complete
until Task 7 lands and this test passes; if running tasks out of order
is unavoidable for some reason, note it clearly in your report.

- [ ] **Step 9: Commit this task's files** (the test will go green once Task 7 lands; committing now keeps history granular)

```bash
git add src/rinchen/cli src/rinchen/__main__.py src/rinchen/py.typed tests/unit/test_cli_help.py
git commit -m "D0: add CLI skeleton with 12 stub subcommand groups"
```

---

### Task 7: `rinchen doctor`

**Files:**
- Create: `src/rinchen/cli/doctor.py`
- Create: `docs/VERIFY.md`
- Create: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `rinchen.cli.main.app` imports this module (Task 6).
- Produces: `rinchen.cli.doctor.app` (the registered sub-Typer),
  `rinchen.cli.doctor.run_all() -> list[DoctorResult]`,
  `rinchen.cli.doctor.DoctorResult` (dataclass: `check_id: str, status:
  str, message: str`).

This task's checks were designed against real, already-synced output on
this machine (not guessed) — two real discrepancies from the master
plan's literal doctor-check description were found and are handled
explicitly below, not silently papered over:
1. The master doc describes checking `eole`'s CLI via `--help` text,
   naming subcommands including "model average". The REAL, installed
   `eole` 0.6.4 CLI's top-level subcommands are
   `{convert, build_vocab, predict, serve, train, tools}` — there is no
   `model` subcommand group at all, and no `average` command anywhere
   in the installed package (confirmed: no file matching `*average*`
   anywhere under the installed `eole` package). `D-EOLE-CLI` below
   checks the four subcommands that actually exist
   (`build_vocab`/`train`/`predict`/`convert`) and records this
   discrepancy in its message and in `docs/VERIFY.md`, rather than
   checking something that doesn't exist. The master doc's own
   `build_ourselves_specs` section already names Rinchen's own
   `train/average.py` as the fallback for exactly this situation
   (D8/D10 scope, not D0's).
2. `D-WHEELS` checks installed-distribution *metadata* (via
   `importlib.metadata.version`), not `import <module>`, because
   several distribution names don't match their import name
   (`PyYAML`→`yaml`, `beautifulsoup4`→`bs4`, `python-multipart`→
   `multipart`) and guessing that mapping generically is fragile. This
   still proves every pinned package actually resolved and installed,
   which is the check's real purpose.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_doctor.py`:

```python
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
```

(`test_check_py_fails_when_version_is_not_314` fakes `sys.version_info`
via monkeypatch, exactly as the spec requires — it does not actually
run on Python 3.13. The other tests in this file run for real against
the current machine's synced environment — this is deliberate: D0's own
doctor checks ARE the acceptance gate for the pyonmttok/fasttext-wheel
work, so faking them with mocks would defeat their purpose.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_doctor.py -v`
Expected: FAILS with `ModuleNotFoundError: No module named 'rinchen.cli.doctor'`.

- [ ] **Step 3: Write `src/rinchen/cli/doctor.py`**

```python
"""`rinchen doctor`: environment checks with clear pass/fail/skip results."""
from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=False)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass(slots=True, frozen=True)
class DoctorResult:
    check_id: str
    status: str  # "pass" | "fail" | "skip"
    message: str


def check_py() -> DoctorResult:
    if sys.version_info[:2] == (3, 14):
        return DoctorResult("D-PY", "pass", f"CPython {sys.version.split()[0]}")
    return DoctorResult(
        "D-PY", "fail", f"expected CPython 3.14.x, got {sys.version.split()[0]}"
    )


def _parse_pinned_package_names(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    names = []
    for entry in data["project"]["dependencies"]:
        name = entry.split("=")[0].split(">")[0].split("<")[0].split("[")[0].strip()
        names.append(name)
    return names


def check_wheels() -> DoctorResult:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    names = _parse_pinned_package_names(pyproject_path)
    failures: list[str] = []
    versions: list[str] = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
            versions.append(f"{name}=={version}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(name)
    if failures:
        return DoctorResult("D-WHEELS", "fail", f"not installed: {', '.join(failures)}")
    return DoctorResult("D-WHEELS", "pass", "; ".join(versions))


def check_eole_py() -> DoctorResult:
    try:
        importlib.import_module("eole")
    except Exception as exc:
        return DoctorResult("D-EOLE-PY", "fail", f"import eole failed: {exc}")
    return DoctorResult("D-EOLE-PY", "pass", "import eole succeeded")


def _run_help(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout + result.stderr)[-500:]


def check_eole_cli() -> DoctorResult:
    eole_bin = shutil.which("eole")
    if eole_bin is None:
        return DoctorResult(
            "D-EOLE-CLI",
            "fail",
            "'eole' console script not found on PATH (run via `uv run rinchen doctor`)",
        )
    subcommands = ["build_vocab", "train", "predict", "convert"]
    failures = []
    for sub in subcommands:
        ok, output = _run_help([eole_bin, sub, "--help"])
        if not ok:
            failures.append(f"{sub}: {output}")
    if failures:
        return DoctorResult("D-EOLE-CLI", "fail", "; ".join(failures))
    note = (
        " (NOTE: 'eole model average' from the master plan does not exist in"
        " installed eole 0.6.4's CLI -- top-level subcommands are"
        " {convert,build_vocab,predict,serve,train,tools}, no 'model' group;"
        " see docs/VERIFY.md)"
    )
    return DoctorResult(
        "D-EOLE-CLI", "pass", f"{len(subcommands)} subcommands parsed --help{note}"
    )


def check_ct2_conv() -> DoctorResult:
    ok, output = _run_help(
        [sys.executable, "-m", "ctranslate2.converters.eole_ct2", "--help"]
    )
    if ok:
        return DoctorResult(
            "D-CT2-CONV", "pass", "python -m ctranslate2.converters.eole_ct2 --help"
        )
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep.name for ep in entry_points if "eole" in ep.name.lower()]
    if matches:
        return DoctorResult("D-CT2-CONV", "pass", f"found console scripts: {matches}")
    return DoctorResult(
        "D-CT2-CONV",
        "fail",
        f"module --help failed ({output}) and no eole console script found",
    )


def check_ct2_import() -> DoctorResult:
    torch_was_loaded_before = "torch" in sys.modules
    try:
        importlib.import_module("ctranslate2")
    except Exception as exc:
        return DoctorResult("D-CT2-IMPORT", "fail", f"import ctranslate2 failed: {exc}")
    torch_loaded_as_side_effect = not torch_was_loaded_before and "torch" in sys.modules
    note = (
        " (torch imported as a side effect, see CTranslate2 issue #2079)"
        if torch_loaded_as_side_effect
        else ""
    )
    return DoctorResult("D-CT2-IMPORT", "pass", f"import ctranslate2 succeeded{note}")


def check_torch_cuda() -> DoctorResult:
    try:
        import torch
    except ImportError as exc:
        return DoctorResult("D-TORCH-CUDA", "skip", f"torch not importable: {exc}")
    if torch.cuda.is_available():
        return DoctorResult(
            "D-TORCH-CUDA", "pass", f"CUDA available: {torch.cuda.get_device_name(0)}"
        )
    return DoctorResult("D-TORCH-CUDA", "skip", "no GPU available")


ALL_CHECKS = [
    check_py,
    check_wheels,
    check_eole_py,
    check_eole_cli,
    check_ct2_conv,
    check_ct2_import,
    check_torch_cuda,
]


def run_all() -> list[DoctorResult]:
    return [check() for check in ALL_CHECKS]


@app.callback(invoke_without_command=True)
def main() -> None:
    """Run every doctor check and print a summary; exit non-zero on any failure."""
    results = run_all()
    failed = False
    for result in results:
        marker = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[result.status]
        typer.echo(f"[{marker}] {result.check_id}: {result.message}")
        if result.status == "fail":
            failed = True
    raise typer.Exit(code=1 if failed else 0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_doctor.py -v`
Expected: all 4 pass, run for real against this machine's synced
environment.

- [ ] **Step 5: Run `rinchen doctor` for real**

Run: `uv run rinchen doctor`
Expected: exit 0, with every check printing `[OK]` except
`D-TORCH-CUDA`, which prints `[SKIP]` on this CPU-only dev machine. If
any check prints `[FAIL]`, stop and report the exact message — this is
the real acceptance proof for the pyonmttok/fasttext-wheel work, so a
failure here would be serious.

- [ ] **Step 6: Now go back and confirm Task 6's CLI test passes**

Run: `uv run pytest tests/unit/test_cli_help.py -v`
Expected: all 14 pass (previously blocked on this module not existing).

- [ ] **Step 7: Write `docs/VERIFY.md`**

```markdown
# VERIFY.md

Every unverified fact from the master plan gets a row here once its
owning milestone implements a doctor check for it, per
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`'s state-file
convention. This file grows across D0-D15; it is not meant to be
complete until the project is.

## D0 checks

| Check ID | Result | Notes |
|---|---|---|
| D-PY | pass | CPython 3.14.x confirmed via `sys.version_info`. |
| D-WHEELS | pass | Every `[project.dependencies]` entry resolves via `importlib.metadata.version` in the synced environment. |
| D-EOLE-PY | pass | `import eole` succeeds (closes the pyonmttok cp314 blocker's real acceptance test). |
| D-EOLE-CLI | pass (with discrepancy noted) | `build_vocab`/`train`/`predict`/`convert` all parse `--help`. The master plan's doctor-check description also named "model average" (citing eole issue #136/PR #137); the installed eole 0.6.4 CLI has no `model` subcommand group at all (`eole --help` lists only `{convert,build_vocab,predict,serve,train,tools}`), and no file matching `*average*` exists anywhere in the installed package. Fallback: Rinchen's own `train/average.py` checkpoint averager (`build_ourselves_specs`, D8/D10 scope) is the path forward when that milestone needs averaging — not re-attempted here. |
| D-CT2-CONV | pass | `python -m ctranslate2.converters.eole_ct2 --help` succeeds directly. |
| D-CT2-IMPORT | pass | `import ctranslate2` succeeds; confirmed CTranslate2 issue #2079 (`import ctranslate2` eagerly imports torch) reproduces on this machine -- also independently observed that `eole --help` itself imports torch eagerly too. |
| D-TORCH-CUDA | skip | No GPU on this dev machine; CI also skips (ubuntu-latest has no GPU). |
```

- [ ] **Step 8: Run ruff and mypy**

Run: `uv run ruff check src/rinchen/cli/doctor.py tests/unit/test_doctor.py`
Run: `uv run mypy src/rinchen/cli/doctor.py`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add src/rinchen/cli/doctor.py docs/VERIFY.md tests/unit/test_doctor.py
git commit -m "D0: add rinchen doctor with 7 checks, document eole-model-average discrepancy"
```

---

### Task 8: Runbook, docs updates, and final STOP-gate verification

**Files:**
- Create: `docs/runbooks/00_environment.md`
- Modify: `STATE.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7 — this task is the final
  acceptance check for the whole plan.

- [ ] **Step 1: Write `docs/runbooks/00_environment.md`**

```markdown
# Runbook: 00 — Environment

## Dev machine setup (CPU)

1. Install Python 3.14.x and `uv` (https://docs.astral.sh/uv/).
2. Install `just` (https://github.com/casey/just) — e.g. `winget install
   Casey.Just` on Windows, or your platform's package manager.
3. Clone the repo, then: `uv sync --dev`
4. Verify: `uv run rinchen doctor` — every check should print `[OK]` or
   `[SKIP]` (GPU checks skip on a CPU-only machine). Any `[FAIL]` means
   something in the pinned stack isn't installed correctly; see
   `docs/VERIFY.md` for what each check id verifies and the known
   `D-EOLE-CLI` discrepancy (eole 0.6.4 has no `model average`
   subcommand, contrary to the master plan's original description).

## GPU host setup

1. Confirm a CUDA 13-compatible driver is installed:
   `nvidia-smi` should report a driver version supporting CUDA 13.x.
2. Follow the CPU setup steps above (same `uv sync --dev`; `torch` is
   pinned to a CUDA 13.0 wheel via `[tool.uv.sources]`'s
   `pytorch-cu130` index).
3. Verify: `uv run rinchen doctor --gpu` is NOT yet a real flag at D0
   (the `D-TORCH-CUDA` check always runs as part of the normal `rinchen
   doctor` invocation and reports `pass` with the detected GPU name, or
   `skip` if none is found) — a `--gpu`-specific flag that fails loudly
   instead of skipping is left for a later milestone if it proves
   useful; for now, read the `D-TORCH-CUDA` line in the normal output.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on `ubuntu-latest`
with CPython 3.14, syncs with `uv sync --frozen --group dev`, then runs
`just gate` (ruff format check, ruff check, mypy, fast tests) and `just
tiny` (currently a stub that always succeeds — real from D15 onward).
GPU checks are not run in CI (no GPU runner); `D-TORCH-CUDA` reports
`skip` there, which is expected and does not fail the gate.
```

- [ ] **Step 2: Update `STATE.md`**

Read the current file first (it has a banner describing both cp314
blockers as resolved and "Current next step: start D0"). Replace that
banner's "Current next step" line and add a new section documenting
D0's completion:

```markdown
Current next step: start D1 (Data acquisition). D0 (environment,
skeleton, doctor) is complete: `pyproject.toml` restructured to the
master plan's toolchain (see `docs/STACK.md` for resolved versions),
`justfile`/CI/pre-commit in place, `rinchen doctor` implemented with 7
checks (all pass or skip-as-expected on this machine; see
`docs/VERIFY.md`), config loader / logging / seed / manifest modules
implemented with passing tests, CLI skeleton registers all 13
subcommand groups. [EVIDENCE:
`docs/superpowers/plans/2026-10-30-d0-environment-skeleton-doctor.md`,
`docs/VERIFY.md`, 2026-10-30.] One real discrepancy from the master
plan found and documented: eole 0.6.4's CLI has no `model average`
subcommand (see `docs/VERIFY.md`'s `D-EOLE-CLI` row) -- the
build-ourselves checkpoint averager (D8/D10) is the planned fallback,
not yet built.
```

- [ ] **Step 3: Update `CLAUDE.md`**

Update the "Current phase" section to reflect D0's completion and point
at D1 next, mirroring `STATE.md`'s wording but kept to 1-2 sentences per
`CLAUDE.md`'s own brevity convention. Also add `docs/STACK.md`,
`docs/VERIFY.md`, `docs/runbooks/`, `justfile`, and
`.pre-commit-config.yaml` to the repo layout block if they aren't
already implied by existing lines.

- [ ] **Step 4: Run the full real STOP-gate check**

Run: `just gate`
Expected: exits 0 (ruff format --check, ruff check, mypy, and `pytest -m
"not network"` all pass).

Run: `uv run rinchen doctor`
Expected: exit 0, matching Task 7 Step 5's output.

Run: `uv run pytest -q` (the full suite, including network-marked tests)
Expected: all pass.

If anything fails here that passed in its own task, investigate before
closing out — a regression introduced by a later task against an
earlier one's work is exactly what this final check exists to catch.

- [ ] **Step 5: Commit**

```bash
git add docs/runbooks STATE.md CLAUDE.md
git commit -m "D0: add environment runbook, update STATE.md/CLAUDE.md for D0 completion"
```
