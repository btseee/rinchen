# Repo Scaffold + CLAUDE.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Rinchen repo scaffold — uv workspace, full directory tree, CLAUDE.md, STATE.md, DECISIONS.md, and a CI stub — so every later phase (data, tokenizer, training, serving) has a home. No training/data/metric code is written here.

**Architecture:** A `src/rinchen/` package with one empty subpackage per future subsystem (data, text, align, filter, synth, metrics, eval, convert, serve, cli), a `pyproject.toml` pinning the Python-3.14-verified dependency versions from `docs/research/2026-10-28-python314-stack-and-prompt.md`, and three root control files (CLAUDE.md, STATE.md, DECISIONS.md) that later phases read before acting.

**Tech Stack:** Python 3.14, uv workspace, hatchling build backend, GitHub Actions CI.

## Global Constraints

- `requires-python = "==3.14.*"` — no 3.12/3.13 fallback, per spec `docs/superpowers/specs/2026-10-28-repo-scaffold-design.md`.
- Dependency versions pinned exactly as listed in `docs/research/2026-10-28-python314-stack-and-prompt.md` Section 2 ("use" rows) — do not substitute newer/older versions in this plan.
- `CLAUDE.md` must stay under 200 lines.
- No `uv sync` / package install in this plan — that is Phase 0's job, out of scope here.
- No code under `src/rinchen/*` beyond empty `__init__.py` files in this plan.
- State lives in files (`STATE.md`, `DECISIONS.md`), not chat memory.

---

### Task 1: uv workspace (`pyproject.toml`)

**Files:**
- Create: `pyproject.toml`

**Interfaces:**
- Produces: a `[project]` table with pinned dependencies that Task 6 (CI) references via `uv sync`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "rinchen"
version = "0.0.0"
description = "Open-source, Mongolian-centric machine translation trained from scratch."
requires-python = "==3.14.*"
dependencies = [
    "torch==2.14.1",
    "eole==0.6.4",
    "ctranslate2==4.8.4",
    "transformers==5.21.0",
    "tokenizers==0.23.3",
    "sentencepiece==0.2.1",
    "sacrebleu==2.6.0",
    "kiwipiepy==0.24.0",
    "fasttext-new==0.1.0",
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
]

[dependency-groups]
dev = ["pytest", "ruff", "mypy"]
gen = ["vllm==0.32.0"]

[tool.uv]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rinchen"]
```

- [ ] **Step 2: Verify it's valid TOML**

Run: `python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add uv workspace pyproject.toml pinned to Python 3.14 stack"
```

---

### Task 2: Directory tree

**Files:**
- Create: `configs/spm/.gitkeep`, `configs/eole/.gitkeep`, `configs/serve/.gitkeep`
- Create: `src/rinchen/__init__.py`
- Create: `src/rinchen/data/__init__.py`
- Create: `src/rinchen/text/__init__.py`
- Create: `src/rinchen/align/__init__.py`
- Create: `src/rinchen/filter/__init__.py`
- Create: `src/rinchen/synth/__init__.py`
- Create: `src/rinchen/metrics/__init__.py`
- Create: `src/rinchen/eval/__init__.py`
- Create: `src/rinchen/convert/__init__.py`
- Create: `src/rinchen/serve/__init__.py`
- Create: `src/rinchen/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/.gitkeep`, `plans/.gitkeep`, `reports/.gitkeep`, `manifests/.gitkeep`, `annotation_app/.gitkeep`
- Create: `data/raw/.gitkeep`, `data/clean/.gitkeep`, `data/mined/.gitkeep`, `data/synth/.gitkeep`, `data/test/.gitkeep`

**Interfaces:**
- Produces: `rinchen` as an importable namespace (`import rinchen`) with ten empty subpackages that later phases fill in one file at a time.

- [ ] **Step 1: Create all directories and placeholder files**

```bash
mkdir -p configs/spm configs/eole configs/serve
mkdir -p src/rinchen/data src/rinchen/text src/rinchen/align src/rinchen/filter \
         src/rinchen/synth src/rinchen/metrics src/rinchen/eval src/rinchen/convert \
         src/rinchen/serve src/rinchen/cli
mkdir -p tests scripts plans reports manifests annotation_app
mkdir -p data/raw data/clean data/mined data/synth data/test

touch configs/spm/.gitkeep configs/eole/.gitkeep configs/serve/.gitkeep
touch src/rinchen/__init__.py
touch src/rinchen/data/__init__.py src/rinchen/text/__init__.py src/rinchen/align/__init__.py \
      src/rinchen/filter/__init__.py src/rinchen/synth/__init__.py src/rinchen/metrics/__init__.py \
      src/rinchen/eval/__init__.py src/rinchen/convert/__init__.py src/rinchen/serve/__init__.py \
      src/rinchen/cli/__init__.py
touch tests/__init__.py
touch scripts/.gitkeep plans/.gitkeep reports/.gitkeep manifests/.gitkeep annotation_app/.gitkeep
touch data/raw/.gitkeep data/clean/.gitkeep data/mined/.gitkeep data/synth/.gitkeep data/test/.gitkeep
```

- [ ] **Step 2: Verify the tree and that `rinchen` imports**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import rinchen; import rinchen.data, rinchen.text, rinchen.align, rinchen.filter, rinchen.synth, rinchen.metrics, rinchen.eval, rinchen.convert, rinchen.serve, rinchen.cli; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Add `.gitignore` entries so `data/` content stays untracked but directories are**

(This step's content is written in Task 5 — `data/*/.gitkeep` files from this task are tracked explicitly via `git add -f` in the commit below, before the Task 5 `.gitignore` rule would otherwise hide them.)

- [ ] **Step 4: Commit**

```bash
git add configs src/rinchen tests scripts plans reports manifests annotation_app
git add -f data/raw/.gitkeep data/clean/.gitkeep data/mined/.gitkeep data/synth/.gitkeep data/test/.gitkeep
git commit -m "chore: scaffold repo directory tree and empty rinchen subpackages"
```

---

### Task 3: `CLAUDE.md`

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Consumes: hard constraints verbatim from `docs/research/2026-10-28-python314-stack-and-prompt.md` (`<hard_constraints>` block in Section 3).
- Produces: the file every future session reads first, per the Global Constraints above.

- [ ] **Step 1: Write `CLAUDE.md`**

```markdown
# CLAUDE.md

Read `STATE.md` first for current phase status.

## Hard constraints

ALWAYS:
- Run everything on CPython 3.14.x. Check `python -c "import sys; assert sys.version_info[:2]==(3,14)"` at the start of every script entrypoint.
- Use uv; one workspace; extra envs only for real dependency conflicts, all on 3.14.
- Train every Rinchen model from random init.
- Use position_encoding_type: SinusoidalInterleaved in every Eole model.
- Record every resolved package version and checkpoint hash in manifests/.
- Write tests before code; numeric acceptance thresholds are gates.
- Keep state in files (STATE.md, DECISIONS.md, manifests/), not in chat memory.

NEVER:
- Use Python 3.12/3.13 or any fallback env, container, or conda env with another Python.
- Fine-tune, LoRA, or continue-train any pretrained model. External models only as evaluation baselines or synthetic-data generators whose outputs become training text.
- Use Inner Mongolian data, varieties, datasets, tools, or FLORES+ khk_Mong.
- Use galig/Latin informal Mongolian.
- Write anything about licensing or legal topics.
- Use Relative, rotary, ALiBi, sliding window, or GQA in Eole models meant for CT2.
- Tune on FLORES+ devtest. Use FLORES+ dev for model selection only; report devtest.

## Commands

- `uv run pytest -q`
- `uv run ruff check`
- `uv run mypy src`

## Repo layout

```
CLAUDE.md STATE.md DECISIONS.md pyproject.toml uv.lock
configs/{spm,eole,serve}/
docs/research/  docs/superpowers/{specs,plans}/
src/rinchen/{data,text,align,filter,synth,metrics,eval,convert,serve,cli}/
tests/ scripts/ plans/ reports/ manifests/ annotation_app/
data/{raw,clean,mined,synth,test}/ (gitignored content, tracked dirs)
.github/workflows/ci.yml
```

## Current phase

Phase 0 (environment and benchmark) — not started. See `STATE.md`.

## Evidence and full phase plan

The complete package evidence, version decisions, and all 14 phase specs live in
`docs/research/2026-10-28-python314-stack-and-prompt.md`. This file only carries
the hard constraints and pointers; consult the research doc before pinning or
changing any dependency.
```

- [ ] **Step 2: Verify line count is under 200**

Run: `wc -l CLAUDE.md`
Expected: a number less than 200.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with hard constraints and repo pointers"
```

---

### Task 4: `STATE.md` and `DECISIONS.md`

**Files:**
- Create: `STATE.md`
- Create: `DECISIONS.md`

**Interfaces:**
- Produces: the two files `CLAUDE.md` (Task 3) tells every session to read for current status and prior decisions.

- [ ] **Step 1: Write `STATE.md`**

```markdown
# STATE.md

Phase 0 (environment and benchmark) — not started.

Repo scaffold (this plan) is the only work completed so far: uv workspace,
directory tree, CLAUDE.md, CI stub. No package has been installed or verified
on Python 3.14 yet.
```

- [ ] **Step 2: Write `DECISIONS.md`**

```markdown
# DECISIONS.md

Each entry is tagged `[EVIDENCE: source]` (verified fact) or `[DESIGN: rationale]`
(a choice we made). Full backing detail is in
`docs/research/2026-10-28-python314-stack-and-prompt.md`.

- Pin `eole==0.6.4`, not the GitHub-tagged 0.6.5. [EVIDENCE: 0.6.4 is on PyPI;
  0.6.5 was only tagged on GitHub as of 2026-10-28]. Upgrade only after Phase 0's
  parity smoke test passes on 0.6.5.
- Build our own COMET-22 inference instead of using `unbabel-comet`. [EVIDENCE:
  unbabel-comet 2.2.7 pins `transformers<5`, incompatible with our transformers
  5.21.0 pin].
- Build our own MetricX-24 inference instead of the official `metricx` repo.
  [EVIDENCE: official code pins `transformers==4.30.2`].
- Build Korean `ko-mecab` tokenizer from source (python-mecab-ko sdist) for BLEU;
  fall back to `kiwipiepy`-based `BLEU[ko-kiwi]` labeled non-comparable if the
  build fails. [DESIGN: no cp314 wheel found for python-mecab-ko as of 2026-10-28].
- Use `position_encoding_type: SinusoidalInterleaved` in every Eole model, never
  Relative/rotary/ALiBi. [EVIDENCE: CTranslate2 issue #2134 (open as of
  2026-10-27) shows the Eole→CT2 converter mis-scales relative position keys;
  CT2 also rejects rotary for encoder-decoder].
- Keep vLLM in a separate `gen` dependency group, installed only if its torch
  pin conflicts with Eole's `torch>=2.10,<2.15`. [DESIGN: vLLM 0.32.0 declares
  3.11-3.14 support but its own CI/Docker run on Python 3.12, so treat the
  3.14 install as a runtime risk, not a certainty].
```

- [ ] **Step 3: Verify both files exist and are non-empty**

Run: `wc -l STATE.md DECISIONS.md`
Expected: both files report more than 0 lines.

- [ ] **Step 4: Commit**

```bash
git add STATE.md DECISIONS.md
git commit -m "docs: add STATE.md and DECISIONS.md tracking phase status and pinned decisions"
```

---

### Task 5: `.gitignore` additions for data/runs/manifests artifacts

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: ignore rules that keep `data/` content, `runs/` training output, and generated `manifests/*.parquet` files out of git, while the `.gitkeep` files from Task 2 stay tracked (they were added with `git add -f`, which overrides these new ignore rules for those specific paths).

- [ ] **Step 1: Append to `.gitignore`**

```gitignore

# Rinchen data/training artifacts
data/*
!data/*/.gitkeep
runs/
manifests/*.parquet
!manifests/.gitkeep
```

- [ ] **Step 2: Verify `.gitkeep` files are still tracked and new data files are ignored**

Run: `git check-ignore -v data/raw/.gitkeep; echo "exit=$?"`
Expected: `exit=1` (not ignored, because it's force-tracked) — command prints nothing from `check-ignore` and the echo shows `exit=1`.

Run: `touch data/raw/sample.txt && git status --porcelain data/raw/sample.txt`
Expected: no output (file is ignored, not shown as untracked). Then: `rm data/raw/sample.txt`

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore data/runs/manifest artifacts while keeping tracked placeholders"
```

---

### Task 6: CI stub (`.github/workflows/ci.yml`)

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `pyproject.toml` from Task 1 (runs `uv sync` against it).
- Produces: nothing consumed by other tasks in this plan; this is the terminal task.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-type-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python 3.14
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Sync dependencies
        run: uv sync --dev

      - name: Ruff check
        run: uv run ruff check .

      - name: Mypy
        run: uv run mypy src

      - name: Pytest
        run: uv run pytest -q
```

- [ ] **Step 2: Verify it's valid YAML**

Run: `python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())"`
Expected: no output, exit code 0. (If `yaml` isn't installed locally, run `python -c "import json,sys; sys.path"` is not a substitute — install PyYAML ad hoc with `pip install --user pyyaml` first, or just eyeball the indentation; do not skip this check silently.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint/type/test workflow for Python 3.14"
```

---

## Final verification

- [ ] Run `git log --oneline -6` and confirm six commits exist for Tasks 1-6 in order.
- [ ] Run `git status` and confirm the working tree is clean except for the pre-existing `README.md` modification and `images/` (both out of scope for this plan).
