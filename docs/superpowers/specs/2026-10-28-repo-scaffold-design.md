# Repo scaffold + CLAUDE.md — design

Date: 2026-10-28

## Context

Rinchen repo currently holds only README.md, LICENSE, images/logo.png. A research
doc (`docs/research/2026-10-28-python314-stack-and-prompt.md`) lays out a 14-phase
plan to build a from-scratch, Mongolian-centric MT system on CPython 3.14, including
a full agent-driving prompt (`RINCHEN_SONNET_PROMPT.md`) with hard constraints,
stack pins, and phase-by-phase specs.

That plan spans many independent subsystems (env/stack, data acquisition, tokenizer,
filtering, custom metric inference, training, distillation, serving). It is too large
for one spec. This document covers only the first slice: the repo scaffold and
CLAUDE.md that every later phase will build on. Phase 0 (environment probe/pinning)
and all later phases get their own spec + plan cycle later.

## Scope

In:
- uv workspace (`pyproject.toml`) with dependencies pinned to the versions the
  research doc marks "use", plus a dev group and an optional `gen` group for vLLM.
- Full directory tree from the doc's `repo_layout` block, as empty placeholders.
- `CLAUDE.md`, `STATE.md`, `DECISIONS.md` at repo root.
- `docs/research/2026-10-28-python314-stack-and-prompt.md` — the full pasted
  research content, saved verbatim as the evidence source.
- `.github/workflows/ci.yml` stub (lint/type/test, no GPU jobs).
- `.gitignore` additions for `data/`, `runs/`, `manifests/*.parquet` artifacts.

Out of scope (future slices):
- Phase 0 probe script (`P0.1`) and any actual `uv sync` / package install.
- Any code under `src/rinchen/*` beyond empty `__init__.py`.
- Data acquisition, tokenizer training, metrics, model training, serving code.

## Repo layout

```
CLAUDE.md STATE.md DECISIONS.md pyproject.toml uv.lock
configs/{spm,eole,serve}/
docs/research/2026-10-28-python314-stack-and-prompt.md
docs/superpowers/specs/2026-10-28-repo-scaffold-design.md
src/rinchen/{data,text,align,filter,synth,metrics,eval,convert,serve,cli}/__init__.py
tests/__init__.py
scripts/ plans/ reports/ manifests/ annotation_app/
data/{raw,clean,mined,synth,test}/  (gitignored)
.github/workflows/ci.yml
```

Empty directories get a `.gitkeep` so git tracks them.

## pyproject.toml

- `requires-python = "==3.14.*"`.
- Main dependency list, pinned per the research doc's package matrix ("use" rows):
  torch==2.14.1 (cu130 index), eole==0.6.4, ctranslate2==4.8.4, transformers==5.21.0,
  tokenizers==0.23.3, sentencepiece==0.2.1, sacrebleu==2.6.0, kiwipiepy==0.24.0,
  fasttext-new==0.1.0, datasketch==2.0.0, datasets, huggingface_hub, pyarrow, polars,
  duckdb, numpy, pandas, scipy, trafilatura, warcio, httpx, selectolax,
  beautifulsoup4, sentence-transformers, fastapi, uvicorn, pydantic.
- `[dependency-groups] dev`: pytest, ruff, mypy.
- `[dependency-groups] gen` (optional, separate env): vllm==0.32.0 — only installed
  if Phase 0 finds a torch pin conflict with Eole's `torch>=2.10,<2.15`.
- No version resolution is run as part of this slice; pins are copied from the
  doc's evidence, to be verified by the (future) Phase 0 probe script.

## CLAUDE.md

Under 200 lines. Contents:
- Hard constraints block, copied verbatim from the research doc's `<hard_constraints>`.
- Commands: `uv run pytest -q`, `uv run ruff check`, `uv run mypy src`.
- Repo layout (short form of the tree above).
- "Current phase: 0 (not started) — read STATE.md first."
- Pointer to `docs/research/2026-10-28-python314-stack-and-prompt.md` for full
  evidence and the phase-by-phase plan, so CLAUDE.md itself stays short.

## STATE.md / DECISIONS.md

- `STATE.md`: one line, "Phase 0 (environment and benchmark) not started."
- `DECISIONS.md`: seeded with the doc's already-made calls (pin Eole 0.6.4 not
  0.6.5; build COMET-22 and MetricX-24 inference ourselves; build Korean mecab-ko
  from source with Kiwi as labeled fallback; keep `SinusoidalInterleaved` for every
  Eole model), each tagged `[EVIDENCE: ...]` or `[DESIGN: ...]` as in the doc.

## CI

`.github/workflows/ci.yml`: `actions/setup-python` with `python-version: "3.14"`,
`uv sync`, `ruff check`, `mypy src`, `pytest -q`. No GPU runners in this slice.

## Testing

No automated tests for a pure scaffold. Verification is manual: confirm the tree
exists, `pyproject.toml` parses, CLAUDE.md is under 200 lines and matches the hard
constraints verbatim, and CI workflow is valid YAML.

## Out of scope / explicitly deferred

- Running `uv sync` or validating any package actually installs on Python 3.14 —
  that is Phase 0's job (task P0.1 in the research doc), not this scaffold.
- Writing the probe script, FLORES+ loader, or any metric/model code.
