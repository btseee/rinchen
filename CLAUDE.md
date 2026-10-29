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
vendor/wheels/ (committed binary wheels with no cp314 upstream release)
vendor/fasttext-wheel-shim/ (pure-Python shim package, see DECISIONS.md)
.github/workflows/ci.yml
```

## Current phase

Master plan switched 2026-10-29 to milestones D0-D15 (see below). Both the
pyonmttok and fasttext-wheel cp314 blockers are resolved: `uv sync --dev`
succeeds, and CI's full step sequence (ruff, mypy, pytest) passes locally on
Windows (not yet observed on CI's Linux runner). Next up: D0, which has not
started. See `STATE.md`.

## Evidence and full plan

The current master plan — milestones D0-D15, repo layout, hard constraints,
build-ourselves specs — lives in
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`. The prior Phase 0-14
plan, `docs/research/2026-10-28-python314-stack-and-prompt.md`, is superseded
but kept as the evidence source for decisions already recorded in
`DECISIONS.md` (the package wheel matrix, the pyonmttok/eole finding's
origin). This file only carries the hard constraints and pointers; consult
the master plan before pinning or changing any dependency.
