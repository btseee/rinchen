# Adopt RINCHEN_DEV_AGENT_PROMPT.md as the master plan — design

Date: 2026-10-29

## Context

The project has been following `docs/research/2026-10-28-python314-stack-and-prompt.md`,
a phase-by-phase (Phase 0-14) research doc and agent prompt. The repo scaffold
and Phase 0 / P0.1 (environment probe) were built against it, producing real
evidence: a verified Python-3.14-compatible package matrix, and — critically —
a confirmed, structural finding that `eole` 0.6.4 cannot be imported or run at
all on this machine without `pyonmttok` (which has no cp314 wheel), because
`eole/bin/__init__.py` unconditionally auto-imports modules with unguarded
`import pyonmttok` statements before any CLI argument parsing happens. No
workaround (`uv override-dependencies`, `eole --no-deps`) avoids this; the
only remaining option is building `pyonmttok` from source for cp314 (or an
equivalent eole-side fix), which is a separate follow-up task, not this one.

The user has now provided a new, much larger planning document,
`RINCHEN_DEV_AGENT_PROMPT.md` (v1.0, dated 2026-10-29), and asked to replace
the project's master plan with it. It restructures the work into milestones
D0-D15 with a different (more detailed) repo layout, different tooling
choices (typer, structlog, pydantic-settings, polars/duckdb, `just` instead
of ad hoc scripts, a `rinchen doctor` CLI command), and a "definition of
done" checklist culminating in a CPU-only tiny end-to-end pipeline. The user
confirmed: migrate/adapt existing work into the new structure (don't discard
it), and fix the pyonmttok blocker before attempting D0, since D0's own
doctor checks (`D-EOLE-PY`, `D-EOLE-CLI`) and CI step (`uv sync --frozen
--group dev`) require a working `eole` to pass.

## Scope

In:
- Save the full pasted `RINCHEN_DEV_AGENT_PROMPT.md` content verbatim to
  `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, as the new
  authoritative plan document (mirroring how the original research doc was
  saved on 2026-10-28).
- Mark the old research doc as superseded: add a short note at the top of
  `docs/research/2026-10-28-python314-stack-and-prompt.md` stating it's
  superseded by the new doc as of 2026-10-29, but kept as-is because it's the
  evidence source for decisions already made (the package matrix, the
  pyonmttok finding's origin, and other `DECISIONS.md` entries still cite it).
- Update `CLAUDE.md`'s "Evidence and full phase plan" section to point at the
  new doc as the primary plan, with a note that the old doc remains as
  historical evidence.
- Update `STATE.md` to record: the plan switch, that D0-D15 replaces
  Phase 0-14 going forward, and the current blocking sequence — pyonmttok
  must be resolved (separate upcoming task) before any D0 work starts, since
  D0's own doctor checks require a working `eole`.

Out of scope:
- Any D0 implementation (pyproject.toml restructure to the new dependency
  set, `justfile`, `ci.yml` changes, `src/rinchen/` reorganization to the new
  module layout, `rinchen doctor` command, config loader, manifest
  store/stage runner, etc.) — follows once pyonmttok is resolved.
- Building `pyonmttok` from source for cp314 — a separate, upcoming
  brainstorm/spec/plan cycle, not bundled into this doc-adoption step.
- Reconciling every detail between the old and new doc's package pins (e.g.
  the new doc's `justfile`/`typer`/`structlog`/`pydantic-settings` additions,
  or its slightly different eole version uncertainty) — that reconciliation
  happens when D0 is actually brainstormed and planned, not here.

## Testing

None — this is a pure documentation change (file additions/edits), no code.
Verification is that `CLAUDE.md` still reads coherently and stays under its
stated line budget, and that both research docs exist and are internally
consistent about which one is authoritative.

## Out of scope / explicitly deferred

- D0 implementation.
- Building pyonmttok from source.
