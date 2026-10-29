# Resolving the pyonmttok / eole gap — design

Date: 2026-10-29

## Context

Phase 0's P0.1 environment probe (`scripts/probe_env.py`) confirmed that `uv lock`
fails on this project's real `pyproject.toml`: `eole==0.6.4` depends on
`pyonmttok>=1.38.1`, and `pyonmttok` has no cp314 wheel (only cp310/cp311/cp312).
This is recorded in `STATE.md` and `DECISIONS.md` as the sole confirmed blocker —
every other direct dependency, including `spacy` (checked fresh today: 3.8.18
now ships cp314 wheels, so it is no longer a concern), resolves cleanly.

The research doc's own prescribed fallback
(`docs/research/2026-10-28-python314-stack-and-prompt.md:184`) is: "install
`eole --no-deps` and add only deps that `python -c \"import eole.bin.main\"` and
`eole train --help` require." This document designs how to carry that fallback
out and, if it works, how to make the real project's `uv lock` succeed.

Rinchen's planned usage of eole (per the research doc's Phase 5/8/etc. example
configs) only uses SentencePiece-based transforms (`sentencepiece`,
`filtertoolong`) — not OpenNMT's own `pyonmttok` tokenizer — so there's reason
to expect eole's core training/conversion path doesn't need `pyonmttok` at
import or runtime, only at install time (as a declared, not actually exercised,
dependency). This document verifies that empirically rather than assuming it.

## Scope

In:
- An isolated scratch-venv investigation reproducing the research doc's exact
  check, to confirm whether `eole.bin.main` and `eole train --help` work
  without `pyonmttok` installed.
- If confirmed: a `pyproject.toml` fix via `uv`'s `override-dependencies` to
  exclude `pyonmttok` from resolution, verified by running the real `uv lock`
  (on the actual project, not a temp copy) and committing the resulting
  `uv.lock`.
- Updating `STATE.md`/`DECISIONS.md` to close out or refine this blocker based
  on the real outcome.

Out of scope:
- Running `uv sync` to actually install the full stack — deferred until the
  project needs real package installs (e.g. starting Phase 0's remaining
  P0.2-P0.7 tasks or later phases).
- Building `pyonmttok` from source for cp314 — only relevant if the
  investigation shows eole genuinely needs it; a separate, bigger follow-up if
  so, not started here.
- Any other Phase 0 task (P0.2-P0.7).

## Investigation procedure

1. Create a throwaway Python 3.14 environment with `uv venv` in a temp
   directory outside the project (so nothing here touches the real project's
   environment or lock file).
2. `pip install eole==0.6.4 --no-deps` into it.
3. Fetch eole's declared dependency list from its PyPI metadata (already
   known to be a `py3-none-any` pure-Python wheel from the P0.1 probe), and
   `pip install` each of those dependencies except `pyonmttok`.
4. Run `python -c "import eole.bin.main"`. If it fails with
   `ModuleNotFoundError` naming some other package (not `pyonmttok`), install
   that package and retry. If it fails specifically because of `pyonmttok`,
   stop — the hypothesis is wrong, record exactly what failed and why.
5. If step 4 succeeds, run `eole train --help` the same way — install
   whatever else surfaces as missing, short of `pyonmttok` itself.
6. Record the outcome: whether `pyonmttok` was ever actually imported,
   and the final minimal dependency set beyond eole's own declared ones that
   turned out to be needed.
7. Delete the scratch venv.

## Fix, if the investigation confirms eole works without pyonmttok

Add to `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["pyonmttok; python_version >= '4.0'"]
```

This is the standard uv idiom for excluding a transitive dependency from
resolution: the marker `python_version >= '4.0'` is always false on any real
Python, so uv treats the `pyonmttok` constraint as satisfied without actually
needing to find or install a matching wheel.

Then:
- Run the real `uv lock` (on the actual project root, not a temp copy this
  time — the goal now is to fix the real lock, not just probe it).
- Confirm it succeeds (exit 0) and a `uv.lock` file is produced.
- Commit `uv.lock` — per the hard constraint "record every resolved package
  version... in manifests/" and standard practice for an application (not a
  library) to commit its lockfile for reproducibility.
- Update `STATE.md`: remove the "uv lock fails" blocker, replace it with a
  short note that the gap is resolved via `override-dependencies`, with the
  evidence (what the scratch-venv investigation showed).
- Update `DECISIONS.md`: add a bullet documenting the override, tagged
  `[EVIDENCE: ...]`, citing the scratch-venv investigation's result.

## Fix, if eole genuinely needs pyonmttok at import or runtime

Stop. Do not attempt `override-dependencies` (it would just move the failure
from install-time to a runtime `ImportError` the moment that code path is
hit). Update `STATE.md` with the refined finding: exactly which import fails,
at what point (`eole.bin.main` import, or only when running `eole train`),
and what `pyonmttok` functionality eole's code appears to need. Leave the
blocker open and note that building `pyonmttok` from source for cp314 is the
next option to evaluate, as a separate follow-up — do not start that build in
this plan.

## Testing

This is an investigation plus a small configuration change, not application
code. There is no new unit-testable logic. Verification is the real commands'
actual exit codes and output (the scratch-venv import/CLI checks, and the
real `uv lock` exit code), captured in a report the same way P0.1's findings
were captured and recorded in `STATE.md`/`DECISIONS.md`.

## Out of scope / explicitly deferred

- `uv sync` (full install) on this machine.
- Building `pyonmttok` from source.
- Any other Phase 0 task.
