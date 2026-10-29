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
- A static source-code investigation of eole's own wheel (downloaded from
  PyPI, extracted, no installation) to determine where and how `pyonmttok` is
  imported relative to `eole.bin.main`'s import chain and the `eole train`
  CLI's entrypoint — without installing eole's heavy dependency stack.
- If the investigation shows `pyonmttok` is not imported at module level
  anywhere in that chain (only inside specific functions that Rinchen's usage
  wouldn't call): a `pyproject.toml` fix via `uv`'s `override-dependencies` to
  exclude `pyonmttok` from resolution, verified by running the real `uv lock`
  (on the actual project, not a temp copy) and committing the resulting
  `uv.lock`.
- Updating `STATE.md`/`DECISIONS.md` to close out or refine this blocker based
  on the real finding.

Out of scope:
- Running `uv sync` to actually install the full stack, or any live
  import/`--help` test requiring eole's heavy dependencies (torch,
  ctranslate2, tensorboard, datasets, pandas, spacy, etc.) to be installed —
  eole's `requires_dist` shows these are base (non-extra) dependencies, so a
  genuinely live test would cost close to the same multi-GB download as a
  full `uv sync`, which this plan explicitly defers. Static analysis is the
  verification method here; a live check can happen naturally whenever the
  project does its first real `uv sync`.
- Building `pyonmttok` from source for cp314 — only relevant if the
  investigation shows eole genuinely needs it; a separate, bigger follow-up if
  so, not started here.
- Any other Phase 0 task (P0.2-P0.7).

## Investigation procedure

eole 0.6.4's PyPI `requires_dist` (fetched directly, no install) shows its
base dependencies include `torch`, `ctranslate2`, `tensorboard`, `datasets`,
`pandas`, `spacy`, and more — the same heavy stack a real `uv sync` would
pull. `eole.bin.main` almost certainly imports several of these at module
level as the project's CLI coordinator, so a genuinely live
`import eole.bin.main` test would require installing most of that stack
regardless of whether `pyonmttok` itself is fixed — the same multi-GB cost
this plan is trying to defer. Static source analysis avoids that cost and
still answers the real question: is `pyonmttok` imported eagerly (at module
load time, in a module that's always reached) or only lazily (inside a
specific function body, reached only by code paths Rinchen doesn't use)?

1. Download eole 0.6.4's wheel file directly from PyPI (it is a small
   `py3-none-any` pure-Python wheel, confirmed by the P0.1 probe) to a
   scratch directory — no installation, just the file.
2. Extract it (a wheel is a zip file).
3. Grep the extracted source for every occurrence of `pyonmttok` (import
   statements and any other reference).
4. For each occurrence, read the surrounding function/module to classify it:
   module-level import (runs whenever that module is imported) vs.
   import inside a function body (runs only when that function is called).
5. Trace whether any module-level `pyonmttok` import sits in the import chain
   reached by `eole/bin/main.py` at module load time, or by the `eole train`
   CLI command's argument-parsing path (the `--help` path, which typically
   only builds an `argparse` parser, not the full training pipeline).
6. Record the outcome: every file/line where `pyonmttok` appears, whether any
   of them is module-level in a module `eole.bin.main` or `eole train --help`
   would actually load, and therefore whether removing `pyonmttok` is safe
   for Rinchen's planned usage (SentencePiece-based transforms, never
   OpenNMT's own tokenizer transforms).
7. Delete the scratch directory.

## Fix, if the investigation shows pyonmttok is not imported eagerly

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
  evidence (the source-analysis findings — file/line references for every
  `pyonmttok` occurrence and why none of them is reached by Rinchen's usage).
  Note explicitly that this is confirmed by static analysis, not a live
  install/run, and should be re-confirmed the first time `uv sync` actually
  runs.
- Update `DECISIONS.md`: add a bullet documenting the override, tagged
  `[EVIDENCE: ...]`, citing the source-analysis result.

## Fix, if the investigation shows pyonmttok IS imported eagerly

Stop. Do not attempt `override-dependencies` (it would just move the failure
from install-time to a certain runtime `ImportError` the moment that code
path is hit). Update `STATE.md` with the refined finding: the exact
file/line where the eager import happens, in which module, and what
`pyonmttok` functionality eole's code appears to need there. Leave the
blocker open and note that building `pyonmttok` from source for cp314 is the
next option to evaluate, as a separate follow-up — do not start that build in
this plan.

## Testing

This is an investigation plus a small configuration change, not application
code. There is no new unit-testable logic. Verification is the real source
analysis findings (file/line evidence) and the real `uv lock` exit code once
the fix is applied, captured in a report the same way P0.1's findings were
captured and recorded in `STATE.md`/`DECISIONS.md`.

## Out of scope / explicitly deferred

- `uv sync` (full install) on this machine.
- Building `pyonmttok` from source.
- Any other Phase 0 task.
