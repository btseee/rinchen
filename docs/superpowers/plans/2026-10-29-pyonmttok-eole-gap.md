# Resolving the pyonmttok / eole Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, via static source analysis of eole's own wheel (no heavy installs), whether `pyonmttok` is imported eagerly anywhere `eole.bin.main` or `eole train --help` would reach — and if not, fix the real project's `uv lock` by excluding `pyonmttok` from resolution via `uv`'s `override-dependencies`.

**Architecture:** Download and extract eole 0.6.4's small pure-Python wheel to a scratch directory (no installation). Grep its source for every `pyonmttok` reference, classify each as module-level or function-local, and trace whether any module-level occurrence sits on the import path of `eole/bin/main.py` or the `train` CLI subcommand's argument-parsing setup. Branch on the finding: if safe, apply the `override-dependencies` fix and verify with a real `uv lock`; if not, document the refined blocker and stop.

**Tech Stack:** Python 3.14 (stdlib `zipfile`, `urllib.request`), `grep` / `Grep` tool, `uv`.

## Global Constraints

- No installation of eole or any of its dependencies — static analysis of the extracted wheel only.
- Never suggest or apply a fix that requires running any part of the project on Python 3.12/3.13 or any fallback interpreter.
- Do not run `uv sync` (full install) — only `uv lock` (resolution only).
- Every claim in the final `STATE.md`/`DECISIONS.md` update must cite the real file/line evidence found in this investigation — no guessing.

---

### Task 1: Static source analysis of eole's pyonmttok usage

**Files:**
- Create (scratch, not committed): a temp directory holding the downloaded/extracted wheel — cleaned up at the end of this task, never added to git.
- No permanent repo files are created by this task; its output feeds Task 2's branch decision.

**Interfaces:**
- Produces: a written finding (every `pyonmttok` occurrence's file:line, whether module-level or function-local, and whether it's reachable from `eole.bin.main`'s import chain or the `train` subcommand's `--help` path) that Task 2 consumes to decide which branch to take.

- [ ] **Step 1: Download eole's wheel to a scratch directory**

```bash
SCRATCH_DIR="$(mktemp -d)"
echo "Scratch dir: $SCRATCH_DIR"
curl -sSL "https://files.pythonhosted.org/packages/c1/35/e79589b4466192c5b9713fc98c248915bf357e83b5a31b97395ed3c82433/eole-0.6.4-py3-none-any.whl" -o "$SCRATCH_DIR/eole-0.6.4-py3-none-any.whl"
ls -la "$SCRATCH_DIR"
```

Expected: the `.whl` file downloads successfully (it's a small pure-Python
wheel, a few hundred KB to a few MB, not a heavy binary wheel).

- [ ] **Step 2: Extract the wheel**

A wheel file is a zip archive. Extract it with Python's stdlib (no new
dependency needed):

```bash
python -c "
import zipfile, sys
with zipfile.ZipFile('$SCRATCH_DIR/eole-0.6.4-py3-none-any.whl') as zf:
    zf.extractall('$SCRATCH_DIR/extracted')
print('extracted ok')
"
ls "$SCRATCH_DIR/extracted"
```

Expected: an `eole/` package directory appears under
`$SCRATCH_DIR/extracted`, alongside an `eole-0.6.4.dist-info/` directory.

- [ ] **Step 3: Grep for every `pyonmttok` reference in the extracted source**

```bash
grep -rn "pyonmttok" "$SCRATCH_DIR/extracted/eole" --include="*.py"
```

Expected: a list of `file:line: content` matches (there may be zero, one, or
several). Record this full list verbatim — it is the primary evidence for
this investigation.

- [ ] **Step 4: Classify each match as module-level or function-local**

For each file:line found in Step 3, open that file (read it directly — these
are small Python source files) and determine: is the `import pyonmttok` (or
`from pyonmttok import ...`) statement at the top of the file (module level,
runs whenever the module is imported), or is it indented inside a function
or method body (runs only when that specific function is called)?

Record this classification for each occurrence, e.g.:
"`eole/transforms/tokenize.py:12`: module-level import, inside the
`ONMTTokenizerTransform` class's module" or
"`eole/bin/some_file.py:84`: function-local, inside `def
build_opennmt_tokenizer(...)`, only called when that transform is selected."

- [ ] **Step 5: Trace reachability from `eole.bin.main` and the `train` CLI**

Read `$SCRATCH_DIR/extracted/eole/bin/main.py` directly to see what it
imports at module level and how it dispatches to subcommands (look for
something like an argparse subparser setup or a dictionary mapping
subcommand names like `"train"` to handler functions/modules).

For the `train` subcommand specifically, find the module/function that
implements it (likely `eole/bin/train.py` or similar — check
`$SCRATCH_DIR/extracted/eole/bin/` for the actual filename) and read its
top-level imports.

Cross-reference against Step 4's classification: does any module-level
`pyonmttok` import sit in a module that `eole/bin/main.py` imports directly
or transitively through its own module-level imports, or that the `train`
subcommand's argument-parser-building code (the code that runs for
`eole train --help`, before any actual training starts) would load?

- [ ] **Step 6: Record the finding and clean up**

Write the finding as plain text (you'll use this directly in Task 2) covering:
- The full list of `pyonmttok` occurrences (file:line) from Step 3.
- Each occurrence's classification (module-level vs. function-local) from
  Step 4.
- Whether any module-level occurrence is reachable from `eole.bin.main`
  import time or from `eole train --help`'s argument-parsing path, per Step 5.
- A clear verdict: "SAFE — pyonmttok is not imported eagerly on any path
  Rinchen would exercise (SentencePiece-based transforms only)" or
  "NOT SAFE — pyonmttok is imported eagerly at `<file:line>`, reachable via
  `<the specific path>`."

Then delete the scratch directory:

```bash
rm -rf "$SCRATCH_DIR"
```

No commit for this task — it produced no permanent repo files, only the
written finding that Task 2 uses.

---

### Task 2: Apply the fix or document the refined blocker, based on Task 1's finding

**Files:**
- Modify (if SAFE verdict): `pyproject.toml`, `STATE.md`, `DECISIONS.md`
- Create (if SAFE verdict): `uv.lock`
- Modify (if NOT SAFE verdict): `STATE.md` only

**Interfaces:**
- Consumes: Task 1's written finding (the full occurrence list, classification, and SAFE/NOT SAFE verdict).
- Produces: nothing consumed by other tasks — this is the terminal task.

#### If Task 1's verdict is SAFE

- [ ] **Step 1: Add the override to `pyproject.toml`**

Open `pyproject.toml` and add this table (if a `[tool.uv]` table already
exists from earlier work, add the key to it rather than duplicating the
table header):

```toml
[tool.uv]
override-dependencies = ["pyonmttok; python_version >= '4.0'"]
```

- [ ] **Step 2: Run the real `uv lock`**

```bash
cd "D:\Personal documents\rinchen"
uv lock
echo "exit=$?"
```

Expected: `exit=0`, and a `uv.lock` file is created in the repo root. If it
is NOT 0, read the error output — if `uv` reports a different, unrelated
resolution failure (not `pyonmttok`), that's a new, separate finding: record
it in `STATE.md` as a new blocker, do not try to fix it in this task, and
treat this step as inconclusive for the pyonmttok question specifically
(the override itself may still be correct). If `uv` still fails specifically
on `pyonmttok` even with the override in place, something is wrong with the
override syntax — re-check Step 1's TOML against uv's documented
`override-dependencies` behavior before concluding the approach doesn't
work.

- [ ] **Step 3: Update `STATE.md`**

Read the current `STATE.md`. Replace the "uv lock fails" blocker bullet
(the one with the `pyonmttok`/`eole` error detail) with a resolved note.
Use this as a template, filling in the real file/line evidence from Task 1's
Step 6 finding in place of the bracketed placeholders:

```markdown
- Resolved: the `pyonmttok`/`eole` gap. [EVIDENCE: static source analysis of
  eole 0.6.4's wheel, 2026-10-29 — pyonmttok appears at [FILE:LINE LIST FROM
  TASK 1], all [function-local / classification from Task 1], none reachable
  from `eole.bin.main` import time or `eole train --help`'s argument-parsing
  path]. Fixed via `[tool.uv] override-dependencies` in pyproject.toml,
  excluding pyonmttok from resolution. `uv lock` now succeeds (exit 0);
  `uv.lock` is committed. Not yet live-tested with a real `uv sync` — if
  eole's actual runtime behavior differs from what the static analysis
  suggests, that will surface the first time the full stack is installed.
```

Keep everything else in `STATE.md` unchanged.

- [ ] **Step 4: Update `DECISIONS.md`**

Add a new bullet (do not remove the existing pyonmttok-confirmation bullet —
add this as a new one documenting the fix):

```markdown
- Exclude `pyonmttok` from dependency resolution via `[tool.uv]
  override-dependencies` in pyproject.toml. [EVIDENCE: static source
  analysis of eole 0.6.4's wheel (2026-10-29) shows pyonmttok is not
  imported at module level anywhere reachable from `eole.bin.main` or
  `eole train --help`; see STATE.md for the file/line evidence]. This
  unblocks `uv lock`, which now succeeds. Re-verify the first time `uv sync`
  actually installs eole for real, since this conclusion is from source
  reading, not a live run.
```

- [ ] **Step 5: Commit**

```bash
cd "D:\Personal documents\rinchen"
git add pyproject.toml uv.lock STATE.md DECISIONS.md
git commit -m "fix: exclude pyonmttok from resolution via uv override-dependencies, resolving the eole lock gap"
```

#### If Task 1's verdict is NOT SAFE

- [ ] **Step 1: Update `STATE.md` with the refined blocker**

Read the current `STATE.md`. Replace the existing pyonmttok blocker bullet
with a refined version using this template, filling in the real evidence
from Task 1:

```markdown
- Confirmed (refined): `pyonmttok` is imported eagerly at `[FILE:LINE FROM
  TASK 1]`, inside `[MODULE/FUNCTION NAME]`, which IS reachable from
  `[eole.bin.main import time / eole train --help's argument-parsing path —
  whichever Task 1 found]`. [EVIDENCE: static source analysis of eole
  0.6.4's wheel, 2026-10-29]. This means `uv`'s `override-dependencies`
  trick would only defer the failure to a runtime `ImportError` — not
  applied. `uv lock` still fails on this machine; the project cannot
  install the full pinned stack on Python 3.14 yet. Next option to evaluate
  (not started): building `pyonmttok` from source for cp314, or finding an
  eole version/fork that makes this import optional.
```

Keep everything else in `STATE.md` unchanged. Do not modify
`pyproject.toml` or `DECISIONS.md`'s existing entries — this is a refinement
of an existing confirmed blocker, not a new decision.

- [ ] **Step 2: Commit**

```bash
cd "D:\Personal documents\rinchen"
git add STATE.md
git commit -m "docs: refine pyonmttok/eole blocker with eager-import evidence from source analysis"
```

---

## Final verification

- [ ] Run `git log --oneline -2` and confirm the Task 2 commit (either branch) landed.
- [ ] Run `cat STATE.md` (or open it) and confirm the pyonmttok section reads as accurate, evidence-backed prose a future session could act on without re-deriving the investigation.
- [ ] If the SAFE branch was taken: run `uv lock` once more from the real repo root and confirm it still exits 0, confirming the committed `uv.lock` reflects a real, reproducible resolution.
