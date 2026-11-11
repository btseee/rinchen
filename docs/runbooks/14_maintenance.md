# 14 - Maintenance

## Scope

D14 says whether the stack the project stands on is still the stack its lock file names, what `uv` could take next and which of that a recorded decision says not to take blindly, and whether the numbers the project pinned with golden tests and parity
checks still come out the same. It builds: the command `rinchen maintain check` and its library (`rinchen.maintain`: the lock reader, the environment check, the parser of a dry run and the drift table, the command runner, the parity reruns, the report and its
Markdown, the order of the steps), the closed configuration `configs/maintain/maintain.yaml` (the uv argument lists and their bounds, the pin rules, the reruns with their cadences and bounds; `rinchen.config.maintain`), the report `maintain_report.json` with
its Markdown summary and a manifest that records every resolved package version and the lock file's SHA-256, the workflow `.github/workflows/nightly.yml`, and this runbook (the upgrade procedure, the rollback, what each failure means). The design, with the
rulings M1 to M19 and a section of the amendments made while building (A1 to A15), is `docs/superpowers/specs/2026-11-11-d14-maintenance-design.md`; the decisions are condensed in the entry "2026-11-11: D14 maintenance loop: the rulings and the limits" of
`DECISIONS.md`; the upstream facts are in `docs/VERIFY.md` (section "D14 findings"). The facts about `uv`, `uv.lock`, the doctor, pytest, the parity commands, the costs and the workflow draft come from a spike made before the design
(`.superpowers/sdd/2026-11-11-d14-maintenance/spike-findings.md`, findings F1 to F18; uv 0.13.7, Eole 0.6.4, CTranslate2 4.8.4, sacrebleu 2.6.0, numpy 2.5.4, CPU, offline).

**What D14 is.** A read-only check and a schedule for it. One command answers three questions with states that say what was and was not looked at: is this machine the stack the lock names and does the toolchain start (the CPU doctor checks, `uv lock --check`,
the installed packages against the lock); what could `uv` take next within the constraints of `pyproject.toml`, and which of that does a recorded decision hold back (the dependency drift table); do the pinned numbers still come out the same (the parity reruns,
the cheap ones every night, the heavy ones every week). It reports; it never changes anything. An upgrade is an operator's procedure on a branch (below), and every number and rule of the shipped configuration is an unvalidated prior.

**What D14 is NOT.** Nothing below is a statement about the quality of any model, and nothing in it has run on GitHub.

- **No upgrade.** Nothing in D14 writes `uv.lock` or `pyproject.toml`, runs `uv lock --upgrade` without `--dry-run`, `uv add`, `uv sync` or `pip`, or makes a branch, a commit, a push or a pull request. The loader refuses a configuration whose drift arguments lack `--dry-run`
  or hold a flag outside `--upgrade`, `--dry-run`, `--offline`, `--check` (and the lock check's arguments hold `--check` and no flag but `--offline`); static scans and the gate hold the rest (see "The STOP gate").
- **No view of what exists on PyPI.** The drift table is what `uv lock --upgrade --dry-run` would take within the constraints of `pyproject.toml`. 12 of the 38 direct dependencies are pinned with `==` (`torch`, `eole`, `pyonmttok`, `ctranslate2`, `transformers`,
  `tokenizers`, `sentencepiece`, `sacrebleu`, `kiwipiepy`, `fasttext-new`, `fasttext-wheel`, `datasketch`; **measured** here by a parse of `pyproject.toml`, the same count as the spike's F13) and never appear in it. Whether they have a newer release is a hand check (below).
- **No GPU.** The doctor's GPU check is never called, no rerun touches a device and `auto` is not a value of anything here. The machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run; every process of the gate ran with
  `CUDA_VISIBLE_DEVICES=""` on the CPU (one mutation run of Task 7 started the command for a few seconds with `CUDA_VISIBLE_DEVICES=0`: no compute, no manifest, and the doctor's CPU checks ask no device question; see "What was not verified"). Everything GPU-side is **UNMEASURED**.
- **No network in any test, measurement or gate run, with one disclosed exception.** The one network step of a night is uv's own dry run and it is `uv`'s; the gate answers it from a HAND-WRITTEN sample through a shim and never forwards it. One mutation run of Task 7 ran the
  real online dry run once in a copy of the tree by a slip (it writes nothing); what the agent reported of that output is in "The drift table" as second-hand and is not a fixture.
- **No GitHub Actions.** The workflow was checked statically and its own command ran natively; it never ran on a runner. "Nightly workflow green on tiny" is met only in that sense (see "The STOP gate").
- **No real model, corpus or export.** The reruns run on the toy, which saturates and ranks nothing; the `parity_export` rerun ran only on the D11 toy export; no robustness or evaluation rerun over a real export exists because D12's command has no floors.
- **No automation of `docs/STACK.md`**, no `/feedback` endpoint, no retraining loop: the older Phase 14 text of the superseded plan is not the master plan's D14. The dependency report lists names, versions and the lock's hashes and nothing else about a package.

Tags used below: **measured** (run for this runbook on 2026-11-11 with the code at `51c4489`: `rinchen maintain check --help` and `rinchen --help` through the real binary, and read-only in-process calls of pure functions (the dry-run parser, `result_lines` over the golden
run built with fakes, the timeout arithmetic of the shipped files, a parse of `pyproject.toml`); **no** `maintain check` run, no uv process and no network were started for this runbook), **gate** (printed in the docstrings of the D14 gate, run by Task 7 on 2026-11-11 and recorded in
the ledger), **spike** (run in the design spike), **source-read** (read from the code, never run), **ARITHMETIC** (a formula applied to values that were measured), **second-hand** (reported by an agent, the log no longer exists), **UNMEASURED** (nothing was run). A TIME moves by a third from one
run to the next on this shared machine; a count does not. A number that was not obtained is not given.

What was checked: 19,818 tests passed after Task 7 (1 skipped, 10 deselected; 19,819 selected collected) against 17,852 at the end of D13 (1,966 new, ARITHMETIC: 29 new unit test files, 4 property, the two gate modules and 12 support modules; 22,065 added lines over 76 files, of which 3,308 are source (counting the 2 lines of `cli/main.py`), 18,239 tests and fixtures and 518 spec, plan, configuration and workflow,
`git diff --numstat 0f3f6ba..51c4489`), each new test mutation-checked by hand during the build (70, 75, 116, 160, 73 and 134 mutants for Tasks 1 to 6, 46 for Task 7 and one for the controller's fix of the workflow: 675 in all, ARITHMETIC; one equivalent survivor, in Task 3), differential
checks against independent references (the lock reader against a regular-expression reader on the real `uv.lock` and on 300 generated locks; the environment check against brute force on 500 random pairs; the dry-run parser against a split-based reader on 20,000 generated and 20,000 corrupted outputs; the JUnit reader against `ElementTree` on 100,000 documents; the
runner against `subprocess.run` on 200 random children; the rerun decision against a model on 60,000 points and 300 real pytest projects; the report's states on 5,000 vectors and its Markdown against an independent renderer on 3,000 random reports; the whole command against an oracle on 1,000 scenarios; the version parser on 40,000 strings; the uv shim against an oracle on 238,741
argument lists; every one with 0 mismatches), and the real-binary gate (two modules, 197 tests, plus the 43 tests of the shim). **The combined review of the code has not run yet**: this runbook is written before it, and a follow-up commit will record its verdicts (see "The review").

## Architecture

```
pyproject.toml, uv.lock, configs/maintain/maintain.yaml --hashed first, parsed from the bytes hashed--> Inputs
doctor: the six CPU checks, in process ------------------------------------------------------------+
lock_file: uv.lock read with tomllib --------------------------------------------------------------+
lock_consistent: `uv lock --check` (runner) -------------------------------------------------------+--> run_check --> Report --> maintain_report.json, maintain_report.md
environment: importlib.metadata against the lock --------------------------------------------------+                      --> logs/, junit/, parity_export/ (local)
drift: `uv lock --upgrade --dry-run` (runner) -> parse -> cross-check against the lock -> rules ----+                      --> manifest.json (written LAST)
reruns: pytest (exit code AND JUnit counters) or `rinchen export parity` (runner) -----------------+
```

| Module | Job | Lines |
|---|---|---:|
| `rinchen.config.maintain` (+ `configs/maintain/maintain.yaml`) | the closed configuration: skipped doctor checks, the uv argument lists and bounds, the pin rules, the reruns; `reruns_due`, `rule_for` | 292 (140) |
| `rinchen.maintain.states` | the four states and the closed lists of skip and fail reasons | 129 |
| `rinchen.maintain.lock`, `environment` | `normalize_name`, `LockFile`, `read_lock`; `installed_versions`, `check_environment` | 104 + 95 |
| `rinchen.maintain.uv_output`, `drift` | the parser of a dry run (`parse_dry_run`); `drift_outcome`: the order of the decision, the cross-check, the classification | 121 + 136 |
| `rinchen.maintain.commands`, `reruns` | `CommandRunner`, `SubprocessRunner`, `child_environment`; `read_junit`, the argument lists, `run_pytest_rerun`, `run_parity_rerun` | 195 + 387 |
| `rinchen.maintain.report`, `report_markdown` | `Report`, `Step`, the aggregation, the bytes, the exit code; the Markdown | 159 + 148 |
| `rinchen.maintain.check`, `check_types`, `check_steps`, `check_reruns` | `run_check`: the order of the steps with every seam injected; the data it takes and gives; the four first steps; the reruns step | 162 + 99 + 245 + 114 |
| `rinchen.cli.maintain`, `maintain_inputs`, `maintain_doctor`, `maintain_output`, `maintain_signals` | the command; the hashing and the options that must agree; the six CPU checks; the documents, the manifest and the printed lines; SIGTERM and SIGHUP | 357 + 132 + 71 + 307 + 53 |
| `.github/workflows/nightly.yml` | the schedule | 52 |
| tests only: `tests/support/{maintain_*,uv_outputs,fake_uv,nightly_workflow,workflow_expression}.py`, `tests/fixtures/maintain/{golden_report.json,golden_report.md,uv.lock.sample}` | fakes, the fixture project, the recorded and hand-written uv outputs, the shim of the gate, the goldens (6,132 + 1,806 bytes) | |

13 library modules (and an empty `__init__.py`) with 2,094 lines, 5 command modules with 920 and the configuration module with 292 (`wc -l`, ARITHMETIC); no source file is over 450 lines. `rinchen.maintain` and `rinchen.config.maintain` import only the standard library, pydantic, PyYAML, `rinchen.errors` and each other (an
AST scan, `tests/unit/test_maintain_layering.py`); the command modules import `rinchen.cli.doctor`, `cli.errors`, `cli.train_common`, `export.verified`, `manifest` and `rinchen.maintain` and none of `train`, `serve`, `eval`, `synth`, `infer`, `annotate`, `filter`, `ctranslate2`, `eole`, `torch`,
`sentencepiece`, `polars` or `huggingface_hub`; nothing but `cli/main.py` imports `cli.maintain`. The only edits to existing files are `src/rinchen/cli/main.py` (+2 lines: the registration of the `maintain` group) and `tests/unit/test_cli_help.py` (+1: `"maintain"` in `SUBCOMMANDS`); `pyproject.toml` and `uv.lock` are untouched
(`git diff --quiet 0f3f6ba HEAD -- pyproject.toml uv.lock`, **measured** here; `uv.lock` SHA-256 `559b3954...b16`).

## What the check looks at

The command runs six steps, each once, in this order, and a step never disappears from the report: what was not looked at is `skipped` with a reason, what could not be looked at is `fail` with a reason.

| Step | What it does | Cost | Can it be `skipped`? |
|---|---|---|---|
| `doctor` | the six CPU checks of `rinchen doctor`, called in process (`D-PY`, `D-WHEELS`, `D-EOLE-PY`, `D-EOLE-CLI`, `D-CT2-CONV`, `D-CT2-IMPORT`); `D-TORCH-CUDA` is listed as `skipped` and never called | about 13 s (**gate**), 15.6 s (**spike**) | per check |
| `lock_file` | `uv.lock` hashed, then read with `tomllib` from the bytes hashed; format version 1 only; every entry needs a string name and version | 0.024 s (**spike**) | no |
| `lock_consistent` | `uv lock --check`: does the lock still satisfy `pyproject.toml` (`ci.yml` syncs with `--frozen`, which never checks) | 0.075 s offline (**spike**) | no |
| `environment` | the installed distributions (`importlib.metadata`, names in PEP 503 form) against the names and versions of the lock | 0.058 s (**spike**) | no |
| `drift` | `uv lock --upgrade --dry-run`, parsed, cross-checked against the lock and classified by the pin rules | the dry run: UNMEASURED online | `disabled by --no-drift` |
| `reruns` | the parity reruns that are due (or the ones named with `--rerun`) | 136 s nightly (**gate**), about 1,220 s weekly (ARITHMETIC: 136.2 + 1,082) | per rerun |

**States.** `ok`; `drift` (only the drift step, when its table is not empty); `fail`; `skipped`. A step that holds several items (the doctor's checks, the reruns) is `fail` if any item failed, else `ok` if any is `ok`, else `skipped`. The whole run is `fail` if any step failed, else `drift`
if the drift step is `drift`, else `ok`. The steps are independent and all attempted, except that the reruns are `skipped (a doctor or environment check failed)` when the doctor or the environment step failed: their numbers would be about another stack and they cost minutes. `lock_file` is `ok` in
every report that exists: a lock that cannot be read ends the command with exit 3 before a report is made. **Reasons** are a closed list in `rinchen.maintain.states`, written once and pinned by a test, so a report never holds a line of another program's output, an exception text or a path: the
skips `a GPU check: maintain check never asks a device question`, `disabled by --no-drift`, `disabled by --no-reruns`, `no export was given`, `a doctor or environment check failed`, `reported by the doctor itself` and `not due at the <cadence> cadence`; the failures `uv was not found`, `uv exited with code <n>`,
`timed out after <s> s`, `the output of uv has <n> lines this version does not recognise`, `uv printed no recognisable result`, `uv printed a no-change line next to change lines`, `uv reported a locked version that differs from uv.lock for <n> packages`, `<m> packages differ from the lock, <e> are not in it`, `the check failed`
(a doctor check), `pytest exited with code <n>`, `<f> failed and <e> errors`, `<p> passed, fewer than the floor <m>`, `<k> skipped, more than the allowed <m>`, `no JUnit report was written`, `a target does not exist`, `the command exited with code <n>` and `the command wrote no readable parity.json`.

## The command

```
uv run --frozen --no-sync rinchen maintain check --out-dir DIR [--cadence nightly|weekly] [--config FILE] [--project-dir DIR] [--uv FILE]
    [--no-drift] [--no-reruns] [--rerun ID ...] [--fail-on-drift] [--export DIR --checkpoint DIR --corpus FILE [--corpus FILE ...]]
```

`--frozen --no-sync` is the form the workflow uses: a plain `uv run` locks and syncs the project first, which would repair a `pyproject.toml` that moved without `uv.lock` before the `lock_consistent` step could report it (amendment A1; commit `8d701b0`: 11 workflow tests fail when the workflow is reverted to a plain `uv run`, **measured** by the controller).
The real option names are those of `rinchen maintain check --help` (**measured** here, exit 0):

| Option | Meaning |
|---|---|
| `--out-dir DIR` (required) | receives the report, its Markdown, `logs/`, `junit/` and, last, `manifest.json`; it may not be the directory of an input (the project's directory, the configuration's) and may not hold another stage's manifest |
| `--cadence nightly\|weekly` (default `nightly`) | `nightly` selects the reruns whose cadence is `nightly`; `weekly` selects every rerun |
| `--config FILE` | the pin rules, the uv argument lists and the reruns (default `configs/maintain/maintain.yaml`) |
| `--project-dir DIR` | holds `pyproject.toml`, `uv.lock` and the tests the reruns select (default: the repository root) |
| `--uv FILE` | the uv program (default: the one on `PATH`; the variable `UV` is never read: `uv run` sets it to the uv that started it, **spike** F17, and the gate relies on a shim first on `PATH`) |
| `--no-drift` | skip the drift step (it is the only step that queries the index) |
| `--no-reruns` | run none of the reruns; it wins over `--rerun` and `--cadence` |
| `--rerun ID` (repeatable) | run exactly these reruns, whatever the cadence; an id the configuration does not hold is exit 2 |
| `--fail-on-drift` | exit 4 when the drift table is not empty (the workflow does not use it) |
| `--export DIR`, `--checkpoint DIR`, `--corpus FILE` (repeatable) | the inputs of the `parity_export` rerun; all three or none (exit 2) |

Order of the work: the options are checked; `pyproject.toml`, `uv.lock` and the configuration are hashed BEFORE any is read; an output directory that is an input's, and the manifest of another stage in it, are refused; the three files are parsed from the bytes that were hashed (a file that changed in between is
refused); only then is anything written: the output directory is made, the old manifest and the files it listed are removed, the check runs, the messages of the doctor checks that failed are printed to standard output as `[FAIL] <id>: <message>` (they hold an exception text and can hold a path, and they reach no file),
the version of uv is read for the manifest, and the report, the Markdown and the manifest are written. Nothing the command writes goes outside `--out-dir`, and `--out-dir` should be a directory outside the working tree or one that git ignores: the workflow's `maintain-out` is not in `.gitignore`.

**Worked examples** (the commands are an operator's: none was run for this runbook).

1. The preflight, with nothing a network or a model needs (the doctor, the lock check and the environment check, about 13 s, **gate**; this is what D15 may call):
   `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --no-drift --no-reruns`
2. The nightly set, as the workflow runs it (this one queries the index through uv's dry run): `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --cadence nightly`
3. The weekly set, by hand or after an upgrade without the drift step: `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --cadence weekly --no-drift`
4. One rerun, for example the one a pin rule requires: `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --no-drift --rerun parity_tiny`
5. The parity of a real export, recorded by name (the toy export of D11 is the only one that ever ran, **gate** R6): `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --no-drift --rerun parity_export --export EXPORT_DIR --checkpoint CHECKPOINT_DIR --corpus CORPUS.parquet`.
   It runs `python -m rinchen export parity ... --max-lines 50 --threads 2 --timeout <bound - 30>`; the other reruns are skipped because the selection wins over the cadence.

The lines the command prints, here for the golden run (the nightly run of the shipped configuration over the sample lock of the tests, with the HAND-WRITTEN dry run; the lock figures are the sample's six entries, not the real lock's 187; **measured** here by calling `result_lines` in process over the golden run, which uses fakes):

```
maintain: drift (nightly)
doctor: ok (6 ok, 0 fail, 1 skipped)
lock: ok (6 entries, 6 names)
environment: ok (6 installed, 0 differ, 0 extra, 0 not installed)
drift: drift (7 entries: 5 update, 1 add, 1 remove; 4 blocked)
rerun numerics: ok (1110 passed)
rerun parity_tiny: ok (13 passed)
rerun parity_export: skipped (no export was given)
rerun robustness_tiny: skipped (not due at the nightly cadence)
rerun evaluation_tiny: skipped (not due at the nightly cadence)
rerun serving_tiny: skipped (not due at the nightly cadence)
report: OUT/maintain_report.json
summary: OUT/maintain_report.md
manifest: OUT/manifest.json
```

Every line before the last three holds states, counts and closed reasons; the last three hold paths. Through the shim of the gate the real lock gave 187 entries and 186 names, 186 installed, none differing and none extra (**gate**, R1).

### Exit codes

| Code | Meaning | Where measured |
|---:|---|---|
| 0 | the run is `ok` or `drift` (drift is information: with 186 locked packages something is almost always newer, and a job that is red every week is a job nobody reads) | **gate** R1, R5, R6, R8 |
| 4 | the run is `fail` (a measured result: everything is written and printed), or it is `drift` and `--fail-on-drift` was given | **gate** R6 (a flipped byte, another model's checkpoint), R7 (the reruns), R8 (bad uv outputs, `--fail-on-drift`) |
| 2 | a bad option or configuration (a `--cadence` or `--rerun` the configuration does not know, `--export`, `--checkpoint` and `--corpus` not given together, a drift argument list without `--dry-run`), a file that changed after it was hashed, an output directory that is the project's or an input's | **gate** R8 (the configuration), R9 (`--export` alone, unknown `--rerun`, the project's directory as output); fakes for the rest |
| 3 | `pyproject.toml` or `uv.lock` missing or invalid (TOML, a lock format version other than 1, an entry without a name or version), another stage's manifest in the output directory | **gate** R9 (a missing `uv.lock`, a foreign manifest: nothing written, the old files untouched) |
| 1 | the output cannot be written | fakes only: UNMEASURED with a real full disk |
| 130, 143, 129 | interrupted by SIGINT, SIGTERM or SIGHUP; the commands the check started are stopped through their process group; no manifest is written and the old one is already gone | **gate** R9 (130 and 143 on the real binary, 13.4 s); 129 in the unit tests of `maintain_signals` with real signals |

## Reading the files

**`maintain_report.json`** (schema `rinchen.maintain/1`; UTF-8, sorted keys, two-space indent, one final LF; no timestamp, no absolute path, no environment value, no host name and no line of any program's output: the same observations give the same bytes). The top level holds `schema`, `cadence`,
`state` (the whole run), `inputs` (SHA-256 of `config`, `lock` and `pyproject`), `lock` (`packages`: the entries, `names`: the distinct names, `format`: `version` and `revision`) and `steps`, the six of the table above in order, each `{id, state, reason, detail}`. The shapes of `detail`: `doctor` `{checks: [{id, state, reason}]}`;
`environment` `{installed, mismatched: [{name, installed, locked}], mismatched_more, extra, extra_more, not_installed}` (at most 20 names of each kind and a count of the rest); `drift` `{entries: [{package, change, locked, available, class, rule, requires}], counts: {update, add, remove, drift, blocked}, warnings}` (`{}` when the step failed or was skipped);
`reruns` `{items: [{id, kind, cadence, state, reason, tests: {passed, failed, errors, skipped} or null, figures: {exact, sentences, mean_abs_delta} or null}]}`; `lock_file` and `lock_consistent` `{}`. Every configured rerun is listed, due or not. The golden example is `tests/fixtures/maintain/golden_report.json` (6,132 bytes, SHA-256 `ac7f08b1...eeec`);
the real lock's 187 entries, 186 names and 186 installed make the report of the gate 6,138 bytes (**gate**; 6,132 + 6 = 6,138, ARITHMETIC: the three counts of 6 become three of 3 digits). A reading, drift step of the golden run:

```
{"package": "numpy", "change": "update", "locked": "2.5.4", "available": "2.5.5", "class": "blocked", "rule": "sacrebleu-numpy", "requires": ["numerics", "evaluation_tiny"]}
{"package": "rich", "change": "update", "locked": "15.0.0", "available": "15.0.1", "class": "drift", "rule": null, "requires": []}
```

`locked` is always the lock file's version (several, joined by `, `, when the lock holds more than one; null for an add), never uv's; `available` is uv's new version (null for a remove). `class` is `drift` (nothing says not to take it: take it through the procedure) or `blocked` (a pin rule names the package: run its `requires` first).

**`maintain_report.md`** (golden `tests/fixtures/maintain/golden_report.md`, 1,806 bytes, SHA-256 `7e7774f8...d07a`; line feeds only, rendered from the report alone, cells escaped): the heading `# Maintenance check: <state> (<cadence> cadence)`, a table of checks (one row per doctor check and one per other step), the drift section (the state and reason, a count line, the table
of at most 200 rows with `needs` for the rules' `requires`, a line saying how many more are in the JSON, and the fixed sentence of the blind spot), the table of reruns (passed, failed, skipped, reason) and, for each failing STEP, a line `- What to do (<step id>): docs/runbooks/14_maintenance.md#failure-<step id>`. The workflow appends this file to the run page.

**`manifest.json`** (stage `maintain_check`, written last): `inputs` (`pyproject.toml`, `uv.lock` and the configuration, each with the SHA-256 taken before it was read), `outputs` (the two documents, the logs the run wrote, the JUnit report of each pytest rerun that ran and `parity_export/parity.json` when the figures were read, each with its digest), `config_sha256`, `seed` -1,
`tool_versions` (EVERY name of the lock with its locked version, a name with several versions joined by `, ` in sorted order, plus `python` and `uv`, or `python (tool)` and `uv (tool)` if the lock holds a package of that name; these are the RESOLVED versions, equal to the installed ones when `environment` is `ok`), `metrics` (`packages`, `entries`, `blocked`, `reruns_run`,
`seconds.<step>`) and `status` (each step's state and `state` for the whole run). The gate measured 188 `tool_versions` keys (186 names, `python` 3.14.10, `uv` 0.13.7) and 8 outputs (the two documents, four logs, two JUnit reports; **gate** R2). The manifest holds the machine's absolute paths and the time of creation: it is a local record.
**`logs/`** (`lock_consistent.log`, `drift.log`, `<rerun id>.log`), **`junit/`** and **`parity_export/`** are local files that can hold paths and uv's or pytest's own words; they are never uploaded and the report quotes none of them. `drift.log` is where the real format of a dry run is learned.

## The drift table and the pin rules

`drift_outcome` decides in a fixed order and the first thing that applies ends it: the run timed out; uv did not run to an exit (not found); its exit code is not 0; then its text, read by `parse_dry_run`, which sorts EVERY non-blank line (after removing ANSI sequences and a trailing CR) into exactly one class: a boilerplate line
(`Using CPython <x> interpreter at: <path>`, `Resolved <n> package(s) in <one token>`), a `warning:` line (counted), the positive line `No lockfile changes detected`, a change line (`Update`, `Updated`, `Add`, `Added`, `Remove`, `Removed`, a name, a version with an optional `v`, and for an update `->` and the new version) or UNRECOGNISED. An unrecognised line is a
failure, the positive line next to change lines is a failure, no result at all is a failure; then the cross-check against the lock: an update's `old` and a remove's version must be versions the lock lists for that name, an add must name a package the lock does not hold, and a `(package, change)` listed twice breaks it. Only then: the positive line alone is `ok`, changes are `drift`. A mis-parse, or a format that changed, cannot get past
the cross-check: it needs the lock's own table. The first rule that names a package (a normalised name or a prefix ending in `*`: `torch-*` covers `torch-cuda` and not `torchvision`) labels its row `blocked`. `blocked` is information and never a failure: the dry run proposing a newer numpy is normal and the row says what to run first. The shipped rules are priors written from the recorded decisions:

| Rule | Packages | Requires (`gate` is `just gate`) | Decision it cites |
|---|---|---|---|
| `eole` | `eole` | `parity_tiny parity_export gate` | the `eole==0.6.4` pin; `D-EOLE-PY`, `D-EOLE-CLI`, `D-STRICT-PARITY` |
| `ctranslate2` | `ctranslate2` | `parity_tiny parity_export serving_tiny` | `D-CT2-CONV`, `D-CT2-IMPORT`, `D-CT2-EOLE-CUDA`, `D-CT2-QUANT-CPU`, `D-CT2-SUPPRESS`; SinusoidalInterleaved only |
| `torch-family` | `torch`, `torch-*`, `torchcodec`, `triton`, `nvidia-*`, `cuda-*` | `parity_tiny robustness_tiny gate` | torch 2.14.1 from the cu130 index, vLLM out of the workspace |
| `sacrebleu-numpy` | `sacrebleu`, `numpy` | `numerics evaluation_tiny` | `D13-SACREBLEU-STATISTICS`, `D13-SACREBLEU-PAIRED-TEST`, `D13-OWN-BOOTSTRAP` |
| `sentencepiece` | `sentencepiece` | `numerics gate` | the D3 tokenizer entries; `D-EOLE-VOCAB-NOT-SPM` |
| `transformers-stack` | `transformers`, `tokenizers`, `safetensors`, `sentence-transformers` | `gate` | the `transformers==5.21.0` pin |
| `polars-arrow` | `polars`, `polars-*`, `pyarrow` | `gate` | `D-POLARS-GLOB`, `D-POLARS-PANIC`, `D-POLARS-SIGINT` |
| `serving-stack` | `fastapi`, `starlette`, `uvicorn`, `h11` | `serving_tiny gate` | `D-SERVE-SIGTERM-EXIT`, `D-UVICORN-LIFESPAN-ORDER` |
| `hub-client` | `huggingface-hub`, `datasets`, `hf-xet` | `evaluation_tiny gate` | `D13-HUB-OFFLINE`, `D13-HUB-CREATE-REPO-EXIST-OK` |
| `vendored-wheels` | `pyonmttok`, `fasttext-wheel`, `fasttext-new` | `gate` | the vendored builds under `vendor/` (the `pyonmttok` wheels, the `fasttext-wheel` shim over `fasttext-new`) |

**The blind spot, stated in every `maintain_report.md`:** a package pinned with `==` never appears in the dry run, so the rules for `eole`, `ctranslate2`, `sacrebleu`, `sentencepiece`, `torch`, `transformers` and `tokenizers` fire only after a pin is loosened; they fire today for the packages the dry run can see (numpy, polars, pyarrow, starlette, uvicorn, `huggingface-hub`, the nvidia
and cuda wheels, `safetensors` and others).

**The lines of a real change are UNMEASURED.** The offline dry run that could print them fails on this machine (the cache holds the Linux resolution only; **spike** F3), so the parser's sample is HAND-WRITTEN (`CHANGES_HAND_WRITTEN` of `tests/support/uv_outputs.py`: `Update numpy v2.5.4 -> v2.5.5`, `Add tomli-w v1.2.0`, `Remove sortedcontainers v2.4.0`) and the two measured outputs
are the no-change text (`Using CPython 3.14.10 interpreter at: <path>`, `Resolved 187 packages in 1ms`, `No lockfile changes detected`, exit 0) and the offline failure (exit 1; the exit code decides first). **Second-hand:** the Task 7 agent reports that a mutation run of its own, by a slip, ran the REAL online `uv lock --upgrade --dry-run` once in a copy of the tree, and that uv 0.13.7 printed
`Resolved 179 packages in 1.11s` as its first line (no `Using CPython` line), then `Update <name> v<old> -> v<new>` and `Remove <name> v<ver>` lines (51 entries: 43 updates, 8 removes, no add), all recognised by this parser, every old version equal to the lock's, the state `drift` with 9 blocked rows. That log was overwritten and is NOT a recorded fixture; a count consistent with it is
ARITHMETIC: the lock has 187 entries and 187 - 8 removes + 0 adds = 179. Still UNMEASURED: an `Add` line, the exit code when there are changes (the design requires 0), whether the torch index is queried in the same run, and uv's elapsed form for a long resolution: the parser takes one token after `in` (`1ms`, `1.11s`, `2.31s`, `3m` are recognised, **measured** here
in process), so a form with a space (`1m 02s`, recalled from uv's source and never seen) is one unrecognised line and the night is red with `the output of uv has 1 lines this version does not recognise`: the intended failure, fixed by one regular expression and a fixture. **The first real night's `logs/drift.log` must become a fixture of `tests/support/uv_outputs.py` and an amendment.** The nightly job
keeps no log and uploads nothing, so the operator who needs it runs the command once on a machine they trust (`rinchen maintain check --out-dir DIR`, the nightly form of example 2) and reads `DIR/logs/drift.log`.

**How to add a pin rule.** Add an entry under `drift.rules` in `configs/maintain/maintain.yaml`: `id` (`[a-z0-9-]{1,40}`, unique), `packages` (1 to 12 normalised names, or a prefix ending in `*`), `decision` (one line naming the `DECISIONS.md` entry or the `docs/VERIFY.md` row ids that pin the package) and `requires` (reruns of the same file, and `gate`). Order matters: the first match wins. The shipped table is
pinned literal by literal in `tests/unit/test_config_maintain_shipped.py` (the ten ids in order, the six reruns, every row id a decision names must exist in `docs/VERIFY.md`): the edit of that test is the decision. A wrong table mislabels and never blocks; the cost of a missing rule is a free `drift` row for a package that deserved a gate, and the upgrade procedure runs the full gate after every
upgrade anyway.

## The parity reruns

A rerun is an entry of `reruns` in the configuration. `kind: pytest` runs `python -m pytest -q -m "not network" -p no:cacheprovider --junitxml=<out>/junit/<id>.xml <targets>` from the project directory with the interpreter that runs the command and the hidden environment (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`);
the outcome is read from the exit code AND the JUnit counters (the only thing read from the XML, which holds an absolute path and a host name). It decides in this order and stops at the first that holds: a target that does not exist (before anything runs); a timeout; a process that did not start; no readable JUnit report; failures or errors; fewer passed than `min_passed`; more skipped than
`max_skipped` (default 0); a non-zero exit code; else `ok`. A selection that shrank or collected nothing (pytest exit 5) is therefore a failure and never a pass (**gate** R7: `0 passed, fewer than the floor 1`). `kind: export_parity` runs `python -m rinchen export parity --checkpoint C --export E --corpus F ... --out-dir <out>/parity_export --max-lines 50 --threads 2 --timeout <bound - 30>`, reads the exit code
(0 `ok`, anything else `fail`: `the command exited with code N`) and the three figures of `parity.json`; without `--export`, `--checkpoint` and `--corpus` it is `skipped (no export was given)` whatever the cadence. A rerun never selects a test of the maintenance check itself (the loader refuses a target that names `maintain`), so a rerun cannot rerun the gate of the command that makes it. Shipped, with
the bounds (at least three times the spike's time and at least 300 s: priors for a runner):

| Rerun | Cadence | Bound | Floor | Targets | Measured on this machine |
|---|---|---:|---:|---|---|
| `numerics` | nightly | 300 s | 1,100 | 15 test files (the sacrebleu statistics and their private API, `corpus_scores`, the bootstrap with its pure-Python reference and determinism, D13's golden report and golden suite, the significance test, D12's numbers, placeholders and script metrics) and `tests/golden` (the 289 normalization goldens) | 1,110 passed in 4.64 s (**spike**), 1,110 (**gate**) |
| `parity_tiny` | nightly | 600 s | 13 | `tests/integration/test_export_parity.py`: Eole against CTranslate2 on the toy models, the greedy output and the log-probabilities | 13 passed in 137.24 s, 120.07 s of it the setup (**spike**); 13 (**gate**) |
| `parity_export` | nightly | 600 s | | the parity command over an operator's export, checkpoint and corpus | 200 of 200 exact, mean abs delta 7.19e-07, max 3.43e-06, about 14 s with the doctor (**gate** R6); 4.06 s alone (**spike**) |
| `robustness_tiny` | weekly | 1,200 s | 31 | D12's two gate modules, which hold the floors for numbers, placeholders, names, terms and script purity | 31 passed in 388.27 s (**spike**) |
| `evaluation_tiny` | weekly | 1,200 s | 99 | D13's three gate modules: `eval run`, `eval report`, `release stage` and the dry-run upload plan | 99 passed in 384.40 s (**spike**) |
| `serving_tiny` | weekly | 1,000 s | 55 | D11's serving gate, its process tests, the stop-budget test and the engine surfaces | 55 passed in 309.78 s (**spike**) |

What the reruns cover: Eole against CTranslate2 (`parity_tiny`, `parity_export`), D13's golden report and bootstrap byte for byte and the sacrebleu private API (`numerics`), D12's floors (`robustness_tiny`: the gate modules hold them as literals; `numerics` holds the metric goldens), D13's evaluation and release gate (`evaluation_tiny`) and D11's serving gate (`serving_tiny`). Robustness is not rerun over an operator's export because D12's command has no threshold
(`eval robustness` exits 0 whenever it measured; **spike** F9).

**A floor is never lowered.** The floors are the thresholds that D11 to D13 wrote as literals of their gate modules and said are never lowered; `min_passed` is the guard on a selection that shrank. A rerun that misses a floor is answered by finding out what moved (below), never by editing a number. Note the margins (ARITHMETIC): `numerics` has 10 tests of margin (1,110 against 1,100) and the other four pytest reruns none (13, 31, 99 and 55 against the same numbers),
so one deleted test in a target of those four fails its rerun, while ten deleted goldens still pass `numerics`; a new file in a rerun's targets raises the count and the floor stays, so raising a floor is a decision to take on purpose.

**How to add a rerun or a cadence.** Add an entry to `reruns` (`id` `[a-z][a-z0-9_]{0,31}` and never `gate`; `kind`; `cadence` `nightly` or `weekly`; `timeout_seconds` 1 to 14,400; a one-line `why`; for `pytest` 1 to 20 `targets` under `tests/` and a `min_passed` of at least 1 measured on the target; `max_skipped`), then edit the pinned list of reruns in `tests/unit/test_config_maintain_shipped.py` and the timeout of
`.github/workflows/nightly.yml` if the weekly bounds grew (a test computes the least `timeout-minutes` from the configuration: 104 minutes for the shipped table, **measured** here, against the workflow's 150). The cadence of a rerun is data, not code: moving a rerun from `weekly` to `nightly` is one word, and a cadence that is too slow is a number in a file. A flaky weekly rerun would need a retry policy;
none is built (the toy trains from a seed, and the gate's training is byte-identical across runs on one machine).

## The workflow

`.github/workflows/nightly.yml` (52 lines): triggers `schedule` (`17 3 * * *` every night and `47 4 * * 0` on Sunday, so on Sunday both fire) and `workflow_dispatch` with a `cadence` choice input (`nightly` or `weekly`); `permissions: contents: read`; one job `maintain` on `ubuntu-24.04` with `timeout-minutes: 150`; the cadence in an `env` expression
`${{ github.event.inputs.cadence || (github.event.schedule == '47 4 * * 0' && 'weekly' || 'nightly') }}`; steps: `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.3`, `actions/setup-python@v7` (3.14), `extractions/setup-just@v4` (the same actions at the same versions as `ci.yml`, checked by a test), `uv sync --frozen --group dev`, the check
`uv run --frozen --no-sync rinchen maintain check --out-dir maintain-out --cadence "$CADENCE"`, `just tiny` (the stub until D15: the real pipeline is picked up without a workflow edit) and, with `always()` and only when the file exists, `cat maintain-out/maintain_report.md >> "$GITHUB_STEP_SUMMARY"`. **Where the report appears:** the run page's summary. The job fails when the command exits 4, which is GitHub's own notification;
nothing needs a secret. If the check step fails `just tiny` is skipped and the summary step still runs.

**What it never does** (a static test with PyYAML, `tests/unit/test_nightly_workflow.py`, scans the file and broken copies of it; the trigger is under the key `True`): use a secret; upload (no `upload-artifact`, no `--upload`); push, commit, open or merge a pull request; run `uv lock`, `uv add`, `uv pip`, `uv sync` without `--frozen` or any `--upgrade` (the dry run happens only inside the command, whose loader refuses a configuration without `--dry-run`);
mention docker, a GPU runner, CUDA, nvidia or a self-hosted runner; print the environment. No step caches a virtual environment, a Hugging Face directory or a model. Because it uploads nothing, `maintain-out/` vanishes with the runner: the job log holds the command's printed lines and the doctor's `[FAIL]` messages, and the summary holds the Markdown, and nothing else survives.

**The weekly set by hand.** On GitHub: the Actions tab, the workflow `Nightly`, `Run workflow`, `cadence` `weekly` (or `gh workflow run nightly.yml -f cadence=weekly`): UNMEASURED, the workflow never ran. Locally: example 3 above.

**Cost.** The job's cost is UNMEASURED on a runner. ARITHMETIC on two assumptions that are themselves UNMEASURED (a runner three times slower than this 20-core machine, which is also why the bounds are three times the spike's times, and 2.5 minutes of set-up for the sync): the nightly job takes about 10 minutes (3 x 150 s = 7.5 minutes), the weekly one about 64 minutes (3 x 1,232 s = 61.6 minutes), and a 30-day
month about 560 minutes (30 x 10 + 4 x 64) if every Sunday runs both. The minutes a private repository gets are not checked here.

## The upgrade procedure

For an operator, on a branch, never on `main`. None of these commands was run for this runbook (each queries or changes the stack); the table of the last report names the packages and which reruns each needs.

1. `git switch -c upgrade/<package>-<date>`. Read the last `maintain_report.md`: pick ONE package or ONE family from the drift table and read its `blocked` row (`requires`).
2. `uv lock --upgrade-package <NAME>` (never `--upgrade` for everything). Read `git diff --stat uv.lock` and the lines that moved: a package you did not name should not have moved; a new transitive package is a finding.
3. `uv sync --group dev`.
4. `just gate` (ruff format, ruff check, mypy, the whole fast suite: about 25 to 40 minutes on this machine, depending on the load).
5. `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --cadence weekly --no-drift`, adding `--export EXPORT_DIR --checkpoint CHECKPOINT_DIR --corpus CORPUS.parquet` for a real export, and keep every rerun the package's rule `requires` green: `ok` in the report, never a lowered floor.
6. **The golden decision rule.** A golden or a parity figure that moved is one of two things: a regression, and the answer is to revert the upgrade ("The rollback" below), or an intended change of the new version, and the answer is to check every moved literal INDEPENDENTLY (the pure-Python references the tests already hold, the spike's scripts), regenerate the golden only then,
   and record in `DECISIONS.md`, in the same commit, the new golden and the reason. A golden is never regenerated to make a rerun green and a floor is never lowered.
7. Record the upgrade in `DECISIONS.md` (package, from and to versions, the rule's id, the reruns run and their results, the date) and refresh the hand snapshot `docs/STACK.md` from `uv pip list` (it is Windows-only and last generated 2026-10-30: say so). Merge by the usual review.

Loosening an exact pin is a decision, not a maintenance step: it is written in `DECISIONS.md` before `pyproject.toml` changes, and the upgrade then follows the same steps. If a uv upgrade writes a lock with a format `version` other than 1, `rinchen maintain check` exits 3 at the lock reader: the reader does not guess a newer layout, and teaching it is a code change with a fixture.

## The rollback

1. `git revert <the commit that moved uv.lock>` (or `git checkout <good commit> -- uv.lock pyproject.toml`).
2. `uv sync --frozen --group dev`.
3. `uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --no-drift` (`--cadence weekly` if the upgrade touched a weekly rerun's subject) until `environment`, `lock_consistent` and the parity reruns are `ok`.
4. A line in `DECISIONS.md` saying what was rolled back and why.

## The monthly hand check of the pins

The dry run cannot see the 12 exact pins, so once a month an operator reads the newest release of each of them (`torch`, `eole`, `pyonmttok`, `ctranslate2`, `transformers`, `tokenizers`, `sentencepiece`, `sacrebleu`, `kiwipiepy`, `fasttext-new`, `fasttext-wheel`, `datasketch`; `pyonmttok` and `fasttext-wheel` are vendored under `vendor/`, so for them a newer upstream release means rebuilding a vendored artifact) against the pin in
`pyproject.toml`, on the package's own index page. `uv tree --outdated` may show it; its output was never seen here (the flag is in the binary's strings, **spike** F13): UNMEASURED and not built. A newer release is a decision to loosen or raise a pin: write it down first, then follow "The upgrade procedure".

## What each failure means

Each row of the report names a reason; the table says what it means and the sections below say what to do. The links of `maintain_report.md` point at `#failure-<step id>`; each rerun has a section of its own too.

| Where | Reason | Meaning |
|---|---|---|
| doctor | `the check failed` | a broken install |
| lock_file | exit 3, no report | a lock this version of rinchen cannot read |
| lock_consistent | `uv exited with code N`, `timed out after 120 s`, `uv was not found` | `pyproject.toml` moved without `uv lock`, or uv is missing |
| environment | `<m> packages differ from the lock, <e> are not in it` | a machine that is not the lock, or a stray package |
| drift | `uv exited with code N`, `timed out after 300 s`, `uv was not found`, `the output of uv has <n> lines this version does not recognise`, `uv printed no recognisable result`, `uv printed a no-change line next to change lines`, `uv reported a locked version that differs from uv.lock for <n> packages` | uv failed, or printed what the parser does not know |
| drift | the state `drift` | the normal state: nothing is wrong |
| reruns | any rerun `fail` | see the rerun's own section |

### failure-doctor

A doctor check failed: the install is broken. The failed checks and their messages are in the job log (or on standard output) as `[FAIL] <id>: <message>`; the report holds the ids and states only. `docs/VERIFY.md` explains each id (`D-PY` CPython 3.14, `D-WHEELS` every direct dependency installed, `D-EOLE-PY`, `D-EOLE-CLI`, `D-CT2-CONV`, `D-CT2-IMPORT`). Repair the environment
with `uv sync --frozen --group dev` and run the preflight (example 1). The reruns are skipped meanwhile (`a doctor or environment check failed`).

### failure-lock_file

`uv.lock` could not be read: the command exits 3 and writes no report (in every report that exists this step is `ok`). The message names the field (`uv.lock: version is not the integer 1`, `package[3].version is missing, empty or not a string`, `not valid TOML`) and quotes no bytes. A lock with another format `version` means uv was upgraded: the reader refuses to guess. Restore the lock
(`git checkout -- uv.lock`) or teach `rinchen.maintain.lock` the new layout with a fixture.

### failure-lock_consistent

`uv lock --check` failed: `pyproject.toml` moved without the lock (the CI sync is `--frozen`, which never notices). Read `logs/lock_consistent.log` on a local run; on a branch run `uv lock`, then the gate, and commit both files. `uv was not found` means no `uv` on `PATH` (or in `--uv`); `timed out after 120 s` is the bound. The shipped arguments are `lock --check` without `--offline`, so on an invalid
lock uv re-resolves, which reaches the index when online: UNMEASURED online (offline it took 0.075 s, **spike** F4).

### failure-environment

The machine is not the stack the lock names. The report lists up to 20 names of each kind: `mismatched` (installed with a version the lock does not list for that name; `installed` and `locked` are the versions) and `extra` (installed and not in the lock at all). Run `uv sync --frozen --group dev` (an exact sync removes the extra packages), then the preflight. A locked package that is NOT installed is only counted
(`not_installed`): the lock holds the packages of other platforms. Versions are strings compared for equality, never numbers (`2.14.1+cu130`).

### failure-drift

uv failed, timed out, was not found, or printed what the parser does not know (the reasons above). A drift table is NOT a failure; `fail` here is the check being unable to tell. Read `logs/drift.log` (a local run: the nightly job keeps none): an exit code is uv's own (an index that cannot be reached, a resolution that fails); an unrecognised-lines count with exit 0 is a new uv format, which is a fixture in
`tests/support/uv_outputs.py` and one regular expression in `rinchen.maintain.uv_output`, never a configuration change; `differs from uv.lock for <n> packages` means a change line named a version the lock does not hold (a mis-parse, or a lock that moved between the hash and the run). For a `blocked` row, run the reruns it `requires` before taking the package.

### failure-reruns

At least one rerun failed. Find the item in `detail.items` (or the rerun table of the Markdown), read its reason and go to the rerun's section. The weekly reruns fail for the same reasons one layer up. A rerun that was `skipped` is not a failure: its reason says why (`not due at the nightly cadence`, `no export was given`, `a doctor or environment check failed`, `disabled by --no-reruns`). To reproduce one by hand:
`uv run --frozen --no-sync rinchen maintain check --out-dir /tmp/maintain --no-drift --rerun <id>` and read `logs/<id>.log`.

### failure-numerics

A library, the Python patch release or the runner's C library moved a number that a golden pins: the sacrebleu statistics or its private API, the bootstrap draws, D13's golden report, D12's metrics, or a normalization golden. The reasons: `<f> failed and <e> errors` (read the failing test's name in the log), `<p> passed, fewer than the floor 1100` (a selection that shrank), `pytest exited with code N`.
Revert the upgrade, or apply "The golden decision rule". BLEU's last digits depend on the C library (D13's limit 6): a failure on a new runner image with no change of the lock is that, and is recorded, not waved through.

### failure-parity_tiny

Eole, CTranslate2 or torch moved the greedy output or the log-probabilities of the toy models (13 tests; the bound 1e-3 per token is a float32 bound). Do not upgrade; read the failing test and `logs/parity_tiny.log`. The rerun trains two toy teachers inside its first test (120 s of its 137 s): a timeout (`timed out after 600 s`) on a slow runner is the bound, not a parity finding: look at the log's last lines
first, and raise the bound in the configuration (a bound, unlike a floor, is a number about a machine).

### failure-parity_export

Eole and CTranslate2 disagree over YOUR export: `the command exited with code N` (4 is a measured miss; 2 and 3 are refusals such as another model's checkpoint or a changed byte of `model.bin`, **gate** R6, which exits 2), or `the command wrote no readable parity.json`. Read `parity_export/parity.json` (`exact`, `sentences`, `mean_abs_delta`, the mismatch list) and `logs/parity_export.log`. The report carries only the three figures of
the file. Do not upgrade; the export is not a good export until this is `ok`.

### failure-robustness_tiny

D12's gate failed on the toy (numbers, placeholders, names, terms, script purity, the traditional-script source, the controls): one layer above `parity_tiny`, the same causes (a moved library, a changed greedy output) plus anything the robustness metrics read. The gate's floors are literals (numbers 0.90, placeholders 0.99, entities 0.90, terms 0.99, script 0.999, retention 0.95, **spike** F9); a miss is answered with more steps of the toy
(`ROBUST_TEACHER_STEPS`), never a lower floor, after finding out what moved.

### failure-evaluation_tiny

D13's gate failed on the toy: the evaluation run, the report, the paired comparisons, the card or the dry-run plan. A moved sacrebleu or numpy fails the golden tests on purpose (`numerics` fails first and says so); a figure under a floor is answered with more steps of the toy, never a lower floor (`docs/runbooks/13_release.md`, "What to do when a floor of the gate is missed").

### failure-serving_tiny

D11's serving gate failed on the toy: the engine surfaces, the HTTP service under 64 concurrent requests, the drain on SIGTERM (exit 143), the stop budget. `fastapi`, `starlette`, `uvicorn` and `h11` are the usual suspects (`D-SERVE-SIGTERM-EXIT`, `D-UVICORN-LIFESPAN-ORDER` in `docs/VERIFY.md`: uvicorn exits by the re-raised signal and starts its lifespan before it binds); read `logs/serving_tiny.log`.

## Costs

**Measured by the gate** (2026-11-11; one thread per command, load varies, a TIME moves by a third): the command of the workflow through the shim, exit 0, about 150 s per night (the manifest's seconds: doctor 13.2, the lock and dry-run steps 0.1, the two nightly reruns 136.2) and 163.6 s on a first look run, with the two toy teachers training inside `parity_tiny`;
a real run of the command with no reruns about 13 s (the doctor); the seven reruns of R7 about 15 s with the doctor. **Spike** (F1 to F11, offline): the doctor 15.6 s (`D-EOLE-CLI` 10.72 s, `D-CT2-CONV` 2.46 s, `D-TORCH-CUDA` 0.74 s when the device is hidden, which is why it is never called); `uv.lock` parsed in 0.024 s; `uv lock --check --offline` 0.075 s; the environment comparison 0.058 s;
`mypy src` 19.4 s; collection of 17,853 tests 3.15 s; the numerics rerun 4.64 s; `parity_tiny` 137.24 s; the parity command 4.06 s; `robustness_tiny` 388.27 s; `evaluation_tiny` 384.40 s; `serving_tiny` 309.78 s; the full gate 25 min 44 s as parallel parts (D13's record). **ARITHMETIC:** the nightly set about 157 s (15 + 0.15 + 4.64 + 137), the weekly set 1,082 s more (388.27 + 384.40 + 309.78), about
21 minutes in all on this machine; the sum of the bounds 4,900 s, 5,620 s with the lock check, the dry run and 300 s of set-up, so the least `timeout-minutes` is 104 (the workflow has 150). **UNMEASURED:** the online dry run (uv's own elapsed time was `1.11s` in the second-hand report, which says nothing about the network), a runner's speed and cores, the minutes budget, and what the setup actions cache.

## What D15 takes

(1) The state vocabulary (`ok`, `skipped` with a reason, `fail`) and the way a step lists its reason: the tiny pipeline's `report.json` can carry its stages the same way. (2) A quick preflight, `rinchen maintain check --no-drift --no-reruns --out-dir DIR`: the doctor, the lock check and the environment check in about 13 s, with nothing a network or a model needs; **D15 decides** whether `just tiny` calls it as a stage. (3) The parity of
D15's own tiny export: `--rerun parity_export --export DIR --checkpoint DIR --corpus FILE` records D11's command by name, or D15 calls `rinchen export parity` itself. (4) The nightly job already runs `just tiny`, so D15's real pipeline is picked up by the schedule without a workflow edit. Nothing in D14 blocks D15 and D15 needs nothing of D14.

## The STOP gate

The master plan's gate for D14 is "nightly workflow green on tiny". It is met ONLY as: the command of the workflow's own maintain step, read from `nightly.yml` and run natively with `CADENCE=nightly` on the CPU with the GPU hidden, through a `uv` shim that never forwards the upgrade dry run, runs green on the tiny profile. GitHub Actions is **UNMEASURED**. The two modules (`tests/integration/test_maintain_gate.py`, 27 tests, R1 to R6, and
`test_maintain_gate_controls.py`, 170 tests, R7 to R10) run the real `rinchen` binary, Eole, CTranslate2 and pytest, every subprocess bounded and every floor a literal of `tests/support/maintain_gate.py` that is never lowered. The `uv` the command finds is the shim of `tests/support/fake_uv.py`: `lock --upgrade --dry-run` (the two flags in any order, nothing else) is answered from the HAND-WRITTEN sample and never forwarded;
`lock --check` is forwarded as `lock --check --offline`; `--version` is forwarded; `run <rest>` is forwarded as `run --project <repository> --no-sync --frozen --offline <rest>` with the caller's duplicate flags stripped (amendment A1); every other argument list, and every list holding `--upgrade`, `--upgrade-package`, `-U`, `-P`, `sync`, `add` or `pip`, exits 97 and is refused. Every forwarded call runs with `UV_OFFLINE=1`. The gate cannot prove the
absence of a socket in a child process: what keeps it off the network is the shim, the offline switches and the empty device list, and it checks the shim's call log and the bytes and modification times of `uv.lock` and `pyproject.toml`. **Gate** (2026-11-11, uv 0.13.7, Eole 0.6.4, CTranslate2 4.8.4):

| Gate | Threshold | Measured |
|---|---|---|
| R1 the night | exit 0; schema `rinchen.maintain/1`, cadence `nightly`, state `drift`, six steps in order; six CPU checks `ok`, the GPU check `skipped`; the lock figures equal the lock's; `lock_consistent` and `environment` `ok`; the table of the sample (7 entries, 5 updates, 1 add, 1 remove, 3 free and 4 blocked: `huggingface-hub`, `numpy`, `nvidia-cudnn-cu13`, `safetensors`); `numerics` at least 1,100 passed and `parity_tiny` exactly 13, none failed or skipped; `parity_export` and the three weekly reruns `skipped` | the command ran twice, exit 0, 163.6 s on a first look and about 150 s inside the gate; the lock 187 entries, 186 names (`pyonmttok` twice at one version), format version 1 revision 5; environment 186 installed, none differing, none extra; numerics 1,110 passed, `parity_tiny` 13 |
| R2 the manifest | stage `maintain_check`; `tool_versions` with every lock name and `python` and `uv`; the three inputs with the digests of their bytes; every output listed with its digest; the six states | 188 `tool_versions` keys; 8 outputs (the two documents of 6,138 and 1,806 bytes, four logs, two JUnit reports), none missing, none extra |
| R3 hygiene of the documents | neither holds an absolute path, a timestamp, a host name, HOME, USER or PATH, the output directory or a line of uv's output | clean |
| R4 nothing written, nothing forwarded | `pyproject.toml` and `uv.lock` keep their bytes and modification times; the shim's log shows the dry run local, the real uv called as `lock --check --offline`, `--version` and `run --project ... --no-sync --frozen --offline ...`, no flag twice, no refusal | four shim calls per run, none refused; `uv.lock` `559b3954...` unchanged |
| R5 determinism | the two documents byte-equal on a rerun under another hash seed | equal under `PYTHONHASHSEED=3` |
| R6 an export | `--no-drift --rerun parity_export` over the toy's float32 export: exit 0, every sentence exact, mean abs delta under 1e-3, the others skipped; one flipped byte of `model.bin`, or another model's checkpoint: the rerun `fail` and exit 4 with everything written | 200 of 200 exact (50 per direction), mean abs delta 7.19e-07 (max 3.43e-06), about 14 s a command; the flipped byte and the other checkpoint: the parity command exits 2, the rerun is `fail`, the check exits 4 |
| R7 the reruns through the real pytest | a failing test, an erroring one, fewer passing tests than the floor, a skipping test with no skip allowed, an empty target (exit 5) and a rerun past its bound are each `fail` and never `ok`; the bound kills the process group; a passing rerun sees the device hidden and the Hub offline; every other step is still reported; exit 4 | `1 failed and 0 errors`, `0 failed and 1 errors`, `2 passed, fewer than the floor 3`, `1 skipped, more than the allowed 0`, `0 passed, fewer than the floor 1`, `timed out after 3 s` within 10 s with both pids of the sleeping test gone |
| R8 the drift | random text, the measured offline failure, a sample whose old version disagrees with the lock are each `fail` and exit 4; the no-change text is `ok`; `--fail-on-drift` turns a table into exit 4; `--no-drift` makes the shim see no dry run; a configuration without `--dry-run` is refused (exit 2, nothing written) | as asserted; about 13 s a run |
| R9 refusals and interrupts | a missing `uv.lock` and a foreign manifest exit 3 and `--export` alone, an unknown `--rerun`, the project's directory as output exit 2, each with nothing written; SIGINT exit 130 and SIGTERM exit 143 leave no manifest, take the old one away and no process | as asserted; the refusals under 1 s; the interrupts end in 13.4 s |
| R10 hygiene of the code | no file D14 adds asks a device question, none names `uv.lock` or `pyproject.toml` next to a write, no argument list holds an add, a sync, a pip or `--upgrade-package`, the one `--upgrade` of the code base is the shipped configuration's next to `--dry-run`, every subprocess has a timeout, the workflow holds none of the words of its scan | the scans of 20 modules and 7 helper files: no finding, and 22 sensitivity cases that each break one rule |

The gate was mutation-checked (26 mutants of the product, 12 of the shim and 8 of the helpers, all killed), so the controls are not vacuous. The real `uv lock --check --offline` forwarded fine through the shim (**gate**: it was not denied by the harness, so that part is measured). **What the figures show:** that the workflow's command runs green natively, that the dry-run text is refused when it is not recognised, that a failing,
shrunken or overrunning rerun is red, that the documents are deterministic and free of paths, and that nothing of the project was written. **What they do not show:** anything about GitHub Actions, the real PyPI, the lines of a real change, a real export, or a GPU. The toy ranks nothing.

## Open gates

Not done by D14, because they need what does not exist here: everything GPU (the doctor's GPU check is never run; a CUDA wheel upgrade, an `nvidia-*` change, cannot be qualified); the real PyPI state and the lines of a real change (the first real `logs/drift.log`); GitHub Actions (that the workflow starts and finishes inside its timeout, that `github.event.schedule` holds the cron that fired, that `hashFiles` and
`env.CADENCE` work in a condition, the runner's speed and the minutes budget); every real export (`parity_export` ran only on the D11 toy export; no robustness or evaluation rerun over a real export exists); and the gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D13 gates that need real data, a GPU, a reader or the Hub.

## What was not verified

- **GitHub Actions.** `nightly.yml` was checked statically only (PyYAML; the same pinned actions as `ci.yml`; no secrets, no upload, nothing about docker, a GPU or a self-hosted runner) and its command ran natively; the cron strings, the cadence expression (modelled by `tests/support/workflow_expression.py` for the four triggers), `hashFiles`, the timeout and the minutes were never run. No CI run covers D3 to D14.
- **The real dry run.** See "The drift table": the lines of an `Add`, the exit code with changes, the torch index, the elapsed form of a long resolution; the second-hand report is not a fixture. One mutation run of Task 7 ran the real online dry run once in a copy of the tree (it writes nothing) and another started the command for a few seconds with `CUDA_VISIBLE_DEVICES=0` (no compute: the doctor's CPU checks ask no
  device question; no manifest was written); the gate and the measurements of this runbook did neither, and the gate now sets `UV_OFFLINE=1` in every environment and the shim sets it on every forwarded call.
- **Anything on a GPU, any real model, corpus, export, benchmark or baseline.** The reruns are the toy's.
- **A runner.** The bounds are three times a 20-core machine's single-thread times; a runner's speed, memory and set-up time are not known.
- **The environment step on a runner.** It fails on any installed distribution the lock does not name; a runner's environment was never seen (uv is recalled not to seed a virtual environment with `pip`, which is why none is expected; this was not run there).
- **The exit code 1 with a real full disk, a `SIGKILL` of the command** (the commands it started would keep running: a limit), Windows and macOS (the runner is POSIX only: `start_new_session`, `killpg`).
- **The combined review of the code**: not yet run (see "The review").

### Known limits, left open on purpose

1. **The blind spot**: 12 of 38 direct dependencies are exact pins and never appear in the drift table; the monthly hand check is a person's, and `uv tree --outdated` is unbuilt.
2. **The format of a dry run is learned, not known**: see "The drift table". A night can be red with `the output of uv has <n> lines this version does not recognise`, which is the intended failure; the nightly job keeps no log, so the first real `logs/drift.log` comes from a local online run. The parser's grammar is strict (single spaces, one token for the elapsed time); `Lockfile changes detected`, which the binary holds as a string, is pinned as unrecognised.
3. **Priors**: the rule table, the floors and the bounds come from the recorded decisions and one machine; the floors are the gate modules' literals and have margins of 10 (`numerics`) and 0 (the other four).
4. **The toy's determinism is one machine's**: the gate's training is byte-identical across runs here (D12-TOY-TRAINING-DETERMINISM), but another CPU or another torch may differ, and a weekly rerun on a runner could miss a floor that this machine met with margin: the answer is more steps of the toy, never a lower floor.
5. **The nightly job keeps nothing**: no upload by rule, so a failed night leaves the job log, the printed lines, the doctor's `[FAIL]` messages and the Markdown summary; the logs of uv and pytest must be reproduced locally.
6. **`lock_consistent` online** (`lock --check` without `--offline`) is UNMEASURED: valid, it does not need the index (1ms offline); invalid, uv re-resolves.
7. **A rerun reports counters only** (`parity_export` its three figures); the mismatch list of `parity.json` and every failing test's name are in the local files. `no JUnit report was written` is also the reason when the file exists but cannot be read as a JUnit report.
8. **Signals**: a SIGKILL of the command leaves the commands it started running until their bounds; an interrupt leaves partial logs that the next run overwrites and that no manifest lists; an `installed` seam that raises (it does not, in the real command) would end the command with exit 1 and no manifest; `exit_on_termination_signals` duplicates a private helper of `rinchen.utils.logged_process` (a shared public helper needs an edit to an existing module).
9. **The doctor's messages** are printed to standard output and can hold an exception text and a path; in the nightly job they land in the run log.
10. **Size**: `tests/unit/test_maintain_check.py` has 852 lines and `tests/integration/test_maintain_gate_controls.py` 892 (the limit is 900): any addition needs a split.
11. **`docs/STACK.md`** stays a hand snapshot (Windows-only, 2026-10-30); the manifest of a check is the machine-readable record, and a script that regenerates the snapshot from it is the obvious next step.
12. **Disclosed process slips of the D14 build** are listed, as facts and one line each, in the `[PROCESS ...]` paragraph of the `DECISIONS.md` entry; the two that limit a claim are in "What was not verified".

### The review

**Not yet run.** A combined review of the code (Tasks 1 to 7, `5477430..51c4489`) has not run when this runbook is written; a follow-up commit, as D13's was, will record its verdicts, the fix wave and the limits it leaves open here and in the spec's section 7.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython 3.14.10, uv 0.13.7, eole 0.6.4, ctranslate2 4.8.4, sentencepiece 0.2.1, sacrebleu 2.6.0, numpy 2.5.4; the GPU was hidden from every process.

| What | Value |
|---|---|
| the lock | 187 entries, 186 names, format version 1, revision 5, SHA-256 `559b3954...b16`; 420,102 bytes parsed in 0.024 s (**spike**, **gate**) |
| the environment | 186 installed, none differing, none extra, none missing (**spike** F5, **gate** R1) |
| direct dependencies | 38, of which 12 pinned with `==`; dev group 10, none pinned (**measured** here, **spike** F13) |
| the night | about 150 s through the shim: doctor 13.2 s, lock steps 0.1 s, two reruns 136.2 s; 163.6 s on a first look (**gate**) |
| the drift table of the sample | 7 entries (5 updates, 1 add, 1 remove), 3 free, 4 blocked (**gate** R1, golden) |
| numerics, `parity_tiny` | 1,110 passed (floor 1,100), 13 passed (**gate**) |
| the export parity | 200 of 200 exact, mean abs delta 7.19e-07, max 3.43e-06, bound 1e-3 (**gate** R6, **spike** F8) |
| the documents | `maintain_report.json` 6,138 B, `maintain_report.md` 1,806 B for the real lock; goldens 6,132 B and 1,806 B (**gate**, golden) |
| the manifest | 188 `tool_versions` keys, 8 outputs (**gate** R2) |
| determinism | the two documents byte-equal under `PYTHONHASHSEED=3` (**gate** R5) |
| the shim | 4 calls per night, none refused (**gate** R4); 238,741 argument lists against an oracle, 0 mismatches |
| the workflow | 52 lines, 150 minutes against a least of 104 (**measured** here) |
| the tests | 19,818 passed, 1 skipped, 10 deselected after Task 7 (17,852 at the end of D13; 18,424, 18,774, 18,972, 19,152, 19,465, 19,576 after Tasks 1 to 6; about 19,578 with the controller's fix, the ledger's approximate figure); `ruff format --check` (833 files), `ruff check --no-cache` and `mypy src` (287 files) clean |

## Where the code lives

`src/rinchen/config/maintain.py`; `configs/maintain/maintain.yaml`; `src/rinchen/maintain/{states,lock,environment,uv_output,drift,commands,reruns,report,report_markdown,check,check_types,check_steps,check_reruns}.py`; `src/rinchen/cli/{maintain,maintain_inputs,maintain_doctor,maintain_output,maintain_signals}.py` and the edit of `src/rinchen/cli/main.py`; `.github/workflows/nightly.yml`;
`tests/fixtures/maintain/{golden_report.json,golden_report.md,uv.lock.sample}`; the gate `tests/integration/{test_maintain_gate,test_maintain_gate_controls}.py`; 29 new unit test files `tests/unit/test_{cli_maintain*,config_maintain*,fake_uv,maintain_*,nightly_workflow,workflow_expression}.py`, 4 property test files `tests/property/test_maintain_{lock,report,reruns,uv_output}_property.py`;
and the helpers `tests/support/{fake_uv,maintain_check,maintain_cli,maintain_fakes,maintain_gate,maintain_gate_hygiene,maintain_project,maintain_reports,maintain_scans,nightly_workflow,uv_outputs,workflow_expression}.py`.
