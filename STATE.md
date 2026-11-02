# STATE.md

**Plan switched 2026-10-29.** The master plan is now
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md` (milestones D0-D15),
superseding the Phase 0-14 plan in
`docs/research/2026-10-28-python314-stack-and-prompt.md` (kept as evidence
source — see that doc's own superseded-notice). The old Phase 0 / P0.1 work
below remains valid; it maps onto the new plan's D0 milestone. Sequencing
decision honored: the pyonmttok/eole blocker (below) was resolved before
D0, since D0's own doctor checks (`D-EOLE-PY`, `D-EOLE-CLI`) and CI's sync
step require a working `eole`.

**D0 and D1 are both done; both STOP gates are fully met, including CI.**
Pushed to `origin/main` 2026-11-02. First CI run (commit `b0eca33`) failed
on `ruff check`'s `EXE001`: two pyonmttok build scripts were tracked as
mode `100644` despite having a shebang (fixed in `abdbb1c` via
`git update-index --chmod=+x`, content unchanged). That run also flagged
two GitHub notices unrelated to the failure: every pinned Action targeted
the deprecated Node 20 runtime, and `ubuntu-latest` is mid-migration to
Ubuntu 26 — both addressed in `334e30f` (bumped `actions/checkout` to
`@v7`, `actions/setup-python` to `@v7`, `actions/upload-artifact` to
`@v7`, `extractions/setup-just` to `@v4`, `astral-sh/setup-uv` to its
exact latest `@v10.3.3` since it has no floating major tag; pinned
`runs-on` to `ubuntu-24.04` instead of `ubuntu-latest` — this is a
deliberate deviation from the master plan's literal "CI: ubuntu-latest"
line, `docs/research/2026-10-29-rinchen-dev-agent-prompt.md:318`, kept
unedited as historical plan text). CI run after both fixes
(`41447258653`, commit `334e30f`) passed clean: `just gate`, `just
doctor`, `just tiny` all green on `ubuntu-24.04`. D0's STOP gate ("`just
gate` green locally AND in CI") and D1's STOP gate (no CI clause, met
locally as detailed below) are both now fully satisfied.

**D2 (human test-set annotation app) is also done; its STOP gate is fully
met, including a whole-branch post-hoc review pass.** See the full D2
paragraph below for the original build (commits `3f94ca9` through
`8d9beab`). A subsequent whole-branch review found and fixed 6 Important
and 6 worthwhile Minor findings (commits `7ace665`, `ed9ce04`, `79edb74`,
`d86dced`, `b7e7771`, and this commit): a real data-loss bug where
concurrent requests crashed with `sqlite3.ProgrammingError` and returned
HTTP 500 (`db.connect()` now passes `check_same_thread=False`); a
permanently-orphaned-claim bug where a refresh/double-click/back-button
abandoned an in-progress task forever (`claim_next_task` now re-serves an
annotator's existing `assigned` task); a non-atomic claim race that could
let two annotators claim the same task (the claiming `UPDATE` now checks
`status='pending'` and `cursor.rowcount`, retrying once on a lost race); a
missing traditional-script export round-trip test (added, passed on
first run); two fixtures and one test file that mislabeled
Latin-transliterated ("galig") Mongolian text as Cyrillic, a direct
violation of CLAUDE.md's permanent "NEVER: Use galig/Latin informal
Mongolian" rule (replaced with genuine Cyrillic Mongolian text, Unicode
block U+0400-U+04FF); a vertical-script toggle button that changed
nothing because no CSS defined the class it toggled, present on only one
of three task templates (now flips the element's own inline
`writing-mode` style directly, on all three templates); and a UI dead end
after every task submission (submit routes now re-render the annotator's
next task instead of a static "Submitted." page). Also fixed: `slots=True`
added to `models.py`'s dataclasses (convention match), a stale
`annotation_app/` line in `README.md`, a misattributed docstring citation
in `agreement.py`, and this file's/`CLAUDE.md`'s/the runbook's own
staleness (D2 was previously undocumented here as done; the runbook's
queue-priority note described pre-fix behavior; see DECISIONS.md's
2026-11-02 entry and the full D2 paragraph below for detail). Full gate after all fixes: `uv run
ruff format --check .` -- 93 files already formatted; `uv run ruff check
.` -- all checks passed; `uv run mypy src` -- no issues found in 50 source
files; `uv run pytest -q -m "not network"` -- **161 passed, 6 deselected**;
`uv run pytest -q` -- **167 passed**. [EVIDENCE: live `just gate` /
`uv run pytest -q` runs, 2026-11-02, commits `7ace665` through this one.]

Next step: push to `origin`, confirm CI green for D2's code (D2 has not
yet been pushed or run on CI -- as of this commit, `main` is 21 commits
ahead of `origin/main`, via `git rev-list --count origin/main..HEAD`),
then start D3.

D0 (environment, skeleton, doctor): `pyproject.toml` restructured to the
master plan's toolchain (see `docs/STACK.md` for resolved versions),
`justfile`/CI/pre-commit in place, `just gate` (ruff format --check, ruff
check, mypy, pytest -m "not network") green locally and in CI, `rinchen
doctor` implemented with 7 checks and showing 6 OK + 1 SKIP (the skip is
the GPU check, expected on this no-GPU dev machine and on CI's runner;
see `docs/VERIFY.md`), config loader / logging / seed / manifest modules
implemented with passing tests, CLI skeleton registers all 13 subcommand
groups. [EVIDENCE:
`docs/superpowers/plans/2026-10-30-d0-environment-skeleton-doctor.md`,
`docs/VERIFY.md`, commits 4a36b13, abdbb1c, 334e30f, CI run 41447258653,
2026-11-02.]
One real discrepancy from the master plan found and documented: eole
0.6.4's CLI has no `model average` subcommand (see `docs/VERIFY.md`'s
`D-EOLE-CLI` row) -- the build-ourselves checkpoint averager (D8/D10) is
the planned fallback, not yet built.

**D1 (Data acquisition) is done; its own STOP gate is fully met.** The
master plan's D1 STOP gate text (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`,
D1 section) is "all D1 tests green; `rinchen data fetch --profile tiny
--all` works offline using `local://` fixtures" -- unlike D0's gate, it
has no CI clause, so D1 did not need a CI run to be considered done --
though as of 2026-11-02 the branch has been pushed and CI has run green
anyway (see the paragraph above). D1 built the schema-validated source
registry
(`src/rinchen/acquire/registry.py`, a `SourceEntry` discriminated union
over five kinds: `http`/`opus`/`hf`/`commoncrawl`/`local`) and its
backends: resumable, retrying, sha256-verifying `http.py` download (with
a `local://` scheme for offline fixtures, also used by `kind: local`
registry entries), an OPUS API client (`opus.py`: `list_corpora`,
`fetch`), WARC iteration and article-text extraction (`warc.py`, proven
against a real 5-record fixture built by
`scripts/build_sample_warc_fixture.py`), a HuggingFace
streaming-to-Parquet-shards writer (`hf.py`), and a CommonCrawl CDX query
plus bounded-range WARC fetch (`commoncrawl.py`). `rinchen data fetch`
and `rinchen data ls` (`src/rinchen/cli/data.py`) dispatch on all five
registry kinds, but `registry.fetch_source`'s `http`/`local` branch (the
two share one code path, `isinstance(entry, (HttpSourceEntry,
LocalSourceEntry))`) is the only one exercised end-to-end via real
fixture bytes, via the tiny-profile `kind: local` fixtures below. The
`opus`, `hf`, and `commoncrawl` branches of `fetch_source` have no
end-to-end test against real fetched content -- their underlying module
functions (`opus.fetch`, `hf.write_shards`, `commoncrawl.query_cdx`/
`fetch_records`) have unit tests in isolation (`opus.py`'s and
`commoncrawl.py`'s via `httpx.MockTransport`, `hf.py`'s via in-memory
rows, not HTTP), and the post-hoc review fixes below added stubbed
`fetch_source`-dispatch tests for the `opus`/`hf`/`commoncrawl` branches
(covering `expected_sha` handling and destination-path nesting) with the
underlying fetch functions replaced by test doubles -- see
`tests/unit/test_acquire_{opus,hf,commoncrawl}.py` and
`test_acquire_registry*.py`.
`configs/data/sources.tiny.yaml` ships 4 offline fixture entries for
`--profile tiny`, and **all 4 use `kind: local`** (not "one per kind" --
there is no tiny-profile entry of kind `http`, `opus`, `hf`, or
`commoncrawl`); `configs/data/sources.yaml` ships with zero real
production entries by design -- see DECISIONS.md's 2026-10-30 entry: 7
candidate real-world parallel corpora were vetted with a
100-random-sample, zero-tolerance review, 6 of 7 failed (script
violations, truncation, misalignment, mistranslation), and the user
chose to build Rinchen's own standard dataset from scratch rather than
seed the registry with a flawed candidate. `docs/runbooks/01_acquisition.md`
documents disk estimates, parallelism, rerun semantics, and the
how-to-add-a-source procedure. D1's own STOP gate was run for real, not
just via the automated test added in its final task: `just gate` green
locally (102 passed, 6 network tests deselected, no ruff/mypy findings
-- a handful of files needed a `ruff format` pass first, fixed as part
of this same check), `uv run rinchen data fetch --profile tiny --all`
fetched all 4 fixtures into `data/raw/` for real, `uv run rinchen data
ls --profile tiny` listed all 4 entries, and the full `uv run pytest -q`
passed at 108. D1 added no `pytest.mark.network` tests: the project's
only network-marked tests are the pre-existing D0/Phase-0
`tests/test_fasttext_wheel_shim.py` and `tests/test_probe_env.py`, which
make real network calls (PyPI JSON API, the torch cu130 index, a real
`uv lock` run) and are unrelated to D1's acquisition code -- they do not
"pass offline against the `local://` fixtures"; they are simply deselected
by `-m "not network"` and were not run as part of this count. [EVIDENCE:
`docs/superpowers/plans/2026-10-30-d1-data-acquisition.md`,
`docs/runbooks/01_acquisition.md`, live `just gate` / `rinchen data
fetch` / `rinchen data ls` / `uv run pytest -q` runs, 2026-10-30.]

**2026-10-30 post-hoc review fixes.** A whole-branch review of D1 found
and fixed several bugs and false claims (see this file's history and the
`D1 fix:`-prefixed commits for detail): a Windows-only bug where
`http.download()` used `Path.rename()` instead of `Path.replace()`,
leaving a stale complete `.part` file behind on any re-run to an existing
`dest`; `expected_sha` being silently ignored for `opus`/`hf`/`commoncrawl`
registry entries (now verified for `opus`, rejected at schema-validation
time for `hf`/`commoncrawl` since those kinds write multiple output
files); a double-nested output directory for `hf`/`commoncrawl` entries
fetched via the CLI; `load_registry` raising a raw `AttributeError`
instead of `ConfigError` when a registry file's top level is a YAML list;
a weak OPUS "pick latest" test that could not fail on its own bug (mock
HTTP responses for every version now return distinguishable bytes); two
CWD-dependent test/script files now resolving fixture paths relative to
`__file__`; the false `hf.fetch` test-coverage claims above (and in
`docs/superpowers/specs/2026-10-30-d1-data-acquisition-design.md` and
`docs/superpowers/plans/2026-10-30-d1-data-acquisition.md`); and this
file's own false claims about D1 fixture kinds, dispatch coverage, and
network-test behavior, corrected in the D1 paragraph above. After these
fixes, `just gate` is green locally with 109 passed, 6 network tests
deselected (115 total with `uv run pytest -q`), no ruff/mypy findings,
and the D1 STOP gate was re-verified for real:
`uv run rinchen data fetch --profile tiny --all` and `uv run rinchen data
ls --profile tiny` both still work against all 4 fixtures.

**D2 (human test-set annotation app) is done; its STOP gate is fully met.**
The master plan's D2 STOP gate text
(`docs/research/2026-10-29-rinchen-dev-agent-prompt.md:343`) is "app serves
locally; 20-item fixture can be fully annotated by a scripted client and
exported." D2 was built across 8 tasks (commits `3f94ca9` schema/migrations/
import, `539f1a3` chrF wrapper, `239e68c`/`fcac627` the task-claiming/
submission engine -- `fcac627` fixed the engine's docstring to match its own
kind-priority tie-break, adjudicate > review > translate within each
annotator's eligible set, then oldest-created -- `ee18987` the agreement
report, `44b23e2`/`7f3be6b` export (jsonl/tsv/parquet, adjudication-aware),
`9109ba2` the FastAPI/Jinja2/htmx app, `8d9beab` the CLI) before this task
(Task 8) added the fixture, e2e test, and runbook proving the STOP gate
itself. `tests/fixtures/annot/e2e_20_items.jsonl` has 20 lines: 17 ordinary
`mn_Cyrl` source-only entries (each line is a single segment for
annotators to translate, not a src/tgt translation pair) across 4
`doc_id`s, plus 3 `mn_Mong` lines -- both scripts built from real
codepoints via `chr()` in the generating script (not a literal non-ASCII
byte in the shell command), and written with `ensure_ascii=True` so the
committed file holds `\uXXXX` escapes for both scripts rather than raw
UTF-8 bytes. (Until the D2 whole-branch review's Findings 4 and 12 -- see
the paragraph above -- the `mn_Cyrl` lines were actually
Latin-transliterated ["galig"] text mislabeled as Cyrillic, a violation of
CLAUDE.md's permanent rule against it, and the traditional-script line
held raw UTF-8 bytes instead of `\uXXXX` escapes; both are now fixed.)
`tests/unit/test_annotate_e2e.py` (one test,
`test_20_item_fixture_fully_annotated_and_exported`) imports all 20 items,
round-robins three named annotators (`alice`, `bob`, `carol`) through
`FastAPI TestClient` calls to `/annotator/<name>/next` and the matching
`/task/<id>/{translate,review,adjudicate}` POST, asserts
`SELECT COUNT(*) FROM tasks WHERE status != 'done'` is 0, then calls
`export_jsonl` and asserts it returns 20. This test passed on its first run
with no changes needed to any Task 1-7 file: `uv run pytest
tests/unit/test_annotate_e2e.py -v` -> `1 passed in 2.64s`; there was no RED
phase to report because it is an integration test over already-implemented,
already-unit-tested code, not new production code written test-first. A
separate, uncommitted instrumented rerun of the identical round-robin logic
(same file, with added per-kind counters, run manually and discarded) showed
the loop finishes in 63 rounds with 40 translate and 20 review tasks done
and **zero adjudicate tasks ever created** -- every submitted translation in
this test is of the form `"work by <name>"` and every review score is fixed
at 95, so chrF between any two translations stays above 40 and the score
never drops below 70, so `db.submit_review`'s adjudicate trigger
(chrF < 40 or score < 70) is never hit for this fixture. State this
precisely: the committed e2e test proves the translate+review half of the
lifecycle end-to-end, plus export of all 20 items; it does not exercise the
`adjudicate` task kind or `adjudicate_task.html`, which still has no direct
test coverage of its own, consistent with Task 6's reviewer note on that
template. `rinchen annotate serve` itself (not just the `TestClient`-wrapped
app) was separately checked live: a scratch DB was initialized, `uv run
rinchen annotate serve --db <scratch> --port 8731` was started as a real
background process, and `curl` got HTTP 200 from both `/healthz`
(`{"status":"ok"}`) and `/` (the real `index.html`, title "Rinchen
Annotate"); the process was then killed by PID and the scratch DB deleted --
neither is part of the repo or this commit. `docs/runbooks/02_human_testsets.md`
was written verbatim per the task-8 brief; its CLI walkthrough
(`init`/`import`/`serve`/`export`/`agreement` with `--db`/`--annotators`/
`--file`/`--set`/`--format`/`--out`) and its quoted Krippendorff's-alpha
"N/A" message text were cross-checked against `src/rinchen/cli/annotate.py`
and match exactly. Full gate, run on win_amd64, 2026-11-02: `uv run ruff
format --check .` -- 92 files already formatted; `uv run ruff check .` --
all checks passed; `uv run mypy src` -- no issues found in 50 source files;
`uv run pytest -q -m "not network"` -- **150 passed, 6 deselected** (run both
via `just gate` and directly); `uv run pytest -q` (including the 6
network-marked tests, pre-existing and unrelated to D2) -- **156 passed**.
No limitation beyond the one stated above (`adjudicate_task.html`
coverage) is newly introduced by this task; the Krippendorff's-alpha "N/A"
behavior and the kind-priority queue-ordering behavior are pre-existing,
already documented in the runbook's Operational notes. (This last claim
was not actually accurate at the time -- the runbook's note described
simple oldest-first ordering across all kinds, not the real
kind-priority-then-oldest-created mechanism, and didn't mention the
3-annotator minimum; both are fixed by the D2 whole-branch review's
Finding 6, see the paragraph above, and the claim is accurate now.)
[EVIDENCE:
`docs/superpowers/plans/2026-11-02-d2-human-annotation.md`,
`docs/superpowers/specs/2026-11-02-d2-human-annotation-design.md`,
`docs/runbooks/02_human_testsets.md`, `tests/unit/test_annotate_e2e.py`,
`tests/fixtures/annot/e2e_20_items.jsonl`, commits `3f94ca9` through
`8d9beab` above, live `just gate` / `uv run pytest -q` / `rinchen annotate
serve` + `curl` runs, 2026-11-02.]

Repo scaffold is done: uv workspace, directory tree, CLAUDE.md, STATE.md,
DECISIONS.md, .gitignore, and a real (not stub) CI workflow
(`.github/workflows/ci.yml`) -- see the "CI step sequence" paragraph below
for what it actually runs; confirmed green on CI as of 2026-11-02 (see
the D0/D1 paragraph above). `pyonmttok` (built from source, see
below) is installed and verified on Python 3.14 via its smoke test on both
win_amd64 and manylinux_2_28_x86_64. `uv sync --dev` now succeeds for the
first time, installing all 145 pinned packages as of the fasttext-wheel fix
below (including `eole`, `torch`, `ctranslate2`, `transformers`) into the
real `.venv` on Python 3.14.7; the current `uv.lock` (after later
dependency additions, e.g. `types-PyYAML`) resolves 187 packages, confirmed
by counting `uv.lock` package entries on 2026-10-30.

P0.1 (environment probe) is complete. Full results: `reports/phase_0_env_probe.md`
(generated by `scripts/probe_env.py`, manifest in
`manifests/phase0_env_probe.json`). All 30 direct dependencies show
wheel-tag status "ok" on Python 3.14.7 (an earlier run falsely flagged
`warcio` as sdist-only; that was a classifier bug, now fixed and re-verified
— `warcio`'s real tag is the compressed `py2.py3-none-any`, which does carry
a wheel).

The current `.github/workflows/ci.yml`'s full step sequence --
`uv sync --frozen --group dev`, then `just gate` (ruff format --check,
ruff check, mypy, `pytest -m "not network" -q`), then `just doctor`
(`uv run rinchen doctor`, added to close the gap where CI never actually
ran the doctor checks), then `just tiny` (an echo stub until D15) -- has
been run locally on win_amd64 (`just gate` green, `rinchen doctor` 6 OK +
1 SKIP, `just tiny` prints its stub message and exits 0) and, as of
2026-11-02, on CI's `ubuntu-24.04` runner (run `41447258653`, all three
steps green -- see the D0/D1 paragraph above). On Linux, besides that CI
run, only the manylinux_2_28 pyonmttok build/smoke test has been run
directly; no interactive Linux `uv sync` + full gate session is recorded.

Known blockers for Phase 0 to resolve (see DECISIONS.md for the vLLM entry):

- Resolved: `pyonmttok` has no cp314 wheel upstream, so it was built from
  source — `OpenNMT/Tokenizer` tag `v1.38.1`
  (commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`), ICU 73.2, pybind11
  3.1.0 — for both win_amd64 and manylinux_2_28_x86_64. Both wheels are
  committed under `vendor/wheels/`, recorded with build provenance and
  sha256 in `manifests/pyonmttok_build.json`, pass the tokenize/detokenize
  round-trip smoke test, and are wired in via `[tool.uv.sources]` path
  overrides keyed on `sys_platform`. Closing this also required two more
  changes beyond the source override itself: adding `pyonmttok==1.38.1`
  directly to `[project.dependencies]` (`[tool.uv.sources]` path overrides
  only apply to direct project/workspace dependencies, not to a package
  that's only pulled in transitively via `eole`, so without this the
  override was silently ignored and `uv lock` failed with the exact same
  "no cp314 wheel" error as before), and adding
  `[tool.uv] environments = ["sys_platform == 'win32'", "sys_platform ==
  'linux'"]` (by default `uv lock` resolves universally across all
  platforms, and the project has no macOS target anywhere in its docs or
  plan — dev machine is Windows, CI/GPU box is Linux — so without this
  restriction `uv lock` tried to also solve a split for other platforms
  and failed there instead). With all three changes in place, `uv lock`
  succeeds on this machine as of this change, for the first time. [EVIDENCE:
  `docs/superpowers/plans/2026-10-29-pyonmttok-cp314-build.md`,
  `manifests/pyonmttok_build.json`, 2026-10-29.] This does not by itself mean
  D0 is unblocked or that `eole`'s doctor checks (`D-EOLE-PY`, `D-EOLE-CLI`)
  will pass: the first real `uv sync` attempt against this lockfile (run
  while verifying this fix) surfaced a new, separate, previously-unreached
  blocker — see the next bullet.
- Resolved: `fasttext-wheel==0.9.2` has no cp314 wheel and fails to build
  from source (confirmed: MSVC lacks `ssize_t`, gcc14 needs `<cstdint>`, and
  its Python wrapper uses NumPy-1-only `copy=False`). Rather than building
  the archived upstream C++ package from source, a first-party pure-Python
  shim (`vendor/fasttext-wheel-shim/`) re-exports the already-pinned
  `fasttext-new==0.1.0` under the `fasttext` name. `eole` uses `fasttext`
  only in its optional `clean` transform's language-identification step
  (`eole/transforms/clean.py`), which this shim was verified against
  directly: it correctly loads the real `lid.176.ftz` model and identifies
  both Mongolian and English test strings. [EVIDENCE:
  `docs/superpowers/plans/2026-10-29-fasttext-wheel-shim.md`,
  `tests/test_fasttext_wheel_shim.py`, 2026-10-29.] `uv sync --dev` succeeds
  for the first time on this project as of this change; `uv run pytest -q`
  and `uv run ruff check` both run against the real synced environment.
  Accepted tradeoff: `fasttext-new` is a small, alpha-status third-party
  package, not from Meta and not yet widely adopted.
- Resolved: the first real `uv run pytest -q` / `uv run ruff check` runs
  against the synced `.venv` surfaced two non-build hygiene findings, both
  now fixed. (1) `tests/test_vendor_wheel_integrity.py` hardcoded
  `manifests/pyonmttok_build.json` and a `uv.lock` package named
  `"pyonmttok"`, so the committed fasttext-wheel shim wheel failed both of
  its checks (2 failures). The test now reads every `manifests/*_build.json`.
  It requires each `vendor/wheels/*.whl` to be claimed by exactly one
  manifest build entry with a matching sha256, and every manifest build entry
  to point at a committed `vendor/wheels/` file. It matches `uv.lock` entries
  by wheel path and filename instead of by package name, so a future vendored
  wheel with its own `manifests/<name>_build.json` is covered without editing
  the test. (2) `ruff`'s `N999` on
  `vendor/fasttext-wheel-shim/src/fasttext/FastText.py` is suppressed by a
  `[tool.ruff.lint.per-file-ignores]` entry in `pyproject.toml`, scoped to
  that one path and that one rule; see DECISIONS.md for why this was chosen
  over a rename or an in-file `# noqa`. This is the project's first
  `[tool.ruff]` table. Diffing `ruff check --show-settings` before and after
  confirms it changes nothing else: same rule set, same 3.14 target version.
  [EVIDENCE: live `uv lock --check`, `uv sync --dev --locked`,
  `uv run ruff check .`, `uv run mypy src`, `uv run pytest -q` (40 passed)
  on win_amd64, 2026-10-29. A mutation check of the generalized test against
  a temp copy of the vendor wheels, manifests, and `uv.lock` covered a
  corrupted manifest or lock hash, a rebuilt wheel, an unrecorded wheel, a
  deleted wheel, and a duplicate manifest claim; each case failed exactly the
  expected assertion.]
- A generator model (vLLM, or an alternative) for Phase 6 synthetic data has
  been deliberately left out of the main workspace; Phase 0 or Phase 6 must
  decide between a separate non-workspace uv project and the llama.cpp path.
