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
- Do not include vLLM in the main uv workspace. [EVIDENCE: vllm==0.32.0 requires
  torch==2.13.0 and transformers<5.20.0, which conflicts with the base
  torch==2.14.1/transformers==5.21.0 pins; uv shares one lockfile across all
  dependency groups, so an in-workspace group cannot isolate this]. If a
  generator model is needed (Phase 6), use a separate non-workspace uv project
  with its own lockfile, or the llama.cpp path the research doc calls
  3.14-safe.
- Track manifests/*.parquet in git rather than ignoring it. [DESIGN: the hard
  constraints require recording resolved versions and checkpoint hashes in
  manifests/, and later phases (release.json) depend on those hashes being
  present in a fresh clone].
- **[SUPERSEDED 2026-10-29 — see the "Build pyonmttok from source for cp314"
  entry below, which resolves this directly; the `eole --no-deps` fallback
  described below was never attempted. Kept as-is, unedited, as the
  historical record of the original finding.]** Confirmed: pyonmttok has no
  cp314 wheel as of 2026-10-28. [EVIDENCE: Phase 0
  P0.1 probe, see reports/phase_0_env_probe.md]. `uv lock` fails outright
  because `eole==0.6.4` requires `pyonmttok>=1.38.1`, and `pyonmttok` only
  ships wheels for cp310/cp311/cp312. Fallback to try first: `eole --no-deps`
  plus only the dependencies `import eole.bin.main` and `eole train --help`
  actually need (see STATE.md). Never work around this by running any part
  of the project on cp310/cp311/cp312 — the hard constraints forbid any
  fallback interpreter.
- All 30 direct dependencies resolve to wheel-tag status "ok" on Python
  3.14.7, including `warcio`. [EVIDENCE: Phase 0 P0.1 probe, 2026-10-29, see
  reports/phase_0_env_probe.md]. An earlier run of the probe had a
  wheel-tag-classifier bug that didn't recognize compressed tags like
  `py2.py3` (warcio's real tag), so it was wrongly reported as sdist-only
  with no wheel at all; the classifier is fixed and this bullet replaces
  that earlier, incorrect finding.

## 2026-10-29: Build pyonmttok from source for cp314

[EVIDENCE: live build, `manifests/pyonmttok_build.json`, tag `v1.38.1`
commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`.] No cp314 wheel or sdist
exists for `pyonmttok` 1.38.1 on PyPI (only cp310/cp311/cp312). Built it
from source for both win_amd64 and manylinux_2_28_x86_64 using ICU 73.2 and
pybind11 3.1.0 (pyonmttok's own `pyproject.toml` requires `pybind11>=2.10.4`
with no upper bound; pybind11 3.1.0 explicitly classifies Python 3.14
support). Wheels committed under `vendor/wheels/`, wired in via
`[tool.uv.sources]` path overrides. Wiring the override in required two
more `pyproject.toml` changes, confirmed by direct experiment rather than
assumed: (1) `pyonmttok==1.38.1` added to `[project.dependencies]` directly
— `[tool.uv.sources]` path overrides only apply to a project's direct or
workspace dependencies, and `pyonmttok` is otherwise only a transitive
dependency of `eole`, so without this the override had no effect and
`uv lock` failed identically to before; (2) `[tool.uv] environments =
["sys_platform == 'win32'", "sys_platform == 'linux'"]` added — `uv lock`
resolves universally across all platforms by default, and this project
targets only Windows (dev) and Linux (CI/GPU) with no macOS target
anywhere in its docs or plan, so without restricting the locked
environments `uv lock` tried to also solve for other platforms and failed
there since no wheel or source exists for them. This unblocks `uv lock`,
which now succeeds for the first time. It does not by itself unblock D0:
the first real `uv sync` against this lockfile surfaced a new, separate,
previously-unreached blocker in `fasttext-wheel==0.9.2` (a transitive
`eole` dependency, distinct from the already-pinned `fasttext-new`
package) — no cp314 wheel upstream, and its C++ extension fails to build
from source under MSVC (`fasttext_pybind.cc(171): error C2065: 'ssize_t':
undeclared identifier`, cascading into pybind11 `C2672` overload-resolution
errors). Same category of problem as this pyonmttok gap (native extension,
no cp314 wheel), but not yet investigated or fixed; see STATE.md's current
next step. **[Resolved later on 2026-10-29: see the "fasttext-wheel cp314
shim via fasttext-new" entry below.]**

**Note on the direct `pyonmttok` dependency vs. the master plan.** The
master plan (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`) lists
`pyonmttok` under "Do NOT depend on: ... (we replace them)" — that line is
about Rinchen's own text-processing pipeline, which must not import
`pyonmttok` directly. The `pyonmttok==1.38.1` entry added to
`[project.dependencies]` above is unrelated to that intent: it exists
*only* as a dependency-resolution mechanism, because `uv`'s
`[tool.uv.sources]` path overrides apply only to a project's direct or
workspace dependencies, not to a package that is otherwise pulled in
transitively via `eole`. Without the direct pin, the locally-built wheel
override is silently ignored and `uv lock` fails exactly as before. Adding
this pin does not reverse or weaken the master plan's "we replace them"
design decision — Rinchen's own code still may not `import pyonmttok`
directly; `eole` is the only thing in this workspace allowed to depend on
it.

## 2026-10-29: fasttext-wheel cp314 shim via fasttext-new

[EVIDENCE: live verification against the real `lid.176.ftz` model,
`tests/test_fasttext_wheel_shim.py`, 2026-10-29.] `fasttext-wheel==0.9.2`
(a transitive, unversioned `eole` dependency) has no cp314 wheel. Building
it from source was investigated and validated as feasible (three confirmed
C++/Python portability bugs, all fixed and build-tested live on both
Windows and Linux during brainstorming), but not applied. [DESIGN: chose
instead to ship a first-party pure-Python shim package
(`vendor/fasttext-wheel-shim/`, distributed as `fasttext-wheel` to satisfy
`eole`'s dependency name) that re-exports the already-pinned, maintained,
cp314-native `fasttext-new==0.1.0`. `eole` only uses `fasttext` for
`load_model`/`predict` in its optional language-ID `clean` transform, which
`fasttext-new` is independently documented and here independently verified
to handle correctly for the `lid.176` models eole actually downloads. This
avoids native compilation entirely and needs no per-platform maintenance,
at the accepted cost of depending on a smaller, alpha-status third-party
package instead of the archived original.]

Wiring this in (`pyproject.toml`'s `[project.dependencies]` plus a
`[tool.uv.sources]` path override to the committed
`vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl`, mirroring the
pyonmttok pattern above) let `uv lock` and then `uv sync --dev` succeed for
the first time ever on this project — the actual milestone both the
pyonmttok and fasttext-wheel cp314 gaps existed to reach. [EVIDENCE: live
`uv lock` / `uv sync --dev` run, 2026-10-29; confirmed
`vendor/fasttext-wheel-shim` was not pulled in as a `uv` workspace member —
`uv.lock`'s `fasttext-wheel` package entry has `source = { path =
"vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl" }`, an ordinary path
dependency on the built wheel file, not an editable/workspace source, and
`pyproject.toml` has no `[tool.uv.workspace]` table at all.]
`tests/test_fasttext_wheel_shim.py`'s network test (real download of
`lid.176.ftz`, real `predict()` calls) passes against the real synced
environment.

The first real `uv run pytest -q` / `uv run ruff check` against this synced
environment also surfaced two non-build hygiene findings, both since fixed
(evidence in STATE.md). [DESIGN: `tests/test_vendor_wheel_integrity.py` was
generalized to read every `manifests/<name>_build.json` and to match
`uv.lock` entries by wheel path/filename rather than by a hardcoded
`pyonmttok` package name. Each vendored package keeps its own build
manifest (no shared or renamed manifest), and a future vendored wheel needs
no test change. `ruff`'s `N999` on the shim's `FastText.py` is suppressed by
a `[tool.ruff.lint.per-file-ignores]` entry in `pyproject.toml`, scoped to
that single path and rule. Renaming the module was rejected because its
name deliberately mirrors upstream's `fasttext.FastText` submodule. An
in-file `# noqa` was rejected because the committed wheel is byte-identical
to the shim source: editing the source would either leave it out of sync
with the wheel, or force a rebuild and a re-hash of the wheel, its manifest,
and `uv.lock`.]

## 2026-10-30: D0 wrap-up decisions and deliberately deferred scope

A few D0 design decisions were made along the way but never recorded here;
collected in one place rather than split across several near-duplicate
entries.

[DESIGN: `ruff`'s `extend-exclude = ["docs/"]` (in `pyproject.toml`'s
`[tool.ruff]` table).] `docs/` holds historical design/plan prose
(`docs/research/`, `docs/superpowers/`) with illustrative code fences, not
real importable source. Without the exclusion, ruff's Markdown support
lints/reformats those example snippets as if they were project code,
producing noise unrelated to the actual codebase.

[DESIGN: `rinchen doctor`'s `D-WHEELS` check (`src/rinchen/cli/doctor.py`)
verifies each `[project.dependencies]` entry via
`importlib.metadata.version(name)` against installed distribution metadata,
rather than `import`-ing each package by its module name.] A PyPI
distribution name does not reliably map to its import name (e.g.
`PyYAML` -> `yaml`, `beautifulsoup4` -> `bs4`, `fasttext-wheel` -> `fasttext`,
`scikit-learn` -> `sklearn`), so a generic "import every dependency by name"
check would need a hand-maintained name-mapping table that silently drifts
as dependencies are added. Checking installed distribution metadata by its
PyPI name is robust to this without any such table, at the accepted cost of
not actually exercising each package's import machinery (separate,
targeted checks like `D-EOLE-PY` and `D-CT2-IMPORT` cover the packages
where a real `import` matters).

[DESIGN: no separate `rinchen doctor --gpu` flag.] The master plan's
wording suggested a GPU-specific doctor flag; D0 instead always runs the
GPU check (`D-TORCH-CUDA`) as part of the normal `rinchen doctor`
invocation, reporting `skip` (not `fail`) when no GPU is present, rather
than adding a separate flag that would need to be remembered and invoked
on purpose on GPU machines. This keeps one invocation path for the whole
doctor suite.

[DESIGN: deliberately deferred CI gaps in `.github/workflows/ci.yml`, not
attempted as part of this fix round.] The current `ci.yml` is a real,
working gate (`uv sync --frozen --group dev` -> `just gate` -> `just
doctor` -> `just tiny`, with a coverage-artifact upload step), but it is
not yet the master plan's full target CI config. Known, deliberate gaps,
each its own future scope:
- No `uv` cache step (e.g. `astral-sh/setup-uv`'s built-in caching) --
  every CI run currently does a full cold dependency resolve/download.
- No `timeout-minutes` set on the `gate` job -- a hang would run until
  GitHub Actions' own default job timeout.
- No nightly/scheduled (`on: schedule`) trigger -- CI only runs on push to
  `main` and on pull requests.
- No `report.json` generation -- the master plan's eventual per-run
  machine-readable report artifact is not produced by this workflow.
- The `Upload coverage` step uploads `coverage.xml`, but no step ever
  passes `--cov` to pytest, so `coverage.xml` is never produced; the
  upload step's `if-no-files-found: ignore` means it silently no-ops on
  every run rather than failing, which could give a false impression that
  coverage is being collected.

None of these should be mistaken for the master plan's full target CI
config; each is real, separately-scoped work for a later milestone to pick
up deliberately, not an oversight to silently work around.

[DESIGN: the config-loader -> manifest integration seam
(`config_sha256`/`code_version` inputs) is deferred, not built in D0.] The
design spec asked the config loader to be able to dump the resolved
config plus its sha256, and for `src/rinchen/__init__.py` to hold a real
`__version__` string -- neither was built in D0, so nothing in the
package today actually produces the `config_sha256`/`code_version` values
that `Manifest`/`should_skip` (`src/rinchen/manifest/`) require as inputs.
`src/rinchen/__init__.py` remains an empty file as of this entry. This is
real scope, not a quick fix: designing the dump mechanism properly (what
gets canonicalized before hashing, where the dump is written, how
`code_version` is derived) is D1+ work. Whichever D1+ task first needs
`config_sha256` for a real manifest should build this seam then, rather
than assuming it already exists.

## 2026-10-30: D1 candidate parallel-corpus QA — 6 of 7 rejected

[EVIDENCE: live, 100-random-sample pair-by-pair review of each candidate,
2026-10-30, conducted via parallel forked review agents (each
independently downloading and reading 100 real EN-MN pairs), under a
zero-tolerance bar the user set explicitly ("if there is the slightest
issue, ignore that dataset").] Before building D1's acquisition machinery,
7 candidate real-world parallel corpora were vetted: 5 OPUS corpora the
user pre-selected (TED2020, QED, NeuLab-TedTalks, GNOME, Tatoeba) plus two
HuggingFace candidates (`ugshanyu/1millionMongolianSentence`,
`davidstap/NTREX`). Verdicts:

- **GNOME: FAIL.** 34 of 100 sampled pairs had issues: truncated verbs
  (e.g. "Remove the queen of hearts." → "Бундангийн хатан", dropping the
  verb entirely), leaked untranslated English developer comments sitting
  in the Mongolian column, clear wrong-meaning mistranslations (e.g.
  "Single card" → "Гурван карт", "three cards"), and at least one
  misaligned/reused string pairing.
- **NeuLab-TedTalks: FAIL.** ~19 of 100 pairs showed systematic
  truncation — the Mongolian side drops clauses or trails off mid-sentence
  relative to the English, a structural artifact of TED subtitle-segment
  alignment, not isolated noise.
- **QED: FAIL.** Galig/Latin-script Mongolian present in at least 9 of 100
  pairs (e.g. `Tegeed ter ShinHwa group-n uv zalgamjlagch yumu?`) — a
  direct hard-constraint violation (CLAUDE.md: "NEVER use galig/Latin
  informal Mongolian") — plus severe misalignment (one pair's English
  field contained Mongolian text verbatim; several pairs' Mongolian text
  was entirely unrelated to the English).
- **TED2020: FAIL.** 19+ distinct issues: 2 fully empty pairs, several
  truncated/incomplete Mongolian sentences, wrong-sense mistranslations
  (e.g. "race" as in ethnicity mistranslated as "уралдаан", a running
  race), content inserted from neighboring subtitle segments with no
  English counterpart at all, a misspelled real proper noun ("Aga Khan"
  → "Агра Хан"), and one stray Cyrillic-homoglyph inconsistency.
- **`ugshanyu/1millionMongolianSentence`: FAIL.** Character-substitution
  defects (Latin `v` substituted for Cyrillic `ү`; Ukrainian `є` substituted
  for Mongolian `ө`), plus real content/name mismatches (a named minister
  in English replaced by two different, unrelated names in Mongolian;
  garbled English proper nouns like "Reikakkal" for Reykjavik) consistent
  with the English side being a poor machine back-translation rather than
  an independent human translation, not a genuine parallel corpus.
- **`davidstap/NTREX`: FAIL.** ~10% defect rate with meaning-inverting
  mistranslations (e.g. "shark" → "аварга засаг", "great ruler"; the idiom
  "full bore" rendered as nonsense; Iowa's agricultural-export status
  misattributed to China). Note: NTREX's Mongolian config is internally
  labeled `mon_Mong`, which was initially (incorrectly) flagged during
  brainstorming as traditional Mongol script — directly downloading and
  reading the real parquet content confirmed it is genuine Khalkha
  Cyrillic; the label is just NTREX's own naming quirk, not a script
  issue. The dataset was excluded for translation-quality reasons, not a
  script violation.
- **Tatoeba: borderline, user chose to exclude anyway.** The only
  candidate without structural/alignment problems or script violations —
  ~10 minor issues in 100 pairs (typos, one case-marking inconsistency, a
  dropped proper name, one imprecise idiom, one geography-fact sentence
  flagged for sensitivity-review rather than being an actual error).

[DESIGN: rather than keep searching for a passing candidate, the user
decided Rinchen will build its own standard dataset from scratch.
`configs/data/sources.yaml` ships with zero real production entries as of
D1 — D1's own scope (the acquisition machinery: `http.py`, `opus.py`,
`hf.py`, `commoncrawl.py`, `warc.py`, `registry.py`) is unaffected by this,
since it only needs a schema-valid registry and offline test fixtures to
prove the machinery works, not a populated catalog. Real corpus curation
is a separate, future, ongoing effort — any future candidate should be
vetted with this same 100-sample zero-tolerance process before being added
to `sources.yaml`, not assumed safe from a smaller spot-check (the
original 5-line-per-corpus spot-check done earlier in this same
brainstorm looked clean for all 7 candidates and would have wrongly
greenlit 6 of them — the 100-sample pass is what actually caught the real
defect rates).]

## 2026-10-30: `expected_sha` narrowed to single-file registry kinds

D1's final whole-branch review found that `expected_sha` was accepted by
the `SourceEntry` schema for every registry kind but silently never
checked for `opus`/`hf`/`commoncrawl` in `registry.fetch_source` —
a config author could set it on those kinds and it would do nothing.

[DESIGN: `expected_sha` verification now applies to `http`, `local`, and
`opus` (each produces exactly one output file, so a single sha256 is
meaningful). `HfSourceEntry` and `CommonCrawlSourceEntry` reject
`expected_sha` at pydantic validation time with a `ConfigError`, since
`hf.fetch` writes multiple Parquet shards and `commoncrawl.fetch_records`
writes multiple WARC records — there is no single file for one sha256 to
describe. A future milestone that needs integrity-checking for
multi-file fetches should design a per-shard/per-record manifest rather
than overload this single-hash field.]

## 2026-11-02: D2 human-annotation design decisions

Three D2 (`src/rinchen/annotate/`) design decisions worth recording for
future reference, collected here rather than split across near-duplicate
entries.

[DESIGN: Krippendorff's alpha (`agreement.py`'s `krippendorff_alpha`) is
hand-implemented rather than adding the `krippendorff` PyPI package as a
new dependency.] This follows the master plan's "Build-ourselves (no
dependency)" principle (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`,
its `<toolchain>` section) — that line doesn't name this statistic specifically,
but the same build-it-ourselves-rather-than-add-a-dependency reasoning
applies: D2's agreement reporting needs exactly one well-understood
formula, not a general-purpose library.

[DESIGN: `db.connect` passes `sqlite3.connect(..., check_same_thread=False)`
(Finding 1 of the D2 whole-branch review).] `sqlite3.connect()` defaults to
`check_same_thread=True`, which raises `sqlite3.ProgrammingError` if a
connection is used or closed from a thread other than the one that created
it. Each request's connection is logically scoped to that one request, but
FastAPI can still run a sync route's dependency-enter (the `yield` in
`get_db`, in `routes.py`) and dependency-exit (the `finally: conn.close()`)
on different threadpool worker threads for the same logical request —
`check_same_thread=False` is required for that reason, not because
connections are actually shared across requests. [EVIDENCE: live
reproduction confirmed ~253/300 request failures with the fix reverted and
0/300 with it in place, firing a burst of concurrent `GET /` requests at a
single `with TestClient(app) as client:` instance via a `ThreadPoolExecutor`
— `tests/unit/test_annotate_app.py::test_concurrent_requests_do_not_return_500`.]

[DESIGN: `claim_next_task`'s priority ordering is kind-priority
(`adjudicate` > `review` > `translate`, oldest-created within a kind),
with re-serving an annotator's own already-`assigned` task taking priority
over all of that — not the originally-planned pure oldest-created-wins
ordering across all kinds.] Found necessary via TDD during D2
implementation: adjudication and review work is time-sensitive and
shouldn't be starved by a backlog of older `translate` tasks on unrelated
items, and an annotator with unfinished in-progress work must always get
that same task back on a subsequent claim (Finding 2) rather than a
different, newly-prioritized one. See `claim_next_task`'s docstring in
`src/rinchen/annotate/db.py` for the full eligibility rules.

