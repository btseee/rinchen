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

## 2026-11-06: D3 normalization and SentencePiece design decisions

D3 (`src/rinchen/{lang,split,spm}/`, `src/rinchen/text/normalize.py`,
`src/rinchen/cli/spm.py`) filled gaps in, or departed from, the approved spec
(`docs/superpowers/specs/`) in the ways below, collected in one entry. "Builder's
call" means the spec was silent and the implementer chose. Items marked FLAG
need a human decision later; the known limitations are recorded so nobody
rediscovers them as surprises. Procedures are in
`docs/runbooks/03_tokenizer.md`.

[DESIGN: The language registry (`configs/languages.yaml`) has 10 entries, not
the spec's "9", and each entry carries a `scripts` list, not a single
`script`.] The 10 are the 8 partner languages (eng_Latn, rus_Cyrl, zho_Hans,
kor_Hang, jpn_Jpan, deu_Latn, kaz_Cyrl, tur_Latn) plus khk_Cyrl and mn_Mong.
mn_Mong is an output variant of Mongolian, not a partner, so there are
8 x 2 = 16 translation directions but 10 direction tags. `scripts` is a list
because jpn_Jpan spans Han, Hiragana and Katakana. The reserved SentencePiece
symbols are 10 tags + 8 structural tags + 100 placeholders = 118, and
`<2mn_Mong>` comes from the registry like the other tags instead of being a
separate hand-added symbol (builder's call). If the count or order of the
registry is wrong, every SentencePiece model and everything trained on it must
be retrained, which is cheap before D8 and expensive after.

[DESIGN: `normalize` runs NFC, zero-width cleanup, whitespace, quotes,
Cyrillic homoglyph repair, Fita fold, NFC again, in that order (builder's
call).] The spec lists zero-width cleanup last. That breaks the spec's own
idempotence property for `a ZWSP b` (stripping after whitespace collapsing
leaves two spaces) and lets the fold and homoglyph steps create composable
pairs that only the closing NFC pass repairs. Uppercase Latin lookalikes
(A E O P C X Y) are repaired as well as lowercase (builder's call). The
zero-width set is U+200B, U+200C, U+200D plus U+2060 and U+FEFF (builder's
call). The Mongolian free variation selectors and vowel separator
U+180B..U+180F are stripped outside mn_Mong and kept inside it. The Fita fold
table is exactly U+0472 to U+04E8 and U+0473 to U+04E9; Kazakh U+04B0/U+04B1
and Greek theta are never touched. `normalize` is meant for raw text, run
before any reserved symbol is inserted, and reserved symbols must be
whitespace-separated from text: the homoglyph step tokenises on spaces, so a
tag glued to a Cyrillic word is rewritten (`<sep>` becomes `<s`, U+0435,
U+0440, `>`). That contract is stated in the docstring and the runbook and
pinned in `tests/unit/test_text_normalize.py`; no code guard, because one
would couple `text/` to `spm/` and change the pinned pipeline. [EVIDENCE:
`tests/fixtures/golden/normalize_200.tsv`, 200 lines derived by hand from the
rule table, never from `normalize` output; a reviewer independently
re-derived all 200 with 0 disagreements. Hypothesis idempotence properties in
`tests/property/test_normalize_properties.py`.] Cost if wrong: a golden-file
update plus a reprocess of any corpus normalized so far.

[DESIGN: FLAG: ZWNJ and ZWJ (U+200C, U+200D) are stripped in mn_Mong too
(spec-literal).] Traditional Mongolian script shaping can rely on both
characters, so stripping them may change how some words render or which
glyph form a model sees. Revisit with a Mongolian shaping-aware reviewer
before any large mn_Mong corpus is normalized. The only cost to change is a
golden-file update and a normalization-rule edit; once corpora are built it
is a reprocess. Concrete gate: before any real mn_Mong corpus is normalized,
count the U+200C and U+200D occurrences in the acquired text and their left
and right neighbours. If ZWNJ is frequently used as the suffix separator
(older web text), map ZWNJ to NNBSP (U+202F) for mn_Mong instead of stripping
it; ZWJ-forced medial or isolated forms in abbreviations and glossaries are
the other lossy case. A mn_Mong target that contains these characters would
train D5 and D8 on the stripped forms.

[DESIGN: FLAG: `normalize` repairs only some look-alikes of the Mongolian
O-umlaut and U-umlaut.] Only Cyrillic Fita (U+0472/U+0473) is folded to
U+04E8/U+04E9. Latin O-stroke (U+00D8/U+00F8), the barred-O letters U+019F
and U+0275 and Greek theta (U+0398/U+03B8) are not folded and stay in the
text. An uppercase Latin `Y` inside a Cyrillic-majority token is repaired to
U+0423, as the pinned golden row for a Latin `Y` that starts a Cyrillic place
name expects, never to U+04AE, which is wrong for a word that should start
with U+04AE. A Mongolian-literate reviewer should say which of these matter;
D4 should count these code points, and U+200C/U+200D, in real khk_Cyrl text
before the vocabulary is frozen. [EVIDENCE: live `normalize` calls with each
code point after a Cyrillic anchor, 2026-11-06.]

[DESIGN: KNOWN LIMITATION of `normalize`: it is not idempotent for Hangul
jamo, a stripped zero-width character and trailing jamo inside a
Cyrillic-majority token that also holds a Latin lookalike.] The final NFC
merges the jamo and shifts the more-than-50%-Cyrillic gate on a second pass;
kor_Hang is in the registry, so the combination is expressible. It is
contrived, and no property test or golden line reaches it. It is not fixed
because the fix (an NFC pass after the zero-width step) changes the pinned
7-step pipeline above and its golden file.

[DESIGN: The splitter's abbreviation sets are module-level frozensets in
`src/rinchen/split/rules.py`; the master plan's `abbrev/*.txt` layout is not
used, per the approved spec.] Sentence-start rules: the Latin family accepts
any letter as a start (Hangul has no case); the Cyrillic family accepts an
uppercase Cyrillic letter, an opening quote or a dash. The single-uppercase-
initial exemption applies to eng, deu and the Cyrillic family only, not tur
or kor (builder's call); German also exempts all-digit ordinals. Spaced
multi-token abbreviations (`z. B.`, `d. h.`, `u. a.`, `i. d. R.`) and dotted
initials (`J.K.`) are handled by a compact-form window of up to 4
dot-terminated tokens, looking both ways (added after a review found German
`z. B.` splitting mid-phrase). Neighbours are consulted
only when whitespace follows the dot directly, because otherwise a closer
after the dot breaks the rejoin and idempotence contract. [EVIDENCE:
`tests/unit/test_split_rules.py`, `tests/property/test_split_properties.py`,
commit `28088da`.]

[DESIGN: FLAG: the Mongolian-specific and Kazakh-specific abbreviations in
`_CYRILLIC_ABBREVIATIONS` were confirmed by web usage evidence, not by an
orthographic authority.] They are the four Mongolian entries (glosses: "and so
on", "for example", "in other words", "before the common era") and the three
Kazakh entries ("and so on", "and the like", "before the common era"), in
Cyrillic spelling in `src/rinchen/split/rules.py`. They need native-speaker
review. A wrong entry costs a missed or a spurious sentence break around that
abbreviation; the fix is a one-line edit to the set.

[DESIGN: KNOWN LIMITATIONS of the splitter, all reproduced by reviewers; a
few re-checked while writing this entry.] `split` expects already-normalized
text: a zero-width character left after a terminator misplaces the boundary
(in mn_Mong the next part starts with it; in the Latin and Cyrillic families
it suppresses the split). Spaced abbreviations in brackets or after leading
punctuation still split (`(z. B. dies)` gives `(z.` and `B. dies)`).
Sentences ending in an initials chain, `U.S.` or `D.C.` followed by a capital
word do not split. The pronoun "I" is swallowed by the single-initial
exemption (`Neither did I. Then we left.` stays one part). The Latin start
rule is letters only, so a quote or a digit after a period does not start a
sentence; the Cyrillic rule excludes Latin uppercase and digits. In mn_Mong,
ASCII `?` and `!` split even without following whitespace (the approved
design specifies it), which cuts URL-like text such as a query string. A
terminator that follows closing characters is split off as a lone terminator
part only when those closing characters themselves follow an earlier
terminator: in the CJK families U+3002 U+300D U+FF01 gives `U+3002 U+300D` and
`U+FF01`, while U+4F60 U+597D U+300D U+FF01 stays one part. A leading
terminator run becomes a part of its own: U+3002 followed by U+4F60 U+597D
gives `U+3002` and `U+4F60 U+597D` (the same holds in mn_Mong, and in the
Latin and Cyrillic families when whitespace follows the run: `. Hello` gives
`.` and `Hello`, `.Hello` stays one part). A very long run of one terminator
at the end of the text, or otherwise not followed by a sentence start, was
quadratic in the first build (80,000 terminators took 2 to 6 s); the fix wave
lets a match start only at the first terminator of a run, and 60,000 now split
in 0.5 to 2.5 ms. The cost of the remaining cases is a mis-split or wrongly
merged sentence, which matters only if such text is common in the corpus.

[DESIGN: SentencePiece policy is fixed in code, not config.] `unigram`,
`byte_fallback=True`, `split_digits=True` and
`normalization_rule_name=identity` are constants in `src/rinchen/spm/train.py`,
because they are project-wide invariants, not per-profile choices, and
`spm check` verifies byte fallback and the symbols on a trained model. The
config carries `vocab_size`, `input_sentence_size`, `character_coverage`,
`temperature`, `seed`, `corpus_dir` and `output_prefix`; the
`input_sentence_size` values 100000, 10000000 and 20000000 are the builder's
call. Sampling uses one RNG stream per language,
`random.Random(f"{seed}-{lang}")` (independent of `PYTHONHASHSEED` and of
language order; adding a language still changes every quota, because the
weights are renormalised, so every uncapped language's sample is redrawn). A
language's quota is `floor(weight * budget)` capped at its size; unused quota
is not redistributed, so a sample can fall below the budget; if the whole
corpus fits the budget, everything is used. Cost if wrong: retrain, with the
budgets the first thing to revisit once real corpora exist.

[DESIGN: The default `add_dummy_prefix` is kept.] A tagged source therefore
encodes with a leading standalone U+2581 piece before the tag, and
`check_spm` defines "atomic" as `[symbol]` or `[U+2581, symbol]`. REVISIT
before D8 (Eole training) whether `add_dummy_prefix=False` is wanted: changing
it later means retraining the vocabulary and anything trained on it. The check
verifies symbol atomicity and byte fallback only, not digit splitting or
normalization, because the sentencepiece 0.2.1 Python API cannot tell
user-defined pieces from ordinary ones, so atomicity is tested on the bare
symbol. [EVIDENCE: live encode of a bare tag with the trained tiny model
gave the two pieces U+2581 and `<2khk_Cyrl>`, 2026-11-06.]

[DESIGN: `rinchen spm train` and `rinchen spm check` take `--config` and
`--model`, not `--profile`, and write no manifest.] SPM config names are not
profile names. The master plan says `train` prints a manifest path; D3 does
not meet that, following the D1 and D2 precedent of writing no manifest from
the CLI. The spm commands map `ConfigError` to exit 2, `DataError` to 3,
`UpstreamError` to 4 and any other `RinchenError` to 1, printing `error: ...`
on stderr; a `check` that reports `failed` exits 3. The mapping is
implemented locally in `cli/spm.py`; the D1 and D2 CLIs remain unmapped.
Cost if wrong: a model's provenance is not recorded automatically, so a later
milestone that consumes a model must record its hash itself.

[DESIGN: Scope is deliberately narrow: only `tiny_1k` is ever trained in D3.]
`student_64k.yaml` and `teacher_128k.yaml` are schema-validated YAML only;
their corpus directory (`data/clean/spm_corpus`) does not exist yet and no
model has been trained on real data. Their `input_sentence_size` budgets, the
memory the training step needs for them, and the sampling behaviour on real
size ratios are untested assumptions until a later milestone supplies a
corpus.

[DESIGN: The tiny fixture is synthetic pseudo-words in genuine scripts, not
real sentences.] Pseudo-words carry no meaning and no real-world text, so
they cannot hold transliterated or Inner Mongolian material, and
`tests/unit/test_spm_fixture.py` checks mechanically that each file is written
in its language's declared script. The committed files
(`tests/fixtures/spm/tiny_corpus/{khk_Cyrl,rus_Cyrl,eng_Latn,mn_Mong}.txt`,
400/200/100/100 lines, reproducible with `scripts/build_spm_fixture.py`)
train to exactly 1000 pieces with the 118 reserved symbols, under
SentencePiece's default hard vocabulary limit. A repetitive corpus of 30
distinct words fails that limit at 1000 ("Vocabulary size too high"), which
is why the fixture is varied. The mn_Mong lines use letters from the basic
block U+1820..U+1841 plus spaces and a closing U+1803. The fixture carries
only the digits 0, 2, 3, 4 and 5, so a digit it lacks (1 was checked) encodes
as a byte piece in the tiny model. [EVIDENCE:
`tests/unit/test_spm_fixture.py`; `SentencePieceProcessor.get_piece_size()`
returned 1000 for the model trained in the STOP-gate run, 2026-11-06.]

## 2026-11-06: D3 follow-up - the three open decisions, resolved by research

This entry resolves the ZWNJ/ZWJ FLAG to keep-stripping, with an audit gate
before D5 (the Mongolian-reader review the old FLAG asked for has not
happened), and resolves the `add_dummy_prefix` revisit-before-D8 note of the D3
entry above. It narrows the abbreviation FLAG of the D3 entry by a published
handbook source (the Kazakh punctuation rules: Syzdykova, handbook chapter,
2000, reprinted at bilim-all.kz), not by an authority: the Kazakh entries other
than `б.з.д` have support in that text, the Mongolian entries still rest on web
usage evidence, and an officially approved Kazakh orthography text was not
located. The third FLAG of the D3 entry, the O-umlaut look-alikes, is untouched
and stays a D4 gate. The D3 entry is left as written (append-only); where the
two differ, this entry is the current decision. The evidence, with its sources,
quotes and the reproduction harness, is in
`docs/research/2026-11-06-mongolian-controls-and-spm-tags.md`; procedures are
in `docs/runbooks/03_tokenizer.md`.

[DESIGN: 1. Target-language tags: keep the SentencePiece default
`add_dummy_prefix=True` and insert the `<2xx>` tag at TOKEN level after
SentencePiece.] Consequences for D8: one Eole corpus per direction (and per
bt/ft variant) with `transforms: [sentencepiece, prefix, ...]` and a
`src_prefix` such as `<2khk_Cyrl>` (a space-separated multi-token prefix, for
example `<2khk_Cyrl> <bt>`, is supported); the training data carries untagged
normalized text; the CTranslate2 wrapper prepends `[tag]` to `sp.encode(text,
out_type=str)` itself (CTranslate2 takes token lists); `tgt_file_prefix` stays
false (the tag is source-side). Keep all 118 symbols as `user_defined_symbols`:
the placeholders and the inline tags `<t> </t> <r> </r> <doc> <sep>` live
inside text and must decode back to their text, and MADLAD-400 also keeps its
tags user-defined. Strip or escape literal reserved-symbol strings from raw
text before encoding, so that user text cannot inject a `<2xx>` tag. Tests to
add when D8 builds it: the wrapper's tokens equal Eole's pipeline output for
the same input; no training source starts with `▁` followed by a tag piece; the
wrapper's source tokens begin with exactly the tag; the decode of an output has
no leading space; every reserved symbol round-trips through encode and decode.
No change to the D3 SentencePiece config or to `check_spm` is needed
(`check_spm` already accepts both `[symbol]` and `['▁', symbol]`). Untested:
running Eole per-direction corpora end to end with `eole train`; if
per-direction corpora prove impractical, a small custom Eole transform is the
fallback.

[EVIDENCE: Eole 0.6.4's `PrefixTransform` (`eole/transforms/misc.py`) asserts
that the example is already a token list ("PrefixTransform needs to be set
after the Tokenizer") and prepends `side_prefix.split(" ")` to it;
`eole/inputters/text_corpus.py` applies a corpus's transforms in the order
listed; the SentencePiece transform (`eole/transforms/tokenize.py`) encodes the
whole string with `sp_model.encode(string, out_type=str)`. A tag written into
the TEXT is therefore matched as a user-defined symbol and gets a standalone
`▁` before it (verified: `['▁', '<2khk_Cyrl>', ...]`); with `transforms:
[sentencepiece, prefix]` the tag becomes a token and no `▁` appears. The
released SentencePiece model files checked on 2026-11-06
(NLLB-200-distilled-600M, M2M-100-418M, mBART-large-50, MADLAD-400-3b-mt,
OPUS-MT en-mul and t5-small) all have `add_dummy_prefix=True`; their addresses
and sha256 prefixes are in the research note (the files can change upstream, so
the prefixes are a snapshot). NLLB, M2M-100 and mBART-50 add the language tag
as an id or token after SentencePiece. The OPUS-MT file contains no `>>xxx<<`
tag: the Hugging Face MarianTokenizer strips it before SentencePiece and
prepends it as a token afterwards, so "Marian adds the tag after SentencePiece"
is true of that tokenizer only, and how Marian was trained was not examined.
Only MADLAD-400 puts `<2xx>` in the text (493 `<2xx>` pieces, all
user-defined), and it shows the same standalone `▁` as ours. The SentencePiece
author says the dummy prefix exists so that a word is segmented the same
wherever it appears (issues 305, 183 and 877 of the SentencePiece repository),
and said in issue 877 that they had not run experiments on it. On a toy
310-piece model, turning it off segmented a sentence-initial word into single
letters while the same word mid-sentence was one piece (the direction is
certain; the size of the effect on the real model was not measured).]

[DESIGN: 2. ZWNJ and ZWJ in `mn_Mong`: keep stripping them (this resolves the
FLAG above), with safeguards.] Stripping is right for a translation system: the
joiners carry no lexical content, preserving them would put two token sequences
per word into the data and let the decoder emit invisible controls (wrong glyph
forms in generated text), and stripping makes the legacy and the standard
orders collapse to the working `<base, FVS>`. The cost is that it changes the
positional context of an isolated letter (an FVS valid for the initial form may
be invalid for the isolate form: 62 against 56 of 117), which matters only for
isolated-letter displays, not for running text. Safeguards: (a) the D5
converter must NEVER emit ZWJ or ZWNJ (use U+180A MONGOLIAN NIRUGU where
joining is wanted), and a validator must fail if converter output contains
either one before normalization; (b) the normalizer should count what it strips
per language (not implemented yet; do it when a real mn_Mong corpus is first
normalized); (c) audit real traditional-script text before the policy is
frozen: counts and neighbours of U+200B, U+200C, U+200D, U+2060, U+FEFF,
U+180A..U+180F and U+202F; words that become identical once the joiners are
removed; a free variation selector that does not directly follow a letter. The
concrete gate of the older FLAG is retained as part of this audit: if the audit
finds ZWNJ used as a suffix separator in legacy text, mapping ZWNJ to NNBSP is
the first thing to try before switching to the preserve policy. Flip condition:
switch to "preserve, move the joiner after the FVS, and gate with a validator"
only if the audit shows source-side joiners carrying display-critical content
that must round-trip. Never apply NFKC or NFKD to `mn_Mong`: NFKC turns U+202F
into a plain space, and SentencePiece's `nmt_nfkc` rule also turns ZWNJ and
NNBSP into spaces (verified locally); the trainer already uses `identity` and a
test pins it.

[EVIDENCE: Unicode 18 core specification, chapter 13 (re-read directly):
"U+200C ZERO WIDTH NON-JOINER (ZWNJ) and U+200D ZERO WIDTH JOINER (ZWJ) may be
used to select a particular positional form of a letter in isolation or to
override the expected positional form within a word". The specification gives
them no lexical role: only MVS and the free variation selectors (FVS) carry
spelling information, and the normalizer preserves those. Unicode Technical
Note #57 (a Technical Note, not normative) says everyday text does not need
them and recommends the visible U+180A for joining. U+180A..U+180F, U+200C,
U+200D and U+202F all have combining class 0 and are stable under NFC (checked
locally with `unicodedata`), so NFC never reorders them. MNS 4932:2000 (the
Mongolian national standard, read from the scan) defines ZWJ and ZWNJ as
positional or joining controls for displaying a letter's initial, medial or
final form alone or for splitting a word for display, and says that when a free
variation selector is also needed for such a standalone or separated letter it
is placed AFTER the ZWJ or ZWNJ; the Unicode standard requires the selector
IMMEDIATELY after the base letter. Real shaping (HarfBuzz 15.0 with Noto Sans
Mongolian v3.103) accepted the selector in 62 of the 117 standardized variants
of UCD 18.0.0 in the Unicode order `<base, FVS, ZWJ>` and in 0 of 117 in the
legacy order `<base, ZWJ, FVS>`; the 2020 font build also diverged between the
two orders in 21 of 117.]

[DESIGN: 3. NNBSP against MVS for separated suffixes: emit NNBSP, preserve
both, map nothing yet.] The converter (D5) emits NNBSP before separated
suffixes: it is the national standard's form and shapes without error in both
fonts tested (no Mongolian reader has judged the result). `normalize` keeps
preserving NNBSP and MVS exactly as they are; there is no NNBSP to MVS mapping
now. A one-way NNBSP to MVS mapping before a Mongolian letter is possible
later, never the reverse, because MVS also has the lexical use. Revisit when
the national standards are revised to MVS or when a renderer matrix shows that
MVS is safe in the fonts users have.

[EVIDENCE: Unicode 16.0 and later (re-read in the Unicode 18 text): "Prior to
Unicode Version 16.0, U+202F NARROW NO-BREAK SPACE (NNBSP) was used to
represent this small whitespace; ... However, its role has been taken over by
U+180E MONGOLIAN VOWEL SEPARATOR (MVS), which not only prevents word breaking
and line breaking, but also triggers special shaping for the following
separated suffix." UTN #57 version 4 discourages NNBSP. The W3C Mongolian
Layout Requirements (a July 2025 draft note) still documents NNBSP. MNS
4932:2000 (read from the scan; it predates Unicode 16) defines U+202F NARROW
NO-BREAK SPACE as the Mongolian space for separated suffixes (section 3.4 and
item 14, with example lists) and defines MVS only as the separator of a
word-final A or E vowel from its stem (item 12), the lexical use in the Unicode
text. Font run (HarfBuzz 15.0, Linux): Noto Sans Mongolian 2.001 (the 2020
system build) shapes NNBSP plus suffix and MVS plus suffix differently (NNBSP:
narrow glyph, suffix in medial form; MVS: the `space` glyph, suffix in initial
form), while Noto Sans Mongolian v3.103 (13 Oct 2026) produces IDENTICAL glyph
sequences and an identical render for the two; in both fonts MVS in the lexical
a/e case still changes the glyphs, and the joiners change forms.]

[DESIGN: 4. Abbreviations: the Cyrillic splitter now has a shared
Russian-origin base plus per-language sets.] Mongolian-only and Kazakh-only
entries no longer leak into the other languages. The single Cyrillic
abbreviation set of the D3 entry is now `_CYRILLIC_COMMON_ABBREVIATIONS`,
`_MONGOLIAN_ABBREVIATIONS` and `_KAZAKH_ABBREVIATIONS`, with one family per
language. The Kazakh set gained the entries of the Kazakh punctuation rules
(Syzdykova, handbook chapter, 2000, reprinted at bilim-all.kz) listed below
(`т.с`, `т.т`, `ж`, `ғ`, `сек`, `мин`, `сағ`); `т.с.с` is used in the chapter's
running examples but is not in its paragraph-8 list, and only `б.з.д` has no
support in the text read. An officially approved Kazakh orthography text was
not located, so the Kazakh and Mongolian entries still lack an orthographic
authority. A wrong abbreviation entry is low-risk: it can only suppress a split
before a capitalised word. Long term, hand lists are a stopgap: D4 should
induce abbreviations from the real corpus (the Punkt approach of Kiss and
Strunk 2006: collocation and length statistics) and measure boundary errors per
language on a small human-labelled sample (the D2 annotation app can produce
it). A pretrained sentence segmenter such as SaT could be used only as an
evaluation baseline (project rule: no external models in the pipeline), and its
Mongolian and Kazakh coverage was not confirmed.

[EVIDENCE: The Kazakh punctuation rules (a handbook chapter by Rabiga Syzdykova
from "Қазақ тілінің анықтағышы", Astana: Elorda, 2000, reprinted at
bilim-all.kz on a learning site; nothing on the page marks it as official or
approved; paragraphs 7-10 of the punctuation chapter) list the dotted
abbreviations `т.б.`, `т.с.`, `т.т.` and `ж.` (year), `ғ.` (century), `сек.`,
`мин.`, `сағ.`; initials keep their dots; unit symbols such as km, m and cm
take no dot; `т.с.с.` occurs 20 times in the running examples of the page. The
Mongolian punctuation rules summary that was read (a blog post crediting
material by Ш. Лувсанвандан, not a primary text; address in the research note)
uses a dot for initials and for abbreviations of names and a hyphen for
contractions of common words (`д-р` doctor, `н-р` comrade), which cannot create
a sentence boundary. The dotted Mongolian forms for "and so on", "for example",
"in other words" and "before the common era" are widely attested in government,
health and news documents, but no authoritative Mongolian list was found.
Tests: `tests/unit/test_split_rules.py` pins every entry of the Kazakh and
Mongolian sets, and that one entry of each (`ж`, `г.м`) does not apply to the
other languages.]

[EVIDENCE: 5. Verification status. Retrieved and read: MNS 4932:2000 (full
scan, hosted in the Unicode document archive at
`www.unicode.org/L2/topical/mongolian/MNS4932-2000(recovered).pdf`; read as
images, so the translations of its items 12-16 are the research pass's own and
want a native-speaker check; its control-character table on page 5 has labels
that do not match their code points and should not be relied on). The
regulator's list (crc.gov.mn, items 148-150) confirms that MNS 4333:1996, MNS
4932:2000 and MNS 5552:2005 exist, which proves titles, not that the editions
are current. NOT retrieved: the full text of MNS 5552:2005 (it needs the
national portal login), its current edition, and MNS 4333:1996. A 2025 revision
with public comment open until 18 Aug 2024 was seen only in a search-result
snippet of a portal notice that could not be re-fetched (the 2025 title next to
a 2024 deadline is unexplained); adoption and edition status are unconfirmed.
Fonts tested: Noto Sans Mongolian 2.001 (system, 2020) and v3.103 (2026) only,
with HarfBuzz 15.0.0 through uharfbuzz 0.57 and Pillow-Raqm (HarfBuzz 14.5.0)
on Linux and Python 3.14.10. Not tested: browsers, Windows, Mongolian Baiti,
Pango (the attempt rendered blank). No Mongolian reader has judged the
orthographic correctness of the renders.]

[DESIGN: 6. Gates.] D4: abbreviation induction and a boundary-error sample per
language; count the o-like substitutes and uppercase Y inside Cyrillic-majority
tokens in real khk_Cyrl (already listed above); audit U+200C, U+200D,
U+180A..U+180F and U+202F in any real traditional-script text before D5. D5:
the converter never emits ZWJ or ZWNJ and a validator enforces it; emit NNBSP;
render-check a sample in the target fonts; decide the abbreviation and suffix
questions that need a Mongolian reader. D8: the tag and prefix design of item 1
with the tests listed there; decide per-direction corpora against a custom
transform; sanitise reserved strings in raw text.

## 2026-11-07: D4 filtering and mining - the rulings, made without a review round

D4 (`src/rinchen/{corpus,filter,mine}/`, `src/rinchen/cli/{filter,mine,errors}.py`,
`configs/data/filters.yaml`) filled gaps in the master plan with the rulings below.
**They were made by the builder without a review round, because the user's
instruction for this milestone was "don't respond back until you finish everything
related to coding".** None is an approved decision; each lists what it costs if wrong
so that it can be undone. The design with the full wording is
`docs/superpowers/specs/2026-11-06-d4-filtering-mining-design.md` (rulings 1 to 17;
the numbers below are the same); procedures, measured numbers and the known
limitations are in `docs/runbooks/04_filtering_mining.md`. The work lives on branch
`d4-filtering-mining`, which was fast-forwarded into `main` on 2026-11-07 (not pushed).

[DESIGN: 1. Twelve tasks on one branch, the pipeline last among the filters.] Cost if
wrong: re-slicing, no code impact. [DESIGN: 2. `corpus/` (schema, readers, writers) is
created now, because the master plan's contracts need it and D1-D3 did not build it;
`dedup.py` and `splits.py` are left to the milestone that needs them (D6 onward).] Cost
if wrong: they are added later. [DESIGN: 3. The numbers rule is the literal "digit
multiset" of the master plan.] It compares the multiset of digit characters, not of
number tokens, so `1,5` equals `1.5` but also `12` equals `21`. Cost if wrong: a few
false accepts; the token form can replace it behind the same `num_ok` score. The lone
CJK numerals `shi` (ten), `bai` and `wan` count as numbers, which gives false rejects
against digit-free English ("very" reads as 10); the runbook lists them.

[DESIGN: 4. Every stage scores every row; only `minhash` ignores rows that are already
rejected.] Cost if wrong: GPU time spent on rejected rows; gain: complete scores for
analysis. [DESIGN: 5. QE is a stage plus a `QEScorer` protocol.] The MetricX-QE
adapter (`filter/qe_eole.py`, over Eole 0.6.4 helpers) was verified on the GPU and
shipped; there is no CometKiwi adapter, and `qe.scorer: cometkiwi` skips the stage.
Cost if wrong: the adapter depends on undocumented Eole helpers, so `eole==0.6.4` stays
pinned for it and the `network` test is the only guard; first use downloads about 6 GB.
[DESIGN: 6. LaBSE runs through sentence-transformers, 7. the aligner embeds overlap
concatenations as text, 8. `Manifest.status` is added (optional, default empty), 9. the
GlotLID parity reference is the raw fastText Python API on the same model because the
command-line binary is not installed, 10. Latin letters in Cyrillic text are tolerated
up to 10%.] Costs: (7) up to `max_k - 1` times more embedding calls; (8) a field in a
shared contract; (9) weaker than the master plan's wording; (10) a brand-name tolerance
that could hide real Latin contamination.

[DESIGN: 11. Source files may hold visible non-ASCII letters (Cyrillic, Han, Hangul);
invisible, look-alike and traditional Mongolian characters are built with `chr()`.]
D3's "ASCII only" was a workaround for the editor tools. Cost if wrong: none.
[DESIGN: 12. Letters of Script=Common or Inherited are neutral in `script_of`.] A
Japanese sentence with the katakana long-vowel mark scored 0.778 and was falsely
rejected at 0.9. Cost if wrong: `script_of` callers see slightly different shares for
Common letters; reverting is three lines. [DESIGN: 13. The `lid` accept table defaults
to `{zho_Hans: [cmn_Hani, wuu_Hani]}`.] The real GlotLID v3 never says `zho_Hans`
(Mandarin is `cmn_Hani` 19 of 20, `wuu_Hani` 1 of 20) and calls traditional Mongolian
`und_Mong`, which the wrapper's pre-detector replaces by `mn_Mong`. Cost if wrong: Wu
text is accepted as Chinese and Simplified versus Traditional is not checked in D4
(`Hani` does not separate them and neither does `script_ratio`).

[DESIGN: 14. The pipeline collects after every stage.] Errors surface at the stage that
caused them. Cost if wrong: no cross-stage lazy fusion, which does not matter.
**Corrected in the final review wave:** the docstring of `run_pipeline` said the LID
model, LaBSE and the QE model are never in memory together, but the CLI kept the LaBSE
embedder alive for the whole run, so about 1,800 MiB of LaBSE stayed on the GPU during
`qe` (6,665 MiB allocated in all, found in the 2026-11-07 smoke run). The pipeline now
releases an embedder that implements `Releasable` right after the `labse` stage, and the
docstring says so. [EVIDENCE: measured after the fix, in the real `filter run` command
over the 24-row smoke corpus, 2026-11-07: 4,866 MiB allocated and 6,387 MiB used on the
device (1,068 idle) after the first QE call, against 6,665 and 8,219 MiB (1,073 idle)
with the release switched off in a scratch script; not tried on an 8 GiB card.]
[DESIGN: 15. The aligner has a `merge_penalty` (edge weight `cost * (a + b) +
merge_penalty * (a + b - 2)`, default 0.3).] With the plain weighting a weak pair rides
on a strong neighbour and F1 on the planted data was 0.906; 0.3 gives 0.989 (mean
0.993, minimum 0.978 over seeds 11 to 30). Cost if wrong: one more constant, tuned on
synthetic additive data only; the real-data value is unknown. [DESIGN: 16. Missing
neighbours in the aligner's margin term count as unrelated (distance 1.0).] Cost if
wrong: a lone pair is accepted when its cosine distance is below about 0.87, which is
lenient for LaBSE. [DESIGN: 17. One shared CLI error mapping, `cli/errors.py`, used by
`cli/spm.py`, `cli/filter.py` and `cli/mine.py`; `mine align` gained `--merge-penalty`.]
Cost if wrong: D3's `cli/spm.py` was edited (behaviour and tests unchanged).

[DESIGN: Final review wave, after the whole-branch review of 2346cea..8cbd39f.] Both
commands now refuse a run that would overwrite its own input (`--corpus` equal to one of
the two output tables; `--out`, or the manifest beside it, equal to `--src` or `--tgt`;
ConfigError, exit 2) and an out dir that holds the manifest of another stage (DataError,
exit 3). They share one skip and commit protocol in `manifest/stage.py`
(`outputs_current`, `refuse_foreign_manifest`, `begin_commit`), the manifest and Parquet
writers write through a uniquely named temporary file, and `fasttext-wheel` (the vendored
shim that provides `import fasttext`) joins `tool_versions` next to `fasttext-new`. Cost
if wrong: a second pass given the first pass's directory as `--out-dir` gets an error
instead of a silent overwrite, and one directory cannot hold both a `filter run` and a
`mine align` output; an unreadable manifest is still removed by the next run, because it
names no stage to protect.

[EVIDENCE: Measured on the development machine (Intel Core Ultra 7 265K, RTX 5070 Ti,
CPython 3.14.10), 2026-11-06 and 2026-11-07. Synthetic planted data: MinHash recall
0.98, precision 1.0 (gates 0.95); aligner F1 0.989 (gate 0.95); planted reason counts
exact (50 rows, 28 kept). Real models: GlotLID parity 200 of 200 top-1 labels (20 of
the golden entries are synthetic traditional-script lines); LaBSE cosine 0.90 for a
translation pair against 0.11 for an unrelated pair; MetricX-QE 0.0 and 0.09 for good
pairs against 9.9 and 25.0 for a mismatched one. Throughput: GlotLID on CPU 77,000
lines/s on all cores and 6,500 on one core (38-character lines); LaBSE about 5,500
sentences/s at batch 128 (2,400 on 117-character lines); MetricX-QE 500 pairs/s at batch
32 (226 on 127/156-character pairs), 4,935 to 5,041 MiB of GPU memory allocated. Integration run: 24
rows through the real CLI with the three real models, 16 natural pairs kept, all 8
planted bad rows rejected for the intended reason, 9.5 s wall; a 12-line alignment gave
12 correct 1-1 rows. The Mongolian sentences of that run were written by the assistant
and not reviewed by a Mongolian reader, so it is an integration check, not evidence of
quality.]

[EVIDENCE: Not verified. No threshold was checked on real data (none exists); the
aligner was calibrated on synthetic additive embeddings only; the no-GPU path of `qe`
was run with a fake scorer only; LaBSE on CPU, other GPUs, bf16 and fp16 QE, and
`filter run` on anything larger than 24 rows were not measured; the `script_ratio` and
`minhash` speeds are reviewer measurements that were not repeated; offline use of the
cached models was not tried. Two small findings from setting up the runbook: the
installed `huggingface_hub` ships `hf`, not `huggingface-cli`, and
`hf download cis-lmu/glotlid model_v3.bin --local-dir models/lid` writes
`model_v3.bin` while the default config reads `models/lid/glotlid_v3.bin`, so the file
must be renamed or `lid.model_path` set.]

[DESIGN: Gates.] The D3 gates that the master plan assigned to D4 stay OPEN because
they need real text: abbreviation induction and a boundary-error sample per language;
counts of o-like Latin substitutes and uppercase `Y` inside Cyrillic-majority tokens in
real khk_Cyrl; an audit of U+200C, U+200D, U+180A..U+180F and U+202F in real
traditional-script text before D5. Also open: a real-data check of every threshold, a
re-tune of `merge_penalty` on real embeddings, and a decision on a CometKiwi adapter.

## 2026-11-07: D5 script converter - the rulings, made without a review round

D5 (`src/rinchen/script_conv/`, `src/rinchen/cli/convert_script.py`,
`configs/eole/converter.yaml`, `configs/eole/converter_tiny.yaml`,
`configs/spm/converter_char.yaml`) filled gaps in the master plan with the rulings below.
**They were made by the builder without a review round, because the user's instructions
for this stretch were "continue dont stop" and "continue" (on top of the earlier "don't
respond back until you finish everything related to coding").** None is an approved
decision; each lists what it costs if wrong so that it can be undone. The design with the
full wording is `docs/superpowers/specs/2026-11-07-d5-script-converter-design.md` (rulings
1 to 13, amended to match the finished code, and the amendments A to U; the numbers and
letters below are the same); procedures, measured numbers and the known limitations are in
`docs/runbooks/05_script_converter.md`. The work lives on branch `d5-script-converter`, cut
from `main` at `4eac98f`; it was fast-forwarded into `main` on 2026-11-07 (tip `862e176`) and is not pushed.

[DESIGN: 1. The engine half of the STOP gate becomes `ScriptConverter`.] The master plan
says "served via engine with `script="Mong"`"; there is no engine until D11 and D12. Cost
if wrong: the engine task adapts to this small interface. [DESIGN: 2. Word-level units.]
The model converts one word per line; text is segmented, words are deduplicated and
cached. Cost if wrong: homograph context is lost; a sentence-level model needs a different
data layout. [DESIGN: 3. No suffix splitting in the rule baseline.] Case-suffix spelling
depends on stem class and the reference holds lemmas only, so rules written here could not
be checked and would put silently wrong suffixes into training data. Cost if wrong: the
model gets no rule-made suffix gaps; real data has to teach them.

[DESIGN: 4. NNBSP before separated suffixes, MVS only where lexical, ZWJ and ZWNJ never.]
This follows the D3 follow-up decision and is not reopened. Cost if wrong: see that
decision. [DESIGN: 5. Both directions in one model, tags `<2mn_Mong>` and `<2mn_Cyrl>`,
one corpus per direction.] Amended while building: Eole takes one validation corpus, so
`valid` holds the Cyrillic-to-script direction and the other direction is trained without
a validation loss. Cost if wrong: the reverse direction is learned only from rule and
reference pairs (many-to-one in the data) and is judged only by fixed-word checks.
[DESIGN: 6. 1000 steps with dropout 0 for the tiny model, CER <= 0.15 unchanged, both
directions, Noam learning rate 0.5.] The first version said 600 to 800 steps; with dropout
0.1, 800 steps kept 54% of the words that carry an NNBSP or selector and 1500 steps were
needed for 90%. Cost if wrong: a tiny test of about 35 s, and a hyperparameter that suits
a memorization fixture only.

[DESIGN: 7. Eole configs are rendered by code and validated.] The master plan's template is
corrected for the three failures found (`tgt_prefix`, `word_vec_size`, `bucket_size`) and
the validator also rejects the real forbidden Eole field names (16, with a drift test) and
a `$` in any string. Cost if wrong: D7's `eole_cfg.render` should absorb `config.py`.
[DESIGN: 8. CTranslate2 export always runs with CUDA hidden.] Cost if wrong: none on CPU
export; an export that wants the GPU does not exist. [DESIGN: 9. Own character
SentencePiece trainer] instead of extending `spm.train.train_spm`, which is unigram,
byte-fallback and per-language by construction. Cost if wrong: about 30 lines of trainer
wiring duplicated. [DESIGN: 10. An invalid neural output is replaced by the rule
baseline's word and counted.] Judged per word (ruling M). Cost if wrong: some outputs are
the phonographic guess instead of the model's.

[DESIGN: 11. Reference provenance.] The Wiktionary harvest (2,029 rows from the English
Wiktionary through the MediaWiki API on 2026-11-07, one revision id per row) is a held-out
reference and the source of the lexicon and the tiny fixture, never training data for a
real model. It carries no guarantee of Mongolia-only provenance; the harvest filters out
entries labelled Inner Mongolia, Chakhar and China and reads only the Mongolian section,
and a Mongolian reader has not checked it. Cost if wrong: a wrong lexicon entry is wrong
training data for the tiny fixture only. [DESIGN: 12. Digits pass unchanged; `.` and `,`
map to U+1803 and U+1802,] except directly between two ASCII letters or digits. Cost if
wrong: one table and one pattern. [DESIGN: 13. Development and held-out halves of the
reference by the parity of the SHA-256 of the word's UTF-8 bytes.] The rules were designed
on the development half (1,005 words, the lexicon); the held-out half (1,022) was measured
once. Cost if wrong: the held-out half is spent after that figure.

[DESIGN: Amendment A. Every pairs TSV is headed `cyrillic<TAB>script`; the reference file
keeps its own columns.] Avoids a silent column mix-up between tasks. Cost if wrong: one
rename. [DESIGN: B. `CorpusFiles.files`, `rows` and `dropped` are keyed by file name and
drop code.] Cost if wrong: none. [DESIGN: C. The integration fixtures `tiny_run` and
`tiny_ct2` are session-scoped, in `tests/integration/conftest.py`.] One training per test
session. Cost if wrong: none; a test that needs another model trains its own (the CLI
chain test does). [DESIGN: D. If the held-out half gives fewer than 70 fallback-correct
words, the golden is filled with lexicon rows to 100 and the shortfall is reported.] It was
not needed (237 held-out words are fallback-correct; the golden is 30 lexicon plus 70
fallback). Cost if wrong: a weaker fallback pin on a thin reference.

[DESIGN: E. The validator accepts letter, FVS, NNBSP, letter.] A stem-final selector may
precede the suffix gap, which is how `segments` reads it; the strict reading would flag
legitimate stems. Cost if wrong: one helper (`_ends_with_stem_letter`). [DESIGN: F. `.`
and `,` directly between two ASCII letters or digits stay unchanged.] `3.5`, `1,000`,
`a.b` and URLs are not sentence punctuation. Cost if wrong: a mark between two non-ASCII
letters or digits (full-width digits) still maps. [DESIGN: G. The data-free fallback lives
in `fallback.py`; `rules.py` re-exports it and the fixture builder imports only
`fallback`.] The builder can then regenerate a lost lexicon. Cost if wrong: one more file.

[DESIGN: H. `rule_pairs` normalizes before the lookup; an empty train split is a
`DataError` raised before any file is written; a side with a U+0020 is dropped under the
new code `multiword`.] An empty valid split stays allowed. Cost if wrong: a very small
`--pairs` input with a high valid fraction fails with an error naming the counts; a
multiword pair is dropped, not split. [DESIGN: I. `converter_char.yaml` has `vocab_size`
and `output_prefix` only.] The trainer is deterministic, so a `seed` would be dead config.
Cost if wrong: adding one later is a schema change. [DESIGN: J. The character trainer
reads a fixed `<prefix>.input.txt`; the `.vocab` hash is the portable identity.] The
`.model` embeds its absolute prefix, so its hash holds per location only; manifests record
both files; the protobuf is not scrubbed. Cost if wrong: a `.model` hash differs between
locations.

[DESIGN: K. The tiny profile uses dropout 0 and 1000 steps.] Chosen by a sweep over 800,
1000, 1200 and 1500 steps: the smallest that keeps at least 93% of the marked words with
both CERs at most 0.10 (the test gate is 90%); no gate was lowered. Cost if wrong: see
ruling 6. [DESIGN: L. `Conversion` has three fields: `text`, `rule_fallbacks`,
`invalid_outputs`.] `rule_fallbacks` counts outputs the rule baseline spelled;
`invalid_outputs` counts occurrences that keep their source text because no usable output
exists. Cost if wrong: the engine reads three fields. [DESIGN: M. A converted word is valid
when `is_script_word` says so:] it passes the validator, holds a letter, and is made of
letters, selectors, MVS, NIRUGU and NNBSP only; a Cyrillic word can convert only to such a
text. Cost if wrong: an output with an ASCII character that someone wants would be
replaced by the rule word.

[DESIGN: N. `predict` pre-tokenises with the character model and runs Eole with
`transforms: []`.] Eole copies the `valid` corpus's prefix into the checkpoint and the
checkpoint overrides the prediction config (see the second finding below). Cost if wrong:
`predict` must track the training-side transform chain by hand, and so must any D8 or D11
per-direction evaluation. [DESIGN: O. A model directory is self-contained: `export` copies
the character model into it as `converter_char.model`.] `apply` and `eval` need only
`--model-dir`. Cost if wrong: a copy of the character model in every export (about 1 KB
for the tiny one). [DESIGN: P. `train` owns the character model:] it trains and checks it
on the corpus words, renders and validates the config and runs Eole; the manifest records
the `.model` and `.vocab` hashes. Cost if wrong: every `train` retrains the character
model, and a failure after that leaves it.

[DESIGN: Q. No skip logic and no `--force`.] A rerun overwrites under the `begin_commit`
and manifest-last protocol. Cost if wrong: a rerun always retrains (35 s for the tiny
profile, GPU hours for the full one). [DESIGN: R. `apply` converts the whole input as one
text and prints the counters on standard error; `eval` prints CER, word accuracy and `n`
(then the two counters) and exits 3 on an empty pairs file.] Cost if wrong: the whole input
is held in memory, in several copies. [DESIGN: S. The commands look up
`train_char_model`, `run_training`, `export_ct2` and `load_converter` at call time.] Test
seams. Cost if wrong: those module-level names are part of the test contract.
[DESIGN: T. The profile is not recorded in `Manifest.status`.] The model has no free-form
field. Cost if wrong: the manifest does not name the profile; the hash of `converter.yaml`
identifies the rendering. [DESIGN: U. The character model is trained on the `cyrl2mong`
files only.] The `mong2cyrl` files hold the same words with the sides swapped. Cost if
wrong: none while both directions come from one pair list.

[EVIDENCE: The spikes, 2026-11-07, this machine (Intel Core Ultra 7 265K, 46 GiB, RTX 5070
Ti not used by training; Eole 0.6.4, CTranslate2 4.8.4, CPython 3.14.10). `eole build_vocab`
and `eole train` run on CPU once `tgt_prefix: ""` is on every corpus (without it,
`TypeError: object of type 'NoneType' has no len()`) and `word_vec_size` equals
`hidden_size` (otherwise a layer-norm shape error at d64). With Eole's default
`bucket_size` of 262144, 300 steps of a d64 model took more than 12 minutes (about 360
tokens/s, 1.4 million log lines of "Weighted corpora loaded"); with `bucket_size` 1024,
`num_workers` 1, `prefetch_factor` 8 and `OMP_NUM_THREADS=2` the same 300 steps took 20 s.
Noam `learning_rate` 2.0 peaks near 0.025 at d64 and does not learn; 0.5 with 100 warmup
steps gave CER 0.935 after 300 steps and 0.007 after 600 on a toy letter-map task (500
words), so the master plan's "300 CPU steps reach CER <= 0.15" is false as written.
CTranslate2 against Eole greedy decoding over the 500 words: 499 identical, the one
difference a degenerate repetition stopped at different lengths. `str.split()` splits at
U+202F, which would cut a script word from its suffix.]

[EVIDENCE: The first upstream finding, `D-CT2-EOLE-CUDA`. `python -m
ctranslate2.converters.eole_ct2` SEGFAULTS when a CUDA device is visible
(`ctypes.string_at` on a device pointer in `ctranslate2/specs/model_spec.py:757`,
`Variable.to_bytes`) and leaves an empty `model.bin`; with `CUDA_VISIBLE_DEVICES=""` it
writes a valid model. Reproduced again on 2026-11-07 on the tiny checkpoint of the runbook
example: exit 139 in a shell (signal 11), `model.bin` 0 bytes, `shared_vocabulary.json`
1,043 bytes. Worked around, not fixed: `export_environment()` hides the GPU; the probe is a
test behind `RINCHEN_CT2_CUDA_PROBE=1`.]

[EVIDENCE: The second upstream finding, `D-EOLE-PREFIX-OVERRIDE`. Eole 0.6.4
(`train_single.ValidationConfigurer`) copies the `valid` corpus's `src_prefix`
(`<2mn_Mong>`) into `transforms_configs.prefix` of the checkpoint's `config.json`, and
`config/run.py` (`_update_with_model_config`) lets the checkpoint's `transforms_configs`
override the prediction config per transform name. Every `eole predict` with the `prefix`
transform therefore ran under the Cyrillic-to-script tag, whatever the config said: the
forward direction was right by coincidence and the reverse wrong (67 of 448 words differed
from CTranslate2, agreement 0.85, reverse CER 0.0533). Evidence: CTranslate2 fed the wrong
tag equals the old Eole output on 448/448; patching the checkpoint's tag makes Eole equal
CTranslate2 on 448/448; Eole with the source pre-tokenised as tag plus pieces and
`transforms: []` equals CTranslate2 on 448/448 in both directions, and the reverse CER
falls to 0.0128. Reproduced again on 2026-11-07 on the runbook example's checkpoint: the
checkpoint holds `prefix.src_prefix` `<2mn_Mong>`; `<2mn_Cyrl>` and `<2mn_Mong>` give the
same output on 448 of 448 lines; 382 lines match the reference against 428 for the
pre-tokenised `predict`. Consequence for D8 and D11: any per-direction Eole evaluation must
bypass the transforms, or a parity gate fails spuriously.]

[EVIDENCE: The rule baseline, measured once after the rules were frozen. Reference: 2,029
rows, 2,027 valid, 1,005 development and 1,022 held-out. Development: letter map alone 129
of 1,005 (12.84%), then five contextual rules, 231 (22.99%), CER 0.2345. Held-out: 237 of
1,022 exact (0.2319), CER 0.2342. The spellings are dictionary spellings (median script
length 1.17 times the Cyrillic length), so about three words in four differ from a
phonographic spelling. Length ratio of script over Cyrillic over the 2,027 reference pairs,
quantiles 0.01, 0.05, 0.5, 0.95, 0.99: 0.8333, 1.0000, 1.1667, 1.5000, 1.9480 (lexicon,
1,005 pairs: 0.8343, 1.0000, 1.1667, 1.5000, 1.9920); not wired into
`configs/data/filters.yaml`.]

[EVIDENCE: The tiny model, 1000 steps, dropout 0, `--threads 2`: CER 0.0051 (Cyrillic to
script) and 0.0128 (script to Cyrillic) on 448 training words; marks kept 78 of 81 words;
CTranslate2 against Eole 448/448 in both directions; `ScriptConverter` word exact match 441
of 448 (0.9844) and 448/448 against Eole in both directions, no rule fallback and no invalid
output; training 34.8 s to 37.4 s over five runs (gate 150 s), export 2.3 s to 3.5 s; the real
CLI chain gives CER 0.0051 and word accuracy 0.9844. On the 52 words the corpus builder held
out as validation, the model gets CER 0.2444 and 0.2982 and 17 and 14 of 52 words exactly:
it memorises, it does not generalise. Throughput, CPU, new process per run, median of ten:
0.715 s for a one-word text, 0.790 s for 500 words, 0.820 s for 1,000 words (1,220 words/s
end to end); about 10,000 words/s once the model is loaded.]

[EVIDENCE: Not verified. No Mongolian reader judged the lexicon, the golden words, the
fallback or any output; the render check in the target fonts was not done (no shaping
library installed); the audit of U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180A..U+180F and
U+202F in real traditional-script text was not done (no real text); the Wiktionary
reference may not be wholly independent of mechanical derivation and cannot be guaranteed
Mongolia-only; the GPU and the full profile were never run; Windows and macOS were not run
(the process-group kill and several tests are Linux only); every threshold is a prior; the
script-to-Cyrillic direction has no validation loss; the wheel was not built to confirm
that the two data files ship.]

[DESIGN: Gates.] Open, because they need real data or a reader: the audit of the control
and joiner characters in real traditional-script text (assigned to D5 by the D3 follow-up),
the render check in Noto Sans Mongolian 2.001 and 3.103, the review of the lexicon, golden
words and fallback by a Mongolian reader, the `mn_Mong` length-ratio bounds for the D4
filter (measured, not wired), and the training of the full profile on real parallel script
data. The D3 gates assigned to D4 (abbreviation induction, o-like substitute counts) stay
open as recorded there.

## 2026-11-07: D5 final review - the `not_script_word` drop code and rulings V, W, X

A whole-branch review of `d5-script-converter` (a reviewer other than the builder) found no
Critical issue and four Important ones, and the build's controller ruled on them. The
rulings are V, W and X; the numbers and letters continue the amendments A to U above, and
the design (`docs/superpowers/specs/2026-11-07-d5-script-converter-design.md`) and the runbook
carry the wording. Like the rulings above, each lists what it costs if wrong.

[DESIGN: `not_script_word`.] `build_corpus` drops a pair whose NORMALIZED script side fails
`validate.is_script_word`, after the validator's codes and before the duplicate check, and
counts it under `not_script_word`; the order is empty, multiword, a validator code,
`not_script_word`, duplicate. Reason: the bare validator lets running text through (ASCII
letters, digits, punctuation, spaces), so a script word holding any of them was training
data. All 1,005 lexicon spellings and the 500 tiny fixture words pass `is_script_word`, so the
tiny gates did not move. Cost if wrong: a word that legitimately carries an ASCII character
in its script spelling is dropped, and counted, not lost silently.

[DESIGN: V. An empty valid split is an error.] This revises ruling H, which allowed it.
The reviewer ran `eole train` on a corpus with no valid row: it died at the first
validation step with a `ZeroDivisionError` (`eole/utils/statistics.py`), after the
character model and the vocabulary were built. `build_corpus` now raises `DataError` for an
empty valid split, before any directory or file is written, like the empty train split;
`train` refuses a corpus whose `cyrl2mong.valid.src` has no row (a corpus built elsewhere),
in a dry run too, before any write. The error text is "valid split is empty (pairs kept: N,
valid_fraction F, seed S)". Cost if wrong: a one-pair or very small corpus can no longer be
built; give it more pairs or another `--valid-fraction`. [EVIDENCE: the reviewer's run; this
fix wave did not repeat it, and pinned the refusal with tests instead.]

[DESIGN: W. The character model belongs to its run.] `train` writes it to
`<run-dir>/converter_char.model` and `.vocab` by default, with `--spm-prefix PATH` to
override, and `output_prefix` is removed from `configs/spm/converter_char.yaml` and from
`CharModelConfig` (it would be a dead key; the YAML holds `vocab_size` only; the loader keeps
its strict contract, so the old key is a `ConfigError`). This amends ruling I. The default
used to be `data/script_conv/converter_char.*`, shared by every run: a second `train`
overwrote the first run's model silently, and `export` bundled the overwritten file unchecked.
`export` now compares the character model's SHA-256 with the run's train manifest
(`<run-dir>/manifest.json`, stage `script_conv_train`) when it lists the model, and exits 3
(`DataError`, both hashes in the message) before any write on a difference; a run directory
without that manifest is trusted. The checked-in Eole configs name the new default path.
Cost if wrong: runs are self-contained, but the character model is retrained for each run
directory, the default path changed and the YAML lost a key.

[DESIGN: X. The harvest scripts are not shipped.] The scripts that fetched and parsed the
Wiktionary reference were scratch work. They are not in the repository, the report copy
`tests/fixtures/script_conv/wiktionary_reference.report.md` stays byte for byte as the
harvest wrote it (its line 4 names the scripts), and the runbook says so in its "What was
not verified" list and no longer sends the reader to a harvester. Cost if wrong: refreshing
the reference needs new tooling; the runbook states the contract it must meet (columns,
validator-clean, one Mongolian-section spelling per word, the development and held-out split
rule).

[DESIGN: The profile reaches the validator.] `validate_converter_config(cfg, profile=None)`
skips the CPU-only checks without a profile, and `run_training` used to validate without one.
`run_training` takes an optional `profile`; the `train` command passes the profile it
rendered, both when it plans the run and to `run_training`. A bare `run_training` call
still skips the checks, and its docstring says so. The validator also rejects a string-valued
`training` and `transforms_configs` (Eole evaluates such strings as code), a `$` in a key, and
the other gaps closed in the parked minors. Cost if wrong: a caller that forgets the profile
trains a tiny config on a GPU without a refusal.

[EVIDENCE: The four gate commands on the tree of the final review fixes, run as one chain
(`just` is not installed here): `ruff format --check .` 223 files already formatted, `ruff
check .` all checks passed, `mypy src` no issues in 99 source files, `pytest -m "not
network" -q` 2833 passed, 1 skipped, 10 deselected in 177 s. Each new test was proved by
mutation: the code under test was broken and the test failed.]
