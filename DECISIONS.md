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

## 2026-11-08: D6 synthetic data - the rulings, made without a review round

D6 (`src/rinchen/mbr/`, `src/rinchen/synth/`, `src/rinchen/config/synth.py`,
`src/rinchen/cli/synth.py`, `configs/data/synth.yaml`) filled gaps in the master plan with the
rulings below. **They were made by the builder without a review round, because the user's
instruction for this stretch was "Proceed. I'm going to sleep now. Don't stop until you finish
everything. Commit everything."** None is an approved decision; each lists what it costs if wrong
so that it can be undone. The design with the full wording is
`docs/superpowers/specs/2026-11-08-d6-synthetic-data-design.md` (rulings 1 to 12, amended to match
the finished code, and the amendments A to V; the numbers and letters below are the same);
procedures, measured numbers and the known limitations are in `docs/runbooks/06_synthetic.md`.
The work lives on branch `d6-synthetic-data`, cut from `main` at `0d534a3`; it is not merged and
not pushed. The design and this entry carry the date 2026-11-08, the day the run was planned for;
the commits and the measurements are stamped 2026-11-07 by the machine's clock.

[DESIGN: 1. Origin encodes role and generator family.] `bt` jobs give `bt`; `ft` jobs give
`synth_llm` or `synth_nmt` by generator kind; the `ft` origin itself stays reserved for forward
translation by our own teacher (D10). The master plan lists the four origins without saying which
job produces which. Cost if wrong: one table in `tags.py` and a re-labelling of produced corpora.
[DESIGN: 2. One `Translator` protocol for every generator.] The master plan's
`ct2_baselines.translate` signature is generalised to the LLM clients. Cost if wrong: LLM-specific
features (system prompts, few-shot) are not expressible without a protocol change.
[DESIGN: 3. Chat completions with one request per text,] server-side batching and `concurrency`
threads on the client, instead of the completions endpoint with prompt lists (which would need the
model's chat template in the client). Cost if wrong: more HTTP overhead per text.

[DESIGN: 4. Prompt wording is the builder's,] not a published Hy-MT prompt: the master plan names
the model but not its template and no source for it is available offline. It is three string
constants joined by plain concatenation, with the Cyrillic instruction present exactly when the
target is `khk_Cyrl`. Cost if wrong: translation quality of a real run may differ from the model's
recommended prompt; the three constants are what to replace, and the shard fingerprint does not
notice the change (run with `--force` or another `--seed`). [DESIGN: 5. Selection order empty,
purity, copy, QE, MBR.] Cheap filters first; QE (GPU) only on survivors. Cost if wrong: a candidate
that QE would have rated best but that fails a threshold never competes in MBR. [DESIGN: 6. QE
thresholds are the D4 ones, reused through `QeConfig`.] `max_metricx_qe` 8.0 (lower is better) and
`min_cometkiwi` 0.55, unvalidated priors. Cost if wrong: the numbers are arbitrary until real scores
exist, and a change to the D4 defaults changes D6 too (the YAML pins them).

[DESIGN: 7. Shard directory keyed by the job fingerprint] (input rows, job, generator config, shard
size, seed). Cost if wrong: changing any config value regenerates everything; old shard
directories stay until removed; the prompt wording and the weights behind a path or a served model
name are not in the key, so changing those silently reuses stale shards. [DESIGN: 8. Per-request
seeds derived from the request content,] not from global order, so a resumed run asks for the
same sampling seed as an uninterrupted one. Cost if wrong: servers that ignore `seed` give
non-reproducible text, and the job's `--seed` only names the shard directory: a new `--seed`
regenerates every shard with the same sampling seeds. [DESIGN: 9. MBR utilities limited to chrF]
(and the negation wrapper). Cost if wrong: no neural MBR until the COMET and MetricX
reference-based scorers exist.

[DESIGN: 10. Monolingual input contract is `id`, `lang`, `text`] (extras kept), not the full
"Monolingual row" of the master plan, because that schema has no builder yet. Cost if wrong: one
adapter when a monolingual stage exists. [DESIGN: 11. No `--force` skip logic beyond shard reuse.]
A rerun reuses valid shards and recomputes selection; `--force` ignores existing shards. Cost if
wrong: selection is recomputed on every run (cheap next to generation). [DESIGN: 12. A job never
involves traditional script.] `JobConfig` rejects any registry language whose scripts include
Mongolian script on either side: a generator produces Cyrillic Mongolian and the D5 converter owns
traditional script. Cost if wrong: no direct synthetic `mn_Mong` data; it has to come from the
converter, which is a scaffold.

[DESIGN: Amendment A. The selection reads the real names of the corpus row-id helper and frame
constructor from `rinchen.corpus.schema`.] No new schema. Cost if wrong: none; the rows follow the
schema. [DESIGN: B. A NaN expected utility is a `ValueError` naming the segment and candidate,]
also when the utilities cannot be averaged (+inf with -inf, overflow); strings and non-sequences are
a `TypeError`. Cost if wrong: a utility that returns NaN stops the run instead of silently winning
or losing at index 0; chrF cannot return NaN, a future neural utility might. [DESIGN: C. `selection.qe`
is D4's `QeConfig` unchanged.] MetricX-QE is lower-is-better, CometKiwi higher-is-better; a wrong
score count is an `UpstreamError`. Cost if wrong: see ruling 6; the first draft had one
`qe_min_score` of -1.0 with "higher is better", which is wrong for MetricX-QE. [DESIGN: D. `JobConfig`
rejects any registry language whose scripts include Mongolian script,] decided by the registry's
scripts and not by a code. Cost if wrong: see ruling 12.

[DESIGN: E. `Ct2Config.model_dir` and `spm_model` are repository-anchored, blank-rejected paths]
like the D4 configs. Cost if wrong: the fingerprint, hence the shard directory, changes when the
checkout moves. [DESIGN: F. `Ct2Config` rejects `n > beam_size`.] Cost if wrong: asking for more
hypotheses than the beam needs a larger `beam_size`. [DESIGN: G. A config that supplies `generators`
and no `jobs` gets no jobs.] Cost if wrong: a config that replaces the generators must list its
jobs. [DESIGN: H. A choice with a null, absent or empty `message.content`, and `choices: []`, give
no candidate and are not errors.] Cost if wrong: a server fault that arrives as an empty content looks
like a refusal and shows as `no_candidates`. [DESIGN: I. The backoff exponent is capped]
(`2**min(attempt, 10)`, and the sleep at 30 s). Cost if wrong: none. [DESIGN: J. Non-2xx errors carry
a 200-character preview of the response body.] Cost if wrong: a server that echoes request fields in
its error body can put a text fragment into an error message.

[DESIGN: K. Both CT2 families append `</s>` to the source tokens and the MADLAD tag is encoded with
the text.] The Hugging Face tokenizers do; the first spec line for MADLAD omitted the `</s>`. Cost if
wrong: unverified against a real converted model; a wrong convention shows as poor or empty
hypotheses on the first real run, and is a one-line change per family. [DESIGN: L. The CT2 backend
passes `max_tokens` as `max_decoding_length`;] the library's default of 256 silently truncated long
outputs; only `temperature` and `top_p` are ignored. Cost if wrong: none beyond longer decodes.
[DESIGN: M. QE pairs follow the final pair orientation:] `bt` (candidate, real text), `ft` (real
text, candidate). Cost if wrong: if the scorer should see the original source text rather than the
synthetic one on a `bt` pair, the orientation is wrong for `bt`; one swap in `select.py`.
[DESIGN: N. A candidate UTF-8 cannot encode is dropped by `clean_candidates`,] and
`ShardStore.write` refuses one with a `DataError`. A server's JSON escape for a lone surrogate
decodes to one, and the write after paid generation crashed with a bare `UnicodeEncodeError`. Cost
if wrong: such a candidate is silently one fewer, or `no_candidates`.

[DESIGN: O. The pipeline order is empty, purity, copy, QE.] A copy is dropped before it costs a QE
call. Cost if wrong: a copy that would also fail QE counts as `dropped_copy`, so the counts differ
from the first spec's order. [DESIGN: P. QE thresholds and the purity share are compared in
float32,] as the D4 stages do. Cost if wrong: a value within float32 rounding of a threshold passes
where float64 would fail. [DESIGN: Q. The copy rule compares the candidate with the real text,
both stripped.] Cost if wrong: a candidate equal to the real text apart from surrounding whitespace
is a copy. [DESIGN: R. `run_job` takes `generator_name` and `generator_cfg` as well.] The
fingerprint and the corpus string need them. Cost if wrong: callers pass them explicitly.

[DESIGN: S. Selection runs per shard and the counts are summed.] Memory is bounded by one shard plus
the accumulated corpus rows instead of every candidate of the job. Cost if wrong: equality with one
global selection holds only for scorers whose score depends on the (source, translation) pair
alone, and a scorer call never spans two shards (a smaller last batch per shard). [DESIGN: T.
`job.generator != generator_name` is a `ConfigError`.] Cost if wrong: none. [DESIGN: U. One decode per
shard per run:] `ShardStore.load` returns `Loaded`, `Missing` or `Invalid(reason)`; `completed` and
`invalid` are built on it and only the dry run calls `completed`. Cost if wrong: two methods with
no production caller besides the dry run.

[DESIGN: V. `created` is the stamp of the run, `begin_commit` goes right before the first output
write, the manifest `status` carries the QE status, `counts.json` is an output, and the scorer is
skipped with a recorded reason.] A rerun's corpus is identical to the first run's apart from
`created` (the plan said byte-identical); the shards are resumable work files and not manifest
outputs, so a failed generation must not erase an earlier manifest; `counts.json` (fingerprint, job,
generator, counts, shard statistics) is hashed as an output; the scorer is skipped when it needs
CUDA and none is available (`rinchen.filter.qe.skip_reason`) or with `--no-qe`. Cost if wrong: the
corpus file is not byte-identical across runs; the earlier manifest stands while newer shards are
being generated.

[EVIDENCE: The first upstream finding, `D-POLARS-GLOB`. polars 2.1.0 `pl.read_parquet(path)` and
`pl.scan_parquet(path)` expand `[`, `*` and `?` in a path as a pattern. Reproduced on 2026-11-07: a
file `run[1]/shard-00000.parquet` raises `ComputeError: failed to retrieve first file schema
(parquet): expanded paths were empty`, and `glob=False` reads it; a file named `a*.parquet` beside
`ab.parquet` and `a1.parquet` returned the rows of all three until `glob=False`. In the shard store
this made every existing shard unreadable and a resume silently regenerated everything (found in
review); `glob=False` is now on every Parquet read in `shards.py` and `corpus/readers.py`. Open
when this was written: `write_corpus` (`corpus/writers.py`) reads its output back through
`pl.scan_parquet(part_path)` without it.]

[EVIDENCE: The second upstream finding, `D-POLARS-PANIC`. A damaged Parquet file can raise
`pyo3_runtime.PanicException` from polars 2.1.0; it derives from `BaseException`, not `Exception`,
so `except Exception` does not catch it. Single-bit flips of a 3,324-byte shard-shaped file (8
rows of 2 candidates), 150,000 trials, a parent process restarting the child after an abort,
2026-11-07: 38,581 `PolarsError`, 45,528 decoded to the same frame, 65,607 decoded without error to
a different frame, 276 `PanicException` and 8 aborts of the process ("memory allocation of
2305843009213693952 bytes failed", SIGABRT, not catchable). A realistic 156,403-byte shard (512
rows of 4 candidates, 3,000 flips): 1,101 errors, 22 unchanged, 1,877 different, no panic, no
abort. The store catches the panic and treats the file as invalid; page checksums are not checked
and no content checksum was added, so silent bit rot inside a stored candidate passes
validation, and a rare damaged file can abort the process. Truncation and partial writes, the
crash cases, are caught.]

[EVIDENCE: The third upstream finding, `D-CT2-NUMHYP`. `ctranslate2.Translator.translate_batch`
(4.8.4, CPU) raises `ValueError: The number of hypotheses cannot be greater than beam_size *
patience` when `num_hypotheses` exceeds `beam_size` (`patience` defaults to 1 in this
version's `translate_batch` signature, as `beam_size` does to 2). Reproduced on 2026-11-07 with the
tiny converter model of a D5 test run: `beam_size=4, num_hypotheses=4` returns 4 hypotheses,
`4, 5` and `1, 2` raise the error, `2, 1` returns 1. It would end a long run at its first batch, so
`Ct2Config` rejects `n > beam_size` when the config is loaded.]

[EVIDENCE: The D4 QE direction correction. The first spec draft had a QE floor `qe_min_score`
(-1.0, "higher is better, the D4 convention"). D4's MetricX-QE threshold is a maximum
(`max_metricx_qe` 8.0 on a 0 to 25 scale where lower is better) and CometKiwi's a minimum
(`min_cometkiwi` 0.55); a floor of -1.0 on the MetricX-QE scale would have rejected nothing. The spec
and plan were corrected before the first implementation task (commit `8182462`) and the selection
config now reuses `QeConfig`.]

[EVIDENCE: Per-shard selection against one global selection. Selection is per source and the QE
batches only group calls, so selecting each shard on its own and summing the counts gives the same
rows and counts as one selection over the whole frame for a scorer that depends on the (source,
translation) pair alone. A test proves equality for shard sizes 1, 3, 7, 8, 40 and 100, for `bt`
and `ft`, with and without a scorer (corpus, counts, status, each pair scored exactly once); the
review of the combined diff compared 746 random cases and all were identical.]

[EVIDENCE: The float32 comparison. Purity is a float64 sum of per-script shares and D4 compares
float32. Three Han, three Hiragana, three Katakana and one Latin letter give shares 0.3, 0.3, 0.3
and 0.1; the three 0.3 sum to 0.8999999999999999, which fails `>= 0.9` in float64 and passes in
float32 (reproduced 2026-11-07 with `rinchen.synth.select`); three pure-Japanese splits sum to
0.9999999999999999 and failed `min_script_share: 1.0` in float64. QE: a MetricX score of 7.30000019
against the threshold 7.3, and 0.699999988 against 0.7, round to the same float32 and pass as in D4,
and fail in float64. The review found it (an Important finding); both comparisons were changed and
pinned with boundary tests.]

[EVIDENCE: The tiny run, measured through the CLI in process on 2026-11-07 (Intel Core Ultra 7
265K, 20 cores, 46 GiB, CPython 3.14.10, polars 2.1.0): the 40 planted rows give sources 40,
candidates 60, empty 5, purity 10, QE 10, copy 0, no candidates 5, no survivor 20, selected 15 for
`bt` and `ft` (without QE: QE 0, no survivor 10, selected 25), in 5 shards, 5 translator calls of 8
texts, 42.5 ms of wall time (median of 7, 40.7 to 44.9 ms); `corpus.parquet` 7,718 bytes,
`counts.json` 492, a shard file 1,217 to 1,225 bytes. MBR (single thread, chrF): `n * (n - 1)`
calls, 12, 56 and 240 for n = 4, 8 and 16, 0.66, 3.06 and 13.57 s per 1,000 segments of 51
characters and 2.16, 9.95 and 43.07 s per 1,000 of 189 characters. Input handling, 200,000 rows of
about 200 Cyrillic characters: `read_monolingual` 0.16 to 0.20 s, `plan_shards` 0.10 to 0.11 s,
`job_fingerprint` 0.55 to 0.57 s; RSS 65.2 MiB at start, 219.5 MiB after the read, 323.1 MiB after
planning (about 544 bytes per row of tuples), peak 323.0 MiB. Mutation checks by hand, one mutant at
a time, per task as recorded during the build: MBR 42, config 110, prompts 49, clients 121, CT2 52
(and 18 in the fix round), shards 80, selection 58 (about 45 in the fix round), runner 60 (40
after the refactor), CLI about 52, end batch 35; every survivor was shown to be equivalent.]

[EVIDENCE: Not verified. No vLLM or llama.cpp server, no NLLB or MADLAD CTranslate2 model and no
GPU was ever used, so the client paths ran against fakes only; the MADLAD and NLLB token
conventions, a server's handling of `seed`, `n` and `Retry-After`, throughput and GPU memory are
unknown; the prompts were never tried on a model; no Mongolian reader judged any output (none
exists); every threshold and default is a prior; the real MetricX-QE scorer was not run inside D6;
Windows and macOS were not run; concurrent runs into one output directory are not guarded; memory
beyond 200,000 rows was extrapolated and the memory of a whole `run_job` was not measured.]

[DESIGN: Gates.] Open, because they need a real generator, data or a reader: running any real
generator, a Mongolian reader's judgement of the output, calibration of the QE floor and the script
share, a language identifier on the synthetic side (purity cannot tell Mongolian from other
Cyrillic text), COMET and MetricX as MBR utilities, and the training-time `<bt>` and `<ft>`
insertion (D7 and D8). The D3 gates assigned to D4 and the D5 gates that need real text stay open as
recorded there.

## 2026-11-08: D6 final review - corrections and fixes

An independent review of the whole branch `d6-synthetic-data` (base `0d534a3`, head `f8a9d67`, 22
commits, read-only, run after the build and the documentation commit) found no Critical issue and
three Important ones; the gate reproduced exactly (4294 passed, 1 skipped, 10 deselected). This entry
records what it found, what was changed and how, CORRECTS in words the earlier statements it shows
to be false, and lists what is left alone. The code and tests are in the commit "D6: final review
fixes" and the documents in "D6: final review fixes, docs"; both are on the branch, which is still
not merged and not pushed. Nothing here is a ruling the user approved: like the rest of D6 these
were decided by the builder, and each says what it costs if wrong.

[CORRECTION: Ctrl-C did not cancel the queued requests in the real program.] The runbook's exit-code
table, the design and the build record of the fix for the Task 4 review ("interrupt now cancels
queued requests") said that an interrupt cancels the requests not yet sent. That was true of the
unit test, which patched `wait` to raise `KeyboardInterrupt` and so never exercised signal
delivery, and false of `rinchen`: once `polars` is imported (always, under `rinchen.cli.main`) a
SIGINT does not wake the main thread's untimed `concurrent.futures.wait()`, and the interrupt is
raised only when the call returns on its own, after every queued request has been sent. Through the
real binary, a loopback server sleeping 1 s per request, `concurrency` 4 and 40 texts in one shard,
a SIGINT 4.5 s after the start gave exit 130 6.33 s later with 40 of 40 requests received. [DESIGN:
Polled wait.] `_fan_out` now waits in slices of 0.2 s (`_INTERRUPT_POLL_S`), keeping every other
behaviour (the first failure cancels the rest and is the one raised, any `BaseException` cancels the
queue, results in input order, `shutdown(wait=True)` so requests in flight finish). The same probe
now gives exit 130 0.27 s after the signal with 16 of 40 received. Also stated plainly now: the
requests in flight are finished, not abandoned, up to `concurrency` of them, each with its own
retries, and a second Ctrl-C during that wait is not delivered until it ends. Cost if wrong: a
5 Hz wake-up of the main thread while a shard is being generated. `docs/VERIFY.md` gets
`D-POLARS-SIGINT`: polars installs a SIGINT handler with `SA_RESTART`, so an untimed lock acquire is
restarted and a timed one is not (a `wait(timeout=5)` is interrupted at once); every `rinchen`
command whose main thread blocks without a timeout is affected the same way, which is out of D6's
scope and was not examined for D4 and D5.

[CORRECTION: silent bit rot in a shard was neither "about 1%" nor an acceptable residual risk.]
The build record of the shard store said that about 1% of random byte flips of a Parquet shard
decode to different data and accepted it (no content checksum). The 2026-11-08 entry above kept
that stance ("silent bit rot inside a stored candidate passes validation", "no content checksum
was added") and measured 65,607 of 150,000 flips of a small shard decoding to a different frame
without error. The review repeated the measurement through `ShardStore.read` with the expected ids,
which is what `run_job` does: 63.7% of 1,500 flips of a 98,112-byte shard (512 rows of 4
candidates) decoded to different candidates with no error and only 0.3% were caught by the store's
own checks; on a 3,248-byte shard 27.5% and 0.9%. The shards are the resumable work files, so a flip
on disk between a crash and a resume wrote wrong text into the corpus with no trace. [DESIGN: Shard
content digest.] `ShardStore.write` stores the SHA-256 of the canonical JSON of the shard's ids and
candidates (the exact recipe is in the `_content_digest` docstring and the runbook) in the Parquet
key-value metadata under `rinchen.content_sha256`; `ShardStore.read` recomputes it from the decoded
rows, after the cheap structural checks, and a missing digest, an unreadable metadata block or a
mismatch is a `DataError` naming the file, so the shard is `Invalid` and is regenerated and counted
under `regenerated_invalid`. Shards written before this change have no digest and are regenerated
once. The digest is not part of `job_fingerprint` (its golden digest and the planted counts did not
change). It is not a signature: whoever can rewrite a shard can rewrite its digest, and the artefact
of record stays the manifest-hashed `corpus.parquet`. Cost if wrong: about 2 to 2.5 microseconds per
KiB of candidate text (0.7 ms for a 512-row shard of 4 candidates, against seconds to minutes to
generate it) and a regeneration of every shard that predates the change. What it does not touch: a
damaged file can still abort the process (a Rust allocation failure no handler catches).

[CORRECTION: `write_corpus` was not "open"; it was a bug that is now fixed.] The `D-POLARS-GLOB`
evidence above, the runbook, the design and STATE.md called the read-back of `write_corpus`
(`pl.scan_parquet(part_path)` without `glob=False`) a known hazard to avoid by keeping `[`, `*` and
`?` out of `--out-dir`. The review ran it: `--out-dir out[1]` generated every shard and then ended in
an uncaught `ComputeError` (exit 1, a traceback, no corpus, counts or manifest, and the earlier
manifest already removed by `begin_commit`); `out*` and `out?` happened to work. One argument,
`glob=False`, and tests (the writer with `[` in the directory and in the name, the CLI with
`out[1]`, `out*?` and `out[a-z]`). The real binary now writes all three outputs for `out[1]`,
`out*`, `out?` and `out[a-z]`. `write_corpus` is shared with D4; its other callers gain the same
protection.

[DESIGN: A scorer's wrong reply and its `OSError` are `UpstreamError`.] A QE scorer returning
text, `None` or a list as a score gave a bare `ValueError` or `TypeError` traceback from `float()`,
and a scorer that raised `OSError` (a missing weights file) was reported as "cannot write the synth
outputs". Now a score that is not a `numbers.Real` (text is rejected even when it reads like a
number), or a reply that is not a sized iterable, is an `UpstreamError` that names the pair and the
type and never a text, like the wrong count it joins (exit 4); NaN and the infinities stay numbers
and the threshold decides; an `OSError` of the scorer is an `UpstreamError` "the QE scorer failed:
...", chained; any other exception of the scorer still propagates unchanged. Cost if wrong: a scorer
that returns numeric strings on purpose is refused.

[EVIDENCE: Bit-flip sweeps with the digest, through `ShardStore.load`, 2026-11-08, one bit at a
time with a parent process restarting the child after an abort.] A 1,236-byte shard, all 9,888
flips: 941 digest mismatches, 4,136 unreadable, 249 structural, 4,561 leaving the rows unchanged, 0
different rows without an error, 1 abort. A 3,339-byte shard, all 26,712 flips: 7,461, 11,046, 232,
7,972, 0 and 1 abort. A 98,203-byte shard, 3,000 random flips: 1,843 digest mismatches, 1,117
unreadable, 8 structural, 32 unchanged, 0 different, no abort; with the digest comparison switched
off the same harness decoded 1,840 of them to different candidates. The two aborts are single
positions (SIGABRT, return code -6); the committed unit test sweeps every 7th byte of the first
shard, with a bit that cycles through all eight, which never pairs the aborting byte with the
aborting bit. Mutations checked by hand, one at a time: the bare wait (the new SIGINT test fails by
assertion, 3.01 s), no failure check in the wait, no `cancel_futures`, `glob=False` removed from the
read-back, the digest comparison skipped (six tests fail), the digest without the ids, with
`ensure_ascii=True`, with default separators, with sorted candidates, with the keys swapped, a
missing digest accepted, the metadata errors unhandled, the digest checked before the structure, a
numeric-text score accepted, the type check dropped, the pair index without the chunk offset, a
reply without `len` accepted, the `OSError` not converted, not chained, and a message that carries
the score; every one was caught except two equivalent mutants: `_INTERRUPT_POLL_S = 5.0`, which
passes because any timed wait is interrupted at once on this platform, and a wait without
`FIRST_EXCEPTION`, which only delays noticing a failure by at most 0.2 s (22 mutants).

[CORRECTION: STATE.md and the runbook said the final review and its fix wave had still to run.]
They have run; this entry and the two commits are the outcome, and the gate counts in STATE.md are
those of the fixed tree.

[DESIGN: Findings the fix wave leaves alone, one line each; the runbook lists them too.] M-1: the
`select._score_in_chunks` loop and count check duplicate D4's `filter/qe.py:_score`, so they must
change together. M-3: `plan.py` imports `Kind` from `select`, `Plan` has five fields for three
values, `bt.py` and `ft.py` exist for a kind-mismatch check that cannot fire, `ShardStore.invalid`
has no production caller and `cli/synth.py` imports `Releasable` from `rinchen.mine.embed`. M-4: a
`base_url` with a space inside the host is accepted (httpx normalises it). M-5: the `rinchen` console
entry has no CPython 3.14 assertion except `doctor` (as in D4 and D5), and an unparseable
`manifest.json` is overwritten by the D4 manifest store's design. Not examined: whether the polars
SIGINT behaviour shows in D4 and D5 commands, and a call into a C++ extension such as CTranslate2's
`translate_batch`.

[EVIDENCE: addendum to the D6 final review entry, same date, from its scoped re-review. The exit 130 delay of 0.27 s
is one sample: the delay depends on where the signal falls in the requests in flight, and five repeats gave 0.77 to
1.06 s, so the bound is the time of one request. The wait is polled every 0.2 s, which costs a pass over the
pending futures: 0.19 ms at the default shard_size of 512, 8.3 ms at 20,000. A future that a worker dequeued after
it was cancelled is `done` to the wait and raises when asked for its exception: the poll skips cancelled futures
(unreachable today, since only the finally block cancels, after the wait). Cost if wrong: none beyond the
documentation.]

## 2026-11-08: D7 multilingual ablations - the rulings, made without a review round

D7 (`src/rinchen/train/`, `src/rinchen/config/ablation.py`, `src/rinchen/cli/train.py`,
`src/rinchen/utils/logged_process.py`,
`configs/eole/{multilingual_tiny,ablation_tiny,multilingual_proxy,ablation_priors}.yaml`) filled
gaps in the master plan with the rulings below. **They were made by the builders and the controller
without a review round of their own, because the user's instruction for this stretch was "Proceed.
I'm going to sleep now. Don't stop until you finish everything. Commit everything."** None is an
approved decision; each lists what it costs if wrong so that it can be undone. (Four reviews of the
code did run during the build, over Task 1, Task 2, Tasks 3 to 5 and Tasks 6 to 8; they reviewed the
work, not these rulings, and no whole-branch review had run when this entry was written.) The design
with the full wording is `docs/superpowers/specs/2026-11-08-d7-multilingual-ablations-design.md`
(rulings R1 to R12, amended to match the finished code, and its section 8, "Amendments made while
building"; the numbers below are the same); procedures, measured numbers and the known limitations
are in `docs/runbooks/07_ablations.md`; the upstream facts are in `docs/VERIFY.md`, section "D7
findings". The work lives on branch `d7-multilingual-ablations`, stacked on `d6-synthetic-data` (cut
from it at `9973319`); it is not merged and not pushed.

[DESIGN: R1. Tag placements are `src` (the default) and `tgt`; a source suffix is out.] The master
plan names "tag placement" as an axis without saying where a tag may go. A tag on the source is the
primary placement (no forcing, no dedup loss, a validation loss that compares across variants); a
tag on the target trains but needs the tag forced at prediction (the harness does it). A suffix
needs Eole's separate `suffix` transform, which raises `KeyError: ''` at the start of training
unless a workaround corpus is configured, so `tag_placement: [suffix]` is a `ConfigError` that names
the bug. Cost if wrong: a suffix experiment needs a third value of `TagPlacement`, the `suffix`
transform in `eole_data` and the workaround corpus (the spike found it works with that workaround).

[DESIGN: R2. Corpora are `(direction, group)` pairs, `auth` and `bt`; other origins are counted, not
used.] `bt` is origin `bt`; `auth` is human, web, mined and converted. Rows of origin `ft`,
`synth_llm` and `synth_nmt` are left out and counted by origin, because forward-translated data
enters training with D8 and D10. Only rows with `split == "train"` train; every other row is counted
once under the first reason that applies, in the order `not_train`, `filtered`, `excluded_origin`,
`empty`, `line_break`, so `rows_read = rows_kept + the exclusions` always holds. Duplicate ids are
kept and counted. Cost if wrong: the origin-to-group table is one place (`mixture.py`,
`EXCLUDED_ORIGINS`, `BT_ORIGIN`); D8 extends it.

[DESIGN: R3. Temperature acts on directions.] The mixing probability of a direction is `rows ** T`
over the directions (`rinchen.utils.sampling.temperature_weights`, D3), where `rows` counts its
authentic and used bt rows; inside a direction the corpora share its probability in proportion to
their rows. So `bt_ratio` fixes the bt share of a direction (`r / (1 + r)` when enough bt rows
exist) and `temperature` the balance between directions. Cost if wrong: a per-corpus temperature
would let the bt share move with the temperature; it is a different rule in
`sampling.mixing_probabilities`.

[DESIGN: R4. The bt subsample is `min(available, round(r * authentic rows))` rows by a seeded hash
rank.] Python `round` (a half goes to the even integer); the rows with the smallest
`sha256(f"{seed}\t{id}")`, ties by id, source, target, so the subsample does not depend on the order
of the rows or files or on the polars version. A direction with bt rows and no authentic rows asks
for none and trains nothing; a shortfall is recorded (`bt_available`, `bt_requested`, `bt_used`),
never an error; no authentic row at all is a `DataError`. Cost if wrong: a reader who expects the
ratio to be exact finds a shortfall in `results.json` (in the tiny run the ratio 0.5 asked for 600
rows in two directions and 300 existed).

[DESIGN: R5. Integer weights are at least 1 and sum to `weight_total`; the replacement of "effective
epochs" by `upsampling`.] Eole's `weight` is the number of consecutive examples per round of a round
robin, so the probabilities become integers by largest-remainder rounding (a minimum of 1; ties by
corpus name, never by input order; a larger probability never gets a smaller weight), and
`weight_total` must be at least the number of corpora (a `ConfigError` names the number needed). The
first draft recorded "effective epochs" per corpus; an epoch count needs the batch composition,
which Eole decides, so each variant records `upsampling = (w / sum w) / (rows / sum rows)` instead
(1.0: drawn as if every row counted alike). Cost if wrong: with 32 corpora and `weight_total` 100
the minimum weight floors the small corpora (computed in the runbook: at temperature 1.0 half of
them sit at weight 1 and the smallest is drawn 5.5 times its proportion; at 200 the worst error
falls from 4.55 to 1.78); it was never trained. The spike's 7/13/7/6/6 for its mix at T=0.3 was the
smallest-sum vector within a cap; the ruling gives 18/34/18/15/15.

[DESIGN: R6. The training corpora are plain text files without tags; the tags are Eole `prefix`
configuration.] One sentence per line, LF only, rows sorted by id, source, target; a `<name>.json`
sidecar holds the row count and the SHA-256 of both files and a corpus is reused only when it
matches. Authentic corpora are written once and shared by every variant; the bt subsample once per
ratio and seed. So placement is a config difference, not a data difference. Cost if wrong: disk for
every distinct bt ratio and seed (634 MB for 902,400 rows at one ratio).

[DESIGN: R7. The validation corpus is pre-tokenised by the harness and carries no prefix keys.] One
concatenation of the dev sets of all directions, sorted by direction, `transforms: []`, the
direction tag as one piece on the source (`src`) or the target (`tgt`). Eole 0.6.4 copies a `valid`
prefix into the checkpoint, where it overrides the prediction (`D-EOLE-PREFIX-OVERRIDE`); this
avoids it for both placements. Dev rows are the `dev` split with no `filter_reason`, at most
`dev_max_lines` per direction by id, source, target. Cost if wrong: Eole's own validation loss and
accuracy are not comparable across placements (the target tag adds an ambiguous token); selection
uses the dev chrF the harness computes, not the validation numbers in the log.

[DESIGN: R8. One `eole predict` call per variant; corpus chrF and BLEU per direction.] Greedy
(`beam_size` 1) on the CPU over the concatenated dev sources, split back by direction; the macro
mean and the minimum chrF over directions; `tag_leaks` counts hypotheses that hold the direction tag
of any registry language or `<bt>`. Cost if wrong: greedy scores are lower than beam scores, which
is fine for comparing variants and not for reporting a model; the evaluation runs on the CPU even
for a GPU-trained model (slow for the proxy size, not timed).

[DESIGN: R9. `source_collision_rate` is recorded, not gated.] The share of kept training rows whose
source text also occurs with another target language, over all kept rows before any bt subsample:
the exposure to Eole's same-source minibatch dedup with the tag on the target. Cost if wrong: it is
a proxy; a real corpus with partial overlap may lose less than the toy numbers suggest, and a user
can ignore a number they do not understand. It is 0.0 on the toy training set, so the gate does not
exercise the hazard.

[DESIGN: R10. Resume by fingerprint, checkpoint digest and `train.json`.] The fingerprint is the
SHA-256 of the canonical JSON of the rendered config text, the digests of every corpus file and of
the validation files, of the raw dev texts and of the SentencePiece model, the thread count, the
versions of Eole, torch, SentencePiece and sacrebleu, and the placement. A variant is complete only
if its `result.json` parses strictly, says `ok`, holds the fingerprint, scores the planned dev sets
line for line AND its checkpoint file has the recorded SHA-256; it is then reused. `train.json`
(fingerprint, checkpoint digest, seconds) is written right after a training so a failed or
interrupted evaluation does not cost the training. Cost if wrong: the config text holds absolute
paths, so moving the output directory, the checkout or the SentencePiece model retrains every
variant, and so does a different `dev_max_lines`; a new torch or Eole version does too, on purpose
(a model changes with them).

[DESIGN: R11. Every Eole config passes `validate_train_config`.] Per-level allow-lists (15 top-level
keys, 6 per corpus, 2 + 2 + 2 for the transforms, 9 for the model, 5 for each of encoder and
decoder, 2 for the embeddings, 27 for training), each checked against the installed Eole's fields by
a test; the 16 forbidden keys (relative, rotary, sliding-window and grouped-query attention) at any
depth; `model.architecture: transformer`; `SinusoidalInterleaved`; `hidden_size` equal to
`word_vec_size`; `overwrite` and `share_vocab` true; plain data only; no `$`; every corpus with
`transforms` and both prefixes; `valid` without prefix, weight or transforms. Cost if wrong: a key
an ablation legitimately needs is refused until it is added to the list (one line and a drift test);
the validator does not check the types of values, so a wrong type fails in `eole train` after
`build_vocab`.

[DESIGN: R12. The shared Eole machinery moved out of `script_conv`; D5's behaviour did not change,
with two exceptions that are rulings below.] `run_logged` is `rinchen.utils.logged_process`; the
allow-lists, `validate_train_config`, `merge`, `render`, `read_config`, `dump_config` and
`reject_dollar` are `rinchen.train.eole_cfg`; `run_training`, `predict_tokens` and the environment
helpers are `rinchen.train.eole_run`; D5's wrappers keep their signatures and error wording. A
differential probe of the old and new D5 runner and prediction over 84 scenarios was byte-identical
except for two error paths, which the second task then restored. Cost if wrong: D5 imports from
`train`; a D5-only change must now respect two other users.

[DESIGN: Controller ruling 1. The matrix `overrides` may not set the keys the run sets itself.]
`data`, `src_vocab`, `seed`, `training.model_path` and `transforms_configs.sentencepiece`, also by
replacing or deleting a parent section, are a `ConfigError` naming the key at plan time (so also for
a dry run); every other override merges last and a value of `null` deletes a key. Cost if wrong: an
operator who wants to point a variant at another vocabulary cannot do it from the matrix; the matrix
has no axis for an Eole key either, so comparing two learning rates takes two matrix files and two
output directories.

[DESIGN: Controller ruling 2. Completeness includes the checkpoint digest, and `train.json` stays
after success.] A `result.json` alone does not make a variant complete: the checkpoint file must
exist with the recorded SHA-256 (a truncated or replaced checkpoint retrains that variant only).
`train.json` is kept beside `result.json` after a successful run, so a missing or corrupt
`result.json` re-evaluates the trained checkpoint (the outcome `evaluated`), while a tampered
`train.json` or checkpoint retrains. Four existing unit tests that asserted a retraining when
`result.json` was not trusted now assert no training call and one evaluation call (`train.calls ==
[]` and `evaluate.variant_ids == [id]`). Cost if wrong: a stale `train.json` beside a deleted
checkpoint is ignored and removed at the next training; the extra file is 240 bytes.

[DESIGN: Controller ruling 3. The plan decides the corpus sizes with one function and validates what
would be rendered.] `planned_corpus_sizes` and `planned_corpus_files` are the functions
`build_corpora` is built from, so a plan cannot disagree with a build, and the plan builds every
variant's Eole config in memory, from the paths its corpora will have, and validates it, together
with the SentencePiece model (it loads and has every direction tag as one piece). So a typo in an
override fails the dry run with exit 2 instead of after the data directory has been written and the
old manifest removed. Cost if wrong: the plan and the preparation share one config builder
(`ablation_config`); a change to how the config is made must go through it.

[DESIGN: Controller ruling 4 (Ruling X). `run_logged` waits in slices of 0.2 s; four D5 test
assertions were edited.] Once polars is imported (every `rinchen` command has it) a blocking untimed
wait on the main thread is not interrupted by Ctrl-C (`D-POLARS-SIGINT`): without `--timeout`,
Ctrl-C during `eole train` was raised only when Eole ended (25.9 s after the signal through the real
binary; 0.47 s with `--timeout 600`). The wait is now made in slices of `_INTERRUPT_POLL_S` = 0.2 s
that never run past a monotonic deadline; the timeout, the kill of the process group and the
messages are unchanged. Four D5 tests asserted the argument that reached `Popen.wait`, which was
`None` for "no timeout" and is now the slice, so the constraint that allowed only import and patch
edits to D5's tests could not hold. Edited, with the new assertion: in
`tests/unit/test_script_conv_train.py`, `test_run_training_does_not_bound_the_commands_unless_asked`
(`waits == [None, None]`, a line that literally asserted the defect, now `[LONG_POLL_S,
LONG_POLL_S]`), `test_run_training_bounds_each_command_by_the_timeout` (`[7.5, 7.5]`, now
`[pytest.approx(7.5, abs=0.5)] * 2`) and
`test_predict_bounds_the_command_by_the_timeout_only_when_asked` (`[None, 3.5]`, now `[LONG_POLL_S,
pytest.approx(3.5, abs=0.5)]`); in `tests/unit/test_script_conv_export.py`,
`test_export_ct2_bounds_the_converter_by_the_timeout_only_when_asked` (`[None, 12.5]`, now
`[LONG_POLL_S, pytest.approx(12.5, abs=0.5)]`). The two fixtures monkeypatch `_INTERRUPT_POLL_S` to
`LONG_POLL_S`, defined in `tests/support/fake_process.py`. The new assertions still distinguish "no
timeout" from "bounded by T": `LONG_POLL_S` is `math.inf` (the review found that a first value of
1000.0 let a mutant with a limit of 3600 survive), so a recorded wait of infinity means no timeout
was asked for and a finite value near T means one was, and a mutant that bounds the unbounded wait
or drops the bound of the bounded one fails them; the reviewer confirmed that no other assertion of
the D5 tests changed, and the real SIGINT test of `tests/unit/test_utils_logged_process.py` fails by
assertion on the old code. Cost if wrong: a 5 Hz wake-up of the main thread while a command runs,
and four D5 assertions that now describe the poll; the slice is one constant.

[DESIGN: Controller ruling 5. Glob characters in the checkpoint directory are refused in the plan,
in `run_training` and in `predict_tokens`, which makes D5 fail earlier.] Eole loads a checkpoint
through `glob(<model_path>/model.*.safetensors)`, so `[`, `]`, `*` and `?` in that directory find no
shard or a sibling's shards (`D-EOLE-GLOB-CKPT`). `reject_glob_characters` raises a `ConfigError`
naming the path and the character; `plan_ablation` applies it to the absolute output directory (so a
dry run refuses too), `run_training` to `training.model_path` and `predict_tokens` to the checkpoint
directory. No D5 test pinned acceptance, so D5's converter training and prediction now refuse such a
run directory at once where they used to train and then fail at the prediction. Not guarded: D5's
`export_ct2`, handed a checkpoint directory, fails the same way; D5 was not changed. Cost if wrong:
a user who put the output in a directory with brackets must rename it.

[DESIGN: Controller ruling 6. SIGTERM and SIGHUP end the run like Ctrl-C.] `run_logged` starts its
child in a session of its own (so a timeout or Ctrl-C can kill the whole group), which also keeps a
SIGTERM or SIGHUP sent to `rinchen` from reaching it while the default action of both ended the
parent at once: `eole train` of hours was left running. While `run_logged` waits on the main thread
both signals now raise `SystemExit(128 + number)` (143 and 129), which takes the existing path that
kills the child's group; the previous handlers are restored afterwards. SIGKILL cannot be caught and
a child can be left running after it. Cost if wrong: handlers are set only on the main thread;
`rinchen` run from another thread gets no handler.

[DESIGN: Controller ruling 7. The module splits of the build are pure moves.] The first plan had one
`mixture` module and one `ablation` module; they did five and four jobs. After the review of Tasks 3
to 5, `mixture.py` was split into `mixture` (classification, statistics, the loaders),
`corpus_files`, `eole_data` and `validation_corpus` (a pure move: 184 tests collected before and
after; the public `direction_tag`, `replace_file` and `lines_as_bytes` made their way into
`config/languages.py` and `utils/part_file.py`). After the review of Tasks 6 to 8, `ablation.py`
(531 lines) was split into `ablation`, `ablation_prepare` and `ablation_records` (5,787 tests
collected before and after) and the fixes added `ablation_layout` and `ablation_config`. Cost if
wrong: more files than a first reading of the master plan suggests (it names `train/eole_cfg.py`
only); the spec's module table lists all of them.

[DESIGN: Other decisions made while building, one line each.] `render` takes the output path as a
third argument (the master plan's sketch is `render(base, overrides)`), so a variant's config lands
in its directory. The dry run prints one fact per line and ends with `nothing was trained and
nothing was written`. Progress lines go to standard output (D6 uses standard error). `begin_commit`
runs right before `run_ablation`, not before the first output, because the variants' files are
outputs of the run. The result tables are written when every variant is done, not per variant, so
after a failed rerun they may be older than the variants. `evaluate_variant` turns a line feed in a
decoded hypothesis into a space and refuses a direction name with `/`, `\` or NUL. The fingerprint
lists versions and the raw dev digests, not the code version, so a rerun after a commit does not
retrain. The toy data is en-mn 1,200 and ru-mn 600 training pairs with 300 bt pairs per language
pair and a SentencePiece vocabulary of 640 (the project's `train_spm` cannot go below about 458
pieces on this text). The matrix defaults (`weight_total` 100, `dev_max_lines` 1000, `max_variants`
64, seed 1234) and every value of the proxy base and of `ablation_priors.yaml` are priors. Cost if
wrong: each is one constant or one YAML value.

[EVIDENCE: Weight semantics, from the spike (`q3_weights.py`, `q3/q3_output.txt`, Eole 0.6.4, CPU,
2026-11-08).] `Dataset.weight` is an integer and `WeightedMixer`
(`eole/inputters/dynamic_iterator.py`) cycles over each corpus name repeated `weight` times; an
exhausted corpus restarts. With Eole's own classes the first 1,000 examples of corpora of 300 and
3,000 lines with weights 1:3 were 250/750, with 3:1 750/250 (the small corpus restarted twice), with
1:1 500/500, with 2:6 the same as 1:3; 50 and 3,000 lines with 1:3 gave 250/750 (five passes over
the small corpus); weights 0:1 gave 0/1,000; 100, 1,000 and 10,000 lines with 1:2:5 gave
125/250/625. End to end, five corpora with weights 3/1/2/2/2 were passed 87/29/29/29/29 times. In
the real 150-step run of the review of Tasks 3 to 5 the per-corpus sentence counts equalled the
weights 22/6/22/6/15/7/15/7. Not run: the sum of the weights against `bucket_size` (read from the
code), and `num_workers` above 1.]

[EVIDENCE: Unknown keys (spike `q5_probe_part1.txt`).] `eole train` rejects an unknown key at every
level (`ValidationError ... training.bogus_key ... Extra inputs are not permitted
[type=extra_forbidden]`); `eole build_vocab` ignores an unknown top-level, `training` or `model` key
(it fails later at `eole train`, after the vocabulary run, which counts the whole corpus) and
rejects an unknown `data.<corpus>` or `transforms_configs` key. A missing `src_prefix` or
`tgt_prefix` on a corpus that uses `prefix` is `TypeError: object of type 'NoneType' has no len()`
already at `build_vocab`. Through the real binary a matrix with `overrides: {training: {train_stepz:
5}}` and `--dry-run` printed `error: invalid Eole config: training.train_stepz: unknown key in
training` and exited 2.]

[EVIDENCE: The suffix transform (spike `q5_probe_part1.txt`, source read in
`eole/transforms/misc.py`).] `SuffixTransform.get_specials` calls `src_specials.remove("")` and
`tgt_specials.remove("")` without the `if "" in ...` guard the prefix transform has; `build_vocab`
is ok and `eole train` fails with `KeyError: ''` unless some corpus that uses `suffix` has an empty
suffix. With the workaround (the `valid` corpus on `[sentencepiece, suffix]` with empty suffixes)
the three placements trained to dev chrF of 98.6 and more on the toy at 2048-token batches and 1,000
steps (source tag 99.4/99.95/100/99.4, target tag 99.3/99.95/99.3/98.6, source suffix
99.7/100/99.5/99.7; one seed, differences are noise).]

[EVIDENCE: Same-source minibatch dedup (spike `q4_dedup_shared_source.py`, `q4_count_dedup.py`;
source read in `eole/inputters/dynamic_iterator.py`, `_should_deduplicate`).] For the training task
Eole skips an example whose tokenised source text equals one already in the same minibatch. Counted
with Eole's own iterator on multi-parallel toy data (identical Mongolian sources in the mn-en and
mn-ru corpora): tag on the source 0.00% dropped; tag on the target 1.94% at 1,024-token batches,
35.6% at 4,096, 50.0% at 16,384. Two trainings of the spike saw 27,861 sentences with the tag on the
source and 27,327 with the tag on the target (1.9% fewer). The toy training set of the gate has
`source_collision_rate` 0.0, so the gate does not exercise this.]

[EVIDENCE: The `valid` prefix leak (spike `q1_native_predict.py`, D5's `D-EOLE-PREFIX-OVERRIDE`).]
With a prefix-free `valid` corpus Eole's own prediction equalled the pre-tokenised path on 100 of
100 lines; with a D5-style `valid` prefix `<2eng_Latn>` its Mongolian to Russian prediction returned
English (chrF 1.5) while the pre-tokenised path scored 99.4. A valid corpus written as text with the
tag in it trains but gives an extra leading `▁` against training and, with the tag on the target, a
bad validation (perplexity 15.7, accuracy 61%). The saved `config.json` of both checkpoints of the
tiny ablation has `data.valid` without prefix keys and without transforms (asserted by the STOP-gate
test).]

[EVIDENCE: Thread determinism and the init plateau (spike `q6_output.txt`, `seed_table.txt`,
`timing_table.txt`).] A rerun with the same seed and two threads gave the same safetensors SHA-256
(prefix `09aaf28a`), also across different `train_steps` and `valid_steps` and under machine load; 4
threads gave another hash and chrF of 64.8/63.3/53.4/38.9, 1 thread 47.2/61.4/45.5/38.9. In the
build, the retrained checkpoint of the tiny ablation after a truncation had the SHA-256 of the first
one (`e126d00d...a00c`) and the same scores. Eole's default initialisation (uniform in -0.1..0.1 on
every parameter, layer norms included) trains from random init but with a plateau until about step
600 to 1000 at 1,024-token batches and a seed-dependent breakout (first step with every direction at
chrF 60 or more over four seeds: 1000, 800, 1000, 1200); `xavier_uniform` reached 97.9 in the worst
direction by step 300 at 2,048-token batches. Not adopted; toy only.]

[EVIDENCE: Glob of the checkpoint directory (review of Tasks 6 to 8; `docs/VERIFY.md`
`D-EOLE-GLOB-CKPT`).] Through the real binary `--out-dir 'out[1]'` trained a variant for 18 s and
then `eole predict failed ... No safetensors file found`; `--out-dir 'out*'` passed because the
pattern matched four sibling checkpoint directories. At the Eole level a real 150-step checkpoint
predicted 400 lines from a plain directory, failed with `ValueError: No safetensors file found` from
`out[1]/ckpt`, and predicted 400 lines from `sib/out*/ckpt` beside three siblings (the glob returned
4 files). The CTranslate2 conversion of a copy in `out[1]/ckpt` fails alike. After the guard the
real binary exits 2 in 0.7 s and creates nothing for `out[1]`, `out*`, `out?` and `out]`.]

[EVIDENCE: The SIGTERM orphan and the Ctrl-C delay (review of Tasks 6 to 8, the Task 8 implementer;
`docs/VERIFY.md` `D-LOGGED-TERM`, `D-POLARS-SIGINT`).] Through the real binary, a SIGTERM 4 s into
`eole train` ended `rinchen` with status -15 after 0.03 s and left `eole train`, its resource
tracker and a spawned worker running (killed by hand); after the guard SIGTERM gave exit 143 in 0.26
s and SIGHUP exit 129 in 0.16 s with no process of the run left, no manifest and no `.part` file.
Without `--timeout` Ctrl-C was raised 25.91 s after the signal (26.8 s in the first measurement) and
with `--timeout 600` after 0.47 s; after the polled wait it is 0.19 s and 0.17 s (0.26 to 0.36 s in
the review, during training and during prediction); a real-SIGINT test fails by assertion on the old
code (5.61 s after the signal).]

[EVIDENCE: The STOP gate and the measurements, 2026-11-08 (Intel Core Ultra 7 265K, 20 cores, 46
GiB, CPython 3.14.10, eole 0.6.4, torch 2.14.1+cu130, sentencepiece 0.2.1, sacrebleu 2.6.0, polars
2.1.0; GPU hidden).] The tiny ablation through the real binary: 72.7 s wall, training 33.7 and 33.0
s, evaluation 2.6 s each, peak resident memory 920 MiB; weights 22 6 22 6 15 7 15 7 = 100 over 1200,
300, 1200, 300, 600, 300, 600, 300 rows; dev chrF (eng-khk, khk-eng, khk-rus, rus-khk) 98.89, 99.23,
97.76, 96.25 for the source tag and 99.72, 98.24, 98.32, 98.33 for the target tag; BLEU 98.85,
99.01, 98.60, 96.19 and 99.67, 98.01, 98.79, 98.43; no tag leak; the review re-ran it and got
identical numbers. A rerun reuses both in 0.76 s, with one `result.json` deleted evaluates in 3.37
s, with a checkpoint truncated retrains in 37.75 s to the same SHA-256. The dry run takes 0.71 s.
Mixture cost on synthetic rows of about 175 characters a side, a fresh process each: 1,000,000 rows
(902,400 kept) load in 0.32 s (peak 1,550 MiB), dev rows in 0.06 s, `build_corpora` 2.27 s (634 MB,
peak 1,661 MiB), the same call on an unchanged directory 2.27 s, a 12-variant plan 0.61 s (1,610
MiB); 4,000,000 rows load in 1.09 s (4,758 MiB) and build in 11.43 s (2.7 GB, 5,127 MiB). Weight
tables for three illustrative mixes are in the runbook. Mutation checks by hand, one mutant at a
time, per task as recorded during the build: Task 1 30, Task 2 122 (one equivalent survivor), Tasks
3 and 4 all but 3 equivalent ones, Task 5 about 135, Interlude A about 45, Task 6 34 and more, Task
7 about 190, Task 8 54 for the CLI, the `run_logged` fix all but one equivalent (a poll of 60 s),
Interlude B about 95; the reviewers' own: 19 of 20, 17 of 17, 29 of 29.]

[EVIDENCE: Not verified.] No real corpus and no GPU run exist; the proxy base and the priors matrix
were only validated against the schema (Eole's `TrainConfig` accepts them); the weight resolution
for many directions is computed, never trained; `num_workers` above 1 with weights, `xavier_uniform`
at scale, Eole-native prediction and the CTranslate2 export of a tag-on-target model, Windows and
macOS were not run; the mixture beyond 4,000,000 rows is an extrapolation (about 1.05 GiB for each
further million rows, 46 GiB near 44M rows); concurrent runs into one output directory are not
guarded; SIGKILL of `rinchen` leaves the child running; only one seed was run anywhere; the value
types of known Eole keys are not validated; Eole's `filtertoolong` drops are not counted in the
statistics.]

[DESIGN: Gates.] Open, because they need real data, a GPU or a reader: any ablation on real corpora
(which temperature, bt ratio and tag placement is better), a measured proxy config, the throughput
and memory of a real training, a streaming loader for hundreds of millions of rows, Eole-native
prediction and the CTranslate2 export of a model trained with the tag on the target. The D3 gates
assigned to D4, the D5 gates that need real text and the D6 gates stay open as recorded there. D8
(Teacher training) takes the config renderer, the runner, the mixture and the weights, the
evaluation harness and the resumable run from here.

## 2026-11-08: D7 final review - corrections and fixes

An independent review of the whole branch (`9973319..87cabe2`, 20 commits, read-only) found no
Critical problem, one Important problem and seven Minor ones. It reproduced the gate (5944 passed, 1
skipped, 10 deselected in 387 s), the STOP gate twice, the D5 regression sets and every number the
earlier entries claim. This entry records what it found, how the findings were fixed (commits `D7:
final review fixes` and `D7: final review fixes, docs`), and CORRECTS in words the statements of the
entries above that the review showed false. The entries above are not rewritten.

[EVIDENCE: Corrections of earlier statements.] (1) Controller ruling 6 above, `docs/VERIFY.md`
`D-LOGGED-TERM` and the spec said that, while `run_logged` waits, SIGTERM and SIGHUP raise
`SystemExit(128 + number)` and that this makes a signal to `rinchen` end the run cleanly. For a
process that was started with SIGHUP ignored it was wrong, and worse than before ruling 6: the
handlers REPLACED the ignore. `nohup` (the usual way to leave a multi-hour `rinchen train ablate`
running over SSH) starts a command with SIGHUP ignored, and the disposition is inherited across
`exec` and left alone by CPython; the handler of ruling 6 turned the hangup that `nohup` exists to
absorb into exit 129 and a killed `eole train`. Before ruling 6 the same run survived the hangup.
Measured through the real binary (`setsid nohup .venv/bin/rinchen train ablate`, one 150-step
variant, SIGHUP sent to `rinchen` 4 s into `eole train`): the code of ruling 6 ended with exit 129 0.17
s after the signal and no manifest; the fixed code finished the run, exit 0, with the manifest and the
tables written 9.55 s after the signal. D5's `convert-script train` and `export_ct2` share `run_logged`
and had the same regression. (2) The docstring of `train/ablation_layout.py` said `train.json` is kept
"until it is scored". Controller ruling 2 and the code keep it after success (a missing or corrupt
`result.json` only re-evaluates); the docstring was wrong, not the rule. (3) Runbook 07 said that
`rinchen.train` imports `spm`. It imports nothing from `rinchen.spm`; the third-party `sentencepiece`
package is imported by `validation_corpus.py` and `evaluate.py` only (the spec's layering sentence
was right). (4) The runbook of D5 (05) said that SIGTERM and SIGHUP to the CLI are not handled and that
the Eole child keeps running; that stopped being true with ruling 6 and is corrected there.

[DESIGN: The fixes.] I-1: `_exit_on_termination_signals` reads the current handler of SIGTERM and of
SIGHUP; one that is `signal.SIG_IGN` is left alone (no handler installed, nothing to restore for it),
any other is replaced by the handler of ruling 6 and put back afterwards, as before. Reason: an
inherited ignore is a request not to die on that signal. Cost if wrong: a process that ignores SIGTERM
on purpose cannot be stopped by SIGTERM while `run_logged` waits (Ctrl-C and SIGKILL still can). M-5:
`bt_ratio` is at most 1000 (`MAX_BT_RATIO`), a `ConfigError` above it; `1e308` had made `bt_requested`
a 309-digit integer in the dry run and in `results.json`. Cost if wrong: a ratio above 1000 needs the
constant raised. M-6: the evaluation of a variant whose score is NaN or infinite raises an
`UpstreamError` that names the variant and the score, after the evaluation and before any result file
is written (the `train.json` stays, so a rerun only evaluates again); `results.json` is also written
with `allow_nan=False`. Unreachable with sacrebleu today (chrF and BLEU are never NaN). M-1 and M-2:
the docstring and the runbook sentence above. M-3: the runbook says that variant directories of an
earlier matrix stay in `variants/` of the same output directory, are not reported, not hashed and not
in the manifest (delete them by hand or use a fresh output directory); no code change. M-4: runbook 05
records that `run_training` and `predict` refuse `[`, `]`, `*` and `?` in the run directory since D7
and that `export_ct2` is still unguarded. M-7: runbook 07 and `D-LOGGED-TERM` say that a shell
background job (`&` without job control) starts with SIGINT ignored (POSIX), so Ctrl-C style SIGINT
does nothing to it; measured: a SIGINT 4 s into `eole train` of such a run was not noticed (exit 0,
9.56 s later). One existing test was edited for the bound: `test_an_exponent_with_a_point_and_a_sign_
is_a_number_in_yaml` used `2.5e+3` for both axes and now uses `2.5e+2`, which keeps the point of the
test (a signed exponent is read as a number) inside the bound.

[EVIDENCE: Tests and measurements of the fix wave.] The new tests fail on the code before the fix (the
signal tests by assertion, bounded by the 2 s command; the bound and the non-finite tests with `DID NOT
RAISE`) and pass after it; each was mutation-checked (the signal guard 5 mutants, the bound 4, the
non-finite score 8, all killed). The full gate: 5966 passed, 1 skipped, 10 deselected (baseline 5944) in 6 min 37 s (6 min 40 s with the checks before it).

[EVIDENCE: Known and left open.] A handler of the caller's own for SIGTERM or SIGHUP is suspended while
`run_logged` waits (the exit is raised instead); no caller has one. SIGKILL of `rinchen` leaves the
child running (`D-LOGGED-TERM`). D5's `export_ct2` is still unguarded against glob characters in the
checkpoint directory. Concurrent runs into one output directory are still not guarded. Every item of
the section above that lists what is open or not verified stays so.

[EVIDENCE: addendum to the D7 final review entry, from its scoped re-review. (1) The correction list there named runbook 05 as the
only document that said SIGTERM and SIGHUP are not handled; the D5 design spec said so too and now points at `D-LOGGED-TERM`.
(2) `signal.getsignal` returns None for a handler installed outside Python (a preloaded library, an embedding program), and
`signal.signal(n, None)` raises TypeError: `run_logged` restored such a handler with that call and so could fail after the wait.
Now a signal whose handler is None is left alone, like an ignored one, and the handlers are installed inside the try block so a
failing second install still restores the first (two tests; reachable only through LD_PRELOAD or embedding, found by the
re-reviewer with a preloaded shared object). (3) The runbook's list of outside imports of `train` named only sentencepiece; it
now lists polars, pyyaml, pydantic and sentencepiece (sacrebleu comes in through `metrics`). Cost if wrong: none beyond the documentation.]

## 2026-11-08: D8 teacher training - the rulings, made without a review round

D8 (`src/rinchen/train/{checkpoints,curves,average,launch,gpu_rules,launch_command,run_lock}.py`
with the extensions of `eole_cfg.py`, `eole_run.py` and `corpus_files.py`,
`src/rinchen/utils/{logged_process,integers}.py`,
`src/rinchen/cli/{train,train_runs,train_common}.py`, `configs/eole/{teacher,teacher_tiny}.yaml`)
filled gaps in the master plan with the rulings below. **They were made by the builders and the
controller without a review round of their own, because the user's instruction for this stretch was
"Proceed. I'm going to sleep now. Don't stop until you finish everything. Commit everything."** None
is an approved decision; each lists what it costs if wrong so that it can be undone. (One review of
the code ran during the build, over Tasks 1 to 6; it reviewed the work, not these rulings, and no
whole-branch review had run when this entry was written.) The design with the full wording is
`docs/superpowers/specs/2026-11-08-d8-teacher-training-design.md` (S1 to S10, amended to match the
finished code, and its section 8, "Amendments made while building"; the numbers below are the same);
procedures, measured numbers and the known limitations are in `docs/runbooks/08_teacher.md`; the
upstream facts are in `docs/VERIFY.md`, section "D8 findings". The work lives on branch
`d8-teacher-training`, stacked on `d7-multilingual-ablations` (cut from it at `b1a8a47`); it is not
merged and not pushed.

[DESIGN: S1. `launch` is a separate runner from D7's `run_training`.] `run_training` treats any
fresh `model.00.safetensors` as success and cannot resume; D7's short ablation runs keep it, and
both use `run_logged`. `run_logged` gained `inherit_fds` for S2's lock. Cost if wrong: making
`run_training` resumable instead would give the short ablation runs the launcher's strictness
(identity, lock, attempt files) they do not need; the two runners share the environment helpers, so
merging them later is a move, not a rewrite.

[DESIGN: S2. Run identity: the hash of the effective config without `train_steps` and `train_from`,
kept in `launch.json`; enforced only when step directories exist.] With Eole's default `reset_optim:
none` the optimiser options come from the CHECKPOINT's config, so a resume YAML with a new learning
rate or warmup is silently ignored (the spike's resume logged the old rate whatever the new file
said); the identity makes such a launch a `ConfigError`. The two excluded keys are the length of the
run, which a longer run extends, and the checkpoint a resume continues. A different GPU count is a
different identity (the batch changes). Paths are in the config and so in the identity. A record
without step directories is rewritten (a fresh start); step directories without a record are
refused. Cost if wrong: a change the identity forbids (a longer warmup in the middle of a run, a
corpus path moved) needs a new run directory and cannot continue the old checkpoint; a looser
identity would let Eole ignore the change without a word. D9, which will continue OUR teacher with
other data and a new schedule, needs a conscious extension of this rule.

[DESIGN: S3. A resume names the newest COMPLETE `step_N`; `train_from` exists only in the config the
launcher renders; every attempt has its own config `launch/train-<k>.yaml` and log
`launch/attempt-<k>.log` (controller ruling).] Eole's saver is not atomic, so a resume never goes
through `model_path`; `validate_train_config(resumes=True)` accepts `train_from` only as
`<training.model_path>/step_<N>` and accepts a config without it (the flag means "allowed"). The
plan's single `resume-<N>.yaml` became one effective config per attempt, so `command.sh` and the
record name exactly what ran and an earlier attempt's log is never read as a later one's. Cost if
wrong: two files per attempt in `launch/`, and a launch that reads the last attempt's log (not all
of them) for the NaN and early-stop checks.

[DESIGN: S4. Outcomes `completed`, `early_stopped`, `already_complete` (and `written`); an explicit
positive `training.train_steps` is required; the NaN gate runs when an attempt ends.] A live abort
on NaN (reading the log while Eole runs) was ruled out of scope; the gate also refuses to resume or
finish a run whose last attempt logged a non-finite figure at or before its newest complete step. An
early stop that lands exactly on `train_steps` is `early_stopped` for the launch that ran it and
`already_complete` for the next. Cost if wrong: a teacher that goes NaN at hour 2 of 60 runs on
until the operator notices; a config that leaves `train_steps` to Eole's default of 100000 is
refused instead of run.

[DESIGN: S5. The GPU rules of `launch` are `world_size`/`gpu_ranks` consistency, `timeout` at least
3600 and at most 80 corpora for `world_size` above 1, one early-stopping criterion, and
`keep_checkpoint` at least `early_stopping + 1`.] They turn Eole's silent cases (a no-op, a
60-second collective timeout, a gather buffer, a deleted best checkpoint) into a message before a
GPU node is spent. The 3600 and the 80 are the design's margins, not measured limits (the pickle was
4.0 KB at 100 corpora, the buffer 4096 bytes). Cost if wrong: a legitimate config with 90 corpora,
or a validation that is known to take under an hour and a timeout of 1800, is refused and needs the
constant changed.

[DESIGN: S6. `command.sh` is deterministic text; `--write-only` writes it, the attempt's config and
`launch.json`; the script ends with `2>&1 | tee -a launch/attempt-<k>.log`.] The record lets a run
started by hand be resumed by `launch` (Task 5 ruling 1); the `tee -a` (with `pipefail`) came after
the review showed that a hand-run attempt left no log, so a later `launch` saw neither a NaN nor an
early stop (a NaN hand run was called `already_complete`, an early-stopped one resumed and trained
to the end). Every path in the script is absolute and it starts with a `cd`, so it must be
regenerated on the target node. Cost if wrong: `--write-only` leaves an existing manifest with a
stale `command.sh` digest (detectable, rewritten by the next launch that runs); a script copied
between machines carries the wrong paths.

[DESIGN: S7. The averager is ours: a plain mean per floating tensor in float64 rounded to the
tensor's dtype, tensors that are not floats or are equal in every source taken from the last, no
optimizer state, never named `step_N`.] Eole 0.6.4 has no averaging command and its `average_decay`
saves the EMA in place of the live weights, so the master plan's `eole model average` does not exist
(`D-EOLE-CLI`). The file is written by hand in the safetensors order (deterministic bytes,
streaming) and checked with `safetensors`; bf16 goes through its bit pattern. Cost if wrong: the
peak memory is about 9 times the largest float tensor (about 7 GB for the 128k embedding,
ARITHMETIC); if Eole adds a command later the output of ours must be compared with it.

[DESIGN: S8. `loss_gate` returns a reason; `launch` applies only the NaN rule.] The gate (no NaN,
`xent` of the last report at most 0.7 times the first, validation perplexity falling) is applied by
the integration test, to the log of the first attempt. Cost if wrong: a launch of a run that does
not learn is accepted; a gate on a real teacher needs a threshold the toy cannot give.

[DESIGN: S9 and S10. The teacher config is a table of unmeasured priors; `valid_metrics` is left
out.] `teacher.yaml` takes the shape and schedule of the master plan (lines 511 to 516), the
optimiser and sequence length of the proxy config, and the rest (label smoothing, `timeout` 7200,
`keep_checkpoint` 8, one early-stopping criterion, `batch_size_multiple` 8) from the design; none
was trained. The validation corpus stays D7's pre-tokenised one, so Eole's BLEU and chrF would score
piece strings; BLEU and chrF of a teacher come from `evaluate_variant` after the training, and early
stopping is on perplexity alone. Cost if wrong: every value may be wrong; the best checkpoint by
perplexity may not be the best by chrF; `tensorboard_log_dir` is not set (Eole's default is a
relative `runs/eole`) and the comment of the file cites master plan lines 511 to 514 where the
teacher block is 511 to 516 (found while writing the records, not fixed).

[DESIGN: Controller ruling. `safetensors` is imported lazily from a locked transitive dependency.]
It is installed and locked at 0.8.0 as a dependency of Eole, not declared in `pyproject.toml`; D8
imports it inside the functions that need it, never at module import time, does not edit
`pyproject.toml` or `uv.lock`, and records its version in the manifests' `tool_versions`. `torch`
and `eole` are never imported by D8's code. Cost if wrong: if a future Eole stops depending on
`safetensors`, `is_complete` and `average` fail at import time of the function; declaring it is a
one-line change.

[DESIGN: Controller ruling. Task 7, the optional two-rank CPU gloo test, is SKIPPED.] The
multi-process classification (a non-zero exit with `Early Stop!` in the log and a complete final
step is an early stop) is covered by unit tests that fake the spike's log and exit code; the one
real observation is the spike's CPU gloo shim run. Cost if wrong: the launcher's handling of a real
multi-process job is untested end to end, here and on a GPU.

[DESIGN: Controller ruling. The run lock, and its descriptor inherited by Eole.] A real launch takes
an exclusive non-blocking `flock` on `launch/lock`; `--write-only` only looks. The first version
held the descriptor in the launcher alone; the review confirmed that a SIGKILL of the launcher then
released the lock while Eole (its own session) kept training, and the next launch started a second
Eole into the same checkpoint directory. The fix passes the descriptor to `eole build_vocab` and
`eole train` (`run_logged(inherit_fds=)`). Cost if wrong: the lock is advisory and local; a hand-run
`command.sh` neither takes nor sees it; `flock` on NFS or Lustre was not checked (an `OSError` ends
the command with exit 1).

[DESIGN: Controller ruling. The `before_writing` hook of `launch()`.] The CLI drops the old manifest
in a hook called right before the first file of an attempt is written, not before the call:
`begin_commit` before `launch()` would have deleted a finished run's manifest on an identity
refusal, a held lock or an `already_complete` rerun. Cost if wrong: one more callback in the
signature; a finished run's manifest is rewritten only when missing or no longer vouching for the
same files, which re-hashes the final step directory (GB scale for a real teacher).

[DESIGN: Controller ruling. `is_complete` never raises.] A bit flip in the zip central directory of
`optimizer.pt` made `zipfile` raise `UnicodeDecodeError` or `NotImplementedError`, and `launch` and
`average` died with a traceback; `_archive_problem` now catches `ValueError`, `NotImplementedError`,
`OverflowError` and `EOFError` too, and a damaged step is skipped like a truncated one. Cost if
wrong: a damaged but "complete" step is reported as a problem phrase and the run falls back to an
older step, losing the steps between.

[DESIGN: Controller rulings, smaller ones.] `average` finds the checkpoints with `--checkpoint-dir`
(the run's `training.model_path`) and writes `averaged` beside it by default; `ablate
--prepare-only` writes a manifest of its own stage `train_prepare`, and each of it and
`train_ablate` refuses the other's manifest, so prepare-then-ablate into one out dir is refused both
ways; `launch()` creates the run directory when it takes the lock;
`rinchen.utils.integers.is_strict_int` replaces three copies of "a bool is not an int" in D8's code
(the copies in D7's `eole_data.py` and D1's `corpus/schema.py` were left alone); the
`default_signal_handlers` fixture makes four older signal tests pass when pytest was started as a
background job with SIGINT and SIGHUP ignored. Cost if wrong: each is one place to change.

[DESIGN: The GPU was not used.] This machine has a GeForce RTX 5070 Ti that the owner did not
authorise for a run, so the spike, every test and every measurement of D8 ran with
`CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`, and everything GPU-side is read from Eole's source or
from a CPU gloo shim and tagged unmeasured. Cost if wrong: none to undo; the cost of the choice is
the list of open gates below. A one-GPU smoke test of the teacher shape needs the owner's
authorisation.

[EVIDENCE: Resume semantics of Eole 0.6.4 (spike, CPU, 2 threads; `docs/VERIFY.md`
`D-EOLE-RESUME-DATA`).] `training.train_from: <model_path>/step_N` restores the weights, the Adam
moments, the step counter and the noam schedule (optimizer.pt says 201 after step 200; the rate at
step 110 was 5.96e-03 in both runs); `train_steps` is the absolute target; the data stream restarts
from line 1 of every corpus (a resumed step 110 showed `sents: 1512; bsz: 1131/1231/151`, the
uninterrupted step 10); the RNG, the EMA and the `keep_checkpoint` queue are not restored; scheduled
`accum_count` and `dropout` lists restart at their first value; the optimiser options come from the
checkpoint (a resume YAML with learning rate 0.05 still logged 4.38e-03 at step 70); resuming a
finished run trains one extra step (step_61 after `train_steps` 60). Three uninterrupted 200-step
runs are byte-identical (weights SHA-256 `32a1c92e...`); a clean stop at 100 and a resume to 200
gave `3bdbb444...`, 58 of 62 tensors different, largest difference 0.538, validation perplexity 8.61
against 8.82; a SIGKILL near 120 and a resume equalled the clean-stop resume. Measured again on
2026-11-08 through the real binary: the uninterrupted run `32a1c92e...`; resumed from step 150 after
the weights of `step_200` were cut `7e62604f...` (validation perplexity 8.80908 against 8.81733); a
separate run extended from 100 to 200, 8.60773.

[EVIDENCE: The non-atomic saver (spike; `D-EOLE-CKPT-NOT-ATOMIC`).] Eole writes `optimizer.pt`, the
weights, the transform artifacts, `vocab.json` and `config.json` into `step_N/`, linking each file
into `model_path` right after, then deletes the oldest step directory, with no temporary name. A
simulated kill mid-save of step 100, resumed through `model_path`, exited 0 and loaded weights from
step 50 with the optimizer of step 100; `train_from: ckpt/step_100` failed (no `config.json`) and
`ckpt/step_50` worked. Eole fails loudly on truncated files (exit 1 in under 4 s). A fresh run into
a `model_path` that holds step directories is not refused. Hence the completeness check and
`launch.json`.

[EVIDENCE: Averaging in Eole (spike; `D-EOLE-EMA-SAVES-AVERAGE`, `D-EOLE-NO-AVERAGE-TOOL`).] `eole
--help` lists serve, build_vocab, predict, train, tools, convert and no `model`; `eole tools` has no
averaging. With `average_decay` the saver swaps the EMA into the parameters, so every
`step_N/model.00.safetensors` IS the EMA and the live weights are never saved (it differs from the
same-seed live run by at most 0.111 at step 100 and 0.201 at step 300); the EMA is not restored. On
the toy, development chrF at 600 steps: live 32.23, average of 500 and 600 29.90, of 400 to 600
25.90, of 200 to 600 18.29, EMA 29.87; at 1500 steps everything within 0.25 chrF (live 98.61,
average of 1400 and 1500 98.68, of 1100 to 1500 98.45). The gate's three-checkpoint average of the
200-step run translates through `evaluate_variant` with finite scores (chrF 1.16, 27.49, 1.11, 1.16:
a poor model; no threshold).

[EVIDENCE: A NaN loss does not stop Eole (spike; `D-EOLE-NAN-CONTINUES`).] With `learning_rate:
1e10` the log shows `ppl: nan; xent: nan; attn_ent: nan` and `Validation perplexity: nan`, training
goes on to the end, NaN checkpoints are written and the exit code is 0; Eole has no `isnan` or
`isfinite` check; early stopping treats NaN as stalled. A harness must read the log.

[EVIDENCE: Early stopping and multi-process jobs (spike; `D-EOLE-EARLYSTOP-RANK0`, `D-EOLE-SPAWN`,
`D-EOLE-SILENT-NOOP`).] Single process, `valid_steps` 20, `early_stopping` 2: `Decreasing patience:
1/2`, `Training finished after not improving. Early Stop!`, `Best model found at step 80`, a final
checkpoint `step_120`, exit 0, 28.5 s; Eole keeps the last `keep_checkpoint` checkpoints, not the
best. `eole train` spawns the per-GPU processes itself (one non-daemon `spawn` process per entry of
`gpu_ranks`; no torchrun, no `RANK` or `WORLD_SIZE` read anywhere); `world_size` 2 with `gpu_ranks:
[]` spawns 0 processes and exits 0 without a checkpoint ("Waiting for 0 training processes to
complete"). On a 2-rank CPU gloo shim (the shim drops `device_id`, swaps `all_gather_list` for
`all_gather_object` and fakes `torch.cuda.current_device`; no change to Eole or the repository) rank
0 logged `Early Stop!`, saved `step_120` complete and the job died with `Timed out waiting 20000ms
for send operation to complete`, exit 1, after 52 s; the same shim ran 100 steps with 2 ranks in
about 18 s (xent 2.82 against 2.81 for one rank). Ctrl-C on the shim ended all ranks with rc 0 and
no final save. NCCL was not tried.

[EVIDENCE: Gradient scaling, the statistics buffer and the batch (spike, source-read;
`D-EOLE-SPAWN`).] The loss is divided by the summed normalization and then the all-reduced gradients
are rescaled by `n_gpu` again, so the update gradient is `1/world_size` of the global mean: Adam
does not mind but `max_grad_norm` (5) becomes `world_size` times looser (40 for 8 GPUs, ARITHMETIC).
`Statistics.all_gather_stats` pickles per-corpus statistics into a 4096-byte buffer; pickle sizes
measured 650 B (9 corpora), 1.4 KB (30), 2.5 KB (60), 4.0 KB (100), 4.46 KB (112); the crash itself
was not reproduced. `padded_tokens_per_update` = `world_size x accum_count x batch_size` (8 x 8 x
16384 = 1,048,576 for the prior). `timeout` is the collective timeout and its default is 60 s.

[EVIDENCE: No recipes are shipped (spike; `D-EOLE-NO-RECIPES`).] The installed wheel has no yaml or
recipe directory (`eole/config/recipes.py` holds Qwen and LiveCodeBench orchestration schemas); the
code defaults are layers 2, hidden 512, heads 8, feed-forward 2048, SGD with learning rate 1.0,
`timeout` 60, `batch_size_multiple` 1 (the "fp16 gives 8" rule is dead code). The teacher shape is
therefore the project's own prior from the master plan and the D7 proxy config, 1,004,601,868
parameters with 64,000 pieces and 1,102,969,868 with 128,000 (ARITHMETIC: a formula that reproduces
the real tiny checkpoint exactly, 820,458 elements and 3,289,000 bytes).

[EVIDENCE: The review's confirmed SIGKILL and orphan reproduction (review of Tasks 1 to 6;
interlude).] Before the fix, with the real Eole, a SIGKILL of the launcher released the lock while
the Eole kept training, a second launch started another Eole into the same `model_path`, and
`attempt-1.log` and `attempt-2.log` both saved `step_60`, `step_90` and `step_120` at the same time
(validation perplexity 11.46 against 10.82); with fake Eoles both wrote one file. After the fix,
with the real Eole: SIGKILL of the launcher at step 30, the orphan trained alone to `step_120`, a
second launch exited 2 naming the lock, a third after the orphan ended said `already_complete`. A
hand run (`--write-only` then `bash command.sh`, 60 steps, then `launch`) is now `already_complete`
with `attempt-1.log` present.

[EVIDENCE: The completeness check and the central-directory fuzz (review; interlude).] 800 of 800
truncations of the step files were flagged; random single-byte flips were flagged for the weights in
3 of 200 trials, for `optimizer.pt` in 1 of 200, for `config.json` in 138 of 200 and for
`vocab.json` in 179 of 200 (the tensor data is not checked, by design; a flip that leaves a JSON
file valid passes); 58 real `eole train` resumes agreed with `is_complete` on truncations and
invalid-JSON flips. A fuzz of 6000 bit flips in the central directory of `optimizer.pt` raised 697
`UnicodeDecodeError` and 57 `NotImplementedError` out of `is_complete`; after the fix 3 x 1000
seeded flips never raise (before: 162 and 24 per 4000 flips) and the real binary skips a damaged
step and falls back to the previous complete one. About 60 fabricated state-machine attacks
(half-written directories, missing files, zero-byte files, symlink loops, `launch.json` missing,
corrupt or foreign, a finished run) behaved correctly; averaging was byte-identical to the spike's
averager and bit-exact against torch for bf16, f16, f64, integers, scalars and empty tensors; its
peak RSS on real files was 71 MB, and 2.6 GB for three float32 checkpoints with one 300 MB tensor.

[EVIDENCE: The STOP gate and the measurements, 2026-11-08 (Intel Core Ultra 7 265K, 20 cores, 46
GiB, CPython 3.14.10, eole 0.6.4, torch 2.14.1+cu130, safetensors 0.8.0, sentencepiece 0.2.1; GPU
hidden).] Through the real binary on the toy set with `teacher_tiny.yaml`: `ablate --prepare-only`
0.78 s; `launch` from nothing 25.61 s, `completed`, `xent` 5.44, 5.40, 4.64, 2.81, 2.28, 2.14 at
steps 10, 20, 50, 100, 150, 200 (a 60.7 % fall; the spike's three seeds 60.7, 60.7, 62.4 %),
validation perplexity 83.6869, 16.6115, 10.1611, 8.81733; a second launch 0.73 s `already_complete`
with the run directory unchanged; the weights of `step_200` cut in half, `launch` resumed from 150
in 9.60 s; a separate run 100 then 200 resumed from 100 (`Step 110/  200`, 13.50 s); `average --last
3` 0.74 s (62 tensors, 58 averaged); the run directory 19,296,131 bytes, a step directory 4,808,113.
The Task 6 gate module (28 tests) took 2 min 38 s; the gate: 7114 passed, 1 skipped, 10 deselected
(5968 on D7's last commit). Mutation checks by hand per task: Task 1 36, Task 2 70 (one equivalent
removed), Task 3 about 145, Task 4 about 110, Task 5 14 and about 74 (three equivalent survivors),
the interlude about 50; the reviewer's own 25 of 25.

[EVIDENCE: Found while writing these records.] `configs/eole/teacher.yaml` sets `tensorboard: true`
and no `tensorboard_log_dir`, whose Eole default is the relative `runs/eole`
(`config/common.py:166`) with a start-time subdirectory (`utils/report_manager.py:117-119`): the
events would go outside the run directory, one directory per attempt. The comment of the config
cites master plan lines 511 to 514; the teacher block is 511 to 516. The ledger of the build records
one review (Tasks 1 to 6), not two. Configs were not touched.

[EVIDENCE: Not verified.] Anything on a GPU (bf16, NCCL, flash attention, multi-GPU, multi-node,
tokens per second, memory, the 13 GB checkpoint write, `GradScaler` with bf16, the dtype of the
saved weights); the real two-rank early stop and the multi-process classification end to end; real
data and the effect of the data stream restarting at line 1 on a long run; `flock` on NFS or Lustre;
the teacher shape's quality and every value of the prior table; whether averaging helps a real
teacher; a live abort on NaN; the size of the log of a long run; Windows and macOS. Verified to
close an open question: `eole build_vocab` on a resume config works, and a resume with `vocab.src`
deleted works.

[DESIGN: Gates.] Open, because they need real data or a GPU: the quality, batch, learning rate,
throughput and memory of the teacher shape; everything on a GPU (a one-GPU smoke test is the
cheapest first step and needs the owner's authorisation); whether averaging helps a real teacher;
the real two-rank early stop; a live abort on NaN. The D3 gates assigned to D4, the D5 gates that
need real text and the D6 and D7 gates stay open as recorded there. D9 (Document-level) takes
`launch`, `average`, `--prepare-only`, the GPU rules and the curve parser from here, and needs a
conscious extension of the identity rule for continued training. The whole-branch review of D8 had
not run when this entry was written; its findings and fixes get an addendum.

## 2026-11-08: D8 final review - corrections and fixes

An independent review of the whole branch (`b1a8a47..90c3865`, 11 commits, read-only) found no
Critical problem, two Important ones (I-1, I-2) and seven Minor ones (M-1 to M-7). It reproduced the
gate twice (7114 passed, 1 skipped, 10 deselected in 658 s and 657 s), the STOP gate twice (every
number of the entries above, the 8-GPU `command.sh` of the runbook byte for byte, the arithmetic,
the signal exits 130, 129 and 143, the lock inheritance with the real Eole); it did not repeat the
mutation checks. This entry records what it found, how the findings were fixed (commits `D8: final
review fixes` and `D8: final review fixes, docs`), and CORRECTS in words the statements of the
entries above that the review showed false. The entries above are not rewritten.

[EVIDENCE: Corrections of earlier statements.] (1) The runbook (the "Orphans" row and "Signals"),
the spec and the entries above said that `launch` runs Eole in a session of its own and kills the
process group. The group was killed only on a timeout, Ctrl-C, SIGTERM and SIGHUP; when Eole's leader exited non-zero by
itself (a Python error, the OOM killer, `kill -9` of `eole train`) the group was left alone. Eole's
per-rank processes are `torch.multiprocessing` "spawn" children started with `close_fds=True`
(`bin/run/train.py:59`), so they do not inherit the lock descriptor either: after such a crash the
ranks of a multi-GPU run would have trained on, writing `step_N`, while `launch` reported exit 4, the
lock was free and the next `launch` started a second training into the same `model_path` (the hazard
of the first review, through another door). The review reproduced it with the real Eole on the CPU
(SIGKILL of `eole train` alone: launcher exit 4 in 0.26 s, lock free, the resource tracker and a
data-loader worker of the group alive after the launcher returned) and with a fake that starts a
`close_fds` child and exits 1 (the child alive, no lock, a second `launch` started Eole again 0.13 s
later). The ranks themselves are a source reading and a fake: no run had several ranks. (2) The
records said that the training curves of `teacher.yaml` (`tensorboard: true`) go to the run
directory, or listed the missing `tensorboard_log_dir` as open without saying what it costs. Eole's
default is the relative path `runs/eole` (`config/common.py:166`) plus a start-time subdirectory
(`utils/report_manager.py:117-119`): the events of every attempt landed in a new directory under the
working directory of whoever launched, outside the run, and from the repository root in `/runs/`,
which `.gitignore` hides. The review reproduced that with the real binary. (3) The ledger and the
runbook said that `eole_cfg.py` had "about 844 lines" (it has 837) and that the STOP gate module has
"28 tests" (28 collected from 25 functions; 30 from 27 after the fix wave). The earlier "two reviews"
of the ledger were one (Tasks 1 to 6), as the entries above already say. The comment of
`configs/eole/teacher.yaml` still cites master plan lines 511 to 514 for what is on 511 to 516; the
fix wave did not touch configs.

[DESIGN: The fixes.] I-1: after the leader of a command has exited with a non-zero code or by a
signal on its own, `run_logged` sends SIGKILL to its process group (`os.killpg`) before it reads the
log for the message, suppressing `ProcessLookupError` (nothing left, the usual case) and
`PermissionError`. The success path is unchanged on purpose: the verified hazard is a crashed leader,
a clean exit has no such survivors, and a command that leaves a process running deliberately is not
`run_logged`'s to stop. The leader has been reaped when the group is signalled: its pid stays
reserved as the group id while any member lives, and with none left the call finds no group; a pid
that the system recycled into a new group between the reap and the call (pids are handed out in
sequence, so it takes a wrap-around within microseconds) would be hit, an accepted residual risk.
D5 and D7 share `run_logged`; the one change to their tests is in `tests/support/fake_process.py`:
`FakeProcess` now has a `pid` (2**31 - 2, a number no system hands out, so a signal to its group
is `ProcessLookupError` and cannot reach a real group) where it had none on purpose, and five D5
tests that make the fake child fail needed it. After a SIGKILL of the LAUNCHER nothing sweeps the
group (unchanged known limit). The lock is held by Eole's leader only, so on a multi-GPU run it
lasts as long as the leader. I-2: `launch` adds `tensorboard_log_dir: <run directory>/tensorboard`
(an absolute path) to the effective config when `tensorboard` is true and the config names none; a
named directory is kept. It is an effective value like the others, so it is in every rendered
`launch/train-<k>.yaml` and in the identity. Checked against `training.model_path` (also a path
inside the run directory), which is in the identity as the config writes it, and done the same way:
a run directory that moved is another run for both (the config's `model_path` has to be edited and
its identity changes; the default changes with the directory), and a relative and an absolute
spelling of one run directory are one run (the default is spelled absolute; `model_path` is whatever
the config says). A run started before this change with `tensorboard: true` and no directory would be
refused as another config (none exists: no real run was ever made). The default is not validated a
second time: it is plain text of a run directory that has passed the `$` check, so a second
`validate_train_config` could not fail and would be dead code; a directory the config names is
validated with the rest of the config. M-1: Eole ignores the `model` section of the config when it
resumes and builds the model of the checkpoint's saved `config.json`, so a step directory copied in
from a run of another architecture was resumed from. `checkpoints.architecture_mismatch(saved_config,
model)` (a pure function over the parsed JSON) compares the keys both name (`architecture`,
`hidden_size`, `heads`, `transformer_ff`, the embeddings' width and position scheme, the layers,
width, heads, feed-forward size and position scheme of the encoder and the decoder; a key either side
lacks is not compared); `latest_complete(model_path, *, model=None)` passes over a complete directory
that differs, like an incomplete one, with `StepDirectory.of_another_model` set; and `launch` (and
`--write-only`) raises a `ConfigError` (exit 2) when no directory is left and some were skipped for
their model, naming the newest one's first differing key and both values, without writing or
starting anything. With a resume point the newer directories of another model are skipped like
incomplete ones (the same `incomplete step directories skipped` line, whose wording now covers them;
Eole overwrites them if the run reaches their step). Best effort: provenance cannot be proven, and a
copy of the SAME architecture from another run is not detected (operator rule: never copy checkpoints
into a run's `model_path`). The layout compared is Eole 0.6.4's, copied from a real step directory
into `tests/support/step_dirs.py`, and the fake Eole of the launcher tests writes it. Cost if wrong:
a later Eole that spells one of these keys differently would make every resume a refusal; the
integration tests resume real checkpoints, so they would fail first. M-4: `is_complete` words an
unreadable file `<name>: cannot be read (<exception class>: <reason>)`; the weights file is opened
first because `safetensors` words every failed open as `FileNotFoundError`, a refused one included.

[EVIDENCE: Tests and measurements of the fix wave.] The new tests fail on the code before the fix
(the sweep, the default, the guard and the message by assertion; the guard's module by an import
error) and pass after it. Mutants, one at a time with the caches cleared, all killed except one
equivalent: I-1 9 (no sweep, sweep on exit 0 too, each suppression dropped, SIGTERM, the leader
only, a wrong group, positive codes only, the launcher-level one), I-2 8 (one equivalent: a falsy
explicit directory replaced by the default cannot be reached, the empty value is refused by the
validator first), M-1 15 (every key of the table, the absent-key rules on each side, the flag, the
launcher's raise, its order, the resume point with and without the model, the newest directory
named), M-4 4. Real Eole on the CPU, GPU hidden (2026-11-08): I-1: a 120-step tiny launch, SIGKILL of
the `eole train` leader alone at step 60 (the group: the leader, the resource tracker, a data-loader
worker): the launcher exited 4 in 0.26 s with "eole train failed: killed by signal 9", no member of
the group was left at once or 3 s later, the lock was free, no manifest was left, and the next launch
printed `resumed from: 50`, `attempt: 2`, `status: completed`, `steps: 120`; an integration test
(scenario I of `tests/integration/test_train_tiny_teacher.py`) pins it and fails on the old code (two
survivors). I-2: a 60-step launch from a scratch working directory with `tensorboard: true`, then a
resume to 80 steps: both attempts' event files under `<run>/tensorboard/<stamp>/`, nothing under
the working directory and no `runs/` in it or in the repository root. M-1: a real 64-wide `step_50`
copied into the checkpoint directory of a 128-wide run with its launch record: exit 2 in 0.8 s,
"... step_50: config.json: model.hidden_size is 64 in the checkpoint but 128 in the config ...",
Eole not started; the same real checkpoint copied into a 64-wide run: accepted, `resumed from: 50`.
The full gate: `ruff format --check`, `ruff check` and `mypy src` clean, 7187 passed, 1 skipped, 10
deselected (baseline 7114) in 10 min 19 s of pytest (10 min 22 s with the checks). D5 and D7
regression set 1: 1839 passed (1831 before). D5 and D7 integration set (`tests/integration -k "script_conv or convert_script or tiny_ablation or train_evaluate or tiny_teacher"`): 75 passed, 1 skipped, 8 deselected in 7 min 46 s (the STOP gate module is 30 of them).

[EVIDENCE: Known and left open on purpose.] M-2: raising `train_steps` after an early stop returns
`early_stopped` at the old step (pinned by a test); to go on, start a new run directory; cost: no
note tells the operator the request was ignored. M-3: running a hand-run `command.sh` twice appends
two runs to one `attempt-<k>.log` and the NaN scan spans the whole log, so a NaN of the first run
keeps refusing a clean second one; start a new attempt or clear the log; cost: a false refusal in the
safe direction (read from `curves.py`, not reproduced with the fake). M-6: `eole_cfg.py` (837 lines)
carries key lists, value rules, the resume rule, merge, render and file I/O, and `average` imports
`manifest.stage` for `sha256_of`; cost: cohesion only. M-7: the `--write-only` look at the lock is
not a hold (documented since Task 5). Also still open: the comment of `teacher.yaml` cites master
plan lines 511 to 514; a skipped step directory of another model shares the `incomplete step
directories skipped` line of the CLI with the incomplete ones (its reason is printed only when no
directory is left); a rank process outliving a leader that died while the launcher was gone too is
covered by nothing; everything on a GPU and under NCCL. A scoped re-review of the fix wave had not
run when this entry was written.

[EVIDENCE: addendum to the D8 final review entry, from its scoped re-review (approved with minors, no Critical or Important
finding). (1) The checkpoint-shape guard compared the config's model numbers with the checkpoint's by plain equality, and Eole
accepts `hidden_size: "64"` (and the other integer keys as text) and saves 64: a run whose config quoted its numbers trained to
step_40, then every relaunch was refused as another model's checkpoint (repro: real Eole, 40 steps). Fixed in `D8: re-review
minors`: a decimal text equals the number it spells (an existing test that used the text "64" as its example of a value of another
kind now uses a list; its intent, a mismatch and not an error, is unchanged). (2) A config that omits `encoder.layers`,
`decoder.layers`, `heads` or `transformer_ff` lets a same-width checkpoint of another depth through: documented in runbook 08
(set them all; `teacher.yaml` does). (3) An explicit relative `tensorboard_log_dir` is kept as written, so its events land relative
to the launch directory: documented (give an absolute path or none). (4) The runbook 05 and 07 descriptions of `run_logged` and
the D5 spec do not mention the sweep of the process group after a non-zero leader exit; nothing in them is false. Left as is: the
documented pid-reuse window of that sweep (the pid_max is 4194304 and pids are handed out sequentially, so a reuse needs a full
wrap within microseconds; an exact fix with `waitid(WNOWAIT)` would replace the polled wait that the D5 and D7 tests record).
Cost if wrong: none beyond the documentation.]

## 2026-11-09: D9 document-level - the rulings, made without a review round

D9 (`src/rinchen/train/{docify,doc_source,doc_pack,doc_corpus,continuation,vocabulary_size,eole_cfg_keys}.py`
with the extensions of `eole_run.py`, `eole_cfg.py`, `checkpoints.py`, `launch.py`, `launch_command.py`,
`ablation_config.py`, `ablation.py`, `corpus_files.py`, `mixture.py` and `validation_corpus.py`,
`src/rinchen/infer/doc_chunker.py`, `src/rinchen/cli/{train_docify,train_runs,train_common,train}.py`,
`configs/eole/{teacher_doc,teacher_doc_tiny}.yaml`) filled gaps in the master plan with the rulings below.
**They were made by the builders and the controller without a review round of their own, because the user's
instruction for this stretch was "Proceed. I'm going to sleep now. Don't stop until you finish everything. Commit
everything." (later "continue").** None is an approved decision; each lists what it costs if wrong so that it can be
undone. (One combined review of the code ran after Task 5; it reviewed the work, not these rulings, and its findings
and fixes are in "2026-11-09: D9 review - corrections and fixes" below.) The design with the full wording is
`docs/superpowers/specs/2026-11-09-d9-document-level-design.md` (T1 to T9, amended to match the finished code, and
its section 8, "Amendments made while building"; the numbers below are the same); procedures, measured numbers and
the known limits are in `docs/runbooks/09_document_level.md`; the upstream facts are in `docs/VERIFY.md`, section
"D9 findings". The work lives on branch `d9-document-level`, stacked on `d8-teacher-training` (cut from it at
`993f32c`); it is not merged and not pushed.

[DESIGN: T1. Every Eole vocabulary holds every D3 reserved symbol (all 118 of `spm.symbols.user_defined_symbols()`),
appended by `eole_run.add_reserved_symbols` after `eole build_vocab`; `src_vocab_size` is the SentencePiece pieces
plus one.] Eole's vocabulary is the file of tokens the data showed, not the SentencePiece vocabulary, so `<doc>` and
`<sep>` were `<unk>` to the D8 tiny teacher and continuing it on document data was silent (alignment 0.00). The spike
proposed adding `<doc>` and `<sep>` only (the tiny teacher's vocabulary stayed 232 with just those two); the
controller ruled for ALL reserved symbols, because the rule is then literal and D12's placeholders and the
`<bt>`/`<ft>`/`<t>`/`<r>` tags need entries too. The `+1` is Eole's pad token `<blank>`, which is no SentencePiece
piece and counts against the size (`pyonmttok`'s `maximum_size` counts the special tokens). `run_training` (D7) takes
`reserved_symbols` and the ablation passes them; D5's converter passes none and is unchanged; `launch` adds them
for a run about to train; `command.sh` does not. Cost if wrong: every tiny-run number of D7 and D8 moved (see the
corrections below) and every earlier ablation variant retrains (its config gained `src_vocab_size`); a config that
reaches `launch` with no `src_vocab_size` gets Eole's default 32,768 and nothing says so; a hand run from
`command.sh` is not topped up.

[DESIGN: T2. Document windows: per `(src_lang, tgt_lang, corpus, doc_id)` the sentences sorted by `seg_idx`, windows of
consecutive `seg_idx` only, sizes 2 to 4 drawn by a seeded hash, shortened to a piece budget, every input row counted
once under a reason.] The corpus is in the key since the review (C1). A tail of fewer than `min_size` sentences is
dropped and counted (`tail`, `gap_single`); `contains_sep` excludes a sentence with a literal `<sep>`; a literal
`<doc>` or direction tag inside a sentence is NOT guarded. The budget counts the joined text, so the piece in front of
a `<sep>` counts, and Eole's `filtertoolong` also counts two boundary tokens of a target, so `docify` passes at most
`tgt_seq_length - 2`. Cost if wrong: a real corpus with fragmented documents will lose its short runs to `tail` and
`gap_single` (the dry run says how many); the hash-drawn sizes are one fixed mix (2, 3 and 4 equally likely), which is
a guess.

[DESIGN: T3. The doc corpora are plain text with the tags in Eole's `prefix` configuration (`src_prefix` `"<2xx>
<doc>"`, `tgt_prefix` `''`); `rinchen train docify` writes them, the validation windows, a rendered continuation
config and a manifest, and trains nothing.] `--max-pieces` defaults to, and may not exceed,
`min(src_seq_length, tgt_seq_length - 2)` of the base config's `filtertoolong`; a split with no window is a
`DataError`. The validation windows are the dev documents windowed with the fixed seed 0. Cost if wrong: the tag and
`<doc>` as prefixes are the project's mechanism and measured no different from text (spike); a placement on the target
side is untested.

[DESIGN: T4. A continuation starts only from OUR OWN checkpoint: the final `step_N` of a run whose `train_launch`
manifest vouches for its four files, or an averaged directory whose `train_average` manifest vouches for its files and
whose `averaged_from.json` agrees with that manifest; anything else is a `ConfigError` (a manifest that cannot be read
is a `DataError`).] The source is recorded in `launch.json` (schema version 2, `continued_from`) and in the manifest's
inputs, the run identity covers the kind and the SHA-256 of its weights, `config.json` and `vocab.json`, and the
files are matched to the manifest by name so a moved run still works. `rinchen train average` refuses a checkpoint
directory with no `launch.json` within three parents. These are guards against mistakes, not against someone who
writes a manifest and a record by hand, and the hash is not atomic with Eole's own read. Cost if wrong: a forged
record passes; a source that moved to another machine with other manifests is refused; a resume needs the source to
exist and verify again (a finished run does not).

[DESIGN: T5 and T6. The first attempt states `train_from: <source>` and `reset_optim: all`; every later attempt states
`train_from: <own step_N>` and `reset_optim: none`; the config must state the architecture, five optimiser options and
the SentencePiece model of the source, and the source's vocabulary must hold `<doc>`, `<sep>` and every tag the data
needs.] The design said a resume of a continuation must DROP `reset_optim`; the real Eole showed that dropping it is not
enough, because Eole fills a key the YAML omits from the config recorded in the checkpoint, and a continuation's
checkpoints record `all` (a "resume" of step 40 ran 200 steps from step 1). Only five optimiser options are required
(`learning_rate`, `warmup_steps`, `decay_method`, `optim`, `adam_beta2`): every other omitted option is inherited from
the checkpoint too and is not checked (review M2, left open). The SentencePiece model is compared by SHA-256 with the
copy Eole saved into the source (review I3). Cost if wrong: an operator config that omits `max_grad_norm`,
`label_smoothing`, `dropout` or the batch size silently continues with the teacher's values; a relative
SentencePiece path is read from the working directory of `rinchen`.

[DESIGN: T7. The chunker is pure functions over three callables; blank sentences pass through and end the window they
are in; a window whose output has the wrong segment count, or an empty segment, falls back as a whole to sentence-level
translation; the budget is 2 plus `count_pieces` of the joined text.] Left open on purpose (review M5, M6, M11): the
budget measures the source only and `eole predict` truncates a target silently, so a cut inside the last segment still
counts as aligned; a `translate_sentence` result with `<sep>` passes through; a literal `<doc>` or tag in a sentence
is not guarded. The module defines its own `<doc>`, `<sep>`, joint and fixed-piece count, because `infer` must not
import `train` (three definitions, pinned equal by a test). Cost if wrong: D11 must close the first two before a
serving path relies on the chunker; the fallback repairs the sentence count, not the content (the spike measured it).

[DESIGN: T8 and T9. `predict_tokens` takes `max_length` and `max_length_ratio` (written only when given); the gate
measures the `<sep>` alignment of held-out dev windows, never translation quality, at 0.9 for each window size after
2,500 continuation steps.] The spike's and the plan's 1,000 steps do not reach 0.9 on the non-overlapping windows
`docify` cuts (.955/.847/.886 at 1,000; the spike trained on 7,260 overlapping windows, docify gives 1,194); the
threshold was not lowered and `teacher_doc_tiny.yaml` trains 2,500 steps (.982/.978/.941), a 3,000-step run was no
better for size 4 and was not adopted. Cost if wrong: the size-4 margin is .941 against .9 on 202 windows and one
seed, and the training is repeatable only on this machine and thread count: the review estimated a 5 to 10 percent
chance of a miss elsewhere; if CI flakes, raise `train_steps`, never lower the gate. Window chrF is not gated (31 at
1,600 steps in the spike).

[DESIGN: Controller rulings, smaller ones.] The D9 branch is stacked on D8 and merging D9 means merging D6, D7 and D8
first. The reviews of D9 use one combined review after Task 5, one fix wave and a scoped re-review, instead of one
per task. `docify` takes a `split` and `DEV_WINDOW_SEED` 0 cuts the validation windows; `read_corpus_rows` loads only
the splits it is asked for; D7's CLI helpers `reject_repeated_corpora`, `numbers_under` and `corpus_manifest_files`
moved to `cli/train_common` with their behaviour unchanged; `eole_cfg_keys.py` took the key lists out of `eole_cfg.py`
(976 to 856 lines). `TRAINING_KEYS` has no `adam_beta1` or `adam_eps`: they stay Eole's defaults, which the teacher
trained with. Cost if wrong: each is one place to change.

[DESIGN: The GPU was not used.] This machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run, so
the spike, every test and every measurement of D9 ran with `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`, and
everything GPU-side is unmeasured. Cost if wrong: none to undo; the cost of the choice is the list of open gates below.

[EVIDENCE: Eole's vocabulary is not the SentencePiece vocabulary (spike; `docs/VERIFY.md` `D-EOLE-VOCAB-NOT-SPM`).]
The `src_vocab` file (the tokens the data showed plus the `prefix` specials) in file order, cut to `src_vocab_size`
(default 32,768), with Eole's four special tokens added and padded to a multiple of 8. The D8 tiny teacher's
`vocab.json` had 232 entries of 640 pieces and neither `<doc>` nor `<sep>`; continuing it naively gave alignment 0.00
and all 768 outputs with an unknown. A symbol appended at the end of the file did not survive `src_vocab_size: 100`;
`update_vocab: true` works but drops tokens missing from the new file. Measured on 2026-11-09 through the real binary:
`build_vocab` wrote 225 lines (4 of the 118 symbols present), 114 were appended (339 lines), `src_vocab_size` 641, the
checkpoint `vocab.json` 344 entries.

[EVIDENCE: `reset_optim` and continuing from another run (spike; `D-EOLE-RESET-OPTIM`, `D-EOLE-TRAIN-FROM-FOREIGN`).]
`train_from: <another run>/step_N` with a new `model_path`, `src_vocab` and `data` works and never writes the source;
the `model:` section is merged unchecked (`heads: 8` on a 4-head checkpoint loaded silently); `reset_optim: all` loads
the weights only, restarts the step at 1 and lets the new YAML's explicit options win (`Step 1/40 lr 3.49e-04`),
`none` takes every option from the checkpoint (`Step 201/240 lr 4.41e-03`, the new 0.25 and 20 ignored), `states`
keeps the step, `keep_states` crashes (`KeyError: 'training_step'`); a key the YAML omits is filled from the
checkpoint's recorded config; an averaged directory (no `optimizer.pt`) works as `train_from` under every value, all
acting as `all`; D8's launcher would have refused a continuation and resumed it with `all` still set (a full 100 more
steps, lr 6.25e-04 at step 10). With the real Eole in Task 3, a resume that omits `reset_optim` restarted at step 1.

[EVIDENCE: Predictor truncation and length limits (spike; `D-EOLE-PREDICT-TRUNCATION`, `D-EOLE-FILTERTOOLONG`).]
`eole predict` has no source-length limit (only the 5,000 sinusoidal positions: a 55,520-piece source gave
`SequenceTooLongError`) and caps the output at `min(max_length, padded_source_length x max_length_ratio + 5)` with
`max_length` 250 and `max_length_ratio` 2: `max_length: 10` gave exactly 10 pieces ending mid-word, rc 0, no warning.
`filtertoolong` counts pieces after `sentencepiece` and `prefix` (the tag and `<doc>` count), drops an example whose
source is over `src_seq_length` or whose target is over `tgt_seq_length - 2`, defaults to 192 and says only a
statistics line; it is a training transform only.

[EVIDENCE: The spike's alignment numbers and the tokenisation facts.] On the toy documents, 768 dev windows of stride 1,
one seed: continued from the tiny teacher with `<doc>` as a `src_prefix`, `<sep>` alignment .38 at 200 steps, .75 at
400, .97, .92, .99 at 600, 800, 1,000; from scratch .92 at 800; `<doc>` in the text .77 at 400 and .97 at 1,600 (no
measurable difference from the prefix); mixing the sentence corpora in gave .81 at 400 and .99 at 3,200; window chrF
31 at 1,600 steps and 95 at 3,200, so the gate is on alignment alone. Eole's `sentencepiece` transform gives exactly
the pieces `sp.encode` gives; `<sep>` written in text gets a standalone word-boundary piece in front of it, a prefix
piece does not; `re.split(r"\s*<sep>\s*", decoded.strip())` splits any output and the segment count is the `<sep>`
count plus one. Window piece lengths on the toy: 10 to 28 pieces for 2 sentences, 16 to 42 for 3, 22 to 51 for 4;
real sentences are about four to five times longer.

[EVIDENCE: The STOP gate and the measurements, 2026-11-09 (Intel Core Ultra 7 265K, 20 cores, 46 GiB, CPython 3.14.10,
eole 0.6.4, torch 2.14.1+cu130, safetensors 0.8.0, sentencepiece 0.2.1; GPU hidden; a machine loaded by other work).]
Through the real binary on the toy documents: the tiny teacher's `launch` (200 steps) and `average --last 2` (1.43 s);
`docify` 2.73 s (4,000 rows read, 3,380 in windows, 1,194 windows of 514, 368 and 312, 220 tail rows, 136 dev
windows, `--max-pieces` 126); `launch --continue-from` 2,500 steps in 107 s (87 s idle in the build), `xent` 3.07 to
0.97, validation perplexity 7.34474, 5.99236, 3.62612, 3.28613, 3.21243; the teacher's tree unchanged; two runs gave
the same weights at all five checkpoints. Alignment of 804 dev windows (334, 268, 202): .9641/.1716/.0000 at 500,
.9521/.8470/.8861 at 1,000, .9731/.9664/.9158 at 1,500, .9880/.9664/.9109 at 2,000, .9820/.9776/.9406 at 2,500; the
chunker over 66 document-direction pairs (400 sentences): 106 windows, 102 aligned, 4 fallbacks, 18 singles, chrF 70.62
against 10.44. Eight refusals of the real binary exit 2 in about a second. Gates: 8214 passed, 1 skipped, 10
deselected after the fix wave (929 s), 7187 at the end of D8. Mutation checks by hand: Task 1 31, Task 2 about 73,
Task 3 109, Task 4 68 and 110.

[EVIDENCE: Not verified.] Anything on a GPU; a real document corpus and the quality of document translation; the
mix of window and sentence data, the window sizes, the steps and the learning rate of a real continuation; the time
to hash and read a real teacher's 13 GB step directory; `read_corpus_rows` beyond toy scale; a different SentencePiece
model inside Eole (refused by hash, never run); the size-4 margin of the gate on another machine; Windows and macOS.

[DESIGN: Gates.] Open, because they need real data or a GPU: a document model on real documents; the real-corpus doc
workflow; everything on a GPU; the CTranslate2 export and serving of a document model (D11). The D3 gates assigned to
D4, the D5 gates that need real text and the D6, D7 and D8 gates stay open as recorded there. D10 (distillation) takes
`launch`, `average`, `predict_tokens` with its length keys and the vocabulary invariant from here.

## 2026-11-09: D9 review - corrections and fixes

A combined review of D9's Tasks 1 to 5 (an independent reviewer, read-only, `993f32c..98e42d1`) found: ready with
fixes, 1 Critical, 5 Important and 12 Minor findings, and reproduced the STOP gate. It checked and found sound: the
held-out dev windows of the gate, a SIGTERM test that is not vacuous, a vocabulary-without-`<doc>` refusal that fires
only from the vocabulary check, the shipped configs, the layering, the D7 refactor (behaviour-preserving) and the
hygiene. The fix wave (one commit each, failing test first) fixed:

- C1 (`3838713`): a document was keyed without its corpus; two corpora sharing a `doc_id` produced a window joining
  both (`A3 <sep> B0`) or interleaving both (`B0 <sep> A1 <sep> B2`). The key is `(src_lang, tgt_lang, corpus,
  doc_id)`.
- I1 (`7d3da62`): `read_corpus_rows` decoded the text of every split, test included; the filter is in the lazy plan.
- I2 (`61fada2`): `rinchen train average` accepted any step directories and so could turn a foreign checkpoint into a
  valid `--continue-from` source; it refuses a directory with no `launch.json` within three parents and `verify_source`
  cross-checks `averaged_from.json` with the manifest. A guard against mistakes, not forgery.
- I3 (`92dddcc`): the SentencePiece model of the config is compared by SHA-256 with the copy Eole saved in the source.
- I4 (`b1c2d12`): an empty `vocab.src` became 118 reserved symbols; it is a `DataError` now (a test that pinned the
  defect was replaced by refusal tests).
- I5 (`63c97d5`): `launch` refuses a STATED `src_vocab_size` or `tgt_vocab_size` below pieces plus one; the D5
  converter configs state sizes equal to their piece count and stay off `launch`.
- Minors (`c2d46b5`): `eole_cfg_keys.py` out of `eole_cfg.py`, the gate's aligned rule is the chunker's (a window with
  an empty segment is not aligned), the chunker closes a window at a blank sentence (`docify` ends a run at any
  excluded sentence), the permission bits of the vocabulary file are kept, dead code, the plan's wording, a constant for
  a literal. A 3,000-step run was measured and not adopted.

Left open on purpose (controller decisions; each is documented in the runbook): M2, only five optimiser options are
required of a continuation config and the others are inherited from the checkpoint; M5, the chunker's budget counts
the source only and a cut inside the last segment still counts as aligned; M6, a `translate_sentence` result that
contains `<sep>` passes through; M8, resuming a continuation requires its source to exist and verify; M9, `<doc>` and
`<sep>` are defined three times and a test pins them equal; M11, a literal `<doc>` or tag in a sentence is not
guarded. Also found and not fixed: `launch.py` has 886 lines and `continuation.py` 777 (the limit is 900); the
SentencePiece comparison reads the model path as given (a relative path from the working directory). A scoped
re-review of the fix wave was running when these records were written and nothing from it is in them. The gate after
the fix wave: 8214 passed, 1 skipped, 10 deselected.

**Corrections of earlier records.** The D7 and D8 entries above quote the numbers of the tiny runs as they were
measured at the time and are left as written. D9's vocabulary invariant changed them; the same commands measured again
on 2026-11-09 (code `c2d46b5`, a machine loaded by other work) gave, for the tiny teacher of D8 (`vocab.src` 225 to 339
lines, checkpoint `vocab.json` 232 to 344 entries): the weights of `step_200` SHA-256 `32a1c92e...` to
`6f5e836c7553563b...` and 3,289,000 to 3,318,120 bytes (two uninterrupted runs, the same); `optimizer.pt` 1,490,955 to
1,549,195 bytes; `xent` at steps 10, 20, 50, 100, 150, 200 from 5.44, 5.40, 4.64, 2.81, 2.28, 2.14 to 5.81, 5.76, 4.93,
2.91, 2.33, 2.13 (a 60.7 % fall to 63.3 %); validation perplexity from 83.6869, 16.6115, 10.1611, 8.81733 to 106.851,
18.3535, 10.4948, 8.73594; the resumed run (weights cut, resumed from 150) from `7e62604f...` and 8.80908 to
`b665b997...` and 8.80202; the separate run extended from 100 to 200 from 8.60773 to 8.78357 (its weights now
`7f0b2899...`); `average --last 3` weights from `4a0c8535...` to `629aa11e...`; the run directory 19,296,131 to
19,659,478 bytes and a step directory 4,808,113 to 4,898,781 (the same arithmetic reproduces the file: 827,738
elements and 3,318,120 bytes). For the tiny ablation of D7 the dev chrF of the source-tag variant (eng-khk, khk-eng,
khk-rus, rus-khk) moved from 98.89, 99.23, 97.76, 96.25 (mean 98.03, minimum 96.25) to 95.94, 95.85, 90.89, 93.67 (mean
94.09, minimum 90.89), BLEU from 98.85, 99.01, 98.60, 96.19 to 95.91, 95.54, 93.11, 93.41; the target-tag variant from
99.72, 98.24, 98.32, 98.33 (mean 98.65, minimum 98.24) to 99.45, 99.16, 98.13, 98.24 (mean 98.75, minimum 98.13), BLEU
from 99.67, 98.01, 98.79, 98.43 to 99.52, 99.01, 98.96, 98.01; the checkpoint 3,289,000 to 3,318,120 bytes; 135.7 s on
the loaded machine against 72.7 s; the gate (chrF at least 60) still holds with a minimum of 90.89. D5's converter
numbers are identical. The spike's figure of 232 entries for a vocabulary with `<doc>` and `<sep>` held only for those
two symbols; with all 118 it is 344. The plan's and the spec's "1,000 steps" for the gate do not hold for the windows
`docify` cuts; the checked-in config trains 2,500. The runbooks 07 and 08, STATE.md and the D9 spec carry the old
values as history beside the new ones.

## 2026-11-09: D9 re-review - the verdict, the minors fixed and the limits recorded

The scoped re-review of the first fix wave (an independent reviewer, read-only, `98e42d1..c2d46b5`) approved it with
minors: no Critical and no Important finding. The records of Task 6 were written while it ran and said nothing of it
(the end of the entry "D9 review - corrections and fixes" above says so, and stays as written); this entry and the
matching edits of the runbook, STATE.md and CLAUDE.md close that. A second small fix wave
(`135e9f9`, `D9: re-review minors`, tests first, every new test mutation-checked) fixed:

- M1: the pre-flight of `launch` (`vocabulary_size.require_sizes_to_fit`, formerly `require_stated_sizes_to_fit`)
  skipped an unstated `src_vocab_size` or `tgt_vocab_size` on the premise that Eole's default is large. It is 32,768
  (`eole/config/data.py`, `VocabConfig`; one named constant, `EOLE_DEFAULT_VOCABULARY_SIZE`, and a test that compares it
  with the installed Eole). The inputter cuts the vocabulary file at load and the reserved symbols are its last lines,
  so a config with a model of more than 32,767 pieces and no size would have lost them. The pre-flight now refuses it
  with a `ConfigError` that names the key and the value to state (for the 128,000-piece model: "state
  `src_vocab_size: 128001` at the top level of the config"). **Decision: `launch` does not inject the value.** It has no per-run override
  but `--world-size` and `--gpu-ranks`, and a silent injection would change what a config says; the operator states
  the size (`ablate --prepare-only` already does; for `docify --config` it goes in the base file, and
  `configs/eole/teacher_doc.yaml` says so in its header). `tgt_vocab_size` is not asked for when `share_vocab: true`
  (Eole ignores it then, every shipped config shares, and `ablate` states `src_vocab_size` alone); an unstated size whose
  model is not a file is left to Eole, which names the missing model when it builds the vocabulary (the many tests that
  name `spm/toy.model` without writing it rely on that); a model that is a file and does not load is a `ConfigError`. The test support gained `tests/support/sized_spm.py` (a model of an exact number of pieces, written
  as a protocol buffer, no training) and the fabricated model of `step_dirs.py` is now one that loads.
- M3: `docify(..., split="test")` on rows read without that split reported every row as `no_doc_id` and made no window,
  silently, and `read_corpus_rows(splits="train")` iterated the characters of the string and loaded nothing. `docify`
  refuses a split outside `("train", "dev")` (`ValueError`, before it reads the SentencePiece model), `classify_rows`
  refuses rows whose rows of the split are ALL empty apart from `split` (`DataError`; one empty row among real ones is
  still counted, not refused) and `read_corpus_rows` refuses a bare string (`TypeError`, the idiom the linter asks for
  an argument of the wrong type).
- M6 and the wording of the `average` guard: stale docstrings (`docify.py`, `launch.py`, whose text said that
  `eole build_vocab` accepting `training.train_from` "is not measured"), two comment lines of 125 and 145 columns, and
  the module-level import of `tests.support.step_dirs` in `tests/conftest.py`. The docstrings, the module text of
  `cli/train_runs.py` and the refusal message of `average` said that only the checkpoints of a run `launch` wrote are
  averaged. What is checked is a `launch.json` that parses within three parent directories of the step directories; a
  hand-made `step_<N>` placed inside a real run directory passes. The text now says that, and the guard is as before a
  guard against a mistake, not against forgery.

Recorded as limits, not fixed (runbook 09, "Known limits" 15 to 17): M2, the `average` guard sees only the three
lexical parents of the checkpoint directory as given (a symlinked `ckpt` outside any run directory and a `model_path`
nested three or more levels inside the run are refused; the same lookup serves `verify_source`; pass the real path);
M4, the fix of C1 keys documents per `corpus` label and not per file, and D4's `mine` sets `doc_id` to the stem of the
source file, so two files with the same label and stem still share a key and only the counted `duplicate_seg_idx`
signals it (make D4's document ids unique per file, or read that count); M5, the pre-flight runs on every plan, also
for `--write-only` and a finished run, and with a stated size needs the SentencePiece model readable from the working
directory of the launcher. The size-4 margin of the gate is unchanged (.9406 against 0.9, one seed; 3,000 steps
measured worse, .9356, so 2,500 stays); if the gate flakes on another machine, raise `train_steps` further, never lower
the gate.

Gate after the second fix wave: 8252 passed, 1 skipped, 10 deselected (1,217 s). The tiny doc integration test (`tests/integration/test_train_tiny_doc.py`, real
Eole on the CPU, GPU hidden) passed: 29 tests in 219 s.

## 2026-11-09: D10 distillation - the rulings, made without a review round

D10 (`src/rinchen/export/eole_ct2.py`, `src/rinchen/train/{distill_export,distill_export_files,distill_saved_config,
distill_teacher,distill_sources,distill_shards,distill_generate,distill_corpus,distill_mix,distill_prepare}.py`,
`src/rinchen/cli/distill*.py`, `configs/eole/student_{server,edge,tiny}.yaml`, one dict entry of `synth/tags.py`, the
`kd_tag` keyword of `train/eole_data.py`, the constant `KD_GROUP` of `train/mixture.py`) filled gaps in the master plan with
the rulings below. **They were made by the builders and the controller without a review round of their own.** None is an
approved decision; each lists what it costs if wrong so that it can be undone. (One combined review of the code ran after
Task 7; it reviewed the work, not these rulings, and its findings and fixes are in "2026-11-09: D10 review - corrections and
fixes" below.) The design with the full wording is `docs/superpowers/specs/2026-11-09-d10-distillation-design.md` (T1 to
T16, amended to match the finished code, and its section 8, "Amendments made while building"; the numbers below are the
same); procedures, measured numbers and the known limits are in `docs/runbooks/10_distillation.md`; the upstream facts are in
`docs/VERIFY.md`, section "D10 findings". The work lives on branch `d10-distillation`, stacked on `d9-document-level` (cut
from it at `7d0c166`); it is not merged and not pushed.

[DESIGN: T1. The teacher is a checkpoint of OURS (the final step of a run `rinchen train launch` finished, or a directory
`rinchen train average` wrote) whose manifest vouches for its files (`continuation.verify_source`); `rinchen distill export`
turns it into a directory of the converter's three files, `spm.model`, `export.json` and a manifest, with the converter MOVED
to `rinchen.export.eole_ct2`.] The layering test forbids `train` to import `script_conv`, so `export_ct2` moved out of D5's
`script_conv.export`, which re-exports the three names with its behaviour unchanged (one edit to an existing test: the CUDA
probe patches the new module). The spike and the first design said the tag placement "is not in the checkpoint": it IS in the
saved `config.json` of a checkpoint our launcher wrote (the `src_prefix` and `tgt_prefix` of its `data` section), so
`--tag-placement` stays a required option, is recorded, and is refused when the saved config places the tag elsewhere (the
review's I1). The record does not say whether the placement was derived or stated. The converter writes into a child of a
private directory made beside the output and the child is renamed onto it. Cost if wrong: an export made before the check
cannot be re-verified; a record schema bump would refuse every existing record; a kill leaves a `.<name>.<random>` directory
beside the output.

[DESIGN: T2. The teacher is a `Translator` (`TeacherTranslator`, CTranslate2 behind D6's protocol) that guards what the library
does silently, and counts what it drops under six counters.] CTranslate2 maps an unknown tag to `<unk>`, cuts a hypothesis at
`max_decoding_length` and a source at 1,024 pieces without a sign, and returns a forced target prefix as the first token of the
hypothesis (spike; `D-CT2-UNKNOWN-TAG`, `D-CT2-END-TOKEN`, `D-CT2-INPUT-TRUNCATION`, `D-CT2-TARGET-PREFIX`). So the tag of every
target language must be an entry of `shared_vocabulary.json` and one SentencePiece piece (`check_directions`), the end token is
always asked for and a hypothesis without it is dropped (`truncated`), a source of more than `max_source_pieces` (default 256, at
most 1,023) is not sent (`source_too_long`), a source with no piece is not sent (`empty_source`), a hypothesis holding a piece no
translation holds is dropped before it is decoded (`reserved_symbol`, added after the review) and one with a line break is dropped
(`line_break`), each under the first reason that applies. The device is `cpu` or `cuda`, never `auto`. Cost if wrong:
`--device cuda` and `--compute-type` are passed through and were never run; the counters cover this process only.

[DESIGN: T3 and T4. The sources are the `src` texts of the AUTHENTIC training rows (D7's `load_training_rows`, group `auth`), one
per direction and text; generation uses D6's shards, selection and `build_rows` with a fingerprint of its own that covers the
teacher's files.] D6's job fingerprint names a model, not the weights behind a path, so a rerun with another teacher would reuse
the wrong shards. The fingerprint is the SHA-256 over the teacher's five-file identity and tag placement, the decoding options
that decide a hypothesis (not the batch size or the threads), `shard_size`, `seed` and the `(id, text)` rows per direction; the
selection thresholds are not in it (as D6). A source with no letter or digit is not sent (the teacher would invent a sentence
for a zero-width character) and is counted. Cost if wrong: a shard directory per fingerprint accumulates; the generation holds
the references of every source in memory (unmeasured at scale).

[DESIGN: T5, T6 and T7. A KD row is a row of the corpus schema with origin `ft` (generator kind `teacher`) and label
`ft:<name>`; the KD target is the beam's top hypothesis (`--n 1`); `--qe` is refused when its scorer cannot run.] **Selecting by
MBR over the n-best lowered the agreement of the KD targets with the authentic ones on the toy** (3,583 of 3,599, 99.6 %, for the
top hypothesis; 2,749 of 3,600, 76.4 %, for n = 4 with MBR; ranks 2 to 4 are about 41 chrF below rank 1, and 871 of 14,400
candidates of n = 4 failed the script rule; `D-KD-NBEST-MBR`), so `n = 1` is the default and n-best as extra targets is not built
(the schema has no rank column). Cost if wrong: a real teacher's n-best may help, which nobody measured; the toy teacher is
overfit, so `matches_reference` is 99.6 % and says nothing about KD. The real QE scorer needs a GPU and was never loaded; its
thresholds (script share 0.9, MetricX-QE floor 8.0) are priors.

[DESIGN: T8. `--mix K:O` is exact in the WEIGHTS, in every direction: the originals weigh `O x` D7's weights and the KD corpus
`K x` their sum, so the KD share is `K/(K+O)` and the directions' shares are the temperature's.] The controller ruled on a case
the design did not cover: **a direction WITHOUT a KD corpus (`generate --direction` covered only some) has its originals scaled
by `K + O` instead of `O`**, so that its share of the draws stays the temperature's and the weights add up to `(K + O) x` the sum
of D7's (the toy: 33, 66, 34, 34 and KD 33, 200 in all). Full coverage is the design's. Cost if wrong: the draw share rests on
`D-EOLE-WEIGHT-SEMANTICS` (measured for Eole's mixer); `num_workers` above 1 with weights was not run and `--mix 10000:9999` gives
a weight sum near 2 million with no guard; `kd.repeat.<direction>` (how many times more often a KD row is drawn than an original
one: 1.25 at 1:1 on the toy) is recorded and never gated.

[DESIGN: T9. KD corpora are tagged `<ft>` by default (`<2xx> <ft>` for the placement `src`, `<ft>` alone as `src_prefix` for
`tgt`); `--no-kd-tag` is allowed and recorded.] Eole drops, silently, a training example whose tokenised source equals one already
in the same minibatch (`D-EOLE-BATCH-DEDUP`); a KD row repeats an original source, so with the KD corpus mixed 1:1 and untagged
**5.19 % of the examples were dropped (798 of 15,360 at 2,048-token batches, 5,317 of 102,400 at 16,384) against 0 % with `<ft>`
and 0 % for the originals alone** (spike, measured again for the runbook: 798 of 15,360 untagged, 0 of 13,312 and 0 of 102,400
tagged; `D-EOLE-BATCH-DEDUP-KD`). The tag cost nothing measurable (chrF 98.58, 98.55 with `<ft>` at inference), and the student is
used without it. Cost if wrong: on real data the KD rows rarely equal the references, so the collisions and the benefit of the tag
are UNMEASURED; whether `<ft>` helps a real student is open.

[DESIGN: T10. The student configs: `student_server.yaml` (encoder 16, decoder 3, hidden 768, heads 12, feed-forward 3072),
`student_edge.yaml` (12, 2, 512, 8, 2048), `student_tiny.yaml` (2, 1, 64, 4, 128); `src_vocab_size` 64,001 for the 64k
SentencePiece model; each student has its OWN SentencePiece model.] The shapes are the master plan's. Parameter counts are
ARITHMETIC (190,845,963, 79,006,218 and the teacher's 1,102,982,164, by a formula that reproduces the element counts of the two tiny
checkpoints exactly). Unequal depths need no code (`D-EOLE-LAYER-COUNTS`); **a 3/1 shape at the 2/1 recipe did not converge in 2,000
steps (dev chrF 38.4 against 98.6)**, so a depth needs its own recipe, unmeasured for the real shapes. A student with a 500-piece
SentencePiece model against the teacher's 640 trained on the KD text unchanged and reached chrF 89.4 against 98.6 (the coarser model
cuts nouns into characters): what changes is the validation corpus (cut with the student's model), `src_vocab_size` (pieces plus
one: 501) and the evaluation. The server and edge schedules are the teacher's scaled to the model, with one difference the headers
state (`attention_dropout: [0.1]`). Cost if wrong: every value of the server and edge configs is an unmeasured prior.

[DESIGN: T11, T12 and T15. `distill prepare` lays out the data and config of a student (reading the KD files only with a valid
manifest) and trains nothing; `distill evaluate` scores any checkpoint of ours on the `dev` rows with its own SentencePiece model;
every command hashes its inputs ONCE before it reads them and writes its manifest last.] The controller ruled on hashing: the CLI
hashes every input once (`hash_input`) and passes the KD digests to `load_kd_rows(digests=)`, which compares them with the manifest
instead of hashing a large file twice (a spy test pins it); `prepare_student(before_writing=)` runs once after every refusal and
before the first write, so a refusal never deletes an old manifest and a dry run writes nothing, not even the directory. The window
between a hash and the read is every command's. Cost if wrong: a file swapped in that window is read as it is; the digests guard
against a mistake or a stale file, not against someone who writes a manifest by hand.

[DESIGN: T13 and T14. D10 builds only the export its gate needs (`export/eole_ct2.py`, `train/distill_export.py` and the command);
D11 takes over quantisation, `rinchen.json`, the engine, serving and the strict parity gate. No document-level teacher, no
monolingual KD sources, no `<ft>` at inference.] The gate's CTranslate2 against Eole figure (95 % of 200) is D11's floor, measured
here on three tiny models (200 of 200 each). Cost if wrong: `verify_export` and the record live in `train`, where `infer/` may not
import them; D11 has to move or re-state the reader.

[DESIGN: T16. The GPU was not used.] This machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run, so the
spike, every test and every measurement of D10 ran with `CUDA_VISIBLE_DEVICES=""`, `gpu_ranks: []` and CTranslate2 on
`device="cpu"`, and the converter runs with the device hidden (`D-CT2-EOLE-CUDA`). Everything GPU-side is unmeasured. Cost if
wrong: none to undo; the cost of the choice is the list of open gates in the runbook.

[DESIGN: Controller rulings, smaller ones.] The D10 branch is stacked on D9, and merging D10 means merging D6, D7, D8 and D9 first.
The reviews of D10 use one combined review after Task 7, one fix wave and a scoped re-review, instead of one per task. A
checkpoint without our manifest is never a teacher; `train_from` and `reset_optim` stay unknown keys in every config and are never
rendered; the edits to D5 to D9 are limited to `script_conv/export.py` (imports), `synth/tags.py` (one entry), `train/mixture.py`
(one constant), `train/eole_data.py` (one keyword) and `cli/main.py` (registration), plus `train/eole_cfg.py` (an optional `reason`
of two refusal helpers, from the review). Cost if wrong: each is one place to change.

[EVIDENCE: the STOP gate.] Through the real `rinchen` binary on the CPU with the GPU hidden, the same numbers in the spike, the gate
and the runbook's run: the tiny teacher (2,000 steps, averaged) exported (`model.bin` 2,034,311 bytes, 344 vocabulary entries); 3,600
sources, 3,599 KD rows, 3,583 equal to an authentic target; the mix 1:1 weights 22, 6, 22, 6, 15, 7, 15, 7 and KD 28, 28, 22, 22
(`3:1`: 84, 84, 66, 66); the student (2 encoder, 1 decoder layers, 2,000 steps) chrF 98.58 against the teacher's 98.20, no tag leak;
its export `model.bin` 1,835,438 bytes, CTranslate2 equal to Eole on 200 of 200 sentences; a second student with a 500-piece
SentencePiece model prepared and trained. **The toy ranks nothing** (three seeds of the student: 98.58, 98.59, 96.68; originals only
97.01), and no threshold was lowered (teacher at least 90.0, student at least 95.0 and the teacher's minus 3.0, CTranslate2 at least
95 %, at least 3,550 rows). Cost if wrong: none; the numbers test machinery, not quality.

## 2026-11-09: D10 review - corrections and fixes

A combined review of D10's Tasks 1 to 7 (an independent reviewer, read-only, `e5ff644..e67b05e`) found: ready with fixes, 0 Critical,
2 Important and nine groups of Minor findings, and reproduced the STOP gate (44 tests passed in 240 s, the numbers as above). It
checked and found sound: the move of the converter, the hashes taken before anything is parsed, the SentencePiece comparison, the
stripping of the target prefix, the alignment when sources are skipped, `auto` refused, the lazy imports, the completeness of the
fingerprint, a truncated shard regenerated, the QE orientation, the mix through the real CLI (partial coverage 33, 66, 34, 34 and KD
33, sum 200; 3:2 giving KD 99 against 66; a back-translated mix at 2:1 with KD 62 = 2 x 31 and 38 = 2 x 19), the `tgt` rendering
(`src_prefix` `<ft>`, `tgt_prefix` the direction tag), the foreign manifests, the dry runs, KeyboardInterrupt (exit 130), the
independence of the gate's tests, the shapes and the arithmetic, the hygiene and the edits to existing files. The fix wave (one
commit each, failing test first) fixed:

- I1 (`95a28c8`): the tag placement of an export was the operator's word and never checked although the checkpoint's saved
  `config.json` records it; a source-tag teacher exported with `--tag-placement tgt` exited 0 with a vouching manifest and
  `generate` over it gave 1,190 rows of which 1 matched an authentic target, Cyrillic garbage that passed the script rule. The new
  `distill_saved_config.py` reads the placement back and `export` and `evaluate` refuse a statement that disagrees (exit 2, before
  anything is made); a saved config without a `data` section or a direction tag keeps the statement; contradictory saved data is a
  `DataError` (exit 3). The export record is unchanged: it does not say derived or stated. 23 mutants; real artifacts: source-tag
  teacher + `tgt` exit 2, + `src` exit 0; target-tag teacher + `src` exit 2; student + `tgt` exit 2.
- I2 (`46c160e`, `cc917e2`): a KD hypothesis holding the unknown piece, the end token inside it, a reserved symbol or a direction tag
  decoded to literal text and passed D6's selection (`dropped_purity 0`); the teacher now drops it before decoding and counts it
  under the new counter `reserved_symbol` (counts file, manifest metrics, result line); the second commit extended the rule to the
  vocabulary entries that are no SentencePiece piece (`<blank>`, `averyunlikelytoken0`). 800 real hypotheses gave 0, so the rule has
  no real positive. The wrong-placement export produces garbage with no reserved symbol, so I2 would not have caught it: I1 prevents it.
- Minors (`6eb778a`): M1 the export cleanup could remove a directory another process made at the output path (the converter now
  writes into a private directory beside the output, renamed into place; a rename onto an empty directory appearing in the last
  instant would still replace it, stated in the code); M2 the record's vocabulary entry count is checked against the converter's
  `shared_vocabulary.json` and a saved config with two SentencePiece models is refused (the converter reads the weights after the
  digest: stated); M6 the student headers say that `attention_dropout: [0.1]` differs from `teacher.yaml`; M9 the glob refusal of
  `generate --out-dir` no longer talks about Eole. `d3ed47f`: a test of the fix wave digested a directory shared with other tests.

Edits to existing tests, all forced: the counter key set in five files, three assertions on the converter's output directory. Gate
after the fix wave: 9706 passed, 1 skipped, 10 deselected (20:57), against 9439 before it and 8252 at the end of D9; the integration
module 44 passed (180 to 190 s) twice.

Left open on purpose (controller decisions; each is documented in the runbook): M3, `generate` exits 0 with few or no rows
(`load_kd_rows` refuses an empty corpus later); M4, a resumed generation reports the teacher counters of this process only and the
manifest key `teacher.name` holds the KD corpus name; M5, "exact" means exact weights (`num_workers` above 1 with weights was not
run, `--mix 10000:9999` gives a weight sum near 2 million above `bucket_size` with no guard); M7, memory and scale (the references
frozensets, the plan and the fingerprint computed twice, all KD rows in memory); M8, `prepare` prints no `source_collision_rate`,
although `D-EOLE-BATCH-DEDUP` applies to multi-parallel sources under `--tag-placement tgt`; the gate's test file has 893 lines.
Also found and not fixed: an export made before I1 cannot be re-checked (the saved config is not in the export directory);
`distill evaluate` on an unparseable or non-Eole `config.json` exits 3 (it was 4); `verify_source`'s refusals say "cannot continue
from"; the student gate's margin (95.0 against 98.58 measured). A scoped re-review of the fix wave (`e67b05e..cc917e2`) was
dispatched at the same time as the records were written; its verdict is not recorded here.
