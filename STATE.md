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

**D2 (human test-set annotation app) is done; its STOP gate is fully met,
including a whole-branch post-hoc review pass and a green CI run.**
Pushed to `origin/main` 2026-11-02 (commit `d7e0248`); CI run `41473011127`
passed clean on `ubuntu-24.04` with no further fixes needed. See the full D2
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

**D3 (normalization, sentence splitting, SentencePiece) is done; its STOP
gate is met on the one profile that was actually trained (`tiny_1k`).** Built:
`rinchen.lang.scripts` (`script_of`), the 10-language registry
(`configs/languages.yaml`, `rinchen.config.languages`), `rinchen.config.spm`
and `configs/spm/{tiny_1k,student_64k,teacher_128k}.yaml`,
`rinchen.utils.sampling.temperature_weights`, `rinchen.text.normalize`,
`rinchen.split` (rules for mn_Mong, CJK, Latin and Cyrillic families),
`rinchen.spm.{symbols,corpus,train,check}` (118 reserved symbols), the
`rinchen spm train|check` CLI (`rinchen.cli.spm`), a synthetic 4-language
tiny corpus (`tests/fixtures/spm/tiny_corpus/`, reproducible with
`scripts/build_spm_fixture.py`), a 200-line golden normalization file, and
`tests/{property,golden}/`. Commits `265476f` through `3623e4a` (24 commits,
6 of them `D3 fix:` and 2 plan-text corrections; design and plan in
`076bb41`; `ed45433` added this paragraph, `docs/runbooks/03_tokenizer.md`
and DECISIONS.md's 2026-11-06 entry), plus this commit. The final fix wave
after the whole-branch review (0 Critical; `f75d7d2`, `3623e4a` and this
commit) guarded the splitter against quadratic terminator runs (a differential
check over 250,000 random strings found no difference in splits or match
spans), rejected a blank `corpus_dir`/`output_prefix`, bounded `spm check`'s
reads to its sample, pinned the trainer options, the exit-4 mapping and the
normalize-before-tags contract, and recorded the open gates in DECISIONS.md.
STOP gate, run for real on Linux with Python 3.14.10, 2026-11-06:
`uv run pytest tests/golden -q` -- **289 passed** in 0.15 s (200 golden
lines, 83 Cyrillic-label checks, 6 file-level checks);
`time (uv run rinchen spm train --config configs/spm/tiny_1k.yaml && uv run
rinchen spm check --model data/tiny/spm/tiny_1k.model --corpus-dir
tests/fixtures/spm/tiny_corpus)` -- exit 0, **real 1.004 s** for the whole
command (three reruns: 0.98 to 0.99 s), so the plan's "under 10 seconds" gate
is met with a wide margin. That time is almost all start-up: `uv run rinchen
--help` alone takes 0.48 s per invocation (the CLI imports every subcommand
group, FastAPI under `rinchen.cli.annotate` the largest), while in-process
`train_spm` took 0.045 to 0.055 s and `check_spm` 0.012 s (scratch timing
script on a temp copy of the tiny config, three runs). The check printed
coverage 1.0000 for all four languages, `byte_fallback: on`, `ok`; the model
has exactly 1000 pieces. The `data/tiny/spm/` artifacts were deleted
afterwards (gitignored). Full gate (the four commands run individually,
`just` is not installed on this box): `ruff format --check .` -- 124 files
already formatted; `ruff check .` -- all checks passed; `mypy src` -- no
issues found in 63 source files; `pytest -m "not network" -q` -- **912
passed, 6 deselected** after the final fix wave (878 before it; plus one
unrelated pydantic UserWarning from `manifest/model.py`). What was and was not
exercised: only `tiny_1k` was ever trained, on synthetic pseudo-word text, so
its 1.0000 coverage says nothing about real data; `student_64k` and
`teacher_128k` are schema-validated YAML only and `data/clean/spm_corpus` does
not exist; no real corpus has been normalized, split or tokenized; no CI run
of this branch is recorded here. The golden file's 200 expected values were
derived by hand and independently re-derived by a reviewer with 0
disagreements across all 200 lines.
`spm train` writes no manifest. Known limitations, three items flagged for
human review (ZWNJ/ZWJ stripped in mn_Mong; the Mongolian and Kazakh
abbreviation extras; the O-umlaut look-alikes `normalize` does not fold) and
the `add_dummy_prefix` revisit-before-D8 note are in DECISIONS.md. The D3
research follow-up below resolved the ZWNJ/ZWJ flag (kept stripping, with an
audit gate before D5) and the `add_dummy_prefix` note (default kept, tag added
as a token), and narrowed the abbreviation flag by a published handbook
source; the O-umlaut flag is still open.
Process gotcha: the editor tools (Write/Edit) convert backslash-u escapes into
the raw characters, so any file that holds invisible or look-alike characters
as escapes must be re-checked with `grep -nP '[^\x00-\x7F]'` after editing.
[EVIDENCE: `docs/superpowers/plans/2026-11-06-d3-normalization-sentencepiece.md`,
`docs/runbooks/03_tokenizer.md`, live `uv run pytest tests/golden -q`, timed
`rinchen spm train` + `spm check`, and the four gate commands, 2026-11-06,
commits `265476f` through `3623e4a`; the final fix wave's differential
splitter check and gate run, same day.]

**Notes for D4 (from D3's final review).** Run `normalize` on raw text first,
before any reserved symbol is inserted. `split` assumes normalized input: the
Cyrillic family under-splits before a Latin-uppercase or digit start and the
Latin family can over-split before a lowercase start after an unlisted
abbreviation, so log boundary errors by family on a ~1k-sentence sample before
tuning. `read_lines` and `sample_balanced` materialise the corpus: budget the
memory for it, or add a streaming reader before the student/teacher runs.
Count the Fita-like substitutes, uppercase `Y` and U+200C/U+200D in the real
corpora before freezing the vocabulary. `add_dummy_prefix` is decided (kept;
see the D3 research follow-up below). `spm train` writes no manifest, so D8
must hash the model itself.

**D3 research follow-up (branch `d3-research-followup`, cut from `main` at
`93ffdf2`; merged into `main` locally by fast-forward to `2346cea` on
2026-11-06, not pushed).** The three open decisions were
researched and recorded in DECISIONS.md ("2026-11-06: D3 follow-up - the three
open decisions, resolved by research"; evidence, sources and the reproduction
harness in `docs/research/2026-11-06-mongolian-controls-and-spm-tags.md`): (1)
the `<2xx>` target-language tag is inserted at token level after SentencePiece
(Eole's `prefix` transform in training, the CTranslate2 wrapper at inference)
and `add_dummy_prefix` stays at its default, which settles the
`add_dummy_prefix` item in the notes above; (2) ZWNJ and ZWJ stay stripped in
`mn_Mong`, with safeguards, which resolves that FLAG to keep-stripping with an
audit gate before D5 (the Mongolian-reader review the old FLAG asked for has
not happened); (3) the converter emits NNBSP before separated suffixes, and
`normalize` keeps NNBSP and MVS and maps neither. One code change: the Cyrillic
splitter has per-language abbreviation sets (a shared base plus Mongolian-only
and Kazakh-only sets), so Kazakh-only entries no longer apply to Mongolian or
Russian and the other way round, and the Kazakh set gained the dotted
abbreviations of the Kazakh punctuation rules (Syzdykova, handbook chapter,
2000, reprinted at bilim-all.kz; the units year, century, second, minute and
hour, and one more form each of "and so on" and "and the like"). Gates: D4 -
abbreviation induction and a boundary-error sample per language, the count of
o-like substitutes and uppercase Y in real khk_Cyrl, an audit of U+200C,
U+200D, U+180A..U+180F and U+202F in any real traditional-script text; D5 - the
converter never emits ZWJ or ZWNJ (a validator enforces it) and emits NNBSP, a
render check in the target fonts, and the abbreviation and suffix questions
that need a Mongolian reader; D8 - the tag and prefix design with its listed
tests, per-direction corpora against a custom transform, and sanitising
reserved strings in raw text. Not verified: no Mongolian reader judged any
render, only two Noto Sans Mongolian builds were tested, the texts of MNS
5552:2005 and MNS 4333:1996 were not retrieved, Eole per-direction corpora were
not run end to end, and the abbreviation entries still lack an orthographic
authority (the Kazakh ones rest on a published handbook chapter, the Mongolian
ones on web usage evidence). Full gate on this branch's head, run individually
(`just` is not installed): `ruff format --check .` -- 124 files already
formatted; `ruff check .` -- all checks passed; `mypy src` -- no issues found
in 63 source files; `pytest -m "not network" -q` -- **938 passed, 6
deselected** (912 before this branch plus 26 new splitter tests), with the same
pydantic UserWarning as before. Commits on top of `93ffdf2`: `ce35cad` (the
splitter change and its tests), `ec611fc` (the research note), `719a741` (the
records) and a last commit that corrects provenance and wording found in
review. [EVIDENCE: live runs of the four gate commands and of the harness
scripts, 2026-11-06.]

**D4 (filtering and mining) is done and merged into `main` locally; its STOP
gate ("tiny filter run green") is met locally.** It was built on branch
`d4-filtering-mining`, cut from `main` at `2346cea`, which was fast-forwarded
into `main` on 2026-11-07 (merge tip `b38eaae`) and then deleted. `main` is not
pushed and no CI run covers D4.
Built: the corpus Parquet contract (`rinchen.corpus`: `CORPUS_SCHEMA`,
`corpus_frame`, `scan_corpus`, `write_corpus`), `rinchen.config.filters` and
`configs/data/filters.yaml`, the seven filter stages and `run_pipeline`
(`rinchen.filter`: `script_ratio`, `numbers`, `length`, `lid` with the B7
`GlotLID` wrapper, `minhash`, `labse`, `qe` with the MetricX-QE adapter
`qe_eole.py`), `rinchen.mine` (`Embedder` and LaBSE, the B6 `align`,
`match_documents`, `pairs`), `rinchen filter run` and `rinchen mine align`
(`rinchen.cli.{filter,mine}`, with the error-to-exit-code mapping shared with
`rinchen spm` in `cli/errors.py`), `Manifest.status`, test helpers in
`tests/support/`, and `docs/runbooks/04_filtering_mining.md`. 22 commits before
this one (`fb8c6d7` design and plan; `e882113` through `102509a` the code, its
review fixes and the test hardening), then the records commit and a final
review-fix commit (below). The design is
`docs/superpowers/specs/2026-11-06-d4-filtering-mining-design.md` (updated to
the finished code); the 17 rulings, made without a review round because the
user asked for no check-ins, are condensed in DECISIONS.md ("2026-11-07: D4
filtering and mining"). STOP gate and gates, run for real on Linux, CPython
3.14.10, 2026-11-07 (`just` is not installed here, so the four gate commands
were run individually; the figures are those after the final review-fix
commit): `ruff format --check .` -- 176 files already formatted;
`ruff check .` -- all checks passed; `mypy src` -- no issues found in 85 source
files; `pytest -m "not network" -q` -- **1702 passed, 10 deselected** (1656
before the final review fixes, 938 on the branch point; the end-to-end test in `tests/unit/test_cli_filter.py` runs
the CLI on the 50-row planted corpus and gets 28 kept rows with the planted
reason counts exact, `qe` skipped). Numeric gates, on
synthetic planted data: MinHash recall 0.98 and precision 1.0 (gate 0.95);
aligner F1 0.989 on seed 11, mean 0.993 and minimum 0.978 over seeds 11 to 30,
with `merge_penalty` 0.3 (0.906 without it; gate 0.95); numbers filter handles
`1,5` and `1.5`; QE stage skipped cleanly without a scorer or CUDA. `network`
checks, run with `uv run pytest -m network -q` on the RTX 5070 Ti machine
(Intel Core Ultra 7 265K, 46 GiB RAM, driver 595.99.02, models in the local
caches): **10 passed** (13.3 s and 12.6 s on two runs); four are D4's and ran against the real models
-- GlotLID parity (wrapper top-1 label equals the raw fastText label for 200
of 200 fixture lines; the 20 `mn_Mong` golden entries are synthetic) and the
default-config check (no fixture line flagged), LaBSE (cosine 0.90 for a
translation pair against 0.11 for an unrelated one) and MetricX-QE (0.0 and
0.09 for good pairs against 9.9 and 25.0 for a mismatched pair); the other six
are the older D0 checks (PyPI, the torch index, `uv lock`, the fasttext shim),
which also passed. None could not run. Measured throughput and memory
(`docs/runbooks/04_filtering_mining.md`, "Estimates"): GlotLID on CPU 77,000
lines/s on all cores and 6,500 on one core (38-character lines; 55,000 and
5,900 at 116 characters); LaBSE about 5,500 sentences/s at batch 128 (2,400 at
117 characters), 2,091 MiB of GPU memory; MetricX-QE 500 pairs/s at batch 32
(226 at 127/156 characters), 4,935 to 5,041 MiB. Real-model integration run
(not a validation): 24 `khk_Cyrl` -> `eng_Latn` rows through `rinchen filter run`
with the real GlotLID, LaBSE and MetricX-QE, 9.5 s wall; the 16 natural pairs
were kept and the 8 planted bad rows (2 Russian sources, 2 changed numbers, 2
mismatched translations, 1 exact duplicate, 1 very long target) were rejected as
`lid_src`, `numbers`, `labse`, `duplicate`, `len_ratio_char`; one natural pair
was close to a floor (`lid_src` 0.623, Buryat 0.377, floor 0.5). The Mongolian
sentences were written by the assistant and not reviewed by a Mongolian
reader. A 12-line `rinchen mine align` run gave 12 correct 1-1 rows in 6.3 s.
Findings worth knowing: GlotLID v3 labels Chinese `cmn_Hani` or `wuu_Hani`,
never `zho_Hans`, and has no `mn_Mong` (the default accept table and the
wrapper's pre-detector handle both; Simplified versus Traditional is not
checked); LaBSE used to stay on the GPU while the QE model ran (6,665 MiB
allocated in all, against a `run_pipeline` docstring that said otherwise): the
pipeline now releases the embedder after the `labse` stage and the QE stage runs
with 4,866 MiB allocated (6,387 MiB used on the device, 1,068 idle; measured in
the real `filter run` command over the 24-row smoke corpus, not tried on an 8
GiB card); the installed `huggingface_hub` ships `hf`, not `huggingface-cli`, and the download
writes `models/lid/model_v3.bin` while the default config reads
`models/lid/glotlid_v3.bin`. NOT verified: every threshold on real data (none
exists), the aligner on real embeddings (calibrated on synthetic additive data
only), the no-GPU path with the real adapter, LaBSE on CPU, `filter run` on more
than 24 rows. The D3 gates assigned to D4 stay OPEN because they need real
text: abbreviation induction and a boundary-error sample per language; counts
of o-like Latin substitutes and uppercase `Y` in real khk_Cyrl; an audit of
U+200C, U+200D, U+180A..U+180F and U+202F in real traditional-script text
before D5. `match_documents` is a library function with no command yet, and
there is no CometKiwi adapter. [EVIDENCE: the four gate commands, `uv run
pytest -m network -q`, the scratch measurement scripts and the smoke runs
described in the runbook, 2026-11-07, commits `fb8c6d7` through `102509a`.]

Final review fixes (whole-branch review of `2346cea..8cbd39f`, one commit after
`8cbd39f`): both commands refuse a run that would overwrite its own input
(`filter run --corpus` equal to `scored.parquet` or `kept.parquet` of the out
dir, `mine align --out` or its manifest equal to `--src` or `--tgt`; exit 2),
and an out dir that holds another stage's manifest (exit 3); the pipeline
releases the LaBSE embedder after the `labse` stage and its docstring is true
(`Releasable`, `SentenceTransformerEmbedder.release`); the skip and commit
protocol of the two commands lives in `manifest/stage.py`; Parquet and manifest
writes go through a uniquely named temporary file; `fasttext-wheel` is recorded
in the manifest's `tool_versions`; the wall-clock assertion of the aligner test
is gone; the filter config shares its path validators with the SentencePiece
config; the runbook and the spec were corrected. The 24-row smoke run was
repeated with the release in place (exit 0, `kept 16 of 24 rows`), and `uv run
pytest -m network -q` passed again (10 passed, 12.4 s).

**D5 (script converter) is done and merged into `main` locally; its STOP gate (tiny
converter exported and served) is met locally.** It was built on branch
`d5-script-converter`, cut from `main` at `4eac98f` on 2026-11-07, fast-forwarded into
`main` the same day (merge tip `862e176`) and deleted; `main` is not pushed and no CI run
covers it. It is a
scaffold, not a quality converter: the rule baseline spells 237 of 1,022 held-out dictionary
words exactly, the tiny model memorises a 500-word fixture (and gets 17 of 52 held-out
fixture words right), the full profile was never trained, and there is no real parallel
script data or Mongolian reader behind any of it.
Built: `rinchen.script_conv` (`alphabet`, `validate` with `is_script_word`, `segments`,
`eval`, `fallback`, `rules`, `data`, `char_model`, `config`, `train`,
`export`, `runtime`, plus the package data `lexicon.tsv` and `standardized_variants.tsv`),
the `rinchen convert-script` commands `build-data`, `train`, `export`, `apply` and `eval`
(`rinchen.cli.convert_script`; its `_subprocess` runner, `run_logged`, now lives in `rinchen.utils.logged_process`, moved by D7), the rendered Eole configs `configs/eole/converter.yaml` and
`converter_tiny.yaml`, `configs/spm/converter_char.yaml`,
`scripts/build_script_conv_fixtures.py`, the Wiktionary reference and the fixtures built from
it (`tests/fixtures/script_conv/`), `tests/integration/` with its helpers in `tests/support/`,
`manifests/ucd_standardized_variants.json`, and `docs/runbooks/05_script_converter.md`.
25 commits ahead of `main` at the merge (`862e176`); `ed7ead9` design and plan,
`2186b83` through `7517915` the code and its review fixes, `14e04ba` the runbook and
records, then the parked minors, the final review fixes and their docs; the last commit is the
one that records this). A whole-branch review found no Critical issue and four Important ones,
all fixed: an empty valid split is refused because `eole train` divides by its row count
(ruling V), the character model lives in the run directory and `export` checks its hash
against the train manifest (W), the harvest scripts are not shipped and the runbook says so
(X), and `run_training` takes the profile so the CPU-only checks run there. The design is
`docs/superpowers/specs/2026-11-07-d5-script-converter-design.md` (updated to the finished
code); its 13 rulings and the 24 amendments A to X, made without a review round because the
user asked for no check-ins, are condensed in DECISIONS.md ("2026-11-07: D5 script
converter" and "2026-11-07: D5 final review"). Gates, run for real on Linux, CPython 3.14.10, 2026-11-07 (`just` is not
installed here, so the four gate commands were run as one chain, on the tree of the final
review fixes, before the docs commit): `ruff format --check .` -- 223 files already formatted;
`ruff check .` -- all checks passed; `mypy src` -- no issues found in 99 source files; `pytest
-m "not network" -q` -- **2833 passed, 1 skipped, 10 deselected** in 177 s (179 s for the four
commands) (1702 passed on `main`; the integration tests train real Eole models and make the
suite slower).
Numeric gates, tiny profile (2/2 layers, d64, 1000 steps, dropout 0, `--threads 2`),
CPU only: CER 0.0051 (Cyrillic to script) and 0.0128 (script to Cyrillic) on the 448
training words (gate 0.15); NNBSP, MVS and selector counts kept in 78 of the 81 words that
carry them (gate 90%); CTranslate2 against Eole on 448 of 448 words in both directions (gate
0.99); `ScriptConverter` word exact match 441 of 448 = 0.984 (gate 0.90) and 448/448
against Eole in both directions, with no rule fallback and no invalid output; training 34.8
s to 37.4 s over five runs (gate 150 s); export 2.3 s to 3.5 s; the real `rinchen` binary
ran `build-data`, `train --profile tiny`, `export`, `apply` and `eval` in a temporary
directory (CER 0.0051, word accuracy 0.9844). Rule baseline, measured once after the rules
were frozen: the fallback (one letter map and five contextual rules) spells 237 of 1,022
held-out words exactly (0.2319, CER 0.2342; development 231 of 1,005 = 22.99%, CER 0.2345),
against dictionary spellings that are not letter for letter. On the 52 fixture words the
corpus builder held out as validation the tiny model reaches CER 0.2444 and 0.2982 (17 and
14 of 52 words exact): it memorises. Throughput, CPU, a new process per run, median of ten:
`apply` takes 0.715 s for one word, 0.79 s for 500 and 0.82 s for 1,000 (about 1,200 words/s
end to end, about 10,000 words/s once the model is loaded). Length ratio of script over
Cyrillic over the 2,027 reference pairs, quantiles 0.01, 0.05, 0.5, 0.95 and 0.99: 0.83,
1.00, 1.17, 1.50 and 1.95; they are word-level and NOT wired into
`configs/data/filters.yaml`.
Findings worth knowing: two upstream defects, both worked around and recorded in
`docs/VERIFY.md`. `D-CT2-EOLE-CUDA`: the CTranslate2 `eole_ct2` converter segfaults (exit
-11, empty `model.bin`) when a CUDA device is visible, so the export always hides the GPU.
`D-EOLE-PREFIX-OVERRIDE`: Eole copies the `valid` corpus's prefix into the checkpoint and the
checkpoint overrides the prediction config, so `eole predict` ignored the requested tag
(reverse direction: 67 of 448 words wrong, CER 0.0533 against 0.0128); `predict`
pre-tokenises and runs Eole with no transforms, and D8 and D11 per-direction Eole
evaluations must do the same or their parity gate fails spuriously. Also: Eole takes one
validation corpus (the script-to-Cyrillic direction has no validation loss); the master
plan's forbidden key names are not Eole fields (the real ones are `heads_kv`,
`sliding_window`, `relative_positions_buckets` and the rotary fields); Eole expands `$` over
the config text; `python -m eole` does not run; the master plan's "300 CPU steps reach CER
<= 0.15" is false as written (the tiny profile needs 1000 steps, dropout 0).
Open gates (none can be done without real data or a reader): the audit of U+200B, U+200C,
U+200D, U+2060, U+FEFF, U+180A..U+180F and U+202F in real traditional-script text, before the
joiner-stripping policy is frozen; the render check in Noto Sans Mongolian 2.001 and 3.103;
the review of the lexicon, the golden words and the fallback by a Mongolian reader; `mn_Mong`
length-ratio bounds for the D4 filter; training the full profile on real parallel script
data. NOT verified: any output by a Mongolian reader; the Wiktionary reference's
independence of mechanical derivation and its Mongolia-only provenance; the GPU path and the
full profile (its YAML never went through Eole's own config class); Windows and macOS; the
built wheel (that the two data files ship); every threshold on real data; throughput beyond
1,000 words on one machine. [EVIDENCE: the four gate commands, the integration runs, the
worked example, the measurement scripts and the reproductions of both upstream defects
described in the runbook and in `docs/VERIFY.md`, 2026-11-07, commits `ed7ead9` through
the final review fixes.]

**D6 (synthetic data) is done on branch `d6-synthetic-data`; its STOP gate (a tiny synth
run through the CLI with a fake generator) is met locally.** The branch was cut from `main`
at `0d534a3` on 2026-11-07, is not merged and not pushed (the overnight run made no merge
and no push), and no CI run covers it. It is 25 commits ahead of `main` as of the commit `D6: re-review minors` (the last commit
of the branch; `38297c4` was the last code commit of the final fix wave, and the re-review's
minors followed it). It is a tested
pipeline, not a pipeline that ever ran: no vLLM or llama.cpp server, no NLLB or MADLAD
CTranslate2 model and no GPU was used by any test or measurement, every client path was
tested against fakes, no real monolingual corpus exists, the prompts are the builder's
wording and were never tried on a model, and no Mongolian reader has judged anything.
Built: `rinchen.mbr` (`Utility`, `ChrfUtility`, `NegatedUtility`, `select`), `rinchen.synth`
(`tags`, `translator`, `prompts`, the shared `_chat_completions` base with the vLLM and
llama.cpp clients, `ct2_baselines` for NLLB and MADLAD, `_candidates`, `_texts`, `shards`,
`select`, `plan`, `bt`, `ft`, `run`), `rinchen.config.synth` with `configs/data/synth.yaml`,
the command `rinchen synth run` (`rinchen.cli.synth`), the test helpers `planted_synth.py`,
`synth_cli.py` and `fake_llm_server.py` in `tests/support/`, and
`docs/runbooks/06_synthetic.md`. The design is
`docs/superpowers/specs/2026-11-08-d6-synthetic-data-design.md` (updated to the finished code);
its 12 rulings and the 22 amendments A to V, made without a review round because the user
said "Proceed. I'm going to sleep now. Don't stop until you finish everything. Commit
everything.", are condensed in DECISIONS.md ("2026-11-08: D6 synthetic data"). Every task was
reviewed and fixed once, a batch of parked review minors followed (`510258d`), and an independent
whole-branch review then ran (no Critical finding, three Important ones: Ctrl-C did not cancel
the queued requests in the real binary, `write_corpus` failed for an `--out-dir` containing `[`,
and a bit flip in a shard could write wrong text into the corpus on a resume); its fix wave is
`38297c4` and corrects earlier statements, in DECISIONS.md ("2026-11-08: D6 final review -
corrections and fixes"). Gates, run for real on Linux, CPython 3.14.10 (`just` is not installed
here, so the four gate commands were run as one chain), the last gate of the code, on `38297c4`
(2026-11-08): `ruff format --check .` clean, `ruff check .` clean, `mypy src` clean, `pytest -m
"not network" -q` -- **4330 passed, 1 skipped, 10 deselected** in 193 s (3 min 16 s for the four
commands; the same counts again with these records added) (4294 on `510258d`, 2833 at the D5
merge); **4331 passed** (199 s) after the re-review's one added test, the cancelled-future guard of
the polled wait.
Evidence, all through the CLI in process with planted fakes: the 40 planted rows give exactly
sources 40, candidates 60, empty 5, purity 10, QE 10, copy 0, no candidates 5, no survivor 20
and selected 15 for both `bt` and `ft` (without QE: QE 0, no survivor 10, selected 25), in 5
shards and 5 translator calls of 8 texts, 42 ms of wall time (median of 7); the corpus
validates against `CORPUS_SCHEMA`; origins `bt` and `synth_llm`; a run that dies after shard 1
exits 4, leaves two shards, and the resumed run requests only shards 3 to 5 and ends with the
rows of an uninterrupted run (identical apart from the `created` stamp). MBR cost: `n * (n - 1)`
chrF calls per segment (12, 56 and 240 for n = 4, 8 and 16), 2.2 s, 9.9 s and 43 s per 1,000
189-character segments. Input handling, 200,000 rows of about 200 Cyrillic characters:
`read_monolingual` 0.16 to 0.20 s, `plan_shards` 0.11 s (about 544 bytes per row of Python
tuples), `job_fingerprint` 0.56 s, peak RSS about 323 MiB with 65 MiB at start (an extrapolation
to ten times the rows, about 2.5 GiB, was not measured). Mutation checks done by hand during
the build, one mutant at a time, as recorded per task: MBR 42, config 110, prompts 49, clients
121, CT2 52 (and 18 more in the fix round), shards 80, selection 58 (about 45 more in the fix
round), runner 60 (40 more after the refactor), CLI about 52, end batch 35; every survivor was
shown to be an equivalent mutant.
Findings worth knowing: four upstream facts, recorded in `docs/VERIFY.md`. `D-POLARS-GLOB`:
`pl.read_parquet` and `pl.scan_parquet` expand `[`, `*` and `?` in a path as a pattern, so a
shard directory named `run[1]` made every shard unreadable and a resume silently regenerated
everything (`glob=False` everywhere, the read-back of `write_corpus` included since the final
fix wave). `D-POLARS-PANIC`: a damaged Parquet file can raise `PanicException`, a
`BaseException` (caught); in 150,000 single-bit flips of a small shard-shaped file 276 panicked,
8 aborted the process with a Rust allocation failure (not catchable) and 65,607 decoded without
error to different data; through `ShardStore.read` the structural checks alone caught 0.3% of
the flips of a 98 KB shard (63.7% decoded to different candidates), which the build had put at
about 1% and accepted, so every shard now carries a SHA-256 of its content in the Parquet
metadata that `read` checks last (no flip of three test shards, 9,888, 26,712 and 3,000
sampled, decoded to different rows without an error; the digest is not a signature and a
damaged file can still abort the process). `D-POLARS-SIGINT`: once polars is imported a SIGINT
does not wake an untimed `concurrent.futures.wait()`, so Ctrl-C did not cancel the queued
requests (through the real binary: exit 130 6.33 s after the signal with 40 of 40 requests
sent); the clients now wait in 0.2 s slices (0.27 s and 16 of 40 once; 0.77 to 1.06 s in five repeats), and the requests already in
flight still finish.
`D-CT2-NUMHYP`: libctranslate2 4.8.4 raises "The number of hypotheses cannot be greater than
beam_size * patience" at the first batch, so `Ct2Config` rejects `n > beam_size` at load. Also:
the first draft of the spec had the QE floor in the wrong direction (MetricX-QE is lower-is-
better, D4's `max_metricx_qe`); the selection compares purity and QE in float32 as D4 does (a
three-script Japanese candidate sums to 0.8999999999999999 and failed 0.9 in float64); purity
is a script check, not a language check (Kazakh and Russian Cyrillic score 1.0); the job
fingerprint covers the operational knobs and omits the prompt wording and the model weights, so
those silently reuse stale shards; `--seed` only names the shard directory and does not reach the
server's sampling seed; per-shard selection equals one global selection for scorers that depend on
the (source, translation) pair alone (746 random comparisons in review).
Open gates (none can be done without a real generator, data or a reader): running any real
generator (the servers, the NLLB and MADLAD models, their token conventions, prompt quality,
throughput, GPU memory); a Mongolian reader's judgement of any output; calibrating the QE floor,
the script share and every default; a language identifier on the synthetic side; COMET and MetricX
as MBR utilities; training-time `<bt>` and `<ft>` insertion (D7 and D8). Known and open:
concurrent runs into one out dir are not guarded; a damaged shard file can still abort the
process; the minors the final review left (a QE chunk loop that duplicates D4's, `plan.py`
importing `Kind` from `select` and the small indirections around it, a `base_url` with a space
inside the host, no CPython 3.14 assertion at the console entry except `doctor`, an `OSError` of
`scorer.release()` worded as an output write failure), listed in the runbook. NOT verified: any
real server or model, the MADLAD and NLLB token conventions, the real MetricX-QE
scorer inside D6, Windows and macOS, memory beyond 200,000 rows and of a whole `run_job` beyond the
tiny run. [EVIDENCE: the gate chain, the integration test, the worked example, the measurement
scripts and the reproductions of the four upstream facts described in the runbook and in
`docs/VERIFY.md`, 2026-11-07 and 2026-11-08, commits `8d2cd7f` through `38297c4`; the final fix
wave added 36 tests, checked 22 mutants by hand (the two that survive are equivalent: a poll
interval of 5 s, because any timed wait is interrupted at once, and a wait without
`FIRST_EXCEPTION`, which only delays noticing a failure by at most 0.2 s), and re-ran the real
binary against a loopback stand-in for a server (the interrupt, and `out[1]`, `out*`, `out?` and
`out[a-z]`).]

**D7 (multilingual ablations) is done on branch `d7-multilingual-ablations`; its STOP gate (a tiny
ablation of 2 variants trains and evaluates through the real CLI) is met locally.** The branch is stacked
on `d6-synthetic-data` (cut from it at `9973319` on 2026-11-08), is not merged and not pushed (the
overnight run made no merge and no push), and no CI run covers it. It is 21 commits ahead of
`d6-synthetic-data` and 48 ahead of `main` as of the commit `D7: re-review minors` (the last
commit of the branch; `cbd2365` was the last code commit of the final fix wave, and the re-review's
minors follow its documents). It is machinery, not a result: the only data it
ever ran on is a templated toy set that the model saturates (chrF 96.25 to 99.72 in every direction of
both variants when D7 was built; 90.89 to 99.45 after D9's vocabulary invariant changed the vocabulary, see
below), no real corpus exists, no GPU was used, and `configs/eole/multilingual_proxy.yaml` and
`ablation_priors.yaml` are unmeasured priors that Eole's schema accepts and nothing has trained. Built:
`rinchen.train` (`eole_cfg` with the per-level allow-lists and `validate_train_config`, `merge`, `render`;
`eole_run` with `run_training` and `predict_tokens`; `sampling` with `mixing_probabilities` and
`integer_weights`; `variants`; `mixture`, `corpus_files`, `eole_data` and `validation_corpus`, the training
mixture; `evaluate`; `ablation_layout`, `ablation_config`, `ablation_plan`, `ablation_prepare`,
`ablation_records`, `ablation_report` and `ablation`, the run), `rinchen.config.ablation`,
`rinchen.utils.logged_process` (D5's runner, moved), `corpus_scores` in `rinchen.metrics.bleu_chrf`,
the command `rinchen train ablate` (`rinchen.cli.train`, a D0 stub until now), the four configs
`configs/eole/{multilingual_tiny,ablation_tiny,multilingual_proxy,ablation_priors}.yaml`, the test
helpers `tiny_parallel.py`, `tiny_variant.py`, `train_cli.py`, `fake_ablation.py`, `corpus_rows.py` and
`eole_train_config.py` in `tests/support/`, and `docs/runbooks/07_ablations.md`. The design is
`docs/superpowers/specs/2026-11-08-d7-multilingual-ablations-design.md` (amended to the finished code, with
a section listing each ruling and decision made while building); its rulings R1 to R12 and the controller
rulings, made without a review round because the user said "Proceed. I'm going to sleep now. Don't stop
until you finish everything. Commit everything.", are condensed in DECISIONS.md ("2026-11-08: D7
multilingual ablations - the rulings, made without a review round"). Every task was implemented with
tests first and reviewed once (Task 1; Task 2; Tasks 3 to 5 together; Tasks 6 to 8 together with the
`run_logged` fix), two interludes applied the reviews' pure moves and minors, and a spike on the real
Eole preceded the design. A whole-branch review (an independent reviewer, read-only, `9973319..87cabe2`)
ran after the records and found 0 Critical, 1 Important and 7 Minor findings; it reproduced the gate and
every number the records claim. The Important one: `run_logged` replaced an inherited ignore of SIGHUP, so a
`nohup rinchen train ablate` was ended (exit 129, `eole train` killed) by the hangup that `nohup` absorbs;
it is fixed (an ignored SIGTERM or SIGHUP now stays ignored; through the real binary a SIGHUP 4 s into
`eole train` of a `nohup` run now lets it finish with exit 0, and without `nohup` it still exits 129 with
no stray `eole`). The Minor ones: the `ablation_layout` docstring and runbook 07 (`train` imports `spm`)
were wrong and are corrected, `bt_ratio` is capped at 1000, a non-finite score is an `UpstreamError`
and `results.json` is written with `allow_nan=False`, and runbooks 05 and 07 and `D-LOGGED-TERM` now
record the stale variant directories of an earlier matrix, the glob refusal in D5's `run_training` and
`predict` (`export_ct2` is still unguarded), `nohup` and the SIGINT that a shell background job ignores.
Corrections of earlier statements are in DECISIONS.md ("2026-11-08: D7 final review - corrections and
fixes"). Gates, run for real on Linux, CPython
3.14.10 (`just` is not installed here, so the four gate commands were run as one chain), the gate of the
build, on `f7d1bd0` (2026-11-08): `ruff format --check .` clean, `ruff check .` clean, `mypy src` clean,
`pytest -m "not network" -q` -- **5944 passed, 1 skipped, 10 deselected** in 6 min 33 s (4331 on D6's last
commit, 4435 after Task 1, 4861 after Task 2, 5278 after Task 5, 5446 after Task 6, 5655 after Task 7,
5769 after Task 8, 5776 after the `run_logged` fix); the D5 regression set (1,245 unit tests and 23
integration tests with 1 skipped, as of the `run_logged` fix) stayed green after every move. After the
final-review fixes, on `cbd2365`: `ruff format --check .`, `ruff check .` and `mypy src` clean, `pytest -m
"not network" -q` -- **5966 passed, 1 skipped, 10 deselected** in 6 min 37 s; the D5 regression sets green;
**5968 passed** (7 min 43 s on a loaded machine) after the re-review's two added tests (a signal handled
outside Python is left alone, a failed second install restores the first).
Evidence of the STOP gate, through the real `rinchen` binary and the real Eole on the CPU with the GPU
hidden, on the toy set of `tests/support/tiny_parallel.py` (en-mn 1,200 and ru-mn 600 training pairs in
both directions, 300 back-translated pairs per language pair, 100 dev lines per direction) with
`configs/eole/ablation_tiny.yaml` (temperature 0.5, bt ratio 0.5, tag on the source and on the target,
1000 steps of 2048-token batches, 2 threads): the whole command takes 72.7 s (training 33.7 s and 33.0 s,
evaluation 2.6 s each); the integer weights are 22 6 22 6 15 7 15 7 = 100 over 1200, 300, 1200, 300, 600,
300, 600 and 300 rows; the dev chrF of eng-khk, khk-eng, khk-rus and rus-khk is 98.89, 99.23, 97.76 and
96.25 for the source tag (mean 98.03, minimum 96.25) and 99.72, 98.24, 98.32 and 98.33 for the target tag
(mean 98.65, minimum 98.24), BLEU 96.19 to 99.67, no direction tag in any hypothesis (**after D9 added every
reserved symbol to the vocabulary, the same command on 2026-11-09, code `c2d46b5`, gave 95.94, 95.85, 90.89 and
93.67 for the source tag (mean 94.09, minimum 90.89) and 99.45, 99.16, 98.13 and 98.24 for the target tag (mean
98.75, minimum 98.13), BLEU 93.11 to 99.52, no tag leak, 135.7 s on a loaded machine, the checkpoint 3,318,120
bytes where it was 3,289,000: the model has another vocabulary, so another initialisation; the values above stay
as the record of D7's build**); the saved
`config.json` of both checkpoints has a `valid` corpus without prefix keys; a rerun reuses both variants
in 0.76 s; with one `result.json` deleted the rerun only evaluates (3.37 s); with one checkpoint cut in
half the rerun trains that variant again (37.75 s), reuses the other, and the retrained checkpoint has the
same SHA-256 as the first. Those scores rank nothing: the model memorises a templated grammar of 37
nouns, one seed each. A dry run reads the same inputs in 0.71 s and creates nothing. The reviews
re-ran the gate for real and got identical numbers, and checked: a differential probe of D5's runner
and prediction against the old code over 84 scenarios (byte-identical but for two error paths), a 40,000
configuration fuzz of the validator (only `ConfigError`), real Eole accepting what `render` wrote for both
placements, a 240,000-row probe of the mixture (0.16 s load, 413 MiB), the real signals (Ctrl-C exit 130
in 0.17 to 0.36 s, SIGTERM 143 in 0.26 s, SIGHUP 129 in 0.16 s, no stray `eole`). Mixture cost, measured
for this record on synthetic rows of about 175 characters a side: 1,000,000 rows load in 0.32 s with a
peak of 1,550 MiB and build the corpus files in 2.27 s (634 MB); 4,000,000 rows load in 1.09 s with a peak
of 4,758 MiB and build in 11.43 s (2.7 GB); extrapolated, 46 GiB would be reached near 44M rows.
Mutation checks done by hand during the build, one mutant at a time, as recorded per task: Task 1 30,
Task 2 122 (one equivalent survivor), Tasks 3 and 4 every mutant killed except 3 equivalent ones, Task 5 about 135, Interlude A
about 45, Task 6 34 and more, Task 7 about 190, Task 8 54 for the CLI (52 killed at once, 1 by a new test, 1
dead code removed), the `run_logged` fix all but one equivalent, Interlude B about 95, the final-review fixes 17 (all killed); the reviewers' own
mutants: 19 of 20, 17 of 17, 29 of 29.
Findings worth knowing: upstream facts about Eole 0.6.4 and the subprocess wait, recorded in
`docs/VERIFY.md` ("D7 findings") and in DECISIONS.md. `D-EOLE-WEIGHT-SEMANTICS`: a corpus `weight` is the
number of consecutive examples per round of a round robin, so the share is `w / sum(w)` whatever the size.
`D-EOLE-UNKNOWN-KEYS`: `eole train` rejects unknown keys at every level and `build_vocab` ignores some, so
the validator carries per-level allow-lists. `D-EOLE-SUFFIX-KEYERROR`: the `suffix` transform raises
`KeyError: ''`, so a suffix tag is not supported. `D-EOLE-BATCH-DEDUP`: Eole silently drops a training
example whose tokenised source equals one in the same minibatch, which loses 1.94%, 35.6% and 50.0% of the
examples of multi-parallel toy data at 1024, 4096 and 16384 tokens when the tag is on the target (0.00%
with the tag on the source); each variant records `source_collision_rate`. The `valid` prefix leak is
`D-EOLE-PREFIX-OVERRIDE` again (a prefix-free, pre-tokenised `valid` corpus avoids it).
`D-EOLE-GLOB-CKPT`: Eole globs the checkpoint directory, so `--out-dir 'out[1]'` trained a whole variant
and then failed, and `out*` scored a sibling's weights (found in review, now refused in the plan).
`D-LOGGED-TERM`: SIGTERM and SIGHUP left `eole train` running (now handled). `D-POLARS-SIGINT` again:
Ctrl-C without `--timeout` was raised only when `eole train` ended, 25.9 s after the signal (now 0.2 s). Same
seed and same thread count give byte-identical checkpoints, different thread counts do not; the default
uniform initialisation shows a plateau until step 600 to 1000 that `xavier_uniform` removes on the toy.
Open gates (none can be done without real data or a GPU): any ablation on real corpora; a measured proxy
config (the priors are untested); the throughput and memory of a real training; Eole-native prediction
and the CTranslate2 export of a tag-on-target model; the weight resolution for many directions (at
`weight_total` 100, half of 32 corpora sit at the minimum weight 1 at temperature 1.0); a streaming
loader for hundreds of millions of rows; the D3 gates assigned to D4, the D5 gates that need real text or
a reader and the D6 gates above stay open. Known and open: concurrent runs into one out dir are not
guarded; SIGKILL of `rinchen` leaves the child running; D5's `export_ct2` fails for an out dir with glob
characters (D5 was not changed); the value types of known Eole keys are not validated; Eole's
`filtertoolong` drops are not counted; the parked review minors are listed in the runbook. NOT verified: any
run on a real corpus or a GPU, `num_workers` above 1 with weights, `xavier_uniform` at scale, Windows and
macOS, the mixture beyond 4,000,000 rows. [EVIDENCE: the gate chain, the integration tests, the runs of the
real binary and the measurement scripts of the runbook, the spike (`spike-findings.md`) and the review
reports, 2026-11-08, commits `c2db16b` through `f7d1bd0`.]

**D8 (teacher training) is done on branch `d8-teacher-training`; its STOP gate (a tiny teacher trains 200
CPU steps without NaN, the loss falls by at least 30 %, it resumes, completes and is averaged) is met
locally.** The branch is stacked on `d7-multilingual-ablations` (cut from it at `b1a8a47` on 2026-11-08), is
not merged and not pushed (the overnight run made no merge and no push), and no CI run covers it. It is 12
commits ahead of `d7-multilingual-ablations` and 60 ahead of `main` as of the commit `D8: final review fixes`
(`0378265`, the last commit before the one that carries these records: with it 13 and 61; at `3e357ae`, the
last commit of the build, it was 10 and 58). It is machinery, not a
result: the only data it ever ran on is the templated toy set of D7, no real corpus exists, **no GPU was
used** (this machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run; every Eole process
of D8 had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`), and `configs/eole/teacher.yaml` is a table of
unmeasured priors that Eole's schema accepts and nothing has trained. Built: `rinchen.train.checkpoints` (step
directories, `is_complete`, the run identity, `launch.json`), `curves` (`parse_train_log`, `loss_gate`),
`average` (`average_checkpoints`), `launch` (`launch`, the resumable launcher), `gpu_rules`, `launch_command`
(`command.sh`), `run_lock` (an exclusive `flock` that the Eole processes inherit), additions to `eole_cfg` (the
teacher keys, the `resumes` flag), `eole_run` and `corpus_files`, `rinchen.utils.integers` and
`run_logged(inherit_fds=)`, the commands `rinchen train launch` and `rinchen train average` and the option
`rinchen train ablate --prepare-only` (`rinchen.cli.train_runs`, `train_common`, `train`), the configs
`configs/eole/{teacher,teacher_tiny}.yaml`, the test helpers `step_dirs.py`, `fake_launch.py`,
`fake_eole_process.py`, `launch_cli.py` and `signal_handlers.py` in `tests/support/`, and
`docs/runbooks/08_teacher.md`. The design is `docs/superpowers/specs/2026-11-08-d8-teacher-training-design.md`
(amended to the finished code, with a section listing each ruling and decision made while building); its
rulings S1 to S10 and the controller rulings, made without a review round because the user said "Proceed. I'm
going to sleep now. Don't stop until you finish everything. Commit everything.", are condensed in DECISIONS.md
("2026-11-08: D8 teacher training - the rulings, made without a review round"). Built task by task with tests
first after a spike on the real Eole (Tasks 1 and 2; 3 and 4; 5; 6, the STOP gate; an interlude; Task 7, the
optional two-rank CPU gloo test, was SKIPPED and the runbook says so). One review ran (an independent
reviewer, Tasks 1 to 6, `ab405e7..7c57d76`): no Critical finding, three Important ones, each reproduced and
fixed in the interlude (a SIGKILL of the launcher released the run lock while Eole kept training, so the next
launch started a second Eole into the same checkpoint directory; a run started by hand from `command.sh` left
no log, so a later launch saw neither a NaN nor an early stop; `is_complete` raised for a damaged zip central
directory), and eleven Minor ones (four fixed, the rest documented in the runbook). **A second, independent review of the
whole branch (`b1a8a47..90c3865`, read-only) then ran: Ready with fixes, no Critical finding, two Important
ones and seven Minor ones; it reproduced the gate twice and every number of the STOP gate.** Its fix wave (the
commits `D8: final review fixes` and `D8: final review fixes, docs`) fixed I-1 (`run_logged` did not kill the
process group when Eole's leader exited non-zero on its own, so the spawned rank processes of a multi-GPU run,
which hold no lock, would have trained on and the next `launch` started a second training into the same
checkpoint directory; the group is now killed after a failed leader, a clean exit is left alone), I-2 (the
TensorBoard events of `tensorboard: true` landed in the working directory of whoever launched: `launch` now
defaults `tensorboard_log_dir` to `<run directory>/tensorboard` in the effective config, so it is in every
attempt config and in the run identity), M-1 (a step directory copied in from a run of another architecture
was resumed from, because Eole ignores the config's `model` section on a resume: `launch` now compares the
saved model with the config's and refuses, best effort, a copy of the same architecture is not detected) and
M-4 (the real exception class in the `is_complete` message). Left open on purpose: M-2 (raising `train_steps`
after an early stop does nothing), M-3 (two hand runs appended to one attempt log: the NaN scan spans it), M-6
and M-7 (design notes); the records in `DECISIONS.md` ("2026-11-08: D8 final review - corrections and fixes")
correct the statements the review showed false (the group was NOT always killed; the TensorBoard events did NOT
stay in the run directory; `eole_cfg.py` has 837 lines, not "about 844"). **A scoped re-review of the fix wave
ran afterwards: approved with minors (a quoted number in a config was taken for a foreign checkpoint: fixed in
`D8: re-review minors`, gate 7194 passed, 1 skipped, 10 deselected in 10 min 23 s; the other minors are documented in
the runbook).** Gates, run for real on Linux, CPython 3.14.10: `ruff format --check .`, `ruff check .` and
`mypy src` clean and `pytest -m "not network" -q` -- 6148 passed after Task 1, 6485 after Task 2, 6588 and 6829
after Tasks 3 and 4, 6853 and 7020 after the two commits of Task 5, 7048 after Task 6 (10 min 36 s on a loaded
machine), 7085 and then **7114 passed, 1 skipped, 10 deselected** after the interlude (10 min 10 s and 10 min 50
s; 5968 on D7's last commit) and again, with the build's records in the tree (only documents had changed), 7114
passed, 1 skipped, 10 deselected in 10 min 34 s of pytest (10 min 37 s with ruff and mypy, which were clean), and
**7187 passed, 1 skipped, 10 deselected** after the final review's fix wave (`D8: final review fixes`, 10 min 19 s
of pytest, 10 min 22 s with ruff and mypy, clean); the D5 and D7 regression sets stayed green (1284 passed at the
interlude, 1839 passed (1831 before) after the fix wave), as the plan required. Mutation checks by hand, one
mutant at a time, as recorded per task: Task 1 36, Task 2 70 (one equivalent removed), Task 3 about 145, Task 4
about 110, Task 5 14 and about 74 (three equivalent survivors), the interlude about 50, the final review's fix wave
36 (one equivalent); the first reviewer's own 25 of 25.
Evidence of the STOP gate, through the real `rinchen` binary and the real Eole on the CPU with the GPU hidden,
on the toy set of `tests/support/tiny_parallel.py` with `configs/eole/teacher_tiny.yaml` (64 wide, 2+2 layers,
2048-token batches, 200 steps, a validation and a save every 50, 2 threads), measured again for the runbook on
2026-11-08 (Intel Core Ultra 7 265K, 20 cores, 46 GiB): `ablate --prepare-only` 0.78 s; `launch` from nothing
25.61 s, `completed`, training `xent` 5.44, 5.40, 4.64, 2.81, 2.28, 2.14 at steps 10, 20, 50, 100, 150, 200 (a
60.7 % fall; the gate asks 30 %), validation perplexity 83.6869, 16.6115, 10.1611, 8.81733, no NaN, the weights
of `step_200` SHA-256 `32a1c92e...` (the spike's value for an uninterrupted run; **after D9's vocabulary
invariant, 2026-11-09: `xent` 5.81, 5.76, 4.93, 2.91, 2.33, 2.13, a 63.3 % fall, validation perplexity 106.851,
18.3535, 10.4948, 8.73594, the weights `6f5e836c7553...`, 3,318,120 bytes, in 34.5 s on a loaded machine; the
values of 2026-11-08 stay as the record of D8's build**); a second `launch` 0.73 s,
`already_complete`, the run directory unchanged; with the weights of `step_200` cut in half, `launch` resumed
from step 150 (9.60 s, `incomplete step directories skipped: [200]`, the log starting at `Step 160/  200`,
`launch.json` unchanged) and finished with other weights (`7e62604f...`, after D9 `b665b997...`: a resumed run is not the
uninterrupted one, because the data stream restarts at line 1 of every corpus), a separate run of 100 steps
extended to 200 resumed from step 100 (`Step 110/  200`, 13.50 s) and lowering `train_steps` or changing the
learning rate was refused (exit 2, nothing written); `average --last 3` (steps 100, 150, 200) 0.74 s, 58 of 62
tensors averaged; a real SIGTERM ended the launcher with 143 in 0.21 to 0.26 s with no Eole process left and
the next launch finished the run; a SIGKILL of the launcher left the Eole running, which kept the lock, so the
next launch was refused (exit 2) and the one after the orphan ended said `already_complete`. Those numbers
rank nothing: the model memorises a templated grammar. ARITHMETIC, not measurement (the formula reproduces the
real tiny checkpoint exactly, 820,458 tensor elements and 3,289,000 bytes; 827,738 elements and 3,318,120 bytes
after D9, with a vocabulary of 344 instead of 232): the teacher shape (24+12 layers,
hidden 1536, feed-forward 4096) has 1,004,601,868 parameters with a 64,000-piece vocabulary and 1,102,969,868
with 128,000, a step directory of 12.1 and 13.3 GB (fp32 weights plus 8 bytes per parameter of Adam state), and
the config's batch is 8 x 8 x 16384 = 1,048,576 padded tokens per update.
Findings worth knowing: upstream facts about Eole 0.6.4, in `docs/VERIFY.md` ("D8 findings") and in
DECISIONS.md. A resume is `training.train_from: <model_path>/step_N` with an absolute `train_steps`; the data
stream restarts, the RNG and the EMA are not restored, scheduled lists restart, the optimiser options come from
the checkpoint (a new learning rate is silently ignored, hence the identity rule); Eole's saver is not atomic
(a kill mid-save leaves a silent mix of two steps when resumed through `model_path`); Eole has no averaging tool
and `average_decay` saves the EMA in place of the live weights; a NaN loss does not stop Eole (exit 0, NaN
checkpoints); `eole train` spawns the per-GPU processes itself (no torchrun), `world_size` 2 with no `gpu_ranks`
is a silent no-op, `timeout` (60 s) is the collective timeout and must be hours because rank 0 alone validates
and saves, early stopping runs on rank 0 only and a multi-process job then dies with exit 1 after the timeout,
the statistics gather has a 4096-byte buffer (about 100 corpora), the loss is divided by `world_size` twice
(`max_grad_norm` acts `world_size` times looser), and the wheel ships no training recipes. The checks of
`is_complete` catch every truncation (800 of 800) but not a flipped byte in the tensor data (random single-byte
flips flagged: weights 3 of 200, optimizer 1 of 200, `config.json` 138 of 200, `vocab.json` 179 of 200).
Averaging neither helps nor hurts beyond noise on the toy and hurts in the steep part of the curve.
Open gates (none can be done without real data or a GPU): everything on a GPU (bf16, NCCL, flash attention,
multi-GPU, multi-node, the tokens per second, the memory and the 13 GB checkpoint write of the teacher shape,
a one-GPU smoke test that the owner has to authorise); the quality of the teacher shape and every value of
`teacher.yaml`; whether averaging helps a real teacher; the real two-rank early stop (only unit fakes and the
spike's CPU shim); a live abort on NaN (it is detected when an attempt ends); `flock` on network file systems;
the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 and D7 gates stay open.
Known and open: `--write-only` leaves an existing manifest with a stale `command.sh` digest; an
`already_complete` rerun re-hashes the final step directory (GB scale); a hand-run `command.sh` is not guarded
by the lock; the lock is held by Eole's leader only (the rank processes of a multi-GPU run hold none), so a rank
can outlive a leader that died while the launcher was gone too; the comment of `teacher.yaml` cites master plan
lines 511 to 514 where the teacher block is 511 to 516 (configs were not touched); M-2, M-3, M-6 and M-7 of the
final review (above); the parked minors are listed in the runbook. NOT verified: any run on a GPU or a real corpus, the multi-process path under NCCL, Windows and
macOS. [EVIDENCE: the gate chain, the integration test module, the real-binary runs and the arithmetic script of
the runbook, the spike (`spike-findings.md`) and the review reports, 2026-11-08, commits `36559a1` through
`0378265`.]

**D9 (document-level) is done on branch `d9-document-level`; its STOP gate (a tiny doc model, made by
continuing the averaged tiny teacher on windows of 2 to 4 sentences joined with `<sep>`, decodes `<sep>`-aligned
output for at least 90 % of held-out dev windows of each size, and the chunker returns exactly one translation per
input sentence for every toy test document) is met locally.** The branch is stacked on `d8-teacher-training` (cut
from it at `993f32c` on 2026-11-09), is not merged and not pushed, and no CI run covers it. It is 16 commits above
`d8-teacher-training` and 78 above `main` as of its tip (the last code commit is `135e9f9`, `D9: re-review minors`;
the earlier code tip was `c2d46b5`, `D9: review minors`; the records are `6746ac5` and, after `135e9f9`, `D9: record
the re-review`). It is machinery, not a result: the only data it ever ran on is the templated
toy documents of `tests/support/tiny_parallel.py` (documents of 4 to 8 sentences from 37 nouns), no real document
corpus exists, **no GPU was used** (the owner did not authorise it; every Eole process of D9 had
`CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`), and `configs/eole/teacher_doc.yaml` is a table of unmeasured priors
that Eole's schema accepts and nothing has trained. Built: the vocabulary invariant (`eole_run.add_reserved_symbols`
and `run_training(reserved_symbols=)`, `vocabulary_size.py`, `src_vocab_size` = pieces + 1), `predict_tokens(max_length,
max_length_ratio)`, `rinchen.train.{docify,doc_source,doc_pack,doc_corpus}` (windows, counted exclusions, doc corpora,
validation windows, the rendered continuation config), `continuation.py` (the provenance rule and the checks of the
config), the extensions of `checkpoints` (`launch.json` version 2, `continued_from`), `launch` (`--continue-from`),
`eole_cfg` (the `continues=` rules; `eole_cfg_keys.py`) and `train_runs` (the `average` guard),
`rinchen.infer.doc_chunker`, the command `rinchen train docify` (`cli/train_docify.py`) and the option
`rinchen train launch --continue-from`, the configs `configs/eole/{teacher_doc_tiny,teacher_doc}.yaml`, the test
helpers `doc_cli.py` and `doc_rows.py` (and extensions of `step_dirs.py`, `fake_launch.py`, `tiny_parallel.py`) in
`tests/support/`, and `docs/runbooks/09_document_level.md`. The design is
`docs/superpowers/specs/2026-11-09-d9-document-level-design.md` (amended to the finished code, with a section listing
each amendment made while building); its rulings T1 to T9 and the controller rulings, made without a review round
because the user said "Proceed. I'm going to sleep now. Don't stop until you finish everything. Commit everything."
(later "continue"), are condensed in DECISIONS.md ("2026-11-09: D9 document-level - the rulings, made without a
review round"). Built task by task with tests first after a spike on the real Eole (Tasks 1 and 2 together; 3; 4; 5,
the STOP gate; then one combined review, one fix wave and a scoped re-review, then these records). **The combined
review (an independent reviewer, read-only, Tasks 1 to 5, `993f32c..98e42d1`) found: ready with fixes, 1 Critical,
5 Important and 12 Minor findings.** The Critical one: a document was keyed without its corpus, so two corpora that
share a `doc_id` (D4 names a document after its file) were joined or interleaved into one window. The Important ones:
`read_corpus_rows` decoded the text of every split, test included; `rinchen train average` let hand-laid step
directories become a valid `--continue-from` source; the SentencePiece model of the config was never compared with
the teacher's; an empty `vocab.src` became 118 reserved symbols and nothing else; `launch` did not enforce
`src_vocab_size >= pieces + 1`. The fix wave (six `D9: fix ...` commits, each with a failing test first, and `D9:
review minors`) fixed all of them, the Minor ones that were cheap, and left open on purpose the five that
`docs/runbooks/09_document_level.md` lists under "Known limits" (only five optimiser options are checked, the
chunker's budget counts the source only, a `<sep>` in a fallback translation passes, resuming needs the source, a
literal `<doc>` in a sentence is not guarded). **A scoped re-review of the fix wave (`98e42d1..c2d46b5`, an
independent reviewer, read-only) approved it with minors: no Critical and no Important finding.** A second small fix
wave (`D9: re-review minors`) fixed: M1 (the pre-flight of `launch` skipped an unstated `src_vocab_size` and
`tgt_vocab_size`, but Eole's default is 32,768 and the reserved symbols are the last lines of the vocabulary file, so
a config with a model of more than 32,767 pieces and no size would lose them; `launch` now refuses it and names the
value to state, never setting it itself), M3 (`docify` refuses a split outside `train` and `dev`, `classify_rows`
refuses rows read without the split, `read_corpus_rows` refuses a bare string for `splits`), M6 (stale docstrings, two
over-long comment lines, the module-level import in `tests/conftest.py`) and the wording of the `average` guard, which
checks only that a `launch.json` that parses exists within three parent directories of the step directories (a
hand-made `step_<N>` inside a real run directory passes). Limits left open and documented (runbook 09, "Known limits"
15 to 17): M2 (the `average` guard and the `verify_source` lookup see only the three lexical parents: a symlinked
`ckpt` outside any run directory and a `model_path` nested three or more levels inside the run are refused; pass the
real path), M4 (C1 is fixed per `corpus` label, not per file: D4 sets `doc_id` to the stem of the source file, so two
files with the same label and stem still share a document key and only the counted `duplicate_seg_idx` signals it), M5
(the `launch` pre-flight runs on every plan, `--write-only` and a finished-run relaunch included, and needs the
SentencePiece model readable from the working directory when a size is stated). The size-4 margin of the gate stays
thin (.9406 against 0.9, one seed; 3,000 steps measured worse, .9356, so 2,500 stays; if it flakes elsewhere raise
`train_steps`, never lower the gate). Gates, run for real on Linux, CPython 3.14.10: `ruff
format --check .`, `ruff check .` and `mypy src` clean and `pytest -m "not network" -q` -- 7279 passed after Task 1,
7446 after Task 2, 7799 after Task 3, 8024 after Task 4, 8075 after Task 5 (1,090 s) and **8214
passed, 1 skipped, 10 deselected** after the fix wave (`D9: review minors`, 929 s), **8252 passed, 1 skipped, 10
deselected** after the second fix wave (`D9: re-review minors`, 1,217 s), against 7187 at the end of D8;
the D5, D7 and D8 regression sets stayed green. Mutation checks by hand, one mutant at a time, as recorded per task:
Task 1 31, Task 2 about 73, Task 3 109, Task 4 68 for the chunker and 110 for the CLI (10 survivors of the first CLI
pass got tests).
Evidence of the STOP gate, through the real `rinchen` binary and the real Eole on the CPU with the GPU hidden, on the
toy documents with `configs/eole/teacher_doc_tiny.yaml` (the tiny teacher's model, 2,500 steps, a validation and a
save every 500), measured again for the runbook on 2026-11-09 (Intel Core Ultra 7 265K, 20 cores, 46 GiB; the machine
was loaded by other work, so the times are longer than idle): the tiny teacher through `launch` (200 steps) and
`average --last 2` (1.43 s); `docify` over the toy train and dev documents (2.73 s; 4,000 rows read, 3,380 in windows,
1,194 windows of sizes 2, 3 and 4 numbering 514, 368 and 312, 220 tail rows, the 400 dev rows `not_train`, 136 dev
windows, `--max-pieces` 126, source windows of 12 to 52 pieces, weights 33 33 17 17); `launch --continue-from <the
averaged teacher>` for 2,500 steps (107 s here, 87 s idle; training `xent` 3.07 at step 50 and 0.97 at the end,
validation perplexity 7.34, 5.99, 3.63, 3.29, 3.21; the teacher's tree byte-identical afterwards; `launch.json`
version 2 with `continued_from`; two runs of the continuation gave the same weights at all five checkpoints). The
gate, the alignment of the 804 held-out stride-1 dev windows (334, 268 and 202 of the sizes 2, 3 and 4; a window is
aligned when its output has one segment per sentence and none is empty): .9641/.1716/.0000 at step 500,
.9521/.8470/.8861 at 1000, .9731/.9664/.9158 at 1500, .9880/.9664/.9109 at 2000 and **.9820/.9776/.9406 at 2500**,
at least 0.9 for every size; a 3,000-step run ended at .982/.974/.936, so the config stays at 2,500. **1,000 steps,
the spike's figure and the plan's, did NOT reach 0.9 on the non-overlapping windows `docify` cuts** (the spike
trained on 7,260 overlapping ones; docify's 1,194 gave .955/.847/.886 at 1,000): the threshold was not lowered, the
steps were raised. The chunker over the toy test split (66 document-direction pairs, 400 sentences): 106 windows, 102 aligned, 4 fallbacks,
18 singles, 0 oversize, one non-empty translation per sentence in every document, chrF 70.62 against 10.44 for pure
sentence-level translation (recorded, not gated). Real-binary refusals, each exit 2 in about a second with Eole not
started: a step directory with no run, a flipped weights byte, a vocabulary without `<doc>`, a config without an
architecture key, another SentencePiece model, `average` of hand-laid step directories, a stated `src_vocab_size` of
100, `docify --max-pieces` above the base config's limit. A SIGTERM after the first checkpoint and a second `launch`
resumed from the run's own step with `reset_optim: none` and continued at the next step. **The vocabulary invariant
changed every D7 and D8 tiny-run number** (the tiny teacher's `vocab.src` 225 to 339 lines, its checkpoint's
`vocab.json` 232 to 344 entries, the weights of `step_200` 3,289,000 to 3,318,120 bytes and SHA-256 `32a1c92e...` to
`6f5e836c...`, `xent` 5.44 and 2.14 to 5.81 and 2.13, the validation perplexities, the D7 ablation chrF of the
source-tag variant by up to 5.9 points); those records were updated with the old values kept as history (D5's
converter is identical).
Findings worth knowing: upstream facts about Eole 0.6.4, in `docs/VERIFY.md` ("D9 findings") and in DECISIONS.md.
**Eole's vocabulary is the file of tokens the data showed, not the SentencePiece vocabulary**, so a symbol the data
never showed (`<doc>`, `<sep>`) is `<unk>` to the model and continuing it on document data is silent (alignment
0.00); hence the invariant. `reset_optim: all` loads the weights only and counts steps from 1; `none` ignores every
new optimiser option; Eole fills any key a YAML omits from the checkpoint's own config, which is why the attempts
after the first of a continuation must STATE `reset_optim: none` (a plan deviation found with the real Eole: an
omitted key restarted a "resume" at step 1). The `model:` section of a continuation is merged unchecked. An averaged
directory works as `train_from`. `eole predict` has no source-length limit and truncates the output silently
(`max_length` 250, `max_length_ratio` 2); `filtertoolong` counts pieces after `prefix` and drops silently.
Open gates (none can be done without real data or a GPU): a document model on real documents and its quality; the
real-corpus doc workflow (the mix with sentence-level data, the window sizes, the number of steps and the learning
rate: `teacher_doc.yaml` is a table of priors); everything on a GPU; the time to hash and to read a real teacher's
13 GB step directory; the CTranslate2 export of a document model and the chunker in a serving path (D11); the D3
gates assigned to D4, the D5 gates that need real text or a reader and the D6, D7 and D8 gates stay open.
Known and open: the limits listed in the runbook (five optimiser options only; the chunker's source-only budget and
the silent output truncation; `<sep>` in a fallback result; resuming needs the source; `<doc>` in a sentence; the
triple definition of `<doc>` and `<sep>` pinned equal by a test); the `average` guard and the provenance rule guard
against mistakes, not forgery; the SentencePiece comparison reads a relative path from the working directory; a
`launch` config that states no `src_vocab_size` gets Eole's 32,768; `launch.py` has 886 lines and `continuation.py`
777 (the limit is 900); the size-4 margin of the gate (.941 against .9, 202 windows, one seed) is thin and the
training is repeatable only on the machine it was measured on: if CI flakes, raise `train_steps`, never lower the
gate. NOT verified: any run on a GPU or a real corpus, a different SentencePiece model inside Eole, `read_corpus_rows`
beyond toy scale, Windows and macOS. [EVIDENCE: the gate chain, the integration test module, the real-binary runs and
the measurement scripts of the runbook, the spike (`spike-findings.md`) and the review reports, 2026-11-08 to
2026-11-09, commits `580871d` through `135e9f9`.]

**D10 (distillation and students) is done on branch `d10-distillation`; its STOP gate (through the real `rinchen`
commands and the real Eole and CTranslate2 on the CPU: a tiny teacher of 2,000 steps is exported, its KD corpus of the
toy training sources is generated, a tiny student of 2 encoder and 1 decoder layers trains for 2,000 steps from random
initialisation on KD text mixed 1:1 with the originals, its dev chrF is at least 95.0 and at least the teacher's minus
3.0, and its own CTranslate2 export decodes what Eole decodes on at least 95 % of 200 dev sentences) is met locally.**
The branch is stacked on `d9-document-level` (cut from it at `7d0c166` on 2026-11-09), is not merged and not pushed, and
no CI run covers it. It is 16 commits above `d9-document-level` and 94 above `main` (the design commit is `e5ff644`; the code tip
is `233ba33`, `D10: re-review minors`; the code tip before it is `cc917e2`, `D10: fix the vocabulary padding that a teacher
hypothesis could hold`; the gate is `e67b05e`; the records are `D10: runbook, spec amendments and state and decision
records` and `D10: record the re-review`). It is machinery, not a distillation result: the only data it ever ran on is the templated
toy sentences of `tests/support/tiny_parallel.py`, no real corpus and no real teacher exist, **no GPU was used** (the owner
did not authorise it; every Eole process had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`, every CTranslate2 translator
was built on `device="cpu"`, the converter runs with the device hidden), and `configs/eole/student_server.yaml` and
`student_edge.yaml` are tables of unmeasured priors that Eole's schema accepts and nothing has trained. Built:
`rinchen.export.eole_ct2` (D5's converter call MOVED out of `script_conv.export`, which re-exports it),
`rinchen.train.{distill_export,distill_export_files,distill_saved_config}` (a checkpoint of ours exported to CTranslate2
with a record and a manifest; the tag placement read back from the saved config; `verify_export`),
`distill_teacher` (the teacher as D6's `Translator`; six counters), `distill_sources`, `distill_shards`,
`distill_generate` (KD generation over D6's shard store and selection, with a fingerprint that covers the teacher's
files), `distill_corpus`, `distill_mix` (the exact integer ratio) and `distill_prepare` (the student's data and config),
the commands `rinchen distill export|generate|prepare|evaluate` (`cli/distill*.py`), the configs
`configs/eole/student_{server,edge,tiny}.yaml`, one dict entry in `synth/tags.py` (generator kind `teacher` gives origin
`ft`), the keyword `kd_tag` of `eole_data.data_section` and the group `kd`, the test helpers `distill_{cli,exports,
gate,gate_refusals,gate_stages,teachers}.py`, `fake_ct2.py`, `planted_kd.py` and `planted_student.py` in `tests/support/`,
and `docs/runbooks/10_distillation.md`. The design is `docs/superpowers/specs/2026-11-09-d10-distillation-design.md`
(amended to the finished code, with section 8 listing each amendment made while building); its rulings T1 to T16 and the
controller's two rulings (partial KD coverage scales the originals of a direction without KD by K+O; the commands hash
every input once) are condensed in DECISIONS.md ("2026-11-09: D10 distillation - the rulings, made without a review
round"), made by the builders and the controller without a review round of their own (none is an approved decision, each lists
what it costs if wrong). Built task by task with tests first after a spike on the real Eole and CTranslate2 (Tasks 1 to
7, then ONE combined review, a fix wave, a scoped re-review with a small wave of its own, and the records). **The
combined review (an independent reviewer, read-only, Tasks 1 to 7,
`e5ff644..e67b05e`) found: ready with fixes, 0 Critical, 2 Important and nine groups of Minor findings.** The Important
ones: the tag placement of an export was the operator's word and never checked although the checkpoint's saved
`config.json` records it (a source-tag teacher exported with `--tag-placement tgt` exited 0 with a vouching manifest, and
`generate` over it gave 1,190 rows of which 1 matched an authentic target); and a KD hypothesis holding the unknown piece,
a reserved symbol or a direction tag decoded to literal text and passed D6's selection. The fix wave (`D10: fix the tag
placement of an export that its checkpoint contradicts`, `D10: fix the reserved and unknown pieces in a teacher
hypothesis`, `D10: review minors`, `D10: fix a placement test that read the files of other tests`, `D10: fix the
vocabulary padding that a teacher hypothesis could hold`) fixed both, the cheap Minor ones (the export is built in a
private directory beside the output and renamed into place; the vocabulary entry count is cross-checked; the student
headers; a message) and left open on purpose the ones that `docs/runbooks/10_distillation.md` lists under "Known limits"
(`generate` exits 0 with few or no rows; a resumed generation reports this process's teacher counters only; "exact" means
exact weights; memory and scale; no collision rate printed for the `tgt` placement; the gate's test file of 893 lines).
**The scoped re-review of the fix wave (an independent reviewer, read-only, `e67b05e..cc917e2`) approved it with minors: 0 Critical, 0 Important, six Minor findings and a few nits.** Its probes on real artifacts (CPU, GPU hidden): a wrong tag placement is refused with exit 2 and leaves nothing (no output directory, no private directory, no log), for `export` and `evaluate`; SIGINT during an export exits 130 and SIGTERM 143, both removing the private directory, while a SIGKILL leaves the hidden `.<name>.<random>/export/` directory; the placement rule derived the right placement of 323 launcher-written `src` configs and 8 `tgt` ones (no false refusal); ten real sources through the teacher with beam 4 and `n` 4 gave `reserved_symbol` 0. The minors fixed in ONE commit (`233ba33`, `D10: re-review minors`, tests first, 21 mutants): the converter's "exited 0 but left no `model.bin`" message now names `--out-dir`, not the private directory that is gone (`export_ct2` takes an optional `reported_dir`); the refusals of `distill_saved_config.py` quote corpus names and prefixes escaped and cut to 80 characters; and `distill export` and `distill evaluate` print one line, `tag placement: checked against the saved config` or `tag placement: stated, not recorded in the checkpoint`, and write `tag_placement_check` (`checked` or `stated`) into the manifest status (the export record is unchanged). Documented, not fixed: a hard kill (SIGKILL, an out-of-memory kill, power loss) of an export leaves a hidden `.<name>.<random>/export/` directory beside the output, as large as the model, safe to delete when no export is running and swept by nothing; exports made before the check hold no saved config and cannot be re-checked (re-export is the remedy); shards generated before `cc917e2` were made without the reserved-symbol filter and the fingerprint has no rule version, so the same inputs reuse them (delete the `shards-*` directories made before it, or use `--force`); and the nits (`saved_tag_placement` is used by tests only, the two file readers live in `distill_saved_config.py`, the reserved set is built after the model load, an interrupt between the rename and the `published` flag). 
Gates, run for real on Linux, CPython 3.14.10: `ruff format --check .`, `ruff check .` and `mypy src` clean and
`pytest -m "not network" -q` -- 8517 passed after Task 1, 8676 after Task 2, 8808 after Task 3, 9064 after Task 4, 9220
after Task 5, 9395 after Task 6, 9439 after Task 7 (21:08), 9706 passed, 1 skipped, 10 deselected after the fix
wave (`cc917e2`, 20:57) and **9750 passed, 1 skipped, 10 deselected** after the re-review wave (`233ba33`, 20:29), against 8252 at the end of D9; the D5 to D9 regression sets stayed green. Mutation checks by hand,
one mutant at a time, as recorded per task: 63, 88 (2 equivalent survivors), 95 (and 22 fingerprint mutants), 109, 87 and 65
for Tasks 1 to 6, ten classes for the gate, 23 for the placement fix and 21 for the re-review wave.
Evidence of the STOP gate, through the real `rinchen` binary on the CPU with the GPU hidden (2 threads), measured by the
gate and again for the runbook on 2026-11-09 (Intel Core Ultra 7 265K, 20 cores, 46 GiB; the machine was loaded by other
work, so the times are longer than idle; every count and score below was the same in the spike, in the gate and in the
runbook's run): the tiny teacher through `ablate --prepare-only`, `launch` (2,000 steps, 78 s) and `average --last 2`;
`distill export` (5.1 s; `model.bin` 2,034,311 bytes, SHA-256 `b7f33d88d5fa...`, `shared_vocabulary.json` 344 entries,
2/2 layers, hidden 64); `distill generate` (2.8 s): 3,600 sources, 3,599 rows (1 `no_survivor` and 1 `dropped_purity`,
both `khk_Cyrl-rus_Cyrl`), `matches reference` 3,583 of 3,599 (1,200, 1,191, 597 and 595), 10 shards, the rerun reuses 10 of
10 with the same rows, the teacher counters all 0 but `sources` 3,600 (`reserved_symbol` 0); `distill prepare --mix 1:1` (1.7 s):
original weights 22, 6, 22, 6, 15, 7, 15, 7 and KD weights 28, 28, 22, 22 (200 in all, 12 corpora, 400 validation lines,
`src_vocab_size` 641), `--mix 3:1` 84, 84, 66, 66 (exactly 3 times, a dry run writes nothing); the student through `launch`
(65 s; 2 encoder and 1 decoder layers in its saved `config.json`); `distill evaluate` (4 s each) on 100 dev lines per
direction: teacher chrF **98.20** (98.51, 98.29, 97.79, 98.20), student **98.58** (98.82, 99.30, 97.54, 98.67), no tag leak,
no dev line in training; `distill export` of the student (3.8 s; `model.bin` 1,835,438 bytes); CTranslate2 greedy against
`eole predict` greedy **200 of 200 equal**, "This is the star." gives the Mongolian "Энэ бол од."; a second student with a
500-piece SentencePiece model prepared (`src_vocab_size` 501) and trained for 200 steps; fifteen refusals, each exit 2 in 1.0
to 1.3 s with Eole and the converter not started. The thresholds (teacher chrF at least 90.0, student at least 95.0 and at
least the teacher's minus 3.0, CTranslate2 equal to Eole on at least 95 %, at least 3,550 KD rows) were not lowered. **The toy
ranks nothing**: three seeds of the student span 1.9 chrF (98.58, 98.59, 96.68), the originals-only student (97.01) is inside
the spread, and 99.6 % of the KD targets equal the authentic targets: do not read the gate as evidence that KD helps.
Findings worth knowing: upstream facts about CTranslate2 4.8.4 and Eole 0.6.4, in `docs/VERIFY.md` ("D10 findings") and in
DECISIONS.md. CTranslate2 maps an unknown tag to `<unk>` silently, cuts a hypothesis at `max_decoding_length` and a source at
1,024 pieces silently (ask for the end token and cap the source yourself), and returns a forced target prefix as the first
token of every hypothesis; its converter keeps the whole padded vocabulary of the checkpoint, so the model can emit `<blank>`
and padding tokens that decode to their letters. **Selecting by MBR over the beam's n-best lowered the agreement of the KD
targets with the references on the toy (99.6 % to 76.4 %; ranks 2 to 4 are 41 chrF below rank 1)**, so the top hypothesis is
the default. **Eole's batch deduplication drops 5.2 % of the examples when the KD corpus is mixed untagged (798 of 15,360), and
0 % with the `<ft>` tag**, which `prepare` adds by default. A student with unequal depths (2/1) trains from random
initialisation with no code change; a 3/1 shape at the same recipe did not converge in 2,000 steps (chrF 38). The tag
placement IS in the saved `config.json` of a checkpoint our launcher wrote, which the spike had missed.
Open gates (none can be done without real data or a GPU): a real teacher's KD corpus and the measured quality of a student
distilled from it; the real mix ratio, the repeat factor that is too high, the n-best and MBR on a real teacher and the QE
thresholds (the real scorer was never loaded); the real students' training (`student_server.yaml` and `student_edge.yaml` are
priors: speed, convergence, learning rate, steps, the depth recipe); the real share of Eole's deduplication and whether `<ft>`
at inference helps; everything on a GPU or in float16 or int8, the export of a 1.1 B-parameter teacher (time, memory, a 13 GB
checkpoint's hashing); the strict parity and quantisation gates (D11); the D3 gates assigned to D4, the D5 gates that need real
text or a reader and the D6 to D9 gates stay open.
Known and open: the limits listed in the runbook (`generate` exits 0 with few or no rows; a resumed generation reports this
process's counters only and the manifest key `teacher.name` is the KD corpus name; "exact" means exact weights, `num_workers`
above 1 with weights was not run and `--mix 10000:9999` has no guard; memory and scale unmeasured; no collision rate printed by
`prepare`; `distill evaluate` on an unparseable `config.json` exits 3; the export record does not say whether the placement was
derived or stated, which the manifest and a printed line now do; the "cannot continue from" wording of `verify_source`;
the hidden directory a hard-killed export leaves; exports made before the check; shards generated before `cc917e2`);
`distill_export.py` has 775 lines, `eole_cfg.py` 860, `launch.py` 886 and the gate's test file 895 (the limit is 900); the student gate's margin (95.0 against 98.58, the lowest seed
96.68): if it flakes elsewhere raise `train_steps`, never lower the gate. NOT verified: any run on a GPU or a real corpus,
CTranslate2 on another device, float16 or int8, `num_workers` above 1 with weights, the `tgt` placement for a real student,
KeyboardInterrupt with a real CTranslate2 call, Windows and macOS. [EVIDENCE: the gate chain, the integration test module, the
real-binary runs and the measurement scripts of the runbook (`scratchpad/d10-t8/`), the spike (`spike-findings.md`) and the
review reports, 2026-11-09, commits `e5ff644` through `233ba33`.]

**D11 (export, quantization, serving, CLI and package) is done on branch `d11-export-serving`; its STOP gate ("`docker compose up` serves a tiny
model") is met IN PARTS locally, and the container part is UNMEASURED: through the real `rinchen` commands and the real Eole and CTranslate2 on the CPU, the
tiny teachers are exported to float32, int8 and int8_float16, the strict parity gate (Eole greedy against CTranslate2 float32 greedy: at least 95 % of 200
sentences equal, mean |delta log-probability| per token at most 1e-3) and the int8 quantisation gate (chrF drop at most 1.0) pass, the script gate (0 forbidden-script
letters in 500 outputs with suppression, 96 without) passes, `rinchen.Translator` and `rinchen translate` agree with the engine, and `rinchen serve` (the command line the
compose file holds, run natively) answers 64 concurrent requests, all 200, p95 far below 2 s, exposes its counters on `/metrics` and drains on SIGTERM. No image was
pulled, built or run and `docker compose up` never ran.** The branch is stacked on `d10-distillation` (cut from it at `b672aa8` on 2026-11-09), is not merged and not
pushed, and no CI run covers it. It is 24 commits above `d10-distillation` at its code tip (one of design, eleven of tasks, nine of the fix wave, three of the final wave that answers the re-review), and two more with the records (the design commit is `17549f8`; the tasks are `05d460e`,
`2fab952`, `48c7125`, `3b94c3f`, `0e7f0e0`, `81ca7e0`, `6ed0e72`, `89ceb57`, `6de792f`, `e98a44c` and `b6d57a4`; the fix wave is `814b57d`, `966387a`, `266dbea`, `765c63c`, `a291106`,
`f464cc0`, `784361d`, `38c490b` and `4bfaa5e`, `D11: review minors (serving)`; the final wave is `330250a`, `55629cd` and the code tip `039d1d7`; the records are the commits `D11: runbook, spec amendments and state and decision
records` and `D11: record the re-review`). It is machinery, not a model: the only data it ever ran on is the templated toy sentences of `tests/support/tiny_parallel.py` (the toy saturates at chrF 98.2 and
ranks nothing), no real corpus or model exists, **no GPU was used** (the owner did not authorise it; every Eole process, the converter and every CTranslate2 translator ran
with `CUDA_VISIBLE_DEVICES=""` and `device="cpu"`, `auto` is refused everywhere, `--device cuda` is validated and never run), and `configs/serve/server.yaml` and `edge.yaml`
are tables of unmeasured priors. Built: `rinchen.export.{record,verified,quantize,ct2}` (the record `rinchen.json`, schema `rinchen.model/1`, replacing D10's `export.json`; the
reader `verify_export` moved out of `train`; quantisation chosen at conversion; `convert` with a private staging directory and the manifest last), `rinchen.infer.{source_tokens,
script_mask,engine_types,engine,translator}` (the engine with the guards CTranslate2 lacks: the tag by placement, the end token, the source limit, the reserved pieces, script
masking in three modes, documents through D9's chunker; the class `rinchen.Translator`, resolved lazily), `rinchen.train.export_parity` (strict Eole/CTranslate2 parity),
`rinchen.serve.{schemas,metrics,batcher,engine,_http,_translate,app}` and `rinchen.config.serve` (FastAPI, a micro-batcher, eleven Prometheus families, a stop with one
budget), the commands `rinchen export convert|verify|parity`, `rinchen translate` and `rinchen serve`, `docker/{Dockerfile.cpu,compose.serve.yaml}`, `.dockerignore`,
`configs/serve/{tiny,server,edge}.yaml`, the hatch `force-include` that ships the language registry in the wheel, the test helpers listed in `CLAUDE.md`, and
`docs/runbooks/11_serving.md`. The design is `docs/superpowers/specs/2026-11-09-d11-export-serving-design.md` (amended to the finished code, section 7 listing each of A1 to A23);
its rulings T1 to T22 are condensed in `DECISIONS.md` ("2026-11-09: D11 export, quantisation, serving: the rulings"), made by the builders and the controller without a review round
of their own. Built task by task with tests first after a spike on the real Eole, CTranslate2, FastAPI and Docker (Tasks 1 to 11, then ONE combined review by two reviewers, a fix wave,
a scoped re-review and these records). **The combined review (two independent reviewers, read-only: Tasks 1 to 7 and Tasks 8 to 11) found: ready with fixes, 0 Critical and 7
Important findings.** R1: a document-level export returned `no_output` for every sentence not decoded inside an aligned window; a literal reserved symbol in a request text became a
control piece (`<2eng_Latn> This is the star.` silently changed the language); `torch` had been dropped from the tool versions. R2: a stop could take twice the shutdown timeout
(SIGKILL under the compose grace period); a valid `n_best` was refused with a misleading message when the configured beam was not 4; a mistyped `RINCHEN_SERVE_*` variable was
silently ignored; readiness and the drain are not observable over HTTP (uvicorn starts the lifespan before it binds and ends it after it closes the listener). The fix wave (nine
commits) fixed them all and left open on purpose the limits that `docs/runbooks/11_serving.md` lists under "Known limits". **The scoped re-review of the fix wave (an independent reviewer, read-only, `b6d57a4..4bfaa5e`) found that it needed one small fix: 0 Critical, 1 Important and 8 Minor findings.** The Important one: the fix of the document path was incomplete, because the suppression list of a lone sentence of a document model still held `<sep>` in the script mode `always` and in the second pass of `retry`, so the model ran on past the sentence and invented text with the status `ok` (the gate's document model, 400 plain rows, beam 1: `always` chrF 81.80 with 358 of 400 texts different from `off`, `retry` 88.42 with invented trailing sentences in its 5 replacements; after the fix `off` 88.23, `retry` 88.49, `always` 88.49 with 5 texts different). Its probes on real servers and exports also measured the stop again (a fake engine stuck 6 s on a real uvicorn ended T + about 0.2 s after the signal: T=1 1.17 s, T=2 2.17 to 2.22 s, T=3 3.18 s) and found the rest sound. The final wave (`330250a`, `55629cd`, `039d1d7`) fixed the Important finding and five of the minors (a hypothesis that starts with `<sep>` is counted; `shutdown_timeout_seconds` has a floor of 1; `rinchen translate` names a refused text by one number counted from 1; the long prose lines; the stop-budget margin) and left three as documented limits (Kubernetes service links of a Service named `rinchen-serve` collide with the refused `RINCHEN_SERVE_` prefix; the requests in flight get half of the stop budget; an unknown client key is echoed up to 40 characters in the 422). The records written while it ran said it was running; this and the matching edits of the runbook, the spec (A24, A25), the plan, `docs/VERIFY.md`, `DECISIONS.md` and `CLAUDE.md` close that. Gates, run for real on Linux, CPython 3.14.10:
`ruff format --check .`, `ruff check .` and `mypy src` clean and `pytest -m "not network" -q` -- 10258 passed after Task 1, 10260 after Task 2, 10410 after Task 3, 10650 after Task 4,
10851 after Task 5, 11004 after Task 6 (four parallel processes), 11049 after Task 7, 11316 after Task 8, 11457 after Task 9, 11534 after Task 10, 11562 after Task 11, 11768 passed, 1 skipped, 10 deselected after the fix wave (`4bfaa5e`, 24 min 1 s, one sequential process) and **11778 passed, 1 skipped, 10 deselected** after the final wave (`039d1d7`, 26 min 51 s, one sequential process), against 9750 at the end of D10; the D5 to D10 regression sets stayed green. Mutation checks by
hand, one mutant at a time, as recorded per task: 92, 61, 62, 159, 106, 132, about 146, 133 and 196 for Tasks 1 to 6 and 8 to 10, the gates' mutants for Tasks 7 and 11, and 4, 9, 11, 30, 10, 8, 14 and 23 for
the fix wave and 5, 14 and 1 for the final wave.
Evidence of the STOP gate, through the real `rinchen` binary on the CPU with the GPU hidden (2 threads), measured by the gates and again for the runbook on 2026-11-10 (Intel Core Ultra 7
265K, 20 cores, 46 GiB; the machine was loaded by other work, so the times are longer than idle; every count and score below was the same in the spike, in the gates and in the runbook's
run): the tiny teachers (2,000 steps source tag, 1,000 steps target tag; averaged weights SHA-256 `f4e32704b958...`, the same as D10's); `export convert` (4.8 s; `model.bin` 2,034,311 bytes at
float32, 1,488,691 at int8 (73.2 %), 844,931 at int8_float16 (41.5 %); `rinchen.json` 1,121 bytes); strict parity **200 of 200, mean |delta| 7.19e-7 per token (max 3.43e-6)** for the
source-tag teacher and **200 of 200, 1.37e-6** for the target-tag teacher, against the gate's 1e-3, **and with the forced tag's log-probability kept 200 of 200 exact but 0.0433 per
token: FAIL** (only the log-probabilities reveal the trap); the int8 export 199 of 200 exact at 1.14e-3 (reported, not gated: the 1e-3 bound is a float32 bound); chrF over the 400 dev rows 98.207
(float32), 98.268 (int8, +0.061), 98.197 (int8_float16); effective compute types `float32`, `int8_float32`, `int8_float32`; script masking over 500 requests the toy never trained: 96 outputs
with a forbidden letter (731 letters) with the mask off, 0 with `always`, `retry` decodes the 96 again and all 96 improve, the 400 trained requests leak nothing, and the D5 converter wrote the
traditional script in 448 of 448 words without the list and 0 with it; a document model (2,500 continuation steps) over 66 test documents: 400 `ok`, chrF 84.747, 101 of 106 windows aligned,
94 of 106 falling back at a 20-piece cap; `rinchen serve` ready 0.80 s after the spawn, 64 x 1 and 64 x 8 concurrent requests all 200 (p95 57.7 and 115.1 ms; the gate's 70.2 and 165.2 ms
against the real server, 28.7 and 87.5 ms in process), `/metrics` counting exactly the 136 requests and 584 sentences, SIGTERM with 24 requests inside the service answered by 1.57 s and the
process gone at 1.68 s with status -15; the stop budget: a fake engine stuck 8 s with a 2 s timeout ends the process 2.17 s after the signal (4.18 s before the fix). The thresholds (exact
share 0.95, delta 1e-3 for float32, int8 chrF drop 1.0, 0 forbidden letters, all 200, p95 below 2.0 s) were not lowered. **The toy ranks nothing**: do not read the gate as evidence that int8 is
lossless, that masking helps a real model or that the thread settings suit one.
Findings worth knowing: upstream facts about CTranslate2 4.8.4, Eole 0.6.4, uvicorn 0.54.0 and the packaging, in `docs/VERIFY.md` ("D11 findings"). `compute_type="auto"` quantises every export to
int8 on a CPU (refused); Eole's score needs `length_penalty: none` and excludes a forced target prefix that `score_batch` includes; `score_batch` appends the end token; `suppress_sequences` entries
must be vocabulary entries and never the end token; a document model continued on windows of 2 to 4 sentences writes `<sep>` after a lone sentence (the engine cuts at the first `<sep>`); uvicorn
exits by the re-raised signal (143), starts the lifespan before it binds the port and ends it after it closes the listener (readiness and the drain are a refused connection on the wire, and the
health check's start period must be sized for the model); the built wheel lacked `configs/languages.yaml` (fixed by a `force-include`); a frozen `uv sync` carries torch and the CUDA wheels (22
packages in this lock, about 4.4 GB) that serving never imports.
Open gates (none can be done without real data, a GPU or the authority to build an image): everything GPU-side (CTranslate2 on CUDA, `float16`, `int8_float16`, token batching, the speed of a
1.1 B-parameter teacher and a 190 M student, the thread settings of a real model); the real quantisation loss and parity margin; whether a hard script mask helps a real model; the container (build,
size, start time, `docker compose up`, the stop of PID 1, an in-container health check, the base and uv tags, the start period for a real model); document throughput and a real document model; the D3
gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D10 gates stay open.
Known and open: the limits listed in the runbook (a line of `rinchen translate` is read whole with no cap; `rinchen serve` hashes the export twice; an engine that fails to open exits with uvicorn's
status 3 after a traceback; no body read timeout and no cancellation of a disconnected client's work; no cap on a client's `beam_size`; uvicorn's plain-text errors cannot be shaped; a blank converter
path from the environment is accepted; the image is amd64 only and its build needs the network; a flake of a D8 test under load, passing alone); `infer/engine.py` has 769 lines and
`infer/engine_types.py` 594 (the limit is 900); three limits of the re-review (a Kubernetes Service named `rinchen-serve` injects variables the prefix refusal rejects, so set `enableServiceLinks: false`; the requests in flight get `ceil(T/2)` of the stop budget, so a legitimate 15 s request is cut at T=20; an unknown client key is echoed up to 40 characters in the 422). NOT verified: any run on a GPU or a real model, the container, `KeyboardInterrupt` with a real CTranslate2 call, Windows and macOS, and CI. [EVIDENCE: the gate chain, the integration modules (112 passed in 604 s: parity 13, surfaces 25, serving gate 13, process 15, stop budget 2, D10 gate 44, D10's gate included), the read-only drivers of the runbook (`scratchpad/d11-t12/`), the
spike (`spike-findings.md`) and the review reports, 2026-11-09 to 2026-11-10, commits `17549f8` through `039d1d7`.]

**D12 (robustness) is done on branch `d12-robustness`; its STOP gate ("robustness block populated in tiny report") is met on the toy: through the real `rinchen eval robustness` binary and
the real Eole, CTranslate2 and SentencePiece on the CPU with the GPU hidden, a tiny model trained for the purpose (6,000 steps, 187 s, from random initialisation) is scored over a generated suite
and `report.json` holds all six values at or above the gate's floors (numbers 892 of 960 = 0.9292 against 0.90, placeholders 599 of 600 = 0.9983 against 0.99, entities 447 of 480 mentions =
0.9313 against 0.90, terms 1044 of 1044, script purity 1.0 over 900 rows, `mn_Mong` source retention 0.9976 against 0.95), while the negative controls score far below (hypotheses moved by 7 rows:
numbers 3 of 960, placeholders 2 of 600; the D11 teacher, which never saw a digit or a placeholder: 0 of 960 and 0 of 600, retention 0.3172; a glossary the model never writes 0 of 1,044;
placeholder indices 10 and up 0 of 28; script mask `off` leaves 101 of 900 rows under the purity floor).** The branch is stacked on `d11-export-serving` (cut from it at `23dc962` on 2026-11-10), is
not merged and not pushed, and no CI run covers it. It is 19 commits above `d11-export-serving` at its code tip `e92687d` (one of design, eight of tasks, five of the fix wave, three of the final wave after the re-review), and two with the records
(the design commit is `37634d1`; the tasks are `a6ecb70`, `60ba4d6`, `93d23b9`, `396e85d`, `3e65275`, `f8087f1`, `4f8cba7` and `b2466ef`; the fix wave is `64e45f0`, `40dc2ee`, `ce794ee`, `338bb02` and
`ae4e816`, `D12: review minors`; the final wave is `d1b25a9` (a currency symbol read as the unit of the wrong number), `e389c2b` (the surface-exact detail sees the layout of a number) and the code
tip `e92687d`, `D12: re-review minors`; the records are the commits `D12: runbook, spec amendments and state and decision records` and `D12: record the re-review`). It is an INSTRUMENT, not a mechanism and not a verdict on any model: D12
MEASURES numbers, placeholders, names, glossary terms, the script of the output and a source in the traditional script, and builds no masking, no terminology engine, no `terminology` request field and no
training row that teaches either (this amends D11's T22; nobody owns the mechanisms or their training data after D12); the only data it ever ran on is the templated toy of `tests/support/robust_toy.py`
(it ranks nothing), no real corpus, glossary or model exists, **no GPU was used** (the owner did not authorise it; every Eole process, the converter and every CTranslate2 translator ran with
`CUDA_VISIBLE_DEVICES=""` and `device="cpu"`, `auto` is refused, `--device cuda` is validated and never run), and the conventions table, the 4-letter case suffix, the currency symbols and the 0.98 floor
are priors. Built: `rinchen.text.numerals` (D4's numeral folding moved out of `filter/numbers.py`, D4 unchanged), `rinchen.eval.robust_{numbers,placeholders,spans,script,mong}` (six metrics as pure
functions), `robust_suite{,_rules}` (the strict suite file `rinchen.robustness.suite/1`), `robust_hypotheses` (the hypotheses JSONL file and the runner: ONE `Engine.translate` call), `robust_outcomes`,
`robust_sections`, `robustness` (`robustness.json`, `rinchen.robustness/1`, one outcome per row, no timestamp; `robustness_block` of SIX keys: the master plan's four plus `entity_acc` and
`mong_source_retention`, amendment C1; `robustness_coverage`), `robust_report` (`report.json`, `rinchen.report.robustness/1`, with a `coverage` map), the command `rinchen eval robustness` (`cli/eval_robustness{,_plan,_output}.py`;
`--model` or `--hypotheses`, the converter pair, hash before read, the manifest last, exit 0 whenever it measured), the golden files in `tests/fixtures/robustness/`, the test helpers listed in `CLAUDE.md`
(the toy, its suites and the gate stages) and `docs/runbooks/12_robustness.md`. The design is `docs/superpowers/specs/2026-11-10-d12-robustness-design.md` (amended to the finished code, section 7 listing
A1 to A24); its rulings T1 to T20 and C1 are condensed in `DECISIONS.md` ("2026-11-10: D12 robustness: the rulings"), made by the design agent, the builders and the controller without a review round of
their own. Built task by task with tests first after a spike on the real Eole and CTranslate2 (Tasks 1 to 8, then ONE combined review by one reviewer, a fix wave, a scoped re-review and these records).
**The combined review (one independent reviewer, read-only, Tasks 1 to 8, `37634d1..b2466ef`) found: ready with fixes, 0 Critical, 4 Important and 9 Minor findings.** I-1: the numbers metric failed
identical text when one value appeared twice with different units (`5 and 5` plus a tugrik sign against itself); I-2: the denominators of the values were invisible (a value over 10 scored rows of 1,000, silence
raising the script mean, a retention over two near-zero chrF scores); I-3: `term_acc` is natural agreement with a glossary, not adherence (the D11 teacher scores 1,041 of 1,044 because the glossary is the toy's noun
table); I-4: the per-row outcomes could not reproduce the retention. The fix wave (five commits, `b2466ef..ae4e816`) fixed I-1, I-2 and I-4, the D8 race test (a process that exits while `/proc/<pid>/stat` is
read raises `ProcessLookupError`, which the helper did not catch; the ONE edit to a D8 test) and the stale figures, left I-3 to documentation and parked the rest as limits (the hyphen over-acceptance, the
converter provenance gap, the trivial-solution scores, no held-out `reorder` row, the floor margins). **The scoped re-review of the fix wave (`b2466ef..ae4e816`, one reader): approved with minors, no Critical and no Important finding.** Its probes (650,000 random cases of the numbers matching against an
independent reference, 0 mismatches; the `coverage` of 400 random sub-suites; the real r3 run through the real command unchanged: 892 of 960 and the other five headlines) found one minor in the code: M-1, the
currency-symbol rule was wider than the vector it pinned (`$5 $7` read as `5 $` and `7`, so its faithful reorder `$7 $5` failed the units). The final wave fixed it with a touch-based rule (`d1b25a9`, A22), made the
surface-exact detail see the layout of a date (`e389c2b`, A23: a copy-through echo is surface-exact on 568 of 960 again, not 720; only the detail and the golden `surface_exact_rows` 4 to 3 changed) and fixed the
minors (`e92687d`: a string detail only for `chrf_signature`, the golden digest comment, `1 row`, one sentence of `--help` that broke in two lines, the docstring of the narrow report); the rest is wording and limits (A24):
the per-row `value_cyrl` supports a ratio-of-mean-sentence-chrF PROXY of the retention, not the corpus statistic (golden 0.5099 against 0.5018; real 0.9987 against 0.9976); `coverage` sits outside
`robustness_block`, so D13 must embed `robustness_coverage` beside it; the real run and the golden numbers rows are unchanged. Gates, run for real on Linux, CPython 3.14.10: `ruff format --check .`, `ruff check .` and `mypy src` clean and `pytest -m "not
network" -q` -- 11983 passed after Task 1, 12155 after Task 2, 12265 after Task 3, 12967 after Task 4, 13217 after Task 5, 13457 after Task 7, 13491 after Task 8 (22 to 29 min each, one sequential process; Task 6's run, 13311 passed,
had the one failure of the D8 race test) and **13543 passed, 1 skipped, 10 deselected** after the fix wave (`ae4e816`; the machine was at load 58 to 130, so the suite ran as five parallel pytest processes
that cover every collected test exactly once, about 19 minutes) and 13,583 passed, 1 skipped, 10 deselected (30 min 44 s, one sequential process, load average 6 to 20; 13,543 after the fix wave, so 40 new tests), `ruff format --check .` (598 files), `ruff check .` and `mypy src` (211 files) clean after the final wave (`e92687d`), against 11778 at the end of D11; the D5 to D11 regression sets stayed green. Mutation checks by hand, one mutant at a time, as recorded per
task: 63, 68, 57, 93, 122, 68 + 5, 93 and six gate mutants for Tasks 1 to 8, and 13, 18 and 12 for the fix wave and 9, 5 and 6 for the final wave (one equivalent: a plural for zero rows that no line prints); differential fuzzes against independent references (300,000 random cases of the numbers
metric against a brute-force unit-aware matching, 0 mismatches; for the final wave 200,000 pairs and 160,308 texts of the touch-based unit rule and 400,000 pairs of the surface detail, 0 mismatches).
Evidence of the STOP gate, measured by the gate and again for the runbook on 2026-11-10 (Intel Core Ultra 7 265K, 20 cores, 46 GiB, one thread per command; every count and score below was the same in the
spike and in the gate wherever the runbook's "Measured numbers" does not say otherwise): the toy's teacher trains in 186.9 s (190.9 s with the export and the suites) and its `model.bin` (2,080,071 bytes, SHA-256
`e28d1566...a6`) is byte-identical to the design spike's and to the gate's earlier retraining, so the gate's figures are exact, not statistical (the gate records the digest and does not assert it); the toy's
generator is pinned by the SHA-256 of its 29,352 rows (`4136d80a...bad1`); the main suite (3,580 rows, 4,060 requests) is scored in 2.10 s at beam 4 and 1.51 s at beam 1 (import and model load included); numbers
by category (beam 4): integer, decimal, percent, date and time 120 of 120, currency 117, grouped 118, two numbers in one sentence 57 of 120 (no floor); placeholders at beam 1 600 of 600; the converted arm
(D5's tiny converter) retains 0.5473 against 0.9976 for the direct arm; the traditional-script output suite is pure (1.0 over 100 rows); the golden files score to the golden `robustness.json` byte for byte;
two runs write equal files. Costs: the metrics 0.04 s per 1,000 requests, the engine 0.11 s (beam 1) and 0.23 s (beam 4) per 1,000 rows. The thresholds (floors and ceilings of the controls) were not lowered.
**The toy ranks nothing**: do not read the gate as evidence about a real model, a real glossary, the traditional script as people write it or a GPU.
Findings worth knowing: facts in `docs/VERIFY.md` ("D12 findings"): the vocabulary of a model is the file of tokens its training data showed, so a model trained on a digit-free corpus cannot read or write a
digit and the engine still says `ok` with purity 1.0; a model copies a placeholder only for the indices it was trained on (0 to 9 copied, 10 and up 0 of 28; the D11 teacher 0 of 600); the engine refuses the
inline terminology markers; the toy's training is byte-for-byte repeatable on this machine; the numbers metric needs a unit-aware matching (D12-NUMBERS-UNIT-MATCHING) and every value needs its coverage
(D12-RETENTION-DENOMINATOR); a source echo scores 960 of 960 on the numbers and 600 of 600 on the placeholders (read these two next to chrF); beam 4 drops a placeholder that beam 1 keeps (D15's
"`placeholder_acc` 1.0" depends on the decode).
Open gates (none can be done without real data, a GPU or a mechanism): everything GPU-side (CTranslate2 on CUDA, `float16`, `int8_float16` and their effect on these metrics); a MECHANISM for placeholders and
terminology and the training data that would make a model use them (nobody owns them); real data and real priors (the conventions table, the suffix, the currency symbols, the 0.98 floor; real entities have many
spellings; number and magnitude words are not read); suites from real parallel text and real glossaries (D13); a real converter and real traditional-script text; D15's own tiny suite and matching training rows
(D12's generator is a test helper); the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D11 gates stay open.
Known and open: the limits listed in the runbook (`term_acc` is agreement, not adherence; the hyphen and prefix over-acceptance of names; the converter's files are hashed but loaded without re-verification and `ScriptConverter`
has no `close`; `read_suite` and `read_hypotheses` are used only by tests and the caps bound the parse, not memory; trivial solutions score high; no held-out `reorder` row; a symbol with a space on both sides is the unit of the number before it (`5 $ 7`); retention above 1 and a PROXY from the per-row chrF; script silence is
excluded by design; `coverage` outside the block; the floors have margin and bit-identical training is not asserted; the sacrebleu signature pinned in the golden file fails on a bump on purpose;
the disclosed process slips of the wave). NOT verified: any run on a GPU, a real model, corpus or glossary, a real converter, the exit codes 4, 1 and 130 with real failures, memory at large suites, CI, Windows and
macOS, and a review of the final wave by another reader. [EVIDENCE: the gate chain, the two integration modules (31 tests: 442 s under load, 374.7 s on a quiet machine), the read-only drivers of the runbook
(`scratchpad/d12-t9/`), the spike (`spike-findings.md`) and the review report, 2026-11-10, commits `37634d1` through `e92687d`.]

**D13 (final evaluation and release tooling) is done on branch `d13-release`; its STOP gate ("tiny report.md + model card generated; dry-run hub upload (mock)") is met on the toy: through the real `rinchen` binary and the real Eole, CTranslate2, SentencePiece and sacrebleu on the CPU with the GPU hidden, no network and no token, the toy export of D12 and the D11 teacher translate a FLORES-like fixture suite built from the toy; `rinchen eval run`, `eval report`, `release stage` and `release hub` (the dry run) write the results, the report (JSON and Markdown), the model card and the upload plan; the paired bootstrap is byte-deterministic and its controls hold (identical systems: delta 0, interval [0, 0], p exactly 1.0; a 5 % perturbation: delta -3.2 to -4.0, p 0.001998), and the stage and the plan refuse a stray file, a changed byte, a symlink, a token-like line and an absolute path.** The branch is stacked on `d12-robustness` (cut from it at `1a378d0` on 2026-11-10), is not merged and not pushed, and no CI run covers it. It is 27 commits above `d12-robustness` at its code tip `413bc94` (one of design, twelve of tasks (eleven tasks, Task 8 in two commits), one of controller import-order fixes, six of the fix wave, seven of the final wave) and two with these records (the design commit is `6d6ad38`; the tasks are `4eee5cb`, `3f33b11`, `1829610`, `3b10d39`, `194c2fa`, `9f7f1fc`, `79e58bc`, `ce8c37f` with `a5ad95e` and `bef542b`, `fe8d592`, `074567e` and `3c52162`; the fix wave is `0000ccb`, `1a95422`, `06b1157`, `3d32b74`, `7ac4404` and `2df73c1`; the final wave is `0a4f8d4`, `e59a740`, `069b232`, `71f3b8b`, `8d84671`, `4bfeb80` and `413bc94`; the records are the commits `D13: runbook, spec amendments and state and decision records` and `D13: record the re-review and the final wave`). It is an INSTRUMENT and a release tool, not a verdict on any model: D13 MEASURES (no threshold, exit 0 whenever it measured), builds no mechanism and no scorer, and the only data it ever ran on is the toy of `tests/support/robust_toy.py` and the golden suite (they rank nothing); no real corpus, FLORES+, NTREX, human set, NLLB, MADLAD, Hy-MT2, COMET or MetricX checkpoint exists here, **no GPU was used** (the owner did not authorise it; every process ran with `CUDA_VISIBLE_DEVICES=""` and `device="cpu"`, `auto` is refused, `--device cuda` is validated and never run), **no network was used** (the real `fetch` ran only with `datasets.load_dataset` replaced by a fake, `rinchen eval suite fetch` and `rinchen release hub --upload` were never run, no Hugging Face endpoint was contacted and no token was read). Built: `rinchen.metrics` (BLEU, chrF and chrF++ from per-segment integer statistics, equal to sacrebleu's `corpus_score` with `==`, the allowed tokenizers `13a`, `zh`, `char`, `intl`, `none` and the six that download or need MeCab refused by name), `rinchen.eval.bootstrap` and `significance` (our own splitmix64 counter stream with integer sums, seed 20,261,110, 1,000 resamples, percentile positions 25 and 974, the paired comparison with ONE set of draws and an add-one p-value, the verdict by p and never by the interval), the suite file and its builders from LOCAL files (`suite_rules`, `suites`, `suite_sources`; `khk_Mong` refused by name; the devtest guard), the systems and the runner (a Rinchen export through the engine, D6's generators for baselines, a hypotheses file; every direction has a state and a model is never asked for a pair it was not trained for), the outputs file, scoring, `result.json` and `run.json` with strict readers, the neural seam (per-segment score files bound to the suite and to the run's hypotheses; no scorer), D12's robustness block embedded WITH its coverage and a verified binding, the targets as measurements, the report (`report.json` and `report.md`, recomputed from the suite and the outputs, byte-deterministic), the model card rendered by our own function (never the Hub's template), the staging directory and release record, the Hub upload plan with a dry run by default and a `HubClient` protocol (`HfHubClient` the only importer of `huggingface_hub`), `.github/workflows/release.yml` (tags only, never uploads), the commands `rinchen eval suite fetch|build|inspect`, `eval run`, `eval report`, `release stage` and `release hub`, the golden files in `tests/fixtures/eval/` (37,096 bytes), the test helpers listed in `CLAUDE.md` and `docs/runbooks/13_release.md`. The design is `docs/superpowers/specs/2026-11-10-d13-release-design.md` (amended to the finished code, section 7 listing A1 to A32); its rulings T1 to T23 and the amendments C2 to C4 are condensed in `DECISIONS.md` ("2026-11-11: D13 release: the rulings, the review and the limits"), made by the design agent and the controller without a review round of their own. Built task by task with tests first after a spike (Tasks 1 to 11, then ONE combined review by one reviewer, a fix wave, a scoped re-review and these records).
**The combined review (one independent reviewer, read-only, Tasks 1 to 11, `1a378d0..3c52162`) found: ready with fixes, 0 Critical, 5 Important and 13 Minor findings.** It re-derived the intervals, the paired delta and the p-value independently (bit-equal, segment and document unit), found no leak and probed the real binary about 50 times. I1: the report believed `ci95`, `statuses`, the three signatures and `trained` of a result file (a re-digested forgery exited 0); I2: the card's evidence banner, `toy` tag and `release.json` `evidence` came from the whole report and not from the carded system; I3: output failures the system flagged were invisible in `report.md` and the card; I4: the dataset, the revision and `pinned` were not printed; I5: private-by-default did not hold for an existing repository (`create_repo(exist_ok=True)` ignores `private`). The fix wave (six commits, `3c52162..2df73c1`) fixed all five and eleven of the Minor ones (the devtest guard bypass, the stage not checking `report.md`, the lost fetch provenance, `--robustness` refused after the whole run, the GPU sentence, the `significant` column read `no` for a worse system, the robustness suite's kind, `--name .`, the Hub scan's blind spots, an upload before the record could be written, `comparison_metrics` without `primary_metric`) and left open on purpose the limits that `docs/runbooks/13_release.md` lists under "Known limits" (dead code used only by tests, a Rinchen export accepted as the baseline, two source files just over 450 lines, a test file of 1,042 lines, transitive imports invisible to the scans). **The scoped re-review of the fix wave (`3c52162..2df73c1`, one independent read-only reviewer) approved it with minors: 0 Critical, 0 Important and 7 Minor findings; all 16 findings of the combined review were verified fixed.** Its own 50 mutants were killed but one equivalent one, its independent cell checkers found 0 mismatches and its differential of the Hub scan 0 in 3,000 cases. The Minor findings: m1 a re-digested result could still plant what the report prints (a neural metric's value, a free-text reason, D12's block values), m2 the dataset and the revision were printed unescaped (a re-digested `report.json` injected a heading into the card), m3 the inverted devtest scan did not see the eval commands `cli/eval*` or an aliased dynamic import, m4 two refusals did not say what to do, m5 `--public` over an existing private repository said nothing, m6 the card printed `n/a ()` for a suite with runs and no results, m7 the card and `report.md` named internal milestones. **The final wave (seven commits, `2df73c1..413bc94`, failing test first, 72 mutants all killed, two of them only after a test was added) fixed all seven** (the neural presence of a result is checked, the report reader holds the dataset and revision to the suite reader's patterns, the inverted scan sees `cli/eval*` and aliased dynamic imports with three more allowlist entries, the two refusals name the command to run again, `release hub --upload` prints `visibility:`, no empty reason is printed, the uploaded text says `the robustness block` and `the COMET and MetricX scorers are not built` in place of milestone names) and left open the free-text reason and D12's block values (the report cannot recompute them), `Suite.selection_view` without a caller and `report_runs.py` at 474 lines; the full gate after it: **17,852 passed, 1 skipped, 10 deselected** (25 min 44 s wall). A run directory made before the final wave carries the old neural reason and the report refuses it: run `eval run` again. **No reader has read the final wave.** Gates, run for real on Linux, CPython 3.14.10: `ruff format --check .`, `ruff check --no-cache .` and `mypy src` clean and `pytest -m "not network" -q` -- 13,900 passed after Task 1, 14,104 after Task 2, 14,551 after Task 3, 14,931 after Task 4, 15,163 after Task 5, 15,739 after Task 6, 16,018 after Task 7, 16,412 after Task 8, 16,866 after Task 9, 17,286 after Task 10, 17,390 after Task 11 and 17,666 passed, 1 skipped, 10 deselected after the fix wave (`2df73c1`; three unit parts and the integration part, 24 min 45 s for the last, about 25 minutes wall) and **17,852 passed, 1 skipped, 10 deselected** after the final wave (`413bc94`; three unit parts and the integration part, 25 min 44 s wall), against 13,583 at the end of D12 (4,269 new tests, 105 new test files at the end of the fix wave and four in the final wave); `ruff check --no-cache` found five import-order errors that ruff's cache had hidden in every earlier gate (fixed in `bef542b`; every gate since uses `--no-cache`). Mutation checks by hand, one mutant at a time: 126, 62, 129, 87, 93, 83, 71, 115, 82, 171 and 45 for Tasks 1 to 11 152 for the fix wave and 72 for the final wave (1,288 in all), with the few equivalent or redundant survivors named in the runbook; differential checks against independent references (30,000 random suite corruptions, 7,000 source cases, 50,000 result mutations, 252 random corpora against sacrebleu's public `corpus_score`, 400 `eval run` cases, 200 report cases with 616 comparisons, 400 card cases, 1,500 random Hub stages twice), 0 mismatches.
Evidence of the STOP gate, measured by the gate on 2026-11-11 (Intel Core Ultra 7 265K, 20 cores, 46 GiB, one thread per command) and again for the runbook through the real binary over the golden fixtures: the toy export (`model.bin` 2,080,071 bytes, SHA-256 `e28d1566...a6`, byte-identical to D12's) scores chrF++ 97.03, 99.05, 98.97 and 98.38 over 480 segments in `eng-khk`, `khk-eng`, `khk-rus` and `rus-khk` (BLEU 95.84, 98.12, 98.15, 96.83; the widest 95 % interval 2.63; floors 95.0, 93.0 and 5.0), the D11 teacher 36.50, 46.09, 29.30 and 31.67 (ceiling 60.0), the export minus the teacher +60.528, +52.965, +69.668 and +66.707 (p 0.001998, the smallest possible with 1,000 resamples), `eng_Latn-rus_Cyrl` and `rus_Cyrl-eng_Latn` `not_trained` with no result, the robustness block attached (numbers 892 of 960, placeholders 599 of 600) with its coverage, the targets `unmeasured` or `not_met` as they are, the card and the nine-file plan generated and deterministic, three reruns byte-equal across hash seeds and locale; the three gate modules take 423.9 s alone, 357.6 s of it the session fixtures. The golden report and card are 14,942 and 6,058 bytes (14,894 and 6,054 before the final wave reworded two sentences of the uploaded text) and `rinchen eval report` over the golden runs writes `report.json` and `report.md` byte-equal to the golden files. **The toy ranks nothing**: do not read the gate as evidence about a real model, a real benchmark, a baseline, a neural metric, the Hub or a GPU.
Findings worth knowing: facts in `docs/VERIFY.md` ("D13 findings"): sacrebleu's per-segment statistics are integers and a corpus score is the score of their sums, exactly; four of its tokenizers download a model on first use (the design spike's first run did: two models of about 9.7 MB were downloaded into `~/.sacrebleu` and removed, a disclosed slip); its paired bootstrap calls a system against itself significant (p 0.000999) and reads a seed from the environment; the engine answers an untrained pair with status `ok` in the wrong language; D6's `CT2Translator` is silent about a family whose language tokens are not in the vocabulary; the Hub client raises before any socket under `HF_HUB_OFFLINE=1`; `create_repo(exist_ok=True)` ignores `private` for an existing repository (source-read, UNMEASURED against the Hub); the export's `manifest.json` holds absolute local paths and is never uploaded; a document unit widens the interval 1.5 to 2.2 times; the verdict of a comparison must come from the p-value because intervals can touch zero while p is 0.108.
Open gates (none can be done without real data, a GPU, a checkpoint or Hub access): everything GPU-side (CTranslate2 on CUDA, `float16`, `int8_float16`, vLLM, the cost of a real 16-direction run); the real FLORES+ dataset (id, configurations, columns, splits, sizes, document ids), NTREX (whether it offers Khalkha), the Mongolia-sourced human sets and the real `fetch`; NLLB, MADLAD and Hy-MT2 as baselines (the vocabulary guard has only seen the toy export); COMET-22, MetricX-24 and MetricX-24-QE (B1 and B2: no scorer); the Japanese and Korean tokenizers (B3 and B4); benchmark decontamination; the off-target rate; zero-shot evaluation; a real upload, the Hub's acceptance of the card's front matter and language codes and the visibility of an existing repository; the release workflow on GitHub Actions; the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D12 gates stay open.
Known and open: the limits listed in the runbook (the segment bootstrap ignores sentence correlation and the document unit needs a document id for every row; no correction for multiple comparisons; `char` BLEU for Japanese and Korean is not comparable with published BLEU and chrF++ is degenerate for Chinese and Japanese; no contamination check; the neural values are believed as the result states them; BLEU's last digits depend on the C library; the devtest guard does not follow transitive imports; `totals_of`, `metric_with_tokenizer`, `require_metric` and `Suite.selection_view` are used only by tests; `cli/eval_suite.py` has 481 lines and `cli/eval_run.py` 451 and `tests/unit/test_eval_error_messages_d13.py` 1,042; a Rinchen export is accepted as the baseline; D12's robustness block does not record its suite's kind; the gate's docstrings quote sizes measured before the fix wave; the disclosed process slips of the waves; a re-digested result can still plant a free-text reason and D12's robustness block values, which the report cannot recompute). NOT verified: any run on a GPU, a real model, corpus, benchmark or baseline, the real fetch and upload, the Hub, GitHub Actions, the exit codes 4, 1 and 130 with real failures, memory at large suites, Windows and macOS, CI, and a review of the final wave. [EVIDENCE: the gate chain, the three gate modules (99 tests), the read-only drivers of the runbook (`scratchpad/d13-t12/`), the spike (`spike-findings.md`) and the review report, 2026-11-10 to 2026-11-11, commits `6d6ad38` through `413bc94`.]

**D14 (maintenance loop) is done on branch `d14-maintenance`; its STOP gate ("nightly workflow green on tiny") is met ONLY as "the workflow's own command, read from `nightly.yml`, runs green natively on the CPU through a uv shim that never forwards the upgrade dry run": through the real `rinchen` binary and the real Eole, CTranslate2 and pytest on the CPU with the GPU hidden, the command `uv run --frozen --no-sync rinchen maintain check --out-dir maintain-out --cadence "$CADENCE"` (`CADENCE=nightly`, run through `sh` in a temporary directory with a `uv` shim first on `PATH`) exits 0 and writes a report, a Markdown summary and a manifest that hold the doctor, the lock, the environment, the dependency drift table (from a HAND-WRITTEN sample) and the parity reruns on the toy, byte-equal on a rerun and free of paths, timestamps and the environment, with every control (a failing, shrunken or overrunning rerun, a bad uv output, a held package, the refusals, the interrupts) failing as it should; GitHub Actions is UNMEASURED.** The branch is stacked on `d13-release` (cut from it at `0f3f6ba` on 2026-11-11), is not merged and not pushed, and no CI run covers it. It is 25 commits above `d13-release` at its code tip `52b9b3b` (one of design `5477430`, seven of tasks `875cabf`, `4cc9576`, `44745ef`, `704370c`, `b30effd`, `93f06fb` and `51c4489`, the controller's fix of the workflow `8d701b0`, one with the first records `cc7b9bd` (`D14: runbook, spec amendments and state and decision records`), the 14 commits of the fix wave `83d09f5` to `ada92a1` and `52b9b3b`) and one with the records of the review (`D14: record the review, the fix wave and the re-review`). It is a CHECK and a schedule, not a verdict on any stack or model: D14 reports and never changes anything (nothing writes `uv.lock` or `pyproject.toml`, nothing upgrades, nothing asks a device question), the dry run is the one network step of a night and is uv's own, the reruns run on the toy of D11 to D13 (which ranks nothing), **no GPU was used** (the one on this machine was not authorised; the doctor's GPU check is never called) and **no network was used by any test or measurement** (the gate answers the dry run from a sample through a shim; one mutation run of Task 7 ran the real online dry run once in a copy of the tree by a slip, disclosed below). Built: `rinchen maintain check` and `rinchen.maintain` (the lock reader with `tomllib`, the environment check against the lock, the parser of a dry run that sorts every line into exactly one class and fails on what it does not know, the drift table with a cross-check against the lock and the pin rules as data, the command runner with a process-group kill, the parity reruns decided by exit code AND JUnit counters, the report and its Markdown, the order of the steps with every seam injected), the closed configuration `configs/maintain/maintain.yaml` (10 pin rules and 6 reruns, every number a prior), `.github/workflows/nightly.yml` (a schedule every night and on Sunday, `workflow_dispatch` with a cadence choice, `contents: read`, 150 minutes, no secret, no upload), the report `maintain_report.json` (schema `rinchen.maintain/1`), its summary and a manifest that records every resolved package version and the lock's SHA-256, the goldens in `tests/fixtures/maintain/` (6,132 and 1,806 bytes), the shim of the gate and the test helpers listed in `CLAUDE.md` and `docs/runbooks/14_maintenance.md`. The design is `docs/superpowers/specs/2026-11-11-d14-maintenance-design.md` (amended to the finished code and to the review, section 7, A1 to A24); its rulings M1 to M19 are condensed in `DECISIONS.md` ("2026-11-11: D14 maintenance loop: the rulings and the limits"), made by the design agent and the controller without a review round of their own, and corrected by its entry "2026-11-11: D14 review, fix wave and re-review". Built task by task with tests first after a spike (Tasks 1 to 7, then the first records, a combined review, a fix wave, a scoped re-review and these records).
**The combined review (one independent read-only reviewer, Tasks 1 to 7, `0f3f6ba..51c4489`) found: ready with fixes, 0 Critical, 0 Important and 11 Minor findings.** It reproduced each defect on a copy of `51c4489` and found the six steps, the closed reasons, the exit codes (130, 143 and 129 by real signals), the parser, the reruns, the report, the manifest, the workflow and the gate's safety sound. The findings: the dry-run log was judged from its first 64 KiB; a configured doctor skip could silence a failing check; the self-selection guard looked at names only; the nightly manifest said `<sha>-dirty`; `--out-dir` could be any project subdirectory; `--rerun X` gave the other reruns a false reason and `--no-reruns` with `--rerun` was accepted; children inherited tokens; the in-process doctor ran with the caller's device visible; `Step` did not enforce a closed reason; a docstring said no program output is in a message; four small ones. **The fix wave (14 commits, `83d09f5` to `ada92a1`, code and tests only, tests first) fixed all eleven and two minors deferred from Task 3** (the dry-run log is read whole or a new closed reason `the output of uv is longer than <limit> bytes`; only `D-TORCH-CUDA` can be skipped and a skip never covers a failure; the guard folds the spelling and refuses `tests/unit`, `tests/integration` and `tests/property`; the code version is read before anything is written and `/maintain-out/` is ignored; an existing `--out-dir` may hold only the command's own names; the reason `not selected with --rerun` and `--no-reruns` with `--rerun` refused; children drop names ending in `_TOKEN`, `_SECRET`, `_PASSWORD`, `_API_KEY`; the command hides every GPU from its own process; uv's `Resolved` line takes `<n>ms`, `<n>.<dd>s` and `<n>m <dd>s`; a bad JUnit report is `the JUnit report could not be read`); the full gate after it, the implementer's in six parallel parts at a load average of about 200 to 460 from another project: **20,443 passed, 1 skipped, 10 deselected** (19,818 at Task 7: 625 new tests in 12 new unit test modules), `ruff format --check` (845 files), `ruff check --no-cache` and `mypy src` (287 files) clean, 119 mutants with none surviving (one test of D5 that was not touched failed once under that load, 201.7 s against its 150 s bound, and passed alone). **The scoped re-review of the fix wave (one independent read-only reviewer, `cc7b9bd..ada92a1`) approved it with minors: 0 Critical, 0 Important, 3 Minor and 3 Nits;** all eleven findings were verified fixed (60 mutants of its own killed, every failing path driven through the real binary through a shim or with `--no-drift`, none of 15 edited existing test files weakened). Its minors: a `*.part` file left by a killed run locks the output directory and the message advised the wrong thing; the doctor's own `--help` children inherit the whole environment; the closed directory list was compared on the raw spelling. The controller fixed the first and the third in `52b9b3b` (7 new tests; 2,392 maintenance, workflow and property tests green; **the full gate was not run again after it, and no independent reader has read it**) and left the second and the nits open on purpose (the suffix rule is the limit: `AWS_SECRET_ACCESS_KEY` is kept; the controls module is at 899 of 900 lines; one docstring line of 123 characters), with four findings the controller chose not to fix (the shared public signal helper against the private duplicate in `utils.logged_process`; an `installed()` seam that raises gives exit 1; the unlisted partial logs after an interrupt; the order of the messages about options): all in the runbook's "Known limits". Gates, run for real on Linux, CPython 3.14.10 (the figures up to Task 7; the fix wave's are above): `ruff format --check .` (833 files), `ruff check --no-cache .` and `mypy src` (287 files) clean and `pytest -m "not network" -q` -- 18,424 passed after Task 1, 18,774 after Task 2, 18,972 after Task 3, 19,152 after Task 4, 19,465 after Task 5, 19,576 after Task 6 and **19,818 passed, 1 skipped, 10 deselected** after Task 7 (`51c4489`; five parallel parts that cover every collected test exactly once, about 2,400 s wall for the last), against 17,852 at the end of D13 (1,966 new tests, 29 new unit test files, 4 property files and the two gate modules). Mutation checks by hand, one mutant at a time: 70, 75, 116, 160, 73, 134 and 46 for Tasks 1 to 7 and one for the controller's fix (675 in all), one equivalent survivor, and 119 for the fix wave and 60 for the re-review (none surviving); differential checks against independent references (the lock reader, the environment check, the dry-run parser on 40,000 outputs, the JUnit reader on 100,000 documents, the runner on 200 children, the rerun decision on 60,000 points, the report's states, the Markdown on 3,000 reports, the whole command on 1,000 scenarios, the version parser on 40,000 strings, the shim on 238,741 argument lists), 0 mismatches.
Evidence of the STOP gate, measured by Task 7 on 2026-11-11 (Intel Core Ultra 7 265K, 20 cores, 46 GiB, one thread per command, uv 0.13.7): the command of the workflow ran twice through the shim, exit 0, 163.6 s on a first look and about 150 s inside the gate (doctor 13.2 s, lock steps about 0.1 s, the two nightly reruns 136.2 s); the lock 187 entries, 186 names, format version 1 revision 5; the environment 186 installed, none differing, none extra; the drift table of the sample 7 entries (5 updates, 1 add, 1 remove; 3 drift, 4 blocked); numerics 1,110 passed (floor 1,100), `parity_tiny` 13 passed; the manifest 188 `tool_versions` keys and 8 outputs; the report 6,138 bytes and its summary 1,806; the two documents byte-equal under `PYTHONHASHSEED=3`; the shim's log 4 calls a night, none refused; `uv.lock` (`559b3954...`) and `pyproject.toml` with the same bytes and modification times; the parity of the toy export through `--rerun parity_export` 200 of 200 exact, mean abs delta 7.19e-07 (max 3.43e-06) against 1e-3, and one flipped byte of `model.bin` or the checkpoint of another model exits 4 with everything written. The real `uv lock --check --offline` forwarded through the shim and was not denied. **The toy ranks nothing**: do not read the gate as evidence about a real stack, a real export, GitHub Actions or a GPU.
Findings worth knowing: facts in `docs/VERIFY.md` ("D14 findings"): uv's offline dry run prints `Using CPython <path> interpreter at: ...`, `Resolved 187 packages in 1ms` and `No lockfile changes detected` when nothing changes and FAILS offline for any change here (the cache holds one platform's resolution), so the lines for an update, an add and a remove were never measured (a second-hand report from a slip: `Resolved 179 packages in 1.11s`, then `Update`/`Remove` lines, 51 entries, no `Add`, not a fixture); the dry run is blind to exact pins (12 of the 38 direct dependencies, among them Eole, CTranslate2, sacrebleu, SentencePiece and torch), so the drift table is "what uv could take", never "what exists"; `uv run` sets `UV` to the uv that started it, and a plain `uv run` locks and syncs first, so the workflow uses `uv run --frozen --no-sync`; pytest exits 0, 1, 4 and 5 and writes `tests`, `failures`, `errors`, `skipped` (a deselected test is not counted), and its report holds an absolute path and a host name that no report may copy; the doctor's GPU check imports torch and asks CUDA, so the check never calls it; `uv.lock` is TOML (187 entries, 186 names, torch `2.14.1+cu130`, a version that is a string); the nightly job keeps no log, so a first real `logs/drift.log` comes from a local online run; uv's elapsed time is read as exactly three spellings, `<n>ms`, `<n>.<dd>s` and `<n>m <dd>s` (from uv's formatter as the reviewer and the fixer remember it: UNMEASURED beyond `1ms`; `1.11s` is second-hand, the others hand-written), and any other spelling is one unrecognised line, a red night with a count; the review found that a dry-run log longer than 64 KiB was judged from its cut (now a failure), that a configured doctor skip silenced a failure (now refused), that a nightly manifest said `<sha>-dirty` because the output directory was inside the checkout (the code version is read first and `/maintain-out/` is ignored), that children inherited tokens and that the in-process doctor saw the caller's device (both closed).
Open gates (none can be done without a runner, the network, a GPU or a real export): everything GPU-side (the doctor's GPU check is never run; a CUDA wheel upgrade cannot be qualified); the real PyPI state and the lines of a real change (the first real `logs/drift.log` must become a fixture); GitHub Actions (that the workflow starts and finishes inside its timeout, that `github.event.schedule` holds the cron that fired, that `hashFiles` and `env.CADENCE` work in a condition, the runner's speed and the minutes budget, about 560 a month on assumptions that are themselves unmeasured); every real export (`parity_export` ran only on the D11 toy export); the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D13 gates stay open.
Known and open: the limits listed in the runbook (the blind spot of the dry run; the format of a real change is learned, not known; the floors, bounds and rules are priors from one machine; the floors' margins are 10 for `numerics` and 0 for the other four; the toy's determinism is one machine's; the nightly job keeps nothing; `lock --check` online is unmeasured; a rerun reports counters only; a SIGKILL of the command leaves its children until their bounds; `tests/unit/test_maintain_check.py` has 856 lines and `test_maintain_gate_controls.py` 899; the doctor's own children inherit the environment; the secret rule is an upper-case suffix rule; the self-selection guard is by name; `torch-*` over-labels `torch-optimi`; `docs/STACK.md` stays a hand snapshot; the disclosed process slips of the build: the first guard test of a mutation run started a real `rinchen maintain check` without a shim, which ran the real online dry run, a lock check and the reruns once in a copy of the tree for about 165 s and, in a second mutation run, the command for a few seconds with `CUDA_VISIBLE_DEVICES=0` (nothing of the repository was written; the gate now sets `UV_OFFLINE=1` everywhere and the guard test runs only `--help`)). NOT verified: any run on a GPU, a real model, corpus or export, GitHub Actions, the real PyPI, the exit code 1 with a real full disk, Windows and macOS, CI, and a reader for `52b9b3b`. [EVIDENCE: the gate chain, the two gate modules (197 tests) and the 43 tests of the shim, the spike (`spike-findings.md`), the read-only checks of the runbook (`rinchen maintain check --help`, pure functions in process), the review reports and the ledger, 2026-11-11, commits `5477430` through `52b9b3b`.]

**D15 (the tiny end-to-end pipeline) is done on branch `d15-tiny-pipeline`; its STOP gate ("CI runs `just tiny` green on 3.14") is met ONLY as "the body of the `tiny` recipe, read from the `justfile` and run through `sh` in a temporary directory with a uv shim first on `PATH`, runs green natively on the CPU": through the real `rinchen` binary and the real Eole, CTranslate2, SentencePiece and sacrebleu on the CPU with the GPU hidden, no network and no token, `rinchen tiny run --profile tiny` takes seven committed fixture files through fifteen stages and writes `report.json`, `report.md` and `manifest.json`, all EIGHT thresholds pass, every control (400 steps, a damaged export or fixture, placeholder indices never trained, a wrong script, a budget of 5 s, the three signals, the refusals) fails as designed and a rerun in another directory under another hash seed, locale and time zone writes equal `report.json` bytes; GitHub Actions is UNMEASURED.** The branch is stacked on `d14-maintenance` (cut from it at `412bf84` on 2026-11-11), is not merged and not pushed, and no CI run covers it. It is 11 commits above `d14-maintenance` at the end of its build (code tip `5dab820`: one of design `971d9fd`, eight of tasks `f0972af`, `0283135`, `cf3de50`, `0141070`, `8ed5445`, `f07b513`, `16403f1` and `83188ca`, one fix found by the first real run `3862635` and the gate `5dab820`), one with the first records (`8386aed`, `D15: runbook, spec amendments and state and decision records`), the 16 commits of the fix wave of the review (`40d6f00` to `08848c1`, all `D15: fix ...`; **code tip `08848c1`**) and one with the records of the review (`D15: records of the combined review and of its fix waves`). It is MACHINERY, not a verdict on any model: the fixtures are templated sentences and the model memorises a grammar of 37 nouns (every number is a toy number and ranks nothing), the stand-ins of `--qe-fake` rank nothing, **no GPU was used** (the one on this machine was not authorised; every process ran with `CUDA_VISIBLE_DEVICES=""` and `device="cpu"`, `auto` is refused, `--device cuda` is never run), **no network was used** (`rinchen data fetch`, `rinchen eval suite fetch` and `rinchen release hub --upload` were never run; the fetch stage copies local files; the one socket is the loopback listener of the synthetic-data stage), no real corpus, generator, LID, LaBSE, QE scorer or neural metric scorer exists, and `just accept` stays a stub. Built: `rinchen tiny run` (`cli/tiny{,_inputs,_output}.py`) and the library `rinchen.tiny` (27 modules: `states`, `context`, `fixtures`, `corpus`, `fakes`, `data_stages`, `commands`, `command_readers`, `command_writers`, `step_runner`, `model_stages`, `model_stages_rest`, `suites`, `suite_matching`, `serve_stage`, `eval_stages`, `eval_readers`, `evaluate_facts`, `thresholds`, `report`, `report_blocks`, `report_markdown`, `report_schema`, `report_shapes`, `pipeline`, `dry_run`, `graceful_runner`), the closed configuration `rinchen.config.tiny` with `configs/tiny/tiny.yaml`, the Eole base config `configs/eole/tiny.yaml` and the local registry `configs/data/sources.tiny_parallel.yaml`, the seven fixture files `tests/fixtures/parallel/*.tsv` (67,104 bytes, each pinned by SHA-256) with their generator `scripts/build_parallel_fixtures.py`, the recordings `tests/fixtures/tiny_run/` (13 files) and the goldens `tests/fixtures/tiny/`, one write-only module in D13's eval package (`eval/fixture_suites.py`), the `tiny` recipe of the `justfile`, the `tiny` job of `ci.yml`, the renamed nightly step, `/artifacts/` in `.gitignore`, a gate of four integration modules, the test helpers listed in `CLAUDE.md` and `docs/runbooks/15_tiny_pipeline.md`. The design is `docs/superpowers/specs/2026-11-11-d15-tiny-pipeline-design.md` (amended to the finished code and to the review, section 7 listing A1 to A40); its rulings P1 to P20 are condensed in `DECISIONS.md` ("2026-11-12: D15 tiny pipeline: the rulings, the build and the limits"), made by the design agent and the controller without a review round of their own, and amended by its entry "2026-11-12: D15 review, fix wave and re-review". Built task by task with tests first after a spike (Tasks 1 to 9, then the first records, a combined review by two readers, a fix wave in three parts, a scoped re-review and these records); the amendments of section 7 of the spec now run A1 to A40.
**The combined review (two independent read-only readers, `412bf84..8386aed`: A the control plane, B the stages and the data) found: both "ready with fixes", 0 Critical, 2 Important and 12 Minor findings.** The Important ones: A-I1 `report.md` printed a FAILING value as one meeting its bound, with two decimals (`train_chrf` 49.996 as `50.00 | fail`, `placeholder_acc` 237 of 238 as `1.00 | fail`, and the known `parity_delta` 1.62e-06 as `0.00`); B-I1 the `synth` stage was `ok` when the fake server answered nothing (nothing read the synth counts). The Minor ones: A-M1 a stage could be skipped for any closed reason and `stages` still passed; A-M2 an `ok` stage silently dropped parts it did not list; A-M3 the strict reader accepted a `wall_time` with a value, a skip with no failure before it and an `evaluate` with 1 of 4 parts; A-M4 the spawn-time interrupt control tolerated orphans for 10 s; A-M5 the report-defect message said nothing of the run was written; A-M6 any `--config` could lower the floors; A-M7 the gate never checked that the `uv` it finds is the shim; A-M8 the report drops a stage's whole detail silently when one value fails the shape; B-M1 the stand-in encoder rejects the authentic row `mm-0031` (a test had been loosened to "at most one"); B-M2 the converter's five lines are read from a log that also holds stderr and its accuracy is never judged; B-M3 the registry check compared link targets, not the names the copies get; B-M4 a generator comment and the spec path `01_fetch/files/`.
**The fix wave (16 commits `8386aed..08848c1`, all `D15: fix ...`, code and tests only, a failing test first, hand-made mutants killed by the new tests, no document touched) fixed both Important findings and nine of the twelve Minor ones (A-M1 to A-M7, B-M3 and the comment of B-M4), corrected the false statements of B-M1 and deferred A-M8, B-M2 and the cause of B-M1, in three parts** (W1: an `ok` stage lists every part, the `stages` threshold counts a skip no failure explains, the reader refuses what the writer cannot write, the loader refuses a lowered floor, one figure rule beside every bound; W2: the closed check `synth_output`, the registry names, the leak check read at once with a grace of at most 1.5 s, the shim check, the synth assertions of R1, the wording, the screen figures at six digits; the scoped re-review's W3 is below). **The scoped re-review (one independent read-only reader, `8386aed..94bf095`) approved the first two parts with minors: 0 Critical, 0 Important, 3 Minor** (M1 an `ok` stage could skip a part for any closed reason, M2 the synth check never compared `sources` with the 64 rows fed, M3 the `stages` detail said no failure came before a skip that one did precede); **W3 (`66548c7`, `843dbe9`, `08848c1`) fixed all three** (`states.OWN_SKIP_REASONS`, four entries; the words of the detail; `counts.sources` against the rows of the monolingual fixture that was fed). Full gates (the implementers', six parallel parts, at most three pytest processes at once): **25,061 passed, 1 skipped, 10 deselected** after W2 (29 min 46 s) and **25,243 passed, 1 skipped, 10 deselected** after W3 (25,254 collected with the deselected; about 27 min; `ruff format --check` 972 files, `ruff check --no-cache` and `mypy src` 319 files clean), against 24,759 passed at `8386aed` (ARITHMETIC: 484 more) and 20,451 run at the end of D14 (ARITHMETIC: 25,244 run, so 4,793 new tests); mutants by hand: W1 76 in the five fixes and 5 extra (one survivor that only the unit tests kill), W2 about 67 (the survivor of the registry fix is equivalent), W3 29 (all killed). The real pipeline passed every change inside those full gates: the documents of its real runs were `report.json` 26,545 bytes, `report.md` 3,218 (the golden is 3,217; 3,228 before the figure rule), `manifest.json` 11,521 to 11,575, the real synthetic block `counts.selected` 64 of `sources` 64, `rows` 64, `chat_requests` 64 and `bt_rows` 60, every threshold `pass`, `report.md` printing `parity_delta` as `1.622e-06`.
**Left open on purpose** (the runbook's "Known limits" 4 to 6 and 14 to 22): A-M8; the broad `except` of `report_blocks` (a null block beside an `ok` stage while the verdict can pass); B-M2; the cause of B-M1 (the encoder cuts words at white space and `. , ? ! : ;` only, so the Mongolian full stop and comma U+1803 and U+1802 stay glued to the word before them and `mm-0031` is rejected: cosine 0.6332 against 0.7, 1.0000 without U+1803 (measured here), 759 of 760 authentic rows pass; a fix would move the kept rows, `train.parquet`, the Mongolian coverage, the digests, the goldens and the real recordings); no gate control shows `wall_time` itself failing; a lowered but self-consistent bound in a `report.json` still parses (only the loader knows the floors); the `has_noted_part` branch of the Markdown rows is reachable only through a damaged report; the 1.5 s grace of the leak check is UNMEASURED beyond a load average of 15; `_copy_name` repeats an expression of `fetch_source` and a link to the fixtures directory itself passes the registry check; the two halves of the skip rule live in different modules; and **one flake, UNEXPLAINED**: D14's control `tests/integration/test_maintain_gate_controls.py::test_an_interrupted_run_leaves_no_manifest_and_no_process_and_takes_the_old_run_away[sigterm]` hit its 60 s `finish()` timeout once in W3's first run of that part (a real pipeline ran in another part at a load average of about 15; it passed alone, 170 passed, and in W2's gate; it imports nothing of `rinchen.tiny` and no D15 change touches it; load-sensitive, cause UNMEASURED). **The `GracefulRunner` decision stays OPEN and is the owner's** (D15's subclass as built, or the one-line fix at the root in D14's `SubprocessRunner`, which changes D14's behaviour), as is merging and pushing the stacked branches. **No independent reader has read the three commits of W3**; the controller read their diffs and ran 2,092 tests over 18 modules.
**Process slips** (as the readers and implementers reported them; DECISIONS.md has the entry): reviewer B started ONE real `synth run` child (a stage command, not the pipeline: GPU hidden, offline switches set, the loopback fake server); both readers used the system `python3` without `-I` for read-only inspections and reviewer A's probes may have left ignored `__pycache__` bytes; the scoped reader used the venv's Python without `-I` in a COPY of the tree and replaced text there; W1 typed a foreground `sleep 1`, ran the system `python3 --version` once, used `sed -i` on a scratch commit-message file and a heredoc into scratch, went to 934 lines in one test module (fixed forward in `e95189a`) and read only part of STATE.md; W2 ran one no-op heredoc, made one Write call with wrong parameters (rejected), launched its last part with a basetemp removal in the same background call and read the pytest signal dispositions for the first wave only; W3 wrote one vacuous assertion (replaced before any run), counted an invalid-syntax mutant as a kill once (redone), ran one stray no-op shell assignment and reran one part by hand after its flake; the controller launched one verification script as `nohup ... &` inside the background mode of the tool. No `git stash`, no `pkill -f`, no GPU question, no docker command and no network request is recorded; every long process ran with `CUDA_VISIBLE_DEVICES=""`.
Gates, run for real on Linux, CPython 3.14.10: `ruff format --check .` (966 files), `ruff check --no-cache .` and `mypy src` (319 files) clean and `pytest -m "not network" -q` -- 20,611 passed after Task 1, 21,908 after Task 2, 22,050 after Task 3, 22,792 after Task 4, 23,257 after Task 5, 23,870 after Task 6, 24,139 after Task 7, 24,340 after Task 8 and **24,758 passed, 1 skipped, 10 deselected** after Task 9 (`5dab820`; in six parallel parts that cover every collected test exactly once, about 37 minutes wall at a load average of 14 to 29), against 20,450 passed at the start (ARITHMETIC: 20,443 at the end of D14's fix wave plus the 7 tests of its last commit; 24,759 collected against 20,451, so 4,308 new tests); the records of Task 10 changed no test and the full gate after them gave 24,759 passed, 1 skipped, 10 deselected (one more case of the hygiene gate's content scan, for the new runbook; six parts, about 46 minutes wall, at most three pytest processes at once); the fix waves of the review then gave 25,061 and, at the code tip `08848c1`, **25,243 passed, 1 skipped, 10 deselected** (above). Mutation checks by hand, one mutant at a time, as recorded per task: 116, 348, 71 and 72 (two rounds), 122, 149, 94, 88 and 72 for Tasks 1 to 8, and 18 gate-level and 30 unit-level mutants for Task 9; differentials against independent references (the reader against `csv`, the thresholds against a table reader, the stand-ins against a numpy model, 1,020 builder calls against the options of the real commands, 20,000 generated sentences against a brute-force matcher, 1,000 random reports, 500 whole-command scenarios), 0 mismatches.
Evidence of the STOP gate, measured by the gate of Task 9 on 2026-11-12 on a 20-core CPU shared with other tenants (a TIME moves by a third, a count, a score or a digest does not): the recipe ran in **113.1 s** alone (121.0 s beside another pipeline; the spike's prototype 120 to 158 s, 120.0 s with every child under `taskset -c 0-3`; ARITHMETIC for a 4-core runner at D14's factor of 3: 370 s, UNMEASURED); the manifest's seconds environment 0.05, fetch 0.00, corpus 0.03, filter 0.43, spm 1.83, converter 41.24, synth 1.05, prepare 0.97, train 44.16, average 0.92, export 6.37, parity 3.52, serve 0.06, robustness 1.28, evaluate 10.27; the thresholds `stages` 0, `train_chrf` 99.3585 (the least of five directions, `mn_Mong-eng_Latn`; against 50.0 at 1,500 steps), `parity_exact` 1.0 (70 of 70), `parity_delta` 1.6221680949143363e-06 (against 1e-3), `int8_chrf_drop` 0.0 (against 1.0), `script_purity` 1.0, `placeholder_acc` 1.0 (238 of 238), `wall_time` pass; the filter 770 rows, 756 kept, the 10 planted rows each with exactly its reason, 4 of 760 authentic rows flagged, `train.parquet` 1,316 rows; `report.json` 26,545 bytes, `report.md` 3,228 bytes (3,218 in the full gates of the fix waves, after the figure rule), `manifest.json` about 11.5 KB (13 inputs, 187 `tool_versions` keys, 7 outputs); under `PYTHONHASHSEED=3`, `LC_ALL=C` and `TZ=Asia/Ulaanbaatar` in another directory the two documents are byte-equal and the digests of the averaged weights (`4e06ce45faef7c4b...`), the float32 and int8 `model.bin` (`f48811a45287337c...`, `8fa90a266950cfc3...`) and the converter's `model.bin` (`e70a4ccacf85c060...`) are equal while the SentencePiece models differ (`66f3f736...`, `58af3a54...`: the file embeds its paths). Controls: 400 steps exit 4 after 102.2 s with `train_chrf` 16.83, `placeholder_acc` 0.0420 and `script_purity` still 1.0; placeholder indices 10 and up 0.0; copied answers `script_purity` 0.5; hypotheses rotated by 7 rows chrF 21.09 against 99.96; `--budget-seconds 5` exit 4 after 5.4 s with the `converter` stage `fail` for `timed out after 1 s`, the later stages skipped and `wall_time` `pass` (each command gets what is left of the budget, so the stage that times out carries the failure); SIGTERM, SIGINT and SIGHUP end the command with 143, 130 and 129 within 0.5 to 1.0 s, nothing left alive, and the same directory then runs to the end in 148 s; a flipped byte of a fixture exits 3 in under 5 s and one of an export exits 2. No floor was lowered; `train.steps` is 1,500 and was not changed. **The toy ranks nothing**: do not read the gate as evidence about a real model, a real corpus, GitHub Actions or a GPU.
Findings worth knowing: facts in `docs/VERIFY.md` ("D15 findings"). 400 steps, the master plan's figure, misses its own bound of 50 on this fixture (16.83) and the rule is "raise the steps, never lower the bound" (1,500 steps); the "1k" SentencePiece vocabulary is not reachable on templated text (the ceiling was 786 to 850 and moves with the text; 640 is used); filtering both directions of a pair flags the reversed rows as near-duplicates (185 of 1,450), so the filter runs once per pair and is expanded afterwards; greedy decoding is what makes the robustness block exact (beam 4 loses a placeholder in one export of nine); the int8 export fails the exact-share rule on held-out lines (64 of 68) and is equal to float32 on the memorised suites, so the gated parity is float32's; the SentencePiece file embeds its paths and the `created` stamp is the clock unless given one; a child process cannot reach a mock transport, so the real `synth run` needs the loopback listener, which `no_network` does not guard; the engine refuses a traditional-script output without a converter even with the mask off; **D14's runner kills the process group of an interrupted command with SIGKILL at once, which left `eole train` running after an interrupted `rinchen tiny run`** (found by the gate's R11, fixed in D15's own `GracefulRunner`: SIGTERM, 2.0 s, then SIGKILL); and **the real `report.md` printed `parity_delta` as `0.00` beside the bound `<= 0.001` for a measured 1.62e-06, a symptom of a wider defect (the two-decimal rule also printed a failing value as one that meets its bound): FIXED by the first fix wave with one rule for both tables and the screen line, `figure_beside_bound` (the real `report.md` now prints `1.622e-06`)**; the review added: a fake chat server that answers nothing still lets `synth run` exit 0 (the stage now judges its counts), and a profile could lower a floor (the loader now refuses it). D15 corrected statements of its own design (among them: the fixtures total 67,104 bytes, not 68,056; the stand-in encoder separates one replaced concept only, maximum cosine 0.446, not up to three; there are EIGHT thresholds, not seven; section 7 of the spec lists the 31 amendments of the build and nine more for the review, A1 to A40) and added one module, `eval/fixture_suites.py`, to D13's package because the devtest guard forbids the suite readers to the pipeline.
The definition of done (`just accept` stays a stub; the eight items of the master plan's `<mission>` are checked by hand, here and in `docs/runbooks/15_tiny_pipeline.md`):

| `<mission>` item | Met by | State |
|---|---|---|
| 1. Every module of the layout exists, typed, tested; `pytest -m "not gpu and not net"` green | D0 to D15 and their gates (the project's marker is `network`): 24,758 passed, 1 skipped, 10 deselected after Task 9 and 25,243 passed, 1 skipped, 10 deselected at the code tip `08848c1` | met locally; no CI run since D2 |
| 2. Coverage of `src/rinchen` at least 85 % | nothing: `ci.yml` uploads `coverage.xml`, no number was ever recorded | UNMEASURED |
| 3. `ruff check`, `ruff format --check`, `mypy` green; `ty check` runs | `just gate`; `ty` was never run | gate met locally; `mypy --strict` and `ty` UNMEASURED |
| 4. CI green on GitHub Actions with 3.14 | `ci.yml` (D0 and D2 green on 2026-11-02) and the new `tiny` job | UNMEASURED for D3 to D15 |
| 5. `just tiny` at most 20 minutes on a 4-core CPU with `report.json` meeting the thresholds | this milestone: 113.1 s here, 370 s ARITHMETIC at D14's factor | met here (20 cores); the 4-core time UNMEASURED |
| 6. A runbook per operator stage | `docs/runbooks/00` to `15` | met |
| 7. Scale-up is configuration only | no branch on a profile name, a closed profile schema, a control profile, the mapping table of the runbook | met as stated; a full run UNMEASURED |
| 8. `rinchen doctor` passes in CI, GPU checks documented | `just doctor` in `ci.yml`; D14's CPU list | CI run UNMEASURED since D2 |

Open gates (none can be done without real data, a GPU, a runner or a reader): everything GPU-side (the GPU-side Eole, `float16` and `int8_float16`, the speed and memory of a real model); GitHub Actions (the `tiny` job inside its 45 minutes, the runner's speed and core count, loopback sockets there, `hashFiles` in a step condition, the artifact upload and the minutes budget); the 4-core time; a real corpus, a real generator behind the synthetic stage, GlotLID, LaBSE and QE scorers and neural metric scorers (the stand-ins rank nothing and the seam of D13 ran with planted scores); cross-machine determinism (byte-equal on one machine at 2 threads); a fixture rich enough for the master plan's "1k" vocabulary; stage resume and a partial-run option (not built); `just accept`; the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D14 gates stay open.
Known and open: the limits listed in the runbook ("Known limits": three inputs hashed for the manifest only and an unhashed registry; `config_sha256` is the digest of the pipeline file and the threads are recorded nowhere; `wall_time` passes when the elapsed time equals the budget and on a budget cut, and no gate control shows it failing; `report._holdable` drops a stage's whole detail silently when one value fails the shape; a malformed fact becomes a null block through a broad `except` while the verdict can pass; `GracefulRunner` duplicates about 20 lines of D14's runner and imports two private names of D14, a SIGKILL of `tiny run` still leaves its children, whether D14's reruns start a child in their own session is UNMEASURED and the decision between the subclass and the root fix is the owner's; the converter stage reads its five lines from a log that also holds stderr and never judges the converter's accuracy; the stand-in encoder rejects `mm-0031` (759 of 760 authentic rows pass; the cause is left); a lowered but self-consistent bound still parses; the `has_noted_part` branch is reachable only through a damaged report; the 1.5 s grace of the leak check is UNMEASURED beyond a load average of 15; `_copy_name` repeats an expression of `fetch_source`; `OWN_SKIP_REASONS` has four entries and the two halves of the skip rule live in two modules; one UNEXPLAINED flake of a D14 control under load; `spm check`'s coverage lines are not read; the pin of 186 packages is a literal in four tests; the step count is stated in two files; a top-level `logs/` exists only after a raising stage; the display of `parity_delta` is FIXED); no file over 450 lines in `src` or over 900 in the tests (the largest 424 and 895 at `08848c1`). NOT verified: any run on a GPU or on GitHub Actions, a real corpus, model, generator or scorer, a sandbox that forbids loopback sockets, exit code 1 with a real full disk, the 4-core time, another machine's digests, a full profile, Windows and macOS, and an independent reader for the three commits of W3 (`94bf095..08848c1`). [EVIDENCE: the gate chain, the four gate modules (224 tests at Task 9, 232 at `08848c1`), the spike (`spike-findings.md`), the read-only `rinchen tiny run --help` and `--dry-run` of the runbook (repeated at `08848c1`: the 39 lines were identical), a pure probe of the stand-in encoder over the fixtures, the three full gates of the fix waves and the ledger, 2026-11-11 and 2026-11-12, commits `971d9fd` through `08848c1`.]

Branch status (updated 2026-11-12, after the merge of D6 to D15 into `main` and the push; the paragraph below describes the branches as they were built): `main` holds D3 to D15 since the fast-forward of 2026-11-12 (see the end of this paragraph); until then `main` held D3, the D3 research follow-up, D4
and D5, and was at `0d534a3` (the commit that records the D5 merge; `862e176` is the D5 merge tip,
`2346cea` was the D3 end, `b38eaae` the D4 merge tip and `4eac98f` the commit that recorded the D4 merge;
no `d3-research-followup`, `d4-filtering-mining` or `d5-script-converter` ref remains). D6 lives on its own
branch, `d6-synthetic-data`, cut from `main` at `0d534a3`; D7 lives on `d7-multilingual-ablations`, stacked
on it (cut from `d6-synthetic-data` at `9973319`); D8 lives on `d8-teacher-training`, stacked on D7 (cut from
`d7-multilingual-ablations` at `b1a8a47`); D9 lives on `d9-document-level`, stacked on D8 (cut from
`d8-teacher-training` at `993f32c`; its last code commit is `135e9f9`, `D9: re-review minors`, and the records are
`6746ac5` and the commit `D9: record the re-review`); D10 lives on `d10-distillation`, stacked on D9 (cut from
`d9-document-level` at `7d0c166`; its last code commit is `233ba33`, `D10: re-review minors`, and the records are the commits `D10: runbook, spec amendments and
state and decision records` and `D10: record the re-review`, `b672aa8`); D11 lives on `d11-export-serving`, stacked on D10 (cut from `d10-distillation` at `b672aa8`; its last code
commit is `039d1d7` (the final wave `330250a`, `55629cd`, `039d1d7` follows `4bfaa5e`, `D11: review minors (serving)`), and the records are the commits `D11: runbook, spec amendments and state and decision records` and `D11: record the re-review`); D12 lives
on `d12-robustness`, stacked on D11 (cut from `d11-export-serving` at `23dc962`; its last code commit is `e92687d`, `D12: re-review minors`, and the records are the commits `D12: runbook, spec amendments and state and decision records` and `D12: record the re-review`); D13 lives on `d13-release`, stacked on D12 (cut from `d12-robustness` at `1a378d0`; its last code commit is `413bc94`, `D13: fix the card printing an empty reason and milestone names in uploaded text`, and the records are the commits `D13: runbook, spec amendments and state and decision records` and `D13: record the re-review and the final wave`); D14 lives on `d14-maintenance`,
stacked on D13 (cut from `d13-release` at `0f3f6ba`; its last code commit is `52b9b3b`, `D14: re-review minors (folded directory list, the .part hint)`, and the records are the commits `D14: runbook, spec amendments and state and decision records` and `D14: record the review, the fix wave and the re-review`); D15 lives on `d15-tiny-pipeline`, stacked on D14 (cut from `d14-maintenance` at `412bf84`; its last code commit is `08848c1`, `D15: fix the synth check and the gate not comparing the sources with the fixture`, the end of the 16 commits of the fix wave of the review (`40d6f00` to `08848c1`) after the records `8386aed`; the end of the build was `5dab820`, `D15: the tiny gate`, after `3862635`, `D15: fix an interrupt that left the trainer running`; the records are the commits `D15: runbook, spec amendments and state and decision records` and `D15: records of the combined review and of its fix waves`). On 2026-11-12, at the owner's request ("merge and push everything to main branch"), `main` was fast-forwarded from `0d534a3` to `3260d1a`, the D15 tip (the tips of the nine earlier branches are ancestors of it, so there is no merge commit), and pushed to `origin/main` (`9e7b0ec`, which held nothing that `main` lacked; 305 commits above it, 82 of them already on `main` at the D5 merge): D6 to D15 are on `main`, the ten branch refs `d6-synthetic-data` to `d15-tiny-pipeline` stay local and were not pushed, and the "not merged and not pushed" of the milestone paragraphs of this file and of `CLAUDE.md` is the state at the time each was written. GitHub Actions had never run for D3 to D15: the first run is the one that push starts and no CI result is recorded here (UNMEASURED). The full local gate was last run
on the D15 branch: 25,243 passed, 1 skipped, 10 deselected (D6 to D15 included), on `08848c1` (the implementer's of the last fix wave, in six parallel parts; 25,061 on `94bf095`, the second fix wave's; 24,946 expected, ARITHMETIC, on `e95189a`, the first's, whose own parts gave 24,330 over everything but the real-pipeline modules; 24,759 on `8386aed`, the records of the build); 24,758 passed, 1 skipped, 10 deselected on `5dab820` (the implementer's, in six parallel parts; 24,340 on `83188ca`; 24,139 on `16403f1`; 23,870 on `f07b513`; 23,257 on `8ed5445`; 22,792 on `0141070`; 22,050 on `cf3de50`; 21,908 on `0283135`; 20,611 on `f0972af`); on the D14 branch it gave 20,443 passed, 1 skipped, 10 deselected (D6 to D14 included), after the fix wave `83d09f5` to `ada92a1` (the implementer's, in six parallel parts; not run again after `52b9b3b`); 19,818 on `51c4489` (19,576 on `93f06fb`, `D14: the nightly workflow`; 19,465 on `b30effd`; 19,152 on `704370c`; 18,972 on `44745ef`; 18,774 on `4cc9576`; 18,424 on `875cabf`); on the D13 branch it gave 17,852 passed, 1 skipped, 10 deselected (D6 to D13 included), on `413bc94` (17,666 on `2df73c1`; 17,390 on `D13: the evaluation and release gate`, `3c52162`; 17,286 on `074567e`; 13,900 after Task 1); on the D12 branch it gave 13,583 passed, 1 skipped, 10 deselected (D6 to D12 included), on `e92687d` (13543 on `ae4e816`, in five parallel parts; 13491 on `D12: the robustness gate`, `b2466ef`; 11983 after Task 1); on the D11 branch it gave
11778 passed, 1 skipped, 10 deselected (D6 to D11 included), on `039d1d7` (11768 on `D11: review minors (serving)`, `4bfaa5e`; 11562 on `D11: the serving gate`, `b6d57a4`; 10258 after Task 1);
on the D10 branch it gave 9750 passed, 1 skipped, 10 deselected, on `D10: re-review minors` (`233ba33`; 9706 on `D10: fix the vocabulary
padding that a teacher hypothesis could hold`, `cc917e2`; 9439 on `D10: the tiny student gate`, `e67b05e`; 8517 after Task 1); on the D9
branch it gave 8252 passed, 1 skipped, 10 deselected on `D9: re-review minors` (`135e9f9`;
8214 on `D9: review minors`, `c2d46b5`; 8075 after Task 5, 7187 at the end of D8); on the D8 branch it gave 7114
passed, 1 skipped, 10 deselected (D6, D7 and D8 included), on `D8: review minors of the
launcher` (`3e357ae`; 7020 after Task 5, 5968 on D7's last commit, `D7: re-review minors`) and again, with the final
review's fixes, on `D8: final review fixes` (`0378265`): 7187 passed, 1 skipped, 10 deselected; on D6's last
commit (`D6: re-review minors`) it gave 4331 passed; on the merged D5 tree `main` gave 2833 passed. The
whole-branch review of D8 ran (above); its fix wave's scoped re-review approved it with minors and the minor
that needed code is fixed (`D8: re-review minors`).

Next step: nothing of D15's own build or review is pending: the combined review (two independent readers, `412bf84..8386aed`), its fix wave in three parts (`8386aed..08848c1`) and a scoped re-review of the first two parts ran and are recorded above, and the limits they left open are in the runbook's "Known limits" (the null block of `report_blocks`, the `GracefulRunner` decision and the others listed above); only an independent reader for the three commits of W3 (`94bf095..08848c1`) was not asked for. D15 is the last milestone of the master plan (D0 to D15): `just tiny` is the real pipeline and the nightly job already runs it after the maintenance check (D15 decided that `just tiny` does not call `rinchen maintain check`; it calls `check_environment` in process). Merging and pushing the stacked branches (D6 to D15) was done on 2026-11-12 (see Branch status). The work that remains is the owner's: the `GracefulRunner` decision (D15's subclass as built, or the one-line fix at the root in D14's `SubprocessRunner`), reading the first run of `ci.yml` (started by that push) and of `nightly.yml` on GitHub (the `tiny` job's 45 minutes, loopback sockets, `hashFiles` and the minutes budget are UNMEASURED until then), the 4-core time, one real run of `rinchen maintain check` whose `logs/drift.log` becomes the first recorded fixture of a real change (D14's open question 1), and everything that needs real data, a GPU or a reader (the open gates above and the D3 to D14 gates). `just accept` stays a stub until the full definition of done is met (the table above).

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
[D15, 2026-11-12: this paragraph describes `ci.yml` as it was on 2026-11-02. Since D15 the stub step is gone from the job `gate` and `just tiny` is the real pipeline in a job of its own, `tiny` (`ubuntu-24.04`, 45 minutes, an upload of `report.json` and `report.md` only); neither ever ran on GitHub (UNMEASURED); see `docs/runbooks/15_tiny_pipeline.md`.]

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
