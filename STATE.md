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
both variants), no real corpus exists, no GPU was used, and `configs/eole/multilingual_proxy.yaml` and
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
(mean 98.65, minimum 98.24), BLEU 96.19 to 99.67, no direction tag in any hypothesis; the saved
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
of `step_200` SHA-256 `32a1c92e...` (the spike's value for an uninterrupted run); a second `launch` 0.73 s,
`already_complete`, the run directory unchanged; with the weights of `step_200` cut in half, `launch` resumed
from step 150 (9.60 s, `incomplete step directories skipped: [200]`, the log starting at `Step 160/  200`,
`launch.json` unchanged) and finished with other weights (`7e62604f...`: a resumed run is not the
uninterrupted one, because the data stream restarts at line 1 of every corpus), a separate run of 100 steps
extended to 200 resumed from step 100 (`Step 110/  200`, 13.50 s) and lowering `train_steps` or changing the
learning rate was refused (exit 2, nothing written); `average --last 3` (steps 100, 150, 200) 0.74 s, 58 of 62
tensors averaged; a real SIGTERM ended the launcher with 143 in 0.21 to 0.26 s with no Eole process left and
the next launch finished the run; a SIGKILL of the launcher left the Eole running, which kept the lock, so the
next launch was refused (exit 2) and the one after the orphan ended said `already_complete`. Those numbers
rank nothing: the model memorises a templated grammar. ARITHMETIC, not measurement (the formula reproduces the
real tiny checkpoint exactly, 820,458 tensor elements and 3,289,000 bytes): the teacher shape (24+12 layers,
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

Branch status (updated 2026-11-08, after the D8 build): `main` holds D3, the D3 research follow-up, D4
and D5, and is at `0d534a3` (the commit that records the D5 merge; `862e176` is the D5 merge tip,
`2346cea` was the D3 end, `b38eaae` the D4 merge tip and `4eac98f` the commit that recorded the D4 merge;
no `d3-research-followup`, `d4-filtering-mining` or `d5-script-converter` ref remains). D6 lives on its own
branch, `d6-synthetic-data`, cut from `main` at `0d534a3`; D7 lives on `d7-multilingual-ablations`, stacked
on it (cut from `d6-synthetic-data` at `9973319`); D8 lives on `d8-teacher-training`, stacked on D7 (cut from
`d7-multilingual-ablations` at `b1a8a47`). None of the three is merged into `main` and none is pushed;
merging D8 means merging D6 and D7 first. At the D5 merge `main` was 81 commits ahead of `origin/main`
(`9e7b0ec`); it is unpushed, and no CI run covers D3, D4, D5, D6, D7 or D8. The full local gate was last run
on the D8 branch: 7114 passed, 1 skipped, 10 deselected (D6, D7 and D8 included), on `D8: review minors of the
launcher` (`3e357ae`; 7020 after Task 5, 5968 on D7's last commit, `D7: re-review minors`) and again, with the final
review's fixes, on `D8: final review fixes` (`0378265`): 7187 passed, 1 skipped, 10 deselected; on D6's last
commit (`D6: re-review minors`) it gave 4331 passed; on the merged D5 tree `main` gave 2833 passed. The
whole-branch review of D8 ran (above); its fix wave's scoped re-review approved it with minors and the minor
that needed code is fixed (`D8: re-review minors`).

Next step: D9 (Document-level). Nothing in D8 blocks it. Its inputs from D8: `rinchen train launch` (the
resumable launcher: start, resume from the newest complete checkpoint, `already_complete`, the NaN gate, the
run lock, `command.sh` for the operator, the manifest of the final step) for continuing OUR teacher
(`train_from` is allowed only as the launcher's own resume, inside the run directory, and the run identity
excludes only `train_steps` and `train_from`, so D9's continued training on document-level data, which changes
the data section and the schedule, will need a CONSCIOUS extension of the identity rule and of the
`resumes` validation, decided and recorded rather than patched in), `rinchen train average` (a plain mean of the
last checkpoints; not a checkpoint to resume from), `rinchen train ablate --prepare-only` (the corpora and the
config of a variant from the mixture the ablations chose), `check_gpu_rules`, `curves.parse_train_log` and
`loss_gate`, and `configs/eole/teacher.yaml` (a prior). What D8 does not give D9: any trained teacher (no real
corpus and no GPU run exist), any measured batch, learning rate, throughput or memory, a live NaN abort, and any
evidence on a GPU; the open gates above and the D3, D5, D6 and D7 gates stay open.

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
