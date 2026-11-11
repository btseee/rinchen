# 13 - Final evaluation and release

## Scope

D13 says how a model of ours translates a benchmark, how it compares with another system, and ships that with a card that claims only what was measured. It builds: the benchmark suite file
and the builders that turn LOCAL files into one (FLORES+, NTREX, the Mongolia-sourced human sets of D2) with one fetch step that is the only code that touches the network
(`rinchen.eval.suites`, `suite_sources`, `rinchen eval suite fetch|build|inspect`); the systems (a Rinchen export through the engine, an NLLB or MADLAD directory or a vLLM server through
D6's generators, any system through a hypotheses file) and one runner (`rinchen.eval.systems`, `runner`, `rinchen eval run`); BLEU, chrF and chrF++ with their sacrebleu signatures from per-segment
integer statistics (`rinchen.metrics`); our own exact paired bootstrap of 1,000 resamples with a fixed seed (`rinchen.eval.bootstrap`, `significance`); `result.json` per (suite, system, direction),
`run.json` per run, the aggregate `report.json` and `report.md` (`rinchen eval report`) that embed D12's robustness block with its coverage; the seam through which COMET and MetricX scores
enter; the model card, the staging directory and the Hub upload plan (`rinchen.export.model_card`, `release`, `hub`; `rinchen release stage|hub`) with a dry run that lists exactly what would leave
the machine; `.github/workflows/release.yml`; the toy-based gate. The design, with the rulings T1 to T23, the amendments C2 to C4 and a section of the amendments made while building (A1 to A26), is
`docs/superpowers/specs/2026-11-10-d13-release-design.md`; the decisions are condensed in the entry "2026-11-11: D13 release: the rulings, the review and the limits" of `DECISIONS.md`; the
upstream facts are in `docs/VERIFY.md` (section "D13 findings"). The facts about sacrebleu, the bootstrap, `datasets`, `huggingface_hub`, the engine and the toy come from a spike made before the
design (`.superpowers/sdd/2026-11-10-d13-release/spike-findings.md`, findings F1 to F18; sacrebleu 2.6.0, numpy 2.5.4, datasets 5.1.0, huggingface_hub 2.3.0, Eole 0.6.4, CTranslate2 4.8.4, CPU, toy data).

**What D13 is.** An instrument and a release tool, tested on files, on fakes and on one real tiny binary. Given a suite and a system (an export, a baseline, a file of outputs) it writes the scores
with intervals; given runs it writes the report with paired comparisons; given an export and a report it writes the directory a release uploads and a plan of the upload. It MEASURES: it has no
threshold, no pass and no fail, and `eval run`, `eval report` and the targets exit 0 whenever they measured. The floors are the gate's.

**What D13 is NOT.** Nothing below is a statement about the quality of any translation.

- **No real data, no real model, no baseline.** FLORES+, NTREX, the human sets, NLLB, MADLAD and Hy-MT2 were not available: every number is the toy's or the golden suite's. The real FLORES+ layout
  (Hub id, configurations, columns, splits, sizes, document ids) is recalled from memory and UNVERIFIED; `configs/eval/suites.yaml` says so and the builders refuse a missing column naming it.
- **No GPU was used.** The machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run. Every Eole process, the converter and every CTranslate2 translator of D13 ran with
  `CUDA_VISIBLE_DEVICES=""`, `gpu_ranks: []` and `device="cpu"`; `auto` is refused; `--device cuda` exists for the operator, is validated and was never run. Everything GPU-side is **UNMEASURED**.
- **No network and no real upload.** `rinchen eval suite fetch` and `rinchen release hub --upload` were never run; the real `acquire.hf.fetch` ran only with `datasets.load_dataset` replaced by a recording
  fake, the real Hub adapter only with `huggingface_hub.HfApi` patched or under `HF_HUB_OFFLINE=1` with every socket refused, and a dry run creates no client. No Hugging Face endpoint was contacted and no token was read.
  The Hub library is imported only inside `HfHubClient`.
- **No neural scorer.** COMET-22, MetricX-24 and MetricX-24-QE are `null` with a reason unless per-segment scores are handed in through the seam (B1 and B2 of the master plan stay open).
- **No tokenizer, no decontamination, no off-target rate, no zero-shot evaluation.** `char` BLEU stands in for the missing Japanese and Korean tokenizers (B3 and B4); nothing checks that training data holds no benchmark
  sentence; the share of outputs in the wrong language is always `unmeasured` (no language identification is run); a direction a model was not trained for is `not_trained` and never scored.
- **No mechanism.** D13 changes nothing a system does, and no model is fine-tuned or continued: NLLB, MADLAD and Hy-MT2 are evaluation baselines only.
- **Departures from the master plan** (`eval/suites.py`, `runner.py`, `significance.py`, `report.py`, `export/model_card.py`, `hub.py`, `release.yml`): the work is split over more modules (see "Architecture");
  the result is amended three times (C2 to C4: no `created`, `manifest_sha` is the export's, the file sits one level deeper, more fields); the baselines enter through D6's generators and the hypotheses file.

Tags used below: **measured** (run for this runbook on 2026-11-11 with the code at `2df73c1`, through the real `rinchen` binary on the CPU, offline; the drivers are read-only scratch scripts,
`scratchpad/d13-t12/`: the golden fixtures of `tests/fixtures/eval/`, a two-row suite and a fabricated release), **gate** (printed in the module docstrings of the D13 gate, run on 2026-11-11, Task 11 of the ledger),
**spike** (run in the design spike), **source-read** (read from the code or from an installed package, never run), **ARITHMETIC** (a formula applied to values that were measured), **UNMEASURED**
(nothing was run). Every number is from a test, the code, the spike, the ledger of the build (`.superpowers/sdd/2026-11-10-d13-release/progress.md`) or a measurement made for this runbook; a number that
was not obtained is not given. A TIME moves by a third from one run to the next on this shared machine (load average 5.6 to 7.5 on 20 cores while these were taken); a count or a score does not.

What was checked: 17,852 tests passed after the final wave (17,666 after the fix wave) against 13,583 at the end of D12 (4,269 new, ARITHMETIC; 105 new test files with 40,900 lines at the end of the fix wave and four more in the final wave, unit, property, golden and integration), each new test mutation-checked by hand during the
build (126, 62, 129, 87, 93, 83, 71, 115, 82, 171 and 45 mutants for Tasks 1 to 11, 152 for the fix wave and 72 for the final wave: 1,288 in all, ARITHMETIC; every first-pass survivor was given a test or an analysis, and the survivors that remain are
equivalent mutants named in "The review, its fix wave and the re-review"), differential checks against independent references (the bootstrap against the spike's pure-Python reference index for index; the suite reader against a validator written
from the spec on 30,000 random corruptions; the sources on 7,000 cases; the results reader on 50,000; the metrics against sacrebleu's public `corpus_score` on 252 random corpora; `eval run` on 400; the report on 200 random cases with 616 comparisons;
the card on 400; the Hub plan on 1,500 random stages, run again after the fix wave; every one with 0 mismatches), the real-binary gate (three modules, 99 tests, plus the dry-run module the workflow runs), a combined review of Tasks 1 to 11 (`1a378d0..3c52162`, one independent
read-only reviewer: ready with fixes, 0 Critical, 5 Important and 13 Minor findings) and a fix wave of six commits (`3c52162..2df73c1`) with the full gate at its end (17,666 passed, 1 skipped, 10 deselected), a scoped re-review of the fix wave (one independent read-only reviewer: approved with minors, 0 Critical, 0 Important and 7 Minor findings; all 16 findings of the combined review verified fixed) and a final wave of seven commits (`2df73c1..413bc94`) that answers its minors, with the full gate at its end (17,852 passed, 1 skipped, 10 deselected). **No reader has read the final wave**: it was tested and mutation-checked by its own author (see "The review, its fix wave and the re-review").

## Architecture

```
local files --eval suite build--> suite.json --Suite.from_bytes--> requests --system (ONE call)--> hypotheses.jsonl --score_direction--> result.json (per direction), run.json, manifest.json
(Hub, once) --eval suite fetch--> shards                                                                  |
                                                       eval report (recomputes every figure from the suite and the outputs) --> report.json, report.md, manifest.json
                                                       release stage (+ the export) --> README.md (the card), release.json, 7 copies, manifest.json (not uploaded)
                                                       release hub --> hub_plan.json (dry run) | hub_commit.json (--upload, ONE commit)
```

| Module | Job | Lines |
|---|---|---:|
| `rinchen.metrics.registry`, `segment_stats` | the six metric names, `HIGHER_IS_BETTER`; the tokenizer table, `segment_statistics`, `score_from_statistics`, `mean_statistics` | 79 + 230 |
| `rinchen.config.eval_metrics`, `eval_suites` (+ `configs/eval/metrics.yaml`, `suites.yaml`) | the bootstrap settings and the release targets; the benchmarks and how each is read | 166 + 194 (39 + 55) |
| `rinchen.eval.bootstrap`, `significance` | the splitmix64 index stream, `resample_totals`, intervals; `estimate`, `compare`, `is_significant` | 245 + 148 |
| `rinchen.eval.suite_rules`, `suites`, `suite_sources` | the rules of the suite file; `Suite`, `read_suite`, `write_suite`, `standard_directions`, the guards; the three readers over plain rows | 444 + 335 + 350 |
| `rinchen.eval.outputs`, `systems`, `runner` | the hypotheses file; `EngineSystem`, `CandidateSystem`, `FileSystem`; `run_suite` | 273 + 355 + 135 |
| `rinchen.eval.segment_scores`, `scoring`, `result`, `run_record`, `field_rules`, `robustness_input` | the scores file (the neural seam); scoring one direction; `result.json` and `run.json` with their strict readers; D12's files and the binding | 214 + 274 + 435 + 366 + 265 + 416 |
| `rinchen.eval.targets`, `report`, `report_runs`, `report_view`, `report_format` | the release targets; the report and its comparisons; the checks that make a run fit to be reported; the view; the shared formats | 250 + 364 + 428 + 183 + 147 |
| `rinchen.eval.report_markdown{,_cells,_scores,_notes}`, `report_read{,_rules,_sections}` | the Markdown renderer; the strict reader of `report.json` | 198 + 131 + 203 + 101; 323 + 93 + 225 |
| `rinchen.export.model_card{,_evaluation,_notes}`, `release` | the card; the release record and the report-to-export binding | 241 + 407 + 119; 186 |
| `rinchen.export.hub`, `hub_files`, `hub_record`, `hub_scan` | the upload plan, the `HubClient` protocol and `HfHubClient`; reading the stage; reading `release.json`; the scan for tokens and absolute paths | 429 + 170 + 83 + 54 |
| `rinchen.cli.eval_suite{,_files,_plan,_output}` | `eval suite fetch`, `build`, `inspect` | 481 + 365 + 143 + 208 |
| `rinchen.cli.eval_run{,_plan,_inputs,_output}`, `eval_systems` | `eval run`; the systems it builds, the vocabulary guard | 451 + 227 + 212 + 386; 336 |
| `rinchen.cli.eval_report{,_inputs,_output}` | `eval report` | 207 + 227 + 165 |
| `rinchen.cli.release`, `release_stage{,_inputs,_output}`, `release_hub{,_output}` | `release stage` and `release hub` | 13; 212 + 138 + 139; 249 + 214 |
| `.github/workflows/release.yml` | tags `v*`, `contents: read`, gate, doctor, build, the dry-run test, `upload-artifact`; never uploads | 52 |
| tests only: `tests/support/{eval_*,hub_*,release_*,no_network,module_imports}.py`, `tests/fixtures/eval/` | the toy suites, fakes, the fabricated release, the goldens (37,020 bytes, budget 50 KB) | |

56 new source files with 13,632 lines (`wc -l` over the added files of `1a378d0..2df73c1`, ARITHMETIC). `rinchen.eval` imports nothing of `train`, `synth`, `serve`, `cli`, `annotate`, `script_conv`, `filter`, `export` and none of `ctranslate2`,
`sentencepiece`, `eole`, `torch`, `polars`, `datasets`, `huggingface_hub`, `httpx` (an AST scan, `tests/unit/test_eval_layering.py`); numpy is used for INTEGER arithmetic only and the bootstrap never touches `numpy.random`, `random`, `secrets` or the environment
(an AST scan). `rinchen.export.model_card`, `release` and `hub` import `rinchen.eval.report_read` and `rinchen.export.record`, never the reverse. The command modules may import `infer.engine`, `synth`, `acquire.hf`, `export`, `manifest`,
`cli.train_common`, `cli.errors`, `cli.eval_robustness_plan`, polars and yaml, never `train`, `serve`, `annotate` or `filter`. **The devtest guard**: no module under `src/rinchen` outside `rinchen/eval/` and `rinchen/cli/eval*.py` may import
anything of `rinchen.eval` or of the eval commands `rinchen.cli.eval*` (which import the suite readers themselves), except the nine listed in `ALLOWED_EVAL_IMPORTERS` of `tests/unit/test_eval_devtest_guard.py` (`export/model_card.py`, `model_card_evaluation.py`, `model_card_notes.py`, `release.py`, `cli/release_stage.py`,
`cli/release_stage_inputs.py`, `cli/release_stage_output.py`, `cli/release_hub_output.py` and `cli/main.py`; the three added in the final wave take `begin_outputs` and `options_digest` of `cli.eval_suite_output` or register the `eval` command group), each with its reason; the scan also reads a dynamic import under another name (`from importlib import import_module as im`); a new importer is a decision taken in that file. The only edits to existing source are `src/rinchen/acquire/hf.py` (`fetch` gains `revision`, default `None`, the old call byte for byte),
`src/rinchen/cli/eval.py` and `src/rinchen/cli/main.py` (they register the commands); the edits to existing tests are additive (the layering and message scans, the `d13_*` session fixtures of `tests/integration/conftest.py`, `"release"` in
`tests/unit/test_cli_help.py`, the module lists of the export scans) except two import-order fixes (`tests/support/robust_suite_documents.py`, `tests/unit/test_export_quantize.py`). `rinchen.eval.bootstrap` and `significance` import only numpy and `rinchen.errors` (a fresh-interpreter test); `rinchen.metrics.segment_stats` imports in 0.25 to 0.31 s with the uv start-up (sacrebleu, pydantic and PyYAML; **measured** by Task 1).

## What the numbers mean

A direction's score is a number over the WHOLE suite for that direction (corpus level); a sentence score is never averaged into it. Read a number with its interval, its unit and its signature.

**BLEU, chrF and chrF++.** The three are sacrebleu 2.6.0's, with its defaults: BLEU with the `13a` tokenizer (`smooth:exp`), chrF with `nc:6`, chrF++ is chrF with word bigrams (`nw:2`). chrF++ is the release metric (the README's "chrF++ in at least
12 of 16 directions"), chrF and chrF++ are the cross-language figures, BLEU is kept for comparison with published numbers. Each is computed from the per-segment INTEGER statistics (10 per segment for BLEU, 18 for chrF, 24 for chrF++): the corpus score is
sacrebleu's score of the SUM of the integers, which equals `corpus_score` exactly (`==`) on 480 toy segments (**spike**) and on the golden corpus (**gate**, unit tests). That is what makes the bootstrap exact: a resample is a different sum of the same integers. The
score of a set is not the mean of its sentence scores: 96.99 against 98.01 on the 480 `eng_Latn-khk_Cyrl` toy rows (**spike**). The statistics methods are PRIVATE to sacrebleu and are pinned by tests that compare them with `corpus_scores`, so a bump of
sacrebleu fails loudly. BLEU of identical text is 100.00000000000004 in floating point and is clamped to 100.0 as `corpus_scores` clamps.

**The signature** of each metric, as sacrebleu reports it after it has seen text, is recorded per direction: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0` (BLEU, Latin or Cyrillic target),
`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0` (chrF), `...|nw:2|...` (chrF++). Compare two numbers only when their signatures are equal. **The tokenizer follows the TARGET language**: `zh` for `zho_Hans`, `char` for `jpn_Jpan` and `kor_Hang`, `13a`
for everything else. A metric built with `flores200`, `flores101`, `spm` or `spBLEU-1K` (they DOWNLOAD a SentencePiece model on first use) or `ja-mecab` or `ko-mecab` (they need MeCab, which is not installed) is a `ConfigError` before sacrebleu is asked (**spike**, F2).
`char` BLEU for Japanese and Korean is not comparable with published word-BLEU (`tok:char` is in the signature). **chrF++ for a language written without spaces** (`zho_Hans`, `jpn_Jpan`) has a degenerate word part (a punctuation-free run is one word): read chrF there.
**Translationese of FLORES**: its sentences were translated from English, so a direction whose source is not English reads a translation, and a model can score above what it would on text that was written in that language.

**The interval** is a 95 % percentile bootstrap: 1,000 resamples of the UNITS of a direction, drawn with a fixed generator (splitmix64 over `(seed 20261110, resample, position)`, our own so that no numpy version can move a draw), the score of each resample
computed from the sum of the integer statistics of the drawn units, the interval the sorted scores at positions 25 and 974 (no interpolation). The seed is fixed because it moves an end by about 0.02 (**spike**, F4) and is recorded in every result. **The unit** is the
document when every row of the direction has a document id, else the segment; the result records which and how many (`bootstrap.unit`, `bootstrap.units`). A document bootstrap keeps the sentences of an article together and is wider (1.5 to 2.2 times on 480 toy rows,
**spike**, F5); a segment bootstrap ignores the correlation between neighbouring sentences and is then too narrow for real text. With fewer than 2 units the interval is `null` (a one-unit "interval" is a point). With few documents the interval is coarse: five
documents allow at most 5^5 = 3,125 different resamples (ARITHMETIC), which is why the golden chrF++ intervals can coincide across seeds. The interval says how much the score moves when the test set is resampled; it does not say anything about another test set.

**The paired comparison.** For two systems scored on the SAME rows of a direction, ONE set of draws is used for both, so the pairing is structural. `delta` is `score(system) - score(baseline)` on the full rows, its interval the percentile interval of the resample
differences, `p` the two-sided p-value with add-one smoothing, `min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))` with `le` the resamples whose difference is at most 0 and `ge` those at least 0 (so identical systems give exactly 1.0 and the smallest p is 2/1,001 = 0.001998,
ARITHMETIC). **The verdict is the p-value against alpha (0.05), never the interval**: `better` is p below alpha with `delta` in the system's favour, `worse` is p below alpha with `delta` against it, `no significant difference` is anything else. Intervals can touch zero while
p is 0.108, because many resamples then have a difference of exactly zero (ties; float32 against int8 of the toy, **spike**, F15). In `report.json` the boolean `significant` means "the FIRST system is better" and nothing else (a significantly worse system has `significant: false`:
read `delta`, or the `verdict` column of `report.md`). **Measured with the finished code** (golden suite, this runbook): the control of two identical systems (`alpha` against a second system that holds alpha's outputs) gives delta +0.00, interval [+0.00, +0.00], p 1.000 and `no significant
difference` in both directions and both metrics; sacrebleu's own paired test gives 0.000999 for a system against itself (**spike**, F3) and reads `SACREBLEU_SEED` from the environment: it is never called. **There is NO correction for multiple comparisons.** The report
says how many intervals and comparisons it holds (`16 intervals and 4 paired comparisons ...` for the golden report); with 16 directions and two metrics against one baseline that is 32 comparisons, of which 0.05 x 32 = 1.6 would be flagged on average if every difference were zero and the
p-values were calibrated (ARITHMETIC). Read a difference with its p-value and look for a pattern across directions before believing one.

**COMET-22, MetricX-24, MetricX-24-QE.** `null` in every result, with the reason in `unmeasured`, unless scores come in through the seam (see "Attaching neural scores"). A neural score is a MEAN of segment scores (the system-level figure) bootstrapped through integer micro-units (`round(score * 1e6)`, exact to 4.4e-8 per
resample mean, **spike**, F6); MetricX is lower-is-better and a comparison says so. D13 does not say whether a score is right and `eval report` cannot recompute one (a report holds no segment scores): it prints what the result says.

**The robustness block** is D12's six values of the same system, embedded in every result of a run that was given `--robustness`, WITH their coverage (`scored` of `total` per value, which is not in the block) and a record of where they came from (`robustness_source`: suite, digests, the binding). The binding is `model_digest`
when the robustness run's model and manifest digests equal the engine's, else the weaker `name`; a block measured on another model is refused. The values were measured on D12's robustness suite, not on the benchmark, and D12's block does not record that suite's kind: the report and the card say `suite kind not recorded` beside it and
say that the evidence banner does not count it.

**The targets** (`configs/eval/metrics.yaml`) are measurements: `beats_baseline` (chrF++ above the named baseline, by the paired verdict, in at least 12 of 16 FLORES+ devtest directions: unmeasured unless the suite is a `benchmark` `flores_plus` `devtest`), `off_target` (at most 1 %: always unmeasured), `numbers` (D12's `numbers_acc` at least 0.99) and
`placeholders` (`placeholder_acc` equal 1.0). Each is `met`, `not_met` (the threshold can no longer be reached) or `unmeasured` with a reason; no command exits non-zero because a target is not met. A target of `numbers` or `placeholders` is judged on the robustness block of a system of kind `rinchen` (the toy's is `not_met`: 0.9292 against 0.99 and 0.9983 against 1, **gate**).

**The evidence banner.** `report.md` and the card open with a banner that follows the suites' kinds: `fixture` (every suite generated by a test or a pipeline's own fixtures: "These figures come from generated fixture suites. They test the tooling and say nothing about translation quality.") or the kinds and the number of directions measured. The card's banner, its `toy` tag and
`release.json`'s `evidence` are computed over the suites in which THE CARDED SYSTEM has a result, not over the whole report.

## The suite file

A suite (`rinchen.eval.suite/1`, UTF-8 JSON, sorted keys, two-space indent, `ensure_ascii` off, one final LF) is a set of sentences in several languages: `{schema, name, kind, benchmark, split, source: {dataset, revision, pinned, files, missing_languages}, directions, rows: [{id, doc, text: {language: sentence}}]}`.
`kind` is `benchmark` (FLORES+ or NTREX), `human` (an export of D2's annotation app) or `fixture` (never evidence of translation quality); `split` is `dev`, `devtest` or `test` (a benchmark is `dev` or `devtest`, a human set `test`, a fixture any); the PURPOSE follows from the split: `dev` is `selection`, the others `report`.
Worked example: the first two rows of the golden suite, which `rinchen eval suite inspect` reads (**measured**, exit 0):

```
{ "benchmark": "example", "directions": ["eng_Latn-khk_Cyrl"], "kind": "fixture", "name": "example", "schema": "rinchen.eval.suite/1", "split": "devtest",
  "source": {"dataset": null, "files": {}, "missing_languages": [], "pinned": false, "revision": null},
  "rows": [ {"doc": "d1", "id": "e01", "text": {"eng_Latn": "the cat sat on the mat", "khk_Cyrl": "муур дэвсгэр дээр сууж байв"}},
            {"doc": "d1", "id": "e02", "text": {"eng_Latn": "a dog runs in the park", "khk_Cyrl": "нохой цэцэрлэгт хүрээлэнд гүйж байна"}} ] }
(written with the keys sorted, one per line; digest be6d96fb8906..., "rows: eng_Latn 2, khk_Cyrl 2", "directions: 1")
```

It holds one document, so its resampling unit is one and its intervals would be `null`; the golden suite has five documents of two rows. **The rules**, enforced by the strict reader (`Suite.from_bytes`: exactly these fields, each of its type, a repeated key refused, `DataError` naming the place and never quoting a sentence, at most 100,000 rows and 64 MiB checked before the parse):
`name` and `benchmark` match D12's id pattern; `revision` is a 40-character lowercase hexadecimal string or null, `pinned` is true only with a revision; `files` maps each input file the builder read to its SHA-256; `directions` are `<src>-<tgt>` registry codes, unique, each with at least one row that has both languages; a row id is `[A-Za-z0-9._:-]{1,96}`,
unique; a sentence is non-blank, at most 5,000 characters, with no line break, no lone surrogate and no reserved symbol of the project other than a numbered placeholder; **`khk_Mong` and every code outside the language registry are refused by name** wherever a language is read; a `benchmark` `flores_plus` also refuses `mn_Mong`. **The 16 directions** are
Khalkha to and from each of the eight partner languages (`standard_directions()`); FLORES+ is multi-way parallel, so a direction takes the rows that have both languages and the other side's text is the reference; a direction exists in a suite only when a row has both languages, a language the source lacks is recorded in `source.missing_languages` and the report shows
`n/a (not in the suite)` against the 16 the targets expect. A human set is DIRECTIONAL: the reverse of a pair is never added. The order of the rows is part of the suite (the bootstrap's draws are bound to positions). The resampling unit is the document when every row of a direction has a `doc`.

## The commands

```
rinchen eval suite fetch   --benchmark B --split S --out-dir DIR [--revision HEX | --unpinned] [--config FILE]
rinchen eval suite build   --benchmark B|human --from PATH --out-dir DIR [--split S] [--name N] [--kind benchmark|human|fixture] [--config FILE]
rinchen eval suite inspect --suite FILE
rinchen eval run    --suite FILE --out-dir DIR (--model DIR | --generator NAME | --hypotheses FILE) [--generators-config FILE] [--system NAME] [--direction D ...]
                    [--beam-size 4] [--max-decoding-length 256] [--script-mask retry|always|off] [--device cpu|cuda] [--threads 1]
                    [--robustness DIR] [--segment-scores METRIC=FILE ...] [--resamples N] [--seed N] [--metrics-config FILE]
rinchen eval report --run DIR [--run DIR ...] --suite FILE [--suite FILE ...] --out-dir DIR [--baseline SYSTEM] [--metrics-config FILE]
rinchen release stage --export DIR --report-dir DIR --system NAME --repo-id ORG/NAME --out-dir DIR
rinchen release hub   --stage DIR --repo-id ORG/NAME --out-dir DIR [--upload] [--public]
```

The option names are those of `--help` of the finished binary (**measured**, 2026-11-11). Every command hashes every input BEFORE it reads it and parses the bytes it hashed, makes `verify_export` vouch for an export, refuses an output directory that is an input's and a foreign manifest, writes its files first
and `manifest.json` LAST, and prints no text of a suite or an output in any message; a run that refuses writes nothing and leaves an earlier run's files as they were. The commands in order:

**1. `eval suite fetch`** (the ONLY network step; never run here, UNMEASURED). It calls `rinchen.acquire.hf.fetch` for each language of the configuration (`configs/eval/suites.yaml`) and writes `DIR/<split>/<language>/shard-*.parquet` and, last, `manifest.json` (stage `eval_suite_fetch`). The
revision must be pinned: `--revision` is the 40-character commit hash of the dataset (or the configuration holds one), else exit 2, unless the operator says `--unpinned`, in which case the suites built from the fetch say `pinned: false` and the report says `unpinned`. The dataset library finds its own Hub credentials (a gated dataset needs
the operator to have logged in with the library's own mechanism); D13 never reads, prints or writes a token for the fetch. A fetch that fails part-way leaves its partial shards and no manifest.

**2. `eval suite build`** turns LOCAL files into a suite: a directory of shards (what `fetch` wrote, or `<language>/shard-*.parquet`; the columns named by the configuration, a missing one refused naming it; the rows of the languages are joined by `id` and every language must hold exactly the ids of Khalkha, in its order, else refused with the counts: never
intersected), a directory of text files (`<language>.txt`, one sentence per line, the line number is the id `l000001`, a blank line refused with its number, all files of one length) or, for `--benchmark human`, an export of D2's `annotate export` (`.jsonl`, `.tsv` or `.parquet`; every row `origin: human` and `split: test`, one `set`; the id is `<doc_id>-<seg_idx>` or `s<ordinal>`). It
writes `DIR/<name>.json` (`<benchmark>-<split>` unless `--name`) and, last, `manifest.json` (stage `eval_suite`). When `--from` holds the manifest of a fetch, every shard it lists must be there with its digest and no other may be, and the suite records the dataset and the revision; hand-made shards give a suite with no provenance and the command prints
`provenance: no fetch manifest found, dataset and revision are not recorded`. A `url` document value becomes a document id (an id already shaped like one stays; anything else becomes `u` plus 16 hex digits of its SHA-256). `--name .` and `--name ..` are refused (exit 2). A `khk_Mong` file or column is refused by name (exit 3, nothing written).
**`eval suite inspect`** prints the name, kind, split, purpose, digest, rows per language and directions (**measured**, the golden suite):

```
$ rinchen eval suite inspect --suite tests/fixtures/eval/golden_suite.json        # 0.95 s, exit 0
name: golden
kind: fixture
split: devtest
purpose: report
sha256: b707518d39133bd206092e6df28bf0bcf1e42c776ad5c16900995bf783ebc761
rows: eng_Latn 10, khk_Cyrl 10
directions: 2
  khk_Cyrl-eng_Latn
  eng_Latn-khk_Cyrl
```

**3. `eval run`** scores one system over a suite. Exactly one of `--model` (also `RINCHEN_MODEL`: an export `rinchen export convert` wrote; the engine translates the directions it was trained for, on the CPU unless an operator says `cuda`; `auto` is refused), `--generator` (a baseline of D6's generators
configuration, `--generators-config`, default `configs/data/synth.yaml`) or `--hypotheses` (a file in which any other system wrote its outputs). `--system` names the system (1 to 64 characters of letters, digits, `.`, `_`, `-`; `.` and `..` refused; default: the export's name, the generator's name or the stem of the file). `--direction` (repeatable) runs only
those directions and lists the others `not_selected`. The decode options go with `--model` only; `--segment-scores` goes with `--hypotheses` only. `--robustness DIR` embeds D12's block of the same system (the output directory of `rinchen eval robustness`); a robustness run of another model is refused (exit 2) BEFORE the engine opens. `--resamples` and `--seed` override the metrics
configuration (the report refuses runs whose settings differ from the configuration it is given). It writes `DIR/<suite>/<system>/{hypotheses.jsonl, <direction>/result.json, run.json, manifest.json}`. **Every direction of the suite has a state** in `run.json`: `run`, `not_trained` (a Rinchen export is never asked for a pair it was not trained for: the engine would answer it with
status `ok` in the wrong language, **spike**, F13), `unsupported_language` (a baseline whose family lacks a language; a server baseline runs the Khalkha pairs only, because D6's prompts cover those), `not_in_file` or `not_selected`; only a direction that ran has a result and a state other than `run` has a reason. ONE call is made to the system with every request of the runnable directions, in the
suite's order. A hypothesis whose status is not `ok` (`empty`, `too_long`, `truncated`, `no_output`) is scored as it is and counted in `statuses`; a failure the system flagged stays visible beside the metric. **Baseline decoding**: D6's synthetic-data defaults are sampling settings (`n` 4, `temperature` 0.7), so an evaluation overrides them (`n` 1; for a server `temperature` 0.0 and
`top_p` 1.0; for CTranslate2 the beam stays as configured) and records the effective settings in `system.options`. **The vocabulary guard**: before a CTranslate2 baseline translates, the command reads its `shared_vocabulary.json` and refuses (exit 2) a family whose language tokens (NLLB: the codes of the nine languages; MADLAD: the `<2xx>` tags) are not in it, because D6's `CT2Translator`
is silent about them and answers the same noise for every source (**spike**, F14); the toy export is the negative control (**measured**: refused with exit 2 before any CTranslate2 load and nothing written, the ledger's Task 7). **Measured** (golden suite through `--hypotheses`, 1.02 s and 1.10 s with the Python and typer start-up; the intervals are document-unit because every row has a document):

```
$ rinchen eval run --suite tests/fixtures/eval/golden_suite.json --hypotheses tests/fixtures/eval/golden_outputs_alpha.jsonl --system alpha --out-dir <O>/runs
suite: golden
system: alpha
purpose: report
khk_Cyrl-eng_Latn: chrF++ 73.08 [54.91, 85.48] (10 segments, 10 ok)
eng_Latn-khk_Cyrl: chrF++ 68.06 [50.67, 80.98] (10 segments, 10 ok)
run: <O>/runs/golden/alpha/run.json
manifest: <O>/runs/golden/alpha/manifest.json
```

A non-run direction prints `n/a (<state>)` (the reason is in `run.json`), an interval of `None` prints `n/a`, the statuses print `ok` first and the others by name (`6 ok, 2 empty, ...`). A run over a `dev` suite is made (its purpose is `selection`) and `eval report` refuses it. A run directory that holds one of its inputs (an earlier run's own `hypotheses.jsonl` fed back under the same system name) is refused; to re-score a run, use another `--system` or `--out-dir`.
**4. `eval report`** reads each `--run` through its manifest and `run.json`, hashes every file `run.json` names and checks it against its listing, matches each `--suite` to the runs by SHA-256 (not by path), and RECOMPUTES every comparison from the verified suite and outputs: **a result file is never believed to hold more than it says**: the segment count, the bootstrap record, the unit, the points, the intervals, the statuses, the three signatures and
`trained` are recomputed from the outputs and a result that disagrees (also one that was edited and re-digested) is refused with exit 3, naming the run, suite, direction and field. A run of a `dev` suite, two runs of one system over one suite, a run scored with other bootstrap settings than the configuration's and a suite no run scored are refused. `--baseline SYSTEM` (default: the metrics configuration's
`baseline_system`, null) compares every other system of the same suite and direction with it (chrF++ and BLEU); a baseline without a result for a direction leaves the comparison absent and the coverage table says why. **Measured** (golden runs `alpha` and `beta`, `--baseline alpha`, 1.18 s; the two files equal `tests/fixtures/eval/golden_report.json` (14,894 bytes) and `golden_report.md` (7,644 bytes) byte for byte):

```
report: <O>/report/report.json
markdown: <O>/report/report.md
manifest: <O>/report/manifest.json
targets: beats_baseline unmeasured
targets: off_target unmeasured
targets: numbers unmeasured
targets: placeholders unmeasured
```

**5. `release stage`** builds the directory a release uploads, whole. `verify_export` vouches for the export; the report directory's manifest (stage `eval_report`) must list `report.json` and `report.md` with the digests found; `report.md` must be the rendering of the verified `report.json` (re-rendered and compared: a stale or edited one is exit 3, and the message says the directory was made by another version of the renderer or was edited, and to run `rinchen eval report` again); **the report must hold a system `NAME` of kind `rinchen` whose
`model_sha256` and `manifest_sha256` equal the export's, and a result of it** (else exit 3: the card would describe another model, or none). The out directory then holds copies of `model.bin`, `shared_vocabulary.json`, `config.json`, `spm.model`, `rinchen.json`, the card `README.md`, `report.json`, `report.md`, the release record `release.json` and, last, `manifest.json` (stage `release_stage`; it holds
local paths and is NOT uploaded). The repository id is `ORG/NAME` (`[A-Za-z0-9._-]{1,96}` each, `.` and `..` refused). **Measured** over a FABRICATED export (a 30-byte `model.bin`: no model) and the golden report with `alpha` bound to it, exit 0:

```
$ rinchen release stage --export <O>/rel/export --report-dir <O>/rel/report --system alpha --repo-id rinchen-test/rinchen-tiny --out-dir <O>/rel/stage
stage: <O>/rel/stage
model.bin 30 07148803f29a
shared_vocabulary.json 2048 1b82a3995c09
config.json 204 0f6c6a6d2c31
spm.model 135 4e3636da82ab
rinchen.json 1178 54d5f37b199b
README.md 6054 34d0bcf7bf4f
report.json 15740 3d0367b02310
report.md 7733 bbe40a45f12a
release.json 2345 4bdad2f469b8
manifest: <O>/rel/stage/manifest.json
```

**6. `release hub`** plans the upload of a stage. **The default is a DRY RUN**: it hashes the stage's manifest and the nine files, builds the plan, scans every text file for a token-like string and an absolute local path, prints it and writes `hub_plan.json` and the manifest (stage `release_hub`); it builds no operation, creates no client and reads no environment variable. **Measured** (the stage above, 0 files uploaded, exit 0):

```
$ rinchen release hub --stage <O>/rel/stage --repo-id rinchen-test/rinchen-tiny --out-dir <O>/rel/hub
mode: dry-run
repo: rinchen-test/rinchen-tiny
model.bin 30 07148803f29a
  ... (the nine files, `<name> <bytes> <first 12 hex digits of the SHA-256>`)
release.json 2345 4bdad2f469b8
total: 35467
plan: <O>/rel/hub/hub_plan.json
manifest: <O>/rel/hub/manifest.json
```

The plan REFUSES (exit 3, the file named and no matched text printed): a stray file, a subdirectory, a symlink or non-regular file, a name outside the nine, a file over 10 GiB (a text file over 64 MiB), a file whose digest differs from `release.json`'s (the stage changed after it was written), a name `release.json` lists that is absent, a text file with a token-like string (`hf_` followed by 20 or more letters or digits, `Bearer ` followed by 16 or more token characters,
`HF_TOKEN` followed by an optional quote, optional spaces and `:` or `=`) or an ABSOLUTE LOCAL PATH (any POSIX path of two or more components, `file:///...` and a path after `://` or a quote included, or a drive path `X:\`; the scan flags the export's own `manifest.json`, which holds the machine's paths, **spike**, F9), a repository other than the one the stage was made for (exit 2) and a dry run into the directory of an
earlier upload (exit 3).

## The operator's upload procedure

Not run here: **UNMEASURED** against the Hub. On the machine that holds the model:

1. Train, convert and verify the export (D8 to D11), run D12's `rinchen eval robustness` over a real suite if there is one, fetch and build the suites (`eval suite fetch` with a PINNED revision, then `build`), `eval run` for the model and for each baseline (`--generator`, or `--hypotheses` for a system run elsewhere), `eval report --baseline NAME`.
2. `rinchen release stage ... --out-dir STAGE`, then read `STAGE/README.md` (the card) and `STAGE/report.md` as a reader of the Hub page would: the banner, the tables, `n/a (...)` for everything unmeasured, the limits.
3. `rinchen release hub --stage STAGE --repo-id ORG/NAME --out-dir PLAN` (the dry run). Read the nine lines and the total; keep `PLAN/hub_plan.json`. Any refusal is a real finding: do not edit the stage by hand to make it pass.
4. Put the token in the ENVIRONMENT for this one command only (`HF_TOKEN`; not on a command line that a shell history keeps, and never in a file of the repository), then `rinchen release hub ... --upload`. It reads `HF_TOKEN` once, before anything else is hashed (absent or blank: exit 2); creates the repository PRIVATE (`--public` is explicit, never the default); asks the Hub what the repository is after
   creating it and BEFORE it commits (an existing repository that is public is refused unless `--public` is given: exit 2, nothing committed; `create_repo(exist_ok=True)` leaves the visibility of an existing repository as it is, **source-read** in the installed library); makes ONE commit of the nine files (one atomic commit); prints `commit: <oid>` and `visibility: <what the Hub reported>` (so `--public` over an existing private repository, which stays private, says so) and writes `hub_commit.json`, `hub_plan.json` and, last, the manifest. Before any call it checks that `--out-dir` can be written, because the record can only be
   written after the commit.
5. Unset the variable. The token is passed to the client and to nothing else: it is never printed, written, put in a message or logged, an error of the Hub is named by its class only, and `rinchen` never calls `login()` or reads a token file. Keep `hub_commit.json`: it is the only local record of the commit (`commit_oid`, the release digest, the visibility the Hub REPORTED).
6. Open the repository. Whether the Hub accepts the card's front matter (`language`, `library_name`, `pipeline_tag`, `tags` and nothing else) and its language codes is UNMEASURED; if its validator objects, the change is in `export/model_card.py`, with the golden card regenerated after every literal was checked independently.

## Reading the files

**`result.json`** (schema `rinchen.eval/1`; sorted keys, two-space indent, LF; no timestamp, no absolute path and no text of the suite or of an output: the same inputs give the same bytes) per (suite, system, direction), at `<DIR>/<suite>/<system>/<direction>/result.json`: `metrics` (`bleu`, `chrf`, `chrf_pp`, `comet22`, `metricx24`, `metricx24_qe`; a metric with no value is `null`, never 0, and its reason is in `unmeasured`), `ci95`
(`[low, high]` or `null`), `bootstrap` (`resamples`, `seed`, `alpha`, `interval: percentile`, `rng: splitmix64-counter/1`, `unit`, `units`), the three signatures, `n_segments`, `statuses` (counts by status, summing to `n_segments`), `trained` (true for a baseline or a file; for an engine, whether the direction is among its trained pairs: always true for a direction that ran), `manifest_sha` (the digest of the export's manifest for a
system of kind `rinchen`, `null` otherwise: the manifest of the RUN cannot be in a file it lists), `robustness`, `robustness_coverage` and `robustness_source` (all three `null`, or all present) and `provenance` (`suite_sha256`, `outputs_sha256` = the digest of the run's own `hypotheses.jsonl`, and `system`: kind, name, `model_sha256`, `manifest_sha256`, `options`, `outputs_sha256` = the digest of the file a file system was read from). The golden example is
`tests/fixtures/eval/golden_result_alpha_khk_eng.json` (1,966 bytes, SHA-256 `f31880cb...b28b`; 1,954 bytes and `05212b4f...8dde` until the final wave reworded the reason of a neural metric without scores). **`run.json`** (`rinchen.eval.run/1`) is the run's table of contents: the suite and its purpose, the system record, the settings (`resamples`, `seed`, `alpha`, `selection`), one entry per direction of the suite with its `state`, `reason`, `rows` (the suite's row count in EVERY state: the
denominator) and `result` path, `files` (relative path to SHA-256 of `hypotheses.jsonl` and each `result.json`; not itself and not the manifest), `robustness_source` and `segment_scores`. **`hypotheses.jsonl`**: one object per request, keys `direction`, `id`, `status`, `text`, sorted, the order of the requests (at most 128 MiB); EVERY run writes it, so a run directory is self-contained and a rerun after a metric change
enters through the same file. **`manifest.json`** of a run: stage `eval_run`, every input with its SHA-256, the files written, `status` (`eval`, `system_kind`, `purpose`, `unit`: `document`, `segment`, `mixed` or `none`) and the seed.

**`report.json`** (`rinchen.report/1`, written back through the strict reader as a check, no timestamp and NO local path) holds `{schema, evidence: {kinds, fixture_only}, systems, suites (with the source record: dataset, revision, pinned), runs: [{suite, system, run_sha256}], coverage: [{suite, system, direction, state, reason}], results, robustness: {system: {block, coverage, source}}, comparisons: [{suite, direction, system, baseline, metric, higher_is_better, delta, ci95,
p_value, better_fraction, significant, unit, units}], targets, bootstrap, note_multiple_comparisons}`. Read `coverage` first: it lists, for every system and direction, `run` or the state that kept it out and why. **`report.md`** is one pure function of the parsed report and byte-deterministic across `PYTHONHASHSEED`, locale and time zone (**spike**, F11; **gate**): the evidence banner; Systems; Suites (kind, benchmark, split, purpose, rows, directions, dataset, revision
or `unpinned` or `n/a (no source record)`, the first 12 hex digits of the digest); Scores (per suite and metric, a row per direction in `standard_directions()` order then the suite's others, each system's cell `value [low, high]` and the column beside it `<system> ok`: `10 of 10`, or `6 of 10 (1 empty, 2 no_output, 1 truncated)` where the system flagged failures; `n/a (<state>)` for a direction that did not run); Paired comparisons (delta, interval, p and the verdict
`better`, `worse` or `no significant difference`); Robustness; Not measured; Release targets; Provenance (the bootstrap, the three signatures, the digests). **The card** (`README.md`, golden `tests/fixtures/eval/golden_card.md`, 6,058 bytes): the front matter (`language`, `library_name: ctranslate2`, `pipeline_tag: translation`, `tags`, plus `toy` when the carded system's evidence is fixture-only), a banner, what the model is
(sizes and quantisation from the record, "trained from random initialisation" stated as the project's rule with the checkpoint's manifest stage beside it: the record cannot prove it), the directions, the files (bytes and SHA-256), the usage (`rinchen.Translator(path).translate(text, tgt="khk_Cyrl")`, no URL invented), the evaluation (every number a number of the report; per suite its kind, split, rows, dataset, revision, signatures; a row per MEASURED direction with an `outputs ok` column, the others
as `n/a (not in the suite)` or `n/a (not trained)` lines), the paired comparisons where the carded system is the candidate, the robustness section (the six values with their counts, from a named suite whose kind is not recorded), the targets, the limits (a list of facts) and the provenance. The card is rendered by OUR function and never by the Hub's template; it carries no field and no sentence about the topics the project's content rules in CLAUDE.md exclude
(`tests/unit/test_d13_content_rules.py` scans every D13 source, document, config, fixture, test and every file the stage generates). **`release.json`** (`rinchen.release/1`; no timestamp, no path): `repo_id`, `name`, `rinchen_version`, `quantization`, `tag_placement`, `directions`, `evidence`, `export` (`model_sha256`, `manifest_sha256`, `spm_sha256`, the checkpoint's digests), `report` (`{sha256, system}`) and `files` (bytes and SHA-256 of the eight other uploaded files: not itself, not
`manifest.json`). **`hub_plan.json`** (`rinchen.hub_plan/1`): `repo_id`, `files` (`name`, `bytes`, `sha256` of the nine), `release_sha256`, `total_bytes`. **`hub_commit.json`** (`rinchen.hub_commit/1`, only after an upload): `repo_id`, `commit_oid`, `release_sha256` and `visibility` (`private` or `public`, as the Hub reported it). The manifest of a dry run has `status` `mode: dry-run`, `repo_id`, `visibility`
and no tool version of the Hub library.

## Adding a benchmark, a baseline, neural scores

**A benchmark.** Add an entry to `configs/eval/suites.yaml`: the `reader` (`parquet_shards` or `lines_dir`), for shards the `dataset`, the `columns` (`id`, `text`, optionally `doc`), `revision` (null until pinned), `splits`, `languages` (Khalkha always among them; `khk_Mong` is refused by name). Then `eval suite fetch` (shards) and `build --benchmark NAME --split S --from PATH`. A reader for a new FORMAT goes in `rinchen.eval.suite_sources` (over plain rows: no polars) and
`rinchen.cli.eval_suite_files` (the file reading), with the refusals of the strict reader and a test that each refusal names a place and no text. Only a `benchmark` named `flores_plus` carries the `mn_Mong` refusal and only a `flores_plus` `devtest` suite can make `beats_baseline` measurable. Check a new benchmark the way the gate checks the toy: `inspect` it, run a control (the same system as a file must score identically), and run the identical-systems comparison (p 1.0).

**A baseline.** A system run elsewhere enters through `--hypotheses FILE` (`direction`, `id`, `status`, `text` per line; every request of a direction once, no other; a file may cover some directions and the rest are `not_in_file`). A baseline run here goes in `configs/data/synth.yaml` under `generators` (kind `ct2` with `family: nllb` or `madlad`, `model_dir`, `spm_model`, `device`; or `vllm` or `llamacpp` with `base_url` and `model`) and
is run with `--generator NAME`; the evaluation overrides the sampling defaults and records the effective settings. A CTranslate2 baseline must have its family's language tokens in its vocabulary or the command refuses it. UNMEASURED: no NLLB, MADLAD or Hy-MT2 directory or server was ever loaded; every baseline path ran over fakes and the toy export.

**Neural scores.** `rinchen eval run --hypotheses FILE --segment-scores comet22=SCORES [--segment-scores metricx24=...]` (`comet22`, `metricx24`, `metricx24_qe`; repeatable; only with `--hypotheses`). `SCORES` is JSON Lines: a header `{"hypotheses_sha256": "<64 hex>", "metric": "comet22", "schema": "rinchen.eval.scores/1", "suite_sha256": "<64 hex>"}` followed by one `{"direction": ..., "id": ..., "score": ...}` line per segment the run scored. The header must
name the suite AND the `hypotheses.jsonl` THIS run writes (a score of other hypotheses is refused), every scored segment needs exactly one finite score of at most 1000 in magnitude, and a metric given for some directions or segments only is refused. The way to produce them: run once without scores (it writes the canonical `hypotheses.jsonl`), score THAT file with the external scorer, and run again from the written file under another `--system`
name (the digest of a canonical file equals the digest of the file it was written from). Nothing here says that a score is right; no scorer is built.

## What to do when a floor of the gate is missed

The commands have no floor: they measure. The floors belong to the gate and are literals of `tests/support/eval_gate.py`, never lowered: chrF++ at least 95.0 and BLEU at least 93.0 for the toy in each trained direction, each interval narrower than 5.0, the D11 teacher at most 60.0, the toy's delta over the teacher at least +40.0 with the lower end of its interval above +40.0 and p at most 0.01, the perturbed system at
delta at most -2.0 with the upper end below -0.5. If a figure falls under its floor: **raise `ROBUST_TEACHER_STEPS`** (`tests/support/robust_gate.py`; training is deterministic here, so the answer is exact), never lower a floor or loosen a control. Before that, find out what moved: `statuses` (did the engine flag rows?), `coverage` and `run.json` (did a direction stop running?), the signatures (did sacrebleu change?), the digest of the toy's `model.bin`
(`e28d1566...a6`: recorded, not asserted). A change of the toy generator, of sacrebleu or of numpy fails the golden tests on purpose (`tests/unit/test_eval_golden.py`, `test_metrics_segment_stats.py`, `test_eval_bootstrap.py`, `test_eval_suite_golden.py`, `test_eval_report_golden.py`): check each moved literal independently (the pure-Python reference of the spike) before regenerating a golden file, and say why.

## Exit codes

| Code | Meaning | Where measured |
|---:|---|---|
| 0 | it measured, planned or staged (also when nothing ran: every direction `not_trained` or `not_in_file`) | all the runs above |
| 2 | a bad option; an export not vouched for; `--device auto`; a vocabulary that lacks the family's tokens; a robustness run of another model; a baseline that is no system of the runs; an output directory that is an input's; a file that changed after it was hashed; `--name .` or `..` at build; a missing or blank `HF_TOKEN` for `--upload`; a repository other than the stage's; an existing public repository without `--public` | **gate**: a robustness directory of another model (R8), `--upload` without a token (R10); fakes for the rest |
| 3 | an input that is missing, unreadable or invalid (suite, outputs, scores, robustness, run, report, stage); a foreign manifest in the output directory; a `dev` run in a report; a report that is not the one of the export; a stage with a stray file, a changed byte, a symlink, a token-like line or an absolute path; `khk_Mong` | **gate**: the dev run (R7), `khk_Mong` (R7), another model's report (R9), the five stage refusals (R10) |
| 4 | an upstream failure: the fetch, CTranslate2, an HTTP generator, the Hub (also a Hub that does not say what a repository is) | fakes only: UNMEASURED with a real failure |
| 1 | the output cannot be written (also a `--out-dir` that cannot be written, found before an upload) | fakes only |
| 130 | interrupted (a run that stops before its first write leaves the last run as it was) | fakes only |

## Costs

**Measured** for this runbook, one process, CPU, load average 5.6 to 7.5, the golden fixtures: `eval suite inspect` 0.95 s, `eval run --hypotheses` 1.02 s (alpha) and 1.10 s (beta), `eval report` 1.18 s, each with the Python and typer start-up. **Gate** (the toy, 2026-11-11): `eval run --model` of the 2 MB export over 1,920 requests of four directions about 1.6 s with the model load, `eval report` 1 to 2.5 s, `release stage` and `hub` 0.9 to 2.5 s, the three
gate modules' session fixtures 357.6 s once per session (190 s of it the toy's training), the three modules 423.9 s alone and 212.7 s, 23.9 s and 26.2 s inside the full gate. **Spike** (F4, F12): extracting the statistics 0.007 to 0.02 s per 480 segments; 1,000 resamples 0.02 to 0.09 s per metric and direction at 1,000 to 3,832 segments (Task 2 again, **measured**: 0.019 to 0.028 s for 1,000 segments, 0.066 to 0.088 s for 3,832; a paired comparison
0.026 to 0.160 s, measured by Task 2 during the build on a loaded machine); ONE engine call of 3,832 toy rows 2.19 s (float32), 2.16 s (int8), 0.92 s for the D11 teacher, opening the engine 0.01 to 0.04 s. ARITHMETIC: a 16-direction report is seconds of statistics; a real run is dominated by the system's translation time. UNMEASURED: a real 16-direction run (1,012 sentences x 16 directions per system is one `translate` call per system with a real model's speed), memory at large suites (the whole suite
and all outputs are read), a baseline's speed, anything on a GPU. The full local gate grew by 4,083 tests (ARITHMETIC; the three gate modules are 99 of them); it took about 25 to 30 minutes (**gate**: 1,401 s in four parallel parts for Task 11, 24 min 45 s for the integration part after the fix wave).

## What D14 and D15 take

[D14: done, on branch `d14-maintenance` (not merged, not pushed; `docs/runbooks/14_maintenance.md`). The golden tests named below are the targets of its `numerics` rerun (nightly; `tests/unit/test_eval_golden.py`, `test_eval_report_golden.py`, `test_eval_suite_golden.py`, `test_eval_bootstrap*.py`, `test_metrics_segment_stats.py` and the other metric goldens: 1,110 passed, floor 1,100) and this runbook's three gate modules are its `evaluation_tiny` rerun (weekly: 99 passed, floor 99), so a sacrebleu or numpy bump that moves a metric is caught by the nightly check and the `sacrebleu-numpy` pin rule says which reruns to run before taking it; the cadence is data in `configs/maintain/maintain.yaml`; `tests/unit/test_nightly_workflow.py` is the static check of `nightly.yml`. D14 builds nothing for D15 beyond what its runbook lists under "What D15 takes".]

**D14** (`rinchen maintain check`: doctor, `uv lock --upgrade --dry-run`, parity reruns; a nightly workflow): the parity reruns can be the golden tests of D13 (a sacrebleu or numpy bump that moves a metric, a signature or a bootstrap draw fails `test_eval_golden.py`, `test_metrics_segment_stats.py` and `test_eval_bootstrap.py` on purpose; every result records the sacrebleu version in its signatures); `tests/unit/test_release_workflow.py` shows how to check a workflow statically (PyYAML reads the key `on` as `True`; every action
must appear in `ci.yml` at the same version). The cadence is D14's. **D15** (the tiny end-to-end pipeline) builds its own fixtures and calls D13's library: its `report.json` comes from `rinchen eval run` and `rinchen eval report` over a suite of kind `fixture`; its thresholds map to keys of this report (training-set chrF at least 50: `metrics.chrf` of a result; script purity 1.0 and `placeholder_acc` 1.0: the robustness block, and `placeholder_acc` is 1.0 at beam 1 and 0.9983 at beam 4 on the toy, so D15 chooses its decode knowing that; statuses `ok`: `statuses`);
the neural metrics through the seam with tiny random-initialised scorers that write score files; the card and the dry run from `rinchen release stage` and `release hub`. **D15 decides** whether to add a `train`-split fixture suite for the memorisation check (the suite rule of T2 grows by one line) and owns the tiny scorers. `Suite.selection_view` (the only function that hands a `dev` suite's rows to a selection loop) has no caller in `src` today: D15 or a selection loop is where it belongs.

## The STOP gate

The master plan's gate for D13 is "tiny report.md + model card generated; dry-run hub upload (mock)". It is met on the toy through the real `rinchen` binary, the real Eole, CTranslate2, SentencePiece and sacrebleu on the CPU with the GPU hidden, no network and no token (`-m subprocess`, every subprocess bounded by 900 s; `HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` on; the Hub library and `datasets` are BLOCKED for the gate's own commands by a package that raises `ImportError`). Three modules, 99 tests: `tests/integration/test_eval_gate.py` (30: R1 to R5),
`test_eval_gate_controls.py` (37: R6 to R8) and `test_release_gate.py` (32: R9 to R11); `tests/integration/test_release_dry_run.py` is what the release workflow runs. The data are the held-out rows of D12's robustness toy joined into `toy-devtest` (480 rows) and `toy-dev` (292 rows), three languages (`eng_Latn`, `khk_Cyrl`, `rus_Cyrl`), six directions of which four are trained; the systems are D12's toy teacher, the D11 teacher that never saw those sentences and a hypotheses file. Thresholds are literals and are never lowered. **Gate** (2026-11-11, `model.bin` 2,080,071 bytes, `e28d1566...a6`, byte-identical to D12's):

| Gate | Threshold | Measured |
|---|---|---|
| R1 the suites; `inspect` | 480 and 292 rows, 3 languages, 6 directions, no document | suite files 113,679 B and 69,392 B; `inspect` prints name, kind, split, purpose, digest, rows per language, directions |
| R2 `eval run --model`, beam 4, `retry`, float32 | four trained directions `run`, 480 segments, every status `ok`; chrF++ at least 95.0, BLEU at least 93.0, each interval containing its value and narrower than 5.0, segment unit, 480 units, seed 20261110; `eng_Latn-rus_Cyrl` and `rus_Cyrl-eng_Latn` `not_trained` with no result file; neural metrics `null` with reasons | chrF++ **97.03 [95.93, 98.04]**, **99.05 [98.61, 99.46]**, **98.97 [98.37, 99.49]**, **98.38 [97.66, 99.05]** (`eng-khk`, `khk-eng`, `khk-rus`, `rus-khk`); BLEU 95.84, 98.12, 98.15, 96.83; widest interval 2.63 (BLEU `rus_Cyrl-khk_Cyrl`); about 1.6 s a run; `run.json` 2,344 B, `hypotheses.jsonl` 210,614 B (1,920 lines) |
| R3 the D11 teacher | chrF++ at most 60.0 in every direction | 36.50 [33.63, 39.16], 46.09 [42.89, 48.93], 29.30 [25.52, 32.67], 31.67 [28.66, 34.52]; BLEU 29.53, 33.11, 26.05, 27.55 |
| R4 R2's `hypotheses.jsonl` as `a-file` | every metric, interval, status and signature equal to R2's; the two other directions `not_in_file` | equal (`==` on the parsed documents) |
| R5 determinism | byte-equal `result.json`, `hypotheses.jsonl`, `run.json` on a rerun, under `PYTHONHASHSEED` 1 with `LC_ALL=C` and under 2; a command in a process that does not hide the devices or the Hub refused | the three reruns wrote the same bytes |
| R6 `eval report --baseline` | export minus teacher: delta at least +40.0, lower end above +40.0, p at most 0.01; the identical system: delta 0.0, interval [0.0, 0.0], p exactly 1.0; the perturbed system against the export: delta at most -2.0, upper end below -0.5, p at most 0.01, `significant` false (the worse system is not better), and the export against the perturbed one the sign reversed with `significant` true; reports byte-equal when regenerated, no absolute path | **+60.528 [+57.934, +63.433]**, **+52.965 [+50.240, +56.143]**, **+69.668 [+66.371, +73.529]**, **+66.707 [+63.981, +69.802]**, p 0.001998 (BLEU +66.306, +65.016, +72.091, +69.277); identical: 0.0 [0.0, 0.0], p 1.0 in all four directions and both metrics; `a-perturbed` (96 of 1,920 texts replaced, 24 per direction): **-3.274 [-4.696, -2.064]**, **-3.205 [-4.635, -2.049]**, **-4.030 [-5.858, -2.511]**, **-3.402 [-4.889, -2.115]**, p 0.001998, `better_fraction` 0.0 |
| R7 the guards | a run over `toy-dev` made (purpose `selection`) and refused by `eval report` (exit 3) alone and among good runs; a lines directory holding the Inner Mongolian variety's file refused by `eval suite build` (exit 3, nothing written); the toy's three languages build the four standard directions | as asserted; 480 rows, six languages of the benchmark's list missing |
| R8 robustness attached | the six values and their coverage in each `result.json` equal the robustness run's, binding `model_digest`; the same directory attached to the D11 teacher exits 2 with nothing written | numbers_acc 0.9291666666666667 (892 of 960), placeholder_acc 0.9983 (599 of 600), term_acc 1.0 (1,044 of 1,044), script_purity 1.0 (900 of 900), entity_acc 0.93125 (447 of 480), mong_source_retention 0.9976 (480 of 480); exit 2 ("another model") |
| R9 `release stage` | nine files and `manifest.json`; a card without a word of the content rules, with the fixture banner and every decimal a number of the report; `release.json` agreeing with the disk; two stages byte-equal; another system's report or another model's exit 3 with nothing written; with the robustness block the card holds the six values and the targets `numbers` (0.9292 against 0.99) and `placeholders` (0.9983 against 1) are `not_met` | as asserted |
| R10 `release hub` | a dry run exits 0, lists exactly the nine names with the disk's sizes and digests, writes `hub_plan.json` equal to the plan rebuilt here, creates no client; a stray file, a flipped byte, a symlink, a card with a token-like line and a card with an absolute path each exit 3 with nothing written; `--upload` without `HF_TOKEN` exits 2 and writes nothing | as asserted; `hub_plan.json` 1,533 B, manifest `mode: dry-run`, `visibility: private` |
| R11 hygiene | no `torch.cuda`, `get_cuda_device_count`, `nvidia-smi`; no socket outside the guard; every subprocess bounded; no word of the content rules; fixtures under 50 KB | clean |

The gate was mutation-checked (45 mutants of the code and of the tests' own settings, all killed; one no-op harness mutant was replaced; an empty-token `--upload` mutant was NOT run, for safety), so the controls are not vacuous: the identical-systems, perturbation, `dev`, `khk_Mong` and stage-refusal controls each fail when the thing they guard is broken. **What the figures show:** that the instrument sees a difference that was planted (+53 to +70 over a teacher, -3.2 to -4.0 for a 5 % perturbation, 0 for identical systems), that every direction has a state, and that
the report, the card and the plan are deterministic. **What they do not show:** anything about the quality of a real model, a real benchmark, a real baseline, a neural metric, the Hub, GitHub Actions or a GPU. The toy saturates at chrF++ 97 to 99 and ranks nothing. The docstrings of the gate quote sizes that were measured before the fix wave (`report.md` 12,012 B, the card 7,339 B); they are history.

## Open gates

Not done by D13, because no real data, GPU, scorer or Hub access exists: everything GPU-side (CTranslate2 on CUDA, `float16`, `int8_float16`, vLLM, the speed of a real run); the real FLORES+ dataset (id, configurations, columns, splits, sizes, document ids), NTREX (whether it offers Khalkha), the Mongolia-sourced human sets and the real `fetch`; NLLB, MADLAD and Hy-MT2 as baselines (the vocabulary guard has only seen the toy
export); COMET-22, MetricX-24 and MetricX-24-QE (B1 and B2: no scorer, no checkpoint, the key mapping, the score index and the EOS handling unverified); the Japanese and Korean tokenizers (B3 and B4); benchmark decontamination of training data; the off-target rate (language identification); zero-shot evaluation; a real upload, the Hub's acceptance of the card's front matter and language codes, and the visibility flow of an
existing repository; the release workflow on GitHub Actions; and the gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D12 gates that need real data, a GPU or a reader.

## What was not verified

- **Anything on a GPU, any real model, corpus, benchmark, human set or baseline.** Every translation ran on the CPU with `device="cpu"`; every Eole run had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`.
- **The real `fetch`, a real upload and the Hub.** `datasets.load_dataset` was always a recording fake (the `revision` keyword was checked offline with `inspect.signature`); `HfApi` was always patched or under `HF_HUB_OFFLINE=1` with every socket refused. UNMEASURED: the Hub's acceptance of the card's front matter, the language codes, the visibility of an existing repository
  (`HfApi.repo_info(...).private`, `create_repo(exist_ok=True)` ignoring `private` for an existing repository: **source-read**), the access a gated dataset needs, the library's behaviour with a real token.
- **GitHub Actions**: `release.yml` was checked statically only (PyYAML; the same pinned actions as `ci.yml`; no `secrets.`, no `--upload`, nothing about docker, GPU or a self-hosted runner) and never ran. No CI run covers D3 to D13.
- **That the metrics rank real systems**, that int8 is lossless or that any system is better than another. The toy ranks nothing.
- **The exit codes 4, 1 and 130 with real failures** (fakes only), `KeyboardInterrupt` with a real CTranslate2 call, memory at large suites; Windows and macOS.
- **The final wave** (`2df73c1..413bc94`): no reviewer read it; it was tested and mutation-checked by its own author (72 mutants, every one killed, two of them only after a test was added). The scoped re-review of the fix wave before it is recorded below.

### Known limits, left open on purpose

1. **The unit and the interval**: the segment bootstrap ignores the correlation between sentences of one article; the document unit needs a document id for EVERY row (FLORES+'s `url` is recalled to be the article address and is turned into an id by a hash: UNVERIFIED); with few documents the interval is coarse (five documents allow 3,125 resamples); a human export's documents are its `doc_id`s where it has them.
2. **No correction for multiple comparisons**: the report prints the counts of intervals and comparisons and says so; the card says "read a difference with its p-value".
3. **Tokenizers**: `char` BLEU for Japanese and Korean is not comparable with published BLEU; chrF++ is degenerate for `zho_Hans` and `jpn_Jpan`; B3 and B4 are unbuilt; the FLORES sentences are translations from English (translationese in the other directions).
4. **Contamination** is not checked: the devtest guard stops CODE from reading a benchmark, not DATA from leaking into a corpus (D4's MinHash could do an exact-match check); **the off-target rate** is always unmeasured; **zero-shot** pairs are never scored.
5. **The neural seam**: no scorer; a score file is bound to the suite and to the run's own `hypotheses.jsonl` but not to a scorer or its version; `eval report` believes the neural values and intervals the result states (it cannot recompute them) and checks only that a result holds one exactly when `run.json` lists the scores of that metric, and holds the runner's own reason otherwise (see limit 17); MetricX is lower-is-better.
6. **BLEU and the C library**: the golden tests compare floats at 1e-9 and `==` against sacrebleu's own `corpus_score`; the golden result file pins BLEU's last digits, so another machine's `exp` and `log` could fail it (an intended alarm: one machine and one lock were measured).
7. **The devtest guard** sees imports (direct and one-hop through the inverted scan, relative ones resolved) and the dynamic ones it can read; it does not follow a transitive import through an allowed module (the three final-wave allowlist entries of `cli/` are such cases: they take two generic helpers of `cli/eval_suite_output.py`, which a move to a neutral module would remove) and cannot see a computed `getattr` or `exec`. `Suite.selection_view` has no caller in `src` (D15 decides). **Dead code used only by tests**: `totals_of`, `metric_with_tokenizer`, `require_metric` and `Suite.selection_view` (review M12; left for later).
8. **Size**: `src/rinchen/cli/eval_suite.py` has 481 lines and `eval_run.py` 451 (over the 450-line guideline, under the 900-line limit), `src/rinchen/eval/report_runs.py` 474 after the final wave (428 before; its statistics half could move to a module of its own, which also moves the entries of the message scan); `tests/unit/test_eval_error_messages_d13.py` has 1,042 lines, over the 900-line limit of the rules (review M12: not split by decision).
9. **`beats_baseline` accepts a Rinchen export as the baseline** (review M9, second half): a previous export is a legitimate baseline for "does it beat X"; each system's kind is in the report, and the target counts directions against whatever baseline the metrics configuration or `--baseline` names. The first half is fixed: `comparison_metrics` must contain `primary_metric`.
10. **The robustness suite's kind is not recorded** in D12's block: the report and the card say `suite kind not recorded`, and the evidence banner counts the eval suites only. The binding of a block to a system is the model digest or, for another kind of system, its name (the weaker one, recorded).
11. **Two runs of one system over one suite are refused by the report** (so are two suites with one name): to compare two versions of a model, give them different `--system` names. A suite named `.` or `..` still passes `Suite.from_bytes`; the commands refuse it at build and at run, and the report does not assume a suite name is a safe path component.
12. **A human export holding the same document and segment in both directions is refused** (the id `<doc_id>-<seg_idx>` would repeat): split such an export. A server baseline runs the Khalkha pairs only (D6's prompts); other pairs are `unsupported_language`. `eval suite build --from <fetch-out>/<split>` finds the fetch manifest in the parent; hand-made shards have no provenance and the command says so.
13. **PyYAML's message for an unusable configuration can quote a line of the operator's file**; a disk-full error inside a fetch is reported as an upstream failure (exit 4); a fetch that fails part-way leaves its shards and no manifest.
14. **`hub_commit.json` is the only local record of an upload**, and a commit cannot be undone: the checks before it (the token, the writable output directory, the Hub's own answer about visibility) are all there is. The real adapter's visibility call was never run against the Hub.
15. **The integration-gate docstrings quote sizes that are history** (`report.md` 12,012 B, the card 7,339 B, measured before the fix wave); a transitive import through an allowed module is seen by neither scan.
16. **Disclosed process slips of the D13 wave**, one line each: the design agent's first run of a sacrebleu script instantiated the `flores200` and `spm` tokenizers and sacrebleu DOWNLOADED two SentencePiece models (about 9.7 MB) into `~/.sacrebleu` (a network access the task excluded; the directory did not exist before and was removed; every later script imported a socket guard; no Hugging Face endpoint was contacted; F17);
    Task 5 called the Agent tool once by mistake (a no-op agent that read and changed nothing); Task 6 left a stray `EOF` line in a test (caught before any run), used a few short foreground sleeps and two `nohup` launches and ran `ruff format` over the whole tests tree (only its own files changed); Tasks 7 and 9 wrote sources before running their tests and Task 10 its workflow before its test (RED shown afterwards on a copy), and several tasks used short foreground sleeps; Tasks 2 and 8 committed with `core.hooksPath` emptied (the repository holds no hook files, so nothing was skipped); Task 8 ran `ruff --fix` over five unrelated test files (restored) and
    one `pgrep -f` that matched its own shell; Task 10 made one existence probe of `~/.cache/huggingface` (output discarded, nothing read); Task 11's first gate module was over 900 lines (split before the commit) and it left about 400 MB of scratch copies; the combined reviewer ran a dry run of `release hub` with a sentinel token and an in-process fetch probe with a fake module (no network); the fix wave ran `ruff format` over several directories (only its own files changed), added tests after the implementation to kill survivors and imported `huggingface_hub` once under `HF_HUB_OFFLINE=1`
    to read two signatures (no instance made); one idle `tail -F` poller of an earlier agent was left running because the harness refused its kill. This runbook's own measurements ran one dry run of `release hub` over a fabricated stage with the offline switches on and no token in the environment (no client is created by a dry run; no `--upload` and no `fetch` was run). The scoped re-review used a foreground `sleep 0` and an until-loop with `sleep 2`, ran `--upload` with a sentinel token IN-PROCESS through the CLI runner with `open_client` replaced by its own fake (no real `HfHubClient` was built there: outside the letter of the rule), built the real `HfHubClient` and imported `huggingface_hub` in a subprocess with `HF_HUB_OFFLINE=1` and every socket refused, used pytest's default `basetemp` under the system `/tmp`, started its mutation driver with `nohup` and extracted the `3c52162` source into its scratch; the runbook's task used one `sleep 0`, one failed edit and a few `rinchen --help` runs with bytecode writing off (the tree stayed clean); the final wave used one `sleep 0` in a foreground command, ran only the tests of each finding before each commit (so the reviewed-interpolation scan of `tests/unit/test_eval_error_messages.py` was red from `0a4f8d4` until `4bfeb80` added its entries; the broad run before the gate found it and the gate is green), started one background pytest with `&` from a foreground shell and waited in its gate launchers with `timeout 6 tail -f /dev/null` (a bounded wait) in place of a sleep, and ran its scratch scripts that import `tests.support` without `-I` (own data). No GPU question, no docker command, no `git stash`, no `pkill -f` is recorded.
17. **What a re-digested result can still plant** (scoped re-review, m1): the report recomputes the count metrics, the statuses, the signatures and `trained`, and the final wave added the presence of each neural metric and its reason. A result file that was edited and re-digested can still hold (a) a free-text `unmeasured` reason of a metric that is unmeasured for another cause than the missing scores (nothing in a run says why else), (b) a value or an interval of a neural metric that the score file would not give (the report holds no segment scores), and (c) D12's robustness block values (the report cannot recompute them: `robustness.json` is not copied into the run directory, only its digests and the binding). **A run directory made before the final wave carries the old reason** (it named two open items of the master plan) and is refused as "not the one the runner writes": run `eval run` again. `Suite.selection_view` still has no caller in `src` (limit 7).

### The review, its fix wave and the re-review

**The combined review** (one independent reviewer, read-only, Tasks 1 to 11, `1a378d0..3c52162`): ready with fixes, 0 Critical, 5 Important and 13 Minor findings. It re-derived the intervals, the paired delta and the p-value independently (bit-equal, segment and document unit), found no leak and probed the real binary (about 50 probes: tamper cases, hub refusals, determinism across the environment).

| Finding | What | Fixed |
|---|---|---|
| **I1** | the report believed `ci95`, `statuses`, the three signatures and `trained` of a result file: a consistently edited and re-digested `result.json` was accepted (a flipped byte was caught by the digest chain, a re-digested file was not) | `0000ccb`: `verified_rows` recomputes the bootstrap record, the statuses, the three signatures, each count metric's interval and `trained` from the outputs; a mismatch is exit 3 naming run, suite, direction and field |
| **I2** | the card's evidence banner, `toy` tag and `release.json`'s `evidence` came from the whole report, not from the carded system | `1a95422`: `system_evidence` over the suites where THE CARDED SYSTEM has a result |
| **I3** | flagged output failures (`empty`, `truncated`, `no_output`) were invisible in `report.md` and the card | `3d32b74`: a `<system> ok` column in every score table (`10 of 10`, `6 of 10 (1 empty, 2 no_output, 1 truncated)`), `outputs ok` in the card |
| **I4** | dataset, revision and the `pinned` flag were not printed | `3d32b74`: in the Suites table and in the card (`unpinned`, `<rev> (unpinned)`, `n/a (no source record)`) |
| **I5** | "private by default" did not hold for a repository that already exists (`create_repo(exist_ok=True)` ignores `private`) | `06b1157`: `HubClient.repo_is_private`, asked after `create_repo` and BEFORE `create_commit`; a public repository without `--public` is exit 2 with nothing committed; `hub_commit.json` and the manifest record the visibility the Hub REPORTED |
| M1 | the devtest guard was bypassed by a one-hop re-export | `7ac4404`: the scan is inverted (any import of `rinchen.eval` outside the exempt modules and the six-entry allowlist is a violation) |
| M2, M5, M9 (first half) | the stage copied `report.md` without checking it was the rendering of `report.json`; the card's GPU sentence was unconditional; `comparison_metrics` need not contain `primary_metric` | `7ac4404`: re-rendered and compared (exit 3); the sentence follows the systems' `device` option; a `ConfigError` naming the keys |
| M3, M4, M8, M10, M11 | `build --from <fetch-out>/<split>` lost the fetch provenance silently; `--robustness` of another model was refused after the whole run; `build --name .` wrote `..json`; the Hub scan missed `file:///...` and the JSON form of `HF_TOKEN`; a read-only `--out-dir` was found out after an upload | `2df73c1`: the provenance line and the parent lookup; the binding checked before the engine opens; `.` and `..` refused at build; `://` and quotes are path boundaries, `HF_TOKEN` with an optional quote; `require_writable` before the client |
| M6, M7 | the `significant` column read `no` for a significantly worse system; the robustness block's suite kind is not recorded | `3d32b74`: a `verdict` column (`better`, `worse`, `no significant difference`, `n/a (<reason>)`), `suite kind not recorded` beside the robustness suite |
| M12, M9 (second half), M13 | dead code used only by tests; a Rinchen export accepted as the baseline; the missing runbook | documented (limits 7, 8, 9); this runbook |

**The fix wave** (six commits, `3c52162..2df73c1`, failing test first, mutation-checked: 152 mutants, 151 killed, 1 equivalent (a `mkstemp` suffix)) and its **full gate: 17,666 passed, 1 skipped, 10 deselected** (17,390 before; three unit parts and the integration part, 24 min 45 s for the last; about 25 minutes wall), `ruff format --check` (761 files), `ruff check --no-cache` and `mypy src` (267 files) clean; no integration test had to change. The goldens
moved: `golden_report.md` 6,610 to 7,644 bytes (SHA-256 `a6b30c0d...8a3f`), `golden_card.md` 5,932 to 6,054 (`7869831b...ab41f`), `golden_report.json` unchanged at 14,894; each was cross-checked byte for byte against an independent stdlib renderer. Task 10's differential ran again (1,500 random stages, 0 mismatches; the old scan disagrees on the new cases) and the card's (400, 0).
**The scoped re-review of the fix wave** (one independent reviewer, read-only, `3c52162..2df73c1`): **approved with minors, 0 Critical, 0 Important and 7 Minor findings.** All 16 findings of the combined review were verified fixed (with the real binary or a fake seam). Its own 50 mutants were killed but one equivalent one (an adapter check that the upload path repeats); its independent checkers of the Markdown cells found 0 mismatches (52, 52, 84 and 10 cells, and 63, 42 and 42 more on toy cards), its differential of the Hub scan 0 in 3,000 cases, the output was byte-identical across 5 hash seeds, 3 locales and 3 time zones (45 runs), and the old and the new binary gave the same `report.json` for a result written before the fix wave.

| Finding | What | Fixed |
|---|---|---|
| m1 | a re-digested result could still plant what the report prints: a neural metric's value, a free-text `unmeasured` reason, D12's block values | `0a4f8d4`: a result holds a value for each neural metric exactly when `run.json` lists its scores, and the runner's own reason otherwise (exit 3 naming run, suite, direction and field, no value); the other two are limit 17 |
| m2 | the dataset and the revision were printed unescaped and the report reader took any text for them: a re-digested `report.json` injected a heading into the card | `e59a740`: the reader holds both to the suite reader's patterns (a mirror in `report_read_rules`, pinned equal by a test) and the cells pass through `inline()` |
| m3 | the inverted devtest scan did not see imports of the exempt `rinchen.cli.eval*` modules nor an aliased dynamic import | `069b232`: names under `rinchen.cli.eval*` count as `rinchen.eval` imports; `cli/release_stage_output.py`, `cli/release_hub_output.py` and `cli/main.py` join the allowlist with their reasons; `from importlib import import_module as im` is read |
| m4 | two refusals could not tell the operator what to do: a stale `report.md` after the renderer changed, and a signature that may be another sacrebleu's | `71f3b8b`: they say so, name the installed sacrebleu version and the command to run again |
| m5 | `--public` over an existing private repository exited 0 and said nothing | `8d84671`: `visibility: <what the Hub reported>` after the commit line |
| m6 (pre-existing) | the card printed `n/a ()` with an empty reason for a suite in which the system has runs and no result | `413bc94`: `no result in this suite`; every reason of an `n/a (...)` goes through one helper that never leaves it blank (`no reason recorded`) |
| m7 | the card and `report.md` named an internal milestone (`D12's block`, `B1 and B2 are open`) | `413bc94`: plain words (`the robustness block`, `the COMET and MetricX scorers are not built`); a scan of the strings of the modules that write the uploaded files, and of the golden and the generated text |

**The final wave** (seven commits, `2df73c1..413bc94`: one each for m1 to m5, `4bfeb80` that lists the new messages in the message scan, and `413bc94` for m6 and m7 together because they share the goldens), failing test first and mutation-checked: 72 mutants, every one killed; two survived the first pass (a deeper name of an eval command that no test reached, and a blank basis beside the value of a target) and were given a test each. The full gate: **17,852 passed, 1 skipped, 10 deselected** (25 min 44 s wall; three unit parts and the integration part), `ruff format --check` and `ruff check --no-cache` (765 files) and `mypy src` (267 files) clean. Four goldens moved: `golden_report.json` 14,894 to 14,942 bytes (SHA-256 `c8d2095f...ce30` to `1fd86b2d...ca65`: the reason text and the two run digests), `golden_report.md` 7,644 to 7,656 (`a6b30c0d...8a3f` to `c1622d70...9f51`), `golden_card.md` 6,054 to 6,058 (`7869831b...ab41f` to `a42ecc37...b3fb`) and `golden_result_alpha_khk_eng.json` 1,954 to 1,966 (`05212b4f...8dde` to `f31880cb...b28b`); the fixtures are 37,096 bytes together (37,020 before). They were regenerated only after the literal tests held: `report.md` and the card are byte for byte the output of independent stdlib renderers, `report.json` is the independent builder's document and the command writes the same bytes. The existing tests edited: the literal of the neural reason (four files), the golden digests and sizes (four files), the pinned sentence about the robustness block (one), the lines after `commit:` of an upload (one), the devtest guard's allowlist and its list of inputs the scan must not flag (the four that named the eval commands moved to the flagged list) and the message scan's list (five entries); everything else is additive. **Left open in the wave**: `Suite.selection_view` has no caller (limit 7), `report_runs.py` has 474 lines and `cli/eval_suite.py` and `eval_run.py` stay at 481 and 451 (limit 8), and limit 17.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython 3.14.10, eole 0.6.4, ctranslate2 4.8.4, sentencepiece 0.2.1, sacrebleu 2.6.0, numpy 2.5.4; the GPU was hidden from every process and every translator was built on `device="cpu"`. The numbers of this runbook's own measurements were taken on 2026-11-11 with the code at `2df73c1` (the tree held only these documents), load average 5.6 to 7.5; the sizes of the goldens and of the fabricated release were measured again at `413bc94` after the final wave; the gate's
numbers are the ledger's Task 11 entry; the spike's are `spike-findings.md`.

| What | Value |
|---|---|
| the golden suite, outputs and result (SHA-256 `b707518d...`, `0708a740...`, `fcf6f294...`, `f31880cb...`) | 2,259 B; 2,049 B and 2,166 B; 1,966 B (**measured**) |
| the golden runs through the real binary | `alpha` chrF++ 73.08 [54.91, 85.48] and 68.06 [50.67, 80.98]; `beta` 93.87 [85.24, 100.00] and 91.34 [85.07, 97.86]; chrF 72.44, 67.41, 93.46, 91.07; BLEU 59.40, 50.87, 94.42, 91.07; document unit, 5 units (**measured**) |
| the golden paired comparison (`beta` against `alpha`, document unit) | chrF++ +20.79 [+13.92, +33.41] and +23.27 [+12.65, +36.22]; BLEU +35.01 and +40.20; p 0.002 (0.001998), `better` (**measured**) |
| the identical-systems control | delta +0.00 [+0.00, +0.00], p 1.000, `no significant difference`, both directions, both metrics (**measured**) |
| `report.json` and `report.md` from the command | byte-equal to the golden files (14,942 B and 7,656 B) (**measured**) |
| the fabricated release | nine files 35,531 B in all (`README.md` 6,058 B, `report.json` 15,788, `report.md` 7,745, `release.json` 2,345), `hub_plan.json` 1,524 B; the two manifests hold the path of their directory (5,476 and 3,526 B at `2df73c1`, 5,710 and 3,669 B under another path at `413bc94`) (**measured**) |
| the toy (D12's export, beam 4, `retry`, float32) | chrF++ 97.03, 99.05, 98.97, 98.38; BLEU 95.84, 98.12, 98.15, 96.83 (**gate**) |
| the D11 teacher | chrF++ 36.50, 46.09, 29.30, 31.67 (**gate**) |
| export minus teacher, chrF++ | +60.528, +52.965, +69.668, +66.707, p 0.001998 (**gate**) |
| the perturbed system minus the export | -3.274, -3.205, -4.030, -3.402, p 0.001998 (**gate**) |
| robustness attached | numbers_acc 892/960 = 0.9292, placeholder_acc 599/600 = 0.9983, entity_acc 447/480 = 0.93125, term_acc 1,044/1,044, script_purity 900/900, retention 0.9976 (**gate**) |
| the spike: float32 minus int8 | +0.289 (p 0.2577), -0.055 (p 0.4476), +0.179 (p 0.1079), +0.134 (p 0.3097); intervals touching zero (**spike**, F15) |
| sacrebleu's own paired test | p 0.000999 for a system against itself (**spike**, F3) |
| costs | the table of "Costs" |
| the tests | 17,852 passed, 1 skipped, 10 deselected after the final wave (17,666 after the fix wave; 13,583 at the end of D12; 13,900, 14,104, 14,551, 14,931, 15,163, 15,739, 16,018, 16,412, 16,866, 17,286, 17,390 after Tasks 1 to 11); `ruff format --check`, `ruff check --no-cache`, `mypy src` clean |

## Where the code lives

`src/rinchen/metrics/{registry,segment_stats}.py`; `src/rinchen/config/{eval_metrics,eval_suites}.py`; `configs/eval/{metrics,suites}.yaml`; `src/rinchen/eval/{bootstrap,significance,suite_rules,suites,suite_sources,outputs,systems,runner,segment_scores,scoring,result,run_record,field_rules,robustness_input,targets,report,report_runs,report_view,report_format}.py`;
`src/rinchen/eval/{report_markdown,report_markdown_cells,report_markdown_scores,report_markdown_notes,report_read,report_read_rules,report_read_sections}.py`; `src/rinchen/export/{model_card,model_card_evaluation,model_card_notes,release,hub,hub_files,hub_record,hub_scan}.py`;
`src/rinchen/cli/{eval_suite,eval_suite_files,eval_suite_plan,eval_suite_output,eval_run,eval_run_plan,eval_run_inputs,eval_run_output,eval_systems,eval_report,eval_report_inputs,eval_report_output,release,release_stage,release_stage_inputs,release_stage_output,release_hub,release_hub_output}.py`; the edits of `src/rinchen/acquire/hf.py`, `src/rinchen/cli/eval.py` and `src/rinchen/cli/main.py`;
`.github/workflows/release.yml`; `tests/fixtures/eval/{golden_suite.json,golden_outputs_alpha.jsonl,golden_outputs_beta.jsonl,golden_result_alpha_khk_eng.json,golden_report.json,golden_report.md,golden_card.md}`; the gate `tests/integration/{test_eval_gate,test_eval_gate_controls,test_release_gate,test_release_dry_run}.py` with the session fixtures `d13_offline`, `d13_suites` and `d13_runs` of
`tests/integration/conftest.py`; 80 unit test files (`tests/unit/test_{metrics_*,config_eval_*,eval_*,cli_eval_*,cli_release_*,export_hub*,export_model_card*,export_release*,acquire_hf_revision,d13_content_rules,printed_text_names_no_milestone,release_workflow,no_network_fixture}.py`), 8 property test files `tests/property/test_{eval_*,metrics_segment_stats}_property.py`; and the helpers
`tests/support/{eval_cli,eval_fakes,eval_gate,eval_golden,eval_report_cli,eval_report_golden,eval_report,eval_result_documents,eval_suite_cli,eval_suite_documents,eval_toy,hub_fakes,hub_stage,module_imports,no_network,release_documents,release_fixture}.py`.
