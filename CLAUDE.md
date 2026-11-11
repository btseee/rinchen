# CLAUDE.md

Read `STATE.md` first for current phase status.

## Hard constraints

ALWAYS:
- Run everything on CPython 3.14.x. Check `python -c "import sys; assert sys.version_info[:2]==(3,14)"` at the start of every script entrypoint.
- Use uv; one workspace; extra envs only for real dependency conflicts, all on 3.14.
- Train every Rinchen model from random init.
- Use position_encoding_type: SinusoidalInterleaved in every Eole model.
- Record every resolved package version and checkpoint hash in manifests/.
- Write tests before code; numeric acceptance thresholds are gates.
- Keep state in files (STATE.md, DECISIONS.md, manifests/), not in chat memory.

NEVER:
- Use Python 3.12/3.13 or any fallback env, container, or conda env with another Python.
- Fine-tune, LoRA, or continue-train any pretrained model. External models only as evaluation baselines or synthetic-data generators whose outputs become training text.
- Use Inner Mongolian data, varieties, datasets, tools, or FLORES+ khk_Mong.
- Use galig/Latin informal Mongolian.
- Write anything about licensing or legal topics.
- Use Relative, rotary, ALiBi, sliding window, or GQA in Eole models meant for CT2.
- Tune on FLORES+ devtest. Use FLORES+ dev for model selection only; report devtest.

## Commands

- `just gate` (ruff format --check, ruff check, mypy, pytest -m "not network" -q -- the full local gate CI also runs)
- `uv run pytest -q`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run rinchen doctor` (or `just doctor`)
- `uv run rinchen eval robustness --suite FILE (--model DIR | --hypotheses FILE) --out-dir DIR` (D12: measures numbers, placeholders, names, glossary terms, script purity and a traditional-script source; no threshold, exit 0 whenever it measured; `docs/runbooks/12_robustness.md`)
- `uv run rinchen eval suite fetch|build|inspect` (D13: `fetch --benchmark B --split S --out-dir DIR (--revision HEX | --unpinned)` is the ONLY network step and was never run; `build --benchmark B|human --from PATH --out-dir DIR` turns local files into a suite file and refuses `khk_Mong` by name; `inspect --suite FILE`)
- `uv run rinchen eval run --suite FILE --out-dir DIR (--model DIR | --generator NAME | --hypotheses FILE)` and `uv run rinchen eval report --run DIR ... --suite FILE ... --out-dir DIR [--baseline SYSTEM]` (D13: BLEU, chrF and chrF++ with 95 % bootstrap intervals, paired comparisons, `report.json` and `report.md`; they measure and exit 0 whenever they measured; `docs/runbooks/13_release.md`)
- `uv run rinchen release stage --export DIR --report-dir DIR --system NAME --repo-id ORG/NAME --out-dir DIR` and `uv run rinchen release hub --stage DIR --repo-id ORG/NAME --out-dir DIR [--upload] [--public]` (D13: the staging directory with the model card; `hub` is a DRY RUN unless `--upload`, which needs `HF_TOKEN` in the environment, creates the repository private and was never run)
- `uv run --frozen --no-sync rinchen maintain check --out-dir DIR [--cadence nightly|weekly] [--no-drift] [--no-reruns] [--rerun ID] [--fail-on-drift] [--export DIR --checkpoint DIR --corpus FILE]` (D14: the doctor's CPU checks, `uv lock --check`, the installed packages against `uv.lock`, the dependency drift table from `uv lock --upgrade --dry-run` (the ONLY step that queries an index: never run it for a test or a measurement) and the parity reruns; writes `maintain_report.json`, `maintain_report.md` and `manifest.json`, only reads the project, exit 0 for ok or drift and 4 for fail; the upgrade procedure and the rollback are `docs/runbooks/14_maintenance.md`)

## Repo layout

```
CLAUDE.md STATE.md DECISIONS.md pyproject.toml uv.lock justfile
.pre-commit-config.yaml
configs/base.yaml configs/profiles/{tiny,full}.yaml configs/{spm,eole,serve}/
configs/languages.yaml configs/spm/{tiny_1k,student_64k,teacher_128k}.yaml
configs/data/sources.yaml configs/data/sources.tiny.yaml configs/data/filters.yaml
configs/data/synth.yaml
configs/eole/converter{,_tiny}.yaml configs/spm/converter_char.yaml
configs/eole/{multilingual_tiny,ablation_tiny,multilingual_proxy,ablation_priors}.yaml
configs/eole/{teacher,teacher_tiny}.yaml configs/eole/{teacher_doc,teacher_doc_tiny}.yaml
configs/eole/student_{server,edge,tiny}.yaml configs/serve/{tiny,server,edge}.yaml
configs/eval/{metrics,suites}.yaml configs/maintain/maintain.yaml
docker/{Dockerfile.cpu,compose.serve.yaml} .dockerignore
docs/STACK.md docs/VERIFY.md
docs/runbooks/00_environment.md docs/runbooks/01_acquisition.md
docs/runbooks/02_human_testsets.md docs/runbooks/03_tokenizer.md
docs/runbooks/04_filtering_mining.md docs/runbooks/05_script_converter.md
docs/runbooks/06_synthetic.md docs/runbooks/07_ablations.md docs/runbooks/08_teacher.md
docs/runbooks/09_document_level.md docs/runbooks/10_distillation.md
docs/runbooks/11_serving.md docs/runbooks/12_robustness.md docs/runbooks/13_release.md
docs/runbooks/14_maintenance.md
docs/research/  docs/superpowers/{specs,plans}/
src/rinchen/{config,manifest,utils,cli,text,lang,split,spm,corpus,filter,mine,script_conv,synth,metrics,eval,serve,export,maintain}/
src/rinchen/acquire/{http,opus,hf,warc,commoncrawl,registry}.py
src/rinchen/annotate/{db,export,agreement,app,routes,models}.py
src/rinchen/spm/{symbols,corpus,train,check}.py src/rinchen/split/rules.py
src/rinchen/text/{normalize,numerals}.py src/rinchen/lang/scripts.py
src/rinchen/corpus/{schema,readers,writers}.py src/rinchen/config/filters.py
src/rinchen/filter/{reasons,script_ratio,numbers,length,lid,minhash,labse,qe,qe_eole,pipeline}.py
src/rinchen/mine/{embed,align,docalign,pairs}.py src/rinchen/cli/{filter,mine,errors}.py
src/rinchen/script_conv/{alphabet,validate,segments,eval,fallback,rules,data,char_model,config,train,export,runtime}.py
src/rinchen/script_conv/{lexicon,standardized_variants}.tsv (package data)
src/rinchen/cli/convert_script.py
src/rinchen/mbr/{utility,mbr}.py src/rinchen/config/synth.py src/rinchen/cli/synth.py
src/rinchen/synth/{tags,translator,prompts,_chat_completions,vllm_client,llamacpp_client}.py
src/rinchen/synth/{ct2_baselines,_candidates,_texts,shards,select,plan,bt,ft,run}.py
src/rinchen/train/{eole_cfg,eole_run,sampling,variants,mixture,corpus_files,eole_data,validation_corpus,evaluate}.py
src/rinchen/train/{ablation_plan,ablation_config,ablation_layout,ablation_prepare,ablation,ablation_records,ablation_report}.py
src/rinchen/train/{checkpoints,curves,average,launch,gpu_rules,launch_command,run_lock}.py
src/rinchen/train/{docify,doc_source,doc_pack,doc_corpus,continuation,vocabulary_size,eole_cfg_keys}.py
src/rinchen/export/{eole_ct2,record,verified,quantize,ct2}.py
src/rinchen/infer/{doc_chunker,source_tokens,script_mask,engine_types,engine,translator}.py
src/rinchen/serve/{schemas,metrics,batcher,engine,_http,_translate,app}.py src/rinchen/config/serve.py
src/rinchen/eval/{robust_numbers,robust_placeholders,robust_spans,robust_script,robust_mong,robust_suite,robust_suite_rules}.py
src/rinchen/eval/{robust_hypotheses,robust_outcomes,robust_sections,robustness,robust_report}.py
src/rinchen/metrics/{registry,segment_stats}.py src/rinchen/config/eval_{metrics,suites}.py
src/rinchen/eval/{bootstrap,significance,suite_rules,suites,suite_sources,outputs,systems,runner}.py
src/rinchen/eval/{segment_scores,scoring,result,run_record,field_rules,robustness_input,targets}.py
src/rinchen/eval/{report,report_runs,report_view,report_format,report_markdown,report_markdown_cells}.py
src/rinchen/eval/{report_markdown_scores,report_markdown_notes,report_read,report_read_rules,report_read_sections}.py
src/rinchen/export/{model_card,model_card_evaluation,model_card_notes,release}.py
src/rinchen/export/{hub,hub_files,hub_record,hub_scan}.py
src/rinchen/train/{distill_export,distill_saved_config,distill_teacher,distill_sources}.py src/rinchen/train/export_parity.py
src/rinchen/train/{distill_shards,distill_generate,distill_corpus,distill_mix,distill_prepare}.py
src/rinchen/config/ablation.py src/rinchen/cli/{train,train_runs,train_common,train_docify}.py
src/rinchen/cli/{distill,distill_export,distill_generate,distill_generate_plan,distill_generate_records}.py
src/rinchen/cli/{distill_prepare,distill_evaluate}.py src/rinchen/cli/{export,export_parity,translate,translate_io,serve}.py
src/rinchen/cli/{eval,eval_robustness,eval_robustness_plan,eval_robustness_output}.py
src/rinchen/cli/{eval_suite,eval_suite_files,eval_suite_plan,eval_suite_output}.py
src/rinchen/cli/{eval_run,eval_run_plan,eval_run_inputs,eval_run_output,eval_systems}.py
src/rinchen/cli/{eval_report,eval_report_inputs,eval_report_output}.py
src/rinchen/cli/{release,release_stage,release_stage_inputs,release_stage_output,release_hub,release_hub_output}.py
src/rinchen/maintain/{states,lock,environment,uv_output,drift,commands,reruns,report,report_markdown,check,check_types,check_steps,check_reruns}.py
src/rinchen/config/maintain.py src/rinchen/cli/{maintain,maintain_inputs,maintain_doctor,maintain_output,maintain_signals}.py
src/rinchen/utils/{logged_process,integers}.py
src/rinchen/errors.py src/rinchen/__init__.py
tests/ scripts/ plans/ reports/ manifests/
manifests/ucd_standardized_variants.json
tests/{unit,property,golden,integration}/ tests/fixtures/{annot,golden,raw,spm,script_conv,robustness,eval,maintain}/
tests/support/ (planted corpus and alignments, fake embedders, tiny LID model,
  tiny_converter, stub_runtime, fake_process, script_reference, planted_synth,
  synth_cli, fake_llm_server, shard_files, corpus_rows, eole_train_config, tiny_parallel,
  tiny_variant, train_cli, fake_ablation, step_dirs, fake_launch, fake_eole_process, launch_cli,
  signal_handlers, doc_cli, doc_rows, distill_cli, distill_exports, distill_gate, distill_gate_refusals,
  distill_gate_stages, distill_teachers, fake_ct2, planted_kd, planted_student, model_exports, engines,
  error_messages, translate_cli, parity_cli, fake_parity_eole, serve_app, serve_batcher, fake_serve_process,
  docker_files, d11_gate, d11_gate_doc, serving_gate, serving_gate_fixtures, robust_toy, robust_toy_text,
  robust_suites, robust_gate, robust_golden, robust_cli, robust_suite_documents, no_network, module_imports,
  eval_golden, eval_fakes, eval_cli, eval_suite_cli, eval_suite_documents, eval_result_documents, eval_report,
  eval_report_cli, eval_report_golden, eval_toy, eval_gate, release_documents, release_fixture, hub_fakes, hub_stage,
  maintain_check, maintain_cli, maintain_fakes, maintain_gate, maintain_gate_hygiene, maintain_project, maintain_reports,
  maintain_scans, uv_outputs, fake_uv, nightly_workflow, workflow_expression)
scripts/build_spm_fixture.py (regenerates tests/fixtures/spm/tiny_corpus)
scripts/build_script_conv_fixtures.py (reference TSV -> lexicon, golden words, tiny fixture)
data/{raw,clean,mined,synth,test}/ (gitignored content, tracked dirs)
vendor/wheels/ (committed binary wheels with no cp314 upstream release)
vendor/fasttext-wheel-shim/ (pure-Python shim package, see DECISIONS.md)
.github/workflows/{ci,release,nightly}.yml
```

Note: `data/`, `align/`, and `convert/` under `src/rinchen/` from the old
Phase 0-14 plan have no home in the new D0-D15 repo layout and were removed;
D1 built `acquire/` there instead of recreating `data/`. No `corpus/`
module was needed because D1's actual scope (per the master plan) is
acquisition only -- fetching/listing raw sources into `data/raw/` -- not
corpus construction, cleaning, or alignment, which are later milestones.

## Current phase

Master plan switched 2026-10-29 to milestones D0-D15 (see below). D0, D1,
and D2 are all done, all STOP gates fully met including green CI runs on
`origin/main` (2026-11-02). D3 (Normalization & SentencePiece) is done: its
STOP gate was met locally (2026-11-06), but only `tiny_1k` was ever trained
(`student_64k`/`teacher_128k` are schema-validated only). D4 (Filtering &
mining) is done: its STOP gate was met locally (2026-11-07) and branch
`d4-filtering-mining` was fast-forwarded into `main` (unpushed); every threshold
is an unvalidated prior because no real data exists, and the D3 gates assigned to
D4 (abbreviation induction, o-like substitute counts, the traditional-script
control-character audit) stay open. `main` is unpushed and no CI run covers D3,
D4 or D5. D5 (Cyrillic/traditional converter) is done locally: branch
`d5-script-converter` (cut from `main` at `4eac98f`) was fast-forwarded into `main`
(unpushed); its STOP gate (the tiny converter exported and served through
`ScriptConverter`) is met on the tiny fixture. It is
a scaffold, not a quality converter: the rule baseline spells 237 of 1,022
held-out dictionary words exactly, the tiny model memorises a 500-word fixture,
the full profile was never trained, and the gates that need real script text or a
Mongolian reader (the control-character audit, the render check, the review of the
lexicon and the fallback) stay open. D6 (Synthetic data) is done locally on branch
`d6-synthetic-data` (cut from `main` at `0d534a3`), not merged and not pushed, and no CI run
covers it: its STOP gate (a tiny synth run through the CLI with a fake generator) is met on
planted data. No generator, server, model or GPU ever ran (every client was tested against
fakes), no real monolingual corpus exists, and the gates that need a real generator, real data
or a Mongolian reader stay open. D7 (Multilingual ablations) is done locally on branch
`d7-multilingual-ablations`, stacked on `d6-synthetic-data` (cut from it at `9973319`), not merged
and not pushed, and no CI run covers it: its STOP gate (a tiny ablation of 2 variants trains and is
evaluated through the real `rinchen train ablate` on the CPU) is met on toy templated data that
saturates at chrF 96 to 99 and ranks nothing. No real corpus, no GPU run and no measured proxy
config exist (`configs/eole/multilingual_proxy.yaml` and `ablation_priors.yaml` are schema-validated
priors only); the gates that need them stay open. D8 (Teacher training) is done locally on branch
`d8-teacher-training`, stacked on `d7-multilingual-ablations` (cut from it at `b1a8a47`), not merged and
not pushed, and no CI run covers it: its STOP gate (a tiny teacher trains 200 CPU steps without NaN, the
loss falls by at least 30 %, it resumes after a damaged checkpoint, completes and is averaged, through
the real `rinchen train launch` and `average` and the real Eole) is met on toy templated data. It is
machinery, not a teacher: no GPU was used (the one on this machine was not authorised), no real corpus
exists, and `configs/eole/teacher.yaml` is a table of unmeasured priors; everything on a GPU (bf16,
NCCL, multi-GPU, the 13 GB checkpoint write) and the quality of the teacher shape stay open. The
whole-branch review of D8 ran: ready with fixes, no Critical finding, two Important ones (the process group of a
crashed Eole leader was not killed, so the rank processes of a multi-GPU run could train on unlocked; the TensorBoard
events landed outside the run directory) and seven Minor ones; the fix wave (two commits, gate 7187 passed, 1 skipped,
10 deselected) fixed the two Important ones, a guard against a resumed checkpoint of another architecture and the
wording of an unreadable-file message, and left four findings open on purpose (`DECISIONS.md`); a scoped re-review of
the fix wave approved it with minors, and the one that needed code is fixed. D9 (Document-level) is done locally
on branch `d9-document-level`, stacked on `d8-teacher-training` (cut from it at `993f32c`), not merged and not
pushed, and no CI run covers it: its STOP gate (a tiny doc model, made by continuing the averaged tiny teacher on
windows of 2 to 4 sentences joined with `<sep>` through the real `rinchen train docify` and `launch --continue-from`
and the real Eole, decodes `<sep>`-aligned output for at least 90 % of held-out dev windows of each size, and the
chunker returns one translation per input sentence for every toy test document) is met on toy templated documents:
alignment .982/.978/.941 for the sizes 2, 3 and 4 after 2,500 continuation steps (1,000 steps, the spike's figure, did
not reach 0.9 on the non-overlapping windows `docify` cuts; the threshold was not lowered). It is machinery, not a
document model: no GPU was used (the one on this machine was not authorised), no real document corpus exists, and
`configs/eole/teacher_doc.yaml` is a table of unmeasured priors; the mix with sentence-level data, the window sizes
and the real continuation schedule are unmeasured, and everything on a GPU stays open. D9 found that Eole's
vocabulary is the file of tokens the data showed (so `<doc>` and `<sep>` were `<unk>` to a teacher that never saw
them) and added the vocabulary invariant: every vocabulary now holds every reserved symbol, which moved every D7 and
D8 tiny-run number (the old values are kept as history in the runbooks). A combined review of the code ran: ready with
fixes, one Critical (a document keyed without its corpus), five Important and twelve Minor findings; its fix wave
(six `D9: fix ...` commits and `D9: review minors`, gate 8214 passed, 1 skipped, 10 deselected) fixed them and left
six findings open on purpose (`docs/runbooks/09_document_level.md`, "Known limits"); a scoped re-review of that fix
wave approved it with minors (no Critical or Important finding) and a second fix wave (`D9: re-review minors`, gate
8252 passed, 1 skipped, 10 deselected) fixed an unstated vocabulary size that Eole's default would cut (`launch` now
refuses it and names the value to state), `docify` on a split that makes no windows or on rows read without it, and
the wording of the `average` guard; three limits stay documented (the `average` guard sees three lexical parents only,
documents are keyed per `corpus` label and not per file, the `launch` pre-flight needs the SentencePiece model
readable from the working directory when a size is stated). D10 (Distillation & students) is done locally on branch
`d10-distillation`, stacked on `d9-document-level` (cut from it at `7d0c166`; code tip `233ba33`), not merged and not pushed,
and no CI run covers it: its STOP gate (through the real `rinchen distill` commands, `rinchen train launch` and the real Eole
and CTranslate2 on the CPU, a tiny teacher of 2,000 steps is exported, its KD corpus of the toy training sources is generated,
a tiny student of 2 encoder and 1 decoder layers trains from random initialisation on that text mixed 1:1 with the originals,
and its own CTranslate2 export decodes what Eole decodes) is met on toy templated data: teacher chrF 98.20, student 98.58,
3,599 KD rows, CTranslate2 equal to Eole on 200 of 200 sentences, no threshold lowered. It is machinery, not a distillation
result: the toy ranks nothing (three seeds of the student span 1.9 chrF), no GPU was used (the one on this machine was not
authorised), no real corpus or teacher exists, and `configs/eole/student_server.yaml` and `student_edge.yaml` are tables of
unmeasured priors; everything on a GPU (float16, int8, the export of a 1.1 B-parameter teacher, the real students' training) and
the quality of any real KD stay open. D10 moved D5's converter call to `rinchen.export.eole_ct2`, exports a checkpoint of ours
with a record and a manifest, reads the tag placement back from the checkpoint's saved config and refuses a statement that
disagrees, and found that CTranslate2 is silent about an unknown tag, a cut hypothesis and a cut source (all guarded), that
selecting by MBR over the toy teacher's n-best lowered the agreement with the references (top-1 is the default), and that Eole
drops 5.2 % of the examples when KD rows are mixed in untagged and 0 % with the `<ft>` tag. A combined review of the code ran:
ready with fixes, no Critical finding and two Important ones (the tag placement of an export was never checked; a KD hypothesis
with an unknown or reserved piece passed the selection); its fix wave (`D10: fix ...` x 4 and `D10: review minors`, gate 9706
passed, 1 skipped, 10 deselected) fixed both and left open on purpose the limits that `docs/runbooks/10_distillation.md` lists
under "Known limits"; a scoped re-review of the fix wave approved it with minors (no Critical or Important finding) and a
small wave of its own (`D10: re-review minors`, gate 9750 passed, 1 skipped, 10 deselected) fixed three of them (a message that named a private path that no
longer exists, a saved config quoted unbounded and unescaped in a message, and a tag placement that nothing could check going
unreported: `export` and `evaluate` now print whether it was checked or only stated) and documented the rest (the hidden
directory a hard-killed export leaves, exports made before the check, shards made before `cc917e2`).
D11 (Export, quantization, serving, CLI, package) is done locally on branch `d11-export-serving`, stacked on
`d10-distillation` (cut from it at `b672aa8`; code tip `039d1d7`), not merged and not pushed, and no CI run covers it: its STOP
gate ("`docker compose up` serves a tiny model") is met IN PARTS and the container part is UNMEASURED. Through the real `rinchen`
commands and the real Eole and CTranslate2 on the CPU the tiny teachers are exported to float32, int8 and int8_float16 (a record
`rinchen.json` replaces D10's `export.json`), strict parity passes (200 of 200 exact, mean |delta| 7.19e-7 per token for the
source-tag teacher and 1.37e-6 for the target-tag one, against 1e-3; the 1e-3 bound is a float32 bound, int8 gave 1.14e-3), the
int8 chrF drop is -0.06 (the gate allows 1.0), script masking leaves 0 forbidden letters in 500 outputs (96 leak without it),
`rinchen.Translator` and `rinchen translate` agree with the engine, and `rinchen serve` (the compose command, run natively)
answers 64 concurrent requests, all 200, p95 far below 2 s, counts them on `/metrics` and drains on SIGTERM (exit 143); no
threshold was lowered. It is machinery, not a model: the toy saturates at chrF 98.2 and ranks nothing, no GPU was used (the one on
this machine was not authorised; `auto` is refused everywhere, `--device cuda` is never run), no real model or corpus exists,
`configs/serve/server.yaml` and `edge.yaml` are tables of unmeasured priors, and no image was pulled, built or run (the Dockerfile
and the compose file are checked statically and the compose command natively). D11 found that `compute_type="auto"` quantises
every export to int8 on a CPU, that Eole's score needs `length_penalty: none` and leaves out a forced target prefix that
`score_batch` counts, that uvicorn exits by the re-raised signal and starts the lifespan before it binds (readiness and the drain
are a refused connection on the wire), and that a document model writes `<sep>` after a lone sentence. A combined review by two
reviewers ran: ready with fixes, no Critical finding and seven Important ones (a document export lost every sentence decoded
alone; a reserved symbol in a request text silently changed the language; `torch` was dropped from the tool versions; a stop could
take twice the shutdown timeout; `n_best` was judged against the wrong beam; a mistyped `RINCHEN_SERVE_*` variable was ignored;
readiness is not observable over HTTP); its fix wave (nine commits, gate 11768 passed, 1 skipped, 10 deselected) fixed them and
left open on purpose the limits that `docs/runbooks/11_serving.md` lists under "Known limits". A scoped re-review of the fix
wave found that it needed one small fix (no Critical finding, one Important, eight Minor): the suppression list of a lone sentence of a document
model still held `<sep>` in the script modes `always` and `retry`'s second pass, so the model ran on and invented text (`always` chrF 81.8
against 88.2 unmasked on the gate's plain rows; 88.5 after the fix); a final wave of three commits (gate 11778 passed, 1 skipped, 10 deselected) fixed it and five of the minors
and documented three (Kubernetes service links under the refused `RINCHEN_SERVE_` prefix, half of the stop budget for requests in flight,
a 40-character echo of an unknown key).
D12 (Robustness) is done locally on branch `d12-robustness`, stacked on `d11-export-serving` (cut from it at `23dc962`; code tip
`ae4e816`), not merged and not pushed, and no CI run covers it: its STOP gate ("robustness block populated in tiny report") is met on the
toy. Through the real `rinchen eval robustness` binary and the real Eole, CTranslate2 and SentencePiece on the CPU a tiny model trained for
the purpose (6,000 steps, 187 s, byte-identical across trainings on this machine) is scored over a generated suite and `report.json` holds all
six values at or above the floors (numbers 892 of 960 = 0.9292 against 0.90, placeholders 599 of 600, entities 447 of 480, terms 1044 of 1044,
script purity 1.0, `mn_Mong` source retention 0.9976 against 0.95), while the controls collapse (hypotheses moved by 7 rows: numbers 3 of 960;
the D11 teacher: 0 of 960 numbers and 0 of 600 placeholders; a glossary the model never writes: 0 of 1,044; placeholder indices 10 and up: 0 of
28; script mask `off`: 101 of 900 rows under the purity floor); no floor was lowered. It is an instrument, not a mechanism: D12 builds no
placeholder masking, no terminology engine and no `terminology` request field (this amends D11's T22; nobody owns them or the training data
that would make a model use them), and the toy ranks nothing; no GPU was used (the one on this machine was not authorised; `auto` is refused,
`--device cuda` is never run), no real corpus, glossary or model exists, and the conventions table, the 4-letter case suffix, the currency symbols
and the 0.98 floor are priors. D12 found that a model trained on a digit-free corpus cannot read or write a digit while the engine says `ok`, that
a model copies a placeholder only for the indices it was trained on (0 to 9 copied, 10 and up 0 of 28), that the engine refuses the inline
terminology markers, and that the numbers metric needs a unit-aware matching and every value its coverage. A combined review by one reviewer ran:
ready with fixes, no Critical finding and four Important ones (identical text failed the numbers metric when one value appeared twice with
different units; the denominators of the values were invisible; `term_acc` is agreement with a glossary, not adherence; the per-row outcomes
could not reproduce the retention); its fix wave (five commits, gate 13543 passed, 1 skipped, 10 deselected, in five parallel parts) fixed three,
repaired a load race in one D8 test and left open on purpose the limits that `docs/runbooks/12_robustness.md` lists under "Known limits". The
scoped re-review of the fix wave approved it with minors (no Critical, no Important); its final wave (three commits, gate 13,583 passed, 1 skipped, 10 deselected) gave a
currency symbol to the number it touches (`$5 $7` against `$7 $5` is exact, `5 $7` against `7 $5` is not), made the surface-exact detail see the layout
of a date (a copy-through echo is surface-exact on 568 of 960 again) and fixed five minors; the per-row `value_cyrl` supports a ratio-of-mean-sentence-chrF
PROXY of the retention, not the corpus statistic, and D13's report must embed `robustness_coverage` beside the six-key block.
D13 (Final eval and release tooling) is done locally on branch `d13-release`, stacked on `d12-robustness` (cut from it at `1a378d0`; code tip
`413bc94`), not merged and not pushed, and no CI run covers it: its STOP gate ("tiny report.md + model card generated; dry-run hub upload (mock)") is met on the
toy. Through the real `rinchen` binary and the real Eole, CTranslate2, SentencePiece and sacrebleu on the CPU, with no network and no token, the toy export of D12
(`model.bin` `e28d1566...a6`) and the D11 teacher translate a FLORES-like fixture suite built from the toy; `rinchen eval run`, `eval report`, `release stage` and
`release hub` (the dry run) write the results, the report, the model card and the nine-file upload plan: chrF++ 97.03, 99.05, 98.97 and 98.38 for the export
against 36.50, 46.09, 29.30 and 31.67 for the teacher (floors 95.0 and 60.0), the paired difference +60.528, +52.965, +69.668 and +66.707 with p 0.001998,
identical systems delta 0 with p exactly 1.0, a 5 % perturbation -3.274 to -4.030 with p 0.001998, three reruns byte-equal across hash seeds and locale, and the
stage and the plan refusing a stray file, a changed byte, a symlink, a token-like line and an absolute path; no floor was lowered. It is an instrument and a release
tool, not a verdict on a model: D13 measures, builds no mechanism and no neural scorer (COMET and MetricX enter only through a seam of score files; B1 to B4 stay open),
the toy ranks nothing, no GPU was used (the one on this machine was not authorised; `auto` is refused, `--device cuda` is never run) and no network was used (`rinchen
eval suite fetch` and `rinchen release hub --upload` were never run; the Hub, FLORES+, NTREX, NLLB, MADLAD, Hy-MT2 and GitHub Actions are unmeasured). D13 found that
sacrebleu's per-segment statistics are integers and a corpus score is the score of their sums, that four of its tokenizers download a model on first use (the design
spike's first run did: a disclosed slip), that its paired bootstrap calls a system against itself significant (so D13 has its own exact one), that the engine answers an
untrained pair with `ok` in the wrong language, that D6's `CT2Translator` is silent about a vocabulary without its family's language tokens, and that
`create_repo(exist_ok=True)` ignores `private` for an existing repository. A combined review by one reviewer ran: ready with fixes, no Critical finding and five
Important ones (the report believed a re-digested result file; the card's evidence came from the whole report; flagged output failures and the dataset and revision were
invisible in `report.md`; private-by-default did not hold for an existing repository); its fix wave (six commits, gate 17,666 passed, 1 skipped, 10 deselected, against
13,583 at the end of D12) fixed them and eleven Minor findings and left open on purpose the limits that `docs/runbooks/13_release.md` lists under "Known limits". A scoped
re-review of the fix wave (one independent reader) ran and approved it with minors: no Critical, no Important and seven Minor findings (a re-digested result could still plant a neural metric or, through the dataset of a suite, a
heading; the devtest guard did not see the eval commands or an aliased dynamic import; two refusals did not say what to do; `--public` over a private repository said nothing; the card printed `n/a ()`; two sentences of the
uploaded text named internal milestones); its final wave (seven commits, `2df73c1..413bc94`, 72 mutants, gate 17,852 passed, 1 skipped, 10 deselected) fixed all seven and left open that a re-digested
result can still plant a free-text reason and D12's robustness block values, which the report cannot recompute; no reader has read the final wave.
D14 (Maintenance loop) is done locally on branch `d14-maintenance`, stacked on `d13-release` (cut from it at `0f3f6ba`; code tip `51c4489`), not merged and not pushed, and no CI run
covers it: its STOP gate ("nightly workflow green on tiny") is met ONLY as "the workflow's own command, read from `nightly.yml`, runs green natively on the CPU through a uv shim that never forwards
the upgrade dry run"; GitHub Actions is UNMEASURED. Through the real `rinchen` binary and the real Eole, CTranslate2 and pytest on the CPU with the GPU hidden, `uv run --frozen --no-sync rinchen maintain
check --out-dir maintain-out --cadence "$CADENCE"` (`CADENCE=nightly`, through `sh` with a `uv` shim first on `PATH`) exits 0 in about 150 s (doctor 13.2 s, the two nightly reruns 136.2 s) and writes a
report, a Markdown summary and a manifest: the lock 187 entries and 186 names, 186 installed packages and none differing, the drift table of a HAND-WRITTEN sample (7 entries, 4 blocked by pin rules),
numerics 1,110 passed (floor 1,100) and `parity_tiny` 13 passed, 188 `tool_versions` keys, the two documents byte-equal across hash seeds and free of paths, timestamps and the environment, `uv.lock`
and `pyproject.toml` untouched, and the controls (a failing, shrunken or overrunning rerun, a bad uv output, a held package, the refusals, the interrupts) fail as they should; no floor was lowered.
It is a CHECK and a schedule, not a verdict on any stack or model: it reports and never changes anything (nothing writes `uv.lock` or `pyproject.toml`, nothing upgrades, nothing asks a device question;
an upgrade is an operator's procedure on a branch, `docs/runbooks/14_maintenance.md`), the reruns run on the toy and rank nothing, no GPU was used (the one on this machine was not authorised; the doctor's
GPU check is never called) and no test or measurement reached a package index (the gate answers the dry run from a sample through a shim; one mutation run of Task 7 ran the real online dry run once in a
copy of the tree by a slip, disclosed). `configs/maintain/maintain.yaml` (10 pin rules, 6 reruns) is a table of unvalidated priors. D14 found that uv's offline dry run fails for any change here, so the lines
for an update, an add and a remove were never measured (a second-hand report of one real run is recorded as such and is not a fixture), that the dry run is blind to the 12 of 38 direct dependencies pinned
with `==` (Eole, CTranslate2, sacrebleu, SentencePiece, torch ...), that a plain `uv run` locks and syncs first (so the workflow uses `--frozen --no-sync`), that the nightly job keeps no log, and that the
floors have margins of 10 (`numerics`) and 0 (the other four reruns). The combined review of the code has NOT run yet; a follow-up commit will record it (gate 19,818 passed, 1 skipped, 10 deselected
before it, against 17,852 at the end of D13). The open gates: everything GPU, the real PyPI state and the lines of a real change, GitHub Actions and its minutes budget, every real export, and the D3 to D13 gates.
Next up: D15 (the tiny end-to-end pipeline; D15 decides whether `just tiny` calls `maintain check --no-drift --no-reruns`). See `STATE.md`, `docs/runbooks/14_maintenance.md` and `docs/runbooks/13_release.md`.

## Evidence and full plan

The current master plan — milestones D0-D15, repo layout, hard constraints,
build-ourselves specs — lives in
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`. The prior Phase 0-14
plan, `docs/research/2026-10-28-python314-stack-and-prompt.md`, is superseded
but kept as the evidence source for decisions already recorded in
`DECISIONS.md` (the package wheel matrix, the pyonmttok/eole finding's
origin). This file only carries the hard constraints and pointers; consult
the master plan before pinning or changing any dependency.
