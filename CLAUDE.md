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
docker/{Dockerfile.cpu,compose.serve.yaml} .dockerignore
docs/STACK.md docs/VERIFY.md
docs/runbooks/00_environment.md docs/runbooks/01_acquisition.md
docs/runbooks/02_human_testsets.md docs/runbooks/03_tokenizer.md
docs/runbooks/04_filtering_mining.md docs/runbooks/05_script_converter.md
docs/runbooks/06_synthetic.md docs/runbooks/07_ablations.md docs/runbooks/08_teacher.md
docs/runbooks/09_document_level.md docs/runbooks/10_distillation.md
docs/runbooks/11_serving.md
docs/research/  docs/superpowers/{specs,plans}/
src/rinchen/{config,manifest,utils,cli,text,lang,split,spm,corpus,filter,mine,script_conv,synth,metrics,eval,serve,export}/
src/rinchen/acquire/{http,opus,hf,warc,commoncrawl,registry}.py
src/rinchen/annotate/{db,export,agreement,app,routes,models}.py
src/rinchen/spm/{symbols,corpus,train,check}.py src/rinchen/split/rules.py
src/rinchen/text/normalize.py src/rinchen/lang/scripts.py
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
src/rinchen/train/{distill_export,distill_saved_config,distill_teacher,distill_sources}.py src/rinchen/train/export_parity.py
src/rinchen/train/{distill_shards,distill_generate,distill_corpus,distill_mix,distill_prepare}.py
src/rinchen/config/ablation.py src/rinchen/cli/{train,train_runs,train_common,train_docify}.py
src/rinchen/cli/{distill,distill_export,distill_generate,distill_generate_plan,distill_generate_records}.py
src/rinchen/cli/{distill_prepare,distill_evaluate}.py src/rinchen/cli/{export,export_parity,translate,translate_io,serve}.py
src/rinchen/utils/{logged_process,integers}.py
src/rinchen/errors.py src/rinchen/__init__.py
tests/ scripts/ plans/ reports/ manifests/
manifests/ucd_standardized_variants.json
tests/{unit,property,golden,integration}/ tests/fixtures/{annot,golden,raw,spm,script_conv}/
tests/support/ (planted corpus and alignments, fake embedders, tiny LID model,
  tiny_converter, stub_runtime, fake_process, script_reference, planted_synth,
  synth_cli, fake_llm_server, shard_files, corpus_rows, eole_train_config, tiny_parallel,
  tiny_variant, train_cli, fake_ablation, step_dirs, fake_launch, fake_eole_process, launch_cli,
  signal_handlers, doc_cli, doc_rows, distill_cli, distill_exports, distill_gate, distill_gate_refusals,
  distill_gate_stages, distill_teachers, fake_ct2, planted_kd, planted_student, model_exports, engines,
  error_messages, translate_cli, parity_cli, fake_parity_eole, serve_app, serve_batcher, fake_serve_process,
  docker_files, d11_gate, d11_gate_doc, serving_gate, serving_gate_fixtures)
scripts/build_spm_fixture.py (regenerates tests/fixtures/spm/tiny_corpus)
scripts/build_script_conv_fixtures.py (reference TSV -> lexicon, golden words, tiny fixture)
data/{raw,clean,mined,synth,test}/ (gitignored content, tracked dirs)
vendor/wheels/ (committed binary wheels with no cp314 upstream release)
vendor/fasttext-wheel-shim/ (pure-Python shim package, see DECISIONS.md)
.github/workflows/ci.yml
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
`d10-distillation` (cut from it at `b672aa8`; code tip `4bfaa5e`), not merged and not pushed, and no CI run covers it: its STOP
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
left open on purpose the limits that `docs/runbooks/11_serving.md` lists under "Known limits". The scoped re-review of the fix
wave was running when this was written and its verdict is not recorded here.
Next up: D12 (Robustness). See `STATE.md`, `docs/runbooks/09_document_level.md`, `docs/runbooks/10_distillation.md` and
`docs/runbooks/11_serving.md`.

## Evidence and full plan

The current master plan — milestones D0-D15, repo layout, hard constraints,
build-ourselves specs — lives in
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`. The prior Phase 0-14
plan, `docs/research/2026-10-28-python314-stack-and-prompt.md`, is superseded
but kept as the evidence source for decisions already recorded in
`DECISIONS.md` (the package wheel matrix, the pyonmttok/eole finding's
origin). This file only carries the hard constraints and pointers; consult
the master plan before pinning or changing any dependency.
