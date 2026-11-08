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
docs/STACK.md docs/VERIFY.md
docs/runbooks/00_environment.md docs/runbooks/01_acquisition.md
docs/runbooks/02_human_testsets.md docs/runbooks/03_tokenizer.md
docs/runbooks/04_filtering_mining.md docs/runbooks/05_script_converter.md
docs/runbooks/06_synthetic.md docs/runbooks/07_ablations.md
docs/research/  docs/superpowers/{specs,plans}/
src/rinchen/{config,manifest,utils,cli,text,lang,split,spm,corpus,filter,mine,script_conv,synth,metrics,eval,serve}/
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
src/rinchen/config/ablation.py src/rinchen/cli/train.py src/rinchen/utils/logged_process.py
src/rinchen/errors.py src/rinchen/__init__.py
tests/ scripts/ plans/ reports/ manifests/
manifests/ucd_standardized_variants.json
tests/{unit,property,golden,integration}/ tests/fixtures/{annot,golden,raw,spm,script_conv}/
tests/support/ (planted corpus and alignments, fake embedders, tiny LID model,
  tiny_converter, stub_runtime, fake_process, script_reference, planted_synth,
  synth_cli, fake_llm_server, shard_files, corpus_rows, eole_train_config, tiny_parallel,
  tiny_variant, train_cli, fake_ablation)
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
priors only); the gates that need them stay open. Next up: D8 (Teacher training). See `STATE.md`
and `docs/runbooks/07_ablations.md`.

## Evidence and full plan

The current master plan — milestones D0-D15, repo layout, hard constraints,
build-ourselves specs — lives in
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`. The prior Phase 0-14
plan, `docs/research/2026-10-28-python314-stack-and-prompt.md`, is superseded
but kept as the evidence source for decisions already recorded in
`DECISIONS.md` (the package wheel matrix, the pyonmttok/eole finding's
origin). This file only carries the hard constraints and pointers; consult
the master plan before pinning or changing any dependency.
