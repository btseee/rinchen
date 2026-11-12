# CLAUDE.md

Read the Wiki page State first for current phase status:
https://github.com/btseee/rinchen/wiki/State (the Wiki is a git repository:
https://github.com/btseee/rinchen.wiki.git).

## Hard constraints

ALWAYS:
- Run everything on CPython 3.14.x. Check `python -c "import sys; assert sys.version_info[:2]==(3,14)"` at the start of every script entrypoint.
- Use uv; one workspace; extra envs only for real dependency conflicts, all on 3.14.
- Train every Rinchen model from random init.
- Use position_encoding_type: SinusoidalInterleaved in every Eole model.
- Record every resolved package version and checkpoint hash in manifests/.
- Write tests before code; numeric acceptance thresholds are gates.
- Keep state in files: the Wiki page State (https://github.com/btseee/rinchen.wiki.git is its git repository; update it in the same change as the code), the committed configs and manifests/, not in chat memory.

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
- `uv run rinchen eval robustness --suite FILE (--model DIR | --hypotheses FILE) --out-dir DIR` (D12: measures numbers, placeholders, names, glossary terms, script purity and a traditional-script source; no threshold, exit 0 whenever it measured; https://github.com/btseee/rinchen/wiki/Runbook-12-Robustness)
- `uv run rinchen eval suite fetch|build|inspect` (D13: `fetch --benchmark B --split S --out-dir DIR (--revision HEX | --unpinned)` is the ONLY network step and was never run; `build --benchmark B|human --from PATH --out-dir DIR` turns local files into a suite file and refuses `khk_Mong` by name; `inspect --suite FILE`)
- `uv run rinchen eval run --suite FILE --out-dir DIR (--model DIR | --generator NAME | --hypotheses FILE)` and `uv run rinchen eval report --run DIR ... --suite FILE ... --out-dir DIR [--baseline SYSTEM]` (D13: BLEU, chrF and chrF++ with 95 % bootstrap intervals, paired comparisons, `report.json` and `report.md`; they measure and exit 0 whenever they measured; https://github.com/btseee/rinchen/wiki/Runbook-13-Release)
- `uv run rinchen release stage --export DIR --report-dir DIR --system NAME --repo-id ORG/NAME --out-dir DIR` and `uv run rinchen release hub --stage DIR --repo-id ORG/NAME --out-dir DIR [--upload] [--public]` (D13: the staging directory with the model card; `hub` is a DRY RUN unless `--upload`, which needs `HF_TOKEN` in the environment, creates the repository private and was never run)
- `uv run --frozen --no-sync rinchen maintain check --out-dir DIR [--cadence nightly|weekly] [--no-drift] [--no-reruns] [--rerun ID] [--fail-on-drift] [--export DIR --checkpoint DIR --corpus FILE]` (D14: the doctor's CPU checks, `uv lock --check`, the installed packages against `uv.lock`, the dependency drift table from `uv lock --upgrade --dry-run` (the ONLY step that queries an index: never run it for a test or a measurement) and the parity reruns; writes `maintain_report.json`, `maintain_report.md` and `manifest.json`, only reads the project, exit 0 for ok or drift and 4 for fail; the upgrade procedure and the rollback are https://github.com/btseee/rinchen/wiki/Runbook-14-Maintenance)
- `uv run --frozen --no-sync rinchen tiny run [--profile tiny] [--out-dir DIR] [--config FILE] [--qe-fake | --no-qe-fake] [--threads N] [--budget-seconds N] [--dry-run]` (D15, the recipe `just tiny`: fifteen stages from the committed fixtures `tests/fixtures/parallel/` to `artifacts/tiny/{report.json,report.md,manifest.json}` on the CPU with no network and no GPU; eight thresholds, exit 0 for pass and 4 for fail with everything written, 2 for a bad option or input or `--profile full` (refused: no full pipeline file), 3 for a changed fixture, a bad lock or a foreign manifest, 130/143/129 for signals; `--dry-run` lists the commands and writes nothing; a miss is answered with more `train.steps`, never a lower bound (the loader refuses a bound below the master plan's); https://github.com/btseee/rinchen/wiki/Runbook-15-Tiny-Pipeline)
- `just wiki-check WIKI_DIR` (`uv run --frozen --no-sync python scripts/check_wiki.py WIKI_DIR --verify-ids-from configs/maintain/maintain.yaml`: checks a clone of the Wiki for the content rules (hits accepted only through `scripts/check_wiki_allowlist.txt`, empty now), dangling internal links, ASCII, line ends, code fences, page names and the Verification-Log rows the maintenance config names; exit 0 clean, 1 problems, 2 not a directory)

## Repo layout

```
CLAUDE.md README.md ACCESSIBILITY.md LICENSE pyproject.toml uv.lock justfile
.pre-commit-config.yaml .gitignore images/logo.png
.github/workflows/{ci,release,nightly}.yml (ci.yml has the jobs gate and tiny)
.github/{CODE_OF_CONDUCT,CONTRIBUTING,SECURITY,PULL_REQUEST_TEMPLATE}.md (community files)
.github/ISSUE_TEMPLATE/{config,bug_report,feature_request,documentation,accessibility}.yml (issue forms)
configs/base.yaml configs/profiles/{tiny,full}.yaml configs/{spm,eole,serve}/
configs/languages.yaml configs/spm/{tiny_1k,student_64k,teacher_128k}.yaml
configs/data/sources.yaml configs/data/sources.tiny.yaml configs/data/filters.yaml
configs/data/synth.yaml
configs/eole/converter{,_tiny}.yaml configs/spm/converter_char.yaml
configs/eole/{multilingual_tiny,ablation_tiny,multilingual_proxy,ablation_priors}.yaml
configs/eole/{teacher,teacher_tiny}.yaml configs/eole/{teacher_doc,teacher_doc_tiny}.yaml
configs/eole/student_{server,edge,tiny}.yaml configs/serve/{tiny,server,edge}.yaml
configs/eval/{metrics,suites}.yaml configs/maintain/maintain.yaml configs/tiny/tiny.yaml
configs/eole/tiny.yaml configs/data/sources.tiny_parallel.yaml (the D15 profile, its Eole base config and its local registry; no configs/tiny/full.yaml)
docker/{Dockerfile.cpu,compose.serve.yaml} .dockerignore
src/rinchen/{config,manifest,utils,cli,text,lang,split,spm,corpus,filter,mine,script_conv,synth,metrics,eval,serve,export,maintain,tiny}/
src/rinchen/__main__.py src/rinchen/py.typed src/rinchen/cli/{main,doctor,data,spm,annotate}.py
src/rinchen/config/{loader,schema,settings,languages,spm}.py src/rinchen/manifest/{model,stage,store}.py
src/rinchen/acquire/{http,opus,hf,warc,commoncrawl,registry}.py
src/rinchen/annotate/{db,export,agreement,app,routes,models}.py src/rinchen/annotate/{templates/,static/htmx.min.js} (package data)
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
src/rinchen/metrics/{registry,segment_stats,bleu_chrf}.py src/rinchen/config/eval_{metrics,suites}.py
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
src/rinchen/tiny/{states,context,fixtures,corpus,fakes,data_stages,commands,command_readers,command_writers,step_runner,model_stages,model_stages_rest}.py
src/rinchen/tiny/{suites,suite_matching,serve_stage,eval_stages,eval_readers,evaluate_facts,thresholds,pipeline,dry_run,graceful_runner}.py
src/rinchen/tiny/{report,report_blocks,report_markdown,report_schema,report_shapes}.py
src/rinchen/config/tiny.py src/rinchen/cli/{tiny,tiny_inputs,tiny_output}.py src/rinchen/eval/fixture_suites.py (write-only)
src/rinchen/utils/{logged_process,integers,logging,part_file,sampling,seed}.py
src/rinchen/errors.py src/rinchen/__init__.py
tests/ scripts/ plans/ reports/ manifests/
manifests/ucd_standardized_variants.json
tests/{unit,property,golden,integration}/ tests/fixtures/{annot,golden,raw,spm,script_conv,robustness,eval,maintain,parallel,tiny,tiny_run}/
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
  maintain_scans, uv_outputs, fake_uv, nightly_workflow, workflow_expression,
  tiny_run, tiny_run_eval, tiny_children, tiny_cli, tiny_config_documents, tiny_fakes_reference, tiny_help,
  tiny_oracle, tiny_pipeline_runs, tiny_profiles, tiny_report_build, tiny_report_inputs, tiny_report_random,
  tiny_report_reference, tiny_report_scans, tiny_report_walk, tiny_scans, tiny_stage_inputs, tiny_threshold_inputs,
  tiny_gate, tiny_gate_processes, tiny_gate_runs, tiny_gate_controls, tiny_gate_hygiene,
  ci_workflow, ci_workflow_edits, ci_workflow_breaks, pseudo_text, sized_spm, community_files)
scripts/build_spm_fixture.py (regenerates tests/fixtures/spm/tiny_corpus)
scripts/build_script_conv_fixtures.py (reference TSV -> lexicon, golden words, tiny fixture)
scripts/build_parallel_fixtures.py (regenerates tests/fixtures/parallel; a changed byte is a re-measurement, not an update)
scripts/check_wiki.py scripts/check_wiki_allowlist.txt (checks a clone of the Wiki; `just wiki-check WIKI_DIR`)
scripts/{probe_env,build_sample_warc_fixture}.py scripts/build_pyonmttok/ (the Phase 0 probe, the WARC fixture, the pyonmttok build)
data/{raw,clean,mined,synth,test}/ (gitignored content, tracked dirs)
artifacts/ (gitignored; `just tiny` writes artifacts/tiny)
vendor/wheels/ (committed binary wheels with no cp314 upstream release)
vendor/fasttext-wheel-shim/ (pure-Python shim package)
```

The documentation (the guide pages, one runbook per operator stage, the stack,
the verification log and the state) lives on the Wiki,
https://github.com/btseee/rinchen/wiki, not in the repository.

Note: `data/`, `align/`, and `convert/` under `src/rinchen/` from the old
Phase 0-14 plan have no home in the new D0-D15 repo layout and were removed;
D1 built `acquire/` there instead of recreating `data/`. No `corpus/`
module was needed because D1's actual scope (per the master plan) is
acquisition only -- fetching/listing raw sources into `data/raw/` -- not
corpus construction, cleaning, or alignment, which are later milestones.

## Current phase

Rinchen is pre-release: no real model, no real corpus and no released artifact
exist. The machinery of every milestone, D0 to D15, is built and on `main`, and
was measured on toy data only (templated fixture sentences: every number ranks
nothing); no GPU was used anywhere. The first GitHub Actions run of the `tiny`
job (run 43244678285) passed: pipeline step about 3 min 19 s, `wall_time`
191.646 s. Open gates: everything GPU, a real corpus, generator and scorers, the
4-core time, `just accept`, and the D3 to D14 gates. Details: the Wiki pages
State (https://github.com/btseee/rinchen/wiki/State) and Roadmap
(https://github.com/btseee/rinchen/wiki/Roadmap).

## Plan and architecture

The milestones, their STOP gates and the open gates are on the Wiki page
Roadmap (https://github.com/btseee/rinchen/wiki/Roadmap); the package map, the
data flow and the design rules are on the Wiki page Architecture
(https://github.com/btseee/rinchen/wiki/Architecture), and the resolved package
versions on the Wiki page Stack (https://github.com/btseee/rinchen/wiki/Stack).
This file only carries the hard constraints and pointers; read those pages
before pinning or changing any dependency.
