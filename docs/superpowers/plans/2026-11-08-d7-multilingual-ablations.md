# D7 Multilingual Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rinchen train ablate`: a schema-validated Eole config renderer, an ablation matrix over temperature sampling, back-translation ratio and tag placement, per-variant training mixtures with integer sampling weights, a development-set evaluation harness, a results table, and the CLI, with a tiny two-variant ablation that trains and evaluates on CPU as the STOP gate.

**Architecture:** Neutral Eole machinery (`utils.logged_process`, `train.eole_cfg`, `train.eole_run`) shared by D5 and D7; `config.ablation` + `train.variants` expand the matrix; `train.mixture` builds corpora and the Eole `data` section; `train.sampling` turns temperature into integer weights; `train.evaluate` translates and scores the dev set; `train.ablation` orchestrates and writes the table; the CLI writes the manifest last.

**Tech Stack:** Python 3.14.10, Eole 0.6.4 (subprocess only), sentencepiece, sacrebleu, polars, pydantic v2, typer, pytest + hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-08-d7-multilingual-ablations-design.md` (binding; findings 1-9, rulings R1-R12). Spike evidence: `.superpowers/sdd/2026-11-08-d7-multilingual-ablations/spike-findings.md` (with scratch scripts and configs that run). Master plan: `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, milestone D7.

## Global Constraints

- CPython 3.14.x only (`uv run`); every script entrypoint asserts `sys.version_info[:2] == (3, 14)`. No new dependency.
- Every model is trained from random init (no `train_from`, no pretrained weights); `position_encoding_type: SinusoidalInterleaved` in every Eole model config; never relative, rotary, ALiBi, sliding-window or grouped-query attention keys (the forbidden list).
- Eole runs only as a subprocess of the `eole` console script next to the interpreter (`python -m eole` does not exist). Unit tests never start Eole (fakes through seams); only the integration tests do, on the CPU with the GPU hidden and the threads pinned.
- Never network access in tests. Nothing written under `data/` or `models/` by tests (`tmp_path`). Fixtures at most 2 MB in total (generate data in tests).
- Never Inner Mongolian data, never galig/Latin informal Mongolian, nothing about licensing or legal topics (code, docs, comments, commit messages). Never tune on FLORES+ devtest or read any test split: D7 reads dev rows only.
- Tests before code; numeric and exact gates are gates (never loosen). Never weaken, skip or delete a failing test. Deterministic seeds everywhere; never silently drop a row or a candidate: every exclusion is counted under a reason.
- D5 must not change behaviour: its tests (`tests/unit/test_script_conv_*.py`, `tests/integration/*script_conv*`, `test_cli_convert_script*`) stay green; the only edits to them allowed are import and patch targets that the moves in Task 1 make necessary.
- Commit subjects start with `D7: `; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Gate: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -m "not network" -q` (never pass `-W error`: it crashes the SentencePiece import). No file mode changes. Match local style (see `src/rinchen/script_conv/`, `src/rinchen/synth/`, `src/rinchen/cli/synth.py`, `src/rinchen/cli/convert_script.py`).
- Invisible, control, look-alike and traditional-script characters in `.py` files only through `chr(0x....)` (the Write and Edit tools turn backslash-u escapes into raw characters); visible Cyrillic letters are fine.
- Work on branch `d7-multilingual-ablations`; do not merge, push or create branches.

## Review Focus

Inputs the spec implies but no happy path exercises, most likely to bite first:

1. Corpora with an unknown language, a direction that has bt rows but no authentic rows, zero bt rows, a bt ratio that needs more rows than exist, duplicate ids, a missing dev set for a direction, text with U+2028, `\r`, tabs, NUL, lone surrogates or an empty string (Tasks 5, 6).
2. Eole config traps: a `$` anywhere, a forbidden key nested deep, an unknown key at each level, `hidden_size` different from `embeddings.word_vec_size`, a `valid` corpus with a prefix, a corpus without `tgt_prefix`, an override of `None` (Task 2).
3. Weights: more corpora than `weight_total`, one corpus, size ratios of 1 to 10^9, temperature tiny or huge, probabilities that round to zero (Task 3).
4. A crash between variants, a reused variant after the corpus bytes or the config changed, two runs into one out dir, an out dir containing `[`, `*`, `?` (use `glob=False` everywhere a path reaches polars) (Tasks 7, 8).
5. Prediction: an output with the wrong line count, a forced tag missing from the output, empty hypotheses, a stray tag in a hypothesis, a SentencePiece piece list containing U+202F (never split with `str.split()`) (Tasks 6, 7).
6. D5 regression through the moves (Task 1) and through the new shared helpers (Tasks 2, 6).

---

### Task 1: Move the shared Eole machinery out of `script_conv`

**Files:**
- Move: `src/rinchen/script_conv/_subprocess.py` -> `src/rinchen/utils/logged_process.py` (git mv); `tests/unit/test_script_conv_subprocess.py` -> `tests/unit/test_utils_logged_process.py` (git mv)
- Create: `src/rinchen/train/__init__.py` (empty), `src/rinchen/train/eole_cfg.py` (only what D5 shares: `FORBIDDEN_KEYS`, `POSITION_ENCODING`, `reject_dollar`, `read_config`, `dump_config`), `src/rinchen/train/eole_run.py` (`CHECKPOINT_FILE`, `training_environment`, `eole_command`, `written_since`, `run_training` with a `validate` callable, `predict_tokens`)
- Modify: `src/rinchen/script_conv/{train,config,export}.py`, `src/rinchen/cli/convert_script.py`, the D5 tests' import and patch targets
- Test: `tests/unit/test_train_eole_run.py`, `tests/unit/test_train_eole_cfg_shared.py` (the moved helpers keep their D5 tests; add only what is new)

**Interfaces:**
- Produces: spec 5.1 (the shared helpers only) and 5.2 (`run_training`, `predict_tokens`, `training_environment`, `eole_command`, `CHECKPOINT_FILE`).
- Consumes: D5's current code in `script_conv/train.py` and `script_conv/config.py` (read them whole first).

- [ ] **Step 1: Run the D5 tests and record the counts** (`uv run pytest tests/unit/test_script_conv_*.py tests/unit/test_cli_convert_script.py tests/integration -q -k "script_conv or convert_script"` and the gate's totals) as the regression baseline.
- [ ] **Step 2: Write failing tests for the new generic pieces.** `run_training` with an injected `validate` that raises `ConfigError` stops before any command; with a fake `run_logged` (patch the symbol in `rinchen.train.eole_run`) it runs `build_vocab` then `train` in that order with the pinned thread environment and returns the model path; a config whose `training.model_path` is outside `run_dir`, a `$` in the config text, the config path or `run_dir`, a missing config file, an exit that wrote no fresh checkpoint: errors exactly as D5's tests state them for the converter (copy the assertions, parametrise over `what`). `predict_tokens` with a fake `run_logged` that writes the output file: the config it dumps has `transforms: []`, `beam_size: 1`, `gpu_ranks: []`, and `tgt` + `tgt_file_prefix: true` exactly when `forced_prefix_tokens` is given; one output line per source line is returned; a wrong line count or a missing output file is an `UpstreamError`; `$` in `ckpt_dir`/`out_dir` is a `ConfigError`; `threads < 1` is a `ValueError`; a prefix list of a different length than the source is a `ValueError`.
- [ ] **Step 3: Run them and see them fail.**
- [ ] **Step 4: Implement the moves surgically.** Keep D5's wording of every message its tests assert (give the generic helpers a `what` argument such as "converter config" so the text is unchanged). D5's `run_training` and `predict` become thin wrappers that keep their signatures and call the generic code; `script_conv/train.py` keeps re-exporting the names its tests import. No D5 behaviour changes.
- [ ] **Step 5: Run the D5 tests again and compare with the baseline (same counts, all green); run the gate.**
- [ ] **Step 6: Commit** (`D7: share the Eole runner and config helpers with script_conv`).

### Task 2: Eole config schema and render

**Files:**
- Modify: `src/rinchen/train/eole_cfg.py`
- Test: `tests/unit/test_train_eole_cfg.py`, `tests/unit/test_train_eole_cfg_drift.py`

**Interfaces:**
- Produces: spec 5.1 in full: the allow-lists, `validate_train_config`, `merge`, `render`.
- Consumes: Task 1's helpers; Eole's pydantic models (`eole.config.run.TrainConfig`, `eole.config.training.TrainingConfig`, `eole.config.models`, `eole.config.data.Dataset`) for the drift test only.

- [ ] **Step 1: Write failing tests.** `merge`: mappings merge recursively, lists and scalars replace, `None` deletes, inputs unmodified, idempotent (property test). `validate_train_config`: the spike's minimal working multilingual config (build it in the test: one `auth` corpus, one `bt` corpus, `valid`, model 64/2+2/4 heads, training keys of the tiny base) passes; each rule of spec 5.1 has a test with its exact message path (unknown key at the top, `data.<corpus>`, `transforms_configs`, `model`, `model.encoder`, `model.embeddings`, `training`; every name of `FORBIDDEN_KEYS` at a deep level; wrong position encoding in embeddings, encoder and decoder; `$`; `overwrite` false or absent; `share_vocab` false or absent; hidden size vs word_vec_size; corpus without `tgt_prefix`, without `src_prefix`, without `path_src`; `valid` with prefix keys or a non-empty `transforms`; weight 0, -1, 1.5, "2", True; `train_from` is an unknown key); a non-mapping config and non-mapping sections give clear errors, never `AttributeError`. `render`: a base with `data: {}` plus overrides renders, validates and round-trips through `yaml.safe_load` equal to the merge; an invalid result is a `ConfigError` and writes nothing (no `out`, no part file); a missing, unreadable, empty, list-valued or non-UTF-8 base is a `ConfigError`; the output directory is created; two renders of the same input are byte-identical (stable key order: keep insertion order, `sort_keys=False`). Drift test: every allowed key at each level is a field of the installed Eole's model at that level (the test imports `eole.config`; the model classes are those the config union selects for `architecture: transformer`); every `FORBIDDEN_KEYS` name is still a real Eole field (D5's drift test, now over the shared constant); the message says which key drifted.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Fix the allow-lists to spec 5.1 and record them; no key without a user.
- [ ] **Step 4: Run the tests and the gate; mutation-check the rule tests (drop each rule, see a test fail).**
- [ ] **Step 5: Commit** (`D7: schema-validated Eole config rendering`).

### Task 3: Sampling weights and corpus metrics

**Files:**
- Create: `src/rinchen/train/sampling.py`
- Modify: `src/rinchen/metrics/bleu_chrf.py`
- Test: `tests/unit/test_train_sampling.py`, `tests/property/test_train_sampling_property.py`, `tests/unit/test_metrics_corpus_scores.py`

**Interfaces:**
- Produces: spec 5.3 and 5.4 (`CorpusSize`, `mixing_probabilities`, `integer_weights`, `CorpusScores`, `corpus_scores`).
- Consumes: `rinchen.utils.sampling.temperature_weights`.

- [ ] **Step 1: Write failing tests.** `mixing_probabilities`: two directions of 100 and 400 rows at T=1 give 0.2/0.8, at T=0.5 give 1/3 and 2/3 (hand computed), probabilities sum to 1 (property), two corpora of one direction (rows 100 and 300) split that direction's probability 1:3, a single corpus gets 1.0, zero-row corpora and a non-positive temperature are errors (`ValueError`), names must be unique. `integer_weights`: the spike's example (sizes 1M/8M/1M/0.6M/0.6M at T=0.3 give weights close to 7/13/7/6/6; compute the expected values by hand from `temperature_weights` and the largest-remainder rule and pin them), every weight at least 1 even for a probability of 1e-9, the sum equals `total` whenever the number of corpora is at most `total`, more corpora than `total` is a `ConfigError` naming the number needed, order preserving (a larger probability never gets a smaller weight), exact on probabilities that are multiples of 1/total, deterministic under input-order changes (the result is keyed by name). Properties (hypothesis) over random size vectors and temperatures. `corpus_scores`: identical text gives chrF 100 and BLEU 100, disjoint gives 0, a hand-checked pair against sacrebleu's own `corpus_chrf`/`corpus_bleu`, signature strings non-empty and stable, different lengths or empty input is a `ValueError`; Cyrillic text works.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** `integer_weights`: floor with a minimum of 1, then distribute the difference by largest remainder, and take surplus from the largest weights one at a time while they stay at least 2.
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D7: temperature mixing weights and corpus metrics`).

### Task 4: Ablation matrix, variants and the shipped configs

**Files:**
- Create: `src/rinchen/config/ablation.py`, `src/rinchen/train/variants.py`, `configs/eole/multilingual_tiny.yaml`, `configs/eole/ablation_tiny.yaml`, `configs/eole/multilingual_proxy.yaml`, `configs/eole/ablation_priors.yaml`
- Test: `tests/unit/test_config_ablation.py`, `tests/unit/test_train_variants.py`, `tests/unit/test_eole_configs_shipped.py`

**Interfaces:**
- Produces: spec 5.5 (`AblationMatrix`, `load_ablation`, `Variant`, `expand_variants`) and the four YAML files.
- Consumes: `rinchen.train.eole_cfg` (`validate_train_config`, `merge`) only in the shipped-configs test.

- [ ] **Step 1: Write failing tests.** Loader: the shipped `ablation_tiny.yaml` loads with the values of spec 5.5; `ConfigError` for a missing file, a non-mapping top level, an unknown field (at the top and in `axes`), a non-positive temperature, a negative bt ratio, a placement outside `src`/`tgt` (including `suffix`, with a message that says it is not supported), duplicate values in any axis, empty axis lists, a negative or non-integer seed, `weight_total` below 1, `dev_max_lines` below 1, `max_variants` below 1, an expansion above `max_variants` (message gives the count), a missing `base` file (checked at load time against the repo anchor), strict types (`"0.5"` for a temperature is an error; `True` for a seed too). `expand_variants`: the product order, ids exactly `t0.5_bt0.5_src_s1234`, `t1_bt0_src_s1234` for the defaults, unique ids (property), count equals the product. Shipped configs: `multilingual_tiny.yaml` and `multilingual_proxy.yaml` each pass `validate_train_config` once a minimal data section is merged in (the base itself may hold `data: {}`); `ablation_tiny.yaml` expands to exactly 2 variants and `ablation_priors.yaml` to 24; the priors file and the proxy base carry a comment that every value is an unmeasured prior.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** The tiny base is the spike's working config: 64/2+2 layers, 4 heads, ff 128, `SinusoidalInterleaved`, shared embeddings, `share_vocab: true`, `overwrite: true`, seed 1234, tokens batch 2048, learning rate 0.5, noam, warmup 100, adamw, dropout 0, label smoothing 0.1, 1000 steps, validation every 500, bucket size 1024, prefetch 8, one worker, `compute_dtype: fp32`, `gpu_ranks: []`, `transforms_configs.sentencepiece` (subword model paths are filled per run), `src_vocab` and `training.model_path` are filled per run. The proxy base is a larger GPU-shaped model (priors, labelled).
- [ ] **Step 4: Run the tests and the gate; mutation-check the validators.**
- [ ] **Step 5: Commit** (`D7: ablation matrix, variants and shipped configs`).

### Task 5: The training mixture (and the toy parallel data)

**Files:**
- Create: `src/rinchen/train/mixture.py`, `tests/support/tiny_parallel.py`
- Test: `tests/unit/test_train_mixture.py`, `tests/unit/test_support_tiny_parallel.py`, `tests/property/test_train_mixture_property.py`

**Interfaces:**
- Produces: spec 5.6 (`load_training_rows`, `load_dev_rows`, `build_corpora`, `data_section`, `write_validation`, `MixtureStats`, `CorpusFiles`) and the toy generator: `build_toy(seed=...)` returning train/dev/test rows per direction for khk_Cyrl, eng_Latn, rus_Cyrl (about 37 basic nouns and six agreement-free templates, en-mn and ru-mn training pairs with different sizes, a bt group built by the same templates with origin `bt`, held-out dev and test combinations with zero overlap asserted), `write_corpus_parquet(rows, path)` in `CORPUS_SCHEMA`, and `train_toy_spm(tmp_dir) -> Path` through `rinchen.spm.train.train_spm` (the spike found 673 is the largest feasible vocabulary for the toy text and about 458 the smallest: pick a constant inside that range and test it).
- Consumes: Task 3's sampling, `rinchen.corpus` (`scan_corpus`, `CORPUS_SCHEMA`), `rinchen.config.languages`, `rinchen.synth.tags.data_origin_tag`, `rinchen.spm`. Port the generator from the spike: `/tmp/claude-1000/-home-tsee-Personal-rinchen/a3e391bc-6f17-4472-b34a-6cd321bc0900/scratchpad/d7-spike/gen_toy.py` and `data/vocab.tsv` (read-only reference; the final code is yours, in the project's style, with the vocabulary inline as a literal table; use only the basic words the spike lists and keep every sentence agreement-free).

- [ ] **Step 1: Write failing tests.** Toy: deterministic (same seed, same rows), zero overlap between train, dev and test in every language, every sentence pair is 3-way consistent (the same item in the three languages), sizes as documented, the Parquet files validate against `CORPUS_SCHEMA`, `train_toy_spm` makes a model in which every direction tag and `<bt>` is one piece. Mixture: R2 exactly (rows with other splits, a non-null `filter_reason`, other origins, and line breaks are excluded and COUNTED under their reasons; the stats add up to the rows read: a conservation test); a language outside the registry is a `ConfigError` naming it; a Parquet file without a required column, an unreadable file, no training rows and an empty dev direction are `DataError`s; paths containing `[`, `*`, `?` work; duplicate ids across files are kept as rows but counted (`duplicate_ids`). `build_corpora`: file names, LF-only UTF-8 lines, line counts equal the stats, src and tgt files stay aligned (a text with U+2028, `\r` or `\n` is excluded and counted under `line_break`; an empty source or target is excluded and counted under `empty`; a tab is kept), `bt_ratio` 0 writes no bt file, ratio r writes `min(available, round(r * n_auth))` rows per direction, a direction with bt rows but no auth rows gets no bt corpus and the stats say so, the subsample is independent of the input row order and of the file order (shuffle the input, same rows), the same seed gives the same rows and another seed gives other rows, the authentic files are reused when present and rewritten when their hash differs. `data_section`: exact dicts for `src` and `tgt` placement, with and without bt (`src_prefix` `<2khk_Cyrl>` / `<2khk_Cyrl> <bt>` for `src`; `''` / `<bt>` with `tgt_prefix` `<2khk_Cyrl>` for `tgt`), weights are the integers given, `valid` carries paths only (no transforms, no prefixes), and the whole section passes `validate_train_config` once merged into the tiny base. `write_validation`: the pre-tokenised lines for both placements against hand-built expectations using the toy SPM (tag is one piece and first on the source side for `src`, first on the target side for `tgt`), one line per dev row, at most `dev_max_lines` per direction by smallest id. `source_collision_rate`: hand-computed on a six-row example.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Hash-ranked sampling per R4; polars with `glob=False`; write files through `utils.part_file`.
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D7: training mixture, validation corpus and the toy parallel data`).

### Task 6: Evaluation harness

**Files:**
- Create: `src/rinchen/train/evaluate.py`
- Test: `tests/unit/test_train_evaluate.py`, `tests/integration/test_train_predict_tokens.py`

**Interfaces:**
- Produces: spec 5.7 (`evaluate_variant`, `DevResult`, per-direction scores).
- Consumes: Task 1's `predict_tokens`, Task 3's `corpus_scores`, Task 5's `DevSet` and toy data, `sentencepiece`.

- [ ] **Step 1: Write failing tests (fake `predict_tokens` through a parameter or a patch of `rinchen.train.evaluate.predict_tokens`).** The source token lines for both placements (tag first for `src`; untagged plus `forced_prefix_tokens` of one tag per line for `tgt`); one call covers all directions; the outputs are split back by direction in order; the forced first token is stripped for `tgt` and a first token that is not the forced tag is an `UpstreamError`; decoding uses `piece.split(" ")` (a piece list containing U+202F survives); per-direction chrF and BLEU equal `corpus_scores` of the decoded hypotheses against the dev targets; the macro mean and the minimum are right; an empty hypothesis scores (as an empty line) without failing; `tag_leaks` counts hypotheses containing any direction tag or `<bt>`; the hypothesis files are written under `out_dir/hyp.<direction>.txt`; an unknown placement is a `ValueError`; an empty dev set is a `DataError`. Integration (real `eole`, no training: reuse D5's tiny trained converter checkpoint fixture from `tests/support/tiny_converter.py` and `tests/integration/conftest.py` if it can be shared without retraining; otherwise train a 30-step toy model once per session): `predict_tokens` with and without a forced prefix returns one line per source line, and with the forced prefix every output line starts with the forced token.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D7: development-set evaluation harness`).

### Task 7: The orchestrator and the results table

**Files:**
- Create: `src/rinchen/train/ablation.py`
- Test: `tests/unit/test_train_ablation.py`, `tests/unit/test_train_ablation_results.py`, `tests/property/test_train_ablation_property.py`

**Interfaces:**
- Produces: spec 5.8 (`plan_ablation`, `AblationPlan`, `run_ablation`, `AblationResult`, `VariantResult`, the table writers, the fingerprint).
- Consumes: Tasks 1 to 6. The `train` and `evaluate` seams are plain callables with the signatures of `run_training` and `evaluate_variant`.

- [ ] **Step 1: Write failing tests (fake seams; toy rows through `tests/support/tiny_parallel.py`).** `plan_ablation`: statistics, variants, per-variant corpus sizes, probabilities, weights (hand-computed for the toy sizes) and effective epochs; no file is created. `run_ablation`: a fake train that records the rendered config path and a fake evaluate that returns scripted scores; per variant the config under `variants/<id>/config.yaml` passes `validate_train_config`, has the weights of the plan, the seed, `training.model_path` inside `variants/<id>/run`, `src_vocab` inside the run dir, the SentencePiece model in `transforms_configs.sentencepiece`, the matrix `overrides` merged last; the variants run in the order of `expand_variants`; a second call reuses every complete variant (the fakes are not called; the progress callback says `reused`); a changed corpus byte, a changed SentencePiece model, a changed override, another thread count or `force=True` reruns exactly the affected variants (fingerprint tests); a variant whose train raises `UpstreamError` stops the run with the cause and leaves earlier variants complete and no `result.json` for the failed one; a corrupt or foreign `result.json` is treated as incomplete (regenerated), never trusted; the tables: `results.tsv` columns and one row per variant with exact values, `results.json` with weights/statistics/signatures, `results.md` with the summary averaging over seeds (two seeds of one variant average; the seed count is shown), numbers formatted with fixed precision; the checkpoint hash of each variant equals the sha256 of its file. Property: for random axes the number of variants equals the product and the ids are unique.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Pure orchestration: sequencing, no business rules (they live in mixture, sampling, eole_cfg).
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D7: ablation orchestrator and results table`).

### Task 8: CLI and the tiny ablation (the STOP gate)

**Files:**
- Modify: `src/rinchen/cli/train.py` (replace the stub), `src/rinchen/cli/main.py` only if needed
- Create: `tests/unit/test_cli_train.py`, `tests/integration/test_train_tiny_ablation.py`, `tests/support/train_cli.py`
- Test: as listed

**Interfaces:**
- Produces: spec 5.9 (`rinchen train ablate`) and the manifest.
- Consumes: Task 7 and `rinchen.cli.errors`, `rinchen.manifest` (study `cli/synth.py` and `cli/convert_script.py` for the D4 to D6 patterns: `rinchen_errors_as_exit_codes`, `refuse_foreign_manifest`, hash-before-read, `begin_commit`, manifest last, `tool_versions`).

- [ ] **Step 1: Write failing unit tests (fake seams; the CliRunner).** Options and defaults (`--help` lists them), required options, exit codes (`ConfigError` 2 for a bad matrix, an unknown language or a missing SPM file path; `DataError` 3 for unreadable corpora; `UpstreamError` 4 from a failing train; an interrupt 130), the dry run prints the variants, the statistics and the weights and creates nothing (not even `out_dir`), a real run creates `out_dir` before paid work (an unwritable out dir fails with no seam called), a foreign manifest is refused, inputs are hashed before they are read (a spy sees the order), the manifest is written last and carries `tool_versions` (eole, torch, sentencepiece, sacrebleu, polars), the input hashes, per-variant checkpoint hashes as outputs and counts as metrics (`count.<key>` style as D6), a rerun reuses variants and the stdout says so, `--force` retrains, `out_dir` with `[`, `*`, `?` works.
- [ ] **Step 2: Write the STOP-gate integration test** with the real binary (`.venv/bin/rinchen` through `subprocess`, as D5's chain test does; skip nothing): toy corpora and the toy SPM in `tmp_path`, `configs/eole/ablation_tiny.yaml`, `--threads 2`, no timeout above 900 s. Assertions per spec 6: exit 0; `results.tsv` has 2 rows and 4 directions each; dev minimum chrF at least 60 for both variants; each rendered config passes `validate_train_config`; weights sum to 100; `tag_leaks` is 0; manifest checkpoint hashes equal the files; the second run exits 0 in a few seconds with both variants reused; `--dry-run` into a fresh directory creates nothing. Record the wall time and the per-variant chrF in the report.
- [ ] **Step 3: Run them and see them fail.**
- [ ] **Step 4: Implement the command.**
- [ ] **Step 5: Run the tests, the integration test and the gate; mutation-check the CLI tests.**
- [ ] **Step 6: Commit** (`D7: rinchen train ablate and the tiny ablation`).

### Task 9: Runbook and records

**Files:**
- Create: `docs/runbooks/07_ablations.md`
- Modify: `docs/superpowers/specs/2026-11-08-d7-multilingual-ablations-design.md` (to the finished code), `STATE.md`, `DECISIONS.md` (CRLF; additions only), `CLAUDE.md` (repo layout and Current phase; not the Hard constraints), `docs/VERIFY.md`, `docs/runbooks/05_script_converter.md` (a pointer to the moved shared helpers, only if it names them)

- [ ] **Step 1: Take the measurements the runbook needs** (read-only scripts in a new scratchpad directory): the tiny ablation's wall time, per-variant chrF per direction, weights, the statistics, file sizes; the shared-machinery timings are not needed; and the weight tables for three illustrative size mixes at T = 0.3, 0.5, 1.0.
- [ ] **Step 2: Write the runbook** (format of runbook 05): what D7 builds and does not, the module table, the matrix reference, commands with the real output of the tiny run, the variant directory layout and resume semantics (the fingerprint contents), the findings 1 to 9 as operator-facing notes (tag on the target: forcing and the dedup hazard, the suffix bug, thread pinning, valid prefix leak), temperature and bt ratio semantics with the weight tables, how to read `results.md` (and that the toy numbers rank nothing), how to add an axis, a "What was not verified" section (no real data, no GPU, priors in the proxy config, scale of the mixture build, the weight resolution, xavier untested at scale, no suffix support).
- [ ] **Step 3: Amend the spec** to the finished code (signatures, error types, defaults, ordering, outputs) and add the list of amendments made while building; update STATE.md (D7 done on branch `d7-multilingual-ablations`, stacked on `d6-synthetic-data`, not merged, not pushed; evidence; open gates; next D8), DECISIONS.md (entry `2026-11-08: D7 multilingual ablations - the rulings, made without a review round`: R1 to R12 with costs, and the evidence blocks for the Eole findings), CLAUDE.md layout, VERIFY.md rows for the Eole facts that are upstream behaviour (`D-EOLE-WEIGHT-SEMANTICS`, `D-EOLE-SUFFIX-KEYERROR`, `D-EOLE-BATCH-DEDUP`, `D-EOLE-UNKNOWN-KEYS`).
- [ ] **Step 4: `rg -n -i "licen|legal"` over the added lines is clean; run the full gate; commit** (`D7: runbook, spec amendments and state and decision records`).
