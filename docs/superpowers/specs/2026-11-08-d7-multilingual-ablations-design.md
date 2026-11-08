# D7 Multilingual ablations: design

Status: binding for the D7 plan (`docs/superpowers/plans/2026-11-08-d7-multilingual-ablations.md`). Written before the build; the build amends it to the finished code (Task 9). Master plan: `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, milestone D7. Evidence for every Eole fact below: `.superpowers/sdd/2026-11-08-d7-multilingual-ablations/spike-findings.md` (a spike on this machine, Eole 0.6.4, CPU, toy templated data).

## 1. Goal and non-goals

Goal: `rinchen train ablate --matrix X` expands an ablation matrix (temperature sampling of the directions, the ratio of back-translated data, the placement of the direction tag) into N variants; for each variant it builds the training mixture, renders a schema-validated Eole config, trains a model from random init, translates the development set of every direction, scores it, and writes a results table. STOP gate: a tiny ablation of 2 variants trains and evaluates through the real CLI on CPU.

Non-goals (stated so nobody assumes them):
- No real data exists, and no GPU was used: every number the gate measures is a toy number on templated sentences that saturate at chrF 98 to 100 and cannot rank variants. The ablation machinery is tested; the ablation CONCLUSIONS (which temperature, which bt ratio, which placement) are not produced by D7 and need real corpora and GPU hours.
- No teacher training (D8), no averaging, no CTranslate2 export or serving (D11), no robustness or final evaluation suites (D12, D13), no document-level data (D9), no distillation (D10).
- Evaluation uses the DEV split only (model selection). The test split and FLORES+ devtest are never read here.
- No fine-tuning, LoRA or continued training: every variant is trained from random init (`train_from` is not an allowed Eole key).

## 2. Findings that bind the design

1. Eole 0.6.4 rejects unknown keys at every level of `eole train` (`extra_forbidden`), but `eole build_vocab` ignores some: our own validation must catch typos before either runs.
2. `data.<corpus>.weight` is an integer: the number of consecutive examples taken from that corpus per round of a cyclic round robin. Proportion is exactly `w_i / sum(w)`, independent of corpus size; an exhausted corpus restarts. Temperature sampling therefore needs integer weights.
3. Eole's own `sentencepiece` + `prefix` transforms work for training with several corpora per direction and a per-corpus `src_prefix` of several pieces (`<2khk_Cyrl> <bt>` splits on the space into two pieces). Both `src_prefix` and `tgt_prefix` must be present on every corpus that uses `prefix` (a missing one is a `TypeError` at `build_vocab`); `''` means none.
4. Eole 0.6.4 takes ONE validation corpus, named `valid`; it copies that corpus's prefix into the saved checkpoint and lets the checkpoint override the prediction config (D5). A validation corpus that carries no prefix keys and is PRE-TOKENISED by the harness (`transforms: []`, the direction tag as a single piece) works for every placement and leaks nothing into the checkpoint.
5. Prediction is done as D5 does it: the harness pre-tokenises with the SentencePiece model, `transforms: []`, greedy decoding, one output line per source line.
6. Tag on the SOURCE (`src_prefix`) works with no forcing and is comparable across variants. Tag on the TARGET (`tgt_prefix`) trains, but prediction needs the tag forced: a `tgt` file with the tag per line plus `tgt_file_prefix: true`; Eole's output then STARTS with the forced tag (strip it), and without forcing an ambiguous direction fails (mn2ru chrF 16). Hazard: Eole drops an example whose tokenised source equals one already in the same minibatch; with the tag on the target, multi-parallel data (the same source in several directions) loses 1.9% of the examples at 1024-token batches, 35.6% at 4096 and 50% at 16384, silently. With the tag on the source, 0.00%.
7. A source-suffix tag needs Eole's separate `suffix` transform and trips an Eole bug (`SuffixTransform.get_specials` raises `KeyError: ''`) unless a workaround corpus is configured: NOT supported in D7 (documented as a finding).
8. Same seed AND same thread count give byte-identical checkpoints and predictions; different thread counts do not. Threads are pinned (default 2) and recorded.
9. SentencePiece: the project's `train_spm` cannot build a vocabulary below about 458 pieces (118 reserved symbols + 256 byte pieces + characters); the toy run uses 673.

## 3. Rulings

R1. Placements are `src` (default) and `tgt`. `suffix` is out (finding 7).
R2. Corpora are `(direction, group)` pairs. Groups: `auth` (origins human, web, mined, converted) and `bt` (origin bt). Rows of other origins (ft, synth_llm, synth_nmt) are NOT used in D7 and are COUNTED by origin in the statistics: forward-translated data enters training with D8/D10. Only rows with `split == "train"` and a null `filter_reason` are training rows; every other row is counted under its reason (`not_train`, `filtered`, `excluded_origin`, `line_break`).
R3. Temperature follows `rinchen.utils.sampling.temperature_weights` (D3): `p ~ size ** T`, T below 1 lifts small corpora, 1 is proportional. It acts on DIRECTIONS (total rows of the direction, auth + bt after subsampling); inside a direction the corpora share its probability in proportion to their rows. So `bt_ratio` fixes the bt share of a direction (`r / (1 + r)` when enough bt rows exist) and `temperature` fixes the balance between directions.
R4. `bt_ratio` r: per direction the bt corpus gets `min(available, round(r * n_auth))` rows, chosen by a seeded hash rank (`sha256(f"{seed}\t{id}")`, smallest first), independent of the input order and of the polars version; `r = 0` or no auth rows in the direction means no bt corpus; a shortfall is not an error, it is recorded (`bt_requested`, `bt_used`).
R5. Integer weights: every corpus gets at least 1; the weights sum to `weight_total` (default 100, at least the number of corpora, else a `ConfigError` that names the number needed); largest-remainder rounding; order preserving. Effective epochs per corpus per variant are computed and recorded.
R6. The training corpora are plain text files (one sentence per line, LF only, no tags): the tags are Eole `prefix` configuration, so placement is a config difference, not a data difference. Authentic corpora are built once and shared by every variant; bt subsamples once per distinct ratio and seed.
R7. The validation corpus and the development evaluation use the dev rows (`split == "dev"`, null `filter_reason`; at most `dev_max_lines` per direction by smallest id). The harness writes the validation corpus pre-tokenised (finding 4): the source tokens `[tag] + pieces` for `src`, `pieces` for `tgt`; the target tokens `pieces` for `src`, `[tag] + pieces` for `tgt`.
R8. A variant's evaluation is ONE `eole predict` call over the concatenated dev sources of all directions (the model loads once), split back by direction; scores are corpus chrF and corpus BLEU (sacrebleu defaults, signatures recorded) per direction, the macro mean and the minimum of chrF over directions, and `tag_leaks` (hypotheses that contain a direction tag or `<bt>`).
R9. `source_collision_rate`: the share of training rows whose source text occurs in at least one other training row of a different target language. It is recorded for every variant and explained in the runbook as the exposure to finding 6's hazard; it is informational, not a gate.
R10. Resume: a variant is complete when its `result.json` exists and its `fingerprint` equals the current one (the rendered config text, the hashes of its corpus files, the SentencePiece model, the dev files, the thread count, the Eole version); a complete variant is reused, `--force` redoes everything. A variant that fails stops the run with the cause; completed variants stay.
R11. Every Eole config rendered by D7 passes `validate_train_config`: curated allow-lists of keys (each checked against the installed Eole's fields by a test), the forbidden-key list (relative, rotary, ALiBi, sliding window, grouped-query attention; moved from D5), `SinusoidalInterleaved` explicit, no `$` anywhere (Eole expands environment variables in the YAML text), `overwrite: true`, `share_vocab: true`, no `train_from`.
R12. The shared Eole machinery D5 built moves out of `script_conv` into neutral homes (`rinchen.utils.logged_process`, `rinchen.train.eole_cfg`, `rinchen.train.eole_run`); D5 imports it from there. D5's behaviour, error wording and tests do not change.

## 4. Modules

| Module | Job | Depends on |
| --- | --- | --- |
| `rinchen.utils.logged_process` | `run_logged` (moved from `script_conv/_subprocess.py`) | errors |
| `rinchen.train.eole_cfg` | allow-lists, forbidden keys, `validate_train_config`, `merge`, `render`, `read_config`, `dump_config`, `reject_dollar` | errors, utils |
| `rinchen.train.eole_run` | `CHECKPOINT_FILE`, `training_environment`, `eole_command`, `run_training`, `predict_tokens` | eole_cfg, utils.logged_process |
| `rinchen.train.sampling` | `mixing_probabilities`, `integer_weights` | utils.sampling |
| `rinchen.metrics.bleu_chrf` | adds `corpus_scores` | sacrebleu |
| `rinchen.config.ablation` | `AblationMatrix`, `load_ablation` | errors (never imports train) |
| `rinchen.train.variants` | `Variant`, `expand_variants` | config.ablation |
| `rinchen.train.mixture` | training/dev rows, corpora files, statistics, the Eole `data` section | corpus, config.languages, synth.tags, sampling |
| `rinchen.train.evaluate` | dev translation and scoring | eole_run, metrics, sentencepiece |
| `rinchen.train.ablation` | the orchestrator and the results table | all of the above, manifest |
| `rinchen.cli.train` | `rinchen train ablate` | train, config |

Layering: `config` imports no feature package; `train` imports `config`, `corpus`, `metrics`, `utils`, `manifest`, `errors`, `spm` (symbols only), `synth.tags` (the origin-to-tag helper) and never `script_conv`; `script_conv` imports `train` and `utils`; `cli` imports everything it needs and nothing imports `cli`.

## 5. Contracts

### 5.1 `rinchen.train.eole_cfg`

- `FORBIDDEN_KEYS: frozenset[str]` (D5's 16 names, moved; D5 re-exports it), `POSITION_ENCODING = "SinusoidalInterleaved"`.
- Allow-lists (frozensets, one per level): `TOP_LEVEL_KEYS` = seed, save_data, src_vocab, tgt_vocab, share_vocab, src_vocab_size, tgt_vocab_size, overwrite, n_sample, report_every, data, transforms, transforms_configs, model, training; `CORPUS_KEYS` = path_src, path_tgt, transforms, weight, src_prefix, tgt_prefix; `TRANSFORMS_CONFIGS_KEYS` (sentencepiece, filtertoolong, and their own keys); `MODEL_KEYS`, `ENCODER_KEYS`/`DECODER_KEYS` (layers, hidden_size, heads, transformer_ff, ...), `EMBEDDINGS_KEYS`; `TRAINING_KEYS` (model_path, batch_type, batch_size, accum_count, dropout, attention_dropout, optim, decay_method, label_smoothing, world_size, train_steps, valid_steps, save_checkpoint_steps, keep_checkpoint, learning_rate, warmup_steps, compute_dtype, gpu_ranks, bucket_size, prefetch_factor, num_workers, param_init, param_init_method, max_grad_norm, adam_beta2, weight_decay, valid_batch_size, accum_steps). The exact lists are fixed in the build and recorded in the runbook; the rule is: only keys a D7 run uses or an operator plausibly overrides in an ablation, each present in the installed Eole's pydantic models (a test imports `eole.config` and checks every allowed key at its level against the model fields; a second test checks every forbidden key is still a real Eole field, as D5 does).
- `validate_train_config(cfg: object) -> None`: raises `ConfigError` whose message starts `invalid Eole config: <dotted.path>: <problem>`: not a mapping; unknown key at any level (the message names the key and the level); a forbidden key at any depth; `model.embeddings.position_encoding_type` (and any encoder/decoder `position_encoding_type` present) not exactly `SinusoidalInterleaved`; a `$` in any string; `overwrite` not true; `share_vocab` not true; `hidden_size` different from `embeddings.word_vec_size` (Eole's default 512 breaks a smaller model); a corpus other than `valid` without `path_src`, `path_tgt`, `src_prefix` or `tgt_prefix` (finding 3); `data.valid` with a `transforms` entry other than an empty list or with prefix keys (finding 4); a `weight` that is not a positive integer.
- `merge(base: Mapping, overrides: Mapping) -> dict`: deep merge; mappings merge, everything else replaces; an override value of `None` deletes the key; neither input is modified.
- `render(base: Path, overrides: Mapping[str, Any], out: Path) -> Path`: reads `base` (a YAML mapping; `ConfigError` for a missing, unreadable, unparseable or non-mapping file), merges the overrides, validates the result, writes `out` (parents created, atomically through `utils.part_file`), returns `out`. The base need not be complete before the merge.
- `read_config(path: Path, what: str) -> dict`, `dump_config(cfg, path)`, `reject_dollar(what, value)`: the D5 helpers, moved.

### 5.2 `rinchen.train.eole_run`

- `run_training(cfg_path, run_dir, *, validate, threads=2, timeout=None, what="Eole config") -> Path`: the D5 flow made generic: `validate(cfg)` (a callable that raises `ConfigError`), `$` rejected, `training.model_path` inside `run_dir`, `eole build_vocab -n_sample -1` then `eole train`, a fresh checkpoint required; returns the checkpoint directory. D5's `run_training` becomes a wrapper that passes its converter validator.
- `predict_tokens(ckpt_dir, source_tokens, *, out_dir, forced_prefix_tokens=None, threads=2, timeout=None, batch_size=64) -> list[str]`: writes the token lines under `out_dir`, runs `eole predict` greedily on the CPU with `transforms: []` (and `tgt` + `tgt_file_prefix: true` when `forced_prefix_tokens` is given), returns Eole's output token lines, exactly one per source line (`UpstreamError` otherwise). D5's `predict` delegates to it.
- Errors as D5: `ConfigError`, `UpstreamError` (non-zero exit, timeout, missing `eole`, no new checkpoint, wrong line count), `ValueError` for fewer than one thread.

### 5.3 `rinchen.train.sampling`

- `mixing_probabilities(corpora: Sequence[CorpusSize], temperature: float) -> dict[str, float]` with `CorpusSize(name, direction, rows)`: R3. Sums to 1; zero-row corpora are not allowed here (the caller drops them first).
- `integer_weights(probabilities: Mapping[str, float], *, total: int = 100) -> dict[str, int]`: R5.

### 5.4 `rinchen.metrics.bleu_chrf.corpus_scores`

`corpus_scores(hypotheses: Sequence[str], references: Sequence[str]) -> CorpusScores`, `CorpusScores(chrf: float, bleu: float, chrf_signature: str, bleu_signature: str)`; `ValueError` for different lengths or empty input. Sentence-level `chrf` stays as it is.

### 5.5 `rinchen.config.ablation` and `rinchen.train.variants`

```yaml
name: tiny
base: configs/eole/multilingual_tiny.yaml   # relative paths are repo-anchored like the other configs
axes:
  temperature: [0.5]          # each > 0, unique; default [1.0]
  bt_ratio: [0.5]             # each >= 0, unique; default [0.0]
  tag_placement: [src, tgt]   # src | tgt, unique; default [src]
seeds: [1234]                 # unique non-negative ints; default [1234]
weight_total: 100
dev_max_lines: 1000
max_variants: 64              # a guard: expansion above it is a ConfigError
overrides: {}                 # merged into every variant's Eole config
```

`AblationMatrix` (pydantic, `extra="forbid"`, strict types) and `load_ablation(path)` (`ConfigError` like `load_synth_config`). `Variant(id, temperature, bt_ratio, tag_placement, seed)` frozen; `expand_variants(matrix)` returns the product in the order temperature, bt_ratio, tag_placement, seed; the id is `t{T:g}_bt{r:g}_{placement}_s{seed}`; the count is checked against `max_variants`.

### 5.6 `rinchen.train.mixture`

- `load_training_rows(paths) -> TrainingRows` and `load_dev_rows(paths, *, max_lines) -> dict[str, DevSet]` read the corpus Parquet files (`scan_corpus`, `glob=False` paths), apply R2 and R7, and return rows plus statistics. Statistics (`MixtureStats`, JSON-ready): rows read, training rows by direction and group, rows excluded by reason, rows excluded by origin, `source_collision_rate`, dev lines by direction.
- `build_corpora(rows, *, bt_ratio, seed, data_dir) -> list[CorpusFiles]`: writes `<name>.src` and `<name>.tgt` (LF only; UTF-8) for the `auth` and, per R4, `bt` corpora; name `f"{src_lang}-{tgt_lang}.{group}"`; the authentic files are written once and reused when present with the right row count and hash.
- `data_section(corpora, weights, placement, *, valid_paths) -> dict`: the Eole `data` section (transforms `[sentencepiece, prefix]` plus `filtertoolong` when the base config configures it; prefixes per placement; `<bt>` after the direction tag in `src_prefix` for `src` and alone in `src_prefix` for `tgt`; the `valid` entry per finding 4). Tags come from `config.languages` (`direction_tag`) and `synth.tags.data_origin_tag`.
- `write_validation(dev, spm_model, placement, out_dir) -> ValidPaths`: R7.
- Errors: `ConfigError` for a language outside the registry or a bt ratio that needs more corpora than `weight_total`; `DataError` for an unreadable corpus file, a missing required column, no training rows at all, or a dev set that is empty for a direction that has training rows.

### 5.7 `rinchen.train.evaluate`

`evaluate_variant(ckpt_dir, dev, *, spm_model, placement, out_dir, threads, timeout) -> DevResult` per R8. It strips the forced first token for `tgt` (a first token that is not the forced tag is an `UpstreamError`), decodes with SentencePiece (`piece.split(" ")`, never `str.split()`), writes `hyp.<direction>.txt`, and never reads anything but dev rows.

### 5.8 `rinchen.train.ablation`

- `plan_ablation(matrix, corpus_paths, spm_model, out_dir) -> AblationPlan`: reads the training and the dev rows from the same corpus files (R2, R7), and returns the statistics, the variants, and per variant the corpus sizes, probabilities, integer weights and effective epochs; no training and no file writes (the dry run prints it).
- `run_ablation(plan, *, threads, timeout, force, train=..., evaluate=...) -> AblationResult`: per variant in order: build corpora, render the config (`render`), `run_training`, `evaluate_variant`, write `result.json`; seams `train` and `evaluate` are injectable for unit tests. Progress callback per variant (`reused`, `trained`).
- Outputs under `out_dir`: `data/`, `variants/<id>/{config.yaml,valid.src,valid.tgt,run/,eval/,result.json}`, `results.tsv` (one row per variant: id, axes, seed, per-direction chrF and BLEU, dev mean chrF, dev min chrF, tag leaks, seconds, source collision rate), `results.json` (everything, including weights, statistics and signatures), `results.md` (the table plus a summary by `(temperature, bt_ratio, tag_placement)` averaging over seeds, with the number of seeds), and `manifest.json` (D4/D5 pattern: inputs hashed before they are read, the manifest written last, `status`, `tool_versions` incl. eole, torch, sentencepiece, sacrebleu, polars, per-variant checkpoint hashes as output files).

### 5.9 CLI `rinchen train ablate`

`--matrix PATH` (required), `--corpus PATH` (repeatable, at least one), `--spm-model PATH`, `--out-dir PATH`, `--threads INT` (default 2), `--timeout FLOAT` (seconds per Eole command), `--force`, `--dry-run`. Exit codes as D4 to D6 (`rinchen_errors_as_exit_codes`: `ConfigError` 2, `DataError` 3, `UpstreamError` 4, other 1; interrupt 130). The dry run reads the inputs, prints the statistics and the planned variants with their weights, and creates nothing. A real run refuses a foreign manifest in `out_dir` (D4's `refuse_foreign_manifest`), hashes its inputs before reading them, creates `out_dir` before paid work, and writes the manifest last.

## 6. Tests and the gate

- Unit tests with fakes for everything but Eole (a fake `run_training` and `evaluate` through the orchestrator's seams; D5's `tests/support/fake_process.py` for the subprocess layer).
- Property tests: `integer_weights` (every weight >= 1, the sum is `total` when it can be, order preserving, close to the probabilities), `mixing_probabilities` (sums to 1, temperature 1 is proportional, a smaller temperature never lowers the smallest direction's share), `merge` (idempotent, no mutation), variant ids unique.
- A drift test against the installed Eole for the allow-lists and the forbidden keys.
- `tests/support/tiny_parallel.py`: the templated three-language toy set (ported from the spike's `gen_toy.py`: about 37 basic nouns, six agreement-free templates, en-mn and ru-mn training pairs, held-out dev and test combinations, zero overlap asserted), Parquet writers in the corpus schema, and a tiny SentencePiece model through `train_spm`.
- STOP gate (integration, real `eole`, real binary through `subprocess` as D5's chain test does): `rinchen train ablate` with `configs/eole/ablation_tiny.yaml` (axes `tag_placement: [src, tgt]`, `temperature: [0.5]`, `bt_ratio: [0.5]`, 1000 steps at 2048-token batches, dropout 0, 2 threads) on the toy corpora: exit 0; two variants each with four directions; dev minimum chrF at least 60 for both (the spike's worst case over four seeds was 98.05); the rendered configs pass `validate_train_config`; the weights sum to 100; `results.tsv`, `results.json`, `results.md` and a manifest whose checkpoint hashes equal the files; a second run reuses both variants (no training); `--dry-run` creates nothing. The two variants take about two minutes together on the reference machine.
- The `valid`-prefix leak (finding 4) has a regression test: a prediction after a training whose validation corpus is prefix-free returns the right language for every direction.

## 7. Open questions, recorded as such

- The proxy config for real ablations (`configs/eole/multilingual_proxy.yaml`, `ablation_priors.yaml`) is a set of unmeasured priors, validated by schema tests only; no GPU run exists.
- Whether the weight resolution of 100 is enough for 16 directions with a bt corpus each (32 corpora) is unmeasured; the guard fails loudly if `weight_total` is below the number of corpora.
- Writing the training text files for hundreds of millions of rows loads one direction at a time; the memory and time at that scale are unmeasured.
- `xavier_uniform` initialisation converged much faster on the toy; it is NOT adopted (an unmeasured effect at scale); `param_init_method` is an allowed key so an ablation can try it.
