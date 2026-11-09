# 07 - Multilingual ablations

## Scope

D7 builds the multilingual ablation stage: the package `rinchen.train` (a schema-validated Eole config
renderer and runner, temperature mixing of the training corpora with integer weights, the training
mixture, a development-set evaluation harness and the ablation run with its results tables), the
matrix config `rinchen.config.ablation` and the command `rinchen train ablate`. The design, with the
rulings that fill the gaps of the master plan and what each costs if wrong, is
`docs/superpowers/specs/2026-11-08-d7-multilingual-ablations-design.md`; the decisions are condensed
in the 2026-11-08 entry of `DECISIONS.md`; the upstream facts it relies on are in `docs/VERIFY.md`
(section "D7 findings").

**What D7 is.** A tested pipeline that expands an ablation matrix (the temperature of the corpus
mixing, the ratio of back-translated to authentic data, the placement of the direction tag, the
seeds) into variants and, for each: builds the training mixture from corpus Parquet files, writes
integer sampling weights, renders and validates an Eole config, trains a model from random
initialisation with `eole`, translates the development set of every direction in one prediction,
scores it (chrF and BLEU per direction) and writes `results.tsv`, `results.json` and `results.md`. A
finished variant is reused on a rerun; a trained one whose evaluation failed is only evaluated again.

**What D7 is NOT.** Nothing below is a statement about which temperature, ratio or tag
placement is better.

- **No ablation conclusion exists.** The only data the machinery ever ran on is the toy set of
  `tests/support/tiny_parallel.py` (37 nouns in six templates, three languages), which the model
  saturates: the two variants of the gate scored chrF 96.25 to 99.72 in every direction when D7 was
  built, and 90.89 to 99.45 after D9's vocabulary invariant changed the model's vocabulary (see
  "The toy numbers rank nothing"). Those numbers show that the tags, weights, prediction and scoring
  work end to end; they rank nothing.
- **No real corpus and no GPU run exist.** Every run was on the CPU (the GPU hidden from Eole), on at
  most 5,200 toy rows. The proxy base config (`configs/eole/multilingual_proxy.yaml`) and the
  ablation that uses it (`configs/eole/ablation_priors.yaml`) are unmeasured priors, checked only by
  schema tests (Eole accepts the files); nothing was trained or timed with them.
- **No teacher, export, serving or final evaluation** (D8 and later). D8 has since built the resumable
  launcher, the checkpoint averager and `ablate --prepare-only` (`docs/runbooks/08_teacher.md`); no
  teacher was trained. Evaluation reads the DEV split only (model selection); the test split and
  FLORES+ devtest are never read here.
- **No fine-tuning.** Every variant is trained from random initialisation; `train_from` is not an
  allowed Eole key of an ablation config. (D8's launcher allows it in one place only: the config it
  renders itself to resume its OWN run from a complete checkpoint of the same run directory,
  `validate_train_config(..., resumes=True)`; see `docs/runbooks/08_teacher.md`. D9's `launch
  --continue-from` starts a NEW run from OUR OWN teacher, with a provenance rule, and is the one other
  place `train_from` and `reset_optim` appear: `docs/runbooks/09_document_level.md`.)

Every number below is from a test, the code, the design, the spike (`spike-findings.md` in the SDD
workspace; toy data) or a measurement made for this runbook on 2026-11-08 ("Measured numbers"); a
number that was not measured is not given. What was checked: the unit and property tests (every new
test checked by mutation by hand); `tests/integration/test_train_evaluate.py` and
`tests/integration/test_train_tiny_ablation.py` (the STOP gate: the real `rinchen` binary and Eole on
the CPU, the second of the two); and the real binary for this runbook (dry run, run, reruns, a damaged checkpoint, failing
invocations, a probe of the mixture build at 1,000,000 and 4,000,000 rows).

## Architecture

| Module | Job |
|---|---|
| `rinchen/utils/logged_process.py` | `run_logged`: a command with its output in a log, a timeout, a wait in 0.2 s slices that Ctrl-C can interrupt, SIGTERM and SIGHUP turned into an exit, and a kill of the whole process group (moved from `script_conv/_subprocess.py`; D8 added `inherit_fds`) |
| `rinchen/train/eole_cfg.py` | Allow-lists of Eole config keys per level (D9 moved the lists to `eole_cfg_keys.py`, which `eole_cfg` re-exports), `FORBIDDEN_KEYS`, `validate_train_config`, `merge`, `merged_config`, `render`, `read_config`, `dump_config`, `reject_dollar`, `reject_glob_characters` (D8 extended the lists and added the `resumes` flag: runbook 08; D9 added the `continues` and `resumes_continuation` rules: runbook 09) |
| `rinchen/train/eole_run.py` | `run_training` (`eole build_vocab`, the reserved symbols appended to the vocabulary file when the caller gives them, then `eole train`), `predict_tokens` (`eole predict` on token lines, optionally with forced prefixes and, since D9, `max_length` and `max_length_ratio`), `add_reserved_symbols`, `training_environment`, `eole_commands`, `CHECKPOINT_FILE` (the D9 additions: runbook 09) |
| `rinchen/train/sampling.py` | `mixing_probabilities` (temperature over directions) and `integer_weights` (largest-remainder rounding to `weight_total`) |
| `rinchen/config/ablation.py` | `AblationMatrix`, `Axes`, `load_ablation`, `TagPlacement`, `id_number`; imports no feature package |
| `rinchen/train/variants.py` | `Variant`, `expand_variants` (the product of the axes and the seeds) |
| `rinchen/train/mixture.py` | Which corpus rows train and which are dev rows, and what was counted: `load_training_rows`, `load_dev_rows`, `MixtureStats`, `DevStats`, `source_collision_rate` |
| `rinchen/train/corpus_files.py` | The plain-text corpus files Eole reads: `build_corpora`, `planned_corpus_sizes`, `planned_corpus_files`, `bt_usage_by_direction`, reuse by digest sidecar |
| `rinchen/train/eole_data.py` | The Eole `data` section: `data_section`, `bt_tag` |
| `rinchen/train/validation_corpus.py` | The dev sets as SentencePiece token lines: `dev_token_lines`, `write_validation`, `require_direction_tags`, `load_processor` |
| `rinchen/train/evaluate.py` | `evaluate_variant`: one prediction over all dev lines, decoding, corpus chrF and BLEU per direction, `tag_leaks` |
| `rinchen/train/ablation_layout.py` | Where the files of a run go (`variant_paths`, `data_dir`); touches no disk |
| `rinchen/train/ablation_config.py` | The Eole config of one variant: the per-run values laid under the matrix overrides (`variant_overrides`, `variant_config`) |
| `rinchen/train/ablation_plan.py` | `plan_ablation`: the rows read, the variants, per variant the corpus sizes, probabilities, weights and upsampling; checks what would be rendered; writes nothing |
| `rinchen/train/ablation_prepare.py` | `prepare_variant`: the corpus files, validation corpus and rendered config of one variant, and its fingerprint |
| `rinchen/train/ablation_records.py` | `VariantResult`, `AblationResult`, `VariantProgress`, and the checks that decide whether a `result.json` or `train.json` on disk may be trusted |
| `rinchen/train/ablation_report.py` | `write_results`: `results.tsv`, `results.json`, `results.md` |
| `rinchen/train/ablation.py` | `run_ablation`: the loop over the variants, reuse, `train.json`, `result.json` |
| `rinchen/metrics/bleu_chrf.py` | `corpus_scores` (added): corpus chrF and BLEU with the sacrebleu signatures |
| `rinchen/cli/train.py` | `rinchen train ablate` (with `--prepare-only` since D8) and its manifest; it also registers `launch` and `average`, which live in `cli/train_runs.py` (runbook 08) |

The layering: `config.ablation` imports no feature package; `train` imports `config`, `corpus`,
`metrics`, `utils`, `manifest`, `errors` and `synth.tags` (the origin-to-tag helper) and, from
outside the repository, `polars`, `pyyaml`, `pydantic` and `sentencepiece` (the last in
`validation_corpus` and `evaluate` only; `sacrebleu` comes in through `metrics`), never
`script_conv`, `spm` or `cli`; `script_conv` imports `train.eole_cfg`, `train.eole_run` and
`utils.logged_process` (D5's runner and validator wrap the shared ones); `cli` imports everything it
needs and nothing imports `cli`. Inside `train` the order is `eole_cfg`, `eole_run`, `mixture`,
`corpus_files` / `validation_corpus` / `eole_data`, `evaluate`, `ablation_layout`, `ablation_config`,
`ablation_plan`, `ablation_prepare`, `ablation_records`, `ablation_report`, `ablation`, each importing
only the ones before it. Eole is reached only through `eole_run`; the tests of the loop and of the
command replace `run_training` and `evaluate_variant` (the `train` and `evaluate` seams of
`run_ablation`).

## Matrix reference

`configs/eole/ablation_tiny.yaml` (the gate) and `configs/eole/ablation_priors.yaml` (the ablation
for real data) are the two shipped matrices. `load_ablation` raises `ConfigError` for a missing,
unreadable or unparseable file, a top level that is not a mapping, an unknown field (also inside
`axes`), a wrongly typed or out-of-range value (numbers and integers are strict: a quoted number or
a YAML boolean is an error, not a coercion), a duplicate value, a base file that does not exist and
an expansion above `max_variants`. Relative paths (`base`) are anchored at the repository root.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Labels the run in `results.md` and the dry run; not blank. Not part of any fingerprint |
| `base` | required | The Eole config every variant starts from: an existing file, not blank |
| `axes.temperature` | `[1.0]` | Each greater than 0 and finite. Mixing weight of a direction is `rows ** T`; below 1 lifts the small directions, 1 is proportional. See "Temperature and bt ratio" |
| `axes.bt_ratio` | `[0.0]` | Each from 0 to 1000, finite, `-0.0` refused (a larger number is a `ConfigError` naming the bound: it is a typo, and the rows it asks for would print as a number of hundreds of digits). Back-translated rows per authentic row of a direction; 0 uses none |
| `axes.tag_placement` | `[src]` | `src` or `tgt`. `suffix` is refused with a message naming the Eole bug (`D-EOLE-SUFFIX-KEYERROR`) |
| `seeds` | `[1234]` | Non-negative integers, distinct. Each is the Eole `seed` of its variant and the seed of its bt subsample |
| `weight_total` | 100 | Sum of the integer corpus weights of a variant, a positive integer, at least the number of corpora |
| `dev_max_lines` | 1000 | Development lines per direction, a positive integer |
| `max_variants` | 64 | A guard: the product of the axis lengths and the seeds above it is a `ConfigError` at load |
| `overrides` | `{}` | A mapping merged over every variant's Eole config, last; a value of `null` deletes the key. Refused for the reserved keys below |

Duplicates are compared as printed, because the printed values make the variant ids: `0.1234567`
and `0.1234568` both print as `0.123457` and would name two variants alike. PyYAML reads `1e-5` (no
decimal point, no sign on the exponent) as text; the loader says so and asks for `1.0e-5`.

The variant id is `t{T:g}_bt{r:g}_{placement}_s{seed}` (for example `t0.5_bt0.5_src_s1234`;
`1.0` prints as `1`). The variants are the product in the order temperature, bt ratio, tag
placement, seed (the seed changes fastest).

**Reserved per-run override keys.** The run sets these itself for every variant, so a matrix
`overrides` that sets, replaces or deletes one of them (or the parent section that holds it) is a
`ConfigError` naming the key at plan time, also for a dry run: `data`, `src_vocab`, `seed`,
`training.model_path`, `transforms_configs.sentencepiece`. Everything else in the allow-lists below
may be overridden, for example `training.train_steps` or `transforms_configs.filtertoolong`.

**Which values are priors.** All of them, until a real run measures otherwise: `weight_total` 100
(coarse for many corpora, see "Temperature and bt ratio"), `dev_max_lines`, `max_variants`, the
default seed, every value of `ablation_priors.yaml` (temperatures 0.3, 0.5, 1.0; bt ratios 0, 0.25,
0.5, 1.0; both placements; one seed: 24 variants) and every value of `multilingual_proxy.yaml` (6+6
layers, 512 wide, 2048 feed-forward, 8 heads, Adam with beta2 0.998, Noam decay with warmup 8000,
label smoothing 0.1, dropout 0.1, 8192-token batches with `accum_count` 2, 50,000 steps, fp16 on one
GPU). The tiny base (`multilingual_tiny.yaml`) is the spike's CPU recipe, measured on the toy set: 64
wide, 2+2 layers, 4 heads, feed-forward 128, 2048-token batches, learning rate 0.5 with Noam warmup
100, AdamW, dropout 0, 1000 steps, `bucket_size` 1024, `prefetch_factor` 8, one worker.

The matrix has no axis for an Eole key. To compare two values of one (a learning rate,
`param_init_method`), write one matrix per value (an override each) and give each its own
`--out-dir`.

## The Eole config

### Allow-lists

`validate_train_config` allows only these keys, level by level (copied from
`src/rinchen/train/eole_cfg_keys.py` since D9 (`eole_cfg` re-exports them), alphabetical here; D8
added 4 top-level and 9 `training` keys for the teacher runs). Eole 0.6.4 rejects any other key of `eole
train` (`D-EOLE-UNKNOWN-KEYS`), and `tests/unit/test_train_eole_cfg_drift.py` checks every listed key
against the fields of the installed Eole.

| Level | Keys |
|---|---|
| top level (19) | `data`, `log_file`, `model`, `n_sample`, `overwrite`, `report_every`, `save_data`, `seed`, `share_vocab`, `src_vocab`, `src_vocab_size`, `tensorboard`, `tensorboard_log_dir`, `tgt_vocab`, `tgt_vocab_size`, `training`, `transforms`, `transforms_configs`, `valid_metrics` |
| `data.<corpus>` (6) | `path_src`, `path_tgt`, `src_prefix`, `tgt_prefix`, `transforms`, `weight` |
| `transforms_configs` (2) | `filtertoolong`, `sentencepiece` |
| `transforms_configs.sentencepiece` (2) | `src_subword_model`, `tgt_subword_model` |
| `transforms_configs.filtertoolong` (2) | `src_seq_length`, `tgt_seq_length` |
| `model` (9) | `architecture`, `decoder`, `embeddings`, `encoder`, `heads`, `hidden_size`, `share_decoder_embeddings`, `share_embeddings`, `transformer_ff` |
| `model.encoder` (5) | `heads`, `hidden_size`, `layers`, `position_encoding_type`, `transformer_ff` |
| `model.decoder` (5) | `heads`, `hidden_size`, `layers`, `position_encoding_type`, `transformer_ff` |
| `model.embeddings` (2) | `position_encoding_type`, `word_vec_size` |
| `training` (36) | `accum_count`, `adam_beta2`, `attention_dropout`, `batch_size`, `batch_size_multiple`, `batch_type`, `bucket_size`, `compute_dtype`, `decay_method`, `dropout`, `early_stopping`, `early_stopping_criteria`, `gpu_backend`, `gpu_ranks`, `keep_checkpoint`, `label_smoothing`, `learning_rate`, `master_ip`, `master_port`, `max_grad_norm`, `model_path`, `normalization`, `num_workers`, `optim`, `param_init`, `param_init_method`, `prefetch_factor`, `save_checkpoint_steps`, `self_attn_backend`, `timeout`, `train_steps`, `valid_batch_size`, `valid_steps`, `warmup_steps`, `weight_decay`, `world_size` |
| forbidden at any depth (16) | `global_heads_kv`, `heads_kv`, `interleave_local`, `layer_types`, `multidimensional_rope`, `partial_rotary_factor`, `qk_norm_post_rope`, `relative_positions_buckets`, `rope_config`, `rotary_dim`, `rotary_dim_global`, `rotary_interleave`, `rotary_theta`, `rotary_theta_local`, `sliding_window`, `xdrope_section` |

Nothing that adapts, continues or quantises a model is listed (no LoRA; `train_from` is accepted only by
`validate_train_config(..., resumes=True)`, which D8's launcher uses for its own resume config, and, with
`reset_optim`, by the `continues=` and `resumes_continuation=True` forms that D9's launcher uses for its
own continuation attempts: runbook 09), and no relative, rotary, ALiBi, sliding-window or
grouped-query key; the lists describe a plain transformer. `average_decay`, `average_every`,
`accum_steps`, `dropout_steps`, `reset_optim` and `update_vocab` are deliberately absent from the lists
(runbook 08; `reset_optim` has the two forms above, `update_vocab` none).

### The rules of `validate_train_config`, in the order they are checked

The first rule broken is reported, as `invalid Eole config: <dotted.path>: <problem>` (a
`ConfigError`, exit 2).

1. The config is a mapping.
2. No key of the forbidden list at any depth.
3. No `$` in any key or string: Eole expands `$NAME` as an environment variable in the whole YAML
   text.
4. Every value is plain YAML data of exactly its type (text, a number, true or false, null, a list or
   a mapping; an enum member or another subclass is refused); a mapping or list that contains itself
   (a recursive anchor) is refused.
5. Each section is a mapping and holds only the keys of its allow-list: the top level, each corpus of
   `data`, `transforms_configs` and its two transforms, `model`, its `encoder`, `decoder` and
   `embeddings`, and `training`.
6. `model.architecture` is `transformer`.
7. `model.embeddings` is present and its `position_encoding_type` is `SinusoidalInterleaved`; so is
   the encoder's and the decoder's when set.
8. `model.hidden_size` and `model.embeddings.word_vec_size` are both set and equal (Eole's default of
   512 breaks a smaller model with a layer-norm shape error); so is the `hidden_size` of the encoder
   or the decoder when set.
9. `overwrite` and `share_vocab` are `true`.
10. Every corpus but `valid` has `path_src`, `path_tgt` (non-empty text), `transforms` as a list of
    strings, and both `src_prefix` and `tgt_prefix` as strings (a missing one is a `TypeError` in
    `eole build_vocab`; `''` means none); a `weight`, when given, is a positive integer (a boolean is
    not). `valid` has `path_src` and `path_tgt`, may have `transforms: []` and nothing else: no
    prefix (Eole copies the `valid` prefix into the checkpoint, where it overrides the prediction,
    `D-EOLE-PREFIX-OVERRIDE`) and no weight.

What it does NOT check: the type of the value of a known key (a `train_steps` of `"ten"` passes the
dry run and fails in `eole train`, which has its own pydantic check; the keys D8 added are the
exception, their kinds of value are checked), and that the config has at least one training corpus
(the plan does, below). D8 added two rules between the ninth and the tenth: the kinds of value of the
teacher keys and, with `resumes`, the form of `train_from`; D9 added two more at the end, for the
launcher's continuation attempts (`continues=`, `resumes_continuation=True`: runbook 09).

`merge(base, overrides)` merges mappings key by key at any depth; every other value, a list included,
replaces; an override of `None` deletes the key; neither input is changed. `render(base, overrides,
out)` merges, validates and writes `out` atomically (a failing config writes nothing; `out` may not be
the base, a directory, or below a file). A variant's config is the base, then the per-run values
(`seed`, `src_vocab` = `variants/<id>/run/vocab.src`, `data`, `transforms_configs.sentencepiece` for
both sides, `training.model_path` = `variants/<id>/run/ckpt`), then the matrix overrides over them,
every path absolute in POSIX form. Since D9 the finished config also has `src_vocab_size`: the
SentencePiece model's pieces plus one (Eole's pad token) unless the base or the overrides give a size,
and a given size below that is a `ConfigError` naming both numbers (runbook 09, "The vocabulary
invariant"); that key is why every variant trained before D9 has another fingerprint and retrains. The corpora run the `filtertoolong` transform when the finished
config configures `transforms_configs.filtertoolong`; the base is the switch.

`$` in the output directory or the SentencePiece model path is refused (it would be expanded). `[`,
`]`, `*` and `?` in the output directory are refused too, for a reason that is easy to meet: Eole loads
a checkpoint through `glob(<model_path>/model.*.safetensors)`, so those characters in the checkpoint
directory make the loader find no shard, or the shards of a sibling directory (`D-EOLE-GLOB-CKPT`).
Before the check was added, `--out-dir 'out[1]'` trained a whole variant and then failed at the first
prediction; `--out-dir 'out*'` passed only because the pattern matched four sibling checkpoints, so
another run's weights could have been scored as this one's. Only the checkpoint directory is globbed;
a corpus, config or SentencePiece path may hold those characters. D5's `export_ct2` is handed a
checkpoint directory and fails the same way; it was not changed.

## Commands

One command, `uv run rinchen train ablate` (`rinchen train` is its group). The runs below were made
on 2026-11-08 on the machine of "Measured numbers", through the real binary (`.venv/bin/rinchen`,
what `uv run rinchen` starts) from the repository root, on the toy data. `$W` is a scratch
directory. The inputs are written by the test helper (a corpus Parquet file with 5,200 rows, 126,698
bytes, and a SentencePiece model, 12,599 bytes; the Parquet file has the same bytes on every build; the
model's size varies by a byte or two with the name of the directory it is written to):

```bash
uv run python -I -c "import sys; sys.path[:0] = ['src', '.']; from pathlib import Path; from tests.support.tiny_variant import build_toy_inputs; build_toy_inputs(Path('$W/in'))"
```

### Dry run

```bash
uv run rinchen train ablate --matrix configs/eole/ablation_tiny.yaml \
    --corpus $W/in/toy.parquet --spm-model $W/in/spm/toy.model --out-dir $W/out --dry-run
```
```
matrix: tiny
out dir: $W/out
weight total: 100
corpus files: 1
rows read: 5200
rows kept: 4800
excluded not_train: 400
excluded filtered: 0
excluded excluded_origin: 0
excluded empty: 0
excluded line_break: 0
excluded by origin ft: 0
excluded by origin synth_llm: 0
excluded by origin synth_nmt: 0
duplicate ids: 0
source collision rate: 0.000000
directions: eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn, khk_Cyrl-rus_Cyrl, rus_Cyrl-khk_Cyrl
dev rows read: 5200
dev rows usable: 400
dev lines eng_Latn-khk_Cyrl: 100
dev lines khk_Cyrl-eng_Latn: 100
dev lines khk_Cyrl-rus_Cyrl: 100
dev lines rus_Cyrl-khk_Cyrl: 100
dev in train: 0
variants: 2
variant t0.5_bt0.5_src_s1234: temperature 0.5, bt ratio 0.5, tag placement src, seed 1234
variant t0.5_bt0.5_src_s1234 bt eng_Latn-khk_Cyrl: available 300, requested 600, used 300
variant t0.5_bt0.5_src_s1234 bt khk_Cyrl-eng_Latn: available 300, requested 600, used 300
variant t0.5_bt0.5_src_s1234 bt khk_Cyrl-rus_Cyrl: available 300, requested 300, used 300
variant t0.5_bt0.5_src_s1234 bt rus_Cyrl-khk_Cyrl: available 300, requested 300, used 300
variant t0.5_bt0.5_src_s1234 corpus eng_Latn-khk_Cyrl.auth: rows 1200, probability 0.225403, weight 22, upsampling 0.880
variant t0.5_bt0.5_src_s1234 corpus eng_Latn-khk_Cyrl.bt: rows 300, probability 0.056351, weight 6, upsampling 0.960
variant t0.5_bt0.5_src_s1234 corpus khk_Cyrl-eng_Latn.auth: rows 1200, probability 0.225403, weight 22, upsampling 0.880
variant t0.5_bt0.5_src_s1234 corpus khk_Cyrl-eng_Latn.bt: rows 300, probability 0.056351, weight 6, upsampling 0.960
variant t0.5_bt0.5_src_s1234 corpus khk_Cyrl-rus_Cyrl.auth: rows 600, probability 0.145497, weight 15, upsampling 1.200
variant t0.5_bt0.5_src_s1234 corpus khk_Cyrl-rus_Cyrl.bt: rows 300, probability 0.072749, weight 7, upsampling 1.120
variant t0.5_bt0.5_src_s1234 corpus rus_Cyrl-khk_Cyrl.auth: rows 600, probability 0.145497, weight 15, upsampling 1.200
variant t0.5_bt0.5_src_s1234 corpus rus_Cyrl-khk_Cyrl.bt: rows 300, probability 0.072749, weight 7, upsampling 1.120
  ... the same 13 lines for variant t0.5_bt0.5_tgt_s1234 (tag placement tgt), identical apart from the id
nothing was trained and nothing was written
```

(The line of dots stands for the second variant's 13 lines, which the real output prints.) A dry run
hashes no input, builds no corpus file and creates no directory (`$W/out` does not exist afterwards;
wall time 0.71 s, peak resident memory 215 MiB). It does everything a real run checks before it writes: the refusals of
the matrix, a foreign manifest and an `--out-dir` that is an input's directory, the reserved
overrides, the mixture, the weights, the SentencePiece model (it loads and has every direction tag as
one piece) and the Eole config of every variant, validated in memory. A typo in an override fails the
dry run with exit 2, not after the data directory has been written.

How to read it: `rows kept` are the training rows; each excluded row is counted under the first
reason that applies, in the order `not_train`, `filtered` (a `filter_reason` is set), `excluded_origin`
(origin `ft`, `synth_llm` or `synth_nmt`: forward-translated data enters training with D8 and D10; also
counted by origin), `empty` (null, empty or whitespace-only text) and `line_break` (a character
`str.splitlines` splits at, U+2028 included, which would turn one sentence into two lines of a text
file); `rows read = rows kept + the exclusions`. `dev in train` counts dev lines whose row id is also
that of a kept training row (such a line inflates the dev score). `requested 600, used 300` is a
shortfall: the ratio 0.5 asks for 600 bt rows (half of 1200) and 300 exist; not an error.

### Run

Same command without `--dry-run`, with `--threads 2` (the default):

```
variant 1/2 t0.5_bt0.5_src_s1234 trained train 33.7 s eval 2.6 s
variant 2/2 t0.5_bt0.5_tgt_s1234 trained train 33.0 s eval 2.6 s
variants: 2 (trained 2, evaluated 0, reused 0)
manifest: $W/out/manifest.json
results.tsv: $W/out/results.tsv
results.json: $W/out/results.json
results.md: $W/out/results.md
```

Wall time 1 min 12.7 s (the whole command, including the two `eole build_vocab`, the two
trainings and the two evaluations), peak resident memory of the largest process 920 MiB (Eole's, the
torch import). The progress lines go to standard output (D6 prints its on standard error). Errors go
to standard error as `error: <message>`.

### Rerun, a lost `result.json`, a damaged checkpoint

The same command again reuses both variants and calls Eole zero times:

```
variant 1/2 t0.5_bt0.5_src_s1234 reused train 0.0 s eval 0.0 s
variant 2/2 t0.5_bt0.5_tgt_s1234 reused train 0.0 s eval 0.0 s
variants: 2 (trained 0, evaluated 0, reused 2)
```

Wall time 0.76 s; `results.tsv` is byte-identical (the tables are rewritten even when everything was
reused). With `variants/t0.5_bt0.5_tgt_s1234/result.json` deleted, the rerun finds the checkpoint
`train.json` vouches for and only evaluates it again (3.37 s, `results.tsv` byte-identical):

```
variant 1/2 t0.5_bt0.5_src_s1234 reused train 0.0 s eval 0.0 s
variant 2/2 t0.5_bt0.5_tgt_s1234 evaluated train 0.0 s eval 2.6 s
variants: 2 (trained 0, evaluated 1, reused 1)
```

With the first variant's checkpoint cut to half its size, the rerun trains that variant again and
reuses the other (37.75 s):

```
variant 1/2 t0.5_bt0.5_src_s1234 trained train 34.4 s eval 2.6 s
variant 2/2 t0.5_bt0.5_tgt_s1234 reused train 0.0 s eval 0.0 s
variants: 2 (trained 1, evaluated 0, reused 1)
```

The retrained checkpoint has the same SHA-256 as the first one
(`e126d00d743a06e04c824018dbab1ea83a66a9f746a83fdf17f92ecc9756a00c`) and every score is
unchanged; only the `train_seconds` cell of `results.tsv` differs (33.7, now 34.4).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--matrix PATH` | required | The ablation matrix |
| `--corpus PATH` | required, repeatable | A corpus Parquet file (the `CORPUS_SCHEMA` table). Its `train` rows train, its `dev` rows score. The same file twice (any spelling or link) is refused: every row would count twice |
| `--spm-model PATH` | required | The SentencePiece model every variant tokenises with; it must hold the direction tag of every target language as one piece |
| `--out-dir PATH` | required | The output directory; may not be the directory of an input file |
| `--threads N` | 2 | OpenMP and MKL threads of each Eole run, at least 1. Part of the fingerprint |
| `--timeout S` | none | Seconds after which each Eole command (the vocabulary, the training, the translation) is killed; a positive, finite number |
| `--force` | off | Train every variant again, ignoring `result.json` and `train.json` |
| `--dry-run` | off | Print the plan and write nothing, not even the directory |
| `--prepare-only` | off | (D8) Write each variant's corpora, validation files and `config.yaml` exactly as a run would, train and score nothing, print `prepared <id>: <config.yaml>` per variant; the manifest is of stage `train_prepare`, which `ablate` and `--prepare-only` refuse in each other's output directory. `--force` and `--timeout` are refused with it, `--threads` is only recorded, `--dry-run` wins (runbook 08) |

The flow, in order: reject a `--timeout` that is not positive and finite and a repeated `--corpus`; hash
the matrix, load it, refuse a foreign manifest and an `--out-dir` that is an input's directory (both
also on a dry run); hash the base config, the corpora and the SentencePiece model (each input is
hashed BEFORE it is read, so the manifest vouches for the bytes the run started from); read the rows
and plan (where the config and the SentencePiece model are checked); on `--dry-run` print and stop;
create the output directory (before any paid work); remove the old manifest (`begin_commit`, so a
failure part-way leaves no manifest that vouches for old and new outputs mixed); run the variants;
write the tables; write the new manifest last.

The manifest (`rinchen.manifest/1`, stage `train_ablate`) records: `inputs` (the matrix as given, the
base config, each corpus and the SentencePiece model, with sha256 and size), `outputs` (the three
tables, then each variant's rendered `config.yaml` and its checkpoint `run/ckpt/model.00.safetensors`
with the digest the variant recorded), `config_sha256` (the matrix file's hash), `seed` (the first
seed; `metrics` lists them all), `code_version`, `metrics` (the mixture statistics flattened as
`count.<key>` and `count.dev.<key>`, e.g. `count.rows_kept` 4800.0; `variants.total|trained|evaluated|reused`;
`threads`; `seed.<n>`), `tool_versions` (`eole` 0.6.4, `torch` 2.14.1+cu130, `sentencepiece` 0.2.1,
`sacrebleu` 2.6.0, `polars` 2.1.0 here; `not installed` for a missing package) and `status`
(`{"ablation": "ok"}`). The variant directories and corpus files are resumable work, not outputs;
`train.json` records that work, the checkpoint it names is the output.

### Exit codes

Failures print `error: <message>` on standard error.

| Code | Meaning |
|---|---|
| 0 | OK, including a dry run |
| 1 | An `OSError` creating the output directory or writing an output ("cannot write the ablation outputs under DIR: ..."), or any other `RinchenError` |
| 2 | `ConfigError`: a missing, unreadable or invalid matrix; a missing base config or SentencePiece model, or one that cannot be loaded or lacks a direction tag piece; a language outside the registry; a reserved or unknown override key or any other invalid Eole config; `weight_total` below the number of corpora; a temperature so large that the sizes raised to it overflow; `$` or `[`, `]`, `*`, `?` in the output directory; `--out-dir` the directory of an input file; a repeated `--corpus`; `--timeout` that is not positive and finite. Also a usage error such as `--threads 0` (click's own code) |
| 3 | `DataError`: a corpus file that is missing, unreadable or not a corpus table; no training row, no authentic training row, or no dev row for a direction that has training rows; a null id, language or origin in a row that would be used (a damaged Parquet file that makes the reader panic counts as unreadable); an output directory that holds another stage's manifest |
| 4 | `UpstreamError`: an Eole command that exits non-zero, runs past `--timeout` or is not installed; a training that left no `model.00.safetensors`; a prediction with another number of lines than dev lines; an output line that does not start with the forced tag; an evaluation that scored other dev sets than the plan's. A message about an Eole command names its log (`run/build_vocab.log`, `run/train.log`, `eval/dev.log`) and ends with the log's last 20 lines |
| 129, 130, 143 | SIGHUP, Ctrl-C, SIGTERM: see "Interrupts" |

Real messages (the real binary, `--dry-run`, 2026-11-08):

| Invocation | Exit | Message |
|---|---|---|
| an override `training.train_stepz: 5` | 2 | `error: invalid Eole config: training.train_stepz: unknown key in training` |
| an override `seed: 7` | 2 | `error: invalid overrides: seed: the ablation sets it for every variant, so it cannot be overridden` |
| `tag_placement: [suffix]` | 2 | `error: invalid ablation config ...: 1 validation error for AblationMatrix`, then `suffix placement is not supported: Eole's suffix transform raises a KeyError when training starts unless a workaround corpus is configured; use 'src' or 'tgt'` |
| `weight_total: 5` (8 corpora) | 2 | `error: weight_total 5 is below the number of corpora (8): every corpus needs a weight of at least 1, so weight_total must be at least 8` |
| an override `model.architecture: transformer_lm` | 2 | `error: invalid Eole config: model.architecture: must be 'transformer', got 'transformer_lm': the allow-lists are those of that architecture` |
| `--out-dir 'out[1]'` | 2 | `error: the output directory .../out[1] holds '[', which Eole reads as a glob pattern when it loads a checkpoint from it` |
| the same corpus twice | 2 | `error: --corpus ... names the same file as --corpus ...: it was given twice and would count every row of it twice` |
| a missing corpus file | 3 | `error: corpus file not found: ...` |

## The output directory, fingerprint and resume

```
$W/out/
  data/auth/<direction>.auth.{src,tgt,json}        authentic corpora, shared by every variant
  data/bt/r<ratio>_s<seed>/<direction>.bt.{src,tgt,json}   the bt subsample of a ratio and a seed
  variants/<id>/config.yaml                         the rendered Eole config
                valid.src, valid.tgt                the pre-tokenised validation corpus
                run/build_vocab.log, run/train.log, run/vocab.src
                run/ckpt/step_1000/*, run/ckpt/model.00.safetensors (a link into step_N)
                eval/dev.{log,predict.yaml,tokens.src,tokens.out}, dev.tokens.tgt (tgt only), hyp.<direction>.txt
                train.json                          written once trained
                result.json                         written last: the variant is complete
  results.tsv  results.json  results.md             the tables
  manifest.json                                     written last of all
```

The tiny run left 71 files and 10 links; the largest are, per variant, the checkpoint (3,318,120
bytes since D9's vocabulary invariant, 3,289,000 before), its `optimizer.pt` (1,549,195; 1,490,955
before) and `train.log` (116 to 130 KB), then the authentic corpus files (17 to 34 KB); `results.tsv`
is 521 bytes and `manifest.json` 5,349 (5,443 in the D9 measurement, whose paths were longer).

**The corpus files** are plain text, one sentence per line, LF only, rows sorted by id then source
then target, no tags (the tags are Eole `prefix` configuration, so the placement is a config
difference, not a data difference). The sidecar `<name>.json` holds the row count and the SHA-256 of
both files; a corpus is reused when the sidecar equals what would be written and both files have
those digests, else it is written again, each file whole or not at all (a temporary name and a
rename; the sidecar last).

**The fingerprint** of a variant is the SHA-256 of the canonical JSON (sorted keys, no spaces) of:
the SHA-256 of the rendered config text; of both files of every corpus; of `valid.src` and
`valid.tgt`; of the raw dev source and reference texts by direction (before tokenisation, because
the evaluation scores the raw text); of the SentencePiece model; the thread count; the installed
versions of Eole, torch, SentencePiece and sacrebleu; and the tag placement. Not in it: the code
version (a rerun after a commit must not retrain every variant), the polars version, `--timeout`,
`--force`, the matrix `name`. The weights, the seed, the training steps and every override reach it
through the config text. The config text holds absolute paths, so moving the output directory (or
the repository checkout, or the SentencePiece model) retrains every variant; the checkpoint hashed
in the manifest is the artefact of record. A different `dev_max_lines` or dev text retrains too: the
validation corpus is an input of the training.

**`result.json`** holds the variant's axes and id, its corpora (name, direction, rows, probability,
weight, upsampling), `bt` usage by direction, `source_collision_rate`, `status` (`ok`), `fingerprint`,
`train_seconds`, `eval_seconds`, `checkpoint_file`, `checkpoint_sha256`, and `dev` (per direction
`lines`, `chrf`, `bleu`, and `chrf_mean`, `chrf_min`, `tag_leaks`, `chrf_signature`, `bleu_signature`).
**`train.json`** holds `status` (`trained`), `fingerprint`, `checkpoint_sha256` and `train_seconds`.

**A variant is complete** when its `result.json` parses (strictly: extra keys are ignored, wrong types
are not), says `ok`, holds the current fingerprint, scores exactly the plan's dev sets with as many
lines as each set has, AND the checkpoint file exists with the recorded SHA-256. Nothing in the file is
trusted before that. A complete variant is `reused`. Otherwise:

| Outcome | When |
|---|---|
| `trained` | no valid record: the stale `result.json` and `train.json` are removed, Eole builds the vocabulary and trains (`eole build_vocab`, then the reserved symbols of `spm.symbols` are appended to `vocab.src` when it lacks them, then `eole train`), `train.json` is written as soon as the checkpoint is hashed, then the evaluation runs and `result.json` is written last |
| `evaluated` | `result.json` is missing or not trusted, but `train.json` parses, holds the current fingerprint and the checkpoint has the recorded digest: only the evaluation runs again. A failed or interrupted evaluation therefore does not cost the training |
| `reused` | complete |

`--force` ignores both records and trains every variant again (`trained`). `train.json` stays after a
success; a missing or corrupt `result.json` re-evaluates, a tampered `train.json` or checkpoint
retrains. The progress line of a reused variant reports 0.0 s for both, a `trained` one its training
and evaluation seconds, an `evaluated` one only the evaluation seconds; `result.json` keeps the
seconds of the runs that did the work.

**What a crash leaves.** Before the manifest is removed (a bad matrix, a bad config, a bad corpus, a
foreign manifest): nothing is created, and the previous manifest stands. After it: the old manifest is
gone; variants before the failed one are complete; the failed one has its `config.yaml`, its
validation files, a partial `run/` and, if it got that far, `train.json` and no `result.json`; the
tables of the previous run are NOT rewritten (they are written when every variant is done), so
`results.*` may be older than the variants; no new manifest exists. A rerun of the same command
resumes: finished variants are reused, a trained variant is only evaluated. A hard kill can leave
`*.part` files beside a file that was being written; they are harmless and may be deleted.

**Concurrent runs into one output directory are not guarded.** Run one `rinchen train ablate` at a
time per `--out-dir`.

**Variant directories of an earlier matrix stay.** A second run into the same `--out-dir` with a
different matrix (another temperature, say) leaves the directories of the earlier variants in
`variants/` untouched, checkpoints included (hundreds of MB each at proxy size). They are not
reported, not hashed and not in the manifest: `results.*` and `manifest.json` describe only the
variants of the matrix of the last run. To clean up, delete such a directory by hand, or use a
fresh `--out-dir`.

## Temperature and bt ratio

**bt ratio.** For each direction `bt_used = min(available, round(r * authentic rows))`, with Python's
`round` (a half goes to the even integer: 2.5 gives 2) and `available` the back-translated rows of the
direction. The rows are the ones with the smallest `sha256(f"{seed}\t{id}")` (ties by id, source,
target), so the subsample does not depend on the order of the rows or files. A direction with
back-translated rows and no authentic rows asks for none and gets no corpus (a `bt` corpus is added to
an authentic one); if no direction has authentic rows at all, that is a `DataError`. A shortfall is recorded
(`bt_available`, `bt_requested`, `bt_used`), never an error.

**Temperature.** The mixing probability of a direction is `rows ** T`, normalised over the directions
(`rinchen.utils.sampling.temperature_weights`, D3), where `rows` counts the authentic and the used bt
rows of the direction. T below 1 lifts the small directions; 1 is proportional; above 1 widens the
gap. Inside a direction its probability is shared by its corpora in proportion to their rows, so the
bt share of a direction is `r / (1 + r)` when enough bt rows exist, whatever the temperature:
`bt_ratio` fixes the bt share and `temperature` the balance between directions.

**Integer weights.** Eole's `weight` is the number of consecutive examples taken from a corpus per
round of a cyclic round robin (`D-EOLE-WEIGHT-SEMANTICS`): the share is `w_i / sum(w)`, independent of
the corpus size, and an exhausted corpus restarts. `integer_weights` therefore turns the
probabilities into integers: every weight is at least 1, the weights sum to `weight_total` (a
`ConfigError` naming the number needed when there are more corpora than that), by largest-remainder
rounding (floor each `total * p`, with a minimum of 1; give the missing units to the largest
remainders; if the minimum overshoots, take units one at a time from the largest weight, never from
a 1); ties go by corpus name, never by input order, and a larger probability never gets a smaller
weight.

**The `upsampling` column.** `upsampling = (w / sum of the weights) / (rows / sum of the rows)`:
1.0 means the corpus is drawn as if every row of the mixture counted alike, more than 1 means its
rows are seen more often than that (the corpus is repeated), less than 1 less often. It is not an
epoch count (an epoch count needs the batch composition, which Eole decides). In the toy run the
small Russian corpora are upsampled 1.2 and 1.12, the English ones 0.88 and 0.96.

Three illustrative size mixes (computed with the repository's own `mixing_probabilities` and
`integer_weights`, `weight_total` 100; the sizes are made up).

Mix A: five directions of 1M, 8M, 1M, 0.6M and 0.6M rows, no bt corpus:

| corpus | rows | T=0.3 p / w / upsampling | T=0.5 p / w / upsampling | T=1 p / w / upsampling |
|---|---:|---|---|---|
| d1 | 1,000,000 | 0.1792 / 18 / 2.02 | 0.1568 / 16 / 1.79 | 0.0893 / 9 / 1.01 |
| d2 | 8,000,000 | 0.3343 / 34 / 0.48 | 0.4435 / 44 / 0.62 | 0.7143 / 72 / 1.01 |
| d3 | 1,000,000 | 0.1792 / 18 / 2.02 | 0.1568 / 16 / 1.79 | 0.0893 / 9 / 1.01 |
| d4 | 600,000 | 0.1537 / 15 / 2.80 | 0.1215 / 12 / 2.24 | 0.0536 / 5 / 0.93 |
| d5 | 600,000 | 0.1537 / 15 / 2.80 | 0.1215 / 12 / 2.24 | 0.0536 / 5 / 0.93 |

(The spike's smallest-sum integer vector for T=0.3, 7/13/7/6/6, has nearly the same shares but sums
to 39; `weight_total` fixes the sum.)

Mix B: the same directions, each with a bt corpus of half its rows (bt ratio 0.5), ten corpora in the
order d1.auth, d1.bt, d2.auth, d2.bt, ..., d5.bt (rows 1M, 0.5M, 8M, 4M, 1M, 0.5M, 0.6M, 0.3M, 0.6M,
0.3M); the weights:

| T | p of d1.auth, d2.auth, d4.auth | weights | upsampling of d1.auth / d2.auth / d4.auth |
|---|---|---|---|
| 0.3 | 0.1194, 0.2229, 0.1025 | 12 6 23 11 12 6 10 5 10 5 | 2.02 / 0.48 / 2.80 |
| 0.5 | 0.1045, 0.2957, 0.0810 | 11 5 30 15 10 5 8 4 8 4 | 1.85 / 0.63 / 2.24 |
| 1.0 | 0.0595, 0.4762, 0.0357 | 6 3 48 24 6 3 3 2 3 2 | 1.01 / 1.01 / 0.84 |

Rounding shows already at ten corpora: d1.auth and d3.auth are the same size and get 11 and 10 at
T=0.5 (the tie goes to the earlier name), and the 2:1 ratio of an authentic corpus to its bt corpus
becomes 3:2 for d4 at T=1.

Mix C: sixteen directions (Mongolian to and from eight partner languages with 8M, 4M, 3M, 1M, 1M,
0.8M, 0.5M and 0.2M rows each way) with a bt corpus each (bt ratio 0.5): 32 corpora, 55.5M rows,
the smallest corpus 100,000 rows. The minimum weight of 1 matters when the weights are few:

| `weight_total` | T | corpora at the minimum weight 1 | largest weight | share of the smallest corpus (p) | worst relative error of a share |
|---|---|---:|---:|---|---:|
| 100 | 0.3 | 4 of 32 | 7 | 0.0100 (0.0113) | 0.33 |
| 100 | 0.5 | 8 of 32 | 9 | 0.0100 (0.0070) | 0.42 |
| 100 | 1.0 | 16 of 32 | 14 | 0.0100 (0.0018) | 4.55 |
| 200 | 0.3 | 0 of 32 | 14 | 0.0100 (0.0113) | 0.12 |
| 200 | 0.5 | 2 of 32 | 18 | 0.0050 (0.0070) | 0.29 |
| 200 | 1.0 | 8 of 32 | 29 | 0.0050 (0.0018) | 1.78 |

At T=1 and `weight_total` 100 half the corpora sit at weight 1 and the smallest one is drawn 5.5 times
its proportion (0.01 against 0.0018): the temperature axis is coarse at its tails when there are many
corpora. Raise `weight_total` (the spike advised a sum of about 100 or less and well below
`bucket_size`; larger sums were not run) or accept that the smallest corpora are floored. The weights at
T=0.3, in the order of the directions khk-eng, eng-khk, khk-rus, rus-khk, khk-zho, zho-khk, khk-kor,
kor-khk, khk-jpn, jpn-khk, khk-deu, deu-khk, khk-tur, tur-khk, khk-kaz, kaz-khk (each `auth` then
`bt`): with `weight_total` 100, 7 3 7 3 5 3 5 3 5 3 5 3 4 2 4 2 4 2 4 2 3 2 3 2 3 1 3 1 2 1 2 1; with
200, 14 7 14 7 11 6 11 6 10 5 10 5 7 4 7 4 7 4 7 4 7 3 7 3 6 3 6 3 4 2 4 2. The guard fails loudly
below one weight per corpus and does not warn about coarseness above that. A temperature so large
that the sizes raised to it do not fit a float is a `ConfigError` naming the variant.

## Tag placement: notes for operators

The direction tag of a corpus is the tag of its target language (`<2khk_Cyrl>`, `<2eng_Latn>`,
`<2rus_Cyrl>`, ...; the registry in `configs/languages.yaml`), one SentencePiece piece. A
back-translated corpus also carries `<bt>`.

| Placement | Training (Eole `prefix` transform) | Prediction (the harness) |
|---|---|---|
| `src` | `src_prefix: '<2xx>'` (`'<2xx> <bt>'` for a bt corpus), `tgt_prefix: ''` | the source line starts with `<2xx>` as one piece |
| `tgt` | `src_prefix: ''` (`'<bt>'` for a bt corpus), `tgt_prefix: '<2xx>'` | the tag is FORCED as the start of the output: a `tgt` file with the tag on every line plus `tgt_file_prefix: true`, the source is untagged; the output then starts with the forced tag, which the harness checks and strips |

**`src` is the primary placement** (spike): no forcing, no dedup loss, a validation loss that compares
across variants. **`tgt` trains, but** it needs three things the harness does and an operator must not
undo: (1) the tag must be forced at prediction, because without it an ambiguous direction fails (in
the spike mn to ru scored chrF 16.0 and mn to en 80.4 against 98.6 and 99.95 forced); the first token of
every output line is checked and a missing forced tag is an `UpstreamError` naming the direction and
the line, never the text; (2) the leading forced tag is stripped before decoding; (3) the validation
loss includes the ambiguous tag (valid accuracy 96.8 against 99.9 for `src` in the spike), so
Eole's `valid` numbers are not comparable across placements; compare the dev chrF.

**Same-source minibatch dedup hazard (`D-EOLE-BATCH-DEDUP`).** Eole silently drops a training example
whose tokenised source equals one already in the same minibatch. With the tag on the target the source
carries no tag, so a sentence that exists in several directions (multi-parallel data: the same
Mongolian source translated into English and Russian) collides with itself. The spike measured, on
multi-parallel toy data, 0.00% dropped for `src` and, for `tgt`, 1.94% at 1024-token batches, 35.6% at
4096 and 50.0% at 16384. Each variant records `source_collision_rate`, the share of kept training rows
whose source text occurs with another target language (one number for the run, before any bt
subsample, repeated on every row of `results.tsv`, printed with four decimals); it is the exposure to
the hazard. It is informational, not a gate: read it before trusting a `tgt` result at a large batch
size, and expect it to be 0.0 on the toy training set (the English and Russian toy sentences are
disjoint content; the gate does not exercise the hazard).

**A source suffix is not supported.** It needs Eole's separate `suffix` transform, which raises
`KeyError: ''` when training starts unless a workaround corpus with an empty suffix is configured
(`D-EOLE-SUFFIX-KEYERROR`, a bug in the installed Eole 0.6.4). `tag_placement: [suffix]` is a
`ConfigError`.

**`<bt>` goes after the direction tag.** With `src`: `<2khk_Cyrl> <bt>` (Eole splits the prefix on a
space into two pieces, both in the vocabulary). With `tgt`: `<bt>` alone as the source prefix and the
direction tag as the target prefix. The SentencePiece model must have `<bt>` and every direction tag
as single pieces (the project's `train_spm` makes them user-defined symbols); a tag written inside the
text of a corpus would get a standalone `▁` before it, so the corpus files carry no tags at all.

**The validation corpus** is one concatenation of the dev sets of all directions (directions in sorted
order), PRE-TOKENISED by the harness with the tag as one piece, `transforms: []`, no prefix keys:
that works for both placements and leaks nothing into the checkpoint (Eole copies the `valid` prefix
into the checkpoint's `config.json`, where it overrides the prediction; `D-EOLE-PREFIX-OVERRIDE`).
The STOP-gate test reads the saved `config.json` of both checkpoints and asserts that `data.valid` has
no `src_prefix`, `tgt_prefix`, `src_suffix` or `tgt_suffix` and no transforms.

## Reading the results

`results.tsv`: a header and one row per variant in plan order. Columns: `variant`, `temperature`,
`bt_ratio`, `tag_placement`, `seed` (the axes as printed in the id), then for each direction, in sorted
order, `chrf_<direction>` and `bleu_<direction>` (interleaved: chrF then BLEU), then `chrf_mean` (the
macro mean over directions: each direction counts once whatever its size), `chrf_min` (the lowest
direction), `tag_leaks`, `train_seconds` and `source_collision_rate`. Scores and means are fixed at two
decimals, seconds at one, the collision rate at four.

`tag_leaks` counts hypotheses, over all directions, that contain a direction tag or `<bt>` as text; a
model that has learned the tags as text would show it. It is 0 for both toy variants.

`results.md`: the same main columns for each variant, then "Summary over the seeds": one row per
`(temperature, bt_ratio, tag_placement)` with the number of seeds and the mean of `chrf_mean` and of
`chrf_min` over them. In the spike's seed sweep (four seeds, 1024-token batches) the step at which a
model first reached chrF 60 in every direction varied from 800 to 1200, so do not read a difference
between two variants from one seed.

`results.json`: everything, unrounded: the matrix, `threads`, both statistics and, per variant, the
`result.json` document above. The scores are corpus chrF2 and BLEU (13a tokeniser) of sacrebleu 2.6.0
(n-gram counts added over the whole direction), signatures
`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0` and
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, of greedy translations of the dev lines,
clamped at 100.0 (sacrebleu reports 100.00000000000004 for identical text); a corpus whose references
are all empty scores 0.0.

**The toy numbers rank nothing.** The two variants of the gate, one seed each, 100 dev lines per
direction. As D7 recorded them on 2026-11-08 (code `f7d1bd0`, before the vocabulary invariant):

| variant | eng-khk | khk-eng | khk-rus | rus-khk | chrf_mean | chrf_min |
|---|---:|---:|---:|---:|---:|---:|
| `t0.5_bt0.5_src_s1234` | 98.89 | 99.23 | 97.76 | 96.25 | 98.03 | 96.25 |
| `t0.5_bt0.5_tgt_s1234` | 99.72 | 98.24 | 98.32 | 98.33 | 98.65 | 98.24 |

(BLEU 98.85, 99.01, 98.60, 96.19 for `src` and 99.67, 98.01, 98.79, 98.43 for `tgt`.) After D9 added
every reserved symbol to the vocabulary (the vocabulary grew from 232 to 344 entries, the checkpoint
from 3,289,000 to 3,318,120 bytes, so the initialisation and every number moved), the same command
on 2026-11-09 (code `c2d46b5`, one run on a loaded machine) gave:

| variant | eng-khk | khk-eng | khk-rus | rus-khk | chrf_mean | chrf_min |
|---|---:|---:|---:|---:|---:|---:|
| `t0.5_bt0.5_src_s1234` | 95.94 | 95.85 | 90.89 | 93.67 | 94.09 | 90.89 |
| `t0.5_bt0.5_tgt_s1234` | 99.45 | 99.16 | 98.13 | 98.24 | 98.75 | 98.13 |

(BLEU 95.91, 95.54, 93.11, 93.41 for `src` and 99.52, 99.01, 98.96, 98.01 for `tgt`; no tag leak; 135.7 s
wall on a machine loaded by other work, `train` 75.1 s and 50.9 s.) The source-tag variant moved by up
to 5.9 points (khk-rus 97.76 to 90.89), the target-tag variant by less than 1.5; both stay far above the
gate's 60. The model memorises a templated grammar of 37 nouns; a 0.6 point difference of means on
one seed said nothing about the placement and a 4.7 point one now says nothing either: the same
variant moved 7 points by a change of vocabulary alone. The only claims these numbers support are the gate's: every direction
of both variants is above chrF 60, the tags select the right language, no tag leaks, and the machinery
runs end to end.

## Threads, determinism and timing

`--threads` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` for every Eole command and enters the
fingerprint. The same seed AND the same thread count give byte-identical checkpoints and predictions
(the spike: the same safetensors SHA-256 across reruns, across different `train_steps` and
`valid_steps` and under machine load, so step N of a long run equals the final checkpoint of an
N-step run); different thread counts do NOT (the spike's reruns: 4 threads gave another hash and chrF
of 64.8, 63.3, 53.4 and 38.9, 1 thread 47.2, 61.4, 45.5 and 38.9, on a short toy run). Measured here:
after truncating a checkpoint, the retraining reproduced the first checkpoint's SHA-256
(`e126d00d...a00c`) and every score (the Task 8 implementer and the review saw the same). Pin the
threads and use the same number for a rerun. The GPU is hidden from Eole (`CUDA_VISIBLE_DEVICES=""`)
when the config's `training.gpu_ranks` is empty, and always for the evaluation: the translation runs on
the CPU even when the training used a GPU, which for the proxy model was not timed. Determinism on a
GPU was not examined. Timing on the toy set (2 threads): `eole build_vocab` 5 to 6 s, 1000 steps of
2048-token batches 28 to 34 s, evaluation of 400 lines 2.6 s.

**Initialisation.** Eole's default `param_init_method: uniform` with `param_init` 0.1 initialises every
parameter, layer-norm gains and biases included, uniformly in -0.1..0.1; a model trains from it but
shows a plateau until about step 600 to 1000 (1024-token batches) with a seed-dependent breakout.
`xavier_uniform` converged much faster on the toy (worst direction 97.9 over four seeds at step 300,
2048-token batches). It is NOT adopted: the effect on a real-size model is untested. `param_init_method`
is an allowed key, so an ablation can try it by override.

## Interrupts

Ctrl-C, SIGTERM and SIGHUP end the run, kill Eole's whole process group (it starts data-loader workers;
`run_logged` puts the command in a session of its own) and exit with 130, 143 and 129 (128 plus the
signal number). Measured through the real binary 4 s into `eole train`, without `--timeout`: Ctrl-C
130 in 0.17 to 0.36 s, SIGTERM 143 in 0.26 s, SIGHUP 129 in 0.16 s, with no `eole` process of the run
left, no manifest and no `.part` file (`D-POLARS-SIGINT`, `D-LOGGED-TERM` in `docs/VERIFY.md`; before
the fixes Ctrl-C without `--timeout` was raised only when `eole train` ended, 25.9 s after the signal,
and SIGTERM left `eole train` running). `--timeout` bounds each Eole command and kills its group when
it runs out (exit 4). A rerun after an interrupt resumes. A signal that was ignored when `rinchen`
started stays ignored (see "Running a long training detached").

**SIGKILL cannot be caught.** After `kill -9` of `rinchen` (or the out-of-memory killer) the child
`eole train` may keep running with nobody to read its result; look for it with `pgrep -f "eole train"`
before starting the next run, or two trainings will write into one directory (`D-LOGGED-TERM`).

### Running a long training detached

A multi-hour run is usually left running over SSH with `nohup` or `setsid`:

```
setsid nohup .venv/bin/rinchen train ablate --matrix M.yaml --corpus C.parquet \
    --spm-model S.model --out-dir OUT > OUT.stdout 2> OUT.stderr < /dev/null &
```

- `nohup` starts the command with SIGHUP ignored. `rinchen` leaves a signal that was ignored when it
  started alone while it waits for Eole: an inherited ignore is a request not to die on that signal,
  so a closed terminal does not end the run. Measured through the real binary, a SIGHUP 4 s into
  `eole train` of a `nohup` run: the run went on and finished with exit 0, the manifest and the three
  tables written, 9.55 s after the signal. (Before this was fixed, the handler that turns SIGHUP into
  an exit replaced the ignore and the same hangup ended the run with 129 after 0.17 s and killed
  `eole train`.)
- Without `nohup` (SIGHUP at its default) SIGHUP and SIGTERM stop the run cleanly: exit 129 and 143,
  Eole's whole process group killed, the variants that were finished kept, no manifest (a rerun
  resumes). Measured: SIGHUP 129 after 0.18 s, no process of the run left.
- SIGKILL leaves the `eole train` process running (known limit, `D-LOGGED-TERM`).
- A shell background job (`command &` in a script or any shell without job control) starts with SIGINT
  ignored (POSIX), and `rinchen` keeps it that way: Ctrl-C style SIGINT, `kill -INT`, does nothing to
  such a run. Measured: a SIGINT 4 s into `eole train` of a run started with `&` from a script was
  not noticed and the run finished (exit 0, 9.56 s later). Stop such a run with SIGTERM or SIGHUP.

## Adding an axis

1. `config/ablation.py`: add the field to `Axes` (default, distinct values, strict types) and to the
   count in `AblationMatrix._expansion_fits_max_variants`; `train/variants.py`: add it to `Variant`,
   the id and the `itertools.product` (its order is the order of the rows).
2. Make it act: on the mixture in `ablation_plan._plan_variant` and `corpus_files`, or on the Eole
   config in `ablation_config.variant_overrides` (the plan and the preparation share that one builder,
   so the dry run checks what the run writes).
3. Make it reach the fingerprint: a changed corpus byte or config text already does; anything else (a
   new input file) goes into `fingerprint_inputs` in `ablation_prepare.prepare_variant`.
4. Report it: `_AXIS_COLUMNS`, `_axis_cells` and the key of `_seed_summary` in `ablation_report`, the
   variant line of `cli/train._plan_lines` (`results.json` takes it from `asdict(variant)`).
5. Tests: `test_config_ablation.py`, `test_train_variants.py`, `test_train_ablation_plan.py`,
   `test_train_ablation_results.py`, `test_cli_train.py`, with `tests/support/fake_ablation.py` and
   `train_cli.py`; break the code once to see each test fail.

## Open gates

Not done by D7, because no real data, GPU or reader exists: any ablation on real corpora; a measured
proxy config (its model size, batch, steps and learning rate are priors); the throughput and memory of
a real training; Eole-native prediction and the CTranslate2 export of a tag-on-target model; the D3
gates assigned to D4 and the D5 gates that need real text stay open as recorded in their runbooks.

## What was not verified

- **No real corpus and no GPU.** Every run was on the CPU, on at most 5,200 toy rows, with the GPU
  hidden. The proxy base and the priors matrix were only validated against the schema (Eole accepts
  the files); no variant of them was trained.
- **The mixture build beyond 4,000,000 rows.** Measured, a fresh process each: a file of 1,000,000
  synthetic rows (8 directions, about 175 characters a side, 902,400 kept as training rows, 294.6 MB of
  Parquet): `load_training_rows` 0.32 s and a peak of 1,550 MiB; `load_dev_rows` 0.06 s and 449 MiB;
  `planned_corpus_sizes` 0.01 s (after the load); `build_corpora` 2.27 s and 1,661 MiB, writing 634
  MB, and again on the unchanged directory (only the digests are checked) also 2.27 s;
  `plan_ablation` for 12 variants 0.61 s and 1,610 MiB. At 4,000,000 rows: `load_training_rows`
  1.09 s and 4,758 MiB, `build_corpora` 11.43 s and 5,127 MiB, 2.7 GB written. The training loader
  holds the kept rows of all directions in one polars frame, so memory grows with the rows, about 1.05
  GiB for each further million of this text length; extrapolated from the two points, 46 GiB would be
  reached near 44M rows and 100M rows would need about 105 GiB (neither run). Hundreds of millions of
  rows need a streaming loader, which D7 does not have.
- **The weight resolution** for many directions is shown above by computation only; no training with
  32 corpora and `weight_total` 100 or 200 was run, and the effect of the floored small corpora on a
  model is unknown. That the sum of the weights must stay well below `bucket_size` is from reading
  the code (`D-EOLE-WEIGHT-SEMANTICS`).
- **`num_workers` above 1 with weights**, weight `None`, GPU or bf16 runs, dropout above 0,
  learning-rate or warmup tuning, larger models: not tested (the spike's list). `train_from` is not
  an ablation key; D8 tested resuming a run's own checkpoint (runbook 08) and D9 continuing our own
  teacher (runbook 09).
- **`xavier_uniform` at scale**: toy only.
- **CTranslate2 export and Eole-native prediction of a tag-on-target model**: not tested (the
  harness predicts from pre-tokenised lines with the tag forced). D5's `export_ct2` fails for an
  output directory with glob characters (`D-EOLE-GLOB-CKPT`); D5 was not changed.
- **SIGKILL** of `rinchen`: the child is not cleaned up (above). `rinchen train launch` (D8) holds a
  lock that the Eole it started keeps, so a second launch is refused while an orphan trains (runbook
  08); `ablate` has no such lock.
- **Concurrent runs into one output directory are not guarded.**
- **Only one seed** in the gate and in every measurement here; the seed-to-seed spread of a variant on
  a real corpus is unknown.
- **Windows and macOS were not run.**
- **The real binary** was run for `--help`, a dry run, the tiny run, its three reruns, failing
  invocations and the signals; every other path ran in process with fakes at the `train` and
  `evaluate` seams.

### Known and left open by the reviews

Findings of the reviews that were parked and not fixed, one line each:

- Value TYPES of known keys are not validated (`train_steps: "ten"` passes the dry run and fails in
  `eole train`); the shipped-config tests build Eole's own `TrainConfig` from the files.
- A mapping with a non-text key is caught only by `dump_config`; `merge()` called outside `render` has
  no guard against a self-containing mapping.
- Eole's own `filtertoolong` drops are not counted in the mixture statistics.
- The SentencePiece model is loaded twice per evaluation (the token lines, then the decoding).
- `script_conv/data.py` keeps a private `_write_lines` beside `eole_run.write_lines`;
  `cli/convert_script.py` still imports through the `script_conv` re-exports.
- D7 prints its progress on standard output and removes the old manifest before the first variant;
  D6 prints on standard error and removes it right before the first output.
- The fingerprint holds absolute paths through the config text (moving a run retrains it).
- `source_collision_rate` is one number for the run, repeated on every row.
- The plan document (`docs/superpowers/plans/2026-11-08-d7-multilingual-ablations.md`) still says
  "effective epochs" where the design says `upsampling`.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux
7.0.0-42-generic, CPython 3.14.10, eole 0.6.4, torch 2.14.1+cu130, sentencepiece 0.2.1, sacrebleu
2.6.0, polars 2.1.0; the GPU (RTX 5070 Ti) was hidden from every Eole run. All on 2026-11-08, with
`/usr/bin/time -v`, once each.

| What | Value |
|---|---|
| The tiny run (2 variants, 1000 steps, 2 threads) | 72.7 s wall; `train` 33.7 and 33.0 s, `eval` 2.6 s each; peak RSS of the largest process 920 MiB |
| The dry run | 0.71 s; 215 MiB |
| A rerun that reuses both / evaluates one / retrains one | 0.76 s / 3.37 s / 37.75 s |
| Weights of the tiny run | 22 6 22 6 15 7 15 7 = 100 over 1200, 300, 1200, 300, 600, 300, 600, 300 rows |
| The gate | dev chrF at least 60 in every direction of both variants (measured 96.25 to 99.72 on 2026-11-08; 90.89 to 99.45 on 2026-11-09, after D9's vocabulary invariant) |
| The tiny run again, 2026-11-09 (code `c2d46b5`, loaded machine) | 135.7 s wall; `train` 75.1 and 50.9 s, `eval` 4.2 and 4.0 s; checkpoint 3,318,120 B, `optimizer.pt` 1,549,195 B, `vocab.src` 339 lines in both variants; 71 files and 10 links |

The mixture probe is under "What was not verified". The measurement scripts were read-only scratch
files (`build_inputs.py`, `weights_tables.py`, `gen_1m.py`, `measure_mixture.py`) run with `python -I`.

## Where the code lives

`src/rinchen/train/` (D8 added the launcher modules, runbook 08), `src/rinchen/config/ablation.py`,
`src/rinchen/cli/train.py`, `src/rinchen/utils/logged_process.py`,
`src/rinchen/metrics/bleu_chrf.py` (`corpus_scores`),
`configs/eole/{multilingual_tiny,ablation_tiny,multilingual_proxy,ablation_priors}.yaml`; unit and
property tests named `test_train_*`, `test_cli_train.py`, `test_config_ablation.py`,
`test_eole_configs_shipped.py`, `test_metrics_corpus_scores.py`, `test_utils_logged_process.py`;
`tests/integration/test_train_evaluate.py` and `test_train_tiny_ablation.py` (the STOP gate); and the
helpers in `tests/support/` (`tiny_parallel.py`, `tiny_variant.py`, `train_cli.py`,
`fake_ablation.py`, `corpus_rows.py`, `eole_train_config.py`).
