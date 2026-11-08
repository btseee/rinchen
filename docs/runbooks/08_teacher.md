# 08 - Teacher training

## Scope

D8 builds what a long Eole training needs around it: `rinchen train launch` (start a run, resume it
after any interruption, recognise how it ended, write the script an operator runs by hand),
`rinchen train average` (the mean of the last checkpoints), `rinchen train ablate --prepare-only`
(the corpora, the validation corpus and the Eole config of a variant, made by the same function the
ablation uses, without training), the modules behind them (checkpoint completeness and run identity,
training-log curves and a loss gate, the GPU rules, the run lock), and two configs:
`configs/eole/teacher.yaml` (the 8-GPU prior) and `configs/eole/teacher_tiny.yaml` (the CPU gate).
The design, with the rulings that fill the gaps of the master plan and what each costs if wrong, is
`docs/superpowers/specs/2026-11-08-d8-teacher-training-design.md` (amended to the finished code, with
a section listing each ruling and decision made while building); the decisions are condensed in the
2026-11-08 entry "D8 teacher training" of `DECISIONS.md`; the upstream facts are in
`docs/VERIFY.md` (section "D8 findings"). The Eole facts come from a spike made before the design
(`.superpowers/sdd/2026-11-08-d8-teacher-training/spike-findings.md`; Eole 0.6.4, CPU, toy data).

**What D8 is.** A tested launcher. Given a validated Eole config and a run directory it builds the
vocabulary if its file is missing, trains, and after any interruption (a kill, a reboot, Ctrl-C, a
crash) starts again from the newest COMPLETE checkpoint of the same run, refusing to resume a run whose
config changed, and recognises a finished run, an early stop, a NaN. It writes `command.sh`, so an
operator can run the same commands by hand (the 8-GPU run), and a manifest of what it did. The
averager makes one model from the last checkpoints.

**What D8 is NOT.** Nothing below is a statement about the quality of any teacher.

- **No GPU was used.** This machine has a GeForce RTX 5070 Ti that the owner did not authorise for a
  run. Every Eole process of D8 (the spike, the tests, this runbook's measurements) ran on the CPU
  with `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`. NCCL, bf16 and fp16, flash attention, a real
  multi-GPU run, multi-node and the write of a 13 GB checkpoint are UNMEASURED; every statement about
  them in this file is read from Eole's source or from a CPU gloo shim and is tagged so. A one-GPU
  smoke test needs the owner's say-so (see "First contact with a GPU").
- **No real data.** The tiny teacher trains on the templated toy corpora of
  `tests/support/tiny_parallel.py`; its loss and perplexity prove that the launcher, the resume and the
  averaging work, nothing else. No real corpus exists.
- **No teacher.** `configs/eole/teacher.yaml` is a table of unmeasured priors (below); nothing was
  trained, timed or tuned with it. Eole's schema accepts the file, a test shows that, nothing more.
- **Task 7 was skipped.** The plan's optional test of the multi-process path (two ranks on the CPU
  through a gloo shim) was not written. The launcher's classification of a multi-process early stop
  (exit code 1 with `Early Stop!` in the log and a complete final step) is covered by unit tests that
  fake the log and the exit code the spike observed; the one real observation is the spike's
  (below).
- **No fine-tuning.** `training.train_from` exists only in the resume config the launcher renders for
  the run's own newest complete step directory; it is refused everywhere else. D9 (continued training
  of our own teacher) will need a conscious extension of the identity rule.
- **No export, serving or evaluation suite** (D11 to D13).

Tags used below: **measured** (run for this runbook or by a test of the build), **spike** (run in the
spike, on the CPU), **source-read, unmeasured** (read from Eole's source, never run), **ARITHMETIC**
(computed from a formula, not measured). Every number is from a test, the code, the spike, the ledger
of the build (`.superpowers/sdd/2026-11-08-d8-teacher-training/progress.md`) or a measurement made for
this runbook on 2026-11-08 ("Measured numbers"); a number that was not obtained is not given.

What was checked: the unit and property tests (every new test mutation-checked by hand during the
build), `tests/integration/test_train_tiny_teacher.py` (the STOP gate: 30 tests collected from 27 test
functions through the real `rinchen` binary and the real Eole on the CPU; 28 from 25 before the final
review's fixes), two reviews (of Tasks 1 to 6, whose findings an interlude fixed; of the whole branch,
whose findings one fix wave handled, "The final review" below), and the real binary for this runbook.

## Architecture

| Module | Job |
|---|---|
| `rinchen/train/checkpoints.py` | `StepDirectory`, `ResumePoint`, `step_directories`, `latest_complete`, `step_of`; `is_complete` (completeness of one step directory); `architecture_mismatch`, `ArchitectureMismatch` (the model-shape guard); `identity_of`, `RunIdentity`, `write_launch_record`, `read_launch_record` (`launch.json`) |
| `rinchen/train/curves.py` | `parse_train_log` (an Eole log to a `TrainCurve` of `TrainReport` rows) and `loss_gate`; reads no file |
| `rinchen/train/average.py` | `average_checkpoints` to an `AveragedModel`; `AVERAGED_FROM_FILE` |
| `rinchen/train/launch.py` | `launch` to a `LaunchResult`; `latest_attempt_log`; `LAUNCH_DIR`, `COMMAND_FILE`, `LOCK_FILE` |
| `rinchen/train/gpu_rules.py` | `check_gpu_rules` to `GpuSettings`; `MIN_MULTI_PROCESS_TIMEOUT_S` (3600), `MAX_TRAINING_CORPORA` (80) |
| `rinchen/train/launch_command.py` | `render_command_sh`: the text of `command.sh`, a pure function |
| `rinchen/train/run_lock.py` | `holding_the_run` (exclusive non-blocking `flock`), `refuse_a_held_run` |
| `rinchen/train/eole_cfg.py` (extended) | the new allow-list keys, `RESUME_TRAINING_KEYS`, `NORMALIZATIONS`, the value rules of the teacher keys, the `resumes` flag of `validate_train_config`, `merged_config` and `render` |
| `rinchen/train/eole_run.py` (extended) | `model_path_inside` made public (it was private) |
| `rinchen/train/corpus_files.py` (extended) | `corpus_sidecar_path`, for the manifest of `--prepare-only` |
| `rinchen/utils/logged_process.py` (extended) | `run_logged(..., inherit_fds=)`: descriptors the child keeps (the lock) |
| `rinchen/utils/integers.py` | `is_strict_int`: an `int` that is not a `bool` |
| `rinchen/cli/train.py` (extended) | `ablate` with `--prepare-only`; registers `launch` and `average` |
| `rinchen/cli/train_runs.py` | `launch`, `average`, their manifests |
| `rinchen/cli/train_common.py` | the checks and exits the three commands share |
| `configs/eole/teacher.yaml`, `teacher_tiny.yaml` | the 8-GPU prior and the CPU gate |

Test helpers in `tests/support/`: `step_dirs.py` (fabricated checkpoint directories with real small
safetensors and zip files, and the damage helpers), `fake_launch.py` (`FakeEole`, a stand-in for
`run_logged` that writes Eole-format logs and step directories), `fake_eole_process.py` (an executable
stand-in for the tests that run `command.sh` with bash), `launch_cli.py` (the CLI workspace),
`signal_handlers.py` with the `default_signal_handlers` fixture of `tests/conftest.py` (signal tests no
longer depend on how pytest was started).

The layering: `train` never imports `script_conv` or `cli`; `launch` imports `eole_cfg`, `eole_run`,
`checkpoints`, `curves`, `gpu_rules`, `launch_command`, `run_lock` and `utils.logged_process`;
`safetensors` (a dependency of Eole, locked in `uv.lock` at 0.8.0, not declared in `pyproject.toml`)
is imported where it is used (`checkpoints._weights_problem`, the check in `average`), and `torch`
and `eole` are never imported by D8's code: Eole runs only as a subprocess of the `eole` script next to
the interpreter.

## The launch state machine

### Order of work

`launch(cfg, run_dir, ...)` does, in this order (the first refusal ends it):

1. Read the config and run `validate_train_config` (the D7 allow-lists plus the D8 keys). The config
   must give a positive integer `training.train_steps` (Eole's default of 100000 would otherwise
   apply without a word), a `src_vocab` file name, and a `training.model_path` inside `run_dir`
   without a `..` component and without `[`, `]`, `*`, `?` (`D-EOLE-GLOB-CKPT`) or `$`.
2. Lay the `--world-size` and `--gpu-ranks` overrides over the config, spell both out, and apply the
   GPU rules (below). The result is the EFFECTIVE config.
3. Take the identity of the run (`identity_of` of the effective config).
4. Take the run lock (`launch/lock`); `--write-only` only checks that nobody holds it.
5. Look at the checkpoint directory: the newest complete step (`latest_complete`), the incomplete
   ones above it. Enforce the identity (below), refuse a NaN left by the last attempt, refuse a
   `train_steps` below the newest complete step, and return `already_complete` or `early_stopped` for
   a finished run.
6. Call the `before_writing` hook (the CLI drops the old manifest here), write this attempt's files
   (`launch/train-<k>.yaml`, `command.sh`) and `launch.json`.
7. With `--write-only`: stop. Otherwise build the vocabulary when the `src_vocab` file is missing, run
   `eole train`, and decide from its log and the step directories how it ended.

### Outcomes

A launch is **fresh** when the checkpoint directory holds no step directory, a **resume** when it holds a
complete one (the newest complete step is the resume point), and **finished** when step `train_steps` is
complete or the last attempt early-stopped at the newest complete step. A fresh launch and a resume both
run Eole; a finished run runs nothing.

| Status | When | Files written |
|---|---|---|
| `completed` | `eole train` exited 0 and step `train_steps` is saved and complete | the attempt's config, log, `command.sh`, `launch.json` (if new), the step directories, the manifest |
| `early_stopped` | the log says `Early Stop!` and the step saved right after it is the newest complete one, whatever the exit code | as above |
| `already_complete` | a complete step `train_steps` exists, or the last attempt early-stopped at the newest complete step | nothing, but the manifest when it is missing or stale |
| `written` | `--write-only`; the CLI prints the path of `command.sh` and nothing else (for a run already finished it writes nothing, prints a note on standard error and no path, and exits 0) | `command.sh`, `launch/train-<k>.yaml`, `launch.json` |

Failures print `error: <message>` and exit with the code of the error class (as D4 to D7):

| Exit | Meaning |
|---|---|
| 2 `ConfigError` | an unusable or unvalidated config; no positive `train_steps`; the GPU rules; a malformed `--gpu-ranks` or `--cuda-visible-devices`; step directories without `launch.json`; a config that is not the run's; `train_steps` below the newest complete step; another launch holds the run |
| 3 `DataError` | `launch.json` is not a launch record (not JSON, wrong schema, over 64 KiB); the checkpoint directory cannot be listed |
| 4 `UpstreamError` | `eole` is not installed; an Eole command exits non-zero without an early stop, or runs past `--timeout`; `eole train` exits 0 without finishing the run (an interrupted run: Ctrl-C makes Eole exit 0 without a final checkpoint); a NaN or infinite figure in the log |
| 1 | an `OSError` writing the run directory, the lock or the manifest ("cannot write the launch outputs under DIR: ..."); any other `RinchenError` |
| 129, 130, 143 | SIGHUP, Ctrl-C, SIGTERM (below) |

### Identity

`launch.json` records the run: `schema_version` 1, `identity`, `world_size`, `gpu_ranks` and `created`
(ISO 8601 with a UTC offset). The identity is the SHA-256 of the canonical JSON (sorted keys, no
spaces, non-ASCII text as it is) of the EFFECTIVE config WITHOUT `training.train_steps` and
`training.train_from`. These two are the only keys a run may change and stay the same run: a longer
run extends `train_steps`, and the resume config names the checkpoint. Everything else is in the
identity, for a reason the spike measured: with Eole's default `reset_optim: none` the optimiser
options (learning rate, warmup, decay) come from the CHECKPOINT's config, so a resume with a new
learning rate would silently ignore it. A different GPU count is a different identity (the batch
changes). Paths are in the config and so in the identity: a run moved to another directory, or a
config prepared on one machine and used on another with other absolute paths, is another run. Not in
the identity: `--threads`, `--timeout`, `--cuda-visible-devices`, the `eole` path.

The identity is enforced when the checkpoint directory holds step directories: with none, the record
is written (or rewritten) and the launch is a fresh start; with step directories and no `launch.json`
(a run started by hand without `--write-only`, or a copied directory) the launch is refused. The
mismatch message names the recorded and the requested world size and ranks and the file, but cannot say
which key differs (an open item).

Measured (the separate run of "Extending a run", finished at step 200, then a launch of a copy of its
config with `learning_rate: 0.25`; in the test, the control with the same values written by another hand
was accepted):

```
error: the config is not the one this run was started with ($W/tiny/separate/run/launch.json): recorded: world_size 1, gpu_ranks []; this launch: world_size 1, gpu_ranks []. Only training.train_steps may change when a run is resumed: Eole keeps the optimiser options of the checkpoint and ignores new ones, and a different GPU count is a different batch. Restore the original config or use a new run directory
```

exit 2, and the tree of the run directory (names, sizes, modification times, SHA-256) unchanged.

### Attempts and the files of a run

Every launch that runs is an attempt; the attempt number is the highest existing `attempt-<k>.log`
plus one (`--write-only` leaves no log, so the next real launch reuses its number and overwrites
`train-<k>.yaml` and `command.sh`). Each attempt has its own effective config and its own log, so what
ran is on disk and an earlier attempt's log is never read as a later one's. `training.train_from`
appears ONLY in the `train-<k>.yaml` of an attempt that resumes (`training.train_from:
<model_path>/step_<N>`, the newest complete step, never `model_path` itself, because Eole's saver is
not atomic); the config file the operator gives is never changed, and `validate_train_config` accepts
`train_from` only under `resumes=True` and only for `<training.model_path>/step_<N>`.

The tiny run of "Commands" left (measured; sizes in bytes; `ckpt/` has the four step directories
`step_50`, `step_100`, `step_150`, `step_200` of the same shape):

```
run/
  command.sh                         2,037    the commands of the latest attempt
  launch.json                          188    the record of the run
  manifest.json                      2,956    written last
  vocab.src                          3,063    built by eole build_vocab
  launch/lock                            0    the run lock; empty, never removed
  launch/build_vocab.log             3,028
  launch/train-1.yaml                5,850    the effective config of attempt 1
  launch/attempt-1.log              46,450    standard output and error of eole train
  ckpt/step_200/model.00.safetensors 3,289,000
  ckpt/step_200/optimizer.pt         1,490,955
  ckpt/step_200/config.json              7,457
  ckpt/step_200/vocab.json               8,081
  ckpt/step_200/toy.model               12,620   the SentencePiece artifact
  ckpt/config.json, model.00.safetensors, optimizer.pt, toy.model, vocab.json
                                              relative symbolic links into step_200/
```

`du -sb` of the run directory: 19,296,131 bytes; of one step directory 4,808,113. The sizes of
`config.json` and the logs depend on the length of the paths in them.

### The NaN gate

Eole does not stop on a NaN loss: it prints `nan`, trains to the end, writes NaN checkpoints and exits
0 (spike, `lr` 1e10). `launch` reads the attempt's log when `eole train` has ended: any NaN or
infinite figure in the log (`acc`, `ppl`, `xent`, `lr`, `aux`, ..., or a validation figure) is an
`UpstreamError` that names the field, the step and the log, and the run is not accepted. A later
launch refuses the same way when the last attempt's log has a non-finite figure at or before the step
of the newest complete checkpoint ("delete the step directories from that step on, or use a new run
directory"). **The gate runs at the END of an attempt: there is no live abort.** A teacher that went
NaN at hour 2 of 60 runs on, burning GPU time, until the operator notices (watch the log or
TensorBoard) or the run ends. A live abort is an open improvement.

### Early stop

Early stopping runs on rank 0 only (spike, source). The launcher calls an attempt `early_stopped`
when the log has `Early Stop!` and the step saved right after it (`Saving optimizer and weights to
step_N`) is the newest complete step directory, whatever the exit code. With one process Eole exits 0.
With several processes the others wait for rank 0 at the next collective and the job dies after
`training.timeout` seconds with exit code 1, after the final checkpoint was written complete
(spike, CPU with gloo through a shim, 2 ranks: `Early Stop!`, `Saving ... step_120`, `Timed out waiting
20000ms for send operation to complete`, exit 1, 52 s; NCCL untested). So exit 1 with `Early Stop!` and
a complete final step is an early stop, not a failure; any other non-zero exit is a failure. A log with
`Early Stop!` whose final checkpoint is not complete is the interrupted-run error.

**Status quirk.** An early stop that lands exactly on `train_steps` is reported `early_stopped` by the
launch that ran it and `already_complete` by the next launch (which sees step `train_steps` complete
first).

### What a resume does not restore

A resumed run is not the run that would have been (spike, `train_from` of a `step_N`; the launcher
changes none of this):

- **The data stream restarts from line 1 of every corpus.** A resumed step 110 showed the same
  `sents: 1512; bsz: 1131/1231/151` as step 10 of the uninterrupted run. On a real corpus with few
  resumes the bias is unmeasured; the teacher sees the first rows of every corpus again.
- The RNG is not restored; scheduled values restart at their first entry (a scheduled `accum_count`
  or `dropout` list; that is why the teacher config uses constant one-entry lists and why
  `accum_steps` and `dropout_steps` are not allowed keys); the EMA (`average_decay`) is not restored
  and replaces the live weights in every saved checkpoint, so `average_decay` and `average_every` are
  not allowed keys either.
- `keep_checkpoint`'s queue is not restored: 100 steps, then a resume to 200 with `keep_checkpoint: 2`
  left `step_50, 100, 150, 200`.
- The weights, the Adam moments, the step counter and the Noam schedule ARE restored (optimizer.pt
  says 201 after step 200; the learning rate at step 110 was 5.96e-03 in both runs).
- **An interrupted run differs from an uninterrupted one.** Spike: three uninterrupted 200-step runs
  are byte-identical (weights SHA-256 `32a1c92e...`); a clean stop at 100 and a resume to 200 gave
  `3bdbb444...`, 58 of 62 tensors different, largest difference 0.538, validation perplexity at 200 of
  8.61 against 8.82; a SIGKILL near step 120 and a resume gave the same bytes as the clean stop (a
  resume itself is deterministic). Measured again for this runbook: the weights of the uninterrupted
  run of "Commands" have SHA-256 `32a1c92e42d8...` (the spike's value); after the weights of `step_200`
  were cut and the run resumed from step 150, the finished `step_200` has `7e62604fb210...`, with
  validation perplexity 8.80908 at 200 against 8.81733; the separate run resumed from step 100 had 8.60773
  at 200 (spike: 8.61).
- Eole's saver is not atomic. It writes `optimizer.pt`, the weights, the transform artifacts,
  `vocab.json` and `config.json` (last) into `step_N/`, linking each into `model_path` right after, and
  then deletes the oldest step directory (`keep_checkpoint`). A kill in the middle leaves a partly
  written `step_N` and root links that mix two steps: the spike's simulated kill mid-save, resumed
  through `model_path`, exited 0 and loaded weights from step 50 with the optimizer of step 100. So a
  resume names a complete `step_N` directory. A fresh run into a `model_path` that holds step
  directories is not refused by Eole either, hence `launch.json`.

## Commands

The runs below were made on 2026-11-08 on the machine of "Measured numbers", through the real binary
(`.venv/bin/rinchen`, what `uv run rinchen` starts) from the repository root with the GPU hidden
(`CUDA_VISIBLE_DEVICES=""`), on the toy data, once each. `$W` is a scratch directory and
`$RUN=$W/tiny/out/variants/t0.5_bt0.5_src_s1234/run`. The inputs are written by the test helper (a
corpus Parquet file of 126,698 bytes and a SentencePiece model of 12,620 bytes):

```bash
uv run python -I -c "import sys; sys.path[:0] = ['src', '.']; from pathlib import Path; from tests.support.tiny_variant import build_toy_inputs; build_toy_inputs(Path('$W/in'))"
```

The matrix is one variant (temperature 0.5, bt ratio 0.5, tag on the source, seed 1234, 100 dev lines
per direction) over `configs/eole/teacher_tiny.yaml`: `name: teacher_tiny_gate`, `base:
configs/eole/teacher_tiny.yaml` (absolute), `axes: {tag_placement: [src], temperature: [0.5],
bt_ratio: [0.5]}`, `seeds: [1234]`, `dev_max_lines: 100`.

### Prepare

```bash
uv run rinchen train ablate --matrix $W/matrix.yaml --corpus $W/in/toy.parquet \
    --spm-model $W/in/spm/toy.model --out-dir $W/out --prepare-only --threads 2
```
```
prepared t0.5_bt0.5_src_s1234: $W/out/variants/t0.5_bt0.5_src_s1234/config.yaml
manifest: $W/out/manifest.json
```

0.78 s. It writes what `ablate` would write before it trains (the corpora, the weights, the
validation corpus, the rendered `config.yaml`) and the manifest (stage `train_prepare`), and trains and
scores nothing. It prints one `prepared <id>: <config.yaml>` line per variant. `--force` and
`--timeout` are refused with `--prepare-only` (they have nothing to do with it), `--threads` is only
recorded, `--dry-run` wins over it.

### Launch (A: from nothing)

```bash
uv run rinchen train launch --config $W/out/variants/t0.5_bt0.5_src_s1234/config.yaml \
    --run-dir $RUN --threads 2 --timeout 600
```
```
status: completed
steps: 200
resumed from: none
attempt: 1
checkpoint: $RUN/ckpt/step_200
log: $RUN/launch/attempt-1.log
command.sh: $RUN/command.sh
manifest: $RUN/manifest.json
```

25.61 s wall, including `eole build_vocab`, 200 steps, four validations and four saves. The curve
(`parse_train_log` of `attempt-1.log`, 20 reports, 4 validations, saved steps 50, 100, 150, 200):

| step | 10 | 20 | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| xent | 5.44 | 5.40 | 4.64 | 2.81 | 2.28 | 2.14 |

The fall from the first report to the last is 60.7 % (the gate asks for 30 %; the spike's three seeds
fell 60.7, 60.7 and 62.4 %). Validation perplexity at steps 50, 100, 150, 200: 83.6869, 16.6115,
10.1611, 8.81733 (accuracy 12.6863, 49.3498, 56.2956, 58.5791). No non-finite figure; `loss_gate`
returns None. The weights of `step_200` have SHA-256 `32a1c92e42d8219fbd9e0a09a0a08a2069590ca8c1f140aaf2adc271c936f548`,
the spike's value for an uninterrupted run. The manifest of the run records `xent_first` 5.44,
`xent_last` 2.14, `valid_ppl_first` 83.6869, `valid_ppl_last` 8.81733. The Task 6 gate saw 27 to 35 s
for the same run.

### Launch again (B: already complete)

The same command: `status: already_complete`, `steps: 200`, `resumed from: none`, `attempt: 0`,
`checkpoint: $RUN/ckpt/step_200`, `log: none`, `command.sh: none`, the manifest line. 0.73 s. The
tree of the run directory (names, sizes, modification times, SHA-256) is unchanged: nothing ran,
nothing was written, the manifest was kept because it still vouches for the same files.

### A damaged final checkpoint (C: truncate and resume)

The weights of `step_200` were cut to half their size (1,644,500 of 3,289,000 bytes). `is_complete`
says `model.00.safetensors: is not a complete safetensors file (Error while deserializing header:
incomplete metadata, file not fully covered)`. The same command:

```
status: completed
steps: 200
resumed from: 150
attempt: 2
checkpoint: $RUN/ckpt/step_200
log: $RUN/launch/attempt-2.log
command.sh: $RUN/command.sh
incomplete step directories skipped: [200]
manifest: $RUN/manifest.json
```

9.60 s. `launch/train-2.yaml` holds `training.train_from: $RUN/ckpt/step_150`; the log starts at
`Step 160/  200;` and has one validation (at 200: 8.80908); `launch.json` is byte-identical to the
first launch's; `step_200` is rewritten and complete again (its weights differ from the first run's, see
"What a resume does not restore"). The manifest was refreshed.

### Extending a run (D)

A separate run with `train_steps: 100` (18.07 s, `completed`, step 100), then the same config with
`train_steps: 200`: `--write-only` first (0.69 s, prints `$W/tiny/separate/run/command.sh`), then the
real launch (13.50 s): `resumed from: 100`, `attempt: 2`, the first report of the log
`Step 110/  200; acc: 57.1; ppl: 12.14; xent: 2.50; ... lr: 5.96e-03; sents:    1512; bsz:
1131/1231/151; ...`, validation perplexity 10.0769 at 150 and 8.60773 at 200. `launch.json` was not
changed. Lowering `train_steps` to 50 afterwards:

```
error: training.train_steps is 50, below step 200, the newest complete checkpoint of this run: a run is not shortened; set train_steps to at least 200 or use a new run directory
```

exit 2, tree unchanged.

**The loss gate is for a first attempt.** `parse_train_log` reads the LAST run of a log, and each
attempt has its own log, so `loss_gate` on the resumed attempt's log compares step 110 (xent 2.50) with
step 200 (2.07), a fall of 17.2 %, and answers `xent went from 2.5 at step 110 to 2.07 at step 200, a
fall of less than 30%`. Apply it to the log of the attempt that started the run.

### Averaging (F)

```bash
uv run rinchen train average --checkpoint-dir $RUN/ckpt --last 3
```
```
averaged: 3 checkpoints (steps 100, 150, 200)
output: $RUN/averaged
weights sha256: 4a0c8535f585e3a33186dc81987208d9e33b9f3182f283114ea6454450217ff0
manifest: $RUN/averaged/manifest.json
```

0.74 s. See "Averaging".

### Options

`rinchen train launch`:

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | required | The Eole training config (YAML) to launch |
| `--run-dir PATH` | required | The run directory: `command.sh`, `launch.json`, `launch/` and the manifest go here, and the config's `training.model_path` must lie in it. Created when the launch takes the lock |
| `--world-size N` | the config's | Replaces `training.world_size` (at least 1) |
| `--gpu-ranks 0,1,2,3` | the config's | Replaces `training.gpu_ranks`: distinct non-negative decimal numbers, spaces allowed. Independent of `--world-size`. There is no way to pass an empty list |
| `--threads N` | 2 | OpenMP and MKL threads of each Eole run |
| `--timeout S` | none | Seconds after which each Eole command is killed (positive, finite). For a real teacher run leave it off |
| `--cuda-visible-devices 0,1,...` | none | Exported to a GPU run as `CUDA_VISIBLE_DEVICES`, at least one device per rank; refused for a run without a GPU, which always hides them all |
| `--write-only` | off | Write `command.sh`, the attempt's config and `launch.json`, print the path of `command.sh`, run nothing |

`rinchen train average`: `--checkpoint-dir PATH` (required; the run's `training.model_path`, not the run
directory), `--last N` (required, at least 2: the newest N complete step directories), `--out PATH`
(default: `averaged` beside the checkpoint directory as it is named; an existing one is refused).

`rinchen train ablate ... --prepare-only`: see "Prepare" above and "Prepare and ablate refuse each
other".

### Refusals (real binary, 2026-11-08, tiny config)

| Invocation | Exit | Message |
|---|---|---|
| `launch --world-size 2` | 2 | `error: invalid Eole config: training.gpu_ranks: is empty although training.world_size is 2: Eole would start no process and exit 0 without training or a checkpoint; list one rank per GPU or set world_size 1` |
| `launch --world-size 2 --gpu-ranks 0,1` (no `timeout`) | 2 | `error: invalid Eole config: training.timeout: must be at least 3600 seconds when world_size is above 1: rank 0 alone validates and saves while the other ranks wait, and a wait beyond the timeout kills the job; got unset (Eole's default is 60)` |
| `launch --gpu-ranks 0,0` | 2 | `error: --gpu-ranks must be distinct GPU numbers separated by commas, such as 0,1,2,3, got '0,0': [0] is repeated` |
| `launch --cuda-visible-devices 0` (CPU run) | 2 | `error: cuda_visible_devices '0' is given for a run that uses no GPU (training.gpu_ranks is empty)` |
| a launch or a `--write-only` while the lock is held | 2 | ``error: another launch holds this run: $W/tiny/separate/run/launch/lock is locked by a running `rinchen train launch` of the same run directory or by an Eole process it started (a launcher that was killed leaves its Eole training: wait for it to end or stop it; the lock goes away with the process)`` |
| `train_steps` lowered, or a changed learning rate | 2 | the two messages above |

None of the first four wrote anything (their run directory was not created); the two on the finished
run left its tree unchanged (names, sizes, modification times and SHA-256 compared); a launch refused
for the lock writes nothing by construction.

### The GPU rules

`check_gpu_rules` runs before anything starts. Eole itself checks only two of these and accepts the
rest silently:

1. `world_size` is a positive integer and `gpu_ranks` a list of distinct non-negative integers.
2. `world_size` above 1 with no `gpu_ranks` starts no process (exit 0, no checkpoint): refused.
3. `world_size` below the number of ranks, or equal to it with no rank 0: refused with Eole's words.
4. With `world_size` above 1, `training.timeout` is at least 3600 seconds (below).
5. With `world_size` above 1, at most 80 training corpora (below).
6. With `early_stopping` on, `early_stopping_criteria` is ONE non-empty string. Several criteria
   make Eole call a validation "improving" only when all improve and "decreasing" only when all
   worsen, else "stalled"; unset, it uses perplexity and accuracy together.
7. With `early_stopping` on and `valid_steps` equal to `save_checkpoint_steps`, `keep_checkpoint` is -1
   or at least `early_stopping + 1` (below).

`--cuda-visible-devices` must name at least as many devices as ranks.

## The run lock

A real `launch` takes an exclusive, non-blocking `flock` on `run_dir/launch/lock` before it reads the
state of the run and holds it until it returns, however it ends (success, error, Ctrl-C, SIGTERM). A
second launch into the same run directory exits 2 naming the lock file (the message above). The lock
belongs to the open file, not to the path: a crash releases it and nothing stale is left. The file is
empty and is never removed.

- **The Eole processes inherit the lock.** The descriptor is passed to `eole build_vocab` and `eole
  train` (`run_logged(inherit_fds=)`, `pass_fds`; no other descriptor is passed). Eole leads a session
  of its own and outlives a SIGKILL of the launcher; the lock then stays held by that Eole until it
  exits, so a second launch cannot start another training into the same checkpoint directory. The
  review reproduced the hole before this: with the lock held only by the launcher, a SIGKILL of the
  launcher released it, the next `launch` started a second Eole into the same `model_path`, and the
  two logs both saved `step_60`, `step_90` and `step_120` at the same time (validation perplexity 11.46
  against 10.82). After the fix, with the real Eole: SIGKILL of the launcher at step 30, the orphan
  trained alone to `step_120`, a second launch exited 2, and a third, after the orphan ended, said
  `already_complete`.
- **Which processes hold it.** The holder is Eole's LEADER, `eole train` itself, which inherited the
  descriptor. The per-rank processes of a multi-GPU run are `torch.multiprocessing` "spawn" children
  started with `close_fds=True` and do NOT hold it (the final review's real run: only the launcher and
  `eole train` held `launch/lock`; the data-loader worker and the resource tracker did not). For a
  multi-GPU run that means the lock lives as long as the leader, not as long as every rank: if the
  leader dies while the launcher is gone too, ranks that outlive it train on without the lock (with the
  launcher alive, `run_logged` kills the whole group when the leader exits non-zero: "Signals").
- **`--write-only` takes no lock.** It only looks, without creating the file, and is refused while
  another launch (or a lingering Eole) holds it, because it would replace the config that launch runs.
  The look is not a hold: a launch that starts after it is not stopped.
- A launch refused for what it finds in the run directory (another config, a NaN) has taken the lock
  by then and leaves the empty lock file and nothing else; one refused for its config or options
  writes nothing.
- The lock is advisory and local. It stops other callers of `launch` on a file system that supports
  `flock`. A hand-run `command.sh` neither takes it nor sees it (nothing guards two hand runs, or a hand
  run next to a launch). `flock` on NFS or Lustre was not checked: an `OSError` there ends the command
  with exit 1.
- The manifest is written after the lock is released (a tiny window, an open minor).

## Signals

`launch` waits for Eole through `run_logged`: it wakes every 0.2 s, kills Eole's whole process group
(Eole starts data-loader workers; the child runs in a session of its own) and exits. The group is killed
in four cases: Ctrl-C, SIGTERM and SIGHUP, a timeout, and **Eole's leader exiting with a non-zero code or
by a signal on its own** (a Python error, the OOM killer, `kill -9` of `eole train` alone). The last case
is the final review's finding I-1: the per-rank processes of a multi-GPU run do not hold the lock (above)
and would otherwise train on, writing `step_N`, while the failed launch reported exit 4 and the next
`launch` started a second training into the same `model_path`. A leader that exits 0 is not followed by
a kill: it has no such survivors, and one that left a process running on purpose is not the launcher's
to stop. Evidence is the Task 6 gate (real Eole, tiny run) and its probes:

| Signal | Exit | What happens |
|---|---|---|
| Ctrl-C (SIGINT) | 130 | the group is killed; the run is resumable |
| SIGTERM | 143 | measured 0.21 to 0.26 s to exit; none of the 2 to 4 descendants survives; the lock is free; no manifest; the next launch finished the run (8 s into a run: `resumed from: 100`; 0.5 s in, before any checkpoint: a fresh attempt 2) |
| SIGHUP | 129 | as SIGTERM |
| Eole's leader exits non-zero or is killed (`kill -9` of `eole train` alone) | 4 | the rest of the group is killed before the launcher returns; the lock is free; the next launch resumes. Real Eole, 120-step tiny run, SIGKILL of the `eole train` leader at step 60 (group: leader, resource tracker, data-loader worker): launcher exit 4 in 0.26 s, "eole train failed: killed by signal 9", no member of the group left (none at once, none 3 s later), no manifest, `step_50` complete, and the next launch `resumed from: 50`, attempt 2, completed at step 120. Before the fix the resource tracker and the worker outlived the launcher (the review) |
| SIGKILL of the launcher | none | cannot be caught: Eole's group keeps running and the lock stays held by Eole's leader; the next launch is refused until it ends (above); no manifest is left, the next launch writes it. Nothing is left to sweep the group if the leader dies afterwards: on a GPU node look for stray `eole` processes before relaunching |
| `nohup` (SIGHUP inherited as ignored) | none | the ignore stays: `run_logged` leaves a signal that was ignored at start alone (D7's fix, measured there for `ablate`: the run finished with exit 0; not re-measured for `launch`) |
| a shell background job (`&`) | none | starts with SIGINT ignored (POSIX) and stays so: stop such a run with SIGTERM or SIGHUP |

SIGINT and SIGHUP were probed once each for their exit code; their timings were not recorded. Nothing of
this was tried under NCCL: the process group of a multi-GPU job (the spawned workers) is expected to be
killed the same way, unmeasured.

## The 8-GPU teacher: operator workflow

All of this is operator guidance, **unrun on a GPU**. What was run is the same chain on the CPU with the
tiny config.

1. **Prepare on the node that will train.** One-variant matrix with `base:
   configs/eole/teacher.yaml` and the temperature, bt ratio and tag placement the ablations chose, then
   `rinchen train ablate --matrix M --corpus C.parquet --spm-model teacher_128k.model --out-dir OUT
   --prepare-only`. It writes `OUT/variants/<id>/config.yaml` with absolute paths (the paths are part
   of the run identity: prepare where the run will be resumed). Use an `--out-dir` that no `ablate`
   run used (below). The SentencePiece model must exist: D3 trained only `tiny_1k`; `teacher_128k` is
   schema-validated and was never trained.
2. **Write the script.** `rinchen train launch --config OUT/variants/<id>/config.yaml --run-dir
   OUT/variants/<id>/run --cuda-visible-devices 0,1,2,3,4,5,6,7 --write-only` prints the path of
   `command.sh` and writes `launch.json` and `launch/train-1.yaml`. Read both.
3. **Run it by hand** (a terminal multiplexer, `setsid nohup bash command.sh > command.out 2>&1 &`, or
   inside a scheduler job script). Because the script appends Eole's output to `launch/attempt-<k>.log`
   (`tee -a`, with `pipefail` so the exit status stays Eole's), a later `launch` reads a NaN or an early
   stop of the hand run exactly as it reads its own. `setsid`, `nohup`, the cluster's NCCL variables
   (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, ...) and the scheduler directives are the operator's: the script
   has none, and none was tried.
4. **Resume.** After an interruption either run `rinchen train launch` with the same config (it
   resumes under the lock and tells how it ended), or `--write-only` again and run the new
   `command.sh` (the resume point is fixed when the script is written). Do not run a hand script
   next to a `launch`: nothing guards it.
5. **Finish.** `rinchen train launch` again says `already_complete` and writes the manifest of the
   final step (it hashes the whole final step directory: several GB for the teacher, a wait of its
   own). Then `rinchen train average --checkpoint-dir OUT/variants/<id>/run/ckpt --last 4`.

### `command.sh`, verbatim

`rinchen train launch --write-only` for `configs/eole/teacher.yaml` merged with run paths and a minimal
data section (two corpora and `valid`; the SentencePiece model `/data/spm/teacher_128k.model`), through
`launch(..., write_only=True)` with a placeholder `eole` path, `/opt/rinchen/.venv/bin/eole`, used only to
render. The GPU rules passed; `bash -n` accepted the script; nothing was run, no GPU was touched. `$W` is
the scratch directory. The `cd` line is the working directory of the command that wrote the script:

```bash
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "$W/gpus8/run".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# launch.json beside it records the run, so `rinchen train launch` can resume a run started
# with this script. The output of `eole train` is appended to "launch/attempt-1.log", which
# `rinchen train launch` reads for a NaN or an early stop when it takes the run over.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /home/tsee/Personal/rinchen
mkdir -p $W/gpus8/run/launch
[ -f $W/gpus8/run/vocab.src ] || /opt/rinchen/.venv/bin/eole build_vocab -config $W/gpus8/run/launch/train-1.yaml -n_sample -1 2>&1 | tee -a $W/gpus8/run/launch/build_vocab.log
/opt/rinchen/.venv/bin/eole train -config $W/gpus8/run/launch/train-1.yaml 2>&1 | tee -a $W/gpus8/run/launch/attempt-1.log
```

With `--world-size 4 --gpu-ranks 0,1,2,3 --cuda-visible-devices 0,1,2,3` the only difference is the
export (`CUDA_VISIBLE_DEVICES=0,1,2,3`) and the paths; with `--world-size 1 --gpu-ranks 0
--cuda-visible-devices 0` it is `CUDA_VISIBLE_DEVICES=0`. Both rendered and passed `bash -n`. Without
`--cuda-visible-devices` a GPU run has no `CUDA_VISIBLE_DEVICES` line (the environment of the script
decides); a CPU run exports `CUDA_VISIBLE_DEVICES=''`. A run written for a resume differs in two
lines, measured for the tiny config (attempt 2 of the separate run of "Extending a run"):

```bash
# Written by `rinchen train launch` for attempt 2 of the run in "$W/tiny/separate/run".
# It resumes from step 100 (training.train_from of the config it names).
...
/home/tsee/Personal/rinchen/.venv/bin/eole train -config $W/tiny/separate/run/launch/train-2.yaml 2>&1 | tee -a $W/tiny/separate/run/launch/attempt-2.log
```

What the script contains and does not:

- **Absolute paths and a `cd`.** Every path is absolute, so it runs from anywhere; the `cd` is for the
  relative paths a config may hold. The `eole` path is the one next to the interpreter that wrote the
  script, and the config paths are those of the machine that wrote it: **regenerate it on the target
  node**, do not copy one.
- `set -euo pipefail`, so a failing `eole build_vocab` stops the script and the exit status of `eole
  train` survives the `tee`. `PYTHONUNBUFFERED` is not needed (the log lines are written as they are
  logged).
- `build_vocab` is guarded by `[ -f vocab.src ]`: a rerun does not count the corpora again.
- No lock, no `trap`, no `setsid`, no `nohup`, no NCCL variable, no TensorBoard setting.
- The script fixes the resume point at the moment it is written.

`--write-only` rewrites `command.sh` and leaves an existing manifest in place, so the manifest's
`command.sh` digest is stale afterwards (measured on the separate run: the digest of `command.sh` in
`manifest.json` no longer matched after `--write-only`); the next launch that runs rewrites the manifest.

### First contact with a GPU

Not done; needs the owner's authorisation. The cheapest first step is one GPU and a few hundred steps of
the teacher shape (`--world-size 1 --gpu-ranks 0 --cuda-visible-devices 0`, a copy of the config with
`train_steps` of a few hundred), which would measure for the first time: that bf16 training runs and
does not go NaN, the tokens per second, the memory, the time and size of a save (13.3 GB for the 128k
vocabulary, below), and whether `GradScaler` with bf16 behaves. Then two GPUs, to see the `timeout`
rule and the rank-0-only validation work, before eight.

## Eole facts for a GPU run

From the spike. **Every row is unmeasured on a GPU or under NCCL.** The tag says what stands behind
it: *source-read* = read from Eole's source, never run; *CPU run* = run on the CPU (with a gloo shim
where several ranks are needed).

| Fact | Tag |
|---|---|
| **Spawn.** `eole train` starts the per-GPU processes itself: `world_size` above 1 gives one non-daemon `torch.multiprocessing` "spawn" process per entry of `gpu_ranks`; world size 1 with one rank is single-GPU; else CPU. There is no torchrun and no `RANK`, `WORLD_SIZE` or `MASTER_*` environment variable read anywhere in the package (`bin/run/train.py:27-103`) | source-read, unmeasured (the spawn and the rendezvous ran on the CPU through a shim) |
| **Ranks.** `world_size` below `len(gpu_ranks)` is an error; equal, positive and no rank 0 is an error (`config/training.py:330-335`). 8 GPUs on one node: `world_size: 8`, `gpu_ranks: [0..7]`. Process `i` uses the `i`-th visible device and has global rank `gpu_ranks[i]` | CPU run (the error messages); source-read (the device mapping) |
| **Silent no-op.** `world_size: 2` with `gpu_ranks: []` spawns 0 processes ("Waiting for 0 training processes to complete..."), exit 0, no checkpoint | CPU run |
| **`timeout` is the collective timeout** (`utils/distributed_workers.py:21-52`, default 60 s). Rank 0 alone validates, reports, early-stops and saves while the others wait at the next collective, so validation, an in-training BLEU and the write of a step directory of 13 GB must fit in it: hours, not a minute. `teacher.yaml` sets 7200; the launcher refuses less than 3600 for `world_size` above 1 | source-read, unmeasured |
| **Multi-GPU early stop.** Rank 0 breaks out of the loop and writes the final checkpoint complete, the others hang, the job dies after `timeout` with exit 1. On a 2-rank CPU gloo shim: `Early Stop!`, `Saving ... step_120`, `Timed out waiting 20000ms for send operation to complete`, exit 1, 52 s | CPU run; NCCL untested |
| **Per-corpus statistics buffer.** With several processes `Statistics.all_gather_stats` pickles the per-corpus statistics into a 4096-byte CUDA buffer. Pickle sizes: 650 B (9 corpora), 1.4 KB (30), 2.5 KB (60), 4.0 KB (100), 4.46 KB (112 corpora with 22-character names), so a `ValueError: encoded data exceeds max_size` is expected around 100 corpora or more; the launcher refuses more than 80 (a margin, not a measured limit). The crash itself was never reproduced | pickle sizes: CPU run; crash: source-read, unmeasured |
| **Batch.** `batch_size` is per GPU per micro-batch; with `batch_type: tokens` a batch holds at most `batch_size` PADDED tokens (`num_sents x max(len src, len tgt)`; the toy log shows about 1,180 real source tokens in a "2048" batch). Each rank reads every `world_size`-th example of every corpus (on the shim, 2 ranks: 3,009 sentences per report against 1,512). `padded_tokens_per_update = world_size x accum_count x batch_size` (`inputters/dynamic_iterator.py:400-415`, `trainer.py:295-306, 444-467`, `train_single.py:258-269`) | source-read; the stride on the shim |
| **Double division by `world_size`.** The loss is the token-loss sum divided by `normalization` (default `sents`; `tokens` is allowed) summed over the ranks with `all_gather_list`; then the gradients are all-reduced and rescaled by `n_gpu` AGAIN (`GradientSynchronizer.synchronize`, `trainer.py` near line 187). The update gradient is `1/world_size` of the global mean. Adam does not mind (its step is invariant to the gradient scale) but `max_grad_norm` (Eole's default 5, not set in `teacher.yaml`) becomes `world_size` times looser: a clip at 5 on the scaled gradient is a clip at 40 on the global mean for 8 GPUs (ARITHMETIC on a source reading) | source-read, unmeasured |
| **`self_attn_backend`.** Eole's default is `flash`, and it falls back to `pytorch` silently when `flash_attn` is missing, the capability is below 8 or the dtype is not fp16/bf16 (`config/common.py:242-256`). `flash_attn` is not installed here (the log says so), so the teacher run would use the pytorch backend unless the operator installs it | source-read; the missing module is seen in every log |
| **`batch_size_multiple`.** Default 1; Eole's "fp16 gives 8" rule is dead code in 0.6.4, so `teacher.yaml` sets 8 | source-read |
| **Precision.** `use_amp` defaults to true: with `compute_dtype` of 16 bits Eole keeps an fp32 copy of the parameters and autocasts (`config/training.py:69, 292-305`). The weights of a step directory are therefore expected to be fp32 on disk (the CPU fp32 run saves fp32); a bf16 file would be half the size. Never run | source-read, unmeasured |
| **Multi-node.** No `n_nodes` or `node_rank`: each node runs its own `eole train` with the same `world_size` (the total), `master_ip` of node 0 and its own global `gpu_ranks` (node 1 of 2x4: `[4,5,6,7]`). The launcher's rules allow that shape; the lock and the checkpoint directory are node-local unless the storage is shared | source-read; untested |
| **Orphans.** The workers are non-daemon torch "spawn" processes started with `close_fds=True`, so they inherit neither the lock descriptor nor anything else: a SIGKILL or a crash of only the parent leaves them training and holding the GPUs. The launcher starts Eole in a session of its own and kills the group on Ctrl-C, SIGTERM, SIGHUP, a timeout and when the leader exits non-zero on its own (a clean exit 0 is not followed by a kill). A SIGKILL of the LAUNCHER still leaves Eole running; the lock is then held by Eole's leader only, so on a multi-GPU run it lasts as long as the leader, not as long as the ranks | CPU run (the group kill, including the leader's own crash); the rank processes: source-read and a fake that starts a `close_fds` child; GPU untested |
| **Ctrl-C on 2 ranks.** All exit with rc 0, "Training interrupted by user" and "Connection reset by peer" noise, about 4.5 s, no final save | CPU run on the shim |
| **TensorBoard.** `tensorboard_log_dir` is not set in `teacher.yaml`; Eole's default is the relative path `runs/eole` (`config/common.py:166`) plus a start-time subdirectory (`utils/report_manager.py:117-119`), so left alone the events would go to `<working directory of the process>/runs/eole/<Mon-DD_HH-MM-SS>`, outside the run directory (the final review reproduced that with the real binary). `launch` therefore adds `tensorboard_log_dir: <run directory>/tensorboard` (an absolute path) to the effective config when `tensorboard` is true and the config names no directory; a directory the config names is kept AS WRITTEN: a RELATIVE one is therefore relative to the working directory of the launch (the original hazard, by explicit choice) and its text enters the identity, so give an absolute path or none. The default is part of every rendered `launch/train-<k>.yaml` and of the run's identity (below). Real run from a scratch working directory: the event files of the first attempt and of the resume were under `<run>/tensorboard/<Mon-DD_HH-MM-SS>/` and nothing was written under the working directory | source-read; CPU run for the effect of the default |

Eole writes nothing to `model_path` besides the `step_N` directories and the links in it. The wheel ships
no training recipes (no yaml under `site-packages/eole`), so the shape of the teacher is the project's
own prior, not a copy of an Eole recipe.

## The effective batch and the size of the teacher (ARITHMETIC)

Not measurements: formulas applied to the values of `teacher.yaml`, with the formula checked against the
real tiny checkpoint.

**Effective batch** = `world_size x accum_count x batch_size` padded tokens per update (batch size 16384):

| Variant | world_size | accum_count | padded tokens per update | `max_grad_norm` 5 acts like |
|---|---:|---:|---:|---:|
| the config (8 GPUs) | 8 | 8 | 1,048,576 | 40 |
| 4 GPUs, same `accum_count` | 4 | 8 | 524,288 | 20 |
| 4 GPUs, `accum_count` 16 | 4 | 16 | 1,048,576 | 20 |
| 1 GPU, same `accum_count` | 1 | 8 | 131,072 | 5 |
| 1 GPU, `accum_count` 64 | 1 | 64 | 1,048,576 | 5 |

A change of the GPU count changes the batch and is a different run identity. At the config's batch,
`train_steps: 100000` is 104,857,600,000 padded tokens (a step is one update, whatever `accum_count`
is).

**Parameters** of the shape (encoder 24 and decoder 12 layers, hidden `H` 1536, feed-forward `F` 4096,
shared vocabulary `V`, shared embeddings, no biases on the linear layers). Per encoder layer: `4H^2 +
2HF + 4H` (attention, feed-forward, two layer norms). Per decoder layer: the same plus `4H^2 + 2H + 1`
(context attention, its layer norm, the `layer_scalar`). Plus `2 x 2H` for the two final layer norms,
`V x H` for the shared embedding and `V` for `generator.bias`; `generator.weight` and the target
embedding are not saved when shared. The weights file also holds two position buffers of `5000 x H`
(`src_emb.pe.pe`, `tgt_emb.pe.pe`). **Check** against the real tiny checkpoint (`H` 64, `F` 128, 2+2
layers, `V` 232): the formula gives 180,458 parameters and 640,000 buffer elements, 820,458 in all, which is
exactly the element count of the 62 tensors of the real `model.00.safetensors`, and `4 x 820,458 + 7,168`
(the header) = 3,289,000 bytes, exactly the real file. `optimizer.pt` (Adam: two moments) is
1,490,955 bytes against `8 x 180,458` = 1,443,664: 3.3 % more, from the pickle and the archive (it does
not hold the buffers).

| | `V` = 64,000 | `V` = 128,000 (`teacher_128k`) |
|---|---:|---:|
| encoder layers (24) | 528,629,760 | 528,629,760 |
| decoder layers (12) | 377,597,964 | 377,597,964 |
| embedding and `generator.bias` | 98,368,000 | 196,736,000 |
| parameters | 1,004,601,868 | 1,102,969,868 |
| weights file, fp32 (`4 x (parameters + 15,360,000)`) | 4,079,847,472 B | 4,473,319,472 B |
| `optimizer.pt` (`8 x parameters`, plus a little) | 8,036,814,944 B | 8,823,758,944 B |
| one step directory (weights + optimizer) | 12.12 GB | 13.30 GB |
| `keep_checkpoint: 8` | 96.9 GB | 106.4 GB |
| peak on disk, 8 kept plus the one being written | 109.0 GB | 119.7 GB |

The master plan's "~1.1 B parameters" is the 128,000 piece column. Eole deletes the oldest step
directory only after the new one is written (`model_saver.py`), hence the peak. A bf16 weights file would
be half the weights row; see "Precision" above. The config, vocabulary and SentencePiece artifact of a
step add a few MB, not computed.

**Memory of the averager** is about 9 times the largest float32 tensor (measured 2.6 GB for three float32
checkpoints with one 300 MB tensor); the largest tensor of the teacher is the embedding, 786,432,000 B for
`V` = 128,000, so about 7 GB of host RAM (ARITHMETIC).

**Throughput.** The master plan asks this runbook for the expected tokens per second. None can be given:
no GPU run exists. The only throughput numbers are CPU, 2 threads, for the 64-wide 2+2 model: 56,900
to 69,114 source/target tokens per second in 15 of the 20 windows, 3,603 to 4,332 in the three windows that
follow a validation and a save (steps 60, 110, 160), 39,348/40,209 in the window of step 100 (not
explained) and 5,053/5,498 in the first window (start-up). They say nothing about the teacher.

## Checkpoints: cadence, `keep_checkpoint`, the best checkpoint

- **Cadence.** `valid_steps` and `save_checkpoint_steps` are equal (5000 in the teacher prior, 50 in the
  tiny one), so each validation is followed by `Saving optimizer and weights to step_N` and the n-th
  validation belongs to the n-th save. `launch` rule 7 holds this together with `keep_checkpoint`.
- **Eole keeps the last `keep_checkpoint` checkpoints, never the best one** (`-1` all, `0` none; a final
  checkpoint is written after the loop, on `train_steps` or an early stop, unless that step was just
  saved; none is written on SIGINT, SIGTERM or a crash). With early stopping after 6 validations without
  improvement the best checkpoint is 6 saves older than the last; `keep_checkpoint: 8` keeps it. The
  launcher does not name the best step. Read it from TensorBoard (`valid/ppl`; the events are under `<run directory>/tensorboard/`) or from the log:

  ```bash
  uv run python -I -c "import sys; from pathlib import Path; from rinchen.train.curves import parse_train_log; c = parse_train_log(Path(sys.argv[1]).read_text(errors='replace')); assert len(c.validations) == len(c.saved_steps); p, s = min(zip(c.validations, c.saved_steps)); print(f'best validation perplexity {p} at step {s}')" $RUN/launch/attempt-1.log
  ```

  which printed `best validation perplexity 8.81733 at step 200` for the tiny run. It pairs the n-th
  validation with the n-th save; a log whose counts differ (a final save at a step that is not a
  validation) needs the pairing by hand, and a resumed run has one log per attempt.
- **Early stopping** is on `ppl` alone (one criterion), strictly better (`<`), checked at validation steps
  on rank 0 only. In the spike (`valid_steps` 20, `early_stopping` 2): `Model is improving ppl: 67.3587
  --> 17.3205.`, `Decreasing patience: 1/2`, ..., `Training finished after not improving. Early Stop!`,
  `Best model found at step 80`, `earlystopper has_stopped!`, then a final checkpoint `step_120`, exit 0.
  A NaN counts as "stalled", so early stopping does not catch a NaN run.
- **`valid_metrics` is not used** (spec S10). The validation corpus is D7's pre-tokenised one
  (`transforms: []`), so Eole would score piece strings, not text; on the CPU, offline, `["BLEU"]` and
  `["BLEU","CHRF"]` do run (spike); `COMET` fails with `ImportError: unbabel-comet is required`; the
  EOLE scorers need converted models (untested). Validation perplexity and accuracy are logged. BLEU and
  chrF of a teacher come from `evaluate_variant` after the training. The master plan's
  `valid_metrics: [BLEU, CHRF]` is therefore left out of `teacher.yaml`, with a comment saying so.
- Validation in a multi-GPU job runs on rank 0 while the others wait inside `timeout`.

## Checkpoint completeness (`is_complete`)

A step directory is a real directory named `step_<N>` (N in ASCII digits, positive, no leading zero; a
link, `step_007`, `step_x`, a file are not listed). `is_complete(step_dir)` returns None or a phrase for
the first problem, checking in this order (Eole writes `config.json` last, so it is the first sign of a
save that did not finish): `config.json`, `vocab.json`, `optimizer.pt`, `model.00.safetensors`. Each must
be a regular file (not a link), non-empty; the two JSON files must parse as JSON objects;
`optimizer.pt` must be a zip archive whose central directory reads and lists an entry; the weights must
open with `safetensors` (it checks the header against the file size) and hold a tensor. The
SentencePiece artifact is not needed. No tensor is loaded, no archive member read, and the function never
raises for a damaged directory (a file that cannot be read is a problem too, worded `<file>: cannot be
read (<exception class>: <reason>)`, for example `model.00.safetensors: cannot be read (PermissionError:
Permission denied)`; the first version said `FileNotFoundError('No such file or directory')` for a weights file
without read permission, because `safetensors` words every failed open that way: the file is now opened
first, so the error is the real one).

What it catches and what it cannot (the review's measurements, on a real tiny checkpoint):

- **Truncations: 800 of 800 flagged** (every cut of every file tried; the spike's own cuts were also all
  flagged).
- **A random single-byte flip is flagged** for the weights in 3 of 200 trials, for `optimizer.pt` in 1
  of 200, for `config.json` in 138 of 200 and for `vocab.json` in 179 of 200. **The tensor data is not
  checked, by design** (a header-only check): a flipped bit in the weights or in the optimizer state is
  not found. A flip that leaves `config.json` or `vocab.json` valid JSON passes. So "complete" means "a
  save that finished", not "bytes that are intact"; the manifest's SHA-256 of the final step is the
  record to compare against.
- 58 real `eole train` resumes from damaged directories agreed with `is_complete` on truncations and
  invalid-JSON flips.
- A bit flip in the zip central directory of `optimizer.pt` made `zipfile` raise `UnicodeDecodeError` or
  `NotImplementedError`, which the first version let through (a traceback and exit 1 from `launch` and
  `average`: 697 and 57 of 6,000 flips); it now catches `ValueError`, `NotImplementedError`,
  `OverflowError` and `EOFError` too: 3 x 1,000 seeded flips never raise (before: 162 + 24 per 4,000),
  and the real binary skips a damaged step and resumes from the previous complete one.

An incomplete directory above the resume point is reported (`incomplete step directories skipped:
[200]`) and left alone; Eole overwrites it when the run reaches that step again. Incomplete directories
are never deleted by D8.

### A checkpoint of another model

Eole IGNORES the `model` section of the config when it resumes: it builds the model that the
checkpoint's own saved `config.json` describes. So a `step_N` directory copied by hand into a run's
`model_path` from a run of another architecture would be trained on as if it were this run's, and nothing
in Eole objects (the final review's finding M-1). Before `launch` chooses a resume point it therefore
compares the `model` section of each candidate's `config.json` with the config's own, for the keys both
name: `architecture`, `hidden_size`, `heads`, `transformer_ff`, `embeddings.word_vec_size` and
`embeddings.position_encoding_type`, and the layers, width, heads, feed-forward size and position scheme of
the encoder and of the decoder (`checkpoints.architecture_mismatch`; the layout is the one Eole 0.6.4 wrote
into `step_20/config.json` of a real tiny run, copied into `tests/support/step_dirs.py`).

- A complete directory that differs in any of those keys is not a resume point: it is skipped like an
  incomplete one (listed in `incomplete step directories skipped:`, a wording that also covers it) and
  Eole overwrites it if the run reaches its step.
- When NO directory is left and some were skipped for their model, `launch` (and `--write-only`) exits 2
  and says which: `no step directory in <model_path> is of the model this config trains ... step_100:
  config.json: model.hidden_size is 64 in the checkpoint but 128 in the config. Remove it, or use a new run
  directory`. It names the newest such directory. Nothing is written and Eole is not started. Real binary,
  a real 64-wide `step_50` copied into the checkpoint directory of a 128-wide run (with its launch
  record): exit 2 in 0.8 s with that message, Eole not started; the same checkpoint copied into a 64-wide
  run was accepted (`resumed from: 50`).
- A number written as decimal text in the config equals the number (Eole accepts `hidden_size: "64"` and
  saves 64), so quoting numbers does not make a run's own checkpoint look foreign.
- A key that either side does not have is not compared, so a checkpoint whose `config.json` has no
  `model` section is resumable, and so is a same-width checkpoint of another DEPTH when the config does not
  set `encoder.layers`/`decoder.layers`, `heads` or `transformer_ff` (the validator requires only the widths
  and the position scheme; `teacher.yaml` sets them all: set them in every config you train). **This is a guard against a mistake, not proof of where a checkpoint came
  from:** a copy of the SAME architecture from another run (another corpus, another seed, another
  optimiser state) passes and would be resumed from. Operator rule: never copy checkpoints into a run's
  `model_path`; start a new run directory.

## Averaging

`rinchen train average --checkpoint-dir $RUN/ckpt --last N` calls `average_checkpoints` on the newest N
COMPLETE step directories (at least two, of one checkpoint directory, none given twice, the same tensor
names, dtypes and shapes). The mean is plain (every checkpoint weighs the same) and taken tensor by
tensor: each floating-point tensor is summed in float64, divided by the number of checkpoints and
rounded to nearest, ties to even, in its dtype. bfloat16 has no numpy dtype and goes through its bit
pattern (the upper 16 bits of a float32), rounded in two steps, float64 to float32 to bfloat16. A tensor
that is not floating point, or that is bit for bit equal in every checkpoint (the position buffers, the
`layer_scalar`s of the tiny model: 4 of 62), is taken from the last checkpoint. A NaN or infinity in any
checkpoint is refused. The weights are read and written tensor by tensor, so memory depends on the largest
tensor (peak about 9x it, float64 temporaries), the safetensors file is written by its format in the order
of the names, so the bytes are a function of the inputs, and `safetensors` itself checks the result.

The output directory (refused if it exists, lies in an input, or is named `step_<N>` beside the inputs):
the averaged `model.00.safetensors`, the files beside the weights in the LAST checkpoint (`config.json`,
`vocab.json`, the SentencePiece artifact; not `optimizer.pt`), `averaged_from.json` (the sources with the
SHA-256 of their weights, the output's SHA-256, the tensor counts) and `manifest.json` (stage
`train_average`, written last; if it cannot be written the directory is removed). Measured for the tiny
run: `averaged_from.json` 1,567 bytes, 62 tensors of which 58 averaged, output 3,322,115 bytes in all.
Three things follow:

- **It is not a checkpoint.** It has no optimizer state (`is_complete` names `optimizer.pt`) and is never
  named `step_N`: a resume or another average cannot pick it up. It is for `eole predict` and the
  CTranslate2 converter (the spike's averager's output ran in both).
- The `config.json` copied from the last checkpoint holds `training.train_from` when the last checkpoint was
  written after a resume (7,642 bytes against 7,457 for the step directory of the uninterrupted run).
- `average_decay` (Eole's EMA) is not an allowed key: it saves the EMA in place of the live weights in
  every checkpoint and does not restore it. Average saved checkpoints afterwards.

**Does averaging help? On the toy, no; and it can hurt.** Spike, development chrF on 400 sentences
(4 directions x 100 lines, greedy): in the steep part of the curve (600 steps) averaging old checkpoints
hurt: live@600 32.23, live@500 25.08, average(500, 600) 29.90, average(400 to 600) 25.90, average(200
to 600) 18.29, EMA(0.01) 29.87. On the plateau (1500 steps) everything is within 0.25 chrF, noise on 400
sentences: live@1500 98.61 (worst direction 96.99), live@1400 98.61, average(1400, 1500) 98.68,
average(1300 to 1500) 98.45, average(1100 to 1500) 98.45, EMA(0.005) 98.45; validation perplexity live
2.524, EMA 2.540. The gate checks plumbing only: the three-checkpoint average of the 200-step run
(steps 100, 150, 200) translated 100 dev lines per direction through `evaluate_variant` with finite
scores and no tag leaks (chrF 1.16, 27.49, 1.11, 1.16 per direction: a poor 200-step model, no threshold).
**Whether averaging helps a real teacher is unmeasured**; the toy cannot show it. The cautious rule:
average only checkpoints from the plateau, near the end, keep `keep_checkpoint` at least as large as `N`,
and compare against the last checkpoint on the dev sets before using the result.

## Reading the log and the loss gate

`parse_train_log(text)` reads the log of one attempt. Eole's lines (from the tiny run, wrapped here; each is
one line in the log):

```
[2026-11-08 15:22:42,144 INFO] Step 10/  200; acc: 0.0; ppl: 229.78; xent: 5.44; aux: 0.000; mtp: 0.000;
    attn_ent: 1.965; lr: 6.25e-04; sents:    1512; bsz: 1131/1231/151; 5053/5498 tok/s;      2 sec;
[2026-11-08 15:22:45,712 INFO] valid stats calculation
                           took: 2.819237470626831 s.
[2026-11-08 15:22:45,713 INFO] Train perplexity: 175.065
[2026-11-08 15:22:45,713 INFO] Train accuracy: 10.1401
[2026-11-08 15:22:45,713 INFO] Sentences processed: 7314
[2026-11-08 15:22:45,713 INFO] Average bsz: 1199/1149/146
[2026-11-08 15:22:45,713 INFO] Validation perplexity: 83.6869
[2026-11-08 15:22:45,713 INFO] Validation accuracy: 12.6863
[2026-11-08 15:22:45,713 INFO] Saving optimizer and weights to step_50, and symlink to $RUN/ckpt
```

- A report line is every `report_every` steps (a top-level key, 10 in the tiny config, 50 in Eole's
  default). The step is printed `Step %2d/%5d`: `Step 10/  200`, `Step 110/  200`, `Step 160/  200`, two
  blanks after the slash for a total of 200. The figures are those of the steps since the previous report.
  `ppl` and `xent` INCLUDE the label smoothing (0.1), so `xent` never reaches 0. The `tok/s` window
  includes any validation and save in it (3,815/3,763 tok/s at step 60, 56,900 to 69,114 in most other windows).
- At a validation the lines have no step number: the validation perplexity (valid xent = its logarithm)
  and accuracy, then the save. `Train perplexity` and `Train accuracy` there are cumulative since the process
  started. The first validation is at `valid_steps`.
- A NaN prints `nan` (`ppl: nan; xent: nan`, `Validation perplexity: nan`, `Validation accuracy: 0`).
- `parse_train_log` returns `reports` (`TrainReport(step, total, acc, ppl, xent, lr)`), `validations`,
  `accuracies`, `saved_steps`, `early_stopped`, `early_stop_step` (the step saved right after `Early
  Stop!`), `non_finite_step` and `non_finite_field` (the first non-finite figure over the whole log, `lr`
  included), `runs` and `unparsed_reports`. A log with several runs appended (the step goes back) is the
  LAST run, except for `non_finite_*`; lines end at `\n` or `\r`; a last line cut in the middle is read only
  as a report that reaches the `;` after `lr`. Python warnings, tracebacks and progress bars are ignored.
- `loss_gate(curve, min_drop=0.3)` returns None or a reason. A run learned when nothing is NaN or
  infinite, the `xent` of the last report is at most `0.7` times that of the first (0.7 times is accepted,
  0.7000001 times is not) and the last validation perplexity is below the first. It raises `DataError`
  with fewer than two reports or two validations. **`launch` itself applies only the NaN rule**; the
  loss gate is applied by the integration test (and is for a first attempt, above). A gate on a real
  teacher would need a threshold that the toy cannot give.

## The teacher prior

`configs/eole/teacher.yaml` holds NO per-run value (`data: {}`; `src_vocab`, `training.model_path`,
`transforms_configs.sentencepiece`, `data` and `seed` come from `--prepare-only`). **No value was
measured; each is a prior.** Master plan = `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, lines
511 to 516 (the comments of the config cite 511 to 514; the dropout, `train_steps`, `valid_steps` and
`early_stopping` of the plan are on 515 and the metrics on 516).

| Key | Value | Source |
|---|---|---|
| `model.hidden_size`, `embeddings.word_vec_size` | 1536 | master plan 512; `word_vec_size` equal to it because Eole's default 512 breaks a bigger model |
| `encoder.layers`, `decoder.layers` | 24, 12 | master plan 512 |
| `heads`, `transformer_ff` | 16, 4096 | master plan 512 |
| `position_encoding_type` | `SinusoidalInterleaved` | the hard constraint; master plan 513 |
| `share_embeddings`, `share_decoder_embeddings` | true | the project's shared vocabulary (the proxy config) |
| `compute_dtype` | `bf16` | master plan 514 |
| `world_size`, `gpu_ranks`, `gpu_backend` | 8, [0..7], `nccl` | master plan 514; `nccl` is Eole's default, spelled out |
| `timeout` | 7200 | the design (S9): longer than a validation plus a 13 GB save should take; not measured |
| `batch_type`, `batch_size`, `accum_count` | `tokens`, 16384, [8] | master plan 514 (`tokens` from the proxy); 1,048,576 padded tokens per update |
| `batch_size_multiple` | 8 | the spike (Eole's own rule is dead code) |
| `dropout`, `attention_dropout` | [0.1], not set (Eole's default [0.1]) | master plan 515; Eole default |
| `label_smoothing` | 0.1 | the proxy config |
| `optim`, `adam_beta2`, `decay_method`, `learning_rate`, `warmup_steps` | `adam`, 0.998, `noam`, 2.0, 8000 | the proxy config (itself an unmeasured prior) |
| `max_grad_norm` | not set (Eole's default 5) | Eole default; acts like `5 x world_size` (above) |
| `train_steps` | 100000 | Eole's own default, spelled out; the plan leaves it to the operator |
| `valid_steps`, `save_checkpoint_steps` | 5000, 5000 | master plan 515; the save equal to the validation by the design |
| `keep_checkpoint` | 8 | the design: more than `early_stopping + 1` |
| `early_stopping`, `early_stopping_criteria` | 6, `ppl` | master plan 515; one criterion (spike) |
| `tensorboard` | true | the design; the log directory is not set here: `launch` defaults it to `<run directory>/tensorboard` (above) |
| `transforms_configs.filtertoolong` | 256, 256 | the proxy config |
| `valid_metrics` | left out | spec S10 (master plan 516 has `[BLEU, CHRF]`) |
| `seed` | 1234 | the project default |

`configs/eole/teacher_tiny.yaml` is the gate: the D7 tiny model (64 wide, 2+2 layers, 4 heads, feed-forward
128, 2048-token batches, AdamW with Noam warmup 100 and learning rate 0.5, dropout 0) with 200 steps,
`report_every` 10, `valid_steps` and `save_checkpoint_steps` 50 and `keep_checkpoint` 4. Its comment says that
`launch` checks only NaN and that the 30 % fall is the test's.

## Prepare and ablate refuse each other

`ablate --prepare-only` writes a manifest of its own stage, `train_prepare`: it vouches for configs and data
files, not for results and checkpoints. `ablate` writes `train_ablate`. Each refuses a directory whose
manifest is the other's (`refuse_foreign_manifest`, also for `--dry-run`), so neither can replace the other's
manifest in an output directory. The consequence: **`--prepare-only` followed by a real `ablate` into the
same `--out-dir` is refused, and the other way round**. Use a separate `--out-dir` for the preparation of a
teacher run. (`launch`'s own manifest, `train_launch`, sits in the run directory and is refused the same way
if another stage's manifest is there.)

## What was not verified

- **Anything on a GPU**: bf16 and fp16, NCCL, flash attention, multi-GPU, multi-node, the memory and the
  tokens per second of the teacher shape, `GradScaler` with bf16, the dtype of the saved weights.
- **The real two-rank early stop and the multi-process classification.** Only unit tests with a faked log
  and exit code, and the spike's CPU gloo shim (Task 7 skipped).
- **Real data**: nothing was trained on a real corpus; the effect of the data stream restarting at line 1 on a
  long run with few resumes is unmeasured.
- **The write of a 13 GB checkpoint** (13.3 GB for the 128k vocabulary: ARITHMETIC) and the time it takes
  against `timeout`; `already_complete` re-hashes the final step directory (GB scale) to decide whether the
  manifest is still true.
- **`flock` on network file systems** (NFS, Lustre); a run directory on one is unchecked.
- **The teacher shape's quality** and every value of the prior table; whether averaging helps a real teacher.
- **A live abort on NaN**: the NaN gate runs at the end of an attempt only.
- **The size of a log of a long run**: the tiny 200-step log is 855 lines and 46,450 bytes (most of it
  Eole's "Weighted corpora loaded so far" blocks); `launch` reads a whole log with `read_text` at the end of an
  attempt and at the next launch. Not measured for 100,000 steps.
- **Windows and macOS** were not run (`flock` is POSIX).
- Verified, to close an open question: `eole build_vocab` on a resume config is fine. A resume with
  `vocab.src` deleted works: the launcher builds the file again from the resume config (the spike: `eole
  train` needs the file to exist, and ignores its content).

### Known and left open by the reviews and by this runbook

- `--write-only` leaves an existing manifest with a stale `command.sh` digest (detectable).
- Curve metrics in the manifest come from the last attempt's log only.
- The manifest is written after the lock is released (a tiny window).
- No CLI way to force CPU on a GPU config (`--gpu-ranks` cannot be empty); the identity-mismatch message
  cannot say which key differs.
- The `bool` is not an `int` check still exists in `train/eole_data.py` and `corpus/schema.py` (left alone:
  D7 and D1 code); D8's own copies were replaced by `is_strict_int`.
- The comment of `configs/eole/teacher.yaml` cites master plan lines 511 to 514 where the teacher block is
  511 to 516 (and the dropout, validation and early-stopping values are on line 515); configs were not
  touched by the fix wave. (The `tensorboard_log_dir` item that stood here is fixed: the launcher defaults
  it.)
- A skipped step directory of another model is listed in the same `incomplete step directories skipped:`
  line as an incomplete one; its reason is in the exit-2 message only when no directory is left.

### The final review (whole branch) and what its fix wave did

An independent whole-branch review (`b1a8a47..90c3865`) found no Critical problem, two Important ones and
seven Minor ones, and reproduced the gate and the STOP gate. Fixed in the commits `D8: final review fixes`
and `D8: final review fixes, docs`: I-1 (the group is killed when Eole's leader exits non-zero, above), I-2
(the TensorBoard default, above), M-1 (the model-shape guard, above) and M-4 (the real exception class in
the `is_complete` message). Left open on purpose, each with its cost:

- **An early-stopped run is finished (M-2).** Raising `train_steps` after an early stop does nothing: the
  launch returns `early_stopped` at the old step (exit 0, `status: early_stopped`, `steps: 100`;
  `test_a_later_train_steps_does_not_restart_an_early_stopped_run` pins it). To train past an early stop
  start a NEW run directory. Cost: an operator who raised the number gets no note that it was ignored.
- **Two hand runs in one attempt log (M-3).** `command.sh` fixes attempt `<k>` when it is written and
  appends (`tee -a`) to `launch/attempt-<k>.log`, so running it twice appends two runs to one log, and the
  NaN scan (`non_finite_*`) spans the whole log: a NaN of the first run keeps refusing the run even if the
  second went on cleanly past it. Start a new attempt (`launch --write-only` again, or `launch`) or clear
  the log deliberately. Cost: a false refusal in the safe direction. Not reproduced with the fake (it
  cannot overwrite steps); read from `curves.py`.
- **Design notes (M-6).** `eole_cfg.py` has 837 lines (an earlier record said "about 844") and carries key
  lists, value rules, the resume rule, merge, render and file I/O; the value rules and the `train_from`
  rule would sit well in a sibling module. `average` imports `manifest.stage` for `sha256_of`. Not changed:
  cohesion, not behaviour.
- **The look is not a hold (M-7).** Between `--write-only`'s look at the lock and its writes a real launch
  can start (documented since Task 5).

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic,
CPython 3.14.10, eole 0.6.4, torch 2.14.1+cu130, safetensors 0.8.0, numpy 2.5.4, sentencepiece 0.2.1; the GPU
(RTX 5070 Ti) was hidden from every Eole run. All on 2026-11-08, once each, code at commit `3e357ae`.

| What | Value |
|---|---|
| `ablate --prepare-only` | 0.78 s |
| `launch`, 200 steps from nothing | 25.61 s; `completed`; xent 5.44 to 2.14 (60.7 %); validation perplexity 83.6869, 16.6115, 10.1611, 8.81733 |
| `launch` of the finished run | 0.73 s; `already_complete`; the tree unchanged |
| `launch` after the weights of `step_200` were cut in half | 9.60 s; `resumed from: 150`, skipped `[200]`, attempt 2 |
| separate run: 100 steps, `--write-only` for 200, then 200 | 18.07 s, 0.69 s, 13.50 s; `resumed from: 100`, first log line `Step 110/  200` |
| `average --last 3` | 0.74 s; steps 100, 150, 200; 62 tensors, 58 averaged |
| refusals and a held lock | 0.69 to 0.73 s each, exit 2 |
| the whole measurement script | 75.5 s |
| `du -sb` | run directory 19,296,131 B; a step directory 4,808,113 B; the averaged directory 3,322,115 B |
| the STOP gate test module (ledger, Task 6) | 28 tests collected from 25 functions, 2 min 38 s (about 184 s of a loaded 11 min gate); 30 from 27 after the final review's fixes |
| the gate (the tree that holds this runbook; only documents differ from `3e357ae`) | `ruff format --check`, `ruff check`, `mypy src` clean; 7114 passed, 1 skipped, 10 deselected in 10 min 34 s of pytest (10 min 37 s in all) |

The measurement scripts were read-only scratch files (`m1_tiny_teacher.py`, `m2_teacher_commands.py`,
`m3_arithmetic.py`) run with `python -I`; the arithmetic and the commands in "The 8-GPU teacher" were not
measurements and are labelled so.

## Where the code lives

`src/rinchen/train/{checkpoints,curves,average,launch,gpu_rules,launch_command,run_lock}.py` with the
extensions of `eole_cfg.py`, `eole_run.py` and `corpus_files.py`; `src/rinchen/utils/{logged_process,
integers}.py`; `src/rinchen/cli/{train,train_runs,train_common}.py`; `configs/eole/{teacher,
teacher_tiny}.yaml`; unit tests `tests/unit/test_train_checkpoints.py`, `test_train_curves.py`,
`test_train_average.py`, `test_train_launch.py`, `test_train_launch_command.py`,
`test_train_launch_handrun.py`, `test_train_launch_lock.py`, `test_train_launch_orphan.py`,
`test_train_gpu_rules.py`, `test_train_eole_cfg.py`, `test_train_eole_cfg_drift.py`,
`test_train_damaged_optimizer_cli.py`, `test_cli_train_launch.py`, `test_cli_train_average.py`,
`test_cli_train_prepare.py`, `test_utils_integers.py`, `test_support_signal_handlers.py` and
`test_eole_configs_shipped.py`; property tests `tests/property/test_train_{checkpoints,curves,
average}_property.py`; `tests/integration/test_train_tiny_teacher.py` (the STOP gate, scenarios A to I);
and the helpers `tests/support/{step_dirs,fake_launch,fake_eole_process,launch_cli,signal_handlers}.py`.
