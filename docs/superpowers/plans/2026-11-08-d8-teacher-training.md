# D8 Teacher Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable Eole launcher (`rinchen train launch`), checkpoint completeness and run-identity checks, training-log parsing with a loss gate, checkpoint averaging (`rinchen train average`), the teacher configs and a `--prepare-only` mode of `rinchen train ablate`, with a tiny teacher that trains 200 CPU steps, resumes, completes and is averaged as the STOP gate.

**Architecture:** `eole_cfg` gains the allow-list keys and the `resumes` rule; `checkpoints` and `curves` are pure readers; `average` is a pure writer; `launch` composes them with `run_logged`; the CLI writes manifests last.

**Tech Stack:** Python 3.14.10, Eole 0.6.4 (subprocess only), safetensors, torch (read only through safetensors), polars, pydantic v2, typer, pytest + hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-08-d8-teacher-training-design.md` (binding; findings 1-9, rulings S1-S10). Spike evidence: `.superpowers/sdd/2026-11-08-d8-teacher-training/spike-findings.md` (with scratch scripts that run: `check_integrity.py` is the completeness check, `average_ckpt.py` the averager). Master plan: `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, milestone D8.

## Global Constraints

- CPython 3.14.x only (`uv run`); every script entrypoint asserts `sys.version_info[:2] == (3, 14)`. No new dependency (safetensors and torch are already installed with Eole; check `uv.lock` before importing anything not declared in `pyproject.toml`: if safetensors is not a declared dependency, read the safetensors header and tensors with the standard library as the spike's checker allows, or report the gap; do NOT edit `pyproject.toml`/`uv.lock`).
- Every model trained from random init. Resuming OUR OWN interrupted run from a complete `step_N` directory inside the same run directory is the only use of `train_from`; it is never accepted for any other path. `position_encoding_type: SinusoidalInterleaved` in every Eole model config; never relative, rotary, ALiBi, sliding-window or grouped-query attention keys.
- NO GPU: this machine has one that the owner did not authorise for a run. Every Eole subprocess runs on the CPU with `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`; no test and no script touches the GPU or NCCL. Anything GPU-side is read from source and labelled UNMEASURED.
- Eole runs only as a subprocess of the `eole` console script next to the interpreter. Unit tests never start Eole (fakes through seams); only integration tests do, on the CPU with the threads pinned.
- Never network access in tests. Nothing written under `data/` or `models/` by tests (`tmp_path`). Fixtures at most 2 MB in total (generate data in tests).
- Never Inner Mongolian data, never galig/Latin informal Mongolian, nothing about licensing or legal topics (code, docs, comments, commit messages). Never tune on FLORES+ devtest or read any test split.
- Tests before code; numeric and exact gates are gates (never loosen). Never weaken, skip or delete a failing test. Deterministic seeds everywhere; never silently drop a thing: report what was skipped (incomplete checkpoints, ignored files) and why.
- D5 and D7 must not change behaviour: their tests stay green; edits to them are limited to import and patch targets a move makes necessary.
- Commit subjects start with `D8: `; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Gate: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -m "not network" -q` (never pass `-W error`: it crashes the SentencePiece import; the gate takes about 7 minutes). No file mode changes. Match local style (see `src/rinchen/train/`, `src/rinchen/cli/train.py`).
- Invisible, control, look-alike and traditional-script characters in `.py` files only through `chr(0x....)`; visible Cyrillic letters are fine.
- Any path that reaches polars uses `glob=False`; any blocking wait on the main thread goes through `run_logged` (it wakes for Ctrl-C and handles SIGTERM/SIGHUP, respecting an inherited ignore); paths given to Eole are refused when they hold `$`, `[`, `]`, `*` or `?` (`reject_dollar`, `reject_glob_characters`).
- Work on branch `d8-teacher-training`; do not merge, push or create branches.

## Review Focus

Inputs the spec implies but no happy path exercises, most likely to bite first:

1. Step directories: a half-written newest directory, a directory missing one file, a zero-byte file, a symlink loop, a non-numeric `step_x`, `step_0`, huge step numbers, a model_path that does not exist or is a file, root symlinks pointing to a directory that was deleted, two directories with the same number after a rename (Tasks 2, 4).
2. Resume decisions: identity mismatch by one character of the config, `train_steps` increased/decreased/equal, `launch.json` missing/corrupt/foreign, a finished run, a run whose newest complete step is below `train_steps`, `train_from` smuggled into a base config or an override, a resume YAML written but never used, two launches into one run dir (Tasks 1, 4, 6).
3. Logs: NaN/inf anywhere (including in `lr`), a log cut in the middle of a line, Python warnings interleaved, `Early Stop!` without a final checkpoint, a log with two runs appended, progress bars with carriage returns, step numbers going backwards after a resume (Task 2).
4. GPU rules: `world_size` 2 with empty `gpu_ranks`, ranks not starting at 0, duplicate ranks, negative ranks, `timeout` too small, 81 corpora, `early_stopping_criteria` as a list, `keep_checkpoint` too small, an override that contradicts the config (Task 4).
5. Averaging: one directory, two runs mixed, different dtypes or shapes, bf16 tensors (float64 mean then cast back), a missing tensor, NaN in an input (refuse), output inside an input, an existing output directory (Task 3).
6. Signals and orphans: Ctrl-C/SIGTERM/SIGHUP during `launch` (the Eole process group must die; the run must be resumable afterwards), `nohup` (inherited ignore) (Tasks 4, 5).
7. D5/D7 regression through the allow-list additions and the new shared helpers (Task 1).

---

### Task 1: Allow-list additions, the `resumes` rule and the teacher configs

**Files:**
- Modify: `src/rinchen/train/eole_cfg.py`
- Create: `configs/eole/teacher.yaml`, `configs/eole/teacher_tiny.yaml`
- Test: `tests/unit/test_train_eole_cfg.py` (extend), `tests/unit/test_train_eole_cfg_drift.py` (extend), `tests/unit/test_eole_configs_shipped.py` (extend)

**Interfaces:**
- Produces: the key additions of spec section 5 and `validate_train_config(cfg, *, resumes: bool = False)`; the two YAML files.
- Consumes: D7's `eole_cfg` (read it whole first), `configs/eole/multilingual_tiny.yaml`, `multilingual_proxy.yaml`, spike findings Q4 to Q6.

- [ ] **Step 1: Write failing tests.** Every added key is accepted at its level and is a field of the installed Eole at that level (drift test, failing message names key and level); `average_decay`, `average_every`, `accum_steps`, `dropout_steps`, `reset_optim`, `update_vocab` stay unknown keys. `train_from`: rejected by default (unknown key message unchanged); with `resumes=True` accepted only when it is a string equal to `<training.model_path>/step_<digits>` (same parent after resolving, no `..`), anything else a `ConfigError` (a foreign path, the model_path itself, a file-like name, a non-string, a `step_N` of another run directory); `resumes=True` does not relax any other rule. Values: `early_stopping` a non-negative int (bool refused), `early_stopping_criteria` a string or a list of strings, `timeout` a positive int, `master_port` an int in 1..65535, `gpu_backend` a non-empty string, `self_attn_backend` a non-empty string, `batch_size_multiple` a positive int, `tensorboard` a bool; the existing value rules for old keys are unchanged. The shipped `teacher.yaml` and `teacher_tiny.yaml` pass `validate_train_config` after a minimal data merge AND `eole.config.run.TrainConfig(**cfg)` (import Eole inside the test function), carry SinusoidalInterleaved, no forbidden keys, no `train_from`; `teacher.yaml` carries a comment that every value is an unmeasured prior and cites the master plan lines (511 to 514); `teacher_tiny.yaml` has 200 steps, `report_every` 10, `valid_steps` = `save_checkpoint_steps` = 50, `keep_checkpoint` 4, dropout 0, the D7 tiny model shape; `teacher.yaml` follows spec S9 (hidden 1536, 24+12 layers, 16 heads, ff 4096, bf16, tokens 16384 with `accum_count` [8], dropout 0.1, label smoothing 0.1, valid/save every 5000, `early_stopping` 6 with criteria `ppl`, `keep_checkpoint` 8, `timeout` 7200, `batch_size_multiple` 8, `tensorboard` true, `world_size` 8 and `gpu_ranks` [0..7], `compute_dtype: bf16`, `gpu_backend: nccl`).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Keep D7's behaviour for every existing call (`resumes` defaults to False).
- [ ] **Step 4: Run the tests and the gate; mutation-check the rules (drop each; a test fails).**
- [ ] **Step 5: Commit** (`D8: allow-list additions, the resume rule and the teacher configs`).

### Task 2: Checkpoint completeness, run identity and log curves

**Files:**
- Create: `src/rinchen/train/checkpoints.py`, `src/rinchen/train/curves.py`, `tests/support/step_dirs.py` (builds fabricated complete step directories with real tiny safetensors, an `optimizer.pt` zip, JSON files)
- Test: `tests/unit/test_train_checkpoints.py`, `tests/unit/test_train_curves.py`, `tests/property/test_train_curves_property.py`

**Interfaces:**
- Produces: `step_directories`, `StepDirectory`, `is_complete`, `latest_complete`, `RunIdentity`, `identity_of(cfg)`, `write_launch_record`, `read_launch_record`, `LAUNCH_FILE`; `parse_train_log`, `TrainCurve`, `loss_gate` (spec section 5; the exact dataclass fields are yours to fix: document them).
- Consumes: spike findings Q1 (completeness recipe: the spike's `check_integrity.py` in the scratchpad `d8-spike/scripts/` is the reference), Q3 (log formats; real lines are in the spike's `work/*/train.log`).

- [ ] **Step 1: Write failing tests.** `is_complete`: a complete directory is None; each of the four files missing, each truncated at several lengths (weights 3,288,000/1,500,000/100 bytes analogues on your tiny files, optimizer, config, vocab), garbage JSON, an empty file, a directory instead of a file, a symlink to nothing: each a distinct problem string naming the file, never raising. `step_directories`: sorted by number; ignores non-numeric `step_x`, `step_`, files named `step_5`, the root symlinks and unrelated files; handles a missing/empty model_path (empty list) and a model_path that is a file (`DataError`); `latest_complete` picks the highest complete one and reports skipped incomplete higher ones; step numbers are exact (`step_007` is not `step_7`: decide and test one rule: only canonical `step_<digits without leading zero>`). Run identity: `identity_of` ignores `training.train_steps` and `training.train_from`, is stable under key order, changes with any other key (property test), the launch record round-trips, a corrupt/foreign/extra-key/wrong-type record is a `DataError`, writes are atomic. `parse_train_log`: real lines copied from the spike logs (two or three literal samples), NaN/inf in xent/ppl/acc/lr and in `Validation perplexity: nan` set `non_finite_step`, early stop detection (`Early Stop!`, the stop step from the following `Saving ... to step_N`), saved steps, carriage-return progress noise and warnings ignored, a log cut mid-line, two runs appended (steps going backwards: the LAST run is parsed, with a flag), empty text gives an empty curve; `loss_gate` per spec S8 (exact 30 % boundary: 0.7 x first accepted, 0.7000001 refused; a single report is a `DataError`; NaN refused). Property tests (hypothesis) for the parser on generated logs.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Read safetensors with the `safetensors` package if it is a declared dependency (check `pyproject.toml`/`uv.lock`), else with the header format directly; read `optimizer.pt` only through `zipfile`.
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D8: checkpoint completeness, run identity and training-log curves`).

### Task 3: Checkpoint averaging

**Files:**
- Create: `src/rinchen/train/average.py`
- Test: `tests/unit/test_train_average.py`, `tests/property/test_train_average_property.py`

**Interfaces:**
- Produces: `average_checkpoints(step_dirs, out_dir) -> AveragedModel` (spec S7).
- Consumes: Task 2's `is_complete`, `tests/support/step_dirs.py`, the spike's `average_ckpt.py` (reference), `utils.part_file`, `manifest.stage.sha256_of`.

- [ ] **Step 1: Write failing tests** on fabricated directories: the float64 mean is exact for hand-computed tensors (float32 and a bfloat16-like dtype if safetensors supports it in your reader: otherwise float16), non-float tensors and equal buffers come from the last directory, the output has the right files and no `optimizer.pt`, `averaged_from.json` lists the sources, every source sha256 and the output sha256, the output weights are byte-identical for the same inputs (determinism) and independent of the order of the input list only when sorted by step (the function sorts by step and refuses duplicates), one directory or an incomplete directory or directories of different shapes/dtypes/names is a `DataError`/`ValueError` naming the problem, NaN/inf in an input refused, an output inside an input or an existing output directory refused (no overwrite), nothing is left behind on a failure (no `.part`), the SPM artifact (any file next to the weights other than the four known ones, copied from the last directory) is preserved. Property test: the mean of N copies of the same tensor equals the tensor; averaging is associative with equal weights within float64 tolerance.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement** (float64 accumulation tensor by tensor so peak memory is one tensor times the number of inputs; atomic directory rename).
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D8: checkpoint averaging`).

### Task 4: The launcher

**Files:**
- Create: `src/rinchen/train/launch.py`
- Test: `tests/unit/test_train_launch.py`, `tests/unit/test_train_launch_command.py`, `tests/support/fake_launch.py`

**Interfaces:**
- Produces: `launch(...) -> LaunchResult`, `render_command_sh(...)`, the S5 rules as a function `check_gpu_rules(cfg, world_size, gpu_ranks)` (names are yours to fix: document them).
- Consumes: Tasks 1 to 3, `eole_run` (`training_environment`, `eole_command`, `eole_commands`), `logged_process.run_logged`, D7's `eole_cfg` (`merge`, `render`, `read_config`, `dump_config`, `reject_dollar`, `reject_glob_characters`).

- [ ] **Step 1: Write failing tests.** Spec S2 to S6 in full, with `run_logged` replaced by a fake that writes a scripted `train.log` and fabricates step directories (build them with `tests/support/step_dirs.py`): a fresh launch (no resume, `launch.json` written BEFORE Eole starts, build_vocab runs only when `vocab.src` is absent); the commands and environment (threads pinned, the GPU hidden exactly when the effective `gpu_ranks` is empty, `CUDA_VISIBLE_DEVICES` untouched otherwise); `already_complete` runs nothing; resume from the newest complete step with the resume YAML content exactly the base plus `train_from` (assert the YAML and that the base file is byte-identical afterwards), incomplete newer directories reported; `train_steps` extended resumes; `train_steps` lowered below the newest complete step is a `ConfigError` naming both numbers; identity mismatch (one character of the data path, the learning rate, the model size, the world size via the override) is a `ConfigError` before anything runs, a missing `launch.json` with step directories present is a `ConfigError`, a corrupt `launch.json` a `DataError`; outcomes: `completed`, `early_stopped` for rc 0 and for rc 1 + `Early Stop!`, `UpstreamError` for rc 1 without it, for rc 0 without the final step, for a NaN in the log (even when the run finished), for a timeout; the GPU rules each refused with its message (spec S5: world_size/gpu_ranks combinations, `timeout`, 81 corpora, criteria list, keep_checkpoint); `write_only` writes `command.sh`, the attempt config and `launch.json`, and runs nothing; a real launch holds a non-blocking `flock` on `launch/lock` (a second launch into the run, and `write_only` beside it, are `ConfigError`s); `command.sh` golden text for a CPU run, an 8-GPU run (`world_size` 8, ranks 0 to 7, `CUDA_VISIBLE_DEVICES` requested) and a resume (the golden strings live in the test, with an injected `eole` path), the file is executable-free text (mode 0644: no mode bits set by us) and `bash -n` accepts it; `$`, glob characters and relative run dirs are handled as D7 does; a launch under a KeyboardInterrupt/SystemExit from the fake leaves the run resumable and the next launch resumes; nothing is written outside `run_dir` (and the config's own paths).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** `launch` is orchestration: the rules live in `eole_cfg`, `checkpoints`, `curves`.
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D8: the resumable launcher`).

### Task 5: CLI

**Files:**
- Modify: `src/rinchen/cli/train.py`; D7's ablation modules only as far as `--prepare-only` needs (an added function, no behaviour change of the existing flow)
- Create: `tests/support/launch_cli.py`
- Test: `tests/unit/test_cli_train_launch.py`, `tests/unit/test_cli_train_average.py`, `tests/unit/test_cli_train_prepare.py`

**Interfaces:**
- Produces: `rinchen train launch`, `rinchen train average`, `rinchen train ablate --prepare-only` (spec section 5); manifests.
- Consumes: Tasks 1 to 4; study `src/rinchen/cli/train.py` (D7's `ablate`), `cli/synth.py` and `cli/errors.py` for the patterns (`rinchen_errors_as_exit_codes`, `refuse_foreign_manifest`, hash-before-read, early `out_dir` creation, `begin_commit`, manifest last, `tool_versions`).

- [ ] **Step 1: Write failing tests** (CliRunner, fake seams, never Eole): options and defaults (`--help`), required options, exit codes (`ConfigError` 2, `DataError` 3, `UpstreamError` 4, other/OSError 1, interrupt 130, SIGTERM 143, SIGHUP 129 via the existing handlers), `launch`: prints one fact per line (status, resumed_from, steps, checkpoint path, log path, command.sh path), `--write-only` prints only the command.sh path and runs nothing, the manifest is written last into the run directory (stage `train_launch`; inputs: the config hashed before it is read; outputs: the final step directory's four files with their sha256, `launch.json`; metrics: steps, resumed_from, early_stopped flag, final train xent, last validation ppl, `launch.attempts`), a foreign manifest is refused, `--gpu-ranks 0,1,2,3` parsing (`0,1,2`, spaces, empty, negative, duplicates, non-numeric), `--world-size` without ranks; `average`: `--checkpoint-dir D --last N` picks the newest N complete steps of D (N at least 2), `--out` default `averaged` beside D, existing output refused, manifest (stage `train_average`) with the sources' and output's hashes, exit codes; `ablate --prepare-only`: renders every variant's corpora, validation files and `config.yaml` exactly as `run_ablation` would (same function) and trains nothing, prints the config paths, writes a manifest (stage `train_prepare`) whose outputs are the configs and the data files' sidecars, refuses `--force` combined with it (nothing to force) and makes `--dry-run` win over it; the existing `ablate` tests stay green.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement the commands.**
- [ ] **Step 4: Run the tests and the gate; mutation-check.**
- [ ] **Step 5: Commit** (`D8: rinchen train launch, average and prepare`).

### Task 6: The tiny teacher (the STOP gate)

**Files:**
- Create: `tests/integration/test_train_tiny_teacher.py`
- Modify: `tests/support/tiny_variant.py` only if a helper must be shared

**Interfaces:**
- Consumes: everything. The matrix for the gate is a scratch copy written by the test: one variant (`tag_placement: [src]`, temperature 0.5, bt ratio 0.5), `base: configs/eole/teacher_tiny.yaml`.

- [ ] **Step 1: Write the integration test** (real `eole`, the real `rinchen` binary through `subprocess` like `tests/integration/test_train_tiny_ablation.py`; marker `subprocess`; no sleeps; subprocess timeouts <= 900 s). Flow: build the toy corpora and SPM in `tmp_path`; `rinchen train ablate --prepare-only` for the one-variant matrix; `rinchen train launch --config <variant config.yaml> --run-dir <variant run dir> --threads 2` with the 200-step config: exit 0, status `completed`, `launch.json`, `command.sh` present, `step_50/100/150/200` exist and are complete, the root symlinks resolve to `step_200`; `parse_train_log` of `train.log`: no NaN, `loss_gate` passes (xent(last) <= 0.7 x xent(first)), last validation ppl below the first; record the measured drop and the wall time in the report; a second `launch` exits 0 in a few seconds with `already_complete` and does not touch the checkpoint (hash unchanged); truncating `step_200`'s weights and launching again reports the incomplete directory, resumes from `step_150`, completes, `step_200` is complete again and the log shows `Step 160/ 200` (or the first report after 150); a SEPARATE run: a config with `train_steps` 100 completes, then the same config with `train_steps` 200 resumes from `step_100` (the log shows `Step 110/ 200`) and completes (identity ignores `train_steps`); a changed learning rate is refused with exit 2 and Eole is not started (no new log lines, no new step dirs); `rinchen train average --last 3` writes the averaged model with `averaged_from.json`, and `evaluate_variant` (the D7 harness) translates the dev set with it: 100 lines per direction, finite scores (NO quality threshold); the manifests' checkpoint hashes equal the files'; Ctrl-C robustness is covered by the unit tests and D7's integration tests, but run ONE real SIGTERM test here: start `launch` of the 200-step run in a session of its own, send SIGTERM to the `rinchen` process 8 s in, assert exit 143, no stray `eole` process of the run (check the process group), and a following `launch` resumes (from the newest complete step if any, else it starts fresh because only `launch.json` exists: decide and test the rule: a launch record without step directories starts fresh).
- [ ] **Step 2: Run it and see it fail** (before Tasks 4/5 are complete it cannot; run it now against the finished code and fix defects found in the earlier tasks with small commits, reporting each).
- [ ] **Step 3: Record the measurements and run the full gate.**
- [ ] **Step 4: Commit** (`D8: the tiny teacher gate`).

### Task 7: Multi-process launch on CPU (optional; stop and report if it needs more than modest code)

**Files:**
- Create: `tests/support/gloo_shim/sitecustomize.py`, `tests/integration/test_train_two_rank_cpu.py`

- [ ] **Step 1:** Port the spike's shim (`/tmp/claude-1000/-home-tsee-Personal-rinchen/a3e391bc-6f17-4472-b34a-6cd321bc0900/scratchpad/d8-spike/shim/sitecustomize.py`: it drops `device_id`, swaps `all_gather_list` for `all_gather_object`, makes `torch.cuda.current_device` return `'cpu'`; activated by `PYTHONPATH` and an environment variable; it changes neither Eole nor the repo) into a TEST-ONLY support directory with a header saying so, and write an integration test: `launch` with `--world-size 2 --gpu-ranks 0,1` on the tiny teacher config with `gpu_backend: gloo` and `timeout` 20 and early stopping provoked (a large learning rate as in the spike's `q5_early_lr8`), the shim enabled through the launcher's environment hook (add a narrow `extra_env` parameter to `launch` for this, tested), asserting the real spawn works, the outcome is `early_stopped` although the exit code is 1, the final checkpoint is complete, and no stray process remains. If the shim breaks with this Eole version or needs more than about 100 lines, STOP, do not commit a broken test, and report.
- [ ] **Step 2:** Commit (`D8: two-rank CPU launch test`) or report why not.

### Task 8: Runbook and records

**Files:**
- Create: `docs/runbooks/08_teacher.md`
- Modify: `docs/superpowers/specs/2026-11-08-d8-teacher-training-design.md` (to the finished code), `STATE.md`, `DECISIONS.md` (CRLF; additions only), `CLAUDE.md` (layout and Current phase; not the Hard constraints), `docs/VERIFY.md`

- [ ] **Step 1: Take the measurements the runbook needs** (read-only scripts in a new scratchpad directory): the tiny teacher through the real CLI once (stdout, timings, curve numbers, directory tree and file sizes), the dry-run/write-only outputs, an 8-GPU `command.sh` for `teacher.yaml` (rendered with a fake eole path), the effective-batch arithmetic table for the teacher prior (world_size x accum x batch tokens), the size of a teacher checkpoint (parameter count of the prior config computed from the model shape, times 4/8 bytes: derived, labelled as arithmetic not measurement).
- [ ] **Step 2: Write the runbook** (format of runbooks 06 and 07): what D8 builds and does not; module table; commands with real output; the launch state machine (fresh/resume/already_complete/early_stopped) and identity rules; the 8-GPU command and what is unmeasured about it (every GPU claim labelled); checkpoint cadence, `keep_checkpoint`, averaging and early-stopping rules with the spike's hazards; reading the loss gate; signals; the data-position-not-restored limitation; teacher prior table with its sources; a "What was not verified" section.
- [ ] **Step 3: Amend the spec**, update STATE.md (D8 done on branch `d8-teacher-training` stacked on `d7-multilingual-ablations`, not merged, not pushed; evidence; open gates; next D9), DECISIONS.md (entry `2026-11-08: D8 teacher training - the rulings, made without a review round`), CLAUDE.md layout, VERIFY.md rows (`D-EOLE-RESUME-DATA`, `D-EOLE-EMA-SAVES-AVERAGE`, `D-EOLE-NAN-CONTINUES`, `D-EOLE-EARLYSTOP-RANK0`, `D-EOLE-SPAWN`, `D-EOLE-CKPT-NOT-ATOMIC`, `D-EOLE-SILENT-NOOP` as the spike supports them).
- [ ] **Step 4: `rg -n -i "licen|legal"` over the added lines is clean; run the full gate; commit** (`D8: runbook, spec amendments and state and decision records`).
