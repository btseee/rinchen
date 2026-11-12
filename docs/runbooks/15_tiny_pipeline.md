# 15 - The tiny pipeline

## Scope

D15 runs the whole project once, small, on the CPU, and says what happened. One command, `rinchen tiny run --profile tiny` (the recipe `just tiny`), takes seven committed fixture files through fifteen stages (the environment check, the fetch of local files,
the corpus, the filter, SentencePiece, the script converter, synthetic data from a fake chat server, the preparation, training and averaging of a tiny multilingual Eole model, two CTranslate2 exports, their parity, the service in process, robustness and evaluation)
and writes `report.json` (schema `rinchen.tiny/1`), `report.md` and, last, `manifest.json`. It builds the command and its library (`rinchen.tiny`), the closed configuration (`rinchen.config.tiny`, `configs/tiny/tiny.yaml`), the Eole base config `configs/eole/tiny.yaml`,
the fixtures `tests/fixtures/parallel/*.tsv` with their generator `scripts/build_parallel_fixtures.py`, the `tiny` job of `ci.yml`, the gate and this runbook. The design, with the rulings P1 to P20 and a section of the amendments made while building (A1 to A31), is
`docs/superpowers/specs/2026-11-11-d15-tiny-pipeline-design.md`; the decisions are condensed in the entry "2026-11-12: D15 tiny pipeline: the rulings, the build and the limits" of `DECISIONS.md`; the upstream facts are in `docs/VERIFY.md` (section "D15 findings"). The facts about the stages,
their costs and the candidate fixtures come from a spike made before the design (`.superpowers/sdd/2026-11-11-d15-tiny-pipeline/spike-findings.md`, findings F1 to F18; Eole 0.6.4, CTranslate2 4.8.4, sacrebleu 2.6.0, SentencePiece 0.2.1, CPU, offline).

**What D15 is.** A machinery check. It proves that the stages of D1 to D13 still fit together: that the commands accept each other's files, that every guard, manifest and exit code is the one the operator's commands have, that a model trained from random initialisation on templated sentences
is learned, exported twice, found equal to itself in Eole and in CTranslate2, served, measured and reported, and that a missed bound or a broken input makes the run red and says where. Every number below is a toy number: the model memorises a grammar of 37 nouns and ranks nothing.

**What D15 is NOT.** Nothing here is a statement about the quality of any real model, and nothing in it has run on GitHub.

- **No GPU.** No stage touches a device; every child runs with `CUDA_VISIBLE_DEVICES=""`, on the CPU, with `gpu_ranks: []`; `auto` is refused everywhere and `--device cuda` is never run. The GPU on this machine was not authorised. Everything GPU-side (the GPU-side Eole, `float16`, `int8_float16`) is **UNMEASURED**.
- **No network and no download.** `rinchen data fetch`, `rinchen eval suite fetch` and `rinchen release hub --upload` are never run; the fetch stage copies local files; no model is downloaded (GlotLID, LaBSE and the QE scorer are replaced or skipped by reason); no FLORES+ is used at all.
  The one socket the pipeline opens is the loopback listener of the synthetic-data stage.
- **No real inputs.** No real corpus, no real generator behind the synthetic stage, no real LID, LaBSE or QE model and no neural metric scorer exist here: the stand-ins rank nothing and the neural seam of D13 is exercised with planted score files.
- **No mechanism.** D15 builds no placeholder masking, no terminology engine, no decoder and no metric (D12 and D13 declined to; nobody owns them); `placeholder_acc` is measured by D12's instrument over rows a model can learn.
- **No resume, no partial run.** A run starts from nothing; there is no `--stages`, `--from-stage` or `--keep`.
- **No `full` run.** `rinchen tiny run --profile full` is refused (exit 2): no `configs/tiny/full.yaml` exists, and scale-up is each stage's own command with its own full configuration (the mapping table below).
- **No release stage and no `maintain check` stage.** `just tiny` does not call `rinchen maintain check` (D14's open question 9, decided here: no, ruling P15 of the spec); `just accept` stays a stub.
- **No GitHub Actions.** The `tiny` job of `ci.yml` and the nightly step were checked statically and their commands ran natively; no runner ever ran them. "CI runs `just tiny` green on 3.14" is met only as "the commands of the workflow run green natively here".

Tags used below: **gate** (measured by the gate of Task 9, commits `3862635` and `5dab820`, on 2026-11-12, on a 20-core CPU shared with other tenants (load average 1 to 29 while it ran), the GPU hidden, no network; the figures are the ones the gate modules print in their docstrings and the ledger records, and the figures of the real-binary gate were
not read again for this runbook: the full gate that closed Task 10 ran its modules again, pass or fail only), **measured here** (run or counted for this runbook with the code at `5dab820`: `rinchen tiny run --help`, `rinchen tiny run --dry-run`, `wc`, `sha256sum` and `git diff --numstat`; no other command of the pipeline was started), **spike** (run in the design spike of 2026-11-11), **source-read** (read from the code, never run),
**ARITHMETIC** (a formula applied to values that were measured), **ledger** (a figure of the build's ledger, not re-measured), **UNMEASURED** (nothing was run). A TIME moves by a third from one run to the next on this shared machine; a count, a score or a digest does not. A number that was not obtained is not given.

## The stages

Fifteen ids, a closed list in this order (`rinchen.config.tiny.STAGE_IDS`; a test pins it). A stage runs as the REAL `rinchen` command through the runner of the context whenever a command exists that does the whole job from files (own session, process group stopped on a timeout or an interrupt, the output in
`<stage dir>/logs/<stage>-<step>.log`, stdin closed, `child_environment`: no `*_TOKEN`, `*_SECRET`, `*_PASSWORD` or `*_API_KEY`, the device hidden, the Hub offline), and as a library call in process only when no command can (the fetch writes under `data/`, the filter takes no stand-in, the corpus build has no command,
the service is in process by the master plan). Results are read from the files a command wrote, never from what it printed (the one exception is the five lines of `convert-script eval`, read by shape). The seconds are the manifest's, **gate**, recipe alone.

| Id | What runs | Process | What it proves | What it does NOT prove | Seconds |
|---|---|---|---|---|---:|
| `environment` | `maintain.environment.check_environment(read_lock(uv.lock), installed_versions())` | library | the installed distributions are the lock's 186 names at the locked versions, so the manifest's `tool_versions` are this run's | anything about behaviour | 0.05 |
| `fetch` | `acquire.registry.fetch_source` over seven `kind: local` entries (`configs/data/sources.tiny_parallel.yaml`) | library | each copy equals its pin in the profile | any network fetch (never run) | 0.00 |
| `corpus` | `normalize`, `split`, `corpus_frame(created=<fixed>)`, `write_corpus`: ONE row per pair, in the direction it was acquired | library | the fixtures become D4's corpus contract: 770 rows (ARITHMETIC: 650 train, 40 dev, 70 test, 10 `holdout`) | a real corpus | 0.03 |
| `filter` | `filter.pipeline.run_pipeline` with the planted stand-ins, then the expansion of the kept `train` and `dev` rows to both directions | library | the filter has teeth: all 10 planted rows get exactly their reasons, 4 of 760 authentic rows are flagged (counts below) | a real filter threshold, LID, LaBSE or QE | 0.43 |
| `spm` | `rinchen spm train`, `rinchen spm check` | CLI | the trainer accepts the text at 640 pieces and `check` exits 0 (the coverage lines are not read) | a 1,000-piece model; coverage of real text | 1.83 |
| `converter` | `convert-script build-data`, `train --profile tiny`, `export`, `eval` over `tests/fixtures/script_conv/tiny_words.tsv` | CLI | D5's chain trains, exports and loads; it feeds the `converted` arm of D12's section | a converter that spells like a reader | 41.24 |
| `synth` | `rinchen synth run bt_eng_khk ... --no-qe` against the loopback `FakeChatServer` | CLI + loopback | D6's real binary talks to a server over a socket and writes `bt` rows | any real generator, prompt or QE | 1.05 |
| `prepare` | `rinchen train ablate ... --prepare-only` over a one-variant matrix | CLI | the training data and Eole config of the variant | nothing is trained here | 0.97 |
| `train` | `rinchen train launch ... --threads N --timeout T` (1,500 steps) | CLI | a model from random initialisation learns the fixture | quality, speed or convergence of a real model | 44.16 |
| `average` | `rinchen train average --last 2` | CLI | the last two checkpoints are averaged | whether averaging helps | 0.92 |
| `export` | `rinchen export convert` at `float32` and at `int8` | CLI | two CTranslate2 exports with records and manifests | `float16`, `int8_float16`, a GPU | 6.37 |
| `parity` | `rinchen export parity` on the float32 export | CLI | Eole greedy equals CTranslate2 greedy on the held-out dev lines (the gated parity) | parity of the int8 export (it is not run on it: P9) | 3.52 |
| `serve` | `TestClient(create_app(...))` over the int8 export: `/healthz`, `/readyz`, `/v1/translate`, a refusal, `/metrics` | library, in process | the service starts, translates, refuses and counts | a socket, concurrency, a container | 0.06 |
| `robustness` | `rinchen eval robustness ... --beam-size 1` with the converter attached | CLI | D12's six values over a suite from the kept train rows | any real glossary, entity or number convention | 1.28 |
| `evaluate` | ten `rinchen eval run` / `eval report` commands (memorised int8 and float32, held-out int8, the neural seam, the report) | CLI | D13's runner, bootstrap and report over fixture suites; the int8 export equals float32 on the memorised suites | a benchmark, a baseline, a neural metric | 10.27 |

The total of the seconds is 112.2 s (**gate**; ARITHMETIC 112.18 s). The stages with parts list them in `parts`: `filter` (`script_ratio`, `numbers`, `length`, `lid`, `minhash`, `labse`, `qe`) and `evaluate` (`memorised`, `heldout`, `neural_seam`, `report`).

**The filter counts** (**gate**, R5): 770 rows scored (760 authentic and 10 planted), 756 kept; reasons duplicate 4, labse 3, len_ratio_char 3, numbers 2, script_tgt 2 (14 in all: the 10 planted rows, each with exactly its expected reason, and 4 authentic rows: two duplicates, one labse, one len_ratio_char); 4 of 760 authentic rows flagged (the profile allows 1 %,
ARITHMETIC: 7.6 rows); `train.parquet` 1,316 rows (1,246 `train` and 70 `dev`), no `holdout` row. Without `--qe-fake` the parts `labse` and `qe` are skipped by reason and the planted rows the CPU stages can detect are the ones that must match (8 of the 10; the two misaligned ones pass: source-read and spike F4; **gate** R5 asserts the planted 10 of 10 WITH the stand-ins).
The filter runs ONCE per pair and is expanded afterwards: filtering both directions makes every row a near-duplicate of its reverse (185 of 1,450 rows at the default 0.8, spike F4).

**The directories.** `01_fetch`, `02_corpus`, `03_filter`, `04_spm`, `05_converter`, `06_synth`, `07_train` (`prepare`, `train` and `average` share it; the variant is `07_train/ablate/variants/<variant>/`, with `run/ckpt` and `run/averaged`), `08_export/{float32,int8}`, `09_parity/float32`, `10_robustness`, `11_evaluate` (`suites`, `runs`, `report`, `scores`).
`environment` and `serve` write none. Each stage that runs commands keeps their output in its own `logs/` directory. A top-level `logs/` directory exists only when a stage RAISED (it holds `<stage>-raised.log`: the class of the exception and its frames, plus the message for a `RinchenError`, whose messages name places and counts and never a text);
the start of the next run removes it. The run directory was 36 MB in 208 files (**spike**, F13: 23 MB of it the four kept checkpoints with their optimiser states); the final code's directory was not measured.

## The command

```
rinchen tiny run [--profile NAME] [--out-dir DIR] [--config FILE] [--qe-fake | --no-qe-fake] [--threads N] [--budget-seconds N] [--dry-run]
```

| Option | Meaning |
|---|---|
| `--profile` (default `tiny`) | selects `configs/tiny/<profile>.yaml` (the pipeline) and `configs/profiles/<profile>.yaml` (the seed, read from the hashed bytes: a non-negative integer, else exit 2). A profile without a pipeline file is exit 2 and the message names this runbook. |
| `--out-dir` (default `artifacts/tiny`) | the output directory. It may not exist, be empty or hold only the names the command writes (`manifest.json`, `report.json`, `report.md`, `logs` and the eleven stage directories); any other entry is exit 2, counted and never named (a leftover `*.part` is counted and the message says to delete it). |
| `--config FILE` | the pipeline file in place of the profile's (the gate's control profiles; `--profile full --config FILE` is allowed and reads `configs/profiles/full.yaml` for the seed). |
| `--qe-fake` / `--no-qe-fake` | the planted stand-ins for LaBSE and the QE scorer (default: the profile's `qe_fake`, true). |
| `--threads N` (1 to 64) | the OpenMP and MKL threads of each training run and the threads of the CTranslate2 runs (default 2). |
| `--budget-seconds N` (1 to 86,400) | the wall-time budget (default 1,200). |
| `--dry-run` | checks the inputs (the fixtures against their pins, the lock, the seed, the output directory), prints the stage list with the argument list of every command and writes nothing; no stage runs and no code version is read. |

`--threads`, `--budget-seconds` and `--qe-fake` are applied to the loaded configuration with `model_copy`: the report's `qe_fake` and the `wall_time` bound are the effective ones, and `config_sha256` is the digest of the pipeline FILE as hashed, not of the effective configuration (A21).

**The order of a run.** The options are checked; the pipeline file is hashed and read through its hash; every other input (the seven fixtures against their pins, the Eole base config, the serve config, the converter's words file, the profile file with the seed, `uv.lock`) is hashed BEFORE any of them is read, and the lock, the seed and the fixtures are parsed from the
bytes that were hashed (a file that changed in between is refused); the output directory is checked; `--dry-run` ends here. Then the code version is read ONCE (before anything is written, so the output directory inside the checkout cannot make it `-dirty`; with uncommitted files in the checkout it reads `<sha>-dirty`, as the gate's first run showed),
the old manifest is removed first and then the files of the old run, the stages run (the first failure ends the run and every later stage is `skipped` for `an earlier stage failed`), the report is built and checked to read back as itself (`parse_report` and `render` must give the same text, else exit 1 and NOTHING is written), `report.json` and `report.md` are written whole, and the manifest LAST.
The command hides every GPU from its own process for the whole run. A rerun, after anything, starts from nothing.

**The worked example: `--help` and `--dry-run`** (**measured here**, the real binary at `5dab820`, `CUDA_VISIBLE_DEVICES=""`, `UV_OFFLINE=1`, an `--out-dir` that did not exist; the directory was not created, exit 0, 39 lines on standard output and none on standard error; each `<repo>` stands for the absolute path of the checkout and each `<out>` for the absolute
output directory, which the real output prints in full). `rinchen tiny run --help` prints (the real help draws its options in a box of line-drawing characters that is not reproduced here):

```
Usage: rinchen tiny run [OPTIONS]

Run the tiny pipeline: the stages from the fixtures to the report, on the CPU.
Writes report.json, report.md and manifest.json and a directory for each stage under --out-dir.
Exit codes: 0 when the verdict is pass; 4 when it is fail; 2 for a bad option, profile, config or seed, a changed input or a directory with other entries; 3 for a missing or changed fixture, a bad uv.lock or another stage's manifest in the directory; 1 when the output cannot be written; 130 when interrupted (143 for SIGTERM, 129 for SIGHUP).

--profile <str>            The profile: selects configs/tiny/<profile>.yaml (the pipeline) and configs/profiles/<profile>.yaml (the seed). A profile without a pipeline file is refused.  [default: tiny]
--out-dir <path>           Output directory: report.json, report.md, the stage directories and manifest.json.  [default: artifacts/tiny]
--config <path>            The pipeline file (YAML) in place of the one of the profile.
--qe-fake / --no-qe-fake   Run the planted stand-ins for LaBSE and the QE scorer (default: the profile's setting).
--threads <int range>      [1<=x<=64]  Threads of each training run and of the CTranslate2 runs (default: the profile's).
--budget-seconds <int range>  [1<=x<=86400]  The wall-time budget of the run in seconds (default: the profile's).
--dry-run                  Check the inputs, list the commands of every stage and write nothing.
--help                     Show this message and exit.
```

`rinchen tiny run --dry-run --out-dir <out>` printed (all 39 lines; `<variant>` is the id the preparation makes from the matrix, named by the real run):

```
environment: in process
fetch: in process
corpus: in process
filter: in process
spm: 2 commands
  <repo>/.venv/bin/python -m rinchen spm train --config <out>/04_spm/spm.yaml
  <repo>/.venv/bin/python -m rinchen spm check --model <out>/04_spm/tiny.model --corpus-dir <out>/04_spm/text
converter: 4 commands
  <repo>/.venv/bin/python -m rinchen convert-script build-data --pairs <repo>/tests/fixtures/script_conv/tiny_words.tsv --out-dir <out>/05_converter/corpus
  <repo>/.venv/bin/python -m rinchen convert-script train --profile tiny --data-dir <out>/05_converter/corpus --run-dir <out>/05_converter/run --spm-config <out>/05_converter/char.yaml --threads 2 --timeout 300
  <repo>/.venv/bin/python -m rinchen convert-script export --run-dir <out>/05_converter/run --out-dir <out>/05_converter/model --timeout 300
  <repo>/.venv/bin/python -m rinchen convert-script eval --model-dir <out>/05_converter/model --pairs <repo>/tests/fixtures/script_conv/tiny_words.tsv
synth: 1 command
  <repo>/.venv/bin/python -m rinchen synth run bt_eng_khk --input <out>/06_synth/mono.parquet --config <out>/06_synth/synth.yaml --out-dir <out>/06_synth/run --seed 1234 --no-qe
prepare: 1 command
  <repo>/.venv/bin/python -m rinchen train ablate --matrix <out>/07_train/matrix.yaml --corpus <out>/03_filter/train.parquet --corpus <out>/06_synth/run/corpus.parquet --spm-model <out>/04_spm/tiny.model --out-dir <out>/07_train/ablate --prepare-only --threads 2
train: 1 command
  <repo>/.venv/bin/python -m rinchen train launch --config '<out>/07_train/ablate/variants/<variant>/config.yaml' --run-dir '<out>/07_train/ablate/variants/<variant>/run' --threads 2 --timeout 600
average: 1 command
  <repo>/.venv/bin/python -m rinchen train average --checkpoint-dir '<out>/07_train/ablate/variants/<variant>/run/ckpt' --last 2
export: 2 commands
  <repo>/.venv/bin/python -m rinchen export convert --checkpoint '<out>/07_train/ablate/variants/<variant>/run/averaged' --spm-model <out>/04_spm/tiny.model --tag-placement src --quantization float32 --out-dir <out>/08_export/float32
  <repo>/.venv/bin/python -m rinchen export convert --checkpoint '<out>/07_train/ablate/variants/<variant>/run/averaged' --spm-model <out>/04_spm/tiny.model --tag-placement src --quantization int8 --out-dir <out>/08_export/int8
parity: 1 command
  <repo>/.venv/bin/python -m rinchen export parity --checkpoint '<out>/07_train/ablate/variants/<variant>/run/averaged' --export <out>/08_export/float32 --corpus <out>/03_filter/train.parquet --out-dir <out>/09_parity/float32 --max-lines 50 --threads 2 --timeout 180
serve: in process
robustness: 1 command
  <repo>/.venv/bin/python -m rinchen eval robustness --suite <out>/10_robustness/suites/tiny-robust.json --model <out>/08_export/int8 --out-dir <out>/10_robustness --beam-size 1 --script-converter-ct2 <out>/05_converter/model --script-converter-spm <out>/05_converter/run/converter_char.model
evaluate: 10 commands
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-eng.json --model <out>/08_export/int8 --system tiny-int8 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-rus.json --model <out>/08_export/int8 --system tiny-int8 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-mong.json --model <out>/08_export/int8 --system tiny-int8 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-eng.json --model <out>/08_export/float32 --system tiny-float32 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-rus.json --model <out>/08_export/float32 --system tiny-float32 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-memorised-mong.json --model <out>/08_export/float32 --system tiny-float32 --beam-size 1 --out-dir <out>/11_evaluate/runs --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-heldout-eng.json --model <out>/08_export/int8 --system tiny-int8 --beam-size 1 --out-dir <out>/11_evaluate/runs --robustness <out>/10_robustness --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-heldout-rus.json --model <out>/08_export/int8 --system tiny-int8 --beam-size 1 --out-dir <out>/11_evaluate/runs --robustness <out>/10_robustness --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval run --suite <out>/11_evaluate/suites/tiny-heldout-eng.json --hypotheses <out>/11_evaluate/runs/tiny-heldout-eng/tiny-int8/hypotheses.jsonl --system tiny-int8-qe --out-dir <out>/11_evaluate/runs --segment-scores metricx24_qe=<out>/11_evaluate/scores/metricx24_qe.jsonl --resamples 1000
  <repo>/.venv/bin/python -m rinchen eval report --suite <out>/11_evaluate/suites/tiny-heldout-eng.json --suite <out>/11_evaluate/suites/tiny-heldout-rus.json --run <out>/11_evaluate/runs/tiny-heldout-eng/tiny-int8 --run <out>/11_evaluate/runs/tiny-heldout-rus/tiny-int8 --out-dir <out>/11_evaluate/report
```

The commands carry no device option, no `auto`, no upload and no download (every list goes through `rinchen_argv`, which refuses those words whatever a caller gives); every `eval run` carries `--beam-size 1`; each command is bounded by its stage's timeout or what is left of the budget, whichever is less.

**The lines a real run prints** (**source-read**, `rinchen.cli.tiny_output.result_lines`; no log of a real run is quoted here): one line per stage, `<id>: ok`, `<id>: skipped (<closed reason>)` or `<id>: fail (<closed reason>)`; one per threshold, `threshold <id>: <pass|fail> (value <number or -> with up to six significant digits, bound <comparison> <limit>)`
(for `wall_time` the value is the elapsed seconds, which the report does not hold); `verdict: <pass|fail>`; and the paths `report:`, `summary:` and `manifest:`. No text of a file is ever printed.

### Exit codes

| Code | Meaning | Where it was measured |
|---|---|---|
| 0 | the verdict is `pass` | **gate** R1 (the recipe) |
| 4 | the verdict is `fail`: a stage failed or a threshold was missed; EVERYTHING is written and printed | **gate** R7 (400 steps: exit 4 after 102.2 s) and R10 (`--budget-seconds 5`: exit 4 after 5.4 s) |
| 2 | a bad option, profile, config or seed (a negative seed too), a file that changed after it was hashed, an output directory with foreign entries, `--profile full` (no `configs/tiny/full.yaml`), `--device auto` (typer's own: there is no such option) | **gate** R12 (foreign entry, `--profile full`, `--device auto`); the rest by the unit tests of the command (**source-read** here) |
| 3 | a fixture missing or different from its pin, a missing or invalid `uv.lock`, the manifest of another stage in the output directory | **gate** R6 (one flipped byte of a fixture: exit 3 in under 5 s, nothing written) and R12 (the foreign manifest, untouched) |
| 1 | the output cannot be written, or the report does not read back as itself (a defect: nothing is written) | unit tests only; a real full disk is **UNMEASURED** |
| 130, 143, 129 | SIGINT, SIGTERM, SIGHUP; no manifest is written and the old one was removed first | **gate** R11, real signals: exit 0.5 to 1.0 s after the signal, 0.7 s after a SIGHUP at the spawn of the vocabulary build, nothing left alive |

## Reading the files

**`report.json`** (UTF-8, sorted keys, two-space indent, one final LF; 26,545 bytes in the real run, **gate**; the golden is 26,309): `schema` `rinchen.tiny/1`, `profile`, `qe_fake`, `verdict`, `inputs` (`config_sha256`, the seven `fixtures` digests, `lock_sha256`), `stages` (fifteen entries, each `id`, `state`, `reason`, `detail` and
`parts`: `parts` is a key of the entry, not of `detail`; a stage that never ran is `skipped` for its closed reason and so is every part), `thresholds` (eight: `id`, `bound`, `value`, `verdict`, `detail`), `blocks` (ELEVEN: `filter`, `synth`, `converter`, `train`, `parity`, `serve`, `robustness` with D12's six values AND its `coverage` beside them (D12 requires both),
`memorised`, `heldout`, `neural_seam`, `int8`; each null when the stage that makes it never gave its figures) and `unmeasured` (a fixed list of six statements). It holds NO timestamp, NO duration, NO absolute path, NO environment value, NO secret, NO line of another program's output and NO `manifest_sha256` or `hypotheses_sha256`
(they embed paths), so two runs in two directories give equal bytes. The length-filter note is `detail["token_ratio_check"]` of the `filter` stage, an `ok` part with a note (there is no fourth state). `parse_report` is strict: it reads only the canonical bytes and checks that every verdict follows from its value and bound.

**`report.md`** is rendered from the parsed `report.json` alone. The text rules are closed: blocks separated by one blank line, line feeds only; a float with two decimals, a null as a hyphen; every cell escaped (a line break becomes a space; a backslash, a pipe and a backtick are escaped) so nothing breaks a table; a row per stage and, named `filter.length`,
a row per part; a fifth `detail` column in the thresholds table and a sentence saying the wall-time figure is in `manifest.json`; the held-out table "reported and never gated"; the neural line (the metric, the system, "planted scores", or the closed reason it was not measured); D12's six values beside its reference floors ("references and not gates of this run"); the fixed "Not measured" list.
Below is the GOLDEN `tests/fixtures/tiny/golden_report.md` (3,227 bytes, **measured here**), built by an independent script from PLANTED results and never from a run: its held-out figures are illustrative (three directions are the spike's F11 values, the chrF++ and the neural scores are made up). The real `report.md` is 3,228 bytes (**gate**).

```
# Tiny pipeline: pass (profile tiny)

## Stages

| stage | state | reason |
| --- | --- | --- |
| environment | ok |  |
| fetch | ok |  |
| corpus | ok |  |
| filter | ok |  |
| filter.script_ratio | ok |  |
| filter.numbers | ok |  |
| filter.length | ok | token_ratio_check skipped: the token-ratio check needs a SentencePiece model, which does not exist yet |
| filter.lid | skipped | no language-identification model: GlotLID is a download |
| filter.minhash | ok |  |
| filter.labse | ok |  |
| filter.qe | ok |  |
| spm | ok |  |
| converter | ok |  |
| synth | ok |  |
| prepare | ok |  |
| train | ok |  |
| average | ok |  |
| export | ok |  |
| parity | ok |  |
| serve | ok |  |
| robustness | ok |  |
| evaluate | ok |  |
| evaluate.memorised | ok |  |
| evaluate.heldout | ok |  |
| evaluate.neural_seam | ok |  |
| evaluate.report | ok |  |

## Thresholds

| threshold | bound | value | verdict | detail |
| --- | --- | --- | --- | --- |
| stages | <= 0 | 0 | pass | no stage failed and every skip has a reason of the closed list |
| train_chrf | >= 50.0 | 99.36 | pass | least of 5 directions, mn_Mong-eng_Latn, int8 export |
| parity_exact | >= 0.95 | 1.00 | pass | exact_share of the parity figures |
| parity_delta | <= 0.001 | 0.00 | pass | mean_abs_delta of the parity figures |
| int8_chrf_drop | <= 1.0 | 0.00 | pass | largest float32 minus int8 chrF over 5 directions, eng_Latn-khk_Cyrl |
| script_purity | >= 1.0 | 1.00 | pass | script_purity of the robustness figures |
| placeholder_acc | >= 1.0 | 1.00 | pass | placeholder_acc of the robustness figures |
| wall_time | <= 1200 | - | pass | seconds from the first line of the run |

The wall-time figure is in manifest.json: this report holds no duration and no timestamp.

## Held-out

Held-out rows: generalisation on templated sentences, reported and never gated.

| direction | system | rows | chrF | chrF++ | BLEU | statuses |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn-khk_Cyrl | int8 | 30 | 72.00 | 71.02 | 54.55 | ok 30 |
| khk_Cyrl-eng_Latn | int8 | 30 | 64.19 | 63.37 | 48.13 | ok 30 |
| khk_Cyrl-rus_Cyrl | int8 | 30 | 55.31 | 54.62 | 43.66 | ok 30 |
| rus_Cyrl-khk_Cyrl | int8 | 30 | 60.80 | 59.95 | 46.78 | ok 30 |

Neural score (metricx24_qe of tiny-int8-qe, planted scores): eng_Latn-khk_Cyrl 3.88 over 30 rows, khk_Cyrl-eng_Latn 4.31 over 30 rows.

## Robustness

The six values of the robustness measurement beside the reference floors of that measurement. The floors are references and not gates of this run.

| measure | value | reference floor | scored | total | excluded |
| --- | --- | --- | --- | --- | --- |
| entity_acc | 0.98 | 0.90 | 240 | 240 | 0 |
| mong_source_retention | 0.99 | 0.95 | 48 | 48 | 0 |
| numbers_acc | 1.00 | 0.90 | 240 | 240 | 0 |
| placeholder_acc | 1.00 | 0.99 | 238 | 238 | 0 |
| script_purity | 1.00 | 0.999 | 600 | 600 | 0 |
| term_acc | 1.00 | 0.99 | 1134 | 1134 | 0 |

## Not measured

- any run on a GPU: no stage touched a device
- a real corpus: the fixtures are templated sentences
- a real generator behind the synthetic data stage
- neural metric scorers: only the seam of score files ran, with planted scores
- GitHub Actions: nothing ran there
- the time on a 4-core CPU runner
```

**OPEN finding (not fixed by D15's build).** In the REAL `report.md` the row `parity_delta` reads `0.00` beside the bound `<= 0.001`, for a measured 1.62e-06 (**gate**): the two-decimal rule of the Markdown hides the figure that the threshold is about (the golden above shows the same `0.00`). `report.json` holds the true value (`1.6221680949143363e-06`). The combined review and its fix wave will address it
(a rule that shows enough digits needs both goldens regenerated by the independent script); until then read `parity_delta` in `report.json`.

**`manifest.json`** (stage `tiny_run`; 11,513 to 11,564 bytes, **gate**; it holds seconds and a creation time, so it differs from run to run): `inputs` (13: the pipeline file, the seven fixtures, the Eole base config, the serve config, the words file, the profile file with the seed and `uv.lock`, each hashed BEFORE it was read), `outputs` (7: `report.json` and `report.md`, then
the checkpoint files that exist: the SentencePiece model, the converter's `model.bin`, the averaged weights `model.00.safetensors` and the `model.bin` of each export; a run that failed lists fewer), `config_sha256` (the pipeline file), `seed`, `code_version`, `tool_versions` (every name of the lock with its locked version, a name locked at several versions joined by `, ` in sorted order, plus
`python`: 187 keys for the 186 names; the key is `python (tool)` if the lock ever holds a package named `python`), `metrics` (`seconds.<stage>` for all fifteen, 0.0 for a stage that did not run; `rows.kept`; `train.steps` when the stage gave it) and `status` (each stage's state). It is written after both documents, so a run that is killed has none.

## The fixtures

Seven UTF-8, LF, tab-separated files with a header line, made by `scripts/build_parallel_fixtures.py OUT_DIR` (self-contained: it embeds its nouns, entities and templates, imports only the standard library and `rinchen.script_conv.rules.transliterate`, asserts Python 3.14; every language stream is seeded from the text `7-<name>`, so the output does not depend on the hash seed or on the order of
the files) and committed byte for byte under `tests/fixtures/parallel/` (67,104 bytes in all, **measured here**; each file far under 2 MB). Production code reads them by the paths the profile names, never by import. To regenerate: `uv run --frozen --no-sync python scripts/build_parallel_fixtures.py tests/fixtures/parallel`; **a changed byte is a re-measurement of the thresholds, not an update**: the profile pins the SHA-256 of every file
(`configs/tiny/tiny.yaml`, `fixtures.sha256`), the pipeline refuses a different byte with exit 3 and nothing written, and `tests/unit/test_parallel_fixtures.py` pins the digest of the whole set (SHA-256 over the sorted `name digest` lines: `fd0ec4d812df84cc...c88e73`, **measured here**: the same value) and the composition.

| File | Bytes | Rows | What it tests |
|---|---:|---:|---|
| `eng_Latn-khk_Cyrl.tsv` | 25,476 | 345 | `train` 300, `dev` 15, `test` 30; kinds `base` 150/8/15, `int` 30/2/3, `decimal` 18/1/2, `grouped` 12/1/2, `placeholder` 60/2/5 (the ten indices 0 to 9), `entity` 30/1/3: the model's data, the validation lines of `train ablate` and of `export parity`, the held-out report |
| `rus_Cyrl-khk_Cyrl.tsv` | 26,436 | 345 | the same composition for Russian |
| `mn_Mong.tsv` | 8,347 | 70 | `train` 50, `dev` 10, `test` 10: the traditional-script spelling (by `transliterate`) of Khalkha sentences the English pair file holds in the same split, so the model has a fifth direction, `mn_Mong-eng_Latn`, for D12's `mong` section; kinds `base` 68 and `placeholder` 2, no entity rows |
| `mono_khk_Cyrl.tsv` | 4,231 | 64 | Khalkha lines with their English oracle: the synthetic job's source and the fake server's answers |
| `noise.tsv` | 819 | 10 | planted bad pairs; `expect` names the reason the filter must give: `script_tgt` 2, `len_ratio_char` 2, `numbers` 2, `duplicate` 2, `labse` 2; they enter the corpus with the split `holdout`, so they are scored and counted and never trained on whatever the filter finds |
| `nouns.tsv` | 962 | 37 | the nouns in three languages: the tables the sentences are made from, the concepts of the planted encoder and the terms of D12's glossary rows |
| `entities.tsv` | 833 | 20 | ten persons and ten cities with one to two spellings: D12's entity rows |

Rows of `train` are the memorisation gate (the master plan's "training-set chrF" and "placeholder_acc on fixture"); `dev` rows are the validation lines; `test` rows are the held-out report, never gated and never used to choose anything (the pipeline selects nothing). Khalkha is Cyrillic everywhere; no sentence is in two splits; none is in two files except the `mn_Mong` rows (same split, copied by design) and the noise rows, four of which
reuse sentences of train rows to make misalignments from authentic text (so those four are `holdout` rows whose sentences also occur as train sentences elsewhere). The `mn_Mong` column holds, besides Mongolian letters, the variation selectors and punctuation the transliteration writes (U+1802, U+1803, U+180B, U+180E), 9 `?` and 2 placeholders. The reader (`rinchen.tiny.fixtures.read_fixtures`) hashes every file before it parses
any, compares to the pins, parses the bytes it hashed and raises a `DataError` that names the file, the row and the field and quotes no text. The sentences are templated: nothing here is a real text and nothing is Inner Mongolian, Latin-written Mongolian or FLORES+.

## The thresholds

Eight, in `configs/tiny/tiny.yaml` (`thresholds:`, `budget_seconds`) and in `report.json` (`id`, `bound`, `value`, `verdict`, `detail`). A bound compares with `>=` or `<=` exactly as written, so a value equal to its bound passes; a value that is missing or cannot be a measurement is a `fail`, never a pass. The master plan's list becomes eight rows: every stage `ok` or skipped with a reason (`stages`), the training-set chrF, the parity as two rows, the int8 chrF drop that holds the product to its reference, script purity, `placeholder_acc` and the wall time; the spec's text said seven (A13).
Measured by the gate of Task 9 (**gate**; the spike's figures are beside them):

| Id | Bound | Value read from | Measured (**gate**) | Spike |
|---|---|---|---|---|
| `stages` | `<= 0` (stages at fault) | the stage records: no stage `fail`, every skip with a reason of the closed list | 0 | all `ok` or `skipped` |
| `train_chrf` | `>= 50.0` | the LEAST corpus chrF of the int8 export over the five memorised directions at beam 1 (`metrics.chrf` of each `result.json`) | 99.3585 (the least direction, `mn_Mong-eng_Latn`) | 99.36 |
| `parity_exact` | `>= 0.95` | `parity.json` of the float32 export: `exact_share` | 1.0 (70 of 70) | 1.0 (70 of 70) |
| `parity_delta` | `<= 0.001` | the same file: `mean_abs_delta` | 1.6221680949143363e-06 | 1.4e-06 to 1.6e-06 |
| `int8_chrf_drop` | `<= 1.0` | the largest (float32 chrF minus int8 chrF) over the memorised directions | 0.0 | at most 0.0 |
| `script_purity` | `>= 1.0` | D12's robustness block of the int8 export | 1.0 | 1.0 |
| `placeholder_acc` | `>= 1.0` | the same block | 1.0 (238 of 238) | 1.0 |
| `wall_time` | `<= 1200` | the monotonic clock, from the first line of the run | pass (the manifest says 112.16 s) | 120 to 158 s |

**`wall_time` in detail.** Its value is ALWAYS null in `report.json` (the report holds no duration: the manifest has the seconds; the screen shows the elapsed seconds) and its verdict comes from the elapsed time against the budget (equality passes). It PASSES on a budget cut: each command gets at most what is left of the budget, so the run ends inside it and the stage whose command timed out carries the failure
(**gate** R10: `--budget-seconds 5` gave exit 4 after 5.4 s, the `converter` stage `fail` for `timed out after 1 s`, the later stages skipped, `wall_time` `pass`). A stage that starts with no budget left is `fail` for `the wall-time budget is spent`.

Reported and never gated (each with D12's floor beside it in `report.md`, as a reference): `numbers_acc`, `entity_acc`, `term_acc`, `mong_source_retention`, the held-out chrF, chrF++ and BLEU per direction (generalisation on templated text), the agreement of the int8 and float32 hypotheses, the filter counts, the neural seam. The int8 export is NOT held to `export parity`: its exact-share rule is about held-out lines, where a tiny model is not confident
(64 of 68, exit 4, spike F9); it is held to `int8_chrf_drop` and to the agreement of its hypotheses with float32's.

### What to do when a threshold is missed

**Raise `train.steps`; never lower a bound.** The master plan's own rule is "if missed, increase the steps in the profile, never lower the threshold", and the 400 steps of the master plan IS the miss: least chrF 16.83 (**gate** R7, equal to the spike's) against 50.0. First read what the run says; most of the time the cause is not the steps.

1. Read the printed stage lines and `report.md`: which stage is `fail`, and its closed reason (below). A failing stage ends the run and every threshold then reads a missing value.
2. `train_chrf` or `placeholder_acc` low with every stage `ok`: look at `blocks.filter` (did the stand-ins reject authentic rows? `authentic_flagged` and `kept`), `blocks.train` (`rows_by_direction`, the steps), `blocks.memorised` (which direction) and `blocks.robustness.coverage` (the denominators). Only then raise the steps: in `configs/tiny/tiny.yaml` (`train.steps`) AND
   in `configs/eole/tiny.yaml` (`train_steps` and `valid_steps`, which state the same count: the profile's steps are applied as overrides of the base and a test pins the two equal, with the literal 1,500), in one commit, with the literals of `tests/unit/test_config_tiny_shipped.py`. The spike's curve (**spike**, F7, beam 4, four directions): 500 steps is the first count above 50 in every direction (56.17),
   600 above 87, 700 above 95, 800 at least 97.65, 1,000 at least 98.80; the robustness block is exact at beam 1 from 1,000 steps in 15 of 15 exports (three seeds). The choice of 1,500 is 1.5 times the count at which it is exact.
3. `parity_exact` or `parity_delta`: read `09_parity/float32/parity.json` (`mismatches`, `placement`, `quantization`). A model that learned little still passes the parity (two models on the same weights agree however little they learned: spike F17, 70 of 70 exact at 400 steps), so a miss here is the export, the tag placement or the checkpoint, not the steps.
4. `int8_chrf_drop`: compare `blocks.int8` and the two `runs/<suite>/<system>/` directories of `11_evaluate`; a damaged export is refused by `verify_export` (exit 2) before it can score.
5. `script_purity`: the converter and the mask; an output in the traditional script without a converter is refused by the engine (exit 2) even with the mask off (**gate** R9(b)), so a purity miss is a wrong script in a measured hypothesis.
6. `wall_time`: the manifest's `seconds.<stage>`; the converter (41.24 s) and the training (44.16 s) are 76 % of the run (ARITHMETIC from the **gate** seconds). A bound that is too tight for a slower runner is a budget decision in the profile (`budget_seconds`), made with a measurement of that runner.

### The controls (the gate shows each bound can fail)

| Control (**gate**) | Result |
|---|---|
| R7: `--config` with `train.steps: 400` | exit 4 after 102.2 s; `train_chrf` `fail` at 16.83 (the least of five directions), `placeholder_acc` `fail` at 0.0420, `script_purity` `pass` at 1.0 (an undertrained model still writes the right script: the purity threshold needs a control of its own); every stage `ok`; report and manifest written |
| R8: one flipped byte of an export | `rinchen eval robustness --model` and `rinchen eval run --model` exit 2 and write nothing |
| R9: placeholders of the indices 10 and up | `placeholder_acc` 0.0 (the model copies a placeholder only for the indices it trained on) |
| R9: a hypotheses file of copied inputs | `script_purity` 0.5 (600 of 600 answers copied, the half whose source is not in the target's script); with the converter attached the traditional-script output suite is 1.0 |
| R9: the checkpoint of another model | `rinchen export parity` exits 2 (the parity stage fails) |
| R9: a parity whose mean abs delta is 2e-3 | `parity_delta` `fail` |
| R9: hypotheses rotated by 7 rows | `int8_chrf_drop` `fail`: chrF 21.09 against 99.96 |
| R10: `--budget-seconds 5` | exit 4 after 5.4 s, see `wall_time` above; no process of the run left |

## The stand-ins and `--qe-fake`

`ConceptEncoder` stands in for LaBSE: a text is a bag of CONCEPTS (the nouns, the entities with their traditional-script spellings, digit strings with their separators ignored and the numbered placeholders), each concept one fixed pseudo-random 64-dimensional unit vector seeded by the CRC-32 of its name, summed in the order of the names and normalised; no weight is trained and nothing is random at run time or depends on the hash seed.
`ConceptQE` stands in for the MetricX scorer: `25 * max(0, 1 - cosine)`, lower is better. They give the filter something that can tell a pair that says one thing from a pair that says two, and **they rank nothing**. Their property tests hold that every authentic row passes and every planted row is caught. **A correction of the design text (A5):** "one concept replaced in a bag of at most three gives a cosine below 0.7" is true for ONE concept only
(exhaustive over the 57 lexicon concepts, maximum cosine 0.446, **measured** at Task 3); for two concepts 1,954 of 175,560 replacements reach 0.7 (maximum 0.832) and for three 7,037 of 20,000 sampled (exhaustive maximum 0.897); the tests assert below 0.7 for one concept and below 0.95 for two or three, and no measured number moved.
Without `--qe-fake` the parts `labse` and `qe` of the filter are skipped by reason, 8 planted rows are detected (the two misaligned rows pass the CPU stages and are `holdout`, so they are never trained on), and the `neural_seam` part of `evaluate` is skipped for `no neural metric scorer: only the seam of score files was built`. With the flag the seam is exercised: the hypotheses of the int8 held-out English run are scored
with `ConceptQE` into a score file by D13's own writer and `eval run --hypotheses --segment-scores` takes them in; the values are planted, so the neural line of `report.md` says so. The tiny random-initialised scorers the master plan names were NOT built: COMET and MetricX enter only through this seam (B1 to B4 stay open).

## The loopback server and the sandboxes that forbid it

`rinchen.tiny.fakes.FakeChatServer` is a standard-library HTTP server bound to `127.0.0.1` on a free port (it refuses any other host), answering `POST /v1/chat/completions` with an OpenAI-style reply from an oracle table (the English line of `mono_khk_Cyrl.tsv` for the text a request asks to translate) and every other request with a 404 (400 for a bad body, 413 over 1 MiB); it lives for the length of a `with` block and is
shut down on every way out. It is the ONLY socket the pipeline opens: the connection is made by the CHILD `rinchen synth run`, outside the guard of `tests/support/no_network.py`, which refuses only `connect` and name resolution in the process that installs it (the gate says it cannot prove the absence of a socket in a child; what keeps the run off the network is the shim, the offline switches and the empty device list).
The stage writes a `synth.yaml` (one `vllm` generator with the real port, `n 1`, `temperature 0`, `retries 0`) and runs `synth run ... --no-qe`, the CLI's own way to skip the GPU scorer, which `counts.json` records as `qe_status: "skipped: --no-qe"`. A sandbox that forbids loopback sockets would make the `synth` stage fail; what it prints then was never seen (**UNMEASURED**: no such sandbox was met), and
whether the GitHub runner allows it is **UNMEASURED**.

## The closed reasons

A report never holds free text: every `skipped` or `fail` carries a reason from `rinchen.tiny.states` (`StageRecord` refuses a skip or a failure without one, and a reason on an `ok`), and a test pins every string.

| Kind | Reason |
|---|---|
| skip | `no language-identification model: GlotLID is a download` |
| skip | `LaBSE needs a model that was not downloaded: --qe-fake runs the planted stand-in` |
| skip | `the QE scorer needs a model that was not downloaded: --qe-fake runs the planted stand-in` |
| skip | `no neural metric scorer: only the seam of score files was built` |
| skip | `the token-ratio check needs a SentencePiece model, which does not exist yet` (the note of the `length` part, which is `ok`) |
| skip | `an earlier stage failed` (the wording says "stage" even for a part of `evaluate` that follows a failed part: A9) |
| fail | `exit code <n>` |
| fail | `timed out after <s> s` |
| fail | `the wall-time budget is spent` |
| fail | `a check of the stage failed: <check>` |
| fail | `an input is missing or differs` |
| fail | `<m> packages differ from the lock, <e> are not in it` (counts only) |

The closed check ids (`[a-z][a-z0-9_]{0,31}`; nine, each defined in the module that uses it, since `states.py` holds only the template and the reader accepts a filled template through `is_closed_reason`): `planted_noise` (a planted row missed its reason, or too many authentic rows were flagged), `stage_skipped` (the filter reported a stage it was asked to run as skipped), `command_not_started` (a program could not be started), `stage_raised` (a stage raised; the class and frames are in
`logs/<stage>-raised.log`), `service_probe` (`/healthz` or `/readyz` or the engine failed upstream), `service_translate`, `service_refusal`, `service_metrics` and `neural_scores`.

**If a stage fails: where to look** (**source-read**). Every command's output is in `<stage dir>/logs/<stage>-<step>.log`. `environment`: a package differs from the lock (`uv sync --frozen --group dev`; the pin of the lock is exact by design, so a package added to the venv fails here). `fetch` or `corpus`: `an input is missing or differs`, a fixture or the registry (exit 3 would have stopped the run before this).
`filter`: `blocks.filter` and `03_filter/counts.json` (`planted` found against expected). `spm`: the vocabulary size against the text (the ceiling was 786 to 850 and moves with the text: 640 is a prior). `converter`: `05_converter/logs`; the five lines of `convert-script eval` are read by shape. `synth`: the loopback listener and `06_synth/logs`. `prepare`: a trained direction needs dev rows in the corpus
(`train ablate` refuses one without). `train` and `average`: `07_train/ablate/variants/<variant>/run` and its `logs`; a NaN loss does not stop Eole (D8) and shows as a missed `train_chrf`. `export` and `parity`: the manifest and record of each export. `serve`: the check id says which of the five calls failed. `robustness` and `evaluate`: the part that failed makes the parts after it `skipped` and the facts of the parts before it are still reported.

## Costs

**Measured by the gate** (recipe alone, 2026-11-12): **113.1 s** (load average 1.2 to 4.1), **121.0 s** beside another pipeline; the manifest's seconds in the table above (total 112.2 s). The runs of the determinism gate took 143.1 s and 128.2 s with three pipelines running at once. A first look at a quiet machine is faster still; these are the figures of a machine shared with other tenants.
**Spike** (the prototype, 1,500 steps, 2 threads per Eole run, load average 7 to 22): 123.4 s, 134.0 s and 158.4 s; **120.0 s with every child under `taskset -c 0-3`** (4 of the 20 cores; the in-process stages unrestricted, 0.7 s in all). The converter (43 s) and the training (48 s) are 74 % of it; the 25 CLI calls cost about 20 s of start-up in all (0.8 to 1.0 s each, spike F1). **ARITHMETIC** (labelled; the 4-core CPU is **UNMEASURED**):
a runner three times slower than this machine's single thread (D14's factor, itself UNMEASURED) takes 3 x 123 s = 370 s, 6.2 minutes; the budget (20 minutes, 1,200 s) is 3.2 times that and about 9.7 times the measured time; the job's 45 minutes cover the budget in whole minutes (20), ten minutes of set-up and a margin. Disk: the run directory was 36 MB in 208 files (**spike**). The gate's own cost: the full local gate ran in six parallel parts, the longest 1,888 s, about 37 minutes wall at a load average of 14 to 29 (**gate**, **ledger**).

## The tiny-to-full mapping

Scale-up is a change of configuration and no change of code: everything that differs between a tiny and a larger run is in the profile read by the closed loader `rinchen.config.tiny`, the code has no branch on the name of a profile (an AST scan of `rinchen.tiny` and `cli/tiny*.py` finds none) and the gate's control profile (`--config` with `train.steps: 400`) runs the same code and misses the bound.
`configs/tiny/full.yaml` is not shipped: nothing measured could fill it, and a table of unmeasured priors that a command would run is how a 14-day GPU job starts by mistake. Each full file below is loaded by the tests of its own milestone (**source-read**; the design's "one test loads each of them" was NOT built as a single test, A27).

| Stage | Tiny (this profile) | The full command and its configuration |
|---|---|---|
| `environment` | the lock against the installed packages, in process | the same (`uv sync --frozen --group dev`); no configuration |
| `fetch` | `configs/data/sources.tiny_parallel.yaml` (seven local files) | `rinchen data fetch` with `configs/data/sources.yaml` (no real entry by design: D1) |
| `corpus` | `rinchen.tiny.corpus` (library, no command) | no command exists for corpus construction from real sources; D4's corpus Parquet contract is the target |
| `filter` | `run_pipeline` with the planted stand-ins and `filter.*` of the profile | `rinchen filter run` with `configs/data/filters.yaml` and the real GlotLID, LaBSE and MetricX-QE (D4) |
| `spm` | `spm.vocab_size` 640 | `rinchen spm train --config configs/spm/student_64k.yaml` (or `teacher_128k.yaml`) |
| `converter` | `convert-script train --profile tiny` | `rinchen convert-script train --profile full --confirm-full` (D5) |
| `synth` | the loopback fake server, `--no-qe` | `rinchen synth run JOB --config configs/data/synth.yaml` against a real server (D6) |
| `prepare` | `rinchen train ablate --prepare-only`, one variant | `rinchen train ablate` with `configs/eole/multilingual_proxy.yaml` and `ablation_priors.yaml` (D7) |
| `train` | `configs/eole/tiny.yaml`, 1,500 steps | `rinchen train launch --config` of `configs/eole/teacher.yaml` (D8), `teacher_doc.yaml` (D9) or `student_server.yaml` and `student_edge.yaml` (D10) |
| `average` | `rinchen train average --last 2` | the same command |
| `export` | `rinchen export convert` at `float32` and `int8` | the same command at the quantisations of the model (the GPU-side ones were never run) |
| `parity` | `rinchen export parity` on float32 | the same command |
| `serve` | `TestClient` over `configs/serve/tiny.yaml` | `rinchen serve` with `configs/serve/server.yaml` or `edge.yaml` (D11) |
| `robustness` | a suite from the fixtures | `rinchen eval robustness` over suites from real data (D12) |
| `evaluate` | fixture suites, `eval run`, `eval report` | `rinchen eval suite fetch|build`, `eval run`, `eval report` with `configs/eval/suites.yaml` (D13) |

## Determinism

**Equal across two runs in two directories under another `PYTHONHASHSEED`, locale and time zone** (**gate** R4, `PYTHONHASHSEED=3`, `LC_ALL=C`, `TZ=Asia/Ulaanbaatar`, byte-compared): `report.json` (26,545 bytes) and `report.md` (3,228 bytes), and the digests of the four model files the pipeline makes: the averaged weights (`4e06ce45faef7c4b...`), the float32 `model.bin` (`f48811a45287337c...`), the int8
`model.bin` (`8fa90a266950cfc3...`) and the converter's `model.bin` (`e70a4ccacf85c060...`, also the design spike's value). The digests are RECORDED by the gate and never asserted against a literal: another machine, another thread count or another build of the libraries may move them, and the claim is about two runs on this machine at 2 threads (**UNMEASURED** elsewhere).
**Different, for a reason each:** the SentencePiece model (`66f3f736...` against `58af3a54...`: the file embeds the paths it was trained from, as D12 found), the manifest (its seconds and its creation time) and every file that embeds a digest of a manifest or of the SentencePiece file; the report embeds none of them. The `created` stamp of every corpus row is the profile's fixed `2026-11-11T00:00:00Z`: `corpus_frame` stamps the
current time unless it is given one, which would make `kept.parquet` differ between two runs (spike F12).

## Interrupts and the runner

The commands of the stages run through `rinchen.tiny.graceful_runner.GracefulRunner`, not through D14's `SubprocessRunner` itself. **Found by the first real run of the gate (R11):** `rinchen tiny run` interrupted by SIGTERM left `eole train` and its two multiprocessing helpers alive for 30 s and more, and a SIGHUP at the spawn of `eole build_vocab` left the build running 5 to 15 s (4 of 4 probes). The cause: D14's runner
kills the process group of an interrupted command with SIGKILL at once, and `rinchen train launch` cannot stop the Eole it started, which leads a session of its own (`rinchen.utils.logged_process`). `GracefulRunner` asks the group to end with SIGTERM, gives the leader 2.0 s and only then kills the group, so the launcher stops what it started. It was fixed in D15's own code only (commit `3862635`, with `tests/unit/test_tiny_graceful_runner.py`, 9 tests).
After the fix the signal is answered in 0.5 to 1.0 s with nothing left alive (**gate** R11). **Limits:** (a) it duplicates about 20 lines of `SubprocessRunner.run` and imports two private names of D14 (`_stop_group`, `_wait_for_exit` of `rinchen.maintain.commands`); (b) a SIGKILL of `tiny run` itself still leaves its children (D14 documents it for its own command);
(c) D14's maintenance reruns use the same SIGKILL-on-interrupt runner, and whether any of them starts a child in its own session is **UNMEASURED**; the one-line alternative lives at the root, in D14's interrupt path (ask in the interrupt path too), and changes D14's behaviour, so it stays a decision for the review and the owner.

## The workflows

**`ci.yml`, the job `tiny`** (**source-read**; a static test, `tests/unit/test_ci_workflow.py`, 201 tests with 121 broken copies of the workflow, in three support files `ci_workflow`, `ci_workflow_edits` and `ci_workflow_breaks`): `ubuntu-24.04`, `permissions: contents: read`, `timeout-minutes: 45`, a comment saying the job is separate so that a failing gate does not hide its report and the other way round; steps `actions/checkout@v7`,
`astral-sh/setup-uv@v10.3.3` ("Install uv"), `actions/setup-python@v7` 3.14, `extractions/setup-just@v4`, "Sync dependencies" (`uv sync --frozen --group dev`), "Tiny pipeline" (`just tiny`), "Upload the tiny report" (`if: always()`, `actions/upload-artifact@v7`, name `tiny-report`, ONLY `artifacts/tiny/report.json` and `artifacts/tiny/report.md`, `if-no-files-found: ignore`: the rest of the run directory holds paths of the machine and is never uploaded), and
"Show the report on the run page" (`if: ${{ always() && hashFiles('artifacts/tiny/report.md') != '' }}`, `cat artifacts/tiny/report.md >> "$GITHUB_STEP_SUMMARY"`). The stub step left the `gate` job; no secret; no cache step of ours. The least `timeout-minutes` is computed by a test from the profile: `ceil(budget_seconds / 60) + 10 = 30`, against 45: a budget over 2,100 s fails that test until the job's 45 is raised.
The recipe is `uv run --frozen --no-sync rinchen tiny run --profile tiny` (`--frozen --no-sync` for D14's reason: a plain `uv run` locks and syncs first). `.gitignore` gains `/artifacts/`.

**`nightly.yml`**: the step is renamed `Tiny pipeline` (the literal of the broken-copy case `no tiny pipeline step` in `tests/unit/test_nightly_workflow.py` was edited to match: the one forced edit of a D14 test) and still runs `just tiny` AFTER the maintenance check; **the nightly now runs the real pipeline**, and a failing check skips it as before. The job's 150 minutes cover the maintenance minimum (104) plus the pipeline's
20-minute budget: 104 + 20 = 124 <= 150 (**source-read**, a test). The nightly job uploads nothing, so the pipeline's report vanishes with the runner; only the printed lines survive in the step log.

**UNMEASURED on GitHub:** that the `tiny` job starts and finishes inside 45 minutes (ARITHMETIC: 370 s at the D14 factor plus set-up), the runner's real speed and core count, that loopback sockets work there, that `hashFiles` works in a step condition, the artifact upload and the minutes budget. The commands of the workflow ran natively (the gate runs the recipe body, read from the `justfile`, through `sh` in a temporary directory with the D14
`uv` shim first on `PATH`, because `just` is not installed on this machine).

## The STOP gate

The master plan's gate is "CI runs `just tiny` green on 3.14". It is met ONLY as: the body of the `tiny` recipe, read from the `justfile` and run natively through the real `rinchen` binary and the real Eole, CTranslate2, SentencePiece and sacrebleu on the CPU with every device hidden, no network and no token, exits 0 inside the budget with every threshold `pass`; GitHub Actions is **UNMEASURED**. Four modules
(`tests/integration/test_tiny_gate.py`, `test_tiny_gate_controls.py`, `test_tiny_gate_determinism.py`, `test_tiny_gate_hygiene.py`: 224 tests, **gate**), their helpers (`tests/support/tiny_gate{,_processes,_runs,_controls,_hygiene}.py`: five modules instead of the plan's one, for the 900-line limit) and four unit-test files; every process starts through ONE function that carries the guards (a session of its own, the environment it was given,
`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `UV_OFFLINE=1`, no `*_TOKEN`, a refusal before the start) and bounds every wait; every floor is a literal of `tests/support/tiny_gate.py` and is never lowered.

| Gate | What it asserts | Measured (**gate**) |
|---|---|---|
| R1 the recipe | exit 0 inside 1,200 s; schema `rinchen.tiny/1`, verdict `pass`, the fifteen ids in order, every stage `ok` (skipped parts: `filter.lid` and the token-ratio note, nothing else); the eight thresholds `pass`; the shim logged one `run` call and no refusal | 113.1 s alone, 121.0 s beside another pipeline; the thresholds as above |
| R2 documents | `report.md` equals `render(parse_report(report.json))`; no absolute path, date or time, duration key, environment value, secret by its shape or fixture text in either | 26,545 and 3,228 bytes |
| R3 manifest | stage `tiny_run`, written after both documents, 13 inputs with fresh digests, `tool_versions` with every lock name and `python`, `seconds.<stage>` for all fifteen, the checkpoint files with fresh digests | 13 inputs, 187 `tool_versions` keys, 7 outputs |
| R4 determinism | see above | `report.json` and the four model digests equal; the SentencePiece digests differ |
| R5 the filter | the counts above; `filter.lid` skipped for its reason; no `holdout` row in `train.parquet` | 770, 756, 10 of 10, 4 of 760 |
| R6 the fixtures | one flipped byte: exit 3, nothing written | under 5 s |
| R7 learned nothing | see the controls | exit 4 after 102.2 s |
| R8 a damaged export | exit 2 from `eval robustness --model` and `eval run --model` | as asserted |
| R9 other thresholds | each fails when its figure does, over the real commands | see the controls |
| R10 the budget | exit 4 inside 30 s (the plan said 60), `wall_time` `pass` | 5.4 s |
| R11 interrupts and the rerun | SIGTERM 143, SIGINT 130, SIGHUP 129 once `train launch` has run for 3 s; the old manifest and documents gone; nothing alive; the same directory then runs to the end and writes the recipe's `report.json` | 0.5 to 1.0 s; the rerun 148 s |
| R12 refusals | a foreign manifest (exit 3), a foreign entry (exit 2), `--profile full` (exit 2), `--device auto` (exit 2) | as asserted |
| R13 the service | statuses `ok`, counter 1.0, the refusal 400 in the report's `serve` block | as asserted |
| R14 hygiene | AST scans of `rinchen.tiny`, `rinchen.config.tiny`, `cli/tiny*.py` and the gate; the content scan of every D15 file | no finding |

**Deviations of the gate from its plan** (A25): R9(b): a traditional-script output asked of a model with no converter is REFUSED by the engine (exit 2, nothing written) even with `--script-mask off`, so the control tests the refusal, the same request passing with the converter attached, and a copied-source hypotheses file scoring `script_purity` 0.5; R10's bound is 30 s because the converter alone takes 40 s;
R11 sends the signal after the trainer ran 3 s (plus a SIGHUP at the spawn of the vocabulary build) and asserts the exit latency and that the stale manifest and documents are gone; five support modules instead of one. **Mutation** (**gate**, Task 9): 18 gate-level mutants, 17 killed, 1 equivalent survivor (the runner signalling only the leader instead of its group gives the same process table, because every stage command leads a group with no other member); 30 unit-level mutants, all killed.
The gate's first R11 control passed by luck (an orphan finished inside its window) and failed once in the full gate, which is how the defect above was found.

## The definition of done

`just accept` stays a stub; the master plan's eight items of `<mission>` are checked by hand and recorded here and in `STATE.md`.

| `<mission>` item | Met by | State |
|---|---|---|
| 1. Every module of the layout exists, typed, tested; `pytest -m "not gpu and not net"` green | D0 to D15 and their gates (the project's marker is `network`): 24,758 passed, 1 skipped, 10 deselected (**gate**) | met locally; no CI run since D2 |
| 2. Coverage of `src/rinchen` at least 85 % | nothing: `ci.yml` uploads `coverage.xml`, no number was ever recorded | **UNMEASURED** |
| 3. `ruff check`, `ruff format --check`, `mypy` green; `ty check` runs | `just gate` (966 files formatted, `mypy src` 319 files, **ledger**); `ty` was never run | gate met locally; `mypy --strict` and `ty` **UNMEASURED** |
| 4. CI green on GitHub Actions with 3.14 | `ci.yml` (D0 and D2 green on 2026-11-02) and the new `tiny` job | **UNMEASURED** for D3 to D15 |
| 5. `just tiny` at most 20 minutes on a 4-core CPU with `report.json` meeting the thresholds | this milestone: 113.1 s here; 370 s ARITHMETIC at D14's factor | met here (20 cores); the 4-core time **UNMEASURED** |
| 6. A runbook per operator stage | `docs/runbooks/00` to `15` | met |
| 7. Scale-up is configuration only | P14: no profile branch, a closed profile schema, a control profile, the mapping table | met as stated; a full run **UNMEASURED** |
| 8. `rinchen doctor` passes in CI, GPU checks documented | `just doctor` in `ci.yml`; D14's CPU list | CI run **UNMEASURED** since D2 |

## Open gates

Not done by D15, because they need what does not exist here: everything GPU (the GPU-side Eole, `float16`, `int8_float16`, speed and memory of a real model); GitHub Actions (the `tiny` job inside 45 minutes, the runner's speed and cores, loopback sockets, `hashFiles` in a step condition, the upload, the minutes budget); the 4-core time (120 s under `taskset` on a 20-core machine, 370 s ARITHMETIC);
a real corpus, a real generator behind `synth`, GlotLID, LaBSE and QE scorers and neural metric scorers (nothing ran; the stand-ins rank nothing); cross-machine determinism (byte-equal on one machine at 2 threads); a richer fixture that reaches the master plan's "1k" vocabulary (the ceiling was 786 to 850 and moves with the text; a fixture with 74 nouns would reach 1,000 at the price of more steps: not measured);
stage resume and a partial-run option (not built); `just accept`; and the gates of D3 to D14 that need real data, a GPU, a reader or the Hub (stay open).

## What was not verified

Any run on a GPU or on GitHub Actions; a real corpus, model, generator, LID or scorer; `rinchen eval suite fetch` and `rinchen release hub --upload` (never run); a sandbox that forbids loopback sockets; exit code 1 with a real full disk; the 4-core time; another machine's digests; a full profile; Windows and macOS; and the figures of the real-binary gate were not read again for this runbook (the full gate that closed Task 10 ran its modules again on 2026-11-12, pass or fail only; the figures here are the gate's of Task 9).
**No reader has read the code of D15 yet.**

## Known limits, left open on purpose

1. **Three inputs are hashed for the manifest only** (the Eole base config, the serve config, the converter's words file) and are read by the commands that use them later: a change in between is not seen. The registry `configs/data/sources.tiny_parallel.yaml` is not hashed at all (the fetch stage checks each copy against its pin).
2. **`config_sha256` is the digest of the pipeline file**, not of the effective configuration: `--threads` is recorded nowhere, `--budget-seconds` and `--qe-fake` only through the report's `qe_fake` and the `wall_time` bound.
3. **`wall_time` is `pass` when the elapsed time equals the budget exactly** while a stage can fail for the spent budget (the stage failure still fails the verdict); its value is null in the report.
4. **`parity_delta` is shown as `0.00` in `report.md`** (OPEN, above).
5. **A malformed fact becomes a null block.** `report_blocks` turns malformed facts into a null block through a broad `except` on purpose (a failed run must still write its report), so a stage `ok` with malformed facts shows a null block while the verdict can still pass; the review decides.
6. **`GracefulRunner`** has the three limits above.
7. **`spm check`'s coverage lines are not read**; its verdict is its exit code.
8. **The pin of 186 locked packages** is a literal in four tests: a lock update that adds or removes a package fails them on purpose, and a package added to the venv fails the `environment` stage by design.
9. **The step count is stated twice** (`train.steps` and the base config's `train_steps` and `valid_steps`) and a test pins them equal.
10. **A top-level `logs/` directory** is written only by a raising stage; the per-stage `logs/` directories are the usual place.
11. **The service stage is in process**, so no timeout can stop it (the budget is read before it); its logging is routed away by swapping `rinchen.utils.logging.get` for the length of the stage (a `logger=` parameter of `create_app` would be cleaner but edits serve code).
12. **The held-out figures** (55 to 72 chrF on 30 sentences per direction, **spike** F11 and the golden) are generalisation on templated text, reported and never gated; the toy ranks nothing.
13. **Files near the size limits**: none over 450 lines in `src` (the largest, `report.py`, has 399) and none over 900 in the tests (the largest, `tests/unit/test_cli_tiny.py`, has 877; `test_tiny_gate_controls.py` 857) (**measured here**, `wc -l`).

## The review

The combined review of the D15 code has NOT run yet. When it has, its findings, a fix wave and the limits it leaves open are recorded by a follow-up commit, as D13's and D14's were; the open `parity_delta` display, the null-block decision and the runner decision above are known before it.

## Measured numbers

The machine: a 20-core CPU shared with other tenants, CPython 3.14.10, uv 0.13.7, Eole 0.6.4, CTranslate2 4.8.4, sentencepiece 0.2.1, sacrebleu 2.6.0; the GPU was hidden from every process.

| What | Value |
|---|---|
| the recipe, alone / beside another pipeline | 113.1 s / 121.0 s (**gate**); spike 123.4, 134.0, 158.4 s, 120.0 s under `taskset -c 0-3` |
| 4-core time | 370 s (ARITHMETIC, 3 x 123 s; **UNMEASURED**) |
| manifest seconds | environment 0.05, fetch 0.00, corpus 0.03, filter 0.43, spm 1.83, converter 41.24, synth 1.05, prepare 0.97, train 44.16, average 0.92, export 6.37, parity 3.52, serve 0.06, robustness 1.28, evaluate 10.27 (**gate**; sum 112.18, ARITHMETIC) |
| thresholds | stages 0; train_chrf 99.3585; parity_exact 1.0 (70/70); parity_delta 1.6221680949143363e-06; int8_chrf_drop 0.0; script_purity 1.0; placeholder_acc 1.0 (238/238); wall_time pass (**gate**) |
| the filter | 770 rows, 756 kept, 4 of 760 authentic flagged, 10 of 10 planted matched; `train.parquet` 1,316 rows (1,246 train, 70 dev) (**gate**) |
| the controls | 400 steps: 16.83 and 0.0420, exit 4 after 102.2 s; budget 5 s: exit 4 after 5.4 s; indices 10 and up 0.0; copied answers purity 0.5; rotated hypotheses chrF 21.09 against 99.96 (**gate**) |
| signals | 143, 130, 129 within 0.5 to 1.0 s, nothing alive; the rerun 148 s (**gate**) |
| documents | `report.json` 26,545 B, `report.md` 3,228 B, `manifest.json` 11,513 to 11,564 B; goldens 26,309 B and 3,227 B (**gate**, **measured here**) |
| the manifest | 13 inputs, 187 `tool_versions` keys, 7 outputs (**gate**) |
| determinism | `report.json` and `report.md` byte-equal; four model digests equal; the SentencePiece models differ (**gate**) |
| the fixtures | 7 files, 67,104 bytes, set digest `fd0ec4d812df84cc...` (**measured here**) |
| the code | 32 source files and 7,868 added source lines, 60 new test modules (54 unit, 2 property, 4 integration), 27 support modules, 21 fixture files, 36,839 added test lines (`git diff --numstat 412bf84..HEAD`, **measured here**, 4,308 new tests: ARITHMETIC from 24,759 collected against 20,451) |
| the tests | 24,758 passed, 1 skipped, 10 deselected after Task 9 (20,611, 21,908, 22,050, 22,792, 23,257, 23,870, 24,139, 24,340 after Tasks 1 to 8; against 20,450 passed at the start, ARITHMETIC from 20,443 and the 7 tests of D14's last commit) (**ledger**); 24,759 passed, 1 skipped, 10 deselected after the records of Task 10, in six parts with at most three pytest processes at once (one more test than Task 9's: the hygiene gate's content scan has a case for each D15 document and this runbook is new) |

## Where the code lives

`src/rinchen/config/tiny.py`; `configs/tiny/tiny.yaml`, `configs/eole/tiny.yaml`, `configs/data/sources.tiny_parallel.yaml` (the profile files `configs/profiles/tiny.yaml` and `configs/serve/tiny.yaml` are earlier ones); `src/rinchen/tiny/{states,context,fixtures,corpus,fakes,data_stages,commands,command_readers,command_writers,step_runner,model_stages,model_stages_rest,suites,suite_matching,serve_stage,eval_stages,eval_readers,evaluate_facts,thresholds,report,report_blocks,report_markdown,report_schema,report_shapes,pipeline,dry_run,graceful_runner}.py`;
`src/rinchen/eval/fixture_suites.py` (the one module D15 added to D13's package: write-only); `src/rinchen/cli/{tiny,tiny_inputs,tiny_output}.py`; `scripts/build_parallel_fixtures.py`; `tests/fixtures/{parallel,tiny,tiny_run}/`; the `tiny` recipe of the `justfile`; `.github/workflows/ci.yml` (job `tiny`) and `nightly.yml` (the step name); `.gitignore` (`/artifacts/`); the gate `tests/integration/test_tiny_gate{,_controls,_determinism,_hygiene}.py`;
the unit tests `tests/unit/test_{tiny_*,cli_tiny*,config_tiny*,parallel_fixtures,eval_fixture_suites,ci_workflow}.py` (54 files) and `tests/property/test_tiny_*.py`; and the helpers `tests/support/{tiny_run,tiny_run_eval,tiny_children,tiny_cli,tiny_config_documents,tiny_fakes_reference,tiny_help,tiny_oracle,tiny_pipeline_runs,tiny_profiles,tiny_report_build,tiny_report_inputs,tiny_report_random,tiny_report_reference,tiny_report_scans,tiny_report_walk,tiny_scans,tiny_stage_inputs,tiny_threshold_inputs,tiny_gate,tiny_gate_processes,tiny_gate_runs,tiny_gate_controls,tiny_gate_hygiene,ci_workflow,ci_workflow_edits,ci_workflow_breaks}.py`.
