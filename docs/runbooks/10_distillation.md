# 10 - Distillation and students

## Scope

D10 builds the path from a teacher to a student: `rinchen distill export` (turn a checkpoint of OURS into a
CTranslate2 directory with a record of where it came from), `rinchen distill generate` (the teacher translates the
source side of the authentic training corpus into a sequence-level knowledge-distillation, KD, corpus, with a
provenance manifest), `rinchen distill prepare` (mix the KD corpus with the original data by an exact integer ratio
and lay out the data and the config of a student of another shape, with its own SentencePiece model),
`rinchen distill evaluate` (chrF and BLEU of any checkpoint of ours on the dev rows of the corpus files), the three
student configs (`configs/eole/student_{server,edge,tiny}.yaml`) and the tiny student that is the STOP gate. Training
is D8's `rinchen train launch`, unchanged. The design, with the rulings T1 to T16 and a section of the amendments made
while building, is `docs/superpowers/specs/2026-11-09-d10-distillation-design.md`; the decisions are condensed in the
2026-11-09 entries "D10 distillation" and "D10 review" of `DECISIONS.md`; the upstream facts are in `docs/VERIFY.md`
(section "D10 findings"). The CTranslate2 and Eole facts come from a spike made before the design
(`.superpowers/sdd/2026-11-09-d10-distillation/spike-findings.md`; Eole 0.6.4, CTranslate2 4.8.4, CPU, toy data).

**What D10 is.** Tested machinery. The teacher is a checkpoint our launcher wrote and our manifest vouches for (the
final step of a run `rinchen train launch` finished, or a directory `rinchen train average` wrote). Its export is a
directory of five files and a manifest. It translates the training sources through CTranslate2 behind D6's `Translator`
seam, D6's shard store and selection keep the best hypothesis of each source, and the winners become rows of origin
`ft` in the project's corpus schema. A student of 2 encoder and 1 decoder layers then trains from random
initialisation on that text mixed 1:1 with the originals, through the real launcher; its dev chrF is within the toy's
noise of the teacher's, and its own export decodes exactly what Eole decodes. The same code takes the shapes of the real
students (16/3 and 12/2 layers); nothing of them was ever trained.

**What D10 is NOT.** Nothing below is a statement about the quality of any translation.

- **No GPU was used.** This machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run. Every Eole
  process and every CTranslate2 translator of D10 (the spike, the tests, this runbook's measurements) ran on the CPU
  with `CUDA_VISIBLE_DEVICES=""`, `gpu_ranks: []` and `device="cpu"`; the converter runs with the device hidden
  (`D-CT2-EOLE-CUDA`). `--device cuda` and `--compute-type` exist for the operator and are passed through unmeasured.
  Everything GPU-side (CTranslate2 on a device, float16 and int8, the export of a 1.1 B-parameter teacher, the real
  students' training) is unmeasured.
- **No real data, no real teacher.** The tiny teacher and the tiny student train on the templated toy sentences of
  `tests/support/tiny_parallel.py` (37 nouns in a few frames), so every number below tests machinery and ranks nothing:
  three seeds of the same student span 1.9 chrF. No real corpus exists. `configs/eole/student_server.yaml` and
  `student_edge.yaml` are tables of unmeasured priors that Eole's schema accepts, and nothing has trained them.
- **No quantisation, no `rinchen.json`, no inference engine, no serving, no `rinchen translate`, no strict
  CTranslate2/Eole parity (log-probabilities), no script masking**: all D11. D10 builds only the export its gate
  needs, shaped so that D11 takes it over (the converter call is `rinchen.export.eole_ct2`, the record is
  `export.json`, `verify_export` is the reader). D11 builds `export/ct2.py::convert(eole_ckpt, out, quantization)`
  around `export_ct2` (`int8`, `int8_float16`), `rinchen.json` (languages, tags, version) in place of `export.json`,
  the engine, the parity and the quantisation gates.
- **No fine-tuning of anything.** A student starts from random initialisation, always: `train_from` and `reset_optim`
  are not keys of any config of this project and `distill prepare` renders neither (and refuses a base that holds
  them). The teacher of a distillation is a checkpoint OUR launcher wrote and OUR manifest vouches for; an external
  model is never a teacher.
- **No document-level teacher, no n-best as extra targets, no monolingual KD sources, no `<ft>` at inference, no QE
  scorer run** (the real scorer needs a GPU): open questions below.
- **No FLORES+ and no test row.** The evaluation reads the corpus's own `dev` rows; no D10 code reads a `test` row
  (the sources of a KD run are authentic TRAIN rows, chosen by `load_training_rows`).

Tags used below: **measured** (run for this runbook on 2026-11-09, code `cc917e2`, or by the STOP gate), **spike** (run
in the design spike, on the CPU), **source-read, unmeasured** (read from the source of Eole or CTranslate2, never
run), **ARITHMETIC** (a formula applied to the values of a config), **UNMEASURED** (nothing was run). Every number is
from a test, the code, the spike, the ledger of the build
(`.superpowers/sdd/2026-11-09-d10-distillation/progress.md`) or a measurement made for this runbook ("Measured numbers");
a number that was not obtained is not given.

What was checked: the unit and property tests (every new test mutation-checked by hand during the build: 63, 88, 95,
109, 87 and 65 mutants for Tasks 1 to 6 and the gate's ten classes), `tests/integration/test_distill_tiny_student.py`
(the STOP gate, real `rinchen`, real Eole, real CTranslate2, CPU, 44 tests), a combined review of Tasks 1 to 7
(`e5ff644..e67b05e`: ready with fixes, no Critical finding, two Important and nine groups of Minor findings), its fix
wave (`D10: fix ...` x 4 and `D10: review minors`, `e67b05e..cc917e2`), and the real binary for this runbook. A scoped
re-review of the fix wave (`e67b05e..cc917e2`) was dispatched at the same time as this runbook was written; its verdict
is NOT recorded here.

## Architecture

| Module | Job |
|---|---|
| `rinchen/export/eole_ct2.py` (new package `export`) | `export_environment`, `export_log_path`, `export_ct2(checkpoint, out_dir, *, timeout, log_path)`: the converter call, MOVED from `script_conv/export.py` (which re-exports the three names and keeps `parity`) |
| `rinchen/train/distill_export.py` | `ExportRecord`, `CheckpointRecord`, `VerifiedExport`, `export_checkpoint`, `verify_export`: the provenance checks, the staging directory, the record |
| `rinchen/train/distill_saved_config.py` | the tag placement read back from the checkpoint's saved `config.json` and the refusal of a statement that disagrees |
| `rinchen/train/distill_export_files.py` | `verified_bytes`, `shared_vocabulary`, `end_token`: the files of an export read only as the bytes that were hashed |
| `rinchen/train/distill_teacher.py` | `TeacherTranslator`, `TeacherOptions`, `ct2_source_tokens`, `COUNTER_KEYS`: CTranslate2 behind D6's `Translator` |
| `rinchen/train/distill_sources.py` | `KdSources`, `kd_sources`: one source per direction and text, from the authentic training rows |
| `rinchen/train/distill_shards.py` | `shard_candidates`: a shard read back from disk or asked of the teacher and written before the next |
| `rinchen/train/distill_generate.py` | `generate_kd`, `kd_fingerprint`, `kd_name`, `KdResult`: the loop that joins D6's shards and selection into KD rows and counts |
| `rinchen/train/distill_corpus.py` | `load_kd_rows`, `KdRows`, `KdStats`, the KD exclusion reasons: reading the KD files a generation wrote |
| `rinchen/train/distill_mix.py` | `Mix`, `parse_mix`, `mixed_weights`, `repeat_factors`: the exact integer ratio |
| `rinchen/train/distill_prepare.py` | `prepare_student`, `PreparedStudent`, `render_student_config`: the student's data and config |
| `rinchen/cli/distill.py` and `distill_{export,generate,generate_plan,generate_records,prepare,evaluate}.py` | the four commands, their manifests and exit codes |
| edits to D5 to D9 | `synth/tags.py` (one entry: generator kind `teacher` gives origin `ft`), `train/eole_data.py` (`kd_tag` keyword, public `ft_tag()`), `train/mixture.py` (the constant `KD_GROUP = "kd"`), `train/eole_cfg.py` (`reject_dollar` and `reject_glob_characters` take an optional `reason`), `script_conv/export.py` (imports), `cli/main.py` (registers `distill`) |

Layering as D7 to D9: `train` never imports `script_conv` or `cli` (a test checks); `export` imports nothing of `train`,
`synth`, `cli` or `script_conv` (the converter is a subprocess, `ctranslate2` is imported lazily and only where a model
is loaded); `infer/` is untouched (a test pins its file set).

## The pipeline, in order

The tiny flow, run once for this runbook through the real `rinchen` binary on the CPU with the GPU hidden (**measured**,
2026-11-09). `$W` stands for the workspace of the run (the gate's helpers build the same one under a pytest temporary
directory). The teacher is the averaged tiny teacher of D8's recipe at 2,000 steps (`configs/eole/teacher_tiny.yaml`,
source tag, trained through `rinchen train ablate --prepare-only` and `rinchen train launch`, averaged with `rinchen
train average --last 2`); everything below starts from that directory.

**1. Export the teacher** (`distill export`, 5.1 s here, 3.4 s in the gate; the converter log is `<O>.export.log`):

```
rinchen distill export --checkpoint $W/teacher_out/variants/t0.5_bt0.5_src_s1234/run/averaged \
  --spm-model $W/in/spm/toy.model --tag-placement src --out-dir $W/teacher_ct2
exported to $W/teacher_ct2
record: $W/teacher_ct2/export.json
manifest: $W/teacher_ct2/manifest.json
```

**2. Plan the generation** (`distill generate --dry-run`, 1.5 s: builds no teacher, loads no model, writes nothing, not
even the output directory):

```
rinchen distill generate --corpus $W/in/toy.parquet --teacher $W/teacher_ct2 --beam-size 4 --n 1 --seed 1234 \
  --out-dir $W/kd --dry-run
corpus files: 1
teacher: rinchen-b7f33d88d5fa (tag placement src)
direction eng_Latn-khk_Cyrl: sources 1200, shards 3
direction khk_Cyrl-eng_Latn: sources 1200, shards 3
direction khk_Cyrl-rus_Cyrl: sources 600, shards 2
direction rus_Cyrl-khk_Cyrl: sources 600, shards 2
sources: 3600
shards: 10 (reusable 0)
decoding: beam size 4, n 1, max decoding length 256, max source pieces 256, batch size 64, device cpu, compute type default, threads 1
qe: not requested
fingerprint: 5c6b165b29067fbadd2a8e3ecf5d1fc3837284ea908207f884cd69cf4f8dea76
shard directory: $W/kd/shards-5c6b165b2906
nothing was written
```

**3. Generate** (the same command without `--dry-run`: 2.8 s here, 1.6 s in the gate; the shard progress goes to
standard error):

```
counts eng_Latn-khk_Cyrl: sources 1200, candidates 1200, dropped_empty 0, dropped_purity 0, dropped_qe 0, dropped_copy 0, no_candidates 0, no_survivor 0, selected 1200
counts khk_Cyrl-eng_Latn: sources 1200, candidates 1200, dropped_empty 0, dropped_purity 0, dropped_qe 0, dropped_copy 0, no_candidates 0, no_survivor 0, selected 1200
counts khk_Cyrl-rus_Cyrl: sources 600, candidates 600, dropped_empty 0, dropped_purity 1, dropped_qe 0, dropped_copy 0, no_candidates 0, no_survivor 1, selected 599
counts rus_Cyrl-khk_Cyrl: sources 600, candidates 600, dropped_empty 0, dropped_purity 0, dropped_qe 0, dropped_copy 0, no_candidates 0, no_survivor 0, selected 600
counts total: sources 3600, candidates 3600, dropped_empty 0, dropped_purity 1, dropped_qe 0, dropped_copy 0, no_candidates 0, no_survivor 1, selected 3599
matches reference: 3583 of 3599
letterless sources: 0 (not sent to the teacher)
teacher counters (this process only): sources 3600, source_too_long 0, empty_source 0, truncated 0, line_break 0, reserved_symbol 0
shards: planned 10, reused 0, generated 10, regenerated_invalid 0
qe: skipped: not requested
kd corpus: $W/kd/kd.parquet
manifest: $W/kd/manifest.json
```

(The first lines are the plan of step 2, printed again.) The same command into a copy of the output directory reuses
the ten shards: `shards: planned 10, reused 10, generated 0, regenerated_invalid 0` and the teacher counters are all 0
(`sources 0`), because the counters cover what THIS process translated. `reserved_symbol 0` is the counter of the
reserved-piece rule ("The six counters"), 0 here as it was for 800 real hypotheses of the spike's teacher.

**4. Plan and prepare the student** (`distill prepare --dry-run`, then without it: 1.3 to 1.9 s; the tiny student's base
config, `--mix 1:1`):

```
rinchen distill prepare --corpus $W/in/toy.parquet --kd $W/kd/kd.parquet --spm-model $W/in/spm/toy.model \
  --config configs/eole/student_tiny.yaml --mix 1:1 --bt-ratio 0.5 --temperature 0.5 --tag-placement src \
  --dev-max-lines 100 --out-dir $W/student
corpus files: 1
kd files: 1
rows read: 5200
rows kept: 4800
excluded not_train: 400
...
dev in train: 0
kd rows read: 3599
kd rows kept: 3599
kd excluded not_train: 0
kd excluded filtered: 0
kd excluded wrong_origin: 0
kd excluded empty: 0
kd excluded line_break: 0
kd excluded unknown_direction: 0
kd duplicate ids: 0
kd rows eng_Latn-khk_Cyrl: 1200
kd rows khk_Cyrl-eng_Latn: 1200
kd rows khk_Cyrl-rus_Cyrl: 599
kd rows rus_Cyrl-khk_Cyrl: 600
mix: 1:1
kd tag: ft
corpus eng_Latn-khk_Cyrl.auth: rows 1200, weight 22
corpus eng_Latn-khk_Cyrl.bt: rows 300, weight 6
corpus khk_Cyrl-eng_Latn.auth: rows 1200, weight 22
corpus khk_Cyrl-eng_Latn.bt: rows 300, weight 6
corpus khk_Cyrl-rus_Cyrl.auth: rows 600, weight 15
corpus khk_Cyrl-rus_Cyrl.bt: rows 300, weight 7
corpus rus_Cyrl-khk_Cyrl.auth: rows 600, weight 15
corpus rus_Cyrl-khk_Cyrl.bt: rows 300, weight 7
corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 28
corpus khk_Cyrl-eng_Latn.kd: rows 1200, weight 28
corpus khk_Cyrl-rus_Cyrl.kd: rows 599, weight 22
corpus rus_Cyrl-khk_Cyrl.kd: rows 600, weight 22
repeat eng_Latn-khk_Cyrl.kd: 1.250
repeat khk_Cyrl-eng_Latn.kd: 1.250
repeat khk_Cyrl-rus_Cyrl.kd: 1.503
repeat rus_Cyrl-khk_Cyrl.kd: 1.500
corpora: 12 (reused 0)
validation lines: 400
config: $W/student/config.yaml
manifest: $W/student/manifest.json
```

(`...` stands for the lines of D7's statistics that the dry run and the real run print alike: `excluded filtered`,
`excluded excluded_origin`, `excluded empty`, `excluded line_break`, the three `excluded by origin` lines, `duplicate ids`
and the four `dev lines` lines.) A dry run prints the same up to `repeat` and then `nothing was written`.

**5. Train the student** (D8's launcher, unchanged: 64.9 s here, 59 to 77 s in the gate; 2,000 steps, `xent` 5.58 at step
50 and 0.95 at 2,000 in the gate):

```
rinchen train launch --config $W/student/config.yaml --run-dir $W/student/run --threads 2 --timeout 600
status: completed
steps: 2000
resumed from: none
attempt: 1
checkpoint: $W/student/run/ckpt/step_2000
```

**6. Evaluate teacher and student on the same dev lines** (`distill evaluate`, 4.1 to 4.4 s each; the teacher's line set):

```
rinchen distill evaluate --checkpoint $W/teacher_out/variants/t0.5_bt0.5_src_s1234/run/averaged \
  --spm-model $W/in/spm/toy.model --corpus $W/in/toy.parquet --tag-placement src --out-dir $W/eval_teacher --max-lines 100
direction eng_Latn-khk_Cyrl: chrF 98.51 BLEU 98.52 (100 lines)
direction khk_Cyrl-eng_Latn: chrF 98.29 BLEU 98.01 (100 lines)
direction khk_Cyrl-rus_Cyrl: chrF 97.79 BLEU 98.68 (100 lines)
direction rus_Cyrl-khk_Cyrl: chrF 98.20 BLEU 98.32 (100 lines)
chrF mean: 98.20
chrF min: 97.79
tag leaks: 0
dev lines in training: 0
result: $W/eval_teacher/result.json
manifest: $W/eval_teacher/manifest.json
```

The student's final step gave 98.82, 99.30, 97.54 and 98.67 per direction (BLEU 98.85, 98.64, 97.76 and 98.67), mean
**98.58**, min 97.54, no tag leak, no dev line in training; the student's `ckpt` link directory gave the same numbers.

**7. Export the student** (3.8 s here, 3.2 s in the gate): the same command with `--checkpoint
$W/student/run/ckpt/step_2000`. Its `model.bin` is 1,835,438 bytes against the teacher's 2,034,311 (the encoder is as
deep, the decoder has one layer less), the other four files as the teacher's. CTranslate2 greedy against `eole predict`
greedy on 200 dev sentences: **200 of 200 equal** in pieces and in text; "This is the star." gives "Энэ бол од.".

**8. A second student with its own SentencePiece model** (500 pieces against the teacher's 640): `distill prepare` 0.95 s
(`src_vocab_size` 501, the validation corpus cut with that model), a 200-step launch 21.9 s.

Every command exits 0; the refusals exit 2, 3, 4 or 1 ("Refusals").

## The teacher export

### What an export is

`rinchen distill export` turns the final `step_<N>` directory of a finished `rinchen train launch` run, or a directory
`rinchen train average` wrote, into a directory CTranslate2 loads. The converter is CTranslate2's own
(`python -I -W ignore -m ctranslate2.converters.eole_ct2`, run through `rinchen.export.eole_ct2.export_ct2` with the GPU
hidden) and writes `model.bin`, `shared_vocabulary.json` and `config.json`; D10 adds `spm.model`, `export.json` and, last,
`manifest.json`. The tiny teacher's export (**measured**; the sizes match the spike's and the gate's):

| File | Bytes | What it is |
|---|---:|---|
| `model.bin` | 2,034,311 | the weights (SHA-256 `b7f33d88d5fa...`, bit-for-bit the spike's and the gate's) |
| `shared_vocabulary.json` | 8,659 | a list of 344 strings: the model's whole padded vocabulary in Eole's order |
| `config.json` | 223 | the converter's: `eos_token`, `bos_token`, `unk_token`, `add_source_bos/eos false`; no layer count, no tag, no tokenizer |
| `spm.model` | 12,612 | a copy of the SentencePiece model the checkpoint was trained with |
| `export.json` | 710 | the record |
| `manifest.json` | 3,360 | stage `distill_export`, written last, with every file's SHA-256 |

(`spm.model` was 12,605 bytes in the gate's run: the toy SentencePiece model is trained afresh in each workspace and the
byte count moved by 7 between two workspaces with the same sentences; the cause was not looked into. `model.bin`, the
vocabulary, `config.json` and every score agree between the runs.)

`shared_vocabulary.json` holds the vocabulary Eole trained with (`<unk>`, `<blank>`, `<s>`, `</s>`, the tokens the data
showed by frequency, the 118 reserved symbols that D9's invariant appended, and made-up padding tokens such as
`averyunlikelytoken0` up to a multiple of 8); its ORDER depends on the training data, so a student's file differs from a
teacher's even with the same entries (`D-CT2-EOLE-VOCAB`).

The record, `export.json` (sorted keys, two-space indent, one final LF; schema `rinchen.distill_export/1`):

```
{
  "checkpoint": {
    "config_sha256": "7f1d51cd...", "kind": "averaged", "manifest_sha256": "087d8444...",
    "manifest_stage": "train_average", "step": null, "vocab_sha256": "9d444b84...", "weights_sha256": "f4e32704..."
  },
  "decoder_layers": 2, "encoder_layers": 2, "hidden_size": 64,
  "schema": "rinchen.distill_export/1", "spm_sha256": "4b676e88...", "tag_placement": "src", "vocabulary_entries": 344
}
```

(one key per line in the file). `kind` is `step` (with its step number) or `averaged` (step null); the layer counts and
`hidden_size` are read from the checkpoint's saved `config.json`, `vocabulary_entries` from its `vocab.json`. The reader
refuses a record of another shape: another schema, a missing or extra key, a repeated JSON key, a count that is not a
positive integer, a digest that is not 64 hex characters, a kind that disagrees with its step.

The manifest's status is `export: ok`, `tag_placement` and `checkpoint_kind`; its metrics `encoder_layers`,
`decoder_layers`, `hidden_size`, `vocabulary_entries` and `model_bytes`; its inputs are the checkpoint's weights,
`config.json` and `vocab.json` and the SentencePiece model, each with the SHA-256 the verification took before it read
the file (the command never hashes them again); its outputs are the five files with the SHA-256 they have now. Its seed
is -1.

### What is checked, in the order of the code

1. The arguments: the checkpoint, `--out-dir` and `--spm-model` must not hold `$`, and the checkpoint and `--out-dir` must
   not hold a glob character (`[ ] * ?`; the checkpoint because Eole globs its directory, `D-EOLE-GLOB-CKPT`, the output
   directory by the command's own stricter rule); `--spm-model` must be a file; `--tag-placement` is `src` or `tgt`.
2. **`continuation.verify_source`** (D9): the directory is a final step of a launch or an averaged directory, a
   manifest of ours (`train_launch` or `train_average`) lists its files, and the SHA-256 of weights, `config.json`,
   `vocab.json` and the manifest are those on disk. A non-final step, a flipped weights byte, a missing manifest and a
   run copied without its manifest are refused. The refusal text says "cannot continue from ..." (a wording debt of D9's
   function, left as it is).
3. **The saved `config.json` and `vocab.json`** are read once, only as the bytes that were hashed, and must say what an
   export records (positive integers at `model.encoder.layers`, `model.decoder.layers` and `model.encoder.hidden_size`, a
   non-empty list of strings at the `src` key of `vocab.json`). A saved config that names ONE SentencePiece model for the
   source and ANOTHER for the target is refused: an export holds one.
4. **The tag placement** ("The tag placement and its check").
5. **The SentencePiece model** `--spm-model`: its SHA-256 must equal that of the copy Eole saved into the checkpoint
   directory (the file the saved `config.json` names under `${MODEL_PATH}`); another model is refused.
6. The output directory must not exist (the converter refuses one): checked first, with the arguments, again after the
   checkpoint was hashed and read, and once more right before the rename.
7. The converter runs; its three files, the vocabulary entry count it wrote (a list of strings with the number of
   entries the record states) and the two extra files are checked; the manifest is written last.

### The tag placement and its check

CTranslate2's files do not hold where the direction tag goes: in the source (`src`: D7's default, `src_prefix` is the tag)
or as a forced prefix of the target (`tgt`). A teacher exported with the wrong one still converts and translates, and
every translation is wrong (the review's end-to-end failure: a source-tag teacher exported with `--tag-placement tgt` gave
1,190 KD rows of which 1 matched an authentic target, Cyrillic garbage that passed the script rule). The saved
`config.json` of a checkpoint our launcher wrote DOES hold the placement: the `data` section keeps the `src_prefix` and
`tgt_prefix` of each corpus (`train/eole_data.py`). The rule (`train/distill_saved_config.py`): a whole-word direction tag
`<2xx>` in `src_prefix` means `src`; one in `tgt_prefix` means `tgt`; the origin markers `<bt>`, `<ft>` and `<doc>` may sit
in `src_prefix` either way and tell nothing. `--tag-placement` is still required, and a statement that disagrees is
refused before the converter starts (**measured**: the real averaged teacher, `--tag-placement tgt`, 1.3 s, exit 2,
nothing made, no `.teacher_ct2_wrong*` directory, no export log):

```
error: the tag placement 'tgt' is not the one the checkpoint was trained with: $W/teacher_out/.../run/averaged/config.json
has data.eng_Latn-khk_Cyrl.auth.src_prefix = '<2khk_Cyrl>', so the direction tag is in the source prefix, which is
placement 'src'; pass --tag-placement src. With the tag in the wrong place the model would be given an input it was never
trained on
```

(one line; wrapped here). `distill evaluate` checks its `--tag-placement` the same way before anything is written (the
same message, exit 2, 1 s, **measured**). What the check does not cover:

- A saved config WITHOUT a `data` section, or whose prefixes hold no direction tag, says nothing: the statement stands and
  is recorded. **The record does not say whether the placement was derived or stated**, and an export made before the
  check (D10's own gate exports were made with the right value) cannot be re-verified afterwards: the saved config is not
  in the export directory. A schema bump would have refused every existing record, so the record is unchanged.
- A saved config whose `data` section is not what Eole writes, or whose corpora place the tag differently (a tag in the
  source prefix of one corpus and in the target prefix of another, or on both sides of one), is a `DataError`, exit 3: no
  statement can be right.
- A target-tag teacher exported as `src` is refused the same way (**measured** by the fix wave on the spike's real
  exports: source-tag teacher + `tgt` exit 2, + `src` exit 0 in 17 s; target-tag teacher + `src`
  exit 2; the student + `tgt` exit 2).

### The staging layout, the converter log and a crash

The converter refuses an output directory that exists, so `export_checkpoint` never lets it write into `--out-dir`. It
makes a private directory `.<name>.<random>` BESIDE `--out-dir` (`tempfile.mkdtemp`, so the parents of `--out-dir` are
made as the converter would have made them), lets the converter write into a not-yet-existing child `export` of it, checks
the three files and the vocabulary entry count, writes `spm.model` and `export.json` into the same child, and renames the
child onto `--out-dir` after one more existence check. Any failure removes the private directory, and `--out-dir` only if
this call put it in place: a directory somebody else makes at that path meanwhile is refused and never touched (the first
version removed whatever sat there, review M1). A `rename` onto an EMPTY directory that appears in the last instant would
still replace it, and nothing else: that window is stated in the code and left. The manifest is written after the
rename; a failure writing it removes `--out-dir`.

**A crash may leave `.<name>.<random>` behind** (a kill that Python cannot catch cannot run the cleanup). **Measured**: a
SIGKILL of `rinchen distill export` 1.2 s after the private directory appeared left `.crash_export.r8szb_ft/export/` holding
the converter's three files (the converter, a process of its own, ran on and finished: a SIGKILL of `rinchen` does not
reach it, `D-LOGGED-TERM`), `crash_export.export.log` beside it and NO `crash_export` directory. The private directory is not an
output and nothing reads it; delete it by hand. **The converter's log, `<O>.export.log`, stays beside `--out-dir`**
(`export_log_path(out_dir)`), also when the export failed: it is the place to read a converter failure (the exception's
message ends with its last lines).

### The vocabulary cross-check

The record's `vocabulary_entries` is the length of the checkpoint's `vocab.json` source vocabulary. At export time the
vocabulary the converter wrote must be a list of strings with exactly that many entries (else `UpstreamError`, exit 4, the
private directory removed, the log kept); `verify_export` repeats the comparison for a directory it is given, when
`shared_vocabulary.json` is a list of strings (a file that is anything else is refused by the reader that opens it, the
teacher). The real artifacts agree: 344 = 344 (**measured**).

### `verify_export`, the reader

Everything that trusts an export (the teacher that translates, the generation that fingerprints it) goes through
`verify_export(directory)`: the manifest must be of stage `distill_export` with status `export: ok` and list each of the
five files once by file name; the SHA-256 of every file, taken once and BEFORE it is parsed, must be the one the manifest
lists; the record must state the SentencePiece model that is beside it. A directory that was moved is accepted (the
names are compared, not the paths). **The verification is not atomic with CTranslate2's later load of `model.bin`**: it
guards against a mistake or a stale directory, not against a writer that swaps a file after the hash; the same holds for
the converter, which reads the weights after the checkpoint's digest was taken (the record's `weights_sha256` says what
was vouched for, and the converter reads the file afterwards, as every command of this project reads an input after it
hashed it).

### Refusals of the four commands (real binary, 2026-11-09, the gate's fifteen)

**Measured**: each of the fifteen exits 2 in 0.8 to 1.1 s (15.4 s in all on the loaded machine, 11 s in the gate), on a
damaged copy of what the pipeline made; every input is byte for byte as before; the output directory is missing or empty
(`export_into_an_existing_directory` keeps the file that was already in it); and neither Eole nor the converter was started
(no export log appeared beside any output). Paths are shortened to `$WORK`.

| Command | Case | The message (start) |
|---|---|---|
| `export` | a step directory without its run | `cannot continue from $WORK/.../step_2000: no launch.json within 3 levels above it, so it is not a checkpoint of a run that `rinchen train launch` wrote` |
| `export` | a flipped weights byte | `cannot continue from ...: model.00.safetensors differs from the manifest ...: its SHA-256 is 0454f23da6fb... and the manifest lists 7035575e665a...; the file changed after the manifest vouched for it` |
| `export` | a non-final step | `cannot continue from ...step_1500: the manifest ... lists the files of step_2000, not of step_1500: only the final step of a finished run is vouched for` |
| `export` | another SentencePiece model | `$WORK/ws/spm500/toy.model (SHA-256 c974dcd43c18...) is not the SentencePiece model the checkpoint was trained with (...averaged/toy.model, SHA-256 4b676e8848eb...): the checkpoint's vocabulary is the pieces that model gave its data, and a piece it lacks would be read as <unk>` |
| `export` | an existing output directory | `$WORK/.../e5 exists: remove it, or name another output directory (the converter refuses a directory that exists)` |
| `generate` | a `model.bin` with a flipped byte | `cannot use the export ...: .../model.bin differs from the manifest ...: its SHA-256 is 0da6466224e3... and the manifest lists b7f33d88d5fa...; the file changed after the manifest vouched for it` |
| `generate` | an export without its manifest | `cannot use the export ...: no manifest vouches for it: .../manifest.json does not exist (`rinchen distill export` writes it when it finishes)` |
| `generate` | a vocabulary without a tag it needs | `the direction tag <2eng_Latn> is not an entry of .../shared_vocabulary.json: CTranslate2 would turn it into <unk> and translate into the wrong language without a word` |
| `generate` | `--n 5 --beam-size 4` | `--n (5) must not exceed --beam-size (4): CTranslate2 returns at most as many hypotheses as the beam holds` |
| `prepare` | a KD file with a flipped byte | `KD file ...kd.parquet: its SHA-256 is cfd7e608... but the manifest vouches for 6ee20512...; the file changed after the generation wrote it` |
| `prepare` | a KD file without its manifest | `KD file ...kd.parquet has no manifest.json beside it: only `rinchen distill generate` writes one, and nothing else vouches for a KD file` |
| `prepare` | `--mix 0:1` | `--mix '0:1' is not K:O: two positive whole numbers of at most 10000, written with digits only and no sign or space, such as 1:1 or 3:1` |
| `prepare` | a base config with `train_from` | `invalid base Eole config ...: training.train_from: a student starts from random initialisation, never from a checkpoint; remove it` |
| `prepare` | a stated `src_vocab_size` of 100 | `invalid Eole config: src_vocab_size: 100 is below 641, what the vocabulary of the SentencePiece model ... needs (640 pieces and Eole's pad token): Eole would cut the reserved symbols that are added at the end of its vocabulary file` |
| `evaluate` | a checkpoint path with a bracket | `the checkpoint directory .../bracketed[1]/step_2000 holds '[', which Eole reads as a glob pattern when it loads a checkpoint from it` |

(The first three `export` messages are D9's `verify_source`, whose text says "cannot continue from" although the command is
an export: the wording debt of "Known limits". The placement refusals, `--tag-placement tgt` on the source-tag teacher for
`export` and for `evaluate`, are shown in "The tag placement and its check": exit 2 in 1.3 s and 1 s.)

## The teacher as a translator

### Source tokens by placement

`ct2_source_tokens(placement, tag, pieces)` is the one pure function that builds what CTranslate2 is given (the place a
D11 engine or a window format would enter):

| Placement | CTranslate2 source | `target_prefix` | the hypotheses |
|---|---|---|---|
| `src` | `[<2xx>, *pieces]` | none | the pieces of the translation, then `</s>` |
| `tgt` | `pieces` | `[[<2xx>]]` | START with the prefix token `<2xx>`, which is stripped once before decoding, then `</s>` |

No BOS or EOS is added (`add_source_bos` and `add_source_eos` are false in the export's `config.json`); the tag is an
ordinary entry of the shared vocabulary. Without a forced prefix a target-tag model starts with a tag of its own
choosing (**spike**: for three English sources, whose only target is Mongolian, the first tokens were `<2khk_Cyrl>` and two pieces of
the sentence): the forced
prefix is what makes the direction a request and not a guess. A `tgt` hypothesis that does not start with the forced tag
is an `UpstreamError`.

### What CTranslate2 does silently

All **spike** facts, measured on the CPU with CTranslate2 4.8.4 and read from its source; the guards are what the code
adds (`VERIFY.md`, "D10 findings").

| Fact | Guard |
|---|---|
| An unknown tag is not an error: the token is mapped to `<unk>` and the output is garbled (`D-CT2-UNKNOWN-TAG`) | `check_directions` and every `translate`: the tag of each target language must be an entry of `shared_vocabulary.json` AND one piece of the SentencePiece model, else `ConfigError` naming it (the gate's refusal: a vocabulary whose `<2eng_Latn>` entry was renamed) |
| `max_decoding_length` (default 256) cuts a hypothesis with no sign; with `return_end_token=True` a finished hypothesis ends with `</s>` and a cut one does not, for greedy and beam search (`D-CT2-END-TOKEN`) | always ask for the end token; a hypothesis without it is dropped and counted (`truncated`) |
| `max_input_length` (default 1,024) cuts the SOURCE with no sign: a 321-piece source with a limit of 20 gave a 5-piece output (`D-CT2-INPUT-TRUNCATION`) | a source of more than `max_source_pieces` (default 256, at most 1,023: the tag takes one place) is NOT sent and counted (`source_too_long`) |
| An empty source returns no hypothesis tokens; a bare tag returns a made-up sentence | a source with no piece is not sent and counted (`empty_source`); a source with no letter or digit (whitespace, punctuation, zero-width characters) is not sent either: it gets no candidates (`KdResult.letterless_sources`) |
| The target prefix is part of the returned hypothesis (`D-CT2-TARGET-PREFIX`) | stripped once, and its presence checked |
| `num_hypotheses <= beam_size x patience` (`D-CT2-NUMHYP`); the hypotheses are distinct and sorted by score | `n <= beam_size`, refused otherwise (the gate's refusal: `--n 5 --beam-size 4`) |
| A batch of sentences gives what each sentence alone gives: 50 of 50 identical | none needed |
| With a GPU visible the converter segfaults (`D-CT2-EOLE-CUDA`) | the converter always runs with `CUDA_VISIBLE_DEVICES=""` |

### The six counters and the reserved-piece rule

`TeacherTranslator.counters()` is what THIS translator did since it was made: `sources` (texts given),
`source_too_long` and `empty_source` (texts not sent), and three kinds of hypotheses dropped, each under the FIRST reason
that applies, in this order (`COUNTER_KEYS`: `sources`, `source_too_long`, `empty_source`, `truncated`, `line_break`,
`reserved_symbol`):

1. `truncated`: no end token at the end.
2. `reserved_symbol`: the hypothesis holds a piece no translation holds. Dropped BEFORE it is decoded, because
   SentencePiece decodes such a piece to text that passes every later rule: the unknown piece decodes to the text of
   U+2047 between spaces (decoding the pieces of "Ном", `<unk>` and "ба" gave a text with U+2047 between the two words,
   which D6's `select_candidates` kept with `dropped_purity 0` in the review's probe), and a reserved symbol or a direction tag decodes to its own letters. The set
   of pieces: the processor's unknown piece, the end token in the middle of a hypothesis, any of the 118 project symbols
   (`spm.symbols.user_defined_symbols()`: the direction tags, `<bt>`, `<ft>`, `<t>`, `</t>`, `<r>`, `</r>`, `<doc>`, `<sep>`
   and the numbered placeholders), and **any entry of `shared_vocabulary.json` that is no piece of the SentencePiece
   model**: Eole's `<blank>` and the made-up padding tokens (`averyunlikelytoken0`) that the model can emit and that decode
   to their letters (the second fix, `cc917e2`). A piece outside the vocabulary is not judged: CTranslate2 cannot emit it.
   The pieces of a byte-fallback model (`<0x41>`) are ordinary and stay.
3. `line_break`: the decoded text holds a line break (`mixture.LINE_BREAK_CHARACTERS`, U+2028 included), which would
   turn one sentence into two lines of a text file.

A failed call adds nothing. The counters reach `counts.json` (`teacher_counters`), the manifest's metrics
(`teacher.sources`, `teacher.source_too_long`, `teacher.empty_source`, `teacher.truncated`, `teacher.line_break`,
`teacher.reserved_symbol`) and the result line of `distill generate` (**measured**: `teacher counters (this process
only): sources 3600, source_too_long 0, empty_source 0, truncated 0, line_break 0, reserved_symbol 0`). On the toy
teacher every counter but `sources` is 0; the reserved-symbol rule found nothing in 800 real hypotheses of the spike's
teacher (beam 4, two per source; **measured** by the fix wave), so the rule is pinned by scripted tests with a real
SentencePiece model and has no real positive.

`max_decoding_length=3` gave `truncated` 2 and `max_source_pieces=3` gave `source_too_long` 1 in Task 2's real-artifact
check on the spike's exports (**measured** by that task, not repeated here).

### Options

`TeacherOptions(beam_size=4, n=1, max_decoding_length=256, max_source_pieces=256, batch_size=64, device="cpu",
compute_type="default", threads=1)`. Every number is an integer of at least 1 (a bool is not one); `device` is `cpu` or
`cuda` and NEVER `auto`, which would pick the GPU (`--device auto` is a usage error); `compute_type` is CTranslate2's
(`default`, `int8`, `float16`, ...) and is passed as given. CTranslate2 is built with `intra_threads=threads` and
`translate_batch(beam_size, num_hypotheses=n, max_batch_size=batch_size, max_decoding_length, return_end_token=True)`.
On the toy (one thread, a model of width 64, so no figure for anything real): 3,600 sources in about 0.7 s at beam 4 and
200 sentences in 0.026 s greedy (**spike**).

## The KD corpus

### The sources

A source is the `src` text of an AUTHENTIC training row. `distill generate` reads the corpus files through D7's
`load_training_rows`, which keeps only rows of the split `train` without a `filter_reason` and leaves out the origins `ft`,
`synth_llm` and `synth_nmt`; of those, only the group `auth` (not the back-translated rows) is used. So no dev, test or
holdout row, no filtered row, no back-translated row and no earlier forward translation is ever a source. There is one
source per direction and text, however often the corpus repeats the text and whatever its targets are, in registry
directions only, SORTED by id (`generate_kd` sorts them itself); the id is `make_row_id(src_lang, tgt_lang, text, "")`.
`--direction <src>-<tgt>` (repeatable) restricts the run to some directions. The authentic targets of a source are kept to
count how many KD targets equal one of them (`matches_reference`). The toy has 3,600 sources: 1,200 `eng_Latn-khk_Cyrl`,
1,200 `khk_Cyrl-eng_Latn`, 600 `khk_Cyrl-rus_Cyrl` and 600 `rus_Cyrl-khk_Cyrl`.

### Generation: shards, the fingerprint, resume

D6's machinery runs over the teacher unchanged (the spike found the fit): `plan_shards` cuts each direction into shards of
`--shard-size` sources (default 512), `ShardStore` writes one Parquet file of candidates per shard BEFORE the next shard is
asked for, `select_candidates(kind="ft", synthetic_lang=<tgt_lang>, ...)` keeps what survives (empty, script purity, copy,
the optional QE floor, minimum Bayes risk among several survivors; a lone survivor wins unscored) and `build_rows` makes
the rows. What D6's job does not have and D10 adds:

- **The fingerprint** names the shard directory `shards-<first 12 hex>` (all directions share it, one subdirectory each,
  `shard-00000.parquet` and so on). It is the SHA-256 over canonical JSON of: the teacher's identity (the SHA-256 of
  `model.bin`, `shared_vocabulary.json`, `config.json` and `spm.model`, and the tag placement), the decoding options that
  decide a hypothesis (`beam_size`, `n`, `max_decoding_length`, `max_source_pieces`, `device`, `compute_type`), `shard_size`,
  `seed` (it names the directory; decoding is deterministic) and, per direction in direction order, the `(id, text)` rows in
  id order. **Not in it**: the batch size and the thread count (speed only), the selection thresholds (a rerun with other
  ones reuses the shards, as D6's jobs do) and `--name`. D6's own fingerprint covers a model's name and not the weights
  behind a path, which is why D10 has its own. A change of any ingredient gives another directory (22 mutants of the
  recipe were each exposed by a test).
- **Resume.** A shard with a valid file is reused; one with a missing file is generated; a damaged one is regenerated
  (`regenerated_invalid`); `--force` ignores the shards. A translator that raises, returns the wrong number of lists or a
  non-list is an error naming the direction and the shard, and the shards written so far stay. The rerun of the toy
  reused 10 of 10 shards and wrote the same rows (**measured**). A Ctrl-C leaves the shards and no manifest (exit 130,
  verified with fakes; a real CTranslate2 call is a blocking C++ call that delivers the signal only when its batch
  returns, UNMEASURED).
- **The old manifest is removed when the translation begins** (`begin_commit`, after every preflight and before
  `generate_kd`): a run that fails or is interrupted leaves none, and the earlier `kd.parquet` stays untouched until the
  new one replaces it. (D6's synth command keeps the old manifest during generation; D10 does not.)
- **Letterless sources.** A source with no letter or digit is not sent: it gets no candidates and is counted in
  `no_candidates` and in `letterless_sources` (a subset of it). A source of only zero-width characters would otherwise be
  translated into a made-up sentence, which D7's empty rule (`str.strip`) does not catch.

### The rows

A KD row is a row of `CORPUS_SCHEMA` (`corpus.schema`), written with `write_corpus`: `origin` `ft` (the generator kind
`teacher` gives `ft` in `synth/tags.py`), `corpus` `ft:<name>` (`<name>` is `--name`, 1 to 64 characters of
`A-Z a-z 0-9 . _ -`, or `rinchen-` and the first 12 hex characters of the SHA-256 of `model.bin`: `ft:rinchen-b7f33d88d5fa`
here), `split` `train`, `src_lang`/`tgt_lang`/`src`/`tgt` the pair (the REAL source and the teacher's text),
`scores.script_tgt` the script purity of the target (1.0 on the toy), `scores.metricx_qe` or `scores.cometkiwi` when a QE
scorer ran, `doc_id`, `seg_idx` and `filter_reason` null, one `created` stamp, and `id` = `make_row_id` of the pair. One row
per source that kept a target; rows are in direction order, then source id order. **Measured** (the toy): 3,599 rows,
90,707 bytes of Parquet, schema as above, for example (`eng_Latn` to `khk_Cyrl`):

```
src: The book and the snow.            tgt: Ном ба цас.          origin: ft   corpus: ft:rinchen-b7f33d88d5fa
src: The river, the tea and the book.  tgt: Гол, цай ба ном.     split: train  scores.script_tgt: 1.0
```

### `counts.json` and the manifest

`counts.json` holds the fingerprint, the teacher's identity (the checkpoint record, the five file digests, the manifest
digest, the tag placement), `decoding`, `selection` (the thresholds that decided), `counts` by direction and in `total`
(every key of D6's `COUNT_KEYS`: `sources`, `candidates`, `dropped_empty`, `dropped_purity`, `dropped_qe`, `dropped_copy`,
`no_candidates`, `no_survivor`, `selected`), `letterless_sources`, `matches_reference`, `teacher_counters` with the note
`teacher_counters_cover`, `shard_stats` (`planned`, `reused`, `generated`, `regenerated_invalid`), `qe_status`, `name`
and `seed`. Conservation holds per direction and in total: `sources = no_candidates + no_survivor + selected`, and
`candidates` is what the teacher kept (the toy: 3,600 sources, 3,600 candidates, 1 `dropped_purity`, 1 `no_survivor`,
3,599 selected; the one dropped source is `khk_Cyrl-rus_Cyrl`).

The manifest (stage `distill_generate`, written last): inputs the corpus files, the selection config
(`configs/data/synth.yaml` by default; its SHA-256 is a listed input and its `config_sha256`), `spm.model`, `model.bin`,
`shared_vocabulary.json`, `config.json`, `export.json` and the export's manifest, each hashed before it was read (the
teacher's digests are the ones `verify_export` took); outputs `kd.parquet` (with its row count) and `counts.json`; metrics
the counts (`count.<direction>.<name>` and `count.total.<name>`), `matches_reference.*`, `letterless_sources.*`,
`shards.*`, `teacher.<counter>` and `decoding.*`; `seed` the `--seed`; status `generate: ok`, `qe: ok | skipped: ...`,
`teacher.name`, `teacher.tag_placement`, `teacher.checkpoint_weights_sha256`, `teacher.checkpoint_manifest_sha256`,
`teacher.model_sha256` and `decoding` (the options as JSON text). **The status key `teacher.name` holds the KD CORPUS name
(`rinchen-b7f33d88d5fa`), not a name of the teacher** (spec T5 names it so; recorded as a limit).

### `matches_reference`

`matches reference: 3583 of 3599` (1,200 of 1,200, 1,191 of 1,200, 597 of 599, 595 of 600): the KD targets that equal an
authentic target of their source, as an EXACT string comparison (reference edge whitespace never matches). A diagnostic,
never a filter. On the toy it is 99.6 % because the teacher is overfit to the sentences it trained on and the templated
data has one right answer; **it says how close the teacher is to the data, not that KD works**: with a real teacher the
KD rows rarely equal the references. The gate requires at least 90 % (which only a nearly-memorising toy teacher gives).

### n-best and the MBR evidence

Default `--n 1` with beam 4: the KD target of a source is the beam's top hypothesis. `--n` up to the beam keeps the `n`
hypotheses in the shards and lets D6's selection pick among the survivors by minimum Bayes risk. **Spike** (the same toy
teacher, its own 3,600 training sources, one thread):

| setting | candidates | dropped | selected | KD target == authentic target |
|---|---:|---|---:|---:|
| source-tag teacher, beam 4, n 1 | 3,600 | purity 1 | 3,599 | 3,583 of 3,599 (99.6 %) |
| the same, n 4 (MBR over the survivors) | 14,400 | empty 16, purity 871 | 3,600 | 2,749 of 3,600 (76.4 %) |
| 1,000-step TARGET-tag teacher, beam 4, n 4 | 14,400 | purity 762 | 3,600 | 3,029 of 3,600 (84.1 %) |

Ranks 2 to 4 are about 41 chrF below rank 1 (95.36, 54.52, 53.09, 53.40 on the 200-line dev set), the n=4 selection drops 871
of 14,400 candidates for script purity (the wrong-script outputs sit in the lower ranks), and **choosing by MBR instead of
taking the top hypothesis cost 834 of 3,600 sources their agreement with the reference**: the lower-ranked hypotheses of a
source are variants of one wrong sentence, so the consensus pick is a variant of the error (`D-KD-NBEST-MBR`). That is a
property of this overfit toy teacher. For a real teacher it is UNMEASURED, which is why `n=1` is the default and `n>1` an
option. Several rows per source (n-best as extra targets) are NOT built: the schema has no rank column.

### QE

`--qe` (off by default) adds D6's QE floor of `SelectionConfig.qe` on the pairs (REAL source, candidate), the orientation of
an `ft` pair. The command REFUSES (`ConfigError`, exit 2) when the scorer cannot run (`qe.skip_reason`): an operator who
asked for the filter never gets an unfiltered corpus silently. The scorer is built and vetted before the output directory
is created and released after the run. The real scorer (`EoleMetricXQEScorer`) needs a GPU and was never loaded; the tests
inject `tests.support.planted_synth.FakeScorer`; with it the spike scored 1,200 pairs in 38 calls (`QeConfig.batch_size`
32) and conservation held. The thresholds (script share 0.9, MetricX-QE floor 8.0) are D4 and D6 priors, never validated.

### Reading the KD files

`distill_corpus.load_kd_rows` reads KD files for `prepare`: each must be a corpus Parquet file with the `manifest.json` of
stage `distill_generate` beside it, status `generate: ok`, listing it with the SHA-256 and row count it has now (a missing,
foreign, unfinished or non-matching manifest is a `ConfigError` naming the file). Every row of every file is counted once,
in the kept rows or under the first reason that matches, in this order: `not_train`, `filtered`, `wrong_origin` (not `ft`),
`empty` (null, empty or whitespace only), `line_break`, `unknown_direction` (no original corpus trains its direction: a KD
corpus is never trained alone). A repeated id is kept and counted (`duplicate_ids`). The kept rows of all the files are merged
per direction into ONE corpus `<direction>.kd` (two files or two teachers of a direction share it; `rows_by_label` counts
them by corpus label), because `mixed_weights` refuses two KD corpora in one direction: its KD share would be above the
mix's.

## The mix and the student's data

### The exact ratio

`--mix K:O` (default `1:1`; two positive whole numbers of at most 10,000 each, digits only, no sign, no space, no leading
zero; reduced by their greatest common divisor, so `2:2` is `1:1` and `6:4` is `3:2`) means that in EVERY direction exactly
`K/(K+O)` of the examples Eole draws come from the KD corpus. Eole's corpus `weight` is the number of consecutive examples
taken from that corpus per round of a cyclic round robin (`D-EOLE-WEIGHT-SEMANTICS`), so a corpus's share of the draws is its
weight over the sum of the weights, whatever the corpus sizes. The original corpora (authentic and back-translated:
`build_corpora`, `--bt-ratio`) keep D7's mixing (temperature over the directions' original rows, `mixing_probabilities`,
`integer_weights`, `--weight-total`). Then, with `S` the sum of the D7 weights of a direction that has a KD corpus and `K`,
`O` the reduced mix:

- every original corpus of the direction weighs `O x` its D7 weight, so the originals of the direction weigh `O x S`;
- its KD corpus weighs `K x S`.

So the KD corpus against the originals of its direction is exactly `K : O` whatever `S` is, every direction weighs
`(K + O) x S` and the share of each direction in the draws is the temperature's, unchanged. The weights add up to `(K + O) x
weight_total`. All of it is integer arithmetic, pinned by property tests for any positive integers and any original
weights (a KD corpus is never subsampled or upsampled; a KD row is drawn `kd.repeat` times as often as an original row).

### The weights (measured, the toy, `--bt-ratio 0.5 --temperature 0.5`)

D7's weights of the original corpora are 22, 6, 22, 6, 15, 7, 15, 7 (`eng-khk` authentic and back-translated, `khk-eng`,
`khk-rus`, `rus-khk`), 100 in all. The KD weights of the four directions and the sum (`dry run`, 1.3 to 1.9 s, nothing
written):

| `--mix` | the originals weigh | KD `eng-khk` / `khk-eng` / `khk-rus` / `rus-khk` | all | `kd.repeat` `eng-khk` | `khk-rus` |
|---|---|---|---:|---:|---:|
| `1:1` | 22, 6, 22, 6, 15, 7, 15, 7 | 28, 28, 22, 22 | 200 | 1.250 | 1.503 |
| `3:1` | 22, 6, 22, 6, 15, 7, 15, 7 | 84, 84, 66, 66 | 400 | 3.750 | 4.508 |
| `3:2` | 44, 12, 44, 12, 30, 14, 30, 14 | 84, 84, 66, 66 | 500 | 1.875 | 2.254 |
| `2:1` | 22, 6, 22, 6, 15, 7, 15, 7 | 56, 56, 44, 44 | 300 | 2.500 | 3.005 |

(`3:1` gives exactly 3 times the `1:1` KD weights, which a test pins; the gate's `3:1` dry run is the one that kills a
K/O swap, which the `1:1` test alone cannot. The ratio of KD to originals in each direction: `28 : 28`, `84 : 28`,
`84 : 56`, `56 : 28`.) The `repeat` factor is `(KD weight / KD rows)` over `(the sum of the original weights of the direction
/ their rows)`: how many times more often one KD row is drawn than one original row. It is recorded as the manifest metric
`kd.repeat.<direction>` and never gated; whether a high factor (a student seeing the same teacher sentence many times per
pass) hurts is UNMEASURED.

### Partial KD coverage (the controller's ruling)

`distill generate --direction` may cover only some directions. The design said the originals would be scaled by `O` and
the KD corpus by `K x S` in the directions that have KD; the directions WITHOUT a KD corpus would then keep the weights
`O x S` too, and their share of the draws would drop below the temperature's. **The ruling (confirmed by the controller):
the originals of a direction without KD are scaled by `K + O` (the reduced numbers) instead**, so that every direction's
share of the draws stays exactly the temperature's and the weights add up to `(K + O) x` the sum of D7's. For full coverage
the result is the design's. A unit test and a property test pin it. Worked example (**measured**, `generate --direction
eng_Latn-khk_Cyrl` so that only that direction has KD, the default options `--bt-ratio 0 --temperature 1 --mix 1:1`):

```
corpus eng_Latn-khk_Cyrl.auth: rows 1200, weight 33
corpus khk_Cyrl-eng_Latn.auth: rows 1200, weight 66
corpus khk_Cyrl-rus_Cyrl.auth: rows 600, weight 34
corpus rus_Cyrl-khk_Cyrl.auth: rows 600, weight 34
corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 33
repeat eng_Latn-khk_Cyrl.kd: 1.000
```

D7 gives the four authentic corpora 33, 33, 17, 17. The covered direction keeps its originals (`O x 33`) and gets KD `K x 33`;
the other three are doubled (`K + O = 2`): 33 + 33 + 66 + 34 + 34 = 200 = `2 x 100`, and each direction's share is 33/100,
33/100, 17/100, 17/100 of the whole, as without KD. With `--bt-ratio 0.5 --temperature 0.5` the same partial run gives the
originals 22, 6 | 44, 12 | 30, 14 | 30, 14, the KD 28 and 56 + 56 + 44 + 44 = 200 = `2 x 100` (**measured**).

### The `<ft>` tag and Eole's batch deduplication

KD corpora are named `<direction>.kd` (group `kd`, `mixture.KD_GROUP`) and, by default, carry the origin tag `<ft>`
(`synth.tags.data_origin_tag("ft")`, `eole_data.ft_tag()`) in `src_prefix`: `<2xx> <ft>` for placement `src` (the rendered
config holds `"src_prefix": "<2khk_Cyrl> <ft>"` and `"tgt_prefix": ""`, **measured**), `<ft>` alone as `src_prefix` with the direction tag as
`tgt_prefix` for placement `tgt` (the D7 convention of `<bt>`). The keyword is `data_section(kd_tag=True)`; no corpus of group
`kd` exists in D7, so D7 is unchanged. `--no-kd-tag` leaves the KD rows untagged, as authentic ones: allowed, recorded
(`kd_tag: 0` in the manifest, `kd tag: none` in the output).

**Why (`D-EOLE-BATCH-DEDUP-KD`, spike, measured again for this runbook).** Eole's batch iterator silently drops, in training,
an example whose TOKENISED source (after `sentencepiece` and `prefix`) equals one already in the same minibatch. A KD row
repeats the source of an original row, always (the KD sources are the authentic sources), and its id and its place in the
id-sorted files are the original's, so the weighted mixer serves the two side by side; when the KD target also equals the
authentic target (99.6 % of the toy rows) the pair collides. Counted with Eole's own iterator in process
(`num_workers: 0`, 100 steps, the rendered student configs, `dedup.py`, the spike's `s6_dedup.py` copied):

| data | batch (tokens) | examples seen | dropped | share |
|---|---:|---:|---:|---:|
| originals only (spike) | 2,048 | 15,360 | 0 | 0 % |
| originals + KD untagged, 1:1 (spike) | 2,048 | 15,360 | 798 | 5.19 % |
| originals + KD untagged, 1:1 (spike) | 16,384 | 102,400 | 5,317 | 5.19 % |
| originals + KD tagged `<ft>`, 1:1 (spike) | 2,048 | 13,312 | 0 | 0 % |
| originals + KD tagged `<ft>`, 1:1 (spike) | 16,384 | 102,400 | 0 | 0 % |
| **this run**, tagged `<ft>`, 1:1 | 2,048 | 13,312 | 0 | 0 % |
| **this run**, tagged `<ft>`, 1:1 | 16,384 | 102,400 | 0 | 0 % |
| **this run**, untagged, 1:1 | 2,048 | 15,360 | 798 | 5.19 % |
| **this run**, untagged, 1:1 | 16,384 | 102,400 | 5,317 | 5.19 % |

The four runs of this run reproduce the spike's counts exactly (152, 228, 173 and 233 s: Eole in process on the CPU, 12
threads, on a loaded machine). (The toy corpora are as small as the buckets: `bucket_size` 1,024, 1,200 to 2,000 rows per corpus. The untagged run used the
tagged run's `vocab.src`: the deduplication compares tokenised sources before any vocabulary lookup.) The dropped rows on
the toy are duplicates of original rows, so no information was lost. With a real teacher the KD rows rarely equal the
references, so the pairs are not in lockstep and a collision needs both rows in one minibatch of a bucket of 262,144
examples (a probability of the size bucket/corpus times batch/bucket): the share on real data is UNMEASURED. A tag in the
source removes the hazard by construction. The tag costs nothing measurable: with `<ft>` in training and none at
inference the student reached chrF 98.58 and with `<ft>` at inference 98.55 (spike; the untagged 1:1 student 98.80 is
inside the toy's noise). **At inference the student is used WITHOUT `<ft>`**; whether the tag helps or hurts a real
student is open, and `--no-kd-tag` exists for that experiment.

### What `prepare` reads and writes

It reads the original corpus files (`load_training_rows`, `load_dev_rows` of D7), the KD files (`load_kd_rows`, above), the
student's SentencePiece model and the base config. It writes, under `--out-dir` (absolute paths in the files, whatever the
working directory), and only when it is not a dry run:

| Path | Content |
|---|---|
| `data/auth/...`, `data/bt/r<ratio>_s<seed>/...` | the original corpora (D7's files: `.src`, `.tgt`, `.json` per corpus) |
| `data/kd/<direction>.kd.src`, `.tgt`, `.json` | the KD corpora, text without tags (the tags are `prefix` configuration), reused when unchanged (`corpora: 12 (reused 12)` on a second run, **measured**) |
| `valid.src`, `valid.tgt` | the dev rows cut with the STUDENT's SentencePiece model, 400 lines on the toy (22,374 and 17,574 bytes) |
| `config.yaml` | the student base with `src_vocab`, `data`, `transforms_configs.sentencepiece`, `training.model_path`, `seed` and `src_vocab_size` laid over it |
| `run/vocab.src`, `run/ckpt` | the run files `rinchen train launch` makes (not written by `prepare`) |
| `manifest.json` | stage `distill_prepare`, last |

The base config must hold no corpora, no `train_from` and no `reset_optim` (`ConfigError`: a student starts from random
initialisation, always). `src_vocab_size` is kept when the base states a size that holds every piece of the student's model
plus Eole's pad token, refused (`ConfigError`) when it states less, and set to the pieces plus one when absent (641 for
the toy's 640-piece model, 501 for the 500-piece one, 64,001 for `student_64k`). The manifest's inputs are the base
config, the corpus files, the KD files with their manifests and the SentencePiece model, each hashed once, before anything
read it; its metrics are the KD counts (`kd.count.*`), D7's counts (`count.*`, including `count.source_collision_rate` of
the original rows), `mix.kd`, `mix.original`, `kd_tag`, `bt_ratio`, `temperature`, `weight_total`, `weight.<corpus>`,
`kd.repeat.<direction>` and `valid.lines`; its status says `prepare: ok`, `tag_placement` and `mix`. The student then
trains with the launcher (`rinchen train launch --config <O>/config.yaml --run-dir <O>/run`), resumable as any D8 run.

### Hashing once and the order of the refusals

**Hash once (the controller's ruling).** The command hashes every input (corpus files, KD files and their manifests, the
SentencePiece model, the base config) ONCE, through `hash_input`, before reading it, and passes the KD digests to
`load_kd_rows`, which compares them with the SHA-256 the KD manifest states instead of hashing a large file again; a spy test
pins that the loader does not hash a file it was given a digest for. The window between the hash and the read is the one
every command of this project has. **The order**: every refusal comes first (the options, the glob and `$` characters, the
output directory being an input's directory, a foreign manifest in it, a KD file without a valid manifest, a base config
that holds `train_from`, a stated vocabulary size that is too small), then `before_writing` runs ONCE (the command uses it
to make the output directory and to remove the old manifest, `begin_commit`), then the first write. A refusal therefore
never deletes an old manifest and never leaves a partial directory; a dry run never calls it, so it writes nothing, not even
the directory (the gate checks a digest of the output directory before and after a `--mix 3:1` dry run over a prepared
student). An unreadable KD manifest is exit 3 (the command's own read), a missing or foreign one exit 2 (the loader's).

### Where "exact" ends

The ratio is exact in the WEIGHTS. The draw share rests on `D-EOLE-WEIGHT-SEMANTICS`, which was measured for Eole's own
mixer (the first 1,000 examples, and a real 150-step run whose corpora counted 22/6/22/6/15/7/15/7 sentences). Not run:
`num_workers` above 1 together with weights (the shipped tiny config has `num_workers: 1`, `bucket_size: 1024`); a ratio
whose weight sum is far above `bucket_size` (`--mix 10000:9999` with the default `--weight-total 100` gives a weight sum
of 1,999,900, far above any bucket, and there is no guard). Raise neither without measuring Eole's mixer at that scale.

## The student configs

### Shapes and parameters (ARITHMETIC)

Three base configs, all `data: {}` with no per-run value (`distill prepare` merges `src_vocab`, `training.model_path`,
`transforms_configs.sentencepiece`, `data`, `seed` and `src_vocab_size` in and validates the merged config before Eole sees
it); all `SinusoidalInterleaved`, shared embeddings, no relative, rotary, ALiBi, sliding-window or grouped-query attention.

| | `student_server.yaml` | `student_edge.yaml` | `student_tiny.yaml` | teacher (`teacher.yaml`) |
|---|---:|---:|---:|---:|
| encoder layers | 16 | 12 | 2 | 24 |
| decoder layers | 3 | 2 | 1 | 12 |
| hidden size, `word_vec_size` | 768 | 512 | 64 | 1,536 |
| heads | 12 | 8 | 4 | 16 |
| feed-forward | 3,072 | 2,048 | 128 | 4,096 |
| `src_vocab_size` | 64,001 | 64,001 | not stated (`prepare` sets pieces + 1) | not stated (128,001 to state) |
| parameters (ARITHMETIC) | 190,845,963 (about 190.8 M) | 79,006,218 (about 79.0 M) | 138,201 for a vocabulary of 344 | 1,102,982,164 (about 1,103.0 M) |
| fp32 weights | 0.76 GB | 0.32 GB | 0.55 MB | 4.41 GB |

The shapes of the server and edge students are the master plan's (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`,
milestone D10: encoder 16, decoder 3, hidden 768 and encoder 12, decoder 2, hidden 512); the vocabulary is the student's own
SentencePiece model (`configs/spm/student_64k.yaml`, 64,000 pieces, plus Eole's pad token: 64,001). **The counts are
ARITHMETIC, not measurements**: the formula of runbook 08 (`V (H + 1) + L_enc (4H^2 + 2 H F + 4H) + L_dec (8H^2 + 2 H F +
6H + 1) + 4H`, `V` the vocabulary rounded up to a multiple of 8, so 64,008 for 64,001) applied to the values of the configs.
The formula is calibrated EXACTLY on the two tiny checkpoints: 187,738 for the 2/2 teacher and 138,201 for the 2/1 student
equal the element counts of their weights files less the 640,000 elements of the two 5,000-position sinusoidal buffers
(**measured**: `s11_params.py`). The size of `model.bin` is consistent with four bytes per parameter plus one position table of
`5,000 x H` (2,034,311 B = 4 x 187,738 + 4 x 5,000 x 64 + 3,359 B; 1,835,438 B = 4 x 138,201 + 1,280,000 + 2,634 B): the
teacher's export would be about 4.4 GB in float32 (ARITHMETIC, an inference from two sizes, not a reading of the converter).
Nothing here was trained or timed at these sizes.

The schedule of the server and edge configs is the teacher's, scaled to the model, and every value is an UNMEASURED prior:
Noam with learning rate 2.0 and 8,000 warm-up steps peaks near 8.1e-4 at width 768 and 9.9e-4 at width 512 (2.0 x H^-0.5 x
8000^-0.5, ARITHMETIC), 16,384-token batches with an accumulation of 8 on 8 GPUs (1,048,576 padded tokens per update),
bf16, 100,000 steps, a validation and a save every 5,000, early stopping after 6 validations on perplexity, `keep_checkpoint`
8. They state `attention_dropout: [0.1]`, which `teacher.yaml` does not; the headers say so (review M6; the values are
unchanged). `student_tiny.yaml` has the optimiser of `teacher_tiny.yaml` (AdamW, Noam, learning rate 0.5, 100 warm-up steps,
batches of 2,048 tokens, dropout 0, 2,000 steps, a validation and a save every 500, all kept, `filtertoolong` 64).

### The depth finding

Unequal encoder and decoder depths need no code: `model.encoder.layers` and `model.decoder.layers` are in the allow-lists,
and Eole, `launch` and CTranslate2 accept a 2/1 shape (**spike**, **measured**: the gate). The checkpoint's saved
`config.json` records `encoder.layers: 2` and `decoder.layers: 1`, and the converter reads the counts from the checkpoint.
**A 3-encoder, 1-decoder student at the same recipe learned much more slowly**: dev chrF 38.4 after 2,000 steps against
98.6 for 2/1, validation accuracy 79.5 % against 99.4 % (**spike**). It is slow and not broken, but a depth needs its own
recipe: do not change `student_tiny.yaml`'s layers without measuring, and expect the real shapes (16/3 and 12/2) to need
their learning rate and steps worked out on real data, which nobody has done.

### The vocabulary rule and a SentencePiece model of its own

The student has its own SentencePiece model; the KD corpus is TEXT, which each model cuts for itself (the KD files hold the
decoded teacher text; Eole's `sentencepiece` transform cuts it with the student's model). What depends on the model:

1. the pre-tokenised VALIDATION corpus, written with the student's model (rebuilt, never reused from another);
2. `src_vocab_size`, at least the model's pieces plus one (`launch` refuses less, D9; `prepare` sets it when absent);
3. the direction tags and `<ft>` must be single pieces of the student's model (`require_one_piece`; true of every project
   SentencePiece model, which holds the 118 reserved symbols);
4. the evaluation uses the model of the checkpoint it scores;
5. Eole's vocabulary is built from the student's data, plus the 118 reserved symbols (D9's invariant).

**Spike** (the chrF) **and gate** (the preparation and the 200 steps): a student with a 500-piece model (the teacher's has 640) trained to a validation accuracy
of 99.3 % and translated at dev chrF 89.4 against 98.6, because the coarser model cuts the nouns into characters (`The
book and the snow.` is cut the same, `Ном ба цас.` into 4 pieces by the 640-piece model and 7 by the 500-piece one; not a
defect of KD). The gate prepares such a student (`src_vocab_size` 501, `valid.src` cut with that model, all 118 reserved
symbols in its vocabulary) and trains it for 200 steps (21.9 s here, 18 s in the gate).

## Evaluation: teacher against student

`distill evaluate` runs D7's `evaluate_variant` on one checkpoint with its own SentencePiece model and tag placement over
the `dev` rows of the corpus files, for the directions their TRAIN rows have, the first `--max-lines` by id of each, as
`rinchen train ablate` scores them. All lines of all directions go to ONE `eole predict` on the CPU with greedy search
(the GPU hidden, `--threads` default 2), and each direction is scored on its own slice. It prints the lines of step 6
above and writes `result.json` (the scores per direction, `chrf_mean`, `chrf_min`, `tag_leaks`, the metric signatures and
`checkpoint`, `spm_sha256`, `placement` and `max_lines`; sorted keys, two-space indent, LF; the same inputs give the same
bytes), `hyp.<direction>.txt`, Eole's own `dev.*` files and, last, `manifest.json` (stage `distill_evaluate`; metrics
`chrf_mean`, `chrf_min`, `tag_leaks`, `dev_in_train`, `lines.<direction>`, `chrf.<direction>`, `bleu.<direction>`,
`count.dev.*`; status `evaluate: ok` and `tag_placement`). The old manifest and result are removed before the prediction
starts, so a failed run leaves none. The checkpoint need not have a manifest (a baseline or a foreign model may be scored;
the command records the SHA-256 of its `config.json`, `vocab.json` and weights, taken before the prediction).

**To compare a teacher and a student** run the command twice with the SAME `--corpus` files and the same `--max-lines`:
the same dev rows give comparable numbers, each checkpoint with its own `--spm-model` (a student and its teacher may have
different ones) and the placement it was trained with. `dev lines in training` counts the dev lines whose id is also a
training row's (such a line inflates a score; 0 here); `tag leaks` counts hypotheses that hold a direction tag. No test row
is read. The student's `ckpt` link directory (a run's, in which Eole links the newest step's files) scores the same as its
`step_2000` directory (**measured**: the same chrF and byte-identical hypotheses).

**Measured** (the gate and this run, 100 dev lines per direction): teacher 98.20 (98.51, 98.29, 97.79, 98.20), student 98.58
(98.82, 99.30, 97.54, 98.67). The toy ranks nothing: the same student over three seeds gave 98.58, 98.59 and 96.68, and the
originals-only student 97.01, inside the spread plus 1.6 points: **do not read 97.01 against 98.80 as a measurement of KD**.

## The provenance chain

The teacher's `train_launch` or `train_average` manifest -> `export.json` and the export's manifest (`distill_export`, the
checkpoint's four digests recorded) -> the KD manifest (`distill_generate`: the five export files, `export.json` and the
export manifest by hash, `teacher.checkpoint_*` and `teacher.model_sha256` as status) -> the prepare manifest
(`distill_prepare`: the KD files and their manifests by hash, the base config, the corpora) -> the student's `train_launch`
manifest (the config by hash, the run identity) -> the student's own export record. Every command hashes its inputs BEFORE
it reads them, refuses a foreign manifest in its output directory (`refuse_foreign_manifest`), refuses an output directory
that is an input's, writes its manifest LAST (`begin_commit`) and exits as D9's commands do: 0; 2 a bad option, config or
provenance; 3 unreadable or empty data; 4 a tool failed; 1 the output cannot be written; 130 interrupted. A checkpoint
without our manifest is never a teacher, and a KD file without its manifest is never a source of training data. These are
guards against mistakes, not against someone who writes a manifest by hand.

## The STOP gate

`tests/integration/test_distill_tiny_student.py` (real `rinchen`, real Eole, real CTranslate2, CPU, GPU hidden, `-m
subprocess`, every subprocess bounded by 900 s; 44 tests, 180 to 240 s for the module). Its stages are the helpers of
`tests/support/distill_gate_stages.py` (the same ones this runbook's measurement script reused). Module fixtures run the
expensive commands once: **T** the tiny teacher (`teacher_tiny.yaml` with 2,000 steps, a save every 500, source tag; `ablate
--prepare-only`, `launch`, `average --last 2`); **X** `distill export` of it; **G** `distill generate` over the 3,600
authentic training sources and a rerun; **P** `distill prepare --mix 1:1` for the tiny student and two dry runs of `--mix
3:1`; **S** `launch` of the student, 2,000 steps from random initialisation; **E** `distill evaluate` of the teacher, the
student's final step and the student's `ckpt`; **Q** the student's export and CTranslate2 against Eole on 200 dev sentences;
**V** a second student with a 500-piece SentencePiece model, 200 steps; **R** fifteen refusals. The toy TEST split is written
to no file a command reads; the test process looks at it only to show that none of its sentences reached the generator.

The thresholds (never lowered; a miss is answered by raising the steps of the checked-in config):

| Gate | Threshold | Measured (gate, this run) |
|---|---|---|
| KD rows | at least 3,550 of 3,600 sources | 3,599 |
| KD targets equal to an authentic target | at least 90 % | 3,583 of 3,599 (99.6 %) |
| teacher mean chrF on the dev lines | at least 90.0 | 98.20 |
| student mean chrF | at least 95.0 and at least the teacher's minus 3.0 | 98.58 |
| CTranslate2 greedy equal to Eole's on 200 dev sentences | at least 95 % (D11's figure; recorded here) | 200 of 200 (pieces and text) |

Also pinned: every count conserved; no source of a dev or test row; origin `ft` and the label; the rerun reuses 10 of 10
shards with the same rows; the KD weight of each direction equals the sum of the originals' (`1:1`), `3:1` gives exactly
3 times; every KD corpus carries `<ft>` and no `train_from` or `reset_optim` is rendered; the student's `config.json` says 2
encoder and 1 decoder layers; the student's export record agrees with its launch manifest's weights hash; a smoke
translation; the 500-piece student's `src_vocab_size` 501, validation corpus and vocabulary; fifteen refusals; and the GPU
claim itself, shown by an AST scan of the new code (no `nvidia-smi` string, no `torch.cuda`, every `ctranslate2.Translator`
with `device="cpu"`) and by Eole's own log lines (`GPU ranks: []`, `Starting training in cpu mode`).

**What the numbers show.** That the whole chain (export with provenance, the teacher behind D6's seam, shards and
selection, the exact mix, the preparation of a student of another shape and vocabulary, the launcher, the evaluation, the
student's export) works through the real binaries on the CPU, that a student of unequal depth trains from random
initialisation on KD text, and that CTranslate2 reproduces Eole's greedy output exactly for three tiny models (a source-tag
teacher, a target-tag teacher with a forced prefix, a student). **What they do not show.** Anything about quality or about
whether KD helps: the toy ranks nothing (three seeds of the student span 1.9 chrF, and the originals-only student is inside
the spread), the teacher's KD targets are 99.6 % the references, and the thresholds sit far from the measurements on
purpose (the student's 95.0 is 3.6 below 98.58 and 1.7 below the lowest of three seeds, 96.68; a margin judged adequate on
one machine). Nothing ran on a GPU, in float16 or int8, or at a real size; the strict parity and quantisation gates are D11's.

The measurements agree between the spike, the gate and this run (a seed-fixed, CPU, two-thread run on this machine): the
teacher's `model.bin` (SHA-256 `b7f33d88d5fa...`), 3,599 rows, 3,583 matches, 28/28/22/22, chrF 98.20 and 98.58, 200 of 200.
A run on another machine or another thread count may differ.

## The operator workflow for a real distillation

UNMEASURED: no real corpus, no real teacher and no GPU run exist, and every value of `student_server.yaml` and
`student_edge.yaml` is a prior. The steps are the ones the tiny gate ran; the numbers to measure are the ones listed.

1. **Train the teacher** with runbook 08 (`rinchen train ablate --prepare-only`, `rinchen train launch`), with the
   vocabulary invariant (D9), then `rinchen train average --checkpoint-dir <run>/ckpt --last N`. Its saved config must state
   the tag placement through the `data` prefixes (every checkpoint our launcher wrote does).
2. **Export it**: `rinchen distill export --checkpoint <averaged or final step> --spm-model <the teacher's model>
   --tag-placement <src|tgt> --out-dir <O>`. UNMEASURED for a real teacher: the time to hash a 13 GB step directory (the
   record hashes the four files and `verify_source` reads them), the converter's time and MEMORY for 1.1 B parameters on the
   CPU (it runs with the GPU hidden by design), the size of `model.bin` (about 4.4 GB in float32, ARITHMETIC), and whether a
   second copy of the checkpoint's weights is needed on disk (the converter reads from the checkpoint, writes `model.bin`).
3. **Dry-run the generation**: `rinchen distill generate --corpus <D4 output> ... --teacher <O> --out-dir <K> --dry-run`
   and read the number of sources and shards per direction; every source is an authentic training row of the corpus files.
   Then without `--dry-run`. Defaults: beam 4, `--n 1`, `--max-decoding-length 256`, `--max-source-pieces 256`, batch 64,
   shards of 512 sources, `--device cpu`. **On a GPU** (UNMEASURED: nothing of it was run; the operator's explicit choice,
   with the owner's authorisation): `--device cuda --compute-type float16` (or `int8_float16`) and a `--batch-size` that
   fits the card; `--threads` is the CPU's intra-op threads only. The fingerprint includes `device` and `compute_type`, so
   another device means other shards. Check `matches reference` (expect it far below the toy's 99.6 %), the six teacher
   counters (a large `truncated` says `--max-decoding-length` is too small; a large `source_too_long` that
   `--max-source-pieces` is; any `reserved_symbol` says the model emits pieces no translation should hold) and the
   dropped-purity and no-survivor counts of every direction before trusting the corpus.
4. **Prepare the student**: a copy of `configs/eole/student_server.yaml` (or `student_edge.yaml`) as the base, the student's
   SentencePiece model (`configs/spm/student_64k.yaml`), `rinchen distill prepare --corpus ... --kd <K>/kd.parquet --spm-model
   <M> --config <base> --out-dir <S> --mix 1:1 --dry-run` first, read the weights and the `repeat` factors, then without
   `--dry-run`. UNMEASURED: the right `--mix`, `--bt-ratio`, `--temperature` and `--weight-total` for real data, and whether
   `<ft>` helps (a student trained with `--no-kd-tag` is the experiment).
5. **Train**: `rinchen train launch --config <S>/config.yaml --run-dir <S>/run --world-size 8 --gpu-ranks 0,1,2,3,4,5,6,7`
   (and `--write-only` first for `command.sh`). Everything of runbook 08 applies. The first contact with a GPU needs the
   owner's authorisation. UNMEASURED: bf16, NCCL, the speed, the learning rate and the steps of a 16/3 and a 12/2 student,
   and whether a deeper-encoder student converges at the teacher's recipe (3/1 did not at the tiny one).
6. **Evaluate** teacher and student with `rinchen distill evaluate` over the same dev rows and the same `--max-lines`, each
   with its own SentencePiece model and placement; then export the student with `distill export`. Quantisation and the
   strict parity gate are D11's.

## Open gates

Not done by D10, because no real data, GPU or reader exists: a real teacher's KD corpus and the measured quality of a
student distilled from it; the real mix ratio, the repeat factor that is too high, the n-best and MBR on a real teacher and
the QE thresholds (the real scorer was never loaded); the real students' training (speed, convergence, learning rate,
steps, the depth recipe); the real share of Eole's deduplication on real data and whether `<ft>` at inference helps; CTranslate2
on a device other than the CPU, in float16 or int8, and the export of a 1.1 B-parameter teacher (D11 owns the strict parity
and the quantisation gates); a document-level teacher, monolingual KD sources and several teachers (extension points:
`ct2_source_tokens`, `KdSources`, the loader that merges KD files per direction, `eole_data.data_section`'s `placement`); the
`tgt` placement for a real student (covered by unit tests only); the D3 gates assigned to D4, the D5 gates that need real
text or a reader and the D6 to D9 gates stay open as recorded in their runbooks.

## What was not verified

- **Anything on a GPU**, and a real corpus. Every translation ran on the CPU with CTranslate2 on `device="cpu"`; every Eole
  run had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`; `--device cuda` and `--compute-type` were passed through a faked
  seam only. Float16, int8, the export of the 1.1 B-parameter teacher (time, memory, a 13 GB checkpoint's hashing) and the
  real students' training are unmeasured.
- **A real teacher's n-best, MBR and QE.** On the toy, ranks 2 to 4 are 41 chrF below rank 1 and MBR lowered the agreement
  with the references; with a real teacher nothing is known. The real QE scorer needs a GPU and was never loaded.
- **The share of Eole's deduplication on real data**, whether `<ft>` at inference helps or hurts, and the effect of the
  mix ratio and of the repeat factor on a real student.
- **The `tgt` placement for a real student**: covered by unit tests and by the target-tag teacher of the spike (CTranslate2
  equal to Eole on 200 of 200), never trained as a student.
- **`num_workers` above 1 with weights, and a weight sum far above `bucket_size`** (the draw share is measured for Eole's
  own mixer with small sums only).
- **Scale.** `kd_sources` holds the authentic targets of every source as a frozenset in memory, the plan and the fingerprint
  are computed twice in a run, and all KD rows are in memory; the memory of any of it for tens of millions of rows is
  unmeasured (D7's loader, for comparison: about 1.05 GiB for each further million rows of this text length).
- **A document-level teacher** (D9's windows `<2xx> <doc> a <sep> b` go through the same code path but were not run).
- **`KeyboardInterrupt` with a real CTranslate2 call** (exit 130 is tested with fakes; the real call is a blocking C++ call
  that delivers the signal only when its batch returns).
- **CI**: no CI run covers D3 to D10. **Windows and macOS** were not run.
- **The real binary** was run for the whole flow above, the refusals and the placement checks; every other path ran in
  process with fakes at the `run_logged` and CTranslate2 seams.

### Known limits, left open on purpose

Findings of the reviews that were parked and not fixed, and limits of the build, one each:

1. **`generate` exits 0 with `generate: ok` when the teacher yields few or no rows** (as D6's synth does): an empty selection
   writes an empty `kd.parquet` and a manifest, and `load_kd_rows` refuses a corpus without rows only later, in `prepare`.
   Read `selected` and `no_survivor` of every direction. (Review M3.)
2. **A resumed generation reports the teacher counters of THIS process only** (all zero when every shard was reused;
   `teacher_counters_cover` says so), while `matches_reference` and `letterless_sources` cover the reused shards too; and the
   manifest key `teacher.name` holds the KD corpus name (`rinchen-<12 hex>` or `--name`), not a name of the teacher (spec T5).
   (Review M4.)
3. **"Exact" means exact WEIGHTS.** The share of the draws rests on `D-EOLE-WEIGHT-SEMANTICS` (measured for Eole's mixer);
   `num_workers` above 1 with weights was not run; `--mix 10000:9999` with the default `--weight-total 100` gives a weight sum
   of 1,999,900, far above `bucket_size`, and there is no guard. (Review M5.)
4. **Memory and scale are unmeasured** (`KdSources.references` frozensets, the plan and the fingerprint computed twice, all KD
   rows in memory). (Review M7.)
5. **`prepare` prints no `source_collision_rate`**, at any placement (the manifest has `count.source_collision_rate` of the
   original rows, 0.0 on the toy, which says nothing about KD rows meeting original rows), although `D-EOLE-BATCH-DEDUP`
   applies to multi-parallel sources under `--tag-placement tgt`. (Review M8.)
6. **`distill evaluate` on an unparseable or non-Eole `config.json` exits 3** (it was 4 before the placement check read the
   file: found by the fix wave, not changed).
7. **Files near the 900-line limit**: `src/rinchen/train/distill_export.py` 754 lines, `eole_cfg.py` 860 (856 in D9), `launch.py`
   886, `continuation.py` 777; the gate `tests/integration/test_distill_tiny_student.py` has 893 lines (it grew from 892 in
   the fix wave) and `tests/unit/test_cli_distill_generate.py` 827: the next test for the gate goes in a module of its own, the
   next change to `launch.py` has to move something. (Review.)
8. **The toy ranks nothing**: three seeds of one student span 1.9 chrF (98.58, 98.59, 96.68), the originals-only student sits
   inside the spread. Do not cite the gate as evidence that KD helps.
9. **The student threshold margin** is 95.0 against 98.58 measured (3.6 points; 1.7 below the lowest seed). If the gate flakes on
   another machine raise `train_steps` in `student_tiny.yaml`, never lower the gate.
10. **The wording debt of `verify_source`**: a checkpoint it refuses is reported as "cannot continue from ..." although the
    command is an export (D9's function, left as it is).
11. **Every GPU-side claim is unmeasured** (see "What was not verified").
12. **The export record does not say whether the placement was derived or stated**, and an export made before the check
    cannot be re-verified (the saved config is not in the export directory). A saved config without a `data` section or a
    direction tag leaves the statement unchecked.
13. **An export's digests are trusted, not re-hashed, by the commands that read it**: `generate` uses the ones `verify_export`
    took, and the weights are read by the converter after the checkpoint's digest was taken. A kill can leave
    `.<name>.<random>` beside `--out-dir`.
14. **A failing preflight leaves an empty output directory** (`generate` creates it early, as D6 does): a teacher without a
    tag it needs (`check_directions`) or a failing teacher build; the gate's vocabulary-without-a-tag case is found AFTER the
    directory is made and leaves it empty.
15. **`export_ct2` and D5's `convert-script export` do not refuse glob characters** in their paths (unchanged since D5,
    `D-EOLE-GLOB-CKPT`); `distill export` refuses them before it calls the converter.
16. **A shard written by other code that holds candidates for a letterless source is used as is**; `matches_reference` is an
    exact string comparison and says only how close the teacher is to the data.
17. **No Python 3.14 assertion in the new CLI modules** (none of the existing CLI modules has one: `requires-python` in
    `pyproject.toml` pins `==3.14.*`).

### The review and what its fix wave did

An independent combined review of Tasks 1 to 7 (`e5ff644..e67b05e`, read-only; it ran the integration gate, 44 passed in
240 s with the numbers above, and probed through the real `rinchen`) found: **ready with fixes, 0 Critical, 2 Important
and nine groups of Minor findings**, and checked what it could not break (the export move, the hashes before the parse, the
SentencePiece comparison, the stripping of the target prefix, the alignment when sources are skipped, `auto` refused, the
lazy imports, the completeness of the fingerprint, a truncated shard regenerated, the QE orientation, the mix through the
real CLI, the dry runs, KeyboardInterrupt, the hygiene). Its fix wave, one commit each with a failing test first:

- **I1** (`95a28c8`): the tag placement was the operator's word and never checked although the checkpoint's saved
  `config.json` records it; a source-tag teacher exported with `--tag-placement tgt` exited 0 with a vouching manifest, and
  `generate` over it produced 1,190 rows of which one matched. The new `distill_saved_config.py` reads the placement back
  and refuses a disagreeing statement, in `export` and in `evaluate` ("The tag placement and its check"; 23 mutants).
  The spec's claim that the placement "is not in the checkpoint" was false for our launcher's checkpoints.
- **I2** (`46c160e`, `cc917e2`): a hypothesis holding the unknown piece, a reserved symbol or a direction tag decoded to
  literal text and passed the selection. The teacher now drops it before decoding and counts it under the new counter
  `reserved_symbol`, which reaches `counts.json`, the manifest metrics and the result lines; the second commit extended the
  rule to the vocabulary entries that are no piece of the SentencePiece model (`<blank>`, `averyunlikelytoken0`). The
  wrong-placement export produces Cyrillic garbage with NO reserved symbol, so I2 would not have caught it: I1 prevents it.
- **Minors** (`6eb778a`, `D10: review minors`): M1 the export cleanup could delete another process's output directory
  (the staging layout above), M2 the vocabulary entry count cross-check and the stated TOCTOU, M6 the headers of the student
  configs (`attention_dropout`), M9 the wording of the glob refusal of `generate --out-dir` (the command starts no Eole;
  `reject_dollar` and `reject_glob_characters` take a `reason`). And `d3ed47f`: one of the new tests took a digest of the
  directory two levels above its checkpoint, which in a full run holds another test's unreadable file.

Edits to existing tests, all forced: the counter key set in five files (`reserved_symbol`), three assertions on the
converter's output directory (a child of a private directory beside `--out-dir`), the CUDA probe of
`test_script_conv_tiny.py` now patches `rinchen.export.eole_ct2.export_environment`. Gate after the fix wave: **9706 passed, 1
skipped, 10 deselected** (20:57), against 9439 before it and 8252 at the end of D9 (8517 after Task 1, 8676 after Task 2,
8808 after Task 3, 9064 after Task 4, 9220 after Task 5, 9395 after Task 6); the integration module 44 passed twice (180 to
190 s). Left open on purpose: the Minor findings M3, M4, M5, M7, M8 and the length of the gate's test file (Known limits 1 to 5,
7). A scoped re-review of the fix wave (`e67b05e..cc917e2`) was running when this runbook was written; its verdict and any
finding it made are not recorded here.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython 3.14.10,
eole 0.6.4, ctranslate2 4.8.4, torch 2.14.1+cu130, sentencepiece 0.2.1, polars 2.1.0, sacrebleu 2.6.0; the GPU (RTX 5070 Ti)
was hidden from every process and every CTranslate2 translator was built on `device="cpu"`. All on 2026-11-09, once each,
code at commit `cc917e2`, **on a machine that was loaded by other work (load average 7 to 11), so every wall time is longer
than an idle machine's** (the gate's times are in parentheses); the numbers that are not times do not depend on the load.

| What | Value |
|---|---|
| the tiny teacher: `ablate --prepare-only`, `launch` (2,000 steps, a save every 500), `average --last 2` | 1.84 s, 78.18 s (65 to 111 s in the gate), 1.29 s; averaged weights SHA-256 `f4e32704b958...` |
| `distill export` of the averaged teacher | 5.08 s (3.4 s); `model.bin` 2,034,311 B, `shared_vocabulary.json` 8,659 B (344 entries), `config.json` 223 B, `spm.model` 12,612 B (12,605 B in the gate's workspace), `export.json` 710 B, `manifest.json` 3,360 B; `du -sb` 2,059,875 B; log `teacher_ct2.export.log` beside it |
| the wrong placement on that checkpoint: `export --tag-placement tgt`, `evaluate --tag-placement tgt` | exit 2 in 1.32 s and 1 s, nothing made, the message above |
| `distill generate --dry-run`, `generate`, the rerun, a one-direction run | 1.46 s, 2.75 s (1.6 s), 2.17 s (10 of 10 shards reused), 2.06 s (1,200 sources); fingerprint `5c6b165b2906...` (`9b2526dd3e65...` for the one-direction run) |
| the KD corpus | 3,599 rows; `kd.parquet` 90,707 B (spike: 90,682 B), `counts.json` 3,735 B, `manifest.json` 7,006 B; `du -sb` of the output directory 175,166 B; 1 `dropped_purity` and 1 `no_survivor` (`khk_Cyrl-rus_Cyrl`); `matches reference` 3,583 of 3,599; every teacher counter 0 but `sources` 3,600 |
| `distill prepare`: the dry runs of `1:1`, `3:1`, `3:2`, `2:1` and of the partial-coverage cases, then `1:1` | 1.3 to 1.9 s each; weights in "The weights"; `config.yaml` 7,689 B, `manifest.json` 16,897 B, `valid.src` 22,374 B and `valid.tgt` 17,574 B (400 lines); a second run into the same directory: `corpora: 12 (reused 12)` |
| the student: `launch` (2/1 layers, 2,000 steps) | 64.86 s (59 to 77 s); `du -sb` of the student directory 17,992,021 B (data, run and checkpoints); `vocab.src` 339 lines |
| `distill evaluate` of the teacher, the student's step and the student's `ckpt` (100 dev lines per direction) | 4.39, 4.16 and 4.10 s (3.8 s each); chrF mean 98.20, 98.58 and 98.58 (the last two have byte-identical hypotheses and a `result.json` that differs only in the checkpoint path), min 97.79, 97.54, 97.54, BLEU per direction in step 6 |
| `distill export` of the student | 3.75 s (3.2 s); `model.bin` 1,835,438 B (the teacher's 2,034,311 B), the other files as the teacher's, `export.json` 705 B (2/1 layers, hidden 64, 344 entries), `du -sb` 1,860,909 B |
| CTranslate2 greedy against `eole predict` greedy, 200 dev sentences, the student | 200 of 200 equal in pieces and in text; "This is the star." gives "Энэ бол од." |
| the second student (a 500-piece SentencePiece model): `prepare`, `launch` of 200 steps | 0.95 s, 21.88 s (18.4 s); `src_vocab_size` 501 |
| the deduplication counts (spike's `s6_dedup.py`, 100 steps, `num_workers: 0`) | tagged 0 of 13,312 (152 s) and 0 of 102,400 (228 s); untagged 798 of 15,360 (173 s) and 5,317 of 102,400 (233 s): the spike's numbers exactly |
| the fifteen refusals | 0.8 to 1.1 s each, 15.4 s in all (11 s), all exit 2 |
| a SIGKILL of `distill export` 1.2 s after its private directory appeared | `.crash_export.<random>/export/` with the converter's three files, the converter's log beside, no output directory |
| parameters (ARITHMETIC, `s11_params.py`) | tiny teacher 187,738 and tiny student 138,201 (V 344, exact); teacher 1,102,982,164, `student_server` 190,845,963, `student_edge` 79,006,218 |
| `ruff format --check .`, `ruff check .`, `mypy src` | clean (449 files formatted; 175 source files) on this tree |
| the full gate | not re-run for this record: only documents differ from `cc917e2`, whose gate gave 9706 passed, 1 skipped, 10 deselected (the unit tests read no document) |

**The numbers agree between the spike, the gate and this run** wherever they are not times: the 3,599 rows, the 3,583
matches (1,200, 1,191, 597, 595), the weights, chrF 98.20 and 98.58 with the per-direction scores, the 200 of 200, the byte
sizes of `model.bin`, `shared_vocabulary.json`, `config.json` and the student's `model.bin`, and the deduplication counts.
The one byte count that differs is `spm.model` (12,612 against 12,605 bytes: the toy model is trained afresh in each
workspace); `kd.parquet` is 90,707 bytes here against 90,682 in the spike (whose prototype labelled the rows `ft:teacher`, not
`ft:rinchen-<12 hex>`).

The measurement scripts were read-only scratch files in the session scratchpad (`d10-t8/run_pipeline.py`,
`run_refusals.py`, `run_dedup.sh` and `dedup.py`, a copy of the spike's `s6_dedup.py` that only changes the scratch directory
and asserts the hidden GPU); they reuse the helpers of `tests/support/distill_gate*.py`, so the numbers are the gate's own. Nothing
was written under `data/` or `models/`.

## Where the code lives

`src/rinchen/export/eole_ct2.py`;
`src/rinchen/train/{distill_export,distill_export_files,distill_saved_config,distill_teacher,distill_sources,distill_shards,
distill_generate,distill_corpus,distill_mix,distill_prepare}.py` with the extensions of `eole_data.py` (`kd_tag`, `ft_tag`),
`mixture.py` (`KD_GROUP`) and `eole_cfg.py` (the `reason` of two refusal helpers); `src/rinchen/synth/tags.py` (the entry
`teacher`); `src/rinchen/script_conv/export.py` (the re-exports); `src/rinchen/cli/{distill,distill_export,distill_generate,
distill_generate_plan,distill_generate_records,distill_prepare,distill_evaluate,main}.py`;
`configs/eole/student_{server,edge,tiny}.yaml`; unit tests `tests/unit/test_export_eole_ct2.py`,
`test_train_distill_{export,export_consistency,export_placement,export_record,export_staging,saved_config,teacher,
teacher_loading,teacher_options,teacher_reserved,sources,generate,generate_fingerprint,generate_shards,corpus,mix,prepare,
prepare_refusals,prepare_seams,student_config}.py`, `test_cli_distill_{export,generate,generate_qe,generate_refusals,
generate_resume,prepare,prepare_refusals,evaluate,evaluate_placement,evaluate_refusals}.py`,
`test_eole_configs_students.py` and the additions to `test_eole_configs_shipped.py`, `test_script_conv_export.py`,
`test_synth_tags.py`, `test_train_eole_cfg_shared.py` and `test_train_eole_data.py`; property tests
`tests/property/test_train_distill_{mix,generate}_property.py`; `tests/integration/test_distill_tiny_student.py` (the STOP
gate, scenarios T, X, G, P, S, E, Q, V, R); and the helpers `tests/support/{distill_cli,distill_exports,distill_gate,
distill_gate_refusals,distill_gate_stages,distill_teachers,fake_ct2,planted_kd,planted_student}.py` (and extensions of
`step_dirs.py`).
