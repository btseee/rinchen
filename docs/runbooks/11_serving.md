# 11 - Export, quantisation and serving

## Scope

D11 makes a model of ours runnable. It builds the export directory with a record of its own (`rinchen.json`) and a quantisation
chosen at conversion (`rinchen export convert`, `verify`, `parity`); the engine that reads such a directory with the guards that
CTranslate2 does not have (`rinchen.infer.engine.Engine`, behind the Python class `rinchen.Translator`); script masking; the
command `rinchen translate` (standard input to standard output); the service `rinchen serve` (FastAPI, a micro-batcher, Prometheus
text, a graceful stop); the CPU container recipe (`docker/`); the packaging that lets an installed wheel load the language registry;
and the gates: strict Eole/CTranslate2 parity, the int8 quantisation gate, the script-masking gate and the serving gate. The design,
with the rulings T1 to T22 and a section of the amendments made while building (A1 to A23), is
`docs/superpowers/specs/2026-11-09-d11-export-serving-design.md`; the decisions are condensed in the entries "2026-11-09: D11
export, quantisation, serving: the rulings" and "2026-11-10: D11 review - corrections and fixes" of `DECISIONS.md`; the upstream facts
are in `docs/VERIFY.md` (section "D11 findings"). The CTranslate2, Eole, FastAPI and Docker facts come from a spike made before the
design (`.superpowers/sdd/2026-11-09-d11-export-serving/spike-findings.md`; Eole 0.6.4, CTranslate2 4.8.4, FastAPI 0.162.0, uvicorn
0.54.0, CPU, toy data).

**What D11 is.** Tested machinery. A checkpoint of ours (the final step of a run `rinchen train launch` finished, or a directory
`rinchen train average` wrote, with the manifest our commands wrote) becomes a directory of five files and a manifest; the engine
verifies that directory by hash before it loads it and answers each text with a status instead of a silent cut; the command and the
service are two front ends over the same engine; and everything is exercised through the real `rinchen` binary, the real Eole and the
real CTranslate2 on the CPU, on the templated toy sentences of `tests/support/tiny_parallel.py`.

**What D11 is NOT.** Nothing below is a statement about the quality of any translation.

- **No GPU was used.** The machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run. Every Eole process, the
  converter and every CTranslate2 translator of D11 (the spike, the tests, this runbook's measurements) ran with
  `CUDA_VISIBLE_DEVICES=""`, `gpu_ranks: []` and `device="cpu"`; `device="auto"` and `compute_type="auto"` are refused everywhere;
  `--device cuda` exists for the operator and is validated, never run. Everything GPU-side (CTranslate2 on a device, `float16`,
  `int8_float16` as a GPU type, token batching, `configs/serve/server.yaml`, the speed of a 1.1 B-parameter teacher or a 190 M
  student) is **UNMEASURED**.
- **No real model and no real corpus.** The toy translates templated sentences (37 nouns in a few frames) and saturates at chrF 98.2,
  so every quality number tests machinery and ranks nothing: it cannot say that int8 is lossless, that masking helps a real model or
  that the thread settings suit one.
- **No container was built or run.** No image was pulled, built or started, and `docker compose up` never ran (no network and no
  authorisation). What stands in for the STOP gate's "`docker compose up` serves a tiny model" is stated in "Docker" and "The STOP
  gate": the compose file parses, static tests read the Dockerfile and the compose file, and the compose `command` ran natively
  as a process.
- **No fine-tuning of anything.** An export is of a checkpoint OUR launcher wrote and OUR manifest vouches for; an external model is
  never exported or served by D11.
- **Departures from the master plan's file list** (`docker/{Dockerfile.cpu,Dockerfile.gpu,compose.serve.yaml}`, `infer/{placeholders,
  terminology,preprocess,postprocess,batching}.py`, `serve/` with rate limiting): no `Dockerfile.gpu` (nothing about it can be built or
  run here); no `placeholders` or `terminology` module and no `terminology` field in a request (D12 owns them and the request schema
  refuses an unknown key until it does); no `preprocess`, `postprocess` or `batching` module (the steps are inline in the engine and the
  batcher is `serve/batcher.py`); no rate limiting by client, no authentication, no hot reload, no multi-model server (a gateway in
  front is assumed); no `export/model_card.py` or `export/hub.py` (D13).

Tags used below: **measured** (run for this runbook on 2026-11-10, code `4bfaa5e`, or by a gate; the numbers the final wave moved were measured again at `039d1d7`), **spike** (run in the design spike,
on the CPU), **source-read** (read from the source of an installed package, never run), **ARITHMETIC** (a formula applied to values
that were measured), **UNMEASURED** (nothing was run). Every number is from a test, the code, the spike, the ledger of the build
(`.superpowers/sdd/2026-11-09-d11-export-serving/progress.md`) or a measurement made for this runbook ("Measured numbers"); a number
that was not obtained is not given. A TIME moves by a third from one run to the next on this shared machine (load average 3 to 14 on 20
cores while these were taken); a count or a score does not.

What was checked: the unit and property tests (every new test mutation-checked by hand during the build: 92, 61, 62, 159, 106, 132,
about 146, 133, 196 mutants for Tasks 1 to 6 and 8 to 10, the gates' mutants for Tasks 7 and 11, 4 + 9 + 11 + 30 + 10 + 8 + 14 + 23
for the fix wave and 5 + 14 + 1 for the final wave, with the first-pass survivors each given a test or an analysis), the integration modules
(`tests/integration/test_export_parity.py`, `test_engine_surfaces.py`, `test_serving_gate.py`, `test_serving_gate_process.py`,
`test_serving_stop_budget.py`, and D10's `test_distill_tiny_student.py` re-run on the new reader: 112 tests in all), a combined review
by two independent reviewers (Tasks 1 to 7, `17549f8..6ed0e72`, and Tasks 8 to 11, `6ed0e72..b6d57a4`: both "ready with fixes", 0
Critical and 7 Important findings between them), a fix wave of nine commits (`b6d57a4..4bfaa5e`) with the full gate at its end (11768
passed, 1 skipped, 10 deselected, 24 min 1 s, one sequential process; the baseline of D10 was 9750), the real binary for this runbook,
a scoped re-review of the fix wave (`b6d57a4..4bfaa5e`: it needed one small fix, an Important finding and eight Minor ones) and a final
wave of three commits (`330250a..039d1d7`) with the full gate again (11778 passed, 1 skipped, 10 deselected, 26 min 51 s): see "The review, its fix wave and the re-review".

## Architecture

| Module | Job |
|---|---|
| `rinchen/export/record.py` | `ModelRecord`, `CheckpointRecord`, the constants (`RECORD_FILE`, `EXPORT_FILES`, `EXPORT_STAGE`, ...), `ContinuationKind` (which `train.checkpoints` imports): the strict reader and writer of `rinchen.json` |
| `rinchen/export/verified.py` | `VerifiedExport`, `verify_export`, `verified_bytes`, `verified_content`, `json_object`, `shared_vocabulary`, `vocabulary_entries`, `end_token`: the reader every consumer of an export goes through (hash once, before any parse) |
| `rinchen/export/quantize.py` | `QUANTIZATIONS`, `require_quantization`, `effective_compute_type` (tests only), `require_compute_type` (refuses `auto` and blank) |
| `rinchen/export/ct2.py` | `convert`, `ExportFacts`, `ConvertResult`: the private staging directory, the converter, `spm.model`, `rinchen.json`, the manifest last |
| `rinchen/export/eole_ct2.py` | the converter call (D10's), now with `quantization` |
| `rinchen/infer/source_tokens.py` | `ct2_source_tokens`, `reserved_pieces`, `source_symbols`: the rules CTranslate2 does not have, shared with D10's teacher |
| `rinchen/infer/script_mask.py` | `forbidden_scripts`, `allowed_scripts`, `script_hint`, `script_purity`, `suppression_list`, `PURITY_FLOOR` |
| `rinchen/infer/engine_types.py` | `LoadOptions`, `DecodeOptions`, `Row`, `Translation`, `DocumentResult`, `ModelInfo`, `ENGINE_COUNTERS`, `RequestTextError`, `document_status`, and the working types of a call (`Route`, `Job`, `Parsed`, `Outcome`, `Ct2Like`, `ProcessorLike`) |
| `rinchen/infer/engine.py` | `Engine` (`open`, `translate`, `translate_documents`, `warmup`, `require_request`, `info`, `counters`, `close`) |
| `rinchen/infer/translator.py`, `rinchen/__init__.py` | `Translator`, the facade; `rinchen.Translator` is resolved lazily (PEP 562) |
| `rinchen/train/export_parity.py` | `strict_parity`, `ParityReport`, `ParitySource`, `parity_config`, `parse_scored_line`, `require_checkpoint_of_export`: Eole predict against CTranslate2 |
| `rinchen/config/serve.py` | `ServeConfig`, `ServeEnvSettings`, `load_serve_config`: four layers, the last one wins |
| `rinchen/serve/schemas.py`, `metrics.py`, `batcher.py` | the request, response and error shapes; the eleven metric families; the micro-batcher |
| `rinchen/serve/_http.py`, `_translate.py`, `engine.py`, `app.py` | the body limit, the request recorder and the error handlers; the plan, the jobs and `run_jobs`; the public `ServingEngine` Protocol; `create_app` and the lifespan |
| `rinchen/cli/export.py`, `distill_export.py`, `export_parity.py` | `rinchen export convert\|verify\|parity` (`convert` is the same function as `rinchen distill export`) |
| `rinchen/cli/translate.py`, `translate_io.py`, `serve.py` | `rinchen translate` and `rinchen serve` |
| `docker/Dockerfile.cpu`, `docker/compose.serve.yaml`, `.dockerignore`, `configs/serve/{tiny,server,edge}.yaml` | the CPU container recipe and the service settings |
| edits to D5 to D10 | `export/eole_ct2.py` (`quantization`), `train/distill_export.py` (the front end), `train/distill_saved_config.py` (`saved_directions`, `saved_document_level`), `train/distill_teacher.py` (imports, the placeholders unioned back, `auto` refused), `train/distill_generate.py` (imports), `train/checkpoints.py` (one import), `config/languages.py` (the packaged registry), `lang/scripts.py` (`counted_letters`), `cli/main.py` (registration), `pyproject.toml` (the hatch `force-include` only), `tests/unit/test_infer_doc_chunker.py` (the pinned file set of `infer/`) |

Layering (AST-scanned by tests): `export` imports nothing of `train`, `synth`, `cli`, `script_conv`, `infer` or `serve`; `infer`
nothing of `train`, `cli`, `serve`, `synth`, `script_conv` or `annotate`; `serve` nothing of `train`, `cli`, `synth`, `script_conv` or
`annotate` (so the command passes an opener that loads D5's converter); `train` never imports `script_conv` or `cli`. `ctranslate2` and
`sentencepiece` are imported inside `Engine.open` only, and the converter runs as a subprocess.

## The export directory

`rinchen export convert --checkpoint C --spm-model M --tag-placement src|tgt --out-dir O [--quantization Q]` writes, for the averaged
tiny teacher (**measured**; `model.bin` is bit-for-bit the one D10 exported):

| File | Bytes | What it is |
|---|---:|---|
| `model.bin` | 2,034,311 (float32) | the weights as the quantisation stores them (SHA-256 `b7f33d88d5fa...`, the name's last 12 digits) |
| `shared_vocabulary.json` | 8,659 | a list of 344 strings in Eole's order, the padded vocabulary of the checkpoint |
| `config.json` | 223 | the converter's: `eos_token`, `bos_token`, `unk_token`, `add_source_bos/eos` false |
| `spm.model` | 12,614 | a copy of the SentencePiece model the checkpoint was trained with |
| `rinchen.json` | 1,121 | the record (below) |
| `manifest.json` | 3,546 | stage `export`, written last, with the SHA-256 of the five files |

(`spm.model` is 12,605 to 12,614 bytes: the toy model is trained afresh in each workspace.) The converter's log is `<O>.export.log`
beside the directory. int8 and int8_float16 exports differ in `model.bin` (1,488,691 and 844,931 bytes), `rinchen.json` and the manifest
only.

### `rinchen.json`, field by field

UTF-8 JSON, sorted keys, two-space indent, one final LF, `ensure_ascii` off, no timestamp: the same inputs give the same bytes. Schema
`rinchen.model/1`; every field is required and no other is allowed.

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `rinchen.model/1` (D10's `rinchen.distill_export/1` is refused) |
| `name` | string | `[A-Za-z0-9._-]{1,64}`; default `rinchen-` and the first 12 hex digits of the SHA-256 of `model.bin`; `--name` sets it |
| `rinchen_version` | string | the installed package's version at export (`0.0.0` while `pyproject.toml` says so; `unknown` without package metadata) |
| `tag_placement` | `src` or `tgt` | where the direction tag goes: in the source, or as the forced prefix of the target |
| `directions` | sorted list of `<src>-<tgt>` | the directions the model was trained for, derived from the saved config's corpus names (`--direction` only when it names none; a statement must equal them) |
| `languages` | sorted list | the union of the codes of `directions` |
| `tags` | map | for each TARGET code of `directions`, its direction tag (`<2khk_Cyrl>`) |
| `document_level` | bool | the model decodes `<doc>` windows; derived from the saved config (a corpus whose `src_prefix` holds `<doc>`), else `--document-level`, else false |
| `quantization` | one of `QUANTIZATIONS` | what the converter was asked for |
| `checkpoint` | object | D10's `CheckpointRecord` unchanged: `kind` (`step` or `averaged`), `step` (number or null), `weights_sha256`, `config_sha256`, `vocab_sha256`, `manifest_sha256`, `manifest_stage` |
| `spm_sha256` | 64 hex | the SentencePiece model beside the record |
| `encoder_layers`, `decoder_layers`, `hidden_size`, `vocabulary_entries` | positive integers | read from the checkpoint |

A worked example (the float32 export of the averaged tiny teacher, digests cut to 12 characters):

```
{
  "checkpoint": {
    "config_sha256": "7290e2b4852a...", "kind": "averaged", "manifest_sha256": "bdc956d1aa34...",
    "manifest_stage": "train_average", "step": null, "vocab_sha256": "9d444b846bc2...",
    "weights_sha256": "f4e32704b958..."
  },
  "decoder_layers": 2,
  "directions": ["eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn", "khk_Cyrl-rus_Cyrl", "rus_Cyrl-khk_Cyrl"],
  "document_level": false,
  "encoder_layers": 2,
  "hidden_size": 64,
  "languages": ["eng_Latn", "khk_Cyrl", "rus_Cyrl"],
  "name": "rinchen-b7f33d88d5fa",
  "quantization": "float32",
  "rinchen_version": "0.0.0",
  "schema": "rinchen.model/1",
  "spm_sha256": "360687814955...",
  "tag_placement": "src",
  "tags": {"eng_Latn": "<2eng_Latn>", "khk_Cyrl": "<2khk_Cyrl>", "rus_Cyrl": "<2rus_Cyrl>"},
  "vocabulary_entries": 344
}
```

(one key per line in the file). Why `directions` and `tags` are in the record: the vocabulary does not say which languages a model was
trained for. Every vocabulary holds all 118 reserved symbols since D9 (the toy teacher has a `<2mn_Mong>` entry and never saw the
language), so "the tag is in the vocabulary" proves nothing about a trained language; the saved `config.json` of a checkpoint our
launcher wrote does (`eng_Latn-khk_Cyrl.auth`, `...bt`, `...kd`, `valid` corpus names). `ModelRecord.from_bytes` is as strict as the
writer is exact: this schema, every field and no other, each of its type, positive counts, 64-hex digests, kind and step agreeing, a
repeated key refused, `languages` the union of `directions`, `tags` one entry per target, `directions` sorted, unique and made of two
distinct registry codes; a refusal is a `DataError` that names the field and never quotes what it held.

### The manifest and the provenance chain

Stage `export`, status `export: ok`, `tag_placement`, `quantization`, `checkpoint_kind`, `tag_placement_check` (`checked` or `stated`),
`directions_check` and `document_level_check` (`derived` or `stated`); metrics `encoder_layers`, `decoder_layers`, `hidden_size`,
`vocabulary_entries`, `model_bytes`; seed -1; `tool_versions` `{"ctranslate2": "4.8.4", "eole": "0.6.4", "torch": "2.14.1+cu130"}`
(measured; `torch` because the converter loads it through Eole); inputs the checkpoint's weights, `config.json`, `vocab.json` and the
SentencePiece model with the SHA-256 the verification took before it read them (the command never hashes them again); outputs the five
files named under the final directory with the SHA-256 they have now. The chain: the checkpoint's `train_launch` or `train_average`
manifest (which, for a D9 continuation or a D10 student, names its own sources) -> `rinchen.json.checkpoint` -> the export manifest
-> `Engine.info` (the SHA-256 of `model.bin` and of the manifest) -> the `serve_start` log line and every parity or evaluation manifest
that names the export. These are guards against mistakes, not against someone who writes a manifest by hand.

### `verify_export`, hash once

Everything that trusts an export (the engine, the commands that name one, the parity, the evaluations) goes through
`verify_export(directory)`. The check of the old format comes FIRST, before anything is hashed. It then requires: a manifest of
stage `export` with status `export: ok` listing each of the five files once by name; the SHA-256 of each file, taken once and BEFORE
the file is parsed, equal to the manifest's; the record's `spm_sha256` equal to the SentencePiece model beside it; and the record's
`vocabulary_entries` equal to the length of `shared_vocabulary.json` (when it is a list of strings); and every `tags` entry still the
registry's tag of that language (a `ConfigError`: the registry changed since the export, re-export). A directory that was moved is
accepted (names are compared, not paths). A reader that opens a file afterwards (`verified_bytes`) reads its bytes once and refuses
them unless they are the bytes that were hashed. **The verification is not atomic with CTranslate2's later read of `model.bin`**: it
guards against a stale or swapped directory, not against a writer that swaps a file after the hash (and the command and the engine each
hash the directory: `rinchen serve` hashes the export twice, M5 of the review, left).

### The staging layout and a crash

The converter refuses a directory that exists, so `convert` never lets it write into `--out-dir`. It makes a private directory
`.<name>.<random>` BESIDE `--out-dir`, lets the converter write into a not-yet-existing child `export`, requires the three files and
that the vocabulary has the number of entries the record states, writes `spm.model` and `rinchen.json`, hashes the five files once,
writes `manifest.json` LAST into the child, and renames the child onto `--out-dir` after one more existence check: the output
appears whole, manifest included, or not at all. Any failure (including an interrupt) removes the private directory, and `--out-dir`
only if this call put it there; a directory someone else makes at that path meanwhile is refused and never touched. A hard kill
(SIGKILL, out-of-memory kill) leaves the hidden directory, as large as the model; it is safe to delete when no export is running and
nothing sweeps it.

### The D10 migration

`rinchen.json` REPLACES D10's `export.json` and stage `export` replaces `distill_export`; there is no second reader. A directory that
holds `export.json` and no `rinchen.json` is refused by every reader, with the remedy (**measured**, exit 2, before anything is
hashed):

```
error: cannot use the export $W/old_format: it holds export.json, the record of the format of the distillation (D10), and no
rinchen.json: export the checkpoint again with `rinchen export convert` (a few seconds for a small model)
```

`rinchen distill export` is the same function as `rinchen export convert`, so the commands of runbook 10 keep working. The KD
fingerprint recipe is unchanged (the diff of `distill_generate.py` is two imports); its VALUE for a workspace moves only with the
workspace's `spm.model`, which differs between workspaces (a dry run for this runbook printed `18a315d3707e...` where runbook 10 shows
`5c6b165b2906...`, for another workspace's SentencePiece model). D10's gate (44 tests) was re-run on the new reader: 3,599 rows, 3,583
matches, chrF 98.2 and 98.58, 200 of 200, `model.bin` 2,034,311 and 1,835,438 bytes, all unchanged.

## Quantisation

Quantisation is applied AT CONVERSION and recorded in `rinchen.json`: `QUANTIZATIONS = float32, float16, int8, int8_float16`.
`float32` is the parity reference and the default, `int8` the CPU product, `float16` and `int8_float16` the GPU products (converted
here, **never run on a GPU: UNMEASURED**). The converter accepts four more (`int8_float32`, `int8_bfloat16`, `int16`, `bfloat16`); they
are not offered, one tuple entry each adds them. At load the engine passes `compute_type="default"` (what the file stores); the device
falls back with CTranslate2's own warning when it cannot run the stored type. **`compute_type="auto"` is refused** (`D-CT2-AUTO-COMPUTE`):
on this CPU it loads EVERY export, a float32 one included, as `int8_float32` (chrF 98.27 against 98.21 for the same weights, spike), so a
parity or quality check that used it would measure int8 and call it float32; `device="auto"` would choose the GPU whenever one is visible.

The tiny teacher at the three offered CPU-relevant quantisations (**measured**, 2026-11-10; the chrF is the engine's greedy translation
of the 400 dev rows, the gate's number):

| `--quantization` | `model.bin` (bytes, share) | `Engine.info.compute_type` on the CPU | chrF | `export convert` |
|---|---:|---|---:|---:|
| `float32` | 2,034,311 (100 %) | `float32` | 98.207 | 4.8 s |
| `int8` | 1,488,691 (73.2 %) | `int8_float32` | 98.268 (+0.061) | 5.3 s |
| `int8_float16` | 844,931 (41.5 %) | `int8_float32` | 98.197 (-0.010) | 5.4 s |

The gate: int8 chrF at least float32's minus 1.0 (passes with 1.06 to spare); the toy saturates at 98.2, so the figure says nothing
about a real model's int8 loss (**UNMEASURED**). An explicit `--compute-type int8_float16` on a CPU is a `ConfigError`
(`CTranslate2 cannot use device 'cpu' with compute type 'int8_float16': Requested int8_float16 compute type, but the target device or
backend do not support efficient int8_float16 computation.`), `auto` another; `float16` and `int8_float16` as stored types load on this
CPU with a warning on stderr and run as `float32` and `int8_float32`.

The spike converted the same checkpoint to all nine settings (**spike**, `model.bin` bytes, and what `compute_type="default"` loads
each as on this CPU, an Intel Core Ultra 7 265K with AVX2 and AVX-VNNI and no AVX-512): none or `float32` 2,034,311 (float32);
`int8` and `int8_float32` 1,488,691 (int8_float32); `int8_float16` and `int8_bfloat16` 844,931 (int8_float32, with a warning);
`int16` 1,663,955 (int16); `float16` and `bfloat16` 1,018,839 (float32, with a warning); an explicit `int8_float16`, `float16` or
`bfloat16` on the CPU is a `ValueError` for every one. The int8 file is 73 % of float32 on the toy because the 344-entry embedding is a
large share of it; a model with real transformer compute is much smaller (below).

**Speed, a mid-size model (spike, UNMEASURED for a real model).** A 6/6 d512 model with a toy vocabulary, trained from random
initialisation for 20 steps (the weights are noise; the speed is the shape's): `model.bin` 187,245,795 bytes at float32 and 54,872,527
at int8 (29 %); fixed 30 decoding steps, beam 4, 256 toy sentences, `translate_batch` from `2 x inter_threads` Python threads.
Sentences per second (p50 call latency in ms), on a machine at load average about 12 (a ratio is good to a factor of about 1.3):

| export | inter | intra | 1 sentence per call | 16 per call |
|---|---:|---:|---:|---:|
| float32 | 1 | 1 | 3.2 (573) | 19.4 (1,664) |
| float32 | 4 | 1 | 14.8 (542) | 69.4 (1,848) |
| float32 | 4 | 2 | 23.5 (334) | 102.8 (1,212) |
| float32 | 8 | 1 | 34.3 (465) | 123.3 (2,036) |
| int8 | 1 | 1 | 32.9 (60) | 44.2 (729) |
| int8 | 1 | 4 | 58.1 (34) | 113.7 (282) |
| int8 | 4 | 1 | 79.8 (100) | 129.0 (979) |
| int8 | 4 | 2 | 110.2 (72) | 192.9 (622) |
| int8 | 8 | 1 | 121.2 (132) | 182.5 (1,301) |

int8 was 10 times faster than float32 for one sentence per call on one thread (32.9 against 3.2) and 2.3 times at 16 per call; the
CPU speed of a quantised export is the reason to ship one, and its quality cost on a real model is the open question.

## `rinchen export convert`, `verify`, `parity`

The tiny flow, run once for this runbook through the real `rinchen` binary on the CPU with the GPU hidden (**measured**, 2026-11-10).
`$W` stands for the workspace of the run (the gates' helpers build the same one under a pytest temporary directory). The teachers are
D10's: the tag in the source (`teacher_tiny.yaml`, 2,000 steps, `train launch` 84.7 s, `train average --last 2` 1.5 s; averaged weights
SHA-256 `f4e32704b958...`, the same as D10's) and the tag as the forced target prefix (1,000 steps, 42.8 s).

**`convert`** (`rinchen distill export` is the same function; 4.8 s):

```
rinchen export convert --checkpoint $W/ws/teacher_out/variants/t0.5_bt0.5_src_s1234/run/averaged \
  --spm-model $W/ws/in/spm/toy.model --tag-placement src --quantization float32 --out-dir $W/ws/exports/src_float32
exported to $W/ws/exports/src_float32
tag placement: checked against the saved config
directions: derived from the saved config (4)
document level: derived from the saved config (false)
quantization: float32
record: $W/ws/exports/src_float32/rinchen.json
manifest: $W/ws/exports/src_float32/manifest.json
```

The second line is `tag placement: stated, not recorded in the checkpoint` when the saved config holds no placement to check against, the
third and fourth `directions: stated (<n>)` and `document level: stated (true|false)` when it names no directions or corpus. Options:
`--checkpoint` (a final `step_<N>` or an averaged directory with the manifest our commands wrote), `--spm-model` (its SHA-256 must be that
of the copy Eole saved), `--tag-placement` (no default; a placement the saved config contradicts is refused before anything is made),
`--out-dir` (must not exist), `--quantization` (default `float32`), `--direction` (repeatable, `<src>-<tgt>`), `--document-level` /
`--no-document-level`, `--name`, `--timeout`. Exit codes: 0; 2 for a bad option, a checkpoint our manifest does not vouch for, a
SentencePiece model that is not the checkpoint's, a placement, direction set or document level the saved config contradicts, directions
nobody states or derives, an output directory that exists or holds `$` or a glob character; 3 for a checkpoint or manifest that cannot be
read or whose `config.json` or `vocab.json` lacks what an export records; 4 when the converter fails, leaves a file out or writes a
vocabulary of another size than the checkpoint's; 1 when the output cannot be written; 130 when interrupted.

**`verify`** reads an export back, changes nothing, and exits 0 only when `verify_export` does (1.4 s):

```
rinchen export verify $W/ws/exports/src_float32
export: $W/ws/exports/src_float32
name: rinchen-b7f33d88d5fa
quantization: float32
tag placement: src
directions (4): eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn, khk_Cyrl-rus_Cyrl, rus_Cyrl-khk_Cyrl
document level: false
model sha256 b7f33d88d5fa
manifest sha256 a1ed0c5bdc5a
```

Exit 2 (nothing on stdout, the reason on stderr) for a directory `verify_export` refuses: the old format above, or one flipped byte
(**measured**, `model.bin`):

```
error: cannot use the export $W/flipped: $W/flipped/model.bin differs from the manifest $W/flipped/manifest.json: its SHA-256 is
2a2a56abdc08... and the manifest lists b7f33d88d5fa...; the file changed after the manifest vouched for it
```

Exit 3 for files that cannot be read or a manifest or record that is not what it must be.

**`parity`** (5.3 to 5.4 s for 200 sentences; "Strict parity" below):

```
rinchen export parity --checkpoint $W/.../run/averaged --export $W/ws/exports/src_float32 --corpus $W/ws/in/toy.parquet \
  --out-dir $W/parity_src_float32 --max-lines 50 --threads 2 --timeout 600
parity: 200/200 exact, mean |delta| 7.19e-07 per token (gated) pass
result: $W/parity_src_float32/parity.json
manifest: $W/parity_src_float32/manifest.json
```

Options: `--checkpoint`, `--export`, `--corpus` (repeatable), `--out-dir`, `--max-lines` (default 50 per direction), `--threads` (default 2),
`--timeout`. The line reads `(not gated: quantized)` for a quantised export. `parity.json` (sorted keys, two-space indent, LF) holds
`sentences`, `exact`, `exact_share`, `mean_abs_delta`, `max_abs_delta`, `quantization`, `placement`, `gated`, `passes`, the thresholds
(`min_exact_share`, `max_mean_delta`), `lines` (per direction), `max_lines`, `mismatches` (the first ten, as token lines), `export_name`,
`model_sha256` and `checkpoint_weights_sha256`. The manifest (stage `export_parity`) names every input with the digest taken before it
was read (the corpus files, the five files of the export and its manifest, the checkpoint's weights, `config.json` and `vocab.json`) and
is written last; the old manifest and result are removed before the run, so a failed run leaves none. Exit codes: 0 when the parity
passes; 4 when it does not (a measured result: everything is written and printed first) or when Eole or CTranslate2 fails; 2 for a bad
option, an export that is not vouched for, a checkpoint that is not the export's, a path Eole reads specially, an output directory that
is an input's; 3 for a missing or unreadable corpus, a direction of the model without dev rows, another stage's manifest in the output
directory; 1 when the output cannot be written; 130 when interrupted. A checkpoint of other weights than the export's (**measured**,
exit 2 in 1.2 s, nothing run):

```
error: the checkpoint's weights have SHA-256 e2410a0f7b0d... but the export rinchen-b7f33d88d5fa was made from weights with
f4e32704b958...: a parity of two different models says nothing
```

## Strict parity

**Method.** 200 dev sentences (the first 50 by id of each of the four directions; no test row is read) are tokenised with the export's
SentencePiece model and `ct2_source_tokens` (the tag goes where the placement says). `eole predict` runs once, greedy
(`beam_size: 1`, `transforms: []`, `max_length: 128`, `max_length_ratio: 0`, `with_score: true`, `length_penalty: none`,
`gpu_ranks: []`, `CUDA_VISIBLE_DEVICES=""`); CTranslate2 translates the same sources greedily on the CPU with the export's own compute type
(`default`, never `auto`). `exact` compares the pieces (without the end token and, for `tgt`, without the forced tag, which both start
with). For the log-probabilities, `score_batch` scores Eole's own pieces and the delta of a sentence is `|sum(log_probs) - pred_score| /
len(log_probs)`; the report holds the mean and the maximum over the sentences.

**The two traps** (both pinned by tests, both found in the spike):

1. Eole's `length_penalty` defaults to `avg`, which divides the score by the length; with `none` the `pred_score` is the sum of the
   log-probabilities of the generated pieces, the end token included (`D-EOLE-SCORE-LENGTH-PENALTY`). `score_batch` APPENDS the end
   token itself, so its token list has one more entry than Eole's line.
2. For the placement `tgt`, Eole's score EXCLUDES the forced prefix token while `score_batch` counts the tag's own log-probability
   (about -0.09): the library drops `log_probs[0]` (`D-EOLE-SCORE-FORCED-PREFIX`). The delta of a `tgt` sentence is divided by the number
   of log-probabilities compared after that drop.

**The numbers** (**measured**, 2026-11-10, 200 sentences each; the spike's, on its own training run, in parentheses):

| export | exact | mean abs delta per token | max | gated | result |
|---|---:|---:|---:|---|---|
| source tag, float32 | 200 / 200 | 7.19e-7 (8.28e-7) | 3.43e-6 (3.48e-6) | yes | pass |
| target tag, float32 | 200 / 200 | 1.37e-6 | 5.05e-6 | yes | pass |
| target tag, float32, the forced tag's log-probability KEPT | 200 / 200 | 0.0433 | - | yes | FAIL |
| source tag, int8 | 199 / 200 | 1.14e-3 (1.45e-3) | 1.60e-2 (2.81e-2) | no | pass |
| source tag, int8_float16 (spike) | 199 / 200 | 1.69e-3 | 3.59e-2 | no | - |

The third row is the trap the library avoids, run through a wrapper that pads the scores: the translations are still 200 of 200 exact,
so the exactness cannot reveal it, only the log-probabilities do. **The 1e-3 bound is a float32 bound** (`D-STRICT-PARITY`): int8 and
int8_float16 exceed it on the same checkpoint, so a quantised export is held to the exact share (0.95) alone, reports its delta, and is
gated by chrF (drop at most 1.0). The thresholds are the master plan's and were never lowered. What the numbers show: CTranslate2
reproduces Eole's greedy translations exactly and its log-probabilities to six places for two tiny models in float32. What they do not
show: the margin for a real model (the toy's 1e-6 says nothing about a 1.1 B-parameter one), whether a quantised real model keeps float32's
greedy output (199 of 200 here), or anything on a GPU. A greedy hypothesis cut at `max_length` 128 has no end token, so the end token
`score_batch` appends adds a spurious delta for that sentence (none occurred on the toy).

The checkpoint is bound to the export: the weights digest, and since the fix wave the digests of `config.json` and `vocab.json` as well,
must equal the record's (exit 2 before anything runs).

## The engine

`Engine.open(directory, LoadOptions(...), script_converter=None)` has `verify_export` vouch for the directory, reads `shared_vocabulary.json`,
`config.json` (`eos_token`, no default) and `spm.model` through the verified bytes, checks EVERY tag of the record (an entry of the
vocabulary and exactly one SentencePiece piece: a model that cannot be asked for its own languages is refused at load, not at the first
request) and only then loads `ctranslate2.Translator`. `LoadOptions(device="cpu", compute_type="default", inter_threads=1,
intra_threads=1, max_queued_batches=-1)` refuse `device="auto"` and `compute_type="auto"` (and a blank one), a count that is not an
integer of at least 1 (a bool is not one). `DecodeOptions(beam_size=4, n=1, max_decoding_length=256, max_source_pieces=256,
max_batch_size=32, script_mask="retry")` take strict integers of at least 1, `n <= beam_size` and `max_source_pieces <= 1023` (the tag
takes one of CTranslate2's 1,024 source places; the same limit D10's teacher keeps). `Engine.info` is a `ModelInfo` (name, quantization,
device, effective compute type, placement, directions, targets, document level, `model_sha256`, `manifest_sha256`); `Engine.counters()` a
copy of the cumulative counters; `Engine.warmup()` one fixed translation into the first target in the default script mode and the build
of every suppression list a call can ask for; `Engine.require_request(src, tgt, script=None, *, documents=False)` raises what `translate`
would raise for the languages, before any text is looked at (so that a command can refuse a typo for an empty input and before it opens an
output file); `Engine.close()` unloads (idempotent; it does not wait for calls in flight: the service drains first).

**The pipeline per row**, in this order: (1) check the language request (a registry source, a target the record has a tag for, a script only
for a Mongolian target, the same source and target refused) and that the text is encodable (a lone surrogate is a `DataError` naming the row);
(2) normalise (`text.normalize.normalize(text, src or hint)`, where `hint` is `mn_Mong` when the text holds a letter of the Mongolian
script); (3) a text with no letter or digit, or no piece, is `empty` and not sent, and a piece that is a reserved symbol is refused (below);
(4) more than `max_source_pieces` pieces is `too_long` and not sent (CTranslate2 would cut a source of 1,024 pieces without a sign); (5) the source by placement (`ct2_source_tokens`); (6) ONE
`translate_batch` per distinct suppression list with `beam_size`, `num_hypotheses=n`, `max_decoding_length`, `max_batch_size`,
`return_end_token=True`, `disable_unk=True`, `suppress_sequences` and, for `tgt`, `target_prefix`; (7) per hypothesis: strip the forced
prefix once (a hypothesis that does not start with it is an `UpstreamError` naming the row, never the text), note a missing end token
(`truncated`), cut a document export's hypothesis at its first `<sep>` (below), drop (counted `reserved_symbol`) a hypothesis holding a
piece no translation holds (the unknown piece, the end token inside, a reserved symbol the model must not write, any vocabulary entry that
is no SentencePiece piece), decode, turn a line-break character into a space (counted `line_break`: the command's one-line contract),
strip; (8) the first surviving hypothesis is `text`, the others are `alternatives` (distinct, in order); (9) script purity and the retry
(next section); (10) the script conversion of a `Mong` request for a model with no `mn_Mong` target.

**Statuses**: `ok`; `empty` (nothing to translate); `too_long` (not sent); `truncated` (the best hypothesis was cut before its end token:
it KEEPS the partial text, which the service and the command say; D10's KD drops it); `no_output` (every hypothesis was dropped, the text
is empty). A request is all-or-nothing in its errors and a text-level problem is a status.

**The 16 counters** (cumulative since `open`; one update per call under one lock; a failed call adds nothing): `sources` (rows, plus
the sentences decoded alone inside a document call), `empty`, `too_long`, `truncated`, `no_output`, `reserved_symbol` (hypotheses
dropped), `line_break`, `zero_shot` (a translation whose pair is not in `directions`), `script_retries`, `script_retry_improved`,
`script_impure` (a final output below the purity floor), `script_conversions`, `converter_rule_fallbacks`, `converter_invalid_outputs`
(the converter's own), `windows` (window decodes of a document call) and `window_fallbacks`.

**Directions and languages.** A target must be a key of `rinchen.json.tags`; any other registry code is a `ConfigError` that names the
model's targets (`the model does not translate into 'xxx_Latn': its targets are eng_Latn, khk_Cyrl, rus_Cyrl`). `src` is a hint (the tag
names only the target): `None` means unknown (the script hint decides the normalisation and the result says `und` or `mn_Mong`). A pair
that is not in `directions` is allowed and counted `zero_shot`, never judged: eng_Latn to rus_Cyrl through the tiny teacher, which never
saw the pair, returns a 200 whose text is not a Russian translation (Khalkha words for one sentence, a nonsense word for another). `script` applies to a Mongolian target only (`khk_Cyrl` or `mn_Mong`): `Cyrl` selects
`khk_Cyrl`, `Mong` selects `mn_Mong`; when the selected target is not a target of the model, a script converter (D5's `ScriptConverter`
fits, as a Protocol: `infer` imports nothing of `script_conv`) attached and a `Mong` request make the engine translate into `khk_Cyrl`
and convert the final text word by word (counters `script_conversions`, ...; D5's limits stand), otherwise `ConfigError`.

**Reserved symbols in the text of a request are refused.** SentencePiece keeps each reserved symbol as one piece wherever it stands, so a
text that spells `<2eng_Latn>` asks the model for another language, and `<sep>` or `<doc>` steer a document model. The 18 symbols of the
project that are not numbered placeholders (the 10 direction tags, `<bt>`, `<ft>`, `<t>`, `</t>`, `<r>`, `</r>`, `<doc>`, `<sep>`) are
refused with a `RequestTextError` (a `DataError`) that names `row N` or `document N` and the SYMBOL, never the text, before anything is sent
or counted (**measured**; exit 3 of `rinchen translate`, HTTP 400 of the service):

```
row 1 contains the reserved symbol <2eng_Latn>, which the model reads as a control piece and not as text (a direction tag would change
the language asked for): remove it
```

Before the fix a row `<2eng_Latn> This is the star.` asked of `khk_Cyrl` gave `ok` and `Машин и стол.` with purity 1.0: the language had
silently changed. The numbered placeholders (`⟦0⟧` to `⟦99⟧`) are text: D12's engine is meant to write them. The windows the engine
assembles itself carry `<sep>` and `<doc>` and are not checked.

**Errors.** `ConfigError` (a request that cannot be honoured), `DataError` (a text the engine refuses: a lone surrogate, a reserved
symbol), `UpstreamError` (the library fails or answers wrongly; it names the exception type or a row, never a text), `TypeError` for a
row that is not a `Row`. No message or log names a text; a test scans `infer/` and `serve/` for every message that interpolates a value
and fails on an unreviewed one.

### Documents

`Engine.translate_documents(documents, src, tgt, options=None, *, script=None)` needs a record with `document_level` true and the placement
`src` (D9 built `src` windows only) and `src` given: it cuts each document into sentences (`split.rules.split`), runs D9's
`translate_document` with two callables, and answers a `DocumentResult(text, translations, stats)` with one `Translation` per sentence. A
window is ONE CTranslate2 call whose source is `[tag, "<doc>", *pieces of the joined window]`, decoded with `<sep>` allowed and the end
token asked for; a window whose hypothesis lacks the end token returns `""`, which makes the chunker fall back to its sentences as a whole
(counted `window_fallbacks`). The sentence callable is the ordinary row path. A document's translations are joined by one space (none for
`zho_Hans` and `jpn_Jpan` targets); a document's status is its worst sentence's; `max_source_pieces` is also the chunker's window budget
(the gate uses 126, what `docify` cuts to).

**A hypothesis of a document export is cut at its first `<sep>`.** A document model is continued on windows of 2 to 4 sentences only
(`docify` cuts none of size 1), so it writes `<sep>` after a LONE sentence, because every training target ended its sentences with it.
Every sentence decoded alone (a single the windows left over, an oversize sentence, a sentence of a window that fell back, a plain row on
a document export) therefore came back as `x <sep>`, and the engine dropped it as a reserved piece. The first gate recorded the hole
instead of failing on it and the review found how large it is. **Before and after** (**measured** on the trees before, `b6d57a4`, and after,
`4bfaa5e`, the fix; 400 toy test sentences in 66 documents; "default" is `DecodeOptions()`, "gate" is beam 1, 128 pieces, windows of at
most 126 source pieces):

| model, options | what | before the fix | after the fix |
|---|---|---|---|
| the spike's document model, default | 400 plain rows | 400 `no_output`, chrF 0.0 | 400 `ok`, chrF 93.861 |
| the spike's document model, default | 66 documents | 382 `ok` + 18 `no_output`, chrF 88.465 | 400 `ok`, chrF 92.061 |
| the gate's own document model, gate | 400 plain rows | 400 `no_output`, chrF 0.0 | 400 `ok`, chrF 88.419 (88.494 after the final wave, below) |
| the gate's own document model, gate | 66 documents | 364 `ok` + 36 `no_output`, chrF 78.549 | 400 `ok`, chrF 84.747 |

The fix (`966387a`): for an export whose record says `document_level` the engine cuts the hypothesis at its first `<sep>` and reads the
head as the translation, finished (the separator is the model saying so); a hypothesis that STARTS with `<sep>` is still `no_output`, a
sentence-level export still drops `x <sep>`, and a window decode lets `<sep>` through as before. Suppressing `<sep>` at decode time was
refused: the runner-up token would be invented text. **How this relates to D9's chunker contract** ("a single goes to the sentence
translator, never to the window model, which was trained on windows of at least two"): a document export has no second, sentence-level
model, so the sentence callable decodes the SAME document model on a lone sentence and cuts at `<sep>`; the quality of that decode on real
text is **UNMEASURED**, and D9's contract is kept in its structure (singles still never form a window), not in the model that decodes
them. The end-token fallback closes the two items D9 left open for D11: a cut inside the last segment is detected by the missing end token
(94 of the 106 windows fell back at a 20-piece cap, and each of the 334 windows of two sentences decoded two pieces below its own length
fell back although it still had one segment per sentence), and a `translate_sentence` result cannot hold `<sep>` (the engine cuts or
drops it).

**The list of a lone sentence lets `<sep>` through too (`330250a`, found by the re-review).** The fix above cut the hypothesis at `<sep>`, but the
suppression list of such a sentence still held `<sep>` in the script mode `always` and in the second pass of `retry` (only a window's list
left it out). With `<sep>` suppressed the model does not stop: it writes the runner-up token and goes on, inventing the sentences that follow,
and the status stays `ok`. **Measured** on the gate's own document model, the 400 toy test sentences as plain rows, beam 1, 128 pieces (the
re-review's `p_modes.py`, run again for this record):

| mode | before `330250a` | after `330250a` |
|---|---|---|
| `off` | 400 `ok`, chrF 88.23 | 400 `ok`, chrF 88.23 |
| `retry` (the default) | 400 `ok`, chrF 88.42; 5 retries, every replaced text carries an invented trailing sentence (`The day is the dog. . Where is the door.`) | 400 `ok`, chrF 88.49; the 5 replaced texts are one sentence each (`The day is the dog.`) |
| `always` | 400 `ok`, chrF 81.80; 358 of 400 texts differ from `off` (`Цай, гар ба ном.` became `Цай, гар ба ном. П, ном ба ном.`) | 400 `ok`, chrF 88.49; 5 of 400 texts differ from `off` |

Now every call of a document export lets `<sep>` through in its list (`Job.lets_separator_through`: a window, or a sentence the engine cuts); the
list of a sentence export still suppresses it, and `Engine.warmup` builds both kinds of list for a document model (a test counts the lists built
after the warm-up in all three modes: none). The list key and the allow flag are read from the one answer (`_suppression` takes the flag from the
key), so the two cannot disagree again. Through `translate_documents` with the gate's options the document numbers do not move (66 documents,
400 `ok`, chrF 84.747, 101 of 106 windows aligned, 18 singles; the spike's model, default options: plain rows 93.861, documents 92.061); only
the plain rows of the gate's model moved, 88.419 to 88.494, because under the default mode `retry` the 5 impure rows are decoded again. A gate
module test (`test_plain_rows_of_a_document_model_do_not_run_on_in_any_script_mode`) now runs the three modes on the real document model:
at most 20 of the 400 texts may differ from the unmasked ones (5 measured; 358 with the defect) and the chrF may drop by 1.0 at most (+0.26
measured; -6.4 with the defect). A hypothesis that STARTS with `<sep>` has an empty head, is dropped as an empty text and is counted
`reserved_symbol` (before, nothing counted it); the row stays `no_output`.

On the gate's own document model (the tiny teacher continued for 2,500 steps on windows made by `rinchen train docify`: 1,194 windows,
`docify` 1.5 s, the continuation 104 s, the export 3.5 s, `document_level: derived from the saved config (true)`; **measured**): 66 test
documents, 400 sentences, one translation per sentence, 106 windows of which 101 line up (95.3 %), 18 singles; per size, windows of 2
sentences line up 325 of 334 (97.3 %), of 3 260 of 268 (97.0 %), of 4 191 of 202 (94.6 %); 5 windows fell back; at `max_decoding_length`
20, 94 of the 106 windows fall back. The throughput of a document request is **UNMEASURED** (one CTranslate2 call per window,
sequential per window), and joiners beyond a space and the `tgt` placement are not built.

### `rinchen.Translator`

```
import rinchen
with rinchen.Translator("models/tiny", device="cpu") as translator:      # verifies the export and loads the model
    result = translator.translate("This is the star.", tgt="khk_Cyrl", src="eng_Latn")      # one Translation
    results = translator.translate(["a", "b"], tgt="khk_Cyrl", n=2)                           # a list of Translation
    document = translator.translate_document("First. Second.", tgt="khk_Cyrl", src="eng_Latn")  # a document export only
```

`Translator(directory, *, device="cpu", compute_type="default", inter_threads=1, intra_threads=1, script_converter=None)`;
`translate(texts, *, tgt, src=None, script=None, beam_size=4, n=1, max_decoding_length=256, max_source_pieces=256, script_mask="retry")`
(a `str` gives one `Translation`, a sequence a list), `translate_document(document, *, tgt, src, ...)` (a string or a sequence of lines),
`info`, `counters()`, `close()` and the context-manager protocol. It is thread-safe as the engine is. A `Translation` is `(text, status,
alternatives, src, tgt, script_purity, retried, zero_shot)` (**measured**: `Translation(text='Энэ бол од.', status='ok', alternatives=(),
src='eng_Latn', tgt='khk_Cyrl', script_purity=1.0, retried=False, zero_shot=False)`). `rinchen.Translator` equals the engine's output on 50
dev sentences and is resolved on first use: `import rinchen` loads neither the engine, `ctranslate2`, `sentencepiece` nor `torch`.

### Threads

`Engine` holds one `ctranslate2.Translator` with `inter_threads` replicas; `translate` may be called from any number of threads (the library
releases the GIL; SentencePiece encode and decode are thread-safe). The only mutable state is the counters (one lock; a reader of the
counters never sees half a call: 1,200 empty sentences by 8 workers with a switch interval of 1 microsecond gave 723,761 readings, none
torn) and the suppression-list cache (each list built once under a lock); calls into D5's converter take a lock of their own. The service
runs `Engine.translate` on `inter_threads` worker threads. `inter_threads` x `intra_threads` must not exceed the cores given to the process.
The thread tables are under "The service".

## Script masking

A model of ours can emit a piece of the wrong script (a Latin piece in Cyrillic output; the D5 converter, which WANTS to write the
traditional script, in a Cyrillic one). `infer/script_mask.py`:

- **Policy by registry.** The forbidden scripts of a target are every member of `rinchen.lang.scripts.Script` that `configs/languages.yaml`
  does not give to it: for `khk_Cyrl` the Mongolian script, Latin, Han, Hangul, Hiragana and Katakana; for `mn_Mong` Cyrillic and the rest. The
  policy for the other eight languages is the registry's and uncalibrated. A letter of a script outside the enum (Greek) is nobody's
  forbidden set and counts against a text's purity all the same.
- **`script_purity(text, scripts)`** is the share of the letters of the text that are in the scripts of its target (1.0 for a text with no
  letter). It is reported on every `Translation`; a final output below `PURITY_FLOOR = 0.98` (a prior, never calibrated) counts
  `script_impure`.
- **`suppression_list(vocabulary, processor, tgt, *, end_token, allow=(), scripts=True)`**: the entries of `shared_vocabulary.json` that are
  no SentencePiece piece (Eole's `<blank>`, `<s>` and the padding tokens), the unknown and control pieces, the reserved symbols the model
  must not write, the pieces with a letter of a forbidden script (the piece with U+2581 read as a space) and the byte-fallback pieces of ASCII
  letters when Latin is forbidden (67 entries for a Cyrillic target of the toy, 194 for a Latin one; 42 for the D5 converter's). It never
  holds `</s>` (a model that cannot end never stops) and holds only vocabulary entries (CTranslate2 refuses any other). `scripts=False`
  keeps the control entries only. A list does not touch a forced prefix.
- **Modes** (`DecodeOptions.script_mask`): `always` decodes every row with its list; `retry` (the default, the master plan's post-check)
  decodes without a script list and decodes again, with the list, only a row whose output has letters and a purity below the floor (counted
  `script_retries`; `script_retry_improved` when the second purity is higher, and only then does the second output replace the first); `off`
  never suppresses a script (the control entries are always suppressed: they are never text). A `no_output` row is not retried. For a
  document export every list lets `<sep>` through, a lone sentence's as a window's ("Documents"): the model writes `<sep>` after a lone
  sentence, and with it suppressed the model goes on and invents the sentences that follow.
- **Cost.** The conversion of the list on every CTranslate2 call: 0.55 microseconds per entry (30,000 entries 16 ms per call, per batch, not
  per sentence; **spike**, extrapolated to a real vocabulary, **UNMEASURED** at real size); the per-step cost is noise (0.6 to 2 ns per entry
  per row-step). `Engine.warmup` builds every list a call can ask for (about 0.3 s per script list for a 64,000-piece vocabulary by the
  reviewer's estimate, unmeasured here).

**The measured leak** (**measured**, 2026-11-10, greedy, float32 export of the source-tag teacher; counts, not times):

| requests | mode | outputs with a forbidden letter | forbidden letters | counters |
|---|---|---:|---:|---|
| 500 the toy never trained (eng-eng, eng-rus, khk-khk, rus-rus, rus-eng) | `off` | 96 | 731 | `script_impure` 96 |
| the same 500 | `always` | 0 | 0 | all 500 `ok` |
| the same 500 | `retry` | 0 | 0 | `script_retries` 96, `script_retry_improved` 96 |
| 400 trained (the four directions) | `off`, `always`, `retry` | 0 | 0 | no retry |
| the D5 tiny converter's 448 words, asked for the traditional script | no list | 448 | 2,909 (spike) | - |
| the same 448 words, the list of a Cyrillic target | with the list | 0 | 0 | 42 entries listed |

The gate cannot pass vacuously: it requires at least 10 outputs with a forbidden letter WITHOUT suppression (96 here), at least 10 retries
(96), the purity floor held in every final output of `retry`, and at least 400 of the converter's words in the forbidden script without
the list (448). **What this shows**: the mechanism removes every forbidden piece the model can emit, also when the model is trying to emit
it. **What it does not**: whether it helps or hurts a real model. The toy's leak comes from requests it was never trained for (a same-language
request, a swapped direction); a real leak, the loss of a legitimate Latin name in Cyrillic output under `always`, the floor 0.98 and the
policy for the other languages are priors, which is why `retry` is the default and `always` is the gate's mode.

## `rinchen translate`

```
rinchen translate --model DIR --tgt CODE [--src CODE] [--script Cyrl|Mong] [--beam-size 4] [--n-best 1]
  [--max-decoding-length 256] [--max-source-pieces 256] [--batch-size 32] [--script-mask retry|always|off]
  [--device cpu|cuda] [--compute-type default] [--inter-threads 1] [--intra-threads 1] [--doc]
  [--format text|jsonl] [--input FILE] [--output FILE] [--strict]
  [--script-converter-ct2 DIR --script-converter-spm FILE]
```

`--model` also reads `RINCHEN_MODEL`. **The contract.** The model is opened (so our manifest vouches for every file of it) BEFORE any input is
read, and nothing is written to standard output when it is refused. Input is UTF-8 read strictly (invalid UTF-8 is exit 3 naming the line
and the byte offset, never quoting it), one text per line, split at LF only (one CR before the LF is dropped), translated in chunks of
`--batch-size` lines as they are read and written, flushed, chunk by chunk. There is **exactly one output line per input line, in order,
whatever the statuses**: a line that is empty, too long for the model or not translated gives an empty line, and `--format jsonl` says why
(`text`, `status`, `alternatives`, `src`, `tgt`, `script_purity`; `--n-best` hypotheses are written by jsonl only). With `--doc` the input is
documents, blocks of lines separated by blank lines (`--src` is then required), each translated as a whole; the output is one translated
document per line (its sentences joined by spaces), blocks separated by a blank line. Standard output carries only translations. The summary
goes to standard error.

**Examples** (**measured**, 2026-11-10; each invocation takes about 1.0 to 1.1 s, most of it the 1.0 to 1.2 s of imports of
`rinchen.cli.main`, which loads every command; the model loads in a few milliseconds). Five lines, of which one is empty, one is 2,000
characters and one has no letter (the toy renders the unknown noun of line 4 as the star sentence: it ranks nothing):

```
printf 'This is the star.\n\n<400 x "star ">\nThis is the house.\n!!!\n' | rinchen translate --model $W/ws/exports/src_float32 \
  --src eng_Latn --tgt khk_Cyrl
Энэ бол од.
(empty line)
(empty line)
Энэ бол од.
(empty line)
```

standard error (after the engine's JSON line `compute_type_effective`):

```
translated 5 lines: ok 2, empty 2, too_long 1, truncated 0, no_output 0
empty: first lines 2, 5
too_long: first lines 3
```

The summary is the line above, then one `<status>: first lines n, ...` line per status that is a problem (at most five line numbers each) and a
`counters:` line with the engine's counters that are not zero (`zero_shot` included). The int8 export gives the same lines. `--format jsonl
--n-best 2` writes one object per line:

```
{"text":"Энэ бол од.","status":"ok","alternatives":["Энэ бол сар."],"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":1.0}
```

and `--doc` over three English documents of the gate's document model (6, 8 and 8 sentences) writes three lines, blank-separated, with
`translated 3 documents: ok 3, empty 0, too_long 0, truncated 0, no_output 0` on standard error.

**Exit codes** (the first column was **measured** for each case):

| Exit | Case |
|---:|---|
| 0 | all lines written (also when some are not `ok`, unless `--strict`) |
| 2 | a bad option; a model our manifest does not vouch for (the old format, a flipped byte); `--tgt xxx_Latn` (`error: the model does not translate into 'xxx_Latn': its targets are eng_Latn, khk_Cyrl, rus_Cyrl`); `--script Mong` with no converter and no `mn_Mong` target; `--device auto` (typer: `'auto' is not one of 'cpu', 'cuda'`); `--compute-type auto`; `--doc` on a sentence model (`the model does not translate documents: its record says document_level is false`); `--doc` without `--src`; `--output` that is `--input` |
| 3 | an input that is not UTF-8 (`error: input line 2 is not valid UTF-8: invalid byte at offset 0 of the line`) or that the engine refuses (`error: line 2 contains the reserved symbol <2eng_Latn>, which the model reads as a control piece ...`) |
| 4 | CTranslate2 fails, or `--strict` meets a line that is not `ok`, after all the output is written (`error: --strict: 3 of 5 lines were not translated ok`) |
| 1 | the output cannot be written, a failure to close after a good run, or a broken standard output (`... \| head -1`: `error: cannot write the output to standard output: [Errno 32] Broken pipe`, one line, not Python's exit 120) |
| 130 | interrupted |

The target and every other language are checked right after the model opens and before the output file opens, also for an empty input: a
typo in `--tgt` with `--output out.txt` leaves an existing `out.txt` as it was. `--device cuda` is validated and was never run. `--script-converter-ct2` and
`--script-converter-spm` (both or neither) attach D5's converter, which loads before the export is verified. A failure of a chunk names its
lines (`lines A to B:`); the chunks before it stay written; no summary is printed when a run fails midway. A refusal of ONE text (a reserved
symbol, a lone surrogate) names it by its place in the input, one number counted from 1 (`line 4`, `document 2`), also in a chunk after the first: the engine
counts from 0 inside the call (`row 1`) and the command adds the chunk's offset to that index and keeps the engine's wording after it (before, the
message was `lines 3 to 4: row 1 contains ...`, which mixed file lines with a chunk-relative row, and `document 1: document 0 contains ...` for a
document). The engine's label `row N` or `document N` is read by a regular expression; a refusal with another label keeps the span (`lines A to B:`).

## The service

`rinchen serve --model DIR [--config FILE] [--host H] [--port P] [flags]` runs FastAPI under `uvicorn` in ONE process (more processes would
load the model more times: scale with `--inter-threads` and with replicas). The command builds the settings and has the manifest vouch for
the export BEFORE it binds a port, so a refusal is an exit code and nothing listens (exit 2 for a setting or a model that is refused, 3 for
files that cannot be read, 1 for any other failure of the command). It listens on 127.0.0.1 by default and has no authentication: put a
gateway in front.

### The endpoints

| Route | Answers |
|---|---|
| `POST /v1/translate` | the translation (below) |
| `GET /healthz` (and `HEAD`) | `{"status":"ok"}` while the process runs |
| `GET /readyz` (and `HEAD`) | `{"status":"ready","model":"rinchen-b7f33d88d5fa","quantization":"float32","device":"cpu","compute_type":"float32"}` once the engine is loaded, a warm-up translation ran and until the stop begins; else 503 |
| `GET /metrics` (and `HEAD`) | Prometheus text, content type `text/plain; version=1.0.0; charset=utf-8` |

Four routes, nothing else: no OpenAPI, `/docs` or `/redoc` (404 in the error shape), no redirect of a path with a trailing slash (a 404, not an
empty 307 whose `Location` comes from the client's `Host` header), no `Server` header, no CORS. A wrong method is a 405 with `Allow`.

### The request and the response

```
POST /v1/translate    Content-Type: application/json
{"texts": ["This is the star."], "src": "eng_Latn", "tgt": "khk_Cyrl"}
```

| Field | Default | Rule |
|---|---|---|
| `texts` | required | 1 to `max_texts` texts (at most 64), each at most `max_text_chars` characters (default 5,000); an empty text is allowed and answered `empty` |
| `src` | `auto` | a registry code, or `auto` (the script hint: reported as `und` or `mn_Mong`) |
| `tgt` | required | a registry code the model writes |
| `script` | null | `Cyrl` or `Mong`, for a Mongolian target only |
| `beam_size` | the config's | 1 to 8; the schema says 4 when none is sent, the service decodes with its configured beam then |
| `n_best` | 1 | at most the beam's size |
| `doc_mode` | false | translate each text as a document (a document model; `src` is then required by the engine) |

The request is strict: a quoted number or a boolean in the wrong place is refused, and a field it does not know is refused (`terminology` until
D12). The response, one entry per text in order (**measured**):

```
{"translations":["Энэ бол од."],"detected_src":["eng_Latn"],"model":"rinchen-b7f33d88d5fa","latency_ms":7.483,"statuses":["ok"],"script_purity":[1.0]}
```

and, for `{"texts": ["This is the star.", "", "!!!"], "tgt": "khk_Cyrl", "n_best": 2}`:

```
{"translations":["Энэ бол од.","",""],"detected_src":["und","und","und"],"model":"rinchen-b7f33d88d5fa","latency_ms":7.344,
 "statuses":["ok","empty","empty"],"script_purity":[1.0,null,null],"alternatives":[["Энэ бол сар."],[],[]]}
```

`alternatives` is present only when `n_best` is above 1. A text that is `empty`, `too_long`, `truncated` or `no_output` is a 200 with its status, never
a failure of the request. A document's `detected_src` is the language given, its status its worst sentence's and its `script_purity` the lowest
of its sentences.

### The errors

Every error is `{"error": {"code": str, "message": str}}`, the 422 included, and a message never holds what a client sent (the one thing of the
client's it names is an unknown key, up to 40 characters and then cut with `...`). The messages below are **measured** on the real server.

| Status, code | When | Message (abridged) |
|---|---|---|
| 400 `bad_request` | a `ConfigError` of the engine | `the model does not translate into 'xxx_Latn': its targets are eng_Latn, khk_Cyrl, rus_Cyrl`; `... into mn_Mong (its targets are ...) and no script converter is attached that could write it from khk_Cyrl`; `a script applies to a Mongolian target only (khk_Cyrl or mn_Mong), not to 'eng_Latn'`; `the model does not translate documents: its record says document_level is false`; `the source and the target language are both khk_Cyrl: nothing to translate`; `the source language 'xxx_Latn' is not in the language registry` |
| 400 `bad_request` | `n_best` above the configured beam when no `beam_size` is sent | `n_best (5) must not exceed beam_size (4): the beam holds no more hypotheses, and this service decodes with a beam_size of 4 when a request names none` |
| 400 `bad_request` | a reserved symbol in a text | `row 1 contains the reserved symbol <2eng_Latn>, which the model reads as a control piece and not as text ...: remove it` (names the row and the symbol, never the text) |
| 400 `bad_request` | any other `DataError` of the engine | the fixed `the model refused a text of this request` |
| 422 `validation_error` | the schema | `tgt: Field required`; `texts: List should have at most 16 items after validation, not 17`; `texts.0: String should have at most 2000 characters`; `beam_size: Input should be less than or equal to 8`; `n_best: Value error, n_best (3) must not exceed beam_size (2): the beam holds no more hypotheses` (a beam that was sent); `beam_size: Input should be a valid integer`; `terminologyAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...: Extra inputs are not permitted`; `body: Invalid JSON: key must be a string at line 1 column 2`; `body: the content type must be application/json` |
| 413 `payload_too_large` | the body is over `max_body_bytes`, by `Content-Length` or, for a chunked body, by counting what is read | `the body is over the limit of 262144 bytes` (with `Connection: close`) |
| 404 `not_found`, 405 `method_not_allowed` | an unknown path, a wrong method | `Not Found`, `Method Not Allowed` |
| 503 `overloaded`, `draining`, `not_ready` | the queue is full; the stop began; the engine is not ready | with `Retry-After: 1` on `/v1/translate` (`/readyz` has none) |
| 504 `timeout` | the request took longer than `request_timeout_seconds` | `the request took longer than the service allows` |
| 500 `internal` | anything else | `the model could not answer this request` (the cause is logged: the exception type, and the message only of an `UpstreamError`) |

A lone surrogate cannot arrive in a JSON body (the parser refuses it as `Invalid JSON`); the engine's own `DataError` for one is for callers of
the library. The 503s `not_ready` and `draining` exist for an app driven in process; over a socket they are a refused connection ("Readiness").
A request cancelled by the server at the end of its stop gets uvicorn's own plain-text 500, which cannot be shaped.

### The micro-batcher

`MicroBatcher(run, max_batch_sentences, max_wait_seconds, max_queue_sentences, workers)`: the items of one request are queued atomically (all or
none) under a key (the decode options CTranslate2 takes once per call; for documents also the languages), so a batch never mixes keys. A key's
batch is dispatched when it holds `max_batch_sentences` or its oldest item has waited `max_wait_ms`, to `workers = inter_threads` threads (at most
that many batches in flight, so under load batches grow by themselves); `QueueFull` (503 `overloaded`) when `queued + len(items)` would pass
`max_queue_sentences`, `Draining` (503) after `close()` began. Every item is answered once, in order (a Hypothesis property test pins it, and the reviewer's 400-round
seeded stress found nothing); an exception of `run` fails every item of that batch with it; a wrong number of results is an `UpstreamError`; the items
of a cancelled request that are still queued are dropped, and one already in a batch is run with it and its result discarded. `run_jobs` re-runs a
multi-request batch that the engine refuses request by request, so that only the guilty request gets its 400 (an `UpstreamError` is not isolated:
it fails the whole batch with a 500). `close(timeout)` dispatches what is queued at once, waits for what is in flight, and fails what is still
unanswered at the timeout with `Draining`; it never blocks the loop on a stuck worker, and a worker thread that is still stuck stays until it
returns.

### Readiness: what a client can see (`D-UVICORN-LIFESPAN-ORDER`)

**Readiness and the drain are not observable over HTTP.** `uvicorn` runs the START of the lifespan (the engine is opened, warmed up, the batcher
built, readiness turned on) BEFORE it binds the port, and closes the listener and waits for the connections BEFORE it runs the END of the
lifespan. Over a socket "not ready" and "draining" are therefore both a refused connection. Measured by polling `GET /readyz` every 5 ms from the
moment of the spawn (**measured**): `[(0.044 s, ConnectError), (0.789 s, 200)]`, never a 503; and around a SIGTERM: `200`, then `ConnectError` 0.10 s after
the signal. The false `/readyz`, the 503 `not_ready` and `draining` exist for an app driven in process (the tests: `S1b` and `S3b`, named
`*_in_process`) or served by something that binds first; nothing makes them visible on a socket. Consequences:

- The port opens only after the export has been hashed (by the command and again by the engine), loaded and warmed up: a health check on
  `/readyz` fails with a refused connection until then, **so its start period must be sized as a function of the model** (hash + load +
  warm-up). The toy is ready 0.7 to 0.8 s after the spawn (0.80 s here; 0.66 s in the gate). Hashing runs at about 3.8 GiB/s from the page
  cache (256 MiB in 0.067 s, spike), so a 4.4 GB export costs about 1.2 s per hash from a warm cache (ARITHMETIC), twice; a cold disk, the load of the model and the
  warm-up (which also builds the suppression lists) are **UNMEASURED for a real model**. An orchestrator that acts on health may restart a
  container that was only loading if the start period is too short: the 20 s of `docker/Dockerfile.cpu` and the compose file are sized for the toy.
- `shutdown_drained completed 0 failed 0` is what a real SIGTERM logs, because `uvicorn`'s own wait answers the requests in flight first: the
  batcher's drain matters only when the work outlasts that wait and the requests were cancelled. The line is not evidence that a drain was exercised.

### The stop and its budget (`D-SERVE-SIGTERM-EXIT`)

On SIGTERM `uvicorn` stops accepting, waits for the requests in flight, cancels what is left, runs the end of the lifespan (readiness off, the
batcher closed, the engine closed last and only when no batch is still running in it; `engine_left_loaded` is logged otherwise) and then
RE-RAISES the signal, so the process ends with status -15 (143 in a shell and for `docker stop`), not 0: treat 143 after a stop as a clean exit.

`shutdown_timeout_seconds` (T, default 20, at least 1) is the budget of the WHOLE stop. `ServeConfig.graceful_wait_seconds` is `ceil(T/2)` whole seconds
and goes to `uvicorn`; `drain_wait_seconds` is T minus that and goes to `batcher.close`. Before the fix of the review the two
waits were a whole T each, so a stuck batch made a stop take 2T: with T=20 and the compose `stop_grace_period: 30s` that is SIGKILL (137) and no
`shutdown_drained`. The requests in flight get `ceil(T/2)` (10 s with the default), an app driven in process waits `drain_wait_seconds` at the end of
its lifespan. T has a **floor of 1** (`SHUTDOWN_TIMEOUT_FLOOR`; found by the re-review): `uvicorn` takes whole seconds, so T=0.5 gave it a wait of 0 and a
request of 0.3 s got uvicorn's plain 500 at +0.13 s although its batch would have finished. A smaller value is a `ConfigError` that names the field, the
layer that supplied it and the floor (`shutdown_timeout_seconds (from --shutdown-timeout-seconds): Input should be greater than or equal to 1`); at T=1
the server waits 1 s and the drain wait is 0. A request cancelled at the end of the server's wait gets uvicorn's
plain-text 500 and is counted `translate/500` with the log code `cancelled`. **Measured** (fake engine on a real `uvicorn`, SIGTERM to the process
group; the stuck request's client got its 500 at half the budget):

| engine | T | process ended after the signal | client |
|---|---:|---:|---|
| stuck 8 s (3 runs) | 2 s | 2.17, 2.17, 2.17 s (the fix wave: 2.12 to 2.22; 4.18 s before the fix) | 500 at +1.13 s |
| stuck 12 s | 4 s | 4.18 s | 500 at +2.14 s |
| stuck 30 s | 20 s (the default) | 20.17 s | 500 at +10.15 s |
| a request of 0.8 s | 6 s | 0.87 s | 200 at +0.78 s; no `engine_left_loaded`, `shutdown_drained` `failed` 0 |

The re-review measured the bound again (fake engine stuck 6 s, real `uvicorn`): T=0.5 ended at 0.72 s (before the floor), T=1 at 1.17 s, T=1.5 at
1.67 s, T=2 at 2.17 to 2.22 s, T=3 at 3.18 s: **T plus about 0.2 s**. **The requests in flight get only `ceil(T/2)`** (10 s with T=20) while
`request_timeout_seconds` is 30 in `tiny.yaml`: a legitimate request of 15 s is cut at a stop (plain 500, counted `cancelled`) although the second half
of the budget is still unused. Size T as twice the longest request that must finish at a stop. The compose file's `stop_grace_period` must be at least T + 5 s (a test pins the formula; the 5 s are room for the server's polling and the exit of the
process): 30 s against T=20 holds. An idle server stops 0.21 s after the signal (**measured**). A real server with 24 requests of 8 sentences held in
the micro-batcher answered all 24 by 1.58 s and the process ended at 1.69 s with status -15 (the gate's S3; "The STOP gate").

### Metrics and logs

One `prometheus_client.CollectorRegistry` per app (no global state; the `_created` series are left out). Eleven families (**measured** on the real
server): `rinchen_requests_total{route,status}` (route in `translate`, `healthz`, `readyz`, `metrics`, `other`), `rinchen_request_duration_seconds{route}`
(histogram), `rinchen_translated_sentences_total{status}`, `rinchen_batch_size` (histogram, buckets 1 to 128), `rinchen_queue_depth` (gauge,
sentences), `rinchen_inflight_batches` (gauge), `rinchen_script_retries_total`, `rinchen_engine_events_total{event}` (the engine's counters, as the
delta of each scrape; a decrease is read as a restart), `rinchen_ready` and `rinchen_draining` (gauges) and `rinchen_model_info{name,quantization,
device,compute_type}` (gauge 1). Every label takes a value of a closed set (300 distinct 404 paths make ONE `other/404` series), so no text of a
request can become a series. Under the run of "Measured numbers" the counts equal the requests made (136 `translate/200`, 584 `ok` sentences, 584
engine sources).

Logs are JSON lines on standard error through `rinchen.utils.logging.get("rinchen.serve")`: `serve_start` (model, quantization, device, effective
compute type, **`model_sha256` and `manifest_sha256`**, threads, limits), `request` (`route`, `status`, `milliseconds`, `texts`, `target` only when
it is a target of the model, `code` on a refusal; a probe that works is logged at debug, a failed one at info, and both are counted),
`shutdown_begin` (`queued`, `inflight`), `shutdown_drained` (`completed`, `failed`), the engine's `compute_type_effective`, and the warning
`engine_left_loaded` and the errors `request_failed` and `unhandled_error`. No text of a request is ever in a log line, a metric label, an
exception message or an error body (the run above planted a marker in the requests and found it nowhere in the log). The access log of `uvicorn` is
off and `log_config=None`, so its own few plain lines (`Started server process`, `Uvicorn running on`, `Shutting down`) are the only non-JSON lines on
standard error.

### Configuration

`configs/serve/{tiny,server,edge}.yaml` are validated by `ServeConfig` (strict, frozen, unknown keys refused). **Precedence, the last wins:**
the defaults, the YAML file (`--config`), the environment (`RINCHEN_SERVE_<FIELD>`), the flags. The model is not a key: it is `--model` or
`RINCHEN_MODEL` (neither is under the prefix).

| Field | Default | Flag | Notes |
|---|---:|---|---|
| `host`, `port` | `127.0.0.1`, 8000 | yes | |
| `device`, `compute_type` | `cpu`, `default` | yes | `auto` refused for both |
| `inter_threads`, `intra_threads` | 2, 1 | yes | batches decoded at once; threads of one |
| `beam_size` | 4 | yes | 1 to 8; a request may name its own |
| `max_decoding_length`, `max_source_pieces` | 256, 256 | no (YAML or environment) | `max_source_pieces` at most 1,023 |
| `script_mask` | `retry` | yes | |
| `max_batch_sentences`, `max_wait_ms`, `max_queue_sentences` | 32, 5, 1024 | yes | `max_queue_sentences >= max_texts` |
| `max_texts`, `max_text_chars`, `max_body_bytes` | 64, 5000, 1 MiB | yes | at most 64, 100,000 and 64 MiB |
| `request_timeout_seconds`, `shutdown_timeout_seconds` | 30, 20 | yes | `shutdown_timeout_seconds` at least 1; at a stop the requests in flight get half of it ("The stop") |
| `log_level` | `info` | yes | |
| `script_converter` (`ct2_dir`, `spm_model`) | none | `--script-converter-ct2` and `-spm` | environment: `RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR` and `..._SPM_MODEL` |

`server.yaml` (CUDA, 4 x 1 threads, 64 sentences per batch, `max_source_pieces` 512) and `edge.yaml` (a small CPU box, 2 x 2 threads, 8 sentences,
timeouts 60 and 30) are **unmeasured priors** and say so in their headers; `tiny.yaml` (16 texts, 2,000 characters, 256 KiB, a 512-sentence queue, 128
decoding pieces) is what the tiny export and the compose file use. Every number is a prior: none was tuned.

**A mistyped environment variable is refused, not ignored** (`ServeEnvSettings` knows its fields): any name under `RINCHEN_SERVE_` (case-insensitive,
even with an empty value) that is no field is a `ConfigError` that names the variables with a "did you mean", and never their values. **Measured**
(exit 2, 0.9 s, the planted value `SECRET-9999` absent from the message):

```
error: unknown environment variable RINCHEN_SERVE_PROT (did you mean RINCHEN_SERVE_PORT?): a setting is RINCHEN_SERVE_<FIELD> for a field of
ServeConfig; the model is --model or RINCHEN_MODEL, not a setting
```

A Kubernetes Service named `rinchen-serve` makes the cluster set `RINCHEN_SERVE_SERVICE_HOST`, `RINCHEN_SERVE_SERVICE_PORT` and `RINCHEN_SERVE_PORT_8000_TCP...` in every pod of
its namespace, and the refusal reads them as typos: the start stops with exit 2 and the message names the variable (known limit 17). `RINCHEN_SERVE__PORT` (a double underscore) is refused the same way, `RINCHEN_SERVE_MAX_QUEUE_SENTENCE` suggests `..._SENTENCES`. **Every refused
value names the layer that supplied it** (**measured**): `invalid serve config: port (from RINCHEN_SERVE_PORT): Input should be less than or equal to
65535`; `max_texts (from $W/serve/bad.yaml): Input should be less than or equal to 64`; `max_text_chars (from --max-text-chars): Input should be greater
than or equal to 1`; `compute_type (from --compute-type): Value error, the compute type `auto` is refused: ...`. An environment variable that is empty
is not a layer; a name that is a field but the wrong type (`true` for an integer, `80.5` for an integer) is refused; relative paths resolve from the
working directory. `--device auto` is a usage error of the command (exit 2).

### Scaling and the thread tables

`inter_threads` x `intra_threads` must not exceed the cores given to the process. Start from `inter_threads` = cores / `intra_threads` with a small
`intra_threads` (1 or 2) and measure; the real optimum for a real model on a real CPU or GPU is **UNMEASURED**. What the toy says, with its limits:

- The mid-size random-init model of "Quantisation" (**spike**, raw `translate_batch`, load average about 12): `inter_threads` raised throughput (int8
  one sentence per call 32.9 -> 121.2 per second from 1 to 8, a factor of 3.7 with the machine loaded) and did not lower a call's latency;
  `intra_threads` lowered a call's latency (int8, one sentence: 59.7, 47.9, 34.3 ms at 1, 2, 4) and raised throughput less per thread used than
  `inter_threads` did; batching 16 per call raised throughput (float32 3.2 -> 19.4) at 10 to 25 times the call latency.
- The tiny export through THE ENGINE (**measured**, 1,200 requests, beam 4, `2 x inter_threads` Python threads calling `Engine.translate`, load
  average 11, sentences per second and p50 call latency):

| export | inter | intra | 1 sentence per call | 16 per call |
|---|---:|---:|---|---|
| float32 | 1 | 1 | 4,970 /s (0.4 ms) | 9,582 /s (3.2 ms) |
| float32 | 1 | 2 | 3,001 /s (0.6 ms) | 10,165 /s (3.1 ms) |
| float32 | 2 | 1 | 4,198 /s (0.9 ms) | 4,818 /s (14.2 ms) |
| float32 | 4 | 1 | 1,644 /s (4.6 ms) | 1,794 /s (72.9 ms) |
| int8 | 1 | 1 | 4,770 /s (0.4 ms) | 10,042 /s (3.1 ms) |
| int8 | 2 | 1 | 3,893 /s (0.9 ms) | 3,558 /s (17.9 ms) |
| int8 | 4 | 1 | 1,273 /s (5.7 ms) | 1,615 /s (80.9 ms) |
| int8 | 4 | 2 | 1,077 /s (6.9 ms) | 1,629 /s (79.8 ms) |

  Here MORE threads LOSE: a call of the toy takes 0.4 ms and a share of it is the engine's own Python (normalisation, SentencePiece, the
  hypothesis loop, locks), which holds the GIL, so Python threads contend. The spike's raw `translate_batch` table scaled with `inter_threads`
  because it had no Python between the calls. The tables do not contradict each other; they say that on a toy the Python side dominates and on a
  model with real compute the library does (the mid-size table). **Do not read the tiny table as advice for a real model**, and do not read the
  mid-size one as measured on one: it is random-init, 20 steps, on a loaded machine. The service's own numbers (64 x 1 and 64 x 8 concurrent
  requests, below) are unaffected by this: a request is one batch item.

**Serving numbers** (**measured**, real `rinchen serve` on 127.0.0.1, `tiny.yaml`, 2 x 1 threads; the first figure of each cell is the runbook's
driver, the second the gate's, taken minutes later on the same machine):

| export | ready after spawn | 64 x 1 sentence: p50 / p95 | 64 x 8 sentences: p50 / p95 |
|---|---:|---:|---:|
| float32, real server | 0.80 s; 0.77 s | 48.5 / 57.7 ms; 58.8 / 70.2 ms | 76.8 / 115.1 ms; 102.8 / 165.2 ms |
| float32, in process (the ASGI transport, no socket) | - | 25.8 / 28.7 ms (the gate) | 61.1 / 87.5 ms (the gate) |
| int8, real server | 0.74 s; 0.75 s | 53.7 / 58.3 ms; 60.7 / 70.6 ms | 70.0 / 115.0 ms; 105.3 / 165.7 ms |

All 200, every translation equal to the engine's, the `/metrics` delta exactly 136 `translate/200` requests and 584 `ok` sentences (8 warm-up requests, 64 x 1,
64 x 8); the gate is "all 200 and p95 below 2.0 s" (a margin of 12 to 70 times). These are toy latencies on a loaded machine, in the same order as the
spike's (p95 48 ms against a real uvicorn). The int8 export differs from float32 in 1 of 100 sentences (a count of a toy, not a quality claim).

### Health checks for an orchestrator

- **Liveness**: `GET /healthz` (or `HEAD`): 200 while the process runs. There is nothing to probe before the port opens, so a liveness probe also needs
  the start period below.
- **Readiness**: `GET /readyz` (or `HEAD`): 200 with the model name once the model is loaded and warm. Over a socket it is 200 or a refused connection.
- **Start period**: at least (twice the hash time of the export) + (the load time) + (the warm-up time) + a margin, all measured for YOUR model: the
  files of the image say 20 s for the toy and mark it **UNMEASURED** for anything else.
- **Stop**: `stop_grace_period` at least `shutdown_timeout_seconds + 5 s`; expect exit status 143 (a clean stop); never SIGKILL before the grace period.
- **Restarts**: no restart policy in the compose file: a refused model or setting is exit 2 and must stop the container, not loop.

## Docker

**What exists.** `docker/Dockerfile.cpu` (two stages, both `FROM python:3.14-slim`; uv copied from `ghcr.io/astral-sh/uv:0.13.7`; two `uv sync
--frozen --no-dev` calls, dependencies first and then the project as a wheel with `--no-editable`, both with the same 22 `--no-install-package`
names; `COPY README.md` kept although `pyproject.toml` has no `readme`; `USER 10001:10001`; `ENV CUDA_VISIBLE_DEVICES=""`; an exec-form `HEALTHCHECK`
(a Python `urllib` probe of `/readyz`, interval 10 s, timeout 3 s, start period 20 s, 3 retries); `ENTRYPOINT ["rinchen"]` with `CMD ["serve",
"--model", "/models/tiny", "--config", "/app/configs/serve/tiny.yaml", "--host", "0.0.0.0", "--port", "8000"]`), `docker/compose.serve.yaml` (one service:
builds the Dockerfile from the repository root, `platform: linux/amd64`, publishes `127.0.0.1:8000:8000` only, mounts `${RINCHEN_MODEL_DIR:-../models/tiny}`
read only at `/models/tiny`, `read_only`, `tmpfs /tmp`, `init: true`, limits of 4 cpus and 2 GB, `stop_grace_period: 30s`, the same health check, no device,
runtime, GPU or reservation, no restart policy), `.dockerignore` and the justfile recipes `serve-tiny` (`up --build`) and `docker-check`
(documentation: `just` is not installed here). There is no `Dockerfile.gpu`.

**The 22 skipped packages.** A frozen `uv sync --no-dev` of this repository would install 160 packages (**measured** as a dry run, offline), among them
`torch==2.14.1+cu130`, `triton`, `torchcodec`, `torch-optimi`, three `cuda-*` and 15 `nvidia-*` wheels; the repository's own environment holds 3.2 GB of
`nvidia/` and 1.2 GB of `torch/` (about 4.4 GB), which the CLI, the engine and the service never import (they import fine with `torch`, `triton`, `eole`,
`transformers`, `sentence_transformers` and `pyonmttok` blocked: the gate's S5). The Dockerfile skips every `torch`, `torch-*`, `torchcodec`, `triton`, `cuda-*`
and `nvidia-*` entry of `uv.lock`: **22 packages** (one reviewer counted 21 and left `torch-optimi` out; the unit test derives the set from the lock and asserts
equality with the Dockerfile's, and pins the number; re-derived on 2026-11-10: 22 and equal), which leaves 138 packages in a dry run. Eole, transformers and
sentence-transformers are still installed and would fail on import; serving imports none of them. The saving is ARITHMETIC from `du` of this repository's
environment, not from an image.

**What is verified here** (and how): `docker compose -f docker/compose.serve.yaml config -q` exits 0 (read only, no daemon, **measured**); static tests read the
Dockerfile (the base images and their tags, `--frozen`, the skip list equal to the lock, the vendored wheels the lock names copied before the sync, a user that is not
root, the hidden GPU, a health check on `/readyz`, the exec form of `ENTRYPOINT` and `CMD`, no `latest` tag, every `COPY` source present and not ignored) and the
compose file (the service, the loopback publish, the read-only model mount, no device or runtime or reservation, `stop_grace_period >= shutdown_timeout_seconds + 5 s`,
the health check equal to the Dockerfile's, the command equal to the image's, every option of it listed by `rinchen serve --help`); the compose `command` runs
natively as a process with only the model path, the config path, the host and the port replaced and the environment of the compose file applied (ready in 0.66 s,
`/readyz` 200, a translation equal to the engine's, SIGTERM to its group, exit -15); and (Task 10's check) a non-editable copy of `src/rinchen` without `configs/` serves the tiny
export with the packaged registry (ready 0.81 s).

**What is UNMEASURED:** the image build (`uv sync` inside it), the image size and start time, `docker compose up`, the base image tag (`python:3.14-slim`) and the uv
image tag (`ghcr.io/astral-sh/uv:0.13.7`, the version installed here: a guess), a health check and the stop of PID 1 inside a container (`init: true` and SIGTERM
reaching `rinchen`, exit 143 inside `stop_grace_period`), the cpus and memory limits, a model-volume layout for a real deployment, any GPU image. The build needs the
network (`uv sync` downloads the wheels, fetches `hatchling` through build isolation, and builds `kiwipiepy-model` from its source distribution because `uv.lock` names
no wheel for it), the image is `linux/amd64` only (the vendored `pyonmttok` wheel is x86_64), the model files on the host must be readable by uid 10001, and a
relative `RINCHEN_MODEL_DIR` resolves against `docker/`.

**The operator's steps** (never run here), from the repository root: `RINCHEN_MODEL_DIR=/absolute/path/to/an/export docker compose -f docker/compose.serve.yaml up
--build`; then `curl -s http://127.0.0.1:8000/readyz` until 200 and a `POST /v1/translate`; `docker stop` should return within the grace period with exit 143; check
the image size (`docker images`) against the 4.4 GB the skip list is meant to save. **Follow-up, not built**: a serving image that also skips Eole, transformers and the
data tools needs an extras split of `pyproject.toml`, which D11 does not make (`rinchen.cli.main` imports every command, so the image needs all their non-torch
dependencies); the `README.md` layer reinstalls everything on an edit of it.

## The packaging

- **The wheel.** `uv build --wheel` works offline here (`UV_OFFLINE=1`, hatchling cached; **measured** 0.3 s from a copy): 211 files, `rinchen-0.0.0-py3-none-any.whl`
  of 538,483 bytes, the entry point `rinchen = rinchen.cli.main:app`, package data included (`annotate/templates`, `script_conv/*.tsv`, `py.typed`), no `configs/` entry,
  and **`rinchen/_data/languages.yaml` byte-equal to `configs/languages.yaml`** (`D-WHEEL-LANGUAGES`). Without the hatch `force-include`
  (`"configs/languages.yaml" = "rinchen/_data/languages.yaml"`, the only edit of `pyproject.toml` in D11) the wheel lacked the registry and an installed package
  could not load it, because `config.languages` located it from the repository root four directories above the module.
- **The registry.** `config.languages.default_languages_path()` is the repository copy when it exists and the packaged one otherwise; `LANGUAGES_PATH` keeps naming the
  repository copy; `load_languages(path=None)` resolves at call time. A test loads the registry from the unpacked wheel with no `configs/` on the path. Other uses of
  the repository root (`configs/data/*.yaml`, `data/...`) belong to commands that need a checkout.
- **The lazy import.** `rinchen/__init__.py` exports `Translator` through a PEP 562 `__getattr__` (with `__dir__`, `__all__` and a `TYPE_CHECKING` import): `import
  rinchen` loads nothing heavy and a subprocess test pins the `sys.modules` of each step; `python -m rinchen` exists.
- **The import costs** (**measured**, 2026-11-10; wall seconds of a fresh interpreter, which alone takes 0.013 s and 10.2 MB, and peak resident set from `/proc`):

| What | Wall | Peak RSS |
|---|---:|---:|
| `import rinchen` | 0.019 s | 11.3 MB |
| `rinchen.Translator` resolved (no `ctranslate2`, `sentencepiece`, `torch`, `eole`) | 0.252 s | 47.6 MB |
| the engine with `ctranslate2` and `sentencepiece` (no `torch`, no `eole`) | 0.278 s | 55.1 MB |
| `rinchen.serve.app` and `uvicorn` (no `torch`, no `eole`) | 0.421 s | 62.6 MB |
| `rinchen.cli.main` (loads every command; no `torch`, no `eole`, no `transformers`) | 1.134 s | 170.9 MB |

  `rinchen translate --help` takes 1.26 s: the command pays the imports of every command because lazy command loading is not built.

## The STOP gate

The master plan's gate for D11 is "`docker compose up` serves a tiny model". **It cannot be proven here and is proven in parts**: no image was built and
`docker compose up` never ran (no network, no authorisation). The parts that are proven are listed under "Docker". What the gate modules prove, all through the real
`rinchen` binary, the real Eole and the real CTranslate2 on the CPU with the GPU hidden (`-m subprocess`, every subprocess bounded by 900 s):

| Module | Tests | What it runs |
|---|---:|---|
| `tests/integration/test_export_parity.py` | 13 | G1 strict parity of the source-tag teacher, G2 of the target-tag teacher (and the trap with the prefix kept), G3 the quantisations (chrF, sizes, compute types, parity of int8), G7 the GPU scan and the logs |
| `tests/integration/test_engine_surfaces.py` | 24 | G4 script masking on 500 and 400 requests and the D5 converter, G5 `Translator`, `rinchen translate` (one line per line, refusals, closed pipe, target tag, `--script Mong`), G6 documents (a document model, windows of each size, the end-token fallback) |
| `tests/integration/test_serving_gate.py` | 13 | S1 the TestClient over the real engine, S1b and S3b in process, S2 64 concurrent requests in process and against a real server, S2c a reader of the counters during 1,200 commits, S7 the int8 export served |
| `tests/integration/test_serving_gate_process.py` | 15 | S3 SIGTERM with 24 requests inside the service, S4 the compose command natively, S5 the GPU stack installed and blocked, S6 the refusals (old format, unknown key, one flipped byte in each of the five files), S8 the AST scan |
| `tests/integration/test_serving_stop_budget.py` | 2 | a fake engine on a real uvicorn: a stuck batch ends within the budget; a request inside the budget is answered |
| `tests/integration/test_distill_tiny_student.py` | 44 | D10's gate, re-run on the new reader |

The thresholds (never lowered; a miss is answered by raising the steps of the checked-in config; none was missed) and the numbers (**measured**, the gate's own
properties in the run of 2026-11-10: 111 tests passed in 517 s, one pytest process):

| Gate | Threshold | Measured |
|---|---|---|
| strict parity, float32, source tag / target tag | exact at least 0.95 and mean abs delta at most 1e-3 per token | 200 of 200, 7.19e-7 (max 3.43e-6) / 200 of 200, 1.37e-6 (max 5.05e-6) |
| the trap: the forced tag's log-probability kept | the parity must FAIL, with a delta above 0.01 and above 1,000 times the right one | 200 of 200 exact, 0.0433 per token: FAIL |
| int8 chrF against float32 | drop at most 1.0 | +0.061 (98.268 against 98.207); int8_float16 -0.010 |
| int8 parity | exact share at least 0.95 only | 199 of 200, delta 1.14e-3 reported, not gated |
| `model.bin` | int8 below float32, int8_float16 below int8 | 1,488,691 (73.2 %) and 844,931 (41.5 %) of 2,034,311 |
| script masking, 500 never-trained requests | `always`: 0 forbidden letters; without suppression at least 10 outputs leak (not vacuous); `retry`: at least 10 retries, floor 0.98 holds | 0; 96 outputs (731 letters); 96 retries, 96 improved, 0 left |
| the D5 converter | at least 400 of its words in the forbidden script without the list, 0 with it | 448, 0 (42 entries listed) |
| `Translator`, `rinchen translate` | equal to the engine's output on 50 sentences; one line per input line; `ok 50`; `--script Mong` 50 of 50 lines with Mongolian letters | all |
| the target-tag engine | equal to raw CTranslate2 with the mask off | 400 of 400, chrF 58.329 (the 1,000-step teacher is weak; a number, not a gate) |
| documents | at least 0.9 of the windows of each size line up; one translation per sentence; no sentence missing | 325/334 (97.3 %), 260/268 (97.0 %), 191/202 (94.6 %); 400 `ok`, chrF 84.747; 94 of 106 windows fall back at a 20-piece cap |
| the service, 64 concurrent requests | all 200 and p95 below 2.0 s, translations equal to the engine's, the `/metrics` delta exactly the requests made | all 200; in process p95 28.7 and 87.5 ms; real server p95 70.2 and 165.2 ms; 136 requests, 584 sentences |
| SIGTERM with 24 requests (192 sentences) inside the service | all answered, the process gone by the signal's status | all 24 answered 1.57 s after the signal, exit -15 at 1.68 s, 4 refused connection attempts, `shutdown_drained` 0/0 (vacuous, "Readiness") |
| the compose command natively | ready, a translation equal to the engine's, exit -15 | ready in 0.75 s |
| the GPU stack blocked | `torch`, `triton`, `eole`, `transformers`, `sentence_transformers`, `pyonmttok` are installed here and refused after the block; `rinchen translate` and `serve` work | `translate` of 5 lines in 0.92 s, byte-equal to unblocked; `serve` ready in 0.75 s and translating |
| the refusals | old format, unknown config key, one flipped byte in each of the five files: exit 2, empty stdout, no `serve_start`, no uvicorn, the port free; an undamaged copy serves | all |
| no GPU question | an AST scan of the new code and tests (no `torch.cuda`, no `get_cuda_device_count`, no `nvidia-smi`; every `ctranslate2.Translator(` with `device="cpu"`) and Eole's own `GPU ranks: []`, `Starting training in cpu mode` | clean |

**What the numbers show.** That the whole chain works through the real binaries on the CPU: a checkpoint of ours becomes an export a manifest vouches for; CTranslate2 reproduces
Eole's greedy translations and, for float32, its log-probabilities to six places in both tag placements; the int8 file is smaller and translates within the toy's noise; a
mask removes every forbidden piece; a document model's hypotheses are cut at `<sep>`, the chunker gets one translation per sentence and a cut window falls back; a service
answers 64 concurrent requests, counts them exactly, drains and exits by the signal; and a serving process needs none of the GPU libraries. **What they do not show.** Anything
about quality (the toy saturates at chrF 98.2 and ranks nothing: they cannot say int8 is lossless, that masking helps a real model or that the thread settings suit one); anything
on a GPU, in `float16` or `int8_float16` as GPU types, or at a real size; the container (not built); the margin of the parity for a real model; and the stop behaviour of PID 1.
The service's latency numbers are those of a toy on a loaded machine.

## The operator workflow for a real model

**UNMEASURED:** no real model, corpus or GPU run exists. The steps are the ones the tiny gates ran; the numbers to measure are the ones listed. Every GPU step needs the owner's
explicit authorisation.

1. **Export** the averaged teacher (or the student of runbook 10) at float32: `rinchen export convert --checkpoint <averaged> --spm-model <the model's> --tag-placement
   <src|tgt> --out-dir <E32>`; read `directions`, `document level` and `tag placement` in the printed lines (a statement that the saved config cannot check says
   `stated`). UNMEASURED for a real teacher: the time to hash a 13 GB step directory, the converter's time and MEMORY for 1.1 B parameters on the CPU, the size of `model.bin` (about
   4.4 GB at float32, ARITHMETIC).
2. **Verify and run the strict parity**: `rinchen export verify <E32>`, then `rinchen export parity --checkpoint <the same checkpoint> --export <E32> --corpus <D4 output> --out-dir
   <P>`. Read the mean delta against 1e-3 and the exact share against 0.95. UNMEASURED for a real model: the margin, and whether a long hypothesis is cut at `max_length` 128 (it adds a
   spurious delta: raise nothing, read `mismatches`).
3. **Quantise** and measure the cost: `--quantization int8` (CPU) or `int8_float16` (GPU, unmeasured), on the same checkpoint; translate the dev sources of the corpus with
   `rinchen translate --format jsonl` or `rinchen.Translator` and score them (`rinchen.metrics.bleu_chrf.corpus_scores`) against float32: the gate is a drop of at most 1.0 chrF; no
   command scores an export yet (D12 and D13 will). UNMEASURED: the real int8 loss, the file's share of float32 (29 % on a 6/6 d512 random-init model, 73 % on the toy), the speed.
4. **Choose the script mode**: keep `retry`; read the engine counters (`script_retries`, `script_retry_improved`, `script_impure` in the `counters:` line of `rinchen translate` and in
   `rinchen_engine_events_total`) over real requests, compare `always`, and calibrate the purity floor 0.98 and the policy for the eight other languages (the registry's) on
   real text. UNMEASURED: whether a hard mask helps or hurts, the loss of a legitimate Latin name in Cyrillic output under `always`, the cost of a 30,000-entry list per call.
5. **Smoke test** with `rinchen translate --strict` on a few real lines and, for a document model, `--doc`.
6. **Serve**: `rinchen serve --model <E> --config configs/serve/<server|edge>.yaml`; the shipped numbers are priors. Measure the start (hash + load + warm-up) and set the health check's
   start period from it; measure the thread settings under your own traffic (`inter_threads` x `intra_threads` within the cores given to the process); set `stop_grace_period` to at
   least `shutdown_timeout_seconds + 5 s`; treat exit 143 as a clean stop. `--device cuda` and `float16` are the operator's choice, never run here.
7. **Container**: the steps of "Docker", then compare the image size with the 4.4 GB the skip list should save; consider the extras split.
8. **A gateway in front**: authentication, TLS, rate limits, a body read timeout and a cap on `beam_size` and `n_best` (the service has none of them).

## Open gates

Not done by D11, because no real data, GPU, container build or reader exists: everything GPU-side (CTranslate2 on CUDA, `float16` and `int8_float16`, token batching, the speed of a
1.1 B-parameter teacher and a 190 M server student, replica memory, the best `inter_threads` and `intra_threads`, the cost of `retry` and of a 30,000-entry suppression list at real
size; `--device cuda` is validated and passed through, never run); the real quantisation loss and parity margin (int8 +0.06 on a toy that saturates at 98.2; float32's delta is 1e-6 on
the toy; the 1.0 and 1e-3 thresholds are the master plan's); whether a hard script mask helps a real model (the purity floor 0.98 and the policy for the other eight languages are
priors); Docker (the image build, its size and start time, `docker compose up`, the base and uv tags, a health check and a SIGTERM inside a container, a model-volume layout, a GPU
image); documents (throughput, joiners beyond a space, the `tgt` placement, the quality of a real document model decoding a lone sentence); the start-up cost of the command (1.0 to
1.2 s of imports; lazy command loading is not built) and an extras split of `pyproject.toml` for a lighter image; directions derived from the saved config (a checkpoint our launcher
did not write needs `--direction`; zero-shot pairs are flagged, never judged; auto-detection of the source is the script hint only); the concurrency limits of the service (one process;
the batcher's `max_wait`, `max_batch_sentences` and queue limit are toy priors; no per-client limit, no authentication, no hot reload); the start period of the health check for a real model;
the D3 gates assigned to D4, the D5 gates that need real text or a reader and the D6 to D10 gates stay open as recorded in their runbooks.

## What was not verified

- **Anything on a GPU**, and any real model or corpus. Every translation ran on the CPU with CTranslate2 on `device="cpu"`; every Eole run had `CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`;
  `--device cuda` was passed through a faked seam only.
- **The container**: the image was never built or run; its size, start time, the stop of PID 1, the in-container health check, the base and uv tags and the resource limits are guesses.
- **Quality**: that int8 is lossless on a real model, that masking helps one, the real parity margin.
- **Throughput and the thread settings of a real model**: the tables are a toy through the engine (more threads lose) and a random-init mid-size model through raw `translate_batch` (more
  threads win), both on a loaded machine.
- **The start period for a real model** (hash + load + warm-up before the port opens).
- **A document model of any quality**, the throughput of a document request and the service's `doc_mode` against a real document model (it ran against a fake engine only).
- **`KeyboardInterrupt` with a real CTranslate2 call** (exit 130 is tested with fakes; the real call is a blocking C++ call that delivers the signal when its batch returns).
- **CI**: no CI run covers D3 to D11. **Windows and macOS** were not run. **A re-review of the final wave** (`330250a..039d1d7`) was not asked for: its fixes are small, each has a failing test
  first and a mutation check, and the full gate ran.

### Known limits, left open on purpose

Findings of the reviews that were parked and not fixed, and limits of the build, one each:

1. **A line of `rinchen translate` is read whole with no cap** (review M-2): an 11.5 MB line took 4.9 s and 1.3 GB of resident memory before it was refused as `too_long`. A pre-check on the
   length of a text before `encode` needs the vocabulary's longest piece to be safe and was not made.
2. **`rinchen serve` hashes the export twice** (review M5): the command's `verify_export` and `Engine.open` each take the SHA-256 of every file; a swap between the engine's hash and
   CTranslate2 reading `model.bin` is not caught. Passing the `VerifiedExport` to `Engine.open` would avoid the second hash.
3. **An engine that fails to open exits with `uvicorn`'s status 3 after a raw traceback** (review M6; for example `--compute-type float16` on a CPU), where the command documents 3 as
   "files cannot be read" and a `ConfigError` is 2 elsewhere.
4. **No body read timeout, and the work of a disconnected client is not cancelled** (review M7): standard `uvicorn`; put a gateway in front. A request keeps its batch slot until it ends.
5. **`/v1/translate` validates the body before the readiness check** (review M8): a bad body is a 422 even while the service is not ready (reachable in process only).
6. **No operator cap on a client's `beam_size` (up to 8) or `n_best`**, and no per-client rate limit or authentication.
7. **`uvicorn`'s own plain-text errors cannot be shaped**: a malformed HTTP request is a plain 400 and a request cancelled at the end of a stop a plain 500 (counted and logged as
   `cancelled`); stderr mixes `uvicorn`'s few plain lines with the JSON events.
8. **A blank converter path from the environment is accepted** (pydantic-settings turns `RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR="   "` into `Path("   ")`).
9. **`rinchen translate` names `DecodeOptions.n` in its own error** for `--n-best` above `--beam-size` (an existing test pins the wording); `doc_chunker._refuse_separators` still raises a
   plain `DataError` (unreachable today: the engine refuses `<sep>` earlier, and an AST test lists every message that interpolates a value).
10. **`quantize.effective_compute_type` is used by tests only** (review M-7); `infer/engine_types.py` mixes the public types with the working types of a call (review M-9); the
    checkpoint's own `manifest.json` is not among the export manifest's inputs (review M-11, as in D10); `ModelRecord` is frozen but not hashable (review M-14); `_load_processor`,
    `_require_no_directory` and `_hashed` exist in more than one module; the parent directories of `--out-dir` are made as the converter would have made them, and two exports into one
    `--out-dir` name share `<out_dir>.export.log`; a hard kill of an export leaves a hidden `.<name>.<random>` directory (as D10).
11. **Files near the 900-line cap**: `infer/engine.py` has 769 lines and `infer/engine_types.py` 594; `tests/support/distill_gate_stages.py` has two pre-existing mypy complaints (tests are not
    type-checked). New behaviour of the engine will have to move code out first.
12. **The Docker recipe**: the image is `linux/amd64` only (the vendored `pyonmttok` wheel is x86_64); the build needs the network (`uv sync`, `hatchling` by build isolation and the
    `kiwipiepy-model` source distribution, which `uv.lock` names no wheel for); a `README.md` edit reinstalls every dependency (the layer-cache note); `cap_drop` and `no-new-privileges`
    are not set; the health check's port 8000 is hard-coded; every image claim is UNMEASURED.
13. **`tests/unit/conftest.py` imports FastAPI and the serve test support in every unit session** (about 0.17 s; a `pytest_plugins` entry would avoid the re-export).
14. **One load flake seen in a D8 test**: `tests/unit/test_train_launch_orphan.py::test_a_crashed_eole_leaves_no_rank_running_and_the_next_launch_is_accepted` timed out once under a load
    average of about 12 (a process-lifetime wait of D8's), passed alone in 4.7 s and did not recur in the final gate (11768 passed). It is D8 code, untouched by D11; a flake of this kind is
    to be reported, not retried away.
15. **The suppression-list cost at a real vocabulary** is extrapolated (0.55 microseconds per entry per call) and the warm-up build of 64,000-piece lists (about 0.3 s per script list, the
    reviewer's estimate) is unmeasured here.
16. **Disclosed process slips of the wave**, one line each: the design agent ran `ctranslate2.get_cuda_device_count()` once without hiding the GPU (a driver query: no model, no kernel);
    Task 10 ran `docker compose version` (read-only, outside the allowed list) and issued a `pkill -f` against its own crashed harness, which the permission classifier denied (nothing was
    killed); Task 11 dispatched a trivial subagent with the prompt "noop" against the rule that implementers dispatch none (it read no file and ran no command).
17. **Kubernetes service links collide with the refused prefix** (re-review m-3): a Service named `rinchen-serve` gets `RINCHEN_SERVE_SERVICE_HOST`, `RINCHEN_SERVE_SERVICE_PORT` and
    `RINCHEN_SERVE_PORT_8000_TCP`, `..._TCP_ADDR`, `..._TCP_PORT`, `..._TCP_PROTO` injected into the environment of every pod of its namespace, and `ServeEnvSettings` refuses every name under
    `RINCHEN_SERVE_` that is no field, so the service does not start (exit 2; the message names the variable and no value, so it is diagnosable). Kubernetes also documents
    `RINCHEN_SERVE_PORT=tcp://<ip>:<port>` for such a Service, which is the name of a real field with a value that is not an integer, refused as a bad `port` (from the documented naming of
    service-link variables; no cluster was available, so the whole item is UNMEASURED). Set `enableServiceLinks: false` in the
    pod spec, or name the Service otherwise. Not fixed: an allow-list of such names would hide real typos of `RINCHEN_SERVE_PORT`.
18. **A stop cuts a request that outlasts half the budget** (re-review m-4): `uvicorn` gets `ceil(T/2)` (10 s at T=20) while `request_timeout_seconds` is 30 in `tiny.yaml`, so a legitimate
    request of 15 s is cut at a stop with a plain 500 (counted `cancelled`). Raise `shutdown_timeout_seconds` to twice the longest request that must finish; the compose
    `stop_grace_period` must follow (T + 5 s). The alternative that gives the requests the whole budget needs a `uvicorn.Server` subclass (fix wave, I1) and was not built. At T=1
    (the floor) the drain wait is 0.
19. **An unknown key of a request is echoed up to 40 characters in the 422** (re-review m-7; the brief of the fix wave allowed a bounded echo): `unknown field 'xxxx...'`. It is a name the
    client sent, never a text of a translation; a fixed `unknown field` would drop even that.
20. **What the stop-budget test does not catch**: `tests/integration/test_serving_stop_budget.py` allows T plus 2.5 s (T=4, stuck 12 s); a server that gave BOTH waits the whole timeout ends at
    about 8 s and fails it, but one that gave only the batcher's wait the whole timeout ends at about 6 s (ARITHMETIC: 2 s of `uvicorn` plus 4 s) and passes it. The unit tests (`graceful + drain == T`, the
    batcher closed with the second share only) kill that mutant: the integration test passed under it, `test_config_serve.py` failed.

### The review, its fix wave and the re-review

**The combined review** was two independent read-only reviewers, as D10: R1 on Tasks 1 to 7 (`17549f8..6ed0e72`) and R2 on Tasks 8 to 11 (`6ed0e72..b6d57a4`). Both: ready with fixes, 0 Critical.
R1 found 3 Important findings: **I-1** a document-level export returned `no_output` for every sentence not decoded inside an aligned window (400 of 400 plain rows with the spike's
document model, 18 of 400 document sentences, chrF 88.47), contradicting D9's own contract; **I-2** a literal reserved symbol in a request text became a control piece (`<2eng_Latn> This is
the star.` into `khk_Cyrl` gave `ok` and the wrong language); **I-3** `torch` was dropped from the tool versions of the export and parity manifests against the hard constraint. R2 found 4
Important findings, all operational: **I1** a stop could take twice `shutdown_timeout_seconds` (SIGKILL under the compose grace period, no `shutdown_drained`); **I2** a valid `n_best` was
refused with a misleading message when the configured beam was not 4; **I3** a mistyped `RINCHEN_SERVE_*` variable was silently ignored; **I4** readiness-false and draining are
unreachable over HTTP, with the consequences for the health check's start period and for the meaning of `shutdown_drained`. Both found the rest of the build sound: the record's
strictness, hash once before any parse, staging and cleanup, the layering (by an independent grep), the alignment of the target prefix, truncation and counters, the lazy imports, the
translate contract, the packaging (offline wheel and sdist), the parity method, the error shape on every path, shared-batch isolation, the batcher under stress, closed metric label
sets, and that no request text reached any log or message (about 40 request shapes with a planted marker).

**The fix wave** (nine commits, `b6d57a4..4bfaa5e`, failing test first, mutation-checked): `814b57d` torch recorded again (I-3); `966387a` a document model's hypothesis cut at its first
`<sep>` (I-1; "Documents"); `266dbea` reserved symbols in a text refused (I-2; it closes the item the ledger had deferred to D12); `765c63c` the minors of the export and the engine (M-1
validate the target before the output opens, M-3 a broken pipe is exit 1, M-4 the parity binds the checkpoint by three digests, M-5 error mapping, M-6 the converter's message not repeated,
M-8 one `MAX_SOURCE_PIECES` import and one `document_status`, M-10 `TeacherOptions` refuses `auto`, M-12 the remedy for a language missing from the registry, M-13 the warm-up builds every
list); `a291106` one budget for a stop (I1); `f464cc0` `n_best` against the configured beam (I2); `784361d` unknown `RINCHEN_SERVE_*` names refused and the layer named (I3); `38c490b` the
docstrings and comments of readiness and the stop made honest, the in-process tests renamed (I4); `4bfaa5e` the minors of the service (redirect_slashes off, HEAD on the probes, a bounded
unknown key, a fixed message for a `DataError`, the single `document_status`, the public `ServingEngine`, a per-probe debug level, compose `platform: linux/amd64` and the Dockerfile's
disclosures). Where the wave decided differently from its brief: the stop budget is split in two rather than passing the REMAINING budget to the batcher (that needs a `uvicorn.Server`
subclass and would lose `uvicorn.run` and its exit status); M-2 was not done; the wording of the 22-package count is the derivation from the lock. The full gate after the wave:
**11768 passed, 1 skipped, 10 deselected** (24 min 1 s, one sequential process; the run before the wave had 11562), `ruff` and `mypy` clean; the integration
modules once each (109 tests, 502 s; 111 with the two stop-budget tests, 517 s, in the run of this runbook) and D10's gate (44) with every number unchanged.

**The scoped re-review** of the fix wave (`b6d57a4..4bfaa5e`, an independent reviewer, read-only) found that the wave **needed one small fix**: 0 Critical, 1 Important
finding and 8 Minor ones. It ran 388 targeted unit tests and the two stop-budget tests (2 passed, 5.96 s) and probed real servers and real exports on the CPU with the GPU hidden. The
records of Task 12 were written while it ran and said so; this section closes that.

- **I-1 (Important): the fix of the review's I-1 was incomplete.** `<sep>` was still suppressed for a lone sentence of a document model in the mode `always` and in the second pass of
  `retry`, so the model ran on and invented text while the status stayed `ok`. Reproduced on the gate's own document model (400 toy test sentences as plain rows, beam 1): `off` 400 `ok`
  chrF 88.23; `retry` 400 `ok` 88.42 with 5 retries whose replaced texts carry invented trailing sentences; `always` 400 `ok` 81.80 with 358 of 400 texts different from `off`. Not a
  regression of the wave, but the shipped default is `retry`. **Fixed in `330250a`**: after it `off` 88.23, `retry` 88.49, `always` 88.49 (5 of 400 texts differ from `off`) and no invented trailing sentence ("Documents"; 2 failing tests, 5 mutants killed).
- **What the probes showed that held**: A-I1's first pass in the default mode, windows and sentence exports unaffected, a truncated hypothesis stays truncated, n-best alternatives cut
  separately, the reserved check on the head; A-I2 (pieces for rows, normalised text for documents, zero-width characters inside `<sep>` refused in both, placeholders excluded, messages
  with the symbol only, exit 3, the service isolating each request, counters untouched on a refusal); `torch` in the tool versions; `Engine.require_request` in the real call order;
  the broken-pipe handling (`/dev/full` is exit 1 with one error line); the parity digests; the converter's message; `MAX_SOURCE_PIECES`; the `auto` refusal of `TeacherOptions`; the
  warm-up prebuild; the stop budget (below); the `n_best` paths over a real `uvicorn` (a 400 in API terms when the service's beam is the limit, a 422 on `n_best` when the request names
  the beam, beam 0 and 9 a 422, 4 and 4 a 200); the environment refusal (case-insensitive, empty values, `RINCHEN_MODEL*` untouched, the compose variables are fields, values never
  quoted); redirect_slashes, HEAD, the probe log level, `RequestTextError`, the AST scan (52 entries with a reason each), the public `ServingEngine`, `platform: linux/amd64`, the 22
  packages (44 flags); and that no existing test was weakened.
- **The stop budget, measured again** (fake engine stuck 6 s on a real `uvicorn`, SIGTERM to the process group): T=0.5 ended at 0.72 s, T=1 at 1.17 s, T=1.5 at 1.67 s, T=2 at 2.17 to
  2.22 s, T=3 at 3.18 s, i.e. T plus about 0.2 s; `graceful + drain` equals T exactly; the cancelled request is recorded once and the `CancelledError` re-raised (nothing swallowed).
- **The minors fixed in `55629cd`** (`D11: re-review minors`; tests first, 14 mutants killed; two survived: a redundant `continue` after the new counter, which was removed, and a graceful wait equal to the whole timeout, which leaves graceful plus drain at T and so changes nothing a test can see): **m-1** a hypothesis (or an n-best
  alternative) that starts with `<sep>` is counted `reserved_symbol` (the row stays `no_output`); **m-2** `shutdown_timeout_seconds` below 1 is refused naming the field, the layer and the
  floor, and `graceful_wait_seconds` lost its branch for T under a second (graceful plus drain still equal T); **m-5** `rinchen translate` names a refused text by one number counted from 1
  (`line 4`, `document 2`), also in a chunk after the first ("`rinchen translate`"); **m-6** the overlong prose lines of the Dockerfile header and of the serve command's docstring are
  reflowed (the exec-form `HEALTHCHECK` and `CMD` stay on one line, as Docker needs them readable); **m-8** the margin of the stop-budget test is 2.5 s (it was 1.5 s).
- **Left as limits, on purpose**: **m-3** Kubernetes service links (limit 17), **m-4** the first half of the budget for requests in flight (limit 18), **m-7** the 40-character echo of an unknown
  key (limit 19). The stop-budget test's margin is also limit 20.
- **Existing tests changed because the code forced it** (the same list is in the commit messages): `test_the_list_of_a_window_is_a_list_of_its_own` took its lone sentence from a
  sentence model (the old form pinned the defect: a lone sentence of a document model suppressing `<sep>`); the bound and budget tests of `shutdown_timeout_seconds`
  (`test_config_serve.py`: values below 1 moved to the refused side, two budget rows and three no-overshoot values removed because T under 1 is no longer valid); the shutdown test
  of `test_serve_app_shutdown.py` that used T=0.2 now uses 1.5; the reserved-symbol test of `test_cli_translate_exits.py` expects the new message.
- **The final wave** (`330250a`, `55629cd` and `039d1d7`, the last a gate test of the three script modes on the real document model): the full gate **11778 passed, 1 skipped, 10 deselected** (26 min 51 s, one sequential process;
  11768 before), `ruff` and `mypy` clean; the integration modules once each: 112 passed in 604 s: parity 13, surfaces 25, serving gate 13, process 15, stop budget 2, D10 gate 44.

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython 3.14.10, eole 0.6.4, ctranslate2 4.8.4, torch 2.14.1+cu130,
sentencepiece 0.2.1, fastapi 0.162.0, uvicorn 0.54.0, httpx 0.28.1; the GPU was hidden from every process and every CTranslate2 translator was built on `device="cpu"`. All on 2026-11-10,
once each, code at commit `4bfaa5e`, **on a machine that was loaded by other work (load average 3 to 14), so every wall time is longer than an idle machine's** (the gates' times, taken in a
separate run minutes later, are in parentheses where they differ); the numbers that are not times do not depend on the load. The drivers were read-only scratch scripts
(`d11-t12/p1_main.py` to `p5_small.py`, `docmeasure.py`) that reuse the helpers of `tests/support/d11_gate*.py`, `serving_gate*.py` and `distill_gate_stages.py`, so the numbers are the
gates' own; nothing was written under `data/` or `models/`.

| What | Value |
|---|---|
| the tiny teachers: `ablate --prepare-only`, `launch` (2,000 steps, source tag; 1,000, target tag), `average --last 2` | 1.2 s, 84.7 s (66.9 s in the gate), 1.5 s; 1.3 s, 42.8 s (38.2), 1.0 s; averaged weights SHA-256 `f4e32704b958...` and `e2410a0f7b0d...` |
| `export convert` (float32, int8, int8_float16, target-tag float32) | 4.8, 5.3, 5.4, 5.3 s (3.6 s each in the gate); `model.bin` 2,034,311 / 1,488,691 / 844,931 / 2,034,311 B; the five files and the manifest of the float32 export sum to 2,060,474 B; `rinchen.json` 1,121 B, `manifest.json` 3,546 B; `rinchen.json` of `distill export` byte-equal to `export convert`'s |
| `export verify` (two good, the old format, one flipped byte) | 1.4 s each; exit 0, 0, 2, 2 |
| `export parity` (source float32, target float32, source int8, a wrong checkpoint) | 5.4, 5.3, 5.3 s, 1.2 s; 200/200 at 7.19e-7, 200/200 at 1.37e-6, 199/200 at 1.14e-3 (not gated); exit 0, 0, 0, 2; the prefix-kept library run 0.0433 |
| the engine over the 400 dev rows (float32, int8, int8_float16) | chrF 98.207, 98.268, 98.197 in 0.07 to 0.09 s; effective compute types `float32`, `int8_float32`, `int8_float32` |
| script masking (500 never trained / 400 trained, off / always / retry) | 96 (731 letters) / 0 / 0 outputs with a forbidden letter; 0 / 0 / 0; 0.05 to 0.10 s per run; `script_retries` 96, `script_retry_improved` 96 |
| the target-tag engine against raw CTranslate2 | mask off: 400 of 400 equal, chrF 58.329; mask `retry`: 2 rows differ (retried), chrF 58.386 |
| `rinchen translate` (5 lines; int8; jsonl; `--strict`; 50 dev lines; closed pipe) | 1.10 s, 1.05 s, 1.02 s, 1.06 s, 1.00 s, 0.98 s; exit 0, 0, 0, 4, 0, 1 |
| the document model: `docify`, `launch --continue-from` (2,500 steps), `export convert` | 1.5 s, 104.3 s (88.1 s in the gate), 3.5 s; 1,194 windows (514, 368, 312 of sizes 2, 3, 4); `document level: derived from the saved config (true)` |
| documents (gate options): 66 documents, 400 sentences | 106 windows, 101 aligned, 18 singles, 5 fallbacks; 400 `ok`; chrF 84.747 (plain rows 88.419 at `4bfaa5e`, 88.494 at `039d1d7`); sizes 325/334, 260/268, 191/202; cap 20: 94 of 106 fall back |
| the three script modes on the gate's document model, 400 plain rows (`off` / `retry` / `always`) | chrF 88.23 / 88.49 / 88.49 and 5 texts different from `off` at `039d1d7`; 88.23 / 88.42 / 81.80 and 358 different at `4bfaa5e` |
| the document path before the fix (the tree at `b6d57a4`) | spike model: 400 `no_output` rows, 382 `ok` + 18 `no_output` documents, chrF 88.465; gate model: 364 + 36, chrF 78.549 |
| `rinchen translate --doc` over three English documents (6, 8, 8 sentences) | 0.89 s, `ok 3` |
| `rinchen serve` (float32): spawn to ready; 64 x 1; 64 x 8; the stop of an idle server | 0.80 s; p50 48.5 / p95 57.7 ms; 76.8 / 115.1 ms; exit -15 at 0.21 s |
| `rinchen serve` (int8): ready; 64 x 1; 64 x 8 | 0.74 s; 53.7 / 58.3 ms; 70.0 / 115.0 ms |
| the stop budget (fake engine on a real uvicorn) | T=2, stuck 8 s: 2.17 s (three runs); T=4, stuck 12 s: 4.18 s; T=20, stuck 30 s: 20.17 s; T=6, a 0.8 s request: 0.87 s; the re-review, stuck 6 s: T=0.5 0.72 s (now refused), 1 1.17 s, 1.5 1.67 s, 2 2.17 to 2.22 s, 3 3.18 s |
| refused settings (thirteen cases: typos, out-of-range values, `auto`, an unknown key, the old format, a flipped byte, no `--model`) | exit 2 each, 0.8 to 1.0 s, nothing listening, the planted value absent |
| `import rinchen`; `rinchen.Translator`; the engine with its libraries; `rinchen.cli.main` | 0.019 s / 11.3 MB; 0.252 s / 47.6 MB; 0.278 s / 55.1 MB; 1.134 s / 170.9 MB |
| `uv build --wheel` offline from a copy | 0.3 s; 211 files, 538,483 B; the registry byte-equal; no `configs/` entry |
| `uv sync --frozen --no-dev --dry-run`, with and without the 22 `--no-install-package` names | 160 packages; 138 packages and none of the torch, triton, cuda or nvidia family |
| `docker --version`; `docker compose -f docker/compose.serve.yaml config -q` | Docker 29.9.3; exit 0 |
| the integration modules once (one pytest process) | 111 passed in 517 s at `4bfaa5e`: parity 13, surfaces 24, serving gate 13, process 15, stop budget 2, D10 gate 44; at `039d1d7`: 112 passed in 604 s: parity 13, surfaces 25, serving gate 13, process 15, stop budget 2, D10 gate 44 |
| `ruff format --check .`, `ruff check .`, `mypy src` | clean (545 files formatted; 195 source files) on this tree, run for this record (the code is unchanged since `4bfaa5e`) |
| the full gate | at `4bfaa5e`: 11768 passed, 1 skipped, 10 deselected (24:01); at `039d1d7`: 11778 passed, 1 skipped, 10 deselected (26 min 51 s) |

**The numbers agree between the spike, the gates and this run** wherever they are not times: the parity counts and deltas, the chrF scores and the `model.bin` sizes, the masking counts, the
document counts, the windows per size, the leak of the D5 converter and the `/metrics` deltas. The differences are times (the stop of 2.17 s against 2.12 to 2.22 s, the serving latencies, the
teachers' 84.7 s against 66.9 s) and two counts that moved with the code since the ledger was written: the wheel holds 211 files (201 at Task 5), `engine.py` and `engine_types.py` have 769 and
581 lines (762 and 574 in the ledger's last reading), and the full dry-run sync leaves 138 packages (the spike wrote 140).

## Where the code lives

`src/rinchen/export/{eole_ct2,record,verified,quantize,ct2}.py`; `src/rinchen/infer/{doc_chunker,source_tokens,script_mask,engine_types,engine,translator}.py` and the lazy
`src/rinchen/__init__.py`; `src/rinchen/serve/{schemas,metrics,batcher,engine,_http,_translate,app}.py`; `src/rinchen/config/{serve,languages}.py`;
`src/rinchen/train/{export_parity,distill_export,distill_saved_config,distill_teacher,distill_generate,checkpoints}.py`; `src/rinchen/cli/{export,distill_export,export_parity,translate,translate_io,serve,main}.py`;
`docker/{Dockerfile.cpu,compose.serve.yaml}`, `.dockerignore`, `configs/serve/{tiny,server,edge}.yaml`, the `force-include` of `pyproject.toml` and the recipes of `justfile`; unit tests
`tests/unit/test_export_{record,verified,verified_reads,quantize,convert,convert_failures,convert_status,eole_ct2,layering}.py`, `test_infer_{engine_translate,engine_open,engine_targets,engine_masking,
engine_documents,engine_reserved_source,engine_threads,request_text_error,script_mask,source_tokens,translator,layering,serve_error_messages,doc_chunker}.py`, `test_train_export_parity{,_refusals}.py`,
`test_cli_{export,export_parity,export_parity_refusals,translate,translate_exits,translate_io,translate_output,serve}.py`, `test_config_{serve,languages}.py`,
`test_serve_{app,app_data_errors,app_decode,app_errors,app_metrics,app_probes,app_shutdown,app_wire,batcher,configs_shipped,layering,metrics,schemas}.py`, `test_docker_files.py`,
`test_compose_file.py`, `test_support_docker_files.py`, `test_packaging.py`, `test_rinchen_toplevel.py`, `test_d11_gpu_scan.py` and the edited D10 tests; property tests
`tests/property/test_{serve_batcher,infer_script_mask,cli_translate}_property.py`; the integration modules of "The STOP gate"; and the helpers `tests/support/{model_exports,engines,error_messages,
translate_cli,parity_cli,fake_parity_eole,serve_app,serve_batcher,fake_serve_process,docker_files,d11_gate,d11_gate_doc,serving_gate,serving_gate_fixtures}.py` (and extensions of `fake_ct2.py`,
`distill_exports.py`, `distill_gate_stages.py`, `distill_teachers.py` and `tests/integration/conftest.py`).
