# 09 - Document-level translation

## Scope

D9 builds what a document-level model needs: the vocabulary invariant that makes `<doc>` and `<sep>` usable by a
model at all (`eole_run.add_reserved_symbols`, `src_vocab_size`), `rinchen train docify` (cut the sentences of a
document into windows of 2 to 4 consecutive sentences joined with `<sep>`, and prepare the continuation of OUR
teacher on them), `rinchen train launch --continue-from` (start a NEW run from one of our own checkpoints, with a
provenance rule, and resume it), the inference chunker `rinchen.infer.doc_chunker` (split a document into windows,
join them, split the model's output back, fall back to sentence-level translation when a window does not line up),
two configs (`configs/eole/teacher_doc_tiny.yaml`, the CPU gate; `configs/eole/teacher_doc.yaml`, the 8-GPU prior)
and the tiny doc model that is the STOP gate. The design, with the rulings T1 to T9 and the section of amendments
made while building, is `docs/superpowers/specs/2026-11-09-d9-document-level-design.md`; the decisions are condensed
in the 2026-11-09 entry "D9 document-level" of `DECISIONS.md`; the upstream facts are in `docs/VERIFY.md` (section
"D9 findings"). The Eole facts come from a spike made before the design
(`.superpowers/sdd/2026-11-08-d9-document-level/spike-findings.md`; Eole 0.6.4, CPU, toy data).

**What D9 is.** Tested machinery. A document is a list of sentences with a `doc_id` and a `seg_idx`; `docify` turns
the train rows into windows (`Sentence one. <sep> Sentence two.`), the continuation trains OUR teacher further on
them (the weights only, a fresh optimiser), and a tiny model made that way decodes `<sep>`-aligned output: one segment
per input sentence for at least 90 % of held-out windows of each size. The chunker puts any document, of any length,
through such a model window by window and still returns exactly one translation per input sentence.

**What D9 is NOT.** Nothing below is a statement about the quality of any document translation.

- **No GPU was used.** This machine has a GeForce RTX 5070 Ti that the owner did not authorise for a run. Every Eole
  process of D9 (the spike, the tests, this runbook's measurements) ran on the CPU with `CUDA_VISIBLE_DEVICES=""` and
  `gpu_ranks: []`. Everything GPU-side (bf16, NCCL, multi-GPU, the 13 GB checkpoint of the real teacher, the length of a
  continuation of the real teacher) is unmeasured.
- **No real documents.** The tiny doc model trains on the templated toy documents of `tests/support/tiny_parallel.py`
  (documents of 4 to 8 sentences built from 37 nouns); the sentences are independent of each other, so the model learns
  the STRUCTURE (one `<sep>` per boundary) long before the lexicon, and the gate measures that and nothing else.
  Window chrF stays low for hundreds of steps after the structure is right. No real document corpus exists.
- **No teacher.** `configs/eole/teacher_doc.yaml` is a table of unmeasured priors; nothing was trained, timed or tuned
  with it. Eole's schema accepts the file, a test shows that, nothing more.
- **No fine-tuning of anything but our own teacher.** A continuation starts from a checkpoint that OUR launcher wrote
  and OUR manifest vouches for (or an average of such), never from another model's weights; `train_from` and
  `reset_optim` stay unknown keys in every config file.
- **No mixing of sentence-level corpora into the doc run** (the spike measured that it lowers the alignment at 1,000
  steps; the real recipe is the operator's and unmeasured), **no target-side tag placement for documents**, **no
  distillation, export, serving or evaluation suite** (D10 to D13): the chunker is a library, not a serving path.

Tags used below: **measured** (run for this runbook or by a test of the build), **spike** (run in the spike, on the
CPU), **source-read, unmeasured** (read from Eole's source, never run). Every number is from a test, the code, the
spike, the ledger of the build (`.superpowers/sdd/2026-11-08-d9-document-level/progress.md`) or a measurement made for
this runbook on 2026-11-09 ("Measured numbers"); a number that was not obtained is not given.

What was checked: the unit and property tests (every new test mutation-checked by hand during the build),
`tests/integration/test_train_tiny_doc.py` (the STOP gate, real `rinchen`, real Eole, CPU), a combined review of
Tasks 1 to 5 (`993f32c..98e42d1`: ready with fixes, one Critical, five Important and twelve Minor findings) and its
fix wave (`D9: fix ...` x 6 and `D9: review minors`, `98e42d1..c2d46b5`), a scoped re-review of that fix wave (approved
with minors, no Critical and no Important finding; "The scoped re-review and the second fix wave" below), a second small
fix wave (`D9: re-review minors`) and the real binary for this runbook.

## Architecture

| Module | Job |
|---|---|
| `rinchen/train/eole_run.py` (extended) | `add_reserved_symbols`; `run_training(reserved_symbols=())`; `predict_tokens(max_length, max_length_ratio)` |
| `rinchen/train/vocabulary_size.py` | `needed_vocabulary_size` (pieces plus one), `require_vocabulary_size`, `require_sizes_to_fit` and `require_unstated_size_to_fit` (the `launch` pre-flight), `EOLE_DEFAULT_VOCABULARY_SIZE` |
| `rinchen/train/doc_source.py` | `read_corpus_rows` (the lazy split filter), `classify_rows`, `require_windowed_split`, the exclusion reasons |
| `rinchen/train/doc_pack.py` | per-document runs of consecutive `seg_idx`, the seeded window sizes, the piece budget, the window frame |
| `rinchen/train/docify.py` | `docify`, `DocWindows`, `DocStats`: the two above and the counts |
| `rinchen/train/doc_corpus.py` | the doc corpora files, the doc `data` section, the validation windows, the weights, the rendered continuation config |
| `rinchen/train/continuation.py` | `verify_source` (provenance of a source checkpoint), `check_continuation` (the checks of the config), `SourceProvenance` |
| `rinchen/train/checkpoints.py` (extended) | `ContinuedFrom`, `launch.json` schema version 2, the identity of a continuation, `run_directory_above` |
| `rinchen/train/launch.py` (extended) | `launch(continue_from=)`, the reserved-symbol top-up, the vocabulary-size pre-flight |
| `rinchen/train/eole_cfg.py` (extended), `eole_cfg_keys.py` | the `continues=` and `resumes_continuation` rules; the key lists moved out of `eole_cfg.py` (it was over 900 lines) |
| `rinchen/infer/doc_chunker.py` | `chunk_document`, `join_window`, `split_output`, `translate_document`, `ChunkStats` (callables only) |
| `rinchen/cli/train_docify.py` | `rinchen train docify` |
| `rinchen/cli/train_runs.py` (extended) | `launch --continue-from`; the guard of `average` |

Layering as D7 and D8: `train` never imports `script_conv` or `cli`; `infer` imports nothing from `train`, so `<doc>`,
`<sep>`, the joint ` <sep> ` and the two fixed pieces are defined three times (`spm.symbols`, `train.doc_source`,
`infer.doc_chunker`) and a unit test pins the three equal.

## The vocabulary invariant

**The hazard (VERIFY `D-EOLE-VOCAB-NOT-SPM`).** Eole's vocabulary is not the SentencePiece vocabulary. It is the
`src_vocab` file that `eole build_vocab` writes (the tokens the training data SHOWED, plus the `prefix` specials), cut
at `src_vocab_size` (default 32,768) and padded to a multiple of 8. A symbol the data never contained is not in the
vocabulary of the model. The D8 tiny teacher's `vocab.json` had 232 entries of the 640 pieces and neither `<doc>` nor
`<sep>`; continuing it on document data with that vocabulary is SILENT: both symbols become `<unk>`, the alignment of
the output is 0.00 and all 768 outputs hold an unknown, with no error and no warning (spike).

**The rule.** Every Rinchen vocabulary holds every D3 reserved symbol (`spm.symbols.user_defined_symbols()`: 118
symbols, the 10 direction tags, the 8 structural ones `<bt>`, `<ft>`, `<t>`, `</t>`, `<r>`, `</r>`, `<doc>` and `<sep>`,
and the 100 numbered placeholders). After `eole build_vocab`,
`eole_run.add_reserved_symbols(vocab_file, symbols)` appends the missing ones as `symbol<TAB>1` lines and returns
what it appended.

- It is called by `run_training(reserved_symbols=...)` (D7's ablation passes the symbols; D5's script converter
  passes none and its model, vocabulary and every number are unchanged) and by `launch` right before `eole train`
  for a run that is about to train (after `eole build_vocab`, or on an existing `vocab.src`). A run found finished,
  `--write-only` and a refused launch leave the file as it is. **`command.sh` does not add the symbols**: a run
  started by hand from it trains on the vocabulary as Eole wrote it, and the next `launch` of that run adds them,
  which is too late for its checkpoints (documented in the `launch` docstring).
- The bytes of the file stay as they are: no line is changed or reordered, a CRLF file keeps its CRLF (the appended
  lines end in LF, which Eole reads beside them), an LF is added first only when the file does not end with one,
  a file with no count column gets bare symbols, nothing is written when nothing is missing (a second call appends
  nothing), the permission bits are kept, and a vocabulary that is a symbolic link becomes a regular file. A symbol
  counts as present only when it is the whole token of a line.
- **An empty vocabulary is refused** (`DataError`: no token, or blank lines only): the first version turned a
  truncated `build_vocab` into 118 lines of reserved symbols, and a model trained on that reads every real piece as
  `<unk>` (review I4).
- The symbols go at the END of the file, where a size cut would take them first, so the size is set with them:
  `src_vocab_size` is the SentencePiece model's `GetPieceSize()` plus ONE, because Eole adds `<blank>` (its pad token,
  no SentencePiece piece) to every vocabulary and the size counts it. `ablate` and `ablate --prepare-only` set it
  when the base config states none and refuse a smaller explicit value (`ConfigError` naming both numbers); `launch`
  refuses a config that STATES a `src_vocab_size` or `tgt_vocab_size` below it, before anything is written (review I5):

```
error: invalid Eole config: src_vocab_size: 100 is below 641, what the vocabulary of the SentencePiece model
$W/ws/in/spm/toy.model needs (640 pieces and Eole's pad token): Eole would cut the reserved symbols that are added at
the end of its vocabulary file
```

  A config that states NO size gets Eole's default, 32,768 (`vocabulary_size.EOLE_DEFAULT_VOCABULARY_SIZE`, taken from
  `eole/config/data.py`, `VocabConfig`; a test compares it with the installed Eole), and since the re-review (M1,
  `D9: re-review minors`) `launch` refuses that too when the model needs more: a `ConfigError` that names the key and
  the value to state, before anything is written (the real binary, 2026-11-09, a model of 128,000 pieces, path
  shortened):

```
error: invalid Eole config: src_vocab_size: not stated, so Eole's default of 32768 applies, which is below 128001, what the
vocabulary of the SentencePiece model $W/spm/big.model needs (128000 pieces and Eole's pad token): Eole would cut the
reserved symbols that are added at the end of its vocabulary file; state `src_vocab_size: 128001` at the top level of
the config
```

  `launch` never sets the size itself, and has no option for it (its per-run values are `--world-size` and
  `--gpu-ranks` only): the operator states it at the top level of the config it passes. `ablate --prepare-only` already
  does; for a config written by hand, or made by `docify --config`, it goes in the base file (a copy of
  `configs/eole/teacher_doc.yaml`, whose header says so). `tgt_vocab_size` is not asked for when `share_vocab: true`
  (Eole builds the target vocabulary from the source one then, and every shipped config shares). An unstated size whose
  model is not a file is left to Eole, which names the missing model when it builds the vocabulary (this keeps the many
  tests that name `spm/toy.model` without writing it); a model that is a file and cannot be loaded is a `ConfigError`,
  as it is for a stated size. The D5 converter configs state sizes equal to their piece count (`converter.yaml` 512,
  `converter_tiny.yaml` 200); they run through `run_training` and never through `launch`, and must stay off it.

**What it changed.** The tiny runs of D7 and D8 changed, because their vocabularies grew and the sizes follow. Measured
on 2026-11-09 through the real binary on the toy data (the "before" values are those recorded in runbooks 07 and 08
on 2026-11-08, code `3e357ae`):

| The tiny teacher of D8 (200 steps) | before D9 | after |
|---|---|---|
| `vocab.src` lines | 225 | 339 (4 of the 118 symbols were there, 114 appended) |
| checkpoint `vocab.json` entries | 232 (4 specials, 225, 3 padding) | 344 (4 specials, 339, 1 padding) |
| `src_vocab_size` of the rendered config | none | 641 |
| weights of `step_200` | 3,289,000 B, SHA-256 `32a1c92e42d8...` | 3,318,120 B, SHA-256 `6f5e836c7553...` |
| training `xent`, first and last report | 5.44 and 2.14 (a 60.7 % fall) | 5.81 and 2.13 (a 63.3 % fall) |
| validation perplexity, steps 50 to 200 | 83.6869, 16.6115, 10.1611, 8.81733 | 106.851, 18.3535, 10.4948, 8.73594 |

The D7 ablation moved too (the dev chrF of the source-tag variant: eng-khk 98.89, khk-eng 99.23, khk-rus 97.76,
rus-khk 96.25 before; see "Measured numbers" for the new values and runbook 07 for both); the tests pin thresholds,
not these values. Every earlier ablation variant's fingerprint changes (its config gained `src_vocab_size`), so a
rerun into an old output directory retrains it.

## Commands

The runs below were made on 2026-11-09 on the machine of "Measured numbers", through the real binary
(`.venv/bin/rinchen`) with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`), on the toy data, once each (the flow is the
STOP gate's, `tests/integration/test_train_tiny_doc.py`). `$W` is a scratch directory, `$TEACHER` the run directory
of the sentence-level tiny teacher (`$W/ws/teacher_out/variants/t0.5_bt0.5_src_s1234/run`, made by `ablate
--prepare-only` and `launch` exactly as in runbook 08 with `configs/eole/teacher_tiny.yaml`, 200 steps) and `$DOC`
the output directory of `docify`. The machine was loaded by other work while these ran (load average 8 to 22 on 20
cores), so every wall time is longer than an idle machine's: the 2,500-step continuation took 107 s here, and 87 s
idle in the build.

### Average the teacher

```bash
uv run rinchen train average --checkpoint-dir $TEACHER/ckpt --last 2
```
```
averaged: 2 checkpoints (steps 150, 200)
output: $TEACHER/averaged
weights sha256: 081f9d54345ea5f10b9ae514060b06a638bb11310d540210500bd5cb77e23513
manifest: $TEACHER/averaged/manifest.json
```

1.43 s. The averaged directory is the natural source of a continuation (it has no `optimizer.pt`, so every
`reset_optim` value acts as `all` in Eole, spike). `average` now refuses a checkpoint directory that is not inside a
run directory with a readable `launch.json` (see "The `average` guard").

### `docify`, the dry run

```bash
uv run rinchen train docify --corpus $W/ws/in/docs.parquet --spm-model $W/ws/in/spm/toy.model \
    --config configs/eole/teacher_doc_tiny.yaml --out-dir $DOC --dry-run
```
```
corpus files: 1
max pieces: 126
rows read: 4000
rows in windows: 3380
windows: 1194
windows of size 1: 0
windows of size 2: 514
windows of size 3: 368
windows of size 4: 312
excluded not_train: 400
excluded filtered: 0
excluded excluded_origin: 0
excluded no_doc_id: 0
excluded no_seg_idx: 0
excluded empty: 0
excluded line_break: 0
excluded contains_sep: 0
excluded duplicate_seg_idx: 0
excluded too_long: 0
excluded gap_single: 0
excluded tail: 220
pieces src: min 12, mean 25.124, p95 45, max 52
pieces tgt: min 10, mean 23.124, p95 43, max 50
dev windows: 136
corpus eng_Latn-khk_Cyrl.doc: windows 392, weight 33
corpus khk_Cyrl-eng_Latn.doc: windows 392, weight 33
corpus khk_Cyrl-rus_Cyrl.doc: windows 205, weight 17
corpus rus_Cyrl-khk_Cyrl.doc: windows 205, weight 17
nothing was written
```

2.31 s; nothing, not even `$DOC`, was created. The corpus file holds the toy train and dev documents (3,600 and 400
rows), no test row. Of the 4,000 rows, 3,600 were train rows and 3,380 of them are in a window, 220 are a `tail` (a
run of consecutive sentences that ended with fewer than two left over) and the 400 dev rows are `not_train` here (the
dev windows are cut by a second pass, `dev windows: 136`: the validation corpus). The piece counts include the
direction tag and `<doc>` on the source side; the longest source is 52 pieces, so the budget of 126 never bites on the
toy.

### `docify`

The same command without `--dry-run`: the same statistics, then

```
corpora: 4 (reused 0)
validation windows: 136
config: $DOC/config.yaml
manifest: $DOC/manifest.json
```

2.73 s. It wrote (sizes in bytes) `data/doc/<src>-<tgt>.doc.src`, `.tgt` and `.json` for each of the four directions
(36,278 and 36,398 for `eng_Latn-khk_Cyrl.doc.src` and `.tgt`, 177 for the sidecar), `valid.src` (21,579) and
`valid.tgt` (19,131), `config.yaml` (3,639) and, last, `manifest.json` (8,050; stage `train_docify`). One window of
`eng_Latn-khk_Cyrl.doc` as the files hold it (the tag and `<doc>` are NOT in the text files: they are Eole `prefix`
configuration, so the same files serve any placement):

```
src: The person and the bird. <sep> The milk, the day and the school. <sep> The car and the moon. <sep> The milk and the hand.
tgt: Хүн ба шувуу. <sep> Сүү, өдөр ба сургууль. <sep> Машин ба сар. <sep> Сүү ба гар.
```

and the first line of `valid.src`, pre-tokenised (Eole reads the validation corpus as it is; the underscore stands
for SentencePiece's word-boundary piece U+2581, which is written as a character of its own in the file and stands
alone in front of a `<sep>` written in text):

```
<2khk_Cyrl> <doc> _Th e _window _ and _the _wind . _ <sep> _Th e _water _ and _the _eye .
```

### Continue the teacher

```bash
uv run rinchen train launch --config $DOC/config.yaml --run-dir $DOC/run \
    --continue-from $TEACHER/averaged --threads 2 --timeout 600
```
```
continued from: averaged $TEACHER/averaged
status: completed
steps: 2500
resumed from: none
attempt: 1
checkpoint: $DOC/run/ckpt/step_2500
log: $DOC/run/launch/attempt-1.log
command.sh: $DOC/run/command.sh
manifest: $DOC/run/manifest.json
```

107.14 s on the loaded machine, including `eole build_vocab`, 2,500 steps, five validations and five saves. The
first attempt's config (`$DOC/run/launch/train-1.yaml`) is the one `docify` wrote plus `training.train_from:
$TEACHER/averaged` and `training.reset_optim: all`; the log says `Loading metadata from $TEACHER/averaged`, logs each
override (`Option: reset_optim, value: all, overriding model: none`) and `Keeping metadata vocabulary` (the
vocabulary is the teacher's; the new `vocab.src` only has to exist). The curve: training `xent` 3.07 at step 50 and
0.97 at step 2500, validation perplexity 7.34474, 5.99236, 3.62612, 3.28613, 3.21243 at steps 500 to 2500; no
non-finite figure. `launch.json` is schema version 2 and records the source (below). The teacher directory is
byte-identical before and after (a tree digest). `command.sh` starts with `# It continues "$TEACHER/averaged"
(training.train_from of the config it names, with reset_optim: all: its weights, a fresh optimizer).`

The same continuation was run a second time in another run directory: the weights of every checkpoint (steps 500,
1000, 1500, 2000, 2500) have the same SHA-256 as the first run's, so the training is bit-for-bit repeatable on this
machine with 2 threads (119.45 s on the loaded machine).

### Launch again, resume

The same command again: `status: already_complete`, `attempt: 0`, `log: none`, nothing written (the gate asserts less
than 30 s and an unchanged tree). A finished continuation is judged by its record and does not need its source any
more; extending it (a larger `train_steps`), resuming it or `--write-only` do, and the source is verified again then.
A SIGTERM during the continuation ends the launcher with 143; the next `launch --continue-from` resumes from the
run's OWN newest complete step with `training.train_from: $DOC/run/ckpt/step_<N>` and `reset_optim: none`, and the
log of that attempt continues at the step after the checkpoint (the gate: SIGTERM after `step_250` of a 750-step
continuation, resumed from 250, reports 300 to 750).

### Refusals (real binary, 2026-11-09)

Each exits 2 in about one second with Eole not started and nothing left in the new run directory but the lock file; the
source is only read (its tree digest is the same afterwards). The messages are, with the paths shortened:

- A step directory with no run around it (`launch.json` is looked for in the three nearest parents): `error: cannot continue from $W/sources/step_alone/step_200: no launch.json within 3 levels above it, so it is not a checkpoint of a run that `rinchen train launch` wrote`
- One byte of the weights flipped: `error: cannot continue from $W/sources/flipped/averaged: model.00.safetensors differs from the manifest $W/sources/flipped/averaged/manifest.json: its SHA-256 is 345538bad9bf... and the manifest lists 081f9d54345e...; the file changed after the manifest vouched for it`
- A teacher whose `vocab.json` lacks `<doc>` (made by rewriting the file and its manifest hash so only this check can fire): `error: cannot continue from $W/sources/no_doc/averaged: vocab.json of the source lacks <doc>: the teacher predates the vocabulary invariant (a vocabulary file made since then holds every reserved symbol), so it would read them as <unk>`
- A config that omits an architecture key: `error: cannot continue from $TEACHER/averaged: model.heads is not stated in the config: a continuation must say which architecture it believes it continues`
- A config whose SentencePiece model is another file than the teacher's copy (a toy model of 500 pieces against the 640 of the source), reported for both sides: `error: cannot continue from $TEACHER/averaged: transforms_configs.sentencepiece.src_subword_model ($W/other_spm/toy.model, SHA-256 466adbb07c3c...) is not the SentencePiece model the source was trained with (toy.model in the source, SHA-256 8d5496f62c11...): the vocabulary of the source is the pieces that model gave its data, and a piece it lacks would be read as <unk>; transforms_configs.sentencepiece.tgt_subword_model (...)`
- Two step directories laid out by hand, averaged: `` error: cannot average $W/foreign/ckpt: no launch.json within 3 levels above its step directories; only a checkpoint directory inside a run directory that has one (`rinchen train launch` writes it) is averaged, because an average is accepted as the model to continue from ``; nothing was written. (Before the second fix wave the message went on "so it is not the checkpoint directory of a run that `rinchen train launch` wrote; only those are averaged", which claimed more than the guard checks.)
- `docify --max-pieces 500` over the tiny base: `error: --max-pieces 500 is above what the base config keeps: filtertoolong drops a source over src_seq_length 128 and a target over tgt_seq_length 128 minus 2, so a window may have at most 126 pieces; the longer ones would be dropped without a word`

## Document windows (`docify`)

**The input.** Corpus Parquet files with the D1 schema; the columns `docify` reads are `id`, `src_lang`, `tgt_lang`,
`src`, `tgt`, `origin`, `corpus`, `split`, `filter_reason`, `doc_id` and `seg_idx`. D4's mining output carries
`doc_id` (the stem of the source file) and `seg_idx`. `read_corpus_rows` reads the files by name (never as a glob)
and loads only the rows of the splits it is asked for: the split filter is part of the lazy plan, as D7's is, so the
text of a test row is never decoded. Each other-split row stays in the frame as one row with only its `split`
filled, so that it is counted (`not_<split>`) and `rows_read` is the number of rows of the files (review I1).

**Which rows make windows.** Only authentic (`auth`) rows of the split with a `doc_id`, a `seg_idx` and usable text.
Every row of every input is counted once, under the first reason that applies, in this order
(`doc_source.exclusion_reasons(split)`): `not_<split>`, `filtered` (it has a `filter_reason`), `excluded_origin`
(`ft`, `synth_llm`, `synth_nmt` and `bt`), `no_doc_id` (null or blank), `no_seg_idx`, `empty`, `line_break` (D7's two
rules), `contains_sep` (a sentence with the literal `<sep>` would be a false boundary), then the reasons that need
the other rows of a document or the piece counts: `duplicate_seg_idx` (the first row by id, source and target wins),
`too_long` (a sentence that alone exceeds the budget), `gap_single` (a run of consecutive sentences shorter than
`min_size`, so no window at all) and `tail` (what remains at the end of a run when fewer than `min_size` sentences are
left). `rows_read = rows_in_windows + the sum of the excluded counts`, always (a property test).

**The key of a document.** `(src_lang, tgt_lang, corpus, doc_id)`. D4 names a document after the file it came from,
so two corpora can both hold a document `en` numbered from 0; before the review's fix (C1) such documents were
joined or interleaved into one window. A window now never holds sentences of two corpora, and the two directions of
one document get the same windows.

**Sizes.** Inside a document the sentences are sorted by `seg_idx`; a window holds only consecutive `seg_idx` (a gap,
which D4's filters can leave, or any left-out sentence ends a run). The size of a window is drawn from `[min_size,
max_size]` (default 2 to 4) by `sha256(seed, doc_id, first seg_idx)`, so the windows do not depend on the order of
the rows or of the files; a window is shortened, down to one sentence, while its source (the tag and `<doc>` count 2
pieces) or its target is over `max_pieces`, both measured on the JOINED text, so the piece in front of a `<sep>`
counts. A window of the toy was between 12 and 52 source pieces; real sentences are about four to five times longer
(spike), so the budget will bite on real data and `windows of size 1` is not zero there. Source and target of a
window have the same number of `<sep>`.

**Weights and files.** The weights of the four corpora follow D7: the probability of a direction is its window count
to the power of `--temperature` (default 1.0), and the integer weights (at least 1) add up to `--weight-total`
(default 100): 33, 33, 17, 17 above. The validation corpus is the same windowing of the DEV rows with the fixed seed 0
(`DEV_WINDOW_SEED`), at most `--dev-max-windows` (default 1000) per direction, those with the smallest ids, written as
token lines. The rendered config is the base config with `data`, `src_vocab`, `transforms_configs.sentencepiece` and
`training.model_path` laid over it; the base must hold no corpora of its own, no `train_from` and no `reset_optim`
(`check_continuation_base`).

**Options.** `rinchen train docify --corpus C [--corpus C2 ...] --spm-model M --config B --out-dir O [--min-size 2]
[--max-size 4] [--max-pieces N] [--temperature 1.0] [--seed 1234] [--dev-max-windows 1000] [--weight-total 100]
[--dry-run]`. `--max-pieces` has a default AND a ceiling: `min(src_seq_length, tgt_seq_length - 2)` of the base
config's `filtertoolong` (Eole's default 192 for a key it leaves out; 126 for the tiny base, 510 for the prior); a
larger value is a `ConfigError`, because Eole would drop the longer windows without a word. Exit codes: 0; 2 for a bad
option, a missing or invalid config, SentencePiece model or output directory (`$` or a glob character in it), or a
budget above the ceiling; 3 for a missing or unreadable corpus, another stage's manifest in the output directory and an
input with no window to cut (the message names the reasons; also for a dry run); 1 when the output cannot be
written. The manifest (stage `train_docify`, written last) lists the base config, the corpus files and the
SentencePiece model as inputs, each hashed before it was read, and its metrics are dotted as D7's:
`count.<name>` for the train rows and `count.dev.<name>` for the dev rows (`count.excluded.<reason>`,
`count.windows_by_size.<n>`, `count.pieces.src.max`), `max_pieces`, `min_size`, `max_size`, `temperature`,
`weight_total`, `dev_max_windows`, `corpora.files`, `valid.windows` and `weight.<corpus>`.

## The continuation

### What may be continued

`rinchen train launch --continue-from SOURCE` starts a NEW run from one of our own checkpoints. SOURCE is one of two
kinds of directory, found by `continuation.verify_source`:

- **step**: the FINAL `step_<N>` directory of a run that `rinchen train launch` finished. The run directory (the one
  with `launch.json`, at most three levels above) holds a `manifest.json` of stage `train_launch` with status ok that
  lists the four files of THIS step directory (`model.00.safetensors`, `optimizer.pt`, `config.json`, `vocab.json`)
  with the SHA-256 they have now. The manifest names the final step only, so an earlier step of the same run is
  refused, and so is a step directory renamed to another number.
- **averaged**: a directory `rinchen train average` wrote: it holds `averaged_from.json` and a `manifest.json` of stage
  `train_average` listing its weights, `config.json`, `vocab.json` and `averaged_from.json`. It must have no
  `optimizer.pt` (a file Eole would load that no manifest vouches for) and no other `model.*.safetensors` than the one
  the manifest lists (Eole loads every shard of the directory). The two accounts of what it was made from must agree
  (review I2): `averaged_from.json` names at least two checkpoints, the manifest lists the weights of the same ones
  (same directories, same SHA-256, same order) as inputs, and the weights the record names are the vouched ones.

Nothing else is a source: no manifest, the manifest of another stage, a status that is not ok, a file the manifest does
not list or whose bytes differ, an incomplete directory (`checkpoints.is_complete`), a symbolic link, a path with `$`
or a glob character (Eole expands the first and globs the checkpoint directory), a directory inside the new run, one
that holds it, or a vocabulary file of the new run that would be written inside it. An unusable source is a
`ConfigError` (exit 2) that names the file and what is wrong and never quotes a file's content; a manifest, run
record or vocabulary that cannot be read is a `DataError` (exit 3). The manifest lists each file by the path it had
when it was written and a run may have been moved since, so a file is matched to its entry by NAME, a name two entries
share is refused, and for a step the entries must lie in a directory named like the source. The SHA-256 of each file is
taken ONCE, before any check reads the file, and kept: `check_continuation` reads `config.json` and `vocab.json` only
through that digest.

**This is a guard against mistakes, not against a forgery.** Someone who can write a manifest and a record by hand can
make any directory pass. The hash is not atomic with Eole's own read of the files either (a writer that swaps a file
after the hash is not caught), and what is hashed is the file as it is on disk when `rinchen` runs; the manifests of
our own commands are the only evidence of origin there is.

### The `average` guard

`rinchen train average` is the one command that can turn several step directories into a source, so it refuses a
checkpoint directory with no `launch.json` that parses within three parent directories of its step directories (review
I2; exit 2, nothing written, see "Refusals"). Without it two step directories laid out by hand became a valid
`--continue-from` source after one command. **That file is all that is checked**: a hand-made `step_<N>` placed inside a
real run directory passes, and so does a `launch.json` written by hand; the guard is against a mistake, not against
forgery. The parents are those of the path as given (`os.path.abspath`, no link is followed), see "Known limits" 15.

### What is recorded

`launch.json` of a plain run stays schema version 1, byte for byte as before D9. A continuation writes version 2 with
`continued_from`: `kind` (`step` or `averaged`), `path` (as given, made absolute), the SHA-256 of the weights,
`config.json`, `vocab.json` and of the source's manifest, `optimizer_sha256` (a step) or null, `manifest_stage`, `step`
or null and `source_run` or null. The real record of the run above (paths shortened, hashes cut):

```json
{
  "schema_version": 2,
  "identity": "cdacbc29ab36...",
  "world_size": 1,
  "gpu_ranks": [],
  "created": "2026-11-09T04:40:52+00:00",
  "continued_from": {
    "kind": "averaged",
    "path": "$TEACHER/averaged",
    "weights_sha256": "081f9d54345e...",
    "optimizer_sha256": null,
    "config_sha256": "cc1b403bb860...",
    "vocab_sha256": "9d444b846bc2...",
    "manifest_sha256": "13f831a87efa...",
    "manifest_stage": "train_average",
    "step": null,
    "source_run": null
  }
}
```

The **identity** of a continuation (the SHA-256 of the effective config without `train_steps` and `train_from`, as D8)
also covers the kind of the source and the SHA-256 of its weights, `config.json` and `vocab.json`, not where it lies
nor its manifest: the same config pointed at other files is another run, and a source that moved is the same source.
The manifest of the new run (`train_launch`) lists the source's files and its manifest among its inputs, with the
digests the verification took (they are not hashed again, and no size is recorded: the source may be gone when a
finished run's manifest is written again), and has the metrics `continued_from.kind` (0 for a step, 1 for an averaged
model) and `continued_from.step`. Eole's own `config.json` of the new checkpoints records only the PATH of the source
(`train_from`) and no hash of it.

### The checks before the first attempt

`check_continuation(cfg, provenance)` (T6) refuses the config that would continue the source, naming every failed
check in one message:

1. The `model` section states every key of the architecture guard (`hidden_size`, `embeddings.word_vec_size`,
   `heads`, `transformer_ff`, `encoder.layers`, `decoder.layers`, `embeddings.position_encoding_type`); the
   `config.json` of the source records each, and `checkpoints.architecture_mismatch` finds no key they differ in. A
   continuation must say which architecture it believes it continues, because Eole merges a new `model:` section over
   the checkpoint's without a check (a wrong `heads` loads silently, VERIFY `D-EOLE-TRAIN-FROM-FOREIGN`).
2. The optimiser options `learning_rate`, `warmup_steps`, `decay_method`, `optim` and `adam_beta2` are stated in
   `training`: Eole inherits an omitted one from the checkpoint. **Only these five are checked** (see "Known limits"):
   every other option the config omits is inherited too, and the shipped doc configs state them all.
3. The SentencePiece model the config states for both sides has the bytes (SHA-256) of the copy Eole saved into the
   source directory, the file the source's `config.json` names (review I3): the vocabulary of the source is the pieces
   that model gave its data, so another model, the 64k student's instead of the 128k teacher's, say, would read pieces
   the teacher lacks as `<unk>` without a word. A source with no recorded or no saved copy cannot be compared and is
   refused. The comparison reads the path as given in the config, so a RELATIVE path is read from the working
   directory of `rinchen` (`docify` writes absolute paths). The manifest does not vouch for the copy and `launch.json`
   does not record its hash.
4. The `vocab.json` of the source holds `<doc>`, `<sep>` and every token written like a symbol (`<...>`) in the
   prefixes of the data corpora, on both sides. A teacher made before the vocabulary invariant lacks the first two and
   is refused with the message that says so (see "Refusals").

The launcher sequences these: it takes the lock of the run before it looks at the source, writes nothing but the lock
file until every check has passed, and a refused launch leaves the run directory as it found it.

### The attempts and `reset_optim`

The first attempt's config is the effective config plus `training.train_from: <SOURCE>` and `training.reset_optim:
all`, and nothing else differs from a fresh start: the weights load, the Adam state is fresh and the step counter and
the Noam schedule count from 1. **Every later attempt resumes the run's OWN newest complete step like any run, with
`reset_optim: none` STATED** (the first design said "drop the key"; the real Eole showed that dropping it is not
enough). Eole fills a key the YAML omits from the config recorded in the checkpoint, and the checkpoints of a
continuation record `reset_optim: all`, so an omitted key made the resume load only the weights and restart at step 1
with the warm-up again (measured in Task 3: a "resume" of step 40 that ran 200 steps from step 1); a resume that keeps
`all` does the same (spike). The validator accepts exactly `all` with `continues=` (and `train_from` exactly the
recorded source), exactly `none` with `resumes_continuation=True`, and neither key anywhere else; `update_vocab` and
`keep_states` are never accepted. A plain run's resume config carries no `reset_optim` at all. An attempt that saved
no step of its own starts from the source again (it is still the first attempt in effect), and `train_steps` of a
continuation may be raised like any run's.

What `reset_optim` does (spike, CPU, Eole 0.6.4; the teacher's learning rate 0.5 with 100 warm-up steps, the new
YAML's 0.25 with 20; VERIFY `D-EOLE-RESET-OPTIM`):

| value | Adam moments | training and decay step | options | evidence |
|---|---|---|---|---|
| `none` (default) | from the checkpoint | from the checkpoint (201) | ALL from the checkpoint's `config.training`; the new YAML's are ignored | `Step 201/240 lr 4.41e-03` (the old 0.5 and 100); `train_steps` is absolute |
| `all` | fresh | restart at 1 | the checkpoint's explicit fields are overridden by the NEW YAML's explicit fields; omitted ones come from the checkpoint | `Step 1/40 lr 3.49e-04` (the new 0.25 and 20), peak 5.0e-3 at step 39; `train_steps` counts from 1 |
| `states` | fresh | kept (201) | from the checkpoint | `Step 201/240 lr 4.41e-03`; Adam step 40 |
| `keep_states` | intended: kept | intended: reset | intended: new | CRASH `KeyError: 'training_step'` (source-read: `optimizers.py:388-390`, `433`) |

An averaged directory has no `optimizer.pt`, so every value acts as `all` with identical numbers (step 1 xent 2.39,
spike). `get_metadata` loads `optimizer.pt` whatever `reset_optim` says (source-read), so a continuation from a step
directory of the real teacher reads 8.8 GB for nothing; the averaged source avoids it.

## The chunker

`rinchen.infer.doc_chunker` is pure functions over three callables: `count_pieces(text) -> int` (the caller's
SentencePiece model), `translate_window(joined_text) -> text` (the doc model, `<sep>` in and out) and
`translate_sentence(sentence) -> text` (the sentence-level model, for the fallback). It knows nothing of Eole or
SentencePiece and what the callables raise is never caught.

**Chunking.** Empty or whitespace-only sentences pass through with the empty translation, call no translator and END
the window they are in (the sentences on either side of one are never in one window, as `docify` ends a run at any
sentence it leaves out; the first version only left a blank out of the window, review M4). A sentence is one line:
`one_line` turns every run of whitespace (line breaks, no-break and ideographic spaces included) into one space, for
both translators. The rest are chunked greedily into windows of at most `max_sentences` (default 4) sentences whose
source costs at most `max_pieces` pieces, where the cost is 2 (the tag and `<doc>`) plus `count_pieces` of the joined
text, so what a `<sep>` costs is the counter's own (two pieces for SentencePiece: the word-boundary piece U+2581 in
front of it). A sentence that does not fit alone is an `oversize` chunk, and a sentence the windows around it left
alone is a `single` chunk; both go to the sentence translator, never to the window model, which was trained on
windows of at least two. A sentence that holds the literal `<sep>` refuses the whole document (`DataError` naming the
index, before anything is translated).

**Splitting and the fallback.** A window's output is split at every `<sep>` with `re.split(r"\s*<sep>\s*",
output.strip())` (robust to `A. <sep> B.`, to the bare `A.<sep> B` that decoding gives, to two in a row, a leading and
a trailing one: the segment count is the `<sep>` count plus one). A count that differs from the number of sentences in
the window, or a right count with an EMPTY segment (a sentence with text and no translation is a sentence lost), makes
the WHOLE window fall back to sentence-level translation of each of its sentences. Nothing is repaired in part: the
fallback is structural, and the spike measured that it does not repair content (composite chrF 27.3 against 26.1 with
the misaligned windows blanked and 98.65 for pure sentence level, on a model not yet trained long enough). The
result, `DocumentTranslation`, has exactly one translation per input sentence, in order, and `ChunkStats`:
`sentences`, `chunks` (`windows + singles + oversize + empty`), `windows`, `aligned`, `fallbacks` (the indices of the
windows that fell back, counted from 0 in document order), `singles`, `oversize`, `empty`.

**What it cannot do (documented, left open on purpose; D11).**

- The budget measures the SOURCE only. `eole predict` caps the OUTPUT at `max_length` (250) and `max_length_ratio` (2)
  pieces and the cut is silent (`D-EOLE-PREDICT-TRUNCATION`), so a window whose output was cut inside its LAST segment
  still has the right segment count and is counted aligned, with a truncated last translation. A cut that drops a
  `<sep>` is caught, one that keeps all of them is not. `predict_tokens(max_length=, max_length_ratio=)` lets a caller
  state both (the gate passes 128 and 0).
- A `translate_sentence` result that itself holds `<sep>` passes through unchecked.
- Only `<sep>` is guarded in the input. A literal `<doc>` or direction tag inside a sentence is not: SentencePiece
  reads it as the symbol, in `docify` and in the chunker alike.
- The fallback does not repair content, only the sentence count.

## Length limits and the predictor's silent truncation

Three limits count PIECES, and the unit has to be the same across them (VERIFY `D-EOLE-FILTERTOOLONG`,
`D-EOLE-PREDICT-TRUNCATION`):

- **Training.** `filtertoolong` drops an example when its source (after `sentencepiece` and `prefix`: the tag and
  `<doc>` count) is longer than `src_seq_length` or its target longer than `tgt_seq_length - 2`; silent apart from a
  statistics line in the log. `docify` therefore cuts to `min(src_seq_length, tgt_seq_length - 2)` and refuses a larger
  `--max-pieces`. A corpus whose `transforms` lack `filtertoolong` has no limit but the 5,000 sinusoidal positions.
- **Inference.** `eole predict` has no source limit (a 55,520-piece source gave `SequenceTooLongError`) and caps the
  output at `int(min(max_length, padded_batch_source_length x max_length_ratio + 5))` pieces; `max_length_ratio: 0`
  turns the ratio off. A cut is silent: `max_length: 10` gave exactly 10 pieces ending in the middle of a word.
- **Model.** A toy doc model that saw windows of at most 4 sentences emitted only 4 segments for a 20-sentence source
  (spike): it does not extrapolate, so the chunker's `max_pieces` has to stay within the length the doc model trained
  with (`src_seq_length` of the base config: 128 for the tiny base, 512 for the prior; `docify` used the same
  numbers, less 2 for the target).

## The STOP gate

`tests/integration/test_train_tiny_doc.py` (real `rinchen`, real Eole, CPU, GPU hidden, `-m subprocess`, every
subprocess bounded by 900 s). Module fixtures run the expensive commands once: **T** the sentence-level tiny teacher
(200 steps) through `ablate --prepare-only` and `launch`, then `average --last 2`; **D** `docify` over the toy train and
dev documents (the test documents are not in the file); **C** `launch --continue-from <averaged>` for the 2,500 steps of
`teacher_doc_tiny.yaml`; **A** every window of 2, 3 and 4 consecutive sentences of every dev document (804 windows:
334, 268 and 202; they overlap, and the model never saw the dev documents) translated by each checkpoint through
`predict_tokens(max_length=128, max_length_ratio=0)`; **K** the chunker over the toy TEST documents (read here on
purpose, for the gate, and nowhere else); **R** five refusals; **S** a SIGTERM and a resume; **B** the finished run
launched again.

The gate: the alignment rate of the final checkpoint is at least 0.9 for EACH window size, where a window is aligned
when its output has one segment per sentence and none is empty (the chunker's rule). Measured again on 2026-11-09
(the fix wave's numbers under that rule; the build's first table, which counted a window with an empty segment as
aligned, differs from these in two cells, see the note):

| checkpoint | size 2 (334) | size 3 (268) | size 4 (202) |
|---:|---:|---:|---:|
| 500 | .9641 | .1716 | .0000 |
| 1000 | .9521 | .8470 | .8861 |
| 1500 | .9731 | .9664 | .9158 |
| 2000 | .9880 | .9664 | .9109 |
| **2500 (the gate)** | **.9820** | **.9776** | **.9406** |

The predict over the 804 windows takes 3.2 to 3.5 s per checkpoint. A 3,000-step run measured by the fix wave ended at
.9820, .9739 and .9356 (size 4 at 1500, 2000, 2500, 3000: .9158, .9109, .9406, .9356), no better for the size that is
closest to the limit, so `teacher_doc_tiny.yaml` stays at 2,500 steps (validation and save every 500, five kept).
Note: the build's first table (the Task 5 record) counted a window aligned when `split_output` returned a list, so it
accepted empty segments, and read .955 at checkpoint 1000 for size 2 and .916 at 2000 for size 4; the fix wave
(review M3) moved the test to the chunker's own definition and these two cells became .9521 and .9109; every other
cell is the same under both.

**Why 2,500 and not the spike's 1,000.** The spike (stride-1 windows, 7,260 of them, which overlap) reached .97, .92
and .99 at 600, 800 and 1,000 steps. `docify` cuts non-overlapping windows (1,194 on the same data), and on those the
dev alignment after 1,000 steps was .955, .847 and .886 for the three sizes: below 0.9. The threshold was NOT lowered
(a gate is never lowered); the checked-in tiny config trains 2,500 steps. T9 and the plan's "1,000 steps" hold for
overlapping windows only (spec section 8).

**What the numbers do and do not show.** They show that continuing the tiny teacher on non-overlapping windows teaches
the `<sep>` structure within a couple of thousand CPU steps and that the whole chain (vocabulary invariant, `docify`,
provenance, continuation, resume, chunker) works through the real binary. They do not show translation quality (the
toy sentences are templated and independent), they are single-seed, and size 4 has the smallest margin (.941 against
.9 on 202 windows): the review estimated a 5 to 10 percent chance of a miss on another CPU or torch build. The
training is bit-for-bit repeatable here (two runs, the same weights at all five checkpoints), not on another machine
or thread count; **if CI flakes raise `train_steps`, never lower the gate.** The docify test's no-test-sentence check
guards the workspace more than it guards `docify` (the test rows are not in the file it reads).

**The chunker over the toy test split (66 document-direction pairs, 400 sentences).** 106 windows went to the window model,
102 came back aligned and 4 fell back to the averaged teacher at sentence level; 18 sentences were singles, none was
oversize or empty; every document got exactly one non-empty translation per sentence. The chunker's chrF against the
references is 70.62 against 10.44 for pure sentence-level translation by the averaged teacher (a model of only 200
steps); recorded, not gated. The scenario took 5.8 s (two `eole predict` calls, one per model).

**The refusals (R).** Five continuations are refused, each exit 2, Eole not started, the run directory holding at most
the lock file and the source's tree digest unchanged: a step directory without its run; a run copied without its
manifest; one weights byte flipped; a vocabulary without `<doc>` (file and manifest rewritten to agree); a config
without an architecture key. The SentencePiece mismatch, the `average` guard, the vocabulary size and the `--max-pieces`
ceiling have unit tests; the real-binary messages are above.

## The operator workflow for a real document model

UNMEASURED: no real document corpus and no GPU run exist, and every value of `configs/eole/teacher_doc.yaml` is a prior.
The steps are the ones the tiny gate ran; the numbers in the config are the ones to measure.

1. **Train the teacher with the invariant.** Build it through `rinchen train ablate --prepare-only` and `rinchen train
   launch` (runbook 08), so that its config states `src_vocab_size` (the SentencePiece pieces plus one) and its
   `vocab.src` holds every reserved symbol. A teacher trained before D9 lacks `<doc>` and `<sep>` and is refused at the
   first `--continue-from`; there is no way to continue it, it has to be trained again.
2. **Average the last checkpoints** (`rinchen train average --checkpoint-dir <run>/ckpt --last N`), or continue from the
   final step directory itself (then `optimizer.pt`, 8.8 GB for the real teacher, is read for nothing).
3. **Cut the windows.** Make a copy of `configs/eole/teacher_doc.yaml` that states `src_vocab_size: 128001` (the pieces
   of the 128,000-piece model and Eole's pad token; `launch` refuses a config that leaves it to Eole's default of
   32,768). `rinchen train docify --corpus <D4 mine output> ... --spm-model <the teacher's model> --config <the copy>
   --out-dir <O> --dry-run` first, and read the statistics: `windows of size n`, every
   `excluded <reason>` (a large `gap_single` or `tail` says the documents are fragmented by the filters, a large
   `too_long` that the budget is too small for real sentences), the piece counts, the weights. Then without
   `--dry-run`. The SentencePiece model must be the teacher's (the launch refuses another).
4. **Continue.** `rinchen train launch --config <O>/config.yaml --run-dir <O>/run --continue-from <averaged> --world-size 8
   --gpu-ranks 0,1,2,3,4,5,6,7` (and `--write-only` first for `command.sh`). Everything of runbook 08 applies (the lock,
   the NaN gate, the resume): an interrupted run is resumed by the same command, WITH `--continue-from`. The first
   contact with a GPU needs the owner's authorisation.
5. **Evaluate.** The alignment rate of held-out dev windows per window size through `predict_tokens(max_length=...,
   max_length_ratio=0)`, the chunker's statistics over real documents (`aligned`, `fallbacks`, `singles`, `oversize`) and
   the chrF of the doc output against the sentence-level teacher's.

Open, because nothing measured them: the mix of document windows with sentence-level data (the spike's toy run with
the sentence corpora mixed in had alignment 0.81 at 400 steps and 0.99 at 3,200, window chrF 73 and 92 at 1,600 and
3,200 steps, against 0.38 and 31 for windows alone at 200 and 1,600 steps; one seed, templated data), the window-size
mix (2 to 4 by default), the number of steps and the learning rate (the prior: 0.25 with 2,000 warm-up steps, 20,000
steps, a validation and a save every 2,000, early stopping after 6 validations), whether the doc model should still
translate single sentences (it was continued on windows only, which is why the chunker takes the fallback from a
callable), and `src_seq_length` 512 for the windows of real sentences.

## Open gates

Not done by D9, because no real data, GPU or reader exists: a document model on real documents and its measured
quality; the real-corpus doc workflow above (mix ratios, window-size mix, the real continuation schedule); anything on a
GPU (bf16, NCCL, the 13 GB checkpoint of the teacher shape, the time of the continuation); the CTranslate2 export of a
document model and the use of the chunker in a serving path (D11); the D3 gates assigned to D4, the D5 gates that need
real text or a reader and the D6, D7 and D8 gates stay open as recorded in their runbooks.

## What was not verified

- **Anything on a GPU**, and a real document corpus. Every continuation ran on the CPU, on 1,194 toy windows.
- **A real teacher as a source**: the 13 GB step directory's hashing time (`verify_source` reads and hashes the four
  files, and `launch` hashes them again before every resume or extension), the 8.8 GB `optimizer.pt` read by Eole for a
  step source, and the write of the new checkpoints.
- **`src_seq_length` 512 and windows of real sentences**: only the toy's 12 to 52 pieces were run.
- **`read_corpus_rows` at scale.** It loads every train and dev row with text into one frame (the other splits only
  their `split`); the memory for tens of millions of rows is unmeasured (D7's loader, for comparison: about 1.05 GiB
  for each further million rows of this text length).
- **A different SentencePiece model under `train_from`** was not run in Eole (the launch refuses it by hash), and `update_vocab` with
  `reset_optim: states` or non-shared embeddings was not tried (the spike).
- **Whether averaging helps a real teacher as the source**, and whether continuing from a step directory differs from an
  averaged one beyond the optimiser file.
- **The size-4 margin of the gate on another machine** (above), and CI: no CI run covers D3 to D9.
- **Windows and macOS** were not run.
- **The real binary** was run for the whole flow above, the refusals and the `average` guard; every other path ran in
  process with fakes at the `run_logged` seam.

### Known limits, left open on purpose

Findings of the reviews that were parked and not fixed, and limits of the build, one each:

1. **Only five optimiser options are required** by `check_continuation` (`learning_rate`, `warmup_steps`,
   `decay_method`, `optim`, `adam_beta2`). Eole inherits EVERY option a continuation config omits from the checkpoint:
   `max_grad_norm`, `label_smoothing`, `dropout`, the batch size, `weight_decay`, `accum_count` and the rest are
   not checked, and an omitted one silently continues with the teacher's value. The shipped configs state them all
   (a test builds Eole's own schema from them), a config of the operator's might not. (Review M2.)
2. **The chunker's budget counts the SOURCE only**, and `eole predict` silently truncates the target at `max_length` and
   `max_length_ratio`, so a cut inside the last segment still counts as aligned. For D11. (Review M5.)
3. **A `translate_sentence` result that contains `<sep>` passes through** the chunker unchecked: the fallback's output
   is not validated. For D11. (Review M6.)
4. **Resuming a continuation requires its source to exist and verify.** The provenance chain is re-verified before
   every resume, extension or `--write-only` of a run that has step directories; a source that was deleted or changed
   stops the resume (a finished run needs none). By design. (Review M8.)
5. **A literal `<doc>` or direction tag inside a sentence is not guarded**, only `<sep>`: SentencePiece reads either as
   the symbol, in `docify` and at inference. (Review M11.)
6. **`<doc>` and `<sep>` are defined three times** (`spm.symbols`, `train.doc_source`, `infer.doc_chunker`) because the
   inference side must not import the training side; a unit test pins the three equal. (Review M9.)
7. **The `average` guard and the provenance rule guard against mistakes, not against forgery**: the `average` guard
   looks for a `launch.json` that parses within three parent directories of the step directories and nothing else, so a
   hand-made `step_<N>` inside a real run directory passes; a record or manifest written by hand passes the provenance
   rule. The source hash is not atomic with Eole's own read of the files.
8. **The SentencePiece comparison reads the model path as given**: a relative path in the config is read from the
   working directory of `rinchen`. The manifest does not vouch for the copy in the source directory and `launch.json`
   does not record its hash.
9. **`launch` checks the vocabulary size against the SentencePiece model** (fixed by the re-review, M1): it refuses a
   STATED size below the pieces plus one and an UNSTATED one when Eole's default, 32,768, is below it (the D5 converter
   configs state sizes equal to their piece count, so they stay off `launch`). It does not set the size.
10. **`command.sh` does not add the reserved symbols** (the next `launch` does, too late for what the hand run trained).
11. **Files near the 900-line limit**: `launch.py` has 886 lines, `continuation.py` 777, `eole_cfg.py` 856 (it had
    976 before `eole_cfg_keys.py`, 147 lines of key lists, was split from it). The next change to `launch.py` has to move something.
12. **The gate's size-4 margin** is thin (.9406 against the 0.9 gate, one seed) and the training is repeatable only on
    the machine and thread count it was measured on (above). A run of 3,000 steps measured worse for that size (.9356), so
    `teacher_doc_tiny.yaml` stays at 2,500. If the gate flakes on another machine, raise `train_steps` further; never
    lower the gate.
13. **`TRAINING_KEYS` has no `adam_beta1` or `adam_eps`**: they stay Eole's defaults, which the teacher trained with.
14. **An ablation run made before D9 retrains** (its config gained `src_vocab_size`, so its fingerprint changed).
15. **The `average` guard sees only the three lexical parents** of the checkpoint directory as given (re-review M2). A
    `ckpt` that is a symbolic link to a place outside any run directory is refused although the run is real, and so is a
    `training.model_path` nested three or more levels inside the run directory (`<run>/a/b/ckpt`: the record is four
    parents up; `<run>/a/ckpt` is accepted). The same lookup (`checkpoints.run_directory_above`) serves `verify_source`
    for a step directory given to `--continue-from`: pass the real path.
16. **Documents are keyed per `corpus` label, not per file** (re-review M4; C1 is fixed at that level). D4's `mine` sets
    `doc_id` to the stem of the source file (`cli/mine.py`), so two files that carry the same `corpus` label and have
    the same stem still share a document key, and their sentences can be joined into one window. Only the counted
    exclusion `duplicate_seg_idx` signals it (the rows of the second file repeat the `seg_idx` of the first). Make D4's
    document ids unique per file before real documents are windowed, or read that count of every dry run.
17. **The `launch` pre-flight runs on every plan** (re-review M5): also with `--write-only` and when the run is already
    finished. With a stated vocabulary size it needs the SentencePiece model readable from the launcher's working
    directory (a relative path in the config is read from there, as Eole reads it); a model that is not a file is
    refused for a stated size and left to Eole for an unstated one.

### The review (Tasks 1 to 5) and what its fix wave did

An independent combined review of Tasks 1 to 5 (`993f32c..98e42d1`, read-only) found one Critical, five Important and
twelve Minor findings and checked what it could not break: the held-out dev windows of the gate, a SIGTERM test that is
not vacuous, a vocabulary-without-`<doc>` refusal that fires only from the vocabulary check, the shipped configs, the
layering, the D7 refactor (behaviour-preserving) and the hygiene (no mode change, no invisible characters, nothing
written under `data/` or `models/`). Its fix wave, one commit each with a failing test first:

- C1 (`3838713`): a document was keyed without its corpus, so two corpora sharing a `doc_id` were joined or interleaved
  into one window (the reviewer's probe: `A3 <sep> B0`, and `B0 <sep> A1 <sep> B2`). The key includes `corpus`.
- I1 (`7d3da62`): `read_corpus_rows` collected the text of every split, test included, then counted the others; the
  filter is in the lazy plan now.
- I2 (`61fada2`): `rinchen train average` accepted any `step_N` directories and wrote a manifest that
  `--continue-from` then accepted; it refuses a checkpoint directory with no `launch.json` within three parents, and
  `verify_source` cross-checks `averaged_from.json` with the manifest.
- I3 (`92dddcc`): the SentencePiece model of the config was never compared with the teacher's.
- I4 (`b1c2d12`): an empty `vocab.src` became 118 reserved symbols; now a `DataError`.
- I5 (`63c97d5`): `launch` did not enforce `src_vocab_size >= pieces + 1`.
- Minors (`c2d46b5`, `D9: review minors`): `eole_cfg_keys.py` out of `eole_cfg.py` (976 to 856 lines), the gate's aligned
  rule is the chunker's, the chunker closes a window at a blank, the permission bits of the vocabulary file are kept,
  dead code, the plan's wording, a constant in place of a literal. The measured 3,000-step run (above) was not
  adopted.

Gate after the fix wave: 8214 passed, 1 skipped, 10 deselected (929 s), against 7187 at the end of D8 (7279 after Task
1, 7446 after Task 2, 7799 after Task 3, 8024 after Task 4, 8075 after Task 5). Left open on purpose: the items of
"Known limits" marked as review findings (M2, M5, M6, M8, M9, M11).

### The scoped re-review and the second fix wave

A scoped re-review of that fix wave (an independent reviewer, read-only, `98e42d1..c2d46b5`) approved it **with minors:
no Critical and no Important finding**. A second small fix wave (`135e9f9`, `D9: re-review minors`) fixed what needed
code:

- M1: the pre-flight of `launch` skipped an UNSTATED `src_vocab_size` or `tgt_vocab_size` on the premise that Eole's
  default is large. It is 32,768 (`eole/config/data.py`, read in the installed Eole), the inputter cuts the vocabulary
  file at load, and the reserved symbols are its last lines: a hand-written config with a model of more than 32,767
  pieces and no size, launched through `launch`, would have lost them. `launch` now refuses it (see "The vocabulary
  invariant"). No shipped path hit it, because `ablate` sets the size; `launch` with `teacher.yaml` and the 128,000-piece
  model would have, and now says which value to state. `require_stated_sizes_to_fit` became `require_sizes_to_fit`.
- M3: `docify(rows, split="test")` on rows read without that split reported every row as `no_doc_id` and made no
  window, and `read_corpus_rows(splits="train")` iterated the characters of the string and loaded nothing. `docify`
  refuses a split outside `("train", "dev")` (`ValueError`) before it reads the SentencePiece model, `classify_rows`
  refuses rows whose rows of the split are all empty apart from their `split` (`DataError`, naming
  `read_corpus_rows(splits=...)`), and `read_corpus_rows` refuses a bare string (`TypeError`).
- M6 and the guard's wording: stale docstrings of `docify.py` and `launch.py` (the latter said that `eole build_vocab`
  accepting `training.train_from` "is not measured"; it was, runbook 08 and the spike), two comment lines of 125 and 145
  columns, and the import of `tests.support.step_dirs` at the top of `tests/conftest.py` (now inside the one fixture
  that needs it). The docstrings and the message of the `average` guard said that only the checkpoints of a run that
  `launch` wrote are averaged, which overclaims: what is checked is a `launch.json` that parses within three parent
  directories of the step directories ("The `average` guard").

Left as limits, not fixed (Known limits 15 to 17): M2 (the guard sees only three lexical parents), M4 (documents are
keyed per `corpus` label, not per file), M5 (the pre-flight runs on every plan and needs the SentencePiece model
readable from the working directory when a size is stated). The reviewer's estimate of the size-4 margin of the gate
(.9406 against 0.9, one seed) stands, with the measurement that 3,000 steps are worse (.9356): see limit 12.

Gate after the second fix wave: 8252 passed, 1 skipped, 10 deselected (1,217 s).

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM, Linux 7.0.0-42-generic, CPython
3.14.10, eole 0.6.4, torch 2.14.1+cu130, safetensors 0.8.0, sentencepiece 0.2.1; the GPU (RTX 5070 Ti) was hidden from
every Eole run. All on 2026-11-09, once each, code at commit `c2d46b5`, **on a machine that was loaded by other work
(load average 8 to 22), so every wall time is longer than an idle machine's**; the numbers that are not times do not
depend on the load (two continuations gave the same weights).

| What | Value |
|---|---|
| vocabulary: `build_vocab`, then the invariant | 225 lines, 4 of 118 symbols present, 114 appended, 339 lines; a second call appends 0; 5.98 s for `build_vocab` |
| `ablate --prepare-only` (the tiny teacher) | 0.75 s |
| the tiny teacher, `launch`, 200 steps | 49.38 s (25.61 s on 2026-11-08, before D9 and with less load); `xent` 5.81 to 2.13; validation perplexity 106.851, 18.3535, 10.4948, 8.73594; weights SHA-256 `6f5e836c7553...`, 3,318,120 B |
| `average --last 2` | 1.43 s; averaged weights `081f9d543...`; the averaged directory 3,353,925 B |
| `docify --dry-run`, `docify` | 2.31 s and 2.73 s; 1,194 windows (514, 368, 312 of sizes 2, 3, 4), weights 33 33 17 17 |
| the continuation, 2,500 steps | 107.14 s and 119.45 s (87 s idle in the build); `xent` 3.07 (step 50) to 0.97; validation perplexity 7.34474, 5.99236, 3.62612, 3.28613, 3.21243 |
| repeat of the continuation | the weights of steps 500, 1000, 1500, 2000 and 2500 are byte-identical to the first run's |
| alignment of 804 dev windows, `predict_tokens` | 3.2 to 3.5 s per checkpoint; the table above |
| the chunker over the 66 test document-direction pairs | 5.8 s; 106 windows, 102 aligned, 4 fallbacks, 18 singles; chrF 70.62 against 10.44 |
| refusals (8 real-binary cases: the 7 listed and the stated vocabulary size) | 0.97 to 1.37 s each, exit 2 |
| `du -sb` | run directory of the continuation 24,678,163 B; one step directory 4,896,614 B; `$DOC` outputs: corpora 36,278 and 36,398 B for the biggest pair, `valid.src` 21,579 B |
| the tiny ablation of D7 (2 variants, 1,000 steps) | 135.7 s (72.7 s before; train 75.1 s and 50.9 s, evaluation 4.2 s and 4.0 s); dev chrF in runbook 07 |
| the gate | not re-run for this record: only documents differ from `c2d46b5`, whose gate gave 8214 passed, 1 skipped, 10 deselected in 929 s; `ruff format --check`, `ruff check` and `mypy src` were run on this tree |

The piece counts of the windows `docify` cut (1,194 windows, source with the tag and `<doc>`, target without; the
sentences alone are 4 to 13 pieces, mean 6.9, over the 3,600 train rows):

| window size | windows | source pieces min / mean / p95 / max | target pieces min / mean / p95 / max |
|---:|---:|---|---|
| 2 | 514 | 12 / 17.8 / 26 / 29 | 10 / 15.8 / 24 / 27 |
| 3 | 368 | 18 / 26.6 / 36 / 39 | 16 / 24.6 / 34 / 37 |
| 4 | 312 | 24 / 35.4 / 49 / 52 | 22 / 33.4 / 47 / 50 |
| all | 1,194 | 12 / 25.1 / 45 / 52 | 10 / 23.1 / 43 / 50 |

The measurement scripts were read-only scratch files (`measure.py`, `measure_d8.py`) run with `python -I`; they reuse
the helpers of `tests/integration/test_train_tiny_doc.py`, so the numbers are the gate's own.

## Where the code lives

`src/rinchen/train/{docify,doc_source,doc_pack,doc_corpus,continuation,vocabulary_size,eole_cfg_keys}.py` with the
extensions of `eole_run.py`, `eole_cfg.py`, `checkpoints.py`, `launch.py`, `launch_command.py`, `ablation_config.py`,
`ablation.py`, `corpus_files.py`, `mixture.py` and `validation_corpus.py`; `src/rinchen/infer/doc_chunker.py`;
`src/rinchen/cli/{train_docify,train_runs,train_common,train}.py`; `src/rinchen/utils/part_file.py` (`mode`);
`configs/eole/{teacher_doc,teacher_doc_tiny}.yaml`; unit tests `tests/unit/test_train_{docify,docify_corpora,doc_source,
doc_corpus,continuation,launch_continue,reserved_symbols,vocabulary_size,checkpoints,eole_cfg,eole_cfg_drift,
eole_cfg_shared,eole_run,launch,launch_command,mixture,validation_corpus,write_corpus_files,ablation,
ablation_config}.py`, `test_infer_doc_chunker.py`, `test_cli_train_{docify,docify_refusals,launch,average}.py`,
`test_support_tiny_documents.py`, `test_utils_part_file.py` and `test_eole_configs_shipped.py`; property tests
`tests/property/test_{train_docify,infer_doc_chunker}_property.py`; `tests/integration/test_train_tiny_doc.py` (the
STOP gate, scenarios T, D, C, A, K, R, S, B); and the helpers `tests/support/{doc_cli,doc_rows,sized_spm,step_dirs,fake_launch,
tiny_parallel,tiny_variant}.py`.
