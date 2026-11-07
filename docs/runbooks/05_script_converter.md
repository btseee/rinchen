# 05 - Script converter

## Scope

D5 builds the converter between Cyrillic Mongolian (`khk_Cyrl`) and traditional
Mongolian script (`mn_Mong`): the package `rinchen.script_conv` and the command
group `rinchen convert-script` (`build-data`, `train`, `export`, `apply`,
`eval`). The design, with the rulings that fill the gaps of the master plan and
what each costs if wrong, is
`docs/superpowers/specs/2026-11-07-d5-script-converter-design.md`; the decisions
are condensed in the 2026-11-07 entry of `DECISIONS.md`.

**What D5 is.** A tested, trainable pipeline: a validator that keeps forbidden
control characters out of the script it writes, a segmenter, the metrics, a rule
baseline measured against a held-out dictionary reference, a corpus builder, the
converter's own character-level SentencePiece model, a rendered and validated
Eole configuration, a hardened subprocess runner for Eole, the CTranslate2 export
with a parity check, and `ScriptConverter`, the small runtime wrapper the serving
engine will call.

**What D5 is NOT.** It is not a quality converter, and nothing below should be
read as one.

- The only model ever trained is the tiny one (2 encoder and 2 decoder layers,
  hidden size 64). It memorises a 500-word fixture. Its error rates on its
  training words say nothing about words it has not seen: on the 52 fixture
  words that were held out of its training, its character error rate is 0.24 and
  0.30 (see "Tiny model", below).
- The rule baseline is a scaffold. On a held-out half of the dictionary reference
  it spells 237 of 1,022 words exactly (23.2%).
- No real parallel script data exists in the repository. The lexicon (1,005
  words), the golden words and the tiny fixture all derive from one harvest of
  English Wiktionary (2,029 rows). No Mongolian reader has judged any of it.
- A real converter needs real data: Mongolia-sourced parallel text in both
  scripts, aligned, and a reader to review what comes out. The full-profile
  configuration is rendered and checked by `validate_converter_config`, but it
  has never been loaded through Eole's own configuration class or trained.
- Conversion is word by word. A homograph gets one spelling whatever its
  sentence says.
- There is no serving engine yet (D11, D12); `ScriptConverter` is the interface
  it will call.

**Every threshold in this runbook is a prior, and every gate was measured on the
tiny memorisation fixture.** A gate that passes shows that the machinery works
end to end; it does not show that the converter is good.

What was checked, and how: the unit and property tests (planted inputs with exact
counts, mutation checks done by hand during review); the integration tests in
`tests/integration/`, which run a real Eole training on CPU, the CTranslate2 export,
the parity of the two decoders and the real `rinchen` binary; and one run of the
whole command chain by hand for this runbook (the worked example below). The
sections "Measured numbers" and "What was not verified" say which figure comes
from where.

## Architecture

| Module (`src/rinchen/script_conv/`) | Job |
|---|---|
| `alphabet.py` | Code points: the 35 letters U+1820..U+1842, the free variation selectors, MVS, NNBSP, NIRUGU, the joiners, the two punctuation marks |
| `validate.py` | `validate_traditional(text)` returns every violation (joiner, foreign character, orphan selector, misplaced NNBSP, variant the Unicode table does not list); `is_script_word(text)` says whether a model output is one converted word |
| `standardized_variants.tsv` | The 45 letter and selector pairs of the Unicode Character Database's `StandardizedVariants.txt` (version 18.0.0) that Mongolian text uses; its source URL and SHA-256 are in `manifests/ucd_standardized_variants.json` |
| `segments.py` | `segment(text)` cuts text into Cyrillic, Mongolian and other runs; splits on U+0020 only, so an NNBSP does not cut a word from its suffix |
| `eval.py` | `edit_distance`, `cer`, `word_accuracy`, `score_conversions`, `length_ratio_quantiles` |
| `fallback.py` | The data-free phonographic spelling: one letter map and five contextual rules; reads no file |
| `rules.py` | `transliterate_word` (lexicon first, then the fallback), `transliterate`, the `LEXICON` mapping |
| `lexicon.tsv` | 1,005 verified Cyrillic to script spellings, the development half of the reference; package data |
| `data.py` | Pair readers, `rule_pairs`, `align_words`, and `build_corpus`, which screens, deduplicates, splits and writes the eight Eole corpus files |
| `char_model.py` | Trains, checks and loads the character SentencePiece model with the two direction tags as single pieces |
| `config.py` | Renders the Eole configuration of a profile, validates it, writes it as YAML, reads the checkpoint and character model paths back |
| `_subprocess.py` | `run_logged`: a command with its output in a log, a timeout, and a kill of its whole process group |
| `train.py` | `run_training` (`eole build_vocab` then `eole train`), `eole_commands`, `predict` (`eole predict` on CPU) |
| `export.py` | `export_ct2` (CTranslate2 export with CUDA hidden) and `parity` (CTranslate2 against Eole) |
| `runtime.py` | `ScriptConverter`: CTranslate2 plus the character model, a word cache, validation of every output, rule fallback |

`src/rinchen/cli/convert_script.py` holds the five commands. It is the only module
that writes manifests. The layering is: `alphabet` is the leaf; `validate`,
`segments` and `eval` depend on it; `fallback` on nothing of the package; `rules`
on `fallback`, `segments` and `alphabet`; `data` on `rules`, `validate` and
`rinchen.text.normalize`; `config`, `train` and `export` know `data`'s file layout
only; `runtime` on `segments`, `rules`, `validate` and `char_model`.

## Commands

All five commands are under `uv run rinchen convert-script`. The example below is
the real chain, run once on 2026-11-07 on the tiny fixture with the project's
`rinchen` binary (what `uv run rinchen` starts) and everything outside the
repository (`$W` stands for an empty scratch directory). The same chain is the
integration test `test_the_whole_cli_chain_on_the_tiny_fixture` in
`tests/integration/test_cli_convert_script.py`.

The default character model config, `configs/spm/converter_char.yaml`, writes to
`data/script_conv/converter_char.*` inside the repository, so the example points
`--spm-config` at a file of its own (the tiny fixture needs only a small cap, see
"Facts found on this machine"):

```bash
printf 'vocab_size: 200\noutput_prefix: %s/char/converter_char\n' "$W" \
    > $W/converter_char.yaml
```

### build-data

```bash
uv run rinchen convert-script build-data \
    --pairs tests/fixtures/script_conv/tiny_words.tsv --out-dir $W/corpus
```
```
wrote 448 train and 52 valid pairs to $W/corpus
manifest: $W/corpus/manifest.json
```

Options: `--pairs FILE` (a UTF-8 TSV headed `cyrillic<TAB>script`, one pair per
line) or `--words FILE` (Cyrillic words, one per line, spelled by the rule
baseline), exactly one of the two; `--out-dir` (default `data/script_conv/corpus`;
it must not be the directory of the input file); `--valid-fraction` (default 0.1,
in (0, 0.5]); `--seed` (default 1234; it seeds the split); `--allow-fallback`
(with `--words` only: also keep the words the lexicon lacks, spelled by the
phonographic guess, a weak training label). With `--words` the command prints
`N words read, M paired` first; when pairs were dropped it prints
`dropped K: code=n, ...` between the `wrote` line and the manifest line.

It writes `cyrl2mong.{train,valid}.{src,tgt}` and `mong2cyrl.{train,valid}.{src,tgt}`,
UTF-8, LF, one word per line; the `mong2cyrl` files are the `cyrl2mong` files with
the two sides swapped. Each side is normalized (`khk_Cyrl` for Cyrillic, `mn_Mong`
for script, which keeps NNBSP, MVS and the selectors). A pair is dropped on the
first check it fails, in this order: `empty`; `multiword` (a side still holds a
U+0020 after its outer spaces are stripped, because a corpus line is one word);
a violation on the raw script side, counted under its code (`joiner`,
`foreign_character`, `orphan_selector`, `misplaced_nnbsp`, `invalid_variant`);
`duplicate`. A pair goes to the valid split when the SHA-256 of
`"{seed}:{normalized Cyrillic word}"`, read as a fraction of 2**256, is below
`--valid-fraction`, so equal Cyrillic words never straddle the split. An empty
train split is an error; an empty valid split writes empty files.

### train

Look first, then run. A dry run checks the corpus files, the character model
config and the Eole configuration (rendered and validated, never written), refuses
a foreign manifest in the run directory, and prints four lines:

```bash
uv run rinchen convert-script train --profile tiny --data-dir $W/corpus \
    --run-dir $W/run --spm-config $W/converter_char.yaml --dry-run
```
```
$W/char/converter_char.model
$W/run/converter.yaml
.../.venv/bin/eole build_vocab -config $W/run/converter.yaml -n_sample -1
.../.venv/bin/eole train -config $W/run/converter.yaml
```

The real run (37 s on the machine below; the same command without `--dry-run`):

```bash
uv run rinchen convert-script train --profile tiny --data-dir $W/corpus \
    --run-dir $W/run --spm-config $W/converter_char.yaml --threads 2 --timeout 600
```
```
trained tiny: checkpoint $W/run/ckpt/model.00.safetensors
manifest: $W/run/manifest.json
```

`train` owns the character model. In order: it hashes its inputs, reads the
`cyrl2mong` corpus files (the `mong2cyrl` files hold the same words, so they would
only count each word twice), removes the old manifest, trains the character model
at `<output_prefix>.model` and `.vocab`, checks it on those words
(`check_char_model`), writes `converter.yaml` into the run directory, and runs
`eole build_vocab -n_sample -1` and `eole train` from random initialisation, each
with its output in `build_vocab.log` and `train.log` of the run directory. The
checks of the plan (profile, character model config, corpus files present, the
rendered and validated Eole config, no `$` in a path, the `eole` command
installed) run in the dry run too, so a real run does not fail on something its
dry run passed, nor after the character model is trained.

Options: `--profile tiny|full` (required), `--data-dir` (required), `--run-dir`
(required; made absolute), `--spm-config` (default
`configs/spm/converter_char.yaml`), `--threads` (default 2; sets `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` of each Eole run), `--dry-run`, `--confirm-full`,
`--timeout` (seconds for each of the two Eole commands, no limit by default).
`--profile full` without `--dry-run` refuses unless `--confirm-full` is given
(exit 2).

### export

```bash
uv run rinchen convert-script export --run-dir $W/run --out-dir $W/model --timeout 600
```
```
exported to $W/model
manifest: $W/model/manifest.json
```

Converts the newest checkpoint with CTranslate2's own converter (CUDA hidden, see
"Facts found on this machine"), copies the character model into the result as
`converter_char.model`, and writes the manifest last. 3.5 s here. Options:
`--run-dir` (required), `--config` (default `<run-dir>/converter.yaml`), `--out-dir`
(default `<run-dir>/ct2`; it must not exist), `--timeout` (default 3600 seconds).
The converter's output goes to `<out-dir>.export.log`, beside the directory.

### apply

```bash
printf 'аав ааг\n' | uv run rinchen convert-script apply --model-dir $W/model --target Mong
```
```
ᠠᠪᠤ ᠠᠭᠠᠭ
rule_fallbacks=0 invalid_outputs=0
```

The first output line is standard output and the second is standard error. The
command reads a UTF-8 text from `FILE` or standard input, converts the whole text as one
string (its line breaks are text between words), writes the result to standard
output exactly, with no newline added, and prints the two counters of
`ScriptConverter` to standard error. Empty input exits 0 and writes nothing.
`--target Mong` converts to traditional script, `--target Cyrl` to Cyrillic; any
other value exits 2. The whole input is held in memory, in several copies.

### eval

```bash
uv run rinchen convert-script eval --model-dir $W/model --pairs $W/pairs_train.tsv
```
```
cer 0.0051
word_accuracy 0.9844
n 448
rule_fallbacks 0
invalid_outputs 0
```

`pairs_train.tsv` here is the 448 training pairs as a pairs file: the header, then
each line of `cyrl2mong.train.src` and `cyrl2mong.train.tgt` joined by a tab. The
command converts one side of each reference pair (`--target Mong`, the default, converts
the Cyrillic side and scores against the script side; `Cyrl` the other way round)
and prints the character error rate, the share of exactly right words, the number
of words and the two counters. Exit 3 for a pairs file with no pairs.

### Exit codes

Failures print `error: <message>` on standard error.

| Code | Meaning |
|---|---|
| 0 | OK, including an empty `apply` input and a `--dry-run` |
| 1 | an `OSError` while writing outputs, or any other `RinchenError` |
| 2 | `ConfigError`: both or neither of `--pairs` and `--words`, a `--valid-fraction` outside (0, 0.5], an unknown profile or target, `--profile full` without `--confirm-full`, a `$` in a path or a rendered config string, an `--out-dir` that is the input's directory (`build-data`) or that exists (`export`), a character model config that does not hold |
| 3 | `DataError`: a missing or unreadable corpus, pairs, checkpoint, character model or model file, input that is not UTF-8, no pair left after dropping, an empty train split, a pairs file with no pairs (`eval`), an out directory that holds another command's manifest |
| 4 | `UpstreamError`: an Eole or CTranslate2 command exits non-zero, times out, or leaves no usable output, the `eole` command is not installed, the translator answers wrongly (the message ends with the tail of the log) |

## Model directory and manifests

A model directory is self-contained: `apply` and `eval` need only `--model-dir`.

```
model/
  model.bin               the CTranslate2 weights (1,965,671 bytes for the tiny model)
  config.json             CTranslate2 model config (no BOS or EOS added to the source)
  shared_vocabulary.json  CTranslate2 vocabulary
  converter_char.model    the character SentencePiece model (copied in by `export`)
  manifest.json           stage script_conv_export
model.export.log          the converter's log, beside the directory
```

`ScriptConverter(model_dir, model_dir / "converter_char.model")` opens it.

The run directory of `train` holds `converter.yaml` (the rendered config),
`vocab.src` (one vocabulary: both directions share it; 70 pieces for the tiny
fixture), `build_vocab.log`, `train.log`, `manifest.json`, and `ckpt/` with one
`step_N/` directory per saved checkpoint plus links to the newest one
(`model.00.safetensors`, `config.json`, `vocab.json`, `converter_char.model`).

Each command writes one `manifest.json` (`rinchen.manifest/1`), last, into its
output directory:

| Command | `stage` | `inputs` | `outputs` | Other fields |
|---|---|---|---|---|
| `build-data` | `script_conv_data` | the pairs or words file | the eight corpus files with row counts | `seed`; `config_sha256` is the SHA-256 of the parameters as canonical JSON (source kind, valid fraction, seed, allow-fallback); `metrics`: `valid_fraction`, `pairs_in`, `dropped.<code>` |
| `train` | `script_conv_train` | the eight corpus files and the character model config | the character `.model` and `.vocab`, `converter.yaml`, the newest checkpoint, listed as `ckpt/model.00.safetensors` (a link; the hash is of the file it points to) | `config_sha256` is the hash of the rendered `converter.yaml`; `metrics`: `threads`; `tool_versions`: eole, torch, sentencepiece, ctranslate2 |
| `export` | `script_conv_export` | the checkpoint file itself (the file in its `step_N/` directory), the character model, and the checkpoint's `config.json` and `vocab.json` | every file under the model directory but the manifest | `config_sha256` of the `converter.yaml` used; `tool_versions`: ctranslate2, eole |

Every manifest also carries `code_version` (the first seven characters of the HEAD
commit, `-dirty` when the tree has uncommitted changes) and the SHA-256 and size
in bytes of each listed file. Both character model files are recorded because the `.model`
embeds its own absolute prefix, so its hash holds for one location only; the
`.vocab` file (pieces and scores) is the identity that holds anywhere. Training
the same words into the same prefix gives byte-identical files.

The profile is not a manifest field; it is visible in the hash of `converter.yaml`
and in the file itself.

## Gates and thresholds

The numeric gates are the master plan's and the spec's; a missed gate means more
steps or a fix, never a lower threshold. All are for the tiny fixture. "Measured"
is the figure of the final code unless a note says otherwise.

| Gate | Threshold | Measured | Test |
|---|---|---|---|
| CER, Cyrillic to script, Eole greedy, 448 training words | <= 0.15 | 0.0051 | `tests/integration/test_script_conv_tiny.py` |
| CER, script to Cyrillic, same | <= 0.15 | 0.0128 | same |
| NNBSP, MVS and selector counts kept in a predicted word | >= 90% of the 81 words that have them | 78/81 = 96.3% | same |
| Training wall time | < 150 s | 34.8 s to 37.4 s over five runs | same |
| CTranslate2 against Eole, both directions | >= 0.99 | 448/448 each | same |
| The requested tag changes `predict`'s output | >= 10% of words differ under the wrong tag | 67/448 = 15.0% | same |
| `ScriptConverter` word exact match against the reference script | >= 0.90 | 441/448 = 0.9844 | `tests/integration/test_script_conv_runtime.py` |
| `ScriptConverter` against Eole, both directions | >= 0.99 | 448/448 each | same |
| Real CLI chain: CER and word accuracy of `eval` | CER <= 0.15, accuracy >= 0.90 | 0.0051 and 0.9844 | `tests/integration/test_cli_convert_script.py` |
| Rule baseline golden words | all 100 right | 100/100 (30 from the lexicon, 70 from the fallback) | `tests/unit/test_script_conv_rules.py` |
| Held-out fallback exact match and CER | measured value within 0.02 | 0.2319 and 0.2342 | `tests/unit/test_script_conv_fallback.py` |

By kind of mark, the words that kept the mark out of the words that carry it:
NNBSP 4/5, MVS 44/44, first selector 30/32, second selector 1/1 (a word can carry
more than one kind). The training time is a range because it was measured five
times, all with `--threads 2`: 34.8 s (the 1,000-step sweep row), 35.0 s and 36.8 s
(the two integration runs of the review rounds), 35.2 s (the CLI chain test) and
37.4 s (the worked example).

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM,
NVIDIA GeForce RTX 5070 Ti (driver 595.99.02; visible to every process but used
by none of these runs), Linux 7.0.0-42, CPython 3.14.10, eole 0.6.4, torch
2.14.1+cu130, sentencepiece 0.2.1, ctranslate2 4.8.4. CPU only. All dates
2026-11-07.

### Rule baseline

The reference is the English Wiktionary harvest in
`tests/fixtures/script_conv/wiktionary_reference.tsv`: 2,029 Khalkha Cyrillic
words with their traditional-script spelling, fetched through the MediaWiki API on
2026-11-07, one revision id per row, with its report beside it
(`wiktionary_reference.report.md`). Entries labelled Inner Mongolia, China,
Chakhar or Southern Mongolia were filtered out, as were entries marked for
verification, but Wiktionary gives no guarantee of Mongolia-only provenance. The
spellings are dictionary (etymological) spellings, not letter for letter: the
median script length is 1.17 times the Cyrillic length. Two rows fail
`validate_traditional` (`агентлаг`, a doubled selector; `руу`, a leading NNBSP)
and are dropped, which leaves 2,027.

The split is the parity of the SHA-256 of the word's UTF-8 bytes: 1,005 words
form the development half, which is the lexicon, and 1,022 the held-out half.
The fallback rules were designed on the development half only. Held-out was
measured once, after the rules were frozen:

| Split | Exact match | CER |
|---|---|---|
| Development (1,005 words) | 231 = 22.99% | 0.2345 |
| Held-out (1,022 words) | 237 = 23.19% | 0.2342 |

A word in the lexicon is spelled exactly (the lexicon is the development half
itself, so its hit rate on those words measures nothing). The held-out figure is
therefore the fallback's, and it is the baseline's honest figure for a word the
lexicon lacks: about three words in four come out different from the dictionary.

The fallback is one letter map plus five contextual rules, each kept only because
it raised the development exact-match rate by at least 0.5 percentage points,
each stated in one sentence in the comment above it in `fallback.py`:

| Part | Sentence | Development exact match after it |
|---|---|---|
| Letter map | One script spelling per Cyrillic letter (for example `в` is written BA, `й` and `ь` I, `ж` and `з` JA, `ц` and `ч` CHA, `ъ` is dropped; `я`, `ю`, `е`, `ё` are YA plus a vowel) | 129 = 12.84% (a first draft with `в` as WA, `й` as YA and `ь` dropped gave 100 = 9.95%) |
| 1 | After the word's first vowel letter, `о` is written U and `ө` is written UE | 138 = 13.73% |
| 2 | The second of two equal vowel letters (`аа`, `ээ`, `оо`, `уу`, `өө`, `үү`) is written GA and then its own vowel | 157 = 15.62% |
| 3 | A word-final `х` takes U in a back-vowel word and UE in a front-vowel word | 209 = 20.80% |
| 4 | A word-final `ч` or `ж` takes I | 220 = 21.89% |
| 5 | A word-final `т`, `ц` or `з` takes A in a back-vowel word and E in a front-vowel word | 231 = 22.99% |

A front-vowel word is one that holds `э`, `ө`, `ү` or `е`. A sixth contextual rule
(`нг` written ANG plus GA, +7 words on development) was left out because of the
cap of five. The fallback never emits a joiner, a selector, an MVS or an NNBSP:
those appear only in lexicon spellings, and a suffix gap is not guessed (ruling
3). A letter outside its map, such as the Kazakh `қ`, is dropped without a count.

Run `uv run pytest tests/unit/test_script_conv_fallback.py -q` to see these pinned.

### Tiny model

The profile (`configs/eole/converter_tiny.yaml`, rendered by `config.py`): 2 and 2
layers, hidden size 64 (word vector size 64), 4 heads, feed-forward 128,
`SinusoidalInterleaved` positions, shared vocabulary of at most 200 pieces
(70 are used), AdamW with Noam decay (`learning_rate` 0.5, `warmup_steps` 100),
dropout 0.0, label smoothing 0.1, 1,000 steps in batches of 1,024 tokens,
`fp32`, no GPU, `bucket_size` 1024, one worker. It trains from random
initialisation on 448 training words per direction (the 500-word fixture less 52
validation words).

| Figure | Value |
|---|---|
| Training, `--threads 2` | 34.8 s to 37.4 s (five runs) |
| Export | 2.3 s to 3.5 s |
| CER on the 448 training words, Cyrillic to script / script to Cyrillic | 0.0051 / 0.0128 |
| Word accuracy on the same words, `eval` | 0.9844 / 0.9554 |
| Words with a gap or selector that keep their marks | 78/81 = 96.3% |
| CTranslate2 against Eole, both directions | 448/448 |
| `ScriptConverter` word exact match against the reference | 441/448 = 0.9844 |
| `rule_fallbacks`, `invalid_outputs` over the training words | 0, 0 |
| The 52 validation words, Cyrillic to script: CER, word accuracy | 0.2444 (77 edits over 315 characters), 17/52 = 0.3269 |
| The 52 validation words, script to Cyrillic: CER, word accuracy | 0.2982, 14/52 = 0.2692 |

The last two rows were measured for this runbook with `eval` on the 52 pairs of
`cyrl2mong.valid.*`. They are the figure to remember: on words it did not train on
the tiny model is no better than the phonographic fallback, which makes the same
77 edits over the same 315 characters (and gets 13 of the 52 words exactly right
to the model's 17). That equal total is a coincidence of the sums, not a copy: the
two agree on only 16 of the 52 words. With 52 words and one training run, this is
an illustration, not a rate. Training is repeatable here (identical numbers in
repeated runs); another CPU or BLAS could move a word or two.

Why 1,000 steps with dropout 0. The tiny profile exists to memorise a fixture, so
the question is the least training that keeps the NNBSP and selector marks
reliably. The first choice, dropout 0.1, needed 1,500 steps to reach the 90% gate
by 0.1 point (73/81 = 90.1%). The marks-kept share by steps, dropout 0.1:

| Steps | 800 | 1000 | 1200 | 1400 | 1500 |
|---|---|---|---|---|---|
| Marks kept | 54% | 71% | 85% | 88% | 90.1% |

With dropout 0 the sweep was run one run at a time, to settle the smallest step
count with at least 93% of marks kept and both CERs at most 0.10 (a selection
rule; the test gate stays 90%):

| Steps | Train s | CER c2m | CER m2c (note) | Marks kept | NNBSP | MVS | FVS-1 | FVS-2 |
|---|---|---|---|---|---|---|---|---|
| 800 | 29.1 | 0.0192 | 0.0369 | 74/81 = 91.4% | 3/5 | 43/44 | 28/32 | 1/1 |
| 1000 | 34.8 | 0.0051 | 0.0533 | 78/81 = 96.3% | 4/5 | 44/44 | 30/32 | 1/1 |
| 1200 | 39.8 | 0.0113 | 0.0192 | 78/81 = 96.3% | 4/5 | 44/44 | 30/32 | 1/1 |
| 1500 | 46.9 | 0.0031 | 0.0373 | 80/81 = 98.8% | 5/5 | 44/44 | 31/32 | 1/1 |

The note: the CER m2c column was measured while `predict` ran every file under
the script tag (see "The Eole checkpoint overrides the prediction tag"), so it is
not comparable between rows or with the final 0.0128; the forward direction and
the marks column were not affected. 1,000 steps is the smallest row that meets
the rule. That is a hyperparameter change on a memorisation fixture, and no gate
was lowered.

### Throughput

Measured on the machine above, CPU only, with the model directory of the worked
example, `apply --target Mong` on a text of Cyrillic words joined by spaces. The
fixture has 500 distinct words, not 1,000, so two texts were used: the 500 words of
`tiny_words.tsv`, and a 1,000-word text made of those 500 plus 500 further lexicon
words that were not in the training data (for 117 of the 1,000 words the model
gave no usable script word and the rule baseline spelled it: `rule_fallbacks=117`,
all among the 500 unseen words). Every
`apply` is a new process, so every run is cold: a new interpreter, a new model
load, an empty word cache. Ten runs each, median, wall clock:

| Text | Words | Wall time of the command | Words per second |
|---|---|---|---|
| one word | 1 | 0.715 s | (the fixed cost: interpreter start-up and model load) |
| `tiny_words.tsv` words | 500 | 0.790 s | 630 |
| those plus 500 lexicon words | 1,000 | 0.820 s | 1,220 |

The command is dominated by its fixed cost. Net of the 0.715 s, the conversion
itself takes about 0.075 s for 500 words and 0.105 s for 1,000. Measured inside a
process (a fresh `ScriptConverter` per repetition, so the cache is cold, and only
`convert` timed; ten repetitions, median): 500 words in 0.043 s, 11,700 words per
second; 1,000 words in 0.096 s, 10,400 words per second. The figure to quote is
**about 10,000 words per second for the conversion once the model is loaded, and
about 1,200 words per second for the whole `apply` command on 1,000 words**,
one CTranslate2 thread, greedy decoding, batches of up to 1,024 words. Each
distinct word of a call is decoded once, so a text with many repeated words costs
less. Not measured: any other machine, a GPU, a text longer than 1,000 distinct
words, many threads.

### Length ratio of script over Cyrillic

`length_ratio_quantiles` over the (Cyrillic, script) pairs: the 2,027 reference
pairs that pass `validate_traditional`, and the 1,005 lexicon pairs. The ratio is
the number of code points of the script word over those of the Cyrillic word.

| Quantile | 0.01 | 0.05 | 0.5 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| Reference (2,027 pairs) | 0.8333 | 1.0000 | 1.1667 | 1.5000 | 1.9480 |
| Lexicon (1,005 pairs) | 0.8343 | 1.0000 | 1.1667 | 1.5000 | 1.9920 |

A note for the D4 length filter. These ratios are for single dictionary words, not
sentences, so a sentence-level ratio will be narrower; no sentence-level
measurement exists. They count every code point, NNBSP included (21 of the 2,027
reference words carry one), while the D4 length stage skips every character for
which `str.isspace()` is true, and that includes U+202F: without the NNBSP the 0.99
quantile of the reference falls to 1.7500 and the others do not move. **These
bounds are NOT wired into `configs/data/filters.yaml`.** The `mn_Mong` row of the
per-direction table in `docs/runbooks/04_filtering_mining.md` still has no
override.

## Facts found on this machine

All measured with Eole 0.6.4, CTranslate2 4.8.4 and CPython 3.14.10 on the machine
above. The spec's "Evidence from the spikes" holds the original measurements.

| Fact | Symptom when it is missed | What the code does |
|---|---|---|
| Every corpus with the `prefix` transform needs `tgt_prefix: ""` | `eole build_vocab` fails with `TypeError: object of type 'NoneType' has no len()` | Rendered on every corpus; the validator rejects a corpus without it |
| `model.embeddings.word_vec_size` must equal `hidden_size` | A d64 model fails with a layer-norm shape error (Eole's default word vector size is 512 and the master plan's template omits it) | `config.py` sets it from the profile; the validator rejects a difference |
| `training.bucket_size` must be small for a tiny corpus | With the default 262144, 300 steps of a d64 model took more than 12 minutes (about 360 tokens/s, 1.4 million log lines of "Weighted corpora loaded"): the loader cycles a 500-line corpus to fill the bucket | Tiny profile: `bucket_size` 1024, `num_workers` 1, `prefetch_factor` 8 and `OMP_NUM_THREADS=2` made the same 300 steps take 20 s. The full profile keeps Eole's defaults, untested |
| The Noam learning rate of the template is too high for d64 | `learning_rate: 2.0` peaks near 0.025 and the model does not learn | Tiny profile: `learning_rate` 0.5, `warmup_steps` 100: CER 0.935 after 300 steps and 0.007 after 600 on the toy letter-map task (500 words). The master plan's "300 CPU steps reach CER <= 0.15" is false as written |
| `python -m eole` does not run | `No module named eole.__main__; 'eole' is a package and cannot be directly executed` | `train.py` runs the console script that sits next to the interpreter (`.venv/bin/eole`) |
| Eole expands `$NAME` over the text of the YAML file | A `$` in a path or a string is silently replaced by an environment variable's value | The validator, `run_training` and `predict` reject a `$` in any path or rendered string (ConfigError, exit 2) |
| Eole reads ONE validation corpus, the one named `valid`, and trains on every other corpus name | A second validation corpus named `valid_mong2cyrl` would be TRAINED on | `valid` holds the Cyrillic-to-script direction only; the script-to-Cyrillic direction is trained but has no validation loss |
| The master plan's forbidden key names are not Eole fields | `model.max_relative_positions`, `model.alibi` and `model.num_kv` are rejected by Eole as unknown; the real fields `heads_kv`, `sliding_window` and `relative_positions_buckets` are accepted silently | `FORBIDDEN_KEYS` holds 16 real field names (rotary, relative position buckets, sliding window, layer types, grouped-query heads); a drift test imports Eole and checks that each is a real field and that no rotary, relative, sliding-window or grouped-query field of the installed Eole is missing |
| `eole_ct2` export SEGFAULTS when a CUDA device is visible | Exit code -11 (139 in a shell) and an empty `model.bin`; the fault is `ctypes.string_at` on a device pointer in `ctranslate2/specs/model_spec.py:757` (`Variable.to_bytes`) | `export_environment()` sets `CUDA_VISIBLE_DEVICES=""`. Reproduced again on 2026-11-07 on the tiny checkpoint: exit 139, `model.bin` 0 bytes. `RINCHEN_CT2_CUDA_PROBE=1` runs the probe as a test |
| Eole copies the `valid` corpus's `src_prefix` into the checkpoint and the checkpoint overrides the prediction config | `eole predict` ran every file under the Cyrillic-to-script tag, whatever the config said; see below | `predict()` tokenises and tags itself and runs Eole with `transforms: []` |
| SentencePiece merges unknown characters into one unknown piece spelled like the text | A tag whose characters the model never saw looks atomic: `encode(tag, out_type=str)` is `['▁', tag]` | `check_char_model` compares ids and requires `piece_to_id(tag) != unk_id` |
| `hard_vocab_limit=False` makes `vocab_size` a cap | A cap below the number of characters silently drops the rarest; the fixture trains to 73 pieces at a cap of 512. Eole's `src_vocab_size` is a cap too | `train` runs `check_char_model` on the corpus words after training; the tiny config uses 200 |
| CTranslate2 source tokens are `[tag, *sp.encode(text, out_type=str)]` | The dummy-prefix piece is a token of its own; `config.json` has `add_source_bos` and `add_source_eos` false | `ScriptConverter`, `parity` and `predict` build tokens this way |
| `str.split()` with no argument splits at U+202F | A script word is cut from its separated suffix | Every word split in the package is `split(" ")` or `segment()` |
| `-W error` on the command line crashes the SentencePiece SWIG import | A test run dies on import | Do not pass it |

The decoders: CTranslate2 and Eole agree on 448/448 words in both directions on the
tiny model, but on the 500-word spike model they differed on one word where both
fell into a degenerate repetition and stopped at different lengths, which is why
the parity gate is 0.99 and not 1.0.

### The Eole checkpoint overrides the prediction tag

Found when the reverse direction of the runtime disagreed with Eole on 67 of 448
words (agreement 0.85) while the forward direction agreed on all 448.
`ScriptConverter` and CTranslate2 were right; Eole was run wrongly.

Eole 0.6.4 (`train_single.ValidationConfigurer`) copies the `valid` corpus's
`src_prefix` (`<2mn_Mong>`, the Cyrillic-to-script direction) into
`transforms_configs.prefix` of the checkpoint's `config.json`, and
`config/run.py` (`_update_with_model_config`) lets the checkpoint's
`transforms_configs` override the prediction config, per transform name. So every
`eole predict` with the `prefix` transform ran under `<2mn_Mong>`, whatever tag
the prediction config asked for: forward output right by coincidence, reverse
output wrong (script words fed under the to-script tag).

Evidence, repeated for this runbook on the checkpoint of the worked example
(448 script words): the checkpoint's `config.json` holds
`transforms_configs.prefix.src_prefix` `<2mn_Mong>`; an `eole predict` with the
`sentencepiece` and `prefix` transforms and `src_prefix` `<2mn_Cyrl>` gave the same
output as one with `<2mn_Mong>` on 448 of 448 lines, and 382 lines matched the
reference, against 428 for `predict()` as it is now (67 lines differ between the
two). Earlier diagnostics (2026-11-07): CTranslate2 fed the wrong tag equals the
old Eole output on 448/448; patching the checkpoint's `config.json` to
`<2mn_Cyrl>` makes Eole equal CTranslate2 on 448/448; Eole with the source
pre-tokenised as `<tag> pieces...` and `transforms: []` equals CTranslate2 on
448/448 in both directions. The reverse CER went from 0.0533 (wrong tag) to
0.0128.

**Consequence for D8 and D11 (and any milestone that trains per-direction
corpora with the `prefix` transform).** The checkpoint remembers the validation
corpus's prefix and it wins over the prediction config. A per-direction
evaluation through Eole must therefore bypass the transforms: write the source
as tag plus SentencePiece pieces, joined by spaces, and run `eole predict` with
`transforms: []` and no `transforms_configs` (what `train.predict` does), or the
Eole side of a parity gate will be run under the wrong tag and the gate will
fail spuriously (or pass by coincidence in one direction). The tag has a weak
effect on Cyrillic-to-script output (a Cyrillic source under the other tag changes
25 of 448 words, 5.6%: the source script already tells the model its direction);
it matters mainly for the reverse direction and for homographs.

## Replacing the lexicon and the reference with real data

Real parallel data enters through `build-data --pairs`; the lexicon and the
reference are the stand-ins.

1. **Real pairs.** Write a UTF-8 TSV, LF, headed `cyrillic<TAB>script`, one
   Cyrillic word and its script spelling per line (the spelling may carry NNBSP,
   MVS and selectors; it must pass `validate_traditional`). Run `build-data
   --pairs`, `train`, `export` as above. `build-data` validates the script side
   only; the Cyrillic side is normalized and screened for emptiness and spaces.
   For aligned sentence lines, `data.align_words(cyrillic_line, script_line)`
   returns word pairs when both lines split on U+0020 into the same number of
   words, and `None` otherwise; it has no command, and it takes word order to
   match, which only the caller can know.
2. **The lexicon.** `src/rinchen/script_conv/lexicon.tsv` is package data, read
   when `rules.py` is imported: header `cyrillic<TAB>script`, lower-case keys, no
   duplicates, no blank lines (a malformed file is a `DataError` naming the line).
   Every spelling should satisfy `is_script_word`, so that it can stand in for an
   invalid model output; all 1,005 current ones do.
3. **The reference.** Replace `tests/fixtures/script_conv/wiktionary_reference.tsv`
   (columns `cyrillic` and `traditional` are read; the others are provenance) and
   run `uv run python scripts/build_script_conv_fixtures.py`: it rewrites the
   lexicon (development half), `rules_golden.tsv` (30 lexicon plus 70 fallback
   rows, 100 in all; `DataError` when the reference cannot give 100) and
   `tiny_words.tsv` (the first 500 pairs by sorted Cyrillic word). A test
   compares the committed files with its output byte for byte.
4. **Re-measure once.** The held-out constants in
   `tests/unit/test_script_conv_fallback.py` (`HELD_OUT_COUNT`,
   `HELD_OUT_EXACT_MATCH`, `HELD_OUT_CER`) describe the current reference and fail
   on another; measure the new held-out half once, set the constants from it, and
   treat that half as spent. Do not tune the fallback on it.
5. **Do not train a real model on the Wiktionary pairs.** They are a held-out
   reference, a lexicon of 1,005 words and a fixture, not parallel text.

## The full profile

The full profile (6 and 6 layers, hidden size 512, 8 heads, feed-forward 2048,
`bf16`, `gpu_ranks: [0]`, 100,000 steps, `learning_rate` 2.0, `warmup_steps` 4000,
batch size 8192 tokens, a vocabulary cap of 512, filtering at 128 pieces, Eole's
default bucket and loader settings) is rendered into `configs/eole/converter.yaml`
and passes `validate_converter_config`; unlike the tiny profile, it was never
loaded through Eole's own configuration class (`eole train` will do that on the
first real run). **Its values are untuned priors and its GPU path was never
run.** The operator step, once real data is in `data/script_conv/corpus`:

```bash
uv run rinchen convert-script train --profile full --confirm-full \
    --data-dir data/script_conv/corpus --run-dir data/script_conv/run_full --timeout 259200
```

Run it with `--dry-run` first (allowed without `--confirm-full`). Without
`--timeout` nothing bounds the run. The GPU stays visible to a full run (only the
tiny profile hides it); the export always hides it. The thresholds of the gates
table do not apply to a model trained on real data; define them from a
Mongolian-reviewed sample.

## Operational notes

- **Timeouts are per command.** `train --timeout` bounds each of `eole
  build_vocab` and `eole train` separately, so a run may take twice the value;
  `export --timeout` (default 3600 s) bounds the one converter command. A command
  that runs past its timeout is killed with everything it started and the error
  (exit 4) ends with the last 20 lines of its log, each cut to 300 characters.
- **Interrupts.** Each Eole or CTranslate2 command runs in a session of its own,
  so that its data-loader workers can be killed with it. `Ctrl-C` (SIGINT) in the
  terminal kills the whole process group of the running command and then ends the
  CLI. SIGTERM and SIGHUP to the CLI process are not handled: the CLI ends and
  the Eole child, in its own session, keeps running. Find it with `pgrep -af
  eole` and stop it by hand. These paths are tested on Linux only (the group kill
  and the tests that fork use POSIX facilities; off POSIX only the child is
  killed).
- **Manifest last.** Each command removes its old manifest before it touches an
  output (`begin_commit`) and writes the new one after every output exists, so a
  manifest never vouches for a mix of old and new files, and a directory that holds
  the manifest of another command is refused (exit 3) rather than erased. What a
  crash leaves: `build-data` leaves the corpus files finished so far (each
  appears whole, written split by split) and no manifest; a bad `--valid-fraction`
  is refused before the old manifest is touched, while a pairs file with no usable
  pair or with a line break in a field fails after it was removed. `train` leaves
  the character model, `converter.yaml`, the logs and any checkpoint written, and
  no manifest; a rerun into the same run directory overwrites them (the config
  sets `overwrite: true`) and a checkpoint older than the launch of `eole train`
  is never taken for the new one. `export` creates its model directory fresh; on a
  failure the directory is removed and the converter's log stays at
  `<out-dir>.export.log`; a failure to write the manifest removes the directory
  too, because a model directory without a manifest is untrusted. A power loss or
  SIGKILL during `export` can leave an unfinished directory: remove it by hand
  before retrying, because the converter refuses an existing one.
- **A `.part` file** beside an output is a temporary file of a killed run; it is
  safe to delete.
- **`predict`** has no command. It is a library function (`rinchen.script_conv.train.predict`)
  that the integration tests and later milestones use; it writes
  `<out>.log`, `.predict.yaml`, `.tokens.src` and `.tokens.out` beside its output.
- **`ScriptConverter` is not thread-safe** (its word cache is a plain dict): give
  each thread its own. The cache drops whole when a call would pass 100,000 words.
- **Editor tools rewrite escapes.** The Write and Edit tools of coding agents turn
  backslash-u escapes into raw characters. Invisible, look-alike and traditional
  Mongolian characters in source and test files are built with `chr(0x....)`;
  after editing run `grep -nP '[^\x00-\x7F]'`. In this runbook the only non-ASCII
  text is Cyrillic letters, the two script words of the worked example (plain
  letters, no selector or joiner) and the SentencePiece dummy-prefix mark.
- **Never write under `data/` or `models/` from tests or probes.** The tests and
  this runbook's measurements used temporary directories; the commands' default
  paths (`data/script_conv/...`) are for an operator.
- **The tests.** The unit and property tests run in about a minute; the integration
  tests (marked `subprocess`) add real Eole trainings and bring the full gate to
  about three minutes. `uv run pytest tests/integration -q` runs them alone.

## Open gates

None of these was done, because no real data or reader exists:

- **Audit of U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180A..U+180F and U+202F in
  real traditional-script text** (counts and neighbours), before the
  joiner-stripping policy is frozen. The D3 follow-up assigned it to D5; it is
  blocked on data. Until it is done, `normalize` keeps stripping the joiners and
  the converter never emits them.
- **Render check of the converter's output in Noto Sans Mongolian 2.001 and 3.103.**
  No shaping library is installed here.
- **Review of the lexicon, the golden words, the fallback and any converter output
  by a Mongolian reader.**
- **`mn_Mong` length-ratio bounds for the D4 filter.** Measured on the reference
  pairs (above), not wired into `configs/data/filters.yaml`, and never checked on
  sentences.
- **Training the full profile on real parallel script data.**

The D3 gates that the master plan assigned to D4 (abbreviation induction,
o-like substitute counts) are also still open; they are listed in
`docs/runbooks/04_filtering_mining.md`.

## What was not verified

- **No Mongolian reader has judged anything.** Not the lexicon, the golden words,
  the fallback's spellings, the model's output, nor the two script words printed in
  this runbook.
- **The render check in the target fonts was not done.** No shaping library is
  installed.
- **The audit of U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180A..U+180F and
  U+202F in real traditional-script text was not done.** There is no real text.
- **The Wiktionary reference may not be wholly independent of mechanical
  derivation,** and it cannot be guaranteed Mongolia-only although entries labelled
  Inner Mongolia, China or Chakhar were filtered out. Some editors may have derived
  spellings by a mechanical transliteration, and nothing in the wikitext tells
  those entries apart. The spellings are dictionary spellings, so the held-out
  exact match is low by construction.
- **The GPU and the full profile were never run.** Not the training, not
  `device="cuda"` in `ScriptConverter`, not bf16, not the default bucket
  settings; the full profile's YAML passed only our own validator.
- **Windows and macOS were not run.** The process-group kill and several tests are
  Linux only.
- **Every threshold is a prior.** The gates come from the master plan and the
  spec and were set on a fixture the model memorises.
- **The script-to-Cyrillic direction has no validation loss during training,**
  because Eole takes one validation corpus. Its quality is known only from the
  fixed-word checks above.
- **The wheel was not built or inspected** to confirm that `lexicon.tsv` and
  `standardized_variants.tsv` ship with it; the editable install and the tests
  read them.
- **Throughput was measured on one machine** with 500 to 1,000 distinct words and
  CPU only; nothing larger, no GPU, no concurrent callers.
- **Repeatability on other hardware** of the tiny run was not checked; the same
  machine gave identical numbers across runs.

## Where the code lives

`src/rinchen/script_conv/` (modules above, plus `lexicon.tsv` and
`standardized_variants.tsv`), `src/rinchen/cli/convert_script.py`,
`configs/eole/converter.yaml` and `converter_tiny.yaml` (renderings, never
hand-edited), `configs/spm/converter_char.yaml`,
`scripts/build_script_conv_fixtures.py`, `tests/fixtures/script_conv/`,
`tests/integration/`, unit and property tests named `test_script_conv_*.py` and
`test_cli_convert_script.py`, test helpers in `tests/support/` (`tiny_converter.py`,
`stub_runtime.py`, `fake_process.py`, `script_reference.py`) and
`manifests/ucd_standardized_variants.json`.
