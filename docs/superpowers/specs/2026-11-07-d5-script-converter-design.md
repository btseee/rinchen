# D5 - Cyrillic <-> traditional-script converter: design

Date: 2026-11-07. Milestone D5 of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`
(section "D5 - Cyrillic<->traditional converter"). Branch `d5-script-converter`. The
operating procedure and measured numbers are in `docs/runbooks/05_script_converter.md`.
Updated after the build to match the finished implementation: the contracts below are
the ones the code has, and "Amendments made while building" lists what changed against
the first version and why.

## Provenance

The user's instructions for this stretch were "continue dont stop" and "continue", on top of
the earlier "don't respond back until you finish everything related to coding". This design,
its plan and every ruling below were therefore made by the builder without a review round.
Nothing here is an approved decision; each ruling lists what it costs if wrong. The master
plan is the binding authority; where it is silent or wrong the rulings fill the gap.

## Intent

The system writes Cyrillic Mongolian by default and traditional script (`mn_Mong`) only
when the caller asks for it (`<2mn_Mong>`, `script="Mong"`). D5 builds the converter that
sits between the two: a trainable character-level model (Eole, from random init), the
deterministic rule baseline that bootstraps and sanity-checks it, the validator that keeps
forbidden control characters out of its output, the metrics, the data builder, the export
to CTranslate2 and a small runtime wrapper the serving engine will call.

There is no real parallel script data in the repository. The converter that D5 ships is
therefore a tested, trainable pipeline plus a tiny memorization model, not a quality
converter. The runbook says so.

## Goals

- `rinchen.script_conv` package with the modules listed under "Architecture".
- A rule baseline whose accuracy is measured against an independent reference (a
  Wiktionary harvest) with a held-out split, and reported as a number.
- A validator enforcing the D3 follow-up safeguard: the converter never emits ZWJ or ZWNJ.
- The tiny converter (2/2 layers, d64, character SentencePiece model) trains from random init
  on CPU, reaches CER <= 0.15 on its training fixture in both directions, exports to
  CTranslate2, matches Eole's own decoding, and is loadable through `ScriptConverter`.
- The full-profile converter config (6/6 layers, d512) is rendered and checked by
  `validate_converter_config` but not trained (no data, no GPU hours); only the tiny config is
  loaded through Eole's own config class.
- CLI: `rinchen convert-script build-data | train | export | apply | eval`.

## Non-goals

- Linguistic completeness. Cyrillic Mongolian follows pronunciation; traditional script
  follows historical spelling (reduced vowels, elided consonants). The mapping is lexical,
  not derivable by rule. The rule baseline is a scaffold and says so in its output type.
- Suffix morphology. The baseline does not split case suffixes or emit NNBSP itself (see
  ruling 3).
- Contextual disambiguation of homographs (the converter is word-level).
- Engine integration (`serve/` is D11/D12). D5 delivers the wrapper the engine will call.
- Rendering quality of the output in fonts (open gate, see "Open gates").
- Any Inner Mongolian data, variety or tool; any Latin-script informal Mongolian.

## Evidence from the spikes (2026-11-07, this machine, CPU)

Measured before this design was written; the rulings below depend on them.

- `eole build_vocab` and `eole train` run on CPU (`gpu_ranks: []`) with Eole 0.6.4 on
  Python 3.14.10 once these are fixed in the config: a corpus with the `prefix` transform
  needs an explicit `tgt_prefix: ""` (otherwise `TypeError: object of type 'NoneType' has no
  len()`); `model.embeddings.word_vec_size` must equal `hidden_size` (the master plan's
  template omits it, default 512, so a d64 model fails with a layer-norm shape error);
  and `training.bucket_size` must be small for a tiny corpus.
- The default `bucket_size: 262144` made 300 steps of a d64 model take more than 12 minutes
  (about 360 tokens/s, 1.4M log lines of "Weighted corpora loaded"): the loader cycles a
  500-line corpus to fill the bucket. With `bucket_size: 1024`, `num_workers: 1`,
  `prefetch_factor: 8` and `OMP_NUM_THREADS=2` the same 300 steps took 20 s.
- Noam `learning_rate: 2.0` (the template value) peaks near 0.025 at d64 and does not learn.
  `learning_rate: 0.5` with `warmup_steps: 100`: CER on the training set 0.935 after 300
  steps and 0.007 after 600 steps (toy letter-map task, 500 words). The master plan's
  "300 CPU steps reach CER <= 0.15" is therefore false as written; D15's rule applies
  (raise the steps, never lower the threshold).
- `eole predict` on CPU works through a YAML config (`transforms: [sentencepiece, prefix]`,
  `transforms_configs.prefix.src_prefix`); 500 lines in 3 s. (Found later, while building:
  with per-corpus prefixes Eole copies the validation corpus's prefix into the checkpoint
  and the checkpoint overrides this config, so the tag cannot be chosen at predict time
  this way. `predict` therefore tokenises and tags the source itself, see "`train` and
  `export`" and ruling N.)
- `python -m ctranslate2.converters.eole_ct2 --model_path <ckpt> --output_dir <dir>`
  SEGFAULTS when a CUDA device is visible (`ctypes.string_at` on a device pointer in
  `ctranslate2/specs/model_spec.py:757`, `Variable.to_bytes`) and leaves an empty
  `model.bin`. With `CUDA_VISIBLE_DEVICES=""` it writes a valid model.
- CTranslate2 4.8.4 vs Eole greedy decoding of the same checkpoint over the 500 training
  words: 499 identical; the one difference is a degenerate repetition where the two stop at
  different lengths. Source tokens for CTranslate2 are `["<2mn_Mong>", *sp.encode(text,
  out_type=str)]` (the dummy-prefix `▁` is its own piece); `config.json` has
  `add_source_bos/eos: false`.
- Python `str.split()` with no argument splits at U+202F NNBSP (`isspace()` is true), which
  would cut a traditional-script word from its separated suffix. Word segmentation must
  split on U+0020 only.
- Eole lists `[sentencepiece, prefix, filtertoolong]` as applied in the order
  `sentencepiece, filtertoolong, prefix`; the prefix is still added after SentencePiece, as
  the D3 follow-up decided.

Measured while building (Eole 0.6.4, CTranslate2 4.8.4, same machine):

- Eole takes ONE validation corpus: it reads `data["valid"]` and TRAINS on every other
  corpus name (`eole/inputters/text_corpus.py`, `get_corpora`).
- Eole expands `$NAME` over the text of the YAML config (`os.path.expandvars`), so a `$` in
  a path is rewritten silently.
- `python -m eole` does not run (no `__main__`); the console script next to the interpreter
  does.
- The master plan's forbidden key names `max_relative_positions`, `alibi` and `num_kv` are
  not Eole fields; the real ones are `heads_kv`, `sliding_window`,
  `relative_positions_buckets` and the rotary fields, which Eole accepts silently.
- The tiny profile at 1,000 steps with dropout 0 reaches CER 0.0051 (Cyrillic to script) and
  0.0128 (script to Cyrillic) on its 448 training words, keeps 78 of the 81 words that carry
  an NNBSP, MVS or selector, trains in 35 s and exports in 2.3 s; CTranslate2 and Eole agree
  on 448 of 448 words in both directions. With dropout 0.1 the marks needed 1,500 steps to
  reach 90%.

## Architecture

```
src/rinchen/script_conv/
  alphabet.py    code points: letters, FVS, MVS, NNBSP, NIRUGU, joiners, punctuation tables
  validate.py    validate_traditional(text) -> tuple[Violation, ...]; is_script_word(text)
  standardized_variants.tsv   UCD letter and selector pairs, package data
  segments.py    segment(text) -> tuple[Segment, ...]  (cyrillic / mongolian / other runs)
  eval.py        edit_distance, cer, word_accuracy, score_conversions, length_ratio_quantiles
  fallback.py    the data-free phonographic spelling: one letter map, five contextual rules
  rules.py       transliterate_word / transliterate: lexicon first, fallback second
  lexicon.tsv    verified Cyrillic -> script pairs (dev split of the reference), package data
  data.py        build converter corpora (Eole files) from word pairs
  char_model.py  train, check and load the converter's own character SentencePiece model
  config.py      render + validate the Eole converter config for a profile
  _subprocess.py run a command with its log, a timeout and a process-group kill
  train.py       run eole build_vocab / train / predict in a hardened subprocess
  export.py      CTranslate2 export (CUDA hidden) and the Eole-vs-CT2 parity check
  runtime.py     ScriptConverter: CT2 + SentencePiece, word cache, validity check, rule fallback
src/rinchen/cli/convert_script.py   build-data, train, export, apply, eval
configs/spm/converter_char.yaml  configs/eole/converter.yaml  configs/eole/converter_tiny.yaml
scripts/build_script_conv_fixtures.py   reference TSV -> lexicon, golden, tiny fixture
tests/fixtures/script_conv/   wiktionary_reference.tsv (+ report)  rules_golden.tsv  tiny_words.tsv
tests/integration/   conftest.py (session-scoped tiny run and export) and the tests that use it
tests/support/   tiny_converter.py  stub_runtime.py  fake_process.py  script_reference.py
manifests/ucd_standardized_variants.json
docs/runbooks/05_script_converter.md
```

Layering: `alphabet` is the leaf; `validate`, `segments`, `eval` depend on it; `fallback` on
nothing of the package; `rules` on `fallback`, `segments`, `alphabet`; `data` on `rules`,
`validate`, `alphabet`, `text.normalize`; `config` on `char_model`'s tags (and it hard-codes
`data`'s file layout by convention, which a test pins against `CORPUS_FILE_NAMES`); `train` on
`config`, `char_model` and `_subprocess`; `export` on `char_model` and `_subprocess`;
`runtime` on `segments`, `rules`, `validate`, `alphabet`, `char_model`, `text.normalize`.
The CLI is the only module that touches the manifest. No cycles.

## Contracts

### `alphabet`

- `LETTERS`: U+1820..U+1842 (the Mongolian letters of the Unicode block that Khalkha text
  uses; Todo, Sibe and Manchu letters from U+1843 up are not allowed).
- `FVS = {U+180B, U+180C, U+180D, U+180F}`, `MVS = U+180E`, `NNBSP = U+202F`, `NIRUGU =
  U+180A`, `JOINERS = {U+200C, U+200D}`.
- `CYRILLIC_TO_SCRIPT_PUNCTUATION = {".": U+1803, ",": U+1802}` and its inverse
  `SCRIPT_TO_CYRILLIC_PUNCTUATION`. Every other punctuation mark and every digit passes through
  unchanged (and `.` and `,` themselves stay inside numbers and Latin text, ruling F).

### `validate`

`validate_traditional(text) -> tuple[Violation, ...]`, `Violation(index: int, code: str,
detail: str)`. The codes (`VIOLATION_CODES`), each tested: `joiner` (ZWJ or ZWNJ anywhere),
`foreign_character` (anything that is not a letter, FVS, MVS, NNBSP, NIRUGU, space, digit,
punctuation in the mapping table, ASCII punctuation or a Latin letter run; a newline and a tab
are foreign, so callers validate one word or one stripped line at a time), `orphan_selector` (an
FVS or MVS that does not directly follow a letter, or an MVS that is not followed by a letter),
`misplaced_nnbsp` (an NNBSP not between two letters; a stem-final letter may carry its own
selector, so letter, FVS, NNBSP, letter is accepted, ruling E, because that is how `segments`
reads a suffix gap), `invalid_variant` (an FVS that `StandardizedVariants.txt` of the Unicode
Character Database does not list for its base letter). The standardized-variant table
(`standardized_variants.tsv`, 45 pairs) is a committed data file derived once from UCD 18.0.0,
with its source URL and SHA-256 recorded in a header comment and in
`manifests/ucd_standardized_variants.json`.

`is_script_word(text) -> bool` (ruling M) says whether a model output is one converted word: it
passes `validate_traditional`, holds at least one letter, and is made of letters, FVS, MVS,
NIRUGU and NNBSP only. A Cyrillic word can convert only to such a text, so an ASCII letter, a
digit, punctuation or a space inside a converted word is a defect, although running text may hold
them.

### `segments`

`segment(text) -> tuple[Segment, ...]`, `Segment(text: str, kind: Literal["cyrillic",
"mongolian", "other"])`. A Cyrillic segment is a maximal run of Cyrillic letters (including
combining marks inside it). A Mongolian segment is a maximal run of letters, FVS, MVS,
joiners and NIRUGU, plus NNBSP only when the character before it is such a character and the
character after it is a letter. A run of only selectors or joiners (no letter) is still kind
"mongolian"; callers that need a letter test for one. Everything else is "other". Concatenating the segments reproduces the input exactly. Split on U+0020 only;
never call `str.split()` without an argument on `mn_Mong` text.

### `rules` and `fallback`

- `transliterate_word(word: str) -> WordResult`, `WordResult(text: str, source:
  Literal["lexicon", "fallback"])`. Lookup is on the lower-cased word. A lexicon hit returns the
  verified spelling. The fallback is the table-driven phonographic transliteration of
  `fallback.py` (ruling G; `rules.py` re-exports its names): one letter map plus five contextual
  rules (the cap was "a handful"), each explainable in one sentence (the comment above it and the
  runbook) and each kept only because it raised the development exact-match rate by at least 0.5
  percentage points. A character outside the map is dropped without a count. The fallback never
  emits an FVS, MVS, NNBSP or joiner, and reads no file, so the fixture builder can regenerate a
  lost lexicon. Input is NFC: callers normalize (`data.rule_pairs` and the runtime do).
- `transliterate(text: str) -> str`: segments the text, converts Cyrillic segments with
  `transliterate_word`, maps `.` and `,` to U+1803 and U+1802 in the stretches between words
  unless the mark sits directly between two ASCII letters or digits (`3.5`, `1,000`, `a.b` stay;
  `5.`, `.5`, `...` and `1,` change; ruling F), passes everything else through.
- Every output of `transliterate_word` passes `validate_traditional` (a test over the whole
  lexicon and over generated words).
- `LEXICON: Mapping[str, str]` is loaded from `lexicon.tsv` with `importlib.resources`; a file
  that is missing or malformed (header, field count, empty field, blank line, upper-case key,
  duplicate key) is a `DataError` naming `lexicon.tsv:<line>`.
- `development_split(word)` is the parity of the SHA-256 of the word's UTF-8 bytes: development
  half (the lexicon) or held-out half.

### `eval`

`edit_distance(a, b) -> int` (Levenshtein over code points), `cer(references, hypotheses) ->
float` (total edits over total reference characters; an empty reference set is a `ValueError`,
not a value), `word_accuracy(references, hypotheses) -> float`, `ConversionScores(cer,
word_accuracy, n)` and `score_conversions(references, hypotheses)` that returns it,
`length_ratio_quantiles(pairs, quantiles) -> dict[float, float]` for the D4 note about `mn_Mong`
length-ratio bounds (each pair is `(cyrillic, script)`, the ratio is `len(script) /
len(cyrillic)`, linear interpolation as `numpy.quantile`).

### `data`

- `WordPair(cyrillic: str, script: str, origin: Literal["reference", "rule", "parallel"])`.
- `read_pairs_tsv(path, *, origin="reference") -> tuple[WordPair, ...]` reads a UTF-8 TSV headed
  `cyrillic<TAB>script` (ruling A; the reference file keeps its own columns), a `DataError`
  naming `path:line` for any defect.
- `align_words(cyrillic_line: str, script_line: str) -> tuple[WordPair, ...] | None`: splits
  both on U+0020, returns pairs only when each line holds a word and the word counts are equal,
  else `None`.
- `rule_pairs(words: Iterable[str], *, allow_fallback: bool = False) -> tuple[WordPair, ...]`:
  normalizes each word (`khk_Cyrl`) before the lookup (ruling H), and keeps only words whose
  `WordResult.source` is "lexicon" (verified), or all words when `allow_fallback=True`; a word
  that spells to nothing is skipped.
- `check_valid_fraction(f)` (`ConfigError` unless `0 < f <= 0.5`) and `CORPUS_FILE_NAMES`.
- `build_corpus(pairs, out_dir, *, valid_fraction, seed) -> CorpusFiles`: normalizes each side
  with `rinchen.text.normalize.normalize` (`khk_Cyrl` and `mn_Mong`) and drops a pair on the first
  check it fails, in this order, counted per code: `empty` (a side of only U+0020, or a Cyrillic
  side empty once normalized), `multiword` (a side that still holds U+0020 after its outer spaces
  are stripped, because a corpus line is one word; ruling H), the code of the first
  `validate_traditional` violation on the raw script side, `not_script_word` (the normalized
  script side is not one script word, `validate.is_script_word`: it has no letter, or holds an
  ASCII letter, a digit, punctuation or a space), `duplicate`. It splits train and
  valid by the SHA-256 of `"{seed}:{normalized Cyrillic word}"` read as a fraction of 2**256, so
  equal Cyrillic words never straddle the split, and writes, under `out_dir`,
  `cyrl2mong.{train,valid}.{src,tgt}` and `mong2cyrl.{train,valid}.{src,tgt}` (the second four
  are the first four with the sides swapped; UTF-8, LF, one word per line, input order kept).
  An empty train split and an empty valid split are both a `DataError` raised before any file is
  written (rulings H and V: Eole divides by the number of validation rows at its first validation
  step). A side holding a line break or tab is a `DataError` naming the
  pair index. Each file appears whole (through a temporary file) but the eight are not written
  atomically as a set: the caller owns `begin_commit` before the call and the manifest last.
  `CorpusFiles(directory, files, rows, dropped)` are read-only mappings keyed by file name and,
  for `dropped`, by the codes that fired (ruling B).

### `char_model`

`train_char_model(texts: Iterable[str], prefix: Path, *, vocab_size: int) -> Path` returns
`<prefix>.model`, with `<prefix>.vocab` beside it. Options pinned by a test: `model_type=char`,
`character_coverage=1.0`, `byte_fallback=False`, `normalization_rule_name=identity`,
`add_dummy_prefix=True` (kept, as in the D3 follow-up), `user_defined_symbols=[<2mn_Mong>,
<2mn_Cyrl>]` (`CONVERTER_TAGS`), `hard_vocab_limit=False`. The texts pass through
`<prefix>.input.txt`, which is removed afterwards (ruling J): the same texts into the same
prefix give byte-identical files, but the `.model` embeds its absolute prefix, so its hash holds
per location only, and the `.vocab` hash is the portable identity; manifests record both.
`vocab_size` is a cap, not a size: the model has one piece per distinct character plus the
special pieces and tags (73 pieces for the tiny fixture at a cap of 200 or 512), and with
`hard_vocab_limit=False` a cap below the character count silently drops the rarest characters.
`check_char_model(model, texts)` therefore runs after training, on a collection that can be read
again: it asserts that both tags are single pieces (compared by id, because a tag made of
characters the model has never seen merges into one unknown piece spelled like the tag) and that
every text has only known characters and decodes back to itself, so that every letter, FVS, MVS
and NNBSP of the training words round-trips. `load_char_processor(model)` opens a model with a
directory guard (SentencePiece aborts the interpreter on a directory), and
`load_char_model_config(path)` reads `configs/spm/converter_char.yaml`, which holds `vocab_size`
(512) only (rulings I and W: the trainer is deterministic, so a `seed` would be dead config, and
where the model goes is chosen per run, so an `output_prefix` would be one path shared by every
run). The loader stays strict: any other key, `output_prefix` included, is a `ConfigError`.

### `config`

`render_converter_config(profile: Literal["tiny", "full"], *, data_dir, spm_model, run_dir) ->
dict[str, Any]` returns the Eole config, every path a POSIX string;
`validate_converter_config(cfg, *, profile=None)` is the schema check (the `train` command and
`run_training` give it the profile of the rendering; a bare call skips the CPU-only checks): known top-level keys
only; no key of `FORBIDDEN_KEYS` at any depth, even with a neutral value (16 real Eole field
names: `rope_config` and the other rotary fields, `relative_positions_buckets`,
`sliding_window`, `layer_types`, `heads_kv`, `global_heads_kv`; a drift test imports Eole);
`position_encoding_type == "SinusoidalInterleaved"` wherever it appears, and an
`embeddings.position_encoding` that is present is `true`; `word_vec_size == hidden_size`, and so
does a `hidden_size` set on the encoder or the decoder; no string-valued `encoder`, `decoder`,
`training` or `transforms_configs` (Eole evaluates such a string as code); `tgt_prefix == ""` on every corpus; the `transforms` of
every corpus list `sentencepiece` before `prefix`; the corpora `cyrl2mong`, `mong2cyrl` and
`valid` present, each direction with its target-script tag as `src_prefix`; `share_vocab`; no
`$` in any key or string value (Eole expands `$NAME` over the YAML text); and, with
`profile="tiny"`, CPU only (`gpu_ranks == []`, `compute_dtype fp32`; without `profile` these two
checks are skipped). `dump_config(cfg, path)` writes it as YAML
through a temporary file; `read_converter_paths(cfg_path) -> ConverterPaths(checkpoint_dir,
char_model)` reads the two paths `export` needs back from a rendered file.

One validation corpus only: Eole reads `data["valid"]` for validation and trains on every other
name, and the two validation files cannot be concatenated (their prefixes differ). `valid`
therefore holds the Cyrillic-to-script direction; the script-to-Cyrillic direction is trained but
has no validation loss.

Tiny: encoder 2, decoder 2, hidden 64, heads 4, ff 128, `learning_rate 0.5`, `warmup_steps
100`, dropout 0.0 and attention dropout 0.0, `train_steps 1000`, `valid_steps 200`,
`save_checkpoint_steps 1000`, batch 1024 tokens, `bucket_size 1024`, `num_workers 1`,
`prefetch_factor 8`, `compute_dtype fp32`, a vocabulary cap of 200 and `filtertoolong` at 64
pieces (ruling K). Full (untuned priors, never trained): encoder 6, decoder 6, hidden 512, heads
8, ff 2048, `compute_dtype bf16`, `gpu_ranks [0]`, 100000 steps, `learning_rate 2.0`,
`warmup_steps 4000`, batch 8192 tokens, a vocabulary cap of 512, `filtertoolong` at 128 and
Eole's default bucket settings. The two profile files `configs/eole/converter_tiny.yaml` and
`converter.yaml` are the rendered output, checked in, and a test fails if a rendering differs
from its file.

### `train` and `export`

- `run_logged(command, *, env, log_path, what, timeout=None)` (`_subprocess.py`) runs a command
  with its combined output in a log file. On POSIX the child leads a session of its own; a
  timeout, or any exception raised while waiting (a `KeyboardInterrupt`), kills the child's whole
  process group (`os.killpg`, SIGKILL) and reaps it, so a data-loader worker does not outlive its
  command. SIGTERM and SIGHUP sent to the parent are not handled. An `UpstreamError` names the
  command, its exit code or signal, the log and the last 20 lines of it (each cut to 300
  characters). A timeout is per command, not a total.
- `run_training(cfg_path, run_dir, *, threads: int = 2, timeout: float | None = None,
  profile: Profile | None = None) -> Path` validates the config (for `profile` when it is
  given: the CPU-only checks of the tiny profile run only then; the `train` command passes it),
  runs `eole build_vocab -n_sample -1` then `eole train` as subprocesses
  (logs `build_vocab.log` and `train.log` in `run_dir`) with `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` set and the GPU hidden unless the config trains on one, and returns the
  config's `training.model_path`, which must lie inside `run_dir`. After a zero exit it requires a
  fresh `model.00.safetensors` there (`CHECKPOINT_FILE`; modified after `eole train` was launched,
  with 2 s of slack), so a stale checkpoint is never taken for the new one. `ConfigError` for a
  config that does not hold or a `$` in a path or the text; `UpstreamError` with the tail of the
  log on a non-zero exit, a timeout, a missing `eole` command or a missing checkpoint. The
  command is the console script beside the interpreter (`eole_commands(cfg_path)` returns both
  commands), because `python -m eole` does not run.
- `predict(ckpt_dir, src_file, *, tag, spm_model, out_file, threads=2, timeout=None) -> Path`
  (ruling N) converts the lines of `src_file` under `tag` greedily on CPU and returns `out_file`
  (one line per source line). It does not use Eole's `sentencepiece` and `prefix` transforms:
  Eole 0.6.4 copies the `valid` corpus's `src_prefix` into the checkpoint's
  `transforms_configs.prefix` and lets the checkpoint override the prediction config, so every
  prediction ran under the Cyrillic-to-script tag whatever `tag` said. It pre-tokenises with the
  character model (`<tag> piece piece ...`, one line per source line), runs `eole predict` with
  `transforms: []` and no `transforms_configs`, and decodes the raw token lines with the character
  model. Side files go next to the output (`predict_side_paths(out_file)` returns the named tuple
  `PredictSidePaths(log, config, token_source, token_output)`: `<out>.log`, `.predict.yaml`,
  `.tokens.src`, `.tokens.out`). `ConfigError` for a `tag` that is not one of `CONVERTER_TAGS` (or
  not a piece of the model) or a `$` in `ckpt_dir` or `out_file` (the two paths its config names;
  `$` is fine in `src_file` and `spm_model`); `DataError` for an
  unloadable model or an unreadable or empty source; `UpstreamError` as for training, or for a
  line count that differs from the source's. There is no command for it.
- `export_ct2(ckpt_dir, out_dir, *, timeout=None) -> Path` runs `python -I -W ignore -m
  ctranslate2.converters.eole_ct2` in a subprocess with `export_environment()` (`CUDA_VISIBLE_DEVICES=""`),
  its log at `<out_dir>.export.log`, and fails if `model.bin` is missing or empty. `out_dir` must
  not exist (the converter refuses it); a failed export can leave a partial one, which the CLI
  removes.
- `parity(ct2_dir, spm_model, src_lines, eole_lines, *, tag, max_decoding_length=128) ->
  ParityReport(agree, total, mismatches)` with the property `agreement`; the gate is
  `agreement >= 0.99`, not 1.0, because the two decoders can stop at different lengths on a rare
  degenerate repetition.

### `runtime`

`ScriptConverter(ct2_dir: Path, spm_model: Path, *, device="cpu")`, the constructor `from_parts(
translator, processor)` for stand-ins, and `convert(text: str, target: Literal["Mong", "Cyrl"])
-> Conversion`, `Conversion(text: str, rule_fallbacks: int, invalid_outputs: int)` (ruling L).

It segments the text; normalizes each distinct word of the source script like the training words
(lower-case and `khk_Cyrl` for Cyrillic, `mn_Mong` for script); converts each distinct word once
with a batched CTranslate2 call (`beam_size=1`, `max_decoding_length=min(256, 4 * longest + 8)`,
`max_batch_size=1024`, tokens `[tag, *pieces]`); and puts the text together again. A word longer
than 80 characters is not sent to the model. When converting to Cyrillic, a script run with no
letter (a lone selector, say) is not a word and passes through. The word cache is per instance, dropped whole when a call would pass 100,000 words, and an instance
is not thread-safe.

Validity is word-level (ruling M). For `target="Mong"` an output must satisfy `is_script_word`;
an empty, invalid or missing one (a too-long word included) is replaced by the rule baseline's
spelling and counted in `rule_fallbacks`; if the rules cannot spell the word either (only
letters they drop), the source word stays and counts in `invalid_outputs`. For `target="Cyrl"` an
output that is not exactly one Cyrillic word keeps the source text and counts in
`invalid_outputs`; `rule_fallbacks` is 0, since the baseline does not convert back. Between
words, `.` and `,` become the script's marks for Mong (as in `transliterate`, ruling F) and the
script's marks become `.` and `,` for Cyrl. Text with no word of the source script is returned
unchanged.

Errors: `ConfigError` for an unknown target or a device CTranslate2 rejects; `DataError` for a
character model or CTranslate2 model that is missing or cannot be loaded (the character model
loads first, being the cheap one); `UpstreamError` when the translator raises, returns another
number of results than words, or returns a result with no hypothesis (nothing of such a call is
cached).

### CLI (`rinchen convert-script`)

```
build-data (--pairs FILE | --words FILE) [--out-dir DIR] [--valid-fraction 0.1] [--seed 1234]
           [--allow-fallback]
train --profile tiny|full --data-dir DIR --run-dir DIR [--spm-config FILE] [--spm-prefix PREFIX]
      [--threads 2] [--dry-run] [--confirm-full] [--timeout SECONDS]
export --run-dir DIR [--config FILE] [--out-dir DIR] [--timeout 3600]
apply --model-dir DIR --target Mong|Cyrl [FILE]
eval --model-dir DIR --pairs FILE [--target Mong|Cyrl]
```

Errors map to exit codes through `rinchen.cli.errors` (`ConfigError` 2, `DataError` 3,
`UpstreamError` 4, anything else 1). `build-data`, `train` and `export` write a manifest last
with the commit protocol of D4 (`begin_commit`, temporary files, the manifest written after the
outputs).

- `build-data` takes exactly one of `--pairs` (a TSV headed `cyrillic<TAB>script`) and `--words`
  (Cyrillic words, spelled by `rule_pairs`; `--allow-fallback` also keeps the words the lexicon
  lacks); `--out-dir` defaults to `data/script_conv/corpus` and must not be the input's directory;
  a foreign manifest in it is refused (exit 3). It prints `wrote N train and M valid pairs to
  DIR`, `dropped K: code=n, ...` when pairs were dropped (and `W words read, P paired` first with
  `--words`), and `manifest: PATH`. Stage `script_conv_data`; `config_sha256` is the SHA-256 of
  the canonical JSON of the parameters; the drop counts are `metrics`.
- `train` owns the character model (ruling P): it reads the `cyrl2mong` corpus files only, since
  the `mong2cyrl` files hold the same words (ruling U), trains the character model with the
  `vocab_size` of `--spm-config` (default `configs/spm/converter_char.yaml`) at `--spm-prefix`
  (default `<run-dir>/converter_char`, ruling W: the model belongs to its run), runs
  `check_char_model` on the words, renders, validates (for the profile) and dumps the Eole
  config into `<run-dir>/converter.yaml`, and runs `run_training` with the same profile. All
  checks of the plan run before any write, in a dry run too, among them that
  `cyrl2mong.valid.src` holds a row (a corpus built elsewhere may not; `DataError`, exit 3,
  ruling V); `--dry-run` prints the character model path, the config
  path and the two Eole commands and writes and runs nothing; `--profile full` without
  `--dry-run` refuses (exit 2) unless `--confirm-full`; `--timeout` bounds each Eole command and
  has no default. It prints `trained PROFILE: checkpoint PATH` and `manifest: PATH`. The
  manifest (stage `script_conv_train`) has the eight corpus files and the character model config
  as inputs, the SHA-256 of the rendered `converter.yaml` as `config_sha256`, and lists the
  character `.model` and `.vocab`, `converter.yaml` and the checkpoint as outputs (ruling J); the
  profile is not in it (ruling T). There is no skip logic and no `--force`: a rerun overwrites
  under the commit protocol (ruling Q).
- `export` converts the newest checkpoint, copies the character model into the result as
  `converter_char.model` so that the model directory is self-contained (ruling O), and writes the
  manifest (stage `script_conv_export`) last. `--out-dir` defaults to `<run-dir>/ct2` and must
  not exist (checked again just before the converter runs, outside the cleanup that removes the
  directory); `--timeout` defaults to 3600 s. When `<run-dir>/manifest.json` is a
  `script_conv_train` manifest that lists the character model, `export` compares its SHA-256
  with the recorded one and refuses a difference (`DataError`, exit 3, both hashes in the message)
  before any write; without that manifest a hand-built run directory is trusted (ruling W). On a
  failure the model directory is removed and the converter's log stays beside it.
- `apply` converts the whole input as one text, its line breaks included (ruling R), writes it to
  standard output exactly (no newline added) and prints `rule_fallbacks=N invalid_outputs=M` on
  standard error; empty input exits 0 with no output; an unknown `--target` exits 2.
- `eval` converts one side of each pair of a pairs TSV in one call and prints `cer`,
  `word_accuracy` and `n`, then `rule_fallbacks` and `invalid_outputs`; a pairs file with no pairs
  exits 3.
- `apply` and `eval` need only `--model-dir`. The commands look up `train_char_model`,
  `run_training`, `export_ct2` and `load_converter` at call time so that tests can replace them
  (ruling S).

## Testing strategy and gates

Tests first, per task. Numeric gates (a missed gate means more steps or a fix, never a lower
threshold):

- Rules golden: exactly 100 words in `tests/fixtures/script_conv/rules_golden.tsv`, 30 whose
  output comes from the lexicon and 70 held-out words the fallback converts correctly (ruling D:
  fewer fallback rows and more lexicon rows if the reference cannot give 70; it gave 70); all
  100 pass, and no fallback word is in the lexicon.
- Reference accuracy: the fallback's exact-match rate and CER on the held-out split (lexicon
  disabled) were measured once at the end of the rules task (237/1022 = 0.2319, CER 0.2342) and
  written to the runbook; a test pins the measured values to within 0.02.
- Validator: every code in the list has a failing and a passing example; the whole lexicon and
  the reference script column pass; `hypothesis` property: `validate_traditional` never raises.
- `cer`: identical strings 0.0; known edit counts; `hypothesis`: `edit_distance` is symmetric,
  bounded by the longer length, and zero only for equal strings.
- `segments`: concatenation reproduces the input (`hypothesis`); an NNBSP between letters does
  not split the word; splitting on U+0020 only.
- Corpus: counts exact on a planted list (duplicates, a violating pair, a multiword pair, an
  empty pair, a pair whose script side is not one script word); train and valid disjoint;
  deterministic for a seed; an empty train split and an empty valid split are errors.
- Char model: tags single pieces (by id); the training words round-trip, letters, FVS, MVS and
  NNBSP included; the same texts into the same prefix give identical files.
- Config: rendered YAML equals the checked-in files; every forbidden key and every violated
  invariant (a `$` included) is rejected by `validate_converter_config`; Eole's own config class
  loads the rendered tiny config.
- Tiny training (marked `subprocess`, ruling C: two SESSION-scoped fixtures in
  `tests/integration/conftest.py`, `tiny_run` for the corpus, character model, config,
  checkpoint and Eole predictions in both directions, and `tiny_ct2` for the export, shared by
  every integration test): CER <= 0.15 on the training fixture in BOTH directions, measured with
  the greedy Eole predictions; at least 90% of the words that carry an NNBSP, MVS or selector
  keep their marks; wall time under 150 s on this machine; the tiny profile trains 1000 steps
  with dropout 0 (ruling K).
- Export parity: CTranslate2 vs Eole, `agree / total >= 0.99` on the training fixture in BOTH
  directions; the export subprocess runs with CUDA hidden and the test also runs it once with CUDA
  visible only when `RINCHEN_CT2_CUDA_PROBE=1` (documents the segfault; off by default). A
  regression test pins that `predict` converts under the tag it is given: script sources under the
  wrong tag differ on at least 10% of the words (67/448 = 15% measured).
- Runtime: word exact match of `ScriptConverter` against the reference script words >= 0.90
  (441/448 measured) and agreement with the Eole predictions >= 0.99 in both directions (448/448);
  an output forced to be invalid (a joiner, an ASCII letter, a digit, a space, empty) is replaced
  by the rule word and counted; an unknown target raises; text without a word of the source
  script is returned unchanged; suffix gaps survive a round trip.
- CLI: `--help` for each command, `build-data` then `train --dry-run` golden command
  rendering, error exit codes, manifest-last behaviour (a failed `export` leaves no manifest and
  no partial directory), hashes recorded before the work. Integration: the real `rinchen` binary
  exports the session checkpoint, `apply` equals `ScriptConverter` byte for byte, `eval` is under
  the CER gate in both directions, and one test drives the whole chain (`build-data`, `train
  --profile tiny`, `export`, `apply`, `eval`) on the tiny fixture in a temporary directory.
- STOP gate: the tiny model trained from the fixture is exported and loaded through
  `ScriptConverter`, and `apply --target Mong` converts a fixture line, all in the tests above.

The master plan also lists "CT2 export parity (D9 utilities)": D9 is document-level work; no
such utilities exist, so D5 builds the parity check it needs in `export.py`.

## Rulings

Each is the builder's call; "cost" is what a wrong ruling costs.

1. **The engine half of the STOP gate becomes `ScriptConverter`.** The master plan says "served
   via engine with `script="Mong"`"; there is no engine until D11/D12. Its result is
   `Conversion(text, rule_fallbacks, invalid_outputs)` (ruling L). Cost if wrong: the engine
   task adapts to this small interface.
2. **Word-level units.** The model converts one word per line; text is segmented, words are
   deduplicated and cached. Aligned sentence pairs are usable only when word counts match.
   Cost: homograph context is lost; a sentence-level model would need a different data layout.
3. **No suffix splitting in the rule baseline.** The master plan lists "suffix rules + NNBSP
   before case suffixes" under `rules.py`. Case-suffix spelling depends on stem class (for
   example stems with a hidden final -n take different genitives) and the reference harvest
   holds lemmas only, so rules written here could not be checked and would put silently wrong
   suffixes into training data. NNBSP therefore reaches the training data only through lexicon
   or parallel entries that already carry it. Cost: the model gets no rule-made suffix gaps;
   real data has to teach them.
4. **NNBSP before separated suffixes, MVS only where lexical, ZWJ/ZWNJ never.** This follows
   the D3 follow-up decision and is not reopened, even though the current Unicode text and the
   W3C notes prefer MVS for the suffix gap. Cost: see that decision.
5. **Both directions in one model**, tags `<2mn_Mong>` and `<2mn_Cyrl>`, one corpus per
   direction (the D3 follow-up's prefix design). The rule baseline covers only Cyrillic to
   script. Eole takes one validation corpus, so `valid` is the Cyrillic-to-script direction and
   the other direction has no validation loss. Cost: the reverse direction is learned only from
   rule/reference pairs, which makes it many-to-one in the data; its CER is reported, not
   assumed, and it is known only from fixed-word checks.
6. **1000 steps with dropout 0 for the tiny model, threshold unchanged (CER <= 0.15), both
   directions, Noam lr 0.5.** The first version of this ruling said 600 to 800 steps: that
   passes the CER gate on the toy letter-map task but keeps only 54% of the words that carry an
   NNBSP or selector (dropout 0.1, 800 steps), and 1500 steps were needed to reach 90% with
   dropout 0.1. With dropout 0, the smallest of 800, 1000, 1200 and 1500 steps that keeps at least
   93% of the marked words with both CERs at most 0.10 is 1000 (78/81 = 96.3%, CER 0.0051 and
   0.0128). Cost: a slower tiny test than the master plan expects (about 35 s of training), and a
   hyperparameter that suits a memorization fixture only (ruling K).
7. **Eole configs are rendered by code and validated** (`config.py`), not hand-edited; the
   master plan's template is corrected for the three failures found (`tgt_prefix`,
   `word_vec_size`, `bucket_size`), and the validator also rejects the real forbidden Eole field
   names and a `$` in any key or string (Eole expands it). Cost: D7's `eole_cfg.render` should absorb
   `config.py`.
8. **CTranslate2 export always runs with CUDA hidden.** Cost: none on CPU export; an export
   that wants the GPU does not exist.
9. **Own character SentencePiece trainer** instead of extending `spm.train.train_spm` (which is
   unigram, byte-fallback and per-language-directory by construction, with a test that pins
   its options). Cost: about 30 lines of trainer wiring duplicated.
10. **An invalid neural output is replaced by the rule baseline's word** and counted
    (`rule_fallbacks`), rather than dropped or left in Cyrillic; an output with no usable
    spelling at all keeps its source word and is counted separately (`invalid_outputs`, ruling
    L). "Invalid" is judged per word (ruling M). Cost: some outputs are the phonographic guess
    instead of the model's.
11. **Reference provenance.** The Wiktionary harvest is used as a held-out reference and as
    the source of the lexicon and tiny fixture, never as training data for a real model. Its
    revision ids are in the TSV. Wiktionary entries carry no guarantee of Mongolia-only
    provenance; the harvest filters out entries labelled Inner Mongolia, Chakhar and China and
    takes only the `==Mongolian==` section, but a Mongolian reader has not checked it. The
    harvest scripts are not in the repository (ruling X), so the fetch cannot be re-run from the
    tree. Cost: a wrong lexicon entry is wrong training data for the tiny fixture only.
12. **Digits and punctuation.** Digits pass through unchanged; `.` and `,` map to U+1803 and
    U+1802 (the marks the D3 splitter already treats as `mn_Mong` boundaries), except where the
    mark sits directly between two ASCII letters or digits (`3.5`, `1,000`, `a.b`; ruling F).
    Cost: one table and one pattern.
13. **Dev and held-out split** of the reference by the parity of the SHA-256 of the word's UTF-8
    bytes (1005 development words, 1022 held-out), the rule development sees only the dev half;
    the held-out figure (237/1022 = 0.2319, CER 0.2342) was computed once. Cost: the held-out
    half is spent after that figure is published.

### Amendments made while building

The controller of the build made these rulings without a review round; each is one line here and
a decision with its cost in `DECISIONS.md`.

- A. Every pairs TSV (`tiny_words.tsv`, `--pairs`) is headed `cyrillic<TAB>script`; the reference
  file keeps its own columns.
- B. `CorpusFiles.files`, `rows` and `dropped` are keyed by file name and drop code.
- C. The integration fixtures `tiny_run` and `tiny_ct2` are session-scoped, in
  `tests/integration/conftest.py`.
- D. If the held-out split gives fewer than 70 fallback-correct words, the golden is filled with
  lexicon rows to 100 and the shortfall is reported (no shortfall occurred).
- E. The validator accepts letter, FVS, NNBSP, letter: a stem-final selector may precede the
  suffix gap.
- F. `.` and `,` between two ASCII letters or digits are not punctuation and stay unchanged.
- G. The data-free fallback lives in `fallback.py`; `rules.py` re-exports it and the fixture
  builder imports only `fallback`.
- H. `rule_pairs` normalizes before the lookup; an empty train split is a `DataError` before any
  file is written (an empty valid split was allowed here; ruling V revises that); a side with a
  space is dropped under the new code `multiword`.
- I. `converter_char.yaml` has `vocab_size` and `output_prefix` only; no `seed` (ruling W later
  removes `output_prefix`).
- J. The character trainer reads `<prefix>.input.txt` (deterministic); the `.vocab` hash is the
  portable identity and manifests record both files.
- K. The tiny profile uses dropout 0 and 1000 steps (ruling 6).
- L. `Conversion` has three fields: `text`, `rule_fallbacks`, `invalid_outputs`.
- M. A converted word is valid when `is_script_word` says so.
- N. `predict` pre-tokenises and runs Eole with no transforms, because the checkpoint overrides
  the prediction tag.
- O. A model directory is self-contained: `export` copies the character model into it as
  `converter_char.model`.
- P. `train` owns the character model: it trains and checks it, renders the config and runs
  Eole.
- Q. No skip logic and no `--force`: a rerun overwrites under the commit protocol.
- R. `apply` converts the whole input as one text and prints the counters on standard error;
  `eval` prints CER, word accuracy and `n` and exits 3 on an empty pairs file.
- S. The commands look up `train_char_model`, `run_training`, `export_ct2` and `load_converter` at
  call time (test seams).
- T. The profile is not recorded in `Manifest.status`.
- U. The character model is trained on the `cyrl2mong` files only.
- V. An empty valid split is a `DataError` in `build_corpus`, before any write (revises H: `eole
  train` dies with a `ZeroDivisionError` at its first validation step on a corpus with no valid
  row), and `train` refuses a `cyrl2mong.valid.src` without a row before any write, in a dry run
  too. Cost: a one-pair or very small corpus can no longer be built.
- W. `train` writes the character model into the run directory, `<run-dir>/converter_char.*`,
  unless `--spm-prefix` says otherwise, and `output_prefix` is removed from
  `converter_char.yaml` and `CharModelConfig` (amends I); `export` compares the character model's
  SHA-256 with the run's train manifest when that lists it. Cost: runs are self-contained, but
  the character model is retrained per run directory, the default path changed and the YAML lost
  a key.
- X. The scripts that harvested the Wiktionary reference are scratch and are not in the
  repository. Cost: refreshing the reference needs new tooling that meets the contract in the
  runbook.

## Open gates (not done by D5, because no real data or reader exists)

- Audit of U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180A..U+180F and U+202F in real
  traditional-script text, before the joiner-stripping policy is frozen (assigned to D5 by the
  D3 follow-up; blocked on data).
- Render check of converter output in Noto Sans Mongolian 2.001 and v3.103 (no shaping
  library is installed; the previous check used a throwaway environment).
- Review of the lexicon, the golden words and the fallback by a Mongolian reader.
- `mn_Mong` length-ratio bounds for the D4 filter: the quantiles were measured on the
  reference pairs (script over Cyrillic length 0.83, 1.00, 1.17, 1.50 and 1.95 at the 0.01,
  0.05, 0.5, 0.95 and 0.99 quantiles) and written to the runbook, not wired into
  `configs/data/filters.yaml`; no sentence-level measurement exists.
- Training the full profile on real parallel script data.

## Limitations

- The rule baseline is a measured scaffold, not an orthography. Its held-out accuracy is in
  the runbook.
- The tiny model memorizes a fixture; its CER on unseen words says nothing about quality.
- Word-level conversion cannot use sentence context.
- The CT2 converter's CUDA segfault is an upstream defect worked around, not fixed.
- The checkpoint's override of the prediction tag is Eole behaviour worked around in `predict`,
  not fixed; any per-direction Eole evaluation must bypass the transforms.
- The script-to-Cyrillic direction has no validation loss (one validation corpus).
- The full profile's values are untuned priors and its GPU path was never run.
