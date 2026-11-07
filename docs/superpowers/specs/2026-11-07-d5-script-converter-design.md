# D5 - Cyrillic <-> traditional-script converter: design

Date: 2026-11-07. Milestone D5 of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`
(section "D5 - Cyrillic<->traditional converter"). Branch `d5-script-converter`. The
operating procedure and measured numbers will be in `docs/runbooks/05_script_converter.md`.

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
- The full-profile converter config (6/6 layers, d512) is rendered and schema-validated but
  not trained (no data, no GPU hours).
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
  `transforms_configs.prefix.src_prefix`); 500 lines in 3 s.
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

## Architecture

```
src/rinchen/script_conv/
  alphabet.py    code points: letters, FVS, MVS, NNBSP, NIRUGU, joiners, punctuation table
  validate.py    validate_traditional(text) -> tuple[Violation, ...]
  segments.py    segment(text) -> tuple[Segment, ...]  (cyrillic / mongolian / other runs)
  rules.py       transliterate_word / transliterate: lexicon first, phonographic fallback
  lexicon.tsv    verified Cyrillic -> script pairs (dev split of the reference), package data
  eval.py        edit_distance, cer, word_accuracy, ConversionScores
  data.py        build converter corpora (Eole files) from word pairs or aligned lines
  char_model.py  train the converter's own character SentencePiece model
  config.py      render + validate the Eole converter config for a profile
  train.py       run eole build_vocab / train / predict in a hardened subprocess
  export.py      CTranslate2 export (CUDA hidden) and the Eole-vs-CT2 parity check
  runtime.py     ScriptConverter: CT2 + SentencePiece, word cache, validator fallback
src/rinchen/cli/convert_script.py   build-data, train, export, apply, eval
configs/spm/converter_char.yaml  configs/eole/converter.yaml  configs/eole/converter_tiny.yaml
scripts/build_script_conv_fixtures.py   reference TSV -> lexicon, golden, tiny fixture
tests/fixtures/script_conv/   wiktionary_reference.tsv  rules_golden.tsv  tiny_words.tsv
docs/runbooks/05_script_converter.md
```

Layering: `alphabet` is the leaf; `validate`, `segments`, `eval` depend on it; `rules`
on `alphabet`, `segments`; `data` on `rules`, `validate`, `text.normalize`; `config`, `train`,
`export` on `data`'s file layout only; `runtime` on `segments`, `rules`, `validate`. The CLI is
the only module that touches the manifest. No cycles.

## Contracts

### `alphabet`

- `LETTERS`: U+1820..U+1842 (the Mongolian letters of the Unicode block that Khalkha text
  uses; Todo, Sibe and Manchu letters from U+1843 up are not allowed).
- `FVS = {U+180B, U+180C, U+180D, U+180F}`, `MVS = U+180E`, `NNBSP = U+202F`, `NIRUGU =
  U+180A`, `JOINERS = {U+200C, U+200D}`.
- `CYRILLIC_TO_SCRIPT_PUNCTUATION = {".": U+1803, ",": U+1802}` and its inverse. Every other
  punctuation mark and every digit passes through unchanged.

### `validate`

`validate_traditional(text) -> tuple[Violation, ...]`, `Violation(index: int, code: str,
detail: str)`. The codes, each tested: `joiner` (ZWJ or ZWNJ anywhere), `foreign_character`
(anything that is not a letter, FVS, MVS, NNBSP, NIRUGU, space, digit, punctuation in the
mapping table, ASCII punctuation or a Latin letter run), `orphan_selector` (an FVS or MVS
that does not directly follow a letter, or an MVS that is not followed by a letter),
`misplaced_nnbsp` (an NNBSP not between two letters), `invalid_variant` (an FVS that
`StandardizedVariants.txt` of the Unicode Character Database does not list for its base
letter). The standardized-variant table is a committed data file derived once from UCD 18.0.0,
with its source URL and sha256 recorded in a header comment and in `manifests/`.

### `segments`

`segment(text) -> tuple[Segment, ...]`, `Segment(text: str, kind: Literal["cyrillic",
"mongolian", "other"])`. A Cyrillic segment is a maximal run of Cyrillic letters (including
combining marks inside it). A Mongolian segment is a maximal run of letters, FVS, MVS,
joiners and NIRUGU, plus NNBSP only when it sits between two such characters. Everything else
is "other". Concatenating the segments reproduces the input exactly. Split on U+0020 only;
never call `str.split()` without an argument on `mn_Mong` text.

### `rules`

- `transliterate_word(word: str) -> WordResult`, `WordResult(text: str, source:
  Literal["lexicon", "fallback"])`. Lookup is on the lower-cased word. A lexicon hit returns
  the verified spelling. The fallback is a table-driven phonographic transliteration (one
  letter map plus at most a handful of contextual rules, each explainable in one sentence in
  the runbook and each shown to improve the development split).
- `transliterate(text: str) -> str`: segments the text, converts Cyrillic segments with
  `transliterate_word`, maps the two punctuation marks, passes everything else through.
- Every output of `transliterate_word` passes `validate_traditional` (a test over the whole
  lexicon and over generated words).
- `LEXICON: Mapping[str, str]` is loaded from `lexicon.tsv` with `importlib.resources`.

### `eval`

`edit_distance(a, b) -> int` (Levenshtein over code points), `cer(references, hypotheses)
-> float` (total edits over total reference characters; an empty reference set is an
error, not a value), `word_accuracy(references, hypotheses) -> float`,
`ConversionScores(cer, word_accuracy, n)`, `length_ratio_quantiles(pairs, qs)` for the D4
note about `mn_Mong` length-ratio bounds.

### `data`

- `WordPair(cyrillic: str, script: str, origin: Literal["reference", "rule", "parallel"])`.
- `align_words(cyrillic_line: str, script_line: str) -> tuple[WordPair, ...] | None`: splits
  both on U+0020, returns pairs only when the word counts are equal, else `None`.
- `rule_pairs(words: Iterable[str]) -> tuple[WordPair, ...]`: only words whose
  `WordResult.source` is "lexicon" (verified) or all words when `allow_fallback=True`.
- `build_corpus(pairs, out_dir, *, valid_fraction, seed) -> CorpusFiles`: normalizes each
  side with `rinchen.text.normalize.normalize` (`khk_Cyrl` and `mn_Mong`), drops a pair whose
  script side fails `validate_traditional` (counted per reason in the return value), dedups,
  splits train and valid by a stable hash of the Cyrillic word, and writes, under `out_dir`,
  `cyrl2mong.{train,valid}.{src,tgt}` and `mong2cyrl.{train,valid}.{src,tgt}` (UTF-8, LF, one
  word per line). `CorpusFiles` lists the paths, row counts and the drop counts.

### `char_model`

`train_char_model(texts: Iterable[str], prefix: Path, *, vocab_size: int) -> Path`. Options
pinned by a test: `model_type=char`, `character_coverage=1.0`, `byte_fallback=False`,
`normalization_rule_name=identity`, `add_dummy_prefix=True` (kept, as in the D3 follow-up),
`user_defined_symbols=[<2mn_Mong>, <2mn_Cyrl>]`, `hard_vocab_limit=False`. `check_char_model`
asserts that both tags are single pieces and that every letter, FVS, MVS and NNBSP in the
training text round-trips through encode and decode unchanged.

### `config`

`render_converter_config(profile: Literal["tiny", "full"], *, data_dir, spm_model, run_dir)
-> dict[str, Any]` returns the Eole config; `validate_converter_config(cfg)` is the schema
check (known keys only; `position_encoding_type == "SinusoidalInterleaved"`; none of rotary,
alibi, relative position, sliding window or GQA keys; `word_vec_size == hidden_size`;
`tgt_prefix == ""` on every corpus; both direction corpora present; `share_vocab`). Tiny:
encoder 2, decoder 2, hidden 64, heads 4, ff 128, `learning_rate 0.5`, `warmup_steps 100`,
`train_steps 800`, `bucket_size 1024`, `num_workers 1`, `prefetch_factor 8`, `compute_dtype
fp32`. Full: encoder 6, decoder 6, hidden 512, heads 8, ff 2048, `compute_dtype bf16`,
default bucket settings. The two profile files `configs/eole/converter_tiny.yaml` and
`converter.yaml` are the rendered output, checked in, and a test fails if a rendering differs
from its file.

### `train` and `export`

- `run_training(cfg_path, run_dir, *, threads: int = 2) -> Path` runs `eole build_vocab -n_sample
  -1` then `eole train` as subprocesses with `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set,
  returns the checkpoint directory, raises `UpstreamError` with the tail of the log on a
  non-zero exit.
- `predict(ckpt_dir, src_file, *, tag: str, spm_model: Path, out_file: Path)` runs `eole predict`
  on CPU with the same prefix transform.
- `export_ct2(ckpt_dir, out_dir) -> Path` runs the converter in a subprocess with
  `CUDA_VISIBLE_DEVICES=""` and fails if `model.bin` is empty.
- `parity(ct2_dir, spm_model, src_lines, eole_lines, tag) -> ParityReport(agree, total,
  mismatches)`; the gate is `agree / total >= 0.99`.

### `runtime`

`ScriptConverter(ct2_dir: Path, spm_model: Path, *, device="cpu")`,
`convert(text: str, target: Literal["Mong", "Cyrl"]) -> Conversion`,
`Conversion(text: str, rule_fallbacks: int)`. It segments the text, converts each distinct word
once (cache) with a batched CTranslate2 call (`beam_size=1`, `max_decoding_length` from the
word length), validates every traditional-script output, and replaces an invalid output by
the rule baseline's word (counted in `rule_fallbacks`; only possible for `target="Mong"`;
for `"Cyrl"` an output that is not Cyrillic stays and is counted). `target="Cyrl"` on text
that has no Mongolian-script segment, and the reverse, return the input unchanged. Raises
`ConfigError` for another target.

### CLI (`rinchen convert-script`)

`build-data --words FILE | --pairs FILE --out-dir DIR [--valid-fraction 0.1] [--seed N]`,
`train --profile tiny|full --data-dir DIR --run-dir DIR [--dry-run]`, `export --run-dir DIR
--out-dir DIR`, `apply --model-dir DIR --target Mong|Cyrl [FILE]` (stdin to stdout when no
file), `eval --model-dir DIR --pairs FILE`. Errors map to exit codes through
`rinchen.cli.errors`; `build-data`, `train` and `export` write a manifest last with the
same commit protocol as D4 (`begin_commit`, `.part` files, manifest written last). `train
--profile full` without `--dry-run` refuses unless `--confirm-full` is given.

## Testing strategy and gates

Tests first, per task. Numeric gates (a missed gate means more steps or a fix, never a lower
threshold):

- Rules golden: exactly 100 words in `tests/fixtures/script_conv/rules_golden.tsv`, 30 whose
  output comes from the lexicon and 70 held-out words the fallback converts correctly; all
  100 pass, and no fallback word is in the lexicon.
- Reference accuracy: the fallback's exact-match rate and CER on the held-out split (lexicon
  disabled) are measured once at the end of the rules task and written to the runbook; a test
  pins a floor of (measured - 0.02).
- Validator: every code in the list has a failing and a passing example; the whole lexicon and
  the reference script column pass; `hypothesis` property: `validate_traditional` never raises.
- `cer`: identical strings 0.0; known edit counts; `hypothesis`: `edit_distance` is symmetric,
  bounded by the longer length, and zero only for equal strings.
- `segments`: concatenation reproduces the input (`hypothesis`); an NNBSP between letters does
  not split the word; splitting on U+0020 only.
- Corpus: counts exact on a planted list (duplicates, a violating pair, a count-mismatch
  line); train and valid disjoint; deterministic for a seed.
- Char model: tags single pieces; letters, FVS, MVS, NNBSP round-trip.
- Config: rendered YAML equals the checked-in files; every forbidden key and every violated
  invariant is rejected by `validate_converter_config`.
- Tiny training (marked `subprocess`, one module-scoped fixture shared by the next two
  gates): CER <= 0.15 on the training fixture in BOTH directions, measured with the
  greedy Eole predictions; wall time under 150 s on this machine.
- Export parity: CTranslate2 vs Eole, `agree / total >= 0.99` on the training fixture; the
  export subprocess runs with CUDA hidden and the test also runs it once with CUDA visible
  only when `RINCHEN_CT2_CUDA_PROBE=1` (documents the segfault; off by default).
- Runtime: `ScriptConverter.convert("...", "Mong")` equals the Eole output for fixture
  words; an output forced to contain ZWJ is replaced by the rule word and counted; an unknown
  target raises; text without Cyrillic words is returned unchanged.
- CLI: `--help` for each command, `build-data` then `train --dry-run` golden command
  rendering, error exit codes, manifest-last behaviour (a failed `export` leaves no manifest).
- STOP gate: the tiny model trained from the fixture is exported and loaded through
  `ScriptConverter`, and `apply --target Mong` converts a fixture line, all in the tests above.

The master plan also lists "CT2 export parity (D9 utilities)": D9 is document-level work; no
such utilities exist, so D5 builds the parity check it needs in `export.py`.

## Rulings

Each is the builder's call; "cost" is what a wrong ruling costs.

1. **The engine half of the STOP gate becomes `ScriptConverter`.** The master plan says "served
   via engine with `script="Mong"`"; there is no engine until D11/D12. Cost if wrong: the engine
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
   script. Cost: the reverse direction is learned only from rule/reference pairs, which makes
   it many-to-one in the data; its CER is reported, not assumed.
6. **600 to 800 steps for the tiny model, threshold unchanged (CER <= 0.15), both directions,
   Noam lr 0.5.** Measured above. Cost: a slower tiny test than the master plan expects.
7. **Eole configs are rendered by code and validated** (`config.py`), not hand-edited; the
   master plan's template is corrected for the three failures found (`tgt_prefix`,
   `word_vec_size`, `bucket_size`). Cost: D7's `eole_cfg.render` should absorb `config.py`.
8. **CTranslate2 export always runs with CUDA hidden.** Cost: none on CPU export; an export
   that wants the GPU does not exist.
9. **Own character SentencePiece trainer** instead of extending `spm.train.train_spm` (which is
   unigram, byte-fallback and per-language-directory by construction, with a test that pins
   its options). Cost: about 30 lines of trainer wiring duplicated.
10. **An invalid neural output is replaced by the rule baseline's word** and counted, rather
    than dropped or left in Cyrillic. Cost: some outputs are the phonographic guess instead of
    the model's.
11. **Reference provenance.** The Wiktionary harvest is used as a held-out reference and as
    the source of the lexicon and tiny fixture, never as training data for a real model. Its
    revision ids are in the TSV. Wiktionary entries carry no guarantee of Mongolia-only
    provenance; the harvest filters out entries labelled Inner Mongolia, Chakhar and China and
    takes only the `==Mongolian==` section, but a Mongolian reader has not checked it. Cost: a
    wrong lexicon entry is wrong training data for the tiny fixture only.
12. **Digits and punctuation.** Digits pass through unchanged; `.` and `,` map to U+1803 and
    U+1802 (the marks the D3 splitter already treats as `mn_Mong` boundaries). Cost: one table.
13. **Dev and held-out split** of the reference by `sha256(word) % 2`, the rule development
    sees only the dev half; the held-out figure is computed once. Cost: the held-out half is
    spent after that figure is published.

## Open gates (not done by D5, because no real data or reader exists)

- Audit of U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180A..U+180F and U+202F in real
  traditional-script text, before the joiner-stripping policy is frozen (assigned to D5 by the
  D3 follow-up; blocked on data).
- Render check of converter output in Noto Sans Mongolian 2.001 and v3.103 (no shaping
  library is installed; the previous check used a throwaway environment).
- Review of the lexicon, the golden words and the fallback by a Mongolian reader.
- `mn_Mong` length-ratio bounds for the D4 filter: the quantiles are measured on the
  reference pairs and written to the runbook, not wired into `configs/data/filters.yaml`.
- Training the full profile on real parallel script data.

## Limitations

- The rule baseline is a measured scaffold, not an orthography. Its held-out accuracy is in
  the runbook.
- The tiny model memorizes a fixture; its CER on unseen words says nothing about quality.
- Word-level conversion cannot use sentence context.
- The CT2 converter's CUDA segfault is an upstream defect worked around, not fixed.
