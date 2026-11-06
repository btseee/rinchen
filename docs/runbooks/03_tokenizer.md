# 03 — Tokenizer

## Scope

D3 is the text layer under every later milestone: `normalize(text, lang)`
(`src/rinchen/text/normalize.py`), `split(text, lang)`
(`src/rinchen/split/rules.py`), and the SentencePiece tooling
(`rinchen spm train` / `rinchen spm check`). This runbook covers retraining a
model, adding a language, the sampling temperature, and updating the golden
file. Why the rules are what they are lives in `DECISIONS.md`
(2026-11-06 entry).

Every Rinchen SentencePiece model reserves 118 single-piece symbols: one
`<2xx>` direction tag per registry language (10), `<bt> <ft> <t> </t> <r>
</r> <doc> <sep>`, and the placeholders 0..99 written with U+27E6 / U+27E7.
The model policy (`unigram`, byte fallback on, digits split, identity
normalization) is a set of constants in `src/rinchen/spm/train.py`, not
config. Corpus text is expected to be normalized and sentence-split already.

Only `tiny_1k` has ever been trained. `student_64k` and `teacher_128k` are
schema-validated YAML; the corpus they read (`data/clean/spm_corpus`) is
produced by later milestones and does not exist yet.

## Walkthrough

### Train and check a model

```bash
# Tiny profile: the committed synthetic corpus, a 1000-piece model.
uv run rinchen spm train --config configs/spm/tiny_1k.yaml
uv run rinchen spm check --model data/tiny/spm/tiny_1k.model \
    --corpus-dir tests/fixtures/spm/tiny_corpus
```

`spm train` prints one line, `trained <absolute path to the .model>`, and
writes three files next to each other: `<output_prefix>.model`,
`<output_prefix>.vocab`, and `<output_prefix>.sampled.txt` (the balanced
sample the model was trained on). It overwrites them on a rerun. Relative
`corpus_dir` and `output_prefix` values in a config are anchored at the repo
root; `--model` and `--corpus-dir` of `spm check` are taken as given, so run
it from the repo root or pass absolute paths. `--sample-lines N` (default
1000) sets how many leading lines per language the coverage is measured on.

`spm check` prints, in this order:

```
eng_Latn	1.0000
khk_Cyrl	1.0000
mn_Mong	1.0000
rus_Cyrl	1.0000
byte_fallback: on
ok
```

- `<lang>\t<coverage>`: one line per `<lang>.txt` file, in language-code
  order. Coverage is the share of encoded pieces that are neither `<unk>` nor
  a byte piece. It is informational and never fails the check.
- `byte_fallback: on|off`: `on` when all 256 pieces `<0x00>`..`<0xFF>` exist.
- `non-atomic symbol: <symbol>`: one line per reserved symbol that does not
  encode to itself (alone, or after the dummy-prefix piece).
- `ok` or `failed`: `failed` when byte fallback is off or any symbol is
  non-atomic.

The check verifies symbol atomicity and byte fallback only. It does not
verify digit splitting or normalization.

### Retrain each profile

| Profile | Config | vocab_size | input_sentence_size | Output prefix | Corpus dir |
|---|---|---|---|---|---|
| tiny | `configs/spm/tiny_1k.yaml` | 1000 | 100000 | `data/tiny/spm/tiny_1k` | `tests/fixtures/spm/tiny_corpus` |
| student | `configs/spm/student_64k.yaml` | 64000 | 10000000 | `models/spm/student_64k` | `data/clean/spm_corpus` |
| teacher | `configs/spm/teacher_128k.yaml` | 128000 | 20000000 | `models/spm/teacher_128k` | `data/clean/spm_corpus` |

```bash
uv run rinchen spm train --config configs/spm/student_64k.yaml
uv run rinchen spm check --model models/spm/student_64k.model \
    --corpus-dir data/clean/spm_corpus
# Same two commands with teacher_128k for the teacher model.
```

Every profile shares `character_coverage: 0.9999`, `temperature: 0.3` and
`seed: 1234`. The student and teacher commands cannot run until a later
milestone writes `data/clean/spm_corpus/<lang_code>.txt`; today they exit 2
with `error: corpus directory is missing or not a directory: ...` and write
nothing. The files are UTF-8, one normalized sentence per line, one file per
language, each named after a registry code. A `.txt` file named anything else
is a `ConfigError` (other extensions are ignored); blank lines are ignored; a
leading BOM is dropped; lines end at LF only. `data/tiny/` and `models/` are
gitignored.

### Exit codes of the spm commands

Failures print `error: <message>` to stderr.

| Code | Meaning |
|---|---|
| 0 | OK; for `check`, the last line is `ok` |
| 1 | Any other `RinchenError`, or an `OSError` while writing training outputs |
| 2 | `ConfigError`: config missing or invalid, corpus dir missing, a corpus file not named after a registry code; also a command-line usage error (for example `--sample-lines 0`) |
| 3 | `DataError`: corpus empty, unreadable or not UTF-8, model file cannot be loaded; also a `check` whose last line is `failed` |
| 4 | `UpstreamError`: SentencePiece rejected the run, with its message appended |

For example, a `vocab_size` above what the corpus supports ends with exit 4
and `error: sentencepiece training failed for <prefix>: ... Vocabulary size
too high (5000). Please set it to a value <= 3795.` (the tiny corpus).

### Add a language

1. Add an entry to `configs/languages.yaml`: `code`, a `scripts` list (values
   of the `Script` enum in `src/rinchen/lang/scripts.py`), and
   `direction_tag: "<2{code}>"` (the loader rejects any other tag). The
   registry order is the order of the tag symbols in the vocabulary.
2. Nothing to regenerate by hand: `spm.symbols.user_defined_symbols()` reads
   the registry, so the symbol set grows by one. The tests that pin the
   registry (`tests/unit/test_config_languages.py`: the code list and the
   8-partner, 16-direction count) and the symbol count
   (`tests/unit/test_spm_symbols.py` and `tests/unit/test_spm_train.py`, both
   118) must be updated in the same change. `normalize` needs no change; it
   accepts every registry code.
3. Register a sentence-splitting family in `_SPLITTERS` in
   `src/rinchen/split/rules.py`: point the code at an existing family or add
   one. `test_every_registry_language_has_a_split_family` fails until you do.
4. Add golden lines for the language (see the golden section below); the
   passthrough category must contain a line for every registry language.
5. Add `<code>.txt` to the corpus directories that should cover the language:
   `data/clean/spm_corpus` for student and teacher. The tiny corpus
   (`tests/fixtures/spm/tiny_corpus`, built by
   `scripts/build_spm_fixture.py`) does not need every language; it has four.
6. Retrain every profile and run `spm check` on each. Existing models become
   invalid, because the reserved symbol set changed, so retrain even if the
   new language is irrelevant to a profile.

### Sampling temperature

`train_spm` draws about `input_sentence_size` lines from the corpus files,
tilted toward small languages. With `n_i` lines in language `i` and
temperature `T`, language `i` gets weight `n_i**T / sum_j(n_j**T)`
(`rinchen.utils.sampling.temperature_weights`), and a quota of
`floor(weight * input_sentence_size)` lines, capped at `n_i`.

`T=1` is proportional sampling; `T` near 0 approaches equal shares per
language. A size ratio `r` between two languages becomes `r**T`, so at
`T=0.3` a 100:1 ratio becomes about 4:1: for two languages of 100,000,000 and
1,000,000 lines the weights are 0.7992 and 0.2008 instead of 0.9901 and
0.0099. That is why `T=0.3` boosts small languages: they would otherwise
supply almost none of the text the vocabulary is learned from.

Two consequences. When all files together hold no more than
`input_sentence_size` lines (the tiny corpus: 800 lines against 100000),
nothing is sampled and the temperature has no effect. When a small language
is capped at its own size, the unused quota is not given to the others: in
the 100,000,000 and 1,000,000 example with a 10,000,000-line budget, the
quotas are 7,992,399 and 1,000,000, so the sample holds 8,992,399 lines.
Each language is sampled from its own RNG stream seeded `f"{seed}-{lang}"`,
so adding a language does not change the other languages' samples.

### Update the golden file

`tests/fixtures/golden/normalize_200.tsv` is the zero-tolerance gate for
`normalize`: 200 data lines `input<TAB>expected<TAB>lang` in 8 categories
(`# category: <name>` comment lines mark them) with fixed quotas: nfc 15,
zero_width 20, whitespace 30, quotes 25, homoglyph 40, fold 20,
mn_mong_preservation 30, passthrough 20. The file is pure ASCII: every
non-ASCII or control character is a `\uXXXX` or `\UXXXXXXXX` escape, and a
literal backslash is `\\`. There is no `just golden-update` recipe in D3:
expected values are never generated from `normalize` output.

1. Edit the file by hand, inside the category block the line belongs to.
2. Derive each `expected` value from the rule table (the pipeline order in
   the `text/normalize.py` docstring), never by pasting `normalize` output.
   When a line fails, decide whether the derivation or the implementation is
   wrong before changing either.
3. Keep exactly 200 data lines and every category quota, and keep at least
   one passthrough line per registry language. A rule change updates every
   affected line in the same commit.
4. Check the file is still pure ASCII; this must print nothing:

   ```bash
   grep -nP '[^\x00-\x7F]' tests/fixtures/golden/normalize_200.tsv
   ```

5. Run `uv run pytest tests/golden -q`, then review `git diff` line by line.

## Operational notes

- **Editor tools rewrite escapes.** The Write and Edit tools coding agents
  use convert backslash-u escapes in a file into the raw characters they
  stand for. In the golden file that breaks the ASCII test; in a test file
  it hides what a string contains (an invisible character, a Latin letter
  that looks Cyrillic). After editing any file that holds such text, run
  `grep -nP '[^\x00-\x7F]'` on it. The golden file, `test_text_normalize.py`,
  `test_normalize_properties.py` and `test_normalize_golden.py` are pure
  ASCII today, so any hit there is a conversion. Quote code points in prose
  as `U+XXXX`.
- **`split` expects normalized text.** Run `normalize` first. A
  zero-width character left after a terminator misplaces the boundary (in
  mn_Mong the next part starts with it; in the Latin and Cyrillic families
  it suppresses the split). The splitter's other known gaps (spaced
  abbreviations in brackets, sentences ending in `U.S.` followed by a capital
  word, the pronoun "I", ASCII `?` and `!` splitting inside mn_Mong URL-like
  text) are listed in `DECISIONS.md`.
- **Memory.** `train_spm` reads every `<lang>.txt` in full
  (`read_lines`) before sampling, so memory follows the total corpus size,
  not the budget. A Python `str` costs roughly 94 bytes for a short line
  (real sentences cost more), and the configured budgets are 10 million
  (student) and 20 million (teacher) lines. Size the machine for the corpus,
  not for the sample.
- **`spm check` reads whole files too.** It loads each corpus file to take
  the first N lines, so `--sample-lines` bounds the encoding work, not the
  reading. The first N lines are not a random sample.
- **Dummy prefix.** The default `add_dummy_prefix` is kept: a bare tag
  encodes as a standalone U+2581 piece followed by the tag, not as the tag
  alone. `spm check` accepts both shapes. Whether to turn it off is to be
  decided before D8 (Eole training).
- **No manifest.** `spm train` writes no manifest and prints only the
  `trained ...` line.
- **SentencePiece failures** surface as exit 4 with the SentencePiece
  message appended. SentencePiece enforces a hard vocabulary limit, so a
  corpus that is too small or too repetitive for the requested `vocab_size`
  fails fast (the tiny corpus reported `<= 3795` when asked for 5000); the
  committed tiny corpus is made of varied pseudo-words for that reason.
