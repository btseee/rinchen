# 04 - Filtering and mining

## Scope

D4 turns parallel text into scored and filtered corpora and mines new parallel
text from unaligned documents. It builds the corpus Parquet contract
(`src/rinchen/corpus/`), seven filter stages with an ordered pipeline
(`src/rinchen/filter/`), the B7 GlotLID wrapper (`filter/lid.py`), the B6
sentence aligner and a document matcher (`src/rinchen/mine/`), and two
commands: `rinchen filter run` and `rinchen mine align`. The design, with the
rulings that fill the gaps of the master plan and what each costs if wrong, is
`docs/superpowers/specs/2026-11-06-d4-filtering-mining-design.md`; the
decisions are condensed in the 2026-11-07 entry of `DECISIONS.md`.

**Every threshold in this runbook is a prior that has never been tested on real
data.** No real corpus exists in the repository yet (D1 fetched fixtures only),
so no number below was tuned or validated on text a person cares about. The
tables mark each one "prior, unvalidated". Treat the defaults as a starting
point for a first look at a real corpus, not as a quality bar.

What was checked, and how, is in the sections "Estimates", "GlotLID label
findings" and "Smoke run". In short: the unit and property tests (planted
failures with exact reason counts, MinHash recall and precision, aligner F1)
use fakes and synthetic data; four `network`-marked tests use the real models
on the development machine; and one 24-row integration run drove the real CLI
with the real GlotLID, LaBSE and MetricX-QE models.

## Order of operations

1. **Prepare text.** Normalize with `normalize(text, lang)` and split into
   sentences with `split(text, lang)` (D3, `docs/runbooks/03_tokenizer.md`)
   before anything is written to a corpus table. The filters do not normalize.
2. **Get a corpus table.** Sentence-aligned pairs go into a corpus Parquet file
   (`rinchen.corpus.schema.corpus_frame` builds the frame from row mappings,
   `rinchen.corpus.writers.write_corpus` writes it; both check the schema).
   D4 ships no converter from `data/raw/` to this table. Documents that are not
   yet aligned go through `rinchen mine align`, which writes the same table
   with `origin = mined`.
3. **Filter.** `rinchen filter run` scores every row and keeps those that no
   stage rejected. The stage order is the precedence of the rejection reason:
   `script_ratio`, `numbers`, `length`, `lid`, `minhash`, `labse`, `qe`. The
   cheap CPU checks come first and the GPU checks last, and a row keeps the
   reason of the first stage it fails. Every stage still scores every row (a
   rejected row gets its later scores too), except `minhash`, which looks only
   at rows that have no reason yet, so a rejected row never makes a later copy
   look like a duplicate.
4. **Use `kept.parquet`.** Later milestones read it; `scored.parquet` is for
   analysis and for choosing thresholds.

Models, in the order the pipeline loads them: GlotLID (CPU, in `lid`), LaBSE
(GPU when available, in `labse`), MetricX-QE (GPU only, in `qe`). When
`labse` is listed before `qe` (the default order), LaBSE is released as soon as
`labse` has run, so MetricX-QE does not share the GPU with it; with `qe` listed
first the two are resident together.

## Models

| Model | Used by | How it arrives | Size | Needs |
|---|---|---|---|---|
| GlotLID v3 (`cis-lmu/glotlid`, file `model_v3.bin`) | `lid` stage, `GlotLID` | you download it into `models/lid/` | 1,687,094,687 bytes (1.6 GiB) | network once; CPU |
| LaBSE (`sentence-transformers/LaBSE`) | `labse` stage, `mine align` | downloaded on first use into the Hugging Face cache (`~/.cache/huggingface/hub`) | 1.8 GiB of weights | network once; GPU recommended |
| MetricX-24 hybrid QE (`eole-nlp/metricx-24-hybrid-large-v2p6-eole`) | `qe` stage | downloaded on first use by Eole's resolver into `~/.cache/eole/hf/` | 5.6 GiB on disk (about 6 GB) | network once; CUDA GPU, about 5 GB of GPU memory |

**Nothing in D4 downloads into the repository.** GlotLID is the only model you
place yourself:

```bash
uv run hf download cis-lmu/glotlid model_v3.bin --local-dir models/lid
mv models/lid/model_v3.bin models/lid/glotlid_v3.bin
```

The installed `huggingface_hub` (2.3.0) provides the `hf` command; its older
name, `huggingface-cli`, is not installed here. The download lands at `models/lid/model_v3.bin` (verified in a
scratch directory) next to a `.cache/` folder of download metadata. The
default config reads `models/lid/glotlid_v3.bin`, hence the `mv`; instead of
renaming you can set `lid.model_path: models/lid/model_v3.bin` in your config.
`models/` is gitignored. A relative `lid.model_path` is anchored at the
repository root; the tests and the smoke run below use the file in the Hugging
Face cache instead.

LaBSE and MetricX-QE download when the first run needs them, so the first
`filter run` or `mine align` on a machine takes network time and disk. The Eole
resolver swallows connectivity errors (seen while building the adapter), so a
machine without network and without the cached model gets the misleading
`error: '<model>' is neither a directory nor a pre-converted Eole repo on
Hugging Face` and exit 4 rather than a network error. Eole's `EOLE_MODEL_DIR` is not honoured by
the adapter. Running offline after the models are cached was not tested.

## Run the filter

```bash
# Look first: validates the config, the model files and the corpus schema,
# prints the stage names (and why `qe` is skipped, when that is known without
# loading a model), writes nothing.
uv run rinchen filter run --corpus data/clean/mycorpus.parquet --dry-run

# Run. Default config: configs/data/filters.yaml.
uv run rinchen filter run --corpus data/clean/mycorpus.parquet

# Own config and output directory.
uv run rinchen filter run --corpus data/clean/mycorpus.parquet \
    --config configs/data/my_filters.yaml --out-dir data/clean/mycorpus_filtered
```

Options: `--corpus` (required), `--config` (default
`configs/data/filters.yaml`), `--out-dir` (default `data/clean/<corpus file
stem>/`), `--force`, `--dry-run`, `--seed N`. A config file may leave out any
section or field; a missing one keeps its default, and an unknown field is an
error.

**Output.** Into the out dir: `scored.parquet` (every input row, with `scores`
and `filter_reason`), `kept.parquet` (the rows with no reason; its
`filter_reason` column is kept and is all null) and `manifest.json`. The
command prints two lines, `kept K of N rows` and `manifest: <path>`. Both
tables follow the corpus schema (`src/rinchen/corpus/schema.py`).

**Inspect.** Count rows per reason:

```bash
uv run python -c "import polars as pl; print(pl.read_parquet('data/clean/mycorpus/scored.parquet').group_by('filter_reason').len())"
```

**Manifest** (`rinchen.manifest/1`): `stage` is `filter`; `config_sha256` is
the SHA-256 of the raw bytes of the config file; `code_version` is the first 7
characters of the HEAD commit, with `-dirty` for uncommitted changes (`unknown`
with a logged warning when git is unavailable); `seed` is `--seed`; `inputs`
holds the corpus and each model file the config reads (the LID model and, when
set, the SentencePiece model) with path, SHA-256 and size; `outputs` holds
both tables with their row counts; `metrics` holds `rows_in`, `rows_kept` and
`reason.<name>` for each of the ten reasons (0 when unused; kept rows are not
counted); `tool_versions` holds polars, datasketch, fasttext-new (the native
binding), fasttext-wheel (the vendored shim that provides `import fasttext` on
top of it), sentence-transformers, sentencepiece, torch and eole; and `status`
maps every configured stage to `ok`, `partial: <what was skipped>` or
`skipped: <reason>`.
The statuses a run can show today:

| Status | Meaning |
|---|---|
| `length: partial: token ratio skipped (no spm_model)` | `length.spm_model` is not set, so only the character ratio ran |
| `qe: skipped: no QE scorer configured` | `qe.scorer: cometkiwi` (no adapter exists, see Limitations) |
| `qe: skipped: no CUDA device` | the MetricX-QE scorer needs a CUDA device and torch sees none |

A skipped stage is not applied, writes no scores and rejects nothing, and the
run still succeeds. The no-GPU path was exercised only with a fake scorer in
the unit tests, not on a machine without a GPU. `--dry-run` shows `qe: skipped`
only when it can know that without importing torch (the cometkiwi case); for
`metricx_qe` it prints a plain `qe` and the real run decides.

**Skip semantics.** A run is skipped, printing `skipped: inputs and config are
unchanged since the last run` and the manifest path, with exit 0, when all of
these hold: no `--force`; `scored.parquet` and `kept.parquet` both exist; a
manifest exists whose `stage` is `filter`, whose `outputs` list both files and
whose `config_sha256` equals the config file's hash; and every input in
`inputs` (the corpus and each model file the config names) has the recorded
hash. Notes: the model files named by the config must exist even for a skipped
run (exit 2 otherwise); the skip check hashes them, which costs about 0.5 s
with the 1.7 GB GlotLID file (0.44 to 0.48 s, measured with the file in the
page cache; a cold read is slower and was not measured); the code version, the LaBSE and MetricX
weights and the package versions are not part of the check, so use `--force`
after changing code or after the cached models change; the manifest is written
last and removed before any output is touched, so a failed run leaves no
manifest and the next run starts over. `--seed` is recorded and nothing else:
the pipeline is deterministic and MinHash has its own `minhash.seed`.

**Output directory.** The run replaces `scored.parquet` and `kept.parquet` in
the out dir, so `--corpus` must not be either of them, also through a symlink:
the command exits 2 before it reads anything, saying to pass a different
`--out-dir`. An out dir whose `manifest.json` belongs to another command (a
`mine_align` run, say) exits 3 and leaves that manifest alone, with or without
`--force`: a directory holds the outputs of one command. Each output and the
manifest are written to a uniquely named temporary file beside the target
(`<name>.<random>.part`) and renamed into place; a killed run can leave such a
file behind, and it is safe to delete.

**Exit codes.** Failures print `error: <message>` on stderr.

| Code | Meaning |
|---|---|
| 0 | OK, including a skipped run and `--dry-run` |
| 1 | any other `RinchenError`, or an `OSError` while writing outputs |
| 2 | `ConfigError`: config missing or invalid (the message names the field), a model file the config names is not a file or not a model, `--corpus` is one of the files the run writes; also a command-line usage error such as a missing `--corpus` |
| 3 | `DataError`: corpus missing, unreadable or not a corpus table (the message lists the schema problems); a language code in the rows that is not in `configs/languages.yaml`; an unreadable manifest; an out dir that holds another command's manifest |
| 4 | `UpstreamError`: a model could not be loaded or run (LaBSE, MetricX-QE), with the reason appended |

A corrupt model file exits 2 where its loader raises: 12 corrupt GlotLID files
(garbage bytes, empty, truncated at 4 KiB, a tenth and a half, 4 bytes, six with
header bytes changed) all did. The fastText and SentencePiece loaders are
native code, though, and a file that crashes one of them ends the interpreter
instead of exiting 2: SentencePiece does so for a directory, which the command
rules out beforehand, and a corrupt fastText file of a kind not tried may do
the same.

Errors that are not `RinchenError`s are not mapped and end as a Python
traceback: the main case is a Parquet file whose metadata is fine but whose
body is corrupt, which polars reports when a stage collects. Out-of-memory is
also a traceback (see Limitations).

## Run the aligner

`rinchen mine align` aligns two files of one sentence per line, taken as one
document pair, with the B6 aligner:

```bash
uv run rinchen mine align --src en.txt --tgt mn.txt \
    --src-lang eng_Latn --tgt-lang khk_Cyrl --corpus mydocs
```

Options: `--src`, `--tgt`, `--src-lang`, `--tgt-lang`, `--corpus` (all
required; the languages must be registry codes and the corpus name a plain
name without path separators), `--out` (default
`data/mined/<corpus>/aligned.parquet`), `--max-k` (default 4), `--skip-cost`
(0.9), `--merge-penalty` (0.3), `--force`. Files are read as UTF-8 (a byte
order mark is ignored); lines are stripped and blank lines are dropped (and
counted).

**Output.** `aligned.parquet` in the corpus schema, with `origin = mined`,
`split = train`, `doc_id` = the stem of the `--src` file, `seg_idx` numbering
the emitted rows from 0, and the sentences of a group joined by one space; a
sentence aligned to nothing (a deletion) emits no row. The command prints
`aligned N rows (unaligned: A source, B target)` and `manifest: <path>`.
`manifest.json` is written beside the output (one manifest per directory, so
use one directory per `--out` or the default path): `stage` is `mine_align`;
`config_sha256` is the SHA-256 of the parameters as canonical JSON, because
the run has no config file; `inputs` are the two sentence files, hashed before
they are read; `metrics` hold `aligned`, `unaligned_src`, `unaligned_tgt`,
`src_sentences`, `tgt_sentences`, `blank_lines_dropped`, `param.max_k`,
`param.skip_cost` and `param.merge_penalty`; there is no `status`. A rerun is
skipped, like `filter run`, when the output exists, the manifest in its
directory is a `mine_align` manifest that lists it, and inputs and parameters
are unchanged. `--out` must not be `--src` or `--tgt`, and the `manifest.json`
beside it must not be one of them either (the run would destroy its own
input); a directory whose manifest belongs to another command (a `filter run`)
is refused and its manifest kept, with or without `--force`.

Exit codes as above: 2 for a bad parameter (`--max-k` below 2, a negative or
non-finite cost), an unknown language code, a bad corpus name or an `--out`
that is an input, 3 for a missing, unreadable or non-UTF-8 sentence file or a
directory that holds another command's manifest, 4 for an embedding model that
cannot be loaded (also for an `align` failure on bad embedder output).

The aligner is vecalign-style: every span of 1 to `max_k - 1` consecutive
sentences is embedded as one text, the cost of a pairing is margin-normalised
cosine distance, and dynamic programming picks the cheapest path over 1-1, 1-2,
2-1, 2-2, 1-3 and 3-1 groups plus single-sentence deletions; the module
docstring of `src/rinchen/mine/align.py` has the formula. `match_documents` in
`src/rinchen/mine/docalign.py` pairs documents by language-stripped URL and
then by mutual nearest embedding neighbours; it is a library function with no
command yet.

## Thresholds

Defaults come from `configs/data/filters.yaml`, which must load to exactly the
`FilterConfig()` defaults in `src/rinchen/config/filters.py` (a test pins
this). Every value in the Default column is a prior, unvalidated.

| Setting | Default | Meaning | Status |
|---|---|---|---|
| `stages` | script_ratio, numbers, length, lid, minhash, labse, qe | Stages to run, in order; the order is the precedence of `filter_reason`; no repeats | prior, unvalidated |
| `script_ratio.min_ratio` | 0.9 | Least share of a side's letters that must belong to the scripts of its language (`configs/languages.yaml`). 10% of other letters (brand names in Latin inside Cyrillic) is tolerated. Letters of Common or Inherited script are neutral. Text without letters scores 1.0 | prior, unvalidated |
| `length.char_ratio` | min 0.5, max 2.0 | Accepted range of effective length of `tgt` over `src`, inclusive. Effective length is the sum of per-character weights over non-whitespace characters. A side with effective length 0 gives a null ratio and is rejected as `len_ratio_char` | prior, unvalidated |
| `length.token_ratio` | min 0.4, max 2.5 | Accepted range of the SentencePiece token count of `tgt` over `src` (reason `len_ratio_token`) | prior, unvalidated |
| `length.char_weights` | Han 2.5, Hangul 2.0, Hiragana 1.5, Katakana 1.5 | Weight of a character of that script; every other character weighs 1.0. Listing it replaces the table, it is not merged | prior, unvalidated |
| `length.by_direction` | empty | Per-direction replacement bounds, see below | prior, unvalidated |
| `length.spm_model` | unset | SentencePiece model for the token check; the check runs only when it is set (status `partial` otherwise) | prior, unvalidated |
| `lid.model_path` | `models/lid/glotlid_v3.bin` | GlotLID file; relative paths are anchored at the repository root | fixed path convention |
| `lid.min_prob` | 0.5 | Least summed probability of the accepted labels among the top `top_k` predictions (reasons `lid_src`, `lid_tgt`) | prior, unvalidated |
| `lid.top_k` | 3 | Predictions summed | prior, unvalidated |
| `lid.accept` | `zho_Hans: [cmn_Hani, wuu_Hani]` | Labels counted as a language; a code not listed accepts only its own label. Listing it replaces the table | prior, unvalidated; evidence in "GlotLID label findings" |
| `minhash.num_perm` | 128 | MinHash permutations | prior, unvalidated |
| `minhash.ngram` | 5 | Character shingle length (a text shorter than that is one shingle) | prior, unvalidated |
| `minhash.threshold` | 0.8 | Estimated Jaccard of `src + TAB + tgt` against an earlier kept row at or above which a row is `duplicate` | prior, unvalidated |
| `minhash.seed` | 1 | Seed of the hash permutations | fixed convention |
| `labse.model` | `sentence-transformers/LaBSE` | Embedding model | fixed choice |
| `labse.min_cosine` | 0.7 | Least cosine of the two sentence embeddings (reason `labse`; NaN is rejected) | prior, unvalidated |
| `labse.batch_size` | 128 | Sentences per GPU batch | prior, unvalidated |
| `labse.device` | `auto` | `auto` is CUDA when torch sees one, else CPU; also `cpu`, `cuda` | convention |
| `qe.scorer` | `metricx_qe` | `metricx_qe` has an adapter; `cometkiwi` has none (stage skipped) | fixed choice |
| `qe.max_metricx_qe` | 8.0 | Highest accepted MetricX-QE score (0 to 25, lower is better; a score equal to the threshold passes) | prior, unvalidated |
| `qe.min_cometkiwi` | 0.55 | Lowest accepted CometKiwi score (higher is better); unused without an adapter | prior, unvalidated |
| `qe.batch_size` | 32 | Pairs per scorer call | prior, unvalidated |

Aligner and document matcher defaults:

| Setting | Default | Meaning | Status |
|---|---|---|---|
| `align` `max_k` / `--max-k` | 4 | A group of `a` source and `b` target sentences is allowed when `a + b <= max_k` | prior, unvalidated |
| `align` `skip_cost` / `--skip-cost` | 0.9 | Cost of leaving one sentence unaligned | prior, unvalidated |
| `align` `merge_penalty` / `--merge-penalty` | 0.3 | Extra cost of `merge_penalty * (a + b - 2)` per group; 0.0 is the plain vecalign weighting | prior, tuned on synthetic data only |
| `align` `margin_k` (library only) | 4 | Neighbours averaged in the margin term | prior, unvalidated |
| `match_documents` `max_date_gap_days` | 2 | Largest date difference of a candidate pair when both dates are known | prior, unvalidated |
| `match_documents` `min_cosine` | 0.7 | Least cosine of document embeddings for a match | prior, unvalidated |

Fixed in code, not configurable: a text whose non-whitespace characters are at
least half traditional Mongolian script (U+1800..U+18AF) is labelled `mn_Mong`
with probability 1.0 before the model runs; GlotLID is asked for rows in slices
of 100,000; a blank text scores 0.0 in `lid`.

### Per-direction bounds

The ratio is `tgt / src`, so the reverse direction has the inverse ratio. The
stage-wide bounds are symmetric under inversion (1 / 2.0 = 0.5 and 1 / 2.5 =
0.4), so one set serves both directions; an override must be inverted by hand
for the other direction (`min' = 1 / max`, `max' = 1 / min`). The registry has
ten codes: `khk_Cyrl`, `mn_Mong` and eight partners. The table lists the nine
codes that pair with `khk_Cyrl`; `mn_Mong` is the traditional-script variant of
Mongolian, not a partner. Mongolian Cyrillic letters weigh 1.0. **No
direction has an override today, so every row uses the stage-wide priors
(characters 0.5 to 2.0, tokens 0.4 to 2.5); the Reasoning column is why a
direction might need one, not a measured expectation.**

| `khk_Cyrl` -> | Target script | Weight of target characters | Bounds in effect | Reasoning (prior, unvalidated) |
|---|---|---|---|---|
| `eng_Latn` | Latin | 1.0 | char 0.5 to 2.0, token 0.4 to 2.5 | Both sides count letters one for one, so the ratio reflects word length and morphology only (Mongolian is agglutinative). Needs no weight |
| `rus_Cyrl` | Cyrillic | 1.0 | as above | Same script and weight on both sides, so no script weighting is involved |
| `zho_Hans` | Han | 2.5 | as above | One Han character carries several letters of meaning, so each counts 2.5. Illustration with made-up numbers: 20 Han characters against 60 Mongolian letters give 0.33 unweighted and 0.83 weighted. The factor is a guess |
| `kor_Hang` | Hangul | 2.0 | as above | A Hangul syllable block packs two or three letters; 2.0 approximates that |
| `jpn_Jpan` | Han, Hiragana, Katakana | 2.5 / 1.5 / 1.5 | as above | The average weight depends on the kanji-to-kana mix of the sentence, so the ratio varies more than for Chinese or Korean; the first candidate for a wider override |
| `deu_Latn` | Latin | 1.0 | as above | As English; German compounds lengthen words, and whether sentences get longer in letters is unmeasured |
| `kaz_Cyrl` | Cyrillic | 1.0 | as above | As Russian; Kazakh is agglutinative like Mongolian |
| `tur_Latn` | Latin | 1.0 | as above | As English; Turkish is agglutinative like Mongolian |
| `mn_Mong` | Mongolian script | 1.0 | as above | Not a translation: the pair is the same text in two scripts (D5's converter produces these pairs). The letter-count ratio is measured for single words only (next paragraph); look at it on sentences before trusting any bound |

**Measured `mn_Mong` ratios (D5, not wired in).** D5 measured the length of a
traditional-script word over the length of its Cyrillic word on its 2,027-pair
dictionary reference: quantiles 0.01, 0.05, 0.5, 0.95 and 0.99 are 0.83, 1.00,
1.17, 1.50 and 1.95 (`docs/runbooks/05_script_converter.md`, "Length ratio of
script over Cyrillic", which also says how NNBSP changes the 0.99 quantile).
They are word-level figures from dictionary spellings, no sentence-level
measurement exists, and **they are not wired into `configs/data/filters.yaml`**:
the `mn_Mong` row still uses the stage-wide priors.

To override a direction, add its key to `length.by_direction`; a side left
unset keeps the stage-wide bounds, at least one of `char_ratio` and
`token_ratio` must be set, and a key must look like `khk_Cyrl->eng_Latn`
(otherwise `ConfigError`). The key is matched against each row's own `src_lang`
and `tgt_lang`, so a corpus with rows in both directions needs both keys. A
syntax example that loads (not a recommendation):

```yaml
length:
  by_direction:
    khk_Cyrl->eng_Latn:
      char_ratio: {min: 0.6, max: 1.8}
    eng_Latn->khk_Cyrl:
      char_ratio: {min: 0.55, max: 1.67}   # the inverse of the first
```

## Estimates

All figures were measured on one machine: Intel Core Ultra 7 265K (20 cores),
46 GiB of RAM, NVIDIA GeForce RTX 5070 Ti (16 GiB, driver 595.99.02), Linux
7.0, CPython 3.14.10, torch 2.14.1+cu130, 2026-11-07. The sentences are the
200-line GlotLID fixture (`tests/fixtures/golden/glotlid_200_lines.tsv`), which
holds 20 short sentences per registry language: "short" below is those lines
as they are (mean 38 characters), "long" is three consecutive lines of the same
language joined by a space (mean 116 to 156 characters). They are not
translations of each other, which does not matter for speed. The measuring
scripts were scratch files outside the repository and are not committed; each
ran as `uv run --project <repo> python -I <script>.py` from an empty scratch
directory.

| Component | Measured | Basis |
|---|---|---|
| GlotLID, CPU, all cores | 77,000 lines/s (short, 3 runs of 10,000 lines: 75,000 to 78,500); 55,000 lines/s (long: 52,000 to 58,000) | `GlotLID.predict(lines, k=3)`; the process used 11 to 15 cores' worth of CPU time (CPU time 11 to 15 times wall time) |
| GlotLID, CPU, one core | 6,500 lines/s (short); 5,900 lines/s (long) | the same under `taskset -c 0`; model load 1.5 s |
| LaBSE, GPU, batch 128 | about 5,500 sentences/s (short: 5,400 to 5,600 in three runs; an earlier measurement in Task 6 gave 6,100); 2,400 sentences/s (long, mean 117 characters) | `SentenceTransformerEmbedder.embed` on 2,000 sentences; model load 4.5 s; GPU memory peak 2,091 MiB allocated, 2,638 MiB reserved |
| MetricX-QE, GPU, batch 32 | 500 pairs/s (short: mean 41 and 51 characters, three runs 499 to 503); 226 pairs/s (long: 127 and 156 characters) | `EoleMetricXQEScorer.score` in chunks of 32 over 500 pairs, fp32; model load and first call 2.1 s from the local cache; GPU memory peak 4,935 MiB allocated (5,044 MiB reserved, 5,322 MiB as `nvidia-smi` shows it for the process) on short pairs and 5,041 MiB allocated (5,286 reserved, 5,566 as `nvidia-smi` shows it) on long pairs |
| `script_ratio` | about 80 s per million rows | reviewer measurement in the Task 4 review (the per-letter script lookup is uncached); not repeated |
| `minhash` | about 3.6 s per 20,000 rows of 200 characters | reviewer measurement in the Task 5 review, single thread; not repeated |

Derived estimates for a corpus of N rows (LaBSE embeds two sentences per row,
QE scores one pair, GlotLID reads two lines). They are arithmetic on the rates
above, assume sentences like the fixture's, and are lower bounds for a real
run, which also reads, writes and hashes the data:

| Stage | N = 1 million rows | N = 10 million rows |
|---|---|---|
| `lid`, all cores | 26 to 36 s | 4 to 6 min |
| `lid`, one core | 5 to 6 min | 51 to 57 min |
| `labse`, GPU | 6 to 14 min | 1.0 to 2.3 h |
| `qe`, GPU | 33 to 74 min | 5.6 to 12.3 h |
| `script_ratio` | about 80 s | about 13 min |
| `minhash` | about 3 min | about 30 min |

The GPU time of the full pipeline is therefore dominated by `qe`. Because every
stage scores every row, a row rejected early still costs GPU time in `labse`
and `qe`; running only the first four stages first (a config with
`stages: [script_ratio, numbers, length, lid]`) and feeding its `kept.parquet`
to a second run with the GPU stages is a way to spend less, at the price of no
scores on the rejected rows (not tried). Give the second run its own out dir,
for example `--corpus data/clean/mycorpus/kept.parquet --out-dir
data/clean/mycorpus_gpu`. Never pass the first pass's directory: the run would
overwrite its own input, and the command refuses it (exit 2). Without
`--out-dir` the default is named after the corpus file stem, so every second
pass lands in `data/clean/kept/`.

**GPU memory in a full run.** MetricX-QE needs about 5 GB alone (above). The
pipeline releases the LaBSE embedder as soon as the `labse` stage has run
(`SentenceTransformerEmbedder.release` drops the model, calls `gc.collect()` and
`torch.cuda.empty_cache()`), so during `qe` the GPU holds MetricX-QE only. The
QE model is released when the command ends, also after a failure. Measured on
2026-11-07 in the real `filter run` command over the 24-row smoke corpus below,
with the real GlotLID, LaBSE and MetricX-QE, after the first QE call: torch
reported 4,866 MiB allocated (5,034 MiB reserved) and `nvidia-smi` showed the
device at 6,387 MiB against 1,068 MiB idle, 5,319 MiB above it; the peak
allocated over the whole run was 4,941 MiB, and 32 MiB were still allocated
after the command's QE release (`nvidia-smi` then showed 1,385 MiB). The same
run with LaBSE's release switched off (a scratch script, which is what the code
did before this fix) held 6,665 MiB allocated (6,862 MiB reserved) after the
first QE call, `nvidia-smi` showed 8,219 MiB against 1,073 MiB idle (7,146 MiB
above it) and the peak allocated was 6,739 MiB. Budget about 5.5 GiB of GPU
memory for the default pipeline. Not tried on an 8 GiB card. An embedder given
to `run_pipeline` that has no `release` is left alone.

**Not measured:** LaBSE on CPU; any GPU other than this one; MetricX-QE in
bf16 or fp16 (the adapter defaults to fp32); real sentence-length
distributions (the figures above are for 38-character and 116-character
sentences, and QE time grows with length); the end-to-end time or peak memory
of `filter run` on anything larger than 24 rows; the `script_ratio` and
`minhash` rates on this machine; the memory figures in "Known limitations" for
corpora of millions of rows.

## GlotLID label findings

Found by running the real model (`model_v3.bin`) over 20 lines per registry
language (Task 7), and confirmed by the parity test:

- **Chinese is not labelled `zho_Hans`.** GlotLID v3 labels Mandarin
  `cmn_Hani` (19 of 20 fixture lines) and, for a short sentence, `wuu_Hani`
  (Wu, 1 of 20), never the registry code. With the "accept only its own label"
  rule every Chinese row would fail `lid`. The shipped default is therefore
  `lid.accept: {zho_Hans: [cmn_Hani, wuu_Hani]}`. Cost: text that GlotLID calls
  Wu is accepted as Chinese, and `Hani` labels do not separate Simplified from
  Traditional characters, so **Simplified versus Traditional is not checked in
  D4** (`script_ratio` cannot do it either, both are Han). A table you supply
  replaces the default, so list `zho_Hans` again if you still need it.
- **GlotLID has no `mn_Mong`.** Traditional Mongolian script gets the label
  `und_Mong`. The wrapper's pre-detector (half or more of the non-whitespace
  characters in U+1800..U+18AF) supplies `mn_Mong` with probability 1.0 before
  the model runs; the model is never asked about such lines.
- **Every other registry code is GlotLID's own label** (`khk_Cyrl`, `eng_Latn`,
  `rus_Cyrl`, `kor_Hang`, `jpn_Jpan`, `deu_Latn`, `kaz_Cyrl`, `tur_Latn`).
  With the default config, none of the 200 fixture lines is flagged (network
  test `test_the_default_config_flags_none_of_the_fixture_lines`).
- **Parity:** the wrapper's top-1 label equals the raw fastText label for 200
  of 200 fixture lines (probabilities within 1e-5). The golden file's 20
  `mn_Mong` entries are synthetic (short traditional-script phrases no
  Mongolian reader has validated), and the fixture's Mongolian and Kazakh
  sentences were not reviewed by native speakers.
- **Buryat is the near neighbour of Halh.** In the smoke run, one correct Halh
  sentence scored `khk_Cyrl` 0.623 with `bxr_Cyrl` (Buryat) 0.377, only 0.12
  above the 0.5 floor (see the smoke run). Short Halh sentences can lose a good
  share of their probability to Buryat; the other 15 correct sentences scored
  0.95 to 1.0. One sentence is an anecdote, not a rate, and the sentence was
  written by the assistant.

## Smoke run

This is an **integration check, not a validation**. It shows that the whole
`filter run` command works with the real GlotLID, LaBSE and MetricX-QE models
on a small input; it says nothing about how good the thresholds are. **The
Mongolian sentences were written by the assistant and were not reviewed by a
Mongolian reader**, so they are not evidence of quality, and 24 rows prove no
rate. No threshold was changed to improve the outcome.

The corpus has 24 rows `khk_Cyrl` -> `eng_Latn`: 16 natural pairs (an English
sentence of 8 to 14 words and its Mongolian translation) and 8 deliberately
bad rows. It was built with `corpus_frame` and `write_corpus` in a scratch
directory; the config was a copy of `configs/data/filters.yaml` with
`lid.model_path` pointing at the GlotLID file in the Hugging Face cache and
`length.spm_model` unset. The rows (index: Mongolian, English):

```
 0: Нэгдүгээр сард Улаанбаатарт цаг агаар маш хүйтэн байдаг. | The weather in Ulaanbaatar is very cold in January.
 1: Ээж маань ням гараг бүр бүх гэр бүлдээ бууз жигнэдэг. | My mother cooks dumplings for the whole family every Sunday.
 2: Оюутнууд номын санд шинэ ном уншиж байна. | The students are reading new books in the library.
 3: Бид ирэх зун хөдөө өвөө эмээгийндээ очно. | We will visit our grandparents in the countryside next summer.
 4: Дархан руу явах галт тэрэг найман цагт төв буудлаас хөдөлнө. | The train to Darkhan leaves from the central station at eight.
 5: Тэр гурван жилийн турш морин хуур тоглож сурсаар байна. | He has been learning to play the horse-head fiddle for three years.
 6: Гадаа салхи ихэсч байгаа тул цонхоо хаана уу. | Please close the window because it is getting windy outside.
 7: Манай компани хаврын улиралд шинэ оффис нээхээр төлөвлөж байна. | Our company plans to open a new office in the spring.
 8: Эмч түүнд илүү их ус уух, сайн унтахыг зөвлөсөн. | The doctor advised her to drink more water and sleep well.
 9: Эрт цагт энэ голын эрэг дээр жижигхэн тосгон байсан. | A long time ago, a small village stood beside this river.
10: Хүүхдүүд цасан дээр тоглож байхдаа чанга инээлдэв. | The children laughed loudly while playing in the snow.
11: Би өнөө өглөө зах дээрээс гурван кило алим авсан. | I bought three kilograms of apples at the market this morning.
12: Хурал 9 цагт эхэлж 11 цагт дуусна. | The meeting starts at 9 o'clock and ends at 11.
13: Монголчууд жил бүрийн долдугаар сард Наадмын баяр тэмдэглэдэг. | Mongolian people celebrate the Naadam festival every July.
14: Тэр гадаадад байгаа хамгийн сайн найздаа урт захидал бичсэн. | She wrote a long letter to her best friend abroad.
15: Музей даваа гарагаас бусад өдөр бүр арав цагаас зургаан цаг хүртэл нээлттэй. | The museum is open every day except Monday from ten to six.
16: Мы вчера долго гуляли по парку возле нашего дома. | Yesterday we walked in the park near our house for a long time.
17: Библиотека закрывается в шесть часов вечера. | The library closes at six o'clock in the evening.
18: Галт тэрэг 8 цагт хөдөлж 10 цагт хүрнэ. | The train leaves at 7 and arrives at 10.
19: Дэлгүүрт 120 ном, 36 сэтгүүл байна. | The shop has 120 books and 35 magazines.
20: Тэр өдөр бүр өглөө эрт босож гүйдэг. | The government announced a new budget for road construction.
21: Манай багш маргааш шинэ хичээл эхлүүлнэ. | A large flock of birds flew over the frozen lake in autumn.
22: Оюутнууд номын санд шинэ ном уншиж байна. | The students are reading new books in the library.
23: Тэр маш хурдан гүйдэг. | He runs very fast, and every morning before the sun rises over the hills he crosses the river, climbs the long path through the forest, and returns home in time to cook breakfast for his whole family and neighbours.
```

Rows 16 and 17 have a Russian source (wrong language); 18 and 19 change a
number; 20 and 21 pair sentences that are not translations of each other; 22
repeats row 2 exactly; 23 has a target far longer than its source.

Commands (from the repository, with the scratch directory `$S`):

```bash
uv run rinchen filter run --corpus $S/smoke.parquet --config $S/filters.yaml \
    --out-dir $S/out --dry-run     # exit 0, prints the seven stage names
uv run rinchen filter run --corpus $S/smoke.parquet --config $S/filters.yaml \
    --out-dir $S/out               # exit 0, 9.5 s wall
```

It printed `kept 16 of 24 rows`. Outcome per row type:

| Rows | Flagged as | Notable scores |
|---|---|---|
| 0 to 15, natural | none: all 16 kept | `lid_src` 0.62 to 1.0, `lid_tgt` 0.997 to 1.0, `len_ratio` 0.74 to 1.51, `labse` 0.77 to 0.96, `metricx_qe` 0.81 to 4.77 |
| 16, 17, Russian source | `lid_src` | `lid_src` 0.0 (GlotLID: `rus_Cyrl` 0.999 and 1.0); `script_src` 1.0, so `script_ratio` did not catch them; `labse` 0.89 and 0.92, `metricx_qe` 2.7 and 0.8: the translations are right |
| 18, 19, changed number | `numbers` | `num_ok` false; `metricx_qe` 4.4 and 1.3, well under the 8.0 limit, and `labse` 0.83 and 0.92: neither model notices a changed number |
| 20, 21, different sentences | `labse` | `labse` 0.21 and 0.15 (limit 0.7); `metricx_qe` 25.0 on both (the maximum), so `qe` would have caught them too |
| 22, exact duplicate | `duplicate` | same scores as row 2 |
| 23, very long target | `len_ratio_char` | `len_ratio` 9.3 (limit 2.0); `labse` was 0.44, so `labse` would have rejected it too |

Manifest: `metrics` `rows_in` 24, `rows_kept` 16, `reason.lid_src` 2,
`reason.len_ratio_char` 1, `reason.numbers` 2, `reason.duplicate` 1,
`reason.labse` 2, every other reason 0 (including `reason.qe`); `status`: all
stages `ok` except `length: partial: token ratio skipped (no spm_model)`; the
inputs were the corpus and the 1,687,094,687-byte GlotLID file;
`code_version` `102509a`.

Timings: 9.5 s wall for the command, which includes the Python start-up, the
SHA-256 of the 1.7 GB model file (about 0.5 s, once on a first run), and every
model load from the local cache. Per stage, run one at a time: `script_ratio` 0.04 s, `numbers`
under 0.01 s, `length` 0.01 s, `lid` 1.2 s (the model load), `minhash` 0.01 s,
`labse` 4.5 s (the model load and 48 sentences), `qe` 1.3 s (load and 24
pairs). A repeated run printed `skipped: inputs and config are unchanged since
the last run` after 1.1 s; a config with `script_ratio.min_ratio: 1.5` exited 2
with the pydantic message naming the field; a missing corpus exited 3.

After the LaBSE release was added, the same command was run again over the same
rows from a scratch script (exit 0, `kept 16 of 24 rows`, 7.0 s wall, with the
models in the page cache and the memory figures of "Estimates").

**What stood out.** No legitimate pair was rejected by the default priors, but
two natural rows were near a floor: row 6 had `lid_src` 0.623 (Buryat took
0.377 of the probability, floor 0.5) and `labse` 0.766 (floor 0.7), and row 5
had `lid_src` 0.988. The `qe` stage added no rejection of its own here: it
confirmed what `labse` had rejected and missed the number changes. `lid` and
`script_ratio` are different checks: the Russian rows are all Cyrillic.

### Mine-align smoke

A 12-line English file and its 12-line Mongolian translation (a short story,
one sentence per line, also written by the assistant), real LaBSE on the GPU:

```bash
uv run rinchen mine align --src $S/mine/en.txt --tgt $S/mine/mn.txt \
    --src-lang eng_Latn --tgt-lang khk_Cyrl --corpus smoke_story \
    --out $S/mine/out/aligned.parquet
```

Output: `aligned 12 rows (unaligned: 0 source, 0 target)`, exit 0, 6.3 s wall
(most of it start-up and the LaBSE load). All 12 rows were 1-1 pairs in order
(`seg_idx` 0 to 11, checked row by row against the input; `doc_id` is `en`, the
stem of the source file). Manifest metrics: `aligned` 12, `unaligned_src` 0,
`unaligned_tgt` 0, `src_sentences` 12, `tgt_sentences` 12,
`blank_lines_dropped` 0, `param.max_k` 4, `param.skip_cost` 0.9,
`param.merge_penalty` 0.3. A rerun was skipped; an unregistered language code
exited 2; a missing sentence file exited 3. This is the easy case, a
sentence-for-sentence translation with no merges, splits or omissions, so it
shows that the command works and not that the aligner copes with real
documents.

## Known limitations

Priors and validation

- Every threshold is a prior that no real data has tested. The tests prove that
  the stages implement the rules, not that the rules are right.
- The aligner's `merge_penalty` of 0.3 was chosen on synthetic data whose
  embeddings are additive (the embedding of a concatenation is the sum of the
  embeddings of its parts). With the plain vecalign weighting (`merge_penalty`
  0.0) a weak pair rides on a strong neighbour and neighbouring 1-1 pairs merge
  into 2-2: F1 0.906 on the planted data; with 0.3, F1 0.989 on seed 11 (mean
  0.993, minimum 0.978 over seeds 11 to 30); a harder noise setting drops it to
  0.939. LaBSE does not embed additively, so the value for real data is
  unknown.
- The aligner is lenient about a lone pair: a missing neighbour in the margin
  term counts as unrelated, so a side with fewer than `margin_k` candidates
  accepts a pair whenever its cosine distance is below about 0.87 at the
  default `skip_cost`. That is permissive for a strong multilingual encoder;
  the downstream filters must judge such pairs.

Stages

- **The duplicate key is `src + TAB + tgt` only.** It ignores the language
  direction and the split: a corpus with both directions of the same pair keeps
  one of them, and a train row can shadow a later dev row. Filter one direction
  and one split per run, or accept this.
- **Number rules.** The digit multiset ignores order and grouping, so `120` and
  `210` are equal (a few false accepts), and a digit on one side against a
  number word on the other (`3` and `three`) is a false reject. A lone CJK
  `shi` (ten, U+5341), `bai` (hundred, U+767E) or `wan` (ten thousand, U+4E07)
  counts as a number: the adverb "very" (U+5341 U+5206) reads as 10 and "in
  case" (U+4E07 U+4E00) as 10001, and either gives a Chinese row digits its
  digit-free English side lacks, so it is rejected as `numbers`. A lone one
  (U+4E00) is ignored.
  [D12: the numeral folding behind `digit_multiset` (every Unicode decimal digit to its ASCII digit, every run of CJK numerals to the
  digits of its value, a lone CJK digit character to nothing) now lives in `rinchen.text.numerals.fold_numerals`, so that D12's number
  metric (`rinchen.eval.robust_numbers`) folds the same way without importing `rinchen.filter` (polars). D4 behaves exactly as before:
  `tests/unit/test_filter_numbers.py` is unchanged, and `tests/unit/test_filter_numbers_fold.py` compares `digit_multiset` with a frozen
  copy of the old formula over random text. `scores.num_ok` keeps this meaning, the digit multiset; D12's `numbers_acc`
  (`docs/runbooks/12_robustness.md`) is a different metric, a comparison of values, and replaces nothing here: it rejects the 15 of a toy
  model's 67 wrong number rows that the digit multiset accepted (digit transpositions, a lost decimal point; measured in the D12 spike, beam 1).]
- **Script score of a lone Common letter.** A text whose only letters are of
  Common or Inherited script (for example a lone katakana long-vowel mark)
  scores 0.0, not the vacuous 1.0.
- **`lid` accepts Wu as Chinese and does not separate Simplified from
  Traditional** (see the findings). `und_Mong` text is handled by the
  pre-detector only.
- **`qe`** has an adapter for MetricX-QE only. `qe.scorer: cometkiwi` is
  accepted by the config but the command builds no scorer for it, so the stage
  is skipped with `no QE scorer configured`. The adapter drives Eole-internal,
  undocumented helpers (`resolve_preconverted_eole_hf_repo`,
  `EncoderDecoderScoringModel.from_model_dir`, `build_scorer_sentencepiece_transform`,
  `build_template_inputs`, `encode_scorer_inputs`,
  `predict_scores_with_oom_retry`); `eole==0.6.4` is pinned for that reason and
  the `network` test is the only guard against an upgrade that changes them.
  Eole halves the batch down to one pair when the GPU runs out of memory.
- `length.token_ratio` runs only with `length.spm_model`; there is no trained
  student or teacher SentencePiece model yet (D3 trained only the tiny one).

Memory and size (all stages work on the whole corpus in memory; none streams)

- `filter run` holds the input and each stage's result: budget **two to three
  times the corpus size**.
- `minhash` keeps a MinHash signature and index entry per kept row, about 1 to
  2 KB each (1 to 2 GB per million rows), single-threaded, and cannot be
  chunked without losing cross-chunk duplicates.
- `labse` embeds each side of all rows in one call and holds two `n x 768`
  float32 matrices, about 6 KB per row (6 GB per million rows, 60 GB per ten
  million) on top of the corpus. Split large corpora across runs; the
  duplicate check then only sees within a run.
- `lid` scores in slices of 100,000 rows and holds the 1.6 GiB model;
  `qe` slices by batch. `script_ratio` is slow (see Estimates).
- `mine align` treats each file as one document pair: time and memory are
  O(n * m * transitions), unbanded. Measured with `max_k` 4 and a cheap
  embedder (peak resident memory of the process): 600 by 600 sentences take
  0.48 s and 109 MB, 1,500 by 1,500 take 2.7 s and 468 MB, 2,500 by 2,500 take
  6.9 s and 1.1 GB, and the embedder comes on top. The command checks no size
  up front: files of 100,000 lines end in a `MemoryError` traceback. Split
  long documents before aligning.
- `match_documents` holds a float32 cosine matrix: 10,000 by 10,000 documents
  is 400 MB; at 4,000 per side the measured peak was 74 MB without dates and 83
  MB with dates. It is a library function with no command and no caller outside
  its tests, and its URL rules are heuristics.

Commands

- `kept.parquet` keeps the all-null `filter_reason` column, to stay a valid
  corpus table.
- `--seed` is metadata only.
- `--dry-run` shows a QE skip only for the cometkiwi case, not for a machine
  without CUDA.
- Corrupt Parquet bodies and out-of-memory end as tracebacks, not exit codes.
- An error raised by numpy inside `align` is reported as bad embedder output
  (exit 4).
- On a rerun that is not skipped, the inputs are hashed twice (the skip check,
  then the manifest). The manifest does not record the revision of the LaBSE or
  MetricX weights.
- LaBSE prints a "Loading weights" progress bar on stderr.

## Open gates

The D3 gates that the master plan assigned to D4 need real text. **No real
data exists, so none of them has been done**:

- **Abbreviation induction and a boundary-error sample per language.** Induce
  abbreviations from a real corpus (the Punkt approach of Kiss and Strunk,
  2006) and measure sentence-boundary errors on a human-labelled sample of
  about a thousand sentences per language (the D2 annotation app can produce
  it) before tuning the splitter. Not done.
- **Counts of o-like Latin substitutes and uppercase `Y` inside
  Cyrillic-majority tokens in real `khk_Cyrl` text**, before the SentencePiece
  vocabulary is frozen. Not done.
- **Audit of U+200C, U+200D, U+180A..U+180F and U+202F in real
  traditional-script text** (counts and neighbours; words that become identical
  once joiners are removed; a free variation selector that does not follow a
  letter), before D5 and before the strip-the-joiners policy is frozen. Not
  done.

Also open, from the design: a real-data check of every threshold in this
runbook, a re-tune of `merge_penalty` on real embeddings, and a CometKiwi
adapter if that scorer is wanted.

## Operational notes

- **Editor tools rewrite escapes.** The Write and Edit tools of coding agents
  turn backslash-u escapes into raw characters. Source files may hold visible
  non-ASCII letters (Cyrillic, Han, Hangul); invisible, look-alike and
  traditional Mongolian characters are written with `chr(0x....)`. After
  editing, run `grep -nP '[^\x00-\x7F]'` on the file. In this runbook only the
  smoke-run sentences are non-ASCII.
- **Never write under `data/` or `models/` from tests or probes.** The tests
  use `tmp_path`; the `network` tests read the Hugging Face cache and the Eole
  cache only.
- **The `network` tests.** `uv run pytest -m network -q` runs the real-model
  checks (GlotLID parity and default-accept, LaBSE, MetricX-QE) and six older
  ones from D0; they need network or a populated cache and a GPU, and are
  excluded from `just gate`.
- **Where the code lives.** `src/rinchen/filter/` (one module per stage plus
  `pipeline.py`, `reasons.py`, `qe_eole.py`), `src/rinchen/mine/` (`embed.py`,
  `align.py`, `docalign.py`, `pairs.py`), `src/rinchen/corpus/`,
  `src/rinchen/config/filters.py`, `src/rinchen/cli/{filter,mine,errors}.py`,
  test helpers in `tests/support/`.
