# D4 - Filtering & mining: design

Date: 2026-11-06 (updated 2026-11-07 to match the finished implementation). Milestone D4
of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md` (section "D4 - Filtering &
mining", builds B6 and B7). Branch `d4-filtering-mining`. The operating procedure, the
measured numbers and the known limitations are in `docs/runbooks/04_filtering_mining.md`.

## Provenance

The user's instruction for this milestone was: "don't respond back until you finish
everything related to coding". This design, its plan and every ruling below were therefore
made by the builder without a review round. Nothing here is an approved decision; each
ruling lists what it costs if wrong so the user can undo it. The master plan is the binding
authority; where it is silent the rulings fill the gap.

## Intent

Turn raw parallel text into scored, filtered corpora, and mine new parallel text from
unaligned documents. D4 delivers (a) the corpus Parquet contract that the master plan
names but D1-D3 never built, (b) the filter stages and an ordered, configurable pipeline
that writes `scores` plus a first-failure `filter_reason` and a manifest with reason
counts, (c) a vecalign-style sentence aligner and a document matcher, and (d) the two CLI
commands `rinchen filter run` and `rinchen mine align`.

There is no real data in the repository yet (D1 fetched only fixtures). Every threshold is
therefore a documented prior, not a tuned value; the runbook says so.

## Goals

- Corpus contract: `rinchen.corpus.schema.CORPUS_SCHEMA` exactly as in the master plan's
  `<contracts>` block, with readers/writers that refuse a mismatching schema.
- Stages, each `apply(df: pl.LazyFrame, cfg) -> pl.LazyFrame`: `script_ratio`, `numbers`,
  `length`, `lid`, `minhash`, `labse`, `qe`. Each adds its score columns (inside the
  `scores` struct) and sets `filter_reason` only on rows that have none yet.
- `pipeline.run_pipeline` runs the configured stages in order; the order is the precedence
  of `filter_reason`.
- B7 `GlotLID` wrapper and B6 `align`, per the master plan's build-ourselves specs.
- Gates (numeric): see "Gates".

## Non-goals

- No tuning of thresholds on real data (none exists). No FLORES+ use.
- No monolingual filtering (the `MONO_SCHEMA` contract constant is defined; nothing
  consumes it in D4).
- No `corpus/dedup.py` or `corpus/splits.py` (listed in the master-plan layout; owned by
  the milestone that needs cross-corpus dedup and split assignment, D6 onward).
- No profiles (`--profile`), as in the D3 CLI.
- The D3 gates that the master plan assigned to D4 (abbreviation induction,
  boundary-error sample, o-like/uppercase-Y counts, U+200C/U+200D audit) need real text.
  They stay open and are restated in STATE.md.

## Architecture

```
src/rinchen/corpus/{schema,readers,writers}.py   corpus contract (polars schema, Parquet zstd I/O)
src/rinchen/config/filters.py                    pydantic config for every stage + the stage order
src/rinchen/filter/reasons.py                    FilterReason vocabulary + flag()/with_scores() helpers
src/rinchen/filter/{script_ratio,numbers,length,lid,minhash,labse,qe}.py
src/rinchen/filter/qe_eole.py                    MetricX-QE scorer through Eole (the QEScorer adapter)
src/rinchen/filter/pipeline.py                   stage registry, run_pipeline, reason counts, status
src/rinchen/mine/{embed,align,docalign}.py       Embedder + Releasable protocols, LaBSE, B6 aligner, doc matcher
src/rinchen/mine/pairs.py                        alignments -> mined corpus rows and their counts
src/rinchen/cli/{filter,mine}.py                 the two commands
src/rinchen/cli/errors.py                        RinchenError -> exit code mapping shared by spm, filter, mine
src/rinchen/utils/part_file.py                   unique temporary name for write-then-rename
configs/data/filters.yaml                        default pipeline config (documented priors)
tests/support/                                   planted-failure corpus, planted alignments, fake embedders,
                                                 pseudo-text, tiny LID model
docs/runbooks/04_filtering_mining.md
```

Changes to existing modules: `manifest/model.py` (`Manifest.status`), `manifest/stage.py`
(`code_version`, `tool_versions`, `build_inputs`, `output_file`, and the commit protocol both
commands share: `outputs_current`, `refuse_foreign_manifest`, `begin_commit`),
`manifest/store.py` (atomic write through a unique temporary file, strict read), `lang/scripts.py` (`script_of` ignores Common/Inherited letters),
`cli/spm.py` (uses `cli/errors.py`, behaviour unchanged), `pyproject.toml` (mypy overrides
for the untyped `datasketch`, `eole`, `fasttext`). No dependency was added.

Dependency direction: `cli -> filter/mine -> corpus, config, lang`; `filter.labse`
uses `mine.embed` and `filter.qe` a scorer protocol; `filter.qe_eole` implements that
protocol and only the CLI builds it; `mine` never imports `filter`; nothing imports `cli`
except other CLI modules (`cli/errors.py`). Models are details behind two protocols (`Embedder`, `QEScorer`) injected by
the CLI wiring; stages receive them as keyword-only arguments that default to "build from
config".

## Contracts

### Corpus schema (`corpus/schema.py`)

`CORPUS_SCHEMA: pl.Schema`, field order as the master plan:
`id` String (16 hex, `make_row_id`), `src_lang`, `tgt_lang`, `src`, `tgt` String,
`origin` Enum(human, web, mined, bt, ft, synth_llm, synth_nmt, converted), `corpus` String,
`doc_id` String, `seg_idx` Int32, `split` Enum(train, dev, test, holdout), `scores`
Struct(lid_src, lid_tgt, script_src, script_tgt, len_ratio: Float32; num_ok: Boolean;
labse, metricx_qe, cometkiwi: Float32), `filter_reason` String, `created`
Datetime("us", "UTC"). Polars columns are nullable; "unscored" is null. `make_row_id(src_lang,
tgt_lang, src, tgt)` is the first 16 hex characters of the SHA-256 of the four fields joined
by TAB. `MONO_SCHEMA`: id, lang, text, corpus, doc_id, seg_idx, url, scores
Struct(lid, script: Float32).

### Filter reasons (`filter/reasons.py`)

`FilterReason` (StrEnum, values are the strings written to `filter_reason`):
`script_src`, `script_tgt`, `lid_src`, `lid_tgt`, `len_ratio_char`, `len_ratio_token`,
`numbers`, `duplicate`, `labse`, `qe`.

### Stage semantics (all thresholds are priors)

| Stage | Score(s) written | Fails with reason when |
|---|---|---|
| `script_ratio` | `script_src`, `script_tgt`: share of the side's letters that belong to the language's registry scripts (`configs/languages.yaml`); letters of Script=Common or Inherited are neutral (ruling 12); text without letters scores 1.0, text whose only letters are neutral scores 0.0 | score < `min_ratio` (0.9), reason `script_src` / `script_tgt` (src first) |
| `numbers` | `num_ok`: multiset of ASCII digits equal on both sides after normalising every Unicode decimal digit (incl. Mongolian U+1810..U+1819) and CJK numeral runs to ASCII digits | `num_ok` is false |
| `length` | `len_ratio` = effective_length(tgt) / effective_length(src); effective length = sum over non-whitespace characters of the script weight (default 1.0) | char ratio outside `[min, max]` (0.5, 2.0), or either side has effective length 0 (`len_ratio` null): `len_ratio_char`; token-count ratio (SentencePiece encode of tgt / src) outside `[0.4, 2.5]`: `len_ratio_token`; token check runs only when `spm_model` is configured |
| `lid` | `lid_src`, `lid_tgt`: summed probability of the labels accepted for the expected language among the top `top_k` (3) predictions (a code with no entry in `accept` accepts its own label; the default `accept` is `zho_Hans: [cmn_Hani, wuu_Hani]`, ruling 13); traditional-script input (share of U+1800..U+18AF among non-whitespace characters >= 0.5) is labelled `mn_Mong` with probability 1.0 before the model runs; a blank side scores 0.0 | score < `min_prob` (0.5), reason `lid_src` / `lid_tgt` |
| `minhash` | none | the row's `src + TAB + tgt` has estimated Jaccard >= `threshold` (0.8) with an earlier surviving row (5-character shingles, 128 permutations): reason `duplicate`; rows already rejected never shadow later rows |
| `labse` | `labse`: cosine of the sentence embeddings | score < `min_cosine` (0.7) |
| `qe` | `metricx_qe` or `cometkiwi` (by configured scorer) | `metricx_qe` above `max_metricx_qe` (8.0), `cometkiwi` below `min_cometkiwi` (0.55), or NaN; GPU only. Only the MetricX-QE scorer exists (`filter/qe_eole.py`); `cometkiwi` has no adapter and the stage is skipped |

Character weights (default): Han 2.5, Hangul 2.0, Hiragana 1.5, Katakana 1.5; Latin,
Cyrillic, Mongolian and every other character 1.0. `length.by_direction` maps
`"<src_lang>-><tgt_lang>"` to replacement ratio bounds for that direction.

CJK numeral runs: digits `0`-`9` in the CJK forms (`U+3007`, `U+96F6`, `U+4E00..`) with the
multipliers ten, hundred, thousand and ten-thousand are converted to their value (`2024` for
the positional form, `12` for ten-two, `12000` for one ten-thousand two thousand); a lone
digit character (`U+4E00` alone) is not a number and is ignored. The verified prototype
cases are in the plan, Task 4.

Because every stage scores every row, a row rejected early still gets later scores; only
`filter_reason` is first-failure. Exception: `minhash` considers only rows without a
`filter_reason`.

### Pipeline (`filter/pipeline.py`)

`run_pipeline(df: pl.LazyFrame, cfg: FilterConfig, *, embedder: Embedder | None = None,
qe_scorer: QEScorer | None = None) -> PipelineResult`. `PipelineResult` carries the
collected frame, `reason_counts: dict[str, int]` (every `FilterReason`, 0 when unused; kept
rows are not counted), and `stage_status: dict[str, str]`. A status is `ok`,
`partial: <what was skipped>` (`length` without an `spm_model`: `partial: token ratio
skipped (no spm_model)`) or `skipped: <reason>` (`qe`: `no QE scorer configured` or `no CUDA
device`). A stage that reports `skipped` is not applied. The frame given to a stage is
always valid against `CORPUS_SCHEMA`. The frame is collected after every stage (ruling 14),
so an error surfaces at the stage that caused it. The embedder it was given is released, when
it implements `Releasable`, as soon as the `labse` stage has run (also when that stage fails),
so LaBSE is not on the GPU during `qe`; the QE scorer is the caller's to release.

The manifest gets (a) `metrics`: `rows_in`, `rows_kept`, and `reason.<name>` for each reason,
(b) a new optional field `status: dict[str, str]` on the `Manifest` model carrying
`stage_status`. The field defaults to empty, so existing manifests stay valid and the schema
string stays `rinchen.manifest/1`.

### CLI

`rinchen filter run --corpus PATH [--config PATH] [--out-dir DIR] [--force] [--dry-run]
[--seed N]`: reads a corpus Parquet, runs the pipeline, writes `scored.parquet` (all rows,
with `filter_reason`), `kept.parquet` (rows with no reason) and `manifest.json` into the out
dir (default `data/clean/<corpus file stem>/`). Skips when `manifest.stage.outputs_current`
says the previous run is reusable: both output tables exist, the manifest there is a `filter`
manifest that lists them, and the inputs (the corpus and each model file the config names)
and the config are unchanged; `--force` never skips. The manifest is the commit marker:
`begin_commit` removes it before any output is written, and it is written atomically last.
Two guards run before any output is written or model loaded: `--corpus` may not be one of
the two output tables (ConfigError, 2: the run would overwrite its own input; checked first,
before anything is read), and an out dir that holds the manifest of another stage is
refused (DataError, 3) and its manifest kept (also under `--dry-run`).
`--dry-run` validates the config, the model files it names and the input schema and prints
the stage names, with the reason a `qe` stage is skipped only when that can be known
without loading a model; it writes nothing. Exit codes as the D3 CLI
(ConfigError 2, DataError 3, UpstreamError 4, other RinchenError 1; message `error: <msg>`
on stderr). On success it prints the manifest path.

`rinchen mine align --src FILE --tgt FILE --src-lang L --tgt-lang L --corpus NAME [--out PATH]
[--max-k 4] [--skip-cost 0.9] [--merge-penalty 0.3] [--force]`: each file holds one sentence
per line (blank lines dropped and counted); writes a corpus-schema Parquet (origin `mined`,
split `train`, `doc_id` = source file stem, `seg_idx` = the position among the emitted rows,
from 0; one row per alignment with both sides non-empty, the sentences of a side joined by a
space; a deletion emits no row) and a manifest whose metrics count `aligned`,
`unaligned_src`, `unaligned_tgt`, `src_sentences`, `tgt_sentences`, `blank_lines_dropped`
and record the parameters as `param.max_k`, `param.skip_cost`, `param.merge_penalty`. The
parameters stand in for a config file in `config_sha256`, and the inputs are hashed before
they are read. It shares the commit protocol of `filter run` (`outputs_current` with the
stage `mine_align`, `begin_commit`) and its two guards: `--out`, or the manifest beside it,
may not be `--src` or `--tgt` (ConfigError, 2), and a directory that holds the manifest of
another stage is refused (DataError, 3).

### Mining

`mine/embed.py`: `Embedder` protocol (`embed(texts: Sequence[str]) -> numpy float32 array`,
shape `(n, d)`, rows L2-normalised) and `SentenceTransformerEmbedder` (LaBSE by default,
batched, `device="auto"` = CUDA when available). A `Releasable` protocol (`release() -> None`)
names what gives its memory back; `SentenceTransformerEmbedder.release` drops the loaded
model, runs `gc.collect()` and empties the CUDA cache when torch is loaded, and is safe before
the first `embed` and when called twice.

`mine/align.py` (B6): `align(src, tgt, embed, max_k=4, *, skip_cost=0.9, margin_k=4,
merge_penalty=0.3) -> list[Alignment]`; `Alignment(src: tuple[int, ...], tgt: tuple[int, ...], cost: float)`; an
alignment with an empty side is a deletion. Per the master plan: overlap concatenations (a
span of consecutive sentences joined by a space and embedded as one text), cosine cost with
margin normalisation by the mean cost of the `margin_k` nearest neighbours, dynamic
programming over the transitions `(a, b)` with `a + b <= max_k` (for `max_k = 4` exactly
1-1, 1-2, 2-1, 2-2, 1-3, 3-1 and the single deletions 1-0 and 0-1, which is the master
plan's set), edge weight `cost * (a + b) + merge_penalty * (a + b - 2)`, deletion weight
`skip_cost`. The alignment covers every sentence of both sides exactly once, in order.
`cost` is `(1 - cos(x, y)) / (0.5 * (knn_x + knn_y))`, where `knn_x` is the mean of the
`margin_k` smallest cosine distances from `x` to the spans of the other side that have the
size of `y` (and `knn_y` the same for `y`); when fewer than `margin_k` spans exist, the
missing neighbours count as unrelated, at distance 1.0 (ruling 16), and the denominator is
floored at 1e-6. `merge_penalty` departs from plain vecalign (`merge_penalty=0.0`): ruling 15.
Ties in the DP go to 1-1, then the other pairings by `a + b` and then `a`, then 0-1, then
1-0.

`mine/docalign.py`: `Document(doc_id, text, url=None, date=None)`, `DocMatch(src_id, tgt_id,
score, method)`; `normalise_url`; `match_documents(src_docs, tgt_docs, embed, *,
max_date_gap_days=2, min_cosine=0.7)`: first an exact match on language-stripped URLs
(unambiguous keys only, method `url`, score 1.0), then mutual nearest neighbours by document
embedding among the remaining documents whose dates (when both are known) differ by at most
`max_date_gap_days`, cosine >= `min_cosine` (method `embedding`). A date-blocked pair never
matches, whatever `min_cosine` is; a document whose embedded part (the first 2000
characters) is blank is left out of the embedding pass. It is a library function with no
command yet.

`mine/pairs.py`: `count_alignments` and `aligned_corpus_frame` turn alignments into the mined
corpus rows described under "CLI".

## Testing strategy and gates

- Tests first; numeric thresholds are gates and are never loosened to make a test pass.
- Shared fixture `tests/support/planted_corpus.py`: a 50-row corpus (khk_Cyrl -> eng_Latn) with
  a planted ledger, built so that each planted row violates exactly one stage. Planted
  counts, which are the exact reason counts asserted by the stage tests and by the
  end-to-end run: clean 28, `script_src` 2, `script_tgt` 2, `lid_src` 2, `lid_tgt` 2,
  `len_ratio_char` 3, `len_ratio_token` 2, `numbers` 3, `duplicate` 3, `labse` 3 (`qe` skipped,
  so no `qe` rows).
- Fake embedders (`tests/support/embedders.py`): `ConceptEmbedder` (exact text -> concept id,
  same concept = same vector) for the filter tests, `HashEmbedder` (bag of hashed tokens,
  so concatenations are additive) for the aligner and document matcher.
- Tiny LID model (`tests/support/tiny_lid.py`): a supervised fastText model trained in the test
  from the tiny-corpus pseudo-languages; it is never committed.
- Gates: planted reason counts exact; MinHash recall >= 0.95 and precision >= 0.95 against the
  exact Jaccard ground truth; aligner F1 >= 0.95 on planted alignments with a deterministic
  fake embedder; numbers filter handles `1,5` / `1.5`; QE stage "skipped" without GPU or
  scorer; B7 parity 100% top-1 against golden JSON (network marker); tiny end-to-end
  `rinchen filter run` green.
- `network`-marked tests use the Hugging Face cache (outside the repository); nothing is
  written under `data/` or `models/`.

## Rulings

1. **Scope split into twelve tasks on one branch**, with the pipeline last among the filters
   because it imports every stage. Cost if wrong: re-slicing; no code impact.
2. **`corpus/` is created now** (schema, readers, writers only) because the master plan's
   `<contracts>` require it and D1-D3 did not build it. `dedup.py` and `splits.py` are left
   out. Cost: they are added later in the owning milestone.
3. **Digit-multiset reading of the numbers rule**: the master plan says "digit multiset
   match"; it is implemented literally (multiset of digit characters, not of number
   tokens), which makes `1,5` / `1.5` and re-ordered dates equal and `12` / `21` equal.
   Cost: a few false accepts. The stricter token form can replace it behind the same
   `num_ok` score.
4. **Every stage scores every row; only `minhash` ignores already-rejected rows.** Cost:
   wasted GPU time on rejected rows (script mismatch is a minority). Gain: scores are
   complete for analysis.
5. **QE ships as a stage plus a `QEScorer` protocol.** The Eole-native MetricX/CometKiwi
   scorers exist in the installed Eole 0.6.4 (`eole.scorers.eole_metricx`) but need a model
   download and the master plan marks the invocation UNVERIFIED. Outcome: Task 8 verified
   the MetricX-QE invocation on the GPU and shipped the adapter (`filter/qe_eole.py`; a
   matched pair scores 0.0 and 0.09, a mismatched one 9.9 and 25.0); there is no CometKiwi
   adapter, and `qe.scorer: cometkiwi` skips the stage with `skipped: no QE scorer
   configured`. Cost: the adapter drives undocumented Eole helpers, so `eole==0.6.4` is
   pinned for it and the `network` test is the only guard; first use downloads about 6 GB.
6. **LaBSE through sentence-transformers** (already a dependency; inference only, no training)
   rather than raw `transformers`, so pooling and the dense head match the published model.
7. **The aligner embeds overlap concatenations as text** (as vecalign does), not as mean
   vectors. Cost: up to `max_k - 1` times more embedding calls.
8. **`Manifest.status`** added (optional, defaults empty) to record the "skipped" status the
   master plan requires. Cost: a model field in a shared contract.
9. **GlotLID parity reference** is the raw fastText Python API on the same model: the fastText
   command-line binary is not installed and the Python API is the same library. The golden
   JSON is captured once and stored. Cost: weaker than the master plan's wording; documented.
10. **Latin letters in Cyrillic text are tolerated up to 10%** (`min_ratio` 0.9) because
    Mongolian web text carries brand names.

Rulings 11 to 17 were made during the build (ledger tags R-b to R-h) and, like rulings 1 to
10, without a review round.

11. **Source files may hold visible non-ASCII letters (R-b).** Cyrillic, Han and Hangul
    letters are written as themselves in source and tests; invisible, look-alike and
    traditional Mongolian characters are built with `chr(0x....)`. D3's "ASCII only" rule was
    a workaround for the editor tools turning escapes into raw characters. Cost if wrong:
    none; it is a convention.
12. **Letters of Script=Common or Inherited are neutral in `script_of` (R-c).** The katakana
    long-vowel mark (U+30FC) and modifier letters such as U+02BC are skipped in the numerator
    and the denominator; letters of a real script that D3 does not name (Greek, Arabic) still
    count in the denominator only. Why: the Japanese sentence "coffee, please" written with
    U+30FC scored 0.778 and was falsely rejected at 0.9. Cost if wrong: callers of
    `script_of` (the spm fixture test, the golden test, the length stage's character lookup)
    see slightly different shares for Common letters; reverting is three lines. A text whose
    only letters are neutral still scores 0.0.
13. **The `lid` accept table defaults to `{zho_Hans: [cmn_Hani, wuu_Hani]}` (R-d, amended
    after review).** Run on 20 lines per registry language, the real GlotLID v3 labels
    Chinese `cmn_Hani` (19 of 20) or `wuu_Hani` (1 of 20), never `zho_Hans`, and traditional
    Mongolian `und_Mong`; the first version of the fix (`[cmn_Hani]` alone) still flagged one
    of twenty. A table the user supplies replaces the default; the wrapper's pre-detector
    supplies `mn_Mong`. Cost if wrong: text labelled Wu is also accepted as Chinese, and
    `Hani` labels do not separate Simplified from Traditional (neither can `script_ratio`,
    both are Han), so Simplified versus Traditional is not checked in D4.
14. **The pipeline collects after every stage (R-e).** The frame is materialised between
    stages and made lazy again for the next, so an error surfaces at the stage that caused it
    and no query plans the LID model, LaBSE and the QE model together. Cost if wrong: no
    cross-stage lazy fusion, which does not matter because every stage runs Python batch
    functions. Correction from the 2026-11-07 smoke run, fixed in the final review wave:
    collecting after every stage did not by itself keep LaBSE and the QE model apart on the
    GPU, because the CLI kept the embedder alive for the whole run (6,665 MiB allocated during
    `qe`, about 1,800 MiB of it LaBSE). The pipeline now releases an embedder that implements
    `Releasable` right after the `labse` stage; measured after the fix, `qe` runs with 4,866
    MiB allocated (the runbook has the numbers). Cost if wrong: an embedder given to
    `run_pipeline` without a `release` keeps its model.
15. **The aligner has a `merge_penalty` (R-f).** Edge weight `cost * (a + b) +
    merge_penalty * (a + b - 2)`, keyword-only, default 0.3; 0.0 is plain vecalign. Why: with
    the pinned weighting `cost * (a + b)` a weak pair rides on a strong neighbour (the cosine
    of a concatenation lets one 2-2 beat two 1-1) and F1 on the planted data was 0.906; 0.3
    gives 0.989 on seed 11 (mean 0.993, minimum 0.978 over seeds 11 to 30), and 0.3 to 0.5
    passes seeds 31 to 50. Cost if wrong: one more tunable constant, tuned on synthetic data
    with additive embeddings only; the real-data value is unknown.
16. **Missing neighbours in the margin term count as unrelated (R-g).** When fewer than
    `margin_k` spans exist, the missing ones are taken at cosine distance 1.0, so a lone or
    tiny pair is not judged against itself and can align. Cost if wrong: lenient; at the
    default `skip_cost` a lone pair is accepted when its cosine distance is below about
    0.87, which is permissive for LaBSE, so the downstream filters must judge it.
17. **One shared CLI error mapping (R-h).** `cli/errors.py` owns the `RinchenError` to exit
    code mapping (ConfigError 2, DataError 3, UpstreamError 4, other 1) and `cli/spm.py`,
    `cli/filter.py` and `cli/mine.py` use it; it was a third copy otherwise. `mine align`
    also gained `--merge-penalty` and records its parameters in the manifest. Cost if wrong:
    D3's `cli/spm.py` was edited (its behaviour and tests are unchanged and green).

## Limitations

The runbook (`docs/runbooks/04_filtering_mining.md`, "Known limitations") has the full list
with numbers. In short: every threshold is a prior that no real data has tested; the
token-ratio check needs an SPM model path; the duplicate key ignores direction and split;
the digit-multiset rule has false accepts (`120` and `210`) and the lone CJK `shi` (ten),
`bai` and `wan` count as numbers although they can mean "very" or "in case", which gives
false rejects; Simplified versus Traditional Chinese is not checked; the aligner and
`match_documents` are O(n * m) and unbanded per document pair (measured memory in the
runbook), their URL rules are heuristics, `match_documents` has no command, and
`merge_penalty` was tuned on synthetic data only; every stage holds the corpus in memory;
there is no CometKiwi adapter; the MetricX-QE adapter depends on Eole-internal helpers pinned
by `eole==0.6.4`; `--seed` is metadata only; errors from a corrupt Parquet body escape as
tracebacks.
