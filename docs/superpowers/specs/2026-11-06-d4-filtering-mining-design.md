# D4 - Filtering & mining: design

Date: 2026-11-06. Milestone D4 of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`
(section "D4 - Filtering & mining", builds B6 and B7). Branch `d4-filtering-mining`.

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
src/rinchen/filter/pipeline.py                   stage registry, run_pipeline, reason counts, status
src/rinchen/mine/{embed,align,docalign}.py       Embedder protocol + LaBSE, B6 aligner, doc matcher
src/rinchen/cli/{filter,mine}.py                 the two commands
configs/data/filters.yaml                        default pipeline config (documented priors)
tests/support/                                   planted-failure corpus, fake embedders, tiny LID model
docs/runbooks/04_filtering_mining.md
```

Dependency direction: `cli -> filter/mine -> corpus, config, lang, text, spm`; `filter.labse`
and `filter.qe` use `mine.embed` / a scorer protocol; `mine` never imports `filter`; nothing
imports `cli`. Models are details behind two protocols (`Embedder`, `QEScorer`) injected by
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
| `script_ratio` | `script_src`, `script_tgt`: share of the side's letters that belong to the language's registry scripts (`configs/languages.yaml`); text without letters scores 1.0 | score < `min_ratio` (0.9), reason `script_src` / `script_tgt` (src first) |
| `numbers` | `num_ok`: multiset of ASCII digits equal on both sides after normalising every Unicode decimal digit (incl. Mongolian U+1810..U+1819) and CJK numeral runs to ASCII digits | `num_ok` is false |
| `length` | `len_ratio` = effective_length(tgt) / effective_length(src); effective length = sum over non-whitespace characters of the script weight (default 1.0) | char ratio outside `[min, max]` (0.5, 2.0), or either side has effective length 0 (`len_ratio` null): `len_ratio_char`; token-count ratio (SentencePiece encode of tgt / src) outside `[0.4, 2.5]`: `len_ratio_token`; token check runs only when `spm_model` is configured |
| `lid` | `lid_src`, `lid_tgt`: summed probability of the labels accepted for the expected language among the top `top_k` (3) predictions; traditional-script input (share of U+1800..U+18AF among non-whitespace characters >= 0.5) is labelled `mn_Mong` with probability 1.0 before the model runs | score < `min_prob` (0.5), reason `lid_src` / `lid_tgt` |
| `minhash` | none | the row's `src + TAB + tgt` has estimated Jaccard >= `threshold` (0.8) with an earlier surviving row (5-character shingles, 128 permutations): reason `duplicate`; rows already rejected never shadow later rows |
| `labse` | `labse`: cosine of the sentence embeddings | score < `min_cosine` (0.7) |
| `qe` | `metricx_qe` or `cometkiwi` (by configured scorer) | score outside the scorer's acceptance side of `threshold`; GPU only |

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
collected frame, `reason_counts: dict[str, int]` (kept rows are not counted), and
`stage_status: dict[str, str]`. A status is `ok`, `partial: <what was skipped>` or
`skipped: <reason>`. A stage that reports `skipped` is not applied. The frame given to a
stage is always valid against `CORPUS_SCHEMA`.

The manifest gets (a) `metrics`: `rows_in`, `rows_kept`, and `reason.<name>` for each reason,
(b) a new optional field `status: dict[str, str]` on the `Manifest` model carrying
`stage_status`. The field defaults to empty, so existing manifests stay valid and the schema
string stays `rinchen.manifest/1`.

### CLI

`rinchen filter run --corpus PATH [--config PATH] [--out-dir DIR] [--force] [--dry-run]
[--seed N]`: reads a corpus Parquet, runs the pipeline, writes `scored.parquet` (all rows,
with `filter_reason`), `kept.parquet` (rows with no reason) and `manifest.json` into the out
dir (default `data/clean/<corpus file stem>/`). Skips when `manifest.should_skip` says the
inputs and config are unchanged, unless `--force`. `--dry-run` validates the config and the
input schema and prints the stage plan; it writes nothing. Exit codes as the D3 CLI
(ConfigError 2, DataError 3, UpstreamError 4, other RinchenError 1; message `error: <msg>`
on stderr). On success it prints the manifest path.

`rinchen mine align --src FILE --tgt FILE --src-lang L --tgt-lang L --corpus NAME [--out PATH]
[--max-k 4] [--skip-cost F] [--force]`: each file holds one sentence per line; writes a
corpus-schema Parquet (origin `mined`, split `train`, `doc_id` = source file stem, `seg_idx`
= alignment index, one row per alignment with both sides non-empty, sides joined by a space)
and a manifest whose metrics count `aligned`, `unaligned_src`, `unaligned_tgt`.

### Mining

`mine/embed.py`: `Embedder` protocol (`embed(texts: Sequence[str]) -> numpy float32 array`,
shape `(n, d)`, rows L2-normalised) and `SentenceTransformerEmbedder` (LaBSE by default,
batched, `device="auto"` = CUDA when available).

`mine/align.py` (B6): `align(src, tgt, embed, max_k=4, *, skip_cost=0.9, margin_k=4) ->
list[Alignment]`; `Alignment(src: tuple[int, ...], tgt: tuple[int, ...], cost: float)`; an
alignment with an empty side is a deletion. Per the master plan: overlap concatenations (a
span of consecutive sentences joined by a space and embedded as one text), cosine cost with
margin normalisation by the mean cost of the `margin_k` nearest neighbours, dynamic
programming over the transitions `(a, b)` with `a + b <= max_k` (for `max_k = 4` exactly
1-1, 1-2, 2-1, 2-2, 1-3, 3-1 and the single deletions 1-0 and 0-1, which is the master
plan's set), edge weight `cost * (a + b)`, deletion weight `skip_cost`. The alignment
covers every sentence of both sides exactly once, in order.

`mine/docalign.py`: `Document(doc_id, text, url=None, date=None)`, `DocMatch(src_id, tgt_id,
score, method)`; `normalise_url`; `match_documents(src_docs, tgt_docs, embed, *,
max_date_gap_days=2, min_cosine=0.7)`: first an exact match on language-stripped URLs
(unambiguous keys only, method `url`, score 1.0), then mutual nearest neighbours by document
embedding among the remaining documents whose dates (when both are known) differ by at most
`max_date_gap_days`, cosine >= `min_cosine` (method `embedding`).

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
   download and the master plan marks the invocation UNVERIFIED. Task 8 investigates with a
   bounded effort and adds an adapter only if it can be verified; otherwise the stage skips
   with `skipped: no QE scorer configured`. Cost: no real QE in D4.
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

## Limitations to state in the runbook

Priors only; token-ratio check needs an SPM model path; the CJK lone `shi` (ten) counts as a
number although it can mean "very"; the aligner is O(n*m) per document and unbanded; the
document matcher's URL rules are heuristics; no real-data validation of any threshold.
