# D6 - Synthetic data: design

Date: 2026-11-08. Milestone D6 of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`
(section "D6 - Synthetic data"; builds the `mbr` module of the build-ourselves specs). Branch
`d6-synthetic-data`. The operating procedure will be in `docs/runbooks/06_synthetic.md`.

## Provenance

The user's instruction for this stretch was "Proceed. I'm going to sleep now. Don't stop until you
finish everything. Commit everything." This design, its plan and every ruling below were therefore
made by the builder without a review round and without anyone to ask. Nothing here is an approved
decision; each ruling lists what it costs if wrong. The master plan is the binding authority;
where it is silent the rulings fill the gap. Nothing is merged to `main` and nothing is pushed:
the work stays on the milestone branch until the user decides.

## Intent

Turn monolingual text into synthetic parallel text with external generators used only for
inference (an LLM behind an OpenAI-compatible server, a CTranslate2 NLLB or MADLAD model), choose
the best candidate per source with a script-purity filter, a QE floor and minimum Bayes risk
(MBR) selection, and write the result as corpus rows (`CORPUS_SCHEMA`) with the right data
origin. D6 also delivers the `rinchen.mbr` module the master plan lists.

No GPU server, no real generator model and no real monolingual corpus exist in the repository, so
every network and model path is tested against fakes (`httpx.MockTransport`, a stub CTranslate2
translator, a fake QE scorer). The runbook says which paths were never run for real.

## Goals

- `rinchen.mbr`: a `Utility` protocol, a chrF utility, a negation wrapper, and
  `select(cands, utility, pseudo_refs) -> list[int]` exactly as the master plan specifies.
- `rinchen.synth`: prompts, an OpenAI-compatible client (vLLM) and a llama.cpp client, a CTranslate2
  baseline translator, deterministic sharding with crash-safe resume, selection, bt/ft job runners
  and `rinchen synth run`.
- Gates (numeric or exact): the fake-server run resumes after a crash without re-requesting finished
  shards; MBR picks the planted best candidate; the purity filter rejects non-Cyrillic outputs;
  sharding is deterministic; the tiny synth run through the CLI yields exact planted counts.

## Non-goals

- Running any real generator. The vLLM launch command, GPU memory and throughput figures in the
  runbook are operator guidance, not measurements.
- COMET and MetricX as MBR utilities (needs the B1 scorers that D4 deferred); the `Utility`
  protocol is the seam.
- Training-time tag insertion (`<bt>`, `<ft>` prepended to the source belongs to the D7/D8 data
  builders); D6 supplies the origin column and a tag helper.
- Document-level synthesis, terminology data, back-translation iteration.
- Any Inner Mongolian data or variety, any galig text, any fine-tuning of a generator.

## Architecture

```
src/rinchen/mbr/
  utility.py      Utility protocol, ChrfUtility, NegatedUtility
  mbr.py          select(cands, utility, pseudo_refs) -> list[int]
src/rinchen/config/synth.py        SynthConfig (generators, jobs, selection), load_synth_config
configs/data/synth.yaml            defaults = the SynthConfig defaults; every number a prior
src/rinchen/synth/
  tags.py         job kind + generator kind -> corpus origin; origin -> data-origin tag
  translator.py   Translator protocol: translate(texts, src_lang, tgt_lang) -> list[list[str]]
  prompts.py      per-direction prompt templates, language names, Cyrillic-output instruction
  vllm_client.py  OpenAICompatTranslator (chat completions, threads, retries, n, seed)
  llamacpp_client.py  LlamaCppTranslator (same interface, one request per candidate)
  ct2_baselines.py    CT2Translator (NLLB and MADLAD families, n-best)
  shards.py       monolingual input, deterministic sharding, shard store, resume
  select.py       purity -> QE floor -> MBR, with per-reason counts and row construction
  bt.py / ft.py   job runners: plan the translation direction and build the rows
  run.py          run_job: shards -> generate -> select -> corpus + metrics
src/rinchen/cli/synth.py           rinchen synth run
tests/support/fake_llm_server.py, planted_synth.py
docs/runbooks/06_synthetic.md
```

Layering: `mbr` depends on `metrics.bleu_chrf` and numpy only. `translator` is the leaf of
`synth`; `prompts` on `translator` types only; the three client modules implement `translator`;
`shards` on polars and `corpus`; `select` on `mbr`, `lang.scripts`, `filter.qe.QEScorer` and
`corpus.schema`; `bt`/`ft`/`run` compose them; the CLI is the only module that writes the manifest.

## Contracts

### `mbr`

- `Utility` (Protocol): `def score(self, hyp: str, ref: str) -> float` (higher is better).
- `ChrfUtility`: sentence-level chrF2 through `rinchen.metrics.bleu_chrf.chrf`, in [0, 100].
- `NegatedUtility(inner)`: `score = -inner.score` for metrics where lower is better (MetricX).
- `select(cands: Sequence[Sequence[str]], utility: Utility, pseudo_refs: Sequence[Sequence[str]] |
  None = None) -> list[int]`: for each segment `i` the index of the candidate with the highest
  expected utility against `pseudo_refs[i]` (or against `cands[i]` itself when `pseudo_refs` is
  None). When the references are the candidates, a candidate is not scored against itself.
  Expected utility is the mean over the references. Ties go to the lowest index; a segment with one
  candidate returns 0; a segment with no candidates raises `ValueError` naming the segment;
  `pseudo_refs` of a different length than `cands`, or a segment whose reference list is empty
  while it has more than one candidate, raises `ValueError`. Deterministic.

### `config` (`SynthConfig`, frozen, extra forbidden, every number an unvalidated prior)

- `generators: dict[str, GeneratorConfig]`, `GeneratorConfig = VllmConfig | LlamaCppConfig |
  Ct2Config` discriminated on `kind`. Common fields: `n` (candidates per source, default 4),
  `temperature` (0.7), `top_p` (0.95), `max_tokens` (256). `VllmConfig`: `base_url`
  (default `http://127.0.0.1:8000/v1`), `model`, `concurrency` (32), `timeout_s` (120), `retries` (4),
  `backoff_s` (1.0). `LlamaCppConfig`: the same with default `base_url http://127.0.0.1:8080/v1`
  and `concurrency` 8. `Ct2Config`: `model_dir`, `spm_model`, `family` (`nllb` | `madlad`), `device`
  (`cpu`), `beam_size` (4), `batch_size` (16).
- `jobs: dict[str, JobConfig]`: `kind` (`bt` | `ft`), `mono_lang`, `other_lang` (both must be
  language registry codes and one of the two must be `khk_Cyrl`), `generator` (a key of
  `generators`), `shard_size` (512). For `bt`, the monolingual text is the target of the pairs
  and the generator translates it into `other_lang` to make the source; for `ft`, the monolingual
  text is the source and the generator translates it into `other_lang` to make the target.
- `selection`: `min_script_share` (0.9, the share of the synthetic side's letters that must belong
  to the registry scripts of its language); `qe`: exactly the D4 `rinchen.config.filters.QeConfig`
  (`scorer` `metricx_qe` | `cometkiwi`, `max_metricx_qe` 8.0 where LOWER is better, `min_cometkiwi`
  0.55 where higher is better, `batch_size` 32), reused unchanged so both stages mean the same
  thing by "QE"; `mbr_utility` (`chrf`).
- `load_synth_config(path=SYNTH_CONFIG_PATH)` raises `ConfigError` on any violation;
  `configs/data/synth.yaml` loads to exactly the `SynthConfig()` defaults.

### `tags`

`origin_for(kind: str, generator_kind: str) -> str`: `bt` jobs give origin `bt`; `ft` jobs give
`synth_llm` for the `vllm` and `llamacpp` generator kinds and `synth_nmt` for `ct2` (ruling 1).
`data_origin_tag(origin: str) -> str | None`: `bt` -> `<bt>`; `ft`, `synth_llm`, `synth_nmt` ->
`<ft>`; every other origin -> None. Raises `ConfigError` for an unknown kind.

### `translator`

`Translator` (Protocol): `def translate(self, texts: Sequence[str], src_lang: str, tgt_lang: str)
-> list[list[str]]`: one list of candidates per text, in input order, best first when the backend
ranks them; a candidate is a plain string; the list may be shorter than `n` or empty (a refusal or
an empty generation), never longer.

### `prompts`

`render_prompt(src_lang, tgt_lang, text) -> str` and `render_messages(...) -> list[dict[str, str]]`
(one user message). Templates exist for every direction between `khk_Cyrl` and each of the eight
partner codes of `configs/languages.yaml`; a direction outside that set raises `ConfigError`. The
prompt names both languages in English ("Mongolian (Khalkha, Cyrillic script)"), forbids
explanations and quotation marks, and, when the target is `khk_Cyrl`, instructs "Write the
translation in Mongolian Cyrillic script only." A prompt never mentions traditional script or
Inner Mongolia. Template text is the builder's, not a published template of any model (ruling 4).

### Clients

- `OpenAICompatTranslator(cfg: VllmConfig, *, client: httpx.Client | None = None, sleep=time.sleep)`:
  `POST {base_url}/chat/completions` with `model`, `messages`, `temperature`, `top_p`, `max_tokens`,
  `n`, and a per-request `seed` derived from the text, the language pair and the candidate count
  (deterministic across runs). Each text is one request; up to `concurrency` run in a
  `ThreadPoolExecutor`; results keep input order. Retries on `httpx.TransportError`, HTTP 429, 500,
  502, 503 and 504 with exponential backoff `backoff_s * 2**attempt` (capped at 30 s) up to
  `retries`; 4xx other than 429 is not retried; exhaustion, a non-JSON body or a body without
  `choices[*].message.content` raises `UpstreamError` naming the status and the attempt count.
  Candidate strings are stripped; an empty string is dropped; duplicates are removed keeping order.
- `LlamaCppTranslator(cfg: LlamaCppConfig, ...)`: the same contract, but it issues `n` separate
  requests per text (the server's `n` support varies by build), each with `seed + k`, and merges
  them.
- `CT2Translator(cfg: Ct2Config, *, translator=None, processor=None)`: loads `ctranslate2.Translator`
  and a SentencePiece processor (injectable for tests). `nllb`: source tokens `[src_lang_code,
  *pieces, "</s>"]` with FLORES codes (identical to ours) and `target_prefix=[[tgt_lang_code]]`;
  `madlad`: source tokens `["<2xx>", *pieces]` with the codes `en ru zh ko ja de kk tr mn`, no
  target prefix. `translate_batch(..., beam_size=beam_size, num_hypotheses=n, return_scores=False)`
  with `max_batch_size=batch_size`; hypotheses are decoded with the processor and the NLLB target
  language token is removed. Raises `ConfigError` for a language outside the family's table.

### `shards`

- `read_monolingual(path, lang=None) -> pl.DataFrame` requires columns `id`, `lang`, `text` (any
  others are kept), non-empty strings, unique ids; filters to `lang` when given; raises `DataError`
  naming the problem. Sorted by `id`.
- `plan_shards(frame, shard_size) -> tuple[Shard, ...]`: contiguous slices of the id-sorted frame of
  `shard_size` rows (the last may be shorter); `Shard(index, ids, texts)`. Deterministic: the same
  input and `shard_size` give the same shards.
- `job_fingerprint(...) -> str`: sha256 over the input frame's (id, text) rows, the job and its
  generator config, `shard_size` and the seed; shards live in `<out_dir>/shards-<fingerprint[:12]>/`
  so a changed input or config never reuses a stale shard.
- `ShardStore(directory)`: `write(index, ids, candidates)` writes `shard-<index:05d>.parquet`
  (columns `id: String`, `candidates: List(String)`) through a `.part` file and an atomic rename;
  `read(index)` validates the file (columns, row count against the planned ids) and returns it;
  `completed(shards)` lists the indexes with a valid file; a corrupt or short file counts as not
  completed and is regenerated.

### `select`

`select_candidates(rows: SourceRows, candidates, *, cfg, scorer, kind, ...) -> Selection` applies,
in this order and per candidate: empty string (dropped, reason `empty`); script purity below
`min_script_share` for the synthetic side's language (reason `purity`); QE worse than the D4
threshold of the configured scorer (above `qe.max_metricx_qe` for `metricx_qe`, below
`qe.min_cometkiwi` for `cometkiwi`; a score exactly at the threshold passes; a NaN score fails;
reason `qe`; the scorer is called with `qe.batch_size` pairs per call and a wrong number of scores
raises `UpstreamError`, as in the D4 stage; when no scorer is given the QE step is skipped and the
status says so); then MBR
among the survivors (`mbr.select(survivors, utility)` with `pseudo_refs=None`, so each survivor is
scored against the OTHER survivors: passing the survivors as explicit pseudo-references would
score every candidate against itself; a single survivor wins unscored). A source with no survivor produces no row and counts under `no_survivor`. The result
carries the selected index per source, the QE score and the purity of the winner, and counts per
reason (candidates dropped per reason, sources without survivor, sources selected).
`build_rows(...) -> pl.DataFrame` makes `CORPUS_SCHEMA` rows: `bt`: `src` synthetic, `tgt` real,
`src_lang` the job's `other_lang`, `tgt_lang` its `mono_lang`, `scores.script_src` the purity,
`scores.metricx_qe` or `scores.cometkiwi` (by the configured scorer) the QE score when a scorer ran; `ft`: `src` real, `tgt` synthetic, the languages
the other way round, `scores.script_tgt`; `origin` from `tags.origin_for`; `corpus` =
`"<kind>:<generator name>"`; `split` `train`; `id` from `corpus.schema` row-id helper over (src,
tgt, languages); `created` now. Pairs whose two sides are identical are dropped (reason `copy`).

### `run`

`run_job(job: JobConfig, generator: Translator, generator_kind: str, frame: pl.DataFrame, *,
out_dir: Path, cfg: SynthConfig, scorer: QEScorer | None, seed: int, force: bool = False,
progress=None) -> RunResult`: plans shards, generates the missing ones (`ShardStore`), selects, and
returns the corpus frame, the counts, the shard statistics (`planned`, `reused`, `generated`) and
the fingerprint. A generator exception on a shard raises `UpstreamError` with the shard index after
the finished shards are on disk; a rerun resumes. It never writes the corpus or a manifest.

### CLI (`rinchen synth run`)

`rinchen synth run JOB --input FILE.parquet [--config configs/data/synth.yaml] [--out-dir DIR]
[--seed 0] [--force] [--dry-run]`. `JOB` names a job of the config. Default out dir
`data/synth/<job>`. Seams (module-level, looked up at call time, replaced by tests):
`build_translator(generator_cfg)` and `build_qe_scorer(selection_cfg)`. `--dry-run` prints the job,
the generator, the number of source rows and shards and the fingerprint, and writes nothing. A run
writes `corpus.parquet` (through `write_corpus`), `counts.json`, and the manifest last
(`begin_commit`, `.part` files, the D4 commit protocol; stage `synth`), refuses a foreign manifest
before any write (also under `--dry-run`), refuses an `--out-dir` that is the input's directory, and
maps errors through `rinchen.cli.errors` (ConfigError 2, DataError 3, UpstreamError 4, other 1).

## Testing strategy and gates

Tests first. Fakes only: no network, no model, no GPU.

- MBR: a planted candidate set where the best candidate is the consensus is chosen; ties go to the
  lowest index; one candidate returns 0; `pseudo_refs` given explicitly changes the winner exactly
  as computed by hand; `NegatedUtility` flips an order; determinism; `ValueError` cases;
  `hypothesis`: the index is always in range and equal for permutation-invariant identical inputs.
- Config: defaults equal the YAML; every violation rejected; a job whose pair lacks `khk_Cyrl`
  rejected; a job naming an unknown generator rejected.
- Tags: the origin table exactly; the tag table exactly.
- Prompts: every direction renders; the Cyrillic instruction is present exactly when the target is
  `khk_Cyrl`; no template mentions traditional script or Inner Mongolia; an unsupported direction
  raises.
- Clients (`httpx.MockTransport` fake server in `tests/support/fake_llm_server.py`): request body
  fields exact; `n` and seed handling; order preserved under concurrency; retry on 429/503 then
  success; retry exhaustion raises `UpstreamError` with the attempt count; a 400 is not retried;
  malformed bodies raise; backoff sleeps recorded through the injected `sleep`; the llama.cpp client
  issues `n` requests with `seed + k`.
- CT2: family token conventions asserted with literals through stubs; language table errors.
- Shards: deterministic plan (property: concatenating shards reproduces the input; same input and
  size, same shards); fingerprint changes with each ingredient; store atomicity (a failed write
  leaves no final file); a corrupt shard is regenerated; shard resume after a crash: a fake generator
  that fails on shard 2 leaves shards 0-1 on disk, a second run with a working generator requests
  only shards 2..N and the final corpus equals an uninterrupted run's.
- Select: exact reason counts on a planted candidate set (empty, non-Cyrillic output rejected by
  purity, low-QE, NaN QE, survivors) with a fake scorer; MBR picks the planted best among survivors;
  the QE-skipped status without a scorer; identical-sides dropped as `copy`.
- Run and CLI: the tiny synth run (the STOP gate): a planted monolingual file of 40 rows, the fake
  server, the fake scorer: exact counts of rows kept and dropped per reason, the corpus validates
  against `CORPUS_SCHEMA`, origins and corpora as ruled, manifest written last, rerun resumes
  (zero generator calls) and produces an identical corpus; `--dry-run` writes nothing; error exit
  codes; foreign-manifest refusal.

## Rulings

1. **Origin encodes role and generator family.** `bt` jobs give `bt`; `ft` jobs give `synth_llm` or
   `synth_nmt` by generator kind; the `ft` origin itself stays reserved for forward translation by
   our own teacher (D10). The master plan lists the four origins without saying which job produces
   which. Cost if wrong: one table in `tags.py` and a re-labelling of produced corpora.
2. **One `Translator` protocol for every generator.** The master plan's `ct2_baselines.translate`
   signature is generalised to the LLM clients, so jobs are generator-agnostic. Cost: LLM-specific
   features (system prompts, few-shot) are not expressible without a protocol change.
3. **Chat completions with one request per text** and server-side batching (vLLM's continuous
   batching), `concurrency` threads on the client, instead of the completions endpoint with prompt
   lists (which would need the model's chat template in the client). Cost: more HTTP overhead per
   text.
4. **Prompt templates are the builder's wording**, not a published Hy-MT prompt: the master plan names
   the model but not its template and no source for it is available offline. They state the task
   plainly and carry the Cyrillic instruction. Cost: translation quality of a real run may differ
   from the model's recommended prompt; the templates are one dictionary to replace.
5. **Selection order purity, QE, MBR.** Cheap filters first; QE (GPU) only on purity survivors; MBR
   only among those that passed both. Cost: a candidate that QE would have rated best but that fails
   a threshold never competes in MBR.
6. **QE thresholds are the D4 ones, reused through `QeConfig`** (`max_metricx_qe` 8.0, lower is
   better; `min_cometkiwi` 0.55), unvalidated priors. Cost: the numbers are arbitrary until real
   scores exist, and a change to the D4 defaults changes D6 too (the YAML pins them).
7. **Shard directory keyed by the job fingerprint** (input rows, job, generator config, shard size,
   seed). Cost: changing any config value regenerates everything; old shard directories stay until
   removed.
8. **Per-request seeds derived from the request content**, not from global order, so a resumed run
   asks for the same sampling seed as an uninterrupted one. Cost: servers that ignore `seed` give
   non-reproducible text; the manifest records the seed anyway.
9. **MBR utilities limited to chrF** (+ the negation wrapper). Cost: no neural MBR until the COMET
   and MetricX reference-based scorers exist.
10. **Monolingual input contract is `id`, `lang`, `text`** (extras kept), not the full "Monolingual
    row" of the master plan, because that schema has no builder yet. Cost: the full contract
    needs one adapter when a monolingual stage exists.
11. **No `--force` skip logic beyond shard reuse.** A rerun reuses valid shards and recomputes
    selection; `--force` ignores existing shards. Cost: selection is recomputed on every run (cheap
    next to generation).

## Open gates and limitations (not done by D6)

- Every client path was tested only against fakes: the real vLLM and llama.cpp servers, real NLLB and
  MADLAD CT2 models and the real prompt quality were never run. Nothing in the runbook about GPU
  memory or throughput is measured.
- Whether the generators' Mongolian is acceptable is unjudged (no Mongolian reader).
- COMET and MetricX utilities for MBR; training-time `<bt>`/`<ft>` insertion (D7/D8).
- The QE floor and the purity share are priors.
