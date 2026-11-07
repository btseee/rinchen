# D6 - Synthetic data: design

Date: 2026-11-08. Milestone D6 of `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`
(section "D6 - Synthetic data"; builds the `mbr` module of the build-ourselves specs). Branch
`d6-synthetic-data`. The operating procedure and the measured numbers are in
`docs/runbooks/06_synthetic.md`. Updated after the build to match the finished implementation:
the contracts below are the ones the code has, and "Amendments made while building" lists what
changed against the first version and why.

## Provenance

The user's instruction for this stretch was "Proceed. I'm going to sleep now. Don't stop until you
finish everything. Commit everything." This design, its plan and every ruling below were therefore
made by the builder without a review round and without anyone to ask. Nothing here is an approved
decision; each ruling lists what it costs if wrong. The master plan is the binding authority;
where it is silent the rulings fill the gap. Nothing is merged to `main` and nothing is pushed:
the work stays on the milestone branch until the user decides.

## Intent

Turn monolingual text into synthetic parallel text with external generators used only for
inference (an LLM behind an OpenAI-compatible server, a llama.cpp server, a CTranslate2 NLLB or
MADLAD model), choose the best candidate per source with a script-purity filter, a copy filter, a
QE floor and minimum Bayes risk (MBR) selection, and write the result as corpus rows
(`CORPUS_SCHEMA`) with the right data origin. D6 also delivers the `rinchen.mbr` module the master
plan lists.

No GPU server, no real generator model and no real monolingual corpus exist in the repository, so
every network and model path is tested against fakes (`httpx.MockTransport`, a stub CTranslate2
translator, a fake QE scorer). The runbook says which paths were never run for real.

## Goals

- `rinchen.mbr`: a `Utility` protocol, a chrF utility, a negation wrapper, and
  `select(cands, utility, pseudo_refs) -> list[int]` exactly as the master plan specifies.
- `rinchen.synth`: prompts, an OpenAI-compatible client (vLLM) and a llama.cpp client, a CTranslate2
  baseline translator, deterministic sharding with crash-safe resume, selection, bt/ft job planning,
  a bounded-memory job runner and `rinchen synth run`.
- Gates (numeric or exact): the fake-server run resumes after a crash without re-requesting finished
  shards; MBR picks the planted best candidate; the purity filter rejects non-Cyrillic outputs;
  sharding is deterministic; the tiny synth run through the CLI yields exact planted counts.

## Non-goals

- Running any real generator. The vLLM and llama.cpp launch commands in the runbook are operator
  guidance, not measurements; it gives no GPU memory or throughput figure.
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
  prompts.py      per-direction prompt: three string constants, LANGUAGE_NAMES, supported_directions()
  _chat_completions.py  the shared base of both HTTP clients (requests, threads, retries, reply checks)
  vllm_client.py  OpenAICompatTranslator (n candidates per request)
  llamacpp_client.py  LlamaCppTranslator (one request per candidate)
  ct2_baselines.py    CT2Translator (NLLB and MADLAD families, n-best)
  _candidates.py  clean_candidates, has_alphanumeric, is_utf8_encodable (shared by every backend)
  _texts.py       require_encodable (shared by every backend)
  shards.py       monolingual input, deterministic sharding, fingerprint, shard store, resume
  select.py       empty -> purity -> copy -> QE -> MBR, with per-reason counts and row construction
  plan.py         Plan, plan_job: the direction of a job
  bt.py / ft.py   one-line planners: plan_job("bt" | "ft", ...)
  run.py          plan_run, run_job: per shard, generate or reuse, select; RunResult
src/rinchen/cli/synth.py           rinchen synth run
tests/support/fake_llm_server.py, planted_synth.py, synth_cli.py
docs/runbooks/06_synthetic.md
```

Layering: `mbr` depends on `metrics.bleu_chrf` only. `translator` is the leaf of `synth`; `prompts`
depends on `config.synth` (which reads the language registry and `lang.scripts`) and `errors`, not
on the translator types; the two HTTP clients implement `translator` through `_chat_completions`
and `CT2Translator` implements it directly, all three through `_candidates` and `_texts`; `shards`
on polars, `config.synth` and `_candidates`; `select` on `mbr`, `lang.scripts`, `config.filters`,
`filter.qe.QEScorer`, `corpus.schema` and `tags`; `plan` on `select` (for `Kind`) and `config.synth`;
`bt` and `ft` on `plan`; `run` composes `plan`, `bt`, `ft`, `shards` and `select`. `plan.py` is a
module of its own (it was to be in `bt.py`/`ft.py`) because `run.py` imports `bt` and `ft`. The CLI
is the only module that writes the corpus, the counts file or the manifest.

## Contracts

### `mbr`

- `Utility` (Protocol): `def score(self, hyp: str, ref: str) -> float` (higher is better).
- `ChrfUtility`: sentence-level chrF2 through `rinchen.metrics.bleu_chrf.chrf`, in [0, 100].
- `NegatedUtility(inner)`: `score = -inner.score` for metrics where lower is better (MetricX).
- `select(cands: Sequence[Sequence[str]], utility: Utility, pseudo_refs: Sequence[Sequence[str]] |
  None = None) -> list[int]`: for each segment `i` the index of the candidate with the highest
  expected utility against `pseudo_refs[i]` (or against the other candidates of `cands[i]` when
  `pseudo_refs` is None). Without `pseudo_refs` a candidate is never scored against its own index,
  but an equal string at another index is a reference like any other. Expected utility is the
  mean (`statistics.fmean`) over the references; every candidate of a segment is averaged over the
  same number of references. The first of equal best candidates wins; a segment with one candidate
  returns 0 and the utility is not called for it. Deterministic when the utility is.
  Raises, before the utility is called at all: `TypeError` for `cands` or `pseudo_refs` that is a
  string or not a `Sequence` (a generator), and for a segment of either that is a string (what a
  flat list of strings holds) or not a `Sequence`; `ValueError` for a segment with no candidates,
  for `pseudo_refs` of another length than `cands`, and for a segment of several candidates whose
  reference list is empty. Raises `ValueError` during scoring, so after earlier segments were
  scored, for a candidate whose expected utility is NaN or cannot be averaged (+inf with -inf, a
  sum beyond the float range); the message names the segment, the candidate index and the first
  60 characters of its text, and an error from the averaging is its `__cause__`. An expected
  utility of +inf or -inf is accepted and ranks as the extreme it is (ruling B). Selection calls
  `select(survivors, utility)` with `pseudo_refs=None`.

### `config` (`SynthConfig`, frozen, extra forbidden, every number an unvalidated prior)

- Numbers and integers are strict: a quoted number or a YAML boolean is an error, not a coercion;
  `nan` and `inf` are rejected.
- `generators: dict[str, GeneratorConfig]`, `GeneratorConfig = VllmConfig | LlamaCppConfig |
  Ct2Config` discriminated on `kind`; a `kind` that is missing or not one of the three is reported
  as "kind must be one of vllm, llamacpp, ct2". Common fields: `n` (candidates per source, 1 to 64,
  default 4), `temperature` (0 to 2, 0.7), `top_p` (greater than 0, at most 1, 0.95), `max_tokens`
  (greater than 0, 256). `VllmConfig`: `base_url` (default `http://127.0.0.1:8000/v1`; not blank,
  and an http or https URL with a host, which the client checks when it is built and reports as
  `ConfigError`), `model` (placeholder `Hy-MT2-7B`), `concurrency` (32), `timeout_s` (greater than
  0, 120), `retries` (0 or more, 4: retries after the first attempt), `backoff_s` (0 or more, 1.0).
  `LlamaCppConfig`: the same with default `base_url http://127.0.0.1:8080/v1` and `concurrency` 8.
  `Ct2Config`: `model_dir` and `spm_model` (both required; paths anchored at the repository root
  when relative and rejected when blank, exactly as the D4 configs do, so the generator config and
  therefore the job fingerprint depend on the checkout's location), `family` (required, `nllb` |
  `madlad`), `device` (`cpu` | `cuda` | `auto`, default `cpu`), `beam_size` (4), `batch_size` (16).
  `n` may not exceed `beam_size` (libctranslate2 raises "The number of hypotheses cannot be greater
  than beam_size * patience" on the first batch), so such a config is rejected at load. The CT2
  backend uses `n` as `num_hypotheses`, `beam_size`, `batch_size` as `max_batch_size` and
  `max_tokens` as `max_decoding_length` (the library's default of 256 would cut longer output
  silently); it ignores `temperature` and `top_p` and leaves `max_input_length` at the library's
  default. When a config supplies `generators` but not `jobs`, `jobs` is empty (the default jobs
  would otherwise name generators the user replaced); a `generators` or `jobs` table that is
  supplied replaces the default one.
- `jobs: dict[str, JobConfig]`: `kind` (`bt` | `ft`), `mono_lang`, `other_lang` (both must be
  language registry codes whose scripts do not include Mongolian script (ruling 12), they differ,
  and exactly one of the two must be `khk_Cyrl`; the error lists the supported codes, which never
  include a Mongolian-script entry), `generator` (a key of `generators`; `SynthConfig` reports the
  first job naming an unknown one), `shard_size` (greater than 0, 512). For `bt`, the monolingual
  text is the target of the pairs and the generator translates it into `other_lang` to make the
  source; for `ft`, the monolingual text is the source and the generator translates it into
  `other_lang` to make the target. The monolingual language is `mono_lang`, either side of the
  pair. `supported_partner_languages()` (the eight partners) and `MONGOLIAN` (`khk_Cyrl`) are
  public in `rinchen.config.synth`.
- `selection`: `min_script_share` (0.9, greater than 0, at most 1: the share of the synthetic
  side's letters that must belong to the registry scripts of its language); `qe`: exactly the D4
  `rinchen.config.filters.QeConfig` (`scorer` `metricx_qe` | `cometkiwi`, `max_metricx_qe` 8.0
  where LOWER is better, `min_cometkiwi` 0.55 where higher is better, `batch_size` 32), reused
  unchanged so both stages mean the same thing by "QE"; `mbr_utility` (`chrf`, the only one).
- `load_synth_config(path=SYNTH_CONFIG_PATH)` raises `ConfigError` on any violation (missing,
  unreadable or unparseable file, non-mapping top level, unknown field, out-of-range or wrongly
  typed value, a job naming an unknown generator or an unsupported language);
  `configs/data/synth.yaml` loads to exactly the `SynthConfig()` defaults.

### `tags`

`origin_for(kind: str, generator_kind: str) -> str`: `bt` jobs give origin `bt`; `ft` jobs give
`synth_llm` for the `vllm` and `llamacpp` generator kinds and `synth_nmt` for `ct2` (ruling 1); it
validates the generator kind first and raises `ConfigError` for an unknown generator kind or job
kind. `data_origin_tag(origin: str) -> str | None`: `bt` -> `<bt>`; `ft`, `synth_llm`, `synth_nmt`
-> `<ft>`; every other origin of the corpus schema -> None; `ConfigError` for an origin outside
the schema.

### `translator`

`Translator` (Protocol, not runtime-checkable): `def translate(self, texts: Sequence[str],
src_lang: str, tgt_lang: str) -> list[list[str]]`: one list of candidates per text, in input
order, best first when the backend ranks them; a candidate is a plain string; the list may be
shorter than `n` or empty (a refusal or an empty generation), never longer. The parameters are
positional-or-keyword with no defaults (a test pins it).

### `prompts`

`render_prompt(src_lang, tgt_lang, text) -> str` and `render_messages(...) -> list[dict[str, str]]`
(one user message). The prompt is three string constants and a plain concatenation, not a
per-direction table: `"Translate the following text from " + name(src) + " to " + name(tgt) + ". "`
+ `"Output only the translation, without explanations, notes or quotation marks."`, then, only when
the target is `khk_Cyrl`, `" Write the translation in Mongolian Cyrillic script only."`, then a
blank line and the text appended unchanged. `LANGUAGE_NAMES` (a read-only mapping) names the nine
languages in English, `khk_Cyrl` as "Mongolian (Khalkha, Cyrillic script)". `supported_directions()`
is a cached function (the registry is read on first use, not at import) that returns the 16 ordered
pairs between `khk_Cyrl` and each partner of `supported_partner_languages()`; a direction outside
the set raises `ConfigError` listing the partners, and a supported code without an English name
raises `ConfigError`. A prompt never mentions traditional script or Inner Mongolia. The wording is
the builder's, not a published template of any model (ruling 4).

### Clients

Both HTTP clients subclass `ChatCompletionsTranslator` in the private module
`_chat_completions.py`; they differ only in `_request_shapes()`.

- `OpenAICompatTranslator(cfg: VllmConfig, *, client: httpx.Client | None = None,
  sleep=time.sleep)`: `POST {base_url}/chat/completions` (a trailing `/` of `base_url` is
  stripped) with `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `n` and a per-request
  `seed`: the first four bytes of the SHA-256 of `"{src_lang}|{tgt_lang}|{n}|{text}"` read as a
  big-endian integer, modulo 2**31 (deterministic across runs and independent of the job's
  `--seed`). Each text is one request. Up to `concurrency` run in a `ThreadPoolExecutor`; results
  keep input order. A `client` passed in is never closed; an owned one is made with
  `httpx.Limits(max_connections=concurrency)` and closed when each call ends.
  Before any request every text is checked for UTF-8 encodability (`_texts.require_encodable`,
  after the empty-texts check): a lone surrogate is a `DataError` naming the text index, never the
  text.
- Retries and errors: `httpx.TransportError` (timeouts included), HTTP 429, 500, 502, 503 and 504
  are retried with the sleep `backoff_s * 2**min(attempt, 10)` capped at 30 s, up to `retries`
  retries; a `Retry-After` that is a whole number of seconds on a 429 or 503 replaces the backoff
  (still capped at 30 s). Exhaustion raises `UpstreamError` "`<url>` failed with `<failure>`,
  attempts: n". Another status, any other `httpx.HTTPError` or `httpx.InvalidURL` (an undecodable
  body, a redirect loop) is not retried. Non-2xx messages carry a 200-character preview of the
  response body (decoded leniently, whitespace collapsed); the client adds nothing from the request.
  Any 2xx is accepted; a body that is not JSON, not an object, has an `error` that is not null
  (`"error": null` beside valid choices is tolerated), has no `choices` list, has more choices than
  requested or has a choice without a `message` object is an `UpstreamError`; `choices: []` and a
  choice whose `message.content` is null, absent or empty (a refusal) give no candidate, not an
  error (ruling H); a `content` that is not a string is an error. `finish_reason` is ignored, so
  a generation cut at `max_tokens` is kept and the selection filters decide.
- Candidates go through `clean_candidates`: stripped with `str.strip` (which also removes narrow
  no-break spaces at the ends, harmless for Cyrillic output), dropped when they have no letter or
  digit (only whitespace, punctuation, zero-width characters or a byte order mark) or cannot be
  encoded as UTF-8 (a JSON escape such as the one for U+D800; ruling N), duplicates removed keeping
  the first. These drops are not counted anywhere.
- A failure of any request fails the whole call, and requests not yet started are cancelled, as they
  are on an interrupt (`KeyboardInterrupt`); the first failure in submission order is the one raised.
  The requests already in flight finish first (the pool is shut down with `wait=True`), each with
  its own retries. The main thread waits in slices of `_INTERRUPT_POLL_S` (0.2 s), not in one
  untimed `wait()`: once polars is imported a SIGINT does not wake an untimed wait (the handler
  polars installs restarts it), so the interrupt would be raised only after every queued request had
  been sent (`D-POLARS-SIGINT`; found by the final review, which measured exit 130 6.3 s after the
  signal with all 40 requests received; a polled wait gives 0.27 s and 16 of 40).
- `LlamaCppTranslator(cfg: LlamaCppConfig, ...)`: the same contract, but it issues `n` separate
  requests per text with `n` 1 (the server's `n` support varies by build), each with
  `seed + k` (modulo 2**31), and merges the candidates in that order.
- `CT2Translator(cfg: Ct2Config, *, translator=None, processor=None)`: both injectable for tests.
  Loading: the SentencePiece file first (`rinchen.script_conv.char_model.load_char_processor`, a
  generic guarded open: a missing file, a directory or a file that is not a model is a
  `DataError`), then the model directory (`DataError` when it is not a directory), then
  `import ctranslate2` (lazily, so importing the module never loads the library) and
  `ctranslate2.Translator(str(model_dir), device=cfg.device)`: a library `ValueError` (an
  unsupported device) is a `ConfigError`, an `OSError` or `RuntimeError` a `DataError`. Both
  families append `</s>` to the source tokens (the Hugging Face tokenizers do; ruling K): `nllb`:
  `[src_code, *pieces, "</s>"]` with FLORES codes (identical to ours) and `target_prefix=[[tgt_code]]`
  per text; the leading target code is removed from a hypothesis only when it is its first token.
  `madlad`: the tag is a user-defined SentencePiece symbol, matched only when it is encoded inside
  the text, so the source tokens are `[*encode(f"{tag} {text}"), "</s>"]` with the codes `en ru zh ko
  ja de kk tr mn` as `<2en>` ... `<2mn>`, no target prefix, nothing stripped from hypotheses, and
  the source language is validated but unused. One `translate_batch(source, [target_prefix=...,]
  beam_size=beam_size, num_hypotheses=n, max_batch_size=batch_size,
  max_decoding_length=max_tokens, return_scores=False)` call per `translate`; empty `texts` make no
  call. A language outside the family's table raises `ConfigError`, even for empty `texts`; a
  library `RuntimeError` or `ValueError` and a result count other than the text count raise
  `UpstreamError`; a result without hypotheses gives `[]` for its text. Candidates go through the
  shared `clean_candidates`. The token conventions were tested through stubs only; models must be
  converted from Hugging Face checkpoints with `ct2-transformers-converter`, because Fairseq- or
  Marian-converted models append `</s>` themselves and this client would double it.

### `shards`

- `read_monolingual(path, lang=None) -> pl.DataFrame` needs the string columns `id`, `lang`, `text`
  (any others are kept). With `lang` only the rows of that language are checked and returned (the
  others never reach a generator). The text is not normalised or stripped. Raises `DataError`
  (naming the file, a count and the first offending id, never a text) for an unreadable file, a
  missing or non-string required column, no rows (also none of the requested language), a null or
  empty id, a null, empty or whitespace-only text (Python `str.strip` semantics, as the code that
  later strips a text uses) and a duplicated id. Sorted by `id` in string order.
- `plan_shards(frame, shard_size) -> tuple[Shard, ...]`: contiguous slices of the id-sorted frame
  of `shard_size` rows (the last may be shorter); `Shard(index, ids, texts)` is a frozen dataclass of
  tuples. Deterministic: the same rows and `shard_size` give the same shards whatever order the
  frame arrives in; `ValueError` for `shard_size < 1`. Ids are assumed unique.
- `job_fingerprint(frame, job, generator, shard_size, seed) -> str`: SHA-256 hex over a header
  (canonical JSON, sorted keys, ASCII: the job config, the generator config, `shard_size`, `seed`)
  followed by one canonical JSON array per `(id, text)` row in id order, hashed row by row; no
  other column and not `lang` counts. The generator config includes the operational knobs
  (`concurrency`, `timeout_s`, `retries`, `backoff_s`, `base_url`) and, for `ct2`, the
  repository-anchored paths; it does not include the prompt wording or the weights behind a path or
  a served model name, so changing those silently reuses stale shards (run with `--force` or another
  seed). A golden digest test pins the canonical form. `shard_directory(out_dir, fingerprint)` is
  `<out_dir>/shards-<fingerprint[:12]>/`, so a changed input or config never reuses a stale shard.
- `ShardStore(directory)`: `write(index, ids, candidates)` writes `shard-<index:05d>.parquet`
  (columns `id: String`, `candidates: List(String)`, zstd) through a uniquely named `.part` file
  and an atomic rename, with no fsync; it raises `ValueError` for a caller's mistake (ids and
  candidates of different lengths, a string where a list belongs, a non-string candidate, a negative
  index) and `DataError` for a candidate UTF-8 cannot encode, before the disk is touched, naming the
  shard and the id. `write` also stores the content digest of what it wrote in the Parquet
  key-value metadata, under `rinchen.content_sha256`: the SHA-256 hex digest of the UTF-8 bytes of
  the JSON text `{"ids":[...],"candidates":[[...],...]}` (keys in that order, no whitespace,
  `ensure_ascii=False`, the ids in shard order and one candidate list per id in order; a golden
  test pins it). `read(index, expected_ids)` returns the candidate lists or raises `DataError`
  naming the file for a missing file, a file that is not Parquet, other columns or types, a row
  count or an id (in order) that is not the expected one, a null list or a null candidate, and then,
  last (it hashes every text, and the structural checks say more about what is wrong), for a
  missing content digest (a shard written before the digest existed, or by something else), an
  unreadable metadata block (`pl.read_parquet_metadata`, which reads the file named and has no
  `glob`) or a digest that differs from the one recomputed from the decoded rows. The digest is not
  part of `job_fingerprint`.
  `load(index, expected_ids)` is the one place a file is judged: `Loaded(candidates)`, `Missing()`
  (no file; a stale `.part` beside it does not count) or `Invalid(reason)` (the `DataError`
  message, which names the file and never a text), decoding the file at most once.
  `completed(shards)` and `invalid(shards)` are built on `load`. Every Parquet read passes
  `glob=False`: `pl.read_parquet` expands `[`, `*` and `?` in a path (see `D-POLARS-GLOB`). A damaged
  file can make polars panic with `PanicException`, a `BaseException`, which is caught and treated
  as an invalid file (`D-POLARS-PANIC`). Page checksums are not checked by polars, so a bit flip
  that leaves the structure intact decodes to different candidates without error (63.7% of random
  flips of a 98 KB shard; the structural checks caught 0.3%); the content digest closes that: of
  every single-bit flip of two small shards (9,888 and 26,712 flips) and 3,000 random flips of a 98
  KB one, none decoded to different rows without an error. It is not a signature (whoever can
  rewrite a shard can rewrite its digest; `corpus.parquet` and the manifest's hash are the record).
  Residual risk, accepted: a rare damaged file aborts the process (a Rust allocation failure no
  handler catches); partial writes and truncation are caught. A shard without a digest is invalid
  and regenerated.

### `select`

`select_candidates(sources: SourceRows, candidates, *, kind, synthetic_lang, cfg: SelectionConfig,
scorer: QEScorer | None) -> Selection` applies, per candidate and in this order, counting the first
rule that fails (rulings O, P, Q):

1. empty (`dropped_empty`): no letter or digit (the same predicate as the translators' cleaning);
2. script purity (`dropped_purity`): the share of the candidate's letters in the registry scripts
   of `synthetic_lang`, summed in registry order, below `min_script_share`; a text without letters
   has purity 0.0. The comparison is float32 (`np.float32(purity) >= np.float32(min_script_share)`),
   as in D4's `script_ratio`: three Han, three Hiragana, three Katakana and one Latin letter sum to
   0.8999999999999999 in float64 and must pass 0.9. Purity is a script check, not a language check
   (Kazakh or Russian Cyrillic scores 1.0 for `khk_Cyrl`); language identification is D4's `lid`
   stage's job;
3. copy (`dropped_copy`): the candidate equals the real text, both stripped, so a copy never costs a
   QE call;
4. QE (`dropped_qe`), only when a scorer is given: the score is worse than the D4 threshold of the
   configured scorer (above `qe.max_metricx_qe` for `metricx_qe`, below `qe.min_cometkiwi` for
   `cometkiwi`), compared in float32 exactly as D4 does (a score exactly at the threshold passes, a
   NaN fails, a magnitude beyond float32 is the infinity of its sign, without a warning). The
   scorer is called with `qe.batch_size` pairs per call, as `(source, translation)` in the
   orientation of the FINAL pair (ruling M): for `bt` `(candidate, real text)`, for `ft` `(real
   text, candidate)`. A wrong number of scores is an `UpstreamError`, and so is a reply that is not
   a sized iterable (`None`, a number, an iterator) and a score that is not a `numbers.Real`
   (text, even when it reads like a number, `None`, a list, bytes), naming the pair index and the type,
   never a text; NaN and the infinities are numbers and the threshold decides; an `OSError` of the
   scorer is an `UpstreamError` "the QE scorer failed: ..." (chained); any other exception of the
   scorer propagates unchanged.

Then MBR among the survivors of a source: `mbr.select(survivors, utility)` with `pseudo_refs=None`,
so each survivor is scored against the OTHER survivors (passing the survivors as explicit
pseudo-references would score every candidate against itself); a single survivor wins unscored.
`Selection(chosen, qe_scores, purity, counts, qe_status)` has one entry per source: the index of the
winner in the source's original candidate list (None without a winner), its QE score and purity.
`qe_status` is `Literal["ok", "skipped: no scorer"]`. `counts` is a read-only mapping with the nine
keys `sources`, `candidates`, `dropped_empty`, `dropped_purity`, `dropped_qe`, `dropped_copy`,
`no_candidates` (a refusal: an empty candidate list), `no_survivor` and `selected`; `candidates` and
the `dropped_*` keys count candidates, the rest count sources, and two invariants hold:
`sources == no_candidates + no_survivor + selected` and `candidates == dropped_empty +
dropped_purity + dropped_copy + dropped_qe + the survivors`. Candidates the translators already
dropped are not in `candidates`. `ValueError` for a `kind` other than bt or ft and for a candidate
list count that differs from the source count, `TypeError` for a candidate list that is a string,
`ConfigError` for a `synthetic_lang` outside the registry or an unknown `mbr_utility`.
`build_rows(sources, candidates, selection, *, kind, generator_kind, generator_name, mono_lang,
other_lang, qe_scorer: str | None, created=None) -> pl.DataFrame` makes `CORPUS_SCHEMA` rows through
`corpus_frame` (ruling A): `bt`: `src` synthetic, `tgt` real, `src_lang` the job's `other_lang`,
`tgt_lang` its `mono_lang`, `scores.script_src` the purity; `ft`: `src` real, `tgt` synthetic, the
languages the other way round, `scores.script_tgt`; the QE score goes to `scores.metricx_qe` or
`scores.cometkiwi` as `qe_scorer` names (the caller passes `cfg.qe.scorer`, or None when no scorer
ran; a mismatch with `selection.qe_status` is a `ValueError`); `origin` from `tags.origin_for`;
`corpus` = `"<kind>:<generator name>"`; `split` `train`; `id` from `make_row_id` over (src_lang,
tgt_lang, src, tgt) inside `corpus_frame`; `created` the given stamp or now. Duplicate pairs are
not removed (two sources with the same real text and winner give two rows with one id).

### `run`

- `plan_run(job, generator_name, generator_cfg, frame, *, out_dir, cfg, seed) -> RunPlan`
  (frozen: `plan: Plan`, `shards`, `fingerprint`, `shard_directory`): the checks that `job.generator`
  is `generator_name`, that it is in `cfg.generators` and that `generator_cfg` is its config
  (`ConfigError`), the job's planner, `plan_shards` and `job_fingerprint`; it creates nothing on
  disk. `run_job` and the CLI's dry run both call it, so they name the same shard directory.
- `plan_job(kind, job, frame) -> Plan` (`plan.py`; `bt.plan` and `ft.plan` are one-line wrappers):
  `Plan(kind, generator_src_lang, generator_tgt_lang, real_lang, synthetic_lang)`; in both kinds
  the generator reads `job.mono_lang` and writes `job.other_lang`, and `real_lang`/`synthetic_lang`
  name the same two languages in the other roles. `ConfigError` for a job of another kind,
  `DataError` for a frame without an `id` or `text` column, without rows, without a `lang` column
  or with rows in another language.
- `run_job(job, generator_name, generator_cfg, translator, frame, *, out_dir, cfg, scorer, seed,
  force=False, progress=None) -> RunResult` (ruling R: the fingerprint and the corpus string need
  the generator's name and config). The shards are handled one at a time in index order: a shard
  with a valid file is reused (decoded once), otherwise the translator is called with the shard's
  texts, its reply is validated and the shard is written before the next one starts; the shard's
  candidates are then selected on their own and dropped (ruling S). Selection is per source, so the
  rows and the summed counts are those of one selection over the whole frame for a scorer whose
  score depends on the (source, translation) pair alone; only the grouping of the scorer calls
  differs, and a call never spans two shards. All rows carry one `created` stamp. `progress` is
  called with `ShardProgress(index, total, reused)` once per shard, after the shard is obtained
  and before it is selected. `seed` only names the shard directory; the translators derive their
  per-request seeds themselves. With `force` the store is ignored and every shard is generated
  again.
- `RunResult(corpus, counts, qe_status, shard_stats, fingerprint, shard_directory)`; `shard_stats`
  has the keys `planned`, `reused`, `generated` and `regenerated_invalid` (the generated shards
  whose file was present but invalid; always 0 under `force`).
- Errors: a `DataError` or `ConfigError` of the translator is raised again as the same type with
  `shard i of n: ` in front; an `UpstreamError` or `OSError` becomes an `UpstreamError` "shard i of
  n failed: ..." chained to it; a reply that is not a list, has a length other than the text count
  or has an entry that is not a list of strings is a `DataError` naming the shard and nothing is
  written for it; any other exception, `KeyboardInterrupt` and `SystemExit` included, propagates
  unchanged with the finished shards on disk, and a rerun resumes. `run_job` never writes the corpus
  or a manifest.
- Memory: the peak is about the caller's frame (which `run_job` cannot free) plus the planned
  shards (every text copied into Python tuples: 544 bytes per row measured for texts of about 200
  Cyrillic characters) plus the corpus rows accumulated so far plus the candidates of one shard; the
  final `pl.concat` transiently doubles the corpus part.

### CLI (`rinchen synth run`)

`rinchen synth run JOB --input FILE.parquet [--config configs/data/synth.yaml] [--out-dir DIR]
[--seed 0] [--force] [--dry-run] [--no-qe]`. `JOB` names a job of the config. Default out dir
`data/synth/<job>`. Seams (module-level, looked up at call time, replaced by tests):
`build_translator(generator_cfg)` (a `match` on the config class) and `build_qe_scorer(selection_cfg)`
(`EoleMetricXQEScorer` for `metricx_qe`, None for `cometkiwi`).

Flow: hash the config file (a missing one is a `ConfigError`, an unreadable one a `DataError`
naming the path) and load it; find the job (an unknown one is a `ConfigError` listing the configured
jobs); refuse a foreign manifest in the out dir (`DataError`, also under `--dry-run`); refuse an
out dir that is the input file's directory, resolved, for the file as named and for the file it
links to (`ConfigError`); hash the monolingual file (a missing one is a `DataError`) and read it
with `read_monolingual(lang=job.mono_lang)`; on `--dry-run` print nine lines and stop; create the
out dir; build the translator; build the scorer (`--no-qe` builds none) and ask `qe.skip_reason`
(the scorer is skipped with a recorded reason when it needs CUDA and none is available); `run_job`
with progress lines on standard error (`shard 1/5 generated`, `reused` for a reused shard); then the
commit: `begin_commit` removes the old manifest, `write_corpus` writes `corpus.parquet`,
`counts.json` is written through a `.part` file and a rename, the manifest is written last; the
scorer is released afterwards when it is `Releasable`, also after a failure. Each input is hashed
before it is read so the manifest vouches for the bytes the run started from;
`begin_commit` goes right before the first OUTPUT write, not before the generation,
because shards are resumable work files and not manifest outputs, so a failed generation must not
erase an earlier manifest (ruling V). An `OSError` creating the out dir or in the commit exits 1 with
one line "cannot write the synth outputs under DIR: ..."; one raised by a translator or the scorer
is an `UpstreamError` (exit 4) before it gets there. `write_corpus` reads its output back with
`glob=False`, so an out dir whose name holds `[`, `*` or `?` works.

The dry run prints `job`, `kind`, `generator` (name and kind), `input rows`, `shards` (count and
size), `fingerprint`, `shard directory`, `valid shards present` (the shard files that pass
validation) and `out dir`; it builds no translator or scorer and writes nothing. A run prints
`corpus:`, `counts:` and `manifest:` paths and `rows N (selected S of T sources, qe <status>)`.
The QE status is `ok`, `skipped: --no-qe`, `skipped: no scorer` or `skipped: <skip_reason>` (for
example `no CUDA device`); it is on the last line, in the manifest's `status` (`{"qe": ...}`) and in
`counts.json`. `counts.json` (hashed as an output, sorted keys) has `fingerprint`, `job`,
`generator`, `counts`, `shard_stats` and `qe_status`. The manifest (stage `synth`) has `inputs`
(monolingual file, config file), `outputs` (`corpus.parquet` with its row count, `counts.json`),
`config_sha256` (the config file's hash), `seed`, `metrics` with `count.<key>` and `shards.<key>`
entries, `tool_versions` (`httpx`, `polars`, `sacrebleu`, `ctranslate2`, `sentencepiece`, `eole`,
`torch`) and `status`. Exit codes through `rinchen.cli.errors` (ConfigError 2, DataError 3,
UpstreamError 4, other `RinchenError` and output `OSError` 1; an interrupt 130). A rerun reuses valid
shards and produces a corpus identical to the first run's apart from `created` (ruling V).

## Testing strategy and gates

Tests first. Fakes only: no network, no model, no GPU.

- MBR: a planted candidate set where the best candidate is the consensus is chosen; ties go to the
  lowest index (also in the order-dependent summation case); one candidate returns 0 without scoring;
  `pseudo_refs` given explicitly changes the winner exactly as computed by hand; `NegatedUtility`
  flips an order; determinism; every `ValueError` and `TypeError` case, NaN and non-finite utilities;
  `hypothesis`: the index is always in range and equal for permutation-invariant identical inputs.
- Config: defaults equal the YAML; every violation rejected; a job whose pair lacks `khk_Cyrl`
  rejected; a job naming an unknown generator rejected; `mn_Mong` rejected by the registry's scripts,
  not by code; the `Ct2Config` constraints.
- Tags: the origin table exactly; the tag table exactly.
- Prompts: every direction renders to the literal prompt; the Cyrillic instruction is present
  exactly when the target is `khk_Cyrl`; no template mentions traditional script or Inner Mongolia;
  awkward texts (empty, braces, CRLF, NUL, decomposed Cyrillic) pass through unchanged; an
  unsupported direction raises; importing the module reads no registry.
- Clients (`httpx.MockTransport` fake server in `tests/support/fake_llm_server.py`): request body
  fields exact; `n` and seed handling; order preserved under concurrency; retry on 429/503 then
  success; `Retry-After`; retry exhaustion raises `UpstreamError` with the attempt count; a 400 is not
  retried; malformed bodies, refusals and `choices: []`; the interrupt cancels queued requests (one
  test patches `wait` to raise, one sends a real SIGINT with polars imported and fails by assertion
  with an untimed wait);
  backoff sleeps recorded through the injected `sleep`; the llama.cpp client issues `n` requests
  with `seed + k`.
- CT2: family token conventions asserted with literals through stubs; language table errors; the
  `ctranslate2` module is never loaded by a test.
- Shards: deterministic plan (property: concatenating shards reproduces the input; same input and
  size, same shards); fingerprint changes with each ingredient and a golden digest; store atomicity
  (a failed write leaves no final file); a corrupt shard is regenerated; a `[1]` directory name; the
  content digest (a golden value for awkward text, every part of the content tells shards apart, changed
  text with the digest kept, no digest, an unreadable metadata block, the structural errors first, a
  deterministic bit-flip sweep through `load` where no flip yields different rows without an error, a
  `run_job` resume that regenerates a shard whose text changed on disk); shard
  resume after a crash: a fake generator that fails on shard 2 leaves shards 0-1 on disk, a second
  run with a working generator requests only shards 2..N and the final corpus equals an
  uninterrupted run's.
- Select: exact reason counts on a planted candidate set (empty, non-Cyrillic output rejected by
  purity, low-QE, NaN QE, survivors) with a fake scorer; scorer contract violations (text, None,
  lists, bytes, a reply that is not a list, the pair index across chunks, an `OSError`); boundary
  cases in float32; MBR picks the
  planted best among survivors; the QE-skipped status without a scorer; copies dropped as `copy`.
- Run and CLI: the tiny synth run (the STOP gate): a planted monolingual file of 40 rows, the fake
  server and the scripted fakes, the fake scorer: exact counts of rows kept and dropped per reason
  (sources 40, candidates 60, empty 5, purity 10, QE 10, copy 0, no candidates 5, no survivor 20,
  selected 15; without QE: QE 0, no survivor 10, selected 25), the corpus validates against `CORPUS_SCHEMA`,
  origins and corpora as ruled, manifest written last, rerun resumes (zero generator calls) and
  produces an identical corpus apart from `created`; per-shard selection equals one global
  selection for several shard sizes; `--dry-run` writes nothing; error exit codes; foreign-manifest
  refusal; the interrupt exit.

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
4. **Prompt wording is the builder's**, not a published Hy-MT prompt: the master plan names the model
   but not its template and no source for it is available offline. The wording is three string
   constants in `prompts.py`, joined by plain concatenation; they state the task plainly and carry
   the Cyrillic instruction. Cost: translation quality of a real run may differ from the model's
   recommended prompt; the constants are the three lines to replace, and the shard fingerprint does
   not notice the change.
5. **Selection order empty, purity, copy, QE, MBR.** Cheap filters first; QE (GPU) only on survivors;
   MBR only among those that passed all of them. Cost: a candidate that QE would have rated best but
   that fails a threshold never competes in MBR.
6. **QE thresholds are the D4 ones, reused through `QeConfig`** (`max_metricx_qe` 8.0, lower is
   better; `min_cometkiwi` 0.55), unvalidated priors. Cost: the numbers are arbitrary until real
   scores exist, and a change to the D4 defaults changes D6 too (the YAML pins them).
7. **Shard directory keyed by the job fingerprint** (input rows, job, generator config, shard size,
   seed). Cost: changing any config value regenerates everything; old shard directories stay until
   removed; the prompt wording and the weights behind a path are not in the key.
8. **Per-request seeds derived from the request content**, not from global order, so a resumed run
   asks for the same sampling seed as an uninterrupted one. Cost: servers that ignore `seed` give
   non-reproducible text; the manifest records the job's `--seed`, which only names the shard
   directory and does not reach the server.
9. **MBR utilities limited to chrF** (+ the negation wrapper). Cost: no neural MBR until the COMET
   and MetricX reference-based scorers exist.
10. **Monolingual input contract is `id`, `lang`, `text`** (extras kept), not the full "Monolingual
    row" of the master plan, because that schema has no builder yet. Cost: the full contract
    needs one adapter when a monolingual stage exists.
11. **No `--force` skip logic beyond shard reuse.** A rerun reuses valid shards and recomputes
    selection; `--force` ignores existing shards. Cost: selection is recomputed on every run (cheap
    next to generation).
12. **A job never involves traditional script.** `JobConfig` rejects any registry language whose
    scripts include Mongolian script on either side: a generator produces Cyrillic Mongolian and the
    D5 converter owns traditional script, and prompts never mention it. Cost: no direct synthetic
    `mn_Mong` data; it has to come from the converter, which is a scaffold.

### Amendments made while building

The controller of the build made these rulings without a review round; each is one line here and a
decision with its cost in `DECISIONS.md`.

- A. The selection reads the real names of the corpus row-id helper and frame constructor from
  `rinchen.corpus.schema` (`make_row_id`, `corpus_frame`); no new schema.
- B. A NaN expected utility (or one that cannot be averaged) is a `ValueError` naming the segment
  and candidate, raised during scoring; strings and non-sequences are a `TypeError`.
- C. `selection.qe` is D4's `QeConfig` unchanged; MetricX-QE is lower-is-better (`max_metricx_qe`),
  CometKiwi higher-is-better; a wrong score count is an `UpstreamError`. This corrects the first
  draft, which had one `qe_min_score` of -1.0 with "higher is better".
- D. `JobConfig` rejects any registry language whose scripts include Mongolian script (ruling 12).
- E. `Ct2Config.model_dir` and `spm_model` are repository-anchored, blank-rejected paths like the D4
  configs.
- F. `Ct2Config` rejects `n > beam_size`.
- G. A config that supplies `generators` and no `jobs` gets no jobs.
- H. A choice with a null, absent or empty `message.content` and `choices: []` give no candidate;
  they are not errors.
- I. The backoff exponent is capped (`2**min(attempt, 10)`) so a large `retries` cannot overflow.
- J. Non-2xx errors carry a 200-character preview of the response body.
- K. Both CT2 families append `</s>` to the source tokens; the MADLAD tag is encoded with the text.
- L. The CT2 backend passes `max_tokens` as `max_decoding_length`; it ignores only `temperature` and
  `top_p`.
- M. QE pairs follow the final pair orientation: `bt` (candidate, real text), `ft` (real text,
  candidate).
- N. A candidate that UTF-8 cannot encode is dropped by `clean_candidates`, and `ShardStore.write`
  refuses one with a `DataError` as defence in depth.
- O. The pipeline order is empty, purity, copy, QE: a copy is dropped before it costs a QE call.
- P. QE thresholds and the purity share are compared in float32, as D4 does.
- Q. The copy rule compares the candidate with the real text, both stripped.
- R. `run_job` takes `generator_name` and `generator_cfg` as well (the fingerprint and the corpus
  string need them).
- S. Selection runs per shard and the counts are summed, so memory is bounded by one shard plus the
  accumulated corpus rows; the result equals one global selection for scorers that depend on the
  pair alone.
- T. `job.generator != generator_name` is a `ConfigError`.
- U. One decode per shard per run: `ShardStore.load` returns `Loaded | Missing | Invalid`;
  `completed` and `invalid` are built on it; a freshly generated shard is used from memory.
- V. `created` is the stamp of the run, so a rerun's corpus is identical apart from `created` (the
  plan said byte-identical); `begin_commit` goes right before the first output write; the manifest
  `status` carries the QE status; `counts.json` (hashed as an output) carries the fingerprint, job,
  generator, counts and shard statistics; the scorer is skipped with a recorded reason when it needs
  CUDA and none is available, or with `--no-qe`.

Changes from the review fixes that are not lettered rulings, all described in the contracts above:
`select(survivors, utility)` is called with `pseudo_refs=None`; `SUPPORTED_DIRECTIONS` became the
lazy, cached `supported_directions()`, and the Mongolian-script rule is one public helper in
`config.synth`; both HTTP clients share `_chat_completions.py`, cancel queued requests on an
interrupt, validate `base_url`, wrap raw httpx errors, tolerate `"error": null`, honour
`Retry-After` and drop candidates without a letter or digit; the encodability check is shared
(`_texts.require_encodable`) and runs in `CT2Translator` too; every Parquet read uses `glob=False`
(`corpus/readers.py` and the read-back of `corpus/writers.py` too) and a polars `PanicException` is
handled; `plan.py`, `plan_run` and
`RunPlan` exist; the translator reply and error taxonomy of `run_job`; `Invalid.reason`; the CLI
hashes each input before reading it, creates the out dir before the work, records `eole` and
`torch` in `tool_versions`, prefixes metric keys with `count.` and `shards.` and releases the
scorer. The final review's fix wave added the polled wait of the HTTP clients, `glob=False` on the
`write_corpus` read-back, the per-shard content digest, the scorer return-type and `OSError`
errors, and `tests/support/shard_files.py`.

## Open gates and limitations (not done by D6)

- Every client path was tested only against fakes: the real vLLM and llama.cpp servers, real NLLB and
  MADLAD CT2 models, their token conventions and the real prompt quality were never run. Nothing in
  the runbook about GPU memory or throughput is measured.
- Whether the generators' Mongolian is acceptable is unjudged (no Mongolian reader). Purity cannot
  tell Mongolian from other Cyrillic text.
- COMET and MetricX utilities for MBR; training-time `<bt>`/`<ft>` insertion (D7/D8).
- The QE floor, the purity share and every default are priors; the real MetricX-QE scorer was not
  run in D6.
- Concurrent runs into one out dir are not guarded (a second run rewrites the first's outputs and
  manifest); the fingerprint does not cover the prompt wording or the weights behind a path; a
  damaged shard file can still abort the process, and the content digest is not a signature; an
  `OSError` of `scorer.release()` is worded as an output write failure; the minors the final review
  left (a QE chunk loop that duplicates D4's, `plan.py` importing `Kind` from `select` and the
  small indirections around it, a `base_url` with a space inside the host, no CPython 3.14 assertion
  at the console entry except `doctor`) are listed in the runbook.
