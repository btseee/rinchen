# D6 Synthetic Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rinchen.mbr` and `rinchen.synth`: prompts, an OpenAI-compatible client and a llama.cpp client, a CTranslate2 baseline translator, deterministic sharding with crash-safe resume, purity/QE/MBR selection, bt and ft job runners, and `rinchen synth run`, all tested against fakes.

**Architecture:** A `Translator` protocol is the seam; clients implement it; `shards` plans and stores generation work; `select` filters and ranks candidates and builds `CORPUS_SCHEMA` rows; `run` composes them; the CLI writes the manifest last.

**Tech Stack:** Python 3.14.10, httpx (MockTransport fakes), polars, numpy, sacrebleu (chrF), ctranslate2 + sentencepiece (stubbed in tests), pydantic v2, typer, pytest + hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-08-d6-synthetic-data-design.md` (binding; rulings 1-11). Master plan: `docs/research/2026-10-29-rinchen-dev-agent-prompt.md`, sections D6 and MBR.

## Global Constraints

- CPython 3.14.x only (`uv run`); every script entrypoint asserts `sys.version_info[:2] == (3, 14)`. No new dependency.
- External models are inference-only generators; never fine-tune, LoRA or continue-train anything. Never network access in unit tests: fakes only (`httpx.MockTransport`, stub translators, fake scorers). No real server, model or GPU is used by any test.
- Never Inner Mongolian data/varieties/tools, never galig/Latin informal Mongolian, nothing about licensing or legal topics (code, docs, comments, commit messages). Prompts never mention traditional script or Inner Mongolia.
- Tests before code; numeric and exact gates are gates (never loosen). Never weaken, skip or delete a failing test.
- Deterministic seeds everywhere; never silently drop text: every dropped candidate and source is counted under a reason code.
- Output Cyrillic Mongolian by default: the Cyrillic instruction is present exactly when the target is `khk_Cyrl`.
- Nothing written under `data/` or `models/` by tests (`tmp_path`). Fixtures at most 2 MB in total (generate data in tests, do not commit big files).
- Commit subjects start with `D6: `; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Gate: `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -m "not network" -q` (never pass `-W error`: it crashes the SentencePiece import). No file mode changes. Match local style (see `src/rinchen/mine/`, `src/rinchen/filter/qe.py`, `src/rinchen/cli/filter.py`, `src/rinchen/script_conv/`).
- Invisible, control, look-alike and traditional-script characters in `.py` files only through `chr(0x....)` (the Write and Edit tools turn backslash-u escapes into raw characters); visible Cyrillic letters are fine.
- Work on branch `d6-synthetic-data`; do not merge, push or create branches.

## Review Focus

Inputs the spec implies but no happy path exercises, most likely to bite first:

1. A generator returning fewer candidates than `n`, an empty list, empty strings, duplicates, or only candidates that fail the purity or QE filter (Tasks 4, 7, 8).
2. A QE scorer returning NaN or the wrong number of scores; a scorer that raises mid-run (Task 7).
3. A crash after a shard file was renamed into place but before the next shard; a truncated or corrupt shard file; two runs into one out dir (Task 6, 9).
4. Input rows with duplicate ids, empty text, a very long text, text that is already the target script, non-Cyrillic input for a `khk_Cyrl` job, or a row whose synthetic side equals its real side (Tasks 6, 7).
5. Server behaviour: 429 with a long backoff, a 200 with an error body, malformed JSON, a response whose `n` choices differ from the request, a hung request (timeout), concurrency greater than the number of texts (Task 4).

---

### Task 1: MBR

**Files:**
- Create: `src/rinchen/mbr/__init__.py` (empty), `src/rinchen/mbr/utility.py`, `src/rinchen/mbr/mbr.py`
- Test: `tests/unit/test_mbr_utility.py`, `tests/unit/test_mbr_select.py`, `tests/property/test_mbr_select_property.py`

**Interfaces:**
- Produces: `Utility` (Protocol, `score(self, hyp: str, ref: str) -> float`), `ChrfUtility`, `NegatedUtility(inner: Utility)`, `select(cands: Sequence[Sequence[str]], utility: Utility, pseudo_refs: Sequence[Sequence[str]] | None = None) -> list[int]`.
- Consumes: `rinchen.metrics.bleu_chrf.chrf(hyp, ref)`.

- [ ] **Step 1: Write failing tests.** `ChrfUtility().score("abc", "abc") == 100.0`; `score("abc", "xyz") == 0.0`; symmetric inputs differ (chrF is not symmetric: pin one known pair against `chrf`). `NegatedUtility(inner).score == -inner.score` for a stub. `select`: planted set `["the cat sat", "the cat sat down", "a dog ran"]` with the first two near each other: the winner is index 0 or 1 by hand-computed expected utility (compute the expected values in the test with `chrf` directly and assert the argmax); ties go to the lowest index (a stub utility returning a constant); one candidate returns `[0]`; several segments return one index each in order; explicit `pseudo_refs` change the winner exactly as hand-computed; with `pseudo_refs=None` a candidate is not scored against itself (a stub utility that raises when `hyp is ref` of the same index proves it: use a stub that records pairs and assert no (c, c) self-pair for equal indexes, while duplicate strings at different indexes ARE compared); `NegatedUtility` inverts the order; `ValueError` for an empty candidate list (message names the segment index), for `len(pseudo_refs) != len(cands)`, and for an empty reference list on a segment with more than one candidate. Empty input `select([], utility)` returns `[]`. Property: `select` returns an index in range for every segment and the same result when called twice.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Expected utility = mean over references; when references are the candidates, skip the same index. Plain loops over small `n` (no vectorised cleverness): clarity over speed, documented O(n^2) per segment.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: minimum Bayes risk selection`).

### Task 2: Config and tags

**Files:**
- Create: `src/rinchen/config/synth.py`, `configs/data/synth.yaml`, `src/rinchen/synth/tags.py`
- Test: `tests/unit/test_config_synth.py`, `tests/unit/test_synth_tags.py`

**Interfaces:**
- Produces: `SynthConfig`, `GeneratorConfig` (discriminated union `VllmConfig | LlamaCppConfig | Ct2Config`), `JobConfig`, `SelectionConfig`, `load_synth_config(path=SYNTH_CONFIG_PATH) -> SynthConfig`, `SYNTH_CONFIG_PATH`; `origin_for(kind, generator_kind) -> str`, `data_origin_tag(origin) -> str | None`.
- Consumes: `rinchen.config.languages.load_languages` for registry codes, `rinchen.errors.ConfigError`, the style of `rinchen.config.filters`.

- [ ] **Step 1: Write failing tests.** Fields, defaults and bounds exactly as in the spec ("config" contract); `configs/data/synth.yaml` loads to equal `SynthConfig()`; every violation raises `ConfigError` (unknown key, negative `n`, `temperature` outside [0, 2], `top_p` outside (0, 1], `shard_size` < 1, `min_script_share` outside (0, 1], `selection.qe` is the D4 `QeConfig` reused unchanged (an invalid QE field is rejected exactly as D4 rejects it, a QE key outside `QeConfig` is an unknown key), a job whose languages are not registry codes, a job where neither language is `khk_Cyrl`, a job naming an unknown generator, a `ct2` generator without `model_dir`, an unknown `kind`, an unknown `family`); a quoted number or boolean is a typo, not coerced (strict types, as `config/spm.py`); a missing or unparseable file raises `ConfigError`. Tags: `origin_for` exact table (`bt` with every generator kind gives `bt`; `ft` with `vllm` or `llamacpp` gives `synth_llm`; `ft` with `ct2` gives `synth_nmt`; unknown kind or generator kind raises `ConfigError`); `data_origin_tag` exact table (`bt` -> `<bt>`; `ft`, `synth_llm`, `synth_nmt` -> `<ft>`; `human`, `web`, `mined`, `converted` -> None; unknown origin raises). Every origin returned by `origin_for` is a member of `rinchen.corpus.schema.ORIGINS`, and `<bt>`/`<ft>` are members of `rinchen.spm.symbols.STRUCTURAL_SYMBOLS`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** The shipped YAML defines one generator of each kind and one `bt` and one `ft` job (`bt_eng_khk`, `ft_khk_eng`) with comments saying every number is an unvalidated prior and that `model` and paths are placeholders for the operator.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: synthetic-data config and origin tags`).

### Task 3: Translator protocol and prompts

**Files:**
- Create: `src/rinchen/synth/translator.py`, `src/rinchen/synth/prompts.py`
- Test: `tests/unit/test_synth_prompts.py`

**Interfaces:**
- Produces: `Translator` (Protocol), `render_prompt(src_lang, tgt_lang, text) -> str`, `render_messages(src_lang, tgt_lang, text) -> list[dict[str, str]]`, `SUPPORTED_DIRECTIONS: frozenset[tuple[str, str]]`.
- Consumes: `rinchen.config.languages.load_languages` (the eight partner codes), `rinchen.errors.ConfigError`.

- [ ] **Step 1: Write failing tests.** Every direction between `khk_Cyrl` and each partner code of the registry renders in both orders (16 directions); the text appears verbatim in the prompt; `render_messages` returns exactly one `{"role": "user", "content": ...}` message equal to the prompt; the sentence "Write the translation in Mongolian Cyrillic script only." is present exactly when the target is `khk_Cyrl`; no rendered prompt contains `Mongolian script` without the word Cyrillic, "traditional", "Inner" or "vertical"; the language names are the English names of the registry (assert the table against `configs/languages.yaml` display names if it has them, otherwise against literals); a direction outside the supported set (for example two partner languages, or an unknown code) raises `ConfigError` naming the pair; the text is not stripped or altered (leading/trailing whitespace kept inside the prompt as given); a text containing braces or `{}` format characters renders unchanged (no `str.format` injection).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Templates as one dictionary keyed by target-language family, filled with plain concatenation (no `str.format` on user text).
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: translator protocol and prompt templates`).

### Task 4: OpenAI-compatible and llama.cpp clients

**Files:**
- Create: `src/rinchen/synth/vllm_client.py`, `src/rinchen/synth/llamacpp_client.py`, `tests/support/fake_llm_server.py`
- Test: `tests/unit/test_synth_vllm_client.py`, `tests/unit/test_synth_llamacpp_client.py`, `tests/unit/test_support_fake_llm_server.py`

**Interfaces:**
- Produces: `OpenAICompatTranslator(cfg, *, client=None, sleep=time.sleep)`, `LlamaCppTranslator(cfg, *, client=None, sleep=time.sleep)`; in `fake_llm_server.py`: `FakeLlmServer(responder, *, failures=())` with `.transport` (an `httpx.MockTransport`), `.requests` (decoded request bodies in arrival order), and helpers to script a failure sequence per request number.
- Consumes: `VllmConfig`, `LlamaCppConfig`, `render_messages`, `Translator`, `UpstreamError`.

- [ ] **Step 1: Write failing tests** per the spec "Clients" and Review Focus 5: request body fields exact (`model`, `messages`, `temperature`, `top_p`, `max_tokens`, `n`, `seed`); the URL is `{base_url}/chat/completions`; seeds are deterministic across runs and differ per (text, pair, candidate count) and per candidate for llama.cpp (`seed + k`); results keep input order under `concurrency` > 1 (the fake server answers out of order); `concurrency` greater than the number of texts; retry on 429 and 503 then success with `backoff_s * 2**attempt` sleeps recorded through the injected `sleep` (capped at 30 s); a 400 is not retried; retry exhaustion raises `UpstreamError` whose message names the status and attempt count; `httpx.TransportError` retried; a 200 with non-JSON, with an `error` body, or without `choices[*].message.content` raises `UpstreamError`; fewer choices than `n` is accepted; candidates are stripped, empty ones dropped, duplicates removed keeping order; an empty `texts` list makes no request and returns `[]`; the llama.cpp client issues exactly `n` requests per text. The fake server itself is tested (scripted failures, request capture). No real network: every client gets `client=httpx.Client(transport=fake.transport)`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** `ThreadPoolExecutor(max_workers=min(concurrency, len(texts)))`; one shared retry helper used by both clients (put it in `vllm_client.py` and import it, or in a small private module: choose the smaller diff and say why); `timeout_s` passed to every request.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: vLLM and llama.cpp translator clients`).

### Task 5: CTranslate2 baselines

**Files:**
- Create: `src/rinchen/synth/ct2_baselines.py`
- Test: `tests/unit/test_synth_ct2_baselines.py`

**Interfaces:**
- Produces: `CT2Translator(cfg: Ct2Config, *, translator=None, processor=None)` implementing `Translator`; `NLLB_CODES`, `MADLAD_CODES`.
- Consumes: `Ct2Config`, `rinchen.script_conv.char_model.load_char_processor` as the model for a guarded SentencePiece load (reuse it for the real load path; tests inject a stub processor), `ctranslate2`.

- [ ] **Step 1: Write failing tests** with stub translator and processor (stubs record calls; the stub translator returns objects with `.hypotheses` as `ctranslate2` does: copy the shape from `tests/support/stub_runtime.py`): NLLB source tokens are `[src_code, *pieces, "</s>"]` and `target_prefix` is `[[tgt_code]]` per text; the target-language token is stripped from decoded hypotheses; MADLAD source tokens are `["<2mn>", *pieces]` for a Mongolian target, `<2en>` for English, and the other six codes per the spec; no `target_prefix` for MADLAD; `beam_size`, `num_hypotheses=n`, `max_batch_size=batch_size` passed; n-best order preserved; one list per text in input order; an empty text list makes no call; a language outside the family's table raises `ConfigError` naming it; missing model dir or SentencePiece file raises `DataError` (via the guarded loader); duplicates and empty hypotheses are dropped like the LLM clients.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Lazy import of `ctranslate2` inside the loader so importing the module is cheap.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: CTranslate2 NLLB and MADLAD baseline translator`).

### Task 6: Shards

**Files:**
- Create: `src/rinchen/synth/shards.py`
- Test: `tests/unit/test_synth_shards.py`, `tests/property/test_synth_shards_property.py`

**Interfaces:**
- Produces: `read_monolingual(path, lang=None) -> pl.DataFrame`, `Shard(index, ids, texts)` (frozen), `plan_shards(frame, shard_size) -> tuple[Shard, ...]`, `job_fingerprint(frame, job, generator, shard_size, seed) -> str`, `ShardStore(directory)` with `write`, `read`, `completed`.
- Consumes: polars, `rinchen.utils.part_file.unique_part_path`, `JobConfig`, `GeneratorConfig`, `DataError`.

- [ ] **Step 1: Write failing tests** per the spec "shards" contract and Review Focus 3 and 4: `read_monolingual` rejects a missing column, a non-string column, empty or whitespace-only text, duplicate ids, an empty file, with `DataError` naming the problem; keeps extra columns; filters by `lang`; sorts by `id`. `plan_shards`: contiguous id-sorted slices, last shard shorter, `shard_size` larger than the frame gives one shard, an empty frame gives `()`; property: concatenating all shards reproduces the sorted input exactly and the same input and size give identical shards. `job_fingerprint` changes when any one of the input text, an id, the job kind, the generator config (any field), `shard_size`, `seed` changes and is stable otherwise (including across dict key order). `ShardStore.write` then `read` round-trips ids and candidate lists (including empty lists and strings with NNBSP, newlines and non-BMP characters); the write goes through a `.part` file and rename (a monkeypatched failing replace leaves no final file and no `.part`); `read` of a file with the wrong columns, a wrong row count against the planned ids, or invalid Parquet raises `DataError` and `completed` treats it as not completed; `completed` lists only valid shards; two writers for the same index do not corrupt each other (last rename wins, file valid).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: deterministic shards with crash-safe resume`).

### Task 7: Selection

**Files:**
- Create: `src/rinchen/synth/select.py`
- Test: `tests/unit/test_synth_select.py`

**Interfaces:**
- Produces: `SourceRows` (ids, texts, real-side language), `Selection` (selected index per source or None, QE score, purity, counts per reason, `qe_status`), `select_candidates(...)`, `build_rows(...) -> pl.DataFrame`.
- Consumes: `mbr.select`, `ChrfUtility`, `rinchen.lang.scripts.script_of`, the language registry, `QEScorer`, `rinchen.corpus.schema` (`CORPUS_SCHEMA`, the row-id helper and frame constructor: read the module for the real names), `tags.origin_for`, `SelectionConfig`.

- [ ] **Step 1: Write failing tests** per the spec "select" contract and Review Focus 1 and 2: a planted candidate set per source (empty string, a Latin/Greek output for a `khk_Cyrl` target rejected by purity, a Cyrillic candidate whose fake MetricX-QE score is above `max_metricx_qe` (lower is better: so a HIGH value is the bad one), one with NaN, one exactly at the threshold that passes, and survivors; also the `cometkiwi` direction (below `min_cometkiwi` fails)) gives exact counts per reason; the winner among survivors is the planted MBR best; a single survivor wins without scoring; no survivor gives no row and counts `no_survivor`; with no scorer the QE step is skipped, `qe_status` says so and survivors go to MBR; a scorer returning the wrong number of scores raises `UpstreamError` (as the D4 QE stage does; no silent truncation) and one that raises propagates as `UpstreamError`; the scorer is called only on purity survivors, in batches of `qe_batch_size`, with (source, candidate) pairs in the direction the model reads them; purity is computed on the synthetic side with the registry scripts of its language; `build_rows` for `bt` puts the synthetic text in `src` and the real text in `tgt` with the languages as in the spec, for `ft` the other way round, `origin` and `corpus` as ruled, `scores.script_src` or `scores.script_tgt` filled and the other null, `scores.metricx_qe` (or `scores.cometkiwi` for that scorer) filled only when a scorer ran, `split` `train`, `created` set, the frame validates against `CORPUS_SCHEMA`, ids are the corpus row-id of (src, tgt, languages); a pair with identical sides is dropped as `copy`; row order follows source order; deterministic.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Reason codes are module constants; keep the three filters as small separate functions.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: candidate selection and corpus rows`).

### Task 8: Job runners

**Files:**
- Create: `src/rinchen/synth/bt.py`, `src/rinchen/synth/ft.py`, `src/rinchen/synth/run.py`, `tests/support/planted_synth.py`
- Test: `tests/unit/test_synth_run.py`, `tests/unit/test_synth_bt_ft.py`

**Interfaces:**
- Produces: `bt.plan(job, frame) -> Plan`, `ft.plan(job, frame) -> Plan` (translation direction: which language the generator reads and writes, which side is real), `run_job(...) -> RunResult` per the spec, `RunResult(corpus, counts, shard_stats, fingerprint)`; `planted_synth.py`: a monolingual frame builder and a scripted fake generator with planted failures.
- Consumes: Tasks 1-7.

- [ ] **Step 1: Write failing tests** per the spec "run" contract and the gates: `bt.plan` and `ft.plan` give the right directions and real sides; `run_job` with the scripted fake generator on a planted frame produces exact corpus rows and counts; shard resume after a crash: a generator that raises on shard 2 leaves shards 0 and 1 on disk and `run_job` raises `UpstreamError` naming shard 2; a second run with a working generator requests only the texts of shards 2..N (assert the generator's call log), reports `reused` and `generated` counts, and produces a corpus equal to an uninterrupted run's; a corrupt shard file is regenerated; `force=True` regenerates everything; a changed `shard_size` or generator config uses a new shard directory and leaves the old one untouched; progress callback called once per shard; the generator is never called with an empty batch; a generator returning fewer lists than texts raises `DataError`; deterministic corpus for a seed.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: back- and forward-translation job runners`).

### Task 9: CLI and the tiny synth run

**Files:**
- Modify: `src/rinchen/cli/synth.py` (replace the stub)
- Test: `tests/unit/test_cli_synth.py`, `tests/integration/test_synth_tiny_run.py`

**Interfaces:**
- Produces: `rinchen synth run JOB --input ... [--config] [--out-dir] [--seed] [--force] [--dry-run]` with module-level seams `build_translator`, `build_qe_scorer`.
- Consumes: everything above, `rinchen.cli.errors.rinchen_errors_as_exit_codes`, `rinchen.manifest.stage` (`begin_commit`, `build_inputs`, `output_file`, `tool_versions`, `code_version`, `refuse_foreign_manifest`), `rinchen.manifest.store.write`, `rinchen.corpus.writers.write_corpus`; the pattern of `src/rinchen/cli/filter.py` and `src/rinchen/cli/convert_script.py`.

- [ ] **Step 1: Write failing tests** per the spec "CLI" contract: `--help`; `--dry-run` prints the job, generator, row and shard counts and the fingerprint and writes nothing; a real run writes `corpus.parquet`, `counts.json` and the manifest last (absent while the seams run; inputs hashed before the work; outputs with sha256 and rows; seed recorded; tool versions: httpx, polars, sacrebleu, ctranslate2); a rerun reuses the shards (zero generator calls) and produces a byte-identical `corpus.parquet`; a foreign manifest is refused before any write, also under `--dry-run`; `--out-dir` equal to the input's directory exits 2; an unknown job exits 2; a missing input exits 3; generator failure exits 4 and leaves the finished shards, no manifest and no `.part` files; the QE seam returning None skips the QE step and the manifest status says so; error messages are one line. The integration test (the milestone STOP gate, `@pytest.mark.subprocess` NOT needed: it uses the CLI in-process through `CliRunner` with the fake server as the `build_translator` seam): a planted 40-row monolingual file, a scripted generator with planted failures (an empty candidate, a Latin output, a low-QE candidate, NaN QE, a copy), a fake scorer: exact counts of rows kept and dropped per reason, the corpus validates against `CORPUS_SCHEMA`, origins and `corpus` strings as ruled for one `bt` and one `ft` run, and the resume-after-crash scenario through the CLI (first run exits 4 after shard 1, second run succeeds and requests only the missing shards).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Thin commands; no selection or sharding logic in the CLI.
- [ ] **Step 4: Run the tests and the gate.**
- [ ] **Step 5: Commit** (`D6: rinchen synth run`).

### Task 10: Runbook and records

**Files:**
- Create: `docs/runbooks/06_synthetic.md`
- Modify: `STATE.md`, `DECISIONS.md` (CRLF file: append through a scratch script with `\r\n`, additions only), `CLAUDE.md` (repo layout and "Current phase"), `docs/VERIFY.md` (only if a new upstream fact was found), the D6 spec (amend to the finished code)

- [ ] **Step 1: Measure** what can be measured offline: the unit of work counts of the tiny run (shards, requests, rows kept and dropped per reason), the time of the tiny run, the MBR cost for n = 4, 8, 16 candidates on a synthetic segment (calls to chrF and seconds), and the size of a shard file. Everything in the runbook comes from the ledger, the spec or these measurements.
- [ ] **Step 2: Write `06_synthetic.md`:** what D6 builds and what it does not; commands with worked examples from the tiny run; the config reference (every key, default, and which are priors); the vLLM and llama.cpp launch commands as operator guidance clearly labelled unmeasured and unverified (model name placeholders; GPU memory and throughput are estimates to be replaced by measurements; do not state numbers you did not measure); the shard directory layout and resume semantics; the selection pipeline and reason codes with the tiny-run counts; origin and tag table; limitations and open gates (the spec's list); how to add a generator or a prompt.
- [ ] **Step 3: Update the records:** the spec amended to the finished code (rulings, deviations), STATE.md (D6 done on branch `d6-synthetic-data`, not merged, not pushed, evidence, open gates, next D7), DECISIONS.md entry `2026-11-08: D6 synthetic data - the rulings, made without a review round`, CLAUDE.md layout (`mbr/`, `synth/`, `config/synth.py`, `configs/data/synth.yaml`, `cli/synth.py`, tests support, the runbook).
- [ ] **Step 4: Run the full gate;** `rg -n -i "licen|legal"` over every D6-added file returns nothing except pre-existing rule text; `git status` shows only intended files.
- [ ] **Step 5: Commit** (`D6: runbook, spec amendments and state and decision records`).
