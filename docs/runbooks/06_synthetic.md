# 06 - Synthetic data

## Scope

D6 builds the synthetic-data stage: the package `rinchen.mbr` (minimum Bayes risk
selection), the package `rinchen.synth` (prompts, three translator backends,
deterministic sharding with crash-safe resume, candidate selection, and the back- and
forward-translation job runners) and the command `rinchen synth run`. The design, with
the rulings that fill the gaps of the master plan and what each costs if wrong, is
`docs/superpowers/specs/2026-11-08-d6-synthetic-data-design.md`; the decisions are
condensed in the 2026-11-08 entry of `DECISIONS.md`.

**What D6 is.** A tested pipeline that turns a monolingual Parquet file into synthetic
parallel text: a generator (an LLM behind an OpenAI-compatible or llama.cpp server, or a
CTranslate2 NLLB or MADLAD model) writes several candidate translations of each text; a
selection step drops empty candidates, candidates in the wrong script, copies of the real
text and candidates a quality estimator rates too low, then picks one survivor per text by
minimum Bayes risk; the winners become corpus rows (`CORPUS_SCHEMA`) with the right data
origin. The generators are used for inference only: nothing here trains, fine-tunes or
adapts any model from them.

**What D6 is NOT.** Nothing below is a statement about the quality of any generator's
Mongolian.

- **No generator ever ran.** No vLLM or llama.cpp server, no NLLB or MADLAD CTranslate2
  model and no GPU was used by any test or measurement of D6. Every client path was tested
  against fakes: `httpx.MockTransport` for the two HTTP clients, stubs for CTranslate2, a
  scripted translator and fake QE scorers for the runner and the command. The launch
  commands in "Running a real generator" are operator guidance, labelled unmeasured; the
  flags beyond the few named there are unverified.
- **No real monolingual corpus exists in the repository.** The tiny run uses 40 planted
  rows, `Монгол улс 000` to `Монгол улс 039`. The prompts are the builder's wording, never
  tried on a model, and no Mongolian reader has judged any output (none exists).
- **Every threshold and default is an unvalidated prior**, and the real MetricX-QE scorer
  was not run in D6 (D4 measured it on its own data; the D6 tests use fake scorers).
- **Training-time tags are not inserted here.** D6 writes the `origin` column and ships
  `data_origin_tag`; prepending `<bt>` or `<ft>` to a source is for the D7 and D8 builders.

Every number below is from a test, the code, the design or a measurement made for this
runbook ("Measured numbers" says how each was taken); numbers that were not measured are
not given. What was checked: the unit and property tests (planted inputs with exact counts,
each new test checked by mutation by hand); the integration test
`tests/integration/test_synth_tiny_run.py` (the STOP gate: the CLI in process with a
scripted generator); one run of the command by hand for this runbook (the worked example);
and the real `rinchen` binary for `--help`, a dry run, a failing run against a closed
port and, after the final review, runs against a loopback stand-in for a server (the interrupt
and out directories whose names hold `[`, `*` or `?`).

## Architecture

| Module | Job |
|---|---|
| `rinchen/mbr/utility.py` | `Utility` protocol (`score(hyp, ref)`, higher is better), `ChrfUtility` (sentence chrF2 through `rinchen.metrics.bleu_chrf`), `NegatedUtility` for lower-is-better metrics |
| `rinchen/mbr/mbr.py` | `select(cands, utility, pseudo_refs=None)`: the index of the candidate with the highest expected utility per segment |
| `rinchen/config/synth.py` | `SynthConfig` (generators, jobs, selection), `load_synth_config`, `supported_partner_languages()`, `MONGOLIAN` |
| `configs/data/synth.yaml` | The defaults, written out; loads to exactly `SynthConfig()` |
| `rinchen/synth/tags.py` | Job kind and generator kind to corpus origin (`origin_for`); origin to training-time tag (`data_origin_tag`) |
| `rinchen/synth/translator.py` | The `Translator` protocol every backend implements |
| `rinchen/synth/prompts.py` | `render_prompt`, `render_messages`, `LANGUAGE_NAMES`, `supported_directions()` |
| `rinchen/synth/_chat_completions.py` | The shared base of both HTTP clients: requests, bounded concurrency, retries, reply checks |
| `rinchen/synth/vllm_client.py` | `OpenAICompatTranslator`: one request per text, `n` candidates per request |
| `rinchen/synth/llamacpp_client.py` | `LlamaCppTranslator`: `n` requests of one candidate per text |
| `rinchen/synth/ct2_baselines.py` | `CT2Translator`: NLLB and MADLAD, n-best by beam search, in process |
| `rinchen/synth/_candidates.py`, `_texts.py` | What every backend does to the candidates it returns and checks about the texts it is given |
| `rinchen/synth/shards.py` | `read_monolingual`, `plan_shards`, `job_fingerprint`, `ShardStore` (crash-safe, validated shard files) |
| `rinchen/synth/select.py` | `select_candidates` (empty, purity, copy, QE, MBR with per-reason counts) and `build_rows` |
| `rinchen/synth/plan.py`, `bt.py`, `ft.py` | The translation direction of a job: which language the generator reads and writes, which side is real |
| `rinchen/synth/run.py` | `plan_run` and `run_job`: shard by shard, get candidates (disk or generator), select, return rows and counts |
| `rinchen/cli/synth.py` | `rinchen synth run`; the only module that writes the corpus, the counts and the manifest |

The layering: `mbr` depends on `rinchen.metrics.bleu_chrf` only. `translator` is the
leaf of `synth`. `prompts` depends on `config.synth` (which reads the language registry and
`lang.scripts`) and `errors`, not on the translator types. The three backends implement `translator`
(the two HTTP clients through `_chat_completions`); `shards` depends on polars and
`config.synth`; `select` on `mbr`, `lang.scripts`, `filter.qe.QEScorer`, `corpus.schema`
and `tags`; `plan` on `select` (for the `Kind` literal); `run` composes `shards`,
`select`, `plan`, `bt` and `ft`. `bt` and `ft` only plan: they are one-line wrappers of
`plan_job`, and `plan.py` is a module of its own because `run.py` imports `bt` and `ft`.
The CLI is the only module that writes anything besides the shard files.

## Config reference

`configs/data/synth.yaml` holds the defaults and loads to exactly the `SynthConfig()`
defaults (a test pins this; change both together). `load_synth_config` raises
`ConfigError` for a missing, unreadable or unparseable file, a non-mapping top level, an
unknown field, an out-of-range or wrongly typed value (numbers and integers are strict:
a quoted number or a YAML boolean is an error, not a coercion) and a job that names an
unknown generator. A `generators` or `jobs` table you supply replaces the default one
instead of merging into it; when `generators` is supplied and `jobs` is not, `jobs` is
empty, because the default jobs name the default generators.

Every number is an unvalidated prior until a real generator has run.

### Generators (`generators.<name>`, chosen by `kind`: `vllm`, `llamacpp`, `ct2`)

A `kind` that is missing or not one of the three is reported as "kind must be one of
vllm, llamacpp, ct2".

| Key | vllm | llamacpp | ct2 | Meaning |
|---|---|---|---|---|
| `n` | 4 | 4 | 4 | Candidates per source text, 1 to 64. For `ct2` also the number of hypotheses, and it may not exceed `beam_size` |
| `temperature` | 0.7 | 0.7 | 0.7 | 0 to 2. Ignored by `ct2` |
| `top_p` | 0.95 | 0.95 | 0.95 | Greater than 0, at most 1. Ignored by `ct2` |
| `max_tokens` | 256 | 256 | 256 | Longest output in tokens. For `ct2` it is passed as `max_decoding_length` (the library's own default of 256 would cut longer output silently) |
| `base_url` | `http://127.0.0.1:8000/v1` | `http://127.0.0.1:8080/v1` | - | Not blank; an `http` or `https` URL with a host, checked when the client is built (`ConfigError`) |
| `model` | `Hy-MT2-7B` | `Hy-MT2-7B` | - | A placeholder: the model name the server serves |
| `concurrency` | 32 | 8 | - | Requests in flight at once |
| `timeout_s` | 120 | 120 | - | Per request |
| `retries` | 4 | 4 | - | Retries after the first attempt; 0 means one attempt |
| `backoff_s` | 1.0 | 1.0 | - | Base of the exponential backoff, 0 allowed |
| `model_dir` | - | - | required | The CTranslate2 model directory |
| `spm_model` | - | - | required | The SentencePiece model file |
| `family` | - | - | required | `nllb` or `madlad` |
| `device` | - | - | `cpu` | `cpu`, `cuda` or `auto` (passed to the library as given) |
| `beam_size` | - | - | 4 | Beam search width |
| `batch_size` | - | - | 16 | `max_batch_size` of the library |

`ct2` constraints: `n` must not exceed `beam_size` (checked at load; libctranslate2
4.8.4 otherwise raises "The number of hypotheses cannot be greater than beam_size *
patience" at the first batch, which would end a long run late; see `docs/VERIFY.md`,
`D-CT2-NUMHYP`); `model_dir` and `spm_model` are paths that are anchored at the
repository root when relative and rejected when blank, exactly as the D4 configs do, so
the fingerprint changes if the checkout moves; `temperature` and `top_p` are accepted
only so that every generator has the same fields. The shipped `ct2_nllb_default` points
at `models/ct2/nllb-200-3.3B`; nothing creates it.

### Jobs (`jobs.<name>`)

| Key | Default of the two shipped jobs | Meaning |
|---|---|---|
| `kind` | `bt` (`bt_eng_khk`), `ft` (`ft_khk_eng`) | `bt`: the monolingual text is the target of the pairs and the generator writes the source. `ft`: the monolingual text is the source and the generator writes the target |
| `mono_lang` | `khk_Cyrl` | The language of the monolingual rows |
| `other_lang` | `eng_Latn` | The language the generator writes, in both kinds |
| `generator` | `vllm_default` | A key of `generators` |
| `shard_size` | 512 | Rows per shard, greater than 0 |

Both languages must be codes of `configs/languages.yaml` whose scripts do not include
Mongolian script (ruling 12: a generator writes Cyrillic Mongolian and the D5 converter
owns traditional script); they must differ and exactly one must be `khk_Cyrl`. The
error lists the supported codes: `khk_Cyrl` and the eight partners `deu_Latn`,
`eng_Latn`, `jpn_Jpan`, `kaz_Cyrl`, `kor_Hang`, `rus_Cyrl`, `tur_Latn`, `zho_Hans`.
`mono_lang` may be either side, so a job can back-translate English text into
Mongolian sources as well as the shipped Mongolian-monolingual jobs.

### Selection (`selection`)

| Key | Default | Meaning |
|---|---|---|
| `min_script_share` | 0.9 | Share of the synthetic side's letters that must belong to the registry scripts of its language, greater than 0, at most 1 |
| `qe.scorer` | `metricx_qe` | `metricx_qe` or `cometkiwi` (D4's `QeConfig`, reused unchanged) |
| `qe.max_metricx_qe` | 8.0 | The highest accepted MetricX-QE score (0 to 25, LOWER is better); a score equal to it passes |
| `qe.min_cometkiwi` | 0.55 | The lowest accepted CometKiwi score (higher is better); a score equal to it passes |
| `qe.batch_size` | 32 | Pairs per scorer call |
| `mbr_utility` | `chrf` | The only utility there is |

A `cometkiwi` scorer has no adapter yet (D4's state), so a run configured for it skips
the QE floor and says so.

## Commands

One command, `uv run rinchen synth run`. The example below was run on 2026-11-07 on the
machine of "Measured numbers", in process through the CLI with the planted fakes of the
integration test behind the two seams (the real binary has no server to talk to; its own
outputs follow). `$W` is an empty scratch directory.

The config is the one the tests write (`write_job_config` in `tests/support/synth_cli.py`):
one vllm generator with its defaults and the two shipped jobs with `shard_size: 8`, which
cuts the 40 rows into 5 shards.

### Dry run

```bash
uv run rinchen synth run bt_eng_khk --input $W/in/mono.parquet --config $W/synth.yaml \
    --out-dir $W/out --dry-run
```
```
job: bt_eng_khk
kind: bt
generator: vllm_default (vllm)
input rows: 40
shards: 5 (shard_size 8)
fingerprint: 8fde2bbdadcd029e3d7826c1ae47741c0a26967f19678e5e7255d060f8992f79
shard directory: $W/out/shards-8fde2bbdadcd
valid shards present: 0
out dir: $W/out
```

A dry run builds no translator and no scorer, writes nothing (`$W/out` still does not
exist afterwards) and generates nothing; it reads and hashes the two input files. It
refuses a foreign manifest and an `--out-dir` that is the input's directory exactly as a
real run does. `valid shards present` counts the shard files that pass validation, so
after a crash it says how many shards a rerun will reuse.

### Run

Same command without `--dry-run`:

```
corpus: $W/out/corpus.parquet
counts: $W/out/counts.json
manifest: $W/out/manifest.json
rows 15 (selected 15 of 40 sources, qe ok)
```

Standard error carries one line per shard as it is obtained, before its selection:
`shard 1/5 generated` to `shard 5/5 generated` (`reused` for a shard found on disk).
Files after the run (sizes in bytes): `corpus.parquet` 7,718, `counts.json` 492,
`manifest.json` about 1.7 KB, and `shards-8fde2bbdadcd/shard-00000.parquet` to
`shard-00004.parquet` of 1,217 to 1,225 bytes each.

`counts.json` (sorted keys) holds `counts` (the nine keys of the selection table below),
`fingerprint`, `generator`, `job`, `qe_status` and `shard_stats` (`generated` 5, `planned`
5, `regenerated_invalid` 0, `reused` 0).

The manifest (`rinchen.manifest/1`, stage `synth`) records: `inputs` (the monolingual
file and the config file with sha256 and size), `outputs` (`corpus.parquet` with its row
count and `counts.json`, with sha256 and size), `config_sha256` (the config file's hash),
`seed`, `code_version`, `metrics` with the counts as `count.<key>` and the shard
statistics as `shards.<key>` (for example `count.selected` 15.0 and `shards.reused`
0.0), `tool_versions` (`httpx` 0.28.1, `polars` 2.1.0, `sacrebleu` 2.6.0, `ctranslate2`
4.8.4, `sentencepiece` 0.2.1, `eole` 0.6.4, `torch` 2.14.1+cu130 in the example; a
package that is not installed is recorded as `not installed`) and `status`
(`{"qe": "ok"}`). The shards are resumable work files, not outputs, so the manifest does
not list them.

The first corpus row: `src_lang` `eng_Latn`, `tgt_lang` `khk_Cyrl`, `src` "The state of
Mongolia 000 stands", `tgt` "Монгол улс 000", `origin` `bt`, `corpus` `bt:vllm_default`,
`split` `train`.

`scores.script_src` holds the purity of the winner (1.0 here) and `scores.metricx_qe` its
QE score (1.0, 8.0 and 1.0 for these three rows: the second is exactly at the threshold
and passes); every other score is null, `doc_id`, `seg_idx` and `filter_reason` are
null, and one UTC `created` stamp is shared by every row of a run.

### Rerun, damaged shard, `--force`

A rerun into the same directory reuses every shard and calls the generator zero times;
standard error says `shard 1/5 reused` and so on, and `counts.json` says `"reused": 5`.
The corpus is identical to the first run's apart from `created`, which is the stamp of
the run.

With shard 1's file truncated to 50 bytes, a dry run says `valid shards present: 4` and
the rerun regenerates exactly that shard (one translator call of 8 texts; standard error
says `reused` for shards 1, 3, 4 and 5 and `generated` for shard 2), with `shard_stats`
`generated` 1, `planned` 5, `regenerated_invalid` 1, `reused` 4. `--force` ignores the
files on disk: all five shards are generated again and overwritten, and
`regenerated_invalid` stays 0 even when a damaged file was present.

### Crash and resume

A generator that fails on its third call (shard index 2), run with `--seed 7`, prints
`shard 1/5 generated`, `shard 2/5 generated` and `error: shard 2 of 5 failed: server down`,
and exits 4. The directory then holds only `shards-<fp>/shard-00000.parquet` and
`shard-00001.parquet`: no corpus, no counts, no manifest, no `.part` file. The same command
with a working generator reuses shards 1 and 2, requests only shards 3 to 5 (three calls
of 8 texts) and ends with the same rows as an uninterrupted run.

### Interrupt

Ctrl-C while an HTTP client is generating a shard cancels the requests not yet sent and
raises `KeyboardInterrupt`; the command exits 130 with the finished shards on disk, nothing
written for the interrupted shard and no manifest of that run. The requests already in
flight are not abandoned: the client's thread pool is shut down with `wait=True`, so up to
`concurrency` requests (32 by default for `vllm`, 8 for `llamacpp`) are finished first, and
a request that fails goes on with its own retries, so the worst case for each is
`retries + 1` attempts of up to `timeout_s` plus the backoff sleeps (each at most 30 s). A
second Ctrl-C during that wait is not delivered until it ends.

The main thread waits for the requests in slices of 0.2 s. Once `polars` is imported (it
always is under `rinchen`), a SIGINT does not wake an untimed `concurrent.futures.wait()`,
and the interrupt would have been raised only after every queued request had been sent;
`docs/VERIFY.md`, `D-POLARS-SIGINT`. Measured through the real binary on 2026-11-08 against
a loopback server that sleeps 1 s per request (`concurrency` 4, 40 texts in one shard,
SIGINT 4.5 s after the start): exit 130, 0.27 s after the signal, 16 of the 40 requests
received (before the fix: 6.33 s after the signal, 40 of 40). The delay depends on where the
signal falls in the requests in flight: five repeats by the re-reviewer gave 0.77 to 1.06 s, so
the bound is the time of one request, not 0.27 s. The poll costs a pass over the pending futures
every 0.2 s: 0.19 ms at the default `shard_size` of 512 and 8.3 ms at 20,000 (measured by the
re-reviewer), negligible unless `shard_size` is made very large. The CTranslate2 backend was
not examined: it runs in process and is not a request queue.

### Options

| Option | Default | Meaning |
|---|---|---|
| `JOB` (argument) | - | A job of the config; an unknown one exits 2 and lists the configured jobs |
| `--input FILE` | required | Monolingual Parquet with string columns `id`, `lang`, `text` |
| `--config FILE` | `configs/data/synth.yaml` | The synth config |
| `--out-dir DIR` | `data/synth/<JOB>` | Output directory; must not be the input file's directory |
| `--seed N` | 0 | Part of the fingerprint (see below) and recorded in the manifest |
| `--force` | off | Generate every shard again |
| `--dry-run` | off | Print the plan, write nothing |
| `--no-qe` | off | Build no scorer and skip the QE floor; the status is `skipped: --no-qe` |

The flow, in order: hash the config file, load it, find the job; refuse a foreign
manifest in the output directory; refuse an output directory that is the input's;
hash the monolingual file, read it (only the rows of `job.mono_lang` are checked and
kept); on `--dry-run` print the plan and stop; create the output directory; build the
translator and the scorer; `run_job`; remove the old manifest (`begin_commit`), write
`corpus.parquet` and `counts.json`, write the new manifest last; release the scorer.
Each input is hashed before it is read, so that the manifest vouches for the bytes the
run started from. `begin_commit` happens right before the first OUTPUT is written, not
before the generation: the shards are not outputs, so a failed generation leaves an
earlier run's manifest and outputs standing.

`--no-qe` and the QE statuses: `ok` (a scorer ran), `skipped: --no-qe`, `skipped: no
scorer` (the configured scorer has no adapter) and `skipped: no CUDA device` (MetricX-QE
needs CUDA and torch sees none). A skipped QE step keeps more candidates (the tiny run
without QE: 25 rows instead of 15). The stdout last line and `status` in the manifest
carry the status.

One output directory holds one job's outputs: a second job run into the same
`--out-dir` creates its own `shards-<fp>` directory beside the first but replaces
`corpus.parquet`, `counts.json` and the manifest. Use one directory per job; the default
`data/synth/<JOB>` does.

### Exit codes

Failures print `error: <message>` on standard error; messages are one line.

| Code | Meaning |
|---|---|
| 0 | OK, including a dry run |
| 1 | An `OSError` creating the output directory or writing an output ("cannot write the synth outputs under DIR: ..."), or any other `RinchenError` |
| 2 | `ConfigError`: unknown job, a missing, unreadable or invalid config, an `--out-dir` that is the input's directory, a `base_url` that is not an http(s) URL with a host, a language outside the CT2 family's table |
| 3 | `DataError`: a missing or unreadable input, not Parquet, a missing or non-string column, no rows of the job's language, a null or empty id, a null, empty or whitespace-only text, a duplicated id, a translator reply of the wrong shape, a text that cannot be encoded as UTF-8, a CT2 model or SentencePiece file that cannot be loaded, an output directory that holds another stage's manifest |
| 4 | `UpstreamError`: a shard's generation failed (server unreachable after the retries, a status that is not retried, a malformed reply, a CT2 failure), named as `shard i of n`; a QE scorer that returns the wrong number of scores, a score that is not a number or an `OSError` ("the QE scorer failed: ...") |
| 130 | An interrupt (Ctrl-C): the queued requests are cancelled and the finished shards stay; the requests already in flight still complete first (up to `concurrency` of them, each up to `timeout_s` plus its retries and backoff), see "Interrupt" |

### The real binary

`uv run rinchen synth run --help` prints the usage, the argument and the options above. A
dry run through the real binary on the same planted file gives the plan lines above (with
another fingerprint, because that config named `http://127.0.0.1:9/v1` with `retries: 0`).
A real run against that closed port ends with `error: shard 0 of 5 failed:
http://127.0.0.1:9/v1/chat/completions failed with ConnectError, attempts: 1`, exit 4,
2.2 s of wall time, and an empty output directory (it is made before the work). After the
final review the binary was also run against a loopback stand-in for a chat server (not a
model; 40 planted texts, one shard, `--no-qe`): `--out-dir` `out[1]`, `out*`, `out?` and
`out[a-z]` each exit 0 and write `corpus.parquet`, `counts.json` and `manifest.json`, and
the interrupt probe above.

## Shard directory, fingerprint and resume

```
<out-dir>/
  corpus.parquet        the corpus rows (CORPUS_SCHEMA)
  counts.json           fingerprint, job, generator, counts, shard_stats, qe_status
  manifest.json         stage synth, last
  shards-<fingerprint[:12]>/
    shard-00000.parquet ... shard-NNNNN.parquet   id: String, candidates: List(String)
```

A shard file holds one row per source id of the shard, in order, with the candidates the
translator returned (an empty list for a refusal) and, in its Parquet key-value metadata,
the content digest `rinchen.content_sha256` (below). Files are written through a uniquely
named `.part` file and a rename, one shard at a time, in index order, each before the
next shard is started; there is no fsync.

**The fingerprint** is the SHA-256 over: the `(id, text)` rows of the input in id order
(no other column and not `lang`), the job config, the whole generator config, the
`shard_size` and the `--seed`. A changed ingredient names another directory, so a stale
shard is never reused; the old directory stays until you remove it.

What is in it, and what follows from that:

- The generator config includes the operational knobs `concurrency`, `timeout_s`,
  `retries`, `backoff_s` and `base_url`, so changing the timeout or pointing at another
  host starts from scratch. For `ct2` it includes the repository-anchored `model_dir`
  and `spm_model` paths, so moving the checkout changes it.
- It does NOT include the prompt wording, the code version, the model weights behind a path
  or behind a served model name, or `--no-qe` and `--force`. Editing the prompt constants,
  replacing the weights in the same directory, or restarting the server with another model
  under the same `model` name silently reuses the stale shards: after such a change run
  with `--force`, another `--seed` or another `--out-dir`. The selection config
  (`min_script_share`, the QE thresholds) is not in it either, deliberately: it does not
  change what a shard holds, and a rerun with other thresholds reuses the shards and
  recomputes the selection (ruling 11).
- `--seed` only names the shard directory and is recorded in the manifest. The clients
  derive the per-request sampling seed themselves, from the text, the language pair and
  `n` (the first four bytes of the SHA-256 of `"{src}|{tgt}|{n}|{text}"` read as an
  integer, modulo 2**31; llama.cpp adds the candidate's index). A new `--seed` therefore
  regenerates every shard but asks the server for the same sampling seeds, so a server
  that honours `seed` returns the same text.

**Resume semantics.** `run_job` handles the shards in index order. A shard whose file
passes validation is reused (the file is decoded once); any other is sent to the
translator and written before the next one starts. After a crash the finished shards are
on disk and the rerun picks up after them. What a crash leaves: the finished shards;
possibly one `.part` file if the process was killed between the write and the rename
(safe to delete); no manifest of the failed run, but an earlier run's manifest and
outputs, if any, stay untouched until a later run reaches its commit.

**A damaged shard.** A file is valid when it is Parquet with exactly the columns `id` and
`candidates` of the right types, as many rows as the shard has ids, the same ids in the
same order, no null list or null candidate, and a content digest in its metadata that
matches its rows. Anything else (truncated, foreign, another shard's file, changed text, or
written before shards carried a digest) is invalid: the rerun regenerates it and counts it
under `regenerated_invalid`. The structural checks run first, because they are cheap and say
more about what is wrong; the digest, which hashes every text, runs last.

The digest is the SHA-256 hex digest of the UTF-8 bytes of the JSON text
`{"ids":[...],"candidates":[[...],...]}`: the keys in that order, no whitespace,
`ensure_ascii=False` (non-ASCII text stays as it is; JSON escapes only quotes, backslashes
and control characters), the ids in shard order and one list of candidates per id in
generation order. `ShardStore.write` stores it under the key `rinchen.content_sha256`;
`ShardStore.read` recomputes it from the decoded rows. It detects any change to an id, a
candidate, an order or a count, which the file reader does not notice: polars does not
verify Parquet page checksums, so a flipped bit often decodes to other text without an
error (before the digest, 63.7% of random single-bit flips of a 98 KB shard decoded to
different candidates, and the structural checks caught 0.3%). It is not a signature:
whoever can rewrite a shard can rewrite its digest, and the artefact of record is
`corpus.parquet`, whose SHA-256 is in the manifest. It is not part of the job fingerprint.
A shard written before the digest existed has none and is regenerated, and counted as
`regenerated_invalid`, by the first rerun. A flip that leaves the decoded rows unchanged is
not reported; it did no harm. One limit remains: a few damaged files make polars panic
(`PanicException`, a `BaseException`, which the code catches and treats as an invalid file)
or, rarer still, abort the process with a Rust allocation failure that cannot be caught. A
partial write or a truncation, the crash cases, are caught.

**Two runs into one directory** are not guarded: a second run rewrites the first's
outputs and manifest, and either manifest may then describe files the other wrote. Run
one command per output directory at a time; only sequential runs are tested. Shard
writes are safe against each other (each renames its own complete file), the outputs are
not.

## Selection

`select_candidates` applies, per candidate and in this order, the first rule that fails
counting:

| Order | Reason code | Rule |
|---|---|---|
| 1 | `dropped_empty` | The candidate has no letter or digit |
| 2 | `dropped_purity` | The share of its letters in the registry scripts of the synthetic language is below `min_script_share` (0.0 for a text without letters) |
| 3 | `dropped_copy` | The candidate equals the real text, both stripped |
| 4 | `dropped_qe` | The QE score is worse than the threshold of the configured scorer; NaN fails; exactly at the threshold passes. Only when a scorer ran |

Then, per source with several survivors, minimum Bayes risk over chrF picks the survivor
the others agree with most (`mbr.select(survivors, utility)` with `pseudo_refs=None`, so
each survivor is scored against the OTHER survivors); a lone survivor wins unscored, the
first of equal best candidates wins. A source with no survivor gets no row and is
counted. The counts have nine keys:

| Key | Counts | |
|---|---|---|
| `sources` | sources | input rows of the job's language |
| `candidates` | candidates | after the translators' own cleaning (below) |
| `dropped_empty`, `dropped_purity`, `dropped_copy`, `dropped_qe` | candidates | by the first failing rule |
| `no_candidates` | sources | the generator returned nothing (a refusal) |
| `no_survivor` | sources | candidates existed, none survived |
| `selected` | sources | a row was written |

Two invariants hold: `sources == no_candidates + no_survivor + selected`, and
`candidates == dropped_empty + dropped_purity + dropped_copy + dropped_qe + the number of
survivors`.

Things to know:

- **Purity is a script check, not a language check.** Kazakh or Russian text in Cyrillic
  scores 1.0 for `khk_Cyrl`. Telling languages apart is the D4 `lid` stage's job; D6
  never calls a language identifier. The real side is not checked at all.
- **Comparisons are float32**, as in the D4 stages. The shares of a candidate with several
  scripts are summed in float64: three Han, three Hiragana, three Katakana and one Latin
  letter sum to 0.8999999999999999, which fails `>= 0.9` in float64 and passes in float32;
  likewise a MetricX score of 7.30000019 against the threshold 7.3 passes (both round to
  the same float32), as D4's stage would have it.
- **QE orientation follows the final pair.** `bt`: `(source=candidate, translation=real
  text)`; `ft`: `(source=real text, translation=candidate)`. A scorer is called with
  `qe.batch_size` pairs per call; a wrong number of scores, a score that is not a number
  (text, even text that reads like a number, `None`, a list, bytes) or a reply that is not
  a list of scores is an `UpstreamError` naming the pair and the type, never a text; NaN
  and the infinities are numbers, and the threshold decides; an `OSError` of the scorer
  (a missing weights file, say) is an `UpstreamError` "the QE scorer failed: ...".
- **A refusal shows as `no_candidates`.** A null, absent or empty `message.content`, an
  empty `choices` list and a CT2 result without hypotheses all give an empty candidate
  list for that text.
- **What the translators drop is not counted.** Each backend strips its candidates and
  drops empty ones, ones without a letter or digit, duplicates (first kept) and ones UTF-8
  cannot encode, before `candidates` is counted. With the three shipped backends
  `dropped_empty` is therefore expected to stay 0; the tiny run's scripted generator
  returns an empty string so that the rule is exercised.
- **MBR is O(n^2)** in the candidates per source; see the measured cost below.
- **Selection runs per shard** and the counts are summed. Selection is per source, so the
  rows and counts equal those of one selection over the whole input when a scorer's
  score for a pair depends on that pair alone (tests for shard sizes 1, 3, 7, 8, 40 and
  100, and 746 random comparisons in the review); only the grouping of the scorer calls
  differs, and a scorer call never spans two shards.

### The tiny run's counts

The 40 planted rows are laid out by position modulo 8 for a job whose synthetic side is
English and whose scorer is `metricx_qe` (threshold 8.0):

| Position mod 8 | Planted generator output | Fate |
|---|---|---|
| 0 | two good candidates | MBR picks the first ("The state of Mongolia n stands") |
| 1 | no candidates (a refusal) | `no_candidates` |
| 2 | an empty candidate, then a Cyrillic one | `dropped_empty`, `dropped_purity`; `no_survivor` |
| 3 | a copy of the real (Cyrillic) text | `dropped_purity` (a Cyrillic copy fails the English purity check before the copy rule); `no_survivor` |
| 4 | one candidate the scorer rates 9.0 | `dropped_qe`; `no_survivor` |
| 5 | one candidate the scorer rates NaN | `dropped_qe`; `no_survivor` |
| 6 | one candidate the scorer rates exactly 8.0 | at the threshold: passes; selected |
| 7 | four good candidates | MBR picks the third ("Mongolia is a country n indeed") |

Measured through the CLI (bt and ft give the same counts; 60 candidates; wall time and the
rest under "Measured numbers"):

| Run | `dropped_empty` | `dropped_purity` | `dropped_copy` | `dropped_qe` | `no_candidates` | `no_survivor` | `selected` |
|---|---|---|---|---|---|---|---|
| with QE (`qe ok`) | 5 | 10 | 0 | 10 | 5 | 20 | 15 |
| `--no-qe` | 5 | 10 | 0 | 0 | 5 | 10 | 25 |

`dropped_copy` is 0 because a copy is only reachable with a real text in the synthetic
script; a unit test covers it with a mislabelled Latin real text.

## Origins and tags

| Job kind | Generator kind | `origin` | `corpus` | `data_origin_tag(origin)` |
|---|---|---|---|---|
| `bt` | any | `bt` | `bt:<generator name>` | `<bt>` |
| `ft` | `vllm`, `llamacpp` | `synth_llm` | `ft:<generator name>` | `<ft>` |
| `ft` | `ct2` | `synth_nmt` | `ft:<generator name>` | `<ft>` |
| (D10, not a job kind of `rinchen synth`) | `teacher` | `ft` | `ft:<name>` | `<ft>` |

The plain `ft` origin is forward translation by our own teacher: since D10 the generator kind `teacher`
gives it (`tags.origin_for`), and `rinchen distill generate` (runbook 10) is the only writer; D6's
configurable generators (`vllm`, `llamacpp`, `ct2`) never give it. Every other origin (`human`, `web`, `mined`, `converted`) has no tag
(`data_origin_tag` returns `None`); an origin outside the corpus schema raises
`ConfigError`. Inserting the tag into a training source belongs to D7 and D8.

## Running a real generator

Everything in this section is operator guidance. It is UNMEASURED: no generator ran for
D6. Flags not named here are UNVERIFIED. Replace the placeholders; read the real
documentation of the release you install.

### vLLM

```bash
# UNVERIFIED: not run by D6. <model> is a placeholder for the checkpoint you chose.
vllm serve <model> --host 127.0.0.1 --port 8000
```

Then `base_url: http://127.0.0.1:8000/v1` (the shipped default) and `model:` set to the
name the server reports for the model (the shipped `Hy-MT2-7B` is a placeholder). Context
length, memory fraction, a served-model name, parallelism and quantisation flags are not
verified here. What the client sends: `POST {base_url}/chat/completions` with `model`,
`messages` (one user message), `temperature`, `top_p`, `max_tokens`, `n` and a `seed`, one
request per text. A server that rejects `n` or `seed`, or answers in another shape, will
fail as an `UpstreamError` on the first shard; check one request by hand first.

### llama.cpp

```bash
# UNVERIFIED as a launch recipe: not run by D6. <model.gguf> is a placeholder.
llama-server -m <model.gguf> --host 127.0.0.1 --port 8080 -c <total context> -np <slots>
```

Confirmed on this machine: `--host`, `--port`, `-m` and `-c` are in the command lines of
the `llama-server` processes running here (`ps`), and the `--help` of that build
(version 0.9.0-dev, build 12178, commit 954de248a) lists `-c, --ctx-size`,
`-np, --parallel` ("number of server slots"; the running servers spell it `--parallel`),
`--host`, `--port`, `-m` and `--jinja` (chat template handling, default on). Not
confirmed: how the context is divided between slots, whether the server honours `n` and a
per-request `seed`, the template of any model, throughput. The client asks for the
candidates one request at a time because support for `n` varies between builds, and its
`concurrency` default is 8; relating it to the number of slots is a choice to make by
measuring. Port 8080, the default `base_url` of `llamacpp_default`, was already in use on
this machine by another service, so a server for D6 here needs another port and a
matching `base_url`.

### CTranslate2 NLLB and MADLAD

The model is converted once from a Hugging Face checkpoint, not run from the original
framework's files:

```bash
# UNVERIFIED: not run for NLLB or MADLAD. Placeholders in angle brackets.
uv run ct2-transformers-converter --model <hf checkpoint or directory> \
    --output_dir models/ct2/<name> --copy_files <sentencepiece file> --quantization <type>
```

Confirmed from the `--help` of the converter in the project environment: `--model`,
`--output_dir`, `--copy_files`, `--quantization` (with `int8`, `int8_float32`,
`int8_float16`, `int8_bfloat16`, `int16`, `float16`, `bfloat16`, `float32`), `--force`,
`--dry-run`, `--report`, `--revision`. Whether it converts these two models was not
tried, and which SentencePiece file the checkpoint holds (the shipped config assumes
`sentencepiece.bpe.model` beside an NLLB model) is unverified. The client follows the
Hugging Face tokenizer conventions: it appends `</s>` to the source tokens of both
families, names both languages as tokens for NLLB (`[src_code, *pieces, "</s>"]` with
the target code forced as a prefix) and puts the MADLAD tag `<2xx>` inside the text before
SentencePiece encodes it. Models converted from Fairseq or Marian files append `</s>`
themselves, so this client would double it: convert from the Hugging Face checkpoint. The
two conventions were tested through stubs only and are unverified against a real
converted model.

### Expectations, as questions to measure

No figures are given, because none was measured. Before trusting a run, answer:

- What is the peak GPU memory of the chosen model, at the context and slot settings you
  chose, with the QE scorer on the same GPU? (D4 measured MetricX-QE alone: a peak of
  4,935 MiB allocated on short pairs and 5,041 MiB on long ones, at batch 32, and 500 and
  226 pairs per second; see `docs/runbooks/04_filtering_mining.md`. Sharing the device with
  a generator was not tried.)
- How many texts per second does the server sustain at `concurrency` 8, 32 and 64, and at
  `n` 1, 4 and 8 for the texts you have? The MBR step below grows with the square of `n`.
- Does the server honour `seed`, and does `n > 1` give distinct candidates?
- Are the answers Cyrillic Mongolian, and not Russian, Kazakh or a refusal? The purity
  check cannot tell Mongolian from other Cyrillic text; sample the output and have a
  Mongolian reader look.
- How long does one shard (512 texts, one translator call, at most `concurrency` requests
  in flight) take, and is the default `timeout_s` of 120 enough for the slowest single
  request?

## Measured numbers

The machine: Intel Core Ultra 7 265K (20 cores as `nproc` reports), 46 GiB of RAM,
Linux 7.0.0-42, CPython 3.14.10, polars 2.1.0, httpx 0.28.1, sacrebleu 2.6.0,
ctranslate2 4.8.4. The GPU (RTX 5070 Ti) was not used by any of these runs. All dates
2026-11-07 by the machine's clock, except the interrupt probe, the shard digest and the
bit-flip sweeps with it, which are 2026-11-08.

### The tiny run

Through the CLI in process (`CliRunner`), the planted 40-row file, the scripted generator
and the planted scorer, in a temporary directory; one unreported warm-up run, then 7
runs per configuration:

| Run | Shards | Translator calls (texts per call) | Rows kept | Wall time median (min to max) | `corpus.parquet` | `counts.json` |
|---|---|---|---|---|---|---|
| bt, QE | 5 | 5 (8 each) | 15 | 42.5 ms (40.7 to 44.9) | 7,718 bytes | 492 bytes |
| bt, `--no-qe` | 5 | 5 (8 each) | 25 | 41.6 ms (39.9 to 42.7) | 7,860 bytes | 505 bytes |
| ft, QE | 5 | 5 (8 each) | 15 | 42.8 ms (40.7 to 45.1) | 7,770 bytes | 492 bytes |
| ft, `--no-qe` | 5 | 5 (8 each) | 25 | 45.4 ms (44.2 to 47.0) | 7,919 bytes | 505 bytes |

A shard file of 8 rows is 1,217 to 1,225 bytes. The wall time includes hashing, the
Parquet writes and the manifest and excludes process start-up. These are the cost of
the machinery on 40 rows and say nothing about a real generator.

### MBR cost

`mbr.select` on one segment of `n` distinct candidates with `ChrfUtility`, single
thread, plain Python; chrF calls counted with a wrapper, seconds are the median of 200
(short) and 100 (long) repeats:

| Segment | n | chrF calls (n*(n-1)) | Seconds per segment | Seconds per 1,000 segments |
|---|---|---|---|---|
| 51 characters | 4 | 12 | 0.000661 | 0.66 |
| 51 characters | 8 | 56 | 0.003061 | 3.06 |
| 51 characters | 16 | 240 | 0.013574 | 13.57 |
| 189 characters | 4 | 12 | 0.002165 | 2.16 |
| 189 characters | 8 | 56 | 0.009946 | 9.95 |
| 189 characters | 16 | 240 | 0.043069 | 43.07 |

About 55 microseconds per call for the short segment and about 180 for the long one. At
the default `n` of 4 and a long segment, MBR costs about two seconds per thousand sources
that reach it; it is paid on the main thread after the shard's candidates arrive.

### Memory and time of the input handling (200,000 rows)

A synthetic Mongolian-looking frame of 200,000 rows (`id`, `lang`, `text`; Cyrillic text
of 157 to 246 characters, mean 200.1; a 19,195,988-byte Parquet file), `shard_size`
512, each configuration in a fresh process (baseline after imports: 65.2 MiB), three
repeats, `resource.getrusage` for the peak:

| Step | Time | RSS before to after | Peak RSS |
|---|---|---|---|
| `read_monolingual(path, lang="khk_Cyrl")` | 0.16 to 0.20 s | 65.2 to 219.5 MiB (frame estimated at 72.4 MiB by polars) | 217.8 MiB at most |
| `plan_shards` (391 shards) | 0.10 to 0.11 s | 219.4 to 323.1 MiB | 323.0 MiB |
| `job_fingerprint` | 0.55 to 0.57 s | no growth (hashes row by row) | unchanged |

So planning costs about 104 MiB for 200,000 rows (about 544 bytes per row: every text is
copied into a Python tuple), the frame read about 154 MiB of resident memory, and the three
steps together peak at about 323 MiB. A linear extrapolation to ten times as many rows is
about 1.5 GiB for the read and 1.0 GiB for the plan; that extrapolation was not measured.
The corpus rows and candidates of the run itself were not part of this probe.

### Damaged shard files

Before the content digest (2026-11-07): single-bit flips of a 3,324-byte shard-shaped
Parquet file (8 rows of 2 candidates), 150,000 trials with polars 2.1.0 (a parent process
restarts the child after an abort): 38,581 raised a polars error (an invalid file,
regenerated), 45,528 still decoded to the same frame, 65,607 decoded to a different frame
without error, 276 raised `PanicException` (caught) and 8 aborted the process with "memory
allocation of 2305843009213693952 bytes failed". On a realistic 156,403-byte shard (512 rows
of 4 candidates, 3,000 flips) 1,101 raised an error, 22 decoded unchanged and 1,877 decoded
to different data; no panic and no abort occurred. Through `ShardStore.read` (the structural
checks alone) the final review measured 63.7% different without error and 0.3% caught on a
98,112-byte shard, and 27.5% different and 0.9% caught on a 3,248-byte one.

With the digest (2026-11-08), every flip judged through `ShardStore.load`, one bit at a
time, a parent process restarting the child after an abort:

| Shard file | Flips | Digest mismatch | Unreadable | Structural | Rows unchanged | Different rows, no error | Aborted |
|---|---|---|---|---|---|---|---|
| 1,236 bytes (8 rows x 4, the unit test's shard), all bits | 9,888 | 941 | 4,136 | 249 | 4,561 | 0 | 1 |
| 3,339 bytes (8 rows, random words), all bits | 26,712 | 7,461 | 11,046 | 232 | 7,972 | 0 | 1 |
| 98,203 bytes (512 rows x 4), 3,000 random bits | 3,000 | 1,843 | 1,117 | 8 | 32 | 0 | 0 |

The same harness with the digest comparison switched off decoded 1,840 of the 3,000 flips
of the large shard to different candidates (61.3%). The aborts are single positions (bit 0
of byte 63 of the first file, bit 0 of byte 66 of the second); the unit test sweeps every
7th byte of the first shard with a bit that cycles through all eight, which never pairs byte 63
with bit 0, and says so.

Cost of the digest (medians of 200 repeats, 40 for the largest; canonical JSON text of 6.0,
350.6 and 2,806.6 KiB for 8, 512 and 4,096 rows of 4 candidates of 10 to 30 words): 0.015,
0.71 and 7.0 ms, about 2.0 to 2.5 microseconds per KiB of text; reading the metadata block
7 microseconds; a whole `ShardStore.read` of the 512-row shard (95.9 KiB on disk) 2.2 ms, of
which the digest is 0.7 ms and the decoding 0.7 ms; `write` of it 5.4 ms. A shard takes
seconds to minutes to generate.

## Memory model of `run_job`

The caller's frame cannot be freed by `run_job` (the CLI holds the monolingual rows in
memory). The peak is about that frame, plus the planned shards (about 544 bytes per row
in the measurement above for texts of about 200 Cyrillic characters), plus the corpus rows
accumulated so far, plus the candidates of ONE shard at a time (selection is per shard);
the final `pl.concat` of the per-shard rows transiently doubles the corpus part. The
fingerprint adds nothing. Not measured: a whole run beyond the 40-row tiny one, the
corpus part and the candidates of a shard. For many millions of rows the input alone is
gigabytes (the extrapolation above): cut the input into files and run them one at a time.

## Errors

| Where | What | Error | Exit |
|---|---|---|---|
| Config | invalid file, unknown job, job names unknown generator, `n > beam_size`, `mn_Mong`, bad `base_url` | `ConfigError` | 2 |
| Input | not Parquet, missing or non-string column, no rows of the language, null or empty id, blank text, duplicate id | `DataError` | 3 |
| Client | lone surrogate in a text (names the index, never the text) | `DataError` | 3 |
| Client | transport error, 429, 500, 502, 503, 504: retried with backoff `backoff_s * 2**attempt` (the exponent capped at 10, the sleep at 30 s; a whole-number `Retry-After` on 429 and 503 is used, capped at 30 s), up to `retries` | `UpstreamError` "failed with ..., attempts: n" | 4 |
| Client | another status (400, 404, ...), any 2xx with a body that is not JSON, not an object, has an `error` that is not null, has no `choices` list, has more choices than asked or a choice without a message object | `UpstreamError` (not retried; non-2xx messages carry a 200-character preview of the response body) | 4 |
| CT2 | model directory or SentencePiece file missing or unloadable / an unsupported `device` or a language outside the family's table | `DataError` / `ConfigError` | 3 / 2 |
| CT2 | library error while translating; a result count other than the text count | `UpstreamError` | 4 |
| Runner | a translator `DataError` or `ConfigError` | the same type, naming `shard i of n` | 3 or 2 |
| Runner | a translator `UpstreamError` or `OSError` | `UpstreamError` "shard i of n failed: ...", chained | 4 |
| Runner | a reply that is not a list, has the wrong length or a non-list or non-string entry | `DataError` naming the shard (nothing is written for it) | 3 |
| Runner | `KeyboardInterrupt`, `SystemExit` | unchanged, with the finished shards on disk | 130 for Ctrl-C |
| Selection | a QE scorer that returns the wrong number of scores, a score that is not a number, or raises an `OSError` / raises another kind of exception | `UpstreamError` / propagates as it is (a traceback) | 4 / 1 |
| CLI | an `OSError` creating the output directory or writing outputs | "cannot write the synth outputs under DIR: ..." | 1 |
| CLI | the output directory holds another stage's manifest | `DataError` | 3 |

### Known limitations

The final review's findings about the interrupt, the out directory with `[` and the shard
digest are fixed (above). What it found and D6 deliberately leaves, one line each:

- `select._score_in_chunks` repeats the chunk loop and the count check of D4's
  `filter/qe.py:_score` (with "pairs" where D4 says "rows"); the two must change together
  if the scorer contract changes.
- `plan.py` imports `Kind` from `select`; `Plan` has five fields for three values; `bt.py`
  and `ft.py` exist for a kind-mismatch check that cannot fire through `run_job` or the CLI;
  `ShardStore.invalid` has no production caller; the CLI imports `Releasable` from
  `rinchen.mine.embed`.
- A `base_url` with a space inside the host (`http://h h/v1`) is accepted; httpx normalises it.
- The `rinchen` console entry has no CPython 3.14 assertion except `doctor`, as in D4 and
  D5, and a `manifest.json` that cannot be parsed is overwritten by the D4 manifest store's
  design.
- An `OSError` raised by `scorer.release()` is worded as a failure to write the outputs
  (read from the code, not run).
- Two runs into one output directory are not guarded (above).

## Extending

**A generator.** Add its config class to `rinchen/config/synth.py` (a `kind` literal and a
`Tag` in the discriminated union; keep the YAML equal to `SynthConfig()`); write a class
with `translate(texts, src_lang, tgt_lang) -> list[list[str]]` (one list per text in input
order, never more than `n`, possibly empty; `UpstreamError` for a service failure,
`DataError` for a bad text, `ConfigError` for a language it does not cover; candidates
cleaned with `clean_candidates`); add a `case` to `build_translator` in
`rinchen/cli/synth.py` and its origin to `_FORWARD_ORIGIN_BY_GENERATOR_KIND` in `tags.py`
(`origin_for` raises for an unknown kind); test it with fakes, never a real server. The
fingerprint covers its config automatically.

**A prompt or a language.** The registry decides which directions exist
(`supported_directions()` derives them from `configs/languages.yaml`, leaving out entries
written in Mongolian script). A new partner language needs a `LANGUAGE_NAMES` entry in
`prompts.py` (a test fails and rendering raises until it exists) and, for CTranslate2, an
entry in `NLLB_CODES` and `MADLAD_CODES` in `ct2_baselines.py`. The prompt is three
constants in `prompts.py` (`_TASK`, `_OUTPUT_RULE`, `_CYRILLIC_OUTPUT_INSTRUCTION`) and a
plain concatenation; the text to translate is appended unchanged after a blank line.
The Cyrillic sentence is present exactly when the target is `khk_Cyrl`, and no prompt
may mention traditional script or Inner Mongolia (a test checks it). The rendered prompt for
English to Mongolian: `Translate the following text from English to Mongolian (Khalkha,
Cyrillic script). Output only the translation, without explanations, notes or quotation
marks. Write the translation in Mongolian Cyrillic script only.` followed by a blank line
and the text. Changing the wording does NOT change the fingerprint: run with `--force`
or another `--seed` afterwards, or old shards are reused. The tests pin the literal
prompts, so a wording change edits them too.

## Open gates

Not done by D6, because no real generator, data or reader exists: running any real
generator (the servers, the NLLB and MADLAD models, prompt quality, throughput, GPU
memory); a Mongolian reader's judgement of any output; calibrating the QE floor, the
script share and every default on real output; a language identifier on the synthetic
side (purity cannot tell Mongolian from other Cyrillic text); COMET and MetricX as MBR
utilities (`Utility` is the seam); training-time `<bt>` and `<ft>` insertion (D7 and D8).
The D3 gates assigned to D4 stay open as recorded in `docs/runbooks/04_filtering_mining.md`.

## What was not verified

- **No real server or model ran.** `ctranslate2.Translator` was exercised only with the
  tiny D5 converter model, for one upstream fact (`D-CT2-NUMHYP`). Whether a server honours
  `seed`, `n`, the `Retry-After` header or a chat template is unknown.
- **The MADLAD and NLLB token conventions** (a trailing `</s>`, the MADLAD tag inside the
  text, the leading target code in an NLLB hypothesis) follow the Hugging Face tokenizers
  and are unverified against real converted models.
- **Prompt quality, any Mongolian output and every threshold and default** are untested or
  priors; no Mongolian reader has judged anything.
- **The real MetricX-QE scorer was not run in D6**; the QE tests use fake scorers, and
  `cometkiwi` has no adapter.
- **Windows and macOS were not run.**
- **Concurrent runs into one output directory are not guarded** (see above); only
  sequential runs are tested.
- **A damaged shard file can still abort the process** (a Rust allocation failure that no
  handler catches; measured above), and the content digest is not a signature.
- **Memory beyond 200,000 rows** was extrapolated, and the memory of a whole `run_job`
  beyond the tiny run was not measured.
- **The real binary** was run for `--help`, a dry run, a failing run against a closed port
  and, against a loopback stand-in for a chat server that is not a model, an interrupt and
  four out directories with glob characters; every other successful run used the fake seams
  in process.

## Where the code lives

`src/rinchen/mbr/`, `src/rinchen/synth/` (`translator`, `prompts`, `tags`,
`_chat_completions`, `vllm_client`, `llamacpp_client`, `ct2_baselines`, `_candidates`,
`_texts`, `shards`, `select`, `plan`, `bt`, `ft`, `run`), `src/rinchen/config/synth.py`,
`configs/data/synth.yaml`, `src/rinchen/cli/synth.py`, unit and property tests named
`test_mbr_*`, `test_synth_*`, `test_config_synth.py`, `test_cli_synth.py`,
`tests/integration/test_synth_tiny_run.py`, and the test helpers in `tests/support/`
(`planted_synth.py`, `synth_cli.py`, `fake_llm_server.py`, `shard_files.py`).
