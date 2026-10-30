# D1 — Data acquisition — design

Date: 2026-10-30

## Context

D1 of the master plan (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`,
`<milestones>` section) builds the data-acquisition machinery: generic
HTTP download with resume, an OPUS API client, a HuggingFace `datasets`
streaming client, a CommonCrawl CDX + WARC extraction pipeline, and a
registry that drives all of them from a YAML config. D0 (environment,
skeleton, doctor) is complete; this is the first milestone that writes
real data-handling code.

A significant part of brainstorming this milestone turned into real,
rigorous data-source vetting, at the user's explicit direction: for every
candidate corpus (5 OPUS corpora the user pre-selected — TED2020, QED,
NeuLab-TedTalks, GNOME, Tatoeba — plus two HuggingFace candidates,
`ugshanyu/1millionMongolianSentence` and `davidstap/NTREX`), 100 randomly
sampled English-Mongolian pairs were downloaded for real and reviewed
pair-by-pair for script correctness, translation fidelity, and alignment,
under a zero-tolerance bar ("if there is the slightest issue, ignore that
dataset"). Result: **6 of 7 failed** (GNOME: 34% defect rate including
leaked English dev-comments in the Mongolian column; NeuLab-TedTalks: ~19%
systematic truncation; QED: galig/Latin-script Mongolian present — a hard-
constraint violation — plus severe misalignment; TED2020: empty pairs,
wrong-sense mistranslations, cross-segment content bleed; `ugshanyu`:
Latin/Cyrillic character-substitution defects plus real name/fact
mismatches consistent with machine back-translation; NTREX: ~10% defect
rate with meaning-inverting mistranslations). Only Tatoeba came close to
clean (typo-level noise, no structural problems, no script violations),
and even that was borderline under the stated bar.

Given this, the user decided: **Rinchen will build its own standard
dataset from scratch**, not seed from these candidates. D1's own scope —
building the acquisition *machinery* — is unaffected by this; it just
means `configs/data/sources.yaml` ships with zero real production entries
rather than a pre-populated catalog. [EVIDENCE: live 100-sample QA review
of all 7 candidates, 2026-10-30, conducted via parallel forked review
agents, each independently sampling and reading 100 real pairs — see
`DECISIONS.md` for the full per-dataset verdict.]

This also means test fixtures must NOT be derived from any of the
reviewed-and-rejected (or even the borderline-passed) real corpora, to
keep "data used to test the machinery" cleanly separated from "data
judged fit for training" — the fixtures are hand-authored, obviously
synthetic content, structurally accurate to the real API/file shapes
verified during research (OPUS API JSON shape, WARC format, moses zip
layout) but containing no text carried over from the vetting process.

## Scope

In:

**1. `src/rinchen/acquire/http.py`**: `download(url, dest, sha256=None,
resume=True, retries=5) -> Path`. Streams to a `.part` file; on a dropped
connection, retries with an HTTP `Range` header resuming from the `.part`
file's current size, exponential backoff (hand-rolled, no new
dependency), up to `retries` attempts. Verifies sha256 on the assembled
file if given, raising `DataError` (and deleting the bad file) on
mismatch. Also handles a `local://<relative-path>` URL scheme used only
by the `kind: local` registry entries described below — resolves the
path relative to the repo root and copies the file to `dest` (optionally
still sha256-checking it), with no network involved at all.

**2. `src/rinchen/acquire/opus.py`**: `list_corpora(src, tgt) ->
list[dict]` queries `https://opus.nlpl.eu/opusapi` (confirmed live: no
trailing slash before the query string — a trailing slash breaks the
response; the API is source/target-order-independent, each corpus entry
carries its own canonical `source`/`target` fields). `fetch(corpus, src,
tgt, preprocessing="moses", version="latest") -> Path` finds the matching
corpus entry from `list_corpora()` and calls `http.download()` on its
real `url` field. Confirmed live: passing the literal string
`version="latest"` as a query parameter returns zero results from the
real API (it isn't a recognized filter value) — `fetch()` must omit the
`version` parameter from the actual API query entirely when the caller
asks for `"latest"`, and instead pick the entry with `latest == "True"`
from the unfiltered result set (or the highest version string if multiple
are marked latest, as seen with Tatoeba's many dated releases).

**3. `src/rinchen/acquire/hf.py`**: streams a HuggingFace dataset via the
already-pinned `datasets` library and writes Parquet shards via
`polars`/`pyarrow` (both already dependencies). The master doc's test
list doesn't name a dedicated HF no-network test, but the hard constraint
("never network-access in unit tests") still applies — the shard-writing
function takes an injected row iterator (not a `datasets.load_dataset`
call baked in), so the shard-writing logic itself is unit-testable
without touching the network. The thin wrapper that actually calls
`datasets.load_dataset(..., streaming=True)` is not unit-tested (would
require network) and has no coverage at all -- it is NOT reachable via the
registry's `kind: local` path in the offline tiny-profile fixture flow,
since every tiny-profile entry is `kind: local`, which dispatches to
`http.download`, never to `hf.fetch`.

**4. `src/rinchen/acquire/commoncrawl.py` + `src/rinchen/acquire/warc.py`**:
`commoncrawl.py`'s `query_cdx(url_pattern, collection) -> list[dict]`
hits the real CDX shape confirmed live (newline-delimited JSON, one
record per line, with `filename`/`offset`/`length` for a WARC byte-range
fetch and a CLD3-tagged `languages` field usable to pre-filter for `mon`
before downloading anything). `fetch_records(records, dest) ->
list[Path]` does bounded HTTP Range fetches of each record — a single
bounded range per record, a different operation from `http.download()`'s
large-file resume semantics, implemented as its own small function in
`commoncrawl.py` (it can still reuse `http.py`'s retry/backoff helper if
that's factored out as a small reusable piece, but it is not a resumable
`.part`-file download). `warc.py`'s `iterate(warc_path) ->
Iterator[WarcRecord]` uses `warcio`; `extract_text(record) -> str` uses
`trafilatura` (both already dependencies).

**5. `src/rinchen/acquire/registry.py` + `configs/data/sources.yaml`**: a
`SourceKind` enum (`http`, `opus`, `hf`, `commoncrawl`, `local`), each
entry validated via a pydantic `SourceEntry` model (`id`, `kind`, `langs`,
`url` or `query` depending on kind, `expected_sha` optional, `notes`).
Invalid schema raises `ConfigError` (matching D0's config-loader
convention: `ConfigError` for malformed configuration, `DataError`
reserved for corrupt/mismatched *data*). `kind: local`'s `url` is a
literal `local://...` fixture path — fetching it is a file copy via
`http.download()`'s local-scheme handling, no per-kind special-casing
needed in the registry itself. `configs/data/sources.yaml` ships with
**zero real entries** (see Context) — just the schema made real and
validatable, with a comment explaining why it's currently empty and
pointing at wherever the project's future curation process will live.
`configs/data/sources.tiny.yaml` has the same `SourceEntry` schema but
entries use `kind: local` pointing at the four fixtures below, used only
when `--profile tiny` is passed.

**6. CLI**: `rinchen data fetch --source <id>|--all [--profile tiny]`,
`rinchen data ls` (lists registry entries). Registered under the existing
`cli/data.py` stub (currently empty per D0).

**7. Tests first** (no network; httpx `MockTransport` + local fixtures):
- `test_download_resumes_after_truncated_connection`: a custom httpx
  transport that cuts off a response mid-stream on the first request,
  then correctly honors a `Range` header on a second request, continuing
  from the `.part` file's byte count.
- `test_download_raises_data_error_on_sha_mismatch`.
- `test_download_local_scheme_copies_fixture_file`.
- `test_opus_list_corpora_parses_real_response_shape`: fed the
  `opus_api_mn_en.json` fixture via `MockTransport`, asserts the parsed
  corpus list matches expected fields.
- `test_opus_fetch_omits_literal_latest_version_param` (regression test
  for the real API gotcha above).
- `test_warc_iterate_and_extract_matches_golden_text`: real 5-record WARC
  fixture, golden extracted text computed once and stored as the
  expected value.
- `test_registry_validates_schema`: valid entries pass; a handful of
  deliberately invalid ones (missing required field, bad `kind` value)
  each raise `ConfigError` with a clear message.
- `test_hf_writes_parquet_shards_from_injected_iterator`: no network,
  feeds a fake row iterator directly to the shard-writing function.

**8. Fixtures** (`tests/fixtures/raw/`), all hand-authored, none derived
from the rejected/borderline real corpora:
- `opus_api_mn_en.json`: a realistic OPUS API response (structurally
  accurate to the confirmed real shape — `corpora` list with
  `alignment_pairs`/`corpus`/`size`/`source`/`target`/`url`/`version`
  fields — but with an invented, clearly-fictional corpus name and URL,
  not a real captured response).
- `sample.warc.gz`: a real, valid gzipped WARC file with 5 records, built
  with `warcio.WARCWriter`, each record a tiny synthetic HTML page with
  known, simple text content chosen so `trafilatura`'s extraction is
  unambiguous (e.g. a single `<article><p>...</p></article>` per page).
- `mn_news.html`: one standalone synthetic HTML page (simulating a single
  extracted page), simple fictional Mongolian-language placeholder text
  (NOT claiming to be real news, just plausible-looking markup structure
  for testing HTML extraction).
- `moses_mn_en.zip`: a small zip in the real OPUS moses format (two
  aligned `.txt` files, one per language) with a handful of simple,
  obviously-fictional synthetic sentence pairs.

**9. Runbook**: `docs/runbooks/01_acquisition.md` — disk estimates,
parallelism, rerun semantics (how the registry's hash-based dedup works,
reusing D0's `manifest`/`stage` module where it makes sense), how to add
a new source to `sources.yaml` once real sources are curated.

Out of scope:
- Any real, vetted production entries in `sources.yaml` — deferred to a
  dedicated future data-curation effort with the same rigor just
  demonstrated (100-sample zero-tolerance review per candidate), not
  bundled into this machinery-building milestone.
- `lang/codes.py`/`lang/tags.py`/`lang/scripts.py` and `configs/languages.yaml`
  — not named in D1's own "Implement:" line; `sources.yaml`'s `langs`
  field uses plain language-code strings for now.
- The config→manifest integration seam (`config_sha256`/`code_version`)
  — already documented as deferred in `DECISIONS.md` from D0; D1's own
  manifest usage (if any, for the hash-based rerun-skip semantics in the
  runbook) works with explicit hashes, not a wired `config_sha256`.
- Mining/alignment (`mine/`), filtering (`filter/`), and the human
  annotation app (`annotate/`) — D2/D4 scope.

## Testing

TDD as practiced throughout this project: each test above is written and
confirmed failing before its implementation exists. The acceptance gate
matching D1's own STOP condition: `rinchen data fetch --profile tiny
--all` works fully offline using the `local://` fixtures (no real network
call made), proving the registry → CLI → `kind`-dispatch → `http.py`
local-scheme chain works end-to-end, independent of whether any real
network source is ever configured.

## Decisions resolved during design (not left open for the plan)

- **`commoncrawl.py`'s bounded-range fetch and `http.py`'s resumable
  download stay separate functions, not a shared abstraction.** At D1's
  scale there are exactly two call sites doing genuinely different
  operations (resumable large-file `.part`-file download vs. a single
  bounded byte-range fetch of an already-known-size WARC record) — per
  YAGNI, forcing them into one shared helper now risks an awkward
  abstraction for a sharing need that doesn't clearly exist yet. Revisit
  only if a third similar call site appears later.
- **`SourceEntry` is a pydantic discriminated union, not one flat
  optional-everything model**, so a misconfigured entry fails validation
  immediately instead of silently carrying irrelevant fields:
  ```python
  class SourceKind(str, Enum):
      HTTP = "http"
      OPUS = "opus"
      HF = "hf"
      COMMONCRAWL = "commoncrawl"
      LOCAL = "local"

  class _BaseSourceEntry(BaseModel):
      model_config = ConfigDict(extra="forbid")
      id: str
      langs: list[str]
      notes: str = ""
      expected_sha: str | None = None

  class HttpSourceEntry(_BaseSourceEntry):
      kind: Literal[SourceKind.HTTP]
      url: str

  class OpusSourceEntry(_BaseSourceEntry):
      kind: Literal[SourceKind.OPUS]
      corpus: str
      src: str
      tgt: str
      preprocessing: str = "moses"
      version: str = "latest"

  class HfSourceEntry(_BaseSourceEntry):
      kind: Literal[SourceKind.HF]
      dataset: str
      config: str | None = None
      split: str = "train"

  class CommonCrawlSourceEntry(_BaseSourceEntry):
      kind: Literal[SourceKind.COMMONCRAWL]
      url_pattern: str
      collection: str

  class LocalSourceEntry(_BaseSourceEntry):
      kind: Literal[SourceKind.LOCAL]
      url: str  # local://... -- resolved relative to the repo root

  SourceEntry = Annotated[
      HttpSourceEntry
      | OpusSourceEntry
      | HfSourceEntry
      | CommonCrawlSourceEntry
      | LocalSourceEntry,
      Field(discriminator="kind"),
  ]
  ```
  The plan writes this exact schema into `registry.py` (or a dedicated
  `registry_schema.py` if that reads cleaner once the file is assembled)
  rather than re-deriving it.
