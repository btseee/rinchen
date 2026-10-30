# Runbook: 01 — Data Acquisition

## Overview

`rinchen data fetch` pulls raw data into `data/raw/<source-id>/` per the
registry at `configs/data/sources.yaml` (real sources) or
`configs/data/sources.tiny.yaml` (offline fixtures, used by
`--profile tiny`). `rinchen data ls` lists registry entries without
fetching anything.

## Disk estimates

Real production sources are not yet populated in `sources.yaml` (see
`DECISIONS.md`'s 2026-10-30 entry — Rinchen is building its own standard
dataset rather than seeding from the 7 candidates reviewed during D1).
Once real sources are added, estimate disk usage per source from its
registry entry before fetching `--all` on a constrained machine; OPUS
corpora report their `size` (segment count) via the API, HuggingFace
datasets report their size on the dataset card, and CommonCrawl WARC
ranges are bounded by the `length` field in each CDX record queried.

## Parallelism

`rinchen data fetch --all` currently fetches sources sequentially. This
is intentional at D1: the registry and CLI are the proven mechanism;
parallelizing fetches (e.g. via a thread pool, respecting each remote's
rate limits) is a reasonable follow-up once there are enough real sources
for it to matter, not before.

## Rerun semantics

`acquire/http.py`'s `download()` resumes a truncated transfer via its
`.part` file and verifies `sha256` if the registry entry sets
`expected_sha` -- this applies to `http`, `local`, and `opus` registry
entries (each produces one output file); `hf` and `commoncrawl` entries
reject `expected_sha` at config-validation time, since they write
multiple output files with no single hash to check against (see
`DECISIONS.md`'s 2026-10-30 "expected_sha narrowed to single-file kinds"
entry). It does not currently skip a fetch whose destination
already exists and is already complete (no hash-based rerun-skip wired to
`rinchen.manifest`'s stage runner yet) — that integration is part of the
still-deferred config→manifest seam noted in `DECISIONS.md`'s 2026-10-29
D0 wrap-up entry. For now, re-running `rinchen data fetch` for an
already-fetched source re-downloads it.

## How to add a new source

1. Vet the candidate with the same rigor used during D1's own design:
   download at least 100 random sentence pairs (for parallel data) or
   100 random documents (for monolingual data) and review every one for
   script correctness, translation fidelity (if parallel), and content
   cleanliness. Any issue found means the candidate is rejected — see
   `DECISIONS.md`'s 2026-10-30 entry for the bar this project holds to.
2. Add an entry to `configs/data/sources.yaml` matching one of the
   `SourceEntry` variants in `src/rinchen/acquire/registry.py` (`http`,
   `opus`, `hf`, `commoncrawl`).
3. Confirm it: `uv run rinchen data ls` should list it; `uv run rinchen
   data fetch --source <id>` should fetch it for real.
4. Record the vetting result (pass/fail, with the specific evidence) in
   `DECISIONS.md`, whether the source is accepted or rejected.
