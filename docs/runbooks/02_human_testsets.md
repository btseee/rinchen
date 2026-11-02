# 02 — Human Test-Set Annotation

## Scope

`rinchen annotate` lets a small trusted group of annotators build a human
test set: every item gets 2 independent translations, 1 review that picks
the better one, and an adjudication pass when the two translations
disagree too much (chrF < 40) or the review score is low (< 70).

Recruiting annotators is explicitly out of scope. There is no login --
annotators are identified by a plain name in the URL
(`/annotator/<name>/next`), suitable only for a small trusted group.

## Walkthrough

```bash
# 1. Create the DB and seed the annotator roster.
uv run rinchen annotate init --db test.sqlite --annotators alice,bob,carol

# 2. Import source segments (JSONL: src, src_lang, tgt_lang, doc_id, seg_idx).
uv run rinchen annotate import --db test.sqlite --file items.jsonl --set myset

# 3. Serve the app locally.
uv run rinchen annotate serve --db test.sqlite
# Visit http://127.0.0.1:8000/, pick a name, work through tasks.

# 4. Export once enough items are resolved (partial exports are fine --
#    unresolved items are silently skipped, not an error).
uv run rinchen annotate export --db test.sqlite --set myset --format parquet --out myset.parquet

# 5. Check inter-translator/inter-reviewer agreement.
uv run rinchen annotate agreement --db test.sqlite --set myset
```

## Operational notes

- **At least 3 annotators are required for the workflow to ever resolve
  an item.** Each item needs 2 independent translations (from 2 different
  annotators) plus 1 review from a third annotator who did not translate
  that item -- a translator cannot review their own item's translations.
  With only 2 annotators seeded, every item's review task is permanently
  ineligible for both of them, so no item ever resolves past the
  translate stage. Seed at least 3 names with `annotate init
  --annotators` (or `annotate init` + re-running with more names) before
  starting real annotation work.
- **Task priority: adjudicate, then review, then translate -- not simple
  oldest-first.** `claim_next_task` does NOT serve the oldest-created
  eligible task across all three kinds. It first checks priority by kind
  (`adjudicate` > `review` > `translate`), and only within the
  highest-priority kind that has an eligible task does it break ties by
  oldest-created. In practice this means ALL eligible adjudicate tasks are
  served before ANY eligible review task, and ALL eligible review tasks
  are served before ANY eligible translate task -- an adjudicate task
  jumps ahead of every translate task for an eligible annotator, not just
  newer ones. It also re-serves an annotator's own already-`assigned`
  task before claiming anything new, so refreshing or revisiting the page
  never orphans an in-progress task. Encourage whoever can adjudicate to
  clear that queue promptly, since unresolved adjudicate/review work
  blocks all translate throughput for annotators eligible to do it.
- **Krippendorff's alpha reports "N/A" by design.** D2 captures exactly
  one review per item, which is a single-rater vector -- Krippendorff's
  alpha is undefined for that shape. The CLI reports "N/A (needs 2+
  independent reviews per item)" rather than a misleading number. A later
  milestone that captures multiple independent reviews per item will get
  a real, meaningful value from the same (already-tested) function.
- **Backups.** The DB is a single SQLite file in WAL mode, which also
  creates `<db>-wal` and `<db>-shm` sibling files while open. Either run
  `PRAGMA wal_checkpoint(FULL);` before copying just the main file, or
  copy all three files together.
- **Deployment.** `rinchen annotate serve` binds to `127.0.0.1` by
  default. Exposing it beyond localhost needs a reverse proxy (not built
  here) -- this app has no authentication and is not meant to be
  internet-facing as-is.
