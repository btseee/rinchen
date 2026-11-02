# D2 — Human Test-Set Annotation App: Design

**Status:** approved by user, 2026-11-02. See master plan
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md` lines 337-343 for
D2's one-paragraph spec and STOP gate; this document fills in the
implementation detail that one-liner leaves open.

## Goal

Build `rinchen annotate` (`init|import|serve|export|agreement`): a small
FastAPI + Jinja2 + vendored-htmx web app, backed by SQLite (stdlib
`sqlite3`, WAL mode), that lets a small trusted group of human annotators
translate and review source segments to build held-out human test sets.
Every item gets 2 independent translations, 1 review that picks the
better one, and an adjudication pass when the two translations disagree
too much or the review score is low. Output exports in the same
origin/split-tagged corpus schema the rest of the pipeline uses.

## Scope boundaries (explicit, from the master plan)

- Recruiting annotators is out of scope.
- A real auth system is out of scope — this serves a small trusted group.
- A custom on-screen keyboard for traditional Mongol script is out of
  scope — annotators use their OS's IME; the app only needs to *render*
  U+1800–U+18AF correctly (with a `vertical-lr` toggle) and round-trip it
  byte-exact through the DB and every export format.
- Deployment is "serves locally" for D2; the runbook notes a reverse-proxy
  caveat for exposing it beyond localhost, but no actual infra is built.

## Repo changes

- New: `src/rinchen/annotate/{app.py, db.py, models.py, routes.py,
  export.py, agreement.py}`, `src/rinchen/annotate/templates/*.html`,
  `src/rinchen/annotate/static/htmx.min.js` (vendored, not CDN-loaded, to
  keep the app usable fully offline).
- New: `src/rinchen/metrics/bleu_chrf.py` — a thin wrapper around the
  already-pinned `sacrebleu==2.6.0` exposing `chrf(hyp: str, ref: str) ->
  float` (sentence-level, chrF2 default args). Lives in `metrics/` rather
  than inside `annotate/` because D4 (filtering/mining, per the master
  plan's `metrics/bleu_chrf.py` filename) will need the same function.
- Modify: `src/rinchen/cli/annotate.py` — replace the D0 stub with 5 real
  `typer` commands, following `src/rinchen/cli/data.py`'s pattern
  (`typer.Typer(no_args_is_help=True)`, `REPO_ROOT`-relative defaults,
  `typer.echo` for output, `typer.Exit(code=2)` for usage errors).
- New: `tests/fixtures/annot/` — hand-authored fixtures: a ~20-item JSONL
  import file (mixing Cyrillic and a few deliberate U+1800–U+18AF
  traditional-script segments to exercise the round-trip test), plus
  smaller fixtures for migration/assignment/agreement unit tests.
- New: `docs/runbooks/02_human_testsets.md`.
- Remove: the top-level `annotation_app/` directory (leftover from the
  superseded Phase 0-14 plan; D2 builds under `src/rinchen/annotate/`
  instead, matching D1's `src/rinchen/acquire/` precedent). Remove its
  line from `CLAUDE.md`'s repo layout block.
- Errors: reuse the existing taxonomy in `src/rinchen/errors.py` —
  `ConfigError` for malformed CLI input / bad JSONL / unknown annotator
  name in `--annotators`, `DataError` for DB-level integrity problems
  (e.g. an item with no tasks, a review task submitted before both
  translations exist). No new error types.

## Database schema

SQLite, WAL mode, opened via stdlib `sqlite3` (no ORM — matches the
project's "build ourselves" bias and keeps the schema inspectable with
plain SQL in tests).

```sql
CREATE TABLE annotators (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL  -- ISO 8601
);

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    set_name TEXT NOT NULL,   -- "set" is a SQL keyword; column is set_name
    src TEXT NOT NULL,
    src_lang TEXT NOT NULL,
    tgt_lang TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    seg_idx INTEGER NOT NULL
);

CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id),
    annotator_id INTEGER REFERENCES annotators(id),  -- NULL until claimed
    kind TEXT NOT NULL CHECK (kind IN ('translate', 'review', 'adjudicate')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'assigned', 'done'))
        DEFAULT 'pending',
    created_at TEXT NOT NULL,
    assigned_at TEXT,
    completed_at TEXT
);

CREATE TABLE translations (
    task_id INTEGER PRIMARY KEY REFERENCES tasks(id),
    text TEXT NOT NULL,
    minutes REAL NOT NULL,  -- time spent, self-reported by the annotator
    flags TEXT NOT NULL DEFAULT '[]'  -- JSON array of flag strings
);

CREATE TABLE reviews (
    task_id INTEGER PRIMARY KEY REFERENCES tasks(id),
    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    errors TEXT NOT NULL DEFAULT '[]',  -- JSON array of error-type strings
    chosen_translation_id INTEGER NOT NULL REFERENCES tasks(id)
);
```

`chosen_translation_id` is a deliberate addition beyond the master plan's
one-line `reviews(task_id, score 0–100, errors JSON)` — it records which
of the item's two `translate` tasks the reviewer judged better, which the
export step needs (see below). `tasks.status` is likewise an addition,
needed to drive the lazy-pull assignment queue.

Migrations: `db.py`'s `init_db(conn)` runs `CREATE TABLE IF NOT EXISTS`
for all five tables plus any future `ALTER TABLE` migrations gated on a
`schema_version` pragma (`PRAGMA user_version`), so `rinchen annotate
init` is idempotent against an existing DB — re-running it must not error
or duplicate schema, and must not touch existing rows.

## Task lifecycle

1. **`import`** inserts each item, then immediately inserts its 2
   `translate` tasks (`status='pending'`, `annotator_id=NULL`) and 1
   `review` task (`status='pending'`, `annotator_id=NULL`). The review
   task exists from the start but is not eligible to be claimed until
   both its item's `translate` tasks are `done` (see claiming rule below)
   — it is not held back structurally, just filtered out of eligibility.
2. **Claiming** (`GET /annotator/{name}/next`): finds the oldest-created
   eligible task for that annotator name:
   - `kind='translate'`, `status='pending'`, and this annotator has no
     other task (of any kind) on the same `item_id`.
   - `kind='review'`, `status='pending'`, both the item's `translate`
     tasks are `done`, and this annotator is not one of the two
     translators.
   - `kind='adjudicate'`, `status='pending'`, and this annotator is
     neither of the item's two translators (per your "any annotator
     except the two translators" answer — the reviewer *may* adjudicate
     their own item, since they already know it's contested).
   On claim: `status='assigned'`, `annotator_id=<claimer>`,
   `assigned_at=now`. If nothing is eligible, the endpoint returns a
   "nothing to do" state (200, not an error — an empty queue is a normal
   condition, not a failure).
3. **Submitting a `translate` task** (`POST /task/{id}/translate`):
   writes `translations(task_id, text, minutes, flags)`, sets the task
   `status='done'`, `completed_at=now`.
4. **Submitting the `review` task** (`POST /task/{id}/review`): writes
   `reviews(task_id, score, errors, chosen_translation_id)`, sets
   `status='done'`. Then computes `chrf(t1, t2)` via
   `metrics.bleu_chrf.chrf` on the item's two translation texts; if
   `chrf < 40 or score < 70`, inserts a new `adjudicate` task
   (`status='pending'`) for that item.
5. **Submitting the `adjudicate` task** (`POST /task/{id}/adjudicate`):
   writes a `translations` row keyed to the adjudicate task's own `id`
   (same table as step 3 — an adjudication *is* a translation, just the
   final authoritative one), sets `status='done'`.

An item is "fully resolved" once: both translate tasks are `done`, the
review task is `done`, and either no adjudicate task was created or the
one that was is also `done`.

## Export (`export.py`)

For each fully-resolved item, the exported text is:

- the adjudicator's `translations.text` (keyed by the adjudicate task's
  id), if an adjudicate task exists and is done; else
- the `translations.text` of whichever translate task's id matches the
  review's `chosen_translation_id`.

Output schema (one row per item): `src, tgt, src_lang, tgt_lang, doc_id,
seg_idx, set, origin, split` with `origin="human"`, `split="test"` —
matching the corpus schema the rest of the pipeline (D1's fetched data,
D4's filtered data) is expected to converge on. `rinchen annotate export
--db PATH --set NAME --format {jsonl,tsv,parquet}` writes one file;
Parquet via the already-pinned `polars`/`pyarrow`. An item that isn't
fully resolved is skipped from export (not an error — exporting a
test set mid-annotation is a normal partial-progress operation).

## Agreement report (`agreement.py`)

`rinchen annotate agreement --db PATH --set NAME` prints:

- **Pairwise chrF**: mean of `chrf(t1, t2)` (via `metrics.bleu_chrf`)
  across all items in the set that have both translations done.
- **Krippendorff's alpha** on review scores: hand-implemented (not a new
  dependency — matches D1's build-ourselves precedent for small,
  well-defined, testable algorithms) using the standard interval-metric
  formula (observed vs. expected disagreement from pairable coincidence
  matrices). Input is the per-item review score as the single "unit";
  since D2 only ever has exactly one review per item, this reduces to a
  one-rater-per-unit degenerate case for score agreement — the design
  computes it over whatever review scores exist in the set, documented in
  the runbook as useful primarily once the project captures more than one
  independent review per item in a later milestone. (If this degenerate
  case turns out not to be useful in practice, that's a runbook note, not
  a blocker — the function itself is tested against literature-standard
  worked examples independent of how D2 happens to use it.)

## CLI contracts

```text
rinchen annotate init --db PATH --annotators NAME[,NAME...]
rinchen annotate import --db PATH --file FILE.jsonl --set NAME
rinchen annotate serve --db PATH [--host HOST] [--port PORT]
rinchen annotate export --db PATH --set NAME --format {jsonl,tsv,parquet} --out PATH
rinchen annotate agreement --db PATH --set NAME
```

Import JSONL line shape: `{"src": str, "src_lang": str, "tgt_lang": str,
"doc_id": str, "seg_idx": int}` — `set` comes from the `--set` flag, not
the file, so one file can be imported under different set names if
needed.

## Web app (`app.py`, `routes.py`, `templates/`)

- `create_app(db_path: Path) -> FastAPI`, lifespan opens the sqlite
  connection once (WAL mode tolerates concurrent readers/single writer,
  fine for this scale).
- Routes: `GET /` (name picker, listing the roster from `annotators`),
  `GET /annotator/{name}/next` (claims + renders the right template for
  the task kind), `POST /task/{id}/translate`, `POST /task/{id}/review`,
  `POST /task/{id}/adjudicate` (HTMX form posts, re-rendering the "next
  task" partial on success), `GET /healthz`.
- Traditional-script segments (`src_lang == "mn_Mong"` or `tgt_lang ==
  "mn_Mong"` — an exact match on the lang code; D2 does not depend on
  D3's not-yet-built `script_of` detector) render in a `<div
  style="writing-mode: vertical-lr">` with a plain toggle button to flip
  to horizontal — no custom input method, per scope.
- No login, no sessions, no cookies — the annotator name is just a path
  segment, matching your "name/ID picker, no auth" answer. This is
  explicitly a small-trusted-group tool, documented as such in the
  runbook.

## Testing strategy

Written test-first per task, mirroring the master plan's list plus two
additions for the schema extensions:

1. `init_db` is idempotent — calling it twice against the same file
   produces identical schema and doesn't touch existing rows.
2. Assignment never gives one annotator two roles on the same item
   (translate+translate, translate+review, translate+adjudicate, or
   review+adjudicate all forbidden; review+adjudicate by the *same*
   non-translator annotator is allowed, per the lifecycle above).
3. Adjudication trigger fires exactly on `chrf < 40 OR score < 70`, and
   not on boundary-passing values (`chrf == 40`, `score == 70` do NOT
   trigger — test both edges).
4. Export schema validates across all three formats; a partially-resolved
   item is correctly skipped.
5. **(added)** Export picks `chosen_translation_id`'s text when no
   adjudicate task exists, and the adjudicator's text when one does and
   is done.
6. Agreement: two identical translations → chrF 100 (fixture); Krippendorff
   alpha against at least one literature-sourced worked example with a
   known expected value.
7. HTMX routes return 200 via FastAPI's `TestClient` for the full
   translate → review → (sometimes) adjudicate → export flow.
8. A traditional-script (U+1800–U+18AF) segment round-trips byte-exact
   through import → DB → every export format.

## STOP gate

Verbatim from the master plan: app serves locally (`rinchen annotate
serve`); a 20-item fixture (`tests/fixtures/annot/`) can be fully
annotated by a scripted client (FastAPI `TestClient`, not a real browser)
and exported, with zero items left pending at the end.

## Runbook (`docs/runbooks/02_human_testsets.md`)

Covers: `init`/`import`/`serve`/`export`/`agreement` walkthrough end to
end on the fixture; SQLite backup = copy the `.sqlite` file (WAL mode
means also copying the `-wal`/`-shm` siblings, or running `PRAGMA
wal_checkpoint` first); a deployment note that `serve` binds to
localhost by default and needs a reverse proxy (not built here) to expose
it beyond that. Recruiting is explicitly out of scope, stated up front.
