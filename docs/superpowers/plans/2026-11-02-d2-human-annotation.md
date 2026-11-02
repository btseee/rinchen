# D2 — Human Annotation App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rinchen annotate` (`init|import|serve|export|agreement`): a
FastAPI + Jinja2 + vendored-htmx app over a SQLite DB, so a small trusted
group of annotators can translate/review/adjudicate source segments into a
human test set, exported in the project's corpus schema.

**Architecture:** A pure-SQL `db.py` task-lifecycle engine (no ORM) is the
single source of truth for task assignment/eligibility; `app.py`/`routes.py`
are a thin FastAPI/Jinja2/htmx layer over it; `export.py` and `agreement.py`
are read-only consumers of the same schema. `rinchen metrics.bleu_chrf.chrf`
is a small reusable wrapper other milestones (D4) will also use.

**Tech Stack:** FastAPI 0.162.0, Jinja2 3.1.6, stdlib `sqlite3`, vendored
htmx 2.0.11 (not CDN), `sacrebleu==2.6.0` for chrF, `polars`/`pyarrow` for
Parquet export, `typer` CLI, `uvicorn` for `serve`. All already pinned in
`pyproject.toml` — this plan adds zero new dependencies.

**All code in this plan was pre-verified empirically** (written as scratch
files outside the project tree, run against the project's real installed
dependencies, then deleted) before being written here — see
`docs/superpowers/specs/2026-11-02-d2-human-annotation-design.md` for the
design rationale. One real bug was caught and is already fixed below (a
missing `conn.row_factory = sqlite3.Row` in export's query helper), and the
Krippendorff's alpha implementation below was cross-checked against the
independent `krippendorff` PyPI package on 4 inputs including the canonical
Hayes & Krippendorff (2007) worked example (both give ≈0.862 -- see the
corrected value in Task 4's test below; an earlier draft of this plan had
mistranscribed this as ≈0.849, caught and fixed during Task 4's
implementation when the two independently-computed values disagreed).

## Global Constraints

- CPython 3.14.x only, via `uv run` — every command in this plan assumes
  that environment.
- No new dependencies — `fastapi`, `jinja2`, `sacrebleu==2.6.0`, `polars`,
  `pyarrow`, `uvicorn`, `typer`, `httpx`, `python-multipart` are already
  pinned in `pyproject.toml`; confirmed installed versions: fastapi
  0.162.0, jinja2 3.1.6, sacrebleu 2.6.0, polars 2.1.0, pyarrow 26.0.0,
  uvicorn 0.54.0, typer 0.27.3.
- Reuse `src/rinchen/errors.py`'s existing taxonomy — `ConfigError` for
  malformed CLI input/bad JSONL, `DataError` for DB-level integrity
  problems (e.g. an unknown annotator name). No new error types.
- Tests before code (write the failing test, watch it fail, then
  implement) for every step below, per `CLAUDE.md`.
- `set` is a SQL keyword — the `items` table's column is `set_name`, not
  `set` (the CLI flag is still `--set` for the human-facing contract).
- Krippendorff's alpha is hand-implemented in `agreement.py`, not a new
  PyPI dependency (matches D1's build-ourselves precedent for small,
  well-defined, independently-testable algorithms).
- The old top-level `annotation_app/` directory (leftover from the
  superseded Phase 0-14 plan) is removed as part of this plan — D2 builds
  under `src/rinchen/annotate/` instead, matching D1's `src/rinchen/acquire/`
  precedent.
- Traditional-script detection in the UI uses an exact match
  (`src_lang == "mn_Mong"` or `tgt_lang == "mn_Mong"`) — D2 does not depend
  on D3's not-yet-built `script_of` detector.
- On Windows, never put raw U+1800–U+18AF literal characters into a pytest
  assertion message, test id, or anything that might get printed to the
  console on failure — the default `cp1252` console encoding raises
  `UnicodeEncodeError` on them, which would obscure the real test failure
  with an unrelated crash. Write these characters as `\uXXXX` escapes in
  test source and keep them out of `-k` test ids / `pytest.param(id=...)`.

---

### Task 1: DB schema, migrations, models, import

**Files:**
- Create: `src/rinchen/annotate/__init__.py` (empty)
- Create: `src/rinchen/annotate/models.py`
- Create: `src/rinchen/annotate/db.py`
- Test: `tests/unit/test_annotate_db.py`
- Test fixture: `tests/fixtures/annot/import_sample.jsonl`

**Interfaces:**
- Produces: `db.connect(db_path: Path) -> sqlite3.Connection` (WAL mode,
  foreign keys on, `row_factory=sqlite3.Row`); `db.init_db(conn) -> None`
  (idempotent schema creation); `db.seed_annotators(conn, names:
  list[str]) -> None`; `db.import_items(conn, file_path: Path, set_name:
  str) -> int` (returns item count); `models.Annotator`, `models.Item`,
  `models.Task`, `models.Translation`, `models.Review` dataclasses, each
  with a `from_row(row) -> cls` classmethod.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_annotate_db.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rinchen.annotate import db

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def test_init_db_creates_all_five_tables(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"annotators", "items", "tasks", "translations", "reviews"} <= tables


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, ["alice"])
    db.init_db(conn)  # second call must not error or touch existing rows
    rows = conn.execute("SELECT name FROM annotators").fetchall()
    assert [r["name"] for r in rows] == ["alice"]


def test_connect_enables_wal_mode(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_seed_annotators_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.seed_annotators(conn, ["alice", "bob"])
    db.seed_annotators(conn, ["alice", "carol"])  # alice re-seeded, carol new
    names = {r["name"] for r in conn.execute("SELECT name FROM annotators")}
    assert names == {"alice", "bob", "carol"}


def test_import_items_inserts_items_and_tasks(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)

    count = db.import_items(conn, FIXTURES_DIR / "import_sample.jsonl", "demo")

    assert count == 3
    items = conn.execute("SELECT * FROM items WHERE set_name = 'demo'").fetchall()
    assert len(items) == 3
    for item in items:
        tasks = conn.execute(
            "SELECT kind, status, annotator_id FROM tasks WHERE item_id = ?",
            (item["id"],),
        ).fetchall()
        kinds = sorted(t["kind"] for t in tasks)
        assert kinds == ["review", "translate", "translate"]
        assert all(t["status"] == "pending" and t["annotator_id"] is None for t in tasks)


def test_import_items_round_trips_traditional_script_byte_exact(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.import_items(conn, FIXTURES_DIR / "import_sample.jsonl", "demo")

    # import_sample.jsonl's 3rd line uses mn_Mong traditional script.
    row = conn.execute(
        "SELECT src FROM items WHERE set_name='demo' AND src_lang='mn_Mong'"
    ).fetchone()
    assert row is not None
    # Built via chr() rather than a literal/escaped string so this source
    # file never contains a raw traditional-script character (see Global
    # Constraints: Windows console encoding can't print these on failure).
    expected = "".join(
        chr(c) for c in (0x1820, 0x1821, 0x1828, 0x182C, 0x1860, 0x180B, 0x202F)
    )
    assert row["src"] == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_annotate_db.py -v`
Expected: FAIL / collection error — `rinchen.annotate.db` doesn't exist yet.

- [ ] **Step 3: Create the fixture**

```json
{"src": "\u0421\u0430\u0439\u043d \u0431\u0430\u0439\u043d\u0430 \u0443\u0443", "src_lang": "mn_Cyrl", "tgt_lang": "en", "doc_id": "doc1", "seg_idx": 0}
{"src": "\u0411\u0430\u044f\u0440\u043b\u0430\u043b\u0430\u0430", "src_lang": "mn_Cyrl", "tgt_lang": "en", "doc_id": "doc1", "seg_idx": 1}
{"src": "\u1820\u1821\u1828\u182c\u1860\u180b\u202f", "src_lang": "mn_Mong", "tgt_lang": "en", "doc_id": "doc2", "seg_idx": 0}
```

Save as `tests/fixtures/annot/import_sample.jsonl` (JSON lines, UTF-8, every
`src` value uses `\uXXXX` escapes -- for the 3rd line's real
traditional-script codepoints, and (post D2 whole-branch-review Finding 4)
for the two `mn_Cyrl` lines' real Cyrillic Mongolian text
("Сайн байна уу" / "Баярлалаа") -- to keep the source file itself
plain-ASCII-safe to edit on any OS; `json.loads` resolves the escapes to
the real characters). An earlier version of this fixture used
Latin-transliterated ("galig") text mislabeled as `mn_Cyrl`, which violates
CLAUDE.md's permanent "NEVER: Use galig/Latin informal Mongolian" rule;
corrected per the D2 whole-branch review.

- [ ] **Step 4: Implement `models.py`**

```python
# src/rinchen/annotate/models.py
"""Typed row representations for the annotation app's SQLite tables."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Annotator:
    id: int
    name: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Annotator":
        return cls(id=row["id"], name=row["name"], created_at=row["created_at"])


@dataclass(frozen=True)
class Item:
    id: int
    set_name: str
    src: str
    src_lang: str
    tgt_lang: str
    doc_id: str
    seg_idx: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Item":
        return cls(
            id=row["id"],
            set_name=row["set_name"],
            src=row["src"],
            src_lang=row["src_lang"],
            tgt_lang=row["tgt_lang"],
            doc_id=row["doc_id"],
            seg_idx=row["seg_idx"],
        )


@dataclass(frozen=True)
class Task:
    id: int
    item_id: int
    annotator_id: int | None
    kind: str
    status: str
    created_at: str
    assigned_at: str | None
    completed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            id=row["id"],
            item_id=row["item_id"],
            annotator_id=row["annotator_id"],
            kind=row["kind"],
            status=row["status"],
            created_at=row["created_at"],
            assigned_at=row["assigned_at"],
            completed_at=row["completed_at"],
        )


@dataclass(frozen=True)
class Translation:
    task_id: int
    text: str
    minutes: float
    flags: list[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Translation":
        return cls(
            task_id=row["task_id"],
            text=row["text"],
            minutes=row["minutes"],
            flags=json.loads(row["flags"]),
        )


@dataclass(frozen=True)
class Review:
    task_id: int
    score: int
    errors: list[str]
    chosen_translation_id: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Review":
        return cls(
            task_id=row["task_id"],
            score=row["score"],
            errors=json.loads(row["errors"]),
            chosen_translation_id=row["chosen_translation_id"],
        )
```

- [ ] **Step 5: Implement `db.py` (schema, connect, init, seed, import)**

```python
# src/rinchen/annotate/db.py
"""SQLite schema, migrations, and the import step for the annotation app.

Task-claiming/submission functions (the assignment engine) are added in a
later task, in this same module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    set_name TEXT NOT NULL,
    src TEXT NOT NULL,
    src_lang TEXT NOT NULL,
    tgt_lang TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    seg_idx INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id),
    annotator_id INTEGER REFERENCES annotators(id),
    kind TEXT NOT NULL CHECK (kind IN ('translate', 'review', 'adjudicate')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'assigned', 'done'))
        DEFAULT 'pending',
    created_at TEXT NOT NULL,
    assigned_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS translations (
    task_id INTEGER PRIMARY KEY REFERENCES tasks(id),
    text TEXT NOT NULL,
    minutes REAL NOT NULL,
    flags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS reviews (
    task_id INTEGER PRIMARY KEY REFERENCES tasks(id),
    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
    errors TEXT NOT NULL DEFAULT '[]',
    chosen_translation_id INTEGER NOT NULL REFERENCES tasks(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create schema if absent. Idempotent: safe to call on an existing DB."""
    conn.executescript(_SCHEMA_SQL)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        # Future ALTER TABLE migrations go here, gated on `current`.
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


def seed_annotators(conn: sqlite3.Connection, names: list[str]) -> None:
    for name in names:
        conn.execute(
            "INSERT OR IGNORE INTO annotators (name, created_at) VALUES (?, ?)",
            (name, _now()),
        )
    conn.commit()


def import_items(conn: sqlite3.Connection, file_path: Path, set_name: str) -> int:
    """Parse JSONL, insert items + their 2 translate + 1 review tasks.

    Returns the number of items imported.
    """
    count = 0
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cur = conn.execute(
                "INSERT INTO items (set_name, src, src_lang, tgt_lang, doc_id, seg_idx) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    set_name,
                    row["src"],
                    row["src_lang"],
                    row["tgt_lang"],
                    row["doc_id"],
                    row["seg_idx"],
                ),
            )
            item_id = cur.lastrowid
            now = _now()
            conn.execute(
                "INSERT INTO tasks (item_id, kind, status, created_at) "
                "VALUES (?, 'translate', 'pending', ?)",
                (item_id, now),
            )
            conn.execute(
                "INSERT INTO tasks (item_id, kind, status, created_at) "
                "VALUES (?, 'translate', 'pending', ?)",
                (item_id, now),
            )
            conn.execute(
                "INSERT INTO tasks (item_id, kind, status, created_at) "
                "VALUES (?, 'review', 'pending', ?)",
                (item_id, now),
            )
            count += 1
    conn.commit()
    return count
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_annotate_db.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/rinchen/annotate/__init__.py src/rinchen/annotate/models.py \
        src/rinchen/annotate/db.py tests/unit/test_annotate_db.py \
        tests/fixtures/annot/import_sample.jsonl
git commit -m "D2: add annotation DB schema, migrations, and import"
```

---

### Task 2: chrF metric wrapper

**Files:**
- Create: `src/rinchen/metrics/__init__.py` (if not already present — check
  first; D0 created an empty `metrics/` package, so this may already exist)
- Create: `src/rinchen/metrics/bleu_chrf.py`
- Test: `tests/unit/test_metrics_bleu_chrf.py`

**Interfaces:**
- Produces: `chrf(hyp: str, ref: str) -> float`, used by Task 3's
  `submit_review` (adjudication trigger) and Task 4's `agreement.py`
  (pairwise chrF).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_metrics_bleu_chrf.py
from __future__ import annotations

from rinchen.metrics.bleu_chrf import chrf


def test_chrf_identical_strings_is_100():
    assert chrf("Sain baina uu", "Sain baina uu") == 100.0


def test_chrf_returns_plain_float_in_range():
    score = chrf("Sain baina uu", "Goodbye forever")
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_chrf_dissimilar_strings_below_threshold():
    # Used by the adjudication trigger (chrf < 40) -- confirm a genuinely
    # dissimilar pair actually lands below that threshold, not just "low".
    score = chrf("Sain baina uu", "The weather today is sunny and warm")
    assert score < 40.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_metrics_bleu_chrf.py -v`
Expected: FAIL — `rinchen.metrics.bleu_chrf` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# src/rinchen/metrics/bleu_chrf.py
"""Sentence-level chrF via sacrebleu."""

from __future__ import annotations

from sacrebleu.metrics import CHRF

_CHRF = CHRF()


def chrf(hyp: str, ref: str) -> float:
    """Sentence-level chrF2 (sacrebleu defaults). Returns a float in [0, 100]."""
    return float(_CHRF.sentence_score(hyp, [ref]).score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_metrics_bleu_chrf.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rinchen/metrics/bleu_chrf.py tests/unit/test_metrics_bleu_chrf.py
git commit -m "D2: add chrF metric wrapper (metrics/bleu_chrf.py)"
```

---

### Task 3: Task-lifecycle engine (claim, submit, adjudication trigger)

**Files:**
- Modify: `src/rinchen/annotate/db.py` (append to the module from Task 1)
- Test: `tests/unit/test_annotate_lifecycle.py`

**Interfaces:**
- Consumes: `db.connect`, `db.init_db`, `db.seed_annotators`,
  `db.import_items` (Task 1); `chrf` from `rinchen.metrics.bleu_chrf`
  (Task 2); `DataError` from `rinchen.errors`.
- Produces: `db.ClaimedTask` (dataclass: `id: int, item_id: int, kind:
  str`); `db.claim_next_task(conn, annotator_name: str) -> ClaimedTask |
  None`; `db.submit_translate(conn, task_id: int, text: str, minutes:
  float, flags: list[str]) -> None`; `db.submit_review(conn, task_id: int,
  score: int, errors: list[str], chosen_translation_id: int) -> bool`
  (returns whether an adjudicate task was created); `db.submit_adjudicate(conn,
  task_id: int, text: str, minutes: float, flags: list[str]) -> None`. All
  four are consumed by Task 6's routes and Task 7's CLI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_annotate_lifecycle.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.annotate import db
from rinchen.errors import DataError

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def _setup(tmp_path, annotators):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.seed_annotators(conn, annotators)
    db.import_items(conn, FIXTURES_DIR / "import_sample.jsonl", "demo")
    return conn


def test_claim_returns_none_when_nothing_eligible(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.seed_annotators(conn, ["alice"])
    assert db.claim_next_task(conn, "alice") is None


def test_claim_unknown_annotator_raises_data_error(tmp_path):
    conn = _setup(tmp_path, ["alice"])
    with pytest.raises(DataError):
        db.claim_next_task(conn, "nobody")


def test_same_annotator_cannot_claim_both_translate_slots(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]

    first = db.claim_next_task(conn, "alice")
    assert first.kind == "translate"
    assert first.item_id == item_id

    # alice still has other items' translate tasks available, so claiming
    # again must not hand her the SAME item's second translate slot.
    second = db.claim_next_task(conn, "alice")
    assert not (second is not None and second.item_id == item_id and second.kind == "translate")


def test_translator_cannot_review_own_item(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]

    # Force both translate tasks on this item to alice and bob, done.
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()["id"]
    conn.execute(
        "UPDATE tasks SET status='done', annotator_id=? WHERE id=?",
        (alice_id, translate_ids[0]),
    )
    bob_id = conn.execute("SELECT id FROM annotators WHERE name='bob'").fetchone()["id"]
    conn.execute(
        "UPDATE tasks SET status='done', annotator_id=? WHERE id=?",
        (bob_id, translate_ids[1]),
    )
    conn.commit()

    claimed = db.claim_next_task(conn, "alice")
    assert not (claimed is not None and claimed.kind == "review" and claimed.item_id == item_id)


def test_review_not_eligible_until_both_translations_done(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()["id"]
    conn.execute(
        "UPDATE tasks SET status='done', annotator_id=? WHERE id=?",
        (alice_id, translate_ids[0]),
    )
    conn.commit()  # only ONE of the two translate tasks is done

    claimed = db.claim_next_task(conn, "carol")
    assert not (claimed is not None and claimed.kind == "review" and claimed.item_id == item_id)


def test_reviewer_can_adjudicate_own_item_but_translators_cannot(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()["id"]
    bob_id = conn.execute("SELECT id FROM annotators WHERE name='bob'").fetchone()["id"]
    db.submit_translate(conn, translate_ids[0], "translation one", 1.0, [])
    conn.execute("UPDATE tasks SET annotator_id=? WHERE id=?", (alice_id, translate_ids[0]))
    db.submit_translate(conn, translate_ids[1], "translation two totally different", 1.0, [])
    conn.execute("UPDATE tasks SET annotator_id=? WHERE id=?", (bob_id, translate_ids[1]))
    conn.commit()

    review_task = db.claim_next_task(conn, "carol")
    assert review_task.kind == "review"
    created_adjudication = db.submit_review(conn, review_task.id, 10, ["bad"], translate_ids[0])
    assert created_adjudication is True  # score=10 < 70 triggers it

    # translators excluded from adjudicating their own item
    alice_claim = db.claim_next_task(conn, "alice")
    assert not (alice_claim is not None and alice_claim.kind == "adjudicate" and alice_claim.item_id == item_id)

    # the reviewer themself IS allowed to adjudicate
    carol_claim = db.claim_next_task(conn, "carol")
    assert carol_claim is not None
    assert carol_claim.kind == "adjudicate"
    assert carol_claim.item_id == item_id


@pytest.mark.parametrize(
    "chrf_below_40,score_below_70,expect_adjudication",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_adjudication_trigger_boolean_combinations(
    tmp_path, monkeypatch, chrf_below_40, score_below_70, expect_adjudication
):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    db.submit_translate(conn, translate_ids[0], "t1", 1.0, [])
    db.submit_translate(conn, translate_ids[1], "t2", 1.0, [])

    monkeypatch.setattr(
        "rinchen.annotate.db.chrf",
        lambda a, b: 30.0 if chrf_below_40 else 90.0,
    )
    score = 50 if score_below_70 else 90
    review_id = conn.execute(
        "SELECT id FROM tasks WHERE item_id=? AND kind='review'", (item_id,)
    ).fetchone()["id"]
    created = db.submit_review(conn, review_id, score, [], translate_ids[0])
    assert created is expect_adjudication


def test_adjudication_trigger_boundary_values_do_not_trigger(tmp_path, monkeypatch):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    db.submit_translate(conn, translate_ids[0], "t1", 1.0, [])
    db.submit_translate(conn, translate_ids[1], "t2", 1.0, [])
    monkeypatch.setattr("rinchen.annotate.db.chrf", lambda a, b: 40.0)
    review_id = conn.execute(
        "SELECT id FROM tasks WHERE item_id=? AND kind='review'", (item_id,)
    ).fetchone()["id"]

    created = db.submit_review(conn, review_id, 70, [], translate_ids[0])

    assert created is False  # exactly chrf==40 and score==70: strict '<', no trigger


def test_submit_translate_writes_row_and_marks_done(tmp_path):
    conn = _setup(tmp_path, ["alice"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    task_id = conn.execute(
        "SELECT id FROM tasks WHERE item_id=? AND kind='translate' LIMIT 1", (item_id,)
    ).fetchone()["id"]

    db.submit_translate(conn, task_id, "hello", 2.5, ["flag1"])

    row = conn.execute("SELECT text, minutes, flags FROM translations WHERE task_id=?", (task_id,)).fetchone()
    assert row["text"] == "hello"
    assert row["minutes"] == 2.5
    assert json.loads(row["flags"]) == ["flag1"]
    status = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()["status"]
    assert status == "done"


def test_submit_adjudicate_writes_translation_row(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    now = conn.execute("SELECT created_at FROM tasks LIMIT 1").fetchone()["created_at"]
    cur = conn.execute(
        "INSERT INTO tasks (item_id, kind, status, created_at) VALUES (?, 'adjudicate', 'pending', ?)",
        (item_id, now),
    )
    adjudicate_id = cur.lastrowid
    conn.commit()

    db.submit_adjudicate(conn, adjudicate_id, "final text", 3.0, [])

    row = conn.execute("SELECT text FROM translations WHERE task_id=?", (adjudicate_id,)).fetchone()
    assert row["text"] == "final text"
    status = conn.execute("SELECT status FROM tasks WHERE id=?", (adjudicate_id,)).fetchone()["status"]
    assert status == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_annotate_lifecycle.py -v`
Expected: FAIL — `claim_next_task`/`submit_*` don't exist yet.

- [ ] **Step 3: Append to `db.py`**

```python
# append to src/rinchen/annotate/db.py

from dataclasses import dataclass

from rinchen.errors import DataError
from rinchen.metrics.bleu_chrf import chrf


@dataclass(frozen=True)
class ClaimedTask:
    id: int
    item_id: int
    kind: str


def _annotator_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM annotators WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise DataError(f"unknown annotator: {name}")
    return row["id"]


def claim_next_task(conn: sqlite3.Connection, annotator_name: str) -> ClaimedTask | None:
    """Claim the oldest-created eligible task for `annotator_name`.

    Eligibility: a 'translate' task on an item this annotator has no other
    task on; a 'review' task only once both the item's translate tasks are
    done and this annotator is not one of the two translators; an
    'adjudicate' task only once created and this annotator is not one of
    the two translators (the reviewer MAY adjudicate their own item).

    Returns None (not an error) when nothing is eligible -- an empty queue
    is a normal condition.
    """
    annotator_id = _annotator_id(conn, annotator_name)

    translate_row = conn.execute(
        """
        SELECT t.id, t.item_id, t.kind, t.created_at
        FROM tasks t
        WHERE t.kind = 'translate' AND t.status = 'pending'
          AND NOT EXISTS (
              SELECT 1 FROM tasks t2
              WHERE t2.item_id = t.item_id AND t2.annotator_id = ?
          )
        ORDER BY t.created_at, t.id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()

    review_row = conn.execute(
        """
        SELECT t.id, t.item_id, t.kind, t.created_at
        FROM tasks t
        WHERE t.kind = 'review' AND t.status = 'pending'
          AND (
              SELECT COUNT(*) FROM tasks tt
              WHERE tt.item_id = t.item_id AND tt.kind = 'translate' AND tt.status = 'done'
          ) = 2
          AND NOT EXISTS (
              SELECT 1 FROM tasks tt
              WHERE tt.item_id = t.item_id AND tt.kind = 'translate' AND tt.annotator_id = ?
          )
        ORDER BY t.created_at, t.id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()

    adjudicate_row = conn.execute(
        """
        SELECT t.id, t.item_id, t.kind, t.created_at
        FROM tasks t
        WHERE t.kind = 'adjudicate' AND t.status = 'pending'
          AND NOT EXISTS (
              SELECT 1 FROM tasks tt
              WHERE tt.item_id = t.item_id AND tt.kind = 'translate' AND tt.annotator_id = ?
          )
        ORDER BY t.created_at, t.id
        LIMIT 1
        """,
        (annotator_id,),
    ).fetchone()

    candidates = [r for r in (translate_row, review_row, adjudicate_row) if r is not None]
    if not candidates:
        return None

    chosen = min(candidates, key=lambda r: (r["created_at"], r["id"]))

    conn.execute(
        "UPDATE tasks SET status='assigned', annotator_id=?, assigned_at=? WHERE id=?",
        (annotator_id, _now(), chosen["id"]),
    )
    conn.commit()
    return ClaimedTask(id=chosen["id"], item_id=chosen["item_id"], kind=chosen["kind"])


def submit_translate(
    conn: sqlite3.Connection, task_id: int, text: str, minutes: float, flags: list[str]
) -> None:
    conn.execute(
        "INSERT INTO translations (task_id, text, minutes, flags) VALUES (?, ?, ?, ?)",
        (task_id, text, minutes, json.dumps(flags)),
    )
    conn.execute(
        "UPDATE tasks SET status='done', completed_at=? WHERE id=?", (_now(), task_id)
    )
    conn.commit()


def submit_review(
    conn: sqlite3.Connection,
    task_id: int,
    score: int,
    errors: list[str],
    chosen_translation_id: int,
) -> bool:
    """Writes the review, marks the task done, and -- if chrF(t1,t2) < 40
    or score < 70 -- creates an 'adjudicate' task. Returns whether one was
    created.
    """
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) "
        "VALUES (?, ?, ?, ?)",
        (task_id, score, json.dumps(errors), chosen_translation_id),
    )
    conn.execute(
        "UPDATE tasks SET status='done', completed_at=? WHERE id=?", (_now(), task_id)
    )

    item_id = conn.execute("SELECT item_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate' ORDER BY id",
            (item_id,),
        ).fetchall()
    ]
    t1_text = conn.execute(
        "SELECT text FROM translations WHERE task_id=?", (translate_ids[0],)
    ).fetchone()[0]
    t2_text = conn.execute(
        "SELECT text FROM translations WHERE task_id=?", (translate_ids[1],)
    ).fetchone()[0]

    needs_adjudication = chrf(t1_text, t2_text) < 40 or score < 70

    if needs_adjudication:
        conn.execute(
            "INSERT INTO tasks (item_id, kind, status, created_at) "
            "VALUES (?, 'adjudicate', 'pending', ?)",
            (item_id, _now()),
        )
    conn.commit()
    return needs_adjudication


def submit_adjudicate(
    conn: sqlite3.Connection, task_id: int, text: str, minutes: float, flags: list[str]
) -> None:
    conn.execute(
        "INSERT INTO translations (task_id, text, minutes, flags) VALUES (?, ?, ?, ?)",
        (task_id, text, minutes, json.dumps(flags)),
    )
    conn.execute(
        "UPDATE tasks SET status='done', completed_at=? WHERE id=?", (_now(), task_id)
    )
    conn.commit()
```

Note: the test file monkeypatches `rinchen.annotate.db.chrf` (the name
`chrf` imported into `db.py`'s own namespace via `from
rinchen.metrics.bleu_chrf import chrf`), not `rinchen.metrics.bleu_chrf.chrf`
directly — `db.py` must use a bare `from ... import chrf` (not `from ...
import bleu_chrf` then `bleu_chrf.chrf(...)`) for that monkeypatch target
to work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_annotate_lifecycle.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/rinchen/annotate/db.py tests/unit/test_annotate_lifecycle.py
git commit -m "D2: add task-claiming/submission engine with adjudication trigger"
```

---

### Task 4: Agreement report (pairwise chrF, Krippendorff's alpha)

**Files:**
- Create: `src/rinchen/annotate/agreement.py`
- Test: `tests/unit/test_annotate_agreement.py`

**Interfaces:**
- Consumes: `chrf` from `rinchen.metrics.bleu_chrf` (Task 2); reads the
  same schema as Task 1/3 but takes a `conn` directly (no dependency on
  Task 3's lifecycle functions).
- Produces: `agreement.krippendorff_alpha(reliability_data: list[list[float
  | None]]) -> float`; `agreement.pairwise_chrf_mean(conn, set_name: str)
  -> float`; `agreement.review_score_alpha(conn, set_name: str) -> float`;
  `agreement.agreement_report(conn, set_name: str) -> dict[str, float]`
  (`{"pairwise_chrf_mean": ..., "krippendorff_alpha": ...}`). Consumed by
  Task 7's CLI `agreement` command.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_annotate_agreement.py
from __future__ import annotations

import math
from pathlib import Path

from rinchen.annotate import db
from rinchen.annotate.agreement import (
    agreement_report,
    krippendorff_alpha,
    pairwise_chrf_mean,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def test_krippendorff_alpha_perfect_agreement_is_one():
    data = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert krippendorff_alpha(data) == 1.0


def test_krippendorff_alpha_matches_hayes_krippendorff_2007_worked_example():
    # Canonical worked example from Hayes & Krippendorff (2007),
    # "Answering the Call for a Standard Reliability Measure for Coding
    # Data", Table 1 (interval metric). Cross-checked directly against the
    # independent `krippendorff` PyPI package in an isolated scratch venv
    # (not a project dependency): `krippendorff.alpha(reliability_data=data,
    # level_of_measurement="interval")` on this exact dataset gives
    # 0.8621041879468846, matching this hand-rolled implementation.
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    ]
    alpha = krippendorff_alpha(data)
    assert math.isclose(alpha, 0.8621, abs_tol=0.001)


def test_pairwise_chrf_mean_on_identical_translations(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.import_items(conn, FIXTURES_DIR / "import_sample.jsonl", "demo")
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    db.submit_translate(conn, translate_ids[0], "same text", 1.0, [])
    db.submit_translate(conn, translate_ids[1], "same text", 1.0, [])

    mean = pairwise_chrf_mean(conn, "demo")

    assert mean == 100.0


def test_agreement_report_returns_both_keys(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    report = agreement_report(conn, "demo")
    assert set(report.keys()) == {"pairwise_chrf_mean", "krippendorff_alpha"}
    # No data yet in this fresh DB -- both are NaN, not an exception.
    assert math.isnan(report["pairwise_chrf_mean"])
    assert math.isnan(report["krippendorff_alpha"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_annotate_agreement.py -v`
Expected: FAIL — `rinchen.annotate.agreement` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# src/rinchen/annotate/agreement.py
"""Agreement statistics: pairwise chrF and Krippendorff's alpha.

Krippendorff's alpha is hand-implemented (not a new dependency) per
CLAUDE.md's build-ourselves precedent. D2 produces exactly one review per
item, so `review_score_alpha` operates on a single-rater vector --
Krippendorff's alpha is formally undefined (NaN) for that shape until a
later milestone captures 2+ independent reviews per item. The function
itself is still the real, tested formula (not a stub), ready for that
future data shape -- see the design spec's "Spec findings" note.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from itertools import permutations

from rinchen.metrics.bleu_chrf import chrf


def krippendorff_alpha(reliability_data: list[list[float | None]]) -> float:
    """Krippendorff's alpha for interval-level data.

    `reliability_data` is a list of rater rows, each row a list of values
    per unit (same length across rows), with None for a missing rating.
    """
    n_raters = len(reliability_data)
    n_units = len(reliability_data[0]) if n_raters else 0

    units: list[list[float]] = []
    for u in range(n_units):
        vals = [
            reliability_data[r][u]
            for r in range(n_raters)
            if reliability_data[r][u] is not None
        ]
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan")

    coincidence: dict[tuple[float, float], float] = defaultdict(float)
    for vals in units:
        weight = 1.0 / (len(vals) - 1)
        for v1, v2 in permutations(vals, 2):
            coincidence[(v1, v2)] += weight

    n = sum(coincidence.values())

    marginal: dict[float, float] = defaultdict(float)
    for (v1, _v2), w in coincidence.items():
        marginal[v1] += w

    def metric(v1: float, v2: float) -> float:
        return (v1 - v2) ** 2

    d_o = sum(w * metric(v1, v2) for (v1, v2), w in coincidence.items()) / n

    d_e_sum = 0.0
    for v1, n_v1 in marginal.items():
        for v2, n_v2 in marginal.items():
            d_e_sum += n_v1 * n_v2 * metric(v1, v2)
    d_e = d_e_sum / (n * (n - 1))

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def pairwise_chrf_mean(conn: sqlite3.Connection, set_name: str) -> float:
    conn.row_factory = sqlite3.Row
    items = conn.execute("SELECT id FROM items WHERE set_name = ?", (set_name,)).fetchall()
    scores = []
    for item in items:
        rows = conn.execute(
            "SELECT tr.text FROM tasks t JOIN translations tr ON tr.task_id = t.id "
            "WHERE t.item_id = ? AND t.kind = 'translate' AND t.status = 'done' "
            "ORDER BY t.id",
            (item["id"],),
        ).fetchall()
        if len(rows) == 2:
            scores.append(chrf(rows[0]["text"], rows[1]["text"]))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def review_score_alpha(conn: sqlite3.Connection, set_name: str) -> float:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT r.score FROM reviews r "
        "JOIN tasks t ON t.id = r.task_id "
        "JOIN items i ON i.id = t.item_id "
        "WHERE i.set_name = ? AND t.status = 'done'",
        (set_name,),
    ).fetchall()
    scores: list[float | None] = [float(r["score"]) for r in rows]
    return krippendorff_alpha([scores])


def agreement_report(conn: sqlite3.Connection, set_name: str) -> dict[str, float]:
    return {
        "pairwise_chrf_mean": pairwise_chrf_mean(conn, set_name),
        "krippendorff_alpha": review_score_alpha(conn, set_name),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_annotate_agreement.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rinchen/annotate/agreement.py tests/unit/test_annotate_agreement.py
git commit -m "D2: add agreement report (pairwise chrF, Krippendorff alpha)"
```

---

### Task 5: Export

**Files:**
- Create: `src/rinchen/annotate/export.py`
- Test: `tests/unit/test_annotate_export.py`

**Interfaces:**
- Consumes: reads the same schema as Task 1/3 directly via `conn` (no
  dependency on Task 3's lifecycle functions for reading, only for
  building its own test fixtures).
- Produces: `export.export_jsonl(conn, set_name: str, out_path: Path) ->
  int`; `export.export_tsv(conn, set_name: str, out_path: Path) -> int`;
  `export.export_parquet(conn, set_name: str, out_path: Path) -> int`
  (each returns the number of rows written). Consumed by Task 7's CLI
  `export` command and Task 8's end-to-end test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_annotate_export.py
from __future__ import annotations

import csv
import json

import polars as pl
import pytest

from rinchen.annotate import db
from rinchen.annotate.export import export_jsonl, export_parquet, export_tsv

_COLUMNS = {
    "src", "tgt", "src_lang", "tgt_lang", "doc_id", "seg_idx",
    "set", "origin", "split",
}


def _build_fixture_db(tmp_path):
    """4 items: review-only resolved, adjudicated, review-pending (skip),
    adjudicate-pending (skip)."""
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    conn.executemany(
        "INSERT INTO items (set_name, src, src_lang, tgt_lang, doc_id, seg_idx) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("demo", "src-review-only", "mn_Cyrl", "en", "d", 0),
            ("demo", "src-adjudicated", "mn_Cyrl", "en", "d", 1),
            ("demo", "src-review-pending", "mn_Cyrl", "en", "d", 2),
            ("demo", "src-adjudicate-pending", "mn_Cyrl", "en", "d", 3),
        ],
    )
    conn.commit()
    item_ids = [
        r["id"] for r in conn.execute("SELECT id FROM items ORDER BY id").fetchall()
    ]

    def add_tasks(item_id, kinds_statuses):
        ids = {}
        for kind, status in kinds_statuses:
            cur = conn.execute(
                "INSERT INTO tasks (item_id, kind, status, created_at) VALUES (?, ?, ?, 'now')",
                (item_id, kind, status),
            )
            ids.setdefault(kind, []).append(cur.lastrowid)
        return ids

    # Item 0: review-only resolved.
    ids0 = add_tasks(item_ids[0], [("translate", "done"), ("translate", "done"), ("review", "done")])
    db.submit_translate(conn, ids0["translate"][0], "translation A", 1.0, [])
    db.submit_translate(conn, ids0["translate"][1], "translation B", 1.0, [])
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) VALUES (?, 90, '[]', ?)",
        (ids0["review"][0], ids0["translate"][0]),
    )

    # Item 1: adjudicated.
    ids1 = add_tasks(
        item_ids[1],
        [("translate", "done"), ("translate", "done"), ("review", "done"), ("adjudicate", "done")],
    )
    db.submit_translate(conn, ids1["translate"][0], "bad translation A", 1.0, [])
    db.submit_translate(conn, ids1["translate"][1], "bad translation B", 1.0, [])
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) VALUES (?, 20, '[]', ?)",
        (ids1["review"][0], ids1["translate"][0]),
    )
    db.submit_adjudicate(conn, ids1["adjudicate"][0], "final adjudicated text", 2.0, [])

    # Item 2: review still pending -- must be skipped.
    add_tasks(item_ids[2], [("translate", "done"), ("translate", "done"), ("review", "pending")])

    # Item 3: review done but adjudicate still pending -- must be skipped.
    ids3 = add_tasks(
        item_ids[3],
        [("translate", "done"), ("translate", "done"), ("review", "done"), ("adjudicate", "pending")],
    )
    db.submit_translate(conn, ids3["translate"][0], "translation C", 1.0, [])
    db.submit_translate(conn, ids3["translate"][1], "translation D", 1.0, [])
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) VALUES (?, 10, '[]', ?)",
        (ids3["review"][0], ids3["translate"][0]),
    )

    conn.commit()
    return conn


def test_export_jsonl_skips_unresolved_and_picks_right_text(tmp_path):
    conn = _build_fixture_db(tmp_path)
    out = tmp_path / "out.jsonl"

    count = export_jsonl(conn, "demo", out)

    assert count == 2
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    by_src = {row["src"]: row for row in lines}
    assert by_src["src-review-only"]["tgt"] == "translation A"
    assert by_src["src-adjudicated"]["tgt"] == "final adjudicated text"
    for row in lines:
        assert set(row.keys()) == _COLUMNS
        assert row["origin"] == "human"
        assert row["split"] == "test"


def test_export_tsv_has_right_row_count(tmp_path):
    conn = _build_fixture_db(tmp_path)
    out = tmp_path / "out.tsv"

    count = export_tsv(conn, "demo", out)

    assert count == 2
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 2


def test_export_parquet_round_trips_via_polars(tmp_path):
    conn = _build_fixture_db(tmp_path)
    out = tmp_path / "out.parquet"

    count = export_parquet(conn, "demo", out)

    assert count == 2
    df = pl.read_parquet(out)
    assert df.height == 2
    assert set(df.columns) == _COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_annotate_export.py -v`
Expected: FAIL — `rinchen.annotate.export` doesn't exist yet.

- [ ] **Step 3: Implement**

```python
# src/rinchen/annotate/export.py
"""Export resolved annotation items into the project's corpus schema."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import polars as pl

_COLUMNS = [
    "src", "tgt", "src_lang", "tgt_lang", "doc_id", "seg_idx",
    "set", "origin", "split",
]


def _resolved_rows(conn: sqlite3.Connection, set_name: str) -> list[dict]:
    """One row per fully-resolved item in `set_name`.

    Final text = the adjudicator's translation if an 'adjudicate' task
    exists and is done for that item; else the review's
    chosen_translation_id translation. An item with no review done yet, or
    with a pending/assigned adjudicate task, is skipped (not an error --
    exporting mid-annotation is a normal partial-progress operation).
    """
    conn.row_factory = sqlite3.Row
    items = conn.execute(
        "SELECT id, src, src_lang, tgt_lang, doc_id, seg_idx, set_name "
        "FROM items WHERE set_name = ?",
        (set_name,),
    ).fetchall()

    rows: list[dict] = []
    for item in items:
        item_id = item["id"]

        review = conn.execute(
            "SELECT r.chosen_translation_id FROM reviews r "
            "JOIN tasks t ON t.id = r.task_id "
            "WHERE t.item_id = ? AND t.status = 'done'",
            (item_id,),
        ).fetchone()
        if review is None:
            continue

        adjudicate_task = conn.execute(
            "SELECT id, status FROM tasks WHERE item_id = ? AND kind = 'adjudicate'",
            (item_id,),
        ).fetchone()

        if adjudicate_task is not None:
            if adjudicate_task["status"] != "done":
                continue
            final_task_id = adjudicate_task["id"]
        else:
            final_task_id = review["chosen_translation_id"]

        final = conn.execute(
            "SELECT text FROM translations WHERE task_id = ?", (final_task_id,)
        ).fetchone()
        assert final is not None, f"missing translation for task {final_task_id}"

        rows.append(
            {
                "src": item["src"],
                "tgt": final["text"],
                "src_lang": item["src_lang"],
                "tgt_lang": item["tgt_lang"],
                "doc_id": item["doc_id"],
                "seg_idx": item["seg_idx"],
                "set": item["set_name"],
                "origin": "human",
                "split": "test",
            }
        )
    return rows


def export_jsonl(conn: sqlite3.Connection, set_name: str, out_path: Path) -> int:
    rows = _resolved_rows(conn, set_name)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_tsv(conn: sqlite3.Connection, set_name: str, out_path: Path) -> int:
    rows = _resolved_rows(conn, set_name)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_parquet(conn: sqlite3.Connection, set_name: str, out_path: Path) -> int:
    rows = _resolved_rows(conn, set_name)
    df = (
        pl.DataFrame(rows, schema=_COLUMNS)
        if rows
        else pl.DataFrame(schema={c: pl.Utf8 for c in _COLUMNS})
    )
    df.write_parquet(out_path)
    return len(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_annotate_export.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rinchen/annotate/export.py tests/unit/test_annotate_export.py
git commit -m "D2: add export (jsonl/tsv/parquet) with adjudication-aware text selection"
```

---

### Task 6: Web app (FastAPI + Jinja2 + vendored htmx)

**Files:**
- Create: `src/rinchen/annotate/routes.py`
- Create: `src/rinchen/annotate/app.py`
- Create: `src/rinchen/annotate/templates/base.html`
- Create: `src/rinchen/annotate/templates/index.html`
- Create: `src/rinchen/annotate/templates/partials/no_task.html`
- Create: `src/rinchen/annotate/templates/partials/submitted.html`
- Create: `src/rinchen/annotate/templates/partials/translate_task.html`
- Create: `src/rinchen/annotate/templates/partials/review_task.html`
- Create: `src/rinchen/annotate/templates/partials/adjudicate_task.html`
- Create: `src/rinchen/annotate/static/htmx.min.js` (vendored, via `curl`,
  not hand-written)
- Test: `tests/unit/test_annotate_app.py`

**Interfaces:**
- Consumes: `db.connect`, `db.claim_next_task`, `db.submit_translate`,
  `db.submit_review`, `db.submit_adjudicate` (Task 3); `DataError` from
  `rinchen.errors`.
- Produces: `app.create_app(db_path: Path) -> FastAPI`. Consumed by Task
  7's CLI `serve` command and Task 8's end-to-end test.

- [ ] **Step 1: Vendor htmx**

```bash
curl -sL -o src/rinchen/annotate/static/htmx.min.js \
  "https://unpkg.com/htmx.org@2.0.11/dist/htmx.min.js"
```

Verify (pre-confirmed during this plan's pre-verification pass — the
command above must reproduce this exact file):

```bash
wc -c src/rinchen/annotate/static/htmx.min.js   # expect 52182
sha256sum src/rinchen/annotate/static/htmx.min.js
# expect d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_annotate_app.py
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rinchen.annotate import db
from rinchen.annotate.app import create_app

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "rinchen" / "annotate" / "static"


def _client(tmp_path, annotators=("alice", "bob", "carol")):
    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, list(annotators))
    conn.close()
    app = create_app(db_path)
    return TestClient(app)


def test_htmx_is_vendored_correctly():
    htmx_path = STATIC_DIR / "htmx.min.js"
    content = htmx_path.read_bytes()
    assert len(content) == 52182
    assert hashlib.sha256(content).hexdigest() == (
        "d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717"
    )
    assert content.startswith(b"var htmx=")


def test_healthz(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_index_lists_annotators(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "alice" in resp.text


def test_next_task_returns_no_task_partial_when_empty(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/annotator/alice/next")
    assert resp.status_code == 200
    assert "Nothing to do" in resp.text or "nothing" in resp.text.lower()


def test_next_task_unknown_annotator_is_404(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/annotator/nobody/next")
    assert resp.status_code == 404


def test_full_translate_review_flow_via_testclient(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, ["alice", "bob", "carol"])
    item_id_cur = conn.execute(
        "INSERT INTO items (set_name, src, src_lang, tgt_lang, doc_id, seg_idx) "
        "VALUES ('demo', 'hello', 'mn_Cyrl', 'en', 'd', 0)"
    )
    item_id = item_id_cur.lastrowid
    now = "2026-01-01T00:00:00"
    conn.execute(
        "INSERT INTO tasks (item_id, kind, status, created_at) VALUES (?, 'translate', 'pending', ?)",
        (item_id, now),
    )
    conn.execute(
        "INSERT INTO tasks (item_id, kind, status, created_at) VALUES (?, 'translate', 'pending', ?)",
        (item_id, now),
    )
    conn.execute(
        "INSERT INTO tasks (item_id, kind, status, created_at) VALUES (?, 'review', 'pending', ?)",
        (item_id, now),
    )
    conn.commit()
    conn.close()

    client = TestClient(create_app(db_path))

    r1 = client.get("/annotator/alice/next")
    assert r1.status_code == 200
    # extract the task id from the rendered form action -- the template
    # posts to /task/{id}/translate
    import re
    task1_id = int(re.search(r"/task/(\d+)/translate", r1.text).group(1))
    resp = client.post(
        f"/task/{task1_id}/translate", data={"text": "translation A", "minutes": "1.5"}
    )
    assert resp.status_code == 200

    r2 = client.get("/annotator/bob/next")
    task2_id = int(re.search(r"/task/(\d+)/translate", r2.text).group(1))
    resp = client.post(
        f"/task/{task2_id}/translate", data={"text": "translation B", "minutes": "1.0"}
    )
    assert resp.status_code == 200

    r3 = client.get("/annotator/carol/next")
    assert "review" in r3.text.lower() or "translation A" in r3.text
    review_task_id = int(re.search(r"/task/(\d+)/review", r3.text).group(1))
    resp = client.post(
        f"/task/{review_task_id}/review",
        data={"score": "90", "chosen_translation_id": str(task1_id), "errors": ""},
    )
    assert resp.status_code == 200
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_annotate_app.py -v`
Expected: FAIL — `rinchen.annotate.app`/`routes` don't exist yet (the
`test_htmx_is_vendored_correctly` test should already PASS if Step 1 was
done correctly — if it fails, redo Step 1 before continuing).

- [ ] **Step 4: Implement templates**

`src/rinchen/annotate/templates/base.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rinchen Annotate</title>
  <script src="/static/htmx.min.js"></script>
</head>
<body>
  <div id="main">
    {% block content %}{% endblock %}
  </div>
</body>
</html>
```

`src/rinchen/annotate/templates/index.html`:
```html
{% extends "base.html" %}
{% block content %}
<h1>Pick annotator</h1>
<ul>
  {% for name in names %}
  <li><a href="/annotator/{{ name }}/next"
         hx-get="/annotator/{{ name }}/next"
         hx-target="#main"
         hx-swap="innerHTML">{{ name }}</a></li>
  {% endfor %}
</ul>
{% endblock %}
```

`src/rinchen/annotate/templates/partials/no_task.html`:
```html
<div><p>Nothing to do right now.</p></div>
```

`src/rinchen/annotate/templates/partials/submitted.html`:
```html
<div>
  <p>Submitted.</p>
  {% if needs_adjudication %}
  <p>This item now needs adjudication.</p>
  {% endif %}
</div>
```

`src/rinchen/annotate/templates/partials/translate_task.html`:
```html
<div>
  <h2>Translate</h2>
  {% if is_vertical %}
  <button type="button" onclick="this.nextElementSibling.classList.toggle('vertical')">Toggle direction</button>
  <div class="src-text vertical" style="writing-mode: vertical-lr;">{{ src }}</div>
  {% else %}
  <div class="src-text">{{ src }}</div>
  {% endif %}
  <form hx-post="/task/{{ task_id }}/translate" hx-target="#main" hx-swap="innerHTML">
    <textarea name="text" required></textarea>
    <input type="number" name="minutes" step="0.1" required>
    <button type="submit">Submit</button>
  </form>
</div>
```

`src/rinchen/annotate/templates/partials/review_task.html`:
```html
<div>
  <h2>Review</h2>
  {% if is_vertical %}
  <div class="src-text vertical" style="writing-mode: vertical-lr;">{{ src }}</div>
  {% else %}
  <div class="src-text">{{ src }}</div>
  {% endif %}
  <form hx-post="/task/{{ task_id }}/review" hx-target="#main" hx-swap="innerHTML">
    {% for choice in choices %}
    <div>
      <label>
        <input type="radio" name="chosen_translation_id" value="{{ choice['task_id'] }}" required>
        {{ choice['text'] }}
      </label>
    </div>
    {% endfor %}
    <label>Score (0-100): <input type="number" name="score" min="0" max="100" required></label>
    <label>Errors (comma-separated): <input type="text" name="errors"></label>
    <button type="submit">Submit</button>
  </form>
</div>
```

`src/rinchen/annotate/templates/partials/adjudicate_task.html`:
```html
<div>
  <h2>Adjudicate</h2>
  {% if is_vertical %}
  <div class="src-text vertical" style="writing-mode: vertical-lr;">{{ src }}</div>
  {% else %}
  <div class="src-text">{{ src }}</div>
  {% endif %}
  <p>Translation candidates:</p>
  <ul>
    {% for choice in choices %}
    <li>{{ choice['text'] }}</li>
    {% endfor %}
  </ul>
  <form hx-post="/task/{{ task_id }}/adjudicate" hx-target="#main" hx-swap="innerHTML">
    <textarea name="text" required></textarea>
    <input type="number" name="minutes" step="0.1" required>
    <button type="submit">Submit final translation</button>
  </form>
</div>
```

- [ ] **Step 5: Implement `routes.py`**

```python
# src/rinchen/annotate/routes.py
"""HTTP routes for the annotation app, over db.py's lifecycle engine."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rinchen.annotate import db
from rinchen.errors import DataError

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = db.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _item_row(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row:
    return conn.execute(
        "SELECT src, src_lang, tgt_lang FROM items WHERE id = ?", (item_id,)
    ).fetchone()


def _translation_choices(conn: sqlite3.Connection, item_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT t.id AS task_id, tr.text AS text
        FROM tasks t JOIN translations tr ON tr.task_id = t.id
        WHERE t.item_id = ? AND t.kind = 'translate'
        ORDER BY t.id
        """,
        (item_id,),
    ).fetchall()


@router.get("/", response_class=HTMLResponse)
def index(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> HTMLResponse:
    rows = conn.execute("SELECT name FROM annotators ORDER BY name").fetchall()
    names = [row["name"] for row in rows]
    return templates.TemplateResponse(request=request, name="index.html", context={"names": names})


@router.get("/annotator/{name}/next", response_class=HTMLResponse)
def next_task(
    request: Request, name: str, conn: sqlite3.Connection = Depends(get_db)
) -> HTMLResponse:
    try:
        claimed = db.claim_next_task(conn, name)
    except DataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if claimed is None:
        return templates.TemplateResponse(request=request, name="partials/no_task.html", context={})

    item = _item_row(conn, claimed.item_id)
    is_vertical = item["src_lang"] == "mn_Mong" or item["tgt_lang"] == "mn_Mong"

    if claimed.kind == "translate":
        return templates.TemplateResponse(
            request=request,
            name="partials/translate_task.html",
            context={"task_id": claimed.id, "src": item["src"], "is_vertical": is_vertical},
        )

    choices = _translation_choices(conn, claimed.item_id)
    template_name = (
        "partials/review_task.html" if claimed.kind == "review" else "partials/adjudicate_task.html"
    )
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "task_id": claimed.id,
            "src": item["src"],
            "is_vertical": is_vertical,
            "choices": choices,
        },
    )


@router.post("/task/{task_id}/translate", response_class=HTMLResponse)
def submit_translate(
    request: Request,
    task_id: int,
    text: str = Form(...),
    minutes: float = Form(...),
    flags: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    db.submit_translate(conn, task_id, text, minutes, flag_list)
    return templates.TemplateResponse(request=request, name="partials/submitted.html", context={})


@router.post("/task/{task_id}/review", response_class=HTMLResponse)
def submit_review(
    request: Request,
    task_id: int,
    score: int = Form(...),
    chosen_translation_id: int = Form(...),
    errors: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    error_list = [e.strip() for e in errors.split(",") if e.strip()]
    needs_adjudication = db.submit_review(conn, task_id, score, error_list, chosen_translation_id)
    return templates.TemplateResponse(
        request=request,
        name="partials/submitted.html",
        context={"needs_adjudication": needs_adjudication},
    )


@router.post("/task/{task_id}/adjudicate", response_class=HTMLResponse)
def submit_adjudicate(
    request: Request,
    task_id: int,
    text: str = Form(...),
    minutes: float = Form(...),
    flags: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    db.submit_adjudicate(conn, task_id, text, minutes, flag_list)
    return templates.TemplateResponse(request=request, name="partials/submitted.html", context={})
```

- [ ] **Step 6: Implement `app.py`**

```python
# src/rinchen/annotate/app.py
"""FastAPI app factory for the annotation app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rinchen.annotate.routes import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI()
    app.state.db_path = db_path
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_annotate_app.py -v`
Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add src/rinchen/annotate/routes.py src/rinchen/annotate/app.py \
        src/rinchen/annotate/templates src/rinchen/annotate/static \
        tests/unit/test_annotate_app.py
git commit -m "D2: add FastAPI/Jinja2/htmx web app over the task-lifecycle engine"
```

---

### Task 7: CLI wiring + stale directory removal

**Files:**
- Modify: `src/rinchen/cli/annotate.py` (replace the D0 stub)
- Test: `tests/unit/test_cli_annotate.py`
- Remove: `annotation_app/` (top-level directory, just a `.gitkeep`)
- Modify: `CLAUDE.md` (drop `annotation_app/` from the repo layout block)

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 4, 5, 6.
- Produces: the 5 `rinchen annotate` subcommands, consumed by Task 8's
  end-to-end test and by human operators per the runbook.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_cli_annotate.py
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rinchen.cli.main import app

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def test_init_creates_db_and_seeds_annotators(tmp_path):
    db_path = tmp_path / "test.sqlite"
    result = runner.invoke(
        app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice,bob"]
    )
    assert result.exit_code == 0, result.output
    assert db_path.exists()


def test_import_reports_count(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice,bob"])
    result = runner.invoke(
        app,
        [
            "annotate", "import",
            "--db", str(db_path),
            "--file", str(FIXTURES_DIR / "import_sample.jsonl"),
            "--set", "demo",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3" in result.output


def test_export_unknown_format_is_usage_error(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice"])
    result = runner.invoke(
        app,
        [
            "annotate", "export",
            "--db", str(db_path),
            "--set", "demo",
            "--format", "xml",
            "--out", str(db_path.parent / "out.xml"),
        ],
    )
    assert result.exit_code == 2


def test_agreement_reports_na_on_empty_set(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice"])
    result = runner.invoke(app, ["annotate", "agreement", "--db", str(db_path), "--set", "demo"])
    assert result.exit_code == 0, result.output
    assert "N/A" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_annotate.py -v`
Expected: FAIL — stub `annotate.py` has no real commands yet.

- [ ] **Step 3: Implement `cli/annotate.py`**

```python
# src/rinchen/cli/annotate.py
"""`rinchen annotate`: human test-set annotation app."""

from __future__ import annotations

import math
from pathlib import Path

import typer
import uvicorn

from rinchen.annotate import db
from rinchen.annotate import export as export_mod
from rinchen.annotate.agreement import agreement_report
from rinchen.annotate.app import create_app

app = typer.Typer(no_args_is_help=True)


@app.command()
def init(
    db_path: Path = typer.Option(..., "--db", help="SQLite database file."),
    annotators: str = typer.Option(..., "--annotators", help="Comma-separated annotator names."),
) -> None:
    names = [n.strip() for n in annotators.split(",") if n.strip()]
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.seed_annotators(conn, names)
    finally:
        conn.close()
    typer.echo(f"initialized {db_path} with {len(names)} annotators")


@app.command(name="import")
def import_(
    db_path: Path = typer.Option(..., "--db"),
    file: Path = typer.Option(..., "--file"),
    set_: str = typer.Option(..., "--set"),
) -> None:
    conn = db.connect(db_path)
    try:
        count = db.import_items(conn, file, set_)
    finally:
        conn.close()
    typer.echo(f"imported {count} items into set '{set_}'")


@app.command()
def serve(
    db_path: Path = typer.Option(..., "--db"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    fastapi_app = create_app(db_path)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command(name="export")
def export_cmd(
    db_path: Path = typer.Option(..., "--db"),
    set_: str = typer.Option(..., "--set"),
    format: str = typer.Option(..., "--format"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    exporters = {
        "jsonl": export_mod.export_jsonl,
        "tsv": export_mod.export_tsv,
        "parquet": export_mod.export_parquet,
    }
    if format not in exporters:
        typer.echo(f"unknown format: {format} (expected one of {sorted(exporters)})", err=True)
        raise typer.Exit(code=2)
    conn = db.connect(db_path)
    try:
        count = exporters[format](conn, set_, out)
    finally:
        conn.close()
    typer.echo(f"exported {count} items to {out}")


@app.command()
def agreement(
    db_path: Path = typer.Option(..., "--db"),
    set_: str = typer.Option(..., "--set"),
) -> None:
    conn = db.connect(db_path)
    try:
        report = agreement_report(conn, set_)
    finally:
        conn.close()

    chrf_mean = report["pairwise_chrf_mean"]
    if math.isnan(chrf_mean):
        typer.echo("pairwise chrF: N/A (no resolved pairs yet)")
    else:
        typer.echo(f"pairwise chrF: {chrf_mean:.2f}")

    alpha = report["krippendorff_alpha"]
    if math.isnan(alpha):
        typer.echo("Krippendorff alpha: N/A (needs 2+ independent reviews per item)")
    else:
        typer.echo(f"Krippendorff alpha: {alpha:.3f}")
```

Note: `export` is shadowed as a Typer command name via
`@app.command(name="export")` on a differently-named Python function
(`export_cmd`), since this module already does `from rinchen.annotate import
export as export_mod` — naming the command function `export` directly
would collide with that import.

- [ ] **Step 4: Remove the stale `annotation_app/` directory**

```bash
git rm annotation_app/.gitkeep
rmdir annotation_app 2>/dev/null || true
```

Then edit `CLAUDE.md`'s repo layout code block (`## Repo layout` section)
to delete the line `tests/ scripts/ plans/ reports/ manifests/
annotation_app/` → `tests/ scripts/ plans/ reports/ manifests/` (drop only
the `annotation_app/` token, keep the rest of that line).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_annotate.py -v`
Expected: 4 passed.

Also run the full suite once here to catch any import-time regressions
from the stale-directory removal:

Run: `uv run pytest -q -m "not network"`
Expected: all passing (no change in count from removing an empty,
untracked-by-tests directory).

- [ ] **Step 6: Commit**

```bash
git add src/rinchen/cli/annotate.py tests/unit/test_cli_annotate.py CLAUDE.md
git add -u annotation_app  # stages the deletion
git commit -m "D2: wire rinchen annotate CLI; remove stale annotation_app/ dir"
```

---

### Task 8: End-to-end fixture, STOP gate, runbook, STATE.md

**Files:**
- Create: `tests/fixtures/annot/e2e_20_items.jsonl`
- Test: `tests/unit/test_annotate_e2e.py`
- Create: `docs/runbooks/02_human_testsets.md`
- Modify: `STATE.md` (add D2 paragraph)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: the D2 STOP gate proof, referenced by `STATE.md` and future
  milestones' "D0-D2 done" status.

- [ ] **Step 1: Create the 20-item fixture**

Generate `tests/fixtures/annot/e2e_20_items.jsonl` with 20 lines, same
JSONL shape as `import_sample.jsonl` (Task 1). Mix: 17 ordinary Cyrillic
Mongolian pairs across a few `doc_id`s, plus 3 lines with `src_lang:
"mn_Mong"` using real U+1800–U+18AF codepoints. Per CLAUDE.md's permanent
"NEVER: Use galig/Latin informal Mongolian" rule, the `mn_Cyrl` lines must
be genuine Cyrillic text (Unicode block U+0400–U+04FF), not a Latin
transliteration mislabeled as Cyrillic -- an earlier version of this
fixture violated that rule and was corrected (see Finding 4 of the D2
whole-branch review, DECISIONS.md). The generation script builds BOTH the
Cyrillic and the traditional-script text via `chr()` on integer codepoints
rather than embedding a literal non-ASCII character directly in the shell
command text, so there's no non-ASCII byte in the command line itself to
risk shell/console encoding mangling on Windows, and writes the file with
`ensure_ascii=True` so the committed JSONL holds `\uXXXX` escapes for both
scripts (consistent with `import_sample.jsonl`'s style) rather than raw
UTF-8 bytes:

```bash
uv run python -c "
import json
cyr_chars = ''.join(chr(c) for c in (0x0421, 0x0430, 0x0439, 0x043D, 0x0020, 0x0431, 0x0430, 0x0439, 0x043D, 0x0430, 0x0020, 0x0443, 0x0443))
mong_chars = ''.join(chr(c) for c in (0x1820, 0x1821, 0x1828, 0x182C, 0x1860, 0x180B, 0x202F))
lines = []
for i in range(17):
    lines.append({'src': f'{cyr_chars} {i}', 'src_lang': 'mn_Cyrl', 'tgt_lang': 'en', 'doc_id': f'doc{i % 4}', 'seg_idx': i})
for i in range(3):
    lines.append({'src': mong_chars + str(i), 'src_lang': 'mn_Mong', 'tgt_lang': 'en', 'doc_id': 'doc_mong', 'seg_idx': i})
with open('tests/fixtures/annot/e2e_20_items.jsonl', 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=True) + chr(10))
"
```

- [ ] **Step 2: Write the failing end-to-end test**

```python
# tests/unit/test_annotate_e2e.py
"""D2 STOP gate: a 20-item fixture can be fully annotated by a scripted
client (FastAPI TestClient) and exported, with zero items left pending."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from rinchen.annotate import db
from rinchen.annotate.app import create_app
from rinchen.annotate.export import export_jsonl

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"
ANNOTATORS = ["alice", "bob", "carol"]


def _run_round(client: TestClient, name: str, action: str) -> bool:
    """Claims and submits one task for `name`. Returns True if a task was
    available and submitted, False if the queue was empty for them."""
    resp = client.get(f"/annotator/{name}/next")
    if "Nothing to do" in resp.text:
        return False

    translate_match = re.search(r"/task/(\d+)/translate", resp.text)
    review_match = re.search(r"/task/(\d+)/review", resp.text)
    adjudicate_match = re.search(r"/task/(\d+)/adjudicate", resp.text)

    if translate_match:
        task_id = translate_match.group(1)
        client.post(f"/task/{task_id}/translate", data={"text": f"{action} by {name}", "minutes": "1.0"})
    elif review_match:
        task_id = review_match.group(1)
        choice_id = re.search(r'value="(\d+)"', resp.text).group(1)
        client.post(
            f"/task/{task_id}/review",
            data={"score": "95", "chosen_translation_id": choice_id, "errors": ""},
        )
    elif adjudicate_match:
        task_id = adjudicate_match.group(1)
        client.post(f"/task/{task_id}/adjudicate", data={"text": f"adjudicated by {name}", "minutes": "2.0"})
    else:
        raise AssertionError(f"unrecognized task page: {resp.text[:200]}")
    return True


def test_20_item_fixture_fully_annotated_and_exported(tmp_path):
    db_path = tmp_path / "e2e.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, ANNOTATORS)
    imported = db.import_items(conn, FIXTURES_DIR / "e2e_20_items.jsonl", "e2e")
    conn.close()
    assert imported == 20

    client = TestClient(create_app(db_path))

    # Round-robin the 3 annotators through the queue until it's empty.
    # Upper bound: 20 items x up to 4 tasks each (2 translate + review +
    # possible adjudicate) = 80 tasks max; 3x that as a generous safety
    # margin against round-robin starvation order.
    max_rounds = 240
    rounds = 0
    idle_streak = 0
    while idle_streak < len(ANNOTATORS) and rounds < max_rounds:
        name = ANNOTATORS[rounds % len(ANNOTATORS)]
        did_work = _run_round(client, name, "work")
        idle_streak = 0 if did_work else idle_streak + 1
        rounds += 1

    conn = db.connect(db_path)
    pending = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE status != 'done'"
    ).fetchone()[0]
    assert pending == 0, f"{pending} tasks still not done after {rounds} rounds"

    out_path = tmp_path / "export.jsonl"
    count = export_jsonl(conn, "e2e", out_path)
    assert count == 20
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `uv run pytest tests/unit/test_annotate_e2e.py -v`
Expected first: FAIL only if any earlier task's code has a real bug this
end-to-end path exercises that the narrower unit tests didn't catch — if
so, fix the root cause in the relevant Task's file (per
`superpowers:systematic-debugging`, not a workaround here). Expected
after: 1 passed.

- [ ] **Step 4: Write the runbook**

```markdown
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

- **Adjudication tasks jump the queue.** `claim_next_task` serves the
  oldest-created eligible task across all three kinds (translate, review,
  adjudicate) for each annotator. If adjudicate tasks pile up unresolved,
  they will sit at the front of every eligible annotator's queue ahead of
  newer translate/review work. Encourage whoever can adjudicate to clear
  that queue promptly rather than letting it block throughput.
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
```

Save as `docs/runbooks/02_human_testsets.md`.

- [ ] **Step 5: Run the full gate**

Run: `uv run ruff format --check . && uv run ruff check . && uv run mypy
src && uv run pytest -q -m "not network"`
Expected: all green. Note the exact pass count for STATE.md.

- [ ] **Step 6: Update `STATE.md`**

Add a new paragraph after the existing D0/D1 paragraph, following the same
style (what was built, STOP gate evidence, exact commands run, pass
counts) — do not overclaim coverage or test counts; state exactly what was
verified, matching the discipline established in D1's post-hoc review
fixes earlier this project.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/annot/e2e_20_items.jsonl tests/unit/test_annotate_e2e.py \
        docs/runbooks/02_human_testsets.md STATE.md
git commit -m "D2: add 20-item e2e fixture (STOP gate), runbook, STATE.md update"
```
