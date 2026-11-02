# src/rinchen/annotate/db.py
"""SQLite schema, migrations, and the import step for the annotation app.

Task-claiming/submission functions (the assignment engine) are added in a
later task, in this same module.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rinchen.errors import DataError
from rinchen.metrics.bleu_chrf import chrf

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
    return datetime.now(UTC).isoformat()


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


def claim_next_task(
    conn: sqlite3.Connection, annotator_name: str
) -> ClaimedTask | None:
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

    candidates = [
        r for r in (translate_row, review_row, adjudicate_row) if r is not None
    ]
    if not candidates:
        return None

    # Prefer finishing in-flight items (adjudicate, then review) over
    # starting fresh translate work, since a freshly-created adjudicate
    # task is otherwise always newer than untouched translate tasks on
    # other items and would never win a pure created_at comparison.
    _kind_priority = {"adjudicate": 0, "review": 1, "translate": 2}
    chosen = min(
        candidates, key=lambda r: (_kind_priority[r["kind"]], r["created_at"], r["id"])
    )

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

    item_id = conn.execute(
        "SELECT item_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()[0]
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
