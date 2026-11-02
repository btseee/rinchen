# src/rinchen/annotate/db.py
"""SQLite schema, migrations, and the import step for the annotation app.

Task-claiming/submission functions (the assignment engine) are added in a
later task, in this same module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
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
