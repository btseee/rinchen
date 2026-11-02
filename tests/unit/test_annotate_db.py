# tests/unit/test_annotate_db.py
from __future__ import annotations

from pathlib import Path

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
        assert all(
            t["status"] == "pending" and t["annotator_id"] is None for t in tasks
        )


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
