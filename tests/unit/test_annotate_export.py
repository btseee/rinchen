# tests/unit/test_annotate_export.py
from __future__ import annotations

import csv
import json

import polars as pl

from rinchen.annotate import db
from rinchen.annotate.export import export_jsonl, export_parquet, export_tsv

_COLUMNS = {
    "src",
    "tgt",
    "src_lang",
    "tgt_lang",
    "doc_id",
    "seg_idx",
    "set",
    "origin",
    "split",
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
    ids0 = add_tasks(
        item_ids[0], [("translate", "done"), ("translate", "done"), ("review", "done")]
    )
    db.submit_translate(conn, ids0["translate"][0], "translation A", 1.0, [])
    db.submit_translate(conn, ids0["translate"][1], "translation B", 1.0, [])
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) VALUES (?, 90, '[]', ?)",
        (ids0["review"][0], ids0["translate"][0]),
    )

    # Item 1: adjudicated.
    ids1 = add_tasks(
        item_ids[1],
        [
            ("translate", "done"),
            ("translate", "done"),
            ("review", "done"),
            ("adjudicate", "done"),
        ],
    )
    db.submit_translate(conn, ids1["translate"][0], "bad translation A", 1.0, [])
    db.submit_translate(conn, ids1["translate"][1], "bad translation B", 1.0, [])
    conn.execute(
        "INSERT INTO reviews (task_id, score, errors, chosen_translation_id) VALUES (?, 20, '[]', ?)",
        (ids1["review"][0], ids1["translate"][0]),
    )
    db.submit_adjudicate(conn, ids1["adjudicate"][0], "final adjudicated text", 2.0, [])

    # Item 2: review still pending -- must be skipped.
    add_tasks(
        item_ids[2],
        [("translate", "done"), ("translate", "done"), ("review", "pending")],
    )

    # Item 3: review done but adjudicate still pending -- must be skipped.
    ids3 = add_tasks(
        item_ids[3],
        [
            ("translate", "done"),
            ("translate", "done"),
            ("review", "done"),
            ("adjudicate", "pending"),
        ],
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


def test_export_parquet_empty_and_populated_schemas_match(tmp_path):
    # Regression: an empty export (e.g. exporting a set with nothing
    # resolved yet) must have the exact same column dtypes as a populated
    # export, not just the same column names -- otherwise concatenating or
    # type-checking across multiple per-set exports breaks the first time
    # one of them happens to be empty.
    conn = _build_fixture_db(tmp_path)

    populated_out = tmp_path / "populated.parquet"
    export_parquet(conn, "demo", populated_out)
    populated_schema = pl.read_parquet(populated_out).schema

    empty_out = tmp_path / "empty.parquet"
    count = export_parquet(conn, "nonexistent-set", empty_out)
    assert count == 0
    empty_df = pl.read_parquet(empty_out)
    assert empty_df.height == 0
    assert empty_df.schema == populated_schema
    assert empty_df.schema["seg_idx"] == pl.Int64
