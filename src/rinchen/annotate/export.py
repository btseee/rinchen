"""Export resolved annotation items into the project's corpus schema."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import polars as pl

_COLUMNS = [
    "src",
    "tgt",
    "src_lang",
    "tgt_lang",
    "doc_id",
    "seg_idx",
    "set",
    "origin",
    "split",
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
