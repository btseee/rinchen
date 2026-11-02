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
        client.post(
            f"/task/{task_id}/translate",
            data={"text": f"{action} by {name}", "minutes": "1.0"},
        )
    elif review_match:
        task_id = review_match.group(1)
        choice_id = re.search(r'value="(\d+)"', resp.text).group(1)
        client.post(
            f"/task/{task_id}/review",
            data={"score": "95", "chosen_translation_id": choice_id, "errors": ""},
        )
    elif adjudicate_match:
        task_id = adjudicate_match.group(1)
        client.post(
            f"/task/{task_id}/adjudicate",
            data={"text": f"adjudicated by {name}", "minutes": "2.0"},
        )
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
