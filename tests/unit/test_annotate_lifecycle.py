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
    assert not (
        second is not None and second.item_id == item_id and second.kind == "translate"
    )


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
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()[
        "id"
    ]
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
    assert not (
        claimed is not None and claimed.kind == "review" and claimed.item_id == item_id
    )


def test_review_not_eligible_until_both_translations_done(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()[
        "id"
    ]
    conn.execute(
        "UPDATE tasks SET status='done', annotator_id=? WHERE id=?",
        (alice_id, translate_ids[0]),
    )
    conn.commit()  # only ONE of the two translate tasks is done

    claimed = db.claim_next_task(conn, "carol")
    assert not (
        claimed is not None and claimed.kind == "review" and claimed.item_id == item_id
    )


def test_reviewer_can_adjudicate_own_item_but_translators_cannot(tmp_path):
    conn = _setup(tmp_path, ["alice", "bob", "carol"])
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    alice_id = conn.execute("SELECT id FROM annotators WHERE name='alice'").fetchone()[
        "id"
    ]
    bob_id = conn.execute("SELECT id FROM annotators WHERE name='bob'").fetchone()["id"]
    db.submit_translate(conn, translate_ids[0], "translation one", 1.0, [])
    conn.execute(
        "UPDATE tasks SET annotator_id=? WHERE id=?", (alice_id, translate_ids[0])
    )
    db.submit_translate(
        conn, translate_ids[1], "translation two totally different", 1.0, []
    )
    conn.execute(
        "UPDATE tasks SET annotator_id=? WHERE id=?", (bob_id, translate_ids[1])
    )
    conn.commit()

    review_task = db.claim_next_task(conn, "carol")
    assert review_task.kind == "review"
    created_adjudication = db.submit_review(
        conn, review_task.id, 10, ["bad"], translate_ids[0]
    )
    assert created_adjudication is True  # score=10 < 70 triggers it

    # translators excluded from adjudicating their own item
    alice_claim = db.claim_next_task(conn, "alice")
    assert not (
        alice_claim is not None
        and alice_claim.kind == "adjudicate"
        and alice_claim.item_id == item_id
    )

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

    row = conn.execute(
        "SELECT text, minutes, flags FROM translations WHERE task_id=?", (task_id,)
    ).fetchone()
    assert row["text"] == "hello"
    assert row["minutes"] == 2.5
    assert json.loads(row["flags"]) == ["flag1"]
    status = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()[
        "status"
    ]
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

    row = conn.execute(
        "SELECT text FROM translations WHERE task_id=?", (adjudicate_id,)
    ).fetchone()
    assert row["text"] == "final text"
    status = conn.execute(
        "SELECT status FROM tasks WHERE id=?", (adjudicate_id,)
    ).fetchone()["status"]
    assert status == "done"
