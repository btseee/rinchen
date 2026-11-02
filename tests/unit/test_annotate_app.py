from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from rinchen.annotate import db
from rinchen.annotate.app import create_app

STATIC_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "rinchen"
    / "annotate"
    / "static"
)


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


def test_connection_can_be_closed_from_a_different_thread(tmp_path):
    # Regression for Finding 1: sqlite3.connect() defaults to
    # check_same_thread=True, which raises sqlite3.ProgrammingError if the
    # connection is used/closed from a thread other than the one that
    # created it. FastAPI's sync-generator dependency (get_db in routes.py)
    # can run its "yield" (dependency-enter, on the request's threadpool
    # thread) and its "finally: conn.close()" (dependency-exit) on
    # different threadpool threads for the same logical request, so
    # db.connect() must pass check_same_thread=False. This test proves the
    # mechanism directly, independent of FastAPI's actual thread scheduling.
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)

    errors: list[BaseException] = []

    def _close_elsewhere():
        try:
            conn.execute("SELECT 1").fetchone()
            conn.close()
        except BaseException as exc:  # noqa: BLE001 -- capture for the assertion below
            errors.append(exc)

    t = threading.Thread(target=_close_elsewhere)
    t.start()
    t.join()

    assert errors == []


def test_concurrent_requests_do_not_return_500(tmp_path):
    # Regression for Finding 1: fires a burst of concurrent requests at a
    # real TestClient-backed app and asserts none of them come back as a
    # 500. TestClient's own threading is more synchronous than a real
    # deployed server's threadpool, so this cannot perfectly reproduce the
    # original crash, but it's a real concurrency smoke test on top of the
    # deterministic thread-close test above, which directly proves the
    # fix's mechanism.
    client = _client(tmp_path)

    def _hit(_i: int) -> int:
        resp = client.get("/healthz")
        return resp.status_code

    with ThreadPoolExecutor(max_workers=10) as pool:
        statuses = list(pool.map(_hit, range(30)))

    assert all(status == 200 for status in statuses), statuses


def test_submit_translate_advances_to_next_task_instead_of_dead_end(tmp_path):
    # Regression for Finding 8: submitting a task is not a dead end -- the
    # response re-renders the annotator's NEXT task (claim_next_task called
    # again server-side) instead of the static "Submitted." partial, so an
    # annotator can keep working without extra navigation.
    import re

    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, ["alice", "bob"])
    import_sample = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "annot"
        / "import_sample.jsonl"
    )
    db.import_items(conn, import_sample, "demo")
    conn.close()

    client = TestClient(create_app(db_path))

    r1 = client.get("/annotator/alice/next")
    task1_id = int(re.search(r"/task/(\d+)/translate", r1.text).group(1))
    resp = client.post(
        f"/task/{task1_id}/translate", data={"text": "translation A", "minutes": "1.5"}
    )
    assert resp.status_code == 200
    assert "Submitted." not in resp.text
    # Alice still has other items' translate slots available -- the
    # response should be the next task's own form, not a dead end.
    assert re.search(r"/task/(\d+)/translate", resp.text) is not None


def test_submit_translate_renders_no_task_partial_when_queue_empty(tmp_path):
    import re

    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)
    db.seed_annotators(conn, ["alice", "bob"])
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
    conn.commit()
    conn.close()

    client = TestClient(create_app(db_path))

    r1 = client.get("/annotator/alice/next")
    task1_id = int(re.search(r"/task/(\d+)/translate", r1.text).group(1))
    resp = client.post(
        f"/task/{task1_id}/translate", data={"text": "translation A", "minutes": "1.5"}
    )
    assert resp.status_code == 200
    assert "Submitted." not in resp.text
    # Alice's only other task is the SAME item's second translate slot,
    # which she's not eligible for -- nothing else is left for her.
    assert "Nothing to do" in resp.text


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
