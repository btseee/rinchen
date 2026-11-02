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
def index(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    rows = conn.execute("SELECT name FROM annotators ORDER BY name").fetchall()
    names = [row["name"] for row in rows]
    return templates.TemplateResponse(
        request=request, name="index.html", context={"names": names}
    )


def _render_claimed_or_no_task(
    request: Request, conn: sqlite3.Connection, annotator_name: str
) -> HTMLResponse:
    """Claim the annotator's next task (which may be their current
    in-progress one, per db.claim_next_task's Finding-2 fix) and render its
    partial, or partials/no_task.html if nothing is eligible.

    Shared by GET /annotator/{name}/next and every POST /task/{id}/...
    submit route, so a submission re-renders the annotator's next task
    instead of a dead-end "Submitted." page (Finding 8).
    """
    claimed = db.claim_next_task(conn, annotator_name)

    if claimed is None:
        return templates.TemplateResponse(
            request=request, name="partials/no_task.html", context={}
        )

    item = _item_row(conn, claimed.item_id)
    is_vertical = item["src_lang"] == "mn_Mong" or item["tgt_lang"] == "mn_Mong"

    if claimed.kind == "translate":
        return templates.TemplateResponse(
            request=request,
            name="partials/translate_task.html",
            context={
                "task_id": claimed.id,
                "src": item["src"],
                "is_vertical": is_vertical,
            },
        )

    choices = _translation_choices(conn, claimed.item_id)
    template_name = (
        "partials/review_task.html"
        if claimed.kind == "review"
        else "partials/adjudicate_task.html"
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


def _annotator_name_for_task(conn: sqlite3.Connection, task_id: int) -> str:
    row = conn.execute(
        "SELECT a.name AS name FROM tasks t JOIN annotators a ON a.id = t.annotator_id "
        "WHERE t.id = ?",
        (task_id,),
    ).fetchone()
    return row["name"]


@router.get("/annotator/{name}/next", response_class=HTMLResponse)
def next_task(
    request: Request,
    name: str,
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    try:
        return _render_claimed_or_no_task(request, conn, name)
    except DataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/task/{task_id}/translate", response_class=HTMLResponse)
def submit_translate(
    request: Request,
    task_id: int,
    text: str = Form(...),
    minutes: float = Form(...),
    flags: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    annotator_name = _annotator_name_for_task(conn, task_id)
    db.submit_translate(conn, task_id, text, minutes, flag_list)
    return _render_claimed_or_no_task(request, conn, annotator_name)


@router.post("/task/{task_id}/review", response_class=HTMLResponse)
def submit_review(
    request: Request,
    task_id: int,
    score: int = Form(...),
    chosen_translation_id: int = Form(...),
    errors: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    error_list = [e.strip() for e in errors.split(",") if e.strip()]
    annotator_name = _annotator_name_for_task(conn, task_id)
    db.submit_review(conn, task_id, score, error_list, chosen_translation_id)
    return _render_claimed_or_no_task(request, conn, annotator_name)


@router.post("/task/{task_id}/adjudicate", response_class=HTMLResponse)
def submit_adjudicate(
    request: Request,
    task_id: int,
    text: str = Form(...),
    minutes: float = Form(...),
    flags: str = Form(""),
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    annotator_name = _annotator_name_for_task(conn, task_id)
    db.submit_adjudicate(conn, task_id, text, minutes, flag_list)
    return _render_claimed_or_no_task(request, conn, annotator_name)
