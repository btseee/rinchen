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


@router.get("/annotator/{name}/next", response_class=HTMLResponse)
def next_task(
    request: Request,
    name: str,
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    try:
        claimed = db.claim_next_task(conn, name)
    except DataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
    db.submit_translate(conn, task_id, text, minutes, flag_list)
    return templates.TemplateResponse(
        request=request, name="partials/submitted.html", context={}
    )


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
    needs_adjudication = db.submit_review(
        conn, task_id, score, error_list, chosen_translation_id
    )
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
    conn: sqlite3.Connection = Depends(get_db),  # noqa: B008 -- FastAPI DI pattern
) -> HTMLResponse:
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    db.submit_adjudicate(conn, task_id, text, minutes, flag_list)
    return templates.TemplateResponse(
        request=request, name="partials/submitted.html", context={}
    )
