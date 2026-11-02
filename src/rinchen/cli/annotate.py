# src/rinchen/cli/annotate.py
"""`rinchen annotate`: human test-set annotation app."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from rinchen.annotate import db
from rinchen.annotate import export as export_mod
from rinchen.annotate.agreement import agreement_report
from rinchen.annotate.app import create_app

app = typer.Typer(no_args_is_help=True)


@app.command()
def init(
    db_path: Annotated[Path, typer.Option("--db", help="SQLite database file.")],
    annotators: Annotated[
        str, typer.Option("--annotators", help="Comma-separated annotator names.")
    ],
) -> None:
    names = [n.strip() for n in annotators.split(",") if n.strip()]
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.seed_annotators(conn, names)
    finally:
        conn.close()
    typer.echo(f"initialized {db_path} with {len(names)} annotators")


@app.command(name="import")
def import_(
    db_path: Annotated[Path, typer.Option("--db")],
    file: Annotated[Path, typer.Option("--file")],
    set_: Annotated[str, typer.Option("--set")],
) -> None:
    conn = db.connect(db_path)
    try:
        count = db.import_items(conn, file, set_)
    finally:
        conn.close()
    typer.echo(f"imported {count} items into set '{set_}'")


@app.command()
def serve(
    db_path: Annotated[Path, typer.Option("--db")],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    fastapi_app = create_app(db_path)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command(name="export")
def export_cmd(
    db_path: Annotated[Path, typer.Option("--db")],
    set_: Annotated[str, typer.Option("--set")],
    format: Annotated[str, typer.Option("--format")],
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    exporters = {
        "jsonl": export_mod.export_jsonl,
        "tsv": export_mod.export_tsv,
        "parquet": export_mod.export_parquet,
    }
    if format not in exporters:
        typer.echo(
            f"unknown format: {format} (expected one of {sorted(exporters)})", err=True
        )
        raise typer.Exit(code=2)
    conn = db.connect(db_path)
    try:
        count = exporters[format](conn, set_, out)
    finally:
        conn.close()
    typer.echo(f"exported {count} items to {out}")


@app.command()
def agreement(
    db_path: Annotated[Path, typer.Option("--db")],
    set_: Annotated[str, typer.Option("--set")],
) -> None:
    conn = db.connect(db_path)
    try:
        report = agreement_report(conn, set_)
    finally:
        conn.close()

    chrf_mean = report["pairwise_chrf_mean"]
    if math.isnan(chrf_mean):
        typer.echo("pairwise chrF: N/A (no resolved pairs yet)")
    else:
        typer.echo(f"pairwise chrF: {chrf_mean:.2f}")

    alpha = report["krippendorff_alpha"]
    if math.isnan(alpha):
        typer.echo("Krippendorff alpha: N/A (needs 2+ independent reviews per item)")
    else:
        typer.echo(f"Krippendorff alpha: {alpha:.3f}")
