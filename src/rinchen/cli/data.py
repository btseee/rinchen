"""`rinchen data`: fetch and list registry sources."""

from __future__ import annotations

from pathlib import Path

import typer

from rinchen.acquire.registry import fetch_source, find_entry, load_registry

app = typer.Typer(no_args_is_help=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCES_PATH = REPO_ROOT / "configs" / "data" / "sources.yaml"
SOURCES_TINY_PATH = REPO_ROOT / "configs" / "data" / "sources.tiny.yaml"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"


def _registry_path(profile: str | None) -> Path:
    return SOURCES_TINY_PATH if profile == "tiny" else SOURCES_PATH


@app.command()
def fetch(
    source: str | None = typer.Option(
        None, "--source", help="Fetch a single source by id."
    ),
    all_: bool = typer.Option(False, "--all", help="Fetch every registered source."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Use this profile's registry (e.g. 'tiny' for offline fixtures).",
    ),
) -> None:
    if not source and not all_:
        typer.echo("specify --source <id> or --all", err=True)
        raise typer.Exit(code=2)

    entries = load_registry(_registry_path(profile))
    if all_:
        targets = entries
    else:
        assert source is not None  # guaranteed by the guard above
        targets = [find_entry(entries, source)]

    for entry in targets:
        dest = fetch_source(entry, DATA_RAW_DIR / entry.id)
        typer.echo(f"fetched {entry.id} -> {dest}")


@app.command(name="ls")
def list_sources(
    profile: str | None = typer.Option(
        None, "--profile", help="Use this profile's registry."
    ),
) -> None:
    entries = load_registry(_registry_path(profile))
    for entry in entries:
        typer.echo(f"{entry.id}\t{entry.kind.value}\t{','.join(entry.langs)}")
