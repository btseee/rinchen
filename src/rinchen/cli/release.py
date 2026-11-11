"""CLI sub-app for `rinchen release` (D13): `stage` builds the directory a release uploads, whole, from
an export and the report that measured it (the card, the release record and the copies of the files)."""

import typer

from rinchen.cli.release_stage import stage_command

app = typer.Typer(no_args_is_help=True)

app.command(name="stage")(stage_command)
