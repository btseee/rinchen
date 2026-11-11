"""CLI sub-app for `rinchen release` (D13): `stage` builds the directory a release uploads, whole, from
an export and the report that measured it (the card, the release record and the copies of the files);
`hub` plans the upload of that directory to the Hub (a dry run) and, only with `--upload`, makes it."""

import typer

from rinchen.cli.release_hub import hub_command
from rinchen.cli.release_stage import stage_command

app = typer.Typer(no_args_is_help=True)

app.command(name="stage")(stage_command)
app.command(name="hub")(hub_command)
