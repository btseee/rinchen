"""CLI sub-app for `rinchen eval`: `robustness` (D12) and `suite` (D13: fetch, build and inspect
an evaluation suite); D13 (Final eval) adds the rest."""

import typer

from rinchen.cli.eval_robustness import robustness_command
from rinchen.cli.eval_suite import suite_app

app = typer.Typer(no_args_is_help=True)

app.command(name="robustness")(robustness_command)
app.add_typer(suite_app, name="suite")
