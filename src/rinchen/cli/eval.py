"""CLI sub-app for `rinchen eval`: `robustness` (D12), `suite` (D13: fetch, build and inspect an
evaluation suite) and `run` (D13: score one system over a suite); D13 adds `report`."""

import typer

from rinchen.cli.eval_robustness import robustness_command
from rinchen.cli.eval_run import run_command
from rinchen.cli.eval_suite import suite_app

app = typer.Typer(no_args_is_help=True)

app.command(name="robustness")(robustness_command)
app.add_typer(suite_app, name="suite")
app.command(name="run")(run_command)
