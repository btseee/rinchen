"""CLI sub-app for `rinchen eval`: `robustness` (D12); D13 (Final eval) adds the rest."""

import typer

from rinchen.cli.eval_robustness import robustness_command

app = typer.Typer(no_args_is_help=True)

app.command(name="robustness")(robustness_command)
