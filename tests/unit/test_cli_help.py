"""Every rinchen CLI subcommand must respond to --help with exit code 0."""

import pytest
from typer.testing import CliRunner

from rinchen.cli.main import app

runner = CliRunner()

SUBCOMMANDS = [
    "data",
    "spm",
    "filter",
    "mine",
    "convert-script",
    "synth",
    "train",
    "export",
    "eval",
    "serve",
    "annotate",
    "doctor",
    "tiny",
]


def test_root_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand):
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0
