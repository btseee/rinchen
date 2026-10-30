"""Top-level `rinchen` typer app, registering one sub-app per CLI module."""
from __future__ import annotations

import typer

from rinchen.cli import (
    annotate,
    convert_script,
    data,
    doctor,
    export,
    mine,
    serve,
    spm,
    synth,
    tiny,
    train,
)
from rinchen.cli import (
    eval as eval_cli,
)
from rinchen.cli import (
    filter as filter_cli,
)

app = typer.Typer(name="rinchen", no_args_is_help=True)

app.add_typer(data.app, name="data")
app.add_typer(spm.app, name="spm")
app.add_typer(filter_cli.app, name="filter")
app.add_typer(mine.app, name="mine")
app.add_typer(convert_script.app, name="convert-script")
app.add_typer(synth.app, name="synth")
app.add_typer(train.app, name="train")
app.add_typer(export.app, name="export")
app.add_typer(eval_cli.app, name="eval")
app.add_typer(serve.app, name="serve")
app.add_typer(annotate.app, name="annotate")
app.add_typer(doctor.app, name="doctor")
app.add_typer(tiny.app, name="tiny")
