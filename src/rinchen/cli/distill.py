"""`rinchen distill`: the commands of the distillation of the teacher into students (D10).

`export` turns a checkpoint of ours into a CTranslate2 model directory with a record of where
it came from, `generate` has such a teacher translate the authentic training sources into a
knowledge-distillation (KD) corpus, `prepare` lays out the data and the config of a student
that trains on that corpus mixed with the original data, and `evaluate` scores a checkpoint
(a teacher or a student) on the dev rows. Each command is a plain function in its own module
that this group registers, as `rinchen.cli.train` registers `launch`, `average` and `docify`;
the work happens in `rinchen.train`.
"""

from __future__ import annotations

import typer

from rinchen.cli import (
    distill_evaluate,
    distill_export,
    distill_generate,
    distill_prepare,
)

app = typer.Typer(no_args_is_help=True)
app.command("export")(distill_export.export_command)
app.command("generate")(distill_generate.generate_command)
app.command("prepare")(distill_prepare.prepare_command)
app.command("evaluate")(distill_evaluate.evaluate_command)
