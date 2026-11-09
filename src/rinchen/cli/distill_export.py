"""`rinchen export convert` and `rinchen distill export`: turn a checkpoint of ours into an export
directory (CTranslate2's files, a copy of the SentencePiece model, the record `rinchen.json`
and the manifest). Parsing, the printed lines and the exit codes only; the work happens in
`rinchen.train.distill_export` (`export_checkpoint`) and `rinchen.export.ct2` (`convert`),
which writes the manifest, last. The command is ONE function registered under both names:
`rinchen.cli.distill` registers it as `distill export` (D10's name, which the runbook of the
distillation keeps) and `rinchen.cli.export` as `export convert` (D11).

    rinchen export convert --checkpoint C --spm-model M --tag-placement src|tgt --out-dir O
        [--quantization float32|float16|int8|int8_float16] [--direction CODE-CODE ...]
        [--document-level | --no-document-level] [--name NAME] [--timeout S]

`--checkpoint` is the final `step_<N>` directory of a run `rinchen train launch` finished,
or a directory `rinchen train average` wrote, in either case with the manifest the command
wrote, which must vouch for the files as they are now. `--spm-model` must be the SentencePiece
model the checkpoint was trained with: its SHA-256 must be that of the copy Eole saved into
the checkpoint. `--tag-placement` (no default) says where the direction tag goes, in the
source (`src`) or as a forced prefix of the target (`tgt`): CTranslate2's files do not hold
it, so the export records what the caller says. The saved config.json of a checkpoint our
launcher wrote records it in the prefixes of its `data` section, and a placement that
disagrees is refused (exit 2) before anything is made; a saved config without a `data`
section, or without a direction tag in it, leaves the statement unchecked, and the run says
so (below).

`--quantization` (default float32, the parity reference) is what the converter stores: `int8`
is the CPU product, `float16` and `int8_float16` the GPU products, which are converted here and
never run on a GPU (unmeasured). `--direction` (repeatable, `<src>-<tgt>`) states the
directions the model was trained for; the saved config names them in its corpora, a
statement must equal them (exit 2 before the converter starts) and is required only when the
config names none. `--document-level` / `--no-document-level` states whether the model decodes
`<doc>` windows, with the same rule; with nothing derived and nothing stated it is false.
`--name` is the model's name (1 to 64 characters of letters, digits, `.`, `_` and `-`;
default `rinchen-` and 12 hexadecimal digits of the SHA-256 of `model.bin`).

`--out-dir` must not exist (the converter refuses a directory that exists, so it works in a
private directory made beside `--out-dir`, which is renamed onto it when the export is
complete; a directory someone else makes there in the meantime is refused and kept); the
converter's log is `<O>.export.log` beside it, and `--timeout` (seconds) bounds the converter.

A run prints

    exported to <O>
    tag placement: checked against the saved config
    directions: derived from the saved config (4)
    document level: derived from the saved config (false)
    quantization: float32
    record: <O>/rinchen.json
    manifest: <O>/manifest.json

where the second line is `tag placement: stated, not recorded in the checkpoint` when the
saved config holds no placement to check the statement against, and the third and fourth are
`directions: stated (<n>)` and `document level: stated (true|false)` when it names no
directions or no corpus to read the level from.

and writes, under O: `model.bin`, `shared_vocabulary.json` and `config.json` (the converter's
files, run with the GPU hidden), `spm.model` (a copy of the SentencePiece model),
`rinchen.json` (the record, `rinchen.model/1`) and, last, `manifest.json`. No GPU is used.

The manifest (stage `export`) lists the five files as outputs with the SHA-256 they have now;
its inputs are the checkpoint's weights, `config.json` and `vocab.json` and the SentencePiece
model, with the SHA-256 the verification took before it read any of them (the command never
hashes them again). Its status says `export: ok`, the `tag_placement`, the `quantization`, the
`checkpoint_kind` and how each fact was settled: `tag_placement_check` (`checked` or `stated`),
`directions_check` and `document_level_check` (`derived` or `stated`), which the record does
not hold. A failure while it is written removes the output directory, as a failure of the
converter does.

Exit codes: 0; 2 for a bad option, a checkpoint our manifest does not vouch for, a
SentencePiece model that is not the checkpoint's, a tag placement, a direction set or a
document level the checkpoint's saved config contradicts, directions nobody states or
derives, an output directory that exists or holds `$` or a glob character; 3 for a
checkpoint or manifest that cannot be read or whose `config.json` or `vocab.json` lacks what
an export records or contradicts itself about the tag placement; 4 when the converter fails,
leaves a file out or writes a vocabulary with another number of entries than the checkpoint's;
1 when the output cannot be written; 130 when interrupted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Final

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    exit_cannot_write,
    require_usable_timeout,
    tag_placement_line,
)
from rinchen.config.ablation import TagPlacement
from rinchen.export.quantize import QUANTIZATIONS
from rinchen.export.record import RECORD_FILE
from rinchen.manifest import store
from rinchen.train.distill_export import ExportOutcome, export_checkpoint

_OUTPUTS: Final = "CTranslate2 export"
_FACT_LINES: Final = {
    "derived": "derived from the saved config",
    "stated": "stated",
}


def export_command(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help=(
                "The final step_<N> directory of a finished `rinchen train launch` run, "
                "or a directory `rinchen train average` wrote."
            ),
        ),
    ],
    spm_model: Annotated[
        Path,
        typer.Option(
            "--spm-model",
            help="The SentencePiece model the checkpoint was trained with.",
        ),
    ],
    tag_placement: Annotated[
        TagPlacement,
        typer.Option(
            "--tag-placement",
            help=(
                "Where the direction tag goes: in the source (src) or as a forced "
                "prefix of the target (tgt). The export records it; it is refused "
                "when the checkpoint's saved config records the other placement."
            ),
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory; it must not exist (the converter makes it).",
        ),
    ],
    quantization: Annotated[
        str,
        typer.Option(
            "--quantization",
            help=(
                f"What the converter stores: {', '.join(QUANTIZATIONS)}. float32 is the "
                "parity reference, int8 the CPU product; float16 and int8_float16 are "
                "GPU products that were never run on a GPU."
            ),
        ),
    ] = "float32",
    direction: Annotated[
        list[str] | None,
        typer.Option(
            "--direction",
            help=(
                "A direction the model was trained for, <src>-<tgt> (repeatable). The "
                "saved config names them in its corpora and a statement must equal "
                "them; needed only when it names none."
            ),
        ),
    ] = None,
    document_level: Annotated[
        bool | None,
        typer.Option(
            "--document-level/--no-document-level",
            help=(
                "Whether the model decodes <doc> windows. The saved config says when "
                "its corpora do and a statement must equal it; default: that, else "
                "false."
            ),
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help=(
                "The model's name: 1 to 64 characters of letters, digits, '.', '_' and "
                "'-'. Default: rinchen- and 12 hexadecimal digits of the SHA-256 of "
                "model.bin."
            ),
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Seconds the converter may run before it is killed; default: no limit.",
        ),
    ] = None,
) -> None:
    """Export a checkpoint of ours to a CTranslate2 directory with a record and a manifest.

    The converter runs as a process of its own with the GPU hidden; the checkpoint is never
    written. The output directory holds the five files of the export and the manifest that
    vouches for them."""
    with rinchen_errors_as_exit_codes():
        require_usable_timeout(timeout)
        target = Path(os.path.abspath(out_dir))
        try:
            outcome = export_checkpoint(
                Path(os.path.abspath(checkpoint)),
                target,
                spm_model=Path(os.path.abspath(spm_model)),
                tag_placement=tag_placement,
                quantization=quantization,
                directions=direction,
                document_level=document_level,
                name=name,
                timeout=timeout,
            )
        except OSError as exc:
            exit_cannot_write(target, exc, what=_OUTPUTS)
        typer.echo(f"exported to {target}")
        typer.echo(tag_placement_line(outcome.placement_check))
        typer.echo(_directions_line(outcome))
        typer.echo(_document_level_line(outcome))
        typer.echo(f"quantization: {outcome.record.quantization}")
        typer.echo(f"record: {target / RECORD_FILE}")
        typer.echo(f"manifest: {store.manifest_path(target)}")


def _directions_line(outcome: ExportOutcome) -> str:
    how = _FACT_LINES[outcome.directions_check]
    return f"directions: {how} ({len(outcome.record.directions)})"


def _document_level_line(outcome: ExportOutcome) -> str:
    how = _FACT_LINES[outcome.document_level_check]
    level = "true" if outcome.record.document_level else "false"
    return f"document level: {how} ({level})"
