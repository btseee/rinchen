"""`rinchen export`: the commands on the export directory of a model (D11).

`convert` turns a checkpoint of ours into an export directory (the same command function as
`rinchen distill export`, D10's name for it, which is registered beside it) and `verify`
reads one back: it accepts a directory only when `rinchen.export.verified.verify_export`
does, prints what the record says and the digests of the model and the manifest, and
changes nothing.

    rinchen export verify DIR

prints `export: <DIR>`, `name`, `quantization`, `tag placement`, `directions (<n>)`,
`document level` and the first 12 hexadecimal digits of the SHA-256 of `model.bin` and of the
manifest, and exits 0. Exit codes: 2 for a directory `verify_export` refuses (no manifest, a
file that does not match it, a record that disagrees with the files, the format of the
distillation of D10 without `rinchen.json`: the message on stderr, nothing on stdout), 3 for
files that cannot be read or a manifest or record that is not what it must be.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer

from rinchen.cli import distill_export
from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.export.record import CT2_FILES
from rinchen.export.verified import verify_export

_DIGEST_SHOWN: Final = 12
_MODEL_FILE: Final = CT2_FILES[0]

app = typer.Typer(no_args_is_help=True)
app.command("convert")(distill_export.export_command)


@app.command("verify")
def verify_command(
    directory: Annotated[
        Path, typer.Argument(help="The export directory `rinchen export convert` made.")
    ],
) -> None:
    """Check an export directory against its manifest and print what its record says.

    Nothing is written. A directory that is not vouched for is refused with the reason."""
    with rinchen_errors_as_exit_codes():
        verified = verify_export(directory)
    record = verified.record
    typer.echo(f"export: {verified.directory}")
    typer.echo(f"name: {record.name}")
    typer.echo(f"quantization: {record.quantization}")
    typer.echo(f"tag placement: {record.tag_placement}")
    typer.echo(f"directions ({len(record.directions)}): {', '.join(record.directions)}")
    typer.echo(f"document level: {'true' if record.document_level else 'false'}")
    typer.echo(f"model sha256 {verified.sha256[_MODEL_FILE][:_DIGEST_SHOWN]}")
    typer.echo(f"manifest sha256 {verified.manifest_sha256[:_DIGEST_SHOWN]}")
