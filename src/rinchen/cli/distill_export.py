"""`rinchen distill export`: turn a checkpoint of ours into a CTranslate2 model directory with
a record of where it came from (D10). Parsing, file I/O and the manifest only; the work
happens in `rinchen.train.distill_export` (`export_checkpoint`) and the converter it runs
(`rinchen.export.eole_ct2`). The command is registered on the `distill` group by
`rinchen.cli.distill`.

    rinchen distill export --checkpoint C --spm-model M --tag-placement src|tgt
        --out-dir O [--timeout S]

`--checkpoint` is the final `step_<N>` directory of a run `rinchen train launch` finished,
or a directory `rinchen train average` wrote, in either case with the manifest the command
wrote, which must vouch for the files as they are now. `--spm-model` must be the SentencePiece
model the checkpoint was trained with: its SHA-256 must be that of the copy Eole saved into
the checkpoint. `--tag-placement` (no default) says where the direction tag goes, in the
source (`src`) or as a forced prefix of the target (`tgt`): CTranslate2's files do not hold
it, so the export records what the caller says. The saved config.json of a checkpoint our
launcher wrote records it in the prefixes of its `data` section, and a placement that
disagrees is refused (exit 2) before anything is made; a saved config without a `data`
section, or without a direction tag in it, leaves the statement unchecked. `--out-dir` must not
exist (the converter refuses a directory that exists, so it works in a private directory made
beside `--out-dir`, which is renamed onto it when the export is complete; a directory someone
else makes there in the meantime is refused and kept); the converter's log is `<O>.export.log`
beside it, and `--timeout` (seconds) bounds the converter.

A run prints

    exported to <O>
    record: <O>/export.json
    manifest: <O>/manifest.json

and writes, under O: `model.bin`, `shared_vocabulary.json` and `config.json` (the converter's
files, run with the GPU hidden), `spm.model` (a copy of the SentencePiece model),
`export.json` (the record: placement, the checkpoint's kind, step and digests, the layer
counts, the hidden size and the vocabulary size) and, last, `manifest.json`. No GPU is used.

The manifest (stage `distill_export`) lists the five files as outputs with the SHA-256 they
have now (rows None); its inputs are the checkpoint's weights, `config.json` and `vocab.json`
and the SentencePiece model, with the SHA-256 the verification took before it read any of
them (the record's; the command never hashes them again). Its metrics are `encoder_layers`,
`decoder_layers`, `hidden_size`, `vocabulary_entries` and `model_bytes`; its status says
`export: ok`, the `tag_placement` and the `checkpoint_kind`; its seed is -1 and its config
digest that of the canonical JSON `{"tag_placement": ...}`. A failure while it is written
removes the output directory, as a failure of the converter does.

Exit codes: 0; 2 for a bad option, a checkpoint our manifest does not vouch for, a
SentencePiece model that is not the checkpoint's, a tag placement the checkpoint's saved
config contradicts, an output directory that exists or holds `$` or a glob character; 3 for a
checkpoint or manifest that cannot be read or whose `config.json` or `vocab.json` lacks what
an export records or contradicts itself about the tag placement; 4 when the converter fails,
leaves a file out or writes a vocabulary with another number of entries than the checkpoint's;
1 when the output cannot be written; 130 when interrupted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Annotated, Final

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    exit_cannot_write,
    require_usable_timeout,
    write_manifest,
)
from rinchen.config.ablation import TagPlacement
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import output_file
from rinchen.train.checkpoints import CONFIG_FILE, VOCAB_FILE, WEIGHTS_FILE
from rinchen.train.distill_export import (
    EXPORT_FILES,
    EXPORT_STAGE,
    RECORD_FILE,
    ExportRecord,
    export_checkpoint,
)

_TOOL_PACKAGES: Final = ("ctranslate2", "eole", "torch")
_OUTPUTS: Final = "CTranslate2 export"
_MODEL_FILE: Final = "model.bin"
# The export has no seed: nothing random happens in it.
_NO_SEED: Final = -1


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
        source = Path(os.path.abspath(checkpoint))
        target = Path(os.path.abspath(out_dir))
        spm = Path(os.path.abspath(spm_model))
        try:
            record = export_checkpoint(
                source,
                target,
                spm_model=spm,
                tag_placement=tag_placement,
                timeout=timeout,
            )
            manifest_path = _write_manifest(source, spm, target, record)
        except OSError as exc:
            exit_cannot_write(target, exc, what=_OUTPUTS)
        typer.echo(f"exported to {target}")
        typer.echo(f"record: {target / RECORD_FILE}")
        typer.echo(f"manifest: {manifest_path}")


def _write_manifest(
    source: Path, spm: Path, target: Path, record: ExportRecord
) -> Path:
    """Write the manifest of the export in `target`, last. Anything that goes wrong removes
    `target`: an export directory without its manifest is vouched for by nothing, and the
    next attempt would find the directory in its way."""
    try:
        return write_manifest(
            target,
            stage=EXPORT_STAGE,
            config_sha256=_placement_sha256(record.tag_placement),
            seed=_NO_SEED,
            inputs=_inputs(source, spm, record),
            outputs=[output_file(target / name, None) for name in EXPORT_FILES],
            metrics={
                "encoder_layers": float(record.encoder_layers),
                "decoder_layers": float(record.decoder_layers),
                "hidden_size": float(record.hidden_size),
                "vocabulary_entries": float(record.vocabulary_entries),
                "model_bytes": float((target / _MODEL_FILE).stat().st_size),
            },
            packages=_TOOL_PACKAGES,
            status={
                "export": "ok",
                "tag_placement": record.tag_placement,
                "checkpoint_kind": record.checkpoint.kind,
            },
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _inputs(source: Path, spm: Path, record: ExportRecord) -> list[ManifestFile]:
    """The checkpoint's files and the SentencePiece model, with the SHA-256 the verification
    took before it read them: the record's, never taken again."""
    checkpoint = record.checkpoint
    return [
        ManifestFile(path=str(source / WEIGHTS_FILE), sha256=checkpoint.weights_sha256),
        ManifestFile(path=str(source / CONFIG_FILE), sha256=checkpoint.config_sha256),
        ManifestFile(path=str(source / VOCAB_FILE), sha256=checkpoint.vocab_sha256),
        ManifestFile(path=str(spm), sha256=record.spm_sha256),
    ]


def _placement_sha256(tag_placement: TagPlacement) -> str:
    """The export has no config file, so its one choice stands in for one: the SHA-256 of the
    canonical JSON (sorted keys, no spaces) of `{"tag_placement": ...}`."""
    canonical = json.dumps(
        {"tag_placement": tag_placement}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
