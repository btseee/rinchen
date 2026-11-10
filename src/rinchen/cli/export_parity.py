"""`rinchen export parity`: check an export against the checkpoint it was made from (D11).
Parsing, file I/O and the manifest only; the work happens in `rinchen.train.export_parity`
(`strict_parity`, the Eole run and the CTranslate2 run under it) and `load_dev_rows`. The
command is registered on the `export` group by `rinchen.cli.export`.

    rinchen export parity --checkpoint C --export E --corpus F [--corpus F2 ...] --out-dir O
        [--max-lines 50] [--threads 2] [--timeout S]

`--export` is an export directory `rinchen export convert` wrote: `verify_export` must vouch for
it (its record names the tag placement, the quantisation and the directions the model was
trained for). `--checkpoint` is the model directory it was made from (`config.json`,
`vocab.json` and `model.00.safetensors`): the SHA-256 of its weights must be the one the record
names, else nothing runs. The dev rows of the corpus files for every direction of the record
(the first `--max-lines` of each by id, as `rinchen distill evaluate` reads them; no test row is
read) go through `eole predict` and through CTranslate2, both greedy and on the CPU with the GPU
hidden, and a run prints

    parity: <exact>/<n> exact, mean |delta| <x> per token (<gated|not gated: quantized>) <pass|FAIL>
    result: <O>/parity.json
    manifest: <O>/manifest.json

The exact share must reach 0.95 and, for a float32 export, the mean log-probability delta stay
within 1e-3 per token; a quantised export reports the delta and is held to the exact share
alone (see `rinchen.train.export_parity`). `parity.json` holds the report, the thresholds, the
number of lines of each direction, the name and `model.bin` digest of the export and the
digest of the weights (sorted keys, two-space indent, LF; the same inputs give the same bytes);
the manifest (stage `export_parity`) names every input with the digest taken before it was read
(the corpus files, the five files of the export and its manifest, the checkpoint's weights,
`config.json` and `vocab.json`) and is written last. The old manifest and result are removed
before the run starts, so a failed run leaves none.

Exit codes: 0 when the parity passes; 4 when it does not (a measured result: everything is
written and printed) or when Eole or CTranslate2 fails; 2 for a bad option, an export that is
not vouched for, a checkpoint of other weights than the export's, a path Eole reads specially
(`$`, or a glob character in the checkpoint's), an output directory that is an input's; 3 for a
missing or unreadable corpus, a direction of the model without dev rows and another stage's
manifest in the output directory; 1 when the output cannot be written; 130 when interrupted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import typer

from rinchen.cli.errors import UPSTREAM_ERROR_EXIT_CODE, rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    create_out_dir,
    exit_cannot_write,
    hash_input,
    reject_repeated_corpora,
    require_out_apart_from_inputs,
    require_usable_timeout,
    write_manifest,
)
from rinchen.errors import ConfigError, DataError
from rinchen.export.record import EXPORT_FILES
from rinchen.export.verified import VerifiedExport, verify_export
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file, refuse_foreign_manifest
from rinchen.train.checkpoints import CONFIG_FILE, VOCAB_FILE, WEIGHTS_FILE
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.train.export_parity import (
    MAX_MEAN_DELTA,
    MIN_EXACT_SHARE,
    ParityReport,
    ParitySource,
    require_checkpoint_of_export,
    strict_parity,
)
from rinchen.train.mixture import DevRows, load_dev_rows
from rinchen.utils.part_file import replace_file

_STAGE: Final = "export_parity"
PARITY_FILE: Final = "parity.json"
# `eole predict` runs on torch.
_TOOL_PACKAGES: Final = ("eole", "torch", "ctranslate2", "sentencepiece", "polars")
_OUTPUTS: Final = "parity result"
_CORPUS: Final = "corpus file"
_MODEL_FILE: Final = EXPORT_FILES[0]
# The parity has no seed: nothing random happens in it.
_NO_SEED: Final = -1


class _Hashed(NamedTuple):
    """The inputs of a run, hashed before anything read them, in the order of the manifest:
    the corpus files, the files of the export and the files of the checkpoint."""

    entries: list[ManifestFile]
    sources: list[Path]
    weights_sha256: str


def parity_command(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="The model directory the export was made from (the weights Eole loads).",
        ),
    ],
    export: Annotated[
        Path,
        typer.Option(
            "--export", help="The export directory `rinchen export convert` wrote."
        ),
    ],
    corpus: Annotated[
        list[Path],
        typer.Option(
            "--corpus",
            help=(
                "Corpus Parquet file; repeat it for more. Its dev rows are translated, "
                "for every direction of the export."
            ),
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the result, Eole's files and the manifest.",
        ),
    ],
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            min=1,
            help="Most dev lines of each direction (those with the smallest ids).",
        ),
    ] = 50,
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            min=1,
            help="OpenMP and MKL threads of Eole, and the threads of CTranslate2.",
        ),
    ] = 2,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Seconds after which the Eole run is killed; default: no limit.",
        ),
    ] = None,
) -> None:
    """Check an export against its checkpoint: the same dev lines through Eole and CTranslate2.

    Both run greedy on the CPU with the GPU hidden. Exit 0 when the exact share reaches 0.95
    and, for a float32 export, the mean log-probability delta stays within 1e-3 per token;
    exit 4 when it does not."""
    with rinchen_errors_as_exit_codes():
        require_usable_timeout(timeout)
        reject_repeated_corpora(corpus)
        source = Path(os.path.abspath(checkpoint))
        target = Path(os.path.abspath(out_dir))
        reject_dollar("checkpoint directory", source)
        reject_glob_characters("checkpoint directory", source)
        reject_dollar("output directory", target)
        verified = verify_export(export)
        hashed = _hash_inputs(corpus, verified, source)
        require_out_apart_from_inputs(out_dir, hashed.sources, option="--out-dir")
        refuse_foreign_manifest(target, _STAGE)
        # Before the old result is removed: a model that cannot be checked must not cost it.
        require_checkpoint_of_export(verified.record, hashed.weights_sha256)
        dev = load_dev_rows(
            corpus, max_lines=max_lines, directions=verified.record.directions
        )
        create_out_dir(target, what=_OUTPUTS)
        try:
            # Just before the run, after everything that can fail on bad input: the old
            # manifest and result go first, so that a failure leaves none.
            begin_commit(target, _STAGE)
            (target / PARITY_FILE).unlink(missing_ok=True)
            report = strict_parity(
                source,
                verified,
                _sources_of(dev),
                out_dir=target,
                threads=threads,
                timeout=timeout,
                weights_sha256=hashed.weights_sha256,
            )
            manifest_path = _commit(
                report, target, hashed, verified=verified, dev=dev, max_lines=max_lines
            )
        except OSError as exc:
            exit_cannot_write(target, exc, what=_OUTPUTS)
        typer.echo(_parity_line(report))
        typer.echo(f"result: {target / PARITY_FILE}")
        typer.echo(f"manifest: {manifest_path}")
    if not report.passes:
        raise typer.Exit(code=UPSTREAM_ERROR_EXIT_CODE)


def _hash_inputs(
    corpus: Sequence[Path], verified: VerifiedExport, source: Path
) -> _Hashed:
    """The manifest entries of the corpus files, the files of the export (hashed by
    `verify_export`, before it parsed them) and the checkpoint's weights, `config.json` and
    `vocab.json`, each taken before anything reads it."""
    corpora = [hash_input(path, what=_CORPUS, missing=DataError) for path in corpus]
    exported = _export_entries(verified)
    weights = hash_input(
        source / WEIGHTS_FILE, what="checkpoint weights", missing=ConfigError
    )
    config = hash_input(
        source / CONFIG_FILE, what="checkpoint config.json", missing=ConfigError
    )
    vocabulary = hash_input(
        source / VOCAB_FILE, what="checkpoint vocab.json", missing=ConfigError
    )
    return _Hashed(
        entries=[*corpora, *exported, weights, config, vocabulary],
        sources=[
            *corpus,
            *(Path(entry.path) for entry in (*exported, weights, config, vocabulary)),
        ],
        weights_sha256=weights.sha256,
    )


def _export_entries(verified: VerifiedExport) -> list[ManifestFile]:
    """The five files of the export and its manifest with the digests `verify_export`
    took."""
    files = [
        (verified.directory / name, digest) for name, digest in verified.sha256.items()
    ]
    files.append((store.manifest_path(verified.directory), verified.manifest_sha256))
    return [_entry(path, digest) for path, digest in files]


def _entry(path: Path, sha256: str) -> ManifestFile:
    return ManifestFile(path=str(path), sha256=sha256, bytes=path.stat().st_size)


def _sources_of(dev: DevRows) -> list[ParitySource]:
    return [
        ParitySource(text, dev_set.tgt_lang)
        for dev_set in dev.sets.values()
        for text in dev_set.src
    ]


def _commit(
    report: ParityReport,
    target: Path,
    hashed: _Hashed,
    *,
    verified: VerifiedExport,
    dev: DevRows,
    max_lines: int,
) -> Path:
    """Write `parity.json`, then the manifest, which comes last so that it never describes
    outputs that are not there."""
    result_path = target / PARITY_FILE
    lines = dev.stats.lines_by_direction
    document = {
        **report.to_dict(),
        "passes": report.passes,
        "min_exact_share": MIN_EXACT_SHARE,
        "max_mean_delta": MAX_MEAN_DELTA,
        "max_lines": max_lines,
        "lines": lines,
        "export_name": verified.record.name,
        "model_sha256": verified.sha256[_MODEL_FILE],
        "checkpoint_weights_sha256": hashed.weights_sha256,
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    replace_file(result_path, f"{text}\n".encode())
    return write_manifest(
        target,
        stage=_STAGE,
        config_sha256=_options_sha256(max_lines),
        seed=_NO_SEED,
        inputs=hashed.entries,
        outputs=[output_file(result_path, None)],
        metrics=_metrics(report, lines),
        packages=_TOOL_PACKAGES,
        status={
            "parity": "pass" if report.passes else "fail",
            "quantization": report.quantization,
            "tag_placement": report.placement,
        },
    )


def _options_sha256(max_lines: int) -> str:
    """The parity has no config file, so the option that decides its numbers stands in for
    one: the SHA-256 of its canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(
        {"max_lines": max_lines}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metrics(report: ParityReport, lines: Mapping[str, int]) -> dict[str, float]:
    return {
        "sentences": float(report.sentences),
        "exact": float(report.exact),
        "exact_share": report.exact_share,
        "mean_abs_delta": report.mean_abs_delta,
        "max_abs_delta": report.max_abs_delta,
        "gated": float(report.gated),
        "passes": float(report.passes),
        **{f"lines.{direction}": float(count) for direction, count in lines.items()},
    }


def _parity_line(report: ParityReport) -> str:
    bound = "gated" if report.gated else "not gated: quantized"
    verdict = "pass" if report.passes else "FAIL"
    return (
        f"parity: {report.exact}/{report.sentences} exact, mean |delta| "
        f"{report.mean_abs_delta:.2e} per token ({bound}) {verdict}"
    )
