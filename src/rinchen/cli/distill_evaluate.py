"""`rinchen distill evaluate`: score a checkpoint on the dev rows of the corpus files (D10).
Parsing, file I/O and the manifest only; the work happens in `rinchen.train` (`evaluate_variant`
and the Eole prediction under it, `load_dev_rows`). The command is registered on the `distill`
group by `rinchen.cli.distill`.

    rinchen distill evaluate --checkpoint C --spm-model M --corpus F [--corpus F2 ...]
        --tag-placement src|tgt --out-dir O [--max-lines 1000] [--threads 2] [--timeout S]

`--checkpoint` is a `step_<N>` directory, a directory `rinchen train average` wrote or the
`ckpt` directory of a run: a model directory with `config.json`, `vocab.json` and
`model.*.safetensors`. It is scored with `--spm-model`, which must be ITS OWN SentencePiece
model (a student and its teacher may have different ones), and with the tag placement it was
trained with: when the checkpoint's saved config.json records the placement (the prefixes of
its `data` section, as our launcher writes them), a `--tag-placement` that disagrees is
refused before anything is written, and a config without a `data` section or a direction tag
leaves the statement as it is. The checkpoint need not have a manifest: a baseline or a
foreign model may be evaluated; what the command records of it is the SHA-256 of those files,
taken before the prediction. Teacher against student is two runs with the same `--corpus` files and
`--max-lines`, which score the same dev rows.

The dev rows are those of the corpus files for the directions their train rows have (the
first `--max-lines` of each by id), as `rinchen train ablate` scores them: no test row is
read, and `dev lines in training` counts the dev lines whose id is also a training row's (such
a line inflates the score). All lines of all directions go to ONE Eole prediction, on the CPU
with greedy search, and each direction is scored on its own slice. A run prints

    direction <direction>: chrF <x> BLEU <y> (<n> lines)       (one per direction, sorted)
    chrF mean: <x>
    chrF min: <x>
    tag leaks: <n>
    dev lines in training: <n>
    result: <O>/result.json
    manifest: <O>/manifest.json

and writes, under O: `result.json` (the scores of each direction, the mean and the lowest
chrF, the tag leaks, the signatures of the metrics, and `checkpoint` as given, `spm_sha256`,
`placement` and `max_lines`: sorted keys, two-space indent, LF; the same inputs give the same
bytes), `hyp.<direction>.txt` (the hypotheses, one line each), Eole's own `dev.*` files and,
last, `manifest.json` (stage `distill_evaluate`; its inputs are the corpus files, the
SentencePiece model and the checkpoint's `config.json`, `vocab.json` and weights; its metrics
`chrf_mean`, `chrf_min`, `tag_leaks`, `dev_in_train` and `lines.<direction>`,
`chrf.<direction>` and `bleu.<direction>`). The old manifest and result are removed before the
prediction starts, so a failed run leaves none.

Exit codes: 0; 2 for a bad option, a checkpoint directory without the files Eole loads or
whose path holds `$` or a glob character (Eole globs it), a tag placement its saved config
contradicts, a SentencePiece model that lacks a direction tag, an output directory that holds
`$` or is the directory of an input; 3 for a missing or unreadable corpus, a saved config that
is not what Eole writes, no row to train on or no dev row for a direction and another stage's
manifest in the output directory; 4 when the prediction fails; 1 when the output
cannot be written; 130 when interrupted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    create_out_dir,
    exit_cannot_write,
    hash_input,
    reject_repeated_corpora,
    require_out_apart_from_inputs,
    require_usable_timeout,
    write_manifest,
)
from rinchen.config.ablation import TagPlacement
from rinchen.errors import ConfigError, DataError
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file, refuse_foreign_manifest
from rinchen.train.checkpoints import CONFIG_FILE, VOCAB_FILE, WEIGHTS_FILE
from rinchen.train.distill_saved_config import require_saved_placement
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.train.eole_run import predict_tokens
from rinchen.train.evaluate import DevResult, evaluate_variant, hypothesis_path
from rinchen.train.mixture import DevRows, load_dev_rows, load_training_rows
from rinchen.train.validation_corpus import require_direction_tags
from rinchen.utils.part_file import replace_file

_STAGE: Final = "distill_evaluate"
RESULT_FILE: Final = "result.json"
_TOOL_PACKAGES: Final = ("eole", "torch", "sentencepiece", "sacrebleu", "polars")
_OUTPUTS: Final = "evaluation"
_CORPUS: Final = "corpus file"
_SPM_MODEL: Final = "SentencePiece model"
# Eole loads a checkpoint through `glob(<model path>/model.*.safetensors)`.
_WEIGHTS_PATTERN: Final = "model.*.safetensors"
# The evaluation has no seed: nothing random happens in it.
_NO_SEED: Final = -1


class _Hashed(NamedTuple):
    """The inputs of a run, hashed before anything read them, in the order of the manifest:
    the corpus files, the SentencePiece model and the files of the checkpoint."""

    entries: list[ManifestFile]
    spm_sha256: str
    config_sha256: str
    sources: list[Path]


def evaluate_command(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help=(
                "The model directory to score: a step_<N> directory, an averaged "
                "directory or a run's ckpt directory."
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
    corpus: Annotated[
        list[Path],
        typer.Option(
            "--corpus",
            help=(
                "Corpus Parquet file; repeat it for more. Its dev rows are scored, for "
                "the directions its train rows have."
            ),
        ),
    ],
    tag_placement: Annotated[
        TagPlacement,
        typer.Option(
            "--tag-placement",
            help=(
                "Where the direction tag goes: in the source (src) or as a forced "
                "prefix of the target (tgt); the one the checkpoint was trained with "
                "(refused when its saved config records the other)."
            ),
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the result, the hypotheses, Eole's files and the manifest.",
        ),
    ],
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            min=1,
            help="Most dev lines of each direction (those with the smallest ids).",
        ),
    ] = 1000,
    threads: Annotated[
        int,
        typer.Option(
            "--threads", min=1, help="OpenMP and MKL threads of the prediction."
        ),
    ] = 2,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Seconds after which the prediction is killed; default: no limit.",
        ),
    ] = None,
) -> None:
    """Score a checkpoint on the dev rows of the corpus files: chrF and BLEU per direction.

    The checkpoint need not have a manifest: a baseline or a foreign model may be
    evaluated. The prediction runs on the CPU with the GPU hidden, no test row is read, and
    two runs of one model with the same inputs write the same result.json."""
    with rinchen_errors_as_exit_codes():
        require_usable_timeout(timeout)
        reject_repeated_corpora(corpus)
        source = Path(os.path.abspath(checkpoint))
        target = Path(os.path.abspath(out_dir))
        reject_dollar("checkpoint directory", source)
        reject_glob_characters("checkpoint directory", source)
        reject_dollar("output directory", target)
        hashed = _hash_inputs(corpus, spm_model, source)
        require_out_apart_from_inputs(out_dir, hashed.sources, option="--out-dir")
        refuse_foreign_manifest(target, _STAGE)
        require_saved_placement(
            source / CONFIG_FILE, hashed.config_sha256, tag_placement
        )
        training = load_training_rows(corpus)
        dev = load_dev_rows(
            corpus,
            max_lines=max_lines,
            directions=training.directions,
            training=training,
        )
        # Before the old result is removed: a model that cannot be used must not cost it.
        require_direction_tags(spm_model, dev.sets)
        create_out_dir(target, what=_OUTPUTS)
        try:
            # Just before the prediction, after everything that can fail on bad input: the
            # old manifest and result go first, so that a failure leaves none.
            begin_commit(target, _STAGE)
            (target / RESULT_FILE).unlink(missing_ok=True)
            result = evaluate_variant(
                source,
                dev.sets,
                spm_model=spm_model,
                placement=tag_placement,
                out_dir=target,
                threads=threads,
                timeout=timeout,
                predict=predict_tokens,
            )
            manifest_path = _commit(
                result,
                target,
                hashed,
                checkpoint=checkpoint,
                tag_placement=tag_placement,
                max_lines=max_lines,
                dev=dev,
            )
        except OSError as exc:
            exit_cannot_write(target, exc, what=_OUTPUTS)
        for line in _result_lines(result, dev, target, manifest_path):
            typer.echo(line)


def _hash_inputs(corpus: Sequence[Path], spm_model: Path, source: Path) -> _Hashed:
    """The manifest entries of the corpus files, the SentencePiece model and the files of
    the checkpoint (`config.json`, `vocab.json` and every `model.*.safetensors`), each taken
    before anything reads it."""
    corpora = [hash_input(path, what=_CORPUS, missing=DataError) for path in corpus]
    model = hash_input(spm_model, what=_SPM_MODEL, missing=ConfigError)
    config = hash_input(
        source / CONFIG_FILE, what="checkpoint config.json", missing=ConfigError
    )
    vocabulary = hash_input(
        source / VOCAB_FILE, what="checkpoint vocab.json", missing=ConfigError
    )
    weights = [
        hash_input(path, what="checkpoint weights", missing=ConfigError)
        for path in _weight_files(source)
    ]
    return _Hashed(
        entries=[*corpora, model, config, vocabulary, *weights],
        spm_sha256=model.sha256,
        config_sha256=config.sha256,
        sources=[
            *corpus,
            spm_model,
            *(Path(entry.path) for entry in (config, vocabulary, *weights)),
        ],
    )


def _weight_files(source: Path) -> list[Path]:
    found = sorted(source.glob(_WEIGHTS_PATTERN))
    if not found:
        raise ConfigError(f"checkpoint weights not found: {source / WEIGHTS_FILE}")
    return found


def _commit(
    result: DevResult,
    target: Path,
    hashed: _Hashed,
    *,
    checkpoint: Path,
    tag_placement: TagPlacement,
    max_lines: int,
    dev: DevRows,
) -> Path:
    """Write `result.json`, then the manifest, which comes last so that it never describes
    outputs that are not there."""
    result_path = target / RESULT_FILE
    document = {
        **result.to_dict(),
        "checkpoint": str(checkpoint),
        "spm_sha256": hashed.spm_sha256,
        "placement": tag_placement,
        "max_lines": max_lines,
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    replace_file(result_path, f"{text}\n".encode())
    return write_manifest(
        target,
        stage=_STAGE,
        config_sha256=_options_sha256(tag_placement, max_lines),
        seed=_NO_SEED,
        inputs=hashed.entries,
        outputs=[
            output_file(result_path, None),
            *(
                output_file(hypothesis_path(target, direction), scores.lines)
                for direction, scores in result.directions.items()
            ),
        ],
        metrics=_metrics(result, dev),
        packages=_TOOL_PACKAGES,
        status={"evaluate": "ok", "tag_placement": tag_placement},
    )


def _options_sha256(tag_placement: TagPlacement, max_lines: int) -> str:
    """The evaluation has no config file, so the options that decide its numbers stand in
    for one: the SHA-256 of their canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(
        {"max_lines": max_lines, "placement": tag_placement},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metrics(result: DevResult, dev: DevRows) -> dict[str, float]:
    return {
        "chrf_mean": result.chrf_mean,
        "chrf_min": result.chrf_min,
        "tag_leaks": float(result.tag_leaks),
        "dev_in_train": float(dev.stats.dev_in_train or 0),
        **{f"lines.{d}": float(s.lines) for d, s in result.directions.items()},
        **{f"chrf.{d}": s.chrf for d, s in result.directions.items()},
        **{f"bleu.{d}": s.bleu for d, s in result.directions.items()},
    }


def _result_lines(
    result: DevResult, dev: DevRows, target: Path, manifest_path: Path
) -> list[str]:
    return [
        *(
            f"direction {direction}: chrF {scores.chrf:.2f} BLEU {scores.bleu:.2f} "
            f"({scores.lines} lines)"
            for direction, scores in sorted(result.directions.items())
        ),
        f"chrF mean: {result.chrf_mean:.2f}",
        f"chrF min: {result.chrf_min:.2f}",
        f"tag leaks: {result.tag_leaks}",
        f"dev lines in training: {dev.stats.dev_in_train}",
        f"result: {target / RESULT_FILE}",
        f"manifest: {manifest_path}",
    ]
