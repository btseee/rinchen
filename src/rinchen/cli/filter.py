"""`rinchen filter`: score and filter a corpus Parquet file with the configured
pipeline and record the run in a manifest. Presentation, file I/O and the
manifest only; the work happens in `rinchen.filter`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from rinchen.cli.errors import (
    FAILURE_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.config.filters import (
    FILTERS_CONFIG_PATH,
    FilterConfig,
    LabseConfig,
    QeConfig,
    load_filter_config,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.writers import write_corpus
from rinchen.errors import ConfigError, RinchenError
from rinchen.filter import qe
from rinchen.filter.pipeline import PipelineResult, kept_rows, run_pipeline
from rinchen.filter.qe import QEScorer
from rinchen.filter.qe_eole import EoleMetricXQEScorer
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    begin_commit,
    build_inputs,
    code_version,
    output_file,
    outputs_current,
    refuse_foreign_manifest,
    sha256_of,
    tool_versions,
)
from rinchen.mine.embed import Embedder, SentenceTransformerEmbedder
from rinchen.utils.logging import get as get_logger

app = typer.Typer(no_args_is_help=True)

# Polars appends this note, naming the failing expression, to an exception
# raised inside a stage. Stripping it is best-effort: it couples to the text of
# polars' message, so a polars release that words the note differently leaves
# the note in the printed error (the error and its exit code are unaffected).
_POLARS_EXPRESSION_NOTE = "\n\nThis error occurred in the following expression"

_STAGE_NAME = "filter"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "clean"
_SCORED_FILE = "scored.parquet"
_KEPT_FILE = "kept.parquet"
_TOOL_PACKAGES = (
    "polars",
    "datasketch",
    # `import fasttext` is served by the vendored `fasttext-wheel` shim, which
    # wraps the native binding of `fasttext-new`: a LID run depends on both.
    "fasttext-new",
    "fasttext-wheel",
    "sentence-transformers",
    "sentencepiece",
    "torch",
    "eole",
)


def build_embedder(cfg: LabseConfig) -> Embedder:
    """The LaBSE embedder of a run; it loads its model on first use. Tests
    replace this function."""
    return SentenceTransformerEmbedder(
        cfg.model, batch_size=cfg.batch_size, device=cfg.device
    )


def build_qe_scorer(cfg: QeConfig) -> QEScorer | None:
    """The QE scorer of a run, or None when none exists for `cfg.scorer`
    (the pipeline then skips the stage). It loads its model on first use.
    Tests replace this function."""
    return EoleMetricXQEScorer() if cfg.scorer == "metricx_qe" else None


def _error_message(error: RinchenError) -> str:
    """The text of an error without polars' note naming the failing expression,
    which is logged instead."""
    text = str(error)
    message = text.split(_POLARS_EXPRESSION_NOTE, 1)[0]
    if message != text:
        get_logger(__name__).error("filter_error_detail", detail=text)
    return message


@app.command("run")
def run(
    corpus: Annotated[
        Path, typer.Option("--corpus", help="Corpus Parquet file to filter.")
    ],
    config: Annotated[
        Path, typer.Option("--config", help="Filter pipeline config (YAML).")
    ] = FILTERS_CONFIG_PATH,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Output directory; default data/clean/<corpus file stem>.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Run even when inputs and config are unchanged."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Check the config, the model files and the corpus, print the "
                "stages, write nothing. A QE skip is shown only when it can be "
                "known without importing torch (no scorer to build)."
            ),
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=(
                "Recorded in the manifest only: the pipeline is deterministic "
                "(minhash has its own seed in the config), and the seed does "
                "not affect skipping."
            ),
        ),
    ] = 0,
) -> None:
    """Score a corpus with the configured stages and split it into the scored
    corpus (every row, with its filter reason) and the kept corpus. The corpus
    is held in memory: budget two to three times its size."""
    with rinchen_errors_as_exit_codes(message_of=_error_message):
        target = out_dir if out_dir is not None else DEFAULT_OUT_ROOT / corpus.stem
        # First of all: nothing is read or hashed for a run that would replace
        # its own input.
        _require_corpus_apart_from_outputs(corpus, target)
        cfg = load_filter_config(config)
        _require_model_files(cfg)
        frame = scan_corpus(corpus)
        if dry_run:
            for line in _plan_lines(cfg):
                typer.echo(line)
            return

        refuse_foreign_manifest(target, _STAGE_NAME)
        input_paths = [corpus, *(path for _, path in _model_files(cfg))]
        config_sha256 = sha256_of(config)
        if outputs_current(
            target,
            _STAGE_NAME,
            input_paths,
            config_sha256,
            list(_output_paths(target)),
            force=force,
        ):
            typer.echo("skipped: inputs and config are unchanged since the last run")
            typer.echo(f"manifest: {store.manifest_path(target)}")
            return

        input_files = build_inputs(input_paths)
        result = _run_with_models(frame, cfg)
        kept = kept_rows(result.frame)
        try:
            manifest_path = _write_run(
                result, kept, target, config_sha256, seed, input_files
            )
        except OSError as exc:
            exit_with_error(
                f"cannot write the filter outputs under {target}: {exc}",
                FAILURE_EXIT_CODE,
            )
        typer.echo(f"kept {kept.height} of {result.frame.height} rows")
        typer.echo(f"manifest: {manifest_path}")


def _plan_lines(cfg: FilterConfig) -> list[str]:
    """One line per configured stage. Only a QE stage that has no scorer to
    build is known to be skipped without loading a model."""
    lines = []
    for name in cfg.stages:
        if name == "qe" and build_qe_scorer(cfg.qe) is None:
            lines.append(f"{name}: skipped: {qe.NO_SCORER_REASON}")
        else:
            lines.append(name)
    return lines


def _model_files(cfg: FilterConfig) -> list[tuple[str, Path]]:
    """The config key and path of each LID and SentencePiece model file that
    a configured stage reads."""
    files = []
    if "lid" in cfg.stages:
        files.append(("lid.model_path", cfg.lid.model_path))
    if "length" in cfg.stages and cfg.length.spm_model is not None:
        files.append(("length.spm_model", cfg.length.spm_model))
    return files


def _require_model_files(cfg: FilterConfig) -> None:
    """Fail before any work, and before the skip check, when a model file the
    config names is gone: unchanged inputs must not hide a missing model."""
    for key, path in _model_files(cfg):
        if not path.is_file():
            raise ConfigError(f"{key} is not a file: {path}")


def _output_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / _SCORED_FILE, out_dir / _KEPT_FILE


def _require_corpus_apart_from_outputs(corpus: Path, out_dir: Path) -> None:
    """A run replaces its two output tables, so a corpus that is one of them
    would be overwritten while the manifest recorded its old hash. Compares
    resolved paths, so a symlink or a `..` does not hide the clash."""
    resolved = corpus.resolve()
    for output in _output_paths(out_dir):
        if output.resolve() == resolved:
            raise ConfigError(
                f"--corpus {corpus} is the {output.name} that this run writes "
                f"in {out_dir}; pass a different --out-dir"
            )


def _run_with_models(frame: pl.LazyFrame, cfg: FilterConfig) -> PipelineResult:
    """Run the pipeline with the models its configured stages need, and give
    the QE model's GPU memory back afterwards, also after a failure. The
    pipeline gives LaBSE back itself, right after its stage."""
    embedder = build_embedder(cfg.labse) if "labse" in cfg.stages else None
    qe_scorer = build_qe_scorer(cfg.qe) if "qe" in cfg.stages else None
    try:
        return run_pipeline(frame, cfg, embedder=embedder, qe_scorer=qe_scorer)
    finally:
        if isinstance(qe_scorer, EoleMetricXQEScorer):
            qe_scorer.release()


def _write_run(
    result: PipelineResult,
    kept: pl.DataFrame,
    out_dir: Path,
    config_sha256: str,
    seed: int,
    inputs: list[ManifestFile],
) -> Path:
    """Write the scored and the kept corpus, then the manifest, which comes
    last so that it never describes outputs that are not there. Returns the
    manifest path."""
    begin_commit(out_dir, _STAGE_NAME)
    scored_path, kept_path = _output_paths(out_dir)
    outputs = [
        output_file(scored_path, write_corpus(result.frame, scored_path)),
        output_file(kept_path, write_corpus(kept, kept_path)),
    ]
    manifest = Manifest(
        stage=_STAGE_NAME,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=config_sha256,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        metrics=_metrics(result, kept.height),
        tool_versions=tool_versions(_TOOL_PACKAGES),
        status=result.stage_status,
    )
    return store.write(out_dir, manifest)


def _metrics(result: PipelineResult, rows_kept: int) -> dict[str, float]:
    return {
        "rows_in": float(result.frame.height),
        "rows_kept": float(rows_kept),
        **{
            f"reason.{reason}": float(count)
            for reason, count in result.reason_counts.items()
        },
    }
