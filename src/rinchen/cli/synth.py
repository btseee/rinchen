"""`rinchen synth run`: run one synthetic-data job (back- or forward-translation of
a monolingual corpus) with its generator and record the run in a manifest.
Parsing, file I/O and the manifest only; the work happens in `rinchen.synth`.

The generator's shards are resumable work files under the output directory, not
outputs of the run: the outputs are `corpus.parquet` and `counts.json`, and the
manifest that vouches for them comes last.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NamedTuple, NoReturn

import polars as pl
import typer

from rinchen.cli.errors import (
    FAILURE_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.config.synth import (
    SYNTH_CONFIG_PATH,
    Ct2Config,
    GeneratorConfig,
    JobConfig,
    LlamaCppConfig,
    SelectionConfig,
    SynthConfig,
    VllmConfig,
    load_synth_config,
)
from rinchen.corpus.writers import write_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.filter import qe
from rinchen.filter.qe import QEScorer
from rinchen.filter.qe_eole import EoleMetricXQEScorer
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    begin_commit,
    build_inputs,
    code_version,
    output_file,
    refuse_foreign_manifest,
    tool_versions,
)
from rinchen.mine.embed import Releasable
from rinchen.synth.ct2_baselines import CT2Translator
from rinchen.synth.llamacpp_client import LlamaCppTranslator
from rinchen.synth.run import RunResult, ShardProgress, plan_run, run_job
from rinchen.synth.select import QE_SKIPPED_NO_SCORER
from rinchen.synth.shards import ShardStore, read_monolingual
from rinchen.synth.translator import Translator
from rinchen.synth.vllm_client import OpenAICompatTranslator
from rinchen.utils.part_file import unique_part_path

app = typer.Typer(no_args_is_help=True)

_STAGE_NAME = "synth"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "synth"
_CORPUS_FILE = "corpus.parquet"
_COUNTS_FILE = "counts.json"
_TOOL_PACKAGES = (
    "httpx",
    "polars",
    "sacrebleu",
    "ctranslate2",
    "sentencepiece",
    "eole",
    "torch",
)
_NO_QE_STATUS = "skipped: --no-qe"


def build_translator(cfg: GeneratorConfig) -> Translator:
    """The translator of a generator. The client for a server only connects when
    it is asked to translate; the CTranslate2 one loads its model here. Tests
    replace this function."""
    match cfg:
        case VllmConfig():
            return OpenAICompatTranslator(cfg)
        case LlamaCppConfig():
            return LlamaCppTranslator(cfg)
        case Ct2Config():
            return CT2Translator(cfg)


def build_qe_scorer(cfg: SelectionConfig) -> QEScorer | None:
    """The QE scorer of a run, or None when none exists for `cfg.qe.scorer`
    (the run then skips the QE floor). It loads its model on first use. Tests
    replace this function."""
    return EoleMetricXQEScorer() if cfg.qe.scorer == "metricx_qe" else None


@app.command("run")
def run(
    job: Annotated[
        str,
        typer.Argument(
            metavar="JOB", help="Name of a job of the config, e.g. bt_eng_khk."
        ),
    ],
    monolingual: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Monolingual Parquet file (columns id, lang, text) to translate.",
        ),
    ],
    config: Annotated[
        Path, typer.Option("--config", help="Synthetic-data config (YAML).")
    ] = SYNTH_CONFIG_PATH,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Output directory; default data/synth/<JOB>."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help=(
                "Part of the job fingerprint, which names the shard directory, "
                "and recorded in the manifest."
            ),
        ),
    ] = 0,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Generate every shard again instead of reusing the files."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Print the job, the generator, the rows, the shards and the "
                "fingerprint; build no model, write nothing."
            ),
        ),
    ] = False,
    no_qe: Annotated[
        bool,
        typer.Option("--no-qe", help="Skip the QE floor of the selection."),
    ] = False,
) -> None:
    """Run one synthetic-data job and write its corpus, counts and manifest.

    The job's generator translates each text of the monolingual corpus, the best
    candidate of each text is kept, and a rerun resumes after the shards already
    generated (kept in <out-dir>/shards-<fingerprint>/). The monolingual rows are
    held in memory."""
    with rinchen_errors_as_exit_codes():
        # Each input is hashed before it is read, so that the manifest vouches
        # for the bytes the run started from.
        config_file = _hash_input(config, what="synth config file", missing=ConfigError)
        cfg = load_synth_config(config)
        job_cfg = _job_named(cfg, job)
        target = out_dir if out_dir is not None else DEFAULT_OUT_ROOT / job
        # Refused before anything is read or written, also for a dry run, which
        # must not be followed by a refusal for this.
        refuse_foreign_manifest(target, _STAGE_NAME)
        _require_out_apart_from_input(target, monolingual)
        monolingual_file = _hash_input(
            monolingual, what="monolingual file", missing=DataError
        )
        frame = read_monolingual(monolingual, lang=job_cfg.mono_lang)
        generator_cfg = cfg.generators[job_cfg.generator]
        if dry_run:
            for line in _plan_lines(
                job, job_cfg, generator_cfg, frame, target, cfg=cfg, seed=seed
            ):
                typer.echo(line)
            return

        # Before any model is built or any text is generated.
        _create_out_dir(target)
        record = _RunRecord(
            job=job,
            generator=job_cfg.generator,
            seed=seed,
            config_sha256=config_file.sha256,
            inputs=[monolingual_file, config_file],
        )
        translator = build_translator(generator_cfg)
        scorer, skipped_status = _qe_scorer_of(cfg, no_qe=no_qe)
        try:
            with _released_after(scorer):
                result = run_job(
                    job_cfg,
                    job_cfg.generator,
                    generator_cfg,
                    translator,
                    frame,
                    out_dir=target,
                    cfg=cfg,
                    scorer=scorer,
                    seed=seed,
                    force=force,
                    progress=_report_shard,
                )
            qe_status = skipped_status or result.qe_status
            written = _commit(target, record, result, qe_status)
        except OSError as exc:
            _exit_cannot_write(target, exc)
        typer.echo(f"corpus: {written.corpus}")
        typer.echo(f"counts: {written.counts}")
        typer.echo(f"manifest: {written.manifest}")
        typer.echo(
            f"rows {result.corpus.height} (selected {result.counts['selected']} "
            f"of {result.counts['sources']} sources, qe {qe_status})"
        )


def _job_named(cfg: SynthConfig, job: str) -> JobConfig:
    job_cfg = cfg.jobs.get(job)
    if job_cfg is None:
        configured = ", ".join(sorted(cfg.jobs)) or "none"
        raise ConfigError(f"unknown job {job!r}; configured jobs: {configured}")
    return job_cfg


def _plan_lines(
    job: str,
    job_cfg: JobConfig,
    generator_cfg: GeneratorConfig,
    frame: pl.DataFrame,
    out_dir: Path,
    *,
    cfg: SynthConfig,
    seed: int,
) -> list[str]:
    """What a run would do with `frame`: the job, its generator, the rows, the
    shards (and how many are already on disk) and the fingerprint that names
    their directory, all as `plan_run` plans them for the run."""
    run_plan = plan_run(
        job_cfg,
        job_cfg.generator,
        generator_cfg,
        frame,
        out_dir=out_dir,
        cfg=cfg,
        seed=seed,
    )
    valid = ShardStore(run_plan.shard_directory).completed(run_plan.shards)
    return [
        f"job: {job}",
        f"kind: {job_cfg.kind}",
        f"generator: {job_cfg.generator} ({generator_cfg.kind})",
        f"input rows: {frame.height}",
        f"shards: {len(run_plan.shards)} (shard_size {job_cfg.shard_size})",
        f"fingerprint: {run_plan.fingerprint}",
        f"shard directory: {run_plan.shard_directory}",
        f"valid shards present: {len(valid)}",
        f"out dir: {out_dir}",
    ]


def _hash_input(
    path: Path, *, what: str, missing: type[ConfigError | DataError]
) -> ManifestFile:
    """The manifest entry of the input file `path`, taken before anything reads
    it. A path that is not a file is `missing`, with the message of the loader
    that would have refused it; a file that cannot be read is a DataError."""
    if not path.is_file():
        raise missing(f"{what} not found: {path}")
    try:
        return build_inputs([path])[0]
    except OSError as exc:
        raise DataError(f"cannot read {what} {path}: {exc}") from exc


def _create_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _exit_cannot_write(out_dir, exc)


def _exit_cannot_write(out_dir: Path, exc: OSError) -> NoReturn:
    exit_with_error(
        f"cannot write the synth outputs under {out_dir}: {exc}", FAILURE_EXIT_CODE
    )


def _require_out_apart_from_input(out_dir: Path, source: Path) -> None:
    """The run writes into `out_dir`, so an input file that lives there could be
    replaced by an output. Compares resolved paths, so a symlink or a `..` does
    not hide the clash, for the file as named and for the file it links to."""
    resolved = out_dir.resolve()
    if resolved in (source.resolve().parent, source.parent.resolve()):
        raise ConfigError(
            f"--out-dir {out_dir} is the directory of the input file {source}; "
            "pass a different --out-dir"
        )


def _qe_scorer_of(
    cfg: SynthConfig, *, no_qe: bool
) -> tuple[QEScorer | None, str | None]:
    """The scorer the run uses and, when it uses none, the QE status that says
    why. `--no-qe` builds no scorer at all."""
    if no_qe:
        return None, _NO_QE_STATUS
    scorer = build_qe_scorer(cfg.selection)
    if scorer is None:
        return None, QE_SKIPPED_NO_SCORER
    reason = qe.skip_reason(cfg.selection.qe, scorer)
    if reason is not None:
        return None, f"skipped: {reason}"
    return scorer, None


@contextmanager
def _released_after(scorer: QEScorer | None) -> Iterator[None]:
    """Give the scorer's model back after the block, also after a failure."""
    try:
        yield
    finally:
        if isinstance(scorer, Releasable):
            scorer.release()


def _report_shard(progress: ShardProgress) -> None:
    how = "reused" if progress.reused else "generated"
    typer.echo(f"shard {progress.index + 1}/{progress.total} {how}", err=True)


@dataclass(frozen=True, slots=True)
class _RunRecord:
    """What the manifest and the counts file say about a run besides its result."""

    job: str
    generator: str
    seed: int
    config_sha256: str
    inputs: list[ManifestFile]


class _Written(NamedTuple):
    corpus: Path
    counts: Path
    manifest: Path


def _commit(
    out_dir: Path, record: _RunRecord, result: RunResult, qe_status: str
) -> _Written:
    """Write the corpus and the counts, then the manifest, which comes last so
    that it never describes outputs that are not there. The old manifest goes
    first, so that a failure part-way leaves none; it is still in place while the
    shards are generated, which are not outputs."""
    begin_commit(out_dir, _STAGE_NAME)
    corpus_path = out_dir / _CORPUS_FILE
    counts_path = out_dir / _COUNTS_FILE
    rows = write_corpus(result.corpus, corpus_path)
    _write_counts(counts_path, record, result, qe_status)
    manifest = Manifest(
        stage=_STAGE_NAME,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=record.config_sha256,
        seed=record.seed,
        inputs=record.inputs,
        outputs=[output_file(corpus_path, rows), output_file(counts_path, None)],
        metrics={
            **{f"count.{key}": float(value) for key, value in result.counts.items()},
            **{
                f"shards.{key}": float(value)
                for key, value in result.shard_stats.items()
            },
        },
        tool_versions=tool_versions(_TOOL_PACKAGES),
        status={"qe": qe_status},
    )
    return _Written(corpus_path, counts_path, store.write(out_dir, manifest))


def _write_counts(
    path: Path, record: _RunRecord, result: RunResult, qe_status: str
) -> None:
    """Write the counts file so that it appears whole or not at all."""
    document = {
        "fingerprint": result.fingerprint,
        "job": record.job,
        "generator": record.generator,
        "counts": dict(result.counts),
        "shard_stats": dict(result.shard_stats),
        "qe_status": qe_status,
    }
    part_path = unique_part_path(path)
    try:
        part_path.write_text(
            json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)
