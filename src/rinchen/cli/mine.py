"""`rinchen mine`: align the sentences of two text files into a mined corpus
Parquet file and record the run in a manifest. Presentation, file I/O and the
manifest only; the alignment happens in `rinchen.mine.align`."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
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
from rinchen.config.languages import known_language_codes
from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    build_inputs,
    code_version,
    sha256_of,
    should_skip,
    tool_versions,
)
from rinchen.mine.align import Alignment, align
from rinchen.mine.embed import Embedder, SentenceTransformerEmbedder

app = typer.Typer(no_args_is_help=True)

_STAGE_NAME = "mine_align"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "mined"
_OUT_FILE = "aligned.parquet"
_TOOL_PACKAGES = ("polars", "numpy", "sentence-transformers", "torch")

# The options default to the aligner's own parameters.
_ALIGN_PARAMETERS = inspect.signature(align).parameters
_DEFAULT_MAX_K: int = _ALIGN_PARAMETERS["max_k"].default
_DEFAULT_SKIP_COST: float = _ALIGN_PARAMETERS["skip_cost"].default
_DEFAULT_MERGE_PENALTY: float = _ALIGN_PARAMETERS["merge_penalty"].default
_MARGIN_K: int = _ALIGN_PARAMETERS["margin_k"].default
_MIN_MAX_K = 2


@dataclass(frozen=True, slots=True)
class _AlignRun:
    """Everything that, besides the input files, decides what a run writes."""

    src_lang: str
    tgt_lang: str
    corpus: str
    max_k: int
    skip_cost: float
    merge_penalty: float
    margin_k: int = _MARGIN_K


@dataclass(frozen=True, slots=True)
class _Sentences:
    lines: list[str]
    blank_lines_dropped: int


def build_embedder() -> Embedder:
    """The LaBSE embedder of a run; it loads its model on first use. Tests
    replace this function."""
    return SentenceTransformerEmbedder()


@app.command("align")
def align_files(
    src: Annotated[
        Path, typer.Option("--src", help="Source text file, one sentence per line.")
    ],
    tgt: Annotated[
        Path, typer.Option("--tgt", help="Target text file, one sentence per line.")
    ],
    src_lang: Annotated[
        str, typer.Option("--src-lang", help="Source language code, e.g. khk_Cyrl.")
    ],
    tgt_lang: Annotated[
        str, typer.Option("--tgt-lang", help="Target language code, e.g. eng_Latn.")
    ],
    corpus: Annotated[
        str,
        typer.Option(
            "--corpus", help="Corpus name, recorded in every row and in the out path."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Output Parquet file; default data/mined/<corpus>/aligned.parquet. "
                "The manifest is written beside it as manifest.json."
            ),
        ),
    ] = None,
    max_k: Annotated[
        int,
        typer.Option(
            "--max-k",
            help=(
                "Largest group: a source and b target sentences are grouped "
                "only if a + b <= max-k."
            ),
        ),
    ] = _DEFAULT_MAX_K,
    skip_cost: Annotated[
        float,
        typer.Option("--skip-cost", help="Cost of leaving one sentence unaligned."),
    ] = _DEFAULT_SKIP_COST,
    merge_penalty: Annotated[
        float,
        typer.Option(
            "--merge-penalty",
            help="Extra cost per sentence beyond a 1-1 group.",
        ),
    ] = _DEFAULT_MERGE_PENALTY,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Run even when inputs and parameters are unchanged."
        ),
    ] = False,
) -> None:
    """Align two sentence files into a corpus of mined sentence pairs.

    The files are taken as one document pair, so the alignment needs memory in
    proportion to the product of their lengths."""
    with rinchen_errors_as_exit_codes():
        run = _validated_run(
            src_lang, tgt_lang, corpus, max_k, skip_cost, merge_penalty
        )
        for path in (src, tgt):
            _require_sentence_file(path)

        target = out if out is not None else DEFAULT_OUT_ROOT / corpus / _OUT_FILE
        config_sha256 = _config_sha256(run)
        if _outputs_are_current(target, [src, tgt], config_sha256, force=force):
            typer.echo(
                "skipped: inputs and parameters are unchanged since the last run"
            )
            typer.echo(f"manifest: {store.manifest_path(target.parent)}")
            return

        src_sentences = _read_sentences(src)
        tgt_sentences = _read_sentences(tgt)
        alignments = _align_sentences(run, src_sentences.lines, tgt_sentences.lines)
        frame = _aligned_frame(
            alignments, src_sentences.lines, tgt_sentences.lines, run, src.stem
        )
        metrics = _metrics(run, alignments, src_sentences, tgt_sentences)
        try:
            manifest_path = _write_run(
                frame, target, config_sha256, build_inputs([src, tgt]), metrics
            )
        except OSError as exc:
            exit_with_error(
                f"cannot write the alignment outputs under {target.parent}: {exc}",
                FAILURE_EXIT_CODE,
            )
        typer.echo(
            f"aligned {frame.height} rows (unaligned: "
            f"{int(metrics['unaligned_src'])} source, {int(metrics['unaligned_tgt'])} target)"
        )
        typer.echo(f"manifest: {manifest_path}")


def _validated_run(
    src_lang: str,
    tgt_lang: str,
    corpus: str,
    max_k: int,
    skip_cost: float,
    merge_penalty: float,
) -> _AlignRun:
    """Raises ConfigError for a parameter the aligner or the output cannot
    take, before any file is read."""
    if max_k < _MIN_MAX_K:
        raise ConfigError(f"--max-k must be at least {_MIN_MAX_K}, got {max_k}")
    for option, cost in (
        ("--skip-cost", skip_cost),
        ("--merge-penalty", merge_penalty),
    ):
        if not (math.isfinite(cost) and cost >= 0.0):
            raise ConfigError(f"{option} must be finite and non-negative, got {cost}")
    for option, language in (("--src-lang", src_lang), ("--tgt-lang", tgt_lang)):
        if language not in known_language_codes():
            raise ConfigError(
                f"{option} {language!r} is not in the language registry "
                "(configs/languages.yaml)"
            )
    if not corpus.strip() or corpus in (".", "..") or "/" in corpus or "\\" in corpus:
        raise ConfigError(
            f"--corpus must be a plain name, without path separators, got {corpus!r}"
        )
    return _AlignRun(src_lang, tgt_lang, corpus, max_k, skip_cost, merge_penalty)


def _config_sha256(run: _AlignRun) -> str:
    """The run has no config file, so its parameters stand in for one: the
    SHA-256 of their canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(asdict(run), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_sentence_file(path: Path) -> None:
    if not path.is_file():
        raise DataError(f"sentence file not found: {path}")


def _read_sentences(path: Path) -> _Sentences:
    """The stripped, non-blank lines of a UTF-8 file (a byte order mark is
    ignored), and how many blank lines were left out. Only CR, LF and CRLF end
    a line: a sentence may contain other Unicode line separators."""
    lines: list[str] = []
    blank = 0
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    lines.append(line)
                else:
                    blank += 1
    except UnicodeDecodeError as exc:
        raise DataError(f"{path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    return _Sentences(lines, blank)


def _outputs_are_current(
    out: Path, input_paths: list[Path], config_sha256: str, *, force: bool
) -> bool:
    # `should_skip` compares the manifest only; a deleted output must not pass,
    # nor an output that the manifest in this directory does not describe.
    if force or not out.is_file():
        return False
    recorded = store.read(out.parent)
    describes_out = recorded is not None and any(
        item.path == str(out) for item in recorded.outputs
    )
    return describes_out and should_skip(out.parent, input_paths, config_sha256)


def _align_sentences(run: _AlignRun, src: list[str], tgt: list[str]) -> list[Alignment]:
    embedder = build_embedder()
    try:
        return align(
            src,
            tgt,
            embedder,
            run.max_k,
            skip_cost=run.skip_cost,
            margin_k=run.margin_k,
            merge_penalty=run.merge_penalty,
        )
    except ValueError as exc:
        # The parameters were checked, so this is the embedder's output.
        raise UpstreamError(
            f"embedding model returned unusable vectors: {exc}"
        ) from exc


def _aligned_frame(
    alignments: list[Alignment],
    src: list[str],
    tgt: list[str],
    run: _AlignRun,
    doc_id: str,
) -> pl.DataFrame:
    """One row per alignment with sentences on both sides; the sentences of a
    group are joined by one space. Deletions have no row."""
    pairs = [a for a in alignments if a.src and a.tgt]
    return corpus_frame(
        [
            {
                "src_lang": run.src_lang,
                "tgt_lang": run.tgt_lang,
                "src": _joined(src, pair.src),
                "tgt": _joined(tgt, pair.tgt),
                "origin": "mined",
                "corpus": run.corpus,
                "split": "train",
                "doc_id": doc_id,
                "seg_idx": seg_idx,
            }
            for seg_idx, pair in enumerate(pairs)
        ]
    )


def _joined(sentences: list[str], indexes: tuple[int, ...]) -> str:
    return " ".join(sentences[index] for index in indexes)


def _metrics(
    run: _AlignRun,
    alignments: list[Alignment],
    src: _Sentences,
    tgt: _Sentences,
) -> dict[str, float]:
    return {
        "aligned": float(sum(1 for a in alignments if a.src and a.tgt)),
        "unaligned_src": float(sum(1 for a in alignments if a.src and not a.tgt)),
        "unaligned_tgt": float(sum(1 for a in alignments if a.tgt and not a.src)),
        "src_sentences": float(len(src.lines)),
        "tgt_sentences": float(len(tgt.lines)),
        "blank_lines_dropped": float(src.blank_lines_dropped + tgt.blank_lines_dropped),
        "param.max_k": float(run.max_k),
        "param.skip_cost": run.skip_cost,
        "param.merge_penalty": run.merge_penalty,
    }


def _write_run(
    frame: pl.DataFrame,
    out: Path,
    config_sha256: str,
    inputs: list[ManifestFile],
    metrics: dict[str, float],
) -> Path:
    """Write the corpus, then the manifest, which comes last so that it never
    describes an output that is not there. Returns the manifest path."""
    # The manifest is the commit marker: drop the old one before touching the
    # output, so a failure part-way leaves no manifest that vouches for it.
    store.manifest_path(out.parent).unlink(missing_ok=True)
    output = _written_file(frame, out)
    manifest = Manifest(
        stage=_STAGE_NAME,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=config_sha256,
        seed=0,
        inputs=inputs,
        outputs=[output],
        metrics=metrics,
        tool_versions=tool_versions(_TOOL_PACKAGES),
    )
    return store.write(out.parent, manifest)


def _written_file(frame: pl.DataFrame, path: Path) -> ManifestFile:
    rows = write_corpus(frame, path)
    return ManifestFile(
        path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size, rows=rows
    )
