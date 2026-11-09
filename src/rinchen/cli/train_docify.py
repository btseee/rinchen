"""`rinchen train docify`: cut a corpus into document windows and prepare the continuation
of our teacher on them (D9). Parsing, file I/O and the manifest only; the work happens in
`rinchen.train` (`docify`, `doc_corpus`). The command is registered on the `train` group
by `rinchen.cli.train`; the checks and the manifest it shares with the other commands are
in `rinchen.cli.train_common`.

    rinchen train docify --corpus C [--corpus C2 ...] --spm-model M --config B
        --out-dir O [--min-size 2] [--max-size 4] [--max-pieces N] [--temperature 1.0]
        [--seed 1234] [--dev-max-windows 1000] [--weight-total 100] [--dry-run]

`--config` is the base config of the continuation (`configs/eole/teacher_doc.yaml`): the
model and the optimiser options the run states for itself, no corpora, no `train_from`
and no `reset_optim` (`rinchen train launch --continue-from` adds those). The windows of
the TRAIN split make the training corpora and those of the DEV split the validation
corpus; no row of the test split is ever read into a window. Only authentic rows with a
document id and a `seg_idx` make windows, and every row of the input is counted once, in a
window or under the reason it is in none (`rows_read = rows_in_windows + the sum of the
excluded counts`).

The piece budget of a window (`--max-pieces`) is what the base config's `filtertoolong`
keeps: a source of at most `src_seq_length` pieces and a target of at most
`tgt_seq_length - 2` (Eole's own length filter counts the two boundary tokens of a
target), `192` for a length the base leaves out, which is Eole's default. A larger value
is refused, because Eole would drop the longer windows without a word; with none given
the budget is that limit.

With `--dry-run` it prints the statistics, the windows and the weight of each direction
and writes nothing, not even the output directory. Without it it prints the same lines,
then

    corpora: <n> (reused <m>)       (reused: written by an earlier run, unchanged)
    validation windows: <n>
    config: <O>/config.yaml
    manifest: <O>/manifest.json

and writes, under O (an absolute path in the files, whatever the working directory):

    data/doc/<src>-<tgt>.doc.src, .tgt, .json    the corpora, one per direction
    valid.src, valid.tgt                         the dev windows, pre-tokenised
    config.yaml                                  the continuation config; its run files
                                                 are O/run/vocab.src and O/run/ckpt
    manifest.json                                stage `train_docify`, written last

Train the continuation with `rinchen train launch --config O/config.yaml --run-dir O/run
--continue-from <teacher>`: the launcher builds the vocabulary file and starts from the
teacher's weights. The manifest's inputs are the base config, the corpus files and the
SentencePiece model, each hashed before it was read; its metrics are every statistic
(`count.<name>` for the train rows, `count.dev.<name>` for the dev rows, dotted as D7's:
`count.excluded.<reason>`, `count.windows_by_size.<n>`, `count.pieces.src.max`), the
options, `valid.windows` and `weight.<corpus>`.

Exit codes: 0; 2 for a bad option, a missing or invalid config, SentencePiece model or
output directory (`$` or a glob character in it), a budget above the base config's; 3 for
a missing or unreadable corpus, another stage's manifest in the output directory, and an
input with no window to cut (the message names the reasons); 1 when the output cannot be
written.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    corpus_manifest_files,
    create_out_dir,
    exit_cannot_write,
    hash_input,
    numbers_under,
    reject_repeated_corpora,
    require_file,
    require_out_apart_from_inputs,
    write_manifest,
)
from rinchen.errors import ConfigError, DataError
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file, refuse_foreign_manifest
from rinchen.train.corpus_files import CorpusFiles
from rinchen.train.doc_corpus import (
    build_doc_corpora,
    check_continuation_base,
    doc_data_section,
    doc_weights,
    planned_doc_corpora,
    render_continuation_config,
    write_doc_validation,
)
from rinchen.train.docify import (
    DEV_WINDOW_SEED,
    DocWindows,
    PieceSummary,
    docify,
    read_corpus_rows,
)
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.train.sampling import CorpusSize
from rinchen.train.validation_corpus import ValidPaths

_STAGE: Final = "train_docify"
_TOOL_PACKAGES: Final = ("sentencepiece", "polars")
_OUTPUTS: Final = "document corpora"
_BASE_CONFIG: Final = "base Eole config"
_CORPUS: Final = "corpus file"
_SPM_MODEL: Final = "SentencePiece model"
# Eole's `filtertoolong` keeps a length it was given no value for at this many pieces, and
# counts the boundary tokens of a target against `tgt_seq_length`.
_FILTERTOOLONG_DEFAULT: Final = 192
_TARGET_BOUNDARY_TOKENS: Final = 2
_RUN_DIR: Final = "run"


@dataclass(frozen=True, slots=True)
class _Options:
    min_size: int
    max_size: int
    max_pieces: int | None
    temperature: float
    seed: int
    dev_max_windows: int
    weight_total: int


@dataclass(frozen=True, slots=True, eq=False)
class _Plan:
    """What a run would do: the windows of both splits, the piece budget they were cut
    with and the corpora with their integer weights."""

    options: _Options
    corpus_files: int
    max_pieces: int
    train: DocWindows
    dev: DocWindows
    corpora: list[CorpusSize]
    weights: dict[str, int]


def docify_command(
    corpus: Annotated[
        list[Path],
        typer.Option(
            "--corpus",
            help=(
                "Corpus Parquet file with document ids and seg_idx; repeat it for more. "
                "Its train rows make the training windows and its dev rows the "
                "validation windows."
            ),
        ),
    ],
    spm_model: Annotated[
        Path,
        typer.Option(
            "--spm-model",
            help="SentencePiece model the windows are measured and tokenised with.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help=(
                "Base Eole config of the continuation (no corpora, no train_from, no "
                "reset_optim): the model and the optimiser options it states."
            ),
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: the corpora, the validation files, config.yaml and the manifest.",
        ),
    ],
    min_size: Annotated[
        int,
        typer.Option(
            "--min-size",
            min=1,
            help="Fewest sentences of a window the sizes are drawn from.",
        ),
    ] = 2,
    max_size: Annotated[
        int,
        typer.Option(
            "--max-size",
            min=1,
            help="Most sentences of a window the sizes are drawn from.",
        ),
    ] = 4,
    max_pieces: Annotated[
        int | None,
        typer.Option(
            "--max-pieces",
            min=1,
            help=(
                "Pieces a window may have on each side (the source counts the tag and "
                "<doc>). Default and ceiling: what the base config's filtertoolong keeps."
            ),
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help=(
                "Mixing temperature over the directions: a direction weighs its window "
                "count to this power (1 is proportional, below 1 lifts the small ones)."
            ),
        ),
    ] = 1.0,
    seed: Annotated[
        int,
        typer.Option(
            "--seed", help="Seed of the window sizes of the training windows."
        ),
    ] = 1234,
    dev_max_windows: Annotated[
        int,
        typer.Option(
            "--dev-max-windows",
            min=1,
            help="Most validation windows of each direction (those with the smallest ids).",
        ),
    ] = 1000,
    weight_total: Annotated[
        int,
        typer.Option(
            "--weight-total",
            min=1,
            help="What the integer weights of the corpora add up to.",
        ),
    ] = 100,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Print the statistics, the windows and the weight of each direction; "
                "write nothing, not even the output directory."
            ),
        ),
    ] = False,
) -> None:
    """Cut a corpus into document windows for the continuation of the teacher.

    Windows of consecutive sentences of one document, joined with <sep>, make one corpus
    per direction; the dev windows make the validation corpus; config.yaml is the base
    config with them, ready for `rinchen train launch --continue-from`. Train nothing."""
    with rinchen_errors_as_exit_codes():
        options = _checked_options(
            _Options(
                min_size,
                max_size,
                max_pieces,
                temperature,
                seed,
                dev_max_windows,
                weight_total,
            )
        )
        reject_repeated_corpora(corpus)
        target = Path(os.path.abspath(out_dir))
        reject_dollar("output directory", out_dir)
        reject_glob_characters("output directory", target)
        reject_dollar(f"{_SPM_MODEL} path", spm_model)
        inputs = _check_or_hash_inputs(config, corpus, spm_model, dry_run=dry_run)
        refuse_foreign_manifest(target, _STAGE)
        require_out_apart_from_inputs(
            out_dir, [config, *corpus, spm_model], option="--out-dir"
        )
        plan = _plan(config, corpus, spm_model, options)
        lines = _statistic_lines(plan)
        if dry_run:
            lines.append("nothing was written")
        else:
            lines += _write(plan, config, spm_model, target, inputs)
        for line in lines:
            typer.echo(line)


def _checked_options(options: _Options) -> _Options:
    if options.min_size > options.max_size:
        raise ConfigError(
            f"--min-size {options.min_size} is above --max-size {options.max_size}"
        )
    if not math.isfinite(options.temperature) or options.temperature <= 0:
        raise ConfigError(
            f"--temperature must be a positive, finite number, got {options.temperature}"
        )
    return options


def _check_or_hash_inputs(
    config: Path, corpus: Sequence[Path], spm_model: Path, *, dry_run: bool
) -> list[ManifestFile]:
    """The manifest entries of the base config, the corpora and the SentencePiece model,
    each taken before anything reads it. A dry run writes no manifest: it only checks
    that the files exist, with the messages a run gives."""
    if dry_run:
        require_file(config, what=_BASE_CONFIG, missing=ConfigError)
        for path in corpus:
            require_file(path, what=_CORPUS, missing=DataError)
        require_file(spm_model, what=_SPM_MODEL, missing=ConfigError)
        return []
    return [
        hash_input(config, what=_BASE_CONFIG, missing=ConfigError),
        *(hash_input(path, what=_CORPUS, missing=DataError) for path in corpus),
        hash_input(spm_model, what=_SPM_MODEL, missing=ConfigError),
    ]


def _plan(
    config: Path, corpus: Sequence[Path], spm_model: Path, options: _Options
) -> _Plan:
    """Read the base config and the corpora, cut the training and the validation
    windows, and weigh the corpora; nothing is written."""
    base = check_continuation_base(config)
    max_pieces = _piece_budget(base, options.max_pieces)
    rows = read_corpus_rows(corpus)
    train = docify(
        rows,
        spm_model=spm_model,
        min_size=options.min_size,
        max_size=options.max_size,
        max_pieces=max_pieces,
        seed=options.seed,
        split="train",
    )
    dev = docify(
        rows,
        spm_model=spm_model,
        min_size=options.min_size,
        max_size=options.max_size,
        max_pieces=max_pieces,
        seed=DEV_WINDOW_SEED,
        split="dev",
    )
    _require_windows(train, "train", max_pieces)
    _require_windows(dev, "dev", max_pieces)
    corpora = planned_doc_corpora(train)
    try:
        weights = doc_weights(
            corpora, temperature=options.temperature, total=options.weight_total
        )
    except ValueError as exc:
        raise ConfigError(f"--temperature {options.temperature}: {exc}") from exc
    return _Plan(options, len(corpus), max_pieces, train, dev, corpora, weights)


def _piece_budget(base: object, requested: int | None) -> int:
    """The pieces a window may have: what `filtertoolong` keeps (see the module's text),
    or `requested` when that is not above it."""
    src_length, tgt_length = _filtertoolong_lengths(base)
    limit = min(src_length, tgt_length - _TARGET_BOUNDARY_TOKENS)
    if limit < 1:
        raise ConfigError(
            f"the {_BASE_CONFIG} keeps no window: filtertoolong has src_seq_length "
            f"{src_length} and tgt_seq_length {tgt_length}, and a target needs "
            f"{_TARGET_BOUNDARY_TOKENS} more tokens than its pieces"
        )
    if requested is None:
        return limit
    if requested > limit:
        raise ConfigError(
            f"--max-pieces {requested} is above what the base config keeps: "
            f"filtertoolong drops a source over src_seq_length {src_length} and a target "
            f"over tgt_seq_length {tgt_length} minus {_TARGET_BOUNDARY_TOKENS}, so a "
            f"window may have at most {limit} pieces; the longer ones would be dropped "
            "without a word"
        )
    return requested


def _filtertoolong_lengths(base: object) -> tuple[int, int]:
    """`(src_seq_length, tgt_seq_length)` of the base config's `filtertoolong`, Eole's
    default for one it leaves out."""
    configs = base.get("transforms_configs") if isinstance(base, Mapping) else None
    section = configs.get("filtertoolong") if isinstance(configs, Mapping) else None
    given: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
    return (
        _seq_length(given, "src_seq_length"),
        _seq_length(given, "tgt_seq_length"),
    )


def _seq_length(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key, _FILTERTOOLONG_DEFAULT)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"invalid {_BASE_CONFIG}: transforms_configs.filtertoolong.{key}: must be a "
            f"positive integer, got {value!r}"
        )
    return value


def _require_windows(windows: DocWindows, split: str, max_pieces: int) -> None:
    """Refuse a split that has no window, saying why every row was left out: a corpus
    of nothing must never be written, or trained on, silently."""
    if not windows.frame.is_empty():
        return
    stats = windows.stats
    reasons = ", ".join(
        f"{reason} {count}" for reason, count in stats.excluded.items() if count
    )
    what = "document windows" if split == "train" else f"{split} document windows"
    raise DataError(
        f"no {what}: none of the {stats.rows_read} rows read can be in a {split} window"
        f"{f' ({reasons})' if reasons else ''}; the piece budget is --max-pieces "
        f"{max_pieces}, and a window needs at least one sentence that fits it"
    )


def _statistic_lines(plan: _Plan) -> list[str]:
    """What the run counted and decided, one fact per line."""
    stats = plan.train.stats
    return [
        f"corpus files: {plan.corpus_files}",
        f"max pieces: {plan.max_pieces}",
        f"rows read: {stats.rows_read}",
        f"rows in windows: {stats.rows_in_windows}",
        f"windows: {stats.windows}",
        *(
            f"windows of size {size}: {count}"
            for size, count in stats.windows_by_size.items()
        ),
        *(f"excluded {reason}: {count}" for reason, count in stats.excluded.items()),
        *(
            f"pieces {side}: {_pieces_line(summary)}"
            for side, summary in stats.pieces.items()
        ),
        f"dev windows: {plan.dev.stats.windows}",
        *(
            f"corpus {corpus.name}: windows {corpus.rows}, weight {plan.weights[corpus.name]}"
            for corpus in plan.corpora
        ),
    ]


def _pieces_line(summary: PieceSummary) -> str:
    return f"min {summary.min}, mean {summary.mean:.3f}, p95 {summary.p95}, max {summary.max}"


def _write(
    plan: _Plan,
    config: Path,
    spm_model: Path,
    target: Path,
    inputs: list[ManifestFile],
) -> list[str]:
    """Write the corpora, the validation files and the config, then the manifest, last;
    return the lines that say so. `target` is the absolute output directory."""
    create_out_dir(target, what=_OUTPUTS)
    spm = Path(os.path.abspath(spm_model))
    try:
        # Just before the first write, and after everything that can fail on bad input:
        # the old manifest goes first, so that a failure part-way leaves none.
        begin_commit(target, _STAGE)
        corpora = build_doc_corpora(plan.train, target / "data")
        valid = write_doc_validation(
            plan.dev, spm, target, max_windows=plan.options.dev_max_windows
        )
        data = doc_data_section(corpora, plan.weights, valid_paths=valid)
        run_dir = target / _RUN_DIR
        config_path = render_continuation_config(
            config,
            data=data,
            spm_model=spm,
            src_vocab=run_dir / "vocab.src",
            model_path=run_dir / "ckpt",
            out=target / "config.yaml",
        )
        valid_windows = _line_count(valid.src)
        manifest_path = write_manifest(
            target,
            stage=_STAGE,
            config_sha256=inputs[0].sha256,
            seed=plan.options.seed,
            inputs=inputs,
            outputs=_output_files(corpora, valid, config_path, valid_windows),
            metrics=_metrics(plan, corpora, valid_windows),
            packages=_TOOL_PACKAGES,
            status={"docify": "ok"},
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)
    reused = sum(corpus.reused for corpus in corpora)
    return [
        f"corpora: {len(corpora)} (reused {reused})",
        f"validation windows: {valid_windows}",
        f"config: {config_path}",
        f"manifest: {manifest_path}",
    ]


def _output_files(
    corpora: Sequence[CorpusFiles],
    valid: ValidPaths,
    config_path: Path,
    valid_windows: int,
) -> list[ManifestFile]:
    """The config, the validation files, then each corpus's files with the digests and
    row counts the build recorded or checked against the files."""
    return [
        output_file(config_path, None),
        output_file(valid.src, valid_windows),
        output_file(valid.tgt, valid_windows),
        *(file for corpus in corpora for file in corpus_manifest_files(corpus)),
    ]


def _metrics(
    plan: _Plan, corpora: Sequence[CorpusFiles], valid_windows: int
) -> dict[str, float]:
    """The statistics of the train rows (`count.*`) and the dev rows (`count.dev.*`), the
    options the windows were cut with and the weight of each corpus."""
    options = plan.options
    return {
        **numbers_under("count", plan.train.stats.to_dict()),
        **numbers_under("count.dev", plan.dev.stats.to_dict()),
        "max_pieces": float(plan.max_pieces),
        "min_size": float(options.min_size),
        "max_size": float(options.max_size),
        "temperature": options.temperature,
        "weight_total": float(options.weight_total),
        "dev_max_windows": float(options.dev_max_windows),
        "corpora.files": float(len(corpora)),
        "valid.windows": float(valid_windows),
        **{f"weight.{name}": float(weight) for name, weight in plan.weights.items()},
    }


def _line_count(path: Path) -> int:
    return path.read_bytes().count(b"\n")
