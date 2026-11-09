"""`rinchen distill prepare`: lay out the data and the config a student trains on, from the
original corpora and the KD corpora a teacher made (D10). Parsing, file I/O and the manifest
only; the work happens in `rinchen.train` (`prepare_student`, `load_kd_rows`, `parse_mix`).
The command is registered on the `distill` group by `rinchen.cli.distill`.

    rinchen distill prepare --corpus C [--corpus C2 ...] --kd K [--kd K2 ...]
        --spm-model M --config B --out-dir O [--mix 1:1] [--no-kd-tag]
        [--tag-placement src|tgt] [--bt-ratio 0.0] [--temperature 1.0] [--seed 1234]
        [--dev-max-lines 1000] [--weight-total 100] [--dry-run]

`--corpus` files give the original corpora (authentic and back-translated train rows) and the
validation (dev rows); `--kd` files are KD corpora that `rinchen distill generate` wrote, each
with the `manifest.json` of that run beside it, which must vouch for the file as it is now.
`--spm-model` is the STUDENT's SentencePiece model (it cuts the validation corpus and sets the
vocabulary size); `--config` is the student's base config (`configs/eole/student_*.yaml`: the
shape and the recipe, no corpora, no `train_from`, no `reset_optim`: a student starts from
random initialisation, always).

The original corpora are weighed as D7 does (`--bt-ratio`, `--temperature`, `--weight-total`);
`--mix K:O` (two positive whole numbers, reduced by their greatest common divisor) then makes
exactly K/(K+O) of the examples Eole draws, in every direction, come from the KD corpus. A KD
corpus carries the origin tag `<ft>` unless `--no-kd-tag` leaves it plain: untagged, Eole
drops (silently) a KD row whose source repeats an original source in the same minibatch,
which was 5.2 % of the examples on the toy. `--tag-placement` says where the direction tag
goes, as for `rinchen train ablate`; use the same one when the student is evaluated.

A run prints the statistics (the rows read and left out of the original corpora, the dev
lines of each direction, the KD rows read, left out by reason and kept), `mix: <K>:<O>`,
`kd tag: ft | none`, for every corpus its rows and integer weight and, for each KD corpus,
how many times more often one of its rows is drawn than one original row of its direction
(`repeat <direction>.kd: <factor>`: recorded, never a gate). With `--dry-run` it then says
`nothing was written` and writes nothing, not even the output directory. Without it it prints

    corpora: <n> (reused <m>)       (reused: written by an earlier run, unchanged)
    validation lines: <n>
    config: <O>/config.yaml
    manifest: <O>/manifest.json

and writes, under O (an absolute path in the files, whatever the working directory):

    data/auth/..., data/bt/r<ratio>_s<seed>/...   the original corpora (D7's files)
    data/kd/<direction>.kd.src, .tgt, .json       the KD corpora, one per direction
    valid.src, valid.tgt                          the dev lines cut with the student's model
    config.yaml                                   the student's config; its run files are
                                                  O/run/vocab.src and O/run/ckpt
    manifest.json                                 stage `distill_prepare`, written last

Train it with `rinchen train launch --config O/config.yaml --run-dir O/run`. The manifest's
inputs are the base config, the corpus files, the KD files with their manifests and the
SentencePiece model, each hashed once, before anything read it (a KD file's digest is the one
the loader checks against its manifest); its metrics are the statistics of the original rows
(`count.*`, `count.dev.*`) and of the KD rows (`kd.count.*`, `kd.count.excluded.<reason>`), the
mix (`mix.kd`, `mix.original`), `kd_tag`, `weight.<corpus>`, `kd.repeat.<direction>`,
`valid.lines` and the options. The old manifest of the directory is removed only after every
input was read and every check passed, just before the first file is written.

Exit codes: 0; 2 for a bad option (`--mix` is checked before anything is read), a base config
or SentencePiece model that cannot be used, a KD file its manifest does not vouch for, an
output directory (or model path) that holds `$` or a glob character, or that is the directory
of an input; 3 for a missing or unreadable corpus, KD file or KD manifest, no row to train on
and another stage's manifest in the output directory; 1 when the output cannot be written; 130
when interrupted.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, NamedTuple

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    corpus_manifest_files,
    create_out_dir,
    exit_cannot_write,
    hash_input,
    numbers_under,
    reject_repeated_corpora,
    require_out_apart_from_inputs,
    write_manifest,
)
from rinchen.config.ablation import TagPlacement
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file, refuse_foreign_manifest
from rinchen.train.distill_mix import Mix, parse_mix
from rinchen.train.distill_prepare import PreparedStudent, prepare_student
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.train.validation_corpus import validation_paths

_STAGE: Final = "distill_prepare"
_TOOL_PACKAGES: Final = ("sentencepiece", "polars")
_OUTPUTS: Final = "student run"
_BASE_CONFIG: Final = "base Eole config"
_CORPUS: Final = "corpus file"
_KD_FILE: Final = "KD file"
_KD_MANIFEST: Final = "KD manifest"
_SPM_MODEL: Final = "SentencePiece model"


@dataclass(frozen=True, slots=True)
class _Paths:
    """The inputs of a run as given on the command line, and the absolute output
    directory."""

    config: Path
    corpus: Sequence[Path]
    kd: Sequence[Path]
    spm_model: Path
    out_dir: Path


@dataclass(frozen=True, slots=True)
class _Options:
    """The options that decide what a run makes."""

    mixture: Mix
    kd_tag: bool
    tag_placement: TagPlacement
    bt_ratio: float
    temperature: float
    seed: int
    dev_max_lines: int
    weight_total: int


class _Hashed(NamedTuple):
    """The inputs of a run, hashed before anything read them: the manifest entries in the
    order of the manifest, the digest of each KD file (the loader checks them against the
    KD manifests) and every path whose directory the output directory must not be."""

    entries: list[ManifestFile]
    base_sha256: str
    kd_digests: dict[Path, str]
    sources: list[Path]


class _Output:
    """The output directory of a run: made, and the old manifest removed, when the library
    is about to write its first file. `begun` says whether that happened, so that an OSError
    is told apart as a failure to write and a failure to read."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.begun = False

    def begin(self) -> None:
        self.begun = True
        create_out_dir(self.target, what=_OUTPUTS)
        begin_commit(self.target, _STAGE)


def prepare_command(
    corpus: Annotated[
        list[Path],
        typer.Option(
            "--corpus",
            help=(
                "Corpus Parquet file; repeat it for more. Its train rows are the "
                "original corpora and its dev rows the validation."
            ),
        ),
    ],
    kd: Annotated[
        list[Path],
        typer.Option(
            "--kd",
            help=(
                "KD corpus Parquet file written by `distill generate`, with its "
                "manifest.json beside it; repeat it for more."
            ),
        ),
    ],
    spm_model: Annotated[
        Path,
        typer.Option(
            "--spm-model",
            help="The student's SentencePiece model (not necessarily the teacher's).",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help=(
                "Base Eole config of the student (no corpora, no train_from, no "
                "reset_optim): its shape and recipe."
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
    mix: Annotated[
        str,
        typer.Option(
            "--mix",
            help=(
                "K:O, two positive whole numbers: in every direction K/(K+O) of the "
                "examples drawn come from the KD corpus."
            ),
        ),
    ] = "1:1",
    no_kd_tag: Annotated[
        bool,
        typer.Option(
            "--no-kd-tag",
            help="Leave the KD corpora without the origin tag <ft> (not recommended).",
        ),
    ] = False,
    tag_placement: Annotated[
        TagPlacement,
        typer.Option(
            "--tag-placement",
            help=(
                "Where the direction tag goes: in the source (src) or as a forced prefix "
                "of the target (tgt)."
            ),
        ),
    ] = "src",
    bt_ratio: Annotated[
        float,
        typer.Option(
            "--bt-ratio",
            help="Back-translated rows per authentic row of a direction (D7's ratio).",
        ),
    ] = 0.0,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help=(
                "Mixing temperature over the directions of the original corpora: a "
                "direction weighs its rows to this power (1 is proportional, below 1 "
                "lifts the small ones)."
            ),
        ),
    ] = 1.0,
    seed: Annotated[
        int,
        typer.Option(
            "--seed", help="Seed of the back-translated subsample and of the training."
        ),
    ] = 1234,
    dev_max_lines: Annotated[
        int,
        typer.Option(
            "--dev-max-lines",
            min=1,
            help="Most validation lines of each direction (those with the smallest ids).",
        ),
    ] = 1000,
    weight_total: Annotated[
        int,
        typer.Option(
            "--weight-total",
            min=1,
            help="What the integer weights of the original corpora add up to.",
        ),
    ] = 100,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Print the statistics, the weights and the repeat factors; write "
                "nothing, not even the output directory."
            ),
        ),
    ] = False,
) -> None:
    """Prepare the data and the config of a student trained on KD text mixed with the originals.

    The original corpora and the KD corpora of the teacher are mixed by an exact integer
    ratio; the validation corpus is cut with the student's own SentencePiece model and
    config.yaml is the student's base config with them, ready for `rinchen train launch`.
    Train nothing."""
    with rinchen_errors_as_exit_codes():
        # The mix and the numbers are checked before anything is read or hashed.
        options = _checked_options(
            _Options(
                parse_mix(mix),
                not no_kd_tag,
                tag_placement,
                bt_ratio,
                temperature,
                seed,
                dev_max_lines,
                weight_total,
            )
        )
        reject_repeated_corpora(corpus)
        _reject_repeated_kd_files(kd)
        paths = _Paths(config, corpus, kd, spm_model, Path(os.path.abspath(out_dir)))
        reject_dollar("output directory", out_dir)
        reject_glob_characters("output directory", paths.out_dir)
        spm = Path(os.path.abspath(spm_model))
        reject_dollar("SentencePiece model path", spm)
        reject_glob_characters("SentencePiece model path", spm)
        hashed = _hash_inputs(paths)
        require_out_apart_from_inputs(out_dir, hashed.sources, option="--out-dir")
        refuse_foreign_manifest(paths.out_dir, _STAGE)
        output = _Output(paths.out_dir)
        try:
            prepared = _prepare(paths, hashed, output, options, dry_run=dry_run)
            lines = _statistic_lines(prepared, paths, options)
            if dry_run:
                lines.append("nothing was written")
            else:
                lines += _commit(prepared, paths.out_dir, hashed, options)
        except OSError as exc:
            if not output.begun:
                raise DataError(
                    f"cannot read an input of the student run: {exc}"
                ) from exc
            exit_cannot_write(paths.out_dir, exc, what=_OUTPUTS)
        for line in lines:
            typer.echo(line)


def _checked_options(options: _Options) -> _Options:
    if not math.isfinite(options.temperature) or options.temperature <= 0:
        raise ConfigError(
            f"--temperature must be a positive, finite number, got {options.temperature}"
        )
    if not math.isfinite(options.bt_ratio) or options.bt_ratio < 0:
        raise ConfigError(
            f"--bt-ratio must be a finite number of at least 0, got {options.bt_ratio}"
        )
    return options


def _reject_repeated_kd_files(paths: Sequence[Path]) -> None:
    """A KD file given twice (whatever the spelling or link) would be read twice and double
    every row of it."""
    given: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in given:
            raise ConfigError(
                f"--kd {path} names the same file as --kd {given[resolved]}: it was "
                "given twice and would count every row of it twice"
            )
        given[resolved] = path


def _hash_inputs(paths: _Paths) -> _Hashed:
    """The manifest entries of the base config, the corpora, the KD files with their
    manifests and the SentencePiece model, each taken before anything reads it, in this
    order. A KD manifest that exists must be readable (a DataError); one that is missing is
    the loader's to refuse, with its own message."""
    base = hash_input(paths.config, what=_BASE_CONFIG, missing=ConfigError)
    corpora = [
        hash_input(path, what=_CORPUS, missing=DataError) for path in paths.corpus
    ]
    entries = [base, *corpora]
    digests: dict[Path, str] = {}
    sources = [paths.config, *paths.corpus, *paths.kd, paths.spm_model]
    seen: set[Path] = set()
    for path in paths.kd:
        file_entry = hash_input(path, what=_KD_FILE, missing=DataError)
        digests[path] = file_entry.sha256
        entries.append(file_entry)
        manifest = store.manifest_path(path.parent)
        sources.append(manifest)
        if manifest.resolve() not in seen and manifest.is_file():
            seen.add(manifest.resolve())
            entries.append(hash_input(manifest, what=_KD_MANIFEST, missing=DataError))
            store.read(path.parent)
    entries.append(hash_input(paths.spm_model, what=_SPM_MODEL, missing=ConfigError))
    return _Hashed(entries, base.sha256, digests, sources)


def _prepare(
    paths: _Paths,
    hashed: _Hashed,
    output: _Output,
    options: _Options,
    *,
    dry_run: bool,
) -> PreparedStudent:
    """Read everything, plan and, unless `dry_run`, write the corpora, the validation files
    and the config. The output directory is made, and the old manifest removed, by the
    library's hook after every refusal and before its first write."""
    try:
        return prepare_student(
            corpora=paths.corpus,
            kd_files=paths.kd,
            spm_model=paths.spm_model,
            base=paths.config,
            out_dir=paths.out_dir,
            mix=options.mixture,
            kd_tag=options.kd_tag,
            placement=options.tag_placement,
            bt_ratio=options.bt_ratio,
            temperature=options.temperature,
            seed=options.seed,
            dev_max_lines=options.dev_max_lines,
            weight_total=options.weight_total,
            dry_run=dry_run,
            kd_digests=hashed.kd_digests,
            before_writing=output.begin,
        )
    except ValueError as exc:
        # The numbers are checked above; what is left is a combination the weights cannot
        # take, such as a temperature so large that the sizes raised to it overflow.
        raise ConfigError(f"cannot plan the student run: {exc}") from exc


def _statistic_lines(
    prepared: PreparedStudent, paths: _Paths, options: _Options
) -> list[str]:
    """What the run counted and decided, one fact per line."""
    training, dev, kd = prepared.training, prepared.dev, prepared.kd_stats
    return [
        f"corpus files: {len(paths.corpus)}",
        f"kd files: {len(paths.kd)}",
        f"rows read: {training.rows_read}",
        f"rows kept: {training.rows_kept}",
        *(f"excluded {reason}: {n}" for reason, n in training.excluded.items()),
        *(
            f"excluded by origin {origin}: {n}"
            for origin, n in training.excluded_by_origin.items()
        ),
        f"duplicate ids: {training.duplicate_ids}",
        *(f"dev lines {name}: {n}" for name, n in dev.lines_by_direction.items()),
        f"dev in train: {dev.dev_in_train}",
        f"kd rows read: {kd.rows_read}",
        f"kd rows kept: {kd.rows_kept}",
        *(f"kd excluded {reason}: {n}" for reason, n in kd.excluded.items()),
        f"kd duplicate ids: {kd.duplicate_ids}",
        *(f"kd rows {name}: {n}" for name, n in kd.rows_by_direction.items()),
        f"mix: {options.mixture.kd}:{options.mixture.original}",
        f"kd tag: {'ft' if options.kd_tag else 'none'}",
        *(
            f"corpus {size.name}: rows {size.rows}, weight {prepared.weights[size.name]}"
            for size in prepared.sizes
        ),
        *(
            f"repeat {name}: {float(factor):.3f}"
            for name, factor in prepared.repeat.items()
        ),
    ]


def _commit(
    prepared: PreparedStudent, target: Path, hashed: _Hashed, options: _Options
) -> list[str]:
    """Write the manifest, last, and return the lines that say what was written."""
    if prepared.config is None:
        raise ValueError("a run that wrote nothing has no manifest to write")
    valid = validation_paths(target)
    manifest_path = write_manifest(
        target,
        stage=_STAGE,
        config_sha256=hashed.base_sha256,
        seed=options.seed,
        inputs=hashed.entries,
        outputs=[
            output_file(prepared.config, None),
            output_file(valid.src, prepared.valid_lines),
            output_file(valid.tgt, prepared.valid_lines),
            *(
                file
                for corpus in prepared.corpora
                for file in corpus_manifest_files(corpus)
            ),
        ],
        metrics=_metrics(prepared, options),
        packages=_TOOL_PACKAGES,
        status={
            "prepare": "ok",
            "tag_placement": options.tag_placement,
            "mix": f"{options.mixture.kd}:{options.mixture.original}",
        },
    )
    reused = sum(corpus.reused for corpus in prepared.corpora)
    return [
        f"corpora: {len(prepared.corpora)} (reused {reused})",
        f"validation lines: {prepared.valid_lines}",
        f"config: {prepared.config}",
        f"manifest: {manifest_path}",
    ]


def _metrics(prepared: PreparedStudent, options: _Options) -> dict[str, float]:
    """The statistics of the original rows (`count.*`, `count.dev.*`) and of the KD rows
    (`kd.count.*`), the mix, the weight of each corpus, how often a KD row is drawn against
    an original one in each direction, and the options."""
    direction_of = {size.name: size.direction for size in prepared.sizes}
    return {
        "mix.kd": float(options.mixture.kd),
        "mix.original": float(options.mixture.original),
        "kd_tag": 1.0 if options.kd_tag else 0.0,
        "bt_ratio": options.bt_ratio,
        "temperature": options.temperature,
        "weight_total": float(options.weight_total),
        "valid.lines": float(prepared.valid_lines),
        **numbers_under("count", prepared.training.to_dict()),
        **numbers_under("count.dev", prepared.dev.to_dict()),
        **numbers_under("kd.count", prepared.kd_stats.to_dict()),
        **{f"weight.{name}": float(w) for name, w in prepared.weights.items()},
        **{
            f"kd.repeat.{direction_of[name]}": float(factor)
            for name, factor in prepared.repeat.items()
        },
    }
