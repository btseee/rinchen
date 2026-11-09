"""Preparing a student run (D10): the data and the config a student trains on, before Eole
runs. Nothing is trained here.

A student trains from random initialisation on the original corpora (D7: authentic rows and
a subsample of the back-translated ones) mixed with the KD corpora a teacher made
(`distill_corpus`), by an exact integer ratio (`distill_mix`). `prepare_student` reads the
rows, plans the corpora and their weights (one planner, so a dry run and a real run agree)
and, unless it is a dry run, writes under `out_dir`:

    data/auth/..., data/bt/r<ratio>_s<seed>/...   the original corpora (D7's files)
    data/kd/<direction>.kd.src, .tgt, .json        the KD corpora, one per direction
    valid.src, valid.tgt                           the dev rows cut with the STUDENT's
                                                   SentencePiece model
    config.yaml                                    the student config; its run files are
                                                   out_dir/run/vocab.src and out_dir/run/ckpt

so that `rinchen train launch --config <out>/config.yaml --run-dir <out>/run` trains the
student, resumable as any D8 run. The corpus files are text and do not depend on the
student's SentencePiece model; the validation corpus (pre-tokenised) and `src_vocab_size`
(the model's pieces plus Eole's pad token) do. A KD corpus carries the origin tag `<ft>`
(`eole_data`) unless asked not to: without it Eole drops, silently, a KD row whose source
repeats an original source in the same minibatch (5.2 % of the examples on the toy).

The config is the base (the student's shape and recipe: `configs/eole/student_*.yaml`) with
the per-run values laid over it. A base holds no corpora and never `train_from` or
`reset_optim`: a student starts from random initialisation, always.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.errors import ConfigError
from rinchen.train.corpus_files import (
    CorpusFiles,
    build_corpora,
    corpus_name,
    planned_corpus_sizes,
    write_corpus_files,
)
from rinchen.train.distill_corpus import KdRows, KdStats, load_kd_rows
from rinchen.train.distill_mix import Mix, mixed_weights, repeat_factors
from rinchen.train.eole_cfg import (
    merged_config,
    read_config,
    reject_dollar,
    reject_glob_characters,
    render,
)
from rinchen.train.eole_data import data_section, ft_tag
from rinchen.train.mixture import (
    KD_GROUP,
    DevRows,
    DevStats,
    MixtureStats,
    TrainingRows,
    load_dev_rows,
    load_training_rows,
)
from rinchen.train.sampling import CorpusSize, integer_weights, mixing_probabilities
from rinchen.train.validation_corpus import (
    load_processor,
    require_direction_tags,
    require_one_piece,
    write_validation,
)
from rinchen.train.vocabulary_size import (
    VOCABULARY_SIZE_KEYS,
    needed_vocabulary_size,
    require_vocabulary_size,
)

_RUN_DIR: Final = "run"
_BASE: Final = "base Eole config"
# Keys that make a model start from a checkpoint: never in the config of a student.
_START_FROM_CHECKPOINT: Final = ("train_from", "reset_optim")


@dataclass(frozen=True, slots=True)
class PreparedStudent:
    """What `prepare_student` decided and, unless it was a dry run, wrote.

    `sizes` are the corpora of the data section in its order (the original ones by direction,
    `auth` before `bt`, then the KD ones by direction) with their row counts, `weights` the
    integer weight of each and `repeat` for each KD corpus how many times more often one of
    its rows is drawn than one original row of its direction. `corpora` are the corpora
    written (empty for a dry run), `config` the config file (None for a dry run) and
    `run_dir` the directory the run files will be in. `valid_lines` is the number of dev
    lines of the validation corpus."""

    config: Path | None
    run_dir: Path
    sizes: tuple[CorpusSize, ...]
    corpora: tuple[CorpusFiles, ...]
    weights: dict[str, int]
    repeat: dict[str, Fraction]
    kd_stats: KdStats
    training: MixtureStats
    dev: DevStats
    valid_lines: int
    written: bool


@dataclass(frozen=True, slots=True)
class _Plan:
    sizes: tuple[CorpusSize, ...]
    weights: dict[str, int]
    repeat: dict[str, Fraction]


def prepare_student(
    *,
    corpora: Sequence[Path],
    kd_files: Sequence[Path],
    spm_model: Path,
    base: Path,
    out_dir: Path,
    mix: Mix,
    kd_tag: bool,
    placement: TagPlacement,
    bt_ratio: float,
    temperature: float,
    seed: int,
    dev_max_lines: int,
    weight_total: int,
    dry_run: bool,
) -> PreparedStudent:
    """Read the original corpus files and the KD files, plan the corpora and their weights
    and, unless `dry_run`, write the files of the module's text. Everything that can be
    refused is refused before the first write; a dry run writes nothing, not even `out_dir`.

    `corpora` are the original corpus files (their `train` rows are the authentic and
    back-translated rows, their `dev` rows the validation), `kd_files` the KD files, each
    with its manifest. `bt_ratio`, `temperature` and `weight_total` are D7's, applied to the
    original corpora; `mix` is applied on top of them. `spm_model` is the student's own
    SentencePiece model. Paths in the config are absolute.

    Raises:
        ValueError: a placement other than `src` or `tgt`, a `bt_ratio`, `temperature` or
            `dev_max_lines` out of range.
        ConfigError: the base config is not one a student may start from or is not valid;
            `out_dir` or `spm_model` holds a character Eole reads specially; the model cannot
            be loaded or lacks a tag as one piece; a stated vocabulary size is too small;
            `weight_total` is below the number of corpora; a KD file is not vouched for by
            its manifest.
        DataError: a corpus or KD file is missing or unreadable, or holds no row to train
            on.
        OSError: a file cannot be written.
    """
    require_placement(placement)
    target = Path(os.path.abspath(out_dir))
    spm = Path(os.path.abspath(spm_model))
    reject_dollar("output directory", target)
    reject_glob_characters("output directory", target)
    reject_dollar("SentencePiece model path", spm)
    base_config = check_student_base(base)

    training = load_training_rows(corpora)
    dev = load_dev_rows(
        corpora,
        max_lines=dev_max_lines,
        directions=training.directions,
        training=training,
    )
    kd = load_kd_rows(kd_files, directions=training.directions)
    _require_tags_and_vocabulary(spm, dev, base_config, kd_tag=kd_tag)
    plan = _plan(
        training,
        kd,
        mix=mix,
        bt_ratio=bt_ratio,
        temperature=temperature,
        weight_total=weight_total,
    )
    valid_lines = sum(dev.stats.lines_by_direction.values())
    if dry_run:
        return PreparedStudent(
            config=None,
            run_dir=target / _RUN_DIR,
            sizes=plan.sizes,
            corpora=(),
            weights=plan.weights,
            repeat=plan.repeat,
            kd_stats=kd.stats,
            training=training.stats,
            dev=dev.stats,
            valid_lines=valid_lines,
            written=False,
        )
    run_dir = target / _RUN_DIR
    built = build_corpora(
        training, bt_ratio=bt_ratio, seed=seed, data_dir=target / "data"
    )
    kd_written = [
        write_corpus_files(part, target / "data" / KD_GROUP, direction, KD_GROUP)
        for direction, part in _by_direction(kd.frame)
    ]
    written = (*built.corpora, *kd_written)
    valid = write_validation(dev.sets, spm, placement, target)
    data = data_section(
        written,
        plan.weights,
        placement,
        valid_paths=valid,
        filtertoolong=_configures_filtertoolong(base_config),
        kd_tag=kd_tag,
    )
    config = render_student_config(
        base,
        data=data,
        spm_model=spm,
        src_vocab=run_dir / "vocab.src",
        model_path=run_dir / "ckpt",
        seed=seed,
        out=target / "config.yaml",
    )
    return PreparedStudent(
        config=config,
        run_dir=run_dir,
        sizes=plan.sizes,
        corpora=written,
        weights=plan.weights,
        repeat=plan.repeat,
        kd_stats=kd.stats,
        training=training.stats,
        dev=dev.stats,
        valid_lines=valid_lines,
        written=True,
    )


def check_student_base(base: Path) -> dict[str, Any]:
    """The base config of a student, checked before anything is written: it reads, has no
    `train_from` and no `reset_optim` (a student starts from random initialisation), holds
    no corpora of its own (the data is `prepare_student`'s) and is valid as it stands.
    Returns it.

    Raises:
        ConfigError: the base is missing, unreadable or not valid (what
            `eole_cfg.merged_config` raises), or holds one of the above.
    """
    raw = read_config(base, _BASE)
    _require_random_initialisation(raw, base)
    _require_no_corpora(raw, base)
    return merged_config(base, {})


def render_student_config(
    base: Path,
    *,
    data: Mapping[str, Any],
    spm_model: Path,
    src_vocab: Path,
    model_path: Path,
    seed: int,
    out: Path,
) -> Path:
    """Write to `out` the base config of a student with the per-run values laid over it,
    validated, and return `out`: `data` (the section of `eole_data.data_section`),
    `src_vocab` (the vocabulary file Eole wants), `transforms_configs.sentencepiece` (the
    student's model, both sides), `training.model_path`, `seed` and `src_vocab_size`: the one
    the base states when it holds every piece of the model and Eole's pad token, the model's
    pieces plus one when it states none. The model and the recipe of the base stay as they
    are; `train_from` and `reset_optim` are never added.

    Raises:
        ConfigError: what `check_student_base` raises; the model cannot be loaded; the base
            states a `src_vocab_size` or `tgt_vocab_size` below what the model needs (the
            message names both numbers); the finished config is not valid (what
            `eole_cfg.render` raises: a path that holds `$`, `out` is the base or a
            directory, ...).
    """
    base_config = check_student_base(base)
    spm = spm_model.as_posix()
    overrides: dict[str, Any] = {
        "seed": seed,
        "src_vocab": src_vocab.as_posix(),
        "data": dict(data),
        "transforms_configs": {
            "sentencepiece": {"src_subword_model": spm, "tgt_subword_model": spm}
        },
        "training": {"model_path": model_path.as_posix()},
        **_vocabulary_size(base_config, spm_model),
    }
    return render(base, overrides, out)


def _plan(
    training: TrainingRows,
    kd: KdRows,
    *,
    mix: Mix,
    bt_ratio: float,
    temperature: float,
    weight_total: int,
) -> _Plan:
    """The corpora of the student's data with their sizes and weights: the one place that
    decides them, for a dry run and for a real run."""
    originals = planned_corpus_sizes(training, bt_ratio)
    original_weights = integer_weights(
        mixing_probabilities(originals, temperature), total=weight_total
    )
    kd_sizes = [
        CorpusSize(corpus_name(direction, KD_GROUP), direction, rows)
        for direction, rows in sorted(kd.stats.rows_by_direction.items())
    ]
    direction_of = {size.name: size.direction for size in originals}
    kd_corpora = {size.name: size.direction for size in kd_sizes}
    weights = mixed_weights(original_weights, direction_of, kd_corpora, mix)
    rows = {size.name: size.rows for size in (*originals, *kd_sizes)}
    return _Plan(
        sizes=(*originals, *kd_sizes),
        weights=weights,
        repeat=repeat_factors(rows, weights, direction_of, kd_corpora),
    )


def _by_direction(frame: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """The KD rows of each direction, sorted by direction."""
    parts = frame.partition_by("direction", as_dict=True)
    return [(direction, part) for (direction,), part in sorted(parts.items())]


def _require_tags_and_vocabulary(
    spm_model: Path, dev: DevRows, base_config: Mapping[str, Any], *, kd_tag: bool
) -> None:
    """Refuse, before anything is written, a SentencePiece model that lacks a tag the data
    carries as one piece, and a stated vocabulary size that cannot hold its pieces."""
    require_direction_tags(spm_model, dev.sets)
    if kd_tag:
        require_one_piece(
            load_processor(spm_model), ft_tag(), spm_model, what="origin tag"
        )
    _vocabulary_size(base_config, spm_model)


def _vocabulary_size(base_config: Mapping[str, Any], spm_model: Path) -> dict[str, int]:
    """`src_vocab_size` for the finished config when the base leaves it out, else nothing;
    either size the base states is checked against the model's pieces plus Eole's pad
    token."""
    needed = needed_vocabulary_size(spm_model)
    for key in VOCABULARY_SIZE_KEYS:
        stated = base_config.get(key)
        if stated is not None:
            require_vocabulary_size(key, stated, needed, spm_model)
    return (
        {}
        if base_config.get("src_vocab_size") is not None
        else {"src_vocab_size": needed}
    )


def _configures_filtertoolong(base_config: Mapping[str, Any]) -> bool:
    sections = base_config.get("transforms_configs")
    return isinstance(sections, Mapping) and "filtertoolong" in sections


def _require_random_initialisation(raw: object, base: Path) -> None:
    sections = raw.get("training") if isinstance(raw, Mapping) else None
    for section, prefix in ((raw, ""), (sections, "training.")):
        if not isinstance(section, Mapping):
            continue
        for key in _START_FROM_CHECKPOINT:
            if key in section:
                raise ConfigError(
                    f"invalid {_BASE} {base}: {prefix}{key}: a student starts from "
                    "random initialisation, never from a checkpoint; remove it"
                )


def _require_no_corpora(raw: object, base: Path) -> None:
    data = raw.get("data") if isinstance(raw, Mapping) else None
    if isinstance(data, Mapping) and data:
        raise ConfigError(
            f"invalid {_BASE} {base}: data: holds the corpora {sorted(data)}, and a "
            "student trains on the data `rinchen distill prepare` makes: leave `data` "
            "empty in the base"
        )
