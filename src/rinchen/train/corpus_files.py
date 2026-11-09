"""The plain-text corpus files Eole reads: one `<name>.src` and `<name>.tgt` per
direction and group, the back-translated subsample, and the reuse of files already
written.

The corpus files of a corpus are `<name>.src` and `<name>.tgt` (UTF-8, one sentence
per line, LF only, rows sorted by id, then source, then target), and `<name>.json`
beside them records the row count and the SHA-256 of both files. A set of files is
reused when that record equals the rows that would be written and the files on disk
have those digests; else it is written again, each file whole or not at all.
Authentic corpora are written to `<data_dir>/auth`; the back-translated subsample to
`<data_dir>/bt/r<ratio>_s<seed>`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

from rinchen.config.ablation import id_number
from rinchen.manifest.stage import sha256_of
from rinchen.train.mixture import AUTH_GROUP, BT_GROUP, TrainingRows
from rinchen.train.sampling import CorpusSize
from rinchen.utils.part_file import lines_as_bytes, replace_file

_SIDECAR_SUFFIX: Final = ".json"


@dataclass(frozen=True, slots=True)
class BtUsage:
    """The back-translated rows of a direction: those there are, those the ratio asks
    for (0 when the direction has no authentic rows) and those used."""

    bt_available: int
    bt_requested: int
    bt_used: int


@dataclass(frozen=True, slots=True)
class CorpusFiles:
    """One corpus on disk: its two text files, row count and file digests, and whether
    the files were already there (`reused`)."""

    name: str
    direction: str
    group: str
    src_lang: str
    tgt_lang: str
    src_path: Path
    tgt_path: Path
    rows: int
    src_sha256: str
    tgt_sha256: str
    reused: bool


@dataclass(frozen=True, slots=True)
class PlannedFiles:
    """One corpus a build would write: where its two files will be and how many rows
    they will hold (the fields `data_section` needs of a corpus, before it exists)."""

    name: str
    direction: str
    group: str
    tgt_lang: str
    src_path: Path
    tgt_path: Path
    rows: int


@dataclass(frozen=True, slots=True)
class BuiltCorpora:
    """The corpora of one variant, sorted by direction and then group, and the bt
    usage of every direction that has bt or authentic rows."""

    corpora: tuple[CorpusFiles, ...]
    bt_by_direction: dict[str, BtUsage]


def build_corpora(
    rows: TrainingRows, *, bt_ratio: float, seed: int, data_dir: Path
) -> BuiltCorpora:
    """Write the corpus files of one variant and describe them.

    Every direction with authentic rows gets an `auth` corpus. It also gets a `bt`
    corpus of the back-translated rows `bt_usage_by_direction` says to use, when that
    is more than 0: those with the smallest `sha256(f"{seed}\\t{id}")` (hex digest; ties
    by id, source, target), so the rows do not depend on the order of the input rows
    or files or on the polars version. A direction with bt rows and no authentic rows
    gets no corpus; `bt_by_direction` records what was available, asked for and used.
    The corpora written are those `planned_corpus_sizes` lists, with those row counts, at
    the paths `planned_corpus_files` lists.

    Raises:
        ValueError: `bt_ratio` is not a finite number of at least 0.
    """
    usage = bt_usage_by_direction(rows, bt_ratio)
    parts = rows.frame.partition_by("direction", "group", as_dict=True)
    auth_directory = _corpus_directory(data_dir, AUTH_GROUP, bt_ratio, seed)
    bt_directory = _corpus_directory(data_dir, BT_GROUP, bt_ratio, seed)
    corpora: list[CorpusFiles] = []
    for direction, direction_usage in usage.items():
        authentic = parts.get((direction, AUTH_GROUP))
        back_translated = parts.get((direction, BT_GROUP))
        if authentic is not None:
            corpora.append(
                write_corpus_files(authentic, auth_directory, direction, AUTH_GROUP)
            )
        if back_translated is not None and direction_usage.bt_used:
            sample = _smallest_hashes(back_translated, direction_usage.bt_used, seed)
            corpora.append(
                write_corpus_files(sample, bt_directory, direction, BT_GROUP)
            )
    return BuiltCorpora(corpora=tuple(corpora), bt_by_direction=usage)


def bt_usage_by_direction(rows: TrainingRows, bt_ratio: float) -> dict[str, BtUsage]:
    """The back-translated rows of every direction that has rows, by direction in
    sorted order, as a build with `bt_ratio` would use them: `min(available,
    round(bt_ratio * authentic rows))`, where `round` is Python's (a half goes to the
    even integer: 2.5 gives 2, 3.5 gives 4) and a ratio so large that the product
    overflows a float asks for more than there are. A direction without authentic rows
    asks for none. Nothing is written.

    Raises:
        ValueError: `bt_ratio` is not a finite number of at least 0.
    """
    if not math.isfinite(bt_ratio) or bt_ratio < 0:
        raise ValueError(
            f"bt_ratio must be a finite number of at least 0, got {bt_ratio}"
        )
    counts = _rows_by_direction_and_group(rows)
    usage: dict[str, BtUsage] = {}
    for direction in sorted({key[0] for key in counts}):
        authentic_rows = counts.get((direction, AUTH_GROUP), 0)
        available = counts.get((direction, BT_GROUP), 0)
        asked = bt_ratio * authentic_rows
        usage[direction] = BtUsage(
            bt_available=available,
            bt_requested=_rows_asked(bt_ratio, authentic_rows, asked),
            # `min` before `round`: a product that overflowed a float is `inf`, which
            # `round` cannot turn into an integer.
            bt_used=round(min(available, asked)),
        )
    return usage


def planned_corpus_sizes(rows: TrainingRows, bt_ratio: float) -> list[CorpusSize]:
    """The corpora `build_corpora(rows, bt_ratio=bt_ratio, ...)` writes, with their
    row counts, in the order it writes them (by direction, `auth` before `bt`): one
    `auth` corpus per direction with authentic rows and a `bt` corpus where the usage
    is above 0. The one place the sizes of a variant's corpora are decided, so a plan
    cannot disagree with the build. Nothing is written.

    Raises:
        ValueError: `bt_ratio` is not a finite number of at least 0.
    """
    return [
        CorpusSize(
            corpus_name(planned.direction, planned.group),
            planned.direction,
            planned.rows,
        )
        for planned in _planned_groups(rows, bt_ratio)
    ]


def planned_corpus_files(
    rows: TrainingRows, bt_ratio: float, seed: int, data_dir: Path
) -> list[PlannedFiles]:
    """The corpora `build_corpora(rows, bt_ratio=bt_ratio, seed=seed,
    data_dir=data_dir)` writes, in its order, with the paths of their files and their
    row counts (those of `planned_corpus_sizes`). The paths come from the functions the
    build writes through, so a config made from them names the files the build makes.
    Nothing is written.

    Raises:
        ValueError: `bt_ratio` is not a finite number of at least 0.
    """
    planned = _planned_groups(rows, bt_ratio)
    languages = dict(
        rows.frame.group_by("direction").agg(pl.col("tgt_lang").first()).iter_rows()
    )
    files: list[PlannedFiles] = []
    for direction, group, row_count in planned:
        name = corpus_name(direction, group)
        directory = _corpus_directory(data_dir, group, bt_ratio, seed)
        src_path, tgt_path, _ = _corpus_paths(directory, name)
        files.append(
            PlannedFiles(
                name=name,
                direction=direction,
                group=group,
                tgt_lang=languages[direction],
                src_path=src_path,
                tgt_path=tgt_path,
                rows=row_count,
            )
        )
    return files


class _PlannedGroup(NamedTuple):
    direction: str
    group: str
    rows: int


def _planned_groups(rows: TrainingRows, bt_ratio: float) -> list[_PlannedGroup]:
    """The one place that decides which corpora a build with `bt_ratio` writes and how
    many rows each has: by direction, `auth` before `bt`."""
    counts = _rows_by_direction_and_group(rows)
    planned: list[_PlannedGroup] = []
    for direction, usage in bt_usage_by_direction(rows, bt_ratio).items():
        authentic_rows = counts.get((direction, AUTH_GROUP), 0)
        if authentic_rows:
            planned.append(_PlannedGroup(direction, AUTH_GROUP, authentic_rows))
        if usage.bt_used:
            planned.append(_PlannedGroup(direction, BT_GROUP, usage.bt_used))
    return planned


def corpus_name(direction: str, group: str) -> str:
    """The name of the corpus of `group` rows in `direction`: `<direction>.<group>`."""
    return f"{direction}.{group}"


def _corpus_directory(data_dir: Path, group: str, bt_ratio: float, seed: int) -> Path:
    """Where the files of `group` corpora go: the authentic ones are shared by every
    variant, the back-translated subsample is per ratio and seed."""
    if group == AUTH_GROUP:
        return data_dir / AUTH_GROUP
    return data_dir / BT_GROUP / f"r{id_number(bt_ratio)}_s{seed}"


def _corpus_paths(directory: Path, name: str) -> tuple[Path, Path, Path]:
    """The source file, the target file and the sidecar record of the corpus `name`."""
    return (
        directory / f"{name}.src",
        directory / f"{name}.tgt",
        directory / f"{name}{_SIDECAR_SUFFIX}",
    )


def corpus_sidecar_path(corpus: CorpusFiles) -> Path:
    """The record `corpus` was written with (`<name>.json` beside its two text files):
    the row count and the SHA-256 of both files."""
    return corpus.src_path.parent / f"{corpus.name}{_SIDECAR_SUFFIX}"


def _rows_by_direction_and_group(rows: TrainingRows) -> dict[tuple[str, str], int]:
    counts = rows.frame.group_by("direction", "group").len().iter_rows()
    return {(direction, group): count for direction, group, count in counts}


def _rows_asked(bt_ratio: float, authentic_rows: int, product: float) -> int:
    """The rows `bt_ratio` asks for, `round(product)`; when the product overflowed
    a float, the same rounding of the exact product."""
    if math.isfinite(product):
        return round(product)
    return round(Fraction(bt_ratio) * authentic_rows)


def _smallest_hashes(frame: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    digests = [
        hashlib.sha256(f"{seed}\t{row_id}".encode()).hexdigest()
        for row_id in frame["id"]
    ]
    return (
        frame.with_columns(pl.Series("rank", digests, dtype=pl.String))
        .sort("rank", "id", "src", "tgt")
        .head(count)
    )


def write_corpus_files(
    frame: pl.DataFrame, directory: Path, direction: str, group: str
) -> CorpusFiles:
    """Write the corpus of `group` rows in `direction` into `directory`: the rows of
    `frame` (columns `id`, `src`, `tgt`, `src_lang`, `tgt_lang`) sorted by id, source and
    target as `<name>.src` and `<name>.tgt`, then the sidecar `<name>.json`; files that
    already are what would be written are left as they are (`CorpusFiles.reused`)."""
    name = corpus_name(direction, group)
    ordered = frame.sort("id", "src", "tgt")
    src_bytes = lines_as_bytes(ordered["src"])
    tgt_bytes = lines_as_bytes(ordered["tgt"])
    src_sha256 = hashlib.sha256(src_bytes).hexdigest()
    tgt_sha256 = hashlib.sha256(tgt_bytes).hexdigest()
    record = {
        "rows": ordered.height,
        "src_sha256": src_sha256,
        "tgt_sha256": tgt_sha256,
    }
    src_path, tgt_path, sidecar = _corpus_paths(directory, name)
    reused = _is_current(record, src_path, tgt_path, sidecar)
    if not reused:
        directory.mkdir(parents=True, exist_ok=True)
        replace_file(src_path, src_bytes)
        replace_file(tgt_path, tgt_bytes)
        # Last: a crash before it leaves files that the next run does not trust.
        replace_file(sidecar, json.dumps(record, sort_keys=True).encode("utf-8"))
    return CorpusFiles(
        name=name,
        direction=direction,
        group=group,
        src_lang=ordered["src_lang"][0],
        tgt_lang=ordered["tgt_lang"][0],
        src_path=src_path,
        tgt_path=tgt_path,
        rows=ordered.height,
        src_sha256=src_sha256,
        tgt_sha256=tgt_sha256,
        reused=reused,
    )


def _is_current(
    record: Mapping[str, object], src_path: Path, tgt_path: Path, sidecar: Path
) -> bool:
    """Whether the sidecar holds exactly `record` and both files have its digests."""
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            recorded == record
            and sha256_of(src_path) == record["src_sha256"]
            and sha256_of(tgt_path) == record["tgt_sha256"]
        )
    except OSError, ValueError:
        return False
