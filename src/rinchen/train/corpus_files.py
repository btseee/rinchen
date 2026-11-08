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
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

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
class BuiltCorpora:
    """The corpora of one variant, sorted by direction and then group, and the bt
    usage of every direction that has bt or authentic rows."""

    corpora: tuple[CorpusFiles, ...]
    bt_by_direction: dict[str, BtUsage]

    def bt_to_dict(self) -> dict[str, dict[str, int]]:
        """The bt usage by direction as a JSON-ready mapping, for a results record."""
        return {
            direction: asdict(usage)
            for direction, usage in self.bt_by_direction.items()
        }


def build_corpora(
    rows: TrainingRows, *, bt_ratio: float, seed: int, data_dir: Path
) -> BuiltCorpora:
    """Write the corpus files of one variant and describe them.

    Every direction with authentic rows gets an `auth` corpus. It also gets a `bt`
    corpus of `min(available, round(bt_ratio * authentic rows))` back-translated rows
    (a ratio so large that the product overflows a float asks for more than there are),
    when that is more than 0: those with the smallest `sha256(f"{seed}\\t{id}")` (hex
    digest; ties by id, source, target), so the rows do not depend on the order of the
    input rows or files or on the polars version. `round` is Python's: a half goes to
    the even integer (2.5 gives 2, 3.5 gives 4). A direction with bt rows and no
    authentic rows gets no corpus; `bt_by_direction` records what was available,
    asked for and used.

    Raises:
        ValueError: `bt_ratio` is not a finite number of at least 0.
    """
    if not math.isfinite(bt_ratio) or bt_ratio < 0:
        raise ValueError(
            f"bt_ratio must be a finite number of at least 0, got {bt_ratio}"
        )
    parts = rows.frame.partition_by("direction", "group", as_dict=True)
    bt_directory = data_dir / "bt" / f"r{id_number(bt_ratio)}_s{seed}"
    corpora: list[CorpusFiles] = []
    usage: dict[str, BtUsage] = {}
    for direction in sorted({key[0] for key in parts}):
        authentic = parts.get((direction, AUTH_GROUP))
        back_translated = parts.get((direction, BT_GROUP))
        authentic_rows = 0 if authentic is None else authentic.height
        available = 0 if back_translated is None else back_translated.height
        asked = bt_ratio * authentic_rows
        # `min` before `round`: a product that overflowed a float is `inf`, which
        # `round` cannot turn into an integer.
        used = round(min(available, asked))
        requested = _rows_asked(bt_ratio, authentic_rows, asked)
        if authentic is not None:
            corpora.append(
                _write_corpus(authentic, data_dir / AUTH_GROUP, direction, AUTH_GROUP)
            )
        if back_translated is not None and used:
            sample = _smallest_hashes(back_translated, used, seed)
            corpora.append(_write_corpus(sample, bt_directory, direction, BT_GROUP))
        usage[direction] = BtUsage(
            bt_available=available, bt_requested=requested, bt_used=used
        )
    return BuiltCorpora(corpora=tuple(corpora), bt_by_direction=usage)


def _rows_asked(bt_ratio: float, authentic_rows: int, product: float) -> int:
    """The rows `bt_ratio` asks for, `round(product)`; when the product overflowed
    a float, the same rounding of the exact product."""
    if math.isfinite(product):
        return round(product)
    return round(Fraction(bt_ratio) * authentic_rows)


def corpus_sizes(corpora: Iterable[CorpusFiles]) -> list[CorpusSize]:
    """The corpora as the sampler takes them."""
    return [
        CorpusSize(corpus.name, corpus.direction, corpus.rows) for corpus in corpora
    ]


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


def _write_corpus(
    frame: pl.DataFrame, directory: Path, direction: str, group: str
) -> CorpusFiles:
    name = f"{direction}.{group}"
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
    src_path = directory / f"{name}.src"
    tgt_path = directory / f"{name}.tgt"
    sidecar = directory / f"{name}{_SIDECAR_SUFFIX}"
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
