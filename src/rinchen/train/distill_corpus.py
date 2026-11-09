"""Reading the KD files a generation wrote (D10): the rows that a student trains on next to
the original corpora.

A KD file is a corpus Parquet file made by `rinchen distill generate`, with the
`manifest.json` of stage `distill_generate` beside it. Nothing else vouches for a KD file,
so `load_kd_rows` refuses one without that manifest, with a manifest of another stage or of
a generation that did not finish, with a manifest that does not list it, or whose SHA-256 or
row count is not the file's now (a `ConfigError` naming the file). The file is hashed once,
before anything of it is read: by `load_kd_rows`, or by the caller, who passes the digests it
took (`digests`) so that a command that vouches for its inputs does not hash a large file
twice.

Every row of every file is counted once, in the kept rows or under the first of these
reasons that matches (`KD_EXCLUSION_REASONS`, in this order):

1. `not_train`: `split` is not `train`. The text of such a row is never decoded.
2. `filtered`: `filter_reason` is set.
3. `wrong_origin`: the origin is not `ft`.
4. `empty`: `src` or `tgt` is null, empty or only whitespace (Python's `str.strip`).
5. `line_break`: `src` or `tgt` holds a character `str.splitlines` splits at, which would
   turn one sentence into two lines of a text file.
6. `unknown_direction`: the direction (`src_lang-tgt_lang`) is not one of the directions
   given, the ones the original corpora train: a KD corpus is never trained alone.

So `rows_read = rows_kept + the sum of the exclusions`. A row whose id repeats is kept and
counted (`duplicate_ids`): two teachers, or one teacher twice, may produce the same pair.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.corpus.readers import scan_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import sha256_of
from rinchen.train.mixture import blank_text, has_line_break

GENERATE_STAGE: Final = "distill_generate"
KD_ORIGIN: Final = "ft"

# The reasons a KD row is left out, in the order they are tried.
KD_EXCLUSION_REASONS: Final = (
    "not_train",
    "filtered",
    "wrong_origin",
    "empty",
    "line_break",
    "unknown_direction",
)

_REASON: Final = "reason"
_GENERATE_STATUS: Final = "generate"
_COLUMNS: Final = ("id", "src_lang", "tgt_lang", "src", "tgt", "direction")


@dataclass(frozen=True, slots=True)
class KdStats:
    """What `load_kd_rows` counted; `to_dict` is JSON-ready.

    `excluded` has every reason of `KD_EXCLUSION_REASONS` with 0 where nothing was left
    out, and `rows_read` is `rows_kept` plus their sum. `rows_by_direction` and
    `rows_by_label` count the kept rows by direction and by corpus label (`ft:<name>`)."""

    rows_read: int
    rows_kept: int
    excluded: dict[str, int]
    duplicate_ids: int
    rows_by_direction: dict[str, int]
    rows_by_label: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class KdRows:
    """The kept rows (columns `id`, `src_lang`, `tgt_lang`, `src`, `tgt`, `direction`,
    sorted by id, then source, then target) and what was counted."""

    frame: pl.DataFrame
    stats: KdStats


@dataclass(frozen=True, slots=True, eq=False)
class _FileRows:
    """One KD file sorted by reason: how many rows, how many per reason, and the rows that
    have none, with their corpus label."""

    rows_read: int
    reasons: Counter[str]
    usable: pl.DataFrame


def load_kd_rows(
    paths: Sequence[Path],
    *,
    directions: Collection[str],
    digests: Mapping[Path, str] | None = None,
) -> KdRows:
    """Read the KD files, each vouched for by its manifest, and keep the rows that train
    (see the module's text); `directions` are those of the original corpora.

    `digests` maps each path to the SHA-256 the caller took of the file before anything
    read it; it is compared with the manifest's in place of a hash of the file here. The
    caller's digest is trusted to be that of the file it hashed: nothing reads the bytes it
    hashed again.

    Raises:
        ValueError: `digests` is given and lacks one of the paths.
        ConfigError: a file has no manifest, or one that does not vouch for it (for the
            digest it has, or the one `digests` gives).
        DataError: no file is given; a file is missing, unreadable or not a corpus table; a
            kept row has no id or no corpus label; no row is kept.
    """
    if not paths:
        raise DataError("no KD file was given")
    if digests is not None:
        absent = [str(path) for path in paths if path not in digests]
        if absent:
            raise ValueError(f"no digest was given for the KD files {absent}")
    known = sorted(set(directions))
    files = [
        _read_file(path, known, None if digests is None else digests[path])
        for path in paths
    ]
    rows_read = sum(file.rows_read for file in files)
    reasons: Counter[str] = Counter()
    for file in files:
        reasons.update(file.reasons)
    kept = pl.concat(file.usable for file in files)
    if kept.is_empty():
        left_out = {
            name: reasons[name] for name in KD_EXCLUSION_REASONS if reasons[name]
        }
        raise DataError(
            f"no KD rows: {rows_read} rows read, 0 kept (left out by reason: {left_out})"
        )
    kept = kept.with_columns(
        (pl.col("src_lang") + "-" + pl.col("tgt_lang")).alias("direction")
    )
    stats = KdStats(
        rows_read=rows_read,
        rows_kept=kept.height,
        excluded={name: reasons[name] for name in KD_EXCLUSION_REASONS},
        duplicate_ids=kept.height - kept["id"].n_unique(),
        rows_by_direction=_counts(kept, "direction"),
        rows_by_label=_counts(kept, "corpus"),
    )
    frame = kept.select(_COLUMNS).sort("id", "src", "tgt")
    return KdRows(frame=frame, stats=stats)


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return dict(sorted(frame.group_by(column).len().iter_rows()))


def _read_file(
    path: Path, known_directions: Sequence[str], digest: str | None
) -> _FileRows:
    recorded_rows = _digest_the_vouched_file(path, digest)
    try:
        lazy = scan_corpus(path)
        rows_read = lazy.select(pl.len()).collect().item()
        if rows_read != recorded_rows:
            raise ConfigError(
                f"KD file {path}: its manifest records {recorded_rows} rows but the "
                f"file has {rows_read}"
            )
        frame = (
            lazy.filter(pl.col("split") == "train")
            .select(
                "id",
                "src_lang",
                "tgt_lang",
                "src",
                "tgt",
                "corpus",
                "filter_reason",
                pl.col("origin").cast(pl.String),
            )
            .with_columns(_reason(known_directions))
            .collect()
        )
    except ConfigError, DataError:
        raise
    except (pl.exceptions.PolarsError, pl.exceptions.PanicException, OSError) as exc:
        # A damaged file can make the Parquet reader panic, which polars raises as a
        # BaseException (docs/VERIFY.md, D-POLARS-PANIC): it is a bad file too.
        raise DataError(f"cannot read KD file {path}: {exc}") from exc
    usable = frame.filter(pl.col(_REASON).is_null()).select(
        "id", "src_lang", "tgt_lang", "src", "tgt", "corpus"
    )
    for column in ("id", "corpus"):
        missing = usable[column].null_count()
        if missing:
            raise DataError(
                f"{path}: {missing} KD rows that would train have a null {column}"
            )
    reasons = Counter(
        {
            reason: count
            for reason, count in frame.group_by(_REASON).len().iter_rows()
            if reason is not None
        }
    )
    reasons["not_train"] += rows_read - frame.height
    return _FileRows(rows_read=rows_read, reasons=reasons, usable=usable)


def _reason(known_directions: Sequence[str]) -> pl.Expr:
    direction = pl.col("src_lang") + "-" + pl.col("tgt_lang")
    return (
        pl.when(pl.col("filter_reason").is_not_null())
        .then(pl.lit("filtered"))
        .when(pl.col("origin").ne_missing(KD_ORIGIN))
        .then(pl.lit("wrong_origin"))
        .when(blank_text("src") | blank_text("tgt"))
        .then(pl.lit("empty"))
        .when(has_line_break("src") | has_line_break("tgt"))
        .then(pl.lit("line_break"))
        .when(~direction.is_in(list(known_directions)).fill_null(False))
        .then(pl.lit("unknown_direction"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(_REASON)
    )


def _digest_the_vouched_file(path: Path, given: str | None) -> int:
    """Check the manifest beside `path` and the file against it, hashing the file once and
    before anything of it is parsed (or comparing the digest `given` by the caller, which
    hashed it); return the row count the manifest records for it.

    Raises:
        DataError: `path` is not a file or cannot be read.
        ConfigError: the manifest is missing, unreadable, of another stage, of a generation
            that did not finish, or does not list the file with the digest it has now.
    """
    if not path.is_file():
        raise DataError(f"KD file not found: {path}")
    entry = _manifest_entry(path)
    try:
        digest = sha256_of(path) if given is None else given
    except OSError as exc:
        raise DataError(f"cannot read KD file {path}: {exc}") from exc
    if digest != entry.sha256:
        raise ConfigError(
            f"KD file {path}: its SHA-256 is {digest} but the manifest vouches for "
            f"{entry.sha256}; the file changed after the generation wrote it"
        )
    if entry.rows is None:
        raise ConfigError(
            f"KD file {path}: its manifest lists it without a row count, which a "
            "generation always records"
        )
    return entry.rows


def _manifest_entry(path: Path) -> ManifestFile:
    try:
        manifest = store.read(path.parent)
    except DataError as exc:
        raise ConfigError(
            f"KD file {path}: its manifest cannot be read: {exc}"
        ) from exc
    if manifest is None:
        raise ConfigError(
            f"KD file {path} has no manifest.json beside it: only `rinchen distill "
            "generate` writes one, and nothing else vouches for a KD file"
        )
    if manifest.stage != GENERATE_STAGE:
        raise ConfigError(
            f"KD file {path}: the manifest beside it is of stage {manifest.stage!r}, "
            f"not {GENERATE_STAGE!r}"
        )
    status = manifest.status.get(_GENERATE_STATUS)
    if status != "ok":
        raise ConfigError(
            f"KD file {path}: the manifest beside it records `{_GENERATE_STATUS}: "
            f"{status}`, not `ok`: the generation did not finish"
        )
    listed = [item for item in manifest.outputs if Path(item.path).name == path.name]
    if not listed:
        raise ConfigError(
            f"KD file {path}: the manifest beside it does not list {path.name}"
        )
    if len(listed) > 1:
        raise ConfigError(
            f"KD file {path}: the manifest beside it lists {path.name} twice"
        )
    return listed[0]
