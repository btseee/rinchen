"""Which corpus rows train, which are development rows, and what was counted.

Corpus rows (the corpus Parquet files) become TRAINING rows by these rules, applied
in this order, the first that matches deciding (so every row of every file is counted
once: `rows_read = rows_kept + the sum of the exclusions`):

1. `not_train`: `split` is not `train`.
2. `filtered`: `filter_reason` is set.
3. `excluded_origin`: the origin is `ft`, `synth_llm` or `synth_nmt` (forward
   translation enters training with D8 and D10); also counted by origin.
4. `empty`: `src` or `tgt` is null, empty or only whitespace (Python's `str.strip`).
5. `line_break`: `src` or `tgt` holds a character `str.splitlines` splits at, which
   would turn one sentence into two lines of a text file.

The origin sorts a kept row into a group: `bt` for back-translated rows, `auth` for
the rest (human, web, mined, converted). A corpus is a direction (`src_lang-tgt_lang`)
and a group, named `<direction>.<group>`.

Development rows (`split` is `dev`, no `filter_reason`) go through the `empty` and
`line_break` rules and are counted apart. Rows of a language outside the registry
are an error only among the rows that would train.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.config.languages import known_language_codes
from rinchen.corpus.readers import scan_corpus
from rinchen.errors import ConfigError, DataError

AUTH_GROUP: Final = "auth"
BT_GROUP: Final = "bt"
BT_ORIGIN: Final = "bt"
EXCLUDED_ORIGINS: Final = ("ft", "synth_llm", "synth_nmt")

# The reasons a training row is left out, in the order they are tried.
EXCLUSION_REASONS: Final = (
    "not_train",
    "filtered",
    "excluded_origin",
    "empty",
    "line_break",
)
_DEV_EXCLUSION_REASONS: Final = ("not_dev", "filtered", "empty", "line_break")

# The characters `str.splitlines` splits at: LF, CR, VT, FF, FS, GS, RS, NEL, LS, PS.
LINE_BREAK_CHARACTERS: Final = tuple(
    chr(code)
    for code in (0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
)

# Python's whitespace (`str.strip`): Unicode White_Space, which `\s` matches, and the
# four separator controls U+001C to U+001F, which Python adds.
_BLANK_PATTERN: Final = r"\A[\s\x1c-\x1f]*\z"

# The columns the kept rows keep.
_KEPT_COLUMNS: Final = ("id", "src_lang", "tgt_lang", "src", "tgt", "origin")
_REASON: Final = "reason"


@dataclass(frozen=True, slots=True)
class MixtureStats:
    """What `load_training_rows` counted; `to_dict` is JSON-ready.

    `excluded` has every reason of `EXCLUSION_REASONS` and `excluded_by_origin` every
    origin of `EXCLUDED_ORIGINS`, with 0 where nothing was left out; the origin counts
    add up to `excluded["excluded_origin"]`. `duplicate_ids` is the number of kept rows
    whose id an earlier kept row has: they are kept. `rows_by_corpus` maps the name of
    each corpus with kept rows to its row count. `source_collision_rate` is
    `source_collision_rate` over all kept rows, before any bt subsample."""

    rows_read: int
    rows_kept: int
    excluded: dict[str, int]
    excluded_by_origin: dict[str, int]
    duplicate_ids: int
    rows_by_corpus: dict[str, int]
    source_collision_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class TrainingRows:
    """The kept rows (columns `id`, `src_lang`, `tgt_lang`, `src`, `tgt`, `origin`,
    `direction`, `group`) and their statistics; `directions` are those that have
    authentic rows, sorted: the ones that train."""

    frame: pl.DataFrame
    stats: MixtureStats
    directions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevStats:
    """What `load_dev_rows` counted; `to_dict` is JSON-ready.

    `rows_read` is `rows_usable` plus the `excluded` counts (`not_dev`, `filtered`,
    `empty`, `line_break`). The usable rows of a direction without training rows are
    counted in `ignored_by_direction`; those of a training direction in
    `available_by_direction`, of which `lines_by_direction` are returned.

    `dev_in_train` is the number of returned dev lines whose id is also the id of a
    kept training row (the id is derived from the languages and both texts): such a
    line inflates the dev score, as the model has seen it. It is None when no
    training rows were given to compare with."""

    rows_read: int
    rows_usable: int
    excluded: dict[str, int]
    ignored_by_direction: dict[str, int]
    available_by_direction: dict[str, int]
    lines_by_direction: dict[str, int]
    dev_in_train: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DevSet:
    """The development rows of one direction, in order of id, source, target."""

    direction: str
    src_lang: str
    tgt_lang: str
    ids: tuple[str, ...]
    src: tuple[str, ...]
    tgt: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevRows:
    sets: dict[str, DevSet]
    stats: DevStats


# --- reading the corpus files ------------------------------------------------------


def _blank(column: str) -> pl.Expr:
    return pl.col(column).is_null() | pl.col(column).str.contains(_BLANK_PATTERN)


def _has_line_break(column: str) -> pl.Expr:
    return pl.col(column).str.contains_any(list(LINE_BREAK_CHARACTERS))


def _training_reason() -> pl.Expr:
    return (
        pl.when(pl.col("filter_reason").is_not_null())
        .then(pl.lit("filtered"))
        .when(pl.col("origin").is_in(EXCLUDED_ORIGINS))
        .then(pl.lit("excluded_origin"))
        .when(_blank("src") | _blank("tgt"))
        .then(pl.lit("empty"))
        .when(_has_line_break("src") | _has_line_break("tgt"))
        .then(pl.lit("line_break"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(_REASON)
    )


def _dev_reason() -> pl.Expr:
    return (
        pl.when(pl.col("filter_reason").is_not_null())
        .then(pl.lit("filtered"))
        .when(_blank("src") | _blank("tgt"))
        .then(pl.lit("empty"))
        .when(_has_line_break("src") | _has_line_break("tgt"))
        .then(pl.lit("line_break"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(_REASON)
    )


@dataclass(frozen=True, slots=True, eq=False)
class _FileRows:
    """One corpus file sorted by `reason`: how many rows, how many per reason (and per
    origin for the excluded ones) and the rows with no reason. The rows of another
    split than the one asked for are counted under `not_<split>` and never loaded."""

    rows_read: int
    reasons: Counter[str]
    excluded_origins: Counter[str]
    usable: pl.DataFrame


def _read_file(
    path: Path,
    split: str,
    reason: pl.Expr,
    identity_columns: Sequence[str],
    what: str,
) -> _FileRows:
    """Read the rows of `split` of a corpus file and sort them by `reason`.

    The split filter is part of the lazy plan, so the text of the other rows is not
    decoded: a file with none of `split` loads next to nothing.

    Raises:
        DataError: the file is missing, cannot be read or is not a corpus table; a
            usable row has a null in one of `identity_columns`. A damaged file can make
            the Parquet reader panic, which polars raises as a BaseException
            (docs/VERIFY.md, D-POLARS-PANIC): it is a bad file too.
    """
    try:
        lazy = scan_corpus(path)
        rows_read = lazy.select(pl.len()).collect().item()
        frame = (
            lazy.filter(pl.col("split") == split)
            .select(
                *_KEPT_COLUMNS[:5],
                "filter_reason",
                pl.col("origin").cast(pl.String),
            )
            .with_columns(reason)
            .collect()
        )
    except DataError:
        raise
    except (pl.exceptions.PolarsError, pl.exceptions.PanicException, OSError) as exc:
        raise DataError(f"cannot read corpus file {path}: {exc}") from exc
    usable = frame.filter(pl.col(_REASON).is_null()).select(_KEPT_COLUMNS)
    for column in identity_columns:
        missing = usable[column].null_count()
        if missing:
            raise DataError(f"{path}: {missing} rows {what} have a null {column}")
    reasons = _counts(frame, _REASON)
    if rows_read > frame.height:
        reasons[f"not_{split}"] = rows_read - frame.height
    return _FileRows(
        rows_read=rows_read,
        reasons=reasons,
        excluded_origins=_counts(
            frame.filter(pl.col(_REASON) == "excluded_origin"), "origin"
        ),
        usable=usable,
    )


def _counts(frame: pl.DataFrame, column: str) -> Counter[str]:
    return Counter(
        {
            value: count
            for value, count in frame.group_by(column).len().iter_rows()
            if value is not None
        }
    )


def _total(counters: Iterable[Counter[str]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return total


def _with_direction(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("src_lang") + "-" + pl.col("tgt_lang")).alias("direction")
    )


def source_collision_rate(frame: pl.DataFrame) -> float:
    """The share of the rows of `frame` (columns `src` and `tgt_lang`) whose source
    text occurs in a row of another target language, 0.0 for no rows. The text is
    compared as it is; a source that occurs twice into one language does not count."""
    if frame.is_empty():
        return 0.0
    colliding = frame.select(
        (pl.col("tgt_lang").n_unique().over("src") > 1).sum()
    ).item()
    return int(colliding) / frame.height


def load_training_rows(paths: Sequence[Path]) -> TrainingRows:
    """Read the corpus files and keep the rows that train (see the module docstring).

    Raises:
        DataError: a file is missing, unreadable or not a corpus table; a row that
            would be kept has no id, language or origin; no row is kept; no kept row
            is authentic.
        ConfigError: a kept row has a language that is not in the language registry.
    """
    files = [
        _read_file(
            path,
            "train",
            _training_reason(),
            ("id", "src_lang", "tgt_lang", "origin"),
            "that would train",
        )
        for path in paths
    ]
    rows_read = sum(file.rows_read for file in files)
    excluded = _total(file.reasons for file in files)
    if not any(file.usable.height for file in files):
        raise DataError(
            f"no training rows: {rows_read} rows read, 0 kept "
            f"(left out by reason: {dict(excluded)})"
        )
    kept = _with_direction(pl.concat(file.usable for file in files))
    _require_known_languages(kept)
    kept = kept.with_columns(
        pl.when(pl.col("origin") == BT_ORIGIN)
        .then(pl.lit(BT_GROUP))
        .otherwise(pl.lit(AUTH_GROUP))
        .alias("group")
    )
    authentic = kept.filter(pl.col("group") == AUTH_GROUP)
    if authentic.is_empty():
        raise DataError(
            f"no authentic training rows: the {kept.height} kept rows are all "
            "back-translated (bt), and a bt corpus is added to an authentic one"
        )
    excluded_origins = _total(file.excluded_origins for file in files)
    stats = MixtureStats(
        rows_read=rows_read,
        rows_kept=kept.height,
        excluded={reason: excluded[reason] for reason in EXCLUSION_REASONS},
        excluded_by_origin={
            origin: excluded_origins[origin] for origin in EXCLUDED_ORIGINS
        },
        duplicate_ids=kept.height - kept["id"].n_unique(),
        rows_by_corpus=_rows_by_corpus(kept),
        source_collision_rate=source_collision_rate(kept),
    )
    return TrainingRows(
        frame=kept,
        stats=stats,
        directions=tuple(sorted(authentic["direction"].unique())),
    )


def _require_known_languages(kept: pl.DataFrame) -> None:
    used = set(kept["src_lang"].unique()) | set(kept["tgt_lang"].unique())
    unknown = sorted(used - known_language_codes())
    if unknown:
        raise ConfigError(
            f"training rows use languages that are not in the language registry: "
            f"{unknown}"
        )


def _rows_by_corpus(kept: pl.DataFrame) -> dict[str, int]:
    counts = kept.group_by("direction", "group").len().iter_rows()
    return dict(
        sorted((f"{direction}.{group}", count) for direction, group, count in counts)
    )


def load_dev_rows(
    paths: Sequence[Path],
    *,
    max_lines: int,
    directions: Collection[str],
    training: TrainingRows | None = None,
) -> DevRows:
    """Read the development rows of the corpus files: for each of `directions` (those
    with training rows, `TrainingRows.directions`) the first `max_lines` usable rows
    by id, then source, then target. Usable rows of any other direction are counted and
    ignored. Only the dev rows are loaded; the text of the other rows is not decoded.

    With `training`, `DevStats.dev_in_train` counts the returned lines whose id is that
    of a kept training row; without it the count is None.

    Raises:
        ValueError: `max_lines` is below 1.
        DataError: a file is missing, unreadable or not a corpus table; a usable row
            has no id or language; a direction has no usable dev row.
    """
    if max_lines < 1:
        raise ValueError(f"max_lines must be at least 1, got {max_lines}")
    files = [
        _read_file(
            path,
            "dev",
            _dev_reason(),
            ("id", "src_lang", "tgt_lang"),
            "that are usable",
        )
        for path in paths
    ]
    if not files:
        raise DataError("no usable dev rows: no corpus file was given")
    rows_read = sum(file.rows_read for file in files)
    excluded = _total(file.reasons for file in files)
    usable = _with_direction(pl.concat(file.usable for file in files))
    available = _counts(usable, "direction")

    sets: dict[str, DevSet] = {}
    for direction in sorted(directions):
        chosen = (
            usable.filter(pl.col("direction") == direction)
            .sort("id", "src", "tgt")
            .head(max_lines)
        )
        if chosen.is_empty():
            raise DataError(
                f"no usable dev rows for direction {direction}, which has training "
                f"rows (dev rows read: {rows_read}, left out by reason: "
                f"{dict(excluded)})"
            )
        sets[direction] = DevSet(
            direction=direction,
            src_lang=chosen["src_lang"][0],
            tgt_lang=chosen["tgt_lang"][0],
            ids=tuple(chosen["id"]),
            src=tuple(chosen["src"]),
            tgt=tuple(chosen["tgt"]),
        )
    stats = DevStats(
        rows_read=rows_read,
        rows_usable=usable.height,
        excluded={reason: excluded[reason] for reason in _DEV_EXCLUSION_REASONS},
        ignored_by_direction={
            direction: count
            for direction, count in sorted(available.items())
            if direction not in sets
        },
        available_by_direction={direction: available[direction] for direction in sets},
        lines_by_direction={direction: len(sets[direction].ids) for direction in sets},
        dev_in_train=None if training is None else _lines_in_training(sets, training),
    )
    return DevRows(sets=sets, stats=stats)


def _lines_in_training(sets: Mapping[str, DevSet], training: TrainingRows) -> int:
    dev_ids = pl.DataFrame(
        {"id": [dev_id for dev_set in sets.values() for dev_id in dev_set.ids]},
        schema={"id": pl.String},
    )
    training_ids = training.frame.select("id").unique()
    return dev_ids.join(training_ids, on="id", how="semi").height
