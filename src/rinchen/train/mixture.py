"""The training mixture of an ablation: which corpus rows train, the development
sets, the plain-text corpus files Eole reads, the Eole `data` section and the
pre-tokenised validation corpus.

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

The corpus files of a corpus are `<name>.src` and `<name>.tgt` (UTF-8, one sentence
per line, LF only, rows sorted by id, then source, then target), and `<name>.json`
beside them records the row count and the SHA-256 of both files. A set of files is
reused when that record equals the rows that would be written and the files on disk
have those digests; else it is written again, each file whole or not at all.
Authentic corpora are written to `<data_dir>/auth`; the back-translated subsample to
`<data_dir>/bt/r<ratio>_s<seed>`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, get_args

import polars as pl
import sentencepiece

from rinchen.config.ablation import TagPlacement, id_number
from rinchen.config.languages import load_languages
from rinchen.corpus.readers import scan_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.synth.tags import data_origin_tag
from rinchen.train.eole_cfg import reject_dollar
from rinchen.train.sampling import CorpusSize
from rinchen.utils.part_file import unique_part_path

AUTH_GROUP: Final = "auth"
BT_GROUP: Final = "bt"
_BT_ORIGIN: Final = "bt"
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
_PLACEMENTS: Final = get_args(TagPlacement)
_SIDECAR_SUFFIX: Final = ".json"


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
    `available_by_direction`, of which `lines_by_direction` are returned."""

    rows_read: int
    rows_usable: int
    excluded: dict[str, int]
    ignored_by_direction: dict[str, int]
    available_by_direction: dict[str, int]
    lines_by_direction: dict[str, int]

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
        return {
            direction: asdict(usage)
            for direction, usage in self.bt_by_direction.items()
        }


@dataclass(frozen=True, slots=True)
class ValidPaths:
    src: Path
    tgt: Path


# --- reading the corpus files ------------------------------------------------------


@functools.cache
def _direction_tags() -> dict[str, str]:
    return {entry.code: entry.direction_tag for entry in load_languages()}


def _direction_tag(language: str) -> str:
    try:
        return _direction_tags()[language]
    except KeyError:
        raise ConfigError(
            f"language {language!r} is not in the language registry"
        ) from None


def _blank(column: str) -> pl.Expr:
    return pl.col(column).is_null() | pl.col(column).str.contains(_BLANK_PATTERN)


def _has_line_break(column: str) -> pl.Expr:
    return pl.col(column).str.contains_any(list(LINE_BREAK_CHARACTERS))


def _training_reason() -> pl.Expr:
    return (
        pl.when(pl.col("split").ne_missing("train"))
        .then(pl.lit("not_train"))
        .when(pl.col("filter_reason").is_not_null())
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
        pl.when(pl.col("split").ne_missing("dev"))
        .then(pl.lit("not_dev"))
        .when(pl.col("filter_reason").is_not_null())
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
    origin for the excluded ones) and the rows with no reason."""

    rows_read: int
    reasons: Counter[str]
    excluded_origins: Counter[str]
    usable: pl.DataFrame


def _read_file(
    path: Path, reason: pl.Expr, identity_columns: Sequence[str], what: str
) -> _FileRows:
    """Read a corpus file and sort its rows by `reason`.

    Raises:
        DataError: the file is missing, cannot be read or is not a corpus table; a
            usable row has a null in one of `identity_columns`. A damaged file can make
            the Parquet reader panic, which polars raises as a BaseException
            (docs/VERIFY.md, D-POLARS-PANIC): it is a bad file too.
    """
    try:
        frame = (
            scan_corpus(path)
            .select(
                *_KEPT_COLUMNS[:5],
                "filter_reason",
                pl.col("origin").cast(pl.String),
                pl.col("split").cast(pl.String),
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
    return _FileRows(
        rows_read=frame.height,
        reasons=_counts(frame, _REASON),
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
        pl.when(pl.col("origin") == _BT_ORIGIN)
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
    unknown = sorted(used - set(_direction_tags()))
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
    paths: Sequence[Path], *, max_lines: int, directions: Collection[str]
) -> DevRows:
    """Read the development rows of the corpus files: for each of `directions` (those
    with training rows, `TrainingRows.directions`) the first `max_lines` usable rows
    by id, then source, then target. Usable rows of any other direction are counted and
    ignored.

    Raises:
        ValueError: `max_lines` is below 1.
        DataError: a file is missing, unreadable or not a corpus table; a usable row
            has no id or language; a direction has no usable dev row.
    """
    if max_lines < 1:
        raise ValueError(f"max_lines must be at least 1, got {max_lines}")
    files = [
        _read_file(
            path, _dev_reason(), ("id", "src_lang", "tgt_lang"), "that are usable"
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
    )
    return DevRows(sets=sets, stats=stats)


# --- writing the corpus files ------------------------------------------------------


def build_corpora(
    rows: TrainingRows, *, bt_ratio: float, seed: int, data_dir: Path
) -> BuiltCorpora:
    """Write the corpus files of one variant and describe them.

    Every direction with authentic rows gets an `auth` corpus. It also gets a `bt`
    corpus of `min(available, round(bt_ratio * authentic rows))` back-translated rows,
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
        requested = round(bt_ratio * authentic_rows)
        used = min(available, requested)
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


def _text_bytes(lines: pl.Series) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_corpus(
    frame: pl.DataFrame, directory: Path, direction: str, group: str
) -> CorpusFiles:
    name = f"{direction}.{group}"
    ordered = frame.sort("id", "src", "tgt")
    src_bytes, tgt_bytes = _text_bytes(ordered["src"]), _text_bytes(ordered["tgt"])
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
        _replace_file(src_path, src_bytes)
        _replace_file(tgt_path, tgt_bytes)
        # Last: a crash before it leaves files that the next run does not trust.
        _replace_file(sidecar, json.dumps(record, sort_keys=True).encode("utf-8"))
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
    except OSError, ValueError:
        return False
    return (
        recorded == record
        and _file_digest(src_path) == record["src_sha256"]
        and _file_digest(tgt_path) == record["tgt_sha256"]
    )


def _file_digest(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def _replace_file(path: Path, content: bytes) -> None:
    """Write `content` so that `path` appears whole: it is written to a temporary file
    beside it and renamed over it, and a failure leaves any earlier file."""
    part_path = unique_part_path(path)
    try:
        part_path.write_bytes(content)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


# --- the Eole data section ---------------------------------------------------------


def _require_placement(placement: object) -> None:
    if placement not in _PLACEMENTS:
        raise ValueError(
            f"placement must be one of {', '.join(_PLACEMENTS)}, got {placement!r}"
        )


def _bt_tag() -> str:
    tag = data_origin_tag(_BT_ORIGIN)
    if tag is None:
        raise ConfigError(f"the origin {_BT_ORIGIN!r} has no data tag")
    return tag


def _prefixes(corpus: CorpusFiles, placement: str) -> tuple[str, str]:
    """(`src_prefix`, `tgt_prefix`) of a corpus: the tag of its target language and,
    for a back-translated corpus, the bt tag, on the side the placement names."""
    tag = _direction_tag(corpus.tgt_lang)
    back_translated = corpus.group == BT_GROUP
    if placement == "src":
        return (f"{tag} {_bt_tag()}" if back_translated else tag), ""
    return (_bt_tag() if back_translated else ""), tag


def _weight(name: str, weights: Mapping[str, int]) -> int:
    if name not in weights:
        raise ValueError(f"no weight for corpus {name!r}")
    weight = weights[name]
    if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
        raise ValueError(
            f"the weight of corpus {name!r} must be a positive integer, got {weight!r}"
        )
    return weight


def data_section(
    corpora: Sequence[CorpusFiles],
    weights: Mapping[str, int],
    placement: TagPlacement,
    *,
    valid_paths: ValidPaths,
    filtertoolong: bool,
) -> dict[str, Any]:
    """The Eole `data` mapping: one entry per corpus with its paths, `transforms`
    (`sentencepiece`, `prefix` and, when `filtertoolong`, `filtertoolong`), `weight`
    and the two prefixes that carry the direction tag, and `valid` with its two paths.

    The tag is that of the corpus's target language; with placement `src` it is the
    source prefix and a bt corpus adds `<bt>` after it, with `tgt` it is the target
    prefix and a bt corpus has `<bt>` alone as source prefix. Paths are written as
    given, as POSIX text. The text files carry no tags: the placement is configuration.

    Raises:
        ValueError: a placement other than `src` or `tgt`; no corpus or two corpora of
            one name; a weight that is missing, not an integer or below 1; a weight
            for a name that is no corpus.
        ConfigError: a path holds `$`, which Eole would expand as a variable.
    """
    _require_placement(placement)
    if not corpora:
        raise ValueError("at least one corpus is needed")
    names = [corpus.name for corpus in corpora]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"corpus names must be unique, repeated: {repeated}")
    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(f"weights name corpora that are not in the mixture: {unknown}")
    transforms = [
        "sentencepiece",
        "prefix",
        *(["filtertoolong"] if filtertoolong else []),
    ]
    section: dict[str, Any] = {}
    for corpus in corpora:
        reject_dollar(f"source path of corpus {corpus.name}", corpus.src_path)
        reject_dollar(f"target path of corpus {corpus.name}", corpus.tgt_path)
        src_prefix, tgt_prefix = _prefixes(corpus, placement)
        section[corpus.name] = {
            "path_src": corpus.src_path.as_posix(),
            "path_tgt": corpus.tgt_path.as_posix(),
            "transforms": list(transforms),
            "weight": _weight(corpus.name, weights),
            "src_prefix": src_prefix,
            "tgt_prefix": tgt_prefix,
        }
    reject_dollar("source path of the validation corpus", valid_paths.src)
    reject_dollar("target path of the validation corpus", valid_paths.tgt)
    section["valid"] = {
        "path_src": valid_paths.src.as_posix(),
        "path_tgt": valid_paths.tgt.as_posix(),
    }
    return section


# --- the validation corpus ---------------------------------------------------------


def dev_line_ranges(dev: Mapping[str, DevSet]) -> dict[str, range]:
    """The lines each direction occupies in the validation corpus `write_validation`
    writes (and in any file that concatenates the dev sets in the same order):
    directions in sorted order, each a run of one line per dev row."""
    ranges: dict[str, range] = {}
    start = 0
    for direction in sorted(dev):
        stop = start + len(dev[direction].src)
        ranges[direction] = range(start, stop)
        start = stop
    return ranges


def _load_processor(spm_model: Path) -> sentencepiece.SentencePieceProcessor:
    # SentencePiece aborts the interpreter on a directory, so only files pass.
    if not spm_model.is_file():
        raise ConfigError(f"SentencePiece model is not a file: {spm_model}")
    try:
        return sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"SentencePiece model {spm_model} cannot be loaded: {exc}"
        ) from exc


def _require_one_piece(
    processor: sentencepiece.SentencePieceProcessor, tag: str, spm_model: Path
) -> None:
    piece_id = processor.piece_to_id(tag)
    if piece_id == processor.unk_id() or processor.id_to_piece(piece_id) != tag:
        raise ConfigError(
            f"the direction tag {tag} is not one piece of the SentencePiece model "
            f"{spm_model}"
        )


def write_validation(
    dev: Mapping[str, DevSet],
    spm_model: Path,
    placement: TagPlacement,
    out_dir: Path,
) -> ValidPaths:
    """Write `valid.src` and `valid.tgt` to `out_dir`: the dev rows of all directions,
    sorted by direction, one line each, pre-tokenised with the SentencePiece model
    (pieces joined by one U+0020) and with the tag of the target language, one piece,
    first on the source line for placement `src` and first on the target line for `tgt`.

    Eole then reads the lines as they are (the corpus has no transforms and no
    prefix), so nothing of the validation leaks into the checkpoint.

    Raises:
        ValueError: a placement other than `src` or `tgt`; no dev set.
        ConfigError: the model cannot be loaded, or a tag is not one of its pieces.
    """
    _require_placement(placement)
    if not dev:
        raise ValueError("at least one dev set is needed")
    processor = _load_processor(spm_model)
    src_lines: list[str] = []
    tgt_lines: list[str] = []
    for direction in sorted(dev):
        dev_set = dev[direction]
        tag = _direction_tag(dev_set.tgt_lang)
        _require_one_piece(processor, tag, spm_model)
        sources = processor.encode(list(dev_set.src), out_type=str)
        targets = processor.encode(list(dev_set.tgt), out_type=str)
        for source, target in zip(sources, targets, strict=True):
            if placement == "src":
                source = [tag, *source]
            else:
                target = [tag, *target]
            src_lines.append(" ".join(source))
            tgt_lines.append(" ".join(target))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ValidPaths(src=out_dir / "valid.src", tgt=out_dir / "valid.tgt")
    _replace_file(paths.src, ("\n".join(src_lines) + "\n").encode("utf-8"))
    _replace_file(paths.tgt, ("\n".join(tgt_lines) + "\n").encode("utf-8"))
    return paths
