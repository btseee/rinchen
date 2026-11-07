"""Work division for a synthetic-data job: the monolingual input, the
deterministic shards it is cut into, the fingerprint that names a job's shard
directory and the crash-safe store of one file of generated candidates per
shard, so a rerun after a crash resumes instead of generating everything
again."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from rinchen.config.synth import GeneratorConfig, JobConfig
from rinchen.errors import DataError
from rinchen.synth._candidates import is_utf8_encodable
from rinchen.utils.part_file import unique_part_path

_MONOLINGUAL_COLUMNS = ("id", "lang", "text")
_SHARD_SCHEMA = pl.Schema({"id": pl.String(), "candidates": pl.List(pl.String())})
_LANGUAGES_SHOWN = 5


@dataclass(frozen=True, slots=True)
class Shard:
    """A contiguous slice of the id-sorted input: the ids and texts of one unit
    of generation, which the store keeps or loses as a whole."""

    index: int
    ids: tuple[str, ...]
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Loaded:
    """A shard whose file is valid: the candidate lists of its ids, in order."""

    candidates: list[list[str]]


@dataclass(frozen=True, slots=True)
class Missing:
    """A shard without a file."""


@dataclass(frozen=True, slots=True)
class Invalid:
    """A shard whose file exists but fails `ShardStore.read`."""


ShardLoad = Loaded | Missing | Invalid


def _read_parquet(path: Path, what: str) -> pl.DataFrame:
    """The file as a frame, or a DataError when it is missing or is not
    readable Parquet. A truncated or damaged file can make the Parquet reader
    panic, which polars raises as a BaseException: it is a bad file too. The
    path is a file name, never a glob: `[`, `*` and `?` in a directory or file
    name must not match some other file."""
    if not path.is_file():
        raise DataError(f"{what} not found: {path}")
    try:
        return pl.read_parquet(path, glob=False)
    except (pl.exceptions.PolarsError, pl.exceptions.PanicException, OSError) as exc:
        raise DataError(f"cannot read {what} {path}: {exc}") from exc


def _require_dtypes(
    frame: pl.DataFrame, expected: Mapping[str, pl.DataType], path: Path
) -> None:
    for name, dtype in expected.items():
        if frame.schema[name] != dtype:
            raise DataError(
                f"{path}: column {name!r} must be {dtype}, found {frame.schema[name]}"
            )


def _sorted_by_id(frame: pl.DataFrame) -> pl.DataFrame:
    # Ids are unique by contract; the stable sort keeps equal ids in input order
    # rather than leaving them to the sort's whim.
    return frame.select("id", "text").sort("id", maintain_order=True)


def read_monolingual(path: Path, lang: str | None = None) -> pl.DataFrame:
    """The monolingual Parquet file at `path`, sorted by `id` (string order).

    It needs the string columns `id`, `lang` and `text`; any other column is
    kept. With `lang` only the rows of that language are kept, and only those
    rows are checked and returned: the other rows never reach a generator. The
    text is not normalised or stripped.

    Raises DataError for an unreadable file, a missing or non-string required
    column, no rows (also none of the requested language), a null or empty id,
    a null, empty or whitespace-only text and a duplicated id. The error names
    the file, the count and the first offending id, never a text."""
    frame = _read_parquet(path, "monolingual file")

    missing = [name for name in _MONOLINGUAL_COLUMNS if name not in frame.columns]
    if missing:
        raise DataError(f"{path}: missing required columns: {missing}")
    _require_dtypes(frame, dict.fromkeys(_MONOLINGUAL_COLUMNS, pl.String()), path)
    if frame.is_empty():
        raise DataError(f"{path}: no rows")

    if lang is not None:
        selected = frame.filter(pl.col("lang") == lang)
        if selected.is_empty():
            present = sorted(frame["lang"].drop_nulls().unique())[:_LANGUAGES_SHOWN]
            raise DataError(
                f"{path}: no rows with lang {lang!r} (languages in the file, "
                f"up to {_LANGUAGES_SHOWN}: {present})"
            )
        frame = selected

    frame = frame.sort("id", maintain_order=True)
    _require_usable_rows(frame, path)
    return frame


def _require_usable_rows(frame: pl.DataFrame, path: Path) -> None:
    ids = frame["id"]
    unusable_ids = ids.fill_null("") == ""
    if unusable_ids.any():
        raise DataError(f"{path}: rows with a null or empty id: {unusable_ids.sum()}")

    blank = [
        position
        for position, text in enumerate(frame["text"])
        # Python's notion of whitespace, as the code that later strips a text uses.
        if text is None or not text.strip()
    ]
    if blank:
        raise DataError(
            f"{path}: rows with a null, empty or whitespace-only text: {len(blank)} "
            f"(first id {ids[blank[0]]!r})"
        )

    repeated = frame.filter(ids.is_duplicated())["id"].unique(maintain_order=True)
    if not repeated.is_empty():
        raise DataError(
            f"{path}: duplicated ids: {repeated.len()} (first {repeated[0]!r})"
        )


def plan_shards(frame: pl.DataFrame, shard_size: int) -> tuple[Shard, ...]:
    """Contiguous slices of `shard_size` rows of the frame sorted by id (the
    last may be shorter), numbered from 0. The frame may arrive in any order:
    the same rows and size always give the same shards."""
    if shard_size < 1:
        raise ValueError(f"shard_size must be at least 1, got {shard_size}")
    ordered = _sorted_by_id(frame)
    ids = ordered["id"].to_list()
    texts = ordered["text"].to_list()
    return tuple(
        Shard(
            index=index,
            ids=tuple(ids[start : start + shard_size]),
            texts=tuple(texts[start : start + shard_size]),
        )
        for index, start in enumerate(range(0, len(ids), shard_size))
    )


def _canonical_json(value: object) -> bytes:
    """`value` as canonical JSON (sorted keys, no spaces, ASCII only). The
    values hashed here are objects and arrays, so one cannot run into the next
    when the bytes are concatenated."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def job_fingerprint(
    frame: pl.DataFrame,
    job: JobConfig,
    generator: GeneratorConfig,
    shard_size: int,
    seed: int,
) -> str:
    """A SHA-256 hex digest over everything that decides what a shard holds: the
    (id, text) rows of the frame in id order (no other column counts), the job
    and generator configs, `shard_size` and `seed`. A changed ingredient names
    another shard directory, so a stale shard is never reused. The rows are
    hashed one at a time: a large input never becomes one string."""
    rows = _sorted_by_id(frame)
    digest = hashlib.sha256()
    header = {
        "job": job.model_dump(mode="json"),
        "generator": generator.model_dump(mode="json"),
        "shard_size": shard_size,
        "seed": seed,
    }
    digest.update(_canonical_json(header))
    for row in rows.iter_rows():
        digest.update(_canonical_json(row))
    return digest.hexdigest()


def shard_directory(out_dir: Path, fingerprint: str) -> Path:
    """Where the shards of the job with this fingerprint live."""
    return out_dir / f"shards-{fingerprint[:12]}"


class ShardStore:
    """One Parquet file per shard (`shard-<index:05d>.parquet`) holding the
    candidates generated for the shard's ids, in `directory`.

    A file appears only whole, through a unique temporary file and a rename, so
    a crash leaves either the earlier file or none; `read` validates a file
    against the planned ids, so a damaged or foreign one is regenerated instead
    of trusted."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, index: int) -> Path:
        if index < 0:
            raise ValueError(f"shard index must not be negative, got {index}")
        return self._directory / f"shard-{index:05d}.parquet"

    def write(
        self,
        index: int,
        ids: Sequence[str],
        candidates: Sequence[Sequence[str]],
    ) -> Path:
        """Store `candidates[i]` for `ids[i]` as shard `index`, replacing any
        earlier file. A text without candidates keeps its empty list. Two
        writers for one index never corrupt each other: each renames its own
        complete file and the last rename wins.

        Raises ValueError for a caller's mistake (ids and candidates of
        different lengths, a string where a list of candidates belongs, a
        candidate that is not a string) and DataError for a candidate UTF-8
        cannot encode (a lone surrogate), before the disk is touched. The
        messages name the shard and the id, never a text."""
        if len(ids) != len(candidates):
            raise ValueError(
                f"ids and candidates differ in length ({len(ids)} vs {len(candidates)})"
            )
        path = self.path_for(index)
        for id_, texts in zip(ids, candidates, strict=True):
            self._require_storable(index, id_, texts)
        frame = pl.DataFrame(
            {"id": list(ids), "candidates": [list(texts) for texts in candidates]},
            schema=_SHARD_SCHEMA,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = unique_part_path(path)
        try:
            frame.write_parquet(part_path, compression="zstd")
            part_path.replace(path)
        finally:
            part_path.unlink(missing_ok=True)
        return path

    @staticmethod
    def _require_storable(index: int, id_: str, texts: Sequence[str]) -> None:
        # ValueError for every bad argument, as `write` documents, although
        # ruff would prefer TypeError for the type checks.
        where = f"shard {index}: candidates for id {id_!r}"
        if isinstance(texts, str):
            raise ValueError(  # noqa: TRY004
                f"{where} must be a sequence of strings, not a string"
            )
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(  # noqa: TRY004
                    f"{where} include a value that is not a string"
                )
            if not is_utf8_encodable(text):
                raise DataError(f"{where} include one UTF-8 cannot encode")

    def read(self, index: int, expected_ids: Sequence[str]) -> list[list[str]]:
        """The candidate lists of shard `index`, one per expected id in order.
        Raises DataError naming the file when it is missing, is not Parquet,
        has other columns or types than the store writes, has other ids than
        `expected_ids` (rows missing, extra or in another order) or holds a
        null where a list or a candidate belongs."""
        path = self.path_for(index)
        frame = _read_parquet(path, "shard file")

        if frame.columns != _SHARD_SCHEMA.names():
            raise DataError(
                f"{path}: expected columns {_SHARD_SCHEMA.names()}, "
                f"found {frame.columns}"
            )
        _require_dtypes(frame, _SHARD_SCHEMA, path)
        if frame.height != len(expected_ids):
            raise DataError(
                f"{path}: row count is {frame.height}, expected {len(expected_ids)}"
            )
        for position, (found, wanted) in enumerate(
            zip(frame["id"], expected_ids, strict=True)
        ):
            if found != wanted:
                raise DataError(
                    f"{path}: id at index {position} is {found!r}, expected {wanted!r}"
                )

        null_lists = frame["candidates"].null_count()
        if null_lists:
            raise DataError(f"{path}: null candidate lists: {null_lists}")
        lists: list[list[str]] = frame["candidates"].to_list()
        null_candidates = sum(text is None for texts in lists for text in texts)
        if null_candidates:
            raise DataError(f"{path}: null candidates: {null_candidates}")
        return lists

    def load(self, index: int, expected_ids: Sequence[str]) -> ShardLoad:
        """Shard `index` as one of `Loaded`, `Missing` (no file) and `Invalid`
        (a file that fails `read` against `expected_ids`), decoding the file at
        most once. A stale `.part` file beside a missing shard does not count.
        This is the single place a shard's file is judged: `completed` and
        `invalid` are built on it."""
        if not self.path_for(index).is_file():
            return Missing()
        try:
            return Loaded(self.read(index, expected_ids))
        except DataError:
            return Invalid()

    def _indexes_loading_as(
        self, shards: Sequence[Shard], outcome: type[Loaded | Invalid]
    ) -> frozenset[int]:
        return frozenset(
            shard.index
            for shard in shards
            if isinstance(self.load(shard.index, shard.ids), outcome)
        )

    def completed(self, shards: Sequence[Shard]) -> frozenset[int]:
        """The indexes of the shards whose file passes `read` against the
        shard's own ids. A missing, corrupt, short or foreign file is not
        completed; an invalid one stays on disk until the next `write` replaces
        it."""
        return self._indexes_loading_as(shards, Loaded)

    def invalid(self, shards: Sequence[Shard]) -> frozenset[int]:
        """The indexes of the shards whose file exists but fails `read`: the
        files a rerun will replace. A shard without a file is not invalid (a
        stale `.part` file beside it does not count as one)."""
        return self._indexes_loading_as(shards, Invalid)
