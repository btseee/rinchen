"""What `rinchen eval suite build` reads from disk (D13, Task 4): which files a source directory
holds for which language, the rows of a Parquet shard, a TSV or a JSON Lines export of a human set,
and the manifest of the fetch that wrote the shards. Polars and `csv` are used here and nowhere in
`rinchen.eval`; the hashing, the order of the steps and the exit codes are the command's
(`rinchen.cli.eval_suite`), the rows' joining and the suite are `rinchen.eval.suite_sources`'s.

A shard is parsed from the bytes the command hashed (`pl.read_parquet(io.BytesIO(data))`). A
message names a language of the registry, a shard, a column of the configuration, a line or a row
and never a cell of a file; an error of polars is named by its class only.

The document of a sentence (the `url` column of FLORES+, as recalled) is turned into an id the
suite accepts: a value that already is one is kept, any other is replaced by `u` and the first 16
hexadecimal characters of its SHA-256, so that the same address is the same document in every
build and no address is written into a suite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

from rinchen.cli.eval_suite_plan import is_revision
from rinchen.config.eval_suites import ColumnsConfig
from rinchen.errors import DataError
from rinchen.eval.suite_rules import checked_language, is_row_id, refusal
from rinchen.eval.suite_sources import HUMAN_COLUMNS, SourceRow
from rinchen.manifest.model import Manifest, ManifestFile

FETCH_STAGE: Final = "eval_suite_fetch"
UNPINNED: Final = "unpinned"
SHARD_PATTERN: Final = "shard-*.parquet"
EXPORT_SUFFIXES: Final = (".jsonl", ".tsv", ".parquet")
_DOCUMENT_DIGITS: Final = 16


# ---- which files are there ----------------------------------------------------------------


def language_inputs(
    root: Path, *, reader: str, configured: Collection[str], benchmark: str
) -> dict[str, list[Path]]:
    """The files of `root` by language: for `parquet_shards` the shards of each language
    directory, for `lines_dir` the file `<code>.txt` of each language, both sorted. Every
    language directory (or text file) must be a language of the benchmark: `khk_Mong` and a code
    outside the registry are refused by name, and any other entry that is not one of the
    `configured` languages is refused too, so that nothing in the directory is left out
    silently.

    Raises:
        DataError: `root` is not a directory, an entry names a language the benchmark does not
            look for, a language directory holds no shard, or nothing is there.
    """
    if not root.is_dir():
        raise DataError(f"the source {root} is not a directory")
    found: dict[str, list[Path]] = {}
    if reader == "parquet_shards":
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                language = _language_of(entry.name, entry, configured, benchmark)
                shards = sorted(entry.glob(SHARD_PATTERN))
                if not shards:
                    raise DataError(f"{entry} holds no shard ({SHARD_PATTERN})")
                found[language] = shards
    else:
        for entry in sorted(root.glob("*.txt")):
            found[_language_of(entry.stem, entry, configured, benchmark)] = [entry]
    if not found:
        raise DataError(
            f"the source {root} holds no language of the benchmark {benchmark}"
        )
    return found


def _language_of(
    name: str, path: Path, configured: Collection[str], benchmark: str
) -> str:
    code = checked_language(name, f"the input {path}", "name")
    if code not in configured:
        raise DataError(
            f"{path}: {code} is not a language the benchmark {benchmark} looks for"
        )
    return code


# ---- the rows of a shard ------------------------------------------------------------------


def parquet_rows(data: bytes, columns: ColumnsConfig, *, where: str) -> list[SourceRow]:
    """The rows of the Parquet shard `data` as the configuration's `columns` name them. `where`
    says which shard it is, for a message.

    Raises:
        DataError: the bytes are not a Parquet file, a column the configuration names is not in
            it, an id or a text is null or of another type, or a document is not text.
    """
    frame = _read_parquet(data, where)
    ids = _column(frame, columns.id, where)
    texts = _column(frame, columns.text, where)
    documents: list[str | None] = [None] * frame.height
    if columns.doc is not None:
        documents = [
            _document(value, columns.doc, index, where)
            for index, value in enumerate(_column(frame, columns.doc, where))
        ]
    return [
        SourceRow(
            _identifier(ids[index], columns.id, index, where),
            _text(texts[index], columns.text, index, where),
            documents[index],
        )
        for index in range(frame.height)
    ]


def parquet_height(path: Path) -> int:
    """The number of rows of the Parquet file at `path`."""
    return int(pl.scan_parquet(path, glob=False).select(pl.len()).collect().item())


def _read_parquet(data: bytes, where: str) -> pl.DataFrame:
    try:
        return pl.read_parquet(io.BytesIO(data))
    except (pl.exceptions.PolarsError, OSError, ValueError) as error:
        raise DataError(
            f"{where} is not a readable Parquet file ({type(error).__name__})"
        ) from error


def _column(frame: pl.DataFrame, name: str, where: str) -> list[object]:
    if name not in frame.columns:
        raise DataError(
            f"{where}: the column {name!r} that the configuration names is not in the shard"
        )
    return list(frame.get_column(name).to_list())


def _identifier(value: object, column: str, index: int, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise _bad_cell(where, column, index, "must be text or a whole number")
    return str(value)


def _text(value: object, column: str, index: int, where: str) -> str:
    if not isinstance(value, str):
        raise _bad_cell(where, column, index, "must be text")
    return value


def _document(value: object, column: str, index: int, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad_cell(where, column, index, "must be text")
    return document_id(value)


def _bad_cell(where: str, column: str, index: int, problem: str) -> DataError:
    return DataError(f"{where}: the column {column!r} {problem} in row {index}")


def document_id(value: str) -> str | None:
    """The document id of the address or name `value`: itself when it already is an id, None when
    it is blank, else `u` and 16 hexadecimal characters of its SHA-256."""
    if not value.strip():
        return None
    if is_row_id(value):
        return value
    return "u" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:_DOCUMENT_DIGITS]


# ---- the export of a human set ------------------------------------------------------------


def export_rows(path: Path, data: bytes) -> list[dict[str, object]]:
    """The rows of the export `data` held by the file `path`, whose suffix names the format
    (`.jsonl`, `.tsv` or `.parquet`): the nine columns of D2's export, a blank document or
    segment of a TSV read as null.

    Raises:
        DataError: the file is not what its suffix says, is not UTF-8 text (JSON Lines, TSV),
            has a blank or malformed line, or does not have exactly the columns of the export;
            the message names the line or the row and never quotes a cell.
    """
    if path.suffix == ".parquet":
        return _parquet_export(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DataError(f"the export {path} is not UTF-8 text") from None
    return _json_lines(text) if path.suffix == ".jsonl" else _tsv(text)


class _RepeatedKey(ValueError):
    """A JSON object that holds a key twice."""


def _object_without_repeats(pairs: list[tuple[str, object]]) -> dict[str, object]:
    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            raise _RepeatedKey
        found[key] = value
    return found


def _json_lines(text: str) -> list[dict[str, object]]:
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    rows: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DataError(f"the export: line {number} is blank")
        try:
            row = json.loads(line, object_pairs_hook=_object_without_repeats)
        except _RepeatedKey:
            raise DataError(f"the export: line {number} repeats a key") from None
        except ValueError, RecursionError:
            raise DataError(f"the export: line {number} is not valid JSON") from None
        if not isinstance(row, dict):
            raise DataError(f"the export: line {number} is not a JSON object")
        rows.append(row)
    return rows


def _tsv(text: str) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    rows: list[dict[str, object]] = []
    try:
        if sorted(reader.fieldnames or ()) != sorted(HUMAN_COLUMNS):
            raise DataError("the header of the export is not its nine columns")
        for record in reader:
            if None in record or any(cell is None for cell in record.values()):
                raise DataError(
                    f"the export: line {reader.line_num} does not hold nine columns"
                )
            rows.append(_tsv_row(record, len(rows)))
    except csv.Error as error:
        raise DataError(
            f"the export is not valid TSV (line {reader.line_num}, {type(error).__name__})"
        ) from error
    return rows


def _tsv_row(record: Mapping[str, str], index: int) -> dict[str, object]:
    row: dict[str, object] = dict(record)
    row["doc_id"] = row["doc_id"] or None
    segment = row["seg_idx"]
    if segment == "":
        row["seg_idx"] = None
    else:
        try:
            row["seg_idx"] = int(str(segment))
        except ValueError:
            raise refusal(
                f"export row index {index}", "seg_idx", "must be a whole number"
            ) from None
    return row


def _parquet_export(data: bytes) -> list[dict[str, object]]:
    frame = _read_parquet(data, "the export")
    if sorted(frame.columns) != sorted(HUMAN_COLUMNS):
        raise DataError("the columns of the export are not its nine columns")
    return [dict(row) for row in frame.to_dicts()]


# ---- the manifest of the fetch ------------------------------------------------------------


class FetchRecord(NamedTuple):
    """What the manifest of a fetch vouches for: the `revision` the fetch was pinned to (None
    when it was unpinned), its `dataset` and the SHA-256 of each shard it wrote, by (language,
    file name)."""

    revision: str | None
    dataset: str | None
    digests: Mapping[tuple[str, str], str]


def read_fetch_manifest(data: bytes, *, benchmark: str, split: str) -> FetchRecord:
    """The record in the manifest `data` of a fetch of `benchmark` and `split`. The shards are
    found by the last three parts of their listed paths (the split, the language and the file),
    so a fetch directory that was moved is still read.

    Raises:
        DataError: `data` is not a manifest, is the manifest of another stage, of another
            benchmark or split, or of a fetch whose revision is neither a commit hash nor
            `unpinned`, or lists a shard outside `<split>/<language>/`.
    """
    try:
        manifest = Manifest.model_validate_json(data)
    except ValueError as error:
        raise DataError(
            f"the manifest in the source directory cannot be read ({type(error).__name__})"
        ) from error
    if manifest.stage != FETCH_STAGE:
        raise DataError(
            "the manifest in the source directory is not the one of a suite fetch "
            f"(stage {FETCH_STAGE})"
        )
    status = manifest.status
    if status.get("benchmark") != benchmark:
        raise DataError(
            f"the fetch in the source directory is not of the benchmark {benchmark}"
        )
    if status.get("split") != split:
        raise DataError(
            f"the fetch in the source directory is not of the split {split}"
        )
    revision = status.get("revision", "")
    if revision != UNPINNED and not is_revision(revision):
        raise DataError(
            "the fetch manifest records neither a commit hash nor `unpinned` as its revision"
        )
    digests: dict[tuple[str, str], str] = {}
    for item in manifest.outputs:
        parts = Path(item.path).parts
        if len(parts) < 3 or parts[-3] != split:
            raise DataError(
                f"a shard the fetch manifest lists is not under {split}/<language>/"
            )
        digests[(parts[-2], parts[-1])] = item.sha256
    return FetchRecord(
        None if revision == UNPINNED else revision, status.get("dataset"), digests
    )


def check_against_fetch(
    entries: Mapping[str, list[ManifestFile]], record: FetchRecord
) -> None:
    """Every shard in `entries` (by language, hashed) must be one the fetch listed with the
    digest it lists, and every shard the fetch listed must be there.

    Raises:
        DataError: a shard is not listed, differs from its listed digest or is missing.
    """
    found: set[tuple[str, str]] = set()
    for language, files in entries.items():
        for entry in files:
            key = (language, Path(entry.path).name)
            listed = record.digests.get(key)
            if listed is None:
                raise DataError(
                    f"the fetch manifest does not list the shard {language}/{key[1]}"
                )
            if listed != entry.sha256:
                raise DataError(
                    f"the shard {language}/{key[1]} changed after the fetch: its SHA-256 "
                    "is not the one the fetch manifest lists"
                )
            found.add(key)
    missing = sorted(set(record.digests) - found)
    if missing:
        language, name = missing[0]
        raise DataError(
            f"the shard {language}/{name} that the fetch manifest lists is not in the source"
        )
