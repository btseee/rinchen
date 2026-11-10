"""What the file of an evaluation suite may hold (D13, spec T2 and T3), field by field.

A suite is what systems are asked to translate and are scored against, so the reader is as strict
as the writer is exact: exactly the fields of the format, each of its type, every language a code
of the registry (`khk_Mong` is refused BY NAME, whatever the registry holds: Inner Mongolian data
is outside the project), every id and document of its pattern, every sentence checked (not blank,
at most 5,000 characters, no line break, no lone surrogate, no reserved symbol of the project that
a model reads as a control piece; the numbered placeholders are text).

Every refusal is a `DataError` that names where the reader was (a field of the format, a row index
and an id that matched its pattern) and NEVER quotes what the file held: a language code is named
only when it has the shape of one, a key of the file never. `parse_object` (D12's hook) keeps a
repeated key so that the refusal can say where it was repeated.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Final

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval.robust_suite_rules import JsonObject, is_id
from rinchen.infer.engine_types import LINE_BREAKS
from rinchen.infer.source_tokens import source_symbols

__all__ = [
    "FORBIDDEN_LANGUAGES",
    "HUB_LANGUAGE",
    "KINDS",
    "MAX_ROWS",
    "MAX_SENTENCE_CHARS",
    "MAX_SUITE_BYTES",
    "PURPOSE_OF_SPLIT",
    "SPLITS",
    "SPLITS_OF_KIND",
    "SUITE_SCHEMA",
    "SuiteRow",
    "SuiteSource",
    "checked_language",
    "checked_name",
    "checked_sentence",
    "direction_name",
    "excluded_languages",
    "is_row_id",
    "object_at",
    "read_directions",
    "read_rows",
    "read_source",
    "refusal",
    "require_rows_for_directions",
    "split_direction",
]

SUITE_SCHEMA: Final = "rinchen.eval.suite/1"
KINDS: Final = ("benchmark", "human", "fixture")
SPLITS: Final = ("dev", "devtest", "test")
SPLITS_OF_KIND: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "benchmark": ("dev", "devtest"),
        "human": ("test",),
        "fixture": SPLITS,
    }
)
# `dev` is for choosing a model; `devtest` and `test` are for reporting one (never for tuning).
PURPOSE_OF_SPLIT: Final[Mapping[str, str]] = MappingProxyType(
    {"dev": "selection", "devtest": "report", "test": "report"}
)
MAX_ROWS: Final = 100_000
MAX_SUITE_BYTES: Final = 64 * 2**20
MAX_SENTENCE_CHARS: Final = 5000
FORBIDDEN_LANGUAGES: Final = ("khk_Mong",)
HUB_LANGUAGE: Final = "khk_Cyrl"
FLORES_BENCHMARK: Final = "flores_plus"
FLORES_EXCLUDED: Final = ("mn_Mong",)

_FORBIDDEN_RULE: Final = "refused by name: Inner Mongolian data is never used"
_MAX_FILE_NAME_CHARS: Final = 200
_ROW_ID: Final = re.compile(r"[A-Za-z0-9._:-]{1,96}")
_CODE_SHAPE: Final = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}")
_DATASET: Final = re.compile(r"[A-Za-z0-9._/-]{1,200}")
_REVISION: Final = re.compile(r"[0-9a-f]{40}")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_ROW_FIELDS: Final = ("id", "doc", "text")
_SOURCE_FIELDS: Final = (
    "dataset",
    "revision",
    "pinned",
    "files",
    "missing_languages",
)
_NAME_RULE: Final = "must be 1 to 64 characters: letters, digits, '.', '_' and '-'"
_ROW_ID_RULE: Final = (
    "must be 1 to 96 characters: letters, digits, '.', '_', ':' and '-'"
)
_LANGUAGE_RULE: Final = "must be a language code of the registry"
_DIRECTION_RULE: Final = (
    "must be <source>-<target>: two language codes of the registry joined by '-'"
)
_FILE_NAME_RULE: Final = (
    "must be a relative path of at most 200 characters: no '..', no leading '/' and "
    "no control character"
)


@dataclass(frozen=True, slots=True)
class SuiteSource:
    """Where a suite came from: the Hub `dataset` it was fetched from (None for a local
    file), the `revision` (a commit hash) the fetch was pinned to, whether it was `pinned`,
    the SHA-256 of every input file the builder read (`files`: the suite is bound to its
    inputs) and the registry languages the source lacks (`missing_languages`, sorted)."""

    dataset: str | None
    revision: str | None
    pinned: bool
    files: Mapping[str, str]
    missing_languages: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "missing_languages", tuple(self.missing_languages))


@dataclass(frozen=True, slots=True)
class SuiteRow:
    """One sentence in one or more languages: its `id` (unique in the suite), the `doc` it
    belongs to (None when the source has no document) and `text`, language code to sentence.
    A direction reads the rows that have both of its languages."""

    id: str
    doc: str | None
    text: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", MappingProxyType(dict(self.text)))


def refusal(place: str, field: str | None, problem: str) -> DataError:
    """The error for a suite that breaks a rule. `place` and `field` are names the reader
    chose (a field of the format, a row index) plus an id that matched its pattern: never a
    text, a key or any value of the file."""
    where = place if field is None else f"{place}, field {field!r}"
    return DataError(f"not an evaluation suite: {where}: {problem}")


def object_at(
    value: object, place: str, field: str | None, *, known: Collection[str] = ()
) -> JsonObject:
    """`value` as a JSON object without a repeated key. `known` are the keys the refusal may
    name when one is repeated."""
    if not isinstance(value, JsonObject):
        raise refusal(place, field, "must be an object")
    if value.repeated:
        named = next((key for key in value.repeated if key in known), None)
        problem = (
            f"the field {named!r} appears more than once"
            if named is not None
            else "a key appears more than once"
        )
        raise refusal(place, field, problem)
    return value


def checked_name(value: object, place: str, field: str) -> str:
    """A name of the format (a suite or a benchmark): the id pattern of D12's files."""
    if not isinstance(value, str) or not is_id(value):
        raise refusal(place, field, _NAME_RULE)
    return value


def is_row_id(value: object) -> bool:
    """Whether `value` is a well-formed row id or document: the only kind of string of a
    row that a message may name."""
    return isinstance(value, str) and _ROW_ID.fullmatch(value) is not None


def checked_row_id(value: object, place: str, field: str) -> str:
    if not isinstance(value, str) or not is_row_id(value):
        raise refusal(place, field, _ROW_ID_RULE)
    return value


def checked_language(
    value: object, place: str, field: str, *, excluded: Collection[str] = ()
) -> str:
    """A language code of the registry. `khk_Mong` is refused by name, before the registry
    is asked; a code in `excluded` is refused as not a language of this benchmark. Any other
    string is named only when it has the shape of a code (`abc_Latn`)."""
    if not isinstance(value, str):
        raise refusal(place, field, _LANGUAGE_RULE)
    for forbidden in FORBIDDEN_LANGUAGES:
        if value == forbidden:
            raise refusal(place, field, f"{forbidden} is {_FORBIDDEN_RULE}")
    for code in excluded:
        if value == code:
            raise refusal(place, field, f"{code} is not a language of this benchmark")
    if value in known_language_codes():
        return value
    shaped = value if _CODE_SHAPE.fullmatch(value) else None
    if shaped is None:
        raise refusal(place, field, _LANGUAGE_RULE)
    raise refusal(place, field, f"{shaped} is not a language code of the registry")


def checked_sentence(value: object, place: str, field: str) -> str:
    """A sentence a model will be given or judged by: a string that is not blank, at most
    `MAX_SENTENCE_CHARS` characters, with no line break, no lone surrogate and no reserved
    symbol other than a numbered placeholder."""
    if not isinstance(value, str) or not value.strip():
        raise refusal(place, field, "must be a non-blank sentence")
    if len(value) > MAX_SENTENCE_CHARS:
        raise refusal(place, field, f"must be at most {MAX_SENTENCE_CHARS} characters")
    if LINE_BREAKS.search(value):
        raise refusal(place, field, "must not hold a line break")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise refusal(place, field, "holds a lone surrogate") from None
    if _reserved_symbols().search(value):
        raise refusal(
            place,
            field,
            "holds a reserved symbol of the project (a direction tag or a structural symbol)",
        )
    return value


@cache
def _reserved_symbols() -> re.Pattern[str]:
    return re.compile("|".join(re.escape(symbol) for symbol in source_symbols()))


def direction_name(src: str, tgt: str) -> str:
    """The name of the direction from `src` to `tgt`, as a suite lists it."""
    return f"{src}-{tgt}"


def split_direction(
    direction: str, *, place: str = "direction", field: str | None = None
) -> tuple[str, str]:
    """The source and the target language of a direction name.

    Raises:
        DataError: the name is not two non-empty parts joined by one hyphen (the message
            does not quote it; `place` and `field` say where it was read).
    """
    src, separator, tgt = direction.partition("-")
    if not separator or not src or not tgt or "-" in tgt:
        raise refusal(place, field, _DIRECTION_RULE)
    return src, tgt


def excluded_languages(kind: str, benchmark: str) -> tuple[str, ...]:
    """The registry languages a suite of this kind and benchmark has no text in: FLORES+ has
    no traditional-script Mongolian that the project uses."""
    if kind == "benchmark" and benchmark == FLORES_BENCHMARK:
        return FLORES_EXCLUDED
    return ()


# ---- the source -----------------------------------------------------------------------------


def read_source(raw: object, *, excluded: Collection[str] = ()) -> SuiteSource:
    """The `source` object of a suite file."""
    source = object_at(raw, "suite", "source", known=_SOURCE_FIELDS)
    for key in _SOURCE_FIELDS:
        if key not in source:
            raise refusal("source", key, "is missing")
    if len(source) != len(_SOURCE_FIELDS):
        raise refusal("source", None, "has a field that is not part of the format")
    dataset = source["dataset"]
    if dataset is not None and not (
        isinstance(dataset, str) and _DATASET.fullmatch(dataset)
    ):
        raise refusal(
            "source",
            "dataset",
            "must be a dataset id of the Hub (letters, digits, '.', '_', '-' and '/', "
            "at most 200 characters) or null",
        )
    revision = source["revision"]
    if revision is not None and not (
        isinstance(revision, str) and _REVISION.fullmatch(revision)
    ):
        raise refusal(
            "source", "revision", "must be 40 lowercase hexadecimal characters or null"
        )
    pinned = source["pinned"]
    if not isinstance(pinned, bool):
        raise refusal("source", "pinned", "must be true or false")
    if pinned and revision is None:
        raise refusal("source", "pinned", "can be true only when a revision is set")
    return SuiteSource(
        dataset=dataset,
        revision=revision,
        pinned=pinned,
        files=_files(source["files"]),
        missing_languages=_missing_languages(source["missing_languages"], excluded),
    )


def _files(raw: object) -> dict[str, str]:
    files = object_at(raw, "source", "files")
    found: dict[str, str] = {}
    for index, (name, digest) in enumerate(files.items()):
        field = f"files[{index}]"
        if not _is_file_name(name):
            raise refusal("source", field, _FILE_NAME_RULE)
        if not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
            raise refusal(
                "source",
                field,
                "must have a digest of 64 lowercase hexadecimal characters",
            )
        found[name] = digest
    return found


def _is_file_name(name: str) -> bool:
    return (
        0 < len(name) <= _MAX_FILE_NAME_CHARS
        and name.isprintable()
        # A leading '/' or a doubled one is an empty component.
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def _missing_languages(raw: object, excluded: Collection[str]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise refusal("source", "missing_languages", "must be a list of language codes")
    codes = tuple(
        checked_language(
            item, "source", f"missing_languages[{index}]", excluded=excluded
        )
        for index, item in enumerate(raw)
    )
    if list(codes) != sorted(set(codes)):
        raise refusal(
            "source", "missing_languages", "must be sorted, without a repeated code"
        )
    return codes


# ---- the directions -------------------------------------------------------------------------


def read_directions(raw: object, *, excluded: Collection[str] = ()) -> tuple[str, ...]:
    """The `directions` list of a suite file: unique names of two different registry
    languages, at least one."""
    if not isinstance(raw, list):
        raise refusal("suite", "directions", "must be a list of directions")
    if not raw:
        raise refusal("suite", "directions", "must hold at least one direction")
    found: list[str] = []
    for index, item in enumerate(raw):
        field = f"directions[{index}]"
        if not isinstance(item, str):
            raise refusal("suite", field, _DIRECTION_RULE)
        src, tgt = split_direction(item, place="suite", field=field)
        checked_language(src, "suite", field, excluded=excluded)
        checked_language(tgt, "suite", field, excluded=excluded)
        if src == tgt:
            raise refusal("suite", field, "must name two different languages")
        if item in found:
            raise refusal("suite", field, "appears more than once")
        found.append(item)
    return tuple(found)


def require_rows_for_directions(
    directions: tuple[str, ...], rows: tuple[SuiteRow, ...]
) -> None:
    """Every direction needs at least one row that has both of its languages: a direction
    nothing can be scored on does not exist in a suite."""
    for index, direction in enumerate(directions):
        src, tgt = split_direction(direction)
        if not any(src in row.text and tgt in row.text for row in rows):
            raise refusal(
                "suite", f"directions[{index}]", "has no row with both languages"
            )


# ---- the rows -------------------------------------------------------------------------------


def row_place(index: int, row_id: object = None) -> str:
    """Where a row is, for a message: its index and, when it is a well-formed id, the id."""
    place = f"row index {index}"
    if is_row_id(row_id):
        place = f"{place} (id {row_id!r})"
    return place


def read_rows(raw: object, *, excluded: Collection[str] = ()) -> tuple[SuiteRow, ...]:
    """The rows of a suite file, in file order, ids unique: at least one and at most
    `MAX_ROWS` (counted before any row is read)."""
    if not isinstance(raw, list):
        raise refusal("suite", "rows", "must be a list of rows")
    if not raw:
        raise refusal("suite", "rows", "must hold at least one row")
    if len(raw) > MAX_ROWS:
        raise refusal("suite", "rows", f"must hold at most {MAX_ROWS} rows")
    rows: list[SuiteRow] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = _read_row(index, item, excluded)
        if row.id in seen:
            raise refusal(row_place(index, row.id), "id", "appears more than once")
        seen.add(row.id)
        rows.append(row)
    return tuple(rows)


def _read_row(index: int, raw: object, excluded: Collection[str]) -> SuiteRow:
    place = row_place(index, raw.get("id") if isinstance(raw, dict) else None)
    row = object_at(raw, place, None, known=_ROW_FIELDS)
    for key in _ROW_FIELDS:
        if key not in row:
            raise refusal(place, key, "is missing")
    if len(row) != len(_ROW_FIELDS):
        raise refusal(place, None, "has a field that is not part of the format")
    row_id = checked_row_id(row["id"], place, "id")
    doc = row["doc"]
    return SuiteRow(
        id=row_id,
        doc=None if doc is None else checked_row_id(doc, place, "doc"),
        text=_read_text(row["text"], place, excluded),
    )


def _read_text(raw: object, place: str, excluded: Collection[str]) -> dict[str, str]:
    text = object_at(raw, place, "text", known=known_language_codes())
    if not text:
        raise refusal(place, "text", "must hold a sentence in at least one language")
    found: dict[str, str] = {}
    for language, sentence in text.items():
        code = checked_language(language, place, "text", excluded=excluded)
        found[code] = checked_sentence(sentence, place, f"text.{code}")
    return found
