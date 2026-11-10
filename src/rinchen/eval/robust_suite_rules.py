"""What the rows of a suite file may hold (D12, spec T11), field by field.

A suite may come from anywhere and its texts are what a model is judged on, so the reader is
as strict as the writer is exact: a row has exactly the fields of its section, each of its
type, and every text is checked (not blank, at most 2,000 characters, no lone surrogate, no
reserved symbol of the project that a model reads as a control piece; the numbered
placeholders are text). Every refusal is a `DataError` that names where the reader was (the
section, the row index and id, the field) and NEVER quotes what the file held: an id is named
only after it matched its pattern, a key of the file never.

`parse_object` is the `object_pairs_hook` of the parse: it keeps a repeated key instead of
raising, so that the refusal can say where the key was repeated.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from functools import cache
from typing import Any, Final, Literal, NamedTuple

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval.robust_numbers import MAX_NUMBER_TOKENS, numbers_in
from rinchen.eval.robust_spans import EntityTable
from rinchen.infer.script_mask import script_hint
from rinchen.infer.source_tokens import source_symbols

SECTIONS: Final = ("numbers", "placeholders", "entities", "terms", "script", "mong")
MAX_TEXT_CHARS: Final = 2000
MAX_TERM_CHARS: Final = 200

ScriptName = Literal["Mong", "Cyrl"]

_COMMON_FIELDS: Final = (
    "id",
    "src_lang",
    "tgt_lang",
    "source",
    "reference",
    "category",
)
_EXTRA_FIELDS: Final = {
    "numbers": (),
    "placeholders": (),
    "entities": ("entities",),
    "terms": ("terms",),
    "script": ("script",),
    "mong": ("source_mong",),
}
# The fields a row may not carry unless its section writes them, with their defaults.
_OPTIONAL_DEFAULTS: Final[dict[str, object]] = {
    "entities": (),
    "terms": (),
    "script": None,
    "source_mong": None,
}
_REFERENCE_REQUIRED: Final = ("numbers", "placeholders", "mong")
_NUMBER_SECTIONS: Final = ("numbers", "placeholders")
_SCRIPT_TARGETS: Final = ("khk_Cyrl", "mn_Mong")
_ID: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")
_CATEGORY: Final = re.compile(r"[a-z0-9_]{1,24}")
_ID_RULE: Final = "must be 1 to 64 characters: letters, digits, '.', '_' and '-'"
_REGISTRY: Final = "must be a language code of the registry"
_SRC_OR_NULL: Final = f"{_REGISTRY} or null"
_SRC_REQUIRED: Final = f"{_REGISTRY} (null is allowed in the 'script' section only)"


class TermPair(NamedTuple):
    """A glossary entry: the term in the source and the term the hypothesis must hold. A
    tuple, so a row's `terms` is what `score_terms` takes."""

    src: str
    tgt: str


@dataclass(frozen=True, slots=True)
class SuiteRow:
    """One row of a section. `id`, `src_lang` (None only in `script`), `tgt_lang`, `source`,
    `reference` (None only where the section allows), `category` are in every section; the
    last four fields belong to one section each: `entities` (ids of the entity table),
    `terms`, `script` (the script the output must be in, for a target that has two) and
    `source_mong` (the same source in traditional script)."""

    id: str
    src_lang: str | None
    tgt_lang: str
    source: str
    reference: str | None
    category: str
    entities: tuple[str, ...] = ()
    terms: tuple[TermPair, ...] = ()
    script: ScriptName | None = None
    source_mong: str | None = None


class JsonObject(dict[str, Any]):
    """A parsed JSON object that remembers the keys it repeated."""

    __slots__ = ("repeated",)
    repeated: tuple[str, ...]


def parse_object(pairs: list[tuple[str, Any]]) -> JsonObject:
    found = JsonObject()
    repeated: list[str] = []
    for key, value in pairs:
        if key in found:
            repeated.append(key)
        found[key] = value
    found.repeated = tuple(repeated)
    return found


def refusal(place: str, field: str | None, problem: str) -> DataError:
    """The error for a suite that breaks a rule. `place` and `field` are names the reader
    chose (a section, an index, a field of the format), plus an id that matched its
    pattern: never a text, a key or any value of the file."""
    where = place if field is None else f"{place}, field {field!r}"
    return DataError(f"not a robustness suite: {where}: {problem}")


def object_at(
    value: object, place: str, field: str | None, *, known: Collection[str] = ()
) -> JsonObject:
    """`value` as a JSON object without a repeated key. `known` are the keys the refusal
    may name when one is repeated."""
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


def is_id(value: object) -> bool:
    """Whether `value` is a well-formed id or name of the format: the only kind of string
    of a file that a message may name."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def checked_name(value: object, place: str, field: str) -> str:
    """A name or an id of the format: 1 to 64 letters, digits, '.', '_' and '-'."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise refusal(place, field, _ID_RULE)
    return value


def checked_text(
    value: object, place: str, field: str, *, limit: int = MAX_TEXT_CHARS
) -> str:
    """A text a model will be given or judged by: a string that is not blank, at most
    `limit` characters, without a lone surrogate and without a reserved symbol."""
    if not isinstance(value, str) or not value.strip():
        raise refusal(place, field, "must be a non-blank text")
    if len(value) > limit:
        raise refusal(place, field, f"must be at most {limit} characters")
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


def row_place(section: str, index: int, row_id: object = None) -> str:
    """Where a row is, for a message: its section, its index and, when it is a well-formed
    id, the id."""
    place = f"section {section!r}, row index {index}"
    if is_id(row_id):
        place = f"{place} (id {row_id!r})"
    return place


# ---- the rows ----------------------------------------------------------------------------


def read_section(section: str, raw: object, table: EntityTable) -> tuple[SuiteRow, ...]:
    """The rows of `section`, in file order, ids unique within the section."""
    if not isinstance(raw, list):
        raise refusal(f"section {section!r}", None, "must be a list of rows")
    rows: list[SuiteRow] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = read_row(section, index, item, table)
        if row.id in seen:
            raise refusal(
                row_place(section, index, row.id),
                "id",
                "appears more than once in the section",
            )
        seen.add(row.id)
        rows.append(row)
    return tuple(rows)


def read_row(section: str, index: int, raw: object, table: EntityTable) -> SuiteRow:
    place = row_place(section, index, raw.get("id") if isinstance(raw, dict) else None)
    row = _row_object(section, raw, place)
    row_id = checked_name(row["id"], place, "id")
    src_lang = _source_language(section, row["src_lang"], place)
    tgt_lang = _language(row["tgt_lang"], place, "tgt_lang", _REGISTRY)
    source = checked_text(row["source"], place, "source")
    reference = _reference(
        row["reference"], place, required=section in _REFERENCE_REQUIRED
    )
    category = row["category"]
    if not isinstance(category, str) or not _CATEGORY.fullmatch(category):
        raise refusal(place, "category", "must be 1 to 24 characters: a-z, 0-9 and '_'")
    if section in _NUMBER_SECTIONS and reference is not None:
        _require_few_numbers(reference, tgt_lang, place)
    extras = _section_fields(section, row, place, src_lang, tgt_lang, table)
    return SuiteRow(
        id=row_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        source=source,
        reference=reference,
        category=category,
        **extras,
    )


def _row_object(section: str, raw: object, place: str) -> JsonObject:
    keys = (*_COMMON_FIELDS, *_EXTRA_FIELDS[section])
    row = object_at(raw, place, None, known=keys)
    for key in keys:
        if key not in row:
            raise refusal(place, key, "is missing")
    if len(row) != len(keys):
        raise refusal(place, None, "has a field that is not part of the format")
    return row


def _source_language(section: str, value: object, place: str) -> str | None:
    if section == "script":
        return (
            None if value is None else _language(value, place, "src_lang", _SRC_OR_NULL)
        )
    return _language(value, place, "src_lang", _SRC_REQUIRED)


def _language(value: object, place: str, field: str, problem: str) -> str:
    if isinstance(value, str) and value in known_language_codes():
        return value
    raise refusal(place, field, problem)


def _reference(value: object, place: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    return checked_text(value, place, "reference")


def _require_few_numbers(reference: str, tgt_lang: str, place: str) -> None:
    if len(numbers_in(reference, tgt_lang)) > MAX_NUMBER_TOKENS:
        raise refusal(
            place,
            "reference",
            f"must not hold more than {MAX_NUMBER_TOKENS} numbers",
        )


def _section_fields(
    section: str,
    row: JsonObject,
    place: str,
    src_lang: str | None,
    tgt_lang: str,
    table: EntityTable,
) -> dict[str, Any]:
    if section == "entities":
        return {"entities": _entity_ids(row["entities"], place, tgt_lang, table)}
    if section == "terms":
        return {"terms": _term_pairs(row["terms"], place)}
    if section == "script":
        return {"script": _script(row["script"], place, tgt_lang)}
    if section == "mong":
        return {
            "source_mong": _mong_source(row["source_mong"], place, src_lang, tgt_lang)
        }
    return {}


def _entity_ids(
    value: object, place: str, tgt_lang: str, table: EntityTable
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise refusal(place, "entities", "must be a list of at least one entity id")
    listed: list[str] = []
    for position, item in enumerate(value):
        field = f"entities[{position}]"
        spellings = table.spellings.get(item) if isinstance(item, str) else None
        if spellings is None:
            raise refusal(place, field, "must be an entity id of the table")
        if not spellings.get(tgt_lang):
            raise refusal(
                place, field, "has no spelling for the target language of the row"
            )
        if item in listed:
            raise refusal(place, field, "appears more than once in the row")
        listed.append(item)
    return tuple(listed)


def _term_pairs(value: object, place: str) -> tuple[TermPair, ...]:
    if not isinstance(value, list) or not value:
        raise refusal(place, "terms", "must be a list of at least one {src, tgt} pair")
    pairs: list[TermPair] = []
    for position, item in enumerate(value):
        field = f"terms[{position}]"
        pair = object_at(item, place, field, known=("src", "tgt"))
        for key in ("src", "tgt"):
            if key not in pair:
                raise refusal(place, f"{field}.{key}", "is missing")
        if len(pair) != 2:
            raise refusal(place, field, "has a field that is not part of the format")
        pairs.append(
            TermPair(
                checked_text(pair["src"], place, f"{field}.src", limit=MAX_TERM_CHARS),
                checked_text(pair["tgt"], place, f"{field}.tgt", limit=MAX_TERM_CHARS),
            )
        )
    return tuple(pairs)


def _script(value: object, place: str, tgt_lang: str) -> ScriptName | None:
    if value is None:
        return None
    if tgt_lang not in _SCRIPT_TARGETS:
        raise refusal(
            place, "script", "is allowed only when the target is khk_Cyrl or mn_Mong"
        )
    if value == "Mong":
        return "Mong"
    if value == "Cyrl":
        return "Cyrl"
    raise refusal(place, "script", "must be 'Mong', 'Cyrl' or null")


def _mong_source(value: object, place: str, src_lang: str | None, tgt_lang: str) -> str:
    if src_lang != "khk_Cyrl":
        raise refusal(place, "src_lang", "must be 'khk_Cyrl' in the mong section")
    if tgt_lang in _SCRIPT_TARGETS:
        raise refusal(
            place,
            "tgt_lang",
            "must be neither khk_Cyrl nor mn_Mong in the mong section",
        )
    text = checked_text(value, place, "source_mong")
    if script_hint(text) != "mn_Mong":
        raise refusal(
            place, "source_mong", "must hold a letter of the Mongolian script"
        )
    return text


# ---- writing a row --------------------------------------------------------------------------


def row_document(section: str, index: int, row: SuiteRow) -> dict[str, Any]:
    """The JSON form of `row`. A field its section does not write must hold its default:
    writing the row would drop it without a sign.

    Raises:
        ValueError: `row` holds a field the section does not write.
    """
    written = _EXTRA_FIELDS[section]
    for field, default in _OPTIONAL_DEFAULTS.items():
        if field not in written and getattr(row, field) != default:
            raise ValueError(
                f"{row_place(section, index)}: the field {field!r} is not written in this "
                "section, so it would be dropped"
            )
    document: dict[str, Any] = {
        "id": row.id,
        "src_lang": row.src_lang,
        "tgt_lang": row.tgt_lang,
        "source": row.source,
        "reference": row.reference,
        "category": row.category,
    }
    if "entities" in written:
        document["entities"] = list(row.entities)
    if "terms" in written:
        document["terms"] = [{"src": pair.src, "tgt": pair.tgt} for pair in row.terms]
    if "script" in written:
        document["script"] = row.script
    if "source_mong" in written:
        document["source_mong"] = row.source_mong
    return document
