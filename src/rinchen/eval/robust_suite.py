"""The suite file of D12 (spec T11, T12): the rows a robustness run is made of, and the requests
that ask a system to write the hypotheses.

A suite is one JSON document, schema `rinchen.robustness.suite/1`, UTF-8 with sorted keys,
two-space indent and one final LF, so the same suite gives the same bytes:

    {"schema", "name", "entities": {id: {language: [spellings]}}, "sections": {section: [row]}}

with up to six sections (`numbers`, `placeholders`, `entities`, `terms`, `script`, `mong`), at
least one of them holding a row. `robust_suite_rules` says what a row and the entity table may
hold. The file is read ONCE (`read_suite` returns the suite and the SHA-256 of the bytes it
parsed) and is at most 8 MiB, which is checked before anything is parsed.

`suite_requests` lists what the runner sends the engine, in a fixed order: the sections in the
order of `SECTIONS`, the rows in file order, and for a `mong` row its arms `cyrl`, `mong` and,
when a converted text is given, `converted`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite_rules import (
    MAX_TERM_CHARS,
    MAX_TEXT_CHARS,
    SECTIONS,
    ScriptName,
    SuiteRow,
    TermPair,
    checked_name,
    checked_text,
    object_at,
    parse_object,
    read_section,
    refusal,
    row_document,
)
from rinchen.utils.part_file import replace_file

__all__ = [
    "MAX_SUITE_BYTES",
    "MAX_TEXT_CHARS",
    "SECTIONS",
    "SUITE_SCHEMA",
    "Request",
    "RequestKey",
    "ScriptName",
    "Suite",
    "SuiteRow",
    "TermPair",
    "read_suite",
    "suite_requests",
    "write_suite",
]

SUITE_SCHEMA: Final = "rinchen.robustness.suite/1"
MAX_SUITE_BYTES: Final = 8 * 1024 * 1024

_TOP_KEYS: Final = ("schema", "name", "entities", "sections")
_TABLE_ID: Final = re.compile(r"[a-z0-9_]{1,32}")
_MAX_SPELLINGS: Final = 8


@dataclass(frozen=True)
class Suite:
    """A robustness suite: its `name`, the `entities` table the `entities` rows refer to
    and the rows of each section, in file order (`sections` is read-only and keeps the
    order of `SECTIONS`; a section the file did not have is absent, one it had empty is
    an empty tuple).

    Raises:
        ValueError: `sections` has a key that is no section of the format (a typo would
            otherwise drop a whole section without a sign).
    """

    name: str
    entities: EntityTable
    sections: Mapping[str, tuple[SuiteRow, ...]]

    def __post_init__(self) -> None:
        if set(self.sections) - set(SECTIONS):
            raise ValueError(
                f"a suite has no such section (the sections are {', '.join(SECTIONS)})"
            )
        ordered = {
            section: tuple(self.sections[section])
            for section in SECTIONS
            if section in self.sections
        }
        object.__setattr__(self, "sections", MappingProxyType(ordered))

    def row_counts(self) -> dict[str, int]:
        """The number of rows of each section the suite has, in the order of `SECTIONS`."""
        return {section: len(rows) for section, rows in self.sections.items()}

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of the suite file: sorted keys, two-space indent, one final LF.
        It writes what it is given and does not validate: `from_bytes` is the validator.

        Raises:
            ValueError: a row holds a field its section does not write.
        """
        document = {
            "schema": SUITE_SCHEMA,
            "name": self.name,
            "entities": {
                entity_id: {
                    language: list(spellings)
                    for language, spellings in by_language.items()
                }
                for entity_id, by_language in self.entities.spellings.items()
            },
            "sections": {
                section: [
                    row_document(section, index, row) for index, row in enumerate(rows)
                ]
                for section, rows in self.sections.items()
            },
        }
        text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
        return f"{text}\n".encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> Suite:
        """The suite in `data`, which must be exactly the document `to_bytes` writes in
        shape: this schema, every field and no other, each of its type.

        Raises:
            DataError: `data` is too large, not UTF-8, not JSON, or not that document;
                the message names the section, the row and the field and never quotes
                what the file held.
        """
        top = _document(data)
        name = checked_name(top["name"], "suite", "name")
        table = _entity_table(top["entities"])
        return cls(
            name=name, entities=table, sections=_sections(top["sections"], table)
        )


def read_suite(path: Path) -> tuple[Suite, str]:
    """The suite in the file `path` and the SHA-256 of the bytes that were parsed. The
    file is opened once and at most one byte over the cap is read.

    Raises:
        DataError: the file cannot be read, or is not a suite (`Suite.from_bytes`).
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_SUITE_BYTES + 1)
    except OSError as error:
        raise DataError(f"cannot read the suite file {path}") from error
    return Suite.from_bytes(data), hashlib.sha256(data).hexdigest()


def write_suite(suite: Suite, path: Path) -> str:
    """Write `suite` to `path` so that the file appears whole; return the SHA-256 of the
    bytes written.

    Raises:
        DataError: the suite is larger than the reader accepts.
        ValueError: a row holds a field its section does not write.
    """
    data = suite.to_bytes()
    if len(data) > MAX_SUITE_BYTES:
        raise _too_large()
    replace_file(path, data)
    return hashlib.sha256(data).hexdigest()


def _too_large() -> DataError:
    return DataError(
        f"not a robustness suite: the file is larger than {MAX_SUITE_BYTES // 2**20} MiB"
    )


def _document(data: bytes) -> dict[str, object]:
    if len(data) > MAX_SUITE_BYTES:
        raise _too_large()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise refusal("suite", None, "the text is not UTF-8") from None
    try:
        parsed = json.loads(text, object_pairs_hook=parse_object)
    except ValueError, RecursionError:
        raise refusal("suite", None, "the text is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise refusal("suite", None, "the document is not a JSON object")
    top = object_at(parsed, "suite", None, known=_TOP_KEYS)
    if "schema" not in top:
        raise refusal("suite", "schema", "is missing")
    if top["schema"] != SUITE_SCHEMA:
        raise refusal("suite", "schema", f"must be {SUITE_SCHEMA!r}")
    for key in _TOP_KEYS:
        if key not in top:
            raise refusal("suite", key, "is missing")
    if len(top) != len(_TOP_KEYS):
        raise refusal("suite", None, "has a field that is not part of the format")
    return top


def _entity_table(raw: object) -> EntityTable:
    """The entity table: `id -> language -> 1 to 8 spellings`."""
    place = "entity table"
    entries = object_at(raw, place, None)
    spellings: dict[str, dict[str, tuple[str, ...]]] = {}
    for index, (entity_id, languages) in enumerate(entries.items()):
        entry = f"{place}, entry index {index}"
        if not _TABLE_ID.fullmatch(entity_id):
            raise refusal(entry, "id", "must be 1 to 32 characters: a-z, 0-9 and '_'")
        spellings[entity_id] = _spellings_by_language(
            languages, f"{entry} (id {entity_id!r})"
        )
    return EntityTable(spellings)


def _spellings_by_language(languages: object, entry: str) -> dict[str, tuple[str, ...]]:
    codes = known_language_codes()
    found: dict[str, tuple[str, ...]] = {}
    for language, items in object_at(
        languages, entry, "languages", known=codes
    ).items():
        if language not in codes:
            raise refusal(
                entry,
                "languages",
                "holds a language that is not a code of the registry",
            )
        found[language] = _spellings(items, entry, language)
    return found


def _spellings(items: Any, entry: str, language: str) -> tuple[str, ...]:
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_SPELLINGS:
        raise refusal(
            entry, language, f"must be a list of 1 to {_MAX_SPELLINGS} spellings"
        )
    return tuple(
        checked_text(item, entry, f"{language}[{position}]", limit=MAX_TERM_CHARS)
        for position, item in enumerate(items)
    )


def _sections(raw: object, table: EntityTable) -> dict[str, tuple[SuiteRow, ...]]:
    found = object_at(raw, "suite", "sections", known=SECTIONS)
    if set(found) - set(SECTIONS):
        raise refusal(
            "suite", "sections", "has a key that is not a section of the format"
        )
    sections = {
        section: read_section(section, found[section], table)
        for section in SECTIONS
        if section in found
    }
    if not any(sections.values()):
        raise refusal("suite", "sections", "must hold at least one row")
    return sections


# ---- the requests ---------------------------------------------------------------------------


class RequestKey(NamedTuple):
    """Which request of a suite: its section, the row id and the arm (`main`, or for a
    `mong` row `cyrl`, `mong` or `converted`)."""

    section: str
    id: str
    arm: str


@dataclass(frozen=True, slots=True)
class Request:
    """One text the runner asks a system to translate into `tgt_lang` (in `script` when
    the target has two), from `src_lang` (None: the model reads the script itself)."""

    key: RequestKey
    text: str
    src_lang: str | None
    tgt_lang: str
    script: ScriptName | None


def suite_requests(
    suite: Suite, converted: Mapping[str, str] | None = None
) -> list[Request]:
    """The requests of `suite`: the sections in the order of `SECTIONS`, the rows in file
    order, one `main` request per row. A `mong` row makes the arms `cyrl` (its Cyrillic
    `source` as `khk_Cyrl`), `mong` (`source_mong` as `mn_Mong`) and, when `converted`
    maps its id to a Cyrillic text, `converted` (that text as `khk_Cyrl`).

    Raises:
        DataError: `converted` has ids that are no row of the `mong` section.
    """
    converted = converted or {}
    mong_ids = {row.id for row in suite.sections.get("mong", ())}
    stray = sum(1 for row_id in converted if row_id not in mong_ids)
    if stray:
        raise DataError(
            f"converted holds {stray} id(s) that are no row of the 'mong' section"
        )
    requests: list[Request] = []
    for section, rows in suite.sections.items():
        for row in rows:
            if section == "mong":
                requests.extend(_mong_requests(row, converted.get(row.id)))
            else:
                requests.append(
                    _request(section, row, "main", row.source, row.src_lang)
                )
    return requests


def _request(
    section: str, row: SuiteRow, arm: str, text: str, src_lang: str | None
) -> Request:
    return Request(
        key=RequestKey(section, row.id, arm),
        text=text,
        src_lang=src_lang,
        tgt_lang=row.tgt_lang,
        script=row.script,
    )


def _mong_requests(row: SuiteRow, converted: str | None) -> list[Request]:
    if row.source_mong is None:
        raise ValueError("a row of the mong section has no source_mong")
    requests = [
        _request("mong", row, "cyrl", row.source, "khk_Cyrl"),
        _request("mong", row, "mong", row.source_mong, "mn_Mong"),
    ]
    if converted is not None:
        requests.append(_request("mong", row, "converted", converted, "khk_Cyrl"))
    return requests
