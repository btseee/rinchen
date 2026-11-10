"""The readers that turn rows of LOCAL files into a suite (D13, spec T4), over plain rows so that no
test needs a dataset or a network: reading a Parquet shard, a TSV or a JSON Lines file into rows is
the command's (`rinchen.cli.eval_suite_files`), which may use polars; this module never does.

`lines_rows` reads a text file with one sentence per line (NTREX, or any multi-way parallel text):
the id of a sentence is its line number. `build_multiway` joins the rows of several languages by
id: the hub language Khalkha sets the order of the rows and is one side of every direction, every
other language must hold exactly the ids of the hub language (a language with more or fewer is
refused with the counts, never intersected: no row is dropped), and the directions are Khalkha to
and from each partner present, in the order of `standard_directions`. `human_rows` reads D2's
export of a human set: every row a human translation for testing, one set, the directions the
pairs of its rows and the reverse of a pair never added (the source of a reverse direction would
be a translation).

Every builder returns the suite its own bytes read back to, so a suite that the reader of
`rinchen.eval.suites` would refuse is refused here, with its row index and field. A message names a
language of the registry, a count, a line or a row index and never a text of the source.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

from rinchen.errors import ConfigError, DataError
from rinchen.eval.suite_rules import (
    HUB_LANGUAGE,
    SuiteRow,
    SuiteSource,
    checked_language,
    checked_sentence,
    direction_name,
    excluded_languages,
    is_row_id,
    refusal,
    split_direction,
)
from rinchen.eval.suites import Suite, standard_directions

__all__ = [
    "HUMAN_BENCHMARK",
    "HUMAN_COLUMNS",
    "SourceRow",
    "build_multiway",
    "human_rows",
    "lines_rows",
]

# D2's export (`rinchen.annotate.export`), in the order it writes them.
HUMAN_COLUMNS: Final = (
    "src",
    "tgt",
    "src_lang",
    "tgt_lang",
    "doc_id",
    "seg_idx",
    "set",
    "origin",
    "split",
)
HUMAN_BENCHMARK: Final = "human"
_HUMAN_KIND: Final = "human"
_HUMAN_SPLIT: Final = "test"
_DOC_ID_RULE: Final = (
    "must be 1 to 96 characters (letters, digits, '.', '_', ':' and '-') or null"
)


class SourceRow(NamedTuple):
    """One sentence of one language as a source file holds it: its `id`, its `text` and the `doc`
    it belongs to (None when the source has no document)."""

    id: str
    text: str
    doc: str | None


# ---- a text file --------------------------------------------------------------------------


def lines_rows(language: str, data: bytes) -> list[SourceRow]:
    """The sentences of the text file `data` of `language`: UTF-8, one per line, lines ended by LF
    (the last may lack it). The id of a line is `l` and its 1-based number to six digits.

    Raises:
        DataError: `language` is not a registry language (`khk_Mong` by name), the bytes are not
            UTF-8 or are empty, a line ends with CR or is blank; the message names the language
            and the line and never what the line held.
    """
    checked_language(language, "lines file", "language")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DataError(f"the file of {language} is not UTF-8 text") from None
    if not text:
        raise DataError(f"the file of {language} is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    rows: list[SourceRow] = []
    for number, line in enumerate(lines, start=1):
        if "\r" in line:
            raise DataError(
                f"the file of {language}: line {number} holds a carriage return; lines "
                "must end with LF, not CRLF"
            )
        if not line.strip():
            raise DataError(f"the file of {language}: line {number} is blank")
        rows.append(SourceRow(f"l{number:06d}", line, None))
    return rows


# ---- languages that share their ids -------------------------------------------------------


def build_multiway(
    *,
    name: str,
    kind: str,
    benchmark: str,
    split: str,
    languages: Mapping[str, Sequence[SourceRow]],
    configured: Sequence[str],
    source: SuiteSource,
) -> Suite:
    """The suite of the rows of `languages` (a multi-way parallel benchmark), joined by id.

    The rows follow the order of the hub language. The document of a row is the hub's. The
    directions are Khalkha to and from each partner present, in the order of
    `standard_directions` (a partner that list does not know follows, sorted). The languages of
    `configured` that `languages` lacks are the `missing_languages` of the source, sorted; the
    ones `source` was given are replaced.

    Raises:
        ConfigError: the hub language is not among `languages`.
        DataError: a language is not a registry language of this benchmark (`khk_Mong` by name),
            a language has an id twice or not exactly the ids of the hub language, no partner
            is present, or the suite is not one (`Suite.from_bytes`).
    """
    excluded = excluded_languages(kind, benchmark)
    for code in (*languages, *configured):
        checked_language(code, "languages", "code", excluded=excluded)
    if HUB_LANGUAGE not in languages:
        raise ConfigError(
            f"the language {HUB_LANGUAGE} is required: it sets the order of the rows and "
            "is one side of every direction"
        )
    by_id = {language: _by_id(language, rows) for language, rows in languages.items()}
    hub = by_id[HUB_LANGUAGE]
    if not hub:
        raise DataError(f"{HUB_LANGUAGE} has no row")
    for language, rows in by_id.items():
        if rows.keys() != hub.keys():
            raise DataError(
                f"{language} has {len(rows)} ids and {HUB_LANGUAGE} has {len(hub)}, not "
                f"the same ones: every language must hold exactly the ids of {HUB_LANGUAGE} "
                "(the rows of the languages are joined by id and never intersected)"
            )
    suite_rows = tuple(
        SuiteRow(
            id=row_id,
            doc=hub_row.doc,
            text={language: by_id[language][row_id].text for language in languages},
        )
        for row_id, hub_row in hub.items()
    )
    present = set(languages)
    suite = Suite(
        name=name,
        kind=kind,
        benchmark=benchmark,
        split=split,
        source=dataclasses.replace(
            source, missing_languages=tuple(sorted(set(configured) - present))
        ),
        directions=_partner_directions(present),
        rows=suite_rows,
    )
    return Suite.from_bytes(suite.to_bytes())


def _by_id(language: str, rows: Sequence[SourceRow]) -> dict[str, SourceRow]:
    found = {row.id: row for row in rows}
    if len(found) != len(rows):
        raise DataError(
            f"{language} has an id more than once: {len(found)} different ids in "
            f"{len(rows)} rows"
        )
    return found


def _partner_directions(present: set[str]) -> tuple[str, ...]:
    """Khalkha to and from each partner in `present`: the standard directions that apply, then
    those of any other partner (traditional-script Mongolian in a fixture), sorted."""
    standard = [
        direction
        for direction in standard_directions()
        if all(language in present for language in split_direction(direction))
    ]
    known = {
        language for direction in standard for language in split_direction(direction)
    }
    extra = sorted(present - known - {HUB_LANGUAGE})
    directions = [
        *standard,
        *(
            direction_name(*pair)
            for partner in extra
            for pair in ((HUB_LANGUAGE, partner), (partner, HUB_LANGUAGE))
        ),
    ]
    if not directions:
        raise DataError(
            f"no language besides {HUB_LANGUAGE} is present, so the suite would have no "
            "direction"
        )
    return tuple(directions)


# ---- the export of a human set ------------------------------------------------------------


class _Item(NamedTuple):
    src_lang: str
    tgt_lang: str
    doc_id: str | None
    seg_idx: int | None
    src: str
    tgt: str


def human_rows(
    rows: Sequence[Mapping[str, object]], *, name: str, source: SuiteSource
) -> Suite:
    """The suite of the human set `rows` (the rows of D2's export, in any order).

    Every row must be `origin` `human` and `split` `test`, all in one `set`. A row's text holds
    its two languages; the directions are the (source, target) pairs present, sorted, and one
    export never holds both ways of a pair. The rows are sorted by (source language, target
    language, document, segment, source, target) before they are numbered, so the suite does not
    depend on the order of the export: the id of a row is `<doc_id>-<seg_idx:05d>`, or
    `s<place:06d>` (its place in that order) when it has no document or no segment number.

    Raises:
        DataError: a row is not what the export writes (the message names its index in `rows` and
            the field), the export holds two sets or both ways of a pair, or the suite is not one
            (`Suite.from_bytes`).
    """
    if not rows:
        raise DataError("the export holds no row")
    items = [_item(index, row) for index, row in enumerate(rows)]
    _require_one_set(rows)
    pairs = {(item.src_lang, item.tgt_lang) for item in items}
    for src, tgt in sorted(pairs):
        if (tgt, src) in pairs and src < tgt:
            raise DataError(
                f"the export holds {src} to {tgt} and {tgt} to {src}: a suite row cannot "
                "say which way it translates, so one export holds one way of each pair"
            )
    ordered = sorted(
        items,
        key=lambda item: (
            item.src_lang,
            item.tgt_lang,
            item.doc_id or "",
            item.seg_idx or 0,
            item.src,
            item.tgt,
        ),
    )
    suite = Suite(
        name=name,
        kind=_HUMAN_KIND,
        benchmark=HUMAN_BENCHMARK,
        split=_HUMAN_SPLIT,
        source=source,
        directions=tuple(direction_name(src, tgt) for src, tgt in sorted(pairs)),
        rows=tuple(
            SuiteRow(
                id=_human_id(place, item),
                doc=item.doc_id,
                text={item.src_lang: item.src, item.tgt_lang: item.tgt},
            )
            for place, item in enumerate(ordered, start=1)
        ),
    )
    return Suite.from_bytes(suite.to_bytes())


def _human_id(place: int, item: _Item) -> str:
    if item.doc_id is not None and item.seg_idx is not None:
        return f"{item.doc_id}-{item.seg_idx:05d}"
    return f"s{place:06d}"


def _item(index: int, row: Mapping[str, object]) -> _Item:
    place = f"export row index {index}"
    for column in HUMAN_COLUMNS:
        if column not in row:
            raise refusal(place, column, "is missing")
    if len(row) != len(HUMAN_COLUMNS):
        raise refusal(place, None, "has a column that is not part of the export")
    if row["origin"] != "human":
        raise refusal(
            place, "origin", "must be human: only human translations are tested"
        )
    if row["split"] != _HUMAN_SPLIT:
        raise refusal(place, "split", f"must be {_HUMAN_SPLIT}")
    if not isinstance(row["set"], str) or not row["set"].strip():
        raise refusal(place, "set", "must be the name of the set")
    src_lang = checked_language(row["src_lang"], place, "src_lang")
    tgt_lang = checked_language(row["tgt_lang"], place, "tgt_lang")
    if src_lang == tgt_lang:
        raise refusal(place, "tgt_lang", "must differ from src_lang")
    return _Item(
        src_lang,
        tgt_lang,
        _optional_document(row["doc_id"], place),
        _optional_segment(row["seg_idx"], place),
        checked_sentence(row["src"], place, "src"),
        checked_sentence(row["tgt"], place, "tgt"),
    )


def _optional_document(value: object, place: str) -> str | None:
    if value is None:
        return None
    if not (isinstance(value, str) and is_row_id(value)):
        raise refusal(place, "doc_id", _DOC_ID_RULE)
    return value


def _optional_segment(value: object, place: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise refusal(place, "seg_idx", "must be a whole number from 0 up, or null")
    return value


def _require_one_set(rows: Sequence[Mapping[str, object]]) -> None:
    first = rows[0]["set"]
    for index, row in enumerate(rows):
        if row["set"] != first:
            raise refusal(
                f"export row index {index}",
                "set",
                "is not the set of the first row: an export holds one set",
            )
