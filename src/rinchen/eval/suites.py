"""The evaluation suite of D13 (spec T2 and T3): the sentences systems are asked to translate and
are scored against, one JSON document, schema `rinchen.eval.suite/1`, UTF-8 with sorted keys,
two-space indent and one final LF, so the same suite gives the same bytes:

    {"schema", "name", "kind", "benchmark", "split",
     "source": {"dataset", "revision", "pinned", "files", "missing_languages"},
     "directions": [...], "rows": [{"id", "doc", "text": {language: sentence}}]}

`kind` is `benchmark`, `human` or `fixture` and `split` is `dev`, `devtest` or `test`; the
purpose of a suite follows from its split (`dev` is for selecting a model, `devtest` and `test`
are for reporting one). `suite_rules` says what each field may hold. The file is read ONCE
(`read_suite` returns the suite and the SHA-256 of the bytes it parsed) and is at most 64 MiB,
which is checked before anything is parsed.

The devtest guard: `Suite.selection_view` is the only function that hands rows to a selection
loop and it accepts a `dev` suite only. Code outside `rinchen.eval` and the `rinchen eval`
commands never imports this module (`tests/unit/test_eval_devtest_guard.py` reads the source).

A direction reads the rows that have both of its languages, in the order of the file (the draws
of the bootstrap are bound to positions, so the order of the rows is part of the suite).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.config.synth import supported_partner_languages
from rinchen.errors import ConfigError, DataError
from rinchen.eval.robust_suite_rules import parse_object
from rinchen.eval.suite_rules import (
    FORBIDDEN_LANGUAGES,
    HUB_LANGUAGE,
    KINDS,
    MAX_ROWS,
    MAX_SENTENCE_CHARS,
    MAX_SUITE_BYTES,
    PURPOSE_OF_SPLIT,
    SPLITS,
    SPLITS_OF_KIND,
    SUITE_SCHEMA,
    SuiteRow,
    SuiteSource,
    checked_name,
    direction_name,
    excluded_languages,
    object_at,
    read_directions,
    read_rows,
    read_source,
    refusal,
    require_rows_for_directions,
    split_direction,
)
from rinchen.utils.part_file import replace_file

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
    "DirectionView",
    "Suite",
    "SuiteRow",
    "SuiteSource",
    "read_suite",
    "standard_directions",
    "write_suite",
]

_TOP_KEYS: Final = (
    "schema",
    "name",
    "kind",
    "benchmark",
    "split",
    "source",
    "directions",
    "rows",
)
_WELL_FORMED_DIRECTION: Final = re.compile(
    r"[a-z]{3}_[A-Z][a-z]{3}-[a-z]{3}_[A-Z][a-z]{3}"
)


@dataclass(frozen=True, slots=True)
class DirectionView:
    """The rows of one direction, in file order, as parallel columns: `ids`, the `sources`
    (text in `src_lang`) and the `references` (text in `tgt_lang`). `docs` has the document of
    every row, or is None unless EVERY row of the direction has one (a document unit of the
    bootstrap exists only then)."""

    direction: str
    src_lang: str
    tgt_lang: str
    ids: tuple[str, ...]
    docs: tuple[str, ...] | None
    sources: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Suite:
    """An evaluation suite: its `name`, `kind`, `benchmark` and `split`, where it came from
    (`source`), the `directions` it declares and its `rows` in file order. It holds what it is
    given and does not validate it: `from_bytes` is the validator (`to_bytes` writes what it
    holds, so a suite built wrongly is refused when it is read back).

    A suite holds mappings (the text of a row, the files of the source), so it cannot be
    hashed."""

    name: str
    kind: str
    benchmark: str
    split: str
    source: SuiteSource
    directions: tuple[str, ...]
    rows: tuple[SuiteRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def purpose(self) -> str:
        """`selection` for a `dev` suite, `report` for `devtest` and `test`."""
        return PURPOSE_OF_SPLIT[self.split]

    def languages(self) -> dict[str, int]:
        """The number of sentences of each language, by language code."""
        counts: dict[str, int] = {}
        for row in self.rows:
            for language in row.text:
                counts[language] = counts.get(language, 0) + 1
        return dict(sorted(counts.items()))

    def view(self, direction: str) -> DirectionView:
        """The rows of `direction` that have both of its languages, in file order.

        Raises:
            ConfigError: the suite does not declare `direction`.
        """
        if direction not in self.directions:
            raise ConfigError(
                f"the suite {self.name!r} has no direction {_named(direction)}"
            )
        src, tgt = split_direction(direction)
        chosen = [row for row in self.rows if src in row.text and tgt in row.text]
        every_row_has_a_document = bool(chosen) and all(
            row.doc is not None for row in chosen
        )
        return DirectionView(
            direction=direction,
            src_lang=src,
            tgt_lang=tgt,
            ids=tuple(row.id for row in chosen),
            docs=(
                tuple(row.doc for row in chosen if row.doc is not None)
                if every_row_has_a_document
                else None
            ),
            sources=tuple(row.text[src] for row in chosen),
            references=tuple(row.text[tgt] for row in chosen),
        )

    def selection_view(self, direction: str) -> DirectionView:
        """`view`, for a loop that chooses a model: only a `dev` suite may be read this way.

        Raises:
            ConfigError: the split is not `dev` (devtest and test are for reporting only), or
                the suite does not declare `direction`.
        """
        if self.split != "dev":
            raise ConfigError(
                "only the rows of a dev suite may be read for a selection: the rows of a "
                "devtest or test suite are for reporting only, never for choosing or "
                "tuning a model"
            )
        return self.view(direction)

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of the suite file: sorted keys, two-space indent, one final LF,
        the directions and the rows in the order the suite holds them. It writes what it is
        given and does not validate."""
        document = {
            "schema": SUITE_SCHEMA,
            "name": self.name,
            "kind": self.kind,
            "benchmark": self.benchmark,
            "split": self.split,
            "source": {
                "dataset": self.source.dataset,
                "revision": self.source.revision,
                "pinned": self.source.pinned,
                "files": dict(self.source.files),
                "missing_languages": list(self.source.missing_languages),
            },
            "directions": list(self.directions),
            "rows": [
                {"id": row.id, "doc": row.doc, "text": dict(row.text)}
                for row in self.rows
            ],
        }
        text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
        return f"{text}\n".encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> Suite:
        """The suite in `data`, which must be exactly the document `to_bytes` writes in
        shape: this schema, every field and no other, each of its type.

        Raises:
            DataError: `data` is too large, not UTF-8, not JSON, or not that document; the
                message names the field, the row index and a well-formed id and never quotes
                what the file held.
        """
        top = _document(data)
        name = checked_name(top["name"], "suite", "name")
        kind = _member(top["kind"], KINDS, "kind")
        benchmark = checked_name(top["benchmark"], "suite", "benchmark")
        split = _member(top["split"], SPLITS, "split")
        if split not in SPLITS_OF_KIND[kind]:
            raise refusal(
                "suite",
                "split",
                f"a {kind} suite has the split {' or '.join(SPLITS_OF_KIND[kind])}",
            )
        excluded = excluded_languages(kind, benchmark)
        source = read_source(top["source"], excluded=excluded)
        directions = read_directions(top["directions"], excluded=excluded)
        rows = read_rows(top["rows"], excluded=excluded)
        require_rows_for_directions(directions, rows)
        return cls(
            name=name,
            kind=kind,
            benchmark=benchmark,
            split=split,
            source=source,
            directions=directions,
            rows=rows,
        )


def read_suite(path: Path) -> tuple[Suite, str]:
    """The suite in the file `path` and the SHA-256 of the bytes that were parsed. The file is
    opened once and at most one byte over the cap is read.

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
    """Write `suite` to `path` so that the file appears whole; return the SHA-256 of the bytes
    written.

    Raises:
        DataError: the suite is larger than the reader accepts.
    """
    data = suite.to_bytes()
    if len(data) > MAX_SUITE_BYTES:
        raise _too_large()
    replace_file(path, data)
    return hashlib.sha256(data).hexdigest()


def standard_directions() -> tuple[str, ...]:
    """The directions the targets expect: Khalkha to and from each partner language of the
    synthetic-data configuration, the partners sorted (16 for the registry)."""
    pairs: list[str] = []
    for partner in sorted(supported_partner_languages()):
        pairs.append(direction_name(HUB_LANGUAGE, partner))
        pairs.append(direction_name(partner, HUB_LANGUAGE))
    return tuple(pairs)


def _named(direction: str) -> str:
    """The direction for a message when it is well formed, and no quotation of it otherwise."""
    if _WELL_FORMED_DIRECTION.fullmatch(direction):
        return repr(direction)
    return "of that name (the name is not well formed)"


def _too_large() -> DataError:
    return DataError(
        f"not an evaluation suite: the file is larger than {MAX_SUITE_BYTES // 2**20} MiB"
    )


def _member(value: object, allowed: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise refusal("suite", field, f"must be one of {', '.join(allowed)}")
    return value


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
