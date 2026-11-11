"""The reader of the tiny pipeline's seven fixture files (D15, spec P5).

`read_fixtures` reads the directory it is given and never the repository by itself. It reads the
bytes of every file, hashes them all and compares each digest with its pin BEFORE it parses any
file, and then parses those very bytes, so a file that changes between the check and the parse is
not a way in. Nothing here is random at run time.

A refusal is a `DataError` that names the file, the row (counted from 1: the first line after the
header is row 1) and the field, and quotes no text of a fixture: the files are templated sentences
and names, and a message that quoted a row would put them into every log.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from rinchen.errors import DataError

ENGLISH_PAIR_FILE: Final = "eng_Latn-khk_Cyrl.tsv"
RUSSIAN_PAIR_FILE: Final = "rus_Cyrl-khk_Cyrl.tsv"
MONGOLIAN_FILE: Final = "mn_Mong.tsv"
MONO_FILE: Final = "mono_khk_Cyrl.tsv"
NOISE_FILE: Final = "noise.tsv"
NOUNS_FILE: Final = "nouns.tsv"
ENTITIES_FILE: Final = "entities.tsv"
FIXTURE_FILES: Final[tuple[str, ...]] = (
    ENGLISH_PAIR_FILE,
    RUSSIAN_PAIR_FILE,
    MONGOLIAN_FILE,
    MONO_FILE,
    NOISE_FILE,
    NOUNS_FILE,
    ENTITIES_FILE,
)

SPLITS: Final = ("train", "dev", "test")
KINDS: Final = ("base", "int", "decimal", "grouped", "placeholder", "entity")
NOISE_REASONS: Final = ("script_tgt", "len_ratio_char", "numbers", "duplicate", "labse")
LANGUAGES: Final = ("khk_Cyrl", "eng_Latn", "rus_Cyrl")

_HEADERS: Final = {
    ENGLISH_PAIR_FILE: ("id", "split", "kind", "eng_Latn", "khk_Cyrl"),
    RUSSIAN_PAIR_FILE: ("id", "split", "kind", "rus_Cyrl", "khk_Cyrl"),
    MONGOLIAN_FILE: ("id", "split", "kind", "mn_Mong", "khk_Cyrl", "eng_Latn"),
    MONO_FILE: ("id", "khk_Cyrl", "eng_Latn"),
    NOISE_FILE: ("id", "expect", "src_lang", "tgt_lang", "src", "tgt"),
    NOUNS_FILE: LANGUAGES,
    ENTITIES_FILE: ("id", *LANGUAGES),
}

Cells = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairRow:
    """A row of a pair file: the source sentence and its Khalkha counterpart."""

    id: str
    split: str
    kind: str
    src: str
    tgt: str


@dataclass(frozen=True, slots=True)
class MongRow:
    """A traditional-script sentence with the Khalkha and English sentences it spells."""

    id: str
    split: str
    kind: str
    mong: str
    khk: str
    eng: str


@dataclass(frozen=True, slots=True)
class MonoRow:
    """A monolingual Khalkha line with its English oracle."""

    id: str
    khk: str
    eng: str


@dataclass(frozen=True, slots=True)
class NoiseRow:
    """A planted bad pair; `expect` is the reason the filter must give for it."""

    id: str
    expect: str
    src_lang: str
    tgt_lang: str
    src: str
    tgt: str


@dataclass(frozen=True, slots=True)
class Fixtures:
    """The seven files, parsed. `pairs` is keyed `eng_Latn-khk_Cyrl` and `rus_Cyrl-khk_Cyrl`;
    `nouns` rows are (Khalkha, English, Russian); `entities` maps an id to a language to its
    spellings, the first being the one a sentence writes."""

    pairs: Mapping[str, tuple[PairRow, ...]]
    mong: tuple[MongRow, ...]
    mono: tuple[MonoRow, ...]
    noise: tuple[NoiseRow, ...]
    nouns: tuple[tuple[str, str, str], ...]
    entities: Mapping[str, Mapping[str, tuple[str, ...]]]


def _read_bytes(directory: Path, name: str) -> bytes:
    try:
        return (directory / name).read_bytes()
    except OSError as error:
        raise DataError(f"{name}: the file is missing or cannot be read") from error


def fixture_digests(directory: Path) -> dict[str, str]:
    """The SHA-256, in hex, of each of the seven files of `directory`, by file name."""
    return {
        name: hashlib.sha256(_read_bytes(directory, name)).hexdigest()
        for name in FIXTURE_FILES
    }


def _check_pins(contents: Mapping[str, bytes], pins: Mapping[str, str]) -> None:
    unknown = len(set(pins) - set(FIXTURE_FILES))
    missing = len(set(FIXTURE_FILES) - set(pins))
    if unknown or missing:
        raise DataError(
            f"the pins name {unknown} files that are not fixtures and miss {missing}"
            f" of the {len(FIXTURE_FILES)} fixtures"
        )
    for name in FIXTURE_FILES:
        if hashlib.sha256(contents[name]).hexdigest() != pins[name]:
            raise DataError(f"{name}: the SHA-256 differs from its pin")


def _rows(name: str, data: bytes) -> list[Cells]:
    """The data rows of a file: UTF-8 lines of tab-separated fields, the header checked."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"{name}: not valid UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise DataError(
            f"{name}: lines must end with a line feed and none may hold a carriage return"
        )
    header, *lines = text[:-1].split("\n")
    expected = _HEADERS[name]
    if tuple(header.split("\t")) != expected:
        raise DataError(f"{name}: the header is not the expected one")
    rows: list[Cells] = []
    for index, line in enumerate(lines, start=1):
        cells = tuple(line.split("\t"))
        if len(cells) != len(expected):
            raise DataError(
                f"{name}: row {index} has {len(cells)} fields, expected {len(expected)}"
            )
        rows.append(cells)
    return rows


def _require(
    name: str, index: int, field: str, value: str, allowed: Sequence[str]
) -> None:
    if value not in allowed:
        raise DataError(
            f"{name}: row {index}: field {field} is not one of {', '.join(allowed)}"
        )


def _pair_rows(name: str, rows: Sequence[Cells]) -> tuple[PairRow, ...]:
    found: list[PairRow] = []
    for index, (id_, split, kind, src, tgt) in enumerate(rows, start=1):
        _require(name, index, "split", split, SPLITS)
        _require(name, index, "kind", kind, KINDS)
        found.append(PairRow(id_, split, kind, src, tgt))
    return tuple(found)


def _mong_rows(rows: Sequence[Cells]) -> tuple[MongRow, ...]:
    found: list[MongRow] = []
    for index, (id_, split, kind, mong, khk, eng) in enumerate(rows, start=1):
        _require(MONGOLIAN_FILE, index, "split", split, SPLITS)
        _require(MONGOLIAN_FILE, index, "kind", kind, KINDS)
        found.append(MongRow(id_, split, kind, mong, khk, eng))
    return tuple(found)


def _noise_rows(rows: Sequence[Cells]) -> tuple[NoiseRow, ...]:
    found: list[NoiseRow] = []
    for index, (id_, expect, src_lang, tgt_lang, src, tgt) in enumerate(rows, start=1):
        _require(NOISE_FILE, index, "expect", expect, NOISE_REASONS)
        found.append(NoiseRow(id_, expect, src_lang, tgt_lang, src, tgt))
    return tuple(found)


def _entities(rows: Sequence[Cells]) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    entities: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for index, (id_, *cells) in enumerate(rows, start=1):
        if id_ in entities:
            raise DataError(f"{ENTITIES_FILE}: row {index}: field id repeats a row")
        spellings: dict[str, tuple[str, ...]] = {}
        for language, cell in zip(LANGUAGES, cells, strict=True):
            names = tuple(cell.split("|"))
            if "" in names:
                raise DataError(
                    f"{ENTITIES_FILE}: row {index}: field {language} has an empty spelling"
                )
            spellings[language] = names
        entities[id_] = MappingProxyType(spellings)
    return MappingProxyType(entities)


def read_fixtures(directory: Path, pins: Mapping[str, str]) -> Fixtures:
    """The seven files of `directory`, parsed, after every file matched its pin.

    `pins` maps each fixture file name to its SHA-256 in hex and holds exactly the seven names.
    Raises DataError for a missing or unreadable file, a pin that is missing or belongs to no
    fixture, a digest that differs, text that is not LF-terminated UTF-8, a header that is not
    the expected one, a row of the wrong width, a `split`, `kind` or noise `expect` outside its
    closed list, an empty entity spelling and a repeated entity id.
    """
    contents = {name: _read_bytes(directory, name) for name in FIXTURE_FILES}
    _check_pins(contents, pins)
    rows = {name: _rows(name, contents[name]) for name in FIXTURE_FILES}
    return Fixtures(
        pairs=MappingProxyType(
            {
                name.removesuffix(".tsv"): _pair_rows(name, rows[name])
                for name in (ENGLISH_PAIR_FILE, RUSSIAN_PAIR_FILE)
            }
        ),
        mong=_mong_rows(rows[MONGOLIAN_FILE]),
        mono=tuple(MonoRow(*cells) for cells in rows[MONO_FILE]),
        noise=_noise_rows(rows[NOISE_FILE]),
        nouns=tuple((khk, eng, rus) for khk, eng, rus in rows[NOUNS_FILE]),
        entities=_entities(rows[ENTITIES_FILE]),
    )
