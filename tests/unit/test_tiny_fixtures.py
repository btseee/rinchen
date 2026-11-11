"""`rinchen.tiny.fixtures`: the reader of the tiny pipeline's seven fixture files (D15, spec P5).

The reader reads the directory it is given and never the repository by itself, hashes the bytes of
every file before it parses any, parses those very bytes, and refuses with a `DataError` that names
the file, the row and the field and quotes no text. The tests plant each defect in a copy of the
committed fixtures with the pins of the planted bytes (so the defect, not the digest, is what the
reader finds), and a recorder asserts that no message holds a text of a fixture or a planted value.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import shutil
import sys
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError
from rinchen.tiny import fixtures as fixtures_module
from rinchen.tiny.fixtures import (
    FIXTURE_FILES,
    Fixtures,
    MongRow,
    MonoRow,
    NoiseRow,
    PairRow,
    fixture_digests,
    read_fixtures,
)

FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
PACKAGE: Final = REPO_ROOT / "src" / "rinchen" / "tiny"

ENGLISH_PAIR: Final = "eng_Latn-khk_Cyrl.tsv"
RUSSIAN_PAIR: Final = "rus_Cyrl-khk_Cyrl.tsv"
MONGOLIAN: Final = "mn_Mong.tsv"
MONO: Final = "mono_khk_Cyrl.tsv"
NOISE: Final = "noise.tsv"
NOUNS: Final = "nouns.tsv"
ENTITIES: Final = "entities.tsv"
SENTINEL: Final = "ZZ-planted-value"
LANGUAGES: Final = ("khk_Cyrl", "eng_Latn", "rus_Cyrl")
# The columns of each file that hold sentences or names (the others hold ids and closed values).
TEXT_COLUMNS: Final = {
    ENGLISH_PAIR: (3, 4),
    RUSSIAN_PAIR: (3, 4),
    MONGOLIAN: (3, 4, 5),
    MONO: (1, 2),
    NOISE: (4, 5),
    NOUNS: (0, 1, 2),
    ENTITIES: (1, 2, 3),
}


def csv_rows(directory: Path, name: str) -> list[list[str]]:
    """The data rows of a file, parsed by the `csv` module (the check on the reader)."""
    with (directory / name).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True)
        return list(reader)[1:]


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    directory = tmp_path / "parallel"
    shutil.copytree(FIXTURES, directory)
    return directory


@pytest.fixture(scope="module")
def pins() -> dict[str, str]:
    return fixture_digests(FIXTURES)


def refusal(directory: Path, given: Mapping[str, str]) -> str:
    with pytest.raises(DataError) as error:
        read_fixtures(directory, given)
    return str(error.value)


@cache
def fixture_texts() -> frozenset[str]:
    """Every sentence and name of the committed files, of eight characters or more."""
    return frozenset(
        row[column]
        for name, columns in TEXT_COLUMNS.items()
        for row in csv_rows(FIXTURES, name)
        for column in columns
        if len(row[column]) >= 8
    )


def assert_names_no_text(message: str) -> None:
    """The recorder: a message holds no text of a fixture and no value the test planted."""
    leaked = [text for text in fixture_texts() if text in message]
    assert leaked == [], "a message quotes fixture text"
    assert SENTINEL not in message
    assert "holdout" not in message
    assert " qe" not in message


def rewrite(
    directory: Path, name: str, transform: Callable[[list[str]], None]
) -> dict[str, str]:
    """Apply `transform` to the lines of a file (the header is `lines[0]`, row `n` is
    `lines[n]`, the last item is the empty string after the final line feed) and return the pins
    of the changed directory."""
    path = directory / name
    lines = path.read_text(encoding="utf-8").split("\n")
    transform(lines)
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return fixture_digests(directory)


def set_cell(row: int, column: int, value: str) -> Callable[[list[str]], None]:
    def transform(lines: list[str]) -> None:
        cells = lines[row].split("\t")
        cells[column] = value
        lines[row] = "\t".join(cells)

    return transform


# ---- the names and the digests ------------------------------------------------------------


def test_the_fixture_files_are_the_seven_of_the_spec() -> None:
    assert FIXTURE_FILES == (
        ENGLISH_PAIR,
        RUSSIAN_PAIR,
        MONGOLIAN,
        MONO,
        NOISE,
        NOUNS,
        ENTITIES,
    )


def test_the_digests_are_the_sha256_of_each_file(pins: dict[str, str]) -> None:
    assert list(pins) == list(FIXTURE_FILES)
    for name, digest in pins.items():
        assert digest == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


def test_the_digests_of_a_directory_with_a_missing_file_name_the_file(
    staged: Path,
) -> None:
    (staged / NOISE).unlink()

    with pytest.raises(DataError, match=NOISE):
        fixture_digests(staged)


# ---- the committed directory --------------------------------------------------------------


def test_the_committed_directory_reads_into_the_counts_of_the_spec(
    pins: dict[str, str],
) -> None:
    read = read_fixtures(FIXTURES, pins)

    assert list(read.pairs) == ["eng_Latn-khk_Cyrl", "rus_Cyrl-khk_Cyrl"]
    assert [len(rows) for rows in read.pairs.values()] == [345, 345]
    assert len(read.mong) == 70
    assert len(read.mono) == 64
    assert len(read.noise) == 10
    assert len(read.nouns) == 37
    assert len(read.entities) == 20
    assert all(isinstance(row, PairRow) for rows in read.pairs.values() for row in rows)
    assert all(isinstance(row, MongRow) for row in read.mong)
    assert all(isinstance(row, MonoRow) for row in read.mono)
    assert all(isinstance(row, NoiseRow) for row in read.noise)


def test_the_reader_agrees_with_an_independent_csv_parse(pins: dict[str, str]) -> None:
    read = read_fixtures(FIXTURES, pins)

    for name, rows in zip(
        (ENGLISH_PAIR, RUSSIAN_PAIR), read.pairs.values(), strict=True
    ):
        assert [dataclasses.astuple(row) for row in rows] == [
            tuple(cells) for cells in csv_rows(FIXTURES, name)
        ]
    for name, rows in ((MONGOLIAN, read.mong), (MONO, read.mono), (NOISE, read.noise)):
        assert [dataclasses.astuple(row) for row in rows] == [
            tuple(cells) for cells in csv_rows(FIXTURES, name)
        ]
    assert list(read.nouns) == [tuple(cells) for cells in csv_rows(FIXTURES, NOUNS)]
    expected = {
        cells[0]: {
            language: tuple(cell.split("|"))
            for language, cell in zip(LANGUAGES, cells[1:], strict=True)
        }
        for cells in csv_rows(FIXTURES, ENTITIES)
    }
    assert {key: dict(value) for key, value in read.entities.items()} == expected
    assert list(read.entities) == list(expected)


def test_a_row_carries_its_columns_by_name(pins: dict[str, str]) -> None:
    read = read_fixtures(FIXTURES, pins)

    first = read.pairs["eng_Latn-khk_Cyrl"][0]
    assert (first.id, first.split, first.kind) == ("ek-0001", "train", "base")
    assert csv_rows(FIXTURES, ENGLISH_PAIR)[0][3:5] == [first.src, first.tgt]
    assert [read.mong[0].mong, read.mong[0].khk, read.mong[0].eng] == (
        csv_rows(FIXTURES, MONGOLIAN)[0][3:6]
    )
    assert read.noise[6].expect == "duplicate"
    assert read.entities["ulaanbaatar"]["eng_Latn"] == ("Ulaanbaatar", "Ulan Bator")


def test_the_result_cannot_be_changed(pins: dict[str, str]) -> None:
    read = read_fixtures(FIXTURES, pins)

    with pytest.raises(dataclasses.FrozenInstanceError):
        read.mong = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        read.pairs["eng_Latn-khk_Cyrl"][0].id = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        read.pairs["other"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        read.entities["other"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        read.entities["bat"]["other"] = ()  # type: ignore[index]


def test_it_reads_the_directory_it_is_given_and_nothing_else(
    staged: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rewrite(staged, NOUNS, lambda lines: lines.pop(-2))
    monkeypatch.chdir(tmp_path)

    read = read_fixtures(staged, fixture_digests(staged))

    assert len(read.nouns) == 36


def test_the_bytes_that_were_hashed_are_the_bytes_that_are_parsed(
    monkeypatch: pytest.MonkeyPatch, pins: dict[str, str]
) -> None:
    reads: list[str] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self.name)
        return original(self)

    def forbidden(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError("a second read of a fixture file")

    monkeypatch.setattr(Path, "read_bytes", counting)
    monkeypatch.setattr(Path, "read_text", forbidden)

    read_fixtures(FIXTURES, pins)

    assert sorted(reads) == sorted(FIXTURE_FILES)


# ---- refusals: the digest and the set of files --------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_flipped_byte_is_refused_naming_the_file(
    staged: Path, pins: dict[str, str], name: str
) -> None:
    data = bytearray((staged / name).read_bytes())
    data[len(data) // 2] ^= 0x01
    (staged / name).write_bytes(bytes(data))

    message = refusal(staged, pins)

    assert name in message
    assert_names_no_text(message)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_missing_file_is_refused_naming_the_file(
    staged: Path, pins: dict[str, str], name: str
) -> None:
    (staged / name).unlink()

    message = refusal(staged, pins)

    assert name in message
    assert_names_no_text(message)


def test_a_directory_that_does_not_exist_is_refused(
    tmp_path: Path, pins: dict[str, str]
) -> None:
    assert FIXTURE_FILES[0] in refusal(tmp_path / "nowhere", pins)


def test_a_pin_that_is_missing_or_for_another_file_is_refused_with_counts(
    pins: dict[str, str],
) -> None:
    short = {name: digest for name, digest in pins.items() if name != NOISE}
    extra = {**pins, "other.tsv": pins[NOISE]}

    assert refusal(FIXTURES, short) == (
        "the pins name 0 files that are not fixtures and miss 1 of the 7 fixtures"
    )
    assert refusal(FIXTURES, extra) == (
        "the pins name 1 files that are not fixtures and miss 0 of the 7 fixtures"
    )


def test_the_digests_of_every_file_are_compared_before_any_file_is_parsed(
    staged: Path,
) -> None:
    rewrite(staged, ENGLISH_PAIR, lambda lines: lines.append("not a row"))
    given = fixture_digests(staged)
    given[ENTITIES] = "0" * 64

    message = refusal(staged, given)

    assert ENTITIES in message
    assert "digest" in message or "SHA-256" in message
    assert ENGLISH_PAIR not in message


# ---- refusals: the shape of a file ------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_row_with_one_field_too_many_is_refused_naming_file_and_row(
    staged: Path, name: str
) -> None:
    given = rewrite(
        staged, name, lambda lines: lines.__setitem__(3, lines[3] + "\t" + SENTINEL)
    )

    message = refusal(staged, given)

    assert name in message and "row 3" in message
    assert_names_no_text(message)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_row_with_one_field_too_few_is_refused_naming_file_and_row(
    staged: Path, name: str
) -> None:
    def drop_last_field(lines: list[str]) -> None:
        lines[2] = lines[2].rsplit("\t", 1)[0]

    given = rewrite(staged, name, drop_last_field)

    message = refusal(staged, given)

    assert name in message and "row 2" in message
    assert_names_no_text(message)


@pytest.mark.parametrize(
    ("name", "row", "column", "value", "field"),
    [
        (ENGLISH_PAIR, 5, 1, "holdout", "split"),
        (RUSSIAN_PAIR, 7, 1, "holdout", "split"),
        (MONGOLIAN, 2, 1, "holdout", "split"),
        (ENGLISH_PAIR, 9, 2, "other", "kind"),
        (RUSSIAN_PAIR, 4, 2, "other", "kind"),
        (MONGOLIAN, 3, 2, "other", "kind"),
        (NOISE, 4, 1, "qe", "expect"),
    ],
)
def test_a_value_outside_its_closed_list_is_refused_naming_file_row_and_field(
    staged: Path, name: str, row: int, column: int, value: str, field: str
) -> None:
    given = rewrite(staged, name, set_cell(row, column, value))

    message = refusal(staged, given)

    assert name in message and f"row {row}" in message and field in message
    assert_names_no_text(message)


def test_every_value_of_the_closed_lists_is_accepted(staged: Path) -> None:
    for split in ("train", "dev", "test"):
        given = rewrite(staged, ENGLISH_PAIR, set_cell(5, 1, split))
        read_fixtures(staged, given)
    for kind in ("base", "int", "decimal", "grouped", "placeholder", "entity"):
        given = rewrite(staged, RUSSIAN_PAIR, set_cell(5, 2, kind))
        read_fixtures(staged, given)
    for reason in ("script_tgt", "len_ratio_char", "numbers", "duplicate", "labse"):
        given = rewrite(staged, NOISE, set_cell(5, 1, reason))
        read_fixtures(staged, given)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_header_that_is_not_the_expected_one_is_refused(
    staged: Path, name: str
) -> None:
    given = rewrite(
        staged, name, lambda lines: lines.__setitem__(0, lines[0] + "\t" + SENTINEL)
    )

    message = refusal(staged, given)

    assert name in message and "header" in message
    assert_names_no_text(message)


def test_an_empty_spelling_in_the_entity_table_is_refused(staged: Path) -> None:
    given = rewrite(staged, ENTITIES, set_cell(2, 2, "Saraa|"))

    message = refusal(staged, given)

    assert ENTITIES in message and "row 2" in message and "empty" in message
    assert_names_no_text(message)


def test_a_repeated_entity_id_is_refused_instead_of_overwritten(staged: Path) -> None:
    given = rewrite(staged, ENTITIES, set_cell(3, 0, "bat"))

    message = refusal(staged, given)

    assert ENTITIES in message and "row 3" in message and "field id" in message
    assert_names_no_text(message)


@pytest.mark.parametrize(
    ("damage", "phrase"),
    [
        pytest.param(lambda data: data + b"\xff", "UTF-8", id="not utf-8"),
        pytest.param(
            lambda data: data.replace(b"\n", b"\r\n", 1),
            "line feed",
            id="carriage return",
        ),
        pytest.param(lambda data: data[:-1], "line feed", id="no final line feed"),
        pytest.param(lambda data: b"", "line feed", id="empty file"),
    ],
)
def test_text_that_is_not_lf_terminated_utf8_lines_is_refused(
    staged: Path, damage: Callable[[bytes], bytes], phrase: str
) -> None:
    (staged / NOUNS).write_bytes(damage((staged / NOUNS).read_bytes()))

    message = refusal(staged, fixture_digests(staged))

    assert NOUNS in message and phrase in message
    assert_names_no_text(message)


# ---- layering ----------------------------------------------------------------------------


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_the_reader_imports_the_standard_library_and_the_errors_only() -> None:
    foreign = {
        module
        for module in imported_modules(Path(fixtures_module.__file__))
        if module.split(".")[0] not in sys.stdlib_module_names
    }

    assert foreign == {"rinchen.errors"}


def test_no_module_of_the_package_imports_the_tests_the_scripts_or_the_generator() -> (
    None
):
    for path in sorted(PACKAGE.glob("*.py")):
        roots = {module.split(".")[0] for module in imported_modules(path)}
        assert not roots & {"tests", "scripts", "build_parallel_fixtures"}, path.name


def test_the_result_type_is_the_frozen_dataclass_of_the_spec() -> None:
    assert dataclasses.is_dataclass(Fixtures)
    assert [field.name for field in dataclasses.fields(Fixtures)] == [
        "pairs",
        "mong",
        "mono",
        "noise",
        "nouns",
        "entities",
    ]
