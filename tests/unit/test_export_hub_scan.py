"""The scan of the text files of a release (D13, Task 10, spec T18): `kinds_in` says whether a text
looks like it holds a token (`hf_` and 20 letters or digits, `Bearer` and 16 token characters,
`HF_TOKEN=`) or an absolute local path (a POSIX path of two components or more at the start of a
word, under any root, or a drive path), and never says what matched. The boundaries of each pattern
are pinned here, with the texts a release really holds (the golden card and report, a vocabulary of
pieces) that must pass. The inputs that look like a token or a path are put together from pieces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from rinchen.export.hub_scan import KIND_PATH, KIND_TOKEN, kinds_in
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

BACKSLASH: Final = chr(0x5C)
SLASH: Final = chr(0x2F)
FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
HUB_PREFIX: Final = "hf" + "_"
VARIABLE: Final = "HF_" + "TOKEN"


def kinds(text: str) -> list[str]:
    return kinds_in(text.encode("utf-8"))


def posix(*parts: str) -> str:
    return SLASH + SLASH.join(parts)


# ---- a Hub token ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "length, expected", [(19, []), (20, [KIND_TOKEN]), (64, [KIND_TOKEN])]
)
def test_a_hub_token_is_hf_underscore_and_twenty_letters_or_digits(
    length: int, expected: list[str]
) -> None:
    assert kinds(f"key {HUB_PREFIX}{'aB3' * 30}"[: 4 + 3 + length]) == expected


def test_a_run_of_letters_that_is_broken_by_a_punctuation_mark_is_not_a_token() -> None:
    assert kinds(HUB_PREFIX + "a" * 19 + "-" + "a" * 19) == []


def test_a_token_is_found_inside_json_and_inside_a_word() -> None:
    text = json.dumps({"note": "x" + HUB_PREFIX + "Z" * 25 + "y"})

    assert kinds(text) == [KIND_TOKEN]


@pytest.mark.parametrize(
    "length, expected", [(15, []), (16, [KIND_TOKEN]), (40, [KIND_TOKEN])]
)
def test_a_bearer_credential_is_bearer_and_sixteen_token_characters(
    length: int, expected: list[str]
) -> None:
    assert kinds("Authorization: Bearer " + "q" * length) == expected


@pytest.mark.parametrize("word", ["bearer", "BEARER", "Bearer"])
@pytest.mark.parametrize("gap", [" ", "  ", "\t", "\n"])
def test_the_word_bearer_is_read_in_any_case_with_any_blank_after_it(
    word: str, gap: str
) -> None:
    assert kinds(f"{word}{gap}{'z' * 16}") == [KIND_TOKEN]


def test_the_characters_of_a_bearer_credential_are_those_of_an_http_credential() -> (
    None
):
    assert kinds("Bearer " + "a.b_c~d+e/f=g-h" + "9") == [KIND_TOKEN]
    assert kinds("Bearer " + "a b c d e f g h i j k l") == []


def test_the_word_bearer_in_a_sentence_is_not_a_credential() -> None:
    assert kinds("the bearer of good news, a bearer bond") == []


@pytest.mark.parametrize(
    "text", [VARIABLE + "=x", VARIABLE + " = x", "export " + VARIABLE + "=y"]
)
def test_the_assignment_of_the_token_variable_is_a_token(text: str) -> None:
    assert kinds(text) == [KIND_TOKEN]


def test_the_name_of_the_token_variable_alone_is_not_an_assignment() -> None:
    assert kinds(f"set {VARIABLE} in your environment") == []


# ---- an absolute local path ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lead",
    [" ", "\n", "\t", '"', "'", "`", "(", "[", "{", "<", "=", ",", ";", ":", "|", ""],
)
def test_a_path_is_found_after_each_character_that_can_start_a_word(lead: str) -> None:
    assert kinds(lead + posix("srv", "data", "x")) == [KIND_PATH]


@pytest.mark.parametrize(
    "root", ["home", "tmp", "var", "root", "Users", "mnt", "opt", "usr", "srv", "data"]
)
def test_a_path_is_found_under_any_root(root: str) -> None:
    assert kinds("see " + posix(root, "x")) == [KIND_PATH]


@pytest.mark.parametrize("name", ["a.b", "a_b", "a-b", "~a", "A1"])
def test_the_components_of_a_path_may_hold_dots_dashes_and_tildes(name: str) -> None:
    assert kinds("see " + posix(name, name)) == [KIND_PATH]


@pytest.mark.parametrize(
    "text",
    [
        "and/or",
        "12/16",
        "org/name",
        "path/to/this/directory",
        "see https://huggingface.co/org/name for it",
        "an address http://host/a/b",
        "a root alone: " + posix("usr") + " and " + posix("tmp"),
        "a slash: / and a pair: //",
        "x" + posix("a", "b"),
        "1" + posix("a", "b"),
        "2026/11/10",
        "no path in here at all",
    ],
)
def test_a_slash_inside_a_word_an_address_or_a_single_component_is_not_a_local_path(
    text: str,
) -> None:
    assert kinds(text) == []


@pytest.mark.parametrize("letter", ["C", "c", "Z", "d"])
def test_a_drive_path_is_a_letter_a_colon_and_a_backslash(letter: str) -> None:
    assert kinds(f"in {letter}:{BACKSLASH}Users") == [KIND_PATH]
    assert kinds(f"{letter}:{BACKSLASH}") == [KIND_PATH]


def test_a_drive_path_is_found_as_a_json_file_writes_it() -> None:
    text = json.dumps({"path": "C:" + BACKSLASH + "Users" + BACKSLASH + "x"})

    assert kinds(text) == [KIND_PATH]


@pytest.mark.parametrize(
    "text", ["ab:" + BACKSLASH + "x", "1C:" + BACKSLASH + "x", "C:x", "C: " + BACKSLASH]
)
def test_a_colon_after_a_word_is_not_a_drive(text: str) -> None:
    assert kinds(text) == []


def test_a_drive_with_a_forward_slash_is_read_as_the_posix_path_it_is() -> None:
    assert kinds("C:" + posix("Users", "x")) == [KIND_PATH]
    assert kinds("C:" + SLASH) == []


# ---- both kinds, any text ------------------------------------------------------------------------------------------


def test_a_text_with_both_says_both_once_and_the_token_first() -> None:
    text = f"{HUB_PREFIX}{'a' * 30} and {posix('srv', 'data', 'x')} and {posix('srv', 'y', 'z')}"

    assert kinds(text) == [KIND_TOKEN, KIND_PATH]


def test_a_text_that_is_not_utf_8_is_scanned_with_its_bad_bytes_replaced() -> None:
    data = b"\xff\xfe " + (HUB_PREFIX + "a" * 25).encode() + b" \xff"

    assert kinds_in(data) == [KIND_TOKEN]
    assert kinds_in(b"\xff\xfe\x00 plain bytes") == []


def test_a_byte_that_is_not_utf_8_ends_a_word_it_does_not_join_the_two_halves() -> None:
    data = (HUB_PREFIX + "a" * 10).encode() + b"\xff" + b"a" * 10

    assert kinds_in(data) == []
    assert kinds_in(data.replace(b"\xff", b"a")) == [KIND_TOKEN]


def test_the_scan_never_returns_what_matched() -> None:
    text = f"{HUB_PREFIX}{'m' * 30} {posix('srv', 'secretplace', 'x')}"

    found = kinds(text)

    assert all("m" * 5 not in kind and "secretplace" not in kind for kind in found)
    assert set(found) == {KIND_TOKEN, KIND_PATH}


# ---- what a release really holds -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["golden_card.md", "golden_report.md", "golden_report.json"]
)
def test_the_golden_card_and_report_hold_nothing_the_scan_finds(name: str) -> None:
    assert kinds_in((FIXTURES / name).read_bytes()) == []


def test_a_vocabulary_of_pieces_holds_nothing_the_scan_finds() -> None:
    lower_one_eighth_block = chr(0x2581)
    pieces = [
        lower_one_eighth_block + SLASH,
        SLASH + "a",
        SLASH + "or",
        "://",
        "s:" + SLASH,
        lower_one_eighth_block + HUB_PREFIX,
        BACKSLASH,
        BACKSLASH * 2,
        VARIABLE,
        "bearer",
        SLASH,
    ]

    assert kinds(json.dumps(pieces)) == []
