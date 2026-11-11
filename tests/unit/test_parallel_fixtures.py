"""The committed fixtures of the tiny pipeline (D15, spec P5) and their generator.

`scripts/build_parallel_fixtures.py` writes seven tab-separated files; the files under
`tests/fixtures/parallel/` are its output, byte for byte, and the whole set is pinned by a digest
(a new digest is a decision about the fixtures, never an update to make a test pass). The other
tests state what the files hold: their composition, that no sentence leaks from one split or one
file into another, which scripts and number formats each language uses, and that none of them
holds a word or name the project's content rules exclude.

This module writes no Khalkha and no Mongolian-script text: characters a reader could mistake for
another are spelled with `chr`.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Final

import pytest
from build_parallel_fixtures import build, main

from rinchen.config.loader import REPO_ROOT
from rinchen.script_conv.rules import transliterate
from rinchen.tiny.fixtures import FIXTURE_FILES
from tests.support.tiny_scans import (
    EM_DASH,
    PLACEHOLDER_CLOSE,
    PLACEHOLDER_OPEN,
    character_classes,
    excluded_names,
    excluded_words,
    hits,
)

FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
SCRIPT: Final = REPO_ROOT / "scripts" / "build_parallel_fixtures.py"

ENGLISH_PAIR: Final = "eng_Latn-khk_Cyrl.tsv"
RUSSIAN_PAIR: Final = "rus_Cyrl-khk_Cyrl.tsv"
MONGOLIAN: Final = "mn_Mong.tsv"
MONO: Final = "mono_khk_Cyrl.tsv"
NOISE: Final = "noise.tsv"
NOUNS: Final = "nouns.tsv"
ENTITIES: Final = "entities.tsv"

MONGOLIAN_FULL_STOP: Final = chr(0x1803)

# The digest of every file (its first 16 hex digits) and of the set: the SHA-256 of the lines
# `<name> <digest>` of the seven files in the order of their names. Changing a fixture changes
# these on purpose and is a re-measurement of the pipeline's thresholds, not an update.
FILE_DIGESTS: Final = {
    ENGLISH_PAIR: "7d1d15d7b7fc8e6e",
    RUSSIAN_PAIR: "dce119b354e90781",
    MONGOLIAN: "16000ecdd3c09917",
    MONO: "79f488e2ffe757dd",
    NOISE: "fa16dba1968d182f",
    NOUNS: "62fe9458a7d8592c",
    ENTITIES: "6cb6bb4ec68a042f",
}
SET_DIGEST: Final = "fd0ec4d812df84ccab0662487745c23b3382d877cff8e9f3c423dbe768c88e73"
TOTAL_BYTES: Final = 67_104
SIZE_BOUND: Final = 2 * 1024 * 1024

KINDS: Final = ("base", "int", "decimal", "grouped", "placeholder", "entity")
KINDS_PER_SPLIT: Final = {
    "train": dict(zip(KINDS, (150, 30, 18, 12, 60, 30), strict=True)),
    "dev": dict(zip(KINDS, (8, 2, 1, 1, 2, 1), strict=True)),
    "test": dict(zip(KINDS, (15, 3, 2, 2, 5, 3), strict=True)),
}
NOISE_REASONS: Final = ("script_tgt", "len_ratio_char", "numbers", "duplicate", "labse")
PERSON_IDS: Final = (
    *("bat", "saraa", "dorj", "tuya", "naran"),
    *("bold", "oyun", "ganaa", "munkh", "enkh"),
)
CITY_IDS: Final = (
    *("ulaanbaatar", "darkhan", "erdenet", "khovd", "choibalsan"),
    *("murun", "bayankhongor", "sukhbaatar", "zuunmod", "uliastai"),
)


def table(name: str) -> list[list[str]]:
    """The data rows of a committed file, read without the library under test."""
    lines = (FIXTURES / name).read_text(encoding="utf-8").split("\n")
    assert lines.pop() == "", name
    return [line.split("\t") for line in lines[1:]]


def header(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").split("\n")[0].split("\t")


def letter_scripts(text: str) -> set[str]:
    """The scripts of the letters of `text`: the first word of each letter's Unicode name."""
    return {unicodedata.name(char).split()[0] for char in text if char.isalpha()}


def script_environment(hash_seed: str) -> dict[str, str]:
    secret = ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")
    return {
        **{k: v for k, v in os.environ.items() if not k.endswith(secret)},
        "PYTHONHASHSEED": hash_seed,
        "CUDA_VISIBLE_DEVICES": "",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


# ---- G1.1: the generator's bytes ----------------------------------------------------------


def test_the_generator_writes_the_committed_files_byte_for_byte_twice(
    tmp_path: Path,
) -> None:
    build(tmp_path / "first")
    build(tmp_path / "second")

    for directory in (tmp_path / "first", tmp_path / "second"):
        assert sorted(path.name for path in directory.iterdir()) == sorted(
            FIXTURE_FILES
        )
        for name in FIXTURE_FILES:
            assert (directory / name).read_bytes() == (FIXTURES / name).read_bytes(), (
                name
            )


@pytest.mark.subprocess
@pytest.mark.parametrize("hash_seed", ["0", "3"])
def test_the_script_writes_the_same_bytes_under_another_hash_seed(
    tmp_path: Path, hash_seed: str
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "out")],
        env=script_environment(hash_seed),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for name in FIXTURE_FILES:
        assert (tmp_path / "out" / name).read_bytes() == (
            (FIXTURES / name).read_bytes()
        ), name


def test_the_script_asks_for_exactly_one_directory(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)  # a script that wrongly built would write here

    assert main([]) == 2
    assert main(["one", "two"]) == 2

    assert "usage" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_the_generator_imports_the_standard_library_and_the_transliteration_only() -> (
    None
):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [(node.module or "", alias.name) for alias in node.names]

    foreign = [
        pair
        for pair in imported
        if pair[0].split(".")[0] not in sys.stdlib_module_names
    ]
    assert foreign == [("rinchen.script_conv.rules", "transliterate")]


# ---- G1.2: the set's digest ---------------------------------------------------------------


def set_digest(directory: Path) -> str:
    lines = "".join(
        f"{name} {hashlib.sha256((directory / name).read_bytes()).hexdigest()}\n"
        for name in sorted(FIXTURE_FILES)
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_the_digest_of_the_set_is_the_pinned_literal() -> None:
    assert set_digest(FIXTURES) == SET_DIGEST, (
        "a new digest is a decision about the fixtures, never an update"
    )


@pytest.mark.parametrize("name", sorted(FILE_DIGESTS))
def test_each_file_has_its_pinned_digest(name: str) -> None:
    digest = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()

    assert digest.startswith(FILE_DIGESTS[name])


def test_the_directory_holds_the_seven_files_and_nothing_else() -> None:
    assert sorted(path.name for path in FIXTURES.iterdir()) == sorted(FIXTURE_FILES)
    assert set(FILE_DIGESTS) == set(FIXTURE_FILES)


def test_every_file_is_far_under_two_megabytes_and_the_set_has_its_size() -> None:
    sizes = {name: (FIXTURES / name).stat().st_size for name in FIXTURE_FILES}

    assert all(size < SIZE_BOUND for size in sizes.values())
    assert sum(sizes.values()) == TOTAL_BYTES


# ---- G1.3: the composition ----------------------------------------------------------------


@pytest.mark.parametrize("name", [ENGLISH_PAIR, RUSSIAN_PAIR])
def test_a_pair_file_has_345_rows_with_the_kinds_of_each_split(name: str) -> None:
    rows = table(name)

    assert Counter(row[1] for row in rows) == {"train": 300, "dev": 15, "test": 30}
    for split, kinds in KINDS_PER_SPLIT.items():
        found = Counter(row[2] for row in rows if row[1] == split)
        assert found == kinds, split
    assert len(rows) == 345


def test_the_pair_files_say_which_sentence_is_in_which_column() -> None:
    assert header(ENGLISH_PAIR) == ["id", "split", "kind", "eng_Latn", "khk_Cyrl"]
    assert header(RUSSIAN_PAIR) == ["id", "split", "kind", "rus_Cyrl", "khk_Cyrl"]


def test_the_other_files_have_their_rows() -> None:
    mongolian = table(MONGOLIAN)
    noise = table(NOISE)

    assert len(mongolian) == 70
    assert Counter(row[1] for row in mongolian) == {"train": 50, "dev": 10, "test": 10}
    assert len(table(MONO)) == 64
    assert len(noise) == 10
    assert Counter(row[1] for row in noise) == dict.fromkeys(NOISE_REASONS, 2)
    assert len(table(NOUNS)) == 37
    assert [row[0] for row in table(ENTITIES)] == [*PERSON_IDS, *CITY_IDS]


# ---- G1.4: uniqueness and splits ----------------------------------------------------------


def test_ids_are_unique_and_count_up_from_one_in_every_file() -> None:
    for name, prefix in (
        (ENGLISH_PAIR, "ek"),
        (RUSSIAN_PAIR, "rk"),
        (MONGOLIAN, "mm"),
        (MONO, "mo"),
        (NOISE, "no"),
    ):
        ids = [row[0] for row in table(name)]
        assert ids == [f"{prefix}-{n:04d}" for n in range(1, len(ids) + 1)], name


def authentic_sentences() -> list[tuple[str, str]]:
    """`(split, sentence)` of every sentence of the pair and monolingual files, once per row
    and column. The monolingual file has no split."""
    found = [
        (row[1], text)
        for name in (ENGLISH_PAIR, RUSSIAN_PAIR)
        for row in table(name)
        for text in row[3:5]
    ]
    return found + [("mono", text) for row in table(MONO) for text in row[1:3]]


def test_no_sentence_is_in_two_rows_of_the_pair_and_monolingual_files() -> None:
    counts = Counter(text for _, text in authentic_sentences())

    assert [text for text, count in counts.items() if count > 1] == []


def test_a_mongolian_row_copies_one_row_of_the_english_pair_in_its_own_split() -> None:
    english = {(row[1], row[2], row[4], row[3]) for row in table(ENGLISH_PAIR)}
    copied = [(row[1], row[2], row[4], row[5]) for row in table(MONGOLIAN)]

    assert set(copied) <= english
    assert len(set(copied)) == len(copied)


def test_no_mongolian_script_text_is_in_two_rows_or_in_another_file() -> None:
    texts = [row[3] for row in table(MONGOLIAN)]
    elsewhere = {text for _, text in authentic_sentences()}

    assert len(set(texts)) == len(texts)
    assert not set(texts) & elsewhere


def test_the_duplicate_noise_rows_repeat_train_rows_and_say_so() -> None:
    train = {(row[3], row[4]) for row in table(ENGLISH_PAIR) if row[1] == "train"}
    duplicates = [row for row in table(NOISE) if row[1] == "duplicate"]

    assert len(duplicates) == 2
    assert all((row[4], row[5]) in train for row in duplicates)
    assert [row for row in table(NOISE) if (row[4], row[5]) in train] == duplicates


# A planted pair is made of authentic sentences: the wrong-script, length-ratio and
# misaligned rows reuse sentences of TRAIN rows (never of dev, test or monolingual rows), and the
# duplicate rows repeat whole train rows. Every planted row is split `holdout` in the corpus.
def test_a_planted_pair_is_made_of_sentences_that_no_dev_test_or_monolingual_row_holds() -> (
    None
):
    held_back = {text for split, text in authentic_sentences() if split != "train"}

    planted = {text for row in table(NOISE) for text in row[4:6]}

    assert not planted & held_back


# ---- G1.5: scripts, numbers and placeholders ----------------------------------------------


def khalkha_texts() -> list[str]:
    """Every Khalkha text of the committed files, but the planted wrong-script ones."""
    texts = [row[4] for name in (ENGLISH_PAIR, RUSSIAN_PAIR) for row in table(name)]
    texts += [row[4] for row in table(MONGOLIAN)] + [row[1] for row in table(MONO)]
    return texts + [row[5] for row in table(NOISE) if row[1] != "script_tgt"]


def test_every_khalkha_text_is_cyrillic_and_has_no_latin_letter() -> None:
    for text in khalkha_texts():
        assert letter_scripts(text) == {"CYRILLIC"}, "a Khalkha text is not Cyrillic"


def test_the_two_script_tgt_noise_rows_have_english_in_the_khalkha_column() -> None:
    wrong_script = [row for row in table(NOISE) if row[1] == "script_tgt"]

    assert len(wrong_script) == 2
    assert all(letter_scripts(row[5]) == {"LATIN"} for row in wrong_script)


def test_every_english_text_is_latin_and_every_russian_text_cyrillic() -> None:
    english = [row[3] for row in table(ENGLISH_PAIR)] + [
        row[5] for row in table(MONGOLIAN)
    ]
    english += [row[2] for row in table(MONO)] + [row[4] for row in table(NOISE)]
    russian = [row[3] for row in table(RUSSIAN_PAIR)]

    assert all(letter_scripts(text) == {"LATIN"} for text in english)
    assert all(letter_scripts(text) == {"CYRILLIC"} for text in russian)


def test_the_noise_rows_name_their_languages() -> None:
    rows = table(NOISE)

    assert {(row[2], row[3]) for row in rows} == {("eng_Latn", "khk_Cyrl")}


PLACEHOLDER: Final = re.compile(f"{PLACEHOLDER_OPEN}([0-9]+){PLACEHOLDER_CLOSE}")


def test_a_mongolian_script_text_holds_mongolian_letters_and_its_punctuation() -> None:
    for row in table(MONGOLIAN):
        text = row[3]
        bare = PLACEHOLDER.sub("", text)
        assert letter_scripts(bare) == {"MONGOLIAN"}, row[0]
        assert all(char in " ?" or 0x1800 <= ord(char) <= 0x18AF for char in bare), row[
            0
        ]
        assert text == text.strip() and "  " not in text, row[0]
        assert text[-1] in (MONGOLIAN_FULL_STOP, "?"), row[0]


def test_every_mongolian_script_row_is_the_transliteration_of_its_khalkha_text() -> (
    None
):
    for row in table(MONGOLIAN):
        assert row[3] == transliterate(row[4]), row[0]


def test_every_text_is_nfc_and_has_no_leading_or_trailing_space() -> None:
    for name in FIXTURE_FILES:
        text = (FIXTURES / name).read_text(encoding="utf-8")
        assert unicodedata.is_normalized("NFC", text), name
        assert all(
            cell == cell.strip()
            for line in text.split("\n")
            for cell in line.split("\t")
        ), name


def test_the_placeholders_are_exactly_zero_to_nine_and_every_one_is_trained_often() -> (
    None
):
    everything = "".join(
        (FIXTURES / name).read_text(encoding="utf-8") for name in FIXTURE_FILES
    )
    indices = PLACEHOLDER.findall(everything)
    assert len(indices) == everything.count(PLACEHOLDER_OPEN)
    assert len(indices) == everything.count(PLACEHOLDER_CLOSE)
    assert set(indices) == {str(n) for n in range(10)}

    per_file = [
        Counter(
            index
            for row in table(name)
            if row[1] == "train"
            for index in PLACEHOLDER.findall(row[4])
        )
        for name in (ENGLISH_PAIR, RUSSIAN_PAIR)
    ]
    together = per_file[0] + per_file[1]
    assert min(together[str(n)] for n in range(10)) >= 20
    assert all(counter[str(n)] >= 1 for counter in per_file for n in range(10))


NUMBER: Final = {
    # language -> kind -> (the sentence around the number, the number's own shape)
    "eng_Latn": {
        "int": r"The \w+ is ([0-9]+)\.",
        "decimal": r"The \w+ is ([0-9]+\.[0-9]+)\.",
        "grouped": r"The \w+ is ([0-9]{1,3}(?:,[0-9]{3})+)\.",
    },
    "khk_Cyrl": {
        "int": r"\S+ ([0-9]+) \S+\.",
        "decimal": r"\S+ ([0-9]+,[0-9]+) \S+\.",
        "grouped": r"\S+ ([0-9]{1,3}(?: [0-9]{3})+) \S+\.",
    },
    "rus_Cyrl": {
        "int": rf"\S+ {EM_DASH} ([0-9]+)\.",
        "decimal": rf"\S+ {EM_DASH} ([0-9]+,[0-9]+)\.",
        "grouped": rf"\S+ {EM_DASH} ([0-9]{{1,3}}(?: [0-9]{{3}})+)\.",
    },
}


def number_value(language: str, kind: str, text: str) -> str:
    """The number of a sentence in the language's own format, as `1250.5` would be written."""
    match = re.fullmatch(NUMBER[language][kind], text)
    assert match, f"a {kind} sentence of {language} does not follow its format"
    value = match.group(1)
    return (
        value.replace(",", "")
        if language == "eng_Latn"
        else (value.replace(" ", "").replace(",", "."))
    )


@pytest.mark.parametrize(
    ("name", "language"), [(ENGLISH_PAIR, "eng_Latn"), (RUSSIAN_PAIR, "rus_Cyrl")]
)
def test_numbers_follow_each_languages_format_and_say_the_same_value(
    name: str, language: str
) -> None:
    checked: Counter[str] = Counter()
    for row in table(name):
        if row[2] not in ("int", "decimal", "grouped"):
            continue
        source = number_value(language, row[2], row[3])
        target = number_value("khk_Cyrl", row[2], row[4])
        assert source == target, row[0]
        checked[row[2]] += 1

    assert checked == {"int": 35, "decimal": 21, "grouped": 15}


# ---- G1.6: the content scan and the characters --------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_no_fixture_holds_an_excluded_word_or_name(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")

    assert hits(text) == [], "(line, label) pairs, never the text"


ALLOWED_CLASSES: Final = {
    ENGLISH_PAIR: {"ascii", "cyrillic", "placeholder bracket"},
    RUSSIAN_PAIR: {"ascii", "cyrillic", "placeholder bracket", "em dash"},
    MONGOLIAN: {"ascii", "cyrillic", "placeholder bracket", "mongolian"},
    MONO: {"ascii", "cyrillic", "placeholder bracket"},
    NOISE: {"ascii", "cyrillic"},
    NOUNS: {"ascii", "cyrillic"},
    ENTITIES: {"ascii", "cyrillic"},
}


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_fixture_holds_the_characters_of_its_languages_and_no_other(
    name: str,
) -> None:
    classes = character_classes((FIXTURES / name).read_text(encoding="utf-8"))

    assert set(classes) <= ALLOWED_CLASSES[name], sorted(classes)


def test_only_the_files_of_those_languages_hold_traditional_script_or_the_dash() -> (
    None
):
    wanted = {MONGOLIAN: "mongolian", RUSSIAN_PAIR: "em dash"}
    for name in FIXTURE_FILES:
        classes = character_classes((FIXTURES / name).read_text(encoding="utf-8"))
        for file, class_name in wanted.items():
            assert (class_name in classes) == (name == file), (name, class_name)


def test_the_files_are_utf8_with_line_feeds_one_header_line_and_no_byte_order_mark() -> (
    None
):
    for name in FIXTURE_FILES:
        data = (FIXTURES / name).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), name
        assert b"\r" not in data and data.endswith(b"\n"), name
        widths = {len(line.split(b"\t")) for line in data.split(b"\n")[:-1]}
        assert len(widths) == 1, name
        data.decode("utf-8")


# ---- the scan is itself tested ---------------------------------------------------------------


@pytest.mark.parametrize("number", range(len(excluded_words())))
def test_the_scan_finds_each_excluded_word_in_any_case_inside_a_sentence(
    number: int,
) -> None:
    word = excluded_words()[number]

    for planted in (word, word.upper(), word.capitalize()):
        assert hits(f"a line that says {planted} in the middle") == [
            (1, f"word {number}")
        ]


@pytest.mark.parametrize("number", range(len(excluded_names())))
def test_the_scan_finds_each_excluded_name_in_any_case(number: int) -> None:
    name = excluded_names()[number]

    assert hits(f"fine\n{name.upper()}\nfine") == [(2, f"name {number}")]


def test_the_scan_names_the_line_and_the_entry_and_never_the_text() -> None:
    planted = excluded_words()[3]

    found = hits(f"fine\nx {planted}\nfine again\n")

    assert found == [(2, "word 3")]
    assert planted not in repr(found)


def test_the_scan_lets_plain_sentences_through() -> None:
    clean = "The book is here. Where is the dog? The river, the window and the tree."

    assert hits(clean) == []


def test_the_scan_lists_are_distinct_and_long_enough_to_mean_something() -> None:
    words = excluded_words()

    assert len(set(words)) == len(words) == 18
    assert all(len(word) >= 4 for word in words)
    assert len(set(excluded_names())) == len(excluded_names()) == 2


def test_the_helper_and_this_module_spell_out_none_of_it() -> None:
    for path in (Path(__file__), REPO_ROOT / "tests" / "support" / "tiny_scans.py"):
        assert hits(path.read_text(encoding="utf-8")) == [], path.name


def test_the_audit_names_a_character_outside_the_classes_by_code_point() -> None:
    audited = character_classes("a" + chr(0x00A0) + chr(0x200B) + "b")

    assert audited == {"ascii": 2, "other U+00A0": 1, "other U+200B": 1}


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0x09, "ascii"),
        (0x0A, "ascii"),
        (0x0D, "other U+000D"),
        (0x1F, "other U+001F"),
        (0x20, "ascii"),
        (0x7E, "ascii"),
        (0x7F, "other U+007F"),
        (0x03FF, "other U+03FF"),
        (0x0400, "cyrillic"),
        (0x04FF, "cyrillic"),
        (0x0500, "other U+0500"),
        (0x17FF, "other U+17FF"),
        (0x1800, "mongolian"),
        (0x18AF, "mongolian"),
        (0x18B0, "other U+18B0"),
        (0x27E5, "other U+27E5"),
        (0x27E6, "placeholder bracket"),
        (0x27E7, "placeholder bracket"),
        (0x27E8, "other U+27E8"),
        (0x2013, "other U+2013"),
        (0x2014, "em dash"),
    ],
)
def test_each_character_class_has_its_exact_boundaries(
    code: int, expected: str
) -> None:
    assert character_classes(chr(code)) == {expected: 1}
