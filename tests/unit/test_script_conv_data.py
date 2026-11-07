"""Converter corpus builder: word pairs in, Eole training files out.

Mongolian and invisible characters are written as `chr()` calls so the exact code
points stay reviewable. The split is checked against an independent hash in this
module, so a test failure cannot be a bug the builder shares with its oracle.
"""

from __future__ import annotations

import codecs
import dataclasses
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.script_conv.alphabet import LETTERS, MVS, NNBSP
from rinchen.script_conv.data import (
    CorpusFiles,
    WordPair,
    align_words,
    build_corpus,
    read_pairs_tsv,
    rule_pairs,
)
from rinchen.script_conv.rules import LEXICON, transliterate_word
from rinchen.script_conv.validate import validate_traditional
from tests.support.script_reference import FIXTURE_DIR

TINY_WORDS = FIXTURE_DIR / "tiny_words.tsv"
HEADER = "cyrillic\tscript\n"

A = chr(0x1820)
NA = chr(0x1828)
FVS_SECOND = chr(0x180B)
FVS_FIFTH = chr(0x180F)  # not a listed variant of A
ZWJ = chr(0x200D)
CYRILLIC_A = chr(0x0430)
CYRILLIC_O = chr(0x043E)
LATIN_O = chr(0x006F)
# Mongolian "ongol" (the word MONGOL) and the genitive suffix after an NNBSP.
MONGOL = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
SUFFIX_UN = chr(0x1824) + chr(0x1828)

SCRIPT_WITH_NNBSP = MONGOL + NNBSP + SUFFIX_UN
SCRIPT_WITH_FVS = A + FVS_SECOND + NA
SCRIPT_WITH_MVS = A + MVS + NA

SEED = 7
VALID_FRACTION = 0.25
SORTED_LETTERS = sorted(LETTERS)
CYRILLIC_LETTERS = "абвгдежзик"

DIRECTIONS = ("cyrl2mong", "mong2cyrl")
SPLITS = ("train", "valid")
SIDES = ("src", "tgt")
FILE_NAMES = {
    f"{direction}.{split}.{side}"
    for direction in DIRECTIONS
    for split in SPLITS
    for side in SIDES
}

# Word lists for the rule_pairs tests: a word whose lexicon spelling differs from
# the fallback's, a word the lexicon does not hold, and a word made only of a letter
# the fallback drops.
LEXICON_WORD = next(word for word in sorted(LEXICON) if len(word) > 3)
FALLBACK_WORD = "абвгд"
DROPPED_LETTER_WORD = "ъ"


def _cyrillic_word(index: int) -> str:
    """A distinct two-letter Cyrillic word for each index below 100."""
    first, second = divmod(index, len(CYRILLIC_LETTERS))
    return CYRILLIC_LETTERS[first] + CYRILLIC_LETTERS[second]


def _script_word(index: int) -> str:
    """A distinct two-letter script word for each index below 1225."""
    first, second = divmod(index, len(SORTED_LETTERS))
    return SORTED_LETTERS[first] + SORTED_LETTERS[second]


def _pair(index: int) -> WordPair:
    return WordPair(_cyrillic_word(index), _script_word(index), "reference")


def _in_valid(
    cyrillic: str, seed: int = SEED, fraction: float = VALID_FRACTION
) -> bool:
    """The split rule restated independently of the builder."""
    digest = hashlib.sha256(f"{seed}:{cyrillic}".encode()).hexdigest()
    return int(digest, 16) / 2**256 < fraction


def _planted_pairs() -> tuple[list[WordPair], list[WordPair]]:
    """The 40 planted pairs and the 34 of them that must survive, in input order.

    Six are bad: three exact duplicates, one ZWJ in the script, one empty side, one
    Cyrillic letter in the script.
    """
    # Stepping by 7 (coprime to 34) keeps the words distinct but out of sorted order.
    kept = [_pair(index * 7 % 34) for index in range(34)]
    kept[3] = WordPair(kept[3].cyrillic, SCRIPT_WITH_NNBSP, "reference")
    kept[4] = WordPair(kept[4].cyrillic, SCRIPT_WITH_FVS, "reference")
    kept[5] = WordPair(kept[5].cyrillic, SCRIPT_WITH_MVS, "reference")
    zwj = WordPair(_cyrillic_word(40), A + ZWJ + NA, "reference")
    empty = WordPair("   ", _script_word(41), "reference")
    foreign = WordPair(_cyrillic_word(42), A + CYRILLIC_A + NA, "reference")
    planted = (
        kept[:20]
        + [kept[0], zwj]
        + kept[20:30]
        + [kept[1], empty, foreign]
        + kept[30:]
        + [kept[2]]
    )
    assert len(planted) == 40
    return planted, kept


def _lines(path: Path) -> list[str]:
    """The lines of a corpus file, after checking its bytes follow the file format."""
    raw = path.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8)
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    if not text:
        return []
    assert text.endswith("\n")
    return text[:-1].split("\n")


def _splits_by_cyrillic_word(result: CorpusFiles) -> dict[str, set[str]]:
    splits: defaultdict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        for line in _lines(result.files[f"cyrl2mong.{split}.src"]):
            splits[line].add(split)
    return dict(splits)


def _write_tsv(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


# --- WordPair ---------------------------------------------------------------


def test_word_pair_is_a_frozen_value() -> None:
    assert WordPair("а", A, "rule") == WordPair("а", A, "rule")
    with pytest.raises(dataclasses.FrozenInstanceError):
        WordPair("а", A, "rule").cyrillic = "б"  # type: ignore[misc]


# --- align_words ------------------------------------------------------------


def test_align_words_pairs_the_words_of_two_equal_length_lines() -> None:
    assert align_words("Монгол улс", MONGOL + " " + A + NA) == (
        WordPair("Монгол", MONGOL, "parallel"),
        WordPair("улс", A + NA, "parallel"),
    )


def test_align_words_returns_none_for_unequal_word_counts() -> None:
    assert align_words("Монгол улс", MONGOL) is None
    assert align_words("Монгол", MONGOL + " " + A) is None


def test_a_script_word_with_an_nnbsp_stays_one_word() -> None:
    assert align_words("Монголын", SCRIPT_WITH_NNBSP) == (
        WordPair("Монголын", SCRIPT_WITH_NNBSP, "parallel"),
    )


def test_an_nnbsp_does_not_make_the_word_counts_equal() -> None:
    assert align_words("Монгол ын", SCRIPT_WITH_NNBSP) is None


@pytest.mark.parametrize(
    ("cyrillic_line", "script_line"),
    [("", ""), ("Монгол", ""), ("", MONGOL), ("   ", "  ")],
)
def test_align_words_returns_none_when_a_line_has_no_word(
    cyrillic_line: str, script_line: str
) -> None:
    assert align_words(cyrillic_line, script_line) is None


def test_repeated_spaces_do_not_make_empty_words() -> None:
    assert align_words("а  б ", " " + A + "  " + NA) == (
        WordPair("а", A, "parallel"),
        WordPair("б", NA, "parallel"),
    )


# --- rule_pairs -------------------------------------------------------------


def test_rule_pairs_keeps_only_lexicon_words_by_default() -> None:
    assert FALLBACK_WORD not in LEXICON
    assert rule_pairs([FALLBACK_WORD, LEXICON_WORD]) == (
        WordPair(LEXICON_WORD, LEXICON[LEXICON_WORD], "rule"),
    )


def test_rule_pairs_with_fallback_keeps_the_fallback_spelling_too() -> None:
    fallback = transliterate_word(FALLBACK_WORD)
    assert fallback.source == "fallback"
    assert rule_pairs([FALLBACK_WORD, LEXICON_WORD], allow_fallback=True) == (
        WordPair(FALLBACK_WORD, fallback.text, "rule"),
        WordPair(LEXICON_WORD, LEXICON[LEXICON_WORD], "rule"),
    )


def test_rule_pairs_skips_a_word_that_spells_to_nothing() -> None:
    assert transliterate_word(DROPPED_LETTER_WORD).text == ""
    assert rule_pairs([DROPPED_LETTER_WORD], allow_fallback=True) == ()


def test_rule_pairs_accepts_any_iterable_of_words() -> None:
    words = (word for word in [LEXICON_WORD])
    assert [pair.cyrillic for pair in rule_pairs(words)] == [LEXICON_WORD]


# --- read_pairs_tsv ---------------------------------------------------------


def test_read_pairs_tsv_reads_the_tiny_fixture() -> None:
    pairs = read_pairs_tsv(TINY_WORDS)
    first_line = TINY_WORDS.read_text(encoding="utf-8").split("\n")[1]
    cyrillic, script = first_line.split("\t")
    assert len(pairs) == 500
    assert pairs[0] == WordPair(cyrillic, script, "reference")
    assert {pair.origin for pair in pairs} == {"reference"}


def test_read_pairs_tsv_can_label_the_origin() -> None:
    pairs = read_pairs_tsv(TINY_WORDS, origin="parallel")
    assert {pair.origin for pair in pairs} == {"parallel"}


def test_read_pairs_tsv_leaves_quotes_and_nnbsp_alone(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path / "pairs.tsv", HEADER + '"а\t' + SCRIPT_WITH_NNBSP + '"\n'
    )
    assert read_pairs_tsv(path) == (
        WordPair('"а', SCRIPT_WITH_NNBSP + '"', "reference"),
    )


def test_read_pairs_tsv_accepts_a_file_without_a_final_newline(tmp_path: Path) -> None:
    path = _write_tsv(tmp_path / "pairs.tsv", HEADER + "а\t" + A)
    assert read_pairs_tsv(path) == (WordPair("а", A, "reference"),)


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("", 1),
        ("cyrillic\ttraditional\n", 1),
        ("script\tcyrillic\n", 1),
        ("cyrillic\tscript\textra\n", 1),
        (HEADER + "а\t" + A + "\n" + "\n", 3),
        (HEADER + "а\t" + A + "\n" + "б\n", 3),
        (HEADER + "а\t" + A + "\n" + "б\t" + A + "\textra\n", 3),
        (HEADER + "\t" + A + "\n", 2),
        (HEADER + "а\t\n", 2),
    ],
)
def test_read_pairs_tsv_names_the_line_of_a_malformed_file(
    tmp_path: Path, text: str, line: int
) -> None:
    path = _write_tsv(tmp_path / "pairs.tsv", text)
    with pytest.raises(DataError, match=f"{path}:{line}"):
        read_pairs_tsv(path)


def test_read_pairs_tsv_rejects_a_byte_order_mark(tmp_path: Path) -> None:
    path = _write_tsv(tmp_path / "pairs.tsv", chr(0xFEFF) + HEADER)
    with pytest.raises(DataError, match=f"{path}:1"):
        read_pairs_tsv(path)


def test_read_pairs_tsv_rejects_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "pairs.tsv"
    path.write_bytes(HEADER.encode("utf-8") + b"\xff\t\xfe\n")
    with pytest.raises(DataError, match=str(path)):
        read_pairs_tsv(path)


def test_read_pairs_tsv_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing.tsv"):
        read_pairs_tsv(tmp_path / "missing.tsv")


# --- build_corpus: layout ---------------------------------------------------


def test_build_corpus_writes_the_eight_files_and_reports_their_rows(
    tmp_path: Path,
) -> None:
    planted, _ = _planted_pairs()
    out_dir = tmp_path / "nested" / "corpus"

    result = build_corpus(planted, out_dir, valid_fraction=VALID_FRACTION, seed=SEED)

    assert result.directory == out_dir
    assert set(result.files) == FILE_NAMES
    assert set(result.rows) == FILE_NAMES
    assert {path.name for path in out_dir.iterdir()} == FILE_NAMES
    for name, path in result.files.items():
        assert path == out_dir / name
        assert result.rows[name] == len(_lines(path))


def test_corpus_files_is_a_frozen_value_with_read_only_mappings(
    tmp_path: Path,
) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rows = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.rows["cyrl2mong.train.src"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.dropped["empty"] = 0  # type: ignore[index]


# --- build_corpus: the planted list -----------------------------------------


def test_build_corpus_counts_rows_and_drops_exactly_on_the_planted_list(
    tmp_path: Path,
) -> None:
    planted, kept = _planted_pairs()
    expected_valid = [pair for pair in kept if _in_valid(pair.cyrillic)]
    assert len(expected_valid) == 8

    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)

    assert result.dropped == {
        "duplicate": 3,
        "joiner": 1,
        "empty": 1,
        "foreign_character": 1,
    }
    assert dict(result.rows) == {
        "cyrl2mong.train.src": 26,
        "cyrl2mong.train.tgt": 26,
        "cyrl2mong.valid.src": 8,
        "cyrl2mong.valid.tgt": 8,
        "mong2cyrl.train.src": 26,
        "mong2cyrl.train.tgt": 26,
        "mong2cyrl.valid.src": 8,
        "mong2cyrl.valid.tgt": 8,
    }


def test_the_split_follows_the_hash_of_the_cyrillic_word_in_input_order(
    tmp_path: Path,
) -> None:
    planted, kept = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)

    for split, wanted in (("train", False), ("valid", True)):
        chosen = [pair for pair in kept if _in_valid(pair.cyrillic) == wanted]
        assert _lines(result.files[f"cyrl2mong.{split}.src"]) == [
            pair.cyrillic for pair in chosen
        ]
        assert _lines(result.files[f"cyrl2mong.{split}.tgt"]) == [
            pair.script for pair in chosen
        ]


def test_train_and_valid_hold_disjoint_cyrillic_words(tmp_path: Path) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    train = set(_lines(result.files["cyrl2mong.train.src"]))
    valid = set(_lines(result.files["cyrl2mong.valid.src"]))
    assert train
    assert valid
    assert not train & valid


def test_the_dropped_pairs_reach_no_file(tmp_path: Path) -> None:
    planted, kept = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    kept_words = {pair.cyrillic for pair in kept}
    written = {
        line
        for split in SPLITS
        for line in _lines(result.files[f"cyrl2mong.{split}.src"])
    }
    assert written == kept_words


def test_mong2cyrl_is_the_cyrl2mong_columns_swapped(tmp_path: Path) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    for split in SPLITS:
        forward_src = _lines(result.files[f"cyrl2mong.{split}.src"])
        forward_tgt = _lines(result.files[f"cyrl2mong.{split}.tgt"])
        assert _lines(result.files[f"mong2cyrl.{split}.src"]) == forward_tgt
        assert _lines(result.files[f"mong2cyrl.{split}.tgt"]) == forward_src
        assert forward_src
        assert all(validate_traditional(script) == () for script in forward_tgt)


@pytest.mark.parametrize(
    "script", [SCRIPT_WITH_NNBSP, SCRIPT_WITH_FVS, SCRIPT_WITH_MVS]
)
def test_nnbsp_fvs_and_mvs_survive_unchanged(tmp_path: Path, script: str) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    for name in ("cyrl2mong.train.tgt", "cyrl2mong.valid.tgt"):
        if script in _lines(result.files[name]):
            break
    else:
        pytest.fail(f"{script!r} did not reach a target file")
    sources = _lines(result.files["mong2cyrl.train.src"]) + _lines(
        result.files["mong2cyrl.valid.src"]
    )
    assert script in sources


def test_files_are_utf8_with_lf_and_a_final_newline(tmp_path: Path) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    for path in result.files.values():
        raw = path.read_bytes()
        assert not raw.startswith(codecs.BOM_UTF8)
        assert b"\r" not in raw
        assert raw == b"" or raw.endswith(b"\n")
        raw.decode("utf-8")


def test_the_same_seed_gives_identical_files_and_another_seed_another_split(
    tmp_path: Path,
) -> None:
    planted, _ = _planted_pairs()
    first = build_corpus(
        planted, tmp_path / "a", valid_fraction=VALID_FRACTION, seed=SEED
    )
    again = build_corpus(
        planted, tmp_path / "b", valid_fraction=VALID_FRACTION, seed=SEED
    )
    other = build_corpus(
        planted, tmp_path / "c", valid_fraction=VALID_FRACTION, seed=SEED + 1
    )

    for name in FILE_NAMES:
        assert first.files[name].read_bytes() == again.files[name].read_bytes()
    assert (
        first.files["cyrl2mong.valid.src"].read_bytes()
        != other.files["cyrl2mong.valid.src"].read_bytes()
    )


def test_building_again_replaces_the_files_and_leaves_no_part_file(
    tmp_path: Path,
) -> None:
    planted, _ = _planted_pairs()
    build_corpus(planted[:10], tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    result = build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    assert sum(result.rows[f"cyrl2mong.{split}.src"] for split in SPLITS) == 34
    assert {path.name for path in tmp_path.iterdir()} == FILE_NAMES


def test_a_failed_write_leaves_no_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted, _ = _planted_pairs()

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        build_corpus(planted, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_valid_split_is_written_as_empty_files(tmp_path: Path) -> None:
    train_word = next(
        word
        for word in map(_cyrillic_word, range(100))
        if not _in_valid(word, SEED, VALID_FRACTION)
    )
    result = build_corpus(
        [WordPair(train_word, A, "reference")],
        tmp_path,
        valid_fraction=VALID_FRACTION,
        seed=SEED,
    )
    assert result.rows["cyrl2mong.valid.src"] == 0
    assert result.rows["mong2cyrl.valid.tgt"] == 0
    assert result.files["cyrl2mong.valid.src"].read_bytes() == b""
    assert result.rows["cyrl2mong.train.src"] == 1


# --- build_corpus: the split key --------------------------------------------


def test_pairs_sharing_a_cyrillic_word_never_straddle_the_split(
    tmp_path: Path,
) -> None:
    pairs = [
        WordPair(_cyrillic_word(index), _script_word(index + shift), "reference")
        for index in range(30)
        for shift in (0, 50)
    ]
    result = build_corpus(pairs, tmp_path, valid_fraction=0.5, seed=SEED)

    splits = _splits_by_cyrillic_word(result)
    assert len(splits) == 30
    assert all(len(where) == 1 for where in splits.values())
    assert {"train", "valid"} == set().union(*splits.values())


def test_the_split_key_is_the_normalized_cyrillic_word(tmp_path: Path) -> None:
    pairs: list[WordPair] = []
    for index in range(30):
        word = "мо" + _cyrillic_word(index)
        lookalike = word.replace(CYRILLIC_O, LATIN_O)
        pairs.append(WordPair(word, _script_word(index), "reference"))
        pairs.append(WordPair(lookalike, _script_word(index + 50), "reference"))

    result = build_corpus(pairs, tmp_path, valid_fraction=0.5, seed=SEED)

    splits = _splits_by_cyrillic_word(result)
    assert len(splits) == 30
    assert all(LATIN_O not in word for word in splits)
    assert all(len(where) == 1 for where in splits.values())
    assert {"train", "valid"} == set().union(*splits.values())


# --- build_corpus: drop order -----------------------------------------------


def _dropped(tmp_path: Path, pairs: list[WordPair]) -> dict[str, int]:
    """The drop counts of `pairs` plus one good pair that keeps the build legal."""
    good = _pair(99)
    result = build_corpus(
        [*pairs, good], tmp_path, valid_fraction=VALID_FRACTION, seed=SEED
    )
    return dict(result.dropped)


@pytest.mark.parametrize(
    "pair",
    [
        WordPair("", A, "reference"),
        WordPair("а", "", "reference"),
        WordPair("   ", A, "reference"),
        WordPair("а", "  ", "reference"),
        # Empty only once normalization strips the zero-width space.
        WordPair(chr(0x200B), A, "reference"),
    ],
)
def test_a_pair_with_an_empty_side_is_dropped_as_empty(
    tmp_path: Path, pair: WordPair
) -> None:
    assert _dropped(tmp_path, [pair]) == {"empty": 1}


@pytest.mark.parametrize("cyrillic", ["", "   "])
def test_empty_outranks_a_violation_on_the_script_side(
    tmp_path: Path, cyrillic: str
) -> None:
    pair = WordPair(cyrillic, A + ZWJ + NA, "reference")
    assert _dropped(tmp_path, [pair]) == {"empty": 1}


def test_the_validator_sees_the_raw_script_side_before_normalization(
    tmp_path: Path,
) -> None:
    # Normalization would strip the ZWJ, leaving a pair equal to the first one.
    first = WordPair("аб", A + NA, "reference")
    joined = WordPair("аб", A + ZWJ + NA, "reference")
    assert _dropped(tmp_path, [first, joined]) == {"joiner": 1}


def test_a_violation_outranks_duplicate(tmp_path: Path) -> None:
    joined = WordPair("аб", A + ZWJ + NA, "reference")
    assert _dropped(tmp_path, [joined, joined]) == {"joiner": 2}


def test_a_pair_counts_once_under_the_first_violation_by_position(
    tmp_path: Path,
) -> None:
    pair = WordPair("аб", A + ZWJ + CYRILLIC_A + NA, "reference")
    assert _dropped(tmp_path, [pair]) == {"joiner": 1}


@pytest.mark.parametrize(
    ("script", "code"),
    [
        (A + ZWJ + NA, "joiner"),
        (A + CYRILLIC_A, "foreign_character"),
        (FVS_SECOND + A, "orphan_selector"),
        (A + NNBSP, "misplaced_nnbsp"),
        (A + FVS_FIFTH, "invalid_variant"),
    ],
)
def test_every_violation_code_is_counted_under_its_own_name(
    tmp_path: Path, script: str, code: str
) -> None:
    assert validate_traditional(script)[0].code == code
    assert _dropped(tmp_path, [WordPair("аб", script, "reference")]) == {code: 1}


def test_duplicates_are_found_after_normalization_and_the_first_is_kept(
    tmp_path: Path,
) -> None:
    plain = "мо" + "нгол"
    lookalike = plain.replace(CYRILLIC_O, LATIN_O)
    pairs = [
        WordPair(plain, MONGOL, "reference"),
        WordPair(lookalike, MONGOL, "rule"),
        WordPair(plain, A + NA, "reference"),
    ]
    result = build_corpus(pairs, tmp_path, valid_fraction=0.5, seed=SEED)

    assert dict(result.dropped) == {"duplicate": 1}
    words = _splits_by_cyrillic_word(result)
    assert list(words) == [plain]
    scripts = _lines(result.files["cyrl2mong.train.tgt"]) + _lines(
        result.files["cyrl2mong.valid.tgt"]
    )
    assert sorted(scripts) == sorted([MONGOL, A + NA])


def test_a_cyrillic_side_is_normalized_with_khk_cyrl(tmp_path: Path) -> None:
    plain = "мо" + "нгол"
    lookalike = plain.replace(CYRILLIC_O, LATIN_O)
    assert lookalike != plain
    result = build_corpus(
        [WordPair(lookalike, MONGOL, "reference")],
        tmp_path,
        valid_fraction=0.5,
        seed=SEED,
    )
    assert list(_splits_by_cyrillic_word(result)) == [plain]


# --- build_corpus: refusals -------------------------------------------------


@pytest.mark.parametrize(
    "valid_fraction", [0.0, -0.1, 0.5000001, 1.0, math.nan, math.inf]
)
def test_valid_fraction_outside_zero_to_one_half_is_a_config_error(
    tmp_path: Path, valid_fraction: float
) -> None:
    planted, _ = _planted_pairs()
    with pytest.raises(ConfigError, match="valid_fraction"):
        build_corpus(planted, tmp_path, valid_fraction=valid_fraction, seed=SEED)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("valid_fraction", [0.5, 0.01])
def test_the_bounds_of_the_valid_fraction_that_are_allowed(
    tmp_path: Path, valid_fraction: float
) -> None:
    planted, _ = _planted_pairs()
    result = build_corpus(planted, tmp_path, valid_fraction=valid_fraction, seed=SEED)
    assert set(result.files) == FILE_NAMES


def test_no_pair_left_after_dropping_is_a_data_error(tmp_path: Path) -> None:
    bad = [
        WordPair("", A, "reference"),
        WordPair("аб", A + ZWJ, "reference"),
        WordPair("вг", CYRILLIC_A, "reference"),
    ]
    with pytest.raises(DataError, match="no pair"):
        build_corpus(bad, tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)
    assert list(tmp_path.iterdir()) == []


def test_no_pairs_at_all_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no pair"):
        build_corpus([], tmp_path, valid_fraction=VALID_FRACTION, seed=SEED)


@pytest.mark.parametrize("separator", ["\n", "\r", "\t"])
@pytest.mark.parametrize("side", ["cyrillic", "script"])
def test_a_line_break_or_tab_in_a_side_is_a_data_error_not_a_drop(
    tmp_path: Path, separator: str, side: str
) -> None:
    cyrillic, script = "аб", A + NA
    if side == "cyrillic":
        cyrillic = "а" + separator + "б"
    else:
        script = A + separator + NA
    with pytest.raises(DataError, match="pair 1"):
        build_corpus(
            [_pair(0), WordPair(cyrillic, script, "reference")],
            tmp_path,
            valid_fraction=VALID_FRACTION,
            seed=SEED,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_line_break_is_an_error_even_in_a_pair_that_would_be_dropped(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataError, match="pair 0"):
        build_corpus(
            [WordPair("", A + "\n", "reference"), _pair(0)],
            tmp_path,
            valid_fraction=VALID_FRACTION,
            seed=SEED,
        )


# --- end to end with the tiny fixture ---------------------------------------


def test_the_tiny_fixture_builds_without_a_drop(tmp_path: Path) -> None:
    pairs = read_pairs_tsv(TINY_WORDS)
    result = build_corpus(pairs, tmp_path, valid_fraction=0.1, seed=SEED)

    assert dict(result.dropped) == {}
    assert sum(result.rows[f"cyrl2mong.{split}.src"] for split in SPLITS) == len(pairs)
    assert result.rows["cyrl2mong.valid.src"] > 0
