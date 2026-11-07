"""Converter corpora: word pairs in, Eole training files out.

A pair is one Cyrillic word and its traditional-script spelling. `build_corpus`
screens the pairs, splits them into train and valid by a hash of the Cyrillic word,
and writes one corpus per direction. The words come from a reference list
(`read_pairs_tsv`), from the rule baseline (`rule_pairs`) or from aligned lines of
real parallel text (`align_words`).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TextIO

from rinchen.errors import ConfigError, DataError
from rinchen.script_conv.rules import transliterate_word
from rinchen.script_conv.validate import validate_traditional
from rinchen.text.normalize import normalize
from rinchen.utils.part_file import unique_part_path

PairOrigin = Literal["reference", "rule", "parallel"]

EMPTY = "empty"
MULTIWORD = "multiword"
DUPLICATE = "duplicate"

_CYRILLIC = "khk_Cyrl"
_SCRIPT = "mn_Mong"
_TSV_HEADER = ("cyrillic", "script")
_MAX_VALID_FRACTION = 0.5
# A line break would split a word across lines and a tab would split a TSV field, so
# a side holding one is a malformed input, not a low-quality pair.
_LINE_BREAK_OR_TAB = re.compile(r"[\n\r\t]")
_HASH_SPACE = 2**256

# The files `build_corpus` writes, in the order it writes them.
CORPUS_FILE_NAMES = tuple(
    f"{direction}.{split}.{side}"
    for split in ("train", "valid")
    for direction in ("cyrl2mong", "mong2cyrl")
    for side in ("src", "tgt")
)


@dataclass(frozen=True)
class WordPair:
    """A Cyrillic word, its script spelling and where the pair came from."""

    cyrillic: str
    script: str
    origin: PairOrigin


@dataclass(frozen=True)
class CorpusFiles:
    """What `build_corpus` wrote.

    `files` and `rows` are keyed by file name (`cyrl2mong.train.src`, ...); `rows`
    counts lines. `dropped` is keyed by drop code and lists only the codes that
    fired: `empty`, `multiword`, `duplicate` and each code of
    `rinchen.script_conv.validate.VIOLATION_CODES`.
    """

    directory: Path
    files: Mapping[str, Path]
    rows: Mapping[str, int]
    dropped: Mapping[str, int]


@dataclass(frozen=True)
class _Drop:
    code: str


def _words(line: str) -> list[str]:
    # U+0020 only: str.split() with no argument also splits at U+202F, which joins
    # a suffix to its stem inside one script word.
    return [word for word in line.split(" ") if word]


def align_words(cyrillic_line: str, script_line: str) -> tuple[WordPair, ...] | None:
    """The word pairs of two aligned lines, or None when they cannot be paired.

    Both lines are split on U+0020 only. They pair up only when each holds at least
    one word and the word counts are equal; word order is then taken to match, which
    is the caller's claim about the parallel text, not something checked here.
    """
    cyrillic_words = _words(cyrillic_line)
    script_words = _words(script_line)
    if not cyrillic_words or len(cyrillic_words) != len(script_words):
        return None
    return tuple(
        WordPair(cyrillic, script, "parallel")
        for cyrillic, script in zip(cyrillic_words, script_words, strict=True)
    )


def rule_pairs(
    words: Iterable[str], *, allow_fallback: bool = False
) -> tuple[WordPair, ...]:
    """Pairs of each word with the rule baseline's spelling of it.

    Only a word the lexicon holds yields a pair, unless `allow_fallback` also admits
    the phonographic guess. A word that spells to nothing (only dropped letters) is
    skipped either way. Each word is normalized (`khk_Cyrl`) first, so the lookup and
    the pair both carry the normalized word.
    """
    pairs: list[WordPair] = []
    for raw_word in words:
        word = normalize(raw_word, _CYRILLIC)
        result = transliterate_word(word)
        if result.text and (allow_fallback or result.source == "lexicon"):
            pairs.append(WordPair(word, result.text, "rule"))
    return tuple(pairs)


def read_pairs_tsv(
    path: Path, *, origin: PairOrigin = "reference"
) -> tuple[WordPair, ...]:
    """The pairs of a two-column TSV (UTF-8, LF) headed `cyrillic<TAB>script`.

    Quotes are ordinary characters. Raises DataError, naming `path:line`, for a wrong
    header, a row that is not two fields or an empty field, and DataError for a file
    that cannot be read or decoded.
    """
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return _pairs_from_handle(handle, path, origin)
    except (UnicodeDecodeError, OSError, csv.Error) as exc:
        raise DataError(f"cannot read pairs file {path}: {exc}") from exc


def _pairs_from_handle(
    handle: TextIO, path: Path, origin: PairOrigin
) -> tuple[WordPair, ...]:
    reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader, None)
    if header is None:
        raise DataError(f"{path}:1: the file is empty")
    if tuple(header) != _TSV_HEADER:
        raise DataError(f"{path}:1: expected header 'cyrillic<TAB>script'")
    pairs: list[WordPair] = []
    for row in reader:
        if len(row) != 2:
            raise DataError(
                f"{path}:{reader.line_num}: expected 2 tab-separated fields,"
                f" found {len(row)}"
            )
        cyrillic, script = row
        if not cyrillic or not script:
            raise DataError(f"{path}:{reader.line_num}: empty field")
        pairs.append(WordPair(cyrillic, script, origin))
    return tuple(pairs)


def _is_blank(text: str) -> bool:
    return not text.strip(" ")


def _reject_line_breaks(index: int, pair: WordPair) -> None:
    for side, text in (("cyrillic", pair.cyrillic), ("script", pair.script)):
        if _LINE_BREAK_OR_TAB.search(text):
            raise DataError(
                f"pair index {index}: the {side} side holds a line break or tab:"
                f" {text!r}"
            )


def _screen_pair(pair: WordPair) -> WordPair | _Drop:
    """The normalized pair, or the first check it fails.

    The checks run in order: empty, multiword, a script-side violation, then (in the
    caller) duplicate. The validator reads the raw script side because `normalize`
    strips joiners, which would hide them. A Cyrillic side is normalized first because
    normalization turns other whitespace into spaces. A script side that passes the
    validator is never empty once normalized: it holds a character `normalize` keeps.
    """
    cyrillic = normalize(pair.cyrillic, _CYRILLIC)
    if not cyrillic or _is_blank(pair.script):
        return _Drop(EMPTY)
    if " " in cyrillic or " " in pair.script.strip(" "):
        return _Drop(MULTIWORD)
    violations = validate_traditional(pair.script)
    if violations:
        return _Drop(violations[0].code)
    return WordPair(cyrillic, normalize(pair.script, _SCRIPT), pair.origin)


def _screen_pairs(pairs: Sequence[WordPair]) -> tuple[list[WordPair], Counter[str]]:
    kept: list[WordPair] = []
    seen: set[tuple[str, str]] = set()
    dropped: Counter[str] = Counter()
    for index, pair in enumerate(pairs):
        _reject_line_breaks(index, pair)
        screened = _screen_pair(pair)
        if isinstance(screened, _Drop):
            dropped[screened.code] += 1
        elif (screened.cyrillic, screened.script) in seen:
            dropped[DUPLICATE] += 1
        else:
            seen.add((screened.cyrillic, screened.script))
            kept.append(screened)
    return kept, dropped


def _goes_to_valid(cyrillic: str, valid_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{cyrillic}".encode()).hexdigest()
    return int(digest, 16) / _HASH_SPACE < valid_fraction


def _split(
    pairs: Sequence[WordPair], valid_fraction: float, seed: int
) -> tuple[list[WordPair], list[WordPair]]:
    train: list[WordPair] = []
    valid: list[WordPair] = []
    for pair in pairs:
        side = valid if _goes_to_valid(pair.cyrillic, valid_fraction, seed) else train
        side.append(pair)
    return train, valid


def _corpus_lines(
    train: Sequence[WordPair], valid: Sequence[WordPair]
) -> dict[str, list[str]]:
    lines: dict[str, list[str]] = {}
    for split, pairs in (("train", train), ("valid", valid)):
        cyrillic = [pair.cyrillic for pair in pairs]
        script = [pair.script for pair in pairs]
        lines[f"cyrl2mong.{split}.src"] = cyrillic
        lines[f"cyrl2mong.{split}.tgt"] = script
        lines[f"mong2cyrl.{split}.src"] = script
        lines[f"mong2cyrl.{split}.tgt"] = cyrillic
    return lines


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write `lines` (UTF-8, LF, a newline after each) so the file appears whole."""
    part_path = unique_part_path(path)
    try:
        with part_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(f"{line}\n" for line in lines)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def check_valid_fraction(valid_fraction: float) -> None:
    """Raises ConfigError unless `valid_fraction` is in (0, 0.5]; `build_corpus`
    checks it first, and a caller that must not touch its outputs for a bad value
    checks it before."""
    if not 0 < valid_fraction <= _MAX_VALID_FRACTION:
        raise ConfigError(
            f"valid_fraction must be in (0, {_MAX_VALID_FRACTION}], got {valid_fraction}"
        )


def build_corpus(
    pairs: Sequence[WordPair], out_dir: Path, *, valid_fraction: float, seed: int
) -> CorpusFiles:
    """Write the train and valid corpora of both directions under `out_dir`.

    Files: `cyrl2mong.{train,valid}.{src,tgt}` and the same four for `mong2cyrl`,
    whose columns are the swapped ones. Each is UTF-8, LF, one word per line.

    A pair is dropped on the first check it fails, in this order, and counted under
    the code in brackets:
    1. an empty side: nothing but U+0020, or (Cyrillic side) nothing left once
       normalized [`empty`];
    2. a side that still holds U+0020 after its outer spaces are stripped, because a
       corpus line is one word (NNBSP does not count) [`multiword`];
    3. a violation on the raw script side, counted under the first one's code;
    4. a repeat of an earlier pair once both sides are normalized, the first is kept
       [`duplicate`].
    Cyrillic sides go through `normalize(..., "khk_Cyrl")` and script sides through
    `normalize(..., "mn_Mong")`, which leaves NNBSP, MVS and FVS alone.

    A pair goes to valid when the SHA-256 of `"{seed}:{normalized Cyrillic word}"`,
    read as a fraction of 2**256, is below `valid_fraction`; so equal Cyrillic words
    never straddle the split. Output keeps input order. A small input can leave
    valid with no pair; its files are then written empty. An empty train split is an
    error.

    Each of the eight files appears whole, through a temporary file, but they are
    not written atomically as a set: a failure part-way leaves the files finished so
    far (written split by split, a src and its tgt together) and no temporary file.
    The caller owns the commit marker: call `begin_commit(out_dir, stage)` before
    this function and write the manifest last.

    Raises:
        ConfigError: `valid_fraction` is outside (0, 0.5].
        DataError: a side holds a line break or tab (it names the pair's index in
            `pairs`), no pair is left after dropping, or the train split is empty.
    """
    check_valid_fraction(valid_fraction)
    kept, dropped = _screen_pairs(pairs)
    if not kept:
        raise DataError(
            f"no pair is left after dropping {sum(dropped.values())} of {len(pairs)}"
            f" ({dict(dropped)})"
        )
    train, valid = _split(kept, valid_fraction, seed)
    if not train:
        raise DataError(
            f"train split is empty: {len(kept)} pairs, valid_fraction"
            f" {valid_fraction}, seed {seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    lines_by_name = _corpus_lines(train, valid)
    for name, lines in lines_by_name.items():
        _write_lines(out_dir / name, lines)
    return CorpusFiles(
        directory=out_dir,
        files=MappingProxyType({name: out_dir / name for name in lines_by_name}),
        rows=MappingProxyType(
            {name: len(lines) for name, lines in lines_by_name.items()}
        ),
        dropped=MappingProxyType(dict(dropped)),
    )
