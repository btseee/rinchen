"""The D4 planted-failure corpus: 50 khk_Cyrl -> eng_Latn rows in which 22 rows
each violate exactly one filter stage and 28 are clean.

Every text is pseudo-text (see `pseudo_text`). The construction is structural
and deterministic: row types differ in script mix, language, length ratio or
token fragmentation, and the clean rows come from the tiny SentencePiece
fixture. The margins were measured with the tiny SentencePiece model
(`tiny_spm_model`) and the tiny LID model (`tiny_lid`); the self-test
`tests/unit/test_support_planted_corpus.py` recomputes every property
independently of the filter stages.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import load_spm_config
from rinchen.corpus.schema import corpus_frame
from rinchen.spm.train import train_spm
from tests.support import pseudo_text as pt
from tests.support.embedders import concept_vector

PLANTED_REASON_COUNTS: dict[str, int] = {
    "script_src": 2,
    "script_tgt": 2,
    "lid_src": 2,
    "lid_tgt": 2,
    "len_ratio_char": 3,
    "len_ratio_token": 2,
    "numbers": 3,
    "duplicate": 3,
    "labse": 3,
}

SRC_LANG = "khk_Cyrl"
TGT_LANG = "eng_Latn"
CLEAN_ROWS = 28
TINY_SPM_CONFIG = REPO_ROOT / "configs" / "spm" / "tiny_1k.yaml"

# The stage that flags each reason, named as in `FilterConfig.stages`.
_STAGE_OF_REASON = {
    "script_src": "script_ratio",
    "script_tgt": "script_ratio",
    "numbers": "numbers",
    "len_ratio_char": "length",
    "len_ratio_token": "length",
    "lid_src": "lid",
    "lid_tgt": "lid",
    "duplicate": "minhash",
    "labse": "labse",
}
_STAGES = ("script_ratio", "numbers", "length", "lid", "minhash", "labse")

_SEED = "d4-planted"
_CREATED = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)
_DUPLICATE_SUFFIX = "!"
_MIN_CLEAN_LENGTH = 90  # src + tgt characters, so one more keeps Jaccard > 0.98
_CLEAN_BALANCE = (0.85, 1.2)  # tgt / src non-space length
_CLEAN_SRC_MIN_MARKED_WORDS = 3
_DUPLICATED_CLEAN_ROWS = (5, 12, 20)
_MAX_CONCEPT_COSINE = 0.3
_MAX_ATTEMPTS = 1000

# Letters that only one of two look-alike languages uses, so that a line with
# enough words containing them is unambiguous for a language identifier.
_KHK_MARKERS = "өү"
_RUS_MARKERS = "ыеёюжфщ"
_DEU_MARKERS = "äöüß"


# Compared by identity: the frame's own `==` is elementwise, not a bool.
@dataclass(frozen=True, slots=True, eq=False)
class PlantedCorpus:
    frame: pl.DataFrame
    # Every src and tgt text -> its concept id; the labse rows' tgt differs.
    concepts: dict[str, int]
    # Row id -> the reason the full pipeline gives it; None for clean rows.
    expected_reason: dict[str, str | None]
    # Stage name -> ids of the rows that stage flags when run alone.
    isolated_failures: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class _Row:
    src: str
    tgt: str
    reason: str | None


def tiny_spm_model(directory: Path) -> Path:
    """Train the 1000-piece tiny SentencePiece model into `directory` and
    return the path of its `.model` file."""
    cfg = load_spm_config(TINY_SPM_CONFIG).model_copy(
        update={"output_prefix": directory / "tiny"}
    )
    return train_spm(cfg)


def build_planted_corpus() -> PlantedCorpus:
    clean = _clean_rows()
    planted = [
        *_script_rows(),
        *_lid_rows(),
        *_length_rows(),
        *_number_rows(),
        *_labse_rows(),
    ]
    rows = _interleave(clean, planted)
    frame = corpus_frame(
        [
            {
                "src_lang": SRC_LANG,
                "tgt_lang": TGT_LANG,
                "src": row.src,
                "tgt": row.tgt,
                "origin": "web",
                "corpus": "planted",
                "split": "train",
                "doc_id": "planted",
                "seg_idx": position,
            }
            for position, row in enumerate(rows)
        ],
        created=_CREATED,
    )
    ids = frame["id"].to_list()
    expected_reason = {
        row_id: row.reason for row_id, row in zip(ids, rows, strict=True)
    }
    return PlantedCorpus(
        frame=frame,
        concepts=_concepts(rows),
        expected_reason=expected_reason,
        isolated_failures={
            stage: frozenset(
                row_id
                for row_id, reason in expected_reason.items()
                if reason is not None and _STAGE_OF_REASON[reason] == stage
            )
            for stage in _STAGES
        },
    )


def _rng(*parts: object) -> random.Random:
    # A string seed is hashed with SHA-512, so it does not depend on
    # PYTHONHASHSEED.
    return random.Random("-".join(str(part) for part in (_SEED, *parts)))


def _marked_words(line: str, markers: str) -> int:
    return sum(any(char in markers for char in word) for word in line.split())


def _marked_line(
    rng: random.Random,
    syllables: tuple[str, ...],
    markers: str,
    target: int,
    *,
    min_marked: int = 4,
    max_syllables: int = 3,
) -> str:
    for _ in range(_MAX_ATTEMPTS):
        line = pt.make_line_of_length(
            rng, syllables, target, max_syllables=max_syllables
        )
        if _marked_words(line, markers) >= min_marked:
            return line
    raise RuntimeError(f"no line of {target} characters with {min_marked} marked words")


def _khk_line(rng: random.Random, target: int, **options: int) -> str:
    return _marked_line(rng, pt.KHK_SYLLABLES, _KHK_MARKERS, target, **options)


def _eng_line(rng: random.Random, target: int) -> str:
    return pt.make_line_of_length(rng, pt.ENG_SYLLABLES, target)


def _balanced_lengths(rng: random.Random) -> tuple[int, int]:
    src = rng.randint(40, 58)
    return src, round(src * rng.uniform(0.9, 1.15))


def _insert_word(rng: random.Random, line: str, word: str) -> str:
    # Before the last word, so the full stop stays at the end of the line.
    words = line.split()
    words.insert(rng.randint(0, len(words) - 1), word)
    return " ".join(words)


def _clean_rows() -> list[_Row]:
    rng = _rng("clean")
    eng_lines = [
        line
        for line in pt.fixture_lines("eng_Latn")
        if not any(char.isdigit() for char in line)
    ]
    khk_lines = [
        line
        for line in pt.fixture_lines("khk_Cyrl")
        if not any(char.isdigit() for char in line)
        and _marked_words(line, _KHK_MARKERS) >= _CLEAN_SRC_MIN_MARKED_WORDS
    ]
    rng.shuffle(eng_lines)
    rows: list[_Row] = []
    for tgt in eng_lines:
        candidates = [
            src
            for src in khk_lines
            if _CLEAN_BALANCE[0]
            <= pt.non_space_length(tgt) / pt.non_space_length(src)
            <= _CLEAN_BALANCE[1]
            and len(src) + len(tgt) >= _MIN_CLEAN_LENGTH
        ]
        if not candidates:
            continue
        src = rng.choice(candidates)
        khk_lines.remove(src)
        rows.append(_Row(src, tgt, None))
        if len(rows) == CLEAN_ROWS:
            return _with_decimal_numbers(rng, rows)
    raise RuntimeError(f"the fixtures hold fewer than {CLEAN_ROWS} balanced pairs")


def _with_decimal_numbers(rng: random.Random, rows: list[_Row]) -> list[_Row]:
    """Give the first three rows numbers whose separators differ by side
    (`1,5` / `1.5`) while the digits stay the same."""
    numbers = (
        (("1,5",), ("1.5",)),
        (("1,5", "2024"), ("1.5", "2024")),
        (("1,5", "12 000"), ("1.5", "12,000")),
    )
    with_numbers = []
    for row, (src_numbers, tgt_numbers) in zip(rows, numbers, strict=False):
        src, tgt = row.src, row.tgt
        for src_number in src_numbers:
            src = _insert_word(rng, src, src_number)
        for tgt_number in tgt_numbers:
            tgt = _insert_word(rng, tgt, tgt_number)
        with_numbers.append(_Row(src, tgt, None))
    return [*with_numbers, *rows[len(numbers) :]]


def _script_rows() -> list[_Row]:
    """Lines that quote long words of the other script: 40% of the letters
    are foreign, so the script share is far below the threshold, while a
    language identifier still follows the word majority."""
    rows = []
    for index in range(PLANTED_REASON_COUNTS["script_src"]):
        rng = _rng("script_src", index)
        src = pt.make_mixed_line(
            rng, pt.KHK_SYLLABLES, pt.ENG_SYLLABLES, native_words=7, foreign_words=2
        )
        tgt = _eng_line(rng, pt.non_space_length(src))
        rows.append(_Row(src, tgt, "script_src"))
    for index in range(PLANTED_REASON_COUNTS["script_tgt"]):
        rng = _rng("script_tgt", index)
        tgt = pt.make_mixed_line(
            rng, pt.ENG_SYLLABLES, pt.KHK_SYLLABLES, native_words=7, foreign_words=2
        )
        src = _khk_line(rng, pt.non_space_length(tgt))
        rows.append(_Row(src, tgt, "script_tgt"))
    return rows


def _lid_rows() -> list[_Row]:
    rows = []
    for index in range(PLANTED_REASON_COUNTS["lid_src"]):
        rng = _rng("lid_src", index)
        src_length, tgt_length = _balanced_lengths(rng)
        src = _marked_line(rng, pt.RUS_SYLLABLES, _RUS_MARKERS, src_length)
        rows.append(_Row(src, _eng_line(rng, tgt_length), "lid_src"))
    for index in range(PLANTED_REASON_COUNTS["lid_tgt"]):
        rng = _rng("lid_tgt", index)
        src_length, _ = _balanced_lengths(rng)
        # Shorter than the source: umlauts cost two byte tokens each.
        tgt = _marked_line(
            rng,
            pt.DEU_SYLLABLES,
            _DEU_MARKERS,
            round(src_length * 0.85),
            min_marked=3,
            max_syllables=2,
        )
        rows.append(_Row(_khk_line(rng, src_length), tgt, "lid_tgt"))
    return rows


def _length_rows() -> list[_Row]:
    return [*_long_target_rows(), *_short_target_rows(), *_fragmenting_rows()]


def _long_target_rows() -> list[_Row]:
    """Target 2.4x and 2.6x the source. The source is written with the soft
    sign, which the tiny SentencePiece model never saw, so its token count
    grows with its length and the token ratio stays inside the bounds."""
    rows = []
    for index, ratio in enumerate((2.4, 2.6)):
        rng = _rng("len_ratio_char_long", index)
        src_length = rng.randint(22, 28)
        src = pt.make_line_of_length(
            rng, pt.RARE_KHK_SYLLABLES, src_length, max_syllables=2
        )
        tgt = _eng_line(rng, round(pt.non_space_length(src) * ratio))
        rows.append(_Row(src, tgt, "len_ratio_char"))
    return rows


def _short_target_rows() -> list[_Row]:
    """Target about a third of the source, written in rare letters so its
    token count does not shrink with it."""
    rng = _rng("len_ratio_char_short")
    src = _khk_line(rng, 52)
    tgt = pt.make_line_of_length(
        rng,
        pt.RARE_ENG_SYLLABLES,
        round(pt.non_space_length(src) * 0.35),
        max_syllables=1,
    )
    return [_Row(src, tgt, "len_ratio_char")]


def _fragmenting_rows() -> list[_Row]:
    """Target 1.6x to 1.75x the source in characters, all rare letters: about
    1.5 tokens per letter against 0.7 for the source, so 3.5 times the tokens."""
    rows = []
    for index, ratio in enumerate((1.75, 1.6)):
        rng = _rng("len_ratio_token", index)
        src = _khk_line(rng, rng.randint(34, 40), max_syllables=4)
        tgt = pt.make_line_of_length(
            rng,
            pt.RARE_ENG_SYLLABLES,
            round(pt.non_space_length(src) * ratio),
            max_syllables=1,
        )
        rows.append(_Row(src, tgt, "len_ratio_token"))
    return rows


def _number_rows() -> list[_Row]:
    mismatches = (("2024", "2025"), ("15", "150"), ("10 000", "1,000"))
    rows = []
    for index, (src_number, tgt_number) in enumerate(mismatches):
        rng = _rng("numbers", index)
        src_length, tgt_length = _balanced_lengths(rng)
        src = _insert_word(rng, _khk_line(rng, src_length), src_number)
        tgt = _insert_word(rng, _eng_line(rng, tgt_length), tgt_number)
        rows.append(_Row(src, tgt, "numbers"))
    return rows


def _labse_rows() -> list[_Row]:
    """Well-formed rows whose two sides are unrelated sentences."""
    rows = []
    for index in range(PLANTED_REASON_COUNTS["labse"]):
        rng = _rng("labse", index)
        src_length, tgt_length = _balanced_lengths(rng)
        rows.append(
            _Row(_khk_line(rng, src_length), _eng_line(rng, tgt_length), "labse")
        )
    return rows


def _interleave(clean: list[_Row], planted: list[_Row]) -> list[_Row]:
    """Shuffle the clean and planted rows together, then put each duplicate
    (a clean row plus one trailing character) after its original."""
    rng = _rng("order")
    rows = [*clean, *planted]
    rng.shuffle(rows)
    for position in _DUPLICATED_CLEAN_ROWS:
        original = clean[position]
        duplicate = _Row(original.src, original.tgt + _DUPLICATE_SUFFIX, "duplicate")
        rows.insert(rng.randint(rows.index(original) + 1, len(rows)), duplicate)
    return rows


def _separated_concept_ids() -> Iterator[int]:
    """Concept ids whose Gaussian vectors are pairwise nearly orthogonal; the
    ids 0, 1, 2, ... themselves have pairs with a cosine above 0.4."""
    chosen: list[NDArray[np.float32]] = []
    for seed in range(_MAX_ATTEMPTS):
        vector = concept_vector(seed)
        if all(abs(float(vector @ other)) < _MAX_CONCEPT_COSINE for other in chosen):
            chosen.append(vector)
            yield seed
    raise RuntimeError(
        f"only {len(chosen)} pairwise separated concept ids among the first "
        f"{_MAX_ATTEMPTS} seeds"
    )


def _concepts(rows: list[_Row]) -> dict[str, int]:
    """One concept per row, shared by its two sides, except that a labse row's
    tgt gets a concept of its own. A duplicate shares its original's src text
    and so its concept."""
    ids = _separated_concept_ids()
    concepts: dict[str, int] = {}
    for row in rows:
        if row.src not in concepts:
            concepts[row.src] = next(ids)
        concepts[row.tgt] = next(ids) if row.reason == "labse" else concepts[row.src]
    return concepts
