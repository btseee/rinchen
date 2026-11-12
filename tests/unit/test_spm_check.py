"""Tests for `rinchen.spm.check.check_spm`, which validates a trained
SentencePiece model against the project invariants (every reserved symbol is a
single piece, byte fallback is on) and reports per-language coverage.

Models are trained once per module on the tiny synthetic fixture corpus, and
every file lands under pytest's temporary directories only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sentencepiece

from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import load_spm_config
from rinchen.errors import DataError
from rinchen.spm.check import LanguageCoverage, SpmCheckReport, check_spm
from rinchen.spm.corpus import discover_corpus
from rinchen.spm.symbols import user_defined_symbols
from rinchen.spm.train import (
    MODEL_TYPE,
    NORMALIZATION_RULE_NAME,
    SPLIT_DIGITS,
    train_spm,
)

TINY_CONFIG_PATH = REPO_ROOT / "configs" / "spm" / "tiny_1k.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"
FIXTURE_LANGUAGES = {"khk_Cyrl", "rus_Cyrl", "eng_Latn", "mn_Mong"}

# Measured on the tiny model: the fixture covers every digit and character, so
# no piece of any fixture language falls back.
MIN_FIXTURE_COVERAGE = 0.95

PLACEHOLDER_ZERO = chr(0x27E6) + "0" + chr(0x27E7)
PLACEHOLDER_COUNT = 100
# Same size as the production tiny model, which the fixture corpus supports even
# without the 256 byte pieces.
VARIANT_VOCAB_SIZE = 1000
# Room for the 118 reserved symbols plus 256 byte-named symbols.
BYTE_NAMED_VOCAB_SIZE = 800
# Han script, which no fixture language uses.
CHINESE_LINE = "中文中文"


def _train_variant(
    sampled_path: Path,
    prefix: Path,
    *,
    byte_fallback: bool,
    symbols: list[str],
    add_dummy_prefix: bool = True,
    vocab_size: int = VARIANT_VOCAB_SIZE,
) -> Path:
    """Train a model the way `train_spm` does, except for the one setting a
    test needs to change."""
    sentencepiece.SentencePieceTrainer.train(
        input=str(sampled_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type=MODEL_TYPE,
        byte_fallback=byte_fallback,
        split_digits=SPLIT_DIGITS,
        normalization_rule_name=NORMALIZATION_RULE_NAME,
        user_defined_symbols=symbols,
        add_dummy_prefix=add_dummy_prefix,
        shuffle_input_sentence=True,
        minloglevel=2,
    )
    return Path(f"{prefix}.model")


def _write_corpus(directory: Path, name: str, lines: list[str]) -> Path:
    path = directory / f"{name}.txt"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _fixture_lines(language: str) -> list[str]:
    path = FIXTURE_DIR / f"{language}.txt"
    return path.read_text(encoding="utf-8").splitlines()


def _piece_count(model_path: Path, lines: list[str]) -> int:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    return sum(len(ids) for ids in processor.encode(lines, out_type=int))


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = load_spm_config(TINY_CONFIG_PATH).model_copy(
        update={"output_prefix": tmp_path_factory.mktemp("spm_check") / "tiny"}
    )
    return train_spm(cfg)


@pytest.fixture(scope="module")
def tiny_corpus() -> dict[str, Path]:
    return discover_corpus(FIXTURE_DIR)


@pytest.fixture(scope="module")
def model_without_byte_fallback(
    tiny_model: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return _train_variant(
        tiny_model.with_suffix(".sampled.txt"),
        tmp_path_factory.mktemp("spm_check_no_bytes") / "no_bytes",
        byte_fallback=False,
        symbols=user_defined_symbols(),
    )


@pytest.fixture(scope="module")
def model_without_symbols(
    tiny_model: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return _train_variant(
        tiny_model.with_suffix(".sampled.txt"),
        tmp_path_factory.mktemp("spm_check_no_symbols") / "no_symbols",
        byte_fallback=True,
        symbols=[],
    )


@pytest.fixture(scope="module")
def model_without_placeholders(
    tiny_model: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return _train_variant(
        tiny_model.with_suffix(".sampled.txt"),
        tmp_path_factory.mktemp("spm_check_no_placeholders") / "no_placeholders",
        byte_fallback=True,
        symbols=user_defined_symbols()[:-PLACEHOLDER_COUNT],
    )


@pytest.fixture(scope="module")
def model_without_dummy_prefix(
    tiny_model: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return _train_variant(
        tiny_model.with_suffix(".sampled.txt"),
        tmp_path_factory.mktemp("spm_check_no_prefix") / "no_prefix",
        byte_fallback=True,
        symbols=user_defined_symbols(),
        add_dummy_prefix=False,
    )


@pytest.fixture(scope="module")
def model_with_byte_named_symbols_only(
    tiny_model: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    byte_named = [f"<0x{value:02X}>" for value in range(256)]
    return _train_variant(
        tiny_model.with_suffix(".sampled.txt"),
        tmp_path_factory.mktemp("spm_check_byte_named") / "byte_named",
        byte_fallback=False,
        symbols=[*user_defined_symbols(), *byte_named],
        vocab_size=BYTE_NAMED_VOCAB_SIZE,
    )


def test_tiny_model_passes_all_checks(
    tiny_model: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(tiny_model, tiny_corpus)

    assert report.ok
    assert report.non_atomic_symbols == ()
    assert report.byte_fallback_on
    assert {entry.lang for entry in report.coverage} == FIXTURE_LANGUAGES
    for entry in report.coverage:
        assert entry.pieces > 0, entry.lang
        assert entry.coverage >= MIN_FIXTURE_COVERAGE, entry.lang


def test_detects_byte_fallback_off(
    model_without_byte_fallback: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(model_without_byte_fallback, tiny_corpus)

    assert report.byte_fallback_on is False
    assert not report.ok
    assert report.non_atomic_symbols == ()


def test_pieces_named_like_bytes_are_not_byte_fallback(
    model_with_byte_named_symbols_only: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(model_with_byte_named_symbols_only, tiny_corpus)

    assert report.byte_fallback_on is False
    assert not report.ok


def test_symbols_are_atomic_without_a_dummy_prefix(
    model_without_dummy_prefix: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(model_without_dummy_prefix, tiny_corpus)

    assert report.non_atomic_symbols == ()
    assert report.ok


def test_detects_non_atomic_symbols(
    model_without_symbols: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(model_without_symbols, tiny_corpus)

    assert "<2eng_Latn>" in report.non_atomic_symbols
    assert PLACEHOLDER_ZERO in report.non_atomic_symbols
    assert not report.ok
    assert report.byte_fallback_on


def test_non_atomic_symbols_follow_reserved_symbol_order_and_leave_out_atomic_ones(
    model_without_placeholders: Path, tiny_corpus: dict[str, Path]
) -> None:
    report = check_spm(model_without_placeholders, tiny_corpus)

    assert report.non_atomic_symbols == tuple(
        user_defined_symbols()[-PLACEHOLDER_COUNT:]
    )


def test_out_of_vocabulary_script_counts_as_fallback(
    tiny_model: Path, tmp_path: Path
) -> None:
    chinese = _write_corpus(tmp_path, "zho_Hans", [CHINESE_LINE] * 3)

    report = check_spm(tiny_model, {"zho_Hans": chinese})

    (entry,) = report.coverage
    assert entry.lang == "zho_Hans"
    assert entry.fallback_pieces > 0
    assert entry.coverage < 0.5


def test_unknown_pieces_count_as_fallback(
    model_without_byte_fallback: Path, tmp_path: Path
) -> None:
    lines = [CHINESE_LINE] * 3
    chinese = _write_corpus(tmp_path, "zho_Hans", lines)

    report = check_spm(model_without_byte_fallback, {"zho_Hans": chinese})

    (entry,) = report.coverage
    # Each line is the dummy prefix plus one merged unknown piece.
    assert entry.pieces == 2 * len(lines)
    assert entry.fallback_pieces == len(lines)


def test_sample_lines_limits_work(
    tiny_model: Path, tiny_corpus: dict[str, Path]
) -> None:
    default_report = check_spm(tiny_model, tiny_corpus)
    sampled_report = check_spm(tiny_model, tiny_corpus, sample_lines=5)

    for sampled, full in zip(
        sampled_report.coverage, default_report.coverage, strict=True
    ):
        assert sampled.lang == full.lang
        assert sampled.pieces < full.pieces, sampled.lang


@pytest.mark.parametrize("sample_lines", [1, 5, 50])
def test_coverage_counts_the_pieces_of_exactly_the_first_lines(
    tiny_model: Path, tiny_corpus: dict[str, Path], sample_lines: int
) -> None:
    report = check_spm(
        tiny_model, {"khk_Cyrl": tiny_corpus["khk_Cyrl"]}, sample_lines=sample_lines
    )

    (entry,) = report.coverage
    expected = _piece_count(tiny_model, _fixture_lines("khk_Cyrl")[:sample_lines])
    assert entry.pieces == expected


def test_sample_lines_beyond_the_file_length_use_every_line(
    tiny_model: Path, tiny_corpus: dict[str, Path]
) -> None:
    lines = _fixture_lines("eng_Latn")

    report = check_spm(
        tiny_model, {"eng_Latn": tiny_corpus["eng_Latn"]}, sample_lines=len(lines) + 900
    )

    (entry,) = report.coverage
    assert entry.pieces == _piece_count(tiny_model, lines)


def test_blank_lines_do_not_count_toward_sample_lines(
    tiny_model: Path, tmp_path: Path
) -> None:
    first, second, third = _fixture_lines("eng_Latn")[:3]
    corpus = tmp_path / "eng_Latn.txt"
    corpus.write_text(f"\n{first}\n\n\n{second}\n\n{third}\n", encoding="utf-8")

    report = check_spm(tiny_model, {"eng_Latn": corpus}, sample_lines=2)

    (entry,) = report.coverage
    assert entry.pieces == _piece_count(tiny_model, [first, second])


def test_empty_corpus_file_reports_no_pieces_and_full_coverage(
    tiny_model: Path, tmp_path: Path
) -> None:
    empty = _write_corpus(tmp_path, "eng_Latn", [])

    report = check_spm(tiny_model, {"eng_Latn": empty})

    (entry,) = report.coverage
    assert entry.pieces == 0
    assert entry.fallback_pieces == 0
    assert entry.coverage == 1.0


def test_coverage_follows_the_corpus_mapping_order(
    tiny_model: Path, tiny_corpus: dict[str, Path]
) -> None:
    mapping = {
        "rus_Cyrl": tiny_corpus["rus_Cyrl"],
        "eng_Latn": tiny_corpus["eng_Latn"],
        "mn_Mong": tiny_corpus["mn_Mong"],
        "khk_Cyrl": tiny_corpus["khk_Cyrl"],
    }

    report = check_spm(tiny_model, mapping)

    assert tuple(entry.lang for entry in report.coverage) == tuple(mapping)


def test_an_empty_corpus_mapping_yields_no_coverage(tiny_model: Path) -> None:
    report = check_spm(tiny_model, {})

    assert report.coverage == ()
    assert report.ok


@pytest.mark.parametrize(
    ("pieces", "fallback_pieces", "expected"),
    [
        (0, 0, 1.0),
        (10, 0, 1.0),
        (10, 3, 0.7),
        (8, 2, 0.75),
        (4, 4, 0.0),
    ],
)
def test_language_coverage_is_the_share_of_pieces_that_do_not_fall_back(
    pieces: int, fallback_pieces: int, expected: float
) -> None:
    entry = LanguageCoverage("khk_Cyrl", pieces=pieces, fallback_pieces=fallback_pieces)

    assert entry.coverage == pytest.approx(expected)


@pytest.mark.parametrize(
    ("non_atomic_symbols", "byte_fallback_on", "expected_ok"),
    [
        ((), True, True),
        (("<t>",), True, False),
        ((), False, False),
        (("<t>",), False, False),
    ],
)
def test_report_is_ok_only_when_both_checks_pass(
    non_atomic_symbols: tuple[str, ...], byte_fallback_on: bool, expected_ok: bool
) -> None:
    report = SpmCheckReport(
        non_atomic_symbols=non_atomic_symbols,
        byte_fallback_on=byte_fallback_on,
        coverage=(),
    )

    assert report.ok is expected_ok


@pytest.mark.parametrize(
    ("content", "cause_type"),
    [
        # SentencePiece 0.2.2 raises RuntimeError for a missing file too (0.2.1: OSError).
        (None, RuntimeError),
        (b"this is not a sentencepiece model", RuntimeError),
        (b"", RuntimeError),
    ],
    ids=["missing_file", "garbage_file", "empty_file"],
)
def test_unloadable_model_raises_data_error_naming_the_path_with_cause(
    tmp_path: Path,
    tiny_corpus: dict[str, Path],
    content: bytes | None,
    cause_type: type[Exception],
) -> None:
    model_path = tmp_path / "bad.model"
    if content is not None:
        model_path.write_bytes(content)

    with pytest.raises(DataError) as excinfo:
        check_spm(model_path, tiny_corpus)

    assert f"cannot load SentencePiece model {model_path}" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, cause_type)


def test_model_path_that_is_a_directory_raises_data_error_instead_of_aborting(
    tmp_path: Path, tiny_corpus: dict[str, Path]
) -> None:
    # SentencePiece's loader terminates the interpreter on a directory.
    with pytest.raises(DataError, match=re.escape(str(tmp_path))):
        check_spm(tmp_path, tiny_corpus)


@pytest.mark.parametrize("sample_lines", [0, -1])
def test_non_positive_sample_lines_raises_value_error(
    tiny_model: Path, tiny_corpus: dict[str, Path], sample_lines: int
) -> None:
    with pytest.raises(ValueError, match="sample_lines"):
        check_spm(tiny_model, tiny_corpus, sample_lines=sample_lines)


def test_undecodable_corpus_file_raises_data_error_naming_the_file(
    tiny_model: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "eng_Latn.txt"
    broken.write_bytes(b"fine line\n\xff\xfe not utf-8\n")

    with pytest.raises(DataError, match=re.escape(str(broken))):
        check_spm(tiny_model, {"eng_Latn": broken})


def test_missing_corpus_file_raises_data_error_naming_the_file(
    tiny_model: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "eng_Latn.txt"

    with pytest.raises(DataError, match=re.escape(str(missing))):
        check_spm(tiny_model, {"eng_Latn": missing})
