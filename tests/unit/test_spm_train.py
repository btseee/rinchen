"""Tests for `rinchen.spm.train.train_spm`, which drives the real SentencePiece
trainer on the tiny synthetic fixture corpus.

Every run writes under pytest's `tmp_path` only: the tiny config's own
`output_prefix` (under `data/`) is always replaced.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import sentencepiece

from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import SpmConfig, load_spm_config
from rinchen.errors import ConfigError, UpstreamError
from rinchen.spm.symbols import user_defined_symbols
from rinchen.spm.train import train_spm

TINY_CONFIG_PATH = REPO_ROOT / "configs" / "spm" / "tiny_1k.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"

TINY_VOCAB_SIZE = 1000
TINY_LINE_COUNT = 800
TRAINING_TIME_BUDGET_SECONDS = 10.0


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> SpmConfig:
    return load_spm_config(TINY_CONFIG_PATH).model_copy(
        update={"output_prefix": tmp_path / "tiny"}
    )


def _sampled_file(cfg: SpmConfig) -> Path:
    return Path(f"{cfg.output_prefix}.sampled.txt")


def _fixture_language_of_each_line() -> dict[str, str]:
    """Map every fixture line to its language; the fixture has no line shared
    by two languages, which the assertion below would otherwise flag."""
    owner: dict[str, str] = {}
    for path in sorted(FIXTURE_DIR.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert owner.setdefault(line, path.stem) == path.stem, line
    return owner


def test_train_tiny_returns_model_with_1000_pieces(tiny_cfg: SpmConfig) -> None:
    model_path = train_spm(tiny_cfg)

    assert model_path == Path(f"{tiny_cfg.output_prefix}.model")
    assert model_path.suffix == ".model"
    assert model_path.is_file()
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    assert processor.get_piece_size() == TINY_VOCAB_SIZE


def test_vocab_file_is_written_next_to_the_model(tiny_cfg: SpmConfig) -> None:
    train_spm(tiny_cfg)

    assert Path(f"{tiny_cfg.output_prefix}.vocab").is_file()


def test_sampled_training_file_holds_the_whole_small_corpus(
    tiny_cfg: SpmConfig,
) -> None:
    train_spm(tiny_cfg)

    sampled = _sampled_file(tiny_cfg)
    assert sampled.is_file()
    lines = sampled.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= tiny_cfg.input_sentence_size
    assert len(lines) == TINY_LINE_COUNT


def test_sampled_training_file_is_utf8_with_lf_endings(tiny_cfg: SpmConfig) -> None:
    train_spm(tiny_cfg)

    raw = _sampled_file(tiny_cfg).read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    raw.decode("utf-8")


def test_tiny_training_finishes_under_ten_seconds(tiny_cfg: SpmConfig) -> None:
    started = time.perf_counter()
    train_spm(tiny_cfg)
    elapsed = time.perf_counter() - started

    assert elapsed < TRAINING_TIME_BUDGET_SECONDS


def test_trained_model_knows_every_user_defined_symbol(tiny_cfg: SpmConfig) -> None:
    model_path = train_spm(tiny_cfg)

    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    unknown = processor.unk_id()
    symbols = user_defined_symbols()
    missing = [s for s in symbols if processor.piece_to_id(s) == unknown]
    assert len(symbols) == 118
    assert missing == []


def test_over_budget_corpus_is_sampled_to_the_budget_across_all_languages(
    tiny_cfg: SpmConfig,
) -> None:
    budget = 500
    cfg = tiny_cfg.model_copy(update={"input_sentence_size": budget})

    train_spm(cfg)

    sampled_lines = _sampled_file(cfg).read_text(encoding="utf-8").splitlines()
    owner = _fixture_language_of_each_line()
    languages_seen = {owner[line] for line in sampled_lines}
    assert len(sampled_lines) <= budget
    assert languages_seen == {"khk_Cyrl", "rus_Cyrl", "eng_Latn", "mn_Mong"}


def test_over_budget_corpus_still_trains_a_1000_piece_model(
    tiny_cfg: SpmConfig,
) -> None:
    cfg = tiny_cfg.model_copy(update={"input_sentence_size": 500})

    model_path = train_spm(cfg)

    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    assert processor.get_piece_size() == TINY_VOCAB_SIZE


def test_missing_output_directory_is_created(
    tiny_cfg: SpmConfig, tmp_path: Path
) -> None:
    nested_prefix = tmp_path / "a" / "b" / "tiny"
    cfg = tiny_cfg.model_copy(update={"output_prefix": nested_prefix})

    model_path = train_spm(cfg)

    assert model_path == nested_prefix.parent / "tiny.model"
    assert model_path.is_file()


def test_vocab_too_large_raises_upstream_error_with_cause(
    tiny_cfg: SpmConfig,
) -> None:
    cfg = tiny_cfg.model_copy(update={"vocab_size": 50000})

    with pytest.raises(UpstreamError, match=str(cfg.output_prefix)) as excinfo:
        train_spm(cfg)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "Vocabulary size too high" in str(excinfo.value)


def test_missing_corpus_dir_raises_config_error(
    tiny_cfg: SpmConfig, tmp_path: Path
) -> None:
    cfg = tiny_cfg.model_copy(update={"corpus_dir": tmp_path / "no_such_corpus"})

    with pytest.raises(ConfigError, match="no_such_corpus"):
        train_spm(cfg)

    assert not _sampled_file(cfg).exists()
