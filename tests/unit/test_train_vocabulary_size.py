"""Tests for `rinchen.train.vocabulary_size`: the size of Eole's vocabulary against the
SentencePiece model it is cut from.

Eole keeps the first `src_vocab_size` tokens of the vocabulary file, special tokens
included, and the reserved symbols are the last lines of it (`add_reserved_symbols`): a
stated size below the model's pieces plus Eole's pad token would cut exactly them. A size
that is not stated is Eole's default (32,768), which the ablations raise themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.train.vocabulary_size import (
    needed_vocabulary_size,
    require_stated_sizes_to_fit,
)
from tests.support.tiny_parallel import train_toy_spm

PIECES = 640  # the toy model's
NEEDED = PIECES + 1  # and Eole's pad token


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("vocabulary_size_spm"))


def config_with(spm: Path | None, **keys: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(keys)
    if spm is not None:
        cfg["transforms_configs"] = {
            "sentencepiece": {
                "src_subword_model": str(spm),
                "tgt_subword_model": str(spm),
            }
        }
    return cfg


def test_the_size_needed_is_the_pieces_and_the_pad_token(toy_spm: Path) -> None:
    assert needed_vocabulary_size(toy_spm) == NEEDED


@pytest.mark.parametrize("size", [NEEDED, NEEDED + 1, 32768])
def test_a_stated_size_that_holds_every_piece_is_accepted(
    toy_spm: Path, size: int
) -> None:
    require_stated_sizes_to_fit(config_with(toy_spm, src_vocab_size=size))
    require_stated_sizes_to_fit(config_with(toy_spm, tgt_vocab_size=size))


@pytest.mark.parametrize("size", [PIECES, PIECES - 1, 200, 1])
@pytest.mark.parametrize("key", ["src_vocab_size", "tgt_vocab_size"])
def test_a_stated_size_below_the_pieces_and_the_pad_token_is_refused_naming_both(
    toy_spm: Path, key: str, size: int
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_stated_sizes_to_fit(config_with(toy_spm, **{key: size}))

    message = str(raised.value)
    assert message.startswith(f"invalid Eole config: {key}: {size} is below {NEEDED}")
    assert str(toy_spm) in message
    assert "reserved symbols" in message


def test_a_config_that_states_no_size_is_accepted_whatever_its_model(
    tmp_path: Path,
) -> None:
    # Nothing is read: the model need not exist.
    require_stated_sizes_to_fit(config_with(tmp_path / "absent.model"))
    require_stated_sizes_to_fit({})


def test_a_stated_size_with_no_model_to_compare_with_is_accepted() -> None:
    require_stated_sizes_to_fit({"src_vocab_size": 10})
    require_stated_sizes_to_fit(
        {"src_vocab_size": 10, "transforms_configs": {"sentencepiece": {}}}
    )
    require_stated_sizes_to_fit({"src_vocab_size": 10, "transforms_configs": []})


def test_each_size_is_compared_with_the_model_of_its_own_side(
    toy_spm: Path, tmp_path: Path
) -> None:
    cfg = config_with(toy_spm, src_vocab_size=NEEDED, tgt_vocab_size=NEEDED)
    cfg["transforms_configs"]["sentencepiece"]["tgt_subword_model"] = str(
        tmp_path / "absent.model"
    )

    with pytest.raises(ConfigError, match="absent.model"):
        require_stated_sizes_to_fit(cfg)


@pytest.mark.parametrize("size", [0, -5, 1.5, "641", True, [641]], ids=repr)
def test_a_stated_size_that_is_not_a_positive_integer_is_refused(
    toy_spm: Path, size: object
) -> None:
    with pytest.raises(
        ConfigError,
        match="^invalid Eole config: src_vocab_size: must be a positive integer",
    ):
        require_stated_sizes_to_fit(config_with(toy_spm, src_vocab_size=size))


def test_a_size_of_none_is_not_stated(tmp_path: Path) -> None:
    require_stated_sizes_to_fit(
        config_with(tmp_path / "absent.model", src_vocab_size=None)
    )


def test_a_stated_size_with_a_model_that_cannot_be_loaded_is_a_config_error(
    tmp_path: Path,
) -> None:
    cfg = config_with(tmp_path / "absent.model", src_vocab_size=1000)

    with pytest.raises(ConfigError, match="absent.model"):
        require_stated_sizes_to_fit(cfg)
