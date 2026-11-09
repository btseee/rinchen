"""Tests for `rinchen.train.vocabulary_size`: the size of Eole's vocabulary against the
SentencePiece model it is cut from.

Eole keeps the first `src_vocab_size` tokens of the vocabulary file, special tokens
included, and the reserved symbols are the last lines of it (`add_reserved_symbols`): a
size below the model's pieces plus Eole's pad token would cut exactly them. A size that is
not stated is Eole's default (32,768), which a model of 32,768 pieces or more outgrows and
the ablations raise themselves; models of exact sizes come from `tests.support.sized_spm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.train.vocabulary_size import (
    EOLE_DEFAULT_VOCABULARY_SIZE,
    needed_vocabulary_size,
    require_sizes_to_fit,
)
from tests.support.sized_spm import write_sized_spm
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
    require_sizes_to_fit(config_with(toy_spm, src_vocab_size=size))
    require_sizes_to_fit(config_with(toy_spm, tgt_vocab_size=size))


@pytest.mark.parametrize("size", [PIECES, PIECES - 1, 200, 1])
@pytest.mark.parametrize("key", ["src_vocab_size", "tgt_vocab_size"])
def test_a_stated_size_below_the_pieces_and_the_pad_token_is_refused_naming_both(
    toy_spm: Path, key: str, size: int
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_sizes_to_fit(config_with(toy_spm, **{key: size}))

    message = str(raised.value)
    assert message.startswith(f"invalid Eole config: {key}: {size} is below {NEEDED}")
    assert str(toy_spm) in message
    assert "reserved symbols" in message


def test_a_config_that_states_no_size_and_names_no_model_that_exists_is_accepted(
    tmp_path: Path,
) -> None:
    # Nothing to compare: Eole names a model that is missing when it builds the vocabulary.
    require_sizes_to_fit(config_with(tmp_path / "absent.model"))
    require_sizes_to_fit({})


def test_an_unstated_size_with_a_model_that_cannot_be_loaded_is_a_config_error(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.model"
    broken.write_bytes(b"not a model")

    with pytest.raises(ConfigError, match="broken.model"):
        require_sizes_to_fit(config_with(broken))


def test_a_stated_size_with_no_model_to_compare_with_is_accepted() -> None:
    require_sizes_to_fit({"src_vocab_size": 10})
    require_sizes_to_fit(
        {"src_vocab_size": 10, "transforms_configs": {"sentencepiece": {}}}
    )
    require_sizes_to_fit({"src_vocab_size": 10, "transforms_configs": []})


def test_each_size_is_compared_with_the_model_of_its_own_side(
    toy_spm: Path, tmp_path: Path
) -> None:
    cfg = config_with(toy_spm, src_vocab_size=NEEDED, tgt_vocab_size=NEEDED)
    cfg["transforms_configs"]["sentencepiece"]["tgt_subword_model"] = str(
        tmp_path / "absent.model"
    )

    with pytest.raises(ConfigError, match="absent.model"):
        require_sizes_to_fit(cfg)


@pytest.mark.parametrize("size", [0, -5, 1.5, "641", True, [641]], ids=repr)
def test_a_stated_size_that_is_not_a_positive_integer_is_refused(
    toy_spm: Path, size: object
) -> None:
    with pytest.raises(
        ConfigError,
        match="^invalid Eole config: src_vocab_size: must be a positive integer",
    ):
        require_sizes_to_fit(config_with(toy_spm, src_vocab_size=size))


def test_a_size_of_none_is_not_stated(tmp_path: Path) -> None:
    require_sizes_to_fit(config_with(tmp_path / "absent.model", src_vocab_size=None))


def test_a_stated_size_with_a_model_that_cannot_be_loaded_is_a_config_error(
    tmp_path: Path,
) -> None:
    cfg = config_with(tmp_path / "absent.model", src_vocab_size=1000)

    with pytest.raises(ConfigError, match="absent.model"):
        require_sizes_to_fit(cfg)


# ---- a size that is not stated is Eole's default ----------------------------------

DEFAULT = 32768  # `src_vocab_size` and `tgt_vocab_size` of Eole 0.6.4 (config/data.py)


def test_the_default_is_the_one_of_the_installed_eole() -> None:
    from eole.config.data import VocabConfig

    assert EOLE_DEFAULT_VOCABULARY_SIZE == DEFAULT
    for key in ("src_vocab_size", "tgt_vocab_size"):
        assert VocabConfig.model_fields[key].default == EOLE_DEFAULT_VOCABULARY_SIZE


def test_an_unstated_size_is_refused_when_the_default_would_cut_the_vocabulary(
    tmp_path: Path,
) -> None:
    # 32,768 pieces and the pad token need 32,769 tokens: one more than the default.
    big = write_sized_spm(tmp_path / "big.model", DEFAULT)

    with pytest.raises(ConfigError) as raised:
        require_sizes_to_fit(config_with(big))

    message = str(raised.value)
    assert message.startswith("invalid Eole config: src_vocab_size: not stated")
    assert f"Eole's default of {DEFAULT}" in message
    assert str(big) in message
    assert "32768 pieces" in message
    assert "reserved symbols" in message
    assert message.endswith(
        "state `src_vocab_size: 32769` at the top level of the config"
    )


def test_the_target_size_is_the_one_named_when_only_it_is_left_out(
    tmp_path: Path,
) -> None:
    big = write_sized_spm(tmp_path / "big.model", 128_000)

    with pytest.raises(ConfigError) as raised:
        require_sizes_to_fit(config_with(big, src_vocab_size=128_001))

    message = str(raised.value)
    assert message.startswith("invalid Eole config: tgt_vocab_size: not stated")
    assert "state `tgt_vocab_size: 128001`" in message


def test_an_unstated_size_is_accepted_when_the_default_holds_every_piece(
    tmp_path: Path,
) -> None:
    # 32,767 pieces and the pad token are exactly the 32,768 tokens the default keeps.
    fits = write_sized_spm(tmp_path / "fits.model", DEFAULT - 1)

    require_sizes_to_fit(config_with(fits))


def test_the_tiny_models_need_no_size_stated(toy_spm: Path) -> None:
    require_sizes_to_fit(config_with(toy_spm))


@pytest.mark.parametrize("size", [128_001, 200_000])
def test_a_size_stated_above_the_default_for_a_big_model_is_accepted(
    tmp_path: Path, size: int
) -> None:
    big = write_sized_spm(tmp_path / "big.model", 128_000)

    require_sizes_to_fit(config_with(big, src_vocab_size=size, tgt_vocab_size=size))


def test_an_unstated_target_size_is_not_needed_when_the_vocabulary_is_shared(
    tmp_path: Path,
) -> None:
    # Eole builds the target vocabulary from the source one then and ignores
    # `tgt_vocab_size`; the ablations state `src_vocab_size` alone.
    big = write_sized_spm(tmp_path / "big.model", 128_000)

    require_sizes_to_fit(config_with(big, src_vocab_size=128_001, share_vocab=True))


@pytest.mark.parametrize("share_vocab", [False, None])
def test_an_unstated_target_size_is_needed_when_the_vocabulary_is_not_shared(
    tmp_path: Path, share_vocab: bool | None
) -> None:
    big = write_sized_spm(tmp_path / "big.model", 128_000)
    cfg = config_with(big, src_vocab_size=128_001, share_vocab=share_vocab)

    with pytest.raises(ConfigError, match="tgt_vocab_size: not stated"):
        require_sizes_to_fit(cfg)


def test_a_stated_target_size_is_still_checked_when_the_vocabulary_is_shared(
    tmp_path: Path,
) -> None:
    big = write_sized_spm(tmp_path / "big.model", 128_000)
    cfg = config_with(big, src_vocab_size=128_001, tgt_vocab_size=100, share_vocab=True)

    with pytest.raises(ConfigError, match="tgt_vocab_size: 100 is below 128001"):
        require_sizes_to_fit(cfg)


def test_an_unstated_source_size_is_needed_even_when_the_vocabulary_is_shared(
    tmp_path: Path,
) -> None:
    big = write_sized_spm(tmp_path / "big.model", 128_000)

    with pytest.raises(ConfigError, match="src_vocab_size: not stated"):
        require_sizes_to_fit(config_with(big, share_vocab=True))
