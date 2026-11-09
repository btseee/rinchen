"""How large Eole's vocabulary must be for the SentencePiece model it is cut from.

Eole keeps the first `src_vocab_size` tokens of the vocabulary file, special tokens
included, and its default is 32,768: a SentencePiece model of 128,000 pieces would lose
the rarest of them to `<unk>`, and the reserved symbols that
`eole_run.add_reserved_symbols` appends at the END of the file would be cut first. So the
size has to be at least the model's pieces plus Eole's own pad token (`<blank>`, which is
no piece). The ablations set it themselves (`ablation_config`); a launch refuses a config
that states a smaller one and one that states none while Eole's default is smaller than the
size needed (`require_sizes_to_fit`). It never sets a size itself: the operator states it
in the config, and the refusal says which value.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from rinchen.errors import ConfigError
from rinchen.train.validation_corpus import load_processor
from rinchen.utils.integers import is_strict_int

# Eole adds `<blank>`, its pad token, to the vocabulary of every model, and a SentencePiece
# model has no such piece (it has no pad piece), so a vocabulary of all the pieces holds
# one token more than the model has pieces.
EOLE_EXTRA_TOKENS: Final = 1
# What Eole uses when a config states no size: the default of `src_vocab_size` and of
# `tgt_vocab_size` in `eole.config.data.VocabConfig` (Eole 0.6.4); a drift test compares it
# with the installed Eole.
EOLE_DEFAULT_VOCABULARY_SIZE: Final = 32768
VOCABULARY_SIZE_KEYS: Final = ("src_vocab_size", "tgt_vocab_size")
# The model each size is compared with: the one that cuts the text of its side.
_MODEL_KEY: Final = {
    "src_vocab_size": "src_subword_model",
    "tgt_vocab_size": "tgt_subword_model",
}


def needed_vocabulary_size(spm_model: Path) -> int:
    """The smallest `src_vocab_size` that keeps every piece of `spm_model` and Eole's pad
    token.

    Raises:
        ConfigError: the model cannot be loaded.
    """
    return load_processor(spm_model).GetPieceSize() + EOLE_EXTRA_TOKENS


def require_vocabulary_size(
    key: str, size: object, needed: int, spm_model: Path
) -> None:
    """Refuse a stated `size` (of the config key `key`) that is not a positive integer
    or is below `needed`, the size `needed_vocabulary_size(spm_model)` gave.

    Raises:
        ConfigError: as above; the message names both numbers and the model.
    """
    if not is_strict_int(size) or size < 1:
        raise ConfigError(
            f"invalid Eole config: {key}: must be a positive integer, got {size!r}"
        )
    if size < needed:
        pieces = needed - EOLE_EXTRA_TOKENS
        raise ConfigError(
            f"invalid Eole config: {key}: {size} is below {needed}, what the "
            f"vocabulary of the SentencePiece model {spm_model} needs ({pieces} pieces "
            "and Eole's pad token): Eole would cut the reserved symbols that are added "
            "at the end of its vocabulary file"
        )


def require_unstated_size_to_fit(key: str, needed: int, spm_model: Path) -> None:
    """Refuse an unstated `key` when Eole's default size is below `needed`, the size
    `needed_vocabulary_size(spm_model)` gave: the message names the value to state.

    Raises:
        ConfigError: as above; the message names the key, both numbers and the model.
    """
    if EOLE_DEFAULT_VOCABULARY_SIZE >= needed:
        return
    pieces = needed - EOLE_EXTRA_TOKENS
    raise ConfigError(
        f"invalid Eole config: {key}: not stated, so Eole's default of "
        f"{EOLE_DEFAULT_VOCABULARY_SIZE} applies, which is below {needed}, what the "
        f"vocabulary of the SentencePiece model {spm_model} needs ({pieces} pieces and "
        "Eole's pad token): Eole would cut the reserved symbols that are added at the "
        f"end of its vocabulary file; state `{key}: {needed}` at the top level of the "
        "config"
    )


def _is_left_to_eole(cfg: Mapping[str, Any], key: str, spm_model: Path) -> bool:
    """Whether an unstated `key` has nothing to be compared with: Eole ignores
    `tgt_vocab_size` when the vocabulary is shared (the target one is the source one),
    and names a model that is not a file itself, when it builds the vocabulary."""
    shared_target = key == "tgt_vocab_size" and cfg.get("share_vocab") is True
    return shared_target or not spm_model.is_file()


def require_sizes_to_fit(cfg: Mapping[str, Any]) -> None:
    """Refuse a config whose `src_vocab_size` or `tgt_vocab_size` is below what the
    SentencePiece model of its side needs (`transforms_configs.sentencepiece`
    `src_subword_model` or `tgt_subword_model`): a size it states, or Eole's default when
    it states none. A side with no model stated has nothing to compare its size with, and
    neither has an unstated size whose model is not a file (Eole reports that itself, when
    it builds the vocabulary) or an unstated `tgt_vocab_size` of a shared vocabulary
    (`share_vocab: true`, which Eole ignores it for); a stated size needs a model that
    loads.

    Raises:
        ConfigError: a stated size is not a positive integer or is too small, an
            unstated one is below what the model needs, or a model that is needed
            cannot be loaded.
    """
    configs = cfg.get("transforms_configs")
    section = configs.get("sentencepiece") if isinstance(configs, Mapping) else None
    models = section if isinstance(section, Mapping) else {}
    for key in VOCABULARY_SIZE_KEYS:
        size = cfg.get(key)
        model = models.get(_MODEL_KEY[key])
        if not isinstance(model, str) or not model:
            continue
        spm_model = Path(model)
        if size is None and _is_left_to_eole(cfg, key, spm_model):
            continue
        needed = needed_vocabulary_size(spm_model)
        if size is None:
            require_unstated_size_to_fit(key, needed, spm_model)
        else:
            require_vocabulary_size(key, size, needed, spm_model)
