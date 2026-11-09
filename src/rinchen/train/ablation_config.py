"""The Eole config of one variant: the values the run sets (seed, vocabulary, SentencePiece
model, corpora with their weights, model path) laid under the matrix's overrides, over
the base config.

The plan builds it in memory to check it before anything is written; the preparation of
a variant builds it from the corpora it wrote and renders it to a file. Both go through
`variant_overrides`, so the config that was checked is the config that is written.

The config also states the size of the vocabulary. Eole keeps the first `src_vocab_size`
tokens of the vocabulary file, special tokens included, and its default is 32,768: a
SentencePiece model of 128,000 pieces would lose the rarest of them to `<unk>`, and the
reserved symbols that `eole_run.add_reserved_symbols` appends at the END of the file would
be cut first. So the size is the model's pieces plus Eole's own pad token (`<blank>`, which
is no piece), and a smaller one that the base or the overrides ask for is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rinchen.config.ablation import AblationMatrix
from rinchen.train.ablation_layout import VariantPaths
from rinchen.train.eole_cfg import merged_config, read_config
from rinchen.train.eole_data import CorpusEntry, data_section
from rinchen.train.variants import Variant
from rinchen.train.vocabulary_size import (
    VOCABULARY_SIZE_KEYS,
    needed_vocabulary_size,
    require_vocabulary_size,
)


def variant_overrides(
    matrix: AblationMatrix,
    variant: Variant,
    *,
    spm_model: Path,
    paths: VariantPaths,
    corpora: Sequence[CorpusEntry],
    weights: Mapping[str, int],
) -> dict[str, Any]:
    """What is laid over the base config for `variant`: the per-run values, then the
    matrix overrides over them. The per-run values are the variant's seed, the
    `vocab.src` and `ckpt` of its run directory, `spm_model` for both sides, and the
    `data` section of `corpora` with `weights` (running `filtertoolong` when the
    finished config configures it), every path as absolute POSIX text. The finished
    config also has `src_vocab_size`: the one the base or the overrides give when it
    holds every token of the SentencePiece model, else (when none is given) the number
    of its pieces plus one for Eole's pad token.

    Raises:
        ConfigError: the base config cannot be read, a path holds `$`, the SentencePiece
            model cannot be loaded, or a `src_vocab_size` or `tgt_vocab_size` the base
            or the overrides give is not a positive integer or is below what the
            model's vocabulary needs.
        ValueError: `data_section` refuses the corpora or the weights.
    """
    base = read_config(matrix.base, "base Eole config")
    data = data_section(
        corpora,
        weights,
        variant.tag_placement,
        valid_paths=paths.valid,
        filtertoolong=_configures_filtertoolong(base, matrix.overrides),
    )
    spm = spm_model.as_posix()
    per_run: dict[str, Any] = {
        "seed": variant.seed,
        "src_vocab": paths.vocab.as_posix(),
        "data": data,
        "transforms_configs": {
            "sentencepiece": {"src_subword_model": spm, "tgt_subword_model": spm}
        },
        "training": {"model_path": paths.ckpt_dir.as_posix()},
    }
    layered = _layered(per_run, matrix.overrides)
    layered.update(_vocabulary_size(base, matrix.overrides, spm_model))
    return layered


def variant_config(
    matrix: AblationMatrix,
    variant: Variant,
    *,
    spm_model: Path,
    paths: VariantPaths,
    corpora: Sequence[CorpusEntry],
    weights: Mapping[str, int],
) -> dict[str, Any]:
    """The finished, validated Eole config of `variant`, in memory (nothing is
    written): the base with `variant_overrides` merged over it.

    Raises:
        ConfigError: the finished config is not valid, or what `variant_overrides`
            raises.
        ValueError: as `variant_overrides`.
    """
    overrides = variant_overrides(
        matrix,
        variant,
        spm_model=spm_model,
        paths=paths,
        corpora=corpora,
        weights=weights,
    )
    return merged_config(matrix.base, overrides)


def _vocabulary_size(
    base: object, overrides: Mapping[str, Any], spm_model: Path
) -> dict[str, int]:
    """`src_vocab_size` for the finished config when the base and the overrides leave
    it out (an override of `None` deletes the base's), as an override over the layered
    ones; nothing when a size is given. Either size that is given is checked against the
    model's vocabulary."""
    needed = needed_vocabulary_size(spm_model)
    given = {
        key: overrides[key] if key in overrides else _lookup(base, key)
        for key in VOCABULARY_SIZE_KEYS
    }
    for key, size in given.items():
        if size is not None:
            require_vocabulary_size(key, size, needed, spm_model)
    return {} if given["src_vocab_size"] is not None else {"src_vocab_size": needed}


def _lookup(config: object, key: str) -> Any:
    return config.get(key) if isinstance(config, Mapping) else None


def _layered(
    per_run: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """`per_run` with `overrides` laid over it: mappings join key by key, every other
    value of `overrides` (`None` included) is kept as it is. `merged_config` merges the
    result over the base, so a `None` still deletes the base's key, also inside a
    section the run itself sets."""
    layered = dict(per_run)
    for key, value in overrides.items():
        current = layered.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            layered[key] = _layered(current, value)
        else:
            layered[key] = value
    return layered


def _configures_filtertoolong(base: object, overrides: Mapping[str, Any]) -> bool:
    """Whether the finished config has `transforms_configs.filtertoolong`: the base's
    unless the overrides set it (a `None` deletes it, as in `eole_cfg.merge`). An
    override of `transforms_configs` itself that is no mapping is refused when the
    plan is made."""
    sections = base.get("transforms_configs") if isinstance(base, Mapping) else None
    configured = isinstance(sections, Mapping) and "filtertoolong" in sections
    override = overrides.get("transforms_configs")
    if not isinstance(override, Mapping) or "filtertoolong" not in override:
        return configured
    return override["filtertoolong"] is not None
