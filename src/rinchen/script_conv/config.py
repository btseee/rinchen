"""The Eole training configuration of the converter.

`render_converter_config` builds the config of one profile as a dict, `dump_config`
writes it as YAML and `validate_converter_config` checks the invariants a config
must keep whatever its profile. The two checked-in files `configs/eole/converter.yaml`
and `converter_tiny.yaml` are renderings, never hand-edited.

The tiny profile is the configuration the spike trained on CPU with Eole 0.6.4, with
more steps: every key in it is needed. The full profile is a set of untuned priors: no
real data and no GPU hours exist to tune them on.

This module relies on the corpus file layout of `rinchen.script_conv.data`
(`<direction>.<split>.<side>`) and on the tags of `rinchen.script_conv.char_model`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn

import yaml

from rinchen.errors import ConfigError
from rinchen.script_conv.char_model import CONVERTER_TAGS
from rinchen.utils.part_file import unique_part_path

Profile = Literal["tiny", "full"]

# The tag names the TARGET script: a corpus that converts Cyrillic to the traditional
# script starts every source line with the script tag. A test pins the tuple's order.
_TO_SCRIPT_TAG, _TO_CYRILLIC_TAG = CONVERTER_TAGS

_CYRL2MONG: Final = "cyrl2mong"
_MONG2CYRL: Final = "mong2cyrl"
_DIRECTION_TAGS: Final = {_CYRL2MONG: _TO_SCRIPT_TAG, _MONG2CYRL: _TO_CYRILLIC_TAG}

# Eole 0.6.4 takes ONE validation corpus: it reads `data["valid"]` for validation and
# treats every other name as a training corpus (`eole/inputters/text_corpus.py`,
# `get_corpora`), so a second validation corpus cannot be configured, and one that is
# named differently would train on its own validation pairs. The two validation files
# have different prefixes and so cannot be concatenated. `valid` therefore holds the
# Cyrillic-to-script direction only: the script-to-Cyrillic direction is trained but
# has no validation loss.
_VALID: Final = "valid"
_VALID_DIRECTION: Final = _CYRL2MONG

_POSITION_ENCODING: Final = "SinusoidalInterleaved"
_SEED: Final = 1234

# Eole's `prefix` transform fails with a TypeError when a corpus has no
# `tgt_prefix`; an empty string means "no target prefix".
_NO_TARGET_PREFIX: Final = ""

# Field names of the installed Eole (0.6.4, `eole/config/models.py`) that select or
# tune a position scheme other than sinusoidal, or restrict or group attention.
# Eole has no field named for ALiBi (it is a `position_encoding_type` value, rejected
# below). A key that is present is rejected even with a neutral value, so a config
# never holds a switch that could be flipped.
FORBIDDEN_KEYS: Final = frozenset(
    {
        # Rotary: `rope_config` holds every other rotary setting.
        "rope_config",
        "rotary_interleave",
        "rotary_theta",
        "rotary_theta_local",
        "rotary_dim",
        "rotary_dim_global",
        "partial_rotary_factor",
        "interleave_local",
        "multidimensional_rope",
        "xdrope_section",
        "qk_norm_post_rope",
        # Relative position bias.
        "relative_positions_buckets",
        # Sliding window and per-layer attention types.
        "sliding_window",
        "layer_types",
        # Grouped-query attention.
        "heads_kv",
        "global_heads_kv",
    }
)

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "seed",
        "save_data",
        "src_vocab",
        "tgt_vocab",
        "share_vocab",
        "src_vocab_size",
        "tgt_vocab_size",
        "overwrite",
        "data",
        "transforms_configs",
        "model",
        "training",
    }
)


@dataclass(frozen=True)
class _ProfileSettings:
    vocab_size: int  # a cap: Eole keeps the characters it saw
    max_length: int  # `filtertoolong`, in pieces
    hidden_size: int
    layers: int  # encoder and decoder alike
    heads: int
    transformer_ff: int
    training: Mapping[str, Any]  # the keys that differ per profile


# Everything both profiles train alike.
_BASE_TRAINING: Final[Mapping[str, Any]] = {
    "batch_type": "tokens",
    "accum_count": [1],
    "dropout": [0.1],
    "attention_dropout": [0.1],
    "optim": "adamw",
    "decay_method": "noam",
    "label_smoothing": 0.1,
    "world_size": 1,
}

_PROFILES: Final[Mapping[str, _ProfileSettings]] = {
    "tiny": _ProfileSettings(
        vocab_size=200,
        max_length=64,
        hidden_size=64,
        layers=2,
        heads=4,
        transformer_ff=128,
        training={
            # Measured on the tiny reference list (tests/integration): 800 steps pass
            # the CER gate but keep the NNBSP and selector counts in only 54% of the
            # words that have them; 1400 steps keep 88% and 1500 keep 90%, the gate.
            # Raise the steps, never the gate.
            "train_steps": 1500,
            "valid_steps": 200,
            "save_checkpoint_steps": 1500,
            "keep_checkpoint": 2,
            "batch_size": 1024,
            # The Noam factor of 2.0 peaks near 0.025 at d64 and does not learn.
            "learning_rate": 0.5,
            "warmup_steps": 100,
            "compute_dtype": "fp32",
            "gpu_ranks": [],
            # With the default bucket of 262144 the loader cycles a small corpus to
            # fill it: 300 steps took over 12 minutes, 20 seconds with these.
            "bucket_size": 1024,
            "prefetch_factor": 8,
            "num_workers": 1,
        },
    ),
    # Untuned priors; the bucket and loader settings stay at Eole's defaults.
    "full": _ProfileSettings(
        vocab_size=512,
        max_length=128,
        hidden_size=512,
        layers=6,
        heads=8,
        transformer_ff=2048,
        training={
            "train_steps": 100000,
            "valid_steps": 2000,
            "save_checkpoint_steps": 2000,
            "keep_checkpoint": 5,
            "batch_size": 8192,
            "learning_rate": 2.0,
            "warmup_steps": 4000,
            "compute_dtype": "bf16",
            "gpu_ranks": [0],
        },
    ),
}


def render_converter_config(
    profile: Profile, *, data_dir: Path, spm_model: Path, run_dir: Path
) -> dict[str, Any]:
    """The Eole config of `profile`, with every path as a POSIX string.

    `data_dir` holds the corpus files `rinchen.script_conv.data.build_corpus` wrote,
    `spm_model` is the converter's character model, and `run_dir` takes the
    vocabulary (`vocab.src`, `vocab.tgt`), `data` and the checkpoints (`ckpt`). Each
    call returns a new dict that shares nothing with the next one.

    Raises:
        ConfigError: `profile` is not `tiny` or `full`.
    """
    settings = _profile_settings(profile)
    return {
        "seed": _SEED,
        "save_data": (run_dir / "data").as_posix(),
        "src_vocab": (run_dir / "vocab.src").as_posix(),
        "tgt_vocab": (run_dir / "vocab.tgt").as_posix(),
        "share_vocab": True,
        "src_vocab_size": settings.vocab_size,
        "tgt_vocab_size": settings.vocab_size,
        "overwrite": True,
        "data": _data_section(data_dir),
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": spm_model.as_posix(),
                "tgt_subword_model": spm_model.as_posix(),
            },
            "filtertoolong": {
                "src_seq_length": settings.max_length,
                "tgt_seq_length": settings.max_length,
            },
        },
        "model": _model_section(settings),
        "training": {
            "model_path": (run_dir / "ckpt").as_posix(),
            **copy.deepcopy(_BASE_TRAINING),
            **copy.deepcopy(settings.training),
        },
    }


def _profile_settings(profile: str) -> _ProfileSettings:
    settings = _PROFILES.get(profile)
    if settings is None:
        raise ConfigError(
            f"unknown converter profile {profile!r}: expected one of {sorted(_PROFILES)}"
        )
    return settings


def _corpus_path(data_dir: Path, direction: str, split: str, side: str) -> str:
    return (data_dir / f"{direction}.{split}.{side}").as_posix()


def _data_section(data_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        direction: {
            "path_src": _corpus_path(data_dir, direction, "train", "src"),
            "path_tgt": _corpus_path(data_dir, direction, "train", "tgt"),
            "transforms": ["sentencepiece", "prefix", "filtertoolong"],
            "src_prefix": tag,
            "tgt_prefix": _NO_TARGET_PREFIX,
            "weight": 1,
        }
        for direction, tag in _DIRECTION_TAGS.items()
    }
    data[_VALID] = {
        "path_src": _corpus_path(data_dir, _VALID_DIRECTION, "valid", "src"),
        "path_tgt": _corpus_path(data_dir, _VALID_DIRECTION, "valid", "tgt"),
        "transforms": ["sentencepiece", "prefix"],
        "src_prefix": _DIRECTION_TAGS[_VALID_DIRECTION],
        "tgt_prefix": _NO_TARGET_PREFIX,
    }
    return data


def _model_section(settings: _ProfileSettings) -> dict[str, Any]:
    return {
        "architecture": "transformer",
        "hidden_size": settings.hidden_size,
        "encoder": {"layers": settings.layers},
        "decoder": {"layers": settings.layers},
        "heads": settings.heads,
        "transformer_ff": settings.transformer_ff,
        # Eole's default word vector size is 512; a smaller model fails with a
        # layer-norm shape error unless it equals `hidden_size`.
        "embeddings": {
            "word_vec_size": settings.hidden_size,
            "position_encoding_type": _POSITION_ENCODING,
        },
        "share_embeddings": True,
        "share_decoder_embeddings": True,
    }


def dump_config(cfg: dict[str, Any], path: Path) -> None:
    """Write `cfg` as YAML (UTF-8, LF, keys in their dict order) so the file appears
    whole: a failure leaves the old file, if any, and no temporary file. The
    directory of `path` must exist."""
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    part_path = unique_part_path(path)
    try:
        with part_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def validate_converter_config(
    cfg: Mapping[str, Any], *, profile: Profile | None = None
) -> None:
    """Check `cfg` against the converter's invariants; return None when it holds.

    Always checked: only the known top-level keys, all present; no key of
    `FORBIDDEN_KEYS` at any depth; `position_encoding_type` is
    `SinusoidalInterleaved` wherever it appears and is set on the embeddings;
    `word_vec_size` equals `hidden_size`; the corpora `cyrl2mong`, `mong2cyrl` and
    `valid` exist, each has `tgt_prefix` `""`, and each direction has its target
    script tag as `src_prefix` (`valid` holds the Cyrillic-to-script direction);
    `share_vocab` is true. With `profile="tiny"` the config must also be CPU-only:
    `gpu_ranks` is `[]` and `compute_dtype` is `fp32`.

    Raises:
        ConfigError: any of these fails; the message names the key path.
    """
    if profile is not None:
        _profile_settings(profile)
    if not isinstance(cfg, Mapping):
        _fail("config", f"expected a mapping, got {type(cfg).__name__}")
    _check_top_level_keys(cfg)
    _check_no_forbidden_keys(cfg)
    _check_position_encoding(cfg)
    _check_embedding_size(cfg)
    _check_corpora(cfg)
    if _lookup(cfg, "share_vocab") is not True:
        _fail("share_vocab", "must be true: both directions share one vocabulary")
    if profile == "tiny":
        _check_cpu_only(cfg)


def _fail(path: str, problem: str) -> NoReturn:
    raise ConfigError(f"invalid converter config: {path}: {problem}")


def _lookup(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    walked: list[str] = []
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            _fail(".".join(walked), f"expected a mapping, got {type(node).__name__}")
        if part not in node:
            _fail(".".join([*walked, part]), "is missing")
        node = node[part]
        walked.append(part)
    return node


def _entries(node: object, path: str) -> Iterator[tuple[str, object, object]]:
    """Every mapping entry below `node` as (key path, key, value), through nested
    mappings and lists; a list item's path carries its index."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, key, value
            yield from _entries(value, child)
    elif isinstance(node, list | tuple):
        for index, item in enumerate(node):
            yield from _entries(item, f"{path}[{index}]")


def _check_top_level_keys(cfg: Mapping[str, Any]) -> None:
    for key in cfg:
        if key not in _TOP_LEVEL_KEYS:
            _fail(str(key), "unknown top-level key")
    for key in sorted(_TOP_LEVEL_KEYS):
        if key not in cfg:
            _fail(key, "is missing")


def _check_no_forbidden_keys(cfg: Mapping[str, Any]) -> None:
    for path, key, _ in _entries(cfg, ""):
        if key in FORBIDDEN_KEYS:
            _fail(path, "is not allowed: the converter uses full multi-head attention")


def _check_position_encoding(cfg: Mapping[str, Any]) -> None:
    embeddings_path = "model.embeddings.position_encoding_type"
    if _lookup(cfg, embeddings_path) != _POSITION_ENCODING:
        _fail(embeddings_path, f"must be {_POSITION_ENCODING}")
    for path, key, value in _entries(cfg, ""):
        if key == "position_encoding_type" and value != _POSITION_ENCODING:
            _fail(path, f"must be {_POSITION_ENCODING}")


def _check_embedding_size(cfg: Mapping[str, Any]) -> None:
    hidden_size = _lookup(cfg, "model.hidden_size")
    word_vec_size = _lookup(cfg, "model.embeddings.word_vec_size")
    if word_vec_size != hidden_size:
        _fail(
            "model.embeddings.word_vec_size",
            f"must equal model.hidden_size ({hidden_size!r}), got {word_vec_size!r}",
        )


def _check_corpora(cfg: Mapping[str, Any]) -> None:
    data = _lookup(cfg, "data")
    if not isinstance(data, Mapping):
        _fail("data", f"expected a mapping, got {type(data).__name__}")
    for name in (*_DIRECTION_TAGS, _VALID):
        if name not in data:
            _fail(f"data.{name}", "is missing")
    for name, corpus in data.items():
        if not isinstance(corpus, Mapping):
            _fail(f"data.{name}", f"expected a mapping, got {type(corpus).__name__}")
        if corpus.get("tgt_prefix") != _NO_TARGET_PREFIX:
            _fail(
                f"data.{name}.tgt_prefix",
                'must be "": the prefix transform fails without it',
            )
    expected_tags = {**_DIRECTION_TAGS, _VALID: _DIRECTION_TAGS[_VALID_DIRECTION]}
    for name, tag in expected_tags.items():
        if data[name].get("src_prefix") != tag:
            _fail(f"data.{name}.src_prefix", f"must be {tag!r}")


def _check_cpu_only(cfg: Mapping[str, Any]) -> None:
    if _lookup(cfg, "training.gpu_ranks") != []:
        _fail("training.gpu_ranks", "must be [] for the tiny profile")
    if _lookup(cfg, "training.compute_dtype") != "fp32":
        _fail("training.compute_dtype", "must be fp32 for the tiny profile")
