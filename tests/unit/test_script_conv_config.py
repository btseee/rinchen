"""Tests for `rinchen.script_conv.config`: the Eole training configuration of the
converter, rendered per profile and validated.

Most tests build the config as a plain dict and mutate one key. The Eole tests share
one module-scoped fixture: a real tiny corpus and character model in a temporary
directory, a real `eole build_vocab` run, and Eole's own loader. The checked-in files
under `configs/eole/` must equal what `render_converter_config` renders for the
placeholder paths below.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pydantic
import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.script_conv.char_model import CONVERTER_TAGS, train_char_model
from rinchen.script_conv.config import (
    FORBIDDEN_KEYS,
    ConverterPaths,
    dump_config,
    read_converter_paths,
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import CORPUS_FILE_NAMES, build_corpus, read_pairs_tsv
from tests.support.script_reference import FIXTURE_DIR

# Importing Eole warns about its missing C++ extension and flash_attn and about
# deprecations in Eole, pydantic and torch; none of it is ours to fix.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:Could not import:RuntimeWarning"),
    pytest.mark.filterwarnings(
        "ignore:.torch.jit.script. is not supported:FutureWarning"
    ),
]

Profile = Literal["tiny", "full"]
PROFILES: tuple[Profile, ...] = ("tiny", "full")

CONFIG_DIR = REPO_ROOT / "configs" / "eole"
CHECKED_IN = {
    "tiny": CONFIG_DIR / "converter_tiny.yaml",
    "full": CONFIG_DIR / "converter.yaml",
}
PLACEHOLDER_DATA_DIR = Path("data/script_conv/corpus")
PLACEHOLDER_SPM_MODEL = Path("data/script_conv/converter_char.model")
PLACEHOLDER_RUN_DIR = Path("data/script_conv/run")

DATA_DIR = Path("work/corpus")
SPM_MODEL = Path("work/converter_char.model")
RUN_DIR = Path("work/run")

TO_SCRIPT_TAG = "<2mn_Mong>"
TO_CYRILLIC_TAG = "<2mn_Cyrl>"
SINUSOIDAL = "SinusoidalInterleaved"

TOP_LEVEL_KEYS = (
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
)
CORPORA = ("cyrl2mong", "mong2cyrl", "valid")


def _render(profile: Profile) -> dict[str, Any]:
    return render_converter_config(
        profile, data_dir=DATA_DIR, spm_model=SPM_MODEL, run_dir=RUN_DIR
    )


def _render_placeholders(profile: Profile) -> dict[str, Any]:
    return render_converter_config(
        profile,
        data_dir=PLACEHOLDER_DATA_DIR,
        spm_model=PLACEHOLDER_SPM_MODEL,
        run_dir=PLACEHOLDER_RUN_DIR,
    )


def _node(cfg: dict[str, Any], dotted: str) -> dict[str, Any]:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _set(cfg: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    parent, _, leaf = dotted.rpartition(".")
    _node(cfg, parent)[leaf] = value
    return cfg


def _set_top(cfg: dict[str, Any], key: str, value: object) -> dict[str, Any]:
    cfg[key] = value
    return cfg


def _delete(cfg: dict[str, Any], dotted: str) -> dict[str, Any]:
    parent, _, leaf = dotted.rpartition(".")
    del _node(cfg, parent)[leaf]
    return cfg


# --- rendering: values per profile -----------------------------------------------------

TINY_MODEL = {
    "architecture": "transformer",
    "hidden_size": 64,
    "encoder": {"layers": 2},
    "decoder": {"layers": 2},
    "heads": 4,
    "transformer_ff": 128,
    "embeddings": {"word_vec_size": 64, "position_encoding_type": SINUSOIDAL},
    "share_embeddings": True,
    "share_decoder_embeddings": True,
}
FULL_MODEL = {
    "architecture": "transformer",
    "hidden_size": 512,
    "encoder": {"layers": 6},
    "decoder": {"layers": 6},
    "heads": 8,
    "transformer_ff": 2048,
    "embeddings": {"word_vec_size": 512, "position_encoding_type": SINUSOIDAL},
    "share_embeddings": True,
    "share_decoder_embeddings": True,
}
# The tiny values are the spike's (Eole 0.6.4, CPU), except the step counts and the
# dropout: the tiny model memorises a fixture, so it trains without dropout.
TINY_TRAINING = {
    "model_path": "work/run/ckpt",
    "train_steps": 1000,
    "valid_steps": 200,
    "save_checkpoint_steps": 1000,
    "keep_checkpoint": 2,
    "batch_type": "tokens",
    "batch_size": 1024,
    "accum_count": [1],
    "dropout": [0.0],
    "attention_dropout": [0.0],
    "optim": "adamw",
    "learning_rate": 0.5,
    "decay_method": "noam",
    "warmup_steps": 100,
    "label_smoothing": 0.1,
    "compute_dtype": "fp32",
    "bucket_size": 1024,
    "prefetch_factor": 8,
    "num_workers": 1,
    "world_size": 1,
    "gpu_ranks": [],
}
# Untuned priors: no real data and no GPU hours exist yet.
FULL_TRAINING = {
    "model_path": "work/run/ckpt",
    "train_steps": 100000,
    "valid_steps": 2000,
    "save_checkpoint_steps": 2000,
    "keep_checkpoint": 5,
    "batch_type": "tokens",
    "batch_size": 8192,
    "accum_count": [1],
    "dropout": [0.1],
    "attention_dropout": [0.1],
    "optim": "adamw",
    "learning_rate": 2.0,
    "decay_method": "noam",
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "compute_dtype": "bf16",
    "world_size": 1,
    "gpu_ranks": [0],
}
EXPECTED = {
    "tiny": {
        "model": TINY_MODEL,
        "training": TINY_TRAINING,
        "vocab_size": 200,
        "sequence_length": 64,
    },
    "full": {
        "model": FULL_MODEL,
        "training": FULL_TRAINING,
        "vocab_size": 512,
        "sequence_length": 128,
    },
}


@pytest.mark.parametrize("profile", PROFILES)
def test_the_model_section_has_the_profile_values(profile: Profile) -> None:
    assert _render(profile)["model"] == EXPECTED[profile]["model"]


@pytest.mark.parametrize("profile", PROFILES)
def test_the_training_section_has_the_profile_values(profile: Profile) -> None:
    assert _render(profile)["training"] == EXPECTED[profile]["training"]


@pytest.mark.parametrize("profile", PROFILES)
def test_the_vocabulary_caps_and_length_filter_have_the_profile_values(
    profile: Profile,
) -> None:
    cfg = _render(profile)
    expected = EXPECTED[profile]
    assert cfg["src_vocab_size"] == expected["vocab_size"]
    assert cfg["tgt_vocab_size"] == expected["vocab_size"]
    assert cfg["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": expected["sequence_length"],
        "tgt_seq_length": expected["sequence_length"],
    }


def test_the_full_profile_leaves_the_bucket_settings_at_eole_defaults() -> None:
    training = _render("full")["training"]
    assert not {"bucket_size", "prefetch_factor", "num_workers"} & set(training)


@pytest.mark.parametrize("profile", PROFILES)
def test_the_word_vector_size_equals_the_hidden_size(profile: Profile) -> None:
    model = _render(profile)["model"]
    assert model["embeddings"]["word_vec_size"] == model["hidden_size"]


# --- rendering: the parts both profiles share ------------------------------------------


@pytest.mark.parametrize("profile", PROFILES)
def test_the_top_level_keys_are_the_known_ones(profile: Profile) -> None:
    assert set(_render(profile)) == set(TOP_LEVEL_KEYS)


@pytest.mark.parametrize("profile", PROFILES)
def test_the_vocabulary_is_shared_and_lives_in_the_run_directory(
    profile: Profile,
) -> None:
    cfg = _render(profile)
    assert cfg["share_vocab"] is True
    assert cfg["overwrite"] is True
    assert cfg["src_vocab"] == "work/run/vocab.src"
    assert cfg["tgt_vocab"] == "work/run/vocab.tgt"
    assert cfg["save_data"] == "work/run/data"
    assert cfg["training"]["model_path"] == "work/run/ckpt"


@pytest.mark.parametrize("profile", PROFILES)
def test_the_character_model_serves_both_sides(profile: Profile) -> None:
    sentencepiece = _render(profile)["transforms_configs"]["sentencepiece"]
    assert sentencepiece == {
        "src_subword_model": "work/converter_char.model",
        "tgt_subword_model": "work/converter_char.model",
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_each_direction_trains_from_its_own_corpus_with_the_target_script_tag(
    profile: Profile,
) -> None:
    data = _render(profile)["data"]
    for corpus, tag in (("cyrl2mong", TO_SCRIPT_TAG), ("mong2cyrl", TO_CYRILLIC_TAG)):
        assert data[corpus] == {
            "path_src": f"work/corpus/{corpus}.train.src",
            "path_tgt": f"work/corpus/{corpus}.train.tgt",
            "transforms": ["sentencepiece", "prefix", "filtertoolong"],
            "src_prefix": tag,
            "tgt_prefix": "",
            "weight": 1,
        }


@pytest.mark.parametrize("profile", PROFILES)
def test_validation_uses_one_corpus_named_valid_for_the_cyrillic_to_script_direction(
    profile: Profile,
) -> None:
    data = _render(profile)["data"]
    assert set(data) == set(CORPORA)
    assert data["valid"] == {
        "path_src": "work/corpus/cyrl2mong.valid.src",
        "path_tgt": "work/corpus/cyrl2mong.valid.tgt",
        "transforms": ["sentencepiece", "prefix"],
        "src_prefix": TO_SCRIPT_TAG,
        "tgt_prefix": "",
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_every_rendered_corpus_path_names_a_file_build_corpus_writes(
    profile: Profile,
) -> None:
    data = _render(profile)["data"]
    paths = [
        Path(corpus[side])
        for corpus in data.values()
        for side in ("path_src", "path_tgt")
    ]

    assert len(paths) == 2 * len(CORPORA)
    assert {path.name for path in paths} <= set(CORPUS_FILE_NAMES)
    assert {path.parent for path in paths} == {DATA_DIR}


def test_the_prefixes_are_the_character_model_tags() -> None:
    assert CONVERTER_TAGS == (TO_SCRIPT_TAG, TO_CYRILLIC_TAG)


@pytest.mark.parametrize("profile", PROFILES)
def test_rendering_is_deterministic(profile: Profile) -> None:
    assert _render(profile) == _render(profile)
    assert list(_render(profile)) == list(_render(profile))


@pytest.mark.parametrize("profile", PROFILES)
def test_a_rendering_does_not_share_state_with_the_next_one(profile: Profile) -> None:
    first = _render(profile)
    first["model"]["encoder"]["layers"] = 99
    first["training"]["dropout"].append(0.5)
    first["data"]["valid"]["tgt_prefix"] = "x"

    assert _render(profile)["model"] == EXPECTED[profile]["model"]
    assert _render(profile)["training"] == EXPECTED[profile]["training"]
    assert _render(profile)["data"]["valid"]["tgt_prefix"] == ""


def test_every_path_is_a_posix_string() -> None:
    cfg = render_converter_config(
        "tiny",
        data_dir=Path("a") / "b",
        spm_model=Path("c") / "d.model",
        run_dir=Path("e"),
    )
    assert cfg["data"]["cyrl2mong"]["path_src"] == "a/b/cyrl2mong.train.src"
    for value in (
        cfg["save_data"],
        cfg["src_vocab"],
        cfg["tgt_vocab"],
        cfg["training"]["model_path"],
        cfg["transforms_configs"]["sentencepiece"]["src_subword_model"],
    ):
        assert type(value) is str


def test_an_unknown_profile_is_rejected() -> None:
    with pytest.raises(ConfigError, match="profile"):
        render_converter_config(
            "huge",  # type: ignore[arg-type]
            data_dir=DATA_DIR,
            spm_model=SPM_MODEL,
            run_dir=RUN_DIR,
        )


@pytest.mark.parametrize("profile", PROFILES)
def test_a_rendering_passes_validation(profile: Profile) -> None:
    validate_converter_config(_render(profile))
    validate_converter_config(_render(profile), profile=profile)


# --- validation: the top level ---------------------------------------------------------


def test_an_unknown_top_level_key_is_rejected() -> None:
    cfg = _set_top(_render("tiny"), "tensorboard", True)
    with pytest.raises(ConfigError, match="tensorboard"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_a_missing_top_level_key_is_rejected(key: str) -> None:
    cfg = _render("tiny")
    del cfg[key]
    with pytest.raises(ConfigError, match=re.escape(key)):
        validate_converter_config(cfg)


def test_a_non_mapping_config_is_rejected() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        validate_converter_config(["model"])  # type: ignore[arg-type]


# --- validation: position encoding and attention variants ------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "Rotary",
        "Relative",
        "Alibi",
        "Learned",
        "SinusoidalConcat",
        "sinusoidalinterleaved",
        None,
    ],
)
def test_a_position_encoding_other_than_sinusoidal_interleaved_is_rejected(
    value: object,
) -> None:
    cfg = _set(_render("tiny"), "model.embeddings.position_encoding_type", value)
    with pytest.raises(ConfigError, match=r"model\.embeddings\.position_encoding_type"):
        validate_converter_config(cfg)


def test_a_missing_position_encoding_type_is_rejected() -> None:
    cfg = _delete(_render("tiny"), "model.embeddings.position_encoding_type")
    with pytest.raises(ConfigError, match=r"model\.embeddings\.position_encoding_type"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("where", ["model.encoder", "model.decoder"])
def test_a_position_encoding_override_below_the_model_is_checked_too(
    where: str,
) -> None:
    cfg = _set(_render("tiny"), f"{where}.position_encoding_type", "Rotary")
    with pytest.raises(ConfigError, match=re.escape(f"{where}.position_encoding_type")):
        validate_converter_config(cfg)

    ok = _set(_render("tiny"), f"{where}.position_encoding_type", SINUSOIDAL)
    validate_converter_config(ok)


@pytest.mark.parametrize("key", sorted(FORBIDDEN_KEYS))
def test_every_forbidden_key_is_rejected_and_named(key: str) -> None:
    cfg = _set(_render("tiny"), f"model.encoder.{key}", 0)
    with pytest.raises(ConfigError, match=re.escape(f"model.encoder.{key}")):
        validate_converter_config(cfg)


@pytest.mark.parametrize(
    "where",
    [
        "model",
        "model.decoder",
        "model.embeddings",
        "training",
        "transforms_configs.sentencepiece",
        "data.valid",
    ],
)
def test_a_forbidden_key_is_rejected_wherever_it_nests(where: str) -> None:
    cfg = _set(_render("tiny"), f"{where}.sliding_window", 128)
    with pytest.raises(ConfigError, match=re.escape(f"{where}.sliding_window")):
        validate_converter_config(cfg)


def test_a_forbidden_key_inside_a_list_is_rejected_with_its_index() -> None:
    cfg = _render("tiny")
    cfg["model"]["layer_overrides"] = [{"heads": 4}, {"heads_kv": 2}]
    with pytest.raises(
        ConfigError, match=re.escape("model.layer_overrides[1].heads_kv")
    ):
        validate_converter_config(cfg)


def test_a_forbidden_key_is_rejected_even_with_a_neutral_value() -> None:
    cfg = _set(_render("tiny"), "model.sliding_window", 0)
    with pytest.raises(ConfigError, match=r"model\.sliding_window"):
        validate_converter_config(cfg)


def test_the_forbidden_keys_cover_rotary_relative_sliding_window_and_grouped_heads() -> (
    None
):
    assert {
        "rope_config",
        "rotary_interleave",
        "rotary_theta",
        "rotary_dim",
        "relative_positions_buckets",
        "sliding_window",
        "heads_kv",
    } <= FORBIDDEN_KEYS
    assert isinstance(FORBIDDEN_KEYS, frozenset)


# --- validation: sizes, prefixes, vocabulary, profile ----------------------------------


@pytest.mark.parametrize("word_vec_size", [63, 65, 512])
def test_a_word_vector_size_that_differs_from_the_hidden_size_is_rejected(
    word_vec_size: int,
) -> None:
    cfg = _set(_render("tiny"), "model.embeddings.word_vec_size", word_vec_size)
    with pytest.raises(ConfigError, match=r"model\.embeddings\.word_vec_size"):
        validate_converter_config(cfg)


@pytest.mark.parametrize(
    "dotted", ["model.embeddings.word_vec_size", "model.hidden_size"]
)
def test_a_missing_size_is_rejected(dotted: str) -> None:
    cfg = _delete(_render("tiny"), dotted)
    with pytest.raises(ConfigError, match=re.escape(dotted)):
        validate_converter_config(cfg)


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("value", ["missing", None, "x", " "])
def test_a_corpus_without_an_empty_target_prefix_is_rejected(
    corpus: str, value: str | None
) -> None:
    cfg = _render("tiny")
    if value == "missing":
        del cfg["data"][corpus]["tgt_prefix"]
    else:
        cfg["data"][corpus]["tgt_prefix"] = value
    with pytest.raises(ConfigError, match=re.escape(f"data.{corpus}.tgt_prefix")):
        validate_converter_config(cfg)


def test_an_extra_training_corpus_needs_an_empty_target_prefix_too() -> None:
    cfg = _render("tiny")
    cfg["data"]["more"] = copy.deepcopy(cfg["data"]["cyrl2mong"])
    validate_converter_config(cfg)

    del cfg["data"]["more"]["tgt_prefix"]
    with pytest.raises(ConfigError, match=r"data\.more\.tgt_prefix"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("corpus", CORPORA)
def test_a_missing_corpus_is_rejected(corpus: str) -> None:
    cfg = _render("tiny")
    del cfg["data"][corpus]
    with pytest.raises(ConfigError, match=re.escape(f"data.{corpus}")):
        validate_converter_config(cfg)


@pytest.mark.parametrize(
    ("corpus", "tag"),
    [
        ("cyrl2mong", TO_CYRILLIC_TAG),
        ("mong2cyrl", TO_SCRIPT_TAG),
        ("valid", TO_CYRILLIC_TAG),
        ("cyrl2mong", ""),
        ("mong2cyrl", None),
    ],
)
def test_a_direction_with_the_wrong_tag_is_rejected(
    corpus: str, tag: str | None
) -> None:
    cfg = _set(_render("tiny"), f"data.{corpus}.src_prefix", tag)
    with pytest.raises(ConfigError, match=re.escape(f"data.{corpus}.src_prefix")):
        validate_converter_config(cfg)


def test_a_missing_source_prefix_is_rejected() -> None:
    cfg = _delete(_render("tiny"), "data.mong2cyrl.src_prefix")
    with pytest.raises(ConfigError, match=r"data\.mong2cyrl\.src_prefix"):
        validate_converter_config(cfg)


# --- validation: transforms, layer sizes, position switch, string sections -------------

_NO_TRANSFORMS = "missing"


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize(
    "transforms",
    [
        _NO_TRANSFORMS,
        None,
        [],
        ["filtertoolong"],
        ["sentencepiece"],
        ["prefix"],
        ["prefix", "sentencepiece"],
        ["filtertoolong", "prefix", "sentencepiece"],
        "sentencepiece prefix",
    ],
    ids=str,
)
def test_a_corpus_that_does_not_split_into_pieces_before_it_prefixes_is_rejected(
    corpus: str, transforms: object
) -> None:
    cfg = _render("tiny")
    if transforms == _NO_TRANSFORMS:
        del cfg["data"][corpus]["transforms"]
    else:
        cfg["data"][corpus]["transforms"] = transforms
    with pytest.raises(ConfigError, match=re.escape(f"data.{corpus}.transforms")):
        validate_converter_config(cfg)


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize(
    "transforms",
    [
        ["sentencepiece", "prefix"],
        ["sentencepiece", "prefix", "filtertoolong"],
        ["filtertoolong", "sentencepiece", "prefix"],
        ["sentencepiece", "filtertoolong", "prefix"],
    ],
    ids=str,
)
def test_a_corpus_may_add_other_transforms_around_sentencepiece_and_prefix(
    corpus: str, transforms: list[str]
) -> None:
    cfg = _set(_render("tiny"), f"data.{corpus}.transforms", transforms)
    validate_converter_config(cfg)


def test_an_extra_training_corpus_needs_the_two_transforms_too() -> None:
    cfg = _render("tiny")
    cfg["data"]["more"] = copy.deepcopy(cfg["data"]["cyrl2mong"])
    cfg["data"]["more"]["transforms"] = ["prefix"]
    with pytest.raises(ConfigError, match=r"data\.more\.transforms"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("where", ["model.encoder", "model.decoder"])
@pytest.mark.parametrize("hidden_size", [63, 128, "64", None])
def test_a_layer_level_hidden_size_that_differs_from_the_vector_size_is_rejected(
    where: str, hidden_size: object
) -> None:
    cfg = _set(_render("tiny"), f"{where}.hidden_size", hidden_size)
    with pytest.raises(ConfigError, match=re.escape(f"{where}.hidden_size")):
        validate_converter_config(cfg)


@pytest.mark.parametrize("where", ["model.encoder", "model.decoder"])
def test_a_layer_level_hidden_size_equal_to_the_vector_size_is_accepted(
    where: str,
) -> None:
    validate_converter_config(_set(_render("tiny"), f"{where}.hidden_size", 64))


@pytest.mark.parametrize("value", [False, None, 0, "false"])
def test_switching_the_absolute_position_encoding_off_is_rejected(
    value: object,
) -> None:
    cfg = _set(_render("tiny"), "model.embeddings.position_encoding", value)
    with pytest.raises(ConfigError, match=r"model\.embeddings\.position_encoding:"):
        validate_converter_config(cfg)


def test_the_absolute_position_encoding_switch_may_be_on_or_left_out() -> None:
    assert "position_encoding" not in _render("tiny")["model"]["embeddings"]
    validate_converter_config(_render("tiny"))
    validate_converter_config(
        _set(_render("tiny"), "model.embeddings.position_encoding", True)
    )


@pytest.mark.parametrize("where", ["model.encoder", "model.decoder"])
@pytest.mark.parametrize("text", ["{'layers': 2, 'sliding_window': 4}", "None", ""])
def test_a_string_valued_encoder_or_decoder_is_rejected(where: str, text: str) -> None:
    # Eole evaluates such a string as Python code, and the string would hide any
    # forbidden key from the checks above.
    cfg = _set(_render("tiny"), where, text)
    with pytest.raises(ConfigError, match=rf"{re.escape(where)}:.*string"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("value", [False, None, "true"])
def test_a_vocabulary_that_is_not_shared_is_rejected(value: object) -> None:
    cfg = _set_top(_render("tiny"), "share_vocab", value)
    with pytest.raises(ConfigError, match="share_vocab"):
        validate_converter_config(cfg)


@pytest.mark.parametrize("gpu_ranks", [[0], [0, 1]])
def test_a_tiny_profile_on_a_gpu_is_rejected(gpu_ranks: list[int]) -> None:
    cfg = _set(_render("tiny"), "training.gpu_ranks", gpu_ranks)
    with pytest.raises(ConfigError, match=r"training\.gpu_ranks"):
        validate_converter_config(cfg, profile="tiny")


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32 ", None])
def test_a_tiny_profile_with_a_dtype_other_than_fp32_is_rejected(
    dtype: str | None,
) -> None:
    cfg = _set(_render("tiny"), "training.compute_dtype", dtype)
    with pytest.raises(ConfigError, match=r"training\.compute_dtype"):
        validate_converter_config(cfg, profile="tiny")


def test_the_cpu_only_rules_apply_to_the_tiny_profile_only() -> None:
    gpu_tiny = _render("tiny")
    gpu_tiny["training"].update({"gpu_ranks": [0], "compute_dtype": "bf16"})
    validate_converter_config(gpu_tiny)
    validate_converter_config(gpu_tiny, profile="full")

    validate_converter_config(_render("full"), profile="full")


def test_an_unknown_profile_argument_is_rejected() -> None:
    with pytest.raises(ConfigError, match="profile"):
        validate_converter_config(_render("tiny"), profile="huge")  # type: ignore[arg-type]


def test_validation_does_not_change_the_config() -> None:
    cfg = _render("tiny")
    before = copy.deepcopy(cfg)
    validate_converter_config(cfg, profile="tiny")
    assert cfg == before


def test_validation_reports_every_failure_as_a_config_error() -> None:
    mutations: list[Callable[[dict[str, Any]], object]] = [
        lambda cfg: cfg["data"].pop("cyrl2mong"),
        lambda cfg: cfg.pop("model"),
        lambda cfg: cfg.update(model="transformer"),
        lambda cfg: cfg["model"].update(embeddings="Sinusoidal"),
        lambda cfg: cfg.update(data=[]),
        lambda cfg: cfg["data"].update(cyrl2mong="path"),
        lambda cfg: cfg["training"].pop("gpu_ranks"),
    ]
    for mutate in mutations:
        cfg = _render("tiny")
        mutate(cfg)
        with pytest.raises(ConfigError):
            validate_converter_config(cfg, profile="tiny")


# --- dump_config -----------------------------------------------------------------------


@pytest.mark.parametrize("profile", PROFILES)
def test_a_dumped_config_loads_back_equal_with_its_key_order(
    tmp_path: Path, profile: Profile
) -> None:
    cfg = _render(profile)
    target = tmp_path / "config.yaml"

    dump_config(cfg, target)

    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == cfg
    assert list(loaded) == list(cfg)
    assert list(loaded["model"]) == list(cfg["model"])
    assert list(loaded["training"]) == list(cfg["training"])


def test_a_dumped_config_is_utf_8_with_lf_endings_and_keeps_non_ascii(
    tmp_path: Path,
) -> None:
    cfg = render_converter_config(
        "tiny",
        data_dir=Path("корпус"),
        spm_model=SPM_MODEL,
        run_dir=RUN_DIR,
    )
    target = tmp_path / "config.yaml"

    dump_config(cfg, target)

    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert "корпус/cyrl2mong.train.src".encode() in raw
    assert b"\\u" not in raw and b"\\x" not in raw


def test_dumping_again_replaces_the_file_and_leaves_no_part_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config.yaml"
    dump_config(_render("full"), target)

    dump_config(_render("tiny"), target)

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == _render("tiny")
    assert [path.name for path in tmp_path.iterdir()] == ["config.yaml"]


def test_a_failed_dump_keeps_the_old_file_and_leaves_no_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.yaml"
    dump_config(_render("full"), target)
    before = target.read_bytes()

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        dump_config(_render("tiny"), target)

    assert target.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == ["config.yaml"]


def test_dumping_into_a_missing_directory_fails_without_creating_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError):
        dump_config(_render("tiny"), tmp_path / "absent" / "config.yaml")
    assert list(tmp_path.iterdir()) == []


# --- the checked-in files --------------------------------------------------------------


@pytest.mark.parametrize("profile", PROFILES)
def test_the_checked_in_file_is_exactly_the_rendering(
    tmp_path: Path, profile: Profile
) -> None:
    rendered = tmp_path / "rendered.yaml"
    dump_config(_render_placeholders(profile), rendered)

    assert CHECKED_IN[profile].read_bytes() == rendered.read_bytes()


@pytest.mark.parametrize("profile", PROFILES)
def test_the_checked_in_file_passes_validation(profile: Profile) -> None:
    loaded = yaml.safe_load(CHECKED_IN[profile].read_text(encoding="utf-8"))
    validate_converter_config(loaded, profile=profile)


def test_the_checked_in_tiny_file_names_the_placeholder_paths() -> None:
    loaded = yaml.safe_load(CHECKED_IN["tiny"].read_text(encoding="utf-8"))
    assert loaded["data"]["cyrl2mong"]["path_src"] == (
        "data/script_conv/corpus/cyrl2mong.train.src"
    )
    assert loaded["src_vocab"] == "data/script_conv/run/vocab.src"
    assert loaded["transforms_configs"]["sentencepiece"]["src_subword_model"] == (
        "data/script_conv/converter_char.model"
    )


# --- Eole's own view -------------------------------------------------------------------


def test_the_forbidden_keys_are_real_eole_fields_and_cover_every_position_variant() -> (
    None
):
    from eole.config import models

    fields = {
        field
        for _, cls in inspect.getmembers(models, inspect.isclass)
        if issubclass(cls, pydantic.BaseModel)
        for field in cls.model_fields
    }
    variants = {
        field
        for field in fields
        if re.search(r"rotary|rope|alibi|relative|sliding|heads_kv|layer_types", field)
    }

    assert FORBIDDEN_KEYS <= fields, sorted(FORBIDDEN_KEYS - fields)
    assert variants <= FORBIDDEN_KEYS, sorted(variants - FORBIDDEN_KEYS)
    assert set(models.RotaryPositionConfig.model_fields) - FORBIDDEN_KEYS <= {
        # generic RoPE scaling knobs, reachable only through `rope_config`, which is
        # forbidden
        "scaling_type",
        "alpha",
        "scaling_factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
        "tmax_index",
    }


@dataclasses.dataclass(frozen=True)
class EoleRun:
    """Everything the Eole tests need from the one real run."""

    config: Any  # eole.config.run.TrainConfig
    config_path: Path
    config_dict: dict[str, Any]
    run_dir: Path
    corpus_dir: Path


@pytest.fixture(scope="module")
def eole_run(tmp_path_factory: pytest.TempPathFactory) -> EoleRun:
    """The rendered tiny config on a real corpus and character model, through
    `eole build_vocab` and Eole's own `eole train` config loader."""
    from eole.bin.run.train import Train

    root = tmp_path_factory.mktemp("converter_config")
    corpus_dir = root / "corpus"
    run_dir = root / "run"
    run_dir.mkdir()

    pairs = read_pairs_tsv(FIXTURE_DIR / "tiny_words.tsv")
    build_corpus(pairs, corpus_dir, valid_fraction=0.1, seed=1234)
    spm_model = train_char_model(
        [pair.cyrillic for pair in pairs] + [pair.script for pair in pairs],
        root / "converter_char",
        vocab_size=200,
    )
    config_dict = render_converter_config(
        "tiny", data_dir=corpus_dir, spm_model=spm_model, run_dir=run_dir
    )
    config_path = run_dir / "converter_tiny.yaml"
    dump_config(config_dict, config_path)

    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "eole.bin.main",
            "build_vocab",
            "-c",
            str(config_path),
            "-n_sample",
            "-1",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    parser = ArgumentParser()
    Train.add_args(parser)
    config = Train.build_config(parser.parse_args(["-c", str(config_path)]))
    return EoleRun(config, config_path, config_dict, run_dir, corpus_dir)


def _split_count(line: str) -> tuple[str, str]:
    piece, _, count = line.partition("\t")
    return piece, count


@pytest.mark.subprocess
def test_eole_loads_the_rendered_tiny_config_and_keeps_every_value(
    eole_run: EoleRun,
) -> None:
    config = eole_run.config

    assert config.model.architecture == "transformer"
    assert config.model.hidden_size == 64
    assert config.model.embeddings.src_word_vec_size == 64
    assert config.model.embeddings.tgt_word_vec_size == 64
    assert config.model.embeddings.position_encoding_type == SINUSOIDAL
    assert config.model.encoder.layers == 2
    assert config.model.decoder.layers == 2
    assert config.training.learning_rate == 0.5
    assert config.training.train_steps == 1000
    assert config.training.dropout == [0.0]
    assert config.training.attention_dropout == [0.0]
    assert config.training.bucket_size == 1024
    assert config.training.gpu_ranks == []
    assert config.share_vocab is True
    assert set(config.data) == set(CORPORA)


@pytest.mark.subprocess
def test_eole_accepts_the_data_and_vocabulary_checks_that_eole_train_runs(
    eole_run: EoleRun,
) -> None:
    # `eole train` runs the vocabulary check (eole/bin/run/train.py) and, in
    # `TrainingInitializer.initialize`, the data check (eole/train_single.py): each
    # raises IOError for a corpus or vocabulary path that is not a file.
    eole_run.config._validate_vocab_config()
    eole_run.config._validate_data_config()


def test_eole_accepts_the_full_profile_rendering_without_any_file() -> None:
    # Constructing the config checks every key and value against Eole's own schema;
    # it reads no file, so the placeholder paths do not have to exist.
    from eole.config.run import TrainConfig

    config = TrainConfig(**_render("full"))

    assert config.training.gpu_ranks == [0]
    assert str(config.training.compute_dtype) == "torch.bfloat16"
    assert config.training.batch_size == 8192
    assert config.model.hidden_size == 512
    assert config.model.embeddings.position_encoding_type == SINUSOIDAL


@pytest.mark.subprocess
def test_every_training_line_carries_the_tag_of_its_direction_in_the_vocabulary_eole_builds(
    eole_run: EoleRun,
) -> None:
    lines = (eole_run.run_dir / "vocab.src").read_text(encoding="utf-8").splitlines()
    counts = {piece: int(count) for piece, count in map(_split_count, lines)}

    for direction, tag in (
        ("cyrl2mong", TO_SCRIPT_TAG),
        ("mong2cyrl", TO_CYRILLIC_TAG),
    ):
        rows = (eole_run.corpus_dir / f"{direction}.train.src").read_text(
            encoding="utf-8"
        )
        assert counts[tag] == len(rows.splitlines())


@pytest.mark.subprocess
def test_eole_reads_only_the_corpus_named_valid_for_validation(
    eole_run: EoleRun,
) -> None:
    """Eole 0.6.4 takes exactly one validation corpus: `get_corpora` for the
    validation task looks up `data["valid"]` and nothing else, and every other name
    is a training corpus. A second validation corpus is therefore not possible, and
    a corpus named like one (`valid_mong2cyrl`) would train on its own validation
    pairs."""
    from eole.constants import CorpusTask
    from eole.inputters.text_corpus import get_corpora

    cfg = copy.deepcopy(eole_run.config_dict)
    cfg["data"]["valid_mong2cyrl"] = {
        "path_src": str(eole_run.corpus_dir / "mong2cyrl.valid.src"),
        "path_tgt": str(eole_run.corpus_dir / "mong2cyrl.valid.tgt"),
        "transforms": ["sentencepiece", "prefix"],
        "src_prefix": TO_CYRILLIC_TAG,
        "tgt_prefix": "",
    }
    config = type(eole_run.config)(**cfg)
    config._validate_data_config()

    assert set(get_corpora(config, task=CorpusTask.TRAIN)) == {
        "cyrl2mong",
        "mong2cyrl",
        "valid_mong2cyrl",
    }
    assert set(get_corpora(config, task=CorpusTask.VALID)) == {"valid"}


# --- validation: no variable references -------------------------------------------------


@pytest.mark.parametrize(
    ("render_arguments", "key_path"),
    [
        ({"data_dir": Path("work/corp$us")}, "data.cyrl2mong.path_src"),
        ({"spm_model": Path("work/$HOME.model")}, "transforms_configs.sentencepiece"),
        ({"run_dir": Path("work/run$x")}, "save_data"),
    ],
    ids=["data_dir", "spm_model", "run_dir"],
)
def test_a_dollar_sign_in_a_rendered_path_is_rejected_with_its_key_path(
    render_arguments: dict[str, Path], key_path: str
) -> None:
    arguments = {"data_dir": DATA_DIR, "spm_model": SPM_MODEL, "run_dir": RUN_DIR}
    arguments.update(render_arguments)
    cfg = render_converter_config("tiny", **arguments)

    with pytest.raises(ConfigError, match=re.escape(key_path)) as raised:
        validate_converter_config(cfg)

    assert "$" in str(raised.value)


def test_a_dollar_sign_in_a_list_item_is_rejected_with_its_index() -> None:
    cfg = _render("tiny")
    cfg["data"]["cyrl2mong"]["transforms"][1] = "pre$fix"

    with pytest.raises(ConfigError, match=re.escape("data.cyrl2mong.transforms[1]")):
        validate_converter_config(cfg)


@pytest.mark.parametrize(
    ("where", "key", "key_path"),
    [
        ("training", "cost$", "training.cost$"),
        ("model", "$HOME", "model.$HOME"),
        ("data.valid", "x${y}", "data.valid.x${y}"),
    ],
)
def test_a_dollar_sign_in_a_key_is_rejected_with_its_key_path(
    where: str, key: str, key_path: str
) -> None:
    # Eole expands the whole YAML text, so a key is rewritten like a value.
    cfg = _render("tiny")
    _node(cfg, where)[key] = 1

    with pytest.raises(ConfigError, match=re.escape(key_path)) as raised:
        validate_converter_config(cfg)

    assert "$" in str(raised.value)


def test_a_dollar_sign_in_a_key_below_a_list_is_rejected_with_its_index() -> None:
    cfg = _render("tiny")
    cfg["model"]["layer_overrides"] = [{"heads": 4}, {"note$": 1}]

    with pytest.raises(ConfigError, match=re.escape("model.layer_overrides[1].note$")):
        validate_converter_config(cfg)


def test_a_non_string_value_is_not_a_variable_reference() -> None:
    cfg = _render("tiny")
    cfg["training"]["note"] = 5
    cfg["training"]["rate"] = 1.5
    cfg["training"]["flag"] = True

    validate_converter_config(cfg)


# --- read_converter_paths ----------------------------------------------------------


def _write_rendered(tmp_path: Path, profile: Profile = "tiny") -> Path:
    path = tmp_path / "converter.yaml"
    dump_config(_render(profile), path)
    return path


@pytest.mark.parametrize("profile", PROFILES)
def test_the_converter_paths_are_the_checkpoint_directory_and_the_character_model(
    tmp_path: Path, profile: Profile
) -> None:
    paths = read_converter_paths(_write_rendered(tmp_path, profile))

    assert paths == ConverterPaths(
        checkpoint_dir=RUN_DIR / "ckpt", char_model=SPM_MODEL
    )


def test_reading_converter_paths_of_a_missing_file_is_a_config_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="not found"):
        read_converter_paths(tmp_path / "absent.yaml")


@pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", "a: [unclosed\n"])
def test_reading_converter_paths_of_text_that_is_not_a_config_is_a_config_error(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "converter.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="converter.yaml"):
        read_converter_paths(path)


@pytest.mark.parametrize(
    "dotted",
    [
        "training.model_path",
        "transforms_configs.sentencepiece.src_subword_model",
    ],
)
@pytest.mark.parametrize("problem", ["missing", "not_a_path"])
def test_reading_converter_paths_names_the_key_that_is_missing_or_not_a_path(
    tmp_path: Path, dotted: str, problem: str
) -> None:
    cfg = _render("tiny")
    *parents, leaf = dotted.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    if problem == "missing":
        del node[leaf]
    else:
        node[leaf] = ["a", "list"]
    path = tmp_path / "converter.yaml"
    dump_config(cfg, path)

    with pytest.raises(ConfigError, match=re.escape(dotted)):
        read_converter_paths(path)
