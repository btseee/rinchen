"""Tests for the schema half of `rinchen.train.eole_cfg`: `validate_train_config`
(what a multilingual training config may hold), `merge` and `render`.

The shared half (forbidden keys, `$` guard, reading the YAML) has its tests in
`test_train_eole_cfg_shared.py`, the allow-lists against the installed Eole in
`test_train_eole_cfg_drift.py`, the `merge` properties in
`tests/property/test_train_eole_cfg_merge_property.py`. No test starts Eole.

The baseline is `tests.support.eole_train_config.make_config`, the spike's working
multilingual config. Each rule is tested by breaking that config in one place and
reading the message: `invalid Eole config: <path>: ...`.
"""

from __future__ import annotations

import collections
import copy
import datetime
import enum
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError
from rinchen.train.eole_cfg import (
    FORBIDDEN_KEYS,
    merge,
    merged_config,
    render,
    validate_train_config,
)
from tests.support.eole_train_config import BT_CORPUS, CORPUS, make_config

PREFIX = "invalid Eole config: "
DROP = object()  # in a case: remove the key instead of setting it


def broken(*path_and_value: Any) -> dict[str, Any]:
    """The baseline with the entry at the key path set to the value, or removed
    when the value is `DROP`. The last argument is the value."""
    *path, value = path_and_value
    cfg = make_config()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    if value is DROP:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return cfg


def message_of(cfg: object) -> str:
    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg)
    return str(raised.value)


def test_the_spikes_working_config_is_valid() -> None:
    assert validate_train_config(make_config()) is None


def test_the_baseline_is_not_changed_by_validation() -> None:
    cfg = make_config()

    validate_train_config(cfg)

    assert cfg == make_config()


# ---- a config of the wrong shape ------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "type_name"),
    [([], "list"), ("seed: 1", "str"), (None, "NoneType"), (3, "int"), (True, "bool")],
    ids=["list", "string", "none", "int", "bool"],
)
def test_a_config_that_is_not_a_mapping_is_refused_by_type(
    cfg: object, type_name: str
) -> None:
    assert message_of(cfg) == f"{PREFIX}config: expected a mapping, got {type_name}"


@pytest.mark.parametrize(
    ("path", "value", "where"),
    [
        (("data",), [], "data"),
        (("data",), "en-mn", "data"),
        (("data",), None, "data"),
        (("data", CORPUS), "path_src: x", f"data.{CORPUS}"),
        (("data", CORPUS), ["path_src"], f"data.{CORPUS}"),
        (("data", CORPUS), None, f"data.{CORPUS}"),
        (("data", "valid"), "path_src: x", "data.valid"),
        (("transforms_configs",), "dict()", "transforms_configs"),
        (("transforms_configs",), [], "transforms_configs"),
        (
            ("transforms_configs", "sentencepiece"),
            3,
            "transforms_configs.sentencepiece",
        ),
        (
            ("transforms_configs", "filtertoolong"),
            [],
            "transforms_configs.filtertoolong",
        ),
        (("model",), "transformer", "model"),
        (("model",), [], "model"),
        (("model", "encoder"), "dict(sliding_window=4)", "model.encoder"),
        (("model", "decoder"), "__import__('os')", "model.decoder"),
        (("model", "encoder"), None, "model.encoder"),
        (("model", "embeddings"), [], "model.embeddings"),
        (("model", "embeddings"), "x", "model.embeddings"),
        (("training",), "dict(train_steps=1)", "training"),
        (("training",), [], "training"),
        (("training",), None, "training"),
    ],
)
def test_a_section_that_is_not_a_mapping_is_refused_by_its_path_never_by_a_crash(
    path: tuple[str, ...], value: object, where: str
) -> None:
    # A string given for `encoder`, `decoder`, `training` or `transforms_configs` is
    # evaluated as Python by Eole, so it is not a spelling of a mapping.
    message = message_of(broken(*path, value))

    assert message.startswith(f"{PREFIX}{where}: expected a mapping, got ")


def test_the_model_section_is_required() -> None:
    assert message_of(broken("model", DROP)) == f"{PREFIX}model: is missing"


@pytest.mark.parametrize("name", [1, None, 2.5, True])
def test_a_corpus_name_that_is_not_text_is_refused(name: object) -> None:
    cfg = make_config()
    cfg["data"][name] = cfg["data"].pop(BT_CORPUS)

    assert message_of(cfg).startswith(f"{PREFIX}data.{name}: the corpus name must be")


# ---- unknown keys, level by level -----------------------------------------------


@pytest.mark.parametrize(
    ("path", "level"),
    [
        ((), "the top level"),
        (("data", CORPUS), "a data corpus"),
        (("data", "valid"), "a data corpus"),
        (("transforms_configs",), "transforms_configs"),
        (("transforms_configs", "sentencepiece"), "transforms_configs.sentencepiece"),
        (("transforms_configs", "filtertoolong"), "transforms_configs.filtertoolong"),
        (("model",), "model"),
        (("model", "encoder"), "model.encoder"),
        (("model", "decoder"), "model.decoder"),
        (("model", "embeddings"), "model.embeddings"),
        (("training",), "training"),
    ],
)
def test_an_unknown_key_is_named_with_its_path_and_its_level(
    path: tuple[str, ...], level: str
) -> None:
    cfg = make_config()
    node = cfg
    for key in path:
        node = node[key]
    node["bogus_key"] = 1

    where = ".".join([*path, "bogus_key"])
    assert message_of(cfg) == f"{PREFIX}{where}: unknown key in {level}"


@pytest.mark.parametrize(
    ("path", "level"),
    [
        (("train_from",), "the top level"),
        (("training", "train_from"), "training"),
        (("training", "lora_rank"), "training"),
        (("training", "update_vocab"), "training"),
        (("transforms_configs", "prefix"), "transforms_configs"),
        (("model", "layers"), "model"),
        (("model", "embeddings", "src_word_vec_size"), "model.embeddings"),
        (("data", CORPUS, "src_suffix"), "a data corpus"),
        (("data", CORPUS, "path_align"), "a data corpus"),
    ],
)
def test_a_key_that_continues_from_or_adapts_a_model_or_that_no_run_sets_is_unknown(
    path: tuple[str, ...], level: str
) -> None:
    cfg = make_config()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = 1

    assert message_of(cfg) == f"{PREFIX}{'.'.join(path)}: unknown key in {level}"


def test_a_key_that_is_not_text_is_unknown_at_its_level() -> None:
    cfg = make_config()
    cfg["training"][7] = 1

    assert message_of(cfg) == f"{PREFIX}training.7: unknown key in training"


def test_a_typo_in_a_training_key_is_named() -> None:
    cfg = broken("training", "train_stpes", 5)

    assert message_of(cfg) == f"{PREFIX}training.train_stpes: unknown key in training"


# ---- forbidden keys, at any depth -----------------------------------------------

FORBIDDEN_PLACES = [
    pytest.param((), id="top-level"),
    pytest.param(("model",), id="model"),
    pytest.param(("model", "encoder"), id="encoder"),
    pytest.param(("model", "decoder"), id="decoder"),
    pytest.param(("model", "embeddings"), id="embeddings"),
    pytest.param(("training",), id="training"),
    pytest.param(("data", CORPUS), id="corpus"),
    pytest.param(("transforms_configs", "filtertoolong"), id="transform"),
]


@pytest.mark.parametrize("place", FORBIDDEN_PLACES)
@pytest.mark.parametrize("key", sorted(FORBIDDEN_KEYS))
def test_every_forbidden_key_is_refused_at_every_level_before_the_unknown_key_check(
    key: str, place: tuple[str, ...]
) -> None:
    cfg = make_config()
    node = cfg
    for part in place:
        node = node[part]
    node[key] = None  # even a neutral value: the key itself is the switch

    where = ".".join([*place, key])
    assert message_of(cfg).startswith(f"{PREFIX}{where}: is not allowed")


@pytest.mark.parametrize("key", sorted(FORBIDDEN_KEYS))
def test_a_forbidden_key_nested_deep_or_inside_a_list_is_found(key: str) -> None:
    cfg = make_config()
    cfg["training"]["accum_count"] = [{"nested": [{key: 4}]}]

    assert message_of(cfg).startswith(
        f"{PREFIX}training.accum_count[0].nested[0].{key}: is not allowed"
    )


def test_the_forbidden_message_says_what_every_model_uses() -> None:
    message = message_of(broken("model", "sliding_window", 128))

    assert message == (
        f"{PREFIX}model.sliding_window: is not allowed: every Rinchen model uses "
        "sinusoidal positions and full multi-head attention"
    )


# ---- position encoding ----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["Rotary", "Relative", "Alibi", "sinusoidal", "SinusoidalConcat", "", None, 1],
)
def test_the_embeddings_position_encoding_must_be_exactly_sinusoidal_interleaved(
    value: object,
) -> None:
    cfg = broken("model", "embeddings", "position_encoding_type", value)

    assert message_of(cfg) == (
        f"{PREFIX}model.embeddings.position_encoding_type: must be "
        f"SinusoidalInterleaved, got {value!r}"
    )


def test_the_embeddings_position_encoding_must_be_set_not_left_to_eoles_default() -> (
    None
):
    cfg = broken("model", "embeddings", "position_encoding_type", DROP)

    assert message_of(cfg) == (
        f"{PREFIX}model.embeddings.position_encoding_type: must be "
        "SinusoidalInterleaved, but it is missing"
    )


def test_a_missing_embeddings_section_is_named_before_its_position_encoding() -> None:
    assert message_of(broken("model", "embeddings", DROP)) == (
        f"{PREFIX}model.embeddings: is missing"
    )


@pytest.mark.parametrize("layer", ["encoder", "decoder"])
@pytest.mark.parametrize("value", ["Rotary", "Relative", "sinusoidal", None])
def test_a_position_encoding_on_the_encoder_or_decoder_must_match_too(
    layer: str, value: object
) -> None:
    cfg = broken("model", layer, "position_encoding_type", value)

    assert message_of(cfg) == (
        f"{PREFIX}model.{layer}.position_encoding_type: must be "
        f"SinusoidalInterleaved, got {value!r}"
    )


@pytest.mark.parametrize("layer", ["encoder", "decoder"])
def test_a_matching_position_encoding_on_the_encoder_or_decoder_is_accepted(
    layer: str,
) -> None:
    cfg = broken("model", layer, "position_encoding_type", "SinusoidalInterleaved")

    assert validate_train_config(cfg) is None


# ---- model.architecture ---------------------------------------------------------


@pytest.mark.parametrize("value", ["transformer_lm", "rnn", "Transformer", None, 1])
def test_the_architecture_must_be_the_one_the_allow_lists_describe(
    value: object,
) -> None:
    cfg = broken("model", "architecture", value)

    assert message_of(cfg) == (
        f"{PREFIX}model.architecture: must be 'transformer', got {value!r}: the "
        "allow-lists are those of that architecture"
    )


def test_a_missing_architecture_is_refused_before_eole_does() -> None:
    cfg = broken("model", "architecture", DROP)

    assert message_of(cfg) == (
        f"{PREFIX}model.architecture: must be 'transformer', but it is missing: "
        "the allow-lists are those of that architecture"
    )


# ---- no variable references -----------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", CORPUS, "path_src"), "data/$CORPUS/en.src"),
        (("training", "model_path"), "$HOME/ckpt"),
        (("src_vocab",), "run/vocab$1.src"),
        (("transforms_configs", "sentencepiece", "src_subword_model"), "$"),
        (("data", CORPUS, "src_prefix"), "<2khk_Cyrl>$"),
        (("model", "architecture"), "transformer$"),
    ],
)
def test_a_dollar_sign_in_a_string_is_refused_with_its_path(
    path: tuple[str, ...], value: str
) -> None:
    cfg = broken(*path, value)

    assert message_of(cfg) == (
        f"{PREFIX}{'.'.join(path)}: holds '$', which Eole would expand as a "
        f"variable: {value!r}"
    )


def test_a_dollar_sign_in_a_list_item_is_refused_with_the_items_index() -> None:
    cfg = broken("data", CORPUS, "transforms", ["sentencepiece", "pre$fix"])

    assert message_of(cfg).startswith(f"{PREFIX}data.{CORPUS}.transforms[1]: holds '$'")


def test_a_dollar_sign_in_a_key_is_refused_before_the_unknown_key_check() -> None:
    cfg = broken("training", "lr$", 1)

    assert message_of(cfg) == (
        f"{PREFIX}training.lr$: the key holds '$', which Eole would expand as a "
        "variable"
    )


def test_a_dollar_sign_in_a_corpus_name_is_refused() -> None:
    cfg = make_config()
    cfg["data"]["en$mn"] = cfg["data"].pop(BT_CORPUS)

    assert message_of(cfg).startswith(f"{PREFIX}data.en$mn: the key holds '$'")


# ---- plain data only ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (Path("run/ckpt"), "PosixPath"),
        ({"a"}, "set"),
        (b"bytes", "bytes"),
        (datetime.date(2026, 11, 8), "date"),
        (object(), "object"),
    ],
)
def test_a_value_that_is_not_plain_yaml_data_is_refused_so_that_it_cannot_be_dumped_wrongly(
    value: object, type_name: str
) -> None:
    cfg = broken("training", "model_path", value)

    assert message_of(cfg) == (
        f"{PREFIX}training.model_path: must be plain YAML data (text, a number, "
        f"true or false, null, a list or a mapping), got {type_name}"
    )


class _Color(enum.StrEnum):
    RED = "red"


class _Count(enum.IntEnum):
    ONE = 1


class _Rate(float):
    pass


class _Names(list[str]):
    pass


class _Section(collections.OrderedDict[str, Any]):
    pass


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (_Color.RED, "_Color"),
        (_Count.ONE, "_Count"),
        (_Rate(0.5), "_Rate"),
        (_Names(["a"]), "_Names"),
        (_Section(a=1), "_Section"),
    ],
    ids=["StrEnum", "IntEnum", "float subclass", "list subclass", "dict subclass"],
)
def test_a_subclass_of_a_plain_type_is_refused_because_yaml_cannot_dump_it(
    value: object, type_name: str
) -> None:
    # `isinstance` would let these through, and `render` would then raise yaml's own
    # RepresenterError.
    cfg = broken("training", "model_path", value)

    assert message_of(cfg) == (
        f"{PREFIX}training.model_path: must be plain YAML data (text, a number, "
        f"true or false, null, a list or a mapping), got {type_name}"
    )


def test_a_subclass_in_a_list_is_refused_with_the_index_in_the_path() -> None:
    cfg = broken("training", "gpu_ranks", [0, _Count.ONE])

    assert message_of(cfg).startswith(
        f"{PREFIX}training.gpu_ranks[1]: must be plain YAML data"
    )


def test_the_plain_types_themselves_are_not_refused_as_subclasses() -> None:
    cfg = broken("training", "learning_rate", 0.5)
    cfg["training"]["dropout"] = True
    cfg["training"]["gpu_ranks"] = (0, 1)
    cfg["seed"] = None

    assert validate_train_config(cfg) is None


def test_numbers_booleans_null_lists_tuples_and_floats_are_plain_data() -> None:
    cfg = broken("training", "learning_rate", 0.5)
    cfg["training"]["gpu_ranks"] = (0, 1)
    cfg["seed"] = None

    assert validate_train_config(cfg) is None


def test_a_mapping_that_contains_itself_is_refused_instead_of_recursing_forever() -> (
    None
):
    cfg = make_config()
    cfg["training"]["accum_count"] = [cfg["training"]]

    assert message_of(cfg) == (
        f"{PREFIX}training.accum_count[0]: refers back to a mapping or list that "
        "contains it"
    )


def test_the_same_list_used_twice_is_not_a_cycle() -> None:
    cfg = make_config()
    shared = ["sentencepiece", "prefix", "filtertoolong"]
    cfg["data"][CORPUS]["transforms"] = shared
    cfg["data"][BT_CORPUS]["transforms"] = shared

    assert validate_train_config(cfg) is None


# ---- overwrite and share_vocab --------------------------------------------------


@pytest.mark.parametrize("value", [False, DROP, "true", 1, None])
def test_overwrite_must_be_true_so_a_rerun_into_a_used_directory_does_not_stop(
    value: object,
) -> None:
    cfg = broken("overwrite", value)

    got = "but it is missing" if value is DROP else f"got {value!r}"
    assert message_of(cfg) == (
        f"{PREFIX}overwrite: must be true: Eole stops at an existing vocabulary "
        f"file otherwise, {got}"
    )


@pytest.mark.parametrize("value", [False, DROP, "true", 1, None])
def test_share_vocab_must_be_true_for_the_shared_embeddings(value: object) -> None:
    cfg = broken("share_vocab", value)

    got = "but it is missing" if value is DROP else f"got {value!r}"
    assert message_of(cfg) == (
        f"{PREFIX}share_vocab: must be true: every Rinchen model shares one "
        f"vocabulary, {got}"
    )


# ---- hidden size and embedding size ---------------------------------------------


@pytest.mark.parametrize("word_vec_size", [128, 512, 32, "64", None, 64.5])
def test_the_embedding_size_must_equal_the_hidden_size(word_vec_size: object) -> None:
    cfg = broken("model", "embeddings", "word_vec_size", word_vec_size)

    assert message_of(cfg) == (
        f"{PREFIX}model.embeddings.word_vec_size: must equal model.hidden_size "
        f"(64), got {word_vec_size!r}"
    )


def test_a_hidden_size_that_is_left_to_eoles_default_is_refused() -> None:
    cfg = broken("model", "hidden_size", DROP)

    assert message_of(cfg) == (
        f"{PREFIX}model.hidden_size: is missing: Eole's default is 512, which "
        "breaks a smaller model"
    )


def test_an_embedding_size_that_is_left_to_eoles_default_is_refused() -> None:
    cfg = broken("model", "embeddings", "word_vec_size", DROP)

    assert message_of(cfg) == (
        f"{PREFIX}model.embeddings.word_vec_size: is missing: Eole's default is "
        "512, which breaks a smaller model"
    )


@pytest.mark.parametrize("layer", ["encoder", "decoder"])
def test_a_hidden_size_set_on_the_encoder_or_decoder_must_equal_the_embedding_size(
    layer: str,
) -> None:
    cfg = broken("model", layer, "hidden_size", 128)

    assert message_of(cfg) == (
        f"{PREFIX}model.{layer}.hidden_size: must equal "
        "model.embeddings.word_vec_size (64), got 128"
    )


@pytest.mark.parametrize("layer", ["encoder", "decoder"])
def test_a_hidden_size_on_the_encoder_or_decoder_that_equals_the_rest_is_accepted(
    layer: str,
) -> None:
    cfg = broken("model", layer, "hidden_size", 64)

    assert validate_train_config(cfg) is None


def test_a_larger_model_with_equal_sizes_is_accepted() -> None:
    cfg = make_config()
    cfg["model"]["hidden_size"] = 512
    cfg["model"]["embeddings"]["word_vec_size"] = 512
    cfg["model"]["encoder"]["layers"] = 6
    cfg["model"]["decoder"]["hidden_size"] = 512

    assert validate_train_config(cfg) is None


# ---- the training corpora -------------------------------------------------------


@pytest.mark.parametrize("key", ["path_src", "path_tgt", "transforms"])
def test_a_training_corpus_without_a_path_or_transforms_is_refused(key: str) -> None:
    cfg = broken("data", CORPUS, key, DROP)

    assert message_of(cfg) == f"{PREFIX}data.{CORPUS}.{key}: is missing"


@pytest.mark.parametrize("key", ["src_prefix", "tgt_prefix"])
def test_a_training_corpus_without_a_prefix_key_is_refused_for_the_type_error_it_makes(
    key: str,
) -> None:
    cfg = broken("data", CORPUS, key, DROP)

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.{key}: is missing: Eole's prefix transform fails "
        "with a TypeError without it; use '' for no prefix"
    )


@pytest.mark.parametrize("key", ["src_prefix", "tgt_prefix"])
@pytest.mark.parametrize("value", [None, 5, ["<2xx>"], True])
def test_a_prefix_that_is_not_a_string_is_refused_as_it_makes_the_same_type_error(
    key: str, value: object
) -> None:
    cfg = broken("data", CORPUS, key, value)

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.{key}: must be a string ('' for none), got {value!r}"
    )


@pytest.mark.parametrize("key", ["path_src", "path_tgt"])
@pytest.mark.parametrize("value", ["", None, 3, ["a.src"]])
def test_a_corpus_path_must_be_a_non_empty_string(key: str, value: object) -> None:
    cfg = broken("data", CORPUS, key, value)

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.{key}: must be a non-empty path, got {value!r}"
    )


@pytest.mark.parametrize("value", ["sentencepiece", None, {"a": 1}, 3])
def test_the_transforms_of_a_training_corpus_must_be_a_list_of_names(
    value: object,
) -> None:
    cfg = broken("data", CORPUS, "transforms", value)

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.transforms: must be a list of transform names, "
        f"got {value!r}"
    )


def test_a_transform_name_that_is_not_a_string_is_refused() -> None:
    cfg = broken("data", CORPUS, "transforms", ["sentencepiece", 3])

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.transforms: must be a list of transform names, "
        "got ['sentencepiece', 3]"
    )


def test_every_training_corpus_is_checked_not_only_the_first() -> None:
    cfg = broken("data", BT_CORPUS, "tgt_prefix", DROP)

    assert message_of(cfg).startswith(
        f"{PREFIX}data.{BT_CORPUS}.tgt_prefix: is missing"
    )


def test_a_corpus_named_like_a_language_pair_is_not_mistaken_for_valid() -> None:
    cfg = make_config()
    cfg["data"]["valid_en-mn"] = cfg["data"][CORPUS]
    cfg["data"]["valid_en-mn"]["src_prefix"] = "<2khk_Cyrl>"

    assert validate_train_config(cfg) is None


def test_an_empty_prefix_is_a_valid_prefix() -> None:
    cfg = broken("data", CORPUS, "src_prefix", "")
    cfg["data"][CORPUS]["tgt_prefix"] = "<2khk_Cyrl>"

    assert validate_train_config(cfg) is None


def test_a_training_corpus_may_leave_the_weight_out_which_means_one() -> None:
    cfg = broken("data", CORPUS, "weight", DROP)

    assert validate_train_config(cfg) is None


@pytest.mark.parametrize("weight", [1, 2, 7, 100, 10**9])
def test_a_positive_integer_weight_is_accepted(weight: int) -> None:
    assert validate_train_config(broken("data", CORPUS, "weight", weight)) is None


@pytest.mark.parametrize(
    "weight", [0, -1, 1.5, 2.0, "2", True, False, None, [1], 0.0], ids=repr
)
def test_a_weight_that_is_not_a_positive_integer_is_refused(weight: object) -> None:
    cfg = broken("data", CORPUS, "weight", weight)

    assert message_of(cfg) == (
        f"{PREFIX}data.{CORPUS}.weight: must be a positive integer (the number of "
        f"consecutive examples taken per round), got {weight!r}"
    )


# ---- the validation corpus ------------------------------------------------------


def test_the_validation_corpus_may_be_pre_tokenised_with_no_transforms() -> None:
    assert validate_train_config(make_config()) is None


def test_the_validation_corpus_may_leave_the_transforms_out() -> None:
    assert validate_train_config(broken("data", "valid", "transforms", DROP)) is None


def test_a_config_may_have_no_validation_corpus() -> None:
    assert validate_train_config(broken("data", "valid", DROP)) is None


@pytest.mark.parametrize("key", ["src_prefix", "tgt_prefix"])
@pytest.mark.parametrize("value", ["", "<2khk_Cyrl>"])
def test_the_validation_corpus_takes_no_prefix_key_even_an_empty_one(
    key: str, value: str
) -> None:
    cfg = broken("data", "valid", key, value)

    assert message_of(cfg) == (
        f"{PREFIX}data.valid.{key}: not allowed on the validation corpus: Eole "
        "copies a prefix into the checkpoint, where it overrides every prediction"
    )


def test_the_validation_corpus_takes_no_weight() -> None:
    cfg = broken("data", "valid", "weight", 1)

    assert message_of(cfg).startswith(
        f"{PREFIX}data.valid.weight: not allowed on the validation corpus"
    )


@pytest.mark.parametrize(
    "transforms",
    [["sentencepiece"], ["sentencepiece", "prefix"], ["prefix"], "sentencepiece", None],
)
def test_the_validation_corpus_takes_no_transform_because_the_harness_tokenises_it(
    transforms: object,
) -> None:
    cfg = broken("data", "valid", "transforms", transforms)

    assert message_of(cfg) == (
        f"{PREFIX}data.valid.transforms: must be [] or left out: the harness "
        f"tokenises the validation corpus, got {transforms!r}"
    )


@pytest.mark.parametrize("key", ["path_src", "path_tgt"])
def test_the_validation_corpus_needs_both_paths(key: str) -> None:
    cfg = broken("data", "valid", key, DROP)

    assert message_of(cfg) == f"{PREFIX}data.valid.{key}: is missing"


@pytest.mark.parametrize("key", ["path_src", "path_tgt"])
def test_the_validation_corpus_paths_must_be_non_empty_strings(key: str) -> None:
    cfg = broken("data", "valid", key, "")

    assert message_of(cfg) == (
        f"{PREFIX}data.valid.{key}: must be a non-empty path, got ''"
    )


# ---- the order in which the rules are checked -----------------------------------


def test_the_forbidden_key_is_reported_before_the_unknown_key_that_it_also_is() -> None:
    cfg = broken("model", "encoder", "sliding_window", 4)
    cfg["model"]["encoder"]["bogus"] = 1

    assert "is not allowed" in message_of(cfg)


def test_a_forbidden_key_is_reported_before_a_dollar_sign() -> None:
    cfg = broken("model", "sliding_window", 128)
    cfg["training"]["model_path"] = "$HOME/ckpt"

    assert "is not allowed" in message_of(cfg)


def test_a_dollar_sign_is_reported_before_an_unknown_key_in_the_same_section() -> None:
    cfg = broken("training", "bogus", 1)
    cfg["training"]["model_path"] = "$HOME/ckpt"

    assert "holds '$'" in message_of(cfg)


def test_an_unknown_key_is_reported_before_a_wrong_position_encoding() -> None:
    cfg = broken("model", "embeddings", "position_encoding_type", "Rotary")
    cfg["model"]["bogus"] = 1

    assert message_of(cfg) == f"{PREFIX}model.bogus: unknown key in model"


def test_a_wrong_architecture_is_reported_before_a_wrong_position_encoding() -> None:
    cfg = broken("model", "embeddings", "position_encoding_type", "Rotary")
    cfg["model"]["architecture"] = "rnn"

    assert message_of(cfg).startswith(f"{PREFIX}model.architecture")


def test_a_wrong_position_encoding_is_reported_before_the_hidden_size() -> None:
    cfg = broken("model", "embeddings", "position_encoding_type", "Rotary")
    cfg["model"]["embeddings"]["word_vec_size"] = 128

    assert "position_encoding_type" in message_of(cfg)


def test_the_hidden_size_is_reported_before_the_switches() -> None:
    cfg = broken("model", "embeddings", "word_vec_size", 128)
    cfg["overwrite"] = False

    assert message_of(cfg).startswith(f"{PREFIX}model.embeddings.word_vec_size")


def test_the_switches_are_reported_before_the_corpora() -> None:
    cfg = broken("overwrite", False)
    cfg["data"][CORPUS]["weight"] = 0

    assert message_of(cfg).startswith(f"{PREFIX}overwrite")


def test_corpora_are_checked_in_the_order_of_the_config() -> None:
    cfg = broken("data", CORPUS, "weight", 0)
    del cfg["data"][BT_CORPUS]["src_prefix"]

    assert message_of(cfg).startswith(f"{PREFIX}data.{CORPUS}.weight")


# ---- the keys a teacher run sets -------------------------------------------------

TEACHER_KEYS = [
    pytest.param(("tensorboard",), True, id="tensorboard"),
    pytest.param(("tensorboard_log_dir",), "run/tensorboard", id="tensorboard_log_dir"),
    pytest.param(("log_file",), "run/train.log", id="log_file"),
    pytest.param(("valid_metrics",), ["BLEU", "CHRF"], id="valid_metrics"),
    pytest.param(("valid_metrics",), [], id="valid_metrics-empty"),
    pytest.param(("training", "early_stopping"), 6, id="early_stopping"),
    pytest.param(("training", "early_stopping"), 0, id="early_stopping-off"),
    pytest.param(("training", "early_stopping_criteria"), "ppl", id="criteria-text"),
    pytest.param(
        ("training", "early_stopping_criteria"), ["ppl", "accuracy"], id="criteria-list"
    ),
    pytest.param(("training", "timeout"), 7200, id="timeout"),
    pytest.param(("training", "master_ip"), "10.0.0.7", id="master_ip"),
    pytest.param(("training", "master_port"), 1, id="master_port-low"),
    pytest.param(("training", "master_port"), 65535, id="master_port-high"),
    pytest.param(("training", "gpu_backend"), "gloo", id="gpu_backend"),
    pytest.param(("training", "normalization"), "tokens", id="normalization-tokens"),
    pytest.param(("training", "normalization"), "sents", id="normalization-sents"),
    pytest.param(("training", "batch_size_multiple"), 8, id="batch_size_multiple"),
    pytest.param(("training", "self_attn_backend"), "pytorch", id="self_attn_backend"),
]


@pytest.mark.parametrize(("path", "value"), TEACHER_KEYS)
def test_a_key_a_teacher_run_sets_is_accepted_at_its_level(
    path: tuple[str, ...], value: object
) -> None:
    assert validate_train_config(broken(*path, value)) is None


@pytest.mark.parametrize(
    ("path", "value", "problem"),
    [
        (("tensorboard",), 1, "must be true or false"),
        (("tensorboard",), "true", "must be true or false"),
        (("tensorboard",), None, "must be true or false"),
        (("tensorboard_log_dir",), "", "must be a non-empty string"),
        (("tensorboard_log_dir",), 5, "must be a non-empty string"),
        (("log_file",), "", "must be a non-empty string"),
        (("log_file",), ["a.log"], "must be a non-empty string"),
        (("valid_metrics",), "BLEU", "must be a list of non-empty strings"),
        (("valid_metrics",), ["BLEU", 5], "must be a list of non-empty strings"),
        (("valid_metrics",), ["BLEU", ""], "must be a list of non-empty strings"),
        (("valid_metrics",), None, "must be a list of non-empty strings"),
        (("training", "early_stopping"), True, "must be a non-negative integer"),
        (("training", "early_stopping"), False, "must be a non-negative integer"),
        (("training", "early_stopping"), -1, "must be a non-negative integer"),
        (("training", "early_stopping"), 2.0, "must be a non-negative integer"),
        (("training", "early_stopping"), "6", "must be a non-negative integer"),
        (("training", "early_stopping_criteria"), "", "must be a non-empty string"),
        (("training", "early_stopping_criteria"), [], "must be a non-empty string"),
        (("training", "early_stopping_criteria"), ["ppl", 1], "must be a non-empty"),
        (("training", "early_stopping_criteria"), ["ppl", ""], "must be a non-empty"),
        (("training", "early_stopping_criteria"), 3, "must be a non-empty string"),
        (("training", "early_stopping_criteria"), None, "must be a non-empty string"),
        (("training", "timeout"), 0, "must be a positive integer"),
        (("training", "timeout"), -60, "must be a positive integer"),
        (("training", "timeout"), True, "must be a positive integer"),
        (("training", "timeout"), 60.0, "must be a positive integer"),
        (("training", "timeout"), "7200", "must be a positive integer"),
        (("training", "master_ip"), "", "must be a non-empty string"),
        (("training", "master_ip"), 7, "must be a non-empty string"),
        (("training", "master_port"), 0, "must be an integer from 1 to 65535"),
        (("training", "master_port"), 65536, "must be an integer from 1 to 65535"),
        (("training", "master_port"), -1, "must be an integer from 1 to 65535"),
        (("training", "master_port"), True, "must be an integer from 1 to 65535"),
        (("training", "master_port"), "10000", "must be an integer from 1 to 65535"),
        (("training", "gpu_backend"), "", "must be a non-empty string"),
        (("training", "gpu_backend"), None, "must be a non-empty string"),
        (("training", "normalization"), "words", "must be 'sents' or 'tokens'"),
        (("training", "normalization"), "", "must be 'sents' or 'tokens'"),
        (("training", "batch_size_multiple"), 0, "must be a positive integer"),
        (("training", "batch_size_multiple"), True, "must be a positive integer"),
        (("training", "batch_size_multiple"), 8.0, "must be a positive integer"),
        (("training", "self_attn_backend"), "", "must be a non-empty string"),
        (("training", "self_attn_backend"), None, "must be a non-empty string"),
    ],
)
def test_a_wrong_value_for_a_teacher_key_is_named_with_its_path_and_the_value(
    path: tuple[str, ...], value: object, problem: str
) -> None:
    message = message_of(broken(*path, value))

    assert message.startswith(f"{PREFIX}{'.'.join(path)}: {problem}")
    assert message.endswith(f", got {value!r}")


@pytest.mark.parametrize(
    "key",
    ["average_decay", "average_every", "accum_steps", "dropout_steps"],
)
def test_a_key_whose_value_would_not_survive_a_resume_stays_unknown(key: str) -> None:
    # The EMA replaces the live weights in every saved checkpoint and is not
    # restored; a scheduled value restarts at its first entry after a resume.
    assert message_of(broken("training", key, 1)) == (
        f"{PREFIX}training.{key}: unknown key in training"
    )


@pytest.mark.parametrize("key", ["reset_optim", "update_vocab"])
def test_a_key_that_changes_how_a_checkpoint_is_loaded_stays_unknown(key: str) -> None:
    assert message_of(broken("training", key, "none")) == (
        f"{PREFIX}training.{key}: unknown key in training"
    )


@pytest.mark.parametrize(
    "key",
    ["average_decay", "average_every", "accum_steps", "dropout_steps"]
    + ["reset_optim", "update_vocab"],
)
def test_the_resume_flag_does_not_make_those_keys_known(key: str) -> None:
    cfg = resume_config()
    cfg["training"][key] = 1

    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg, resumes=True)

    assert str(raised.value) == f"{PREFIX}training.{key}: unknown key in training"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training", "timeout"), 0),
        (("training", "early_stopping"), -1),
        (("tensorboard",), "yes"),
    ],
)
def test_a_teacher_value_rule_is_reported_after_the_switches_and_before_the_corpora(
    path: tuple[str, ...], value: object
) -> None:
    cfg = broken(*path, value)
    cfg["data"][CORPUS]["weight"] = 0

    assert message_of(cfg).startswith(f"{PREFIX}{'.'.join(path)}: must be")

    cfg["overwrite"] = False
    assert message_of(cfg).startswith(f"{PREFIX}overwrite: must be true")


# ---- train_from, only in a resume config ------------------------------------------

RUN_CKPT = "run/ckpt"


def resume_config(train_from: object = f"{RUN_CKPT}/step_500") -> dict[str, Any]:
    """The baseline (model_path `run/ckpt`) as a resume config: it names a step
    directory of its own checkpoint directory."""
    cfg = make_config()
    cfg["training"]["train_from"] = train_from
    return cfg


def resume_message(train_from: object, model_path: object = RUN_CKPT) -> str:
    cfg = resume_config(train_from)
    if model_path is DROP:
        del cfg["training"]["model_path"]
    else:
        cfg["training"]["model_path"] = model_path
    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg, resumes=True)
    return str(raised.value)


@pytest.mark.parametrize(
    ("path", "level"),
    [(("train_from",), "the top level"), (("training", "train_from"), "training")],
)
def test_train_from_is_an_unknown_key_unless_the_config_resumes(
    path: tuple[str, ...], level: str
) -> None:
    cfg = make_config()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = "run/ckpt/step_500"

    assert message_of(cfg) == f"{PREFIX}{'.'.join(path)}: unknown key in {level}"


def test_train_from_is_unknown_when_resumes_is_false_explicitly() -> None:
    cfg = resume_config()

    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg, resumes=False)

    assert str(raised.value) == f"{PREFIX}training.train_from: unknown key in training"


def test_train_from_stays_unknown_at_the_top_level_even_when_the_config_resumes() -> (
    None
):
    cfg = make_config()
    cfg["train_from"] = f"{RUN_CKPT}/step_500"

    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg, resumes=True)

    assert str(raised.value) == f"{PREFIX}train_from: unknown key in the top level"


@pytest.mark.parametrize(
    "train_from",
    [
        "run/ckpt/step_500",
        "run/ckpt/step_1",
        "run/ckpt/step_5000000000",
        "run/ckpt/step_500/",
        "run//ckpt/step_500",
        "./run/ckpt/step_500",
        "run/./ckpt/step_500",
    ],
)
def test_a_resume_config_may_name_a_step_directory_of_its_own_model_path(
    train_from: str,
) -> None:
    assert validate_train_config(resume_config(train_from), resumes=True) is None


def test_a_resume_config_may_leave_train_from_out() -> None:
    # The flag allows the key; the resume file's content is the launcher's to check.
    assert validate_train_config(make_config(), resumes=True) is None


def test_the_same_parent_is_compared_as_a_normalised_path_for_an_absolute_model_path() -> (
    None
):
    cfg = resume_config("/runs/a//ckpt/step_70")
    cfg["training"]["model_path"] = "/runs/a/ckpt/"

    assert validate_train_config(cfg, resumes=True) is None


@pytest.mark.parametrize(
    ("model_path", "train_from"),
    [(".", "step_5"), (".", "./step_5"), ("ckpt", "ckpt/step_5"), ("/", "/step_5")],
)
def test_a_model_path_that_is_the_current_or_the_root_directory_still_matches(
    model_path: str, train_from: str
) -> None:
    cfg = resume_config(train_from)
    cfg["training"]["model_path"] = model_path

    assert validate_train_config(cfg, resumes=True) is None


@pytest.mark.parametrize(
    ("train_from", "problem"),
    [
        (None, "must be a string"),
        (500, "must be a string"),
        (["run/ckpt/step_500"], "must be a string"),
        ({"path": "run/ckpt/step_500"}, "must be a string"),
        (True, "must be a string"),
        ("", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/model.00.safetensors", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_500/model.00.safetensors", "must be '<training.model_path>/"),
        ("run/ckpt/step_500/step_500", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_x", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_0", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_007", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_-5", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_5.0", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/step_5 ", "must be '<training.model_path>/step_<N>'"),
        (
            "run/ckpt/step_" + chr(0x665),  # an Arabic-Indic digit five
            "must be '<training.model_path>/step_<N>'",
        ),
        ("run/ckpt/Step_5", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/xstep_5", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt2/step_500", "must be '<training.model_path>/step_<N>'"),
        ("run/step_500", "must be '<training.model_path>/step_<N>'"),
        ("other/run/ckpt/step_500", "must be '<training.model_path>/step_<N>'"),
        ("/run/ckpt/step_500", "must be '<training.model_path>/step_<N>'"),
        ("run/ckpt/sub/step_500", "must be '<training.model_path>/step_<N>'"),
        ("elsewhere/ckpt/step_500", "must be '<training.model_path>/step_<N>'"),
        ("run2/ckpt/step_500", "must be '<training.model_path>/step_<N>'"),
    ],
)
def test_a_resume_config_may_name_nothing_but_a_step_directory_of_its_own_model_path(
    train_from: object, problem: str
) -> None:
    message = resume_message(train_from)

    assert message.startswith(f"{PREFIX}training.train_from: {problem}")
    if not isinstance(train_from, str):
        assert message.endswith(f"got {type(train_from).__name__}")
    else:
        assert repr(train_from) in message


def test_another_run_directorys_step_is_foreign_even_when_its_name_looks_right() -> (
    None
):
    message = resume_message("run-b/ckpt/step_500")

    assert message.startswith(f"{PREFIX}training.train_from: must be '<training.model")
    assert "run/ckpt" in message


@pytest.mark.parametrize(
    "train_from",
    [
        "run/ckpt/../ckpt/step_500",
        "run/../run/ckpt/step_500",
        "../run/ckpt/step_500",
        "run/ckpt/step_500/..",
        "run/ckpt/step_500/../step_500",
        "run/ckpt/..",
    ],
)
def test_a_parent_directory_component_is_refused_even_when_it_cancels_out(
    train_from: str,
) -> None:
    assert resume_message(train_from).startswith(
        f"{PREFIX}training.train_from: holds a '..' component"
    )


@pytest.mark.parametrize("char", ["*", "?", "[", "]"])
def test_a_glob_character_in_train_from_is_refused(char: str) -> None:
    message = resume_message(f"run/ckpt/step_5{char}")

    assert message.startswith(
        f"{PREFIX}training.train_from: holds {char!r}, which Eole reads as a glob"
    )


def test_a_dollar_sign_in_train_from_is_refused_as_in_every_string() -> None:
    message = resume_message("run/ckpt/step_$N")

    assert message.startswith(f"{PREFIX}training.train_from: holds '$'")


def test_train_from_cannot_be_checked_without_a_model_path_to_compare_with() -> None:
    message = resume_message("run/ckpt/step_500", DROP)

    assert message == (
        f"{PREFIX}training.train_from: cannot be checked, training.model_path "
        "must be a non-empty string to compare with, but it is missing"
    )


@pytest.mark.parametrize("model_path", [None, 5, ["run/ckpt"], ""])
def test_train_from_cannot_be_checked_against_a_model_path_that_is_not_text(
    model_path: object,
) -> None:
    message = resume_message("run/ckpt/step_500", model_path)

    assert message.startswith(
        f"{PREFIX}training.train_from: cannot be checked, training.model_path "
        "must be a non-empty string"
    )


def test_a_resume_config_is_held_to_every_other_rule() -> None:
    cfg = resume_config()
    cfg["model"]["embeddings"]["position_encoding_type"] = "Rotary"

    with pytest.raises(ConfigError) as raised:
        validate_train_config(cfg, resumes=True)

    assert "position_encoding_type" in str(raised.value)


def test_a_resume_config_still_refuses_an_unknown_key_and_a_forbidden_one() -> None:
    cfg = resume_config()
    cfg["training"]["bogus"] = 1
    with pytest.raises(ConfigError) as unknown:
        validate_train_config(cfg, resumes=True)
    assert str(unknown.value) == f"{PREFIX}training.bogus: unknown key in training"

    cfg = resume_config()
    cfg["model"]["sliding_window"] = 4
    with pytest.raises(ConfigError) as forbidden:
        validate_train_config(cfg, resumes=True)
    assert "model.sliding_window: is not allowed" in str(forbidden.value)


def test_a_resume_config_is_not_changed_by_validation() -> None:
    cfg = resume_config()

    validate_train_config(cfg, resumes=True)

    assert cfg == resume_config()


def test_the_resume_flag_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        validate_train_config(resume_config(), True)  # type: ignore[misc]


# ---- render ---------------------------------------------------------------------


def write_base(tmp_path: Path, base: object | None = None) -> Path:
    """A base config file; by default the baseline with an empty `data` section and
    without `src_vocab` and `model_path`, which a run fills in."""
    if base is None:
        cfg = make_config()
        cfg["data"] = {}
        del cfg["src_vocab"]
        del cfg["training"]["model_path"]
        base = cfg
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


def run_overrides() -> dict[str, Any]:
    full = make_config()
    return {
        "src_vocab": full["src_vocab"],
        "data": full["data"],
        "training": {"model_path": full["training"]["model_path"]},
    }


def test_merged_config_is_the_validated_merge_and_what_render_writes(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    overrides = {**run_overrides(), "training": {"model_path": "x/ckpt", "seed": None}}
    before = sorted(tmp_path.iterdir())

    cfg = merged_config(base, overrides)

    assert cfg == merge(yaml.safe_load(base.read_text(encoding="utf-8")), overrides)
    render(base, overrides, tmp_path / "config.yaml")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == cfg
    assert before == [base]  # merged_config itself wrote nothing


def test_merged_config_refuses_what_render_refuses_and_names_the_key(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)

    with pytest.raises(ConfigError) as raised:
        merged_config(base, {**run_overrides(), "training": {"train_stepz": 5}})

    assert PREFIX in str(raised.value) and "train_stepz" in str(raised.value)
    assert sorted(tmp_path.iterdir()) == [base]


@pytest.mark.parametrize("problem", ["missing", "not a mapping", "recursive overrides"])
def test_merged_config_refuses_a_bad_base_or_overrides_like_render(
    tmp_path: Path, problem: str
) -> None:
    base = write_base(tmp_path)
    overrides: dict[str, Any] = run_overrides()
    if problem == "missing":
        base = tmp_path / "missing.yaml"
    elif problem == "not a mapping":
        base.write_text("- a\n- b\n", encoding="utf-8")
    else:
        loop: dict[str, Any] = {}
        loop["self"] = loop
        overrides = {"training": loop}

    with pytest.raises(ConfigError) as from_merged:
        merged_config(base, overrides)
    with pytest.raises(ConfigError) as from_render:
        render(base, overrides, tmp_path / "config.yaml")

    assert str(from_merged.value) == str(from_render.value)


def test_render_merges_the_overrides_validates_and_writes_what_the_merge_is(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    out = tmp_path / "variants" / "t1" / "config.yaml"

    returned = render(base, run_overrides(), out)

    assert returned == out
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == make_config()


def test_render_round_trips_the_merge_of_the_base_and_the_overrides(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    overrides = {**run_overrides(), "training": {"model_path": "x/ckpt", "seed": None}}
    out = tmp_path / "config.yaml"

    render(base, overrides, out)

    expected = merge(yaml.safe_load(base.read_text(encoding="utf-8")), overrides)
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == expected


def test_render_applies_a_none_override_as_a_deletion(tmp_path: Path) -> None:
    base = write_base(tmp_path)
    overrides = {**run_overrides(), "training": {"model_path": "r", "gpu_ranks": None}}
    out = tmp_path / "config.yaml"

    render(base, overrides, out)

    written = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "gpu_ranks" not in written["training"]
    assert written["training"]["model_path"] == "r"


def test_render_creates_the_output_directory(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "config.yaml"

    render(write_base(tmp_path), run_overrides(), out)

    assert out.is_file()


def test_render_leaves_no_part_file_beside_a_written_config(tmp_path: Path) -> None:
    out = tmp_path / "out" / "config.yaml"

    render(write_base(tmp_path), run_overrides(), out)

    assert [path.name for path in out.parent.iterdir()] == ["config.yaml"]


def test_render_replaces_an_existing_output_whole(tmp_path: Path) -> None:
    out = tmp_path / "config.yaml"
    out.write_text("old: config\n", encoding="utf-8")

    render(write_base(tmp_path), run_overrides(), out)

    assert yaml.safe_load(out.read_text(encoding="utf-8")) == make_config()


def test_two_renders_of_the_same_input_are_byte_identical(tmp_path: Path) -> None:
    base = write_base(tmp_path)

    render(base, run_overrides(), tmp_path / "one.yaml")
    render(base, run_overrides(), tmp_path / "two.yaml")

    assert (tmp_path / "one.yaml").read_bytes() == (tmp_path / "two.yaml").read_bytes()


def test_render_writes_through_a_part_file_so_a_failed_write_keeps_the_old_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "config.yaml"
    out.write_text("old: config\n", encoding="utf-8")

    def fail_to_move(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail_to_move)
    with pytest.raises(OSError, match="disk full"):
        render(write_base(tmp_path), run_overrides(), out)

    assert out.read_text(encoding="utf-8") == "old: config\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "base.yaml",
        "config.yaml",
    ]


def test_render_keeps_the_order_of_the_base_then_adds_new_keys_in_override_order(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    overrides = {"zeta": None, "src_vocab": "v", "data": make_config()["data"]}
    out = tmp_path / "config.yaml"

    render(base, overrides, out)

    keys = list(yaml.safe_load(out.read_text(encoding="utf-8")))
    base_keys = list(yaml.safe_load(base.read_text(encoding="utf-8")))
    assert keys == [*base_keys, "src_vocab"]
    data_keys = list(yaml.safe_load(out.read_text(encoding="utf-8"))["data"])
    assert data_keys == [CORPUS, BT_CORPUS, "valid"]


def test_render_does_not_sort_keys_alphabetically(tmp_path: Path) -> None:
    out = tmp_path / "config.yaml"

    render(write_base(tmp_path), run_overrides(), out)

    assert out.read_text(encoding="utf-8").startswith("seed: 1234\nshare_vocab: true\n")


def test_render_changes_neither_the_base_file_nor_the_overrides(tmp_path: Path) -> None:
    base = write_base(tmp_path)
    before = base.read_bytes()
    overrides = run_overrides()
    snapshot = copy.deepcopy(overrides)

    render(base, overrides, tmp_path / "config.yaml")

    assert base.read_bytes() == before
    assert overrides == snapshot


def test_render_refuses_an_invalid_result_and_writes_nothing(tmp_path: Path) -> None:
    base = write_base(tmp_path)
    out = tmp_path / "new_dir" / "config.yaml"
    overrides = {**run_overrides(), "overwrite": False}

    with pytest.raises(
        ConfigError, match=r"^invalid Eole config: overwrite: must be true"
    ):
        render(base, overrides, out)

    assert not out.parent.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["base.yaml"]


def test_render_keeps_an_existing_output_when_the_result_is_invalid(
    tmp_path: Path,
) -> None:
    out = tmp_path / "config.yaml"
    out.write_text("old: config\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        render(write_base(tmp_path), {"bogus": 1}, out)

    assert out.read_text(encoding="utf-8") == "old: config\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "base.yaml",
        "config.yaml",
    ]


def test_render_validates_the_base_itself_when_the_overrides_change_nothing(
    tmp_path: Path,
) -> None:
    cfg = make_config()
    cfg["model"]["embeddings"]["word_vec_size"] = 128
    out = tmp_path / "config.yaml"

    with pytest.raises(ConfigError, match="model.embeddings.word_vec_size"):
        render(write_base(tmp_path, cfg), {}, out)

    assert not out.exists()


def test_render_refuses_a_dollar_sign_an_override_brings(tmp_path: Path) -> None:
    out = tmp_path / "config.yaml"
    overrides = {**run_overrides(), "src_vocab": "$HOME/vocab.src"}

    with pytest.raises(ConfigError, match=r"src_vocab: holds '\$'"):
        render(write_base(tmp_path), overrides, out)

    assert not out.exists()


def test_render_refuses_an_override_that_yaml_cannot_dump_and_writes_nothing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "new_dir" / "config.yaml"
    overrides = {**run_overrides(), "training": {"model_path": Path("run/ckpt")}}

    with pytest.raises(ConfigError, match="must be plain YAML data"):
        render(write_base(tmp_path), overrides, out)

    assert not out.parent.exists()


@pytest.mark.parametrize(
    "value", [_Color.RED, _Count.ONE, _Rate(0.5), _Names(["a"])], ids=repr
)
def test_render_refuses_an_override_that_is_a_subclass_of_a_plain_type_as_a_config_error(
    tmp_path: Path, value: object
) -> None:
    out = tmp_path / "new_dir" / "config.yaml"
    overrides = {**run_overrides(), "training": {"model_path": value}}

    with pytest.raises(ConfigError, match="must be plain YAML data"):
        render(write_base(tmp_path), overrides, out)

    assert not out.parent.exists()


def test_render_refuses_a_recursive_anchor_in_the_base_before_it_merges(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("training: &x\n  self: *x\n", encoding="utf-8")
    out = tmp_path / "new_dir" / "config.yaml"

    with pytest.raises(ConfigError) as raised:
        render(base, run_overrides(), out)

    assert str(raised.value) == (
        f"invalid base Eole config {base}: training.self: refers back to a mapping "
        "or list that contains it"
    )
    assert not out.parent.exists()


def test_render_refuses_overrides_that_contain_themselves_before_it_merges(
    tmp_path: Path,
) -> None:
    # `AblationMatrix.overrides` comes from YAML, where an anchor can do this.
    section: dict[str, Any] = {}
    section["self"] = section
    out = tmp_path / "new_dir" / "config.yaml"

    with pytest.raises(ConfigError) as raised:
        render(write_base(tmp_path), {"training": section}, out)

    assert str(raised.value) == (
        "invalid Eole config overrides: training.self: refers back to a mapping or "
        "list that contains it"
    )
    assert not out.parent.exists()


def test_render_refuses_an_output_that_is_a_directory_and_writes_nothing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "config.yaml"
    out.mkdir()

    with pytest.raises(ConfigError, match=rf"cannot write {out}: it is a directory"):
        render(write_base(tmp_path), run_overrides(), out)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "base.yaml",
        "config.yaml",
    ]
    assert list(out.iterdir()) == []


def test_render_refuses_an_output_whose_directory_is_a_regular_file(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "variants"
    parent.write_text("not a directory\n", encoding="utf-8")
    out = parent / "config.yaml"

    with pytest.raises(
        ConfigError, match=rf"cannot write {out}: {parent} is not a directory"
    ):
        render(write_base(tmp_path), run_overrides(), out)

    assert parent.read_text(encoding="utf-8") == "not a directory\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "base.yaml",
        "variants",
    ]


def test_render_refuses_an_output_below_a_regular_file_further_up(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "variants"
    ancestor.write_text("not a directory\n", encoding="utf-8")
    out = ancestor / "v1" / "run" / "config.yaml"

    with pytest.raises(
        ConfigError, match=rf"cannot write {out}: {ancestor} is not a directory"
    ):
        render(write_base(tmp_path), run_overrides(), out)

    assert ancestor.read_text(encoding="utf-8") == "not a directory\n"


@pytest.mark.parametrize("spelling", ["same", "dotted", "link"])
def test_render_refuses_an_output_that_is_the_base_and_leaves_the_base_alone(
    tmp_path: Path, spelling: str
) -> None:
    base = write_base(tmp_path)
    before = base.read_bytes()
    out = {
        "same": base,
        "dotted": tmp_path / "sub" / ".." / "base.yaml",
        "link": tmp_path / "link.yaml",
    }[spelling]
    if spelling == "dotted":
        (tmp_path / "sub").mkdir()
    if spelling == "link":
        out.symlink_to(base)

    with pytest.raises(
        ConfigError, match=rf"cannot write {re.escape(str(out))}: it is the base"
    ):
        render(base, run_overrides(), out)

    assert base.read_bytes() == before
    assert not list(tmp_path.glob("*.part"))


def test_render_checks_the_output_before_it_creates_anything(tmp_path: Path) -> None:
    ancestor = tmp_path / "variants"
    ancestor.write_text("x", encoding="utf-8")

    with pytest.raises(ConfigError, match="is not a directory"):
        render(write_base(tmp_path), {}, ancestor / "a" / "config.yaml")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "base.yaml",
        "variants",
    ]


def test_render_refuses_a_missing_base(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base Eole config file not found"):
        render(tmp_path / "missing.yaml", {}, tmp_path / "config.yaml")

    assert [path.name for path in tmp_path.iterdir()] == []


def test_render_refuses_a_base_that_is_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base Eole config file not found"):
        render(tmp_path, {}, tmp_path / "config.yaml")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_render_refuses_a_base_it_cannot_read(tmp_path: Path) -> None:
    base = write_base(tmp_path)
    base.chmod(0)
    try:
        with pytest.raises(ConfigError, match="cannot read or parse"):
            render(base, {}, tmp_path / "config.yaml")
    finally:
        base.chmod(0o600)

    assert not (tmp_path / "config.yaml").exists()


def test_render_refuses_a_base_that_is_not_utf8(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_bytes(b"seed: \xff\xfe\n")

    with pytest.raises(ConfigError, match="cannot read or parse"):
        render(base, {}, tmp_path / "config.yaml")

    assert not (tmp_path / "config.yaml").exists()


def test_render_refuses_a_base_that_is_not_yaml(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("key: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="cannot read or parse"):
        render(base, {}, tmp_path / "config.yaml")


def test_render_refuses_an_empty_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("", encoding="utf-8")
    out = tmp_path / "config.yaml"

    with pytest.raises(ConfigError, match="base Eole config .*base.yaml: is empty"):
        render(base, run_overrides(), out)

    assert not out.exists()


@pytest.mark.parametrize(
    ("text", "type_name"),
    [("- a\n- b\n", "list"), ("just text\n", "str"), ("3\n", "int")],
)
def test_render_refuses_a_base_that_is_not_a_mapping_by_type(
    tmp_path: Path, text: str, type_name: str
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(text, encoding="utf-8")
    out = tmp_path / "config.yaml"

    with pytest.raises(
        ConfigError, match=rf"base Eole config .*: must be a mapping, got {type_name}$"
    ):
        render(base, run_overrides(), out)

    assert not out.exists()


def test_render_refuses_a_dollar_sign_in_the_base_text(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("src_vocab: $HOME/vocab.src\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"holds '\$'"):
        render(base, {}, tmp_path / "config.yaml")


def test_render_accepts_a_base_with_an_empty_data_section_once_the_overrides_fill_it(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    assert yaml.safe_load(base.read_text(encoding="utf-8"))["data"] == {}

    render(base, run_overrides(), tmp_path / "config.yaml")


# ---- render and merged_config for a resume config ---------------------------------


def resume_overrides(train_from: object = f"{RUN_CKPT}/step_500") -> dict[str, Any]:
    return {
        **run_overrides(),
        "training": {"model_path": RUN_CKPT, "train_from": train_from},
    }


def test_render_refuses_train_from_unless_it_is_told_the_config_resumes(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    out = tmp_path / "launch" / "resume-500.yaml"

    with pytest.raises(ConfigError) as raised:
        render(base, resume_overrides(), out)

    assert str(raised.value) == f"{PREFIX}training.train_from: unknown key in training"
    assert not out.parent.exists()


def test_render_writes_the_resume_config_when_told_the_config_resumes(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    out = tmp_path / "launch" / "resume-500.yaml"
    before = base.read_bytes()

    assert render(base, resume_overrides(), out, resumes=True) == out

    written = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert written["training"]["train_from"] == f"{RUN_CKPT}/step_500"
    assert written["training"]["model_path"] == RUN_CKPT
    assert base.read_bytes() == before


def test_render_with_the_resume_flag_refuses_a_foreign_train_from_and_writes_nothing(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)
    out = tmp_path / "launch" / "resume-500.yaml"

    with pytest.raises(ConfigError) as raised:
        render(base, resume_overrides("elsewhere/ckpt/step_500"), out, resumes=True)

    assert str(raised.value).startswith(f"{PREFIX}training.train_from: must be '")
    assert not out.parent.exists()


def test_the_resume_flag_of_render_is_keyword_only(tmp_path: Path) -> None:
    base = write_base(tmp_path)

    with pytest.raises(TypeError):
        render(base, run_overrides(), tmp_path / "o.yaml", True)  # type: ignore[misc]


def test_merged_config_passes_the_resume_flag_on_and_defaults_to_no_resume(
    tmp_path: Path,
) -> None:
    base = write_base(tmp_path)

    with pytest.raises(ConfigError, match="train_from: unknown key in training"):
        merged_config(base, resume_overrides())

    cfg = merged_config(base, resume_overrides(), resumes=True)
    assert cfg["training"]["train_from"] == f"{RUN_CKPT}/step_500"


# ---- merge: the cases -----------------------------------------------------------


def test_merge_merges_mappings_recursively_and_keeps_what_the_overrides_leave() -> None:
    base = {"a": {"b": 1, "c": {"d": 2, "e": 3}}, "f": 4}

    merged = merge(base, {"a": {"c": {"d": 20}}, "g": 5})

    assert merged == {"a": {"b": 1, "c": {"d": 20, "e": 3}}, "f": 4, "g": 5}


def test_merge_replaces_lists_and_scalars_whole() -> None:
    base = {"a": [1, 2, 3], "b": 1, "c": "text", "d": {"x": 1}}

    merged = merge(base, {"a": [9], "b": 2, "c": 3, "d": [1]})

    assert merged == {"a": [9], "b": 2, "c": 3, "d": [1]}


def test_merge_replaces_a_scalar_by_a_mapping_and_a_mapping_by_a_scalar() -> None:
    assert merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}


def test_merge_deletes_a_key_whose_override_is_none() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}

    assert merge(base, {"a": None, "b": {"c": None}}) == {"b": {"d": 3}}


def test_merge_deletes_nothing_for_a_none_override_of_an_absent_key() -> None:
    assert merge({"a": 1}, {"z": None, "y": {"x": None}}) == {"a": 1, "y": {}}


def test_merge_drops_none_values_inside_a_mapping_that_replaces_a_scalar() -> None:
    assert merge({"a": 1}, {"a": {"b": None, "c": 2}}) == {"a": {"c": 2}}


def test_merge_keeps_a_none_the_base_already_holds() -> None:
    assert merge({"a": None, "b": 1}, {"b": 2}) == {"a": None, "b": 2}


def test_merge_with_no_overrides_copies_the_base() -> None:
    base = {"a": {"b": [1, {"c": 2}]}}

    merged = merge(base, {})

    assert merged == base
    assert merged is not base
    assert merged["a"] is not base["a"]
    assert merged["a"]["b"] is not base["a"]["b"]


def test_merge_modifies_neither_input_and_shares_no_list_with_them() -> None:
    base = {"a": {"b": [1, 2]}, "c": [3]}
    overrides = {"a": {"d": [4, 5]}, "e": {"f": [6]}}
    base_before = copy.deepcopy(base)
    overrides_before = copy.deepcopy(overrides)

    merged = merge(base, overrides)
    merged["a"]["b"].append(99)
    merged["a"]["d"].append(99)
    merged["c"].append(99)
    merged["e"]["f"].append(99)

    assert base == base_before
    assert overrides == overrides_before


def test_merge_keeps_the_key_order_of_the_base_and_appends_new_keys_in_order() -> None:
    merged = merge({"b": 1, "a": {"y": 1, "x": 2}}, {"d": 4, "a": {"z": 3}, "c": 5})

    assert list(merged) == ["b", "a", "d", "c"]
    assert list(merged["a"]) == ["y", "x", "z"]


def test_merge_returns_a_plain_dict() -> None:
    from collections import OrderedDict

    merged = merge(OrderedDict(a=OrderedDict(b=1)), {"a": {"c": 2}})

    assert type(merged) is dict
    assert type(merged["a"]) is dict
