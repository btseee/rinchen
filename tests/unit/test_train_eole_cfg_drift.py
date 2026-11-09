"""The allow-lists of `rinchen.train.eole_cfg` against the Eole that is installed.

Every key a list allows must still be a field of the Eole model at its level, or the
validator would pass a key `eole train` then rejects (Eole refuses unknown keys at
every level). The models are found from Eole's own `TrainConfig`, so a renamed class
does not hide a drifted key. When a test here fails, the message names the key, the
level and the Eole class: drop or rename the key in the list, then check whether the
D7 configs still say what they meant.

These tests import Eole (a few seconds) but start no process and read no file.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping

import pydantic
import pytest

from rinchen.train import eole_cfg
from tests.support.eole_train_config import make_config


def _classes(annotation: object) -> list[type[pydantic.BaseModel]]:
    """The model classes an Eole field annotation names, through `Annotated`,
    unions, `Optional` and `dict` values."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _classes(typing.get_args(annotation)[0])
    if origin is typing.Union or origin is types.UnionType:
        return [c for arg in typing.get_args(annotation) for c in _classes(arg)]
    if origin is dict:
        return _classes(typing.get_args(annotation)[1])
    if annotation is type(None):
        return []
    assert isinstance(annotation, type), annotation
    return [annotation]


def _the_class_tagged(
    candidates: list[type[pydantic.BaseModel]], tag_field: str, tag: str
) -> type[pydantic.BaseModel]:
    """The candidate whose discriminator field `tag_field` is the literal `tag`."""
    matching = [
        cls
        for cls in candidates
        if typing.get_args(cls.model_fields[tag_field].annotation) == (tag,)
    ]
    assert len(matching) == 1, [cls.__name__ for cls in candidates]
    return matching[0]


def _levels() -> list[tuple[str, frozenset[str], type[pydantic.BaseModel]]]:
    """(level, the keys the list allows, the Eole class that must have them)."""
    from eole.config.data import NestedAllTransformsConfig
    from eole.config.run import TrainConfig

    fields = TrainConfig.model_fields
    (dataset,) = _classes(fields["data"].annotation)
    (training,) = _classes(fields["training"].annotation)
    model = _the_class_tagged(
        _classes(fields["model"].annotation), "architecture", "transformer"
    )
    encoder = _the_class_tagged(
        _classes(model.model_fields["encoder"].annotation),
        "encoder_type",
        "transformer",
    )
    decoder = _the_class_tagged(
        _classes(model.model_fields["decoder"].annotation),
        "decoder_type",
        "transformer",
    )
    (embeddings,) = _classes(model.model_fields["embeddings"].annotation)
    transforms = NestedAllTransformsConfig.model_fields
    (sentencepiece,) = _classes(transforms["sentencepiece"].annotation)
    (filtertoolong,) = _classes(transforms["filtertoolong"].annotation)
    return [
        ("the top level", eole_cfg.TOP_LEVEL_KEYS, TrainConfig),
        ("a data corpus", eole_cfg.CORPUS_KEYS, dataset),
        (
            "transforms_configs",
            eole_cfg.TRANSFORMS_CONFIGS_KEYS,
            NestedAllTransformsConfig,
        ),
        (
            "transforms_configs.sentencepiece",
            eole_cfg.SENTENCEPIECE_KEYS,
            sentencepiece,
        ),
        (
            "transforms_configs.filtertoolong",
            eole_cfg.FILTERTOOLONG_KEYS,
            filtertoolong,
        ),
        ("model", eole_cfg.MODEL_KEYS, model),
        ("model.encoder", eole_cfg.ENCODER_KEYS, encoder),
        ("model.decoder", eole_cfg.DECODER_KEYS, decoder),
        ("model.embeddings", eole_cfg.EMBEDDINGS_KEYS, embeddings),
        ("training", eole_cfg.TRAINING_KEYS, training),
    ]


def keys_eole_dropped(
    level: str, allowed: frozenset[str], eole_class: type[pydantic.BaseModel]
) -> list[str]:
    """One sentence per allowed key that `eole_class` has no field for."""
    return [
        f"the key {key!r} at level {level!r} is no longer a field of "
        f"{eole_class.__name__}: Eole dropped or renamed it"
        for key in sorted(allowed - set(eole_class.model_fields))
    ]


def test_every_allowed_key_is_still_a_field_of_the_eole_model_at_its_level() -> None:
    drifted = [
        sentence
        for level, allowed, eole_class in _levels()
        for sentence in keys_eole_dropped(level, allowed, eole_class)
    ]

    assert drifted == []


def test_the_drift_check_names_the_key_the_level_and_the_class_of_a_dropped_key() -> (
    None
):
    class Renamed(pydantic.BaseModel):
        batch_size: int = 1
        keep_checkpoint: int = 1

    sentences = keys_eole_dropped(
        "training", frozenset({"batch_size", "train_steps"}), Renamed
    )

    assert sentences == [
        (
            "the key 'train_steps' at level 'training' is no longer a field of "
            "Renamed: Eole dropped or renamed it"
        )
    ]


def test_the_drift_check_finds_nothing_to_report_when_every_key_is_present() -> None:
    class Same(pydantic.BaseModel):
        a: int = 1
        b: int = 1

    assert keys_eole_dropped("model", frozenset({"a", "b"}), Same) == []


def test_every_level_the_validator_knows_has_a_checked_list() -> None:
    levels = [level for level, _, _ in _levels()]

    assert levels == [
        "the top level",
        "a data corpus",
        "transforms_configs",
        "transforms_configs.sentencepiece",
        "transforms_configs.filtertoolong",
        "model",
        "model.encoder",
        "model.decoder",
        "model.embeddings",
        "training",
    ]


def test_every_forbidden_key_is_still_a_field_of_some_eole_model() -> None:
    from eole.config import models

    fields = {
        field
        for _, cls in inspect.getmembers(models, inspect.isclass)
        if issubclass(cls, pydantic.BaseModel)
        for field in cls.model_fields
    }

    gone = [
        f"the forbidden key {key!r} is no longer a field of any Eole model config"
        for key in sorted(eole_cfg.FORBIDDEN_KEYS - fields)
    ]
    assert gone == []


def test_the_allow_lists_hold_no_forbidden_key() -> None:
    allowed_everywhere: Mapping[str, frozenset[str]] = {
        level: allowed for level, allowed, _ in _levels()
    }

    clashes = [
        f"{level}: {sorted(allowed & eole_cfg.FORBIDDEN_KEYS)}"
        for level, allowed in allowed_everywhere.items()
        if allowed & eole_cfg.FORBIDDEN_KEYS
    ]
    assert clashes == []


def test_the_allow_lists_are_frozensets_of_names() -> None:
    for _, allowed, _ in _levels():
        assert isinstance(allowed, frozenset)
        assert allowed
        assert all(isinstance(key, str) and key for key in allowed)


@pytest.mark.parametrize(
    "key",
    [
        "train_from",
        "update_vocab",
        "reset_optim",
        "lora_rank",
        "lora_layers",
        "rl_algorithm",
        "quant_type",
        "pre_word_vecs_enc",
        "freeze_encoder",
    ],
)
def test_no_list_allows_a_key_that_starts_from_or_adapts_an_existing_model(
    key: str,
) -> None:
    # Every Rinchen model trains from random init; adapters, reinforcement learning
    # and quantisation have no user. `train_from` is allowed only in the resume
    # config of a run that continues its own checkpoint, through its own list.
    assert all(key not in allowed for _, allowed, _ in _levels())


@pytest.mark.parametrize(
    "key", ["average_decay", "average_every", "accum_steps", "dropout_steps"]
)
def test_no_list_allows_a_key_whose_effect_a_resume_would_lose_or_replace(
    key: str,
) -> None:
    # The EMA replaces the live weights in every saved checkpoint and is not restored;
    # a scheduled value restarts at its first entry after a resume.
    assert all(key not in allowed for _, allowed, _ in _levels())
    assert key not in eole_cfg.RESUME_TRAINING_KEYS


def test_the_one_key_only_a_resume_config_may_hold_is_a_field_of_the_eole_training_model() -> (
    None
):
    training = {level: cls for level, _, cls in _levels()}["training"]

    assert eole_cfg.RESUME_TRAINING_KEYS == {"train_from"}
    assert (
        keys_eole_dropped(
            "training (resume only)", eole_cfg.RESUME_TRAINING_KEYS, training
        )
        == []
    )
    assert not eole_cfg.RESUME_TRAINING_KEYS & eole_cfg.TRAINING_KEYS


def test_the_training_list_names_the_keys_the_tiny_proxy_and_teacher_runs_set() -> None:
    assert {
        "model_path",
        "train_steps",
        "valid_steps",
        "save_checkpoint_steps",
        "keep_checkpoint",
        "batch_type",
        "batch_size",
        "valid_batch_size",
        "accum_count",
        "world_size",
        "gpu_ranks",
        "compute_dtype",
        "learning_rate",
        "warmup_steps",
        "decay_method",
        "optim",
        "adam_beta2",
        "weight_decay",
        "max_grad_norm",
        "param_init",
        "param_init_method",
        "dropout",
        "attention_dropout",
        "label_smoothing",
        "bucket_size",
        "prefetch_factor",
        "num_workers",
        # the teacher runs (D8)
        "early_stopping",
        "early_stopping_criteria",
        "timeout",
        "master_ip",
        "master_port",
        "gpu_backend",
        "normalization",
        "batch_size_multiple",
        "self_attn_backend",
    } == eole_cfg.TRAINING_KEYS


def test_the_other_lists_hold_exactly_the_keys_a_d7_or_d8_config_sets() -> None:
    assert eole_cfg.TOP_LEVEL_KEYS == {
        "seed",
        "save_data",
        "src_vocab",
        "tgt_vocab",
        "share_vocab",
        "src_vocab_size",
        "tgt_vocab_size",
        "overwrite",
        "n_sample",
        "report_every",
        "data",
        "transforms",
        "transforms_configs",
        "model",
        "training",
        # the teacher runs (D8); `valid_metrics` is allowed, and no teacher config
        # sets it: the validation corpus is pre-tokenised
        "tensorboard",
        "tensorboard_log_dir",
        "log_file",
        "valid_metrics",
    }
    assert eole_cfg.CORPUS_KEYS == {
        "path_src",
        "path_tgt",
        "transforms",
        "weight",
        "src_prefix",
        "tgt_prefix",
    }
    assert eole_cfg.TRANSFORMS_CONFIGS_KEYS == {"sentencepiece", "filtertoolong"}
    assert eole_cfg.SENTENCEPIECE_KEYS == {"src_subword_model", "tgt_subword_model"}
    assert eole_cfg.FILTERTOOLONG_KEYS == {"src_seq_length", "tgt_seq_length"}
    assert eole_cfg.MODEL_KEYS == {
        "architecture",
        "hidden_size",
        "heads",
        "transformer_ff",
        "encoder",
        "decoder",
        "embeddings",
        "share_embeddings",
        "share_decoder_embeddings",
    }
    layer_keys = {
        "layers",
        "hidden_size",
        "heads",
        "transformer_ff",
        "position_encoding_type",
    }
    assert eole_cfg.ENCODER_KEYS == layer_keys
    assert eole_cfg.DECODER_KEYS == layer_keys
    assert eole_cfg.EMBEDDINGS_KEYS == {"word_vec_size", "position_encoding_type"}


def test_eoles_own_schema_accepts_the_config_the_validator_is_tested_against() -> None:
    # Constructing the config checks every key and value against Eole's models and
    # reads no file, so the baseline's placeholder paths need not exist.
    from eole.config.run import TrainConfig

    config = TrainConfig(**make_config())

    assert config.model.architecture == "transformer"
    assert config.model.embeddings.position_encoding_type == (
        eole_cfg.POSITION_ENCODING
    )
    assert config.share_vocab is True
    assert config.overwrite is True
    assert set(config.data) == {"en-mn.auth", "en-mn.bt", "valid"}


def test_the_validator_accepts_exactly_the_normalisations_eole_knows() -> None:
    import typing

    from eole.config.training import TrainingConfig

    eole_values = set(
        typing.get_args(TrainingConfig.model_fields["normalization"].annotation)
    )

    assert eole_values == set(eole_cfg.NORMALIZATIONS)


def test_eoles_own_schema_accepts_a_config_that_sets_every_added_key() -> None:
    from eole.config.run import TrainConfig

    cfg = make_config()
    cfg.update(
        tensorboard=True,
        tensorboard_log_dir="run/tensorboard",
        log_file="run/train.log",
        valid_metrics=["BLEU", "CHRF"],
    )
    cfg["training"].update(
        early_stopping=6,
        early_stopping_criteria="ppl",
        timeout=7200,
        master_ip="localhost",
        master_port=10000,
        gpu_backend="nccl",
        normalization="tokens",
        batch_size_multiple=8,
        self_attn_backend="pytorch",
    )

    config = TrainConfig(**cfg)

    assert config.training.early_stopping == 6
    assert config.training.timeout == 7200
    assert config.training.normalization == "tokens"
    assert config.tensorboard is True
    assert config.valid_metrics == ["BLEU", "CHRF"]


def test_eoles_own_schema_accepts_the_resume_key_of_a_resume_config() -> None:
    from eole.config.run import TrainConfig

    cfg = make_config()
    cfg["training"]["train_from"] = "run/ckpt/step_500"

    assert TrainConfig(**cfg).training.train_from == "run/ckpt/step_500"


def test_the_two_keys_only_a_continuation_config_may_hold_are_eole_training_fields() -> (
    None
):
    training = {level: cls for level, _, cls in _levels()}["training"]

    assert eole_cfg.CONTINUATION_TRAINING_KEYS == {"train_from", "reset_optim"}
    assert (
        keys_eole_dropped(
            "training (continuation only)",
            eole_cfg.CONTINUATION_TRAINING_KEYS,
            training,
        )
        == []
    )
    assert not eole_cfg.CONTINUATION_TRAINING_KEYS & eole_cfg.TRAINING_KEYS
    assert eole_cfg.RESUME_TRAINING_KEYS < eole_cfg.CONTINUATION_TRAINING_KEYS


def test_the_continuation_keys_leave_out_the_ones_that_adapt_or_crash() -> None:
    # `update_vocab` remaps the embeddings of a checkpoint to a new vocabulary, and the
    # `keep_states` value of `reset_optim` crashes Eole 0.6.4 (spike, D9).
    assert "update_vocab" not in eole_cfg.CONTINUATION_TRAINING_KEYS
    assert "keep_states" not in eole_cfg.CONTINUATION_TRAINING_KEYS
    assert all(
        "update_vocab" not in allowed and "reset_optim" not in allowed
        for _, allowed, _ in _levels()
    )


def test_eole_still_has_the_reset_optim_value_the_launcher_writes() -> None:
    import typing

    from eole.config.training import TrainingConfig

    values = set(typing.get_args(TrainingConfig.model_fields["reset_optim"].annotation))

    assert eole_cfg.CONTINUATION_RESET_OPTIM in values
    assert eole_cfg.CONTINUATION_RESET_OPTIM == "all"


def test_eoles_own_schema_accepts_the_two_keys_of_a_continuation_config() -> None:
    from eole.config.run import TrainConfig

    cfg = make_config()
    cfg["training"]["train_from"] = "/runs/teacher/ckpt/step_200"
    cfg["training"]["reset_optim"] = "all"

    config = TrainConfig(**cfg)

    assert config.training.train_from == "/runs/teacher/ckpt/step_200"
    assert config.training.reset_optim == "all"


def test_eole_still_has_the_reset_optim_value_a_resume_of_a_continuation_states() -> (
    None
):
    import typing

    from eole.config.training import TrainingConfig

    values = set(typing.get_args(TrainingConfig.model_fields["reset_optim"].annotation))

    assert eole_cfg.CONTINUATION_RESUME_RESET_OPTIM in values
    assert eole_cfg.CONTINUATION_RESUME_RESET_OPTIM == "none"
    # What Eole does when the key is left out is the default: the same value.
    default = TrainingConfig.model_fields["reset_optim"].default
    assert default == eole_cfg.CONTINUATION_RESUME_RESET_OPTIM
