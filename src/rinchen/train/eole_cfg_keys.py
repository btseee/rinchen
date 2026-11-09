"""The keys a Rinchen Eole training config may hold, level by level (the allow-lists
`eole_cfg.validate_train_config` checks), and the three values of the continuation
(D9). Data only; `eole_cfg` re-exports every name and applies them.
"""

from __future__ import annotations

from typing import Final

# The keys a D7 or D8 run sets or an ablation plausibly overrides, level by level. Eole
# 0.6.4 rejects any other key of `eole train` (`extra_forbidden`), `eole build_vocab`
# ignores some, and a typo must not cost a build_vocab run: so a key that no config
# needs is not here. `test_train_eole_cfg_drift.py` checks every key against the
# installed Eole.

# Eole's `TrainConfig`.
TOP_LEVEL_KEYS: Final = frozenset(
    {
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
        # The teacher runs (D8). `valid_metrics` is allowed and no teacher config sets
        # it: the validation corpus is pre-tokenised, so a metric would score pieces.
        "tensorboard",
        "tensorboard_log_dir",
        "log_file",
        "valid_metrics",
    }
)

# Eole's `Dataset`: one corpus of `data`; the prefixes carry the direction tag.
CORPUS_KEYS: Final = frozenset(
    {"path_src", "path_tgt", "transforms", "weight", "src_prefix", "tgt_prefix"}
)

# Only the transforms the D7 corpora run: Eole's own tokenisation and length filter.
TRANSFORMS_CONFIGS_KEYS: Final = frozenset({"sentencepiece", "filtertoolong"})
SENTENCEPIECE_KEYS: Final = frozenset({"src_subword_model", "tgt_subword_model"})
FILTERTOOLONG_KEYS: Final = frozenset({"src_seq_length", "tgt_seq_length"})

# `TransformerModelConfig`, the model class `architecture: transformer` selects.
MODEL_KEYS: Final = frozenset(
    {
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
)

# The two stacks differ in Eole's classes, so they have a list each.
ENCODER_KEYS: Final = frozenset(
    {"layers", "hidden_size", "heads", "transformer_ff", "position_encoding_type"}
)
DECODER_KEYS: Final = frozenset(
    {"layers", "hidden_size", "heads", "transformer_ff", "position_encoding_type"}
)

# Only the size and the position scheme: the source and target sizes stay Eole's.
EMBEDDINGS_KEYS: Final = frozenset({"word_vec_size", "position_encoding_type"})

# Eole's `TrainingConfig`: the tiny CPU base, a GPU proxy base and the teachers, nothing
# that adapts, continues or quantises a model. Not here, on purpose: `average_decay` and
# `average_every` (the moving average replaces the live weights in every saved
# checkpoint and is not restored), `accum_steps` and `dropout_steps` (a scheduled value
# restarts at its first entry after a resume), `reset_optim` and `update_vocab` (they
# change how a checkpoint is loaded; `reset_optim` has its own list below, and
# `update_vocab`, which remaps a checkpoint's embeddings to a new vocabulary, none).
TRAINING_KEYS: Final = frozenset(
    {
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
        # The teacher runs (D8).
        "early_stopping",
        "early_stopping_criteria",
        "timeout",
        "master_ip",
        "master_port",
        "gpu_backend",
        "normalization",
        "batch_size_multiple",
        "self_attn_backend",
    }
)

# The one `training` key only the resume config of a run may hold: it continues the
# run's own checkpoint (`validate_train_config(..., resumes=True)`). Everywhere else it
# is an unknown key, because every Rinchen model trains from random init.
RESUME_TRAINING_KEYS: Final = frozenset({"train_from"})

# The two `training` keys only the FIRST attempt of a continuation may hold (D9,
# `validate_train_config(..., continues=...)`): the launcher starts a new run from one of
# our own checkpoints with `train_from` set to that checkpoint and `reset_optim` set to
# `CONTINUATION_RESET_OPTIM`, which loads the weights only. Everywhere else they are
# unknown keys. `update_vocab` is never a key, and the other `reset_optim` values are
# never accepted: `none` and `states` keep the optimizer options of the checkpoint and
# ignore the new config's, `keep_states` crashes Eole 0.6.4.
CONTINUATION_TRAINING_KEYS: Final = frozenset({"train_from", "reset_optim"})
CONTINUATION_RESET_OPTIM: Final = "all"
# A resume of a run that began as a continuation must say `reset_optim: none` (Eole's own
# default, here stated): the checkpoints of such a run record `reset_optim: all`, and Eole
# fills a key the resume config omits from the checkpoint's record, so a resume that left
# it out would load only the weights, restart the step count at 1 and repeat the warm-up
# (measured with Eole 0.6.4: a "resume" of step 40 that ran 200 steps from step 1).
CONTINUATION_RESUME_RESET_OPTIM: Final = "none"
