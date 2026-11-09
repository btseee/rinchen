"""Eole configs: what every Rinchen model shares, and the multilingual training
config's schema.

Shared: the keys no config may carry, the guard against `$`, and reading and writing
the YAML. The schema (`validate_train_config`, with the allow-lists it checks and
`merge` and `render` that build a config from a base file and overrides) is that of
the multilingual models trained from the D7 configs. What another model's config must
hold (the converter's tags and corpora, say) belongs to that model's own module, which
keeps its own validator.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import yaml

from rinchen.errors import ConfigError
from rinchen.utils.integers import is_strict_int
from rinchen.utils.part_file import unique_part_path

# Eole's own default, spelled out in every config so none depends on it.
POSITION_ENCODING: Final = "SinusoidalInterleaved"

# Field names of the installed Eole (0.6.4, `eole/config/models.py`) that select or
# tune a position scheme other than sinusoidal, or restrict or group attention.
# Eole has no field named for ALiBi (it is a `position_encoding_type` value, rejected
# by the validators). A key that is present is rejected even with a neutral value, so
# a config never holds a switch that could be flipped.
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

# What Eole's loss is divided by (`TrainingConfig.normalization`).
NORMALIZATIONS: Final = ("sents", "tokens")

# Eole 0.6.4 takes one validation corpus and reads it under this name; every other
# name is a training corpus.
_VALID: Final = "valid"
_GLOB_CHARACTERS: Final = frozenset("[]*?")
_ARCHITECTURE: Final = "transformer"
_TRAIN_FROM: Final = "train_from"
_RESET_OPTIM: Final = "reset_optim"
_RESUME_WHY: Final = (
    "the checkpoints of a continuation record reset_optim: all, which Eole would "
    "apply again to a resume that leaves it out: the step count would restart at 1"
)
_CONTINUE_WHY: Final = (
    "a continuation loads the weights of the source and nothing else, or Eole keeps "
    "the optimizer options of the source and ignores the new ones"
)
# `step_<N>` as Eole names a checkpoint directory; [0-9], not \d, which matches the
# digits of other scripts.
_STEP_DIRECTORY: Final = re.compile(r"step_[1-9][0-9]*")
_MAX_PORT: Final = 65535
# Exactly these types: YAML's safe dumper looks a value's type up by identity, so a
# subclass (an enum member, a numpy float) is not dumped although `isinstance` says it
# is plain.
_PLAIN_TYPES: Final = (str, int, float, bool, type(None), dict, list, tuple)
_MISSING: Final = object()

# Where each section's own keys are checked, parent before child; a corpus's keys are
# checked by `_check_sections` through `data`.
_SECTION_LEVELS: Final = (
    (("transforms_configs",), TRANSFORMS_CONFIGS_KEYS),
    (("transforms_configs", "sentencepiece"), SENTENCEPIECE_KEYS),
    (("transforms_configs", "filtertoolong"), FILTERTOOLONG_KEYS),
    (("model",), MODEL_KEYS),
    (("model", "encoder"), ENCODER_KEYS),
    (("model", "decoder"), DECODER_KEYS),
    (("model", "embeddings"), EMBEDDINGS_KEYS),
    (("training",), TRAINING_KEYS),
)
# The keys a section also allows in a resume config, and in the first attempt of a
# continuation.
_RESUME_KEYS: Final[dict[tuple[str, ...], frozenset[str]]] = {
    ("training",): RESUME_TRAINING_KEYS
}
_CONTINUATION_KEYS: Final[dict[tuple[str, ...], frozenset[str]]] = {
    ("training",): CONTINUATION_TRAINING_KEYS
}
_CONTINUATION_RESUME_KEYS: Final[dict[tuple[str, ...], frozenset[str]]] = {
    ("training",): RESUME_TRAINING_KEYS | {"reset_optim"}
}


@dataclass(frozen=True, slots=True)
class _ValueRule:
    """A value rule of a key the teacher runs set: where the key is, whether a value is
    acceptable, and the words `must be ...` that say what is."""

    parts: tuple[str, ...]
    accepts: Callable[[object], bool]
    must_be: str


def reject_dollar(what: str, value: Path | str) -> None:
    """Raise ConfigError, naming `what`, when `value` holds `$`."""
    # Eole runs `os.path.expandvars` over the text of a config file, so a `$NAME`
    # in a path would be silently replaced by an environment variable's value.
    if "$" in str(value):
        raise ConfigError(
            f"the {what} holds '$', which Eole would expand as a variable"
        )


def reject_glob_characters(what: str, value: Path | str) -> None:
    """Raise ConfigError, naming `what`, the path and the character, when `value` holds
    `[`, `]`, `*` or `?`."""
    # Eole loads a checkpoint through `glob(<model_path>/model.*.safetensors)`
    # (eole/models/model.py, `_load_safetensors_shards`), so these characters in the
    # checkpoint directory make it match no shard, or the shards of its siblings
    # (docs/VERIFY.md, D-EOLE-GLOB-CKPT).
    text = str(value)
    found = next((char for char in text if char in _GLOB_CHARACTERS), None)
    if found is not None:
        raise ConfigError(
            f"the {what} {text} holds {found!r}, which Eole reads as a glob pattern "
            "when it loads a checkpoint from it"
        )


def read_config(path: Path, what: str) -> Any:
    """What the YAML file at `path` holds, which the caller's validator then checks
    is a mapping of the right shape; `what` names the kind of config in messages.

    Raises:
        ConfigError: the file is missing, cannot be read or parsed, its text holds
            `$`, which Eole would expand as a variable, or a mapping or list in it
            contains itself (a recursive YAML anchor).
    """
    if not path.is_file():
        raise ConfigError(f"{what} file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        reject_dollar(f"text of the config {path}", text)
        loaded = yaml.safe_load(text)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid {what} {path}: cannot read or parse: {exc}"
        ) from exc
    _reject_self_reference(loaded, f"{what} {path}")
    return loaded


def _reject_self_reference(node: object, subject: str) -> None:
    # A recursive anchor (`a: &x {self: *x}`) loads as a mapping that holds itself,
    # which `merge` would follow until it ran out of stack.
    def refuse(where: str, problem: str) -> NoReturn:
        raise ConfigError(f"invalid {subject}: {where}: {problem}")

    for _ in _entries(node, "", fail=refuse):
        pass


def dump_config(cfg: dict[str, Any], path: Path) -> None:
    """Write `cfg` as YAML (UTF-8, LF, keys in their dict order) so the file appears
    whole: a failure leaves the old file, if any, and no temporary file. The
    directory of `path` must exist.

    Raises:
        ConfigError: YAML cannot dump a value of `cfg`; nothing is written.
    """
    try:
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"cannot write the config {path}: YAML cannot dump a value: {exc}"
        ) from exc
    part_path = unique_part_path(path)
    try:
        with part_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        part_path.replace(path)
    finally:
        part_path.unlink(missing_ok=True)


def validate_train_config(
    cfg: object,
    *,
    resumes: bool = False,
    continues: Path | str | None = None,
    resumes_continuation: bool = False,
) -> None:
    """Check that `cfg`, a multilingual training config as YAML gave it, is one
    `eole build_vocab` and `eole train` run as written; return None when it is.

    `resumes` says that `cfg` is the resume config of a run, the one file that may
    hold `training.train_from` (rule 11). It relaxes nothing else: everywhere
    else `train_from` is an unknown key (rule 5).

    `continues` names the checkpoint of ours that `cfg` is the first attempt of a
    continuation of (D9): the other place `training.train_from` may be, and the only
    one `training.reset_optim` may (rule 13). It relaxes nothing else, and it excludes
    `resumes`: an attempt resumes its own run or continues another's, not both.
    `resumes_continuation` (with `resumes`) says that the run being resumed began as a
    continuation: `training.reset_optim` is then allowed, and required, as `none` (rule 14).

    The rules, in the order they are checked (the first one broken is reported):

    1. `cfg` is a mapping.
    2. No key of `FORBIDDEN_KEYS` at any depth: every model uses sinusoidal positions
       and full multi-head attention.
    3. No `$` in any key or string: Eole expands it as an environment variable in the
       whole YAML text.
    4. Every value is plain YAML data (text, a number, a boolean, null, a list or a
       mapping), and of exactly that type, so a render can write it: an enum member
       or another subclass is refused, though `isinstance` would let it through.
    5. Each section is a mapping (Eole evaluates a string given for one as Python)
       and holds only the keys its allow-list names: the top level, each corpus of
       `data`, `transforms_configs` and the two transforms, `model`, its `encoder`,
       `decoder` and `embeddings`, and `training`.
    6. `model.architecture` is `transformer`, the architecture the allow-lists describe.
    7. `model.embeddings.position_encoding_type` is `SinusoidalInterleaved`, and so is
       the one of the encoder or the decoder when set.
    8. `model.hidden_size` and `model.embeddings.word_vec_size` are both set and equal
       (Eole's default of 512 breaks a smaller model), and so is the `hidden_size` of
       the encoder or the decoder when set.
    9. `overwrite` and `share_vocab` are true.
    10. The keys a teacher run sets hold values of the right kind: `tensorboard` is
       a boolean, `tensorboard_log_dir` and `log_file` non-empty text, `valid_metrics`
       a list of non-empty text, `early_stopping` a non-negative integer,
       `early_stopping_criteria` non-empty text or a list of it, `timeout` and
       `batch_size_multiple` positive integers, `master_port` an integer from 1 to
       65535, `master_ip`, `gpu_backend` and `self_attn_backend` non-empty text,
       `normalization` one of `NORMALIZATIONS`. A boolean is not an integer.
    11. With `resumes`, `training.train_from`, when present, is text that after
       normalisation (`os.path.normpath`, the file system is not touched) is
       `<training.model_path>/step_<N>` for a positive `N` without a leading zero,
       with no `..` component and no glob character. `training.model_path` must then
       be non-empty text, in the same form (relative or absolute) as `train_from`.
       The directory itself is the launcher's to find complete. A resume config that
       leaves `train_from` out is accepted.
    12. Every corpus but `valid` has `path_src`, `path_tgt`, a list of `transforms`,
       and both `src_prefix` and `tgt_prefix` as strings (a missing one is a
       TypeError in `build_vocab`; `''` means none), and when it has a `weight` that
       is a positive integer (a boolean is not one; no weight means 1). `valid` has
       `path_src` and `path_tgt`, may have `transforms: []`, and nothing else: Eole
       copies its prefix into the checkpoint, where it overrides every prediction.
    13. With `continues`, `training.train_from`, when present, is text that after
       normalisation (`os.path.normpath`) is `continues` normalised, with no `..`
       component and no glob character, and `training.reset_optim` is then present
       and exactly the text `all` (it loads the weights only; the other values keep
       the optimizer options of the checkpoint, or crash). `reset_optim` without
       `train_from` means nothing and is refused. A config that holds neither key is
       accepted. `update_vocab` stays an unknown key.
    14. With `resumes_continuation`, `training.reset_optim` is present and exactly the
       text `none` whenever `training.train_from` is (see
       `CONTINUATION_RESUME_RESET_OPTIM`), and means nothing without it.

    Raises:
        ValueError: `resumes` and `continues` are both given, `continues` is empty, or
            `resumes_continuation` is given without `resumes`.
        ConfigError: a rule is broken; the message is
            `invalid Eole config: <dotted.path>: <problem>`.
    """
    if resumes and continues is not None:
        raise ValueError("a config either resumes or continues, not both")
    if continues is not None and not str(continues):
        raise ValueError("continues must name the checkpoint the config continues")
    if resumes_continuation and not resumes:
        raise ValueError("resumes_continuation needs resumes")
    if not isinstance(cfg, Mapping):
        _fail("config", f"expected a mapping, got {type(cfg).__name__}")
    _check_no_forbidden_keys(cfg)
    _check_no_variable_references(cfg)
    _check_plain_data(cfg)
    _check_sections(
        cfg,
        _extra_keys(
            resumes=resumes,
            continues=continues,
            resumes_continuation=resumes_continuation,
        ),
    )
    _check_architecture(cfg)
    _check_position_encoding(cfg)
    _check_hidden_size(cfg)
    _check_switches(cfg)
    _check_run_settings(cfg)
    if resumes:
        _check_train_from(cfg)
    if resumes_continuation:
        _check_reset_optim(cfg, CONTINUATION_RESUME_RESET_OPTIM, why=_RESUME_WHY)
    if continues is not None:
        _check_continuation(cfg, str(continues))
    _check_corpora(cfg)


def merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """`base` with `overrides` laid over it, as new dicts and lists that share nothing
    with either input.

    Mappings merge key by key at any depth; every other value, a list included,
    replaces what it meets. An override of `None` deletes the key (nothing happens
    when `base` has none), also inside a mapping that replaces a value, so merging the
    same overrides again changes nothing. Keys keep the order of `base`, then the
    order in which the overrides add them.
    """
    merged = {key: _copied(value) for key, value in base.items()}
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping):
            existing = merged.get(key)
            merged[key] = merge(
                existing if isinstance(existing, Mapping) else {}, value
            )
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merged_config(
    base: Path,
    overrides: Mapping[str, Any],
    *,
    resumes: bool = False,
    continues: Path | str | None = None,
) -> dict[str, Any]:
    """The training config that is the YAML mapping in `base` with `overrides` merged
    over it (`merge`), validated (`validate_train_config`, with `resumes` and
    `continues`); nothing is written. The base need not be complete: the merged config
    is what is validated.

    Raises:
        ConfigError: `base` is missing, cannot be read or parsed, is empty, is not a
            mapping, holds `$` or a recursive anchor; `overrides` contains itself; or
            the merged config is not valid.
    """
    raw = read_config(base, "base Eole config")
    if not isinstance(raw, Mapping):
        problem = (
            "is empty"
            if raw is None
            else f"must be a mapping, got {type(raw).__name__}"
        )
        raise ConfigError(f"invalid base Eole config {base}: {problem}")
    _reject_self_reference(overrides, "Eole config overrides")
    cfg = merge(raw, overrides)
    validate_train_config(cfg, resumes=resumes, continues=continues)
    return cfg


def render(
    base: Path,
    overrides: Mapping[str, Any],
    out: Path,
    *,
    resumes: bool = False,
    continues: Path | str | None = None,
) -> Path:
    """Write the training config that is the YAML mapping in `base` with `overrides`
    merged over it (`merged_config`, with `resumes` and `continues`) to `out`, and
    return `out`.

    A config that fails writes nothing, so a file at `out` stays as it was. The
    directory of `out` is created; the file appears whole (`dump_config`), keys in
    merge order.

    Raises:
        ConfigError: `out` is `base` (the same file once links and `..` are resolved),
            a directory, or below something that is not a directory; and what
            `merged_config` raises.
    """
    _require_writable_output(base, out)
    cfg = merged_config(base, overrides, resumes=resumes, continues=continues)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    return out


def _require_writable_output(base: Path, out: Path) -> None:
    # Checked before anything is created, so a refused render leaves no directory.
    if out.resolve() == base.resolve():
        raise ConfigError(
            f"cannot write {out}: it is the base config, which it would overwrite"
        )
    if out.is_dir():
        raise ConfigError(f"cannot write {out}: it is a directory")
    nearest = next((path for path in out.parents if os.path.lexists(path)), None)
    if nearest is not None and not nearest.is_dir():
        raise ConfigError(f"cannot write {out}: {nearest} is not a directory")


def _copied(value: Any) -> Any:
    return merge(value, {}) if isinstance(value, Mapping) else copy.deepcopy(value)


def _fail(path: str, problem: str) -> NoReturn:
    raise ConfigError(f"invalid Eole config: {path}: {problem}")


def _join(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def _got(value: object) -> str:
    return "but it is missing" if value is _MISSING else f"got {value!r}"


def _entries(
    node: object,
    path: str,
    ancestors: tuple[int, ...] = (),
    *,
    fail: Callable[[str, str], NoReturn] = _fail,
) -> Iterator[tuple[str, object, object]]:
    """Every entry below `node` as (key path, key, value), through nested mappings,
    lists and tuples; a list item's key is its index and its path ends `[index]`.

    A mapping or list that contains itself is reported by calling `fail` with the
    key path and the problem; it must raise.
    """
    if isinstance(node, Mapping):
        entries: list[tuple[object, object]] = list(node.items())
    elif isinstance(node, list | tuple):
        entries = list(enumerate(node))
    else:
        return
    inside = (*ancestors, id(node))
    for key, value in entries:
        child = _join(path, key) if isinstance(node, Mapping) else f"{path}[{key}]"
        if id(value) in inside:
            fail(child, "refers back to a mapping or list that contains it")
        yield child, key, value
        yield from _entries(value, child, inside, fail=fail)


def _check_no_forbidden_keys(cfg: Mapping[Any, Any]) -> None:
    for path, key, _ in _entries(cfg, ""):
        if key in FORBIDDEN_KEYS:
            _fail(
                path,
                "is not allowed: every Rinchen model uses sinusoidal positions and "
                "full multi-head attention",
            )


def _check_no_variable_references(cfg: Mapping[Any, Any]) -> None:
    for path, key, value in _entries(cfg, ""):
        if isinstance(key, str) and "$" in key:
            _fail(path, "the key holds '$', which Eole would expand as a variable")
        if isinstance(value, str) and "$" in value:
            _fail(
                path,
                f"holds '$', which Eole would expand as a variable: {value!r}",
            )


def _check_plain_data(cfg: Mapping[Any, Any]) -> None:
    for path, _, value in _entries(cfg, ""):
        if type(value) not in _PLAIN_TYPES:
            _fail(
                path,
                "must be plain YAML data (text, a number, true or false, null, a "
                f"list or a mapping), got {type(value).__name__}",
            )


def _lookup(cfg: Mapping[Any, Any], *parts: str) -> Any:
    """The value at the key path `parts`, or `_MISSING` when a key is absent. Every
    key but the last must lead to a section, which `_check_sections` has shown to be a
    mapping where present."""
    node: Any = cfg
    for part in parts:
        if part not in node:
            return _MISSING
        node = node[part]
    return node


def _require_present(cfg: Mapping[Any, Any], *parts: str) -> Any:
    value = _lookup(cfg, *parts)
    if value is _MISSING:
        _fail(".".join(parts), "is missing")
    return value


def _as_mapping(value: object, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(
    section: Mapping[Any, Any], path: str, allowed: frozenset[str], level: str
) -> None:
    for key in section:
        if key not in allowed:
            _fail(_join(path, key), f"unknown key in {level}")


def _extra_keys(
    *, resumes: bool, continues: Path | str | None, resumes_continuation: bool
) -> Mapping[tuple[str, ...], frozenset[str]]:
    """The keys a section may hold besides its allow-list, for the kind of config."""
    if resumes_continuation:
        return _CONTINUATION_RESUME_KEYS
    if resumes:
        return _RESUME_KEYS
    if continues is not None:
        return _CONTINUATION_KEYS
    return {}


def _check_sections(
    cfg: Mapping[Any, Any], extra_keys: Mapping[tuple[str, ...], frozenset[str]]
) -> None:
    # Later checks rely on this one: a section that is present is a mapping.
    _check_keys(cfg, "", TOP_LEVEL_KEYS, "the top level")
    if "data" in cfg:
        for name, corpus in _as_mapping(cfg["data"], "data").items():
            if not isinstance(name, str):
                _fail(
                    f"data.{name}",
                    f"the corpus name must be text, got {type(name).__name__}",
                )
            path = f"data.{name}"
            _check_keys(_as_mapping(corpus, path), path, CORPUS_KEYS, "a data corpus")
    for parts, allowed in _SECTION_LEVELS:
        section = _lookup(cfg, *parts)
        if section is not _MISSING:
            path = ".".join(parts)
            allowed = allowed | extra_keys.get(parts, frozenset())
            _check_keys(_as_mapping(section, path), path, allowed, path)


def _check_architecture(cfg: Mapping[Any, Any]) -> None:
    _require_present(cfg, "model")
    architecture = _lookup(cfg, "model", "architecture")
    if architecture != _ARCHITECTURE:
        _fail(
            "model.architecture",
            f"must be {_ARCHITECTURE!r}, {_got(architecture)}: the allow-lists are "
            "those of that architecture",
        )


def _check_position_encoding(cfg: Mapping[Any, Any]) -> None:
    _require_present(cfg, "model", "embeddings")
    where = [("model", "embeddings")]
    where += [
        ("model", layer)
        for layer in ("encoder", "decoder")
        if _lookup(cfg, "model", layer, "position_encoding_type") is not _MISSING
    ]
    for parts in where:
        value = _lookup(cfg, *parts, "position_encoding_type")
        if value != POSITION_ENCODING:
            _fail(
                ".".join([*parts, "position_encoding_type"]),
                f"must be {POSITION_ENCODING}, {_got(value)}",
            )


def _check_hidden_size(cfg: Mapping[Any, Any]) -> None:
    for parts in (("model", "hidden_size"), ("model", "embeddings", "word_vec_size")):
        if _lookup(cfg, *parts) is _MISSING:
            _fail(
                ".".join(parts),
                "is missing: Eole's default is 512, which breaks a smaller model",
            )
    hidden_size = _lookup(cfg, "model", "hidden_size")
    word_vec_size = _lookup(cfg, "model", "embeddings", "word_vec_size")
    if word_vec_size != hidden_size:
        _fail(
            "model.embeddings.word_vec_size",
            f"must equal model.hidden_size ({hidden_size!r}), got {word_vec_size!r}",
        )
    for layer in ("encoder", "decoder"):
        layer_size = _lookup(cfg, "model", layer, "hidden_size")
        if layer_size is not _MISSING and layer_size != word_vec_size:
            _fail(
                f"model.{layer}.hidden_size",
                f"must equal model.embeddings.word_vec_size ({word_vec_size!r}), "
                f"got {layer_size!r}",
            )


def _check_switches(cfg: Mapping[Any, Any]) -> None:
    for key, why in (
        ("overwrite", "Eole stops at an existing vocabulary file otherwise"),
        ("share_vocab", "every Rinchen model shares one vocabulary"),
    ):
        value = cfg.get(key, _MISSING)
        if value is not True:
            _fail(key, f"must be true: {why}, {_got(value)}")


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_non_negative_int(value: object) -> bool:
    return is_strict_int(value) and value >= 0


def _is_positive_int(value: object) -> bool:
    return is_strict_int(value) and value > 0


def _is_port(value: object) -> bool:
    return is_strict_int(value) and 1 <= value <= _MAX_PORT


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_text_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_text(item) for item in value)


def _is_text_or_nonempty_text_list(value: object) -> bool:
    return _is_text(value) or (_is_text_list(value) and bool(value))


def _is_normalization(value: object) -> bool:
    return value in NORMALIZATIONS


_VALUE_RULES: Final = (
    _ValueRule(("tensorboard",), _is_bool, "true or false"),
    _ValueRule(("tensorboard_log_dir",), _is_text, "a non-empty string"),
    _ValueRule(("log_file",), _is_text, "a non-empty string"),
    _ValueRule(("valid_metrics",), _is_text_list, "a list of non-empty strings"),
    _ValueRule(
        ("training", "early_stopping"), _is_non_negative_int, "a non-negative integer"
    ),
    _ValueRule(
        ("training", "early_stopping_criteria"),
        _is_text_or_nonempty_text_list,
        "a non-empty string or a list of non-empty strings",
    ),
    _ValueRule(("training", "timeout"), _is_positive_int, "a positive integer"),
    _ValueRule(("training", "master_ip"), _is_text, "a non-empty string"),
    _ValueRule(
        ("training", "master_port"), _is_port, f"an integer from 1 to {_MAX_PORT}"
    ),
    _ValueRule(("training", "gpu_backend"), _is_text, "a non-empty string"),
    _ValueRule(
        ("training", "normalization"),
        _is_normalization,
        " or ".join(repr(name) for name in NORMALIZATIONS),
    ),
    _ValueRule(
        ("training", "batch_size_multiple"), _is_positive_int, "a positive integer"
    ),
    _ValueRule(("training", "self_attn_backend"), _is_text, "a non-empty string"),
)


def _check_run_settings(cfg: Mapping[Any, Any]) -> None:
    for rule in _VALUE_RULES:
        value = _lookup(cfg, *rule.parts)
        if value is not _MISSING and not rule.accepts(value):
            _fail(".".join(rule.parts), f"must be {rule.must_be}, got {value!r}")


def _check_free_of_dotdot_and_globs(train_from: object, path: str) -> None:
    """`train_from` is text that no `..` component could turn into another directory and
    that Eole will not read as a glob pattern when it loads a checkpoint from it."""
    if not isinstance(train_from, str):
        _fail(path, f"must be a string, got {type(train_from).__name__}")
    if ".." in train_from.split("/"):
        _fail(
            path,
            "holds a '..' component, which could leave the run's checkpoint "
            f"directory: {train_from!r}",
        )
    glob_character = next((c for c in train_from if c in _GLOB_CHARACTERS), None)
    if glob_character is not None:
        _fail(
            path,
            f"holds {glob_character!r}, which Eole reads as a glob pattern when it "
            f"loads a checkpoint from it: {train_from!r}",
        )


def _check_train_from(cfg: Mapping[Any, Any]) -> None:
    path = f"training.{_TRAIN_FROM}"
    train_from = _lookup(cfg, "training", _TRAIN_FROM)
    if train_from is _MISSING:
        return
    _check_free_of_dotdot_and_globs(train_from, path)
    model_path = _lookup(cfg, "training", "model_path")
    if not _is_text(model_path):
        _fail(
            path,
            "cannot be checked, training.model_path must be a non-empty string "
            f"to compare with, {_got(model_path)}",
        )
    parent, name = os.path.split(os.path.normpath(train_from))
    if (parent or ".") != os.path.normpath(model_path) or not _STEP_DIRECTORY.fullmatch(
        name
    ):
        _fail(
            path,
            "must be '<training.model_path>/step_<N>' of this run (N a positive "
            f"number without a leading zero; model_path is {model_path!r}), got "
            f"{train_from!r}",
        )


def _check_continuation(cfg: Mapping[Any, Any], source: str) -> None:
    train_from = _lookup(cfg, "training", _TRAIN_FROM)
    if train_from is not _MISSING:
        _check_free_of_dotdot_and_globs(train_from, f"training.{_TRAIN_FROM}")
        if os.path.normpath(train_from) != os.path.normpath(source):
            _fail(
                f"training.{_TRAIN_FROM}",
                f"must be the continuation source {os.path.normpath(source)!r}, got "
                f"{train_from!r}",
            )
    _check_reset_optim(cfg, CONTINUATION_RESET_OPTIM, why=_CONTINUE_WHY)


def _check_reset_optim(cfg: Mapping[Any, Any], wanted: str, *, why: str) -> None:
    """`training.reset_optim` is exactly `wanted` whenever `training.train_from` is
    there, and is not there without it (it would mean nothing)."""
    train_from = _lookup(cfg, "training", _TRAIN_FROM)
    reset_optim = _lookup(cfg, "training", _RESET_OPTIM)
    if train_from is _MISSING:
        if reset_optim is not _MISSING:
            _fail(
                f"training.{_TRAIN_FROM}",
                f"is missing: training.{_RESET_OPTIM} means nothing without the "
                "checkpoint it applies to",
            )
        return
    if not isinstance(reset_optim, str) or reset_optim != wanted:
        _fail(
            f"training.{_RESET_OPTIM}",
            f"must be {wanted!r}, {_got(reset_optim)}: {why}",
        )


def _check_corpora(cfg: Mapping[Any, Any]) -> None:
    for name, corpus in cfg.get("data", {}).items():
        if name == _VALID:
            _check_validation_corpus(corpus, f"data.{name}")
        else:
            _check_training_corpus(corpus, f"data.{name}")


def _check_paths(corpus: Mapping[Any, Any], path: str) -> None:
    for key in ("path_src", "path_tgt"):
        value = corpus.get(key, _MISSING)
        if value is _MISSING:
            _fail(f"{path}.{key}", "is missing")
        if not isinstance(value, str) or not value:
            _fail(f"{path}.{key}", f"must be a non-empty path, got {value!r}")


def _check_training_corpus(corpus: Mapping[Any, Any], path: str) -> None:
    _check_paths(corpus, path)
    transforms = corpus.get("transforms", _MISSING)
    if transforms is _MISSING:
        _fail(f"{path}.transforms", "is missing")
    if not isinstance(transforms, list) or not all(
        isinstance(name, str) for name in transforms
    ):
        _fail(
            f"{path}.transforms",
            f"must be a list of transform names, got {transforms!r}",
        )
    for key in ("src_prefix", "tgt_prefix"):
        prefix = corpus.get(key, _MISSING)
        if prefix is _MISSING:
            _fail(
                f"{path}.{key}",
                "is missing: Eole's prefix transform fails with a TypeError without "
                "it; use '' for no prefix",
            )
        if not isinstance(prefix, str):
            _fail(f"{path}.{key}", f"must be a string ('' for none), got {prefix!r}")
    if "weight" in corpus:
        weight = corpus["weight"]
        # A boolean is an int to Python, and no weight to Eole's round robin.
        if not is_strict_int(weight) or weight < 1:
            _fail(
                f"{path}.weight",
                "must be a positive integer (the number of consecutive examples "
                f"taken per round), got {weight!r}",
            )


def _check_validation_corpus(corpus: Mapping[Any, Any], path: str) -> None:
    _check_paths(corpus, path)
    if "transforms" in corpus and corpus["transforms"] != []:
        _fail(
            f"{path}.transforms",
            "must be [] or left out: the harness tokenises the validation corpus, "
            f"got {corpus['transforms']!r}",
        )
    for key in ("src_prefix", "tgt_prefix"):
        if key in corpus:
            _fail(
                f"{path}.{key}",
                "not allowed on the validation corpus: Eole copies a prefix into the "
                "checkpoint, where it overrides every prediction",
            )
    if "weight" in corpus:
        _fail(
            f"{path}.weight",
            "not allowed on the validation corpus: it takes no weight",
        )
