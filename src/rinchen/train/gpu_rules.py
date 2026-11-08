"""The GPU and stopping rules a long Eole run is held to before it starts (D8, S5).

Eole checks two things about `world_size` and `gpu_ranks` and silently accepts the rest
of what goes wrong on a GPU machine. These rules turn the silent cases into a message,
before an hour of a GPU node is spent on them. They read the effective training config
(the file with the launcher's overrides laid over it). Every GPU, NCCL and multi-process
statement here is read from Eole's source or from a CPU test with a shim, none was
measured on a GPU (spike, D8).

The rules:

1. `training.world_size` is a positive integer, `training.gpu_ranks` a list of distinct
   non-negative integers (Eole's defaults: 1 and none).
2. `world_size` above 1 with no `gpu_ranks` starts no process: Eole exits 0 without
   training and without a checkpoint. Refused.
3. `world_size` below the number of ranks, or equal to it, positive, with no rank 0,
   are the two rules Eole itself checks. Refused here, with the same words.
4. With `world_size` above 1 only rank 0 validates and saves while the others wait at
   the next collective, so `training.timeout` (Eole's default is 60 seconds) must cover
   a validation and a multi-gigabyte save: at least `MIN_MULTI_PROCESS_TIMEOUT_S`.
5. With `world_size` above 1 the per-corpus statistics are gathered through a
   4096-byte buffer: at most `MAX_TRAINING_CORPORA` training corpora.
6. With `training.early_stopping` on, `training.early_stopping_criteria` is one string:
   a list, or none (Eole then uses perplexity and accuracy together), stops on a
   different rule than the one the operator reads.
7. With early stopping on and a validation at every save, Eole keeps the last
   `keep_checkpoint` saves and never the best one: `keep_checkpoint` is -1 (all) or at
   least `early_stopping` + 1, so the best of the last validations is still on disk.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn

from rinchen.errors import ConfigError

MIN_MULTI_PROCESS_TIMEOUT_S: Final = 3600
MAX_TRAINING_CORPORA: Final = 80
# Eole 0.6.4's defaults (`TrainingConfig`), for the keys a config leaves out.
_DEFAULT_WORLD_SIZE: Final = 1
_DEFAULT_VALID_STEPS: Final = 10000
_DEFAULT_SAVE_STEPS: Final = 5000
_KEEP_ALL: Final = -1
_VALID_CORPUS: Final = "valid"
_UNSET: Final = object()


@dataclass(frozen=True, slots=True)
class GpuSettings:
    """The `world_size` and `gpu_ranks` a config trains with, checked."""

    world_size: int
    gpu_ranks: tuple[int, ...]

    @property
    def uses_gpu(self) -> bool:
        return bool(self.gpu_ranks)


def check_gpu_rules(cfg: Mapping[str, Any]) -> GpuSettings:
    """The GPU settings of the effective training config `cfg`, after the rules of this
    module hold for it.

    `cfg` has passed `eole_cfg.validate_train_config`, which checks the keys and the
    kinds of value of the teacher settings but not these.

    Raises:
        ConfigError: a rule is broken; the message is
            `invalid Eole config: <dotted.path>: <problem>`.
    """
    training = cfg.get("training")
    if not isinstance(training, Mapping):
        _refuse("training", "is missing or not a mapping")
    settings = GpuSettings(_world_size(training), _gpu_ranks(training))
    _check_ranks(settings)
    if settings.world_size > 1:
        _check_collective_timeout(training)
        _check_corpus_count(cfg)
    _check_early_stopping(training)
    return settings


def _refuse(path: str, problem: str) -> NoReturn:
    raise ConfigError(f"invalid Eole config: {path}: {problem}")


def _is_int(value: object) -> bool:
    # A boolean is an int to Python and neither a count nor a rank to Eole.
    return isinstance(value, int) and not isinstance(value, bool)


def _world_size(training: Mapping[str, Any]) -> int:
    value = training.get("world_size", _DEFAULT_WORLD_SIZE)
    if not _is_int(value) or value < 1:
        _refuse("training.world_size", f"must be a positive integer, got {value!r}")
    return int(value)


def _gpu_ranks(training: Mapping[str, Any]) -> tuple[int, ...]:
    value = training.get("gpu_ranks", [])
    if not isinstance(value, list | tuple) or not all(
        _is_int(rank) and rank >= 0 for rank in value
    ):
        _refuse(
            "training.gpu_ranks",
            f"must be a list of non-negative integers, got {value!r}",
        )
    ranks = tuple(value)
    for rank in ranks:
        if ranks.count(rank) > 1:
            _refuse("training.gpu_ranks", f"lists rank {rank} twice: {list(ranks)!r}")
    return ranks


def _check_ranks(settings: GpuSettings) -> None:
    world_size, ranks = settings.world_size, settings.gpu_ranks
    if world_size > 1 and not ranks:
        _refuse(
            "training.gpu_ranks",
            f"is empty although training.world_size is {world_size}: Eole would start "
            "no process and exit 0 without training or a checkpoint; list one rank per "
            "GPU or set world_size 1",
        )
    if world_size < len(ranks):
        _refuse(
            "training.world_size",
            f"is {world_size}, below the {len(ranks)} entries of gpu_ranks "
            f"{list(ranks)!r}: Eole needs world_size to be at least their number",
        )
    if ranks and world_size == len(ranks) and 0 not in ranks:
        _refuse(
            "training.gpu_ranks",
            f"{list(ranks)!r} has no master rank 0 although world_size is "
            f"{world_size}, the number of ranks: Eole rejects it",
        )


def _check_collective_timeout(training: Mapping[str, Any]) -> None:
    timeout = training.get("timeout", _UNSET)
    if not _is_int(timeout) or timeout < MIN_MULTI_PROCESS_TIMEOUT_S:
        got = "unset (Eole's default is 60)" if timeout is _UNSET else repr(timeout)
        _refuse(
            "training.timeout",
            f"must be at least {MIN_MULTI_PROCESS_TIMEOUT_S} seconds when world_size is "
            "above 1: rank 0 alone validates and saves while the other ranks wait, "
            f"and a wait beyond the timeout kills the job; got {got}",
        )


def _check_corpus_count(cfg: Mapping[str, Any]) -> None:
    data = cfg.get("data")
    corpora = (
        [name for name in data if name != _VALID_CORPUS]
        if isinstance(data, Mapping)
        else []
    )
    if len(corpora) > MAX_TRAINING_CORPORA:
        _refuse(
            "data",
            f"has {len(corpora)} training corpora, more than the "
            f"{MAX_TRAINING_CORPORA} Eole's statistics gather can carry when "
            "world_size is above 1 (a 4096-byte buffer); merge corpora",
        )


def _check_early_stopping(training: Mapping[str, Any]) -> None:
    patience = training.get("early_stopping", 0)
    if not _is_int(patience) or patience < 1:
        return
    criteria = training.get("early_stopping_criteria")
    if not isinstance(criteria, str) or not criteria:
        got = "unset" if criteria is None else repr(criteria)
        _refuse(
            "training.early_stopping_criteria",
            "must be a single string (one criterion, such as 'ppl') when "
            f"early_stopping is {patience}: unset or a list, Eole judges by several "
            f"criteria at once; got {got}",
        )
    valid_steps = training.get("valid_steps", _DEFAULT_VALID_STEPS)
    save_steps = training.get("save_checkpoint_steps", _DEFAULT_SAVE_STEPS)
    if valid_steps != save_steps:
        return
    keep = training.get("keep_checkpoint", _KEEP_ALL)
    if keep != _KEEP_ALL and (not _is_int(keep) or keep < patience + 1):
        _refuse(
            "training.keep_checkpoint",
            f"is {keep!r}: with early_stopping {patience} and a validation at every "
            "save, Eole keeps only the last saves and never the best one, so it must "
            f"be -1 or at least {patience + 1}",
        )
