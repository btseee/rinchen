"""Tests for `rinchen.train.gpu_rules`: the GPU and stopping rules a run is held to
before it starts.

The baseline is the D7 test config (`tests.support.eole_train_config.make_config`), a
CPU run with two training corpora and one validation corpus; each rule is tested by
changing the `training` section (or the corpora) and reading the verdict. No test
starts Eole and none touches a GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.train.gpu_rules import (
    MAX_TRAINING_CORPORA,
    MIN_MULTI_PROCESS_TIMEOUT_S,
    GpuSettings,
    check_gpu_rules,
)
from tests.support.eole_train_config import make_config

PREFIX = "invalid Eole config: "


def config_with(**training: Any) -> dict[str, Any]:
    cfg = make_config()
    cfg["training"].update(training)
    return cfg


def gpu_config(ranks: int = 8, **training: Any) -> dict[str, Any]:
    """A multi-process config: `ranks` GPUs, world size equal, a long timeout."""
    settings: dict[str, Any] = {
        "world_size": ranks,
        "gpu_ranks": list(range(ranks)),
        "timeout": 7200,
    }
    settings.update(training)
    return config_with(**settings)


def corpora(count: int) -> dict[str, Any]:
    template = make_config()["data"]["en-mn.auth"]
    data = {f"c{index}": dict(template) for index in range(count)}
    data["valid"] = make_config()["data"]["valid"]
    return data


def refusal_of(cfg: dict[str, Any]) -> str:
    with pytest.raises(ConfigError) as raised:
        check_gpu_rules(cfg)
    message = str(raised.value)
    assert message.startswith(PREFIX)
    return message


# ---- what is accepted -----------------------------------------------------------


def test_a_cpu_config_is_accepted_and_reported_as_using_no_gpu() -> None:
    settings = check_gpu_rules(make_config())

    assert settings == GpuSettings(world_size=1, gpu_ranks=())
    assert not settings.uses_gpu


def test_eoles_defaults_apply_when_the_config_leaves_the_keys_out() -> None:
    cfg = make_config()
    del cfg["training"]["world_size"]
    del cfg["training"]["gpu_ranks"]

    assert check_gpu_rules(cfg) == GpuSettings(1, ())


def test_one_gpu_is_accepted() -> None:
    settings = check_gpu_rules(config_with(world_size=1, gpu_ranks=[0]))

    assert settings == GpuSettings(1, (0,))
    assert settings.uses_gpu


def test_the_eight_gpu_teacher_prior_is_accepted() -> None:
    settings = check_gpu_rules(gpu_config(8))

    assert settings == GpuSettings(8, (0, 1, 2, 3, 4, 5, 6, 7))


def test_the_second_node_of_two_is_accepted() -> None:
    # Node 1 of 2x4: the world is 8, this node runs ranks 4 to 7.
    cfg = gpu_config(8, gpu_ranks=[4, 5, 6, 7])

    assert check_gpu_rules(cfg) == GpuSettings(8, (4, 5, 6, 7))


def test_the_ranks_may_come_as_a_tuple() -> None:
    assert check_gpu_rules(config_with(world_size=2, gpu_ranks=(0, 1), timeout=3600))


def test_the_config_is_not_changed() -> None:
    cfg = gpu_config(2)
    before = repr(cfg)

    check_gpu_rules(cfg)

    assert repr(cfg) == before


# ---- world size and ranks -------------------------------------------------------


@pytest.mark.parametrize("world_size", [0, -1, 1.0, "2", None, True, [2]])
def test_a_world_size_that_is_not_a_positive_integer_is_refused(
    world_size: object,
) -> None:
    message = refusal_of(config_with(world_size=world_size))

    assert message == (
        f"{PREFIX}training.world_size: must be a positive integer, got {world_size!r}"
    )


@pytest.mark.parametrize(
    "ranks", [[-1], [0, 1.5], ["0"], [True], [None], "0,1", 0, {0: 1}, [[0]]]
)
def test_ranks_that_are_not_a_list_of_non_negative_integers_are_refused(
    ranks: object,
) -> None:
    message = refusal_of(config_with(gpu_ranks=ranks))

    assert message.startswith(
        f"{PREFIX}training.gpu_ranks: must be a list of non-negative integers"
    )


def test_a_rank_listed_twice_is_refused_naming_it() -> None:
    message = refusal_of(
        config_with(world_size=4, gpu_ranks=[0, 1, 1, 2], timeout=3600)
    )

    assert "training.gpu_ranks: lists rank 1 twice" in message


@pytest.mark.parametrize("world_size", [2, 8])
def test_several_processes_with_no_ranks_is_the_silent_no_op_and_is_refused(
    world_size: int,
) -> None:
    message = refusal_of(config_with(world_size=world_size, gpu_ranks=[], timeout=3600))

    assert "training.gpu_ranks: is empty although training.world_size is" in message
    assert "exit 0 without training" in message


def test_several_processes_with_the_ranks_left_out_is_refused_too() -> None:
    cfg = config_with(world_size=4, timeout=3600)
    del cfg["training"]["gpu_ranks"]

    assert "is empty although training.world_size is 4" in refusal_of(cfg)


@pytest.mark.parametrize(
    ("world_size", "ranks"), [(1, [0, 1]), (2, [0, 1, 2]), (4, [0, 1, 2, 3, 4])]
)
def test_a_world_size_below_the_number_of_ranks_is_refused(
    world_size: int, ranks: list[int]
) -> None:
    message = refusal_of(
        config_with(world_size=world_size, gpu_ranks=ranks, timeout=3600)
    )

    assert (
        f"training.world_size: is {world_size}, below the {len(ranks)} entries"
        in message
    )


@pytest.mark.parametrize(
    ("world_size", "ranks"), [(1, [1]), (2, [1, 2]), (3, [5, 6, 7])]
)
def test_as_many_ranks_as_the_world_without_rank_zero_is_refused(
    world_size: int, ranks: list[int]
) -> None:
    message = refusal_of(
        config_with(world_size=world_size, gpu_ranks=ranks, timeout=3600)
    )

    assert "has no master rank 0" in message


def test_fewer_ranks_than_the_world_need_not_hold_rank_zero() -> None:
    # The other half of a two-node job.
    assert check_gpu_rules(gpu_config(8, gpu_ranks=[7]))


def test_an_override_that_contradicts_the_config_is_caught_in_the_merged_config() -> (
    None
):
    # The launcher lays `--world-size 1` over a config that lists eight ranks.
    cfg = gpu_config(8)
    cfg["training"]["world_size"] = 1

    assert "below the 8 entries" in refusal_of(cfg)


# ---- the collective timeout -----------------------------------------------------


def test_the_minimum_timeout_is_an_hour() -> None:
    assert MIN_MULTI_PROCESS_TIMEOUT_S == 3600


def test_a_collective_timeout_of_exactly_an_hour_is_accepted() -> None:
    assert check_gpu_rules(gpu_config(2, timeout=3600))


@pytest.mark.parametrize("timeout", [3599, 60, 0, -1, 3600.5, "7200", True, [7200]])
def test_a_short_or_malformed_timeout_is_refused_when_several_processes_run(
    timeout: object,
) -> None:
    message = refusal_of(gpu_config(2, timeout=timeout))

    assert message.startswith(f"{PREFIX}training.timeout: must be at least 3600")
    assert f"got {timeout!r}" in message


def test_an_unset_timeout_is_refused_with_eoles_default_named() -> None:
    cfg = gpu_config(2)
    del cfg["training"]["timeout"]

    message = refusal_of(cfg)

    assert "training.timeout" in message
    assert "unset (Eole's default is 60)" in message


@pytest.mark.parametrize("timeout", [None, 1, 59])
def test_a_single_process_run_does_not_need_a_long_timeout(timeout: int | None) -> None:
    cfg = config_with(world_size=1, gpu_ranks=[0])
    if timeout is None:
        cfg["training"].pop("timeout", None)
    else:
        cfg["training"]["timeout"] = timeout

    assert check_gpu_rules(cfg) == GpuSettings(1, (0,))


# ---- the number of corpora ------------------------------------------------------


def test_eighty_training_corpora_are_accepted_with_several_processes() -> None:
    cfg = gpu_config(2)
    cfg["data"] = corpora(MAX_TRAINING_CORPORA)

    assert check_gpu_rules(cfg).world_size == 2


def test_eighty_one_training_corpora_are_refused_with_several_processes() -> None:
    cfg = gpu_config(2)
    cfg["data"] = corpora(MAX_TRAINING_CORPORA + 1)

    message = refusal_of(cfg)

    assert message.startswith(f"{PREFIX}data: has 81 training corpora")
    assert "4096-byte buffer" in message


def test_the_validation_corpus_is_not_a_training_corpus() -> None:
    cfg = gpu_config(2)
    cfg["data"] = corpora(MAX_TRAINING_CORPORA)

    assert "valid" in cfg["data"]
    assert check_gpu_rules(cfg)


def test_a_single_process_run_may_have_any_number_of_corpora() -> None:
    cfg = make_config()
    cfg["data"] = corpora(200)

    assert check_gpu_rules(cfg) == GpuSettings(1, ())


def test_a_config_without_data_has_no_corpora_to_count() -> None:
    cfg = gpu_config(2)
    del cfg["data"]

    assert check_gpu_rules(cfg).world_size == 2


# ---- early stopping -------------------------------------------------------------


def early_stopping(**training: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "early_stopping": 6,
        "early_stopping_criteria": "ppl",
        "valid_steps": 5000,
        "save_checkpoint_steps": 5000,
        "keep_checkpoint": 8,
    }
    settings.update(training)
    return config_with(**settings)


def test_early_stopping_with_one_criterion_and_enough_checkpoints_is_accepted() -> None:
    assert check_gpu_rules(early_stopping()) == GpuSettings(1, ())


@pytest.mark.parametrize("criteria", [["ppl"], ["ppl", "accuracy"], [], 3, {"ppl": 1}])
def test_a_criteria_list_is_refused_when_early_stopping_is_on(criteria: object) -> None:
    message = refusal_of(early_stopping(early_stopping_criteria=criteria))

    assert message.startswith(
        f"{PREFIX}training.early_stopping_criteria: must be a single string"
    )
    assert f"got {criteria!r}" in message


def test_unset_criteria_are_refused_when_early_stopping_is_on() -> None:
    cfg = early_stopping()
    del cfg["training"]["early_stopping_criteria"]

    message = refusal_of(cfg)

    assert "training.early_stopping_criteria" in message
    assert "got unset" in message


def test_the_criteria_are_not_looked_at_when_early_stopping_is_off() -> None:
    assert check_gpu_rules(
        early_stopping(early_stopping=0, early_stopping_criteria=["a", "b"])
    )


def test_a_boolean_is_not_a_patience() -> None:
    # `true` is 1 to Python, and no number of validations to wait for: early stopping is
    # not on, so the criteria list is not looked at (the Eole config check refuses the
    # boolean itself).
    assert check_gpu_rules(
        early_stopping(early_stopping=True, early_stopping_criteria=["a", "b"])
    )


def test_early_stopping_left_out_is_off() -> None:
    cfg = config_with(early_stopping_criteria=["a", "b"], keep_checkpoint=0)

    assert check_gpu_rules(cfg)


@pytest.mark.parametrize("keep", [0, 1, 6, -2])
def test_too_few_kept_checkpoints_are_refused_when_every_validation_is_saved(
    keep: int,
) -> None:
    message = refusal_of(early_stopping(keep_checkpoint=keep))

    assert message.startswith(f"{PREFIX}training.keep_checkpoint: is {keep}")
    assert "at least 7" in message


@pytest.mark.parametrize("keep", [7, 8, 100, -1])
def test_enough_kept_checkpoints_are_accepted(keep: int) -> None:
    assert check_gpu_rules(early_stopping(keep_checkpoint=keep))


def test_keep_all_is_accepted_and_keep_none_is_not() -> None:
    assert check_gpu_rules(early_stopping(keep_checkpoint=-1))
    assert "is 0" in refusal_of(early_stopping(keep_checkpoint=0))


@pytest.mark.parametrize("keep", [6.5, "8", None, True])
def test_a_kept_count_that_is_not_an_integer_is_refused(keep: object) -> None:
    assert "training.keep_checkpoint" in refusal_of(
        early_stopping(keep_checkpoint=keep)
    )


def test_an_unset_keep_checkpoint_is_eoles_keep_all_and_accepted() -> None:
    cfg = early_stopping()
    del cfg["training"]["keep_checkpoint"]

    assert check_gpu_rules(cfg)


def test_the_kept_count_is_not_checked_when_saves_do_not_follow_every_validation() -> (
    None
):
    # A validation every 5000 steps but a save every 10000: not the case the rule is for.
    cfg = early_stopping(
        valid_steps=5000, save_checkpoint_steps=10000, keep_checkpoint=1
    )

    assert check_gpu_rules(cfg)


def test_eoles_defaults_of_valid_and_save_steps_differ_so_the_rule_is_off_by_default() -> (
    None
):
    cfg = early_stopping(keep_checkpoint=1)
    del cfg["training"]["valid_steps"]
    del cfg["training"]["save_checkpoint_steps"]

    assert check_gpu_rules(cfg)


def test_one_equal_default_and_one_equal_setting_is_not_a_match() -> None:
    cfg = early_stopping(keep_checkpoint=1, valid_steps=5000)
    del cfg["training"]["save_checkpoint_steps"]

    # Eole's save_checkpoint_steps default is 5000, equal to valid_steps here.
    assert "training.keep_checkpoint" in refusal_of(cfg)


# ---- the order the rules are applied in -----------------------------------------


def test_the_first_broken_rule_is_the_one_reported() -> None:
    cfg = gpu_config(2, gpu_ranks=[], timeout=60, early_stopping=2, keep_checkpoint=0)

    assert "gpu_ranks: is empty" in refusal_of(cfg)


def test_a_config_without_a_training_section_is_refused() -> None:
    assert refusal_of({}) == f"{PREFIX}training: is missing or not a mapping"
