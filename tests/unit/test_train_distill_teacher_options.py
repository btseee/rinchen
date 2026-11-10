"""The options of the teacher and the source tokens of both tag placements (D10)."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rinchen.config import serve as serve_config
from rinchen.errors import ConfigError
from rinchen.infer import engine_types
from rinchen.train import distill_teacher
from rinchen.train.distill_teacher import TeacherOptions, ct2_source_tokens
from tests.support.distill_teachers import SP, TAG

# -- the source tokens of the two placements


def test_a_source_tag_goes_in_front_of_the_pieces_and_nothing_is_forced():
    assert ct2_source_tokens("src", TAG, [SP + "a", SP + "b"]) == (
        [TAG, SP + "a", SP + "b"],
        None,
    )


def test_a_target_tag_is_a_forced_prefix_and_the_source_stays_untagged():
    assert ct2_source_tokens("tgt", TAG, [SP + "a", SP + "b"]) == (
        [SP + "a", SP + "b"],
        [TAG],
    )


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_source_tokens_are_a_new_list_and_the_pieces_are_not_changed(placement):
    pieces = (SP + "a",)

    tokens, _ = ct2_source_tokens(placement, TAG, pieces)

    assert isinstance(tokens, list)
    assert pieces == (SP + "a",)


def test_no_pieces_give_the_bare_tag_for_a_source_tag_and_nothing_for_a_target_tag():
    assert ct2_source_tokens("src", TAG, []) == ([TAG], None)
    assert ct2_source_tokens("tgt", TAG, []) == ([], [TAG])


@pytest.mark.parametrize("placement", ["both", "", "SRC", None])
def test_another_placement_is_a_value_error(placement):
    with pytest.raises(ValueError, match="placement"):
        ct2_source_tokens(placement, TAG, [SP + "a"])


# -- the options


def test_the_defaults_are_the_documented_ones():
    options = TeacherOptions()

    assert (
        options.beam_size,
        options.n,
        options.max_decoding_length,
        options.max_source_pieces,
        options.batch_size,
        options.device,
        options.compute_type,
        options.threads,
    ) == (4, 1, 256, 256, 64, "cpu", "default", 1)


def test_the_options_are_frozen_and_slotted():
    options = TeacherOptions()

    with pytest.raises(FrozenInstanceError):
        options.beam_size = 5  # type: ignore[misc]
    assert not hasattr(options, "__dict__")


@pytest.mark.parametrize(
    "field",
    [
        "beam_size",
        "n",
        "max_decoding_length",
        "max_source_pieces",
        "batch_size",
        "threads",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "4", None])
def test_every_number_must_be_an_integer_of_at_least_one(field, value):
    with pytest.raises(ConfigError, match=f"TeacherOptions.{field}"):
        TeacherOptions(**{field: value})


def test_one_is_the_smallest_value_of_every_number():
    options = TeacherOptions(
        beam_size=1,
        n=1,
        max_decoding_length=1,
        max_source_pieces=1,
        batch_size=1,
        threads=1,
    )

    assert options.beam_size == 1


@pytest.mark.parametrize(("n", "beam_size"), [(5, 4), (2, 1), (4, 3)])
def test_more_hypotheses_than_the_beam_are_refused(n, beam_size):
    with pytest.raises(ConfigError, match=r"TeacherOptions\.n.*TeacherOptions\.beam"):
        TeacherOptions(n=n, beam_size=beam_size)


def test_as_many_hypotheses_as_the_beam_are_accepted():
    assert TeacherOptions(n=4, beam_size=4).n == 4


def test_a_source_of_1023_pieces_is_the_longest_the_library_takes():
    assert TeacherOptions(max_source_pieces=1023).max_source_pieces == 1023

    with pytest.raises(ConfigError, match=r"max_source_pieces.*1023"):
        TeacherOptions(max_source_pieces=1024)


def test_auto_is_refused_because_it_would_choose_the_gpu():
    with pytest.raises(ConfigError, match="auto") as excinfo:
        TeacherOptions(device="auto")  # type: ignore[arg-type]

    assert "GPU" in str(excinfo.value)


@pytest.mark.parametrize("device", ["gpu", "CPU", "", "cuda:0", None, 0])
def test_a_device_is_cpu_or_cuda_and_nothing_else(device):
    with pytest.raises(ConfigError, match="TeacherOptions.device"):
        TeacherOptions(device=device)


def test_cuda_is_an_explicit_choice_that_is_accepted():
    assert TeacherOptions(device="cuda").device == "cuda"


@pytest.mark.parametrize("compute_type", ["", "  ", None, 8])
def test_a_compute_type_is_a_non_empty_string(compute_type):
    with pytest.raises(ConfigError, match="TeacherOptions.compute_type"):
        TeacherOptions(compute_type=compute_type)


@pytest.mark.parametrize("compute_type", ["auto", "AUTO", " auto "])
def test_a_compute_type_of_auto_is_refused_because_it_would_quantise_the_teacher(
    compute_type,
):
    with pytest.raises(ConfigError, match="compute type `auto` is refused"):
        TeacherOptions(compute_type=compute_type)


def test_the_other_compute_types_are_unchanged():
    for compute_type in ("default", "float32", "int8", "int8_float32"):
        assert TeacherOptions(compute_type=compute_type).compute_type == compute_type


def test_the_source_limit_is_one_number_in_the_teacher_the_engine_and_the_service():
    teacher = ast.parse(Path(distill_teacher.__file__).read_text(encoding="utf-8"))
    assigned = {
        target.id
        for node in teacher.body
        if isinstance(node, ast.AnnAssign | ast.Assign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    assert "MAX_SOURCE_PIECES" not in assigned
    assert distill_teacher.MAX_SOURCE_PIECES == engine_types.MAX_SOURCE_PIECES == 1023
    # The service's configuration may not import the engine, so it keeps a copy.
    assert serve_config._MAX_SOURCE_PIECES == engine_types.MAX_SOURCE_PIECES
