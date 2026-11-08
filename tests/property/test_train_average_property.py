"""Property tests for `rinchen.train.average`: the mean of copies, the mean against an
exact one, and the mean of means, over generated tensors."""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from safetensors.numpy import load_file

from rinchen.train.average import average_checkpoints
from tests.support.step_dirs import WEIGHTS_FILE, make_step_dir

_DTYPES = st.sampled_from([np.float16, np.float32, np.float64])


def _finite_arrays(dtype: type, count: int, size: int, bound: float):
    width = np.dtype(dtype).itemsize * 8
    element = st.floats(
        min_value=-bound,
        max_value=bound,
        allow_nan=False,
        allow_infinity=False,
        width=width,  # type: ignore[arg-type]
    )
    return st.lists(
        st.lists(element, min_size=size, max_size=size).map(
            lambda values: np.array(values, dtype=dtype)
        ),
        min_size=count,
        max_size=count,
    )


def _contents(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _average(tensor_lists: list[NDArray], work: Path) -> NDArray:
    """Average `tensor_lists` (one tensor `w` per checkpoint, steps 10, 20, ...)."""
    model_path = work / "ckpt"
    dirs = [
        make_step_dir(model_path, 10 * (index + 1), tensors={"w": tensor})
        for index, tensor in enumerate(tensor_lists)
    ]
    average_checkpoints(dirs, work / "averaged")
    return load_file(work / "averaged" / WEIGHTS_FILE)["w"]


@settings(max_examples=40)
@given(
    dtype=_DTYPES,
    copies=st.integers(2, 5),
    data=st.data(),
)
def test_the_mean_of_copies_of_one_tensor_is_that_tensor(
    dtype: type, copies: int, data: st.DataObject
) -> None:
    tensor = data.draw(_finite_arrays(dtype, 1, 6, 1e3))[0]

    with tempfile.TemporaryDirectory() as work:
        got = _average([tensor] * copies, Path(work))

    assert got.dtype == tensor.dtype
    assert got.tobytes() == tensor.tobytes()


@settings(max_examples=60)
@given(dtype=_DTYPES, count=st.integers(2, 5), data=st.data())
def test_the_mean_is_within_one_step_of_the_exact_mean(
    dtype: type, count: int, data: st.DataObject
) -> None:
    tensors = data.draw(_finite_arrays(dtype, count, 5, 1e3))

    with tempfile.TemporaryDirectory() as work:
        got = _average(tensors, Path(work))

    biggest = max(float(np.abs(t.astype(np.float64)).max()) for t in tensors)
    for index in range(5):
        exact = sum(Fraction(float(t[index])) for t in tensors) / count
        # The result is the exact mean rounded to the dtype, so it is within the
        # spacing of its dtype; the sum in float64 can lose bits when the terms differ
        # by more than 2**53, hence the second term.
        tolerance = Fraction(float(np.spacing(np.abs(got[index])))) + Fraction(
            biggest * 2.0**-52
        )
        assert abs(Fraction(float(got[index])) - exact) <= tolerance


@settings(max_examples=40)
@given(
    arrays=_finite_arrays(np.float64, 4, 4, 1e6),
)
def test_the_mean_of_four_equals_the_mean_of_the_means_of_two_pairs(
    arrays: list[NDArray],
) -> None:
    with tempfile.TemporaryDirectory() as work_name:
        work = Path(work_name)
        whole = _average(arrays, work / "whole")
        first = _average(arrays[:2], work / "pair_a")
        second = _average(arrays[2:], work / "pair_b")
        of_means = _average([first, second], work / "pairs")

    np.testing.assert_allclose(whole, of_means, rtol=1e-12, atol=1e-9)


@settings(max_examples=40)
@given(arrays=_finite_arrays(np.float32, 3, 4, 1e3))
def test_the_output_does_not_depend_on_the_order_the_directories_are_listed_in(
    arrays: list[NDArray],
) -> None:
    with tempfile.TemporaryDirectory() as work_name:
        work = Path(work_name)
        model_path = work / "ckpt"
        dirs = [
            make_step_dir(model_path, 10 * (index + 1), tensors={"w": tensor})
            for index, tensor in enumerate(arrays)
        ]
        average_checkpoints(dirs, work / "forward")
        average_checkpoints(dirs[::-1], work / "backward")

        # Every file: the config comes from the last directory, which is the same
        # one whatever the order, and so does the record of the sources.
        assert _contents(work / "forward") == _contents(work / "backward")
