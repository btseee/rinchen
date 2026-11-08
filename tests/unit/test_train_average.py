"""Tests for `rinchen.train.average`: the mean of the last checkpoints of a run.

The step directories are fabricated by `tests.support.step_dirs` with real tiny
safetensors files (bfloat16 and the dtypes numpy cannot write, by hand), so the
completeness check and the averaging run for real. Expected values are computed by
hand, in the test, from the bit patterns or from numbers whose mean is known; the
outputs are read back by `safetensors` or by the test support's own raw reader, never
by the module under test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from rinchen.errors import ConfigError, DataError
from rinchen.train import average
from rinchen.train.average import AveragedModel, average_checkpoints
from rinchen.train.checkpoints import is_complete, latest_complete, step_directories
from tests.support import step_dirs
from tests.support.step_dirs import (
    CONFIG_FILE,
    OPTIMIZER_FILE,
    SPM_ARTIFACT,
    VOCAB_FILE,
    WEIGHTS_FILE,
    bfloat16_bytes,
    make_run,
    make_step_dir,
    read_safetensors_raw,
    safetensors_bytes,
    safetensors_metadata,
)

SIDECAR = "averaged_from.json"


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "ckpt"


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "averaged"


def _f32(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _runs_of(
    model_path: Path, tensors_by_step: dict[int, dict[str, np.ndarray]]
) -> list[Path]:
    return [
        make_step_dir(model_path, step, tensors=tensors)
        for step, tensors in tensors_by_step.items()
    ]


def _bf16_run(model_path: Path, *per_step: tuple[int, ...]) -> list[Path]:
    """Steps 100, 200, ... each holding one bfloat16 tensor `w` of the given patterns."""
    return [
        make_step_dir(
            model_path,
            100 * (index + 1),
            weights=safetensors_bytes({"w": ("BF16", (len(p),), bfloat16_bytes(*p))}),
        )
        for index, p in enumerate(per_step)
    ]


def _bf16_mean_of(model_path: Path, out_dir: Path, *per_step: tuple[int, ...]) -> bytes:
    average_checkpoints(_bf16_run(model_path, *per_step), out_dir)
    dtype, _shape, data = read_safetensors_raw(out_dir / WEIGHTS_FILE)["w"]
    assert dtype == "BF16"
    return data


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leftovers(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.name.endswith(".part"))


# ---- the mean -------------------------------------------------------------------


def test_the_mean_of_three_float32_checkpoints_is_exact_for_hand_computed_values(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300))

    average_checkpoints(dirs, out_dir)

    got = load_file(out_dir / WEIGHTS_FILE)
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert sorted(got) == [
        "decoder.layer.0.weight",
        "encoder.layer.0.weight",
        "generator.bias",
    ]
    np.testing.assert_array_equal(got["encoder.layer.0.weight"], base + 200)
    np.testing.assert_array_equal(got["decoder.layer.0.weight"], base * 2 - 200)
    np.testing.assert_array_equal(got["generator.bias"], np.full(4, 200, np.float32))
    assert all(tensor.dtype == np.float32 for tensor in got.values())


def test_the_sum_is_accumulated_in_float64_not_in_the_dtype_of_the_tensors(
    model_path: Path, out_dir: Path
) -> None:
    # In float32, 1e8 + 1 is 1e8 (the spacing there is 8), so the sum would be 0.
    dirs = _runs_of(
        model_path,
        {100: {"w": _f32(1e8)}, 200: {"w": _f32(1.0)}, 300: {"w": _f32(-1e8)}},
    )

    average_checkpoints(dirs, out_dir)

    assert load_file(out_dir / WEIGHTS_FILE)["w"].tolist() == [float(np.float32(1 / 3))]


def test_two_float16_checkpoints_are_averaged_and_rounded_to_nearest_even(
    model_path: Path, out_dir: Path
) -> None:
    half = np.float16
    # 1.0 and 1.0009765625 (one step above 1.0) meet exactly between two float16
    # numbers: the tie goes to the even one, 1.0; the second pair's tie goes up to
    # 1.001953125, the even one of its two neighbours.
    dirs = _runs_of(
        model_path,
        {
            100: {"w": np.array([1.0, 1.0009765625, 2.0], half)},
            200: {"w": np.array([1.0009765625, 1.001953125, 4.0], half)},
        },
    )

    average_checkpoints(dirs, out_dir)

    got = load_file(out_dir / WEIGHTS_FILE)["w"]
    assert got.dtype == np.float16
    assert got.tolist() == [1.0, 1.001953125, 3.0]


def test_float64_checkpoints_keep_their_dtype_and_average_exactly(
    model_path: Path, out_dir: Path
) -> None:
    dirs = _runs_of(
        model_path,
        {100: {"w": np.array([0.1, 1e-300])}, 200: {"w": np.array([0.3, 3e-300])}},
    )

    average_checkpoints(dirs, out_dir)

    got = load_file(out_dir / WEIGHTS_FILE)["w"]
    assert got.dtype == np.float64
    assert got.tolist() == [(0.1 + 0.3) / 2, (1e-300 + 3e-300) / 2]


def test_bfloat16_ties_go_to_the_even_neighbour(
    model_path: Path, out_dir: Path
) -> None:
    # 0x3F80 is 1.0, 0x3F81 1.0078125, 0x3F82 1.015625, 0x4000 2.0.
    # (1.0 + 1.0078125) / 2 sits between 0x3F80 and 0x3F81 -> 0x3F80 (even);
    # (1.0078125 + 1.015625) / 2 between 0x3F81 and 0x3F82 -> 0x3F82 (even);
    # (1.0 + 2.0) / 2 = 1.5 = 0x3FC0 exactly.
    data = _bf16_mean_of(
        model_path, out_dir, (0x3F80, 0x3F81, 0x3F80), (0x3F81, 0x3F82, 0x4000)
    )

    assert data == bfloat16_bytes(0x3F80, 0x3F82, 0x3FC0)


def test_bfloat16_means_that_are_not_ties_round_to_the_nearer_neighbour(
    model_path: Path, out_dir: Path
) -> None:
    # (1 + 1 + 1.0078125) / 3 = 1.0026..., nearer 1.0 than 1.0078125;
    # (1 + 1.0078125 + 1.0078125) / 3 = 1.0052..., nearer 1.0078125;
    # (1 + 2 + 3) / 3 = 2 exactly.
    data = _bf16_mean_of(
        model_path,
        out_dir,
        (0x3F80, 0x3F80, 0x3F80),
        (0x3F80, 0x3F81, 0x4000),
        (0x3F81, 0x3F81, 0x4040),
    )

    assert data == bfloat16_bytes(0x3F80, 0x3F81, 0x4000)


def test_bfloat16_means_of_negative_values_round_symmetrically(
    model_path: Path, out_dir: Path
) -> None:
    negative = 0x8000
    data = _bf16_mean_of(
        model_path,
        out_dir,
        (negative | 0x3F80, negative | 0x3F81),
        (negative | 0x3F81, negative | 0x3F82),
    )

    assert data == bfloat16_bytes(negative | 0x3F80, negative | 0x3F82)


def test_bfloat16_output_is_loadable_by_torch_and_matches_its_own_rounding(
    model_path: Path, out_dir: Path
) -> None:
    import torch
    from safetensors.torch import load_file as torch_load
    from safetensors.torch import save as torch_save

    generator = torch.Generator().manual_seed(7)
    inputs = [torch.randn(4, 25, generator=generator).bfloat16() for _ in range(3)]
    for index, tensor in enumerate(inputs):
        make_step_dir(
            model_path,
            100 * (index + 1),
            weights=torch_save({"w": tensor.contiguous()}),
        )

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    got = torch_load(str(out_dir / WEIGHTS_FILE))["w"]
    expected = torch.stack([t.double() for t in inputs]).mean(0).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_each_tensor_keeps_the_dtype_it_had_in_the_checkpoints(
    model_path: Path, out_dir: Path
) -> None:
    def weights(step: int) -> bytes:
        return safetensors_bytes(
            {
                "a.f16": ("F16", (1,), np.array([step], np.float16).tobytes()),
                "b.bf16": ("BF16", (1,), bfloat16_bytes(0x3F80)),
                "c.f32": ("F32", (1,), np.array([step], np.float32).tobytes()),
                "d.f64": ("F64", (1,), np.array([step], np.float64).tobytes()),
            }
        )

    for step in (100, 200):
        make_step_dir(model_path, step, weights=weights(step))

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    raw = read_safetensors_raw(out_dir / WEIGHTS_FILE)
    assert {name: dtype for name, (dtype, _, _) in raw.items()} == {
        "a.f16": "F16",
        "b.bf16": "BF16",
        "c.f32": "F32",
        "d.f64": "F64",
    }
    assert raw["c.f32"][2] == np.array([150], np.float32).tobytes()


def test_integer_tensors_are_taken_from_the_last_directory(
    model_path: Path, out_dir: Path
) -> None:
    for step, values in ((100, [1, 2, 3]), (200, [4, 5, 6]), (300, [7, 8, 9])):
        make_step_dir(
            model_path,
            step,
            tensors={"ids": np.array(values, np.int64), "w": _f32(step)},
        )

    result = average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    got = load_file(out_dir / WEIGHTS_FILE)
    assert got["ids"].tolist() == [7, 8, 9]
    assert got["ids"].dtype == np.int64
    assert got["w"].tolist() == [200.0]
    assert (result.tensor_count, result.averaged_count) == (2, 1)


def test_a_float_tensor_equal_bit_for_bit_in_every_directory_is_taken_unchanged(
    model_path: Path, out_dir: Path
) -> None:
    # The positional-encoding buffers are such tensors. A float64 sum of three
    # negative zeros starts from +0.0 and would turn them into +0.0.
    buffer = _f32(-0.0, 1.5, 0.1)
    for step in (100, 200, 300):
        make_step_dir(model_path, step, tensors={"pe": buffer, "w": _f32(step)})

    result = average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    got = load_file(out_dir / WEIGHTS_FILE)
    assert got["pe"].tobytes() == buffer.tobytes()
    assert np.signbit(got["pe"][0])
    assert (result.tensor_count, result.averaged_count) == (2, 1)


def test_a_tensor_that_differs_in_one_bit_only_is_averaged(
    model_path: Path, out_dir: Path
) -> None:
    low = np.array([1.0], np.float32)
    high = np.nextafter(low, np.float32(2))
    make_step_dir(model_path, 100, tensors={"w": low})
    make_step_dir(model_path, 200, tensors={"w": high})

    result = average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    # The mean of two neighbours is the tie; it goes to the even one, 1.0 (the first).
    assert load_file(out_dir / WEIGHTS_FILE)["w"].tobytes() == low.tobytes()
    assert result.averaged_count == 1


def test_a_tensor_with_no_element_and_a_scalar_are_handled(
    model_path: Path, out_dir: Path
) -> None:
    def weights(value: float) -> bytes:
        return safetensors_bytes(
            {
                "empty": ("F32", (0, 3), b""),
                "scalar": ("F32", (), np.array(value, np.float32).tobytes()),
            }
        )

    make_step_dir(model_path, 100, weights=weights(1.0))
    make_step_dir(model_path, 200, weights=weights(3.0))

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    raw = read_safetensors_raw(out_dir / WEIGHTS_FILE)
    assert raw["empty"] == ("F32", (0, 3), b"")
    assert raw["scalar"] == ("F32", (), np.array(2.0, np.float32).tobytes())


# ---- the output -----------------------------------------------------------------


def test_the_output_holds_the_weights_the_json_and_the_files_of_the_last_directory(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300))

    average_checkpoints(dirs, out_dir)

    assert sorted(p.name for p in out_dir.iterdir()) == [
        SIDECAR,
        CONFIG_FILE,
        WEIGHTS_FILE,
        SPM_ARTIFACT,
        VOCAB_FILE,
    ]
    for name in (CONFIG_FILE, VOCAB_FILE, SPM_ARTIFACT):
        assert (out_dir / name).read_bytes() == (dirs[-1] / name).read_bytes()
    assert (out_dir / CONFIG_FILE).read_bytes() != (dirs[0] / CONFIG_FILE).read_bytes()


def test_the_output_has_no_optimizer_state(model_path: Path, out_dir: Path) -> None:
    average_checkpoints(make_run(model_path, (100, 200)), out_dir)

    assert not (out_dir / OPTIMIZER_FILE).exists()


def test_files_beside_the_weights_are_copied_from_the_last_directory_only(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300))
    (dirs[0] / "only_in_the_first.txt").write_text("first", encoding="utf-8")
    (dirs[-1] / "extra.sentencepiece.model").write_bytes(b"another artifact")

    result = average_checkpoints(dirs, out_dir)

    assert not (out_dir / "only_in_the_first.txt").exists()
    assert (out_dir / "extra.sentencepiece.model").read_bytes() == b"another artifact"
    assert result.copied_files == (
        CONFIG_FILE,
        "extra.sentencepiece.model",
        SPM_ARTIFACT,
        VOCAB_FILE,
    )


def test_what_is_not_a_regular_file_in_the_last_directory_is_left_out_and_reported(
    model_path: Path, out_dir: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    (dirs[-1] / "logs").mkdir()
    (dirs[-1] / "logs" / "inner.txt").write_text("x", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("outside the checkpoint", encoding="utf-8")
    (dirs[-1] / "link.txt").symlink_to(elsewhere)

    result = average_checkpoints(dirs, out_dir)

    assert result.skipped_entries == ("link.txt", "logs")
    assert not (out_dir / "logs").exists()
    assert not (out_dir / "link.txt").exists()
    assert json.loads((out_dir / SIDECAR).read_text(encoding="utf-8"))[
        "skipped_entries"
    ] == ["link.txt", "logs"]


def test_the_metadata_of_the_last_weights_file_is_carried_over(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(model_path, 100, tensors={"w": _f32(1)})
    make_step_dir(model_path, 200, tensors={"w": _f32(3)})
    assert safetensors_metadata(model_path / "step_200" / WEIGHTS_FILE) == {
        "format": "np"
    }

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert safetensors_metadata(out_dir / WEIGHTS_FILE) == {"format": "np"}


def test_a_weights_file_without_metadata_gives_an_output_without_metadata(
    model_path: Path, out_dir: Path
) -> None:
    for step in (100, 200):
        data = np.array([step], np.float32).tobytes()
        make_step_dir(
            model_path, step, weights=safetensors_bytes({"w": ("F32", (1,), data)})
        )

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert safetensors_metadata(out_dir / WEIGHTS_FILE) is None


def test_the_json_names_the_sources_their_hashes_and_the_output_hash(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300))
    source_hashes = [_sha256(d / WEIGHTS_FILE) for d in dirs]

    result = average_checkpoints(dirs, out_dir)

    output_hash = _sha256(out_dir / WEIGHTS_FILE)
    document = json.loads((out_dir / SIDECAR).read_text(encoding="utf-8"))
    assert document["sources"] == [
        {"path": str(d), "step": step, "sha256": digest}
        for d, step, digest in zip(dirs, (100, 200, 300), source_hashes, strict=True)
    ]
    assert document["weights"] == {"file": WEIGHTS_FILE, "sha256": output_hash}
    assert document["tensor_count"] == 3
    assert document["tensors_averaged"] == 3
    assert document["copied_files"] == [CONFIG_FILE, SPM_ARTIFACT, VOCAB_FILE]
    assert "plain mean" in document["note"]
    assert "optimizer" in document["note"]
    assert document["schema_version"] == 1
    assert set(document) == {
        "schema_version",
        "note",
        "sources",
        "weights",
        "tensor_count",
        "tensors_averaged",
        "copied_files",
        "skipped_entries",
    }
    assert result == AveragedModel(
        directory=out_dir,
        weights_sha256=output_hash,
        sources=tuple(zip(dirs, source_hashes, strict=True)),
        tensor_count=3,
        averaged_count=3,
        copied_files=(CONFIG_FILE, SPM_ARTIFACT, VOCAB_FILE),
        skipped_entries=(),
    )


def test_the_inputs_are_sorted_by_step_whatever_the_order_given(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300, 1000))
    in_order = average_checkpoints(dirs, tmp_path / "ordered")

    shuffled = average_checkpoints(
        [dirs[3], dirs[0], dirs[2], dirs[1]], tmp_path / "shuffled"
    )

    # Sorted by number, not by name: step_1000 sorts before step_200 as text.
    assert [path for path, _ in shuffled.sources] == dirs
    assert _snapshot(tmp_path / "shuffled") == _snapshot(tmp_path / "ordered")
    assert shuffled.weights_sha256 == in_order.weights_sha256
    # The last directory, the one 1000, supplies the config.
    assert (tmp_path / "shuffled" / CONFIG_FILE).read_bytes() == (
        dirs[3] / CONFIG_FILE
    ).read_bytes()


def test_the_same_inputs_give_the_same_bytes_in_every_file(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200, 300))

    average_checkpoints(dirs, tmp_path / "first")
    average_checkpoints(dirs, tmp_path / "second")

    assert _snapshot(tmp_path / "first") == _snapshot(tmp_path / "second")


def test_the_tensors_are_written_in_the_order_of_their_names_and_aligned(
    model_path: Path, out_dir: Path
) -> None:
    # The inputs list their tensors in the order z, a, m, b, so an output that keeps
    # the order of its input is not in the order of the names.
    names = ["z.last", "a.first", "m.middle", "b.second"]
    for step in (100, 200):
        data = np.array([step, step + 1], np.float32).tobytes()
        make_step_dir(
            model_path,
            step,
            weights=safetensors_bytes({name: ("F32", (2,), data) for name in names}),
        )

    average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    weights = out_dir / WEIGHTS_FILE
    assert list(read_safetensors_raw(weights)) == sorted(names)
    # The header is padded so that the data starts at a multiple of eight bytes.
    assert int.from_bytes(weights.read_bytes()[:8], "little") % 8 == 0


def test_the_result_names_the_output_directory_and_counts_the_tensors(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))

    result = average_checkpoints(dirs, out_dir)

    assert result.directory == out_dir
    assert result.tensor_count == 3
    assert all(isinstance(path, Path) for path, _ in result.sources)


def test_a_relative_input_is_recorded_by_its_absolute_path(
    model_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = make_run(model_path, (100, 200))
    monkeypatch.chdir(model_path.parent)

    result = average_checkpoints(
        [Path("ckpt/step_100"), Path("ckpt/step_200")], out_dir
    )

    assert [path for path, _ in result.sources] == dirs


def test_the_inputs_are_not_modified(model_path: Path, out_dir: Path) -> None:
    dirs = make_run(model_path, (100, 200, 300))
    before = _snapshot(model_path)

    average_checkpoints(dirs, out_dir)

    assert _snapshot(model_path) == before


def test_the_output_can_be_written_inside_the_checkpoint_directory_of_the_run(
    model_path: Path,
) -> None:
    dirs = make_run(model_path, (100, 200))

    average_checkpoints(dirs, model_path / "averaged")

    assert (model_path / "averaged" / WEIGHTS_FILE).is_file()


# ---- an averaged directory is not a checkpoint to resume ------------------------


def test_an_averaged_directory_lacks_the_optimizer_state_and_nothing_else(
    model_path: Path, out_dir: Path, tmp_path: Path
) -> None:
    average_checkpoints(make_run(model_path, (100, 200, 300)), out_dir)

    assert is_complete(out_dir) == f"{OPTIMIZER_FILE}: is missing"
    # With an optimizer file added the completeness check passes: the config, the
    # vocabulary and the weights are those of a checkpoint Eole could load.
    completed = tmp_path / "completed"
    shutil.copytree(out_dir, completed)
    shutil.copyfile(
        model_path / "step_300" / OPTIMIZER_FILE, completed / OPTIMIZER_FILE
    )
    assert is_complete(completed) is None


def test_an_averaged_directory_inside_the_checkpoint_directory_is_never_a_step(
    model_path: Path,
) -> None:
    dirs = make_run(model_path, (100, 200, 300))
    average_checkpoints(dirs, model_path / "averaged")

    listed = step_directories(model_path)
    point = latest_complete(model_path)

    assert [d.step for d in listed] == [100, 200, 300]
    assert point.directory is not None
    assert point.directory.step == 300
    assert point.incomplete_newer == ()


def test_an_averaged_directory_beside_the_checkpoint_directory_is_never_a_step(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    average_checkpoints(dirs, out_dir)

    assert [d.step for d in step_directories(model_path)] == [100, 200]
    assert step_directories(out_dir) == []


# ---- what is refused ------------------------------------------------------------


def test_fewer_than_two_directories_are_refused(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100,))

    with pytest.raises(ConfigError, match=r"at least two.*got 1"):
        average_checkpoints(dirs, out_dir)
    with pytest.raises(ConfigError, match=r"at least two.*got 0"):
        average_checkpoints([], out_dir)

    assert not out_dir.exists()


def test_the_same_directory_twice_is_refused(model_path: Path, out_dir: Path) -> None:
    dirs = make_run(model_path, (100, 200))

    with pytest.raises(ConfigError, match=r"step 100.*twice"):
        average_checkpoints([dirs[0], dirs[1], dirs[0]], out_dir)

    assert not out_dir.exists()


def test_the_same_directory_spelled_two_ways_is_refused(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    spelled_otherwise = model_path / ".." / "ckpt" / "step_100"

    with pytest.raises(ConfigError, match=r"step 100.*twice"):
        average_checkpoints([dirs[0], dirs[1], spelled_otherwise], out_dir)


def test_two_directories_of_one_step_number_from_different_runs_are_refused(
    tmp_path: Path, out_dir: Path
) -> None:
    first = make_step_dir(tmp_path / "run_a" / "ckpt", 100)
    second = make_step_dir(tmp_path / "run_b" / "ckpt", 100)

    with pytest.raises(ConfigError, match=r"step 100.*twice"):
        average_checkpoints([first, second], out_dir)


def test_checkpoints_of_different_runs_are_refused(
    tmp_path: Path, out_dir: Path
) -> None:
    first = make_step_dir(tmp_path / "run_a" / "ckpt", 100)
    second = make_step_dir(tmp_path / "run_b" / "ckpt", 200)

    with pytest.raises(ConfigError, match=r"not of one run.*run_a.*run_b"):
        average_checkpoints([first, second], out_dir)

    assert not out_dir.exists()


@pytest.mark.parametrize("name", ["averaged", "step_x", "step_0", "step_007", "100"])
def test_a_directory_not_named_after_a_step_is_refused(
    model_path: Path, out_dir: Path, name: str
) -> None:
    dirs = make_run(model_path, (100, 200))
    shutil.copytree(dirs[1], model_path / name)

    with pytest.raises(ConfigError, match=rf"{name}.*step_<N>"):
        average_checkpoints([dirs[0], model_path / name], out_dir)


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: step_dirs.truncate(d, WEIGHTS_FILE, 100),
        lambda d: step_dirs.truncate(d, WEIGHTS_FILE, 0),
        lambda d: step_dirs.delete(d, WEIGHTS_FILE),
        lambda d: step_dirs.delete(d, OPTIMIZER_FILE),
        lambda d: step_dirs.delete(d, CONFIG_FILE),
        lambda d: step_dirs.overwrite(d, VOCAB_FILE, b"{not json"),
    ],
)
def test_an_incomplete_directory_is_refused_by_name_with_its_problem(
    model_path: Path, out_dir: Path, damage: object
) -> None:
    dirs = make_run(model_path, (100, 200, 300))
    damage(dirs[1])  # type: ignore[operator]
    problem = is_complete(dirs[1])
    assert problem is not None

    with pytest.raises(DataError) as raised:
        average_checkpoints(dirs, out_dir)

    assert str(dirs[1]) in str(raised.value)
    assert problem in str(raised.value)
    assert not out_dir.exists()
    assert _leftovers(out_dir.parent) == []


def test_a_directory_that_does_not_exist_is_refused_as_a_missing_one(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))

    with pytest.raises(DataError, match=r"step_300: does not exist"):
        average_checkpoints([*dirs, model_path / "step_300"], out_dir)


def test_a_tensor_missing_from_one_directory_is_named(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(model_path, 100, tensors={"a": _f32(1), "b": _f32(1)})
    make_step_dir(model_path, 200, tensors={"a": _f32(2)})

    with pytest.raises(DataError) as raised:
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    message = str(raised.value)
    assert "step_200" in message
    assert "step_100" in message
    assert "'b'" in message
    assert not out_dir.exists()


def test_a_tensor_added_in_one_directory_is_named(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(model_path, 100, tensors={"a": _f32(1)})
    make_step_dir(model_path, 200, tensors={"a": _f32(2), "extra.weight": _f32(2)})

    with pytest.raises(DataError, match=r"step_200.*'extra\.weight'"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)


def test_a_tensor_of_another_shape_is_named_with_both_shapes(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(model_path, 100, tensors={"a": _f32(1, 2), "w": _f32(1, 2, 3, 4)})
    make_step_dir(
        model_path, 200, tensors={"a": _f32(1, 2), "w": _f32(1, 2, 3, 4)[:3].copy()}
    )

    with pytest.raises(
        DataError, match=r"tensor 'w'.*\(4,\).*step_100.*\(3,\).*step_200"
    ):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert not out_dir.exists()


def test_a_tensor_of_another_dtype_is_named_with_both_dtypes(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(model_path, 100, tensors={"w": _f32(1, 2)})
    make_step_dir(model_path, 200, tensors={"w": np.array([1, 2], np.float16)})

    with pytest.raises(DataError, match=r"tensor 'w'.*F32.*step_100.*F16.*step_200"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)


def test_float32_and_bfloat16_of_one_name_are_not_the_same_dtype(
    model_path: Path, out_dir: Path
) -> None:
    make_step_dir(
        model_path,
        100,
        weights=safetensors_bytes({"w": ("F32", (1,), _f32(1).tobytes())}),
    )
    make_step_dir(
        model_path,
        200,
        weights=safetensors_bytes({"w": ("BF16", (1,), bfloat16_bytes(0x3F80))}),
    )

    with pytest.raises(DataError, match=r"tensor 'w'.*F32.*BF16"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_nan_or_infinity_in_any_input_is_refused_naming_tensor_and_directory(
    model_path: Path, out_dir: Path, bad: float, position: int
) -> None:
    for index, step in enumerate((100, 200, 300)):
        values = _f32(1, bad, 3) if index == position else _f32(1, 2, 3)
        make_step_dir(model_path, step, tensors={"ok": _f32(1), "enc.w": values})

    with pytest.raises(DataError) as raised:
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    message = str(raised.value)
    assert "enc.w" in message
    assert str(model_path / f"step_{100 * (position + 1)}") in message
    assert not out_dir.exists()
    assert _leftovers(out_dir.parent) == []


def test_a_nan_in_a_tensor_that_is_equal_in_every_directory_is_refused_too(
    model_path: Path, out_dir: Path
) -> None:
    for step in (100, 200):
        make_step_dir(model_path, step, tensors={"pe": _f32(1, np.nan)})

    with pytest.raises(DataError, match=r"'pe'.*NaN or infinity"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)


@pytest.mark.parametrize("pattern", [0x7FC0, 0x7F80, 0xFF80])
def test_a_bfloat16_nan_or_infinity_is_refused(
    model_path: Path, out_dir: Path, pattern: int
) -> None:
    for step, bits in ((100, 0x3F80), (200, pattern)):
        make_step_dir(
            model_path,
            step,
            weights=safetensors_bytes({"w": ("BF16", (1,), bfloat16_bytes(bits))}),
        )

    with pytest.raises(DataError) as raised:
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert "'w'" in str(raised.value)
    assert str(model_path / "step_200") in str(raised.value)


def test_a_float16_infinity_is_refused(model_path: Path, out_dir: Path) -> None:
    make_step_dir(model_path, 100, tensors={"w": np.array([1.0], np.float16)})
    make_step_dir(model_path, 200, tensors={"w": np.array([np.inf], np.float16)})

    with pytest.raises(DataError, match=r"'w'"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)


def test_a_float_dtype_it_cannot_average_is_refused_by_name(
    model_path: Path, out_dir: Path
) -> None:
    for step in (100, 200):
        make_step_dir(
            model_path,
            step,
            weights=safetensors_bytes(
                {"q.w": ("F8_E4M3", (2,), bytes([step % 256] * 2))}
            ),
        )

    with pytest.raises(DataError, match=r"'q\.w'.*F8_E4M3"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert not out_dir.exists()


def test_an_existing_output_directory_is_never_overwritten(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    out_dir.mkdir()
    (out_dir / "mine.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"exists.*never overwrites"):
        average_checkpoints(dirs, out_dir)

    assert _snapshot(out_dir) == {"mine.txt": b"keep me"}


def test_an_existing_file_or_dangling_link_as_the_output_is_refused(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    (tmp_path / "a-file").write_text("x", encoding="utf-8")
    (tmp_path / "a-link").symlink_to(tmp_path / "nowhere")

    with pytest.raises(ConfigError, match="exists"):
        average_checkpoints(dirs, tmp_path / "a-file")
    with pytest.raises(ConfigError, match="exists"):
        average_checkpoints(dirs, tmp_path / "a-link")

    assert (tmp_path / "a-file").read_text(encoding="utf-8") == "x"


def test_an_output_directory_inside_an_input_is_refused(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))

    with pytest.raises(ConfigError, match=r"inside the checkpoint.*step_200"):
        average_checkpoints(dirs, dirs[1] / "averaged")
    with pytest.raises(ConfigError, match=r"inside the checkpoint.*step_100"):
        average_checkpoints(dirs, model_path / "step_100" / "deeper" / "averaged")

    assert not (dirs[1] / "averaged").exists()
    assert not (model_path / "step_100" / "deeper").exists()


def test_an_output_directory_reached_through_a_link_into_an_input_is_refused(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    (tmp_path / "shortcut").symlink_to(dirs[1])

    with pytest.raises(ConfigError, match="inside the checkpoint"):
        average_checkpoints(dirs, tmp_path / "shortcut" / "averaged")


def test_an_output_directory_whose_parent_does_not_exist_is_refused(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    out = tmp_path / "not" / "made" / "averaged"

    with pytest.raises(ConfigError, match=r"parent.*does not exist"):
        average_checkpoints(dirs, out)

    assert not (tmp_path / "not").exists()


def test_an_output_directory_whose_parent_is_a_file_is_refused(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    (tmp_path / "plain").write_text("x", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"parent.*not a directory"):
        average_checkpoints(dirs, tmp_path / "plain" / "averaged")


@pytest.mark.parametrize("name", ["step_300", "step_5"])
def test_an_output_named_like_a_step_beside_the_inputs_is_refused(
    model_path: Path, name: str
) -> None:
    dirs = make_run(model_path, (100, 200))

    with pytest.raises(ConfigError, match=r"named like a step directory"):
        average_checkpoints(dirs, model_path / name)

    assert sorted(d.name for d in model_path.iterdir()) == ["step_100", "step_200"]


def test_an_output_named_like_a_step_elsewhere_is_allowed(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))

    average_checkpoints(dirs, tmp_path / "step_300")

    assert (tmp_path / "step_300" / WEIGHTS_FILE).is_file()


def test_a_refusal_leaves_the_output_parent_as_it_was(
    model_path: Path, tmp_path: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(ConfigError):
        average_checkpoints(dirs[:1], tmp_path / "averaged")

    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---- a failure leaves nothing ---------------------------------------------------


def test_a_failure_while_copying_a_file_leaves_no_output_and_no_part_directory(
    model_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = make_run(model_path, (100, 200))
    calls: list[str] = []

    def failing_copy(source: Path, target: Path) -> None:
        calls.append(Path(source).name)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(average.shutil, "copyfile", failing_copy)

    with pytest.raises(OSError, match="No space left"):
        average_checkpoints(dirs, out_dir)

    assert calls, "the failure was not reached"
    assert not out_dir.exists()
    assert _leftovers(out_dir.parent) == []


def test_an_interrupt_while_copying_a_file_leaves_nothing_behind_and_propagates(
    model_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = make_run(model_path, (100, 200))

    def interrupted(source: Path, target: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(average.shutil, "copyfile", interrupted)

    with pytest.raises(KeyboardInterrupt):
        average_checkpoints(dirs, out_dir)

    assert not out_dir.exists()
    assert _leftovers(out_dir.parent) == []


def test_a_refused_nan_leaves_no_part_directory_even_after_other_tensors_were_written(
    model_path: Path, out_dir: Path
) -> None:
    # "a" sorts first and is written before "z" is found to hold a NaN.
    for step, z in ((100, _f32(1)), (200, _f32(np.nan))):
        make_step_dir(model_path, step, tensors={"a": _f32(step), "z": z})

    with pytest.raises(DataError, match="'z'"):
        average_checkpoints(sorted(model_path.glob("step_*")), out_dir)

    assert sorted(p.name for p in out_dir.parent.iterdir()) == ["ckpt"]


# ---- the module ------------------------------------------------------------------


def test_averaging_loads_neither_torch_nor_eole(
    model_path: Path, out_dir: Path
) -> None:
    dirs = make_run(model_path, (100, 200))
    code = (
        "import sys; from pathlib import Path; "
        "from rinchen.train.average import average_checkpoints; "
        "before = sorted(m for m in ('torch', 'eole', 'safetensors') if m in sys.modules); "
        f"average_checkpoints([Path(p) for p in {[str(d) for d in dirs]!r}], "
        f"Path({str(out_dir)!r})); "
        "after = sorted(m for m in ('torch', 'eole') if m in sys.modules); "
        "print(before, after)"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert done.stdout.strip() == "[] []"
    assert (out_dir / WEIGHTS_FILE).is_file()
