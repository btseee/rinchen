"""The mean of the last checkpoints of a run, written as one model.

Eole 0.6.4 has no tool for this, and its own moving average (`average_decay`) saves the
average in place of the live weights in every checkpoint (spike, D8), so the average is
made here, from checkpoints that are saved and complete (`checkpoints.is_complete`).

The mean is plain (every checkpoint weighs the same) and taken tensor by tensor: each
floating-point tensor is summed in float64 over the checkpoints, divided by their
number and rounded to nearest, ties to even, in the dtype it had. bfloat16 has no numpy
dtype, so it goes through its bit pattern (a bfloat16 is the upper 16 bits of a float32)
and is rounded in two steps, float64 to float32 to bfloat16. A tensor that is not
floating point, or that is equal bit for bit in every checkpoint (Eole's positional
encoding buffers), is taken from the last checkpoint unchanged. A NaN or an infinity in
any floating-point tensor of any checkpoint is refused: it would poison the mean.

The weights are read tensor by tensor from the files and written the same way, so the
memory in use is a few copies of the largest tensor, not of the model. The safetensors
file is written here, by its format (an 8-byte header length, the JSON header, the
tensors' bytes in the order of their names), because that keeps the bytes a function of
the inputs and the writing streaming; `safetensors` itself is imported where it checks
the result, and `torch` is not imported at all.

The output directory holds the averaged `model.00.safetensors`, the files beside the
weights in the LAST checkpoint (`config.json`, `vocab.json`, the SentencePiece artifact;
not `optimizer.pt`) and `averaged_from.json`. It is for `eole predict` and the
CTranslate2 converter. It is NOT a checkpoint to resume: it has no optimizer state, so
`checkpoints.is_complete` names `optimizer.pt`, and its name is never `step_<N>`.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

import numpy as np
from numpy.typing import NDArray

from rinchen.errors import ConfigError, DataError
from rinchen.manifest.stage import sha256_of
from rinchen.train.checkpoints import OPTIMIZER_FILE, WEIGHTS_FILE, is_complete, step_of
from rinchen.utils.part_file import unique_part_path

AVERAGED_FROM_FILE: Final = "averaged_from.json"
_SCHEMA_VERSION: Final = 1
_NOTE: Final = (
    "A plain mean (every source weighs the same) of the floating-point tensors of the "
    "sources, summed in float64 and rounded to the dtype of each tensor. A tensor that "
    "is not floating point, or is equal bit for bit in every source, is taken from the "
    "last source. There is no optimizer state: this directory is not a checkpoint to "
    "resume."
)
# Never copied from the last checkpoint: the weights are replaced, the optimizer state
# has no meaning for an average, and the record is written here.
_NOT_COPIED: Final = frozenset({WEIGHTS_FILE, OPTIMIZER_FILE, AVERAGED_FROM_FILE})
# safetensors' own limit on the header.
_MAX_HEADER_BYTES: Final = 100 * 1024 * 1024
_ALIGNMENT: Final = 8
# The floating-point dtypes that are averaged, as numpy reads them (little-endian);
# bfloat16 has none and is handled through its bit pattern.
_NUMPY_FLOATS: Final = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_BFLOAT16: Final = "BF16"
_FLOATS: Final = frozenset({*_NUMPY_FLOATS, _BFLOAT16})
# Taken from the last checkpoint as they are.
_NOT_FLOATS: Final = frozenset(
    {"BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"}
)


@dataclass(frozen=True, slots=True)
class AveragedModel:
    """What `average_checkpoints` wrote.

    directory: the output directory.
    weights_sha256: the SHA-256 of its `model.00.safetensors`.
    sources: each averaged checkpoint directory (absolute) and the SHA-256 of its
        weights, oldest step first.
    tensor_count: the tensors in the weights.
    averaged_count: how many of them were averaged (the others were taken from the
        last checkpoint: not floating point, or equal in every checkpoint).
    copied_files: the files copied from the last checkpoint, by name.
    skipped_entries: what the last checkpoint holds beside the weights that is not a
        regular file (a directory, a link), and so was not copied.
    """

    directory: Path
    weights_sha256: str
    sources: tuple[tuple[Path, str], ...]
    tensor_count: int
    averaged_count: int
    copied_files: tuple[str, ...]
    skipped_entries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Tensor:
    """A tensor of a safetensors file: its dtype code, shape, and where its bytes are."""

    dtype: str
    shape: tuple[int, ...]
    start: int
    size: int


@dataclass(frozen=True, slots=True)
class _WeightsHeader:
    tensors: dict[str, _Tensor]
    metadata: dict[str, str] | None


def average_checkpoints(step_dirs: Sequence[Path], out_dir: Path) -> AveragedModel:
    """Average the weights of the checkpoints `step_dirs` into the new directory
    `out_dir` and return what was written.

    The directories are sorted by step number (the `<N>` of their name `step_<N>`),
    whatever order they are given in. They must be complete (`is_complete`), be
    directories of one checkpoint directory, and hold the same tensors with the same
    dtypes and shapes. The files beside the weights in the last one are copied, its
    `config.json` and `vocab.json` among them, and `averaged_from.json` records the
    sources with their SHA-256 and the output's. The inputs are not changed.

    `out_dir` is built beside its final place under a temporary name and renamed, so it
    appears whole or not at all; a failure, an interrupt included, leaves nothing. It
    must not exist, must have an existing parent, must not lie inside an input, and
    must not be named `step_<N>` in the directory the inputs are in (a resume would take
    it for a checkpoint).

    Raises:
        ConfigError: fewer than two directories, one step given twice, a directory not
            named `step_<N>`, directories of different parents, or a refused `out_dir`.
        DataError: an input is not complete, the inputs differ in tensor names, dtypes
            or shapes, a tensor has a dtype that cannot be averaged, or holds a NaN or
            an infinity.
        OSError: a file cannot be read or written (a full disk, say).
    """
    sources = _ordered_sources(step_dirs)
    directories = [path for _, path in sources]
    output = _checked_output(out_dir, directories)
    for directory in directories:
        problem = is_complete(directory)
        if problem is not None:
            raise DataError(f"cannot average {directory}: {problem}")
    source_hashes = [sha256_of(path / WEIGHTS_FILE) for path in directories]
    headers = [_read_header(path / WEIGHTS_FILE) for path in directories]
    _require_one_layout(directories, headers)

    part = unique_part_path(output)
    part.mkdir()
    try:
        written = _write_directory(part, sources, source_hashes, headers)
        os.rename(part, output)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return AveragedModel(
        directory=output,
        weights_sha256=written.weights_sha256,
        sources=tuple(zip(directories, source_hashes, strict=True)),
        tensor_count=len(headers[-1].tensors),
        averaged_count=written.averaged_count,
        copied_files=written.copied_files,
        skipped_entries=written.skipped_entries,
    )


@dataclass(frozen=True, slots=True)
class _Written:
    weights_sha256: str
    averaged_count: int
    copied_files: tuple[str, ...]
    skipped_entries: tuple[str, ...]


def _write_directory(
    part: Path,
    sources: list[tuple[int, Path]],
    source_hashes: list[str],
    headers: list[_WeightsHeader],
) -> _Written:
    directories = [path for _, path in sources]
    weights = part / WEIGHTS_FILE
    averaged = _write_weights(weights, directories, headers)
    _require_readable(weights, sorted(headers[-1].tensors))
    weights_sha256 = sha256_of(weights)
    copied, skipped = _copy_beside_weights(directories[-1], part)
    document = {
        "schema_version": _SCHEMA_VERSION,
        "note": _NOTE,
        "sources": [
            {"path": str(path), "step": step, "sha256": digest}
            for (step, path), digest in zip(sources, source_hashes, strict=True)
        ],
        "weights": {"file": WEIGHTS_FILE, "sha256": weights_sha256},
        "tensor_count": len(headers[-1].tensors),
        "tensors_averaged": averaged,
        "copied_files": list(copied),
        "skipped_entries": list(skipped),
    }
    (part / AVERAGED_FROM_FILE).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _Written(weights_sha256, averaged, copied, skipped)


def _ordered_sources(step_dirs: Sequence[Path]) -> list[tuple[int, Path]]:
    directories = [Path(os.path.abspath(path)) for path in step_dirs]
    if len(directories) < 2:
        raise ConfigError(
            "need at least two checkpoint directories to average, got "
            f"{len(directories)}"
        )
    numbered: list[tuple[int, Path]] = []
    for path in directories:
        step = step_of(path)
        if step is None:
            raise ConfigError(
                f"{path} is not named step_<N> (N a positive number without a "
                "leading zero), so its step is not known"
            )
        numbered.append((step, path))
    numbered.sort()
    for (step, first), (next_step, second) in itertools.pairwise(numbered):
        if step == next_step:
            raise ConfigError(f"step {step} is given twice: {first} and {second}")
    parents = {path.parent.resolve(): path.parent for _, path in numbered}
    if len(parents) > 1:
        first_parent, second_parent, *_ = parents.values()
        raise ConfigError(
            "the checkpoints are not of one run, they lie in different directories: "
            f"{first_parent} and {second_parent}"
        )
    return numbered


def _checked_output(out_dir: Path, inputs: Sequence[Path]) -> Path:
    output = Path(os.path.abspath(out_dir))
    if os.path.lexists(output):
        raise ConfigError(
            f"the output directory {output} exists; averaging never overwrites"
        )
    where = Path(os.path.realpath(output))
    for path in inputs:
        if where.is_relative_to(path.resolve()):
            raise ConfigError(
                f"the output directory {output} lies inside the checkpoint {path}"
            )
    parent = output.parent
    if not parent.exists():
        raise ConfigError(f"the parent of the output directory {output} does not exist")
    if not parent.is_dir():
        raise ConfigError(
            f"the parent of the output directory {output} is not a directory"
        )
    if step_of(output) is not None and parent.resolve() == inputs[0].parent.resolve():
        raise ConfigError(
            f"the output directory {output} is named like a step directory of the "
            "run; a resume would take it for a checkpoint"
        )
    return output


def _read_header(path: Path) -> _WeightsHeader:
    """The tensors and metadata of the safetensors file `path`, read from its header
    (the file has passed `safetensors`' own check: this reads what that checked)."""
    try:
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            if length > _MAX_HEADER_BYTES:
                raise DataError(f"{path}: the safetensors header is {length} bytes")
            header = json.loads(handle.read(length))
        data_start = 8 + length
        tensors = {
            name: _tensor_of(entry, data_start)
            for name, entry in header.items()
            if name != "__metadata__"
        }
        metadata = header.get("__metadata__")
        if metadata is not None and not (
            isinstance(metadata, dict)
            and all(isinstance(v, str) for v in metadata.values())
        ):
            raise ValueError("__metadata__ is not a mapping of text")
    except (struct.error, ValueError, KeyError, TypeError, AttributeError) as error:
        raise DataError(
            f"{path}: cannot read the safetensors header ({error})"
        ) from error
    return _WeightsHeader(tensors, metadata)


def _tensor_of(entry: object, data_start: int) -> _Tensor:
    if not isinstance(entry, dict):
        raise TypeError("a tensor entry is not a mapping")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if not (
        isinstance(dtype, str)
        and isinstance(shape, list)
        and isinstance(offsets, list)
        and len(offsets) == 2
    ):
        raise TypeError("a tensor entry is malformed")
    begin, end = int(offsets[0]), int(offsets[1])
    if not 0 <= begin <= end:
        raise ValueError("a tensor entry has a negative size")
    return _Tensor(
        dtype=dtype,
        shape=tuple(int(extent) for extent in shape),
        start=data_start + begin,
        size=end - begin,
    )


def _require_one_layout(
    directories: Sequence[Path], headers: Sequence[_WeightsHeader]
) -> None:
    first_dir, first = directories[0], headers[0].tensors
    for directory, header in zip(directories[1:], headers[1:], strict=True):
        if header.tensors.keys() != first.keys():
            missing = sorted(first.keys() - header.tensors.keys())
            unexpected = sorted(header.tensors.keys() - first.keys())
            raise DataError(
                f"{directory}: the tensors differ from those of {first_dir}: "
                f"missing {missing}, unexpected {unexpected}"
            )
    for name in sorted(first):
        for directory, header in zip(directories[1:], headers[1:], strict=True):
            tensor = header.tensors[name]
            if tensor.shape != first[name].shape:
                raise DataError(
                    f"tensor {name!r}: shape {first[name].shape} in {first_dir} but "
                    f"{tensor.shape} in {directory}"
                )
            if tensor.dtype != first[name].dtype:
                raise DataError(
                    f"tensor {name!r}: dtype {first[name].dtype} in {first_dir} but "
                    f"{tensor.dtype} in {directory}"
                )
        dtype = first[name].dtype
        if dtype not in _FLOATS and dtype not in _NOT_FLOATS:
            raise DataError(
                f"tensor {name!r}: dtype {dtype} cannot be averaged (in {first_dir})"
            )


def _write_weights(
    target: Path, directories: Sequence[Path], headers: Sequence[_WeightsHeader]
) -> int:
    """Write the averaged tensors to `target`; return how many were averaged."""
    layout = headers[-1].tensors
    names = sorted(layout)
    entries: dict[str, object] = {}
    if headers[-1].metadata is not None:
        entries["__metadata__"] = dict(sorted(headers[-1].metadata.items()))
    position = 0
    for name in names:
        tensor = layout[name]
        entries[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [position, position + tensor.size],
        }
        position += tensor.size
    text = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % _ALIGNMENT)
    averaged = 0
    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context((path / WEIGHTS_FILE).open("rb"))
            for path in directories
        ]
        output = stack.enter_context(target.open("wb"))
        output.write(struct.pack("<Q", len(text)))
        output.write(text)
        for name in names:
            data, was_averaged = _mean_of(name, directories, headers, files)
            output.write(data)
            averaged += was_averaged
    return averaged


def _mean_of(
    name: str,
    directories: Sequence[Path],
    headers: Sequence[_WeightsHeader],
    files: Sequence[BinaryIO],
) -> tuple[bytes, bool]:
    """The bytes of the output tensor `name`, and whether they are a mean."""
    dtype = headers[-1].tensors[name].dtype
    last = _bytes_of(files[-1], headers[-1].tensors[name])
    if dtype not in _FLOATS:
        return last, False
    total = np.zeros(math.prod(headers[-1].tensors[name].shape), dtype=np.float64)
    identical = True
    for directory, header, handle in zip(directories, headers, files, strict=True):
        raw = last if handle is files[-1] else _bytes_of(handle, header.tensors[name])
        values = _as_float64(raw, dtype)
        _require_finite(values, name, directory)
        identical = identical and raw == last
        total += values
    if identical:
        return last, False
    return _from_float64(total / len(files), dtype), True


def _bytes_of(handle: BinaryIO, tensor: _Tensor) -> bytes:
    handle.seek(tensor.start)
    data = handle.read(tensor.size)
    if len(data) != tensor.size:
        raise DataError(f"{handle.name}: the file ends inside a tensor")
    return data


def _as_float64(raw: bytes, dtype: str) -> NDArray[np.float64]:
    if dtype == _BFLOAT16:
        # A bfloat16 is the upper half of a float32.
        upper = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return upper.view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype=_NUMPY_FLOATS[dtype]).astype(np.float64)


def _from_float64(mean: NDArray[np.float64], dtype: str) -> bytes:
    if dtype != _BFLOAT16:
        return mean.astype(_NUMPY_FLOATS[dtype]).tobytes()
    bits = mean.astype(np.float32).view(np.uint32).astype(np.uint64)
    # Round to nearest, ties to even, by adding half of the dropped part (less one) and
    # the lowest kept bit; a finite float32 cannot carry into the sign.
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes()


def _require_finite(values: NDArray[np.float64], name: str, directory: Path) -> None:
    bad = int(np.count_nonzero(~np.isfinite(values)))
    if bad:
        raise DataError(
            f"{directory}: tensor {name!r} holds NaN or infinity ({bad} of "
            f"{values.size} values); an average of it would be NaN or infinite"
        )


def _require_readable(weights: Path, names: list[str]) -> None:
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(weights), framework="numpy") as tensors:
            written = sorted(tensors.keys())
    except SafetensorError as error:
        raise DataError(
            f"the averaged weights {weights} are not a valid safetensors file: {error}"
        ) from error
    if written != names:
        raise DataError(f"the averaged weights {weights} do not hold the tensors asked")


def _copy_beside_weights(
    last: Path, part: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    copied: list[str] = []
    skipped: list[str] = []
    with os.scandir(last) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.name in _NOT_COPIED:
                continue
            if entry.is_file(follow_symlinks=False):
                shutil.copyfile(entry.path, part / entry.name)
                copied.append(entry.name)
            else:
                skipped.append(entry.name)
    return tuple(copied), tuple(skipped)
