"""Fabricated Eole checkpoint directories for the tests of D8.

A COMPLETE `step_<N>` directory holds what Eole 0.6.4 writes into one (spike, D8): the
weights as a real safetensors file (written through the `safetensors` package), an
`optimizer.pt` that is a real zip archive (its content is not loadable by torch), a
`config.json` and a `vocab.json` that parse as JSON objects, and the SentencePiece
artifact. The files are tiny; the damage helpers truncate, delete, overwrite or replace
one of them the way an interrupted save or a bad disk would.

Eole also links the files of the newest step into the checkpoint directory itself,
through relative symbolic links (`link_newest_step`); a resume never goes through them.
"""

from __future__ import annotations

import json
import os
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

WEIGHTS_FILE = "model.00.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
CONFIG_FILE = "config.json"
VOCAB_FILE = "vocab.json"
SPM_ARTIFACT = "toy.model"
# A member of `optimizer.pt` whose name is not ASCII (Mongolian "state").
UTF8_MEMBER = "optimizer/төлөв.pkl"

# The four files that make a step directory complete, then the artifact Eole also writes.
REQUIRED_FILES = (WEIGHTS_FILE, OPTIMIZER_FILE, CONFIG_FILE, VOCAB_FILE)
ALL_FILES = (*REQUIRED_FILES, SPM_ARTIFACT)


def default_tensors(step: int) -> dict[str, NDArray[np.float32]]:
    """Three small float32 tensors whose values depend on `step`, so the weights of two
    steps differ."""
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    return {
        "encoder.layer.0.weight": base + np.float32(step),
        "decoder.layer.0.weight": base * np.float32(2) - np.float32(step),
        "generator.bias": np.full(4, step, dtype=np.float32),
    }


def safetensors_bytes(
    entries: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    metadata: Mapping[str, str] | None = None,
) -> bytes:
    """The bytes of a safetensors file holding `entries`, each name mapped to its
    safetensors dtype code (`BF16`, `F8_E4M3`, ...), shape and little-endian data, in
    the order given. Written by hand, for the dtypes numpy cannot write (bfloat16 has
    no numpy dtype)."""
    header: dict[str, object] = {} if metadata is None else {"__metadata__": metadata}
    offset = 0
    for name, (dtype, shape, data) in entries.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    body = b"".join(data for _, _, data in entries.values())
    return struct.pack("<Q", len(text)) + text + body


def read_safetensors_raw(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """The tensors of the safetensors file `path` as (dtype code, shape, little-endian
    data) by name, read without safetensors (so any dtype can be read). The reverse of
    `safetensors_bytes`."""
    content = path.read_bytes()
    (length,) = struct.unpack("<Q", content[:8])
    header = json.loads(content[8 : 8 + length])
    body = content[8 + length :]
    found: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        found[name] = (entry["dtype"], tuple(entry["shape"]), body[start:end])
    assert sum(len(data) for _, _, data in found.values()) == len(body)
    return found


def safetensors_metadata(path: Path) -> dict[str, str] | None:
    """The `__metadata__` of the safetensors file `path`, None when it has none."""
    content = path.read_bytes()
    (length,) = struct.unpack("<Q", content[:8])
    metadata = json.loads(content[8 : 8 + length]).get("__metadata__")
    assert metadata is None or isinstance(metadata, dict)
    return metadata


def bfloat16_bytes(*patterns: int) -> bytes:
    """The data of a bfloat16 tensor from the 16-bit patterns `patterns` (0x3F80 is 1.0,
    0x4000 is 2.0, 0x7FC0 a NaN)."""
    return struct.pack(f"<{len(patterns)}H", *patterns)


def make_step_dir(
    model_path: Path,
    step: int,
    *,
    tensors: Mapping[str, NDArray[Any]] | None = None,
    weights: bytes | None = None,
) -> Path:
    """Create `model_path/step_<step>` as a COMPLETE step directory and return it.

    The weights are `tensors` (default `default_tensors(step)`) written by
    `safetensors.numpy.save_file`, or, when `weights` is given, those bytes as the
    weights file (`safetensors_bytes`). `model_path` is created when missing.
    """
    from safetensors.numpy import save_file

    step_dir = model_path / f"step_{step}"
    step_dir.mkdir(parents=True)
    if weights is not None:
        (step_dir / WEIGHTS_FILE).write_bytes(weights)
    else:
        save_file(
            dict(tensors if tensors is not None else default_tensors(step)),
            str(step_dir / WEIGHTS_FILE),
            metadata={"format": "np"},
        )
    with zipfile.ZipFile(step_dir / OPTIMIZER_FILE, "w") as archive:
        archive.writestr("optimizer/data.pkl", b"not a pickle: torch cannot load this")
        archive.writestr("optimizer/byteorder", "little")
        # A member name that is not ASCII is stored with the UTF-8 flag, so the reader
        # decodes it: damage in the central directory can then raise a decoding error.
        archive.writestr(UTF8_MEMBER, b"x")
    (step_dir / VOCAB_FILE).write_text(
        json.dumps({"src": [["a", 0], ["b", 1]], "tgt": [["a", 0]], "specials": {}}),
        encoding="utf-8",
    )
    (step_dir / SPM_ARTIFACT).write_bytes(b"sentencepiece model of the toy corpus")
    # Eole writes the config last, so it is the file that says the rest was saved.
    (step_dir / CONFIG_FILE).write_text(
        json.dumps({"training": {"model_path": str(model_path)}, "saved_at": step}),
        encoding="utf-8",
    )
    return step_dir


def make_run(model_path: Path, steps: tuple[int, ...]) -> list[Path]:
    """Complete step directories for `steps`, in that order."""
    return [make_step_dir(model_path, step) for step in steps]


def link_newest_step(model_path: Path, step: int) -> None:
    """Link the files of `step_<step>` into `model_path` through relative symbolic
    links, as Eole leaves them after a save."""
    for name in ALL_FILES:
        link = model_path / name
        link.unlink(missing_ok=True)
        link.symlink_to(Path(f"step_{step}") / name)


def lengths_to_truncate_to(size: int) -> list[int]:
    """Where to cut a file of `size` bytes: one byte short of whole, half, 100 bytes
    (the size of the header's length field plus a stub), one byte and none. The spike
    cut a 3,289,000-byte file at 3,288,000, 1,500,000 and 100 bytes."""
    cuts = {size - 1, size // 2, min(100, size - 1), 1, 0}
    return sorted(cut for cut in cuts if 0 <= cut < size)


def truncate(step_dir: Path, name: str, length: int) -> None:
    """Cut `step_dir/name` to its first `length` bytes."""
    os.truncate(step_dir / name, length)


def delete(step_dir: Path, name: str) -> None:
    """Remove `step_dir/name`."""
    (step_dir / name).unlink()


def overwrite(step_dir: Path, name: str, content: bytes) -> None:
    """Replace the content of `step_dir/name` by `content`."""
    (step_dir / name).write_bytes(content)


def replace_with_directory(step_dir: Path, name: str) -> None:
    """Make `step_dir/name` a directory instead of a file."""
    (step_dir / name).unlink()
    (step_dir / name).mkdir()


def replace_with_dangling_symlink(step_dir: Path, name: str) -> None:
    """Make `step_dir/name` a symbolic link to a file that does not exist."""
    (step_dir / name).unlink()
    (step_dir / name).symlink_to("deleted-step/" + name)


def replace_with_symlink_to(step_dir: Path, name: str, target: Path) -> None:
    """Make `step_dir/name` a symbolic link to the existing `target`."""
    (step_dir / name).unlink()
    (step_dir / name).symlink_to(target)
