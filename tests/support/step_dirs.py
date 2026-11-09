"""Fabricated Eole checkpoint directories for the tests of D8 and D9.

A COMPLETE `step_<N>` directory holds what Eole 0.6.4 writes into one (spike, D8): the
weights as a real safetensors file (written through the `safetensors` package), an
`optimizer.pt` that is a real zip archive (its content is not loadable by torch), a
`config.json` and a `vocab.json` that parse as JSON objects, and the SentencePiece
artifact. The `config.json` carries a `model` section, laid out as Eole saves it
(`eole_model_section`), when the test gives one. The files are tiny; the damage helpers truncate, delete, overwrite or replace
one of them the way an interrupted save or a bad disk would.

Eole also links the files of the newest step into the checkpoint directory itself,
through relative symbolic links (`link_newest_step`); a resume never goes through them.

D9 adds the SOURCES of a continuation: `make_finished_run` is a run that `rinchen train
launch` finished (its step directories, `launch.json` and a `train_launch` manifest listing
the final step's four files with their real SHA-256, written through `rinchen.manifest`),
and `make_averaged_dir` is what `rinchen train average` leaves (the averaged weights, the
files beside them, `averaged_from.json` and a `train_average` manifest). Their vocabulary
is laid out as Eole 0.6.4 writes it (`eole_vocabulary`): a list of tokens for each side.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.spm.symbols import user_defined_symbols
from rinchen.train.checkpoints import LAUNCH_FILE, RunIdentity, write_launch_record

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
AVERAGED_FROM_FILE = "averaged_from.json"
MANIFEST_FILE = "manifest.json"
# What the specials of a vocabulary file hold in Eole 0.6.4, then the tokens a corpus
# showed, then every reserved symbol of the project: the vocabulary a teacher has when
# `launch` has made it (the vocabulary invariant of D9).
EOLE_SPECIALS = ("<unk>", "<blank>", "<s>", "</s>")
SOURCE_VOCABULARY = (*EOLE_SPECIALS, "a", "b", *user_defined_symbols())


# The `model` section of the `config.json` that Eole 0.6.4 saved into `step_20` of a real
# CPU run of the tiny teacher (64 wide, 2+2 layers, 4 heads, feed-forward 128), copied key
# for key: what the config sets, and what Eole fills in beside it (`tgt_word_vec_size`,
# `n_positions`, `with_cross_attn`, the stacks' own copies of the width and the heads).
_EOLE_MODEL_SECTION: dict[str, Any] = {
    "hidden_size": 64,
    "transformer_ff": 128,
    "position_encoding_type": "SinusoidalInterleaved",
    "architecture": "transformer",
    "share_embeddings": True,
    "heads": 4,
    "share_decoder_embeddings": True,
    "embeddings": {
        "tgt_word_vec_size": 64,
        "src_word_vec_size": 64,
        "position_encoding_type": "SinusoidalInterleaved",
        "word_vec_size": 64,
    },
    "decoder": {
        "layers": 2,
        "transformer_ff": 128,
        "hidden_size": 64,
        "with_cross_attn": True,
        "position_encoding_type": "SinusoidalInterleaved",
        "heads": 4,
        "n_positions": None,
        "tgt_word_vec_size": 64,
        "decoder_type": "transformer",
    },
    "encoder": {
        "layers": 2,
        "transformer_ff": 128,
        "hidden_size": 64,
        "position_encoding_type": "SinusoidalInterleaved",
        "encoder_type": "transformer",
        "heads": 4,
        "n_positions": None,
        "src_word_vec_size": 64,
    },
}


def eole_model_section(
    *,
    hidden_size: int = 64,
    heads: int = 4,
    transformer_ff: int = 128,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
) -> dict[str, Any]:
    """The `model` section Eole saves for a transformer of these sizes: the real one of the
    tiny teacher (64, 4, 128, 2, 2 by default) with the sizes replaced wherever they
    appear."""
    section = copy.deepcopy(_EOLE_MODEL_SECTION)
    section["hidden_size"] = hidden_size
    section["heads"] = heads
    section["transformer_ff"] = transformer_ff
    section["embeddings"]["word_vec_size"] = hidden_size
    section["embeddings"]["tgt_word_vec_size"] = hidden_size
    section["embeddings"]["src_word_vec_size"] = hidden_size
    for name, layers in (("encoder", encoder_layers), ("decoder", decoder_layers)):
        section[name].update(
            layers=layers,
            hidden_size=hidden_size,
            heads=heads,
            transformer_ff=transformer_ff,
        )
    section["encoder"]["src_word_vec_size"] = hidden_size
    section["decoder"]["tgt_word_vec_size"] = hidden_size
    return section


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


def eole_vocabulary(tokens: Sequence[str]) -> dict[str, Any]:
    """The content of the `vocab.json` Eole 0.6.4 saves for a shared vocabulary of
    `tokens`: the list of tokens for each side, the decoder start token and the
    specials."""
    return {
        "src": list(tokens),
        "tgt": list(tokens),
        "decoder_start_token": "<s>",
        "specials": {
            "unk_token": "<unk>",
            "pad_token": "<blank>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
    }


def make_step_dir(
    model_path: Path,
    step: int,
    *,
    tensors: Mapping[str, NDArray[Any]] | None = None,
    weights: bytes | None = None,
    model: Mapping[str, Any] | None = None,
    vocab_tokens: Sequence[str] | None = None,
    vocab_document: Mapping[str, Any] | None = None,
) -> Path:
    """Create `model_path/step_<step>` as a COMPLETE step directory and return it.

    The weights are `tensors` (default `default_tensors(step)`) written by
    `safetensors.numpy.save_file`, or, when `weights` is given, those bytes as the
    weights file (`safetensors_bytes`). `model` is the `model` section of `config.json`
    (`eole_model_section`); without it the config has none. `vocab_tokens` are the
    tokens of `vocab.json` as Eole writes it (`eole_vocabulary`) and `vocab_document` is
    the whole content of that file; without either the file is a small stub.
    `model_path` is created when missing.
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
    vocabulary: Mapping[str, Any] = {
        "src": [["a", 0], ["b", 1]],
        "tgt": [["a", 0]],
        "specials": {},
    }
    if vocab_tokens is not None:
        vocabulary = eole_vocabulary(vocab_tokens)
    if vocab_document is not None:
        vocabulary = vocab_document
    (step_dir / VOCAB_FILE).write_text(json.dumps(vocabulary), encoding="utf-8")
    (step_dir / SPM_ARTIFACT).write_bytes(b"sentencepiece model of the toy corpus")
    # Eole writes the config last, so it is the file that says the rest was saved.
    saved_config: dict[str, Any] = {
        "training": {"model_path": str(model_path)},
        "saved_at": step,
    }
    if model is not None:
        saved_config["model"] = dict(model)
    (step_dir / CONFIG_FILE).write_text(json.dumps(saved_config), encoding="utf-8")
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


# ---- the sources of a continuation -------------------------------------------------

LAUNCH_STAGE = "train_launch"
AVERAGE_STAGE = "train_average"
_NOW = "2026-11-09T10:00:00+00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FabricatedRun:
    """A run `rinchen train launch` finished: `run_dir` holds `ckpt/step_<N>` for each of
    `steps`, `launch.json`, `command.sh` and `manifest.json`."""

    run_dir: Path
    steps: tuple[int, ...]

    @property
    def model_path(self) -> Path:
        return self.run_dir / "ckpt"

    @property
    def final(self) -> Path:
        return self.step(self.steps[-1])

    @property
    def manifest(self) -> Path:
        return self.run_dir / MANIFEST_FILE

    def step(self, step: int) -> Path:
        return self.model_path / f"step_{step}"


def write_manifest(
    directory: Path,
    files: Sequence[Path],
    *,
    stage: str = LAUNCH_STAGE,
    status: Mapping[str, str] | None = None,
    inputs: Sequence[ManifestFile] = (),
) -> Path:
    """The manifest of `stage` in `directory`, written through `rinchen.manifest.store`,
    listing `files` as outputs with their real SHA-256 and size (the way a stage lists
    the files it wrote, by the path they had then) and `inputs` as they are."""
    outputs = [
        ManifestFile(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
        for path in files
    ]
    ok = {"launch": "ok"} if stage == LAUNCH_STAGE else {"average": "ok"}
    manifest = Manifest(
        stage=stage,
        created=_NOW,
        code_version="0000000",
        config_sha256="0" * 64,
        seed=1234,
        inputs=list(inputs),
        outputs=outputs,
        metrics={"steps": 200.0},
        tool_versions={"eole": "0.6.4"},
        status=dict(ok if status is None else status),
    )
    return store.write(directory, manifest)


def write_run_record(run_dir: Path) -> None:
    """The `launch.json` that `rinchen train launch` writes into a run directory (which
    must exist): what makes the directory a run of ours."""
    write_launch_record(
        run_dir,
        RunIdentity(identity="a" * 64, world_size=1, gpu_ranks=(), created=_NOW),
    )


def make_finished_run(
    root: Path,
    *,
    steps: Sequence[int] = (50, 100, 150, 200),
    model: Mapping[str, Any] | None = None,
    vocab_tokens: Sequence[str] | None = SOURCE_VOCABULARY,
    vocab_document: Mapping[str, Any] | None = None,
    manifest: bool = True,
    stage: str = LAUNCH_STAGE,
    status: Mapping[str, str] | None = None,
    listed: Sequence[str] = REQUIRED_FILES,
    listed_step: int | None = None,
) -> FabricatedRun:
    """A finished run in `root/run`: complete step directories for `steps` (the model of
    `eole_model_section` unless `model` says), a real `launch.json`, a `command.sh` and,
    when `manifest`, a manifest of `stage` that lists the files named in `listed` of the
    step `listed_step` (the final one unless said), the other two outputs of a launch
    being `launch.json` and `command.sh`."""
    run_dir = root / "run"
    model_path = run_dir / "ckpt"
    for step in steps:
        make_step_dir(
            model_path,
            step,
            model=eole_model_section() if model is None else model,
            vocab_tokens=vocab_tokens,
            vocab_document=vocab_document,
        )
    write_run_record(run_dir)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    run = FabricatedRun(run_dir, tuple(steps))
    if manifest:
        named = run.step(steps[-1] if listed_step is None else listed_step)
        files = [named / name for name in listed]
        files += [run_dir / LAUNCH_FILE, run_dir / "command.sh"]
        write_manifest(run_dir, files, stage=stage, status=status)
    return run


@dataclass(frozen=True)
class AveragedSource:
    """One checkpoint an averaged model says it was made from: the step directory as
    `averaged_from.json` names it (it need not exist), its step and the SHA-256 of its
    weights."""

    directory: str
    step: int
    sha256: str


def averaged_sources(directory: Path) -> tuple[AveragedSource, AveragedSource]:
    """The two checkpoints the averaged model in `directory` is made from by default."""
    base = directory.parent / "ckpt"
    return (
        AveragedSource(str(base / "step_150"), 150, "1" * 64),
        AveragedSource(str(base / "step_200"), 200, "2" * 64),
    )


def make_averaged_dir(
    directory: Path,
    *,
    model: Mapping[str, Any] | None = None,
    vocab_tokens: Sequence[str] | None = SOURCE_VOCABULARY,
    vocab_document: Mapping[str, Any] | None = None,
    manifest: bool = True,
    stage: str = AVERAGE_STAGE,
    status: Mapping[str, str] | None = None,
    listed: Sequence[str] = (WEIGHTS_FILE, AVERAGED_FROM_FILE, CONFIG_FILE, VOCAB_FILE),
    recorded: Sequence[AveragedSource] | None = None,
    inputs: Sequence[AveragedSource] | None = None,
    recorded_weights_sha256: str | None = None,
    record_text: str | None = None,
) -> Path:
    """What `rinchen train average` leaves in `directory`: the weights, the config and
    vocabulary of the last step, the SentencePiece artifact (never `optimizer.pt`),
    `averaged_from.json` and, when `manifest`, a manifest of `stage` listing `listed`.

    `averaged_from.json` records the sources `recorded` (default `averaged_sources`) and
    the SHA-256 of the weights (`recorded_weights_sha256`, default the real one); the
    manifest lists the weights of the sources `inputs` (default the recorded ones), as
    the command writes them, so the two agree unless a test says otherwise.
    `record_text` is the whole content of `averaged_from.json` instead."""
    scratch = directory.parent / f".{directory.name}.steps"
    step = make_step_dir(
        scratch,
        200,
        model=eole_model_section() if model is None else model,
        vocab_tokens=vocab_tokens,
        vocab_document=vocab_document,
    )
    directory.mkdir(parents=True)
    for name in (WEIGHTS_FILE, CONFIG_FILE, VOCAB_FILE, SPM_ARTIFACT):
        shutil.copyfile(step / name, directory / name)
    shutil.rmtree(scratch)
    sources = averaged_sources(directory) if recorded is None else tuple(recorded)
    weights_sha256 = (
        sha256(directory / WEIGHTS_FILE)
        if recorded_weights_sha256 is None
        else recorded_weights_sha256
    )
    record = {
        "schema_version": 1,
        "sources": [
            {"path": s.directory, "step": s.step, "sha256": s.sha256} for s in sources
        ],
        "weights": {"file": WEIGHTS_FILE, "sha256": weights_sha256},
    }
    (directory / AVERAGED_FROM_FILE).write_text(
        json.dumps(record) if record_text is None else record_text, encoding="utf-8"
    )
    if manifest:
        write_manifest(
            directory,
            [directory / name for name in listed],
            stage=stage,
            status=status,
            inputs=[
                ManifestFile(
                    path=f"{s.directory}/{WEIGHTS_FILE}", sha256=s.sha256, bytes=1
                )
                for s in (sources if inputs is None else inputs)
            ],
        )
    return directory


def flip_one_byte(path: Path, *, marker: bytes | None = None) -> None:
    """Change one byte of `path`: the first byte of `marker` (a text that occurs in it, so
    a JSON file stays valid JSON), or the last byte of the file."""
    content = bytearray(path.read_bytes())
    index = content.index(marker) if marker is not None else len(content) - 1
    content[index] ^= 0x01
    path.write_bytes(bytes(content))


def tree_digest(directory: Path) -> str:
    """One digest of every file (name and bytes), link (target) and directory (name) under
    `directory`: equal before and after exactly when nothing there was written."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        digest.update(str(path.relative_to(directory)).encode())
        if path.is_symlink():
            digest.update(b"->" + str(path.readlink()).encode())
        elif path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
