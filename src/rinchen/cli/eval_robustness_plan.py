"""What the options of `rinchen eval robustness` ask for (D12): the run as a plan, or a
`ConfigError` before anything is read or hashed.

A plan is a `ModelPlan` (the engine translates: the export, the converter's two paths when one is
attached, the name of the system when the operator gave one) or a `HypothesesPlan` (the texts come
from a file, under a name). `RunOptions` are the options that decide what a model run measures:
they make the engine's `LoadOptions` and `DecodeOptions` and the record of the options that
`robustness.json` and the manifest's digest stand on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from rinchen.errors import ConfigError
from rinchen.eval.robust_suite_rules import is_id
from rinchen.infer.engine_types import DecodeOptions, LoadOptions, ScriptMode

_SYSTEM_RULE: Final = "letters, digits, '.', '_' and '-'"


@dataclass(frozen=True)
class ModelPlan:
    """A run in which the engine translates: the export, the converter's two paths when one
    is attached and the name of the system when the operator gave one."""

    model: Path
    converter: tuple[Path, Path] | None
    name: str | None


@dataclass(frozen=True)
class HypothesesPlan:
    """A run that reads the texts from a file, under `name`."""

    path: Path
    name: str


@dataclass(frozen=True)
class RunOptions:
    """The options that decide a model run's numbers."""

    beam_size: int
    max_decoding_length: int
    script_mask: ScriptMode
    device: Literal["cpu", "cuda"]
    threads: int

    def load(self) -> LoadOptions:
        return LoadOptions(device=self.device, intra_threads=self.threads)

    def decode(self) -> DecodeOptions:
        return DecodeOptions(
            beam_size=self.beam_size,
            max_decoding_length=self.max_decoding_length,
            script_mask=self.script_mask,
        )

    def record(self, *, converter: bool) -> dict[str, str | int | bool]:
        return {
            "beam_size": self.beam_size,
            "converter": converter,
            "device": self.device,
            "max_decoding_length": self.max_decoding_length,
            "script_mask": self.script_mask,
            "threads": self.threads,
        }


def plan_of(
    model: Path | None,
    hypotheses: Path | None,
    system: str | None,
    converter_ct2: Path | None,
    converter_spm: Path | None,
) -> ModelPlan | HypothesesPlan:
    """The options as a plan, or a ConfigError: exactly one of the two sources of hypotheses,
    a usable system name, the converter pair and only with a model."""
    if system is not None and not is_id(system):
        raise ConfigError(f"--system must be 1 to 64 characters: {_SYSTEM_RULE}")
    if model is not None and hypotheses is not None:
        raise ConfigError(
            "give --model or --hypotheses, not both (RINCHEN_MODEL counts as --model)"
        )
    if model is not None:
        converter = _converter_paths(converter_ct2, converter_spm)
        return ModelPlan(Path(os.path.abspath(model)), converter, system)
    if hypotheses is None:
        raise ConfigError(
            "give --model or --hypotheses: the texts to score come from an engine or a file"
        )
    if converter_ct2 is not None or converter_spm is not None:
        raise ConfigError(
            "the script converter options need --model: they attach to the engine"
        )
    name = hypotheses.stem if system is None else system
    if not is_id(name):
        raise ConfigError(
            "the name of the hypotheses file is not a usable system name (1 to 64 "
            f"characters: {_SYSTEM_RULE}): give --system"
        )
    return HypothesesPlan(Path(os.path.abspath(hypotheses)), name)


def _converter_paths(
    ct2_dir: Path | None, spm_model: Path | None
) -> tuple[Path, Path] | None:
    if ct2_dir is None and spm_model is None:
        return None
    if ct2_dir is None or spm_model is None:
        raise ConfigError(
            "--script-converter-ct2 and --script-converter-spm go together: the converter "
            "needs its CTranslate2 directory and its SentencePiece model"
        )
    return Path(os.path.abspath(ct2_dir)), Path(os.path.abspath(spm_model))
