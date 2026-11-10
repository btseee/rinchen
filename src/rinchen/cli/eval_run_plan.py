"""What the options of `rinchen eval run` ask for (D13): the system, the selection of directions, the
bootstrap overrides and the files of the optional inputs as a plan, or a `ConfigError` before anything
is read, hashed or opened.

Exactly one of `--model`, `--generator` and `--hypotheses` names the system. The options that set
how the engine decodes (`--beam-size`, `--max-decoding-length`, `--script-mask`, `--device`,
`--threads`) belong to `--model` and are `ConfigError` with anything else; they default to
`None` here so that giving one is told from not giving it, and the defaults of a model run are
D12's (`RunOptions`). `--generators-config` belongs to `--generator`, and `--segment-scores`
(`METRIC=FILE`, one metric once) to `--hypotheses`: the scores are of the texts of a file that
exists before the run.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from rinchen.cli.eval_robustness_plan import RunOptions
from rinchen.config.loader import REPO_ROOT
from rinchen.config.synth import SYNTH_CONFIG_PATH
from rinchen.errors import ConfigError
from rinchen.eval.robust_suite_rules import is_id
from rinchen.infer.engine_types import ScriptMode
from rinchen.metrics.registry import SEGMENT_SCORE_METRICS

__all__ = [
    "DEFAULT_METRICS_PATH",
    "GeneratorPlan",
    "HypothesesPlan",
    "ModelPlan",
    "RunPlan",
    "plan_of",
]

DEFAULT_METRICS_PATH: Final = REPO_ROOT / "configs" / "eval" / "metrics.yaml"
_SYSTEM_RULE: Final = "letters, digits, '.', '_' and '-'"
_SYSTEM_OPTIONS: Final = "--model, --generator and --hypotheses"
_DEFAULT_DECODE: Final = RunOptions(4, 256, "retry", "cpu", 1)


@dataclass(frozen=True)
class ModelPlan:
    """A run in which the engine translates: the export and how it decodes."""

    model: Path
    options: RunOptions


@dataclass(frozen=True)
class GeneratorPlan:
    """A run of a generator of D6's configuration: its `generator` name and the `config` file."""

    generator: str
    config: Path


@dataclass(frozen=True)
class HypothesesPlan:
    """A run that reads the outputs from a file."""

    path: Path


@dataclass(frozen=True)
class RunPlan:
    """What the options ask for: the `system`, its `name` when the operator gave one, the
    `selection` of directions (None: all of them), the robustness directory, the scores files
    by metric, the bootstrap overrides (None: the metrics configuration's) and the metrics
    configuration."""

    system: ModelPlan | GeneratorPlan | HypothesesPlan
    name: str | None
    selection: tuple[str, ...] | None
    robustness: Path | None
    segment_scores: dict[str, Path]
    resamples: int | None
    seed: int | None
    metrics_config: Path


def plan_of(
    *,
    model: Path | None,
    generator: str | None,
    hypotheses: Path | None,
    generators_config: Path | None,
    system: str | None,
    directions: Sequence[str] | None,
    beam_size: int | None,
    max_decoding_length: int | None,
    script_mask: ScriptMode | None,
    device: Literal["cpu", "cuda"] | None,
    threads: int | None,
    robustness: Path | None,
    segment_scores: Sequence[str] | None,
    resamples: int | None,
    seed: int | None,
    metrics_config: Path,
) -> RunPlan:
    """The options as a plan.

    Raises:
        ConfigError: the options contradict one another (see the module's text), a name is not an
            id, a `--segment-scores` argument is not `METRIC=FILE` with one of the three metrics,
            a metric or a direction is given twice.
    """
    if system is not None and not is_id(system):
        raise ConfigError(f"--system must be 1 to 64 characters: {_SYSTEM_RULE}")
    chosen = [option for option in (model, generator, hypotheses) if option is not None]
    if len(chosen) != 1:
        raise ConfigError(
            f"give exactly one of {_SYSTEM_OPTIONS} (RINCHEN_MODEL counts as --model), "
            f"not {len(chosen)}"
        )
    engine_options = _engine_options_given(
        beam_size, max_decoding_length, script_mask, device, threads
    )
    scores = _scores_of(segment_scores, hypotheses is not None)
    selection = _selection_of(directions)
    if model is not None:
        plan: ModelPlan | GeneratorPlan | HypothesesPlan = ModelPlan(
            Path(os.path.abspath(model)),
            RunOptions(
                _DEFAULT_DECODE.beam_size if beam_size is None else beam_size,
                _DEFAULT_DECODE.max_decoding_length
                if max_decoding_length is None
                else max_decoding_length,
                _DEFAULT_DECODE.script_mask if script_mask is None else script_mask,
                _DEFAULT_DECODE.device if device is None else device,
                _DEFAULT_DECODE.threads if threads is None else threads,
            ),
        )
        name = system
    else:
        if engine_options:
            raise ConfigError(
                "these options set how the engine decodes and need --model: "
                + ", ".join(engine_options)
            )
        if generator is not None:
            plan = GeneratorPlan(
                generator, Path(os.path.abspath(generators_config or SYNTH_CONFIG_PATH))
            )
            name = _usable_name(system, generator, "the name of the generator")
        else:
            assert hypotheses is not None
            plan = HypothesesPlan(Path(os.path.abspath(hypotheses)))
            name = _usable_name(
                system, hypotheses.stem, "the name of the hypotheses file"
            )
    if generators_config is not None and generator is None:
        raise ConfigError("--generators-config needs --generator")
    return RunPlan(
        system=plan,
        name=name,
        selection=selection,
        robustness=None if robustness is None else Path(os.path.abspath(robustness)),
        segment_scores=scores,
        resamples=resamples,
        seed=seed,
        metrics_config=Path(os.path.abspath(metrics_config)),
    )


def _engine_options_given(
    beam_size: int | None,
    max_decoding_length: int | None,
    script_mask: ScriptMode | None,
    device: Literal["cpu", "cuda"] | None,
    threads: int | None,
) -> list[str]:
    given = {
        "--beam-size": beam_size,
        "--max-decoding-length": max_decoding_length,
        "--script-mask": script_mask,
        "--device": device,
        "--threads": threads,
    }
    return [option for option, value in given.items() if value is not None]


def _usable_name(system: str | None, default: str, what: str) -> str:
    """`--system` when given, else `default`, which must be an id."""
    name = default if system is None else system
    if not is_id(name):
        raise ConfigError(
            f"{what} is not a usable system name (1 to 64 characters: {_SYSTEM_RULE}): "
            "give --system"
        )
    return name


def _selection_of(directions: Sequence[str] | None) -> tuple[str, ...] | None:
    if not directions:
        return None
    if len(set(directions)) != len(directions):
        raise ConfigError("--direction is given more than once for the same direction")
    return tuple(directions)


def _scores_of(
    arguments: Sequence[str] | None, with_hypotheses: bool
) -> dict[str, Path]:
    """The `METRIC=FILE` arguments as a mapping from the metric to its file."""
    if not arguments:
        return {}
    if not with_hypotheses:
        raise ConfigError(
            "--segment-scores needs --hypotheses: the scores are of the texts of a file "
            "that exists before the run"
        )
    scores: dict[str, Path] = {}
    for argument in arguments:
        metric, separator, file = argument.partition("=")
        if not separator or not file or metric not in SEGMENT_SCORE_METRICS:
            raise ConfigError(
                "--segment-scores must be METRIC=FILE with METRIC one of "
                + ", ".join(SEGMENT_SCORE_METRICS)
            )
        if metric in scores:
            raise ConfigError("--segment-scores names a metric more than once")
        scores[metric] = Path(os.path.abspath(file))
    return scores
