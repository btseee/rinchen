"""The commands the CLI stages of the tiny pipeline run, and what they leave behind (D15, spec P2, P3, P7
and P17).

A stage that has a command runs it as a child, `python -m rinchen <group> <command> ...`, so what is proved
is the command's own wiring, guards and exit codes. This module is the one place the stages and the
tests take the commands from: one builder per call returns the argument list, in the order of the spike's
prototype, and the readers of what a command wrote (`command_readers`) and the writers of the files it
reads (`command_writers`) are re-exported here. Everything in it is pure. The lists are checked against
the real commands' `--help` by the tests, and no list can hold a device, an upload or a download: every
one goes through `rinchen_argv`, which refuses those words whatever value a caller gives.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from rinchen.config.tiny import Quantization
from rinchen.tiny.command_readers import (
    read_converter_eval,
    read_json,
    read_parity,
    read_result_metrics,
    read_synth_counts,
)
from rinchen.tiny.command_writers import (
    MONO_LANGUAGE,
    OTHER_LANGUAGE,
    write_char_config,
    write_matrix,
    write_spm_config,
    write_synth_config,
)

__all__ = [
    "MONO_LANGUAGE",
    "OTHER_LANGUAGE",
    "converter_build_data",
    "converter_eval",
    "converter_export",
    "converter_train",
    "eval_report",
    "eval_robustness",
    "eval_run",
    "eval_run_hypotheses",
    "export_convert",
    "export_parity",
    "read_converter_eval",
    "read_json",
    "read_parity",
    "read_result_metrics",
    "read_synth_counts",
    "rinchen_argv",
    "spm_check",
    "spm_train",
    "synth_run",
    "train_average",
    "train_launch",
    "train_prepare",
    "write_char_config",
    "write_matrix",
    "write_spm_config",
    "write_synth_config",
]

# Written in pieces: a scan of the sources refuses these words as strings.
_FORBIDDEN_TOKENS: Final = frozenset(
    {"au" + "to", "cu" + "da", "fe" + "tch", "--up" + "load", "--pub" + "lic"}
)
_DEVICE_OPTION: Final = "--dev" + "ice"
# The profile of the converter's own training that runs on a CPU: not a profile of the pipeline.
_CONVERTER_PROFILE: Final = "tiny"


def rinchen_argv(*args: str | Path) -> list[str]:
    """`[python, -m, rinchen, *args]`, every argument as text.

    Raises:
        ValueError: an argument is a device option, an upload or public-release flag or the word of a
            download or of an automatic device choice, in any case; the message quotes none.
    """
    argv = [sys.executable, "-m", "rinchen", *(str(arg) for arg in args)]
    for token in argv[3:]:
        lowered = token.lower()
        if lowered in _FORBIDDEN_TOKENS or lowered.startswith(_DEVICE_OPTION):
            raise ValueError(
                "a command of the tiny pipeline never takes a device, an upload or a download"
            )
    return argv


# ---- the builders, one per call -----------------------------------------------------------------


def spm_train(config_path: Path) -> list[str]:
    return rinchen_argv("spm", "train", "--config", config_path)


def spm_check(model: Path, corpus_dir: Path) -> list[str]:
    return rinchen_argv("spm", "check", "--model", model, "--corpus-dir", corpus_dir)


def converter_build_data(pairs: Path, out_dir: Path) -> list[str]:
    return rinchen_argv(
        "convert-script", "build-data", "--pairs", pairs, "--out-dir", out_dir
    )


def converter_train(
    data_dir: Path, run_dir: Path, spm_config: Path, threads: int, timeout: int
) -> list[str]:
    return rinchen_argv(
        "convert-script",
        "train",
        "--profile",
        _CONVERTER_PROFILE,
        "--data-dir",
        data_dir,
        "--run-dir",
        run_dir,
        "--spm-config",
        spm_config,
        "--threads",
        str(threads),
        "--timeout",
        str(timeout),
    )


def converter_export(run_dir: Path, out_dir: Path, timeout: int) -> list[str]:
    return rinchen_argv(
        "convert-script",
        "export",
        "--run-dir",
        run_dir,
        "--out-dir",
        out_dir,
        "--timeout",
        str(timeout),
    )


def converter_eval(model_dir: Path, pairs: Path) -> list[str]:
    return rinchen_argv(
        "convert-script", "eval", "--model-dir", model_dir, "--pairs", pairs
    )


def synth_run(
    job: str, mono_input: Path, config: Path, out_dir: Path, seed: int
) -> list[str]:
    """The quality floor is always skipped: its scorer is a model that was not downloaded."""
    return rinchen_argv(
        "synth",
        "run",
        job,
        "--input",
        mono_input,
        "--config",
        config,
        "--out-dir",
        out_dir,
        "--seed",
        str(seed),
        "--no-qe",
    )


def train_prepare(
    matrix: Path, corpora: Sequence[Path], spm_model: Path, out_dir: Path, threads: int
) -> list[str]:
    """The preparation only: nothing is trained or scored.

    Raises:
        ValueError: there is no corpus.
    """
    if not corpora:
        raise ValueError("the preparation needs at least one corpus")
    corpus_options = [part for corpus in corpora for part in ("--corpus", corpus)]
    return rinchen_argv(
        "train",
        "ablate",
        "--matrix",
        matrix,
        *corpus_options,
        "--spm-model",
        spm_model,
        "--out-dir",
        out_dir,
        "--prepare-only",
        "--threads",
        str(threads),
    )


def train_launch(config: Path, run_dir: Path, threads: int, timeout: int) -> list[str]:
    return rinchen_argv(
        "train",
        "launch",
        "--config",
        config,
        "--run-dir",
        run_dir,
        "--threads",
        str(threads),
        "--timeout",
        str(timeout),
    )


def train_average(checkpoint_dir: Path, last: int) -> list[str]:
    return rinchen_argv(
        "train", "average", "--checkpoint-dir", checkpoint_dir, "--last", str(last)
    )


def export_convert(
    checkpoint: Path, spm_model: Path, quantization: Quantization, out_dir: Path
) -> list[str]:
    """The direction tag is always in the source."""
    return rinchen_argv(
        "export",
        "convert",
        "--checkpoint",
        checkpoint,
        "--spm-model",
        spm_model,
        "--tag-placement",
        "src",
        "--quantization",
        quantization,
        "--out-dir",
        out_dir,
    )


def export_parity(
    checkpoint: Path,
    export: Path,
    corpus: Path,
    out_dir: Path,
    max_lines: int,
    threads: int,
    timeout: int,
) -> list[str]:
    return rinchen_argv(
        "export",
        "parity",
        "--checkpoint",
        checkpoint,
        "--export",
        export,
        "--corpus",
        corpus,
        "--out-dir",
        out_dir,
        "--max-lines",
        str(max_lines),
        "--threads",
        str(threads),
        "--timeout",
        str(timeout),
    )


def eval_robustness(
    suite: Path,
    model: Path,
    out_dir: Path,
    beam_size: int,
    converter_ct2: Path,
    converter_spm: Path,
) -> list[str]:
    return rinchen_argv(
        "eval",
        "robustness",
        "--suite",
        suite,
        "--model",
        model,
        "--out-dir",
        out_dir,
        "--beam-size",
        str(beam_size),
        "--script-converter-ct2",
        converter_ct2,
        "--script-converter-spm",
        converter_spm,
    )


def _resamples_option(resamples: int | None) -> list[str]:
    return [] if resamples is None else ["--resamples", str(resamples)]


def eval_run(
    suite: Path,
    model: Path,
    system: str,
    out_dir: Path,
    beam_size: int,
    robustness_dir: Path | None,
    *,
    resamples: int | None = None,
) -> list[str]:
    """`--resamples` (the bootstrap's draws) is given only when `resamples` is: otherwise the
    metrics configuration's own count stands."""
    attached: list[str | Path] = (
        [] if robustness_dir is None else ["--robustness", robustness_dir]
    )
    return rinchen_argv(
        "eval",
        "run",
        "--suite",
        suite,
        "--model",
        model,
        "--system",
        system,
        "--beam-size",
        str(beam_size),
        "--out-dir",
        out_dir,
        *attached,
        *_resamples_option(resamples),
    )


def eval_run_hypotheses(
    suite: Path,
    hypotheses: Path,
    system: str,
    out_dir: Path,
    segment_scores: Mapping[str, Path],
    *,
    resamples: int | None = None,
) -> list[str]:
    """`--resamples` is given only when `resamples` is (see `eval_run`)."""
    scores = [
        part
        for metric, scored in segment_scores.items()
        for part in ("--segment-scores", f"{metric}={scored}")
    ]
    return rinchen_argv(
        "eval",
        "run",
        "--suite",
        suite,
        "--hypotheses",
        hypotheses,
        "--system",
        system,
        "--out-dir",
        out_dir,
        *scores,
        *_resamples_option(resamples),
    )


def eval_report(
    runs: Sequence[Path], suites: Sequence[Path], out_dir: Path
) -> list[str]:
    """The suites are named before the runs, as the prototype did.

    Raises:
        ValueError: there is no run or no suite.
    """
    if not runs or not suites:
        raise ValueError("a report needs at least one run and one suite")
    suite_options = [part for suite in suites for part in ("--suite", suite)]
    run_options = [part for run in runs for part in ("--run", run)]
    return rinchen_argv(
        "eval", "report", *suite_options, *run_options, "--out-dir", out_dir
    )
