"""The commands a run of the tiny pipeline would start, for the dry run (D15, spec P13): the argument list of
every command of every stage, built by the same builders the stages call (`rinchen.tiny.commands`) over the
paths the stages use under the output directory.

Nothing is read, written or started. The id of the variant that the preparation makes is named by the
preparation (from the matrix), so the plan shows `<variant>` where a run has it. The timeout of a command
is the timeout of its stage, which is what a run gives it while the budget is not nearly spent. The names of
the files a stage makes inside its own directory (`spm.yaml`, `text`, `char.yaml`, `corpus`, `synth.yaml`,
`config.yaml`, `run`, `ckpt`, `averaged`) are the stages' own, written here a second time: a test runs the
real stages over the recording runner and compares the commands they start with these lists, so a stage
that moves one of them breaks it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from rinchen.config.tiny import StageId, TinyConfig
from rinchen.tiny import commands
from rinchen.tiny.context import stage_dir
from rinchen.tiny.eval_stages import (
    HYPOTHESES_FILE,
    NEURAL_METRIC,
    NEURAL_SUFFIX,
    REPORT_DIR,
    RUNS_DIR,
    SCORES_DIR,
    SUITES_DIR,
    system_name,
)
from rinchen.tiny.model_stages import (
    CONVERTER_CHAR_MODEL,
    CONVERTER_MODEL_DIR,
    CONVERTER_RUN_DIR,
    MONO_FILE,
    SPM_MODEL_FILE,
    SYNTH_CORPUS_FILE,
    SYNTH_RUN_DIR,
)
from rinchen.tiny.model_stages_rest import ABLATE_DIR, MATRIX_FILE, TRAIN_FILE
from rinchen.tiny.suites import HELDOUT_NAMES, MEMORISED_NAMES, ROBUSTNESS_NAME
from rinchen.tiny.thresholds import GATED_SYSTEM, REFERENCE_SYSTEM

__all__ = ["VARIANT_PLACEHOLDER", "stage_commands"]

VARIANT_PLACEHOLDER: Final = "<variant>"
Commands = dict[str, list[list[str]]]


class _Plan:
    """The paths of a run under `out` and the values the builders are given."""

    def __init__(
        self, config: TinyConfig, out: Path, threads: int, seed: int, qe_fake: bool
    ) -> None:
        self.config = config
        self.out = out
        self.threads = threads
        self.seed = seed
        self.qe_fake = qe_fake
        self.variant = self.at("prepare", ABLATE_DIR, "variants", VARIANT_PLACEHOLDER)
        self.averaged = self.variant / "run" / "averaged"
        self.spm_model = self.at("spm", SPM_MODEL_FILE)

    def at(self, stage_id: StageId, *parts: str) -> Path:
        directory = stage_dir(self.out, stage_id)
        assert directory is not None, "the stages that have commands have a directory"
        return directory.joinpath(*parts)

    def timeout(self, stage_id: StageId) -> int:
        return self.config.timeouts[stage_id]

    def spm(self) -> list[list[str]]:
        text = self.at("spm", "text")
        return [
            commands.spm_train(self.at("spm", "spm.yaml")),
            commands.spm_check(self.spm_model, text),
        ]

    def converter(self) -> list[list[str]]:
        pairs = self.config.converter.pairs
        corpus = self.at("converter", "corpus")
        run = self.at("converter", CONVERTER_RUN_DIR)
        model = self.at("converter", CONVERTER_MODEL_DIR)
        timeout = self.timeout("converter")
        return [
            commands.converter_build_data(pairs, corpus),
            commands.converter_train(
                corpus, run, self.at("converter", "char.yaml"), self.threads, timeout
            ),
            commands.converter_export(run, model, timeout),
            commands.converter_eval(model, pairs),
        ]

    def synth(self) -> list[list[str]]:
        return [
            commands.synth_run(
                self.config.synth.job,
                self.at("synth", MONO_FILE),
                self.at("synth", "synth.yaml"),
                self.at("synth", SYNTH_RUN_DIR),
                self.seed,
            )
        ]

    def prepare(self) -> list[list[str]]:
        corpora = [
            self.at("filter", TRAIN_FILE),
            self.at("synth", SYNTH_RUN_DIR, SYNTH_CORPUS_FILE),
        ]
        return [
            commands.train_prepare(
                self.at("prepare", MATRIX_FILE),
                corpora,
                self.spm_model,
                self.at("prepare", ABLATE_DIR),
                self.threads,
            )
        ]

    def train(self) -> list[list[str]]:
        return [
            commands.train_launch(
                self.variant / "config.yaml",
                self.variant / "run",
                self.threads,
                self.timeout("train"),
            )
        ]

    def average(self) -> list[list[str]]:
        return [
            commands.train_average(
                self.variant / "run" / "ckpt", self.config.train.average_last
            )
        ]

    def export(self) -> list[list[str]]:
        return [
            commands.export_convert(
                self.averaged,
                self.spm_model,
                quantization,
                self.at("export", quantization),
            )
            for quantization in self.config.export.quantizations
        ]

    def parity(self) -> list[list[str]]:
        return [
            commands.export_parity(
                self.averaged,
                self.at("export", REFERENCE_SYSTEM),
                self.at("filter", TRAIN_FILE),
                self.at("parity", REFERENCE_SYSTEM),
                self.config.parity.max_lines,
                self.threads,
                self.timeout("parity"),
            )
        ]

    def robustness(self) -> list[list[str]]:
        return [
            commands.eval_robustness(
                self.at("robustness", SUITES_DIR, f"{ROBUSTNESS_NAME}.json"),
                self.at("export", GATED_SYSTEM),
                self.at("robustness"),
                self.config.decode.beam_size,
                self.at("converter", CONVERTER_MODEL_DIR),
                self.at("converter", CONVERTER_RUN_DIR, CONVERTER_CHAR_MODEL),
            )
        ]

    def _suite(self, name: str) -> Path:
        return self.at("evaluate", SUITES_DIR, f"{name}.json")

    def _run_dir(self, suite: str, system: str) -> Path:
        return self.at("evaluate", RUNS_DIR, suite, system)

    def evaluate(self) -> list[list[str]]:
        runs = self.at("evaluate", RUNS_DIR)
        beam = self.config.decode.beam_size
        resamples = self.config.evaluation.resamples
        gated = system_name(GATED_SYSTEM)
        found: list[list[str]] = []
        for quantization in (GATED_SYSTEM, REFERENCE_SYSTEM):
            for suite in MEMORISED_NAMES:
                found.append(
                    commands.eval_run(
                        self._suite(suite),
                        self.at("export", quantization),
                        system_name(quantization),
                        runs,
                        beam,
                        None,
                        resamples=resamples,
                    )
                )
        found.extend(
            commands.eval_run(
                self._suite(suite),
                self.at("export", GATED_SYSTEM),
                gated,
                runs,
                beam,
                self.at("robustness"),
                resamples=resamples,
            )
            for suite in HELDOUT_NAMES
        )
        if self.qe_fake:
            found.append(self._neural(runs, gated, resamples))
        found.append(
            commands.eval_report(
                [self._run_dir(suite, gated) for suite in HELDOUT_NAMES],
                [self._suite(suite) for suite in HELDOUT_NAMES],
                self.at("evaluate", REPORT_DIR),
            )
        )
        return found

    def _neural(self, runs: Path, gated: str, resamples: int) -> list[str]:
        suite = HELDOUT_NAMES[0]
        scores = self.at("evaluate", SCORES_DIR, f"{NEURAL_METRIC}.jsonl")
        return commands.eval_run_hypotheses(
            self._suite(suite),
            self._run_dir(suite, gated) / HYPOTHESES_FILE,
            f"{gated}-{NEURAL_SUFFIX}",
            runs,
            {NEURAL_METRIC: scores},
            resamples=resamples,
        )


def stage_commands(
    config: TinyConfig, out_dir: Path, *, threads: int, seed: int, qe_fake: bool
) -> Commands:
    """The argument lists of the commands each stage starts, by stage id in the order of `STAGE_IDS`
    (an empty list for a stage that runs in process), over the absolute form of `out_dir`."""
    plan = _Plan(config, out_dir.resolve(), threads, seed, qe_fake)
    return {
        "environment": [],
        "fetch": [],
        "corpus": [],
        "filter": [],
        "spm": plan.spm(),
        "converter": plan.converter(),
        "synth": plan.synth(),
        "prepare": plan.prepare(),
        "train": plan.train(),
        "average": plan.average(),
        "export": plan.export(),
        "parity": plan.parity(),
        "serve": [],
        "robustness": plan.robustness(),
        "evaluate": plan.evaluate(),
    }
