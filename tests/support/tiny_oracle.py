"""An independent oracle for the whole of `rinchen tiny run` (D15 Task 7, the differential check).

It was written before the command and shares no code with it: it imports nothing of `rinchen.cli.tiny*`
or `rinchen.tiny.pipeline`, restates the closed reasons as literals, and reads only the files on disk
(the inputs of the profile, and what the command left in its output directory) and the scenario it
drew. A scenario is what the fifteen fake stages do (each `ok`, `skipped` by itself, `fail`, raising a
`DataError` or raising something else), for how many seconds of a fake clock, under which budget, with
which options. From it the oracle derives, by a loop of its own:

- which stages ran and which were left after the first failure or by a spent budget, and the record each
  one came to (the reasons as the closed lists write them);
- the verdict and the exit code (0 when no stage failed, the stages that carry the figures of a threshold
  are `ok` and the elapsed time is within the budget, else 4);
- the bytes of `report.json` and the text of `report.md`, from the reports of `build_report` over the
  results it derived and the independent writers of `tiny_report_reference`;
- the keys and values of `manifest.json` (the inputs by digest of the files on disk, the two documents by
  the digest of what is on disk, the lock's every name, the seconds of each stage, the figures of
  `filter` and `train`, the status of each stage);
- the lines printed.

`check` returns the differences as a list of short descriptions, empty when the command did what the
oracle says. Nothing in a description quotes a file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import STAGE_IDS, TinyConfig
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import Inputs, build_report
from rinchen.tiny.states import StageRecord
from tests.support.tiny_cli import (
    CODE_VERSION,
    DATA_ERROR_TEXT,
    SECRET,
    FakeClock,
    Outcome,
    Recording,
)
from tests.support.tiny_report_inputs import golden_results
from tests.support.tiny_report_reference import reference_bytes, reference_markdown
from tests.support.tiny_stage_inputs import shipped_config

EARLIER: Final = "an earlier stage failed"
BUDGET: Final = "the wall-time budget is spent"
INPUT: Final = "an input is missing or differs"
RAISED: Final = "a check of the stage failed: stage_raised"
SKIP_REASONS: Final = (
    "no language-identification model: GlotLID is a download",
    "LaBSE needs a model that was not downloaded: --qe-fake runs the planted stand-in",
    "the QE scorer needs a model that was not downloaded: --qe-fake runs the planted stand-in",
    "no neural metric scorer: only the seam of score files was built",
    "the token-ratio check needs a SentencePiece model, which does not exist yet",
    "an earlier stage failed",
)
# The stages whose facts the thresholds read: a threshold without its facts is a failure.
FIGURE_STAGES: Final = ("evaluate", "parity", "robustness")
DURATIONS: Final = (0.25, 0.5, 1.0, 2.0, 5.0, 25.0, 60.0, 150.0)
BUDGETS: Final = (1, 5, 20, 100, 300, 600, 1200, 1200, 1200, 3600)
THREADS: Final = (None, 1, 4)
SEED: Final = 1234
PROFILE_FILE: Final = REPO_ROOT / "configs" / "profiles" / "tiny.yaml"
LOCK_FILE: Final = REPO_ROOT / "uv.lock"
INPUT_FILES: Final = (
    REPO_ROOT / "configs" / "tiny" / "tiny.yaml",
    *(
        REPO_ROOT / "tests" / "fixtures" / "parallel" / name
        for name in (
            "eng_Latn-khk_Cyrl.tsv",
            "rus_Cyrl-khk_Cyrl.tsv",
            "mn_Mong.tsv",
            "mono_khk_Cyrl.tsv",
            "noise.tsv",
            "nouns.tsv",
            "entities.tsv",
        )
    ),
    REPO_ROOT / "configs" / "eole" / "tiny.yaml",
    REPO_ROOT / "configs" / "serve" / "tiny.yaml",
    REPO_ROOT / "tests" / "fixtures" / "script_conv" / "tiny_words.tsv",
    PROFILE_FILE,
    LOCK_FILE,
)


@dataclass(frozen=True)
class Scenario:
    """What the fifteen fake stages do and how the command is called."""

    outcomes: tuple[Outcome, ...]
    budget: int
    qe_flag: str | None
    threads: int | None

    @property
    def table(self) -> dict[str, Outcome]:
        return dict(zip(STAGE_IDS, self.outcomes, strict=True))

    def arguments(self, out: Path) -> list[str]:
        arguments = ["tiny", "run", "--out-dir", str(out)]
        arguments += ["--budget-seconds", str(self.budget)]
        if self.qe_flag is not None:
            arguments.append(self.qe_flag)
        if self.threads is not None:
            arguments += ["--threads", str(self.threads)]
        return arguments


def draw(seed: int) -> Scenario:
    """The scenario of a seed: a third are clean runs (every stage `ok`, a skip here and there in a stage
    that carries no figure of a threshold), the others fail, skip, crash or run out of budget."""
    rng = random.Random(f"tiny-oracle-{seed}")
    clean = rng.random() < 0.35
    outcomes = []
    for stage_id in STAGE_IDS:
        seconds = rng.choice(DURATIONS)
        draw_kind = rng.random()
        if clean:
            skippable = stage_id not in FIGURE_STAGES
            kind = "skipped" if skippable and draw_kind < 0.08 else "ok"
        elif draw_kind < 0.88:
            kind = "ok"
        else:
            kind = rng.choice(("skipped", "fail", "data_error", "crash"))
        outcomes.append(
            Outcome(
                kind,
                seconds,
                reason=rng.choice(SKIP_REASONS),
                exit_code=rng.choice((1, 2, 3, 4, 137)),
            )
        )
    return Scenario(
        tuple(outcomes),
        rng.choice(BUDGETS),
        rng.choice((None, "--qe-fake", "--no-qe-fake")),
        rng.choice(THREADS),
    )


# ---- what the stages come to ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Expected:
    """The oracle's reading of a scenario."""

    states: dict[str, tuple[str, str | None]]
    ran: list[str]
    seconds: dict[str, float]
    results: list[StageResult]
    total: float
    verdict: str
    qe_fake: bool
    threads: int
    budget: int

    @property
    def exit_code(self) -> int:
        return 0 if self.verdict == "pass" else 4


def _own_record(stage_id: str, outcome: Outcome) -> tuple[str, str | None]:
    if outcome.kind == "ok":
        return ("ok", None)
    if outcome.kind == "skipped":
        return ("skipped", outcome.reason)
    if outcome.kind == "fail":
        return ("fail", f"exit code {outcome.exit_code}")
    if outcome.kind == "data_error":
        return ("fail", INPUT)
    assert outcome.kind == "crash"
    return ("fail", RAISED)


def expect(scenario: Scenario) -> Expected:
    """Walk the stages in order: a failure, or a budget that is gone before a stage, ends the run."""
    golden = {result.record.id: result for result in golden_results()}
    states: dict[str, tuple[str, str | None]] = {}
    seconds: dict[str, float] = {}
    results: list[StageResult] = []
    ran: list[str] = []
    elapsed = 0.0
    stopped = False
    for stage_id, outcome in zip(STAGE_IDS, scenario.outcomes, strict=True):
        if stopped:
            states[stage_id] = ("skipped", EARLIER)
            seconds[stage_id] = 0.0
            continue
        if scenario.budget - elapsed <= 0:
            states[stage_id] = ("fail", BUDGET)
            seconds[stage_id] = 0.0
            results.append(StageResult(StageRecord(stage_id, "fail", BUDGET), {}))
            stopped = True
            continue
        ran.append(stage_id)
        elapsed += outcome.seconds
        seconds[stage_id] = outcome.seconds
        state, reason = _own_record(stage_id, outcome)
        states[stage_id] = (state, reason)
        if state == "ok":
            results.append(golden[stage_id])
        else:
            results.append(StageResult(StageRecord(stage_id, state, reason), {}))  # type: ignore[arg-type]
        stopped = state == "fail"
    nothing_failed = all(state != "fail" for state, _ in states.values())
    figures = all(states[stage_id][0] == "ok" for stage_id in FIGURE_STAGES)
    verdict = (
        "pass" if nothing_failed and figures and elapsed <= scenario.budget else "fail"
    )
    config = shipped_config()
    return Expected(
        states,
        ran,
        seconds,
        results,
        elapsed,
        verdict,
        {"--qe-fake": True, "--no-qe-fake": False, None: config.qe_fake}[
            scenario.qe_flag
        ],
        scenario.threads if scenario.threads is not None else config.threads,
        scenario.budget,
    )


# ---- what is on disk --------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_report_bytes(expected: Expected) -> tuple[bytes, str]:
    """The bytes of `report.json` and the text of `report.md` the oracle expects, from the files of the
    profile as they are on disk now."""
    fixtures = {
        path.name: sha256(path)
        for path in INPUT_FILES
        if path.parent.name == "parallel"
    }
    inputs = Inputs(sha256(INPUT_FILES[0]), fixtures, sha256(LOCK_FILE))
    config: TinyConfig = shipped_config().model_copy(
        update={"budget_seconds": expected.budget}
    )
    report = build_report(
        "tiny", expected.qe_fake, inputs, expected.results, expected.total, config
    )
    return reference_bytes(report), reference_markdown(report)


def _locked_versions() -> dict[str, str]:
    packages = tomllib.loads(LOCK_FILE.read_text(encoding="utf-8"))["package"]
    found: dict[str, set[str]] = {}
    for package in packages:
        name = re.sub(r"[-_.]+", "-", package["name"]).lower()
        found.setdefault(name, set()).add(package["version"])
    return {name: ", ".join(sorted(found[name])) for name in found}


def expected_lines(
    expected: Expected, out: Path, document: dict[str, Any]
) -> list[str]:
    """The lines printed: a line for each stage, one for each threshold (its value, and for the
    wall time the seconds the run took), the verdict and the three paths."""
    lines = []
    for stage_id in STAGE_IDS:
        state, reason = expected.states[stage_id]
        lines.append(f"{stage_id}: {state}" + (f" ({reason})" if reason else ""))
    for threshold in document["thresholds"]:
        value = threshold["value"]
        if threshold["id"] == "wall_time":
            value = expected.total
        shown = "-" if value is None else format(value, ".6g")
        lines.append(
            f"threshold {threshold['id']}: {threshold['verdict']} "
            f"(value {shown}, bound {threshold['bound']})"
        )
    lines.append(f"verdict: {expected.verdict}")
    lines += [
        f"report: {out / 'report.json'}",
        f"summary: {out / 'report.md'}",
        f"manifest: {out / 'manifest.json'}",
    ]
    return lines


def check(
    scenario: Scenario,
    out: Path,
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    recording: Recording,
    clock: FakeClock,
) -> list[str]:
    """What differs between what the command did and what the oracle says it must do."""
    expected = expect(scenario)
    problems: list[str] = []

    def differ(what: str, found: object, wanted: object) -> None:
        if found != wanted:
            problems.append(what)

    differ("exit code", exit_code, expected.exit_code)
    differ("stages called", recording.stages, expected.ran)
    differ("seed given to the table", recording.seeds, [yaml_seed()])
    differ(
        "lock bytes given to the table", recording.lock_bytes, [LOCK_FILE.read_bytes()]
    )
    differ("fake clock at the end", clock.now - 1000.0, expected.total)
    if recording.contexts:
        first = recording.contexts[0]
        differ("qe flag of the context", first.qe_fake, expected.qe_fake)
        differ("threads of the context", first.threads, expected.threads)
        differ("out dir of the context", first.out_dir, out)
    for text in (stdout, stderr):
        if SECRET in text or DATA_ERROR_TEXT in text:
            problems.append("a text of an input was printed")
    problems += _check_files(scenario, expected, out, stdout)
    return problems


def yaml_seed() -> int:
    seed = yaml.safe_load(PROFILE_FILE.read_text(encoding="utf-8"))["seed"]
    assert isinstance(seed, int)
    return seed


def _check_raised_logs(scenario: Scenario, expected: Expected, out: Path) -> list[str]:
    """A stage that raised leaves `logs/<stage>-raised.log`: the class of the exception, the message
    of a `DataError` (it names a place) and the frames, and never the text of another crash."""
    raised = {
        stage_id: outcome.kind
        for stage_id, outcome in scenario.table.items()
        if stage_id in expected.ran and outcome.kind in ("data_error", "crash")
    }
    logs = out / "logs"
    found = sorted(path.name for path in logs.iterdir()) if logs.is_dir() else []
    if found != sorted(f"{stage_id}-raised.log" for stage_id in raised):
        return [f"the raised logs are {found}"]
    problems = []
    for stage_id, kind in raised.items():
        text = (logs / f"{stage_id}-raised.log").read_text(encoding="utf-8")
        if kind == "data_error" and not (
            "DataError" in text and DATA_ERROR_TEXT in text
        ):
            problems.append("the log of a DataError")
        if kind == "crash" and not ("RuntimeError" in text and SECRET not in text):
            problems.append("the log of a crash")
    return problems


def _check_files(
    scenario: Scenario, expected: Expected, out: Path, stdout: str
) -> list[str]:
    problems: list[str] = []
    names = sorted(path.name for path in out.iterdir())
    wanted_names = ["manifest.json", "report.json", "report.md"]
    if any(
        outcome.kind in ("data_error", "crash") and stage_id in expected.ran
        for stage_id, outcome in scenario.table.items()
    ):
        wanted_names.insert(0, "logs")
    if names != wanted_names:
        return [f"the output directory holds {names}"]
    if any(path.name.endswith(".part") for path in out.rglob("*")):
        problems.append("a part file was left")
    problems += _check_raised_logs(scenario, expected, out)
    wanted_json, wanted_markdown = expected_report_bytes(expected)
    found_json = (out / "report.json").read_bytes()
    if found_json != wanted_json:
        problems.append("report.json")
    if (out / "report.md").read_bytes() != wanted_markdown.encode("utf-8"):
        problems.append("report.md")
    for name in ("report.json", "report.md", "manifest.json"):
        content = (out / name).read_bytes()
        if SECRET.encode() in content or DATA_ERROR_TEXT.encode() in content:
            problems.append(f"{name} holds a text of an input")
    document = json.loads(found_json)
    if document["verdict"] != expected.verdict:
        problems.append("verdict")
    if stdout.splitlines() != expected_lines(expected, out, document):
        problems.append("printed lines")
    problems += _check_manifest(expected, out)
    return problems


def _check_manifest(expected: Expected, out: Path) -> list[str]:
    problems: list[str] = []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    keys = {
        "schema",
        "stage",
        "created",
        "code_version",
        "config_sha256",
        "seed",
        "inputs",
        "outputs",
        "metrics",
        "tool_versions",
        "status",
    }
    if set(manifest) != keys:
        return ["the keys of the manifest"]

    def differ(what: str, found: object, wanted: object) -> None:
        if found != wanted:
            problems.append(f"manifest {what}")

    differ("stage", manifest["stage"], "tiny_run")
    differ("code version", manifest["code_version"], CODE_VERSION)
    differ("config digest", manifest["config_sha256"], sha256(INPUT_FILES[0]))
    differ("seed", manifest["seed"], yaml_seed())
    differ(
        "inputs",
        {item["path"]: item["sha256"] for item in manifest["inputs"]},
        {str(path): sha256(path) for path in INPUT_FILES},
    )
    differ(
        "input order",
        [item["path"] for item in manifest["inputs"]],
        [str(p) for p in INPUT_FILES],
    )
    differ(
        "outputs",
        [(item["path"], item["sha256"], item["bytes"]) for item in manifest["outputs"]],
        [
            (str(out / name), sha256(out / name), (out / name).stat().st_size)
            for name in ("report.json", "report.md")
        ],
    )
    wanted_metrics: dict[str, float] = {
        f"seconds.{stage_id}": expected.seconds[stage_id] for stage_id in STAGE_IDS
    }
    if expected.states["filter"] == ("ok", None) and "filter" in expected.ran:
        wanted_metrics["rows.kept"] = 756
    if expected.states["train"] == ("ok", None) and "train" in expected.ran:
        wanted_metrics["train.steps"] = 1500
    differ("metrics", manifest["metrics"], wanted_metrics)
    differ(
        "status",
        manifest["status"],
        {stage_id: expected.states[stage_id][0] for stage_id in STAGE_IDS},
    )
    wanted_versions = {**_locked_versions(), "python": platform.python_version()}
    differ("tool versions", manifest["tool_versions"], wanted_versions)
    return problems


def describe(outcomes: Sequence[Outcome]) -> str:
    """A short label of a scenario for a failure message."""
    return ",".join(f"{outcome.kind[:2]}{outcome.seconds:g}" for outcome in outcomes)


__all__ = [
    "BUDGET",
    "EARLIER",
    "INPUT",
    "RAISED",
    "Expected",
    "Scenario",
    "check",
    "describe",
    "draw",
    "expect",
]
