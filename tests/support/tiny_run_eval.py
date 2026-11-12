"""A recording fake runner that also plays the evaluation commands of the tiny pipeline (D15, Task 5).

`EvalRunner` is `tests/support/tiny_run.RecordingRunner` with three more commands: `eval robustness`,
`eval run` and `eval report`. It plays them as the real commands do from the point of view of a stage:
it reads the suites it is given with the real readers, checks the inputs the real command would need (a
model directory, a robustness directory, a converter that is given whole, an output directory that is
not the directory of an input, `--resamples` of at least 100, `--beam-size` with `--model` only, one of
`--model` and `--hypotheses`) and refuses with the exit code 2 and a line in the log, and for a score
file it uses D13's own reader against the suite and the hypotheses it was given, so a stage that writes
a score file with the wrong digest, the wrong segments or a missing score is refused. Then it writes the
files the real command writes, from the recordings of the spike under `tests/fixtures/tiny_run/`: the
`result.json` of every direction (the metrics of the recorded int8 or float32 run, the rows and the
statuses of what it wrote), the `hypotheses.jsonl` (the recorded outputs for the rows they cover and the
reference of the row for the others) and the `report.json` of the robustness run.

A command is told apart in `fail` and `after` by its key: `eval robustness`, `eval report` and, for a
run, `eval run <suite> <system>` (the stem of the suite file and the `--system`). `chrf` gives a
direction of a system over a suite another corpus chrF (its key is the suite, the system and the
direction), `edit` rewrites what a system wrote for a row, `robustness` changes a value of the block,
`ignore_scores` makes a hypotheses run leave the scores it was given out of its results (a command with
a defect).

The fixtures `model_outputs` (the output of the eight model stages, made once for a module over the plain
recording runner) and `evaluated_inputs` (the same with the robustness stage run) are here for the tests
that need what comes before the evaluation; each test copies them into its own output directory with
`eval_out`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.config.tiny import TinyConfig
from rinchen.errors import DataError
from rinchen.eval import robust_suite as robust_suite_reader
from rinchen.eval import suites as suite_reader
from rinchen.eval.outputs import OutputKey
from rinchen.eval.segment_scores import read_scores
from rinchen.tiny.context import StageResult
from rinchen.tiny.eval_stages import run_evaluate, run_robustness
from rinchen.tiny.fixtures import Fixtures
from tests.support.tiny_run import (
    Invocation,
    RecordingRunner,
    Refused,
    context,
    copy_inputs,
    recorded_text,
    require_dir,
    require_file,
    run_stages,
)
from tests.support.tiny_stage_inputs import offline_in

RESULT_FILES: Final = {
    "int8": "result_memorised_eng_khk_int8.json",
    "float32": "result_memorised_eng_khk_float32.json",
}
HYPOTHESES_FILES: Final = {
    ("tiny-memorised-eng", "int8"): "hypotheses_memorised_eng_int8.jsonl",
    ("tiny-memorised-eng", "float32"): "hypotheses_memorised_eng_float32.jsonl",
    ("tiny-heldout-eng", "int8"): "hypotheses_heldout_eng_int8.jsonl",
}
MIN_RESAMPLES: Final = 100
EditText = Callable[[str, str, str, str, str], str]

# The names the stage gives what it runs, and the keys of its commands, as the tests of it expect them.
INT8: Final = "tiny-int8"
FLOAT32: Final = "tiny-float32"
NEURAL: Final = "tiny-int8-qe"
MEMORISED: Final = ("tiny-memorised-eng", "tiny-memorised-rus", "tiny-memorised-mong")
HELDOUT: Final = ("tiny-heldout-eng", "tiny-heldout-rus")
MODEL_RUNS: Final = [
    *(f"eval run {suite} {INT8}" for suite in MEMORISED),
    *(f"eval run {suite} {FLOAT32}" for suite in MEMORISED),
    *(f"eval run {suite} {INT8}" for suite in HELDOUT),
]
SEAM_RUN: Final = f"eval run tiny-heldout-eng {NEURAL}"
ALL_KEYS: Final = [*MODEL_RUNS, SEAM_RUN, "eval report"]


def stage_dir(out: Path) -> Path:
    """The directory of the evaluation stage under `out`, absolute."""
    return out.resolve() / "11_evaluate"


@cache
def recorded_result(quantization: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(recorded_text(RESULT_FILES[quantization]))
    return document


@cache
def recorded_hypotheses(name: str) -> dict[tuple[str, str], str]:
    rows = [json.loads(line) for line in recorded_text(name).splitlines()]
    return {(row["direction"], row["id"]): row["text"] for row in rows}


def quantization_of(system: str) -> str:
    """`int8` or `float32` from a system name such as `tiny-float32` or `tiny-int8-qe`."""
    return "float32" if "float32" in system else "int8"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_suite(path: Path) -> tuple[suite_reader.Suite, str]:
    try:
        return suite_reader.read_suite(require_file(path))
    except DataError as error:
        raise Refused("the suite is not valid") from error


class _EvalPlayers(dict[str, Callable[[Invocation], None]]):
    """The players of the recording runner, and one for every run of `eval run`."""

    def __init__(
        self,
        players: Mapping[str, Callable[[Invocation], None]],
        run: Callable[[Invocation], None],
    ) -> None:
        super().__init__(players)
        self._run = run

    def get(self, key: str, default: Any = None) -> Any:
        if key.startswith("eval run "):
            return self._run
        return super().get(key, default)


class EvalRunner(RecordingRunner):
    """The recording runner of `tiny_run`, with the evaluation commands played (see the module's text)."""

    def __init__(
        self,
        *,
        chrf: Mapping[tuple[str, str, str], float] | None = None,
        edit: EditText | None = None,
        robustness: Mapping[str, float | None] | None = None,
        ignore_scores: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self._chrf = dict(chrf or {})
        self._edit = edit
        self._robustness_values = dict(robustness or {})
        self._ignore_scores = ignore_scores

    @staticmethod
    def _key(call: Invocation) -> str:
        key = RecordingRunner._key(call)
        if key == "eval run":
            suite = Path(call.options.get("--suite", ["?"])[0]).stem
            return f"eval run {suite} {call.options.get('--system', ['?'])[0]}"
        return key

    def _players(self) -> dict[str, Callable[[Invocation], None]]:
        players = super()._players()
        players["eval robustness"] = self._eval_robustness
        players["eval report"] = self._eval_report
        return _EvalPlayers(players, self._eval_run)

    # ---- eval robustness ------------------------------------------------------------------------

    def _eval_robustness(self, call: Invocation) -> None:
        suite_path = require_file(call.path("--suite"))
        try:
            suite, suite_sha = robust_suite_reader.read_suite(suite_path)
        except DataError as error:
            raise Refused("the suite is not valid") from error
        require_file(require_dir(call.path("--model")) / "model.bin")
        call.number("--beam-size")
        require_dir(call.path("--script-converter-ct2"))
        require_file(call.path("--script-converter-spm"))
        out = call.path("--out-dir")
        self._require_apart(out, suite_path)
        out.mkdir(parents=True, exist_ok=True)
        document = copy.deepcopy(json.loads(recorded_text("robustness_report.json")))
        document["robustness"].update(self._robustness_values)
        document["suite"] = {"name": suite.name, "sha256": suite_sha}
        (out / "robustness.json").write_text("{}\n", encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _require_apart(out: Path, source: Path) -> None:
        if out.resolve() == source.resolve().parent:
            raise Refused("the output directory is the directory of an input")

    # ---- eval run -------------------------------------------------------------------------------

    def _eval_run(self, call: Invocation) -> None:
        suite_path = require_file(call.path("--suite"))
        suite, suite_sha = _read_suite(suite_path)
        model, hypotheses = call.has("--model"), call.has("--hypotheses")
        if model == hypotheses:
            raise Refused("exactly one of --model and --hypotheses")
        if call.has("--resamples") and call.number("--resamples") < MIN_RESAMPLES:
            raise Refused("--resamples is below 100")
        system = call.one("--system")
        out = call.path("--out-dir")
        self._require_apart(out, suite_path)
        robustness = (
            require_file(require_dir(call.path("--robustness")) / "report.json")
            if call.has("--robustness")
            else None
        )
        run_dir = out / suite.name / system
        if model:
            self._run_model(call, suite, system, run_dir, robustness)
        else:
            self._run_hypotheses(call, suite, suite_sha, system, run_dir)

    def _run_model(
        self,
        call: Invocation,
        suite: suite_reader.Suite,
        system: str,
        run_dir: Path,
        robustness: Path | None,
    ) -> None:
        if call.has("--segment-scores"):
            raise Refused("scores go with --hypotheses only")
        directory = require_dir(call.path("--model"))
        require_file(directory / "model.bin")
        call.number("--beam-size")
        quantization = directory.name
        rows = self._outputs(suite, system, quantization)
        extras = (
            {}
            if robustness is None
            else {
                "robustness": json.loads(robustness.read_text("utf-8"))["robustness"],
                "robustness_coverage": json.loads(robustness.read_text("utf-8"))[
                    "coverage"
                ],
            }
        )
        self._write_run(suite, system, quantization, run_dir, rows, extras, {})

    def _run_hypotheses(
        self,
        call: Invocation,
        suite: suite_reader.Suite,
        suite_sha: str,
        system: str,
        run_dir: Path,
    ) -> None:
        if call.has("--beam-size"):
            raise Refused("decode options go with --model only")
        path = require_file(call.path("--hypotheses"))
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        expected = [OutputKey(row["direction"], row["id"]) for row in rows]
        scores: dict[str, dict[OutputKey, float]] = {}
        for entry in call.many("--segment-scores"):
            metric, _, file = entry.partition("=")
            try:
                scores[metric] = read_scores(
                    require_file(Path(file)).read_bytes(),
                    metric=metric,
                    suite_sha256=suite_sha,
                    hypotheses_sha256=digest(path),
                    expected=expected,
                )
            except (DataError, ValueError) as error:
                raise Refused("the scores are not valid") from error
        grouped = {
            metric: self._mean_by_direction(values) for metric, values in scores.items()
        }
        if self._ignore_scores:
            grouped = {}
        quantization = quantization_of(system)
        self._write_run(suite, system, quantization, run_dir, rows, {}, grouped)

    @staticmethod
    def _mean_by_direction(scores: Mapping[OutputKey, float]) -> dict[str, float]:
        by_direction: dict[str, list[float]] = {}
        for key, value in scores.items():
            by_direction.setdefault(key.direction, []).append(value)
        return {
            direction: sum(values) / len(values)
            for direction, values in by_direction.items()
        }

    def _outputs(
        self, suite: suite_reader.Suite, system: str, quantization: str
    ) -> list[dict[str, str]]:
        recorded_name = HYPOTHESES_FILES.get((suite.name, quantization))
        recorded = recorded_hypotheses(recorded_name) if recorded_name else {}
        rows: list[dict[str, str]] = []
        for direction in suite.directions:
            view = suite.view(direction)
            for row_id, reference in zip(view.ids, view.references, strict=True):
                text = recorded.get((direction, row_id), reference)
                if self._edit is not None:
                    text = self._edit(system, suite.name, direction, row_id, text)
                rows.append(
                    {"direction": direction, "id": row_id, "status": "ok", "text": text}
                )
        return rows

    def _write_run(
        self,
        suite: suite_reader.Suite,
        system: str,
        quantization: str,
        run_dir: Path,
        rows: Sequence[Mapping[str, str]],
        extras: Mapping[str, Any],
        scores: Mapping[str, Mapping[str, float]],
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = (
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) for row in rows
        )
        (run_dir / "hypotheses.jsonl").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        for direction in suite.directions:
            count = sum(1 for row in rows if row["direction"] == direction)
            if not count:
                continue
            document = copy.deepcopy(recorded_result(quantization))
            document.update(
                direction=direction,
                suite=suite.name,
                system=system,
                n_segments=count,
                statuses={"ok": count},
                **extras,
            )
            document["bootstrap"]["units"] = count
            document["metrics"]["chrf"] = self._chrf.get(
                (suite.name, system, direction), document["metrics"]["chrf"]
            )
            for metric, by_direction in scores.items():
                document["metrics"][metric] = by_direction[direction]
                document["unmeasured"].pop(metric, None)
            result = run_dir / direction
            result.mkdir(parents=True, exist_ok=True)
            (result / "result.json").write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    # ---- eval report ----------------------------------------------------------------------------

    def _eval_report(self, call: Invocation) -> None:
        suites = [_read_suite(Path(path))[0] for path in call.many("--suite")]
        runs = [require_dir(Path(path)) for path in call.many("--run")]
        if not suites or not runs:
            raise Refused("a report needs a run and a suite")
        for run in runs:
            require_file(run / "hypotheses.jsonl")
        out = call.path("--out-dir")
        for path in call.many("--suite"):
            self._require_apart(out, Path(path))
        out.mkdir(parents=True, exist_ok=True)
        document = {
            "schema": "rinchen.report/1",
            "suites": sorted(suite.name for suite in suites),
            "runs": len(runs),
        }
        (out / "report.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        (out / "report.md").write_text("# report\n", encoding="utf-8")


# ---- the inputs of the stages ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_outputs(
    tmp_path_factory: pytest.TempPathFactory, library_outputs: Path
) -> Iterator[Path]:
    """The output of the corpus, filter and eight model stages, made once for a module."""
    base = tmp_path_factory.mktemp("model")
    copy_inputs(library_outputs, base)
    with offline_in(base):
        results = run_stages(context(base), "parity")
    assert [r.record.state for r in results.values()] == ["ok"] * 8
    yield base


@pytest.fixture(scope="module")
def evaluated_inputs(
    tmp_path_factory: pytest.TempPathFactory, model_outputs: Path
) -> Iterator[Path]:
    """`model_outputs` with the robustness stage run over the recording runner."""
    base = tmp_path_factory.mktemp("robust") / "out"
    shutil.copytree(model_outputs, base)
    with offline_in(base.parent):
        result = run_robustness(context(base, runner=EvalRunner()))
    assert result.record.state == "ok"
    yield base


@pytest.fixture
def eval_out(tmp_path: Path, evaluated_inputs: Path) -> Path:
    """An output directory that holds everything before the evaluation, for one test."""
    out = tmp_path / "out"
    shutil.copytree(evaluated_inputs, out)
    return out


def hypotheses_of(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def run_evaluation(
    out: Path,
    *,
    qe_fake: bool = True,
    config: TinyConfig | None = None,
    fixtures: Fixtures | None = None,
    remaining: Callable[[], float] | None = None,
    **options: Any,
) -> tuple[StageResult, EvalRunner]:
    """The evaluation stage over `out`, with a runner that plays every command (`options` go to it)."""
    runner = EvalRunner(**options)
    ctx = dataclasses.replace(
        context(out, runner=runner, config=config, remaining=remaining),
        qe_fake=qe_fake,
    )
    if fixtures is not None:
        ctx = dataclasses.replace(ctx, fixtures=fixtures)
    return run_evaluate(ctx), runner
