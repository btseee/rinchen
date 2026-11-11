"""Helpers of the D13 STOP gate (`tests/integration/test_eval_gate.py`, `test_eval_gate_controls.py` and
`test_release_gate.py`): run the real `rinchen eval ...` and `rinchen release ...` binaries, read what
they wrote, make the controls the gate needs and hold the floors.

Every command runs in a process that hides every device and reaches no network. `run_eval`,
`run_release` and `run_robustness` REFUSE to start unless `CUDA_VISIBLE_DEVICES` is the empty string,
`HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` are `1` and no `HF_TOKEN` is in the environment (the session
fixtures `d11_gpu_hidden` and `d13_offline` of `tests/integration/conftest.py` set that), and a test can
change the environment of ONE command (a hash seed, a locale) only for a variable that is not one of
those: `changed_environment` refuses the protected ones. Every command also runs with the Hub library
(`huggingface_hub`) and `datasets` BLOCKED: the session fixture `d13_offline` makes a directory whose two
packages raise `ImportError` when imported and `run_eval`, `run_release` and `run_robustness` put it first on
the `PYTHONPATH` of the command, so a command that so much as imports the library fails instead of making a
client. The only process these files start is the one
`distill_gate.run_rinchen` starts, bounded by `SUBPROCESS_TIMEOUT_SECONDS` (900 s). Nothing here asks a
device question, opens a socket or creates a Hub client.

THE FLOORS below are literals of this module and are never lowered: a miss of a model floor is answered
with more steps of the toy (`ROBUST_TEACHER_STEPS`, deterministic), never by editing a number. The
measured values they stand under are in the docstrings of the gate modules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.distill_gate import Invocation, read_json, run_rinchen
from tests.support.eval_toy import Suites
from tests.support.robust_gate import run_robustness as _run_robustness

__all__ = [
    "COMET_AND_METRICX",
    "DEVTEST_SUITE",
    "EVERY_NTH_TEXT",
    "HUB_BLOCKED_VARIABLE",
    "HYPOTHESES_FILE",
    "MAX_CI_WIDTH",
    "MAX_DELTA_OF_PERTURBED",
    "MAX_P_OF_A_SEEN_DIFFERENCE",
    "MAX_TEACHER_CHRF_PP",
    "MAX_UPPER_END_OF_PERTURBED",
    "MIN_BLEU",
    "MIN_CHRF_PP",
    "MIN_DELTA_OVER_TEACHER",
    "MIN_LOWER_END_OF_PERTURBED",
    "MIN_NUMBERS_ACC",
    "NINE_NAMES",
    "NOT_TRAINED_DIRECTIONS",
    "PROTECTED_VARIABLES",
    "REPO_ID",
    "TRAINED_DIRECTIONS",
    "Report",
    "Runs",
    "absolute_paths_in",
    "block_hub_libraries",
    "card_decimals",
    "card_section",
    "changed_environment",
    "comparison",
    "environment_problems",
    "expected_plan",
    "make_report",
    "make_runs",
    "network_findings",
    "perturb_outputs",
    "read_result",
    "read_run",
    "report_numbers",
    "run_eval",
    "run_release",
    "run_robustness",
]

# ---- the floors: literals, never lowered ---------------------------------------------------------

# R2: the export, per trained direction (measured 97.03, 99.05, 98.97, 98.38 chrF++ and 95.84,
# 98.12, 98.15, 96.83 BLEU), each interval narrower than 5.0 and holding its value.
MIN_CHRF_PP: Final = 95.0
MIN_BLEU: Final = 93.0
MAX_CI_WIDTH: Final = 5.0
# R3: the D11 teacher, which never saw the sentences (measured 36.50, 46.09, 29.30, 31.67).
MAX_TEACHER_CHRF_PP: Final = 60.0
# R6: the export against the teacher (measured +60.5, +53.0, +69.7, +66.7), a difference the paired
# test must see (p at most 0.01; its smallest value at 1,000 resamples is 0.001998).
MIN_DELTA_OVER_TEACHER: Final = 40.0
MAX_P_OF_A_SEEN_DIFFERENCE: Final = 0.01
# R6: the sensitivity control, one text in EVERY_NTH_TEXT replaced by the next row's.
EVERY_NTH_TEXT: Final = 20
MAX_DELTA_OF_PERTURBED: Final = -2.0
MAX_UPPER_END_OF_PERTURBED: Final = -0.5
MIN_LOWER_END_OF_PERTURBED: Final = 0.5
# R8: D12's own floor for the number accuracy of the robustness block.
MIN_NUMBERS_ACC: Final = 0.90

# ---- what the toy is ------------------------------------------------------------------------------

DEVTEST_SUITE: Final = "toy-devtest"
HYPOTHESES_FILE: Final = "hypotheses.jsonl"
# The directions the D12 export and the D11 teacher are both trained for, in the suite's order.
TRAINED_DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
NOT_TRAINED_DIRECTIONS: Final = ("eng_Latn-rus_Cyrl", "rus_Cyrl-eng_Latn")
COMET_AND_METRICX: Final = ("comet22", "metricx24", "metricx24_qe")
# The nine files of an upload, in the order the plan lists them (written out here, not read from
# the library: a test compares the two).
NINE_NAMES: Final = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
    "README.md",
    "report.json",
    "report.md",
    "release.json",
)
REPO_ID: Final = "rinchen-test/rinchen-toy"
# What the environment of every command must be, and what a test may not change.
_REQUIRED_ENVIRONMENT: Final = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
_NO_TOKEN: Final = "HF_TOKEN"
# The variable that names the directory of the blocked libraries (set by the session fixture) and
# the libraries it blocks: the Hub client and the dataset loader, the two that reach for the network.
HUB_BLOCKED_VARIABLE: Final = "D13_GATE_HUB_BLOCKED"
_BLOCKED_LIBRARIES: Final = ("huggingface_hub", "datasets")
_BLOCK_MARKER: Final = "blocked: the gate never imports this library"
PROTECTED_VARIABLES: Final = (
    *_REQUIRED_ENVIRONMENT,
    _NO_TOKEN,
    HUB_BLOCKED_VARIABLE,
    "PYTHONPATH",
)


# ---- the environment and the commands -------------------------------------------------------------


def block_hub_libraries(directory: Path) -> Path:
    """Make `directory` (when missing) hold a package for each library of `_BLOCKED_LIBRARIES` whose
    import raises `ImportError`, and return it: put first on a command's `PYTHONPATH` it shadows the
    installed library."""
    for library in _BLOCKED_LIBRARIES:
        package = directory / library
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(
            f'raise ImportError("{_BLOCK_MARKER}")\n', encoding="utf-8", newline="\n"
        )
    return directory


def _read_text(path: Path) -> str:
    try:
        return path.read_text("utf-8")
    except OSError:
        return ""


def _blocked_directory() -> Path | None:
    """The directory of the blocked libraries, when the environment names one that holds them."""
    named = os.environ.get(HUB_BLOCKED_VARIABLE)
    if not named:
        return None
    directory = Path(named)
    holds_them = all(
        _BLOCK_MARKER in _read_text(directory / library / "__init__.py")
        for library in _BLOCKED_LIBRARIES
    )
    return directory if holds_them else None


def environment_problems() -> list[str]:
    """What is wrong with the environment of this process for a command of the gate: a device that
    is not hidden, a Hub switch that is off, a token, or Hub libraries that are not blocked. A token's
    value is never named."""
    problems = [
        f"{name} must be {wanted!r}, not {os.environ.get(name)!r}"
        for name, wanted in _REQUIRED_ENVIRONMENT.items()
        if os.environ.get(name) != wanted
    ]
    if _NO_TOKEN in os.environ:
        problems.append(f"{_NO_TOKEN} must not be set")
    if _blocked_directory() is None:
        problems.append(
            f"{HUB_BLOCKED_VARIABLE} must name a directory of blocked Hub libraries"
        )
    return problems


@contextmanager
def changed_environment(variables: Mapping[str, str | None]) -> Iterator[None]:
    """Set (or, for None, remove) `variables` for the commands started inside the block. The
    variables that hide the devices and the Hub are protected and cannot be changed."""
    protected = sorted(set(variables) & set(PROTECTED_VARIABLES))
    assert not protected, (
        f"{', '.join(protected)} cannot be changed: they are protected"
    )
    with pytest.MonkeyPatch.context() as patch:
        for name, value in variables.items():
            if value is None:
                patch.delenv(name, raising=False)
            else:
                patch.setenv(name, value)
        yield


@contextmanager
def _hub_blocked() -> Iterator[None]:
    """The blocked libraries first on the `PYTHONPATH` of the commands started inside the block."""
    blocked = _blocked_directory()
    assert blocked is not None, "the Hub libraries are not blocked"
    existing = os.environ.get("PYTHONPATH")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "PYTHONPATH",
            str(blocked) if not existing else f"{blocked}{os.pathsep}{existing}",
        )
        yield


def _run(
    group: str, args: Sequence[str | Path], environ: Mapping[str, str | None] | None
) -> Invocation:
    problems = environment_problems()
    assert not problems, "; ".join(problems)
    with changed_environment(environ or {}), _hub_blocked():
        return run_rinchen(group, *args)


def run_eval(
    *args: str | Path, environ: Mapping[str, str | None] | None = None
) -> Invocation:
    """`rinchen eval <args>` through the real binary, to its end."""
    return _run("eval", args, environ)


def run_release(
    *args: str | Path, environ: Mapping[str, str | None] | None = None
) -> Invocation:
    """`rinchen release <args>` through the real binary, to its end."""
    return _run("release", args, environ)


def run_robustness(*args: str | Path) -> Invocation:
    """D12's `rinchen eval robustness <args>`, started only in the environment of the gate."""
    problems = environment_problems()
    assert not problems, "; ".join(problems)
    with _hub_blocked():
        return _run_robustness(*args)


# ---- the runs and the reports ---------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    """The runs of the gate under one output root: the export (`a-model`), the D11 teacher
    (`d11-teacher`) and the export's outputs fed back as a file (`a-file`), and what each command
    printed."""

    root: Path
    invocations: Mapping[str, Invocation]

    def run_dir(self, system: str) -> Path:
        return self.root / DEVTEST_SUITE / system

    @property
    def model(self) -> Path:
        return self.run_dir("a-model")

    @property
    def teacher(self) -> Path:
        return self.run_dir("d11-teacher")

    @property
    def file(self) -> Path:
        return self.run_dir("a-file")


def make_runs(suites: Suites, export: Path, teacher: Path, root: Path) -> Runs:
    """`rinchen eval run` over the devtest suite for the D12 export, the D11 teacher and the
    outputs of the export as a file, each under `root`. Every command must exit 0."""
    invocations: dict[str, Invocation] = {}
    for system, source in (
        ("a-model", ("--model", export)),
        ("d11-teacher", ("--model", teacher)),
        (
            "a-file",
            ("--hypotheses", root / DEVTEST_SUITE / "a-model" / HYPOTHESES_FILE),
        ),
    ):
        invocation = run_eval(
            "run",
            "--suite",
            suites.devtest,
            *source,
            "--system",
            system,
            "--out-dir",
            root,
        )
        assert invocation.returncode == 0, (system, invocation.stderr)
        invocations[system] = invocation
    return Runs(root=root, invocations=invocations)


def read_run(run_dir: Path) -> dict[str, Any]:
    """`run.json` of a run directory."""
    return read_json(run_dir / "run.json")


def read_result(run_dir: Path, direction: str) -> dict[str, Any]:
    """`result.json` of a direction of a run directory."""
    return read_json(run_dir / direction / "result.json")


@dataclass(frozen=True)
class Report:
    """A report that was made: where it is, its parsed `report.json`, its Markdown and what the
    command printed."""

    directory: Path
    document: dict[str, Any]
    markdown: str
    invocation: Invocation


def make_report(
    runs: Sequence[Path],
    suites: Sequence[Path],
    out_dir: Path,
    *,
    baseline: str,
    environ: Mapping[str, str | None] | None = None,
) -> Report:
    """`rinchen eval report` over the run directories and suite files, against `baseline`. The
    command must exit 0."""
    args: list[str | Path] = ["report"]
    for run in runs:
        args += ["--run", run]
    for suite in suites:
        args += ["--suite", suite]
    args += ["--baseline", baseline, "--out-dir", out_dir]
    invocation = run_eval(*args, environ=environ)
    assert invocation.returncode == 0, invocation.stderr
    return Report(
        directory=out_dir,
        document=read_json(out_dir / "report.json"),
        markdown=(out_dir / "report.md").read_text("utf-8"),
        invocation=invocation,
    )


def comparison(
    report: Report, system: str, direction: str, metric: str = "chrf_pp"
) -> dict[str, Any]:
    """The one comparison of `system` in `direction` on `metric` that the report holds."""
    found = [
        item
        for item in report.document["comparisons"]
        if (item["system"], item["direction"], item["metric"])
        == (system, direction, metric)
    ]
    assert len(found) == 1, (system, direction, metric, len(found))
    return found[0]


# ---- the sensitivity control ------------------------------------------------------------------------


def perturb_outputs(path_in: Path, path_out: Path, every: int = EVERY_NTH_TEXT) -> None:
    """Write the outputs file `path_in` to `path_out` with, inside each direction, the text of
    every `every`-th line (the first, the `every + 1`-th, ...) replaced by the next line's (the last
    line of a direction takes the first one's). The keys and the statuses stay, so every request is
    still answered and only what the answer says is another row's. The file keeps the writer's
    form: sorted keys, one JSON object per line."""
    lines = [json.loads(line) for line in path_in.read_text("utf-8").splitlines()]
    members: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        members[line["direction"]].append(line)
    for group in members.values():
        texts = [line["text"] for line in group]
        for position, line in enumerate(group):
            if position % every == 0:
                line["text"] = texts[(position + 1) % len(group)]
    path_out.write_text(
        "".join(
            json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n"
            for line in lines
        ),
        encoding="utf-8",
        newline="\n",
    )


# ---- reading what was written -----------------------------------------------------------------------

_POSIX_PATH: Final = re.compile(
    r"(?:^|[\s\"'`(\[{<=,;:|])/[A-Za-z0-9._~-]+/[A-Za-z0-9._~-]"
)
_DRIVE_PATH: Final = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\")


def absolute_paths_in(text: str) -> list[str]:
    """The absolute local paths `text` holds (a POSIX path of two components or more after a
    delimiter, or a drive path), as the places that matched, never longer than 40 characters. A
    slash inside a word (`and/or`, `org/name`) is not one."""
    found = [match.group(0).strip()[:40] for match in _POSIX_PATH.finditer(text)]
    return found + [match.group(0) for match in _DRIVE_PATH.finditer(text)]


def expected_plan(stage: Path, repo_id: str) -> bytes:
    """`hub_plan.json` as the disk gives it for the nine files of `stage`, written out here with
    the standard library and nothing of the code under test."""
    contents = {name: (stage / name).read_bytes() for name in NINE_NAMES}
    document = {
        "schema": "rinchen.hub_plan/1",
        "repo_id": repo_id,
        "release_sha256": hashlib.sha256(contents["release.json"]).hexdigest(),
        "total_bytes": sum(len(data) for data in contents.values()),
        "files": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in contents.items()
        ],
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    return f"{text}\n".encode()


def card_section(card: str, heading: str) -> str:
    """The text of the `## <heading>` section of a card, up to the next `## `."""
    lines = card.splitlines()
    start = lines.index(f"## {heading}")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end])


def card_decimals(card: str) -> list[str]:
    """Every decimal number of the tables of the Evaluation and Robustness sections of a card."""
    found: list[str] = []
    for heading in ("Evaluation", "Robustness"):
        for line in card_section(card, heading).splitlines():
            if line.startswith("| "):
                found += re.findall(r"[+-]?\d+\.\d+", line)
    return found


def report_numbers(document: Mapping[str, Any]) -> set[str]:
    """The numbers of a report formatted as a card formats them: scores and interval ends with two
    decimals, deltas and their ends with a sign, p-values with three, robustness values with four."""
    numbers: set[str] = set()
    for result in document["results"]:
        for metric, value in result["metrics"].items():
            if value is not None:
                numbers.add(f"{value:.2f}")
                low, high = result["ci95"][metric]
                numbers.update((f"{low:.2f}", f"{high:.2f}"))
    for item in document["comparisons"]:
        numbers.add(f"{item['delta']:+.2f}")
        if item["ci95"] is not None:
            numbers.update(f"{end:+.2f}" for end in item["ci95"])
        if item["p_value"] is not None:
            numbers.add(f"{item['p_value']:.3f}")
    for entry in document["robustness"].values():
        numbers.update(
            f"{value:.4f}" for value in entry["block"].values() if value is not None
        )
    return numbers


# ---- the hygiene scan ---------------------------------------------------------------------------------

_NETWORK_MODULES: Final = ("socket", "urllib", "requests", "httpx", "http")
_NETWORK_CALLS: Final = ("httpx.Client", "httpx.AsyncClient")


def network_findings(path: Path) -> list[str]:
    """What the Python file `path` imports or calls that reaches for the network: the `socket`,
    `urllib`, `requests`, `httpx` and `http` modules and a client of `httpx`. The file is read, not
    imported; a string that only names a module is not an import."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found += [
                f"line {node.lineno}: import of {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in _NETWORK_MODULES
            ]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _NETWORK_MODULES:
                found.append(f"line {node.lineno}: import from {node.module}")
        elif isinstance(node, ast.Call) and ast.unparse(node.func) in _NETWORK_CALLS:
            found.append(f"line {node.lineno}: a call of {ast.unparse(node.func)}")
    return found
