"""Shared setup of the tests of `rinchen eval report` (D13, Task 8).

A `ReportWorkspace` is the golden runs as `rinchen eval run` writes them: the workspace of
`tests.support.eval_cli` (copies of the golden suite and of the hypotheses of `alpha` and `beta`)
and, made through the real command, the run directories `<out>/golden/alpha` and
`<out>/golden/beta`. `report_out` does not exist. Nothing here loads a model, opens a socket or
touches a GPU: a hypotheses file is the system, so the engine is never opened.

`rewrite_run_file` changes one file of a run directory and makes the manifest of that run vouch
for the new bytes (and, for a file `run.json` lists, `run.json` too, then the manifest again), so
that a test can put the fault exactly where it wants it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from click.testing import Result

from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.eval_cli import EvalWorkspace, invoke, make_workspace

FIRST: Final = "khk_Cyrl-eng_Latn"
SECOND: Final = "eng_Latn-khk_Cyrl"
MANIFEST: Final = "manifest.json"
RUN_JSON: Final = "run.json"
MARKER: Final = "SECRET-MARKER-52"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReportWorkspace:
    """The workspace of `eval run` and the two run directories it made."""

    ws: EvalWorkspace
    alpha: Path
    beta: Path

    @property
    def suite(self) -> Path:
        return self.ws.suite

    @property
    def report_out(self) -> Path:
        return self.ws.root / "report"

    def args(
        self,
        *extra: str,
        runs: Sequence[Path] | None = None,
        suites: Sequence[Path] | None = None,
        out_dir: Path | None = None,
        baseline: str | None = "alpha",
    ) -> list[str]:
        """`eval report --run A --run B --suite S --out-dir O --baseline alpha` and `extra`."""
        chosen_runs = [self.alpha, self.beta] if runs is None else list(runs)
        chosen_suites = [self.suite] if suites is None else list(suites)
        arguments = ["eval", "report"]
        for run in chosen_runs:
            arguments += ["--run", str(run)]
        for suite in chosen_suites:
            arguments += ["--suite", str(suite)]
        arguments += ["--out-dir", str(self.report_out if out_dir is None else out_dir)]
        if baseline is not None:
            arguments += ["--baseline", baseline]
        return [*arguments, *extra]


def make_report_workspace(root: Path) -> ReportWorkspace:
    """The golden runs of `alpha` and `beta`, each made by `rinchen eval run --hypotheses`."""
    ws = make_workspace(root)
    for hypotheses in (ws.alpha, ws.beta):
        made = invoke(ws.hypotheses_args(hypotheses=hypotheses))
        assert made.exit_code == 0, made.output
    return ReportWorkspace(
        ws, ws.out_dir / "golden" / "alpha", ws.out_dir / "golden" / "beta"
    )


def invoke_report(workspace: ReportWorkspace, *extra: str, **options: Any) -> Result:
    return invoke(workspace.args(*extra, **options))


def copy_run(run: Path, destination: Path) -> Path:
    """A copy of a run directory somewhere else: the files are the run's and its paths are
    its own."""
    shutil.copytree(run, destination)
    return destination


def rewrite_manifest(run: Path, change: Callable[[Manifest], None]) -> None:
    manifest = store.read(run)
    assert manifest is not None
    change(manifest)
    store.write(run, manifest)


def vouch_for(run: Path, name: str) -> None:
    """Make the manifest of `run` list the current bytes of the file `name` (by its file name)."""
    path = run / name

    def update(manifest: Manifest) -> None:
        manifest.outputs = [
            item.model_copy(
                update={
                    "sha256": sha256_of(path),
                    "bytes": path.stat().st_size,
                }
            )
            if Path(item.path).name == path.name
            else item
            for item in manifest.outputs
        ]

    rewrite_manifest(run, update)


def rewrite_run_json(run: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Change `run.json` and make the manifest vouch for the result."""
    path = run / RUN_JSON
    document = json.loads(path.read_text("utf-8"))
    edit(document)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "utf-8",
    )
    vouch_for(run, RUN_JSON)


def listing(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )
