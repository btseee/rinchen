"""The D14 STOP gate, the night and the export (R1 to R6): the command of the nightly workflow, read from
`.github/workflows/nightly.yml` and run for real, and the parity rerun over an operator's export, through the
real `rinchen` binary, Eole, CTranslate2 and pytest on the CPU with every device hidden. The controls (R7 to R10)
are in `test_maintain_gate_controls.py`.

The `uv` the command finds is a shim (`tests/support/fake_uv.py`): `lock --check` and `--version` go to the REAL
uv (`lock --check --offline`), `run` goes to the real uv as `run --project <repository> --no-sync --frozen
--offline`, and `lock --upgrade --dry-run`, the one call of a night that reaches a package index, is answered
from a HAND-WRITTEN sample (`CHANGES_HAND_WRITTEN` of `tests/support/uv_outputs.py`) and never forwarded: the
lines uv prints for an update, an add and a remove were never measured. The gate cannot prove the absence of a
socket in a child process; what keeps it off the network is the shim, the offline switches and the empty device
list, and it checks the shim's call log and the bytes of the two project files.

Gates (every floor a literal of `tests/support/maintain_gate.py` and never lowered):

    R1  `sh -c` of the workflow's step (`uv run --frozen --no-sync rinchen maintain check --out-dir
        maintain-out --cadence "$CADENCE"`, `CADENCE=nightly`, in a temporary directory) exits 0 and writes a
        report of schema `rinchen.maintain/1`, cadence `nightly`, state `drift`, the six steps in order; the
        doctor's six CPU checks `ok` and the GPU check `skipped`; the lock figures equal what the lock says;
        `lock_consistent` and `environment` `ok`; the drift table of the sample (7 entries, 5 updates, 1 add, 1
        remove, 3 free, 4 blocked: `huggingface-hub`, `numpy`, `nvidia-cudnn-cu13`, `safetensors`); the
        `numerics` rerun at least 1,100 passed and `parity_tiny` exactly 13, none failed or skipped;
        `parity_export` skipped (no export) and the three weekly reruns skipped (not due)
    R2  the manifest: stage `maintain_check`, `tool_versions` holding every name of the lock with its versions
        and `python` and `uv`, the three inputs with the digests of their bytes, every output listed with its
        digest, and the six states of the steps
    R3  neither document holds an absolute path, a timestamp, a host name, a value of HOME, USER or PATH, the
        output directory or a line of the hand-written sample or of uv's own output
    R4  `pyproject.toml` and `uv.lock` have the same bytes and modification times after the runs; the shim's
        call log shows the dry run local (`forwarded` false), the real uv called as `lock --check --offline`,
        `--version` and `run --project ... --no-sync --frozen --offline ...` once each, no flag twice, no refusal
    R5  the command run twice gives byte-equal `maintain_report.json` and `maintain_report.md` (the second time
        under another hash seed)
    R6  `maintain check --no-drift --rerun parity_export --export E --checkpoint C --corpus F` over the toy's
        float32 export: exit 0, the rerun `ok` with every sentence exact and the mean |delta| under 1e-3, the
        other reruns skipped (the selection wins over the cadence); with one byte of `model.bin` flipped, or with
        the checkpoint of another model, the rerun is `fail` (`the command exited with code 2`), the exit is 4
        and everything is written

Measured on this machine (CPU, Eole 0.6.4, CTranslate2 4.8.4, uv 0.13.7, 2026-11-11; a time moves by a third
between runs, a count never does):

    R1  the command of the workflow ran twice, exit 0 both times, in 163.6 s (a first look) and
           about 150 s each inside the gate (the manifest's seconds: doctor 13.2, the lock and the dry-run
           steps 0.1, the two reruns 136.2); numerics 1,110 passed (floor 1,100), `parity_tiny` 13 passed (the
           toy's two teachers train inside it); the lock: 187 entries, 186 names (`pyonmttok` twice at one
           version), format version 1 revision 5; the environment 186 installed, none differing, none extra;
           the table 7 entries, 5 updates, 1 add, 1 remove, 3 free and 4 blocked
    R2  `tool_versions` 188 keys (186 names, `python` 3.14.10, `uv` 0.13.7); 8 outputs listed (the two
           documents of 6,138 and 1,806 bytes, four logs, two JUnit reports), none missing, none extra
    R3  both documents free of paths, dates and times, the host, HOME, USER, PATH and the output directory
    R4  four calls of the shim per run (the `run`, the lock check, the dry run answered locally, the version),
           none refused; `uv.lock` (`559b3954...`) and `pyproject.toml` with the same bytes and the same
           modification times after both runs
    R5  the two documents byte-equal between the run and its repeat under `PYTHONHASHSEED=3`
    R6  200 of 200 sentences exact (50 per direction), mean |delta| 7.19e-07 (max 3.43e-06) against the
           bound 1e-3, each command about 14 s (the doctor 13 s of it); the flipped byte and the checkpoint of
           another model: the parity command exits 2, the rerun is `fail`, the check exits 4
    time   this module's 27 tests took about 460 s (the night's two runs 150 s each, the toy teachers of
           the D11 session fixtures 133 s)

UNMEASURED: GitHub Actions (the workflow was never run there), the real state of PyPI, the lines uv prints for a
change, anything on a GPU and every real model. The toy ranks nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.maintain.report import STEP_IDS
from tests.support.d11_gate import Exports, Teachers
from tests.support.distill_gate import Invocation
from tests.support.distill_gate_stages import Workspace
from tests.support.fake_uv import read_calls, write_uv_shim
from tests.support.maintain_gate import (
    MAX_MEAN_ABS_DELTA,
    MIN_EXACT_SHARE,
    MIN_NUMERICS_PASSED,
    PARITY_TINY_PASSED,
    REPOSITORY,
    WORKFLOW,
    absolute_paths,
    file_identity,
    lock_table,
    project_files_unchanged,
    read_manifest,
    read_report,
    real_uv,
    rerun_item,
    run_maintain,
    run_workflow_command,
    step_of,
    timestamps,
    workflow_command,
)
from tests.support.nightly_workflow import CHECK_PREFIX
from tests.support.uv_outputs import CHANGES_HAND_WRITTEN, NO_CHANGE_MEASURED

pytestmark = [pytest.mark.subprocess, pytest.mark.usefixtures("d11_gpu_hidden")]

SHIPPED_CONFIG = REPOSITORY / "configs" / "maintain" / "maintain.yaml"
OUT_DIR_NAME = "maintain-out"
PROJECT_FILES = ("pyproject.toml", "uv.lock")
GPU_REASON = "a GPU check: maintain check never asks a device question"
NOT_DUE = "not due at the nightly cadence"
NOT_SELECTED = "not selected with --rerun"
CPU_CHECKS = [
    "D-PY",
    "D-WHEELS",
    "D-EOLE-PY",
    "D-EOLE-CLI",
    "D-CT2-CONV",
    "D-CT2-IMPORT",
]
RERUN_IDS = [
    "numerics",
    "parity_tiny",
    "parity_export",
    "robustness_tiny",
    "evaluation_tiny",
    "serving_tiny",
]
BLOCKED = ["huggingface-hub", "numpy", "nvidia-cudnn-cu13", "safetensors"]
SENTENCES_PER_DIRECTION = 50
DIRECTIONS = 4
THE_COMMAND = [
    "run",
    "--frozen",
    "--no-sync",
    "rinchen",
    "maintain",
    "check",
    "--out-dir",
    OUT_DIR_NAME,
    "--cadence",
    "nightly",
]


@pytest.fixture(autouse=True)
def the_project_files_keep_their_bytes_and_times() -> Iterator[None]:
    """No test of this module may change `pyproject.toml` or `uv.lock` of the repository."""
    with project_files_unchanged():
        yield


def _identities() -> dict[str, tuple[str, int]]:
    return {name: file_identity(REPOSITORY / name) for name in PROJECT_FILES}


# ---- the night: the command of the workflow, run twice ---------------------------------------------


@dataclass(frozen=True)
class NightRun:
    """One run of the workflow's command: what it printed, where it wrote, what its shim logged and
    the identity (bytes and modification time) of the two project files before and after it."""

    invocation: Invocation
    out_dir: Path
    shim_dir: Path
    before: dict[str, tuple[str, int]]
    after: dict[str, tuple[str, int]]

    @property
    def report(self) -> dict[str, Any]:
        return read_report(self.out_dir)

    @property
    def manifest(self) -> dict[str, Any]:
        return read_manifest(self.out_dir)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return read_calls(self.shim_dir)


def _night_run(root: Path, extra: dict[str, str] | None) -> NightRun:
    shim = write_uv_shim(
        root / "bin",
        real_uv=real_uv(),
        project=REPOSITORY,
        dry_run_output=CHANGES_HAND_WRITTEN,
    )
    cwd = root / "cwd"
    cwd.mkdir()
    before = _identities()
    invocation = run_workflow_command(
        shim.parent,
        workflow_command(WORKFLOW, "nightly"),
        cwd=cwd,
        cadence="nightly",
        tmpdir=root / "tmp",
        extra=extra,
    )
    return NightRun(invocation, cwd / OUT_DIR_NAME, shim.parent, before, _identities())


@pytest.fixture(scope="module")
def night(tmp_path_factory: pytest.TempPathFactory) -> NightRun:
    """The command of the workflow, run in a temporary directory of its own."""
    return _night_run(tmp_path_factory.mktemp("night_first"), None)


@pytest.fixture(scope="module")
def night_again(night: NightRun, tmp_path_factory: pytest.TempPathFactory) -> NightRun:
    """The same command again, in another directory and under another hash seed."""
    return _night_run(tmp_path_factory.mktemp("night_second"), {"PYTHONHASHSEED": "3"})


@pytest.fixture(scope="module")
def uv_version() -> str:
    """The version of the real uv, from its own `--version`."""
    return subprocess.run(
        [real_uv(), "--version"], capture_output=True, text=True, timeout=30, check=True
    ).stdout.split()[1]


# =================================================================================
# R1: the workflow's own command, natively.


def test_the_command_of_the_workflow_is_the_one_that_was_run(night: NightRun) -> None:
    command = workflow_command(WORKFLOW, "nightly")

    assert command.startswith(CHECK_PREFIX)
    assert command.endswith('--out-dir maintain-out --cadence "$CADENCE"')
    assert night.calls[0]["argv"] == THE_COMMAND


def test_the_night_exits_zero_and_says_drift_for_the_nightly_cadence(
    night: NightRun,
) -> None:
    assert night.invocation.returncode == 0, night.invocation.stderr
    assert night.invocation.stdout[0] == "maintain: drift (nightly)"
    report = night.report
    assert report["schema"] == "rinchen.maintain/1"
    assert report["cadence"] == "nightly"
    assert report["state"] == "drift"


def test_the_steps_are_the_six_of_the_check_in_order(night: NightRun) -> None:
    ids = [step["id"] for step in night.report["steps"]]

    assert ids == list(STEP_IDS)
    assert ids == [
        "doctor",
        "lock_file",
        "lock_consistent",
        "environment",
        "drift",
        "reruns",
    ]


def test_the_doctor_runs_the_six_cpu_checks_and_skips_the_gpu_check(
    night: NightRun,
) -> None:
    doctor = step_of(night.report, "doctor")
    rows = {check["id"]: check for check in doctor["detail"]["checks"]}

    assert doctor["state"] == "ok"
    assert list(rows) == [*CPU_CHECKS, "D-TORCH-CUDA"]
    for check_id in CPU_CHECKS:
        assert (rows[check_id]["state"], rows[check_id]["reason"]) == ("ok", None)
    assert (rows["D-TORCH-CUDA"]["state"], rows["D-TORCH-CUDA"]["reason"]) == (
        "skipped",
        GPU_REASON,
    )


def test_the_figures_of_the_lock_are_what_the_lock_file_says(night: NightRun) -> None:
    table = lock_table(REPOSITORY / "uv.lock")
    report = night.report

    assert report["lock"]["packages"] == table.package_lines
    assert report["lock"]["names"] == len(table.versions)
    assert report["lock"]["format"] == {
        "version": table.format_version,
        "revision": table.revision,
    }
    assert step_of(report, "lock_file")["state"] == "ok"


def test_the_lock_check_goes_to_the_real_uv_offline_and_the_lock_is_consistent(
    night: NightRun,
) -> None:
    checks = [call for call in night.calls if call["argv"][:1] == ["lock"]]
    forwarded = [call for call in checks if call["forwarded"]]

    assert step_of(night.report, "lock_consistent")["state"] == "ok"
    assert [call["argv"] for call in forwarded] == [["lock", "--check"]]
    assert forwarded[0]["final"] == [real_uv(), "lock", "--check", "--offline"]


def test_the_installed_packages_are_the_locked_ones(night: NightRun) -> None:
    environment = step_of(night.report, "environment")

    assert environment["state"] == "ok"
    assert environment["detail"]["mismatched"] == []
    assert environment["detail"]["extra"] == []
    assert environment["detail"]["installed"] > 0


def test_the_drift_table_is_the_hand_written_sample_classified_by_the_rules(
    night: NightRun,
) -> None:
    drift = step_of(night.report, "drift")
    entries = drift["detail"]["entries"]
    locked = lock_table(REPOSITORY / "uv.lock").versions

    assert drift["state"] == "drift"
    assert len(entries) == 7
    assert drift["detail"]["counts"] == {
        "update": 5,
        "add": 1,
        "remove": 1,
        "drift": 3,
        "blocked": 4,
    }
    assert [e["package"] for e in entries if e["class"] == "blocked"] == BLOCKED
    for entry in entries:
        if entry["change"] != "add":
            assert (entry["locked"],) == locked[entry["package"]]


def test_the_nightly_reruns_pass_and_the_others_say_why_they_did_not_run(
    night: NightRun,
) -> None:
    report = night.report
    items = step_of(report, "reruns")["detail"]["items"]
    numerics = rerun_item(report, "numerics")
    parity = rerun_item(report, "parity_tiny")

    assert [item["id"] for item in items] == RERUN_IDS
    assert numerics["state"] == "ok"
    assert numerics["tests"]["passed"] >= MIN_NUMERICS_PASSED
    assert (numerics["tests"]["failed"], numerics["tests"]["skipped"]) == (0, 0)
    assert parity["state"] == "ok"
    assert parity["tests"] == {
        "passed": PARITY_TINY_PASSED,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    export = rerun_item(report, "parity_export")
    assert (export["state"], export["reason"]) == ("skipped", "no export was given")
    for weekly in ("robustness_tiny", "evaluation_tiny", "serving_tiny"):
        item = rerun_item(report, weekly)
        assert (item["state"], item["reason"], item["cadence"]) == (
            "skipped",
            NOT_DUE,
            "weekly",
        )


def test_the_shipped_floors_of_the_nightly_reruns_are_not_below_the_floors_of_the_gate() -> (
    None
):
    shipped = yaml.safe_load(SHIPPED_CONFIG.read_text("utf-8"))
    floors = {
        rerun["id"]: rerun["min_passed"]
        for rerun in shipped["reruns"]
        if "min_passed" in rerun
    }

    assert floors["numerics"] >= MIN_NUMERICS_PASSED
    assert floors["parity_tiny"] >= PARITY_TINY_PASSED


# =================================================================================
# R2: the manifest.


def test_the_manifest_records_every_locked_version_and_the_tools(
    night: NightRun, uv_version: str
) -> None:
    versions = lock_table(REPOSITORY / "uv.lock").versions
    tools = night.manifest["tool_versions"]

    assert night.manifest["stage"] == "maintain_check"
    assert set(tools) == {*versions, "python", "uv"}
    for name, found in versions.items():
        assert tools[name] == ", ".join(found)
    assert tools["python"].startswith("3.14.")
    assert tools["uv"] == uv_version


def test_the_manifest_lists_the_three_inputs_with_the_digests_of_their_bytes(
    night: NightRun,
) -> None:
    manifest = night.manifest
    wanted = {
        "pyproject.toml": REPOSITORY / "pyproject.toml",
        "uv.lock": REPOSITORY / "uv.lock",
        "maintain.yaml": REPOSITORY / "configs" / "maintain" / "maintain.yaml",
    }
    listed = {Path(item["path"]).name: item for item in manifest["inputs"]}

    assert list(listed) == list(wanted)
    for name, path in wanted.items():
        assert Path(listed[name]["path"]) == path
        assert listed[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert listed[name]["bytes"] == path.stat().st_size
    assert manifest["config_sha256"] == listed["maintain.yaml"]["sha256"]
    report_inputs = night.report["inputs"]
    assert report_inputs == {
        "config": listed["maintain.yaml"]["sha256"],
        "lock": listed["uv.lock"]["sha256"],
        "pyproject": listed["pyproject.toml"]["sha256"],
    }


def test_every_output_is_listed_with_the_digest_it_has(night: NightRun) -> None:
    out = night.out_dir
    outputs = night.manifest["outputs"]
    listed = {Path(item["path"]).relative_to(out).as_posix(): item for item in outputs}
    on_disk = {
        path.relative_to(out).as_posix()
        for path in out.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }

    assert set(listed) == on_disk
    assert {
        "maintain_report.json",
        "maintain_report.md",
        "logs/drift.log",
        "logs/lock_consistent.log",
        "junit/numerics.xml",
        "junit/parity_tiny.xml",
    } <= on_disk
    for name, item in listed.items():
        data = (out / name).read_bytes()
        assert item["sha256"] == hashlib.sha256(data).hexdigest(), name
        assert item["bytes"] == len(data), name


def test_the_manifest_says_the_state_of_each_step(night: NightRun) -> None:
    manifest = night.manifest

    assert manifest["status"] == {
        "doctor": "ok",
        "lock_file": "ok",
        "lock_consistent": "ok",
        "environment": "ok",
        "drift": "drift",
        "reruns": "ok",
        "state": "drift",
    }
    assert manifest["seed"] == -1
    assert "created" in manifest
    assert manifest["metrics"]["reruns_run"] == 2.0


# =================================================================================
# R3: what the documents never hold.


def _texts(run: NightRun) -> dict[str, str]:
    return {
        name: (run.out_dir / name).read_text("utf-8")
        for name in ("maintain_report.json", "maintain_report.md")
    }


def test_neither_document_holds_a_path_a_timestamp_a_host_or_a_value_of_the_environment(
    night: NightRun,
) -> None:
    run = night
    secrets = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "PATH": os.environ["PATH"],
        "host": socket.gethostname(),
        "out dir": str(run.out_dir),
        "out dir name": run.out_dir.name,
        "working directory": run.out_dir.parent.parent.name,
    }

    for name, text in _texts(run).items():
        assert absolute_paths(text) == [], name
        assert timestamps(text) == [], name
        for label, value in secrets.items():
            if value:
                assert value not in text, (name, label)


def test_neither_document_holds_a_line_of_the_sample_or_of_the_output_of_uv(
    night: NightRun,
) -> None:
    lines = [
        *CHANGES_HAND_WRITTEN.splitlines(),
        *NO_CHANGE_MEASURED.splitlines(),
        "Resolved 189",
        "Resolved 187",
        "Using CPython",
    ]

    for name, text in _texts(night).items():
        for line in lines:
            assert line not in text, (name, line[:20])


def test_the_printed_lines_hold_a_path_only_in_the_last_three(night: NightRun) -> None:
    printed = night.invocation.stdout

    assert [line for line in printed[:-3] if absolute_paths(line)] == []
    assert [line.split(":")[0] for line in printed[-3:]] == [
        "report",
        "summary",
        "manifest",
    ]


# =================================================================================
# R4: the inputs and the shim.


def test_the_lock_and_the_project_file_keep_their_bytes_and_their_modification_times(
    night: NightRun, night_again: NightRun
) -> None:
    assert night.after == night.before
    assert night_again.before == night.before
    assert night_again.after == night.before
    manifest_inputs = {
        Path(i["path"]).name: i["sha256"] for i in night.manifest["inputs"]
    }
    for name in PROJECT_FILES:
        assert manifest_inputs[name] == night.before[name][0]


def test_the_dry_run_never_left_the_shim_and_the_real_uv_ran_only_what_the_gate_allows(
    night: NightRun, night_again: NightRun
) -> None:
    for run in (night, night_again):
        calls = run.calls
        dry = [c for c in calls if c["argv"] == ["lock", "--upgrade", "--dry-run"]]
        forwarded = [c for c in calls if c["forwarded"]]
        finals = sorted(" ".join(c["final"][1:]) for c in forwarded)

        assert len(dry) == 1
        assert dry[0]["forwarded"] is False
        assert [c for c in calls if c["refused"]] == []
        assert len(calls) == 4
        assert finals == sorted(
            [
                "--version",
                "lock --check --offline",
                (
                    f"run --project {REPOSITORY} --no-sync --frozen --offline "
                    f"rinchen maintain check --out-dir {OUT_DIR_NAME} --cadence nightly"
                ),
            ]
        )
        for call in forwarded:
            assert call["final"][0] == real_uv()
            assert "--upgrade" not in call["final"]
        run_final = next(c["final"] for c in forwarded if c["final"][1] == "run")
        counts = Counter(token for token in run_final if token.startswith("--"))
        assert max(counts.values()) == 1, counts


def test_the_shim_forwarded_the_run_without_the_flags_of_the_caller_twice(
    night: NightRun,
) -> None:
    caller, forwarded = (
        night.calls[0]["argv"],
        night.calls[0]["final"],
    )

    assert caller[:3] == ["run", "--frozen", "--no-sync"]
    assert forwarded[1:7] == [
        "run",
        "--project",
        str(REPOSITORY),
        "--no-sync",
        "--frozen",
        "--offline",
    ]
    assert forwarded[7:] == caller[3:]


# =================================================================================
# R5: the same bytes every time.


def test_the_command_run_again_gives_the_same_documents_byte_for_byte(
    night: NightRun, night_again: NightRun
) -> None:
    assert night_again.invocation.returncode == 0, night_again.invocation.stderr
    for name in ("maintain_report.json", "maintain_report.md"):
        first = (night.out_dir / name).read_bytes()
        second = (night_again.out_dir / name).read_bytes()
        assert first == second, name
        assert len(first) > 1000


# =================================================================================
# R6: the parity of an operator's export.


@dataclass(frozen=True)
class ExportRun:
    invocation: Invocation
    out_dir: Path
    shim_dir: Path

    @property
    def report(self) -> dict[str, Any]:
        return read_report(self.out_dir)


def _export_run(
    root: Path, workspace: Workspace, export: Path, checkpoint: Path
) -> ExportRun:
    shim = write_uv_shim(
        root / "bin",
        real_uv=real_uv(),
        project=REPOSITORY,
        dry_run_output=NO_CHANGE_MEASURED,
    )
    out_dir = root / "out"
    invocation = run_maintain(
        [
            "--out-dir",
            out_dir,
            "--no-drift",
            "--rerun",
            "parity_export",
            "--export",
            export,
            "--checkpoint",
            checkpoint,
            "--corpus",
            workspace.inputs.corpus,
            "--uv",
            shim,
        ],
        cwd=root,
    )
    return ExportRun(invocation, out_dir, shim.parent)


@pytest.fixture(scope="module")
def clean_export_run(
    d11_workspace: Workspace,
    d11_teacher: Teachers,
    d11_exports: Exports,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExportRun:
    return _export_run(
        tmp_path_factory.mktemp("export_clean"),
        d11_workspace,
        d11_exports.float32.directory,
        d11_teacher.source.averaged,
    )


def test_the_parity_of_the_toy_export_passes_with_every_sentence_exact(
    clean_export_run: ExportRun,
) -> None:
    run = clean_export_run
    item = rerun_item(run.report, "parity_export")
    figures = item["figures"]

    assert run.invocation.returncode == 0, run.invocation.stderr
    assert (item["state"], item["reason"]) == ("ok", None)
    assert figures["sentences"] == SENTENCES_PER_DIRECTION * DIRECTIONS
    assert figures["exact"] == figures["sentences"]
    assert figures["exact"] / figures["sentences"] >= MIN_EXACT_SHARE
    assert figures["mean_abs_delta"] <= MAX_MEAN_ABS_DELTA
    assert run.report["state"] == "ok"


def test_the_selection_wins_over_the_cadence_and_the_drift_is_not_asked(
    clean_export_run: ExportRun,
) -> None:
    report = clean_export_run.report
    skipped = {
        item["id"]: item["reason"]
        for item in step_of(report, "reruns")["detail"]["items"]
        if item["state"] == "skipped"
    }
    drift = step_of(report, "drift")

    assert skipped == dict.fromkeys(
        [
            "numerics",
            "parity_tiny",
            "robustness_tiny",
            "evaluation_tiny",
            "serving_tiny",
        ],
        NOT_SELECTED,
    )
    assert (drift["state"], drift["reason"]) == ("skipped", "disabled by --no-drift")
    assert not any(
        "--dry-run" in call["argv"] for call in read_calls(clean_export_run.shim_dir)
    )


def test_the_parity_figures_are_in_the_manifest_outputs(
    clean_export_run: ExportRun,
) -> None:
    out = clean_export_run.out_dir
    listed = {
        Path(item["path"]).relative_to(out).as_posix()
        for item in read_manifest(out)["outputs"]
    }

    assert "parity_export/parity.json" in listed
    assert (out / "parity_export" / "parity.json").is_file()


def _damaged_export(root: Path, source: Path) -> Path:
    copy = root / "export_one_byte_flipped"
    shutil.copytree(source, copy)
    model = copy / "model.bin"
    data = bytearray(model.read_bytes())
    data[len(data) // 2] ^= 0x01
    model.write_bytes(bytes(data))
    return copy


def _assert_not_vouched_for(run: ExportRun) -> None:
    item = rerun_item(run.report, "parity_export")

    assert run.invocation.returncode == 4, run.invocation.stderr
    assert (item["state"], item["reason"]) == ("fail", "the command exited with code 2")
    assert item["figures"] is None
    assert run.report["state"] == "fail"
    assert (run.out_dir / "manifest.json").is_file()
    assert (run.out_dir / "maintain_report.md").is_file()


def test_an_export_with_one_byte_flipped_is_not_vouched_for(
    d11_workspace: Workspace,
    d11_teacher: Teachers,
    d11_exports: Exports,
    tmp_path: Path,
) -> None:
    source = d11_exports.float32.directory
    before = hashlib.sha256((source / "model.bin").read_bytes()).hexdigest()
    damaged = _damaged_export(tmp_path, source)

    run = _export_run(tmp_path, d11_workspace, damaged, d11_teacher.source.averaged)

    _assert_not_vouched_for(run)
    assert hashlib.sha256((source / "model.bin").read_bytes()).hexdigest() == before


def test_an_export_checked_against_the_checkpoint_of_another_model_is_not_vouched_for(
    d11_workspace: Workspace,
    d11_teacher: Teachers,
    d11_exports: Exports,
    tmp_path: Path,
) -> None:
    run = _export_run(
        tmp_path,
        d11_workspace,
        d11_exports.float32.directory,
        d11_teacher.target.averaged,
    )

    _assert_not_vouched_for(run)


def test_the_parity_file_of_the_rerun_says_it_passes(
    clean_export_run: ExportRun,
) -> None:
    figures = json.loads(
        (clean_export_run.out_dir / "parity_export" / "parity.json").read_text("utf-8")
    )

    assert figures["passes"] is True
    assert figures["exact"] == figures["sentences"]
