"""The D13 STOP gate, release half (R9 to R11): `rinchen release stage` and `rinchen release hub` through
the real binary over the export of the D12 toy and the report of the evaluation gate, on the CPU with
every device hidden and no network (`HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` on, no token in the
environment). The Hub is never contacted: only the DRY RUN is run, no Hub client is made (the Hub library
and `datasets` are BLOCKED for every command, a package of each name that raises `ImportError` first on the
command's path, and the manifest of a dry run names no Hub package), and `--upload` is run once, WITHOUT a
token, to show that it stops before it does anything. A real upload, the Hub's acceptance of the card and
a run in GitHub Actions are UNMEASURED.

Gates, every one a literal of `tests/support/eval_gate.py` or of this file and never lowered:

    R9  `release stage` of the D12 export and the report of the evaluation gate (system `a-model`): the
        nine files of the upload and `manifest.json`; the card without a word of the content rules (the
        scan of `tests/unit/test_d13_content_rules.py` over every generated text file), with the fixture
        banner, every decimal of its evaluation tables a number of the report, `release.json` agreeing
        with the disk, two stages byte-equal; a report built without the export's system, with another
        model's, or naming a system that is not a model of ours exits 3 with nothing written; a report
        that carries the robustness block gives a card with its six values and their counts and targets
        that say `not_met` where the toy misses them (a target is a measurement, no command fails)
    R10 `release hub` over the stage: a dry run exits 0, lists exactly the nine names with the sizes and
        the digests of the disk, writes `hub_plan.json` (equal to the plan rebuilt here from the disk) and
        a manifest that names no Hub package, and creates no client; a stray file, a flipped byte, a
        symlink, a card with a token-like line and a card with an absolute local path each exit 3 with
        nothing written and the matched text in no message; `--upload` with no `HF_TOKEN` exits 2 and
        writes nothing
    R11 hygiene: the new tests and support files ask no device question, open no socket (no `socket`,
        `urllib`, `requests` or `httpx.Client(`), start no process of their own, every command they run is
        bounded by 900 seconds and refused unless the devices are hidden; none holds a word of the content
        rules, a non-ASCII character or an invisible one; the committed fixtures of D13 stay under 50 KB

Measured on this machine (CPU, one thread per command, Eole 0.6.4, CTranslate2 4.8.4, sacrebleu 2.6.0,
2026-11-11; a time moves by a third between runs, a count or a score never does):

    setup  the report of the export and the D11 teacher (`report.json` 30,013 bytes, `report.md` 7,687) and
           the stage: `model.bin` 2,080,071 bytes, `shared_vocabulary.json` 12,610, `config.json` 223,
           `spm.model` 15,210, `rinchen.json` 1,184, `README.md` 7,339, `report.json` 30,013, `report.md`
           7,687, `release.json` 2,360 (nine files, 2,156,697 bytes in all) and the stage's `manifest.json`
           4,468; each `release` command takes 1 to 3 s
    R9     the card of the export against the teacher; with the robustness block (`report.json` 31,586
           bytes, card 7,763) the card's Robustness table holds the six values with their counts (960 of
           960, 600 of 600, 1,044 of 1,044, 900 of 900, 480 of 480, 480 of 480) and the targets `numbers`
           (0.9292 against 0.99) and `placeholders` (0.9983 against 1) are `not_met`, `beats_baseline` and
           `off_target` `unmeasured`: a target is a measurement and no command fails
    R10    the dry run writes `hub_plan.json` (1,533 bytes) and a manifest of stage `release_hub` with the
           status `mode: dry-run`, `visibility: private` and no tool version of the Hub library
    time   this module's 32 tests took 26.2 s in the full gate (every `release` command takes 0.9 to 2.5 s)

UNMEASURED: the Hub (a real upload, the acceptance of the card's front matter, the visibility of an
existing repository), GitHub Actions, every real dataset, baseline and neural metric and anything on a GPU.
The toy ranks nothing.
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.export.release import UPLOAD_NAMES
from tests.support.d11_gate import forbidden_gpu_findings
from tests.support.distill_gate import SUBPROCESS_TIMEOUT_SECONDS, read_json, sha256_of
from tests.support.eval_gate import (
    NINE_NAMES,
    REPO_ID,
    Report,
    Runs,
    absolute_paths_in,
    card_decimals,
    card_section,
    expected_plan,
    make_report,
    network_findings,
    report_numbers,
    run_eval,
    run_release,
    run_robustness,
)
from tests.support.eval_toy import Suites
from tests.support.hub_stage import rewrite
from tests.support.robust_gate import Suites as RobustSuites
from tests.unit.test_d13_content_rules import (
    lines_with_a_needle,
    needles_in,
    scanned_files,
)

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden", "d13_offline"),
]

MANIFEST: Final = "manifest.json"
PLAN_FILE: Final = "hub_plan.json"
GENERATED_TEXT: Final = (
    "README.md",
    "release.json",
    "report.json",
    "report.md",
    "rinchen.json",
)
EXPORT_FILES: Final = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
)
NEW_FILES: Final = (
    REPO_ROOT / "tests" / "support" / "eval_toy.py",
    REPO_ROOT / "tests" / "support" / "eval_gate.py",
    REPO_ROOT / "tests" / "integration" / "test_eval_gate.py",
    REPO_ROOT / "tests" / "integration" / "test_eval_gate_controls.py",
    REPO_ROOT / "tests" / "integration" / "test_release_gate.py",
)
FIXTURE_BUDGET_BYTES: Final = 50 * 1024


# ---------------------------------------------------------------------------------
# The fixtures: each runs the real command once per module.


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("d13_release_gate")


@pytest.fixture(scope="module")
def release_report(d13_suites: Suites, d13_runs: Runs, scratch: Path) -> Report:
    """The export (`a-model`) and the D11 teacher, the teacher as the baseline."""
    return make_report(
        [d13_runs.model, d13_runs.teacher],
        [d13_suites.devtest],
        scratch / "report",
        baseline="d11-teacher",
    )


def _stage_args(
    export: Path, report: Report, out_dir: Path, system: str = "a-model"
) -> list[str | Path]:
    return [
        "stage",
        "--export",
        export,
        "--report-dir",
        report.directory,
        "--system",
        system,
        "--repo-id",
        REPO_ID,
        "--out-dir",
        out_dir,
    ]


@pytest.fixture(scope="module")
def stage(d12_robust_export: Path, release_report: Report, scratch: Path) -> Path:
    out_dir = scratch / "stage"
    staged = run_release(*_stage_args(d12_robust_export, release_report, out_dir))
    assert staged.returncode == 0, staged.stderr
    return out_dir


@pytest.fixture(scope="module")
def robust_report(
    d13_suites: Suites,
    d13_runs: Runs,
    d12_robust_export: Path,
    d12_suites: RobustSuites,
    scratch: Path,
) -> Report:
    """A report whose export carries D12's robustness block with its coverage."""
    robustness = scratch / "robustness"
    measured = run_robustness(
        "--suite",
        d12_suites.main,
        "--model",
        d12_robust_export,
        "--out-dir",
        robustness,
    )
    assert measured.returncode == 0, measured.stderr
    ran = run_eval(
        "run",
        "--suite",
        d13_suites.devtest,
        "--model",
        d12_robust_export,
        "--system",
        "a-model-robust",
        "--robustness",
        robustness,
        "--out-dir",
        scratch / "robust_runs",
    )
    assert ran.returncode == 0, ran.stderr
    return make_report(
        [scratch / "robust_runs" / "toy-devtest" / "a-model-robust", d13_runs.teacher],
        [d13_suites.devtest],
        scratch / "robust_report",
        baseline="d11-teacher",
    )


# =================================================================================
# R9: `release stage`.


def test_the_stage_holds_the_nine_files_of_the_upload_and_its_own_manifest(
    stage: Path,
) -> None:
    assert NINE_NAMES == UPLOAD_NAMES
    assert sorted(path.name for path in stage.iterdir()) == sorted(
        [*NINE_NAMES, MANIFEST]
    )
    assert all(path.is_file() and not path.is_symlink() for path in stage.iterdir())


def test_the_stage_prints_the_nine_files_with_their_sizes_and_digests(
    d12_robust_export: Path, release_report: Report, scratch: Path
) -> None:
    out_dir = scratch / "stage_printed"
    staged = run_release(*_stage_args(d12_robust_export, release_report, out_dir))

    assert staged.returncode == 0, staged.stderr
    assert staged.stdout[0] == f"stage: {out_dir}"
    assert [line.split()[0] for line in staged.stdout[1:10]] == list(NINE_NAMES)
    for line in staged.stdout[1:10]:
        name, size, digest = line.split()
        data = (out_dir / name).read_bytes()
        assert size == str(len(data)), name
        assert sha256_of(out_dir / name).startswith(digest), name
    assert staged.stdout[10] == f"manifest: {out_dir / MANIFEST}"


def test_the_five_export_files_and_the_two_reports_are_the_bytes_that_were_vouched_for(
    stage: Path, d12_robust_export: Path, release_report: Report
) -> None:
    for name in EXPORT_FILES:
        assert (stage / name).read_bytes() == (d12_robust_export / name).read_bytes()
    for name in ("report.json", "report.md"):
        assert (stage / name).read_bytes() == (
            release_report.directory / name
        ).read_bytes()


def test_the_card_and_every_generated_text_hold_no_word_of_the_content_rules(
    stage: Path,
) -> None:
    for name in GENERATED_TEXT:
        text = (stage / name).read_text("utf-8")
        assert lines_with_a_needle(text) == [], name
        assert needles_in(text) == [], name


def test_the_card_says_fixture_in_its_banner_and_ranks_nothing(stage: Path) -> None:
    card = (stage / "README.md").read_text("utf-8")
    banner = next(line for line in card.splitlines() if line.startswith("> "))

    assert "fixture" in banner
    assert "say nothing about translation quality" in banner
    assert card.index(banner) < card.index("## What it is")
    lowered = card.lower()
    for phrase in ("outperform", "state of the art", "superior", "the best", "wins"):
        assert phrase not in lowered, phrase


def test_every_decimal_of_the_evaluation_tables_of_the_card_is_a_number_of_the_report(
    stage: Path,
) -> None:
    card = (stage / "README.md").read_text("utf-8")
    report = read_json(stage / "report.json")
    shown = card_decimals(card)

    assert len(shown) >= 30
    assert not set(shown) - report_numbers(report), sorted(
        set(shown) - report_numbers(report)
    )
    assert "a-model" in card_section(card, "Evaluation")


def test_the_card_lists_the_files_it_ships_with_the_digests_of_the_disk(
    stage: Path,
) -> None:
    table = card_section((stage / "README.md").read_text("utf-8"), "Files")
    listed = [name for name in NINE_NAMES if name not in ("README.md", "release.json")]

    assert len(listed) == 7
    for name in listed:
        row = next(
            line for line in table.splitlines() if line.startswith(f"| {name} |")
        )
        size, digest = [cell.strip() for cell in row.strip("|").split("|")[1:3]]
        assert size == str((stage / name).stat().st_size), name
        assert digest == sha256_of(stage / name), name


def test_the_release_record_agrees_with_the_disk(
    stage: Path, d12_robust_export: Path
) -> None:
    release = read_json(stage / "release.json")
    text = (stage / "release.json").read_text("utf-8")

    assert release["schema"] == "rinchen.release/1"
    assert release["repo_id"] == REPO_ID
    assert release["evidence"] == "fixture"
    assert set(release["files"]) == set(NINE_NAMES) - {"release.json"}
    for name, listed in release["files"].items():
        assert listed == {
            "bytes": (stage / name).stat().st_size,
            "sha256": sha256_of(stage / name),
        }, name
    assert release["report"] == {
        "sha256": sha256_of(stage / "report.json"),
        "system": "a-model",
    }
    assert release["export"]["model_sha256"] == sha256_of(stage / "model.bin")
    assert release["export"]["manifest_sha256"] == sha256_of(
        d12_robust_export / MANIFEST
    )
    assert release["export"]["spm_sha256"] == sha256_of(stage / "spm.model")
    assert absolute_paths_in(text) == []
    assert '"created"' not in text


def test_the_manifest_of_the_stage_lists_the_nine_files_with_the_digests_of_the_disk(
    stage: Path,
) -> None:
    manifest = read_json(stage / MANIFEST)

    assert manifest["stage"] == "release_stage"
    assert manifest["status"]["release"] == "ok"
    assert {
        Path(item["path"]).name: item["sha256"] for item in manifest["outputs"]
    } == {name: sha256_of(stage / name) for name in NINE_NAMES}


def test_two_stages_of_the_same_inputs_are_byte_equal(
    stage: Path, d12_robust_export: Path, release_report: Report, scratch: Path
) -> None:
    again = scratch / "stage_again"

    staged = run_release(*_stage_args(d12_robust_export, release_report, again))

    assert staged.returncode == 0, staged.stderr
    for name in NINE_NAMES:
        assert (again / name).read_bytes() == (stage / name).read_bytes(), name


@pytest.mark.parametrize(
    ("system", "why"),
    [
        ("a-model", "the report has no such system"),
        ("d11-teacher", "the system of the report is another model"),
        ("a-file", "the system of the report is not a model of ours"),
    ],
)
def test_a_report_that_is_not_the_exports_is_refused_and_nothing_is_written(
    d12_robust_export: Path,
    d13_suites: Suites,
    d13_runs: Runs,
    scratch: Path,
    system: str,
    why: str,
) -> None:
    report = make_report(
        [d13_runs.teacher, d13_runs.file],
        [d13_suites.devtest],
        scratch / f"report_without_a_model_{system}",
        baseline="d11-teacher",
    )
    out_dir = scratch / f"refused_stage_{system}"

    refused = run_release(*_stage_args(d12_robust_export, report, out_dir, system))

    assert refused.returncode == 3, (why, refused.stderr)
    assert refused.stdout == []
    assert not out_dir.exists()


def test_a_malformed_repository_is_refused_before_anything_is_read(
    d12_robust_export: Path, release_report: Report, scratch: Path
) -> None:
    out_dir = scratch / "refused_repo"
    args = _stage_args(d12_robust_export, release_report, out_dir)
    args[args.index(REPO_ID)] = "not a repository"

    refused = run_release(*args)

    assert refused.returncode == 2, refused.stderr
    assert refused.stdout == []
    assert not out_dir.exists()


def test_a_report_with_the_robustness_block_gives_a_card_with_its_values_and_its_counts(
    d12_robust_export: Path, robust_report: Report, scratch: Path
) -> None:
    out_dir = scratch / "robust_stage"

    staged = run_release(
        *_stage_args(d12_robust_export, robust_report, out_dir, "a-model-robust")
    )

    assert staged.returncode == 0, staged.stderr
    card = (out_dir / "README.md").read_text("utf-8")
    report = read_json(out_dir / "report.json")
    entry = report["robustness"]["a-model-robust"]
    table = card_section(card, "Robustness")
    for key, value in entry["block"].items():
        row = next(line for line in table.splitlines() if line.startswith(f"| {key} |"))
        counts = entry["coverage"][key]
        assert f"| {value:.4f} |" in row, key
        assert f"{counts['scored']} of {counts['total']}" in row, key
    targets = {target["id"]: target for target in report["targets"]}
    assert targets["numbers"]["status"] == "not_met"
    assert targets["placeholders"]["status"] == "not_met"
    assert targets["beats_baseline"]["status"] == "unmeasured"
    assert not set(card_decimals(card)) - report_numbers(report)
    assert lines_with_a_needle(card) == []
    assert "| numbers | not_met |" in card_section(card, "Release targets")


# =================================================================================
# R10: `release hub`, the dry run and its refusals.


def _hub(stage: Path, out_dir: Path, *extra: str, repo_id: str = REPO_ID) -> Any:
    return run_release(
        "hub", "--stage", stage, "--repo-id", repo_id, "--out-dir", out_dir, *extra
    )


def test_the_dry_run_lists_exactly_the_nine_files_with_the_sizes_and_digests_of_the_disk(
    stage: Path, scratch: Path
) -> None:
    out_dir = scratch / "plan"

    planned = _hub(stage, out_dir)

    assert planned.returncode == 0, planned.stderr
    assert planned.stderr == ""
    assert planned.stdout[:2] == ["mode: dry-run", f"repo: {REPO_ID}"]
    assert [line.split()[0] for line in planned.stdout[2:11]] == list(NINE_NAMES)
    assert MANIFEST not in " ".join(planned.stdout[2:11])
    for line in planned.stdout[2:11]:
        name, size, digest = line.split()
        assert size == str((stage / name).stat().st_size), name
        assert sha256_of(stage / name).startswith(digest), name
    assert planned.stdout[11] == (
        f"total: {sum((stage / name).stat().st_size for name in NINE_NAMES)}"
    )
    assert planned.facts["plan"] == str(out_dir / PLAN_FILE)


def test_the_plan_is_the_one_the_disk_gives_and_the_manifest_names_no_hub_package(
    stage: Path, scratch: Path
) -> None:
    out_dir = scratch / "plan_checked"

    planned = _hub(stage, out_dir)

    assert planned.returncode == 0, planned.stderr
    assert (out_dir / PLAN_FILE).read_bytes() == expected_plan(stage, REPO_ID)
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(
        [MANIFEST, PLAN_FILE]
    )
    manifest = read_json(out_dir / MANIFEST)
    assert manifest["stage"] == "release_hub"
    assert manifest["status"] == {
        "mode": "dry-run",
        "repo_id": REPO_ID,
        "visibility": "private",
    }
    # A dry run makes no client: no Hub package is recorded, no commit exists.
    assert "huggingface_hub" not in manifest["tool_versions"]
    assert not (out_dir / "hub_commit.json").exists()
    assert "HF_TOKEN" not in (out_dir / MANIFEST).read_text("utf-8")


def test_two_dry_runs_write_the_same_plan(stage: Path, scratch: Path) -> None:
    first, second = scratch / "plan_one", scratch / "plan_two"

    assert _hub(stage, first).returncode == 0
    assert _hub(stage, second).returncode == 0

    assert (first / PLAN_FILE).read_bytes() == (second / PLAN_FILE).read_bytes()


def _damaged_copy(stage: Path, scratch: Path, name: str) -> Path:
    damaged = scratch / f"damaged_{name}"
    shutil.copytree(stage, damaged)
    return damaged


def _token_like() -> str:
    """What the scan must take for a token, built here so that no file holds it."""
    return "hf" + "_" + "".join(chr(ord("a") + index % 26) for index in range(30))


def _local_path() -> str:
    """What the scan must take for an absolute path, built here so that no file holds one."""
    slash = chr(0x2F)
    return f"{slash}home{slash}someone{slash}x"


def _refused_and_nothing_written(
    damaged: Path, scratch: Path, name: str, *, shows: str, hides: str | None = None
) -> None:
    out_dir = scratch / f"refused_{name}"

    refused = _hub(damaged, out_dir)

    assert refused.returncode == 3, refused.stderr
    assert refused.stdout == []
    assert shows in refused.stderr
    if hides is not None:
        assert hides not in refused.stderr
    assert not out_dir.exists()


def test_a_stray_file_is_refused_by_name(stage: Path, scratch: Path) -> None:
    damaged = _damaged_copy(stage, scratch, "stray")
    (damaged / "notes.txt").write_bytes(b"x")

    _refused_and_nothing_written(damaged, scratch, "stray", shows="notes.txt")


def test_a_flipped_byte_is_refused_by_the_name_of_the_file(
    stage: Path, scratch: Path
) -> None:
    damaged = _damaged_copy(stage, scratch, "flipped")
    weights = bytearray((damaged / "model.bin").read_bytes())
    weights[len(weights) // 2] ^= 0xFF
    (damaged / "model.bin").write_bytes(bytes(weights))

    _refused_and_nothing_written(damaged, scratch, "flipped", shows="model.bin")


def test_a_symlink_in_place_of_a_file_is_refused_even_when_it_points_at_the_same_bytes(
    stage: Path, scratch: Path
) -> None:
    damaged = _damaged_copy(stage, scratch, "symlink")
    outside = scratch / "outside_report.md"
    outside.write_bytes((stage / "report.md").read_bytes())
    (damaged / "report.md").unlink()
    (damaged / "report.md").symlink_to(outside)

    # The link reads the very bytes the stage lists: only the refusal of a link can stop it.
    _refused_and_nothing_written(
        damaged, scratch, "symlink", shows="report.md (not a regular file)"
    )


def test_a_card_with_a_token_like_line_is_refused_and_the_token_is_in_no_message(
    stage: Path, scratch: Path
) -> None:
    damaged = _damaged_copy(stage, scratch, "token")
    card = (damaged / "README.md").read_bytes()
    rewrite(damaged, "README.md", card + b"\n" + _token_like().encode() + b"\n")

    _refused_and_nothing_written(
        damaged, scratch, "token", shows="README.md", hides=_token_like()
    )


def test_a_card_with_an_absolute_local_path_is_refused_and_the_path_is_in_no_message(
    stage: Path, scratch: Path
) -> None:
    damaged = _damaged_copy(stage, scratch, "path")
    card = (damaged / "README.md").read_bytes()
    rewrite(damaged, "README.md", card + b"\nsee " + _local_path().encode() + b"\n")

    _refused_and_nothing_written(
        damaged, scratch, "path", shows="README.md", hides=_local_path()
    )


def test_an_upload_without_a_token_stops_before_it_does_anything(
    stage: Path, scratch: Path
) -> None:
    out_dir = scratch / "upload_without_token"

    refused = _hub(stage, out_dir, "--upload")

    assert refused.returncode == 2, refused.stderr
    assert refused.stdout == []
    assert "HF_TOKEN" in refused.stderr
    assert not out_dir.exists()


def test_another_repository_than_the_one_the_stage_was_made_for_is_refused(
    stage: Path, scratch: Path
) -> None:
    out_dir = scratch / "other_repository"

    refused = _hub(stage, out_dir, repo_id="rinchen-test/another")

    assert refused.returncode == 2, refused.stderr
    assert refused.stdout == []
    assert not out_dir.exists()


def test_a_release_command_in_a_process_that_does_not_hide_the_devices_is_refused(
    stage: Path, d12_robust_export: Path, release_report: Report, scratch: Path
) -> None:
    out_dir = scratch / "must_not_start"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
            run_release(*_stage_args(d12_robust_export, release_report, out_dir))
        with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
            _hub(stage, out_dir)

    assert not out_dir.exists()


# =================================================================================
# R11: hygiene.


def test_no_new_file_asks_a_device_question_or_reaches_for_the_network() -> None:
    assert len(NEW_FILES) == 5
    for path in NEW_FILES:
        assert path.is_file(), path
        assert forbidden_gpu_findings(path) == [], path
        assert network_findings(path) == [], path


def test_the_new_files_start_no_process_of_their_own_and_every_command_is_bounded() -> (
    None
):
    assert SUBPROCESS_TIMEOUT_SECONDS <= 900
    for path in NEW_FILES:
        tree = ast.parse(path.read_text("utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "subprocess" not in imported, path
        assert "ctranslate2" not in imported, path
    runner = ast.parse(
        (REPO_ROOT / "tests" / "support" / "distill_gate.py").read_text("utf-8")
    )
    runs = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "subprocess.run"
    ]
    assert len(runs) == 1
    assert [
        ast.unparse(keyword.value)
        for keyword in runs[0].keywords
        if keyword.arg == "timeout"
    ] == ["SUBPROCESS_TIMEOUT_SECONDS"]


def test_the_environment_of_the_session_is_the_one_every_command_inherits() -> None:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in os.environ


def test_the_new_files_hold_no_word_of_the_content_rules_and_are_scanned_by_the_repository_scan() -> (
    None
):
    scanned = {path.resolve() for path in scanned_files()}

    for path in NEW_FILES:
        assert path.resolve() in scanned, path
        assert lines_with_a_needle(path.read_text("utf-8")) == [], path


def test_the_new_files_are_plain_ascii_and_none_is_too_long() -> None:
    for path in NEW_FILES:
        text = path.read_text("utf-8")
        assert text.isascii(), path
        assert len(text.splitlines()) < 900, path


def test_the_committed_fixtures_of_d13_stay_within_their_budget() -> None:
    fixtures = sorted((REPO_ROOT / "tests" / "fixtures" / "eval").iterdir())

    assert fixtures
    assert sum(path.stat().st_size for path in fixtures) < FIXTURE_BUDGET_BYTES
