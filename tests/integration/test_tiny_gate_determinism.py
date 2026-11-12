"""The D15 STOP gate, determinism (R4): the recipe run a second time, in another directory, under another hash
seed, another locale and another time zone, writes the same `report.json` bytes, and the four model files the
pipeline makes (the averaged weights, the float32 and the int8 `model.bin` and the converter's `model.bin`)
have the same digests. The digests are RECORDED, never compared with a literal: another machine, another
thread count or another build of the libraries may move them, and the claim is about two runs on this machine.

What is and is not equal (the design spike's F12, measured again here): the report, its Markdown, the four
model files and the parity figures are equal; the SentencePiece model (it embeds the paths it was trained
from), the manifest (its seconds and its creation time) and every file that embeds a digest of a manifest are
not, and the report embeds none of them.

The second run is the recipe again, through its own shim: nothing is cheaper than the real thing, and the run
shows that the recipe is what is deterministic, not a path of its own.

Measured on this machine (CPU, 2026-11-12; the second run under `PYTHONHASHSEED=3`, `LC_ALL=C` and
`TZ=Asia/Ulaanbaatar`; when the outer environment already has one of the three (the launcher of the full gate sets
`LC_ALL=C`), the second run takes another value of it, `C.UTF-8` for the locale):

    R4  the two runs took 143.1 and 128.2 s by the manifests' seconds (three pipelines were running at once);
           `report.json` (26,545 bytes) and `report.md` (its size is not pinned) byte-equal; the digests of the averaged
           weights (`4e06ce45faef7c4b...`), the float32 `model.bin` (`f48811a45287337c...`), the int8 `model.bin`
           (`8fa90a266950cfc3...`) and the converter's `model.bin` (`e70a4ccacf85c060...`) equal; the
           SentencePiece models differ (`66f3f736...` and `58af3a54...`), as they must (the converter's digest
           is also the one of the design spike's runs; the spike's other digests were of an earlier corpus)

UNMEASURED: another machine, another thread count, GitHub Actions. The toy ranks nothing.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.fake_uv import read_calls
from tests.support.tiny_gate import (
    RecipeRun,
    make_shim,
    read_manifest,
    read_report,
    run_the_recipe,
)
from tests.support.tiny_gate_controls import (
    averaged_directory,
    export_directory,
    stage_path,
)

pytestmark = [pytest.mark.subprocess, pytest.mark.usefixtures("tiny_gpu_hidden")]

# The environment of the second run (the brief's) and, for a variable that the outer environment already
# has at that value, another one, so that the two runs differ in all three.
WANTED_ENVIRONMENT: Final = {
    "PYTHONHASHSEED": "3",
    "LC_ALL": "C",
    "TZ": "Asia/Ulaanbaatar",
}
FALLBACK_ENVIRONMENT: Final = {
    "PYTHONHASHSEED": "4",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}


def other_environment(outer: Mapping[str, str]) -> dict[str, str]:
    """The hash seed, locale and time zone of the second run: the ones the brief names, unless `outer` (the
    environment the first run saw) already has one of them, which would make that part of the comparison say
    nothing: the full gate's launcher sets `LC_ALL=C` to order its files, and the second run then takes another
    locale."""
    return {
        name: wanted if outer.get(name) != wanted else FALLBACK_ENVIRONMENT[name]
        for name, wanted in WANTED_ENVIRONMENT.items()
    }


def _model_files(out_dir: Path) -> dict[str, Path]:
    """The four model files of a run, by what they are."""
    return {
        "averaged weights": averaged_directory(out_dir) / "model.00.safetensors",
        "float32 model.bin": export_directory(out_dir, "float32") / "model.bin",
        "int8 model.bin": export_directory(out_dir, "int8") / "model.bin",
        "converter model.bin": stage_path(out_dir, "converter", "model", "model.bin"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def second_run(
    tiny_run_dir: RecipeRun, tmp_path_factory: pytest.TempPathFactory
) -> RecipeRun:
    """The recipe once more, in another directory with a shim of its own, under another hash seed, another
    locale and another time zone."""
    root = tmp_path_factory.mktemp("tiny_gate_second_run")
    return run_the_recipe(
        root,
        shim_dir=make_shim(root / "bin"),
        env_extra=other_environment(os.environ),
    )


def test_the_second_run_differs_from_the_first_in_what_the_claim_says_it_may(
    tiny_run_dir: RecipeRun, second_run: RecipeRun
) -> None:
    assert second_run.cwd != tiny_run_dir.cwd
    # The first run saw the environment of the test process: if that already was the second run's,
    # the comparison would say nothing.
    for name, value in other_environment(os.environ).items():
        assert os.environ.get(name) != value, name


@pytest.mark.parametrize(
    ("outer", "expected"),
    [
        (
            {},
            {"PYTHONHASHSEED": "3", "LC_ALL": "C", "TZ": "Asia/Ulaanbaatar"},
        ),
        # The launcher of the full gate sets LC_ALL=C to order the files: the second run then takes another one.
        (
            {"LC_ALL": "C"},
            {"PYTHONHASHSEED": "3", "LC_ALL": "C.UTF-8", "TZ": "Asia/Ulaanbaatar"},
        ),
        (
            {"PYTHONHASHSEED": "3", "TZ": "Asia/Ulaanbaatar", "LC_ALL": "C"},
            {"PYTHONHASHSEED": "4", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
        ),
        (
            {"PYTHONHASHSEED": "7", "LC_ALL": "en_US.UTF-8", "TZ": "Europe/Berlin"},
            {"PYTHONHASHSEED": "3", "LC_ALL": "C", "TZ": "Asia/Ulaanbaatar"},
        ),
    ],
    ids=["nothing-set", "the-gates-locale", "everything-the-same", "all-different"],
)
def test_the_second_environment_is_the_briefs_and_differs_from_the_outer_one_in_all_three(
    outer: dict[str, str], expected: dict[str, str]
) -> None:
    assert other_environment(outer) == expected
    assert all(other_environment(outer)[name] != outer.get(name) for name in expected)


def test_the_second_run_passes_and_asked_uv_for_one_run_only(
    second_run: RecipeRun,
) -> None:
    invocation = second_run.invocation
    calls = read_calls(second_run.shim_dir)

    assert invocation.returncode == 0, invocation.stderr[-3000:]
    assert "verdict: pass" in invocation.stdout
    assert [call["refused"] for call in calls] == [False]
    assert [call["argv"][:1] for call in calls] == [["run"]]


def test_the_two_reports_are_byte_equal(
    tiny_run_dir: RecipeRun, second_run: RecipeRun, record_property: Any
) -> None:
    first = (tiny_run_dir.out_dir / "report.json").read_bytes()
    second = (second_run.out_dir / "report.json").read_bytes()
    record_property("report_json_bytes", len(first))

    assert first == second
    assert (tiny_run_dir.out_dir / "report.md").read_bytes() == (
        second_run.out_dir / "report.md"
    ).read_bytes()
    assert read_report(tiny_run_dir.out_dir) == read_report(second_run.out_dir)


def test_the_four_model_files_have_the_same_digests_in_both_runs(
    tiny_run_dir: RecipeRun, second_run: RecipeRun, record_property: Any
) -> None:
    first = _model_files(tiny_run_dir.out_dir)
    second = _model_files(second_run.out_dir)

    assert (
        list(first)
        == list(second)
        == [
            "averaged weights",
            "float32 model.bin",
            "int8 model.bin",
            "converter model.bin",
        ]
    )
    for name in first:
        digest = _sha256(first[name])
        record_property(name.replace(" ", "_"), digest)
        assert len(digest) == 64
        assert _sha256(second[name]) == digest, name


def test_the_digests_the_manifests_list_are_the_digests_of_the_files(
    tiny_run_dir: RecipeRun, second_run: RecipeRun
) -> None:
    for run in (tiny_run_dir, second_run):
        listed = {
            Path(item["path"]).relative_to(run.out_dir).as_posix(): item["sha256"]
            for item in read_manifest(run.out_dir)["outputs"]
        }
        for path in _model_files(run.out_dir).values():
            assert listed[path.relative_to(run.out_dir).as_posix()] == _sha256(path)


def test_the_parity_figures_and_the_inputs_of_the_two_runs_are_equal(
    tiny_run_dir: RecipeRun, second_run: RecipeRun
) -> None:
    first = read_manifest(tiny_run_dir.out_dir)
    second = read_manifest(second_run.out_dir)

    assert first["inputs"] == second["inputs"]
    assert first["tool_versions"] == second["tool_versions"]
    assert first["config_sha256"] == second["config_sha256"]
    assert first["seed"] == second["seed"]
    assert (
        stage_path(tiny_run_dir.out_dir, "parity", "float32", "parity.json")
    ).read_bytes() == (
        stage_path(second_run.out_dir, "parity", "float32", "parity.json")
    ).read_bytes()
