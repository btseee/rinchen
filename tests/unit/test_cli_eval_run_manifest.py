"""`rinchen eval run`: what its manifest says, the files it replaces and the order in which it
writes (D13, Task 7). The manifest (stage `eval_run`) names every input with the digest taken before
it was read, every output with the digest of the bytes on disk, the bootstrap seed and the
measurements, and is written LAST. The files an earlier manifest of the same (suite, system) listed
go before the first new file, by that manifest's list and never by listing the directory.

The suite and the outputs are the golden files, the export a fabricated one that `verify_export`
vouches for, the engine and the translator fakes (`tests.support.eval_cli`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from tests.support.eval_cli import (
    FIRST,
    SECOND,
    EvalWorkspace,
    install,
    invoke,
    make_workspace,
    read_json,
    run_dir_of,
)
from tests.support.eval_fakes import GOLDEN_OUTPUTS_FILES, FakeCandidateTranslator
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import fail_replace, record_replaces

pytestmark = pytest.mark.usefixtures("no_network")

PACKAGES = ("ctranslate2", "sentencepiece", "sacrebleu", "numpy", "httpx")


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path: Path, rows: int | None = None) -> ManifestFile:
    """The entry a manifest holds for the file `path` as it is on disk."""
    return ManifestFile(
        path=str(path), sha256=sha(path), bytes=path.stat().st_size, rows=rows
    )


def manifest_of(directory: Path) -> Any:
    manifest = store.read(directory)
    assert manifest is not None
    return manifest


def listing(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


# --- a hypotheses run --------------------------------------------------------------------------------


def test_the_manifest_of_a_hypotheses_run_names_the_stage_the_inputs_and_the_outputs(
    workspace: EvalWorkspace,
) -> None:
    assert (
        invoke(
            workspace.hypotheses_args("--metrics-config", str(workspace.metrics))
        ).exit_code
        == 0
    )
    run_dir = workspace.run_dir

    manifest = manifest_of(run_dir)

    assert manifest.stage == "eval_run"
    assert manifest.seed == 20261110
    assert manifest.inputs == [
        entry(workspace.suite),
        entry(workspace.metrics),
        entry(workspace.alpha),
    ]
    assert manifest.outputs == [
        entry(run_dir / "hypotheses.jsonl", 20),
        entry(run_dir / FIRST / "result.json"),
        entry(run_dir / SECOND / "result.json"),
        entry(run_dir / "run.json"),
    ]
    assert manifest.created != ""
    assert manifest.code_version != ""
    assert len(manifest.config_sha256) == 64


def test_the_metrics_config_defaults_to_the_one_of_the_repository(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    shipped = REPO_ROOT / "configs" / "eval" / "metrics.yaml"
    assert manifest_of(workspace.run_dir).inputs[1] == entry(shipped)


def test_the_manifest_says_what_the_run_was(workspace: EvalWorkspace) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    assert manifest_of(workspace.run_dir).status == {
        "eval": "ok",
        "system_kind": "hypotheses",
        "purpose": "report",
        "unit": "document",
    }


def test_the_manifest_records_the_installed_versions_of_the_five_tools(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    assert manifest_of(workspace.run_dir).tool_versions == {
        package: importlib.metadata.version(package) for package in PACKAGES
    }


def test_the_metrics_of_the_manifest_are_the_segment_counts_and_the_values_that_were_measured(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0

    metrics = manifest_of(workspace.run_dir).metrics

    expected: dict[str, float] = {}
    for direction in (FIRST, SECOND):
        result = read_json(workspace.run_dir / direction / "result.json")
        expected[f"n_segments.{direction}"] = 10.0
        for name in ("bleu", "chrf", "chrf_pp"):
            expected[f"{name}.{direction}"] = result["metrics"][name]
    assert metrics == expected
    assert not any("comet22" in key for key in metrics), "a null metric is not a metric"


def test_a_neural_metric_that_was_given_scores_is_in_the_metrics(
    workspace: EvalWorkspace,
) -> None:
    from tests.support.eval_cli import scores_bytes

    scores = workspace.root / "in" / "comet22.jsonl"
    scores.write_bytes(
        scores_bytes(
            "comet22", sha(workspace.suite), sha(GOLDEN_OUTPUTS_FILES["alpha"]), 0.5
        )
    )

    assert (
        invoke(
            workspace.hypotheses_args("--segment-scores", f"comet22={scores}")
        ).exit_code
        == 0
    )

    manifest = manifest_of(workspace.run_dir)
    assert manifest.metrics[f"comet22.{FIRST}"] == 0.5
    assert manifest.inputs[-1] == entry(scores)


def test_the_seed_of_the_manifest_is_the_seed_of_the_bootstrap(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args("--seed", "7")).exit_code == 0

    assert manifest_of(workspace.run_dir).seed == 7


def test_the_config_digest_stands_on_the_options_and_the_metrics_config(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    first = manifest_of(workspace.run_dir).config_sha256
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    again = manifest_of(workspace.run_dir).config_sha256
    assert invoke(workspace.hypotheses_args("--resamples", "500")).exit_code == 0
    resampled = manifest_of(workspace.run_dir).config_sha256
    assert invoke(workspace.hypotheses_args("--direction", FIRST)).exit_code == 0
    selected = manifest_of(workspace.run_dir).config_sha256
    workspace.metrics.write_text(
        workspace.metrics.read_text("utf-8").replace("0.05", "0.1"), "utf-8"
    )
    assert (
        invoke(
            workspace.hypotheses_args("--metrics-config", str(workspace.metrics))
        ).exit_code
        == 0
    )
    reconfigured = manifest_of(workspace.run_dir).config_sha256

    assert first == again
    assert len({first, resampled, selected, reconfigured}) == 4


def test_the_manifest_of_a_model_run_names_the_five_files_of_the_export_and_its_manifest(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert (
        invoke(
            workspace.model_args("--metrics-config", str(workspace.metrics))
        ).exit_code
        == 0
    )
    run_dir = run_dir_of(workspace.out_dir, workspace.export.record.name)

    manifest = manifest_of(run_dir)

    export = workspace.export
    expected = [
        ManifestFile(
            path=str(export.directory / name),
            sha256=digest,
            bytes=(export.directory / name).stat().st_size,
        )
        for name, digest in export.sha256.items()
    ]
    expected.append(entry(export.directory / "manifest.json"))
    assert manifest.inputs == [
        entry(workspace.suite),
        entry(workspace.metrics),
        *expected,
    ]
    assert manifest.status["system_kind"] == "rinchen"


def test_the_manifest_of_a_baseline_run_names_the_config_and_the_files_of_the_model(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())
    assert (
        invoke(
            workspace.generator_args(
                "nllb_toy", "--metrics-config", str(workspace.metrics)
            )
        ).exit_code
        == 0
    )
    run_dir = run_dir_of(workspace.out_dir, "nllb_toy")

    manifest = manifest_of(run_dir)

    directory = workspace.root / "nllb"
    assert manifest.inputs[:3] == [
        entry(workspace.suite),
        entry(workspace.metrics),
        entry(workspace.generators),
    ]
    assert [Path(item.path) for item in manifest.inputs[3:]] == [
        directory / "model.bin",
        directory / "shared_vocabulary.json",
        directory / "config.json",
        directory / "sentencepiece.bpe.model",
    ]
    assert manifest.status["system_kind"] == "baseline"


def test_the_robustness_files_of_a_run_are_inputs_with_the_digests_that_were_hashed(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.eval_cli import robustness_directories

    robust = robustness_directories(workspace, monkeypatch)
    assert (
        invoke(
            workspace.hypotheses_args("--robustness", str(robust.named_alpha))
        ).exit_code
        == 0
    )

    inputs = manifest_of(workspace.run_dir).inputs

    assert inputs[-3:] == [
        entry(robust.named_alpha / "robustness.json"),
        entry(robust.named_alpha / "report.json"),
        entry(robust.named_alpha / "manifest.json"),
    ]


# --- the files a run replaces --------------------------------------------------------------------------


def test_a_second_run_into_the_same_directory_replaces_every_file(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    first = manifest_of(workspace.run_dir)
    before = listing(workspace.run_dir)

    assert invoke(workspace.hypotheses_args("--resamples", "500")).exit_code == 0

    second = manifest_of(workspace.run_dir)
    assert second.config_sha256 != first.config_sha256
    assert listing(workspace.run_dir) == before
    assert (
        read_json(workspace.run_dir / FIRST / "result.json")["bootstrap"]["resamples"]
        == 500
    )
    for item in second.outputs:
        assert sha(Path(item.path)) == item.sha256


def test_a_run_of_fewer_directions_removes_the_result_files_the_old_manifest_listed(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    assert (workspace.run_dir / SECOND / "result.json").is_file()

    assert invoke(workspace.hypotheses_args("--direction", FIRST)).exit_code == 0

    assert listing(workspace.run_dir) == [
        "hypotheses.jsonl",
        f"{FIRST}/result.json",
        "manifest.json",
        "run.json",
    ]
    assert not (workspace.run_dir / SECOND).exists(), (
        "the empty directory goes with its file"
    )
    assert [
        Path(item.path).name for item in manifest_of(workspace.run_dir).outputs
    ] == [
        "hypotheses.jsonl",
        "result.json",
        "run.json",
    ]


def test_only_the_files_the_old_manifest_listed_are_removed(
    workspace: EvalWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    run_dir = workspace.run_dir
    (run_dir / "notes.txt").write_text("the operator's own file", "utf-8")
    (run_dir / SECOND / "keep.txt").write_text("another file of the operator", "utf-8")

    assert invoke(workspace.hypotheses_args("--direction", FIRST)).exit_code == 0

    assert (run_dir / "notes.txt").read_text("utf-8") == "the operator's own file"
    assert (run_dir / SECOND / "keep.txt").read_text(
        "utf-8"
    ) == "another file of the operator"
    assert not (run_dir / SECOND / "result.json").exists()


def test_a_listed_path_is_taken_inside_the_run_directory_whatever_the_manifest_says(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.json").write_text("not ours", "utf-8")
    (outside / "result.json").write_text("not ours either", "utf-8")
    manifest = manifest_of(workspace.run_dir)
    manifest.outputs = [
        *manifest.outputs,
        ManifestFile(path=str(outside / "run.json"), sha256="0" * 64),
        ManifestFile(path=f"{outside}/../outside/result.json", sha256="0" * 64),
    ]
    store.write(workspace.run_dir, manifest)

    assert invoke(workspace.hypotheses_args()).exit_code == 0

    assert (outside / "run.json").read_text("utf-8") == "not ours"
    assert (outside / "result.json").read_text("utf-8") == "not ours either"


def test_a_directory_that_was_moved_is_still_cleaned_by_the_names_in_the_old_manifest(
    workspace: EvalWorkspace, tmp_path: Path
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    moved = tmp_path / "moved-out"
    shutil.move(str(workspace.out_dir), str(moved))

    assert (
        invoke(workspace.hypotheses_args("--direction", FIRST, out_dir=moved)).exit_code
        == 0
    )

    assert listing(run_dir_of(moved, "alpha")) == [
        "hypotheses.jsonl",
        f"{FIRST}/result.json",
        "manifest.json",
        "run.json",
    ]


def test_a_manifest_of_another_stage_in_the_run_directory_removes_nothing(
    workspace: EvalWorkspace,
) -> None:
    from tests.support.distill_cli import write_foreign_manifest

    run_dir = workspace.run_dir
    write_foreign_manifest(run_dir)
    (run_dir / "hypotheses.jsonl").write_text("not ours", "utf-8")
    before = snapshot(run_dir)

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 3, result.output
    assert snapshot(run_dir) == before


@pytest.mark.parametrize("name", ["hypotheses.jsonl", "result.json", "run.json"])
def test_a_file_that_cannot_be_written_exits_1_with_no_manifest_and_the_old_results_gone(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    run_dir = run_dir_of(workspace.out_dir, workspace.export.record.name)
    assert (run_dir / "manifest.json").is_file()
    engine = workspace.engine()
    install(monkeypatch, engine)
    fail_replace(monkeypatch, name)

    result = invoke(workspace.model_args())

    assert_one_error(result, 1, f"cannot write the evaluation run under {run_dir}")
    assert "disk full" in result.stderr
    assert result.stdout == ""
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "run.json").exists()
    assert engine.closed == 1


def test_a_manifest_that_cannot_be_written_exits_1_and_leaves_none(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, "manifest.json")

    result = invoke(workspace.hypotheses_args())

    assert_one_error(
        result, 1, f"cannot write the evaluation run under {workspace.run_dir}"
    )
    assert result.stdout == ""
    assert not (workspace.run_dir / "manifest.json").exists()
    assert (workspace.run_dir / "run.json").is_file()


def test_a_failed_second_run_leaves_no_manifest_and_a_good_one_replaces_every_file(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    first = manifest_of(workspace.run_dir)
    with monkeypatch.context() as failing:
        fail_replace(failing, "run.json")
        assert invoke(workspace.hypotheses_args()).exit_code == 1
    assert not (workspace.run_dir / "manifest.json").exists()

    result = invoke(workspace.hypotheses_args("--resamples", "500"))

    assert result.exit_code == 0, result.output
    second = manifest_of(workspace.run_dir)
    assert second.config_sha256 != first.config_sha256
    assert listing(workspace.run_dir) == [
        f"{SECOND}/result.json",
        "hypotheses.jsonl",
        f"{FIRST}/result.json",
        "manifest.json",
        "run.json",
    ]


# --- the manifest is last ---------------------------------------------------------------------------------


def test_the_files_appear_in_the_order_outputs_results_run_and_the_manifest_last(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = record_replaces(monkeypatch)

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert names == [
        "hypotheses.jsonl",
        "result.json",
        "result.json",
        "run.json",
        "manifest.json",
    ]


def test_the_manifest_and_the_old_files_are_gone_before_the_first_file_is_written(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    run_dir = workspace.run_dir
    seen: list[tuple[bool, bool]] = []
    real = Path.replace

    def replacing(self: Path, target: Any) -> Any:
        seen.append(
            ((run_dir / "manifest.json").exists(), (run_dir / "run.json").exists())
        )
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replacing)

    assert invoke(workspace.hypotheses_args()).exit_code == 0

    # Every file of the new run is written after the old manifest and the old run.json went;
    # the manifest, the last of the five, is written when the new run.json is there.
    assert seen[:4] == [(False, False)] * 4
    assert seen[4] == (False, True)


def test_nothing_is_written_before_every_score_and_every_input_was_read(
    workspace: EvalWorkspace,
) -> None:
    # A scores file that is refused (exit 3) comes after the run and before the first write,
    # so an earlier run is untouched.
    assert invoke(workspace.hypotheses_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    bad = workspace.root / "in" / "bad.jsonl"
    bad.write_text("not a scores file\n", "utf-8")

    result = invoke(workspace.hypotheses_args("--segment-scores", f"comet22={bad}"))

    assert result.exit_code == 3, result.output
    assert snapshot(workspace.out_dir) == before
