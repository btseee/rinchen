"""`rinchen eval run` with the two optional inputs of a result (D13, Task 7): D12's robustness
directory (`--robustness`) and the per-segment scores of the neural seam (`--segment-scores`).

The robustness files are the product of D12's own command over D12's golden suite
(`tests.support.eval_cli.robustness_directories`); the block, its coverage and its source reach
every `result.json` WITH the coverage and a verified binding to the system (`model_digest`, or the
weaker `name` for a system that is not a Rinchen model). The scores are written by the helper of
the seam (`scores_to_bytes`); no scorer exists, so what the mean and the interval of a given score
are is what the bootstrap of integer micro-units makes of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.eval.outputs import OutputKey
from rinchen.eval.result import run_from_bytes
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_cli import (
    FIRST,
    SECOND,
    EvalWorkspace,
    RobustnessDirectories,
    install,
    invoke,
    make_workspace,
    read_json,
    rewrite_robustness_file,
    robustness_directories,
    run_dir_of,
    scores_bytes,
)
from tests.support.eval_fakes import GOLDEN_OUTPUTS_FILES, FakeCandidateTranslator
from tests.support.launch_cli import assert_one_error
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_golden import GOLDEN_SUITE_SHA256 as ROBUST_SUITE_SHA256

pytestmark = pytest.mark.usefixtures("no_network")

NEURAL = ("comet22", "metricx24", "metricx24_qe")


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def robust(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> RobustnessDirectories:
    return robustness_directories(workspace, monkeypatch)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_sha(workspace: EvalWorkspace) -> str:
    return sha(workspace.suite)


ALPHA_OUTPUTS_SHA = hashlib.sha256(
    GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()
).hexdigest()


def varying(key: OutputKey) -> float:
    """0.51 to 0.60 over the ten rows of a direction (row g01 to g10)."""
    return 0.5 + 0.01 * int(key.id[1:])


# --- the robustness block ----------------------------------------------------------------------------


def assert_block_embedded(result: dict[str, Any], directory: Path) -> None:
    """The block, the coverage and the source of the result are D12's files' own."""
    d12 = read_json(directory / "robustness.json")
    report = read_json(directory / "report.json")
    assert result["robustness"] == d12["robustness"] == report["robustness"]
    assert result["robustness"]["numbers_acc"] == 0.5
    assert result["robustness_coverage"] == report["coverage"]
    assert result["robustness_coverage"]["numbers_acc"] == {
        "scored": 6,
        "excluded": 2,
        "total": 8,
    }
    source = result["robustness_source"]
    assert source["suite"] == "golden"
    assert source["suite_sha256"] == ROBUST_SUITE_SHA256
    assert source["robustness_sha256"] == sha(directory / "robustness.json")
    assert source["report_sha256"] == sha(directory / "report.json")


def test_a_hypotheses_run_named_like_the_robustness_run_embeds_the_block_bound_by_name(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert result.exit_code == 0, result.output
    for direction in (FIRST, SECOND):
        written = read_json(workspace.run_dir / direction / "result.json")
        assert_block_embedded(written, robust.named_alpha)
        assert written["robustness_source"]["binding"] == "name"
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert (
        dict(record.robustness_source or {})
        == read_json(workspace.run_dir / FIRST / "result.json")["robustness_source"]
    )


def test_a_model_run_is_bound_to_the_robustness_run_by_the_digest_of_the_model(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--robustness", str(robust.model)))

    assert result.exit_code == 0, result.output
    run_dir = run_dir_of(workspace.out_dir, workspace.export.record.name)
    for direction in (FIRST, SECOND):
        written = read_json(run_dir / direction / "result.json")
        assert_block_embedded(written, robust.model)
        assert written["robustness_source"]["binding"] == "model_digest"


def test_the_robustness_of_another_model_is_refused_and_nothing_is_written(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    result = invoke(workspace.model_args("--robustness", str(robust.other_model)))

    assert_one_error(result, 2, "the robustness run is of another model")
    assert result.stdout == ""
    assert seams.opens == [] and engine.closed == 0
    assert not workspace.out_dir.exists()


def test_the_robustness_of_another_model_is_refused_before_the_engine_is_opened(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    result = invoke(workspace.model_args("--robustness", str(robust.other_model)))

    assert result.exit_code == 2
    assert seams.opens == [] and seams.events == []
    assert engine.calls == [] and engine.closed == 0


def test_a_model_without_a_robustness_model_is_refused_before_the_engine_is_opened(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = workspace.engine()
    seams = install(monkeypatch, engine)

    result = invoke(
        workspace.model_args(
            "--system", "alpha", "--robustness", str(robust.named_alpha)
        )
    )

    assert_one_error(result, 2, "the robustness run was made without a model")
    assert seams.opens == [] and engine.calls == []


def test_a_baseline_of_another_name_is_refused_before_its_translator_is_built(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    result = invoke(
        workspace.generator_args("nllb_toy", "--robustness", str(robust.named_alpha))
    )

    assert_one_error(result, 2, "the robustness run is of another system")
    assert seams.builds == []


def test_the_model_that_is_bound_still_opens_the_engine_once(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = install(monkeypatch, workspace.engine())

    result = invoke(workspace.model_args("--robustness", str(robust.model)))

    assert result.exit_code == 0, result.output
    assert len(seams.opens) == 1


def test_a_hypotheses_system_of_another_name_than_the_robustness_run_is_refused(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    result = invoke(
        workspace.hypotheses_args(
            "--system", "beta", "--robustness", str(robust.named_alpha)
        )
    )

    assert_one_error(result, 2, "the robustness run is of another system")
    assert "'alpha'" in result.stderr
    assert "'beta'" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_model_is_never_bound_by_name_to_a_robustness_run_that_had_no_model(
    workspace: EvalWorkspace,
    robust: RobustnessDirectories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, workspace.engine())

    result = invoke(
        workspace.model_args(
            "--system", "alpha", "--robustness", str(robust.named_alpha)
        )
    )

    assert_one_error(result, 2, "the robustness run was made without a model")
    assert not workspace.out_dir.exists()


def test_a_run_in_which_no_direction_ran_still_checks_the_binding(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    # Every direction is left out, yet a block that describes another system must not pass.
    only = workspace.root / "in" / "only.jsonl"
    only.write_bytes(b"")
    result = invoke(
        workspace.hypotheses_args(
            "--robustness", str(robust.named_alpha), hypotheses=only
        )
    )

    assert_one_error(result, 2, "the robustness run is of another system")


def test_a_robustness_directory_without_its_files_is_a_data_error(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    (robust.named_alpha / "report.json").unlink()

    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert_one_error(result, 3, "robustness report not found")
    assert not workspace.out_dir.exists()


def test_a_directory_whose_manifest_is_of_another_stage_is_a_data_error(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    (robust.named_alpha / "manifest.json").unlink()
    write_foreign_manifest(robust.named_alpha)

    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert_one_error(result, 3, "")
    assert "eval_robustness" in result.stderr
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("name", ["robustness.json", "report.json"])
def test_a_manifest_that_does_not_vouch_for_the_files_as_they_are_is_a_data_error(
    workspace: EvalWorkspace, robust: RobustnessDirectories, name: str
) -> None:
    # The same document with other whitespace: it still parses and still agrees with its twin,
    # so only the manifest's own digest can tell that the file is not the one it listed.
    changed = robust.named_alpha / name
    document = json.loads(changed.read_text("utf-8"))
    changed.write_text(json.dumps(document, indent=4, sort_keys=True) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert_one_error(
        result, 3, "the robustness directory's manifest does not vouch for "
    )
    assert name in result.stderr
    assert not workspace.out_dir.exists()


def test_a_file_the_manifest_does_not_list_is_a_data_error(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    from rinchen.manifest import store

    manifest = store.read(robust.named_alpha)
    assert manifest is not None
    manifest.outputs = [
        i for i in manifest.outputs if not i.path.endswith("report.json")
    ]
    store.write(robust.named_alpha, manifest)

    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert_one_error(
        result, 3, "the robustness directory's manifest does not vouch for report.json"
    )
    assert not workspace.out_dir.exists()


def test_a_report_that_names_another_digest_of_robustness_json_is_a_data_error(
    workspace: EvalWorkspace, robust: RobustnessDirectories
) -> None:
    # The manifest vouches for the changed report, so only the report itself is at fault.
    rewrite_robustness_file(
        robust.named_alpha,
        "report.json",
        lambda document: document.__setitem__("robustness_sha256", "f" * 64),
    )

    result = invoke(workspace.hypotheses_args("--robustness", str(robust.named_alpha)))

    assert result.exit_code == 3, result.output
    assert "robustness_sha256" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_robustness_directory_that_was_moved_still_works(
    workspace: EvalWorkspace, robust: RobustnessDirectories, tmp_path: Path
) -> None:
    moved = tmp_path / "moved-robustness"
    robust.named_alpha.rename(moved)

    result = invoke(workspace.hypotheses_args("--robustness", str(moved)))

    assert result.exit_code == 0, result.output


# --- the segment scores --------------------------------------------------------------------------------


def write_scores(workspace: EvalWorkspace, metric: str, **options: Any) -> Path:
    path = workspace.root / "in" / f"{metric}.jsonl"
    path.write_bytes(
        scores_bytes(
            metric,
            options.pop("suite_sha256", suite_sha(workspace)),
            options.pop("hypotheses_sha256", ALPHA_OUTPUTS_SHA),
            **options,
        )
    )
    return path


def test_a_metric_with_scores_is_the_mean_with_its_interval_and_the_others_stay_null(
    workspace: EvalWorkspace,
) -> None:
    scores = write_scores(workspace, "comet22", value=varying)

    result = invoke(workspace.hypotheses_args("--segment-scores", f"comet22={scores}"))

    assert result.exit_code == 0, result.output
    for direction in (FIRST, SECOND):
        written = read_json(workspace.run_dir / direction / "result.json")
        assert written["metrics"]["comet22"] == pytest.approx(0.555, abs=1e-9)
        low, high = written["ci95"]["comet22"]
        assert 0.51 <= low <= 0.555 <= high <= 0.60
        for name in ("metricx24", "metricx24_qe"):
            assert written["metrics"][name] is None
            assert written["ci95"][name] is None
            assert written["unmeasured"][name]
        assert "comet22" not in written["unmeasured"]
        assert written["bootstrap"]["unit"] == "document"
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert dict(record.segment_scores) == {"comet22": sha(scores)}


def test_two_metrics_can_be_given_and_the_one_not_given_stays_unmeasured(
    workspace: EvalWorkspace,
) -> None:
    comet = write_scores(workspace, "comet22", value=0.8)
    metricx = write_scores(workspace, "metricx24", value=3.25)

    result = invoke(
        workspace.hypotheses_args(
            "--segment-scores",
            f"comet22={comet}",
            "--segment-scores",
            f"metricx24={metricx}",
        )
    )

    assert result.exit_code == 0, result.output
    written = read_json(workspace.run_dir / FIRST / "result.json")
    assert (written["metrics"]["comet22"], written["metrics"]["metricx24"]) == (
        0.8,
        3.25,
    )
    assert written["metrics"]["metricx24_qe"] is None
    assert sorted(written["unmeasured"]) == ["metricx24_qe"]
    record = run_from_bytes((workspace.run_dir / "run.json").read_bytes())
    assert dict(record.segment_scores) == {
        "comet22": sha(comet),
        "metricx24": sha(metricx),
    }


def test_the_scores_must_name_the_hypotheses_file_this_run_writes_not_the_input_file(
    workspace: EvalWorkspace,
) -> None:
    # The same outputs, the lines in another order: the file is not canonical, so its digest is
    # not the digest of the `hypotheses.jsonl` the run writes and re-scores from.
    lines = workspace.alpha.read_text("utf-8").splitlines()
    workspace.alpha.write_text("\n".join(reversed(lines)) + "\n", "utf-8")
    input_sha = sha(workspace.alpha)
    assert input_sha != ALPHA_OUTPUTS_SHA
    named_after_input = write_scores(workspace, "comet22", hypotheses_sha256=input_sha)

    refused = invoke(
        workspace.hypotheses_args("--segment-scores", f"comet22={named_after_input}")
    )

    assert refused.exit_code == 3, refused.output
    assert "hypotheses_sha256" in refused.stderr
    assert not workspace.out_dir.exists()
    named_after_run = write_scores(workspace, "comet22")

    accepted = invoke(
        workspace.hypotheses_args("--segment-scores", f"comet22={named_after_run}")
    )

    assert accepted.exit_code == 0, accepted.output
    assert (workspace.run_dir / "hypotheses.jsonl").read_bytes() == (
        GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()
    )


@pytest.mark.parametrize(
    "options",
    [
        {"hypotheses_sha256": "e" * 64},
        {"suite_sha256": "e" * 64},
    ],
    ids=["other-hypotheses", "other-suite"],
)
def test_scores_of_other_hypotheses_or_another_suite_are_a_data_error(
    workspace: EvalWorkspace, options: dict[str, Any]
) -> None:
    scores = write_scores(workspace, "comet22", **options)

    result = invoke(workspace.hypotheses_args("--segment-scores", f"comet22={scores}"))

    assert result.exit_code == 3, result.output
    assert not workspace.out_dir.exists()


def test_scores_of_a_header_that_names_another_metric_are_a_data_error(
    workspace: EvalWorkspace,
) -> None:
    scores = write_scores(workspace, "metricx24")

    result = invoke(workspace.hypotheses_args("--segment-scores", f"comet22={scores}"))

    assert result.exit_code == 3, result.output
    assert "metric" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_scores_file_that_lacks_a_segment_is_a_data_error(
    workspace: EvalWorkspace,
) -> None:
    scores = write_scores(workspace, "comet22")
    lines = scores.read_text("utf-8").splitlines()
    scores.write_text("\n".join(lines[:-1]) + "\n", "utf-8")

    result = invoke(workspace.hypotheses_args("--segment-scores", f"comet22={scores}"))

    assert result.exit_code == 3, result.output
    assert not workspace.out_dir.exists()


def test_scores_for_a_direction_that_was_not_run_are_a_data_error(
    workspace: EvalWorkspace,
) -> None:
    # The run scores one direction, so the file must hold the segments of that one only.
    scores = write_scores(workspace, "comet22")

    result = invoke(
        workspace.hypotheses_args(
            "--direction", FIRST, "--segment-scores", f"comet22={scores}"
        )
    )

    assert result.exit_code == 3, result.output
    assert not workspace.out_dir.exists()


def test_a_scores_file_that_is_missing_is_a_data_error(
    workspace: EvalWorkspace,
) -> None:
    result = invoke(
        workspace.hypotheses_args(
            "--segment-scores", f"comet22={workspace.root / 'none.jsonl'}"
        )
    )

    assert_one_error(result, 3, "scores file not found")
    assert not workspace.out_dir.exists()


def test_scores_given_with_a_model_or_a_generator_are_a_config_error(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = write_scores(workspace, "comet22")
    seams = install(monkeypatch, workspace.engine(), FakeCandidateTranslator())

    with_model = invoke(workspace.model_args("--segment-scores", f"comet22={scores}"))
    with_generator = invoke(
        workspace.generator_args("vllm_toy", "--segment-scores", f"comet22={scores}")
    )

    assert_one_error(with_model, 2, "--segment-scores needs --hypotheses")
    assert_one_error(with_generator, 2, "--segment-scores needs --hypotheses")
    assert seams.opens == []
    assert seams.builds == []
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("bleu={file}", "--segment-scores must be METRIC=FILE"),
        ("chrf_pp={file}", "--segment-scores must be METRIC=FILE"),
        ("comet22", "--segment-scores must be METRIC=FILE"),
        ("comet22=", "--segment-scores must be METRIC=FILE"),
        ("=file", "--segment-scores must be METRIC=FILE"),
    ],
)
def test_a_metric_that_is_not_one_of_the_three_or_a_malformed_argument_is_a_config_error(
    workspace: EvalWorkspace, argument: str, message: str
) -> None:
    scores = write_scores(workspace, "comet22")

    result = invoke(
        workspace.hypotheses_args("--segment-scores", argument.format(file=scores))
    )

    assert_one_error(result, 2, message)
    for name in NEURAL:
        assert name in result.stderr
    assert not workspace.out_dir.exists()


def test_a_metric_given_twice_is_a_config_error(workspace: EvalWorkspace) -> None:
    scores = write_scores(workspace, "comet22")

    result = invoke(
        workspace.hypotheses_args(
            "--segment-scores",
            f"comet22={scores}",
            "--segment-scores",
            f"comet22={scores}",
        )
    )

    assert_one_error(result, 2, "--segment-scores names a metric more than once")
    assert not workspace.out_dir.exists()
