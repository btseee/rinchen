"""`rinchen distill generate --qe`: the QE floor runs on the pairs (real source, candidate),
a scorer that cannot run is a refusal and never a silent skip, and the scorer is released
afterwards however the run ends.

The scorer is `tests.support.planted_synth.FakeScorer` behind the command's `build_qe_scorer`
seam: the real scorer needs a GPU and is never loaded (`EoleMetricXQEScorer` is only
constructed, which loads nothing).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from rinchen.cli import distill_generate
from rinchen.config.synth import SelectionConfig
from rinchen.errors import UpstreamError
from rinchen.filter import qe
from rinchen.filter.qe_eole import EoleMetricXQEScorer
from rinchen.manifest import store
from tests.support.corpus_rows import EN_MN
from tests.support.distill_cli import (
    CliTeacher,
    GenerateWorkspace,
    ReleasingScorer,
    install_scorer,
    install_teacher,
    invoke,
    make_generate_workspace,
    run_ok,
    write_selection_config,
)
from tests.support.launch_cli import assert_one_error

EN_PAIRS = {
    ("Hello", "Сайн байна уу"),
    ("Good morning", "Өглөөний мэнд"),
    ("Thank you", "Баярлалаа танд"),
    ("Water", "Хүйтэн ус"),
}


@pytest.fixture
def workspace(tmp_path: Path) -> GenerateWorkspace:
    return make_generate_workspace(tmp_path)


def pairs_scored(scorer: ReleasingScorer) -> set[tuple[str, str]]:
    return {
        pair
        for sources, translations in scorer.calls
        for pair in zip(sources, translations, strict=True)
    }


def test_the_floor_runs_on_the_real_source_and_the_candidate_and_drops_the_bad_ones(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    scorer = ReleasingScorer({("Water", "Хүйтэн ус"): 9.0})
    seam = install_scorer(monkeypatch, scorer)

    result = run_ok(workspace.args("--qe"))

    assert pairs_scored(scorer) == EN_PAIRS
    lines = result.stdout.splitlines()
    assert "qe: requested" in lines
    assert lines[-3] == "qe: ok"
    assert seam.built == [SelectionConfig()]
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert manifest.status["qe"] == "ok"
    assert manifest.metrics["count.total.dropped_qe"] == 1.0
    assert manifest.metrics["count.total.selected"] == 3.0
    assert manifest.metrics[f"count.{EN_MN}.no_survivor"] == 1.0
    assert workspace.counts_document()["qe_status"] == "ok"


def test_the_qe_scores_of_the_kept_rows_are_in_the_corpus(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    install_scorer(monkeypatch, ReleasingScorer({("Water", "Хүйтэн ус"): 9.0}))

    run_ok(workspace.args("--qe"))

    frame = pl.read_parquet(workspace.kd_corpus)
    assert frame.height == 3
    assert frame["scores"].struct.field("metricx_qe").to_list() == [1.0, 1.0, 1.0]
    assert "Хүйтэн ус" not in set(frame["tgt"])


def test_without_the_option_no_scorer_is_built_and_the_status_says_so(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    seam = install_scorer(monkeypatch, ReleasingScorer())

    result = run_ok(workspace.args())

    assert seam.built == []
    assert "qe: skipped: not requested" in result.stdout.splitlines()
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert manifest.status["qe"] == "skipped: not requested"
    frame = pl.read_parquet(workspace.kd_corpus)
    assert frame["scores"].struct.field("metricx_qe").null_count() == frame.height


def test_the_scorer_is_released_once_after_a_run(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    scorer = ReleasingScorer()
    install_scorer(monkeypatch, scorer)

    run_ok(workspace.args("--qe"))

    assert scorer.released == 1


def test_the_scorer_is_released_when_the_teacher_fails(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(
        monkeypatch, CliTeacher(fail_on_call={0: UpstreamError("the teacher is down")})
    )
    scorer = ReleasingScorer()
    install_scorer(monkeypatch, scorer)

    result = invoke(workspace.args("--qe"))

    assert result.exit_code == 4, result.output
    assert scorer.released == 1


class FailingScorer(ReleasingScorer):
    """A scorer whose first call raises `error`."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        raise self._error


def test_the_scorer_is_released_when_the_scorer_fails(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    scorer = FailingScorer(RuntimeError("scorer out of memory"))
    install_scorer(monkeypatch, scorer)

    result = invoke(workspace.args("--qe"))

    assert isinstance(result.exception, RuntimeError)
    assert scorer.released == 1
    assert not workspace.manifest.exists()


def test_a_scorer_that_cannot_reach_its_files_is_an_upstream_error(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    scorer = FailingScorer(FileNotFoundError("no such model: metricx-24"))
    install_scorer(monkeypatch, scorer)

    result = invoke(workspace.args("--qe"))

    assert_one_error(result, 4, "the QE scorer failed: no such model: metricx-24")
    assert "cannot write" not in result.stderr
    assert scorer.released == 1
    assert not workspace.manifest.exists()


def test_the_scorer_is_released_when_the_run_is_interrupted(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(fail_on_call={0: KeyboardInterrupt()}))
    scorer = ReleasingScorer()
    install_scorer(monkeypatch, scorer)

    result = invoke(workspace.args("--qe"))

    assert result.exit_code == 130, result.output
    assert scorer.released == 1


# --- a scorer that cannot run is a refusal ------------------------------------------------------------


def test_a_scorer_that_needs_cuda_is_refused_when_there_is_none_and_released(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam_teacher = install_teacher(monkeypatch)
    scorer = ReleasingScorer(requires_cuda=True)
    install_scorer(monkeypatch, scorer)
    monkeypatch.setattr(qe, "cuda_is_available", lambda: False)

    result = invoke(workspace.args("--qe"))

    assert_one_error(result, 2, "--qe was asked for but the QE scorer cannot run")
    assert "no CUDA device" in result.stderr
    assert scorer.released == 1
    assert scorer.calls == []
    assert not workspace.out_dir.exists()
    assert seam_teacher.built == []


def test_no_scorer_for_the_configured_kind_is_refused_not_skipped(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam_teacher = install_teacher(monkeypatch)
    install_scorer(monkeypatch, None)

    result = invoke(workspace.args("--qe"))

    assert_one_error(result, 2, "--qe was asked for but the QE scorer cannot run")
    assert "no QE scorer configured" in result.stderr
    assert not workspace.out_dir.exists()
    assert seam_teacher.built == []


def test_the_real_seam_has_no_scorer_for_cometkiwi_and_the_command_refuses(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_teacher(monkeypatch)
    config = write_selection_config(tmp_path / "kiwi.yaml", qe={"scorer": "cometkiwi"})

    result = invoke(workspace.args("--qe", "--selection-config", str(config)))

    assert_one_error(result, 2, "--qe was asked for but the QE scorer cannot run")
    assert "no QE scorer configured" in result.stderr
    assert not workspace.out_dir.exists()


def test_the_real_seam_builds_the_metricx_scorer_which_needs_a_gpu_this_machine_hides(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    monkeypatch.setattr(qe, "cuda_is_available", lambda: False)

    result = invoke(workspace.args("--qe"))

    assert_one_error(result, 2, "--qe was asked for but the QE scorer cannot run")
    assert "no CUDA device" in result.stderr
    assert not workspace.out_dir.exists()


def test_build_qe_scorer_builds_the_eole_scorer_only_for_metricx() -> None:
    scorer = distill_generate.build_qe_scorer(SelectionConfig())

    assert isinstance(scorer, EoleMetricXQEScorer)
    assert (
        distill_generate.build_qe_scorer(
            SelectionConfig.model_validate({"qe": {"scorer": "cometkiwi"}})
        )
        is None
    )
