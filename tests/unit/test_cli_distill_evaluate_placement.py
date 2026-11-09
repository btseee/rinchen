"""`rinchen distill evaluate` against the tag placement the checkpoint was trained with (D10).

A checkpoint our launcher wrote records the placement in the `data` section of its saved
`config.json`. Scoring it with the other placement would only show as a low chrF, so the
command reads that section (the file was hashed before, and what is read is what was hashed)
and refuses a `--tag-placement` that disagrees, before anything is written. A checkpoint
without a `data` section, such as a baseline of another origin, is scored as before with the
placement the caller states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import distill_evaluate
from rinchen.manifest import store
from tests.support.distill_cli import (
    EvaluateWorkspace,
    OraclePredict,
    install_predictor,
    invoke,
    make_evaluate_workspace,
    run_ok,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.planted_student import Inputs, build_inputs
from tests.support.step_dirs import launcher_data
from tests.support.tiny_parallel import build_toy

FAST = ("--max-lines", "3")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("cli_distill_evaluate_placement"))


@pytest.fixture
def workspace(tmp_path: Path, inputs: Inputs) -> EvaluateWorkspace:
    return make_evaluate_workspace(tmp_path, inputs)


@pytest.fixture
def predictor(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> OraclePredict:
    oracle = OraclePredict(inputs.spm, build_toy().dev)
    install_predictor(monkeypatch, oracle)
    return oracle


def save_config(workspace: EvaluateWorkspace, data: Any) -> Path:
    config = workspace.checkpoint / "config.json"
    config.write_text(json.dumps({"data": data}), encoding="utf-8")
    return config


def test_a_placement_the_saved_config_contradicts_is_refused_before_anything_is_written(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    config = save_config(workspace, launcher_data("src"))
    before = snapshot(workspace.root)

    result = invoke(workspace.args(*FAST, tag_placement="tgt"))

    assert_one_error(
        result,
        2,
        "the tag placement 'tgt' is not the one the checkpoint was trained with",
    )
    assert str(config) in result.stderr
    assert "pass --tag-placement src" in result.stderr
    assert predictor.calls == []
    assert snapshot(workspace.root) == before


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_placement_the_saved_config_records_is_scored(
    workspace: EvaluateWorkspace, predictor: OraclePredict, placement: str
) -> None:
    save_config(workspace, launcher_data(placement))

    run_ok(workspace.args(*FAST, tag_placement=placement))

    assert len(predictor.calls) == 1
    assert (workspace.out_dir / "result.json").is_file()


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_a_checkpoint_without_a_direction_tag_in_its_config_is_scored_as_stated(
    workspace: EvaluateWorkspace, predictor: OraclePredict, placement: str
) -> None:
    # The fabricated checkpoint's config has no `data` section: a baseline of another origin.
    run_ok(workspace.args(*FAST, tag_placement=placement))

    assert len(predictor.calls) == 1


def test_a_saved_config_that_is_not_what_eole_writes_is_a_data_error(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    config = save_config(workspace, ["<2khk_Cyrl>"])

    result = invoke(workspace.args(*FAST))

    assert_one_error(result, 3, f"{config}: 'data' is not an object")
    assert predictor.calls == []
    assert not workspace.out_dir.exists()


def test_the_config_is_read_only_as_the_file_that_was_hashed(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = save_config(workspace, launcher_data("src"))
    seen: list[tuple[Path, str]] = []
    real = distill_evaluate.require_saved_placement

    def spying(file: Path, digest: str, stated: Any) -> None:
        seen.append((file, digest))
        real(file, digest, stated)

    monkeypatch.setattr(distill_evaluate, "require_saved_placement", spying)

    run_ok(workspace.args(*FAST))

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert seen == [(config, hashlib.sha256(config.read_bytes()).hexdigest())]
    assert recorded[config] == seen[0][1]
