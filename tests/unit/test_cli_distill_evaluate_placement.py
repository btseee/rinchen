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
from rinchen.train.distill_saved_config import PlacementCheck
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


def entry(src_prefix: str, tgt_prefix: str) -> dict[str, str]:
    return {"src_prefix": src_prefix, "tgt_prefix": tgt_prefix}


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


def test_a_foreign_config_with_a_hostile_corpus_name_is_reported_on_one_clean_line(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    # A foreign model may be evaluated, so what its config holds is quoted, never copied raw.
    escape = chr(0x1B)
    name = f"odd\n{escape}[31mred{escape}[0m"
    save_config(workspace, {name: entry("<2khk_Cyrl> " + "y" * 3000, "")})

    result = invoke(workspace.args(*FAST, tag_placement="tgt"))

    assert_one_error(
        result,
        2,
        "the tag placement 'tgt' is not the one the checkpoint was trained with",
    )
    assert escape not in result.stderr
    assert "data['odd\\n\\x1b[31mred\\x1b[0m'].src_prefix" in result.stderr
    assert len(result.stderr) < 800
    assert predictor.calls == []


def test_the_config_is_read_only_as_the_file_that_was_hashed(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = save_config(workspace, launcher_data("src"))
    seen: list[tuple[Path, str]] = []
    real = distill_evaluate.require_saved_placement

    def spying(file: Path, digest: str, stated: Any) -> PlacementCheck:
        seen.append((file, digest))
        return real(file, digest, stated)

    monkeypatch.setattr(distill_evaluate, "require_saved_placement", spying)

    run_ok(workspace.args(*FAST))

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    recorded = {Path(item.path): item.sha256 for item in manifest.inputs}
    assert seen == [(config, hashlib.sha256(config.read_bytes()).hexdigest())]
    assert recorded[config] == seen[0][1]


# ---- whether the placement was checked ---------------------------------------------------

CHECKED_LINE = "tag placement: checked against the saved config"
STATED_LINE = "tag placement: stated, not recorded in the checkpoint"


def manifest_status(workspace: EvaluateWorkspace) -> dict[str, str]:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest.status


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_a_placement_the_saved_config_records_is_reported_as_checked(
    workspace: EvaluateWorkspace, predictor: OraclePredict, placement: str
) -> None:
    save_config(workspace, launcher_data(placement))

    result = run_ok(workspace.args(*FAST, tag_placement=placement))

    lines = result.stdout.splitlines()
    assert lines.count(CHECKED_LINE) == 1
    assert STATED_LINE not in lines
    assert lines[lines.index(CHECKED_LINE) - 1] == "dev lines in training: 0"
    assert manifest_status(workspace) == {
        "evaluate": "ok",
        "tag_placement": placement,
        "tag_placement_check": "checked",
    }


@pytest.mark.parametrize("placement", ["src", "tgt"])
@pytest.mark.parametrize(
    "data",
    [None, {}, {"corpus": {"src_prefix": "<bt>", "tgt_prefix": ""}}],
    ids=["no data section", "empty data section", "no direction tag"],
)
def test_a_placement_the_checkpoint_does_not_record_is_reported_as_stated(
    workspace: EvaluateWorkspace,
    predictor: OraclePredict,
    placement: str,
    data: Any,
) -> None:
    if data is not None:
        save_config(workspace, data)

    result = run_ok(workspace.args(*FAST, tag_placement=placement))

    lines = result.stdout.splitlines()
    assert lines.count(STATED_LINE) == 1
    assert CHECKED_LINE not in lines
    assert lines[lines.index(STATED_LINE) - 1] == "dev lines in training: 0"
    assert manifest_status(workspace) == {
        "evaluate": "ok",
        "tag_placement": placement,
        "tag_placement_check": "stated",
    }


def test_a_refused_placement_prints_no_line_about_the_check(
    workspace: EvaluateWorkspace, predictor: OraclePredict
) -> None:
    save_config(workspace, launcher_data("src"))

    result = invoke(workspace.args(*FAST, tag_placement="tgt"))

    assert result.exit_code == 2
    assert "tag placement:" not in result.stdout
