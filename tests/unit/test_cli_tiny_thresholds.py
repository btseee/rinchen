"""`rinchen tiny run` over fakes, the thresholds at the command (D15 fix wave, findings A-M1 and A-M6).

The `stages` threshold counts a stage that was skipped with no failure before it, so a run in which a
stage was dropped without a cause exits 4 although no stage failed and no figure is missing; and a
profile never holds a bound looser than the master plan's, so a `--config` that lowers one is exit 2,
names the field and the bound, quotes no value of the file and writes and starts nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import Outcome, go, lines_of, report_of
from tests.support.tiny_profiles import control_profile

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.mark.parametrize(
    "reason",
    [
        "an earlier stage failed",
        "no language-identification model: GlotLID is a download",
    ],
)
def test_a_stage_skipped_when_nothing_failed_exits_4_and_fails_the_stages_threshold(
    reason: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `serve` carries no figure of a threshold: only the `stages` threshold can notice it was dropped.
    out = tmp_path / "out"

    result, _, _ = go(
        monkeypatch, out, outcomes={"serve": Outcome("skipped", reason=reason)}
    )

    assert result.exit_code == 4
    assert not [line for line in lines_of(result)[:15] if ": fail" in line]
    thresholds = {item["id"]: item["verdict"] for item in report_of(out)["thresholds"]}
    assert thresholds["stages"] == "fail"
    assert {name for name, verdict in thresholds.items() if verdict == "fail"} == {
        "stages"
    }
    assert report_of(out)["verdict"] == "fail"


@pytest.mark.parametrize(
    ("override", "floor"),
    [
        ({"thresholds_train_chrf_min": 12.375}, "at least 50.0"),
        ({"thresholds_parity_exact_min": 0.5}, "at least 0.95"),
        ({"thresholds_parity_delta_max": 0.5}, "at most 0.001"),
        ({"thresholds_int8_chrf_drop_max": 7.375}, "at most 1.0"),
        ({"thresholds_script_purity_min": 0.5}, "at least 1.0"),
        ({"thresholds_placeholder_acc_min": 0.5}, "at least 1.0"),
    ],
)
def test_a_config_that_lowers_a_bound_is_exit_2_and_names_the_field_and_the_bound(
    override: dict[str, float],
    floor: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = control_profile(tmp_path, validate=False, **override)

    result, recording, _ = go(monkeypatch, tmp_path / "out", config=config)

    (name,) = override
    assert result.exit_code == 2
    assert f"thresholds.{name.removeprefix('thresholds_')}" in result.stderr
    assert floor in result.stderr
    assert str(next(iter(override.values()))) not in result.stderr
    assert recording.stages == []
    assert not (tmp_path / "out").exists()
