"""The control profiles of the tests of `rinchen tiny run` (D15 Task 7, `tests/support/tiny_profiles.py`): a
copy of the shipped pipeline file with named values changed, paths made absolute and, when it points at a
copy of the fixtures, a registry of its own that the fetch stage accepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.acquire.registry import load_registry
from rinchen.config.tiny import load_tiny_config
from rinchen.errors import ConfigError
from rinchen.tiny.data_stages import run_fetch
from rinchen.tiny.fixtures import read_fixtures
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_profiles import control_profile, copy_fixtures
from tests.support.tiny_stage_inputs import make_context, shipped_config

pytestmark = pytest.mark.usefixtures("no_network")


def test_a_control_profile_is_the_shipped_one_with_the_named_values_changed(
    tmp_path: Path,
) -> None:
    path = control_profile(
        tmp_path, train_steps=400, budget_seconds=5, **{"filter.char_ratio_max": 3.0}
    )

    config = load_tiny_config(path.read_bytes())

    assert (
        config.train.steps,
        config.budget_seconds,
        config.filter.char_ratio_max,
    ) == (400, 5, 3.0)
    assert config.fixtures.sha256 == shipped_config().fixtures.sha256
    assert config.timeouts == shipped_config().timeouts
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for keys in (
        ("fixtures", "directory"),
        ("registry",),
        ("converter", "pairs"),
        ("train", "base"),
        ("serve", "config"),
    ):
        value: Any = document
        for key in keys:
            value = value[key]
        assert Path(value).is_absolute()


def test_a_control_profile_that_points_at_a_copy_of_the_fixtures_gets_a_registry_of_its_own(
    tmp_path: Path,
) -> None:
    copy = copy_fixtures(tmp_path / "copy")

    path = control_profile(tmp_path, fixtures_directory=str(copy))

    config = load_tiny_config(path.read_bytes())
    assert config.registry != shipped_config().registry
    entries = load_registry(config.registry)
    assert (
        sorted(Path(entry.url.removeprefix("local://")).parent for entry in entries)
        == [copy] * 7
    )


def test_the_fetch_stage_accepts_the_registry_of_a_control_profile_over_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    copy = copy_fixtures(tmp_path / "copy")
    config = load_tiny_config(
        control_profile(tmp_path, fixtures_directory=str(copy)).read_bytes()
    )
    fixtures = read_fixtures(copy, config.fixtures.sha256)

    result = run_fetch(make_context(tmp_path / "out", config=config, fixtures=fixtures))

    assert result.record.state == "ok"
    assert result.facts["files"] == 7


def test_a_control_profile_for_a_key_the_profile_does_not_have_is_an_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(KeyError):
        control_profile(tmp_path, no_such_section_here=1)
    with pytest.raises(ConfigError):
        control_profile(tmp_path, budget_seconds=0)


def test_a_profile_that_is_not_to_be_validated_is_written_as_it_is(
    tmp_path: Path,
) -> None:
    path = control_profile(tmp_path, validate=False, budget_seconds=0)

    with pytest.raises(ConfigError):
        load_tiny_config(path.read_bytes())
