"""`rinchen.tiny.context`: what a stage is handed and what it hands back, and the directories of the
output (D15, spec P2 and P11).

The stage ids are a closed list and so are the eleven directories a run writes under `--out-dir`:
`environment` and `serve` write none, and `prepare`, `train` and `average` share `07_train`. A test
pins both tables, because a renamed or reordered directory moves every path of the report.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.context import (
    STAGE_DIRECTORIES,
    StageContext,
    StageResult,
    stage_dir,
)
from rinchen.tiny.states import StageRecord
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

EXPECTED_DIRECTORIES = {
    "fetch": "01_fetch",
    "corpus": "02_corpus",
    "filter": "03_filter",
    "spm": "04_spm",
    "converter": "05_converter",
    "synth": "06_synth",
    "prepare": "07_train",
    "train": "07_train",
    "average": "07_train",
    "export": "08_export",
    "parity": "09_parity",
    "robustness": "10_robustness",
    "evaluate": "11_evaluate",
}


def test_the_directories_are_the_eleven_of_the_spec() -> None:
    assert dict(STAGE_DIRECTORIES) == EXPECTED_DIRECTORIES
    assert len(set(STAGE_DIRECTORIES.values())) == 11


def test_a_directory_name_is_two_digits_and_a_word() -> None:
    for name in STAGE_DIRECTORIES.values():
        assert re.fullmatch(r"[0-9]{2}_[a-z]+", name), name


def test_the_directories_are_numbered_in_the_order_of_the_stages() -> None:
    numbers = [
        int(STAGE_DIRECTORIES[stage][:2])
        for stage in STAGE_IDS
        if stage in STAGE_DIRECTORIES
    ]
    assert numbers == sorted(numbers)
    assert sorted(set(numbers)) == list(range(1, 12))


def test_the_stages_without_a_directory_are_the_environment_and_the_service() -> None:
    assert [stage for stage in STAGE_IDS if stage not in STAGE_DIRECTORIES] == [
        "environment",
        "serve",
    ]


def test_the_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        STAGE_DIRECTORIES["fetch"] = "x"  # type: ignore[index]


@pytest.mark.parametrize("stage_id", sorted(EXPECTED_DIRECTORIES))
def test_stage_dir_is_the_directory_under_the_output(stage_id: str) -> None:
    out = Path("/some/out")
    assert stage_dir(out, stage_id) == out / EXPECTED_DIRECTORIES[stage_id]


@pytest.mark.parametrize("stage_id", ["environment", "serve"])
def test_a_stage_that_writes_no_directory_has_none(stage_id: str) -> None:
    assert stage_dir(Path("/some/out"), stage_id) is None


@pytest.mark.parametrize("stage_id", ["", "tiny", "01_fetch", "Fetch", "logs"])
def test_an_id_outside_the_stage_ids_is_an_error_that_quotes_nothing(
    stage_id: str,
) -> None:
    with pytest.raises(ValueError, match="stage") as caught:
        stage_dir(Path("/some/out"), stage_id)
    assert stage_id not in str(caught.value) or stage_id == ""


def test_the_three_stages_of_the_training_share_one_directory() -> None:
    out = Path("out")
    assert (
        stage_dir(out, "prepare")
        == stage_dir(out, "train")
        == stage_dir(out, "average")
    )


def test_a_result_holds_its_record_and_its_facts() -> None:
    record = StageRecord("fetch", "ok")
    result = StageResult(record, {"files": 7})
    assert result.record is record
    assert result.facts == {"files": 7}
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.facts = {}  # type: ignore[misc]


def test_a_context_is_frozen_and_holds_the_fields_of_the_interface() -> None:
    names = [field.name for field in dataclasses.fields(StageContext)]
    assert names == [
        "config",
        "fixtures",
        "out_dir",
        "runner",
        "qe_fake",
        "threads",
        "remaining_seconds",
    ]
    assert StageContext.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_a_context_hands_its_fields_to_a_stage() -> None:
    clock: Callable[[], float] = lambda: 12.5
    marker: Any = object()
    context = StageContext(
        config=marker,
        fixtures=marker,
        out_dir=Path("out"),
        runner=marker,
        qe_fake=True,
        threads=2,
        remaining_seconds=clock,
    )
    assert context.config is marker
    assert context.out_dir == Path("out")
    assert (context.qe_fake, context.threads) == (True, 2)
    assert context.remaining_seconds() == 12.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.threads = 3  # type: ignore[misc]
