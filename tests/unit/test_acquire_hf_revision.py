"""`rinchen.acquire.hf.fetch` and the `revision` keyword D13 gives it (Task 4).

The real `fetch` runs with `datasets.load_dataset` replaced by a recording fake (the design spike's
finding F7: `fetch` has no network-free test of its own), so no socket is opened and nothing is
downloaded. Without a revision the call to `load_dataset` is exactly the one D1 wrote (the two
positional arguments, `split=` and `streaming=True`, and nothing else); with one, the revision
arrives as the only extra keyword. `tests/unit/test_acquire_hf.py` (the tests of `write_shards`)
passes unchanged.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rinchen.acquire import hf
from tests.support.eval_suite_cli import (
    REVISION,
    FakeLoadDataset,
    install_fake_load_dataset,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

DATASET = "openlanguagedata/flores_plus"
ROWS = [
    {"id": "a", "text": "one"},
    {"id": "b", "text": "two"},
    {"id": "c", "text": "three"},
]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLoadDataset:
    return install_fake_load_dataset(monkeypatch, FakeLoadDataset({"eng_Latn": ROWS}))


def test_without_a_revision_the_call_is_the_one_d1_wrote(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    hf.fetch(DATASET, tmp_path, config="eng_Latn", split="dev")

    assert [call.as_tuple() for call in fake.calls] == [
        ("openlanguagedata/flores_plus", "eng_Latn", "dev", True, {})
    ]


def test_a_revision_of_none_changes_nothing_in_the_call(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    hf.fetch(DATASET, tmp_path, config="eng_Latn", split="dev", revision=None)

    assert [call.as_tuple() for call in fake.calls] == [
        ("openlanguagedata/flores_plus", "eng_Latn", "dev", True, {})
    ]
    assert "revision" not in fake.calls[0].extra


def test_a_revision_arrives_as_the_only_extra_keyword(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    hf.fetch(DATASET, tmp_path, config="eng_Latn", split="dev", revision=REVISION)

    assert [call.as_tuple() for call in fake.calls] == [
        (
            "openlanguagedata/flores_plus",
            "eng_Latn",
            "dev",
            True,
            {"revision": "a" * 40},
        )
    ]


def test_the_rows_of_the_stream_are_written_as_shards_whatever_the_revision(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    paths = hf.fetch(
        DATASET, tmp_path / "out", config="eng_Latn", split="dev", revision=REVISION
    )

    assert [path.name for path in paths] == ["shard-00000.parquet"]
    assert pl.read_parquet(paths[0]).to_dicts() == ROWS


def test_the_shard_size_still_reaches_the_writer(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    paths = hf.fetch(
        DATASET, tmp_path, config="eng_Latn", split="dev", shard_size=2, revision=None
    )

    assert [pl.read_parquet(path).height for path in paths] == [2, 1]
