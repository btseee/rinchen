"""What the card's limits say about the GPU (D13 fix wave 1, M5): it used to say "any run on a GPU
(none was made)" whatever the report recorded, while the card prints `device=` from the run options.
Now the sentence follows the report: when every run of the report that records a device used `cpu`
(or none records one) it says that no GPU run is recorded in the report, and when a run used another
device it says that the figures of that run come from a GPU and are not reproduced here.
"""

from __future__ import annotations

import pytest

from rinchen.export.record import ModelRecord
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    ENGINE_OPTIONS,
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    OTHER,
    SYSTEM,
    bound_document,
    view_of,
)
from tests.support.release_fixture import SIX_DIRECTIONS, card_of

pytestmark = pytest.mark.usefixtures("no_network")

NO_GPU = "- No run on a GPU is recorded in this report.\n"
GPU = (
    "- A run of this report used the device cuda: its figures come from that run on a GPU "
    "and are not reproduced here.\n"
)


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def bound(**options: object) -> dict:  # type: ignore[type-arg]
    return bound_document(
        GOLDEN_MODEL_SHA256,
        GOLDEN_EXPORT_MANIFEST_SHA256,
        options={**ENGINE_OPTIONS, **options},  # type: ignore[arg-type]
    )


def limits_of(card: str) -> str:
    start = card.index("\n## Limits\n")
    return card[start : card.index("\n## Provenance\n")]


def test_every_run_on_the_cpu_says_that_no_gpu_run_is_recorded(
    record: ModelRecord,
) -> None:
    limits = limits_of(card_of(record, bound()))

    assert NO_GPU in limits
    assert "(none was made)" not in limits
    assert "any run on a GPU" not in limits
    assert GPU not in limits


def test_a_run_on_cuda_says_that_its_figures_come_from_a_gpu_and_are_not_reproduced(
    record: ModelRecord,
) -> None:
    limits = limits_of(card_of(record, bound(device="cuda")))

    assert GPU in limits
    assert NO_GPU not in limits
    assert "(none was made)" not in limits


def test_a_baseline_on_cuda_counts_though_the_carded_system_ran_on_the_cpu(
    record: ModelRecord,
) -> None:
    document = bound()
    for entry in document["systems"]:
        if entry["name"] == OTHER:
            entry["options"] = {"device": "cuda", "beam_size": 4}

    limits = limits_of(card_of(record, document))

    assert GPU in limits
    assert NO_GPU not in limits


def test_a_report_that_records_no_device_says_that_no_gpu_run_is_recorded(
    record: ModelRecord,
) -> None:
    document = bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, options={}
    )

    limits = limits_of(card_of(record, document))

    assert NO_GPU in limits
    assert GPU not in limits


def test_the_gpu_sentence_is_not_in_the_not_measured_list(record: ModelRecord) -> None:
    for document in (bound(), bound(device="cuda")):
        limits = limits_of(card_of(record, document))
        measured = next(
            line for line in limits.splitlines() if line.startswith("- Not measured:")
        )
        assert "GPU" not in measured
        assert measured.endswith("and real-world input beyond the suites above.")


def test_the_sentence_is_decided_by_the_report_alone(record: ModelRecord) -> None:
    document = bound(device="cuda")

    assert GPU in limits_of(card_of(record, document, system=SYSTEM))
    assert view_of(document).systems[0].options["device"] == "cuda"
