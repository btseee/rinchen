"""The Eole MetricX-QE adapter. Unit tests stand in for the Eole model and its
tokenizer but run Eole's own input and batching helpers; the network test runs
the real model on the GPU."""

from __future__ import annotations

import subprocess
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from rinchen.errors import UpstreamError
from rinchen.filter import qe
from rinchen.filter.qe import QEScorer
from rinchen.filter.qe_eole import EoleMetricXQEScorer

# Importing Eole warns about its missing C++ extension and flash_attn and about
# deprecations in Eole, pydantic and torch; none of it is ours to fix.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:Could not import:RuntimeWarning"),
    pytest.mark.filterwarnings(
        "ignore:.torch.jit.script. is not supported:FutureWarning"
    ),
]

QE_TEMPLATE_WORDS = 2  # "source:" and "candidate:" in the QE input template


class StandInVocab:
    def lookup_token(self, piece: str) -> int:
        return len(piece) + 10


class StandInTokenizer:
    def tokenize_string(self, text: str, side: str, is_train: bool) -> list[str]:
        return text.split()


class StandInModel:
    """Scores a pair by its number of words, template words included."""

    scoring_type = "metricx"
    input_templates: ClassVar[dict[str, str]] = {"qe": "source: {src} candidate: {tgt}"}
    vocabs: ClassVar[dict[str, StandInVocab]] = {"src": StandInVocab()}
    eos_id = -1
    strip_eos = True
    max_length = 512

    def __init__(self, failure: RuntimeError | None = None) -> None:
        self.failure = failure
        self.batch_sizes: list[int] = []

    def validate_input_mode(self, input_mode: str) -> None:
        assert input_mode == "qe"

    def predict_scores(self, rows: list[dict[str, list[int]]]) -> NDArray[np.float64]:
        self.batch_sizes.append(len(rows))
        if self.failure is not None:
            raise self.failure
        return np.array([float(len(row["input_ids"])) for row in rows])


class StandInEole:
    """Replaces what the adapter fetches from Eole: the model finder, the model
    loader and the tokenizer builder. Every model it hands out is remembered
    weakly, so a test can tell whether the adapter let go of it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loads: list[tuple[str, str]] = []
        self.models: list[weakref.ref[StandInModel]] = []
        self.failure: RuntimeError | None = None
        self.model_factory: Callable[[], StandInModel] = self._new_model
        self._patch(monkeypatch)

    def _new_model(self) -> StandInModel:
        return StandInModel(self.failure)

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import eole.models.hf_resolver
        import eole.models.model
        import eole.utils.scorer

        def resolve(model_name: str) -> str:
            return model_name

        def from_model_dir(model_dir: str, device: Any, compute_dtype: Any) -> Any:
            self.loads.append((model_dir, str(device)))
            model = self.model_factory()
            self.models.append(weakref.ref(model))
            return model

        monkeypatch.setattr(
            eole.models.hf_resolver, "resolve_preconverted_eole_hf_repo", resolve
        )
        monkeypatch.setattr(
            eole.models.model.EncoderDecoderScoringModel,
            "from_model_dir",
            from_model_dir,
        )
        monkeypatch.setattr(
            eole.utils.scorer,
            "build_scorer_sentencepiece_transform",
            lambda model, model_dir: StandInTokenizer(),
        )


@pytest.fixture
def stand_in_eole(monkeypatch: pytest.MonkeyPatch) -> StandInEole:
    return StandInEole(monkeypatch)


def _patch_resolver(
    monkeypatch: pytest.MonkeyPatch, resolve: Callable[[str], str | None]
) -> None:
    import eole.models.hf_resolver

    monkeypatch.setattr(
        eole.models.hf_resolver, "resolve_preconverted_eole_hf_repo", resolve
    )


# --- what the adapter is ------------------------------------------------------------


def test_the_eole_scorer_needs_cuda_and_is_a_scorer() -> None:
    scorer: QEScorer = EoleMetricXQEScorer()

    assert scorer.requires_cuda is True


def test_a_compute_dtype_that_eole_does_not_offer_is_a_value_error() -> None:
    # A constructor argument, not a config field, so ValueError rather than
    # ConfigError.
    with pytest.raises(
        ValueError, match="compute_dtype must be one of fp32, bf16, fp16, got 'int4'"
    ):
        EoleMetricXQEScorer(compute_dtype="int4")  # type: ignore[arg-type]


def test_the_eole_scorer_scores_no_pairs_without_loading_a_model(
    tmp_path: Path,
) -> None:
    scorer = EoleMetricXQEScorer(str(tmp_path / "not-a-model"))

    assert scorer.score([], []) == []


# --- finding and loading the model --------------------------------------------------


def test_a_model_that_is_neither_a_directory_nor_a_repo_is_an_upstream_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = str(tmp_path / "not-a-model")
    _patch_resolver(monkeypatch, lambda model_name: None)

    with pytest.raises(UpstreamError) as raised:
        EoleMetricXQEScorer(name).score(["hello"], ["world"])

    assert str(raised.value) == (
        f"{name!r} is neither a directory nor a pre-converted Eole repo on Hugging Face"
    )


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, ValueError])
def test_a_failure_while_finding_the_model_is_wrapped_and_chained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_type: type[Exception]
) -> None:
    original = failure_type("download went wrong")

    def resolve(model_name: str) -> str | None:
        raise original

    _patch_resolver(monkeypatch, resolve)
    name = str(tmp_path / "some-repo")

    with pytest.raises(UpstreamError) as raised:
        EoleMetricXQEScorer(name).score(["hello"], ["world"])

    assert raised.value.__cause__ is original
    assert str(raised.value) == (
        f"could not load MetricX QE model {name!r}: download went wrong"
    )


def test_a_model_that_is_not_a_metricx_scorer_is_an_upstream_error(
    stand_in_eole: StandInEole,
) -> None:
    class NotMetricX(StandInModel):
        scoring_type = "comet"

    stand_in_eole.model_factory = NotMetricX

    with pytest.raises(UpstreamError, match="is not a MetricX scoring model"):
        EoleMetricXQEScorer("some/repo").score(["hello"], ["world"])


def test_a_local_model_directory_is_used_as_it_is(
    stand_in_eole: StandInEole, tmp_path: Path
) -> None:
    EoleMetricXQEScorer(str(tmp_path)).score(["hello"], ["world"])

    assert stand_in_eole.loads == [(str(tmp_path), "cuda")]


# --- scoring ------------------------------------------------------------------------


def test_each_pair_gets_its_own_score_in_order_and_the_model_loads_once(
    stand_in_eole: StandInEole,
) -> None:
    scorer = EoleMetricXQEScorer("some/repo")

    first = scorer.score(["a b", "a"], ["c", "c d e"])
    second = scorer.score(["x"], ["y"])

    assert first == [QE_TEMPLATE_WORDS + 3, QE_TEMPLATE_WORDS + 4]
    assert second == [QE_TEMPLATE_WORDS + 2]
    assert all(isinstance(score, float) for score in first + second)
    assert len(stand_in_eole.loads) == 1


@pytest.mark.parametrize(
    ("message", "batch_sizes"),
    [
        ("CUDA error: device-side assert", [2]),
        # Eole halves the batch on an out-of-memory error before giving up.
        ("CUDA out of memory", [2, 1]),
    ],
)
def test_a_runtime_error_while_scoring_is_wrapped_and_chained(
    stand_in_eole: StandInEole, message: str, batch_sizes: list[int]
) -> None:
    original = RuntimeError(message)
    stand_in_eole.failure = original
    scorer = EoleMetricXQEScorer("some/repo")

    with pytest.raises(UpstreamError, match="2 pairs") as raised:
        scorer.score(["a", "b"], ["c", "d"])

    assert raised.value.__cause__ is original
    assert message in str(raised.value)
    model = stand_in_eole.models[0]()
    assert model is not None
    assert model.batch_sizes == batch_sizes


# --- giving the memory back ---------------------------------------------------------


def test_release_before_the_model_is_loaded_does_nothing() -> None:
    scorer = EoleMetricXQEScorer()

    scorer.release()
    scorer.release()

    assert scorer.score([], []) == []


def test_release_drops_the_model_and_a_later_score_loads_it_again(
    stand_in_eole: StandInEole, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    emptied: list[None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(None))
    scorer = EoleMetricXQEScorer("some/repo")
    scorer.score(["a"], ["b"])
    assert [ref() is not None for ref in stand_in_eole.models] == [True]

    scorer.release()
    scorer.release()

    assert [ref() is None for ref in stand_in_eole.models] == [True]
    assert emptied
    assert scorer.score(["a"], ["b"]) == [QE_TEMPLATE_WORDS + 2]
    assert len(stand_in_eole.loads) == 2


# --- import cost --------------------------------------------------------------------


@pytest.mark.subprocess
def test_importing_the_qe_modules_does_not_import_torch_or_eole() -> None:
    snippet = (
        "import sys\n"
        "import rinchen.filter.qe\n"
        "import rinchen.filter.qe_eole\n"
        "heavy = [m for m in ('torch', 'eole', 'transformers')"
        " if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


# --- the real model -----------------------------------------------------------------


@pytest.mark.network
def test_metricx_qe_scores_a_translation_lower_than_a_mismatched_pair() -> None:
    if not qe.cuda_is_available():
        pytest.skip("MetricX-QE through Eole needs a CUDA device")
    sources = [
        "The weather is nice today.",
        "The weather is nice today.",
        "I bought three apples at the market.",
        "I bought three apples at the market.",
    ]
    translations = [
        "Өнөөдөр цаг агаар сайхан байна.",
        "Хадны оройд цас орж байна.",
        "Би зах дээрээс гурван алим худалдаж авлаа.",
        "Манай компани шинэ оффис нээлээ.",
    ]
    scorer = EoleMetricXQEScorer()

    try:
        scores = scorer.score(sources, translations)
    finally:
        scorer.release()

    assert len(scores) == 4
    assert all(0.0 <= score <= 25.0 for score in scores)
    assert scores[0] < scores[1]
    assert scores[2] < scores[3]
