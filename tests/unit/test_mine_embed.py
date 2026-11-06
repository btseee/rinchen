"""The `Embedder` protocol and the sentence-transformers embedder behind it."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from rinchen.mine import embed as embed_module
from rinchen.mine.embed import Embedder, SentenceTransformerEmbedder
from tests.support.embedders import ConceptEmbedder, HashEmbedder


class _FakeModel:
    """Stands in for a SentenceTransformer and records every `encode` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, inputs: Sequence[str], **options: Any) -> NDArray[np.float64]:
        self.calls.append((list(inputs), options))
        # float64 on purpose: the embedder has to hand back float32.
        return np.arange(len(inputs) * 3, dtype=np.float64).reshape(len(inputs), 3)


class _LoaderSpy:
    """Replaces `embed._load_model` and records each (model name, device)."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.loads: list[tuple[str, str]] = []

    def __call__(self, model_name: str, device: str) -> _FakeModel:
        self.loads.append((model_name, device))
        return self.model


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> _LoaderSpy:
    spy = _LoaderSpy(_FakeModel())
    monkeypatch.setattr(embed_module, "_load_model", spy)
    return spy


# --- the protocol -------------------------------------------------------------------


@pytest.mark.parametrize(
    "embedder",
    [ConceptEmbedder({}), HashEmbedder(), SentenceTransformerEmbedder()],
    ids=["concept", "hash", "sentence_transformer"],
)
def test_every_embedder_satisfies_the_protocol(embedder: object) -> None:
    assert isinstance(embedder, Embedder)


def test_an_object_without_embed_does_not_satisfy_the_protocol() -> None:
    assert not isinstance(object(), Embedder)


# --- loading ------------------------------------------------------------------------


def test_construction_does_not_load_a_model(loader: _LoaderSpy) -> None:
    SentenceTransformerEmbedder("some/model", batch_size=4, device="cpu")

    assert loader.loads == []


def test_the_first_embed_loads_the_model_once_and_later_calls_reuse_it(
    loader: _LoaderSpy,
) -> None:
    embedder = SentenceTransformerEmbedder("some/model", device="cpu")

    embedder.embed(["a", "b"])
    embedder.embed(["c"])

    assert loader.loads == [("some/model", "cpu")]
    assert len(loader.model.calls) == 2


def test_the_default_model_is_labse(loader: _LoaderSpy) -> None:
    SentenceTransformerEmbedder(device="cpu").embed(["a"])

    assert loader.loads == [("sentence-transformers/LaBSE", "cpu")]


def test_embed_asks_for_normalised_numpy_batches_without_a_progress_bar(
    loader: _LoaderSpy,
) -> None:
    SentenceTransformerEmbedder(batch_size=7, device="cpu").embed(["x", "y", "z"])

    ((texts, options),) = loader.model.calls
    assert texts == ["x", "y", "z"]
    assert options == {
        "batch_size": 7,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_embed_returns_one_float32_row_per_text_in_order(loader: _LoaderSpy) -> None:
    result = SentenceTransformerEmbedder(device="cpu").embed(["x", "y"])

    assert result.dtype == np.float32
    assert result.shape == (2, 3)
    assert result.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_embed_of_no_texts_is_an_empty_array_and_loads_nothing(
    loader: _LoaderSpy,
) -> None:
    result = SentenceTransformerEmbedder(device="cpu").embed([])

    assert result.shape == (0, 0)
    assert result.dtype == np.float32
    assert loader.loads == []


# --- the device ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cuda_available", "expected"), [(False, "cpu"), (True, "cuda")]
)
def test_auto_device_follows_cuda_availability(
    loader: _LoaderSpy,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected: str,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    SentenceTransformerEmbedder(device="auto").embed(["a"])

    assert loader.loads == [("sentence-transformers/LaBSE", expected)]


def test_auto_is_the_default_device(
    loader: _LoaderSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    SentenceTransformerEmbedder().embed(["a"])

    assert loader.loads == [("sentence-transformers/LaBSE", "cuda")]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_an_explicit_device_is_used_whatever_cuda_says(
    loader: _LoaderSpy, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: device != "cuda")

    SentenceTransformerEmbedder(device=device).embed(["a"])

    assert loader.loads == [("sentence-transformers/LaBSE", device)]


# --- import cost --------------------------------------------------------------------


@pytest.mark.subprocess
def test_importing_the_embedder_stage_and_cli_does_not_import_torch() -> None:
    snippet = (
        "import sys\n"
        "import rinchen.mine.embed\n"
        "import rinchen.filter.labse\n"
        "import rinchen.cli.main\n"
        "heavy = [m for m in ('torch', 'sentence_transformers') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == ""


# --- the real model -----------------------------------------------------------------


@pytest.mark.network
def test_labse_embeds_translation_pair_closer_than_unrelated_pair() -> None:
    english = "Good morning"
    mongolian = "Өглөөний мэнд"
    unrelated = "Хадны оройд цас орж байна"
    embedder = SentenceTransformerEmbedder()

    vectors = embedder.embed([english, mongolian, unrelated])

    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    translation = float(vectors[0] @ vectors[1])
    other = float(vectors[0] @ vectors[2])
    assert translation > other, (translation, other)
