"""Sentence embedders for filtering and mining. `Embedder` is the protocol the
`labse` filter and the sentence aligner depend on; `SentenceTransformerEmbedder`
is the LaBSE implementation. Models are used for inference only, and torch and
sentence-transformers are imported on the first `embed`, so importing this
module stays cheap."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One L2-normalised float32 row per text, in order: shape `(n, d)`."""
        ...


class SentenceTransformerEmbedder:
    """Embeds with a sentence-transformers model, LaBSE by default.

    The model loads on the first `embed` and is reused afterwards. `device`
    is `"auto"` (CUDA when torch sees one, else CPU), `"cpu"` or `"cuda"`."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        *,
        batch_size: int = 128,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Shape `(n, d)`; with no texts, shape `(0, 0)` because `d` is only
        known once the model is loaded, and the model is not loaded for that."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._loaded_model().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _loaded_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self._model_name, _resolve_device(self._device))
        return self._model


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)
