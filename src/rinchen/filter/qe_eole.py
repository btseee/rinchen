"""MetricX-24 quality estimation through Eole's scorer model, as a `QEScorer`
for the `qe` stage. Inference only. Eole and torch are imported on first use,
so importing this module stays cheap."""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from rinchen.errors import UpstreamError

EOLE_METRICX_QE_MODEL = "eole-nlp/metricx-24-hybrid-large-v2p6-eole"

ComputeDtype = Literal["fp32", "bf16", "fp16"]
_COMPUTE_DTYPES = get_args(ComputeDtype)


@dataclass(frozen=True)
class _LoadedMetricX:
    model: Any
    tokenizer: Any


class EoleMetricXQEScorer:
    """MetricX-24 hybrid in quality-estimation mode (source and translation, no
    reference), run by Eole's scorer model on the GPU. Scores run from 0 to 25
    and lower is better.

    Eole's own scorer class returns only the corpus mean and reloads the model
    on every call, so this drives Eole-internal, undocumented helpers instead:
    `resolve_preconverted_eole_hf_repo`, `EncoderDecoderScoringModel.from_model_dir`,
    `build_scorer_sentencepiece_transform`, `build_template_inputs`,
    `encode_scorer_inputs` and `predict_scores_with_oom_retry`. `eole==0.6.4` is
    pinned for that reason, and the network test is the only guard against an
    upgrade that changes them.

    `model` is a pre-converted Eole MetricX directory or Hugging Face repo; a
    repo is downloaded on first use to `~/.cache/eole/hf/`, about 6 GB. The
    model loads on the first `score` and stays on the GPU until `release`.
    `compute_dtype` defaults to `fp32`, Eole's own default; a value Eole does
    not offer raises ValueError, because it is a constructor argument and not
    a config field."""

    requires_cuda = True

    def __init__(
        self,
        model: str = EOLE_METRICX_QE_MODEL,
        *,
        compute_dtype: ComputeDtype = "fp32",
    ) -> None:
        if compute_dtype not in _COMPUTE_DTYPES:
            raise ValueError(
                f"compute_dtype must be one of {', '.join(_COMPUTE_DTYPES)}, "
                f"got {compute_dtype!r}"
            )
        self._model_name = model
        self._compute_dtype = compute_dtype
        self._loaded: _LoadedMetricX | None = None

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        """Per-pair scores, in order. The pairs of one call are scored as one
        batch, which Eole halves, down to one pair, when the GPU runs out of
        memory. Raises UpstreamError when the model cannot be loaded or
        scoring fails."""
        if not sources:
            return []
        from eole.utils.scorer import (
            build_template_inputs,
            encode_scorer_inputs,
            predict_scores_with_oom_retry,
        )

        loaded = self._loaded_model()
        texts = build_template_inputs(
            list(sources),
            list(translations),
            None,
            loaded.model.input_templates,
            input_mode="qe",
        )
        input_ids = encode_scorer_inputs(texts, loaded.tokenizer, loaded.model)
        rows = [{"input_ids": ids} for ids in input_ids]
        try:
            scores: list[float] = predict_scores_with_oom_retry(
                loaded.model, rows, len(rows)
            )
        except RuntimeError as exc:
            raise UpstreamError(
                f"MetricX QE scoring failed for a call of {len(rows)} pairs: {exc}"
            ) from exc
        return scores

    def release(self) -> None:
        """Drop the loaded model and give its GPU memory back. Safe to call
        before the first `score` and more than once; a later `score` loads the
        model again."""
        self._loaded = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.cuda.empty_cache()

    def _loaded_model(self) -> _LoadedMetricX:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _load(self) -> _LoadedMetricX:
        """Raises UpstreamError when the model cannot be found, downloaded or
        read, is not a MetricX QE model, or CUDA is not usable."""
        import torch
        from eole.constants import TORCH_DTYPES
        from eole.models.hf_resolver import resolve_preconverted_eole_hf_repo
        from eole.models.model import EncoderDecoderScoringModel
        from eole.utils.scorer import build_scorer_sentencepiece_transform

        try:
            model_dir = (
                self._model_name
                if os.path.isdir(self._model_name)
                else resolve_preconverted_eole_hf_repo(self._model_name)
            )
            if model_dir is None:
                raise UpstreamError(
                    f"{self._model_name!r} is neither a directory nor a "
                    "pre-converted Eole repo on Hugging Face"
                )
            model = EncoderDecoderScoringModel.from_model_dir(
                model_dir,
                device=torch.device("cuda"),
                compute_dtype=TORCH_DTYPES[self._compute_dtype],
            )
            if getattr(model, "scoring_type", None) != "metricx":
                raise UpstreamError(
                    f"{self._model_name!r} is not a MetricX scoring model"
                )
            model.validate_input_mode("qe")
            tokenizer = build_scorer_sentencepiece_transform(model, model_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise UpstreamError(
                f"could not load MetricX QE model {self._model_name!r}: {exc}"
            ) from exc
        return _LoadedMetricX(model=model, tokenizer=tokenizer)
