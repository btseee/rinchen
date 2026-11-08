"""The multilingual training config the D7 tests start from.

It is the spike's minimal working config (Eole 0.6.4 trained it on CPU): one
authentic and one back-translated corpus of a direction, a pre-tokenised `valid`, a
64-wide model with 2+2 layers and 4 heads, and the training keys of the tiny base.
Tests copy it, break one thing and read the verdict.
"""

from __future__ import annotations

from typing import Any

CORPUS = "en-mn.auth"  # a name with a dot, as the mixture builds them
BT_CORPUS = "en-mn.bt"


def make_config() -> dict[str, Any]:
    """The spike's minimal working multilingual config, which must validate."""
    pieces = ["sentencepiece", "prefix", "filtertoolong"]
    return {
        "seed": 1234,
        "src_vocab": "run/vocab.src",
        "share_vocab": True,
        "overwrite": True,
        "data": {
            CORPUS: {
                "path_src": "data/en-mn.auth.src",
                "path_tgt": "data/en-mn.auth.tgt",
                "transforms": list(pieces),
                "weight": 3,
                "src_prefix": "<2khk_Cyrl>",
                "tgt_prefix": "",
            },
            BT_CORPUS: {
                "path_src": "data/en-mn.bt.src",
                "path_tgt": "data/en-mn.bt.tgt",
                "transforms": list(pieces),
                "weight": 1,
                "src_prefix": "<2khk_Cyrl> <bt>",
                "tgt_prefix": "",
            },
            "valid": {
                "path_src": "variants/t1/valid.src",
                "path_tgt": "variants/t1/valid.tgt",
                "transforms": [],
            },
        },
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": "spm/toy.model",
                "tgt_subword_model": "spm/toy.model",
            },
            "filtertoolong": {"src_seq_length": 64, "tgt_seq_length": 64},
        },
        "model": {
            "architecture": "transformer",
            "hidden_size": 64,
            "encoder": {"layers": 2},
            "decoder": {"layers": 2},
            "heads": 4,
            "transformer_ff": 128,
            "embeddings": {
                "word_vec_size": 64,
                "position_encoding_type": "SinusoidalInterleaved",
            },
            "share_embeddings": True,
            "share_decoder_embeddings": True,
        },
        "training": {
            "model_path": "run/ckpt",
            "batch_type": "tokens",
            "batch_size": 2048,
            "accum_count": [1],
            "dropout": [0.0],
            "attention_dropout": [0.0],
            "optim": "adamw",
            "decay_method": "noam",
            "label_smoothing": 0.1,
            "world_size": 1,
            "train_steps": 1000,
            "valid_steps": 500,
            "save_checkpoint_steps": 500,
            "keep_checkpoint": 2,
            "learning_rate": 0.5,
            "warmup_steps": 100,
            "compute_dtype": "fp32",
            "gpu_ranks": [],
            "bucket_size": 1024,
            "prefetch_factor": 8,
            "num_workers": 1,
        },
    }
