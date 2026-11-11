"""The document of the tiny pipeline's loader tests (D15): a literal of the interface, with digests
that are not the fixtures' own, and the helpers that edit it and read what the loader refuses.

The loader tests (`tests/unit/test_config_tiny.py` and `test_config_tiny_values.py`) edit this
document one field at a time, so that the refusal of a field is a test of that field alone and does
not move when a prior of the shipped profile does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest
import yaml

from rinchen.config.tiny import load_tiny_config
from rinchen.errors import ConfigError
from rinchen.tiny.fixtures import FIXTURE_FILES

SENTINEL: Final = "ZZ-planted-value"
TIMEOUTS: Final = {
    "environment": 30,
    "fetch": 30,
    "corpus": 60,
    "filter": 120,
    "spm": 120,
    "converter": 300,
    "synth": 120,
    "prepare": 120,
    "train": 600,
    "average": 60,
    "export": 120,
    "parity": 180,
    "serve": 60,
    "robustness": 120,
    "evaluate": 300,
}
SECTIONS: Final = (
    "fixtures",
    "spm",
    "filter",
    "converter",
    "synth",
    "train",
    "export",
    "parity",
    "decode",
    "serve",
    "evaluation",
    "thresholds",
)


def document() -> dict[str, Any]:
    """The document of the interface, with digests that are not the fixtures' own."""
    return {
        "fixtures": {
            "directory": "tests/fixtures/parallel",
            "sha256": {
                name: f"{index:x}" * 64 for index, name in enumerate(FIXTURE_FILES, 1)
            },
        },
        "registry": "configs/data/sources.tiny_parallel.yaml",
        "spm": {
            "vocab_size": 640,
            "input_sentence_size": 100000,
            "character_coverage": 0.9999,
            "temperature": 0.3,
        },
        "filter": {
            "char_ratio_min": 0.4,
            "char_ratio_max": 2.5,
            "minhash_threshold": 0.8,
            "labse_min_cosine": 0.7,
            "qe_max_metricx": 8.0,
            "max_authentic_flagged_share": 0.01,
        },
        "converter": {
            "pairs": "tests/fixtures/script_conv/tiny_words.tsv",
            "char_vocab_size": 200,
        },
        "synth": {"job": "bt_eng_khk", "shard_size": 32, "n": 1},
        "train": {
            "base": "configs/eole/tiny.yaml",
            "steps": 1500,
            "average_last": 2,
            "temperature": 1.0,
            "bt_ratio": 0.2,
            "tag_placement": "src",
            "dev_max_lines": 100,
        },
        "export": {"quantizations": ["float32", "int8"]},
        "parity": {"max_lines": 50},
        "decode": {"beam_size": 1},
        "serve": {"config": "configs/serve/tiny.yaml"},
        "evaluation": {
            "memorised_split": "test",
            "heldout_split": "devtest",
            "resamples": 1000,
        },
        "thresholds": {
            "train_chrf_min": 50.0,
            "parity_exact_min": 0.95,
            "parity_delta_max": 0.001,
            "int8_chrf_drop_max": 1.0,
            "script_purity_min": 1.0,
            "placeholder_acc_min": 1.0,
        },
        "budget_seconds": 1200,
        "timeouts": dict(TIMEOUTS),
        "threads": 2,
        "qe_fake": True,
        "created": "2026-11-11T00:00:00Z",
    }


def dump(doc: Any) -> bytes:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True).encode("utf-8")


def with_value(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    doc = document()
    node = doc
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    return doc


def without_key(path: tuple[str, ...]) -> dict[str, Any]:
    doc = document()
    node = doc
    for part in path[:-1]:
        node = node[part]
    del node[path[-1]]
    return doc


def key_paths(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """The path of every key of every mapping of the document (lists are not entered)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield (*prefix, key)
            yield from key_paths(value, (*prefix, key))


def refusal(data: bytes | dict[str, Any]) -> str:
    with pytest.raises(ConfigError) as caught:
        load_tiny_config(data if isinstance(data, bytes) else dump(data))
    return str(caught.value)


def mapping_paths() -> list[tuple[str, ...]]:
    """The empty path (the top level) and the path of every mapping."""
    return [(), *[p for p in key_paths(document()) if isinstance(_at(p), dict)]]


def _at(path: tuple[str, ...]) -> Any:
    node: Any = document()
    for part in path:
        node = node[part]
    return node
