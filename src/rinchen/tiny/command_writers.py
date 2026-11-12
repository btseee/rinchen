"""The writers of the files the CLI stages of the tiny pipeline's commands read (D15, spec P2, P7 and P17):
the SentencePiece config, the config of the converter's character model, the ablation matrix of the one
variant and the synthetic-data config.

Each is the file its command's own loader reads, and the tests load it through that loader. A writer
refuses a path that is not absolute (the loaders anchor a relative one at the checkout, which would put a
model or a corpus there) and a server that is not the loopback listener of the pipeline (it opens a
connection to nothing else). A file appears whole or not at all.

`rinchen.tiny.commands` re-exports every writer, which is where the stages and the tests take them from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from rinchen.config.tiny import SpmSection, TrainSection
from rinchen.tiny.fakes import LOOPBACK
from rinchen.utils.part_file import replace_file

# The back-translation job: Khalkha text translated into English by the oracle.
MONO_LANGUAGE: Final = "khk_Cyrl"
OTHER_LANGUAGE: Final = "eng_Latn"
_MATRIX_NAME: Final = "tiny"
_GENERATOR: Final = "tiny_oracle"
_GENERATOR_MODEL: Final = "tiny-oracle"
_SERVER_URL: Final = re.compile(rf"http://{re.escape(LOOPBACK)}:([1-9][0-9]{{0,4}})/v1")
_MAX_PORT: Final = 65535


def _require_absolute(path: Path, what: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{what} must be an absolute path")


def _write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    replace_file(path, yaml.safe_dump(dict(document), sort_keys=False).encode("utf-8"))


def write_spm_config(
    path: Path, section: SpmSection, seed: int, corpus_dir: Path, output_prefix: Path
) -> None:
    """The config `rinchen spm train` reads: the knobs of the profile, the seed and where the text
    is and the model goes.

    Raises:
        ValueError: a directory or prefix is not an absolute path.
    """
    _require_absolute(corpus_dir, "the corpus directory")
    _require_absolute(output_prefix, "the output prefix")
    _write_yaml(
        path,
        {
            "vocab_size": section.vocab_size,
            "input_sentence_size": section.input_sentence_size,
            "character_coverage": section.character_coverage,
            "temperature": section.temperature,
            "seed": seed,
            "corpus_dir": str(corpus_dir),
            "output_prefix": str(output_prefix),
        },
    )


def write_char_config(path: Path, vocab_size: int) -> None:
    """The config of the converter's character model, which holds the size of its vocabulary.

    Raises:
        ValueError: `vocab_size` is not positive.
    """
    if vocab_size < 1:
        raise ValueError("the character vocabulary must be at least 1")
    _write_yaml(path, {"vocab_size": vocab_size})


def write_matrix(
    path: Path, config: TrainSection, base_config: Path, seed: int
) -> None:
    """The ablation matrix of the one variant the pipeline trains: the mixing of the profile, one seed,
    and the steps of the profile as overrides of `base_config`, which `train ablate` merges over the
    base file (the save interval and the number of checkpoints kept stay the base's).

    Raises:
        ValueError: `base_config` is not an absolute path.
    """
    _require_absolute(base_config, "the base config")
    steps = config.steps
    _write_yaml(
        path,
        {
            "name": _MATRIX_NAME,
            "base": str(base_config),
            "axes": {
                "tag_placement": [config.tag_placement],
                "temperature": [float(config.temperature)],
                "bt_ratio": [float(config.bt_ratio)],
            },
            "seeds": [seed],
            "dev_max_lines": config.dev_max_lines,
            "overrides": {"training": {"train_steps": steps, "valid_steps": steps}},
        },
    )


def write_synth_config(
    path: Path, job: str, base_url: str, shard_size: int, n: int
) -> None:
    """The config `rinchen synth run` reads: one `vllm` generator at `base_url`, greedy, with no
    retry, and one back-translation job of Khalkha text into English.

    Raises:
        ValueError: `base_url` is not `http://127.0.0.1:<port>/v1`.
    """
    found = _SERVER_URL.fullmatch(base_url)
    if found is None or int(found.group(1)) > _MAX_PORT:
        raise ValueError(
            "the server of the synthetic config must be on the loopback address"
        )
    _write_yaml(
        path,
        {
            "generators": {
                _GENERATOR: {
                    "kind": "vllm",
                    "base_url": base_url,
                    "model": _GENERATOR_MODEL,
                    "n": n,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "concurrency": 4,
                    "timeout_s": 20,
                    "retries": 0,
                }
            },
            "jobs": {
                job: {
                    "kind": "bt",
                    "mono_lang": MONO_LANGUAGE,
                    "other_lang": OTHER_LANGUAGE,
                    "generator": _GENERATOR,
                    "shard_size": shard_size,
                }
            },
        },
    )
