"""Session fixtures of the converter integration tests.

`tiny_run` trains the tiny converter once per test session, from the tiny reference
list, and predicts the training sources in both directions; `tiny_ct2` exports that
checkpoint to CTranslate2. Every test of the session that needs a trained or
exported model takes these two, so the run is paid for once. Everything is written
under pytest's temporary directories.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from rinchen.script_conv.char_model import (
    CONVERTER_TAGS,
    check_char_model,
    train_char_model,
)
from rinchen.script_conv.config import (
    dump_config,
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import CorpusFiles, build_corpus, read_pairs_tsv
from rinchen.script_conv.export import export_ct2
from rinchen.script_conv.train import predict, run_training
from tests.support.script_reference import FIXTURE_DIR

TO_SCRIPT_TAG, TO_CYRILLIC_TAG = CONVERTER_TAGS
DIRECTION_TAGS = {"cyrl2mong": TO_SCRIPT_TAG, "mong2cyrl": TO_CYRILLIC_TAG}

_VALID_FRACTION = 0.1
_SEED = 1234
_CHAR_VOCAB_SIZE = 200


def read_lines(path: Path) -> tuple[str, ...]:
    """The lines of a corpus or prediction file, split on LF only: other line
    separators and NNBSP are part of a word's line."""
    return tuple(path.read_text(encoding="utf-8").removesuffix("\n").split("\n"))


@dataclass(frozen=True)
class TinyRun:
    """One finished tiny training run.

    `predictions` maps each direction (`cyrl2mong`, `mong2cyrl`) to Eole's greedy
    output for the lines of its training source file, in file order.
    """

    corpus: CorpusFiles
    spm_model: Path
    config_path: Path
    run_dir: Path
    ckpt_dir: Path
    train_seconds: float
    predictions: Mapping[str, tuple[str, ...]]

    def lines(self, name: str) -> tuple[str, ...]:
        """The lines of the corpus file `name`, for example `cyrl2mong.train.tgt`."""
        return read_lines(self.corpus.files[name])


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> TinyRun:
    root = tmp_path_factory.mktemp("converter_tiny")
    run_dir = root / "run"
    pairs = read_pairs_tsv(FIXTURE_DIR / "tiny_words.tsv", origin="reference")
    corpus = build_corpus(
        pairs, root / "corpus", valid_fraction=_VALID_FRACTION, seed=_SEED
    )

    texts = [pair.cyrillic for pair in pairs] + [pair.script for pair in pairs]
    spm_model = train_char_model(
        texts, root / "converter_char", vocab_size=_CHAR_VOCAB_SIZE
    )
    check_char_model(spm_model, texts)

    cfg = render_converter_config(
        "tiny", data_dir=corpus.directory, spm_model=spm_model, run_dir=run_dir
    )
    validate_converter_config(cfg, profile="tiny")
    config_path = root / "converter_tiny.yaml"
    dump_config(cfg, config_path)

    started = time.perf_counter()
    ckpt_dir = run_training(config_path, run_dir)
    train_seconds = time.perf_counter() - started

    predictions = {
        direction: read_lines(
            predict(
                ckpt_dir,
                corpus.files[f"{direction}.train.src"],
                tag=tag,
                spm_model=spm_model,
                out_file=run_dir / f"{direction}.train.pred",
            )
        )
        for direction, tag in DIRECTION_TAGS.items()
    }
    return TinyRun(
        corpus=corpus,
        spm_model=spm_model,
        config_path=config_path,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        train_seconds=train_seconds,
        predictions=predictions,
    )


@pytest.fixture(scope="session")
def tiny_ct2(tiny_run: TinyRun, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The CTranslate2 model directory exported from `tiny_run`'s checkpoint."""
    return export_ct2(
        tiny_run.ckpt_dir, tmp_path_factory.mktemp("converter_ct2") / "model"
    )
