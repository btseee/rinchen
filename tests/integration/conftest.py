"""Session fixtures of the converter integration tests.

`tiny_run` trains the tiny converter once per test session, from the tiny reference
list, and predicts the training sources in both directions; `tiny_ct2` exports that
checkpoint to CTranslate2. Every test of the session that needs a trained or
exported model takes these two, so the run is paid for once. Everything is written
under pytest's temporary directories. Each Eole or CTranslate2 command is bounded
by `_COMMAND_TIMEOUT_SECONDS`, so a hung subprocess fails the session instead of
blocking it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rinchen.script_conv.char_model import check_char_model, train_char_model
from rinchen.script_conv.config import (
    dump_config,
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import build_corpus, read_pairs_tsv
from rinchen.script_conv.export import export_ct2
from rinchen.script_conv.train import predict, run_training
from tests.support.script_reference import FIXTURE_DIR
from tests.support.tiny_converter import DIRECTION_TAGS, TinyRun, read_lines

_VALID_FRACTION = 0.1
_SEED = 1234
_CHAR_VOCAB_SIZE = 200
_COMMAND_TIMEOUT_SECONDS = 600


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
    ckpt_dir = run_training(config_path, run_dir, timeout=_COMMAND_TIMEOUT_SECONDS)
    train_seconds = time.perf_counter() - started

    predictions = {
        direction: read_lines(
            predict(
                ckpt_dir,
                corpus.files[f"{direction}.train.src"],
                tag=tag,
                spm_model=spm_model,
                out_file=run_dir / f"{direction}.train.pred",
                timeout=_COMMAND_TIMEOUT_SECONDS,
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
        tiny_run.ckpt_dir,
        tmp_path_factory.mktemp("converter_ct2") / "model",
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
