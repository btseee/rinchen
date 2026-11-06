"""Train a Rinchen SentencePiece model from a directory of per-language text.

`train_spm` samples the corpus with `sample_balanced`, writes the sample next to
the model, and runs the SentencePiece trainer on it with the project's policy
invariants.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece

from rinchen.config.spm import SpmConfig
from rinchen.errors import UpstreamError
from rinchen.spm.corpus import discover_corpus, read_lines, sample_balanced
from rinchen.spm.symbols import user_defined_symbols

# Project invariants, not config fields. A test pins the options passed to the
# trainer; `check_spm` verifies that a trained model keeps its symbols atomic and
# has byte fallback on.
MODEL_TYPE = "unigram"
BYTE_FALLBACK = True
SPLIT_DIGITS = True
NORMALIZATION_RULE_NAME = "identity"

# 2 keeps the trainer's ERROR and FATAL messages and drops its INFO/WARNING flood.
_TRAINER_MIN_LOG_LEVEL = 2


def train_spm(cfg: SpmConfig) -> Path:
    """Train the SentencePiece model described by `cfg` and return its path,
    `<output_prefix>.model`. The balanced sample the model was trained on is
    kept as `<output_prefix>.sampled.txt`, and the trainer writes
    `<output_prefix>.vocab` beside the model.

    Raises:
        ConfigError: the corpus directory is missing or holds a file that is not
            named after a registry language code.
        DataError: the corpus has no usable text.
        UpstreamError: SentencePiece rejected the training run, for example
            because `vocab_size` exceeds what the corpus can support; the
            trainer's `RuntimeError` is chained as the cause.
    """
    corpus = discover_corpus(cfg.corpus_dir)
    lines_by_language = {
        language: read_lines(path) for language, path in corpus.items()
    }
    sample = sample_balanced(
        lines_by_language,
        input_sentence_size=cfg.input_sentence_size,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )
    sampled_path = _write_sample(sample, cfg.output_prefix)
    try:
        sentencepiece.SentencePieceTrainer.train(**_trainer_options(cfg, sampled_path))
    except RuntimeError as exc:
        raise UpstreamError(
            f"sentencepiece training failed for {cfg.output_prefix}: {exc}"
        ) from exc
    return Path(f"{cfg.output_prefix}.model")


def _write_sample(sample: list[str], output_prefix: Path) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sampled_path = Path(f"{output_prefix}.sampled.txt")
    with sampled_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in sample:
            handle.write(f"{line}\n")
    return sampled_path


def _trainer_options(cfg: SpmConfig, sampled_path: Path) -> dict[str, object]:
    return {
        "input": str(sampled_path),
        "model_prefix": str(cfg.output_prefix),
        "vocab_size": cfg.vocab_size,
        "character_coverage": cfg.character_coverage,
        "input_sentence_size": cfg.input_sentence_size,
        "model_type": MODEL_TYPE,
        "byte_fallback": BYTE_FALLBACK,
        "split_digits": SPLIT_DIGITS,
        "normalization_rule_name": NORMALIZATION_RULE_NAME,
        "user_defined_symbols": user_defined_symbols(),
        "shuffle_input_sentence": True,
        "minloglevel": _TRAINER_MIN_LOG_LEVEL,
    }
