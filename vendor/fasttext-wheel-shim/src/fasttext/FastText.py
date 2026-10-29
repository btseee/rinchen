"""Re-export of fasttext_new's FastText submodule under the fasttext.FastText
name, matching the real fasttext-wheel package's internal module layout, for
any code that imports this submodule path directly rather than the package's
top level."""

from fasttext_new.FastText import (
    _FastText,
    cbow,
    load_model,
    skipgram,
    supervised,
    tokenize,
    train_supervised,
    train_unsupervised,
    unsupervised_default,
)

__all__ = [
    "_FastText",
    "cbow",
    "load_model",
    "skipgram",
    "supervised",
    "tokenize",
    "train_supervised",
    "train_unsupervised",
    "unsupervised_default",
]
