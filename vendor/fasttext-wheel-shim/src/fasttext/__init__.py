"""Compatibility shim for eole's hard dependency on the PyPI distribution
"fasttext-wheel". Rinchen does not build the original facebookresearch/fastText
C++ package from source: it has no cp314 wheel, and building it from source hits
several pre-existing C++ portability bugs (MSVC-only `ssize_t`, gcc14 missing
`<cstdint>`, and a NumPy-1-only `copy=False` call in its Python wrapper) --
see docs/superpowers/specs/2026-10-29-fasttext-wheel-shim-design.md.

eole uses `fasttext` in exactly one place (`eole/transforms/clean.py`'s lazy,
optional `clean` transform), only for `fasttext.load_model(...)` to run
language identification via the `lid.176` model. `fasttext-new` (already a
direct Rinchen dependency) is a maintained, cp314-native, NumPy-2-compatible
Rust/PyO3 reimplementation verified bit-identical to the original on the
lid.176 models. This package re-exports it under the `fasttext` name so
`import fasttext` resolves correctly wherever eole (or anything else) expects
the real package -- no native compilation involved.
"""

from fasttext_new import (
    FastText,
    load_model,
    tokenize,
    train_supervised,
    train_unsupervised,
)

__version__ = "0.9.2"

__all__ = [
    "FastText",
    "__version__",
    "load_model",
    "tokenize",
    "train_supervised",
    "train_unsupervised",
]
