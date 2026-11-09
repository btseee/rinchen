"""Rinchen: low-resource Mongolian machine translation.

`rinchen.Translator` is the Python way to translate with a model of ours (D11). It is
resolved on first use (PEP 562): `import rinchen` stays cheap and loads neither the engine nor
CTranslate2 nor SentencePiece; those are loaded when a model is opened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rinchen.infer.translator import Translator

__all__ = ["Translator"]


def __getattr__(name: str) -> Any:
    if name == "Translator":
        from rinchen.infer.translator import Translator

        return Translator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
