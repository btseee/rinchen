"""The interface every generator backend implements: the LLM clients and the
CTranslate2 baseline. A synthetic-data job only sees this protocol, so it does
not know which kind of generator produced a candidate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Translator(Protocol):
    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        """Translate every text from `src_lang` to `tgt_lang` (registry codes).

        Returns one list of candidates per text, in input order, best first
        when the backend ranks them. A candidate is a plain string. A list may
        hold fewer candidates than the backend was asked for, or none at all (a
        refusal or an empty generation), but never more."""
        ...
