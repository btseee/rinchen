"""The `Translator` protocol every generator backend implements: the clients
and the synthetic-data runner call it by these parameter names."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import get_type_hints

from rinchen.synth.translator import Translator


def test_translate_takes_texts_and_both_languages():
    parameters = inspect.signature(Translator.translate).parameters

    assert list(parameters) == ["self", "texts", "src_lang", "tgt_lang"]


def test_translate_parameters_are_positional_or_keyword_without_defaults():
    parameters = inspect.signature(Translator.translate).parameters

    for name in ("texts", "src_lang", "tgt_lang"):
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[name].default is inspect.Parameter.empty


def test_translate_maps_a_sequence_of_texts_to_one_candidate_list_per_text():
    hints = get_type_hints(Translator.translate)

    assert hints == {
        "texts": Sequence[str],
        "src_lang": str,
        "tgt_lang": str,
        "return": list[list[str]],
    }
