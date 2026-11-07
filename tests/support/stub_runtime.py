"""Stand-ins for the two libraries `ScriptConverter` drives, for tests that need no model.

`StubProcessor` splits a word into one piece per character after a dummy prefix and
decodes the way SentencePiece does (the dummy prefix becomes a space, and a space at
the very start is dropped). `StubTranslator` answers each source line, a tag and then
the pieces of one word, with whatever its `answer` function returns for that word,
and records every call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DUMMY_PREFIX = chr(0x2581)


class StubProcessor:
    def encode(self, text: str, out_type: type = str) -> list[str]:
        return [DUMMY_PREFIX, *text]

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens).replace(DUMMY_PREFIX, " ").removeprefix(" ")


class StubTranslation:
    def __init__(self, pieces: list[str]) -> None:
        self.hypotheses = [pieces]


def source_word(tokens: list[str]) -> str:
    """The word a source line stands for: its pieces without the tag and the dummy prefix."""
    return "".join(tokens[2:])


class StubTranslator:
    def __init__(self, answer: Callable[[str], str]) -> None:
        self.answer = answer
        self.calls: list[tuple[list[list[str]], dict[str, Any]]] = []

    def translate_batch(
        self, source: list[list[str]], **kwargs: Any
    ) -> list[StubTranslation]:
        self.calls.append((source, kwargs))
        return [
            StubTranslation([DUMMY_PREFIX, *self.answer(source_word(tokens))])
            for tokens in source
        ]

    @property
    def words_sent(self) -> list[str]:
        """Every word of every call, in order."""
        return [source_word(tokens) for source, _ in self.calls for tokens in source]
