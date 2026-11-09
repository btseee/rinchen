"""A scripted CTranslate2 translator and a SentencePiece stand-in for the tests of the teacher.

`ScriptedCt2` has the call shape of `ctranslate2.Translator.translate_batch` and answers
each source line from a script (a dict from the source tokens to the hypotheses, or a
function of them): a hypothesis is `(tokens, score)` and the result of a source is an object
with `.hypotheses` and `.scores`, as the library's `TranslationResult` has them (the scores
only when the call asked for `return_scores`). It records every call and starts no
process and loads no model: nothing here imports `ctranslate2`.

`FakeProcessor` cuts a text into one piece per whitespace-separated word, each with the
dummy prefix of SentencePiece (U+2581), and `decode` is the inverse. It also answers the
questions `require_one_piece` asks of a processor, over the pieces it was given: that is
how a test makes a direction tag one piece of the model or not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SPIECE = chr(0x2581)
_UNKNOWN_ID = 0

Hypothesis = tuple[Sequence[str], float]
Script = (
    Mapping[tuple[str, ...], Sequence[Hypothesis]]
    | Callable[[tuple[str, ...]], Sequence[Hypothesis]]
)


@dataclass(frozen=True)
class Ct2Result:
    """The two attributes of a `ctranslate2.TranslationResult` the teacher reads."""

    hypotheses: list[list[str]]
    scores: list[float]


@dataclass(frozen=True)
class Ct2Call:
    """One call of `translate_batch`: the source lines, the forced prefixes (None when
    none were passed) and every other keyword."""

    source: list[list[str]]
    target_prefix: list[list[str]] | None
    options: dict[str, Any]


@dataclass
class ScriptedCt2:
    """Answers each source line of `translate_batch` with its script entry.

    A source the script does not hold is a bug of the test and raises `AssertionError`.
    `error` is raised by every call instead of answering (after the call is recorded).
    `result_offset` changes the number of results: positive appends that many empty
    results, negative drops that many from the end."""

    script: Script
    error: BaseException | None = None
    result_offset: int = 0
    calls: list[Ct2Call] = field(default_factory=list)

    def translate_batch(
        self,
        source: list[list[str]],
        target_prefix: list[list[str]] | None = None,
        **options: Any,
    ) -> list[Ct2Result]:
        self.calls.append(Ct2Call(source, target_prefix, dict(options)))
        if self.error is not None:
            raise self.error
        with_scores = bool(options.get("return_scores", False))
        results = [self._answer(tuple(line), with_scores) for line in source]
        if self.result_offset >= 0:
            return results + [Ct2Result([], []) for _ in range(self.result_offset)]
        return results[: self.result_offset]

    def _answer(self, line: tuple[str, ...], with_scores: bool) -> Ct2Result:
        if callable(self.script):
            hypotheses = self.script(line)
        elif line in self.script:
            hypotheses = self.script[line]
        else:
            raise AssertionError(f"the script has no answer for the source {line}")
        return Ct2Result(
            hypotheses=[list(tokens) for tokens, _ in hypotheses],
            scores=[score for _, score in hypotheses] if with_scores else [],
        )


class FakeProcessor:
    """`encode`: one `U+2581` + word piece per whitespace-separated word, no piece for a
    text without a word. `decode`: the inverse. `piece_to_id`, `unk_id` and `id_to_piece`
    over `pieces` (the ids are the positions from 1; 0 is the unknown piece). Records
    every encoded text in `encoded`."""

    def __init__(self, pieces: Sequence[str] = ()) -> None:
        self._pieces = list(pieces)
        self.encoded: list[str] = []

    def encode(self, text: str, out_type: type = str) -> list[str]:
        self.encoded.append(text)
        return [SPIECE + word for word in text.split()]

    def decode(self, pieces: list[str]) -> str:
        return "".join(pieces).replace(SPIECE, " ").removeprefix(" ")

    def piece_to_id(self, piece: str) -> int:
        if piece in self._pieces:
            return self._pieces.index(piece) + 1
        return _UNKNOWN_ID

    def unk_id(self) -> int:
        return _UNKNOWN_ID

    def id_to_piece(self, piece_id: int) -> str:
        return "<unk>" if piece_id == _UNKNOWN_ID else self._pieces[piece_id - 1]
