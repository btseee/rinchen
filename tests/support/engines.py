"""An `Engine` over a fabricated export, for the tests of D11.

`scripted_engine` writes a COMPLETE model export (`write_model_export`), and opens an
`Engine` over it through the two seams the engine has: a CTranslate2 stand-in and a
SentencePiece stand-in. Nothing loads a model, starts a process or touches a GPU, and
`ctranslate2` is never imported.

The vocabulary (`ENGINE_TOKENS`) is what a model of ours has: Eole's own entries that are no
piece of the SentencePiece model (`<blank>`, a padding token), the pieces a test writes, one
of each script, two byte pieces, then every reserved symbol of the project. `ENGINE_PIECES`
are the pieces of the stand-in processor: the same, without Eole's entries and without the
unknown piece (which is id 0).

The stand-ins of the library:

- `echo` answers a source line with its own pieces (the tag of the placement `src` left off)
  and the end token: the "translation" of a text is the text, so the output has the script of
  the input;
- `LiveCt2` is a `ScriptedCt2` that also puts a forced target prefix in front of every
  hypothesis, as the library does, and can answer by what the call suppresses (the way a
  model that wants to write the wrong script answers once the pieces of that script are
  suppressed);
- `FakeConverter` stands for D5's `ScriptConverter`.
"""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import pytest

from rinchen.config.ablation import TagPlacement
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import LoadOptions
from rinchen.spm.symbols import user_defined_symbols
from tests.support.fake_ct2 import (
    SPIECE,
    Ct2Result,
    FakeProcessor,
    Hypothesis,
    Script,
    ScriptedCt2,
)
from tests.support.model_exports import TOY_DIRECTIONS, WrittenModel, write_model_export

SP: Final = chr(0x2581)
END: Final = "</s>"
ENG, KHK, RUS, MON = "eng_Latn", "khk_Cyrl", "rus_Cyrl", "mn_Mong"
TAG_ENG, TAG_KHK, TAG_RUS, TAG_MON = (
    "<2eng_Latn>",
    "<2khk_Cyrl>",
    "<2rus_Cyrl>",
    "<2mn_Mong>",
)
# Three letters of the Mongolian script (U+1820, U+1821, U+1822).
MONGOLIAN: Final = "".join(chr(code) for code in (0x1820, 0x1821, 0x1822))
NNBSP: Final = chr(0x202F)

# What Eole adds to the vocabulary and the SentencePiece model has no piece for.
FOREIGN: Final = ("<blank>", "averyunlikelytoken0")
WORDS: Final = (
    SP + "hello",
    SP + "world",
    SP + "Th",
    SP + "Энэ",
    SP + "ном",
    SP + "ус",
    SP + MONGOLIAN,
)
BYTES: Final = ("<0x41>", "<0xE0>")
ENGINE_PIECES: Final = ("<s>", END, *WORDS, *BYTES, *user_defined_symbols())
ENGINE_TOKENS: Final = (
    "<unk>",
    "<blank>",
    "<s>",
    END,
    FOREIGN[1],
    *WORDS,
    *BYTES,
    *user_defined_symbols(),
)


def hyp(*tokens: str, score: float = -0.5) -> Hypothesis:
    """One scripted hypothesis: its tokens and a score."""
    return (list(tokens), score)


def echo(line: tuple[str, ...]) -> Sequence[Hypothesis]:
    """The pieces of the source (the tag of the placement `src` left off) and the end token."""
    return [hyp(*line[1:], END)]


def echo_untagged(line: tuple[str, ...]) -> Sequence[Hypothesis]:
    """The pieces of an untagged source (placement `tgt`) and the end token."""
    return [hyp(*line, END)]


def copy_windows(line: tuple[str, ...]) -> Sequence[Hypothesis]:
    """A model of documents: a window (`[tag, "<doc>", *pieces]`) is copied with the `<sep>`
    pieces turned into the symbol itself, and a sentence is copied as `echo` does."""
    if len(line) > 1 and line[1] == "<doc>":
        return [hyp(*["<sep>" if p == SP + "<sep>" else p for p in line[2:]], END)]
    return echo(line)


class SymbolProcessor(FakeProcessor):
    """A `FakeProcessor` that cuts a reserved symbol out of a text as SentencePiece does: the
    symbol is one piece of its own, without the dummy prefix, wherever it stands in a word
    (the model keeps every reserved symbol as a user-defined piece)."""

    def __init__(self, pieces: Sequence[str], symbols: Sequence[str]) -> None:
        super().__init__(pieces)
        self._symbol_pattern = re.compile(
            "("
            + "|".join(re.escape(s) for s in sorted(symbols, key=len, reverse=True))
            + ")"
        )
        self._symbols = frozenset(symbols)

    def encode(self, text: str, out_type: type = str) -> list[str]:
        self.encoded.append(text)
        pieces: list[str] = []
        for word in text.split():
            for part in self._symbol_pattern.split(word):
                if part in self._symbols:
                    pieces.append(part)
                elif part:
                    pieces.append(SPIECE + part)
        return pieces


def symbol_processor() -> SymbolProcessor:
    """The processor of the engine's pieces that knows every reserved symbol of the project."""
    return SymbolProcessor(ENGINE_PIECES, user_defined_symbols())


class LiveCt2(ScriptedCt2):
    """A `ScriptedCt2` for the engine.

    With a forced target prefix every hypothesis starts with it, as the library's do. With
    `answer`, the script is `answer(line, suppressed)`: `suppressed` is the set of the entries
    the call asked to suppress, `frozenset()` when it asked for none."""

    def __init__(
        self,
        script: Script | None = None,
        *,
        answer: Callable[[tuple[str, ...], frozenset[str]], Sequence[Hypothesis]]
        | None = None,
        **options: Any,
    ) -> None:
        self._seen = threading.local()
        self._answer_with = answer
        self.unloaded = 0
        super().__init__(echo if script is None else script, **options)

    def translate_batch(
        self,
        source: list[list[str]],
        target_prefix: list[list[str]] | None = None,
        **options: Any,
    ) -> list[Ct2Result]:
        self._seen.suppressed = frozenset(
            entry[0] for entry in options.get("suppress_sequences") or []
        )
        results = super().translate_batch(source, target_prefix, **options)
        if target_prefix is None:
            return results
        return [
            Ct2Result(
                [[*prefix, *tokens] for tokens in result.hypotheses], result.scores
            )
            for prefix, result in zip(target_prefix, results, strict=False)
        ]

    def _answer(self, line: tuple[str, ...], with_scores: bool) -> Ct2Result:
        if self._answer_with is None:
            return super()._answer(line, with_scores)
        hypotheses = self._answer_with(line, self._seen.suppressed)
        return Ct2Result(
            hypotheses=[list(tokens) for tokens, _ in hypotheses],
            scores=[score for _, score in hypotheses] if with_scores else [],
        )

    def unload_model(self) -> None:
        self.unloaded += 1


@dataclass
class FakeConverter:
    """D5's `ScriptConverter` as the engine sees it: `convert(text, target)` records its
    call and returns `prefix + text` with the two counts it was given."""

    prefix: str = "[mong]"
    rule_fallbacks: int = 2
    invalid_outputs: int = 1
    calls: list[tuple[str, str]] = field(default_factory=list)

    def convert(self, text: str, target: str) -> SimpleNamespace:
        self.calls.append((text, target))
        return SimpleNamespace(
            text=self.prefix + text,
            rule_fallbacks=self.rule_fallbacks,
            invalid_outputs=self.invalid_outputs,
        )


@dataclass
class ScriptedEngine:
    """An engine and the stand-ins it was opened over."""

    engine: Engine
    ct2: ScriptedCt2
    processor: FakeProcessor
    written: WrittenModel
    vocabulary: tuple[str, ...]


def scripted_engine(
    tmp_path: Path,
    *,
    placement: TagPlacement = "src",
    directions: Sequence[str] = TOY_DIRECTIONS,
    document_level: bool = False,
    quantization: str = "float32",
    script: Script | None = None,
    answer: Callable[[tuple[str, ...], frozenset[str]], Sequence[Hypothesis]]
    | None = None,
    converter: FakeConverter | None = None,
    load: LoadOptions | None = None,
    tokens: Sequence[str] = ENGINE_TOKENS,
    pieces: Sequence[str] = ENGINE_PIECES,
    processor: FakeProcessor | None = None,
    compute_type: str = "float32",
    ct2: ScriptedCt2 | None = None,
) -> ScriptedEngine:
    """An engine over a fabricated export in `tmp_path / "model"`.

    The library answers from `script` (default: `echo` for the placement `src`,
    `echo_untagged` for `tgt`) or, with `answer`, by what the call suppresses; `ct2` is a
    library of the test's own."""
    written = write_model_export(
        tmp_path / "model",
        placement=placement,
        directions=directions,
        quantization=quantization,
        document_level=document_level,
        tokens=tokens,
    )
    if ct2 is None:
        if script is None and answer is None:
            script = echo if placement == "src" else echo_untagged
        ct2 = LiveCt2(script, answer=answer)
    ct2.compute_type = compute_type
    fake_processor = FakeProcessor(pieces) if processor is None else processor
    engine = Engine.open(
        written.directory,
        LoadOptions() if load is None else load,
        script_converter=converter,
        translator=ct2,
        processor=fake_processor,
    )
    return ScriptedEngine(engine, ct2, fake_processor, written, tuple(tokens))


class SilentLog:
    """The logger `rinchen.utils.logging.get` returns, saying nothing."""

    def info(self, event: str, **fields: Any) -> None:
        return None


@pytest.fixture
def engine_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a test that opens an engine over the stand-ins: the real `ctranslate2` is made
    unimportable, so a test that reached it would fail instead of loading it, and the logger
    of the engine is silent, because the first call of `rinchen.utils.logging.get` configures
    the root logger for the rest of the session and `test_logging` expects to be the first."""
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setattr("rinchen.utils.logging.get", lambda name: SilentLog())
