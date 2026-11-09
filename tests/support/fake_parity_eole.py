"""Stand-ins for the two halves of a strict parity run (D11), so that no test starts Eole or
loads a model.

`FakeParityEole` replaces `run_logged` in `rinchen.train.export_parity`: it records every
launch (the command, the environment, the config and the files the config names) and writes,
as a finished `eole predict` would, the output file the config names, one line
`tokens<TAB>score<TAB>1.0` per source line, from the answers it was given. `raw_lines`
replaces those lines by any text, and `writes_output`, `extra_lines` and `error` make the run
fail in the ways a real one does.

`ParityCt2` is a `ScriptedCt2` that answers as the library does for a parity run: with a
forced target prefix every hypothesis starts with it, and the end token closes a hypothesis
only when the call asks for it with `return_end_token`.

`make_case` puts the two together over a fabricated export and checkpoint: `count` sources
(`w0`, `w1`, ...) that Eole and the library translate into the same pieces, except for the
sources in `differ`, with the log-probabilities of the library worked out by hand. Eole's
score of a sentence is -0.125 for each of its pieces and for the end token (a sum that floats
hold exactly); the library's log-probabilities are the same, plus `deltas[i]` on the end
token, and for the placement `tgt` the log-probability `prefix_log_prob` in front for the tag
(which Eole's score leaves out).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from rinchen.config.ablation import TagPlacement
from rinchen.config.languages import direction_tag
from rinchen.export.record import CheckpointRecord
from rinchen.export.verified import VerifiedExport, verify_export
from rinchen.train import export_parity
from rinchen.train.export_parity import ParitySource
from tests.support.fake_ct2 import (
    SPIECE,
    Ct2Result,
    FakeProcessor,
    Hypothesis,
    ScriptedCt2,
)
from tests.support.model_exports import (
    CHECKPOINT_MANIFEST_SHA256,
    CONFIG_SHA256,
    VOCAB_SHA256,
    write_model_export,
)

END: Final = "</s>"
TARGET: Final = "khk_Cyrl"
TAG: Final = direction_tag(TARGET)
# What Eole scores each piece of a sentence and the end token at, in these tests.
PIECE_LOG_PROB: Final = -0.125
WEIGHTS_BYTES: Final = b"fabricated weights of the checkpoint"
WEIGHTS_SHA256: Final = hashlib.sha256(WEIGHTS_BYTES).hexdigest()


@dataclass(frozen=True)
class EoleLaunch:
    """One launch of the fake: what `run_logged` was given, and what the config's files
    held when the command started."""

    command: list[str]
    env: dict[str, str]
    log_path: Path
    what: str
    timeout: float | None
    config: dict[str, Any]
    source_lines: list[str]
    prefix_lines: list[str]


def _lines_of(config: Mapping[str, Any], key: str) -> list[str]:
    if key not in config:
        return []
    return Path(config[key]).read_text(encoding="utf-8").removesuffix("\n").split("\n")


@dataclass
class FakeParityEole:
    """Stands in for `run_logged`. `answers` are the tokens and the score of each source
    line, in order (for the placement `tgt` the tokens start with the forced tag, as Eole's
    output does)."""

    answers: Sequence[tuple[Sequence[str], float]] = ()
    raw_lines: Sequence[str] | None = None
    writes_output: bool = True
    extra_lines: int = 0
    error: BaseException | None = None
    launches: list[EoleLaunch] = field(default_factory=list)

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        log_path: Path,
        what: str,
        timeout: float | None = None,
    ) -> None:
        config = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        self.launches.append(
            EoleLaunch(
                command=list(command),
                env=dict(env),
                log_path=log_path,
                what=what,
                timeout=timeout,
                config=config,
                source_lines=_lines_of(config, "src"),
                prefix_lines=_lines_of(config, "tgt"),
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("eole log\n", encoding="utf-8")
        if self.error is not None:
            raise self.error
        if not self.writes_output:
            return
        lines = (
            [self._line(tokens, score) for tokens, score in self.answers]
            if self.raw_lines is None
            else list(self.raw_lines)
        )
        lines.extend(self._line([], 0.0) for _ in range(self.extra_lines))
        Path(config["output"]).write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )

    @staticmethod
    def _line(tokens: Sequence[str], score: float) -> str:
        return f"{' '.join(tokens)}\t{score!r}\t1.0"


class ParityCt2(ScriptedCt2):
    """A `ScriptedCt2` whose hypotheses have no end token unless the call asks for it, and
    start with the forced prefix when there is one (unless `keeps_prefix` is turned off, which
    the real library never does)."""

    keeps_prefix = True

    def translate_batch(
        self,
        source: list[list[str]],
        target_prefix: list[list[str]] | None = None,
        **options: Any,
    ) -> list[Ct2Result]:
        results = super().translate_batch(source, target_prefix, **options)
        end = [self.end_token] if options.get("return_end_token") else []
        shaped = []
        for index, result in enumerate(results):
            prefix = (
                target_prefix[index]
                if self.keeps_prefix
                and target_prefix is not None
                and index < len(target_prefix)
                else []
            )
            shaped.append(
                Ct2Result(
                    [[*prefix, *tokens, *end] for tokens in result.hypotheses], []
                )
            )
        return shaped


@dataclass
class ParityCase:
    """The fabricated parts of a strict parity run, and what Eole will answer."""

    export: VerifiedExport
    checkpoint: Path
    out_dir: Path
    processor: FakeProcessor
    sources: list[ParitySource]
    eole: FakeParityEole
    ct2: ParityCt2
    placement: TagPlacement
    answers: list[tuple[list[str], float]]


def write_checkpoint(directory: Path) -> Path:
    """A checkpoint directory with the three files an export's record names."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.00.safetensors").write_bytes(WEIGHTS_BYTES)
    (directory / "config.json").write_bytes(b"{}")
    (directory / "vocab.json").write_bytes(b"{}")
    return directory


def make_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int = 3,
    placement: TagPlacement = "src",
    quantization: str = "float32",
    piece_count: int = 2,
    differ: Sequence[int] = (),
    deltas: Sequence[float] | None = None,
    prefix_log_prob: float = -0.09,
) -> ParityCase:
    """A case of `count` sources, and the fake Eole installed in `rinchen.train.export_parity`.

    Each translation is `piece_count` pieces. Eole scores a sentence -0.125 per piece and
    for the end token. The library answers the same pieces, except that the last piece of
    the sources in `differ` is another one; its log-probabilities for Eole's pieces are
    -0.125 each, the end token's moved by `deltas[i]` (default none) and, for `tgt`, the
    tag's `prefix_log_prob` in front."""
    moves = [0.0] * count if deltas is None else list(deltas)
    assert len(moves) == count
    checkpoint = write_checkpoint(tmp_path / "ckpt")
    written = write_model_export(
        tmp_path / "export",
        placement=placement,
        quantization=quantization,
        checkpoint=CheckpointRecord(
            kind="step",
            step=200,
            weights_sha256=WEIGHTS_SHA256,
            config_sha256=CONFIG_SHA256,
            vocab_sha256=VOCAB_SHA256,
            manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
            manifest_stage="train_launch",
        ),
    )
    export = verify_export(written.directory)
    sources = [ParitySource(f"w{index}", TARGET) for index in range(count)]
    lines: list[tuple[str, ...]] = []
    eole_pieces: list[list[str]] = []
    for index in range(count):
        piece = SPIECE + f"w{index}"
        lines.append((piece,) if placement == "tgt" else (TAG, piece))
        eole_pieces.append([SPIECE + f"h{index}x{each}" for each in range(piece_count)])
    answers = [
        (
            [TAG, *pieces] if placement == "tgt" else pieces,
            PIECE_LOG_PROB * (len(pieces) + 1),
        )
        for pieces in eole_pieces
    ]
    index_of = {line: index for index, line in enumerate(lines)}

    def translated(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        index = index_of[line]
        pieces = list(eole_pieces[index])
        if index in differ:
            pieces[-1] = SPIECE + "other"
        return [(pieces, -0.5)]

    def log_probs(line: tuple[str, ...], target: tuple[str, ...]) -> Sequence[float]:
        index = index_of[line]
        scored = len(eole_pieces[index]) + 1
        generated = [PIECE_LOG_PROB] * scored
        generated[-1] += moves[index]
        return [prefix_log_prob, *generated] if placement == "tgt" else generated

    ct2 = ParityCt2(translated, score_script=log_probs)
    eole = FakeParityEole(answers)
    monkeypatch.setattr(export_parity, "run_logged", eole)
    monkeypatch.setattr(export_parity, "eole_command", lambda: "eole")
    return ParityCase(
        export=export,
        checkpoint=checkpoint,
        out_dir=tmp_path / "parity",
        processor=FakeProcessor(),
        sources=sources,
        eole=eole,
        ct2=ct2,
        placement=placement,
        answers=answers,
    )
