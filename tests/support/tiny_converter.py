"""What the tiny converter run of the integration tests hands to its tests.

`tests/integration/conftest.py` builds a `TinyRun` once per session; the tests read
its corpus files and Eole's predictions through it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rinchen.script_conv.char_model import CONVERTER_TAGS
from rinchen.script_conv.data import CorpusFiles

TO_SCRIPT_TAG, TO_CYRILLIC_TAG = CONVERTER_TAGS
DIRECTION_TAGS = {"cyrl2mong": TO_SCRIPT_TAG, "mong2cyrl": TO_CYRILLIC_TAG}
# Bounds each Eole or CTranslate2 command of the integration tests, so a hung
# subprocess fails the session instead of blocking it.
COMMAND_TIMEOUT_SECONDS = 600


def read_lines(path: Path) -> tuple[str, ...]:
    """The lines of a corpus or prediction file, split on LF only: other line
    separators and NNBSP are part of a word's line."""
    return tuple(path.read_text(encoding="utf-8").removesuffix("\n").split("\n"))


@dataclass(frozen=True)
class TinyRun:
    """One finished tiny training run.

    `predictions` maps each direction (`cyrl2mong`, `mong2cyrl`) to Eole's greedy
    output for the lines of its training source file, in file order.
    """

    corpus: CorpusFiles
    spm_model: Path
    config_path: Path
    run_dir: Path
    ckpt_dir: Path
    train_seconds: float
    predictions: Mapping[str, tuple[str, ...]]

    def lines(self, name: str) -> tuple[str, ...]:
        """The lines of the corpus file `name`, for example `cyrl2mong.train.tgt`."""
        return read_lines(self.corpus.files[name])
