"""Planted inputs for the synthetic-data job tests: a monolingual frame, a
generator that answers each text with scripted candidates (and can fail on
chosen calls) and a QE scorer that rates scripted pairs. No model, server or
GPU is involved.

The planted outcomes are laid out by row position modulo 8 for a job whose
synthetic side is English (`other_lang` is `eng_Latn`, as in both default jobs
of `SynthConfig`) and whose scorer is `metricx_qe` (lower is better, the
default threshold is 8.0). Row i has the real text `Монгол улс <i:03d>`; with
n = `<i:03d>`:

  0  two good candidates: "The state of Mongolia n stands" and the shorter
     "The state of Mongolia n"; chrF consensus picks the first
  1  a refusal: no candidates at all
  2  an empty candidate, then a Cyrillic one (the wrong script for English)
  3  a copy of the real text; on an English side it fails purity before the
     copy rule is reached (a copy needs a real text in the synthetic script)
  4  one candidate the scorer rates 9.0, above the threshold
  5  one candidate the scorer rates NaN
  6  one candidate the scorer rates exactly 8.0, at the threshold: it passes
  7  four good candidates, the third ("Mongolia is a country n indeed") is the
     chrF consensus
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from rinchen.synth.select import Kind

PLANT_PERIOD = 8
MAX_ROWS = 1000  # ids `s000`..`s999` sort in row order

# The planted QE scores by position modulo `PLANT_PERIOD`; every other pair gets
# the scorer's default.
_PLANTED_QE = {4: 9.0, 5: math.nan, 6: 8.0}

# What a run of the 40 planted rows with the planted scorer keeps and drops, in
# either direction (the counts of `Selection`), and the shard statistics of such
# a run in 5 shards when nothing was on disk.
PLANTED_COUNTS = {
    "sources": 40,
    "candidates": 60,
    "dropped_empty": 5,
    "dropped_purity": 10,
    "dropped_qe": 10,
    "dropped_copy": 0,
    "no_candidates": 5,
    "no_survivor": 20,
    "selected": 15,
}
FIRST_RUN_STATS = {"planned": 5, "reused": 0, "generated": 5, "regenerated_invalid": 0}


def planted_monolingual(n: int = 40, lang: str = "khk_Cyrl") -> pl.DataFrame:
    """`n` rows `id`, `lang`, `text`: ids `s000`.., distinct Cyrillic Mongolian
    texts `Монгол улс <i:03d>`. `lang` is only the label: the text stays
    Cyrillic whatever it says. Sorted by id, as `read_monolingual` returns it."""
    if not 1 <= n <= MAX_ROWS:
        raise ValueError(f"n must be between 1 and {MAX_ROWS}, got {n}")
    return pl.DataFrame(
        {
            "id": [f"s{index:03d}" for index in range(n)],
            "lang": [lang] * n,
            "text": [f"Монгол улс {index:03d}" for index in range(n)],
        }
    )


def _outcome(index: int, real: str) -> list[str]:
    number = f"{index:03d}"
    match index % PLANT_PERIOD:
        case 0:
            return [
                f"The state of Mongolia {number} stands",
                f"The state of Mongolia {number}",
            ]
        case 1:
            return []
        case 2:
            return ["", f"Англи хэл {number}"]
        case 3:
            return [real]
        case 4:
            return [f"Mongolia is poor {number}"]
        case 5:
            return [f"Mongolia is nan {number}"]
        case 6:
            return [f"Mongolia is edge {number}"]
        case _:
            return [
                f"A completely different sentence {number}",
                f"Mongolia is a country {number}",
                f"Mongolia is a country {number} indeed",
                f"Mongolia is the country {number}",
            ]


def planted_script(frame: pl.DataFrame) -> dict[str, list[str]]:
    """The planted candidates for each text of `frame` (see the module
    docstring), keyed by the real text, for `ScriptedTranslator`."""
    return {real: _outcome(index, real) for index, real in enumerate(frame["text"])}


class ScriptedTranslator:
    """A `Translator` that answers each text with its scripted candidate list.

    An unknown text is a test mistake and raises KeyError. Every call is
    recorded in `calls` as `(texts, src_lang, tgt_lang)` before it is answered.
    A call whose number (0-based, counting every call) is in `fail_on_call`
    raises that exception instead of answering."""

    def __init__(
        self,
        script: Mapping[str, Sequence[str]],
        *,
        fail_on_call: Mapping[int, BaseException] | None = None,
    ) -> None:
        self._script = script
        self._fail_on_call = dict(fail_on_call or {})
        self.calls: list[tuple[tuple[str, ...], str, str]] = []

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        call_number = len(self.calls)
        self.calls.append((tuple(texts), src_lang, tgt_lang))
        if call_number in self._fail_on_call:
            raise self._fail_on_call[call_number]
        return [list(self._script[text]) for text in texts]


class FakeScorer:
    """A `QEScorer` that rates a (source, translation) pair with `scores[pair]`,
    else `default`; the default is a good score for `metricx_qe` (lower is
    better), so a `cometkiwi` test passes its own. Records every call as the
    tuples of sources and translations it was given."""

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float],
        *,
        default: float = 1.0,
        requires_cuda: bool = False,
    ) -> None:
        self._scores = dict(scores)
        self._default = default
        self.requires_cuda = requires_cuda
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.calls.append((tuple(sources), tuple(translations)))
        return [
            self._scores.get(pair, self._default)
            for pair in zip(sources, translations, strict=True)
        ]


def planted_scorer(frame: pl.DataFrame, kind: Kind) -> FakeScorer:
    """A `FakeScorer` that rates the planted candidates of rows 4, 5 and 6
    (modulo 8) as the module docstring says. The pair is in the orientation the
    selection scores it in: (candidate, real text) for `bt`, (real text,
    candidate) for `ft`."""
    script = planted_script(frame)
    scores: dict[tuple[str, str], float] = {}
    for index, real in enumerate(frame["text"]):
        score = _PLANTED_QE.get(index % PLANT_PERIOD)
        if score is not None:
            (candidate,) = script[real]
            scores[(candidate, real) if kind == "bt" else (real, candidate)] = score
    return FakeScorer(scores)
