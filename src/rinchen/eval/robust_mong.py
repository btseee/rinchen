"""The `mn_Mong` source metric of D12: does a model read the traditional script as well as
the Cyrillic one?

The same Khalkha sentences are translated from three arms: `cyrl` (the control: the Cyrillic
source), `mong` (the sentence in traditional script, read by the model itself) and, when a
script converter is attached, `converted` (the traditional source converted to Cyrillic
first). The key figure is the retention, the corpus chrF of the direct arm over the corpus
chrF of the control, both against the references; it is not clipped (a direct arm may do
better than the control) and it is `None`, never 0 or 1, when the control scores 0 (there is
nothing to retain); it says nothing when the control's chrF is near zero (two poor scores give a
ratio near 1), so a report shows both. The converted arm measures the converter as much as the
model, which is why it is optional and reported beside the direct one."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rinchen.metrics.bleu_chrf import corpus_scores


@dataclass(frozen=True, slots=True)
class MongScores:
    """The corpus chrF of each arm, the retention of the direct and of the converted arm
    (`None` without a converted arm, or when the control scores 0), `agreement`, the
    share of rows whose `cyrl` and `mong` texts are equal, and the `chrf_signature` of the
    sacrebleu scorer (its settings and version: no timestamp, the same on every run)."""

    rows: int
    chrf_cyrl: float
    chrf_mong: float
    retention: float | None
    agreement: float
    chrf_converted: float | None
    retention_converted: float | None
    chrf_signature: str


def score_mong(
    references: Sequence[str],
    cyrl: Sequence[str],
    mong: Sequence[str],
    converted: Sequence[str] | None = None,
) -> MongScores:
    """Score the arms, one text per reference each.

    Raises:
        ValueError: no rows, or an arm that does not hold one text per reference.
    """
    if not references:
        raise ValueError("at least one row is needed to score the arms")
    _require_one_text_per_reference("cyrl", cyrl, references)
    _require_one_text_per_reference("mong", mong, references)
    if converted is not None:
        _require_one_text_per_reference("converted", converted, references)
    control = corpus_scores(cyrl, references)
    chrf_cyrl = control.chrf
    chrf_mong = _chrf(mong, references)
    chrf_converted = None if converted is None else _chrf(converted, references)
    return MongScores(
        rows=len(references),
        chrf_cyrl=chrf_cyrl,
        chrf_mong=chrf_mong,
        retention=_retention(chrf_mong, chrf_cyrl),
        agreement=sum(a == b for a, b in zip(cyrl, mong, strict=True)) / len(cyrl),
        chrf_converted=chrf_converted,
        retention_converted=(
            None if chrf_converted is None else _retention(chrf_converted, chrf_cyrl)
        ),
        chrf_signature=control.chrf_signature,
    )


def _require_one_text_per_reference(
    arm: str, texts: Sequence[str], references: Sequence[str]
) -> None:
    if len(texts) != len(references):
        raise ValueError(
            f"the {arm} arm holds {len(texts)} texts for {len(references)} references"
        )


def _chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    return corpus_scores(hypotheses, references).chrf


def _retention(chrf_arm: float, chrf_control: float) -> float | None:
    return None if chrf_control == 0.0 else chrf_arm / chrf_control
