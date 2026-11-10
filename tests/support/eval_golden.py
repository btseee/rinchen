"""The golden corpus of D13: ten sentences in English and in Khalkha, five documents of two
sentences, and the outputs of two systems, `alpha` (with planted errors, the last sentence
missing) and `beta` (close to the references).

Every expected figure that the tests pin for these sentences was computed with the pure-Python
reference of the design spike and sacrebleu's own score-from-statistics, never with the code
under test. Two directions use them: `khk_Cyrl-eng_Latn` (sources `GOLDEN_KHK`, references
`GOLDEN_ENG`, outputs `ALPHA_ENG` or `BETA_ENG`) and `eng_Latn-khk_Cyrl` (sources `GOLDEN_ENG`,
references `GOLDEN_KHK`, outputs `ALPHA_KHK` or `BETA_KHK`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, NamedTuple

GOLDEN_ENG: Final = (
    "the cat sat on the mat",
    "a dog runs in the park",
    "she reads a long book",
    "we eat bread and cheese",
    "the sun is hot today",
    "he drinks cold water",
    "birds fly over the river",
    "my brother plays the piano",
    "they walk to school",
    "the train is late again",
)
GOLDEN_KHK: Final = (
    "муур дэвсгэр дээр сууж байв",
    "нохой цэцэрлэгт хүрээлэнд гүйж байна",
    "тэр урт ном уншдаг",
    "бид талх бяслаг иддэг",
    "өнөөдөр нар халуун байна",
    "тэр хүйтэн ус уудаг",
    "шувууд голын дээгүүр нисдэг",
    "миний ах төгөлдөр хуур тоглодог",
    "тэд сургууль руу алхаж явна",
    "галт тэрэг дахин хоцорлоо",
)
GOLDEN_IDS: Final = tuple(f"g{number:02d}" for number in range(1, 11))
GOLDEN_DOCS: Final = ("d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d5", "d5")

ALPHA_ENG: Final = (
    "the cat sat on the mat",
    "a dog run in the park",
    "she reads a book",
    "we eat bread and cheese",
    "the sun is hot",
    "he drinks water",
    "birds fly over the river",
    "my brother plays piano",
    "they walk to the school",
    "",
)
BETA_ENG: Final = (
    "the cat sat on the mat",
    "a dog runs in the park",
    "she reads a long book",
    "we eat bread and cheese",
    "the sun is hot today",
    "he drinks water",
    "birds fly over the river",
    "my brother plays the piano",
    "they walk to school",
    "the train is late",
)
ALPHA_KHK: Final = (
    GOLDEN_KHK[0],
    "нохой цэцэрлэгт гүйж байна",
    "тэр ном уншдаг",
    GOLDEN_KHK[3],
    "өнөөдөр нар байна",
    "тэр ус уудаг",
    GOLDEN_KHK[6],
    "миний ах хуур тоглодог",
    "тэд сургууль руу явна",
    "",
)
BETA_KHK: Final = (
    GOLDEN_KHK[0],
    GOLDEN_KHK[1],
    GOLDEN_KHK[2],
    GOLDEN_KHK[3],
    "өнөөдөр нар халуун",
    GOLDEN_KHK[5],
    GOLDEN_KHK[6],
    "миний ах төгөлдөр хуур",
    GOLDEN_KHK[8],
    "галт тэрэг хоцорлоо",
)

# ---- The bootstrap (Task 2) ----------------------------------------------------------------
# Derived with the pure-Python reference of the design spike (a loop over `index_py`, sacrebleu's
# score-from-statistics of the integer sums, clamped to 100.0), never with the code under test.
GOLDEN_SEED: Final = 20_261_110
GOLDEN_RESAMPLES: Final = 1000
GOLDEN_ALPHA: Final = 0.05
GOLDEN_POSITIONS: Final = (25, 974)
GOLDEN_DIRECTIONS: Final = ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl")
GOLDEN_SYSTEMS: Final = ("alpha", "beta")
GOLDEN_COUNT_METRICS: Final = ("bleu", "chrf", "chrf_pp")


class SegmentGolden(NamedTuple):
    """The point and the interval of one metric, the segments being the units."""

    point: float
    low: float
    high: float


class PairedGolden(NamedTuple):
    """`beta` minus `alpha`, chrF++, 1,000 resamples: the difference of the full corpora, the
    interval of the resample differences, the two-sided p-value and the share of resamples in
    which `beta` is better."""

    delta: float
    low: float
    high: float
    p_value: float
    better_fraction: float


# (direction, system, metric) -> point and interval over the ten segments.
GOLDEN_SEGMENT_ESTIMATES: Final[Mapping[tuple[str, str, str], SegmentGolden]] = {
    ("khk_Cyrl-eng_Latn", "alpha", "bleu"): SegmentGolden(
        59.40481339822236, 29.956257939026496, 82.32464002694157
    ),
    ("khk_Cyrl-eng_Latn", "alpha", "chrf"): SegmentGolden(
        72.43951493593927, 54.63535272961181, 87.7566570801851
    ),
    ("khk_Cyrl-eng_Latn", "alpha", "chrf_pp"): SegmentGolden(
        73.08276683393322, 55.12895236792601, 88.16473036218294
    ),
    ("khk_Cyrl-eng_Latn", "beta", "bleu"): SegmentGolden(
        94.41920983915246, 84.10619176533582, 100.0
    ),
    ("khk_Cyrl-eng_Latn", "beta", "chrf"): SegmentGolden(
        93.46167452409257, 83.52653935304194, 100.0
    ),
    ("khk_Cyrl-eng_Latn", "beta", "chrf_pp"): SegmentGolden(
        93.87300204175257, 84.33507979458035, 100.0
    ),
    ("eng_Latn-khk_Cyrl", "alpha", "bleu"): SegmentGolden(
        50.86894326533609, 19.468104644353417, 76.07804902091874
    ),
    ("eng_Latn-khk_Cyrl", "alpha", "chrf"): SegmentGolden(
        67.41404584044409, 49.57133463224348, 83.44113512503841
    ),
    ("eng_Latn-khk_Cyrl", "alpha", "chrf_pp"): SegmentGolden(
        68.06307127183763, 50.69295803014816, 83.90991060392365
    ),
    ("eng_Latn-khk_Cyrl", "beta", "bleu"): SegmentGolden(
        91.06781579902749, 80.80592469957207, 97.8495461444958
    ),
    ("eng_Latn-khk_Cyrl", "beta", "chrf"): SegmentGolden(
        91.06726556207575, 82.0321502072076, 98.08255567765114
    ),
    ("eng_Latn-khk_Cyrl", "beta", "chrf_pp"): SegmentGolden(
        91.33781025459385, 82.45607260440461, 98.05731214482036
    ),
}

# (direction, system, metric) -> the interval over the five documents (`GOLDEN_DOCS`); the
# points are the segment ones.
GOLDEN_DOCUMENT_INTERVALS: Final[Mapping[tuple[str, str, str], tuple[float, float]]] = {
    ("khk_Cyrl-eng_Latn", "alpha", "bleu"): (36.33559622758536, 72.13730354815289),
    ("khk_Cyrl-eng_Latn", "alpha", "chrf"): (53.758259687838795, 85.62747863272854),
    ("khk_Cyrl-eng_Latn", "alpha", "chrf_pp"): (54.90789038028186, 85.48385757545739),
    ("khk_Cyrl-eng_Latn", "beta", "bleu"): (85.51323882363222, 100.0),
    ("khk_Cyrl-eng_Latn", "beta", "chrf"): (84.43197075527794, 100.0),
    ("khk_Cyrl-eng_Latn", "beta", "chrf_pp"): (85.23908863844622, 100.0),
    ("eng_Latn-khk_Cyrl", "alpha", "bleu"): (25.5366832841316, 67.87818182443375),
    ("eng_Latn-khk_Cyrl", "alpha", "chrf"): (49.837684278291874, 80.58562671841165),
    ("eng_Latn-khk_Cyrl", "alpha", "chrf_pp"): (50.67413054994893, 80.98143244841619),
    ("eng_Latn-khk_Cyrl", "beta", "bleu"): (83.56447475603727, 97.8022872484601),
    ("eng_Latn-khk_Cyrl", "beta", "chrf"): (84.67226850152994, 97.8429577643661),
    ("eng_Latn-khk_Cyrl", "beta", "chrf_pp"): (85.0715197211207, 97.8580969310857),
}

# (direction, unit) -> the paired comparison of `beta` against `alpha` on chrF++. Resamples at or
# below zero and at or above it: khk_Cyrl-eng_Latn segments 1 and 1,000, every other row 0 and
# 1,000, which is how a p-value of 4/1001 and of 2/1001 comes about.
GOLDEN_PAIRED: Final[Mapping[tuple[str, str], PairedGolden]] = {
    ("khk_Cyrl-eng_Latn", "segment"): PairedGolden(
        20.79023520781935,
        8.618638523199792,
        34.58603066716192,
        0.003996003996003996,
        0.999,
    ),
    ("khk_Cyrl-eng_Latn", "document"): PairedGolden(
        20.79023520781935,
        13.91888026941568,
        33.41274507632882,
        0.001998001998001998,
        1.0,
    ),
    ("eng_Latn-khk_Cyrl", "segment"): PairedGolden(
        23.27473898275622,
        11.521101928419966,
        36.40141307486019,
        0.001998001998001998,
        1.0,
    ),
    ("eng_Latn-khk_Cyrl", "document"): PairedGolden(
        23.27473898275622,
        12.652693037229753,
        36.215446705381645,
        0.001998001998001998,
        1.0,
    ),
}

# The mean-type golden: twelve segment scores of two systems as integer micro-units, each
# a row `[score, 1]` of a mean-type metric.
MEAN_MICRO_A: Final = (
    912000,
    850000,
    770000,
    990000,
    640000,
    880000,
    930000,
    720000,
    810000,
    960000,
    700000,
    870000,
)
MEAN_MICRO_B: Final = (
    900000,
    860000,
    700000,
    980000,
    600000,
    870000,
    940000,
    650000,
    800000,
    950000,
    720000,
    840000,
)
# The sorted resample sums at the positions 25 and 974 of `GOLDEN_POSITIONS`.
MEAN_SUMS_A: Final = (9342000, 10680000)
MEAN_SUMS_B: Final = (9040000, 10600000)
MEAN_SUMS_A_MINUS_B: Final = (40000, 412000)
MEAN_AT_OR_BELOW: Final = 7
MEAN_AT_OR_ABOVE: Final = 994
MEAN_ABOVE: Final = 993
MEAN_P_VALUE: Final = 0.015984015984015984
MEAN_BETTER_FRACTION: Final = 0.993
# As means (the scorer divides the sum by `count * 1_000_000`).
MEAN_POINT_A: Final = 0.836
MEAN_POINT_B: Final = 0.8175
MEAN_INTERVAL_A: Final = (0.7785, 0.89)
MEAN_INTERVAL_DIFFERENCE: Final = (0.0033333333333333335, 0.034333333333333334)

# ---- Scoring and the result documents (Task 6) ----------------------------------------------
# The signatures sacrebleu 2.6.0 reports for the golden corpus (target `eng_Latn` and `khk_Cyrl`:
# BLEU's tokenizer is `13a` for both) and what the bootstrap of the golden suite records.
GOLDEN_SIGNATURES: Final[Mapping[str, str]] = {
    "bleu": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0",
    "chrf": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0",
    "chrf_pp": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0",
}
GOLDEN_BOOTSTRAP_DOCUMENT: Final[Mapping[str, object]] = {
    "resamples": GOLDEN_RESAMPLES,
    "seed": GOLDEN_SEED,
    "alpha": GOLDEN_ALPHA,
    "interval": "percentile",
    "rng": "splitmix64-counter/1",
    "unit": "document",
    "units": 5,
}
GOLDEN_BOOTSTRAP_SEGMENT: Final[Mapping[str, object]] = {
    **GOLDEN_BOOTSTRAP_DOCUMENT,
    "unit": "segment",
    "units": 10,
}
# The words of the reason every neural metric carries when no segment scores were given.
GOLDEN_NO_SCORER_REASON: Final = (
    "no segment scores were given: no scorer is built (B1 and B2 are open)"
)
