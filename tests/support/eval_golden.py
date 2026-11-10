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

from typing import Final

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
