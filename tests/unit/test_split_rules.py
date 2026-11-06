"""Unit tests for `rinchen.split` (dispatcher and every rule family).

Traditional-script strings are written with code-point escapes so the exact
input is reviewable; U+1803 MONGOLIAN FULL STOP, U+1809 MONGOLIAN MANCHU FULL
STOP and U+1802 MONGOLIAN COMMA are the punctuation under test. The Chinese,
Japanese, Korean, Latin-script and Cyrillic sentences are written as plain
characters.
"""

from __future__ import annotations

import time

import pytest

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError
from rinchen.split import split

S1 = "\u1820\u1821"  # two traditional-script letters standing for a word
S2 = "\u1822\u1823"
NBSP = "\N{NO-BREAK SPACE}"  # spaced abbreviations may use it between tokens

_SPLIT_LANGUAGES = (
    "khk_Cyrl",
    "mn_Mong",
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
)


def test_mn_mong_splits_on_full_stops_and_question() -> None:
    assert split(f"{S1}\u1803 {S2}\u1809 {S1}?", "mn_Mong") == [
        f"{S1}\u1803",
        f"{S2}\u1809",
        f"{S1}?",
    ]


def test_mn_mong_comma_never_splits() -> None:
    assert split(f"{S1}\u1802 {S2}\u1803", "mn_Mong") == [f"{S1}\u1802 {S2}\u1803"]


def test_mn_mong_splits_without_following_space() -> None:
    assert split(f"{S1}\u1803{S2}\u1803", "mn_Mong") == [f"{S1}\u1803", f"{S2}\u1803"]


def test_mn_mong_stacked_terminators_stay_together() -> None:
    assert split(f"{S1}?! {S2}", "mn_Mong") == [f"{S1}?!", S2]


def test_mn_mong_closing_quote_stays_with_its_sentence() -> None:
    assert split(f'{S1}?" {S2}', "mn_Mong") == [f'{S1}?"', S2]


@pytest.mark.parametrize("tail", ["\u1803", "?!", "?!  ", "\u1803\u1803\n"])
def test_mn_mong_trailing_terminator_creates_no_extra_part(tail: str) -> None:
    assert split(f"{S1}{tail}", "mn_Mong") == [f"{S1}{tail}".strip()]


def test_cjk_splits_after_fullwidth_terminators() -> None:
    assert split("今天天气很好。你好吗？我很好！", "zho_Hans") == [
        "今天天气很好。",
        "你好吗？",
        "我很好！",
    ]
    assert split("今日は晴れです。明日は雨ですか？はい。", "jpn_Jpan") == [
        "今日は晴れです。",
        "明日は雨ですか？",
        "はい。",
    ]


def test_cjk_splits_when_whitespace_follows_terminator() -> None:
    assert split("今天天气很好。 你好吗？\n我很好！", "zho_Hans") == [
        "今天天气很好。",
        "你好吗？",
        "我很好！",
    ]


def test_cjk_closers_stay_with_their_sentence() -> None:
    assert split("他说：「你好。」然后走了。", "zho_Hans") == [
        "他说：「你好。」",
        "然后走了。",
    ]


@pytest.mark.parametrize("tail", ["。", "？！", "。」  ", "！\n"])
def test_cjk_trailing_terminator_creates_no_extra_part(tail: str) -> None:
    assert split(f"你好{tail}", "zho_Hans") == [f"你好{tail}".strip()]


_LATIN_AND_CYRILLIC_CASES = [
    # English: terminators, abbreviations, initials, decimals, ellipsis, closers.
    (
        "eng_Latn",
        "Hello world. How are you? Fine!",
        ["Hello world.", "How are you?", "Fine!"],
    ),
    (
        "eng_Latn",
        "Dr. Smith arrived. He sat.",
        ["Dr. Smith arrived.", "He sat."],
    ),
    (
        "eng_Latn",
        "DR. Smith arrived. He sat.",
        ["DR. Smith arrived.", "He sat."],
    ),
    (
        "eng_Latn",
        "J. K. Rowling wrote it. Then left.",
        ["J. K. Rowling wrote it.", "Then left."],
    ),
    (
        "eng_Latn",
        "J.K. Rowling wrote it. Then left.",
        ["J.K. Rowling wrote it.", "Then left."],
    ),
    ("eng_Latn", "Pi is 3.14 today. Ok.", ["Pi is 3.14 today.", "Ok."]),
    ("eng_Latn", "I have 3. Then go.", ["I have 3.", "Then go."]),
    ("eng_Latn", "Wait… Then go.", ["Wait…", "Then go."]),
    ("eng_Latn", "Really?! Yes.", ["Really?!", "Yes."]),
    (
        "eng_Latn",
        'He said "stop." Then left.',
        ['He said "stop."', "Then left."],
    ),
    ("eng_Latn", "no terminator here", ["no terminator here"]),
    ("eng_Latn", "See also Fig.", ["See also Fig."]),
    # German: ordinals and abbreviations with inner dots.
    (
        "deu_Latn",
        "Am 3. Oktober feiern wir. Das ist schön.",
        ["Am 3. Oktober feiern wir.", "Das ist schön."],
    ),
    (
        "deu_Latn",
        "Das ist z.B. ein Test. Gut.",
        ["Das ist z.B. ein Test.", "Gut."],
    ),
    (
        "deu_Latn",
        "Das ist z. B. ein Test. Gut.",
        ["Das ist z. B. ein Test.", "Gut."],
    ),
    (
        "deu_Latn",
        f"Das ist z.{NBSP}B. ein Test. Gut.",
        [f"Das ist z.{NBSP}B. ein Test.", "Gut."],
    ),
    (
        "deu_Latn",
        "Wir kommen u. a. am Montag. Gut.",
        ["Wir kommen u. a. am Montag.", "Gut."],
    ),
    (
        "deu_Latn",
        "Das gilt i. d. R. für alle. Gut.",
        ["Das gilt i. d. R. für alle.", "Gut."],
    ),
    (
        "deu_Latn",
        "Das ist d. h. ein Test. Gut.",
        ["Das ist d. h. ein Test.", "Gut."],
    ),
    # A window of dotted tokens must not swallow a real sentence boundary.
    (
        "deu_Latn",
        "Ich bin hier. B. kommt später.",
        ["Ich bin hier.", "B. kommt später."],
    ),
    # Turkish and Korean use the base rule: no abbreviations, no initials.
    (
        "tur_Latn",
        "Bugün hava güzel. Yarın yağmur yağacak.",
        ["Bugün hava güzel.", "Yarın yağmur yağacak."],
    ),
    ("tur_Latn", "Dr. Ali geldi.", ["Dr.", "Ali geldi."]),
    (
        "kor_Hang",
        "오늘은 날씨가 좋습니다. 내일은 비가 옵니다.",
        ["오늘은 날씨가 좋습니다.", "내일은 비가 옵니다."],
    ),
    ("kor_Hang", "A. 홍길동이 왔습니다.", ["A.", "홍길동이 왔습니다."]),
    # Mongolian in Cyrillic.
    (
        "khk_Cyrl",
        "Монгол Улсын нийслэл Улаанбаатар хот юм. Энэ хот Туул голын хөндийд оршдог.",
        [
            "Монгол Улсын нийслэл Улаанбаатар хот юм.",
            "Энэ хот Туул голын хөндийд оршдог.",
        ],
    ),
    (
        "khk_Cyrl",
        "Чи хаанаас ирсэн бэ? Би Монголоос ирсэн.",
        ["Чи хаанаас ирсэн бэ?", "Би Монголоос ирсэн."],
    ),
    (
        "khk_Cyrl",
        "Б. Баяр, Д. Дорж нар ирлээ. Дараа нь явав.",
        ["Б. Баяр, Д. Дорж нар ирлээ.", "Дараа нь явав."],
    ),
    (
        "khk_Cyrl",
        "А.Б. Баяр ирлээ. Дараа нь явав.",
        ["А.Б. Баяр ирлээ.", "Дараа нь явав."],
    ),
    (
        "khk_Cyrl",
        "Үнэ 1.5 сая төгрөг. Огноо 12.03.2024 байна. Сайн.",
        ["Үнэ 1.5 сая төгрөг.", "Огноо 12.03.2024 байна.", "Сайн."],
    ),
    (
        "khk_Cyrl",
        "Тэр ирлээ. — Сайн уу? — гэв.",
        ["Тэр ирлээ.", "— Сайн уу?", "— гэв."],
    ),
    (
        "khk_Cyrl",
        'Би "за" гэв. "Явъя" гэв.',
        ['Би "за" гэв.', '"Явъя" гэв.'],
    ),
    (
        "khk_Cyrl",
        "Тэр ирлээ... дараа нь явав.",
        ["Тэр ирлээ... дараа нь явав."],
    ),
    (
        "khk_Cyrl",
        "Тэр ирлээ. дараа нь явав.",
        ["Тэр ирлээ. дараа нь явав."],
    ),
    ("khk_Cyrl", "Тэр ирлээ. Өнөөдөр сайхан.", ["Тэр ирлээ.", "Өнөөдөр сайхан."]),
    ("khk_Cyrl", "Тэр ирлээ. Үнэхээр.", ["Тэр ирлээ.", "Үнэхээр."]),
    ("khk_Cyrl", "Үнэхээр үү?! Тийм.", ["Үнэхээр үү?!", "Тийм."]),
    # Russian and Kazakh.
    (
        "rus_Cyrl",
        "В 2020 г. Москва выросла. Это хорошо.",
        ["В 2020 г. Москва выросла.", "Это хорошо."],
    ),
    (
        "rus_Cyrl",
        "Он живёт на ул. Ленина. Это далеко.",
        ["Он живёт на ул. Ленина.", "Это далеко."],
    ),
    ("rus_Cyrl", "Он ушёл. Ёж спит.", ["Он ушёл.", "Ёж спит."]),
    (
        "rus_Cyrl",
        "Мы купили хлеб, молоко и т. д. Потом ушли.",
        ["Мы купили хлеб, молоко и т. д. Потом ушли."],
    ),
    (
        "kaz_Cyrl",
        "Қазақстан — үлкен ел. Астана — оның астанасы.",
        ["Қазақстан — үлкен ел.", "Астана — оның астанасы."],
    ),
    # Kazakh dotted abbreviations from the official punctuation rules: the units
    # year, century, minute, hour and second, and "and so on".
    (
        "kaz_Cyrl",
        "1991 ж. Қазақстан тәуелсіздік алды. Бұл маңызды күн.",
        ["1991 ж. Қазақстан тәуелсіздік алды.", "Бұл маңызды күн."],
    ),
    (
        "kaz_Cyrl",
        "19 ғ. Абай өмір сүрді. Ол ақын болды.",
        ["19 ғ. Абай өмір сүрді.", "Ол ақын болды."],
    ),
    (
        "kaz_Cyrl",
        "Жиналыс 40 мин. Астанада өтті. Бәрі келді.",
        ["Жиналыс 40 мин. Астанада өтті.", "Бәрі келді."],
    ),
    (
        "kaz_Cyrl",
        "Біз 3 сағ. Астанада болдық. Содан кейін кеттік.",
        ["Біз 3 сағ. Астанада болдық.", "Содан кейін кеттік."],
    ),
    (
        "kaz_Cyrl",
        "Ол 30 сек. Алматыда тұрды. Содан кейін кетті.",
        ["Ол 30 сек. Алматыда тұрды.", "Содан кейін кетті."],
    ),
    (
        "kaz_Cyrl",
        "Алма, өрік т.б. Алматыда өседі. Бұл жақсы.",
        ["Алма, өрік т.б. Алматыда өседі.", "Бұл жақсы."],
    ),
]


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    _LATIN_AND_CYRILLIC_CASES,
    ids=[
        f"{case[0]}-{number}" for number, case in enumerate(_LATIN_AND_CYRILLIC_CASES)
    ],
)
def test_latin_and_cyrillic_split_cases(
    lang: str, text: str, expected: list[str]
) -> None:
    assert split(text, lang) == expected


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("tur_Latn", "Bu z. B. gibi. Tamam.", ["Bu z.", "B.", "gibi.", "Tamam."]),
        ("kor_Hang", "예: z. B. 안녕. 확인.", ["예: z.", "B.", "안녕.", "확인."]),
    ],
)
def test_turkish_and_korean_have_no_abbreviation_handling(
    lang: str, text: str, expected: list[str]
) -> None:
    # These families have no abbreviation set, so "z. B." is cut like any other
    # sentence end; the spaced-abbreviation rule must not apply to them.
    assert split(text, lang) == expected


def test_closer_after_a_dot_keeps_each_part_stable_when_split_again() -> None:
    # The dot of "a.)" has a closer after it and joins no window. If it consulted
    # its neighbours it would pair with "u." and be suppressed while "u." is cut,
    # and the part "a.) o." would split again on its own.
    text = "u. a.) o. Foo"

    parts = split(text, "deu_Latn")

    assert all(split(part, "deu_Latn") == [part] for part in parts)
    assert "".join("".join(parts).split()) == "".join(text.split())


@pytest.mark.parametrize("lang", ["khk_Cyrl", "rus_Cyrl", "kaz_Cyrl"])
def test_cyrillic_languages_share_one_rule_family(lang: str) -> None:
    text = "Б. Баяр жил на ул. Ленина. Это далеко."

    assert split(text, lang) == ["Б. Баяр жил на ул. Ленина.", "Это далеко."]


_KAZAKH_YEAR_TEXT = "1991 ж. Қазақстан тәуелсіздік алды."
_MONGOLIAN_ETC_TEXT = "Улаанбаатар, Дархан г.м. Монгол хотууд олон."


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("kaz_Cyrl", [_KAZAKH_YEAR_TEXT]),
        ("khk_Cyrl", ["1991 ж.", "Қазақстан тәуелсіздік алды."]),
        ("rus_Cyrl", ["1991 ж.", "Қазақстан тәуелсіздік алды."]),
    ],
)
def test_kazakh_only_abbreviation_does_not_apply_to_mongolian_or_russian(
    lang: str, expected: list[str]
) -> None:
    assert split(_KAZAKH_YEAR_TEXT, lang) == expected


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("khk_Cyrl", [_MONGOLIAN_ETC_TEXT]),
        ("kaz_Cyrl", ["Улаанбаатар, Дархан г.м.", "Монгол хотууд олон."]),
        ("rus_Cyrl", ["Улаанбаатар, Дархан г.м.", "Монгол хотууд олон."]),
    ],
)
def test_mongolian_only_abbreviation_does_not_apply_to_kazakh_or_russian(
    lang: str, expected: list[str]
) -> None:
    assert split(_MONGOLIAN_ETC_TEXT, lang) == expected


# The minimum abbreviations each family must know, with a capitalised word to
# follow them (an abbreviation never ends a sentence here).
_MINIMUM_ABBREVIATIONS = [
    (
        "eng_Latn",
        "Smith",
        [
            "mr",
            "mrs",
            "ms",
            "dr",
            "prof",
            "st",
            "vs",
            "etc",
            "e.g",
            "i.e",
            "fig",
            "inc",
        ],
    ),
    (
        "deu_Latn",
        "Müller",
        [
            "z.b",
            "d.h",
            "u.a",
            "bzw",
            "usw",
            "dr",
            "prof",
            "nr",
            "str",
            "ca",
            "vgl",
        ],
    ),
    (
        "khk_Cyrl",
        "Баяр",
        [
            "т.д",
            "т.е",
            "т.п",
            "др",
            "см",
            "ул",
            "г",
            "гг",
            "им",
            "проф",
            "тыс",
            "млн",
            "руб",
            "рис",
            "стр",
        ],
    ),
    # Mongolian-only entries: and so on, for example, in other words, before the
    # common era.
    ("khk_Cyrl", "Монгол", ["г.м", "ж.нь", "ө.х", "м.э.ө"]),
    # Kazakh-only entries: year, century, second, minute, hour, and so on, and the
    # like, and the like (long form), before the common era.
    (
        "kaz_Cyrl",
        "Қазақстан",
        ["ж", "ғ", "сек", "мин", "сағ", "т.б", "т.с", "т.т", "т.с.с", "б.з.д"],
    ),
]


@pytest.mark.parametrize(
    ("lang", "abbreviation", "next_word"),
    [
        (lang, abbreviation, next_word)
        for lang, next_word, abbreviations in _MINIMUM_ABBREVIATIONS
        for abbreviation in abbreviations
    ],
)
def test_minimum_abbreviations_do_not_end_a_sentence(
    lang: str, abbreviation: str, next_word: str
) -> None:
    text = f"{abbreviation}. {next_word}"

    assert split(text, lang) == [text]


@pytest.mark.parametrize(
    ("lang", "word"),
    [("eng_Latn", "Hello"), ("kor_Hang", "안녕하세요"), ("khk_Cyrl", "Монгол")],
)
@pytest.mark.parametrize("tail", [".", "?!", "...", "?!  ", ".\n"])
def test_trailing_terminator_creates_no_extra_part(
    lang: str, word: str, tail: str
) -> None:
    assert split(f"{word}{tail}", lang) == [f"{word}{tail}".strip()]


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("mn_Mong", f"{S1} {S2}"),
        ("zho_Hans", "今天天气很好"),
        ("eng_Latn", "Hello world"),
        ("khk_Cyrl", "Монгол Улс"),
    ],
)
def test_no_terminator_returns_single_part(lang: str, text: str) -> None:
    assert split(text, lang) == [text]


@pytest.mark.parametrize("lang", _SPLIT_LANGUAGES)
def test_empty_and_whitespace_only_return_empty_list(lang: str) -> None:
    assert split("", lang) == []
    assert split("  \n ", lang) == []


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("mn_Mong", f"  {S1}\u1803 \n {S2}\u1809{S1}?  {S2}  "),
        ("zho_Hans", "  今天天气很好。 \n 你好吗？我很好！  "),
        ("jpn_Jpan", "\u3000今日は晴れです。\u3000明日は雨ですか？ "),
        ("eng_Latn", "  Dr. Smith left.  \n Then Mr. Jones came?!  Fine.  "),
        ("khk_Cyrl", "  Б. Баяр ирлээ. \n Дараа нь явав?!  Сайн.  "),
    ],
)
def test_parts_are_stripped_nonempty_and_cover_the_text(lang: str, text: str) -> None:
    parts = split(text, lang)

    assert parts
    assert all(part and part == part.strip() for part in parts)
    assert "".join("".join(parts).split()) == "".join(text.split())
    position = 0
    for part in parts:  # each part is a verbatim slice, found in order
        position = text.index(part, position) + len(part)


@pytest.mark.parametrize("lang", ["mn_mong", "", "xx_Xxxx"])
def test_unknown_language_raises_config_error(lang: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        split(f"{S1}\u1803 {S2}", lang)


def test_unknown_language_raises_even_for_empty_text() -> None:
    with pytest.raises(ConfigError):
        split("", "mn_mong")


def test_every_registry_language_has_a_split_family() -> None:
    for code in sorted(known_language_codes()):
        assert split("x", code) == ["x"], code


_RUN_LENGTH = 60_000
_MAX_SECONDS_FOR_TERMINATOR_RUN = 0.5


@pytest.mark.parametrize(
    ("lang", "terminator", "tail"),
    [
        ("mn_Mong", chr(0x1803), ""),
        ("mn_Mong", "?", ""),
        ("zho_Hans", chr(0x3002), ""),
        ("eng_Latn", ".", ""),
        ("eng_Latn", ".", " 3"),
        ("khk_Cyrl", ".", ""),
        ("khk_Cyrl", ".", " 3"),
    ],
    ids=lambda value: repr(value),
)
def test_long_terminator_run_without_a_sentence_start_splits_in_linear_time(
    lang: str, terminator: str, tail: str
) -> None:
    # Nothing starts a sentence after the run (a digit starts none in the spaced
    # families; the end of the text starts none anywhere), so it is one part.
    text = terminator * _RUN_LENGTH + tail

    started = time.perf_counter()
    parts = split(text, lang)
    elapsed = time.perf_counter() - started

    assert parts == [text]
    assert elapsed < _MAX_SECONDS_FOR_TERMINATOR_RUN
