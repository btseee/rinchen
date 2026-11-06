"""Unit tests for `rinchen.split` (dispatcher, `mn_Mong` and CJK rules).

Traditional-script strings are written with code-point escapes so the exact
input is reviewable; U+1803 MONGOLIAN FULL STOP, U+1809 MONGOLIAN MANCHU FULL
STOP and U+1802 MONGOLIAN COMMA are the punctuation under test. The Chinese
and Japanese sentences are written as plain characters.
"""

from __future__ import annotations

import pytest

from rinchen.errors import ConfigError
from rinchen.split import split

S1 = "\u1820\u1821"  # two traditional-script letters standing for a word
S2 = "\u1822\u1823"

_SPLIT_LANGUAGES = ("mn_Mong", "zho_Hans", "jpn_Jpan")


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


@pytest.mark.parametrize(
    ("lang", "text"),
    [("mn_Mong", f"{S1} {S2}"), ("zho_Hans", "今天天气很好")],
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
    ],
)
def test_parts_are_stripped_nonempty_and_cover_the_text(lang: str, text: str) -> None:
    parts = split(text, lang)

    assert parts
    assert all(part and part == part.strip() for part in parts)
    assert "".join(parts) == "".join(text.split())


@pytest.mark.parametrize("lang", ["mn_mong", "", "xx_Xxxx"])
def test_unknown_language_raises_config_error(lang: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        split(f"{S1}\u1803 {S2}", lang)


def test_unknown_language_raises_even_for_empty_text() -> None:
    with pytest.raises(ConfigError):
        split("", "mn_mong")
