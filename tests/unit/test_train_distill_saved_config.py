"""Tests for `rinchen.train.distill_saved_config`: what the saved `config.json` of a checkpoint
of ours says about the tag placement (D10), and the read of that file that only trusts bytes
whose SHA-256 was taken before.

`saved_tag_placement` is the inverse of how `train/eole_data.py` writes the prefixes of every
corpus: the direction tag `<2xx>` is in `src_prefix` for placement `src` and in `tgt_prefix`
for placement `tgt` (an origin marker such as `<bt>`, `<ft>` or `<doc>` may sit in
`src_prefix` either way). `require_stated_placement` refuses a statement that disagrees and
names both values. The export (`test_train_distill_export_placement.py`) and `distill
evaluate` use them through the files of a checkpoint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.train.distill_saved_config import (
    require_saved_placement,
    require_stated_placement,
    saved_tag_placement,
)
from tests.support.step_dirs import launcher_data

FILE = Path("/ckpt/step_200/config.json")


def entry(src_prefix: Any, tgt_prefix: Any) -> dict[str, Any]:
    return {"path_src": "/a.src", "path_tgt": "/a.tgt", "src_prefix": src_prefix,
            "tgt_prefix": tgt_prefix}  # fmt: skip


@pytest.mark.parametrize(
    ("src_prefix", "tgt_prefix", "expected"),
    [
        ("<2khk_Cyrl>", "", "src"),
        ("<2khk_Cyrl> <bt>", "", "src"),
        ("<2khk_Cyrl> <ft>", "", "src"),
        ("<2khk_Cyrl> <doc>", "", "src"),
        ("<2eng_Latn>", None, "src"),
        ("", "<2khk_Cyrl>", "tgt"),
        ("<bt>", "<2khk_Cyrl>", "tgt"),
        ("<ft>", "<2rus_Cyrl>", "tgt"),
        (None, "<2khk_Cyrl>", "tgt"),
        ("", "", None),
        ("<bt>", "", None),
        ("<doc>", "<ft>", None),
        (None, None, None),
    ],
)
def test_one_corpus_places_the_tag_where_its_prefix_holds_it(
    src_prefix: str | None, tgt_prefix: str | None, expected: str | None
) -> None:
    config = {"data": {"corpus": entry(src_prefix, tgt_prefix)}}

    assert saved_tag_placement(config, FILE) == expected


@pytest.mark.parametrize(
    "config",
    [{}, {"data": None}, {"data": {}}, {"model": {"hidden_size": 64}}],
    ids=["no key", "null", "empty", "other keys"],
)
def test_a_config_without_corpora_places_nothing(config: dict[str, Any]) -> None:
    assert saved_tag_placement(config, FILE) is None


def test_the_launchers_data_places_the_tag_for_either_placement() -> None:
    assert saved_tag_placement({"data": launcher_data("src")}, FILE) == "src"
    assert saved_tag_placement({"data": launcher_data("tgt")}, FILE) == "tgt"


def test_a_tag_is_a_whole_word_of_the_form_2_and_a_language_code() -> None:
    # `<2` starts a tag; a marker that merely contains the letters, or text that is not
    # delimited, is not one.
    for prefix in ("<bt>", "<doc>", "2khk_Cyrl", "<22>x", "x<2khk_Cyrl>", "<2>"):
        config = {"data": {"corpus": entry(prefix, "")}}
        assert saved_tag_placement(config, FILE) is None, prefix


@pytest.mark.parametrize(
    "data",
    [[], "src", 7, {"corpus": "<2khk_Cyrl>"}, {"corpus": ["<2khk_Cyrl>"]}],
)
def test_data_that_is_not_a_mapping_of_mappings_is_a_data_error(data: Any) -> None:
    with pytest.raises(DataError, match="config.json"):
        saved_tag_placement({"data": data}, FILE)


@pytest.mark.parametrize("prefix", [7, ["<2khk_Cyrl>"], {"tag": "<2khk_Cyrl>"}, True])
@pytest.mark.parametrize("side", ["src_prefix", "tgt_prefix"])
def test_a_prefix_that_is_not_text_is_a_data_error_naming_the_key(
    side: str, prefix: Any
) -> None:
    corpus = entry("<2khk_Cyrl>", "")
    corpus[side] = prefix

    with pytest.raises(DataError, match=f"data.corpus.{side}"):
        saved_tag_placement({"data": {"corpus": corpus}}, FILE)


def test_a_corpus_with_a_tag_on_both_sides_is_a_data_error_naming_it() -> None:
    config = {"data": {"fine": entry("<2khk_Cyrl>", ""), "odd": entry("<2a>", "<2b>")}}

    with pytest.raises(DataError, match="data.odd") as raised:
        saved_tag_placement(config, FILE)

    assert "both" in str(raised.value)


def test_corpora_that_disagree_are_a_data_error_naming_the_first_of_each_side() -> None:
    config = {
        "data": {"a": entry("<2x>", ""), "b": entry("", "<2x>"), "c": entry("<2y>", "")}
    }

    with pytest.raises(DataError) as raised:
        saved_tag_placement(config, FILE)

    message = str(raised.value)
    assert "data.a" in message
    assert "data.b" in message
    assert "data.c" not in message


# ---- require_stated_placement ----------------------------------------------------------


def test_a_statement_that_agrees_or_cannot_be_checked_is_accepted() -> None:
    require_stated_placement({"data": launcher_data("src")}, FILE, "src")
    require_stated_placement({"data": launcher_data("tgt")}, FILE, "tgt")
    require_stated_placement({}, FILE, "src")
    require_stated_placement({}, FILE, "tgt")
    require_stated_placement({"data": {"c": entry("", "")}}, FILE, "tgt")


def test_a_statement_that_disagrees_is_a_config_error_naming_both_values_and_the_key() -> (
    None
):
    config = {"data": {"corpus": entry("<2khk_Cyrl> <bt>", "")}}

    with pytest.raises(ConfigError) as raised:
        require_stated_placement(config, FILE, "tgt")

    message = str(raised.value)
    assert "'tgt'" in message
    assert "'src'" in message
    assert "data.corpus.src_prefix" in message
    assert "'<2khk_Cyrl> <bt>'" in message
    assert str(FILE) in message
    assert "--tag-placement src" in message


def test_the_target_side_key_is_named_when_the_tag_is_a_target_prefix() -> None:
    config = {"data": {"corpus": entry("<bt>", "<2khk_Cyrl>")}}

    with pytest.raises(ConfigError) as raised:
        require_stated_placement(config, FILE, "src")

    message = str(raised.value)
    assert "data.corpus.tgt_prefix" in message
    assert "'<2khk_Cyrl>'" in message
    assert "--tag-placement tgt" in message


# ---- require_saved_placement: the bytes that were hashed are the bytes that are read --------


def write_config(directory: Path, text: str) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "config.json"
    file.write_text(text, encoding="utf-8")
    return file, hashlib.sha256(file.read_bytes()).hexdigest()


def test_the_saved_config_is_read_once_and_only_when_it_is_the_hashed_file(
    tmp_path: Path,
) -> None:
    file, digest = write_config(
        tmp_path, '{"data": {"c": {"src_prefix": "<2khk_Cyrl>", "tgt_prefix": ""}}}'
    )

    require_saved_placement(file, digest, "src")
    with pytest.raises(ConfigError, match="'tgt'.*'src'"):
        require_saved_placement(file, digest, "tgt")
    file.write_text('{"data": {"c": {"src_prefix": "", "tgt_prefix": "<2x>"}}}')
    with pytest.raises(ConfigError, match="changed after it was hashed"):
        require_saved_placement(file, digest, "src")


def test_a_saved_config_that_cannot_be_read_or_parsed_is_a_data_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "gone" / "config.json"
    with pytest.raises(DataError, match="cannot read"):
        require_saved_placement(missing, "0" * 64, "src")
    for text in ("not json", "[1, 2]"):
        file, digest = write_config(tmp_path / "bad", text)
        with pytest.raises(DataError, match="config.json"):
            require_saved_placement(file, digest, "src")
