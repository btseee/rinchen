# tests/unit/test_config_languages.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import rinchen
from rinchen.config import languages as languages_module
from rinchen.config.languages import (
    LANGUAGES_PATH,
    PACKAGED_LANGUAGES_PATH,
    default_languages_path,
    direction_tag,
    known_language_codes,
    load_languages,
)
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

SPEC_CODES = [
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
]

NON_PARTNER_CODES = {"khk_Cyrl", "mn_Mong"}

MISSING_FIELD_YAML = """
languages:
  - code: eng_Latn
    scripts: [Latin]
"""

EXTRA_FIELD_YAML = """
languages:
  - code: eng_Latn
    scripts: [Latin]
    direction_tag: "<2eng_Latn>"
    unexpected_field: 1
"""

UNKNOWN_SCRIPT_YAML = """
languages:
  - code: ell_Grek
    scripts: [Greek]
    direction_tag: "<2ell_Grek>"
"""

EMPTY_SCRIPTS_YAML = """
languages:
  - code: eng_Latn
    scripts: []
    direction_tag: "<2eng_Latn>"
"""

DUPLICATE_CODE_YAML = """
languages:
  - code: eng_Latn
    scripts: [Latin]
    direction_tag: "<2eng_Latn>"
  - code: eng_Latn
    scripts: [Latin]
    direction_tag: "<2eng_Latn>"
"""

WRONG_DIRECTION_TAG_YAML = """
languages:
  - code: eng_Latn
    scripts: [Latin]
    direction_tag: "<2en>"
"""

TOP_LEVEL_LIST_YAML = "- code: eng_Latn\n- code: rus_Cyrl\n"

LANGUAGES_KEY_MISSING_YAML = "other: []\n"

EMPTY_LANGUAGES_YAML = "languages: []\n"


def test_default_registry_has_ten_entries_in_spec_order():
    entries = load_languages()

    assert [entry.code for entry in entries] == SPEC_CODES


def test_direction_tag_is_2_plus_code():
    for entry in load_languages():
        assert entry.direction_tag == f"<2{entry.code}>"


def test_codes_and_tags_are_unique():
    entries = load_languages()

    assert len({entry.code for entry in entries}) == len(entries)
    assert len({entry.direction_tag for entry in entries}) == len(entries)


def test_sixteen_directions():
    partners = [
        entry.code for entry in load_languages() if entry.code not in NON_PARTNER_CODES
    ]

    assert len(partners) == 8
    assert 2 * len(partners) == 16


def test_known_language_codes_matches_registry():
    assert known_language_codes() == frozenset(entry.code for entry in load_languages())


def test_direction_tag_of_a_code_is_the_registry_tag():
    assert direction_tag("khk_Cyrl") == "<2khk_Cyrl>"
    assert direction_tag("eng_Latn") == "<2eng_Latn>"


def test_every_registry_code_has_its_own_direction_tag():
    for entry in load_languages():
        assert direction_tag(entry.code) == entry.direction_tag


@pytest.mark.parametrize(
    "code", ["xx_Latn", "", "khk_cyrl", " khk_Cyrl", "<2khk_Cyrl>"]
)
def test_direction_tag_of_a_code_outside_the_registry_is_a_config_error(code):
    with pytest.raises(
        ConfigError, match=f"language {code!r} is not in the language registry"
    ):
        direction_tag(code)


def test_jpn_has_three_scripts():
    by_code = {entry.code: entry for entry in load_languages()}

    assert by_code["jpn_Jpan"].scripts == [Script.HAN, Script.HIRAGANA, Script.KATAKANA]


@pytest.mark.parametrize(
    "yaml_text",
    [
        pytest.param(MISSING_FIELD_YAML, id="missing-field"),
        pytest.param(EXTRA_FIELD_YAML, id="extra-field"),
        pytest.param(UNKNOWN_SCRIPT_YAML, id="unknown-script"),
        pytest.param(EMPTY_SCRIPTS_YAML, id="empty-scripts"),
        pytest.param(DUPLICATE_CODE_YAML, id="duplicate-code"),
        pytest.param(WRONG_DIRECTION_TAG_YAML, id="wrong-direction-tag"),
        pytest.param(TOP_LEVEL_LIST_YAML, id="top-level-list"),
        pytest.param(LANGUAGES_KEY_MISSING_YAML, id="languages-key-missing"),
        pytest.param(EMPTY_LANGUAGES_YAML, id="empty-languages"),
    ],
)
def test_malformed_registry_raises_config_error(tmp_path: Path, yaml_text: str):
    path = tmp_path / "languages.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_languages(path)


def test_yaml_syntax_error_raises_config_error_with_cause(tmp_path: Path):
    path = tmp_path / "languages.yaml"
    path.write_text("languages: [\n  - code: x", encoding="utf-8")

    with pytest.raises(ConfigError, match="languages.yaml") as excinfo:
        load_languages(path)

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_non_utf8_registry_raises_config_error_with_cause(tmp_path: Path):
    path = tmp_path / "languages.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="languages.yaml") as excinfo:
        load_languages(path)

    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_missing_registry_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_languages(tmp_path / "nonexistent.yaml")


def test_default_path_points_into_repo_configs():
    assert LANGUAGES_PATH.name == "languages.yaml"
    assert LANGUAGES_PATH.parent.name == "configs"


# --- the registry of the installed package (D11, ruling T14) -------------------------------

ONE_LANGUAGE_YAML = """
languages:
  - code: eng_Latn
    scripts: [Latin]
    direction_tag: "<2eng_Latn>"
"""


def test_the_packaged_registry_is_named_inside_the_package():
    assert PACKAGED_LANGUAGES_PATH == (
        Path(rinchen.__file__).parent / "_data" / "languages.yaml"
    )


def test_the_default_path_is_the_repository_copy_when_it_exists():
    assert LANGUAGES_PATH.is_file()
    assert default_languages_path() == LANGUAGES_PATH


def test_the_default_path_is_the_packaged_copy_when_the_repository_has_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        languages_module, "LANGUAGES_PATH", tmp_path / "configs" / "l.yaml"
    )

    assert default_languages_path() == PACKAGED_LANGUAGES_PATH


def test_the_repository_copy_wins_over_the_packaged_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    packaged = tmp_path / "packaged.yaml"
    packaged.write_text(ONE_LANGUAGE_YAML, encoding="utf-8")
    monkeypatch.setattr(languages_module, "PACKAGED_LANGUAGES_PATH", packaged)

    assert default_languages_path() == LANGUAGES_PATH
    assert len(load_languages()) == len(SPEC_CODES)


def test_load_languages_without_a_path_reads_the_packaged_copy_when_there_is_no_repository_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    packaged = tmp_path / "packaged.yaml"
    packaged.write_text(ONE_LANGUAGE_YAML, encoding="utf-8")
    monkeypatch.setattr(languages_module, "LANGUAGES_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(languages_module, "PACKAGED_LANGUAGES_PATH", packaged)

    assert [entry.code for entry in load_languages()] == ["eng_Latn"]


def test_load_languages_without_a_path_resolves_the_copy_at_each_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    packaged = tmp_path / "packaged.yaml"
    packaged.write_text(ONE_LANGUAGE_YAML, encoding="utf-8")
    before = len(load_languages())
    monkeypatch.setattr(languages_module, "LANGUAGES_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(languages_module, "PACKAGED_LANGUAGES_PATH", packaged)
    during = len(load_languages())
    monkeypatch.undo()

    assert (before, during, len(load_languages())) == (
        len(SPEC_CODES),
        1,
        len(SPEC_CODES),
    )


def test_without_either_copy_the_error_names_the_packaged_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    packaged = tmp_path / "packaged.yaml"
    monkeypatch.setattr(languages_module, "LANGUAGES_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(languages_module, "PACKAGED_LANGUAGES_PATH", packaged)

    with pytest.raises(ConfigError, match="not found") as excinfo:
        load_languages()

    assert str(packaged) in str(excinfo.value)


def test_an_explicit_path_is_still_read_whatever_the_default_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(ONE_LANGUAGE_YAML, encoding="utf-8")

    assert [entry.code for entry in load_languages(explicit)] == ["eng_Latn"]
    assert [entry.code for entry in load_languages(path=explicit)] == ["eng_Latn"]
