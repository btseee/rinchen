"""Tests for what a reader of a verified export takes from its files
(`rinchen.export.verified`): `verified_bytes`, `shared_vocabulary`, `end_token`,
`verified_content` and `json_object`.

A reader that opens a file of the export after `verify_export` hashed it reads its bytes
ONCE and refuses them unless they are the bytes that were hashed, and parses those bytes:
what is parsed is what was vouched for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.export.verified import (
    VerifiedExport,
    end_token,
    json_object,
    shared_vocabulary,
    string_list,
    verified_bytes,
    verified_content,
    verify_export,
    vocabulary_entries,
)
from tests.support.distill_exports import sha256_of_bytes
from tests.support.model_exports import write_model_export
from tests.support.step_dirs import SOURCE_VOCABULARY


def verified_with(tmp_path: Path, **options: object) -> VerifiedExport:
    return verify_export(write_model_export(tmp_path / "export", **options).directory)  # type: ignore[arg-type]


# ---- verified_bytes -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["model.bin", "shared_vocabulary.json", "config.json", "spm.model", "rinchen.json"],
)
def test_the_bytes_of_a_file_are_returned_when_they_are_the_ones_that_were_hashed(
    tmp_path: Path, name: str
) -> None:
    export = verified_with(tmp_path)

    assert verified_bytes(export, name) == (export.directory / name).read_bytes()


def test_a_file_that_changed_after_the_export_was_verified_is_refused(
    tmp_path: Path,
) -> None:
    export = verified_with(tmp_path)
    (export.directory / "config.json").write_bytes(b"{}")

    with pytest.raises(ConfigError) as raised:
        verified_bytes(export, "config.json")

    message = str(raised.value)
    assert (
        f"{export.directory / 'config.json'} changed after the export was verified"
        in (message)
    )


def test_a_file_that_cannot_be_read_is_a_data_error_naming_it(tmp_path: Path) -> None:
    export = verified_with(tmp_path)
    (export.directory / "spm.model").unlink()

    with pytest.raises(DataError, match=r"cannot read .*spm\.model"):
        verified_bytes(export, "spm.model")


# ---- shared_vocabulary ----------------------------------------------------------------


def test_the_vocabulary_is_the_set_of_the_entries_of_the_converters_list(
    tmp_path: Path,
) -> None:
    export = verified_with(tmp_path)

    assert shared_vocabulary(export) == frozenset(SOURCE_VOCABULARY)


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"a": 1}', b'["a", 2]', b"[]", b""],
    ids=["not json", "object", "non-string", "empty list", "empty file"],
)
def test_a_vocabulary_that_is_not_a_non_empty_list_of_strings_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    export = verified_with(tmp_path, vocabulary_bytes=content)

    with pytest.raises(DataError, match=r"shared_vocabulary\.json"):
        shared_vocabulary(export)


def test_a_vocabulary_that_changed_after_the_export_was_verified_is_refused(
    tmp_path: Path,
) -> None:
    export = verified_with(tmp_path)
    (export.directory / "shared_vocabulary.json").write_bytes(
        json.dumps(["x"]).encode()
    )

    with pytest.raises(ConfigError, match=r"shared_vocabulary\.json changed after"):
        shared_vocabulary(export)


# ---- vocabulary_entries ---------------------------------------------------------------


def test_the_entries_of_the_vocabulary_are_in_the_order_of_the_file(
    tmp_path: Path,
) -> None:
    tokens = ("<unk>", "zeta", "alpha", "mid", "</s>")
    export = verified_with(tmp_path, tokens=tokens)

    assert vocabulary_entries(export) == tokens
    assert shared_vocabulary(export) == frozenset(tokens)


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"a": 1}', b'["a", 2]', b"[]", b""],
    ids=["not json", "object", "non-string", "empty list", "empty file"],
)
def test_the_entries_of_a_vocabulary_that_is_no_list_of_strings_are_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    export = verified_with(tmp_path, vocabulary_bytes=content)

    with pytest.raises(DataError, match=r"shared_vocabulary\.json"):
        vocabulary_entries(export)


def test_the_entries_of_a_vocabulary_that_changed_after_the_verification_are_refused(
    tmp_path: Path,
) -> None:
    export = verified_with(tmp_path)
    (export.directory / "shared_vocabulary.json").write_bytes(
        json.dumps(["x"]).encode()
    )

    with pytest.raises(ConfigError, match=r"shared_vocabulary\.json changed after"):
        vocabulary_entries(export)


# ---- end_token ------------------------------------------------------------------------


def test_the_end_token_is_the_one_in_the_converters_config(tmp_path: Path) -> None:
    export = verified_with(tmp_path)

    assert end_token(export) == "</s>"


def rewritten_config(tmp_path: Path, content: bytes) -> VerifiedExport:
    """An export whose manifest vouches for `config.json` holding `content`."""
    directory = write_model_export(tmp_path / "export").directory
    (directory / "config.json").write_bytes(content)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["outputs"]:
        if item["path"].endswith("config.json"):
            item["sha256"] = sha256_of_bytes(content)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return verify_export(directory)


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"[]",
        b"{}",
        json.dumps({"eos_token": ""}).encode(),
        json.dumps({"eos_token": 3}).encode(),
        json.dumps({"eos_token": None}).encode(),
    ],
    ids=["not json", "list", "no key", "empty", "number", "null"],
)
def test_a_config_without_a_usable_end_token_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    export = rewritten_config(tmp_path, content)

    with pytest.raises(DataError, match=r"config\.json"):
        end_token(export)


def test_an_end_token_other_than_the_default_is_the_one_returned(
    tmp_path: Path,
) -> None:
    export = rewritten_config(tmp_path, json.dumps({"eos_token": "<end>"}).encode())

    assert end_token(export) == "<end>"


def test_a_config_that_changed_after_the_export_was_verified_is_refused(
    tmp_path: Path,
) -> None:
    export = verified_with(tmp_path)
    (export.directory / "config.json").write_bytes(
        json.dumps({"eos_token": "x"}).encode()
    )

    with pytest.raises(ConfigError, match=r"config\.json changed after"):
        end_token(export)


# ---- verified_content -----------------------------------------------------------------


def test_the_content_of_a_file_is_returned_when_it_has_the_digest(
    tmp_path: Path,
) -> None:
    file = tmp_path / "a.txt"
    file.write_bytes(b"hello")

    assert verified_content(file, sha256_of_bytes(b"hello")) == b"hello"


def test_a_file_that_does_not_have_the_digest_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    file = tmp_path / "a.txt"
    file.write_bytes(b"hello")

    with pytest.raises(ConfigError) as raised:
        verified_content(file, sha256_of_bytes(b"other"))

    assert f"{file} changed after it was hashed" in str(raised.value)


def test_a_file_that_cannot_be_read_is_a_data_error(tmp_path: Path) -> None:
    file = tmp_path / "missing.txt"

    with pytest.raises(DataError, match="cannot read"):
        verified_content(file, sha256_of_bytes(b""))


def test_a_directory_in_place_of_a_file_is_a_data_error(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    os.mkdir(folder)

    with pytest.raises(DataError, match="cannot read"):
        verified_content(folder, sha256_of_bytes(b""))


# ---- json_object and string_list ------------------------------------------------------


def test_a_json_object_is_parsed(tmp_path: Path) -> None:
    assert json_object(b'{"a": [1, 2]}', tmp_path / "x.json") == {"a": [1, 2]}


@pytest.mark.parametrize(
    ("content", "says"),
    [(b"SECRET{", "is not valid JSON"), (b'["SECRET"]', "is not a JSON object")],
)
def test_bytes_that_are_not_a_json_object_are_a_data_error_that_never_quotes_them(
    tmp_path: Path, content: bytes, says: str
) -> None:
    file = tmp_path / "x.json"

    with pytest.raises(DataError) as raised:
        json_object(content, file)

    assert str(raised.value) == f"{file}: {says}"


def test_deeply_nested_json_is_not_valid_and_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="is not valid JSON"):
        json_object(b"[" * 100_000, tmp_path / "x.json")


@pytest.mark.parametrize(
    ("content", "entries"),
    [
        (b'["a", "b"]', ["a", "b"]),
        (b"[]", []),
        (b'["a", 2]', None),
        (b'{"a": 1}', None),
        (b"not json", None),
        (b"[" * 100_000, None),
        (b"", None),
    ],
)
def test_a_list_of_strings_is_parsed_and_anything_else_is_none(
    content: bytes, entries: list[str] | None
) -> None:
    assert string_list(content) == entries
