"""Tests for the record of a model export (`ModelRecord`, `rinchen.json`, schema
`rinchen.model/1`): the document that says what an export directory is.

`to_bytes` writes one exact text (sorted keys, two-space indent, no timestamp: the same
inputs give the same bytes) and `from_bytes` reads back exactly that document: this schema,
every field and no other, each of its type, directions and languages and tags that agree
with one another. A refusal is a `DataError` that names the field and never quotes what it
held (the file may come from anywhere).
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import rinchen
from rinchen.errors import ConfigError, DataError
from rinchen.export import record as record_module
from rinchen.export.record import (
    CT2_FILES,
    EXPORT_FILES,
    EXPORT_STAGE,
    LEGACY_RECORD_FILE,
    RECORD_FILE,
    RECORD_SCHEMA,
    SPM_FILE,
    CheckpointRecord,
    ModelRecord,
    default_name,
    languages_of,
    require_directions,
    require_name,
    targets_of,
)
from rinchen.train import checkpoints
from tests.support.model_exports import checkpoint_record

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
CANARY = "CANARY-7f3a"


def a_record(
    *,
    checkpoint: CheckpointRecord | None = None,
    placement: Any = "src",
    directions: tuple[str, ...] = ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"),
    document_level: bool = False,
    quantization: str = "int8",
) -> ModelRecord:
    return ModelRecord(
        name="rinchen-abababababab",
        rinchen_version="0.0.0",
        tag_placement=placement,
        directions=directions,
        languages=languages_of(directions),
        tags={
            "eng_Latn": "<2eng_Latn>",
            "khk_Cyrl": "<2khk_Cyrl>",
        }
        if directions == ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn")
        else {target: f"<2{target}>" for target in targets_of(directions)},
        document_level=document_level,
        quantization=quantization,
        checkpoint=checkpoint
        or CheckpointRecord(
            kind="step",
            step=2000,
            weights_sha256=DIGEST_A,
            config_sha256=DIGEST_B,
            vocab_sha256=DIGEST_C,
            manifest_sha256=DIGEST_D,
            manifest_stage="train_launch",
        ),
        spm_sha256=DIGEST_E,
        encoder_layers=2,
        decoder_layers=1,
        hidden_size=64,
        vocabulary_entries=344,
    )


def document_of(record: ModelRecord) -> dict[str, Any]:
    document = json.loads(record.to_bytes())
    assert isinstance(document, dict)
    return document


def encoded(document: object) -> bytes:
    return json.dumps(document).encode("utf-8")


def refuses_to_read(content: bytes) -> str:
    with pytest.raises(DataError) as raised:
        ModelRecord.from_bytes(content)
    return str(raised.value)


def refuses_the_document(document: object) -> str:
    return refuses_to_read(encoded(document))


# ---- the constants ----------------------------------------------------------------------


def test_the_names_of_the_export_are_the_agreed_ones() -> None:
    assert EXPORT_STAGE == "export"
    assert RECORD_FILE == "rinchen.json"
    assert LEGACY_RECORD_FILE == "export.json"
    assert SPM_FILE == "spm.model"
    assert RECORD_SCHEMA == "rinchen.model/1"
    assert CT2_FILES == ("model.bin", "shared_vocabulary.json", "config.json")
    assert EXPORT_FILES == (
        "model.bin",
        "shared_vocabulary.json",
        "config.json",
        "spm.model",
        "rinchen.json",
    )


def test_the_kind_of_a_continuation_is_defined_once_and_the_checkpoints_use_it() -> (
    None
):
    defining: list[str] = []
    for source in sorted(Path(rinchen.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ContinuationKind"
                for target in node.targets
            ):
                defining.append(source.name)

    assert defining == ["record.py"]
    assert checkpoints.ContinuationKind is record_module.ContinuationKind


# ---- the document -----------------------------------------------------------------------


def test_a_record_is_written_as_sorted_two_space_indented_json_with_a_final_line_feed() -> (
    None
):
    expected = (
        "{\n"
        '  "checkpoint": {\n'
        f'    "config_sha256": "{DIGEST_B}",\n'
        '    "kind": "step",\n'
        f'    "manifest_sha256": "{DIGEST_D}",\n'
        '    "manifest_stage": "train_launch",\n'
        '    "step": 2000,\n'
        f'    "vocab_sha256": "{DIGEST_C}",\n'
        f'    "weights_sha256": "{DIGEST_A}"\n'
        "  },\n"
        '  "decoder_layers": 1,\n'
        '  "directions": [\n'
        '    "eng_Latn-khk_Cyrl",\n'
        '    "khk_Cyrl-eng_Latn"\n'
        "  ],\n"
        '  "document_level": false,\n'
        '  "encoder_layers": 2,\n'
        '  "hidden_size": 64,\n'
        '  "languages": [\n'
        '    "eng_Latn",\n'
        '    "khk_Cyrl"\n'
        "  ],\n"
        '  "name": "rinchen-abababababab",\n'
        '  "quantization": "int8",\n'
        '  "rinchen_version": "0.0.0",\n'
        '  "schema": "rinchen.model/1",\n'
        f'  "spm_sha256": "{DIGEST_E}",\n'
        '  "tag_placement": "src",\n'
        '  "tags": {\n'
        '    "eng_Latn": "<2eng_Latn>",\n'
        '    "khk_Cyrl": "<2khk_Cyrl>"\n'
        "  },\n"
        '  "vocabulary_entries": 344\n'
        "}\n"
    )

    assert a_record().to_bytes() == expected.encode("utf-8")


def test_an_averaged_checkpoint_is_written_with_a_null_step() -> None:
    record = a_record(checkpoint=checkpoint_record("averaged"))

    assert b'"kind": "averaged"' in record.to_bytes()
    assert b'"step": null' in record.to_bytes()


def test_the_same_record_gives_the_same_bytes_whatever_order_its_tags_were_given_in() -> (
    None
):
    first = a_record()
    second = dataclasses.replace(
        first, tags={"khk_Cyrl": "<2khk_Cyrl>", "eng_Latn": "<2eng_Latn>"}
    )

    assert first.to_bytes() == second.to_bytes()
    assert first == second


@pytest.mark.parametrize(
    "record",
    [
        a_record(),
        a_record(placement="tgt"),
        a_record(checkpoint=checkpoint_record("averaged")),
        a_record(document_level=True, quantization="float32"),
        a_record(
            directions=("eng_Latn-mn_Mong", "khk_Cyrl-mn_Mong"), quantization="float16"
        ),
        a_record(directions=("khk_Cyrl-eng_Latn",), quantization="int8_float16"),
    ],
    ids=["step", "tgt", "averaged", "document", "mongolian-script", "one-direction"],
)
def test_a_record_reads_back_as_it_was_written(record: ModelRecord) -> None:
    assert ModelRecord.from_bytes(record.to_bytes()) == record


def test_the_targets_of_a_record_are_the_sorted_keys_of_its_tags() -> None:
    record = a_record(directions=("khk_Cyrl-eng_Latn", "khk_Cyrl-mn_Mong"))

    assert record.targets == ("eng_Latn", "mn_Mong")


def test_a_record_cannot_be_changed_and_its_tags_are_read_only() -> None:
    record = a_record()

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.tags["xxx_Latn"] = "<2xxx_Latn>"  # type: ignore[index]


# ---- a document that is not a record --------------------------------------------------------


def test_bytes_that_are_not_utf_8_are_not_a_record() -> None:
    assert "UTF-8" in refuses_to_read(b"\xff\xfe{}")


def test_text_that_is_not_json_is_not_a_record() -> None:
    assert "not valid JSON" in refuses_to_read(b'{"schema": ')


def test_json_that_is_not_an_object_is_not_a_record() -> None:
    assert "not a JSON object" in refuses_to_read(b"[1, 2]")


@pytest.mark.parametrize(
    "schema",
    ["rinchen.distill_export/1", "rinchen.model/2", "rinchen.model", "", None, 1],
)
def test_a_document_with_another_schema_is_not_a_record(schema: object) -> None:
    document = document_of(a_record())
    document["schema"] = schema

    message = refuses_the_document(document)

    assert "'schema'" in message
    assert "rinchen.model/1" in message


def test_a_document_without_a_schema_is_not_a_record() -> None:
    document = document_of(a_record())
    del document["schema"]

    assert "'schema'" in refuses_the_document(document)


def test_the_old_record_of_the_distillation_is_not_a_record_of_a_model() -> None:
    old = {
        "schema": "rinchen.distill_export/1",
        "tag_placement": "src",
        "checkpoint": document_of(a_record())["checkpoint"],
        "spm_sha256": DIGEST_E,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "hidden_size": 64,
        "vocabulary_entries": 344,
    }

    assert "'schema' must be 'rinchen.model/1'" in refuses_the_document(old)


def test_a_document_with_a_key_twice_is_not_a_record() -> None:
    text = a_record().to_bytes().decode("utf-8")
    twice = text.replace(
        '"hidden_size": 64,', '"hidden_size": 64,\n  "hidden_size": 128,'
    )
    assert twice != text

    assert "'hidden_size' twice" in refuses_to_read(twice.encode("utf-8"))


TOP_LEVEL_KEYS = (
    "name",
    "rinchen_version",
    "tag_placement",
    "directions",
    "languages",
    "tags",
    "document_level",
    "quantization",
    "checkpoint",
    "spm_sha256",
    "encoder_layers",
    "decoder_layers",
    "hidden_size",
    "vocabulary_entries",
)
CHECKPOINT_KEYS = (
    "kind",
    "step",
    "weights_sha256",
    "config_sha256",
    "vocab_sha256",
    "manifest_sha256",
    "manifest_stage",
)


def test_the_fields_the_tests_name_are_all_the_fields_a_record_has() -> None:
    assert sorted((*TOP_LEVEL_KEYS, "schema")) == sorted(document_of(a_record()))
    assert sorted(CHECKPOINT_KEYS) == sorted(document_of(a_record())["checkpoint"])


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_a_record_with_a_field_missing_is_refused_naming_it(key: str) -> None:
    document = document_of(a_record())
    del document[key]

    assert f"'{key}' is missing" in refuses_the_document(document)


@pytest.mark.parametrize("key", CHECKPOINT_KEYS)
def test_a_record_with_a_checkpoint_field_missing_is_refused_naming_it(
    key: str,
) -> None:
    document = document_of(a_record())
    del document["checkpoint"][key]

    assert f"'checkpoint.{key}' is missing" in refuses_the_document(document)


def test_a_record_with_an_extra_field_is_refused_naming_it() -> None:
    document = document_of(a_record())
    document["created"] = "2026-11-09"

    assert "'created'" in refuses_the_document(document)


def test_a_record_with_an_extra_checkpoint_field_is_refused_naming_it() -> None:
    document = document_of(a_record())
    document["checkpoint"]["optimizer_sha256"] = DIGEST_A

    assert "'checkpoint.optimizer_sha256'" in refuses_the_document(document)


def test_a_checkpoint_that_is_not_an_object_is_refused() -> None:
    document = document_of(a_record())
    document["checkpoint"] = "step_2000"

    assert "'checkpoint' must be an object" in refuses_the_document(document)


# ---- the fields of text ---------------------------------------------------------------------


@pytest.mark.parametrize("placement", ["up", "both", "", 1, None, "SRC"])
def test_a_placement_other_than_src_or_tgt_is_refused(placement: object) -> None:
    document = document_of(a_record())
    document["tag_placement"] = placement

    assert "'tag_placement' must be 'src' or 'tgt'" in refuses_the_document(document)


@pytest.mark.parametrize("value", ["int4", "int8_float32", "INT8", "", None, 8])
def test_a_quantization_the_export_does_not_offer_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["quantization"] = value

    message = refuses_the_document(document)

    assert "'quantization' must be one of" in message
    assert "float32, float16, int8, int8_float16" in message


@pytest.mark.parametrize(
    "value",
    [
        "",
        "has space",
        "x" * 65,
        "a/b",
        "line\nbreak",
        "caf" + chr(0xE9),
        7,
        None,
        ["x"],
    ],
    ids=["empty", "space", "65", "slash", "newline", "accent", "int", "none", "list"],
)
def test_a_name_that_is_not_one_to_64_plain_characters_is_refused(
    value: object,
) -> None:
    document = document_of(a_record())
    document["name"] = value

    assert "'name' must be 1 to 64" in refuses_the_document(document)


@pytest.mark.parametrize("value", ["x", "a.b_c-D9", "x" * 64])
def test_a_name_of_plain_characters_is_accepted(value: str) -> None:
    document = document_of(a_record())
    document["name"] = value

    assert ModelRecord.from_bytes(encoded(document)).name == value


@pytest.mark.parametrize("value", ["", "1 2", "x" * 65, 0, None, "1.0\n"])
def test_a_version_that_is_not_a_package_version_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["rinchen_version"] = value

    assert "'rinchen_version' must be" in refuses_the_document(document)


@pytest.mark.parametrize("value", ["0.0.0", "1.2.3+local", "2026.11.0.dev3", "unknown"])
def test_a_package_version_is_accepted(value: str) -> None:
    document = document_of(a_record())
    document["rinchen_version"] = value

    assert ModelRecord.from_bytes(encoded(document)).rinchen_version == value


@pytest.mark.parametrize(
    "value", ["abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, 5, None]
)
def test_a_sentencepiece_digest_that_is_not_a_sha256_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["spm_sha256"] = value

    assert "'spm_sha256' must be a SHA-256" in refuses_the_document(document)


# ---- the counts and the flag ----------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["encoder_layers", "decoder_layers", "hidden_size", "vocabulary_entries"]
)
@pytest.mark.parametrize("value", ["2", True, 2.0, None, -1, 0])
def test_a_count_that_is_not_a_positive_integer_is_refused_naming_the_field(
    key: str, value: object
) -> None:
    document = document_of(a_record())
    document[key] = value

    assert f"'{key}' must be a positive integer" in refuses_the_document(document)


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, [], 1.0])
def test_a_document_level_that_is_not_a_boolean_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["document_level"] = value

    assert "'document_level' must be true or false" in refuses_the_document(document)


# ---- the checkpoint -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["weights_sha256", "config_sha256", "vocab_sha256", "manifest_sha256"]
)
@pytest.mark.parametrize(
    "value", ["abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n", 5, None]
)
def test_a_checkpoint_digest_that_is_not_a_sha256_is_refused_naming_the_field(
    key: str, value: object
) -> None:
    document = document_of(a_record())
    document["checkpoint"][key] = value

    assert f"'checkpoint.{key}' must be a SHA-256" in refuses_the_document(document)


@pytest.mark.parametrize("value", ["both", 1, None])
def test_a_checkpoint_kind_other_than_step_or_averaged_is_refused(
    value: object,
) -> None:
    document = document_of(a_record())
    document["checkpoint"]["kind"] = value

    assert "'checkpoint.kind' must be 'step' or 'averaged'" in refuses_the_document(
        document
    )


@pytest.mark.parametrize("value", [None, "2000", True, 0, -3, 1.5])
def test_a_step_checkpoint_needs_a_positive_step(value: object) -> None:
    document = document_of(a_record())
    document["checkpoint"]["step"] = value

    assert "'checkpoint.step' must be a positive integer" in refuses_the_document(
        document
    )


def test_an_averaged_checkpoint_has_no_step() -> None:
    document = document_of(a_record(checkpoint=checkpoint_record("averaged")))
    document["checkpoint"]["step"] = 200

    assert "'checkpoint.step' must be null" in refuses_the_document(document)


def test_a_manifest_stage_that_is_not_text_is_refused() -> None:
    document = document_of(a_record())
    document["checkpoint"]["manifest_stage"] = 7

    assert "'checkpoint.manifest_stage' must be text" in refuses_the_document(document)


# ---- directions, languages and tags ---------------------------------------------------------


@pytest.mark.parametrize(
    ("directions", "problem"),
    [
        ([], "at least one direction"),
        (["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"], "sorted and without repeats"),
        (["eng_Latn-khk_Cyrl", "eng_Latn-khk_Cyrl"], "sorted and without repeats"),
        (["eng_Latn-eng_Latn"], "from a language to itself"),
        (["eng_Latn-xxx_Latn"], "not in the language registry"),
        (["xxx_Latn-eng_Latn"], "not in the language registry"),
        (["eng_Latn-khk_Cyrl-rus_Cyrl"], "not <source>-<target>"),
        (["eng_Latn"], "not <source>-<target>"),
        (["eng_Latn-"], "not <source>-<target>"),
        ([""], "not <source>-<target>"),
        (["eng_Latn-khk_Cyrl", 7], "must be a list of text"),
        ("eng_Latn-khk_Cyrl", "must be a list of text"),
        ({"eng_Latn-khk_Cyrl": 1}, "must be a list of text"),
        (None, "must be a list of text"),
    ],
    ids=[
        "empty",
        "unsorted",
        "repeated",
        "same-language",
        "unknown-target",
        "unknown-source",
        "three-parts",
        "no-target",
        "empty-target",
        "empty-entry",
        "not-text",
        "a-string",
        "an-object",
        "null",
    ],
)
def test_directions_that_are_not_sorted_distinct_registry_pairs_are_refused(
    directions: object, problem: str
) -> None:
    document = document_of(a_record())
    document["directions"] = directions

    message = refuses_the_document(document)

    assert "'directions'" in message
    assert problem in message
    assert "xxx_Latn" not in message


@pytest.mark.parametrize(
    "languages",
    [
        ["eng_Latn"],
        ["eng_Latn", "khk_Cyrl", "rus_Cyrl"],
        ["khk_Cyrl", "eng_Latn"],
        ["eng_Latn", "eng_Latn", "khk_Cyrl"],
        [],
        "eng_Latn",
        None,
        ["eng_Latn", 7],
    ],
    ids=["missing", "extra", "unsorted", "repeated", "empty", "string", "null", "int"],
)
def test_languages_that_are_not_the_union_of_the_directions_are_refused(
    languages: object,
) -> None:
    document = document_of(a_record())
    document["languages"] = languages

    assert "'languages' must be" in refuses_the_document(document)


@pytest.mark.parametrize(
    "tags",
    [
        {"khk_Cyrl": "<2khk_Cyrl>"},
        {
            "eng_Latn": "<2eng_Latn>",
            "khk_Cyrl": "<2khk_Cyrl>",
            "rus_Cyrl": "<2rus_Cyrl>",
        },
        {"eng_Latn": "<2eng_Latn>", "mn_Mong": "<2mn_Mong>"},
        {},
    ],
    ids=["a-target-missing", "a-target-too-many", "another-target", "empty"],
)
def test_tags_whose_keys_are_not_the_targets_of_the_directions_are_refused(
    tags: object,
) -> None:
    document = document_of(a_record())
    document["tags"] = tags

    assert "'tags' must have one entry for each target" in refuses_the_document(
        document
    )


@pytest.mark.parametrize("tags", ["<2khk_Cyrl>", ["<2khk_Cyrl>"], None, 3])
def test_tags_that_are_not_an_object_are_refused(tags: object) -> None:
    document = document_of(a_record())
    document["tags"] = tags

    assert "'tags' must be an object of text" in refuses_the_document(document)


@pytest.mark.parametrize("value", [7, None, ["<2khk_Cyrl>"], ""])
def test_a_tag_that_is_not_text_is_refused(value: object) -> None:
    document = document_of(a_record())
    document["tags"]["khk_Cyrl"] = value

    assert "'tags' must be an object of text" in refuses_the_document(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", f"{CANARY} with spaces"),
        ("rinchen_version", f"{CANARY} 1"),
        ("tag_placement", CANARY),
        ("quantization", CANARY),
        ("spm_sha256", CANARY),
        ("document_level", CANARY),
        ("encoder_layers", CANARY),
        ("directions", [CANARY]),
        ("directions", [f"{CANARY}-khk_Cyrl"]),
        ("directions", [f"eng_Latn-{CANARY}"]),
        ("languages", [CANARY]),
        ("tags", {CANARY: CANARY}),
        ("tags", CANARY),
        ("checkpoint", CANARY),
        ("schema", CANARY),
    ],
)
def test_a_refusal_names_the_field_and_never_quotes_what_it_held(
    field: str, value: object
) -> None:
    document = document_of(a_record())
    document[field] = value

    assert CANARY not in refuses_the_document(document)


def test_an_unknown_field_is_named_but_the_text_after_it_is_not_quoted() -> None:
    document = document_of(a_record())
    document["extra_field"] = CANARY

    message = refuses_the_document(document)

    assert "'extra_field'" in message
    assert CANARY not in message


@pytest.mark.parametrize("key", ["weights_sha256", "kind", "step"])
def test_a_checkpoint_refusal_never_quotes_what_it_held(key: str) -> None:
    document = document_of(a_record())
    document["checkpoint"][key] = CANARY

    assert CANARY not in refuses_the_document(document)


# ---- the helpers ----------------------------------------------------------------------------


def test_the_default_name_is_rinchen_and_the_first_12_hex_digits_of_the_model() -> None:
    assert default_name("ab" * 32) == "rinchen-abababababab"
    assert default_name("0123456789abcdef" * 4) == "rinchen-0123456789ab"


@pytest.mark.parametrize("value", ["m", "rinchen-abababababab", "a.b_c-D9", "x" * 64])
def test_a_usable_name_is_returned_as_it_is(value: str) -> None:
    assert require_name(value) == value


@pytest.mark.parametrize(
    "value", ["", " ", "has space", "x" * 65, "a/b", "a\nb", chr(0xE9)]
)
def test_an_unusable_name_is_a_config_error_that_says_what_is_usable(
    value: str,
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_name(value)

    assert "1 to 64" in str(raised.value)


def test_the_languages_and_the_targets_of_directions_are_sorted_and_distinct() -> None:
    directions = ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn", "khk_Cyrl-rus_Cyrl")

    assert languages_of(directions) == ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
    assert targets_of(directions) == ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
    assert targets_of(("khk_Cyrl-eng_Latn", "rus_Cyrl-eng_Latn")) == ("eng_Latn",)
    assert languages_of(("khk_Cyrl-eng_Latn", "rus_Cyrl-eng_Latn")) == (
        "eng_Latn",
        "khk_Cyrl",
        "rus_Cyrl",
    )


def test_good_directions_are_returned_as_a_tuple() -> None:
    assert require_directions(["eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"]) == (
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
    )


@pytest.mark.parametrize(
    "directions",
    [
        [],
        ["b-a"],
        ["eng_Latn-eng_Latn"],
        ["eng_Latn-xxx_Latn"],
        ["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"],
    ],
)
def test_bad_directions_are_a_config_error_that_shows_what_was_given(
    directions: list[str],
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_directions(directions)

    assert "'directions'" in str(raised.value)
    assert repr(directions) in str(raised.value)
