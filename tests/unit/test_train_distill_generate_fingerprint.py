"""The name of a KD corpus and the fingerprint of a generation (D10): what the fingerprint
is made of (each ingredient changes it; the batch size, the thread count, the selection and
the order of the source frames do not) and the refusals of a name.

The teacher is a fabricated export (`tests.support.distill_teachers.verified_export`); an
ingredient is changed by replacing one digest or option of the verified export."""

from __future__ import annotations

import dataclasses
import re

import pytest

from rinchen.errors import ConfigError
from rinchen.train.distill_generate import kd_fingerprint, kd_name
from rinchen.train.distill_teacher import TeacherOptions
from tests.support.corpus_rows import EN_MN, RU_MN
from tests.support.distill_exports import MODEL_BYTES, sha256_of_bytes
from tests.support.distill_teachers import verified_export
from tests.support.planted_kd import (
    PLANTED_REFERENCES,
    planted_sources,
    sources_of,
)

SEED = 1234
GOLDEN_FINGERPRINT = "66dbb50ad46fa3702ce2efaccd0e683f8fc176675ff1e6448a91f72c1c035bea"
# Digests that belong to no file: the golden depends on the recipe, not on fabricated bytes.
PINNED_DIGESTS = {
    "model.bin": "a" * 64,
    "shared_vocabulary.json": "b" * 64,
    "config.json": "c" * 64,
    "spm.model": "d" * 64,
    "rinchen.json": "e" * 64,
}
SHARD_SIZE = 512


@pytest.fixture
def export(tmp_path):
    return verified_export(tmp_path)


def _fingerprint(
    export,
    *,
    options=None,
    sources=None,
    shard_size=SHARD_SIZE,
    seed=SEED,
):
    return kd_fingerprint(
        export,
        TeacherOptions() if options is None else options,
        planted_sources() if sources is None else sources,
        shard_size=shard_size,
        seed=seed,
    )


def _with_digest(export, name):
    return dataclasses.replace(
        export, sha256={**export.sha256, name: sha256_of_bytes(name.encode())}
    )


# -- the name


def test_the_default_name_is_rinchen_and_the_first_12_hex_of_the_model_digest(export):
    assert sha256_of_bytes(MODEL_BYTES)[:12] == "07148803f29a"
    assert kd_name(export, None) == "rinchen-07148803f29a"


def test_a_given_name_is_kept(export):
    assert kd_name(export, "teacher-v1.2_a") == "teacher-v1.2_a"
    assert kd_name(export, "x" * 64) == "x" * 64


@pytest.mark.parametrize(
    "name",
    ["", "x" * 65, "a b", "a/b", "a\nb", "teacher\n", "m" + chr(0x0430), "a:b"],
)
def test_a_name_outside_the_allowed_characters_or_length_is_refused(export, name):
    with pytest.raises(ConfigError, match="name"):
        kd_name(export, name)


# -- the fingerprint


def test_the_fingerprint_is_64_hex_characters_and_stable(export):
    first = _fingerprint(export)

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert _fingerprint(export) == first


def test_the_fingerprint_of_fixed_inputs_is_a_fixed_value(export):
    # A change of the recipe renames the shard directory of every earlier run, so it must
    # be deliberate: this value is the recipe's golden.
    pinned = dataclasses.replace(export, sha256=PINNED_DIGESTS)

    assert _fingerprint(pinned) == GOLDEN_FINGERPRINT


@pytest.mark.parametrize(
    "name", ["model.bin", "shared_vocabulary.json", "config.json", "spm.model"]
)
def test_the_digest_of_each_file_of_the_teacher_is_in_the_fingerprint(export, name):
    assert _fingerprint(_with_digest(export, name)) != _fingerprint(export)


def test_the_digest_of_the_record_is_not_a_separate_ingredient(export):
    # `rinchen.json` says where the export came from; what the teacher IS is its four files
    # and its tag placement.
    assert _fingerprint(_with_digest(export, "rinchen.json")) == _fingerprint(export)


def test_the_tag_placement_is_in_the_fingerprint(export):
    other = dataclasses.replace(
        export, record=dataclasses.replace(export.record, tag_placement="tgt")
    )

    assert _fingerprint(other) != _fingerprint(export)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("beam_size", 5),
        ("n", 2),
        ("max_decoding_length", 128),
        ("max_source_pieces", 255),
        ("device", "cuda"),
        ("compute_type", "int8"),
    ],
)
def test_each_decoding_option_that_decides_a_hypothesis_is_in_the_fingerprint(
    export, field, value
):
    options = dataclasses.replace(TeacherOptions(), **{field: value})

    assert _fingerprint(export, options=options) != _fingerprint(export)


@pytest.mark.parametrize("field", ["batch_size", "threads"])
def test_the_batch_size_and_the_thread_count_are_not_in_the_fingerprint(export, field):
    options = dataclasses.replace(TeacherOptions(), **{field: 7})

    assert _fingerprint(export, options=options) == _fingerprint(export)


def test_the_shard_size_and_the_seed_are_in_the_fingerprint(export):
    assert _fingerprint(export, shard_size=256) != _fingerprint(export)
    assert _fingerprint(export, seed=SEED + 1) != _fingerprint(export)


def test_one_source_text_is_in_the_fingerprint(export):
    changed = {
        EN_MN: {**PLANTED_REFERENCES[EN_MN], "Water": ["Ус"]},
        RU_MN: PLANTED_REFERENCES[RU_MN],
    }
    references = {
        direction: {
            ("Waters" if text == "Water" else text): targets
            for text, targets in rows.items()
        }
        for direction, rows in changed.items()
    }

    sources = [sources_of(direction, rows) for direction, rows in references.items()]

    assert _fingerprint(export, sources=sources) != _fingerprint(export)


def test_one_source_id_is_in_the_fingerprint(export):
    sources = planted_sources()
    first = sources[0]
    renamed = first.frame.with_columns(
        id=first.frame["id"].str.replace("^29bc", "29bd")
    )
    sources[0] = dataclasses.replace(first, frame=renamed)

    assert _fingerprint(export, sources=sources) != _fingerprint(export)


def test_a_direction_added_changes_the_fingerprint(export):
    sources = planted_sources()
    more = [*sources, sources_of("rus_Cyrl-eng_Latn", {"Привет": ["Hello"]})]

    assert _fingerprint(export, sources=more) != _fingerprint(export)


def test_a_source_moved_to_another_direction_changes_the_fingerprint(export):
    one = [
        sources_of(EN_MN, {"Hello": ["Сайн"]}),
        sources_of(RU_MN, {"Привет": ["Сайн"]}),
    ]
    other = [sources_of(EN_MN, {"Hello": ["Сайн"], "Привет": ["Сайн"]})]

    assert _fingerprint(export, sources=one) != _fingerprint(export, sources=other)


def test_the_target_language_of_a_direction_is_in_the_fingerprint(export):
    # The same rows translated into another language are other shards, whatever the ids say.
    sources = planted_sources()
    sources[1] = dataclasses.replace(
        sources[1], direction="rus_Cyrl-eng_Latn", tgt_lang="eng_Latn"
    )

    assert _fingerprint(export, sources=sources) != _fingerprint(export)


def test_a_shard_size_below_one_is_refused(export):
    with pytest.raises(ValueError, match="shard_size"):
        _fingerprint(export, shard_size=0)


def test_the_row_order_inside_a_frame_does_not_change_the_fingerprint(export):
    turned = [
        dataclasses.replace(source, frame=source.frame.reverse())
        for source in planted_sources()
    ]

    assert _fingerprint(export, sources=turned) == _fingerprint(export)


def test_the_order_of_the_source_frames_does_not_change_the_fingerprint(export):
    sources = planted_sources()

    assert _fingerprint(export, sources=list(reversed(sources))) == _fingerprint(
        export, sources=sources
    )


def test_the_references_are_not_in_the_fingerprint(export):
    sources = planted_sources()
    first = sources[0]
    sources[0] = dataclasses.replace(first, references={})

    assert _fingerprint(export, sources=sources) == _fingerprint(export)


def test_two_frames_of_one_direction_are_refused(export):
    twice = [*planted_sources(), planted_sources()[0]]

    with pytest.raises(ConfigError, match=EN_MN):
        _fingerprint(export, sources=twice)
