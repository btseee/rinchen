"""The document of the robustness result and the system behind it (D12, spec T13): the bytes
are deterministic (sorted keys, one final LF, no timestamp, no NaN, no text of a suite or of a
hypothesis), the system is recorded with its digests and options and validated, and a suite
that was built by hand and is wrong is a programming error and not a silent drop."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import pytest

from rinchen.eval.robust_hypotheses import Hypothesis
from rinchen.eval.robustness import (
    RobustnessSystem,
    compute_robustness,
    robustness_block,
)
from tests.support.robust_golden import (
    GOLDEN_SYSTEM,
    KHK,
    MONGOLIAN,
    golden_hypotheses,
    golden_result,
    golden_suite,
    golden_suite_sha256,
    key,
    row,
    run,
    suite_of,
    texts,
)

MARKER = "SECRET-MARKER-63"


# ---- the document ---------------------------------------------------------------------------


def test_the_same_inputs_give_the_same_bytes() -> None:
    assert golden_result().to_bytes() == golden_result().to_bytes()


def test_the_bytes_end_with_one_line_feed_and_sort_their_keys_at_every_level() -> None:
    data = golden_result().to_bytes()
    text = data.decode("utf-8")

    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    reformatted = json.dumps(
        json.loads(text), sort_keys=True, indent=2, ensure_ascii=False
    )
    assert text == f"{reformatted}\n"


# What the counts are keyed by is the suite's data (a category, a direction, a status).
DATA_KEYED = ("by_category", "by_direction", "statuses", "excluded")


def field_names(value: object) -> set[str]:
    """The names of the fields of a document, leaving out the keys that are data."""
    if isinstance(value, dict):
        found = set(value)
        for name, item in value.items():
            if name not in DATA_KEYED:
                found |= field_names(item)
        return found
    if isinstance(value, list):
        return {name for item in value for name in field_names(item)}
    return set()


def test_the_result_holds_no_timestamp() -> None:
    names = field_names(golden_result().to_dict())

    assert not {"created", "timestamp", "time", "date", "generated", "at"} & names


def test_the_result_never_holds_a_text_of_the_suite_or_of_a_hypothesis() -> None:
    suite = suite_of(
        numbers=(row("a1", source=f"{MARKER} 15", reference=f"{MARKER} 15"),)
    )

    data = compute_robustness(
        suite,
        texts(a1=f"{MARKER} 15"),
        suite_sha256="0" * 64,
        system=GOLDEN_SYSTEM,
    ).to_bytes()

    assert MARKER.encode() not in data


def test_a_value_that_is_not_a_number_cannot_be_written() -> None:
    result = golden_result()
    numbers = result.sections["numbers"]
    broken = dataclasses.replace(
        result,
        sections={
            **result.sections,
            "numbers": dataclasses.replace(numbers, value=math.nan),
        },
    )

    with pytest.raises(ValueError, match="Out of range float"):
        broken.to_bytes()


def test_the_result_is_frozen() -> None:
    result = golden_result()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.system = GOLDEN_SYSTEM  # type: ignore[misc]


def test_the_block_of_a_result_is_what_its_document_says() -> None:
    result = golden_result()

    assert robustness_block(result) == result.to_dict()["robustness"]


# ---- the system ---------------------------------------------------------------------------


def test_a_model_system_is_recorded_with_its_digests_and_options() -> None:
    system = RobustnessSystem(
        name="tiny.v1",
        model_sha256="a" * 64,
        manifest_sha256="b" * 64,
        options={"beam_size": 4, "converter": False, "script_mask": "retry"},
    )

    document = compute_robustness(
        golden_suite(),
        golden_hypotheses(),
        suite_sha256=golden_suite_sha256(),
        system=system,
    ).to_dict()

    assert document["system"] == {
        "name": "tiny.v1",
        "model_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "options": {"beam_size": 4, "converter": False, "script_mask": "retry"},
    }


def test_the_options_of_a_system_are_a_copy() -> None:
    options = {"beam_size": 4}
    system = RobustnessSystem(name="s", options=options)

    options["beam_size"] = 9

    assert dict(system.options or {}) == {"beam_size": 4}
    with pytest.raises(TypeError):
        system.options["beam_size"] = 2  # type: ignore[index]


@pytest.mark.parametrize("name", ["", "a b", "x" * 65, chr(0xE9), MARKER + "!", 5])
def test_a_system_name_that_is_not_an_id_is_refused(name: object) -> None:
    with pytest.raises(ValueError, match="name") as info:
        RobustnessSystem(name=name)  # type: ignore[arg-type]

    assert MARKER not in str(info.value)


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64, 5, "a" * 65])
@pytest.mark.parametrize("field", ["model_sha256", "manifest_sha256"])
def test_a_system_digest_must_be_a_sha256_or_none(field: str, digest: object) -> None:
    with pytest.raises(ValueError, match=field):
        RobustnessSystem(name="s", **{field: digest})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1.5, None, [1], {"a": 1}, math.nan])
def test_an_option_of_a_system_is_a_string_an_integer_or_a_boolean(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="options"):
        RobustnessSystem(name="s", options={"a": value})  # type: ignore[dict-item]


def test_an_option_name_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="options"):
        RobustnessSystem(name="s", options={1: 2})  # type: ignore[dict-item]


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_the_digest_of_the_suite_must_be_a_sha256(digest: str) -> None:
    with pytest.raises(ValueError, match="suite_sha256"):
        compute_robustness(
            golden_suite(),
            golden_hypotheses(),
            suite_sha256=digest,
            system=GOLDEN_SYSTEM,
        )


def test_the_digest_in_the_result_is_the_one_given() -> None:
    digest = hashlib.sha256(b"x").hexdigest()

    document = compute_robustness(
        golden_suite(), golden_hypotheses(), suite_sha256=digest, system=GOLDEN_SYSTEM
    ).to_dict()

    assert document["suite"]["sha256"] == digest


# ---- a suite that was built by hand and is wrong -----------------------------------------


def test_an_entities_row_that_lists_no_entity_is_a_programming_error() -> None:
    suite = suite_of(entities=(row("e1", entities=()),))

    with pytest.raises(ValueError, match="entities"):
        run(suite, {key("entities", "e1"): Hypothesis("x", "ok")})


def test_a_terms_row_that_lists_no_term_is_a_programming_error() -> None:
    suite = suite_of(terms=(row("t1", terms=()),))

    with pytest.raises(ValueError, match="terms"):
        run(suite, {key("terms", "t1"): Hypothesis("x", "ok")})


@pytest.mark.parametrize("section", ["numbers", "placeholders", "mong"])
def test_a_row_without_a_reference_where_one_is_required_is_a_programming_error(
    section: str,
) -> None:
    extra = {"source_mong": MONGOLIAN, "src_lang": KHK} if section == "mong" else {}
    suite = suite_of(**{section: (row("a1", reference=None, **extra),)})
    arm = "cyrl" if section == "mong" else "main"
    hypotheses = {key(section, "a1", arm): Hypothesis("x", "ok")}
    if section == "mong":
        hypotheses[key(section, "a1", "mong")] = Hypothesis("x", "ok")

    with pytest.raises(ValueError, match="reference"):
        run(suite, hypotheses)


def test_text_is_written_as_it_is_and_not_as_escapes() -> None:
    system = RobustnessSystem(name="s", options={"note": "Монгол"})

    data = compute_robustness(
        golden_suite(),
        golden_hypotheses(),
        suite_sha256=golden_suite_sha256(),
        system=system,
    ).to_bytes()

    assert "Монгол".encode() in data
    assert b"\\u" not in data
