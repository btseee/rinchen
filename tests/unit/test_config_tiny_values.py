"""The values of the tiny pipeline's profile, `rinchen.config.tiny` (D15, spec P7, P8 and P12):
the whole numbers, the numbers with a range, the paths, the digests and the stamp, each refused by the
loader with a `ConfigError` that names the field and quotes nothing the file said.

A count is a whole number and not a boolean, a float or a quoted text; a ratio, a share or a bound
has the range of the stage or the threshold it feeds, so that no bound is one that cannot fail; a
path is not blank, has only printable characters and holds no `..` component; a digest is 64
lower-case hexadecimal digits; and the stamp is a UTC time of one closed shape. The document is the
literal of `tests/support/tiny_config_documents.py`, edited one field at a time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import ThresholdsSection, load_tiny_config
from rinchen.tiny.fixtures import FIXTURE_FILES
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_config_documents import (
    document,
    dump,
    refusal,
    with_value,
)

pytestmark = pytest.mark.usefixtures("no_network")

# ---- the whole numbers ----------------------------------------------------------------------------------

POSITIVE_INTS = [
    ("spm", "vocab_size"),
    ("spm", "input_sentence_size"),
    ("converter", "char_vocab_size"),
    ("synth", "shard_size"),
    ("synth", "n"),
    ("train", "steps"),
    ("train", "average_last"),
    ("train", "dev_max_lines"),
    ("parity", "max_lines"),
    ("threads",),
]


@pytest.mark.parametrize("path", POSITIVE_INTS, ids=lambda p: ".".join(p))
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, 2.5, "5", None, [1]])
def test_a_count_must_be_a_positive_whole_number(
    path: tuple[str, ...], bad: Any
) -> None:
    assert ".".join(path) in refusal(with_value(path, bad))


@pytest.mark.parametrize("path", POSITIVE_INTS, ids=lambda p: ".".join(p))
def test_a_count_of_one_is_accepted(path: tuple[str, ...]) -> None:
    load_tiny_config(dump(with_value(path, 1)))


@pytest.mark.parametrize("threads", [65, 1000])
def test_more_than_64_threads_are_refused(threads: int) -> None:
    assert "threads" in refusal(with_value(("threads",), threads))


def test_64_threads_are_accepted() -> None:
    assert load_tiny_config(dump(with_value(("threads",), 64))).threads == 64


@pytest.mark.parametrize("bad", [1, 0, "true", "yes", None, 1.0])
def test_the_stand_in_switch_must_be_a_boolean(bad: Any) -> None:
    assert "qe_fake" in refusal(with_value(("qe_fake",), bad))


@pytest.mark.parametrize("good", [True, False])
def test_the_stand_in_switch_takes_either_boolean(good: bool) -> None:
    assert load_tiny_config(dump(with_value(("qe_fake",), good))).qe_fake is good


# ---- the numbers with a range ---------------------------------------------------------------------------

# (path, values refused, values accepted)
RANGES: list[tuple[tuple[str, ...], list[Any], list[Any]]] = [
    (
        ("spm", "character_coverage"),
        [0, 0.0, 1.0001, -0.1, True, "0.9", None],
        [0.0001, 0.5, 1, 1.0],
    ),
    (("spm", "temperature"), [0, 0.0, -1, True, "0.3", None], [0.0001, 0.3, 1, 5]),
    (("filter", "char_ratio_min"), [0, -0.4, True, "0.4", None], [0.0001, 0.4, 2.4]),
    (("filter", "char_ratio_max"), [0, -2.5, True, None], [0.4001, 2.5, 10]),
    (
        ("filter", "minhash_threshold"),
        [0, 0.0, 1.0001, True, "0.8", None],
        [0.0001, 0.8, 1, 1.0],
    ),
    (
        ("filter", "labse_min_cosine"),
        [-1.0001, 1.0001, True, "0.7", None],
        [-1, -1.0, 0, 0.7, 1.0],
    ),
    (("filter", "qe_max_metricx"), [-0.0001, -1, True, "8", None], [0, 0.0, 8, 25.0]),
    (
        ("filter", "max_authentic_flagged_share"),
        [-0.0001, 1.0001, True, "0.01", None],
        [0, 0.0, 0.01, 1, 1.0],
    ),
    (("train", "temperature"), [0, 0.0, -1, True, "1.0", None], [0.0001, 1, 1.0, 5]),
    (
        ("train", "bt_ratio"),
        [-0.0001, 1000.0001, True, "0.2", None],
        [0, 0.0, 0.2, 1000, 1000.0],
    ),
    # The accepted values of a threshold are the ones at or beyond the master plan's bound (below it
    # the profile is refused, see FLOORS): that range is what the schema of the field itself allows.
    (
        ("thresholds", "train_chrf_min"),
        [0, 0.0, -1, 100.0001, True, "50", None],
        [50, 50.0, 50.0001, 100, 100.0],
    ),
    (
        ("thresholds", "parity_exact_min"),
        [0, 0.0, -0.1, 1.0001, True, "0.95", None],
        [0.95, 0.99, 1, 1.0],
    ),
    (
        ("thresholds", "parity_delta_max"),
        [0, 0.0, -0.001, 1.0001, True, "0.001", None],
        [1e-9, 0.0005, 0.001],
    ),
    (
        ("thresholds", "int8_chrf_drop_max"),
        [-0.0001, -1, 100.0001, True, "1.0", None],
        [0, 0.0, 0.5, 1, 1.0],
    ),
    (
        ("thresholds", "script_purity_min"),
        [0, 0.0, -0.1, 1.0001, True, "1.0", None],
        [1, 1.0],
    ),
    (
        ("thresholds", "placeholder_acc_min"),
        [0, 0.0, -0.1, 1.0001, True, "1.0", None],
        [1, 1.0],
    ),
]


@pytest.mark.parametrize(
    ("path", "refused", "accepted"), RANGES, ids=[".".join(r[0]) for r in RANGES]
)
def test_a_number_outside_its_range_or_of_the_wrong_kind_is_refused(
    path: tuple[str, ...], refused: list[Any], accepted: list[Any]
) -> None:
    for bad in refused:
        assert ".".join(path) in refusal(with_value(path, bad)), bad
    for good in accepted:
        node: Any = load_tiny_config(dump(with_value(path, good)))
        for part in path:
            node = getattr(node, part)
        assert node == good


@pytest.mark.parametrize("path", [r[0] for r in RANGES], ids=lambda p: ".".join(p))
@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"]
)
def test_a_number_that_is_not_finite_is_refused(
    path: tuple[str, ...], bad: float
) -> None:
    assert ".".join(path) in refusal(with_value(path, bad))


# ---- the floors of the thresholds ----------------------------------------------------------------------

# Per threshold: the master plan's bound as the message writes it, values on the wrong side of it (the
# first is a marker the message must not repeat) and values at or beyond it.
FLOORS = [
    (
        "train_chrf_min",
        "at least 50.0",
        [41.123456, 49.999, 0.0001],
        [50.0, 50.0001, 100],
    ),
    (
        "parity_exact_min",
        "at least 0.95",
        [0.912345, 0.9499, 0.0001],
        [0.95, 0.99, 1.0],
    ),
    (
        "parity_delta_max",
        "at most 0.001",
        [0.012345, 0.0011, 1.0],
        [0.001, 0.0005, 1e-9],
    ),
    ("int8_chrf_drop_max", "at most 1.0", [7.123456, 1.0001, 100.0], [0, 0.5, 1.0]),
    ("script_purity_min", "at least 1.0", [0.912345, 0.9999, 0.0001], [1, 1.0]),
    ("placeholder_acc_min", "at least 1.0", [0.812345, 0.9999, 0.0001], [1, 1.0]),
]


@pytest.mark.parametrize(
    ("name", "bound", "lowered", "kept"), FLOORS, ids=[floor[0] for floor in FLOORS]
)
def test_a_bound_below_the_master_plans_is_refused_naming_the_field_and_the_bound(
    name: str, bound: str, lowered: list[float], kept: list[float]
) -> None:
    for value in lowered:
        message = refusal(with_value(("thresholds", name), value))
        assert message.startswith("invalid tiny config: "), message
        assert f"thresholds.{name} " in message
        assert bound in message
        if value == lowered[0]:
            assert str(value) not in message


@pytest.mark.parametrize(
    ("name", "bound", "lowered", "kept"), FLOORS, ids=[floor[0] for floor in FLOORS]
)
def test_a_bound_at_or_beyond_the_master_plans_is_kept(
    name: str, bound: str, lowered: list[float], kept: list[float]
) -> None:
    for value in kept:
        config = load_tiny_config(dump(with_value(("thresholds", name), value)))
        assert getattr(config.thresholds, name) == value


def test_every_bound_that_was_lowered_is_named_in_one_message() -> None:
    doc = document()
    for name, _, lowered, _ in FLOORS:
        doc["thresholds"][name] = lowered[0]
    message = refusal(doc)
    for name, bound, _, _ in FLOORS:
        assert f"thresholds.{name} " in message
        assert bound in message


def test_the_floors_are_checked_for_a_profile_and_not_for_a_section_built_in_a_test() -> (
    None
):
    # The logic of the thresholds is tested with bounds that a profile may not hold.
    assert ThresholdsSection(train_chrf_min=10.0).train_chrf_min == 10.0


def test_the_character_ratio_bounds_must_be_in_order() -> None:
    doc = document()
    doc["filter"]["char_ratio_min"] = 2.5
    doc["filter"]["char_ratio_max"] = 2.5
    assert "filter" in refusal(doc)
    doc["filter"]["char_ratio_min"] = 3.0
    assert "filter" in refusal(doc)
    doc["filter"]["char_ratio_min"] = 2.4999
    load_tiny_config(dump(doc))


def test_an_integer_where_a_float_is_wanted_is_read_as_that_float() -> None:
    config = load_tiny_config(dump(with_value(("thresholds", "script_purity_min"), 1)))
    assert config.thresholds.script_purity_min == 1.0
    assert isinstance(config.thresholds.script_purity_min, float)


# ---- the paths --------------------------------------------------------------------------------------------

PATH_FIELDS = [
    ("fixtures", "directory"),
    ("registry",),
    ("converter", "pairs"),
    ("train", "base"),
    ("serve", "config"),
]


@pytest.mark.parametrize("path", PATH_FIELDS, ids=lambda p: ".".join(p))
@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "   ",
        "\t",
        "..",
        "../x",
        "a/../b",
        "a/..",
        "/abs/../x",
        "a\nb",
        "a\x00b",
        "a\tb",
        None,
        7,
        ["a"],
        {"a": "b"},
    ],
)
def test_a_path_that_is_blank_or_holds_dotdot_or_is_not_text_is_refused(
    path: tuple[str, ...], bad: Any
) -> None:
    assert ".".join(path) in refusal(with_value(path, bad))


@pytest.mark.parametrize("path", PATH_FIELDS, ids=lambda p: ".".join(p))
def test_a_relative_path_is_anchored_at_the_repository_and_an_absolute_one_is_kept(
    path: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def read(value: str) -> Path:
        node: Any = load_tiny_config(dump(with_value(path, value)))
        for part in path:
            node = getattr(node, part)
        assert isinstance(node, Path)
        return node

    assert read("some/place.txt") == REPO_ROOT / "some" / "place.txt"
    assert read("a..b/c") == REPO_ROOT / "a..b" / "c"
    assert read(str(tmp_path / "copy.txt")) == tmp_path / "copy.txt"


def test_a_path_with_dots_that_is_not_a_parent_component_is_accepted() -> None:
    config = load_tiny_config(dump(with_value(("registry",), "configs/data/x..y.yaml")))
    assert config.registry == REPO_ROOT / "configs" / "data" / "x..y.yaml"


# ---- the digests ------------------------------------------------------------------------------------------


def digest_path(name: str) -> tuple[str, ...]:
    return ("fixtures", "sha256", name)


@pytest.mark.parametrize("name", FIXTURE_FILES)
@pytest.mark.parametrize(
    "bad",
    [
        "a" * 63,
        "a" * 65,
        "",
        "g" * 64,
        "A" * 64,
        "a" * 63 + "G",
        " " + "a" * 63,
        "a" * 64 + "\n",
        "0x" + "a" * 62,
        "\u0661" * 64,
        1,
        None,
        True,
        ["a" * 64],
    ],
)
def test_a_digest_that_is_not_64_lowercase_hex_digits_is_refused(
    name: str, bad: Any
) -> None:
    message = refusal(with_value(digest_path(name), bad))
    assert f"fixtures.sha256.{name}" in message


def test_a_digest_of_64_hex_digits_is_accepted() -> None:
    digest = "0123456789abcdef" * 4
    config = load_tiny_config(dump(with_value(digest_path(FIXTURE_FILES[0]), digest)))
    assert config.fixtures.sha256[FIXTURE_FILES[0]] == digest


def test_the_digests_must_name_exactly_the_seven_fixtures() -> None:
    doc = document()
    doc["fixtures"]["sha256"]["eight.tsv"] = "a" * 64
    message = refusal(doc)
    assert "fixtures.sha256" in message
    doc = document()
    del doc["fixtures"]["sha256"][FIXTURE_FILES[3]]
    message = refusal(doc)
    assert "fixtures.sha256" in message
    assert FIXTURE_FILES[3] in message


@pytest.mark.parametrize("table", [[], "digest", 5, None, [["a", "b"]]])
def test_the_digests_must_be_a_mapping(table: Any) -> None:
    assert "fixtures.sha256" in refusal(with_value(("fixtures", "sha256"), table))


# ---- the time that is stamped on every row ---------------------------------------------------------------


@pytest.mark.parametrize(
    "created",
    [
        "2026-11-11",
        "2026-11-11T00:00:00",
        "2026-11-11T00:00:00+00:00",
        "2026-11-11T00:00:00Z ",
        " 2026-11-11T00:00:00Z",
        "2026-11-11T00:00:00z",
        "2026-11-11t00:00:00Z",
        "2026-11-11 00:00:00Z",
        "2026-13-11T00:00:00Z",
        "2026-11-31T00:00:00Z",
        "2026-02-30T00:00:00Z",
        "2026-11-11T24:00:00Z",
        "2026-11-11T00:60:00Z",
        "2026-11-11T23:59:60Z",
        "2026-11-11T23:59:61Z",
        "2026-11-11T00:00:00.5Z",
        "26-11-11T00:00:00Z",
        "2026-1-1T0:0:0Z",
        "\u0662\u0660\u0662\u0666-11-11T00:00:00Z",
        "2026-11-11T00:00:00Z\n",
        "",
        "now",
        None,
        20261111,
        True,
        ["2026-11-11T00:00:00Z"],
    ],
)
def test_a_stamp_that_is_not_an_iso_utc_time_is_refused(created: Any) -> None:
    assert "created" in refusal(with_value(("created",), created))


def test_an_unquoted_time_is_refused_because_yaml_reads_it_as_a_date() -> None:
    text = dump(document()).decode("utf-8")
    quoted = "created: '2026-11-11T00:00:00Z'"
    assert quoted in text
    message = refusal(
        text.replace(quoted, "created: 2026-11-11T00:00:00Z").encode("utf-8")
    )
    assert "created" in message
    assert "quoted" in message


@pytest.mark.parametrize(
    "created", ["2026-11-11T00:00:00Z", "2000-01-01T23:59:59Z", "2026-02-28T12:30:45Z"]
)
def test_a_stamp_of_the_closed_shape_is_accepted(created: str) -> None:
    assert load_tiny_config(dump(with_value(("created",), created))).created == created
