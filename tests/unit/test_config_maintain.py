"""The configuration of the maintenance check (D14, spec M6, M7, M8 and M13): the pin rules, the
uv argument lists and the parity reruns, as data in `configs/maintain/maintain.yaml`.

The loader enforces the guards of M6 at load time: no configuration reaches the runner with an
argument list that can write the lock file or upgrade one package. Every refusal is a
`ConfigError` that names the field and quotes nothing of the value that was refused. The shipped
table, `reruns_due` and `rule_for` are in `test_config_maintain_shipped.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import yaml

from rinchen.config.maintain import (
    ALLOWED_UV_FLAGS,
    CADENCES,
    GATE,
    MaintainConfig,
    load_maintain_config,
)
from rinchen.errors import ConfigError
from rinchen.maintain.states import SKIP_GPU_CHECK
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def document() -> dict[str, Any]:
    """A valid document, as a dict that a test then breaks in one place."""
    return {
        "skipped_doctor_checks": {"D-TORCH-CUDA": SKIP_GPU_CHECK},
        "lock_check": {"uv_args": ["lock", "--check"], "timeout_seconds": 120},
        "drift": {
            "uv_args": ["lock", "--upgrade", "--dry-run"],
            "timeout_seconds": 300,
            "rules": [
                {
                    "id": "eole",
                    "packages": ["eole"],
                    "decision": "Wiki page Decisions: the eole pin",
                    "requires": ["parity_tiny", GATE],
                },
                {
                    "id": "torch-family",
                    "packages": ["torch", "torch-*"],
                    "decision": "Wiki page Decisions: the cu130 pin",
                    "requires": ["parity_tiny"],
                },
            ],
        },
        "reruns": [
            {
                "id": "parity_tiny",
                "kind": "pytest",
                "cadence": "nightly",
                "timeout_seconds": 600,
                "why": "Eole against CTranslate2 on the toy",
                "targets": ["tests/integration/test_export_parity.py"],
                "min_passed": 13,
            },
            {
                "id": "parity_export",
                "kind": "export_parity",
                "cadence": "nightly",
                "timeout_seconds": 600,
                "why": "the parity command over an operator's export",
            },
        ],
    }


def dump(doc: dict[str, Any]) -> bytes:
    return yaml.safe_dump(doc, sort_keys=False).encode("utf-8")


def load(doc: dict[str, Any]) -> MaintainConfig:
    return load_maintain_config(dump(doc))


def edited(*path: str | int, value: Any) -> dict[str, Any]:
    """The valid document with the value at `path` replaced."""
    doc = document()
    node: Any = doc
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return doc


def without(*path: str | int) -> dict[str, Any]:
    """The valid document with the key at `path` removed."""
    doc = document()
    node: Any = doc
    for step in path[:-1]:
        node = node[step]
    del node[path[-1]]
    return doc


def refusal(data: bytes) -> str:
    with pytest.raises(ConfigError) as caught:
        load_maintain_config(data)
    return str(caught.value)


def test_the_valid_document_loads() -> None:
    config = load(document())
    assert [rule.id for rule in config.drift.rules] == ["eole", "torch-family"]
    assert [rerun.id for rerun in config.reruns] == ["parity_tiny", "parity_export"]
    assert config.skipped_doctor_checks == {"D-TORCH-CUDA": SKIP_GPU_CHECK}
    assert config.lock_check.uv_args == ["lock", "--check"]


def test_the_module_constants_are_those_of_the_interface() -> None:
    assert CADENCES == ("nightly", "weekly")
    assert GATE == "gate"
    assert ALLOWED_UV_FLAGS == frozenset(
        {"--upgrade", "--dry-run", "--offline", "--check"}
    )


# --- an unknown key and a missing key, at every level -------------------------------------------

UNKNOWN_KEY_PLACES: list[tuple[Any, ...]] = [
    (),
    ("lock_check",),
    ("drift",),
    ("drift", "rules", 0),
    ("reruns", 0),
    ("reruns", 1),
]


@pytest.mark.parametrize("place", UNKNOWN_KEY_PLACES)
def test_an_unknown_key_is_refused_at_every_level(place: tuple[Any, ...]) -> None:
    doc = document()
    node: Any = doc
    for step in place:
        node = node[step]
    node["zqx_unknown"] = 1
    message = refusal(dump(doc))
    assert "zqx_unknown" in message


MISSING_KEYS: list[tuple[tuple[Any, ...], str]] = [
    (("skipped_doctor_checks",), "skipped_doctor_checks"),
    (("lock_check",), "lock_check"),
    (("drift",), "drift"),
    (("reruns",), "reruns"),
    (("lock_check", "uv_args"), "lock_check.uv_args"),
    (("lock_check", "timeout_seconds"), "lock_check.timeout_seconds"),
    (("drift", "uv_args"), "drift.uv_args"),
    (("drift", "timeout_seconds"), "drift.timeout_seconds"),
    (("drift", "rules"), "drift.rules"),
    (("drift", "rules", 0, "id"), "drift.rules.0.id"),
    (("drift", "rules", 0, "packages"), "drift.rules.0.packages"),
    (("drift", "rules", 0, "decision"), "drift.rules.0.decision"),
    (("drift", "rules", 0, "requires"), "drift.rules.0.requires"),
    (("reruns", 0, "id"), "reruns.0.id"),
    (("reruns", 0, "kind"), "reruns.0.kind"),
    (("reruns", 0, "cadence"), "reruns.0.cadence"),
    (("reruns", 0, "timeout_seconds"), "reruns.0.timeout_seconds"),
    (("reruns", 0, "why"), "reruns.0.why"),
    (("reruns", 0, "targets"), "targets"),
    (("reruns", 0, "min_passed"), "min_passed"),
    (("reruns", 1, "id"), "reruns.1.id"),
    (("reruns", 1, "kind"), "reruns.1.kind"),
    (("reruns", 1, "cadence"), "reruns.1.cadence"),
    (("reruns", 1, "timeout_seconds"), "reruns.1.timeout_seconds"),
    (("reruns", 1, "why"), "reruns.1.why"),
]


@pytest.mark.parametrize(("path", "field"), MISSING_KEYS)
def test_a_missing_key_is_refused_at_every_level(
    path: tuple[Any, ...], field: str
) -> None:
    assert field in refusal(dump(without(*path)))


# --- the argument lists of uv: the guards of M6 ---------------------------------------------------

DRIFT_ARGS_REFUSED = [
    pytest.param(["--upgrade", "--dry-run"], id="not-starting-with-lock"),
    pytest.param(["Lock", "--upgrade", "--dry-run"], id="a-capital-lock"),
    pytest.param(["--dry-run", "lock"], id="lock-not-first"),
    pytest.param([], id="empty"),
    pytest.param(["lock"], id="no-dry-run-no-flag"),
    pytest.param(["lock", "--upgrade"], id="upgrade-without-dry-run"),
    pytest.param(["lock", "--offline"], id="offline-without-dry-run"),
    pytest.param(["lock", "--check"], id="check-without-dry-run"),
    pytest.param(
        ["lock", "--upgrade", "--dry-run", "--upgrade-package"], id="upgrade-package"
    ),
    pytest.param(
        ["lock", "--upgrade-package", "numpy", "--dry-run"], id="upgrade-package-numpy"
    ),
    pytest.param(["lock", "--upgrade", "--dry-run", "-U"], id="short-upgrade"),
    pytest.param(["lock", "--upgrade", "--dry-run", "--refresh"], id="refresh"),
    pytest.param(["lock", "--upgrade", "--dry-run", "--no-cache"], id="no-cache"),
    pytest.param(["lock", "--upgrade", "--dry-run", "--frozen"], id="frozen"),
    pytest.param(
        ["lock", "--upgrade", "--dry-run", "--dry-run"], id="repeated-dry-run"
    ),
    pytest.param(
        ["lock", "--upgrade", "--upgrade", "--dry-run"], id="repeated-upgrade"
    ),
    pytest.param(["lock", "--dry-run=true"], id="flag-with-a-value"),
    pytest.param(["lock", "lock", "--dry-run"], id="lock-twice"),
    pytest.param(["lock", "--dry-run", ""], id="an-empty-argument"),
    pytest.param(["lock", "--dry-run", 3], id="a-number-argument"),
    pytest.param("lock --upgrade --dry-run", id="a-string"),
    pytest.param(None, id="null"),
]


@pytest.mark.parametrize("args", DRIFT_ARGS_REFUSED)
def test_a_drift_argument_list_that_could_write_or_upgrade_is_refused(
    args: Any,
) -> None:
    message = refusal(dump(edited("drift", "uv_args", value=args)))
    assert "drift.uv_args" in message
    for text in ("--upgrade-package", "numpy", "--refresh", "--no-cache", "--frozen"):
        assert text not in message


@pytest.mark.parametrize(
    "args",
    [
        ["lock", "--upgrade", "--dry-run"],
        ["lock", "--dry-run", "--upgrade"],
        ["lock", "--dry-run"],
        ["lock", "--offline", "--dry-run"],
        ["lock", "--upgrade", "--dry-run", "--offline"],
        ["lock", "--dry-run", "--check"],
    ],
)
def test_a_drift_argument_list_of_the_closed_flags_with_dry_run_is_accepted(
    args: list[str],
) -> None:
    config = load(edited("drift", "uv_args", value=args))
    assert config.drift.uv_args == args


LOCK_CHECK_ARGS_REFUSED = [
    pytest.param(["lock"], id="no-check"),
    pytest.param(["lock", "--offline"], id="offline-without-check"),
    pytest.param(["lock", "--dry-run"], id="dry-run-without-check"),
    pytest.param(["lock", "--upgrade"], id="upgrade-without-check"),
    pytest.param(["--check"], id="not-starting-with-lock"),
    pytest.param(["lock", "--check", "--check"], id="repeated-check"),
    pytest.param(["lock", "--check", "--upgrade-package"], id="upgrade-package"),
    pytest.param(["lock", "--check", "-U"], id="short-upgrade"),
    pytest.param(["lock", "--check", "--frozen"], id="frozen"),
    pytest.param(["lock", "--check", "--upgrade"], id="check-with-upgrade"),
    pytest.param(["lock", "--upgrade", "--check"], id="upgrade-before-check"),
    pytest.param(
        ["lock", "--check", "--offline", "--upgrade"], id="offline-check-with-upgrade"
    ),
    pytest.param(["lock", "--check", "--dry-run"], id="check-with-dry-run"),
    pytest.param(
        ["lock", "--dry-run", "--check", "--upgrade"], id="upgrade-dry-run-check"
    ),
    pytest.param([], id="empty"),
]


@pytest.mark.parametrize("args", LOCK_CHECK_ARGS_REFUSED)
def test_a_lock_check_argument_list_without_check_or_with_another_flag_is_refused(
    args: list[str],
) -> None:
    message = refusal(dump(edited("lock_check", "uv_args", value=args)))
    assert "lock_check.uv_args" in message
    assert "--frozen" not in message


@pytest.mark.parametrize(
    "args",
    [
        ["lock", "--check"],
        ["lock", "--offline", "--check"],
        ["lock", "--check", "--offline"],
    ],
)
def test_a_lock_check_argument_list_with_check_is_accepted(args: list[str]) -> None:
    assert load(edited("lock_check", "uv_args", value=args)).lock_check.uv_args == args


def test_the_lock_check_is_lock_check_with_at_most_offline_and_no_other_flag() -> None:
    for flag in sorted(ALLOWED_UV_FLAGS - {"--check", "--offline"}):
        args = edited("lock_check", "uv_args", value=["lock", "--check", flag])
        assert "lock_check.uv_args" in refusal(dump(args)), flag


TIMEOUT_PLACES = [
    ("lock_check", "timeout_seconds"),
    ("drift", "timeout_seconds"),
    ("reruns", 0, "timeout_seconds"),
    ("reruns", 1, "timeout_seconds"),
]


@pytest.mark.parametrize("place", TIMEOUT_PLACES)
@pytest.mark.parametrize("bad", [0, -1, 14_401, True, False, 300.5, 300.0, "300", None])
def test_a_timeout_that_is_not_an_integer_of_1_to_14400_is_refused(
    place: tuple[Any, ...], bad: Any
) -> None:
    message = refusal(dump(edited(*place, value=bad)))
    assert ".".join(str(step) for step in place) in message


@pytest.mark.parametrize("place", TIMEOUT_PLACES)
@pytest.mark.parametrize("good", [1, 14_400])
def test_the_bounds_of_a_timeout_are_accepted(
    place: tuple[Any, ...], good: int
) -> None:
    load(edited(*place, value=good))


# --- the reruns -----------------------------------------------------------------------------------


def test_a_rerun_id_that_repeats_is_refused() -> None:
    doc = document()
    doc["reruns"][1]["id"] = "parity_tiny"
    message = refusal(dump(doc))
    assert "reruns" in message
    assert "parity_tiny" in message


@pytest.mark.parametrize(
    "bad",
    [
        "Numerics",
        "nUmerics",
        "1numerics",
        "_x",
        "a-b",
        "a b",
        "",
        "a" * 33,
        "gate",
        "ab\n",
        "n" + chr(0xFC) + "merics",
    ],
)
def test_a_rerun_id_outside_its_pattern_or_named_gate_is_refused(bad: str) -> None:
    message = refusal(dump(edited("reruns", 0, "id", value=bad)))
    assert "reruns.0.id" in message


@pytest.mark.parametrize(
    "good", ["a", "numerics", "parity_other", "a" + "9" * 31, "x_1"]
)
def test_a_rerun_id_inside_its_pattern_is_accepted(good: str) -> None:
    # Rerun 1 is the one that no rule requires, so it can be renamed.
    load(edited("reruns", 1, "id", value=good))


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("cadence", "daily"),
        ("cadence", "Nightly"),
        ("cadence", ""),
        ("cadence", None),
        ("kind", "shell"),
        ("kind", "Pytest"),
        ("kind", ""),
        ("kind", None),
    ],
)
def test_a_cadence_or_kind_outside_the_closed_lists_is_refused(
    key: str, bad: Any
) -> None:
    message = refusal(dump(edited("reruns", 0, key, value=bad)))
    assert f"reruns.0.{key}" in message
    assert "daily" not in message
    assert "shell" not in message


@pytest.mark.parametrize("cadence", ["nightly", "weekly"])
def test_both_cadences_are_accepted(cadence: str) -> None:
    assert (
        load(edited("reruns", 0, "cadence", value=cadence)).reruns[0].cadence == cadence
    )


@pytest.mark.parametrize(
    "bad", ["", "   ", "a\nb", "a\r", "line\n", "a" * 301, 3, None]
)
def test_a_why_that_is_not_one_line_of_1_to_300_characters_is_refused(bad: Any) -> None:
    assert "reruns.0.why" in refusal(dump(edited("reruns", 0, "why", value=bad)))


def test_a_why_of_300_characters_is_accepted() -> None:
    load(edited("reruns", 0, "why", value="w" * 300))


PYTEST_TARGETS_REFUSED = [
    pytest.param([], id="none"),
    pytest.param(["../x"], id="parent"),
    pytest.param(["/abs"], id="absolute"),
    pytest.param(["/tests/unit/test_x.py"], id="absolute-under-tests"),
    pytest.param(["data/x"], id="outside-tests"),
    pytest.param(["zqxcanary/x"], id="canary-outside-tests"),
    pytest.param(["tests/../zqxcanary"], id="canary-dotdot"),
    pytest.param(["tests/zqxcanary\\x"], id="canary-backslash"),
    pytest.param(["tests/../x"], id="dotdot-inside"),
    pytest.param(["tests/unit/../../x"], id="dotdot-deep"),
    pytest.param(["tests\\x"], id="backslash"),
    pytest.param(["tests/unit\\test_x.py"], id="backslash-inside"),
    pytest.param(["tests/unit/test_maintain_gate.py"], id="maintain-file"),
    pytest.param(["tests/maintain"], id="maintain-directory"),
    pytest.param(["tests/integration/test_Maintain_gate.py"], id="maintain-capital"),
    pytest.param(["tests/unit/test_x.py::test_maintain_it"], id="maintain-node-id"),
    pytest.param(["tests"], id="tests-itself"),
    pytest.param(["tests/"], id="tests-slash"),
    pytest.param(["tests//x.py"], id="empty-component"),
    pytest.param(["tests/./x.py"], id="dot-component"),
    pytest.param(["tests/golden/"], id="trailing-slash"),
    pytest.param(["tests/unit/test_x.py\n"], id="newline"),
    pytest.param(["tests/unit/test_x.py\t"], id="tab"),
    pytest.param(["TESTS/unit/test_x.py"], id="capital-tests"),
    pytest.param([""], id="empty-string"),
    pytest.param(["tests/unit/test_x.py", "x/y"], id="one-good-one-bad"),
    pytest.param(["tests/unit/test_x.py", 3], id="a-number"),
    pytest.param([f"tests/unit/test_{n}.py" for n in range(21)], id="twenty-one"),
    pytest.param("tests/unit/test_x.py", id="a-string"),
]


@pytest.mark.parametrize("targets", PYTEST_TARGETS_REFUSED)
def test_a_pytest_rerun_with_a_bad_target_list_is_refused(targets: Any) -> None:
    message = refusal(dump(edited("reruns", 0, "targets", value=targets)))
    assert "targets" in message
    assert "zqxcanary" not in message


@pytest.mark.parametrize(
    "targets",
    [
        ["tests/golden"],
        ["tests/unit/test_x.py"],
        ["tests/unit/test_x.py", "tests/property/test_y.py", "tests/golden"],
        [f"tests/unit/test_{n}.py" for n in range(20)],
        ["tests/unit/test_x.py::test_case"],
    ],
)
def test_a_pytest_rerun_with_targets_under_tests_is_accepted(
    targets: list[str],
) -> None:
    assert (
        load(edited("reruns", 0, "targets", value=targets)).reruns[0].targets == targets
    )


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, 13.0, "13", None])
def test_a_pytest_rerun_floor_that_is_not_an_integer_of_at_least_1_is_refused(
    bad: Any,
) -> None:
    assert "min_passed" in refusal(dump(edited("reruns", 0, "min_passed", value=bad)))


def test_a_floor_of_one_is_accepted() -> None:
    assert load(edited("reruns", 0, "min_passed", value=1)).reruns[0].min_passed == 1


@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None])
def test_a_max_skipped_that_is_not_a_non_negative_integer_is_refused(bad: Any) -> None:
    assert "max_skipped" in refusal(dump(edited("reruns", 0, "max_skipped", value=bad)))


def test_max_skipped_defaults_to_zero_and_may_be_raised() -> None:
    assert load(document()).reruns[0].max_skipped == 0
    assert load(edited("reruns", 0, "max_skipped", value=2)).reruns[0].max_skipped == 2


def test_the_fields_of_a_rerun_that_take_a_default_are_not_required() -> None:
    config = load(document())
    export = config.reruns[1]
    assert (export.targets, export.min_passed, export.max_skipped) == ([], None, 0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("targets", ["tests/unit/test_x.py"]),
        ("min_passed", 5),
        ("min_passed", 0),
        ("max_skipped", 1),
    ],
)
def test_an_export_parity_rerun_has_no_targets_and_no_floor(
    key: str, value: Any
) -> None:
    message = refusal(dump(edited("reruns", 1, key, value=value)))
    assert key in message


def test_an_export_parity_rerun_may_state_the_default_skip_allowance() -> None:
    load(edited("reruns", 1, "max_skipped", value=0))


# --- the pin rules ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "Eole",
        "eole*x",
        "*",
        "**",
        "-eole",
        "eole-",
        "eole_x",
        "eole ",
        "e le",
        "",
        "ab\n",
        chr(0xFC) + "n" + chr(0xEF),
        3,
        None,
    ],
)
def test_a_package_outside_its_pattern_is_refused(bad: Any) -> None:
    message = refusal(dump(edited("drift", "rules", 0, "packages", value=[bad])))
    assert "drift.rules.0.packages.0" in message
    assert "eole*x" not in message


@pytest.mark.parametrize(
    "good",
    [
        "a",
        "eole",
        "torch-*",
        "nvidia-*",
        "zope.interface",
        "a1",
        "a1-b2.c3",
        "x*",
        "a.b*",
        "9z",
    ],
)
def test_a_package_name_or_prefix_inside_its_pattern_is_accepted(good: str) -> None:
    load(edited("drift", "rules", 0, "packages", value=[good]))


def test_a_rule_with_no_packages_is_refused() -> None:
    assert "drift.rules.0.packages" in refusal(
        dump(edited("drift", "rules", 0, "packages", value=[]))
    )


def test_a_rule_with_thirteen_packages_is_refused_and_one_with_twelve_is_not() -> None:
    names = [f"p{n}" for n in range(13)]
    assert "drift.rules.0.packages" in refusal(
        dump(edited("drift", "rules", 0, "packages", value=names))
    )
    load(edited("drift", "rules", 0, "packages", value=names[:12]))


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "a\nb", "a\rb", "a\n", "a" + chr(0x2028) + "b", "d" * 301, 3, None],
)
def test_a_decision_that_is_not_one_line_of_1_to_300_characters_is_refused(
    bad: Any,
) -> None:
    assert "drift.rules.0.decision" in refusal(
        dump(edited("drift", "rules", 0, "decision", value=bad))
    )


def test_a_decision_of_300_characters_is_accepted() -> None:
    load(edited("drift", "rules", 0, "decision", value="d" * 300))


@pytest.mark.parametrize(
    "bad", ["Eole", "a_b", "", "a" * 41, "a b", "ab\n", chr(0xFC) + "n" + chr(0xEF), 3]
)
def test_a_rule_id_outside_its_pattern_is_refused(bad: Any) -> None:
    assert "drift.rules.0.id" in refusal(
        dump(edited("drift", "rules", 0, "id", value=bad))
    )


@pytest.mark.parametrize("good", ["a", "torch-fam", "a-b-1", "0", "x" * 40])
def test_a_rule_id_inside_its_pattern_is_accepted(good: str) -> None:
    load(edited("drift", "rules", 0, "id", value=good))


def test_two_rules_with_one_id_are_refused() -> None:
    doc = document()
    doc["drift"]["rules"][1]["id"] = "eole"
    message = refusal(dump(doc))
    assert "drift.rules" in message
    assert "eole" in message


def test_a_rule_requires_a_rerun_of_the_file_or_the_gate() -> None:
    doc = edited(
        "drift", "rules", 0, "requires", value=["parity_tiny", "parity_export", GATE]
    )
    assert load(doc).drift.rules[0].requires == ["parity_tiny", "parity_export", GATE]


def test_a_requires_that_names_no_rerun_is_refused() -> None:
    message = refusal(
        dump(edited("drift", "rules", 0, "requires", value=["parity_tiny", "numerics"]))
    )
    assert "drift.rules" in message
    assert "requires" in message
    assert "numerics" in message


@pytest.mark.parametrize("bad", ["Gate", "gate ", "full_gate", "", "GATE", 3, None])
def test_a_requires_that_is_not_a_rerun_or_the_word_gate_is_refused(bad: Any) -> None:
    refusal(dump(edited("drift", "rules", 0, "requires", value=[bad])))


def test_a_rule_may_require_nothing() -> None:
    assert (
        load(edited("drift", "rules", 0, "requires", value=[])).drift.rules[0].requires
        == []
    )


def test_a_document_without_rules_is_accepted() -> None:
    assert load(edited("drift", "rules", value=[])).drift.rules == []


# --- the skipped doctor checks ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["torch", "D-", "d-torch", "D-lower", "D_TORCH", "D-" + "A" * 31, "D-A B", "D-A\n"],
)
def test_a_skipped_doctor_check_key_that_is_not_a_check_id_is_refused(bad: str) -> None:
    message = refusal(
        dump(edited("skipped_doctor_checks", value={bad: SKIP_GPU_CHECK}))
    )
    assert "skipped_doctor_checks" in message


def test_the_one_skipped_doctor_check_key_that_is_accepted_is_the_device_check() -> (
    None
):
    load(edited("skipped_doctor_checks", value={"D-TORCH-CUDA": SKIP_GPU_CHECK}))


@pytest.mark.parametrize("other", ["D-A", "D-" + "A" * 30, "D-A1-B2", "D-PY"])
def test_a_check_id_that_is_not_a_device_check_is_refused_as_a_skipped_key(
    other: str,
) -> None:
    message = refusal(
        dump(edited("skipped_doctor_checks", value={other: SKIP_GPU_CHECK}))
    )
    assert "skipped_doctor_checks" in message


@pytest.mark.parametrize("bad", ["because", "", "a GPU check", 3, None])
def test_a_skipped_doctor_check_reason_outside_the_closed_list_is_refused(
    bad: Any,
) -> None:
    message = refusal(
        dump(edited("skipped_doctor_checks", value={"D-TORCH-CUDA": bad}))
    )
    assert "skipped_doctor_checks" in message
    assert "because" not in message


def test_no_skipped_doctor_check_is_accepted() -> None:
    assert load(edited("skipped_doctor_checks", value={})).skipped_doctor_checks == {}


# --- the document ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"", b"- a\n- b\n", b"3", b"just text", b"null", b"~"],
)
def test_a_document_that_is_not_a_mapping_is_refused(data: bytes) -> None:
    assert "mapping" in refusal(data)


def test_bytes_that_are_not_utf_8_are_refused() -> None:
    assert "UTF-8" in refusal(b"\xff\xfe\x00")


def test_yaml_that_does_not_parse_is_refused_without_quoting_it() -> None:
    message = refusal(b"drift: [zqxcanary,\n  - : x")
    assert "YAML" in message
    assert "zqxcanary" not in message


def test_the_loader_reads_nothing_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = load(document())
    for name in (
        "UV",
        "RINCHEN_CONFIG",
        "RINCHEN_MAINTAIN_TIMEOUT",
        "UV_ARGS",
        "CADENCE",
    ):
        monkeypatch.setenv(name, "1")
    assert load(document()) == expected


def test_a_loaded_configuration_is_frozen() -> None:
    config = load(document())
    with pytest.raises(ValueError):
        config.lock_check = config.lock_check.model_copy()  # type: ignore[misc]


def test_a_validated_configuration_survives_a_second_validation() -> None:
    config = load(document())
    assert MaintainConfig.model_validate(config.model_dump()) == config


def test_all_the_loader_refusals_are_config_errors_of_one_prefix() -> None:
    builders: list[Callable[[], bytes]] = [
        lambda: dump(edited("drift", "uv_args", value=["lock", "--upgrade"])),
        lambda: b"- a",
        lambda: b"\xff",
        lambda: dump(without("reruns")),
    ]
    for build in builders:
        assert refusal(build()).startswith("invalid maintain config")
