"""The shipped maintenance table `configs/maintain/maintain.yaml`, `reruns_due` and `rule_for`
(D14, spec M7, M8 and M9).

The table is read from the file's bytes and pinned literal by literal: it is a table of priors
written from the recorded decisions and one machine's times, and a change of one of its numbers
is a deliberate edit of this test. `rule_for` matches the first rule that names the package or
a prefix of it, after the name is put in PEP 503 form; a name that only starts like a prefix
without its separator (`torchvision` for `torch-*`) is not matched, and the test pins that.
"""

from __future__ import annotations

import re

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import (
    GATE,
    MaintainConfig,
    PinRule,
    load_maintain_config,
    reruns_due,
    rule_for,
)
from rinchen.errors import ConfigError
from rinchen.maintain.states import SKIP_GPU_CHECK
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
VERIFY = REPO_ROOT / "docs" / "VERIFY.md"


def three_reruns() -> MaintainConfig:
    """Four reruns: two nightly ones and two weekly ones, the first of them weekly."""
    pytest_rerun = {
        "kind": "pytest",
        "timeout_seconds": 600,
        "why": "a gate module",
        "targets": ["tests/integration/test_serving_gate.py"],
        "min_passed": 5,
    }
    export_rerun = {"kind": "export_parity", "timeout_seconds": 600, "why": "parity"}
    return MaintainConfig.model_validate(
        {
            "skipped_doctor_checks": {},
            "lock_check": {"uv_args": ["lock", "--check"], "timeout_seconds": 120},
            "drift": {
                "uv_args": ["lock", "--upgrade", "--dry-run"],
                "timeout_seconds": 300,
                "rules": [],
            },
            "reruns": [
                {"id": "first_weekly", "cadence": "weekly", **export_rerun},
                {"id": "parity_tiny", "cadence": "nightly", **pytest_rerun},
                {"id": "parity_export", "cadence": "nightly", **export_rerun},
                {"id": "heavy", "cadence": "weekly", **pytest_rerun},
            ],
        }
    )


def test_reruns_due_at_the_nightly_cadence_are_the_nightly_ones_in_file_order() -> None:
    due = reruns_due(three_reruns(), "nightly")
    assert isinstance(due, tuple)
    assert [rerun.id for rerun in due] == ["parity_tiny", "parity_export"]


def test_reruns_due_at_the_weekly_cadence_are_all_of_them_in_file_order() -> None:
    due = reruns_due(three_reruns(), "weekly")
    assert isinstance(due, tuple)
    assert [rerun.id for rerun in due] == [
        "first_weekly",
        "parity_tiny",
        "parity_export",
        "heavy",
    ]


@pytest.mark.parametrize("cadence", ["daily", "Nightly", "", "weekly ", "monthly"])
def test_reruns_due_for_another_cadence_is_a_config_error(cadence: str) -> None:
    with pytest.raises(ConfigError) as caught:
        reruns_due(three_reruns(), cadence)
    assert "cadence" in str(caught.value)
    assert "monthly" not in str(caught.value)


def rules_of(*entries: tuple[str, list[str]]) -> list[PinRule]:
    return [
        PinRule(id=name, packages=packages, decision="a decision", requires=[])
        for name, packages in entries
    ]


def test_the_first_rule_that_matches_wins() -> None:
    rules = rules_of(
        ("first", ["numpy", "torch-*"]), ("second", ["numpy", "torch-*", "x"])
    )
    for package in ("numpy", "torch-cuda"):
        found = rule_for(rules, package)
        assert found is not None
        assert found.id == "first"
    found = rule_for(rules, "x")
    assert found is not None
    assert found.id == "second"


def test_a_prefix_rule_needs_its_whole_prefix_and_a_name_that_only_starts_alike_is_not_matched() -> (
    None
):
    rules = rules_of(("torch-family", ["torch", "torch-*"]))
    assert rule_for(rules, "torchvision") is None
    assert rule_for(rules, "torchcodec") is None
    assert rule_for(rules, "tor") is None
    found = rule_for(rules, "torch-lightning")
    assert found is not None
    assert found.id == "torch-family"


def test_a_prefix_rule_without_a_separator_matches_every_name_that_starts_with_it() -> (
    None
):
    rules = rules_of(("t", ["torch*"]))
    assert rule_for(rules, "torchvision") is not None
    assert rule_for(rules, "torch") is not None
    assert rule_for(rules, "tor") is None


def test_a_package_name_is_normalised_before_the_match() -> None:
    rules = rules_of(("hub", ["huggingface-hub"]), ("nv", ["nvidia-*"]))
    found = rule_for(rules, "Huggingface_Hub")
    assert found is not None
    assert found.id == "hub"
    found = rule_for(rules, "NVIDIA_cudnn.cu13")
    assert found is not None
    assert found.id == "nv"


def test_a_rule_written_with_a_dot_matches_the_normalised_name() -> None:
    rules = rules_of(("zope", ["zope.interface", "a.b*"]))
    assert rule_for(rules, "zope-interface") is not None
    assert rule_for(rules, "zope.interface") is not None
    assert rule_for(rules, "a-bc") is not None


def test_no_rules_match_nothing() -> None:
    assert rule_for([], "numpy") is None
    assert rule_for(rules_of(("a", ["b"])), "") is None


def test_a_name_is_matched_whole() -> None:
    rules = rules_of(("a", ["numpy"]))
    assert rule_for(rules, "numpy2") is None
    assert rule_for(rules, "xnumpy") is None
    assert rule_for(rules, "num") is None


# --- the shipped table ----------------------------------------------------------------------------------------


def shipped() -> MaintainConfig:
    return load_maintain_config(SHIPPED.read_bytes())


EXPECTED_RULES = [
    ("eole", ["eole"], ["parity_tiny", "parity_export", "gate"]),
    ("ctranslate2", ["ctranslate2"], ["parity_tiny", "parity_export", "serving_tiny"]),
    (
        "torch-family",
        ["torch", "torch-*", "torchcodec", "triton", "nvidia-*", "cuda-*"],
        ["parity_tiny", "robustness_tiny", "gate"],
    ),
    ("sacrebleu-numpy", ["sacrebleu", "numpy"], ["numerics", "evaluation_tiny"]),
    ("sentencepiece", ["sentencepiece"], ["numerics", "gate"]),
    (
        "transformers-stack",
        ["transformers", "tokenizers", "safetensors", "sentence-transformers"],
        ["gate"],
    ),
    ("polars-arrow", ["polars", "polars-*", "pyarrow"], ["gate"]),
    (
        "serving-stack",
        ["fastapi", "starlette", "uvicorn", "h11"],
        ["serving_tiny", "gate"],
    ),
    (
        "hub-client",
        ["huggingface-hub", "datasets", "hf-xet"],
        ["evaluation_tiny", "gate"],
    ),
    ("vendored-wheels", ["pyonmttok", "fasttext-wheel", "fasttext-new"], ["gate"]),
]

NUMERICS_TARGETS = [
    "tests/unit/test_metrics_segment_stats.py",
    "tests/unit/test_metrics_corpus_scores.py",
    "tests/unit/test_metrics_bleu_chrf.py",
    "tests/property/test_metrics_segment_stats_property.py",
    "tests/unit/test_eval_bootstrap.py",
    "tests/unit/test_eval_bootstrap_reference.py",
    "tests/unit/test_eval_bootstrap_determinism.py",
    "tests/property/test_eval_bootstrap_property.py",
    "tests/unit/test_eval_golden.py",
    "tests/unit/test_eval_report_golden.py",
    "tests/unit/test_eval_suite_golden.py",
    "tests/unit/test_eval_significance.py",
    "tests/unit/test_eval_robust_numbers.py",
    "tests/unit/test_eval_robust_placeholders.py",
    "tests/unit/test_eval_robust_script.py",
    "tests/golden",
]

# id, kind, cadence, timeout, floor, targets
EXPECTED_RERUNS: list[tuple[str, str, str, int, int | None, list[str]]] = [
    ("numerics", "pytest", "nightly", 300, 1100, NUMERICS_TARGETS),
    (
        "parity_tiny",
        "pytest",
        "nightly",
        600,
        13,
        ["tests/integration/test_export_parity.py"],
    ),
    ("parity_export", "export_parity", "nightly", 600, None, []),
    (
        "robustness_tiny",
        "pytest",
        "weekly",
        1200,
        31,
        [
            "tests/integration/test_robustness_gate.py",
            "tests/integration/test_robustness_gate_refusals.py",
        ],
    ),
    (
        "evaluation_tiny",
        "pytest",
        "weekly",
        1200,
        99,
        [
            "tests/integration/test_eval_gate.py",
            "tests/integration/test_eval_gate_controls.py",
            "tests/integration/test_release_gate.py",
        ],
    ),
    (
        "serving_tiny",
        "pytest",
        "weekly",
        1000,
        55,
        [
            "tests/integration/test_serving_gate.py",
            "tests/integration/test_serving_gate_process.py",
            "tests/integration/test_serving_stop_budget.py",
            "tests/integration/test_engine_surfaces.py",
        ],
    ),
]


def test_the_shipped_table_has_the_ten_rules_in_order() -> None:
    config = shipped()
    assert [rule.id for rule in config.drift.rules] == [r[0] for r in EXPECTED_RULES]
    assert [
        (r.id, r.packages, r.requires) for r in config.drift.rules
    ] == EXPECTED_RULES


def test_the_shipped_table_has_the_six_reruns_in_order() -> None:
    config = shipped()
    assert [rerun.id for rerun in config.reruns] == [r[0] for r in EXPECTED_RERUNS]
    found = [
        (r.id, r.kind, r.cadence, r.timeout_seconds, r.min_passed, r.targets)
        for r in config.reruns
    ]
    assert found == EXPECTED_RERUNS
    assert all(rerun.max_skipped == 0 for rerun in config.reruns)


def test_the_shipped_table_skips_the_gpu_doctor_check_with_the_gpu_reason() -> None:
    assert shipped().skipped_doctor_checks == {"D-TORCH-CUDA": SKIP_GPU_CHECK}


def test_the_shipped_uv_argument_lists_and_bounds() -> None:
    config = shipped()
    assert config.drift.uv_args == ["lock", "--upgrade", "--dry-run"]
    assert config.drift.timeout_seconds == 300
    assert config.lock_check.uv_args == ["lock", "--check"]
    assert config.lock_check.timeout_seconds == 120


def test_every_pytest_target_of_the_shipped_table_exists() -> None:
    for rerun in shipped().reruns:
        for target in rerun.targets:
            assert (REPO_ROOT / target).exists(), target


def test_every_shipped_rerun_and_rule_says_why_in_one_line() -> None:
    config = shipped()
    for rerun in config.reruns:
        assert rerun.why.strip() == rerun.why
    for rule in config.drift.rules:
        assert rule.decision.strip() == rule.decision
        assert (
            "Wiki page Decisions" in rule.decision
            or "Wiki page Verification-Log" in rule.decision
        ), rule.id
        assert ".md" not in rule.decision, rule.id


def test_every_id_of_a_recorded_row_that_a_decision_names_is_in_the_verify_table() -> (
    None
):
    verify = VERIFY.read_text(encoding="utf-8")
    named = 0
    for rule in shipped().drift.rules:
        for row in re.findall(
            r"\bD(?:[0-9]+)?-[A-Z0-9]+(?:-[A-Z0-9]+)*", rule.decision
        ):
            named += 1
            assert f"| {row} " in verify, (rule.id, row)
    assert named >= 10


def test_every_requires_of_the_shipped_table_names_a_rerun_or_the_gate() -> None:
    config = shipped()
    known = {rerun.id for rerun in config.reruns} | {GATE}
    for rule in config.drift.rules:
        assert set(rule.requires) <= known, rule.id


def test_no_shipped_target_is_a_test_of_the_maintenance_check_itself() -> None:
    for rerun in shipped().reruns:
        assert all("maintain" not in target.casefold() for target in rerun.targets)


def test_the_shipped_reruns_due_at_each_cadence() -> None:
    config = shipped()
    assert [r.id for r in reruns_due(config, "nightly")] == [
        "numerics",
        "parity_tiny",
        "parity_export",
    ]
    assert [r.id for r in reruns_due(config, "weekly")] == [
        r[0] for r in EXPECTED_RERUNS
    ]
    with pytest.raises(ConfigError):
        reruns_due(config, "daily")


SHIPPED_LOOKUPS: list[tuple[str, str | None]] = [
    ("numpy", "sacrebleu-numpy"),
    ("sacrebleu", "sacrebleu-numpy"),
    ("nvidia-cudnn-cu13", "torch-family"),
    ("nvidia-nccl-cu13", "torch-family"),
    ("cuda-bindings", "torch-family"),
    ("torch", "torch-family"),
    ("torchcodec", "torch-family"),
    ("triton", "torch-family"),
    ("torch-lightning", "torch-family"),
    ("torchvision", None),
    ("huggingface-hub", "hub-client"),
    ("Huggingface_Hub", "hub-client"),
    ("hf-xet", "hub-client"),
    ("eole", "eole"),
    ("ctranslate2", "ctranslate2"),
    ("sentencepiece", "sentencepiece"),
    ("sentence-transformers", "transformers-stack"),
    ("polars-u64-idx", "polars-arrow"),
    ("pyarrow", "polars-arrow"),
    ("uvicorn", "serving-stack"),
    ("pyonmttok", "vendored-wheels"),
    ("fasttext-wheel", "vendored-wheels"),
    ("rich", None),
    ("tomli-w", None),
    ("sortedcontainers", None),
]


@pytest.mark.parametrize(("package", "rule"), SHIPPED_LOOKUPS)
def test_the_shipped_rules_find_the_package_they_pin(
    package: str, rule: str | None
) -> None:
    found = rule_for(shipped().drift.rules, package)
    assert (found.id if found else None) == rule


def test_the_shipped_file_says_in_its_header_that_its_numbers_are_priors() -> None:
    header = SHIPPED.read_text(encoding="utf-8").split("\n\n", 1)[0]
    assert header.startswith("#")
    assert "unvalidated prior" in re.sub(r"\s*#\s*", " ", header).lower()


def test_the_shipped_file_is_ascii_text_of_modest_size() -> None:
    data = SHIPPED.read_bytes()
    assert data.isascii()
    assert data.endswith(b"\n")
    assert len(data.splitlines()) < 200
