"""The configuration of the evaluation metrics (D13, T9 and T13): the bootstrap settings, the
metrics a comparison reports and the release targets, as data. The shipped table
`configs/eval/metrics.yaml` loads and says what the spec says; every violation of the schema is
a `ConfigError` that names the field."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from rinchen.config.eval_metrics import (
    EvalMetricsConfig,
    TargetConfig,
    load_eval_metrics_config,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from tests.support.module_imports import (
    loaded_by_importing,
    project_imports,
    source_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "eval" / "metrics.yaml"


def document() -> dict[str, Any]:
    """A valid document, as a dict the test then breaks in one place."""
    return {
        "bootstrap": {"resamples": 1000, "seed": 20261110, "alpha": 0.05},
        "primary_metric": "chrf_pp",
        "comparison_metrics": ["chrf_pp", "bleu"],
        "baseline_system": None,
        "targets": [
            {
                "id": "beats_baseline",
                "description": "better than the baseline",
                "kind": "beats_baseline",
                "min_directions": 12,
                "of_directions": 16,
            },
            {
                "id": "off_target",
                "description": "off-target share",
                "kind": "off_target",
                "threshold": 0.01,
            },
            {
                "id": "numbers",
                "description": "numbers kept",
                "kind": "robustness_value",
                "key": "numbers_acc",
                "threshold": 0.99,
            },
        ],
    }


def load(value: dict[str, Any] | str | bytes) -> EvalMetricsConfig:
    if isinstance(value, dict):
        value = yaml.safe_dump(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return load_eval_metrics_config(data)


def refused(value: dict[str, Any] | str | bytes) -> str:
    """The message of the ConfigError a document is refused with."""
    with pytest.raises(ConfigError) as caught:
        load(value)
    return str(caught.value)


def with_change(path: str, value: object) -> dict[str, Any]:
    """The valid document with the entry at `path` (dotted; digits index a list) replaced; a
    `...` value removes it."""
    data = document()
    node: Any = data
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    key: Any = int(last) if isinstance(node, list) else last
    if value is ...:
        del node[key]
    else:
        node[key] = value
    return data


# ---- the shipped table ------------------------------------------------------------------


def test_the_shipped_table_is_the_one_of_the_spec() -> None:
    config = load(SHIPPED.read_bytes())

    assert config.model_dump() == {
        "bootstrap": {"resamples": 1000, "seed": 20261110, "alpha": 0.05},
        "primary_metric": "chrf_pp",
        "comparison_metrics": ["chrf_pp", "bleu"],
        "baseline_system": None,
        "targets": [
            {
                "id": "beats_baseline",
                "description": config.targets[0].description,
                "kind": "beats_baseline",
                "threshold": None,
                "min_directions": 12,
                "of_directions": 16,
                "key": None,
            },
            {
                "id": "off_target",
                "description": config.targets[1].description,
                "kind": "off_target",
                "threshold": 0.01,
                "min_directions": None,
                "of_directions": None,
                "key": None,
            },
            {
                "id": "numbers",
                "description": config.targets[2].description,
                "kind": "robustness_value",
                "threshold": 0.99,
                "min_directions": None,
                "of_directions": None,
                "key": "numbers_acc",
            },
            {
                "id": "placeholders",
                "description": config.targets[3].description,
                "kind": "robustness_value",
                "threshold": 1.0,
                "min_directions": None,
                "of_directions": None,
                "key": "placeholder_acc",
            },
        ],
    }
    assert all(target.description.strip() for target in config.targets)


def test_the_shipped_table_says_that_every_number_is_an_unvalidated_prior() -> None:
    header = SHIPPED.read_text(encoding="utf-8").split("\nbootstrap:")[0]

    assert "unvalidated prior" in header


def test_the_shipped_table_names_no_baseline_because_none_was_ever_run() -> None:
    assert load(SHIPPED.read_bytes()).baseline_system is None


# ---- what a valid document gives --------------------------------------------------------


def test_a_valid_document_loads_and_is_frozen() -> None:
    config = load(document())

    assert config.bootstrap.resamples == 1000
    assert isinstance(config.targets[0], TargetConfig)
    with pytest.raises(ValueError, match="frozen"):
        config.primary_metric = "chrf_pp"  # type: ignore[misc]


def test_a_baseline_system_is_an_id_or_null() -> None:
    assert (
        load(with_change("baseline_system", "nllb-1.3b")).baseline_system == "nllb-1.3b"
    )
    assert load(with_change("baseline_system", None)).baseline_system is None


def test_the_document_is_read_from_the_bytes_it_is_given_and_nowhere_else() -> None:
    other = with_change("bootstrap.seed", 7)

    assert load_eval_metrics_config(yaml.safe_dump(other).encode()).bootstrap.seed == 7


# ---- every violation names its field ----------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "named"),
    [
        ("bootstrap.resamples", 99, "bootstrap.resamples"),
        ("bootstrap.resamples", "1000", "bootstrap.resamples"),
        ("bootstrap.resamples", 1000.5, "bootstrap.resamples"),
        ("bootstrap.resamples", True, "bootstrap.resamples"),
        ("bootstrap.alpha", 0, "bootstrap.alpha"),
        ("bootstrap.alpha", 0.0, "bootstrap.alpha"),
        ("bootstrap.alpha", 0.5, "bootstrap.alpha"),
        ("bootstrap.alpha", 1, "bootstrap.alpha"),
        ("bootstrap.alpha", -0.05, "bootstrap.alpha"),
        ("bootstrap.alpha", float("nan"), "bootstrap.alpha"),
        ("bootstrap.alpha", "0.05", "bootstrap.alpha"),
        ("bootstrap.seed", -1, "bootstrap.seed"),
        ("bootstrap.seed", True, "bootstrap.seed"),
        ("bootstrap.seed", 1.0, "bootstrap.seed"),
        ("bootstrap.seed", None, "bootstrap.seed"),
        ("bootstrap", ..., "bootstrap"),
        ("bootstrap.seed", ..., "bootstrap.seed"),
        ("bootstrap.surprise", 1, "bootstrap.surprise"),
        ("surprise", 1, "surprise"),
        ("primary_metric", "bleu", "primary_metric"),
        ("primary_metric", ..., "primary_metric"),
        ("comparison_metrics", ..., "comparison_metrics"),
        ("comparison_metrics", ["chrf_pp", "comet22"], "comparison_metrics.1"),
        ("comparison_metrics", ["chrf_pp", "metricx24"], "comparison_metrics.1"),
        ("comparison_metrics", ["chrf_pp", "blue"], "comparison_metrics.1"),
        ("comparison_metrics", [], "comparison_metrics"),
        ("comparison_metrics", ["bleu", "bleu"], "comparison_metrics"),
        ("comparison_metrics", "bleu", "comparison_metrics"),
        ("baseline_system", ..., "baseline_system"),
        ("baseline_system", "", "baseline_system"),
        ("baseline_system", "a b", "baseline_system"),
        ("baseline_system", "../x", "baseline_system"),
        ("baseline_system", 7, "baseline_system"),
        ("targets", ..., "targets"),
        ("targets", "none", "targets"),
        ("targets.0.kind", "wins", "targets.0.kind"),
        ("targets.0.kind", ..., "targets.0.kind"),
        ("targets.0.id", "", "targets.0.id"),
        ("targets.0.id", "has space", "targets.0.id"),
        ("targets.0.description", "  ", "targets.0.description"),
        ("targets.0.description", ..., "targets.0.description"),
        ("targets.0.surprise", 1, "targets.0.surprise"),
        ("targets.0.min_directions", ..., "min_directions"),
        ("targets.0.of_directions", ..., "of_directions"),
        ("targets.0.min_directions", 0, "targets.0.min_directions"),
        ("targets.0.min_directions", 17, "min_directions"),
        ("targets.0.of_directions", True, "targets.0.of_directions"),
        ("targets.0.threshold", 0.5, "threshold"),
        ("targets.0.key", "numbers_acc", "key"),
        ("targets.1.threshold", ..., "threshold"),
        ("targets.1.threshold", -0.1, "targets.1.threshold"),
        ("targets.1.threshold", float("inf"), "targets.1.threshold"),
        ("targets.1.threshold", "0.01", "targets.1.threshold"),
        ("targets.1.threshold", True, "targets.1.threshold"),
        ("targets.1.key", "numbers_acc", "key"),
        ("targets.1.min_directions", 1, "min_directions"),
        ("targets.2.key", ..., "key"),
        ("targets.2.key", "Numbers Acc", "targets.2.key"),
        ("targets.2.threshold", ..., "threshold"),
        ("targets.2.of_directions", 16, "of_directions"),
    ],
)
def test_every_violation_is_a_config_error_that_names_the_field(
    path: str, value: object, named: str
) -> None:
    assert named in refused(with_change(path, value))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("bootstrap.resamples", 100),
        ("bootstrap.seed", 0),
        ("bootstrap.alpha", 0.0001),
        ("bootstrap.alpha", 0.4999),
        ("targets.0.min_directions", 1),
        ("targets.0.min_directions", 16),
        ("targets.1.threshold", 0),
        ("targets.1.threshold", 1),
        ("targets.2.threshold", 0.0),
        ("comparison_metrics", ["chrf"]),
        ("comparison_metrics", ["bleu", "chrf", "chrf_pp"]),
    ],
)
def test_the_value_at_the_edge_of_a_bound_is_accepted(path: str, value: object) -> None:
    load(with_change(path, value))


def test_a_duplicate_target_id_is_refused_by_name() -> None:
    data = document()
    data["targets"][1]["id"] = "beats_baseline"

    message = refused(data)

    assert "targets" in message
    assert "beats_baseline" in message


# ---- what is not a document -------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["", "[]", "- a\n- b\n", "just text", "null", "42"], ids=repr
)
def test_a_document_that_is_not_a_mapping_is_refused(text: str) -> None:
    assert "mapping" in refused(text)


def test_a_document_that_is_not_yaml_is_refused() -> None:
    assert "parse" in refused("bootstrap: [unclosed")


def test_a_document_that_is_not_utf_8_is_refused() -> None:
    assert "UTF-8" in refused(b"bootstrap: \xff\xfe")


def test_nothing_is_taken_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SACREBLEU_SEED", "7")
    monkeypatch.setenv("RINCHEN_EVAL_SEED", "7")

    assert load(document()).bootstrap.seed == 20261110


# ---- layering ---------------------------------------------------------------------------


def test_the_module_imports_of_the_project_only_the_registry_and_the_errors() -> None:
    assert project_imports(source_of("rinchen.config.eval_metrics")) == {
        "rinchen.errors",
        "rinchen.metrics.registry",
    }


def test_importing_the_configuration_loads_no_model_library_and_no_metric_engine() -> (
    None
):
    loaded = loaded_by_importing(
        ["rinchen.config.eval_metrics"],
        ("ctranslate2", "sentencepiece", "eole", "torch", "polars", "sacrebleu"),
    )

    assert loaded == []
