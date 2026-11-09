"""The three serve configs the repository ships (D11, T18): each loads through the real loader,
and the two that no measurement backs say so in their header."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.serve import ServeConfig, load_serve_config

SERVE_DIR = REPO_ROOT / "configs" / "serve"
SHIPPED = ("tiny", "server", "edge")
PRIORS = ("server", "edge")


def header_of(path: Path) -> str:
    """The comment block at the top of a YAML file, without the hashes."""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(lines)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_config_loads_through_the_loader(name: str) -> None:
    config = load_serve_config(SERVE_DIR / f"{name}.yaml", env={})

    assert isinstance(config, ServeConfig)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_config_states_every_field_it_could(name: str) -> None:
    document = yaml.safe_load((SERVE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))

    missing = set(ServeConfig.model_fields) - set(document) - {"script_converter"}

    assert missing == set(), f"{name}.yaml leaves {sorted(missing)} to a default"


@pytest.mark.parametrize("name", PRIORS)
def test_the_priors_say_in_their_header_that_they_are_unmeasured(name: str) -> None:
    header = header_of(SERVE_DIR / f"{name}.yaml")

    assert "unmeasured" in header.lower()
    assert "prior" in header.lower()


def test_the_tiny_config_says_it_is_for_the_tiny_export() -> None:
    header = header_of(SERVE_DIR / "tiny.yaml").lower()

    assert "tiny" in header
    assert "toy" in header


def test_the_gpu_server_prior_asks_for_cuda_and_says_it_never_ran() -> None:
    config = load_serve_config(SERVE_DIR / "server.yaml", env={})

    assert config.device == "cuda"
    assert "gpu" in header_of(SERVE_DIR / "server.yaml").lower()


def test_the_edge_and_tiny_configs_run_on_the_cpu() -> None:
    for name in ("edge", "tiny"):
        assert load_serve_config(SERVE_DIR / f"{name}.yaml", env={}).device == "cpu"


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_config_never_asks_for_automatic_choices(name: str) -> None:
    config = load_serve_config(SERVE_DIR / f"{name}.yaml", env={})

    assert config.device in ("cpu", "cuda")
    assert config.compute_type != "auto"
