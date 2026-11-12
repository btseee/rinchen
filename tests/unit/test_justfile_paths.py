"""The paths the recipes of the justfile name exist in the repository, and the recipe `accept` points
at the Wiki page State, which holds the definition of done (the documents moved to the Wiki)."""

from __future__ import annotations

import re
from typing import Final

from rinchen.config.loader import REPO_ROOT

JUSTFILE: Final = REPO_ROOT / "justfile"
# A repository path: a token with a slash and a file suffix, outside a URL.
_PATH: Final = re.compile(
    r"(?<![\w/.:-])((?:[\w.-]+/)+[\w.-]+\.(?:md|ya?ml|py|toml))\b"
)


def recipe_lines() -> list[str]:
    return [
        line.strip()
        for line in JUSTFILE.read_text(encoding="utf-8").splitlines()
        if line.startswith("    ")
    ]


def test_every_repository_path_a_recipe_names_exists() -> None:
    paths = [found for line in recipe_lines() for found in _PATH.findall(line)]

    assert "docker/compose.serve.yaml" in paths
    assert [path for path in paths if not (REPO_ROOT / path).is_file()] == []


def test_the_path_pattern_sees_a_path_and_leaves_a_url_alone() -> None:
    assert _PATH.findall("see docs/runbooks/00_environment.md now") == [
        "docs/runbooks/00_environment.md"
    ]
    assert _PATH.findall("see https://example.org/a/b.md now") == []


def test_the_accept_recipe_points_at_the_state_page_of_the_wiki() -> None:
    lines = JUSTFILE.read_text(encoding="utf-8").splitlines()
    body = lines[lines.index("accept:") + 1].strip()

    assert body.startswith('echo "just accept: not implemented')
    assert "https://github.com/btseee/rinchen/wiki/State." in body
    assert "docs/" not in body
    assert "Research-" not in body
