"""Control profiles for `rinchen tiny run` (D15 Task 7): a copy of `configs/tiny/tiny.yaml` with named
values changed, for the tests of the command, the scale-up test and the controls of the gate.

`control_profile(tmp_path, train_steps=400)` writes `<tmp_path>/<name>.yaml` and returns its path. An
override is a dotted key (`**{"train.steps": 400}`), a key of the top level (`budget_seconds=5`,
`qe_fake=False`) or a keyword in which the first underscore separates the section from its key
(`train_steps=400` is `train.steps`, `evaluation_resamples=200` is `evaluation.resamples`). Every path of
the shipped profile is made absolute (the profile is read from a temporary directory, and a relative path
means the repository root only for the profiles of the repository), the digests are kept as they are, and
the result is loaded through `load_tiny_config` unless `validate=False`.

When `fixtures_directory` names a copy of the fixtures (`copy_fixtures`) and `registry` is not given, the
profile gets a registry file of its own, beside it, whose `local://` entries name that copy: the fetch
stage accepts only a registry that lists the fixtures of the profile's own directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Final

import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import load_tiny_config
from rinchen.tiny.fixtures import FIXTURE_FILES

SHIPPED: Final = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"
FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
# The keys that hold a path; a relative one is anchored at the repository root.
PATH_KEYS: Final = (
    ("fixtures", "directory"),
    ("registry",),
    ("converter", "pairs"),
    ("train", "base"),
    ("serve", "config"),
)
LOCAL_SCHEME: Final = "local://"


def copy_fixtures(directory: Path) -> Path:
    """Copy the seven fixture files into `directory` (made) and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_FILES:
        shutil.copyfile(FIXTURES / name, directory / name)
    return directory


def _keys_of(override: str, document: dict[str, Any]) -> list[str]:
    """The path of keys an override names: a dotted key as it is, a key of the top level as it is,
    any other keyword split at its first underscore."""
    if "." in override:
        return override.split(".")
    if override in document:
        return [override]
    section, _, rest = override.partition("_")
    return [section, rest]


def _parent(document: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    parent = document
    for key in keys[:-1]:
        parent = parent[key]
    return parent


def _absolute(document: dict[str, Any]) -> None:
    for keys in PATH_KEYS:
        parent = _parent(document, list(keys))
        value = Path(parent[keys[-1]])
        parent[keys[-1]] = str(value if value.is_absolute() else REPO_ROOT / value)


def _write_registry(path: Path, fixtures_directory: Path) -> None:
    """The shipped registry with every `local://` URL naming the file of the same name in
    `fixtures_directory`."""
    shipped = yaml.safe_load(
        (REPO_ROOT / "configs" / "data" / "sources.tiny_parallel.yaml").read_text(
            encoding="utf-8"
        )
    )
    for entry in shipped["sources"]:
        entry["url"] = f"{LOCAL_SCHEME}{fixtures_directory / Path(entry['url']).name}"
    path.write_text(yaml.safe_dump(shipped, sort_keys=False), encoding="utf-8")


def control_profile(
    tmp_path: Path,
    *,
    name: str = "control",
    validate: bool = True,
    **overrides: Any,
) -> Path:
    """The file `<tmp_path>/<name>.yaml`: the shipped profile with `overrides` applied (see the
    module's text).

    Raises:
        ConfigError: the result does not load (only with `validate`).
        KeyError: an override names a section or a key the profile does not have.
    """
    document = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    _absolute(document)
    named = set()
    for override, value in overrides.items():
        keys = _keys_of(override, document)
        parent = _parent(document, keys)
        if keys[-1] not in parent:
            raise KeyError(override)
        parent[keys[-1]] = value
        named.add(keys[0])
    fixtures_directory = Path(document["fixtures"]["directory"])
    if fixtures_directory != FIXTURES and "registry" not in named:
        registry = tmp_path / f"{name}_sources.yaml"
        _write_registry(registry, fixtures_directory)
        document["registry"] = str(registry)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    if validate:
        load_tiny_config(path.read_bytes())
    return path
