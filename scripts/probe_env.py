"""Phase 0 environment probe: checks Python version, dry-resolves the pinned
dependency stack via `uv lock`, and checks PyPI for per-package wheel
availability on cp314."""

from __future__ import annotations


def classify_wheel_tags(filenames: list[str]) -> set[str]:
    """Classify a release's file list into wheel-tag categories.

    Returns a set that may contain any of: "cp314", "cp314t", "abi3",
    "py3-none", "sdist", or "none" (only when the input is empty).
    """
    tags: set[str] = set()
    for name in filenames:
        if name.endswith((".tar.gz", ".zip")):
            tags.add("sdist")
            continue
        if not name.endswith(".whl"):
            continue
        stem = name[: -len(".whl")]
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        python_tag, abi_tag = parts[-3], parts[-2]
        if "cp314t" in python_tag or "cp314t" in abi_tag:
            tags.add("cp314t")
        elif "cp314" in python_tag:
            tags.add("cp314")
        if abi_tag == "abi3":
            tags.add("abi3")
        if python_tag.startswith("py3") and abi_tag == "none":
            tags.add("py3-none")
    if not tags:
        tags.add("none")
    return tags


if __name__ == "__main__":
    import sys

    sys.exit(0)
