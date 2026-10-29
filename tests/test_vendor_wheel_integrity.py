"""Guard against vendor/wheels/*.whl drifting from manifests/*_build.json or
uv.lock.

`uv lock --check` does not catch a hand-corrupted or stale hash in uv.lock
for a local path dependency (it trusts the lockfile's recorded hash rather
than re-hashing the file on disk), so nothing else in this repo's tooling
would notice if a wheel were rebuilt (or corrupted) without updating both
of these downstream records. This test recomputes the real sha256 of every
committed wheel and cross-checks it against both places that also claim to
know that hash.

Nothing here is specific to one vendored package: every
`manifests/<name>_build.json` (currently `pyonmttok_build.json` and
`fasttext_wheel_build.json`) is read, and uv.lock entries are matched by the
wheel's own path/filename rather than by package name. A future vendored
wheel is covered automatically once it is committed under `vendor/wheels/`,
recorded in its own `manifests/<name>_build.json`, and wired into uv.lock.
"""

import hashlib
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_WHEELS_DIR = REPO_ROOT / "vendor" / "wheels"
MANIFESTS_DIR = REPO_ROOT / "manifests"
UV_LOCK_PATH = REPO_ROOT / "uv.lock"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vendor_wheels() -> list[Path]:
    wheels = sorted(VENDOR_WHEELS_DIR.glob("*.whl"))
    assert wheels, f"no wheels found in {VENDOR_WHEELS_DIR}"
    return wheels


def _manifest_builds_by_wheel() -> dict[str, list[tuple[str, dict]]]:
    """Map each recorded wheel filename to every (manifest filename, build
    entry) pair that claims it, across all manifests/*_build.json files."""
    manifest_paths = sorted(MANIFESTS_DIR.glob("*_build.json"))
    assert manifest_paths, f"no *_build.json manifests found in {MANIFESTS_DIR}"

    builds_by_wheel: dict[str, list[tuple[str, dict]]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for build in manifest["builds"]:
            wheel_name = Path(build["wheel"]).name
            builds_by_wheel.setdefault(wheel_name, []).append((manifest_path.name, build))
    return builds_by_wheel


def test_vendor_wheels_match_manifest_sha256():
    builds_by_wheel = _manifest_builds_by_wheel()

    for wheel in _vendor_wheels():
        claims = builds_by_wheel.get(wheel.name, [])
        assert len(claims) == 1, (
            f"{wheel.name} must have exactly one build entry across "
            f"manifests/*_build.json, found {len(claims)}: "
            f"{[manifest_name for manifest_name, _ in claims]}"
        )
        manifest_name, build = claims[0]
        assert build["sha256"] == _sha256_of(wheel), (
            f"{wheel.name}: manifests/{manifest_name} sha256 does not match "
            "the real file contents -- the wheel was rebuilt (or edited) "
            "without re-running record_manifest.py"
        )


def test_manifest_build_entries_point_at_committed_wheels():
    for wheel_name, claims in _manifest_builds_by_wheel().items():
        for manifest_name, build in claims:
            assert build["wheel"] == f"vendor/wheels/{wheel_name}", (
                f"manifests/{manifest_name} records wheel path {build['wheel']!r}, "
                f"expected 'vendor/wheels/{wheel_name}'"
            )
            assert (VENDOR_WHEELS_DIR / wheel_name).is_file(), (
                f"manifests/{manifest_name} records {build['wheel']}, but no "
                "such wheel is committed -- it was removed or renamed without "
                "updating the manifest"
            )


def test_vendor_wheels_match_uv_lock_hash():
    lock = tomllib.loads(UV_LOCK_PATH.read_text(encoding="utf-8"))

    for wheel in _vendor_wheels():
        rel_path = f"vendor/wheels/{wheel.name}"
        matching_packages = [
            pkg for pkg in lock["package"] if pkg.get("source", {}).get("path") == rel_path
        ]
        assert matching_packages, f"uv.lock has no [[package]] entry with source path {rel_path}"

        expected_hash = f"sha256:{_sha256_of(wheel)}"
        for pkg in matching_packages:
            wheel_entries = [w for w in pkg.get("wheels", []) if w.get("filename") == wheel.name]
            assert wheel_entries, (
                f"uv.lock {pkg['name']} entry for {rel_path} has no wheels[] "
                f"entry named {wheel.name}"
            )
            for entry in wheel_entries:
                assert entry["hash"] == expected_hash, (
                    f"uv.lock hash for {wheel.name} does not match the real "
                    "file contents -- the wheel was rebuilt (or edited) "
                    "without regenerating uv.lock"
                )
