"""Guard against vendor/wheels/*.whl drifting from manifests/pyonmttok_build.json
or uv.lock.

`uv lock --check` does not catch a hand-corrupted or stale hash in uv.lock
for a local path dependency (it trusts the lockfile's recorded hash rather
than re-hashing the file on disk), so nothing else in this repo's tooling
would notice if a wheel were rebuilt (or corrupted) without updating both
of these downstream records. This test recomputes the real sha256 of every
committed wheel and cross-checks it against both places that also claim to
know that hash.
"""

import hashlib
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_WHEELS_DIR = REPO_ROOT / "vendor" / "wheels"
MANIFEST_PATH = REPO_ROOT / "manifests" / "pyonmttok_build.json"
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


def test_vendor_wheels_match_manifest_sha256():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    builds_by_wheel = {Path(build["wheel"]).name: build for build in manifest["builds"]}

    for wheel in _vendor_wheels():
        build = builds_by_wheel.get(wheel.name)
        assert build is not None, f"{wheel.name} has no manifests/pyonmttok_build.json build entry"
        assert build["sha256"] == _sha256_of(wheel), (
            f"{wheel.name}: manifests/pyonmttok_build.json sha256 does not "
            "match the real file contents -- the wheel was rebuilt (or "
            "edited) without re-running record_manifest.py"
        )


def test_vendor_wheels_match_uv_lock_hash():
    lock = tomllib.loads(UV_LOCK_PATH.read_text(encoding="utf-8"))
    pyonmttok_packages = [pkg for pkg in lock["package"] if pkg.get("name") == "pyonmttok"]

    for wheel in _vendor_wheels():
        rel_path = f"vendor/wheels/{wheel.name}"
        matching_packages = [
            pkg for pkg in pyonmttok_packages if pkg.get("source", {}).get("path") == rel_path
        ]
        assert matching_packages, f"uv.lock has no pyonmttok [[package]] entry with source path {rel_path}"

        expected_hash = f"sha256:{_sha256_of(wheel)}"
        for pkg in matching_packages:
            wheel_entries = [w for w in pkg.get("wheels", []) if w.get("filename") == wheel.name]
            assert wheel_entries, (
                f"uv.lock pyonmttok entry for {rel_path} has no wheels[] entry named {wheel.name}"
            )
            for entry in wheel_entries:
                assert entry["hash"] == expected_hash, (
                    f"uv.lock hash for {wheel.name} does not match the real "
                    "file contents -- the wheel was rebuilt (or edited) "
                    "without regenerating uv.lock"
                )
