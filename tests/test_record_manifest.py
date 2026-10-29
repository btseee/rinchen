import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "build_pyonmttok"))

import record_manifest  # noqa: E402


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps({"source": {"commit": "deadbeef"}, "builds": []}),
        encoding="utf-8",
    )


def test_sha256_of_matches_hashlib(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"hello pyonmttok")
    expected = hashlib.sha256(b"hello pyonmttok").hexdigest()
    assert record_manifest.sha256_of(sample) == expected


def test_main_appends_new_platform_record(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    wheel = tmp_path / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"fake wheel contents")

    assert record_manifest.main("win_amd64", str(wheel)) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["builds"]) == 1
    record = manifest["builds"][0]
    assert record["platform"] == "win_amd64"
    assert record["wheel"] == "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    assert record["sha256"] == hashlib.sha256(b"fake wheel contents").hexdigest()
    assert "built_at" in record


def test_main_replaces_existing_platform_record(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {"commit": "deadbeef"},
                "builds": [{"platform": "win_amd64", "wheel": "old.whl", "sha256": "old", "built_at": "old"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    wheel = tmp_path / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"new contents")

    assert record_manifest.main("win_amd64", str(wheel)) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["builds"]) == 1
    assert manifest["builds"][0]["sha256"] == hashlib.sha256(b"new contents").hexdigest()


def test_main_returns_1_when_wheel_missing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    assert record_manifest.main("win_amd64", str(tmp_path / "missing.whl")) == 1
