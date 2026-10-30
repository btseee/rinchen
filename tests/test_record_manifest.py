import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "build_pyonmttok"))

import record_manifest


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
                "builds": [
                    {
                        "platform": "win_amd64",
                        "wheel": "old.whl",
                        "sha256": "old",
                        "built_at": "old",
                    }
                ],
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
    assert (
        manifest["builds"][0]["sha256"] == hashlib.sha256(b"new contents").hexdigest()
    )


def test_main_returns_1_when_wheel_missing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    assert record_manifest.main("win_amd64", str(tmp_path / "missing.whl")) == 1


def test_main_merges_extra_provenance_fields(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    wheel = tmp_path / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"fake wheel contents")

    extra = {"cpython_version": "3.14.7", "cmake_version": "3.30.1"}
    assert record_manifest.main("win_amd64", str(wheel), extra) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["builds"][0]
    assert record["cpython_version"] == "3.14.7"
    assert record["cmake_version"] == "3.30.1"


def test_main_falls_back_to_wheel_name_when_outside_repo_root(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    # REPO_ROOT points at a directory that does NOT contain the wheel, so
    # Path.relative_to raises ValueError and main() must fall back to the
    # wheel's bare filename instead of a repo-relative path.
    other_root = tmp_path / "unrelated_repo_root"
    other_root.mkdir()
    monkeypatch.setattr(record_manifest, "REPO_ROOT", other_root)

    wheel_dir = tmp_path / "elsewhere"
    wheel_dir.mkdir()
    wheel = wheel_dir / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"fake wheel contents")

    assert record_manifest.main("win_amd64", str(wheel)) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["builds"][0]
    assert record["wheel"] == "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"


def test_main_rejects_reserved_field_override(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    wheel = tmp_path / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"fake wheel contents")

    real_sha256 = hashlib.sha256(b"fake wheel contents").hexdigest()
    assert record_manifest.main("win_amd64", str(wheel), {"sha256": "clobbered"}) == 1

    # The manifest must not have been written with the overridden value.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["builds"] == []
    # And the real hash, if it had been recorded, would differ from the
    # rejected override -- sanity check the test itself isn't vacuous.
    assert real_sha256 != "clobbered"


def test_main_rejects_empty_extra_field_key(tmp_path, monkeypatch):
    manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    wheel = tmp_path / "pyonmttok-1.38.1-cp314-cp314-win_amd64.whl"
    wheel.write_bytes(b"fake wheel contents")

    assert record_manifest.main("win_amd64", str(wheel), {"": "x"}) == 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["builds"] == []


def test_main_writes_to_custom_manifest_path(tmp_path, monkeypatch):
    default_manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(default_manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", default_manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    custom_manifest_path = tmp_path / "fasttext_wheel_build.json"
    _write_manifest(custom_manifest_path)

    wheel = tmp_path / "fasttext_wheel-0.9.2-py3-none-any.whl"
    wheel.write_bytes(b"fake shim wheel")

    assert (
        record_manifest.main(
            "universal", str(wheel), manifest_path=custom_manifest_path
        )
        == 0
    )

    # The custom manifest got the record...
    custom_manifest = json.loads(custom_manifest_path.read_text(encoding="utf-8"))
    assert len(custom_manifest["builds"]) == 1
    assert custom_manifest["builds"][0]["platform"] == "universal"

    # ...and the default manifest was left untouched.
    default_manifest = json.loads(default_manifest_path.read_text(encoding="utf-8"))
    assert default_manifest["builds"] == []


def test_parse_cli_extracts_manifest_flag(tmp_path):
    manifest_path, platform, wheel, extra = record_manifest._parse_cli(
        ["--manifest", str(tmp_path / "custom.json"), "universal", "x.whl", "k=v"]
    )
    assert manifest_path == tmp_path / "custom.json"
    assert platform == "universal"
    assert wheel == "x.whl"
    assert extra == {"k": "v"}


def test_parse_cli_without_manifest_flag_defaults_to_none():
    manifest_path, platform, wheel, extra = record_manifest._parse_cli(
        ["win_amd64", "x.whl"]
    )
    assert manifest_path is None
    assert platform == "win_amd64"
    assert wheel == "x.whl"
    assert extra == {}
