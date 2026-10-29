#!/usr/bin/env python
"""Upsert one platform's build record into manifests/pyonmttok_build.json."""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "manifests" / "pyonmttok_build.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(platform: str, wheel_path_str: str, extra: dict[str, str] | None = None) -> int:
    wheel_path = Path(wheel_path_str).resolve()
    if not wheel_path.is_file():
        print(f"wheel not found: {wheel_path}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    try:
        rel_wheel = wheel_path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel_wheel = wheel_path.name
    record = {
        "platform": platform,
        "wheel": rel_wheel,
        "sha256": sha256_of(wheel_path),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        record.update(extra)
    builds = [b for b in manifest.get("builds", []) if b["platform"] != platform]
    builds.append(record)
    manifest["builds"] = builds
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {platform} build: {record['wheel']} ({record['sha256'][:12]}...)")
    return 0


def _parse_extra(args: list[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            print(f"invalid extra field (expected key=value): {arg}", file=sys.stderr)
            raise SystemExit(2)
        key, _, value = arg.partition("=")
        extra[key] = value
    return extra


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    if len(sys.argv) < 3:
        print("usage: record_manifest.py <platform> <wheel_path> [key=value ...]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2], _parse_extra(sys.argv[3:])))
