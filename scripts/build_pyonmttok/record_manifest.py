#!/usr/bin/env python
"""Upsert one platform's build record into manifests/pyonmttok_build.json."""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "manifests" / "pyonmttok_build.json"

# Core fields main() itself computes; a `key=value` extra field may not
# override any of these (e.g. `sha256=anything` on the CLI must not replace
# the real computed hash).
RESERVED_FIELDS = frozenset({"platform", "wheel", "sha256", "built_at"})


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(
    platform: str,
    wheel_path_str: str,
    extra: dict[str, str] | None = None,
    manifest_path: Path | None = None,
) -> int:
    target_manifest = manifest_path if manifest_path is not None else MANIFEST_PATH
    wheel_path = Path(wheel_path_str).resolve()
    if not wheel_path.is_file():
        print(f"wheel not found: {wheel_path}", file=sys.stderr)
        return 1

    if extra:
        for key in extra:
            if not key:
                print("invalid extra field: empty key", file=sys.stderr)
                return 1
            if key in RESERVED_FIELDS:
                print(
                    f"invalid extra field: {key!r} is a reserved core field "
                    "and cannot be overridden",
                    file=sys.stderr,
                )
                return 1

    manifest = json.loads(target_manifest.read_text(encoding="utf-8"))
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
    target_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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


def _parse_cli(argv: list[str]) -> tuple[Path | None, str, str, dict[str, str]]:
    manifest_path: Path | None = None
    if argv[:1] == ["--manifest"]:
        if len(argv) < 2:
            print("--manifest requires a path argument", file=sys.stderr)
            raise SystemExit(2)
        manifest_path = Path(argv[1])
        argv = argv[2:]
    if len(argv) < 2:
        print(
            "usage: record_manifest.py [--manifest PATH] <platform> <wheel_path> [key=value ...]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return manifest_path, argv[0], argv[1], _parse_extra(argv[2:])


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    _manifest_path, _platform, _wheel, _extra = _parse_cli(sys.argv[1:])
    raise SystemExit(main(_platform, _wheel, _extra, _manifest_path))
