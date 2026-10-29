# Build pyonmttok from source for CPython 3.14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce working `pyonmttok` 1.38.1 wheels for CPython 3.14 on both
win_amd64 and manylinux_2_28_x86_64, wire them into the project via
`[tool.uv.sources]`, and get a real `uv lock` to succeed — closing the sole
remaining blocker to `eole` importing on Python 3.14.

**Architecture:** Clone `OpenNMT/Tokenizer` at the exact commit PyPI's
`1.38.1` release was built from. Build the C++ tokenizer library and its
pybind11 Python extension natively per platform (MSVC + prebuilt ICU on
Windows; manylinux container + ICU-from-source on Linux), verify each wheel
with a tokenize/detokenize round-trip, commit both wheels under
`vendor/wheels/`, and point `pyproject.toml` at them with platform markers.

**Tech Stack:** CMake, MSVC (Windows), Docker + `quay.io/pypa/manylinux_2_28_x86_64`
(Linux), pybind11 3.1.0, `auditwheel`, Python 3.14.7, `uv`.

## Global Constraints

- Every Python invocation runs on CPython 3.14.x (`python -c "import sys;
  assert sys.version_info[:2]==(3,14)"`), per `CLAUDE.md`. No 3.12/3.13
  fallback environment, container, or conda env — the manylinux container's
  own cp314 interpreter is still CPython 3.14, not a fallback.
- Source is pinned to `OpenNMT/Tokenizer` tag `v1.38.1`, commit
  `ae4b83b0b2af2172ed306c372835e84b51d5d77e` (verified against the PyPI
  `pyonmttok==1.38.1` release). Every build step verifies this commit before
  building — never build against whatever `HEAD` happens to be.
- ICU version is pinned to `73.2` (matches upstream's own
  `prepare_build_environment_{windows,linux}.sh`). Both download URLs were
  verified live during planning (HTTP 200):
  `https://github.com/unicode-org/icu/releases/download/release-73-2/icu4c-73_2-Win64-MSVC2019.zip`
  and `.../icu4c-73_2-src.tgz`.
- pybind11 version is pinned to `3.1.0` (PyPI's latest; classifiers
  explicitly list `Programming Language :: Python :: 3.14`). Satisfies
  `pyonmttok`'s `pybind11>=2.10.4` build requirement.
- Linux wheel target is `manylinux_2_28_x86_64` specifically (matches the
  project's CI/training-box platform) — not `manylinux2014` or musllinux.
- If the `manylinux_2_28_x86_64` image has no cp314 interpreter under
  `/opt/python/`, STOP and report back (exit code 2 from the gate script).
  Do not substitute a different base image, build CPython from source inside
  the container, or fall back to a different manylinux tag without asking —
  each of those changes what the wheel's compliance tag actually promises.
- Record every build (platform, wheel filename, sha256, build timestamp) in
  `manifests/pyonmttok_build.json`. This is the project's existing
  "record every resolved package version and checkpoint hash in manifests/"
  constraint applied to a built artifact instead of a resolved package.
- Scratch/intermediate build state lives under `build/pyonmttok/` (already
  covered by the existing `build/` gitignore rule — no `.gitignore` change
  needed for that path). Only the final wheel files are committed, under
  `vendor/wheels/`.
- Never fine-tune, LoRA, or continue-train anything — not applicable to this
  plan (no model weights involved) but stated for completeness per
  `CLAUDE.md`.

---

### Task 1: Pin and vendor the pyonmttok source checkout

**Files:**
- Create: `scripts/build_pyonmttok/clone_source.sh`
- Create: `manifests/pyonmttok_build.json`
- Test: manual run + assertion (see Step 2) — no pytest for this step; it's
  a one-shot checkout script, not library logic.

**Interfaces:**
- Produces: `build/pyonmttok/src/` — a git checkout of `OpenNMT/Tokenizer`
  pinned at commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`, submodules
  initialized. Tasks 2 and 3 read from this path and verify the same commit
  before building.
- Produces: `manifests/pyonmttok_build.json` with a `source` key populated
  and an empty `builds` list. Tasks 2 and 3 append to `builds`.

- [ ] **Step 1: Write the clone script**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/build/pyonmttok/src"
PINNED_COMMIT="ae4b83b0b2af2172ed306c372835e84b51d5d77e"
PINNED_TAG="v1.38.1"

rm -rf "$SRC_DIR"
mkdir -p "$(dirname "$SRC_DIR")"
git clone --quiet --branch "$PINNED_TAG" --depth 1 \
  https://github.com/OpenNMT/Tokenizer.git "$SRC_DIR"
git -C "$SRC_DIR" submodule update --init --recursive

actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "FAIL: expected commit $PINNED_COMMIT, got $actual_commit" >&2
  exit 1
fi

echo "OK: $SRC_DIR checked out at $actual_commit"
```

Save as `scripts/build_pyonmttok/clone_source.sh`.

- [ ] **Step 2: Run it and verify the commit check passes**

Run: `bash scripts/build_pyonmttok/clone_source.sh`
Expected output ends with: `OK: .../build/pyonmttok/src checked out at ae4b83b0b2af2172ed306c372835e84b51d5d77e`

Also verify the exit code is non-zero if the pin is wrong — temporarily edit
`PINNED_COMMIT` in a throwaway copy to a wrong value, confirm it prints
`FAIL: expected commit ...` and exits 1, then discard the throwaway edit (the
committed script keeps the correct pin).

- [ ] **Step 3: Write the initial manifest**

```json
{
  "source": {
    "repo": "https://github.com/OpenNMT/Tokenizer",
    "tag": "v1.38.1",
    "commit": "ae4b83b0b2af2172ed306c372835e84b51d5d77e"
  },
  "icu_version": "73.2",
  "pybind11_version": "3.1.0",
  "builds": []
}
```

Save as `manifests/pyonmttok_build.json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_pyonmttok/clone_source.sh manifests/pyonmttok_build.json
git commit -m "build: pin pyonmttok source checkout for cp314 build"
```

---

### Task 2: Build and verify the Windows (win_amd64) wheel

**Files:**
- Create: `scripts/build_pyonmttok/smoke_test.py`
- Create: `scripts/build_pyonmttok/record_manifest.py`
- Test: `tests/test_record_manifest.py`
- Create: `scripts/build_pyonmttok/build_windows.sh`
- Modify: `.gitignore` (the blanket `wheels/` rule at line 22 would hide
  `vendor/wheels/` — add explicit un-ignores)
- Modify: `manifests/pyonmttok_build.json` (gets a `win_amd64` build record
  appended by running `record_manifest.py`, not by hand-editing)

**Interfaces:**
- Consumes: `build/pyonmttok/src/` from Task 1, already pinned and verified.
- Produces: `scripts/build_pyonmttok/smoke_test.py` — a standalone script,
  `python smoke_test.py`, exit 0 on success, exit 1 with an assertion
  message on failure. Task 3 reuses this unchanged.
- Produces: `scripts/build_pyonmttok/record_manifest.py` — CLI
  `python record_manifest.py <platform> <wheel_path>`, upserts one entry in
  `manifests/pyonmttok_build.json` keyed by `platform`. Task 3 reuses this
  unchanged.
- Produces: `vendor/wheels/pyonmttok-1.38.1-cp314-cp314-win_amd64.whl`.

- [ ] **Step 1: Write the failing test for `record_manifest.py`**

```python
# tests/test_record_manifest.py
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
```

- [ ] **Step 2: Run it to verify it fails (module doesn't exist yet)**

Run: `uv run pytest tests/test_record_manifest.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'record_manifest'`

- [ ] **Step 3: Write `record_manifest.py`**

```python
#!/usr/bin/env python
"""Upsert one platform's build record into manifests/pyonmttok_build.json."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "manifests" / "pyonmttok_build.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(platform: str, wheel_path_str: str) -> int:
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
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    builds = [b for b in manifest.get("builds", []) if b["platform"] != platform]
    builds.append(record)
    manifest["builds"] = builds
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {platform} build: {record['wheel']} ({record['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: record_manifest.py <platform> <wheel_path>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
```

Save as `scripts/build_pyonmttok/record_manifest.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_record_manifest.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/build_pyonmttok/record_manifest.py tests/test_record_manifest.py
git commit -m "build: add pyonmttok build manifest recorder"
```

- [ ] **Step 6: Write the smoke test script**

```python
#!/usr/bin/env python
"""Round-trip smoke test for a built pyonmttok wheel (from its own README)."""
import sys


def main() -> int:
    assert sys.version_info[:2] == (3, 14), f"expected cp314, got {sys.version}"

    import pyonmttok

    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
    tokens, _ = tokenizer.tokenize("Hello World!")
    detokenized = tokenizer.detokenize(tokens)
    assert detokenized == "Hello World!", f"round-trip mismatch: {detokenized!r}"

    print("OK: pyonmttok import and tokenize/detokenize round-trip succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Save as `scripts/build_pyonmttok/smoke_test.py`. This has no pytest
coverage of its own — it IS the acceptance test for a built wheel, run
against a real installed wheel in Steps 9 and (Task 3) inside the manylinux
container. There is nothing to unit-test here without a built wheel to test
against.

- [ ] **Step 7: Fix `.gitignore` so `vendor/wheels/*.whl` can be committed**

The existing line 22, `wheels/`, matches any directory named `wheels`
anywhere in the tree, including `vendor/wheels/`. Add after it:

```gitignore
!vendor/
!vendor/wheels/
!vendor/wheels/*.whl
```

- [ ] **Step 8: Write the Windows build script**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/build/pyonmttok/src"
WORK_DIR="$REPO_ROOT/build/pyonmttok/windows"
ICU_VERSION="73.2"
PYBIND11_VERSION="3.1.0"
PINNED_COMMIT="ae4b83b0b2af2172ed306c372835e84b51d5d77e"

python -c "import sys; assert sys.version_info[:2] == (3, 14), sys.version"

if [ ! -d "$SRC_DIR/.git" ]; then
  echo "Source not found at $SRC_DIR -- run clone_source.sh first" >&2
  exit 1
fi
actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "Source commit mismatch: expected $PINNED_COMMIT, got $actual_commit" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

export TOKENIZER_ROOT="$WORK_DIR/tokenizer-install"

# --- ICU (prebuilt MSVC2019 binaries) ---
curl --fail -L -o icu4c-win.zip \
  "https://github.com/unicode-org/icu/releases/download/release-${ICU_VERSION/./-}/icu4c-${ICU_VERSION/./_}-Win64-MSVC2019.zip"
unzip -q icu4c-win.zip
unzip -q ./*_Release/icu-windows.zip -d icu
rm -rf ./*_Release icu4c-win.zip

# --- Build the C++ tokenizer library ---
rm -rf "$SRC_DIR/build"
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
  -DLIB_ONLY=ON \
  -DICU_ROOT="$WORK_DIR/icu" \
  -DCMAKE_INSTALL_PREFIX="$TOKENIZER_ROOT"
cmake --build "$SRC_DIR/build" --config Release --target install

# --- Stage ICU + tokenizer DLLs next to the extension source ---
cp "$WORK_DIR"/icu/bin64/icudt*.dll "$SRC_DIR/bindings/python/pyonmttok/"
cp "$WORK_DIR"/icu/bin64/icuuc*.dll "$SRC_DIR/bindings/python/pyonmttok/"
cp "$TOKENIZER_ROOT/bin/OpenNMTTokenizer.dll" "$SRC_DIR/bindings/python/pyonmttok/"

# --- Build the wheel (no build isolation, so the pinned pybind11 is used) ---
python -m pip install --quiet --upgrade pip setuptools wheel build "pybind11==${PYBIND11_VERSION}"
rm -rf "$WORK_DIR/dist"
( cd "$SRC_DIR/bindings/python" && TOKENIZER_ROOT="$TOKENIZER_ROOT" python -m build --wheel --no-isolation --outdir "$WORK_DIR/dist" )

echo "Wheel(s) built:"
ls "$WORK_DIR/dist"
```

Save as `scripts/build_pyonmttok/build_windows.sh`.

- [ ] **Step 9: Run the build, then verify with the smoke test in a fresh venv**

Run: `bash scripts/build_pyonmttok/build_windows.sh`
Expected: ends with `Wheel(s) built:` followed by one `.whl` filename
matching `pyonmttok-1.38.1-cp314-cp314-win_amd64.whl` (or an equivalent
cp314 win_amd64 tag — if the tag differs, note the actual filename and use
it consistently in the following steps).

If this fails with a compiler or linker error, do not guess at a different
ICU or pybind11 version — report the exact error. This is a new build
target with no upstream precedent; a real toolchain mismatch is a legitimate
blocked state, not something to route around silently.

Then verify the wheel actually works:

```bash
WHEEL="$(ls build/pyonmttok/windows/dist/*.whl)"
rm -rf build/pyonmttok/windows/venv
python -m venv build/pyonmttok/windows/venv
build/pyonmttok/windows/venv/Scripts/python -m pip install --quiet "$WHEEL"
build/pyonmttok/windows/venv/Scripts/python scripts/build_pyonmttok/smoke_test.py
```

Expected: `OK: pyonmttok import and tokenize/detokenize round-trip succeeded`

- [ ] **Step 10: Commit the wheel, record it in the manifest, commit again**

```bash
mkdir -p vendor/wheels
cp build/pyonmttok/windows/dist/*.whl vendor/wheels/
python scripts/build_pyonmttok/record_manifest.py win_amd64 vendor/wheels/pyonmttok-1.38.1-cp314-cp314-win_amd64.whl
git add .gitignore scripts/build_pyonmttok/smoke_test.py scripts/build_pyonmttok/build_windows.sh \
  vendor/wheels/pyonmttok-1.38.1-cp314-cp314-win_amd64.whl manifests/pyonmttok_build.json
git commit -m "build: add working pyonmttok cp314 win_amd64 wheel"
```

(Adjust the wheel filename in these commands if Step 9's actual output
filename differs from the one assumed above.)

---

### Task 3: Build and verify the Linux (manylinux_2_28_x86_64) wheel

**Files:**
- Create: `scripts/build_pyonmttok/build_linux_inner.sh`
- Create: `scripts/build_pyonmttok/run_linux_build.sh`
- Modify: `manifests/pyonmttok_build.json` (gets a `manylinux_2_28_x86_64`
  build record appended by `record_manifest.py`)

**Interfaces:**
- Consumes: `build/pyonmttok/src/` from Task 1 (same pinned checkout, no
  re-clone).
- Consumes: `scripts/build_pyonmttok/smoke_test.py` and
  `scripts/build_pyonmttok/record_manifest.py` from Task 2, unchanged.
- Produces: `vendor/wheels/pyonmttok-1.38.1-cp314-cp314-manylinux_2_28_x86_64.whl`.

- [ ] **Step 1: Write the in-container build script**

This runs *inside* the `manylinux_2_28_x86_64` container, where `/work/src`
is the (host-verified, already-pinned) source checkout and `/work/out` is
where the final repaired wheel must land.

```bash
#!/usr/bin/env bash
set -euo pipefail

PY_BIN="$1"
export PATH="$(dirname "$PY_BIN"):$PATH"

SRC_DIR=/work/src
OUT_DIR=/work/out
WORK_DIR=/root/pyonmttok-build
ICU_VERSION="73.2"
PYBIND11_VERSION="3.1.0"

"$PY_BIN" --version

"$PY_BIN" -m pip install --quiet --upgrade pip cmake setuptools wheel build \
  "pybind11==${PYBIND11_VERSION}" auditwheel

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"

export ICU_ROOT="$WORK_DIR/icu"
export TOKENIZER_ROOT="$WORK_DIR/tokenizer-install"

# --- ICU (built from source, static, for manylinux portability) ---
curl --fail -L -O "https://github.com/unicode-org/icu/releases/download/release-${ICU_VERSION/./-}/icu4c-${ICU_VERSION/./_}-src.tgz"
tar xf icu4c-*-src.tgz
( cd icu/source && CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC" \
    ./configure --disable-shared --enable-static --prefix="$ICU_ROOT" && \
    make -j"$(nproc)" install )

# --- Build the C++ tokenizer library ---
rm -rf "$SRC_DIR/build"
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
  -DLIB_ONLY=ON \
  -DICU_ROOT="$ICU_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$TOKENIZER_ROOT"
cmake --build "$SRC_DIR/build" --target install -- -j"$(nproc)"

# --- Build the wheel (no build isolation, so the pinned pybind11 is used) ---
( cd "$SRC_DIR/bindings/python" && TOKENIZER_ROOT="$TOKENIZER_ROOT" \
    "$PY_BIN" -m build --wheel --no-isolation --outdir "$WORK_DIR/dist" )

# --- Repair to manylinux_2_28 compliance (bundles the tokenizer .so) ---
RAW_WHEEL="$(ls "$WORK_DIR"/dist/*.whl)"
"$PY_BIN" -m auditwheel repair "$RAW_WHEEL" --plat manylinux_2_28_x86_64 -w "$OUT_DIR"

# --- Smoke test, using the same interpreter that built it ---
REPAIRED_WHEEL="$(ls "$OUT_DIR"/*.whl)"
"$PY_BIN" -m pip install --quiet --force-reinstall --no-deps "$REPAIRED_WHEEL"
"$PY_BIN" /work/scripts/smoke_test.py

echo "Repaired wheel(s) in $OUT_DIR:"
ls "$OUT_DIR"
```

Save as `scripts/build_pyonmttok/build_linux_inner.sh`.

- [ ] **Step 2: Write the host-side Docker orchestrator**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/build/pyonmttok/src"
OUT_DIR="$REPO_ROOT/build/pyonmttok/linux-out"
IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
PINNED_COMMIT="ae4b83b0b2af2172ed306c372835e84b51d5d77e"

if [ ! -d "$SRC_DIR/.git" ]; then
  echo "Source not found at $SRC_DIR -- run clone_source.sh first" >&2
  exit 1
fi
actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "Source commit mismatch: expected $PINNED_COMMIT, got $actual_commit" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "BLOCKED: Docker daemon not reachable. Start Docker Desktop and retry." >&2
  exit 2
fi

docker pull "$IMAGE"

CP314_DIR="$(docker run --rm "$IMAGE" bash -c 'ls -d /opt/python/cp314-cp314 2>/dev/null | head -1')"
if [ -z "$CP314_DIR" ]; then
  echo "BLOCKED: $IMAGE has no /opt/python/cp314-cp314 interpreter. Available interpreters:" >&2
  docker run --rm "$IMAGE" ls /opt/python/ >&2
  exit 2
fi
echo "Found cp314 interpreter: $CP314_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

docker run --rm \
  -v "$SRC_DIR:/work/src" \
  -v "$OUT_DIR:/work/out" \
  -v "$REPO_ROOT/scripts/build_pyonmttok:/work/scripts:ro" \
  "$IMAGE" \
  bash /work/scripts/build_linux_inner.sh "$CP314_DIR/bin/python3.14"

echo "Build finished. Wheel(s) in $OUT_DIR:"
ls "$OUT_DIR"
```

Save as `scripts/build_pyonmttok/run_linux_build.sh`.

- [ ] **Step 3: Run it**

Run: `bash scripts/build_pyonmttok/run_linux_build.sh`

Expected outcomes, in order of what to check:
1. If it prints `BLOCKED: Docker daemon not reachable...` and exits 2: start
   Docker Desktop, wait for it to report ready, and re-run. This is a known
   pre-existing condition (Docker Desktop's Linux engine was not running as
   of 2026-10-29) — not a plan defect.
2. If it prints `BLOCKED: ... has no /opt/python/cp314-cp314 interpreter...`
   and exits 2: STOP. Report the printed interpreter list back — do not
   substitute a different base image or build Python from source without
   asking (see Global Constraints).
3. Otherwise: expect it to end with `Build finished. Wheel(s) in .../linux-out:`
   followed by one `.whl` filename matching
   `pyonmttok-1.38.1-cp314-cp314-manylinux_2_28_x86_64.whl`, and — inside the
   container, before that — the line
   `OK: pyonmttok import and tokenize/detokenize round-trip succeeded`
   from the in-container smoke test.

If the build fails with a compiler/linker/auditwheel error (not one of the
two BLOCKED cases above), report the exact error rather than guessing at a
different ICU/pybind11 version or manylinux policy.

- [ ] **Step 4: Commit the wheel, record it in the manifest, commit again**

```bash
cp build/pyonmttok/linux-out/*.whl vendor/wheels/
python scripts/build_pyonmttok/record_manifest.py manylinux_2_28_x86_64 \
  vendor/wheels/pyonmttok-1.38.1-cp314-cp314-manylinux_2_28_x86_64.whl
git add scripts/build_pyonmttok/build_linux_inner.sh scripts/build_pyonmttok/run_linux_build.sh \
  vendor/wheels/pyonmttok-1.38.1-cp314-cp314-manylinux_2_28_x86_64.whl manifests/pyonmttok_build.json
git commit -m "build: add working pyonmttok cp314 manylinux_2_28_x86_64 wheel"
```

(Adjust the wheel filename in these commands if Step 3's actual output
filename differs from the one assumed above.)

---

### Task 4: Wire the wheels into the project and close the blocker

**Files:**
- Modify: `pyproject.toml:46-47` (`[tool.uv.sources]`)
- Modify: `STATE.md` (replace the open pyonmttok blocker with a resolved entry)
- Modify: `DECISIONS.md` (add the resolution decision, evidence-tagged)
- Modify: `CLAUDE.md` (repo layout gains `vendor/wheels/`)

**Interfaces:**
- Consumes: both wheel files committed in Tasks 2 and 3, and
  `manifests/pyonmttok_build.json` populated by both.

- [ ] **Step 1: Add the `[tool.uv.sources]` entries**

In `pyproject.toml`, change:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
```

to:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
pyonmttok = [
    { path = "vendor/wheels/pyonmttok-1.38.1-cp314-cp314-win_amd64.whl", marker = "sys_platform == 'win32'" },
    { path = "vendor/wheels/pyonmttok-1.38.1-cp314-cp314-manylinux_2_28_x86_64.whl", marker = "sys_platform == 'linux'" },
]
```

(Use the actual filenames recorded in `manifests/pyonmttok_build.json` if
either differs from what's shown here.)

- [ ] **Step 2: Run the real `uv lock` and verify it succeeds**

Run: `uv lock`
Expected: exits 0, updates `uv.lock`, with no "No solution found" error.
This is the actual close of the blocker — `STATE.md` currently records this
exact command failing; this step is where it starts succeeding.

If it still fails, read the actual error: it may mean the wheel's declared
`Requires-Python` or dependency metadata doesn't satisfy what `eole`
constrains `pyonmttok` to. Report the exact `uv lock` error rather than
re-running it unchanged.

- [ ] **Step 3: Update `STATE.md`**

Replace the bullet starting "Confirmed (refined): `pyonmttok` is imported
eagerly..." (the whole paragraph through "...or finding an eole version/fork
that makes this import optional.") with:

```markdown
- Resolved: `pyonmttok` has no cp314 wheel upstream, so it was built from
  source — `OpenNMT/Tokenizer` tag `v1.38.1`
  (commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`), ICU 73.2, pybind11
  3.1.0 — for both win_amd64 and manylinux_2_28_x86_64. Both wheels are
  committed under `vendor/wheels/`, recorded with build provenance and
  sha256 in `manifests/pyonmttok_build.json`, pass the tokenize/detokenize
  round-trip smoke test, and are wired in via `[tool.uv.sources]` path
  overrides keyed on `sys_platform`. `uv lock` succeeds on this machine as
  of this change. [EVIDENCE: `docs/superpowers/plans/2026-10-29-pyonmttok-cp314-build.md`,
  `manifests/pyonmttok_build.json`, 2026-10-29.] D0 is no longer blocked on
  this; `eole`'s doctor checks (`D-EOLE-PY`, `D-EOLE-CLI`) have not yet been
  run — that's D0 work, not this fix.
```

Also update the top banner's "Current next step" line to point at D0
instead of "resolve pyonmttok".

- [ ] **Step 4: Update `DECISIONS.md`**

Add a new dated entry:

```markdown
## 2026-10-29: Build pyonmttok from source for cp314

[EVIDENCE: live build, `manifests/pyonmttok_build.json`, tag `v1.38.1`
commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`.] No cp314 wheel or sdist
exists for `pyonmttok` 1.38.1 on PyPI (only cp310/cp311/cp312). Built it
from source for both win_amd64 and manylinux_2_28_x86_64 using ICU 73.2 and
pybind11 3.1.0 (pyonmttok's own `pyproject.toml` requires `pybind11>=2.10.4`
with no upper bound; pybind11 3.1.0 explicitly classifies Python 3.14
support). Wheels committed under `vendor/wheels/`, wired in via
`[tool.uv.sources]` path overrides. This unblocks `uv lock` and, in turn, D0.
```

- [ ] **Step 5: Update `CLAUDE.md`'s repo layout**

In the fenced repo layout block, add `vendor/wheels/` on its own line (near
`manifests/`), e.g.:

```
data/{raw,clean,mined,synth,test}/ (gitignored content, tracked dirs)
vendor/wheels/ (committed binary wheels with no cp314 upstream release)
.github/workflows/ci.yml
```

- [ ] **Step 6: Run the full test suite and lint to confirm nothing broke**

Run: `uv run pytest -q`
Expected: all existing tests still pass (including the new
`tests/test_record_manifest.py` from Task 2).

Run: `uv run ruff check`
Expected: no new findings.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock STATE.md DECISIONS.md CLAUDE.md
git commit -m "build: wire pyonmttok cp314 wheels into pyproject, close blocker"
```

---

**Note added after execution, not part of the original plan:**

A few points below the per-task per-review checkpoints, surfaced only by
the final whole-branch review, are worth recording here for anyone reading
this plan after the fact:

- The real Linux wheel built by Task 3 is dual-tagged:
  `pyonmttok-1.38.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
  not the plain `manylinux_2_28_x86_64` filename assumed throughout this
  plan's text (Task 3's own "adjust the wheel filename if it differs"
  caveats covered this, but it's worth stating plainly: the dual tag is
  `auditwheel`'s doing, declaring compatibility with both manylinux
  policies, and every reference to the wheel elsewhere in the repo uses the
  real, dual-tagged filename).
- The real wiring that closed the blocker needed `pyonmttok` added as a
  direct `[project.dependencies]` entry *and* `[tool.uv] environments` set
  — `[tool.uv.sources]` path overrides alone do not override a transitive
  dependency's resolution; see the "Build pyonmttok from source for cp314"
  entry in `DECISIONS.md` for the confirmed (not assumed) mechanism.
- Task 4's own in-body claims that "D0 is no longer blocked on this" and
  "This unblocks `uv lock` and, in turn, D0" turned out to be incomplete.
  `uv lock` succeeding was real, but it surfaced a new, separate blocker —
  `fasttext-wheel==0.9.2` has no cp314 wheel and fails to build from source
  on both Windows/MSVC and Linux/gcc14 — that still stands between this
  plan's result and D0. See `STATE.md` for the current status of that
  blocker.
