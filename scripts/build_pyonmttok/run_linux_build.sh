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

# On Windows/Git Bash, MSYS auto-converts bare POSIX-looking path *arguments*
# before they reach docker.exe (a native, non-MSYS binary). Two distinct
# forms of that bite here:
#   1. The container-side arguments below ("/work/scripts/...") are
#      themselves standalone POSIX-looking arguments, so MSYS wrongly
#      rewrites them into Windows paths too. Fixed with the well-known
#      escape of doubling the leading slash (Linux collapses "//" to "/", so
#      this is a no-op once inside the container, and a no-op on Linux/macOS
#      hosts where this script also runs).
#   2. The "-v host:container" host-side halves (computed from $REPO_ROOT,
#      itself an MSYS POSIX-style path like "/d/Personal documents/...") hit
#      a known MSYS conversion bug when the path contains a space: the
#      auto-conversion silently produces a bind mount that resolves to
#      nothing, so /work/* ends up empty in the container. Converting these
#      explicitly with `cygpath -w` (a no-op, absent, command on non-Windows
#      hosts, where $REPO_ROOT is already a plain host path) avoids relying
#      on that auto-conversion at all.
host_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

docker run --rm \
  -v "$(host_path "$SRC_DIR"):/work/src" \
  -v "$(host_path "$OUT_DIR"):/work/out" \
  -v "$(host_path "$REPO_ROOT/scripts/build_pyonmttok"):/work/scripts:ro" \
  "$IMAGE" \
  bash "//work/scripts/build_linux_inner.sh" "/$CP314_DIR/bin/python3.14"

echo "Build finished. Wheel(s) in $OUT_DIR:"
ls "$OUT_DIR"
