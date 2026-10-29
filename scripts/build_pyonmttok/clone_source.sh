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
