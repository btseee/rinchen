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
python -m pip install --quiet --upgrade pip setuptools wheel build "pybind11==${PYBIND11_VERSION}" pytest-runner pytest
rm -rf "$WORK_DIR/dist"
( cd "$SRC_DIR/bindings/python" && TOKENIZER_ROOT="$TOKENIZER_ROOT" python -m build --wheel --no-isolation --outdir "$WORK_DIR/dist" )

echo "Wheel(s) built:"
ls "$WORK_DIR/dist"
