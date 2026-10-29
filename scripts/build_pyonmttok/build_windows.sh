#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/build/pyonmttok/src"
WORK_DIR="$REPO_ROOT/build/pyonmttok/windows"
BUILD_VENV_DIR="$WORK_DIR/buildenv"
ICU_VERSION="73.2"
ICU_ZIP_SHA256="27e3794025eac675ed5cff2921062baca547291dc5ad623aa36ebd5eadc54f77"
PYBIND11_VERSION="3.1.0"
PIP_VERSION="26.2.2"
SETUPTOOLS_VERSION="84.2.1"
WHEEL_VERSION="0.48.0"
BUILD_PKG_VERSION="1.6.1"
PYTEST_RUNNER_VERSION="6.0.1"
PYTEST_VERSION="9.2.0"
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

# Wipe any stale build artifacts left in the pinned source checkout by a
# previous run of this script (or of build_linux_inner.sh against the same
# shared checkout). setuptools' own build cache
# (bindings/python/build/, bindings/python/pyonmttok.egg-info/) is
# untracked, so without this `python -m build` sees the already-compiled
# `_ext` extension module as up to date and silently skips recompiling it
# -- only the separately-built OpenNMTTokenizer.dll would actually be
# fresh. `git clean -xfd` also clears the ICU/tokenizer DLLs this script
# itself stages into bindings/python/pyonmttok/ below on every prior run.
git -C "$SRC_DIR" clean -xfd

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

export TOKENIZER_ROOT="$WORK_DIR/tokenizer-install"

# --- ICU (prebuilt MSVC2019 binaries) ---
curl --fail -L -o icu4c-win.zip \
  "https://github.com/unicode-org/icu/releases/download/release-${ICU_VERSION/./-}/icu4c-${ICU_VERSION/./_}-Win64-MSVC2019.zip"
actual_icu_sha256="$(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" icu4c-win.zip)"
if [ "$actual_icu_sha256" != "$ICU_ZIP_SHA256" ]; then
  echo "ICU download sha256 mismatch: expected $ICU_ZIP_SHA256, got $actual_icu_sha256" >&2
  exit 1
fi
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

# --- Build the wheel in an isolated venv, never the real per-user CPython
#     3.14 install on PATH (no build isolation inside this venv, so the
#     pinned pybind11 is used) ---
rm -rf "$BUILD_VENV_DIR"
python -m venv "$BUILD_VENV_DIR"
BUILD_PY="$BUILD_VENV_DIR/Scripts/python"
"$BUILD_PY" -m pip install --quiet \
  "pip==${PIP_VERSION}" \
  "setuptools==${SETUPTOOLS_VERSION}" \
  "wheel==${WHEEL_VERSION}" \
  "build==${BUILD_PKG_VERSION}" \
  "pybind11==${PYBIND11_VERSION}" \
  "pytest-runner==${PYTEST_RUNNER_VERSION}" \
  "pytest==${PYTEST_VERSION}"
rm -rf "$WORK_DIR/dist"
( cd "$SRC_DIR/bindings/python" && TOKENIZER_ROOT="$TOKENIZER_ROOT" "$BUILD_PY" -m build --wheel --no-isolation --outdir "$WORK_DIR/dist" )

echo "Wheel(s) built:"
ls "$WORK_DIR/dist"

# --- Record build provenance for record_manifest.py ---
cpython_version="$("$BUILD_PY" -c "import platform; print(platform.python_version())")"
cmake_version="$(cmake --version | head -1 | sed -E 's/^cmake version //')"
# Best-effort MSVC *compiler* version (cl.exe, e.g. 19.51.x -- NOT the
# "14.x" MSVC toolset version): read back what CMake itself detected during
# configure, from CMakeFiles/<ver>/CMakeCXXCompiler.cmake (CMake does not
# cache this in CMakeCache.txt). This works regardless of how MSVC was
# located -- PATH, vswhere, or a Visual Studio generator -- unlike invoking
# cl.exe directly, which is only on PATH inside a Developer Command Prompt.
# `|| true` keeps a no-match here from tripping `set -e`/pipefail and
# aborting an otherwise-successful build.
msvc_compiler_version="$(grep -h '^set(CMAKE_CXX_COMPILER_VERSION ' "$SRC_DIR"/build/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -1 | sed -E 's/^set\(CMAKE_CXX_COMPILER_VERSION "([^"]*)"\).*/\1/' || true)"
msvc_compiler_version="${msvc_compiler_version:-unknown}"
# Since pip/setuptools/wheel/build are installed above with exact `==`
# pins, the pinned version *is* the resolved version (pip install would
# have failed under `set -e` otherwise) -- no need for a separate query.
{
  echo "cpython_version=${cpython_version}"
  echo "cmake_version=${cmake_version}"
  echo "pybind11_version=${PYBIND11_VERSION}"
  echo "icu_version=${ICU_VERSION}"
  echo "msvc_compiler_version=${msvc_compiler_version}"
  echo "pip_version=${PIP_VERSION}"
  echo "setuptools_version=${SETUPTOOLS_VERSION}"
  echo "wheel_version=${WHEEL_VERSION}"
  echo "build_version=${BUILD_PKG_VERSION}"
} > "$WORK_DIR/provenance.env"
echo "Build provenance:"
cat "$WORK_DIR/provenance.env"

WHEEL_PATH="$(ls "$WORK_DIR"/dist/*.whl | head -1)"
WHEEL_NAME="$(basename "$WHEEL_PATH")"
PROVENANCE_ARGS=()
while IFS= read -r line; do
  [ -n "$line" ] && PROVENANCE_ARGS+=("$line")
done < "$WORK_DIR/provenance.env"
echo "Suggested commands (vendor/wheels/ is the real destination -- the"
echo "build output above lives under the gitignored build/ directory):"
echo "  cp \"$WHEEL_PATH\" \"$REPO_ROOT/vendor/wheels/\""
echo "  python \"$REPO_ROOT/scripts/build_pyonmttok/record_manifest.py\" win_amd64 \"$REPO_ROOT/vendor/wheels/$WHEEL_NAME\" ${PROVENANCE_ARGS[*]}"
