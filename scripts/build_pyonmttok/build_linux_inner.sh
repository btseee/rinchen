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

# pytest-runner/pytest are required because the bindings' setup.py declares
# `setup_requires=["pytest-runner"]` and nothing else in this sequence pulls
# it in without build isolation (same fix as Task 2's Windows build).
"$PY_BIN" -m pip install --quiet --upgrade pip cmake setuptools wheel build \
  "pybind11==${PYBIND11_VERSION}" auditwheel pytest-runner pytest

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"

export ICU_ROOT="$WORK_DIR/icu"
export TOKENIZER_ROOT="$WORK_DIR/tokenizer-install"
# auditwheel resolves an extension's shared-library dependencies via the
# dynamic loader's search path, not the -L/-l flags used at link time, so
# without this it cannot find the just-built, non-system-installed
# libOpenNMTTokenizer.so when repairing the wheel below.
export LD_LIBRARY_PATH="$TOKENIZER_ROOT/lib:${LD_LIBRARY_PATH:-}"

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
